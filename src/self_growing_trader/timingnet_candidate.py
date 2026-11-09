"""Isolated, causal TimingNet training and frozen evaluation.

This module intentionally does not import a policy artifact and never sees
lifecycle P/L, selected action, account, position or execution fields.  It
trains only two delayed M15 labels from immutable market/timing/Wyckoff input:

* whether the direct M15 context changes after a configured number of closed
  M15 samples;
* whether the causal timing trend score decelerates toward zero.

The resulting pickle is a *TimingNet candidate capsule*, not a policy
checkpoint.  A later, explicit integration experiment is required before it
can affect an MT5 action.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Iterable, Sequence
import csv
import json
import pickle

import numpy as np

from .features import MARKET_FEATURE_COLUMNS, TAIL_FEATURE_COLUMNS, add_features
from .market_regime_classifier import (
    MarketRegimeClassifier,
    REGIME_NAMES,
    REGIME_VOLATILE,
)
from .input_normalization import normalize_input_vector
from .timing_network import (
    REGIME_TRANSITION_LABELS,
    TimingNetwork,
    TIMING_NETWORK_INPUT_SIZE,
    extract_timing_features,
    timing_trend_score,
)


TIMINGNET_CANDIDATE_SCHEMA_VERSION = 1
TIMINGNET_TRANSITION_TARGET = "context_change_v1"
TIMINGNET_TRANSITION_TARGETS = frozenset({
    "context_change_v1",
    "trend_support_deterioration_v1",
})


@dataclass(frozen=True)
class TimingSample:
    """One causal observation representing one closed M15 state."""

    timestamp: str
    observation: np.ndarray
    regime: str
    trend_score: float


@dataclass
class CausalTimingFeatureAdapter:
    """Append compact causal M15 dynamics without touching the EA wire.

    The live 78 inputs are preserved verbatim as the prefix.  Twelve appended
    values summarize only present/past M15 observations, giving an isolated
    candidate direct access to deceleration and volatility-change evidence
    which a small recurrent state otherwise has to reconstruct implicitly.
    """

    width: int = 12
    mtf_scaling_mode: str = "legacy_m1_atr_v1"
    _history: list[tuple[np.ndarray, float]] = field(default_factory=list, repr=False)
    _last_regime: str = field(default="", repr=False)
    _regime_age: int = field(default=0, repr=False)

    @property
    def input_size(self) -> int:
        return TIMING_NETWORK_INPUT_SIZE + self.width

    def reset(self) -> None:
        self._history.clear()
        self._last_regime = ""
        self._regime_age = 0

    def transform_one(self, sample: TimingSample) -> np.ndarray:
        base = extract_timing_features(sample.observation)
        trend = float(timing_trend_score(base, mtf_scaling_mode=self.mtf_scaling_mode))

        def previous(offset: int) -> tuple[np.ndarray, float]:
            return self._history[-offset] if len(self._history) >= offset else (base, trend)

        one, trend_one = previous(1)
        three, trend_three = previous(3)
        six, trend_six = previous(6)
        regime = str(sample.regime or "")
        self._regime_age = self._regime_age + 1 if regime == self._last_regime else 1
        self._last_regime = regime
        extras = np.asarray([
            trend,
            trend - trend_one,
            trend - trend_three,
            trend - trend_six,
            abs(trend) - abs(trend_three),
            base[36] - one[36],
            base[36] - three[36],
            base[37] - one[37],
            base[37] - three[37],
            base[38] - one[38],
            base[38] - three[38],
            min(self._regime_age, 32) / 32.0,
        ], dtype=float)
        self._history.append((base.copy(), trend))
        self._history = self._history[-6:]
        return np.concatenate([base, np.clip(extras, -3.0, 3.0)])

    def transform(self, samples: Sequence[TimingSample]) -> np.ndarray:
        self.reset()
        return np.vstack([self.transform_one(sample) for sample in samples])


def build_causal_m15_tensors(
    feature_rows: np.ndarray,
    *,
    sequence_length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize causal ``[sample, 32 M15, feature]`` tensors.

    Rows are oldest-to-newest.  Only the valid suffix is populated during
    warm-up, and callers receive its length so zero padding cannot be treated
    as a genuine flat market.  This is shared conceptually with
    ``TimingNetwork.forward``: train and frozen inference use identical
    rolling-window semantics.
    """

    rows = np.asarray(feature_rows, dtype=float)
    if rows.ndim != 2 or rows.shape[0] <= 0:
        raise ValueError("feature_rows must be a non-empty [sample, feature] matrix")
    width = int(np.clip(int(sequence_length), 1, 128))
    result = np.zeros((rows.shape[0], width, rows.shape[1]), dtype=float)
    valid = np.zeros(rows.shape[0], dtype=int)
    for index in range(rows.shape[0]):
        start = max(0, index - width + 1)
        history = rows[start:index + 1]
        result[index, -history.shape[0]:] = history
        valid[index] = int(history.shape[0])
    return result, valid


def _context_label(classifier: MarketRegimeClassifier, market: np.ndarray) -> str:
    regime_id = classifier.rule_based_label(market)
    name = REGIME_NAMES.get(regime_id, "RANGING")
    if regime_id == REGIME_VOLATILE:
        name = f"VOLATILE_{classifier.volatile_subregime(market)}"
    return f"CTX_{name}"


def _trend_side(regime: str) -> int:
    value = str(regime or "").upper()
    if value.startswith("CTX_TANG_"):
        return 1
    if value.startswith("CTX_GIAM_"):
        return -1
    return 0


def _trend_strength(regime: str) -> int:
    value = str(regime or "").upper()
    if value.endswith("_MANH") and (value.startswith("CTX_TANG_") or value.startswith("CTX_GIAM_")):
        return 2
    if value.endswith("_YEU") and (value.startswith("CTX_TANG_") or value.startswith("CTX_GIAM_")):
        return 1
    return 0


def _transition_target(
    source_regime: str,
    future_regime: str,
    *,
    target_mode: str = TIMINGNET_TRANSITION_TARGET,
) -> int | None:
    """Target whether the present trend materially deteriorates before entry.

    A strong-to-weak transition is harmful for a fresh trend-follow entry;
    weak-to-strong is not.  Range/volatile sources are excluded because the
    candidate only supervises a trend-entry veto.
    """

    mode = str(target_mode or TIMINGNET_TRANSITION_TARGET).strip().lower()
    if mode == "context_change_v1":
        return int(str(source_regime) != str(future_regime))
    if mode != "trend_support_deterioration_v1":
        raise ValueError(f"unsupported TimingNet transition target: {target_mode}")
    source_side = _trend_side(source_regime)
    if source_side == 0:
        return None
    future_side = _trend_side(future_regime)
    if future_side != source_side:
        return 1
    return int(_trend_strength(future_regime) < _trend_strength(source_regime))


def _append_m15_sample(
    rows: list[TimingSample],
    *,
    timestamp: object,
    observation: Iterable[float] | np.ndarray,
    classifier: MarketRegimeClassifier,
    last_signature: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    raw = np.asarray(observation, dtype=float).reshape(-1)
    if raw.size < 58 or not np.isfinite(raw).all():
        return last_signature
    timing = extract_timing_features(raw)
    market = timing[:58]
    signature = classifier.m15_signature(
        market,
        mtf_scaling_mode=classifier.mtf_scaling_mode,
    )
    if signature == last_signature:
        return last_signature
    rows.append(TimingSample(
        timestamp=str(timestamp),
        observation=timing.copy(),
        regime=_context_label(classifier, market),
        trend_score=timing_trend_score(
            timing,
            mtf_scaling_mode=classifier.mtf_scaling_mode,
        ),
    ))
    return signature


def load_mt5_feature_audit_samples(
    path: str | Path,
    *,
    mtf_scaling_mode: str,
) -> list[TimingSample]:
    """Load real MT5 audit rows and retain one row per closed M15 state."""

    target = Path(path)
    if not target.is_file() or target.stat().st_size == 0:
        raise FileNotFoundError(f"MT5 feature audit missing or empty: {target}")
    with target.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        first = handle.readline()
        if not first:
            raise ValueError(f"MT5 feature audit has no header: {target}")
        delimiter = "\t" if "\t" in first else ","
        handle.seek(0)
        reader = csv.DictReader(handle, delimiter=delimiter)
        if not reader.fieldnames or "inputs" not in reader.fieldnames:
            raise ValueError("MT5 feature audit is missing the inputs field")
        classifier = MarketRegimeClassifier(mtf_scaling_mode=mtf_scaling_mode)
        samples: list[TimingSample] = []
        signature: tuple[float, float, float, float] | None = None
        for row in reader:
            try:
                values = np.fromstring(str(row.get("inputs", "")), sep="|", dtype=float)
            except ValueError:
                continue
            # Do not accidentally accept a partially-written live row.
            if values.size != 137 or not np.isfinite(values).all():
                continue
            signature = _append_m15_sample(
                samples,
                timestamp=row.get("bar_time", row.get("logged_at", "")),
                observation=values,
                classifier=classifier,
                last_signature=signature,
            )
    if len(samples) < 64:
        raise ValueError(f"only {len(samples)} unique M15 samples in {target}; need at least 64")
    return samples


def load_reference_csv_samples(
    path: str | Path,
    *,
    feature_config: object,
    symbol_config: object,
    start: str,
    end: str,
    mtf_scaling_mode: str,
    warmup_days: int = 21,
) -> list[TimingSample]:
    """Build causal M15 samples from the canonical feature-reference CSV.

    The reference is used only for pretraining and validation.  Promotion is
    still judged against the distinct MT5 FeatureAudit holdout.
    """

    import pandas as pd

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"reference CSV does not exist: {target}")
    start_at = pd.Timestamp(start)
    end_at = pd.Timestamp(end)
    read_from = start_at - pd.Timedelta(days=max(1, int(warmup_days)))
    chunks: list[object] = []
    names = ["<DATE>", "<TIME>", "<OPEN>", "<HIGH>", "<LOW>", "<CLOSE>", "<TICKVOL>", "<VOL>", "<SPREAD>"]
    for chunk in pd.read_csv(target, sep="\t", usecols=names, chunksize=200_000):
        stamp = pd.to_datetime(
            chunk["<DATE>"].astype(str) + " " + chunk["<TIME>"].astype(str),
            errors="coerce",
        )
        keep = (stamp >= read_from) & (stamp <= end_at)
        if bool(keep.any()):
            kept = chunk.loc[keep].copy()
            kept["Datetime"] = stamp.loc[keep]
            chunks.append(kept)
    if not chunks:
        raise ValueError(f"reference CSV has no rows in [{read_from}, {end_at}]")
    raw = pd.concat(chunks, ignore_index=True)
    frame = pd.DataFrame({
        "Datetime": raw["Datetime"],
        "Open": pd.to_numeric(raw["<OPEN>"], errors="coerce"),
        "High": pd.to_numeric(raw["<HIGH>"], errors="coerce"),
        "Low": pd.to_numeric(raw["<LOW>"], errors="coerce"),
        "Close": pd.to_numeric(raw["<CLOSE>"], errors="coerce"),
        "TickVolume": pd.to_numeric(raw["<TICKVOL>"], errors="coerce"),
        "Vol": pd.to_numeric(raw["<VOL>"], errors="coerce"),
        "Spread": pd.to_numeric(raw["<SPREAD>"], errors="coerce"),
    }).dropna(subset=["Datetime", "Open", "High", "Low", "Close", "Spread"])
    engineered = add_features(frame, feature_config, symbol_config)
    engineered = engineered.loc[
        (engineered["Datetime"] >= start_at) & (engineered["Datetime"] <= end_at)
    ].copy()
    classifier = MarketRegimeClassifier(mtf_scaling_mode=mtf_scaling_mode)
    samples: list[TimingSample] = []
    signature: tuple[float, float, float, float] | None = None
    columns = list(MARKET_FEATURE_COLUMNS) + list(TAIL_FEATURE_COLUMNS)
    for row in engineered[["Datetime", *columns]].itertuples(index=False, name=None):
        timestamp, *values = row
        observation = np.asarray(values, dtype=float)
        signature = _append_m15_sample(
            samples,
            timestamp=timestamp.isoformat(),
            observation=observation,
            classifier=classifier,
            last_signature=signature,
        )
    if len(samples) < 128:
        raise ValueError(f"only {len(samples)} unique M15 reference samples; need at least 128")
    return samples


def _binary_metrics(
    probability: Sequence[float],
    target: Sequence[int],
    *,
    baseline_rate: float,
    threshold: float,
) -> dict[str, float]:
    p = np.clip(np.asarray(probability, dtype=float).reshape(-1), 0.0, 1.0)
    y = np.asarray(target, dtype=int).reshape(-1)
    if p.size != y.size or not p.size:
        return {"count": 0.0, "positives": 0.0, "brier": 1.0, "baseline_brier": 1.0, "auc": 0.5, "average_precision": 0.0, "threshold": float(threshold), "support": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    y = (y > 0).astype(int)
    positives = int(y.sum())
    negatives = int(y.size - positives)
    brier = float(np.mean(np.square(p - y)))
    base = float(np.clip(baseline_rate, 0.0, 1.0))
    baseline_brier = float(np.mean(np.square(base - y)))
    if positives and negatives:
        order = np.argsort(p, kind="mergesort")
        ranks = np.empty(p.size, dtype=float)
        ranks[order] = np.arange(1, p.size + 1, dtype=float)
        for value in np.unique(p):
            tied = p == value
            ranks[tied] = float(np.mean(ranks[tied]))
        auc = float((ranks[y == 1].sum() - positives * (positives + 1) / 2.0) / max(positives * negatives, 1))
        descending = np.argsort(-p, kind="mergesort")
        y_desc = y[descending]
        cumulative = np.cumsum(y_desc)
        precision_at = cumulative / np.arange(1, y.size + 1, dtype=float)
        average_precision = float(precision_at[y_desc == 1].sum() / positives)
    else:
        auc = 0.5
        average_precision = float(positives / max(y.size, 1))
    predicted = p >= float(threshold)
    support = int(predicted.sum())
    true_positive = int(np.logical_and(predicted, y == 1).sum())
    precision = float(true_positive / support) if support else 0.0
    recall = float(true_positive / positives) if positives else 0.0
    f1 = float(2.0 * precision * recall / max(precision + recall, 1e-12))
    return {
        "count": float(y.size),
        "positives": float(positives),
        "positive_rate": float(positives / y.size),
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_improvement": float((baseline_brier - brier) / max(baseline_brier, 1e-12)),
        "auc": auc,
        "average_precision": average_precision,
        "threshold": float(threshold),
        "support": float(support),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fit_isotonic_calibrator(
    probability: Sequence[float],
    target: Sequence[int],
) -> dict[str, object]:
    """Fit monotone PAV calibration on a reference-only calibration split.

    The network retains the learned ordering.  The serialized map only turns
    a raw score into an empirical probability and is never fit on MT5 OOS.
    """

    p = np.clip(np.asarray(probability, dtype=float).reshape(-1), 0.0, 1.0)
    y = (np.asarray(target, dtype=int).reshape(-1) > 0).astype(float)
    if p.size != y.size or p.size == 0:
        return {"kind": "isotonic_pav_v1", "thresholds": [], "values": [], "count": 0}
    order = np.argsort(p, kind="mergesort")
    sorted_p, sorted_y = p[order], y[order]
    blocks: list[list[float]] = []  # [upper raw score, positives, count]
    start = 0
    while start < sorted_p.size:
        stop = start + 1
        while stop < sorted_p.size and sorted_p[stop] == sorted_p[start]:
            stop += 1
        blocks.append([float(sorted_p[stop - 1]), float(sorted_y[start:stop].sum()), float(stop - start)])
        while len(blocks) >= 2:
            previous, current = blocks[-2], blocks[-1]
            if previous[1] / previous[2] <= current[1] / current[2]:
                break
            blocks[-2:] = [[current[0], previous[1] + current[1], previous[2] + current[2]]]
        start = stop
    return {
        "kind": "isotonic_pav_v1",
        "thresholds": [float(block[0]) for block in blocks],
        "values": [float(block[1] / block[2]) for block in blocks],
        "count": int(p.size),
        "positives": int(y.sum()),
    }


def apply_probability_calibrator(
    probability: float | Sequence[float] | np.ndarray,
    calibrator: object | None,
) -> np.ndarray:
    """Apply serialized monotone calibration without moving any network row."""

    raw = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    if not isinstance(calibrator, dict) or calibrator.get("kind") != "isotonic_pav_v1":
        return raw.copy()
    thresholds = np.asarray(calibrator.get("thresholds", ()), dtype=float)
    values = np.asarray(calibrator.get("values", ()), dtype=float)
    if thresholds.ndim != 1 or values.shape != thresholds.shape or thresholds.size == 0:
        return raw.copy()
    index = np.searchsorted(thresholds, raw, side="left")
    return np.clip(values[np.clip(index, 0, thresholds.size - 1)], 0.0, 1.0)


def choose_validation_threshold(probability: Sequence[float], target: Sequence[int]) -> float:
    """Choose a fixed threshold from validation only, never from MT5 OOS."""

    y = np.asarray(target, dtype=int).reshape(-1)
    minimum_support = max(4, min(12, int(y.sum())))
    candidates = np.arange(0.10, 0.96, 0.05)
    ranked = []
    for threshold in candidates:
        metric = _binary_metrics(probability, y, baseline_rate=float(y.mean()) if y.size else 0.0, threshold=float(threshold))
        if metric["support"] >= minimum_support:
            ranked.append((metric["f1"], metric["precision"], -metric["threshold"], float(threshold)))
    return max(ranked)[-1] if ranked else 0.75


def _labels_for_index(
    samples: Sequence[TimingSample],
    index: int,
    *,
    deceleration_horizon: int,
    deceleration_min_trend_strength: float,
    deceleration_min_drop: float,
    transition_horizons: Sequence[int],
    transition_target_mode: str = TIMINGNET_TRANSITION_TARGET,
) -> tuple[int | None, dict[int, int]]:
    sample = samples[index]
    deceleration_target: int | None = None
    target_index = index + int(deceleration_horizon)
    if target_index < len(samples) and abs(sample.trend_score) >= deceleration_min_trend_strength:
        deceleration_target = int(
            abs(sample.trend_score) - abs(samples[target_index].trend_score) >= deceleration_min_drop
        )
    transitions = {
        int(horizon): _transition_target(
            sample.regime,
            samples[index + int(horizon)].regime,
            target_mode=transition_target_mode,
        )
        for horizon in transition_horizons
        if index + int(horizon) < len(samples)
    }
    return deceleration_target, transitions


def train_timingnet(
    network: TimingNetwork,
    samples: Sequence[TimingSample],
    *,
    epochs: int,
    batch_size: int = 256,
    transition_target: str = TIMINGNET_TRANSITION_TARGET,
    feature_adapter: CausalTimingFeatureAdapter | None = None,
) -> None:
    """Train the transition representation with causal truncated-BPTT batches.

    The former candidate loop called the online settlement writer once for
    every horizon at every M15 sample.  A six-seed experiment therefore made
    hundreds of thousands of tiny AdamW calls while changing no policy state.
    Here each sequential batch makes one AdamW update for the same causal
    future-context labels.  The recurrent encoder, embedding and transition
    head remain trainable; deceleration is intentionally observational and
    gets no surrogate outcome update in this candidate.
    """

    if len(samples) < 2:
        return
    network.ensure_compatible()
    horizons = tuple(int(value) for value in network.regime_transition_horizons)
    if not horizons:
        return
    size = max(16, int(batch_size))
    raw_feature_rows = feature_adapter.transform(samples) if feature_adapter is not None else np.vstack([
        extract_timing_features(sample.observation) for sample in samples
    ])
    if raw_feature_rows.shape[1] != network.input_size:
        raise ValueError("TimingNet candidate feature adapter/output dimension mismatch")
    feature_rows = np.vstack([
        normalize_input_vector(row, mode="soft_clip_v1", size=network.input_size)
        for row in raw_feature_rows
    ])
    sequence_tensors, sequence_valid_lengths = build_causal_m15_tensors(
        feature_rows,
        sequence_length=network.sequence_length,
    )
    regime_index = {name: index for index, name in enumerate(REGIME_TRANSITION_LABELS)}
    labels = np.asarray([regime_index.get(str(sample.regime), -1) for sample in samples], dtype=int)

    def forward_tensor_batch(
        tensors: np.ndarray,
        valid_lengths: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unroll each independent 32-M15 tensor with exact truncated BPTT."""

        batch, steps, _ = tensors.shape
        hidden = np.zeros((batch, steps, network.hidden_size), dtype=float)
        candidate = np.zeros_like(hidden)
        previous = np.zeros_like(hidden)
        current = np.zeros((batch, network.hidden_size), dtype=float)
        retention = float(network.memory_decay)
        injection = 1.0 - retention
        for step in range(steps):
            active = (step >= (steps - valid_lengths))[:, None]
            previous[:, step] = current
            proposed = np.tanh(tensors[:, step] @ network.w_in.T + current @ network.w_rec.T + network.b_hidden)
            candidate[:, step] = proposed
            updated = retention * current + injection * proposed
            current = np.where(active, updated, current)
            hidden[:, step] = current
        embedding = np.tanh(current @ network.w_embed.T + network.b_embed)
        return previous, candidate, hidden, embedding

    for _epoch in range(max(1, int(epochs))):
        for start in range(0, len(samples), size):
            stop = min(len(samples), start + size)
            previous, candidate, hidden, embedding = forward_tensor_batch(
                sequence_tensors[start:stop],
                sequence_valid_lengths[start:stop],
            )
            gradients = {
                name: np.zeros_like(value, dtype=float)
                for name, value in network._parameter_arrays()
            }
            d_embedding = np.zeros_like(embedding)
            settled = 0
            for head, horizon in enumerate(horizons):
                global_index = np.arange(start, stop, dtype=int)
                future_index = global_index + horizon
                valid = future_index < len(samples)
                if not bool(valid.any()):
                    continue
                local = np.flatnonzero(valid)
                source = labels[global_index[valid]]
                future = labels[future_index[valid]]
                source_names = np.asarray([samples[index].regime for index in global_index[valid]], dtype=object)
                future_names = np.asarray([samples[index].regime for index in future_index[valid]], dtype=object)
                event = np.asarray([
                    _transition_target(
                        str(current),
                        str(later),
                        target_mode=transition_target,
                    )
                    for current, later in zip(source_names, future_names, strict=True)
                ], dtype=object)
                # The first partial windows may be useful to warm a live
                # session but are not supervised as if they contained 32
                # bars.  This prevents an artificial all-zero prefix from
                # becoming a transition label shortcut.
                mature = sequence_valid_lengths[global_index[valid]] >= network.sequence_length
                usable = (source >= 0) & (future >= 0) & (event != None) & mature
                if not bool(usable.any()):
                    continue
                local, source, event = local[usable], source[usable], event[usable].astype(bool)
                logits = embedding[local] @ network.w_regime_transition[head].T + network.b_regime_transition[head]
                shifted = logits - np.max(logits, axis=1, keepdims=True)
                probabilities = np.exp(np.clip(shifted, -30.0, 30.0))
                probabilities /= np.maximum(probabilities.sum(axis=1, keepdims=True), 1e-12)
                # Direct binary transition likelihood.  The old online
                # objective predicted every future regime class, then used
                # ``1 - P(current_class)`` as an indirect transition score.
                # This candidate instead optimizes that exact score against
                # ``current != future`` while retaining the checkpoint's
                # softmax layout and public q_transition output contract.
                current_probability = probabilities[np.arange(local.size), source]
                changed = event
                error = -probabilities
                stable = ~changed
                error[stable, source[stable]] += 1.0
                if bool(changed.any()):
                    changed_rows = np.flatnonzero(changed)
                    p_current = current_probability[changed_rows]
                    remaining = np.maximum(1.0 - p_current, 1e-8)
                    error[changed_rows] = (
                        p_current[:, None] * probabilities[changed_rows] / remaining[:, None]
                    )
                    error[changed_rows, source[changed_rows]] = -p_current
                normalizer = float(max(1, local.size))
                gradients["w_regime_transition"][head] = (
                    error.T @ embedding[local] / normalizer
                    - network.regime_transition_l2 * network.w_regime_transition[head]
                )
                gradients["b_regime_transition"][head] = error.mean(axis=0)
                d_embedding[local] += (
                    error @ network.w_regime_transition[head] / normalizer
                ) * network.regime_transition_encoder_scale
                settled += int(local.size)
            if not settled:
                continue
            d_embedding_pre = d_embedding * (1.0 - np.square(np.clip(embedding, -1.0, 1.0)))
            gradients["w_embed"] = d_embedding_pre.T @ hidden[:, -1]
            gradients["b_embed"] = d_embedding_pre.sum(axis=0)
            carry = d_embedding_pre @ network.w_embed
            retention = float(network.memory_decay)
            injection = 1.0 - retention
            steps = sequence_tensors.shape[1]
            lengths = sequence_valid_lengths[start:stop]
            for step in range(steps - 1, -1, -1):
                active = (step >= (steps - lengths))[:, None]
                d_candidate = active * injection * carry * (1.0 - np.square(candidate[:, step]))
                gradients["w_in"] += d_candidate.T @ sequence_tensors[start:stop, step]
                gradients["w_rec"] += d_candidate.T @ previous[:, step]
                gradients["b_hidden"] += d_candidate.sum(axis=0)
                propagated = retention * carry + d_candidate @ network.w_rec
                carry = np.where(active, propagated, carry)
            norm = float(np.sqrt(sum(float(np.sum(np.square(value))) for value in gradients.values())))
            if norm > network.gradient_clip:
                factor = network.gradient_clip / max(norm, 1e-12)
                gradients = {name: value * factor for name, value in gradients.items()}
            network._optimizer_update(
                gradients,
                learning_rate=float(network.regime_transition_learning_rate),
                realized_r=0.0,
                auxiliary=True,
            )
            network.regime_transition_update_count += settled
            network.regime_transition_last_error = float(norm)
        network.reset_runtime_state()
    network.reset_runtime_state()


def evaluate_timingnet(
    network: TimingNetwork,
    samples: Sequence[TimingSample],
    *,
    warmup: Sequence[TimingSample] = (),
    deceleration_horizon: int,
    deceleration_min_trend_strength: float,
    deceleration_min_drop: float,
    transition_horizons: Sequence[int],
    transition_target_mode: str = TIMINGNET_TRANSITION_TARGET,
    thresholds: dict[str, float] | None = None,
    baseline_rates: dict[str, float] | None = None,
    calibrators: dict[str, object] | None = None,
    feature_adapter: CausalTimingFeatureAdapter | None = None,
    include_raw: bool = False,
) -> dict[str, object]:
    """Frozen chronological evaluation.  ``network`` weights never change."""

    frozen = network.copy()
    frozen.reset_runtime_state()
    adapter = (
        CausalTimingFeatureAdapter(mtf_scaling_mode=feature_adapter.mtf_scaling_mode)
        if feature_adapter is not None else None
    )
    for sample in warmup:
        observation = adapter.transform_one(sample) if adapter is not None else sample.observation
        frozen.forward(observation, regime_label=sample.regime)
    deceleration_probability: list[float] = []
    deceleration_target: list[int] = []
    transition_probability = {int(horizon): [] for horizon in transition_horizons}
    transition_target = {int(horizon): [] for horizon in transition_horizons}
    for index, sample in enumerate(samples):
        observation = adapter.transform_one(sample) if adapter is not None else sample.observation
        output = frozen.forward(observation, regime_label=sample.regime)
        deceleration, transitions = _labels_for_index(
            samples,
            index,
            deceleration_horizon=deceleration_horizon,
            deceleration_min_trend_strength=deceleration_min_trend_strength,
            deceleration_min_drop=deceleration_min_drop,
            transition_horizons=transition_horizons,
            transition_target_mode=transition_target_mode,
        )
        if deceleration is not None:
            deceleration_probability.append(float(apply_probability_calibrator(
                output.deceleration_probability,
                dict(calibrators or {}).get("deceleration"),
            )))
            deceleration_target.append(int(deceleration))
        for head, horizon in enumerate(transition_horizons):
            target = transitions.get(int(horizon))
            if target is None:
                continue
            transition_probability[int(horizon)].append(float(apply_probability_calibrator(
                output.regime_transition_probabilities[head],
                dict(calibrators or {}).get(f"transition_{int(horizon)}"),
            )))
            transition_target[int(horizon)].append(int(target))
    threshold_map = dict(thresholds or {})
    rate_map = dict(baseline_rates or {})
    decel_threshold = float(threshold_map.get("deceleration", 0.75))
    result: dict[str, object] = {
        "deceleration": _binary_metrics(
            deceleration_probability,
            deceleration_target,
            baseline_rate=float(rate_map.get("deceleration", np.mean(deceleration_target) if deceleration_target else 0.0)),
            threshold=decel_threshold,
        ),
        "transition": {},
    }
    for horizon in transition_horizons:
        key = f"transition_{int(horizon)}"
        result["transition"][str(int(horizon))] = _binary_metrics(
            transition_probability[int(horizon)],
            transition_target[int(horizon)],
            baseline_rate=float(rate_map.get(key, np.mean(transition_target[int(horizon)]) if transition_target[int(horizon)] else 0.0)),
            threshold=float(threshold_map.get(key, 0.75)),
        )
    if include_raw:
        result["_raw"] = {
            "deceleration_probability": deceleration_probability,
            "deceleration_target": deceleration_target,
            "transition_probability": transition_probability,
            "transition_target": transition_target,
        }
    return result


def timingnet_requirements(metrics: dict[str, object]) -> dict[str, object]:
    """Conservative publish gate for the sole deployable head: transition h3.

    Deceleration stays observational until it separately earns OOS evidence;
    a weak auxiliary head cannot gate, or silently invalidate, h3 transition.
    """

    deceleration = dict(metrics.get("deceleration", {}) or {})
    transitions = dict(metrics.get("transition", {}) or {})
    transition = dict(transitions.get("3", {}) or {})

    def passes(values: dict[str, object]) -> bool:
        return bool(
            float(values.get("count", 0.0)) >= 180
            and float(values.get("positives", 0.0)) >= 8
            and float(values.get("brier_improvement", -1.0)) >= 0.05
            and float(values.get("auc", 0.0)) >= 0.60
            and float(values.get("support", 0.0)) >= 8
            and float(values.get("precision", 0.0)) >= 0.50
        )

    transition_passed = passes(transition)
    return {
        "target": "trend_support_deterioration_h3",
        "deceleration_observational": passes(deceleration),
        "transition_h3": passes(transition),
        "passed": transition_passed,
        "rule": {
            "min_samples": 180,
            "min_positives": 8,
            "min_brier_improvement": 0.05,
            "min_auc": 0.60,
            "min_high_confidence_support": 8,
            "min_high_confidence_precision": 0.50,
        },
    }


def source_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_timingnet_candidate(
    path: str | Path,
    *,
    network: TimingNetwork,
    report: dict[str, object],
    feature_adapter: CausalTimingFeatureAdapter | None = None,
) -> None:
    """Persist only the isolated candidate and immutable training report."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    network.reset_runtime_state()
    capsule = {
        "schema_version": TIMINGNET_CANDIDATE_SCHEMA_VERSION,
        "kind": "isolated_timingnet_candidate",
        "network": network,
        "report": report,
        "feature_adapter": feature_adapter,
    }
    with target.open("wb") as handle:
        pickle.dump(capsule, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_json_report(path: str | Path, report: dict[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
