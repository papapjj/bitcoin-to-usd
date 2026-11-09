"""Causal pairwise preference feedback for the trading policy.

This is the trading equivalent of an RLHF preference layer.  It deliberately
does not pretend that a language-model DPO record is safe for an MT5 policy:
each pair must refer to the same pre-decision context and risk geometry, and
only the two competing entry actions are compared.  The pair is an auxiliary
candidate signal; it never authorises an inference-side action or a promotion.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import json
from math import exp, log
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


RLHF_PREFERENCE_SCHEMA_VERSION = "rlhf_preference_v1"
RLHF_ENTRY_ACTIONS = tuple(range(7))
RLHF_LABEL_SOURCES = frozenset({"human", "expert", "teacher", "mt5_rater"})
RLHF_SPLITS = frozenset({"train", "validation", "test", "shadow"})


def _text(value: object) -> str:
    return str(value or "").strip()


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _integer(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _boolean(value: object, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return bool(default)


def _reason_codes(value: object) -> tuple[str, ...]:
    if isinstance(value, (tuple, list, set)):
        values = value
    else:
        values = _text(value).replace(",", "|").split("|")
    return tuple(sorted({_text(item).lower() for item in values if _text(item)}))


def _parse_timestamp(value: str) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


@dataclass(frozen=True)
class RLHFPreferencePair:
    """One auditable chosen/rejected comparison for the entry surface.

    ``pre_decision_state_hash`` and ``risk_geometry_hash`` are hashes of the
    immutable state available at decision time.  A pair that does not share
    those hashes cannot distinguish preference from a changed market state or
    a changed SL/TP contract, so it is rejected before training.
    """

    pair_id: str
    context_id: str
    chosen_trace_id: str
    rejected_trace_id: str
    chosen_action: int
    rejected_action: int
    decision_time: str
    feedback_time: str
    pre_decision_state_hash: str
    risk_geometry_hash: str
    label_source: str
    rater_id: str
    rater_version: str
    confidence: float
    split: str = "train"
    horizon_bars: int = 48
    reason_codes: tuple[str, ...] = ()
    source_run_id: str = ""
    # Filled for teacher/RLAIF rows so an offline outcome-trained teacher is
    # never mistaken for a human rater.  Human/expert rows may leave it empty.
    teacher_training_evidence: str = ""
    feature_causal: bool = True
    candidate_only: bool = True
    outcome_audit_only: bool = True
    schema_version: str = RLHF_PREFERENCE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "reason_codes", _reason_codes(self.reason_codes))
        errors = validate_rlhf_preference_pair(self)
        if errors:
            raise ValueError("invalid RLHF preference pair: " + "; ".join(errors))

    def to_row(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "pair_id": self.pair_id,
            "context_id": self.context_id,
            "chosen_trace_id": self.chosen_trace_id,
            "rejected_trace_id": self.rejected_trace_id,
            "chosen_action": self.chosen_action,
            "rejected_action": self.rejected_action,
            "decision_time": self.decision_time,
            "feedback_time": self.feedback_time,
            "pre_decision_state_hash": self.pre_decision_state_hash,
            "risk_geometry_hash": self.risk_geometry_hash,
            "label_source": self.label_source,
            "rater_id": self.rater_id,
            "rater_version": self.rater_version,
            "confidence": self.confidence,
            "split": self.split,
            "horizon_bars": self.horizon_bars,
            "reason_codes": "|".join(self.reason_codes),
            "source_run_id": self.source_run_id,
            "teacher_training_evidence": self.teacher_training_evidence,
            "feature_causal": self.feature_causal,
            "candidate_only": self.candidate_only,
            "outcome_audit_only": self.outcome_audit_only,
        }


def validate_rlhf_preference_pair(pair: RLHFPreferencePair) -> tuple[str, ...]:
    """Return fail-closed validation errors without making a promotion claim."""

    errors: list[str] = []
    for name in (
        "pair_id", "context_id", "chosen_trace_id", "rejected_trace_id",
        "pre_decision_state_hash", "risk_geometry_hash", "decision_time",
        "feedback_time", "label_source", "rater_id", "rater_version",
    ):
        if not _text(getattr(pair, name, "")):
            errors.append(f"missing_{name}")
    if _text(getattr(pair, "schema_version", "")) != RLHF_PREFERENCE_SCHEMA_VERSION:
        errors.append("unsupported_schema_version")
    if _text(getattr(pair, "chosen_trace_id", "")) == _text(getattr(pair, "rejected_trace_id", "")):
        errors.append("chosen_and_rejected_trace_must_differ")
    try:
        chosen_action = int(pair.chosen_action)
        rejected_action = int(pair.rejected_action)
    except (TypeError, ValueError):
        chosen_action = rejected_action = -1
    if chosen_action not in RLHF_ENTRY_ACTIONS:
        errors.append("chosen_action_out_of_entry_surface")
    if rejected_action not in RLHF_ENTRY_ACTIONS:
        errors.append("rejected_action_out_of_entry_surface")
    if chosen_action == rejected_action and chosen_action in RLHF_ENTRY_ACTIONS:
        errors.append("chosen_and_rejected_action_must_differ")
    source = _text(getattr(pair, "label_source", "")).lower()
    if source not in RLHF_LABEL_SOURCES:
        errors.append("unsupported_label_source")
    split = _text(getattr(pair, "split", "")).lower()
    if split not in RLHF_SPLITS:
        errors.append("unsupported_split")
    confidence = _finite_float(getattr(pair, "confidence", float("nan")))
    if not 0.0 <= confidence <= 1.0:
        errors.append("confidence_out_of_range")
    if int(getattr(pair, "horizon_bars", 0) or 0) < 1:
        errors.append("horizon_bars_must_be_positive")
    if not bool(getattr(pair, "feature_causal", False)):
        errors.append("future_feature_leakage_declared")
    if not bool(getattr(pair, "candidate_only", False)):
        errors.append("candidate_only_must_be_true")
    if not bool(getattr(pair, "outcome_audit_only", False)):
        errors.append("outcome_must_be_audit_only")
    decision = _parse_timestamp(_text(getattr(pair, "decision_time", "")))
    feedback = _parse_timestamp(_text(getattr(pair, "feedback_time", "")))
    if decision is None:
        errors.append("invalid_decision_time")
    if feedback is None:
        errors.append("invalid_feedback_time")
    if decision is not None and feedback is not None:
        try:
            if feedback < decision:
                errors.append("feedback_before_decision")
        except TypeError:
            errors.append("incomparable_timestamps")
    return tuple(errors)


def parse_rlhf_preference_row(row: Mapping[str, object]) -> RLHFPreferencePair | None:
    """Parse one JSONL/CSV-like row; malformed or unsafe rows return ``None``."""

    try:
        pair = RLHFPreferencePair(
            pair_id=_text(row.get("pair_id")),
            context_id=_text(row.get("context_id")),
            chosen_trace_id=_text(row.get("chosen_trace_id")),
            rejected_trace_id=_text(row.get("rejected_trace_id")),
            chosen_action=_integer(row.get("chosen_action"), -1),
            rejected_action=_integer(row.get("rejected_action"), -1),
            decision_time=_text(row.get("decision_time")),
            feedback_time=_text(row.get("feedback_time")),
            pre_decision_state_hash=_text(row.get("pre_decision_state_hash")),
            risk_geometry_hash=_text(row.get("risk_geometry_hash")),
            label_source=_text(row.get("label_source")).lower(),
            rater_id=_text(row.get("rater_id")),
            rater_version=_text(row.get("rater_version")),
            confidence=_finite_float(row.get("confidence")),
            split=_text(row.get("split", "train")).lower(),
            horizon_bars=_integer(row.get("horizon_bars", 48), 48),
            reason_codes=_reason_codes(row.get("reason_codes", "")),
            source_run_id=_text(row.get("source_run_id")),
            teacher_training_evidence=_text(row.get("teacher_training_evidence")),
            feature_causal=_boolean(row.get("feature_causal"), False),
            candidate_only=_boolean(row.get("candidate_only"), False),
            outcome_audit_only=_boolean(row.get("outcome_audit_only"), False),
            schema_version=_text(row.get("schema_version")),
        )
    except (TypeError, ValueError):
        return None
    return pair


def load_rlhf_preference_jsonl(path: str | Path) -> tuple[tuple[RLHFPreferencePair, ...], tuple[int, ...]]:
    """Load valid pairs and return 1-based line numbers rejected fail-closed."""

    valid: list[RLHFPreferencePair] = []
    invalid_lines: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            pair = parse_rlhf_preference_row(row) if isinstance(row, Mapping) else None
            if pair is None:
                invalid_lines.append(line_number)
            else:
                valid.append(pair)
    return tuple(valid), tuple(invalid_lines)


@dataclass(frozen=True)
class RLHFPairwiseResult:
    loss: float
    probability_chosen: float
    margin: float
    ascent: np.ndarray
    effective_confidence: float


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = exp(-min(float(value), 60.0))
        return 1.0 / (1.0 + z)
    z = exp(max(float(value), -60.0))
    return z / (1.0 + z)


def rlhf_pairwise_loss_and_ascent(
    logits: Sequence[float] | np.ndarray,
    pair: RLHFPreferencePair,
    *,
    temperature: float = 0.25,
    min_confidence: float = 0.0,
) -> RLHFPairwiseResult:
    """Bradley--Terry loss and ascent direction on the seven entry logits.

    The returned ``ascent`` has the same sign convention as the existing
    Direct-RNN policy directions: it increases the chosen logit and lowers the
    rejected logit.  The actual parameter update still goes through the one
    Full-RL AdamW writer.
    """

    values = np.asarray(logits, dtype=float).reshape(-1)
    if values.size < len(RLHF_ENTRY_ACTIONS):
        raise ValueError("RLHF logits must contain the seven entry actions")
    if validate_rlhf_preference_pair(pair):
        raise ValueError("cannot train on an invalid RLHF preference pair")
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("RLHF temperature must be positive")
    confidence = float(np.clip(pair.confidence, 0.0, 1.0))
    if confidence < float(np.clip(min_confidence, 0.0, 1.0)):
        confidence = 0.0
    margin = float(
        (values[int(pair.chosen_action)] - values[int(pair.rejected_action)]) / temperature
    )
    probability = _sigmoid(margin)
    loss = confidence * float(np.logaddexp(0.0, -margin))
    signal = confidence * (1.0 - probability) / temperature
    ascent = np.zeros(values.size, dtype=float)
    ascent[int(pair.chosen_action)] = signal
    ascent[int(pair.rejected_action)] = -signal
    return RLHFPairwiseResult(
        loss=float(loss),
        probability_chosen=float(probability),
        margin=float(margin),
        ascent=ascent,
        effective_confidence=float(confidence),
    )


def rlhf_action_target(
    pair: RLHFPreferencePair,
    *,
    confidence_floor: float = 0.5,
    smoothing: float = 0.02,
) -> np.ndarray:
    """Create a seven-action soft target for adapters using existing CE code."""

    confidence = float(np.clip(pair.confidence, 0.0, 1.0))
    preferred = float(np.clip(confidence, float(confidence_floor), 1.0))
    smoothing = float(np.clip(smoothing, 0.0, 0.25))
    target = np.full(len(RLHF_ENTRY_ACTIONS), smoothing, dtype=float)
    target[int(pair.chosen_action)] += preferred
    target[int(pair.rejected_action)] += max(1.0 - preferred, 1e-6)
    target /= max(float(target.sum()), 1e-12)
    return target


def pairwise_preference_metrics(
    pairs: Sequence[RLHFPreferencePair],
    chosen_scores: Sequence[float],
    rejected_scores: Sequence[float],
    *,
    temperature: float = 0.25,
) -> dict[str, object]:
    """Score a held-out preference set; this is not an MT5/LCB gate."""

    if len(pairs) != len(chosen_scores) or len(pairs) != len(rejected_scores):
        raise ValueError("preference pairs and score arrays must have equal length")
    if not pairs:
        return {
            "pair_count": 0,
            "weighted_count": 0.0,
            "accuracy": 0.0,
            "weighted_accuracy": 0.0,
            "log_loss": 0.0,
            "brier": 0.0,
            "mean_margin": 0.0,
            "mean_probability_chosen": 0.0,
            "source_counts": {},
            "split_counts": {},
            "evidence_class": "offline_preference_only",
        }
    rows: list[RLHFPairwiseResult] = []
    for pair, chosen, rejected in zip(pairs, chosen_scores, rejected_scores):
        if validate_rlhf_preference_pair(pair):
            raise ValueError(f"invalid RLHF pair {pair.pair_id}")
        values = np.zeros(7, dtype=float)
        values[pair.chosen_action] = float(chosen)
        values[pair.rejected_action] = float(rejected)
        rows.append(rlhf_pairwise_loss_and_ascent(values, pair, temperature=temperature))
    weights = np.asarray([max(float(row.effective_confidence), 1e-12) for row in rows], dtype=float)
    probabilities = np.asarray([row.probability_chosen for row in rows], dtype=float)
    margins = np.asarray([row.margin for row in rows], dtype=float)
    labels = np.ones(len(rows), dtype=float)
    correct = (probabilities >= 0.5).astype(float)
    log_losses = -np.log(np.maximum(probabilities, 1e-12))
    brier = (probabilities - labels) ** 2
    return {
        "pair_count": len(rows),
        "weighted_count": float(weights.sum()),
        "accuracy": float(correct.mean()),
        "weighted_accuracy": float(np.average(correct, weights=weights)),
        "log_loss": float(np.average(log_losses, weights=weights)),
        "brier": float(np.average(brier, weights=weights)),
        "mean_margin": float(np.average(margins, weights=weights)),
        "mean_probability_chosen": float(np.average(probabilities, weights=weights)),
        "mean_confidence": float(np.mean([pair.confidence for pair in pairs])),
        "source_counts": dict(Counter(pair.label_source for pair in pairs)),
        "split_counts": dict(Counter(pair.split for pair in pairs)),
        "evidence_class": "offline_preference_only",
    }


def rlhf_inter_rater_agreement(
    pairs: Iterable[RLHFPreferencePair],
) -> dict[str, object]:
    """Measure agreement when multiple raters label the same action pair."""

    groups: dict[tuple[object, ...], dict[str, int]] = {}
    for pair in pairs:
        left = (pair.chosen_trace_id, int(pair.chosen_action))
        right = (pair.rejected_trace_id, int(pair.rejected_action))
        canonical = tuple(sorted((left, right)))
        orientation = 1 if left == canonical[0] else 0
        key = (
            pair.context_id,
            pair.pre_decision_state_hash,
            pair.risk_geometry_hash,
            canonical,
        )
        votes = groups.setdefault(key, {})
        existing = votes.get(pair.rater_id)
        if existing is not None and existing != orientation:
            # Keep the group conservative: a rater contradicting themself is
            # not silently converted into agreement.
            votes[pair.rater_id] = -1
        else:
            votes[pair.rater_id] = orientation
    rated_groups = [
        votes for votes in groups.values()
        if len(votes) >= 2
    ]
    valid_groups: list[dict[str, int]] = []
    for votes in rated_groups:
        clean = [vote for vote in votes.values() if vote in (0, 1)]
        if len(clean) >= 2:
            valid_groups.append({"ones": clean.count(1), "zeros": clean.count(0)})
    if not valid_groups:
        return {
            "group_count": len(groups),
            "rated_group_count": 0,
            "agreement": 0.0,
            "disagreement_group_count": 0,
            "evidence_class": "rater_calibration_insufficient",
        }
    agreement_values = [
        max(group["ones"], group["zeros"])
        / max(group["ones"] + group["zeros"], 1)
        for group in valid_groups
    ]
    return {
        "group_count": len(groups),
        "rated_group_count": len(valid_groups),
        "agreement": float(np.mean(agreement_values)),
        "disagreement_group_count": int(
            sum(value < 1.0 for value in agreement_values)
        ),
        "evidence_class": "rater_calibration",
    }


def rlhf_preference_gate(
    metrics: Mapping[str, object],
    *,
    min_pairs: int = 32,
    min_weighted_accuracy: float = 0.55,
    max_log_loss: float = 0.69,
) -> dict[str, object]:
    """Return an offline-quality result without granting MT5 promotion."""

    reasons: list[str] = []
    pair_count = int(metrics.get("pair_count", 0) or 0)
    weighted_accuracy = float(metrics.get("weighted_accuracy", 0.0) or 0.0)
    raw_log_loss = metrics.get("log_loss", float("inf"))
    try:
        log_loss = float(raw_log_loss)
    except (TypeError, ValueError):
        log_loss = float("inf")
    if pair_count < int(min_pairs):
        reasons.append("insufficient_pairs")
    if weighted_accuracy < float(min_weighted_accuracy):
        reasons.append("weighted_accuracy_below_floor")
    if log_loss > float(max_log_loss):
        reasons.append("log_loss_above_ceiling")
    return {
        "passed": not reasons,
        "reasons": reasons,
        "promotion_allowed": False,
        "evidence_class": "offline_preference_only",
        "requires_mt5_frozen_oos_lcb": True,
    }


def validate_rlhf_dataset(pairs: Iterable[RLHFPreferencePair]) -> dict[str, object]:
    """Validate uniqueness and summarize a dataset before any training call."""

    rows = tuple(pairs)
    errors: list[str] = []
    pair_ids = [pair.pair_id for pair in rows]
    duplicates = sorted(pair_id for pair_id, count in Counter(pair_ids).items() if count > 1)
    if duplicates:
        errors.append("duplicate_pair_ids:" + ",".join(duplicates))
    pair_keys = [(
        pair.context_id,
        pair.chosen_trace_id,
        pair.rejected_trace_id,
        int(pair.chosen_action),
        int(pair.rejected_action),
        pair.rater_id,
    ) for pair in rows]
    duplicate_pair_keys = sorted(
        repr(key) for key, count in Counter(pair_keys).items() if count > 1
    )
    if duplicate_pair_keys:
        errors.append("duplicate_pair_keys:" + ",".join(duplicate_pair_keys))
    context_splits: dict[str, set[str]] = {}
    for pair in rows:
        context_splits.setdefault(pair.context_id, set()).add(pair.split)
    split_overlaps = {
        context: sorted(splits)
        for context, splits in context_splits.items()
        if len(splits) > 1
    }
    if split_overlaps:
        errors.append("context_split_overlap")
    context_count = len({pair.context_id for pair in rows})
    human_feedback_raters = {
        pair.rater_id for pair in rows if pair.label_source in {"human", "expert"}
    }
    chosen_actions = [int(pair.chosen_action) for pair in rows]
    chosen_counts = Counter(chosen_actions)
    total_pairs = len(rows)
    buy_count = sum(chosen_counts[a] for a in (1, 3, 5))
    sell_count = sum(chosen_counts[a] for a in (2, 4, 6))
    hold_count = chosen_counts[0]
    p_buy = (buy_count / total_pairs) if total_pairs > 0 else 0.0
    p_sell = (sell_count / total_pairs) if total_pairs > 0 else 0.0
    p_hold = (hold_count / total_pairs) if total_pairs > 0 else 0.0
    min_action_prob = min(p_buy, p_sell, p_hold)

    return {
        "valid": not errors,
        "errors": errors,
        "pair_count": len(rows),
        "context_count": context_count,
        "rater_count": len({pair.rater_id for pair in rows}),
        "human_feedback_rater_count": len(human_feedback_raters),
        "context_split_overlaps": split_overlaps,
        "source_counts": dict(Counter(pair.label_source for pair in rows)),
        "split_counts": dict(Counter(pair.split for pair in rows)),
        "chosen_action_counts": dict(chosen_counts),
        "action_distribution": {
            "BUY": float(p_buy),
            "SELL": float(p_sell),
            "HOLD": float(p_hold),
        },
        "min_action_probability": float(min_action_prob),
        "action_distribution_symmetric": bool(min_action_prob >= 0.20),
        "inter_rater": rlhf_inter_rater_agreement(rows),
        "candidate_only": all(pair.candidate_only for pair in rows),
        "evidence_class": "dataset_contract_only",
    }


def rlhf_dataset_gate(
    dataset_summary: Mapping[str, object],
    *,
    min_pairs: int = 32,
    min_raters: int = 2,
    required_splits: Sequence[str] = ("train", "validation", "test"),
    require_human_feedback: bool = False,
    require_inter_rater_groups: bool = False,
    min_inter_rater_agreement: float | None = None,
    require_action_balance: bool = False,
    min_action_balance: float | None = None,
) -> dict[str, object]:
    """Gate dataset readiness without claiming policy quality or promotion."""

    reasons = list(dataset_summary.get("errors", ()) or ())
    if int(dataset_summary.get("pair_count", 0) or 0) < int(min_pairs):
        reasons.append("insufficient_pairs")
    if int(dataset_summary.get("rater_count", 0) or 0) < int(min_raters):
        reasons.append("insufficient_raters")
    if require_human_feedback and int(
        dataset_summary.get("human_feedback_rater_count", 0) or 0
    ) <= 0:
        reasons.append("no_human_feedback")
    inter_rater = dict(dataset_summary.get("inter_rater", {}) or {})
    if require_inter_rater_groups and int(inter_rater.get("rated_group_count", 0) or 0) <= 0:
        reasons.append("no_inter_rater_groups")
    if min_inter_rater_agreement is not None and int(
        inter_rater.get("rated_group_count", 0) or 0
    ) > 0 and float(inter_rater.get("agreement", 0.0) or 0.0) < float(
        min_inter_rater_agreement
    ):
        reasons.append("inter_rater_agreement_below_floor")
    split_counts = dict(dataset_summary.get("split_counts", {}) or {})
    for split in required_splits:
        if int(split_counts.get(str(split), 0) or 0) <= 0:
            reasons.append(f"missing_split:{split}")

    if require_action_balance or min_action_balance is not None:
        floor = 0.20 if min_action_balance is None else float(min_action_balance)
        min_prob = float(dataset_summary.get("min_action_probability", 0.0) or 0.0)
        if min_prob < floor:
            reasons.append("action_distribution_asymmetric")

    return {
        "ready_for_training": not reasons,
        "reasons": sorted(set(str(reason) for reason in reasons)),
        "promotion_allowed": False,
        "evidence_class": "dataset_contract_only",
        "requires_offline_preference_gate": True,
        "requires_mt5_frozen_oos_lcb": True,
        "requires_human_feedback": bool(require_human_feedback),
        "requires_inter_rater_groups": bool(require_inter_rater_groups),
        "min_inter_rater_agreement": (
            None if min_inter_rater_agreement is None else float(min_inter_rater_agreement)
        ),
    }


def rlhf_dataset_balance_gate(
    dataset_or_pairs: Iterable[object] | Mapping[str, object],
    *,
    min_action_probability: float = 0.20,
) -> dict[str, object]:
    """Validate that action distribution is sufficiently balanced across BUY, SELL, HOLD.

    Enforces: min(p(BUY), p(SELL), p(HOLD)) >= min_action_probability (default 0.20).
    Rejects imbalanced datasets (e.g. 100% BUY) with 'action_distribution_asymmetric'.
    """
    reasons: list[str] = []
    min_threshold = float(min_action_probability)

    if isinstance(dataset_or_pairs, Mapping) and "action_distribution" in dataset_or_pairs:
        # Precomputed dataset summary
        dist = dataset_or_pairs.get("action_distribution") or {}
        p_buy = float(dist.get("BUY", 0.0) or 0.0)
        p_sell = float(dist.get("SELL", 0.0) or 0.0)
        p_hold = float(dist.get("HOLD", 0.0) or 0.0)
        total_pairs = int(dataset_or_pairs.get("pair_count", 0) or 0)
        counts = dict(dataset_or_pairs.get("chosen_action_counts", {}) or {})
    else:
        if isinstance(dataset_or_pairs, Mapping) and "pairs" in dataset_or_pairs:
            raw_items = dataset_or_pairs["pairs"]
        else:
            raw_items = dataset_or_pairs  # type: ignore[assignment]

        actions: list[int] = []
        for item in raw_items:
            if hasattr(item, "chosen_action"):
                actions.append(int(getattr(item, "chosen_action")))
            elif hasattr(item, "pair") and hasattr(getattr(item, "pair"), "chosen_action"):
                actions.append(int(getattr(getattr(item, "pair"), "chosen_action")))
            elif isinstance(item, Mapping):
                if "chosen_action" in item:
                    actions.append(int(item["chosen_action"]))
                elif "chosen" in item and isinstance(item["chosen"], Mapping) and "action" in item["chosen"]:
                    actions.append(int(item["chosen"]["action"]))
                elif "annotation" in item and isinstance(item["annotation"], Mapping):
                    # Check if action is specified or find from candidates
                    annot = item["annotation"]
                    c_id = annot.get("chosen_trace_id")
                    cands = item.get("candidates", [])
                    found = False
                    if c_id and isinstance(cands, Sequence):
                        for c in cands:
                            if isinstance(c, Mapping) and str(c.get("trace_id")) == str(c_id):
                                actions.append(int(c.get("action", -1)))
                                found = True
                                break
                    if not found and "chosen_action" in annot:
                        actions.append(int(annot["chosen_action"]))

        total_pairs = len(actions)
        counts = Counter(actions)
        buy_count = sum(counts[a] for a in (1, 3, 5))
        sell_count = sum(counts[a] for a in (2, 4, 6))
        hold_count = counts[0]

        p_buy = (buy_count / total_pairs) if total_pairs > 0 else 0.0
        p_sell = (sell_count / total_pairs) if total_pairs > 0 else 0.0
        p_hold = (hold_count / total_pairs) if total_pairs > 0 else 0.0

    if total_pairs == 0:
        reasons.append("insufficient_pairs")

    min_p = min(p_buy, p_sell, p_hold)
    if min_p < min_threshold:
        reasons.append("action_distribution_asymmetric")

    return {
        "passed": not reasons,
        "ready_for_training": not reasons,
        "reasons": sorted(set(reasons)),
        "pair_count": total_pairs,
        "action_probabilities": {
            "BUY": float(p_buy),
            "SELL": float(p_sell),
            "HOLD": float(p_hold),
        },
        "chosen_action_counts": dict(counts),
        "min_action_probability": float(min_p),
        "min_action_probability_floor": min_threshold,
        "evidence_class": "dataset_balance_gate_only",
        "promotion_allowed": False,
    }


rlhf_dataset_balance_validation_gate = rlhf_dataset_balance_gate
