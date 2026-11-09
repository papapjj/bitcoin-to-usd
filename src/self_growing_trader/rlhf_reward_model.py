"""A small, auditable reward model for the causal trading RLHF pipeline.

This module is deliberately separate from MT5 outcome accounting.  A reward
model learns a scalar preference score from rater-labelled chosen/rejected
behaviour pairs; it does not read PnL, close rows, future bars, or the
promotion ledger.  The resulting score is still candidate-only until the
policy and frozen-MT5 gates pass.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from math import exp
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .rlhf_preference import (
    RLHFPreferencePair,
    parse_rlhf_preference_row,
    pairwise_preference_metrics,
    validate_rlhf_preference_pair,
)


RLHF_REWARD_SCHEMA_VERSION = "rlhf_reward_pair_v1"


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        z = exp(-min(float(value), 60.0))
        return 1.0 / (1.0 + z)
    z = exp(max(float(value), -60.0))
    return z / (1.0 + z)


def _finite_vector(value: object) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0 or not bool(np.all(np.isfinite(vector))):
        return None
    return vector


@dataclass(frozen=True)
class RLHFRewardPair:
    """One preference pair with decision-time feature vectors.

    The vectors are the features shown to the rater for the two candidate
    behaviours.  They must be causal and use one versioned feature schema.
    Any future/outcome feature is rejected before fitting.
    """

    pair: RLHFPreferencePair
    chosen_features: tuple[float, ...]
    rejected_features: tuple[float, ...]
    feature_schema: str
    feature_causal: bool = True
    candidate_only: bool = True
    schema_version: str = RLHF_REWARD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        chosen = _finite_vector(self.chosen_features)
        rejected = _finite_vector(self.rejected_features)
        errors = validate_rlhf_reward_pair(self)
        if errors:
            raise ValueError("invalid RLHF reward pair: " + "; ".join(errors))
        assert chosen is not None and rejected is not None
        object.__setattr__(self, "chosen_features", tuple(float(x) for x in chosen))
        object.__setattr__(self, "rejected_features", tuple(float(x) for x in rejected))

    @property
    def pair_id(self) -> str:
        return self.pair.pair_id

    def to_row(self) -> dict[str, object]:
        row = self.pair.to_row()
        row.update({
            "reward_schema_version": self.schema_version,
            "feature_schema": self.feature_schema,
            "chosen_features": list(self.chosen_features),
            "rejected_features": list(self.rejected_features),
            "reward_feature_causal": self.feature_causal,
            "reward_candidate_only": self.candidate_only,
        })
        return row


def validate_rlhf_reward_pair(pair: RLHFRewardPair) -> tuple[str, ...]:
    errors = list(validate_rlhf_preference_pair(pair.pair))
    if str(getattr(pair, "schema_version", "")) != RLHF_REWARD_SCHEMA_VERSION:
        errors.append("unsupported_reward_schema_version")
    if not str(getattr(pair, "feature_schema", "") or "").strip():
        errors.append("missing_feature_schema")
    chosen = _finite_vector(getattr(pair, "chosen_features", None))
    rejected = _finite_vector(getattr(pair, "rejected_features", None))
    if chosen is None:
        errors.append("chosen_features_must_be_finite_nonempty")
    if rejected is None:
        errors.append("rejected_features_must_be_finite_nonempty")
    if chosen is not None and rejected is not None and chosen.shape != rejected.shape:
        errors.append("chosen_rejected_feature_shape_mismatch")
    if not bool(getattr(pair, "feature_causal", False)):
        errors.append("reward_future_feature_leakage_declared")
    if not bool(getattr(pair, "candidate_only", False)):
        errors.append("reward_candidate_only_must_be_true")
    return tuple(errors)


def parse_rlhf_reward_row(row: Mapping[str, object]) -> RLHFRewardPair | None:
    preference = parse_rlhf_preference_row(row)
    if preference is None:
        return None
    try:
        return RLHFRewardPair(
            pair=preference,
            chosen_features=tuple(float(x) for x in row.get("chosen_features", ())),
            rejected_features=tuple(float(x) for x in row.get("rejected_features", ())),
            feature_schema=str(row.get("feature_schema", "") or "").strip(),
            feature_causal=row.get("reward_feature_causal") is True,
            candidate_only=row.get("reward_candidate_only") is True,
            schema_version=str(row.get("reward_schema_version", "") or "").strip(),
        )
    except (TypeError, ValueError):
        return None


def load_rlhf_reward_jsonl(
    path: str | Path,
) -> tuple[tuple[RLHFRewardPair, ...], tuple[int, ...]]:
    rows: list[RLHFRewardPair] = []
    invalid_lines: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                invalid_lines.append(line_number)
                continue
            parsed = parse_rlhf_reward_row(value) if isinstance(value, Mapping) else None
            if parsed is None:
                invalid_lines.append(line_number)
            else:
                rows.append(parsed)
    return tuple(rows), tuple(invalid_lines)


@dataclass(frozen=True)
class RLHFRewardModel:
    """Deterministic linear Bradley--Terry reward model.

    A linear model is intentional for V1: every feature contribution can be
    audited, serialized, replayed, and compared without hiding a new neural
    writer inside the trading policy.
    """

    feature_schema: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[float, ...]
    bias: float
    model_version: str = "rlhf_reward_linear_v1"

    def __post_init__(self) -> None:
        arrays = [_finite_vector(value) for value in (self.mean, self.scale, self.weights)]
        if any(value is None for value in arrays):
            raise ValueError("reward model parameters must be finite non-empty vectors")
        mean, scale, weights = arrays
        assert mean is not None and scale is not None and weights is not None
        if not (mean.shape == scale.shape == weights.shape):
            raise ValueError("reward model parameter shapes must match")
        if np.any(scale <= 0.0):
            raise ValueError("reward model scale must be positive")
        if not np.isfinite(float(self.bias)):
            raise ValueError("reward model bias must be finite")
        if not str(self.feature_schema).strip():
            raise ValueError("reward model feature schema is required")

    @property
    def feature_count(self) -> int:
        return len(self.weights)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "RLHFRewardModel":
        version = str(row.get("model_version", "rlhf_reward_linear_v1") or "")
        if "bilinear" in version:
            return RLHFBilinearRewardModel.from_row(row)
        model = cls(
            feature_schema=str(row.get("feature_schema", "") or "").strip(),
            mean=tuple(float(value) for value in row.get("mean", ())),
            scale=tuple(float(value) for value in row.get("scale", ())),
            weights=tuple(float(value) for value in row.get("weights", ())),
            bias=float(row.get("bias")),
            model_version=str(row.get("model_version", "rlhf_reward_linear_v1") or ""),
        )
        declared_hash = str(row.get("sha256", "") or "").strip()
        if declared_hash and declared_hash != model.sha256:
            raise ValueError("RLHF reward model sha256 mismatch")
        return model

    def score(self, features: Sequence[float] | np.ndarray) -> float:
        values = _finite_vector(features)
        if values is None or values.size != self.feature_count:
            raise ValueError("reward feature vector has the wrong shape")
        normalized = (values - np.asarray(self.mean)) / np.asarray(self.scale)
        return float(np.dot(np.asarray(self.weights), normalized) + float(self.bias))

    def score_pair(self, pair: RLHFRewardPair) -> tuple[float, float]:
        if validate_rlhf_reward_pair(pair):
            raise ValueError("cannot score an invalid RLHF reward pair")
        if pair.feature_schema != self.feature_schema:
            raise ValueError("reward feature schema mismatch")
        return self.score(pair.chosen_features), self.score(pair.rejected_features)

    def to_row(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "feature_schema": self.feature_schema,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": list(self.weights),
            "bias": float(self.bias),
            "sha256": self.sha256,
            "evidence_class": "offline_reward_model_only",
            "promotion_allowed": False,
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            {
                "model_version": self.model_version,
                "feature_schema": self.feature_schema,
                "mean": list(self.mean),
                "scale": list(self.scale),
                "weights": list(self.weights),
                "bias": float(self.bias),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


@dataclass(frozen=True)
class RLHFBilinearRewardModel(RLHFRewardModel):
    r"""Deterministic Bilinear State-Action reward model.

    Scores state-action pairs via R(s, a) = w_a^T \tilde{s} + b_a,
    where W \in \mathbb{R}^{7 \times 137} and b \in \mathbb{R}^7.
    Intra-state candidate pairs (s_c = s_r = s) produce non-zero gradients on
    w_{a_c} and w_{a_r}:
        \partial \Delta R / \partial w_{a_c} = \tilde{s} \ne 0
        \partial \Delta R / \partial w_{a_r} = -\tilde{s} \ne 0
    guaranteeing market features receive meaningful updates during preference learning.
    """

    feature_schema: str
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    weights: tuple[tuple[float, ...], ...]
    bias: tuple[float, ...] = (0.0,) * 7
    model_version: str = "rlhf_reward_bilinear_v2"

    def __post_init__(self) -> None:
        mean_arr = _finite_vector(self.mean)
        scale_arr = _finite_vector(self.scale)
        if mean_arr is None or scale_arr is None:
            raise ValueError("bilinear reward model mean and scale must be finite non-empty vectors")
        if mean_arr.size < 137 or scale_arr.size < 137:
            raise ValueError("bilinear reward model mean and scale must have at least 137 state features")
        if np.any(scale_arr[:137] <= 0.0):
            raise ValueError("bilinear reward model scale must be positive")
        if not str(self.feature_schema).strip():
            raise ValueError("reward model feature schema is required")

        raw_w = self.weights
        if isinstance(raw_w, np.ndarray):
            w_arr = np.asarray(raw_w, dtype=np.float64)
        elif isinstance(raw_w, (tuple, list)):
            if len(raw_w) == 959 and not isinstance(raw_w[0], (tuple, list)):
                w_arr = np.asarray(raw_w, dtype=np.float64).reshape(7, 137)
            else:
                w_arr = np.asarray([np.asarray(row, dtype=np.float64).reshape(-1) for row in raw_w], dtype=np.float64)
        else:
            raise ValueError("bilinear reward model weights must be a 7x137 sequence or array")

        if w_arr.shape != (7, 137) or not bool(np.all(np.isfinite(w_arr))):
            raise ValueError(f"bilinear reward model weights must have shape (7, 137) and be finite, got {w_arr.shape}")

        raw_b = self.bias
        if isinstance(raw_b, (int, float)) and not isinstance(raw_b, bool):
            b_arr = np.full(7, float(raw_b), dtype=np.float64)
        else:
            b_arr = np.asarray(raw_b, dtype=np.float64).reshape(-1)
        if b_arr.size != 7 or not bool(np.all(np.isfinite(b_arr))):
            raise ValueError(f"bilinear reward model bias must have 7 finite elements, got shape {b_arr.shape}")

        object.__setattr__(self, "mean", tuple(float(x) for x in mean_arr))
        object.__setattr__(self, "scale", tuple(float(x) for x in scale_arr))
        object.__setattr__(self, "weights", tuple(tuple(float(x) for x in row) for row in w_arr))
        object.__setattr__(self, "bias", tuple(float(x) for x in b_arr))
        object.__setattr__(self, "model_version", str(self.model_version or "rlhf_reward_bilinear_v2"))

    @property
    def feature_count(self) -> int:
        return 137

    @property
    def weights_array(self) -> np.ndarray:
        return np.asarray(self.weights, dtype=np.float64)

    @property
    def flattened_weights(self) -> tuple[float, ...]:
        return tuple(float(x) for row in self.weights for x in row)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "RLHFBilinearRewardModel":
        raw_weights = row.get("weights", ())
        if isinstance(raw_weights, (list, tuple)) and len(raw_weights) == 959 and not isinstance(raw_weights[0], (list, tuple)):
            arr = np.asarray(raw_weights, dtype=np.float64).reshape(7, 137)
            weights = tuple(tuple(float(x) for x in arr[a]) for a in range(7))
        elif isinstance(raw_weights, (list, tuple)) and len(raw_weights) == 7:
            weights = tuple(tuple(float(x) for x in row_w) for row_w in raw_weights)
        else:
            arr = np.asarray(raw_weights, dtype=np.float64).reshape(7, 137)
            weights = tuple(tuple(float(x) for x in arr[a]) for a in range(7))

        raw_bias = row.get("bias", 0.0)
        if isinstance(raw_bias, (list, tuple)):
            bias = tuple(float(x) for x in raw_bias)
        else:
            bias = (float(raw_bias),) * 7

        model = cls(
            feature_schema=str(row.get("feature_schema", "") or "").strip(),
            mean=tuple(float(value) for value in row.get("mean", ())),
            scale=tuple(float(value) for value in row.get("scale", ())),
            weights=weights,
            bias=bias,
            model_version=str(row.get("model_version", "rlhf_reward_bilinear_v2") or ""),
        )
        declared_hash = str(row.get("sha256", "") or "").strip()
        if declared_hash and declared_hash != model.sha256:
            raise ValueError("RLHF bilinear reward model sha256 mismatch")
        return model

    def score_action(self, state: Sequence[float] | np.ndarray, action: int) -> float:
        s = _finite_vector(state)
        if s is None or s.size < 137:
            raise ValueError("state vector must be finite and at least 137-dimensional")
        s_137 = s[:137]
        act = int(action)
        if not 0 <= act < 7:
            raise ValueError(f"action {act} is outside valid entry action range [0..6]")
        mean = np.asarray(self.mean[:137], dtype=np.float64)
        scale = np.asarray(self.scale[:137], dtype=np.float64)
        scale = np.where(scale < 1e-8, 1.0, scale)
        norm_s = (s_137 - mean) / scale
        w_a = np.asarray(self.weights[act], dtype=np.float64)
        b_a = float(self.bias[act])
        return float(np.dot(w_a, norm_s) + b_a)

    def score(self, features: Sequence[float] | np.ndarray) -> float:
        values = _finite_vector(features)
        if values is None:
            raise ValueError("reward feature vector must be finite and non-empty")
        if values.size >= 144:
            state = values[:137]
            one_hot = values[137:144]
            action = int(np.argmax(one_hot)) if np.any(one_hot > 0.0) else 0
            return self.score_action(state, action)
        elif values.size == 137:
            return self.score_action(values, 0)
        else:
            raise ValueError(f"feature vector has unexpected shape {values.shape}, expected 137 or 144")

    def score_pair(self, pair: RLHFRewardPair) -> tuple[float, float]:
        if validate_rlhf_reward_pair(pair):
            raise ValueError("cannot score an invalid RLHF reward pair")
        if pair.feature_schema != self.feature_schema:
            raise ValueError("reward feature schema mismatch")
        return self.score(pair.chosen_features), self.score(pair.rejected_features)

    def to_row(self) -> dict[str, object]:
        return {
            "model_version": self.model_version,
            "feature_schema": self.feature_schema,
            "mean": list(self.mean),
            "scale": list(self.scale),
            "weights": [list(row) for row in self.weights],
            "bias": list(self.bias),
            "sha256": self.sha256,
            "evidence_class": "offline_reward_model_only",
            "promotion_allowed": False,
        }

    @property
    def sha256(self) -> str:
        canonical = json.dumps(
            {
                "model_version": self.model_version,
                "feature_schema": self.feature_schema,
                "mean": list(self.mean),
                "scale": list(self.scale),
                "weights": [list(row) for row in self.weights],
                "bias": list(self.bias),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()


def load_rlhf_reward_model(
    path: str | Path,
    *,
    require_test_gate: bool = False,
    expected_feedback_mode: str | None = None,
) -> RLHFRewardModel:
    """Load one immutable reward-model artifact for the policy phase.

    The loader accepts either the serialized ``RLHFRewardModel.to_row`` shape
    or the report written by ``train_rlhf_reward_model.py``.  A report wrapper
    is accepted only when its held-out reward-model ``test_gate`` passed; this
    prevents a failed calibration report from silently becoming an MT5 policy
    scorer.  The MT5 worker passes ``require_test_gate=True`` so a raw row is
    never enough for a policy update.  The declared model SHA256 is checked by
    ``from_row`` before the object can reach PPO.  Training a reward model is
    deliberately kept out of the MT5 worker.
    """

    artifact = Path(path)
    with artifact.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError("RLHF reward model artifact must be a JSON object")
    model_payload: Mapping[str, object] = payload
    if "model" in payload:
        test_gate = payload.get("test_gate")
        if not isinstance(test_gate, Mapping) or test_gate.get("passed") is not True:
            raise ValueError(
                "RLHF reward model report test_gate must pass before MT5 loading"
            )
        nested = payload.get("model")
        if not isinstance(nested, Mapping):
            raise ValueError("RLHF reward model report has an invalid model payload")
        expected_mode = str(expected_feedback_mode or "").strip().lower()
        if expected_mode:
            actual_mode = str(payload.get("feedback_mode", "")).strip().lower()
            if actual_mode != expected_mode:
                raise ValueError(
                    "RLHF reward model feedback_mode mismatch: "
                    f"expected {expected_mode!r}, got {actual_mode or '<missing>'!r}"
                )
        model_payload = nested
    elif require_test_gate:
        raise ValueError(
            "MT5 RLHF reward model artifact must be a report with a passed test_gate"
        )
    return RLHFRewardModel.from_row(model_payload)


def fit_rlhf_reward_model(
    pairs: Sequence[RLHFRewardPair],
    *,
    epochs: int = 200,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    model_type: str = "linear",
) -> tuple[RLHFRewardModel, dict[str, object]]:
    """Fit only on already split train pairs using weighted BT likelihood."""

    if str(model_type or "linear").lower() == "bilinear":
        return fit_rlhf_bilinear_reward_model(
            pairs, epochs=epochs, learning_rate=learning_rate, l2=l2
        )

    rows = tuple(pairs)
    if not rows:
        raise ValueError("reward model needs at least one training pair")
    errors = [error for pair in rows for error in validate_rlhf_reward_pair(pair)]
    if errors:
        raise ValueError("invalid reward training data: " + ",".join(sorted(set(errors))))
    schemas = {pair.feature_schema for pair in rows}
    if len(schemas) != 1:
        raise ValueError("reward training data mixes feature schemas")
    epochs = int(epochs)
    learning_rate = float(learning_rate)
    l2 = float(l2)
    if epochs <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("reward fit hyperparameters are invalid")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("reward model l2 must be non-negative")

    all_features = np.asarray(
        [feature for pair in rows for feature in (pair.chosen_features, pair.rejected_features)],
        dtype=np.float64,
    )
    mean = all_features.mean(axis=0)
    scale = all_features.std(axis=0)
    scale = np.where(scale < 1e-8, 1.0, scale)
    chosen = (np.asarray([pair.chosen_features for pair in rows]) - mean) / scale
    rejected = (np.asarray([pair.rejected_features for pair in rows]) - mean) / scale
    confidence = np.asarray([float(pair.pair.confidence) for pair in rows], dtype=np.float64)
    if not bool(np.any(confidence > 0.0)):
        raise ValueError("reward training data has no positive-confidence pair")
    weights = np.zeros(chosen.shape[1], dtype=np.float64)
    bias = 0.0
    losses: list[float] = []
    for _ in range(epochs):
        margins = np.dot(chosen - rejected, weights)
        probabilities = np.asarray([_sigmoid(value) for value in margins])
        signal = confidence * (1.0 - probabilities)
        gradient = np.mean(signal[:, None] * (chosen - rejected), axis=0) - l2 * weights
        weights += learning_rate * gradient
        losses.append(float(np.average(np.logaddexp(0.0, -margins), weights=confidence)))

    # The pairwise objective is invariant to a constant.  Keeping the bias at
    # zero makes the model hash stable and avoids inventing a calibration claim.
    model = RLHFRewardModel(
        feature_schema=next(iter(schemas)),
        mean=tuple(float(value) for value in mean),
        scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
        bias=float(bias),
    )
    return model, {
        "pair_count": len(rows),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "model_sha256": model.sha256,
        "evidence_class": "offline_reward_model_only",
        "promotion_allowed": False,
    }


def fit_rlhf_bilinear_reward_model(
    pairs: Sequence[RLHFRewardPair | Mapping[str, object]] | None = None,
    *,
    tasks: Sequence[Mapping[str, object]] | None = None,
    epochs: int = 200,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
) -> tuple[RLHFBilinearRewardModel, dict[str, object]]:
    r"""Fit a Bilinear State-Action reward model using weighted Bradley--Terry likelihood.

    R(s, a) = w_a^T \tilde{s} + b_a
    where W \in \mathbb{R}^{7 \times 137} and b \in \mathbb{R}^7.
    Intra-state candidate pairs (s_c = s_r = s) produce non-zero gradients on
    w_{a_c} and w_{a_r}:
        \partial \Delta R / \partial w_{a_c} = \tilde{s} \ne 0
        \partial \Delta R / \partial w_{a_r} = -\tilde{s} \ne 0
    guaranteeing market features receive meaningful updates during preference learning.
    """
    raw_data = tasks if pairs is None and tasks is not None else pairs
    if raw_data is None or len(raw_data) == 0:
        raise ValueError("reward model needs at least one training pair or task")

    chosen_states: list[np.ndarray] = []
    rejected_states: list[np.ndarray] = []
    chosen_actions: list[int] = []
    rejected_actions: list[int] = []
    confidences: list[float] = []
    schemas: set[str] = set()

    for item in raw_data:
        if isinstance(item, RLHFRewardPair):
            errs = validate_rlhf_reward_pair(item)
            if errs:
                raise ValueError("invalid reward training data: " + ",".join(sorted(set(errs))))
            c_f = np.asarray(item.chosen_features, dtype=np.float64)
            r_f = np.asarray(item.rejected_features, dtype=np.float64)
            if c_f.size < 137 or r_f.size < 137:
                raise ValueError("candidate features must have at least 137 state dimensions")
            chosen_states.append(c_f[:137])
            rejected_states.append(r_f[:137])
            chosen_actions.append(int(item.pair.chosen_action))
            rejected_actions.append(int(item.pair.rejected_action))
            confidences.append(float(item.pair.confidence))
            schemas.add(item.feature_schema)
        elif isinstance(item, Mapping) and "candidates" in item and "annotation" in item:
            annot = item.get("annotation", {})
            if not isinstance(annot, Mapping):
                raise ValueError("task annotation must be a mapping")
            chosen_id = str(annot.get("chosen_trace_id", "") or "").strip()
            rejected_id = str(annot.get("rejected_trace_id", "") or "").strip()
            if not chosen_id or not rejected_id or chosen_id == rejected_id:
                raise ValueError("annotation task must have distinct chosen_trace_id and rejected_trace_id")
            cands = item.get("candidates", [])
            cand_map = {str(c.get("trace_id")): c for c in cands if isinstance(c, Mapping)}
            if chosen_id not in cand_map or rejected_id not in cand_map:
                raise ValueError("annotation task candidates missing chosen or rejected trace ID")
            c_c = cand_map[chosen_id]
            r_c = cand_map[rejected_id]
            c_f = np.asarray(c_c.get("features", ()), dtype=np.float64).reshape(-1)
            r_f = np.asarray(r_c.get("features", ()), dtype=np.float64).reshape(-1)
            if c_f.size < 137 or r_f.size < 137:
                raise ValueError("task candidate features must have at least 137 state dimensions")
            chosen_states.append(c_f[:137])
            rejected_states.append(r_f[:137])
            chosen_actions.append(int(c_c.get("action", -1)))
            rejected_actions.append(int(r_c.get("action", -1)))
            confidences.append(float(annot.get("confidence", 1.0) or 1.0))
            schemas.add("mt5_raw_obs137_action_onehot_v1")
        elif isinstance(item, Mapping) and "chosen_features" in item and "rejected_features" in item:
            c_f = np.asarray(item["chosen_features"], dtype=np.float64).reshape(-1)
            r_f = np.asarray(item["rejected_features"], dtype=np.float64).reshape(-1)
            if c_f.size < 137 or r_f.size < 137:
                raise ValueError("reward row features must have at least 137 state dimensions")
            c_act = int(item.get("chosen_action", np.argmax(c_f[137:144]) if c_f.size >= 144 else 0))
            r_act = int(item.get("rejected_action", np.argmax(r_f[137:144]) if r_f.size >= 144 else 0))
            chosen_states.append(c_f[:137])
            rejected_states.append(r_f[:137])
            chosen_actions.append(c_act)
            rejected_actions.append(r_act)
            confidences.append(float(item.get("confidence", 1.0) or 1.0))
            schemas.add(str(item.get("feature_schema", "mt5_raw_obs137_action_onehot_v1")))
        else:
            raise ValueError(f"unsupported item type in reward training data: {type(item)}")

    epochs = int(epochs)
    learning_rate = float(learning_rate)
    l2 = float(l2)
    if epochs <= 0 or not np.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("reward fit hyperparameters are invalid")
    if not np.isfinite(l2) or l2 < 0.0:
        raise ValueError("reward model l2 must be non-negative")

    c_acts = np.asarray(chosen_actions, dtype=np.int64)
    r_acts = np.asarray(rejected_actions, dtype=np.int64)
    if np.any(c_acts < 0) or np.any(c_acts >= 7) or np.any(r_acts < 0) or np.any(r_acts >= 7):
        raise ValueError("action index out of range [0..6]")
    if np.any(c_acts == r_acts):
        raise ValueError("chosen and rejected actions must differ")

    conf_arr = np.asarray(confidences, dtype=np.float64)
    if not bool(np.any(conf_arr > 0.0)):
        raise ValueError("reward training data has no positive-confidence pair")

    c_s = np.asarray(chosen_states, dtype=np.float64)
    r_s = np.asarray(rejected_states, dtype=np.float64)
    all_s = np.concatenate([c_s, r_s], axis=0)
    raw_scale = all_s.std(axis=0)
    mean = np.where(raw_scale < 1e-8, 0.0, all_s.mean(axis=0))
    scale = np.where(raw_scale < 1e-8, 1.0, raw_scale)

    norm_c_s = (c_s - mean) / scale
    norm_r_s = (r_s - mean) / scale

    N = len(chosen_actions)
    W = np.zeros((7, 137), dtype=np.float64)
    b = np.zeros(7, dtype=np.float64)
    losses: list[float] = []

    for _ in range(epochs):
        r_c = np.sum(W[c_acts] * norm_c_s, axis=1) + b[c_acts]
        r_r = np.sum(W[r_acts] * norm_r_s, axis=1) + b[r_acts]
        margins = r_c - r_r
        probs = np.asarray([_sigmoid(m) for m in margins])
        signal = conf_arr * (1.0 - probs)

        grad_W = np.zeros_like(W)
        grad_b = np.zeros_like(b)
        np.add.at(grad_W, c_acts, signal[:, None] * norm_c_s)
        np.add.at(grad_W, r_acts, -signal[:, None] * norm_r_s)
        np.add.at(grad_b, c_acts, signal)
        np.add.at(grad_b, r_acts, -signal)

        grad_W = grad_W / N - l2 * W
        grad_b = grad_b / N

        W += learning_rate * grad_W
        b += learning_rate * grad_b
        losses.append(float(np.average(np.logaddexp(0.0, -margins), weights=conf_arr)))

    model = RLHFBilinearRewardModel(
        feature_schema=next(iter(schemas)) if schemas else "mt5_raw_obs137_action_onehot_v1",
        mean=tuple(float(v) for v in mean),
        scale=tuple(float(v) for v in scale),
        weights=tuple(tuple(float(v) for v in row) for row in W),
        bias=tuple(float(v) for v in b),
    )
    return model, {
        "pair_count": N,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "model_sha256": model.sha256,
        "model_version": model.model_version,
        "evidence_class": "offline_reward_model_only",
        "promotion_allowed": False,
        "market_weights_non_zero_count": int(np.count_nonzero(W != 0.0)),
    }


def score_rlhf_reward_pairs(
    model: RLHFRewardModel,
    pairs: Sequence[RLHFRewardPair],
) -> dict[str, object]:
    rows = tuple(pairs)
    chosen: list[float] = []
    rejected: list[float] = []
    for pair in rows:
        left, right = model.score_pair(pair)
        chosen.append(left)
        rejected.append(right)
    metrics = pairwise_preference_metrics(
        [pair.pair for pair in rows], chosen, rejected
    )
    metrics["model_sha256"] = model.sha256
    metrics["feature_schema"] = model.feature_schema
    metrics["evidence_class"] = "offline_reward_model_only"
    metrics["promotion_allowed"] = False
    return metrics


def rlhf_reward_model_gate(
    metrics: Mapping[str, object],
    *,
    min_pairs: int = 32,
    min_weighted_accuracy: float = 0.55,
    max_log_loss: float = 0.69,
) -> dict[str, object]:
    reasons: list[str] = []
    pair_count = int(metrics.get("pair_count", 0) or 0)
    weighted_accuracy = float(metrics.get("weighted_accuracy", 0.0) or 0.0)
    try:
        log_loss = float(metrics.get("log_loss", float("inf")))
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
        "evidence_class": "offline_reward_model_only",
        "requires_rlhf_ppo_policy_gate": True,
        "requires_mt5_frozen_oos_lcb": True,
    }
