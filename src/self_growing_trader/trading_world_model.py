"""Causal, multi-head trading world model used as evidence—not an entry bootstrap.

The module intentionally keeps the model small and inspectable.  It predicts
observable market/execution targets at several horizons and exposes calibrated
uncertainty.  A prediction is never converted directly into a BUY/SELL logit;
it may only influence shadow/replay work and risk/critic context until the
separate contextual-expectancy system has verified a positive lower bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

WORLD_MODEL_SCHEMA_VERSION = 1
RETURN_HORIZONS: tuple[int, ...] = (1, 5, 15, 60)
RETURN_FEATURE_INDEX = {1: 0, 5: 2, 15: 3, 60: 5}
TARGET_NAMES: tuple[str, ...] = (
    "return_1",
    "return_5",
    "return_15",
    "return_60",
    "volatility_1",
    "spread_atr_1",
    "execution_cost_r",
    "net_r",
    "fill_probability",
    "tp_before_sl",
    "mae_r",
    "mfe_r",
)
_PROBABILITY_TARGETS = frozenset({"fill_probability", "tp_before_sl"})


def _finite_vector(value: object, size: int) -> np.ndarray:
    try:
        raw = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raw = np.zeros(0, dtype=float)
    out = np.zeros(max(1, int(size)), dtype=float)
    if raw.size:
        out[: min(out.size, raw.size)] = raw[:out.size]
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _pool(values: object, size: int) -> np.ndarray:
    source = _finite_vector(values, max(1, np.asarray(values).size if hasattr(values, "__len__") else 1))
    out = np.zeros(max(1, int(size)), dtype=float)
    if source.size == 0:
        return out
    boundaries = np.linspace(0, source.size, out.size + 1)
    for index in range(out.size):
        start = int(np.floor(boundaries[index]))
        stop = min(source.size, max(start + 1, int(np.ceil(boundaries[index + 1]))))
        if start < stop:
            out[index] = float(np.mean(source[start:stop]))
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30.0, 30.0)))


def _float(mapping: Mapping[str, object] | None, key: str, default: float = 0.0) -> float:
    try:
        value = float((mapping or {}).get(key, default))
    except (TypeError, ValueError):
        value = default
    return value if np.isfinite(value) else float(default)


@dataclass
class _PendingExample:
    """A causal decision record waiting for multi-step forward labels."""

    sequence: int
    features: np.ndarray
    steps_observed: int = 0
    cumulative_return: float = 0.0


@dataclass
class TradingWorldModel:
    """Bounded online linear heads with calibrated residual uncertainty.

    The heads deliberately use only causal observations.  Multi-horizon return
    labels are attached after the required number of later decisions arrives.
    Execution heads update only when execution/lifecycle evidence exists.
    """

    input_size: int = 40
    target_names: tuple[str, ...] = TARGET_NAMES
    learning_rate: float = 0.004
    max_history: int = 4096
    min_samples_for_evidence: int = 32
    weights: np.ndarray | None = field(default=None, repr=False)
    bias: np.ndarray | None = field(default=None, repr=False)
    samples: np.ndarray | None = field(default=None, repr=False)
    residual_ema: np.ndarray | None = field(default=None, repr=False)
    target_mean: np.ndarray | None = field(default=None, repr=False)
    target_var: np.ndarray | None = field(default=None, repr=False)
    sequence: int = 0
    pending: list[_PendingExample] = field(default_factory=list, repr=False)
    last_prediction: dict[str, object] = field(default_factory=dict)

    @classmethod
    def seed(
        cls,
        *,
        observation_size: int = 119,
        action_size: int = 20,
        seed: int | None = None,
        input_size: int = 40,
    ) -> "TradingWorldModel":
        del observation_size, action_size
        rng = np.random.default_rng(seed)
        model = cls(input_size=max(8, int(input_size)))
        model.ensure_compatible()
        # Small non-zero state breaks complete symmetry without encoding a
        # directional prior.  The amplitude is deliberately tiny.
        model.weights = rng.normal(0.0, 1e-4, size=model.weights.shape)
        return model

    def ensure_compatible(self, *, observation_size: int | None = None, action_size: int | None = None) -> None:
        del observation_size, action_size
        self.input_size = max(8, int(getattr(self, "input_size", 40) or 40))
        self.target_names = tuple(getattr(self, "target_names", TARGET_NAMES) or TARGET_NAMES)
        target_count = len(self.target_names)
        if self.weights is None or np.asarray(self.weights).shape != (target_count, self.input_size):
            self.weights = np.zeros((target_count, self.input_size), dtype=float)
        else:
            self.weights = np.nan_to_num(np.asarray(self.weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
        for name, default in (("bias", 0.0), ("samples", 0.0), ("residual_ema", 1.0), ("target_mean", 0.0), ("target_var", 1.0)):
            value = getattr(self, name, None)
            array = np.asarray(value if value is not None else np.full(target_count, default), dtype=float).reshape(-1)
            if array.size != target_count:
                resized = np.full(target_count, default, dtype=float)
                resized[: min(target_count, array.size)] = array[:target_count]
                array = resized
            setattr(self, name, np.nan_to_num(array, nan=default, posinf=default, neginf=default))
        self.learning_rate = float(np.clip(float(getattr(self, "learning_rate", 0.004) or 0.004), 1e-5, 0.05))
        self.max_history = max(64, int(getattr(self, "max_history", 4096) or 4096))
        self.min_samples_for_evidence = max(4, int(getattr(self, "min_samples_for_evidence", 32) or 32))
        self.sequence = max(0, int(getattr(self, "sequence", 0) or 0))
        raw_pending = list(getattr(self, "pending", []) or [])
        clean: list[_PendingExample] = []
        for item in raw_pending[-self.max_history:]:
            if isinstance(item, _PendingExample):
                clean.append(_PendingExample(
                    int(item.sequence),
                    _finite_vector(item.features, self.input_size),
                    max(0, int(getattr(item, "steps_observed", 0) or 0)),
                    float(getattr(item, "cumulative_return", 0.0) or 0.0),
                ))
            elif isinstance(item, Mapping):
                clean.append(_PendingExample(
                    int(item.get("sequence", 0) or 0),
                    _finite_vector(item.get("features"), self.input_size),
                    max(0, int(item.get("steps_observed", 0) or 0)),
                    float(item.get("cumulative_return", 0.0) or 0.0),
                ))
        self.pending = clean
        self.last_prediction = dict(getattr(self, "last_prediction", {}) or {})

    def copy(self) -> "TradingWorldModel":
        self.ensure_compatible()
        return TradingWorldModel(
            input_size=self.input_size,
            target_names=tuple(self.target_names),
            learning_rate=self.learning_rate,
            max_history=self.max_history,
            min_samples_for_evidence=self.min_samples_for_evidence,
            weights=self.weights.copy(),
            bias=self.bias.copy(),
            samples=self.samples.copy(),
            residual_ema=self.residual_ema.copy(),
            target_mean=self.target_mean.copy(),
            target_var=self.target_var.copy(),
            sequence=int(self.sequence),
            pending=[
                _PendingExample(item.sequence, item.features.copy(), int(item.steps_observed), float(item.cumulative_return))
                for item in self.pending
            ],
            last_prediction={key: value for key, value in self.last_prediction.items()},
        )

    def reset_runtime_state(self) -> None:
        self.ensure_compatible()
        self.pending = []
        self.last_prediction = {}
        self.sequence = 0

    def encode(self, observation: object, action: object | None = None) -> np.ndarray:
        """Encode market, position, account, execution and action separately."""
        obs = _finite_vector(observation, 119)
        act = _finite_vector(action, 20)
        # The exact slices mirror the canonical 119-slot protocol. Pooling keeps
        # the online heads bounded while preserving semantic partitions.
        market = np.concatenate([obs[:34], obs[82:108]])
        positions = obs[34:74]
        account = obs[74:82]
        execution = obs[108:119]
        encoded = np.concatenate([
            _pool(market, 16),
            _pool(positions, 8),
            _pool(account, 4),
            _pool(execution, 4),
            _pool(act[:12], 4),
            _pool(act[12:20], 4),
        ])
        return _finite_vector(encoded, self.input_size)

    def _raw_prediction(self, features: np.ndarray) -> np.ndarray:
        self.ensure_compatible()
        return self.weights @ features + self.bias

    def predict(self, observation: object, action: object | None = None) -> dict[str, object]:
        self.ensure_compatible()
        features = self.encode(observation, action)
        raw = self._raw_prediction(features)
        values = np.tanh(raw) * 5.0
        for index, name in enumerate(self.target_names):
            if name in _PROBABILITY_TARGETS:
                values[index] = float(_sigmoid(raw[index]))
        residual = np.maximum(0.02, np.asarray(self.residual_ema, dtype=float))
        support = np.asarray(self.samples, dtype=float)
        data_uncertainty = np.sqrt(1.0 / np.maximum(support, 1.0))
        uncertainty = np.clip(residual * (0.35 + data_uncertainty), 0.01, 5.0)
        names = {name: float(values[index]) for index, name in enumerate(self.target_names)}
        uncertainty_by_head = {name: float(uncertainty[index]) for index, name in enumerate(self.target_names)}
        return {
            "schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "features": features.copy(),
            "predictions": names,
            "uncertainty_by_head": uncertainty_by_head,
            "samples_by_head": {name: int(support[index]) for index, name in enumerate(self.target_names)},
            "mean_uncertainty": float(np.mean(uncertainty)) if uncertainty.size else 1.0,
            "min_support": int(np.min(support)) if support.size else 0,
        }

    def context(self, observation: object, action: object | None = None) -> dict[str, object]:
        prediction = self.predict(observation, action)
        values = dict(prediction["predictions"])
        uncertainty = dict(prediction["uncertainty_by_head"])
        support = dict(prediction["samples_by_head"])
        residual_error = float(np.mean(np.asarray(self.residual_ema, dtype=float)))
        data_gap = bool(
            prediction["mean_uncertainty"] > 0.55
            or int(prediction["min_support"]) < self.min_samples_for_evidence
        )
        return {
            "world_model_schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "world_model_return_lcb_1": float(values.get("return_1", 0.0) - 1.64 * uncertainty.get("return_1", 1.0)),
            "world_model_return_lcb_5": float(values.get("return_5", 0.0) - 1.64 * uncertainty.get("return_5", 1.0)),
            "world_model_volatility": float(values.get("volatility_1", 0.0)),
            "world_model_spread_atr": float(values.get("spread_atr_1", 0.0)),
            "world_model_uncertainty": float(prediction["mean_uncertainty"]),
            "world_model_prediction_error": residual_error,
            "world_model_min_support": int(prediction["min_support"]),
            "world_model_data_gap": data_gap,
            # This is deliberately not market_opportunity. It is a request for
            # offline/shadow evidence only, and must never become an entry logit.
            "world_model_shadow_replay_requested": bool(
                data_gap
                or (support.get("return_1", 0) >= self.min_samples_for_evidence and values.get("return_1", 0.0) > 0.0)
            ),
            "world_model_prediction": prediction,
        }

    def _target_index(self, name: str) -> int:
        return self.target_names.index(name)

    def _update_target(self, name: str, features: np.ndarray, target: float) -> None:
        if name not in self.target_names or not np.isfinite(target):
            return
        index = self._target_index(name)
        raw = float(self.weights[index] @ features + self.bias[index])
        if name in _PROBABILITY_TARGETS:
            prediction = float(_sigmoid(raw))
            error = float(np.clip(target, 0.0, 1.0) - prediction)
        else:
            prediction = float(np.tanh(raw) * 5.0)
            error = float(np.clip(target, -5.0, 5.0) - prediction)
        support = float(self.samples[index])
        lr = float(np.clip(self.learning_rate / np.sqrt(1.0 + 0.02 * support), 0.00005, self.learning_rate))
        gradient = np.clip(error, -2.0, 2.0)
        self.weights[index] = np.clip(self.weights[index] + lr * gradient * features, -3.0, 3.0)
        self.bias[index] = float(np.clip(self.bias[index] + lr * gradient, -3.0, 3.0))
        self.samples[index] = support + 1.0
        abs_error = abs(error)
        self.residual_ema[index] = float(np.clip(0.985 * self.residual_ema[index] + 0.015 * abs_error, 0.01, 5.0))
        delta = float(target - self.target_mean[index])
        self.target_mean[index] += delta / self.samples[index]
        self.target_var[index] = float(np.clip(0.99 * self.target_var[index] + 0.01 * delta * delta, 1e-6, 25.0))

    @staticmethod
    def _maybe(mapping: Mapping[str, object] | None, keys: Iterable[str]) -> float | None:
        source = mapping or {}
        for key in keys:
            try:
                value = float(source.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                return value
        return None

    def observe_transition(
        self,
        observation: object,
        action: object,
        next_observation: object,
        *,
        reward: float = 0.0,
        cost: float = 0.0,
        execution_details: Mapping[str, object] | None = None,
        lifecycle_tags: Iterable[str] = (),
        execution_feedback_eligible: bool = False,
    ) -> dict[str, object]:
        """Attach only realised future state/outcome labels to prior decisions."""
        self.ensure_compatible()
        self.sequence += 1
        features = self.encode(observation, action)
        future = _finite_vector(next_observation, 119)
        updates = 0
        # Attach actual *forward* returns, not a future rolling-return feature
        # that includes bars before the original decision. Each pending record
        # accumulates the causal next-bar return observed after its decision.
        step_return = float(np.clip(future[0], -5.0, 5.0))
        retained: list[_PendingExample] = []
        for example in self.pending:
            example.steps_observed = int(example.steps_observed) + 1
            example.cumulative_return = float(np.clip(example.cumulative_return + step_return, -20.0, 20.0))
            if example.steps_observed in RETURN_HORIZONS and example.steps_observed > 1:
                self._update_target(
                    f"return_{example.steps_observed}",
                    example.features,
                    float(np.clip(example.cumulative_return, -5.0, 5.0)),
                )
                updates += 1
            if example.steps_observed < max(RETURN_HORIZONS):
                retained.append(example)

        # One-step labels belong to the current trace immediately. This avoids
        # a one-decision shift in the old pending-only implementation.
        self._update_target("return_1", features, step_return)
        self._update_target("volatility_1", features, abs(float(future[10])))
        self._update_target("spread_atr_1", features, abs(float(future[15])))
        updates += 3
        self.pending = retained[-self.max_history:]

        # Execution heads are not inferred from market movement. They update
        # only when the bridge/lifecycle has established feedback eligibility.
        if execution_feedback_eligible:
            net_r = float(np.clip(float(reward) - float(cost), -5.0, 5.0))
            self._update_target("execution_cost_r", features, float(np.clip(cost, 0.0, 5.0)))
            self._update_target("net_r", features, net_r)
            details = execution_details or {}
            filled = self._maybe(details, ("filled", "fill_confirmed", "placed"))
            if filled is not None:
                self._update_target("fill_probability", features, 1.0 if filled > 0.0 else 0.0)
            mae = self._maybe(details, ("mae_r", "MAE_R", "mae"))
            mfe = self._maybe(details, ("mfe_r", "MFE_R", "mfe"))
            if mae is not None:
                self._update_target("mae_r", features, float(np.clip(mae, -5.0, 5.0)))
            if mfe is not None:
                self._update_target("mfe_r", features, float(np.clip(mfe, -5.0, 5.0)))
            tags = {str(item).lower() for item in lifecycle_tags}
            if tags & {"tp", "take_profit", "tp_hit"}:
                self._update_target("tp_before_sl", features, 1.0)
            elif tags & {"sl", "stop_loss", "sl_hit"}:
                self._update_target("tp_before_sl", features, 0.0)
            updates += 2

        self.pending.append(_PendingExample(self.sequence, features.copy(), 1, step_return))
        if len(self.pending) > self.max_history:
            self.pending = self.pending[-self.max_history:]
        prediction = self.predict(observation, action)
        self.last_prediction = prediction
        return {
            "world_model_schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "world_model_updates": int(updates),
            "world_model_mean_uncertainty": float(prediction["mean_uncertainty"]),
            "world_model_min_support": int(prediction["min_support"]),
            "world_model_data_gap": bool(prediction["mean_uncertainty"] > 0.55),
        }

    def summary(self) -> dict[str, object]:
        self.ensure_compatible()
        prediction = self.last_prediction or self.predict(np.zeros(119), np.zeros(20))
        return {
            "schema_version": WORLD_MODEL_SCHEMA_VERSION,
            "heads": list(self.target_names),
            "samples_by_head": {name: int(self.samples[index]) for index, name in enumerate(self.target_names)},
            "residual_ema_by_head": {name: float(self.residual_ema[index]) for index, name in enumerate(self.target_names)},
            "pending_examples": int(len(self.pending)),
            "last_mean_uncertainty": float(prediction.get("mean_uncertainty", 1.0)),
        }
