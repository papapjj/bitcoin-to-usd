"""Canonical decision context for the self-growing trading policy.

A ContextFrame is the single immutable description of one decision tick.  It
keeps market, portfolio, position, execution, goal and learning-drive signals
separate until they are deliberately projected into a graph region.  This
prevents raw observations, latent M1 activations and bridge-only constraints
from being compared as if they were the same vector space.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping

import numpy as np

from .protocol_contract import (
    ACCOUNT_OFFSET,
    BASE_MARKET_FEATURE_COUNT,
    EXECUTION_FEATURE_COUNT,
    EXECUTION_OFFSET,
    EXTRA_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    POSITION_FEATURE_COUNT,
    POSITION_OFFSET,
    TIMING_FEATURE_COUNT,
    TIMING_OFFSET,
    WYCKOFF_FEATURE_COUNT,
    WYCKOFF_OFFSET,
)

ACTION_COUNT = 12
CANONICAL_MEMORY_SCHEMA_VERSION = 10
MARKET_MEMORY_KEY_SIZE = 32
ACCOUNT_MEMORY_KEY_SIZE = 10
POSITION_MEMORY_KEY_SIZE = 6
MEMORY_KEY_SIZE = MARKET_MEMORY_KEY_SIZE + ACCOUNT_MEMORY_KEY_SIZE + POSITION_MEMORY_KEY_SIZE

# Exact v10 bridge feature contract. Keep these ranges centralised: component
# masks and the canonical encoder must not approximate the schema by percentages.
MARKET_BASE_SLICE = slice(0, BASE_MARKET_FEATURE_COUNT)
POSITION_SLOTS_SLICE = slice(POSITION_OFFSET, ACCOUNT_OFFSET)
ACCOUNT_SLICE = slice(ACCOUNT_OFFSET, EXTRA_MARKET_OFFSET)
MARKET_EXTRA_SLICE = slice(EXTRA_MARKET_OFFSET, EXECUTION_OFFSET)
EXECUTION_SLICE = slice(EXECUTION_OFFSET, TIMING_OFFSET)
TIMING_SLICE = slice(TIMING_OFFSET, WYCKOFF_OFFSET)
WYCKOFF_SLICE = slice(WYCKOFF_OFFSET, INPUT_COUNT)

ACCOUNT_CONTEXT_FIELDS = (
    "equity_to_balance",
    "floating_pnl_r",
    "total_pnl",
    "cumulative_pnl_pct",
    "current_equity_drawdown",
    "balance_drawdown",
    "open_risk_ratio",
    "equity_return_1_bar",
    "equity_return_5_bar",
    "equity_volatility_20",
    "losing_streak_norm",
    "time_under_water_norm",
    "position_count_norm",
    "day_trade_pressure",
)


def _finite_vector(values: object, size: int | None = None) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        vector = np.zeros(0, dtype=float)
    vector = np.nan_to_num(vector, nan=0.0, posinf=2.0, neginf=-2.0)
    if size is None:
        return vector.astype(float, copy=True)
    result = np.zeros(max(0, int(size)), dtype=float)
    result[: min(result.size, vector.size)] = vector[: result.size]
    return result


def _frozen_vector(values: object, size: int | None = None) -> np.ndarray:
    vector = _finite_vector(values, size)
    vector.setflags(write=False)
    return vector


def pool_vector(values: object, size: int) -> np.ndarray:
    """Mean-pool every source dimension into an exact target width.

    Padding/truncation would silently drop the v4 extra-market and execution
    features.  Pooling is deterministic, length-preserving in aggregate and
    safe for a checkpoint-compatible fixed-width memory key.
    """
    source = _finite_vector(values)
    target = max(0, int(size))
    result = np.zeros(target, dtype=float)
    if target <= 0 or source.size == 0:
        return result
    edges = np.linspace(0, source.size, target + 1)
    for index in range(target):
        start = int(np.floor(edges[index]))
        stop = int(np.ceil(edges[index + 1]))
        stop = max(stop, start + 1)
        stop = min(stop, source.size)
        if start < source.size:
            result[index] = float(np.mean(source[start:stop]))
    return np.clip(result, -2.0, 2.0)


def account_vector_from_mapping(
    account_context: Mapping[str, object] | None,
    fallback: object | None = None,
) -> np.ndarray:
    fallback_vector = _finite_vector(fallback, ACCOUNT_MEMORY_KEY_SIZE)
    if not isinstance(account_context, Mapping) or not account_context:
        return np.clip(fallback_vector, -2.0, 2.0)
    values = np.zeros(len(ACCOUNT_CONTEXT_FIELDS), dtype=float)
    for index, key in enumerate(ACCOUNT_CONTEXT_FIELDS):
        try:
            values[index] = float(account_context.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            values[index] = 0.0
    values = np.clip(values, -2.0, 2.0)
    # The bridge mapping is richer than the eight raw account columns.  Pool it
    # to eight and blend with raw fields only when both are available.
    mapped = pool_vector(values, ACCOUNT_MEMORY_KEY_SIZE)
    if np.any(np.abs(mapped) > 1e-12):
        return np.clip(0.80 * mapped + 0.20 * fallback_vector, -2.0, 2.0)
    return np.clip(fallback_vector, -2.0, 2.0)


def learning_drive_vector(starvation: Mapping[str, object] | None) -> np.ndarray:
    source = starvation if isinstance(starvation, Mapping) else {}
    def value(name: str) -> float:
        try:
            return float(source.get(name, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0
    return np.clip(np.array([
        value("no_trade_streak") / 12.0,
        value("unfilled_streak") / 8.0,
        value("cadence_deficit"),
        value("runtime_starvation") / 3.0,
        value("outcome_starvation") / 3.0,
    ], dtype=float), 0.0, 1.0)


def normalize_action_mask(mask: object | None, action_count: int = ACTION_COUNT) -> np.ndarray:
    count = max(1, int(action_count))
    result = np.ones(count, dtype=bool)
    if mask is None:
        return result
    if isinstance(mask, Mapping):
        result[:] = False
        for key, value in mask.items():
            try:
                index = int(key)
            except (TypeError, ValueError):
                continue
            if 0 <= index < count:
                result[index] = bool(value)
    else:
        raw = np.asarray(mask, dtype=bool).reshape(-1)
        if raw.size:
            result[:] = False
            result[: min(count, raw.size)] = raw[:count]
    if not np.any(result):
        result[0] = True  # HOLD remains the universal fail-closed action.
    result.setflags(write=False)
    return result


def blockwise_cosine(left: object, right: object) -> float:
    """Similarity that preserves market/account/position weighting by block."""
    a = _finite_vector(left, MEMORY_KEY_SIZE)
    b = _finite_vector(right, MEMORY_KEY_SIZE)
    slices = (
        (slice(0, MARKET_MEMORY_KEY_SIZE), 0.50),
        (slice(MARKET_MEMORY_KEY_SIZE, MARKET_MEMORY_KEY_SIZE + ACCOUNT_MEMORY_KEY_SIZE), 0.30),
        (slice(MARKET_MEMORY_KEY_SIZE + ACCOUNT_MEMORY_KEY_SIZE, MEMORY_KEY_SIZE), 0.20),
    )
    total = 0.0
    denominator = 0.0
    for part, weight in slices:
        av = a[part]
        bv = b[part]
        norm = float(np.linalg.norm(av) * np.linalg.norm(bv))
        if norm <= 1e-12:
            continue
        total += float(weight) * float(np.dot(av, bv) / norm)
        denominator += float(weight)
    return float(np.clip(total / denominator, -1.0, 1.0)) if denominator > 1e-12 else 0.0


@dataclass(frozen=True)
class ContextFrame:
    """Immutable canonical state for exactly one policy decision."""

    frame_id: str
    sequence_id: int
    timestamp: float
    raw_observation: np.ndarray
    market_vector: np.ndarray
    position_vector: np.ndarray
    account_vector: np.ndarray
    execution_vector: np.ndarray
    execution_external_vector: np.ndarray
    goal_vector: np.ndarray
    learning_drive: np.ndarray
    allowed_action_mask: np.ndarray
    canonical_memory_key: np.ndarray
    account_scalars: dict[str, float]
    execution_scalars: dict[str, float]
    temporal_current: np.ndarray | None = None
    temporal_delta: np.ndarray | None = None
    temporal_short: np.ndarray | None = None
    temporal_regime: np.ndarray | None = None
    schema_version: int = CANONICAL_MEMORY_SCHEMA_VERSION

    @classmethod
    def from_observation(
        cls,
        observation: object,
        *,
        frame_id: str,
        sequence_id: int,
        timestamp: float,
        account_context: Mapping[str, object] | None = None,
        execution_context: Mapping[str, object] | None = None,
        starvation: Mapping[str, object] | None = None,
        goal: object | None = None,
        allowed_action_mask: object | None = None,
    ) -> "ContextFrame":
        raw = _finite_vector(observation, INPUT_COUNT)
        timing = raw[TIMING_SLICE].copy()
        wyckoff = raw[WYCKOFF_SLICE].copy()
        core_market = np.concatenate([raw[MARKET_BASE_SLICE], raw[MARKET_EXTRA_SLICE], timing])
        market = np.concatenate([core_market, wyckoff])
        position = raw[POSITION_SLOTS_SLICE].copy()
        account = account_vector_from_mapping(account_context, raw[ACCOUNT_SLICE])
        execution = raw[EXECUTION_SLICE].copy()
        ext = execution_context if isinstance(execution_context, Mapping) else {}
        def scalar(name: str, scale: float = 1.0) -> float:
            try:
                return float(np.clip(float(ext.get(name, 0.0) or 0.0) / max(scale, 1e-12), -2.0, 2.0))
            except (TypeError, ValueError):
                return 0.0
        execution_external = np.array([
            scalar("spread_points", 100.0),
            scalar("position_capacity_remaining", 1.0),
            scalar("pending_orders", 1.0),
            scalar("allowed_action_fraction", 1.0),
            scalar("day_trade_pressure", 1.0),
        ], dtype=float)
        # Schema v10 reserves one quarter of the market key for Wyckoff. This
        # prevents its event sequence from disappearing inside a pooled 78-wide
        # market vector while keeping the established memory-key width stable.
        market_key = np.concatenate([
            pool_vector(core_market, 16),
            pool_vector(wyckoff, 8),
            pool_vector(np.concatenate([execution, execution_external]), 8),
        ])
        position_key = pool_vector(position, POSITION_MEMORY_KEY_SIZE)
        key = np.concatenate([market_key, account, position_key])
        goal_vector = _finite_vector(goal)
        drive = learning_drive_vector(starvation)
        mask = normalize_action_mask(allowed_action_mask, ACTION_COUNT)
        account_scalars = {
            str(key): float(value)
            for key, value in (account_context.items() if isinstance(account_context, Mapping) else [])
            if _is_number(value)
        }
        execution_scalars = {
            str(key): float(value)
            for key, value in ext.items()
            if _is_number(value)
        }
        return cls(
            frame_id=str(frame_id),
            sequence_id=int(sequence_id),
            timestamp=float(timestamp),
            raw_observation=_frozen_vector(raw, INPUT_COUNT),
            market_vector=_frozen_vector(market),
            position_vector=_frozen_vector(position),
            account_vector=_frozen_vector(account, ACCOUNT_MEMORY_KEY_SIZE),
            execution_vector=_frozen_vector(execution),
            execution_external_vector=_frozen_vector(execution_external),
            goal_vector=_frozen_vector(goal_vector),
            learning_drive=_frozen_vector(drive),
            allowed_action_mask=mask,
            canonical_memory_key=_frozen_vector(key, MEMORY_KEY_SIZE),
            account_scalars=account_scalars,
            execution_scalars=execution_scalars,
        )

    def with_temporal(
        self,
        current: object,
        delta: object,
        short: object,
        regime: object,
    ) -> "ContextFrame":
        return replace(
            self,
            temporal_current=_frozen_vector(current, MEMORY_KEY_SIZE),
            temporal_delta=_frozen_vector(delta, MEMORY_KEY_SIZE),
            temporal_short=_frozen_vector(short, MEMORY_KEY_SIZE),
            temporal_regime=_frozen_vector(regime, MEMORY_KEY_SIZE),
        )

    @property
    def temporal_ready(self) -> bool:
        return all(value is not None for value in (
            self.temporal_current,
            self.temporal_delta,
            self.temporal_short,
            self.temporal_regime,
        ))

    def as_m1_context(self, memory_query: Mapping[str, object] | None = None) -> dict[str, object]:
        result: dict[str, object] = dict(memory_query or {})
        result.update({
            "context_frame_id": self.frame_id,
            "context_schema_version": int(self.schema_version),
            "market_context": self.market_vector.copy(),
            "portfolio_context": self.account_vector.copy(),
            "position_context": self.position_vector.copy(),
            "execution_context": np.concatenate([self.execution_vector, self.execution_external_vector]),
            "goal_vector": self.goal_vector.copy(),
            "learning_drive": self.learning_drive.copy(),
            "allowed_action_mask": self.allowed_action_mask.copy(),
            "canonical_memory_key": self.canonical_memory_key.copy(),
            "temporal_current": self.temporal_current.copy() if self.temporal_current is not None else self.canonical_memory_key.copy(),
            "temporal_delta": self.temporal_delta.copy() if self.temporal_delta is not None else np.zeros(MEMORY_KEY_SIZE, dtype=float),
            "temporal_short": self.temporal_short.copy() if self.temporal_short is not None else self.canonical_memory_key.copy(),
            "temporal_regime": self.temporal_regime.copy() if self.temporal_regime is not None else self.canonical_memory_key.copy(),
        })
        return result

    def audit_summary(self) -> dict[str, object]:
        return {
            "frame_id": self.frame_id,
            "sequence_id": int(self.sequence_id),
            "schema_version": int(self.schema_version),
            "allowed_actions": [int(index) for index, allowed in enumerate(self.allowed_action_mask) if bool(allowed)],
            "market_norm": float(np.linalg.norm(self.market_vector)),
            "account_norm": float(np.linalg.norm(self.account_vector)),
            "position_norm": float(np.linalg.norm(self.position_vector)),
            "execution_norm": float(np.linalg.norm(self.execution_vector)),
            "goal_norm": float(np.linalg.norm(self.goal_vector)),
            "learning_drive_norm": float(np.linalg.norm(self.learning_drive)),
        }


def _is_number(value: object) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False
