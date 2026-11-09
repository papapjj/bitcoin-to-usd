"""Audit-first expectancy by entry condition.

This ledger answers a narrow causal question that the aggregate action ledger
cannot: was a result produced by a MARKET entry chasing an extended/high-vol
move, or by a STOP breakout with aligned trend/location?  It stores realised
final R only after lifecycle settlement.  It never changes logits, masks,
selector ranking, risk or optimizer ownership; Full-RL remains the only writer
of Direct-RNN parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from .contextual_expectancy import ExpectancyEstimate, robust_expectancy
from .protocol_contract import INPUT_COUNT


CONDITIONED_ENTRY_EXPECTANCY_SCHEMA_VERSION = 1
ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})

_EMA_DIFF_ATR = 11
_VOLATILITY_15 = 3
_VOLATILITY_60 = 5
_DISTANCE_SWING_HIGH_ATR = 118
_DISTANCE_SWING_LOW_ATR = 119


def _wire(values: object) -> np.ndarray:
    try:
        source = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        source = np.zeros(0, dtype=float)
    result = np.zeros(INPUT_COUNT, dtype=float)
    result[: min(result.size, source.size)] = source[: result.size]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _bucket_distance(value: float) -> str:
    distance = max(0.0, float(value))
    if distance <= 0.35:
        return "near"
    if distance <= 0.85:
        return "mid"
    return "far"


def _action_name(action: int) -> str:
    return {
        1: "MARKET_BUY", 2: "MARKET_SELL", 3: "BUY_LIMIT", 4: "SELL_LIMIT",
        5: "BUY_STOP", 6: "SELL_STOP",
    }.get(int(action), f"ACTION_{int(action)}")


def entry_condition(
    observation: object,
    *,
    action_index: int,
    context: object = "",
) -> dict[str, str | int]:
    """Create a coarse causal bucket from the exact input available at entry."""

    action = int(action_index)
    if action not in ENTRY_ACTIONS:
        return {}
    wire = _wire(observation)
    trend_signal = float(np.tanh(wire[_EMA_DIFF_ATR]))
    trend = "up" if trend_signal >= 0.20 else "down" if trend_signal <= -0.20 else "neutral"
    is_long = action in {1, 3, 5}
    location = _bucket_distance(
        wire[_DISTANCE_SWING_LOW_ATR] if is_long else wire[_DISTANCE_SWING_HIGH_ATR]
    )
    expansion = float(np.tanh(wire[_VOLATILITY_15] - wire[_VOLATILITY_60]))
    level = 0.5 * (float(wire[_VOLATILITY_15]) + float(wire[_VOLATILITY_60]))
    if expansion >= 0.20:
        volatility = "expanding_high" if level >= 0.85 else "expanding"
    elif expansion <= -0.20:
        volatility = "contracting"
    else:
        volatility = "high_stable" if level >= 0.85 else "stable"
    return {
        "context": str(context or "UNKNOWN").strip().upper() or "UNKNOWN",
        "action": _action_name(action),
        "action_index": action,
        "trend": trend,
        "location": location,
        "volatility": volatility,
    }


def condition_key(condition: Mapping[str, object]) -> str:
    return "|".join(
        str(condition.get(key, "UNKNOWN") or "UNKNOWN")
        for key in ("context", "action", "trend", "location", "volatility")
    )


@dataclass
class ConditionedEntryExpectancyRecord:
    condition: dict[str, str | int]
    values: list[float] = field(default_factory=list)
    trace_ids: list[str] = field(default_factory=list)

    def observe(self, realized_r: float, *, trace_id: str, max_samples: int) -> None:
        self.values.append(float(np.clip(float(realized_r), -12.0, 12.0)))
        self.trace_ids.append(str(trace_id or ""))
        cap = max(8, int(max_samples))
        if len(self.values) > cap:
            self.values = self.values[-cap:]
            self.trace_ids = self.trace_ids[-cap:]

    def estimate(self, *, min_samples: int, prior_strength: float) -> ExpectancyEstimate:
        return robust_expectancy(
            self.values,
            min_samples=max(1, int(min_samples)),
            prior_strength=max(0.0, float(prior_strength)),
            policy_origin_ratio=1.0,
        )


@dataclass
class ConditionedEntryExpectancyLedger:
    records: dict[str, ConditionedEntryExpectancyRecord] = field(default_factory=dict)
    schema_version: int = CONDITIONED_ENTRY_EXPECTANCY_SCHEMA_VERSION
    min_samples: int = 12
    prior_strength: float = 12.0
    max_samples_per_record: int = 256

    def ensure_compatible(self) -> None:
        if int(getattr(self, "schema_version", 0) or 0) != CONDITIONED_ENTRY_EXPECTANCY_SCHEMA_VERSION:
            self.records = {}
            self.schema_version = CONDITIONED_ENTRY_EXPECTANCY_SCHEMA_VERSION
        self.min_samples = max(1, int(getattr(self, "min_samples", 12) or 12))
        self.prior_strength = max(0.0, float(getattr(self, "prior_strength", 12.0) or 12.0))
        self.max_samples_per_record = max(8, int(getattr(self, "max_samples_per_record", 256) or 256))
        if not isinstance(getattr(self, "records", None), dict):
            self.records = {}
        self.records = {
            str(key): value
            for key, value in self.records.items()
            if isinstance(value, ConditionedEntryExpectancyRecord)
        }

    def observe(
        self,
        condition: Mapping[str, object],
        realized_r: float,
        *,
        trace_id: str = "",
    ) -> tuple[str, ExpectancyEstimate]:
        self.ensure_compatible()
        normalized = {
            "context": str(condition.get("context", "UNKNOWN") or "UNKNOWN"),
            "action": str(condition.get("action", "UNKNOWN") or "UNKNOWN"),
            "action_index": int(condition.get("action_index", -1) or -1),
            "trend": str(condition.get("trend", "neutral") or "neutral"),
            "location": str(condition.get("location", "unknown") or "unknown"),
            "volatility": str(condition.get("volatility", "unknown") or "unknown"),
        }
        key = condition_key(normalized)
        record = self.records.get(key)
        if record is None:
            record = ConditionedEntryExpectancyRecord(condition=normalized)
            self.records[key] = record
        record.observe(realized_r, trace_id=trace_id, max_samples=self.max_samples_per_record)
        return key, record.estimate(
            min_samples=self.min_samples,
            prior_strength=self.prior_strength,
        )

    def report(self) -> dict[str, object]:
        self.ensure_compatible()
        rows: list[dict[str, object]] = []
        for key, record in self.records.items():
            estimate = record.estimate(
                min_samples=self.min_samples,
                prior_strength=self.prior_strength,
            )
            rows.append({
                "key": key,
                "condition": dict(record.condition),
                **estimate.as_dict(),
            })
        rows.sort(key=lambda item: (-int(item["sample_count"]), str(item["key"])))
        # ``record_count`` alone is easy to misread as the number of realised
        # samples. In a short MT5 run the causal five-tuple intentionally
        # splits support across several regimes, so many settled samples can
        # exist while no bucket has passed ``min_samples`` yet. Publish both
        # quantities explicitly. This remains audit-only and cannot become a
        # decision-time gate, mask, selector input, or second optimizer writer.
        sample_count = int(sum(int(item["sample_count"]) for item in rows))
        eligible_rows = [item for item in rows if bool(item.get("eligible", False))]
        eligible_sample_count = int(sum(int(item["sample_count"]) for item in eligible_rows))
        max_record_sample_count = max((int(item["sample_count"]) for item in rows), default=0)
        return {
            "schema_version": CONDITIONED_ENTRY_EXPECTANCY_SCHEMA_VERSION,
            "min_samples": int(self.min_samples),
            "prior_strength": float(self.prior_strength),
            "record_count": len(rows),
            "sample_count": sample_count,
            "eligible_record_count": int(len(eligible_rows)),
            "eligible_sample_count": eligible_sample_count,
            "max_record_sample_count": int(max_record_sample_count),
            "samples_to_first_eligible": int(max(0, int(self.min_samples) - max_record_sample_count)),
            "records": rows,
            "telemetry_only": True,
        }
