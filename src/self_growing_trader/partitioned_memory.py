"""Separate short-term memory for market regime, position lifecycle and execution.

The stores intentionally use separate write/reset semantics so a position close
cannot erase regime knowledge and a market regime change cannot preserve stale
order/execution details indefinitely.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


WORLD_MEMORY_SCHEMA_VERSION = 1


def _finite(values: object, size: int) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raw = np.zeros(0, dtype=float)
    out = np.zeros(max(1, int(size)), dtype=float)
    if raw.size:
        out[: min(out.size, raw.size)] = raw[:out.size]
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=-1.0)


def _pool(values: object, size: int) -> np.ndarray:
    raw = np.asarray(values, dtype=float).reshape(-1) if values is not None else np.zeros(0)
    raw = np.nan_to_num(raw, nan=0.0, posinf=1.0, neginf=-1.0)
    out = np.zeros(max(1, int(size)), dtype=float)
    if raw.size == 0:
        return out
    bounds = np.linspace(0, raw.size, out.size + 1)
    for index in range(out.size):
        start = int(np.floor(bounds[index]))
        stop = min(raw.size, max(start + 1, int(np.ceil(bounds[index + 1]))))
        if start < stop:
            out[index] = float(np.mean(raw[start:stop]))
    return out


@dataclass
class PartitionedWorkingMemory:
    market_size: int = 16
    position_size: int = 12
    execution_size: int = 8
    market_state: np.ndarray = field(default_factory=lambda: np.zeros(16, dtype=float), repr=False)
    position_state: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=float), repr=False)
    execution_state: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=float), repr=False)
    age: int = 0

    def ensure_compatible(self) -> None:
        self.market_size = max(4, int(getattr(self, "market_size", 16) or 16))
        self.position_size = max(4, int(getattr(self, "position_size", 12) or 12))
        self.execution_size = max(4, int(getattr(self, "execution_size", 8) or 8))
        self.market_state = _finite(getattr(self, "market_state", None), self.market_size)
        self.position_state = _finite(getattr(self, "position_state", None), self.position_size)
        self.execution_state = _finite(getattr(self, "execution_state", None), self.execution_size)
        self.age = max(0, int(getattr(self, "age", 0) or 0))

    def copy(self) -> "PartitionedWorkingMemory":
        self.ensure_compatible()
        return PartitionedWorkingMemory(
            market_size=self.market_size,
            position_size=self.position_size,
            execution_size=self.execution_size,
            market_state=self.market_state.copy(),
            position_state=self.position_state.copy(),
            execution_state=self.execution_state.copy(),
            age=self.age,
        )

    def reset(self) -> None:
        self.ensure_compatible()
        self.market_state.fill(0.0)
        self.position_state.fill(0.0)
        self.execution_state.fill(0.0)
        self.age = 0

    def summary(self) -> dict[str, object]:
        """Return bounded diagnostics without exposing mutable arrays."""
        self.ensure_compatible()
        return {
            "schema_version": WORLD_MEMORY_SCHEMA_VERSION,
            "age": int(self.age),
            "market_norm": float(np.linalg.norm(self.market_state)),
            "position_lifecycle_norm": float(np.linalg.norm(self.position_state)),
            "execution_norm": float(np.linalg.norm(self.execution_state)),
            "market_size": int(self.market_size),
            "position_size": int(self.position_size),
            "execution_size": int(self.execution_size),
        }

    @staticmethod
    def _ema(previous: np.ndarray, current: np.ndarray, *, decay: float) -> np.ndarray:
        return np.clip(decay * previous + (1.0 - decay) * current, -3.0, 3.0)

    def step(self, observation: object) -> dict[str, object]:
        self.ensure_compatible()
        obs = _finite(observation, 119)
        market = np.concatenate([obs[:34], obs[82:108]])
        position = obs[34:74]
        execution = obs[108:119]
        # Market regime changes more slowly; high short-term range makes the
        # store adapt faster rather than carrying a stale trend representation.
        regime_volatility = float(np.clip(abs(obs[10]) / 8.0, 0.0, 1.0))
        market_decay = float(np.clip(0.93 - 0.30 * regime_volatility, 0.55, 0.93))
        self.market_state = self._ema(self.market_state, _pool(market, self.market_size), decay=market_decay)

        position_present = bool(np.any(np.abs(position) > 1e-10))
        if not position_present:
            # Closing all positions immediately clears lifecycle memory. It does
            # not touch the market regime or execution-quality memory.
            self.position_state.fill(0.0)
        else:
            self.position_state = self._ema(self.position_state, _pool(position, self.position_size), decay=0.62)

        execution_active = bool(np.any(np.abs(execution) > 1e-10))
        if not execution_active:
            self.execution_state *= 0.45
        else:
            self.execution_state = self._ema(self.execution_state, _pool(execution, self.execution_size), decay=0.50)
        self.age += 1
        return {
            "partitioned_memory_schema_version": WORLD_MEMORY_SCHEMA_VERSION,
            "market_memory_context": self.market_state.copy(),
            "position_lifecycle_memory_context": self.position_state.copy(),
            "execution_memory_context": self.execution_state.copy(),
            "partitioned_memory_position_active": float(position_present),
            "partitioned_memory_execution_active": float(execution_active),
            "partitioned_memory_market_decay": market_decay,
            "market_memory_norm": float(np.linalg.norm(self.market_state)),
            "position_lifecycle_memory_norm": float(np.linalg.norm(self.position_state)),
            "execution_memory_norm": float(np.linalg.norm(self.execution_state)),
        }
