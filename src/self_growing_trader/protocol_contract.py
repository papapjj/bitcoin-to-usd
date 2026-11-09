"""Versioned fail-closed wire contracts shared by Python and MT5.

Bridge v10 keeps the 137-input feature layout and upgrades the action side to a
packet of one or five explicitly identified agents.  Execution feedback is
partitioned by ``trace_id`` so concurrent intents can never share one receipt.
"""
from __future__ import annotations
from typing import Iterable
import numpy as np

FEATURE_SCHEMA_VERSION = 10
# V10 keeps Python as the sole input-normalization owner and adds atomic
# one/five-agent action packets with per-agent execution attribution.
BRIDGE_SCHEMA_VERSION = 10
# V17 keeps V16 causal unrealized/MFE/MAE evidence and appends exact pending
# fill age inside geometry_entry_pipe while preserving the CSV column count.
EXECUTION_LEDGER_SCHEMA_VERSION = 17
# Provenance sidecar keeps the full Python audit beyond the display-safe owner
# fields carried directly by bridge v10.
CONTEXT_EXPERT_PROVENANCE_SCHEMA_VERSION = 1
BASE_MARKET_FEATURE_COUNT = 29
EXTRA_MARKET_FEATURE_COUNT = 29
MARKET_FEATURE_COUNT = BASE_MARKET_FEATURE_COUNT + EXTRA_MARKET_FEATURE_COUNT
MAX_POSITION_SLOTS = 5
UNLIMITED_POSITION_LIMIT = 0
POSITION_FEATURE_COUNT = 8
ACCOUNT_FEATURE_COUNT = 8
EXECUTION_FEATURE_COUNT = 1 + MAX_POSITION_SLOTS * 2
TIMING_FEATURE_COUNT = 4
WYCKOFF_FEATURE_COUNT = 16
TAIL_FEATURE_COUNT = TIMING_FEATURE_COUNT + WYCKOFF_FEATURE_COUNT
POSITION_OFFSET = BASE_MARKET_FEATURE_COUNT
ACCOUNT_OFFSET = POSITION_OFFSET + MAX_POSITION_SLOTS * POSITION_FEATURE_COUNT
EXTRA_MARKET_OFFSET = ACCOUNT_OFFSET + ACCOUNT_FEATURE_COUNT
EXECUTION_OFFSET = EXTRA_MARKET_OFFSET + EXTRA_MARKET_FEATURE_COUNT
TIMING_OFFSET = EXECUTION_OFFSET + EXECUTION_FEATURE_COUNT
WYCKOFF_OFFSET = TIMING_OFFSET + TIMING_FEATURE_COUNT
INPUT_COUNT = (
    MARKET_FEATURE_COUNT
    + MAX_POSITION_SLOTS * POSITION_FEATURE_COUNT
    + ACCOUNT_FEATURE_COUNT
    + EXECUTION_FEATURE_COUNT
    + TAIL_FEATURE_COUNT
)
ACTION_COUNT = 12
OUTPUT_COUNT = 20

def require_supported_position_limit(max_positions: int) -> int:
    value = int(max_positions)
    if value == UNLIMITED_POSITION_LIMIT:
        return value
    if not 1 <= value <= MAX_POSITION_SLOTS:
        raise ValueError(
            f"risk.max_positions must be 0 (unlimited) or within [1,{MAX_POSITION_SLOTS}] "
            f"for feature schema v{FEATURE_SCHEMA_VERSION}; got {value}"
        )
    return value


def position_limit_enabled(max_positions: int) -> bool:
    """Return whether a software position/order quota is configured.

    Zero deliberately means that only broker validity and available margin may
    reject another order. ``MAX_POSITION_SLOTS`` remains an observation-window
    width and is not an execution limit.
    """
    return int(max_positions) > UNLIMITED_POSITION_LIMIT


def position_count_reference(max_positions: int) -> int:
    """Stable denominator for the fixed 137-input account observation."""
    return int(max_positions) if position_limit_enabled(max_positions) else MAX_POSITION_SLOTS

def finite_vector(values: Iterable[float] | np.ndarray, *, size: int | None = None, name: str = "vector") -> np.ndarray:
    result = np.asarray(values, dtype=float).reshape(-1)
    if size is not None and result.size != int(size):
        raise ValueError(f"{name} has {result.size} values; expected {size}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result

def sanitize_finite(values: Iterable[float] | np.ndarray, default: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return np.where(np.isfinite(array), array, float(default))
