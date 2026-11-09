"""Canonical Python side of the MT5 execution-output contract.

The EA deliberately receives *decoded* risk percentage on the action wire.
Contract v7 keeps the historic 20-output width and extends the TP/RR codebook
below 2R so the opposite-pivot teacher can represent every positive target.
Marker 6 vectors retain their exact v6 decoder. The matching MQL constants live in
``MQL5/Include/self_growing_execution_contract.mqh``.
Keep this module small and deterministic: it is a bridge contract, not policy
logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EXECUTION_CONTRACT_VERSION = 7
ACTION_COUNT = 12
OUTPUT_COUNT = 20
RISK_OUTPUT_INDEX = 12
# Compatibility-only names for the historic dynamic-geometry rows. Contract
# v6 reuses them for delay and the explicit contract marker.
SL_OUTPUT_INDEX = 13
RR_OUTPUT_INDEX = 14
TP_OUTPUT_INDEX = RR_OUTPUT_INDEX
# Contract v7 reuses the two historic fixed-geometry compatibility rows for
# the factorized Five-style action surface.
ACTION_DELAY_OUTPUT_INDEX = 13
FACTORIZED_ACTION_MARKER_OUTPUT_INDEX = 14
LEGACY_FACTORIZED_ACTION_MARKER_VALUE = 6.0
FACTORIZED_ACTION_MARKER_VALUE = 7.0
ENTRY_OFFSET_OUTPUT_INDEX = 15
GEOMETRY_PACKED_OUTPUT_INDEX = 16
# Backward import name. Row 16 now contains all three geometry choices.
TRAILING_OUTPUT_INDEX = GEOMETRY_PACKED_OUTPUT_INDEX
PARTIAL_CLOSE_OUTPUT_INDEX = 17
MANAGEMENT_MODE_OUTPUT_INDEX = 18
TARGET_SLOT_OUTPUT_INDEX = 19

LEGACY_LEARNED_GEOMETRY_MARKER_THRESHOLD = 5.5
LEARNED_GEOMETRY_MARKER_THRESHOLD = 6.5
LEGACY_SL_ATR_BINS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
SL_ATR_BINS = (0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
LEGACY_TP_RR_BINS = (2.0, 2.5, 3.0, 4.0, 5.0)
TP_RR_BINS = (0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0)
TRAILING_ATR_BINS = (0.0, 0.50, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
DEFAULT_SL_BIN_INDEX = SL_ATR_BINS.index(2.5)
DEFAULT_TP_BIN_INDEX = TP_RR_BINS.index(2.0)
DEFAULT_TRAILING_BIN_INDEX = TRAILING_ATR_BINS.index(0.0)
GEOMETRY_PACKED_BIN_COUNT = len(SL_ATR_BINS) * len(TP_RR_BINS) * len(TRAILING_ATR_BINS)
LEGACY_GEOMETRY_PACKED_BIN_COUNT = (
    len(LEGACY_SL_ATR_BINS) * len(LEGACY_TP_RR_BINS) * len(TRAILING_ATR_BINS)
)

FIXED_GEOMETRY_MODE = "fixed_atr_v1"
LEGACY_LEARNED_GEOMETRY_MODE = "learned_factorized_atr_v1"
LEARNED_GEOMETRY_MODE = "learned_factorized_atr_v2"

# ``legacy_tanh_v1`` was the historical wire behavior: every output row was
# squashed before the decoder saw it.  That accidentally made a parameter
# decoder of the form ``sigmoid(tanh(z))`` while Full-RL trained
# ``sigmoid(z)``.  V2 keeps the action scores bounded but leaves management
# rows as decoder logits, so every execution parameter is transformed exactly
# once by its canonical decoder.
PARAMETER_ACTIVATION_LEGACY_TANH_V1 = "legacy_tanh_v1"
PARAMETER_ACTIVATION_RAW_LOGITS_V2 = "raw_logits_v2"
PARAMETER_ACTIVATION_MODES = frozenset({
    PARAMETER_ACTIVATION_LEGACY_TANH_V1,
    PARAMETER_ACTIVATION_RAW_LOGITS_V2,
})


def normalize_parameter_activation_mode(mode: object) -> str:
    """Return a safe parameter-wire activation mode.

    The legacy default is intentional: protected artifacts must retain their
    historical execution surface until a candidate explicitly opts into V2.
    """

    value = str(mode or PARAMETER_ACTIVATION_LEGACY_TANH_V1).strip().lower()
    return value if value in PARAMETER_ACTIVATION_MODES else PARAMETER_ACTIVATION_LEGACY_TANH_V1


def activate_policy_output(
    values: np.ndarray | list[float],
    *,
    parameter_activation_mode: object = PARAMETER_ACTIVATION_LEGACY_TANH_V1,
    action_count: int = ACTION_COUNT,
) -> np.ndarray:
    """Apply the final policy activation without double-transforming parameters.

    Discrete action scores are bounded for the action wire in both modes.
    ``raw_logits_v2`` deliberately keeps rows after ``action_count`` raw so
    risk/SL/RR/trailing decoders own their single sigmoid.  The returned vector
    is the exact vector that must be stored in the trace and sent to MT5.
    """

    result = np.asarray(values, dtype=float).reshape(-1).copy()
    result[~np.isfinite(result)] = 0.0
    count = min(max(0, int(action_count)), result.size)
    if count:
        result[:count] = np.tanh(result[:count])
    if normalize_parameter_activation_mode(parameter_activation_mode) == PARAMETER_ACTIVATION_LEGACY_TANH_V1:
        result[count:] = np.tanh(result[count:])
    return result


def sigmoid(value: float) -> float:
    clipped = float(np.clip(float(value), -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _output(values: np.ndarray, index: int) -> float:
    return float(values[index]) if 0 <= int(index) < values.size else 0.0


def pack_geometry_choices(sl_index: int, tp_index: int, trailing_index: int) -> float:
    """Pack three categorical choices into one canonical decoder logit."""

    sl = min(max(int(sl_index), 0), len(SL_ATR_BINS) - 1)
    tp = min(max(int(tp_index), 0), len(TP_RR_BINS) - 1)
    trailing = min(max(int(trailing_index), 0), len(TRAILING_ATR_BINS) - 1)
    packed = sl + len(SL_ATR_BINS) * (tp + len(TP_RR_BINS) * trailing)
    probability = (float(packed) + 0.5) / float(GEOMETRY_PACKED_BIN_COUNT)
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(probability / (1.0 - probability)))


def pack_legacy_geometry_choices(sl_index: int, tp_index: int, trailing_index: int) -> float:
    """Pack a marker-6 vector for backward-compatibility verification only."""

    sl = min(max(int(sl_index), 0), len(LEGACY_SL_ATR_BINS) - 1)
    tp = min(max(int(tp_index), 0), len(LEGACY_TP_RR_BINS) - 1)
    trailing = min(max(int(trailing_index), 0), len(TRAILING_ATR_BINS) - 1)
    packed = sl + len(LEGACY_SL_ATR_BINS) * (tp + len(LEGACY_TP_RR_BINS) * trailing)
    probability = (float(packed) + 0.5) / float(LEGACY_GEOMETRY_PACKED_BIN_COUNT)
    probability = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
    return float(np.log(probability / (1.0 - probability)))


def unpack_geometry_choices(value: float) -> tuple[int, int, int]:
    """Inverse of :func:`pack_geometry_choices`, mirrored exactly by MQL."""

    packed = min(
        GEOMETRY_PACKED_BIN_COUNT - 1,
        max(0, int(np.floor(sigmoid(float(value)) * GEOMETRY_PACKED_BIN_COUNT))),
    )
    sl = packed % len(SL_ATR_BINS)
    packed //= len(SL_ATR_BINS)
    tp = packed % len(TP_RR_BINS)
    trailing = packed // len(TP_RR_BINS)
    return int(sl), int(tp), int(trailing)


def unpack_legacy_geometry_choices(value: float) -> tuple[int, int, int]:
    """Decode an immutable marker-6 geometry vector with the v6 codebook."""

    packed = min(
        LEGACY_GEOMETRY_PACKED_BIN_COUNT - 1,
        max(0, int(np.floor(sigmoid(float(value)) * LEGACY_GEOMETRY_PACKED_BIN_COUNT))),
    )
    sl = packed % len(LEGACY_SL_ATR_BINS)
    packed //= len(LEGACY_SL_ATR_BINS)
    tp = packed % len(LEGACY_TP_RR_BINS)
    trailing = packed // len(LEGACY_TP_RR_BINS)
    return int(sl), int(tp), int(trailing)


@dataclass(frozen=True)
class ExecutionGeometry:
    """Canonical fixed order geometry shared by Python and MT5."""

    risk_pct: float
    sl_atr: float
    rr: float
    tp_atr: float
    trailing_stop_atr: float
    mode: str = FIXED_GEOMETRY_MODE


def decode_execution_geometry(
    outputs: np.ndarray | list[float],
    *,
    min_risk_pct: float,
    max_risk_pct: float,
    technical_min_rr: float,
    fixed_sl_atr: float,
    fixed_tp_atr: float,
    fixed_risk_pct: float | None = None,
    parameter_offset: int = ACTION_COUNT,
) -> ExecutionGeometry:
    """Decode risk and versioned entry geometry once for Python and MT5."""

    values = np.asarray(outputs, dtype=float).reshape(-1)
    offset = int(parameter_offset)
    low = float(min(min_risk_pct, max_risk_pct))
    high = float(max(min_risk_pct, max_risk_pct))
    risk_pct = low + sigmoid(_output(values, offset)) * (high - low)
    if fixed_risk_pct is not None and float(fixed_risk_pct) > 0.0:
        risk_pct = float(fixed_risk_pct)
    marker = _output(values, FACTORIZED_ACTION_MARKER_OUTPUT_INDEX)
    learned_geometry = bool(
        values.size > GEOMETRY_PACKED_OUTPUT_INDEX
        and marker >= LEGACY_LEARNED_GEOMETRY_MARKER_THRESHOLD
    )
    if learned_geometry:
        if marker >= LEARNED_GEOMETRY_MARKER_THRESHOLD:
            sl_index, tp_index, trailing_index = unpack_geometry_choices(
                _output(values, GEOMETRY_PACKED_OUTPUT_INDEX)
            )
            rr_bins = TP_RR_BINS
            sl_bins = SL_ATR_BINS
            geometry_mode = LEARNED_GEOMETRY_MODE
        else:
            sl_index, tp_index, trailing_index = unpack_legacy_geometry_choices(
                _output(values, GEOMETRY_PACKED_OUTPUT_INDEX)
            )
            rr_bins = LEGACY_TP_RR_BINS
            sl_bins = LEGACY_SL_ATR_BINS
            geometry_mode = LEGACY_LEARNED_GEOMETRY_MODE
        sl_atr = float(sl_bins[sl_index])
        # V7 deliberately represents every positive opposite-pivot target.
        # The teacher contract is spread-aware positive PnL, not a minimum-RR
        # filter. Marker-6 packets retain their historic technical floor.
        rr = float(rr_bins[tp_index])
        if geometry_mode == LEGACY_LEARNED_GEOMETRY_MODE:
            rr = max(rr, float(technical_min_rr))
        tp_atr = float(sl_atr * rr)
        trailing_stop_atr = float(TRAILING_ATR_BINS[trailing_index])
    else:
        sl_atr = float(fixed_sl_atr)
        tp_atr = float(fixed_tp_atr)
        rr = tp_atr / sl_atr if sl_atr > 0.0 else 0.0
        trailing_stop_atr = 0.0
        geometry_mode = FIXED_GEOMETRY_MODE
    if not np.isfinite(sl_atr) or not np.isfinite(tp_atr) or sl_atr <= 0.0 or tp_atr <= 0.0:
        raise ValueError("SL/TP ATR values must be finite and positive")
    rr = tp_atr / sl_atr
    if geometry_mode != LEARNED_GEOMETRY_MODE and rr + 1e-12 < float(technical_min_rr):
        raise ValueError("fixed TP/SL RR must satisfy technical_min_rr")
    return ExecutionGeometry(
        risk_pct=float(risk_pct),
        sl_atr=float(sl_atr),
        rr=float(rr),
        tp_atr=float(tp_atr),
        trailing_stop_atr=float(trailing_stop_atr),
        mode=geometry_mode,
    )


# ---------------------------------------------------------------------------
# Milestone 2: Regime-Adaptive Volatility Target Contracts (Requirement R2)
# ---------------------------------------------------------------------------

REGIME_ADAPTIVE_GEOMETRY: dict[str, dict[str, float | int]] = {
    # Strong Trend: Wide barriers to capture large moves (Payoff 2.0R)
    "STRONG_TREND": {
        "sl_atr": 5.0,
        "tp_rr": 2.0,
        "sl_idx": 7,          # SL_ATR_BINS[7] = 5.0
        "tp_idx": 5,          # TP_RR_BINS[5] = 2.0
        "trailing_idx": 0,    # TRAILING_ATR_BINS[0] = 0.0
        "sl_index": 7,
        "tp_index": 5,
        "trailing_index": 0,
    },
    # Weak Trend: Balanced barriers for moderate directional momentum (Payoff 1.5R)
    "WEAK_TREND": {
        "sl_atr": 5.0,
        "tp_rr": 1.5,
        "sl_idx": 7,          # SL_ATR_BINS[7] = 5.0
        "tp_idx": 4,          # TP_RR_BINS[4] = 1.5
        "trailing_idx": 0,    # TRAILING_ATR_BINS[0] = 0.0
        "sl_index": 7,
        "tp_index": 4,
        "trailing_index": 0,
    },
    # Consolidation / Sideways: Tight TP to bank profits within 20-50 bars (Payoff 0.75R)
    "RANGING": {
        "sl_atr": 5.0,
        "tp_rr": 0.75,
        "sl_idx": 7,          # SL_ATR_BINS[7] = 5.0
        "tp_idx": 2,          # TP_RR_BINS[2] = 0.75
        "trailing_idx": 0,    # TRAILING_ATR_BINS[0] = 0.0
        "sl_index": 7,
        "tp_index": 2,
        "trailing_index": 0,
    },
    # High Volatility Chop: 1:1 risk-reward to exit before mean-reversion (Payoff 1.0R)
    "VOLATILE": {
        "sl_atr": 5.0,
        "tp_rr": 1.0,
        "sl_idx": 7,          # SL_ATR_BINS[7] = 5.0
        "tp_idx": 3,          # TP_RR_BINS[3] = 1.0
        "trailing_idx": 0,    # TRAILING_ATR_BINS[0] = 0.0
        "sl_index": 7,
        "tp_index": 3,
        "trailing_index": 0,
    },
}


def classify_regime_for_geometry(
    *,
    chop_index: float = 0.5,
    m15_direction: float = 0.0,
    trend_strength: float = 0.0,
    volatility: float = 0.0,
) -> str:
    """Classify market state into one of four geometry regimes.

    Order of priority:
    1. VOLATILE: High volatility chop with weak directional momentum
    2. RANGING: High choppiness or low directional momentum (sideways)
    3. STRONG_TREND: Strong directional momentum
    4. WEAK_TREND: Moderate directional momentum
    """
    abs_dir = abs(float(m15_direction))
    chop = float(chop_index)
    vol = float(volatility)
    strength = float(trend_strength)

    if vol >= 0.85 and abs_dir < 0.20:
        return "VOLATILE"
    if chop >= 0.65 or abs_dir < 0.20 or (chop >= 0.55 and strength < 0.30):
        return "RANGING"
    if abs_dir >= 0.45 or strength >= 0.70:
        return "STRONG_TREND"
    return "WEAK_TREND"


def get_regime_adaptive_geometry(regime: str) -> dict[str, float | int]:
    """Return geometry parameters and bin indices for a named regime."""
    key = str(regime).strip().upper()
    if key not in REGIME_ADAPTIVE_GEOMETRY:
        key = "RANGING"
    return REGIME_ADAPTIVE_GEOMETRY[key]


def resolve_adaptive_geometry_indices(
    *,
    chop_index: float = 0.5,
    m15_direction: float = 0.0,
    trend_strength: float = 0.0,
    volatility: float = 0.0,
) -> tuple[int, int, int, str]:
    """Return (sl_index, tp_index, trailing_index, regime_name) for given market features."""
    regime = classify_regime_for_geometry(
        chop_index=chop_index,
        m15_direction=m15_direction,
        trend_strength=trend_strength,
        volatility=volatility,
    )
    geom = REGIME_ADAPTIVE_GEOMETRY[regime]
    return int(geom["sl_idx"]), int(geom["tp_idx"]), int(geom["trailing_idx"]), regime

