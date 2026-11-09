"""Causal entry representation repair for the context-expert candidate.

This is deliberately an *input representation*, not an action mask, a score
prior, or a second learner.  Direct-RNN/Full-RL remains the sole owner of all
weights.  The module does three narrowly scoped things for an opted-in
candidate:

* appends a small set of bounded trend/location/volatility interactions;
* exposes causal structural-stop anchors to the SL/RR head without turning
  them into a fixed stop-loss rule;
* exposes upper-wick evidence in explicit long/short orientations so output
  rows can learn their own use of it; and
* attenuates the direct encoder drive of empirically unstable inputs while
  keeping their information available through conditional interactions.

The immutable 137-column MT5 wire is never changed.  Local fields are appended
after the optional causal context-history tokens and are denied to management
routes by ``semantic_input_routing``.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np

from .protocol_contract import INPUT_COUNT


ENTRY_FEATURE_ENRICHMENT_LEGACY = "legacy"
ENTRY_FEATURE_ENRICHMENT_QUALITY_V1 = "quality_interactions_v1"
ENTRY_FEATURE_ENRICHMENT_QUALITY_NO_QUARANTINED_V1 = "quality_interactions_no_quarantined_v1"
ENTRY_FEATURE_ENRICHMENT_MODES = frozenset({
    ENTRY_FEATURE_ENRICHMENT_LEGACY,
    ENTRY_FEATURE_ENRICHMENT_QUALITY_V1,
    ENTRY_FEATURE_ENRICHMENT_QUALITY_NO_QUARANTINED_V1,
})
ENTRY_FEATURE_ENRICHMENT_VERSION = 1

# Values are intentionally named in the audit output.  Keep these exact wire
# indexes local: changing any of them is a representation-contract change.
_UPPER_WICK_ATR = 8
_EMA_DIFF_ATR = 11
_VOLATILITY_15 = 3
_VOLATILITY_60 = 5
_RANGE_ATR = 10
_SPREAD_ATR = 15
_M15_EMA_DIFF_ATR = 84
_H1_EMA_DIFF_ATR = 88
_WYCKOFF_SWING_POSITION = 117
_DISTANCE_SWING_HIGH_ATR = 118
_DISTANCE_SWING_LOW_ATR = 119
_LIQUIDITY_SWEEP_REJECTION = 120
_ABSORPTION_SCORE = 126
_RANGE_AGE_NORMALIZED = 128
_SPRING_SCORE = 129
_UPTHRUST_SCORE = 130
_SIGN_OF_STRENGTH = 131
_SIGN_OF_WEAKNESS = 132
_SPRING_TEST_QUALITY = 133
_UPTHRUST_TEST_QUALITY = 134

# The decoder independently enforces [0.5, 5.0] ATR.  These inputs merely
# express a causal candidate anchor in the same bounded scale.  They never
# override output[13] and are intentionally available only to the entry route.
_STRUCTURAL_SL_ATR_MIN = 0.5
_STRUCTURAL_SL_ATR_MAX = 5.0
_STRUCTURAL_SL_BUFFER_MAX = 0.75

# These features showed adverse state/outcome association in the current live
# audit.  They are neither deleted nor rewarded/punished directly: their raw
# encoder drive is reduced only for entry while Full-RL can still learn through
# the explicit conditional features below.
QUARANTINED_ENTRY_INPUTS: Mapping[int, str] = {
    3: "volatility_15_atr",
    5: "volatility_60_atr",
    10: "candle_range_atr",
    17: "rsi_5_normalized",
    78: "hour_sin",
    82: "m5_volatility_ratio",
    84: "m15_ema_diff_atr",
    85: "m15_ema_slope_atr",
    86: "m15_volatility_ratio",
    127: "trading_range_width_atr",
}
H1_DIRECT_INPUTS: Mapping[int, str] = {
    88: "h1_ema_diff_atr",
    89: "h1_ema_slope_atr",
    90: "h1_volatility_ratio",
    91: "h1_gap_atr",
}

ENTRY_INTERACTION_NAMES = (
    "long_trend_location",
    "short_trend_location",
    "long_mtf_wyckoff_alignment",
    "short_mtf_wyckoff_alignment",
    "long_location_volatility_warning",
    "short_location_volatility_warning",
    "upper_wick_directional_buy",
    "upper_wick_directional_sell",
    "long_structural_sl_anchor",
    "short_structural_sl_anchor",
    "long_structural_anchor_quality",
    "short_structural_anchor_quality",
    "sl_execution_buffer",
)


def normalize_entry_feature_enrichment_mode(value: object) -> str:
    mode = str(value or ENTRY_FEATURE_ENRICHMENT_LEGACY).strip().lower()
    if mode not in ENTRY_FEATURE_ENRICHMENT_MODES:
        raise ValueError(
            "entry_feature_enrichment_mode must be legacy, quality_interactions_v1, "
            "or quality_interactions_no_quarantined_v1"
        )
    return mode


def entry_feature_interaction_width(mode: object) -> int:
    return len(ENTRY_INTERACTION_NAMES) if normalize_entry_feature_enrichment_mode(mode) != ENTRY_FEATURE_ENRICHMENT_LEGACY else 0


def _finite_wire(values: object) -> np.ndarray:
    try:
        raw = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raw = np.zeros(0, dtype=float)
    wire = np.zeros(INPUT_COUNT, dtype=float)
    wire[: min(wire.size, raw.size)] = raw[: wire.size]
    return np.nan_to_num(wire, nan=0.0, posinf=0.0, neginf=0.0)


def _value(wire: np.ndarray, index: int) -> float:
    return float(wire[index]) if 0 <= int(index) < wire.size else 0.0


def _signed(value: float) -> float:
    return float(np.tanh(np.clip(float(value), -3.0, 3.0)))


def _near_location(distance_atr: float) -> float:
    """Bounded proximity; it does not encode a directional trade rule."""

    return float(1.0 / (1.0 + max(0.0, float(distance_atr))))


def _unit_interval(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _positive(value: float) -> float:
    return max(0.0, float(value))


def _structural_stop_signal(distance_atr: float, buffer_atr: float) -> float:
    """Encode an ATR stop anchor into ``[-1, 1]`` without fixing an SL.

    The wire is already normalized before this local representation is made,
    so this deliberately preserves order rather than attempting an unsafe
    inverse transform.  The Full-RL geometry target remains the sole teacher
    for the actual decoded SL distance.
    """

    anchored = float(np.clip(
        max(_STRUCTURAL_SL_ATR_MIN, _positive(distance_atr) + max(0.0, buffer_atr)),
        _STRUCTURAL_SL_ATR_MIN,
        _STRUCTURAL_SL_ATR_MAX,
    ))
    ratio = (anchored - _STRUCTURAL_SL_ATR_MIN) / (
        _STRUCTURAL_SL_ATR_MAX - _STRUCTURAL_SL_ATR_MIN
    )
    return float(2.0 * ratio - 1.0)


def _structural_anchor_quality(
    *,
    sweep_rejection: float,
    absorption: float,
    directional_event: float,
    directional_test: float,
    directional_strength: float,
    range_age: float,
    side: int,
) -> float:
    """Return a directional, causal confidence score in ``[-1, 1]``.

    Positive sweep/absorption supports a long structural low; negative values
    support a short structural high.  A stale range deliberately reduces
    confidence but never changes a stop by itself.
    """

    signed_rejection = float(side) * float(sweep_rejection)
    signed_absorption = float(side) * float(absorption)
    quality = (
        0.24 * _unit_interval(signed_rejection)
        + 0.18 * _unit_interval(signed_absorption)
        + 0.24 * _unit_interval(directional_event)
        + 0.18 * _unit_interval(directional_test)
        + 0.16 * _unit_interval(directional_strength)
    )
    quality *= 1.0 - 0.25 * _unit_interval(range_age)
    return float(2.0 * _unit_interval(quality) - 1.0)


def _execution_buffer_atr(
    *, volatility_15: float, volatility_60: float, candle_range: float, spread: float,
) -> float:
    """Causal noise/execution cushion; volatility never chooses direction."""

    noise = max(abs(float(volatility_15)), abs(float(volatility_60)), abs(float(candle_range)))
    return float(np.clip(
        0.08 + 0.28 * min(noise, 1.5) + 0.20 * min(max(0.0, float(spread)), 1.0),
        0.0,
        _STRUCTURAL_SL_BUFFER_MAX,
    ))


def build_entry_quality_interactions(
    observation: object,
    *,
    include_quarantined: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Return bounded, causal interactions from the normalized wire vector.

    No fitted normalizer and no outcome/PnL is read here.  A positive/negative
    interaction is merely an input to learned output rows; the function never
    adds to BUY, SELL or HOLD logits itself.
    """

    wire = _finite_wire(observation)
    trend = _signed(_value(wire, _EMA_DIFF_ATR))
    m15_trend = _signed(_value(wire, _M15_EMA_DIFF_ATR))
    h1_trend = _signed(_value(wire, _H1_EMA_DIFF_ATR))
    trend_up = max(0.0, trend)
    trend_down = max(0.0, -trend)
    mtf = _signed((trend + m15_trend + h1_trend) / 3.0)
    mtf_up = max(0.0, mtf)
    mtf_down = max(0.0, -mtf)
    near_low = _near_location(_value(wire, _DISTANCE_SWING_LOW_ATR))
    near_high = _near_location(_value(wire, _DISTANCE_SWING_HIGH_ATR))
    wyckoff = float(np.clip(_value(wire, _WYCKOFF_SWING_POSITION), 0.0, 1.0))

    # Only relative expansion is used, and only in a location-conditioned
    # warning.  Thus volatility cannot independently create a BUY/SELL score.
    volatility_expansion = abs(_signed(
        _value(wire, _VOLATILITY_15) - _value(wire, _VOLATILITY_60)
    ))
    upper_wick = _signed(_value(wire, _UPPER_WICK_ATR))
    execution_buffer = _execution_buffer_atr(
        volatility_15=_value(wire, _VOLATILITY_15),
        volatility_60=_value(wire, _VOLATILITY_60),
        candle_range=_value(wire, _RANGE_ATR),
        spread=_value(wire, _SPREAD_ATR),
    )
    long_anchor = _structural_stop_signal(
        _value(wire, _DISTANCE_SWING_LOW_ATR), execution_buffer,
    )
    short_anchor = _structural_stop_signal(
        _value(wire, _DISTANCE_SWING_HIGH_ATR), execution_buffer,
    )
    long_quality = _structural_anchor_quality(
        sweep_rejection=_value(wire, _LIQUIDITY_SWEEP_REJECTION),
        absorption=_value(wire, _ABSORPTION_SCORE),
        directional_event=_value(wire, _SPRING_SCORE),
        directional_test=_value(wire, _SPRING_TEST_QUALITY),
        directional_strength=_value(wire, _SIGN_OF_STRENGTH),
        range_age=_value(wire, _RANGE_AGE_NORMALIZED),
        side=1,
    )
    short_quality = _structural_anchor_quality(
        sweep_rejection=_value(wire, _LIQUIDITY_SWEEP_REJECTION),
        absorption=_value(wire, _ABSORPTION_SCORE),
        directional_event=_value(wire, _UPTHRUST_SCORE),
        directional_test=_value(wire, _UPTHRUST_TEST_QUALITY),
        directional_strength=_value(wire, _SIGN_OF_WEAKNESS),
        range_age=_value(wire, _RANGE_AGE_NORMALIZED),
        side=-1,
    )
    values = np.asarray((
        trend_up * near_low,
        trend_down * near_high,
        mtf_up * wyckoff if include_quarantined else 0.0,
        mtf_down * (1.0 - wyckoff) if include_quarantined else 0.0,
        -(1.0 - near_low) * volatility_expansion if include_quarantined else 0.0,
        -(1.0 - near_high) * volatility_expansion if include_quarantined else 0.0,
        -upper_wick,
        upper_wick,
        long_anchor,
        short_anchor,
        long_quality,
        short_quality,
        2.0 * execution_buffer / _STRUCTURAL_SL_BUFFER_MAX - 1.0,
    ), dtype=float)
    values = np.clip(np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0), -1.0, 1.0)
    return values, {
        name: float(value) for name, value in zip(ENTRY_INTERACTION_NAMES, values, strict=True)
    }


def append_entry_quality_interactions(
    observation: object,
    *,
    mode: object = ENTRY_FEATURE_ENRICHMENT_LEGACY,
) -> tuple[np.ndarray, dict[str, object]]:
    """Append local interaction values without modifying the MT5 wire prefix."""

    normalized_mode = normalize_entry_feature_enrichment_mode(mode)
    try:
        base = np.asarray(observation, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        base = np.zeros(0, dtype=float)
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    if normalized_mode == ENTRY_FEATURE_ENRICHMENT_LEGACY:
        return base, {
            "entry_feature_enrichment_mode": normalized_mode,
            "entry_feature_interaction_count": 0,
            "entry_feature_interactions": {},
        }
    interactions, audit = build_entry_quality_interactions(
        base,
        include_quarantined=(normalized_mode == ENTRY_FEATURE_ENRICHMENT_QUALITY_V1),
    )
    return np.concatenate((base, interactions)), {
        "entry_feature_enrichment_mode": normalized_mode,
        "entry_feature_interaction_count": int(interactions.size),
        "entry_feature_interactions": audit,
    }


def apply_entry_feature_quarantine(
    observation: object,
    *,
    mode: object,
    quarantine_scale: float,
    h1_direct_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Attenuate only raw entry drives; interaction fields remain full-scale."""

    normalized_mode = normalize_entry_feature_enrichment_mode(mode)
    values = np.asarray(observation, dtype=float).reshape(-1).copy()
    if normalized_mode == ENTRY_FEATURE_ENRICHMENT_LEGACY:
        return values, {
            "entry_feature_quarantine_active": False,
            "entry_feature_quarantine_scales": {},
        }
    quarantine = float(np.clip(quarantine_scale, 0.0, 1.0))
    h1_scale = float(np.clip(h1_direct_scale, 0.0, 1.0))
    applied: dict[str, float] = {}
    for index, name in QUARANTINED_ENTRY_INPUTS.items():
        if index < values.size:
            values[index] *= quarantine
            applied[name] = quarantine
    for index, name in H1_DIRECT_INPUTS.items():
        if index < values.size:
            values[index] *= h1_scale
            applied[name] = h1_scale
    return values, {
        "entry_feature_quarantine_active": True,
        "entry_feature_quarantine_scales": applied,
    }
