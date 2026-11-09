"""Deterministic, leakage-free normalization for the MT5 input contract.

The MT5 bridge and the offline environment must apply exactly the same
transform. The transforms are deliberately parameter-free and do not estimate
statistics from train/OOS data, so they cannot leak future regime information
into a candidate. ``soft_clip_v1`` preserves the useful ``[-1, 1]`` region and
smoothly compresses only the extreme tail; this is the less disruptive
alternative to globally applying tanh to a raw v13 seed.

``raw`` remains available for legacy v13 artifacts.  Normalized experiments
must use a separate candidate/config and may not overwrite the protected seed.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .protocol_contract import INPUT_COUNT, finite_vector

INPUT_NORMALIZATION_VERSION = 3
INPUT_NORMALIZATION_RAW = "raw"
INPUT_NORMALIZATION_TANH_V1 = "tanh_v1"
INPUT_NORMALIZATION_SOFT_CLIP_V1 = "soft_clip_v1"
INPUT_NORMALIZATION_GROUPWISE_V2 = "groupwise_v2"
INPUT_NORMALIZATION_GROUPWISE_MTF_V3 = "groupwise_mtf_v3"
INPUT_NORMALIZATION_STRUCTURE_AWARE_V4 = "structure_aware_v4"
INPUT_NORMALIZATION_MODES = frozenset({
    INPUT_NORMALIZATION_RAW,
    INPUT_NORMALIZATION_TANH_V1,
    INPUT_NORMALIZATION_SOFT_CLIP_V1,
    INPUT_NORMALIZATION_GROUPWISE_V2,
    INPUT_NORMALIZATION_GROUPWISE_MTF_V3,
    INPUT_NORMALIZATION_STRUCTURE_AWARE_V4,
})
INPUT_NORMALIZATION_CLIP = 20.0
INPUT_NORMALIZATION_SOFT_CLIP_KNEE = 1.0
INPUT_NORMALIZATION_SOFT_CLIP_TAIL_SCALE = 2.0

# The bridge layout is fixed at 137 values.  This transform intentionally
# uses only the semantic *group* and known bounded fields; it never estimates
# a mean/std from a train or OOS window, therefore it is causal and has no
# future-regime leakage.  Position/account/execution values are the only
# groups observed to have heavy absolute tails in the live audit.
_POSITION_SLICE = slice(29, 69)
_ACCOUNT_SLICE = slice(69, 77)
_EXTRA_MARKET_SLICE = slice(77, 106)
_EXECUTION_SLICE = slice(106, 117)
_TIMING_WYCKOFF_SLICE = slice(117, 137)
# M5/M15/H1 EMA spread, EMA slope, relative volatility and gap.  V3 applies a
# separate smooth bounded map *after* each feature has been expressed in its
# own timeframe ATR.  This prevents a residual long-H1 trend tail from being
# pinned at the generic soft-clip endpoint.
_MTF_NATIVE_SLICE = slice(80, 92)
# Exact 137-contract indices. The EA emits ``chop_index`` as a unit interval,
# not a percentage. V2/V3 auto-scales values: raw > 1.0 is treated as legacy
# percentage (0..100) while raw <= 1.0 is treated as canonical unit-interval (0..1).
# This prevents collapsing live values near -1 while preserving backward compatibility.
_CHOP_INDEX = 93
_BARS_SINCE_INDICES = (135, 136)
# Positive distances already expressed in ATR units. Their old soft-clip map
# became almost flat above 3--4 ATR, even though 6 ATR and 14 ATR represent
# materially different entry locations. A signed/asinh map is not a fitted
# scaler: it is deterministic, monotonic and preserves local resolution.
_POSITIVE_ATR_DISTANCE_INDICES = (100, 101, 102, 103, 118, 119)
_MTF_VOLATILITY_RATIO_INDICES = (82, 86, 90)
_MTF_SIGNED_ATR_INDICES = (80, 81, 83, 84, 85, 87, 88, 89, 91)
_ATR_DISTANCE_REFERENCE = 2.0
_ATR_DISTANCE_MAX = 20.0
_MTF_SIGNED_ATR_REFERENCE = 1.5
_MTF_VOLATILITY_LOG_SCALE = 0.35


def _soft_clip(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -INPUT_NORMALIZATION_CLIP, INPUT_NORMALIZATION_CLIP)
    magnitude = np.abs(clipped)
    tail = np.maximum(magnitude - INPUT_NORMALIZATION_SOFT_CLIP_KNEE, 0.0)
    compressed = INPUT_NORMALIZATION_SOFT_CLIP_KNEE + (
        1.0 - np.exp(-tail / INPUT_NORMALIZATION_SOFT_CLIP_TAIL_SCALE)
    )
    return np.where(magnitude <= INPUT_NORMALIZATION_SOFT_CLIP_KNEE, clipped, np.sign(clipped) * compressed)


def _groupwise_v2(vector: np.ndarray) -> np.ndarray:
    """Normalize by feature family while preserving sign and local ordering.

    Market/TIMING/Wyckoff values are already mostly dimensionless ATR scores,
    so they retain soft-clip semantics.  Stateful account/order fields use a
    signed log compressor.  This avoids a high lot/count/value field drowning
    market features, without inventing a running normalizer whose state could
    drift between MT5 train and frozen replay.
    """

    result = _soft_clip(vector)
    for group in (_POSITION_SLICE, _ACCOUNT_SLICE, _EXECUTION_SLICE):
        raw = np.clip(vector[group], -INPUT_NORMALIZATION_CLIP, INPUT_NORMALIZATION_CLIP)
        result[group] = np.sign(raw) * np.log1p(np.abs(raw)) / np.log1p(INPUT_NORMALIZATION_CLIP)
    if vector.size > _CHOP_INDEX:
        raw_chop = float(vector[_CHOP_INDEX])
        normalized_chop = raw_chop / 100.0 if raw_chop > 1.0 else raw_chop
        result[_CHOP_INDEX] = float(np.clip(2.0 * np.clip(normalized_chop, 0.0, 1.0) - 1.0, -1.0, 1.0))
    for index in _BARS_SINCE_INDICES:
        if vector.size > index:
            result[index] = float(np.clip(np.log1p(max(0.0, vector[index])) / np.log1p(60.0), 0.0, 1.0))
    return result


def _groupwise_mtf_v3(vector: np.ndarray) -> np.ndarray:
    """V2 plus a time-frame-local smooth map for native MTF features.

    It deliberately has no running mean/variance: the ATR denominator is the
    causal local scale and ``1.2*tanh`` preserves order without an endpoint
    plateau.  Keeping the output near ``[-1.2, 1.2]`` leaves the ordinary
    market signal comparable to the rest of groupwise_v2 while retaining
    useful variation in strong H1 trends.
    """

    result = _groupwise_v2(vector)
    upper = min(vector.size, _MTF_NATIVE_SLICE.stop)
    if upper > _MTF_NATIVE_SLICE.start:
        raw = np.clip(vector[_MTF_NATIVE_SLICE.start:upper], -INPUT_NORMALIZATION_CLIP, INPUT_NORMALIZATION_CLIP)
        result[_MTF_NATIVE_SLICE.start:upper] = 1.2 * np.tanh(raw)
    return result


def _asinh(values: np.ndarray | float) -> np.ndarray | float:
    """Portable inverse-hyperbolic-sine used by Python and MQL5 parity code."""

    source = np.asarray(values, dtype=float)
    result = np.log(source + np.sqrt(source * source + 1.0))
    if source.ndim == 0:
        return float(result)
    return result


def _structure_aware_v4(vector: np.ndarray) -> np.ndarray:
    """V4 candidate normalizer for saturated structural/MTF observations.

    It keeps the causal group boundaries from V2, but uses semantic reference
    points where they exist: ``0`` for signed ATR values, ``1`` for volatility
    ratios and ``[0, 1]`` for chop. No rolling fit is stored, so frozen replay
    and MT5 use the exact same transform without future-regime leakage.
    """

    result = _groupwise_v2(vector)
    # Correct the unit-interval EA contract. This intentionally differs from
    # V2/V3, where the historical 0..100 assumption is retained for old runs.
    if vector.size > _CHOP_INDEX:
        result[_CHOP_INDEX] = float(np.clip(2.0 * np.clip(vector[_CHOP_INDEX], 0.0, 1.0) - 1.0, -1.0, 1.0))
    distance_denominator = float(_asinh(_ATR_DISTANCE_MAX / _ATR_DISTANCE_REFERENCE))
    for index in _POSITIVE_ATR_DISTANCE_INDICES:
        if vector.size > index:
            raw = float(np.clip(vector[index], 0.0, _ATR_DISTANCE_MAX))
            result[index] = float(np.clip(_asinh(raw / _ATR_DISTANCE_REFERENCE) / distance_denominator, 0.0, 1.0))
    signed_denominator = float(_asinh(INPUT_NORMALIZATION_CLIP / _MTF_SIGNED_ATR_REFERENCE))
    for index in _MTF_SIGNED_ATR_INDICES:
        if vector.size > index:
            raw = float(np.clip(vector[index], -INPUT_NORMALIZATION_CLIP, INPUT_NORMALIZATION_CLIP))
            result[index] = float(np.clip(_asinh(raw / _MTF_SIGNED_ATR_REFERENCE) / signed_denominator, -1.0, 1.0))
    for index in _MTF_VOLATILITY_RATIO_INDICES:
        if vector.size > index:
            ratio = float(np.clip(vector[index], 1.0 / _ATR_DISTANCE_MAX, _ATR_DISTANCE_MAX))
            # A ratio of one means ordinary local volatility, so it must map to
            # zero rather than a persistent positive entry bias.
            result[index] = float(np.tanh(np.log(ratio) / _MTF_VOLATILITY_LOG_SCALE))
    return result


def normalize_input_vector(
    values: Iterable[float] | np.ndarray,
    *,
    mode: str = INPUT_NORMALIZATION_RAW,
    size: int = INPUT_COUNT,
) -> np.ndarray:
    """Return one canonical observation vector for policy inference/learning.

    ``raw`` is an identity transform with finite/shape validation. ``tanh_v1``
    is globally bounded and symmetric. ``soft_clip_v1`` leaves values through
    +/-1 unchanged and maps the clipped tail smoothly into +/-2. Both bounded
    transforms are deterministic and have matching Python/MT5 implementations.
    """

    normalized_mode = str(mode or INPUT_NORMALIZATION_RAW).strip().lower()
    if normalized_mode not in INPUT_NORMALIZATION_MODES:
        raise ValueError(
            f"unsupported input normalization mode {mode!r}; "
            f"expected one of {sorted(INPUT_NORMALIZATION_MODES)}"
        )
    vector = finite_vector(values, size=size, name="input observation")
    if normalized_mode == INPUT_NORMALIZATION_RAW:
        return vector
    if normalized_mode == INPUT_NORMALIZATION_TANH_V1:
        return np.tanh(np.clip(vector, -INPUT_NORMALIZATION_CLIP, INPUT_NORMALIZATION_CLIP))
    if normalized_mode == INPUT_NORMALIZATION_SOFT_CLIP_V1:
        return _soft_clip(vector)
    if normalized_mode == INPUT_NORMALIZATION_GROUPWISE_V2:
        return _groupwise_v2(vector)
    if normalized_mode == INPUT_NORMALIZATION_GROUPWISE_MTF_V3:
        return _groupwise_mtf_v3(vector)
    return _structure_aware_v4(vector)


def normalize_input_matrix(
    values: Iterable[Iterable[float]] | np.ndarray,
    *,
    mode: str = INPUT_NORMALIZATION_RAW,
    size: int = INPUT_COUNT,
) -> np.ndarray:
    """Vectorized counterpart used by audit/tests without changing semantics."""

    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != int(size):
        raise ValueError(f"input matrix must have shape (N, {size}); got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("input matrix contains non-finite values")
    if str(mode or INPUT_NORMALIZATION_RAW).strip().lower() == INPUT_NORMALIZATION_RAW:
        return matrix.copy()
    return np.vstack(
        [normalize_input_vector(row, mode=mode, size=size) for row in matrix]
    )


def input_normalization_payload(mode: str = INPUT_NORMALIZATION_RAW) -> dict[str, object]:
    """Stable contract metadata included in normalized feature fingerprints."""

    normalized_mode = str(mode or INPUT_NORMALIZATION_RAW).strip().lower()
    if normalized_mode not in INPUT_NORMALIZATION_MODES:
        raise ValueError(
            f"unsupported input normalization mode {mode!r}; "
            f"expected one of {sorted(INPUT_NORMALIZATION_MODES)}"
        )
    payload: dict[str, object] = {
        "mode": normalized_mode,
        "version": INPUT_NORMALIZATION_VERSION,
    }
    if normalized_mode == INPUT_NORMALIZATION_TANH_V1:
        payload.update({"function": "tanh", "clip": INPUT_NORMALIZATION_CLIP, "range": "(-1,1)"})
    elif normalized_mode == INPUT_NORMALIZATION_SOFT_CLIP_V1:
        payload.update({
            "function": "soft_clip",
            "clip": INPUT_NORMALIZATION_CLIP,
            "knee": INPUT_NORMALIZATION_SOFT_CLIP_KNEE,
            "tail_scale": INPUT_NORMALIZATION_SOFT_CLIP_TAIL_SCALE,
            "range": "(-2,2)",
        })
    elif normalized_mode == INPUT_NORMALIZATION_GROUPWISE_V2:
        payload.update({
            "function": "groupwise_soft_clip_signed_log",
            "stateful_groups": "position,account,execution",
            "known_bounded": "chop_index,bars_since",
            "range": "approximately [-2,2]",
        })
    elif normalized_mode == INPUT_NORMALIZATION_GROUPWISE_MTF_V3:
        payload.update({
            "function": "groupwise_soft_clip_signed_log_plus_mtf_tanh",
            "stateful_groups": "position,account,execution",
            "mtf_native_indexes": "80:92",
            "mtf_native_function": "1.2*tanh",
            "range": "approximately [-2,2], MTF [-1.2,1.2]",
        })
    elif normalized_mode == INPUT_NORMALIZATION_STRUCTURE_AWARE_V4:
        payload.update({
            "function": "groupwise_plus_structural_asinh_and_ratio_log",
            "corrected_chop_contract": "unit_interval_to[-1,1]",
            "positive_atr_distance_indexes": ",".join(str(index) for index in _POSITIVE_ATR_DISTANCE_INDICES),
            "positive_atr_distance_function": "asinh(x/2)/asinh(10)",
            "mtf_volatility_ratio_indexes": ",".join(str(index) for index in _MTF_VOLATILITY_RATIO_INDICES),
            "mtf_volatility_ratio_function": "tanh(log(ratio)/0.35)",
            "range": "approximately [-1,1] for structural and MTF fields",
        })
    return payload


def input_normalization_summary(values: Iterable[float] | np.ndarray, *, mode: str) -> dict[str, object]:
    """Small audit summary proving that a row was bounded without storing data."""

    raw = finite_vector(values, size=INPUT_COUNT, name="input observation")
    transformed = normalize_input_vector(raw, mode=mode)
    return {
        "mode": str(mode or INPUT_NORMALIZATION_RAW).strip().lower(),
        "version": INPUT_NORMALIZATION_VERSION,
        "raw_max_abs": float(np.max(np.abs(raw), initial=0.0)),
        "normalized_max_abs": float(np.max(np.abs(transformed), initial=0.0)),
        "normalized_mean_abs": float(np.mean(np.abs(transformed))) if transformed.size else 0.0,
        "saturated_count": int(np.count_nonzero(np.abs(transformed) >= 0.999999)),
    }
