"""Leakage-free normalization for policy action outputs.

The policy emits twelve discrete-action scores followed by eight management
parameters.  ``rms_v1`` only rescales the discrete block when its RMS exceeds
the configured target; ``layernorm_v1`` centers and scales that block using
only the current forward pass.  The management parameters are intentionally
untouched.  Neither mode uses outcomes, replay, percentiles, or running
statistics, so the contract stays leakage-free and candidate-only.
"""
from __future__ import annotations

import numpy as np

OUTPUT_NORMALIZATION_VERSION = 1
OUTPUT_NORMALIZATION_RAW = "raw"
OUTPUT_NORMALIZATION_RMS_V1 = "rms_v1"
OUTPUT_NORMALIZATION_LAYERNORM_V1 = "layernorm_v1"
OUTPUT_NORMALIZATION_MODES = frozenset({
    OUTPUT_NORMALIZATION_RAW,
    OUTPUT_NORMALIZATION_RMS_V1,
    OUTPUT_NORMALIZATION_LAYERNORM_V1,
})
OUTPUT_NORMALIZATION_ACTION_COUNT = 12
OUTPUT_NORMALIZATION_TARGET_RMS = 0.70
OUTPUT_NORMALIZATION_CLIP = 3.0


def normalize_policy_output(
    values: np.ndarray,
    *,
    mode: str = OUTPUT_NORMALIZATION_RAW,
    target_rms: float = OUTPUT_NORMALIZATION_TARGET_RMS,
    clip: float = OUTPUT_NORMALIZATION_CLIP,
    action_count: int = OUTPUT_NORMALIZATION_ACTION_COUNT,
    action_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return a finite output vector with candidate-only action calibration.

    ``raw`` is an exact finite identity.  ``rms_v1`` scales only the first
    ``action_count`` rows down when their RMS is above ``target_rms``; it never
    amplifies weak evidence and leaves parameter rows unchanged.  ``layernorm_v1``
    applies per-forward-pass centering followed by a target-RMS scale to the
    same action block, with no state carried between decisions.
    """
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    arr[~np.isfinite(arr)] = 0.0
    normalized_mode = str(mode or OUTPUT_NORMALIZATION_RAW).strip().lower()
    if normalized_mode == OUTPUT_NORMALIZATION_RAW or arr.size == 0:
        return arr
    if normalized_mode not in {OUTPUT_NORMALIZATION_RMS_V1, OUTPUT_NORMALIZATION_LAYERNORM_V1}:
        raise ValueError(f"unsupported output normalization mode: {mode!r}")
    target = float(np.clip(target_rms, 1e-6, 10.0))
    limit = float(np.clip(clip, target, 20.0))
    count = min(max(1, int(action_count)), arr.size)
    action = arr[:count]
    if action_mask is None:
        mask = np.ones(count, dtype=bool)
    else:
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.size != count:
            raise ValueError("action_mask must match the normalized action block")
        if not np.any(mask):
            mask = mask.copy()
            mask[0] = True
    active = action[mask]
    if normalized_mode == OUTPUT_NORMALIZATION_LAYERNORM_V1:
        mean = float(np.mean(active)) if active.size else 0.0
        variance = float(np.mean(np.square(active - mean))) if active.size else 0.0
        std = float(np.sqrt(max(variance, 0.0) + 1e-6))
        active = (active - mean) / std
        rms = float(np.sqrt(np.mean(np.square(active)))) if active.size else 0.0
        if rms > 1e-9:
            active *= target / rms
    else:
        rms = float(np.sqrt(np.mean(np.square(active)))) if active.size else 0.0
        if rms > target:
            active *= target / rms
    action = action.copy()
    action[mask] = np.clip(active, -limit, limit)
    arr[:count] = action
    return arr


def policy_output_normalization_vjp(
    raw_action: np.ndarray,
    upstream: np.ndarray,
    *,
    mode: str = OUTPUT_NORMALIZATION_RAW,
    target_rms: float = OUTPUT_NORMALIZATION_TARGET_RMS,
    clip: float = OUTPUT_NORMALIZATION_CLIP,
    action_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Return the exact action-block VJP for :func:`normalize_policy_output`.

    Full-RL must optimise the logits actually used to choose the MT5 action.
    The older path cached pre-normalization logits, then optimised them as if
    normalization were an identity.  This helper closes that gap for the two
    candidate modes and treats clipped coordinates as locally constant.
    """
    raw = np.asarray(raw_action, dtype=float).reshape(-1)
    gradient = np.asarray(upstream, dtype=float).reshape(-1).copy()
    if raw.size == 0 or gradient.size != raw.size:
        return np.zeros_like(raw)
    raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
    normalized_mode = str(mode or OUTPUT_NORMALIZATION_RAW).strip().lower()
    if normalized_mode == OUTPUT_NORMALIZATION_RAW:
        return gradient
    if normalized_mode not in {OUTPUT_NORMALIZATION_RMS_V1, OUTPUT_NORMALIZATION_LAYERNORM_V1}:
        raise ValueError(f"unsupported output normalization mode: {mode!r}")

    target = float(np.clip(target_rms, 1e-6, 10.0))
    limit = float(np.clip(clip, target, 20.0))
    if action_mask is None:
        mask = np.ones(raw.size, dtype=bool)
    else:
        mask = np.asarray(action_mask, dtype=bool).reshape(-1)
        if mask.size != raw.size:
            raise ValueError("action_mask must match raw_action")
        if not np.any(mask):
            mask = mask.copy()
            mask[0] = True
    normalized = normalize_policy_output(
        raw,
        mode=normalized_mode,
        target_rms=target,
        clip=limit,
        action_count=raw.size,
        action_mask=mask,
    )
    # The clip in the forward pass has zero local derivative at the boundary.
    gradient *= mask.astype(float)
    gradient *= (np.abs(normalized) < limit - 1e-12).astype(float)
    active_raw = raw[mask]
    active_gradient = gradient[mask]
    if normalized_mode == OUTPUT_NORMALIZATION_RMS_V1:
        rms = float(np.sqrt(np.mean(np.square(active_raw))))
        if rms <= target or rms <= 1e-12:
            return gradient
        result = np.zeros_like(gradient)
        result[mask] = (target / rms) * (
            active_gradient
            - active_raw * float(np.mean(active_gradient * active_raw)) / max(rms * rms, 1e-12)
        )
        return result

    centered = active_raw - float(np.mean(active_raw))
    variance = float(np.mean(np.square(centered)))
    if variance <= 1e-24:
        return np.zeros_like(gradient)
    denom = float(np.sqrt(variance))
    result = np.zeros_like(gradient)
    result[mask] = (target / denom) * (
        active_gradient
        - float(np.mean(active_gradient))
        - centered * float(np.mean(active_gradient * centered)) / max(variance, 1e-12)
    )
    return result


def output_normalization_payload(
    mode: str,
    *,
    target_rms: float = OUTPUT_NORMALIZATION_TARGET_RMS,
    clip: float = OUTPUT_NORMALIZATION_CLIP,
    action_count: int = OUTPUT_NORMALIZATION_ACTION_COUNT,
) -> dict[str, object]:
    """Return the serializable output contract stored with a candidate run."""
    normalized_mode = str(mode or OUTPUT_NORMALIZATION_RAW).strip().lower()
    if normalized_mode not in OUTPUT_NORMALIZATION_MODES:
        raise ValueError(f"unsupported output normalization mode: {mode!r}")
    return {
        "mode": normalized_mode,
        "version": OUTPUT_NORMALIZATION_VERSION,
        "target_rms": float(np.clip(target_rms, 1e-6, 10.0)),
        "clip": float(np.clip(clip, 1e-6, 20.0)),
        "action_count": int(max(1, action_count)),
        "parameter_rows_untouched": True,
        "leakage_free": True,
    }
