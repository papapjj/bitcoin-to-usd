"""Leakage-free mandatory hidden-state normalization for Direct-RNN layers.

The transform is per forward pass and has no running statistics or outcome
inputs. Every hidden block must use RMS or LayerNorm before ``tanh``.
"""
from __future__ import annotations

import numpy as np

HIDDEN_NORMALIZATION_RMS_V1 = "rms_v1"
HIDDEN_NORMALIZATION_LAYERNORM_V1 = "layernorm_v1"
HIDDEN_NORMALIZATION_MODES = frozenset({
    HIDDEN_NORMALIZATION_RMS_V1,
    HIDDEN_NORMALIZATION_LAYERNORM_V1,
})
HIDDEN_NORMALIZATION_VERSION = 1


def normalize_hidden(
    values: np.ndarray,
    *,
    mode: str = HIDDEN_NORMALIZATION_LAYERNORM_V1,
    target_rms: float = 1.0,
    clip: float = 3.0,
    eps: float = 1e-5,
) -> np.ndarray:
    """Normalize one hidden preactivation vector without state or leakage."""
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    arr[~np.isfinite(arr)] = 0.0
    normalized_mode = str(mode or HIDDEN_NORMALIZATION_LAYERNORM_V1).strip().lower()
    if arr.size == 0:
        return arr
    if normalized_mode not in {HIDDEN_NORMALIZATION_RMS_V1, HIDDEN_NORMALIZATION_LAYERNORM_V1}:
        raise ValueError(f"unsupported hidden normalization mode: {mode!r}")
    target = float(np.clip(target_rms, 1e-6, 10.0))
    limit = float(np.clip(clip, target, 20.0))
    safe_eps = float(np.clip(eps, 1e-9, 1.0))
    if normalized_mode == HIDDEN_NORMALIZATION_LAYERNORM_V1:
        centered = arr - float(np.mean(arr))
        variance = float(np.mean(np.square(centered)))
        arr = centered / float(np.sqrt(max(variance, 0.0) + safe_eps))
        rms = float(np.sqrt(np.mean(np.square(arr))))
        if rms > 1e-9:
            arr *= target / rms
    else:
        rms = float(np.sqrt(np.mean(np.square(arr))))
        if rms > target:
            arr *= target / rms
    return np.clip(arr, -limit, limit)


def hidden_normalization_payload(
    mode: str,
    *,
    target_rms: float = 1.0,
    clip: float = 3.0,
    eps: float = 1e-5,
) -> dict[str, object]:
    normalized_mode = str(mode or HIDDEN_NORMALIZATION_LAYERNORM_V1).strip().lower()
    if normalized_mode not in HIDDEN_NORMALIZATION_MODES:
        raise ValueError(f"unsupported hidden normalization mode: {mode!r}")
    return {
        "mode": normalized_mode,
        "version": HIDDEN_NORMALIZATION_VERSION,
        "target_rms": float(np.clip(target_rms, 1e-6, 10.0)),
        "clip": float(np.clip(clip, 1e-6, 20.0)),
        "eps": float(np.clip(eps, 1e-9, 1.0)),
        "per_forward_pass": True,
        "running_statistics": False,
        "leakage_free": True,
    }
