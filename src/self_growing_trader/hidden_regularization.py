"""Layer-wise dropout controls shared by the recurrent policy cores.

The project has two compatible policy implementations (the context-expert
runtime and the older self-growing runner).  Keeping the small parsing and
masking contract here prevents the two ModuleState variants from drifting.
Dropout is deterministic for a given seed/step/layer, is inverted (so the
expected activation is preserved), and is never applied unless the caller
explicitly marks the forward pass as training.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np


MAX_LAYERWISE_VALUES = 64
MAX_DROPOUT_RATE = 0.95
DEFAULT_REQUIRED_HIDDEN_DROPOUT_RATES = (0.05, 0.10, 0.10, 0.05, 0.10, 0.10)


def _as_sequence(value: object) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        return tuple(part.strip() for part in text.split(",") if part.strip())
    if isinstance(value, Iterable):
        return tuple(value)
    return (value,)


def layerwise_bools(value: object, count: int, *, default: bool = False) -> tuple[bool, ...]:
    """Expand a config sequence to exactly ``count`` layer switches."""

    size = max(0, int(count))
    if size == 0:
        return ()
    raw = _as_sequence(value)
    if not raw:
        return tuple(bool(default) for _ in range(size))
    parsed: list[bool] = []
    for item in raw[:MAX_LAYERWISE_VALUES]:
        if isinstance(item, str):
            parsed.append(item.strip().lower() in {"1", "true", "yes", "on"})
        else:
            parsed.append(bool(item))
    if not parsed:
        parsed = [bool(default)]
    return tuple(parsed[min(index, len(parsed) - 1)] for index in range(size))


def layerwise_rates(value: object, count: int, *, default: float = 0.0) -> tuple[float, ...]:
    """Expand and clamp per-layer dropout rates.

    A one-value config is intentionally broadcast to all hidden blocks.  A
    shorter list repeats its final value so adding depth does not silently
    disable regularization in newly grown layers.
    """

    size = max(0, int(count))
    if size == 0:
        return ()
    raw = _as_sequence(value)
    if not raw:
        raw = (default,)
    parsed: list[float] = []
    for item in raw[:MAX_LAYERWISE_VALUES]:
        try:
            rate = float(item)
        except (TypeError, ValueError):
            rate = float(default)
        parsed.append(float(np.clip(rate, 0.0, MAX_DROPOUT_RATE)))
    if not parsed:
        parsed = [float(np.clip(default, 0.0, MAX_DROPOUT_RATE))]
    return tuple(parsed[min(index, len(parsed) - 1)] for index in range(size))


def mandatory_layerwise_rates(value: object, count: int) -> tuple[float, ...]:
    """Return positive dropout rates for every hidden block.

    Config validation rejects an off value. This second boundary upgrades old
    checkpoint state or hand-built runtime state that bypassed the parser.
    A valid positive per-layer schedule is retained unchanged.
    """

    rates = layerwise_rates(value, count)
    if rates and all(0.0 < float(rate) <= MAX_DROPOUT_RATE for rate in rates):
        return rates
    return layerwise_rates(DEFAULT_REQUIRED_HIDDEN_DROPOUT_RATES, count)


def deterministic_dropout(
    values: np.ndarray,
    rate: float,
    *,
    seed: int,
    step: int,
    layer_index: int,
) -> np.ndarray:
    """Apply inverted dropout using a reproducible per-layer mask."""

    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    probability = float(np.clip(rate, 0.0, MAX_DROPOUT_RATE))
    if probability <= 0.0 or arr.size == 0:
        return arr
    # Mix the coordinates with large odd constants rather than relying on a
    # process-global RNG.  This makes replay/checkpoint tests reproducible and
    # keeps masks independent across the six recurrent blocks.
    mixed = (
        (int(seed) & 0x7FFFFFFF)
        ^ ((int(step) + 1) * 0x9E3779B1)
        ^ ((int(layer_index) + 1) * 0x85EBCA6B)
    ) & 0x7FFFFFFF
    rng = np.random.default_rng(mixed)
    keep = rng.random(arr.size) >= probability
    scale = 1.0 / max(1.0 - probability, 1e-6)
    return arr * keep.astype(float) * scale


__all__ = [
    "MAX_DROPOUT_RATE",
    "MAX_LAYERWISE_VALUES",
    "DEFAULT_REQUIRED_HIDDEN_DROPOUT_RATES",
    "deterministic_dropout",
    "layerwise_bools",
    "layerwise_rates",
    "mandatory_layerwise_rates",
]
