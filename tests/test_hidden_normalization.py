from __future__ import annotations

import numpy as np
import pytest

from self_growing_trader.hidden_normalization import normalize_hidden


def test_none_mode_is_rejected() -> None:
    values = np.asarray([-2.0, 0.5, 3.0], dtype=float)
    with pytest.raises(ValueError, match="unsupported hidden normalization mode"):
        normalize_hidden(values, mode="none")


def test_rms_hidden_normalization_only_reduces_large_drive() -> None:
    values = np.asarray([-10.0, 0.0, 10.0], dtype=float)
    result = normalize_hidden(values, mode="rms_v1", target_rms=1.0, clip=3.0)
    assert np.isfinite(result).all()
    assert np.sqrt(np.mean(np.square(result))) <= 1.0 + 1e-6


def test_layernorm_hidden_normalization_is_centered_and_bounded() -> None:
    values = np.asarray([2.0, 4.0, 8.0], dtype=float)
    result = normalize_hidden(values, mode="layernorm_v1", target_rms=1.0, clip=3.0)
    assert abs(float(np.mean(result))) < 1e-5
    assert np.isclose(float(np.sqrt(np.mean(np.square(result)))), 1.0, atol=1e-4)
    assert np.max(np.abs(result)) <= 3.0
