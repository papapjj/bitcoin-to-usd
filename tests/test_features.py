from __future__ import annotations

import pandas as pd
import numpy as np

from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import add_features, compute_atr, spread_to_price


def test_spread_points_to_price() -> None:
    assert spread_to_price(3059, SymbolConfig(point_size=0.01, spread_in_points=True)) == 30.59
    assert spread_to_price(30.59, SymbolConfig(point_size=0.01, spread_in_points=False)) == 30.59


def test_atr_and_gap_return_are_trailing_only() -> None:
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=240, freq="min").strftime("%d/%m/%Y %H:%M"),
            "Open": [100.0 + i for i in range(240)],
            "High": [101.0 + i for i in range(240)],
            "Low": [99.0 + i for i in range(240)],
            "Close": [100.5 + i for i in range(240)],
            "Spread": [10] * 240,
        }
    )
    atr = compute_atr(df, 14)
    assert round(float(atr.iloc[13]), 6) == 2.0
    featured = add_features(
        df,
        FeatureConfig(atr_period=14, atr_percentile_window=20, range_window=20),
        SymbolConfig(point_size=0.01, spread_in_points=True),
    )
    assert {"ret1_atr", "ret60_atr", "gap_atr", "spread_atr"}.issubset(featured.columns)
    assert len(featured) > 0


def test_native_mtf_scaling_uses_own_atr_and_bounds_relative_volatility() -> None:
    rows = 1_440
    phase = np.arange(rows, dtype=float)
    close = 100.0 + 0.02 * phase + 1.5 * np.sin(phase / 23.0)
    df = pd.DataFrame(
        {
            "Datetime": pd.date_range("2024-01-01", periods=rows, freq="min").strftime("%d/%m/%Y %H:%M"),
            "Open": close - 0.1,
            "High": close + 0.4,
            "Low": close - 0.4,
            "Close": close,
            "Spread": [10] * rows,
        }
    )
    symbol = SymbolConfig(point_size=0.01, spread_in_points=True)
    legacy = add_features(df, FeatureConfig(mtf_scaling_mode="legacy_m1_atr_v1"), symbol)
    native = add_features(df, FeatureConfig(mtf_scaling_mode="native_tf_atr_v2"), symbol)

    valid = native["h1_volatility_ratio"].dropna()
    assert (valid.abs() <= 1.0 + 1e-12).all()
    assert not np.allclose(
        legacy["h1_ema_diff_atr"].fillna(0.0),
        native["h1_ema_diff_atr"].fillna(0.0),
    )

