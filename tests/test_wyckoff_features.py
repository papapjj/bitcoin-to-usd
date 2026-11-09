from __future__ import annotations

import numpy as np
import pandas as pd

from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    BASE_MARKET_FEATURE_COLUMNS,
    EXTRA_MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
    add_features,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, TIMING_OFFSET, WYCKOFF_OFFSET


def _bars(with_volume: bool = True) -> pd.DataFrame:
    rows = 320
    close = 100.0 + 0.03 * np.sin(np.arange(rows) / 9.0)
    frame = pd.DataFrame(
        {
            "Datetime": pd.date_range("2025-01-01", periods=rows, freq="min"),
            "Open": close.copy(),
            "High": close + 0.20,
            "Low": close - 0.20,
            "Close": close.copy(),
            "Spread": np.full(rows, 10.0),
        }
    )
    if with_volume:
        frame["TickVolume"] = 100.0
    return frame


def _config() -> FeatureConfig:
    return FeatureConfig(atr_percentile_window=60, regime_window=60)


def test_feature_groups_match_the_versioned_137_layout() -> None:
    assert len(BASE_MARKET_FEATURE_COLUMNS) == 29
    assert len(EXTRA_MARKET_FEATURE_COLUMNS) == 29
    assert len(TIMING_FEATURE_COLUMNS) == 4
    assert len(WYCKOFF_FEATURE_COLUMNS) == 16
    assert TIMING_OFFSET == 117
    assert WYCKOFF_OFFSET == 121
    assert INPUT_COUNT == 137


def test_missing_volume_is_neutral_instead_of_fabricating_a_signal() -> None:
    featured = add_features(_bars(with_volume=False), _config(), SymbolConfig())
    row = featured.iloc[-1]

    assert np.isclose(float(row["relative_volume_20"]), 0.2)
    assert np.isclose(float(row["relative_volume_60"]), 0.2)
    assert np.isclose(float(row["volume_zscore_60"]), 0.0)
    assert np.isclose(float(row["volume_trend_20"]), 0.0)


def test_mt5_export_tickvol_alias_drives_volume_features() -> None:
    raw = _bars(with_volume=True).rename(columns={"TickVolume": "TickVol"})
    raw["Vol"] = 0.0
    raw.loc[250, "TickVol"] = 500.0
    timestamp = raw.loc[250, "Datetime"]

    featured = add_features(raw, _config(), SymbolConfig()).set_index("Datetime")
    row = featured.loc[pd.Timestamp(timestamp)]

    assert float(row["relative_volume_20"]) > 0.2
    assert float(row["volume_zscore_60"]) > 0.0


def test_spring_upthrust_and_low_volume_retests_are_sequence_aware() -> None:
    raw = _bars(with_volume=True)
    spring_time = raw.loc[200, "Datetime"]
    spring_test_time = raw.loc[205, "Datetime"]
    upthrust_time = raw.loc[230, "Datetime"]
    upthrust_test_time = raw.loc[235, "Datetime"]

    raw.loc[200, ["Open", "High", "Low", "Close", "TickVolume"]] = [99.9, 100.1, 98.8, 100.0, 350.0]
    raw.loc[205, ["Open", "High", "Low", "Close", "TickVolume"]] = [99.7, 100.0, 98.9, 99.6, 45.0]
    raw.loc[230, ["Open", "High", "Low", "Close", "TickVolume"]] = [100.1, 101.2, 99.9, 100.0, 360.0]
    raw.loc[235, ["Open", "High", "Low", "Close", "TickVolume"]] = [100.3, 101.1, 100.0, 100.4, 40.0]

    featured = add_features(raw, _config(), SymbolConfig()).set_index("Datetime")
    spring = featured.loc[pd.Timestamp(spring_time)]
    spring_test = featured.loc[pd.Timestamp(spring_test_time)]
    upthrust = featured.loc[pd.Timestamp(upthrust_time)]
    upthrust_test = featured.loc[pd.Timestamp(upthrust_test_time)]

    assert float(spring["spring_score"]) > 0.0
    assert float(spring["relative_volume_20"]) > 0.2
    assert float(spring_test["spring_test_quality"]) > 0.0
    assert 0.0 < float(spring_test["bars_since_spring"]) < 1.0
    assert float(upthrust["upthrust_score"]) > 0.0
    assert float(upthrust["relative_volume_20"]) > 0.2
    assert float(upthrust_test["upthrust_test_quality"]) > 0.0
    assert 0.0 < float(upthrust_test["bars_since_upthrust"]) < 1.0


def test_wyckoff_features_are_finite_and_bounded() -> None:
    raw = _bars(with_volume=True)
    raw["TickVolume"] = 100.0 + 80.0 * np.sin(np.arange(len(raw)) / 11.0)
    featured = add_features(raw, _config(), SymbolConfig())
    values = featured[WYCKOFF_FEATURE_COLUMNS].to_numpy(dtype=float)

    assert np.isfinite(values).all()
    assert np.max(np.abs(values)) <= 1.0 + 1e-12
