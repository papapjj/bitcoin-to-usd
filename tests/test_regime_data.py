"""
tests/test_regime_data.py

Automated unit and integration test suite for BTCUSD M1 Regime Data Pipeline.
Covers:
- Raw data loading and shape integrity on Q1 2025 BTCUSD M1 data.
- 15-dimensional feature shapes and column contracts.
- Strict causality verification: mutating future bars t+1..t+H must NOT alter features at bar t.
- Column name invariance (lowercase, uppercase, MT5 angle bracket tags).
- Deterministic Triple Barrier mechanics (synthetic Bull, Bear, Chop paths).
- Tri-modal balanced label distribution (~30-40% each for Bull, Bear, Chop).
- Boundary embargo purge: last 60 bars of January are purged to eliminate leakage into Feb OOS.
- Scaler isolation: standardizer fitted strictly on In-Sample January data.
- Warmup bar handling and absence of NaNs / Infs in prepared dataset.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler

try:
    from self_growing_trader.regime_data import (
        DEFAULT_CSV_PATH,
        REGIME_FEATURE_NAMES,
        REGIME_LABEL_BULL,
        REGIME_LABEL_BEAR,
        REGIME_LABEL_CHOP,
        REGIME_LABEL_PURGED,
        compute_regime_features,
        compute_triple_barrier_labels,
        extract_regime_features,
        generate_triple_barrier_labels,
        load_and_preprocess_q1_data,
        load_raw_btcusd_m1,
        prepare_dataset,
        split_q1_data,
    )
except ImportError:
    from src.self_growing_trader.regime_data import (
        DEFAULT_CSV_PATH,
        REGIME_FEATURE_NAMES,
        REGIME_LABEL_BULL,
        REGIME_LABEL_BEAR,
        REGIME_LABEL_CHOP,
        REGIME_LABEL_PURGED,
        compute_regime_features,
        compute_triple_barrier_labels,
        extract_regime_features,
        generate_triple_barrier_labels,
        load_and_preprocess_q1_data,
        load_raw_btcusd_m1,
        prepare_dataset,
        split_q1_data,
    )


@pytest.fixture(scope="module")
def real_csv_available() -> bool:
    return os.path.exists(DEFAULT_CSV_PATH)


@pytest.fixture(scope="module")
def q1_df(real_csv_available) -> pd.DataFrame:
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return load_raw_btcusd_m1(DEFAULT_CSV_PATH, "2025-01-01", "2025-03-31")


def test_load_raw_btcusd_m1_properties(q1_df: pd.DataFrame):
    """Verifies that BTCUSDm.csv Q1 2025 is loaded with exact row count and chronological order."""
    assert len(q1_df) == 129581, f"Expected 129,581 rows for Q1 2025, got {len(q1_df)}"
    assert q1_df["datetime"].is_monotonic_increasing, "Datetime must be strictly chronological"

    expected_cols = ["datetime", "open", "high", "low", "close", "tickvol", "vol", "spread", "<CLOSE>"]
    for col in expected_cols:
        assert col in q1_df.columns, f"Missing expected column: {col}"

    assert q1_df["datetime"].iloc[0] == pd.Timestamp("2025-01-01 00:00:00")
    assert q1_df["datetime"].iloc[-1] == pd.Timestamp("2025-03-31 23:59:00")
    assert not q1_df[["open", "high", "low", "close"]].isna().any().any(), "No NaN allowed in OHLC"


def test_load_raw_btcusd_m1_missing_file():
    """Verifies that non-existent CSV path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_btcusd_m1("c:\\non_existent_path\\bogus_btcusd.csv")


def test_split_q1_data(q1_df: pd.DataFrame):
    """Verifies splitting into In-Sample (Jan 2025) and Out-of-Sample (Feb & Mar 2025)."""
    df_train, df_oos = split_q1_data(q1_df, train_end="2025-01-31 23:59:59")
    assert len(df_train) == 44635, f"Expected 44,635 January bars, got {len(df_train)}"
    assert len(df_oos) == 84946, f"Expected 84,946 Feb+Mar bars, got {len(df_oos)}"
    assert len(df_train) + len(df_oos) == 129581

    assert df_train["datetime"].max() <= pd.Timestamp("2025-01-31 23:59:59")
    assert df_oos["datetime"].min() >= pd.Timestamp("2025-02-01 00:00:00")


def test_load_and_preprocess_q1_data_contract(real_csv_available):
    """Verifies PROJECT.md interface contract: load_and_preprocess_q1_data()."""
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    df_train, df_oos = load_and_preprocess_q1_data(DEFAULT_CSV_PATH)
    assert isinstance(df_train, pd.DataFrame)
    assert isinstance(df_oos, pd.DataFrame)
    assert len(df_train) == 44635
    assert len(df_oos) == 84946


def test_compute_regime_features_shape_and_names(q1_df: pd.DataFrame):
    """Verifies that compute_regime_features produces [N, 15] DataFrame with exact feature names."""
    sample = q1_df.iloc[:500].copy()
    features = compute_regime_features(sample)

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (500, 15)
    assert list(features.columns) == REGIME_FEATURE_NAMES

    # extract_regime_features contract
    raw_array = extract_regime_features(sample)
    assert isinstance(raw_array, np.ndarray)
    assert raw_array.shape == (500, 15)
    assert raw_array.dtype == np.float64


def test_compute_regime_features_strict_causality(q1_df: pd.DataFrame):
    """
    CRUCIAL CAUSALITY TEST:
    Mutating future bars t+1..T must NOT alter ANY feature at bar t or earlier.
    Zero lookahead bias guarantee.
    """
    # Use first 600 bars
    df_base = q1_df.iloc[:600].copy().reset_index(drop=True)
    feats_base = compute_regime_features(df_base)

    target_bar = 350

    # Create mutated future scenario: alter future prices, volumes, spreads drastically
    df_mutated = df_base.copy()
    df_mutated.loc[target_bar + 1 :, "close"] *= 3.5
    df_mutated.loc[target_bar + 1 :, "high"] *= 4.0
    df_mutated.loc[target_bar + 1 :, "low"] *= 0.2
    df_mutated.loc[target_bar + 1 :, "open"] *= 2.0
    df_mutated.loc[target_bar + 1 :, "tickvol"] += 50000
    df_mutated.loc[target_bar + 1 :, "spread"] += 1500

    feats_mutated = compute_regime_features(df_mutated)

    # For all bars s <= target_bar, features must be identical down to floating point precision
    for col in REGIME_FEATURE_NAMES:
        base_vals = feats_base.loc[:target_bar, col].to_numpy()
        mut_vals = feats_mutated.loc[:target_bar, col].to_numpy()

        # Ignore initial warmup NaNs
        valid = ~np.isnan(base_vals)
        assert np.all(valid == ~np.isnan(mut_vals)), f"Warmup NaN mismatch in {col}"
        np.testing.assert_allclose(
            base_vals[valid],
            mut_vals[valid],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Causality violation detected in feature '{col}' at or before bar {target_bar}!",
        )


def test_column_name_invariance():
    """Verifies that features are invariant to column casing and naming variations."""
    n = 100
    rng = np.random.default_rng(42)
    close = 90000.0 + np.cumsum(rng.standard_normal(n) * 20.0)
    high = close + rng.uniform(5.0, 25.0, n)
    low = close - rng.uniform(5.0, 25.0, n)
    open_ = close + rng.uniform(-10.0, 10.0, n)
    tickvol = rng.integers(100, 500, n).astype(float)
    spread = rng.integers(2000, 3000, n).astype(float)

    df_lower = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "tickvol": tickvol, "spread": spread,
    })
    df_upper = pd.DataFrame({
        "<OPEN>": open_, "<HIGH>": high, "<LOW>": low, "<CLOSE>": close,
        "<TICKVOL>": tickvol, "<SPREAD>": spread,
    })
    df_capital = pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "TickVolume": tickvol, "Spread": spread,
    })

    f_lower = compute_regime_features(df_lower)
    f_upper = compute_regime_features(df_upper)
    f_capital = compute_regime_features(df_capital)

    for col in REGIME_FEATURE_NAMES:
        v1 = f_lower[col].to_numpy()
        v2 = f_upper[col].to_numpy()
        v3 = f_capital[col].to_numpy()
        valid = ~np.isnan(v1)
        np.testing.assert_allclose(v1[valid], v2[valid], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(v1[valid], v3[valid], rtol=1e-12, atol=1e-12)


def test_triple_barrier_deterministic_mechanics():
    """
    Synthetic tests verifying exact mechanics of Triple Barrier Method:
    - Path hitting upper barrier first -> 0 (BULL)
    - Path hitting lower barrier first -> 1 (BEAR)
    - Path staying within barriers for H bars -> 2 (CHOP)
    - Tail bars -> -1 (PURGED)
    """
    h = 10
    k = 2.0
    n = 60

    # Base flat prices
    close = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    open_ = np.full(n, 100.0)
    # TR = 2.0, ATR14 = 2.0, k*ATR = 4.0
    # Upper barrier = 104.0, Lower barrier = 96.0

    # Bar 30: trigger bull breakout at bar 30 (high reaches 105.0)
    high[30] = 105.0
    # Bar 40: trigger bear breakdown at bar 40 (low reaches 95.0)
    low[40] = 95.0

    df_synth = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    labels = compute_triple_barrier_labels(df_synth, horizon=h, k_atr=k, purge_tail=True)

    # Bar 14 has horizon [15..24] where high=101 < 104 and low=99 > 96
    # -> stays strictly within barriers for all 10 bars -> CHOP (2)
    assert labels[14] == REGIME_LABEL_CHOP, f"Expected CHOP for bar 14, got {labels[14]}"

    # Bar 25 has horizon [26..35] hitting high=105.0 at bar 30 first -> BULL (0)
    assert labels[25] == REGIME_LABEL_BULL, f"Expected BULL for bar 25, got {labels[25]}"

    # Bar 35 has horizon [36..45] hitting low=95.0 at bar 40 first -> BEAR (1)
    assert labels[35] == REGIME_LABEL_BEAR, f"Expected BEAR for bar 35, got {labels[35]}"

    # Tail bars (last h=10 bars: index 50 to 59) must be purged to -1
    for t in range(n - h, n):
        assert labels[t] == REGIME_LABEL_PURGED, f"Tail bar {t} must be purged (-1), got {labels[t]}"

    # Test PROJECT.md contract alias
    labels_alias = generate_triple_barrier_labels(df_synth, horizon=h, k_atr=k)
    np.testing.assert_array_equal(labels, labels_alias)


def test_triple_barrier_distribution_balance(q1_df: pd.DataFrame):
    """
    Verifies that Triple Barrier Ground Truth (H=60, k=5.0 ATR) produces a balanced
    tri-modal distribution (~30-40% each for Bull, Bear, and Chop) on real Q1 data.
    """
    labels = compute_triple_barrier_labels(q1_df, horizon=60, k_atr=5.0, purge_tail=True)
    valid_labels = labels[labels != REGIME_LABEL_PURGED]

    assert len(valid_labels) > 0
    counts = pd.Series(valid_labels).value_counts(normalize=True)

    prop_bull = counts.get(REGIME_LABEL_BULL, 0.0)
    prop_bear = counts.get(REGIME_LABEL_BEAR, 0.0)
    prop_chop = counts.get(REGIME_LABEL_CHOP, 0.0)

    # Verify each regime is between 25% and 45% (target ~30-40%)
    assert 0.25 <= prop_bull <= 0.45, f"Bull proportion out of expected range: {prop_bull:.3f}"
    assert 0.25 <= prop_bear <= 0.45, f"Bear proportion out of expected range: {prop_bear:.3f}"
    assert 0.25 <= prop_chop <= 0.45, f"Chop proportion out of expected range: {prop_chop:.3f}"

    # Verify symmetry between Bull and Bear
    assert abs(prop_bull - prop_bear) < 0.05, f"Bull/Bear imbalance: {prop_bull} vs {prop_bear}"


def test_embargo_purge_boundary(real_csv_available):
    """
    Verifies that the Jan/Feb boundary embargo purge eliminates forward leakage into February OOS.
    Last 60 bars of January must be purged and never appear in X_train / y_train.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, horizon=60, k_atr=5.0)

    # Check timestamps of train vs oos
    train_end_ts = pd.Timestamp("2025-01-31 23:59:59")
    max_train_ts = data["train_timestamps"].max()
    min_oos_ts = data["oos_timestamps"].min()

    assert max_train_ts <= train_end_ts
    assert min_oos_ts >= pd.Timestamp("2025-02-01 00:00:00")

    # The last 60 minutes of January are 23:00 to 23:59.
    # Because of the embargo purge, the max train timestamp must be <= 22:59:00
    assert max_train_ts <= pd.Timestamp("2025-01-31 22:59:00"), (
        f"Embargo purge failed: train timestamp {max_train_ts} exceeds embargo cutoff 22:59:00!"
    )


def test_scaler_isolation(real_csv_available):
    """
    Verifies scaler isolation:
    - Scaler must be fit strictly on In-Sample January data.
    - Transform applies to both Train and OOS without leakage.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    # Test RobustScaler
    data_robust = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="robust")
    scaler_r = data_robust["scaler"]
    assert isinstance(scaler_r, RobustScaler)

    # Center of robust scaler must match median of X_train_raw
    expected_center = np.median(data_robust["X_train_raw"], axis=0)
    np.testing.assert_allclose(scaler_r.center_, expected_center, rtol=1e-5, atol=1e-5)

    # Test StandardScaler
    data_std = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="standard")
    scaler_s = data_std["scaler"]
    assert isinstance(scaler_s, StandardScaler)

    expected_mean = np.mean(data_std["X_train_raw"], axis=0)
    np.testing.assert_allclose(scaler_s.mean_, expected_mean, rtol=1e-5, atol=1e-5)


def test_prepare_dataset_full_integrity(real_csv_available):
    """
    Verifies complete end-to-end dataset preparation:
    - Shapes: X_train (44335, 15), y_train (44335,), X_oos (84886, 15), y_oos (84886,).
    - Zero NaNs and zero Infs in features and labels.
    - Balanced class distribution across both train and OOS partitions.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_oos = data["X_oos"]
    y_oos = data["y_oos"]

    assert X_train.shape == (44335, 15)
    assert y_train.shape == (44335,)
    assert X_oos.shape == (84886, 15)
    assert y_oos.shape == (84886,)

    assert not np.isnan(X_train).any(), "NaN found in X_train"
    assert not np.isinf(X_train).any(), "Inf found in X_train"
    assert not np.isnan(X_oos).any(), "NaN found in X_oos"
    assert not np.isinf(X_oos).any(), "Inf found in X_oos"

    # Classes in y_train and y_oos must be strictly in {0, 1, 2}
    assert set(np.unique(y_train)) == {0, 1, 2}
    assert set(np.unique(y_oos)) == {0, 1, 2}


def test_edge_cases_empty_and_short_inputs():
    """Verifies behavior on empty dataframes, short sequences, and invalid parameters."""
    empty_df = pd.DataFrame(columns=["open", "high", "low", "close", "tickvol", "spread"])
    empty_feats = compute_regime_features(empty_df)
    assert empty_feats.empty
    assert list(empty_feats.columns) == REGIME_FEATURE_NAMES

    empty_labels = compute_triple_barrier_labels(empty_df)
    assert len(empty_labels) == 0

    with pytest.raises(ValueError):
        compute_triple_barrier_labels(empty_df, horizon=0)
    with pytest.raises(ValueError):
        compute_triple_barrier_labels(empty_df, k_atr=-1.0)

    # Short df where len(df) < horizon
    short_df = pd.DataFrame({
        "open": [100.0] * 10,
        "high": [101.0] * 10,
        "low": [99.0] * 10,
        "close": [100.0] * 10,
        "tickvol": [100.0] * 10,
        "spread": [10.0] * 10,
    })
    short_labels = compute_triple_barrier_labels(short_df, horizon=20, purge_tail=True)
    assert len(short_labels) == 10
    assert np.all(short_labels == REGIME_LABEL_PURGED)

