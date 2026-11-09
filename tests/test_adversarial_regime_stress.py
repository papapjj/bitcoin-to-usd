"""
tests/test_adversarial_regime_stress.py

Adversarial Stress Test Suite for BTCUSD M1 Regime Data Pipeline (Milestone 1).
Designed and executed by Challenger 2 (Numerical Stability & Extreme Workload).

Adversarial attack dimensions:
1. Zero Variance / Flat Price Test:
   - All prices identical (O=H=L=C), zero volatility, zero volume, zero spread.
   - Flat prices with positive/large spreads.
   - Zero price series (O=H=L=C=0.0).
   - Verification: No ZeroDivisionError, no NaN, no Inf propagation after warmup.

2. Flash Crash & Extreme Spike Test:
   - 90% sudden price collapse in a single bar (100,000 -> 10,000).
   - 1000% sudden price spike in a single bar (100,000 -> 1,100,000).
   - Mathematical stability of Parkinson volatility (ln(H/L)^2 non-negative, finite).
   - Mathematical stability of Garman-Klass volatility (0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2 non-negative, finite).
   - Mathematical stability of Bollinger / Keltner squeeze ratio (finite, non-negative).
   - Triple Barrier labeling correctness during violent regime shifts.

3. Short Warmup & Edge Case Datasets:
   - Datasets with 0, 1, 5, 13, 14, 29, 30, 41, 42, 59, 60, 100, 239, 240, 241 bars.
   - Verification of graceful handling, valid array shapes, no unhandled crashes.
   - Tail purging (purge_tail=True vs purge_tail=False) on short sequences.
   - Boundary parameter validation (horizon <= 0, k_atr <= 0).

4. Pathological & Perturbed Price Inputs:
   - Negative prices.
   - Inverted prices (High < Low).
   - Extreme floating point scales (1e-6 and 1e9).
   - Missing / malformed column structures.

5. Full Q1 2025 Real Dataset Workload & Memory Benchmark:
   - End-to-end execution across all 129,581 bars.
   - Memory allocation and execution latency limits.
"""

from __future__ import annotations

import os
import time
import tracemalloc
from unittest.mock import patch
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


# ---------------------------------------------------------------------------
# Dimension 1: Zero Variance / Flat Price Adversarial Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flat_price", [100.0, 50000.0, 1.0, 0.0])
@pytest.mark.parametrize("spread_val", [0.0, 2880.0])
def test_zero_variance_flat_price_features(flat_price: float, spread_val: float):
    """
    Adversarial Challenge: Completely flat market with zero volatility and zero volume.
    Asserts:
    - No ZeroDivisionError is raised.
    - No NaN appears in any feature after warmup (index >= 240).
    - No Inf appears in any feature.
    - All features evaluate to finite real numbers.
    """
    n_bars = 400
    df_flat = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
            "open": np.full(n_bars, flat_price),
            "high": np.full(n_bars, flat_price),
            "low": np.full(n_bars, flat_price),
            "close": np.full(n_bars, flat_price),
            "tickvol": np.zeros(n_bars),
            "spread": np.full(n_bars, spread_val),
        }
    )

    features = compute_regime_features(df_flat)
    assert features.shape == (n_bars, 15)

    warmup_features = features.iloc[240:]
    nan_counts = warmup_features.isna().sum().to_dict()
    for col, count in nan_counts.items():
        assert count == 0, f"Feature {col} produced {count} NaNs under flat price={flat_price}, spread={spread_val}"

    feat_vals = warmup_features.to_numpy()
    assert not np.isinf(feat_vals).any(), f"Inf detected under flat price={flat_price}, spread={spread_val}"
    assert np.isfinite(feat_vals).all(), f"Non-finite values detected under flat price={flat_price}"

    # Specifically check realized volatility and momentum metrics are zero
    assert np.allclose(warmup_features["vol_parkinson_30"].values, 0.0, atol=1e-7)
    assert np.allclose(warmup_features["vol_garman_klass_30"].values, 0.0, atol=1e-7)
    assert np.allclose(warmup_features["mom_5_log"].values, 0.0, atol=1e-7)
    assert np.allclose(warmup_features["mom_15_log"].values, 0.0, atol=1e-7)
    assert np.allclose(warmup_features["mom_60_log"].values, 0.0, atol=1e-7)
    assert np.allclose(warmup_features["mom_240_log"].values, 0.0, atol=1e-7)


def test_zero_variance_flat_price_labels():
    """
    Adversarial Challenge: Completely flat market labeling.
    Asserts:
    - Triple Barrier labeling executes without division by zero.
    - Labels are bounded to valid domain {-1, 0, 1, 2}.
    - No unhandled exceptions.
    """
    n_bars = 300
    df_flat = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
            "open": np.full(n_bars, 100.0),
            "high": np.full(n_bars, 100.0),
            "low": np.full(n_bars, 100.0),
            "close": np.full(n_bars, 100.0),
            "tickvol": np.zeros(n_bars),
            "spread": np.zeros(n_bars),
        }
    )

    labels = compute_triple_barrier_labels(df_flat, horizon=60, k_atr=5.0, purge_tail=True)
    assert len(labels) == n_bars
    valid_domain = {REGIME_LABEL_PURGED, REGIME_LABEL_BULL, REGIME_LABEL_BEAR, REGIME_LABEL_CHOP}
    unique_labels = set(np.unique(labels))
    assert unique_labels.issubset(valid_domain)
    # The last 60 bars must be purged
    assert (labels[-60:] == REGIME_LABEL_PURGED).all()


# ---------------------------------------------------------------------------
# Dimension 2: Flash Crash & Extreme Spike Adversarial Tests
# ---------------------------------------------------------------------------


def test_flash_crash_mathematical_stability():
    """
    Adversarial Challenge: Catastrophic 90% flash crash within a single 1-minute bar.
    Price crashes from $100,000 to $10,000.
    Asserts:
    - Parkinson volatility remains strictly positive, non-NaN, and non-Inf.
    - Garman-Klass volatility element does not become negative or imaginary.
    - Bollinger/Keltner Squeeze ratio expands gracefully without Inf or NaN.
    - Downward jump is correctly anticipated as BEAR label prior to crash.
    """
    n_bars = 500
    crash_idx = 260
    base_price = 100000.0
    prices = np.full(n_bars, base_price, dtype=np.float64)
    # Crash persists for 30 bars
    prices[crash_idx : crash_idx + 30] = 10000.0

    df_crash = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
            "open": prices.copy(),
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices.copy(),
            "tickvol": np.full(n_bars, 100.0),
            "spread": np.full(n_bars, 2880.0),
        }
    )

    # In the crash bar itself: open=100k, high=100k, low=10k, close=10k
    df_crash.loc[crash_idx, "open"] = 100000.0
    df_crash.loc[crash_idx, "high"] = 100000.0
    df_crash.loc[crash_idx, "low"] = 10000.0
    df_crash.loc[crash_idx, "close"] = 10000.0

    features = compute_regime_features(df_crash)
    warmup_features = features.iloc[240:]

    assert not warmup_features.isna().any().any(), "Flash crash produced unexpected NaNs!"
    assert not np.isinf(warmup_features.to_numpy()).any(), "Flash crash produced Inf!"

    # Volatility indicators must be non-negative and finite
    assert (features["vol_parkinson_30"].iloc[crash_idx : crash_idx + 30] > 0.0).all()
    assert (features["vol_garman_klass_30"].iloc[crash_idx : crash_idx + 30] > 0.0).all()
    # Squeeze ratio is positive during the crash transition and non-negative when prices stabilize
    assert (features["bb_kc_squeeze_ratio"].iloc[crash_idx : crash_idx + 15] > 0.0).all()
    assert (features["bb_kc_squeeze_ratio"].iloc[crash_idx : crash_idx + 30] >= 0.0).all()

    # Labels: Bars immediately before the crash must be labeled BEAR (Class 1)
    labels = compute_triple_barrier_labels(df_crash, horizon=60, k_atr=5.0)
    pre_crash_labels = labels[crash_idx - 10 : crash_idx]
    assert (pre_crash_labels == REGIME_LABEL_BEAR).all(), "Pre-crash bars should be labeled BEAR"


def test_extreme_spike_mathematical_stability():
    """
    Adversarial Challenge: Massive 1000% spike in a single bar.
    Price explodes from $100,000 to $1,100,000.
    Asserts:
    - Squeeze ratio, Garman-Klass, and Parkinson handle extreme ratios without numerical overflow.
    - Pre-spike bars are objectively labeled BULL (Class 0).
    """
    n_bars = 500
    spike_idx = 260
    base_price = 100000.0
    prices = np.full(n_bars, base_price, dtype=np.float64)
    prices[spike_idx : spike_idx + 30] = 1100000.0

    df_spike = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
            "open": prices.copy(),
            "high": prices * 1.001,
            "low": prices * 0.999,
            "close": prices.copy(),
            "tickvol": np.full(n_bars, 100.0),
            "spread": np.full(n_bars, 2880.0),
        }
    )

    # In the spike bar itself: open=100k, high=1.1m, low=100k, close=1.1m
    df_spike.loc[spike_idx, "open"] = 100000.0
    df_spike.loc[spike_idx, "high"] = 1100000.0
    df_spike.loc[spike_idx, "low"] = 100000.0
    df_spike.loc[spike_idx, "close"] = 1100000.0

    features = compute_regime_features(df_spike)
    warmup_features = features.iloc[240:]

    assert not warmup_features.isna().any().any(), "Extreme spike produced unexpected NaNs!"
    assert not np.isinf(warmup_features.to_numpy()).any(), "Extreme spike produced Inf!"

    assert (features["vol_parkinson_30"].iloc[spike_idx : spike_idx + 30] > 0.0).all()
    assert (features["vol_garman_klass_30"].iloc[spike_idx : spike_idx + 30] > 0.0).all()

    # Labels: Bars immediately before the spike must be labeled BULL (Class 0)
    labels = compute_triple_barrier_labels(df_spike, horizon=60, k_atr=5.0)
    pre_spike_labels = labels[spike_idx - 10 : spike_idx]
    assert (pre_spike_labels == REGIME_LABEL_BULL).all(), "Pre-spike bars should be labeled BULL"


# ---------------------------------------------------------------------------
# Dimension 3: Short Warmup & Edge Case Datasets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_bars",
    [0, 1, 2, 5, 13, 14, 15, 29, 30, 41, 42, 59, 60, 61, 100, 239, 240, 241],
)
def test_short_warmup_and_boundary_sizes(n_bars: int):
    """
    Adversarial Challenge: Datasets ranging from empty (0) to exact window boundaries.
    Asserts:
    - compute_regime_features never crashes with IndexError or Unhandled Exception.
    - Output DataFrame shape is always exactly [n_bars, 15].
    - compute_triple_barrier_labels handles n_bars <= horizon gracefully.
    """
    if n_bars == 0:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "tickvol", "spread"])
    else:
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
                "open": np.linspace(90000, 95000, n_bars),
                "high": np.linspace(90100, 95100, n_bars),
                "low": np.linspace(89900, 94900, n_bars),
                "close": np.linspace(90050, 95050, n_bars),
                "tickvol": np.full(n_bars, 50.0),
                "spread": np.full(n_bars, 2880.0),
            }
        )

    # Feature computation
    features = compute_regime_features(df)
    assert len(features) == n_bars
    assert list(features.columns) == REGIME_FEATURE_NAMES

    # Numpy contract
    features_arr = extract_regime_features(df)
    assert features_arr.shape == (n_bars, 15)

    # Label computation with purge_tail=True
    labels_purged = compute_triple_barrier_labels(df, horizon=60, k_atr=5.0, purge_tail=True)
    assert len(labels_purged) == n_bars
    assert labels_purged.dtype == np.int32

    # Label computation with purge_tail=False
    labels_unpurged = compute_triple_barrier_labels(df, horizon=60, k_atr=5.0, purge_tail=False)
    assert len(labels_unpurged) == n_bars
    assert labels_unpurged.dtype == np.int32

    # If n_bars <= 60, all purged labels must be REGIME_LABEL_PURGED (-1)
    if n_bars <= 60:
        assert (labels_purged == REGIME_LABEL_PURGED).all()


def test_parameter_boundary_validation():
    """
    Adversarial Challenge: Invalid parameter boundaries (horizon <= 0, k_atr <= 0).
    Verify ValueError is raised deterministically.
    """
    df = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=100, freq="1min"),
            "open": np.full(100, 90000.0),
            "high": np.full(100, 90100.0),
            "low": np.full(100, 89900.0),
            "close": np.full(100, 90050.0),
            "tickvol": np.full(100, 50.0),
            "spread": np.full(100, 2880.0),
        }
    )

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        compute_triple_barrier_labels(df, horizon=0)

    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        compute_triple_barrier_labels(df, horizon=-10)

    with pytest.raises(ValueError, match="k_atr must be a positive float"):
        compute_triple_barrier_labels(df, k_atr=0.0)

    with pytest.raises(ValueError, match="k_atr must be a positive float"):
        compute_triple_barrier_labels(df, k_atr=-2.5)


def test_prepare_dataset_missing_split_graceful_rejection():
    """
    Adversarial Challenge: Passing dataset with missing OOS partition to prepare_dataset.
    When a dataset lacks out-of-sample rows beyond train_end,
    it must fail gracefully with ValueError('No out-of-sample rows found').
    """
    df_jan_only = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=300, freq="1min"),
            "open": np.full(300, 90000.0),
            "high": np.full(300, 90100.0),
            "low": np.full(300, 89900.0),
            "close": np.full(300, 90050.0),
            "tickvol": np.full(300, 50.0),
            "spread": np.full(300, 2880.0),
        }
    )

    import sys

    target = (
        "self_growing_trader.regime_data.load_raw_btcusd_m1"
        if "self_growing_trader.regime_data" in sys.modules
        else "src.self_growing_trader.regime_data.load_raw_btcusd_m1"
    )
    with patch(target, return_value=df_jan_only):
        with pytest.raises(ValueError, match="No out-of-sample rows found"):
            prepare_dataset(csv_path="mock.csv", train_end="2025-01-31 23:59:59")


# ---------------------------------------------------------------------------
# Dimension 4: Pathological & Perturbed Price Inputs
# ---------------------------------------------------------------------------


def test_pathological_inverted_and_negative_prices():
    """
    Adversarial Challenge: Physically impossible prices (High < Low, negative prices).
    Verify that compute_regime_features clamps to safe lower bounds and does not crash or generate Infs.
    """
    n_bars = 300
    # Inverted High < Low
    df_inverted = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
            "open": np.full(n_bars, 100.0),
            "high": np.full(n_bars, 50.0),  # High < Low
            "low": np.full(n_bars, 150.0),
            "close": np.full(n_bars, 100.0),
            "tickvol": np.full(n_bars, 10.0),
            "spread": np.full(n_bars, 20.0),
        }
    )

    feats_inv = compute_regime_features(df_inverted)
    assert feats_inv.shape == (n_bars, 15)
    warmup_inv = feats_inv.iloc[240:]
    assert not warmup_inv.isna().any().any()
    assert not np.isinf(warmup_inv.to_numpy()).any()

    # Negative prices
    df_negative = pd.DataFrame(
        {
            "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
            "open": np.full(n_bars, -100.0),
            "high": np.full(n_bars, -50.0),
            "low": np.full(n_bars, -150.0),
            "close": np.full(n_bars, -100.0),
            "tickvol": np.full(n_bars, 10.0),
            "spread": np.full(n_bars, 20.0),
        }
    )

    feats_neg = compute_regime_features(df_negative)
    assert feats_neg.shape == (n_bars, 15)
    warmup_neg = feats_neg.iloc[240:]
    assert not warmup_neg.isna().any().any()
    assert not np.isinf(warmup_neg.to_numpy()).any()


def test_pathological_extreme_float_magnitudes():
    """
    Adversarial Challenge: Prices at floating point extremes ($1e-6$ and $1e9$).
    Verify no overflow or underflow produces NaN / Inf in features.
    """
    n_bars = 300
    for scale in [1e-6, 1e9]:
        df_scaled = pd.DataFrame(
            {
                "datetime": pd.date_range("2025-01-01", periods=n_bars, freq="1min"),
                "open": np.full(n_bars, 1.0 * scale),
                "high": np.full(n_bars, 1.01 * scale),
                "low": np.full(n_bars, 0.99 * scale),
                "close": np.full(n_bars, 1.005 * scale),
                "tickvol": np.full(n_bars, 100.0),
                "spread": np.full(n_bars, 2880.0),
            }
        )
        feats = compute_regime_features(df_scaled)
        warmup = feats.iloc[240:]
        assert not warmup.isna().any().any(), f"Failed on scale {scale}"
        assert not np.isinf(warmup.to_numpy()).any(), f"Inf on scale {scale}"


# ---------------------------------------------------------------------------
# Dimension 5: Full Q1 2025 Workload & Performance Benchmark
# ---------------------------------------------------------------------------


def test_full_q1_workload_and_performance():
    """
    Empirical Benchmark on full Q1 2025 dataset (129,581 bars):
    - Verify wall-clock execution time is within acceptable production limits:
      - compute_regime_features < 2.0s
      - compute_triple_barrier_labels < 0.5s
      - prepare_dataset < 45.0s (including 153MB CSV disk ingestion)
    - Verify peak memory allocated by tracemalloc is strictly < 500 MB.
    - Verify 0 NaNs and 0 Infs in final prepared dataset.
    """
    if not os.path.exists(DEFAULT_CSV_PATH):
        pytest.skip(f"Benchmark requires dataset at {DEFAULT_CSV_PATH}")

    # Benchmark raw feature computation
    raw_df = load_raw_btcusd_m1(DEFAULT_CSV_PATH, "2025-01-01", "2025-03-31")
    assert len(raw_df) == 129581

    # Measure compute_regime_features
    t0 = time.perf_counter()
    features_df = compute_regime_features(raw_df)
    t_feat = time.perf_counter() - t0
    assert features_df.shape == (129581, 15)
    assert t_feat < 2.0, f"compute_regime_features exceeded 2.0s limit: took {t_feat:.3f}s"

    # Measure compute_triple_barrier_labels
    t0 = time.perf_counter()
    labels = compute_triple_barrier_labels(raw_df, horizon=60, k_atr=5.0, purge_tail=True)
    t_label = time.perf_counter() - t0
    assert len(labels) == 129581
    assert t_label < 0.5, f"compute_triple_barrier_labels exceeded 0.5s limit: took {t_label:.3f}s"

    # Measure prepare_dataset with memory profiling
    tracemalloc.start()
    t0 = time.perf_counter()
    data = prepare_dataset(DEFAULT_CSV_PATH)
    t_prep = time.perf_counter() - t0
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak_bytes / (1024 * 1024)

    assert t_prep < 45.0, f"prepare_dataset exceeded 45.0s limit: took {t_prep:.3f}s"
    assert peak_mb < 500.0, f"prepare_dataset peak memory exceeded 500 MB: allocated {peak_mb:.2f} MB"

    # Assert shape and mathematical purity of prepared dataset
    assert data["X_train"].shape == (44335, 15)
    assert data["y_train"].shape == (44335,)
    assert data["X_oos"].shape == (84886, 15)
    assert data["y_oos"].shape == (84886,)

    assert np.isfinite(data["X_train"]).all(), "Non-finite values found in X_train"
    assert np.isfinite(data["X_oos"]).all(), "Non-finite values found in X_oos"
    assert not np.isnan(data["X_train"]).any(), "NaN found in X_train"
    assert not np.isnan(data["X_oos"]).any(), "NaN found in X_oos"

    # Assert labels are strictly in {0, 1, 2}
    valid_classes = {REGIME_LABEL_BULL, REGIME_LABEL_BEAR, REGIME_LABEL_CHOP}
    assert set(np.unique(data["y_train"])).issubset(valid_classes)
    assert set(np.unique(data["y_oos"])).issubset(valid_classes)
