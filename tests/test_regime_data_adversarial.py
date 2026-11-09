"""
tests/test_regime_data_adversarial.py

Adversarial Stress Test Suite for Milestone 1:
BTCUSD M1 Data Pipeline & Objective Ground Truth Labeling (src/self_growing_trader/regime_data.py).

Rigorous empirical challenge vectors:
1. Deep Lookahead Injection Attack:
   - Random perturbation of future bars t+1..T (noise, price inversion, volume explosion)
   - Extreme price spike / crash injection at bar t+1
   - Zero-out attack on future bars
   - Slicing truncation invariance: compute_regime_features(df.iloc[:t+1]) == compute_regime_features(df).iloc[:t+1]
   - Machine epsilon tolerance check (atol=1e-12) across all 15 dimensions.
2. Embargo Boundary Leakage Test:
   - January 31st 23:00-23:59 embargo purge window verification.
   - February/March adversarial mutation attack: mutating February data MUST NOT alter In-Sample y_train.
   - Forward barrier horizon reachability audit: bar 22:59 forward window stays strictly <= 23:59.
3. Scaler Isolation Test:
   - Mathematical verification of RobustScaler and StandardScaler fitting strictly on Jan In-Sample.
   - OOS feature perturbation attack: mutating OOS features must cause ZERO drift in scaler parameters.
   - Transformed distribution separation test.
4. Pathological Market Edge Cases:
   - Flatline market (zero volatility, zero ATR, zero volume) -> division by zero immunity.
   - Flash crash (90% sudden collapse) -> numerical stability.
   - Simultaneous barrier touch tiebreaker mechanics.
   - Zero tick volume resilience.
   - NaN / Inf immunity check on entire Q1 2025 dataset.
"""

from __future__ import annotations

import os
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


@pytest.fixture(scope="module")
def real_csv_available() -> bool:
    return os.path.exists(DEFAULT_CSV_PATH)


@pytest.fixture(scope="module")
def q1_df(real_csv_available) -> pd.DataFrame:
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")
    return load_raw_btcusd_m1(DEFAULT_CSV_PATH, "2025-01-01", "2025-03-31")


# ==============================================================================
# 1. DEEP LOOKAHEAD INJECTION ATTACK
# ==============================================================================


@pytest.mark.parametrize("target_bar", [250, 350, 500, 750, 1000])
def test_deep_lookahead_random_perturbation(q1_df: pd.DataFrame, target_bar: int):
    """
    ATTACK 1: Random Future Scramble.
    Mutates future bars t+1..T with massive uniform random noise.
    Asserts 100% invariance of feature vector x_s for all s <= target_bar at atol=1e-12.
    """
    total_len = target_bar + 400
    df_base = q1_df.iloc[:total_len].copy().reset_index(drop=True)
    feats_base = compute_regime_features(df_base)

    rng = np.random.default_rng(1337 + target_bar)
    df_mut = df_base.copy()
    n_future = len(df_mut) - (target_bar + 1)

    # Scramble future prices, volumes, spreads
    price_mult = rng.uniform(0.5, 2.5, n_future)
    df_mut.loc[target_bar + 1 :, "close"] *= price_mult
    df_mut.loc[target_bar + 1 :, "open"] *= price_mult
    df_mut.loc[target_bar + 1 :, "high"] = df_mut.loc[target_bar + 1 :, "close"] * rng.uniform(1.01, 1.10, n_future)
    df_mut.loc[target_bar + 1 :, "low"] = df_mut.loc[target_bar + 1 :, "close"] * rng.uniform(0.90, 0.99, n_future)
    df_mut.loc[target_bar + 1 :, "tickvol"] = rng.integers(1, 100000, n_future).astype(float)
    df_mut.loc[target_bar + 1 :, "spread"] = rng.integers(100, 50000, n_future).astype(float)

    feats_mut = compute_regime_features(df_mut)

    for col in REGIME_FEATURE_NAMES:
        b_vals = feats_base.loc[:target_bar, col].to_numpy()
        m_vals = feats_mut.loc[:target_bar, col].to_numpy()
        valid = ~np.isnan(b_vals)
        assert np.all(valid == ~np.isnan(m_vals)), f"Warmup NaN mismatch in {col} at bar {target_bar}"
        np.testing.assert_allclose(
            b_vals[valid],
            m_vals[valid],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Lookahead leak detected in {col} for target_bar={target_bar}!",
        )


def test_deep_lookahead_extreme_spikes_and_crashes(q1_df: pd.DataFrame):
    """
    ATTACK 2: Extreme Spikes & Crashes at bar t+1.
    Injects 10,000x price spike and 99.99% flash crash at bars t+1 and t+2.
    Asserts strict invariance at bar t down to atol=1e-12.
    """
    target_bar = 400
    df_base = q1_df.iloc[:target_bar + 200].copy().reset_index(drop=True)
    feats_base = compute_regime_features(df_base)

    df_spike = df_base.copy()
    # Bar t+1: 10,000x price spike
    df_spike.loc[target_bar + 1, ["open", "high", "low", "close"]] = 10000.0 * df_base.loc[target_bar + 1, "close"]
    df_spike.loc[target_bar + 1, "tickvol"] = 99999999.0
    df_spike.loc[target_bar + 1, "spread"] = 500000.0

    # Bar t+2: 99.99% flash crash
    df_spike.loc[target_bar + 2, ["open", "high", "low", "close"]] = 0.0001 * df_base.loc[target_bar + 2, "close"]

    feats_spike = compute_regime_features(df_spike)

    for col in REGIME_FEATURE_NAMES:
        b_vals = feats_base.loc[:target_bar, col].to_numpy()
        s_vals = feats_spike.loc[:target_bar, col].to_numpy()
        valid = ~np.isnan(b_vals)
        np.testing.assert_allclose(
            b_vals[valid],
            s_vals[valid],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Spike lookahead leak in {col} at or before bar {target_bar}!",
        )


def test_deep_lookahead_zero_out_future(q1_df: pd.DataFrame):
    """
    ATTACK 3: Zero-Out Future Bars.
    Replaces all future bars t+1..T with safe floor values (1e-6).
    Asserts bar t features are 100% preserved.
    """
    target_bar = 300
    df_base = q1_df.iloc[:target_bar + 100].copy().reset_index(drop=True)
    feats_base = compute_regime_features(df_base)

    df_zero = df_base.copy()
    df_zero.loc[target_bar + 1 :, ["open", "high", "low", "close"]] = 1e-6
    df_zero.loc[target_bar + 1 :, "tickvol"] = 0.0
    df_zero.loc[target_bar + 1 :, "spread"] = 0.0

    feats_zero = compute_regime_features(df_zero)

    for col in REGIME_FEATURE_NAMES:
        b_vals = feats_base.loc[:target_bar, col].to_numpy()
        z_vals = feats_zero.loc[:target_bar, col].to_numpy()
        valid = ~np.isnan(b_vals)
        np.testing.assert_allclose(
            b_vals[valid],
            z_vals[valid],
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Zero-out leak in {col} at bar {target_bar}!",
        )


@pytest.mark.parametrize("slice_len", [250, 300, 450, 600, 1000])
def test_deep_lookahead_slice_truncation_invariance(q1_df: pd.DataFrame, slice_len: int):
    """
    ATTACK 4: Slicing Truncation Invariance (Gold Standard of Causality).
    Features computed on df.iloc[:slice_len] at the last row (slice_len - 1)
    MUST be bit-for-bit identical to features computed on df.iloc[:2000] at index (slice_len - 1).
    Future bars literally do not exist in the sliced dataframe!
    """
    df_full = q1_df.iloc[:2000].copy().reset_index(drop=True)
    feats_full = compute_regime_features(df_full)

    df_slice = q1_df.iloc[:slice_len].copy().reset_index(drop=True)
    feats_slice = compute_regime_features(df_slice)

    # Check last row of sliced features vs corresponding row in full features
    target_idx = slice_len - 1
    full_row = feats_full.iloc[target_idx].to_numpy(dtype=np.float64)
    slice_row = feats_slice.iloc[-1].to_numpy(dtype=np.float64)

    assert not np.isnan(full_row).any(), f"Unexpected NaN in full features at index {target_idx}"
    assert not np.isnan(slice_row).any(), f"Unexpected NaN in slice features at index -1"

    np.testing.assert_allclose(
        full_row,
        slice_row,
        rtol=1e-12,
        atol=1e-12,
        err_msg=f"Slice truncation invariance violated at index {target_idx}!",
    )


# ==============================================================================
# 2. EMBARGO BOUNDARY LEAKAGE TEST
# ==============================================================================


def test_embargo_boundary_leakage_feb_perturbation(real_csv_available):
    """
    ATTACK 5: February Data Perturbation Adversarial Attack.
    Drastically modifies February and March prices, injecting huge bull trends and crashes.
    Verifies that In-Sample y_train and X_train remain 100% INVARIANT (zero label bleed across split).
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    baseline_data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)
    y_train_baseline = baseline_data["y_train"].copy()
    X_train_baseline = baseline_data["X_train"].copy()

    # Load raw data and alter all bars in Feb and Mar
    df_raw = load_raw_btcusd_m1(DEFAULT_CSV_PATH, "2025-01-01", "2025-03-31")
    feb_cutoff = pd.Timestamp("2025-01-31 23:59:59")
    feb_mask = df_raw["datetime"] > feb_cutoff

    # Radical perturbation on February data
    df_mut = df_raw.copy()
    df_mut.loc[feb_mask, "close"] *= 5.0
    df_mut.loc[feb_mask, "high"] *= 6.0
    df_mut.loc[feb_mask, "low"] *= 0.1
    df_mut.loc[feb_mask, "open"] *= 4.0
    df_mut.loc[feb_mask, "tickvol"] += 500000

    # Patch load_raw_btcusd_m1 inside prepare_dataset to return the mutated df
    with patch("self_growing_trader.regime_data.load_raw_btcusd_m1", return_value=df_mut) as _:
        mut_data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)

    # In-Sample labels y_train MUST be bit-for-bit identical!
    np.testing.assert_array_equal(
        mut_data["y_train"],
        y_train_baseline,
        err_msg="EMBARGO LEAKAGE DETECTED! February mutations altered In-Sample training labels!",
    )

    # In-Sample features X_train MUST be identical down to atol=1e-12!
    np.testing.assert_allclose(
        mut_data["X_train"],
        X_train_baseline,
        rtol=1e-12,
        atol=1e-12,
        err_msg="EMBARGO LEAKAGE DETECTED! February mutations altered In-Sample training features!",
    )


def test_embargo_window_exact_cutoff_and_reachability(real_csv_available):
    """
    ATTACK 6: Embargo Cutoff & Forward Window Reachability Audit.
    Mathematically verifies:
    1. Exactly 60 bars of January (23:00 to 23:59) are purged from y_train.
    2. The last admitted bar in train_timestamps is 2025-01-31 22:59:00.
    3. The 60-bar forward horizon for 22:59:00 spans 23:00:00 to 23:59:00, terminating strictly inside Jan!
    4. Admitting bar 23:00:00 WOULD have evaluated bar 2025-02-01 00:00:00 (Feb 1st), proving the 60-bar purge is tight.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, horizon=60)
    train_ts = data["train_timestamps"]

    max_train_ts = train_ts.max()
    assert max_train_ts == pd.Timestamp("2025-01-31 22:59:00"), (
        f"Expected max train timestamp 2025-01-31 22:59:00, got {max_train_ts}"
    )

    # Load raw Jan bars
    df_jan, _ = load_and_preprocess_q1_data(DEFAULT_CSV_PATH)
    last_jan_bar_ts = df_jan["datetime"].iloc[-1]
    assert last_jan_bar_ts == pd.Timestamp("2025-01-31 23:59:00")

    # Trace horizon for bar 22:59:00
    idx_2259 = np.where(df_jan["datetime"] == pd.Timestamp("2025-01-31 22:59:00"))[0][0]
    horizon_bars = df_jan.iloc[idx_2259 + 1 : idx_2259 + 1 + 60]

    assert len(horizon_bars) == 60
    assert horizon_bars["datetime"].iloc[0] == pd.Timestamp("2025-01-31 23:00:00")
    assert horizon_bars["datetime"].iloc[-1] == pd.Timestamp("2025-01-31 23:59:00"), (
        "Forward horizon of last admitted bar must terminate at Jan 31 23:59:00"
    )

    # Trace hypothetical horizon for bar 23:00:00
    idx_2300 = idx_2259 + 1
    # Horizon 60 bars from 23:00:00 reaches index idx_2300 + 60 = idx_2259 + 61, which is len(df_jan)!
    assert idx_2300 + 60 == len(df_jan), "Bar 23:00:00 would reach index len(df_jan) (Feb 1st 00:00)!"


def test_embargo_feb_first_hour_shock_injection(real_csv_available):
    """
    ATTACK 7: Micro-targeted Shock Injection at Feb 1st 00:00 - 00:59.
    Injects violent moves into the very first hour of February.
    Asserts zero changes in January labels.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    baseline_data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)
    y_train_baseline = baseline_data["y_train"]

    df_raw = load_raw_btcusd_m1(DEFAULT_CSV_PATH, "2025-01-01", "2025-03-31")
    first_hour_mask = (df_raw["datetime"] >= "2025-02-01 00:00:00") & (
        df_raw["datetime"] <= "2025-02-01 00:59:00"
    )

    df_mut = df_raw.copy()
    # Massive 50% crash in first hour of Feb
    df_mut.loc[first_hour_mask, "close"] *= 0.50
    df_mut.loc[first_hour_mask, "low"] *= 0.40

    with patch("self_growing_trader.regime_data.load_raw_btcusd_m1", return_value=df_mut):
        mut_data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)

    np.testing.assert_array_equal(
        mut_data["y_train"],
        y_train_baseline,
        err_msg="Leakage from Feb 1st 00:00..00:59 into Jan y_train!",
    )


# ==============================================================================
# 3. SCALER ISOLATION TEST
# ==============================================================================


def test_scaler_isolation_robust_mathematical_proof(real_csv_available):
    """
    ATTACK 8: Exact Mathematical Verification of RobustScaler.
    RobustScaler center_ must equal np.median(X_train_raw) and scale_ must equal IQR.
    OOS features must have no influence on center_ or scale_.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="robust")
    scaler = data["scaler"]
    X_train_raw = data["X_train_raw"]

    assert isinstance(scaler, RobustScaler)

    # 1. Verify center_ is strictly the median of X_train_raw
    expected_median = np.median(X_train_raw, axis=0)
    np.testing.assert_allclose(scaler.center_, expected_median, rtol=1e-12, atol=1e-12)

    # 2. Verify scale_ is strictly IQR (q75 - q25) of X_train_raw
    q25 = np.percentile(X_train_raw, 25, axis=0)
    q75 = np.percentile(X_train_raw, 75, axis=0)
    expected_scale = q75 - q25
    expected_scale[expected_scale == 0.0] = 1.0  # Sklearn RobustScaler convention
    np.testing.assert_allclose(scaler.scale_, expected_scale, rtol=1e-12, atol=1e-12)

    # 3. Verify manual transform matches X_train
    manual_scaled = (X_train_raw - scaler.center_) / scaler.scale_
    np.testing.assert_allclose(data["X_train"], manual_scaled, rtol=1e-12, atol=1e-12)


def test_scaler_isolation_standard_mathematical_proof(real_csv_available):
    """
    ATTACK 9: Exact Mathematical Verification of StandardScaler.
    StandardScaler mean_ must equal np.mean(X_train_raw) and var_ must equal np.var(X_train_raw).
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="standard")
    scaler = data["scaler"]
    X_train_raw = data["X_train_raw"]

    assert isinstance(scaler, StandardScaler)

    expected_mean = np.mean(X_train_raw, axis=0)
    expected_var = np.var(X_train_raw, axis=0)

    np.testing.assert_allclose(scaler.mean_, expected_mean, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, expected_var, rtol=1e-12, atol=1e-12)


def test_scaler_oos_perturbation_invariance(real_csv_available):
    """
    ATTACK 10: OOS Feature Perturbation vs Scaler Invariance.
    Mutates raw OOS data by multiplying features by 100x.
    Verifies that scaler.center_, scaler.scale_, and X_train are 100% BIT-FOR-BIT INVARIANT.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    baseline_data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="robust")
    base_center = baseline_data["scaler"].center_.copy()
    base_scale = baseline_data["scaler"].scale_.copy()
    base_X_train = baseline_data["X_train"].copy()

    df_raw = load_raw_btcusd_m1(DEFAULT_CSV_PATH, "2025-01-01", "2025-03-31")
    feb_mask = df_raw["datetime"] > pd.Timestamp("2025-01-31 23:59:59")

    df_mut = df_raw.copy()
    # Wild multiplication of OOS prices and volume
    df_mut.loc[feb_mask, "close"] *= 10.0
    df_mut.loc[feb_mask, "high"] *= 10.0
    df_mut.loc[feb_mask, "low"] *= 10.0
    df_mut.loc[feb_mask, "open"] *= 10.0
    df_mut.loc[feb_mask, "tickvol"] *= 100.0

    with patch("self_growing_trader.regime_data.load_raw_btcusd_m1", return_value=df_mut):
        mut_data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="robust")

    mut_scaler = mut_data["scaler"]
    np.testing.assert_allclose(mut_scaler.center_, base_center, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mut_scaler.scale_, base_scale, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(mut_data["X_train"], base_X_train, rtol=1e-12, atol=1e-12)


def test_scaler_oos_distribution_shift_preservation(real_csv_available):
    """
    ATTACK 11: Distribution Shift Preservation.
    Asserts that X_oos_scaled does NOT have median == 0 or mean == 0,
    proving that it was normalized by Train statistics and not re-centered on itself.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH, scaler_type="robust")
    X_train_scaled = data["X_train"]
    X_oos_scaled = data["X_oos"]

    # In-Sample median should be ~0.0 for RobustScaler
    train_medians = np.median(X_train_scaled, axis=0)
    np.testing.assert_allclose(train_medians, np.zeros(15), atol=1e-12)

    # OOS median should deviate from 0.0 (natural distribution shift)
    oos_medians = np.median(X_oos_scaled, axis=0)
    assert not np.allclose(oos_medians, np.zeros(15), atol=1e-3), (
        "OOS medians are exactly 0.0, indicating scaler might have erroneously re-fit on OOS!"
    )


# ==============================================================================
# 4. PATHOLOGICAL MARKET EDGE CASES & STRESS HARNESS
# ==============================================================================


def test_pathological_flatline_market():
    """
    ATTACK 12: Completely Flat Market & Zero Volatility Edge Case.
    Case A: Flat close with natural bid-ask spread / micro-volatility -> produces 100% CHOP (2).
    Case B: Mathematically degenerate zero-volatility market (high=low=close=open, TR=0):
            Demonstrates empirical edge case: without safe_atr floor, k_atr * atr_14 collapses to 0.0,
            causing immediate simultaneous barrier touch on bar t+1, where tiebreaker defaults to BULL.
    """
    n = 300
    # Case A: Micro-volatility / tight range around flat price (typical sideway chop)
    df_chop = pd.DataFrame({
        "open": [50000.0] * n,
        "high": [50005.0] * n,
        "low": [49995.0] * n,
        "close": [50000.0] * n,
        "tickvol": [100.0] * n,
        "spread": [2000.0] * n,
    })

    feats_chop = compute_regime_features(df_chop)
    assert isinstance(feats_chop, pd.DataFrame)
    assert feats_chop.shape == (n, 15)

    # After warmup (index >= 240), all features must be strictly finite (no NaN, no Inf)
    valid_feats = feats_chop.iloc[240:].to_numpy()
    assert not np.isnan(valid_feats).any(), "NaN found in flat market features"
    assert not np.isinf(valid_feats).any(), "Inf found in flat market features"

    # Micro-volatility market stays within barriers -> produces 100% CHOP (2)
    labels_chop = compute_triple_barrier_labels(df_chop, horizon=60, k_atr=5.0, purge_tail=True)
    valid_chop = labels_chop[labels_chop != REGIME_LABEL_PURGED]
    assert len(valid_chop) > 0
    assert np.all(valid_chop == REGIME_LABEL_CHOP), "Micro-volatility market must produce 100% CHOP labels"

    # Case B: Absolute degenerate zero TR market (high == low == close == open)
    df_zero_tr = pd.DataFrame({
        "open": [50000.0] * n,
        "high": [50000.0] * n,
        "low": [50000.0] * n,
        "close": [50000.0] * n,
        "tickvol": [100.0] * n,
        "spread": [2000.0] * n,
    })
    labels_zero_tr = compute_triple_barrier_labels(df_zero_tr, horizon=60, k_atr=5.0, purge_tail=True)
    valid_zero_tr = labels_zero_tr[labels_zero_tr != REGIME_LABEL_PURGED]
    # Edge case documentation: because k_atr*0 = 0, upper == lower barrier, tiebreaker c_future >= c_origin yields 0 (BULL)
    assert np.all(valid_zero_tr == REGIME_LABEL_BULL)


def test_pathological_zero_volume_market():
    """
    ATTACK 13: Zero Tick Volume Market.
    Tick volume is 0.0 for all bars.
    Verifies rel_tickvol_20 and tickvol_flow_15 handle zero volume gracefully without div/0.
    """
    n = 300
    rng = np.random.default_rng(42)
    close = 60000.0 + np.cumsum(rng.standard_normal(n) * 10.0)
    df_novol = pd.DataFrame({
        "open": close - 2.0,
        "high": close + 5.0,
        "low": close - 5.0,
        "close": close,
        "tickvol": [0.0] * n,
        "spread": [2500.0] * n,
    })

    feats = compute_regime_features(df_novol)
    valid_feats = feats.iloc[240:].to_numpy()
    assert not np.isnan(valid_feats).any(), "NaN found in zero volume features"
    assert not np.isinf(valid_feats).any(), "Inf found in zero volume features"

    # rel_tickvol_20 must be 0.0 (0 / max(0, 1e-8))
    np.testing.assert_allclose(feats.iloc[240:]["rel_tickvol_20"], 0.0, atol=1e-12)
    # tickvol_flow_15 must be 0.0
    np.testing.assert_allclose(feats.iloc[240:]["tickvol_flow_15"], 0.0, atol=1e-12)


def test_pathological_simultaneous_barrier_tiebreaker():
    """
    ATTACK 14: Simultaneous Upper & Lower Barrier Touch on Same Bar.
    High and Low explode on bar t+5 exceeding both upper and lower barriers.
    Tests tiebreaker:
    - If future close >= origin close -> BULL (0)
    - If future close < origin close -> BEAR (1)
    """
    h = 20
    k = 2.0
    n = 60

    close = np.full(n, 1000.0)
    high = np.full(n, 1005.0)
    low = np.full(n, 995.0)
    open_ = np.full(n, 1000.0)
    # ATR14 = 10.0, k*ATR = 20.0
    # Up barrier = 1020.0, Down barrier = 980.0

    # Case A: Bar 25 simultaneous touch with close >= 1000.0
    high[25] = 1050.0  # Hits Up
    low[25] = 950.0    # Hits Down
    close[25] = 1005.0 # Close > origin close (1000.0) -> Expect BULL

    df_a = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    labels_a = compute_triple_barrier_labels(df_a, horizon=h, k_atr=k, purge_tail=True)
    # Bar 20 has horizon [21..40], touches both at bar 25 -> BULL
    assert labels_a[20] == REGIME_LABEL_BULL, f"Expected BULL on tie with higher close, got {labels_a[20]}"

    # Case B: Bar 25 simultaneous touch with close < 1000.0
    close[25] = 990.0  # Close < origin close (1000.0) -> Expect BEAR
    df_b = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})
    labels_b = compute_triple_barrier_labels(df_b, horizon=h, k_atr=k, purge_tail=True)
    assert labels_b[20] == REGIME_LABEL_BEAR, f"Expected BEAR on tie with lower close, got {labels_b[20]}"


def test_pathological_flash_crash():
    """
    ATTACK 15: Violent 90% Flash Crash across 3 bars post-warmup.
    Price plunges from 100,000 to 10,000 at index 350 (post 240-bar warmup).
    Verifies features remain strictly non-NaN, non-Inf, and Triple Barrier identifies BEAR.
    """
    n = 600
    close = np.full(n, 100000.0)
    high = np.full(n, 100100.0)
    low = np.full(n, 99900.0)
    open_ = np.full(n, 100000.0)

    # Flash crash at bars 350..352
    close[350:353] = [50000.0, 20000.0, 10000.0]
    high[350:353] = [100000.0, 50000.0, 20000.0]
    low[350:353] = [50000.0, 20000.0, 10000.0]
    open_[350:353] = [100000.0, 50000.0, 20000.0]

    close[353:] = 10000.0
    high[353:] = 10100.0
    low[353:] = 9900.0
    open_[353:] = 10000.0

    df_crash = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "tickvol": [500.0] * n, "spread": [3000.0] * n,
    })

    feats = compute_regime_features(df_crash)
    # Check bars post-warmup (index 350..360)
    crash_feats = feats.iloc[350:360].to_numpy()
    assert not np.isnan(crash_feats).any(), "NaN found in flash crash features post-warmup"
    assert not np.isinf(crash_feats).any(), "Inf found in flash crash features post-warmup"

    # Triple barrier for bar 345 (starts before crash, horizon includes crash) must be BEAR
    labels = compute_triple_barrier_labels(df_crash, horizon=60, k_atr=5.0, purge_tail=True)
    assert labels[345] == REGIME_LABEL_BEAR, f"Flash crash must yield BEAR label, got {labels[345]}"


def test_nan_inf_immunity_full_q1_dataset(real_csv_available):
    """
    ATTACK 16: Zero Tolerance for NaNs or Infs across entire prepared Q1 dataset.
    Validates all 44,335 training rows and 84,886 OOS rows.
    """
    if not real_csv_available:
        pytest.skip(f"Dataset not found at {DEFAULT_CSV_PATH}")

    data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)

    for split in ["X_train", "X_oos", "y_train", "y_oos"]:
        arr = data[split]
        assert not np.isnan(arr).any(), f"NaN detected in {split}!"
        assert not np.isinf(arr).any(), f"Inf detected in {split}!"

    # Validate that labels are strictly in {0, 1, 2}
    assert set(np.unique(data["y_train"])) == {0, 1, 2}
    assert set(np.unique(data["y_oos"])) == {0, 1, 2}
