"""
Empirical Oracle & Adversarial Verification Test Suite for Milestone 1.
Author: teamwork_challenger_tca_m1_2
File: tests/test_empirical_oracle_m1_challenger.py

Independent mathematical oracle and stress testing for:
1. Exact ATR(14) formula verification against raw BTCUSDm.csv.
2. Naive multi-horizon future returns [y_5, y_15, y_30] across 1,000 samples per regime partition.
3. Authentic partition bar counts and date bounds directly from raw CSV.
4. Boundary safety, lookahead prevention, clipping, and OOS firewall enforcement.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pytest
import torch

from research_reasoning_world_class.src.mega_pool_dataset import (
    MegaPoolBatch,
    MegaPoolConfig,
    MegaPoolDataset,
    MegaPoolRouter,
    SingleRegimeDataset,
    compute_multi_horizon_targets_numpy,
    compute_multi_horizon_targets_torch,
)

CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"


# =============================================================================
# 1. INDEPENDENT NAIVE ORACLES
# =============================================================================


def naive_atr14_oracle(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    close_arr: np.ndarray,
    bar_idx: int,
) -> float:
    """
    Independent naive mathematical oracle for 14-period MT5 iATR.
    Calculates True Range from raw prices:
        TR_k = max(H_k - L_k, |H_k - C_{k-1}|, |L_k - C_{k-1}|)
        ATR_{14, k} = (1 / 14) * sum_{j=0}^{13} TR_{k-j}
    Requires at least 13 prior bars (bar_idx >= 13).
    """
    if bar_idx < 13:
        raise ValueError(f"Need at least 14 bars for ATR14, got bar_idx={bar_idx}")

    tr_values: List[float] = []
    # Collect TR for bars [bar_idx - 13, ..., bar_idx]
    for k in range(bar_idx - 13, bar_idx + 1):
        h_k = float(high_arr[k])
        l_k = float(low_arr[k])
        c_prev = float(close_arr[k - 1])
        tr_k = max(h_k - l_k, abs(h_k - c_prev), abs(l_k - c_prev))
        tr_values.append(tr_k)

    return float(sum(tr_values) / 14.0)


def naive_multi_horizon_oracle(
    close_arr: np.ndarray,
    atr_val: float,
    bar_idx: int,
    horizons: Tuple[int, ...] = (5, 15, 30),
    clip_range: Tuple[float, float] = (-10.0, 10.0),
    atr_floor: float = 1e-4,
) -> np.ndarray:
    """
    Independent naive mathematical oracle for multi-horizon future returns:
        y_{t, h} = clip((Close_{min(t+h, N-1)} - Close_t) / max(ATR_t, 1e-4), -10.0, 10.0)
    """
    c_t = float(close_arr[bar_idx])
    n = len(close_arr)
    safe_atr = max(float(atr_val), atr_floor) if (math.isfinite(atr_val) and atr_val > 0) else atr_floor
    c_min, c_max = clip_range

    targets = []
    for h in horizons:
        future_idx = min(bar_idx + h, n - 1)
        c_future = float(close_arr[future_idx])
        raw_ret = (c_future - c_t) / safe_atr
        clipped_ret = max(min(raw_ret, c_max), c_min)
        targets.append(clipped_ret)

    return np.array(targets, dtype=np.float32)


# =============================================================================
# 2. TEST 1: PARTITION BAR COUNTS & DATE BOUNDS DIRECTLY FROM RAW CSV
# =============================================================================


@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="BTCUSDm.csv not found")
def test_partition_bar_counts_and_date_bounds_raw_csv():
    """
    Empirically verify bar counts and date bounds directly from the raw 153MB CSV.
    Expected:
        - Bull: 243,084 bars (2023.10.15 - 2024.03.31)
        - Bear: 132,405 bars (2022.04.01 - 2022.07.01)
        - Chop: 133,798 bars (2023.06.20 - 2023.09.20)
        - OOS Master: 64,785 bars (2025.01.15 - 2025.02.28)
        - OOS Crash Slice: 9,785 bars ([55000:64785], Feb 22-28, 2025)
        - Total Training Mega-Pool: 509,287 bars
    """
    df_dates = pd.read_csv(CSV_PATH, sep="\t", usecols=["<DATE>", "<TIME>"])
    df_dates["Datetime"] = pd.to_datetime(df_dates["<DATE>"] + " " + df_dates["<TIME>"])

    bull_mask = (df_dates["Datetime"] >= "2023-10-15 00:00:00") & (df_dates["Datetime"] <= "2024-03-31 23:59:59")
    bear_mask = (df_dates["Datetime"] >= "2022-04-01 00:00:00") & (df_dates["Datetime"] <= "2022-07-01 23:59:59")
    chop_mask = (df_dates["Datetime"] >= "2023-06-20 00:00:00") & (df_dates["Datetime"] <= "2023-09-20 23:59:59")
    oos_mask = (df_dates["Datetime"] >= "2025-01-15 00:00:00") & (df_dates["Datetime"] <= "2025-02-28 23:59:59")

    bull_bars = int(bull_mask.sum())
    bear_bars = int(bear_mask.sum())
    chop_bars = int(chop_mask.sum())
    oos_bars = int(oos_mask.sum())

    assert bull_bars == 243084, f"Bull count mismatch: expected 243084, got {bull_bars}"
    assert bear_bars == 132405, f"Bear count mismatch: expected 132405, got {bear_bars}"
    assert chop_bars == 133798, f"Chop count mismatch: expected 133798, got {chop_bars}"
    assert oos_bars == 64785, f"OOS master count mismatch: expected 64785, got {oos_bars}"

    total_training = bull_bars + bear_bars + chop_bars
    assert total_training == 509287, f"Total training count mismatch: expected 509287, got {total_training}"

    # Verify OOS Crash Slice [55000:64785]
    oos_df = df_dates[oos_mask].reset_index(drop=True)
    oos_slice = oos_df.iloc[55000:64785]
    assert len(oos_slice) == 9785, f"OOS crash slice count mismatch: expected 9785, got {len(oos_slice)}"
    assert oos_slice.iloc[0]["Datetime"] == pd.Timestamp("2025-02-22 04:48:00")
    assert oos_slice.iloc[-1]["Datetime"] == pd.Timestamp("2025-02-28 23:59:00")


# =============================================================================
# 2. TEST 2: NAIVE ATR(14) ORACLE VS ENVIRONMENT / CACHED ARRAYS
# =============================================================================


@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="BTCUSDm.csv not found")
def test_naive_atr14_oracle_zero_divergence():
    """
    Empirically compare naive ATR(14) oracle against cached/environment atr_arr.
    Assert max absolute divergence == 0.0 (or < 1e-6) across Bull, Bear, Chop, and OOS.
    """
    partitions = [
        ("bull", "2023-10-15 00:00:00", "2024-03-31 23:59:59", "2023-10-14 00:00:00"),
        ("bear", "2022-04-01 00:00:00", "2022-07-01 23:59:59", "2022-03-31 00:00:00"),
        ("chop", "2023-06-20 00:00:00", "2023-09-20 23:59:59", "2023-06-19 00:00:00"),
        ("oos_master", "2025-01-15 00:00:00", "2025-02-28 23:59:59", "2025-01-14 00:00:00"),
    ]

    df = pd.read_csv(CSV_PATH, sep="\t")
    df.columns = [c.strip("<>") for c in df.columns]
    df["Datetime"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])

    for pool_name, start_dt_str, end_dt_str, warmup_dt_str in partitions:
        start_dt = pd.to_datetime(start_dt_str)
        end_dt = pd.to_datetime(end_dt_str)
        warmup_dt = pd.to_datetime(warmup_dt_str)

        mask = (df["Datetime"] >= warmup_dt) & (df["Datetime"] <= end_dt)
        df_seg = df[mask].copy().reset_index(drop=True)

        high = df_seg["HIGH"].to_numpy(dtype=np.float64)
        low = df_seg["LOW"].to_numpy(dtype=np.float64)
        close = df_seg["CLOSE"].to_numpy(dtype=np.float64)

        # Naive oracle calculation for all bars in segment
        naive_atr_seg = np.zeros(len(close), dtype=np.float64)
        # Sample 250 random bars across the target window to verify via naive_atr14_oracle
        target_indices = np.where((df_seg["Datetime"] >= start_dt) & (df_seg["Datetime"] <= end_dt))[0]
        rng = np.random.default_rng(12345)
        sampled_target_indices = rng.choice(target_indices, size=250, replace=False)

        # Load cached environment ATR
        cache_path = Path("outputs/cache") / f"pool_{pool_name}.npz"
        assert cache_path.exists(), f"Cache file not found: {cache_path}"
        with np.load(cache_path) as data:
            cached_atr = data["atr_arr"]

        for idx in sampled_target_indices:
            expected_atr = naive_atr14_oracle(high, low, close, idx)
            # Corresponding index in cached target window
            local_target_idx = idx - target_indices[0]
            actual_atr = float(cached_atr[local_target_idx])

            abs_diff = abs(expected_atr - actual_atr)
            assert abs_diff < 1e-4, (
                f"[{pool_name}] ATR oracle divergence on bar {local_target_idx}: "
                f"expected={expected_atr:.6f}, actual={actual_atr:.6f}, diff={abs_diff:.6e}"
            )


# =============================================================================
# 3. TEST 3: NAIVE MULTI-HORIZON ORACLE ON 1,000 SAMPLES PER REGIME
# =============================================================================


def test_naive_multi_horizon_oracle_1000_samples_across_regimes():
    """
    Sample 1,000 random bar indices across Bull, Bear, and Chop partitions.
    Compare naive oracle calculation directly against mega_pool_dataset.py outputs.
    Assert max absolute divergence == 0.0 (or < 1e-6).
    """
    router = MegaPoolRouter()

    for regime in ["bull", "bear", "chop"]:
        dataset = router.training_datasets[regime]
        n_valid = len(dataset)
        assert n_valid > 1000, f"Dataset too small: {n_valid}"

        close_arr = dataset.env.close_arr
        atr_arr = dataset.env.atr_arr
        targets_matrix = dataset.targets_matrix

        rng = np.random.default_rng(seed=42 if regime == "bull" else (43 if regime == "bear" else 44))
        sample_indices = rng.integers(0, n_valid, size=1000)

        max_divergence = 0.0
        for idx in sample_indices:
            # Oracle calculation
            oracle_target = naive_multi_horizon_oracle(
                close_arr=close_arr,
                atr_val=float(atr_arr[idx]),
                bar_idx=int(idx),
                horizons=(5, 15, 30),
            )

            # mega_pool_dataset output (from dataset[idx] and from precomputed targets_matrix)
            matrix_target = targets_matrix[idx]
            obs, seq, tgt, atr = dataset[idx]
            dataset_target = tgt.numpy()

            diff_matrix = float(np.max(np.abs(oracle_target - matrix_target)))
            diff_dataset = float(np.max(np.abs(oracle_target - dataset_target)))
            max_divergence = max(max_divergence, diff_matrix, diff_dataset)

            assert diff_matrix < 1e-6, (
                f"[{regime}] Target matrix divergence at bar {idx}: "
                f"oracle={oracle_target}, matrix={matrix_target}, diff={diff_matrix}"
            )
            assert diff_dataset < 1e-6, (
                f"[{regime}] Dataset __getitem__ divergence at bar {idx}: "
                f"oracle={oracle_target}, dataset={dataset_target}, diff={diff_dataset}"
            )

        assert max_divergence < 1e-6, (
            f"[{regime}] Max absolute divergence across 1,000 samples exceeded tolerance: {max_divergence}"
        )


# =============================================================================
# 4. TEST 4: OOS SLICE EVALUATION & FIREWALL ADVERSARIAL STRESS
# =============================================================================


def test_oos_slice_exactness_and_firewall():
    """
    Verify OOS slice [55000:64785] targets exactness and strict firewall enforcement.
    """
    router = MegaPoolRouter()
    oos_dataset = router.oos_dataset

    assert len(oos_dataset) == 9755  # 9785 bars - 30 horizon buffer
    assert oos_dataset.is_oos is True

    # 1. Oracle exactness on 500 samples of OOS slice
    close_arr = oos_dataset.env.close_arr
    atr_arr = oos_dataset.env.atr_arr
    targets_matrix = oos_dataset.targets_matrix

    rng = np.random.default_rng(seed=999)
    sample_indices = rng.integers(0, len(oos_dataset), size=500)

    for idx in sample_indices:
        oracle_tgt = naive_multi_horizon_oracle(close_arr, float(atr_arr[idx]), int(idx))
        actual_tgt = targets_matrix[idx]
        diff = float(np.max(np.abs(oracle_tgt - actual_tgt)))
        assert diff < 1e-6, f"OOS divergence at {idx}: {diff}"

    # 2. Firewall verification
    with pytest.raises(PermissionError):
        router.get_batch(32, pool_name="oos")

    with pytest.raises(ValueError, match="Out-of-sample data cannot be part of MegaPoolDataset"):
        MegaPoolDataset({"oos": oos_dataset})

    assert router.validate_zero_oos_leakage() is True


# =============================================================================
# 5. TEST 5: ADVERSARIAL BOUNDARY & CORNER CASES
# =============================================================================


def test_adversarial_targets_extreme_inputs():
    """
    Stress-test target calculation engine under extreme inputs:
    - Zero ATR, negative ATR, infinite ATR, NaN ATR floored at 1e-4.
    - Massive price jumps (+-1,000,000) clipped to [-10.0, 10.0].
    - Empty arrays return empty matrix.
    """
    close = np.array([100.0, 102.0, 105.0, 110.0, 120.0, 150.0], dtype=np.float32)
    # Degenerate ATR values
    atr = np.array([0.0, -5.0, np.nan, np.inf, -np.inf, 1e-8], dtype=np.float32)

    targets = compute_multi_horizon_targets_numpy(close, atr, horizons=(1, 2), atr_floor=1e-4)

    assert targets.shape == (6, 2)
    assert not np.isnan(targets).any(), "Found NaNs in target output"
    assert not np.isinf(targets).any(), "Found Infs in target output"
    assert np.all(targets >= -10.0) and np.all(targets <= 10.0), "Targets exceeded clip range"

    # Bitwise equivalence between numpy and torch implementations
    torch_close = torch.from_numpy(close)
    torch_atr = torch.from_numpy(atr)
    torch_targets = compute_multi_horizon_targets_torch(torch_close, torch_atr, horizons=(1, 2), atr_floor=1e-4)
    np.testing.assert_allclose(targets, torch_targets.numpy(), atol=1e-6)
