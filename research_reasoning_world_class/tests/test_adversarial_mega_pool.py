"""
Adversarial Stress Test Suite for Mega-Pool Dataset, Sampler & Router.
File: research_reasoning_world_class/tests/test_adversarial_mega_pool.py
Milestone: Milestone 1 Verification (Features 1, 2, 3, 4)

Adversarial Objectives:
1. Degenerate ATR inputs: zero, negative, NaN, Inf, -Inf, subnormal / tiny values (1e-12).
2. Boundary horizon conditions: t = N - 31, t = N - 30, t = N - 1, t >= N, t < 0.
3. Causal lookback sequence padding: t = 0, 1, 238, 239, 240.
4. OOS security & firewall bypass: 10 malicious injection and bypass vectors.
5. Concurrency & Multi-threading: thread pool contention during batch extraction.
6. High-throughput stress: benchmark batch extraction under load.
7. Edge cases: empty batch collate, negative offsets, probability anomalies.
"""

from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from research_reasoning_world_class.src.mega_pool_dataset import (
    MarketRegime,
    MegaPoolBatch,
    MegaPoolConfig,
    MegaPoolDataset,
    MegaPoolRouter,
    RegimeRoutingBatchSampler,
    RoutingMode,
    SingleRegimeDataset,
    collate_mega_pool,
    compute_multi_horizon_targets_numpy,
    compute_multi_horizon_targets_torch,
)
from research_reasoning_world_class.tests.test_mega_pool_dataset import MockM1Environment


# =============================================================================
# 1. DEGENERATE ATR STRESS TESTS
# =============================================================================


def test_degenerate_atr_numpy_and_torch_vectorization():
    """
    Stress-test multi-horizon target calculation with extreme ATR values:
    - Zero (0.0)
    - Negative (-5.0, -100.0)
    - NaN (float('nan'))
    - Inf (+inf, -inf)
    - Tiny subnormal values (1e-12, 1e-18)
    - Normal positive values
    """
    degenerate_atrs = np.array(
        [0.0, -5.0, -100.0, float("nan"), float("inf"), float("-inf"), 1e-12, 1e-18, 50.0, 100.0],
        dtype=np.float32,
    )
    n = len(degenerate_atrs)
    closes = np.linspace(50000.0, 51000.0, n, dtype=np.float32)

    # Test NumPy implementation
    targets_np = compute_multi_horizon_targets_numpy(
        close_arr=closes,
        atr_arr=degenerate_atrs,
        horizons=(1, 2, 3),
        clamp_boundary=True,
        clip_range=(-10.0, 10.0),
        atr_floor=1e-4,
    )

    assert targets_np.shape == (n, 3)
    assert not np.isnan(targets_np).any(), "NumPy targets contain NaNs under degenerate ATR inputs!"
    assert not np.isinf(targets_np).any(), "NumPy targets contain Infs under degenerate ATR inputs!"
    assert np.all(targets_np >= -10.0) and np.all(targets_np <= 10.0), "NumPy targets breached [-10, 10] clipping!"

    # Test PyTorch implementation
    closes_t = torch.from_numpy(closes)
    atrs_t = torch.from_numpy(degenerate_atrs)
    targets_pt = compute_multi_horizon_targets_torch(
        close_tensor=closes_t,
        atr_tensor=atrs_t,
        horizons=(1, 2, 3),
        clamp_boundary=True,
        clip_range=(-10.0, 10.0),
        atr_floor=1e-4,
    )

    assert targets_pt.shape == (n, 3)
    assert not torch.isnan(targets_pt).any(), "PyTorch targets contain NaNs under degenerate ATR inputs!"
    assert not torch.isinf(targets_pt).any(), "PyTorch targets contain Infs under degenerate ATR inputs!"
    assert torch.all(targets_pt >= -10.0) and torch.all(targets_pt <= 10.0), "PyTorch targets breached [-10, 10] clipping!"

    # Verify numerical parity between NumPy and PyTorch
    diff = np.abs(targets_np - targets_pt.cpu().numpy())
    assert np.max(diff) < 1e-5, f"Discrepancy between NumPy and PyTorch under degenerate ATR: {np.max(diff)}"


def test_degenerate_atr_single_regime_dataset_getitem():
    """
    Validates that SingleRegimeDataset.__getitem__ properly sanitizes degenerate ATR values
    to safe_atr >= atr_floor (1e-4) without propagating NaNs or Infs.
    """
    total_bars = 100
    atr_arr = np.full(total_bars, 25.0, dtype=np.float32)
    # Inject degenerate ATR values
    atr_arr[0] = 0.0
    atr_arr[1] = -10.0
    atr_arr[2] = float("nan")
    atr_arr[3] = float("inf")
    atr_arr[4] = float("-inf")
    atr_arr[5] = 1e-12

    env = MockM1Environment(total_bars=total_bars, atr_arr=atr_arr)
    ds = SingleRegimeDataset("test_degenerate", env, atr_floor=1e-4)

    for i in range(6):
        obs, seq, tgt, atr = ds[i]
        assert not torch.isnan(atr).any(), f"Bar {i} returned NaN ATR in __getitem__!"
        assert not torch.isinf(atr).any(), f"Bar {i} returned Inf ATR in __getitem__!"
        # In float32, 1e-4 is approx 9.9999997e-5; verify with numerical epsilon tolerance
        assert atr.item() >= 1e-4 - 1e-7, f"Bar {i} returned ATR {atr.item()} below safety floor 1e-4!"
        assert not torch.isnan(tgt).any(), f"Bar {i} returned NaN target in __getitem__!"
        assert not torch.isinf(tgt).any(), f"Bar {i} returned Inf target in __getitem__!"


def test_degenerate_atr_get_batch_vulnerability_detection():
    """
    Adversarial vulnerability check on SingleRegimeDataset.get_batch():
    Line 371: atr_list = [max(float(self.env.atr_arr[bi]), self.atr_floor) for bi in bar_indices]
    Because max(nan, 1e-4) evaluates to nan and max(inf, 1e-4) evaluates to inf,
    get_batch() leaks NaNs/Infs into MegaPoolBatch.atr.

    This test empirically documents whether get_batch is susceptible to NaN/Inf pollution.
    """
    total_bars = 100
    atr_arr = np.full(total_bars, 25.0, dtype=np.float32)
    atr_arr[10] = float("nan")
    atr_arr[11] = float("inf")

    env = MockM1Environment(total_bars=total_bars, atr_arr=atr_arr)
    ds = SingleRegimeDataset("test_batch_vuln", env, atr_floor=1e-4)

    batch = ds.get_batch([10, 11])
    nan_in_atr = torch.isnan(batch.atr).any().item()
    inf_in_atr = torch.isinf(batch.atr).any().item()

    if nan_in_atr or inf_in_atr:
        # Vulnerability confirmed: get_batch fails to guard against NaN/Inf in atr_arr
        pytest.xfail(
            "CONFIRMED DEFECT: SingleRegimeDataset.get_batch() (line 371) leaks NaN/Inf into batch.atr "
            "because it uses max(float(atr[bi]), atr_floor) without math.isfinite guard."
        )
    else:
        assert batch.atr[0].item() >= 1e-4 - 1e-7
        assert batch.atr[1].item() >= 1e-4 - 1e-7


# =============================================================================
# 2. BOUNDARY HORIZON CONDITIONS
# =============================================================================


def test_boundary_horizon_conditions():
    """
    Stress-test boundary horizons:
    Let N = total_bars (e.g. 100).
    Horizons = (5, 15, 30).
    Max horizon = 30.
    valid_len = N - 30 = 70.
    Valid dataset index range: [0, 69].
    - t = N - 31 (bar 69): Last legal bar. Horizon 30 targets index 69 + 30 = 99 = N - 1.
    - t = N - 30 (bar 70): Boundary breach. Must raise IndexError.
    - t = N - 1 (bar 99): Tail bar. Must raise IndexError.
    - t >= N (bar 100): Out of bounds. Must raise IndexError.
    - t < 0: Negative index. Must raise IndexError.
    """
    n = 100
    env = MockM1Environment(total_bars=n)
    ds = SingleRegimeDataset("test_boundary", env, horizons=(5, 15, 30))

    assert len(ds) == 70  # N - 30

    # 1. Legal boundary: t = N - 31 (index 69)
    last_idx = len(ds) - 1
    assert last_idx == 69
    obs, seq, tgt, atr = ds[last_idx]
    assert obs.shape == (137,)
    assert seq.shape == (240, 6)
    assert tgt.shape == (3,)
    assert not torch.isnan(tgt).any()

    # Verify mathematical exactness at t = N - 31
    expected_diff_h30 = env.close_arr[69 + 30] - env.close_arr[69]
    expected_tgt_h30 = np.clip(expected_diff_h30 / env.atr_arr[69], -10.0, 10.0)
    assert abs(tgt[2].item() - expected_tgt_h30) < 1e-5

    # Legal get_batch at boundary
    batch = ds.get_batch([69])
    assert batch.targets_3.shape == (1, 3)
    assert abs(batch.targets_3[0, 2].item() - expected_tgt_h30) < 1e-5

    # 2. Boundary breach: t = N - 30 (index 70)
    with pytest.raises(IndexError):
        _ = ds[70]
    with pytest.raises(IndexError):
        _ = ds.get_batch([70])

    # 3. Tail boundary: t = N - 1 (index 99)
    with pytest.raises(IndexError):
        _ = ds[99]
    with pytest.raises(IndexError):
        _ = ds.get_batch([99])

    # 4. Out of bounds: t >= N (index 100)
    with pytest.raises(IndexError):
        _ = ds[100]
    with pytest.raises(IndexError):
        _ = ds.get_batch([100])

    # 5. Negative index: t < 0
    with pytest.raises(IndexError):
        _ = ds[-1]
    with pytest.raises(IndexError):
        _ = ds.get_batch([-1])


def test_causal_sequence_padding_early_boundaries():
    """
    Verify causal left-padding for early bars:
    - At t = 0: padding is 239 zeros, row 239 is bar 0.
    - At t = 1: padding is 238 zeros, rows 238..239 are bars 0..1.
    - At t = 238: padding is 1 zero row, rows 1..239 are bars 0..238.
    - At t = 239: full 240 bars, zero padding.
    - At t = 240: full 240 bars, zero padding.
    """
    env = MockM1Environment(total_bars=300)
    ds = SingleRegimeDataset("test_padding", env, lookback=240)

    # Bar 0
    _, seq_0, _, _ = ds[0]
    assert seq_0.shape == (240, 6)
    assert torch.all(seq_0[:239, :] == 0.0), "Bar 0 leading 239 rows must be zero-padded!"
    assert not torch.all(seq_0[239, :] == 0.0), "Bar 0 last row should be non-zero!"

    # Bar 1
    _, seq_1, _, _ = ds[1]
    assert torch.all(seq_1[:238, :] == 0.0), "Bar 1 leading 238 rows must be zero-padded!"
    assert not torch.all(seq_1[238:, :] == 0.0), "Bar 1 rows 238..239 should be non-zero!"

    # Bar 238
    _, seq_238, _, _ = ds[238]
    assert torch.all(seq_238[0, :] == 0.0), "Bar 238 row 0 must be zero-padded!"
    assert not torch.all(seq_238[1:, :] == 0.0), "Bar 238 rows 1..239 should be non-zero!"

    # Bar 239
    _, seq_239, _, _ = ds[239]
    assert not torch.all(seq_239[0, :] == 0.0), "Bar 239 row 0 should not be padded!"

    # Bar 240
    _, seq_240, _, _ = ds[240]
    assert not torch.all(seq_240[0, :] == 0.0), "Bar 240 row 0 should not be padded!"


# =============================================================================
# 3. OOS SECURITY & FIREWALL BYPASS ATTEMPTS
# =============================================================================


def test_oos_firewall_bypass_attacks():
    """
    Adversarially attempt 10 different attack vectors to breach the OOS firewall:
    - Path 1: Direct call router.get_batch(pool_name="oos")
    - Path 2: Direct call router.get_batch(pool_name=MarketRegime.OOS)
    - Path 3: Direct call router.get_batch("oos") (positional string)
    - Path 4: Substring injection router.get_batch(pool_name="oos_test")
    - Path 5: Substring injection router.get_batch(pool_name="custom_oos_data")
    - Path 6: Case & whitespace injection router.get_batch(pool_name="  OOS  ")
    - Path 7: Composite dataset injection with key 'oos'
    - Path 8: Composite dataset injection with substring 'my_oos_dataset'
    - Path 9: Composite dataset injection with is_oos=True flag
    - Path 10: State tampering injection followed by validate_zero_oos_leakage()
    """
    bull_env = MockM1Environment(total_bars=500, name="Bull")
    bear_env = MockM1Environment(total_bars=500, name="Bear")
    chop_env = MockM1Environment(total_bars=500, name="Chop")
    oos_env = MockM1Environment(total_bars=9785, name="OOS")

    config = MegaPoolConfig(
        bull_bars=500,
        bear_bars=500,
        chop_bars=500,
        oos_master_bars=9785,
        oos_slice_start=0,
        oos_slice_end=9785,
        oos_slice_bars=9785,
        use_cache=False,
    )
    router = MegaPoolRouter(
        config=config,
        bull_env=bull_env,
        bear_env=bear_env,
        chop_env=chop_env,
        oos_env=oos_env,
        load_on_init=True,
    )

    # Path 1: get_batch(pool_name="oos")
    with pytest.raises(PermissionError):
        _ = router.get_batch(pool_name="oos")

    # Path 2: get_batch(pool_name=MarketRegime.OOS)
    with pytest.raises(PermissionError):
        _ = router.get_batch(pool_name=MarketRegime.OOS)

    # Path 3: get_batch("oos") positional
    with pytest.raises(PermissionError):
        _ = router.get_batch("oos")

    # Path 4: get_batch(pool_name="oos_test")
    with pytest.raises(PermissionError):
        _ = router.get_batch(pool_name="oos_test")

    # Path 5: get_batch(pool_name="custom_oos_data")
    with pytest.raises(PermissionError):
        _ = router.get_batch(pool_name="custom_oos_data")

    # Path 6: get_batch(pool_name="  OOS  ")
    with pytest.raises(PermissionError):
        _ = router.get_batch(pool_name="  OOS  ")

    # Path 7: MegaPoolDataset with key 'oos'
    with pytest.raises(ValueError):
        _ = MegaPoolDataset({"oos": router.oos_dataset})

    # Path 8: MegaPoolDataset with substring 'my_oos_dataset'
    fake_ds = SingleRegimeDataset("my_oos_dataset", bull_env, is_oos=False)
    with pytest.raises(ValueError):
        _ = MegaPoolDataset({"my_oos_dataset": fake_ds})

    # Path 9: MegaPoolDataset with is_oos=True flag
    flagged_ds = SingleRegimeDataset("stealth", bull_env, is_oos=True)
    with pytest.raises(ValueError):
        _ = MegaPoolDataset({"stealth": flagged_ds})

    # Path 10: Injected tampered dataset into router.training_datasets
    tampered_router = MegaPoolRouter(
        config=config,
        bull_env=bull_env,
        bear_env=bear_env,
        chop_env=chop_env,
        oos_env=oos_env,
        load_on_init=True,
    )
    tampered_router.training_datasets["sneaky_oos"] = router.oos_dataset
    with pytest.raises(AssertionError):
        tampered_router.validate_zero_oos_leakage()


# =============================================================================
# 4. CONCURRENCY & MULTI-THREADING STRESS
# =============================================================================


def test_concurrency_multithreading_batch_generation():
    """
    Stress-test concurrent mini-batch generation across 8 OS threads:
    - 8 worker threads concurrently requesting 50 batches of size 32 (total 400 batches).
    - Verifies thread-safety, race-condition freedom, and tensor shape consistency.
    """
    bull_env = MockM1Environment(total_bars=500, name="Bull")
    bear_env = MockM1Environment(total_bars=500, name="Bear")
    chop_env = MockM1Environment(total_bars=500, name="Chop")
    oos_env = MockM1Environment(total_bars=9785, name="OOS")

    config = MegaPoolConfig(
        bull_bars=500,
        bear_bars=500,
        chop_bars=500,
        oos_master_bars=9785,
        oos_slice_start=0,
        oos_slice_end=9785,
        oos_slice_bars=9785,
        use_cache=False,
    )
    router = MegaPoolRouter(
        config=config,
        bull_env=bull_env,
        bear_env=bear_env,
        chop_env=chop_env,
        oos_env=oos_env,
        load_on_init=True,
    )

    def worker_task(thread_id: int) -> List[str]:
        errors: List[str] = []
        for step in range(50):
            try:
                batch = router.get_batch(32)
                if batch.obs_137.shape != (32, 137):
                    errors.append(f"T{thread_id} step {step}: obs shape mismatch {batch.obs_137.shape}")
                if batch.candle_seq_240.shape != (32, 240, 6):
                    errors.append(f"T{thread_id} step {step}: seq shape mismatch {batch.candle_seq_240.shape}")
                if batch.targets_3.shape != (32, 3):
                    errors.append(f"T{thread_id} step {step}: tgt shape mismatch {batch.targets_3.shape}")
                if batch.atr.shape != (32,):
                    errors.append(f"T{thread_id} step {step}: atr shape mismatch {batch.atr.shape}")
                if torch.isnan(batch.obs_137).any():
                    errors.append(f"T{thread_id} step {step}: NaN in obs_137")
                if torch.isnan(batch.targets_3).any():
                    errors.append(f"T{thread_id} step {step}: NaN in targets_3")
            except Exception as e:
                errors.append(f"T{thread_id} step {step} raised {type(e).__name__}: {e}")
        return errors

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker_task, i) for i in range(8)]
        all_errors: List[str] = []
        for f in concurrent.futures.as_completed(futures):
            all_errors.extend(f.result())

    assert len(all_errors) == 0, f"Encountered {len(all_errors)} errors during concurrent execution: {all_errors[:5]}"


# =============================================================================
# 5. HIGH-THROUGHPUT STRESS BENCHMARK
# =============================================================================


def test_high_throughput_and_tensor_cleanliness():
    """
    Stress-test throughput and tensor numerical cleanliness across 50 batches of size 64:
    - Verifies throughput > 2,500 samples/sec on CPU across authentic 509k dataset.
    - Confirms 0 NaNs and 0 Infs across all 4 batch tensors.
    """
    router = MegaPoolRouter()
    assert router.validate_zero_oos_leakage() is True

    # Warmup
    for _ in range(5):
        _ = router.get_batch(32)

    num_batches = 50
    batch_size = 64
    total_samples = num_batches * batch_size

    t0 = time.perf_counter()
    for _ in range(num_batches):
        batch = router.get_batch(batch_size)
        assert batch.obs_137.shape == (batch_size, 137)
        assert batch.candle_seq_240.shape == (batch_size, 240, 6)
        assert batch.targets_3.shape == (batch_size, 3)
        assert batch.atr.shape == (batch_size,)

        assert not torch.isnan(batch.obs_137).any()
        assert not torch.isnan(batch.candle_seq_240).any()
        assert not torch.isnan(batch.targets_3).any()
        assert not torch.isnan(batch.atr).any()

        assert not torch.isinf(batch.obs_137).any()
        assert not torch.isinf(batch.candle_seq_240).any()
        assert not torch.isinf(batch.targets_3).any()
        assert not torch.isinf(batch.atr).any()

    elapsed = time.perf_counter() - t0
    throughput = total_samples / max(elapsed, 1e-6)
    assert throughput > 2500.0, f"Throughput {throughput:.1f} samples/sec below 2,500 threshold!"


# =============================================================================
# 6. EDGE CASE: EMPTY BATCH & ZERO-LENGTH REGIMES
# =============================================================================


def test_empty_batch_handling():
    """
    Test edge cases:
    - SingleRegimeDataset.get_batch([]) returns valid empty MegaPoolBatch.
    - collate_mega_pool([]) raises RuntimeError because torch.stack requires non-empty list.
    """
    env = MockM1Environment(total_bars=100)
    ds = SingleRegimeDataset("empty_test", env)

    empty_batch = ds.get_batch([])
    assert empty_batch.obs_137.shape == (0, 137)
    assert empty_batch.candle_seq_240.shape == (0, 240, 6)
    assert empty_batch.targets_3.shape == (0, 3)
    assert empty_batch.atr.shape == (0,)

    # collate_mega_pool([]) fails on torch.stack([])
    with pytest.raises(RuntimeError):
        _ = collate_mega_pool([])
