"""
Adversarial Stress Test Suite: Challenger 2 for Solution 3 (Temporal Cross-Attention / TCA).

Stress-tests:
1. RealM1Environment.get_candle_sequence boundary conditions:
   - Bars 0, 1, 238, 239, 240, 64784 (on authentic BTCUSDm.csv dataset).
   - Out of range indices (-1, -100, 64785, 100000) raising IndexError.
   - Strict attribute bound checks across all 6 columns, zero NaNs, zero Infs.
   - Synthetic adversarial feeds: NaN ATR, zero volume, flat prices, lookback > total_bars.
2. Attention telemetry interface (get_attention_summary, get_attention_weights):
   - Varying batch sizes (B=1, 2, 4, 16, 64) and unbatch behaviors.
   - Theoretical Oracles for Uniform, Dirac Delta (latest bar), Dirac Delta (first bar).
   - Shannon entropy, recency ratio, mean attention, and top-k indices verification.
   - Multi-query MCTS evaluation over single-item precomputed KV-cache.
3. MT5 Bridge Latency & Numerical Stability:
   - Full tick cycle latency benchmark (must be strictly < 50ms on CPU).
   - Numerical stability under extreme floats (+-1e12), subnormals (1e-35), Infs, and NaNs.
"""

from __future__ import annotations

import math
import os
import time
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    TemporalKVCache,
)
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment


CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"


# ---------------------------------------------------------------------------
# 1. RealM1Environment Boundary Conditions & Bounds
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not os.path.exists(CSV_PATH), reason="BTCUSDm.csv not found")
def test_realm1_environment_real_data_boundary_conditions():
    """Verify get_candle_sequence across boundary indices on the full 64,785 bar dataset."""
    env = RealM1Environment(csv_path=CSV_PATH)
    total_bars = env.load_data()
    assert total_bars == 64785, f"Expected 64785 bars, got {total_bars}"

    # Target boundary test points:
    # bar 0: 239 leading zero rows, 1 populated row
    # bar 1: 238 leading zero rows, 2 populated rows
    # bar 238: 1 leading zero row, 239 populated rows
    # bar 239: exact boundary where sequence becomes fully populated (0 padding)
    # bar 240: first standard non-boundary step
    # bar 64784: final bar in the dataset
    boundary_cases = [
        (0, 239),
        (1, 238),
        (238, 1),
        (239, 0),
        (240, 0),
        (64784, 0),
    ]

    for bar_idx, expected_pad_zeros in boundary_cases:
        seq = env.get_candle_sequence(bar_idx, lookback=240)
        assert seq.shape == (240, 6), f"Bar {bar_idx} returned shape {seq.shape}"
        assert seq.dtype == np.float32

        # Verify zero NaNs and zero Infs
        assert not np.isnan(seq).any(), f"Bar {bar_idx} contains NaN"
        assert not np.isinf(seq).any(), f"Bar {bar_idx} contains Inf"

        # Verify exact causal left-padding
        if expected_pad_zeros > 0:
            assert np.all(seq[:expected_pad_zeros, :] == 0.0), f"Bar {bar_idx} padding mismatch"
            # Ensure populated slice has non-zero or valid values
            assert not np.all(seq[expected_pad_zeros:, :] == 0.0), f"Bar {bar_idx} populated slice is all zero"
        else:
            # Full sequence should have no systemic zero padding
            assert not np.all(seq[0, :] == 0.0), f"Bar {bar_idx} unexpectedly all-zero at index 0"

        # Verify all 6 relative-normalized attributes are strictly within their bounded contracts
        # 0: Body ATR ratio [-10.0, 10.0]
        assert np.all((seq[:, 0] >= -10.0) & (seq[:, 0] <= 10.0)), f"Bar {bar_idx} col 0 out of bounds"
        # 1: Upper Wick ATR ratio [0.0, 10.0]
        assert np.all((seq[:, 1] >= 0.0) & (seq[:, 1] <= 10.0)), f"Bar {bar_idx} col 1 out of bounds"
        # 2: Lower Wick ATR ratio [0.0, 10.0]
        assert np.all((seq[:, 2] >= 0.0) & (seq[:, 2] <= 10.0)), f"Bar {bar_idx} col 2 out of bounds"
        # 3: Price Velocity ATR ratio [-10.0, 10.0]
        assert np.all((seq[:, 3] >= -10.0) & (seq[:, 3] <= 10.0)), f"Bar {bar_idx} col 3 out of bounds"
        # 4: Relative Volume [-1.0, 15.0]
        assert np.all((seq[:, 4] >= -1.0) & (seq[:, 4] <= 15.0)), f"Bar {bar_idx} col 4 out of bounds"
        # 5: Relative Spread [0.0, 10.0]
        assert np.all((seq[:, 5] >= 0.0) & (seq[:, 5] <= 10.0)), f"Bar {bar_idx} col 5 out of bounds"

    # Out of range checks
    for oob in [-1, -100, 64785, 100000]:
        with pytest.raises(IndexError):
            env.get_candle_sequence(oob, lookback=240)


def test_realm1_environment_synthetic_adversarial_data():
    """Stress-test get_candle_sequence on synthetic environment with pathological data feeds."""
    env = RealM1Environment()
    N = 60
    env.total_bars = N
    # Pathological inputs: zero volume, flat prices, NaN/Inf ATR, zero spread
    env.open_arr = np.full(N, 1000.0, dtype=np.float32)
    env.high_arr = np.full(N, 1000.0, dtype=np.float32)
    env.low_arr = np.full(N, 1000.0, dtype=np.float32)
    env.close_arr = np.full(N, 1000.0, dtype=np.float32)
    env.volume_arr = np.zeros(N, dtype=np.float32)
    env.spread_arr = np.zeros(N, dtype=np.float32)
    env.atr_arr = np.full(N, float("nan"), dtype=np.float32)  # Injected NaN ATR
    env.day_progress_arr = np.zeros(N, dtype=np.float32)
    env.timestamps = ["2025-01-01 00:00:00"] * N
    env.obs_matrix = np.zeros((N, 137), dtype=np.float32)

    # 1. Bar 0 with lookback 240 (lookback > total_bars)
    seq_0 = env.get_candle_sequence(0, lookback=240)
    assert seq_0.shape == (240, 6)
    assert not np.isnan(seq_0).any()
    assert not np.isinf(seq_0).any()
    # Relative volume: 0 / 1e-4 - 1 = -1.0
    assert seq_0[239, 4] == -1.0

    # 2. Bar N-1 (last bar) with lookback 240
    seq_last = env.get_candle_sequence(N - 1, lookback=240)
    assert seq_last.shape == (240, 6)
    assert not np.isnan(seq_last).any()
    assert not np.isinf(seq_last).any()

    # 3. Injected Inf ATR at specific bar
    env.atr_arr[30] = float("inf")
    seq_inf = env.get_candle_sequence(30, lookback=240)
    assert not np.isnan(seq_inf).any()
    assert not np.isinf(seq_inf).any()


# ---------------------------------------------------------------------------
# 2. Attention Telemetry Oracles & Distributions
# ---------------------------------------------------------------------------

def test_attention_telemetry_oracles():
    """Verify quantitative attention telemetry against analytical mathematical oracles."""
    tca = TemporalCrossAttentionEncoder(candle_len=240, num_heads=4)

    # Oracle 1: Uniform Attention Distribution (p_i = 1 / 240 for all i)
    # Expected Shannon entropy: ln(240) = 5.4806389...
    # Expected recency ratio: (15 / 240) / (225 / 240 + 1e-6) = 1/15 = 0.066666...
    tca.last_candle_attn_weights = torch.full((1, 4, 240), 1.0 / 240.0)
    summary_uniform = tca.get_attention_summary()
    assert summary_uniform is not None

    expected_entropy = math.log(240.0)
    assert np.allclose(summary_uniform["head_entropy"], [expected_entropy] * 4, atol=1e-4)

    expected_recency = (15.0 / 240.0) / ((225.0 / 240.0) + 1e-6)
    assert np.isclose(summary_uniform["recency_weight_ratio"], expected_recency, atol=1e-4)
    assert np.allclose(summary_uniform["mean_attention"], np.full(240, 1.0 / 240.0), atol=1e-5)
    assert np.isclose(summary_uniform["last_candle_weight"], 1.0 / 240.0, atol=1e-5)

    # Oracle 2: Dirac Delta Distribution at Current Candle (bar 239)
    # All attention on the newest bar
    # Expected entropy: 0.0
    # Expected recency ratio: 1.0 / 1e-6 = 1,000,000
    dirac_latest = torch.zeros((1, 4, 240))
    dirac_latest[:, :, 239] = 1.0
    tca.last_candle_attn_weights = dirac_latest
    summary_latest = tca.get_attention_summary()
    assert summary_latest is not None

    assert np.allclose(summary_latest["head_entropy"], [0.0] * 4, atol=1e-5)
    assert np.isclose(summary_latest["recency_weight_ratio"], 1.0 / 1e-6, atol=1.0)
    assert summary_latest["last_candle_weight"] == 1.0
    assert summary_latest["top_k_indices"][0] == 239

    # Oracle 3: Dirac Delta Distribution at Oldest Candle (bar 0)
    # All attention on the earliest bar
    # Expected entropy: 0.0
    # Expected recency ratio: 0.0 / (1.0 + 1e-6) = 0.0
    dirac_oldest = torch.zeros((1, 4, 240))
    dirac_oldest[:, :, 0] = 1.0
    tca.last_candle_attn_weights = dirac_oldest
    summary_oldest = tca.get_attention_summary()
    assert summary_oldest is not None

    assert np.allclose(summary_oldest["head_entropy"], [0.0] * 4, atol=1e-5)
    assert np.isclose(summary_oldest["recency_weight_ratio"], 0.0, atol=1e-5)
    assert summary_oldest["last_candle_weight"] == 0.0
    assert summary_oldest["top_k_indices"][0] == 0


def test_attention_telemetry_varying_batch_sizes_and_interfaces():
    """Verify attention telemetry under varying batch sizes (B=1, 2, 8, 32) and unbatch interface."""
    tca = TemporalCrossAttentionEncoder(candle_len=240, num_heads=4)

    # 1. Uninitialized state
    assert tca.get_attention_weights() is None
    assert tca.get_attention_summary() is None

    # 2. Batch size 1
    tca.last_candle_attn_weights = torch.softmax(torch.randn(1, 4, 240), dim=-1)
    w_batched = tca.get_attention_weights(unbatch=False)
    assert w_batched.shape == (1, 4, 240)
    w_unbatched = tca.get_attention_weights(unbatch=True)
    assert w_unbatched.shape == (4, 240)

    # 3. Varying batch sizes
    for B in [2, 4, 16, 64]:
        tca.last_candle_attn_weights = torch.softmax(torch.randn(B, 4, 240), dim=-1)
        w_b = tca.get_attention_weights(unbatch=False)
        assert w_b.shape == (B, 4, 240)

        # When B > 1, unbatch=True should preserve shape [B, 4, 240]
        w_unb = tca.get_attention_weights(unbatch=True)
        assert w_unb.shape == (B, 4, 240)

        summary = tca.get_attention_summary()
        assert summary is not None
        assert summary["mean_attention"].shape == (240,)
        assert np.isclose(summary["mean_attention"].sum(), 1.0, atol=1e-3)
        assert len(summary["head_entropy"]) == 4
        assert len(summary["top_k_indices"]) == 5


def test_mcts_multi_query_over_single_cached_kv():
    """
    Verify that single-bar precomputed KV-cache [1, 4, 240, 32] expands seamlessly
    to handle batched candidate action queries [B, 768] (as encountered in MCTS).
    """
    tca = TemporalCrossAttentionEncoder(candle_len=240)
    tca.eval()

    candle_seq = torch.randn(1, 240, 6)
    kv_cache = tca.precompute_kv_cache(candle_seq)
    assert kv_cache.key.shape == (1, 4, 240, 32)

    for B in [1, 2, 8, 16, 32]:
        z_feat_batch = torch.randn(B, 768)
        z_out, attn_weights = tca.forward_with_kv_cache(z_feat_batch, kv_cache)
        assert z_out.shape == (B, 768)
        assert attn_weights.shape == (B, 4, 240)
        assert not torch.isnan(z_out).any()
        assert not torch.isnan(attn_weights).any()


# ---------------------------------------------------------------------------
# 3. MT5 Bridge Latency & Numerical Stability
# ---------------------------------------------------------------------------

def test_mt5_bridge_latency_benchmark():
    """
    Empirical latency benchmark:
    Verify that full tick pipeline (get_candle_sequence + KV-cache + encode_state + attention summary)
    executes well under the MT5 bridge constraint of 50ms per tick.
    """
    env = RealM1Environment()
    N = 500
    env.total_bars = N
    env.open_arr = np.linspace(50000, 50200, N, dtype=np.float32)
    env.high_arr = env.open_arr + 10.0
    env.low_arr = env.open_arr - 10.0
    env.close_arr = env.open_arr + 2.0
    env.volume_arr = np.ones(N, dtype=np.float32) * 20.0
    env.spread_arr = np.ones(N, dtype=np.float32) * 2.5
    env.atr_arr = np.ones(N, dtype=np.float32) * 25.0

    wm = LatentDynamicsWorldModel()
    wm.enable_temporal_stream()
    wm.eval()

    obs_137 = np.random.randn(137).astype(np.float32)

    # Warmup
    for _ in range(5):
        candle_seq = env.get_candle_sequence(300)
        cache = wm.precompute_temporal_kv_cache(candle_seq)
        _ = wm.encode_state(obs_137, kv_cache=cache)
        _ = wm.get_attention_summary()

    latencies_env: list[float] = []
    latencies_kv: list[float] = []
    latencies_encode: list[float] = []
    latencies_summary: list[float] = []
    latencies_total: list[float] = []

    iterations = 50
    for i in range(iterations):
        bar_idx = 250 + i
        t0 = time.perf_counter()
        candle_seq = env.get_candle_sequence(bar_idx)
        t1 = time.perf_counter()
        cache = wm.precompute_temporal_kv_cache(candle_seq)
        t2 = time.perf_counter()
        z = wm.encode_state(obs_137, kv_cache=cache)
        t3 = time.perf_counter()
        summary = wm.get_attention_summary()
        t4 = time.perf_counter()

        latencies_env.append((t1 - t0) * 1000)
        latencies_kv.append((t2 - t1) * 1000)
        latencies_encode.append((t3 - t2) * 1000)
        latencies_summary.append((t4 - t3) * 1000)
        latencies_total.append((t4 - t0) * 1000)

    mean_total = float(np.mean(latencies_total))
    p99_total = float(np.percentile(latencies_total, 99))
    max_total = float(np.max(latencies_total))

    # Assertions for MT5 bridge contract (< 50ms)
    assert mean_total < 50.0, f"Mean total latency {mean_total:.2f}ms exceeds 50ms"
    assert p99_total < 50.0, f"P99 total latency {p99_total:.2f}ms exceeds 50ms"
    assert max_total < 50.0, f"Max total latency {max_total:.2f}ms exceeds 50ms"


def test_numerical_stability_extreme_and_pathological_inputs():
    """
    Stress-test numerical safeguards:
    - Extreme float magnitude (+-1e12)
    - Subnormal/denormal floats (1e-35)
    - Injected NaNs and Infs in both query features and candle sequence
    - Verify soft clamping [-50.0, 50.0] guarantees finite attention weights summing to 1.0.
    """
    tca = TemporalCrossAttentionEncoder(candle_len=240)
    tca.eval()

    # 1. Extreme float magnitudes (+-1e12)
    z_feat_extreme = torch.full((2, 768), 1e12)
    candle_seq_extreme = torch.full((2, 240, 6), -1e12)
    z_out, attn_weights = tca(z_feat_extreme, candle_seq_extreme)
    assert not torch.isnan(z_out).any(), "NaN found in extreme float z_out"
    assert not torch.isinf(z_out).any(), "Inf found in extreme float z_out"
    assert not torch.isnan(attn_weights).any(), "NaN found in extreme float attn_weights"
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(2, 4), atol=1e-5)

    # 2. Subnormal floats (1e-35)
    z_feat_sub = torch.full((2, 768), 1e-35)
    candle_seq_sub = torch.full((2, 240, 6), 1e-35)
    z_out_sub, attn_sub = tca(z_feat_sub, candle_seq_sub)
    assert not torch.isnan(z_out_sub).any()
    assert not torch.isinf(z_out_sub).any()

    # 3. Pathological Infs and NaNs
    z_feat_nan = torch.tensor([[float("nan"), float("inf"), float("-inf")] + [0.0] * 765])
    candle_seq_nan = torch.zeros((1, 240, 6))
    candle_seq_nan[0, :50, :] = float("nan")
    candle_seq_nan[0, 50:100, :] = float("inf")
    candle_seq_nan[0, 100:150, :] = float("-inf")

    z_out_nan, attn_nan = tca(z_feat_nan, candle_seq_nan)
    assert not torch.isnan(z_out_nan).any(), "NaN propagated into z_out"
    assert not torch.isinf(z_out_nan).any(), "Inf propagated into z_out"
    assert not torch.isnan(attn_nan).any(), "NaN propagated into attn_nan"
    assert not torch.isinf(attn_nan).any(), "Inf propagated into attn_nan"
    assert torch.allclose(attn_nan.sum(dim=-1), torch.ones(1, 4), atol=1e-5)
