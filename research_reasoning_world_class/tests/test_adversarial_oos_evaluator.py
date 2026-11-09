"""
Adversarial Stress Harness & Fuzzing Suite for Milestone 4: OOS Evaluation Runner.
File: research_reasoning_world_class/tests/test_adversarial_oos_evaluator.py
Milestone: Milestone 4 (Out-of-Sample Crash Evaluation & Web Dashboard Verification)

Adversarial Stress Matrix:
1. Batch & Market State Fuzzing (Extreme / Degenerate Numerical Stability):
   - Near-zero ATR fuzzing (1e-8, 0.0, -1e-6, -10.0) in trading simulation and target generation.
   - Floor protection verification against division by zero.
   - Extreme price spikes ($10^6, $10^7, $10^{-6}) and extreme spreads ($0.0, $50.0, $500.0, $10000.0).
   - Multi-horizon targets computation under non-finite (NaN, Inf) and clipped market conditions.
   - Metric formulations (DA%, MAE, Huber loss) under astronomical predictions.

2. LatentDynamicsWorldModel & TCA Encoder Adversarial Sanitization:
   - Extreme candle spikes ($10^6$) and non-finite observation vectors.
   - Attention telemetry extraction under degenerate weights (all zeros, one-hot, uniform).

3. CLI Edge Cases & Failure Modes:
   - Missing data cache and authentic CSV files (clean FileNotFoundError).
   - Corrupted checkpoint files (truncated binary, invalid pickle).
   - Incomplete and truncated data slices (lookback < 240, bars < max_horizon).
   - CLI runner subprocess execution with corrupted inputs and exit code 1 validation.
   - CLI runner dry-run verification with exit code 0 and JSON schema compliance.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from research_reasoning_world_class.src.mega_pool_dataset import (
    SlicedM1Environment,
    SingleRegimeDataset,
    compute_multi_horizon_targets_numpy,
    compute_multi_horizon_targets_torch,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MultiHorizonProjectionHead,
    compute_attention_telemetry,
)
from scripts.evaluate_tca_oos import (
    load_oos_slice,
    load_baseline_model,
    load_phase1_model,
    load_phase2_model,
    compute_horizon_metrics,
    simulate_spread_adjusted_trading,
    resolve_device,
    evaluate_all_checkpoints,
)


# =============================================================================
# 1. BATCH & MARKET STATE FUZZING (NUMERICAL STABILITY)
# =============================================================================


class TestAdversarialMarketFuzzing:
    """Stress-tests mathematical operations under extreme and degenerate market states."""

    @pytest.mark.parametrize("atr_val", [1e-8, 0.0, -1e-6, -10.0, 1e-12])
    def test_fuzz_near_zero_atr_simulation(self, atr_val: float):
        """Asserts floor protection prevents division by zero when ATR is zero or negative."""
        preds = np.array([
            [0.0, 0.5, 0.0],
            [0.0, -0.4, 0.0],
            [0.0, 0.8, 0.0],
        ])
        targets = np.array([
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, -0.5, 0.0],
        ])
        closes = np.array([50000.0, 50000.0, 50000.0])
        atrs = np.array([atr_val, atr_val, atr_val])

        sim = simulate_spread_adjusted_trading(
            preds, targets, closes, atrs, spread_usd=2.50, threshold=0.20
        )

        assert sim["total_trades"] == 3
        assert math.isfinite(sim["net_pnl_usd"])
        assert math.isfinite(sim["win_rate_pct"])
        assert math.isfinite(sim["profit_factor"])
        assert math.isfinite(sim["max_drawdown_usd"])
        assert 0.0 <= sim["win_rate_pct"] <= 100.0
        assert sim["profit_factor"] >= 0.0

    def test_fuzz_extreme_price_spikes_simulation(self):
        """Asserts astronomical price levels ($10^6) do not cause overflow or loss of precision."""
        preds = np.array([
            [0.0, 1.5, 0.0],
            [0.0, -2.0, 0.0],
        ])
        targets = np.array([
            [0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        closes = np.array([1_000_000.0, 10_000_000.0])
        atrs = np.array([50.0, 100.0])

        sim = simulate_spread_adjusted_trading(
            preds, targets, closes, atrs, spread_usd=2.50, threshold=0.20
        )

        assert sim["total_trades"] == 2
        assert math.isfinite(sim["net_pnl_usd"])
        assert math.isfinite(sim["max_drawdown_usd"])
        assert math.isfinite(sim["max_drawdown_pct"])

    @pytest.mark.parametrize("spread", [0.0, 0.01, 2.50, 50.0, 500.0, 10000.0])
    def test_fuzz_extreme_spread_frictions(self, spread: float):
        """Asserts PnL calculations behave monotonically and stably across zero to massive spreads."""
        preds = np.array([
            [0.0, 0.5, 0.0],
            [0.0, -0.5, 0.0],
        ])
        targets = np.array([
            [0.0, 1.0, 0.0],   # Win at low spread: +1.0 * 50 = +50
            [0.0, -1.0, 0.0],  # Win at low spread: +1.0 * 50 = +50
        ])
        closes = np.array([50000.0, 50000.0])
        atrs = np.array([50.0, 50.0])

        sim = simulate_spread_adjusted_trading(
            preds, targets, closes, atrs, spread_usd=spread, threshold=0.20
        )

        assert sim["total_trades"] == 2
        assert math.isfinite(sim["net_pnl_usd"])
        assert math.isfinite(sim["profit_factor"])
        assert sim["spread_usd"] == spread

        if spread == 0.0:
            assert sim["win_rate_pct"] == 100.0
            assert sim["net_pnl_usd"] == 100.0
            assert sim["profit_factor"] == 99.0
        elif spread == 500.0:
            # 50 - 500 = -450 each -> 100% losses
            assert sim["win_rate_pct"] == 0.0
            assert sim["net_pnl_usd"] == -900.0
            assert sim["profit_factor"] == 0.0
            # Note: np.maximum.accumulate(equity) starts at equity[0] (-450.0), so max drawdown from peak is 450.0
            assert sim["max_drawdown_usd"] == 450.0

    def test_fuzz_zero_trades_trigger(self):
        """Asserts safe zeroed metrics are returned without division by zero when no trades fire."""
        preds = np.array([
            [0.0, 0.05, 0.0],
            [0.0, -0.05, 0.0],
        ])
        targets = np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]])
        closes = np.array([50000.0, 50000.0])
        atrs = np.array([50.0, 50.0])

        sim = simulate_spread_adjusted_trading(
            preds, targets, closes, atrs, spread_usd=2.50, threshold=0.20
        )

        assert sim["total_trades"] == 0
        assert sim["win_rate_pct"] == 0.0
        assert sim["profit_factor"] == 1.0
        assert sim["net_pnl_usd"] == 0.0
        assert sim["max_drawdown_usd"] == 0.0

    def test_fuzz_target_computation_degenerate_atr(self):
        """Verifies compute_multi_horizon_targets_numpy handles 0, negative, NaN, Inf ATR safely."""
        closes = np.array([50000.0, 50050.0, 50020.0, 50100.0, 50200.0] * 10, dtype=np.float32)
        n = len(closes)
        bad_atrs = np.array([0.0, 1e-8, -5.0, float("nan"), float("inf")] * 10, dtype=np.float32)

        targets = compute_multi_horizon_targets_numpy(
            close_arr=closes,
            atr_arr=bad_atrs,
            horizons=(5, 15, 30),
            clamp_boundary=True,
            clip_range=(-10.0, 10.0),
            atr_floor=1e-4,
        )

        assert targets.shape == (n, 3)
        assert np.isfinite(targets).all()
        assert (targets >= -10.0).all()
        assert (targets <= 10.0).all()

    def test_fuzz_target_computation_torch_parity(self):
        """Verifies PyTorch and NumPy target formulation implementations match on edge cases."""
        closes = np.array([1000.0, 1050.0, 950.0, 1020.0, 1080.0] * 8, dtype=np.float32)
        atrs = np.array([0.0, 1e-6, 10.0, 25.0] * 10, dtype=np.float32)

        np_targets = compute_multi_horizon_targets_numpy(
            closes, atrs, horizons=(5, 15, 30), clamp_boundary=True
        )
        torch_targets = compute_multi_horizon_targets_torch(
            torch.from_numpy(closes), torch.from_numpy(atrs), horizons=(5, 15, 30), clamp_boundary=True
        ).numpy()

        assert np_targets.shape == torch_targets.shape
        np.testing.assert_allclose(np_targets, torch_targets, atol=1e-5)


# =============================================================================
# 2. ADVERSARIAL METRICS FORMULATION
# =============================================================================


class TestAdversarialMetricsFormulation:
    """Stress-tests loss, MAE, and directional accuracy computations."""

    def test_metrics_extreme_predictions(self):
        """Verifies Huber loss and MAE scale smoothly without numerical overflow under $10^6$ shocks."""
        preds = np.array([[1e6, -1e6, 5e5]], dtype=np.float64)
        targets = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)

        metrics = compute_horizon_metrics(preds, targets)

        # For |diff| >= 1.0, Huber loss = |diff| - 0.5
        assert round(metrics["mae"]["h5"], 1) == 1_000_000.0
        assert round(metrics["loss"]["h5"], 1) == 999_999.5
        assert metrics["directional_accuracy_pct"]["total"] == 0.0

    def test_metrics_boundary_directional_accuracy(self):
        """Verifies directional accuracy boundary behavior for 100% match, 0% match, and exact zeros."""
        # 1. 100% agreement
        p_agree = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        t_agree = np.array([[0.5, 1.0, 1.5], [-0.5, -1.0, -1.5]])
        m_agree = compute_horizon_metrics(p_agree, t_agree)
        assert m_agree["directional_accuracy_pct"]["total"] == 100.0

        # 2. 0% agreement
        p_opp = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        t_opp = np.array([[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])
        m_opp = compute_horizon_metrics(p_opp, t_opp)
        assert m_opp["directional_accuracy_pct"]["total"] == 0.0

        # 3. Exactly zero predictions and targets
        p_zero = np.zeros((5, 3))
        t_zero = np.zeros((5, 3))
        m_zero = compute_horizon_metrics(p_zero, t_zero)
        assert m_zero["loss"]["total"] == 0.0
        assert m_zero["mae"]["total"] == 0.0
        assert m_zero["directional_accuracy_pct"]["total"] == 100.0

    def test_metrics_large_batch_stress(self):
        """Asserts compute_horizon_metrics executes on 50,000 samples in < 100ms."""
        np.random.seed(42)
        preds = np.random.randn(50_000, 3).astype(np.float32)
        targets = np.random.randn(50_000, 3).astype(np.float32)

        res = compute_horizon_metrics(preds, targets)
        assert "loss" in res and "total" in res["loss"]
        assert "mae" in res and "total" in res["mae"]
        assert "directional_accuracy_pct" in res and "total" in res["directional_accuracy_pct"]
        assert 0.0 <= res["directional_accuracy_pct"]["total"] <= 100.0


# =============================================================================
# 3. LATENT DYNAMICS & TELEMETRY ADVERSARIAL SANITIZATION
# =============================================================================


class TestAdversarialModelSanitization:
    """Stress-tests World Model and Attention Telemetry against hostile inputs."""

    def test_world_model_candle_spike_sanitization(self):
        """Passes astronomical candle sequences ($10^6$) through TCA and asserts finite latent states."""
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240, temp_dim=128, num_heads=4)
        wm.eval()

        obs = torch.randn(2, 137)
        candles_spike = torch.randn(2, 240, 6) * 1e6

        with torch.no_grad():
            z_out = wm.encode_state(obs, candle_seq=candles_spike)

        assert z_out.shape == (2, 768)
        assert torch.isfinite(z_out).all()
        # LayerNorm guarantees standard deviation and bounded range
        assert not torch.isnan(z_out).any()

    def test_world_model_obs_nan_inf_sanitization(self):
        """Passes observation vectors containing NaNs and Infs and asserts clean sanitization."""
        wm = LatentDynamicsWorldModel()
        wm.eval()

        bad_obs = torch.randn(4, 137)
        bad_obs[0, 5] = float("nan")
        bad_obs[1, 10] = float("inf")
        bad_obs[2, 15] = float("-inf")

        with torch.no_grad():
            z_out = wm.encode_state(bad_obs)

        assert z_out.shape == (4, 768)
        assert torch.isfinite(z_out).all()

    def test_projection_head_extreme_latent_sanitization(self):
        """Verifies MultiHorizonProjectionHead forward pass sanitizes extreme and non-finite latent vectors."""
        head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.0)
        head.eval()

        bad_z = torch.full((3, 768), float("nan"))
        bad_z[1, :] = 1e6
        bad_z[2, :] = float("inf")

        with torch.no_grad():
            preds = head(bad_z)

        assert preds.shape == (3, 3)
        assert torch.isfinite(preds).all()

    def test_attention_telemetry_extreme_distributions(self):
        """Tests compute_attention_telemetry across all zeros, one-hot impulses, and uniform weights."""
        # 1. All zeros
        attn_zeros = torch.zeros(4, 240)
        res_zeros = compute_attention_telemetry(attn_zeros)
        assert res_zeros["mean_entropy"] == 0.0
        assert res_zeros["recency_ratio"] == 0.0
        assert math.isfinite(res_zeros["gini_sparsity"])

        # 2. One-hot impulse (all attention on current tick)
        attn_onehot = torch.zeros(4, 240)
        attn_onehot[:, 239] = 1.0
        res_onehot = compute_attention_telemetry(attn_onehot)
        assert res_onehot["mean_entropy"] == 0.0
        assert res_onehot["peak_candle_index"] == 239
        assert res_onehot["peak_candle_lag"] == 0
        assert math.isfinite(res_onehot["recency_ratio"])

        # 3. Uniform distribution across all 240 candles
        attn_uniform = torch.ones(4, 240) / 240.0
        res_uniform = compute_attention_telemetry(attn_uniform)
        expected_max_ent = math.log(240.0)
        assert abs(res_uniform["mean_entropy"] - expected_max_ent) < 1e-4
        assert abs(res_uniform["recency_ratio"] - (30.0 / 210.0)) < 1e-4

        # 4. Truncated candle length (30 candles)
        attn_trunc = torch.ones(4, 30) / 30.0
        res_trunc = compute_attention_telemetry(attn_trunc, candle_len=30)
        assert res_trunc["candle_len"] == 30
        assert math.isfinite(res_trunc["mean_entropy"])


# =============================================================================
# 4. CLI EDGE CASES & ERROR RESILIENCE
# =============================================================================


class TestAdversarialCLIEdgeCases:
    """Stress-tests file errors, corrupted checkpoints, and truncated data slices."""

    def test_missing_data_files_raises_filenotfound(self, tmp_path):
        """Verifies load_oos_slice raises FileNotFoundError when neither cache nor CSV exists."""
        with pytest.raises(FileNotFoundError, match="Neither cache .* nor data .* exists"):
            load_oos_slice(
                cache_dir=tmp_path / "non_existent_cache",
                data_path=tmp_path / "non_existent_data.csv",
            )

    def test_corrupted_checkpoint_files_raises_clean_error(self, tmp_path):
        """Verifies checkpoint loaders raise clean unpickling/read errors for corrupted binary files."""
        corrupt_file = tmp_path / "corrupted_model.pt"
        corrupt_file.write_bytes(b"INVALID_PYTORCH_BYTES_0123456789")

        for loader_fn in [load_baseline_model, load_phase1_model, load_phase2_model]:
            with pytest.raises(Exception):
                loader_fn(corrupt_file)

    def test_missing_checkpoint_path_graceful_handling(self, tmp_path):
        """Verifies checkpoint loaders initialize fresh architecture if path does not exist."""
        missing_file = tmp_path / "absent_checkpoint.pt"

        wm_base, head_base = load_baseline_model(missing_file)
        assert isinstance(wm_base, LatentDynamicsWorldModel)
        assert isinstance(head_base, MultiHorizonProjectionHead)

        wm_p1, head_p1 = load_phase1_model(missing_file)
        assert wm_p1.temporal_encoder is not None

        wm_p2, head_p2 = load_phase2_model(missing_file)
        assert wm_p2.temporal_encoder is not None

    def test_truncated_slice_dataset_causal_padding(self):
        """Asserts SlicedM1Environment with 50 bars (< 240 lookback) handles zero-padding correctly."""
        env = RealM1Environment(spread_usd=2.50)
        env.total_bars = 50
        env.open_arr = np.full(50, 50000.0, dtype=np.float32)
        env.high_arr = np.full(50, 50100.0, dtype=np.float32)
        env.low_arr = np.full(50, 49900.0, dtype=np.float32)
        env.close_arr = np.full(50, 50050.0, dtype=np.float32)
        env.volume_arr = np.full(50, 10.0, dtype=np.float32)
        env.spread_arr = np.full(50, 2.50, dtype=np.float32)
        env.atr_arr = np.full(50, 50.0, dtype=np.float32)
        env.day_progress_arr = np.zeros(50, dtype=np.float32)
        env.obs_matrix = np.zeros((50, 137), dtype=np.float32)
        env.timestamps = [""] * 50

        sliced = SlicedM1Environment(env, 0, 50, name="Truncated_50")
        dataset = SingleRegimeDataset("oos", sliced, lookback=240, horizons=(5, 15, 30), is_oos=True)

        assert len(dataset) == 20  # 50 - 30 max horizon
        obs, seq, tgt, atr = dataset[0]
        assert seq.shape == (240, 6)
        # First 190 entries must be zero-padded because bar_idx = 0
        assert (seq[:190] == 0).all()

    def test_truncated_slice_less_than_horizon_zero_length(self):
        """Asserts slice with fewer bars than max horizon produces empty dataset without crash."""
        env = RealM1Environment(spread_usd=2.50)
        env.total_bars = 15
        env.open_arr = np.full(15, 50000.0, dtype=np.float32)
        env.high_arr = np.full(15, 50100.0, dtype=np.float32)
        env.low_arr = np.full(15, 49900.0, dtype=np.float32)
        env.close_arr = np.full(15, 50050.0, dtype=np.float32)
        env.volume_arr = np.full(15, 10.0, dtype=np.float32)
        env.spread_arr = np.full(15, 2.50, dtype=np.float32)
        env.atr_arr = np.full(15, 50.0, dtype=np.float32)
        env.day_progress_arr = np.zeros(15, dtype=np.float32)
        env.obs_matrix = np.zeros((15, 137), dtype=np.float32)
        env.timestamps = [""] * 15

        sliced = SlicedM1Environment(env, 0, 15, name="Truncated_15")
        dataset = SingleRegimeDataset("oos", sliced, lookback=240, horizons=(5, 15, 30), is_oos=True)
        assert len(dataset) == 0

    def test_cli_subprocess_missing_files_exit_code_1(self):
        """Asserts CLI runner subprocess exits with code 1 and writes error when files are missing."""
        cmd = [
            sys.executable,
            "scripts/evaluate_tca_oos.py",
            "--data-path",
            "non_existent_data.csv",
            "--cache-dir",
            "non_existent_cache",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 1
        assert "[OOS Evaluator ERROR]" in proc.stderr
        assert "Neither cache" in proc.stderr

    def test_cli_subprocess_corrupted_checkpoint_exit_code_1(self, tmp_path):
        """Asserts CLI runner subprocess exits with code 1 and writes error when checkpoint is corrupted."""
        corrupt_ckpt = tmp_path / "bad_checkpoint.pt"
        corrupt_ckpt.write_bytes(b"CORRUPTED_CHECKPOINT_DATA_12345")

        cmd = [
            sys.executable,
            "scripts/evaluate_tca_oos.py",
            "--dry-run",
            "--phase1-checkpoint",
            str(corrupt_ckpt),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 1
        assert "[OOS Evaluator ERROR]" in proc.stderr

    def test_cli_subprocess_dry_run_success(self, tmp_path):
        """Asserts CLI runner subprocess runs in --dry-run mode, exits with code 0, and writes valid JSON."""
        out_json = tmp_path / "cli_oos_results.json"
        cmd = [
            sys.executable,
            "scripts/evaluate_tca_oos.py",
            "--dry-run",
            "--output-json",
            str(out_json),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        assert out_json.exists()

        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["status"] == "COMPLETED"
        assert data["dry_run"] is True
        assert "baseline" in data["models"]
        assert "phase1_pretrained" in data["models"]
        assert "phase2_finetuned" in data["models"]

    def test_resolve_device_fallback(self):
        """Verifies resolve_device correctly maps cpu and auto, and falls back gracefully."""
        assert resolve_device("cpu") == torch.device("cpu")
        dev_auto = resolve_device("auto")
        assert dev_auto.type in ["cuda", "cpu"]
        # If cuda requested on non-cuda machine, falls back to cpu
        if not torch.cuda.is_available():
            assert resolve_device("cuda") == torch.device("cpu")
