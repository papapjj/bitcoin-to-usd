"""
Unit and Integration Tests for Feature 14: Out-of-Sample (OOS) Crash Slice Evaluation Runner.
Verifies:
1. Crash slice dimensions [55000:64785] (9,785 bars, 9,755 valid evaluation steps)
2. Metric formulation accuracy (Directional Accuracy %, MAE, Huber Loss)
3. Spread-adjusted trading simulation with $2.50 spread friction
4. Multi-model loading (Baseline 30 keys, Phase 1, Phase 2)
5. Dry-run execution speed and JSON schema compliance
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MultiHorizonProjectionHead,
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


class TestOOSDataLoader:
    """Verifies dataset boundaries and sequence validity on February 2025 crash slice."""

    def test_oos_slice_dimensions(self):
        dataset, sliced_env, master_env = load_oos_slice(dry_run=False)
        assert sliced_env.total_bars == 9785
        assert len(sliced_env.close_arr) == 9785

    def test_oos_valid_evaluation_steps(self):
        dataset, sliced_env, _ = load_oos_slice(dry_run=False)
        # 9,785 bars - max_horizon (30) = 9,755 valid samples
        assert len(dataset) == 9755

    def test_oos_dry_run_sample_count(self):
        dataset, _, _ = load_oos_slice(dry_run=True, dry_run_samples=200)
        assert len(dataset) == 200

    def test_oos_sample_tensor_shapes(self):
        dataset, _, _ = load_oos_slice(dry_run=True, dry_run_samples=5)
        obs_137, candle_seq_240, targets_3, atr_ref = dataset[0]
        assert obs_137.shape == (137,)
        assert candle_seq_240.shape == (240, 6)
        assert targets_3.shape == (3,)
        assert atr_ref.dim() == 0
        assert not torch.isnan(obs_137).any()
        assert not torch.isnan(candle_seq_240).any()
        assert not torch.isnan(targets_3).any()
        assert not torch.isnan(atr_ref).any()


class TestMetricFormulations:
    """Verifies exact calculation of multi-horizon metrics and trading simulation."""

    def test_directional_accuracy_computation(self):
        # Synthetic predictions and targets
        preds = np.array([
            [1.0, -1.0, 0.5],
            [-0.5, 2.0, -0.2],
            [0.1, -0.1, 0.0],
            [-1.0, -2.0, -3.0],
        ])
        targets = np.array([
            [0.5, -0.2, -0.5],  # match, match, mismatch
            [0.2, 1.0, -0.1],   # mismatch, match, match
            [0.8, -0.5, 0.0],   # match, match, match (0 vs 0)
            [-0.4, 1.0, -1.0],  # match, mismatch, match
        ])
        # Horizon 0 matches: 3/4 = 75.0%
        # Horizon 1 matches: 3/4 = 75.0%
        # Horizon 2 matches: 3/4 = 75.0%
        # Total mean: 75.0%
        res = compute_horizon_metrics(preds, targets)
        da = res["directional_accuracy_pct"]
        assert da["h5"] == 75.0
        assert da["h15"] == 75.0
        assert da["h30"] == 75.0
        assert da["total"] == 75.0

    def test_mae_and_huber_computation(self):
        preds = np.array([[1.0, 2.0, 3.0]])
        targets = np.array([[0.5, 1.0, 0.0]])
        # diffs: [0.5, 1.0, 3.0]
        # mae: [0.5, 1.0, 3.0], mean = 1.5
        # huber:
        # diff 0.5: < 1.0 -> 0.5 * (0.5^2) = 0.125
        # diff 1.0: >= 1.0 -> 1.0 - 0.5 = 0.5
        # diff 3.0: >= 1.0 -> 3.0 - 0.5 = 2.5
        # mean huber = (0.125 + 0.5 + 2.5) / 3 = 1.0417
        res = compute_horizon_metrics(preds, targets)
        assert res["mae"]["h5"] == 0.5
        assert res["mae"]["h15"] == 1.0
        assert res["mae"]["h30"] == 3.0
        assert res["mae"]["total"] == 1.5
        assert res["loss"]["h5"] == 0.125
        assert res["loss"]["h15"] == 0.5
        assert res["loss"]["h30"] == 2.5
        assert round(res["loss"]["total"], 4) == 1.0417

    def test_spread_adjusted_simulation_math(self):
        # 3 trade signals at horizon t+15: threshold = 0.20
        preds = np.array([
            [0.0, 0.5, 0.0],   # BUY signal (0.5 > 0.20)
            [0.0, -0.4, 0.0],  # SELL signal (-0.4 < -0.20)
            [0.0, 0.1, 0.0],   # NO TRADE (0.1 < 0.20)
            [0.0, 0.6, 0.0],   # BUY signal (0.6 > 0.20)
        ])
        targets = np.array([
            [0.0, 1.0, 0.0],   # target move = +1.0 ATR
            [0.0, -0.5, 0.0],  # target move = -0.5 ATR
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],  # target move = -1.0 ATR
        ])
        closes = np.array([50000.0, 50000.0, 50000.0, 50000.0])
        atrs = np.array([10.0, 10.0, 10.0, 10.0])  # ATR = $10
        spread_usd = 2.50

        # Trade 1 (BUY): gross = 1.0 * 10 = +$10, net = 10 - 2.5 = +$7.50 (WIN)
        # Trade 2 (SELL): gross = -(-0.5) * 10 = +$5, net = 5 - 2.5 = +$2.50 (WIN)
        # Trade 3: NO TRADE
        # Trade 4 (BUY): gross = -1.0 * 10 = -$10, net = -10 - 2.5 = -$12.50 (LOSS)
        # Total trades = 3
        # Wins = 2, Losses = 1, Win Rate = 66.67%
        # Gross profit = 7.5 + 2.5 = $10.0
        # Gross loss = $12.50
        # Profit factor = 10.0 / 12.50 = 0.80
        # Net PnL = 7.5 + 2.5 - 12.50 = -$2.50
        sim = simulate_spread_adjusted_trading(
            preds, targets, closes, atrs, spread_usd=spread_usd, threshold=0.20
        )
        assert sim["total_trades"] == 3
        assert sim["win_rate_pct"] == 66.67
        assert sim["profit_factor"] == 0.8
        assert sim["gross_profit_usd"] == 10.0
        assert sim["gross_loss_usd"] == 12.50
        assert sim["net_pnl_usd"] == -2.50
        assert sim["spread_usd"] == 2.50


class TestModelLoading:
    """Verifies checkpoints loading without errors."""

    def test_load_baseline_model_structure(self):
        wm, head = load_baseline_model(
            "outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt"
        )
        assert isinstance(wm, LatentDynamicsWorldModel)
        assert isinstance(head, MultiHorizonProjectionHead)
        # Baseline WM has no temporal stream
        assert wm.temporal_encoder is None
        clean_keys = [k for k in wm.state_dict().keys() if not k.startswith("temporal_encoder.")]
        assert len(clean_keys) == 30

    def test_load_phase1_model_structure(self):
        wm, head = load_phase1_model("outputs/checkpoints/tca_phase1_pretrained.pt")
        assert wm.temporal_encoder is not None
        assert len(wm.temporal_encoder.state_dict()) == 41
        assert len(head.state_dict()) == 8

    def test_load_phase2_model_structure(self):
        wm, head = load_phase2_model("outputs/checkpoints/tca_phase2_finetuned.pt")
        assert wm.temporal_encoder is not None
        assert len(wm.temporal_encoder.state_dict()) == 41
        assert len(head.state_dict()) == 8

    def test_resolve_device_graceful(self):
        dev = resolve_device("cpu")
        assert dev == torch.device("cpu")
        dev_auto = resolve_device("auto")
        assert dev_auto.type in ["cuda", "cpu"]


class TestEvaluatorExecution:
    """Verifies end-to-end dry-run execution and output schema."""

    def test_evaluate_all_checkpoints_dry_run(self, tmp_path):
        out_json = tmp_path / "test_oos_results.json"
        res = evaluate_all_checkpoints(
            output_json=out_json,
            dry_run=True,
            dry_run_samples=50,
            batch_size=64,
            spread_usd=2.50,
        )

        assert res["status"] == "COMPLETED"
        assert res["dry_run"] is True
        assert res["dataset"]["evaluated_samples"] == 50
        assert "baseline" in res["models"]
        assert "phase1_pretrained" in res["models"]
        assert "phase2_finetuned" in res["models"]

        for model_key in ["baseline", "phase1_pretrained", "phase2_finetuned"]:
            m = res["models"][model_key]
            assert "loss" in m and "total" in m["loss"]
            assert "mae" in m and "total" in m["mae"]
            assert "directional_accuracy_pct" in m and "total" in m["directional_accuracy_pct"]
            assert "simulation" in m
            assert m["simulation"]["spread_usd"] == 2.50
            assert "net_pnl_usd" in m["simulation"]

        assert out_json.exists()
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["dataset"]["total_bars"] == 9785
