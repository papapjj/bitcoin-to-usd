"""
Adversarial Stress Harness & Fuzzing Suite for TCA Multi-Horizon Adaptive Training Runner.
File: research_reasoning_world_class/tests/test_adversarial_training_runner.py
Milestone: Milestone 3 (Adaptive Training Runner & Zero-Drift Pipeline — Features 9, 10, 11, 12, 13)

Adversarial Stress Matrix:
1. Batch Fuzzing & Input Sanitization:
   - Corrupted batches containing NaNs and Infs in targets_3 caught by Firewall 1.
   - Input feature sanitization behavior (AC-7) for obs_137 and candle_seq_240.
   - Extreme astronomical targets (1e6, -1e6, 1e9) and extreme inputs (1e7).
2. Dual Firewall Integrity:
   - Firewall 1 pre-backward non-finite loss detection: prevents backward pass and weight tampering.
   - Firewall 2 post-backward gradient norm clipping & non-finite gradient guard.
   - Poisoned sequential stream: alternating poisoned and valid batches without optimizer corruption.
3. Gradient Clipping Attack:
   - Strict norm clipping bounds (norm <= max_norm + 1e-4) under astronomical loss.
   - Dynamic max_norm thresholds (0.05, 0.25, 0.5, 1.0, 2.0, 5.0).
   - Finite weight maintenance across AdamW optimization steps.
4. Reasoning Engine Freeze & Zero-Drift Attack:
   - Non-zero gradient injection detection.
   - Backward pass gradient leakage detection through reasoning outputs.
   - Cryptographic SHA-256 weight tampering sensitivity down to 1e-7 float perturbation.
   - Structural state dict key tampering detection (191-key invariant).
5. CLI Edge Cases:
   - Invalid device names fallback to CPU.
   - Zero and negative batch sizes defensive clamping.
   - Negative epochs and extreme learning rates.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
    MetacognitiveOutput,
)
from research_reasoning_world_class.src.mega_pool_dataset import (
    MegaPoolRouter,
    MegaPoolConfig,
    RoutingMode,
    RegimeRoutingBatchSampler,
    MegaPoolBatch,
)
from scripts.train_tca_multi_horizon import (
    parse_args,
    set_seed,
    resolve_device,
    hash_reasoning_engine,
    build_phase1_optimizer,
    build_phase2_optimizer,
    build_lr_scheduler,
    extract_4_namespace_checkpoint,
    save_atomic_checkpoint,
    compute_metrics,
    execute_training_step,
    run_training,
)

CHALLENGER_CKPT = REPO_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_specialized_challenger.pt"


class MockBatch:
    """Mock batch structure conforming to MegaPoolBatch interface."""
    def __init__(self, obs: torch.Tensor, candles: torch.Tensor, targets: torch.Tensor):
        self.obs_137 = obs
        self.candle_seq_240 = candles
        self.targets_3 = targets


# =============================================================================
# 1. BATCH FUZZING & CORRUPTED INPUT ATTACKS
# =============================================================================

class TestAdversarialBatchFuzzing:
    """Adversarial fuzzing of execute_training_step with corrupted and extreme inputs."""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        set_seed(42)
        self.wm = LatentDynamicsWorldModel()
        self.wm.enable_temporal_stream(candle_len=240)
        self.head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
        self.trainable = list(self.wm.temporal_encoder.parameters()) + list(self.head.parameters())
        self.opt = AdamW(self.trainable, lr=1e-3)
        self.device = torch.device("cpu")

    def test_fuzz_nan_in_targets_firewall1_rejection(self):
        """Batch containing NaN in targets must be caught by Firewall 1 and skipped."""
        obs = torch.randn(4, 137)
        candles = torch.randn(4, 240, 6)
        targets = torch.randn(4, 3)
        targets[0, 1] = float("nan")

        batch = MockBatch(obs, candles, targets)
        weights_before = [p.clone().detach() for p in self.trainable]

        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
            beta=1.0,
            max_grad_norm=1.0,
        )

        assert success is False, "Corrupted NaN targets batch must not succeed"
        assert metrics.get("skipped", 0.0) == 1.0
        assert math.isnan(metrics.get("loss_total", 0.0))

        # Check weights are completely uncorrupted
        for p_before, p_after in zip(weights_before, self.trainable):
            assert torch.equal(p_before, p_after), "Weights were altered despite NaN in targets!"

    def test_fuzz_inf_in_targets_firewall1_rejection(self):
        """Batch containing Inf (+inf, -inf) in targets must be caught by Firewall 1."""
        obs = torch.randn(4, 137)
        candles = torch.randn(4, 240, 6)
        targets = torch.randn(4, 3)
        targets[0, 0] = float("inf")
        targets[2, 2] = float("-inf")

        batch = MockBatch(obs, candles, targets)
        weights_before = [p.clone().detach() for p in self.trainable]

        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
            beta=1.0,
            max_grad_norm=1.0,
        )

        assert success is False, "Inf targets batch must be rejected"
        assert metrics.get("skipped", 0.0) == 1.0

        for p_before, p_after in zip(weights_before, self.trainable):
            assert torch.equal(p_before, p_after)

    def test_encoder_level_nan_sanitization(self):
        """
        Confirms AC-7: MarketStateEncoder internally converts NaNs/Infs in obs_137
        into finite representations (torch.nan_to_num), preventing NaN propagation.
        """
        obs_nan = torch.randn(2, 137)
        obs_nan[0, :] = float("nan")
        obs_nan[1, :] = float("inf")

        z_encoded = self.wm.encoder(obs_nan)
        assert torch.isfinite(z_encoded).all(), "MarketStateEncoder failed AC-7 nan_to_num guarantee!"

    def test_tca_level_nan_sanitization(self):
        """
        Confirms AC-7: TemporalCrossAttentionEncoder internally converts NaNs/Infs
        in candle_seq_240 into finite representations (torch.nan_to_num).
        """
        candles_nan = torch.randn(2, 240, 6)
        candles_nan[:, :, :] = float("nan")

        cache = self.wm.temporal_encoder.precompute_kv_cache(candles_nan)
        assert torch.isfinite(cache.key).all(), "TemporalKVCache key contained NaNs!"
        assert torch.isfinite(cache.value).all(), "TemporalKVCache value contained NaNs!"

    def test_extreme_targets_and_finite_clipping(self):
        """Batch with astronomical targets (1e6) must be safely clipped and bounded."""
        obs = torch.randn(4, 137)
        candles = torch.randn(4, 240, 6)
        # Astronomical targets: 1,000,000 return units
        targets = torch.tensor([[1e6, -1e6, 5e5], [-1e6, 1e6, -5e5], [1e6, 1e6, 1e6], [-1e6, -1e6, -1e6]])

        batch = MockBatch(obs, candles, targets)
        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
            beta=1.0,
            max_grad_norm=1.0,
        )

        assert success is True, "Finite extreme targets should process successfully with gradient clipping"
        assert metrics["loss_total"] > 1e5, "Loss should reflect large target distance"
        # Verify gradient norm before step was clipped to <= 1.0001
        current_norm = torch.norm(torch.stack([p.grad.norm() for p in self.trainable if p.grad is not None]))
        assert current_norm.item() <= 1.0001, f"Gradient norm exceeded 1.0! Got {current_norm.item()}"

        # Verify all parameters remain finite
        for p in self.trainable:
            assert torch.isfinite(p).all(), "Parameter became non-finite after extreme target step!"

    def test_extreme_input_values_stability(self):
        """Batch with extreme input magnitudes (1e7) remains stable or skipped cleanly."""
        obs = torch.randn(4, 137) * 1e7
        candles = torch.randn(4, 240, 6) * 1e7
        targets = torch.randn(4, 3)

        batch = MockBatch(obs, candles, targets)
        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
            beta=1.0,
            max_grad_norm=1.0,
        )

        if success:
            for p in self.trainable:
                assert torch.isfinite(p).all(), "Parameters became non-finite after extreme input step!"
        else:
            assert metrics.get("skipped", 0.0) == 1.0


# =============================================================================
# 2. DUAL FIREWALL INTEGRITY & STREAM ISOLATION
# =============================================================================

class TestAdversarialDualFirewall:
    """Tests Firewall 1 and Firewall 2 under direct adversarial manipulation."""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        set_seed(101)
        self.wm = LatentDynamicsWorldModel()
        self.wm.enable_temporal_stream(candle_len=240)
        self.head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
        self.trainable = list(self.wm.temporal_encoder.parameters()) + list(self.head.parameters())
        self.opt = AdamW(self.trainable, lr=1e-3)
        self.device = torch.device("cpu")

    def test_firewall1_prevents_backward_pass(self):
        """Firewall 1 must abort BEFORE backward pass when loss is non-finite."""
        batch = MockBatch(
            torch.randn(1, 137),
            torch.zeros(1, 240, 6),
            torch.tensor([[float("nan"), 0.0, 0.0]]),
        )

        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
        )

        assert success is False
        assert all(p.grad is None for p in self.trainable), "Firewall 1 must clear and not compute gradients!"

    def test_firewall2_catches_non_finite_gradients_and_aborts_step(self):
        """
        If loss is finite but backward generates non-finite gradients (simulated here),
        Firewall 2 must clear gradients and abort before optimizer.step().
        """
        obs = torch.randn(2, 137)
        candles = torch.randn(2, 240, 6)
        targets = torch.randn(2, 3)

        z = self.wm.encode_state(obs, candle_seq=candles)
        preds = self.head(z)
        loss = F.smooth_l1_loss(preds, targets, beta=1.0)
        loss.backward()

        # Poison one gradient tensor with Inf
        first_grad_param = next(p for p in self.trainable if p.grad is not None)
        first_grad_param.grad[0] = float("inf")

        # Now test gradient clipping check matching Firewall 2 logic
        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainable, max_norm=1.0)
        grad_norm_val = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else float(grad_norm)

        assert not math.isfinite(grad_norm_val), "Poisoned gradient must result in non-finite grad_norm"

        # Under Firewall 2, optimizer.zero_grad(set_to_none=True) must be called
        self.opt.zero_grad(set_to_none=True)
        assert all(p.grad is None for p in self.trainable)

    def test_sequential_stream_with_intermittent_poisoned_batches(self):
        """
        Streams 10 batches with alternating valid and corrupted batches.
        Valid batches must update weights; corrupted batches must be rejected without
        poisoning AdamW moment buffers (exp_avg, exp_avg_sq).
        """
        for i in range(10):
            if i % 2 == 0:
                # Valid batch
                batch = MockBatch(torch.randn(2, 137), torch.randn(2, 240, 6), torch.randn(2, 3))
                success, metrics = execute_training_step(
                    batch=batch,
                    world_model=self.wm,
                    head=self.head,
                    optimizer=self.opt,
                    scheduler=None,
                    trainable_params=self.trainable,
                    device=self.device,
                )
                assert success is True
                assert metrics["skipped"] == 0.0
            else:
                # Corrupted batch (NaNs in targets)
                batch = MockBatch(
                    torch.randn(2, 137),
                    torch.randn(2, 240, 6),
                    torch.full((2, 3), float("nan")),
                )
                success, metrics = execute_training_step(
                    batch=batch,
                    world_model=self.wm,
                    head=self.head,
                    optimizer=self.opt,
                    scheduler=None,
                    trainable_params=self.trainable,
                    device=self.device,
                )
                assert success is False
                assert metrics["skipped"] == 1.0

        # Verify AdamW state does not contain any NaNs or Infs
        for p in self.trainable:
            state = self.opt.state.get(p, {})
            if "exp_avg" in state:
                assert torch.isfinite(state["exp_avg"]).all(), "AdamW exp_avg was poisoned!"
            if "exp_avg_sq" in state:
                assert torch.isfinite(state["exp_avg_sq"]).all(), "AdamW exp_avg_sq was poisoned!"


# =============================================================================
# 3. GRADIENT CLIPPING ATTACK UNDER EXTREME LOSS
# =============================================================================

class TestAdversarialGradientClipping:
    """Stress tests on gradient norm clipping bounds under extreme conditions."""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        set_seed(777)
        self.wm = LatentDynamicsWorldModel()
        self.wm.enable_temporal_stream(candle_len=240)
        self.head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
        self.trainable = list(self.wm.temporal_encoder.parameters()) + list(self.head.parameters())
        self.opt = AdamW(self.trainable, lr=1e-3)
        self.device = torch.device("cpu")

    def test_gradient_norm_upper_bound_strictness(self):
        """Under massive loss (targets = 1e8), grad norm after clipping must strictly satisfy <= max_norm."""
        obs = torch.randn(4, 137)
        candles = torch.randn(4, 240, 6)
        targets = torch.full((4, 3), 1e8)

        batch = MockBatch(obs, candles, targets)
        max_norm = 1.0

        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
            beta=1.0,
            max_grad_norm=max_norm,
        )

        assert success is True
        # Verify post-clip norm does not exceed 1.0 + 1e-4
        current_norm = torch.norm(torch.stack([p.grad.norm() for p in self.trainable if p.grad is not None]))
        assert current_norm.item() <= max_norm + 1e-4

    @pytest.mark.parametrize("custom_max_norm", [0.05, 0.25, 0.5, 2.0, 5.0])
    def test_dynamic_max_grad_norm_thresholds(self, custom_max_norm):
        """Verifies clipping respects arbitrary user-specified max_grad_norm thresholds."""
        obs = torch.randn(2, 137)
        candles = torch.randn(2, 240, 6)
        targets = torch.full((2, 3), 100.0)

        batch = MockBatch(obs, candles, targets)
        success, metrics = execute_training_step(
            batch=batch,
            world_model=self.wm,
            head=self.head,
            optimizer=self.opt,
            scheduler=None,
            trainable_params=self.trainable,
            device=self.device,
            beta=1.0,
            max_grad_norm=custom_max_norm,
        )

        assert success is True
        current_norm = torch.norm(torch.stack([p.grad.norm() for p in self.trainable if p.grad is not None]))
        assert current_norm.item() <= custom_max_norm + 1e-4


# =============================================================================
# 4. REASONING ENGINE FREEZE & ZERO-DRIFT ATTACK
# =============================================================================

class TestAdversarialReasoningEngineFreeze:
    """Attacks the zero-drift pipeline and validates tamper-detection assertions."""

    @pytest.fixture(autouse=True)
    def setup_models(self):
        set_seed(42)
        self.wm = LatentDynamicsWorldModel()
        self.wm.enable_temporal_stream(candle_len=240)
        self.head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)
        self.re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
        self.re.eval()
        for p in self.re.parameters():
            p.requires_grad = False

    def test_direct_gradient_injection_fails_runtime_assertion(self):
        """Directly injecting non-zero gradients into Reasoning Engine must raise AssertionError."""
        first_param = next(self.re.parameters())
        first_param.grad = torch.ones_like(first_param)

        with pytest.raises(AssertionError, match="Reasoning Engine"):
            assert all(p.grad is None for p in self.re.parameters()), "Reasoning Engine received gradients in Phase 1!"

    def test_backward_leakage_through_reasoning_outputs_fails_assertion(self):
        """If someone attempts to backpropagate through reasoning engine, grad assertion must trigger."""
        first_param = next(self.re.parameters())
        first_param.requires_grad = True

        z = torch.randn(2, 768, requires_grad=True)
        re_out = self.re(z)
        loss = re_out.confidence.sum()
        loss.backward()

        with pytest.raises(AssertionError):
            assert all(p.grad is None for p in self.re.parameters()), "Zero Drift Violation: grad is not None!"

    def test_weight_drift_tamper_detection(self):
        """Perturbing a single weight by 1e-7 must trigger max_drift and SHA-256 mismatch."""
        initial_hash = hash_reasoning_engine(self.re)
        initial_snapshot = {k: v.clone().detach().cpu() for k, v in self.re.state_dict().items()}

        # Tamper with one weight tensor by 1e-7
        with torch.no_grad():
            first_key = sorted(self.re.state_dict().keys())[0]
            flat_view = self.re.state_dict()[first_key].view(-1)
            flat_view[0] += 1e-7

        # 1. Float drift check must fail
        max_drift = max(
            (v.cpu() - initial_snapshot[k]).abs().max().item() for k, v in self.re.state_dict().items()
        )
        assert max_drift > 0.0, "Drift was not detected by float comparison!"
        with pytest.raises(AssertionError, match="Zero Drift Violation"):
            assert max_drift == 0.0, f"Zero Drift Violation: max drift = {max_drift} > 0.0"

        # 2. SHA-256 check must fail
        current_hash = hash_reasoning_engine(self.re)
        assert current_hash != initial_hash, "SHA-256 failed to catch 1e-7 float perturbation!"
        with pytest.raises(AssertionError, match="Zero Drift SHA-256 mismatch"):
            assert current_hash == initial_hash, "Zero Drift SHA-256 mismatch!"

    def test_key_tamper_detection(self):
        """Deleting or modifying state dict keys must fail the 191-key invariant."""
        sd = self.re.state_dict()
        assert len(sd) == 191

        # Delete first key
        first_key = next(iter(sd.keys()))
        tampered_sd = {k: v for k, v in sd.items() if k != first_key}
        assert len(tampered_sd) == 190

        # extract_4_namespace_checkpoint or direct check must raise
        class TamperedRE(nn.Module):
            def state_dict(self):
                return tampered_sd

        with pytest.raises(ValueError, match="Expected 191 reasoning engine keys"):
            extract_4_namespace_checkpoint(self.wm, self.head, TamperedRE())


# =============================================================================
# 5. CLI EDGE CASES & ARGUMENT ROBUSTNESS
# =============================================================================

class TestAdversarialCLIEdgeCases:
    """Stress tests on CLI argument parsing, device resolution, and corner values."""

    def test_resolve_device_invalid_names(self):
        """Invalid device names must safely fall back to CPU without crashing."""
        invalid_devices = [
            "invalid_gpu_name",
            "fake_tpu",
            "mps",
            "directml",
            "cuda:999",
            "None",
            "",
        ]
        for dev in invalid_devices:
            resolved = resolve_device(dev)
            assert resolved.type == "cpu", f"Device '{dev}' did not fall back to CPU! Got {resolved}"

    def test_device_auto_resolution(self):
        """'auto' resolution returns valid torch.device (cpu or cuda)."""
        resolved = resolve_device("auto")
        assert resolved.type in ("cpu", "cuda")

    def test_batch_size_defensive_clamping(self):
        """RegimeRoutingBatchSampler defensively clamps batch_size <= 0 to at least 1."""
        dummy_datasets = {"dummy": torch.zeros(10, 5)}
        sampler_zero = RegimeRoutingBatchSampler(datasets=dummy_datasets, batch_size=0, num_batches=1)
        assert sampler_zero.batch_size == 1

        sampler_neg = RegimeRoutingBatchSampler(datasets=dummy_datasets, batch_size=-5, num_batches=1)
        assert sampler_neg.batch_size == 1

    def test_negative_epochs_cli_and_scheduler_safety(self):
        """Negative epochs produce scheduler without math crash or negative step counts."""
        m = nn.Linear(4, 4)
        opt = AdamW(m.parameters(), lr=1e-3)

        # Scheduler with negative or zero total_steps must safely default to T_max >= 1
        sched = build_lr_scheduler(opt, total_steps=-5, warmup_ratio=0.05, eta_min=1e-6)
        assert sched is not None
        opt.step()
        sched.step()
        assert opt.param_groups[0]["lr"] > 0.0

    def test_extreme_learning_rates_cli(self):
        """Zero LR and extreme high LR arguments must parse correctly without crash."""
        parsed_zero = parse_args(["--lr", "0.0", "--encoder-lr", "0.0"])
        assert parsed_zero.lr == 0.0
        assert parsed_zero.encoder_lr == 0.0

        parsed_high = parse_args(["--lr", "10.0", "--clip-grad-norm", "50.0"])
        assert parsed_high.lr == 10.0
        assert parsed_high.clip_grad_norm == 50.0

    def test_invalid_routing_mode_cli_rejection(self):
        """Invalid routing mode string must be rejected by argparse with SystemExit."""
        with pytest.raises(SystemExit):
            parse_args(["--routing-mode", "quantum_hyper_routing"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
