"""
Unit and Integration Test Suite for TCA Multi-Horizon Adaptive Training Runner.
File: research_reasoning_world_class/tests/test_tca_training_runner.py
Milestone: Milestone 3 (Features 9, 10, 11, 12, 13)

Comprehensive 14-test suite validating:
- Test 1: CLI argument parsing with defaults and custom overrides.
- Test 2: Dry-run execution of Phase 1 and Phase 2 (--dry-run).
- Test 3: Optimizer parameter grouping and learning rates for Phase 1 (2 groups, 2,072,707 params).
- Test 4: Optimizer parameter grouping and learning rates for Phase 2 (3 groups, 2,539,907 params).
- Test 5: Cosine Annealing learning rate progression down to eta_min.
- Test 6: Base encoder freeze in Phase 1 (gradients strictly None).
- Test 7: Reasoning Engine 191-key freeze and zero gradient leakage.
- Test 8: Cryptographic SHA-256 hash invariant validation on Reasoning Engine.
- Test 9: Gradient norm clipping and stability under non-finite loss (dual firewall).
- Test 10: 4-namespace checkpoint export and file existence under outputs/checkpoints/.
- Test 11: Checkpoint roundtrip strict load into baseline 30-key LatentDynamicsWorldModel.
- Test 12: Stratified Mega-Pool batch routing distribution across Bull, Bear, and Chop.
- Test 13: Multi-horizon loss decomposition, MAE, and directional accuracy calculations.
- Test 14: Deterministic seeding repeatability across multiple runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
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

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHALLENGER_CKPT = REPO_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_specialized_challenger.pt"
CACHE_DIR = REPO_ROOT / "outputs" / "cache"


class TestTCATrainingRunner:
    """14 Institutional Unit & Integration Tests for Milestone 3 Training Pipeline."""

    # -------------------------------------------------------------------------
    # Test 1: CLI Argument Parsing Defaults and Overrides
    # -------------------------------------------------------------------------
    def test_01_cli_argument_parsing_defaults_and_overrides(self):
        # 1. Defaults verification
        defaults = parse_args([])
        assert defaults.phase1_epochs == 5
        assert defaults.phase2_epochs == 10
        assert defaults.batch_size == 64
        assert defaults.lr == 1e-4
        assert defaults.head_lr == 3e-4
        assert defaults.encoder_lr == 1e-5
        assert defaults.phase2_temporal_lr == 5e-5
        assert defaults.phase2_head_lr == 1e-4
        assert defaults.weight_decay == 1e-2
        assert defaults.clip_grad_norm == 1.0
        assert defaults.seed == 42
        assert defaults.output_dir == "outputs/checkpoints"
        assert defaults.checkpoint_interval == 1
        assert defaults.device == "auto"
        assert defaults.dry_run is False
        assert defaults.routing_mode == "stratified"

        # 2. Custom overrides verification
        custom_args = [
            "--phase1-epochs", "2",
            "--phase2-epochs", "4",
            "--batch-size", "128",
            "--lr", "2e-4",
            "--encoder-lr", "3e-5",
            "--seed", "999",
            "--output-dir", "custom/checkpoints",
            "--device", "cpu",
            "--dry-run",
            "--routing-mode", "round_robin",
        ]
        parsed = parse_args(custom_args)
        assert parsed.phase1_epochs == 1  # Adjusted by dry-run
        assert parsed.phase2_epochs == 1  # Adjusted by dry-run
        assert parsed.batch_size == 128
        assert parsed.lr == 2e-4
        assert parsed.encoder_lr == 3e-5
        assert parsed.seed == 999
        assert parsed.output_dir == "custom/checkpoints"
        assert parsed.device == "cpu"
        assert parsed.dry_run is True
        assert parsed.routing_mode == "round_robin"

    # -------------------------------------------------------------------------
    # Test 2: Dry-Run Execution of Phase 1 and Phase 2
    # -------------------------------------------------------------------------
    def test_02_dry_run_execution_phase1_and_phase2(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            args = parse_args([
                "--dry-run",
                "--batches-per-epoch", "2",
                "--output-dir", tmp_dir,
                "--device", "cpu",
                "--seed", "42",
            ])
            summary = run_training(args)

            assert summary["status"] == "COMPLETED"
            assert summary["re_drift_max"] == 0.0
            assert summary["re_initial_sha256"] == summary["re_final_sha256"]

            p1_ckpt = Path(tmp_dir) / "tca_phase1_pretrained.pt"
            p2_ckpt = Path(tmp_dir) / "tca_phase2_finetuned.pt"
            summary_file = Path(tmp_dir) / "training_logs.json"
            metrics_file = Path(tmp_dir) / "train_metrics.jsonl"

            assert p1_ckpt.exists() and p1_ckpt.stat().st_size > 1_000_000
            assert p2_ckpt.exists() and p2_ckpt.stat().st_size > 1_000_000
            assert summary_file.exists() and summary_file.stat().st_size > 0
            assert metrics_file.exists() and metrics_file.stat().st_size > 0

    # -------------------------------------------------------------------------
    # Test 3: Optimizer Parameter Grouping for Phase 1
    # -------------------------------------------------------------------------
    def test_03_phase1_optimizer_parameter_groups_and_trainable_count(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)

        # In Phase 1: Base encoder is frozen, TCA and Head are trainable
        for p in wm.encoder.parameters():
            p.requires_grad = False
        for p in wm.temporal_encoder.parameters():
            p.requires_grad = True
        for p in head.parameters():
            p.requires_grad = True

        opt = build_phase1_optimizer(
            wm.temporal_encoder,
            head,
            lr_temporal=1e-4,
            lr_head=3e-4,
            weight_decay=1e-2,
        )

        assert len(opt.param_groups) == 2
        g_tca = opt.param_groups[0]
        g_head = opt.param_groups[1]

        assert g_tca["name"] == "temporal_encoder"
        assert g_tca["lr"] == 1e-4
        assert g_tca["weight_decay"] == 1e-2
        assert len(g_tca["params"]) == 40  # 40 trainable tensors
        assert sum(p.numel() for p in g_tca["params"]) == 1842048

        assert g_head["name"] == "head"
        assert g_head["lr"] == 3e-4
        assert g_head["weight_decay"] == 1e-2
        assert len(g_head["params"]) == 8  # 8 parameter tensors
        assert sum(p.numel() for p in g_head["params"]) == 230659

        total_trainable = sum(p.numel() for p in g_tca["params"]) + sum(p.numel() for p in g_head["params"])
        assert total_trainable == 2072707

    # -------------------------------------------------------------------------
    # Test 4: Optimizer Parameter Grouping for Phase 2
    # -------------------------------------------------------------------------
    def test_04_phase2_optimizer_parameter_groups_and_trainable_count(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead(d_in=768, num_horizons=3)

        # In Phase 2: Base encoder, TCA, and Head are all trainable
        for p in wm.encoder.parameters():
            p.requires_grad = True
        for p in wm.temporal_encoder.parameters():
            p.requires_grad = True
        for p in head.parameters():
            p.requires_grad = True

        opt = build_phase2_optimizer(
            wm.encoder,
            wm.temporal_encoder,
            head,
            lr_encoder=1e-5,
            lr_temporal=5e-5,
            lr_head=1e-4,
            weight_decay=1e-2,
        )

        assert len(opt.param_groups) == 3
        g_enc = opt.param_groups[0]
        g_tca = opt.param_groups[1]
        g_head = opt.param_groups[2]

        assert g_enc["name"] == "encoder"
        assert g_enc["lr"] == 1e-5
        assert len(g_enc["params"]) == 8
        assert sum(p.numel() for p in g_enc["params"]) == 467200

        assert g_tca["name"] == "temporal_encoder"
        assert g_tca["lr"] == 5e-5
        assert len(g_tca["params"]) == 40
        assert sum(p.numel() for p in g_tca["params"]) == 1842048

        assert g_head["name"] == "head"
        assert g_head["lr"] == 1e-4
        assert len(g_head["params"]) == 8
        assert sum(p.numel() for p in g_head["params"]) == 230659

        total_trainable = (
            sum(p.numel() for p in g_enc["params"])
            + sum(p.numel() for p in g_tca["params"])
            + sum(p.numel() for p in g_head["params"])
        )
        assert total_trainable == 2539907

    # -------------------------------------------------------------------------
    # Test 5: Cosine Annealing Learning Rate Progression to eta_min
    # -------------------------------------------------------------------------
    def test_05_cosine_annealing_lr_progression_to_eta_min(self):
        m1, m2, m3 = nn.Linear(4, 4), nn.Linear(4, 4), nn.Linear(4, 4)
        opt = AdamW([
            {"params": m1.parameters(), "lr": 1e-5},
            {"params": m2.parameters(), "lr": 5e-5},
            {"params": m3.parameters(), "lr": 1e-4},
        ])
        total_steps = 100
        sched = build_lr_scheduler(opt, total_steps=total_steps, warmup_ratio=0.05, eta_min=1e-6)

        initial_lrs = [g["lr"] for g in opt.param_groups]
        assert initial_lrs[0] < 1e-5  # Starts at 10% base LR due to warmup
        assert initial_lrs[1] < 5e-5
        assert initial_lrs[2] < 1e-4

        # Run schedule to the end
        for _ in range(total_steps):
            opt.step()
            sched.step()

        final_lrs = [g["lr"] for g in opt.param_groups]
        for lr in final_lrs:
            assert pytest.approx(lr, abs=1e-7) == 1e-6

    # -------------------------------------------------------------------------
    # Test 6: Base Encoder Freeze in Phase 1 (Gradients None)
    # -------------------------------------------------------------------------
    def test_06_base_encoder_freeze_in_phase1_gradient_none(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead()

        # Freeze encoder, unfreeze temporal & head
        for p in wm.encoder.parameters():
            p.requires_grad = False
        for p in wm.temporal_encoder.parameters():
            p.requires_grad = True
        for p in head.parameters():
            p.requires_grad = True

        init_encoder_weights = [p.clone() for p in wm.encoder.parameters()]

        obs = torch.randn(4, 137)
        candles = torch.randn(4, 240, 6)
        targets = torch.randn(4, 3)

        z = wm.encode_state(obs, candle_seq=candles)
        preds = head(z)
        loss = F.smooth_l1_loss(preds, targets, beta=1.0)
        loss.backward()

        # Base encoder must receive zero gradients
        for p in wm.encoder.parameters():
            assert p.grad is None

        # Temporal encoder & head must receive valid gradients
        assert any(p.grad is not None for p in wm.temporal_encoder.parameters())
        assert any(p.grad is not None for p in head.parameters())

        # Weights must remain bitwise identical
        for p_curr, p_init in zip(wm.encoder.parameters(), init_encoder_weights):
            assert torch.equal(p_curr, p_init)

    # -------------------------------------------------------------------------
    # Test 7: Reasoning Engine 191-Key Freeze & Zero Gradient Leakage
    # -------------------------------------------------------------------------
    def test_07_reasoning_engine_191_key_freeze_zero_gradient_leakage(self):
        re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead()

        # Freeze reasoning engine
        re.eval()
        for p in re.parameters():
            p.requires_grad = False

        obs = torch.randn(2, 137)
        candles = torch.randn(2, 240, 6)
        z = wm.encode_state(obs, candle_seq=candles)
        preds = head(z)

        # Also pass through reasoning engine without gradient tracking
        with torch.no_grad():
            _ = re(z)

        loss = preds.sum()
        loss.backward()

        assert len(re.state_dict()) == 191
        assert sum(p.numel() for p in re.parameters()) == 4196564
        assert all(p.grad is None for p in re.parameters())

    # -------------------------------------------------------------------------
    # Test 8: Cryptographic SHA-256 Hash Invariant Validation on Reasoning Engine
    # -------------------------------------------------------------------------
    def test_08_reasoning_engine_sha256_hash_invariant_validation(self):
        re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
        h1 = hash_reasoning_engine(re)
        h2 = hash_reasoning_engine(re)
        assert h1 == h2, "SHA-256 calculation must be deterministic"

        # Check authentic challenger checkpoint hash if file exists
        if CHALLENGER_CKPT.exists():
            ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)
            re_state = ckpt.get("reasoning_engine_state_dict", ckpt.get("bull_cot_state_dict", {}))
            re.load_state_dict(re_state, strict=True)
            auth_hash = hash_reasoning_engine(re)
            assert auth_hash == "b69d48bd5993f9f989d466247a0242b5b317e96dae32769b84f1fa73b3b8a929"

        # Perturbation sensitivity test: modifying 1 float by 1e-7 flips hash
        with torch.no_grad():
            first_param = next(re.parameters())
            first_param[0, 0] += 1e-7
        h_perturbed = hash_reasoning_engine(re)
        assert h_perturbed != h1, "Hash must be sensitive to 1e-7 float perturbation"

    # -------------------------------------------------------------------------
    # Test 9: Gradient Norm Clipping & Stability under Non-Finite Loss
    # -------------------------------------------------------------------------
    def test_09_gradient_clipping_and_non_finite_loss_dual_firewall(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead()
        trainable = list(wm.temporal_encoder.parameters()) + list(head.parameters())
        opt = AdamW(trainable, lr=1e-3)

        # 1. Firewall 1: Non-finite loss is skipped gracefully
        class MockNaNBatch:
            obs_137 = torch.tensor([[float("nan")] * 137])
            candle_seq_240 = torch.zeros(1, 240, 6)
            targets_3 = torch.tensor([[float("nan"), 0.0, 0.0]])

        success, metrics = execute_training_step(
            batch=MockNaNBatch(),
            world_model=wm,
            head=head,
            optimizer=opt,
            scheduler=None,
            trainable_params=trainable,
            device=torch.device("cpu"),
            beta=1.0,
            max_grad_norm=1.0,
        )
        assert success is False
        assert metrics.get("skipped", 0.0) == 1.0

        # 2. Firewall 2: Valid inputs produce properly clipped gradient norm <= 1.0
        class MockValidBatch:
            obs_137 = torch.randn(4, 137) * 100.0
            candle_seq_240 = torch.randn(4, 240, 6) * 10.0
            targets_3 = torch.randn(4, 3) * 5.0

        success, metrics = execute_training_step(
            batch=MockValidBatch(),
            world_model=wm,
            head=head,
            optimizer=opt,
            scheduler=None,
            trainable_params=trainable,
            device=torch.device("cpu"),
            beta=1.0,
            max_grad_norm=1.0,
        )
        assert success is True
        assert metrics["grad_norm"] > 0.0
        # Post-clip gradient check
        current_norm = torch.norm(torch.stack([p.grad.norm() for p in trainable if p.grad is not None]))
        assert current_norm.item() <= 1.0001

    # -------------------------------------------------------------------------
    # Test 10: 4-Namespace Checkpoint Export and File Existence
    # -------------------------------------------------------------------------
    def test_10_four_namespace_checkpoint_export_and_file_existence(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead()
        re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_checkpoint.pt"
            meta = {"epoch": 1, "phase": "test", "val_loss": 0.543}

            payload = extract_4_namespace_checkpoint(wm, head, re, training_metadata=meta)
            assert "world_model_state_dict" in payload
            assert "temporal_encoder_state_dict" in payload
            assert "projection_head_state_dict" in payload
            assert "reasoning_engine_state_dict" in payload
            assert "training_metadata" in payload

            assert len(payload["world_model_state_dict"]) == 30
            assert len(payload["temporal_encoder_state_dict"]) == 41
            assert len(payload["projection_head_state_dict"]) == 8
            assert len(payload["reasoning_engine_state_dict"]) == 191

            save_atomic_checkpoint(payload, ckpt_path)
            assert ckpt_path.exists()
            assert ckpt_path.stat().st_size > 0

            # Roundtrip verification
            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            assert len(loaded["world_model_state_dict"]) == 30
            assert len(loaded["temporal_encoder_state_dict"]) == 41
            assert len(loaded["projection_head_state_dict"]) == 8
            assert len(loaded["reasoning_engine_state_dict"]) == 191
            assert loaded["training_metadata"]["phase"] == "test"

    # -------------------------------------------------------------------------
    # Test 11: Checkpoint Roundtrip Strict Load into Baseline 30-Key World Model
    # -------------------------------------------------------------------------
    def test_11_checkpoint_roundtrip_strict_load_baseline_30_keys(self):
        wm = LatentDynamicsWorldModel()
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead()
        re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))

        payload = extract_4_namespace_checkpoint(wm, head, re)
        base_wm_sd = payload["world_model_state_dict"]

        # An unaugmented LatentDynamicsWorldModel must load this with strict=True
        fresh_baseline_wm = LatentDynamicsWorldModel()
        assert len(fresh_baseline_wm.state_dict()) == 30

        result = fresh_baseline_wm.load_state_dict(base_wm_sd, strict=True)
        assert len(result.missing_keys) == 0
        assert len(result.unexpected_keys) == 0

    # -------------------------------------------------------------------------
    # Test 12: Stratified Mega-Pool Batch Routing Distribution
    # -------------------------------------------------------------------------
    def test_12_stratified_mega_pool_batch_routing_distribution(self):
        if not CACHE_DIR.exists():
            pytest.skip("outputs/cache directory not found")

        cfg = MegaPoolConfig(cache_dir=str(CACHE_DIR), use_cache=True)
        router = MegaPoolRouter(config=cfg)

        # Sample 10 stratified batches of size 60 (20 Bull, 20 Bear, 20 Chop expected)
        batch_size = 60
        sampler = RegimeRoutingBatchSampler(
            datasets=router.training_datasets,
            batch_size=batch_size,
            num_batches=10,
            mode=RoutingMode.STRATIFIED,
        )

        offsets = sampler.offsets
        names = list(router.training_datasets.keys())  # ['bull', 'bear', 'chop']
        bull_offset = offsets["bull"]
        bear_offset = offsets["bear"]
        chop_offset = offsets["chop"]
        bear_end = bear_offset + len(router.training_datasets["bear"])
        chop_end = chop_offset + len(router.training_datasets["chop"])

        total_bull, total_bear, total_chop = 0, 0, 0
        for batch_indices in sampler:
            assert len(batch_indices) == batch_size
            for idx in batch_indices:
                if idx < bear_offset:
                    total_bull += 1
                elif idx < chop_offset:
                    total_bear += 1
                else:
                    total_chop += 1

        total_samples = 10 * batch_size
        bull_frac = total_bull / total_samples
        bear_frac = total_bear / total_samples
        chop_frac = total_chop / total_samples

        # In stratified mode with equal probabilities, each regime represents ~33.3%
        assert 0.25 <= bull_frac <= 0.42
        assert 0.25 <= bear_frac <= 0.42
        assert 0.25 <= chop_frac <= 0.42

    # -------------------------------------------------------------------------
    # Test 13: Multi-Horizon Loss Decomposition, MAE, & Directional Accuracy
    # -------------------------------------------------------------------------
    def test_13_multi_horizon_loss_decomposition_mae_directional_accuracy(self):
        # Known synthetic predictions and targets
        # Preds:  [[ 1.0, -1.0,  0.5], [-0.5,  2.0, -1.0]]
        # Target: [[ 1.0,  1.0,  0.5], [ 0.5,  2.0, -1.0]]
        preds = torch.tensor([[1.0, -1.0, 0.5], [-0.5, 2.0, -1.0]])
        targets = torch.tensor([[1.0, 1.0, 0.5], [0.5, 2.0, -1.0]])

        loss_mat = F.smooth_l1_loss(preds, targets, beta=1.0, reduction="none")
        metrics = compute_metrics(preds, targets, loss_mat)

        # Horizons 5: pred=[1.0, -0.5], tgt=[1.0, 0.5].
        # Signs: [1.0, -1.0] vs [1.0, 1.0] -> 1 match out of 2 -> 50% DA
        assert metrics["da_5"] == 50.0

        # Horizons 15: pred=[-1.0, 2.0], tgt=[1.0, 2.0].
        # Signs: [-1.0, 1.0] vs [1.0, 1.0] -> 1 match out of 2 -> 50% DA
        assert metrics["da_15"] == 50.0

        # Horizons 30: pred=[0.5, -1.0], tgt=[0.5, -1.0].
        # Signs: [1.0, -1.0] vs [1.0, -1.0] -> 2 matches out of 2 -> 100% DA
        assert metrics["da_30"] == 100.0

        # Mean DA across all 6 elements: 4 matches / 6 = 66.67%
        assert pytest.approx(metrics["da_total"], rel=1e-3) == 66.6667

        # MAE:
        # col 0: (|1 - 1| + |-0.5 - 0.5|) / 2 = 1.0 / 2 = 0.5
        assert pytest.approx(metrics["mae_5"], rel=1e-4) == 0.5
        # col 1: (|-1 - 1| + |2 - 2|) / 2 = 2.0 / 2 = 1.0
        assert pytest.approx(metrics["mae_15"], rel=1e-4) == 1.0
        # col 2: (|0.5 - 0.5| + |-1 - -1|) / 2 = 0.0
        assert pytest.approx(metrics["mae_30"], rel=1e-4) == 0.0

    # -------------------------------------------------------------------------
    # Test 14: Deterministic Seeding Repeatability Across Runs
    # -------------------------------------------------------------------------
    def test_14_deterministic_seeding_repeatability(self):
        set_seed(12345)
        wm1 = LatentDynamicsWorldModel()
        wm1.enable_temporal_stream(candle_len=240)
        head1 = MultiHorizonProjectionHead()
        x1 = torch.randn(2, 137)
        c1 = torch.randn(2, 240, 6)
        with torch.no_grad():
            z1 = wm1.encode_state(x1, candle_seq=c1)
            pred1 = head1(z1)

        set_seed(12345)
        wm2 = LatentDynamicsWorldModel()
        wm2.enable_temporal_stream(candle_len=240)
        head2 = MultiHorizonProjectionHead()
        x2 = torch.randn(2, 137)
        c2 = torch.randn(2, 240, 6)
        with torch.no_grad():
            z2 = wm2.encode_state(x2, candle_seq=c2)
            pred2 = head2(z2)

        assert torch.equal(x1, x2)
        assert torch.equal(c1, c2)
        assert torch.equal(pred1, pred2)
