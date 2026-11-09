"""
Empirical Naive Reference Oracle & Throughput Benchmarking Suite.
File: tests/test_empirical_challenger_m3_2.py
Milestone: Milestone 3 (Adaptive Training Runner & Zero-Drift Pipeline)
Agent: teamwork_challenger_tca_m3_2

Tests:
1. test_cosine_annealing_oracle_step_by_step_divergence:
   Independent analytical reference oracle for Linear Warmup + Cosine Annealing.
   Asserts maximum absolute divergence Delta <= 1e-6 across all steps.
2. test_stratified_regime_batch_routing_300_batches:
   Empirical statistical verification of MegaPoolRouter in STRATIFIED mode over 300 batches.
   Asserts regime proportions match 1/3 Bull, 1/3 Bear, 1/3 Chop within statistical margin
   and passes Chi-Square goodness-of-fit test (p > 0.01).
3. test_training_step_throughput_50_steps:
   Benchmarks training loop throughput across 50 consecutive steps.
4. test_resident_ram_memory_leak_100_steps:
   Profiles resident RAM (WorkingSetSize) across 100 consecutive training steps via Windows API.
   Asserts RAM < 1.0 GB and zero memory leak.
5. test_checkpoint_exactness_phase1_and_phase2:
   Inspects outputs/checkpoints/tca_phase1_pretrained.pt and tca_phase2_finetuned.pt.
   Verifies 4-namespace isolation and strict loadability into baseline LatentDynamicsWorldModel
   with exactly 0 missing and 0 unexpected keys.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple

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
)
from scripts.train_tca_multi_horizon import (
    build_lr_scheduler,
    build_phase1_optimizer,
    build_phase2_optimizer,
    execute_training_step,
    hash_reasoning_engine,
    set_seed,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "outputs" / "cache"
CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"


# =============================================================================
# WINDOWS RESIDENT RAM PROFILER
# =============================================================================

class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    ]


def get_resident_ram_mb() -> float:
    """Returns the current process resident RAM (Working Set Size) in megabytes."""
    pid = os.getpid()
    # PROCESS_QUERY_INFORMATION (0x0400) | PROCESS_VM_READ (0x0010) = 0x0410
    handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
    if not handle:
        return 0.0
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024.0 * 1024.0)
        return 0.0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


# =============================================================================
# INDEPENDENT ANALYTICAL ORACLE
# =============================================================================

def analytical_warmup_cosine_lr(
    step: int,
    base_lr: float,
    total_steps: int,
    warmup_ratio: float = 0.05,
    max_warmup_steps: int = 500,
    eta_min: float = 1e-6,
    start_factor: float = 0.1,
) -> float:
    """
    Independent analytical reference oracle for LinearLR warmup + CosineAnnealingLR.
    Evaluates learning rate at step `step` (0-indexed, where step 0 is before first step()).
    """
    if total_steps <= 2:
        # Fallback to pure cosine annealing
        return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * step / max(1, total_steps)))

    warmup_steps = min(max_warmup_steps, max(1, int(total_steps * warmup_ratio)))
    cosine_steps = max(1, total_steps - warmup_steps)

    if step < warmup_steps:
        # Linear warmup: factor increases linearly from start_factor to 1.0
        factor = start_factor + (1.0 - start_factor) * (step / warmup_steps)
        return base_lr * factor
    else:
        # Cosine annealing decay
        t_cos = step - warmup_steps
        return eta_min + 0.5 * (base_lr - eta_min) * (1.0 + math.cos(math.pi * t_cos / cosine_steps))


# =============================================================================
# EMPIRICAL CHALLENGER TEST SUITE
# =============================================================================

class TestEmpiricalChallengerM3_2:
    """Empirical Reference Oracle, Statistical Routing, & Throughput Benchmarking."""

    # -------------------------------------------------------------------------
    # Mission 1A: Independent Analytical LR Oracle vs PyTorch Scheduler
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize("total_steps", [20, 100, 300, 1000])
    @pytest.mark.parametrize("warmup_ratio", [0.05, 0.10])
    def test_cosine_annealing_oracle_step_by_step_divergence(
        self,
        total_steps: int,
        warmup_ratio: float,
    ):
        """
        Validates step-by-step equivalence between PyTorch SequentialLR scheduler
        and the independent analytical formula. Asserts max divergence Delta <= 1e-6.
        """
        m = nn.Linear(10, 10)
        base_lr_1 = 1e-4
        base_lr_2 = 3e-4
        eta_min = 1e-6

        opt = AdamW([
            {"params": m.weight, "lr": base_lr_1},
            {"params": m.bias, "lr": base_lr_2},
        ])
        sched = build_lr_scheduler(
            opt,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            eta_min=eta_min,
        )

        max_diff_g1 = 0.0
        max_diff_g2 = 0.0

        for step in range(total_steps):
            lr_actual_g1 = opt.param_groups[0]["lr"]
            lr_actual_g2 = opt.param_groups[1]["lr"]

            lr_expected_g1 = analytical_warmup_cosine_lr(
                step=step,
                base_lr=base_lr_1,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
                eta_min=eta_min,
            )
            lr_expected_g2 = analytical_warmup_cosine_lr(
                step=step,
                base_lr=base_lr_2,
                total_steps=total_steps,
                warmup_ratio=warmup_ratio,
                eta_min=eta_min,
            )

            diff_g1 = abs(lr_actual_g1 - lr_expected_g1)
            diff_g2 = abs(lr_actual_g2 - lr_expected_g2)

            if diff_g1 > max_diff_g1:
                max_diff_g1 = diff_g1
            if diff_g2 > max_diff_g2:
                max_diff_g2 = diff_g2

            # Assert strict divergence tolerance at each individual step
            assert diff_g1 <= 1e-6, f"Step {step} Group 1 divergence {diff_g1} > 1e-6"
            assert diff_g2 <= 1e-6, f"Step {step} Group 2 divergence {diff_g2} > 1e-6"

            opt.step()
            sched.step()

        # At the final step, learning rate must reach eta_min
        final_lr_g1 = opt.param_groups[0]["lr"]
        final_lr_g2 = opt.param_groups[1]["lr"]
        assert pytest.approx(final_lr_g1, abs=1e-7) == eta_min
        assert pytest.approx(final_lr_g2, abs=1e-7) == eta_min

        print(
            f"\n[Oracle Validation] Steps={total_steps}, Warmup={warmup_ratio} | "
            f"Max Delta G1: {max_diff_g1:.2e}, Max Delta G2: {max_diff_g2:.2e} (All <= 1e-6)"
        )

    # -------------------------------------------------------------------------
    # Mission 1B: Stratified Regime Batch Routing Empirical Verification (300 Batches)
    # -------------------------------------------------------------------------
    def test_stratified_regime_batch_routing_300_batches(self):
        """
        Samples 300 batches from MegaPoolRouter in STRATIFIED mode and verifies
        that regime proportions match 1/3 Bull, 1/3 Bear, and 1/3 Chop within
        statistical margin and Chi-Square goodness-of-fit test.
        """
        if not CACHE_DIR.exists():
            pytest.skip(f"Cache directory {CACHE_DIR} not found.")

        cfg = MegaPoolConfig(cache_dir=str(CACHE_DIR), use_cache=True)
        router = MegaPoolRouter(config=cfg)
        router.validate_zero_oos_leakage()

        num_batches = 300
        batch_size = 64
        total_samples = num_batches * batch_size  # 19,200 samples

        sampler = RegimeRoutingBatchSampler(
            datasets=router.training_datasets,
            batch_size=batch_size,
            num_batches=num_batches,
            mode=RoutingMode.STRATIFIED,
        )

        offsets = sampler.offsets
        bear_offset = offsets["bear"]
        chop_offset = offsets["chop"]
        total_len = len(router.mega_dataset)

        count_bull = 0
        count_bear = 0
        count_chop = 0

        batch_count = 0
        for batch_indices in sampler:
            batch_count += 1
            assert len(batch_indices) == batch_size, f"Batch size mismatch: {len(batch_indices)} != {batch_size}"

            for idx in batch_indices:
                assert 0 <= idx < total_len, f"Global index {idx} out of range [0, {total_len})"
                if idx < bear_offset:
                    count_bull += 1
                elif idx < chop_offset:
                    count_bear += 1
                else:
                    count_chop += 1

        assert batch_count == num_batches
        assert count_bull + count_bear + count_chop == total_samples

        prop_bull = count_bull / total_samples
        prop_bear = count_bear / total_samples
        prop_chop = count_chop / total_samples

        print(
            f"\n[Stratified Sampling - 300 Batches ({total_samples} samples)]\n"
            f"  Bull: {count_bull:5d} ({prop_bull * 100:.2f}%)\n"
            f"  Bear: {count_bear:5d} ({prop_bear * 100:.2f}%)\n"
            f"  Chop: {count_chop:5d} ({prop_chop * 100:.2f}%)"
        )

        # Expected proportion is 1/3 (0.33333)
        expected_prop = 1.0 / 3.0
        expected_count = total_samples / 3.0

        # Margin check: assert each proportion is within 1/3 +/- 0.02 (2.0%)
        margin = 0.02
        assert abs(prop_bull - expected_prop) <= margin, f"Bull proportion {prop_bull:.4f} outside margin"
        assert abs(prop_bear - expected_prop) <= margin, f"Bear proportion {prop_bear:.4f} outside margin"
        assert abs(prop_chop - expected_prop) <= margin, f"Chop proportion {prop_chop:.4f} outside margin"

        # Chi-Square Goodness of Fit Test
        # chi2 = sum (O_i - E_i)^2 / E_i
        chi2 = (
            (count_bull - expected_count) ** 2 / expected_count
            + (count_bear - expected_count) ** 2 / expected_count
            + (count_chop - expected_count) ** 2 / expected_count
        )
        # Critical value for df=2 at alpha=0.01 is 9.21; at alpha=0.05 is 5.99
        print(f"  Chi-Square Statistic: {chi2:.4f} (df=2, Critical alpha=0.01: 9.21)")
        assert chi2 < 9.21, f"Chi-Square test failed with statistic {chi2:.4f} >= 9.21"

    # -------------------------------------------------------------------------
    # Mission 2A: Throughput Benchmarking across 50 Steps
    # -------------------------------------------------------------------------
    def test_training_step_throughput_50_steps(self):
        """
        Benchmarks execution throughput (batches/second and samples/second)
        across 50 consecutive training steps on CPU.
        """
        set_seed(42)
        device = torch.device("cpu")

        if not CACHE_DIR.exists():
            pytest.skip(f"Cache directory {CACHE_DIR} not found.")

        cfg = MegaPoolConfig(cache_dir=str(CACHE_DIR), use_cache=True)
        router = MegaPoolRouter(config=cfg)

        wm = LatentDynamicsWorldModel().to(device)
        wm.enable_temporal_stream(candle_len=240)
        head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).to(device)

        # Trainable parameters (Phase 1 mode)
        trainable = list(wm.temporal_encoder.parameters()) + list(head.parameters())
        opt = build_phase1_optimizer(wm.temporal_encoder, head)
        sched = build_lr_scheduler(opt, total_steps=50)

        batch_size = 64
        num_steps = 50
        dataloader = router.create_dataloader(
            batch_size=batch_size, mode=RoutingMode.STRATIFIED, num_batches=num_steps
        )

        step_times: List[float] = []

        # Warm-up run (1 step)
        warmup_batch = next(iter(dataloader))
        execute_training_step(
            batch=warmup_batch,
            world_model=wm,
            head=head,
            optimizer=opt,
            scheduler=sched,
            trainable_params=trainable,
            device=device,
            beta=1.0,
        )

        # Benchmark 50 steps
        t0 = time.perf_counter()
        for step, batch in enumerate(dataloader):
            if step >= num_steps:
                break
            s0 = time.perf_counter()
            success, metrics = execute_training_step(
                batch=batch,
                world_model=wm,
                head=head,
                optimizer=opt,
                scheduler=sched,
                trainable_params=trainable,
                device=device,
                beta=1.0,
            )
            assert success is True
            step_times.append(time.perf_counter() - s0)

        total_elapsed = time.perf_counter() - t0
        batches_per_sec = len(step_times) / total_elapsed
        samples_per_sec = (len(step_times) * batch_size) / total_elapsed
        p50_ms = float(np.percentile(step_times, 50)) * 1000.0
        p95_ms = float(np.percentile(step_times, 95)) * 1000.0

        print(
            f"\n[Throughput Benchmark - 50 Steps]\n"
            f"  Total Duration   : {total_elapsed:.2f} s\n"
            f"  Throughput       : {batches_per_sec:.2f} batches/s | {samples_per_sec:.1f} samples/s\n"
            f"  Latency p50 / p95: {p50_ms:.1f} ms / {p95_ms:.1f} ms"
        )
        assert len(step_times) == 50, f"Expected 50 steps, got {len(step_times)}"
        assert batches_per_sec > 0.1, f"Throughput abnormally low: {batches_per_sec:.2f} batches/s"

    # -------------------------------------------------------------------------
    # Mission 2B: Resident RAM & Zero Memory Leak Profiling across 100 Steps
    # -------------------------------------------------------------------------
    def test_resident_ram_memory_leak_100_steps(self):
        """
        Profiles resident RAM across 100 consecutive training steps (batch size 16)
        in an isolated process to prevent test-session memory accumulation.
        Asserts resident RAM remains strictly bounded (< 1.0 GB = 1024 MB) with zero memory leaks.
        """
        import subprocess

        script_path = REPO_ROOT / "tests" / "profile_memory_isolated.py"
        assert script_path.exists(), f"Profile script {script_path} not found"

        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
        )

        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
            raise AssertionError(f"Isolated memory profiler failed with return code {result.returncode}")

        assert "bounded_under_1gb\": true" in result.stdout.lower(), (
            "Resident RAM violated < 1.0 GB threshold in isolated 100-step run"
        )
        assert "zero_memory_leak\": true" in result.stdout.lower(), (
            "Zero memory leak invariant violated in isolated 100-step run"
        )

    # -------------------------------------------------------------------------
    # Mission 3: Checkpoint Exactness Verification
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "checkpoint_filename",
        ["tca_phase1_pretrained.pt", "tca_phase2_finetuned.pt"],
    )
    def test_checkpoint_exactness_phase1_and_phase2(self, checkpoint_filename: str):
        """
        Inspects generated checkpoint files:
        - Asserts file exists and is non-empty.
        - Verifies 4 isolated namespaces: world_model (30 keys), temporal (41 keys),
          projection_head (8 keys), reasoning_engine (191 keys).
        - Strictly loads world_model_state_dict into baseline LatentDynamicsWorldModel
          with 0 missing and 0 unexpected keys.
        - Strictly loads reasoning_engine_state_dict and verifies SHA-256 matches authentic hash.
        """
        ckpt_path = CHECKPOINT_DIR / checkpoint_filename
        assert ckpt_path.exists(), f"Checkpoint {ckpt_path} does not exist!"
        assert ckpt_path.stat().st_size > 1_000_000, f"Checkpoint file too small ({ckpt_path.stat().st_size} bytes)"

        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # 1. 4-namespace structure
        expected_namespaces = [
            "world_model_state_dict",
            "temporal_encoder_state_dict",
            "projection_head_state_dict",
            "reasoning_engine_state_dict",
            "training_metadata",
        ]
        for ns in expected_namespaces:
            assert ns in payload, f"Missing namespace '{ns}' in checkpoint {checkpoint_filename}"

        wm_sd = payload["world_model_state_dict"]
        tca_sd = payload["temporal_encoder_state_dict"]
        head_sd = payload["projection_head_state_dict"]
        re_sd = payload["reasoning_engine_state_dict"]

        assert len(wm_sd) == 30, f"Expected 30 world model keys, got {len(wm_sd)}"
        assert len(tca_sd) == 41, f"Expected 41 temporal encoder keys, got {len(tca_sd)}"
        assert len(head_sd) == 8, f"Expected 8 projection head keys, got {len(head_sd)}"
        assert len(re_sd) == 191, f"Expected 191 reasoning engine keys, got {len(re_sd)}"

        # 2. Strict loadability into baseline LatentDynamicsWorldModel
        fresh_baseline_wm = LatentDynamicsWorldModel()
        assert len(fresh_baseline_wm.state_dict()) == 30
        res = fresh_baseline_wm.load_state_dict(wm_sd, strict=True)
        assert len(res.missing_keys) == 0, f"Missing keys in {checkpoint_filename}: {res.missing_keys}"
        assert len(res.unexpected_keys) == 0, f"Unexpected keys in {checkpoint_filename}: {res.unexpected_keys}"

        # 3. Strict loadability into MetacognitiveReasoningEngine & SHA-256 invariant
        fresh_re = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768))
        res_re = fresh_re.load_state_dict(re_sd, strict=True)
        assert len(res_re.missing_keys) == 0
        assert len(res_re.unexpected_keys) == 0

        re_hash = hash_reasoning_engine(fresh_re)
        expected_hash = "b69d48bd5993f9f989d466247a0242b5b317e96dae32769b84f1fa73b3b8a929"
        assert re_hash == expected_hash, (
            f"SHA-256 mismatch in {checkpoint_filename}: {re_hash} != {expected_hash}"
        )

        print(
            f"\n[Checkpoint Verified: {checkpoint_filename}]\n"
            f"  Size: {ckpt_path.stat().st_size / (1024*1024):.2f} MB\n"
            f"  World Model: 30 keys | Strict load: 0 missing / 0 unexpected keys\n"
            f"  Temporal Encoder: 41 keys\n"
            f"  Projection Head: 8 keys\n"
            f"  Reasoning Engine: 191 keys | SHA-256: {re_hash} (VERIFIED)"
        )
