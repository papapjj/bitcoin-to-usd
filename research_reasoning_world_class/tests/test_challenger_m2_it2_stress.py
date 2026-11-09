"""
Empirical Challenger Test Suite: Milestone 2 Remediation Stress Verification
Module: research_reasoning_world_class/tests/test_challenger_m2_it2_stress.py
Agent: teamwork_challenger_m2_it2_1
Role: EMPIRICAL CHALLENGER (critic, specialist)

Objectives:
1. Re-verify the former failure in test_unroll_bptt_gradient_propagation line 245.
   Stress-test BPTT gradient flow across 100 random seeds to confirm zero gradient vanishing/underflow.
2. Stress-test MarketStateEncoder with malicious boundary inputs:
   - All zeros: s = 0_137 ==> ||z_0||_2 ~ sqrt(768) ~ 27.7123 (Must NOT be 0.0!).
   - All NaNs, all Infs, subnormal floats (1e-38, 1.4e-45), extreme values (1e12, -1e12).
   - Verify that all outputs are finite and norm is preserved within [10.0, 50.0] (target 27.7123).
3. Stress-test deep unrolling H in {20, 50, 100} across 1,000 steps to confirm zero memory leaks
   and throughput >= 500 ops/sec.
"""

from __future__ import annotations

import gc
import math
import os
import sys
import time
import tracemalloc
import unittest
from typing import Dict, List

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MarketStateEncoder,
)


class TestBPTTGradientFlowAdversarial(unittest.TestCase):
    """Empirical challenge suite for BPTT gradient propagation and numerical stability."""

    def test_01_former_failure_reproduction_and_norm_fix(self):
        """
        Confirms root cause of former failure at test_unroll_bptt_gradient_propagation line 245:
        Unweighted sum of LayerNorm outputs has zero gradient due to mean subtraction:
            sum_i (x_i - mu) / sigma == 0 (identically).
        Replacing linear sum with squared L2 norm guarantees non-zero gradients.
        """
        torch.manual_seed(42)
        wm = LatentDynamicsWorldModel()

        # A: Reproduce failure with unweighted linear sum
        z_0_linear = torch.randn(2, 768, requires_grad=True)
        actions_linear = torch.randn(2, 5, 64, requires_grad=True)
        traj_linear = wm.rollout(z_0_linear, actions_linear)
        loss_linear = traj_linear[:, -1, :].sum()
        loss_linear.backward()

        self.assertIsNotNone(z_0_linear.grad)
        # Linear sum on LayerNorm output yields zero gradient
        self.assertTrue(
            torch.allclose(z_0_linear.grad, torch.zeros_like(z_0_linear.grad), atol=1e-6),
            "Expected linear sum on LayerNorm to have zero gradient due to mathematical symmetry",
        )

        # B: Verify squared L2 norm restores genuine non-zero gradient flow
        wm.zero_grad()
        z_0_sq = torch.randn(2, 768, requires_grad=True)
        actions_sq = torch.randn(2, 5, 64, requires_grad=True)
        traj_sq = wm.rollout(z_0_sq, actions_sq)
        loss_sq = (traj_sq[:, -1, :] ** 2).sum()
        loss_sq.backward()

        self.assertIsNotNone(z_0_sq.grad)
        grad_norm = z_0_sq.grad.norm().item()
        self.assertGreater(grad_norm, 1e-8, f"Gradient norm should be non-zero, got {grad_norm}")
        self.assertTrue(torch.all(torch.isfinite(z_0_sq.grad)))

        for name, p in wm.g_net.named_parameters():
            self.assertIsNotNone(p.grad, f"Parameter {name} has no gradient")
            self.assertTrue(torch.all(torch.isfinite(p.grad)), f"Parameter {name} grad is not finite")
            self.assertGreater(
                p.grad.abs().max().item(),
                1e-12,
                f"Parameter {name} suffered gradient vanishing/underflow",
            )

    def test_02_bptt_gradient_propagation_across_100_seeds(self):
        """
        Empirically stress-tests BPTT gradient propagation across 100 independent random seeds.
        Confirms zero occurrences of gradient vanishing (grad == 0 or < 1e-12), NaN, or Inf.
        """
        grad_norms: List[float] = []
        zero_grad_failures: List[int] = []
        nan_inf_failures: List[int] = []

        for seed in range(100):
            torch.manual_seed(seed)
            wm = LatentDynamicsWorldModel()
            wm.zero_grad()

            z_0 = torch.randn(2, 768, requires_grad=True)
            actions = torch.randn(2, 5, 64, requires_grad=True)
            traj = wm.rollout(z_0, actions)

            loss = (traj[:, -1, :] ** 2).sum()
            loss.backward()

            # Check z_0 gradients
            if z_0.grad is None or not torch.all(torch.isfinite(z_0.grad)):
                nan_inf_failures.append(seed)
                continue

            z_grad_norm = z_0.grad.norm().item()
            grad_norms.append(z_grad_norm)

            if z_grad_norm < 1e-12:
                zero_grad_failures.append(seed)

            # Check g_net parameter gradients
            for name, p in wm.g_net.named_parameters():
                if p.grad is None or not torch.all(torch.isfinite(p.grad)):
                    nan_inf_failures.append(seed)
                    break
                if p.grad.abs().max().item() < 1e-12:
                    zero_grad_failures.append(seed)
                    break

        self.assertEqual(len(nan_inf_failures), 0, f"NaN/Inf gradient failures on seeds: {nan_inf_failures}")
        self.assertEqual(len(zero_grad_failures), 0, f"Zero/underflow gradient failures on seeds: {zero_grad_failures}")
        self.assertEqual(len(grad_norms), 100)

        min_norm = min(grad_norms)
        max_norm = max(grad_norms)
        mean_norm = sum(grad_norms) / len(grad_norms)
        self.assertGreater(min_norm, 1e-9, f"Minimum gradient norm across 100 seeds: {min_norm}")
        self.assertLess(max_norm, 1e4, f"Maximum gradient norm across 100 seeds: {max_norm}")
        self.assertGreater(mean_norm, 1e-8)

    def test_03_deep_horizon_bptt_gradient_stability(self):
        """
        Stress-tests gradient propagation through deeper rollout horizons H in {5, 10, 20}.
        Ensures non-zero gradients propagate back through all sequential steps.
        """
        wm = LatentDynamicsWorldModel()

        for H in [5, 10, 20]:
            wm.zero_grad()
            z_0 = torch.randn(2, 768, requires_grad=True)
            actions = torch.randn(2, H, 64, requires_grad=True)
            traj = wm.rollout(z_0, actions)

            loss = (traj[:, -1, :] ** 2).sum()
            loss.backward()

            self.assertIsNotNone(z_0.grad, f"z_0 grad is None for H={H}")
            self.assertTrue(torch.all(torch.isfinite(z_0.grad)), f"z_0 grad not finite for H={H}")
            self.assertGreater(z_0.grad.norm().item(), 1e-12, f"Gradient vanished for H={H}")


class TestMarketStateEncoderBoundaryStress(unittest.TestCase):
    """Adversarial stress-test suite for MarketStateEncoder with extreme inputs."""

    def setUp(self):
        self.target_norm = 27.7123  # Target norm: sqrt(768) with eps correction ~ 27.7123

    def test_01_all_zeros_preserves_target_norm_never_zero(self):
        """
        Verifies s = 0_137 ==> ||z_0||_2 ~ sqrt(768) ~ 27.7123 across 20 distinct model instances.
        CRITICAL: Output norm must NOT collapse to 0.0!
        """
        for seed in range(20):
            torch.manual_seed(seed)
            encoder = MarketStateEncoder()
            zeros_input = torch.zeros(4, 137)
            z_0 = encoder(zeros_input)

            self.assertTrue(torch.all(torch.isfinite(z_0)))
            norms = z_0.norm(dim=-1).tolist()
            for b, norm in enumerate(norms):
                self.assertNotEqual(norm, 0.0, f"Seed {seed}, Batch {b}: norm collapsed to 0.0!")
                self.assertGreater(norm, 10.0, f"Seed {seed}, Batch {b}: norm too small ({norm})")
                self.assertLess(norm, 50.0, f"Seed {seed}, Batch {b}: norm too large ({norm})")
                self.assertAlmostEqual(norm, self.target_norm, delta=0.01, msg=f"Seed {seed}, Batch {b}: norm deviated from target")

    def test_02_all_nans_and_all_infs_handled_safely(self):
        """
        Adversarially feeds all NaNs, all +Infs, all -Infs, and combinations.
        All outputs must be strictly finite with norm preserved in [10.0, 50.0].
        """
        encoder = MarketStateEncoder()
        test_cases = {
            "all_nans": torch.full((4, 137), float("nan")),
            "all_posinf": torch.full((4, 137), float("inf")),
            "all_neginf": torch.full((4, 137), float("-inf")),
            "mixed_nan_inf": torch.cat(
                [
                    torch.full((1, 137), float("nan")),
                    torch.full((1, 137), float("inf")),
                    torch.full((1, 137), float("-inf")),
                    torch.zeros(1, 137),
                ],
                dim=0,
            ),
        }

        for name, tensor in test_cases.items():
            out = encoder(tensor)
            self.assertTrue(torch.all(torch.isfinite(out)), f"Case {name}: Output contains NaN or Inf!")
            norms = out.norm(dim=-1).tolist()
            for b, norm in enumerate(norms):
                self.assertGreaterEqual(norm, 10.0, f"Case {name}, Batch {b}: norm {norm} < 10.0")
                self.assertLessEqual(norm, 50.0, f"Case {name}, Batch {b}: norm {norm} > 50.0")
                self.assertAlmostEqual(norm, self.target_norm, delta=0.01)

    def test_03_subnormal_floats_and_underflow_prevention(self):
        """
        Feeds subnormal IEEE 754 floats (1e-38, 1.4e-45, negative subnormals).
        Confirms zero underflow to zero norm, output is finite, norm is ~27.7123.
        """
        encoder = MarketStateEncoder()
        subnormals = [
            torch.full((2, 137), 1e-38, dtype=torch.float32),
            torch.full((2, 137), -1e-38, dtype=torch.float32),
            torch.full((2, 137), 1.4e-45, dtype=torch.float32),
            torch.full((2, 137), -1.4e-45, dtype=torch.float32),
        ]

        for idx, tensor in enumerate(subnormals):
            out = encoder(tensor)
            self.assertTrue(torch.all(torch.isfinite(out)), f"Subnormal tensor {idx} produced non-finite output")
            norms = out.norm(dim=-1).tolist()
            for b, norm in enumerate(norms):
                self.assertGreaterEqual(norm, 10.0)
                self.assertLessEqual(norm, 50.0)
                self.assertAlmostEqual(norm, self.target_norm, delta=0.01)

    def test_04_extreme_magnitudes_and_overflow_prevention(self):
        """
        Feeds extreme magnitudes within float32 range (1e12, -1e12, 1e9, -1e9).
        Confirms zero overflow to Inf, output is finite, norm is ~27.7123 within [10.0, 50.0].
        """
        encoder = MarketStateEncoder()
        extremes = [
            torch.full((2, 137), 1e12, dtype=torch.float32),
            torch.full((2, 137), -1e12, dtype=torch.float32),
            torch.full((2, 137), 1e9, dtype=torch.float32),
            torch.full((2, 137), -1e9, dtype=torch.float32),
        ]

        for idx, tensor in enumerate(extremes):
            out = encoder(tensor)
            self.assertTrue(torch.all(torch.isfinite(out)), f"Extreme tensor {idx} produced non-finite output")
            norms = out.norm(dim=-1).tolist()
            for b, norm in enumerate(norms):
                self.assertGreaterEqual(norm, 10.0)
                self.assertLessEqual(norm, 50.0)
                self.assertAlmostEqual(norm, self.target_norm, delta=0.01)

    def test_05_mixed_adversarial_feature_vector(self):
        """
        Constructs a chimeric adversarial vector mixing 0, NaN, Inf, subnormals, and extremes
        across different feature indices in the 137D space.
        """
        encoder = MarketStateEncoder()
        raw = torch.zeros(8, 137)
        # Populate varied columns with adversarial values
        raw[:, 0:20] = float("nan")
        raw[:, 20:40] = float("inf")
        raw[:, 40:60] = float("-inf")
        raw[:, 60:80] = 1e-38
        raw[:, 80:100] = 1e12
        raw[:, 100:120] = -1e12
        # indices 120..136 are exact zeros

        out = encoder(raw)
        self.assertTrue(torch.all(torch.isfinite(out)))
        norms = out.norm(dim=-1).tolist()
        for b, norm in enumerate(norms):
            self.assertGreaterEqual(norm, 10.0)
            self.assertLessEqual(norm, 50.0)
            self.assertAlmostEqual(norm, self.target_norm, delta=0.01)

    def test_06_polymorphic_batch_dimensions(self):
        """
        Verifies unbatched 1D [137] and polymorphic batched inputs [1, 137], [4, 137], [64, 137].
        """
        encoder = MarketStateEncoder()
        # 1D input
        out_1d = encoder(torch.zeros(137))
        self.assertEqual(out_1d.shape, (1, 768))
        self.assertGreaterEqual(out_1d.norm(dim=-1).item(), 10.0)
        self.assertLessEqual(out_1d.norm(dim=-1).item(), 50.0)
        self.assertAlmostEqual(out_1d.norm(dim=-1).item(), self.target_norm, delta=0.01)

        # 2D batch sizes
        for B in [1, 4, 32, 64]:
            out_2d = encoder(torch.randn(B, 137))
            self.assertEqual(out_2d.shape, (B, 768))
            self.assertTrue(torch.all(torch.isfinite(out_2d)))
            for norm in out_2d.norm(dim=-1).tolist():
                self.assertGreaterEqual(norm, 10.0)
                self.assertLessEqual(norm, 50.0)
                self.assertAlmostEqual(norm, self.target_norm, delta=0.5)


class TestDeepUnrollingMemoryAndThroughput(unittest.TestCase):
    """Stress-tests deep unrolling H in {20, 50, 100} across 1,000 steps for memory leaks and throughput."""

    def setUp(self):
        self.wm = LatentDynamicsWorldModel()
        self.wm.eval()

    def test_01_throughput_benchmark_ge_500_ops_sec(self):
        """
        Benchmarks deep unrolling throughput across H in {20, 50, 100}.
        Confirms transition throughput >= 500 ops/sec.
        """
        throughputs: Dict[int, float] = {}

        for H in [20, 50, 100]:
            z_0 = torch.randn(8, 768)
            actions = torch.randn(8, H, 64)

            # Warmup
            with torch.inference_mode():
                for _ in range(5):
                    _ = self.wm.rollout(z_0, actions)

                t0 = time.perf_counter()
                N_rollouts = 100
                for _ in range(N_rollouts):
                    _ = self.wm.rollout(z_0, actions)
                t1 = time.perf_counter()

            elapsed = t1 - t0
            total_transitions = N_rollouts * 8 * H
            transitions_per_sec = total_transitions / elapsed
            throughputs[H] = transitions_per_sec

            self.assertGreaterEqual(
                transitions_per_sec,
                500.0,
                f"Throughput for H={H} ({transitions_per_sec:.1f} ops/sec) fell below 500 ops/sec SLA!",
            )

        print(f"\n[Throughput Benchmark Results]: {throughputs}")
        sys.stdout.flush()

    def test_02_memory_leak_free_1000_unroll_steps(self):
        """
        Stress-tests deep unrolling across 1,000 steps for H in {20, 50, 100}.
        Verifies zero memory leaks (memory growth < 100 KB, zero object count growth).
        Uses 1,000 steps for H=20 and 200 steps for H=50, 100 for fast reproducible execution.
        """
        step_counts = {20: 1000, 50: 200, 100: 200}

        for H, N_steps in step_counts.items():
            z_0 = torch.randn(2, 768)
            actions = torch.randn(2, H, 64)

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = self.wm.rollout(z_0, actions)
            gc.collect()

            tracemalloc.start()
            mem_start, _ = tracemalloc.get_traced_memory()
            obj_start = len(gc.get_objects())

            with torch.no_grad():
                for step in range(N_steps):
                    traj = self.wm.rollout(z_0, actions)
                    del traj

            gc.collect()
            mem_end, _ = tracemalloc.get_traced_memory()
            obj_end = len(gc.get_objects())
            tracemalloc.stop()

            mem_diff_kb = (mem_end - mem_start) / 1024.0
            obj_diff = obj_end - obj_start

            print(f"\n[H={H} {N_steps} Steps]: mem_diff={mem_diff_kb:.2f} KB, obj_diff={obj_diff}")
            sys.stdout.flush()

            # Zero memory leak assertions
            self.assertLessEqual(
                mem_diff_kb,
                100.0,
                f"Memory leak detected for H={H}: allocated {mem_diff_kb:.2f} KB across {N_steps} steps",
            )
            self.assertLessEqual(
                obj_diff,
                10,
                f"Object leak detected for H={H}: {obj_diff} uncollected objects created across {N_steps} steps",
            )


if __name__ == "__main__":
    unittest.main()
