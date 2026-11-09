"""
Challenger Empirical Stress and False-Positive Benchmark (Milestone M1 Iteration 2)
Agent: teamwork_preview_challenger_m1_it2_2

Empirically stress-tests:
1. False-positive trigger prevention across >= 1,000 concordant input pairs:
   - Does backtracking trigger falsely (was_backtracked == True) when thoughts agree? Must be 0 / 1000 (0.00%).
   - Detailed distribution of tau_trap, directional_conflict, and revision magnitude.
2. Single-tick reasoning CPU inference latency profiling across >= 1,000 ticks:
   - P50, P90, P95, P99, Mean, Max latency profile on CPU.
   - SLA verification (<= 50.0 ms).
3. Strict N_backtrack <= 1 invariant enforcement:
   - Instrumented backtrack execution across 1,000 passes under adversarial trap pressures.
   - Max backtrack executions per tick must be strictly <= 1.
"""

import math
import os
import sys
import time
from typing import Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MT5 deterministic single-thread convention
torch.set_num_threads(1)

# Ensure project and repository roots are in sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.metacognitive_reasoning import (
    ReasoningConfig,
    BacktrackingRevisionModule,
    ContradictionDetector,
    CoTReasoningHead,
    MetacognitiveReasoningEngine,
)


class TestEmpiricalChallengerM1It2Stress:
    """Empirical Stress and False-Positive Benchmark for M1 Iteration 2."""

    def test_1000_concordant_pairs_false_positive_rate(self):
        """
        Stress-tests 1,000 concordant input pairs across multiple random seeds and configurations.
        Verifies whether was_backtracked is EVER True when thoughts agree.
        """
        torch.manual_seed(42)
        hidden_dim = 256
        smc_dim = 64
        backtracker = BacktrackingRevisionModule(
            hidden_dim=hidden_dim, smc_dim=smc_dim, trap_threshold=0.65
        ).eval()

        N_PAIRS = 1000
        false_positive_count = 0
        tau_traps: List[float] = []
        conflicts: List[bool] = []
        magnitudes: List[float] = []

        torch.manual_seed(20261024)
        with torch.no_grad():
            for i in range(N_PAIRS):
                # Sample random base premise h1
                h1 = torch.randn(1, hidden_dim)

                # Generate concordant thought: positively aligned with h1
                # Mix varying positive multipliers and small noise
                scale = float(torch.empty(1).uniform_(0.3, 3.0).item())
                noise_scale = float(torch.empty(1).uniform_(0.01, 0.10).item())
                hk_concordant = h1 * scale + torch.randn(1, hidden_dim) * noise_scale

                # Benign market context (smc = 0)
                smc_benign = torch.zeros(1, smc_dim)

                # Execute backtracking revision module
                rev_h1, was_bt, mag = backtracker(h1, hk_concordant, smc_benign)

                # Record internal signals
                trap_in = torch.cat([h1, hk_concordant, smc_benign], dim=-1)
                tau = float(backtracker.detector(trap_in).item())
                dir1 = backtracker.contradiction_detector.dir_h1(h1).squeeze(-1)
                dirk = backtracker.contradiction_detector.dir_hk(hk_concordant).squeeze(-1)
                conflict = bool(((dir1 * dirk) < 0.0).item())

                tau_traps.append(tau)
                conflicts.append(conflict)
                magnitudes.append(mag)

                if was_bt:
                    false_positive_count += 1

        tau_arr = np.array(tau_traps)
        conflict_arr = np.array(conflicts)
        fp_rate = (false_positive_count / N_PAIRS) * 100.0

        print("\n" + "=" * 80)
        print("EMPIRICAL CHALLENGER: 1,000 CONCORDANT PAIRS FALSE-POSITIVE STRESS TEST")
        print("=" * 80)
        print(f"Total Evaluated Concordant Pairs: {N_PAIRS}")
        print(f"False Backtracking Triggers (was_backtracked == True): {false_positive_count} / {N_PAIRS} ({fp_rate:.2f}%)")
        print(f"Directional Conflict Count: {np.sum(conflict_arr)} / {N_PAIRS}")
        print("-" * 80)
        print("Raw tau_trap Detector Distribution:")
        print(f"  Min:  {np.min(tau_arr):.4f}")
        print(f"  Mean: {np.mean(tau_arr):.4f}")
        print(f"  P50:  {np.percentile(tau_arr, 50):.4f}")
        print(f"  P95:  {np.percentile(tau_arr, 95):.4f}")
        print(f"  P99:  {np.percentile(tau_arr, 99):.4f}")
        print(f"  Max:  {np.max(tau_arr):.4f}")
        print(f"  Fraction tau_trap > 0.65: {np.sum(tau_arr > 0.65)} / {N_PAIRS} ({(np.sum(tau_arr > 0.65) / N_PAIRS) * 100:.2f}%)")
        print("=" * 80)

        # Invariant 1: False positive rate on concordant thoughts MUST be exactly 0.00%
        assert false_positive_count == 0, (
            f"VIOLATION: False positive backtracking triggered {false_positive_count}/{N_PAIRS} times!"
        )
        assert max(magnitudes) == 0.0, f"Revision magnitude {max(magnitudes)} > 0 on concordant thoughts!"

    def test_single_tick_cpu_latency_profile_1000_ticks(self):
        """
        Executes 1,000 consecutive single-tick CPU inference passes.
        Measures P50, P90, P95, P99, Mean, Max latency profile.
        Strictly verifies latency SLA (<= 50.0 ms).
        """
        torch.manual_seed(42)
        config = ReasoningConfig(
            input_dim=256,
            hidden_dim=256,
            num_hops=4,
            max_hops=8,
            min_hops=2,
            enable_cross_hop=True,
            enable_backtrack=True,
            enable_act=True,
        )
        engine = MetacognitiveReasoningEngine(config).eval()

        # Warm-up phase (10 ticks)
        with torch.no_grad():
            for _ in range(10):
                x_warm = torch.randn(1, 256)
                u_warm = torch.tensor([0.5])
                _ = engine(x_warm, uncertainty=u_warm)

        N_TICKS = 1000
        latencies_ms: List[float] = []

        with torch.no_grad():
            for tick in range(N_TICKS):
                # Cycle through realistic market volatility / uncertainty states
                # Calm sideway: 0.05, Moderate: 0.40, High volatility trap: 0.95
                cycle = tick % 5
                if cycle == 0:
                    sigma = 0.05
                elif cycle == 1:
                    sigma = 0.20
                elif cycle == 2:
                    sigma = 0.50
                elif cycle == 3:
                    sigma = 0.85
                else:
                    sigma = 0.98

                x = torch.randn(1, 256)
                u = torch.tensor([sigma])

                t0 = time.perf_counter()
                out = engine(x, uncertainty=u)
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(elapsed_ms)

                assert not torch.isnan(out.confidence).any(), f"NaN confidence at tick {tick}"

        lat_arr = np.array(latencies_ms)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        max_lat = float(np.max(lat_arr))
        min_lat = float(np.min(lat_arr))

        # Check drift across first 200 vs last 200
        first_200 = float(np.mean(lat_arr[:200]))
        last_200 = float(np.mean(lat_arr[-200:]))
        drift_ratio = last_200 / (first_200 + 1e-6)

        print("\n" + "=" * 80)
        print("EMPIRICAL CHALLENGER: 1,000 SINGLE-TICK CPU LATENCY PROFILE (AC-6 SLA)")
        print("=" * 80)
        print(f"Total Evaluated Ticks: {N_TICKS}")
        print(f"Min Latency:   {min_lat:.3f} ms")
        print(f"Mean Latency:  {mean_lat:.3f} ms")
        print(f"Median (P50):  {p50:.3f} ms")
        print(f"P90 Latency:   {p90:.3f} ms")
        print(f"P95 Latency:   {p95:.3f} ms (SLA <= 50.0 ms)")
        print(f"P99 Latency:   {p99:.3f} ms")
        print(f"Max Latency:   {max_lat:.3f} ms")
        print(f"First 200 Mean: {first_200:.3f} ms | Last 200 Mean: {last_200:.3f} ms | Drift: {drift_ratio:.2f}x")
        print("=" * 80)

        # SLA Assertions
        assert p95 <= 50.0, f"VIOLATION: P95 latency {p95:.2f} ms exceeds SLA 50.0 ms!"
        assert mean_lat <= 30.0, f"VIOLATION: Mean latency {mean_lat:.2f} ms exceeds 30.0 ms budget!"
        assert drift_ratio < 2.0, f"VIOLATION: Latency degradation detected (drift {drift_ratio:.2f}x >= 2.0x)!"

    def test_strict_n_backtrack_le_1_invariant_across_1000_passes(self):
        """
        Stress-tests the strict N_backtrack <= 1 invariant across 1,000 passes
        with adversarial conditions (forcing trap confidence = 0.99 and opposing directions).
        Verifies that backtrack execution NEVER exceeds 1 per tick pass.
        """
        torch.manual_seed(42)
        config = ReasoningConfig(
            input_dim=256,
            hidden_dim=256,
            num_hops=4,
            max_hops=8,
            min_hops=2,
            enable_cross_hop=True,
            enable_backtrack=True,
            enable_act=True,
        )
        head = CoTReasoningHead(config).eval()

        # Force high trap detection on all thoughts
        with torch.no_grad():
            for p in head.backtrack_module.detector.parameters():
                p.fill_(0.9)

        # Instrument forward pass
        call_counts_per_pass: List[int] = []
        original_forward = head.backtrack_module.forward

        current_call_count = [0]

        def instrumented_forward(*args, **kwargs):
            current_call_count[0] += 1
            return original_forward(*args, **kwargs)

        head.backtrack_module.forward = instrumented_forward

        N_PASSES = 1000
        torch.manual_seed(999)

        with torch.no_grad():
            for i in range(N_PASSES):
                current_call_count[0] = 0
                x = torch.randn(1, 256)
                u = torch.rand(1)
                _ = head(x)
                call_counts_per_pass.append(current_call_count[0])

        max_calls = max(call_counts_per_pass)
        mean_calls = sum(call_counts_per_pass) / len(call_counts_per_pass)
        num_zero_calls = sum(1 for c in call_counts_per_pass if c == 0)
        num_one_calls = sum(1 for c in call_counts_per_pass if c == 1)
        num_greater_than_one = sum(1 for c in call_counts_per_pass if c > 1)

        print("\n" + "=" * 80)
        print("EMPIRICAL CHALLENGER: STRICT N_BACKTRACK <= 1 INVARIANT (1,000 PASSES)")
        print("=" * 80)
        print(f"Total Passes Tested: {N_PASSES}")
        print(f"Passes with N_backtrack == 0: {num_zero_calls}")
        print(f"Passes with N_backtrack == 1: {num_one_calls}")
        print(f"Passes with N_backtrack > 1:  {num_greater_than_one}")
        print(f"Max Backtrack Calls in Any Pass: {max_calls}")
        print(f"Mean Backtrack Calls: {mean_calls:.4f}")
        print("=" * 80)

        assert num_greater_than_one == 0, (
            f"VIOLATION: N_backtrack exceeded 1 in {num_greater_than_one} passes! Max={max_calls}"
        )
        assert max_calls <= 1, f"VIOLATION: Max backtrack calls {max_calls} > 1!"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
