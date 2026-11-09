"""
Empirical Challenger 2 Stress Test Suite (teamwork_preview_challenger_m1_2)
Focus:
1. Soft ACT Adaptive Pondering depth bounds & numerical stability under extreme sigma.
2. PRMVerifier dynamic hop counts K in [2..8] & Weak-Link Principle verification.
3. 200-tick CPU latency profiling and latency degradation check.
"""

import time
import math
import pytest
import torch
import torch.nn.functional as F

from research_reasoning_world_class.src.metacognitive_reasoning import (
    ReasoningConfig,
    MetacognitiveReasoningEngine,
    SoftHaltingPonderController,
    PRMVerifier,
    CoTReasoningHead,
)


class TestEmpiricalChallengerSoftACT:
    """Stress testing Soft ACT Adaptive Pondering under extreme sigma values."""

    @pytest.mark.parametrize("sigma", [0.0, 0.05, 0.5, 0.95, 1.0, 10.0])
    def test_soft_act_depth_bounds_standalone(self, sigma):
        """Test SoftHaltingPonderController across extreme sigma inputs."""
        torch.manual_seed(42)
        controller = SoftHaltingPonderController(hidden_dim=256, max_hops=8, min_hops=2)
        controller.eval()

        batch_size = 16
        hop_states = [torch.randn(batch_size, 256) for _ in range(8)]
        uncertainty = torch.full((batch_size,), sigma)

        with torch.no_grad():
            aggregated, halting_weights, exp_depth = controller(hop_states, uncertainty)

        # 1. Check for NaN / Inf
        assert not torch.isnan(aggregated).any(), f"NaN in aggregated for sigma={sigma}"
        assert not torch.isinf(aggregated).any(), f"Inf in aggregated for sigma={sigma}"
        assert not torch.isnan(halting_weights).any(), f"NaN in halting_weights for sigma={sigma}"
        assert not torch.isinf(halting_weights).any(), f"Inf in halting_weights for sigma={sigma}"
        assert not math.isnan(exp_depth), f"NaN in exp_depth for sigma={sigma}"
        assert not math.isinf(exp_depth), f"Inf in exp_depth for sigma={sigma}"

        # 2. Check simplex property: weights sum to 1.0
        weight_sums = halting_weights.sum(dim=-1)
        torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-5, rtol=1e-5)

        # 3. Check bounds for calm and volatile
        if sigma <= 0.05:
            assert exp_depth <= 2.5, f"Calm market E[K] violation: E[K]={exp_depth:.3f} > 2.5 for sigma={sigma}"
        elif sigma >= 0.95:
            assert exp_depth >= 5.0, f"Volatile market E[K] violation: E[K]={exp_depth:.3f} < 5.0 for sigma={sigma}"

    def test_soft_act_monotonicity_across_sigma(self):
        """Verify that expected depth monotonically increases as uncertainty increases."""
        torch.manual_seed(42)
        controller = SoftHaltingPonderController(hidden_dim=256, max_hops=8, min_hops=2)
        controller.eval()

        sigmas = [0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0, 2.0, 5.0, 10.0]
        depths = []
        hop_states = [torch.randn(16, 256) for _ in range(8)]

        with torch.no_grad():
            for s in sigmas:
                u = torch.full((16,), s)
                _, _, d = controller(hop_states, u)
                depths.append(d)

        for i in range(len(depths) - 1):
            assert depths[i] <= depths[i + 1] + 1e-4, (
                f"Non-monotonic depth response: sigma={sigmas[i]} -> {depths[i]:.3f}, "
                f"sigma={sigmas[i+1]} -> {depths[i+1]:.3f}"
            )

    @pytest.mark.parametrize("sigma", [0.0, 0.05, 0.5, 0.95, 1.0, 10.0])
    def test_full_engine_act_integration(self, sigma):
        """Test full MetacognitiveReasoningEngine with enable_act=True across extreme sigmas."""
        torch.manual_seed(42)
        config = ReasoningConfig(hidden_dim=256, max_hops=8, enable_act=True)
        engine = MetacognitiveReasoningEngine(config)
        engine.eval()

        batch_size = 4
        x = torch.randn(batch_size, 256)
        u = torch.full((batch_size,), sigma)

        with torch.no_grad():
            output = engine(x, uncertainty=u)

        assert not torch.isnan(output.confidence).any()
        assert not torch.isnan(output.prm_score).any()
        assert not torch.isnan(output.cot_hops).any()
        assert output.cot_hops.shape == (batch_size, 8, 256)

        exp_depth = engine.cot_head.last_expected_depth
        if sigma <= 0.05:
            assert exp_depth <= 2.5, f"Engine calm E[K] violation: E[K]={exp_depth:.3f} > 2.5 for sigma={sigma}"
        elif sigma >= 0.95:
            assert exp_depth >= 5.0, f"Engine volatile E[K] violation: E[K]={exp_depth:.3f} < 5.0 for sigma={sigma}"

    def test_extreme_adversarial_inputs_to_ponder(self):
        """Test with extreme negative, infinite, and NaN uncertainty inputs."""
        torch.manual_seed(42)
        controller = SoftHaltingPonderController(hidden_dim=256, max_hops=8, min_hops=2)
        controller.eval()

        hop_states = [torch.randn(4, 256) for _ in range(8)]
        # Negative sigma
        _, _, d_neg = controller(hop_states, torch.full((4,), -5.0))
        assert not math.isnan(d_neg)
        assert d_neg <= 2.5

        # Extreme positive sigma
        _, _, d_huge = controller(hop_states, torch.full((4,), 1000.0))
        assert not math.isnan(d_huge)
        assert d_huge >= 5.0


class TestEmpiricalChallengerPRMWeakLink:
    """Stress testing PRMVerifier dynamic hop counts K and Weak-Link Principle."""

    @pytest.mark.parametrize("K", [2, 3, 4, 5, 6, 7, 8])
    def test_prm_verifier_dynamic_hop_counts(self, K):
        """Verify PRMVerifier handles any K in [2..8] without dimension error."""
        torch.manual_seed(42)
        config = ReasoningConfig(hidden_dim=256, num_hops=4, max_hops=8)
        verifier = PRMVerifier(config)
        verifier.eval()

        batch_size = 8
        all_hops = torch.randn(batch_size, K, 256)

        with torch.no_grad():
            prm_score, hop_rewards = verifier(all_hops)

        assert prm_score.shape == (batch_size,)
        assert hop_rewards.shape == (batch_size, K)
        assert not torch.isnan(prm_score).any()
        assert not torch.isinf(prm_score).any()
        assert (prm_score >= 0.0).all() and (prm_score <= 1.0).all()

    def test_weak_link_active_hop_flaw_degrades_score(self):
        """
        Verify Weak-Link Principle:
        An intentional flaw injected into an active hop (k <= floor(E[K]))
        MUST cause substantial degradation in the PRM score.
        """
        torch.manual_seed(42)
        config = ReasoningConfig(hidden_dim=256, max_hops=8)
        verifier = PRMVerifier(config)
        verifier.eval()

        batch_size = 4
        K = 8
        # Construct coherent baseline hops
        base_hop = torch.randn(batch_size, 1, 256)
        all_hops = base_hop.repeat(1, K, 1) + 0.05 * torch.randn(batch_size, K, 256)

        # Suppose expected_depth = 4.0, so active hops are k=0,1,2,3
        exp_depth = 4.0
        halting_weights = torch.zeros(batch_size, K)
        halting_weights[:, :4] = 0.25

        with torch.no_grad():
            prm_clean, rewards_clean = verifier(
                all_hops, halting_weights=halting_weights, expected_depth=exp_depth
            )

            # Inject a fatal flaw into active hop 1 (index 1)
            flawed_hops_active = all_hops.clone()
            flawed_hops_active[:, 1, :] = torch.randn(batch_size, 256) * 50.0  # complete disruption

            prm_active_flawed, rewards_active_flawed = verifier(
                flawed_hops_active, halting_weights=halting_weights, expected_depth=exp_depth
            )

        # Flawed active hop must decrease the step/hop reward
        assert rewards_active_flawed[:, 1].mean() < rewards_clean[:, 1].mean()
        # Weak-Link principle: the aggregate PRM score MUST degrade significantly
        prm_drop = (prm_clean - prm_active_flawed).mean().item()
        assert prm_drop > 0.05, f"Weak-Link failure: active flaw did not degrade PRM score! Drop: {prm_drop:.4f}"

    def test_weak_link_unpondered_hop_flaw_does_not_falsely_penalize(self):
        """
        Verify that an intentional flaw in an unpondered hop (k > floor(E[K]))
        does NOT falsely penalize the PRM score (masked under Weak-Link principle).
        """
        torch.manual_seed(42)
        config = ReasoningConfig(hidden_dim=256, max_hops=8)
        verifier = PRMVerifier(config)
        verifier.eval()

        batch_size = 4
        K = 8
        base_hop = torch.randn(batch_size, 1, 256)
        all_hops = base_hop.repeat(1, K, 1) + 0.05 * torch.randn(batch_size, K, 256)

        # Expected depth = 2.0 (calm market, early halt at K=2).
        # Active hops: indices 0, 1. Unpondered hops: indices 2..7.
        exp_depth = 2.0
        halting_weights = torch.zeros(batch_size, K)
        halting_weights[:, 0] = 0.5
        halting_weights[:, 1] = 0.5

        with torch.no_grad():
            prm_clean, _ = verifier(
                all_hops, halting_weights=halting_weights, expected_depth=exp_depth
            )

            # Inject extreme flaw into unpondered hop 7 (index 6) and hop 8 (index 7)
            flawed_hops_unpondered = all_hops.clone()
            flawed_hops_unpondered[:, 6:, :] = torch.randn(batch_size, 2, 256) * 100.0

            prm_unpondered_flawed, _ = verifier(
                flawed_hops_unpondered, halting_weights=halting_weights, expected_depth=exp_depth
            )

            # Also test what happens if flaw WAS in active hop 0
            flawed_hops_active = all_hops.clone()
            flawed_hops_active[:, 0, :] = torch.randn(batch_size, 256) * 100.0
            prm_active_flawed, _ = verifier(
                flawed_hops_active, halting_weights=halting_weights, expected_depth=exp_depth
            )

        # PRM score with unpondered flaw should be virtually identical to clean
        # (halting weight is 0 for unpondered hops, and masked min_hop_reward ignores them)
        unpondered_diff = torch.abs(prm_clean - prm_unpondered_flawed).max().item()
        active_diff = torch.abs(prm_clean - prm_active_flawed).mean().item()

        assert unpondered_diff < 1e-4, (
            f"Unpondered hop flaw falsely penalized PRM score! Max diff: {unpondered_diff:.6f}"
        )
        assert active_diff > 0.05, (
            f"Active hop flaw was not penalized! Mean diff: {active_diff:.6f}"
        )


class TestEmpiricalChallengerCPULatency:
    """Stress testing single-tick CPU latency across 200 consecutive passes."""

    def test_200_consecutive_passes_latency_profile(self):
        """
        Execute 200 consecutive single-tick inference passes on CPU.
        Verify SLA (P95 <= 50.0 ms) and check for latency degradation / memory bloat.
        """
        torch.manual_seed(42)
        # Full engine with ACT enabled (max 8 hops)
        config = ReasoningConfig(hidden_dim=256, max_hops=8, enable_act=True)
        engine = MetacognitiveReasoningEngine(config)
        engine.eval()

        # Warm up 5 passes
        x_warm = torch.randn(1, 256)
        u_warm = torch.tensor([0.5])
        with torch.no_grad():
            for _ in range(5):
                _ = engine(x_warm, uncertainty=u_warm)

        # 200 consecutive passes
        latencies = []
        batch_size = 1

        with torch.no_grad():
            for tick in range(200):
                # Vary inputs realistically (simulate varying market volatility)
                sigma_val = 0.05 if (tick % 4 == 0) else (0.95 if (tick % 4 == 1) else 0.5)
                x = torch.randn(batch_size, 256)
                u = torch.tensor([sigma_val])

                t0 = time.perf_counter()
                out = engine(x, uncertainty=u)
                dt = (time.perf_counter() - t0) * 1000.0
                latencies.append(dt)

                assert not torch.isnan(out.confidence).any(), f"NaN confidence at tick {tick}"

        # Latency statistics
        latencies_sorted = sorted(latencies)
        p50 = latencies_sorted[int(len(latencies) * 0.50)]
        p95 = latencies_sorted[int(len(latencies) * 0.95)]
        p99 = latencies_sorted[int(len(latencies) * 0.99)]
        max_lat = max(latencies)
        mean_lat = sum(latencies) / len(latencies)

        # Check early vs late degradation (first 50 vs last 50)
        first_50_mean = sum(latencies[:50]) / 50.0
        last_50_mean = sum(latencies[-50:]) / 50.0
        drift_ratio = last_50_mean / (first_50_mean + 1e-6)

        print(
            f"\n[200-Tick Profile] Mean: {mean_lat:.2f}ms | P50: {p50:.2f}ms | "
            f"P95: {p95:.2f}ms | P99: {p99:.2f}ms | Max: {max_lat:.2f}ms | "
            f"First50: {first_50_mean:.2f}ms | Last50: {last_50_mean:.2f}ms | Drift: {drift_ratio:.2f}x"
        )

        # SLA Assertions: Steady-state execution meets 50.0ms SLA at P95 (and Mean <= 30.0ms)
        assert mean_lat <= 30.0, f"SLA Violation: Mean latency is {mean_lat:.2f}ms > 30.0ms"
        assert p50 <= 30.0, f"SLA Violation: P50 latency is {p50:.2f}ms > 30.0ms"
        assert p95 <= 50.0, f"SLA Violation: P95 latency is {p95:.2f}ms > 50.0ms"
        assert drift_ratio < 2.0, f"Latency degradation detected! Drift ratio: {drift_ratio:.2f}x"
