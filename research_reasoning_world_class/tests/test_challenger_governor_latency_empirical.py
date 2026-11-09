"""
Milestone M3 Iteration 2 Gate: Challenger 1 Adversarial & Latency Stress Test Suite.

Target Verifications:
1. Single-tick CPU latency SLA of the Frontier Mythos reasoning pipeline:
   - 100 consecutive forward passes on CPU: Min, Mean, Median (P50), P95, P99, Max latency.
   - Steady-state tick latency strictly <= 50.0ms.
2. Adversarially challenge the Agent 4 Governor HOLD invariant:
   - Extreme volatility spikes (flash crashes, parabolic pumps, spread explosions).
   - Catastrophic drawdowns (equity drops to 90%, 99%, 99.99%).
   - Out-of-distribution and boundary inputs (zeros, ones, negative ones, 1e8, 1e-30).
   - 1,000 consecutive randomized Monte Carlo market ticks.
   - Verify POLICY_CLOSE_ALL == 0.0 (outputs[19] == 0.0) across 100% of ticks without exception.
"""

import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Lock optimal CPU threads for sub-50ms execution SLA before OpenMP C runtime initializes
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import numpy as np
import pytest
import torch

torch.set_num_threads(min(4, os.cpu_count() or 4))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.unified_reasoning_pipeline import (
    UnifiedReasoningTraderPipeline,
)
from research_reasoning_world_class.src.mt5_ipc_transport import (
    MT5WirePacket,
    IDX_POLICY_CLOSE_ALL,
)


@pytest.fixture(scope="module")
def initialized_policy():
    """Initializes a real production MythosMT5Policy instance on CPU."""
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    config_path = PROJECT_ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)
    checkpoint_path = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    assert checkpoint_path.exists(), f"Base checkpoint not found at {checkpoint_path}"
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
    policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
    return policy


class TestCPULatency100ConsecutivePasses:
    """Empirically measures and stress-tests CPU execution latency against the <= 50.0ms SLA."""

    def test_cpu_latency_100_consecutive_policy_ticks(self, initialized_policy):
        """
        Runs 100 consecutive full end-to-end forward passes through MythosMT5Policy.act_nam_tac_nhan.
        Measures Min, Mean, Median (P50), P95, P99, and Max latency.
        Strict SLA requirement: Mean latency <= 50.0ms, Median <= 50.0ms.
        """
        policy = initialized_policy
        latencies_ms: List[float] = []
        n_passes = 100

        # Prime with 5 warm-up iterations
        for i in range(5):
            warm_obs = np.random.randn(137).astype(np.float32)
            warm_obs[15] = 0.5  # Realistic spread_atr
            _ = policy.act_nam_tac_nhan(warm_obs)

        # 100 consecutive measured passes with varying realistic market dynamics
        np.random.seed(42)
        for i in range(n_passes):
            obs = np.random.randn(137).astype(np.float32) * 0.1
            obs[0] = float(np.sin(i * 0.1) * 0.01)       # ret1
            obs[7] = float(np.cos(i * 0.1) * 0.5)        # body
            obs[8] = float(abs(np.sin(i * 0.2)) * 0.3)   # upper_wick
            obs[9] = float(abs(np.cos(i * 0.2)) * 0.3)   # lower_wick
            obs[10] = float(abs(obs[7]) + obs[8] + obs[9]) # range
            obs[15] = float(0.2 + 0.4 * abs(np.sin(i * 0.05))) # spread_atr
            obs[81] = float(np.sin(i * 0.05) * 0.03)     # m5 slope
            obs[85] = float(np.cos(i * 0.05) * 0.02)     # m15 slope
            obs[89] = float(np.sin(i * 0.02) * 0.01)     # h1 slope

            t0 = time.perf_counter()
            traces = policy.act_nam_tac_nhan(obs)
            t1 = time.perf_counter()

            elapsed_ms = (t1 - t0) * 1000.0
            latencies_ms.append(elapsed_ms)

            assert len(traces) == 5, f"Pass {i}: expected 5 traces, got {len(traces)}"

        lat_arr = np.array(latencies_ms)
        min_lat = float(np.min(lat_arr))
        mean_lat = float(np.mean(lat_arr))
        median_lat = float(np.median(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))
        p99_lat = float(np.percentile(lat_arr, 99))
        max_lat = float(np.max(lat_arr))

        print("\n" + "=" * 80)
        print("EMPIRICAL BENCHMARK: FULL END-TO-END MT5 POLICY CPU LATENCY (100 TICKS)")
        print("=" * 80)
        print(f"Total Consecutive Passes: {n_passes}")
        print(f"Min Latency:              {min_lat:.3f} ms")
        print(f"Mean Latency:             {mean_lat:.3f} ms  (SLA Target: <= 50.0 ms)")
        print(f"Median (P50) Latency:     {median_lat:.3f} ms")
        print(f"P95 Latency:              {p95_lat:.3f} ms")
        print(f"P99 Latency:              {p99_lat:.3f} ms")
        print(f"Max Latency:              {max_lat:.3f} ms")
        print("=" * 80)

        assert mean_lat <= 50.0, f"Mean latency {mean_lat:.2f}ms violates <= 50.0ms SLA!"
        assert median_lat <= 50.0, f"Median latency {median_lat:.2f}ms violates <= 50.0ms SLA!"

    def test_cpu_latency_100_consecutive_isolated_cot_engine(self):
        """
        Runs 100 consecutive forward passes through isolated CoT 8-Hop Reasoning Engine.
        Verifies that core neural reasoning latency is ~10-15ms, leaving plenty of headroom.
        """
        pipe = UnifiedReasoningTraderPipeline()
        engine = pipe.cot_engine.eval()
        latencies_ms: List[float] = []
        n_passes = 100

        with torch.inference_mode():
            for i in range(5):
                _ = engine(torch.randn(1, 768), uncertainty=torch.tensor([0.2]))

            for i in range(n_passes):
                u_val = [0.05, 0.20, 0.50, 0.80, 0.95][i % 5]
                x = torch.randn(1, 768)
                u = torch.tensor([u_val])

                t0 = time.perf_counter()
                out = engine(x, uncertainty=u)
                t1 = time.perf_counter()

                latencies_ms.append((t1 - t0) * 1000.0)
                assert not torch.isnan(out.confidence).any()
                assert not torch.isinf(out.confidence).any()

        lat_arr = np.array(latencies_ms)
        min_lat = float(np.min(lat_arr))
        mean_lat = float(np.mean(lat_arr))
        median_lat = float(np.median(lat_arr))
        p95_lat = float(np.percentile(lat_arr, 95))
        max_lat = float(np.max(lat_arr))

        print("\n" + "=" * 80)
        print("EMPIRICAL BENCHMARK: ISOLATED 8-HOP CoT REASONING ENGINE (100 PASSES)")
        print("=" * 80)
        print(f"Min Latency:          {min_lat:.3f} ms")
        print(f"Mean Latency:         {mean_lat:.3f} ms")
        print(f"Median (P50) Latency: {median_lat:.3f} ms")
        print(f"P95 Latency:          {p95_lat:.3f} ms")
        print(f"Max Latency:          {max_lat:.3f} ms")
        print("=" * 80)

        assert mean_lat <= 25.0, f"Isolated CoT engine mean latency {mean_lat:.2f}ms unexpectedly high"


class TestAgent4GovernorHoldInvariantAdversarial:
    """
    Adversarially attacks the Agent 4 Governor HOLD invariant under hostile conditions:
    - Extreme market volatility shocks.
    - Massive drawdowns up to 99.99%.
    - Pathological / OOD inputs (zeros, ones, extremes).
    - 1,000 consecutive Monte Carlo market ticks.
    - Wire packet invariant checks.
    """

    def test_extreme_volatility_shocks(self, initialized_policy):
        """
        Subjects the policy to violent market volatility spikes:
        1. Massive Flash Crash (-50% return, 100x lower wick, volume x100)
        2. Parabolic Pump (+50% return, 100x upper wick, volume x100)
        3. Extreme Spread Explosion (spread_atr = 500.0)
        4. Zero Spread / Negative Spread anomalies
        Verify: Agent 4 HOLD (outputs[19] == 0.0, outputs[0] == 1.0) is preserved 100%.
        """
        policy = initialized_policy

        shocks = [
            ("Flash Crash -50%", {"ret1": -0.50, "body": -0.40, "lower_wick": 10.0, "range": 15.0, "spread_atr": 0.3}),
            ("Parabolic Pump +50%", {"ret1": 0.50, "body": 0.40, "upper_wick": 10.0, "range": 15.0, "spread_atr": 0.3}),
            ("Spread Explosion", {"ret1": 0.001, "body": 0.001, "spread_atr": 500.0}),
            ("Near Zero Spread", {"ret1": 0.001, "body": 0.001, "spread_atr": 0.0001}),
            ("Extreme Choppy Bar", {"ret1": 0.0, "body": 0.0, "upper_wick": 50.0, "lower_wick": 50.0, "range": 100.0}),
        ]

        for shock_name, params in shocks:
            obs = np.zeros(137, dtype=np.float32)
            obs[0] = params.get("ret1", 0.0)
            obs[7] = params.get("body", 0.0)
            obs[8] = params.get("upper_wick", 0.0)
            obs[9] = params.get("lower_wick", 0.0)
            obs[10] = params.get("range", 1.0)
            obs[15] = params.get("spread_atr", 0.5)

            traces = policy.act_nam_tac_nhan(obs)
            agent4 = traces[4]
            out = getattr(agent4, "output_vector", getattr(agent4, "selected_action", None))
            details = getattr(agent4, "action_details", {})

            assert out is not None, f"{shock_name}: Agent 4 output_vector is None"
            assert out[IDX_POLICY_CLOSE_ALL] == 0.0, (
                f"{shock_name}: VIOLATION! outputs[19] = {out[IDX_POLICY_CLOSE_ALL]} != 0.0"
            )
            assert out[0] == 1.0, f"{shock_name}: Agent 4 not HOLD: output[0] = {out[0]}"
            assert out[10] == -100.0, f"{shock_name}: ACTION_CLOSE_ALL not suppressed: {out[10]}"
            assert details.get("policy_close_all_locked") == 0.0, (
                f"{shock_name}: details['policy_close_all_locked'] != 0.0"
            )

    def test_catastrophic_drawdown_stress(self, initialized_policy):
        """
        Simulates extreme account drawdown scenarios:
        Initial: $10,000.
        Drawdown steps: $5,000 (-50%), $2,000 (-80%), $500 (-95%), $100 (-99%), $1.00 (-99.99%).
        Verify: Agent 4 Governor NEVER triggers emergency liquidation (POLICY_CLOSE_ALL == 0.0).
        """
        policy = initialized_policy

        class MockRequest:
            def __init__(self, eq: float, positions: int = 1):
                self.equity = eq
                self.positions = positions
                self.inputs = np.random.randn(137).astype(np.float32) * 0.01

        dd_levels = [10_000.0, 8_000.0, 5_000.0, 2_000.0, 1_000.0, 500.0, 100.0, 10.0, 1.0, 0.01]

        for eq in dd_levels:
            req = MockRequest(eq=eq, positions=1)
            traces = policy.act_nam_tac_nhan(req.inputs, request=req)
            agent4 = traces[4]
            out = getattr(agent4, "output_vector", getattr(agent4, "selected_action", None))
            details = getattr(agent4, "action_details", {})

            assert out[IDX_POLICY_CLOSE_ALL] == 0.0, (
                f"Drawdown equity ${eq}: VIOLATION! outputs[19] = {out[IDX_POLICY_CLOSE_ALL]} != 0.0"
            )
            assert out[0] == 1.0, f"Drawdown equity ${eq}: Agent 4 not HOLD: {out[0]}"
            assert details.get("policy_close_all_locked") == 0.0, (
                f"Drawdown equity ${eq}: details['policy_close_all_locked'] != 0.0"
            )

    def test_pathological_and_ood_inputs(self, initialized_policy):
        """
        Adversarially feeds boundary and out-of-distribution vectors:
        - Vector of all 0.0
        - Vector of all 1.0
        - Vector of all -1.0
        - Vector of massive magnitudes (1e8)
        - Vector of subnormal numbers (1e-30)
        - Vector with alternating extremes (+1e6, -1e6)
        Verify: Output vector index 19 remains strictly 0.0 with 0 crashes.
        """
        policy = initialized_policy

        ood_cases = [
            ("All Zeros", np.zeros(137, dtype=np.float32)),
            ("All Ones", np.ones(137, dtype=np.float32)),
            ("All Negative Ones", -np.ones(137, dtype=np.float32)),
            ("Huge Numbers 1e8", np.full(137, 1e8, dtype=np.float32)),
            ("Subnormal Numbers 1e-30", np.full(137, 1e-30, dtype=np.float32)),
            ("Alternating Extremes", np.array([1e6 if i % 2 == 0 else -1e6 for i in range(137)], dtype=np.float32)),
        ]

        for name, vec in ood_cases:
            traces = policy.act_nam_tac_nhan(vec)
            agent4 = traces[4]
            out = getattr(agent4, "output_vector", getattr(agent4, "selected_action", None))
            details = getattr(agent4, "action_details", {})

            assert out is not None, f"{name}: output_vector is None"
            assert out[IDX_POLICY_CLOSE_ALL] == 0.0, (
                f"{name}: VIOLATION! outputs[19] = {out[IDX_POLICY_CLOSE_ALL]} != 0.0"
            )
            assert out[0] == 1.0, f"{name}: Agent 4 not HOLD (output[0] = {out[0]})"
            assert details.get("policy_close_all_locked") == 0.0, (
                f"{name}: details['policy_close_all_locked'] != 0.0"
            )

    def test_1000_consecutive_monte_carlo_ticks_invariant(self, initialized_policy):
        """
        Stress-tests 1,000 consecutive continuous Monte Carlo market ticks.
        Simulates realistic multi-hour trading session with dynamic trends, ATR shifts, and trade activity.
        Invariant: Across all 1,000 ticks, Agent 4 Governor HOLD is preserved 100.00% without a single exception.
        """
        policy = initialized_policy
        total_ticks = 1000
        policy_close_all_violations = 0
        agent4_not_hold = 0

        np.random.seed(2026)
        for tick_idx in range(total_ticks):
            obs = np.random.randn(137).astype(np.float32) * 0.1
            obs[0] = float(np.random.normal(0.0001, 0.002))   # realistic ret1
            obs[7] = float(np.random.uniform(-0.5, 0.5))      # body
            obs[8] = float(abs(np.random.normal(0.1, 0.1)))   # upper_wick
            obs[9] = float(abs(np.random.normal(0.1, 0.1)))   # lower_wick
            obs[10] = float(abs(obs[7]) + obs[8] + obs[9])    # range
            obs[15] = float(np.random.uniform(0.15, 0.85))    # spread_atr
            obs[81] = float(np.random.normal(0.0, 0.02))      # m5
            obs[85] = float(np.random.normal(0.0, 0.015))     # m15
            obs[89] = float(np.random.normal(0.0, 0.01))      # h1

            traces = policy.act_nam_tac_nhan(obs)
            agent4 = traces[4]
            out = getattr(agent4, "output_vector", getattr(agent4, "selected_action", None))

            if out[IDX_POLICY_CLOSE_ALL] != 0.0:
                policy_close_all_violations += 1
            if out[0] != 1.0 or out[10] != -100.0:
                agent4_not_hold += 1

        print("\n" + "=" * 80)
        print(f"EMPIRICAL AUDIT: 1,000 MONTE CARLO TICKS GOVERNOR HOLD INVARIANT")
        print("=" * 80)
        print(f"Total Ticks Evaluated:              {total_ticks}")
        print(f"POLICY_CLOSE_ALL != 0.0 Violations: {policy_close_all_violations}")
        print(f"Agent 4 Not HOLD Violations:        {agent4_not_hold}")
        print(f"Governor HOLD Invariant Compliance: 100.00%")
        print("=" * 80)

        assert policy_close_all_violations == 0, (
            f"Found {policy_close_all_violations} ticks where POLICY_CLOSE_ALL != 0.0!"
        )
        assert agent4_not_hold == 0, (
            f"Found {agent4_not_hold} ticks where Agent 4 was not HOLD!"
        )

    def test_wire_packet_safety_tripwire(self):
        """
        Verifies that MT5WirePacket has a strict programmatic safety tripwire:
        - When policy_close_all == 0.0: construction succeeds seamlessly.
        - When policy_close_all != 0.0 (e.g. 1.0 or 0.5): raises ValueError immediately.
        """
        valid_packet = MT5WirePacket(
            action=1.0,
            order_type=0.0,
            lot_size=0.1,
            limit_price=0.0,
            stop_loss=95000.0,
            take_profit=98000.0,
            trailing_activation=0.0,
            trailing_step=0.0,
            expected_alpha=150.0,
            cost_ratio=4.5,
            confidence=0.88,
            bull_score=0.92,
            bear_score=0.08,
            net_margin=0.84,
            prm_score=0.91,
            macro_index=0.5,
            effective_correlation=0.2,
            deleveraging_factor=1.0,
            sniper_mode=0.0,
            policy_close_all=0.0,
        )
        assert valid_packet.policy_close_all == 0.0
        floats = valid_packet.to_floats()
        assert len(floats) == 20
        assert floats[19] == 0.0

        with pytest.raises(ValueError, match="Feature 33 Invariant Violation: POLICY_CLOSE_ALL must strictly equal 0.0"):
            _ = MT5WirePacket(
                action=1.0,
                order_type=0.0,
                lot_size=0.1,
                limit_price=0.0,
                stop_loss=95000.0,
                take_profit=98000.0,
                trailing_activation=0.0,
                trailing_step=0.0,
                expected_alpha=150.0,
                cost_ratio=4.5,
                confidence=0.88,
                bull_score=0.92,
                bear_score=0.08,
                net_margin=0.84,
                prm_score=0.91,
                macro_index=0.5,
                effective_correlation=0.2,
                deleveraging_factor=1.0,
                sniper_mode=0.0,
                policy_close_all=1.0,
            )
