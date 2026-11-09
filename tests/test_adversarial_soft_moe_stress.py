"""
tests/test_adversarial_soft_moe_stress.py

Adversarial stress testing harness for Soft-MoE Gating Network,
Capital Preservation Guard, and DualChampionRouter continuous integration.

Author: Challenger M1_1 (Empirical Robustness Challenger)
Roles: critic, specialist
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pytest
import torch

from self_growing_trader.dual_champion_router import DualChampionRouter
from self_growing_trader.soft_moe_gating import (
    NUM_MTS_FEATURES,
    WIRE_MTS_INDICES,
    CapitalPreservationGuard,
    SoftMoEGatingNetwork,
    extract_multi_timescale_features,
)
class MockRecurrentSpecialist:
    """Mock specialist policy tracking 4096-D recurrent state."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.hidden_state = np.zeros((5, 4096), dtype=np.float32)
        self.cell_state = np.zeros((5, 4096), dtype=np.float32)
        self.step_count = 0

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[object, ...]:
        self.step_count += 1
        self.hidden_state += 1.0
        self.cell_state += 2.0

        class MockTrace:
            def __init__(self, agent_id: int, act: int) -> None:
                self.trace_id = f"{agent_id}_trace"
                self.selected_idx = act
                self.selected_action = np.zeros(20, dtype=float)
                self.selected_action[act] = 1.0
                self.action_details = {
                    "five_agent_index": agent_id,
                    "five_action_index": act,
                    "wire_action": act,
                    "base_risk": 0.01,
                    "risk_pct": 0.01,
                }

        return (
            MockTrace(0, 1),  # MARKET_BUY
            MockTrace(1, 2),  # MARKET_SELL
            MockTrace(2, 0),  # HOLD
            MockTrace(3, 0),  # HOLD
            MockTrace(4, 7),  # Non-hold
        )

    def ensure_compatible(self) -> None:
        pass

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state.fill(0.0)
        self.cell_state.fill(0.0)


# ==============================================================================
# 1. SIMPLEX INVARIANT FUZZING & EXTREME BOUNDARY HARNESS
# ==============================================================================

class TestSimplexInvariantFuzzing:
    """Adversarial stress tests for mathematical invariant w_bull + w_bear == 1.0."""

    @pytest.mark.parametrize("scale", [1e-4, 1.0, 10.0, 100.0, 1e4])
    def test_simplex_fuzzing_1000_scaled_random_vectors(self, scale: float) -> None:
        """Verify simplex invariant across 1,000 vectors at varied variance scales."""
        gating = SoftMoEGatingNetwork()
        rng = np.random.RandomState(int(scale * 1000) % 1234567)

        for _ in range(1000):
            obs = (rng.randn(137) * scale).astype(np.float32)
            wb, wr, info = gating.compute_regime_weights(obs)

            assert math.isfinite(wb), f"Non-finite w_bull: {wb}"
            assert math.isfinite(wr), f"Non-finite w_bear: {wr}"
            assert 0.0 <= wb <= 1.0, f"w_bull out of bounds: {wb}"
            assert 0.0 <= wr <= 1.0, f"w_bear out of bounds: {wr}"
            assert abs((wb + wr) - 1.0) <= 1e-7, f"Simplex violation: {wb} + {wr} = {wb + wr}"

    def test_extreme_spikes_100_atr(self) -> None:
        """Verify behavior with extreme +/- 100 ATR momentum shocks."""
        gating = SoftMoEGatingNetwork()

        # Positive 100 ATR shock
        obs_spike_pos = np.zeros(137, dtype=np.float32)
        obs_spike_pos[0] = 100.0   # M1 return 100 ATR
        obs_spike_pos[80] = 100.0  # M5 EMA diff 100 ATR
        obs_spike_pos[81] = 50.0   # M5 EMA slope 50 ATR
        wb, wr, info = gating.compute_regime_weights(obs_spike_pos)

        assert wb > 0.999
        assert wr < 0.001
        assert abs(wb + wr - 1.0) <= 1e-7

        # Negative 100 ATR shock
        obs_spike_neg = np.zeros(137, dtype=np.float32)
        obs_spike_neg[0] = -100.0
        obs_spike_neg[80] = -100.0
        obs_spike_neg[81] = -50.0
        wb, wr, info = gating.compute_regime_weights(obs_spike_neg)

        assert wr > 0.999
        assert wb < 0.001
        assert abs(wb + wr - 1.0) <= 1e-7

    def test_pathological_inputs_137d_all_zeros_and_subnormals(self) -> None:
        """Test with all zeros and subnormal IEEE floating point numbers."""
        gating = SoftMoEGatingNetwork()

        obs_zeros = np.zeros(137, dtype=np.float32)
        wb, wr, info = gating.compute_regime_weights(obs_zeros)
        assert abs(wb + wr - 1.0) <= 1e-7
        assert math.isclose(wb, 0.5, abs_tol=0.1)  # Neutral prior

        # Subnormal float32 (near zero, 1e-40)
        obs_sub = np.full(137, 1e-40, dtype=np.float32)
        wb, wr, info = gating.compute_regime_weights(obs_sub)
        assert abs(wb + wr - 1.0) <= 1e-7
        assert math.isfinite(wb) and math.isfinite(wr)


# ==============================================================================
# 2. TRANSITION LAG & 180-DEGREE INFLECTION HARNESS
# ==============================================================================

class TestTransitionLagAndInflections:
    """Empirically verifies 0-1 bar transition speed vs legacy 15-45m lag."""

    def test_flash_180_degree_reversal_response_bars(self) -> None:
        """Verify that a sudden 180-degree momentum reversal switches regime in 0-1 bars."""
        gating = SoftMoEGatingNetwork()

        # Bar 0: Strongly Bullish
        obs_bull = np.zeros(137, dtype=np.float32)
        obs_bull[0] = 2.0
        obs_bull[80] = 2.5
        obs_bull[81] = 1.0
        obs_bull[88] = 2.0
        wb0, wr0, info0 = gating.compute_regime_weights(obs_bull)
        assert wb0 > 0.85
        assert info0["dominant_regime"] == "BULL"

        # Bar 1: Flash 180-degree reversal to Bearish (H1 still lagging at +2.0)
        obs_rev = obs_bull.copy()
        obs_rev[0] = -2.5
        obs_rev[80] = -3.0
        obs_rev[81] = -1.5
        wb1, wr1, info1 = gating.compute_regime_weights(obs_rev)

        # Soft-MoE must flip dominant weight on Bar 1 (0-1 bar response)
        assert wr1 > wb1, f"Failed 0-1 bar flip: wb1={wb1}, wr1={wr1}"
        assert wr1 > 0.70, f"Bear weight insufficiently dominant: wr1={wr1}"

    def test_smooth_transition_zone_sweep(self) -> None:
        """Verify weight transitions smoothly across continuous momentum sweep."""
        gating = SoftMoEGatingNetwork()
        sweep = np.linspace(-1.0, 1.0, 101)
        prev_wb = 0.0

        for m5 in sweep:
            obs = np.zeros(137, dtype=np.float32)
            obs[80] = m5
            obs[81] = m5 * 0.5
            wb, wr, _ = gating.compute_regime_weights(obs)
            assert 0.0 <= wb <= 1.0
            assert 0.0 <= wr <= 1.0
            if m5 > -1.0:
                assert wb >= prev_wb - 1e-6, "Monotonicity violated in momentum sweep"
            prev_wb = wb


# ==============================================================================
# 3. CAPITAL PRESERVATION GUARD IN CHOPPY REGIMES
# ==============================================================================

class TestCapitalPreservationChoppyMarket:
    """Stress tests Capital Preservation Guard under noisy chop market simulations."""

    def test_choppy_market_simulation_100_percent_activation(self) -> None:
        """Simulate 300 bars of choppy/consolidation regime with chop_index in [0.60, 0.90]."""
        gating = SoftMoEGatingNetwork()
        guard = CapitalPreservationGuard()
        rng = np.random.RandomState(999)

        n_bars = 300
        for _ in range(n_bars):
            obs = np.zeros(137, dtype=np.float32)
            obs[0] = rng.randn() * 0.05
            obs[80] = rng.randn() * 0.05
            obs[81] = rng.randn() * 0.02
            chop = float(rng.uniform(0.60, 0.90))
            obs[93] = chop

            wb, wr, info = gating.compute_regime_weights(obs)
            is_active = guard.is_transition_active(wb, wr, info["chop_index"])
            assert is_active is True, f"Guard failed to activate with chop={chop}, wb={wb}, wr={wr}"

            # Market orders must be vetoed
            v_buy, act_buy, _ = guard.evaluate_order(1, wb, wr, info["chop_index"])
            v_sell, act_sell, _ = guard.evaluate_order(2, wb, wr, info["chop_index"])
            assert v_buy is True and act_buy == 0
            assert v_sell is True and act_sell == 0

            # Scale must be de-leveraged
            scale = guard.compute_defensive_scale(wb, wr, info["chop_index"])
            assert scale <= 0.50

    def test_breakout_orders_suppressed_in_severe_chop(self) -> None:
        """In severe chop (chop >= 0.70), stop orders (Actions 5, 6) must also be vetoed."""
        guard = CapitalPreservationGuard()
        v5, act5, r5 = guard.evaluate_order(5, 0.5, 0.5, chop_index=0.75)
        v6, act6, r6 = guard.evaluate_order(6, 0.5, 0.5, chop_index=0.75)
        assert v5 is True and act5 == 0
        assert v6 is True and act6 == 0


# ==============================================================================
# 4. DISCOVERY TESTS: PATHOLOGICAL INPUTS & ROUTER INTEGRATION BUGS
# ==============================================================================

class TestEmpiricalChallengerDiscoveries:
    """Tests reproducing the concrete vulnerabilities identified during analysis."""

    def test_discover_router_standalone_deadlock_without_regime_detector(self) -> None:
        """
        FINDING 1: When DualChampionRouter is initialized without legacy regime_detector,
        it fails to update self.current_regime from Soft-MoE weights in act_nam_tac_nhan,
        leaving the router frozen in 'CHOP' and routing 100% to Bear Specialist.
        """
        bull = MockRecurrentSpecialist("bull")
        bear = MockRecurrentSpecialist("bear")
        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            regime_detector=None,  # Modern Soft-MoE standalone mode
        )

        # Bullish market observation (strong upward momentum)
        obs = np.zeros(137, dtype=float)
        obs[80] = 4.0
        obs[81] = 2.0
        obs[88] = 3.0

        for _ in range(5):
            router.act_nam_tac_nhan(obs)

        wb, wr = router.soft_moe_weights
        assert wb > 0.90, f"Soft-MoE weights correctly computed bull={wb}"

        # BUG ASSERTION:
        # In current code: router.current_regime is NOT updated, remaining 'CHOP',
        # and active_policy remains bear_policy (bull.step_count == 0).
        is_bug_present = (router.current_regime == "CHOP" and bull.step_count == 0)
        # We record this finding:
        if is_bug_present:
            pytest.fail(
                "VULNERABILITY CONFIRMED: Standalone DualChampionRouter with regime_detector=None "
                "fails to transition to Bull specialist during 100% Bull momentum!"
            )

    def test_discover_20d_nan_input_bypasses_sanitization(self) -> None:
        """
        FINDING 2: When a 20-dim feature vector is passed to SoftMoEGatingNetwork,
        extract_multi_timescale_features returns arr directly without math.isfinite check,
        resulting in NaN weights becoming 1.0 (BULL) with 0% defense.
        """
        gating = SoftMoEGatingNetwork()
        arr20_nan = np.full(NUM_MTS_FEATURES, np.nan, dtype=np.float32)

        wb, wr, info = gating.compute_regime_weights(arr20_nan)

        # If NaN was converted to Bull 1.0 without defense, bug is present
        if wb == 1.0 and wr == 0.0 and info["transition_flag"] is False:
            pytest.fail(
                "VULNERABILITY CONFIRMED: 20-D NaN input bypassed sanitization and produced "
                "100% Bull weight (wb=1.0, wr=0.0) with capital preservation OFF!"
            )


# ==============================================================================
# 5. CPU LATENCY BENCHMARK
# ==============================================================================

class TestCPULatencyBenchmarkUnderLoad:
    """Verifies strict CPU SLA (<= 0.5 ms mean, <= 1.0 ms P99) under 5,000 tick load."""

    def test_benchmark_5000_sequential_ticks(self) -> None:
        gating = SoftMoEGatingNetwork()
        rng = np.random.RandomState(42)

        # Warm-up
        for _ in range(100):
            gating.compute_regime_weights(rng.randn(137).astype(np.float32))

        n_ticks = 5000
        latencies_ms = []

        for _ in range(n_ticks):
            obs = rng.randn(137).astype(np.float32)
            t0 = time.perf_counter()
            wb, wr, info = gating.compute_regime_weights(obs)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        median_lat = float(np.median(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        p99_lat = float(np.percentile(latencies_ms, 99))
        max_lat = float(np.max(latencies_ms))

        print(f"\nLatency Benchmark (5,000 ticks):")
        print(f"  Mean:   {mean_lat:.4f} ms")
        print(f"  Median: {median_lat:.4f} ms")
        print(f"  P95:    {p95_lat:.4f} ms")
        print(f"  P99:    {p99_lat:.4f} ms")
        print(f"  Max:    {max_lat:.4f} ms")

        assert mean_lat <= 0.50, f"Mean latency {mean_lat:.4f} ms > 0.50 ms SLA"
        assert p99_lat <= 1.00, f"P99 latency {p99_lat:.4f} ms > 1.00 ms SLA"
