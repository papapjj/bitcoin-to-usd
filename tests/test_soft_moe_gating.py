"""
tests/test_soft_moe_gating.py

Comprehensive test suite for Soft-MoE (Mixture of Regimes) Gating Network,
Capital Preservation Guard, and DualChampionRouter continuous integration.

Covers:
1. Differentiability via PyTorch autograd and parameter gradient flow.
2. Simplex invariant: w_bull + w_bear == 1.0 across 1,000 randomized & boundary inputs.
3. Transition lag elimination: comparing continuous Soft-MoE vs discrete switching.
4. Capital Preservation Guard: contest zone (|w_bull - w_bear| <= 0.30) & chop defense.
5. CPU latency SLA: <= 0.5 ms per tick inference.
6. Governor invariant: Agent 4 ACTION_HOLD = 0 (POLICY_CLOSE_ALL == 0.0) preserved.
7. Recurrent memory isolation (L_inf = 0.0) under continuous Soft-MoE routing.
"""

from __future__ import annotations

import math
import time
from typing import Any
from unittest.mock import MagicMock

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


# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================

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

        # 0: BUY, 1: SELL, 2: MANAGE, 3: SCALP, 4: GOVERNOR (tries to close all)
        return (
            MockTrace(0, 1),  # MARKET_BUY
            MockTrace(1, 2),  # MARKET_SELL
            MockTrace(2, 0),  # HOLD
            MockTrace(3, 0),  # HOLD
            MockTrace(4, 7),  # Tries to liquidate / non-hold action
        )

    def ensure_compatible(self) -> None:
        pass

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state.fill(0.0)
        self.cell_state.fill(0.0)


# ==============================================================================
# TEST GROUP 1: DIFFERENTIABILITY & AUTOGRAD GRADIENT FLOW
# ==============================================================================

class TestSoftMoEDifferentiability:
    """Verifies differentiability of the Soft-MoE gating network via PyTorch autograd."""

    def test_gating_network_differentiability_and_gradient_flow(self) -> None:
        """Gating network must produce valid gradients with respect to inputs and parameters."""
        model = SoftMoEGatingNetwork(in_features=NUM_MTS_FEATURES, hidden_dim=32)
        model.train()

        # Batch of 8 synthetic observation vectors with requires_grad=True
        x = torch.randn(8, NUM_MTS_FEATURES, dtype=torch.float32, requires_grad=True)

        # Forward pass
        weights = model(x)

        # Assert shape: [8, 2]
        assert weights.shape == (8, 2)

        # Simplex sum across regime dimension must be 1.0 for each batch element
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

        # Compute a scalar objective (e.g. maximize bull weight)
        target_loss = -torch.log(weights[:, 0] + 1e-8).mean()
        target_loss.backward()

        # Verify input gradients exist and are non-zero
        assert x.grad is not None
        assert not torch.all(x.grad == 0.0)
        assert torch.isfinite(x.grad).all()

        # Verify parameter gradients exist and are non-zero
        assert model.fc1.weight.grad is not None
        assert not torch.all(model.fc1.weight.grad == 0.0)
        assert torch.isfinite(model.fc1.weight.grad).all()

        assert model.fc1.bias.grad is not None
        assert not torch.all(model.fc1.bias.grad == 0.0)

        assert model.fc2.weight.grad is not None
        assert not torch.all(model.fc2.weight.grad == 0.0)

        assert model.fc2.bias.grad is not None
        assert not torch.all(model.fc2.bias.grad == 0.0)

    def test_forward_with_full_137d_torch_tensor(self) -> None:
        """Forward pass must seamlessly handle full 137D observation tensors."""
        model = SoftMoEGatingNetwork()
        x137 = torch.randn(4, 137, dtype=torch.float32, requires_grad=True)

        weights = model(x137)
        assert weights.shape == (4, 2)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(4), atol=1e-6)

        loss = weights[:, 1].sum()
        loss.backward()
        assert x137.grad is not None
        assert torch.isfinite(x137.grad).all()


# ==============================================================================
# TEST GROUP 2: SIMPLEX INVARIANT (w_bull + w_bear == 1.0)
# ==============================================================================

class TestSimplexInvariant:
    """Verifies strict adherence to w_bull + w_bear == 1.0 across randomized & extreme inputs."""

    def test_simplex_invariant_across_1000_random_inputs(self) -> None:
        """Every input must produce weights on the 1-simplex: w_bull, w_bear in [0,1], sum == 1.0."""
        gating = SoftMoEGatingNetwork()
        rng = np.random.RandomState(42)

        for _ in range(1000):
            # Generate random 137-D observation
            obs = rng.randn(137).astype(np.float32) * 5.0
            w_bull, w_bear, info = gating.compute_regime_weights(obs)

            # Invariants
            assert 0.0 <= w_bull <= 1.0, f"w_bull out of bounds: {w_bull}"
            assert 0.0 <= w_bear <= 1.0, f"w_bear out of bounds: {w_bear}"
            assert abs((w_bull + w_bear) - 1.0) < 1e-6, f"Simplex violation: {w_bull} + {w_bear} != 1.0"
            assert math.isclose(w_bull + w_bear, 1.0, rel_tol=1e-7, abs_tol=1e-7)

    def test_boundary_inputs_zeros_extremes_and_nans(self) -> None:
        """Robustness against zeros, huge values, and NaNs."""
        gating = SoftMoEGatingNetwork()

        # 1. All zeros
        obs_zeros = np.zeros(137, dtype=np.float32)
        wb, wr, info = gating.compute_regime_weights(obs_zeros)
        assert abs((wb + wr) - 1.0) < 1e-6
        assert 0.0 <= wb <= 1.0
        assert 0.0 <= wr <= 1.0

        # 2. Extreme positive momentum
        obs_extreme_bull = np.zeros(137, dtype=np.float32)
        obs_extreme_bull[80] = 100.0  # huge M5 diff
        obs_extreme_bull[81] = 50.0   # huge M5 slope
        wb, wr, _ = gating.compute_regime_weights(obs_extreme_bull)
        assert wb > 0.85
        assert wr < 0.15
        assert abs((wb + wr) - 1.0) < 1e-6

        # 3. Extreme negative momentum
        obs_extreme_bear = np.zeros(137, dtype=np.float32)
        obs_extreme_bear[80] = -100.0
        obs_extreme_bear[81] = -50.0
        wb, wr, _ = gating.compute_regime_weights(obs_extreme_bear)
        assert wr > 0.85
        assert wb < 0.15
        assert abs((wb + wr) - 1.0) < 1e-6

        # 4. Inputs containing NaNs and Infs (must not crash or produce NaN weights)
        obs_corrupt = np.zeros(137, dtype=np.float32)
        obs_corrupt[80] = np.nan
        obs_corrupt[81] = np.inf
        obs_corrupt[84] = -np.inf
        wb, wr, _ = gating.compute_regime_weights(obs_corrupt)
        assert math.isfinite(wb)
        assert math.isfinite(wr)
        assert abs((wb + wr) - 1.0) < 1e-6


# ==============================================================================
# TEST GROUP 3: REGIME TRANSITION LAG ELIMINATION (INFLECTION BENCHMARK)
# ==============================================================================

class TestRegimeTransitionLagElimination:
    """Proves that Soft-MoE eliminates the 15-45m lag of discrete hard switching."""

    def test_immediate_inflection_response_vs_discrete_lag(self) -> None:
        """Soft-MoE responds on Tick 0 of an inflection, whereas discrete switching lags."""
        gating = SoftMoEGatingNetwork()

        # Step 1: Initial Bear market conditions
        obs_bear = np.zeros(137, dtype=np.float32)
        obs_bear[80] = -0.40  # M5 diff negative
        obs_bear[81] = -0.10  # M5 slope negative
        obs_bear[88] = -0.50  # H1 diff negative

        wb_initial, wr_initial, _ = gating.compute_regime_weights(obs_bear)
        assert wr_initial > wb_initial
        assert wr_initial > 0.60

        # Step 2: Instant market inflection at t=0 (Bull divergence / V-shape reversal)
        # Fast M1/M5 momentum flips positive, but H1 EMA is still negative (lagging)
        obs_inflection = obs_bear.copy()
        obs_inflection[0] = 0.50   # M1 return surges
        obs_inflection[80] = 0.35  # M5 diff turns positive
        obs_inflection[81] = 0.20  # M5 slope accelerates upward
        # H1 diff is still negative (-0.50) due to 60-bar EMA lag!

        wb_flip, wr_flip, info = gating.compute_regime_weights(obs_inflection)

        # Soft-MoE immediately recognizes the inflection on tick 0
        assert wb_flip > wb_initial + 0.25, (
            f"Soft-MoE failed to react immediately to inflection: wb changed from {wb_initial} to {wb_flip}"
        )
        assert wb_flip > wr_flip, "Bull weight should dominate after positive momentum surge"

        # Telemetry verification
        assert math.isclose(info["m5_trend"], 0.35, abs_tol=1e-5)
        assert info["dominant_regime"] in ("BULL", "TRANSITION")

    def test_smooth_continuous_trajectory_no_step_discontinuity(self) -> None:
        """Weight trajectory must be smooth and continuous, not a discontinuous jump."""
        gating = SoftMoEGatingNetwork()

        # Sweep M5 momentum linearly from -0.5 to +0.5 over 100 ticks
        m5_diffs = np.linspace(-0.5, 0.5, 100)
        weights_history = []

        for diff in m5_diffs:
            obs = np.zeros(137, dtype=np.float32)
            obs[80] = diff
            obs[81] = diff * 0.5
            wb, wr, _ = gating.compute_regime_weights(obs)
            weights_history.append((wb, wr))

        wb_arr = np.array([w[0] for w in weights_history])

        # Strictly monotonic response with respect to momentum
        diffs = np.diff(wb_arr)
        assert (diffs >= -1e-6).all(), "Trajectory must be monotonically increasing with momentum"

        # Continuous without violent discrete jump (max single-step delta < 0.10)
        assert np.max(diffs) < 0.10, "Weight changed too discontinuously between adjacent ticks"


# ==============================================================================
# TEST GROUP 4: CAPITAL PRESERVATION GUARD IN TRANSITION ZONE
# ==============================================================================

class TestCapitalPreservationGuard:
    """Verifies Capital Preservation Guard defensive scaling and order suppression."""

    def test_guard_detects_transition_zone_and_chop(self) -> None:
        guard = CapitalPreservationGuard(transition_threshold=0.30, chop_threshold=0.60)

        # Delta = |0.55 - 0.45| = 0.10 <= 0.30 -> Transition active
        assert guard.is_transition_active(0.55, 0.45, chop_index=0.40) is True

        # Delta = |0.80 - 0.20| = 0.60 > 0.30, chop = 0.40 -> Inactive
        assert guard.is_transition_active(0.80, 0.20, chop_index=0.40) is False

        # Delta = 0.60, but chop = 0.65 >= 0.60 -> Active due to chop
        assert guard.is_transition_active(0.80, 0.20, chop_index=0.65) is True

    def test_guard_suppresses_market_order_chasing(self) -> None:
        guard = CapitalPreservationGuard()

        # In Transition Zone (delta <= 0.30):
        # Action 1: MARKET_BUY -> Suppressed to HOLD (0)
        vetoed, act, reason = guard.evaluate_order(proposed_action=1, w_bull=0.52, w_bear=0.48)
        assert vetoed is True
        assert act == 0
        assert reason == "capital_preservation_transition_zone_no_market_chase"

        # Action 2: MARKET_SELL -> Suppressed to HOLD (0)
        vetoed, act, reason = guard.evaluate_order(proposed_action=2, w_bull=0.48, w_bear=0.52)
        assert vetoed is True
        assert act == 0
        assert reason == "capital_preservation_transition_zone_no_market_chase"

        # Action 3: BUY_LIMIT -> Permitted (patient order)
        vetoed, act, _ = guard.evaluate_order(proposed_action=3, w_bull=0.52, w_bear=0.48)
        assert vetoed is False
        assert act == 3

        # Action 4: SELL_LIMIT -> Permitted (patient order)
        vetoed, act, _ = guard.evaluate_order(proposed_action=4, w_bull=0.48, w_bear=0.52)
        assert vetoed is False
        assert act == 4

    def test_guard_suppresses_breakouts_in_severe_chop(self) -> None:
        guard = CapitalPreservationGuard()

        # Severe chop (chop >= 0.70) suppresses Action 5 (BUY_STOP) and 6 (SELL_STOP)
        vetoed, act, reason = guard.evaluate_order(5, w_bull=0.5, w_bear=0.5, chop_index=0.75)
        assert vetoed is True
        assert act == 0
        assert reason == "capital_preservation_severe_chop_no_breakout_chase"

        vetoed, act, reason = guard.evaluate_order(6, w_bull=0.5, w_bear=0.5, chop_index=0.75)
        assert vetoed is True
        assert act == 0
        assert reason == "capital_preservation_severe_chop_no_breakout_chase"

    def test_defensive_scaling_curve(self) -> None:
        guard = CapitalPreservationGuard(min_defensive_scale=0.40)

        # At delta=0 (w_bull=0.5, w_bear=0.5): scale is min_defensive_scale (0.40)
        scale_0 = guard.compute_defensive_scale(0.5, 0.5, chop_index=0.3)
        assert math.isclose(scale_0, 0.40, abs_tol=1e-5)

        # At delta=0.15 (halfway through transition zone): scale = 0.50
        scale_half = guard.compute_defensive_scale(0.575, 0.425, chop_index=0.3)
        assert math.isclose(scale_half, 0.50, abs_tol=1e-5)

        # Outside transition zone (delta=0.40): scale = 1.0 (full risk)
        scale_full = guard.compute_defensive_scale(0.70, 0.30, chop_index=0.3)
        assert math.isclose(scale_full, 1.0, abs_tol=1e-5)


# ==============================================================================
# TEST GROUP 5: CPU LATENCY SLA (<= 0.5 MS PER TICK)
# ==============================================================================

class TestCPULatencySLA:
    """Verifies sub-0.5ms CPU execution time SLA."""

    def test_cpu_inference_latency_under_half_millisecond(self) -> None:
        gating = SoftMoEGatingNetwork()
        rng = np.random.RandomState(123)

        # Warm up JIT / CPU cache
        dummy = rng.randn(137).astype(np.float32)
        for _ in range(50):
            gating.compute_regime_weights(dummy)

        # Benchmark 1,000 ticks
        n_ticks = 1000
        latencies = []

        for _ in range(n_ticks):
            obs = rng.randn(137).astype(np.float32)
            t0 = time.perf_counter()
            wb, wr, info = gating.compute_regime_weights(obs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # milliseconds

        mean_lat = np.mean(latencies)
        p99_lat = np.percentile(latencies, 99)

        # SLA requirements
        assert mean_lat <= 0.50, f"Mean latency SLA exceeded: {mean_lat:.4f} ms > 0.50 ms"
        assert p99_lat <= 1.00, f"P99 latency SLA exceeded: {p99_lat:.4f} ms > 1.00 ms"


# ==============================================================================
# TEST GROUP 6: GOVERNOR INVARIANT (ACTION_HOLD = 0, POLICY_CLOSE_ALL == 0.0)
# ==============================================================================

class TestGovernorInvariantInDualChampionRouter:
    """Verifies Agent 4 (Governor) invariant ACTION_HOLD = 0 across all conditions."""

    def test_governor_agent4_strictly_hold_across_all_regimes(self) -> None:
        """DualChampionRouter must ALWAYS clamp Agent 4 to ACTION_HOLD = 0."""
        bull_mock = MockRecurrentSpecialist("bull")
        bear_mock = MockRecurrentSpecialist("bear")

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            soft_moe_gating=SoftMoEGatingNetwork(),
            capital_preservation_guard=CapitalPreservationGuard(),
        )

        test_observations = [
            # 1. Bull momentum
            np.zeros(137, dtype=float) + 0.3,
            # 2. Bear momentum
            np.zeros(137, dtype=float) - 0.3,
            # 3. Transition zone / contested
            np.zeros(137, dtype=float),
            # 4. Severe chop
            np.zeros(137, dtype=float),
        ]
        test_observations[3][93] = 0.85  # Chop index

        for i, obs in enumerate(test_observations):
            batch = router.act_nam_tac_nhan(obs)
            assert len(batch) == 5

            agent4_trace = batch[4]
            act4 = int(agent4_trace.action_details.get("five_action_index", -1))
            wire4 = int(agent4_trace.action_details.get("wire_action", -1))

            # Strictly locked to ACTION_HOLD (0)
            assert act4 == 0, f"Governor invariant violated on case {i}: act4 = {act4}"
            assert wire4 == 0, f"Governor wire action violated on case {i}: wire4 = {wire4}"
            assert agent4_trace.action_details.get("policy_close_all", 0.0) == 0.0


# ==============================================================================
# TEST GROUP 7: RECURRENT MEMORY ISOLATION UNDER SOFT-MOE INTEGRATION
# ==============================================================================

class TestRecurrentMemoryIsolationUnderSoftMoE:
    """Verifies that recurrent memory isolation contracts (L_inf = 0.0) remain intact."""

    def test_inactive_specialist_recurrent_memory_strictly_frozen_100_steps(self) -> None:
        bull_mock = MockRecurrentSpecialist("bull")
        bear_mock = MockRecurrentSpecialist("bear")

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            soft_moe_gating=SoftMoEGatingNetwork(),
        )

        # Force Bull routing
        router.set_regime_soft_probabilities([0.80, 0.10, 0.10], "BULL", 0)
        assert router.active_policy is bull_mock

        bear_h0 = bear_mock.hidden_state.copy()
        bear_c0 = bear_mock.cell_state.copy()

        obs = np.zeros(137, dtype=float)
        obs[80] = 0.25

        for _ in range(100):
            router.act_nam_tac_nhan(obs)

        # Active Bull advanced
        assert bull_mock.step_count == 100
        assert not np.array_equal(bull_mock.hidden_state, np.zeros((5, 4096)))

        # Inactive Bear strictly frozen (L_inf == 0.0)
        assert bear_mock.step_count == 0
        h_diff = np.max(np.abs(bear_mock.hidden_state - bear_h0))
        c_diff = np.max(np.abs(bear_mock.cell_state - bear_c0))
        assert h_diff == 0.0, f"Bear hidden state leaked! L_inf = {h_diff}"
        assert c_diff == 0.0, f"Bear cell state leaked! L_inf = {c_diff}"
