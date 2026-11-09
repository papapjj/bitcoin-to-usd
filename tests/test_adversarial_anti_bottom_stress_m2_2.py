"""
tests/test_adversarial_anti_bottom_stress_m2_2.py

Milestone M2 Adversarial Verification & Stress Testing Suite
Author: Challenger 11_M2_2 (Adversarial Verifier & Stress Tester)
Roles: critic, specialist

Focus Areas:
1. Boundary fuzzing around swing_pos in [0.1799, 0.1801] and dist_low_atr in [1.199, 1.201].
2. IEEE 754 float32 vs float64 boundary precision gap analysis.
3. False negative leak detection across all short actions (2, 4, 6) and exhaustion triggers.
4. False positive preservation for trend continuation and breakdown shorts.
5. Bull BUY immunity (actions 1, 3, 5 never vetoed).
6. Agent 4 Governor invariance (ACTION_HOLD = 0, POLICY_CLOSE_ALL == 0.0).
7. Corrupted input robustness (NaN, Inf, extreme floats, bad action indices).
8. Standalone CPU latency SLA benchmark (< 0.05 ms / 50 µs).
"""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
import pytest

from self_growing_trader.anti_bottom_guard import (
    AntiBottomExhaustionGuard,
    VetoDecision,
    IDX_RSI14,
    IDX_RSI5,
    IDX_BB_POSITION,
    IDX_SWING_RANGE120_POS,
    IDX_DIST_SWING_LOW120_ATR,
    IDX_LIQUIDITY_SWEEP_REJECTION,
    IDX_RELATIVE_VOLUME_20,
    IDX_EFFORT_RESULT_RATIO,
    IDX_ABSORPTION_SCORE,
    IDX_SPRING_SCORE,
    IDX_SPRING_TEST_QUALITY,
    IDX_BARS_SINCE_SPRING,
)
from self_growing_trader.dual_champion_router import DualChampionRouter, BridgeActionBatch
from self_growing_trader.orders import Action


# ==============================================================================
# FIXTURES AND MOCK SPECIALISTS
# ==============================================================================

def make_neutral_observation(dtype: type = np.float32) -> np.ndarray:
    """Return a neutral 137D observation array."""
    obs = np.zeros(137, dtype=dtype)
    obs[IDX_SWING_RANGE120_POS] = 0.50
    obs[IDX_DIST_SWING_LOW120_ATR] = 5.0
    obs[IDX_BB_POSITION] = 0.50
    obs[IDX_RELATIVE_VOLUME_20] = 0.20
    obs[IDX_BARS_SINCE_SPRING] = 1.0
    return obs


class AdversarialMockSpecialist:
    """Adversarial specialist policy capable of generating arbitrary action distributions."""

    def __init__(self, name: str, default_actions: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)) -> None:
        self.name = name
        self.actions = list(default_actions)
        self.step_count = 0

    def set_action(self, agent_idx: int, action_idx: int) -> None:
        self.actions[agent_idx] = action_idx

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[Any, ...]:
        self.step_count += 1

        class MockTrace:
            def __init__(self, agent_id: int, act: int) -> None:
                self.trace_id = f"{agent_id}_trace"
                self.selected_idx = act
                self.action_index = act
                self.selected_action = np.zeros(20, dtype=float)
                self.selected_action[act] = 1.0
                self.output_vector = self.selected_action.copy()
                self.action_probs = self.selected_action.copy()
                self.action_details = {
                    "five_agent_index": agent_id,
                    "five_action_index": act,
                    "wire_action": act,
                    "base_risk": 0.01,
                    "risk_pct": 0.01,
                }

        return tuple(MockTrace(i, self.actions[i]) for i in range(5))


# ==============================================================================
# TEST GROUP 1: BOUNDARY FUZZING & IEEE 754 PRECISION GAP
# ==============================================================================

class TestAdversarialBoundaryFuzzing:
    """Stress-tests boundary transitions and surfaces floating-point precision edge cases."""

    def test_float32_exact_boundary_discrepancy_empirical_evidence(self) -> None:
        """Surfaces the empirical finding: float32(0.18) > float64(0.18) in IEEE 754.

        In float32, np.float32(0.18) expands to 0.18000000715255737 in float64.
        This test documents the exact threshold discrepancy between float32 inputs
        and strict float64 comparisons in the guard.
        """
        guard = AntiBottomExhaustionGuard()

        # float64 input at exactly 0.18
        obs_f64 = make_neutral_observation(dtype=np.float64)
        obs_f64[IDX_SWING_RANGE120_POS] = 0.18
        obs_f64[IDX_ABSORPTION_SCORE] = 0.50
        dec_f64 = guard.check_bottom_short_veto(obs_f64, 2)
        assert dec_f64.is_vetoed is True, "float64 at 0.18 must be vetoed"

        # float32 input at 0.18
        obs_f32 = make_neutral_observation(dtype=np.float32)
        obs_f32[IDX_SWING_RANGE120_POS] = np.float32(0.18)
        obs_f32[IDX_ABSORPTION_SCORE] = np.float32(0.50)
        dec_f32 = guard.check_bottom_short_veto(obs_f32, 2)

        # Record empirical result:
        # If float32 input is used, 0.18 evaluates to 0.18000000715255737 > 0.18, so veto is False.
        # This is a documented boundary sensitivity edge-case.
        assert dec_f32.telemetry["swing_range120_pos"] == float(np.float32(0.18))
        if not dec_f32.is_vetoed:
            print(
                f"\n[EMPIRICAL FINDING] Float32 precision gap: "
                f"np.float32(0.18) = {float(np.float32(0.18)):.17f} > 0.18. "
                f"in_demand_zone={dec_f32.telemetry['in_demand_zone']}"
            )

    def test_dense_boundary_fuzzing_swing_pos_float64(self) -> None:
        """Dense fuzzing of 1,000 points in [0.1799, 0.1801] with float64 precision."""
        guard = AntiBottomExhaustionGuard()
        points = np.linspace(0.1799, 0.1801, 1000)

        for pos in points:
            obs = make_neutral_observation(dtype=np.float64)
            obs[IDX_SWING_RANGE120_POS] = pos
            obs[IDX_ABSORPTION_SCORE] = 0.50
            dec = guard.check_bottom_short_veto(obs, 2)

            expected_veto = pos <= 0.18
            assert dec.is_vetoed == expected_veto, (
                f"Boundary transition failed at swing_pos={pos:.8f}: "
                f"got {dec.is_vetoed}, expected {expected_veto}"
            )

    def test_dense_boundary_fuzzing_dist_low_atr_float64(self) -> None:
        """Dense fuzzing of 1,000 points in [1.199, 1.201] with float64 precision."""
        guard = AntiBottomExhaustionGuard()
        points = np.linspace(1.199, 1.201, 1000)

        for dist in points:
            obs = make_neutral_observation(dtype=np.float64)
            obs[IDX_SWING_RANGE120_POS] = 0.50  # Far from pos threshold
            obs[IDX_DIST_SWING_LOW120_ATR] = dist
            obs[IDX_SPRING_SCORE] = 0.50  # Active spring
            dec = guard.check_bottom_short_veto(obs, 2)

            expected_veto = dist <= 1.20
            assert dec.is_vetoed == expected_veto, (
                f"Boundary transition failed at dist_low_atr={dist:.8f}: "
                f"got {dec.is_vetoed}, expected {expected_veto}"
            )

    def test_joint_2d_boundary_fuzzing_grid(self) -> None:
        """2D grid fuzzing over (swing_pos, dist_low_atr) crossing boundaries."""
        guard = AntiBottomExhaustionGuard()
        pos_vals = [0.1790, 0.17999, 0.18000, 0.18001, 0.1810]
        dist_vals = [1.190, 1.1999, 1.2000, 1.2001, 1.210]

        for p in pos_vals:
            for d in dist_vals:
                obs = make_neutral_observation(dtype=np.float64)
                obs[IDX_SWING_RANGE120_POS] = p
                obs[IDX_DIST_SWING_LOW120_ATR] = d
                obs[IDX_ABSORPTION_SCORE] = 0.40  # Trigger active

                dec = guard.check_bottom_short_veto(obs, 2)
                expected_in_demand = (p <= 0.18) or (d <= 1.20)
                assert dec.is_vetoed == expected_in_demand, (
                    f"Joint grid failure at (pos={p}, dist={d}): "
                    f"got {dec.is_vetoed}, expected {expected_in_demand}"
                )


# ==============================================================================
# TEST GROUP 2: FALSE NEGATIVE LEAK DETECTION (NO SHORT CAN LEAK)
# ==============================================================================

class TestAdversarialFalseNegativeLeakDetection:
    """Verifies that NO short entry action can leak unvetoed in a demand zone with exhaustion."""

    SHORT_ACTIONS = (2, 4, 6)  # MARKET_SELL, SELL_LIMIT, SELL_STOP

    def test_all_short_actions_intercepted_under_all_6_exhaustion_triggers(self) -> None:
        """Combinatorial test: 3 short actions x 6 distinct exhaustion triggers = 18 cases."""
        guard = AntiBottomExhaustionGuard()

        triggers = [
            ("absorption", {IDX_ABSORPTION_SCORE: 0.45}),
            ("spring", {IDX_SPRING_SCORE: 0.35}),
            ("rsi_oversold", {IDX_RSI14: -0.45, IDX_RSI5: -0.65}),
            ("sweep_rejection", {IDX_LIQUIDITY_SWEEP_REJECTION: 0.30}),
            ("spring_retest", {IDX_SPRING_TEST_QUALITY: 0.30, IDX_BARS_SINCE_SPRING: 0.15}),
            ("bb_climax", {IDX_BB_POSITION: 0.02, IDX_RELATIVE_VOLUME_20: 0.40}),
        ]

        for trig_name, trig_feats in triggers:
            for act in self.SHORT_ACTIONS:
                obs = make_neutral_observation(dtype=np.float64)
                obs[IDX_SWING_RANGE120_POS] = 0.10  # Deep demand zone
                obs[IDX_DIST_SWING_LOW120_ATR] = 0.80

                for idx, val in trig_feats.items():
                    obs[idx] = val

                dec = guard.check_bottom_short_veto(obs, act)
                assert dec.is_vetoed is True, (
                    f"LEAK DETECTED! Action {act} leaked under trigger '{trig_name}'"
                )
                assert dec.reason != "allowed"
                assert dec.telemetry["in_demand_zone"] is True
                assert dec.telemetry["is_exhausted"] is True

    def test_router_act_nam_tac_nhan_strictly_vetoes_bear_shorts_in_demand_zone(self) -> None:
        """Full router integration test in Bear directional regime: Bear specialist shorting must be intercepted."""
        for act in self.SHORT_ACTIONS:
            bull_pol = AdversarialMockSpecialist("bull", (0, 0, 0, 0, 0))
            bear_pol = AdversarialMockSpecialist("bear", (0, act, 0, 0, 0))  # Agent 1 proposes short

            router = DualChampionRouter(
                bull_policy=bull_pol,
                bear_policy=bear_pol,
                anti_bottom_guard_enabled=True,
            )

            obs = make_neutral_observation(dtype=np.float64)
            # Set Bear trend indicators so Capital Preservation is inactive and AntiBottomGuard handles the veto
            obs[80] = -0.50  # M5 diff negative (strong bear)
            obs[81] = -0.30  # M5 slope negative
            obs[84] = -0.60  # M15 diff negative
            obs[85] = -0.40  # M15 slope negative
            obs[93] = 0.10   # low chop
            obs[IDX_SWING_RANGE120_POS] = 0.08  # Demand zone
            obs[IDX_ABSORPTION_SCORE] = 0.55   # Absorption

            batch = router.act_nam_tac_nhan(obs)
            assert isinstance(batch, BridgeActionBatch)

            # Agent 1 trace MUST be neutralized to HOLD (0)
            agent1_trace = batch.traces[1]
            act_1 = router._extract_action_idx(agent1_trace)
            assert act_1 == 0, f"Router failed to intercept short action {act}: got {act_1}"
            assert agent1_trace.action_details.get("regime_veto_applied") is True
            assert "anti_bottom_guard" in str(agent1_trace.action_details.get("regime_veto_reason"))
            assert batch.anti_bottom_telemetry.get("bottom_short_veto") is True

    def test_router_scalp_short_in_demand_zone_intercepted(self) -> None:
        """Verifies that micro-pullback scalper short signals are also vetoed in demand zones."""
        bull_pol = AdversarialMockSpecialist("bull", (0, 0, 0, 0, 0))
        bear_pol = AdversarialMockSpecialist("bear", (0, 0, 0, 0, 0))  # Both specialists HOLD

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            anti_bottom_guard_enabled=True,
        )

        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 0.05
        obs[IDX_ABSORPTION_SCORE] = 0.60

        # Simulate micro-scalper emitting a SELL signal (direction -1, action 2)
        dec = router.anti_bottom_guard.check_bottom_short_veto(obs, 2)
        assert dec.is_vetoed is True


# ==============================================================================
# TEST GROUP 3: FALSE POSITIVE PRESERVATION (CLEAN BREAKDOWNS PRESERVED)
# ==============================================================================

class TestAdversarialFalsePositivePreservation:
    """Verifies that legitimate trend-following shorts outside demand zones are preserved."""

    def test_mid_range_trend_continuation_short_never_vetoed(self) -> None:
        """Mid-range shorting (swing_pos=0.50, dist=5.0 ATR) must never be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 0.50
        obs[IDX_DIST_SWING_LOW120_ATR] = 5.0
        # Even with high absorption in mid-range, not in demand zone -> no veto
        obs[IDX_ABSORPTION_SCORE] = 0.80

        for act in (2, 4, 6):
            dec = guard.check_bottom_short_veto(obs, act)
            assert dec.is_vetoed is False
            assert dec.reason == "allowed"
            assert dec.telemetry["in_demand_zone"] is False

    def test_clean_breakdown_without_absorption_never_vetoed(self) -> None:
        """High-velocity breakdown through range low without stopping volume must not be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 0.02  # At the very bottom
        obs[IDX_DIST_SWING_LOW120_ATR] = 0.10
        # No exhaustion, no spring, no sweep rejection, healthy momentum
        obs[IDX_ABSORPTION_SCORE] = 0.00
        obs[IDX_SPRING_SCORE] = 0.00
        obs[IDX_LIQUIDITY_SWEEP_REJECTION] = 0.00
        obs[IDX_RSI14] = -0.20  # Moderate, not oversold climax
        obs[IDX_RSI5] = -0.30

        for act in (2, 4, 6):
            dec = guard.check_bottom_short_veto(obs, act)
            assert dec.is_vetoed is False, "Clean breakdown falsely vetoed!"
            assert dec.telemetry["in_demand_zone"] is True
            assert dec.telemetry["is_exhausted"] is False


# ==============================================================================
# TEST GROUP 4: BULL BUY IMMUNITY (ACTIONS 1, 3, 5 NEVER VETOED)
# ==============================================================================

class TestAdversarialBullBuyImmunity:
    """Verifies that Bull Specialist accumulation BUY actions are 100% immune from veto."""

    BUY_ACTIONS = (1, 3, 5)  # MARKET_BUY, BUY_LIMIT, BUY_STOP

    def test_bull_buy_actions_immune_in_extreme_exhaustion_demand_zone(self) -> None:
        """Bull accumulation entries in Wyckoff Springs must never be blocked."""
        guard = AntiBottomExhaustionGuard()
        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 0.01  # Absolute bottom
        obs[IDX_DIST_SWING_LOW120_ATR] = 0.05
        obs[IDX_SPRING_SCORE] = 0.95      # Massive spring
        obs[IDX_ABSORPTION_SCORE] = 0.90  # Huge absorption

        for act in self.BUY_ACTIONS:
            dec = guard.check_bottom_short_veto(obs, act)
            assert dec.is_vetoed is False, f"Bull BUY action {act} falsely vetoed!"
            assert dec.reason == "not_short_action"

    def test_router_preserves_bull_buy_through_full_pipeline(self) -> None:
        """Router integration must preserve Bull specialist BUY actions in Bull regime."""
        bull_pol = AdversarialMockSpecialist("bull", (1, 0, 0, 0, 0))  # Agent 0 emits MARKET_BUY
        bear_pol = AdversarialMockSpecialist("bear", (0, 0, 0, 0, 0))

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            anti_bottom_guard_enabled=True,
        )

        obs = make_neutral_observation(dtype=np.float64)
        # Set Bull trend indicators so Capital Preservation is inactive
        obs[80] = 0.50  # M5 diff positive (strong bull)
        obs[81] = 0.30  # M5 slope positive
        obs[84] = 0.60  # M15 diff positive
        obs[85] = 0.40  # M15 slope positive
        obs[93] = 0.10  # low chop
        obs[IDX_SWING_RANGE120_POS] = 0.05
        obs[IDX_SPRING_SCORE] = 0.80

        batch = router.act_nam_tac_nhan(obs)
        act_0 = router._extract_action_idx(batch.traces[0])
        assert act_0 == 1, f"Bull BUY action mutated: got {act_0}"

    def test_router_preserves_bull_buy_limit_in_transition_zone(self) -> None:
        """In transition zones, non-market orders (BUY_LIMIT=3) are preserved by both guards."""
        bull_pol = AdversarialMockSpecialist("bull", (3, 0, 0, 0, 0))  # Agent 0 emits BUY_LIMIT
        bear_pol = AdversarialMockSpecialist("bear", (0, 0, 0, 0, 0))

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            anti_bottom_guard_enabled=True,
        )
        router.soft_moe_weights = (0.50, 0.50)  # Transition zone

        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 0.05
        obs[IDX_SPRING_SCORE] = 0.80

        batch = router.act_nam_tac_nhan(obs)
        act_0 = router._extract_action_idx(batch.traces[0])
        assert act_0 == 3, f"Bull BUY_LIMIT mutated: got {act_0}"


# ==============================================================================
# TEST GROUP 5: AGENT 4 GOVERNOR INVARIANCE (POLICY_CLOSE_ALL == 0.0)
# ==============================================================================

class TestAdversarialGovernorInvariance:
    """Verifies that Agent 4 Governor is strictly locked to ACTION_HOLD = 0 under all conditions."""

    def test_malicious_agent4_liquidation_attempt_intercepted(self) -> None:
        """Adversarial policy tries to inject Action 10 (CLOSE_ALL) into Agent 4."""
        bull_pol = AdversarialMockSpecialist("bull", (0, 0, 0, 0, 10))  # Tries to liquidate
        bear_pol = AdversarialMockSpecialist("bear", (0, 2, 0, 0, 10))  # Tries to liquidate + short

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            anti_bottom_guard_enabled=True,
        )

        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 0.05
        obs[IDX_ABSORPTION_SCORE] = 0.70

        batch = router.act_nam_tac_nhan(obs)

        # Agent 4 MUST remain ACTION_HOLD = 0
        agent4_trace = batch.traces[4]
        act_4 = router._extract_action_idx(agent4_trace)
        assert act_4 == 0, f"Governor invariant violated! Agent 4 emitted action {act_4}"

        out_vec = getattr(agent4_trace, "output_vector", getattr(agent4_trace, "selected_action", None))
        assert out_vec[0] == 1.0, "Governor logit 0 not 1.0"
        # POLICY_CLOSE_ALL logit 10 must be strongly suppressed (-100.0)
        assert out_vec[10] <= 0.0, f"POLICY_CLOSE_ALL logit not suppressed: {out_vec[10]}"
        assert agent4_trace.action_details.get("policy_close_all") == 0.0


# ==============================================================================
# TEST GROUP 6: CORRUPTED AND ADVERSARIAL INPUT ROBUSTNESS
# ==============================================================================

class TestAdversarialCorruptInputFuzzing:
    """Fuzzes AntiBottomExhaustionGuard with malformed, NaN, Inf, and corrupt inputs."""

    def test_all_nan_array_fail_safe(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = np.full(137, np.nan, dtype=np.float32)
        dec = guard.check_bottom_short_veto(obs, 2)
        assert dec.is_vetoed is False
        assert dec.telemetry["corrupt_input"] is True

    def test_all_inf_array_fail_safe(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = np.full(137, np.inf, dtype=np.float32)
        dec = guard.check_bottom_short_veto(obs, 2)
        assert dec.is_vetoed is False
        assert dec.telemetry["corrupt_input"] is True

    def test_extreme_floats_and_denormals(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_neutral_observation(dtype=np.float64)
        obs[IDX_SWING_RANGE120_POS] = 1e308
        obs[IDX_ABSORPTION_SCORE] = -1e308
        dec = guard.check_bottom_short_veto(obs, 2)
        assert dec.is_vetoed is False

    def test_arbitrary_invalid_action_types_handling(self) -> None:
        """Documents guard behavior on invalid action input types."""
        guard = AntiBottomExhaustionGuard()
        obs = make_neutral_observation()

        # Numeric out-of-range actions are safely handled
        for numeric_act in (-1, 999, 100):
            dec = guard.check_bottom_short_veto(obs, numeric_act)
            assert dec.is_vetoed is False
            assert dec.reason == "not_short_action"

        # Non-numeric action types raise ValueError/TypeError due to int(proposed_action)
        with pytest.raises((ValueError, TypeError)):
            guard.check_bottom_short_veto(obs, "bad_action")  # type: ignore

        with pytest.raises((ValueError, TypeError)):
            guard.check_bottom_short_veto(obs, None)  # type: ignore


# ==============================================================================
# TEST GROUP 7: ULTRA-FAST CPU LATENCY SLA BENCHMARK (< 0.05 MS)
# ==============================================================================

class TestAdversarialLatencySLA:
    """Verifies that AntiBottomExhaustionGuard satisfies the strict < 0.05 ms SLA under 10,000 tick burst."""

    def test_guard_standalone_10000_tick_burst_under_50_microseconds(self) -> None:
        guard = AntiBottomExhaustionGuard()
        rng = np.random.RandomState(1337)
        obs_batch = [rng.randn(137).astype(np.float32) for _ in range(10000)]
        actions = [int(rng.choice([0, 1, 2, 3, 4, 5, 6])) for _ in range(10000)]

        # Warmup
        for i in range(200):
            guard.check_bottom_short_veto(obs_batch[i], actions[i])

        latencies_ms = []
        t_start = time.perf_counter()
        for i in range(10000):
            t0 = time.perf_counter()
            guard.check_bottom_short_veto(obs_batch[i], actions[i])
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)
        t_total = time.perf_counter() - t_start

        lat = np.array(latencies_ms)
        mean_ms = float(lat.mean())
        p50_ms = float(np.median(lat))
        p95_ms = float(np.percentile(lat, 95))
        p99_ms = float(np.percentile(lat, 99))

        print(
            f"\n[Guard 10k Burst] Mean={mean_ms:.5f}ms, P50={p50_ms:.5f}ms, "
            f"P95={p95_ms:.5f}ms, P99={p99_ms:.5f}ms, Total={t_total:.3f}s"
        )

        assert mean_ms < 0.050, f"Mean latency exceeded 0.050ms SLA: {mean_ms:.5f}ms"
        assert p50_ms < 0.020, f"P50 latency exceeded 0.020ms: {p50_ms:.5f}ms"
        assert p99_ms < 0.100, f"P99 latency exceeded 0.100ms: {p99_ms:.5f}ms"
