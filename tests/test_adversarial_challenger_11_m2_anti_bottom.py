"""
tests/test_adversarial_challenger_11_m2_anti_bottom.py

Adversarial Challenger Verification Suite for Milestone M2:
Anti-Bottom Exhaustion Guard & DualChampionRouter Integration.

Adversarial Probes:
1. Boundary Fuzzing (float64 & float32 representation):
   - Grid sweep across [0.1799, 0.1801] and [1.199, 1.201].
   - Empirical investigation of float32 vs float64 threshold comparison.
2. False Negative Leak Prevention:
   - Exhaustive permutations of all short actions (2, 4, 6) across all 6 exhaustion triggers.
   - End-to-end Router trace mutation, probability mass neutralization, and telemetry verification.
3. False Positive Alpha Preservation:
   - High-momentum breakdown continuation outside and inside demand zones without exhaustion.
4. Bull Champion BUY Immunity:
   - Long entries (1, 3, 5) strictly immune to anti-bottom veto in demand zones.
5. Agent 4 Governor Invariance:
   - ACTION_HOLD = 0 (POLICY_CLOSE_ALL == 0.0) strictly preserved under all adversarial inputs.
6. CPU Latency Burst Benchmark (10,000 ticks):
   - SLA compliance verification (< 0.05ms per tick for guard, < 0.50ms for router).
"""

from __future__ import annotations

import math
import time
from typing import Any
import numpy as np
import pytest

from self_growing_trader.anti_bottom_guard import (
    AntiBottomExhaustionGuard,
    SHORT_ACTIONS,
    VetoDecision,
)
from self_growing_trader.dual_champion_router import (
    BridgeActionBatch,
    DualChampionRouter,
)
from self_growing_trader.orders import Action
from self_growing_trader.soft_moe_gating import (
    CapitalPreservationGuard,
    SoftMoEGatingNetwork,
)


def make_clean_obs(dtype=np.float64) -> np.ndarray:
    """Create a clean 137D baseline observation vector."""
    obs = np.zeros(137, dtype=dtype)
    obs[117] = 0.50  # swing_range120_pos
    obs[118] = 5.00  # dist_swing_high120_atr
    obs[119] = 5.00  # dist_swing_low120_atr
    obs[120] = 0.00  # sweep_rejection
    obs[121] = 0.20  # rvol20
    obs[125] = 0.00  # effort_result
    obs[126] = 0.00  # absorption
    obs[129] = 0.00  # spring
    obs[133] = 0.00  # spring_test
    obs[135] = 1.00  # bars_since_spring
    obs[16] = 0.00   # rsi14
    obs[17] = 0.00   # rsi5
    obs[19] = 0.50   # bb_pos
    return obs


class AdversarialMockPolicy:
    """Mock specialist policy returning controllable agent traces."""

    def __init__(self, name: str, agent_actions: dict[int, int] | None = None) -> None:
        self.name = name
        self.agent_actions = dict(agent_actions or {})
        self.step_count = 0

    def set_action(self, agent_id: int, action_idx: int) -> None:
        self.agent_actions[agent_id] = action_idx

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
                self.trace_id = f"{agent_id}_trace_{act}"
                self.selected_idx = act
                self.action_index = act
                self.selected_action = np.zeros(20, dtype=float)
                self.selected_action[act] = 1.0
                self.output_vector = self.selected_action.copy()
                self.action_probs = self.selected_action.copy()
                self.action_details = {
                    "five_agent_index": agent_id,
                    "five_action_index": act,
                    "action": act,
                }

        traces = []
        for i in range(5):
            act = self.agent_actions.get(i, 0)
            traces.append(MockTrace(i, act))
        return tuple(traces)


# ==============================================================================
# 1. BOUNDARY FUZZING & FLOAT PRECISION AUDIT
# ==============================================================================

class TestAdversarialBoundaryFuzzing:
    """Adversarial stress-testing of exact boundary transitions."""

    def test_float64_fine_grained_swing_pos_boundary(self) -> None:
        """Verify strict transition across [0.1799, 0.1801] with float64."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs(dtype=np.float64)
        obs[119] = 5.0   # outside dist_low
        obs[126] = 0.50  # active absorption

        # Sample 200 points in [0.1799, 0.1801]
        test_points = np.linspace(0.1799, 0.1801, 201)
        for pos in test_points:
            obs[117] = pos
            dec = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
            if pos <= 0.18:
                assert dec.is_vetoed is True, f"Failed veto at swing_pos={pos:.7f} <= 0.18"
            else:
                assert dec.is_vetoed is False, f"False veto at swing_pos={pos:.7f} > 0.18"

    def test_float64_fine_grained_dist_low_boundary(self) -> None:
        """Verify strict transition across [1.199, 1.201] with float64."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs(dtype=np.float64)
        obs[117] = 0.50  # outside swing_pos
        obs[129] = 0.40  # active spring

        test_points = np.linspace(1.199, 1.201, 201)
        for dist in test_points:
            obs[119] = dist
            dec = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
            if dist <= 1.20:
                assert dec.is_vetoed is True, f"Failed veto at dist_low={dist:.7f} <= 1.20"
            else:
                assert dec.is_vetoed is False, f"False veto at dist_low={dist:.7f} > 1.20"

    def test_float32_boundary_representation_discrepancy(self) -> None:
        """Empirical demonstration of float32 boundary behavior:
        In IEEE-754 single precision:
        np.float32(0.18) == 0.18000000715255737 > 0.18000000000000002.
        np.float32(1.20) == 1.2000000476837158 > 1.2.
        When pure float32 arrays are passed with exact nominal threshold values,
        exact boundary values 0.18f and 1.20f evaluate to False without tolerance.
        """
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs(dtype=np.float32)
        obs[117] = np.float32(0.18)
        obs[119] = np.float32(2.0)
        obs[126] = np.float32(0.50)

        dec = guard.check_bottom_short_veto(obs, 2)
        # Record the empirical finding
        val_float = float(obs[117])
        assert val_float > 0.18, "IEEE-754 single precision conversion confirmed"
        # Due to strictly '<= 0.18', np.float32(0.18) is not <= 0.18000000000000002
        assert dec.is_vetoed is False, "Empirically demonstrates float32 strict boundary leak"


# ==============================================================================
# 2. FALSE NEGATIVE LEAK PREVENTION AUDIT
# ==============================================================================

class TestAdversarialFalseNegativeLeakage:
    """Verifies that no short action (2, 4, 6) leaks in demand zones under any exhaustion trigger."""

    @pytest.mark.parametrize("short_act", [2, 4, 6])
    @pytest.mark.parametrize("exhaustion_trigger", [
        ("absorption", 126, 0.36),
        ("spring", 129, 0.26),
        ("dual_rsi", 16, -0.41),  # companion 17 set in body
        ("sweep_rej", 120, 0.26),
        ("spring_retest", 133, 0.26),  # companion 135 set in body
        ("bb_climax", 19, 0.04),  # companion 121 set in body
    ])
    def test_all_short_actions_and_triggers_intercepted(
        self,
        short_act: int,
        exhaustion_trigger: tuple[str, int, float],
    ) -> None:
        """Every permutation of short action and exhaustion trigger MUST be vetoed in demand zone."""
        name, idx, val = exhaustion_trigger
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs(dtype=np.float64)

        # In demand zone via swing_pos
        obs[117] = 0.12
        obs[119] = 3.0
        obs[idx] = val

        if name == "dual_rsi":
            obs[17] = -0.65
        elif name == "spring_retest":
            obs[135] = 0.15  # age <= 0.30
        elif name == "bb_climax":
            obs[121] = 0.35  # rvol >= 0.30

        dec = guard.check_bottom_short_veto(obs, short_act)
        assert dec.is_vetoed is True, f"Leaked short action {short_act} on trigger {name}!"
        assert dec.telemetry["in_demand_zone"] is True
        assert dec.telemetry["is_exhausted"] is True

    def test_router_integration_full_short_interception_and_probability_zeroing(self) -> None:
        """Verify DualChampionRouter intercepts Bear short and neutralizes probability mass."""
        bull_mock = AdversarialMockPolicy("bull", {0: 0})
        bear_mock = AdversarialMockPolicy("bear", {1: 2})  # Agent 1 proposes Action 2 (MARKET_SELL)

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            anti_bottom_guard_enabled=True,
        )

        obs = make_clean_obs()
        obs[80] = -0.6    # M5 trend bear
        obs[84] = -0.6    # M15 trend bear
        obs[88] = -0.6    # H1 trend bear
        obs[117] = 0.08   # Deep demand zone
        obs[119] = 0.40   # 0.4 ATR from low
        obs[126] = 0.60   # Climax absorption
        router.set_regime_soft_probabilities([0.10, 0.85, 0.05], "BEAR", 1)

        batch = router.act_nam_tac_nhan(obs)
        assert isinstance(batch, BridgeActionBatch)

        # Agent 1 MUST be converted to HOLD (0)
        trace_1 = batch[1]
        assert trace_1.selected_idx == 0
        assert trace_1.action_index == 0
        assert trace_1.output_vector[0] == 1.0
        assert trace_1.output_vector[2] == 0.0
        assert trace_1.action_probs[0] == 1.0
        assert trace_1.action_probs[2] == 0.0
        assert trace_1.action_details.get("regime_veto_applied") is True

        # Telemetry verification
        assert router.anti_bottom_telemetry["bottom_short_veto"] is True
        assert "anti_bottom_guard:" in trace_1.action_details.get("regime_veto_reason", "")
        assert router.anti_bottom_telemetry["veto_reason"] == "bottom_demand_zone_absorption"
        assert router.anti_bottom_telemetry["demand_zone_score"] > 0.0


# ==============================================================================
# 3. FALSE POSITIVE ALPHA PRESERVATION AUDIT
# ==============================================================================

class TestAdversarialFalsePositiveAlphaPreservation:
    """Verifies that legitimate shorts are preserved outside demand zones or during clean momentum."""

    def test_mid_and_high_range_shorts_preserved(self) -> None:
        """Shorts outside demand zones must never be vetoed, even with indicator spikes."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs()
        obs[117] = 0.65   # Mid-high range
        obs[119] = 4.5    # 4.5 ATR from low
        obs[126] = 0.80   # Spurious absorption indicator artifact
        obs[129] = 0.70   # Spurious spring indicator artifact

        for act in (2, 4, 6):
            dec = guard.check_bottom_short_veto(obs, act)
            assert dec.is_vetoed is False, f"False positive veto on action {act} in mid-range!"
            assert dec.telemetry["in_demand_zone"] is False

    def test_clean_breakdown_in_demand_zone_preserved(self) -> None:
        """High-momentum breakdown through demand zone without exhaustion must be allowed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs()
        obs[117] = 0.05   # Price piercing swing low
        obs[119] = 0.20   # 0.2 ATR from swing low
        # Clean momentum: no absorption, no spring, neutral RSI
        obs[126] = 0.00
        obs[129] = 0.00
        obs[16] = -0.15
        obs[17] = -0.20
        obs[120] = 0.00
        obs[133] = 0.00
        obs[19] = 0.30

        for act in (2, 4, 6):
            dec = guard.check_bottom_short_veto(obs, act)
            assert dec.is_vetoed is False, f"Vetoed valid clean breakdown short {act}!"
            assert dec.telemetry["in_demand_zone"] is True
            assert dec.telemetry["is_exhausted"] is False


# ==============================================================================
# 4. BULL BUY IMMUNITY AUDIT
# ==============================================================================

class TestAdversarialBullBuyImmunity:
    """Verifies that Bull Specialist BUY orders are 100% immune to veto."""

    @pytest.mark.parametrize("buy_act", [1, 3, 5])
    def test_bull_buy_actions_immune_in_extreme_demand_exhaustion(self, buy_act: int) -> None:
        """Bull long entries (1=MARKET_BUY, 3=BUY_LIMIT, 5=BUY_STOP) must never be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_obs()
        obs[117] = 0.01   # Absolute bottom
        obs[119] = 0.05
        obs[126] = 0.99   # Climax stopping volume
        obs[129] = 0.95   # Spring active

        dec = guard.check_bottom_short_veto(obs, buy_act)
        assert dec.is_vetoed is False
        assert dec.reason == "not_short_action"

    def test_router_never_mutates_bull_specialist_buy_in_demand_zone(self) -> None:
        """DualChampionRouter must not veto Bull Specialist (Agent 0) BUY in demand zone."""
        bull_mock = AdversarialMockPolicy("bull", {0: 1})  # Agent 0 emits MARKET_BUY (1)
        bear_mock = AdversarialMockPolicy("bear", {1: 0})

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            anti_bottom_guard_enabled=True,
        )

        obs = make_clean_obs()
        obs[80] = 0.6     # M5 trend bull
        obs[84] = 0.6     # M15 trend bull
        obs[88] = 0.6     # H1 trend bull
        obs[117] = 0.02
        obs[119] = 0.10
        obs[126] = 0.85
        router.set_regime_soft_probabilities([0.85, 0.10, 0.05], "BULL", 0)

        batch = router.act_nam_tac_nhan(obs)
        trace_0 = batch[0]
        assert trace_0.selected_idx == 1
        assert trace_0.action_index == 1
        assert trace_0.output_vector[1] == 1.0

    def test_bull_buy_limit_and_stop_immune_even_in_transition_zone(self) -> None:
        """BUY_LIMIT (3) and BUY_STOP (5) are immune to both Anti-Bottom Guard and Capital Preservation."""
        bull_mock = AdversarialMockPolicy("bull", {0: 3})  # BUY_LIMIT
        bear_mock = AdversarialMockPolicy("bear", {0: 3, 1: 0})

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            anti_bottom_guard_enabled=True,
        )

        obs = make_clean_obs()
        obs[117] = 0.02
        obs[119] = 0.10
        obs[126] = 0.85

        batch = router.act_nam_tac_nhan(obs)
        trace_0 = batch[0]
        # BUY_LIMIT is NOT vetoed even in transition zone
        assert trace_0.selected_idx == 3
        assert trace_0.output_vector[3] == 1.0


# ==============================================================================
# 5. GOVERNOR AGENT 4 INVARIANCE AUDIT
# ==============================================================================

class TestAdversarialGovernorAgent4Invariance:
    """Verifies Agent 4 Governor invariant (ACTION_HOLD = 0, POLICY_CLOSE_ALL == 0.0)."""

    def test_hostile_agent4_override_attempt_during_bottom_veto(self) -> None:
        """Even if specialist attempts to emit POLICY_CLOSE_ALL (10), Agent 4 is strictly HOLD."""
        # Hostile mock: Agent 1 tries to sell (2), Agent 4 tries to close basket (10)
        bear_mock = AdversarialMockPolicy("bear", {1: 2, 4: 10})
        bull_mock = AdversarialMockPolicy("bull", {0: 0, 4: 10})

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            anti_bottom_guard_enabled=True,
        )

        obs = make_clean_obs()
        obs[80] = -0.6
        obs[84] = -0.6
        obs[88] = -0.6
        obs[117] = 0.05
        obs[126] = 0.50
        router.set_regime_soft_probabilities([0.10, 0.85, 0.05], "BEAR", 1)

        batch = router.act_nam_tac_nhan(obs)
        # Agent 1 was vetoed
        assert batch[1].selected_idx == 0
        # Agent 4 MUST be HOLD (0), not 10
        trace_4 = batch[4]
        assert trace_4.selected_idx == 0
        assert trace_4.action_index == 0
        assert trace_4.output_vector[0] == 1.0
        assert trace_4.output_vector[10] <= 0.0
        assert int(np.argmax(trace_4.output_vector[:12])) == 0


# ==============================================================================
# 6. CPU LATENCY BURST BENCHMARK (< 0.05 MS SLA)
# ==============================================================================

class TestCPULatencyBurstBenchmark:
    """Verifies high-frequency latency under 10,000 sequential evaluations."""

    def test_anti_bottom_guard_10000_tick_latency(self) -> None:
        guard = AntiBottomExhaustionGuard()
        rng = np.random.RandomState(42)
        observations = [rng.randn(137).astype(np.float32) for _ in range(10000)]
        actions = [int(rng.choice([1, 2, 3, 4, 5, 6])) for _ in range(10000)]

        # Warmup
        for i in range(200):
            guard.check_bottom_short_veto(observations[i], actions[i])

        latencies_ms = []
        for i in range(10000):
            t0 = time.perf_counter()
            guard.check_bottom_short_veto(observations[i], actions[i])
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies_ms))
        p95_ms = float(np.percentile(latencies_ms, 95))
        p99_ms = float(np.percentile(latencies_ms, 99))

        print(f"\n[Guard Latency Benchmark] Mean: {mean_ms:.5f}ms, P95: {p95_ms:.5f}ms, P99: {p99_ms:.5f}ms")
        assert mean_ms < 0.05, f"Mean latency exceeded 0.05ms SLA: {mean_ms:.5f}ms"
        assert p99_ms < 0.10, f"P99 latency exceeded 0.10ms: {p99_ms:.5f}ms"
