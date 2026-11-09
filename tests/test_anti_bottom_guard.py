"""
tests/test_anti_bottom_guard.py

Comprehensive Test Suite for Milestone M2:
Anti-Bottom Exhaustion Guard & Demand Zone Liquidity Shield

Test Coverage:
1. Historical Failure Mode Recreation:
   - Window 1 ($96k bottom exhaustion during bull run, -$3,108.95 loss recreation)
   - Window 3 ($84k-$87k Wyckoff accumulation spring, -$2,233.62 loss recreation)
   - 100% Interception rate on 1,000 Monte Carlo bottom selloff permutations (3,000/3,000 short actions intercepted)
2. Specificity & False Positive Prevention:
   - Mid-range and top-of-range short entries permitted (zero false vetoes)
   - Clean breakdown through demand zone without absorption/spring permitted
3. Boundary & Threshold Precision:
   - Exact boundary transitions for swing_range120_position (0.18), dist_swing_low120_atr (1.20),
     absorption_score (0.35), spring_score (0.25), sweep_rejection (0.25), spring_test (0.25),
     bars_since_spring (0.30), bb_position (0.05), relative_volume (0.30), and dual RSI (-0.40, -0.60)
4. Numerical Edge Cases & Corrupt Inputs:
   - All zeros, NaNs, Infs, -Infs, empty, and truncated arrays
5. Exhaustion Triggers Unit Coverage:
   - Wyckoff absorption, Wyckoff spring, Dual-RSI, Liquidity sweep rejection, Spring retest, Climax band pierce
6. Action Permutations & Invariant Guarantees:
   - Bull Champion BUY actions (1, 3, 5) never vetoed in demand zone
   - Non-entry actions never vetoed
   - Agent 4 Governor invariant (ACTION_HOLD = 0) strictly preserved under all conditions
   - DualChampionRouter integration trace mutation, probability zeroing, and telemetry verified
7. Router Integration with Pre-Trained / Mock Regime Detector:
   - Both regime_detector=None and regime_detector present
8. Ultra-Fast Latency Benchmark:
   - 10,000 sequential evaluations benchmark (< 0.05ms per tick SLA)
"""

from __future__ import annotations

import math
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from self_growing_trader.anti_bottom_guard import AntiBottomExhaustionGuard, VetoDecision
from self_growing_trader.dual_champion_router import DualChampionRouter, BridgeActionBatch
from self_growing_trader.orders import Action


# ==============================================================================
# TEST FIXTURES & HELPERS
# ==============================================================================

def make_clean_observation() -> np.ndarray:
    """Return a neutral 137D float observation vector."""
    obs = np.zeros(137, dtype=float)
    # Neutral mid-range baseline
    obs[117] = 0.50  # swing_range120_position
    obs[118] = 5.00  # dist_swing_high120_atr
    obs[119] = 5.00  # dist_swing_low120_atr
    obs[120] = 0.00  # liquidity_sweep_rejection
    obs[121] = 0.20  # relative_volume_20
    obs[125] = 0.00  # effort_result_ratio
    obs[126] = 0.00  # absorption_score
    obs[129] = 0.00  # spring_score
    obs[133] = 0.00  # spring_test_quality
    obs[135] = 1.00  # bars_since_spring (1.0 = no recent spring)
    obs[16] = 0.00   # rsi14 (normalized: 0.0 = 50 RSI)
    obs[17] = 0.00   # rsi5  (normalized: 0.0 = 50 RSI)
    obs[19] = 0.50   # bb_position (mid-band)
    return obs


class MockSpecialistPolicy:
    """Mock specialist policy tracking 5-agent trace outputs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.step_count = 0

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[object, ...]:
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

        # 0: BUY (Agent 0), 1: SELL (Agent 1), 2: MANAGE, 3: SCALP, 4: GOVERNOR
        return (
            MockTrace(0, int(Action.MARKET_BUY)),   # 1
            MockTrace(1, int(Action.MARKET_SELL)),  # 2
            MockTrace(2, int(Action.HOLD)),         # 0
            MockTrace(3, int(Action.HOLD)),         # 0
            MockTrace(4, int(Action.CLOSE_ALL)),    # Tries to liquidate (10)
        )

    def ensure_compatible(self) -> None:
        pass

    def reset_runtime_state(self) -> None:
        self.step_count = 0


# ==============================================================================
# TEST GROUP 1: HISTORICAL FAILURE MODE RECREATION (OOS WINDOWS 1 & 3)
# ==============================================================================

class TestHistoricalFailureModeRecreation:
    """Recreates the exact failure states of OOS Windows 1 & 3 and proves 100% interception."""

    def test_window_1_bull_run_bottom_selloff_exhaustion_vetoed(self) -> None:
        """Window 1: Bear specialist shorting at $96,000 into Wyckoff Spring must be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()

        # Window 1 signature: extreme 120-bar bottom, high absorption, Wyckoff spring, deeply oversold
        obs[117] = 0.06   # swing_range120_position (bottom 6% of 2-hour range)
        obs[119] = 0.45   # dist_swing_low120_atr (0.45 ATR from low)
        obs[126] = 0.58   # absorption_score (> 0.35)
        obs[129] = 0.65   # spring_score (> 0.25)
        obs[16] = -0.55   # rsi14 (RSI 22.5 <= -0.40)
        obs[17] = -0.72   # rsi5  (RSI 14.0 <= -0.60)

        for sell_act in (Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP):
            is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(sell_act))
            assert is_vetoed is True, f"Failed to veto {sell_act.name} in Window 1 scenario"
            assert "demand_zone" in reason or "spring" in reason or "absorption" in reason
            assert telemetry.get("in_demand_zone") is True
            assert telemetry.get("is_exhausted") is True
            assert telemetry.get("demand_zone_score", 0.0) > 0.0

    def test_window_3_accumulation_spring_liquidity_sweep_vetoed(self) -> None:
        """Window 3: Bear specialist shorting range low liquidity sweep must be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()

        # Window 3 signature: consolidation low sweep with high stopping volume
        obs[117] = 0.12   # swing_range120_position (<= 0.18)
        obs[119] = 0.85   # dist_swing_low120_atr (<= 1.20)
        obs[120] = 0.75   # liquidity_sweep_rejection (strong low sweep)
        obs[126] = 0.45   # absorption_score (> 0.35)
        obs[129] = 0.40   # spring_score (> 0.25)

        for sell_act in (Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP):
            is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(sell_act))
            assert is_vetoed is True, f"Failed to veto {sell_act.name} in Window 3 scenario"
            assert telemetry.get("in_demand_zone") is True
            assert telemetry.get("is_exhausted") is True

    def test_exhaustive_monte_carlo_historical_failure_scenarios_100pct_intercepted(self) -> None:
        """Stochastic test across 1,000 bottom exhaustion states proving 100.0% interception."""
        guard = AntiBottomExhaustionGuard()
        rng = np.random.RandomState(42)
        total_evaluations = 0
        total_intercepted = 0

        for _ in range(1000):
            obs = make_clean_observation()
            # Randomize demand zone proximity (Condition A True)
            if rng.rand() < 0.5:
                obs[117] = float(rng.uniform(0.00, 0.18))
                obs[119] = float(rng.uniform(0.00, 5.00))
            else:
                obs[117] = float(rng.uniform(0.00, 1.00))
                obs[119] = float(rng.uniform(0.00, 1.20))

            # Randomize exhaustion trigger (Condition B True)
            trigger_choice = rng.randint(0, 6)
            if trigger_choice == 0:
                obs[126] = float(rng.uniform(0.36, 1.00))  # absorption trigger
            elif trigger_choice == 1:
                obs[129] = float(rng.uniform(0.26, 1.00))  # spring trigger
            elif trigger_choice == 2:
                obs[16] = float(rng.uniform(-1.00, -0.40))  # dual rsi trigger
                obs[17] = float(rng.uniform(-1.00, -0.60))
            elif trigger_choice == 3:
                obs[120] = float(rng.uniform(0.26, 1.00))  # sweep rejection trigger
            elif trigger_choice == 4:
                obs[133] = float(rng.uniform(0.26, 1.00))  # spring test trigger
                obs[135] = float(rng.uniform(0.00, 0.30))  # recent age
            else:
                obs[19] = float(rng.uniform(0.00, 0.05))   # bb extreme
                obs[121] = float(rng.uniform(0.30, 1.00))  # volume expansion

            # Evaluate all 3 sell actions
            for act in (int(Action.MARKET_SELL), int(Action.SELL_LIMIT), int(Action.SELL_STOP)):
                total_evaluations += 1
                is_vetoed, _, _ = guard.check_bottom_short_veto(obs, act)
                if is_vetoed:
                    total_intercepted += 1

        assert total_evaluations == 3000
        assert total_intercepted == 3000, f"Leaked {total_evaluations - total_intercepted} bottom shorts!"


# ==============================================================================
# TEST GROUP 2: SPECIFICITY & FALSE POSITIVE PREVENTION
# ==============================================================================

class TestSpecificityAndFalsePositivePrevention:
    """Verifies that legitimate Bear short trades are NEVER vetoed outside demand exhaustion."""

    def test_mid_range_trend_continuation_sell_not_vetoed(self) -> None:
        """Healthy mid-range trend shorting must not be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()

        obs[117] = 0.52   # mid-range
        obs[119] = 4.80   # far from swing low
        obs[16] = -0.10   # neutral momentum
        obs[126] = 0.00   # no absorption
        obs[129] = 0.00   # no spring

        for sell_act in (Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP):
            is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(sell_act))
            assert is_vetoed is False, f"False positive veto on {sell_act.name} in mid-range"
            assert telemetry.get("in_demand_zone") is False

    def test_top_of_range_resistance_breakdown_sell_not_vetoed(self) -> None:
        """Top-of-range short at resistance must not be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()

        obs[117] = 0.88   # top of range
        obs[119] = 12.5   # 12.5 ATR above low
        obs[16] = 0.45    # overbought turning down
        obs[126] = 0.00
        obs[129] = 0.00

        is_vetoed, _, telemetry = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False
        assert telemetry.get("in_demand_zone") is False

    def test_demand_zone_continuation_without_absorption_not_vetoed(self) -> None:
        """Clean breakdown through demand zone without absorption must not be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()

        obs[117] = 0.14   # near low
        obs[119] = 0.75   # near low
        # BUT no absorption, no spring, RSI not oversold
        obs[126] = 0.08   # <= 0.35
        obs[129] = 0.00   # <= 0.25
        obs[16] = -0.20   # not oversold
        obs[17] = -0.30   # not oversold
        obs[19] = 0.20    # not band climax
        obs[120] = 0.00   # no sweep
        obs[133] = 0.00   # no spring test

        is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False, "Vetoed clean trend continuation breakdown!"
        assert telemetry.get("in_demand_zone") is True
        assert telemetry.get("is_exhausted") is False


# ==============================================================================
# TEST GROUP 3: BOUNDARY & THRESHOLD PRECISION
# ==============================================================================

class TestBoundaryThresholdPrecision:
    """Verifies razor-sharp mathematical boundaries at exact threshold transitions."""

    def test_exact_threshold_swing_range120_position(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[119] = 5.0    # Far from low in ATR
        obs[126] = 0.50   # Exhaustion present (> 0.35)

        # At exactly 0.1800000 -> In Demand Zone (Veto True)
        obs[117] = 0.1800000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

        # At 0.1800001 -> Outside Demand Zone (Veto False)
        obs[117] = 0.1800001
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

    def test_exact_threshold_dist_swing_low120_atr(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.50   # Outside range percentage
        obs[129] = 0.40   # Spring present (> 0.25)

        # At exactly 1.2000000 -> In Demand Zone (Veto True)
        obs[119] = 1.2000000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

        # At 1.2000001 -> Outside Demand Zone (Veto False)
        obs[119] = 1.2000001
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

    def test_exact_threshold_absorption_score(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.10   # In Demand Zone
        obs[129] = 0.00
        obs[16] = 0.00
        obs[17] = 0.00

        # At exactly 0.3500000 -> Strict inequality requires > 0.35 (Veto False)
        obs[126] = 0.3500000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

        # At 0.3500001 -> Triggered (Veto True)
        obs[126] = 0.3500001
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

    def test_exact_threshold_spring_score(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.10   # In Demand Zone
        obs[126] = 0.00
        obs[16] = 0.00
        obs[17] = 0.00

        # At exactly 0.2500000 -> Strict inequality requires > 0.25 (Veto False)
        obs[129] = 0.2500000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

        # At 0.2500001 -> Triggered (Veto True)
        obs[129] = 0.2500001
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

    def test_exact_threshold_dual_rsi_oversold(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.10   # In Demand Zone
        obs[126] = 0.00
        obs[129] = 0.00

        # Both meet threshold: rsi14 <= -0.40 and rsi5 <= -0.60
        obs[16] = -0.4000000
        obs[17] = -0.6000000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

        # rsi14 just misses: -0.3999999
        obs[16] = -0.3999999
        obs[17] = -0.6000000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

        # rsi5 just misses: -0.5999999
        obs[16] = -0.4000000
        obs[17] = -0.5999999
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

    def test_exact_threshold_liquidity_sweep_rejection(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.10   # In Demand Zone

        # At exactly 0.2500000 -> False (> 0.25 required)
        obs[120] = 0.2500000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

        # At 0.2500001 -> True
        obs[120] = 0.2500001
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

    def test_exact_threshold_spring_retest_and_recency(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.10   # In Demand Zone

        # spring_test > 0.25 and bars_since_spring <= 0.30
        obs[133] = 0.2500001
        obs[135] = 0.3000000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

        # bars_since_spring too old: 0.3000001
        obs[135] = 0.3000001
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False

    def test_exact_threshold_climax_band_pierce(self) -> None:
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.10   # In Demand Zone

        # bb_pos <= 0.05 and rvol >= 0.30
        obs[19] = 0.0500000
        obs[121] = 0.3000000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is True

        # bb_pos just above 0.05
        obs[19] = 0.0500001
        obs[121] = 0.3000000
        is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False


# ==============================================================================
# TEST GROUP 4: NUMERICAL EDGE CASES & CORRUPT INPUTS
# ==============================================================================

class TestNumericalEdgeCasesAndCorruptInputs:
    """Verifies fail-safe handling on uninitialized, NaN, Inf, and truncated arrays."""

    def test_all_zeros_observation_vector(self) -> None:
        """On an all-zeros startup array, guard must evaluate safely to False (no false veto)."""
        guard = AntiBottomExhaustionGuard()
        obs = np.zeros(137, dtype=np.float32)

        is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False
        assert telemetry.get("is_exhausted") is False

    def test_nan_observation_vector_fail_safe(self) -> None:
        """NaN observations must be sanitized without raising exceptions, failing safe."""
        guard = AntiBottomExhaustionGuard()
        obs = np.full(137, np.nan, dtype=np.float32)

        is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False
        assert telemetry.get("corrupt_input") is True
        assert telemetry.get("in_demand_zone") is False

    def test_inf_and_negative_inf_observation_vector(self) -> None:
        """Infinite inputs must not crash or cause undefined veto behavior."""
        guard = AntiBottomExhaustionGuard()
        obs_pos = np.full(137, np.inf, dtype=np.float32)
        obs_neg = np.full(137, -np.inf, dtype=np.float32)

        is_vetoed_pos, _, tel_pos = guard.check_bottom_short_veto(obs_pos, int(Action.MARKET_SELL))
        is_vetoed_neg, _, tel_neg = guard.check_bottom_short_veto(obs_neg, int(Action.MARKET_SELL))
        assert is_vetoed_pos is False
        assert is_vetoed_neg is False
        assert tel_pos.get("corrupt_input") is True
        assert tel_neg.get("corrupt_input") is True

    def test_truncated_observation_vector_safety(self) -> None:
        """Truncated observation vectors (< 137 elements) must handle missing indices safely."""
        guard = AntiBottomExhaustionGuard()
        short_obs = np.zeros(20, dtype=np.float32)

        is_vetoed, _, telemetry = guard.check_bottom_short_veto(short_obs, int(Action.MARKET_SELL))
        assert is_vetoed is False
        assert telemetry.get("in_demand_zone") is False

    def test_empty_observation_vector_safety(self) -> None:
        """Empty array must not crash."""
        guard = AntiBottomExhaustionGuard()
        empty_obs = np.empty(0, dtype=np.float32)

        is_vetoed, reason, telemetry = guard.check_bottom_short_veto(empty_obs, int(Action.MARKET_SELL))
        assert is_vetoed is False
        assert telemetry.get("empty_obs") is True

    def test_disabled_guard_passes_all(self) -> None:
        """When enabled=False, guard never vetoes."""
        guard = AntiBottomExhaustionGuard(enabled=False)
        obs = make_clean_observation()
        obs[117] = 0.05
        obs[126] = 0.80

        is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(Action.MARKET_SELL))
        assert is_vetoed is False
        assert reason == "guard_disabled"


# ==============================================================================
# TEST GROUP 5: ACTION PERMUTATIONS & INVARIANT GUARANTEES
# ==============================================================================

class TestActionPermutationsAndRouterInvariants:
    """Verifies that Bull BUYs are preserved and Agent 4 Governor invariant is 100% locked."""

    def test_bull_champion_buy_actions_never_vetoed_in_demand_zone(self) -> None:
        """Bull Champion BUY actions (1, 3, 5) must NEVER be vetoed in demand zones."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        # Extreme demand zone with active spring (ideal accumulation buy!)
        obs[117] = 0.05
        obs[119] = 0.30
        obs[126] = 0.60
        obs[129] = 0.75

        for buy_act in (Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP):
            is_vetoed, reason, telemetry = guard.check_bottom_short_veto(obs, int(buy_act))
            assert is_vetoed is False, f"Bull {buy_act.name} was erroneously vetoed in demand zone!"
            assert reason == "not_short_action"

    def test_non_entry_actions_never_vetoed(self) -> None:
        """Non-entry actions (HOLD, CANCEL, CLOSE_ALL, MANAGE) must never be vetoed."""
        guard = AntiBottomExhaustionGuard()
        obs = make_clean_observation()
        obs[117] = 0.05
        obs[129] = 0.75

        non_entries = (
            Action.HOLD,
            Action.CANCEL_ORDERS,
            Action.CLOSE_WORST,
            Action.CLOSE_BEST,
            Action.CLOSE_ALL,
            Action.MANAGE_POSITION,
        )
        for act in non_entries:
            is_vetoed, _, _ = guard.check_bottom_short_veto(obs, int(act))
            assert is_vetoed is False

    def test_agent4_governor_strict_hold_invariance_under_veto(self) -> None:
        """Agent 4 Governor must remain strictly ACTION_HOLD = 0 when Agent 1 is vetoed."""
        bull_mock = MockSpecialistPolicy("bull")
        bear_mock = MockSpecialistPolicy("bear")
        guard = AntiBottomExhaustionGuard()

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            regime_detector=None,
            anti_bottom_guard=guard,
        )

        obs = make_clean_observation()
        obs[80] = -0.6    # M5 trend bear
        obs[84] = -0.6    # M15 trend bear
        obs[88] = -0.6    # H1 trend bear
        obs[117] = 0.08   # Demand zone
        obs[129] = 0.60   # Spring exhaustion

        # Force Bear routing so Agent 1 is active
        router.set_regime_soft_probabilities([0.10, 0.80, 0.10], "BEAR", 1)

        batch = router.act_nam_tac_nhan(obs)
        assert len(batch) == 5

        # 1. Agent 1 (Short specialist) was vetoed to HOLD (0)
        trace_1 = batch[1]
        act_1 = int(trace_1.action_details.get("five_action_index", -1))
        assert act_1 == int(Action.HOLD)
        assert trace_1.action_details.get("regime_veto_applied") is True
        assert "anti_bottom_guard" in trace_1.action_details.get("regime_veto_reason", "")

        # 2. Agent 4 (Governor) MUST strictly remain ACTION_HOLD (0)
        trace_4 = batch[4]
        act_4 = int(trace_4.action_details.get("five_action_index", -1))
        wire_4 = int(trace_4.action_details.get("wire_action", -1))
        assert act_4 == int(Action.HOLD)
        assert wire_4 == int(Action.HOLD)
        assert trace_4.action_details.get("policy_close_all", 0.0) == 0.0

        # 3. Telemetry captured on router and batch
        assert router.anti_bottom_telemetry["bottom_short_veto"] is True
        assert batch.anti_bottom_telemetry["bottom_short_veto"] is True

    def test_router_integration_with_pre_trained_regime_detector(self) -> None:
        """Verify DualChampionRouter integration with mock pre-trained detector."""
        mock_detector = MagicMock()
        mock_detector.predict_proba.return_value = np.array([[0.10, 0.85, 0.05]], dtype=float)
        mock_detector.predict.return_value = np.array([1])  # BEAR

        bull_mock = MockSpecialistPolicy("bull")
        bear_mock = MockSpecialistPolicy("bear")
        guard = AntiBottomExhaustionGuard()

        router = DualChampionRouter(
            bull_policy=bull_mock,
            bear_policy=bear_mock,
            regime_detector=mock_detector,
            anti_bottom_guard=guard,
        )

        obs = make_clean_observation()
        obs[80] = -0.6    # M5 trend bear
        obs[84] = -0.6    # M15 trend bear
        obs[88] = -0.6    # H1 trend bear
        obs[117] = 0.05   # Demand zone
        obs[126] = 0.65   # Wyckoff absorption

        batch = router.act_nam_tac_nhan(obs)
        assert len(batch) == 5

        # Agent 1 was vetoed
        trace_1 = batch[1]
        assert trace_1.action_details.get("regime_veto_applied") is True
        assert trace_1.selected_idx == 0
        assert trace_1.action_index == 0
        assert trace_1.output_vector[0] == 1.0
        assert trace_1.output_vector[2] == 0.0

        # Agent 0 BUY was NOT vetoed
        trace_0 = batch[0]
        assert trace_0.selected_idx == int(Action.MARKET_BUY)


# ==============================================================================
# TEST GROUP 6: ULTRA-FAST CPU LATENCY BENCHMARK (< 0.05 MS PER TICK)
# ==============================================================================

class TestCPULatencySLA:
    """Verifies sub-0.05ms (50 microseconds) CPU evaluation speed across 10,000 ticks."""

    def test_10000_sequential_evaluations_latency_sla(self) -> None:
        guard = AntiBottomExhaustionGuard()
        rng = np.random.RandomState(999)

        # Pre-generate 10,000 synthetic observations
        observations = [rng.randn(137).astype(np.float32) for _ in range(10000)]
        actions = [int(rng.choice([1, 2, 3, 4])) for _ in range(10000)]

        # Warm up JIT and CPU instruction cache
        for i in range(100):
            guard.check_bottom_short_veto(observations[i], actions[i])

        # Benchmark 10,000 ticks
        latencies_ms = []
        t_start = time.perf_counter()

        for i in range(10000):
            t0 = time.perf_counter()
            guard.check_bottom_short_veto(observations[i], actions[i])
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

        t_total = time.perf_counter() - t_start

        mean_lat = float(np.mean(latencies_ms))
        p99_lat = float(np.percentile(latencies_ms, 99))

        # Latency assertions: SLA < 0.05ms (50µs)
        assert mean_lat < 0.050, f"Mean latency exceeded SLA: {mean_lat:.5f} ms >= 0.050 ms"
        assert p99_lat < 0.100, f"P99 latency exceeded SLA: {p99_lat:.5f} ms >= 0.100 ms"
        assert t_total < 0.500, f"Total 10,000 evaluations took {t_total:.3f}s (budget: 0.5s)"
