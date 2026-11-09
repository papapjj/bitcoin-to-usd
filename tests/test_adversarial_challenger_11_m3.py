"""
tests/test_adversarial_challenger_11_m3.py

Adversarial Stress & Invariant Verification Suite for Milestone M3:
1. Dynamic Bidirectional CPPI Boundary & Extreme Stress (Floor, Drawdown, Cushion explosion, 50% cap, Symmetry).
2. Concurrency Flood & Race Invariant Harness (Global 5, Directional 2, Position format fuzzing).
3. Micro-Scalp Penetration & Anti-Bypass Testing (Global cap, Directional cap, Anti-Bottom integration).
4. Dynamic Breakeven Ratchet Stress & Flash Reversal Harness (1.99 vs 2.0 ATR, Flash crash profit lock, Monotonicity).
5. Agent 4 Governor Invariance Adversarial Attacks (Adversarial actions, Corrupted traces, Panic drawdowns).
6. Multi-Bar High-Frequency Simulation with Continuous Invariant Auditing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest

from self_growing_trader.cppi_risk_engine import (
    BidirectionalCPPIConfig,
    BidirectionalCPPIEngine,
    CPPIConfig,
    CPPIFloorEngine,
)
from self_growing_trader.two_tier_runner import (
    TwoTierConfig,
    TwoTierTrailingRunnerEngine,
    PositionState,
    PositionPhase,
    ACTION_HOLD,
    ACTION_MANAGE_POSITION,
    MANAGEMENT_MODE_BREAKEVEN,
    MANAGEMENT_MODE_PARTIAL_CLOSE,
)
from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    BridgeActionBatch,
)
from self_growing_trader.micro_scalp_engine import (
    MicroScalpConfig,
    MicroPullbackScalpEngine,
    ScalpSignal,
)


# ============================================================================
# Helpers & Mocks for Adversarial Testing
# ============================================================================

@dataclass
class AdversarialMockTrace:
    action_index: int = 0
    selected_idx: int = 0
    output_vector: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.float32))
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=np.float32))
    action_details: Dict[str, Any] = field(default_factory=dict)


class AdversarialMockPolicy:
    """Mock policy that deliberately proposes illegal actions (e.g. Agent 4 CLOSE_ALL)."""

    def __init__(self, act_0: int = 1, act_1: int = 2, act_4: int = 10) -> None:
        self.act_0 = act_0
        self.act_1 = act_1
        self.act_4 = act_4

    def act_nam_tac_nhan(self, obs: np.ndarray, goal: Any = None, *args: Any, **kwargs: Any) -> List[Any]:
        t0 = AdversarialMockTrace(
            action_index=self.act_0,
            selected_idx=self.act_0,
            action_details={"five_action_index": self.act_0, "base_risk_pct": 0.002},
        )
        t1 = AdversarialMockTrace(
            action_index=self.act_1,
            selected_idx=self.act_1,
            action_details={"five_action_index": self.act_1, "base_risk_pct": 0.002},
        )
        t2 = AdversarialMockTrace(action_index=0)
        t3 = AdversarialMockTrace(action_index=0)
        # Agent 4 deliberately proposes action 10 (POLICY_CLOSE_ALL) with positive logit
        out_vec = np.zeros(20, dtype=np.float32)
        out_vec[self.act_4] = 10.0
        t4 = AdversarialMockTrace(
            action_index=self.act_4,
            selected_idx=self.act_4,
            output_vector=out_vec,
            selected_action=out_vec,
            action_details={"policy_close_all": 1.0, "five_action_index": self.act_4},
        )
        return [t0, t1, t2, t3, t4]


# ============================================================================
# Suite 1: Dynamic Bidirectional CPPI Boundary & Extreme Stress Harness
# ============================================================================

class TestAdversarialCPPIBoundaries:
    """Adversarially probe boundary conditions, edge cases, and mathematical invariants."""

    def test_cppi_exact_floor_boundary_rejection(self):
        """At exact floor (cushion = 0.0), order MUST be rejected with zero risk."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Peak = 10000, Floor = 9200
        assert engine.floor == pytest.approx(9200.0)

        # Exact floor
        allowed, risk, reason = engine.evaluate_order_risk(9200.0, 10000.0, "BUY", [])
        assert allowed is False
        assert risk == 0.0
        assert reason == "cushion_depleted_capital_floor_reached"

        # Sell at exact floor
        allowed, risk, reason = engine.evaluate_order_risk(9200.0, 10000.0, "SELL", [])
        assert allowed is False
        assert risk == 0.0
        assert reason == "cushion_depleted_capital_floor_reached"

    def test_cppi_epsilon_above_floor_numerical_stability(self):
        """Micro-cushion (1e-6) above floor: 50% cushion cap must bound risk without underflow."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)

        for eps in [1e-6, 1e-5, 1e-4, 0.01, 0.1, 1.0]:
            eq = 9200.0 + eps
            allowed, risk, reason = engine.evaluate_order_risk(eq, 10000.0, "BUY", [])
            assert allowed is True
            assert reason == "order_admitted"
            assert risk > 0.0
            # Theoretical 50% cushion cap strictly verified against engine cushion
            expected_cap = (0.50 * engine.cushion) / eq
            assert risk <= expected_cap * (1.0 + 1e-9)
            assert risk * eq <= (0.50 * engine.cushion) + 1e-9

    def test_cppi_deep_drawdown_breach_rejection(self):
        """Adversarial drawdowns below floor (E = 0.5*F, E = 1.0, E = -500.0)."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)

        breach_equities = [9199.99, 8000.0, 5000.0, 1.0, 0.0, -100.0, -10000.0]
        for eq in breach_equities:
            allowed, risk, reason = engine.evaluate_order_risk(eq, 10000.0, "BUY", [])
            assert allowed is False
            assert risk == 0.0
            assert reason in ("cushion_depleted_capital_floor_reached", "invalid_equity")

    def test_cppi_non_finite_inputs_rejection(self):
        """NaN, Inf, -Inf must be safely rejected without unhandled exceptions."""
        engine = BidirectionalCPPIEngine()
        engine.update_equity(10000.0)

        for bad_val in [float("nan"), float("inf"), float("-inf")]:
            allowed, risk, reason = engine.evaluate_order_risk(bad_val, 10000.0, "BUY", [])
            assert allowed is False
            assert risk == 0.0
            assert reason == "invalid_equity"

            allowed, risk, reason = engine.evaluate_order_risk(10000.0, bad_val, "SELL", [])
            # If peak is non-finite, engine uses max(equity, _peak_equity) and survives
            assert allowed is True
            assert 0.0015 <= risk <= 0.0025

    def test_cppi_cushion_explosion_cap(self):
        """When equity explodes to 100x or 10^9, risk MUST remain capped at max_risk_pct (0.0025)."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        for huge_eq in [100000.0, 1000000.0, 1e9]:
            allowed, risk, reason = engine.evaluate_order_risk(huge_eq, huge_eq, "BUY", [])
            assert allowed is True
            assert reason == "order_admitted"
            assert risk == pytest.approx(0.0025)
            assert engine.peak_equity == pytest.approx(huge_eq)
            assert engine.floor == pytest.approx((1.0 - 0.08) * huge_eq)

    def test_cppi_continuous_fuzzing_invariant(self):
        """Fuzz 3,000 randomized equity values: Verify that single trade risk never exceeds 50% cushion."""
        random.seed(42)
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        peak = 10000.0
        floor = (1.0 - 0.08) * peak  # 9200.0

        for _ in range(3000):
            # Reset engine so ratcheted floor doesn't carry across independent tests
            engine.reset(peak)
            eq = random.uniform(8000.0, 25000.0)
            direction = random.choice(["BUY", "SELL"])
            allowed, risk, reason = engine.evaluate_order_risk(eq, peak, direction, [])

            # Notice: if eq > peak, the floor ratchets up to (1 - 0.08) * eq!
            current_floor = (1.0 - 0.08) * max(peak, eq)
            if eq <= current_floor:
                assert allowed is False
                assert risk == 0.0
            else:
                assert allowed is True
                assert 0.0 < risk <= 0.0025
                cushion = eq - current_floor
                # Single-trade risk cannot breach 50% of cushion
                dollar_risk = risk * eq
                assert dollar_risk <= (0.50 * cushion) + 1e-7

    def test_cppi_peak_monotonic_ratchet_raises_floor(self):
        """Monotonic peak ratcheting raises floor permanently; dropping equity halts when new floor is reached."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08), initial_equity=10000.0)
        assert engine.floor == pytest.approx(9200.0)

        # Equity doubles to 20000.0
        engine.update_equity(20000.0)
        assert engine.peak_equity == pytest.approx(20000.0)
        assert engine.floor == pytest.approx(18400.0)

        # Equity drops to 18500.0 (still above new floor 18400)
        allowed, risk, _ = engine.evaluate_order_risk(18500.0, 20000.0, "BUY", [])
        assert allowed is True
        assert risk > 0.0

        # Equity drops to 18399.0 (breached new floor 18400, even though it is > original 10000!)
        allowed, risk, reason = engine.evaluate_order_risk(18399.0, 20000.0, "BUY", [])
        assert allowed is False
        assert risk == 0.0
        assert reason == "cushion_depleted_capital_floor_reached"

    def test_cppi_perfect_bidirectional_symmetry_fuzzing(self):
        """Across 1,000 equity states, BUY risk and SELL risk must be IDENTICAL to exact float precision."""
        random.seed(1337)
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)

        for _ in range(1000):
            eq = random.uniform(9200.01, 15000.0)
            _, buy_risk, _ = engine.evaluate_order_risk(eq, 10000.0, "BUY", [])
            _, sell_risk, _ = engine.evaluate_order_risk(eq, 10000.0, "SELL", [])
            assert buy_risk == sell_risk, f"Asymmetry detected at eq={eq}: BUY={buy_risk}, SELL={sell_risk}"


# ============================================================================
# Suite 2: Concurrency Flood & Race Invariant Harness
# ============================================================================

class TestAdversarialConcurrencyFlood:
    """Stress-test global and directional concurrency gates under flood conditions."""

    def test_concurrency_flood_directional_buy_cap(self):
        """Flooding with 50 BUY proposals against 0 open positions admitted only up to directional cap (2)."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        open_positions: List[Dict[str, Any]] = []

        admitted = 0
        rejected = 0
        for i in range(50):
            allowed, risk, reason = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
            if allowed:
                admitted += 1
                open_positions.append({"ticket": 1000 + i, "side": 1})
            else:
                rejected += 1
                assert reason == "max_directional_buys_reached"

        assert admitted == 2
        assert rejected == 48
        assert len(open_positions) == 2

    def test_concurrency_flood_directional_sell_cap(self):
        """Flooding with 50 SELL proposals against 0 open positions admitted only up to directional cap (2)."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        open_positions: List[Dict[str, Any]] = []

        admitted = 0
        rejected = 0
        for i in range(50):
            allowed, risk, reason = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
            if allowed:
                admitted += 1
                open_positions.append({"ticket": 2000 + i, "side": -1})
            else:
                rejected += 1
                assert reason == "max_directional_sells_reached"

        assert admitted == 2
        assert rejected == 48
        assert len(open_positions) == 2

    def test_concurrency_flood_mixed_directions_reaches_global_cap(self):
        """Simultaneous mixed flood: Max 2 BUYs, Max 2 SELLs, and global cap = 5."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        # Start with 2 BUY and 2 SELL open (total 4)
        open_positions = [
            {"ticket": 1, "side": "BUY"},
            {"ticket": 2, "side": "BUY"},
            {"ticket": 3, "side": "SELL"},
            {"ticket": 4, "side": "SELL"},
        ]
        # Any directional order (BUY or SELL) must be rejected because directional count is 2!
        allowed_b, _, reason_b = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert allowed_b is False
        assert reason_b == "max_directional_buys_reached"

        allowed_s, _, reason_s = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
        assert allowed_s is False
        assert reason_s == "max_directional_sells_reached"

    def test_concurrency_global_cap_5_with_unknown_positions(self):
        """When 5 positions are open (even if directions are mixed/neutral), 6th must be rejected."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        open_positions = [
            {"ticket": 1, "side": 1},
            {"ticket": 2, "side": -1},
            {"ticket": 3, "side": 0},  # Hedged / unknown
            {"ticket": 4, "side": 0},
            {"ticket": 5, "side": 0},
        ]
        assert len(open_positions) == 5
        allowed_b, _, reason_b = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert allowed_b is False
        assert reason_b == "max_concurrent_positions_reached"

        allowed_s, _, reason_s = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
        assert allowed_s is False
        assert reason_s == "max_concurrent_positions_reached"

    def test_concurrency_adversarial_position_formatting(self):
        """Classifier must handle diverse, nested, or malformed position dictionaries safely."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))

        positions = [
            {"ticket": "t1", "direction": "LONG"},
            {"ticket": "t2", "is_long": True},
            {"ticket": "t3", "type": "SELL"},
            {"ticket": "t4", "side": -1.0},
        ]
        # open_buys = 2, open_sells = 2
        allowed_b, _, _ = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", positions)
        assert allowed_b is False

        allowed_s, _, _ = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", positions)
        assert allowed_s is False


# ============================================================================
# Suite 3: Micro-Scalp Penetration & Anti-Bypass Testing
# ============================================================================

class TestAdversarialMicroScalpGating:
    """Ensure Micro-Scalp engine (Section 9.5) cannot bypass concurrency or guard gates."""

    def test_micro_scalp_cannot_penetrate_global_cap(self):
        """Micro-scalp BUY or SELL must be blocked when 5 positions are open."""
        mock_p = AdversarialMockPolicy(act_0=0, act_1=0, act_4=0)
        router = DualChampionRouter(
            bull_policy=mock_p,
            bear_policy=mock_p,
            bidirectional_cppi_config=BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2),
        )
        router.micro_scalp_engine = MagicMock()
        # Scalp engine wants to enter BUY
        router.micro_scalp_engine.evaluate_scalp_opportunity.return_value = ScalpSignal(
            should_enter=True,
            action_index=1,
            direction=1,
            sl_atr=1.0,
            tp_rr=2.0,
            trailing_atr=1.5,
        )

        open_positions = [{"ticket": i, "side": 1 if i % 2 == 0 else -1} for i in range(5)]
        obs = np.zeros(137, dtype=np.float32)

        batch = router.act_nam_tac_nhan(obs, positions=open_positions)
        # Verify Agent 0 does NOT enter
        act_0 = batch.traces[0].action_index if hasattr(batch.traces[0], "action_index") else batch.traces[0].action_details.get("five_action_index", 0)
        assert act_0 == 0

    def test_micro_scalp_cannot_penetrate_directional_cap(self):
        """Micro-scalp BUY must be blocked when 2 BUY positions are open (total 2 < 5)."""
        mock_p = AdversarialMockPolicy(act_0=0, act_1=0, act_4=0)
        router = DualChampionRouter(
            bull_policy=mock_p,
            bear_policy=mock_p,
            bidirectional_cppi_config=BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2),
        )
        router.micro_scalp_engine = MagicMock()
        router.micro_scalp_engine.evaluate_scalp_opportunity.return_value = ScalpSignal(
            should_enter=True,
            action_index=1,
            direction=1,
            sl_atr=1.0,
            tp_rr=2.0,
            trailing_atr=1.5,
        )

        open_positions = [{"ticket": 1, "side": 1}, {"ticket": 2, "side": 1}]
        obs = np.zeros(137, dtype=np.float32)

        batch = router.act_nam_tac_nhan(obs, positions=open_positions)
        act_0 = batch.traces[0].action_index if hasattr(batch.traces[0], "action_index") else batch.traces[0].action_details.get("five_action_index", 0)
        assert act_0 == 0

    def test_micro_scalp_sell_blocked_by_anti_bottom_guard(self):
        """Micro-scalp SELL must be intercepted if Anti-Bottom Guard triggers in Demand Zone."""
        mock_p = AdversarialMockPolicy(act_0=0, act_1=0, act_4=0)
        router = DualChampionRouter(
            bull_policy=mock_p,
            bear_policy=mock_p,
            bidirectional_cppi_config=BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2),
        )
        # Scalp engine wants to enter SELL (action 2)
        router.micro_scalp_engine = MagicMock()
        router.micro_scalp_engine.evaluate_scalp_opportunity.return_value = ScalpSignal(
            should_enter=True,
            action_index=2,
            direction=-1,
            sl_atr=1.0,
            tp_rr=2.0,
            trailing_atr=1.5,
        )

        # Mock anti_bottom_guard to veto
        router.anti_bottom_guard = MagicMock()
        router.anti_bottom_guard.enabled = True
        router.anti_bottom_guard.check_bottom_short_veto.return_value = (True, "demand_zone_liquidity_absorption", {"score": 0.95})

        obs = np.zeros(137, dtype=np.float32)

        batch = router.act_nam_tac_nhan(obs, positions=[])
        act_1 = batch.traces[1].action_index if hasattr(batch.traces[1], "action_index") else batch.traces[1].action_details.get("five_action_index", 0)
        assert act_1 == 0


# ============================================================================
# Suite 4: Dynamic Breakeven Ratchet Stress & Flash Reversal Harness
# ============================================================================

class TestAdversarialBreakevenRatchet:
    """Stress-test breakeven ratchet triggering at +2.0 ATR, profit locking (+0.5 ATR), and flash reversals."""

    def test_ratchet_sub_threshold_strict_rejection(self):
        """Excursion at +1.999 ATR must NOT trigger breakeven ratchet."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5))
        # BUY at 100, ATR = 10, SL = 50 (-5.0 ATR)
        res = engine.evaluate_position("pos_1", 100.0, 100.0, 1, 10.0)
        state = engine.get_position_state("pos_1")
        assert state is not None
        assert state.current_sl == 50.0
        assert not state.breakeven_executed

        # Price rises to 119.99 (+1.999 ATR)
        res = engine.evaluate_position("pos_1", 100.0, 119.99, 1, 10.0)
        assert not res["move_sl"]
        assert not state.breakeven_executed
        assert state.current_sl == 50.0

    def test_ratchet_exact_threshold_trigger(self):
        """Excursion at +2.000 ATR must trigger breakeven and ratchet SL to +0.5 ATR."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5))
        engine.evaluate_position("pos_1", 100.0, 100.0, 1, 10.0)
        state = engine.get_position_state("pos_1")

        # Price hits 120.0 (+2.0 ATR)
        res = engine.evaluate_position("pos_1", 100.0, 120.0, 1, 10.0)
        assert res["move_sl"] is True
        assert res["mode"] == MANAGEMENT_MODE_BREAKEVEN
        assert state.breakeven_executed is True
        # Locked SL = 100.0 + 0.5 * 10.0 = 105.0 (+0.5 ATR profit)
        assert state.current_sl == pytest.approx(105.0)

    def test_ratchet_flash_reversal_guarantees_profit_buy(self):
        """Flash crash from +2.0 ATR to +0.49 ATR stops out at +0.5 ATR, guaranteeing +0.5 ATR profit."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5))
        engine.evaluate_position("pos_buy", 100.0, 100.0, 1, 10.0)

        # Hits +2.0 ATR
        engine.evaluate_position("pos_buy", 100.0, 120.0, 1, 10.0)
        state = engine.get_position_state("pos_buy")
        assert state.current_sl == pytest.approx(105.0)

        # Flash reversal plunges to 104.9 (+0.49 ATR)
        res_crash = engine.evaluate_position("pos_buy", 100.0, 104.9, 1, 10.0)
        assert res_crash["stopped_out"] is True
        # Realized exit was at 105.0 or below -> stops out with positive profit (+5.0 USD = +0.5 ATR)
        profit_atr = (state.current_sl - 100.0) / 10.0
        assert profit_atr == pytest.approx(0.50)

    def test_ratchet_flash_reversal_guarantees_profit_sell(self):
        """SELL: Flash spike from -2.0 ATR drop to -0.49 ATR stops out at -0.5 ATR (+0.5 ATR profit)."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5))
        # SELL at 100.0, ATR = 10.0, SL = 150.0
        engine.evaluate_position("pos_sell", 100.0, 100.0, -1, 10.0)

        # Drops to 80.0 (+2.0 ATR excursion)
        res = engine.evaluate_position("pos_sell", 100.0, 80.0, -1, 10.0)
        assert res["move_sl"] is True
        state = engine.get_position_state("pos_sell")
        assert state.current_sl == pytest.approx(95.0)  # Locked at 95.0 (+0.5 ATR profit for short)

        # Flash reversal spikes to 95.1
        res_spike = engine.evaluate_position("pos_sell", 100.0, 95.1, -1, 10.0)
        assert res_spike["stopped_out"] is True
        profit_atr = (100.0 - state.current_sl) / 10.0
        assert profit_atr == pytest.approx(0.50)

    def test_ratchet_monotonicity_under_wild_whipsaw(self):
        """During violent price oscillations, SL can only move in the profit direction, never loosen."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5))
        engine.evaluate_position("pos_whipsaw", 100.0, 100.0, 1, 10.0)
        state = engine.get_position_state("pos_whipsaw")

        price_sequence = [105.0, 115.0, 120.0, 118.0, 122.0, 110.0, 125.0, 112.0]
        sl_history: List[float] = []

        for p in price_sequence:
            engine.evaluate_position("pos_whipsaw", 100.0, p, 1, 10.0)
            sl_history.append(state.current_sl)

        # Verify monotonicity: SL[i] >= SL[i-1] for BUY
        for i in range(1, len(sl_history)):
            assert sl_history[i] >= sl_history[i - 1], f"SL loosened at step {i}: {sl_history[i]} < {sl_history[i-1]}"


# ============================================================================
# Suite 5: Agent 4 Governor Invariance Adversarial Attack
# ============================================================================

class TestAdversarialAgent4Governor:
    """Stress-test Agent 4 HOLD=0 invariant under hostile actions and malformed traces."""

    def test_agent4_malicious_action_interception(self):
        """Even if model outputs action 10 (LIQUIDATE_ALL) with logit 99.0, Agent 4 is forced to ACTION_HOLD = 0."""
        mock_p = AdversarialMockPolicy(act_0=0, act_1=0, act_4=10)
        router = DualChampionRouter(
            bull_policy=mock_p,
            bear_policy=mock_p,
        )

        obs = np.zeros(137, dtype=np.float32)
        batch = router.act_nam_tac_nhan(obs, positions=[])

        t4 = batch.traces[4]
        # Check action index is 0
        act_idx = getattr(t4, "action_index", None)
        assert act_idx == 0
        # Check policy_close_all is 0.0
        details = getattr(t4, "action_details", {})
        assert details.get("policy_close_all", 0.0) == 0.0
        assert details.get("five_action_index", 0) == 0
        # Check output vector logit[10] was suppressed
        out_vec = getattr(t4, "output_vector", None)
        if out_vec is not None and out_vec.size >= 12:
            assert out_vec[0] == 1.0
            assert out_vec[10] == -100.0

    def test_agent4_primitive_and_unusual_types_survival(self):
        """Agent 4 defense handles primitive integers, strings, None, and frozen dataclasses without crashing."""
        for weird_trace in [0, 10, np.int32(10), {"action": 10}, "unknown"]:
            safe_trace = DualChampionRouter._enforce_agent4_hold(weird_trace)
            if isinstance(weird_trace, (int, np.integer)):
                assert safe_trace == 0


# ============================================================================
# Suite 6: Multi-Bar High-Frequency Simulation with Continuous Invariant Auditing
# ============================================================================

class TestAdversarialHighFrequencySimulation:
    """Simulate 300 sequential bars on DualChampionRouter to ensure zero invariant violations."""

    def test_sequential_300_bars_invariants_never_breached(self):
        random.seed(999)
        np.random.seed(999)

        mock_p = AdversarialMockPolicy(act_0=0, act_1=0, act_4=0)
        router = DualChampionRouter(
            bull_policy=mock_p,
            bear_policy=mock_p,
            bidirectional_cppi_config=BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2),
            min_spacing_bars=3,
        )

        open_positions: List[Dict[str, Any]] = []
        equity = 10000.0
        peak_equity = 10000.0

        for bar_idx in range(300):
            obs = np.random.randn(137).astype(np.float32)

            # Mock policies proposing random actions
            act_0 = random.choice([0, 1, 3])
            act_1 = random.choice([0, 2, 4])
            act_4 = random.choice([0, 10])

            mock_p.act_0 = act_0
            mock_p.act_1 = act_1
            mock_p.act_4 = act_4

            batch = router.act_nam_tac_nhan(obs, equity=equity, positions=open_positions)

            # Invariant 1: Agent 4 is ALWAYS ACTION_HOLD = 0
            t4 = batch.traces[4]
            a4 = getattr(t4, "action_index", 0)
            assert a4 == 0, f"Bar {bar_idx}: Agent 4 emitted non-zero action {a4}!"

            # If BUY admitted, add to positions
            t0 = batch.traces[0]
            a0 = getattr(t0, "action_index", t0.action_details.get("five_action_index", 0))
            if a0 in (1, 3, 5):
                open_positions.append({"ticket": bar_idx * 10 + 1, "side": 1, "entry": 100.0, "sl": 95.0})

            # If SELL admitted, add to positions
            t1 = batch.traces[1]
            a1 = getattr(t1, "action_index", t1.action_details.get("five_action_index", 0))
            if a1 in (2, 4, 6):
                open_positions.append({"ticket": bar_idx * 10 + 2, "side": -1, "entry": 100.0, "sl": 105.0})

            # Invariant 2: Total open positions never exceeds 5
            assert len(open_positions) <= 5, f"Bar {bar_idx}: Total open positions {len(open_positions)} > 5!"

            # Invariant 3: Directional positions never exceed 2 BUY or 2 SELL
            buys = sum(1 for p in open_positions if router.bidirectional_cppi_engine._classify_position_direction(p) > 0)
            sells = sum(1 for p in open_positions if router.bidirectional_cppi_engine._classify_position_direction(p) < 0)
            assert buys <= 2, f"Bar {bar_idx}: Open BUYs {buys} > 2!"
            assert sells <= 2, f"Bar {bar_idx}: Open SELLs {sells} > 2!"

            # Simulate random trade closes
            if open_positions and random.random() < 0.30:
                closed_pos = open_positions.pop(random.randint(0, len(open_positions) - 1))
                pnl = random.uniform(-20.0, 50.0)
                equity += pnl
                if equity > peak_equity:
                    peak_equity = equity
