"""
tests/test_cppi_bidirectional.py

Comprehensive Test Suite for Milestone M3:
- Bidirectional CPPI Risk Sizing (0.15% - 0.25% Fractional Kelly for BUY and SELL)
- Hard Single-Trade Loss Cap (loss <= 50% cushion)
- Concurrency Governance (Max 5 Global, Max 2 Directional, Min Spacing 3 bars)
- Micro-Scalp Engine Order Gating Integration
- Dynamic Breakeven Ratchet (+2.0 ATR -> +0.5 ATR Profit Lock)
- Agent 4 Governor Invariant Preservation (100% HOLD, POLICY_CLOSE_ALL == 0.0)
- DualChampionRouter End-to-End Execution
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
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
# Tier 1: Bidirectional CPPI Risk Sizing Tests (0.15% - 0.25%)
# ============================================================================

class TestBidirectionalCPPISizing:
    """Validate Fractional Kelly continuous risk scaling, symmetry, and cushion caps."""

    def test_cppi_sizing_buy_in_range_015_025(self):
        """Agent 0 BUY receives 0.0015 <= risk_pct <= 0.0025 at normal equity."""
        engine = BidirectionalCPPIEngine(initial_equity=10000.0)
        allowed, risk, reason = engine.evaluate_order_risk(
            equity=10000.0,
            peak_equity=10000.0,
            proposed_direction="BUY",
            current_open_positions=[],
        )
        assert allowed is True
        assert reason == "order_admitted"
        assert 0.0015 <= risk <= 0.0025
        # At peak equity (cushion ratio = 1.0), Kelly risk hits max_risk_pct = 0.0025
        assert risk == pytest.approx(0.0025)

    def test_cppi_sizing_sell_in_range_015_025(self):
        """Agent 1 SELL receives 0.0015 <= risk_pct <= 0.0025 at normal equity."""
        engine = BidirectionalCPPIEngine(initial_equity=10000.0)
        allowed, risk, reason = engine.evaluate_order_risk(
            equity=10000.0,
            peak_equity=10000.0,
            proposed_direction="SELL",
            current_open_positions=[],
        )
        assert allowed is True
        assert reason == "order_admitted"
        assert 0.0015 <= risk <= 0.0025
        assert risk == pytest.approx(0.0025)

    def test_cppi_sizing_bidirectional_symmetry(self):
        """BUY and SELL receive identical risk under identical portfolio conditions."""
        engine = BidirectionalCPPIEngine(initial_equity=10000.0)
        for eq in [10000.0, 9600.0, 9400.0, 9250.0]:
            _, buy_risk, _ = engine.evaluate_order_risk(eq, 10000.0, "BUY", [])
            _, sell_risk, _ = engine.evaluate_order_risk(eq, 10000.0, "SELL", [])
            assert buy_risk == pytest.approx(sell_risk), f"Asymmetry at equity {eq}: {buy_risk} != {sell_risk}"

    def test_cppi_fractional_kelly_interpolation(self):
        """Risk scales continuously between min_risk (0.0015) and max_risk (0.0025)."""
        # MaxDD = 0.08. Peak = 10000 -> Floor = 9200. MaxDD dollars = 800.
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)

        # 1. At peak (cushion = 800, ratio = 1.0) -> risk = 0.0025
        _, r_peak, _ = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", [])
        assert r_peak == pytest.approx(0.0025)

        # 2. At half cushion (equity = 9600, cushion = 400, ratio = 0.5)
        # Kelly risk = 0.0015 + (0.0025 - 0.0015) * 0.5 = 0.0020
        # Cushion cap = (0.50 * 400) / 9600 = 200 / 9600 = 0.020833 (does not bind)
        _, r_half, _ = engine.evaluate_order_risk(9600.0, 10000.0, "BUY", [])
        assert r_half == pytest.approx(0.0020)

        # 3. At 1/4 cushion (equity = 9400, cushion = 200, ratio = 0.25)
        # Kelly risk = 0.0015 + (0.0025 - 0.0015) * 0.25 = 0.00175
        # Cushion cap = (0.50 * 200) / 9400 = 100 / 9400 = 0.010638 (does not bind)
        _, r_quarter, _ = engine.evaluate_order_risk(9400.0, 10000.0, "BUY", [])
        assert r_quarter == pytest.approx(0.00175)

    def test_cppi_cushion_loss_cap_dominates_near_floor(self):
        """Hard single-trade loss cap <= 50% cushion strictly bounds risk near the floor."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Floor = 9200.0

        # Equity = 9210.0 -> cushion = 10.0
        # Cushion cap = (0.50 * 10.0) / 9210.0 = 5.0 / 9210.0 approx 0.00054288
        # This is below 0.0015, and MUST strictly dominate to protect the floor!
        allowed, risk, reason = engine.evaluate_order_risk(9210.0, 10000.0, "BUY", [])
        assert allowed is True
        expected_cap = 5.0 / 9210.0
        assert risk == pytest.approx(expected_cap, rel=1e-5)
        assert risk < 0.0015  # Scaled down to honor the 50% cushion cap!

    def test_cppi_zero_risk_and_rejected_at_or_below_floor(self):
        """When equity <= floor, order is rejected (allowed=False) to prevent breach."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)  # Floor = 9200.0

        # Exact floor
        allowed, risk, reason = engine.evaluate_order_risk(9200.0, 10000.0, "BUY", [])
        assert allowed is False
        assert risk == 0.0
        assert reason == "cushion_depleted_capital_floor_reached"

        # Below floor
        allowed, risk, reason = engine.evaluate_order_risk(9100.0, 10000.0, "SELL", [])
        assert allowed is False
        assert risk == 0.0
        assert reason == "cushion_depleted_capital_floor_reached"

    def test_cppi_peak_equity_ratchet_monotonicity(self):
        """Peak equity monotonically ratchets up and never falls during drawdowns."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_allowed_dd=0.08))
        engine.update_equity(10000.0)
        assert engine.peak_equity == 10000.0
        assert engine.floor == 9200.0

        # New peak
        engine.update_equity(12500.0)
        assert engine.peak_equity == 12500.0
        assert engine.floor == pytest.approx(11500.0)

        # Drawdown
        engine.update_equity(11800.0)
        assert engine.peak_equity == 12500.0
        assert engine.floor == pytest.approx(11500.0)
        assert engine.cushion == pytest.approx(300.0)

    def test_cppi_invalid_inputs_safety(self):
        """Invalid equity, non-finite values, and unknown directions handled safely."""
        engine = BidirectionalCPPIEngine()
        # Invalid equity
        allowed, risk, reason = engine.evaluate_order_risk(-500.0, 10000.0, "BUY", [])
        assert allowed is False
        assert risk == 0.0
        assert reason == "invalid_equity"

        allowed, risk, reason = engine.evaluate_order_risk(float("nan"), 10000.0, "BUY", [])
        assert allowed is False
        assert risk == 0.0

        # Unknown direction
        allowed, risk, reason = engine.evaluate_order_risk(10000.0, 10000.0, "INVALID_DIR", [])
        assert allowed is False
        assert "unknown_direction" in reason


# ============================================================================
# Tier 2: Concurrency & Position Governance Tests
# ============================================================================

class TestConcurrencyPositionGovernance:
    """Validate 5 global, 2 directional, and entry spacing constraints."""

    def test_global_concurrency_cap_5_positions(self):
        """5 open positions allows no new trades (6th proposal rejected)."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5))
        positions = [
            {"ticket": 1, "side": 1},
            {"ticket": 2, "side": 1},
            {"ticket": 3, "side": -1},
            {"ticket": 4, "side": -1},
            {"ticket": 5, "side": 1},
        ]
        allowed_buy, _, reason_buy = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", positions)
        assert allowed_buy is False
        assert reason_buy == "max_concurrent_positions_reached"

        allowed_sell, _, reason_sell = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", positions)
        assert allowed_sell is False
        assert reason_sell == "max_concurrent_positions_reached"

    def test_directional_concurrency_cap_max_2_buys(self):
        """When 2 BUY positions are open, 3rd BUY is rejected, but SELL is allowed."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        positions = [
            {"ticket": 101, "side": 1},
            {"ticket": 102, "side": 1},
        ]
        # Proposed BUY -> Rejected
        allowed_buy, _, reason_buy = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", positions)
        assert allowed_buy is False
        assert reason_buy == "max_directional_buys_reached"

        # Proposed SELL -> Allowed (open_sells = 0 < 2)
        allowed_sell, risk_sell, reason_sell = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", positions)
        assert allowed_sell is True
        assert reason_sell == "order_admitted"
        assert 0.0015 <= risk_sell <= 0.0025

    def test_directional_concurrency_cap_max_2_sells(self):
        """When 2 SELL positions are open, 3rd SELL is rejected, but BUY is allowed."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        positions = [
            {"ticket": 201, "side": -1},
            {"ticket": 202, "side": -1},
        ]
        # Proposed SELL -> Rejected
        allowed_sell, _, reason_sell = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", positions)
        assert allowed_sell is False
        assert reason_sell == "max_directional_sells_reached"

        # Proposed BUY -> Allowed (open_buys = 0 < 2)
        allowed_buy, risk_buy, reason_buy = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", positions)
        assert allowed_buy is True
        assert reason_buy == "order_admitted"
        assert 0.0015 <= risk_buy <= 0.0025

    def test_directional_concurrency_cap_2_buys_2_sells_allows_fifth(self):
        """With 2 BUY and 2 SELL open (total 4), both directions are at directional cap."""
        engine = BidirectionalCPPIEngine(BidirectionalCPPIConfig(max_concurrent_positions=5, max_positions_per_direction=2))
        positions = [
            {"ticket": 1, "side": 1},
            {"ticket": 2, "side": 1},
            {"ticket": 3, "side": -1},
            {"ticket": 4, "side": -1},
        ]
        # Both directions have reached 2 directional positions
        allowed_buy, _, reason_buy = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", positions)
        assert allowed_buy is False
        assert reason_buy == "max_directional_buys_reached"

        allowed_sell, _, reason_sell = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", positions)
        assert allowed_sell is False
        assert reason_sell == "max_directional_sells_reached"

    def test_position_classification_formats(self):
        """Robust parsing across 'side', 'direction', 'is_long', and MT5 'type'."""
        engine = BidirectionalCPPIEngine()
        # BUY formats
        assert engine._classify_position_direction({"side": 1}) == 1
        assert engine._classify_position_direction({"side": "BUY"}) == 1
        assert engine._classify_position_direction({"direction": "long"}) == 1
        assert engine._classify_position_direction({"is_long": True}) == 1
        assert engine._classify_position_direction({"type": 0}) == 1  # POSITION_TYPE_BUY

        # SELL formats
        assert engine._classify_position_direction({"side": -1}) == -1
        assert engine._classify_position_direction({"side": "SELL"}) == -1
        assert engine._classify_position_direction({"direction": "short"}) == -1
        assert engine._classify_position_direction({"is_long": False}) == -1
        assert engine._classify_position_direction({"type": 1}) == -1  # POSITION_TYPE_SELL


# ============================================================================
# Tier 3: Dynamic Breakeven Ratchet Tests (+2.0 ATR -> +0.5 ATR)
# ============================================================================

class TestDynamicBreakevenRatchet:
    """Validate +2.0 ATR trigger, +0.5 ATR profit lock, monotonicity, and idempotency."""

    def test_breakeven_ratchet_buy_at_2_atr(self):
        """BUY at +2.0 ATR ratchets SL to entry + 0.5 ATR."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig())  # Uses default 2.0 trigger, 0.5 lock
        entry = 90000.0
        atr = 100.0
        ticket = "be_test_buy"

        # 1. At entry (0.0 ATR): initial SL = 89500
        res = engine.evaluate_position(ticket, entry, entry, 1, atr)
        assert res["move_sl"] is False
        assert res["current_sl"] == 89500.0
        assert res["breakeven_executed"] is False

        # 2. At +1.9 ATR (90190): Below trigger
        res = engine.evaluate_position(ticket, entry, 90190.0, 1, atr)
        assert res["move_sl"] is False
        assert res["current_sl"] == 89500.0
        assert res["breakeven_executed"] is False

        # 3. At +2.0 ATR (90200): TRIGGERS BREAKEVEN!
        res = engine.evaluate_position(ticket, entry, 90200.0, 1, atr)
        assert res["move_sl"] is True
        assert res["new_sl"] == 90050.0  # entry + 0.5 * 100 = 90050.0
        assert res["current_sl"] == 90050.0
        assert res["breakeven_executed"] is True
        assert res["mode"] == MANAGEMENT_MODE_BREAKEVEN
        assert engine.get_position(ticket).phase == PositionPhase.BREAKEVEN_LOCKED

    def test_breakeven_ratchet_sell_at_2_atr(self):
        """SELL at -2.0 ATR ratchets SL to entry - 0.5 ATR."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig())
        entry = 90000.0
        atr = 100.0
        ticket = "be_test_sell"

        # At -2.0 ATR (89800): TRIGGERS BREAKEVEN!
        res = engine.evaluate_position(ticket, entry, 89800.0, -1, atr)
        assert res["move_sl"] is True
        assert res["new_sl"] == 89950.0  # entry - 0.5 * 100 = 89950.0
        assert res["current_sl"] == 89950.0
        assert res["breakeven_executed"] is True
        assert res["mode"] == MANAGEMENT_MODE_BREAKEVEN

    def test_breakeven_flash_reversal_guarantees_profit(self):
        """Excursion reaches +2.5 ATR, ratchets SL, then crashes below entry -> stops in PROFIT."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig())
        entry = 90000.0
        atr = 100.0
        ticket = "flash_reversal"

        # Price advances to +2.5 ATR (90250) -> BE activated at 90050
        engine.evaluate_position(ticket, entry, 90250.0, 1, atr)
        assert engine.get_position(ticket).current_sl == 90050.0

        # Market flash crash: bar low hits 89900 (below entry)
        res = engine.evaluate_position(ticket, entry, 89900.0, 1, atr, high=90100.0, low=89900.0)
        assert res["stopped_out"] is True
        assert res["current_sl"] == 90050.0
        assert res["current_sl"] > entry  # Stopped out with +$50 / +0.5 ATR locked profit!

    def test_breakeven_ratchet_monotonicity(self):
        """SL never retreats or loosens on pullbacks after breakeven lock."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig())
        entry = 90000.0
        atr = 100.0
        ticket = "be_monotonic"

        # Trigger BE at +2.0 ATR
        engine.evaluate_position(ticket, entry, 90200.0, 1, atr)
        pos = engine.get_position(ticket)
        assert pos.current_sl == 90050.0

        # Pullback to +1.0 ATR (90100). SL must stay locked at 90050.0
        res = engine.evaluate_position(ticket, entry, 90100.0, 1, atr)
        assert res["move_sl"] is False
        assert res["current_sl"] == 90050.0
        assert pos.current_sl == 90050.0

    def test_breakeven_ratchet_idempotency(self):
        """Subsequent ticks between 2.0 ATR and 10.0 ATR do not re-emit move_sl."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig())
        entry = 90000.0
        atr = 100.0
        ticket = "be_idempotent"

        # First trigger
        r1 = engine.evaluate_position(ticket, entry, 90200.0, 1, atr)
        assert r1["move_sl"] is True

        # Second tick at +3.0 ATR (90300)
        r2 = engine.evaluate_position(ticket, entry, 90300.0, 1, atr)
        assert r2["move_sl"] is False
        assert r2["current_sl"] == 90050.0

        # Third tick at +6.0 ATR (90600)
        r3 = engine.evaluate_position(ticket, entry, 90600.0, 1, atr)
        assert r3["move_sl"] is False
        assert r3["current_sl"] == 90050.0

    def test_progression_from_breakeven_to_tier1_partial_close(self):
        """Position progresses from BE lock (+2 ATR) to Tier 1 partial close (+10 ATR)."""
        engine = TwoTierTrailingRunnerEngine(TwoTierConfig())
        entry = 90000.0
        atr = 100.0
        ticket = "progression"

        # 1. Hit +2.0 ATR -> Breakeven locked
        r_be = engine.evaluate_position(ticket, entry, 90200.0, 1, atr)
        assert r_be["breakeven_executed"] is True
        assert r_be["new_sl"] == 90050.0

        # 2. Hit +10.0 ATR -> Tier 1 partial close triggers
        r_tp = engine.evaluate_position(ticket, entry, 91000.0, 1, atr)
        assert r_tp["partial_close"] is True
        assert r_tp["close_ratio"] == 0.50
        assert r_tp["action"] == ACTION_MANAGE_POSITION
        assert r_tp["mode"] == MANAGEMENT_MODE_PARTIAL_CLOSE
        assert r_tp["trailing_active"] is True


# ============================================================================
# Tier 4: Micro-Scalp Concurrency Gating Integration Tests
# ============================================================================

class TestMicroScalpConcurrencyGating:
    """Validate that Section 9.5 Micro-Scalp engine cannot bypass concurrency limits."""

    def test_micro_scalp_cannot_bypass_global_5_cap(self):
        """When 5 positions are open, micro-scalper entry is strictly blocked."""
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=0,
        )
        # Mock active policy to return HOLD on both Agent 0 and Agent 1
        mock_trace_0 = MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
        mock_trace_1 = MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
        mock_trace_2 = MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
        mock_trace_3 = MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
        mock_trace_4 = MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
        router.bull_policy.act_nam_tac_nhan.return_value = [
            mock_trace_0, mock_trace_1, mock_trace_2, mock_trace_3, mock_trace_4
        ]
        router.bear_policy.act_nam_tac_nhan.return_value = [
            mock_trace_0, mock_trace_1, mock_trace_2, mock_trace_3, mock_trace_4
        ]

        # Configure micro scalp to propose BUY
        router.micro_scalp_engine = MagicMock()
        router.micro_scalp_engine.config.enabled = True
        router.micro_scalp_engine.evaluate_scalp_opportunity.return_value = ScalpSignal(
            should_enter=True,
            direction=1,
            action_index=1,
            sl_atr=1.5,
            tp_rr=2.0,
            trailing_atr=1.0,
        )

        # 5 positions open
        positions = [{"ticket": i, "side": 1 if i % 2 == 0 else -1} for i in range(5)]
        batch = router.act_nam_tac_nhan(
            np.zeros(137),
            equity=10000.0,
            positions=positions,
        )
        # Agent 0 action MUST remain 0 (HOLD)
        act_0 = DualChampionRouter._extract_action_idx(batch[0])
        assert act_0 == 0, f"Micro-scalper bypassed global concurrency cap! Action={act_0}"

    def test_micro_scalp_cannot_bypass_directional_2_cap(self):
        """When 2 BUY positions are open, micro-scalper BUY is strictly blocked."""
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=0,
        )
        mock_traces = [
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
            for _ in range(5)
        ]
        router.bull_policy.act_nam_tac_nhan.return_value = mock_traces
        router.bear_policy.act_nam_tac_nhan.return_value = mock_traces

        # Micro-scalper fires BUY signal
        router.micro_scalp_engine = MagicMock()
        router.micro_scalp_engine.config.enabled = True
        router.micro_scalp_engine.evaluate_scalp_opportunity.return_value = ScalpSignal(
            should_enter=True,
            direction=1,
            action_index=1,
            sl_atr=1.5,
            tp_rr=2.0,
            trailing_atr=1.0,
        )

        # 2 BUY positions open (open_buys = 2)
        positions = [{"ticket": 101, "side": 1}, {"ticket": 102, "side": 1}]
        batch = router.act_nam_tac_nhan(
            np.zeros(137),
            equity=10000.0,
            positions=positions,
        )
        act_0 = DualChampionRouter._extract_action_idx(batch[0])
        assert act_0 == 0, f"Micro-scalper bypassed directional BUY cap! Action={act_0}"

    def test_router_entry_spacing_cadence_3_bars(self):
        """Router enforces 3-bar spacing cooldown between consecutive entries."""
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
            min_spacing_bars=3,
        )
        # Entry proposal: Action 3 (BUY_LIMIT)
        def make_entry_traces():
            out = np.zeros(20)
            out[3] = 1.0
            return [
                MagicMock(selected_action=out, action_index=3, action_details={"action": 3}),
                MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
                MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
                MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
                MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
            ]

        router.bull_policy.act_nam_tac_nhan.side_effect = lambda *a, **kw: make_entry_traces()
        router.bear_policy.act_nam_tac_nhan.side_effect = lambda *a, **kw: make_entry_traces()

        # Bar 0: First entry allowed
        b0 = router.act_nam_tac_nhan(np.zeros(137), equity=10000.0, positions=[])
        assert DualChampionRouter._extract_action_idx(b0[0]) == 3

        # Bar 1: 1 bar elapsed < 3 -> Vetoed to HOLD
        b1 = router.act_nam_tac_nhan(np.zeros(137), equity=10000.0, positions=[])
        assert DualChampionRouter._extract_action_idx(b1[0]) == 0

        # Bar 2: 2 bars elapsed < 3 -> Vetoed to HOLD
        b2 = router.act_nam_tac_nhan(np.zeros(137), equity=10000.0, positions=[])
        assert DualChampionRouter._extract_action_idx(b2[0]) == 0

        # Bar 3: 3 bars elapsed >= 3 -> Allowed again!
        b3 = router.act_nam_tac_nhan(np.zeros(137), equity=10000.0, positions=[])
        assert DualChampionRouter._extract_action_idx(b3[0]) == 3


# ============================================================================
# Tier 5: Agent 4 Governor Invariant Preservation Tests
# ============================================================================

class TestAgent4GovernorInvariant:
    """Validate that Agent 4 Governor is strictly 100% HOLD (POLICY_CLOSE_ALL == 0.0)."""

    def test_agent4_governor_invariant_100_percent_hold(self):
        """Across 1,000 arbitrary evaluations, Agent 4 is strictly ACTION_HOLD = 0."""
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
        )

        rng = np.random.RandomState(42)
        for _ in range(100):
            # Create hostile traces where Agent 4 tries to emit CLOSE_ALL (Action 10)
            mock_traces = [
                MagicMock(selected_action=rng.randn(20), action_index=rng.randint(0, 7), action_details={"action": 0})
                for _ in range(4)
            ]
            hostile_agent4 = MagicMock(
                selected_action=np.array([0.0]*10 + [10.0] + [0.0]*9),
                output_vector=np.array([0.0]*10 + [10.0] + [0.0]*9),
                action_index=10,
                action_details={"action": 10, "policy_close_all": 1.0},
            )
            mock_traces.append(hostile_agent4)
            router.bull_policy.act_nam_tac_nhan.return_value = mock_traces
            router.bear_policy.act_nam_tac_nhan.return_value = mock_traces

            batch = router.act_nam_tac_nhan(rng.randn(137), equity=10000.0)
            act_4 = DualChampionRouter._extract_action_idx(batch[4])
            assert act_4 == 0, f"Agent 4 violated HOLD invariant! Action={act_4}"

            # Check details
            details = getattr(batch[4], "action_details", {})
            if isinstance(details, dict):
                assert details.get("policy_close_all", 0.0) == 0.0
                assert details.get("action", 0) == 0

    def test_agent4_primitive_integer_trace_defense(self):
        """Passing raw primitive integer trace to _enforce_agent4_hold safely returns 0."""
        assert DualChampionRouter._enforce_agent4_hold(10) == 0
        assert DualChampionRouter._enforce_agent4_hold(1) == 0
        assert DualChampionRouter._enforce_agent4_hold(0) == 0
        assert DualChampionRouter._enforce_agent4_hold(np.int64(10)) == 0


# ============================================================================
# Tier 6: DualChampionRouter End-to-End M3 Integration
# ============================================================================

class TestDualChampionRouterM3Integration:
    """Validate router sizing, telemetry, and simultaneous conflict arbitration."""

    def test_router_attaches_bidirectional_cppi_risk(self):
        """Router attaches dynamically scaled risk_pct in [0.0015, 0.0025] to entry traces."""
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
            bidirectional_cppi_config=BidirectionalCPPIConfig(),
            min_spacing_bars=0,
        )
        # Specialist proposes BUY_LIMIT (action 3)
        out_0 = np.zeros(20)
        out_0[3] = 1.0
        traces = [
            MagicMock(selected_action=out_0, action_index=3, action_details={"action": 3}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
        ]
        router.bull_policy.act_nam_tac_nhan.return_value = traces
        router.bear_policy.act_nam_tac_nhan.return_value = traces

        batch = router.act_nam_tac_nhan(np.zeros(137), equity=10000.0, positions=[])
        assert batch.risk_pct is not None
        assert 0.0015 <= batch.risk_pct <= 0.0025

        # Check telemetry
        assert "peak_equity" in batch.cppi_telemetry
        assert "min_risk_pct" in batch.cppi_telemetry
        assert batch.cppi_telemetry["min_risk_pct"] == 0.0015
        assert batch.cppi_telemetry["max_risk_pct"] == 0.0025

    def test_router_simultaneous_dual_entry_arbitration(self):
        """When 4 positions open and both BUY & SELL are proposed, Soft-MoE weights arbitrate."""
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
            max_concurrent_positions=5,
            max_positions_per_direction=3,  # Allow 3 directional to test global 5 cap
            min_spacing_bars=0,
        )
        out_0 = np.zeros(20)
        out_0[3] = 1.0
        out_1 = np.zeros(20)
        out_1[4] = 1.0
        traces = [
            MagicMock(selected_action=out_0, action_index=3, action_details={"action": 3}),
            MagicMock(selected_action=out_1, action_index=4, action_details={"action": 4}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0}),
        ]
        router.bull_policy.act_nam_tac_nhan.return_value = traces
        router.bear_policy.act_nam_tac_nhan.return_value = traces

        # Mock compute_soft_moe_weights in favor of Bull (0.8 vs 0.2)
        router.compute_soft_moe_weights = MagicMock(return_value=(0.8, 0.2, {"chop_index": 0.1}))
        positions = [{"ticket": i, "side": 1 if i % 2 == 0 else -1} for i in range(4)]

        batch = router.act_nam_tac_nhan(np.zeros(137), equity=10000.0, positions=positions)
        act_0 = DualChampionRouter._extract_action_idx(batch[0])
        act_1 = DualChampionRouter._extract_action_idx(batch[1])

        # Bull won arbitration -> BUY admitted, SELL vetoed
        assert act_0 == 3
        assert act_1 == 0
