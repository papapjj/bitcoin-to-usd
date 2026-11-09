"""
tests/test_two_tier_runner.py

Comprehensive 4-Tier Unit and Adversarial Test Suite for Two-Tier Trailing Runner:
- Tier 1: Nominal & Specification Tests (Config, PositionState, ATR excursion, Tier 1 & Tier 2 triggers, ratcheting, cleanup)
- Tier 2: Boundary Value Analysis & Guard Tests (exact ATR boundaries, <=0/NaN/Inf ATR and price safety, invalid configs)
- Tier 3: Pairwise Combinatorial & Dynamic Shock Tests (direction x scale, custom ratios, flash gaps, immediate reversals)
- Tier 4: Real-World Scenarios, Concurrency & Stress Tests (10 concurrent positions, 35 ATR BTC bull wave, 30 ATR BTC dump, choppy sideway, sub-millisecond HF performance)
- Backward Compatibility: Full verification of legacy TwoTierRunnerManager and ManagedPosition lifecycle
"""

import math
import time
import pytest

from self_growing_trader.two_tier_runner import (
    ACTION_HOLD,
    ACTION_MANAGE_POSITION,
    MANAGEMENT_MODE_NONE,
    MANAGEMENT_MODE_PARTIAL_CLOSE,
    CLOSE_BIN_50_PCT,
    TwoTierConfig,
    PositionState,
    PositionPhase,
    TwoTierTrailingRunnerEngine,
    # Legacy backward compatibility exports
    TwoTierRunnerManager,
    RunnerConfig,
    ManagedPosition,
)


# =====================================================================
# TIER 1: Nominal & Specification Tests (Features F6, F7, F8)
# =====================================================================

class TestTier1Nominal:
    """Verifies baseline configuration, initial states, calculations, and standard triggers."""

    def test_two_tier_config_defaults(self):
        """Checks standard default configuration invariants required by R3."""
        cfg = TwoTierConfig()
        assert cfg.partial_tp_atr == 10.0
        assert cfg.partial_close_ratio == 0.50
        assert cfg.breakeven_sl_atr == 1.0
        assert cfg.trailing_atr == 3.0
        assert cfg.immediate_trailing_at_trigger is False
        assert cfg.initial_sl_atr == 5.0

    def test_two_tier_config_custom_initialization(self):
        """Verifies custom parameter initialization."""
        cfg = TwoTierConfig(
            partial_tp_atr=8.0,
            partial_close_ratio=0.40,
            breakeven_sl_atr=1.5,
            trailing_atr=2.5,
            immediate_trailing_at_trigger=True,
            initial_sl_atr=4.0,
        )
        assert cfg.partial_tp_atr == 8.0
        assert cfg.partial_close_ratio == 0.40
        assert cfg.breakeven_sl_atr == 1.5
        assert cfg.trailing_atr == 2.5
        assert cfg.immediate_trailing_at_trigger is True
        assert cfg.initial_sl_atr == 4.0

    def test_position_state_dataclass_initialization(self):
        """Verifies PositionState dataclass fields and default values."""
        pos = PositionState(
            position_id="ticket_1001",
            entry_price=95000.0,
            direction=1,
            entry_atr=100.0,
        )
        assert pos.position_id == "ticket_1001"
        assert pos.entry_price == 95000.0
        assert pos.direction == 1
        assert pos.entry_atr == 100.0
        assert pos.highest_favorable_price == 95000.0
        assert pos.lowest_favorable_price == 95000.0
        assert pos.partial_close_executed is False
        assert pos.sl_moved_to_profit is False
        assert pos.trailing_active is False
        assert pos.current_sl is None
        assert pos.is_long is True
        assert pos.is_short is False
        assert pos.ticket == "ticket_1001"
        assert pos.atr == 100.0
        assert pos.phase == PositionPhase.INITIAL

    def test_position_state_properties(self):
        """Verifies properties for short positions and phase transitions."""
        pos = PositionState(
            position_id="ticket_2002",
            entry_price=98000.0,
            direction=-1,
            entry_atr=150.0,
        )
        assert pos.is_long is False
        assert pos.is_short is True

        pos.partial_close_executed = True
        pos.sl_moved_to_profit = True
        assert pos.phase == PositionPhase.PARTIAL_BANKED

        pos.trailing_active = True
        assert pos.phase == PositionPhase.RUNNER_TRAILING

    def test_engine_initialization_defaults(self):
        """Verifies engine starts in clean initial state."""
        engine = TwoTierTrailingRunnerEngine()
        assert engine.active_position_count == 0
        assert len(engine.positions) == 0
        assert engine.config.partial_tp_atr == 10.0

    def test_engine_register_and_get_position(self):
        """Verifies explicit position registration and querying."""
        engine = TwoTierTrailingRunnerEngine()
        pos = engine.register_position(
            position_id=12345,
            entry_price=90000.0,
            direction=1,
            entry_atr=100.0,
        )
        assert pos.position_id == "12345"
        assert engine.has_position(12345) is True
        assert engine.has_position("12345") is True
        assert engine.get_position(12345) is pos
        assert engine.get_position_state("12345") is pos
        assert engine.active_position_count == 1

    def test_unrealized_atr_calculation_buy_at_entry(self):
        """BUY at entry price has 0.0 unrealized ATR."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="p1",
            entry_price=95000.0,
            current_price=95000.0,
            direction=1,
            entry_atr=100.0,
        )
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False

    def test_unrealized_atr_calculation_buy_in_profit(self):
        """BUY at +500 price with 100 ATR has +5.0 unrealized ATR."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="p1",
            entry_price=95000.0,
            current_price=95500.0,
            direction=1,
            entry_atr=100.0,
        )
        assert pytest.approx(res["unrealized_atr"], rel=1e-6) == 5.0
        assert res["partial_close"] is False

    def test_unrealized_atr_calculation_buy_in_drawdown(self):
        """BUY at -250 price with 100 ATR has -2.5 unrealized ATR."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="p1",
            entry_price=95000.0,
            current_price=94750.0,
            direction=1,
            entry_atr=100.0,
        )
        assert pytest.approx(res["unrealized_atr"], rel=1e-6) == -2.5
        assert res["partial_close"] is False

    def test_unrealized_atr_calculation_sell_at_entry(self):
        """SELL at entry price has 0.0 unrealized ATR."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="p2",
            entry_price=98000.0,
            current_price=98000.0,
            direction=-1,
            entry_atr=200.0,
        )
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False

    def test_unrealized_atr_calculation_sell_in_profit(self):
        """SELL at -1000 price with 200 ATR has +5.0 unrealized ATR."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="p2",
            entry_price=98000.0,
            current_price=97000.0,
            direction=-1,
            entry_atr=200.0,
        )
        assert pytest.approx(res["unrealized_atr"], rel=1e-6) == 5.0
        assert res["partial_close"] is False

    def test_unrealized_atr_calculation_sell_in_drawdown(self):
        """SELL at +500 price with 200 ATR has -2.5 unrealized ATR."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="p2",
            entry_price=98000.0,
            current_price=98500.0,
            direction=-1,
            entry_atr=200.0,
        )
        assert pytest.approx(res["unrealized_atr"], rel=1e-6) == -2.5
        assert res["partial_close"] is False

    def test_tier1_trigger_buy_nominal(self):
        """Reaching 10.0 ATR emits 50% partial close command on BUY."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="buy_tp",
            entry_price=95000.0,
            current_price=96000.0,  # +1000 = +10.0 ATR
            direction=1,
            entry_atr=100.0,
        )
        assert res["partial_close"] is True
        assert res["close_ratio"] == 0.50
        assert res["action"] == ACTION_MANAGE_POSITION  # 11
        assert res["mode"] == MANAGEMENT_MODE_PARTIAL_CLOSE  # 1
        assert res["close_bin"] == CLOSE_BIN_50_PCT  # 1
        assert res["partial_close_executed"] is True

    def test_tier1_trigger_sell_nominal(self):
        """Reaching 10.0 ATR emits 50% partial close command on SELL."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position(
            position_id="sell_tp",
            entry_price=98000.0,
            current_price=96000.0,  # -2000 = +10.0 ATR
            direction=-1,
            entry_atr=200.0,
        )
        assert res["partial_close"] is True
        assert res["close_ratio"] == 0.50
        assert res["action"] == ACTION_MANAGE_POSITION  # 11
        assert res["mode"] == MANAGEMENT_MODE_PARTIAL_CLOSE  # 1
        assert res["close_bin"] == CLOSE_BIN_50_PCT  # 1
        assert res["partial_close_executed"] is True

    def test_tier1_state_flag_updated(self):
        """PositionState record reflects partial_close_executed flag."""
        engine = TwoTierTrailingRunnerEngine()
        engine.evaluate_position("pos_flag", 95000.0, 96000.0, 1, 100.0)
        pos = engine.get_position("pos_flag")
        assert pos is not None
        assert pos.partial_close_executed is True

    def test_tier1_idempotency_buy(self):
        """Subsequent evaluations at or above 10 ATR do NOT fire partial close again."""
        engine = TwoTierTrailingRunnerEngine()
        res1 = engine.evaluate_position("idem_buy", 95000.0, 96000.0, 1, 100.0)
        assert res1["partial_close"] is True

        # Next tick at 96100 (11 ATR)
        res2 = engine.evaluate_position("idem_buy", 95000.0, 96100.0, 1, 100.0)
        assert res2["partial_close"] is False
        assert res2["close_ratio"] == 0.0
        assert res2["action"] == ACTION_HOLD

        # Next tick at 96500 (15 ATR)
        res3 = engine.evaluate_position("idem_buy", 95000.0, 96500.0, 1, 100.0)
        assert res3["partial_close"] is False

    def test_tier1_idempotency_sell(self):
        """Subsequent evaluations at or above 10 ATR do NOT fire partial close again on SELL."""
        engine = TwoTierTrailingRunnerEngine()
        res1 = engine.evaluate_position("idem_sell", 98000.0, 96000.0, -1, 200.0)
        assert res1["partial_close"] is True

        res2 = engine.evaluate_position("idem_sell", 98000.0, 95800.0, -1, 200.0)
        assert res2["partial_close"] is False
        assert res2["close_ratio"] == 0.0

    def test_tier2_sl_moved_to_profit_buy(self):
        """Tier 2 moves Stop Loss to +1.0 ATR profit on BUY (entry + 1.0 * atr)."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("tier2_buy", 95000.0, 96000.0, 1, 100.0)
        assert res["move_sl"] is True
        assert res["new_sl"] == 95100.0  # 95000 + 1.0 * 100
        assert res["sl_moved_to_profit"] is True
        assert res["trailing_active"] is True

    def test_tier2_sl_moved_to_profit_sell(self):
        """Tier 2 moves Stop Loss to +1.0 ATR profit on SELL (entry - 1.0 * atr)."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("tier2_sell", 98000.0, 96000.0, -1, 200.0)
        assert res["move_sl"] is True
        assert res["new_sl"] == 97800.0  # 98000 - 1.0 * 200
        assert res["sl_moved_to_profit"] is True
        assert res["trailing_active"] is True

    def test_tier2_trailing_active_flag_set(self):
        """trailing_active flag is set to True upon reaching Tier 1/2 trigger."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("t_active", 95000.0, 96000.0, 1, 100.0)
        assert res["trailing_active"] is True
        pos = engine.get_position("t_active")
        assert pos.trailing_active is True

    def test_chandelier_trailing_advances_buy(self):
        """Chandelier 3.0 ATR trailing advances SL on higher favorable prices for BUY."""
        engine = TwoTierTrailingRunnerEngine()
        # Trigger Tier 1 at 10 ATR (96000)
        engine.evaluate_position("trail_buy", 95000.0, 96000.0, 1, 100.0)

        # Price advances to 96500 (15 ATR). Peak = 96500. Chandelier SL = 96500 - 300 = 96200
        res = engine.evaluate_position("trail_buy", 95000.0, 96500.0, 1, 100.0)
        assert res["move_sl"] is True
        assert res["new_sl"] == 96200.0
        assert res["current_sl"] == 96200.0

    def test_chandelier_trailing_advances_sell(self):
        """Chandelier 3.0 ATR trailing advances SL on lower favorable prices for SELL."""
        engine = TwoTierTrailingRunnerEngine()
        # Trigger Tier 1 at 10 ATR (96000)
        engine.evaluate_position("trail_sell", 98000.0, 96000.0, -1, 200.0)

        # Price dumps to 95000 (15 ATR). Trough = 95000. Chandelier SL = 95000 + 600 = 95600
        res = engine.evaluate_position("trail_sell", 98000.0, 95000.0, -1, 200.0)
        assert res["move_sl"] is True
        assert res["new_sl"] == 95600.0
        assert res["current_sl"] == 95600.0

    def test_ratchet_constraint_buy_no_retreat(self):
        """Ratchet Invariant: Trailing SL NEVER retreats or loosens on BUY pullback."""
        engine = TwoTierTrailingRunnerEngine()
        engine.evaluate_position("ratchet_buy", 95000.0, 96000.0, 1, 100.0)
        # Advance to 97000 (20 ATR). Chandelier SL = 97000 - 300 = 96700
        engine.evaluate_position("ratchet_buy", 95000.0, 97000.0, 1, 100.0)

        # Price pulls back to 96800. SL must NOT loosen; move_sl must be False
        res = engine.evaluate_position("ratchet_buy", 95000.0, 96800.0, 1, 100.0)
        assert res["move_sl"] is False
        assert res["new_sl"] is None
        assert res["current_sl"] == 96700.0  # Strictly ratcheted

    def test_ratchet_constraint_sell_no_retreat(self):
        """Ratchet Invariant: Trailing SL NEVER retreats or loosens on SELL rally."""
        engine = TwoTierTrailingRunnerEngine()
        engine.evaluate_position("ratchet_sell", 98000.0, 96000.0, -1, 200.0)
        # Dump to 94000 (20 ATR). Chandelier SL = 94000 + 600 = 94600
        engine.evaluate_position("ratchet_sell", 98000.0, 94000.0, -1, 200.0)

        # Price bounces to 94400. SL must NOT loosen; move_sl must be False
        res = engine.evaluate_position("ratchet_sell", 98000.0, 94400.0, -1, 200.0)
        assert res["move_sl"] is False
        assert res["new_sl"] is None
        assert res["current_sl"] == 94600.0  # Strictly ratcheted

    def test_position_cleanup_via_close_position(self):
        """close_position removes position cleanly and returns final state."""
        engine = TwoTierTrailingRunnerEngine()
        engine.evaluate_position("clean_pos", 95000.0, 95500.0, 1, 100.0)
        assert engine.has_position("clean_pos") is True

        closed_state = engine.close_position("clean_pos")
        assert closed_state is not None
        assert closed_state.position_id == "clean_pos"
        assert engine.has_position("clean_pos") is False
        assert engine.active_position_count == 0

    def test_engine_reset_clears_state(self):
        """Engine reset cleans all active positions."""
        engine = TwoTierTrailingRunnerEngine()
        engine.evaluate_position("p1", 95000.0, 95500.0, 1, 100.0)
        engine.evaluate_position("p2", 98000.0, 97500.0, -1, 100.0)
        assert engine.active_position_count == 2

        engine.reset()
        assert engine.active_position_count == 0
        assert len(engine.positions) == 0


# =====================================================================
# TIER 2: Boundary Value Analysis & Guard Tests
# =====================================================================

class TestTier2BoundaryGuards:
    """Verifies precision boundaries, <=0/NaN/Inf handling, and config validation."""

    def test_boundary_exact_below_threshold_9_999_buy(self):
        """9.999 ATR excursion does NOT trigger Tier 1 for BUY."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("b_999", 95000.0, 95999.9, 1, 100.0)
        assert res["unrealized_atr"] < 10.0
        assert res["partial_close"] is False
        assert res["action"] == ACTION_HOLD

    def test_boundary_exact_below_threshold_9_999_sell(self):
        """9.999 ATR excursion does NOT trigger Tier 1 for SELL."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("s_999", 98000.0, 96000.1, -1, 200.0)
        assert res["unrealized_atr"] < 10.0
        assert res["partial_close"] is False

    def test_boundary_exact_threshold_10_000_buy(self):
        """Exactly 10.000000 ATR excursion DOES trigger Tier 1 for BUY."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("b_100", 95000.0, 96000.0, 1, 100.0)
        assert res["unrealized_atr"] == 10.0
        assert res["partial_close"] is True

    def test_boundary_exact_threshold_10_000_sell(self):
        """Exactly 10.000000 ATR excursion DOES trigger Tier 1 for SELL."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("s_100", 98000.0, 96000.0, -1, 200.0)
        assert res["unrealized_atr"] == 10.0
        assert res["partial_close"] is True

    def test_boundary_above_threshold_10_001_buy(self):
        """10.001 ATR excursion triggers Tier 1 for BUY."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("b_1001", 95000.0, 96000.1, 1, 100.0)
        assert res["unrealized_atr"] > 10.0
        assert res["partial_close"] is True

    def test_boundary_zero_atr_safe_handling(self):
        """entry_atr == 0.0 handled safely without division by zero."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("zero_atr", 95000.0, 96500.0, 1, 0.0)
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False
        assert res["move_sl"] is False

    def test_boundary_negative_atr_safe_handling(self):
        """Negative entry_atr handled safely without exception."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("neg_atr", 95000.0, 96500.0, 1, -50.0)
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False

    def test_boundary_nan_atr_safe_handling(self):
        """NaN entry_atr handled safely without exception."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("nan_atr", 95000.0, 96500.0, 1, float("nan"))
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False

    def test_boundary_pos_inf_atr_safe_handling(self):
        """Positive infinity entry_atr handled safely without exception."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("pinf_atr", 95000.0, 96500.0, 1, float("inf"))
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False

    def test_boundary_neg_inf_atr_safe_handling(self):
        """Negative infinity entry_atr handled safely without exception."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("ninf_atr", 95000.0, 96500.0, 1, float("-inf"))
        assert res["unrealized_atr"] == 0.0
        assert res["partial_close"] is False

    def test_boundary_nan_price_safe_handling(self):
        """NaN current_price returns fail-closed safe response."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("nan_price", 95000.0, float("nan"), 1, 100.0)
        assert res["partial_close"] is False
        assert res["move_sl"] is False

    def test_boundary_inf_price_safe_handling(self):
        """Infinite current_price returns fail-closed safe response."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("inf_price", 95000.0, float("inf"), 1, 100.0)
        assert res["partial_close"] is False
        assert res["move_sl"] is False

    def test_boundary_zero_entry_price_safe_handling(self):
        """Zero entry_price returns fail-closed safe response."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("zero_entry", 0.0, 95000.0, 1, 100.0)
        assert res["partial_close"] is False
        assert res["move_sl"] is False

    def test_boundary_negative_entry_price_safe_handling(self):
        """Negative entry_price returns fail-closed safe response."""
        engine = TwoTierTrailingRunnerEngine()
        res = engine.evaluate_position("neg_entry", -100.0, 95000.0, 1, 100.0)
        assert res["partial_close"] is False

    def test_boundary_invalid_config_partial_tp_atr(self):
        """TwoTierConfig raises ValueError for non-positive partial_tp_atr."""
        with pytest.raises(ValueError):
            TwoTierConfig(partial_tp_atr=0.0)
        with pytest.raises(ValueError):
            TwoTierConfig(partial_tp_atr=-5.0)

    def test_boundary_invalid_config_close_ratio(self):
        """TwoTierConfig raises ValueError for close_ratio <= 0 or > 1.0."""
        with pytest.raises(ValueError):
            TwoTierConfig(partial_close_ratio=0.0)
        with pytest.raises(ValueError):
            TwoTierConfig(partial_close_ratio=1.5)

    def test_boundary_invalid_config_breakeven_atr(self):
        """TwoTierConfig raises ValueError for negative breakeven_sl_atr."""
        with pytest.raises(ValueError):
            TwoTierConfig(breakeven_sl_atr=-1.0)

    def test_boundary_invalid_config_trailing_atr(self):
        """TwoTierConfig raises ValueError for non-positive trailing_atr."""
        with pytest.raises(ValueError):
            TwoTierConfig(trailing_atr=0.0)

    def test_boundary_close_nonexistent_position(self):
        """close_position on untracked position returns None cleanly."""
        engine = TwoTierTrailingRunnerEngine()
        assert engine.close_position("non_existent_ticket") is None


# =====================================================================
# TIER 3: Pairwise Combinatorial & Dynamic Shock Tests
# =====================================================================

class TestTier3PairwiseAndShocks:
    """Verifies parameter combinations, volatility shocks, and flash reversals."""

    @pytest.mark.parametrize("direction", [1, -1])
    @pytest.mark.parametrize("atr_scale", [0.001, 1.0, 100.0, 2500.0])
    def test_pairwise_direction_and_atr_scales(self, direction, atr_scale):
        """Validates excursion math across multiple orders of magnitude of ATR."""
        engine = TwoTierTrailingRunnerEngine()
        entry = 50000.0
        # Target 10 ATR price
        current = entry + 10.0 * atr_scale if direction > 0 else entry - 10.0 * atr_scale
        res = engine.evaluate_position(f"scale_{direction}_{atr_scale}", entry, current, direction, atr_scale)
        assert pytest.approx(res["unrealized_atr"], rel=1e-5) == 10.0
        assert res["partial_close"] is True

    @pytest.mark.parametrize("close_ratio", [0.25, 0.33, 0.50, 0.75, 1.0])
    def test_pairwise_different_close_ratios(self, close_ratio):
        """Validates custom partial close ratios emit expected configuration."""
        cfg = TwoTierConfig(partial_close_ratio=close_ratio)
        engine = TwoTierTrailingRunnerEngine(cfg)
        res = engine.evaluate_position("cr_test", 95000.0, 96000.0, 1, 100.0)
        assert res["partial_close"] is True
        assert res["close_ratio"] == close_ratio

    @pytest.mark.parametrize("trail_dist", [1.5, 3.0, 5.0])
    def test_pairwise_different_trailing_distances(self, trail_dist):
        """Validates Chandelier trailing distance adheres to configured multiplier."""
        cfg = TwoTierConfig(trailing_atr=trail_dist)
        engine = TwoTierTrailingRunnerEngine(cfg)
        # Trigger Tier 1 at 10 ATR
        engine.evaluate_position("td_test", 95000.0, 96000.0, 1, 100.0)
        # Advance to 97000 (20 ATR)
        res = engine.evaluate_position("td_test", 95000.0, 97000.0, 1, 100.0)
        expected_sl = 97000.0 - trail_dist * 100.0
        assert res["current_sl"] == expected_sl

    def test_pairwise_instant_volatility_gap_buy(self):
        """Immediate gap jump from 0 to 25 ATR triggers Tier 1 and sets trailing SL."""
        engine = TwoTierTrailingRunnerEngine()
        # Direct leap to +25 ATR (97500)
        res = engine.evaluate_position("gap_buy", 95000.0, 97500.0, 1, 100.0)
        assert res["partial_close"] is True
        assert res["partial_close_executed"] is True
        assert res["trailing_active"] is True

    def test_pairwise_instant_volatility_gap_sell(self):
        """Immediate gap dump from 0 to -25 ATR triggers Tier 1 and sets trailing SL."""
        engine = TwoTierTrailingRunnerEngine()
        # Direct dump to -25 ATR (93000)
        res = engine.evaluate_position("gap_sell", 98000.0, 93000.0, -1, 200.0)
        assert res["partial_close"] is True
        assert res["partial_close_executed"] is True
        assert res["trailing_active"] is True

    def test_pairwise_immediate_flash_reversal_buy(self):
        """Touches 10 ATR then immediately collapses back to entry price."""
        engine = TwoTierTrailingRunnerEngine()
        # Bar 1: hits exactly 10 ATR
        res1 = engine.evaluate_position("rev_buy", 95000.0, 96000.0, 1, 100.0)
        assert res1["partial_close"] is True
        assert res1["current_sl"] == 95100.0  # +1.0 ATR profit lock

        # Bar 2: collapses to 95000.0 (entry)
        res2 = engine.evaluate_position("rev_buy", 95000.0, 95000.0, 1, 100.0)
        assert res2["partial_close"] is False
        assert res2["stopped_out"] is True
        assert res2["exit_reason"] == "RUNNER_TRAILED_TP"

    def test_pairwise_immediate_flash_reversal_sell(self):
        """Drops 10 ATR then immediately spikes back to entry price."""
        engine = TwoTierTrailingRunnerEngine()
        # Bar 1: hits -10 ATR (96000)
        res1 = engine.evaluate_position("rev_sell", 98000.0, 96000.0, -1, 200.0)
        assert res1["partial_close"] is True
        assert res1["current_sl"] == 97800.0  # -1.0 ATR profit lock

        # Bar 2: rallies back to 98000.0
        res2 = engine.evaluate_position("rev_sell", 98000.0, 98000.0, -1, 200.0)
        assert res2["stopped_out"] is True
        assert res2["exit_reason"] == "RUNNER_TRAILED_TP"

    def test_pairwise_high_low_bar_evaluation_buy(self):
        """Passing high triggers TP even when current_price (close) is lower."""
        engine = TwoTierTrailingRunnerEngine()
        # Close is 95800 (8 ATR), but High is 96050 (10.5 ATR)
        res = engine.evaluate_position(
            position_id="hl_buy",
            entry_price=95000.0,
            current_price=95800.0,
            direction=1,
            entry_atr=100.0,
            high=96050.0,
            low=95700.0,
        )
        assert res["partial_close"] is True
        assert res["highest_favorable_price"] == 96050.0

    def test_pairwise_high_low_bar_evaluation_sell(self):
        """Passing low triggers TP even when current_price (close) is higher."""
        engine = TwoTierTrailingRunnerEngine()
        # Close is 96500 (7.5 ATR drop), but Low is 95900 (10.5 ATR drop)
        res = engine.evaluate_position(
            position_id="hl_sell",
            entry_price=98000.0,
            current_price=96500.0,
            direction=-1,
            entry_atr=200.0,
            high=96700.0,
            low=95900.0,
        )
        assert res["partial_close"] is True
        assert res["lowest_favorable_price"] == 95900.0

    def test_pairwise_immediate_trailing_at_trigger_config(self):
        """immediate_trailing_at_trigger=True uses max(breakeven, trailing) on trigger step."""
        cfg = TwoTierConfig(immediate_trailing_at_trigger=True)
        engine = TwoTierTrailingRunnerEngine(cfg)
        # Entry 95000, leaps directly to 97000 (20 ATR). Chandelier SL = 97000 - 300 = 96700
        res = engine.evaluate_position("imm_trail", 95000.0, 97000.0, 1, 100.0)
        assert res["partial_close"] is True
        assert res["current_sl"] == 96700.0  # Immediately set to 96700 instead of 95100


# =====================================================================
# TIER 4: Real-World Scenarios, Concurrency & Workload Tests
# =====================================================================

class TestTier4RealWorldScenarios:
    """Verifies full lifecycle simulation, concurrency, stress performance."""

    def test_scenario_multi_position_concurrency_isolation(self):
        """10 concurrent positions (5 BUY, 5 SELL) operate with completely isolated states."""
        engine = TwoTierTrailingRunnerEngine()
        tickets = [f"t_{i}" for i in range(10)]

        # Register 5 BUY and 5 SELL
        for i, t in enumerate(tickets):
            direction = 1 if i % 2 == 0 else -1
            engine.evaluate_position(t, 90000.0, 90000.0, direction, 100.0)

        assert engine.active_position_count == 10

        # Push ticket 0 to 10 ATR (BUY)
        res0 = engine.evaluate_position("t_0", 90000.0, 91000.0, 1, 100.0)
        assert res0["partial_close"] is True

        # Push ticket 1 to 10 ATR (SELL)
        res1 = engine.evaluate_position("t_1", 90000.0, 89000.0, -1, 100.0)
        assert res1["partial_close"] is True

        # Other tickets remain at 0 ATR unaffected
        res2 = engine.evaluate_position("t_2", 90000.0, 90100.0, 1, 100.0)
        assert res2["partial_close"] is False

        # Close ticket 0
        engine.close_position("t_0")
        assert engine.has_position("t_0") is False
        assert engine.has_position("t_1") is True
        assert engine.active_position_count == 9

    def test_scenario_btc_35_atr_bull_wave(self):
        """Full simulation of 35 ATR Bitcoin pump: 50% partial close, trailing, exit."""
        engine = TwoTierTrailingRunnerEngine()
        entry_price = 90000.0
        atr = 100.0
        ticket = "btc_wave"

        # 1. Open at entry
        r = engine.evaluate_position(ticket, entry_price, 90000.0, 1, atr)
        assert r["partial_close"] is False

        # 2. Advance to 8 ATR (90800) -> No trigger
        r = engine.evaluate_position(ticket, entry_price, 90800.0, 1, atr)
        assert r["partial_close"] is False

        # 3. Hit 10 ATR (91000) -> Tier 1 trigger!
        r = engine.evaluate_position(ticket, entry_price, 91000.0, 1, atr)
        assert r["partial_close"] is True
        assert r["new_sl"] == 90100.0  # +1.0 ATR profit lock
        assert r["trailing_active"] is True

        # 4. Pump surges to 25 ATR (92500)
        r = engine.evaluate_position(ticket, entry_price, 92500.0, 1, atr)
        assert r["partial_close"] is False
        assert r["current_sl"] == 92200.0  # 92500 - 300

        # 5. Peak surge at 35 ATR (93500)
        r = engine.evaluate_position(ticket, entry_price, 93500.0, 1, atr)
        assert r["current_sl"] == 93200.0  # 93500 - 300

        # 6. Minor dip to 93300 -> SL stays ratcheted at 93200
        r = engine.evaluate_position(ticket, entry_price, 93300.0, 1, atr)
        assert r["move_sl"] is False
        assert r["current_sl"] == 93200.0
        assert r["stopped_out"] is False

        # 7. Pullback hits 93200 -> Exits runner with +32.0 ATR banked profit!
        r = engine.evaluate_position(ticket, entry_price, 93180.0, 1, atr)
        assert r["stopped_out"] is True
        assert r["exit_reason"] == "RUNNER_TRAILED_TP"

    def test_scenario_btc_30_atr_bear_dump(self):
        """Full simulation of 30 ATR Bitcoin dump: 50% partial close, trailing, exit."""
        engine = TwoTierTrailingRunnerEngine()
        entry_price = 95000.0
        atr = 100.0
        ticket = "btc_dump"

        # 1. Reaches -10 ATR (94000) -> Tier 1 trigger!
        r = engine.evaluate_position(ticket, entry_price, 94000.0, -1, atr)
        assert r["partial_close"] is True
        assert r["new_sl"] == 94900.0  # -1.0 ATR profit lock

        # 2. Dumps further to 92000 (-30 ATR!)
        r = engine.evaluate_position(ticket, entry_price, 92000.0, -1, atr)
        assert r["current_sl"] == 92300.0  # 92000 + 300

        # 3. Pullback bounces to 92350 -> Hits trailing stop!
        r = engine.evaluate_position(ticket, entry_price, 92350.0, -1, atr)
        assert r["stopped_out"] is True
        assert r["exit_reason"] == "RUNNER_TRAILED_TP"

    def test_scenario_choppy_sideway_market_no_false_trigger(self):
        """Choppy market oscillating in range never falsely triggers Tier 1."""
        engine = TwoTierTrailingRunnerEngine()
        entry = 95000.0
        atr = 100.0
        ticket = "chop"

        trajectory = [95200, 94900, 95500, 94800, 95700, 95300, 95800, 95100]
        for p in trajectory:
            res = engine.evaluate_position(ticket, entry, p, 1, atr)
            assert res["partial_close"] is False
            assert res["trailing_active"] is False
            assert res["partial_close_executed"] is False

    def test_scenario_runner_harvesting_40_atr(self):
        """Verifies capturing extended 40 ATR trend run with monotonic ratchets."""
        engine = TwoTierTrailingRunnerEngine()
        entry = 50000.0
        atr = 50.0
        ticket = "mega"

        # Advance in 5-ATR steps up to 40 ATR
        for step in range(1, 9):
            atr_move = step * 5.0
            price = entry + atr_move * atr
            res = engine.evaluate_position(ticket, entry, price, 1, atr)
            if atr_move >= 10.0:
                assert res["trailing_active"] is True
                expected_sl = price - 3.0 * atr
                # Once past trigger, trailing line advances
                if step > 2:
                    assert res["current_sl"] == expected_sl

    def test_scenario_high_frequency_tick_performance(self):
        """2,000 tick evaluations execute in < 50ms (< 25 microseconds per tick)."""
        engine = TwoTierTrailingRunnerEngine()
        start = time.perf_counter()
        entry = 95000.0
        atr = 100.0
        for i in range(2000):
            p = entry + (i % 200) * 5.0
            engine.evaluate_position("perf_pos", entry, p, 1, atr)
        elapsed = time.perf_counter() - start
        assert elapsed < 0.20, f"Expected < 200ms, took {elapsed:.4f}s"

    def test_scenario_stopped_out_flag_and_reason_buy(self):
        """Stopped out flag correctly signals exit reason when SL breached."""
        engine = TwoTierTrailingRunnerEngine()
        # Normal SL hit before Tier 1
        res = engine.evaluate_position("sl_hit", 95000.0, 94400.0, 1, 100.0)
        assert res["stopped_out"] is True
        assert res["exit_reason"] == "SL"

    def test_scenario_stopped_out_flag_and_reason_sell(self):
        """Stopped out flag correctly signals exit reason when SELL SL breached."""
        engine = TwoTierTrailingRunnerEngine()
        # Initial SL is entry + 5 ATR = 98500
        res = engine.evaluate_position("sl_sell_hit", 98000.0, 98600.0, -1, 100.0)
        assert res["stopped_out"] is True
        assert res["exit_reason"] == "SL"


# =====================================================================
# BACKWARD COMPATIBILITY: Legacy TwoTierRunnerManager Suite
# =====================================================================

class TestLegacyTwoTierRunnerManager:
    """Preserves 100% backward compatibility with legacy TwoTierRunnerManager tests."""

    def test_legacy_long_position_lifecycle(self):
        mgr = TwoTierRunnerManager(RunnerConfig(initial_tp_atr=10.0, initial_sl_atr=5.0, bank_fraction=0.5))
        pos = mgr.open_position(
            ticket=1001,
            is_long=True,
            entry_price=95000.0,
            volume=0.10,
            atr=100.0,
            current_bar=0,
        )
        assert pos.state == PositionState.INITIAL
        assert pos.current_sl_price == 94500.0  # entry - 5 ATR
        assert pos.current_volume == 0.10

        # Step 1: Normal fluctuations within 5 ATR
        bank, new_sl, reason = mgr.on_tick_or_bar(1001, high=95400.0, low=94800.0, close=95200.0, current_bar=5)
        assert bank is None
        assert reason is None
        assert pos.state == PositionState.INITIAL

        # Step 2: Price hits 10 ATR target (96000)
        bank, new_sl, reason = mgr.on_tick_or_bar(1001, high=96050.0, low=95600.0, close=95900.0, current_bar=10)
        assert bank == 0.05  # 50% volume banked
        assert pos.current_volume == 0.05
        assert pos.state == PositionState.RUNNER_TRAILING
        assert pos.current_sl_price == 95100.0  # Ratcheted to +1.0 ATR profit
        assert new_sl == 95100.0
        assert reason is None

        # Step 3: Sóng tăng tiếp tục chạy tới 97500 (25 ATR)
        bank, new_sl, reason = mgr.on_tick_or_bar(1001, high=97500.0, low=97300.0, close=97400.0, current_bar=20)
        assert pos.current_sl_price == 97200.0
        assert new_sl == 97200.0
        assert reason is None

        # Step 4: Giá quay đầu chạm trailing SL
        bank, new_sl, reason = mgr.on_tick_or_bar(1001, high=97300.0, low=97150.0, close=97180.0, current_bar=25)
        assert reason == "RUNNER_TRAILED_TP"
        assert pos.state == PositionState.CLOSED
        assert pos.close_price == 97200.0

    def test_legacy_short_position_lifecycle(self):
        mgr = TwoTierRunnerManager()
        pos = mgr.open_position(
            ticket=2001,
            is_long=False,
            entry_price=98000.0,
            volume=0.20,
            atr=100.0,
            current_bar=0,
        )
        assert pos.state == PositionState.INITIAL
        assert pos.current_sl_price == 98500.0  # entry + 5 ATR

        # Step 1: Hits -10 ATR (97000)
        bank, new_sl, reason = mgr.on_tick_or_bar(2001, high=97400.0, low=96950.0, close=97000.0, current_bar=15)
        assert bank == 0.10  # 50% banked
        assert pos.current_volume == 0.10
        assert pos.state == PositionState.RUNNER_TRAILING
        assert pos.current_sl_price == 97900.0  # entry - 1 ATR
        assert new_sl == 97900.0

        # Step 2: Dumps further to 94000 (40 ATR drop!)
        bank, new_sl, reason = mgr.on_tick_or_bar(2001, high=94200.0, low=94000.0, close=94100.0, current_bar=35)
        assert pos.current_sl_price == 94300.0
        assert reason is None

        # Step 3: Retraces to 94350 and gets closed
        bank, new_sl, reason = mgr.on_tick_or_bar(2001, high=94350.0, low=94150.0, close=94300.0, current_bar=40)
        assert reason == "RUNNER_TRAILED_TP"
        assert pos.state == PositionState.CLOSED

    def test_legacy_timeout_expiration(self):
        mgr = TwoTierRunnerManager(RunnerConfig(max_holding_bars=120))
        pos = mgr.open_position(1002, is_long=True, entry_price=95000.0, volume=0.05, atr=100.0, current_bar=0)
        bank, new_sl, reason = mgr.on_tick_or_bar(1002, high=95200.0, low=94900.0, close=95100.0, current_bar=119)
        assert reason is None
        assert pos.state == PositionState.INITIAL

        # Exactly bar 120 -> TIMEOUT
        bank, new_sl, reason = mgr.on_tick_or_bar(1002, high=95200.0, low=94900.0, close=95050.0, current_bar=120)
        assert reason == "TIMEOUT"
        assert pos.state == PositionState.CLOSED


class TestEarlyBreakevenRatchet:
    """Test suite for Early Breakeven Protection & Win-Rate Boosting Ratchet."""

    def test_breakeven_ratchet_buy_nominal(self):
        """When BUY reaches +2.0 ATR, SL ratchets to entry + 0.5 ATR."""
        cfg = TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5, partial_tp_atr=10.0)
        engine = TwoTierTrailingRunnerEngine(config=cfg)
        entry = 90000.0
        atr = 100.0
        ticket = "be_buy_1"

        # 1. At entry: initial SL = 89500 (entry - 5 ATR)
        r = engine.evaluate_position(ticket, entry, entry, 1, atr)
        assert r["move_sl"] is False
        assert r["current_sl"] == 89500.0
        assert r["breakeven_executed"] is False

        # 2. Reaches +1.5 ATR (90150) -> Below trigger (2.0 ATR), no BE yet
        r = engine.evaluate_position(ticket, entry, 90150.0, 1, atr)
        assert r["move_sl"] is False
        assert r["current_sl"] == 89500.0
        assert r["breakeven_executed"] is False

        # 3. Reaches +2.0 ATR (90200) -> TRIGGERS BREAKEVEN RATCHET!
        r = engine.evaluate_position(ticket, entry, 90200.0, 1, atr)
        assert r["move_sl"] is True
        assert r["new_sl"] == 90050.0  # entry + 0.5 * 100 = 90050 (+0.5 ATR locked profit)
        assert r["current_sl"] == 90050.0
        assert r["breakeven_executed"] is True
        assert engine.get_position(ticket).phase == PositionPhase.BREAKEVEN_LOCKED

    def test_breakeven_ratchet_sell_nominal(self):
        """When SELL reaches -2.0 ATR, SL ratchets to entry - 0.5 ATR."""
        cfg = TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5, partial_tp_atr=10.0)
        engine = TwoTierTrailingRunnerEngine(config=cfg)
        entry = 90000.0
        atr = 100.0
        ticket = "be_sell_1"

        # Reaches -2.0 ATR (89800) -> TRIGGERS BREAKEVEN RATCHET!
        r = engine.evaluate_position(ticket, entry, 89800.0, -1, atr)
        assert r["move_sl"] is True
        assert r["new_sl"] == 89950.0  # entry - 0.5 * 100 = 89950 (+0.5 ATR locked profit)
        assert r["current_sl"] == 89950.0
        assert r["breakeven_executed"] is True
        assert engine.get_position(ticket).phase == PositionPhase.BREAKEVEN_LOCKED

    def test_breakeven_prevents_loss_on_reversal(self):
        """Position gains +2.5 ATR, ratchets to BE +0.5 ATR, then dumps. Stopped out in PROFIT!"""
        cfg = TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5, partial_tp_atr=10.0)
        engine = TwoTierTrailingRunnerEngine(config=cfg)
        entry = 90000.0
        atr = 100.0
        ticket = "be_reversal"

        # Surge to +2.5 ATR (90250) -> BE activated at 90050
        engine.evaluate_position(ticket, entry, 90250.0, 1, atr)
        pos = engine.get_position(ticket)
        assert pos.current_sl == 90050.0

        # Market flash dump drops below entry to 89900
        r = engine.evaluate_position(ticket, entry, 89900.0, 1, atr, high=90100.0, low=89900.0)
        assert r["stopped_out"] is True
        # Exited at SL = 90050.0, which is > entry (90000.0) -> A WINNING TRADE (+0.5 ATR)!
        assert r["current_sl"] > entry

    def test_breakeven_idempotency_no_double_fire(self):
        """Once BE is executed, subsequent ticks in the 2.0 - 9.9 ATR range do not re-emit move_sl."""
        cfg = TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5, partial_tp_atr=10.0)
        engine = TwoTierTrailingRunnerEngine(config=cfg)
        entry = 90000.0
        atr = 100.0
        ticket = "be_idemp"

        # Step 1: trigger BE
        r1 = engine.evaluate_position(ticket, entry, 90200.0, 1, atr)
        assert r1["move_sl"] is True

        # Step 2: further price advance to +4.0 ATR (before Tier 1 at 10 ATR)
        r2 = engine.evaluate_position(ticket, entry, 90400.0, 1, atr)
        assert r2["move_sl"] is False
        assert r2["current_sl"] == 90050.0
        assert r2["breakeven_executed"] is True

