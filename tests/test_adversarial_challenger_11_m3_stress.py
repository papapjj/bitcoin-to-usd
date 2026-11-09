"""
tests/test_adversarial_challenger_11_m3_stress.py

Adversarial Challenger Verification & Empirical Stress Suite for Milestone M3:
Dynamic Bidirectional CPPI, Concurrency Position Governance, Dynamic Breakeven Ratchet,
and Agent 4 Governor Invariance.

Role: Empirical Challenger (11_M3_2)
Objective: Stress-test boundaries, flood limits, probe failure modes, and verify invariants.
"""

from __future__ import annotations

import math
import time
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
from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
)
from self_growing_trader.micro_scalp_engine import ScalpSignal
from self_growing_trader.two_tier_runner import (
    ACTION_HOLD,
    ACTION_MANAGE_POSITION,
    MANAGEMENT_MODE_BREAKEVEN,
    MANAGEMENT_MODE_PARTIAL_CLOSE,
    TwoTierConfig,
    TwoTierTrailingRunnerEngine,
)


# ============================================================================
# Section 1: Boundary Fuzzing & Stress on Dynamic Bidirectional CPPI
# ============================================================================

class TestAdversarialBidirectionalCPPIBoundaries:
    """Stress-test edge cases, boundary transitions, and mathematical invariants of CPPI."""

    def test_boundary_cushion_zero_and_near_zero(self):
        """Probe exact floor boundary: cushion = 0.0, micro-cushion, and breach."""
        config = BidirectionalCPPIConfig(max_allowed_dd=0.08, min_risk_pct=0.0015, max_risk_pct=0.0025)
        engine = BidirectionalCPPIEngine(config=config, initial_equity=10000.0)
        # Floor is 9200.0

        # Exact floor: cushion = 0.0 -> Must be strictly rejected
        allowed, risk, reason = engine.evaluate_order_risk(9200.0, 10000.0, "BUY", [])
        assert not allowed, f"Order allowed at exact floor: {reason}"
        assert risk == 0.0
        assert "floor" in reason.lower() or "depleted" in reason.lower()

        # Micro-cushion: equity = 9200.0001 (cushion = 0.0001)
        # 50% cushion cap: 0.5 * 0.0001 / 9200.0001 ~= 5.43e-9
        allowed_micro, risk_micro, reason_micro = engine.evaluate_order_risk(9200.0001, 10000.0, "BUY", [])
        assert allowed_micro, f"Micro-cushion should admit order with capped risk, got: {reason_micro}"
        assert 0.0 < risk_micro <= (0.50 * 0.0001 / 9200.0001) * 1.0001
        assert risk_micro < config.min_risk_pct, "Risk must be dominated by cushion cap near floor"

        # Slight drawdown breach: equity = 9199.9999
        allowed_breach, risk_breach, reason_breach = engine.evaluate_order_risk(9199.9999, 10000.0, "BUY", [])
        assert not allowed_breach
        assert risk_breach == 0.0

    def test_deep_drawdown_breach_resilience(self):
        """Sweep equity deep into drawdown breach down to 0.01; verify no negative risk or crashes."""
        config = BidirectionalCPPIConfig(max_allowed_dd=0.08)
        engine = BidirectionalCPPIEngine(config=config, initial_equity=10000.0)

        for eq in [9199.0, 8000.0, 5000.0, 1000.0, 100.0, 1.0, 0.01]:
            allowed, risk, reason = engine.evaluate_order_risk(eq, 10000.0, "BUY", [])
            assert not allowed, f"Deep breach allowed at equity={eq}"
            assert risk == 0.0, f"Non-zero risk {risk} at equity={eq}"
            assert math.isfinite(risk)

    def test_cushion_explosion_runaway_equity(self):
        """Test runaway equity explosion (10x, 100x, 1000x); verify risk remains bounded in [0.0015, 0.0025]."""
        config = BidirectionalCPPIConfig(max_allowed_dd=0.08, min_risk_pct=0.0015, max_risk_pct=0.0025)
        engine = BidirectionalCPPIEngine(config=config, initial_equity=10000.0)

        for multiplier in [1.5, 2.0, 5.0, 10.0, 100.0, 1000.0]:
            runaway_eq = 10000.0 * multiplier
            allowed, risk, reason = engine.evaluate_order_risk(runaway_eq, 10000.0, "BUY", [])
            assert allowed, f"Rejected at runaway equity={runaway_eq}: {reason}"
            assert engine.peak_equity == pytest.approx(runaway_eq, rel=1e-6)
            assert engine.floor == pytest.approx(0.92 * runaway_eq, rel=1e-6)
            assert engine.cushion == pytest.approx(0.08 * runaway_eq, rel=1e-6)
            assert engine.cushion_ratio == pytest.approx(1.0, rel=1e-6)
            assert risk == pytest.approx(config.max_risk_pct, rel=1e-6)
            assert risk <= 0.0025, f"Risk {risk} exceeded max_risk_pct ceiling"

    def test_continuous_sweep_50_pct_cushion_cap_invariant(self):
        """Fine-grained sweep across 1,000 equity levels: loss on single trade <= 50% cushion."""
        config = BidirectionalCPPIConfig(
            max_allowed_dd=0.08,
            min_risk_pct=0.0015,
            max_risk_pct=0.0025,
            cushion_loss_fraction=0.50,
        )
        engine = BidirectionalCPPIEngine(config=config, initial_equity=10000.0)
        peak = 10000.0
        floor = 9200.0

        # Sweep from 9200.1 to 10000.0 in 1,000 steps
        equities = np.linspace(9200.1, 10000.0, 1000)
        for eq in equities:
            cushion = eq - floor
            allowed, risk, _ = engine.evaluate_order_risk(float(eq), peak, "BUY", [])
            assert allowed
            single_trade_loss = risk * eq
            max_permitted_loss = 0.50 * cushion
            # Floating point tolerance: single_trade_loss must not breach 50% cushion + epsilon
            assert single_trade_loss <= max_permitted_loss + 1e-9, (
                f"Breach at eq={eq}: trade_loss={single_trade_loss} > max_permitted={max_permitted_loss}"
            )
            assert risk <= 0.0025 + 1e-9

    def test_bidirectional_exact_risk_symmetry_grid(self):
        """Prove exact risk symmetry across 100 varied equity states for BUY vs SELL."""
        config = BidirectionalCPPIConfig(max_allowed_dd=0.08)
        engine = BidirectionalCPPIEngine(config=config, initial_equity=10000.0)

        for eq in np.linspace(9210.0, 15000.0, 100):
            allowed_buy, risk_buy, reason_buy = engine.evaluate_order_risk(float(eq), 10000.0, "BUY", [])
            allowed_sell, risk_sell, reason_sell = engine.evaluate_order_risk(float(eq), 10000.0, "SELL", [])
            assert allowed_buy == allowed_sell
            assert risk_buy == pytest.approx(risk_sell, abs=1e-15)
            assert reason_buy == reason_sell

    def test_pathological_inputs_safety(self):
        """Verify engine rejects NaN, Inf, negative, or invalid directions without exceptions."""
        engine = BidirectionalCPPIEngine(initial_equity=10000.0)

        # Invalid direction
        allowed, risk, reason = engine.evaluate_order_risk(10000.0, 10000.0, "HOLD", [])
        assert not allowed and "unknown_direction" in reason

        # NaN equity
        allowed, risk, reason = engine.evaluate_order_risk(float("nan"), 10000.0, "BUY", [])
        assert not allowed and risk == 0.0

        # Inf equity
        allowed, risk, reason = engine.evaluate_order_risk(float("inf"), 10000.0, "BUY", [])
        assert not allowed and risk == 0.0

        # Negative equity
        allowed, risk, reason = engine.evaluate_order_risk(-500.0, 10000.0, "BUY", [])
        assert not allowed and risk == 0.0


# ============================================================================
# Section 2: Concurrency Flood & Position Schema Evasion Stress
# ============================================================================

class TestAdversarialConcurrencyGovernance:
    """Stress-test concurrency bounds, position flood, and schema manipulation."""

    def test_concurrency_flood_alternating_orders(self):
        """Flood engine with 100 alternating BUY/SELL orders; verify max 5 total, max 2 directional."""
        config = BidirectionalCPPIConfig(
            max_concurrent_positions=5,
            max_positions_per_direction=2,
        )
        engine = BidirectionalCPPIEngine(config=config, initial_equity=10000.0)
        open_positions: List[Dict[str, Any]] = []

        # Order 1: BUY -> Allowed (1 BUY, 0 SELL, 1 Total)
        allowed, _, _ = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert allowed
        open_positions.append({"side": "BUY", "ticket": 1})

        # Order 2: BUY -> Allowed (2 BUY, 0 SELL, 2 Total)
        allowed, _, _ = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert allowed
        open_positions.append({"side": "BUY", "ticket": 2})

        # Order 3: BUY -> REJECTED (max 2 BUY reached)
        allowed, _, reason = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert not allowed
        assert reason == "max_directional_buys_reached"

        # Order 4: SELL -> Allowed (2 BUY, 1 SELL, 3 Total)
        allowed, _, _ = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
        assert allowed
        open_positions.append({"side": "SELL", "ticket": 3})

        # Order 5: SELL -> Allowed (2 BUY, 2 SELL, 4 Total)
        allowed, _, _ = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
        assert allowed
        open_positions.append({"side": "SELL", "ticket": 4})

        # Order 6: SELL -> REJECTED (max 2 SELL reached)
        allowed, _, reason = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
        assert not allowed
        assert reason == "max_directional_sells_reached"

        # Order 7: BUY -> REJECTED (max 2 BUY reached)
        allowed, _, reason = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert not allowed
        assert reason == "max_directional_buys_reached"

        # Add a 5th neutral position to test global limit of 5
        open_positions.append({"type": "UNKNOWN", "ticket": 5})
        assert len(open_positions) == 5

        # Order 8: Global cap reached -> REJECTED for any direction
        allowed, _, reason = engine.evaluate_order_risk(10000.0, 10000.0, "BUY", open_positions)
        assert not allowed
        assert reason == "max_concurrent_positions_reached"

        allowed, _, reason = engine.evaluate_order_risk(10000.0, 10000.0, "SELL", open_positions)
        assert not allowed
        assert reason == "max_concurrent_positions_reached"

    def test_adversarial_position_schema_evasion(self):
        """Attempt to evade directional classification with subtle key variations."""
        engine = BidirectionalCPPIEngine()

        # All of these must classify as BUY (+1)
        buy_variants = [
            {"side": 1},
            {"side": "BUY"},
            {"side": "LONG"},
            {"side": "+1"},
            {"direction": 1},
            {"direction": "BUY"},
            {"direction": "LONG"},
            {"is_long": True},
            {"type": "BUY"},
            {"type": "LONG"},
            {"type": 0},  # MT5 POSITION_TYPE_BUY
        ]
        for v in buy_variants:
            cls_dir = engine._classify_position_direction(v)
            assert cls_dir == 1, f"Failed to classify BUY variant: {v}"

        # All of these must classify as SELL (-1)
        sell_variants = [
            {"side": -1},
            {"side": "SELL"},
            {"side": "SHORT"},
            {"side": "-1"},
            {"direction": -1},
            {"direction": "SELL"},
            {"direction": "SHORT"},
            {"is_long": False},
            {"type": "SELL"},
            {"type": "SHORT"},
            {"type": 1},  # MT5 POSITION_TYPE_SELL
        ]
        for v in sell_variants:
            cls_dir = engine._classify_position_direction(v)
            assert cls_dir == -1, f"Failed to classify SELL variant: {v}"

    def test_micro_scalp_concurrency_flood_on_router(self):
        """Verify that micro-scalp engine cannot bypass concurrency limits in DualChampionRouter."""
        mock_traces = [
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
            for _ in range(5)
        ]
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=0,
            bidirectional_cppi_config=BidirectionalCPPIConfig(
                max_concurrent_positions=5,
                max_positions_per_direction=2,
            ),
        )
        router.bull_policy.act_nam_tac_nhan.return_value = mock_traces
        router.bear_policy.act_nam_tac_nhan.return_value = mock_traces

        # Mock micro scalp to aggressively propose BUY entries
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

        # 5 open positions -> Micro-scalp and Champion entries must be completely blocked
        positions_5 = [
            {"side": 1, "ticket": 1},
            {"side": 1, "ticket": 2},
            {"side": -1, "ticket": 3},
            {"side": -1, "ticket": 4},
            {"type": 0, "ticket": 5, "side": 1},
        ]
        batch = router.act_nam_tac_nhan(
            np.zeros(137),
            equity=10000.0,
            positions=positions_5,
        )

        # Agent 0 and Agent 1 MUST be HOLD (0)
        act_0 = DualChampionRouter._extract_action_idx(batch[0])
        act_1 = DualChampionRouter._extract_action_idx(batch[1])
        assert act_0 == 0, f"Micro-scalp bypassed 5-position cap! Action={act_0}"
        assert act_1 == 0, f"Agent 1 emitted non-zero action {act_1}"

    def test_router_positions_string_side_vulnerability(self):
        """Empirically prove vulnerability: DualChampionRouter crashes if positions have string 'BUY'/'SELL'."""
        mock_traces = [
            MagicMock(selected_action=np.zeros(20), action_index=0, action_details={"action": 0})
            for _ in range(5)
        ]
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
        )
        router.bull_policy.act_nam_tac_nhan.return_value = mock_traces
        router.bear_policy.act_nam_tac_nhan.return_value = mock_traces

        # While BidirectionalCPPIEngine correctly classifies string 'BUY'/'SELL',
        # DualChampionRouter line 1319 uses int(pos.get("side", 1)) which raises ValueError.
        positions_with_string_side = [{"side": "BUY", "ticket": 1}]
        with pytest.raises(ValueError, match="invalid literal for int"):
            router.act_nam_tac_nhan(
                np.zeros(137),
                equity=10000.0,
                positions=positions_with_string_side,
            )


# ============================================================================
# Section 3: Dynamic Breakeven Ratchet Stress & Flash Reversal
# ============================================================================

class TestAdversarialDynamicBreakevenRatchet:
    """Stress-test breakeven ratchet trigger boundaries, monotonicity, and flash reversal."""

    def test_breakeven_ratchet_precision_boundaries_buy(self):
        """BUY: Excursion at +1.99 ATR does not ratchet; +2.00 ATR locks +0.5 ATR; flash reversal stops out."""
        config = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            initial_sl_atr=5.0,
        )
        engine = TwoTierTrailingRunnerEngine(config=config)
        entry_price = 60000.0
        atr = 100.0

        # Initial bar: Entry (evaluate_position signature: pid, entry_price, current_price, direction, entry_atr, high, low)
        engine.evaluate_position("pos_buy", entry_price=entry_price, current_price=60000.0, direction=1, entry_atr=atr)
        state = engine._positions["pos_buy"]
        initial_sl = state.current_sl
        assert initial_sl == 60000.0 - 5.0 * 100.0  # 59500.0

        # Excursion to +1.99 ATR: High = 60199.0 -> No ratchet!
        engine.evaluate_position("pos_buy", entry_price=entry_price, current_price=60199.0, direction=1, entry_atr=atr, high=60199.0, low=60150.0)
        assert not state.breakeven_executed
        assert state.current_sl == initial_sl

        # Excursion to +2.00 ATR: High = 60200.0 -> Ratchet triggers, locking +0.5 ATR (60050.0)
        res_trig = engine.evaluate_position("pos_buy", entry_price=entry_price, current_price=60200.0, direction=1, entry_atr=atr, high=60200.0, low=60150.0)
        assert state.breakeven_executed
        assert state.current_sl == 60050.0  # entry + 0.5 * 100.0
        assert res_trig["mode"] == MANAGEMENT_MODE_BREAKEVEN

        # Flash reversal to +0.49 ATR: Low drops to 60049.0 -> Stopped out with profit!
        res_flash = engine.evaluate_position("pos_buy", entry_price=entry_price, current_price=60049.0, direction=1, entry_atr=atr, high=60100.0, low=60049.0)
        assert res_flash["stopped_out"]
        assert state.current_sl == 60050.0
        assert state.current_sl > entry_price

    def test_breakeven_ratchet_precision_boundaries_sell(self):
        """SELL: Excursion at -1.99 ATR does not ratchet; -2.00 ATR locks -0.5 ATR; flash reversal stops out."""
        config = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            initial_sl_atr=5.0,
        )
        engine = TwoTierTrailingRunnerEngine(config=config)
        entry_price = 60000.0
        atr = 100.0

        # Initial bar: Entry
        engine.evaluate_position("pos_sell", entry_price=entry_price, current_price=60000.0, direction=-1, entry_atr=atr)
        state = engine._positions["pos_sell"]
        initial_sl = state.current_sl
        assert initial_sl == 60000.0 + 5.0 * 100.0  # 60500.0

        # Excursion to -1.99 ATR: Low = 59801.0 -> No ratchet!
        engine.evaluate_position("pos_sell", entry_price=entry_price, current_price=59801.0, direction=-1, entry_atr=atr, high=59850.0, low=59801.0)
        assert not state.breakeven_executed
        assert state.current_sl == initial_sl

        # Excursion to -2.00 ATR: Low = 59800.0 -> Ratchet triggers, locking -0.5 ATR (59950.0)
        res_trig = engine.evaluate_position("pos_sell", entry_price=entry_price, current_price=59800.0, direction=-1, entry_atr=atr, high=59850.0, low=59800.0)
        assert state.breakeven_executed
        assert state.current_sl == 59950.0  # entry - 0.5 * 100.0
        assert res_trig["mode"] == MANAGEMENT_MODE_BREAKEVEN

        # Flash reversal to -0.49 ATR: High spikes to 59951.0 -> Stopped out with profit!
        res_flash = engine.evaluate_position("pos_sell", entry_price=entry_price, current_price=59951.0, direction=-1, entry_atr=atr, high=59951.0, low=59900.0)
        assert res_flash["stopped_out"]
        assert state.current_sl == 59950.0
        assert state.current_sl < entry_price

    def test_ratchet_monotonicity_under_random_walk(self):
        """Simulate 200 bars of volatile price steps after ratchet; verify SL NEVER loosens."""
        config = TwoTierConfig(breakeven_trigger_atr=2.0, breakeven_lock_atr=0.5)
        engine = TwoTierTrailingRunnerEngine(config=config)
        entry = 50000.0
        atr = 100.0

        # Trigger ratchet: High = 50250.0 (+2.5 ATR)
        engine.evaluate_position("walk_buy", entry_price=entry, current_price=50000.0, direction=1, entry_atr=atr)
        engine.evaluate_position("walk_buy", entry_price=entry, current_price=50250.0, direction=1, entry_atr=atr, high=50250.0, low=50100.0)
        state = engine._positions["walk_buy"]
        assert state.breakeven_executed
        prev_sl = state.current_sl
        assert prev_sl == 50050.0

        # 200 random fluctuating bars above SL
        rng = np.random.default_rng(42)
        for _ in range(200):
            high = 50250.0 + rng.uniform(0.0, 500.0)
            low = 50060.0 + rng.uniform(0.0, 100.0)  # stays above SL
            cur = (high + low) / 2.0
            engine.evaluate_position("walk_buy", entry_price=entry, current_price=cur, direction=1, entry_atr=atr, high=high, low=low)
            current_sl = state.current_sl
            assert current_sl >= prev_sl, f"SL loosened from {prev_sl} to {current_sl}"
            prev_sl = current_sl

    def test_pathological_atr_handling_in_runner(self):
        """Verify runner fails gracefully on non-finite or zero ATR."""
        engine = TwoTierTrailingRunnerEngine()
        # ATR = 0.0
        res_zero = engine.evaluate_position("pos_zero_atr", entry_price=50000.0, current_price=50200.0, direction=1, entry_atr=0.0, high=50200.0, low=50000.0)
        assert res_zero["action"] == ACTION_HOLD
        assert not res_zero["stopped_out"]

        # ATR = NaN
        res_nan = engine.evaluate_position("pos_nan_atr", entry_price=50000.0, current_price=50200.0, direction=1, entry_atr=float("nan"), high=50200.0, low=50000.0)
        assert res_nan["action"] == ACTION_HOLD


# ============================================================================
# Section 4: Agent 4 Governor Invariant Stress & Corruption Resistance
# ============================================================================

class TestAdversarialAgent4GovernorInvariance:
    """Adversarially probe Agent 4 Governor invariant (100% HOLD, POLICY_CLOSE_ALL == 0.0)."""

    def test_agent4_adversarial_logits_corruption(self):
        """Attempt to force Agent 4 to emit LIQUIDATE_ALL (action 10) or other illegal actions."""
        # Use lightweight mock router
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
        )

        # Malicious trace object mimicking an agent attempting to liquidate
        class MaliciousTrace:
            def __init__(self):
                # 12 action logits with action 10 (LIQUIDATE_ALL) set to +999.0
                logits = np.zeros(12, dtype=np.float32)
                logits[10] = 999.0
                logits[0] = -50.0
                self.output_vector = logits
                self.action_index = 10
                self.selected_idx = 10
                self.action_details = {
                    "action": 10,
                    "policy_close_all": 1.0,
                    "five_action_index": 10,
                }

        malicious_agent4 = MaliciousTrace()
        sanitized = router._enforce_agent4_hold(malicious_agent4)

        # Must be completely sanitized to ACTION_HOLD = 0 and POLICY_CLOSE_ALL = 0.0
        assert getattr(sanitized, "action_index", None) == 0
        assert getattr(sanitized, "selected_idx", None) == 0
        details = getattr(sanitized, "action_details", {})
        assert details.get("policy_close_all") == 0.0
        assert details.get("five_action_index") == 0
        assert details.get("action") == 0

        # Vector logits: action 0 forced to 1.0, action 10 suppressed to -100.0
        vec = getattr(sanitized, "output_vector", None)
        assert vec[0] == 1.0
        assert vec[10] == -100.0

    def test_agent4_router_e2e_invariant(self):
        """Verify that across 50 router evaluations, Agent 4 action is ALWAYS 0 and policy_close_all == 0.0."""
        # Test with mock policies returning diverse actions
        router = DualChampionRouter(
            bull_policy=MagicMock(),
            bear_policy=MagicMock(),
        )
        rng = np.random.default_rng(123)

        for step in range(50):
            # Bull and bear mock traces with varied actions
            mock_traces = [
                MagicMock(selected_action=np.zeros(20), action_index=rng.integers(0, 7), action_details={"action": 0})
                for _ in range(5)
            ]
            # Deliberately make Agent 4 propose action 10 in the mock
            mock_traces[4] = MagicMock(
                selected_action=np.zeros(20),
                action_index=10,
                action_details={"action": 10, "policy_close_all": 1.0},
            )
            router.bull_policy.act_nam_tac_nhan.return_value = mock_traces
            router.bear_policy.act_nam_tac_nhan.return_value = mock_traces

            obs = rng.standard_normal(137).astype(np.float32)
            traces = router.act_nam_tac_nhan(obs, equity=10000.0, positions=[])
            agent4_trace = traces[4]
            act4 = DualChampionRouter._extract_action_idx(agent4_trace)
            assert act4 == 0, f"Agent 4 emitted non-zero action {act4} at step {step}"
            details = getattr(agent4_trace, "action_details", {})
            if isinstance(details, dict):
                assert details.get("policy_close_all", 0.0) == 0.0


# ============================================================================
# Section 5: High-Speed Burst Latency Benchmark (1,000 evaluations)
# ============================================================================

class TestHighSpeedBurstLatency:
    """Empirically measure latency SLA compliance of Bidirectional CPPI Engine."""

    def test_burst_latency_sla(self):
        """1,000 evaluations of evaluate_order_risk must average < 0.05ms per call."""
        engine = BidirectionalCPPIEngine(initial_equity=10000.0)
        positions = [
            {"side": "BUY", "ticket": 1},
            {"side": "SELL", "ticket": 2},
        ]

        t0 = time.perf_counter()
        n_evals = 1000
        for i in range(n_evals):
            eq = 10000.0 + (i % 500)
            engine.evaluate_order_risk(eq, 11000.0, "BUY" if i % 2 == 0 else "SELL", positions)
        t1 = time.perf_counter()

        avg_latency_ms = ((t1 - t0) / n_evals) * 1000.0
        assert avg_latency_ms < 0.05, f"Average latency {avg_latency_ms:.4f}ms exceeded SLA limit 0.05ms"
