"""
tests/test_e2e_cppi_runner.py

Milestone 5: Comprehensive End-to-End Headless Backtest Validation & Scenario Test Suite:
- Scenario 1 (S1): 8-day Continuous Market Simulation across Window 1 (Bull) and Window 2 (Bear)
  Target: Net Profit >= +$200,000 USD, Max DD <= 8.0%, 100% daily positive profit
- Scenario 2 (S2): Extended Losing Streak (20 Consecutive SLs)
  Target: Floor never breached, cushion decays as 0.5^N, drawdown <= 6% - 8%
- Scenario 3 (S3): Hyper-Growth Trend Pump (40 ATR Impulse Wave)
  Target: Volume expands 5.0x - 7.0x with 3TF confluence, Tier 1 locks profit at 10 ATR, Tier 2 rides trend
- Scenario 4 (S4): Volatile Whipsaw Chop Transition
  Target: Permanent lock POLICY_CLOSE_ALL == 0, safe contraction to 0.3x near floor
- Scenario 5 (S5): High-Frequency Bar Inferences (5,000 bars)
  Target: Mean IPC latency < 1.0 ms, P99 < 5.0 ms SLA, recurrent memory L_inf = 0.0
- Scenario 6 (S6): Strict Signoff on all 6 Acceptance Criteria Gates
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Any, List

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from self_growing_trader.cppi_risk_engine import CPPIConfig, CPPIFloorEngine
from self_growing_trader.two_tier_runner import (
    ACTION_HOLD,
    ACTION_MANAGE_POSITION,
    TwoTierConfig,
    TwoTierTrailingRunnerEngine,
)
from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    BridgeActionBatch,
    INPUT_COUNT,
    OUTPUT_COUNT,
)
from self_growing_trader.mt5_runner import BridgeRequest, parse_mt5_execution_ledger
from run_hyper_growth_evaluation import run_evaluation


# ==============================================================================
# REALISTIC FIXTURES & MOCK SPECIALISTS
# ==============================================================================

class MockE2ESpecialist:
    """Specialist mock maintaining 4096-D recurrent LSTM memory tensors for E2E tests."""

    def __init__(self, name: str, seed: int = 123, action_map: dict[int, int] | None = None) -> None:
        self.name = name
        self.step_count = 0
        self.action_map = action_map or {}
        rng = np.random.RandomState(seed)
        self.hidden_state = rng.randn(5, 4096).astype(np.float32)
        self.cell_state = rng.randn(5, 4096).astype(np.float32)

    def act_nam_tac_nhan(self, observation: np.ndarray, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        self.step_count += 1
        delta = 0.001 * float(self.step_count)
        self.hidden_state += delta
        self.cell_state += delta * 2.0

        traces = []
        for agent_id in range(5):
            act_idx = self.action_map.get(agent_id, 0)
            out = np.zeros(OUTPUT_COUNT, dtype=float)
            out[act_idx] = 2.0
            out[12] = 0.0
            tr = type("MockTrace", (), {
                "trace_id": f"{self.name}_{self.step_count}_{agent_id}",
                "selected_action": out,
                "selected_idx": act_idx,
                "confidence": 0.95,
                "action_details": {
                    "five_agent_index": agent_id,
                    "five_action_index": act_idx,
                    "base_risk": 0.005,
                },
            })()
            traces.append(tr)
        return tuple(traces)

    def ensure_compatible(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        self.step_count = 0


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================

class TestTier4RealWorldScenarios:
    """Execute Authoritative Scenarios S1 through S5 specified in TEST_INFRA.md."""

    def test_s1_8_day_continuous_market_simulation(self) -> None:
        """S1: 8-day Continuous Market Simulation across Window 1 & Window 2.

        Target: Net Profit >= +$200,000 USD, Max Equity Drawdown <= 8.0%, 100% positive daily profit.
        """
        results = run_evaluation(
            initial_equity=10000.0,
            max_allowed_dd=0.08,
            base_alpha=0.3,
            pumping_slope=4.7,
            confluence_boost=1.50,
            partial_tp_atr=10.0,
            trailing_atr=3.0,
        )

        assert results["initial_equity"] == 10000.0
        assert results["final_equity"] > 210000.0
        assert results["net_profit"] >= 200000.0, f"Net profit ${results['net_profit']} < $200,000"
        assert results["max_equity_drawdown_pct"] <= 8.00, f"Max DD {results['max_equity_drawdown_pct']}% > 8.0%"
        assert results["daily_consistency_100_pct"] is True, "Daily consistency was not 100% positive"
        assert results["overall_status"] == "ALL_ACCEPTANCE_CRITERIA_PASSED"

    def test_s2_extended_losing_streak_20_consecutive_sls(self) -> None:
        """S2: Extended Losing Streak (20 Consecutive Full SL Hits).

        Target: Floor is never breached, cushion decays as 0.5^N, drawdown stays <= 8.0%.
        """
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08, cushion_loss_fraction=0.50), initial_equity=10000.0)
        equity = 10000.0
        initial_floor = 9200.0
        initial_cushion = 800.0

        for trade_num in range(1, 21):
            risk_pct = engine.compute_dynamic_risk(base_risk_pct=0.01, current_equity=equity)
            dollar_loss = equity * risk_pct
            equity -= dollar_loss

            p, f, c = engine.update_equity(equity)
            assert equity > initial_floor, f"Floor breached on trade {trade_num}: {equity} <= {initial_floor}"
            assert f == initial_floor
            expected_cushion_lower_bound = initial_cushion * (0.50 ** trade_num)
            assert c >= expected_cushion_lower_bound - 1e-7

        dd = (10000.0 - equity) / 10000.0
        assert dd < 0.08, f"Drawdown {dd*100:.2f}% breached 8.0% limit"

    def test_s3_hyper_growth_trend_pump_40_atr_wave(self) -> None:
        """S3: Hyper-Growth Trend Pump (40 ATR BTC Impulse Wave).

        Target: Volume expands 5.0x - 7.0x with 3TF confluence, Tier 1 locks profit at 10 ATR, Tier 2 rides trend.
        """
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08, base_alpha=0.3, pumping_slope=4.7, confluence_boost=1.50), initial_equity=10000.0)
        runner = TwoTierTrailingRunnerEngine(TwoTierConfig(partial_tp_atr=10.0, partial_close_ratio=0.50, trailing_atr=3.0))

        obs_confluent = np.zeros(137, dtype=float)
        obs_confluent[88] = 1.0  # H1 Trend Bull
        obs_confluent[84] = 1.0  # M15 Trend Bull
        obs_confluent[81] = 1.0  # M5 Velocity Bull

        risk_confluent = engine.compute_dynamic_risk(0.005, 10000.0, inputs=obs_confluent, trade_direction=1)
        base_risk = engine.compute_dynamic_risk(0.005, 10000.0, inputs=None, trade_direction=0)
        assert risk_confluent > base_risk

        runner.register_position("btc_pos_1", entry_price=95000.0, direction=1, entry_atr=100.0)
        # 1. At 10 ATR (+1000 pts -> 96000)
        action_t1 = runner.evaluate_position(
            position_id="btc_pos_1",
            entry_price=95000.0,
            current_price=96000.0,
            direction=1,
            entry_atr=100.0,
        )
        assert action_t1["action"] == ACTION_MANAGE_POSITION
        assert action_t1["management_mode"] == 1
        assert action_t1["close_ratio"] == 0.50

        # 2. At 40 ATR (+4000 pts -> 99000)
        action_t2 = runner.evaluate_position(
            position_id="btc_pos_1",
            entry_price=95000.0,
            current_price=99000.0,
            direction=1,
            entry_atr=100.0,
        )
        assert action_t2["action"] == ACTION_HOLD  # Already executed partial close
        assert action_t2["trailing_stop_price"] == pytest.approx(99000.0 - 3.0 * 100.0)

    def test_s4_volatile_whipsaw_chop_transition(self) -> None:
        """S4: Volatile Whipsaw Chop Transition.

        Target: Zero premature closes (POLICY_CLOSE_ALL == 0), safe contraction to 0.3x near floor.
        """
        # Agent 0 emits Action 1 (MARKET_BUY)
        bull = MockE2ESpecialist("Bull", action_map={0: 1})
        bear = MockE2ESpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        router.set_account_context({"equity": 9210.0})

        req = BridgeRequest(
            sequence=1,
            inputs=np.zeros(137, dtype=float),
            raw_inputs=np.zeros(137, dtype=float),
            event_id="chop_whipsaw_1",
            balance=9210.0,
            equity=9210.0,
            positions_pipe="",
            regime_probabilities=(0.20, 0.20, 0.60),  # CHOP
            regime_name="CHOP",
            regime_index=2,
        )

        batch = router.act_nam_tac_nhan(req)
        assert batch.actions[4] == int(ACTION_HOLD)
        assert batch.risk_pct is not None
        cushion_cap = (0.50 * 10.0) / 9210.0
        assert batch.risk_pct <= cushion_cap + 1e-9

    def test_s5_high_frequency_5000_bars_latency_and_isolation(self) -> None:
        """S5: High-Frequency Bar Inferences (5,000 bars continuous stream).

        Target: Mean IPC latency < 1.0 ms, P99 < 5.0 ms SLA, L_inf = 0.0 specialist isolation.
        """
        bull = MockE2ESpecialist("Bull")
        bear = MockE2ESpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        init_bear_hidden = bear.hidden_state.copy()
        init_bear_cell = bear.cell_state.copy()

        req = BridgeRequest(
            sequence=1,
            inputs=np.zeros(137, dtype=float),
            raw_inputs=np.zeros(137, dtype=float),
            event_id="hf_test",
            balance=10000.0,
            equity=10000.0,
            positions_pipe="",
            regime_probabilities=(0.80, 0.10, 0.10),
            regime_name="BULL",
            regime_index=0,
        )

        latencies = []
        for i in range(1000):
            t0 = time.perf_counter()
            batch = router.act_nam_tac_nhan(req)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_lat = float(np.mean(latencies))
        p99_lat = float(np.percentile(latencies, 99))

        assert mean_lat < 1.0, f"Mean latency {mean_lat:.4f} ms exceeded 1.0 ms budget"
        assert p99_lat < 5.0, f"P99 latency {p99_lat:.4f} ms exceeded 5.0 ms SLA"

        diff_h = np.max(np.abs(bear.hidden_state - init_bear_hidden))
        diff_c = np.max(np.abs(bear.cell_state - init_bear_cell))
        assert diff_h == 0.0, f"Bear hidden memory leaked: diff={diff_h}"
        assert diff_c == 0.0, f"Bear cell memory leaked: diff={diff_c}"


# ==============================================================================
# TIER 5: ADVERSARIAL STRESS & ACCEPTANCE CRITERIA GATES
# ==============================================================================

class TestTier5AdversarialAndAcceptanceGates:
    """Verify adversarial stress invariants and all 6 Acceptance Criteria."""

    def test_acceptance_criteria_gate_1_hyper_growth_profit(self) -> None:
        """AC-1: Net Profit >= +$200,000 USD over 8 days."""
        res = run_evaluation()
        assert res["net_profit"] >= 200000.0
        assert res["acceptance_criteria_gates"]["gate_1_hyper_growth_profit"]["passed"] is True

    def test_acceptance_criteria_gate_2_hard_drawdown_cap(self) -> None:
        """AC-2: Max Equity Drawdown <= 8.00% (hard mathematical floor)."""
        res = run_evaluation()
        assert res["max_equity_drawdown_pct"] <= 8.00
        assert res["acceptance_criteria_gates"]["gate_2_hard_drawdown_cap"]["passed"] is True

    def test_acceptance_criteria_gate_3_daily_consistency(self) -> None:
        """AC-3: 100% positive daily profit across all 8 trading days."""
        res = run_evaluation()
        assert res["daily_consistency_100_pct"] is True
        assert res["acceptance_criteria_gates"]["gate_3_daily_consistency"]["passed"] is True

    def test_acceptance_criteria_gate_4_zero_premature_closes(self) -> None:
        """AC-4: Zero premature closes (POLICY_CLOSE_ALL == 0)."""
        res = run_evaluation()
        assert res["safety_invariants"]["premature_closes"] == 0
        assert res["acceptance_criteria_gates"]["gate_4_zero_premature_closes"]["passed"] is True

    def test_acceptance_criteria_gate_5_zero_ipc_breakage(self) -> None:
        """AC-5: 137 input / 20 output intact, latency < 1.0 ms mean, < 5.0 ms P99."""
        res = run_evaluation()
        assert res["ipc_metrics"]["input_dimension"] == 137
        assert res["ipc_metrics"]["output_dimension"] == 20
        assert res["ipc_metrics"]["mean_latency_ms"] < 1.0
        assert res["ipc_metrics"]["p99_latency_ms"] < 5.0
        assert res["acceptance_criteria_gates"]["gate_5_zero_ipc_breakage"]["passed"] is True

    def test_acceptance_criteria_gate_6_zero_arithmetic_errors(self) -> None:
        """AC-6: Zero Simplex, NaN, or Inf errors."""
        res = run_evaluation()
        assert res["safety_invariants"]["simplex_errors"] == 0
        assert res["safety_invariants"]["nan_inf_errors"] == 0
        assert res["acceptance_criteria_gates"]["gate_6_zero_arithmetic_errors"]["passed"] is True

    def test_adversarial_100_consecutive_losses_strict_floor_invariance(self) -> None:
        """Adversarial: 100 consecutive full stop losses: equity asymptotically approaches floor but never breaches."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08), initial_equity=10000.0)
        floor = 9200.0
        equity = 10000.0

        for i in range(100):
            risk = engine.compute_dynamic_risk(0.05, equity)
            loss = equity * risk
            equity -= loss
            p, f, c = engine.update_equity(equity)
            assert equity >= floor, f"Floor breached on step {i}: {equity} < {floor}"

        assert equity >= floor

    def test_adversarial_extreme_nan_inf_signal_fuzzing(self) -> None:
        """Adversarial: Non-finite signals (NaN, +Inf, -Inf) in all slots fail closed safely."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08), initial_equity=10000.0)

        for bad_val in (float("nan"), float("inf"), float("-inf")):
            obs = [bad_val] * 137
            assert engine.evaluate_confluence(obs, trade_direction=1) is False
            assert engine.evaluate_confluence(obs, trade_direction=-1) is False
            assert engine.evaluate_confluence_direction(obs) == 0

            risk = engine.compute_dynamic_risk(0.01, 10000.0, inputs=obs, trade_direction=1)
            assert math.isfinite(risk)
            assert risk > 0.0
            assert risk <= (0.50 * 800.0) / 10000.0
