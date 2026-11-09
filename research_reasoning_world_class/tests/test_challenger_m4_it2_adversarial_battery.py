"""
Milestone 4 Iteration 2 Challenger 1: Adversarial Empirical Battery
Agent: teamwork_challenger_m4_it2_1
Role: critic, specialist

Rigorous adversarial stress testing of:
1. SlicedM1Environment Boundary Edges & Out-of-Range Index Fuzzing:
   - simulate_trade at bar_idx == total_bars - 1, total_bars - 2, 0, -1, -100, total_bars, total_bars + 1000.
   - Strict verification that invalid indices raise IndexError cleanly.
   - Proof that trades initiated at total_bars - 1 never access master data past start_idx + total_bars - 1.
   - Coordinate space invariance: exit_bar_idx is strictly within [0, total_bars) for all valid executions.
2. OOS Metrics Calculation Stress Testing (OOSEvaluationEngine.calculate_metrics):
   - Edge-case trade distributions: zero trades, 100% wins, 100% losses, 100% breakeven, alternating wins/losses.
   - Extreme profit factors (astronomical 1e9, microscopic 1e-10, infinite profit factor with zero loss).
   - Deep drawdown, catastrophic account wipeout, negative final equity.
   - Verification of ZERO ZeroDivisionError, ZERO NaN, ZERO unhandled Inf crashes, clean formatting in print_report.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
    ExitType,
)
from scripts.train_mythos_muzero_coevolution import (
    SlicedM1Environment,
    OOSEvaluationEngine,
    OOSConfig,
    OOSTradeRecord,
    OOSMetrics,
)


@pytest.fixture(scope="module")
def loaded_master_env():
    """Loads authentic master RealM1Environment from BTCUSDm.csv for empirical stress testing."""
    csv_path = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
    if not Path(csv_path).exists():
        pytest.skip(f"Master CSV not found at {csv_path}")
    env = RealM1Environment(csv_path=csv_path)
    env.load_data("2025.01.15", "2025.02.28")
    return env


# =============================================================================
# SUITE 1: ADVERSARIAL BOUNDARY & OUT-OF-RANGE FUZZING ON SlicedM1Environment
# =============================================================================
class TestSlicedM1EnvironmentAdversarialBoundaries:
    """
    Adversarial fuzz testing of SlicedM1Environment at boundary edges and invalid index domains.
    """

    def test_out_of_range_indices_raise_index_error_cleanly(self, loaded_master_env):
        """
        Query simulate_trade with out-of-range indices:
        -1, -100, total_bars, total_bars + 1, total_bars + 1000, 2**31 - 1.
        Must raise clean IndexError without side effects or unhandled exceptions.
        """
        master = loaded_master_env
        start_idx = 40316
        end_idx = 60465
        sliced = SlicedM1Environment(master, start_idx, end_idx, name="Cycle 1 OOS Slice")
        total = sliced.total_bars
        assert total == end_idx - start_idx == 20149

        invalid_indices = [-1, -2, -100, -999999, total, total + 1, total + 1000, 2**31 - 1]
        for bad_idx in invalid_indices:
            with pytest.raises(IndexError) as exc_info:
                sliced.simulate_trade(bar_idx=bad_idx, action=1, sl_atr=2.0, tp_atr=4.0)
            assert f"bar_idx {bad_idx} out of range [0, {total})" in str(exc_info.value)

            # Similarly, get_observation must also cleanly reject bad indices
            with pytest.raises(IndexError):
                sliced.get_observation(bad_idx)

    def test_first_bar_boundary_execution(self, loaded_master_env):
        """
        Query simulate_trade at bar_idx == 0 (lower boundary).
        Must execute normally, and exit_bar_idx must be in slice relative coordinates [0, total_bars).
        """
        master = loaded_master_env
        sliced = SlicedM1Environment(master, 1000, 5000, name="Slice Lower Edge")
        res = sliced.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=3.0, max_holding_bars=60)

        assert 0 <= res.exit_bar_idx < sliced.total_bars
        assert res.exit_bar_idx == 0 + res.holding_bars
        # Observation at exit_bar_idx must succeed
        obs = sliced.get_observation(res.exit_bar_idx)
        assert obs.shape == (137,)

    def test_last_bar_terminal_boundary_zero_peeking(self, loaded_master_env):
        """
        Query simulate_trade at bar_idx == total_bars - 1 (terminal boundary edge).
        Verify:
        - safe_max_holding is clamped to 0.
        - Trade result has holding_bars == 0 and exit_reason == 'DATASET_END'.
        - res.exit_bar_idx is exactly total_bars - 1.
        - Master environment is never accessed at start_idx + total_bars or beyond.
        """
        master = loaded_master_env
        start_idx = 10000
        end_idx = 20000
        sliced = SlicedM1Environment(master, start_idx, end_idx, name="Slice Terminal Edge")
        last_idx = sliced.total_bars - 1  # 9999

        # Spy on master simulate_trade call to inspect exact arguments passed
        original_master_simulate = master.simulate_trade
        calls = []

        def spy_simulate_trade(*args, **kwargs):
            calls.append((args, kwargs))
            return original_master_simulate(*args, **kwargs)

        master.simulate_trade = spy_simulate_trade
        try:
            res = sliced.simulate_trade(
                bar_idx=last_idx,
                action=1,
                sl_atr=2.0,
                tp_atr=3.0,
                max_holding_bars=60,
            )
        finally:
            master.simulate_trade = original_master_simulate

        assert len(calls) == 1
        _, kwargs = calls[0]
        # Verify master was called with exactly master_bar = start_idx + last_idx = end_idx - 1
        assert kwargs["bar_idx"] == end_idx - 1
        # Verify max_holding_bars was clamped to exactly 0 (no future bars allowed)
        assert kwargs["max_holding_bars"] == 0

        # Verify trade outcome
        assert res.holding_bars == 0
        assert res.exit_reason == "DATASET_END"
        assert res.exit_bar_idx == last_idx
        assert 0 <= res.exit_bar_idx < sliced.total_bars

        # Sliced observation must succeed at res.exit_bar_idx
        obs = sliced.get_observation(res.exit_bar_idx)
        assert obs.shape == (137,)

    def test_second_to_last_bar_boundary_execution(self, loaded_master_env):
        """
        Query simulate_trade at bar_idx == total_bars - 2.
        Only 1 future bar remains in slice: total_bars - 1 - (total_bars - 2) = 1.
        safe_max_holding must clamp min(60, 1) = 1.
        holding_bars must be <= 1.
        res.exit_bar_idx must be <= total_bars - 1 (never touching total_bars or beyond).
        """
        master = loaded_master_env
        start_idx = 20000
        end_idx = 25000
        sliced = SlicedM1Environment(master, start_idx, end_idx, name="Slice N-2 Edge")
        penultimate_idx = sliced.total_bars - 2

        res = sliced.simulate_trade(
            bar_idx=penultimate_idx,
            action=1,
            sl_atr=2.0,
            tp_atr=3.0,
            max_holding_bars=60,
        )

        assert res.holding_bars in (0, 1)
        assert res.exit_bar_idx <= sliced.total_bars - 1
        assert 0 <= res.exit_bar_idx < sliced.total_bars
        obs = sliced.get_observation(res.exit_bar_idx)
        assert obs.shape == (137,)

    def test_master_data_strict_boundary_isolation_empirical(self, loaded_master_env):
        """
        Empirically prove that trades initiated across all bars in slice [0, total_bars)
        NEVER read master array data at or past start_idx + total_bars.
        We test bar_idx = total_bars - 1, total_bars - 2, total_bars - 5 with large max_holding_bars=100.
        """
        master = loaded_master_env
        start_idx = 30000
        end_idx = 35000
        sliced = SlicedM1Environment(master, start_idx, end_idx, name="Boundary Isolation")
        total = sliced.total_bars

        for offset in [1, 2, 3, 5, 10]:
            b_idx = total - offset
            res = sliced.simulate_trade(
                bar_idx=b_idx,
                action=1,
                sl_atr=1.5,
                tp_atr=4.0,
                max_holding_bars=100,
            )
            # Guaranteed zero future peeking past end_idx:
            # master bar visited: start_idx + res.exit_bar_idx <= end_idx - 1
            master_exit_bar = start_idx + res.exit_bar_idx
            assert master_exit_bar < end_idx, (
                f"Leaked into future! master_exit_bar={master_exit_bar} >= end_idx={end_idx}"
            )
            assert res.exit_bar_idx < total
            assert res.holding_bars <= offset - 1


# =============================================================================
# SUITE 2: ADVERSARIAL STRESS TESTING ON OOS METRICS (calculate_metrics)
# =============================================================================
class TestOOSEvaluationEngineAdversarialMetricsStress:
    """
    Stress-tests calculate_metrics across pathological, edge-case, and adversarial trade distributions.
    Verifies zero ZeroDivisionError, zero NaN, zero crash on infinite profit factor or extreme metrics.
    """

    @pytest.fixture
    def default_config(self):
        return OOSConfig(
            min_win_rate=0.80,
            min_profit_factor=3.0,
            max_drawdown=0.10,
            min_trades=50,
        )

    def _make_dummy_trade(
        self,
        trade_id: int,
        net_pnl_usd: float,
        holding_bars: int = 10,
        equity_after: float = 10_000.0,
    ) -> OOSTradeRecord:
        return OOSTradeRecord(
            trade_id=trade_id,
            entry_bar=trade_id * 10,
            exit_bar=trade_id * 10 + holding_bars,
            action=1,
            champion="BULL",
            entry_price=100_000.0,
            exit_price=100_000.0 + net_pnl_usd,
            sl_atr=2.0,
            tp_atr=4.0,
            exit_type="tp" if net_pnl_usd > 0 else "sl",
            exit_reason="TP_HIT" if net_pnl_usd > 0 else "SL_HIT",
            holding_bars=holding_bars,
            raw_pnl_pts=net_pnl_usd + 2.50,
            spread_paid_usd=2.50,
            net_pnl_usd=net_pnl_usd,
            return_atr=net_pnl_usd / 50.0,
            equity_after=equity_after,
        )

    def test_metrics_zero_trades_distribution(self, default_config):
        """Zero trades must produce zero metrics and clean rejection without division by zero."""
        m = OOSEvaluationEngine.calculate_metrics(
            trades=[],
            initial_equity=10_000.0,
            final_equity=10_000.0,
            cycle_name="Zero Trades Test",
            config=default_config,
        )
        assert m.total_trades == 0
        assert m.winning_trades == 0
        assert m.losing_trades == 0
        assert m.breakeven_trades == 0
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert m.gross_profit == 0.0
        assert m.gross_loss == 0.0
        assert m.net_profit == 0.0
        assert m.max_drawdown == 0.0
        assert not m.gate_passed
        assert "Zero trades executed in OOS window" in m.failure_reasons

    def test_metrics_all_winning_trades_infinite_profit_factor(self, default_config):
        """
        100% winning trades (zero gross loss).
        profit_factor must be float('inf').
        win_rate must be 1.0.
        Must NOT raise OverflowError, ZeroDivisionError, or NaN.
        print_report must safely handle float('inf') without crashing.
        """
        trades = [
            self._make_dummy_trade(i, net_pnl_usd=50.0, equity_after=10_000.0 + (i + 1) * 50.0)
            for i in range(60)
        ]
        m = OOSEvaluationEngine.calculate_metrics(
            trades=trades,
            initial_equity=10_000.0,
            final_equity=13_000.0,
            cycle_name="All Wins Test",
            config=default_config,
        )
        assert m.total_trades == 60
        assert m.winning_trades == 60
        assert m.losing_trades == 0
        assert m.win_rate == 1.0
        assert math.isinf(m.profit_factor)
        assert m.gross_loss == 0.0
        assert m.gross_profit == 3000.0
        assert m.net_profit == 3000.0
        assert m.max_drawdown == 0.0
        assert m.gate_passed  # 60 trades >= 50, WR 100% >= 80%, PF inf >= 3.0, DD 0% <= 10%

        # Smoke check print_report with infinite PF
        OOSEvaluationEngine.print_report(m)

    def test_metrics_all_losing_trades(self, default_config):
        """
        100% losing trades (zero gross profit).
        profit_factor must be 0.0 (not NaN or Inf).
        win_rate must be 0.0.
        Drawdown must reflect cumulative losses accurately.
        """
        trades = [
            self._make_dummy_trade(i, net_pnl_usd=-100.0, equity_after=10_000.0 - (i + 1) * 100.0)
            for i in range(50)
        ]
        m = OOSEvaluationEngine.calculate_metrics(
            trades=trades,
            initial_equity=10_000.0,
            final_equity=5_000.0,
            cycle_name="All Losses Test",
            config=default_config,
        )
        assert m.total_trades == 50
        assert m.winning_trades == 0
        assert m.losing_trades == 50
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert not math.isnan(m.profit_factor)
        assert m.gross_profit == 0.0
        assert m.gross_loss == 5000.0
        assert m.net_profit == -5000.0
        assert pytest.approx(m.max_drawdown, 1e-4) == 0.50
        assert not m.gate_passed
        assert len(m.failure_reasons) >= 3  # WR, PF, DD

    def test_metrics_all_breakeven_trades(self, default_config):
        """
        All trades have net_pnl_usd == 0.0.
        gross_profit == 0.0 and gross_loss == 0.0.
        profit_factor must be 0.0 (not NaN or division error).
        """
        trades = [self._make_dummy_trade(i, net_pnl_usd=0.0) for i in range(50)]
        m = OOSEvaluationEngine.calculate_metrics(
            trades=trades,
            initial_equity=10_000.0,
            final_equity=10_000.0,
            cycle_name="Breakeven Test",
            config=default_config,
        )
        assert m.total_trades == 50
        assert m.breakeven_trades == 50
        assert m.win_rate == 0.0
        assert m.profit_factor == 0.0
        assert not math.isnan(m.profit_factor)
        assert m.max_drawdown == 0.0
        assert not m.gate_passed

    def test_metrics_alternating_wins_and_losses_drawdown_tracking(self, default_config):
        """
        Alternating trades: +1000, -2000, +1500, -500, +3000...
        Checks that running_peak, equity trajectory, and drawdown are rigorously exact.
        """
        pnls = [1000.0, -2000.0, 1500.0, -500.0, 3000.0, -100.0]
        curr_eq = 10_000.0
        trades = []
        for i, pnl in enumerate(pnls):
            curr_eq += pnl
            trades.append(self._make_dummy_trade(i, net_pnl_usd=pnl, equity_after=curr_eq))

        m = OOSEvaluationEngine.calculate_metrics(
            trades=trades,
            initial_equity=10_000.0,
            final_equity=curr_eq,
            cycle_name="Alternating PnL Test",
            config=default_config,
        )
        # Trajectory:
        # start: 10,000, peak: 10,000
        # +1000 -> 11,000, peak: 11,000, dd: 0
        # -2000 -> 9,000, peak: 11,000, dd: 2000 / 11000 = 18.1818%
        # +1500 -> 10,500, peak: 11,000, dd: 500 / 11000
        # -500 -> 10,000, peak: 11,000, dd: 1000 / 11000
        # +3000 -> 13,000, peak: 13,000, dd: 0
        # -100 -> 12,900, peak: 13,000, dd: 100 / 13000
        expected_max_dd_pct = 2000.0 / 11000.0
        assert pytest.approx(m.max_drawdown, 1e-4) == expected_max_dd_pct
        assert pytest.approx(m.max_drawdown_usd, 1e-4) == 2000.0
        assert m.peak_equity == 13000.0
        assert m.final_equity == 12900.0

    def test_metrics_extreme_profit_factors_and_denormals(self, default_config):
        """
        Adversarial fuzzing with astronomical profits and microscopic losses,
        as well as near-zero floating point numbers.
        """
        # Astronomical PF: win = 1e9, loss = 1.0
        t1 = [
            self._make_dummy_trade(0, net_pnl_usd=1e9),
            self._make_dummy_trade(1, net_pnl_usd=-1.0),
        ]
        m1 = OOSEvaluationEngine.calculate_metrics(
            trades=t1,
            initial_equity=10_000.0,
            final_equity=10_000.0 + 1e9 - 1.0,
            cycle_name="Astronomical PF",
            config=default_config,
        )
        assert pytest.approx(m1.profit_factor, 1e-2) == 1e9
        assert not math.isnan(m1.profit_factor)

        # Microscopic PF: win = 1e-9, loss = 1e6
        t2 = [
            self._make_dummy_trade(0, net_pnl_usd=1e-9),
            self._make_dummy_trade(1, net_pnl_usd=-1e6),
        ]
        m2 = OOSEvaluationEngine.calculate_metrics(
            trades=t2,
            initial_equity=10_000.0,
            final_equity=10_000.0 + 1e-9 - 1e6,
            cycle_name="Microscopic PF",
            config=default_config,
        )
        assert pytest.approx(m2.profit_factor, 1e-15) == 1e-15
        assert not math.isnan(m2.profit_factor)

    def test_metrics_catastrophic_bankruptcy_drawdown_denominator_safety(self, default_config):
        """
        Account suffers catastrophic loss wiping out 100% of capital and entering negative equity.
        Formula dd_pct = dd_usd / max(running_peak, 1.0) must never divide by 0 or negative.
        """
        trades = [
            self._make_dummy_trade(0, net_pnl_usd=-15_000.0, equity_after=-5_000.0)
        ]
        m = OOSEvaluationEngine.calculate_metrics(
            trades=trades,
            initial_equity=10_000.0,
            final_equity=-5_000.0,
            cycle_name="Bankruptcy Test",
            config=default_config,
        )
        # initial_equity = 10,000, peak = 10,000
        # equity = -5,000, dd_usd = 15,000
        # dd_pct = 15,000 / 10,000 = 1.5 (150% drawdown)
        assert pytest.approx(m.max_drawdown, 1e-4) == 1.50
        assert pytest.approx(m.max_drawdown_usd, 1e-4) == 15000.0
        assert not math.isnan(m.max_drawdown)
        assert not math.isinf(m.max_drawdown)
        assert not m.gate_passed
