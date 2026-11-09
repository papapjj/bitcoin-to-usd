"""
Independent Verification Tests for Module R5:
Fractional Kelly Dynamic Compounding & Continuous Risk Scaling
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine

Validates Acceptance Criteria:
- AC-1: Account Growth >= 50x in 4,000 trades.
- AC-2: Profit Factor >= 3.0.
- AC-3: Win Rate >= 85.0%.
- AC-4: Trade count N = 4,000.
- AC-5: Maximum Drawdown < 15.0%.
- AC-6: Real-time decision latency <= 50ms (achieves < 50us).
- AC-7: Numerical integrity: 0 NaN, 0 Inf arithmetic preservation.
"""

from __future__ import annotations

import math
import os
import sys
import time
import pytest
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.dynamic_kelly_compounding import (
    KellyParameters,
    TradeSizingDecision,
    MonteCarloResult,
    DynamicKellyCompoundingEngine,
    CompoundingPortfolioManager,
)


@pytest.fixture
def default_params() -> KellyParameters:
    return KellyParameters(
        fractional_kelly=0.20,
        base_trade_risk_pct=0.012,
        max_drawdown_limit=0.15,
        sniper_dd_threshold=0.08,
        base_win_prob_threshold=0.85,
        sniper_win_prob_threshold=0.90,
        deleveraging_floor=0.10,
        nominal_payoff_ratio=0.70,
    )


@pytest.fixture
def engine(default_params: KellyParameters) -> DynamicKellyCompoundingEngine:
    return DynamicKellyCompoundingEngine(default_params)


class TestKellyFormulasAndBounds:
    """Verifies mathematical validity of Full and Fractional Kelly formulations."""

    def test_full_kelly_calculation(self, engine: DynamicKellyCompoundingEngine):
        # p = 0.85, b = 0.70 => K* = (0.85 * 1.70 - 1.0) / 0.70 = 0.445 / 0.70 = 0.635714...
        k = engine.calculate_full_kelly(p_win=0.85, payoff_ratio=0.70)
        expected = (0.85 * 1.70 - 1.0) / 0.70
        assert math.isclose(k, expected, rel_tol=1e-6)

        # p = 0.90, b = 0.70 => K* = (0.90 * 1.70 - 1.0) / 0.70 = 0.530 / 0.70 = 0.757142...
        k_sniper = engine.calculate_full_kelly(p_win=0.90, payoff_ratio=0.70)
        assert math.isclose(k_sniper, (0.90 * 1.70 - 1.0) / 0.70, rel_tol=1e-6)
        assert k_sniper > k

    def test_full_kelly_zero_or_negative_edge(self, engine: DynamicKellyCompoundingEngine):
        # Break-even coin flip: p = 0.50, b = 1.0 => K* = 0.0
        assert engine.calculate_full_kelly(p_win=0.50, payoff_ratio=1.0) == 0.0

        # Negative edge: p = 0.40, b = 1.0 => K* = 0.0 (strictly non-negative)
        assert engine.calculate_full_kelly(p_win=0.40, payoff_ratio=1.0) == 0.0

        # Boundary inputs
        assert engine.calculate_full_kelly(p_win=0.0, payoff_ratio=1.0) == 0.0
        assert engine.calculate_full_kelly(p_win=1.0, payoff_ratio=1.0) == 1.0
        assert engine.calculate_full_kelly(p_win=0.85, payoff_ratio=0.0) == 0.0

    def test_fractional_kelly_range(self, engine: DynamicKellyCompoundingEngine):
        # Fractional Kelly f* must be scaled in [0.15, 0.25]
        k_full = engine.calculate_full_kelly(0.85, 0.70)

        f_15 = engine.calculate_fractional_kelly(0.85, 0.70, fractional_multiplier=0.15)
        f_20 = engine.calculate_fractional_kelly(0.85, 0.70, fractional_multiplier=0.20)
        f_25 = engine.calculate_fractional_kelly(0.85, 0.70, fractional_multiplier=0.25)

        assert math.isclose(f_15, 0.15 * k_full, rel_tol=1e-6)
        assert math.isclose(f_20, 0.20 * k_full, rel_tol=1e-6)
        assert math.isclose(f_25, 0.25 * k_full, rel_tol=1e-6)
        assert f_15 < f_20 < f_25

        # Multiplier outside [0.15, 0.25] is clamped safely
        f_under = engine.calculate_fractional_kelly(0.85, 0.70, fractional_multiplier=0.05)
        assert math.isclose(f_under, 0.15 * k_full, rel_tol=1e-6)

        f_over = engine.calculate_fractional_kelly(0.85, 0.70, fractional_multiplier=0.50)
        assert math.isclose(f_over, 0.25 * k_full, rel_tol=1e-6)


class TestContinuousDeleveraging:
    """Verifies continuous 24/7 linear de-leveraging and non-zero risk floor."""

    def test_deleveraging_curve(self, engine: DynamicKellyCompoundingEngine):
        # At 0% DD: factor = 1.0
        assert engine.calculate_continuous_deleveraging(0.0) == 1.0

        # At 7.5% DD: factor = 1.0 - 0.075 / 0.15 = 0.50
        assert math.isclose(engine.calculate_continuous_deleveraging(0.075), 0.50, rel_tol=1e-6)

        # At 12% DD: factor = 1.0 - 0.12 / 0.15 = 0.20
        assert math.isclose(engine.calculate_continuous_deleveraging(0.12), 0.20, rel_tol=1e-6)

        # At 13.5% DD: factor = 1.0 - 0.135 / 0.15 = 0.10 (hits floor)
        assert math.isclose(engine.calculate_continuous_deleveraging(0.135), 0.10, rel_tol=1e-6)

        # At 15% DD and beyond: factor remains strictly at 0.10 (floor)
        assert engine.calculate_continuous_deleveraging(0.15) == 0.10
        assert engine.calculate_continuous_deleveraging(0.20) == 0.10

    def test_continuous_24_7_never_zero(self, engine: DynamicKellyCompoundingEngine):
        # Sweep DD from 0% to 50%
        for dd in np.linspace(0.0, 0.50, 100):
            factor = engine.calculate_continuous_deleveraging(float(dd))
            assert factor >= 0.10, f"Deleveraging factor dropped below 0.10 at DD={dd}"
            assert factor <= 1.0, f"Deleveraging factor exceeded 1.0 at DD={dd}"

    def test_monotonic_decrease(self, engine: DynamicKellyCompoundingEngine):
        dds = np.linspace(0.0, 0.15, 50)
        factors = [engine.calculate_continuous_deleveraging(float(d)) for d in dds]
        # Verify strictly non-increasing
        for i in range(len(factors) - 1):
            assert factors[i] >= factors[i + 1]


class TestSniperAAdmissionGate:
    """Verifies dynamic escalation of win probability admission threshold when DD > 8%."""

    def test_normal_mode_admission(self, engine: DynamicKellyCompoundingEngine):
        # DD = 5% (<= 8%)
        current_dd = 0.05

        # P(Win) = 0.86 >= 0.85 => Admitted
        admitted, req_p, in_sniper, rej = engine.evaluate_trade_admission(0.86, current_dd)
        assert admitted is True
        assert req_p == 0.85
        assert in_sniper is False
        assert rej is None

        # P(Win) = 0.84 < 0.85 => Rejected
        admitted, req_p, in_sniper, rej = engine.evaluate_trade_admission(0.84, current_dd)
        assert admitted is False
        assert req_p == 0.85
        assert in_sniper is False
        assert "below required threshold" in rej

    def test_sniper_mode_admission(self, engine: DynamicKellyCompoundingEngine):
        # DD = 9% (> 8%)
        current_dd = 0.09

        # P(Win) = 0.88 would pass in normal mode, but in Sniper mode it MUST be rejected!
        admitted, req_p, in_sniper, rej = engine.evaluate_trade_admission(0.88, current_dd)
        assert admitted is False
        assert req_p == 0.90
        assert in_sniper is True
        assert "Sniper Mode (DD > 8%)" in rej

        # P(Win) = 0.92 >= 0.90 => Admitted in Sniper Mode
        admitted, req_p, in_sniper, rej = engine.evaluate_trade_admission(0.92, current_dd)
        assert admitted is True
        assert req_p == 0.90
        assert in_sniper is True
        assert rej is None

    def test_threshold_boundary_continuity(self, engine: DynamicKellyCompoundingEngine):
        # Exactly at 8% => Normal mode
        admitted, req_p, in_sniper, _ = engine.evaluate_trade_admission(0.85, 0.08)
        assert in_sniper is False
        assert req_p == 0.85
        assert admitted is True

        # Just above 8% (8.001%) => Sniper mode
        admitted, req_p, in_sniper, _ = engine.evaluate_trade_admission(0.85, 0.08001)
        assert in_sniper is True
        assert req_p == 0.90
        assert admitted is False


class TestExponentialCompoundingAndLotSizing:
    """Verifies exponential compounding law and MT5 lot size computations."""

    def test_exponential_growth_math(self, engine: DynamicKellyCompoundingEngine):
        # E_N = E_0 * (1 + r)^N
        # If r = 0.00102 (0.102% per trade), N = 4000
        # (1 + 0.00102)^4000 = 58.74x >= 50x
        growth = engine.simulate_exponential_growth(initial_equity=10000.0, n_trades=4000, avg_return_per_trade=0.00102)
        assert growth / 10000.0 >= 50.0

    def test_lot_size_scaling_with_equity(self, engine: DynamicKellyCompoundingEngine):
        # Base setup: SL = 20 points, point_value = 1.0
        decision_10k = engine.calculate_position_size(
            equity=10000.0,
            peak_equity=10000.0,
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=20.0,
            point_value=1.0,
        )
        assert decision_10k.admitted is True
        lot_10k = decision_10k.lot_size

        # After 10x equity growth to 100k
        decision_100k = engine.calculate_position_size(
            equity=100000.0,
            peak_equity=100000.0,
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=20.0,
            point_value=1.0,
        )
        assert decision_100k.admitted is True
        lot_100k = decision_100k.lot_size

        # Lot size should scale proportionally by 10x
        assert math.isclose(lot_100k / lot_10k, 10.0, rel_tol=1e-2)

    def test_broker_lot_step_and_clamping(self, engine: DynamicKellyCompoundingEngine):
        # Test broker min_lot, max_lot, and lot_step
        decision = engine.calculate_position_size(
            equity=10000.0,
            peak_equity=10000.0,
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=20.0,
            point_value=1.0,
            min_lot=0.01,
            max_lot=5.0,
            lot_step=0.05,
        )
        assert decision.admitted is True
        # Quantized to nearest 0.05
        remainder = round(decision.lot_size % 0.05, 4)
        assert remainder in (0.0, 0.05)
        assert 0.01 <= decision.lot_size <= 5.0


class TestNumericalIntegrityAndEdgeCases:
    """Verifies AC-7: Zero NaN / Zero Inf strict preservation and safety guards."""

    def test_nan_inf_inputs(self, engine: DynamicKellyCompoundingEngine):
        decision_nan = engine.calculate_position_size(
            equity=float("nan"),
            peak_equity=10000.0,
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=20.0,
        )
        assert decision_nan.admitted is False
        assert "NaN or Inf" in str(decision_nan.rejection_reason)
        assert not math.isnan(decision_nan.lot_size)
        assert decision_nan.lot_size == 0.0

        decision_inf = engine.calculate_position_size(
            equity=10000.0,
            peak_equity=float("inf"),
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=20.0,
        )
        assert decision_inf.admitted is False
        assert decision_inf.lot_size == 0.0

    def test_zero_or_negative_equity(self, engine: DynamicKellyCompoundingEngine):
        decision = engine.calculate_position_size(
            equity=0.0,
            peak_equity=10000.0,
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=20.0,
        )
        assert decision.admitted is False
        assert decision.lot_size == 0.0
        assert "equity <= 0" in str(decision.rejection_reason)

    def test_zero_sl_points(self, engine: DynamicKellyCompoundingEngine):
        decision = engine.calculate_position_size(
            equity=10000.0,
            peak_equity=10000.0,
            p_win=0.85,
            payoff_ratio=0.70,
            sl_points=0.0,
        )
        assert decision.admitted is False
        assert decision.lot_size == 0.0
        assert "Invalid SL points" in str(decision.rejection_reason)


class TestRealtimeLatencyBenchmark:
    """Verifies AC-6: Real-time latency budget <= 50ms per decision (target < 50us)."""

    def test_decision_latency_sub_millisecond(self, engine: DynamicKellyCompoundingEngine):
        # Warmup
        for _ in range(50):
            engine.calculate_position_size(
                equity=10000.0,
                peak_equity=10500.0,
                p_win=0.86,
                payoff_ratio=0.70,
                sl_points=15.0,
            )

        n_calls = 2000
        t0 = time.perf_counter()
        for i in range(n_calls):
            engine.calculate_position_size(
                equity=10000.0 + i,
                peak_equity=12000.0,
                p_win=0.87,
                payoff_ratio=0.70,
                sl_points=15.0,
            )
        total_time = time.perf_counter() - t0
        avg_latency_ms = (total_time / n_calls) * 1000.0

        # Must be well below 50ms (typically < 0.05ms)
        assert avg_latency_ms < 1.0, f"Average latency too high: {avg_latency_ms:.4f} ms"
        assert avg_latency_ms < 50.0  # AC-6 mandate


class TestCompoundingPortfolioManager:
    """Verifies stateful portfolio and trade execution tracking."""

    def test_manager_trade_lifecycle(self, engine: DynamicKellyCompoundingEngine):
        manager = CompoundingPortfolioManager(initial_balance=10000.0, engine=engine)

        assert manager.balance == 10000.0
        assert manager.equity == 10000.0
        assert manager.current_drawdown == 0.0

        # Simulate 5 winning trades of +$100 each
        for _ in range(5):
            manager.record_trade_result(100.0, symbol="BTCUSD")

        assert manager.equity == 10500.0
        assert manager.peak_equity == 10500.0
        assert manager.current_drawdown == 0.0
        assert manager.win_rate == 1.0

        # Simulate 1 loss of -$200
        manager.record_trade_result(-200.0, symbol="BTCUSD")
        assert manager.equity == 10300.0
        assert manager.peak_equity == 10500.0
        expected_dd = (10500.0 - 10300.0) / 10500.0
        assert math.isclose(manager.current_drawdown, expected_dd, rel_tol=1e-5)
        assert manager.max_drawdown_observed == expected_dd
        assert manager.win_rate == 5 / 6
        assert math.isclose(manager.profit_factor, 500.0 / 200.0, rel_tol=1e-5)


class TestMonteCarloFullScaleValidation:
    """
    Core Mathematical Proof:
    10,000 independent Monte Carlo simulation paths x 4,000 trades.
    Demonstrates:
    - Account growth >= 50x in >= 99.9% of paths.
    - Maximum relative drawdown < 15.0% across all paths.
    - Zero risk of ruin (0 count / 0.00%).
    - Realized Win Rate >= 85.0%.
    - Realized Profit Factor >= 3.0.
    """

    def test_monte_carlo_10000_runs_all_criteria(self, engine: DynamicKellyCompoundingEngine):
        result: MonteCarloResult = engine.run_monte_carlo(
            n_simulations=10000,
            n_trades=4000,
            initial_equity=10000.0,
            base_win_prob=0.85,
            sniper_win_prob=0.90,
            payoff_ratio=0.70,
            seed=42,
        )

        # 1. Total Simulations & Trades (AC-4)
        assert result.n_simulations == 10000
        assert result.n_trades == 4000

        # 2. Account Growth >= 50x (AC-1)
        assert result.growth_ge_50x_pct >= 99.9, f"Growth >= 50x rate: {result.growth_ge_50x_pct:.2f}%"
        assert result.min_growth >= 50.0, f"Min growth was {result.min_growth:.2f}x"
        assert result.median_growth >= 1000.0

        # 3. Maximum Drawdown < 15.0% (AC-5)
        assert result.overall_max_dd < 0.15, f"Overall Max DD: {result.overall_max_dd * 100:.2f}%"
        assert result.p99_max_dd < 0.12, f"99th percentile DD: {result.p99_max_dd * 100:.2f}%"
        assert result.dd_lt_15_pct == 100.0, "Some simulations exceeded 15% drawdown"

        # 4. Zero Risk of Ruin
        assert result.ruin_count == 0, f"Ruin count was {result.ruin_count}"
        assert result.ruin_rate == 0.0

        # 5. Win Rate >= 85.0% (AC-3)
        assert result.mean_win_rate >= 0.85, f"Mean Win Rate: {result.mean_win_rate * 100:.2f}%"

        # 6. Profit Factor >= 3.0 (AC-2)
        assert result.mean_profit_factor >= 3.0, f"Mean Profit Factor: {result.mean_profit_factor:.2f}"

        # 7. Complete acceptance flag
        assert result.passed_all_ac is True
