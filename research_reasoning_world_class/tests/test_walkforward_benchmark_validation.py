"""
Milestone 5 Acceptance Benchmark & 10,000-Run Monte Carlo Validation Suite
Module: Features 34 & 35 (Milestone 5 - Acceptance Criteria AC-1 through AC-7)
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Quantitative Reasoning Engine
"""

import math
import os
import sys
import time
import pytest
import numpy as np

# Ensure project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

from research_reasoning_world_class.src.walkforward_simulator import (
    WalkForwardSimulator,
    WalkForwardReport,
    run_monte_carlo_validation,
)
from research_reasoning_world_class.src.dynamic_kelly_compounding import (
    DynamicKellyCompoundingEngine,
    CompoundingPortfolioManager,
    KellyParameters,
)


class TestWalkForwardBenchmarkValidationAC1toAC7:
    """Verifies all 7 Acceptance Criteria over a 90-day multi-asset simulation."""

    def test_full_3_month_walk_forward_meets_all_acceptance_criteria(self) -> None:
        """
        Executes 4,050 trades simulation across 6 assets.
        Validates AC-1 through AC-7 simultaneously.
        """
        simulator = WalkForwardSimulator(
            initial_equity=10_000.0,
            fractional_kelly=0.20,
            base_trade_risk=0.012,
            target_trades=4_050,
            seed=42,
        )

        report = simulator.run_simulation(
            base_win_rate=0.865,
            nominal_payoff=0.75,
        )

        # Print consolidated report
        print(f"\n[3-Month Walk-Forward Benchmark Results]")
        print(f"  Closed Trades:   {report.total_trades} (AC-4 target >= 4,000)")
        print(f"  Win Rate:        {report.win_rate*100:.2f}% (AC-3 target >= 85.0%)")
        print(f"  Profit Factor:   {report.profit_factor:.2f} (AC-2 target >= 3.0)")
        print(f"  Account Growth:  {report.net_growth_multiple:.2f}x (AC-1 target >= 50.0x)")
        print(f"  Max Drawdown:    {report.max_relative_drawdown*100:.2f}% (AC-5 target < 15.0%)")
        print(f"  Avg Latency:     {report.average_latency_ms:.4f} ms")
        print(f"  Max Latency:     {report.max_latency_ms:.4f} ms (AC-6 target <= 50.0 ms)")
        print(f"  NaN/Inf Errors:  {report.nan_inf_violations} (AC-7 target == 0)")

        # AC-1: Net Account Growth >= 50x (+5,000%)
        assert report.ac1_passed, f"AC-1 Failed: Growth {report.net_growth_multiple:.2f}x < 50.0x"

        # AC-2: Profit Factor >= 3.0
        assert report.ac2_passed, f"AC-2 Failed: Profit Factor {report.profit_factor:.2f} < 3.0"

        # AC-3: Win Rate >= 85.0%
        assert report.ac3_passed, f"AC-3 Failed: Win Rate {report.win_rate*100:.2f}% < 85.0%"

        # AC-4: Closed Trades >= 4,000
        assert report.ac4_passed, f"AC-4 Failed: Closed trades {report.total_trades} < 4,000"

        # AC-5: Maximum Relative Drawdown < 15.0%
        assert report.ac5_passed, f"AC-5 Failed: Max Drawdown {report.max_relative_drawdown*100:.2f}% >= 15.0%"

        # AC-6: Decision Latency <= 50.0ms
        assert report.ac6_passed, f"AC-6 Failed: Max latency {report.max_latency_ms:.2f}ms > 50.0ms"

        # AC-7: Zero NaN / Zero Inf
        assert report.ac7_passed, f"AC-7 Failed: {report.nan_inf_violations} NaN/Inf violations"

        assert report.all_passed

    def test_feature_35_10000_run_monte_carlo_risk_verification(self) -> None:
        """
        Feature 35: 10,000-run Monte Carlo verification.
        Confirms breach probability for DD >= 15% is under 1% (< 0.01).
        """
        results = run_monte_carlo_validation(
            iterations=10_000,
            trades_per_cycle=400,
            win_rate=0.865,
            payoff=0.75,
        )

        print(f"\n[10,000-Run Monte Carlo Risk Validation]")
        print(f"  Breach Probability (DD >= 15%): {results['breach_probability']*100:.3f}%")
        print(f"  Median 400-Trade Growth:         {results['median_growth']:.2f}x")
        print(f"  P95 400-Trade Growth:            {results['p95_growth']:.2f}x")

        assert results["breach_probability"] < 0.01, (
            f"Risk Violation: Drawdown breach probability {results['breach_probability']*100:.2f}% >= 1.0%"
        )
        assert results["median_growth"] > 1.5, "Median growth insufficient under Monte Carlo."

    def test_continuous_linear_deleveraging_under_adverse_streak(self) -> None:
        """
        Verifies that an adverse streak of 6 losses smoothly reduces lot size
        without triggering any 24h freeze, maintaining continuous execution with floor 10%.
        """
        engine = DynamicKellyCompoundingEngine(
            KellyParameters(
                fractional_kelly=0.20,
                base_trade_risk_pct=0.012,
                max_drawdown_limit=0.15,
                sniper_dd_threshold=0.08,
                base_win_prob_threshold=0.85,
                sniper_win_prob_threshold=0.90,
                deleveraging_floor=0.10,
            )
        )
        manager = CompoundingPortfolioManager(initial_balance=10_000.0, engine=engine)

        lot_sizes = []
        for _ in range(6):
            sizing = manager.calculate_trade_order(p_win=0.88, payoff_ratio=0.75, sl_points=50.0)
            lot_sizes.append(sizing.lot_size)
            manager.record_trade_result(net_profit_usd=-sizing.dollar_risk, symbol="BTCUSD")

        # Verify lot sizes strictly decrease monotonically
        for i in range(len(lot_sizes) - 1):
            assert lot_sizes[i + 1] < lot_sizes[i], f"Lot size failed to scale down: {lot_sizes}"

        # Verify drawdown did not breach 15%
        assert manager.current_drawdown < 0.15, f"Drawdown {manager.current_drawdown*100:.2f}% breached 15%!"
        # Verify deleveraging factor >= floor 0.10
        assert sizing.deleveraging_factor >= 0.10
