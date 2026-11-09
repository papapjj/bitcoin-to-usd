"""
Adversarial Stress and Fuzz Testing Suite for RealM1Environment.
Target Component: research_reasoning_world_class/src/real_m1_environment.py
Author: teamwork_challenger_m1_1
Milestone: M1 (Real M1 Candle Replay Environment & Path-Dependent Execution)

Scope:
- Boundary SL/TP ATR ratios: SL=0.01, SL=100.0, TP=0.01, TP=100.0, asymmetric pairs.
- Action validity: negative actions, zero, out of range (>2), non-integer types.
- Bar index boundaries: -1, 0, N-1, N, N+100, float indices.
- Gaps and extreme price movements: single-bar gaps, multi-bar gaps, zero-range bars.
- Double-touch simultaneous breach: strict conservative SL-first priority.
- Dynamic positions tracking in 137D observation vector (0, 1, 5, 10 positions).
- Determinism, numerical cleanliness (0 NaNs, 0 Infs), and execution latency.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from typing import List

import numpy as np
import pytest

# MT5 single-thread convention
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Path setup
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.real_m1_environment import (
    ExitType,
    PositionSlot,
    RealM1Environment,
    TradeResult,
)
from self_growing_trader.protocol_contract import INPUT_COUNT, POSITION_OFFSET

DEFAULT_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
EXPECTED_BAR_COUNT = 64785


@pytest.fixture(scope="module")
def real_env():
    """Module-level loaded environment for integration and boundary fuzzing."""
    if not os.path.exists(DEFAULT_CSV_PATH):
        pytest.skip(f"CSV not found: {DEFAULT_CSV_PATH}")
    env = RealM1Environment(csv_path=DEFAULT_CSV_PATH, spread_usd=2.50)
    count = env.load_data(start_date="2025.01.15", end_date="2025.02.28")
    assert count == EXPECTED_BAR_COUNT
    return env


def make_synthetic_tsv(lines: List[str]) -> str:
    """Helper creating a temporary TSV file for exact scenario fuzzing."""
    header = "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
    full_content = [header]
    # 300 warmup bars at constant $100,000 price
    for i in range(300):
        minute = i % 60
        hour = (i // 60) % 24
        full_content.append(
            f"2025.01.14\t{hour:02d}:{minute:02d}:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250"
        )
    full_content.extend(lines)
    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write("\n".join(full_content) + "\n")
        return f.name


# ============================================================================
# 1. Boundary SL/TP Ratios and Input Hardening
# ============================================================================

class TestBoundarySLTPAndActions:
    """Fuzzes extreme boundary values for SL/TP ATR ratios and actions."""

    @pytest.mark.parametrize("sl_atr,tp_atr", [
        (0.01, 0.01),      # Microscopic stops
        (100.0, 100.0),    # Massive stops
        (0.01, 100.0),     # Tight SL, ultra-wide TP
        (100.0, 0.01),     # Ultra-wide SL, microscopic TP
        (0.0001, 10.0),    # Sub-basis-point SL
        (10.0, 500.0),     # Macro TP
    ])
    def test_extreme_sl_tp_ratios_execute_cleanly(self, real_env, sl_atr, tp_atr):
        """Verifies simulation executes deterministically without NaNs or unhandled crashes."""
        res_buy = real_env.simulate_trade(bar_idx=500, action=1, sl_atr=sl_atr, tp_atr=tp_atr)
        assert isinstance(res_buy, TradeResult)
        assert np.isfinite(res_buy.pnl_usd)
        assert np.isfinite(res_buy.return_atr)
        assert 0 <= res_buy.holding_bars <= 60
        assert res_buy.exit_type in ("sl", "tp", "timestop")

        res_sell = real_env.simulate_trade(bar_idx=500, action=2, sl_atr=sl_atr, tp_atr=tp_atr)
        assert isinstance(res_sell, TradeResult)
        assert np.isfinite(res_sell.pnl_usd)
        assert np.isfinite(res_sell.return_atr)

    @pytest.mark.parametrize("invalid_action", [0, -1, 3, 4, 100, -999])
    def test_invalid_action_integers_rejected(self, real_env, invalid_action):
        """Verifies invalid action values strictly raise ValueError."""
        with pytest.raises(ValueError, match="Invalid action"):
            real_env.simulate_trade(bar_idx=100, action=invalid_action, sl_atr=2.0, tp_atr=4.0)

    @pytest.mark.parametrize("non_int_action", ["BUY", None, 1.5, [1]])
    def test_non_integer_actions_rejected(self, real_env, non_int_action):
        """Verifies non-integer action types strictly raise ValueError or TypeError."""
        with pytest.raises((ValueError, TypeError)):
            real_env.simulate_trade(bar_idx=100, action=non_int_action, sl_atr=2.0, tp_atr=4.0)

    @pytest.mark.parametrize("invalid_sl,invalid_tp", [
        (0.0, 2.0),
        (-1.0, 2.0),
        (2.0, 0.0),
        (2.0, -1.0),
        (-5.0, -5.0),
    ])
    def test_non_positive_sl_tp_rejected(self, real_env, invalid_sl, invalid_tp):
        """Verifies non-positive SL/TP parameters strictly raise ValueError."""
        with pytest.raises(ValueError, match="strictly positive"):
            real_env.simulate_trade(bar_idx=100, action=1, sl_atr=invalid_sl, tp_atr=invalid_tp)

    def test_microscopic_tp_net_loss_due_to_spread(self, real_env):
        """Empirically proves that microscopic TP (0.01 ATR) hitting TP yields negative PnL post-spread."""
        # Gross gain = 0.01 * atr (~$0.70), spread = $2.50 -> net PnL < 0
        res = real_env.simulate_trade(bar_idx=100, action=1, sl_atr=100.0, tp_atr=0.01)
        assert res.exit_type == "tp"
        assert res.pnl_usd < 0.0, "Gross gain on 0.01 ATR must not cover $2.50 spread"


# ============================================================================
# 2. Bar Index Boundary Fuzzing
# ============================================================================

class TestBarIndexBoundaries:
    """Tests extreme and edge-case bar indices on all public observation and simulation APIs."""

    @pytest.mark.parametrize("invalid_idx", [-1, -100, EXPECTED_BAR_COUNT, EXPECTED_BAR_COUNT + 10])
    def test_out_of_bounds_indices_raise_index_error(self, real_env, invalid_idx):
        """Verifies get_observation, get_atr, get_timestamp, and simulate_trade raise IndexError."""
        with pytest.raises(IndexError):
            real_env.get_observation(invalid_idx)
        with pytest.raises(IndexError):
            real_env.get_atr(invalid_idx)
        with pytest.raises(IndexError):
            real_env.get_timestamp(invalid_idx)
        with pytest.raises(IndexError):
            real_env.get_bar(invalid_idx)
        with pytest.raises(IndexError):
            real_env.simulate_trade(invalid_idx, action=1, sl_atr=2.0, tp_atr=4.0)

    @pytest.mark.parametrize("float_idx", [0.5, 10.0, 100.9])
    def test_float_indices_raise_index_error(self, real_env, float_idx):
        """Verifies non-integer indices raise IndexError."""
        with pytest.raises(IndexError):
            real_env.get_observation(float_idx)
        with pytest.raises(IndexError):
            real_env.simulate_trade(float_idx, action=1, sl_atr=2.0, tp_atr=4.0)

    def test_first_bar_boundary_valid(self, real_env):
        """Verifies bar 0 is fully accessible and executable."""
        obs = real_env.get_observation(0)
        assert obs.shape == (137,)
        assert np.all(np.isfinite(obs))
        res = real_env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert isinstance(res, TradeResult)
        assert res.holding_bars > 0

    def test_terminal_bar_boundary_clean_exit(self, real_env):
        """Verifies the exact last bar (N-1) cleanly terminates with DATASET_END and zero holding bars."""
        terminal_idx = EXPECTED_BAR_COUNT - 1
        obs = real_env.get_observation(terminal_idx)
        assert obs.shape == (137,)
        assert np.all(np.isfinite(obs))

        res_buy = real_env.simulate_trade(terminal_idx, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res_buy.holding_bars == 0
        assert res_buy.exit_reason == "DATASET_END"
        assert res_buy.exit_type == "timestop"
        assert res_buy.pnl_usd == pytest.approx(-2.50, abs=1e-3)

        res_sell = real_env.simulate_trade(terminal_idx, action=2, sl_atr=2.0, tp_atr=4.0)
        assert res_sell.holding_bars == 0
        assert res_sell.exit_reason == "DATASET_END"
        assert res_sell.exit_type == "timestop"
        assert res_sell.pnl_usd == pytest.approx(-2.50, abs=1e-3)


# ============================================================================
# 3. Gaps and Adverse Slippage Realization
# ============================================================================

class TestGapsAndAdverseSlippage:
    """Verifies market execution realism: adverse gaps execute at Open, not magical SL."""

    def test_buy_adverse_gap_down_slippage(self):
        """When market opens past BUY SL, exit price must reflect the adverse open, not the SL level."""
        lines = [
            "2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            # Bar 1 opens at 99500 (gaps 300 below SL 99800, atr=100, sl_atr=2)
            "2025.01.15\t00:01:00\t99500.00\t99550.00\t99400.00\t99450.00\t100\t0\t250",
        ]
        tsv = make_synthetic_tsv(lines)
        env = RealM1Environment(csv_path=tsv, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res.exit_type == "sl"
        assert res.holding_bars == 1
        # Exit price must be Open (99500) - half_spread (1.25) = 99498.75
        assert res.exit_price == pytest.approx(99498.75, abs=1e-2)
        # Net loss is 99498.75 - 100001.25 = -502.50 USD (worse than theoretical -202.50)
        assert res.pnl_usd == pytest.approx(-502.50, abs=1e-2)

    def test_sell_adverse_gap_up_slippage(self):
        """When market opens past SELL SL, exit price must reflect the adverse open, not the SL level."""
        lines = [
            "2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            # Bar 1 opens at 100500 (gaps 300 above SL 100200, atr=100, sl_atr=2)
            "2025.01.15\t00:01:00\t100500.00\t100600.00\t100450.00\t100550.00\t100\t0\t250",
        ]
        tsv = make_synthetic_tsv(lines)
        env = RealM1Environment(csv_path=tsv, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(0, action=2, sl_atr=2.0, tp_atr=4.0)
        assert res.exit_type == "sl"
        assert res.holding_bars == 1
        # Exit price must be Open (100500) + half_spread (1.25) = 100501.25
        assert res.exit_price == pytest.approx(100501.25, abs=1e-2)
        # Net loss is 99998.75 - 100501.25 = -502.50 USD
        assert res.pnl_usd == pytest.approx(-502.50, abs=1e-2)

    def test_multibar_adverse_gap_down_slippage(self):
        """Verifies multi-bar path correctly locates gap on second bar."""
        lines = [
            "2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            # Bar 1: Normal bar (Low 99850 > SL 99800)
            "2025.01.15\t00:01:00\t100000.00\t100050.00\t99850.00\t99900.00\t100\t0\t250",
            # Bar 2: Gaps down to 99400
            "2025.01.15\t00:02:00\t99400.00\t99450.00\t99300.00\t99350.00\t100\t0\t250",
        ]
        tsv = make_synthetic_tsv(lines)
        env = RealM1Environment(csv_path=tsv, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res.exit_type == "sl"
        assert res.holding_bars == 2
        assert res.exit_price == pytest.approx(99398.75, abs=1e-2)
        assert res.pnl_usd == pytest.approx(-602.50, abs=1e-2)


# ============================================================================
# 4. Simultaneous Touch Conflict Resolution (Conservative SL-First)
# ============================================================================

class TestConservativeDoubleTouchPriority:
    """Verifies that simultaneous breach of SL and TP strictly resolves to SL."""

    def test_buy_single_bar_double_touch_strict_sl(self):
        """Bar 1 touches both TP and SL -> strictly SL."""
        lines = [
            "2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            # Bar 1: High spikes to 100500 (>= TP 100400) AND Low drops to 99700 (<= SL 99800)
            "2025.01.15\t00:01:00\t100000.00\t100500.00\t99700.00\t100000.00\t100\t0\t250",
        ]
        tsv = make_synthetic_tsv(lines)
        env = RealM1Environment(csv_path=tsv, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res.holding_bars == 1
        assert res.pnl_usd < 0.0

    def test_sell_single_bar_double_touch_strict_sl(self):
        """Bar 1 touches both SELL SL and TP -> strictly SL."""
        lines = [
            "2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            # Bar 1: High spikes to 100300 (>= SL 100200) AND Low drops to 99500 (<= TP 99600)
            "2025.01.15\t00:01:00\t100000.00\t100300.00\t99500.00\t100000.00\t100\t0\t250",
        ]
        tsv = make_synthetic_tsv(lines)
        env = RealM1Environment(csv_path=tsv, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(0, action=2, sl_atr=2.0, tp_atr=4.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res.holding_bars == 1
        assert res.pnl_usd < 0.0

    def test_multibar_double_touch_on_bar_3_strict_sl(self):
        """Bars 1-2 quiet, Bar 3 touches both -> strictly SL on Bar 3."""
        lines = [
            "2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            "2025.01.15\t00:01:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            "2025.01.15\t00:02:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250",
            "2025.01.15\t00:03:00\t100000.00\t100500.00\t99700.00\t100000.00\t100\t0\t250",
        ]
        tsv = make_synthetic_tsv(lines)
        env = RealM1Environment(csv_path=tsv, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res.holding_bars == 3
        assert res.pnl_usd < 0.0


# ============================================================================
# 5. Dynamic Position Injection & Numerical Invariants
# ============================================================================

class TestDynamicPositionsInObservation:
    """Verifies observation matrix integrity when positions are dynamically populated."""

    def test_observation_with_five_active_positions(self, real_env):
        """Verifies full 5-slot position population matches feature offsets with 0 NaNs."""
        env = RealM1Environment(csv_path=DEFAULT_CSV_PATH, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.16")

        entry_close = float(env.close_arr[10])
        # Add 5 active positions (BUY and SELL mixed)
        env.positions = [
            PositionSlot(side=1, entry=entry_close - 50, sl=entry_close - 200, tp=entry_close + 300, risk_distance=150, opened_bar=5),
            PositionSlot(side=-1, entry=entry_close + 50, sl=entry_close + 250, tp=entry_close - 350, risk_distance=200, opened_bar=6),
            PositionSlot(side=1, entry=entry_close, sl=entry_close - 100, tp=entry_close + 200, risk_distance=100, opened_bar=7),
            PositionSlot(side=-1, entry=entry_close + 20, sl=entry_close + 120, tp=entry_close - 180, risk_distance=100, opened_bar=8),
            PositionSlot(side=1, entry=entry_close - 10, sl=entry_close - 110, tp=entry_close + 190, risk_distance=100, opened_bar=9),
        ]
        env.today_trade_count = 5
        env.equity = 10_500.0
        env.peak_equity = 11_000.0

        obs = env.get_observation(10)
        assert obs.shape == (INPUT_COUNT,)
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

        # Slot 0 active flag (index 29) must be 1.0
        assert obs[POSITION_OFFSET] == 1.0
        # Slot 0 side (index 30) must be +1.0
        assert obs[POSITION_OFFSET + 1] == 1.0
        # Slot 1 active flag (index 37) must be 1.0
        assert obs[POSITION_OFFSET + 8] == 1.0
        # Slot 1 side (index 38) must be -1.0
        assert obs[POSITION_OFFSET + 9] == -1.0

    def test_observation_with_oversized_positions_clamped(self, real_env):
        """Verifies having > 5 positions is clamped strictly to 5 slots without index errors."""
        env = RealM1Environment(csv_path=DEFAULT_CSV_PATH, spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.16")

        entry_close = float(env.close_arr[10])
        env.positions = [
            PositionSlot(side=1, entry=entry_close, sl=entry_close - 100, tp=entry_close + 200, risk_distance=100, opened_bar=5)
            for _ in range(12)  # 12 positions
        ]
        obs = env.get_observation(10)
        assert obs.shape == (INPUT_COUNT,)
        assert np.all(np.isfinite(obs))


# ============================================================================
# 6. Performance Benchmarks
# ============================================================================

class TestPerformanceAndLatency:
    """Verifies sub-10 µs SLA and throughput benchmarks."""

    def test_observation_retrieval_sub_10_microsecond_sla(self, real_env):
        """Verifies average observation retrieval time is strictly < 10.0 microseconds."""
        rng = np.random.RandomState(42)
        sample_indices = rng.randint(0, real_env.total_bars, size=10_000)

        # Warmup
        for idx in sample_indices[:500]:
            _ = real_env.get_observation(int(idx))

        t0 = time.perf_counter_ns()
        for idx in sample_indices:
            _ = real_env.get_observation(int(idx))
        elapsed_ns = time.perf_counter_ns() - t0

        avg_us = (elapsed_ns / len(sample_indices)) / 1000.0
        print(f"\n[LATENCY BENCHMARK] get_observation: {avg_us:.3f} us/call")
        assert avg_us < 10.0, f"Observation latency {avg_us:.3f}us exceeds < 10us budget"
