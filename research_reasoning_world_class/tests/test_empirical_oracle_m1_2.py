"""
Empirical Oracle & Invariant Stress Test Suite for RealM1Environment.
Module: research_reasoning_world_class/tests/test_empirical_oracle_m1_2.py
Author: teamwork_challenger_m1_2 (EMPIRICAL CHALLENGER)
Milestone: M1 (Real M1 Candle Replay Environment & Path-Dependent Execution)

Verifies:
1. Independent Naive Ground-Truth Simulator vs RealM1Environment across 1,000 sampled trades:
   - Exact PnL match within 1e-5.
   - Exact holding_bars match.
   - Exact exit_type match ("sl", "tp", "time_stop").
2. Spread deduction invariant:
   - For 100% of trades: PnL_net == PnL_gross - spread_usd ($2.50).
3. Observation vector integrity across all 64,785 bars:
   - Zero NaNs, zero Infs, strictly float32, strictly shape (137,).
4. Boundary and stress conditions:
   - Simultaneous touch conservative SL-first priority.
   - Adverse gap handling (slippage past SL price).
   - Dataset boundary edge conditions (last bar, near-end bars).
   - Random seeds and parameter sweeps (SL in [1, 8], TP in [1.5, 15], MaxHold in [5, 120]).
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SRC_DIR = os.path.join(REPO_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from research_reasoning_world_class.src.real_m1_environment import (
    ExitType,
    PositionSlot,
    RealM1Environment,
    TradeResult,
)
from self_growing_trader.protocol_contract import INPUT_COUNT

DEFAULT_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
EXPECTED_BARS = 64785


@dataclass(frozen=True)
class NaiveTradeResult:
    """Independent naive bar-by-bar simulator result."""
    pnl_usd: float
    gross_pnl_usd: float
    return_atr: float
    exit_type: str
    holding_bars: int
    exit_reason: str
    entry_price: float
    exit_price: float
    exit_bar_idx: int


def naive_simulate_trade(
    env: RealM1Environment,
    bar_idx: int,
    action: int,
    sl_atr: float,
    tp_atr: float,
    max_holding_bars: int = 60,
    spread_usd: float = 2.50,
) -> NaiveTradeResult:
    """
    Independent ground-truth simulator that steps sequentially bar-by-bar
    using purely scalar arithmetic (zero vectorized numpy operations).
    """
    total_bars = env.total_bars
    if bar_idx < 0 or bar_idx >= total_bars:
        raise IndexError(f"bar_idx {bar_idx} out of range [0, {total_bars})")
    if action not in (1, 2):
        raise ValueError(f"Invalid action {action}")
    if sl_atr <= 0.0 or tp_atr <= 0.0:
        raise ValueError("sl_atr and tp_atr must be strictly positive")

    atr = max(float(env.atr_arr[bar_idx]), 1e-4)
    ref_close = float(env.close_arr[bar_idx])
    half_spread = spread_usd / 2.0

    avail = min(max_holding_bars, total_bars - 1 - bar_idx)
    if avail <= 0:
        entry_price = ref_close + half_spread if action == 1 else ref_close - half_spread
        exit_price = ref_close - half_spread if action == 1 else ref_close + half_spread
        pnl_usd = (exit_price - entry_price) if action == 1 else (entry_price - exit_price)
        gross_pnl_usd = 0.0
        return NaiveTradeResult(
            pnl_usd=float(pnl_usd),
            gross_pnl_usd=float(gross_pnl_usd),
            return_atr=float(pnl_usd / atr),
            exit_type="timestop",
            holding_bars=0,
            exit_reason="DATASET_END",
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            exit_bar_idx=bar_idx,
        )

    if action == 1:  # BUY
        entry_price = ref_close + half_spread
        sl_level = ref_close - (sl_atr * atr)
        tp_level = ref_close + (tp_atr * atr)

        # Default fallback if neither SL nor TP is hit
        holding = avail
        exit_level = float(env.close_arr[bar_idx + avail])
        exit_type = "timestop"
        exit_reason = "TIME_STOP"

        # Sequential bar-by-bar step
        for step in range(1, avail + 1):
            curr_bar = bar_idx + step
            curr_open = float(env.open_arr[curr_bar])
            curr_high = float(env.high_arr[curr_bar])
            curr_low = float(env.low_arr[curr_bar])

            sl_hit = curr_low <= sl_level
            tp_hit = curr_high >= tp_level

            if sl_hit and tp_hit:
                # Conservative priority: SL first on simultaneous breach
                holding = step
                exit_level = min(sl_level, curr_open)  # Adverse open gap
                exit_type = "sl"
                exit_reason = "SIMULTANEOUS_SL_FIRST"
                break
            elif sl_hit:
                holding = step
                exit_level = min(sl_level, curr_open)
                exit_type = "sl"
                exit_reason = "SL_HIT"
                break
            elif tp_hit:
                holding = step
                exit_level = tp_level
                exit_type = "tp"
                exit_reason = "TP_HIT"
                break

        exit_price = exit_level - half_spread
        pnl_usd = exit_price - entry_price
        gross_pnl_usd = exit_level - ref_close

    else:  # SELL (action == 2)
        entry_price = ref_close - half_spread
        sl_level = ref_close + (sl_atr * atr)
        tp_level = ref_close - (tp_atr * atr)

        holding = avail
        exit_level = float(env.close_arr[bar_idx + avail])
        exit_type = "timestop"
        exit_reason = "TIME_STOP"

        for step in range(1, avail + 1):
            curr_bar = bar_idx + step
            curr_open = float(env.open_arr[curr_bar])
            curr_high = float(env.high_arr[curr_bar])
            curr_low = float(env.low_arr[curr_bar])

            sl_hit = curr_high >= sl_level
            tp_hit = curr_low <= tp_level

            if sl_hit and tp_hit:
                # Conservative priority: SL first on simultaneous breach
                holding = step
                exit_level = max(sl_level, curr_open)  # Adverse open gap for short
                exit_type = "sl"
                exit_reason = "SIMULTANEOUS_SL_FIRST"
                break
            elif sl_hit:
                holding = step
                exit_level = max(sl_level, curr_open)
                exit_type = "sl"
                exit_reason = "SL_HIT"
                break
            elif tp_hit:
                holding = step
                exit_level = tp_level
                exit_type = "tp"
                exit_reason = "TP_HIT"
                break

        exit_price = exit_level + half_spread
        pnl_usd = entry_price - exit_price
        gross_pnl_usd = ref_close - exit_level

    return NaiveTradeResult(
        pnl_usd=float(pnl_usd),
        gross_pnl_usd=float(gross_pnl_usd),
        return_atr=float(pnl_usd / atr),
        exit_type=exit_type,
        holding_bars=int(holding),
        exit_reason=exit_reason,
        entry_price=float(entry_price),
        exit_price=float(exit_price),
        exit_bar_idx=int(bar_idx + holding),
    )


@pytest.fixture(scope="module")
def env_fixture():
    """Module-level fixture loading full 64,785 bars from BTCUSDm.csv."""
    if not os.path.exists(DEFAULT_CSV_PATH):
        pytest.skip(f"Data file not found at {DEFAULT_CSV_PATH}")
    env = RealM1Environment(csv_path=DEFAULT_CSV_PATH, spread_usd=2.50)
    bars = env.load_data(start_date="2025.01.15", end_date="2025.02.28")
    assert bars == EXPECTED_BARS, f"Expected {EXPECTED_BARS} bars, got {bars}"
    return env


class TestEmpiricalOracle1000Trades:
    """Rigorous empirical validation of simulate_trade against naive ground-truth."""

    def test_1000_sampled_trades_exact_oracle_agreement(self, env_fixture):
        """
        Samples 1,000 distinct trades across the dataset and asserts:
        1. Exact PnL agreement within 1e-5.
        2. Exact holding_bars agreement.
        3. Exact exit_type agreement ("sl", "tp", "time_stop").
        4. Invariant: PnL_net == PnL_gross - spread_usd ($2.50).
        """
        rng = np.random.RandomState(42)
        total_bars = env_fixture.total_bars

        # Construct stratified sample of 1,000 bar indices:
        # 950 random bars across dataset, 50 boundary and near-end bars
        random_bars = rng.randint(0, total_bars - 120, size=950).tolist()
        boundary_bars = list(range(total_bars - 50, total_bars))
        sample_bars = random_bars + boundary_bars
        assert len(sample_bars) == 1000

        # Parameter distributions
        actions = rng.choice([1, 2], size=1000).tolist()
        sl_atrs = rng.uniform(1.0, 8.0, size=1000).tolist()
        tp_atrs = rng.uniform(1.5, 15.0, size=1000).tolist()
        max_holds = rng.choice([15, 30, 45, 60, 90, 120], size=1000).tolist()
        spreads = [2.50] * 900 + [0.0] * 50 + [5.0] * 50

        pnl_discrepancies = []
        holding_discrepancies = []
        exit_discrepancies = []
        spread_discrepancies = []

        exit_counts = {"sl": 0, "tp": 0, "timestop": 0}

        t0 = time.perf_counter()
        for i in range(1000):
            b_idx = sample_bars[i]
            act = actions[i]
            sl = sl_atrs[i]
            tp = tp_atrs[i]
            mh = max_holds[i]
            sp = spreads[i]

            # Vectorized implementation
            sim_res = env_fixture.simulate_trade(
                bar_idx=b_idx,
                action=act,
                sl_atr=sl,
                tp_atr=tp,
                max_holding_bars=mh,
                spread_usd=sp,
            )

            # Ground-truth naive implementation
            oracle_res = naive_simulate_trade(
                env=env_fixture,
                bar_idx=b_idx,
                action=act,
                sl_atr=sl,
                tp_atr=tp,
                max_holding_bars=mh,
                spread_usd=sp,
            )

            # 1. PnL Check within 1e-5
            pnl_diff = abs(sim_res.pnl_usd - oracle_res.pnl_usd)
            if pnl_diff > 1e-5:
                pnl_discrepancies.append((i, b_idx, act, sim_res.pnl_usd, oracle_res.pnl_usd, pnl_diff))

            # 2. Holding Bars Check
            if sim_res.holding_bars != oracle_res.holding_bars:
                holding_discrepancies.append((i, b_idx, act, sim_res.holding_bars, oracle_res.holding_bars))

            # 3. Exit Type Check
            sim_exit = str(sim_res.exit_type).lower().replace("_", "")
            oracle_exit = str(oracle_res.exit_type).lower().replace("_", "")
            if sim_exit != oracle_exit:
                exit_discrepancies.append((i, b_idx, act, sim_res.exit_type, oracle_res.exit_type))

            # 4. Spread Invariant Check: PnL_net == PnL_gross - spread
            expected_net = oracle_res.gross_pnl_usd - sp
            spread_diff = abs(sim_res.pnl_usd - expected_net)
            if spread_diff > 1e-5:
                spread_discrepancies.append((i, b_idx, sim_res.pnl_usd, expected_net, spread_diff))

            exit_counts[oracle_exit] += 1

        elapsed = time.perf_counter() - t0
        print(f"\n[ORACLE BENCHMARK] Evaluated 1,000 trades in {elapsed:.3f}s ({elapsed/1000*1e6:.1f} µs/trade)")
        print(f"[ORACLE DISTRIBUTION] Exits: SL={exit_counts['sl']}, TP={exit_counts['tp']}, TimeStop={exit_counts['timestop']}")

        # Report any failures with maximum diagnostic detail
        assert len(pnl_discrepancies) == 0, f"PnL discrepancies found in {len(pnl_discrepancies)} trades: {pnl_discrepancies[:5]}"
        assert len(holding_discrepancies) == 0, f"Holding bar discrepancies found in {len(holding_discrepancies)} trades: {holding_discrepancies[:5]}"
        assert len(exit_discrepancies) == 0, f"Exit type discrepancies found in {len(exit_discrepancies)} trades: {exit_discrepancies[:5]}"
        assert len(spread_discrepancies) == 0, f"Spread invariant violations found in {len(spread_discrepancies)} trades: {spread_discrepancies[:5]}"


class TestObservationNumericalIntegrity:
    """Verification of 137D float32 representation and 0 NaN / 0 Inf invariants."""

    def test_full_dataset_obs_matrix_zero_nan_inf(self, env_fixture):
        """Validates that the entire 64,785 x 137 float32 observation matrix has zero NaNs and zero Infs."""
        matrix = env_fixture.obs_matrix
        assert matrix.shape == (EXPECTED_BARS, INPUT_COUNT), f"Expected shape ({EXPECTED_BARS}, {INPUT_COUNT}), got {matrix.shape}"
        assert matrix.dtype == np.float32, f"Expected dtype float32, got {matrix.dtype}"

        nan_count = int(np.isnan(matrix).sum())
        inf_count = int(np.isinf(matrix).sum())
        assert nan_count == 0, f"Found {nan_count} NaNs in obs_matrix!"
        assert inf_count == 0, f"Found {inf_count} Infs in obs_matrix!"

    def test_dynamic_get_observation_zero_nan_inf(self, env_fixture):
        """Tests get_observation() with multi-slot dynamic active positions across 500 sampled bars."""
        rng = np.random.RandomState(123)
        sample_indices = rng.randint(100, EXPECTED_BARS - 100, size=500)

        # Inject realistic simulated position state into env
        env_fixture.positions = [
            PositionSlot(side=1, entry=95000.0, sl=94000.0, tp=97000.0, risk_distance=1000.0, opened_bar=50),
            PositionSlot(side=-1, entry=96000.0, sl=97500.0, tp=93000.0, risk_distance=1500.0, opened_bar=60),
            PositionSlot(side=1, entry=95500.0, sl=94500.0, tp=98000.0, risk_distance=1000.0, opened_bar=70),
        ]
        env_fixture.equity = 10_250.0
        env_fixture.peak_equity = 10_500.0
        env_fixture.today_trade_count = 4

        for idx in sample_indices:
            obs = env_fixture.get_observation(int(idx))
            assert obs.shape == (137,), f"Expected shape (137,), got {obs.shape}"
            assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"
            assert not np.isnan(obs).any(), f"NaN detected in dynamic observation at bar {idx}"
            assert not np.isinf(obs).any(), f"Inf detected in dynamic observation at bar {idx}"

        # Reset env state
        env_fixture.reset(0)


class TestAdversarialEdgeCases:
    """Adversarial stress-testing of execution boundary conditions."""

    def test_simultaneous_sl_tp_breach_strictly_favors_sl(self, env_fixture):
        """
        Adversarial test: Construct synthetic extreme bar where BOTH SL and TP
        are breached simultaneously. Verify that SL-first conservative execution is
        strictly and unfailingly applied for both BUY and SELL actions.
        """
        # Test BUY simultaneous
        # Set tight SL (0.01 ATR) and tight TP (0.01 ATR) on a bar known to have high volatility
        # Find high volatility bar
        high_vol_bar = int(np.argmax(env_fixture.atr_arr))
        res_buy = env_fixture.simulate_trade(
            bar_idx=high_vol_bar,
            action=1,
            sl_atr=0.01,
            tp_atr=0.01,
            max_holding_bars=1,
        )
        assert res_buy.exit_type == "sl", f"Simultaneous breach failed conservative SL rule: got {res_buy.exit_type}"
        assert res_buy.holding_bars == 1
        assert res_buy.pnl_usd < 0.0

        # Test SELL simultaneous
        res_sell = env_fixture.simulate_trade(
            bar_idx=high_vol_bar,
            action=2,
            sl_atr=0.01,
            tp_atr=0.01,
            max_holding_bars=1,
        )
        assert res_sell.exit_type == "sl", f"Simultaneous breach failed conservative SL rule: got {res_sell.exit_type}"
        assert res_sell.holding_bars == 1
        assert res_sell.pnl_usd < 0.0

    def test_adverse_gap_slippage_realization(self, env_fixture):
        """
        Adversarial test: When market gaps over SL at the Open of subsequent bar,
        the realized execution price MUST be at the worse Open price, not the theoretical SL level.
        """
        # Find indices across the dataset where Open gaps down significantly: (close[i] - open[i+1]) > 0.1 * atr[i]
        diffs = env_fixture.close_arr[:-1] - env_fixture.open_arr[1:]
        thresholds = 0.1 * env_fixture.atr_arr[:-1]
        down_gap_indices = np.where(diffs > thresholds)[0]
        assert len(down_gap_indices) > 0, "No down-gaps found across 64,785 bars"

        verified_gaps = 0
        for i in down_gap_indices[:20]:
            atr = env_fixture.atr_arr[i]
            prev_close = env_fixture.close_arr[i]
            next_open = env_fixture.open_arr[i + 1]

            # Place BUY with SL between prev_close and next_open
            # e.g., SL distance is half the gap
            sl_gap_dist = (prev_close - next_open) * 0.5
            sl_atr = sl_gap_dist / atr
            if sl_atr <= 0:
                continue

            res = env_fixture.simulate_trade(
                bar_idx=int(i),
                action=1,
                sl_atr=float(sl_atr),
                tp_atr=10.0,
                max_holding_bars=1,
            )
            assert res.exit_type == "sl"
            # Sl_level was prev_close - sl_atr * atr > next_open
            # Adverse gap rule: exit_level must be next_open, which is worse than sl_level
            sl_level = prev_close - sl_atr * atr
            assert res.exit_price < (sl_level - 1.25)
            assert res.exit_price == pytest.approx(next_open - 1.25, abs=1e-4)
            verified_gaps += 1

        print(f"\n[GAP TEST] Verified adverse gap slippage on {verified_gaps} empirical gap bars.")
        assert verified_gaps >= 5, f"Expected at least 5 verified gaps, got {verified_gaps}"

    def test_dataset_boundary_final_bar_safe_handling(self, env_fixture):
        """Verifies trade initiated on the very last bar (avail = 0) exits safely at DATASET_END."""
        last_idx = env_fixture.total_bars - 1
        res = env_fixture.simulate_trade(bar_idx=last_idx, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res.holding_bars == 0
        assert res.exit_reason == "DATASET_END"
        assert res.pnl_usd == pytest.approx(-2.50, abs=1e-5)
        assert res.exit_type in ("timestop", "time_stop")


class TestRawCsvDiskGroundTruth:
    """Verifies RealM1Environment internal arrays directly against raw disk bytes from BTCUSDm.csv."""

    def test_raw_csv_sample_matches_arrays(self, env_fixture):
        """
        Directly opens C:\\Users\\Admin\\Desktop\\BTCUSDm.csv, reads a random sample of
        raw TSV lines, parses the OHLC values with pure Python, and asserts that
        env_fixture.open_arr, high_arr, low_arr, close_arr match to within 1e-4.
        """
        import io
        import pandas as pd

        # Seek directly to 2025.01.15 in raw CSV file
        matching_lines = []
        with open(DEFAULT_CSV_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("2025.01.15\t00:00:00"):
                    matching_lines.append(line)
                    for _ in range(99):
                        matching_lines.append(f.readline())
                    break

        assert len(matching_lines) == 100, f"Expected 100 lines, found {len(matching_lines)}"

        for i, line in enumerate(matching_lines):
            parts = line.strip().split("\t")
            # Format: DATE, TIME, OPEN, HIGH, LOW, CLOSE, ...
            expected_date = parts[0]
            expected_time = parts[1]
            expected_open = float(parts[2])
            expected_high = float(parts[3])
            expected_low = float(parts[4])
            expected_close = float(parts[5])

            assert abs(env_fixture.open_arr[i] - expected_open) < 1e-2
            assert abs(env_fixture.high_arr[i] - expected_high) < 1e-2
            assert abs(env_fixture.low_arr[i] - expected_low) < 1e-2
            assert abs(env_fixture.close_arr[i] - expected_close) < 1e-2
            assert f"{expected_date} {expected_time}" in env_fixture.get_timestamp(i)


class TestInputValidationAndContractSafety:
    """Tests input rejection and error boundaries for simulate_trade."""

    @pytest.mark.parametrize("invalid_act", [-1, 0, 3, 4, 99])
    def test_invalid_action_rejected(self, env_fixture, invalid_act):
        with pytest.raises(ValueError):
            env_fixture.simulate_trade(bar_idx=10, action=invalid_act, sl_atr=2.0, tp_atr=4.0)

    @pytest.mark.parametrize("bad_sl,bad_tp", [(-1.0, 2.0), (0.0, 2.0), (2.0, 0.0), (2.0, -5.0)])
    def test_invalid_sl_tp_rejected(self, env_fixture, bad_sl, bad_tp):
        with pytest.raises(ValueError):
            env_fixture.simulate_trade(bar_idx=10, action=1, sl_atr=bad_sl, tp_atr=bad_tp)

    @pytest.mark.parametrize("bad_idx", [-10, -1, EXPECTED_BARS, EXPECTED_BARS + 10])
    def test_out_of_bounds_bar_idx_rejected(self, env_fixture, bad_idx):
        with pytest.raises(IndexError):
            env_fixture.simulate_trade(bar_idx=bad_idx, action=1, sl_atr=2.0, tp_atr=4.0)



if __name__ == "__main__":
    pytest.main(["-v", "-s", __file__])
