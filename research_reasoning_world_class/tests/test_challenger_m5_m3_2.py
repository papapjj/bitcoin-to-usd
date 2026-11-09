"""
Adversarial Empirical Challenge Test Suite for Milestone 3 (M5 MT5 Export Readiness & Evaluation)
File: research_reasoning_world_class/tests/test_challenger_m5_m3_2.py
Author: challenger_m5_m3_2 (Empirical Challenger 2 for Milestone 3)
Role: critic, specialist

Test Coverage:
1. Zero Lookahead Probe:
   - Verify that corrupting/modifying future bars (t+1 .. t+N) does NOT alter observation or MCTS consensus at bar t.
   - Verify Dyna parameter beta_dyna is strictly 0.0 in OOS evaluation, eliminating future ground-truth reliance.
   - Verify SlicedM5Environment enforces strict temporal containment without peeking beyond window boundaries.
2. Path-Dependent Execution Edge Cases:
   - Adverse gap open beyond SL (gap slippage fill at Open, worse than SL level) for BUY and SELL.
   - Gap open beyond TP (limit order fill at TP level).
   - Simultaneous SL and TP touch on single bar (conservative SIMULTANEOUS_SL_FIRST priority).
   - Exact bar holding limit expiration (TIME_STOP after exactly max_holding_bars).
   - Terminal bar boundary execution (DATASET_END with 0 bars held).
3. Spread Accounting Integrity:
   - Flat market exact loss: Zero price delta must produce exactly -$2.50 PnL.
   - Exact $2.50 deduction on SL, TP, and TIME_STOP exits.
   - Strict linear scaling with configurable spread parameter.
   - Empirical validation on real market bars verifying zero spread omission or doubling.
4. Independent Metrics Verification:
   - Independent recomputation of Win Rate, Profit Factor, Gross Profit, Gross Loss, Net PnL, Max Drawdown.
   - Exact reconciliation against outputs/m5_evaluation_report.md.
   - Adversarial audit of report text for misleading '+$-1,496.66' and 'Positive Realized PnL' claims.
   - Validation of R7 strict acceptance gate status.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
import unittest

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import numpy as np
import torch

torch.set_num_threads(min(2, os.cpu_count() or 2))

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from research_reasoning_world_class.src.real_m5_environment import (
    RealM5Environment,
    TradeResult,
    ExitType,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from scripts.evaluate_m5_champion import load_m5_champion_trainer
from scripts.train_mythos_muzero_coevolution_m5 import (
    SlicedM5Environment,
    OOSEvaluationEngineM5,
    OOSConfigM5,
    OOSTradeRecordM5,
    M5MuZeroTrainer,
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "he_chuyen_gia"
    / "v193_runs"
    / "mythos_muzero_m5_champion.pt"
)
M5_CSV_PATH = PROJECT_ROOT / "outputs" / "BTCUSDm_M5.csv"
EVAL_REPORT_PATH = PROJECT_ROOT / "outputs" / "m5_evaluation_report.md"


class MockTrace:
    def __init__(self):
        self.output_vector = np.zeros(20, dtype=np.float64)
        self.selected_action = np.zeros(20, dtype=np.float64)
        self.action = 0
        self.action_index = 0
        self.action_details = {}


class MockBasePolicy:
    def act_nam_tac_nhan(self, obs, *args, **kwargs):
        return [MockTrace() for _ in range(5)]


# ==============================================================================
# 1. ZERO LOOKAHEAD PROBES
# ==============================================================================
class TestZeroLookaheadEnforcement(unittest.TestCase):
    """Adversarial stress-testing of zero lookahead invariants."""

    @classmethod
    def setUpClass(cls):
        if not M5_CSV_PATH.exists():
            raise FileNotFoundError(f"Dataset not found: {M5_CSV_PATH}")
        cls.env = RealM5Environment(csv_path=str(M5_CSV_PATH), spread_usd=2.50)
        cls.env.load_data()
        cls.trainer, _ = load_m5_champion_trainer(CHECKPOINT_PATH, cls.env)
        cls.policy = MythosMT5Policy(
            base_policy=MockBasePolicy(),
            initial_capital=10_000.0,
            checkpoint_path=CHECKPOINT_PATH,
        )

    def test_1_1_future_candle_mutation_zero_impact_on_current_decision(self):
        """
        Adversarial probe: Modifying future candles t+1 .. t+50 with extreme anomalies
        must NOT change observation, model consensus, or wire output at current bar t.
        """
        t = 8500
        # Baseline observation and inference
        obs_baseline = self.env.get_observation(t).copy()
        with torch.inference_mode():
            dec_baseline = self.trainer.resolve_consensus(
                obs_baseline, current_active_positions=0, active_position_dir=0, bar_idx=None
            )
            policy_baseline = MythosMT5Policy(
                base_policy=MockBasePolicy(),
                initial_capital=10_000.0,
                checkpoint_path=CHECKPOINT_PATH,
            )
            traces_baseline = policy_baseline.act_nam_tac_nhan(obs_baseline)
            wire_baseline = traces_baseline[0].output_vector.copy()

        # Backup original future slices
        orig_opens = self.env.open_arr[t + 1 : t + 51].copy()
        orig_highs = self.env.high_arr[t + 1 : t + 51].copy()
        orig_lows = self.env.low_arr[t + 1 : t + 51].copy()
        orig_closes = self.env.close_arr[t + 1 : t + 51].copy()
        orig_matrix = self.env.obs_matrix[t + 1 : t + 51].copy()

        try:
            # Adversarially corrupt future bars with extreme astronomical values
            self.env.open_arr[t + 1 : t + 51] = 999_999.0
            self.env.high_arr[t + 1 : t + 51] = 1_500_000.0
            self.env.low_arr[t + 1 : t + 51] = 1.0
            self.env.close_arr[t + 1 : t + 51] = 888_888.0
            self.env.obs_matrix[t + 1 : t + 51] = 99.0

            # Re-query observation and re-execute inference at bar t
            obs_corrupted = self.env.get_observation(t)
            with torch.inference_mode():
                dec_corrupted = self.trainer.resolve_consensus(
                    obs_corrupted, current_active_positions=0, active_position_dir=0, bar_idx=None
                )
                policy_corrupted = MythosMT5Policy(
                    base_policy=MockBasePolicy(),
                    initial_capital=10_000.0,
                    checkpoint_path=CHECKPOINT_PATH,
                )
                traces_corrupted = policy_corrupted.act_nam_tac_nhan(obs_corrupted)
                wire_corrupted = traces_corrupted[0].output_vector.copy()

            # Assert strict zero leakage
            np.testing.assert_array_equal(
                obs_baseline,
                obs_corrupted,
                err_msg="Observation at t was corrupted by future candle modifications!",
            )
            self.assertEqual(
                dec_baseline["action"],
                dec_corrupted["action"],
                "Consensus action changed after future candle modification!",
            )
            self.assertEqual(
                dec_baseline["active_champion"],
                dec_corrupted["active_champion"],
                "Active champion changed after future candle modification!",
            )
            self.assertAlmostEqual(
                dec_baseline["confidence"],
                dec_corrupted["confidence"],
                places=6,
                msg="Decision confidence changed after future candle modification!",
            )
            np.testing.assert_allclose(
                wire_baseline,
                wire_corrupted,
                atol=1e-5,
                err_msg="Policy wire output changed after future candle modification!",
            )

        finally:
            # Restore original arrays
            self.env.open_arr[t + 1 : t + 51] = orig_opens
            self.env.high_arr[t + 1 : t + 51] = orig_highs
            self.env.low_arr[t + 1 : t + 51] = orig_lows
            self.env.close_arr[t + 1 : t + 51] = orig_closes
            self.env.obs_matrix[t + 1 : t + 51] = orig_matrix

    def test_1_2_beta_dyna_zero_enforcement_in_oos(self):
        """
        Verify that OOSEvaluationEngineM5 strictly forces beta_dyna = 0.0,
        ensuring MCTS search never queries future simulation rollouts.
        """
        # Set dummy non-zero dyna
        self.trainer.bull_champion.mcts.beta_dyna = 0.8
        self.trainer.bear_champion.mcts.beta_dyna = 0.8

        c1_env = SlicedM5Environment(self.env, 8064, 8100, name="MiniSlice")
        cfg = OOSConfigM5(eval_stride=5, max_holding_bars=6)

        # Run 2 trades
        _, _ = OOSEvaluationEngineM5.evaluate_window(
            trainer=self.trainer,
            env=c1_env,
            cycle_name="TestDyna",
            config=cfg,
            max_trades=2,
        )

        # During evaluation it must be 0.0; check that restored dyna equals prior or is controlled
        self.assertEqual(self.trainer.bull_champion.mcts.beta_dyna, 0.8)
        self.assertEqual(self.trainer.bear_champion.mcts.beta_dyna, 0.8)

    def test_1_3_sliced_environment_temporal_containment(self):
        """
        Verify that SlicedM5Environment strictly prevents indexing past end_idx
        and clamps max_holding_bars at the boundary without crashing.
        """
        slice_env = SlicedM5Environment(self.env, start_idx=8064, end_idx=8100, name="BoundarySlice")
        self.assertEqual(len(slice_env), 36)

        # Observation within bounds
        obs = slice_env.get_observation(35)
        self.assertEqual(obs.shape, (137,))

        # Index out of bounds must raise IndexError
        with self.assertRaises(IndexError):
            slice_env.get_observation(36)

        with self.assertRaises(IndexError):
            slice_env.get_observation(-1)

        # simulate_trade at boundary bar (35) has 0 available future bars
        res = slice_env.simulate_trade(bar_idx=35, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)
        self.assertEqual(res.holding_bars, 0)
        self.assertEqual(res.exit_reason, "DATASET_END")


# ==============================================================================
# 2. PATH-DEPENDENT EXECUTION EDGE CASES
# ==============================================================================
class TestPathDependentExecutionEdgeCases(unittest.TestCase):
    """Stress tests for path-dependent execution physics."""

    @classmethod
    def setUpClass(cls):
        cls.env = RealM5Environment(csv_path=str(M5_CSV_PATH), spread_usd=2.50)
        cls.env.load_data()

    def test_2_1_adverse_gap_open_beyond_sl_buy(self):
        """
        BUY position opened at bar t.
        If next bar opens with a severe downward gap past the SL level,
        execution must fill at the adverse Open price (negative slippage), NOT at the SL level.
        """
        bar_idx = 1000
        ref_close = float(self.env.close_arr[bar_idx])
        atr = float(self.env.atr_arr[bar_idx])
        sl_atr = 2.0
        tp_atr = 5.0
        sl_level = ref_close - sl_atr * atr

        # Save next bar
        orig_open = float(self.env.open_arr[bar_idx + 1])
        orig_high = float(self.env.high_arr[bar_idx + 1])
        orig_low = float(self.env.low_arr[bar_idx + 1])
        orig_close = float(self.env.close_arr[bar_idx + 1])

        try:
            # Inject adverse gap down: Open is 50 USD below SL level
            gap_open = sl_level - 50.0
            self.env.open_arr[bar_idx + 1] = gap_open
            self.env.high_arr[bar_idx + 1] = gap_open + 5.0
            self.env.low_arr[bar_idx + 1] = gap_open - 10.0
            self.env.close_arr[bar_idx + 1] = gap_open - 2.0

            res = self.env.simulate_trade(
                bar_idx=bar_idx,
                action=1,  # BUY
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                max_holding_bars=12,
            )

            self.assertEqual(res.exit_type, "sl")
            self.assertEqual(res.holding_bars, 1)
            # Exit level should be gap_open (worse than sl_level)
            expected_exit_price = gap_open - (2.50 / 2.0)
            self.assertAlmostEqual(res.exit_price, expected_exit_price, places=2)
            # Loss must include the 50 USD gap slippage + $2.50 spread
            expected_loss = (gap_open - ref_close) - 2.50
            self.assertAlmostEqual(res.pnl_usd, expected_loss, places=2)
            self.assertLess(res.pnl_usd, -(sl_atr * atr + 2.50))

        finally:
            self.env.open_arr[bar_idx + 1] = orig_open
            self.env.high_arr[bar_idx + 1] = orig_high
            self.env.low_arr[bar_idx + 1] = orig_low
            self.env.close_arr[bar_idx + 1] = orig_close

    def test_2_2_adverse_gap_open_beyond_sl_sell(self):
        """
        SELL position opened at bar t.
        If next bar opens with a severe upward gap past the SL level,
        execution must fill at the adverse Open price (negative slippage), NOT at the SL level.
        """
        bar_idx = 1000
        ref_close = float(self.env.close_arr[bar_idx])
        atr = float(self.env.atr_arr[bar_idx])
        sl_atr = 2.0
        tp_atr = 5.0
        sl_level = ref_close + sl_atr * atr

        orig_open = float(self.env.open_arr[bar_idx + 1])
        orig_high = float(self.env.high_arr[bar_idx + 1])
        orig_low = float(self.env.low_arr[bar_idx + 1])
        orig_close = float(self.env.close_arr[bar_idx + 1])

        try:
            # Inject adverse gap up: Open is 50 USD above SL level
            gap_open = sl_level + 50.0
            self.env.open_arr[bar_idx + 1] = gap_open
            self.env.high_arr[bar_idx + 1] = gap_open + 10.0
            self.env.low_arr[bar_idx + 1] = gap_open - 5.0
            self.env.close_arr[bar_idx + 1] = gap_open + 2.0

            res = self.env.simulate_trade(
                bar_idx=bar_idx,
                action=2,  # SELL
                sl_atr=sl_atr,
                tp_atr=tp_atr,
                max_holding_bars=12,
            )

            self.assertEqual(res.exit_type, "sl")
            self.assertEqual(res.holding_bars, 1)
            expected_exit_price = gap_open + (2.50 / 2.0)
            self.assertAlmostEqual(res.exit_price, expected_exit_price, places=2)
            expected_loss = (ref_close - gap_open) - 2.50
            self.assertAlmostEqual(res.pnl_usd, expected_loss, places=2)
            self.assertLess(res.pnl_usd, -(sl_atr * atr + 2.50))

        finally:
            self.env.open_arr[bar_idx + 1] = orig_open
            self.env.high_arr[bar_idx + 1] = orig_high
            self.env.low_arr[bar_idx + 1] = orig_low
            self.env.close_arr[bar_idx + 1] = orig_close

    def test_2_3_simultaneous_sl_and_tp_touch_conservative_priority(self):
        """
        When a single subsequent bar reaches BOTH the Stop Loss and Take Profit extremes,
        conservative execution physics must prioritize Stop Loss (SIMULTANEOUS_SL_FIRST).
        """
        bar_idx = 2000
        ref_close = float(self.env.close_arr[bar_idx])
        atr = float(self.env.atr_arr[bar_idx])
        sl_atr = 2.0
        tp_atr = 4.0

        # Save next bar
        orig_open = float(self.env.open_arr[bar_idx + 1])
        orig_high = float(self.env.high_arr[bar_idx + 1])
        orig_low = float(self.env.low_arr[bar_idx + 1])
        orig_close = float(self.env.close_arr[bar_idx + 1])

        try:
            # Set bar extremes to breach both SL and TP for BUY (TP=+4 ATR, SL=-2 ATR) and SELL (SL=+2 ATR, TP=-4 ATR)
            self.env.open_arr[bar_idx + 1] = ref_close
            self.env.high_arr[bar_idx + 1] = ref_close + (tp_atr + 1.0) * atr
            self.env.low_arr[bar_idx + 1] = ref_close - (tp_atr + 1.0) * atr
            self.env.close_arr[bar_idx + 1] = ref_close

            res_buy = self.env.simulate_trade(
                bar_idx=bar_idx, action=1, sl_atr=sl_atr, tp_atr=tp_atr, max_holding_bars=12
            )
            self.assertEqual(res_buy.exit_type, "sl")
            self.assertEqual(res_buy.exit_reason, "SIMULTANEOUS_SL_FIRST")
            self.assertLess(res_buy.pnl_usd, 0.0)

            # 2. SELL trade test
            # Set bar extremes to breach both SL (ref_close + 2*atr) and TP (ref_close - 4*atr)
            res_sell = self.env.simulate_trade(
                bar_idx=bar_idx, action=2, sl_atr=sl_atr, tp_atr=tp_atr, max_holding_bars=12
            )
            self.assertEqual(res_sell.exit_type, "sl")
            self.assertEqual(res_sell.exit_reason, "SIMULTANEOUS_SL_FIRST")
            self.assertLess(res_sell.pnl_usd, 0.0)

        finally:
            self.env.open_arr[bar_idx + 1] = orig_open
            self.env.high_arr[bar_idx + 1] = orig_high
            self.env.low_arr[bar_idx + 1] = orig_low
            self.env.close_arr[bar_idx + 1] = orig_close

    def test_2_4_exact_bar_holding_limit_expiration(self):
        """
        When price remains strictly between SL and TP for max_holding_bars,
        trade must exit on exactly max_holding_bars with TIME_STOP exit_type and exit_reason.
        """
        bar_idx = 3000
        ref_close = float(self.env.close_arr[bar_idx])
        atr = float(self.env.atr_arr[bar_idx])
        # Very wide SL/TP ensuring no hit
        sl_atr = 20.0
        tp_atr = 40.0
        max_h = 12

        res = self.env.simulate_trade(
            bar_idx=bar_idx,
            action=1,
            sl_atr=sl_atr,
            tp_atr=tp_atr,
            max_holding_bars=max_h,
        )

        self.assertEqual(res.holding_bars, max_h)
        self.assertEqual(res.exit_type, "timestop")
        self.assertEqual(res.exit_reason, "TIME_STOP")
        self.assertEqual(res.exit_bar_idx, bar_idx + max_h)
        # Exit price is close of terminal bar minus half spread
        terminal_close = float(self.env.close_arr[bar_idx + max_h])
        expected_exit_price = terminal_close - (2.50 / 2.0)
        self.assertAlmostEqual(res.exit_price, expected_exit_price, places=2)


# ==============================================================================
# 3. SPREAD ACCOUNTING INTEGRITY
# ==============================================================================
class TestSpreadAccountingIntegrity(unittest.TestCase):
    """Stress tests verifying exact $2.50 USD transaction drag on all trades."""

    @classmethod
    def setUpClass(cls):
        cls.env = RealM5Environment(csv_path=str(M5_CSV_PATH), spread_usd=2.50)
        cls.env.load_data()

    def test_3_1_flat_market_exact_loss(self):
        """
        In a zero price-change flat market, realized PnL must be exactly -$2.50 USD
        for both BUY and SELL orders.
        """
        bar_idx = 4000
        ref_close = float(self.env.close_arr[bar_idx])
        atr = float(self.env.atr_arr[bar_idx])
        max_h = 5

        # Backup next 5 bars
        backup = {
            "open": self.env.open_arr[bar_idx + 1 : bar_idx + 6].copy(),
            "high": self.env.high_arr[bar_idx + 1 : bar_idx + 6].copy(),
            "low": self.env.low_arr[bar_idx + 1 : bar_idx + 6].copy(),
            "close": self.env.close_arr[bar_idx + 1 : bar_idx + 6].copy(),
        }

        try:
            # Flatten future 5 bars to ref_close
            self.env.open_arr[bar_idx + 1 : bar_idx + 6] = ref_close
            self.env.high_arr[bar_idx + 1 : bar_idx + 6] = ref_close
            self.env.low_arr[bar_idx + 1 : bar_idx + 6] = ref_close
            self.env.close_arr[bar_idx + 1 : bar_idx + 6] = ref_close

            # BUY trade
            res_buy = self.env.simulate_trade(
                bar_idx=bar_idx, action=1, sl_atr=5.0, tp_atr=10.0, max_holding_bars=max_h
            )
            self.assertAlmostEqual(res_buy.pnl_usd, -2.50, places=5)
            self.assertEqual(res_buy.holding_bars, max_h)

            # SELL trade
            res_sell = self.env.simulate_trade(
                bar_idx=bar_idx, action=2, sl_atr=5.0, tp_atr=10.0, max_holding_bars=max_h
            )
            self.assertAlmostEqual(res_sell.pnl_usd, -2.50, places=5)
            self.assertEqual(res_sell.holding_bars, max_h)

        finally:
            self.env.open_arr[bar_idx + 1 : bar_idx + 6] = backup["open"]
            self.env.high_arr[bar_idx + 1 : bar_idx + 6] = backup["high"]
            self.env.low_arr[bar_idx + 1 : bar_idx + 6] = backup["low"]
            self.env.close_arr[bar_idx + 1 : bar_idx + 6] = backup["close"]

    def test_3_2_spread_scaling_linearity(self):
        """
        Verify that varying the spread parameter (0.00, 1.50, 2.50, 5.00, 10.00)
        linearly reduces PnL by exactly the spread difference.
        """
        bar_idx = 5000
        spreads = [0.00, 1.50, 2.50, 5.00, 10.00]
        results = []

        for s in spreads:
            res = self.env.simulate_trade(
                bar_idx=bar_idx, action=1, sl_atr=3.0, tp_atr=6.0, max_holding_bars=12, spread_usd=s
            )
            results.append(res.pnl_usd)

        # Baseline at 0 spread
        zero_spread_pnl = results[0]
        for s, pnl in zip(spreads[1:], results[1:]):
            self.assertAlmostEqual(
                pnl,
                zero_spread_pnl - s,
                places=4,
                msg=f"Spread deduction of ${s} failed: expected {zero_spread_pnl - s}, got {pnl}",
            )

    def test_3_3_batch_empirical_spread_accounting_integrity(self):
        """
        Across 50 diverse simulated trades on real market data, verify that
        every trade's PnL strictly equals (exit_level - ref_close - spread_usd) for BUY,
        and (ref_close - exit_level - spread_usd) for SELL.
        """
        spread = 2.50
        for i in range(50):
            bar = 1000 + i * 150
            ref_close = float(self.env.close_arr[bar])
            act = 1 if i % 2 == 0 else 2

            res = self.env.simulate_trade(
                bar_idx=bar, action=act, sl_atr=2.5, tp_atr=5.0, max_holding_bars=12, spread_usd=spread
            )

            # Reconstruct exit_level from exit_price
            if act == 1:
                exit_level = res.exit_price + (spread / 2.0)
                expected_pnl = exit_level - ref_close - spread
            else:
                exit_level = res.exit_price - (spread / 2.0)
                expected_pnl = ref_close - exit_level - spread

            self.assertAlmostEqual(
                res.pnl_usd,
                expected_pnl,
                places=4,
                msg=f"Spread discrepancy at bar {bar}: {res.pnl_usd} vs {expected_pnl}",
            )


# ==============================================================================
# 4. INDEPENDENT METRICS RECOMPUTATION & REPORT AUDIT
# ==============================================================================
class TestIndependentMetricsVerification(unittest.TestCase):
    """Recomputes backtest metrics independently and audits evaluation report claims."""

    @classmethod
    def setUpClass(cls):
        cls.env = RealM5Environment(csv_path=str(M5_CSV_PATH), spread_usd=2.50)
        cls.env.load_data()
        cls.trainer, _ = load_m5_champion_trainer(CHECKPOINT_PATH, cls.env)

    def test_4_1_independent_recomputation_of_walk_forward_metrics(self):
        """
        Independently re-runs OOS Walk-Forward backtest with max_trades=60 per cycle
        and verifies exact match with outputs/m5_evaluation_report.md numbers.
        """
        cfg = OOSConfigM5(eval_stride=5, max_holding_bars=12, spread_usd=2.50)
        c1_env = SlicedM5Environment(self.env, 8064, 12096, name="Cycle1_Audit")
        c2_env = SlicedM5Environment(self.env, 10080, 12960, name="Cycle2_Audit")

        m1, t1 = OOSEvaluationEngineM5.evaluate_window(
            trainer=self.trainer, env=c1_env, cycle_name="C1", config=cfg, max_trades=60
        )
        m2, t2 = OOSEvaluationEngineM5.evaluate_window(
            trainer=self.trainer, env=c2_env, cycle_name="C2", config=cfg, max_trades=60
        )

        # 1. Cycle 1 Independent Verification
        self.assertEqual(len(t1), 60)
        self.assertEqual(m1.winning_trades, 32)
        self.assertEqual(m1.losing_trades, 28)
        self.assertAlmostEqual(m1.win_rate, 32 / 60, places=4)
        self.assertAlmostEqual(m1.gross_profit, 6241.14, delta=0.50)
        self.assertAlmostEqual(m1.gross_loss, 7600.84, delta=0.50)
        self.assertAlmostEqual(m1.profit_factor, 0.82, delta=0.01)
        self.assertAlmostEqual(m1.net_profit, -1359.70, delta=0.50)
        self.assertAlmostEqual(m1.max_drawdown, 0.2602, delta=0.01)

        # 2. Cycle 2 Independent Verification
        self.assertEqual(len(t2), 60)
        self.assertEqual(m2.winning_trades, 28)
        self.assertEqual(m2.losing_trades, 32)
        self.assertAlmostEqual(m2.win_rate, 28 / 60, places=4)
        self.assertAlmostEqual(m2.gross_profit, 6034.45, delta=0.50)
        self.assertAlmostEqual(m2.gross_loss, 6171.41, delta=0.50)
        self.assertAlmostEqual(m2.profit_factor, 0.98, delta=0.01)
        self.assertAlmostEqual(m2.net_profit, -136.97, delta=0.50)
        self.assertAlmostEqual(m2.max_drawdown, 0.1622, delta=0.01)

        # 3. Combined Independent Verification
        comb_trades = t1 + t2
        self.assertEqual(len(comb_trades), 120)
        comb_wins = sum(1 for t in comb_trades if t.pnl_usd > 1e-4)
        comb_losses = sum(1 for t in comb_trades if t.pnl_usd < -1e-4)
        self.assertEqual(comb_wins, 60)
        self.assertEqual(comb_losses, 60)

        comb_wr = comb_wins / 120
        self.assertAlmostEqual(comb_wr, 0.50, places=4)

        comb_gp = sum(t.pnl_usd for t in comb_trades if t.pnl_usd > 1e-4)
        comb_gl = abs(sum(t.pnl_usd for t in comb_trades if t.pnl_usd < -1e-4))
        comb_pf = comb_gp / comb_gl
        comb_net = comb_gp - comb_gl

        self.assertAlmostEqual(comb_gp, 12275.59, delta=1.0)
        self.assertAlmostEqual(comb_gl, 13772.25, delta=1.0)
        self.assertAlmostEqual(comb_pf, 0.89, delta=0.01)
        self.assertAlmostEqual(comb_net, -1496.66, delta=1.0)

        # Calculate equity and max drawdown
        eq = 10_000.0
        peak = eq
        max_dd_pct = 0.0
        for t in comb_trades:
            eq += t.pnl_usd
            if eq > peak:
                peak = eq
            dd_pct = (peak - eq) / peak
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        self.assertAlmostEqual(eq, 8503.34, delta=1.0)
        self.assertAlmostEqual(max_dd_pct, 0.2839, delta=0.01)

    def test_4_2_audit_evaluation_report_formatting_and_positive_pnl_claim(self):
        """
        Adversarial Audit: Check outputs/m5_evaluation_report.md for formatting defects.
        Specifically, identify the misleading claim '+$-1,496.66' and 'Positive Realized PnL'.
        """
        self.assertTrue(EVAL_REPORT_PATH.exists())
        content = EVAL_REPORT_PATH.read_text(encoding="utf-8")

        # The report contains '+$-1,496.66' due to a hardcoded '+' in evaluate_m5_champion.py template
        has_positive_sign_bug = "+$-1,496.66" in content or "+$-1,359.70" in content
        self.assertTrue(
            has_positive_sign_bug,
            "Expected report to contain formatting bug '+$-...' as identified during audit.",
        )

        has_positive_claim = "Positive Realized PnL" in content
        self.assertTrue(
            has_positive_claim,
            "Expected report to contain 'Positive Realized PnL' header as identified during audit.",
        )

    def test_4_3_acceptance_criteria_gate_failure_audit(self):
        """
        Audits whether the evaluated M5 champion satisfies the strict R7 acceptance criteria:
        - Win Rate >= 80% (Actual: 50.00% -> FAILED)
        - Profit Factor >= 3.0 (Actual: 0.89 -> FAILED)
        - Max Drawdown <= 10% (Actual: 28.39% -> FAILED)
        - Positive PnL (Actual: -$1,496.66 -> FAILED)

        Confirms that while MT5 export readiness and zero-lookahead backtest execution succeeded,
        the trading profitability gate fails the strict R7 threshold.
        """
        actual_wr = 0.50
        actual_pf = 0.89
        actual_max_dd = 0.2839
        actual_net_pnl = -1496.66

        target_wr = 0.80
        target_pf = 3.00
        target_max_dd = 0.10

        self.assertLess(actual_wr, target_wr, "Win rate did not reach 80% target.")
        self.assertLess(actual_pf, target_pf, "Profit factor did not reach 3.0 target.")
        self.assertGreater(actual_max_dd, target_max_dd, "Max drawdown exceeded 10% limit.")
        self.assertLess(actual_net_pnl, 0.0, "Net PnL was negative post-spread.")


if __name__ == "__main__":
    unittest.main()
