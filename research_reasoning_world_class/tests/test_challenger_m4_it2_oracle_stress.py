"""
Milestone 4 Iteration 2 Empirical Oracle Adversarial Stress Harness.
File: research_reasoning_world_class/tests/test_challenger_m4_it2_oracle_stress.py

Role: teamwork_challenger_m4_it2_2 (Empirical Oracle Challenger 2)

Empirical Challenge Objectives:
1. Boundary Invariant & Zero-Peeking Trap:
   - Prove that bar_idx == total_bars - 1 NEVER reads into the next slice candle,
     even when adjacent slice candles are poisoned with NaNs/sentinels.
   - Prove that holding_bars == 0, exit_reason == 'DATASET_END', and exit_bar_idx == total_bars - 1.
2. Penultimate Boundary Clamping:
   - Prove that bar_idx == total_bars - 2 clamps safe_max_holding to 1,
     strictly confining candle access to index total_bars - 1 and NEVER reaching total_bars.
3. Multi-Slice Coordinate Space Remapping:
   - Prove that across ALL 4 Walk-Forward slices (Cycle 1 Train/OOS, Cycle 2 Train/OOS),
     every trade produces 0 <= res.exit_bar_idx < total_bars.
   - Prove s.get_observation(res.exit_bar_idx) matches master.get_observation(start_idx + res.exit_bar_idx) bit-for-bit.
4. Strict Bounds Ingress Checking:
   - Prove that negative indices and out-of-range indices strictly raise IndexError.
   - Prove invalid actions and SL/TP distances raise ValueError.
5. Extreme max_holding_bars Parameter Values:
   - Stress-test max_holding_bars in {0, 1, 999999}.
6. Dataclass Immutability Invariant:
   - Prove returned TradeResult adheres to frozen dataclass semantics without mutation side-effects.
"""

from dataclasses import FrozenInstanceError
import math
import os
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REASONING_DIR = ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
    ExitType,
)
from scripts.train_mythos_muzero_coevolution import SlicedM1Environment

CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"


@pytest.fixture(scope="module")
def loaded_master_env():
    """Loads the master RealM1Environment once for testing."""
    if not os.path.exists(CSV_PATH):
        pytest.skip(f"Historical file not found: {CSV_PATH}")
    env = RealM1Environment(csv_path=CSV_PATH, spread_usd=2.50)
    total = env.load_data("2025.01.15", "2025.02.28")
    assert total == 64785, f"Expected 64,785 bars, got {total}"
    return env


class TestAdversarialSliceBoundaryAndZeroPeeking:
    """Stress-tests slice boundary handling with active poison traps."""

    def test_terminal_bar_zero_peeking_with_nan_poison(self, loaded_master_env):
        """
        Adversarial Test:
        Slice Cycle 1 Train is [0, 40316). The adjacent candle is 40316.
        Poison master candle 40316..40400 with NaN values.
        If SlicedM1Environment peeks across the slice boundary, NaN comparisons
        or calculations will pollute pnl_usd, holding_bars, or throw unexpected errors.
        """
        master = loaded_master_env
        start_idx = 0
        end_idx = 40316
        sliced = SlicedM1Environment(master, start_idx, end_idx, name="Cycle 1 Train")
        last_bar_idx = sliced.total_bars - 1  # 40315

        # Backup master arrays starting at the adjacent slice boundary
        poison_start = end_idx  # 40316
        poison_len = 100
        saved_open = master.open_arr[poison_start : poison_start + poison_len].copy()
        saved_high = master.high_arr[poison_start : poison_start + poison_len].copy()
        saved_low = master.low_arr[poison_start : poison_start + poison_len].copy()
        saved_close = master.close_arr[poison_start : poison_start + poison_len].copy()

        try:
            # Inject poison into adjacent slice
            master.open_arr[poison_start : poison_start + poison_len] = np.nan
            master.high_arr[poison_start : poison_start + poison_len] = np.nan
            master.low_arr[poison_start : poison_start + poison_len] = np.nan
            master.close_arr[poison_start : poison_start + poison_len] = np.nan

            # 1. Test BUY at terminal bar
            res_buy = sliced.simulate_trade(
                bar_idx=last_bar_idx,
                action=1,
                sl_atr=2.0,
                tp_atr=3.0,
                max_holding_bars=60,
            )
            assert res_buy.holding_bars == 0, f"BUY expected 0 holding bars, got {res_buy.holding_bars}"
            assert res_buy.exit_reason == "DATASET_END", f"BUY expected DATASET_END, got {res_buy.exit_reason}"
            assert res_buy.exit_bar_idx == last_bar_idx, f"BUY expected exit_bar_idx {last_bar_idx}, got {res_buy.exit_bar_idx}"
            assert not math.isnan(res_buy.pnl_usd), "BUY PnL corrupted by NaN poison!"
            assert not math.isnan(res_buy.return_atr), "BUY return_atr corrupted by NaN poison!"
            assert res_buy.exit_type == ExitType("timestop")
            # Must successfully retrieve observation
            obs_buy = sliced.get_observation(res_buy.exit_bar_idx)
            assert obs_buy.shape == (137,)
            assert not np.isnan(obs_buy).any()

            # 2. Test SELL at terminal bar
            res_sell = sliced.simulate_trade(
                bar_idx=last_bar_idx,
                action=2,
                sl_atr=2.0,
                tp_atr=3.0,
                max_holding_bars=60,
            )
            assert res_sell.holding_bars == 0, f"SELL expected 0 holding bars, got {res_sell.holding_bars}"
            assert res_sell.exit_reason == "DATASET_END", f"SELL expected DATASET_END, got {res_sell.exit_reason}"
            assert res_sell.exit_bar_idx == last_bar_idx, f"SELL expected exit_bar_idx {last_bar_idx}, got {res_sell.exit_bar_idx}"
            assert not math.isnan(res_sell.pnl_usd), "SELL PnL corrupted by NaN poison!"
            assert not math.isnan(res_sell.return_atr), "SELL return_atr corrupted by NaN poison!"
            assert res_sell.exit_type == ExitType("timestop")
            obs_sell = sliced.get_observation(res_sell.exit_bar_idx)
            assert obs_sell.shape == (137,)
            assert not np.isnan(obs_sell).any()

            # 3. Test penultimate bar (last_bar_idx - 1 = 40314)
            # safe_max_holding should clamp to max(0, min(60, 40316 - 1 - 40314)) = 1 bar.
            # That 1 bar is candle 40315 (which is clean). It must NEVER read poisoned candle 40316!
            penultimate_idx = last_bar_idx - 1
            res_penultimate = sliced.simulate_trade(
                bar_idx=penultimate_idx,
                action=1,
                sl_atr=2.0,
                tp_atr=3.0,
                max_holding_bars=60,
            )
            assert res_penultimate.holding_bars in (0, 1)
            assert 0 <= res_penultimate.exit_bar_idx < sliced.total_bars
            assert res_penultimate.exit_bar_idx <= last_bar_idx
            assert not math.isnan(res_penultimate.pnl_usd), "Penultimate trade touched poisoned candle 40316!"
            assert not math.isnan(res_penultimate.return_atr)

        finally:
            # Restore pristine candle arrays in master
            master.open_arr[poison_start : poison_start + poison_len] = saved_open
            master.high_arr[poison_start : poison_start + poison_len] = saved_high
            master.low_arr[poison_start : poison_start + poison_len] = saved_low
            master.close_arr[poison_start : poison_start + poison_len] = saved_close

    @pytest.mark.parametrize(
        "slice_name,start,end",
        [
            ("Cycle 1 Train", 0, 40316),
            ("Cycle 1 OOS", 40316, 60465),
            ("Cycle 2 Train", 10079, 50392),
            ("Cycle 2 OOS", 50392, 64785),
        ],
    )
    def test_all_slices_terminal_boundary_zero_peeking(self, loaded_master_env, slice_name, start, end):
        """Verify terminal boundary invariant holds uniformly across all Walk-Forward slices."""
        master = loaded_master_env
        sliced = SlicedM1Environment(master, start, end, name=slice_name)
        last_bar_idx = sliced.total_bars - 1

        for action in (1, 2):
            res = sliced.simulate_trade(
                bar_idx=last_bar_idx,
                action=action,
                sl_atr=2.0,
                tp_atr=3.0,
                max_holding_bars=60,
            )
            assert res.holding_bars == 0
            assert res.exit_reason == "DATASET_END"
            assert res.exit_bar_idx == last_bar_idx
            assert 0 <= res.exit_bar_idx < sliced.total_bars
            obs = sliced.get_observation(res.exit_bar_idx)
            assert obs.shape == (137,)


class TestCoordinateSpaceRemappingStress:
    """Stress-tests coordinate space remapping across non-zero start_idx offsets."""

    @pytest.mark.parametrize(
        "slice_name,start,end",
        [
            ("Cycle 1 OOS", 40316, 60465),
            ("Cycle 2 Train", 10079, 50392),
            ("Cycle 2 OOS", 50392, 64785),
            ("Mid-Dataset Window", 25000, 35000),
        ],
    )
    def test_coordinate_space_sweep_invariants(self, loaded_master_env, slice_name, start, end):
        """
        Verify that across a broad sweep of bars in slices with start_idx > 0:
        1. res.exit_bar_idx is strictly within [0, sliced.total_bars)
        2. res.exit_bar_idx == bar_idx + res.holding_bars
        3. sliced.get_observation(res.exit_bar_idx) matches master.get_observation(start + res.exit_bar_idx)
        """
        master = loaded_master_env
        sliced = SlicedM1Environment(master, start, end, name=slice_name)
        total = sliced.total_bars

        test_bars = [
            0,
            1,
            5,
            10,
            50,
            100,
            500,
            total // 4,
            total // 2,
            3 * total // 4,
            total - 100,
            total - 61,
            total - 60,
            total - 30,
            total - 10,
            total - 5,
            total - 2,
            total - 1,
        ]

        for bar_idx in test_bars:
            for action in (1, 2):
                res = sliced.simulate_trade(
                    bar_idx=bar_idx,
                    action=action,
                    sl_atr=2.5,
                    tp_atr=5.0,
                    max_holding_bars=60,
                )
                # Invariant 1: exit_bar_idx strictly within slice bounds
                assert 0 <= res.exit_bar_idx < total, (
                    f"[{slice_name}] bar {bar_idx}: exit_bar_idx {res.exit_bar_idx} not in [0, {total})"
                )
                # Invariant 2: exit_bar_idx matches bar_idx + holding_bars
                assert res.exit_bar_idx == bar_idx + res.holding_bars, (
                    f"[{slice_name}] bar {bar_idx}: exit_bar_idx {res.exit_bar_idx} != bar {bar_idx} + holding {res.holding_bars}"
                )
                # Invariant 3: get_observation succeeds within slice and matches master bit-for-bit
                slice_obs = sliced.get_observation(res.exit_bar_idx)
                master_obs = master.get_observation(start + res.exit_bar_idx)
                assert slice_obs.shape == (137,)
                assert np.array_equal(slice_obs, master_obs), (
                    f"[{slice_name}] bar {bar_idx}: observation at exit_bar_idx mismatch with master"
                )
                # Invariant 4: ATR accessor matches
                assert sliced.get_atr(res.exit_bar_idx) == master.get_atr(start + res.exit_bar_idx)
                # Invariant 5: Timestamp matches
                assert sliced.get_timestamp(res.exit_bar_idx) == master.get_timestamp(start + res.exit_bar_idx)


class TestDefensiveBoundsIngressOracle:
    """Stress-tests defensive bounds checking and error propagation."""

    def test_out_of_bounds_indices_raise_indexerror(self, loaded_master_env):
        master = loaded_master_env
        sliced = SlicedM1Environment(master, 40316, 60465, name="Cycle 1 OOS")
        total = sliced.total_bars

        invalid_indices = [
            -1,
            -2,
            -10,
            -100,
            -999999,
            total,
            total + 1,
            total + 10,
            total + 100,
            total + 999999,
        ]

        for inv_idx in invalid_indices:
            with pytest.raises(IndexError):
                sliced.simulate_trade(bar_idx=inv_idx, action=1, sl_atr=2.0, tp_atr=3.0)
            with pytest.raises(IndexError):
                sliced.get_observation(bar_idx=inv_idx)

    def test_invalid_parameters_raise_valueerror(self, loaded_master_env):
        master = loaded_master_env
        sliced = SlicedM1Environment(master, 40316, 60465, name="Cycle 1 OOS")

        # Invalid actions
        for act in (0, 3, -1, 99):
            with pytest.raises(ValueError):
                sliced.simulate_trade(bar_idx=10, action=act, sl_atr=2.0, tp_atr=3.0)

        # Invalid sl_atr / tp_atr <= 0
        with pytest.raises(ValueError):
            sliced.simulate_trade(bar_idx=10, action=1, sl_atr=0.0, tp_atr=3.0)
        with pytest.raises(ValueError):
            sliced.simulate_trade(bar_idx=10, action=1, sl_atr=-1.0, tp_atr=3.0)
        with pytest.raises(ValueError):
            sliced.simulate_trade(bar_idx=10, action=1, sl_atr=2.0, tp_atr=0.0)
        with pytest.raises(ValueError):
            sliced.simulate_trade(bar_idx=10, action=1, sl_atr=2.0, tp_atr=-2.0)


class TestExtremeParametersAndImmutabilityOracle:
    """Stress-tests extreme parameters and immutability semantics."""

    def test_extreme_max_holding_bars(self, loaded_master_env):
        master = loaded_master_env
        sliced = SlicedM1Environment(master, 40316, 60465, name="Cycle 1 OOS")

        # max_holding_bars = 0: must return 0 holding bars immediately
        res_0 = sliced.simulate_trade(bar_idx=100, action=1, sl_atr=2.0, tp_atr=3.0, max_holding_bars=0)
        assert res_0.holding_bars == 0
        assert res_0.exit_reason == "DATASET_END"
        assert res_0.exit_bar_idx == 100

        # max_holding_bars = 1
        res_1 = sliced.simulate_trade(bar_idx=100, action=1, sl_atr=2.0, tp_atr=3.0, max_holding_bars=1)
        assert res_1.holding_bars in (0, 1)
        assert res_1.exit_bar_idx in (100, 101)

        # max_holding_bars = 1_000_000 (huge) near boundary
        near_end = sliced.total_bars - 5
        res_huge = sliced.simulate_trade(bar_idx=near_end, action=1, sl_atr=2.0, tp_atr=3.0, max_holding_bars=1_000_000)
        # safe_max_holding is clamped to total_bars - 1 - near_end = 4
        assert res_huge.holding_bars <= 4
        assert res_huge.exit_bar_idx <= sliced.total_bars - 1
        assert 0 <= res_huge.exit_bar_idx < sliced.total_bars
        obs = sliced.get_observation(res_huge.exit_bar_idx)
        assert obs.shape == (137,)

    def test_trade_result_dataclass_frozen_immutability(self, loaded_master_env):
        master = loaded_master_env
        sliced = SlicedM1Environment(master, 40316, 60465, name="Cycle 1 OOS")
        res = sliced.simulate_trade(bar_idx=10, action=1, sl_atr=2.0, tp_atr=3.0)

        assert isinstance(res, TradeResult)
        assert isinstance(res.exit_bar_idx, int)
        assert isinstance(res.holding_bars, int)

        # Attempting in-place mutation on frozen TradeResult MUST raise FrozenInstanceError
        with pytest.raises(FrozenInstanceError):
            res.exit_bar_idx = 999
        with pytest.raises(FrozenInstanceError):
            res.holding_bars = 999
