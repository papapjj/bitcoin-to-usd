"""
Adversarial Stress Test Suite for Milestone 1: M5 Candlestick Environment and Data Ingestion
Target Modules:
- research_reasoning_world_class/src/real_m5_environment.py
- scripts/export_m5_dataset.py
- src/self_growing_trader/features.py

Author: challenger_m1_2
Milestone: Milestone 1 (M5 Candlestick Environment and Data Ingestion)
"""

from __future__ import annotations

import dataclasses
import os
import sys
import tempfile
import time
from typing import List

import numpy as np
import pandas as pd
import pytest

# Path setup to import repository modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from research_reasoning_world_class.src.real_m5_environment import (
    ExitType,
    PositionSlot,
    RealM5Environment,
    TradeResult,
)
from scripts.export_m5_dataset import (
    detect_delimiter,
    load_raw_candles,
    normalize_candle_columns,
    resample_m1_to_m5,
)
from self_growing_trader.protocol_contract import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    POSITION_FEATURE_COUNT,
    POSITION_OFFSET,
)

DEFAULT_M5_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm_M5.csv"
FALLBACK_M5_CSV_PATH = os.path.join(repo_root, "outputs", "BTCUSDm_M5.csv")


def _generate_synthetic_csv(
    file_path: str,
    bar_count: int = 320,
    base_price: float = 100000.0,
    delimiter: str = ",",
    header_style: str = "standard",
    start_date: str = "2025.01.14 00:00:00",
) -> None:
    """Helper to generate clean synthetic M5 datasets with custom delimiter and header styles."""
    dt_start = pd.to_datetime(start_date.replace(".", "-"))
    lines = []

    if header_style == "standard":
        lines.append("Datetime,Open,High,Low,Close,TickVol,Vol,Spread")
    elif header_style == "tab_meta":
        lines.append("<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>")
    elif header_style == "semicolon":
        lines.append("Date;Time;Open;High;Low;Close;TickVol;Vol;Spread")
    elif header_style == "spaced_brackets":
        lines.append(" <date> , <time> , <open> , <high> , <low> , <close> , <tickvol> , <vol> , <spread> ")

    for i in range(bar_count):
        curr_dt = dt_start + pd.Timedelta(minutes=5 * i)
        date_str = curr_dt.strftime("%Y.%m.%d")
        time_str = curr_dt.strftime("%H:%M:%S")
        dt_full = f"{date_str} {time_str}"

        open_p = base_price
        high_p = base_price + 50.0
        low_p = base_price - 50.0
        close_p = base_price
        tickvol = 100
        vol = 0
        spread = 250

        if header_style == "standard":
            lines.append(f"{dt_full},{open_p:.2f},{high_p:.2f},{low_p:.2f},{close_p:.2f},{tickvol},{vol},{spread}")
        elif header_style == "tab_meta":
            lines.append(f"{date_str}\t{time_str}\t{open_p:.2f}\t{high_p:.2f}\t{low_p:.2f}\t{close_p:.2f}\t{tickvol}\t{vol}\t{spread}")
        elif header_style == "semicolon":
            lines.append(f"{date_str};{time_str};{open_p:.2f};{high_p:.2f};{low_p:.2f};{close_p:.2f};{tickvol};{vol};{spread}")
        elif header_style == "spaced_brackets":
            lines.append(f" {date_str} , {time_str} , {open_p:.2f} , {high_p:.2f} , {low_p:.2f} , {close_p:.2f} , {tickvol} , {vol} , {spread} ")

    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# Section 1: Ingestion Robustness & Hostile Header / Format Stress Tests
# ============================================================================

class TestIngestionRobustnessAdversarial:
    """Stress-tests CSV/TSV loading against malformed headers, corrupt data, and boundary files."""

    def test_missing_mandatory_price_column_raises_value_error(self, tmp_path):
        """Omitting required price columns (e.g. Close) must raise ValueError, not crash downstream."""
        bad_csv = tmp_path / "missing_close.csv"
        bad_csv.write_text(
            "Datetime,Open,High,Low,TickVol,Vol,Spread\n"
            "2025.01.15 00:00:00,100000,100050,99950,100,0,250\n",
            encoding="utf-8",
        )
        env = RealM5Environment(csv_path=str(bad_csv))
        with pytest.raises(ValueError, match="Missing required price column"):
            env.load_data("2025.01.15", "2025.01.15")

    def test_missing_datetime_column_raises_value_error(self, tmp_path):
        """Omitting Datetime/Date/Time must raise ValueError during header normalization."""
        bad_csv = tmp_path / "missing_datetime.csv"
        bad_csv.write_text(
            "Open,High,Low,Close,TickVol,Vol,Spread\n"
            "100000,100050,99950,100000,100,0,250\n",
            encoding="utf-8",
        )
        env = RealM5Environment(csv_path=str(bad_csv))
        with pytest.raises(ValueError, match="Unable to resolve Datetime column"):
            env.load_data("2025.01.15", "2025.01.15")

    def test_semicolon_delimited_csv_ingestion(self, tmp_path):
        """Verifies semicolon delimited datasets are auto-detected and loaded correctly."""
        semi_csv = tmp_path / "semicolon.csv"
        _generate_synthetic_csv(str(semi_csv), bar_count=310, header_style="semicolon")

        env = RealM5Environment(csv_path=str(semi_csv))
        bars = env.load_data("2025.01.14", "2025.01.15")
        assert bars > 0
        obs = env.get_observation(0)
        assert obs.shape == (137,)
        assert np.all(np.isfinite(obs))

    def test_bracketed_headers_without_surrounding_space(self, tmp_path):
        """Verifies standard MT5 bracketed headers <DATE>, <TIME>, <OPEN> parse cleanly."""
        tsv = tmp_path / "meta_brackets.tsv"
        _generate_synthetic_csv(str(tsv), bar_count=310, header_style="tab_meta")

        env = RealM5Environment(csv_path=str(tsv))
        bars = env.load_data("2025.01.14", "2025.01.15")
        assert bars > 0
        obs = env.get_observation(0)
        assert obs.shape == (137,)

    def test_spaced_header_brackets_vulnerability(self, tmp_path):
        """
        EMPIRICAL FINDING / BUG DEMONSTRATION:
        Headers with spaces surrounding brackets (e.g. ' <date> ') fail because
        col_norm = {c: c.strip('<>').strip()} does not strip brackets if spaces precede them.
        Demonstrates the parser vulnerability under space-padded bracketed headers.
        """
        dirty_csv = tmp_path / "spaced_brackets.csv"
        _generate_synthetic_csv(str(dirty_csv), bar_count=310, header_style="spaced_brackets")

        env = RealM5Environment(csv_path=str(dirty_csv))
        # Demonstrates that current implementation raises ValueError because '<date>' is retained
        with pytest.raises(ValueError, match="Unable to resolve Datetime column"):
            env.load_data("2025.01.14", "2025.01.15")

    def test_empty_date_window_returns_zero_and_protects_indexing(self, tmp_path):
        """Loading a date window with 0 matching bars returns 0 and safely raises IndexError on queries."""
        csv_file = tmp_path / "m5_data.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=310, start_date="2025.01.14 00:00:00")

        env = RealM5Environment(csv_path=str(csv_file))
        # Request date range far in the future
        bars = env.load_data("2026.01.01", "2026.01.02")
        assert bars == 0
        assert len(env) == 0

        with pytest.raises(IndexError):
            env.get_observation(0)
        with pytest.raises(IndexError):
            env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)

    def test_flash_crash_and_extreme_volatility_spikes(self, tmp_path):
        """Flash crash bar (Low drops 99%) followed by recovery: obs_matrix must contain 0 NaNs/Infs."""
        crash_csv = tmp_path / "flash_crash.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append(f"2025.01.14 00:{i%60:02d}:00,100000,100050,99950,100000,100,0,250")
        lines.append("2025.01.15 00:00:00,100000,100000,1.00,10.00,1000,0,500")
        lines.append("2025.01.15 00:05:00,10.00,200000.00,10.00,150000.00,1000,0,500")
        crash_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(crash_csv))
        bars = env.load_data("2025.01.15", "2025.01.15")
        assert bars == 2

        obs0 = env.get_observation(0)
        obs1 = env.get_observation(1)
        assert np.all(np.isfinite(obs0)), "obs0 contains NaN/Inf after flash crash"
        assert np.all(np.isfinite(obs1)), "obs1 contains NaN/Inf after flash rebound"


# ============================================================================
# Section 2: Resampling Utility Robustness (`export_m5_dataset.py`)
# ============================================================================

class TestResamplingUtilityAdversarial:
    """Stress-tests M1 to M5 resampling logic, boundary aggregation, and date clipping."""

    def test_resample_m1_to_m5_exact_aggregation_math(self, tmp_path):
        """Verifies exact OHLCV mathematical rules for 5 consecutive M1 bars."""
        m1_csv = tmp_path / "m1_input.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        lines.append("2025.01.15 00:00:00,100.0,105.0,95.0,102.0,10,1,200")
        lines.append("2025.01.15 00:01:00,102.0,110.0,99.0,108.0,20,2,250")
        lines.append("2025.01.15 00:02:00,108.0,109.0,90.0,92.0,30,3,300")
        lines.append("2025.01.15 00:03:00,92.0,98.0,91.0,95.0,40,4,220")
        lines.append("2025.01.15 00:04:00,95.0,101.0,93.0,99.0,50,5,240")
        m1_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        out_csv = tmp_path / "m5_output.csv"
        resampled = resample_m1_to_m5(
            input_path=str(m1_csv),
            output_path=str(out_csv),
            start_date="2025.01.15",
            end_date="2025.01.15",
            warmup_days=0,
        )

        assert len(resampled) == 1
        bar = resampled.iloc[0]
        # Open -> first (100.0)
        assert bar["Open"] == pytest.approx(100.0)
        # High -> max(105, 110, 109, 98, 101) = 110.0
        assert bar["High"] == pytest.approx(110.0)
        # Low -> min(95, 99, 90, 91, 93) = 90.0
        assert bar["Low"] == pytest.approx(90.0)
        # Close -> last (99.0)
        assert bar["Close"] == pytest.approx(99.0)
        # TickVol -> sum(10+20+30+40+50) = 150
        assert bar["TickVol"] == pytest.approx(150.0)
        # Vol -> sum(1+2+3+4+5) = 15
        assert bar["Vol"] == pytest.approx(15.0)
        # Spread -> median(200, 220, 240, 250, 300) = 240.0
        assert bar["Spread"] == pytest.approx(240.0)

    def test_resample_empty_window_raises_value_error(self, tmp_path):
        """Resampling when no source bars fall inside the window must raise ValueError."""
        m1_csv = tmp_path / "m1_empty.csv"
        lines = [
            "Datetime,Open,High,Low,Close,TickVol,Vol,Spread",
            "2025.01.15 00:00:00,100.0,105.0,95.0,102.0,10,1,200",
        ]
        m1_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="No bars found"):
            resample_m1_to_m5(
                input_path=str(m1_csv),
                start_date="2025.02.01",
                end_date="2025.02.02",
                warmup_days=0,
            )


# ============================================================================
# Section 3: Spread Accounting & Financial Invariants
# ============================================================================

class TestFinancialSpreadAccountingAdversarial:
    """Verifies that nominal spread ($2.50) is deducted without leakage across all conditions."""

    @pytest.fixture
    def flat_env(self, tmp_path):
        """Creates an environment with constant 100,000 price and 100 ATR."""
        csv_file = tmp_path / "flat_m5.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=350, base_price=100000.0)
        env = RealM5Environment(csv_path=str(csv_file), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")
        return env

    @pytest.mark.parametrize("spread", [0.0, 1.0, 2.50, 5.0, 25.0, 100.0])
    def test_breakeven_trade_yields_exactly_negative_spread(self, flat_env, spread):
        """On a completely flat candle trajectory, net PnL must equal strictly -spread_usd."""
        res_buy = flat_env.simulate_trade(bar_idx=0, action=1, sl_atr=5.0, tp_atr=5.0, spread_usd=spread)
        res_sell = flat_env.simulate_trade(bar_idx=0, action=2, sl_atr=5.0, tp_atr=5.0, spread_usd=spread)

        assert res_buy.pnl_usd == pytest.approx(-spread, abs=1e-3)
        assert res_sell.pnl_usd == pytest.approx(-spread, abs=1e-3)

    def test_buy_adverse_gap_down_slippage(self, tmp_path):
        """When market gaps down BELOW Stop Loss on bar open, BUY position exits at Open - half_spread."""
        csv_file = tmp_path / "gap_down.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000,100050,99950,100000,100,0,250")
        lines.append("2025.01.15 00:00:00,100000,100050,99950,100000,100,0,250")
        lines.append("2025.01.15 00:05:00,99600,99650,99500,99550,100,0,250")
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_file), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=4.0)

        assert result.exit_type == "sl"
        assert result.entry_price == pytest.approx(100001.25)
        assert result.exit_price == pytest.approx(99598.75)
        assert result.pnl_usd == pytest.approx(-402.50, abs=1e-2)

    def test_sell_adverse_gap_up_slippage(self, tmp_path):
        """When market gaps up ABOVE Stop Loss on bar open, SELL position exits at Open + half_spread."""
        csv_file = tmp_path / "gap_up.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000,100050,99950,100000,100,0,250")
        lines.append("2025.01.15 00:00:00,100000,100050,99950,100000,100,0,250")
        lines.append("2025.01.15 00:05:00,100400,100500,100350,100450,100,0,250")
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_file), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=2, sl_atr=2.0, tp_atr=4.0)

        assert result.exit_type == "sl"
        assert result.entry_price == pytest.approx(99998.75)
        assert result.exit_price == pytest.approx(100401.25)
        assert result.pnl_usd == pytest.approx(-402.50, abs=1e-2)

    def test_conservative_sl_priority_exact_pnl(self, tmp_path):
        """On simultaneous touch (both SL & TP breached), exit reason is SIMULTANEOUS_SL_FIRST and PnL reflects SL."""
        csv_file = tmp_path / "simultaneous_touch.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000,100050,99950,100000,100,0,250")
        lines.append("2025.01.15 00:00:00,100000,100050,99950,100000,100,0,250")
        # Bar 1: High=100500, Low=99500
        # BUY: SL=99800, TP=100400 -> High>=100400 and Low<=99800 (Both touched)
        # SELL: SL=100200, TP=99600 -> High>=100200 and Low<=99600 (Both touched)
        lines.append("2025.01.15 00:05:00,100000,100500,99500,100000,100,0,250")
        csv_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_file), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res_buy = env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert res_buy.exit_type == "sl"
        assert res_buy.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res_buy.pnl_usd == pytest.approx(-202.50, abs=1e-2)

        res_sell = env.simulate_trade(0, action=2, sl_atr=2.0, tp_atr=4.0)
        assert res_sell.exit_type == "sl"
        assert res_sell.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res_sell.pnl_usd == pytest.approx(-202.50, abs=1e-2)


# ============================================================================
# Section 4: Dataset Boundaries & Edge Horizons
# ============================================================================

class TestDatasetBoundariesAdversarial:
    """Stress-tests boundary conditions at the edges of the dataset."""

    def test_decision_at_final_bar_instant_exit(self, tmp_path):
        """When trading at the final bar of the dataset, it exits immediately with DATASET_END."""
        csv_file = tmp_path / "m5_boundary.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=305, base_price=100000.0)

        env = RealM5Environment(csv_path=str(csv_file), spread_usd=2.50)
        bars = env.load_data("2025.01.14", "2025.01.15")
        last_idx = bars - 1

        result = env.simulate_trade(bar_idx=last_idx, action=1, sl_atr=2.0, tp_atr=4.0)
        assert result.exit_reason == "DATASET_END"
        assert result.holding_bars == 0
        assert result.exit_bar_idx == last_idx
        assert result.pnl_usd == pytest.approx(-2.50, abs=1e-3)

    def test_decision_at_penultimate_bar(self, tmp_path):
        """When trading at penultimate bar (last_idx - 1), only 1 future bar is available."""
        csv_file = tmp_path / "m5_boundary2.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=305, base_price=100000.0)

        env = RealM5Environment(csv_path=str(csv_file), spread_usd=2.50)
        bars = env.load_data("2025.01.14", "2025.01.15")
        penultimate_idx = bars - 2

        result = env.simulate_trade(bar_idx=penultimate_idx, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)
        assert result.holding_bars == 1
        assert result.exit_bar_idx == bars - 1
        assert result.exit_type in ("timestop", "time", "time_stop")

    def test_first_bar_decision(self, tmp_path):
        """Decision at bar_idx = 0 must execute without lookbehind or indexing errors."""
        csv_file = tmp_path / "m5_boundary3.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=320, base_price=100000.0)

        env = RealM5Environment(csv_path=str(csv_file))
        env.load_data("2025.01.14", "2025.01.15")

        obs0 = env.get_observation(0)
        assert obs0.shape == (137,)
        res0 = env.simulate_trade(0, action=1, sl_atr=2.0, tp_atr=4.0)
        assert isinstance(res0, TradeResult)


# ============================================================================
# Section 5: Dynamic Position Slots, Memory Immutability & Concurrency
# ============================================================================

class TestMemoryImmutabilityAndStateIsolation:
    """Verifies that static observation matrix cannot be corrupted by external mutations."""

    @pytest.fixture
    def test_env(self, tmp_path):
        csv_file = tmp_path / "m5_mem.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=320)
        env = RealM5Environment(csv_path=str(csv_file))
        env.load_data("2025.01.14", "2025.01.15")
        return env

    def test_observation_vector_is_copy_not_view(self, test_env):
        """Mutating the array returned by get_observation must NOT corrupt the underlying matrix."""
        orig_val = float(test_env.obs_matrix[5, 0])
        obs = test_env.get_observation(5)

        # Adversarially overwrite every element
        obs[:] = 999999.0
        assert obs[0] == 999999.0

        # Query again from environment
        obs_fresh = test_env.get_observation(5)
        assert obs_fresh[0] == pytest.approx(orig_val)
        assert test_env.obs_matrix[5, 0] == pytest.approx(orig_val)
        assert not np.any(obs_fresh == 999999.0)

    def test_dynamic_position_slots_matrix_isolation(self, test_env):
        """Injecting position slots must NOT write into self.obs_matrix."""
        assert test_env.obs_matrix[10, POSITION_OFFSET] == 0.0

        test_env.positions = [
            PositionSlot(side=1, entry=100000.0, sl=99800.0, tp=100400.0, risk_distance=200.0, opened_bar=5),
            PositionSlot(side=-1, entry=100000.0, sl=100200.0, tp=99600.0, risk_distance=200.0, opened_bar=8),
            PositionSlot(side=1, entry=100000.0, sl=99500.0, tp=101000.0, risk_distance=500.0, opened_bar=9),
        ]

        obs = test_env.get_observation(10)

        # Slot 0 active
        assert obs[POSITION_OFFSET] == 1.0
        assert obs[POSITION_OFFSET + 1] == 1.0  # side +1
        # Slot 1 active
        assert obs[POSITION_OFFSET + 8] == 1.0
        assert obs[POSITION_OFFSET + 9] == -1.0  # side -1
        # Slot 2 active
        assert obs[POSITION_OFFSET + 16] == 1.0
        assert obs[POSITION_OFFSET + 17] == 1.0  # side +1
        # Slot 3 and 4 inactive
        assert obs[POSITION_OFFSET + 24] == 0.0
        assert obs[POSITION_OFFSET + 32] == 0.0

        # Verify underlying obs_matrix was NEVER modified
        assert test_env.obs_matrix[10, POSITION_OFFSET] == 0.0
        assert test_env.obs_matrix[10, POSITION_OFFSET + 8] == 0.0
        assert test_env.obs_matrix[10, POSITION_OFFSET + 16] == 0.0

    def test_overflow_position_slots_capped_at_five(self, test_env):
        """Injecting more than MAX_POSITION_SLOTS (5) must be safely clamped to 5 slots."""
        test_env.positions = [
            PositionSlot(side=1, entry=100000.0, sl=99800.0, tp=100400.0, risk_distance=200.0, opened_bar=i)
            for i in range(12)  # 12 positions
        ]

        # Must not raise IndexError or buffer overflow
        obs = test_env.get_observation(15)
        assert obs.shape == (137,)
        for i in range(MAX_POSITION_SLOTS):
            offset = POSITION_OFFSET + i * POSITION_FEATURE_COUNT
            assert obs[offset] == 1.0

    def test_trade_result_is_frozen_immutable(self, test_env):
        """TradeResult dataclass is frozen and cannot be tampered with."""
        result = test_env.simulate_trade(bar_idx=5, action=1, sl_atr=2.0, tp_atr=4.0)

        with pytest.raises(dataclasses.FrozenInstanceError):
            result.pnl_usd = 999999.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.exit_type = "hacked"


# ============================================================================
# Section 6: Parameter Boundaries & Vulnerability Probing
# ============================================================================

class TestParameterBoundariesAdversarial:
    """Stress-tests parameter boundary checks, malicious floats, and malformed queries."""

    @pytest.fixture
    def env(self, tmp_path):
        csv_file = tmp_path / "m5_param.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=320)
        e = RealM5Environment(csv_path=str(csv_file))
        e.load_data("2025.01.14", "2025.01.15")
        return e

    @pytest.mark.parametrize("bad_action", [0, 3, -1, 4, 10, 100])
    def test_invalid_action_strictly_rejected(self, env, bad_action):
        with pytest.raises(ValueError, match="Invalid action"):
            env.simulate_trade(bar_idx=0, action=bad_action, sl_atr=2.0, tp_atr=4.0)

    @pytest.mark.parametrize("bad_sl", [0.0, -0.001, -1.0, -100.0])
    def test_non_positive_sl_strictly_rejected(self, env, bad_sl):
        with pytest.raises(ValueError, match="must be strictly positive"):
            env.simulate_trade(bar_idx=0, action=1, sl_atr=bad_sl, tp_atr=4.0)

    @pytest.mark.parametrize("bad_tp", [0.0, -0.001, -1.0, -100.0])
    def test_non_positive_tp_strictly_rejected(self, env, bad_tp):
        with pytest.raises(ValueError, match="must be strictly positive"):
            env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=bad_tp)

    @pytest.mark.parametrize("bad_idx", [-1, -100, 99999])
    def test_out_of_range_index_strictly_rejected(self, env, bad_idx):
        with pytest.raises(IndexError):
            env.get_observation(bad_idx)
        with pytest.raises(IndexError):
            env.simulate_trade(bad_idx, action=1, sl_atr=2.0, tp_atr=4.0)
        with pytest.raises(IndexError):
            env.get_bar(bad_idx)
        with pytest.raises(IndexError):
            env.get_atr(bad_idx)
        with pytest.raises(IndexError):
            env.get_timestamp(bad_idx)

    def test_initial_equity_zero_behavior(self, tmp_path):
        """
        EMPIRICAL FINDING:
        Demonstrates that setting initial_equity=0.0 causes ZeroDivisionError on get_observation
        due to obs[72] = (self.equity / self.initial_equity) - 1.0 lacking max(initial_equity, 1.0) guard.
        """
        csv_file = tmp_path / "zero_eq.csv"
        _generate_synthetic_csv(str(csv_file), bar_count=320)
        env = RealM5Environment(csv_path=str(csv_file), initial_equity=0.0)
        env.load_data("2025.01.14", "2025.01.15")

        with pytest.raises(ZeroDivisionError):
            env.get_observation(0)

    def test_nan_sl_bypasses_non_positive_validation(self, env):
        """
        EMPIRICAL FINDING:
        NaN comparison in Python (float('nan') <= 0.0 is False) allows NaN SL/TP to bypass
        the 'if sl_atr <= 0.0 or tp_atr <= 0.0' check and fall through to timestop.
        """
        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=float("nan"), tp_atr=4.0)
        # Because nan <= 0.0 is False, it doesn't raise ValueError; instead it exits via timestop
        assert res.exit_type in ("timestop", "time", "time_stop")
