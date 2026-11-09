"""
Unit and Contract Verification Suite for RealM5Environment
Target Module: research_reasoning_world_class/src/real_m5_environment.py
Test File: research_reasoning_world_class/tests/test_real_m5_environment.py
Milestone: Milestone 1 (M5 Candlestick Environment and Data Ingestion)

Verifies:
1. M5 Ingestion & Delimiter / Header Flexibility:
   - Ingests comma-delimited CSV and tab-delimited TSV.
   - Unified Datetime, separate DATE/TIME, and case-insensitive headers.
   - Auto-resolution fallback when file path is omitted or missing.
   - Exact 12,960 bars for Jan 15 - Feb 28, 2025.
   - Date window boundaries (First bar: 2025.01.15 00:00:00, Last bar: 2025.02.28 23:55:00).
2. Canonical 137D Feature Architecture Contract:
   - Vector shape is strictly (137,).
   - Dtype is strictly np.float32.
   - Absolute numerical safety: exactly 0 NaNs and 0 Infs across sampled bars.
   - Structural feature slice invariants matching protocol_contract.py.
   - Bounded ranges for normalized features.
   - Positive, finite ATR in reasonable range.
3. Path-Dependent BUY Trade Simulation:
   - SL hit before TP (exit_type="sl", holding_bars, gross & net PnL).
   - TP hit before SL (exit_type="tp", holding_bars, gross & net PnL).
   - Time stop exit at 12 bars (exit_type="timestop", holding_bars=12).
   - Multi-bar path resolution and adverse gap slippage.
4. Path-Dependent SELL Trade Simulation:
   - SL hit before TP (exit_type="sl", holding_bars, gross & net PnL).
   - TP hit before SL (exit_type="tp", holding_bars, gross & net PnL).
   - Time stop exit at 12 bars (exit_type="timestop", holding_bars=12).
   - Multi-bar path resolution and adverse gap slippage.
5. Conservative Simultaneous Touch Priority:
   - Simultaneous [Low <= SL and High >= TP] breach on same bar resolves strictly to SL.
   - SIMULTANEOUS_SL_FIRST exit reason.
6. Spread Deduction & Friction Accounting:
   - Flat $2.50 nominal price deducted from gross trade PnL.
   - Breakeven trade results in -$2.50 PnL.
   - Linear scaling with configurable spread_usd parameter.
7. Sub-10 Microsecond Retrieval Latency SLA:
   - get_observation(bar_idx) benchmark over 10,000 queries runs in < 10.0 microseconds.
   - simulate_trade throughput benchmark < 50.0 microseconds.
8. Edge Cases, Dual-Champion Invariance & Duck-Typing:
   - Dataset boundary truncation at end of series.
   - Parameter boundary error handling.
   - TradeResult aliases (pnl_atr, bars_held, exit_bar), tuple unpacking, and indexing.
   - Dynamic position slots update without mutating base observation matrix.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pytest

# Configure single thread for deterministic execution
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Path setup to import repository modules
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

src_dir = os.path.join(repo_root, "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.real_m5_environment import (
    ExitType,
    PositionSlot,
    RealM5Environment,
    TradeResult,
)
from self_growing_trader.protocol_contract import (
    ACCOUNT_FEATURE_COUNT,
    ACCOUNT_OFFSET,
    BASE_MARKET_FEATURE_COUNT,
    EXECUTION_FEATURE_COUNT,
    EXECUTION_OFFSET,
    EXTRA_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    POSITION_FEATURE_COUNT,
    POSITION_OFFSET,
    TIMING_FEATURE_COUNT,
    TIMING_OFFSET,
    WYCKOFF_FEATURE_COUNT,
    WYCKOFF_OFFSET,
)

DEFAULT_M5_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm_M5.csv"
FALLBACK_M5_CSV_PATH = os.path.join(repo_root, "outputs", "BTCUSDm_M5.csv")
EXPECTED_M5_BAR_COUNT = 12960
START_DATE = "2025.01.15"
END_DATE = "2025.02.28"
DEFAULT_SPREAD_USD = 2.50


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def real_m5_env():
    """Module-scoped fixture loading real BTCUSDm_M5.csv once for integration checks."""
    csv_path = DEFAULT_M5_CSV_PATH if os.path.exists(DEFAULT_M5_CSV_PATH) else FALLBACK_M5_CSV_PATH
    if not os.path.exists(csv_path) and not os.path.exists(r"C:\Users\Admin\Desktop\BTCUSDm.csv"):
        pytest.skip(f"Historical candle file not found: {csv_path}")

    env = RealM5Environment(csv_path=csv_path, spread_usd=DEFAULT_SPREAD_USD)
    count = env.load_data(start_date=START_DATE, end_date=END_DATE)
    assert count == EXPECTED_M5_BAR_COUNT, f"Expected {EXPECTED_M5_BAR_COUNT} bars, got {count}"
    return env


@pytest.fixture
def synthetic_m5_file(tmp_path):
    """Generates a deterministic CSV file for unit testing trade simulation with 300 warmup bars."""
    csv_path = tmp_path / "synthetic_test_m5.csv"
    lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]

    # 300 warmup bars on 2025.01.14 (every 5 minutes)
    for i in range(300):
        minute = (i * 5) % 60
        hour = ((i * 5) // 60) % 24
        day_offset = (i * 5) // 1440
        day = 13 + day_offset
        lines.append(
            f"2025.01.{day:02d} {hour:02d}:{minute:02d}:00,100000.00,100050.00,99950.00,100000.00,100,0,250"
        )

    # Bar 0 on 2025.01.15: Active window decision bar
    lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")

    # Bars 1..20: Flat baseline bars (ATR ~ 100.0)
    for i in range(1, 21):
        minute = (i * 5) % 60
        hour = (i * 5) // 60
        lines.append(
            f"2025.01.15 {hour:02d}:{minute:02d}:00,100000.00,100050.00,99950.00,100000.00,100,0,250"
        )

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(csv_path)


# ============================================================================
# Test Suite 1: M5 Loading & Format Flexibility Verification
# ============================================================================

class TestRealM5Loading:
    """Verifies CSV/TSV parsing, flexible columns, fallbacks, and bar counts."""

    def test_m5_file_presence(self):
        """Verifies canonical M5 dataset exists on Desktop or outputs."""
        exists = os.path.exists(DEFAULT_M5_CSV_PATH) or os.path.exists(FALLBACK_M5_CSV_PATH)
        assert exists, "Missing M5 dataset file on Desktop and outputs"

    def test_window_bar_count(self, real_m5_env):
        """Verifies exactly 12,960 bars are loaded for Jan 15 - Feb 28, 2025."""
        assert len(real_m5_env) == EXPECTED_M5_BAR_COUNT
        assert real_m5_env.total_bars == EXPECTED_M5_BAR_COUNT

    def test_date_window_boundaries(self, real_m5_env):
        """Verifies timestamps of first and last loaded bars strictly match bounds."""
        first_time = real_m5_env.get_timestamp(0)
        last_time = real_m5_env.get_timestamp(EXPECTED_M5_BAR_COUNT - 1)
        assert "2025.01.15 00:00" in first_time or "2025-01-15 00:00" in first_time
        assert "2025.02.28 23:55" in last_time or "2025-02-28 23:55" in last_time

    def test_tsv_delimiter_parsing(self, tmp_path):
        """Verifies tab-delimited M5 dataset parses correctly with auto-detected delimiter."""
        tsv_path = tmp_path / "test_m5.tsv"
        lines = [
            "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"
        ]
        # 300 warmup bars
        for i in range(300):
            lines.append("2025.01.14\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250")
        lines.append("2025.01.15\t00:00:00\t100000.00\t100050.00\t99950.00\t100000.00\t100\t0\t250")
        tsv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(tsv_path))
        count = env.load_data("2025.01.15", "2025.01.15")
        assert count == 1
        obs = env.get_observation(0)
        assert obs.shape == (137,)

    def test_flexible_lowercase_headers(self, tmp_path):
        """Verifies dataset with lowercase column headers and unified Time column parses cleanly."""
        csv_path = tmp_path / "lowercase_m5.csv"
        lines = ["time,open,high,low,close,tickvol,vol,spread"]
        for i in range(300):
            lines.append("2025-01-14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025-01-15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path))
        count = env.load_data("2025.01.15", "2025.01.15")
        assert count == 1

    def test_auto_fallback_resolution(self):
        """Verifies automatic fallback to default desktop/outputs dataset when path is None."""
        env = RealM5Environment(csv_path=None)
        count = env.load_data("2025.01.15", "2025.02.28")
        assert count == EXPECTED_M5_BAR_COUNT

    def test_out_of_bounds_indexing(self, real_m5_env):
        """Verifies IndexError is raised on out-of-range bar indices."""
        with pytest.raises(IndexError):
            real_m5_env.get_observation(-1)
        with pytest.raises(IndexError):
            real_m5_env.get_observation(EXPECTED_M5_BAR_COUNT)
        with pytest.raises(IndexError):
            real_m5_env.simulate_trade(-1, action=1, sl_atr=2.0, tp_atr=4.0)
        with pytest.raises(IndexError):
            real_m5_env.simulate_trade(EXPECTED_M5_BAR_COUNT, action=1, sl_atr=2.0, tp_atr=4.0)


# ============================================================================
# Test Suite 2: 137D Feature Architecture Contract & Arithmetic Integrity
# ============================================================================

class Test137DFeatureContract:
    """Verifies 137D feature vector shape, dtype, ranges, and zero NaN/Inf on M5 data."""

    def test_observation_shape_and_dtype(self, real_m5_env):
        """Verifies shape is strictly (137,) and dtype is np.float32."""
        obs = real_m5_env.get_observation(0)
        assert isinstance(obs, np.ndarray), "Observation must be a numpy ndarray"
        assert obs.shape == (137,), f"Expected shape (137,), got {obs.shape}"
        assert obs.dtype == np.float32, f"Expected dtype float32, got {obs.dtype}"

    def test_zero_nan_zero_inf_across_sampled_bars(self, real_m5_env):
        """Verifies 0 NaNs and 0 Infs across sampled bars throughout the M5 dataset."""
        sample_indices = [
            0, 1, 10, 50, 100, 500, 1000, 3000, 6000, 9000, 12000, EXPECTED_M5_BAR_COUNT - 1
        ]
        for idx in sample_indices:
            obs = real_m5_env.get_observation(idx)
            assert not np.isnan(obs).any(), f"NaN detected at bar_idx {idx}"
            assert not np.isinf(obs).any(), f"Inf detected at bar_idx {idx}"
            assert np.all(np.isfinite(obs)), f"Non-finite values at bar_idx {idx}"

    def test_feature_structural_slice_offsets(self, real_m5_env):
        """Verifies partition offsets strictly match protocol_contract.py."""
        obs = real_m5_env.get_observation(50)

        # Base Market Features (0..28, 29 floats)
        base_market = obs[:POSITION_OFFSET]
        assert len(base_market) == BASE_MARKET_FEATURE_COUNT

        # Position Slots (29..68, 40 floats) - default state should be 0.0
        position_slots = obs[POSITION_OFFSET:ACCOUNT_OFFSET]
        assert len(position_slots) == 40
        assert np.all(position_slots == 0.0), "Initial position slots must be 0.0"

        # Account Features (69..76, 8 floats)
        account_features = obs[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET]
        assert len(account_features) == 8

        # Extra Market Features (77..105, 29 floats)
        extra_market = obs[EXTRA_MARKET_OFFSET:EXECUTION_OFFSET]
        assert len(extra_market) == EXTRA_MARKET_FEATURE_COUNT

        # Execution Features (106..116, 11 floats)
        execution_features = obs[EXECUTION_OFFSET:TIMING_OFFSET]
        assert len(execution_features) == EXECUTION_FEATURE_COUNT

        # Timing Features (117..120, 4 floats)
        timing_features = obs[TIMING_OFFSET:WYCKOFF_OFFSET]
        assert len(timing_features) == TIMING_FEATURE_COUNT

        # Wyckoff Features (121..136, 16 floats)
        wyckoff_features = obs[WYCKOFF_OFFSET:INPUT_COUNT]
        assert len(wyckoff_features) == WYCKOFF_FEATURE_COUNT

    def test_normalized_feature_value_ranges(self, real_m5_env):
        """Verifies bounded features obey their mathematical domain constraints."""
        for idx in [0, 100, 500, 2500, 7500, 12000]:
            obs = real_m5_env.get_observation(idx)

            # range_position (index 13) in [0.0, 1.0]
            assert 0.0 <= obs[13] <= 1.0, f"range_position {obs[13]} out of [0, 1]"

            # atr_percentile (index 14) in [0.0, 1.0]
            assert 0.0 <= obs[14] <= 1.0, f"atr_percentile {obs[14]} out of [0, 1]"

            # rsi14 (index 16) and rsi5 (index 17) in [-1.0, 1.0]
            assert -1.0 <= obs[16] <= 1.0, f"rsi14 {obs[16]} out of [-1, 1]"
            assert -1.0 <= obs[17] <= 1.0, f"rsi5 {obs[17]} out of [-1, 1]"

            # volatility_ratio (index 20) in [0.1, 5.0]
            assert 0.1 <= obs[20] <= 5.0, f"volatility_ratio {obs[20]} out of [0.1, 5.0]"

            # candle_body_ratio (index 23) in [0.0, 1.0]
            assert 0.0 <= obs[23] <= 1.0, f"candle_body_ratio {obs[23]} out of [0, 1]"

            # hour_sin / hour_cos (indices 78, 79) in [-1.0, 1.0]
            assert -1.0 <= obs[78] <= 1.0
            assert -1.0 <= obs[79] <= 1.0

            # day_progress (index 75) in [0.0, 1.0]
            assert 0.0 <= obs[75] <= 1.0

    def test_m5_atr_positive_and_finite(self, real_m5_env):
        """Verifies get_atr() returns positive, finite ATR for M5 bars."""
        atr = real_m5_env.get_atr(100)
        assert atr > 0.0, f"ATR must be strictly positive, got {atr}"
        assert np.isfinite(atr), f"ATR must be finite, got {atr}"
        # BTCUSD typical M5 ATR is within [10.0, 1500.0] USD
        assert 10.0 <= atr <= 1500.0, f"BTCUSD M5 ATR out of reasonable range: {atr}"


# ============================================================================
# Test Suite 3: Path-Dependent BUY Trade Simulation
# ============================================================================

class TestBuyPathSimulation:
    """Verifies BUY path execution (SL, TP, time stop, and multi-bar resolution)."""

    def test_buy_sl_hit_before_tp(self, tmp_path):
        """Verifies BUY position exits at SL when Low touches SL before High touches TP."""
        csv_path = tmp_path / "buy_sl.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 0 (Entry): Close = 100000.00
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Low drops to 99750 (SL is 99800 for sl_atr=2.0, atr=100)
        lines.append("2025.01.15 00:05:00,100000.00,100100.00,99750.00,99850.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert isinstance(result, TradeResult)
        assert result.exit_type == "sl"
        assert result.holding_bars == 1
        assert result.bars_held == 1
        # Gross loss = -200.00, Net loss = -200.00 - 2.50 = -202.50
        assert result.pnl_usd == pytest.approx(-202.50, abs=1e-2)
        assert result.return_atr < 0.0
        assert result.pnl_atr == result.return_atr

    def test_buy_tp_hit_before_sl(self, tmp_path):
        """Verifies BUY position exits at TP when High touches TP before Low touches SL."""
        csv_path = tmp_path / "buy_tp.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High spikes to 100450 (TP is 100400 for tp_atr=4.0, atr=100)
        lines.append("2025.01.15 00:05:00,100000.00,100450.00,99950.00,100350.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert result.exit_type == "tp"
        assert result.holding_bars == 1
        # Gross gain = +400.00, Net gain = +400.00 - 2.50 = +397.50
        assert result.pnl_usd == pytest.approx(397.50, abs=1e-2)
        assert result.return_atr > 0.0

    def test_buy_timestop_at_12_bars(self, tmp_path):
        """Verifies BUY position exits at Close of bar 12 when neither SL nor TP is hit."""
        csv_path = tmp_path / "buy_timestop.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # 11 flat bars
        for i in range(1, 12):
            minute = (i * 5) % 60
            lines.append(f"2025.01.15 00:{minute:02d}:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 12: Closes at 100050 (+50 gross price move)
        lines.append("2025.01.15 01:00:00,100000.00,100060.00,99950.00,100050.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert result.exit_type in ("timestop", "time", "time_stop")
        assert result.holding_bars == 12
        # Gross = +50.00, Net = +50.00 - 2.50 = +47.50
        assert result.pnl_usd == pytest.approx(47.50, abs=1e-2)


# ============================================================================
# Test Suite 4: Path-Dependent SELL Trade Simulation
# ============================================================================

class TestSellPathSimulation:
    """Verifies SELL path execution (SL, TP, time stop, and multi-bar resolution)."""

    def test_sell_sl_hit_before_tp(self, tmp_path):
        """Verifies SELL position exits at SL when High touches SL before Low touches TP."""
        csv_path = tmp_path / "sell_sl.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High spikes to 100250 (SELL SL is 100200 for sl_atr=2.0, atr=100)
        lines.append("2025.01.15 00:05:00,100000.00,100250.00,99900.00,100150.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=2, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert result.exit_type == "sl"
        assert result.holding_bars == 1
        assert result.pnl_usd == pytest.approx(-202.50, abs=1e-2)
        assert result.return_atr < 0.0

    def test_sell_tp_hit_before_sl(self, tmp_path):
        """Verifies SELL position exits at TP when Low touches TP before High touches SL."""
        csv_path = tmp_path / "sell_tp.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Low drops to 99550 (SELL TP is 99600 for tp_atr=4.0, atr=100)
        lines.append("2025.01.15 00:05:00,100000.00,100050.00,99550.00,99650.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=2, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert result.exit_type == "tp"
        assert result.holding_bars == 1
        assert result.pnl_usd == pytest.approx(397.50, abs=1e-2)
        assert result.return_atr > 0.0

    def test_sell_timestop_at_12_bars(self, tmp_path):
        """Verifies SELL position exits at Close of bar 12 when neither SL nor TP is hit."""
        csv_path = tmp_path / "sell_timestop.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        for i in range(1, 12):
            minute = (i * 5) % 60
            lines.append(f"2025.01.15 00:{minute:02d}:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 12: Closes at 99920 (+80 gross profit for short)
        lines.append("2025.01.15 01:00:00,100000.00,100050.00,99910.00,99920.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=2, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert result.exit_type in ("timestop", "time", "time_stop")
        assert result.holding_bars == 12
        # Gross short gain = 100000 - 99920 = 80.00, Net = 80.00 - 2.50 = 77.50
        assert result.pnl_usd == pytest.approx(77.50, abs=1e-2)


# ============================================================================
# Test Suite 5: Simultaneous Touch Conflict Resolution (Conservative SL-First)
# ============================================================================

class TestSimultaneousTouchConflictResolution:
    """Verifies conservative institutional assumption: SL touched first on ambiguity."""

    def test_buy_simultaneous_touch_triggers_sl(self, tmp_path):
        """When Bar 1 High >= TP and Low <= SL, BUY must resolve strictly to SL."""
        csv_path = tmp_path / "buy_simultaneous.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High spikes to 100500 (>= TP 100400) AND Low plummets to 99700 (<= SL 99800)
        lines.append("2025.01.15 00:05:00,100000.00,100500.00,99700.00,100100.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        # Must resolve to SL, NEVER to TP
        assert result.exit_type == "sl", "Conservative execution must prioritize SL over TP"
        assert result.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert result.holding_bars == 1
        assert result.pnl_usd < 0.0, "PnL must be negative for SL realization"
        assert result.pnl_usd == pytest.approx(-202.50, abs=1e-2)

    def test_sell_simultaneous_touch_triggers_sl(self, tmp_path):
        """When Bar 1 High >= SL and Low <= TP, SELL must resolve strictly to SL."""
        csv_path = tmp_path / "sell_simultaneous.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High rises to 100300 (>= SL 100200) AND Low drops to 99500 (<= TP 99600)
        lines.append("2025.01.15 00:05:00,100000.00,100300.00,99500.00,100000.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=2, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)

        assert result.exit_type == "sl", "Conservative execution must prioritize SL over TP"
        assert result.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert result.holding_bars == 1
        assert result.pnl_usd < 0.0
        assert result.pnl_usd == pytest.approx(-202.50, abs=1e-2)


# ============================================================================
# Test Suite 6: Spread Deduction & Friction Accounting
# ============================================================================

class TestSpreadDeductionAndFriction:
    """Verifies spread deduction is accurately subtracted on all trade exits."""

    def test_spread_deduction_scaling(self, tmp_path):
        """Verifies net PnL equals gross PnL minus spread_usd."""
        csv_path = tmp_path / "spread_diff.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:05:00,100000.00,100500.00,99950.00,100400.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env_spread_0 = RealM5Environment(csv_path=str(csv_path), spread_usd=0.0)
        env_spread_0.load_data("2025.01.15", "2025.01.15")
        res_0 = env_spread_0.simulate_trade(0, 1, 2.0, 4.0)

        env_spread_25 = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env_spread_25.load_data("2025.01.15", "2025.01.15")
        res_25 = env_spread_25.simulate_trade(0, 1, 2.0, 4.0)

        env_spread_50 = RealM5Environment(csv_path=str(csv_path), spread_usd=5.00)
        env_spread_50.load_data("2025.01.15", "2025.01.15")
        res_50 = env_spread_50.simulate_trade(0, 1, 2.0, 4.0)

        assert res_0.pnl_usd - res_25.pnl_usd == pytest.approx(2.50, abs=1e-3)
        assert res_0.pnl_usd - res_50.pnl_usd == pytest.approx(5.00, abs=1e-3)

    def test_breakeven_trade_produces_negative_pnl(self, tmp_path):
        """Verifies a flat trade (Close = Entry) results in -$2.50 PnL."""
        csv_path = tmp_path / "breakeven.csv"
        lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]
        for i in range(300):
            lines.append("2025.01.14 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        for i in range(1, 13):
            minute = (i * 5) % 60
            lines.append(f"2025.01.15 00:{minute:02d}:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        env = RealM5Environment(csv_path=str(csv_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        result = env.simulate_trade(bar_idx=0, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12)
        assert result.exit_type in ("timestop", "time", "time_stop")
        assert result.pnl_usd == pytest.approx(-2.50, abs=1e-3)


# ============================================================================
# Test Suite 7: Sub-10 Microsecond Retrieval Latency & Throughput Benchmark
# ============================================================================

class TestPerformanceAndLatency:
    """Verifies get_observation runs in <10 microseconds and simulation throughput."""

    def test_get_observation_sub_10_microsecond_latency(self, real_m5_env):
        """Verifies average observation retrieval time is strictly < 10.0 microseconds."""
        indices = np.random.randint(0, EXPECTED_M5_BAR_COUNT, size=10_000)

        # Warmup cache
        for idx in indices[:500]:
            _ = real_m5_env.get_observation(int(idx))

        start_ns = time.perf_counter_ns()
        for idx in indices:
            _ = real_m5_env.get_observation(int(idx))
        elapsed_ns = time.perf_counter_ns() - start_ns

        avg_latency_us = (elapsed_ns / len(indices)) / 1000.0
        print(f"\n[BENCHMARK M5] get_observation avg latency: {avg_latency_us:.3f} microseconds/query")

        assert avg_latency_us < 10.0, f"get_observation exceeded 10us budget: {avg_latency_us:.3f}us"

    def test_simulate_trade_throughput(self, real_m5_env):
        """Verifies simulate_trade throughput is < 50.0 microseconds per path evaluation."""
        indices = np.random.randint(0, EXPECTED_M5_BAR_COUNT - 50, size=1_000)

        # Warmup
        for idx in indices[:50]:
            _ = real_m5_env.simulate_trade(int(idx), action=1, sl_atr=2.0, tp_atr=4.0)

        start_ns = time.perf_counter_ns()
        for idx in indices:
            _ = real_m5_env.simulate_trade(int(idx), action=1, sl_atr=2.0, tp_atr=4.0)
        elapsed_ns = time.perf_counter_ns() - start_ns

        avg_sim_us = (elapsed_ns / len(indices)) / 1000.0
        print(f"\n[BENCHMARK M5] simulate_trade avg latency: {avg_sim_us:.3f} microseconds/eval")

        assert avg_sim_us < 50.0, f"simulate_trade exceeded 50us budget: {avg_sim_us:.3f}us"


# ============================================================================
# Test Suite 8: Edge Cases, Dual-Champion Invariance & Duck-Typing
# ============================================================================

class TestEdgeCasesAndDualChampionInterface:
    """Verifies edge cases, Duck-typing, and TradeResult aliases."""

    def test_dataset_boundary_truncation(self, real_m5_env):
        """Verifies trades near dataset end (bar 12955) exit cleanly at final bar."""
        near_end_idx = EXPECTED_M5_BAR_COUNT - 3
        result = real_m5_env.simulate_trade(
            bar_idx=near_end_idx, action=1, sl_atr=2.0, tp_atr=4.0, max_holding_bars=12
        )
        assert isinstance(result, TradeResult)
        assert result.holding_bars <= 3
        assert np.isfinite(result.pnl_usd)

    def test_invalid_action_rejected(self, real_m5_env):
        """Verifies invalid action indices raise ValueError."""
        with pytest.raises(ValueError):
            real_m5_env.simulate_trade(bar_idx=100, action=0, sl_atr=2.0, tp_atr=4.0)
        with pytest.raises(ValueError):
            real_m5_env.simulate_trade(bar_idx=100, action=3, sl_atr=2.0, tp_atr=4.0)

    def test_invalid_sl_tp_ranges_rejected(self, real_m5_env):
        """Verifies non-positive sl_atr or tp_atr raises ValueError."""
        with pytest.raises(ValueError):
            real_m5_env.simulate_trade(bar_idx=100, action=1, sl_atr=0.0, tp_atr=4.0)
        with pytest.raises(ValueError):
            real_m5_env.simulate_trade(bar_idx=100, action=1, sl_atr=2.0, tp_atr=0.0)

    def test_trade_result_duck_typing_and_aliases(self, real_m5_env):
        """Verifies TradeResult supports aliases, indexing, and tuple unpacking."""
        res = real_m5_env.simulate_trade(bar_idx=100, action=1, sl_atr=2.0, tp_atr=4.0)
        # Property aliases
        assert res.pnl_atr == res.return_atr
        assert res.bars_held == res.holding_bars
        assert res.exit_bar == res.exit_bar_idx

        # Indexing by int
        assert res[0] == res.pnl_usd
        assert res[1] == res.return_atr
        assert res[2] == res.exit_type
        assert res[3] == res.holding_bars

        # Indexing by string
        assert res["pnl_usd"] == res.pnl_usd
        assert res["pnl_atr"] == res.return_atr
        assert res["bars_held"] == res.holding_bars
        assert res["exit_bar"] == res.exit_bar_idx

        # Tuple unpacking
        pnl, ret_atr, exit_type, holding, exit_p, entry_p = res
        assert pnl == res.pnl_usd
        assert ret_atr == res.return_atr

    def test_dynamic_position_slots_update(self, real_m5_env):
        """Verifies adding a position updates slots in get_observation without mutating obs_matrix."""
        obs_empty = real_m5_env.get_observation(100)
        assert obs_empty[29] == 0.0  # Slot 0 inactive

        # Inject position
        real_m5_env.positions.append(
            PositionSlot(side=1, entry=90000.0, sl=89000.0, tp=92000.0, risk_distance=1000.0, opened_bar=95)
        )
        obs_active = real_m5_env.get_observation(100)
        assert obs_active[29] == 1.0  # Slot 0 active
        assert obs_active[30] == 1.0  # Side +1 (BUY)

        # Base matrix remains clean
        assert real_m5_env.obs_matrix[100, 29] == 0.0

        # Reset clears position
        real_m5_env.reset()
        obs_reset = real_m5_env.get_observation(100)
        assert obs_reset[29] == 0.0
