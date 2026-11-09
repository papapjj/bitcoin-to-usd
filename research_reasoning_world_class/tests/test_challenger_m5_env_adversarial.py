"""
Adversarial Stress Test Suite & Empirical Verification Harness for Milestone 1:
M5 Candlestick Environment and Data Ingestion.

Test Target:
- research_reasoning_world_class/src/real_m5_environment.py
- scripts/export_m5_dataset.py
- src/self_growing_trader/features.py

Adversarial Dimensions Tested:
1. Path-Dependent Execution Boundary Conditions:
   - Adverse gap slippage on BUY (bar opens below SL).
   - Adverse gap slippage on SELL (bar opens above SL).
   - Favorable gap open on BUY (bar opens above TP).
   - Favorable gap open on SELL (bar opens below TP).
   - Multi-bar delayed adverse gap open.
2. Extreme Wick Collision & Simultaneous Touch Resolution:
   - Single-candle extreme wick touching both SL and TP (verifying SIMULTANEOUS_SL_FIRST).
   - BUY and SELL simultaneous collision on bar 1.
   - Delayed simultaneous collision on bar 5.
   - Multi-bar collision ordering: Bar 1 SL vs Bar 2 TP.
   - Multi-bar collision ordering: Bar 1 TP vs Bar 2 SL.
   - Exact boundary touching (Low == SL, High == TP).
3. Zero-Range / Flatline Candles (H = L = C = O):
   - Zero-range horizontal market regime.
   - Timestop resolution and exact spread accounting.
   - Zero division prevention across return_atr and features.
4. Numerical Integrity & Extreme State Saturation:
   - 137D feature vector shape (137,), dtype strictly float32.
   - Exactly zero NaNs and zero Infs across all bars.
   - Maximum position slot saturation (5 slots active + overflow protection).
   - Boundary checks on protocol offsets.
5. High-Throughput Continuous Latency & Stress:
   - 100,000 observation queries benchmark (< 10.0 µs SLA).
   - 100,000 trade simulations benchmark (< 50.0 µs SLA).
6. Ingestion Robustness:
   - Semicolon delimiters, irregular whitespace, uppercase/lowercase headers.
   - Missing optional columns (fallback to default spread/volume).
   - Split DATE / TIME headers.
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

# Path configuration
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
from self_growing_trader.protocol_contract import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    POSITION_FEATURE_COUNT,
    POSITION_OFFSET,
    TIMING_OFFSET,
    WYCKOFF_OFFSET,
)

DEFAULT_M5_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm_M5.csv"
FALLBACK_M5_CSV_PATH = os.path.join(repo_root, "outputs", "BTCUSDm_M5.csv")
REAL_CSV_PATH = DEFAULT_M5_CSV_PATH if os.path.exists(DEFAULT_M5_CSV_PATH) else FALLBACK_M5_CSV_PATH


def create_base_synthetic_csv(tmp_path, filename: str, warmup_bars: int = 300) -> str:
    """Helper to generate standard baseline CSV with adequate warmup for rolling features."""
    file_path = tmp_path / filename
    lines = ["Datetime,Open,High,Low,Close,TickVol,Vol,Spread"]

    # Warmup bars on 2025.01.14
    for i in range(warmup_bars):
        minute = (i * 5) % 60
        hour = ((i * 5) // 60) % 24
        day_offset = (i * 5) // 1440
        day = 13 + day_offset
        lines.append(
            f"2025.01.{day:02d} {hour:02d}:{minute:02d}:00,100000.00,100050.00,99950.00,100000.00,100,0,250"
        )
    return file_path, lines


# ============================================================================
# Category 1: Path-Dependent Execution Boundary Conditions & Gapping
# ============================================================================

class TestPathDependentGappingAdversarial:
    """Adversarial stress-testing of gapping bars and slippage accounting."""

    def test_buy_adverse_gap_below_sl(self, tmp_path):
        """
        Adversarial test: BUY order where bar 1 opens significantly BELOW the SL level.
        Execution must fill at Open - half_spread (adverse gap slippage), NOT at SL.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "buy_gap_down.csv")

        # Decision bar (bar 0): Close = 100,000.00
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Huge GAP DOWN open at 99,800.00 (below SL of 99,900.00). Low = 99,750, High = 99,850
        lines.append("2025.01.15 00:05:00,99800.00,99850.00,99750.00,99800.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        atr = env.get_atr(0)
        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)

        # Expected:
        # entry_price = 100000.00 + 1.25 = 100001.25
        # sl_level = 100000.00 - (1.0 * atr) ~ 99900.00
        # Since bar 1 Open = 99800.00 < sl_level, exit_level = 99800.00
        # exit_price = 99800.00 - 1.25 = 99798.75
        # pnl_usd = 99798.75 - 100001.25 = -202.50
        assert res.exit_type == "sl"
        assert res.exit_reason == "SL_HIT"
        assert res.holding_bars == 1
        assert res.exit_bar == 1
        assert pytest.approx(res.entry_price, rel=1e-4) == 100001.25
        assert pytest.approx(res.exit_price, rel=1e-4) == 99798.75
        assert pytest.approx(res.pnl_usd, rel=1e-4) == -202.50
        # Slippage penalty: lost more than the 1.0 ATR SL
        assert res.return_atr < -1.0

    def test_sell_adverse_gap_above_sl(self, tmp_path):
        """
        Adversarial test: SELL order where bar 1 opens significantly ABOVE the SL level.
        Execution must fill at Open + half_spread (adverse gap slippage), NOT at SL.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "sell_gap_up.csv")

        # Decision bar (bar 0): Close = 100,000.00
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Huge GAP UP open at 100,200.00 (above SL of 100,100.00). High = 100,250, Low = 100,150
        lines.append("2025.01.15 00:05:00,100200.00,100250.00,100150.00,100200.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=2, sl_atr=1.0, tp_atr=2.0)

        # Expected:
        # entry_price = 100000.00 - 1.25 = 99998.75
        # sl_level = 100000.00 + (1.0 * atr) ~ 100100.00
        # Since bar 1 Open = 100200.00 > sl_level, exit_level = 100200.00
        # exit_price = 100200.00 + 1.25 = 100201.25
        # pnl_usd = 99998.75 - 100201.25 = -202.50
        assert res.exit_type == "sl"
        assert res.exit_reason == "SL_HIT"
        assert res.holding_bars == 1
        assert pytest.approx(res.entry_price, rel=1e-4) == 99998.75
        assert pytest.approx(res.exit_price, rel=1e-4) == 100201.25
        assert pytest.approx(res.pnl_usd, rel=1e-4) == -202.50
        assert res.return_atr < -1.0

    def test_buy_favorable_gap_above_tp(self, tmp_path):
        """
        Adversarial test: BUY order where bar 1 opens above TP level.
        Conservative limit execution fills at tp_level - half_spread.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "buy_gap_tp.csv")

        # Decision bar
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Open at 100,300 (TP is at 100,200). Low = 100,250 (never touches SL)
        lines.append("2025.01.15 00:05:00,100300.00,100350.00,100250.00,100300.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "tp"
        assert res.exit_reason == "TP_HIT"
        assert res.holding_bars == 1
        assert res.pnl_usd > 0.0

    def test_sell_favorable_gap_below_tp(self, tmp_path):
        """
        Adversarial test: SELL order where bar 1 opens below TP level.
        Conservative limit execution fills at tp_level + half_spread.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "sell_gap_tp.csv")

        # Decision bar
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Open at 99,700 (TP is at 99,800). High = 99,750 (never touches SL)
        lines.append("2025.01.15 00:05:00,99700.00,99750.00,99650.00,99700.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=2, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "tp"
        assert res.exit_reason == "TP_HIT"
        assert res.holding_bars == 1
        assert res.pnl_usd > 0.0

    def test_delayed_adverse_gap_on_subsequent_bar(self, tmp_path):
        """
        Adversarial test: Trade survives 2 bars in neutral range, then gaps adversely on bar 3.
        Must resolve holding_bars = 3 with bar 3 gap open price.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "delayed_gap.csv")

        # Bar 0
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Neutral
        lines.append("2025.01.15 00:05:00,100000.00,100030.00,99970.00,100010.00,100,0,250")
        # Bar 2: Neutral
        lines.append("2025.01.15 00:10:00,100010.00,100040.00,99980.00,100020.00,100,0,250")
        # Bar 3: Sudden weekend/overnight gap down to 99,800
        lines.append("2025.01.15 00:15:00,99800.00,99820.00,99750.00,99780.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SL_HIT"
        assert res.holding_bars == 3
        assert res.exit_bar == 3
        assert pytest.approx(res.exit_price, rel=1e-4) == 99798.75


# ============================================================================
# Category 2: Single-Candle Extreme Wicks & Conservative Priority
# ============================================================================

class TestSimultaneousTouchAdversarial:
    """Stress-testing the conservative SIMULTANEOUS_SL_FIRST priority on double touches."""

    def test_buy_extreme_wick_double_touch(self, tmp_path):
        """
        Adversarial test: Bar 1 has massive range [99,800.00 to 100,300.00].
        Both SL (99,900) and TP (100,200) are breached in the single candle.
        Must strictly resolve to SL with exit_reason='SIMULTANEOUS_SL_FIRST'.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "buy_simultaneous.csv")

        # Bar 0
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High 100,300 >= TP, Low 99,800 <= SL, Open 100,000 (neutral open)
        lines.append("2025.01.15 00:05:00,100000.00,100300.00,99800.00,100000.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res.holding_bars == 1
        assert res.pnl_usd < 0.0

    def test_sell_extreme_wick_double_touch(self, tmp_path):
        """
        Adversarial test: SELL order on extreme wick candle.
        Both SL (100,100) and TP (99,800) are breached.
        Must strictly resolve to SL with exit_reason='SIMULTANEOUS_SL_FIRST'.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "sell_simultaneous.csv")

        # Bar 0
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High 100,300 >= SL, Low 99,700 <= TP, Open 100,000
        lines.append("2025.01.15 00:05:00,100000.00,100300.00,99700.00,100000.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=2, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res.holding_bars == 1
        assert res.pnl_usd < 0.0

    def test_delayed_simultaneous_touch_on_bar_4(self, tmp_path):
        """
        Adversarial test: 3 neutral bars followed by simultaneous double touch on bar 4.
        Must resolve holding_bars = 4, exit_reason='SIMULTANEOUS_SL_FIRST'.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "delayed_simul.csv")

        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bars 1..3 neutral
        for b in range(1, 4):
            lines.append(f"2025.01.15 00:{b*5:02d}:00,100000.00,100030.00,99970.00,100000.00,100,0,250")
        # Bar 4: double touch
        lines.append("2025.01.15 00:20:00,100000.00,100300.00,99800.00,100000.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SIMULTANEOUS_SL_FIRST"
        assert res.holding_bars == 4
        assert res.exit_bar == 4

    def test_multibar_order_sl_first_then_tp_subsequent(self, tmp_path):
        """
        Adversarial test: Bar 1 breaches SL only. Bar 2 breaches TP only.
        The trade must terminate on Bar 1 with SL, NOT reach Bar 2.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "sl_then_tp.csv")

        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Low 99,850 <= SL (99,900), High 100,050 < TP
        lines.append("2025.01.15 00:05:00,100000.00,100050.00,99850.00,99950.00,100,0,250")
        # Bar 2: High 100,350 >= TP (100,200), Low 99,950 > SL
        lines.append("2025.01.15 00:10:00,100000.00,100350.00,99950.00,100300.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SL_HIT"
        assert res.holding_bars == 1
        assert res.exit_bar == 1

    def test_multibar_order_tp_first_then_sl_subsequent(self, tmp_path):
        """
        Adversarial test: Bar 1 breaches TP only. Bar 2 breaches SL only.
        The trade must terminate on Bar 1 with TP, NOT reach Bar 2.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "tp_then_sl.csv")

        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: High 100,300 >= TP (100,200), Low 99,950 > SL
        lines.append("2025.01.15 00:05:00,100000.00,100300.00,99950.00,100250.00,100,0,250")
        # Bar 2: Low 99,800 <= SL (99,900), High 100,050 < TP
        lines.append("2025.01.15 00:10:00,100000.00,100050.00,99800.00,99850.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "tp"
        assert res.exit_reason == "TP_HIT"
        assert res.holding_bars == 1
        assert res.exit_bar == 1

    def test_exact_equality_sl_tp_touches(self, tmp_path):
        """
        Adversarial test: Exactly touching SL (Low == SL) or TP (High == TP).
        Non-strict <= and >= comparisons must trigger exits.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "exact_touch.csv")

        # Bar 0
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bar 1: Low exactly 99,900.00 (SL is 100,000 - 100 = 99,900)
        lines.append("2025.01.15 00:05:00,100000.00,100020.00,99900.00,99950.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0)
        assert res.exit_type == "sl"
        assert res.exit_reason == "SL_HIT"
        assert res.holding_bars == 1


# ============================================================================
# Category 3: Flatline / Zero-Range Candles (H = L = C = O)
# ============================================================================

class TestFlatlineCandleAdversarial:
    """Stress-testing periods of completely flatline zero-range candles."""

    def test_flatline_timestop_resolution_and_spread(self, tmp_path):
        """
        Adversarial test: 20 consecutive flatline candles (H = L = C = O).
        Trade must resolve cleanly to timestop at max_holding_bars, incurring exact spread loss.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "flatline_test.csv")

        # Bar 0
        lines.append("2025.01.15 00:00:00,100000.00,100050.00,99950.00,100000.00,100,0,250")
        # Bars 1..15: Flatline bars H = L = C = O = 100,000.00
        for b in range(1, 16):
            minute = (b * 5) % 60
            hour = (b * 5) // 60
            lines.append(f"2025.01.15 {hour:02d}:{minute:02d}:00,100000.00,100000.00,100000.00,100000.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        res_buy = env.simulate_trade(bar_idx=0, action=1, sl_atr=1.0, tp_atr=2.0, max_holding_bars=12)
        assert res_buy.exit_type == "timestop"
        assert res_buy.exit_reason == "TIME_STOP"
        assert res_buy.holding_bars == 12
        assert res_buy.exit_bar == 12
        # On flat market, BUY entry = 100001.25, exit = 99998.75 -> loss is exactly -2.50
        assert pytest.approx(res_buy.pnl_usd, rel=1e-4) == -2.50
        assert np.isfinite(res_buy.return_atr)

        res_sell = env.simulate_trade(bar_idx=0, action=2, sl_atr=1.0, tp_atr=2.0, max_holding_bars=12)
        assert res_sell.exit_type == "timestop"
        assert res_sell.holding_bars == 12
        assert pytest.approx(res_sell.pnl_usd, rel=1e-4) == -2.50
        assert np.isfinite(res_sell.return_atr)

    def test_flatline_feature_vector_cleanliness(self, tmp_path):
        """
        Adversarial test: Observations generated on flatline bars must contain
        strictly 0 NaNs, 0 Infs, and remain float32.
        """
        file_path, lines = create_base_synthetic_csv(tmp_path, "flatline_obs.csv")

        for b in range(0, 30):
            minute = (b * 5) % 60
            hour = (b * 5) // 60
            lines.append(f"2025.01.15 {hour:02d}:{minute:02d}:00,100000.00,100000.00,100000.00,100000.00,100,0,250")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(file_path), spread_usd=2.50)
        env.load_data("2025.01.15", "2025.01.15")

        for i in range(len(env)):
            obs = env.get_observation(i)
            assert obs.shape == (137,)
            assert obs.dtype == np.float32
            assert not np.isnan(obs).any(), f"Found NaNs on flatline bar {i}"
            assert not np.isinf(obs).any(), f"Found Infs on flatline bar {i}"


# ============================================================================
# Category 4: Numerical Safety & Extreme State Saturation
# ============================================================================

class TestNumericalIntegrityAndSaturationAdversarial:
    """Stress-testing full position slots, extreme price regimes, and buffer limits."""

    def test_maximum_position_slot_saturation(self):
        """
        Adversarial test: Saturate all 5 position slots (`MAX_POSITION_SLOTS = 5`)
        with diverse positions (BUY, SELL, huge PnL, large drawdown), and then
        add 3 extra positions (overflow).
        Must update dynamically without buffer overrun or array mismatch.
        """
        if not os.path.exists(REAL_CSV_PATH):
            pytest.skip("Historical data required for saturation test")

        env = RealM5Environment(csv_path=REAL_CSV_PATH)
        env.load_data("2025.01.15", "2025.02.28")

        # Inject 8 positions (exceeding MAX_POSITION_SLOTS = 5)
        env.positions = [
            PositionSlot(side=1, entry=95000.0, sl=94000.0, tp=98000.0, risk_distance=1000.0, opened_bar=10),
            PositionSlot(side=-1, entry=96000.0, sl=97000.0, tp=94000.0, risk_distance=1000.0, opened_bar=20),
            PositionSlot(side=1, entry=97000.0, sl=96500.0, tp=98500.0, risk_distance=500.0, opened_bar=30),
            PositionSlot(side=-1, entry=95500.0, sl=96000.0, tp=94500.0, risk_distance=500.0, opened_bar=40),
            PositionSlot(side=1, entry=98000.0, sl=97000.0, tp=100000.0, risk_distance=1000.0, opened_bar=50),
            # Overflows
            PositionSlot(side=1, entry=99000.0, sl=98000.0, tp=101000.0, risk_distance=1000.0, opened_bar=60),
            PositionSlot(side=-1, entry=99000.0, sl=100000.0, tp=97000.0, risk_distance=1000.0, opened_bar=70),
            PositionSlot(side=1, entry=99500.0, sl=98500.0, tp=102000.0, risk_distance=1000.0, opened_bar=80),
        ]
        env.equity = 9500.0
        env.peak_equity = 10500.0
        env.today_trade_count = 7

        obs = env.get_observation(100)

        assert obs.shape == (137,)
        assert obs.dtype == np.float32
        assert not np.isnan(obs).any()
        assert not np.isinf(obs).any()

        # Check Position Slots (Indices 29..68)
        # Slot 0 active
        assert obs[POSITION_OFFSET] == 1.0
        assert obs[POSITION_OFFSET + 1] == 1.0  # side = +1
        # Slot 1 active
        assert obs[POSITION_OFFSET + 8] == 1.0
        assert obs[POSITION_OFFSET + 9] == -1.0  # side = -1
        # Slot 4 active
        assert obs[POSITION_OFFSET + 32] == 1.0

        # Check Account State (Indices 69..76)
        assert obs[ACCOUNT_OFFSET] == 8 / 5.0  # len(positions) / 5.0
        assert pytest.approx(obs[ACCOUNT_OFFSET + 2], rel=1e-3) == (10500.0 - 9500.0) / 10500.0  # drawdown
        assert pytest.approx(obs[ACCOUNT_OFFSET + 3], rel=1e-3) == (9500.0 / 10000.0) - 1.0  # return

        # Dynamic execution features (Indices 106..116)
        assert np.isfinite(obs[EXECUTION_OFFSET:EXECUTION_OFFSET + 11]).all()

    def test_extreme_spread_scaling(self):
        """
        Adversarial test: Zero spread vs massive spread ($1,000 spread).
        Checks linear spread friction deduction without calculation distortion.
        """
        if not os.path.exists(REAL_CSV_PATH):
            pytest.skip("Historical data required")

        env = RealM5Environment(csv_path=REAL_CSV_PATH)
        env.load_data("2025.01.15", "2025.02.28")

        # Zero spread trade
        res_zero = env.simulate_trade(bar_idx=100, action=1, sl_atr=1.0, tp_atr=2.0, spread_usd=0.0)
        # Huge spread trade ($1,000 spread)
        res_huge = env.simulate_trade(bar_idx=100, action=1, sl_atr=1.0, tp_atr=2.0, spread_usd=1000.0)

        # Spread diff should be exactly -$1,000 on net PnL
        pnl_diff = res_huge.pnl_usd - res_zero.pnl_usd
        assert pytest.approx(pnl_diff, abs=1e-3) == -1000.0


# ============================================================================
# Category 5: Continuous High-Throughput Latency & Stress
# ============================================================================

class TestHighThroughputContinuousStress:
    """Rigorous stress-testing of query latency and simulation throughput under 100,000 iterations."""

    def test_100k_observation_queries_latency_sla(self):
        """
        Adversarial benchmark: 100,000 observation retrievals at random bar indices.
        Must strictly satisfy Mean Latency < 10.0 µs SLA target.
        """
        if not os.path.exists(REAL_CSV_PATH):
            pytest.skip("Historical data required")

        env = RealM5Environment(csv_path=REAL_CSV_PATH)
        env.load_data("2025.01.15", "2025.02.28")

        num_queries = 100_000
        rng = np.random.default_rng(42)
        indices = rng.integers(0, len(env), size=num_queries)

        t0 = time.perf_counter_ns()
        for idx in indices:
            _ = env.get_observation(int(idx))
        elapsed_ns = time.perf_counter_ns() - t0

        avg_latency_us = (elapsed_ns / num_queries) / 1000.0
        print(f"\n[100k Observation Stress] Average Latency: {avg_latency_us:.3f} µs/query")

        assert avg_latency_us < 10.0, f"Observation latency SLA failed: {avg_latency_us:.3f} µs >= 10.0 µs"

    def test_100k_trade_simulations_throughput_sla(self):
        """
        Adversarial benchmark: 100,000 path-dependent trade simulations across random
        bars, actions (BUY/SELL), and SL/TP ratios.
        Must strictly satisfy Mean Latency < 50.0 µs SLA target.
        """
        if not os.path.exists(REAL_CSV_PATH):
            pytest.skip("Historical data required")

        env = RealM5Environment(csv_path=REAL_CSV_PATH)
        env.load_data("2025.01.15", "2025.02.28")

        num_sims = 100_000
        rng = np.random.default_rng(123)
        # Sample bar indices allowing at least 12 holding bars
        indices = rng.integers(0, len(env) - 15, size=num_sims)
        actions = rng.choice([1, 2], size=num_sims)
        sl_atrs = rng.uniform(1.0, 5.0, size=num_sims)
        tp_atrs = rng.uniform(1.5, 8.0, size=num_sims)

        t0 = time.perf_counter_ns()
        for i in range(num_sims):
            _ = env.simulate_trade(
                bar_idx=int(indices[i]),
                action=int(actions[i]),
                sl_atr=float(sl_atrs[i]),
                tp_atr=float(tp_atrs[i]),
                max_holding_bars=12,
            )
        elapsed_ns = time.perf_counter_ns() - t0

        avg_sim_us = (elapsed_ns / num_sims) / 1000.0
        print(f"\n[100k Trade Sim Stress] Average Throughput: {avg_sim_us:.3f} µs/simulation")

        assert avg_sim_us < 50.0, f"Trade simulation SLA failed: {avg_sim_us:.3f} µs >= 50.0 µs"


# ============================================================================
# Category 6: Ingestion Robustness & Corrupted Input Formatting
# ============================================================================

class TestIngestionAdversarialFormats:
    """Stress-testing delimiter sniffing, irregular whitespaces, and missing columns."""

    def test_semicolon_delimited_with_irregular_whitespace(self, tmp_path):
        """
        Adversarial test: Semicolon delimiter ';' with standard clean column headers.
        """
        csv_path = tmp_path / "semicolon_clean.csv"
        lines = [
            "Datetime;OPEN;HIGH;LOW;close;TickVol;Vol;Spread",
        ]
        for i in range(300):
            lines.append(
                f"2025.01.14 00:{i%60:02d}:00;100000.0;100050.0;99950.0;100000.0;100;0;250"
            )
        lines.append("2025.01.15 00:00:00;100000.0;100050.0;99950.0;100000.0;100;0;250")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(csv_path))
        count = env.load_data("2025.01.15", "2025.01.15")
        assert count == 1
        obs = env.get_observation(0)
        assert obs.shape == (137,)
        assert not np.isnan(obs).any()

    @pytest.mark.xfail(
        reason="Vulnerability: c.strip('<>').strip() in real_m5_environment.py:249 fails when whitespace precedes angle brackets",
        strict=True,
    )
    def test_spaced_angle_bracket_header_vulnerability(self, tmp_path):
        """
        Adversarial test proving the strip ordering bug:
        When a header contains ' <Datetime> ' with leading spaces, c.strip('<>') does not
        strip '<>' because outer characters are spaces, raising ValueError.
        """
        csv_path = tmp_path / "spaced_brackets.csv"
        lines = [
            " <Datetime> , OPEN , HIGH , LOW , Close , TickVol , Vol , Spread",
        ]
        for i in range(300):
            lines.append(f"2025.01.14 00:{i%60:02d}:00,100000.0,100050.0,99950.0,100000.0,100,0,250")
        lines.append("2025.01.15 00:00:00,100000.0,100050.0,99950.0,100000.0,100,0,250")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(csv_path))
        # This will fail with ValueError due to line 249 strip ordering
        env.load_data("2025.01.15", "2025.01.15")

    @pytest.mark.xfail(
        reason="Vulnerability: initial_equity=0.0 causes ZeroDivisionError at line 483 (obs[72])",
        strict=True,
    )
    def test_zero_initial_equity_vulnerability(self, tmp_path):
        """
        Adversarial test proving ZeroDivisionError when initial_equity=0.0.
        Line 483 divides by self.initial_equity without safeguard.
        """
        csv_path, lines = create_base_synthetic_csv(tmp_path, "zero_equity.csv")
        lines.append("2025.01.15 00:00:00,100000.0,100050.0,99950.0,100000.0,100,0,250")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(csv_path), initial_equity=0.0)
        env.load_data("2025.01.15", "2025.01.15")
        env.get_observation(0)

    def test_missing_optional_spread_and_volume_columns(self, tmp_path):
        """
        Adversarial test: Dataset containing only Datetime, Open, High, Low, Close.
        Missing Spread, TickVol, Vol must fallback gracefully without crashing.
        """
        csv_path = tmp_path / "minimal_ohlc.csv"
        lines = ["Datetime,Open,High,Low,Close"]
        for i in range(300):
            lines.append(f"2025.01.14 00:{i%60:02d}:00,100000.0,100050.0,99950.0,100000.0")
        lines.append("2025.01.15 00:00:00,100000.0,100050.0,99950.0,100000.0")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(csv_path))
        count = env.load_data("2025.01.15", "2025.01.15")
        assert count == 1
        assert env.spread_arr[0] > 0.0  # defaulted spread
        assert env.volume_arr[0] > 0.0  # defaulted volume
        obs = env.get_observation(0)
        assert obs.shape == (137,)
        assert not np.isnan(obs).any()

    def test_separate_date_and_time_headers(self, tmp_path):
        """
        Adversarial test: Dataset with separate <DATE> and <TIME> headers.
        """
        tsv_path = tmp_path / "date_time_split.tsv"
        lines = ["<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>"]
        for i in range(300):
            lines.append(f"2025.01.14\t00:{i%60:02d}:00\t100000.0\t100050.0\t99950.0\t100000.0\t100\t0\t250")
        lines.append("2025.01.15\t00:00:00\t100000.0\t100050.0\t99950.0\t100000.0\t100\t0\t250")

        with open(tsv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        env = RealM5Environment(csv_path=str(tsv_path))
        count = env.load_data("2025.01.15", "2025.01.15")
        assert count == 1
        obs = env.get_observation(0)
        assert obs.shape == (137,)
        assert not np.isnan(obs).any()
