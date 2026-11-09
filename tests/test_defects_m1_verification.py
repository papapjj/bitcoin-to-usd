"""
tests/test_defects_m1_verification.py

Comprehensive Automated Verification Suite for Milestone 1:
Defects 1, 2, 5, 6, 7, 8 Root-Cause Fixes.

Covers:
1. Defect 1: Price & ATR Synchronization (no dummy 93440.0/28.98 drift, real telemetry ingest).
2. Defect 2: Accurate SELL stop loss, trailing stop classification as SL_HIT, intrabar spike checks.
3. Defect 5: Zero-spacing re-entry prevention, universal cooldown enforcement under direct_action.
4. Defect 6: Spread-compensated Stop Loss sizing (k_SL >= 2.50, sl_dist >= k_SL * ATR + spread).
5. Defect 7: Wilder EWM ATR parity from M1 buffer without inverted spread formula.
6. Defect 8: Static Pandas vs Live MT5 Streaming Feature Parity (Wilder SMMA alpha=1/14).
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.features import compute_atr, add_features
from self_growing_trader.config import FeatureConfig, SymbolConfig
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


class DummyTrace:
    def __init__(self):
        self.action_details = {}
        self.output_vector = np.zeros(20, dtype=np.float64)
        self.allowed_action_mask = tuple([True] + [False] * 11)


class DummyBasePolicy:
    def act_nam_tac_nhan(self, *args, **kwargs):
        return tuple(DummyTrace() for _ in range(5))


# ==============================================================================
# DEFECT 1: Price & ATR Synchronization
# ==============================================================================
def test_defect_1_price_and_atr_synchronization(tmp_path: Path):
    """
    Verify that MythosMT5Policy ingests real terminal prices (Bid, Ask, current price)
    and real ATR directly from requests without dummy 93440.0 assumptions or
    synthetic drift accumulation.
    """
    policy = MythosMT5Policy(base_policy=DummyBasePolicy())

    # 1. Telemetry ingestion via request dictionary
    req = {
        "bid": 95250.25,
        "ask": 95279.24,
        "current_price": 95250.25,
        "current_atr": 45.80,
    }
    policy._sync_request_telemetry(req)

    assert policy.current_bid == 95250.25
    assert policy.current_ask == 95279.24
    assert policy.current_price == 95250.25
    assert policy.current_atr == 45.80
    assert abs((policy.current_ask - policy.current_bid) - 28.99) < 1e-4

    # 2. Telemetry ingestion via CSV file (EA self_growing_request.csv parity)
    req_csv = tmp_path / "self_growing_request.csv"
    req_csv.write_text("bid,ask,current_price,current_atr\n96110.50,96139.49,96110.50,52.30\n", encoding="utf-8")
    policy._sync_request_telemetry(str(req_csv))

    assert policy.current_bid == 96110.50
    assert policy.current_ask == 96139.49
    assert policy.current_price == 96110.50
    assert policy.current_atr == 52.30

    # 3. Check m1_buffer does not accumulate double-counted deltas
    fake_audit = {"Close": 96150.0}
    policy._sync_feature_audit(fake_audit)
    assert policy.m1_buffer[-1, 3] == 96150.0
    assert policy.current_price == 96150.0


# ==============================================================================
# DEFECT 2: Simulation Masking Bug in Backtesting
# ==============================================================================
def test_defect_2_sell_stop_loss_and_trailing_classification():
    """
    Verify that in zero-filter evaluation (scripts/test_5_regimes_zero_filter.py):
    1. SELL positions (act == 2 / pos_dir in (-1, 2)) strictly trigger SL when price rises.
    2. Trailing stop-outs are classified as SL_HIT (never masquerading as TP_HIT).
    3. Exits are evaluated authentically through simulate_trading_regime without look-back bias.
    """
    import torch
    from scripts.test_5_regimes_zero_filter import simulate_trading_regime

    class MockOut:
        def __init__(self, acts, prms):
            self.actions = torch.tensor(acts)
            self.prm_scores = torch.tensor(prms)

    class MockEngine:
        def __init__(self, acts, prms):
            self.acts = acts
            self.prms = prms
        def eval(self): pass
        def forward_batch(self, ingress, batch_size=512):
            return MockOut(self.acts, self.prms)

    # Case 1: SELL Adverse excursion hits stop loss
    regime_data_sl = {
        'ingress': np.zeros((5, 233), dtype=np.float32),
        'closes': np.array([90000.0, 90120.0, 90120.0, 90120.0, 90120.0], dtype=np.float64),
        'highs': np.array([90010.0, 90150.0, 90130.0, 90130.0, 90130.0], dtype=np.float64),
        'lows': np.array([89990.0, 90080.0, 90100.0, 90100.0, 90100.0], dtype=np.float64),
        'opens': np.array([90000.0, 90010.0, 90120.0, 90120.0, 90120.0], dtype=np.float64),
        'atrs': np.array([40.0, 40.0, 40.0, 40.0, 40.0], dtype=np.float64),
        'datetimes': [f'2026-10-19 10:0{i}' for i in range(5)],
        'lookback': 0,
    }
    mock_engine_sl = MockEngine(acts=[2, 0, 0, 0, 0], prms=[0.9, 0.5, 0.5, 0.5, 0.5])
    res_sl = simulate_trading_regime(mock_engine_sl, regime_data_sl, prm_threshold=0.58, sl_mult=2.5, tp_mult=3.5)

    assert len(res_sl["trade_records"]) == 1
    trade_sl = res_sl["trade_records"][0]
    assert trade_sl["direction"] == "SELL"
    assert trade_sl["exit_reason"] == "SL_HIT"
    assert trade_sl["net_pnl_usd"] == -1.0
    assert trade_sl["exit_price"] == 90100.0

    # Case 2: Trailing stop logic for SELL classifies trailing exit as SL_HIT (not TP_HIT)
    regime_data_trail = {
        'ingress': np.zeros((5, 233), dtype=np.float32),
        'closes': np.array([90000.0, 89920.0, 89950.0, 89950.0, 89950.0], dtype=np.float64),
        'highs': np.array([90010.0, 90010.0, 89960.0, 89960.0, 89960.0], dtype=np.float64),
        'lows': np.array([89990.0, 89900.0, 89930.0, 89930.0, 89930.0], dtype=np.float64),
        'opens': np.array([90000.0, 90000.0, 89920.0, 89920.0, 89920.0], dtype=np.float64),
        'atrs': np.array([40.0, 40.0, 40.0, 40.0, 40.0], dtype=np.float64),
        'datetimes': [f'2026-10-19 10:0{i}' for i in range(5)],
        'lookback': 0,
    }
    mock_engine_trail = MockEngine(acts=[2, 0, 0, 0, 0], prms=[0.9, 0.5, 0.5, 0.5, 0.5])
    res_trail = simulate_trading_regime(mock_engine_trail, regime_data_trail, prm_threshold=0.58, sl_mult=2.5, tp_mult=3.5, trailing_atr=1.2)

    assert len(res_trail["trade_records"]) == 1
    trade_trail = res_trail["trade_records"][0]
    assert trade_trail["direction"] == "SELL"
    assert trade_trail["exit_reason"] == "SL_HIT"  # Trailing stop hit MUST be SL_HIT, NOT TP_HIT!
    assert trade_trail["net_pnl_usd"] == pytest.approx(0.52, abs=1e-2)
    assert trade_trail["exit_price"] == 89948.0


# ==============================================================================
# DEFECT 5: Zero-Spacing Re-Entry Flaw
# ==============================================================================
def test_defect_5_min_spacing_and_universal_cooldown():
    """
    Verify that MythosMT5Policy's actual decision dispatch (act_nam_tac_nhan) enforces:
    1. Universal post-exit cooldown (bars_since_exit < post_exit_cooldown_bars -> HOLD)
    2. Minimum spacing bars (bars_since_entry < min_spacing_bars -> HOLD)
    even when direct_action is True, forbidding immediate re-entry.
    """
    from unittest.mock import MagicMock
    from research_reasoning_world_class.src.unified_reasoning_pipeline import PipelineTickResult

    policy = MythosMT5Policy(base_policy=DummyBasePolicy())
    policy.direct_action = True
    policy.min_spacing_bars = 3
    policy.post_exit_cooldown_bars = 3
    policy.loss_cooldown_bars = 5
    policy.max_consecutive_losses = 2

    # Mock pipeline.process_tick to return an authorized BUY signal
    mock_tick = MagicMock(spec=PipelineTickResult)
    mock_tick.action = 1
    mock_tick.confidence = 0.95
    mock_tick.net_margin = 0.5
    mock_tick.prm_score = 0.92
    mock_tick.is_sniper_mode = False
    mock_tick.deleveraging_factor = 1.0
    mock_tick.bull_score = 0.95
    mock_tick.bear_score = 0.05
    mock_tick.wire_packet = None
    policy.pipeline.process_tick = MagicMock(return_value=mock_tick)

    dummy_obs = np.zeros(137, dtype=np.float32)

    # Case 1: Post-exit cooldown active (exited 1 bar ago < post_exit_cooldown_bars=3)
    policy.seq_counter = 100
    policy._last_exit_bar = 100  # next bar seq_counter is 101, bars_since_exit = 1 < 3
    policy._last_entry_bar = 50
    traces_cd = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(traces_cd[0]) == 0  # Re-entry blocked by cooldown!

    # Case 2: Cooldown expired, but minimum spacing active (entered 1 bar ago < min_spacing_bars=3)
    policy._last_exit_bar = 50
    policy._last_entry_bar = policy.seq_counter  # next bar bars_since_entry = 1 < 3
    traces_spacing = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(traces_spacing[0]) == 0  # Re-entry blocked by spacing!

    # Case 3: Both cooldown and spacing satisfied -> Entry permitted!
    policy._last_exit_bar = 50
    policy._last_entry_bar = 50
    policy._last_entry_price = None
    traces_cleared = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(traces_cleared[0]) == 1  # BUY authorized!


# ==============================================================================
# DEFECT 6: Sub-Spread Stop Loss Sizing
# ==============================================================================
def test_defect_6_spread_compensated_stop_loss_sizing():
    """
    Verify spread-compensated SL sizing: k_SL * ATR + Spread with k_SL >= 2.50.
    Ensures stop loss breathing room comfortably exceeds broker spread ($28.99).
    """
    policy = MythosMT5Policy(base_policy=DummyBasePolicy())
    policy.scalp_sl_atr = 2.50
    policy.sniper_sl_atr = 3.50
    policy.current_price = 90000.0
    policy.current_atr = 45.0
    policy.current_bid = 90000.0
    policy.current_ask = 90028.99  # 28.99 USD spread

    # BUY SL calculation in _build_entry_trace
    dummy_trace = DummyTrace()
    spread_atr = 28.99 / 45.0
    built_trace = policy._build_entry_trace(
        dummy_trace,
        action_idx=1,
        dynamic_risk=0.0025,
        sl_atr=2.50,
        tp_rr=1.50,
        trailing_atr=1.20,
        prm_score=0.88,
        confidence=0.85,
        spread_atr=spread_atr,
    )

    details = built_trace.action_details
    assert details["base_sl_atr"] >= 2.50
    # effective sl_atr = k_sl + spread_atr
    assert details["sl_atr"] == pytest.approx(2.50 + spread_atr, abs=1e-4)

    # Sub-spread sizing rejection: if someone passes sl_atr = 0.50, clamped to >= 2.50
    clamped_trace = policy._build_entry_trace(
        dummy_trace,
        action_idx=1,
        sl_atr=0.50,
        spread_atr=spread_atr,
    )
    assert clamped_trace.action_details["base_sl_atr"] == 2.50


# ==============================================================================
# DEFECT 7: Fixed Spread ATR Formula Removal & Wilder EWM Parity
# ==============================================================================
def test_defect_7_wilder_ewm_atr_no_spread_inversion():
    """
    Verify complete removal of inverted spread formula (28.98 / spread_atr) and
    accurate calculation of Wilder EWM ATR directly by MythosMT5Policy's buffer routine
    under dynamic/low spread (e.g. $6.40 on Oct 19, 2026).
    """
    import inspect

    policy = MythosMT5Policy(base_policy=DummyBasePolicy())

    # 1. AST / Code verification: inverted formula '28.98 /' must NOT exist in policy logic
    src = inspect.getsource(MythosMT5Policy)
    assert "28.98 /" not in src, "Inverted spread formula '28.98 /' must be completely removed!"

    # 2. Populate policy.m1_buffer with 60 bars with ~40 USD true range
    np.random.seed(42)
    base_price = 90000.0
    for i in range(60):
        o = base_price + np.random.randn() * 5.0
        h = o + 20.0 + np.random.rand() * 5.0
        l = o - 20.0 - np.random.rand() * 5.0
        c = (h + l) / 2.0
        policy.m1_buffer[i] = [o, h, l, c, 100.0]

    # Align current_price with buffer to avoid artificial gap
    policy.current_price = float(policy.m1_buffer[-1, 3])
    policy.current_atr = 0.0

    # 3. Invoke policy.act_nam_tac_nhan with low-spread observation (Oct 19, 2026 conditions)
    # Under the buggy old code:
    # spread = 6.40, atr = 40.0 -> spread_atr = 0.16
    # est_atr = 28.98 / 0.16 = 181.125 USD (A 4.5X DISTORTION!)
    obs = np.zeros(137, dtype=np.float32)
    obs[15] = 0.16  # spread_atr
    policy.act_nam_tac_nhan(obs)

    # Verify policy.current_atr computed by policy's own Wilder EWM routine
    assert 35.0 <= policy.current_atr <= 55.0
    assert abs(policy.current_atr - 181.125) > 100.0  # Zero 4.5x distortion!


# ==============================================================================
# DEFECT 8: Static Pandas vs Live MT5 Streaming Feature Parity
# ==============================================================================
def test_defect_8_pandas_wilder_smma_atr_parity():
    """
    Verify that `src/self_growing_trader/features.py` compute_atr implements
    Wilder's SMMA (alpha = 1/period) matching MT5 `iATR`, rather than a boxcar SMA.
    """
    N = 100
    np.random.seed(123)
    # Generate synthetic price series with sudden volatility jump at bar 50
    opens = np.full(N, 100.0)
    highs = np.full(N, 102.0)
    lows = np.full(N, 98.0)
    closes = np.full(N, 101.0)

    # Volatility jump
    highs[50:] += 10.0
    lows[50:] -= 10.0

    df = pd.DataFrame({
        "Datetime": pd.date_range("2026-10-19", periods=N, freq="min"),
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Spread": [640] * N,
    })

    period = 14
    atr_series = compute_atr(df, period)

    # Verify return type and length
    assert len(atr_series) == N
    assert not atr_series.iloc[period:].isna().any()

    # Before bar 50, TR = 4.0, so ATR should be exactly 4.0
    assert atr_series.iloc[20] == pytest.approx(4.0, abs=1e-4)

    # At bar 50, TR jumps to 24.0.
    # At bar 64 (14 bars after step):
    # Rolling SMA would be exactly 24.0 (step function).
    # Wilder SMMA decays smoothly: 24.0 - (24.0 - 4.0) * (13/14)^14 = ~16.94
    atr_64 = atr_series.iloc[64]
    assert atr_64 < 23.5  # Proves exponential smoothing rather than boxcar SMA cliff!
    assert atr_64 > 15.0


def test_defects_all_integrated_sanity():
    """Integrated sanity test validating coexistence of all 6 fixes."""
    policy = MythosMT5Policy(base_policy=DummyBasePolicy())
    assert policy.min_spacing_bars >= 2
    assert policy.post_exit_cooldown_bars >= 2
    assert policy.loss_cooldown_bars >= 3
    assert policy.scalp_sl_atr >= 2.50
    assert policy.sniper_sl_atr >= 2.50
    assert policy.current_price > 0.0
    assert policy.current_atr > 0.0
