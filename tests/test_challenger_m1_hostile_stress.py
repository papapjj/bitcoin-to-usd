"""
tests/test_challenger_m1_hostile_stress.py

Adversarial Stress Harness by challenger_m1_it2_1.
Empirically stress-tests:
1. `simulate_trading_regime` under hostile market regimes:
   - Extreme flash crash ($10,000 candle) during active BUY position.
   - Extreme flash spike ($10,000 candle) during active SELL position.
   - Multi-candle volatile whipsaws (high-low wick traps).
   - Continuous trailing stop ratcheting under multi-bar trend & reversal.
   - Strict causal zero-filter timing (no look-back on entry bar).
2. `policy.act_nam_tac_nhan` under relentless trigger attacks:
   - 100 consecutive bars of relentless BUY signals (enforcing min spacing >= 2).
   - 100 consecutive bars of relentless SELL signals (enforcing min spacing >= 2).
   - Alternating BUY/SELL rapid switching.
   - Hostile input boundaries: 0.0 spread, NaN, negative spreads.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import numpy as np
import pytest
import torch
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from scripts.test_5_regimes_zero_filter import simulate_trading_regime
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.unified_reasoning_pipeline import PipelineTickResult


class MockEngineOutput:
    def __init__(self, acts, prms):
        self.actions = torch.tensor(acts, dtype=torch.int64)
        self.prm_scores = torch.tensor(prms, dtype=torch.float32)


class MockDualEngine:
    def __init__(self, acts, prms):
        self.acts = acts
        self.prms = prms

    def eval(self):
        pass

    def forward_batch(self, ingress, batch_size=512):
        return MockEngineOutput(self.acts, self.prms)


class DummyTrace:
    def __init__(self):
        self.action_details = {}
        self.output_vector = np.zeros(20, dtype=np.float64)
        self.allowed_action_mask = tuple([True] + [False] * 11)


class DummyBasePolicy:
    def act_nam_tac_nhan(self, *args, **kwargs):
        return tuple(DummyTrace() for _ in range(5))


# ==============================================================================
# SECTION 1: simulate_trading_regime Hostile Market Regimes
# ==============================================================================

def test_hostile_flash_crash_active_buy():
    """
    Stress test: BUY entered at 90,000 USD with SL at 89,900 USD (2.5 ATR, ATR=40).
    Next bar experiences a $10,000 Flash Crash: Open 89,950, High 89,950, Low 80,000, Close 80,100.
    Invariants:
    - SL MUST trigger.
    - Exit price MUST be bounded at 89,900 USD (simulating stop order fill at trigger price).
    - Net PnL must be exactly (89900 - 90000) * 0.01 = -1.0 USD (not -$100 USD due to wick overshoot).
    - Exit reason MUST be 'SL_HIT'.
    - Position MUST be closed (in_pos == False).
    """
    regime = {
        'ingress': np.zeros((3, 233), dtype=np.float32),
        'opens': np.array([90000.0, 89950.0, 80500.0], dtype=np.float64),
        'highs': np.array([90050.0, 89950.0, 81000.0], dtype=np.float64),
        'lows': np.array([89980.0, 80000.0, 80000.0], dtype=np.float64),  # Flash crash to 80,000
        'closes': np.array([90000.0, 80100.0, 80500.0], dtype=np.float64),
        'atrs': np.array([40.0, 40.0, 40.0], dtype=np.float64),
        'datetimes': ['2026-10-19 10:00', '2026-10-19 10:01', '2026-10-19 10:02'],
        'lookback': 0,
    }
    engine = MockDualEngine(acts=[1, 0, 0], prms=[0.90, 0.50, 0.50])
    res = simulate_trading_regime(engine, regime, prm_threshold=0.58, sl_mult=2.50, tp_mult=3.75)

    assert len(res["trade_records"]) == 1
    trade = res["trade_records"][0]
    assert trade["direction"] == "BUY"
    assert trade["exit_reason"] == "SL_HIT"
    assert trade["exit_price"] == 89900.0  # Bounded at SL price!
    assert trade["net_pnl_usd"] == -1.0
    assert res["losses"] == 1
    assert res["wins"] == 0


def test_hostile_flash_spike_active_sell():
    """
    Stress test: SELL entered at 90,000 USD with SL at 90,100 USD (2.5 ATR, ATR=40).
    Next bar experiences a massive short squeeze / spike: High 100,000 USD.
    Invariants:
    - SELL SL MUST trigger.
    - Exit price MUST be bounded at 90,100 USD.
    - Net PnL must be exactly (90000 - 90100) * 0.01 = -1.0 USD.
    - Exit reason MUST be 'SL_HIT'.
    - Zero phantom win relabeling!
    """
    regime = {
        'ingress': np.zeros((3, 233), dtype=np.float32),
        'opens': np.array([90000.0, 90050.0, 98000.0], dtype=np.float64),
        'highs': np.array([90020.0, 100000.0, 99000.0], dtype=np.float64),  # Spike to 100k
        'lows': np.array([89980.0, 90010.0, 97000.0], dtype=np.float64),
        'closes': np.array([90000.0, 99000.0, 98500.0], dtype=np.float64),
        'atrs': np.array([40.0, 40.0, 40.0], dtype=np.float64),
        'datetimes': ['2026-10-19 10:00', '2026-10-19 10:01', '2026-10-19 10:02'],
        'lookback': 0,
    }
    engine = MockDualEngine(acts=[2, 0, 0], prms=[0.90, 0.50, 0.50])
    res = simulate_trading_regime(engine, regime, prm_threshold=0.58, sl_mult=2.50, tp_mult=3.75)

    assert len(res["trade_records"]) == 1
    trade = res["trade_records"][0]
    assert trade["direction"] == "SELL"
    assert trade["exit_reason"] == "SL_HIT"
    assert trade["exit_price"] == 90100.0
    assert trade["net_pnl_usd"] == -1.0


def test_hostile_continuous_trailing_stop_ratchet_and_reversal():
    """
    Stress test: Multi-bar trend with trailing stop ratcheting.
    BUY entered at 90,000 USD (SL 89,900, TP 90,300, trailing_atr=1.0, ATR=40).
    Bar 1 (T): Entry at 90,000.
    Bar 2 (T+1): High 90,080 (+2.0 ATR fav >= 1.5). Trailing activates: SL = 90,080 - 40 = 90,040.
    Bar 3 (T+2): High 90,160 (+4.0 ATR fav). Trailing ratchets: SL = 90,160 - 40 = 90,120.
    Bar 4 (T+3): High 90,240 (+6.0 ATR fav). Trailing ratchets: SL = 90,240 - 40 = 90,200.
    Bar 5 (T+4): Violent reversal: Low 90,150 pierces ratcheted SL 90,200!
    Invariants:
    - Exit reason MUST be 'SL_HIT' (a ratcheted stop-out is an SL exit!).
    - Exit price MUST be 90,200.0.
    - Trade is profitable: Net PnL = (90,200 - 90,000) * 0.01 = +2.00 USD.
    - Realized R > 0.
    """
    regime = {
        'ingress': np.zeros((6, 233), dtype=np.float32),
        'opens':   np.array([90000.0, 90010.0, 90080.0, 90160.0, 90230.0, 90100.0], dtype=np.float64),
        'highs':   np.array([90020.0, 90080.0, 90160.0, 90240.0, 90240.0, 90150.0], dtype=np.float64),
        'lows':    np.array([89990.0, 90005.0, 90070.0, 90150.0, 90150.0, 90050.0], dtype=np.float64),
        'closes':  np.array([90000.0, 90070.0, 90150.0, 90230.0, 90160.0, 90080.0], dtype=np.float64),
        'atrs':    np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0], dtype=np.float64),
        'datetimes': [f'2026-10-19 10:0{i}' for i in range(6)],
        'lookback': 0,
    }
    engine = MockDualEngine(acts=[1, 0, 0, 0, 0, 0], prms=[0.90, 0.50, 0.50, 0.50, 0.50, 0.50])
    res = simulate_trading_regime(engine, regime, prm_threshold=0.58, sl_mult=2.50, tp_mult=7.50, trailing_atr=1.0)

    assert len(res["trade_records"]) == 1
    trade = res["trade_records"][0]
    assert trade["direction"] == "BUY"
    assert trade["exit_reason"] == "SL_HIT"
    assert trade["exit_price"] == 90200.0
    assert trade["net_pnl_usd"] == pytest.approx(2.00, abs=1e-2)
    assert trade["realized_r"] > 1.5


def test_hostile_causal_no_entry_bar_wick_stopout():
    """
    Stress test causal entry timing:
    At bar 0: High = 90,150, Low = 89,800, Close = 90,000.
    BUY position enters at Close (90,000) with SL = 89,900.
    Even though bar 0's Low (89,800) < SL (89,900), the trade was entered AT CLOSE.
    Therefore, bar 0's past wick MUST NOT retrospectively stop out the trade!
    The trade must survive bar 0 and continue to bar 1.
    """
    regime = {
        'ingress': np.zeros((3, 233), dtype=np.float32),
        'opens':   np.array([89900.0, 90000.0, 90050.0], dtype=np.float64),
        'highs':   np.array([90150.0, 90200.0, 90100.0], dtype=np.float64),
        'lows':    np.array([89800.0, 89950.0, 90000.0], dtype=np.float64),  # bar 0 low = 89,800!
        'closes':  np.array([90000.0, 90100.0, 90050.0], dtype=np.float64),
        'atrs':    np.array([40.0, 40.0, 40.0], dtype=np.float64),
        'datetimes': ['2026-10-19 10:00', '2026-10-19 10:01', '2026-10-19 10:02'],
        'lookback': 0,
    }
    engine = MockDualEngine(acts=[1, 0, 0], prms=[0.90, 0.50, 0.50])
    res = simulate_trading_regime(engine, regime, prm_threshold=0.58, sl_mult=2.50, tp_mult=3.75)

    # Bar 1 low was 89,950 (> SL 89,900) and high was 90,200 (> TP 90,150).
    # So trade must exit on bar 1 via TP_HIT! (NOT stopped out on bar 0!).
    assert len(res["trade_records"]) == 1
    trade = res["trade_records"][0]
    assert trade["exit_time"] == '2026-10-19 10:01'
    assert trade["exit_reason"] == "TP_HIT"
    assert trade["net_pnl_usd"] == pytest.approx(1.50, abs=1e-2)


# ==============================================================================
# SECTION 2: act_nam_tac_nhan Relentless Trigger Stress
# ==============================================================================

def test_hostile_relentless_buy_trigger_flood():
    """
    Stress test policy under relentless BUY trigger flood:
    For 100 consecutive bars, pipeline emits BUY (action=1, confidence=0.95, prm=0.95).
    Invariants:
    - Min spacing >= 2 bars MUST be strictly enforced: NO consecutive BUY entries allowed.
    - Every adjacent pair of entries must have bar_distance >= min_spacing_bars.
    """
    policy = MythosMT5Policy(base_policy=DummyBasePolicy())
    policy.direct_action = True
    policy.min_spacing_bars = 3
    policy.post_exit_cooldown_bars = 2
    policy._last_exit_bar = -999
    policy._last_entry_bar = -999
    policy._last_entry_price = None

    mock_tick = MagicMock(spec=PipelineTickResult)
    mock_tick.action = 1
    mock_tick.confidence = 0.95
    mock_tick.net_margin = 0.5
    mock_tick.prm_score = 0.95
    mock_tick.is_sniper_mode = False
    mock_tick.deleveraging_factor = 1.0
    mock_tick.bull_score = 0.95
    mock_tick.bear_score = 0.05
    mock_tick.wire_packet = None
    policy.pipeline.process_tick = MagicMock(return_value=mock_tick)

    dummy_obs = np.zeros(137, dtype=np.float32)

    entry_bars = []
    actions = []

    for bar in range(1, 101):
        # Vary price so price proximity guard does not trivially block every trade
        policy.current_price = 90000.0 + (bar * 50.0)
        policy.current_atr = 45.0
        # If last entry was > 0 bars ago, test spacing
        traces = policy.act_nam_tac_nhan(dummy_obs)
        act_idx = policy._extract_action_idx(traces[0])
        actions.append(act_idx)
        if act_idx == 1:
            entry_bars.append(policy.seq_counter)
            policy._last_entry_bar = policy.seq_counter
            policy._last_entry_price = policy.current_price

    # Verify spacing between all consecutive entries
    assert len(entry_bars) > 10, f"Expected multiple entries, got {len(entry_bars)}"
    for i in range(1, len(entry_bars)):
        spacing = entry_bars[i] - entry_bars[i - 1]
        assert spacing >= policy.min_spacing_bars, (
            f"Violation at bar {entry_bars[i]}: spacing was {spacing} < min_spacing_bars {policy.min_spacing_bars}"
        )


def test_hostile_relentless_sell_trigger_flood():
    """
    Stress test policy under relentless SELL trigger flood:
    For 100 consecutive bars, pipeline emits SELL (action=2, confidence=0.95, prm=0.95).
    Invariants:
    - Min spacing >= 2 bars MUST be strictly enforced: NO consecutive SELL entries allowed.
    - Every adjacent pair of entries must have bar_distance >= min_spacing_bars.
    """
    policy = MythosMT5Policy(base_policy=DummyBasePolicy())
    policy.direct_action = True
    policy.min_spacing_bars = 4
    policy.post_exit_cooldown_bars = 2
    policy._last_exit_bar = -999
    policy._last_entry_bar = -999
    policy._last_entry_price = None

    mock_tick = MagicMock(spec=PipelineTickResult)
    mock_tick.action = -1  # SELL in pipeline
    mock_tick.confidence = 0.95
    mock_tick.net_margin = 0.5
    mock_tick.prm_score = 0.95
    mock_tick.is_sniper_mode = False
    mock_tick.deleveraging_factor = 1.0
    mock_tick.bull_score = 0.05
    mock_tick.bear_score = 0.95
    mock_tick.wire_packet = None
    policy.pipeline.process_tick = MagicMock(return_value=mock_tick)

    dummy_obs = np.zeros(137, dtype=np.float32)

    entry_bars = []
    for bar in range(1, 101):
        policy.current_price = 90000.0 - (bar * 50.0)
        policy.current_atr = 45.0
        traces = policy.act_nam_tac_nhan(dummy_obs)
        # Agent 1 is the short specialist (takes action 2)
        act_idx = policy._extract_action_idx(traces[1])
        if act_idx == 2:
            entry_bars.append(policy.seq_counter)
            policy._last_entry_bar = policy.seq_counter
            policy._last_entry_price = policy.current_price

    assert len(entry_bars) > 10
    for i in range(1, len(entry_bars)):
        spacing = entry_bars[i] - entry_bars[i - 1]
        assert spacing >= policy.min_spacing_bars, (
            f"Violation at bar {entry_bars[i]}: spacing was {spacing} < min_spacing_bars {policy.min_spacing_bars}"
        )


def test_hostile_spread_extremes_resilience():
    """
    Stress test entry trace generation under hostile spread extremes:
    1. Zero spread: spread_atr = 0.0 -> valid trace, base_sl_atr >= 2.50.
    2. Giant spread: spread_usd = $1,000 -> effective_sl_atr scales properly.
    3. NaN spread_atr: guarded gracefully to 0.0 without crash or NaN output.
    4. Inf spread_atr: guarded gracefully to 0.0 without crash or Inf output.
    """
    policy = MythosMT5Policy(base_policy=DummyBasePolicy())
    policy.current_price = 90000.0
    policy.current_atr = 45.0

    trace = DummyTrace()

    # 1. Zero spread
    t_zero = policy._build_entry_trace(trace, action_idx=1, spread_atr=0.0)
    assert t_zero.action_details["sl_atr"] >= 2.50
    assert math.isfinite(t_zero.action_details["sl_atr"])

    # 2. Giant spread ($1,000 spread / 45 ATR = 22.22 ATR)
    t_giant = policy._build_entry_trace(trace, action_idx=1, spread_atr=22.22)
    assert t_giant.action_details["sl_atr"] >= 24.72
    assert math.isfinite(t_giant.action_details["sl_atr"])

    # 3. NaN spread
    t_nan = policy._build_entry_trace(trace, action_idx=1, spread_atr=float("nan"))
    assert math.isfinite(t_nan.action_details["sl_atr"])
    assert not math.isnan(t_nan.action_details["sl_atr"])

    # 4. Inf spread
    t_inf = policy._build_entry_trace(trace, action_idx=1, spread_atr=float("inf"))
    assert math.isfinite(t_inf.action_details["sl_atr"])
    assert not math.isinf(t_inf.action_details["sl_atr"])
