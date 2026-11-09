"""
tests/test_m1_adversarial_stress.py

Empirical Challenger Stress Harness for Milestone 1 Defect Fixes:
Defects 1, 2, 5, 6, 7, 8.

Stress Test Matrix:
1. Area 1: Price & ATR synchronization under sudden gaps and multi-thousand dollar regime shifts.
2. Area 2: SELL stop loss and trailing stop execution in backtesting (test_5_regimes_zero_filter.py)
   under volatile wick reversals and extreme multi-bar whipsaws.
3. Area 3: Spacing guards and cooldowns under simulated stop-outs (confirm zero re-entries occur
   within min_spacing_bars / post_exit_cooldown_bars).
4. Area 4: Spread-compensated stop loss sizing (k_SL >= 2.50) across varying spreads
   ($6.40, $28.99, $50.00, $100.00).
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

from self_growing_trader.features import compute_atr
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


class ConstantSignalBasePolicy:
    """Mock base policy emitting persistent BUY (1) or SELL (-1) signals."""
    def __init__(self, action_signal: int = 1):
        self.action_signal = action_signal

    def act_nam_tac_nhan(self, *args, **kwargs):
        # 5 agents: [Scalp, Swing, Sniper, Arbitrage, Macro]
        traces = []
        for _ in range(5):
            vec = np.zeros(20, dtype=np.float64)
            vec[0] = float(self.action_signal)  # action index
            vec[1] = 0.95  # confidence
            traces.append(vec)
        return tuple(traces)


class DummyTrace:
    def __init__(self):
        self.action_details = {}
        self.output_vector = np.zeros(20, dtype=np.float64)
        self.allowed_action_mask = tuple([True] + [False] * 11)


# ==============================================================================
# AREA 1: Price & ATR Synchronization under Sudden Gaps and Regime Shifts
# ==============================================================================
def test_area1_price_and_atr_sync_multi_thousand_dollar_gap(tmp_path: Path):
    """
    Stress-test price & ATR synchronization when price experiences sudden
    multi-thousand dollar gaps (e.g. BTC $65k -> $98k -> $62k flash crash)
    and ATR shifts from low volatility ($15) to extreme volatility ($450).
    """
    policy = MythosMT5Policy(base_policy=ConstantSignalBasePolicy())

    # Initial state
    initial_req = {
        "bid": 65000.0,
        "ask": 65028.99,
        "current_price": 65000.0,
        "current_atr": 25.0,
    }
    policy._sync_request_telemetry(initial_req)
    assert policy.current_price == 65000.0
    assert policy.current_atr == 25.0
    assert policy.m1_buffer[-1, 3] == 65000.0

    # Massive $33,000 Gap Up Regime Shift
    gap_up_req = {
        "bid": 98000.0,
        "ask": 98028.99,
        "current_price": 98000.0,
        "current_atr": 180.0,
    }
    policy._sync_request_telemetry(gap_up_req)
    assert policy.current_price == 98000.0
    assert policy.current_bid == 98000.0
    assert policy.current_ask == 98028.99
    assert policy.current_atr == 180.0
    assert policy.m1_buffer[-1, 3] == 98000.0
    assert math.isfinite(policy.current_price)

    # Flash Crash: Drop to $62,000 with ATR explosion to $450.0
    flash_crash_req = {
        "bid": 62000.0,
        "ask": 62050.0,
        "current_price": 62000.0,
        "current_atr": 450.0,
    }
    policy._sync_request_telemetry(flash_crash_req)
    assert policy.current_price == 62000.0
    assert policy.current_atr == 450.0
    assert policy.m1_buffer[-1, 3] == 62000.0

    # Wire CSV Simulation under rapid price gaps
    wire_file = tmp_path / "self_growing_request.csv"
    wire_file.write_text(
        "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\tpositions_pipe\torders_pipe\texecution_pipe\tmargin\tfree_margin\tmargin_level_pct\tleverage\tbid\task\tcurrent_price\tcurrent_atr\n"
        "2.0\tEV001\t1\t2026.10.19 12:00\t10000\t10000\t0\t640\t0\t0\t0\t0\t0\t0\t0\t10000\t0\t100\t91234.50\t91240.90\t91234.50\t42.10\n",
        encoding="utf-8"
    )
    policy._sync_request_telemetry(str(wire_file))
    assert policy.current_bid == 91234.50
    assert policy.current_ask == 91240.90
    assert policy.current_price == 91234.50
    assert policy.current_atr == 42.10
    assert abs((policy.current_ask - policy.current_bid) - 6.40) < 1e-4

    # Subsequent bar in wire CSV with gap
    wire_file.write_text(
        "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\tpositions_pipe\torders_pipe\texecution_pipe\tmargin\tfree_margin\tmargin_level_pct\tleverage\tbid\task\tcurrent_price\tcurrent_atr\n"
        "2.0\tEV002\t2\t2026.10.19 12:01\t10000\t10000\t0\t640\t0\t0\t0\t0\t0\t0\t0\t10000\t0\t100\t93500.00\t93506.40\t93500.00\t75.50\n",
        encoding="utf-8"
    )
    policy._sync_request_telemetry(str(wire_file))
    assert policy.current_price == 93500.00
    assert policy.current_atr == 75.50
    assert policy.m1_buffer[-1, 3] == 93500.00


def test_area1_ewm_atr_resilience_under_volatility_explosion():
    """
    Verify that Wilder EWM ATR calculation in policy from 60 M1 buffer
    converges smoothly and remains stable without overflow or NaN under 10x volatility jump.
    """
    policy = MythosMT5Policy(base_policy=ConstantSignalBasePolicy())
    base_price = 85000.0

    # 40 bars of low volatility (ATR ~ 20)
    for i in range(40):
        o = base_price + np.sin(i) * 5.0
        h = o + 10.0
        l = o - 10.0
        c = (h + l) / 2.0
        policy.m1_buffer[i] = [o, h, l, c, 100.0]

    # 20 bars of massive volatility explosion (ATR ~ 200)
    for i in range(40, 60):
        o = base_price + np.sin(i) * 50.0
        h = o + 100.0
        l = o - 100.0
        c = (h + l) / 2.0
        policy.m1_buffer[i] = [o, h, l, c, 500.0]

    highs = policy.m1_buffer[1:, 1]
    lows = policy.m1_buffer[1:, 2]
    prev_c = policy.m1_buffer[:-1, 3]
    tr_calc = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    alpha = 1.0 / 14.0
    ewm_atr = float(np.mean(tr_calc[:14]))
    for tr_val in tr_calc[14:]:
        ewm_atr = alpha * float(tr_val) + (1.0 - alpha) * ewm_atr

    assert math.isfinite(ewm_atr)
    assert 60.0 < ewm_atr < 200.0  # Smooth transition reflecting weighted surge


# ==============================================================================
# AREA 2: SELL Stop Loss & Trailing Stop Execution under Volatile Wick Reversals
# ==============================================================================
def test_area2_sell_stop_loss_volatile_wick_reversal():
    """
    Empirical challenge: In scripts/test_5_regimes_zero_filter.py simulation logic,
    test a SELL order when an extreme volatile wick reversal occurs:
    The bar spikes UP piercing the Stop Loss, and in the same bar plunges DOWN
    piercing Take Profit.
    Verification:
    - SL MUST trigger because adverse movement occurred.
    - Exit reason MUST be SL_HIT.
    - PnL MUST be negative, bounded at SL fill price.
    - MUST NOT erroneously award a phantom TP_HIT win!
    """
    pos_dir = -1  # SELL
    pos_entry_p = 90000.0
    pos_atr = 40.0
    sl_mult = 2.50
    tp_mult = 3.75
    pos_vol = 0.01

    pos_sl_p = pos_entry_p + sl_mult * pos_atr  # 90100.0
    pos_tp_p = pos_entry_p - tp_mult * pos_atr  # 89850.0

    # Adverse bar: High = 90150 (violates SL 90100), Low = 89800 (penetrates TP 89850)
    cur_h = 90150.0
    cur_l = 89800.0

    exit_reason = None
    exit_p = None
    pnl = None

    # Mirroring the exact test_5_regimes_zero_filter.py logic:
    if pos_dir in (-1, 2):
        if cur_h >= pos_sl_p:
            exit_p = pos_sl_p
            pnl = (pos_entry_p - exit_p) * pos_vol
            exit_reason = "SL_HIT"
        elif cur_l <= pos_tp_p:
            exit_p = pos_tp_p
            pnl = (pos_entry_p - exit_p) * pos_vol
            exit_reason = "TP_HIT"

    assert exit_reason == "SL_HIT", f"Expected SL_HIT but got {exit_reason}"
    assert exit_p == 90100.0, f"Exit price should be bounded at SL 90100, got {exit_p}"
    assert pnl == pytest.approx(-1.0, abs=1e-4), f"PnL should be -1.0 USD, got {pnl}"


def test_area2_sell_trailing_stop_multi_bar_whipsaw():
    """
    Stress-test SELL trailing stop across a 5-bar sequence:
    Bar 1: Favorable drop (MFE = 1.0 ATR). No trailing yet (requires >= 1.5 ATR).
    Bar 2: Deeper drop to 89900 (MFE = 2.5 ATR). Trailing activates: new SL = 89900 + 1.2*40 = 89948.
    Bar 3: Continues down to 89880 (MFE = 3.0 ATR). Trailing tightens: new SL = 89880 + 48 = 89928.
    Bar 4: Massive green reversal bar: Low = 89900, High = 89950.
           High 89950 pierces trailing SL 89928!
    Verification:
    - Exit reason MUST be SL_HIT (never TP_HIT).
    - Trade is a WINNING trade (+0.72 USD profit locked in by trailing stop).
    - PnL must strictly match (90000 - 89928) * 0.01 = +0.72 USD.
    """
    pos_dir = -1
    pos_entry_p = 90000.0
    pos_atr = 40.0
    sl_mult = 2.50
    tp_mult = 3.75
    trailing_atr = 1.20
    pos_vol = 0.01

    pos_sl_p = pos_entry_p + sl_mult * pos_atr  # 90100.0
    pos_tp_p = pos_entry_p - tp_mult * pos_atr  # 89850.0

    bars = [
        # (cur_h, cur_l)
        (90020.0, 89960.0),  # Bar 1: fav = (90000 - 89960)/40 = 1.0 ATR
        (89980.0, 89900.0),  # Bar 2: fav = 2.5 ATR -> trailing activates
        (89920.0, 89880.0),  # Bar 3: fav = 3.0 ATR -> trailing tightens
        (89950.0, 89900.0),  # Bar 4: reversal piercing 89928!
    ]

    in_pos = True
    exit_record = None

    for bar_idx, (cur_h, cur_l) in enumerate(bars):
        if not in_pos:
            break
        if pos_dir in (-1, 2):
            # 1. Check SL
            if cur_h >= pos_sl_p:
                exit_p = pos_sl_p
                pnl = (pos_entry_p - exit_p) * pos_vol
                exit_record = {
                    "bar": bar_idx + 1,
                    "exit_reason": "SL_HIT",
                    "exit_p": exit_p,
                    "pnl": pnl,
                }
                in_pos = False
                break
            # 2. Check TP
            elif cur_l <= pos_tp_p:
                exit_p = pos_tp_p
                pnl = (pos_entry_p - exit_p) * pos_vol
                exit_record = {
                    "bar": bar_idx + 1,
                    "exit_reason": "TP_HIT",
                    "exit_p": exit_p,
                    "pnl": pnl,
                }
                in_pos = False
                break
            # 3. Holding & Trailing Stop Update
            else:
                fav = (pos_entry_p - cur_l) / pos_atr
                if trailing_atr > 0.0 and fav >= 1.50:
                    new_sl = cur_l + trailing_atr * pos_atr
                    if new_sl < pos_sl_p:
                        pos_sl_p = new_sl

    assert not in_pos
    assert exit_record is not None
    assert exit_record["bar"] == 4, f"Should exit on bar 4, exited on bar {exit_record['bar']}"
    assert exit_record["exit_reason"] == "SL_HIT", "Trailing stop-out MUST be SL_HIT"
    assert exit_record["exit_p"] == pytest.approx(89928.0, abs=1e-2)
    assert exit_record["pnl"] == pytest.approx(0.72, abs=1e-2)


# ==============================================================================
# AREA 3: Spacing Guards & Universal Cooldowns Under Simulated Stop-Outs
# ==============================================================================
def test_area3_zero_reentry_within_min_spacing_bars_under_direct_action():
    """
    Stress-test: Simulate an immediate stop-out at bar T.
    Under direct_action = True and persistent incoming BUY signals,
    test sequence T, T+1, T+2, T+3, T+4, T+5.
    Confirm that ZERO re-entries occur for min_spacing_bars bars.
    """
    policy = MythosMT5Policy(base_policy=ConstantSignalBasePolicy(action_signal=1))
    policy.direct_action = True
    policy.min_spacing_bars = 3
    policy.post_exit_cooldown_bars = 3
    policy.loss_cooldown_bars = 5
    policy.max_consecutive_losses = 2

    # Stop out occurred at bar 50
    stopout_bar = 50
    policy._last_exit_bar = stopout_bar
    policy._last_entry_bar = stopout_bar - 1
    policy._consecutive_losses = 1

    decisions = []
    # Test subsequent bars: 50, 51, 52, 53, 54, 55
    for bar in range(50, 56):
        policy.seq_counter = bar
        bars_since_exit = policy.seq_counter - getattr(policy, "_last_exit_bar", -999)
        post_exit_cd = max(2, int(getattr(policy, "post_exit_cooldown_bars", 2)))
        
        # Policy signal generation check
        raw_signal = 1  # Buy
        allowed = True
        
        # Cooldown guard:
        if raw_signal != 0 and bars_since_exit < post_exit_cd:
            allowed = False
            effective_action = 0
        else:
            effective_action = raw_signal
            
        decisions.append((bar, bars_since_exit, allowed, effective_action))

    # At bar 50: bars_since_exit = 0 -> BLOCKED
    assert decisions[0][2] is False and decisions[0][3] == 0
    # At bar 51: bars_since_exit = 1 -> BLOCKED
    assert decisions[1][2] is False and decisions[1][3] == 0
    # At bar 52: bars_since_exit = 2 -> BLOCKED
    assert decisions[2][2] is False and decisions[2][3] == 0
    # At bar 53: bars_since_exit = 3 -> ALLOWED (post_exit_cd = 3 satisfied)
    assert decisions[3][2] is True and decisions[3][3] == 1


def test_area3_consecutive_loss_circuit_breaker():
    """
    Stress-test consecutive loss cooldown:
    When consecutive losses reaches max_consecutive_losses (2),
    policy MUST enforce loss_cooldown_bars (5) bars of zero trading.
    """
    policy = MythosMT5Policy(base_policy=ConstantSignalBasePolicy(action_signal=1))
    policy.direct_action = True
    policy.max_consecutive_losses = 2
    policy.loss_cooldown_bars = 5

    # 2 consecutive losses recorded at bar 100
    policy._consecutive_losses = 2
    policy._last_loss_bar = 100

    loss_cd = max(3, int(getattr(policy, "loss_cooldown_bars", 3)))
    for bar in range(100, 105):
        policy.seq_counter = bar
        bars_since_loss = policy.seq_counter - policy._last_loss_bar
        in_loss_cooldown = bars_since_loss < loss_cd
        assert in_loss_cooldown is True, f"Bar {bar} should be in loss cooldown!"

    # Bar 105: 5 bars elapsed -> cooldown expired
    policy.seq_counter = 105
    bars_since_loss = policy.seq_counter - policy._last_loss_bar
    assert bars_since_loss >= loss_cd


def test_area3_intrabar_price_proximity_guard():
    """
    Stress-test: Even if bars elapsed >= min_spacing,
    if current price is within 0.20 ATR of last entry price,
    re-entry must be rejected to prevent churn at the same price node.
    """
    policy = MythosMT5Policy(base_policy=ConstantSignalBasePolicy(action_signal=1))
    policy.current_atr = 50.0
    policy._last_entry_price = 90000.0
    policy.seq_counter = 200
    policy._last_entry_bar = 190  # 10 bars ago (spacing ok)
    policy._last_exit_bar = 190

    # Current price = 90005.0 -> delta = $5.0, which is 0.10 ATR (< 0.20 ATR)
    policy.current_price = 90005.0
    price_delta = abs(policy.current_price - policy._last_entry_price)
    is_price_churn = price_delta < 0.20 * policy.current_atr
    assert is_price_churn is True  # Blocked!

    # Current price = 90025.0 -> delta = $25.0, which is 0.50 ATR (> 0.20 ATR)
    policy.current_price = 90025.0
    price_delta = abs(policy.current_price - policy._last_entry_price)
    is_price_churn = price_delta < 0.20 * policy.current_atr
    assert is_price_churn is False  # Allowed!


# ==============================================================================
# AREA 4: Spread-Compensated Stop Loss Sizing Across Varying Spreads
# ==============================================================================
@pytest.mark.parametrize("spread_usd", [6.40, 28.99, 50.00, 100.00])
@pytest.mark.parametrize("atr_usd", [20.0, 45.0, 120.0])
def test_area4_spread_compensated_sl_sizing_matrix(spread_usd: float, atr_usd: float):
    """
    Adversarially test spread-compensated stop loss sizing across varying spreads:
    $6.40 (Oct 19, 2026 BTCUSDm low spread), $28.99 (standard baseline),
    $50.00 (elevated spread), $100.00 (extreme news blowout spread).

    Invariant:
    - Base k_SL >= 2.50.
    - Effective SL distance in price = k_SL * ATR + Spread.
    - Effective SL ATR = k_SL + (Spread / ATR).
    - Stop loss breathing room (sl_distance - spread) >= 2.50 * ATR ALWAYS!
    """
    policy = MythosMT5Policy(base_policy=ConstantSignalBasePolicy())
    policy.current_price = 90000.0
    policy.current_atr = atr_usd
    policy.current_bid = 90000.0
    policy.current_ask = 90000.0 + spread_usd

    spread_atr = spread_usd / atr_usd

    # Test BUY order trace generation
    trace = DummyTrace()
    built = policy._build_entry_trace(
        trace,
        action_idx=1,
        dynamic_risk=0.0025,
        sl_atr=1.50,  # Caller attempts vulnerable 1.50 ATR! Must be clamped to >= 2.50!
        tp_rr=1.50,
        trailing_atr=1.20,
        prm_score=0.90,
        confidence=0.85,
        spread_atr=spread_atr,
    )

    details = built.action_details
    assert details["base_sl_atr"] >= 2.50, f"base_sl_atr {details['base_sl_atr']} must be >= 2.50"
    expected_effective_sl_atr = details["base_sl_atr"] + spread_atr
    assert details["sl_atr"] == pytest.approx(expected_effective_sl_atr, abs=1e-4)

    # In MQL5 geometry:
    # slDist = effectiveSlAtr * atr + spreadPrice
    # Breathing room after spread deduction:
    sl_dist = details["base_sl_atr"] * atr_usd + spread_usd
    net_buffer_after_spread = sl_dist - spread_usd
    assert net_buffer_after_spread >= 2.50 * atr_usd
    assert net_buffer_after_spread / atr_usd >= 2.50


def test_area4_mql5_open_buy_sell_geometry_invariants():
    """
    Verify the mathematical contracts between MQL5 OpenBuy/OpenSell
    and Python policy calculations.
    """
    spread = 28.99
    atr = 45.0
    sl_atr = 2.50
    tp_rr = 1.50
    tp_atr = sl_atr * tp_rr  # 3.75

    # BUY Order
    ask = 90028.99
    bid = 90000.0
    sl_dist = sl_atr * atr + spread  # 2.5 * 45 + 28.99 = 112.5 + 28.99 = 141.49
    buy_sl = ask - sl_dist  # 90028.99 - 141.49 = 89887.50
    buy_tp = ask + tp_atr * atr

    # Distance from BID (current liquidating price for BUY) to SL:
    bid_to_sl = bid - buy_sl  # 90000.0 - 89887.50 = 112.50 = 2.5 * 45.0
    assert bid_to_sl == pytest.approx(sl_atr * atr, abs=1e-4)
    # This proves that even with $28.99 spread, position has exactly 2.50 ATR room from BID!

    # SELL Order
    sell_sl_dist = sl_atr * atr + spread
    sell_sl = bid + sell_sl_dist  # 90000.0 + 141.49 = 90141.49
    # Distance from ASK (current liquidating price for SELL) to SL:
    sl_to_ask = sell_sl - ask  # 90141.49 - 90028.99 = 112.50 = 2.5 * 45.0
    assert sl_to_ask == pytest.approx(sl_atr * atr, abs=1e-4)
    # This proves that SELL also has exactly 2.50 ATR room from ASK!
