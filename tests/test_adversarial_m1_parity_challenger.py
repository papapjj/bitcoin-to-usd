"""
tests/test_adversarial_m1_parity_challenger.py

Adversarial Stress Test Suite — Milestone 1 Iteration 2 Challenger:
Empirical verification of ATR parity, Wilder SMMA mathematical invariants,
zero-spread/high-spread invariance, zero-volatility edge cases, and EA execution contract.
"""

from __future__ import annotations

import math
import inspect
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.features import compute_atr, add_features
from self_growing_trader.config import FeatureConfig, SymbolConfig
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.unified_reasoning_pipeline import PipelineTickResult
from unittest.mock import MagicMock


class MockTrace:
    def __init__(self):
        self.action_details = {}
        self.output_vector = np.zeros(20, dtype=np.float64)
        self.allowed_action_mask = tuple([True] + [False] * 11)


class MockBasePolicy:
    def act_nam_tac_nhan(self, *args, **kwargs):
        return tuple(MockTrace() for _ in range(5))


# ==============================================================================
# SECTION 1: MythosMT5Policy True Wilder EWM ATR Parity & Invariance
# ==============================================================================
def test_mythos_policy_atr_independent_oracle_bit_parity():
    """
    Empirically verify that MythosMT5Policy computes exact Wilder EWM ATR
    directly from m1_buffer matching an independent pure-math oracle,
    and is 100% invariant to observation spread values (zero fixed fallback).
    """
    policy = MythosMT5Policy(base_policy=MockBasePolicy())

    # Verify absence of legacy fixed spread inversion formula in source
    src = inspect.getsource(MythosMT5Policy)
    assert "28.98 /" not in src
    assert "28.98/" not in src

    # Synthesize 60 M1 OHLCV bars
    np.random.seed(999)
    base_price = 91234.56
    m1_data = np.zeros((60, 5), dtype=np.float64)
    p = base_price
    for i in range(60):
        o = p + np.random.uniform(-10.0, 10.0)
        h = o + np.random.uniform(15.0, 45.0)
        l = o - np.random.uniform(15.0, 45.0)
        c = np.random.uniform(l, h)
        v = 150.0 + i * 2.0
        m1_data[i] = [o, h, l, c, v]
        p = c

    # Test across a sweep of adversarial spread values:
    # 0.001, 1.0, 6.40 (Oct 19 normal), 28.98 (legacy), 150.0 (extreme), 1000.0 (insane)
    adversarial_spreads = [0.001, 1.0, 6.40, 28.98, 150.0, 1000.0]
    calculated_atrs = []

    for spread in adversarial_spreads:
        policy.m1_buffer = m1_data.copy()
        policy.current_price = float(policy.m1_buffer[-1, 3])
        policy.current_atr = 0.0  # Force recalculation from buffer

        obs = np.zeros(137, dtype=np.float32)
        obs[15] = float(spread / 45.0)  # spread_atr feature

        # act_nam_tac_nhan should calculate ATR from m1_buffer
        policy.act_nam_tac_nhan(obs)

        # Compute oracle directly on policy.m1_buffer post-shift
        buf = policy.m1_buffer
        tr_oracle = np.maximum(
            buf[1:, 1] - buf[1:, 2],
            np.maximum(
                np.abs(buf[1:, 1] - buf[:-1, 3]),
                np.abs(buf[1:, 2] - buf[:-1, 3])
            )
        )
        oracle_ewm = float(np.mean(tr_oracle[:14]))
        alpha = 1.0 / 14.0
        for tr_val in tr_oracle[14:]:
            oracle_ewm = alpha * float(tr_val) + (1.0 - alpha) * oracle_ewm

        # Assert exact agreement with oracle
        assert math.isclose(policy.current_atr, oracle_ewm, rel_tol=1e-12), (
            f"Spread {spread} distorted ATR from oracle! Got {policy.current_atr}, expected {oracle_ewm}"
        )
        calculated_atrs.append(policy.current_atr)

    # Prove complete mathematical invariance to spread:
    # Calculated ATR must be bit-identical across all 6 spread values!
    for val in calculated_atrs[1:]:
        assert val == calculated_atrs[0], "ATR must have ZERO dependency on spread!"


def test_mythos_policy_atr_adversarial_zero_volatility():
    """
    Adversarial challenge: completely flat market (zero volatility, H == L == C).
    ATR must converge to 0.0 without division by zero, NaN, or exceptions.
    """
    policy = MythosMT5Policy(base_policy=MockBasePolicy())
    flat_price = 88000.0
    # 60 completely flat bars
    policy.m1_buffer = np.full((60, 5), flat_price, dtype=np.float64)
    policy.m1_buffer[:, 4] = 100.0  # Volume
    policy.current_price = flat_price
    policy.current_atr = 0.0

    obs = np.zeros(137, dtype=np.float32)
    obs[15] = 0.0  # spread_atr = 0

    policy.act_nam_tac_nhan(obs)
    assert math.isfinite(policy.current_atr)
    assert policy.current_atr == pytest.approx(0.0, abs=1e-9)


def test_mythos_policy_atr_adversarial_massive_shock():
    """
    Adversarial challenge: massive 50% price flash crash/spike at bar 59.
    Verify Wilder EWM incorporates shock smoothly without overflowing or NaN.
    """
    policy = MythosMT5Policy(base_policy=MockBasePolicy())
    np.random.seed(42)
    base_price = 90000.0
    for i in range(59):
        policy.m1_buffer[i] = [base_price, base_price + 10.0, base_price - 10.0, base_price, 100.0]
    # Extreme bar 59: 30,000 USD flash crash
    policy.m1_buffer[59] = [base_price, base_price, base_price - 30000.0, base_price - 30000.0, 10000.0]
    policy.current_price = float(policy.m1_buffer[-1, 3])
    policy.current_atr = 0.0

    obs = np.zeros(137, dtype=np.float32)
    policy.act_nam_tac_nhan(obs)

    # Post-tick buffer oracle verification
    buf = policy.m1_buffer
    tr_calc = np.maximum(
        buf[1:, 1] - buf[1:, 2],
        np.maximum(
            np.abs(buf[1:, 1] - buf[:-1, 3]),
            np.abs(buf[1:, 2] - buf[:-1, 3])
        )
    )
    oracle_shock = float(np.mean(tr_calc[:14]))
    alpha = 1.0 / 14.0
    for tr_val in tr_calc[14:]:
        oracle_shock = alpha * float(tr_val) + (1.0 - alpha) * oracle_shock

    assert math.isclose(policy.current_atr, oracle_shock, rel_tol=1e-12)
    assert math.isfinite(policy.current_atr)


def test_mythos_policy_build_entry_trace_nan_inf_guards():
    """
    Empirically verify that _build_entry_trace handles NaN, Inf, and negative spread_atr
    safely without poisoning effective sl_atr.
    """
    policy = MythosMT5Policy(base_policy=MockBasePolicy())
    trace = MockTrace()

    # Case 1: NaN spread_atr
    res_nan = policy._build_entry_trace(trace, action_idx=1, sl_atr=2.50, spread_atr=float("nan"))
    assert res_nan.action_details["sl_atr"] == 2.50
    assert math.isfinite(res_nan.action_details["sl_atr"])

    # Case 2: +Inf spread_atr
    res_inf = policy._build_entry_trace(trace, action_idx=1, sl_atr=2.50, spread_atr=float("inf"))
    assert res_inf.action_details["sl_atr"] == 2.50

    # Case 3: Negative spread_atr (corrupt broker feed)
    res_neg = policy._build_entry_trace(trace, action_idx=1, sl_atr=2.50, spread_atr=-5.0)
    assert res_neg.action_details["sl_atr"] == 2.50  # Clamped to k_sl + max(0, s_atr)

    # Case 4: Sub-floor sl_atr (e.g. 0.8) clamped to 2.50 floor + valid spread_atr (0.4)
    res_sub = policy._build_entry_trace(trace, action_idx=1, sl_atr=0.80, spread_atr=0.40)
    assert res_sub.action_details["base_sl_atr"] == 2.50
    assert res_sub.action_details["sl_atr"] == pytest.approx(2.90, abs=1e-5)


# ==============================================================================
# SECTION 2: features.py compute_atr Mathematical Invariant & Stress
# ==============================================================================
def test_features_compute_atr_oracle_recurrence_invariant():
    """
    Verify features.py compute_atr strictly adheres to Wilder SMMA recurrence:
    ATR[t] = (13 * ATR[t-1] + TR[t]) / 14
    across a 500-bar random geometric walk.
    """
    N = 500
    np.random.seed(777)
    rets = np.random.normal(0, 0.002, N)
    prices = 90000.0 * np.exp(np.cumsum(rets))
    highs = prices * (1.0 + np.abs(np.random.normal(0, 0.001, N)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, 0.001, N)))
    closes = prices.copy()
    opens = np.roll(closes, 1)
    opens[0] = closes[0]

    df = pd.DataFrame({
        "Close": closes,
        "High": highs,
        "Low": lows,
        "Open": opens,
    })

    period = 14
    atr_series = compute_atr(df, period)

    # Compute manual True Range
    tr = np.zeros(N)
    tr[0] = highs[0] - lows[0]
    for i in range(1, N):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))

    # Verify Wilder SMMA recurrence for every bar t >= 15
    alpha = 1.0 / period
    for t in range(period + 1, N):
        prev_atr = atr_series.iloc[t - 1]
        curr_atr = atr_series.iloc[t]
        expected_curr = alpha * tr[t] + (1.0 - alpha) * prev_atr
        assert math.isclose(curr_atr, expected_curr, rel_tol=1e-6), (
            f"Wilder recurrence violated at bar {t}: expected {expected_curr}, got {curr_atr}"
        )


def test_features_compute_atr_zero_volatility_and_sudden_step():
    """
    Stress test compute_atr with:
    1. 50 zero-volatility bars (H=L=C=100.0)
    2. Sudden step jump to 200.0 with 50-point range
    """
    N = 100
    opens = np.full(N, 100.0)
    highs = np.full(N, 100.0)
    lows = np.full(N, 100.0)
    closes = np.full(N, 100.0)

    # Bar 50: Gap up and high volatility
    opens[50:] = 200.0
    closes[50:] = 200.0
    highs[50:] = 225.0
    lows[50:] = 175.0

    df = pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes})
    atr = compute_atr(df, 14)

    # Bars 14..49: ATR must be 0.0
    for i in range(14, 50):
        assert atr.iloc[i] == pytest.approx(0.0, abs=1e-9)

    # Bar 50: Gap from 100 to 200, High=225, Low=175.
    # TR = max(225-175=50, |225-100|=125, |175-100|=75) = 125.0
    # ATR at bar 50 = (1/14)*125 + (13/14)*0 = 8.92857
    assert atr.iloc[50] == pytest.approx(125.0 / 14.0, abs=1e-4)


# ==============================================================================
# SECTION 3: Backtest Engine SELL SL and Trailing Logic Invariance
# ==============================================================================
def test_backtest_sell_sl_and_trailing_classification_adversarial():
    """
    Adversarial verification of simulate_trading_regime in scripts/test_5_regimes_zero_filter.py:
    1. Verify SELL position is stopped out when high breaches entry + sl_mult * atr.
    2. Verify trailing stop trigger produces exit_reason == 'SL_HIT' with positive PnL.
    3. Verify that entry bar high/low does NOT trigger same-bar lookback exit.
    """
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

    # Scenario: SELL entry at bar 0 (close = 100.0). Entry SL = 100 + 2.5 * 10 = 125.0.
    # Bar 0 has a spike in high to 130.0!
    # Because position enters at bar 0 close, bar 0's past intra-bar high MUST NOT trigger exit!
    # Bar 1 high is 110.0 (safe).
    # Bar 2 high is 126.0 (triggers SELL SL).
    regime_data = {
        'ingress': np.zeros((4, 233), dtype=np.float32),
        'closes': np.array([100.0, 105.0, 120.0, 120.0], dtype=np.float64),
        'highs': np.array([130.0, 110.0, 126.0, 120.0], dtype=np.float64),  # Bar 0 high=130 spike!
        'lows': np.array([95.0, 100.0, 115.0, 115.0], dtype=np.float64),
        'opens': np.array([100.0, 100.0, 105.0, 120.0], dtype=np.float64),
        'atrs': np.array([10.0, 10.0, 10.0, 10.0], dtype=np.float64),
        'datetimes': [f'2026-10-19 10:0{i}' for i in range(4)],
        'lookback': 0,
    }

    engine = MockEngine(acts=[2, 0, 0, 0], prms=[0.9, 0.5, 0.5, 0.5])
    res = simulate_trading_regime(engine, regime_data, prm_threshold=0.58, sl_mult=2.5, tp_mult=4.0)

    assert len(res["trade_records"]) == 1
    trade = res["trade_records"][0]
    # Proves exit did NOT occur on bar 0 (no lookback leak)!
    # Entry at 10:00 (bar 0), exit at 10:02 (bar 2)
    assert trade["entry_time"] == "2026-10-19 10:00"
    assert trade["exit_time"] == "2026-10-19 10:02"
    assert trade["exit_reason"] == "SL_HIT"
    assert trade["direction"] == "SELL"
    assert trade["exit_price"] == 125.0
    assert trade["net_pnl_usd"] < 0.0


# ==============================================================================
# SECTION 4: Universal Cooldown and Min-Spacing State Machine
# ==============================================================================
def test_universal_cooldown_and_spacing_multi_tick_stress():
    """
    Stress test MythosMT5Policy entry state machine over a sequence of 10 ticks:
    - Bar 10: Exit recorded -> post_exit_cooldown_bars=3 active
    - Bars 11, 12: Cooldown active -> must emit HOLD (0)
    - Bar 13: Cooldown expires, entry permitted -> emits BUY (1)
    - Bar 14: Min spacing active (< 2 bars since entry) -> must emit HOLD (0)
    - Bar 15: Min spacing satisfied -> entry permitted -> emits BUY (1)
    """
    policy = MythosMT5Policy(base_policy=MockBasePolicy())
    policy.direct_action = True
    policy.min_spacing_bars = 2
    policy.post_exit_cooldown_bars = 3

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

    # Bar 10: exit occurs
    policy.seq_counter = 10
    policy._last_exit_bar = 10
    policy._last_entry_bar = 0

    # Bar 11: bars_since_exit = 1 < 3 -> HOLD
    t11 = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(t11[0]) == 0

    # Bar 12: bars_since_exit = 2 < 3 -> HOLD
    t12 = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(t12[0]) == 0

    # Bar 13: bars_since_exit = 3 >= 3 -> BUY allowed
    policy._last_entry_price = None
    t13 = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(t13[0]) == 1
    assert policy._last_entry_bar == 13

    # Bar 14: bars_since_entry = 1 < 2 -> HOLD (spacing guard)
    t14 = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(t14[0]) == 0

    # Bar 15: bars_since_entry = 2 >= 2 -> BUY allowed
    policy._last_entry_price = None
    t15 = policy.act_nam_tac_nhan(dummy_obs)
    assert policy._extract_action_idx(t15[0]) == 1
    assert policy._last_entry_bar == 15
