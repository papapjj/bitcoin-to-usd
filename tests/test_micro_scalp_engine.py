"""
tests/test_micro_scalp_engine.py

Comprehensive tests for MicroPullbackScalpEngine:
- Bull pullback detection and BUY signal emission
- Bear rally detection and SELL signal emission
- Cooldown and max open positions enforcement
- Micro-Breakeven Ratchet (+0.75 ATR -> Lock +0.10 ATR)
- Resilience to NaN, zero ATR, and edge cases
"""

import numpy as np
import pytest

from self_growing_trader.micro_scalp_engine import (
    MicroPullbackScalpEngine,
    MicroScalpConfig,
    ScalpSignal,
)


def create_dummy_observation(
    bb_pos: float = 0.50,
    ema20_dist: float = 0.0,
    rsi14_norm: float = 0.0,
    rsi5_norm: float = 0.0,
    ema_diff: float = 0.0,
    bar_return: float = 0.0,
    lower_wick: float = 0.05,
    upper_wick: float = 0.05,
    m5_diff: float = 0.0,
    m15_diff: float = 0.0,
) -> np.ndarray:
    """Create a mock 137-feature observation array."""
    obs = np.zeros(137, dtype=float)
    obs[7] = bar_return
    obs[8] = upper_wick
    obs[9] = lower_wick
    obs[11] = ema_diff
    obs[16] = rsi14_norm
    obs[17] = rsi5_norm
    obs[19] = bb_pos
    obs[26] = ema20_dist
    obs[80] = m5_diff
    obs[84] = m15_diff
    return obs


def test_micro_scalp_initialization():
    engine = MicroPullbackScalpEngine()
    assert engine.config.enabled is True
    assert engine.config.sl_atr == 2.5
    assert engine.config.tp_atr == 1.5
    assert engine.config.micro_be_trigger_atr == 0.65
    assert engine.config.micro_be_lock_atr == 0.15
    assert engine.total_signals == 0


def test_bull_pullback_scalp_buy_signal():
    engine = MicroPullbackScalpEngine()
    # Bull regime: P_Bull = 0.70, P_Bear = 0.10, P_Chop = 0.20
    # Pullback: bb_pos = 0.25 (lower BB), ema20_dist = -0.10, rsi14_norm = -0.15 (RSI 42.5)
    obs = create_dummy_observation(
        bb_pos=0.25,
        ema20_dist=-0.10,
        rsi14_norm=-0.15,
        ema_diff=0.20,
        lower_wick=0.08,
    )
    sig = engine.evaluate_scalp_opportunity(
        observation=obs,
        regime_distribution=(0.70, 0.10, 0.20),
        open_positions_count=0,
    )
    assert sig.should_enter is True
    assert sig.direction == +1
    assert sig.action_index == 3  # BUY_LIMIT
    assert sig.sl_atr == 2.5
    assert sig.tp_atr == 1.5
    assert sig.tp_rr == 0.60
    assert sig.micro_be_trigger_atr == 0.65
    assert sig.micro_be_lock_atr == 0.15
    assert engine.total_signals == 1


def test_bear_rally_scalp_sell_signal():
    engine = MicroPullbackScalpEngine()
    # Bear regime: P_Bull = 0.10, P_Bear = 0.75, P_Chop = 0.15
    # Rally: bb_pos = 0.80 (upper BB), ema20_dist = 0.15, rsi14_norm = +0.20 (RSI 60)
    obs = create_dummy_observation(
        bb_pos=0.80,
        ema20_dist=0.15,
        rsi14_norm=0.20,
        ema_diff=-0.25,
        upper_wick=0.09,
    )
    sig = engine.evaluate_scalp_opportunity(
        observation=obs,
        regime_distribution=(0.10, 0.75, 0.15),
        open_positions_count=0,
    )
    assert sig.should_enter is True
    assert sig.direction == -1
    assert sig.action_index == 4  # SELL_LIMIT
    assert sig.sl_atr == 2.5
    assert sig.tp_atr == 1.5
    assert engine.total_signals == 1


def test_cooldown_and_max_positions():
    cfg = MicroScalpConfig(min_bars_between_entries=2, max_open_scalp_positions=2)
    engine = MicroPullbackScalpEngine(config=cfg)
    obs = create_dummy_observation(bb_pos=0.20, ema20_dist=-0.20, rsi14_norm=-0.20, ema_diff=0.20)

    # First entry triggers
    sig1 = engine.evaluate_scalp_opportunity(obs, regime_distribution=(0.70, 0.10, 0.20), open_positions_count=0)
    assert sig1.should_enter is True

    # Immediate next bar (cooldown active)
    sig2 = engine.evaluate_scalp_opportunity(obs, regime_distribution=(0.70, 0.10, 0.20), open_positions_count=1)
    assert sig2.should_enter is False
    assert sig2.reason == "min_bars_cooldown"

    # Bar 2 passes
    sig3 = engine.evaluate_scalp_opportunity(obs, regime_distribution=(0.70, 0.10, 0.20), open_positions_count=1)
    # Now cooldown expired, should enter
    assert sig3.should_enter is True

    # Max positions reached
    sig4 = engine.evaluate_scalp_opportunity(obs, regime_distribution=(0.70, 0.10, 0.20), open_positions_count=2)
    assert sig4.should_enter is False
    assert sig4.reason == "max_open_positions_reached"


def test_micro_be_ratchet_long():
    engine = MicroPullbackScalpEngine()
    entry_p = 60000.0
    atr = 100.0
    current_sl = 59750.0  # -2.5 ATR

    # Case 1: Profit < +0.65 ATR (+60.0 USD) -> No trigger
    cur_p = 60060.0
    res = engine.evaluate_micro_be_ratchet(
        entry_price=entry_p,
        current_price=cur_p,
        direction=+1,
        entry_atr=atr,
        current_sl=current_sl,
    )
    assert res is None

    # Case 2: Profit >= +0.65 ATR (+70.0 USD) -> Triggers Micro-BE to +0.15 ATR (+15.0 USD -> 60015.0)
    cur_p = 60070.0
    new_sl = engine.evaluate_micro_be_ratchet(
        entry_price=entry_p,
        current_price=cur_p,
        direction=+1,
        entry_atr=atr,
        current_sl=current_sl,
    )
    assert new_sl is not None
    assert new_sl == pytest.approx(60015.0)

    # Case 3: Already moved higher (e.g. 60020.0) -> Does not regress
    res_no_regress = engine.evaluate_micro_be_ratchet(
        entry_price=entry_p,
        current_price=cur_p,
        direction=+1,
        entry_atr=atr,
        current_sl=60020.0,
    )
    assert res_no_regress is None


def test_micro_be_ratchet_short():
    engine = MicroPullbackScalpEngine()
    entry_p = 60000.0
    atr = 100.0
    current_sl = 60250.0  # +2.5 ATR

    # Case 1: Profit < +0.65 ATR (cur_p = 59940.0 -> profit = 60.0) -> No trigger
    cur_p = 59940.0
    res = engine.evaluate_micro_be_ratchet(
        entry_price=entry_p,
        current_price=cur_p,
        direction=-1,
        entry_atr=atr,
        current_sl=current_sl,
    )
    assert res is None

    # Case 2: Profit >= +0.65 ATR (cur_p = 59930.0 -> profit = 70.0) -> Triggers Micro-BE to -0.15 ATR (59985.0)
    cur_p = 59930.0
    new_sl = engine.evaluate_micro_be_ratchet(
        entry_price=entry_p,
        current_price=cur_p,
        direction=-1,
        entry_atr=atr,
        current_sl=current_sl,
    )
    assert new_sl is not None
    assert new_sl == pytest.approx(59985.0)


def test_edge_cases_zero_atr_and_nan():
    engine = MicroPullbackScalpEngine()
    # Zero ATR
    res = engine.evaluate_micro_be_ratchet(
        entry_price=60000.0,
        current_price=60100.0,
        direction=+1,
        entry_atr=0.0,
        current_sl=59750.0,
    )
    assert res is None

    # NaN in observation
    obs = np.full(137, np.nan)
    sig = engine.evaluate_scalp_opportunity(obs)
    assert sig.should_enter is False
