from __future__ import annotations

import numpy as np

from self_growing_trader.model1_execution import Model1ExecutionEngine, ReplaySymbolSpec
from self_growing_trader.orders import Action


def _engine() -> Model1ExecutionEngine:
    return Model1ExecutionEngine(ReplaySymbolSpec(2, 0.01, 0.01, 0.01, 0.01, 200.0, 0.01))


def test_market_buy_uses_ask_and_tick_value_risk_volume() -> None:
    engine = _engine()
    engine.on_tick(time_msc=1_000, bid=100.0, ask=101.0)
    event = engine.execute(action=int(Action.MARKET_BUY), outputs=np.zeros(20), risk_pct=0.01, atr=2.0, reference_close=100.0)
    assert event.price == 101.0
    assert engine.positions[0].sl == 96.0
    assert engine.positions[0].tp == 111.0
    assert event.volume == 20.0


def test_zero_position_limit_allows_more_than_five_positions() -> None:
    engine = _engine()
    engine.max_positions = 0
    engine.on_tick(time_msc=1_000, bid=100.0, ask=101.0)
    for _ in range(8):
        event = engine.execute(
            action=int(Action.MARKET_BUY),
            outputs=np.zeros(20),
            risk_pct=0.001,
            atr=2.0,
            reference_close=100.0,
        )
        assert event.status == "FILLED"
    assert len(engine.positions) == 8


def test_buy_limit_fills_only_when_ask_reaches_entry() -> None:
    engine = _engine()
    engine.on_tick(time_msc=1_000, bid=100.0, ask=101.0)
    outputs = np.zeros(20)
    outputs[15] = 0.0
    placed = engine.execute(action=int(Action.BUY_LIMIT), outputs=outputs, risk_pct=0.01, atr=2.0, reference_close=100.0)
    assert placed.price == 99.1
    engine.on_tick(time_msc=2_000, bid=98.2, ask=99.2)
    assert not engine.positions
    engine.on_tick(time_msc=3_000, bid=98.1, ask=99.1)
    assert engine.positions[0].entry == 99.1


def test_stop_loss_uses_bid_for_long_and_settles_balance() -> None:
    engine = _engine()
    engine.on_tick(time_msc=1_000, bid=100.0, ask=101.0)
    engine.execute(action=int(Action.MARKET_BUY), outputs=np.zeros(20), risk_pct=0.01, atr=2.0, reference_close=100.0)
    engine.on_tick(time_msc=2_000, bid=96.0, ask=97.0)
    assert not engine.positions
    assert engine.balance == 9900.0
