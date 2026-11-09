from __future__ import annotations

from dataclasses import dataclass, field
from math import exp, floor
from typing import Iterable

import numpy as np

from .execution_contract import decode_execution_geometry
from .orders import Action
from .protocol_contract import position_limit_enabled


def _sigmoid(value: float) -> float:
    if value > 60.0:
        return 1.0
    if value < -60.0:
        return 0.0
    return 1.0 / (1.0 + exp(-float(value)))


def _normalize_price(value: float, digits: int) -> float:
    scale = 10.0 ** int(digits)
    return floor(float(value) * scale + 0.5) / scale


@dataclass(frozen=True)
class ReplaySymbolSpec:
    digits: int
    point: float
    tick_size: float
    tick_value: float
    volume_min: float
    volume_max: float
    volume_step: float
    stops_level: int = 0


@dataclass
class ReplayPosition:
    ticket: int
    side: int
    volume: float
    entry: float
    sl: float
    tp: float
    open_time_msc: int
    initial_risk_distance: float
    owner_expert: str = ""
    trailing_stop_atr: float = 0.0
    sl_atr: float = 2.5


@dataclass
class ReplayPendingOrder:
    ticket: int
    action: int
    side: int
    volume: float
    entry: float
    sl: float
    tp: float
    create_time_msc: int
    owner_expert: str = ""
    trailing_stop_atr: float = 0.0
    sl_atr: float = 2.5


@dataclass(frozen=True)
class ReplayExecutionEvent:
    time_msc: int
    kind: str
    status: str
    action: int
    order_ticket: int = 0
    deal_ticket: int = 0
    position_ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    profit: float = 0.0


@dataclass
class Model1ExecutionEngine:
    spec: ReplaySymbolSpec
    balance: float = 10_000.0
    max_positions: int = 5
    fixed_sl_atr: float = 2.5
    fixed_tp_atr: float = 5.0
    technical_min_rr: float = 2.0
    entry_offset_atr: float = 5.0
    positions: list[ReplayPosition] = field(default_factory=list)
    orders: list[ReplayPendingOrder] = field(default_factory=list)
    events: list[ReplayExecutionEvent] = field(default_factory=list)
    next_ticket: int = 2
    bid: float = 0.0
    ask: float = 0.0
    time_msc: int = 0
    last_trailing_bar_msc: int = -1

    def _ticket(self) -> int:
        ticket = int(self.next_ticket)
        self.next_ticket += 1
        return ticket

    def _volume(self, risk_pct: float, entry: float, sl: float) -> float:
        distance = abs(float(entry) - float(sl))
        if min(risk_pct, entry, sl, distance, self.spec.tick_size, self.spec.tick_value) <= 0.0:
            return 0.0
        equity = self.equity
        requested = equity * min(max(float(risk_pct), 0.0), 1.0) / (
            distance / self.spec.tick_size * self.spec.tick_value
        )
        volume = floor(requested / self.spec.volume_step) * self.spec.volume_step
        volume = max(self.spec.volume_min, min(self.spec.volume_max, volume))
        return _normalize_price(volume, 8)

    @property
    def floating_profit(self) -> float:
        total = 0.0
        for position in self.positions:
            exit_price = self.bid if position.side == 1 else self.ask
            total += position.side * (exit_price - position.entry) / self.spec.tick_size * self.spec.tick_value * position.volume
        return total

    @property
    def equity(self) -> float:
        return float(self.balance + self.floating_profit)

    def on_tick(self, *, time_msc: int, bid: float, ask: float) -> None:
        self.time_msc, self.bid, self.ask = int(time_msc), float(bid), float(ask)
        self._fill_pending()
        self._apply_stops()
        bar_msc = (self.time_msc // 60_000) * 60_000
        if bar_msc != self.last_trailing_bar_msc:
            self.last_trailing_bar_msc = bar_msc
            self._apply_trailing_stops()

    def _fill_pending(self) -> None:
        active: list[ReplayPendingOrder] = []
        for order in self.orders:
            fill = (
                (order.action == int(Action.BUY_LIMIT) and self.ask <= order.entry)
                or (order.action == int(Action.SELL_LIMIT) and self.bid >= order.entry)
                or (order.action == int(Action.BUY_STOP) and self.ask >= order.entry)
                or (order.action == int(Action.SELL_STOP) and self.bid <= order.entry)
            )
            if (
                not fill
                or (
                    position_limit_enabled(self.max_positions)
                    and len(self.positions) >= self.max_positions
                )
            ):
                active.append(order)
                continue
            deal = self._ticket()
            position = ReplayPosition(
                ticket=order.ticket, side=order.side, volume=order.volume,
                entry=order.entry, sl=order.sl, tp=order.tp,
                open_time_msc=self.time_msc,
                initial_risk_distance=abs(order.entry - order.sl),
                owner_expert=order.owner_expert,
                trailing_stop_atr=order.trailing_stop_atr,
                sl_atr=order.sl_atr,
            )
            self.positions.insert(0, position)
            self.events.append(ReplayExecutionEvent(
                self.time_msc, "pending_fill", "FILLED", order.action,
                order_ticket=order.ticket, deal_ticket=deal, position_ticket=position.ticket,
                price=order.entry, volume=order.volume,
            ))
        self.orders = active

    def _apply_stops(self) -> None:
        active: list[ReplayPosition] = []
        for position in self.positions:
            if position.side == 1:
                close_price = position.sl if self.bid <= position.sl else position.tp if self.bid >= position.tp else 0.0
            else:
                close_price = position.sl if self.ask >= position.sl else position.tp if self.ask <= position.tp else 0.0
            if close_price <= 0.0:
                active.append(position)
                continue
            reason = "SL" if close_price == position.sl else "TP"
            self._settle(position, close_price, reason, action=0)
        self.positions = active

    def _apply_trailing_stops(self) -> None:
        for position in self.positions:
            distance = float(position.trailing_stop_atr)
            if distance <= 0.0:
                continue
            # Model-1 tapes do not carry a separate execution ATR side channel
            # per tick. The immutable initial risk is SL_ATR * entry ATR.
            entry_atr = position.initial_risk_distance / max(float(position.sl_atr), 1e-12)
            if position.side == 1:
                candidate = _normalize_price(self.bid - distance * entry_atr, self.spec.digits)
                if candidate < self.bid:
                    position.sl = max(position.sl, candidate)
            else:
                candidate = _normalize_price(self.ask + distance * entry_atr, self.spec.digits)
                if candidate > self.ask:
                    position.sl = min(position.sl, candidate)

    def _settle(self, position: ReplayPosition, price: float, reason: str, *, action: int, volume: float | None = None) -> None:
        close_volume = position.volume if volume is None else float(volume)
        profit = position.side * (float(price) - position.entry) / self.spec.tick_size * self.spec.tick_value * close_volume
        self.balance += profit
        self.events.append(ReplayExecutionEvent(
            self.time_msc, "position_close", reason, int(action), deal_ticket=self._ticket(),
            position_ticket=position.ticket, price=float(price), volume=close_volume, profit=profit,
        ))

    def execute(
        self,
        *,
        action: int,
        outputs: Iterable[float],
        risk_pct: float,
        atr: float,
        reference_close: float,
        owner_expert: str = "",
    ) -> ReplayExecutionEvent:
        values = np.asarray(tuple(outputs), dtype=float)
        action = int(action)
        if action == int(Action.HOLD):
            event = ReplayExecutionEvent(self.time_msc, "action", "NOOP_HOLD", action)
            self.events.append(event)
            return event
        if action == int(Action.CANCEL_ORDERS):
            count = len(self.orders)
            self.orders.clear()
            event = ReplayExecutionEvent(self.time_msc, "action", f"CANCELLED_{count}", action)
            self.events.append(event)
            return event
        if action in (int(Action.CLOSE_WORST), int(Action.CLOSE_BEST), int(Action.CLOSE_ALL)):
            return self._close_control(action, owner_expert)
        if action not in range(int(Action.MARKET_BUY), int(Action.SELL_STOP) + 1):
            event = ReplayExecutionEvent(self.time_msc, "action", "NO_EFFECT", action)
            self.events.append(event)
            return event
        if position_limit_enabled(self.max_positions) and len(self.positions) >= self.max_positions:
            event = ReplayExecutionEvent(self.time_msc, "action", "NO_EFFECT", action)
            self.events.append(event)
            return event

        side = 1 if action in (int(Action.MARKET_BUY), int(Action.BUY_LIMIT), int(Action.BUY_STOP)) else -1
        if action in (int(Action.MARKET_BUY), int(Action.MARKET_SELL)):
            entry = self.ask if side == 1 else self.bid
        else:
            raw_offset = (_sigmoid(float(values[15])) * 2.0 - 1.0) * self.entry_offset_atr
            offset = max(0.45, abs(raw_offset)) * float(atr)
            entry = float(reference_close) + (offset if action in (int(Action.SELL_LIMIT), int(Action.BUY_STOP)) else -offset)
            minimum = max((max(0, self.spec.stops_level) + 1) * self.spec.point, 2.0 * self.spec.point)
            if action == int(Action.BUY_LIMIT):
                entry = min(entry, self.ask - minimum)
            elif action == int(Action.SELL_LIMIT):
                entry = max(entry, self.bid + minimum)
            elif action == int(Action.BUY_STOP):
                entry = max(entry, self.ask + minimum)
            else:
                entry = min(entry, self.bid - minimum)
            entry = _normalize_price(entry, self.spec.digits)
        geometry = decode_execution_geometry(
            values,
            min_risk_pct=float(risk_pct),
            max_risk_pct=float(risk_pct),
            technical_min_rr=float(self.technical_min_rr),
            fixed_sl_atr=float(self.fixed_sl_atr),
            fixed_tp_atr=float(self.fixed_tp_atr),
            fixed_risk_pct=float(risk_pct),
        )
        sl = _normalize_price(entry - side * geometry.sl_atr * float(atr), self.spec.digits)
        tp = _normalize_price(entry + side * geometry.tp_atr * float(atr), self.spec.digits)
        volume = self._volume(risk_pct, entry, sl)
        if volume <= 0.0:
            event = ReplayExecutionEvent(self.time_msc, "action", "INVALID_POLICY_VOLUME", action)
            self.events.append(event)
            return event
        if action in (int(Action.MARKET_BUY), int(Action.MARKET_SELL)):
            order_ticket, deal_ticket = self._ticket(), self._ticket()
            position = ReplayPosition(
                order_ticket, side, volume, entry, sl, tp, self.time_msc,
                abs(entry - sl), owner_expert, geometry.trailing_stop_atr,
                geometry.sl_atr,
            )
            self.positions.insert(0, position)
            event = ReplayExecutionEvent(self.time_msc, "market_fill", "FILLED", action, order_ticket, deal_ticket, order_ticket, entry, volume)
        else:
            order_ticket = self._ticket()
            pending = ReplayPendingOrder(
                order_ticket, action, side, volume, entry, sl, tp, self.time_msc,
                owner_expert, geometry.trailing_stop_atr, geometry.sl_atr,
            )
            self.orders.insert(0, pending)
            event = ReplayExecutionEvent(self.time_msc, "pending_place", "PLACED", action, order_ticket=order_ticket, price=entry, volume=volume)
        self.events.append(event)
        return event

    def _close_control(self, action: int, owner_expert: str) -> ReplayExecutionEvent:
        eligible = [item for item in self.positions if item.owner_expert == owner_expert and owner_expert]
        if not eligible:
            event = ReplayExecutionEvent(self.time_msc, "action", "NO_EFFECT", action)
            self.events.append(event)
            return event
        if action == int(Action.CLOSE_ALL):
            selected = list(eligible)
        else:
            def score(position: ReplayPosition) -> float:
                exit_price = self.bid if position.side == 1 else self.ask
                return position.side * (exit_price - position.entry) / max(position.initial_risk_distance, 1e-12)
            selected = [max(eligible, key=score) if action == int(Action.CLOSE_BEST) else min(eligible, key=score)]
        last: ReplayExecutionEvent | None = None
        for position in selected:
            price = self.bid if position.side == 1 else self.ask
            self._settle(position, price, "POLICY", action=action)
            self.positions.remove(position)
            last = self.events[-1]
        assert last is not None
        return last
