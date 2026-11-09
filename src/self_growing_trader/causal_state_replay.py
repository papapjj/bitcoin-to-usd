"""Causal portfolio-state replay for the seven-choice snapshot dataset.

The counterfactual exporter deliberately has no order API, so its position and
account portions of the 137-value vector are flat.  This module adds an
opt-in, deterministic behaviour trajectory *offline*.  At each decision it
emits the state that was produced by earlier ticks and earlier scheduled
actions, then processes the current bar.  Future ticks are therefore never
used to build the current input.

The result is a research dataset.  It is useful for checking whether the
network can read position/risk state, but it is not a replacement for a
stateful MT5 capture: the scheduled actions are synthetic and promotion must
still use a real tester run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np

from .model1_replay import (
    EXECUTION_SPREAD_INDEX,
    ReplayPositionSnapshot,
    compose_canonical_inputs,
    build_dynamic_input_blocks,
)
from .orders import Action, MIN_PENDING_OFFSET_ATR


STATE_REPLAY_SCHEMA_VERSION = 1
RAW_SNAPSHOT_SCHEMA_VERSION = 2
FEATURE_COUNT = 137
# V16's 17-bin offset codebook maps the requested 0.75 ATR setting to bin 7,
# whose actual placement is 0.588235 ATR.  The replay must use the effective
# codebook value so its synthetic orders share the label geometry.
V16_FIXED_ENTRY_OFFSET_ATR = 0.5882352941176471

# One action is scheduled only after the previous synthetic lifecycle has had
# enough time to expire/close.  The cycle contains HOLD plus all six entries,
# so every choice is deliberately represented without making the action depend
# on a future label.
BEHAVIOUR_ACTION_CYCLE = (
    Action.HOLD,
    Action.MARKET_BUY,
    Action.MARKET_SELL,
    Action.BUY_LIMIT,
    Action.SELL_LIMIT,
    Action.BUY_STOP,
    Action.SELL_STOP,
)
LONG_ACTIONS = frozenset((Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP))
PENDING_ACTIONS = frozenset((Action.BUY_LIMIT, Action.SELL_LIMIT, Action.BUY_STOP, Action.SELL_STOP))


@dataclass(frozen=True)
class CausalStateReplayConfig:
    horizon_bars: int = 48
    pending_expiry_bars: int = 5
    cycle_bars: int = 0
    fixed_entry_offset_atr: float = V16_FIXED_ENTRY_OFFSET_ATR
    fixed_sl_atr: float = 1.50
    fixed_tp_rr: float = 2.00
    risk_fraction: float = 0.01
    initial_balance: float = 10_000.0
    max_positions: int = 5
    max_holding_bars: int = 48
    min_daily_trades: int = 0
    max_daily_trades: int = 0

    def validate(self) -> None:
        if self.horizon_bars < 2:
            raise ValueError("horizon_bars phai tu 2 tro len")
        if self.pending_expiry_bars < 1 or self.pending_expiry_bars > self.horizon_bars:
            raise ValueError("pending_expiry_bars khong hop le")
        if self.cycle_bars and self.cycle_bars < self.horizon_bars + self.pending_expiry_bars + 2:
            raise ValueError("cycle_bars qua ngan cho mot vong doi")
        if self.fixed_entry_offset_atr <= 0.0:
            raise ValueError("fixed_entry_offset_atr phai duong")
        if self.fixed_sl_atr <= 0.0 or self.fixed_tp_rr <= 0.0:
            raise ValueError("fixed_sl_atr/fixed_tp_rr phai duong")
        if not 0.0 < self.risk_fraction < 1.0:
            raise ValueError("risk_fraction phai nam trong (0,1)")
        if self.initial_balance <= 0.0:
            raise ValueError("initial_balance phai duong")
        if self.max_positions < 1:
            raise ValueError("max_positions phai it nhat 1")
        if self.max_holding_bars < 1:
            raise ValueError("max_holding_bars phai duong")

    @property
    def effective_cycle_bars(self) -> int:
        return int(self.cycle_bars or (self.horizon_bars + self.pending_expiry_bars + 2))


@dataclass
class _Pending:
    action: Action
    order_price: float
    atr: float
    anchor_price: float
    created_row: int


@dataclass
class _Position:
    action: Action
    side: int
    entry: float
    sl: float
    tp: float
    risk_distance: float
    risk_money: float
    open_time: int
    fill_row: int


def _parse_time(value: object) -> datetime:
    text = str(value).strip()
    for fmt in ("%Y.%m.%d %H:%M:%S", "%Y.%m.%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"thoi gian snapshot khong hop le: {value!r}")


def _parse_pipe(value: object, *, name: str, dtype: Any = np.float64) -> np.ndarray:
    values = np.fromstring(str(value or ""), sep="|", dtype=dtype)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} khong hop le")
    return values


def _parse_inputs(value: object) -> np.ndarray:
    values = _parse_pipe(value, name="inputs_pipe", dtype=np.float64)
    if values.size != FEATURE_COUNT:
        raise ValueError(f"inputs_pipe phai co {FEATURE_COUNT} gia tri")
    return values


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _pending_price(action: Action, close: float, atr: float, bid: float, ask: float, minimum_distance: float, digits: int, offset_atr: float) -> float:
    offset = max(float(MIN_PENDING_OFFSET_ATR), float(offset_atr)) * max(float(atr), 1e-12)
    raw = float(close) + (-offset if action in (Action.BUY_LIMIT, Action.SELL_STOP) else offset)
    if action == Action.BUY_LIMIT:
        value = min(raw, ask - minimum_distance)
    elif action == Action.SELL_LIMIT:
        value = max(raw, bid + minimum_distance)
    elif action == Action.BUY_STOP:
        value = max(raw, ask + minimum_distance)
    elif action == Action.SELL_STOP:
        value = min(raw, bid - minimum_distance)
    else:
        raise ValueError("chi lenh cho moi co gia dat")
    return round(float(value), int(digits))


class _Replay:
    def __init__(self, cfg: CausalStateReplayConfig) -> None:
        self.cfg = cfg
        self.balance = float(cfg.initial_balance)
        self.position: _Position | None = None
        self.pending: _Pending | None = None
        self.day_key: object = None
        self.day_trades = 0
        self.action_counts = {action.name: 0 for action in BEHAVIOUR_ACTION_CYCLE}
        self.fill_counts = {action.name: 0 for action in BEHAVIOUR_ACTION_CYCLE if action != Action.HOLD}
        self.close_counts = {"SL": 0, "TP": 0, "HORIZON": 0}
        self.realized_r = 0.0

    def _sync_day(self, decision_time: datetime) -> None:
        key = decision_time.date()
        if key != self.day_key:
            self.day_key = key
            self.day_trades = 0

    def _position_snapshot(self) -> tuple[ReplayPositionSnapshot, ...]:
        position = self.position
        if position is None:
            return ()
        return (ReplayPositionSnapshot(
            ticket=f"causal-{position.fill_row}",
            side=position.side,
            volume=1.0,
            entry=position.entry,
            sl=position.sl,
            tp=position.tp,
            initial_risk_distance=position.risk_distance,
            open_time=position.open_time,
        ),)

    def _unrealized_r(self, bar_close: float, spread_price: float) -> float:
        position = self.position
        if position is None:
            return 0.0
        exit_price = float(bar_close) if position.side == 1 else float(bar_close) + float(spread_price)
        return float(np.clip(position.side * (exit_price - position.entry) / max(position.risk_distance, 1e-12), -5.0, 5.0))

    def vector_for_row(self, row: dict[str, str]) -> np.ndarray:
        inputs = _parse_inputs(row["inputs_pipe"])
        sample_time = _parse_time(row["sample_time"])
        decision_time = _parse_time(row["decision_bar_time"])
        close = float(row["close_bid"])
        atr = max(float(row["atr"]), 1e-12)
        point = float(row["point"])
        ticks_bid = _parse_pipe(row["tick_bid_pipe"], name="tick_bid_pipe")
        ticks_ask = _parse_pipe(row["tick_ask_pipe"], name="tick_ask_pipe")
        live_spread = max(0.0, float(ticks_ask[0] - ticks_bid[0]))
        spread_points = live_spread / max(point, 1e-12)
        risk_money = max(float(self.balance) * float(self.cfg.risk_fraction), 1e-12)
        unrealized = self._unrealized_r(close, live_spread)
        equity = float(self.balance + unrealized * risk_money)
        open_risk_ratio = float(self.cfg.risk_fraction if self.position is not None else 0.0)
        position_features, account_features, execution_features = build_dynamic_input_blocks(
            positions=self._position_snapshot(),
            bar_time=sample_time,
            decision_time=decision_time,
            bar_close=close,
            atr=atr,
            spread_points=spread_points,
            point=point,
            balance=self.balance,
            equity=equity,
            open_risk_ratio=open_risk_ratio,
            day_trades=self.day_trades,
            max_positions=self.cfg.max_positions,
            max_holding_bars=self.cfg.max_holding_bars,
            min_daily_trades=self.cfg.min_daily_trades,
            max_daily_trades=self.cfg.max_daily_trades,
        )
        # The execution spread field is sampled from the decision tick in the
        # stateful path.  The exporter keeps a completed-bar spread, so update
        # only this one dynamic slot and preserve every market/tail value.
        market = inputs.copy()
        market[EXECUTION_SPREAD_INDEX] = live_spread / atr
        return compose_canonical_inputs(
            market,
            position_features=position_features,
            account_features=account_features,
            execution_position_features=execution_features,
        )

    def _open(self, action: Action, bid: float, ask: float, atr: float, anchor: float, time_msc: int, row_index: int) -> None:
        side = 1 if action in LONG_ACTIONS else -1
        entry = float(ask if side == 1 else bid)
        risk_distance = max(float(self.cfg.fixed_sl_atr) * float(atr), 1e-12)
        sl = float(anchor - risk_distance if side == 1 else anchor + risk_distance)
        tp_distance = risk_distance * float(self.cfg.fixed_tp_rr)
        tp = float(anchor + tp_distance if side == 1 else anchor - tp_distance)
        self.position = _Position(
            action=action,
            side=side,
            entry=entry,
            sl=sl,
            tp=tp,
            risk_distance=risk_distance,
            risk_money=max(float(self.balance) * float(self.cfg.risk_fraction), 1e-12),
            open_time=int(time_msc // 1000),
            fill_row=int(row_index),
        )
        self.fill_counts[action.name] += 1
        self.day_trades += 1

    def _close(self, exit_price: float, reason: str) -> None:
        position = self.position
        if position is None:
            return
        net_r = float(position.side * (float(exit_price) - position.entry) / max(position.risk_distance, 1e-12))
        self.balance += net_r * position.risk_money
        self.realized_r += net_r
        self.close_counts[reason] += 1
        self.position = None

    def _process_tick(self, bid: float, ask: float, row_index: int, tick_index: int, time_msc: int) -> None:
        pending = self.pending
        # Do not let a pending order created at the decision tick fill on that
        # same tick.  This matches the exporter replay, which searches from
        # ``placed + 1`` for pending fills.
        if pending is not None and not (pending.created_row == row_index and tick_index == 0):
            action = pending.action
            if (
                (action == Action.BUY_LIMIT and ask <= pending.order_price)
                or (action == Action.SELL_LIMIT and bid >= pending.order_price)
                or (action == Action.BUY_STOP and ask >= pending.order_price)
                or (action == Action.SELL_STOP and bid <= pending.order_price)
            ):
                self.pending = None
                self._open(action, bid, ask, pending.atr, pending.anchor_price, time_msc, row_index)
                return

        position = self.position
        if position is None:
            return
        # When both levels are touched on one tick, the conservative stop wins,
        # matching the fixed-geometry label replay.
        if position.side == 1:
            if bid <= position.sl:
                self._close(bid, "SL")
            elif bid >= position.tp:
                self._close(bid, "TP")
        else:
            if ask >= position.sl:
                self._close(ask, "SL")
            elif ask <= position.tp:
                self._close(ask, "TP")

    def _scheduled_action(self, row_index: int) -> Action:
        block = row_index // self.cfg.effective_cycle_bars
        phase = row_index % self.cfg.effective_cycle_bars
        if phase != 0:
            return Action.HOLD
        return BEHAVIOUR_ACTION_CYCLE[block % len(BEHAVIOUR_ACTION_CYCLE)]

    def advance_row(self, row: dict[str, str], row_index: int) -> tuple[np.ndarray, Action]:
        decision_time = _parse_time(row["decision_bar_time"])
        self._sync_day(decision_time)
        # Expiry is checked at the start of the expiry bar.  This matches the
        # label replay's [placed + 1, bar_start + expiry) search interval.
        if self.pending is not None and row_index - self.pending.created_row >= self.cfg.pending_expiry_bars:
            self.pending = None
        vector = self.vector_for_row(row)
        action = self._scheduled_action(row_index)
        self.action_counts[action.name] += 1

        bid = _parse_pipe(row["tick_bid_pipe"], name="tick_bid_pipe")
        ask = _parse_pipe(row["tick_ask_pipe"], name="tick_ask_pipe")
        times = _parse_pipe(row["tick_time_msc_pipe"], name="tick_time_msc_pipe")
        atr = max(float(row["atr"]), 1e-12)
        if action != Action.HOLD and self.position is None and self.pending is None:
            first_bid = float(bid[0])
            first_ask = float(ask[0])
            if action in (Action.MARKET_BUY, Action.MARKET_SELL):
                anchor = first_ask if action == Action.MARKET_BUY else first_bid
                self._open(action, first_bid, first_ask, atr, anchor, int(times[0]), row_index)
            elif action in PENDING_ACTIONS:
                minimum_distance = float(row["pending_min_distance"])
                order_price = _pending_price(
                    action,
                    float(row["close_bid"]),
                    atr,
                    first_bid,
                    first_ask,
                    minimum_distance,
                    int(float(row["digits"])),
                    self.cfg.fixed_entry_offset_atr,
                )
                self.pending = _Pending(
                    action=action,
                    order_price=order_price,
                    atr=atr,
                    anchor_price=order_price,
                    created_row=row_index,
                )

        for tick_index, (tick_bid, tick_ask, tick_time) in enumerate(zip(bid, ask, times)):
            self._process_tick(float(tick_bid), float(tick_ask), row_index, tick_index, int(tick_time))

        if self.position is not None and row_index - self.position.fill_row + 1 >= self.cfg.horizon_bars:
            # The row has already been processed, so the last tick is a causal
            # finite-horizon close available before the next decision.
            exit_price = float(bid[-1] if self.position.side == 1 else ask[-1])
            self._close(exit_price, "HORIZON")
        return vector, action


def augment_snapshot_file(
    snapshot_file: str | Path,
    output_file: str | Path,
    *,
    config: CausalStateReplayConfig | None = None,
) -> dict[str, Any]:
    """Write a stateful research snapshot while retaining the v2 input schema."""
    cfg = config or CausalStateReplayConfig()
    cfg.validate()
    source = Path(snapshot_file).resolve()
    output = Path(output_file).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError("output phai khac snapshot nguon")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    replay = _Replay(cfg)
    rows = 0
    first_sample: datetime | None = None
    last_sample: datetime | None = None
    fieldnames: list[str] | None = None
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle, temporary.open("w", encoding="utf-8", newline="") as out:
            reader = csv.DictReader(handle, delimiter="\t")
            required = {
                "schema_version", "sample_time", "close_bid", "atr", "point",
                "pending_min_distance", "digits", "causal_input", "decision_bar_time",
                "tick_count", "tick_time_msc_pipe", "tick_bid_pipe", "tick_ask_pipe", "inputs_pipe",
            }
            missing = required - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"tep snapshot thieu cot: {sorted(missing)}")
            fieldnames = list(reader.fieldnames or ()) + [
                "state_replay_schema_version", "state_replay_action",
                "state_replay_state_source", "state_replay_causal_input",
            ]
            writer = csv.DictWriter(out, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            for raw in reader:
                row = dict(raw)
                if int(float(row["schema_version"])) != RAW_SNAPSHOT_SCHEMA_VERSION:
                    raise ValueError("schema snapshot khong duoc ho tro")
                if int(float(row["causal_input"])) != 1:
                    raise ValueError("snapshot khong duoc danh dau causal_input=1")
                sample = _parse_time(row["sample_time"])
                if last_sample is not None and sample <= last_sample:
                    raise ValueError("sample_time phai tang nghiem ngat")
                first_sample = first_sample or sample
                last_sample = sample
                tick_count = int(float(row["tick_count"]))
                ticks_bid = _parse_pipe(row["tick_bid_pipe"], name="tick_bid_pipe")
                ticks_ask = _parse_pipe(row["tick_ask_pipe"], name="tick_ask_pipe")
                ticks_time = _parse_pipe(row["tick_time_msc_pipe"], name="tick_time_msc_pipe")
                if tick_count != len(ticks_bid) or ticks_bid.shape != ticks_ask.shape or ticks_bid.shape != ticks_time.shape:
                    raise ValueError("tick_count va ba chuoi tick phai cung kich thuoc")
                vector, action = replay.advance_row(row, rows)
                row["inputs_pipe"] = "|".join(f"{value:.10f}" for value in vector)
                row["state_replay_schema_version"] = str(STATE_REPLAY_SCHEMA_VERSION)
                row["state_replay_action"] = str(int(action))
                row["state_replay_state_source"] = "causal_synthetic_behavior_schedule_v1"
                row["state_replay_causal_input"] = "1"
                writer.writerow(row)
                rows += 1
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    report = {
        "schema_version": STATE_REPLAY_SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": _sha256_file(source),
        "output": str(output),
        "rows": int(rows),
        "feature_count": FEATURE_COUNT,
        "config": dict(cfg.__dict__) | {"effective_cycle_bars": cfg.effective_cycle_bars},
        "sample_time_start": first_sample.strftime("%Y-%m-%dT%H:%M:%S") if first_sample else "",
        "sample_time_end": last_sample.strftime("%Y-%m-%dT%H:%M:%S") if last_sample else "",
        "behavior_action_counts": replay.action_counts,
        "filled_action_counts": replay.fill_counts,
        "close_reason_counts": replay.close_counts,
        "realized_r_sum": float(replay.realized_r),
        "production_eligible": False,
        "state_source": "causal_synthetic_behavior_schedule_v1",
        "causal_contract": {
            "input_emitted_before_current_row_ticks": True,
            "future_ticks_used_for_current_input": False,
            "future_labels_used_for_behavior_action": False,
            "scheduled_actions": [int(action) for action in BEHAVIOUR_ACTION_CYCLE],
            "note": "Dùng để kiểm tra khả năng đọc trạng thái; cần capture MT5 thật trước khi quảng bá.",
        },
    }
    report_path = output.with_suffix(output.suffix + ".state.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
