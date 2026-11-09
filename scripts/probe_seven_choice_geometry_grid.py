#!/usr/bin/env python3
"""Đo cận dưới nhân quả khi được chọn cả cách vào và hình học lệnh.

Đây là phép đo nghiên cứu, không tạo ứng viên hay bật cập nhật trọng số. Nó
dùng đúng giá khớp của lệnh chờ, ATR tại lúc khớp và lưới SL/RR của hợp đồng.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.seven_choice_teacher import (  # noqa: E402
    ENTRY_CHOICES,
    LONG_CHOICES,
    PENDING_CHOICES,
    SL_ATR_BINS,
    TP_RR_BINS,
    _causal_regimes,
    _fixed_geometry_ticks,
    _nearest_index,
    _normalized_pending_price,
    _offset_codebook,
    _parse_inputs,
    _parse_tick_pipe,
    _parse_time,
    _placement_on_ticks,
)
from self_growing_trader.orders import Action  # noqa: E402


def _lower(state: list[float], z: float, minimum: int) -> float | None:
    n, total, squared = state
    if n < minimum:
        return None
    mean = total / n
    variance = max(0.0, (squared - total * total / n) / max(1.0, n - 1.0))
    return mean - z * math.sqrt(variance) / math.sqrt(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--horizon-bars", type=int, default=48)
    ap.add_argument("--pending-expiry-bars", type=int, default=5)
    ap.add_argument(
        "--holding-horizon-from-fill", action="store_true",
        help="Tinh 48 nen tu luc khop, dong bo voi POSITION_TIME cua MT5.",
    )
    ap.add_argument(
        "--risk-denominator-from-fill", action="store_true",
        help="Chia R theo khoang gia khop den SL bat bien.",
    )
    ap.add_argument("--fixed-entry-offset-atr", type=float, default=0.75)
    ap.add_argument(
        "--pending-fill-price-mode", choices=("order", "touch"), default="order",
        help="Gia khop lenh cho: dung gia dat MT5; touch chi de truy vet du lieu cu.",
    )
    ap.add_argument("--action-history-min-count", type=int, default=48)
    ap.add_argument("--action-lcb-z", type=float, default=1.0)
    ap.add_argument("--action-lcb-min-r", type=float, default=0.05)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    rows: list[dict[str, object]] = []
    with Path(args.snapshots).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for raw in reader:
            bids = _parse_tick_pipe(raw["tick_bid_pipe"], name="tick_bid_pipe")
            asks = _parse_tick_pipe(raw["tick_ask_pipe"], name="tick_ask_pipe")
            times = _parse_tick_pipe(raw["tick_time_msc_pipe"], name="tick_time_msc_pipe")
            if not (len(bids) == len(asks) == len(times)):
                raise ValueError("chuoi nhip gia khong cung kich thuoc")
            rows.append({
                "time": _parse_time(raw["sample_time"]),
                "open": float(raw["open_bid"]), "close": float(raw["close_bid"]),
                "atr": float(raw["atr"]), "spread": float(raw["spread_price"]),
                "point": float(raw["point"]), "distance": float(raw["pending_min_distance"]),
                "digits": int(float(raw["digits"])), "bid": bids, "ask": asks,
            })
            if args.max_rows and len(rows) >= args.max_rows:
                break
    if len(rows) <= args.horizon_bars + 32:
        raise ValueError("snapshot qua ngan")
    lookahead_bars = args.horizon_bars + (
        args.pending_expiry_bars if args.holding_horizon_from_fill else 0
    )
    usable = len(rows) - lookahead_bars
    close = np.asarray([float(row["close"]) for row in rows], dtype=np.float64)
    open_ = np.asarray([float(row["open"]) for row in rows], dtype=np.float64)
    atr = np.asarray([float(row["atr"]) for row in rows], dtype=np.float64)
    times = np.asarray([row["time"].strftime("%Y-%m-%dT%H:%M:%S") for row in rows], dtype="U19")
    counts = np.asarray([len(row["bid"]) for row in rows], dtype=np.int64)
    global_starts = np.concatenate((np.asarray([0], dtype=np.int64), np.cumsum(counts)))
    all_bid = np.concatenate([row["bid"] for row in rows])
    all_ask = np.concatenate([row["ask"] for row in rows])
    regimes = _causal_regimes(close[:usable], open_[:usable], atr[:usable], times[:usable])
    train_end = int(math.floor(usable * 0.70))
    validation_end = int(math.floor(usable * 0.85))
    splits = np.full(usable, "test", dtype="U10")
    purge_bars = lookahead_bars
    splits[:max(0, train_end - purge_bars)] = "train"
    splits[train_end:max(train_end, validation_end - purge_bars)] = "validation"
    splits[validation_end:] = "test"
    splits[max(0, train_end - purge_bars):train_end] = "purged"
    splits[max(train_end, validation_end - purge_bars):validation_end] = "purged"

    sl_count, tp_count = len(SL_ATR_BINS), len(TP_RR_BINS)
    offsets = _offset_codebook(5.0)
    offset_index = _nearest_index(offsets, args.fixed_entry_offset_atr)
    exact: defaultdict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    direction: defaultdict[tuple[int, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    global_stats: defaultdict[tuple[int, int, int], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    ring: list[np.ndarray | None] = [None] * (lookahead_bars + 1)
    chosen: dict[str, list[float]] = {name: [] for name in ("train", "validation", "test")}
    chosen_actions: dict[str, Counter[str]] = {name: Counter() for name in chosen}
    chosen_geometry: dict[str, Counter[str]] = {name: Counter() for name in chosen}

    def update(state: list[float], value: float) -> None:
        state[0] += 1.0
        state[1] += value
        state[2] += value * value

    for index in range(usable):
        row = rows[index]
        tick_start = int(global_starts[index])
        tick_stop = int(global_starts[index + lookahead_bars])
        bid = all_bid[tick_start:tick_stop]
        ask = all_ask[tick_start:tick_stop]
        bars = global_starts[index:index + lookahead_bars + 1] - tick_start
        grid = np.zeros((6, sl_count, tp_count), dtype=np.float32)
        filled = np.zeros(6, dtype=np.bool_)
        for action in ENTRY_CHOICES:
            placement = _placement_on_ticks(
                action, 0, offset_index if action in PENDING_CHOICES else 0,
                float(row["close"]), float(row["atr"]), float(row["distance"]),
                int(row["digits"]), offsets, bid, ask, bars, args.pending_expiry_bars,
                args.pending_fill_price_mode,
            )
            if placement is None or placement[0] >= len(bid) - 1:
                continue
            fill_pos, order_price, entry_price, entry_bid, _, fill_bar = placement
            exit_horizon = args.horizon_bars if args.holding_horizon_from_fill else len(bid) - fill_pos
            action_bid = bid[fill_pos:fill_pos + exit_horizon]
            action_ask = ask[fill_pos:fill_pos + exit_horizon]
            result = _fixed_geometry_ticks(
                action, order_price, float(entry_price), float(entry_bid),
                float(row["atr"]), action_bid, action_ask, 0, 0,
                geometry_price=(order_price if action in PENDING_CHOICES else entry_price),
                risk_denominator_from_fill=args.risk_denominator_from_fill,
            )
            grid[action - 1] = np.asarray(result["all_net_r"], dtype=np.float32)
            filled[action - 1] = True
        ring[index % len(ring)] = grid
        matured = index - lookahead_bars
        if matured >= 0:
            old = ring[matured % len(ring)]
            assert old is not None
            regime = regimes[matured]
            for action in ENTRY_CHOICES:
                for sl in range(sl_count):
                    for tp in range(tp_count):
                        value = float(old[action - 1, sl, tp])
                        update(exact[tuple(int(x) for x in (*regime, action, sl, tp))], value)
                        update(direction[(int(regime[0]), int(regime[1]), action, sl, tp)], value)
                        update(global_stats[(action, sl, tp)], value)
        scores = np.full((6, sl_count, tp_count), -np.inf, dtype=np.float64)
        for action in ENTRY_CHOICES:
            for sl in range(sl_count):
                for tp in range(tp_count):
                    for state in (
                        exact[tuple(int(x) for x in (*regimes[index], action, sl, tp))],
                        direction[(int(regimes[index, 0]), int(regimes[index, 1]), action, sl, tp)],
                        global_stats[(action, sl, tp)],
                    ):
                        value = _lower(state, args.action_lcb_z, args.action_history_min_count)
                        if value is not None:
                            scores[action - 1, sl, tp] = value
                            break
        role_choices = ((0, (1, 3, 5)), (1, (2, 4, 6)))
        for _, actions in role_choices:
            best_action, best_sl, best_tp, best_score = 0, 0, 0, -np.inf
            for action in actions:
                flat = scores[action - 1]
                sl, tp = np.unravel_index(int(np.argmax(flat)), flat.shape)
                score = float(flat[sl, tp])
                if score > best_score:
                    best_action, best_sl, best_tp, best_score = action, int(sl), int(tp), score
            if best_score <= args.action_lcb_min_r:
                continue
            split = str(splits[index])
            if split not in chosen:
                continue
            value = float(grid[best_action - 1, best_sl, best_tp])
            chosen[split].append(value)
            chosen_actions[split][str(Action(best_action).name)] += 1
            chosen_geometry[split][f"sl{SL_ATR_BINS[best_sl]}_rr{TP_RR_BINS[best_tp]}"] += 1

    report: dict[str, object] = {
        "rows": usable,
        "grid": {"sl_atr": list(SL_ATR_BINS), "tp_rr": list(TP_RR_BINS)},
        "fixed_entry_offset": {
            "requested_atr": float(args.fixed_entry_offset_atr),
            "effective_atr": float(offsets[offset_index]),
            "codebook_index": int(offset_index),
        },
        "selection": {"history_min": args.action_history_min_count, "lcb_z": args.action_lcb_z, "min_lcb_r": args.action_lcb_min_r},
        "lifecycle": {
            "holding_horizon_bars": int(args.horizon_bars),
            "holding_horizon_origin": "from_fill" if args.holding_horizon_from_fill else "from_placement",
            "risk_denominator_origin": "fill_to_sl" if args.risk_denominator_from_fill else "requested_at_placement",
            "lookahead_bars": int(lookahead_bars),
            "pending_fill_price_mode": str(args.pending_fill_price_mode),
        },
        "splits": {},
    }
    for name, values in chosen.items():
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) >= 2:
            mean = float(arr.mean())
            lcb = mean - 1.64 * float(arr.std(ddof=1)) / math.sqrt(len(arr))
            win = float((arr > 0.0).mean())
        else:
            mean = lcb = win = None
        report["splits"][name] = {
            "entries": len(arr), "mean_net_r": mean, "lcb_95_one_sided": lcb,
            "win_rate": win, "actions": dict(chosen_actions[name]),
            "geometry": dict(chosen_geometry[name]),
        }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
        args.output.resolve().write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
