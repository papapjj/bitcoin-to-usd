#!/usr/bin/env python3
"""Causally simulate an online action-direction bias on a saved value head."""
from __future__ import annotations

import argparse
import heapq
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.seven_choice_teacher import load_seven_choice_dataset  # noqa: E402
from sweep_seven_choice_safety_floor import _scores  # noqa: E402


def _metrics(values: list[float]) -> dict[str, float | int | None]:
    a = np.asarray(values, dtype=np.float64)
    n = int(a.size)
    if n == 0:
        return {"entries": 0, "mean_r": None, "lcb_r": None, "win_rate": None}
    mean = float(a.mean())
    lcb = mean - 1.64 * float(a.std(ddof=1)) / np.sqrt(n) if n >= 2 else None
    return {"entries": n, "mean_r": mean, "lcb_r": lcb, "win_rate": float(np.mean(a > 0.0))}


def _simulate(data, values: np.ndarray, wins: np.ndarray, split: str, threshold: float, window: int, scale: float, floor: float) -> dict[str, object]:
    unique = np.flatnonzero(data.agent_ids == 0)
    selected_values = unique
    split_unique = data.splits[unique]
    segment = np.flatnonzero(split_unique == split)
    recent: list[list[float]] = [[] for _ in range(6)]
    sums = np.zeros(6, dtype=np.float64)
    pending: list[list[tuple[int, float]]] = [[] for _ in range(len(segment) + 64)]
    entries: list[float] = []
    entry_times: list[int] = []
    selected_actions: list[int] = []
    for local_t, unique_pos in enumerate(segment):
        for action, result in pending[local_t]:
            bucket = recent[action]
            bucket.append(float(result))
            sums[action] += float(result)
            if len(bucket) > window:
                sums[action] -= bucket.pop(0)
        bias = sums / np.maximum(1, np.asarray([len(bucket) for bucket in recent]))
        raw = np.asarray(values[unique_pos], dtype=np.float64).copy()
        raw[np.asarray(wins[unique_pos]) < float(floor)] = -5.0
        adjusted = raw + float(scale) * bias
        long_scores = adjusted[[0, 2, 4]]
        short_scores = adjusted[[1, 3, 5]]
        for row_offset, scores, actions in ((0, long_scores, (1, 3, 5)), (1, short_scores, (2, 4, 6))):
            best_local = int(np.argmax(scores))
            action = int(actions[best_local])
            if float(scores[best_local]) <= float(threshold):
                action = 0
            selected_actions.append(action)
            if action > 0:
                row = int(unique[unique_pos] + row_offset)
                result = float(data.net_r[row, action])
                entries.append(result)
                entry_times.append(local_t)
                filled = bool(data.filled[row, action])
                delay = int(data.exit_bars[row, action])
                if not filled:
                    delay = 5
                maturity = min(len(segment) - 1, local_t + max(1, delay))
                pending[maturity].append((action - 1, result))
    return _metrics(entries) | {
        "selected_actions": selected_actions,
        "entry_values": entries,
        "entry_times": entry_times,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=-0.197599815706412)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_seven_choice_dataset(args.dataset)
    values, wins = _scores(args.checkpoint, args.config, data)
    rows: list[dict[str, object]] = []
    thresholds = [
        float(args.threshold), -0.50, -0.40, -0.30, -0.25, -0.20, -0.15,
        -0.10, -0.05, 0.0, 0.10, 0.20,
    ]
    for threshold in sorted(set(thresholds)):
        for window in (8, 16, 32, 64, 128, 256):
            for scale in (0.05, 0.10, 0.25, 0.50):
                for floor in (0.0, 1.0 / 3.0, 0.3833333333):
                    validation = _simulate(data, values, wins, "validation", threshold, window, scale, floor)
                    test = _simulate(data, values, wins, "test", threshold, window, scale, floor)
                    rows.append({"threshold": threshold, "window": window, "scale": scale, "floor": floor, "validation": {k: v for k, v in validation.items() if k != "selected_actions"}, "test": {k: v for k, v in test.items() if k != "selected_actions"}})
    rows.sort(key=lambda row: float(row["validation"]["lcb_r"] or -1e9), reverse=True)
    report = {"checkpoint": str(args.checkpoint.resolve()), "threshold": args.threshold, "selection_uses": "validation only", "rows": rows, "best_validation": rows[0] if rows else None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best_validation": report["best_validation"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
