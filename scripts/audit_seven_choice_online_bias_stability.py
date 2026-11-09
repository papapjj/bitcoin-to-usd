#!/usr/bin/env python3
"""Audit validation-block stability for top online-bias configurations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.seven_choice_teacher import load_seven_choice_dataset  # noqa: E402
from simulate_seven_choice_online_bias import _metrics, _scores, _simulate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--top", type=int, default=80)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sweep = json.loads(args.sweep.read_text(encoding="utf-8"))
    data = load_seven_choice_dataset(args.dataset)
    values, wins = _scores(args.checkpoint, args.config, data)
    validation_unique = np.flatnonzero(data.agent_ids == 0)
    validation_times = validation_unique[data.splits[validation_unique] == "validation"]
    block_edges = np.linspace(0, len(validation_times), args.blocks + 1, dtype=int)
    audited: list[dict[str, object]] = []
    for row in sweep["rows"][: max(1, args.top)]:
        v = _simulate(
            data, values, wins, "validation", float(row["threshold"]),
            int(row["window"]), float(row["scale"]), float(row["floor"]),
        )
        blocks = []
        times = np.asarray(v["entry_times"], dtype=int)
        results = np.asarray(v["entry_values"], dtype=np.float64)
        for start, stop in zip(block_edges[:-1], block_edges[1:]):
            blocks.append(_metrics(results[(times >= start) & (times < stop)].tolist()))
        item = dict(row)
        train = _simulate(
            data, values, wins, "train", float(row["threshold"]),
            int(row["window"]), float(row["scale"]), float(row["floor"]),
        )
        item["train"] = {key: value for key, value in train.items() if key not in {"selected_actions", "entry_values", "entry_times"}}
        item["validation_blocks"] = blocks
        item["validation_min_block_lcb"] = min(
            float(block["lcb_r"] or -1e9) for block in blocks
        )
        audited.append(item)
    audited.sort(key=lambda item: float(item["validation_min_block_lcb"]), reverse=True)
    report = {
        "selection_uses": "validation blocks only",
        "blocks": int(args.blocks),
        "audited": audited,
        "best_stable": audited[0] if audited else None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "best_stable": report["best_stable"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
