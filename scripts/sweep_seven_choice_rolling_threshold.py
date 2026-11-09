#!/usr/bin/env python3
"""Choose a causal entry threshold robustly across validation time blocks."""
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

from self_growing_trader.seven_choice_teacher import (  # noqa: E402
    _entry_selection_metrics,
    load_seven_choice_dataset,
)
from sweep_seven_choice_safety_floor import _scores  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--minimum-block-entries", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_seven_choice_dataset(args.dataset)
    report: dict[str, object] = {"selection_uses": "validation blocks only", "models": {}}
    for checkpoint in args.checkpoint:
        predicted, predicted_win = _scores(checkpoint, args.config, data)
        values = np.repeat(predicted[:, None, :], 2, axis=1).reshape(-1, 6)
        wins = np.repeat(predicted_win[:, None, :], 2, axis=1).reshape(-1, 6)
        rows: list[dict[str, object]] = []
        for floor in (0.0, 1.0 / 3.0, 0.3833333333):
            scores = np.zeros((len(data.observations), 7), dtype=np.float32)
            scores[:, 1:7] = np.where(wins >= floor, values, -5.0)
            validation = np.flatnonzero(data.splits == "validation")
            best = np.max(
                np.where(data.allowed_action_mask[validation, 1:7], scores[validation, 1:7], -np.inf),
                axis=1,
            )
            thresholds = np.unique(np.concatenate((
                np.quantile(best, np.linspace(0.05, 0.995, 191)),
                np.asarray((0.0, 0.10, 0.20, 0.30, 0.40), dtype=np.float32),
            )))
            blocks = [part for part in np.array_split(validation, args.blocks) if len(part)]
            candidates: list[tuple[float, float, list[dict[str, object]]]] = []
            for threshold in thresholds:
                metrics = []
                valid = True
                for part in blocks:
                    _, item = _entry_selection_metrics(data, scores, np.isin(np.arange(len(data.observations)), part), float(threshold))
                    if int(item["entries"] or 0) < args.minimum_block_entries:
                        valid = False
                        break
                    metrics.append(item)
                if not valid:
                    continue
                min_lcb = min(float(item["lcb_95_one_sided"] or -1e9) for item in metrics)
                candidates.append((min_lcb, float(threshold), metrics))
            if not candidates:
                continue
            min_lcb, threshold, block_metrics = max(candidates, key=lambda item: item[0])
            _, test = _entry_selection_metrics(data, scores, data.splits == "test", threshold)
            rows.append({"floor": floor, "threshold": threshold, "validation_min_lcb": min_lcb, "validation_blocks": block_metrics, "test": test})
        report["models"][str(checkpoint)] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": list(report["models"].keys())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
