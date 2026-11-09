#!/usr/bin/env python3
"""Compare causal entry score formulas using a saved seven-choice value head."""
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
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_seven_choice_dataset(args.dataset)
    report: dict[str, object] = {"selection_uses": "validation only", "models": {}}
    for checkpoint in args.checkpoint:
        values, wins = _scores(checkpoint, args.config, data)
        values = np.repeat(values[:, None, :], 2, axis=1).reshape(-1, 6)
        wins = np.repeat(wins[:, None, :], 2, axis=1).reshape(-1, 6)
        formulas: dict[str, np.ndarray] = {
            "value": values,
            "win_probability": wins,
            "value_times_win": values * wins,
            "fixed_rr_expected": 3.0 * wins - 1.0,
            "value_plus_win_margin": values + 2.0 * (wins - 1.0 / 3.0),
        }
        rows: list[dict[str, object]] = []
        for formula, score_values in formulas.items():
            scores = np.zeros((len(data.observations), 7), dtype=np.float32)
            scores[:, 1:7] = score_values
            mask = data.splits == "validation"
            best = np.max(np.where(data.allowed_action_mask[mask, 1:7], scores[mask, 1:7], -np.inf), axis=1)
            thresholds = np.unique(np.concatenate((
                np.quantile(best, np.linspace(0.05, 0.995, 191)),
                np.asarray((-1.0, -0.5, -0.25, 0.0, 0.10, 0.20, 0.30, 1.0), dtype=np.float32),
            )))
            candidates: list[tuple[float, float, dict[str, object]]] = []
            for threshold in thresholds:
                _, metrics = _entry_selection_metrics(data, scores, mask, float(threshold))
                if int(metrics["entries"] or 0) < args.minimum_validation_entries:
                    continue
                candidates.append((float(metrics["lcb_95_one_sided"] or -1e9), float(threshold), metrics))
            if not candidates:
                continue
            _, threshold, validation = max(candidates, key=lambda item: item[0])
            _, test = _entry_selection_metrics(data, scores, data.splits == "test", threshold)
            rows.append({"formula": formula, "threshold": threshold, "validation": validation, "test": test})
        rows.sort(key=lambda row: float(row["validation"]["lcb_95_one_sided"] or -1e9), reverse=True)
        report["models"][str(checkpoint)] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "models": list(report["models"].keys())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
