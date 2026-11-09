#!/usr/bin/env python3
"""Search simple causal feature intervals for stable seven-choice edge."""
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


def _metrics(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    n = int(values.size)
    if n == 0:
        return {"entries": 0, "mean_r": None, "lcb_r": None, "win_rate": None}
    mean = float(values.mean())
    lcb = mean - 1.64 * float(values.std(ddof=1)) / np.sqrt(n) if n >= 2 else None
    return {"entries": n, "mean_r": mean, "lcb_r": lcb, "win_rate": float(np.mean(values > 0.0))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.bins < 2 or args.minimum_validation_entries < 2:
        raise ValueError("bins va minimum entries phai >= 2")
    data = load_seven_choice_dataset(args.dataset)
    unique = np.flatnonzero(data.agent_ids == 0)
    obs = data.observations[unique]
    net = data.net_r[unique]
    splits = data.splits[unique]
    names = ("MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP")
    rows: list[dict[str, object]] = []
    for feature in range(obs.shape[1]):
        train_mask = splits == "train"
        values = obs[train_mask, feature]
        edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, args.bins + 1)))
        if len(edges) < 3:
            continue
        bins = np.clip(np.digitize(obs[:, feature], edges[1:-1], right=False), 0, len(edges) - 2)
        for action_index, action_name in enumerate(names, start=1):
            train_means = np.full(len(edges) - 1, float(net[train_mask, action_index].mean()), dtype=np.float64)
            for bucket in range(len(train_means)):
                selected = train_mask & (bins == bucket)
                if selected.any():
                    train_means[bucket] = float(net[selected, action_index].mean())
            validation_mask = splits == "validation"
            candidates: list[tuple[float, int, dict[str, object], dict[str, object]]] = []
            for bucket in range(len(train_means)):
                vmask = validation_mask & (bins == bucket)
                tmask = (splits == "test") & (bins == bucket)
                vm = _metrics(net[vmask, action_index])
                if int(vm["entries"] or 0) < args.minimum_validation_entries:
                    continue
                tm = _metrics(net[tmask, action_index])
                candidates.append((float(vm["lcb_r"] or -1e9), bucket, vm, tm))
            if not candidates:
                continue
            lcb, bucket, vm, tm = max(candidates, key=lambda item: item[0])
            rows.append({
                "feature": feature,
                "action": action_name,
                "bin": bucket,
                "lower": float(edges[bucket]),
                "upper": float(edges[bucket + 1]),
                "train_predicted_mean_r": float(train_means[bucket]),
                "validation": vm,
                "test": tm,
                "validation_lcb_r": lcb,
            })
    rows.sort(key=lambda item: float(item["validation_lcb_r"]), reverse=True)
    report = {
        "dataset": str(args.dataset.resolve()),
        "feature_count": int(obs.shape[1]),
        "bins": int(args.bins),
        "selection_uses": "train quantile edges and validation lower bound only",
        "top_validation_rules": rows[:100],
        # A test-positive rule whose validation lower bound is negative is a
        # post-hoc finding, not a deployable rule. Keep it visible for audit,
        # but report the sequential validation-then-test gate separately.
        "positive_test_lcb_rules_unfiltered": [
            row for row in rows if float(row["test"]["lcb_r"] or -1e9) > 0.0
        ][:100],
        "positive_validation_and_test_lcb_rules": [
            row for row in rows
            if float(row["validation_lcb_r"] or -1e9) > 0.0
            and float(row["test"]["lcb_r"] or -1e9) > 0.0
        ][:100],
        "counts": {
            "rules": len(rows),
            "positive_validation_lcb": sum(float(row["validation_lcb_r"]) > 0.0 for row in rows),
            "positive_test_lcb_unfiltered": sum(
                float(row["test"]["lcb_r"] or -1e9) > 0.0 for row in rows
            ),
            "positive_validation_and_test_lcb": sum(
                float(row["validation_lcb_r"] or -1e9) > 0.0
                and float(row["test"]["lcb_r"] or -1e9) > 0.0
                for row in rows
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": report["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
