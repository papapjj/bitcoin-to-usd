#!/usr/bin/env python3
"""Search strict two-feature causal filters for a seven-choice action."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ACTION_NAMES = {
    1: "MARKET_BUY",
    2: "MARKET_SELL",
    3: "BUY_LIMIT",
    4: "SELL_LIMIT",
    5: "BUY_STOP",
    6: "SELL_STOP",
}
DEFAULT_FEATURES = (10, 14, 15, 21, 25, 26, 27, 28, 77, 85, 86, 88, 90, 97, 104, 105, 121, 124, 127, 132)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _day_lcb(day_sum: np.ndarray, day_count: np.ndarray) -> np.ndarray:
    """Return one lower bound per cell, including inactive days as zero."""
    means = np.divide(
        day_sum,
        day_count,
        out=np.zeros_like(day_sum, dtype=np.float64),
        where=day_count > 0,
    )
    day_count_total = means.shape[0]
    mean = means.mean(axis=0)
    if day_count_total < 2:
        return mean
    std = means.std(axis=0, ddof=1)
    return mean - 1.64 * std / math.sqrt(float(day_count_total))


def _aggregate(
    code: np.ndarray,
    values: np.ndarray,
    filled: np.ndarray,
    times: np.ndarray,
    mask: np.ndarray,
    cells: int,
) -> dict[str, np.ndarray]:
    selected = np.asarray(mask, dtype=np.bool_)
    code_selected = code[selected]
    value_selected = np.asarray(values[selected], dtype=np.float64)
    filled_selected = np.asarray(filled[selected], dtype=np.bool_)
    days = np.array([str(item)[:10] for item in times[selected]], dtype="U10")
    all_days = np.unique(np.array([str(item)[:10] for item in times[mask]], dtype="U10"))
    day_index = {day: index for index, day in enumerate(all_days.tolist())}
    day_ids = np.asarray([day_index[day] for day in days.tolist()], dtype=np.int64)
    day_sum = np.zeros((len(all_days), cells), dtype=np.float64)
    day_count = np.zeros((len(all_days), cells), dtype=np.int64)
    if len(code_selected):
        np.add.at(day_sum, (day_ids, code_selected), value_selected)
        np.add.at(day_count, (day_ids, code_selected), 1)
    count = day_count.sum(axis=0)
    total = day_sum.sum(axis=0)
    filled_day_sum = np.zeros((len(all_days), cells), dtype=np.float64)
    filled_day_count = np.zeros((len(all_days), cells), dtype=np.int64)
    if filled_selected.any():
        np.add.at(
            filled_day_sum,
            (day_ids[filled_selected], code_selected[filled_selected]),
            value_selected[filled_selected],
        )
        np.add.at(
            filled_day_count,
            (day_ids[filled_selected], code_selected[filled_selected]),
            1,
        )
    filled_count = filled_day_count.sum(axis=0)
    filled_total = filled_day_sum.sum(axis=0)
    return {
        "count": count,
        "sum": total,
        "mean": np.divide(total, count, out=np.zeros(cells), where=count > 0),
        "lcb": _day_lcb(day_sum, day_count),
        "filled_count": filled_count,
        "filled_mean": np.divide(
            filled_total,
            filled_count,
            out=np.zeros(cells),
            where=filled_count > 0,
        ),
        "filled_lcb": _day_lcb(filled_day_sum, filled_day_count),
    }


def _cell_metrics(agg: dict[str, np.ndarray], cell: int) -> dict[str, float | int | None]:
    return {
        "n_attempts": int(agg["count"][cell]),
        "attempt_mean_r": float(agg["mean"][cell]),
        "attempt_day_lcb95_r": float(agg["lcb"][cell]),
        "filled": int(agg["filled_count"][cell]),
        "filled_mean_r": float(agg["filled_mean"][cell]) if agg["filled_count"][cell] else None,
        "filled_day_lcb95_r": float(agg["filled_lcb"][cell]) if agg["filled_count"][cell] else None,
    }


def scan(dataset: Path, features: tuple[int, ...], bins: int, minimum_support: int) -> dict[str, object]:
    with np.load(dataset, allow_pickle=True) as raw:
        unique = np.flatnonzero(raw["agent_ids"] == 0)
        observations = np.asarray(raw["observations"][unique], dtype=np.float64)
        agent_ids = np.asarray(raw["agent_ids"][unique], dtype=np.int64)
        sample_times = raw["sample_times"][unique]
        splits = raw["splits"][unique]
        net = np.asarray(raw["counterfactual_net_r"][unique], dtype=np.float64)
        filled = np.asarray(raw["counterfactual_filled"][unique], dtype=np.bool_)
    if any(feature < 0 or feature >= observations.shape[1] for feature in features):
        raise ValueError("feature nam ngoai 137 cot")
    train = splits == "train"
    validation = splits == "validation"
    test = splits == "test"
    bin_values: dict[int, np.ndarray] = {}
    edges: dict[int, np.ndarray] = {}
    for feature in features:
        edge = np.unique(np.quantile(observations[train, feature], np.linspace(0.0, 1.0, bins + 1)))
        if len(edge) < 3:
            continue
        edges[feature] = edge
        bin_values[feature] = np.clip(
            np.digitize(observations[:, feature], edge[1:-1], right=False),
            0,
            len(edge) - 2,
        )
    candidates: list[dict[str, object]] = []
    for feature1_index, feature1 in enumerate(sorted(edges)):
        for feature2 in sorted(edges)[feature1_index + 1:]:
            code = bin_values[feature1] * bins + bin_values[feature2]
            cells = bins * bins
            for action, action_name in ACTION_NAMES.items():
                # The deduplicated row carries all six counterfactual values;
                # role legality is applied only when the policy chooses an
                # action, not when measuring that action's potential edge.
                legal = np.ones(len(observations), dtype=np.bool_)
                train_agg = _aggregate(code, net[:, action], filled[:, action], sample_times, train & legal, cells)
                validation_agg = _aggregate(code, net[:, action], filled[:, action], sample_times, validation & legal, cells)
                test_agg = _aggregate(code, net[:, action], filled[:, action], sample_times, test & legal, cells)
                eligible = (
                    (train_agg["count"] >= minimum_support)
                    & (validation_agg["count"] >= minimum_support)
                    & (train_agg["filled_count"] >= minimum_support)
                    & (validation_agg["filled_count"] >= minimum_support)
                    & (train_agg["mean"] >= 0.0)
                    & (train_agg["lcb"] >= 0.0)
                    & (validation_agg["mean"] >= 0.0)
                    & (validation_agg["lcb"] >= 0.0)
                    & (train_agg["filled_mean"] >= 0.0)
                    & (train_agg["filled_lcb"] >= 0.0)
                    & (validation_agg["filled_mean"] >= 0.0)
                    & (validation_agg["filled_lcb"] >= 0.0)
                )
                cells_ok = np.flatnonzero(eligible)
                if not len(cells_ok):
                    continue
                cell = max(
                    cells_ok.tolist(),
                    key=lambda item: (
                        float(validation_agg["filled_lcb"][item]),
                        float(validation_agg["filled_mean"][item]),
                        float(validation_agg["lcb"][item]),
                    ),
                )
                candidates.append({
                    "feature1": int(feature1),
                    "feature2": int(feature2),
                    "action_index": int(action),
                    "action": action_name,
                    "bin1": int(cell // bins),
                    "bin2": int(cell % bins),
                    "lower1": float(edges[feature1][cell // bins]),
                    "upper1": float(edges[feature1][cell // bins + 1]),
                    "lower2": float(edges[feature2][cell % bins]),
                    "upper2": float(edges[feature2][cell % bins + 1]),
                    "train": _cell_metrics(train_agg, cell),
                    "validation": _cell_metrics(validation_agg, cell),
                    "test": _cell_metrics(test_agg, cell),
                })
    candidates.sort(
        key=lambda row: (
            float(row["validation"]["filled_day_lcb95_r"]),
            float(row["validation"]["filled_mean_r"]),
        ),
        reverse=True,
    )
    return {
        "dataset": str(dataset.resolve()),
        "dataset_sha256": _sha256(dataset),
        "rows": int(len(observations)),
        "features": [int(item) for item in features],
        "bins": int(bins),
        "minimum_support_each_split_metric": int(minimum_support),
        "candidates": candidates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--features", nargs="*", type=int, default=list(DEFAULT_FEATURES))
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--minimum-support", type=int, default=256)
    args = parser.parse_args()
    if args.bins < 2 or args.minimum_support < 2:
        raise ValueError("bins va minimum-support phai >= 2")
    result = scan(args.dataset.resolve(), tuple(dict.fromkeys(args.features)), args.bins, args.minimum_support)
    report = {
        "schema_version": 1,
        "scope": "research_diagnostic_only",
        "method": {
            "selection": "train va validation phai dat trung binh, can duoi theo ngay, va trung binh lenh da khop khong am; xep theo can duoi da khop o validation",
            "daily_lcb": "tat ca ngay cua split deu duoc tinh, ngay khong co mau gan 0",
            "deduplicate": "agent_id=0",
            "test_read_after_selection": True,
        },
        "result": result,
        "decision": {
            "candidate_count": len(result["candidates"]),
            "status": "research_only_not_deployable",
            "reason": "dau ra nay chi la phep loc dau vao; can phat lai tu sinh va MT5 cung hop dong truoc khi dung",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "candidate_count": len(result["candidates"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
