"""Audit realized expectancy by expert and emitted action.

This is intentionally an offline diagnostic.  It does not modify an artifact,
does not update policy state, and does not use the current MT5 result as a
training signal.  Causal ledgers can retain historical reports, so the output
labels the source as retained-ledger evidence and should not be mistaken for
a fresh out-of-sample score.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _stats(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not n:
        return {"n": 0, "sum_r": 0.0, "mean_r": None, "wr": None, "std_r": None, "lcb95": None}
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / max(1, n - 1)
    std = math.sqrt(var)
    return {
        "n": n,
        "sum_r": mean * n,
        "mean_r": mean,
        "wr": sum(1 for x in values if x > 0.0) / n,
        "loss_rate": sum(1 for x in values if x < 0.0) / n,
        "std_r": std,
        "lcb95": mean - 1.96 * std / math.sqrt(n),
    }


def _iter_reports(system: Any):
    experts = getattr(system, "experts", {}) or {}
    for outer_expert, expert in experts.items():
        policy = getattr(expert, "policy", None)
        ledger = getattr(policy, "causal_ledger", None)
        reports = getattr(ledger, "reports", {}) if ledger is not None else {}
        values = reports.values() if hasattr(reports, "values") else reports or ()
        for report in values:
            outcome = getattr(report, "outcome", None)
            snapshot = getattr(report, "snapshot", None)
            if outcome is None or snapshot is None:
                continue
            realized_r = _safe_float(getattr(outcome, "realized_r", None))
            if realized_r is None:
                continue
            details = getattr(snapshot, "action_details", {}) or {}
            metadata = getattr(outcome, "metadata", {}) or {}
            action_index = getattr(snapshot, "selected_action_index", None)
            try:
                action_index = int(action_index)
            except (TypeError, ValueError):
                action_index = -1
            yield {
                "outer_expert": str(outer_expert),
                "intent_context": str(metadata.get("intent_context") or getattr(expert, "ma_context", "")),
                "intent_expert": str(metadata.get("intent_expert") or outer_expert),
                "owner_context": str(metadata.get("position_owner_context") or metadata.get("intent_context") or ""),
                "owner_expert": str(metadata.get("position_owner_expert") or metadata.get("intent_expert") or outer_expert),
                "action_index": action_index,
                "action_name": str(details.get("action_name") or getattr(snapshot, "action_name", "UNKNOWN")),
                "realized_r": realized_r,
                "trace_id": str(getattr(report, "trace_id", "")),
            }


def _ranked(groups: dict[tuple[Any, ...], list[float]], names: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for key, values in groups.items():
        row = dict(zip(names, key))
        row.update(_stats(values))
        rows.append(row)
    return sorted(rows, key=lambda row: (row["mean_r"] if row["mean_r"] is not None else 0.0, -row["n"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.artifact.open("rb") as handle:
        system = pickle.load(handle)

    observations = list(_iter_reports(system))
    by_expert: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    by_expert_action: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    by_owner_action: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for item in observations:
        by_expert[(item["outer_expert"], item["intent_context"], item["intent_expert"])].append(item["realized_r"])
        by_expert_action[(item["outer_expert"], item["intent_context"], item["intent_expert"], item["owner_context"], item["owner_expert"], item["action_index"], item["action_name"])].append(item["realized_r"])
        by_owner_action[(item["owner_context"], item["owner_expert"], item["action_index"], item["action_name"])].append(item["realized_r"])

    result = {
        "artifact": str(args.artifact),
        "evidence_scope": "retained_causal_ledger_reports",
        "fresh_mt5_oos_required": True,
        "observations": len(observations),
        "expert_summary": [
            dict(zip(("outer_expert", "intent_context", "intent_expert"), key), **_stats(values))
            for key, values in sorted(by_expert.items(), key=lambda item: (sum(item[1]) / len(item[1]), -len(item[1])))
        ],
        "expert_action_summary": _ranked(by_expert_action, ("outer_expert", "intent_context", "intent_expert", "owner_context", "owner_expert", "action_index", "action_name")),
        "owner_action_summary": _ranked(by_owner_action, ("owner_context", "owner_expert", "action_index", "action_name")),
    }
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
