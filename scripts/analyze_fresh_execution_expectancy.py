#!/usr/bin/env python3
"""Summarize fresh MT5 lifecycle expectancy by entry expert and action.

Unlike ``analyze_expert_output_expectancy.py``, this tool reads only the
execution ledger produced by one MT5 run.  It intentionally assigns the full
closed lifecycle R to the filled entry intent; broker/system close events are
reported separately as management evidence and are not treated as causal
proof for an exit action.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [_float(row["r"]) for row in rows]
    n = len(values)
    mean = sum(values) / n if n else 0.0
    variance = sum((value - mean) ** 2 for value in values) / max(1, n - 1) if n > 1 else 0.0
    std = math.sqrt(max(0.0, variance))
    wins = sum(value > 0.0 for value in values)
    return {
        "n": n,
        "sum_r": float(sum(values)),
        "mean_r": float(mean),
        "wr": float(wins / n) if n else 0.0,
        "loss_rate": float((n - wins) / n) if n else 0.0,
        "std_r": float(std),
        "lcb95": float(mean - 1.96 * std / math.sqrt(n)) if n else 0.0,
    }


def _group_summary(records: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[tuple(str(record.get(key, "")) for key in keys)].append(record)
    result: list[dict[str, Any]] = []
    for group_key, rows in grouped.items():
        item = {key: value for key, value in zip(keys, group_key)}
        item.update(_summary(rows))
        result.append(item)
    result.sort(key=lambda item: (float(item["mean_r"]), -int(item["n"])))
    return result


def analyze(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    by_position: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        ticket = str(row.get("position_ticket", "") or "").strip()
        if ticket and ticket != "0":
            by_position[ticket].append(row)

    records: list[dict[str, Any]] = []
    management_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for ticket, position_rows in by_position.items():
        entry = next(
            (
                row
                for row in position_rows
                if str(row.get("source", "")) == "policy"
                and str(row.get("transaction_type", "")) == "DEAL_ADD"
                and str(row.get("status", "")) == "FILLED"
                and str(row.get("action_name", "")) != "SYSTEM"
            ),
            None,
        )
        close = next((row for row in reversed(position_rows) if str(row.get("position_final_close", "0")) == "1"), None)
        if entry is None or close is None:
            skipped.append({"position_ticket": ticket, "reason": "missing_entry_or_final_close"})
            continue
        risk = _float(entry.get("initial_risk"))
        if abs(risk) < 1e-12:
            skipped.append({"position_ticket": ticket, "reason": "zero_initial_risk"})
            continue
        realized_r = _float(close.get("profit")) / risk
        records.append(
            {
                "position_ticket": ticket,
                "intent_context": str(entry.get("intent_context", "")),
                "intent_expert": str(entry.get("intent_expert", "")),
                "entry_action": str(entry.get("action_name", "")),
                "close_reason": str(close.get("close_reason", "")),
                "r": float(realized_r),
            }
        )
        management_rows.extend(
            row
            for row in position_rows
            if str(row.get("action_name", "")) in {"CLOSE_BEST", "CLOSE_WORST", "CLOSE_ALL", "MANAGE_POSITION"}
            and str(row.get("source", "")) == "policy"
        )

    return {
        "source": str(path.resolve()),
        "evidence_scope": "fresh_mt5_execution_ledger",
        "causal_warning": (
            "Full lifecycle R is assigned to the filled entry intent. "
            "Exit/manage events are shadow evidence until counterfactual replay."
        ),
        "raw_event_count": len(rows),
        "closed_lifecycle_count": len(records),
        "skipped_position_count": len(skipped),
        "skipped_positions": skipped,
        "entry_expert_action": _group_summary(records, ("intent_expert", "entry_action")),
        "entry_context_expert": _group_summary(records, ("intent_context", "intent_expert")),
        "entry_action": _group_summary(records, ("entry_action",)),
        "management_shadow_event_count": len(management_rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.execution_csv)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
