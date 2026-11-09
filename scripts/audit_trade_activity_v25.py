#!/usr/bin/env python3
"""Audit the newest V25 continuous-session trade funnel.

This script is read-only. It helps distinguish policy HOLD, risk/memory overlay,
legality/capacity, pending orders and EA fill failures before changing config.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                events.append(item)
    return events


def _latest_payload(events: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for item in reversed(events):
        if str(item.get("event_type", "")) == name:
            payload = item.get("payload", {})
            return dict(payload) if isinstance(payload, dict) else {}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", type=Path, help="events.jsonl path; defaults to the most recently modified V25 session")
    parser.add_argument("--root", type=Path, default=Path("outputs/continuous"), help="continuous session root")
    args = parser.parse_args()

    if args.session:
        path = args.session
    else:
        candidates = list(args.root.glob("**/events.jsonl"))
        if not candidates:
            raise SystemExit(f"No events.jsonl found under {args.root}")
        path = max(candidates, key=lambda item: item.stat().st_mtime)
    if not path.exists():
        raise SystemExit(f"Session event file does not exist: {path}")

    events = _load_events(path)
    print(f"SESSION: {path}")
    print(f"EVENTS: {len(events)}")

    counts = Counter(str(item.get("event_type", "")) for item in events)
    funnel_payload = _latest_payload(events, "continuous_trade_funnel")
    funnel = funnel_payload.get("funnel", funnel_payload)
    if not isinstance(funnel, dict):
        funnel = {}

    blockers = Counter()
    actions = Counter()
    gate_stats: list[float] = []
    memory_stats: list[float] = []
    migration = _latest_payload(events, "continuous_session_started").get("trade_activity_migration", {})
    for item in events:
        if str(item.get("event_type", "")) != "continuous_trade_decision":
            continue
        payload = item.get("payload", {})
        if not isinstance(payload, dict):
            continue
        blockers[str(payload.get("root_blocker", "none"))] += 1
        actions[f"policy={payload.get('policy_action', '?')} / legal={payload.get('legal_action', '?')} / emitted={payload.get('emitted_action', '?')}"] += 1
        gate = payload.get("risk_gate", {})
        if isinstance(gate, dict):
            try:
                gate_stats.append(float(gate.get("action_dampen", 0.0) or 0.0))
                memory_stats.append(float(gate.get("memory_hold_bias", 0.0) or 0.0))
            except (TypeError, ValueError):
                pass

    print("\n=== V25 MIGRATION ===")
    print(json.dumps(migration, ensure_ascii=False, indent=2, default=str) if migration else "No V25 migration event found.")
    print("\n=== FUNNEL ===")
    for key in (
        "decisions", "policy_holds", "policy_entry_intents", "legal_entry_intents",
        "emitted_entry_intents", "acknowledged", "filled", "rejected", "final_closes",
        "masked_entries", "blocked_capacity", "blocked_pending", "pending_recovery_cancels",
        "fill_from_emitted_entry_ratio", "legal_from_policy_entry_ratio",
        "emitted_from_policy_entry_ratio", "sequences_since_last_emitted_entry",
    ):
        if key in funnel:
            print(f"{key}: {funnel[key]}")

    print("\n=== ROOT BLOCKER ===")
    for name, value in blockers.most_common():
        print(f"{name}: {value}")
    print("\n=== MOST COMMON ACTION PATHS ===")
    for name, value in actions.most_common(12):
        print(f"{name}: {value}")

    print("\n=== V25 INTEGRITY SIGNALS ===")
    for name in (
        "continuous_final_risk_gate_credit_v25",
        "continuous_final_risk_gate_credit_error_v25",
        "continuous_legacy_rejection_feedback_blocked_v25",
        "continuous_inhibition_created",
        "continuous_inhibition_deferred_action_mapping",
        "continuous_timing_replacement_required",
        "continuous_trade_activity_warning",
    ):
        print(f"{name}: {counts.get(name, 0)}")
    if gate_stats:
        print(f"mean_action_dampen: {sum(gate_stats) / len(gate_stats):.5f}")
        print(f"max_action_dampen: {max(gate_stats):.5f}")
    if memory_stats:
        print(f"mean_memory_hold_bias: {sum(memory_stats) / len(memory_stats):.5f}")
        print(f"max_memory_hold_bias: {max(memory_stats):.5f}")

    print("\n=== INTERPRETATION ===")
    if counts.get("continuous_final_risk_gate_credit_error_v25", 0):
        print("- Final risk-gate credit errors exist: fix bridge trace/lifecycle data before interpreting trade frequency.")
    if blockers.get("selector_hold", 0) > max(1, sum(blockers.values()) // 2):
        print("- Core selector is choosing HOLD; this is not a legality/fill bottleneck.")
    if blockers.get("risk_gate_global_or_contextual", 0):
        print("- Risk gate is still materially suppressing entries; inspect hard-pressure fields and migration result.")
    if blockers.get("pending_order", 0) or blockers.get("pending_order_live", 0):
        print("- Pending orders are consuming entry opportunities; inspect offset/fill data and cancel events.")
    if float(funnel.get("emitted_entry_intents", 0) or 0) > 0 and float(funnel.get("fill_from_emitted_entry_ratio", 0) or 0) < 0.20:
        print("- EA/execution fill path is weak; do not solve it by increasing exploration or min_daily_trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
