"""Persist live learning evidence while a V13 MT5 run is active."""
from __future__ import annotations

import csv
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / os.environ.get("V13_RUNNER_DIR", "mt5_v2_runner")
OUT = ROOT / os.environ.get(
    "V13_CONSERVATIVE_OUT",
    "outputs/he_chuyen_gia/audit/v13_conservative_matrix",
)
INTERVAL = max(5, int(os.environ.get("V13_WATCH_INTERVAL", "30")))
RUN_ID = str(os.environ.get("V13_WATCH_RUN_ID", "") or "")
LEARNING_EVENTS = ROOT / os.environ.get(
    "V13_LEARNING_EVENTS",
    "outputs/quan_sat_chuyen_gia_live.jsonl",
)
MIN_FILE_MTIME_NS = int(os.environ.get("V13_WATCH_MIN_MTIME_NS", "0") or "0")


def _latest(name: str) -> Path | None:
    paths = list((RUNNER / "Tester").glob(f"Agent-*/MQL5/Files/{name}"))
    if MIN_FILE_MTIME_NS:
        paths = [path for path in paths if path.stat().st_mtime_ns >= MIN_FILE_MTIME_NS]
    return max(paths, key=lambda item: item.stat().st_mtime_ns) if paths else None


def _rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))
    except (OSError, PermissionError):
        return []


def _learning_event_snapshot() -> dict[str, object]:
    """Read append-only policy telemetry, not tester/PKL state.

    The bridge/execution CSVs show what was executed.  These event counters
    show whether the policy actually consumed outcomes and changed its
    learning systems.  A missing file is explicit rather than silently
    reported as zero learning.
    """
    result: dict[str, object] = {
        "path": str(LEARNING_EVENTS),
        "available": bool(LEARNING_EVENTS.is_file()),
        "events": 0,
        "feedback_events": 0,
        "development_events": 0,
        "lifecycle_events": 0,
        "lifecycle_credited": 0,
        "lifecycle_positive": 0,
        "lifecycle_negative": 0,
        "expectancy_update_events": 0,
        "expectancy_record_count": 0,
        "selector_update_events": 0,
        "growth_event_count": 0,
        "last_revision": None,
        "lifecycle_source": "none",
        "expectancy_update_source": "none",
        "expectancy_update_exact": False,
    }
    if not LEARNING_EVENTS.is_file():
        return result
    allowed_run_ids = {RUN_ID} if RUN_ID else set()
    active_path = LEARNING_EVENTS.with_name("active_run.json")
    # Some MT5 paths emit the immutable close outcome as a ``feedback`` event
    # and do not emit the optional observatory ``lifecycle`` envelope.  Keep
    # both counters while streaming, then prefer exact lifecycle events when
    # they exist and otherwise fall back to closed feedback outcomes.
    feedback_closed = 0
    feedback_credited = 0
    feedback_positive = 0
    feedback_negative = 0
    feedback_expectancy_updates = 0
    exact_lifecycle_events = 0
    try:
        if active_path.is_file():
            active = json.loads(active_path.read_text(encoding="utf-8"))
            for key in ("run_id", "root_run_id"):
                value = str(active.get(key, "") or "")
                if value:
                    allowed_run_ids.add(value)
    except (OSError, PermissionError, ValueError):
        pass
    try:
        with LEARNING_EVENTS.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except (TypeError, ValueError):
                    continue
                event_ids = {
                    str(event.get("run_id", "") or ""),
                    str(event.get("root_run_id", "") or ""),
                }
                if allowed_run_ids and not (event_ids & allowed_run_ids):
                    continue
                result["events"] = int(result["events"]) + 1
                event_type = str(event.get("event_type", event.get("event", event.get("type", ""))) or "")
                payload = event.get("payload", event.get("data", {}))
                if not isinstance(payload, dict):
                    payload = event
                if event_type == "feedback":
                    result["feedback_events"] = int(result["feedback_events"]) + 1
                    execution = payload.get("execution")
                    if isinstance(execution, dict) and bool(execution.get("closed", False)):
                        feedback_closed += 1
                        profit = float(execution.get("profit", 0.0) or 0.0)
                        feedback_positive += int(profit > 0.0)
                        feedback_negative += int(profit < 0.0)
                        # The feedback path does not carry the final policy
                        # credit/expectancy report; one close is therefore an
                        # inferred update, not an exact observatory count.
                        feedback_expectancy_updates += 1
                elif event_type == "development":
                    result["development_events"] = int(result["development_events"]) + 1
                elif event_type == "selector_update":
                    # Direct selector events are exact updates emitted by the
                    # policy selector, not an inference from a closed trade.
                    result["selector_update_events"] = int(result["selector_update_events"]) + 1
                elif event_type == "lifecycle":
                    exact_lifecycle_events += 1
                    result["lifecycle_events"] = int(result["lifecycle_events"]) + 1
                    credited = int(payload.get("credited", 0) or 0)
                    result["lifecycle_credited"] = int(result["lifecycle_credited"]) + credited
                    realized = float(payload.get("realized_net_r", 0.0) or 0.0)
                    result["lifecycle_positive"] = int(result["lifecycle_positive"]) + int(realized > 0.0)
                    result["lifecycle_negative"] = int(result["lifecycle_negative"]) + int(realized < 0.0)
                    if "expectancy" in payload:
                        result["expectancy_update_events"] = int(result["expectancy_update_events"]) + 1
                        report = payload.get("expectancy")
                        if isinstance(report, dict):
                            records = report.get("record_count", report.get("records", 0))
                            try:
                                result["expectancy_record_count"] = max(int(result["expectancy_record_count"]), int(records or 0))
                            except (TypeError, ValueError):
                                pass
                    causal = payload.get("causal")
                    if isinstance(causal, dict) and any("selector" in str(key).lower() for key in causal):
                        result["selector_update_events"] = int(result["selector_update_events"]) + 1
                growth = payload.get("growth_events", [])
                if isinstance(growth, list):
                    result["growth_event_count"] = int(result["growth_event_count"]) + len(growth)
                if "revision" in payload:
                    result["last_revision"] = payload.get("revision")
    except (OSError, PermissionError):
        result["read_error"] = True
    if exact_lifecycle_events:
        result["lifecycle_source"] = "observatory_lifecycle"
        result["expectancy_update_source"] = "observatory_lifecycle_report"
        result["expectancy_update_exact"] = True
    elif feedback_closed:
        result["lifecycle_events"] = feedback_closed
        result["lifecycle_credited"] = feedback_credited
        result["lifecycle_positive"] = feedback_positive
        result["lifecycle_negative"] = feedback_negative
        result["expectancy_update_events"] = feedback_expectancy_updates
        result["lifecycle_source"] = "feedback_execution_closed_inferred"
        result["expectancy_update_source"] = "feedback_execution_closed_inferred"
        result["expectancy_update_exact"] = False
    return result


def _snapshot(previous: dict[str, object] | None = None) -> dict[str, object]:
    bridge_rows = _rows(_latest("self_growing_bridge.csv"))
    execution_rows = _rows(_latest("self_growing_execution.csv"))
    experts = Counter(
        str(row.get("trace_id", "")).split("~", 1)[0]
        for row in bridge_rows
        if row.get("trace_id")
    )
    closed = [
        row for row in execution_rows
        if str(row.get("position_final_close", "")).strip().lower() in {"1", "true", "yes"}
    ]
    profit = 0.0
    positive = 0
    negative = 0
    net_r: list[float] = []
    for row in closed:
        try:
            value = float(row.get("profit", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        profit += value
        positive += int(value > 0.0)
        negative += int(value < 0.0)
        try:
            risk = float(row.get("initial_risk", 0.0) or 0.0)
            if risk:
                net_r.append(value / risk)
        except (TypeError, ValueError):
            pass
    total_rows = max(1, len(bridge_rows))
    previous_peak = float((previous or {}).get("closed_profit_peak", profit) or profit)
    profit_peak = max(previous_peak, profit)
    selection_rates = {key: value / total_rows for key, value in experts.items()}
    previous_rates = dict((previous or {}).get("expert_selection_rates", {}) or {})
    selection_rate_delta = {
        key: float(selection_rates.get(key, 0.0) - float(previous_rates.get(key, 0.0)))
        for key in sorted(set(selection_rates) | set(previous_rates))
    }
    closed_by_expert: dict[str, dict[str, float | int]] = {}
    for row in closed:
        expert = str(row.get("position_owner_expert", "") or "UNKNOWN")
        try:
            value = float(row.get("profit", 0.0) or 0.0)
        except (TypeError, ValueError):
            value = 0.0
        item = closed_by_expert.setdefault(expert, {"closed": 0, "positive": 0, "negative": 0, "profit": 0.0})
        item["closed"] = int(item["closed"]) + 1
        item["positive"] = int(item["positive"]) + int(value > 0.0)
        item["negative"] = int(item["negative"]) + int(value < 0.0)
        item["profit"] = float(item["profit"]) + value
    return {
        "run_id": RUN_ID,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_rows": len(bridge_rows),
        "last_sim_time": str(bridge_rows[-1].get("time", "")) if bridge_rows else "",
        "execution_events": len(execution_rows),
        "lifecycle_closed": len(closed),
        "closed_profit_sum": profit,
        "closed_profit_peak": profit_peak,
        "closed_drawdown_from_peak": profit_peak - profit,
        "closed_edge_status": "positive" if profit > 0.0 else "non_positive",
        "closed_positive": positive,
        "closed_negative": negative,
        "closed_mean_r": (sum(net_r) / len(net_r)) if net_r else None,
        "closed_profit_factor": (sum(value for value in net_r if value > 0.0) / abs(sum(value for value in net_r if value < 0.0))) if any(value < 0.0 for value in net_r) else None,
        "expert_selection_counts": dict(experts),
        "expert_selection_rates": selection_rates,
        "selection_rate_delta_since_previous": selection_rate_delta,
        "closed_by_expert": closed_by_expert,
        "action_counts": dict(Counter(str(row.get("action", "")) for row in bridge_rows)),
        "learning": _learning_event_snapshot(),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / "live_learning.jsonl"
    previous: dict[str, object] | None = None
    while True:
        snapshot = _snapshot(previous)
        with log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, ensure_ascii=False) + "\n")
        previous = snapshot
        if (OUT / "summary.json").is_file():
            return
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
