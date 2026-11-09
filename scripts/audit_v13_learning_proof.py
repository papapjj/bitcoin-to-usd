"""Audit evidence that a V13 continuation learned, not merely profited."""
from __future__ import annotations

import csv
import json
import os
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from self_growing_trader.learning_integrity import sha256_file


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / os.environ.get(
    "V13_SEED_SYSTEM",
    "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl",
)
CANDIDATE = ROOT / os.environ.get(
    "V13_CONSERVATIVE_CANDIDATE",
    "outputs/he_chuyen_gia/candidates/v13_conservative_candidate.pkl",
)
AUDIT = ROOT / os.environ.get(
    "V13_CONSERVATIVE_OUT",
    "outputs/he_chuyen_gia/audit/v13_conservative_matrix",
)


def _expectancy_counts(policy: Any) -> dict[str, int]:
    records = 0
    observations = 0
    policy_observations = 0
    for expert in getattr(getattr(policy, "registry", None), "chuyen_gia_theo_context", {}).values():
        owner = getattr(expert, "policy", None)
        ledger = getattr(getattr(owner, "memory_supernet", None), "contextual_expectancy", None)
        for record in dict(getattr(ledger, "records", {}) or {}).values():
            records += 1
            values = list(getattr(record, "values", ()) or ())
            origins = list(getattr(record, "origins", ()) or ())
            observations += len(values)
            policy_observations += sum(1 for origin in origins if str(origin).lower() in {"policy", "policy_exploration"})
    return {
        "records": int(records),
        "observations": int(observations),
        "policy_observations": int(policy_observations),
    }


def _snapshot(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        system = pickle.load(handle)
    experts = {}
    total_lifecycle = 0
    total_decisions = 0
    revisions = 0
    for context, expert in getattr(getattr(system, "registry", None), "chuyen_gia_theo_context", {}).items():
        owner = getattr(expert, "policy", None)
        memory = getattr(owner, "memory_supernet", None)
        item = {
            "expert": str(getattr(expert, "ma_chuyen_gia", "")),
            "context": str(context),
            "lifecycle_closed": int(getattr(expert, "so_lifecycle_da_dong", 0) or 0),
            "decisions": int(getattr(expert, "so_quyet_dinh", 0) or 0),
            "feedback_transitions": int(getattr(expert, "so_feedback_transition", 0) or 0),
            "policy_revision": int(getattr(owner, "revision", 0) or 0),
            "memory_neurons": len(getattr(memory, "neural_cells", {}) or {}),
            "memory_synapses": len(getattr(memory, "neural_synapses", []) or []),
            "contextual_expectancy_records": len(getattr(getattr(memory, "contextual_expectancy", None), "records", {}) or {}),
        }
        experts[str(context)] = item
        total_lifecycle += item["lifecycle_closed"]
        total_decisions += item["decisions"]
        revisions += item["policy_revision"]
    selector = getattr(system, "policy_selector", None)
    selector_counts = dict(getattr(selector, "outcome_count", {}) or {})
    selector_losses = dict(getattr(selector, "loss_streak", {}) or {})
    group_counts = dict(getattr(selector, "group_outcome_count", {}) or {})
    group_losses = dict(getattr(selector, "group_loss_streak", {}) or {})
    return {
        "artifact": str(path),
        "sha256": sha256_file(path),
        "experts": experts,
        "expert_count": len(experts),
        "total_lifecycle_closed": total_lifecycle,
        "total_decisions": total_decisions,
        "total_policy_revisions": revisions,
        "expectancy": _expectancy_counts(system),
        "selector": {
            "leaf_outcome_updates": int(sum(int(value) for value in selector_counts.values())),
            "leaf_keys": len(selector_counts),
            "leaf_loss_streak_sum": int(sum(int(value) for value in selector_losses.values())),
            "group_outcome_updates": int(sum(int(value) for value in group_counts.values())),
            "group_keys": len(group_counts),
            "group_loss_streak_sum": int(sum(int(value) for value in group_losses.values())),
            "lossy_leaf_keys": int(sum(1 for value in selector_losses.values() if int(value) > 0)),
            "lossy_group_keys": int(sum(1 for value in group_losses.values() if int(value) > 0)),
        },
    }


def _decision_file(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _selection(rows: list[dict[str, str]]) -> dict[str, Any]:
    experts = Counter()
    actions = []
    for row in rows:
        trace = str(row.get("trace_id", ""))
        expert = trace.split("~", 1)[0] if trace else ""
        if expert:
            experts[expert] += 1
        actions.append(str(row.get("action", "")))
    total = max(1, len(rows))
    return {
        "rows": len(rows),
        "expert_counts": dict(experts),
        "expert_selection_rate": {key: value / total for key, value in experts.items()},
        "action_counts": dict(Counter(actions)),
    }


def _selection_timeline(rows: list[dict[str, str]], buckets: int = 4) -> dict[str, Any]:
    """Show whether expert selection actually changed during one replay."""
    if not rows:
        return {"buckets": [], "rate_delta_first_to_last": {}, "changed": False}
    buckets = max(2, min(int(buckets), len(rows)))
    chunks: list[dict[str, Any]] = []
    for index in range(buckets):
        start = (len(rows) * index) // buckets
        end = (len(rows) * (index + 1)) // buckets
        view = _selection(rows[start:end])
        view["bucket"] = index + 1
        view["start_row"] = start
        view["end_row"] = end
        chunks.append(view)
    first = chunks[0]["expert_selection_rate"]
    last = chunks[-1]["expert_selection_rate"]
    keys = sorted(set(first) | set(last))
    delta = {key: float(last.get(key, 0.0) - first.get(key, 0.0)) for key in keys}
    return {
        "buckets": chunks,
        "rate_delta_first_to_last": delta,
        "changed": bool(any(abs(value) > 1e-12 for value in delta.values())),
    }


def _closed_execution_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if str(row.get("position_final_close", "0")).lower() in {"1", "true", "yes"}]


def _loss_followup(bridge_rows: list[dict[str, str]], execution_path: Path | None, new_experts: set[str]) -> dict[str, Any]:
    """Check whether a newly introduced expert was selected less after a loss."""
    if not new_experts:
        return {"status": "not_applicable_no_new_experts", "experts": {}}
    closed = _closed_execution_rows(execution_path) if execution_path else []
    result: dict[str, Any] = {}
    for expert in sorted(new_experts):
        owned = [row for row in closed if str(row.get("position_owner_expert", "")) == expert]
        first_loss = next((row for row in owned if float(row.get("profit", 0.0) or 0.0) < 0.0), None)
        if first_loss is None:
            result[expert] = {"status": "no_observed_loss", "closed_count": len(owned)}
            continue
        loss_time = str(first_loss.get("logged_at", ""))
        before = [row for row in bridge_rows if str(row.get("time", "")) < loss_time]
        after = [row for row in bridge_rows if str(row.get("time", "")) >= loss_time]
        before_count = sum(1 for row in before if str(row.get("trace_id", "")).split("~", 1)[0] == expert)
        after_count = sum(1 for row in after if str(row.get("trace_id", "")).split("~", 1)[0] == expert)
        result[expert] = {
            "status": "loss_observed",
            "loss_time": loss_time,
            "closed_count": len(owned),
            "before_loss_rows": len(before),
            "after_loss_rows": len(after),
            "selection_before_loss": before_count,
            "selection_after_loss": after_count,
            "selection_rate_before_loss": before_count / max(1, len(before)),
            "selection_rate_after_loss": after_count / max(1, len(after)),
            "reduced_after_loss": (after_count / max(1, len(after))) < (before_count / max(1, len(before))),
        }
    observed = [item for item in result.values() if item.get("status") == "loss_observed"]
    return {
        "status": "checked",
        "experts": result,
        "all_reduced_after_loss": bool(observed) and all(bool(item["reduced_after_loss"]) for item in observed),
    }


def _metric_number(metrics: dict[str, Any], key: str, *fallbacks: tuple[str, str]) -> float | None:
    value: Any = metrics.get(key)
    for parent, child in fallbacks:
        if value is None:
            value = (metrics.get(parent) or {}).get(child)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _performance_comparison() -> dict[str, Any]:
    """Load paired MT5 performance metrics without treating them as learning proof."""
    pairs: list[dict[str, Any]] = []
    summary_path = AUDIT / "summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Normalize research walk-forward and the authoritative champion matrix.
        for row in summary.get("windows", []):
            pairs.append({"window": row.get("window"), "candidate": row.get("candidate_metrics", {}), "baseline": row.get("baseline_metrics", {})})
        for row in summary.get("evaluations", []):
            pairs.append({
                "window": row.get("validation_window"),
                "candidate": dict(dict(row.get("candidate", {}) or {}).get("metrics", {}) or {}),
                "baseline": dict(dict(row.get("champion", {}) or {}).get("metrics", {}) or {}),
            })
    ab_path = AUDIT / "ab_summary.json"
    if ab_path.is_file() and not pairs:
        summary = json.loads(ab_path.read_text(encoding="utf-8"))
        for label in ("baseline", "candidate"):
            if label in summary:
                pairs.append({"window": summary.get("oos_window", {}), "candidate": summary.get("candidate", {}).get("metrics", {}) if label == "candidate" else {}, "baseline": summary.get("baseline", {}).get("metrics", {}) if label == "baseline" else {}})
        if len(pairs) == 2:
            candidate = pairs[1]["candidate"]
            baseline = pairs[0]["baseline"]
            pairs = [{"window": summary.get("oos_window", {}), "candidate": candidate, "baseline": baseline}]
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        converted: dict[str, Any] = {"window": pair.get("window")}
        for label in ("baseline", "candidate"):
            metrics = pair.get(label) or {}
            converted[label] = {
                "net_profit": _metric_number(metrics, "net_profit"),
                "mean_r": _metric_number(metrics, "mean_r", ("execution_summary", "expectancy")),
                "expectancy": _metric_number(metrics, "expectancy", ("execution_summary", "expectancy")),
                "expectancy_lcb": _metric_number(metrics, "expectancy_lcb", ("execution_summary", "expectancy_lcb")),
                "profit_factor": _metric_number(metrics, "profit_factor", ("execution_summary", "profit_factor_shrunk")),
            }
        c, b = converted["candidate"], converted["baseline"]
        converted["delta_candidate_minus_baseline"] = {key: (None if c[key] is None or b[key] is None else c[key] - b[key]) for key in c}
        rows.append(converted)
    return {"pairs": rows, "pair_count": len(rows)}


def _live_learning_report() -> dict[str, Any]:
    path = AUDIT / "live_learning.jsonl"
    if not path.is_file():
        return {"available": False, "snapshot_count": 0}
    snapshots: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            snapshots.append(value)
    run_ids = [str(item.get("run_id", "")) for item in snapshots if item.get("run_id")]
    if run_ids:
        current_run_id = run_ids[-1]
        snapshots = [item for item in snapshots if str(item.get("run_id", "")) == current_run_id]
    if not snapshots:
        return {"available": False, "snapshot_count": 0}
    first, last = snapshots[0], snapshots[-1]
    first_learning = dict(first.get("learning", {}) or {})
    last_learning = dict(last.get("learning", {}) or {})
    first_rates = dict(first.get("expert_selection_rates", {}) or {})
    last_rates = dict(last.get("expert_selection_rates", {}) or {})
    if not first_rates and first.get("expert_selection_counts"):
        total = max(1, int(first.get("bridge_rows", 0) or 0))
        first_rates = {key: float(value) / total for key, value in first["expert_selection_counts"].items()}
    if not last_rates and last.get("expert_selection_counts"):
        total = max(1, int(last.get("bridge_rows", 0) or 0))
        last_rates = {key: float(value) / total for key, value in last["expert_selection_counts"].items()}
    keys = sorted(set(first_rates) | set(last_rates))
    learning_deltas = {
        key: int(last_learning.get(key, 0) or 0) - int(first_learning.get(key, 0) or 0)
        for key in (
            "feedback_events", "development_events", "lifecycle_events",
            "lifecycle_credited", "expectancy_update_events",
            "selector_update_events", "growth_event_count",
        )
    }
    return {
        "available": True,
        "snapshot_count": len(snapshots),
        "first": first,
        "last": last,
        "selection_rate_delta_first_to_last": {key: float(last_rates.get(key, 0.0) - first_rates.get(key, 0.0)) for key in keys},
        "selection_rate_changed": bool(any(abs(last_rates.get(key, 0.0) - first_rates.get(key, 0.0)) > 1e-12 for key in keys)),
        "lifecycle_delta": int(last.get("lifecycle_closed", 0) or 0) - int(first.get("lifecycle_closed", 0) or 0),
        "closed_mean_r_delta": (None if last.get("closed_mean_r") is None or first.get("closed_mean_r") is None else float(last["closed_mean_r"]) - float(first["closed_mean_r"])),
        "closed_profit_factor_last": last.get("closed_profit_factor"),
        "learning_first": first_learning,
        "learning_last": last_learning,
        "learning_deltas": learning_deltas,
        "learning_telemetry_available": bool(last_learning.get("available", False)),
    }


def _frozen_replay_report() -> dict[str, Any]:
    windows = []
    candidate_bridges = sorted(AUDIT.glob("window_*_candidate/bridge.csv"))
    direct_candidate = AUDIT / "candidate" / "bridge.csv"
    if not candidate_bridges and direct_candidate.is_file():
        candidate_bridges = [direct_candidate]
    for candidate_bridge in candidate_bridges:
        if candidate_bridge.parent.name == "candidate":
            index = "direct"
            baseline_bridge = AUDIT / "baseline" / "bridge.csv"
        else:
            index = candidate_bridge.parent.name.split("_", 2)[1]
            baseline_bridge = AUDIT / f"window_{index}_baseline" / "bridge.csv"
        candidate_rows = _decision_file(candidate_bridge)
        baseline_rows = _decision_file(baseline_bridge)
        paired = min(len(candidate_rows), len(baseline_rows))
        action_diff = 0
        expert_diff = 0
        for pos in range(paired):
            if str(candidate_rows[pos].get("action", "")) != str(baseline_rows[pos].get("action", "")):
                action_diff += 1
            c_expert = str(candidate_rows[pos].get("trace_id", "")).split("~", 1)[0]
            b_expert = str(baseline_rows[pos].get("trace_id", "")).split("~", 1)[0]
            if c_expert != b_expert:
                expert_diff += 1
        windows.append({
            "window": index,
            "candidate_selection": _selection(candidate_rows),
            "baseline_selection": _selection(baseline_rows),
            "paired_rows": paired,
            "action_divergence_count": action_diff,
            "action_divergence_rate": action_diff / max(1, paired),
            "expert_divergence_count": expert_diff,
            "expert_divergence_rate": expert_diff / max(1, paired),
            "candidate_selection_timeline": _selection_timeline(candidate_rows),
            "baseline_selection_timeline": _selection_timeline(baseline_rows),
        })
    return {"windows": windows, "window_count": len(windows)}


def main() -> None:
    before = _snapshot(SEED)
    after = _snapshot(CANDIDATE)
    replay = _frozen_replay_report()
    contexts_before = set(before["experts"])
    contexts_after = set(after["experts"])
    diff = {
        "new_experts": sorted(contexts_after - contexts_before),
        "removed_experts": sorted(contexts_before - contexts_after),
        "lifecycle_delta": after["total_lifecycle_closed"] - before["total_lifecycle_closed"],
        "decision_delta": after["total_decisions"] - before["total_decisions"],
        "revision_delta": after["total_policy_revisions"] - before["total_policy_revisions"],
        "expectancy_record_delta": after["expectancy"]["records"] - before["expectancy"]["records"],
        "expectancy_observation_delta": after["expectancy"]["observations"] - before["expectancy"]["observations"],
        "selector_leaf_update_delta": after["selector"]["leaf_outcome_updates"] - before["selector"]["leaf_outcome_updates"],
        "selector_group_update_delta": after["selector"]["group_outcome_updates"] - before["selector"]["group_outcome_updates"],
        "selector_loss_streak_delta": after["selector"]["leaf_loss_streak_sum"] - before["selector"]["leaf_loss_streak_sum"],
    }
    replay_windows = replay.get("windows", [])
    total_action_divergence = sum(int(row.get("action_divergence_count", 0)) for row in replay_windows)
    total_paired_rows = sum(int(row.get("paired_rows", 0)) for row in replay_windows)
    timeline_changed = any(bool(row.get("candidate_selection_timeline", {}).get("changed")) for row in replay_windows)
    execution_paths = sorted(AUDIT.glob("window_*_candidate/execution.csv"))
    bridge_paths = sorted(AUDIT.glob("window_*_candidate/bridge.csv"))
    if not execution_paths and (AUDIT / "candidate" / "execution.csv").is_file():
        execution_paths = [AUDIT / "candidate" / "execution.csv"]
    if not bridge_paths and (AUDIT / "candidate" / "bridge.csv").is_file():
        bridge_paths = [AUDIT / "candidate" / "bridge.csv"]
    loss_followup = _loss_followup(
        _decision_file(bridge_paths[0]) if bridge_paths else [],
        execution_paths[0] if execution_paths else None,
        set(diff["new_experts"]),
    )
    performance = _performance_comparison()
    loss_items = [item for item in (loss_followup.get("experts", {}) or {}).values() if item.get("status") == "loss_observed"]
    loss_response_reasonable = not loss_items or all(bool(item.get("reduced_after_loss")) for item in loss_items)
    live = _live_learning_report()
    live_deltas = dict(live.get("learning_deltas", {}) or {})
    live_learning_available = bool(live.get("learning_telemetry_available", False))
    primary_checks = {
        "lifecycle_updated": (live_deltas.get("lifecycle_credited", 0) > 0 if live_learning_available else diff["lifecycle_delta"] > 0),
        "expectancy_updated": (live_deltas.get("expectancy_update_events", 0) > 0 if live_learning_available else diff["expectancy_observation_delta"] > 0),
        "selector_updated": (
            live_deltas.get("selector_update_events", 0) > 0
            or diff["selector_leaf_update_delta"] > 0
            or diff["selector_group_update_delta"] > 0
        ),
        "selector_rate_changed_over_time": timeline_changed or bool(live.get("selection_rate_changed", False)),
        "new_expert_loss_response_reasonable": loss_response_reasonable,
        "checkpoint_changed": before["sha256"] != after["sha256"],
        "frozen_replay_decisions_changed": total_action_divergence > 0,
        "paired_replay_available": total_paired_rows > 0,
    }
    report = {
        "checkpoint": {"before": before, "after": after, "state_changed": before["sha256"] != after["sha256"]},
        "diff": diff,
        "frozen_replay": replay,
        "performance_comparison": performance,
        "loss_followup": loss_followup,
        "live_learning": live,
        "interpretation": {
            "primary_checks": primary_checks,
            "learning_evidence": bool(all(primary_checks.values())),
            "new_expert_selection_after_loss": "not_applicable_no_new_experts" if not diff["new_experts"] else "inspect_selection_windows",
            "warning": "Frozen replay runs are paired MT5 evaluations; action divergence proves state effect, not necessarily profitable generalization. Performance metrics must still pass the paired non-regression gate.",
        },
    }
    (AUDIT / "learning_proof.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"proof": str(AUDIT / "learning_proof.json"), "diff": diff, "replay_windows": replay["window_count"]}, ensure_ascii=False, indent=2), flush=True)

if __name__ == "__main__":
    main()
