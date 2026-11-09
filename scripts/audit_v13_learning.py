"""Read-only audit for the v13 baseline and the completed V51 candidate.

This script deliberately does not alter source policy code, promote an artifact,
or overwrite either input checkpoint.  It writes only under the audit output
directory and is used to produce the evidence report required by the v13
learning audit objective.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / os.environ.get(
    "V13_AUDIT_BASELINE",
    "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl",
)
CANDIDATE = ROOT / os.environ.get(
    "V13_AUDIT_CANDIDATE",
    "outputs/he_chuyen_gia/v51_v13_v31_hierarchical_role_exposure_parent_train_apr24to_may09_15d_candidate.pkl",
)
EVIDENCE = ROOT / os.environ.get(
    "V13_AUDIT_EVIDENCE",
    "outputs/he_chuyen_gia/evidence/v51_v13_v31_hierarchical_role_exposure_parent_train_apr24to_may09_15d",
)
AUDIT = ROOT / os.environ.get("V13_AUDIT_OUTPUT", "outputs/he_chuyen_gia/audit/v13_learning_audit_v51")
AUDIT.mkdir(parents=True, exist_ok=True)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _artifact_meta(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": int(stat.st_size),
        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "sha256": _sha256(path),
    }


WEIGHT_FIELDS = ("w_in", "w_rec", "b_hidden", "w_out", "b_out", "gate_w", "w_lat", "deep_w", "deep_rec", "deep_b")


def _module_arrays(policy: Any) -> dict[str, list[np.ndarray]]:
    module = (getattr(policy, "modules", None) or [None])[0]
    result: dict[str, list[np.ndarray]] = {}
    for name in WEIGHT_FIELDS:
        value = getattr(module, name, None)
        if isinstance(value, np.ndarray):
            result[name] = [value]
        elif isinstance(value, (list, tuple)):
            result[name] = [item for item in value if isinstance(item, np.ndarray)]
    return result


def _weight_digest(policy: Any) -> dict[str, Any]:
    arrays = _module_arrays(policy)
    total = 0
    nonzero = 0
    squared = 0.0
    digest = hashlib.sha256()
    fields: dict[str, Any] = {}
    for name, values in arrays.items():
        rows = []
        for index, value in enumerate(values):
            raw = np.ascontiguousarray(value).tobytes()
            digest.update(name.encode("utf-8"))
            digest.update(str(index).encode("ascii"))
            digest.update(raw)
            numeric = np.asarray(value, dtype=float)
            total += int(numeric.size)
            nonzero += int(np.count_nonzero(numeric))
            squared += float(np.sum(numeric * numeric))
            rows.append({"index": index, "shape": list(value.shape), "sha256": hashlib.sha256(raw).hexdigest().upper()})
        fields[name] = rows
    return {
        "array_count": sum(len(v) for v in arrays.values()),
        "elements": total,
        "nonzero": nonzero,
        "l2": math.sqrt(squared),
        "sha256": digest.hexdigest().upper(),
        "fields": fields,
    }


def _counter_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (dict, list, tuple, set)):
        return len(value)
    if isinstance(value, np.ndarray):
        return {"shape": list(value.shape), "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest().upper()}
    return type(value).__name__


def _expert_summary(expert: Any) -> dict[str, Any]:
    policy = getattr(expert, "policy", None)
    scalar = {
        key: _counter_value(getattr(expert, key, None))
        for key in ("ma_chuyen_gia", "ma_context", "ma_context_cha", "trang_thai", "so_lifecycle_da_dong", "so_feedback_transition", "so_quyet_dinh", "fitness_cuc_bo", "ma_phien_ban")
    }
    scalar["trang_thai"] = getattr(getattr(expert, "trang_thai", None), "value", str(getattr(expert, "trang_thai", "")))
    scalar["lich_su_len"] = len(getattr(expert, "lich_su", ()) or ())
    if policy is not None:
        for key in ("revision", "best_fitness", "artifact_version", "growth_events", "experience_buffer", "rejection_memory", "_execution_trace_cache", "_causal_trace_archive", "_v9_lifecycle_memory_credited"):
            if hasattr(policy, key):
                scalar[f"policy_{key.lstrip('_')}_len"] = _counter_value(getattr(policy, key))
        scalar["weight_digest"] = _weight_digest(policy)
    return scalar


def _selector_summary(policy: Any) -> dict[str, Any]:
    selector = getattr(policy, "policy_selector", None)
    if selector is None:
        return {}
    result: dict[str, Any] = {}
    for field in ("learned_utility", "outcome_count", "win_signal", "loss_streak", "group_learned_utility", "group_outcome_count", "group_win_signal", "group_loss_streak"):
        value = getattr(selector, field, {}) or {}
        result[field] = {"count": len(value), "sum": float(sum(float(item) for item in value.values()))}
    result["last_trace"] = repr(getattr(selector, "last_trace", None))
    return result


def _policy_state(policy: Any) -> dict[str, Any]:
    experts = getattr(policy, "experts", {}) or {}
    return {
        "expert_count": len(experts),
        "expert_names": sorted(experts),
        "experts": {name: _expert_summary(expert) for name, expert in sorted(experts.items())},
        "selector": _selector_summary(policy),
        "registry_context_count": len(getattr(getattr(policy, "registry", None), "chuyen_gia_theo_context", {}) or {}),
        "registry_children": {str(k): list(v) for k, v in (getattr(getattr(policy, "registry", None), "context_con_theo_cha", {}) or {}).items()},
        "trace_to_expert": len(getattr(policy, "_trace_to_expert", {}) or {}),
        "trace_to_selector_policy": len(getattr(policy, "_trace_to_selector_policy", {}) or {}),
        "selector_credited_trace_ids": len(getattr(policy, "_selector_credited_trace_ids", {}) or {}),
    }


def _learning_state(policy: Any) -> dict[str, Any]:
    """Project out runtime accounting so frozen inference is a valid control.

    Lifecycle counters, trace caches and execution archives can change while
    learning is disabled.  They are useful evidence in ``_policy_state`` but
    must not make a no-learning A/B arm look like it changed its policy.
    """
    state = _policy_state(policy)
    experts: dict[str, Any] = {}
    for name, item in state["experts"].items():
        experts[name] = {
            "weight_digest": item.get("weight_digest"),
            "policy_revision": item.get("policy_revision"),
            "policy_best_fitness": item.get("policy_best_fitness"),
            "policy_artifact_version": item.get("policy_artifact_version"),
        }
    selector = {
        key: value
        for key, value in state.get("selector", {}).items()
        if key != "last_trace"
    }
    return {
        "expert_count": state["expert_count"],
        "expert_names": state["expert_names"],
        "experts": experts,
        "selector": selector,
        "registry_context_count": state["registry_context_count"],
        "registry_children": {key: sorted(value) for key, value in state["registry_children"].items()},
    }


def _weight_delta(base: Any, cand: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in sorted(set(base.experts) & set(cand.experts)):
        left = _module_arrays(base.experts[name].policy)
        right = _module_arrays(cand.experts[name].policy)
        total = nonzero = 0
        squared = base_squared = 0.0
        max_abs = 0.0
        for field in set(left) & set(right):
            for x, y in zip(left[field], right[field]):
                if x.shape != y.shape:
                    continue
                a = np.asarray(x, dtype=float)
                b = np.asarray(y, dtype=float)
                d = b - a
                total += int(d.size)
                nonzero += int(np.count_nonzero(d))
                squared += float(np.sum(d * d))
                base_squared += float(np.sum(a * a))
                if d.size:
                    max_abs = max(max_abs, float(np.max(np.abs(d))))
        result[name] = {
            "elements": total,
            "nonzero": nonzero,
            "l2": math.sqrt(squared),
            "baseline_l2": math.sqrt(base_squared),
            "relative_l2": math.sqrt(squared / base_squared) if base_squared else 0.0,
            "max_abs": max_abs,
            "changed": bool(nonzero),
        }
    return result


def _read_final_rows() -> list[dict[str, str]]:
    path = EVIDENCE / "self_growing_execution.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row for row in rows if str(row.get("position_final_close", "")).strip().lower() in {"1", "true", "yes"}]


def _r(row: dict[str, str]) -> float:
    try:
        return float(row.get("profit", 0.0) or 0.0) / max(abs(float(row.get("initial_risk", 0.0) or 0.0)), 1e-12)
    except (TypeError, ValueError):
        return 0.0


def _group_stats(rows: list[dict[str, str]], field: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(field, "") or "(missing)")].append(_r(row))
    output: dict[str, dict[str, Any]] = {}
    for key, values in sorted(grouped.items()):
        wins = [v for v in values if v > 0]
        losses = [v for v in values if v < 0]
        output[key] = {
            "n": len(values),
            "wr_pct": 100.0 * len(wins) / max(1, len(values)),
            "mean_r": float(np.mean(values)),
            "avg_win_r": float(np.mean(wins)) if wins else 0.0,
            "avg_loss_r": float(np.mean(losses)) if losses else 0.0,
            "net_r": float(np.sum(values)),
            "close_reasons": dict(Counter(str(row.get("close_reason", "")) for row in rows if str(row.get(field, "") or "(missing)") == key)),
        }
    return output


def _credit_audit(candidate: Any, final_rows: list[dict[str, str]]) -> dict[str, Any]:
    owner_missing = sum(not str(row.get("position_owner_trace_id", "")).strip() for row in final_rows)
    owner_mismatch = sum(
        bool(row.get("intent_expert"))
        and bool(row.get("position_owner_expert"))
        and str(row.get("intent_expert")) != str(row.get("position_owner_expert"))
        for row in final_rows
    )
    policy_origin = sum(str(row.get("source", "")) == "policy" for row in final_rows)
    trace_ids = [str(row.get("position_owner_trace_id", "")).strip() for row in final_rows if str(row.get("position_owner_trace_id", "")).strip()]
    credited_ids = set(getattr(candidate, "_selector_credited_trace_ids", {}) or {})
    manager_credits = sum("POSITION_MANAGEMENT" in item for item in credited_ids)
    owner_credit_ids = {item for item in credited_ids if "POSITION_MANAGEMENT" not in item}
    return {
        "final_lifecycle_count": len(final_rows),
        "owner_trace_missing": owner_missing,
        "credit_missing_rate_pct": 100.0 * owner_missing / max(1, len(final_rows)),
        "owner_mismatch": owner_mismatch,
        "owner_mismatch_rate_pct": 100.0 * owner_mismatch / max(1, len(final_rows)),
        "policy_origin_final_count": policy_origin,
        "broker_or_unknown_final_count": len(final_rows) - policy_origin,
        "owner_trace_unique": len(set(trace_ids)),
        "selector_credited_trace_count": len(credited_ids),
        "selector_credited_unique": len(credited_ids) == len(set(credited_ids)),
        "entry_or_owner_credits": len(owner_credit_ids),
        "management_credits": manager_credits,
        "economic_lifecycle_credit_multiplicity": manager_credits / max(1, len(final_rows)),
        "credit_duplicated_by_management_trace": bool(manager_credits),
        "sample_wins": [
            {key: row.get(key, "") for key in ("logged_at", "position_ticket", "position_owner_trace_id", "position_owner_expert", "position_owner_context", "close_reason", "profit", "initial_risk")}
            for row in final_rows if _r(row) > 0
        ][:3],
        "sample_losses": [
            {key: row.get(key, "") for key in ("logged_at", "position_ticket", "position_owner_trace_id", "position_owner_expert", "position_owner_context", "close_reason", "profit", "initial_risk")}
            for row in final_rows if _r(row) < 0
        ][:3],
    }


def _write_roundtrip(candidate: Any) -> dict[str, Any]:
    path = AUDIT / "candidate_roundtrip.pkl"
    with path.open("wb") as handle:
        pickle.dump(candidate, handle, protocol=pickle.HIGHEST_PROTOCOL)
    loaded = pickle.load(path.open("rb"))
    before = _policy_state(candidate)
    after = _policy_state(loaded)
    equal = before == after
    return {"path": str(path.resolve()), "meta": _artifact_meta(path), "equal": equal, "before": before, "after": after}


def main() -> None:
    baseline = pickle.load(BASELINE.open("rb"))
    candidate = pickle.load(CANDIDATE.open("rb"))
    final_rows = _read_final_rows()
    baseline_state = _policy_state(baseline)
    candidate_state = _policy_state(candidate)
    state = {
        "baseline": baseline_state,
        "candidate": candidate_state,
        "weight_delta": _weight_delta(baseline, candidate),
        "candidate_only_experts": sorted(set(candidate.experts) - set(baseline.experts)),
        "baseline_only_experts": sorted(set(baseline.experts) - set(candidate.experts)),
        "parameters_changed": any(item["changed"] for item in _weight_delta(baseline, candidate).values()),
        "policy_relevant_state_changed": baseline_state != candidate_state,
        "credit": _credit_audit(candidate, final_rows),
        "progression": {
            "by_day": _group_stats(final_rows, "logged_at_day"),
            "by_owner_context": _group_stats(final_rows, "position_owner_context"),
            "by_owner_expert": _group_stats(final_rows, "position_owner_expert"),
            "by_close_reason": _group_stats(final_rows, "close_reason"),
        },
    }
    for row in final_rows:
        row["logged_at_day"] = str(row.get("logged_at", "")).split(" ", 1)[0]
    state["progression"] = {
        "by_day": _group_stats(final_rows, "logged_at_day"),
        "by_owner_context": _group_stats(final_rows, "position_owner_context"),
        "by_owner_expert": _group_stats(final_rows, "position_owner_expert"),
        "by_close_reason": _group_stats(final_rows, "close_reason"),
    }
    state["artifacts"] = {"baseline": _artifact_meta(BASELINE), "candidate": _artifact_meta(CANDIDATE)}
    state["save_load"] = _write_roundtrip(candidate)
    (AUDIT / "state_audit.json").write_text(json.dumps(state, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "audit": str((AUDIT / "state_audit.json").resolve()),
        "parameters_changed": state["parameters_changed"],
        "policy_relevant_state_changed": state["policy_relevant_state_changed"],
        "final_lifecycles": state["credit"]["final_lifecycle_count"],
        "credit_missing_rate_pct": state["credit"]["credit_missing_rate_pct"],
        "owner_mismatch_rate_pct": state["credit"]["owner_mismatch_rate_pct"],
        "management_credits": state["credit"]["management_credits"],
        "save_load_equal": state["save_load"]["equal"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
