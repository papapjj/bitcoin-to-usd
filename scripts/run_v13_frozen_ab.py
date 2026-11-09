"""Run a read-only MT5 frozen A/B for the locked seed and V51 candidate."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import shutil
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from self_growing_trader.chuyen_gia_policy import _cau_hinh_policy_chuyen_gia_cho_mt5, _mt5_execution_contract_kwargs
from self_growing_trader.config import load_config
from self_growing_trader.mt5_runner import MT5BacktestConfig, MT5StrategyTesterEvaluator, MT5_MODEL_REAL_TICKS

from audit_v13_learning import _learning_state, _policy_state


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / os.environ.get(
    "V13_FROZEN_BASELINE",
    "outputs/he_chuyen_gia/seeds/v13_original_7experts_locked_for_continuation.pkl",
)
CANDIDATE = ROOT / os.environ.get(
    "V13_FROZEN_CANDIDATE",
    "outputs/he_chuyen_gia/v51_v13_v31_hierarchical_role_exposure_parent_train_apr24to_may09_15d_candidate.pkl",
)
OUT = ROOT / os.environ.get("V13_FROZEN_OUT", "outputs/he_chuyen_gia/audit/v13_frozen_ab_oos_may10to14")
OUT.mkdir(parents=True, exist_ok=True)
FROM_DATE = os.environ.get("V13_FROZEN_FROM", "2025.05.10")
TO_DATE = os.environ.get("V13_FROZEN_TO", "2025.05.14")
FROZEN_MODEL = int(os.environ.get("V13_FROZEN_MODEL", str(MT5_MODEL_REAL_TICKS)) or MT5_MODEL_REAL_TICKS)
FROZEN_MAX_WALL_SEC = max(0, int(os.environ.get("V13_FROZEN_MAX_WALL_SEC", "1800") or 1800))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _latest_runtime_file(runner: Path, name: str) -> Path | None:
    paths = list(runner.glob(f"**/{name}"))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime_ns)


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    raw = dict(getattr(metrics, "__dict__", {}) or {})
    for key in ("report_path", "execution_ledger_path", "bridge_path", "feature_audit_path"):
        if raw.get(key) is not None:
            raw[key] = str(raw[key])
    for key in ("reconciliation", "execution_summary", "feature_audit", "real_tick_evidence"):
        value = raw.get(key)
        if hasattr(value, "__dict__"):
            converted = dict(value.__dict__)
            if key == "execution_summary":
                lifecycles = converted.pop("lifecycles", ())
                converted["lifecycle_count"] = len(lifecycles or ())
            raw[key] = converted
        elif value is not None and not isinstance(value, (dict, list, str, int, float, bool)):
            raw[key] = str(value)
    return raw


def _copy_runtime(runner: Path, target: Path, metrics: Any) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=True)
    copied: dict[str, Any] = {}
    candidates = {
        "report": Path(str(getattr(metrics, "report_path", ""))) if getattr(metrics, "report_path", None) else None,
        "execution": Path(str(getattr(metrics, "execution_ledger_path", ""))) if getattr(metrics, "execution_ledger_path", None) else None,
        "bridge": Path(str(getattr(metrics, "bridge_path", ""))) if getattr(metrics, "bridge_path", None) else None,
        "feature_audit": Path(str(getattr(metrics, "feature_audit_path", ""))) if getattr(metrics, "feature_audit_path", None) else None,
    }
    for name in ("self_growing_action.csv", "self_growing_request.csv"):
        candidates[name.rsplit(".", 1)[0]] = _latest_runtime_file(runner, name)
    for name, path in candidates.items():
        if path is None or not path.exists():
            continue
        dst = target / (path.name if name.startswith("self_") else f"{name}{path.suffix}")
        shutil.copy2(path, dst)
        copied[name] = {"path": str(dst.resolve()), "bytes": dst.stat().st_size, "sha256": _sha256(dst)}
    return copied


def _csv_counts(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    actions = [str(row.get("action", "")) for row in rows]
    return {
        "rows": len(rows),
        "action_counts": dict(Counter(actions)),
        "masked": sum(str(row.get("action_masked", "0")).lower() in {"1", "true", "yes"} for row in rows),
        "trace_count": len({str(row.get("trace_id", "")) for row in rows if row.get("trace_id")}),
        "rows_preview": rows[:2],
    }


def _state_diff(before: Any, after: Any, path: str = "") -> list[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        changes: list[str] = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            changes.extend(_state_diff(before.get(key), after.get(key), child))
            if len(changes) >= 80:
                return changes[:80]
        return changes
    if before != after:
        return [path or "<root>"]
    return []


def _config(run_id: str) -> tuple[Any, MT5BacktestConfig]:
    app = load_config(ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    kwargs = _mt5_execution_contract_kwargs(app)
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except (TypeError, ValueError):
            return int(default)

    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except (TypeError, ValueError):
            return float(default)

    # Frozen A/B is an inference measurement.  Full live observability and
    # the long causal trace archive are useful during training, but retaining
    # graph-sized snapshots while serving every tester request can consume
    # gigabytes and turn a valid OOS run into a runtime stall.  Keep bridge,
    # execution-ledger and feature-audit evidence; disable only duplicate
    # in-memory telemetry/replay state for this read-only arm.
    app = replace(
        app,
        mt5_learning_enabled=False,
        observability=replace(app.observability, enabled=False),
        causal=replace(app.causal, enabled=False, max_decisions=128, max_reports=128),
        learning=replace(
            app.learning,
            context_direction_entry_gate_enabled=_env_bool("V13_CONTEXT_ENTRY_GATE", False),
            context_direction_side_entry_gate_enabled=_env_bool("V13_CONTEXT_SIDE_GATE", False),
            context_expectancy_entry_gate_enabled=_env_bool("V13_CONTEXT_EXPECTANCY_GATE", False),
            context_expectancy_entry_gate_min_support=_env_float("V13_CONTEXT_EXPECTANCY_MIN_SUPPORT", 12.0),
            context_expectancy_entry_gate_max_lcb=_env_float("V13_CONTEXT_EXPECTANCY_MAX_LCB", -0.15),
        ),
    )

    kwargs.update({
        "runner_dir": ROOT / "mt5_v2_runner",
        "from_date": FROM_DATE,
        "to_date": TO_DATE,
        "timeout_sec": 120,
        # A frozen arm is read-only and must fail closed instead of leaving a
        # tester child running after the caller has timed out.  Zero remains
        # available explicitly for a long audit, but the default is bounded.
        "max_wall_sec": FROZEN_MAX_WALL_SEC,
        "max_positions": int(app.risk.max_positions),
        "max_holding_bars": int(app.orders.max_holding_bars),
        "min_daily_trades": 0,
        "max_daily_trades": 0,
        "suppress_flat_idle_starvation": True,
        "visual": False,
        "shutdown_terminal": _env_bool("V13_FROZEN_SHUTDOWN_TERMINAL", False),
        "model": FROZEN_MODEL,
        "experience_journal": None,
        "learning_enabled": False,
        "enable_execution_bootstrap": False,
        "train_entry_probe_enabled": False,
        "entry_loss_guard_enabled": False,
        "runtime_fitness_guard_enabled": _env_bool("V13_RUNTIME_GUARD", False),
        "runtime_fitness_min_closed": _env_int("V13_RUNTIME_MIN_CLOSED", 20),
        "runtime_fitness_max_daily_loss_r": _env_float("V13_RUNTIME_MAX_DAILY_LOSS_R", 3.0),
        "runtime_fitness_max_run_loss_r": _env_float("V13_RUNTIME_MAX_RUN_LOSS_R", 8.0),
        "runtime_fitness_max_drawdown_pct": _env_float("V13_RUNTIME_MAX_DD_PCT", 20.0),
        "require_real_ticks_for_promotion": False,
        "require_real_tick_evidence_for_promotion": False,
        "run_id": run_id,
    })
    return app, MT5BacktestConfig(**kwargs)


def _run(label: str, source: Path, policy_id: int) -> dict[str, Any]:
    print(f"[FROZEN] loading {label}", flush=True)
    policy = pickle.load(source.open("rb"))
    app, cfg = _config(f"v13-frozen-ab-{label}")
    # Apply the same runtime contract to both arms; do not persist this in the
    # source artifact. The evaluator itself is learning-disabled.
    _cau_hinh_policy_chuyen_gia_cho_mt5(policy, app)
    before = _policy_state(policy)
    learning_before = _learning_state(policy)
    evaluator = MT5StrategyTesterEvaluator(cfg)
    print(f"[FROZEN] start {label} {FROM_DATE}->{TO_DATE}", flush=True)
    fitness, metrics = evaluator.evaluate_policy(policy_id, policy, 0)
    after = _policy_state(policy)
    learning_after = _learning_state(policy)
    run_dir = OUT / label
    copied = _copy_runtime(cfg.runner_dir, run_dir, metrics)
    # The action file is a last-row wire snapshot; bridge is the complete
    # decision stream used for A/B action and masking counts.
    action_path = Path(copied["bridge"]["path"]) if "bridge" in copied else None
    result = {
        "label": label,
        "source": str(source.resolve()),
        "source_sha256": _sha256(source),
        "from_date": FROM_DATE,
        "to_date": TO_DATE,
        "model": cfg.model,
        "learning_enabled": cfg.learning_enabled,
        "experience_journal": cfg.experience_journal,
        "fitness": float(fitness),
        "metrics": _metrics_dict(metrics),
        "runtime_files": copied,
        "decision_counts": _csv_counts(action_path),
        "policy_relevant_state_before_after_equal": before == after,
        "learning_state_before_after_equal": learning_before == learning_after,
        "learning_state_diff_paths": _state_diff(learning_before, learning_after),
        "runtime_state_before_after_equal": before == after,
    }
    (run_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[FROZEN] done {label} fitness={fitness} state_equal={before == after}", flush=True)
    return result


def main() -> None:
    arm = str(os.environ.get("V13_FROZEN_ARM", "both") or "both").lower()
    # Audits may be repeated against the same terminal without reusing a
    # stale MT5 report name.  Keep the historical defaults for callers that do
    # not set the override, but allow an isolated policy id per run.
    policy_base = int(os.environ.get("V13_FROZEN_POLICY_BASE", "9200") or "9200")
    results = {"oos_window": {"from": FROM_DATE, "to": TO_DATE, "model": FROZEN_MODEL, "learning_enabled": False}}
    if arm in {"both", "baseline"}:
        results["baseline"] = _run("baseline", BASELINE, policy_base + 1)
    if arm in {"both", "candidate"}:
        results["candidate"] = _run("candidate", CANDIDATE, policy_base + 2)
    (OUT / "ab_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({
        "summary": str((OUT / "ab_summary.json").resolve()),
        **({"baseline_state_equal": results["baseline"]["policy_relevant_state_before_after_equal"], "baseline_fitness": results["baseline"]["fitness"]} if "baseline" in results else {}),
        **({"candidate_state_equal": results["candidate"]["policy_relevant_state_before_after_equal"], "candidate_fitness": results["candidate"]["fitness"]} if "candidate" in results else {}),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
