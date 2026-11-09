"""Run three isolated MT5 arms to attribute online Full-RL policy drift."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import random
import shutil
from dataclasses import replace
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
import yaml

from self_growing_trader.chuyen_gia_policy import (
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_execution_contract_kwargs,
)
from self_growing_trader.config import load_config
from self_growing_trader.mt5_runner import MT5BacktestConfig, MT5StrategyTesterEvaluator

from audit_v13_learning import _learning_state
from run_v13_frozen_ab import _copy_runtime, _metrics_dict, _state_diff


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / os.environ.get(
    "FULL_RL_ABC_SOURCE",
    "outputs/he_chuyen_gia/candidates/v153_v13_compat_cachefix_model1_20260723_15d_keepmt5.pkl",
)
OUT = ROOT / os.environ.get(
    "FULL_RL_ABC_OUT",
    "outputs/he_chuyen_gia/audit/full_rl_drift_abc_20260723",
)
CONFIGS = (
    ROOT / "configs/experiments/full_rl_drift_a_frozen.yaml",
    ROOT / "configs/experiments/full_rl_drift_b_tiny_lr.yaml",
    ROOT / "configs/experiments/full_rl_drift_c_current.yaml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _load_arm(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    arm = dict(raw.get("experiment", {}))
    required = {
        "id", "label", "learning_enabled", "direct_rnn_full_rl_enabled",
        "direct_rnn_adamw_enabled", "direct_rnn_adamw_learning_rate",
        "seed", "from_date", "to_date", "mt5_model",
    }
    missing = sorted(required - set(arm))
    if missing:
        raise ValueError(f"{path}: missing {missing}")
    return arm


def _parameter_snapshot(policy: Any, settlement_index: int, trace_id: str = "") -> dict[str, Any]:
    expert_policies = [
        getattr(expert, "policy", None)
        for _, expert in sorted((getattr(policy, "experts", {}) or {}).items())
    ]
    expert_policies = [item for item in expert_policies if item is not None]
    if not expert_policies:
        expert_policies = [policy]
    modules = [
        module
        for owner in expert_policies
        for module in list(getattr(owner, "modules", []) or [])
    ]
    weights: list[np.ndarray] = []
    optimizer_steps = 0
    for module in modules:
        optimizer_steps += int(getattr(module, "rnn_optimizer_step", 0) or 0)
        for name in ("w_in", "w_rec", "w_out", "value_w", "gate_w"):
            value = getattr(module, name, None)
            if value is not None:
                weights.append(np.asarray(value, dtype=np.float64).reshape(-1))
        for name in ("deep_w", "deep_rec", "deep_b"):
            weights.extend(np.asarray(item, dtype=np.float64).reshape(-1) for item in list(getattr(module, name, []) or []))
    flat = np.concatenate(weights) if weights else np.zeros(0, dtype=np.float64)
    gate_values: list[np.ndarray] = []
    gate_age_total = 0
    for owner in expert_policies:
        gate = getattr(owner, "risk_cost_gate", None)
        if gate is None:
            continue
        gate_age_total += int(getattr(gate, "age", 0) or 0)
        for name in ("w_in", "w_rec", "w_out", "b_hidden", "b_out"):
            value = getattr(gate, name, None)
            if value is not None:
                gate_values.append(np.asarray(value, dtype=np.float64).reshape(-1))
    gate_flat = np.concatenate(gate_values) if gate_values else np.zeros(0, dtype=np.float64)
    return {
        "settlement_index": int(settlement_index),
        "trace_id": str(trace_id),
        "outcome_r": None,
        "optimizer_steps_total": int(optimizer_steps),
        "expert_policy_count": int(len(expert_policies)),
        "direct_rnn_l2": float(np.linalg.norm(flat)),
        "direct_rnn_sum": float(np.sum(flat)),
        "risk_gate_l2": float(np.linalg.norm(gate_flat)),
        "risk_gate_sum": float(np.sum(gate_flat)),
        "risk_gate_age_total": int(gate_age_total),
    }


def _attach_learning_probe(policy: Any, rows: list[dict[str, Any]]) -> None:
    original = policy.apply_execution_lifecycle_credit
    rows.append(_parameter_snapshot(policy, 0, "initial"))

    def probed(self: Any, trace_id: str, outcome: float, **kwargs: Any) -> int:
        result = original(trace_id, outcome, **kwargs)
        snapshot = _parameter_snapshot(self, len(rows), trace_id)
        snapshot["outcome_r"] = float(outcome)
        rows.append(snapshot)
        return result

    policy.apply_execution_lifecycle_credit = MethodType(probed, policy)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _equity_series(bridge_path: Path, target: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with bridge_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            try:
                rows.append({
                    "time": row.get("time", ""),
                    "bar": int(float(row.get("bar", 0) or 0)),
                    "balance": float(row.get("balance", 0.0) or 0.0),
                    "equity": float(row.get("equity", 0.0) or 0.0),
                    "drawdown_pct": float(row.get("drawdown_pct", 0.0) or 0.0),
                    "positions": int(float(row.get("positions", 0) or 0)),
                    "day_trades": int(float(row.get("day_trades", 0) or 0)),
                })
            except (TypeError, ValueError):
                continue
    _write_csv(target, rows)
    return {
        "points": len(rows),
        "ending_balance": rows[-1]["balance"] if rows else None,
        "ending_equity": rows[-1]["equity"] if rows else None,
        "max_stream_drawdown_pct": max((row["drawdown_pct"] for row in rows), default=0.0),
    }


def _configure(arm: dict[str, Any], run_id: str) -> tuple[Any, MT5BacktestConfig]:
    app = load_config(ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    learning_enabled = bool(arm["learning_enabled"])
    app = replace(
        app,
        mt5_learning_enabled=learning_enabled,
        observability=replace(app.observability, enabled=False),
        causal=replace(app.causal, enabled=False, max_decisions=128, max_reports=128),
        learning=replace(
            app.learning,
            direct_rnn_full_rl_enabled=bool(arm["direct_rnn_full_rl_enabled"]),
            direct_rnn_adamw_enabled=bool(arm["direct_rnn_adamw_enabled"]),
            direct_rnn_adamw_learning_rate=float(arm["direct_rnn_adamw_learning_rate"]),
        ),
    )
    kwargs = _mt5_execution_contract_kwargs(app)
    kwargs.update({
        "runner_dir": ROOT / "mt5_v2_runner",
        "from_date": str(arm["from_date"]),
        "to_date": str(arm["to_date"]),
        "model": int(arm["mt5_model"]),
        "timeout_sec": 120,
        "max_wall_sec": int(os.environ.get("FULL_RL_ABC_MAX_WALL_SEC", "1800")),
        "visual": False,
        # This MT5 build forwards a second /config launch to the live terminal
        # without starting a fresh Strategy Tester pass. Restart only this
        # portable runner between arms so every config is actually executed.
        "shutdown_terminal": True,
        "learning_enabled": learning_enabled,
        "experience_journal": str((OUT / str(arm["id"]) / "experience.jsonl").resolve()) if learning_enabled else None,
        "enable_execution_bootstrap": False,
        "train_entry_probe_enabled": False,
        "entry_loss_guard_enabled": False,
        "runtime_fitness_guard_enabled": False,
        "require_real_ticks_for_promotion": False,
        "require_real_tick_evidence_for_promotion": False,
        "run_id": run_id,
    })
    return app, MT5BacktestConfig(**kwargs)


def _run_arm(arm: dict[str, Any], source_hash: str, policy_id: int) -> dict[str, Any]:
    arm_dir = OUT / str(arm["id"])
    arm_dir.mkdir(parents=True, exist_ok=True)
    clone = arm_dir / "initial_model_copy.pkl"
    shutil.copy2(SOURCE, clone)
    if _sha256(clone) != source_hash:
        raise RuntimeError(f"Model clone hash mismatch for arm {arm['id']}")
    seed = int(arm["seed"])
    random.seed(seed)
    np.random.seed(seed)
    with clone.open("rb") as handle:
        policy = pickle.load(handle)
    app, cfg = _configure(arm, f"full-rl-drift-{str(arm['id']).lower()}")
    _cau_hinh_policy_chuyen_gia_cho_mt5(policy, app)
    before = _learning_state(policy)
    snapshots: list[dict[str, Any]] = []
    _attach_learning_probe(policy, snapshots)
    evaluator = MT5StrategyTesterEvaluator(cfg)
    fitness, metrics = evaluator.evaluate_policy(policy_id, policy, 0)
    after = _learning_state(policy)
    copied = _copy_runtime(cfg.runner_dir, arm_dir, metrics)
    _write_csv(arm_dir / "learning_timeline.csv", snapshots)
    equity = {}
    bridge = copied.get("bridge", {}).get("path")
    if bridge:
        equity = _equity_series(Path(bridge), arm_dir / "equity_balance_timeline.csv")
    result = {
        "arm": arm,
        "source": str(SOURCE.resolve()),
        "source_sha256": source_hash,
        "clone": str(clone.resolve()),
        "clone_sha256": _sha256(clone),
        "fitness": float(fitness),
        "metrics": _metrics_dict(metrics),
        "equity_stream": equity,
        "learning_snapshot_count": len(snapshots),
        "learning_state_equal": before == after,
        "learning_state_diff_paths": _state_diff(before, after),
        "initial_parameters": snapshots[0] if snapshots else {},
        "final_parameters": snapshots[-1] if snapshots else {},
        "runtime_files": copied,
    }
    (arm_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256(SOURCE)
    requested = str(os.environ.get("FULL_RL_ABC_ARMS", "ABC") or "ABC").upper()
    loaded_arms = [_load_arm(path) for path in CONFIGS]
    arms = [arm for arm in loaded_arms if str(arm["id"]).upper() in requested]
    results: list[dict[str, Any]] = []
    for arm in arms:
        print(f"[ABC] start arm={arm['id']} label={arm['label']}", flush=True)
        policy_id = 15300 + max(0, ord(str(arm["id"]).upper()[0]) - ord("A"))
        results.append(_run_arm(arm, source_hash, policy_id))
        if _sha256(SOURCE) != source_hash:
            raise RuntimeError("Protected source model changed during experiment")
        print(f"[ABC] done arm={arm['id']}", flush=True)
    completed = {str(item["arm"]["id"]): item for item in results}
    for arm in loaded_arms:
        arm_id = str(arm["id"])
        result_path = OUT / arm_id / "result.json"
        if arm_id not in completed and result_path.exists():
            completed[arm_id] = json.loads(result_path.read_text(encoding="utf-8"))
    results = [completed[key] for key in sorted(completed)]
    summary = {
        "source": str(SOURCE.resolve()),
        "source_sha256_before_after": source_hash,
        "source_unchanged": _sha256(SOURCE) == source_hash,
        "same_seed": len({int(item["arm"]["seed"]) for item in results}) <= 1,
        "same_window": len({(item["arm"]["from_date"], item["arm"]["to_date"]) for item in results}) <= 1,
        "same_mt5_model": len({int(item["arm"]["mt5_model"]) for item in results}) <= 1,
        "results": results,
    }
    (OUT / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps({
        "comparison": str((OUT / "comparison.json").resolve()),
        "source_unchanged": summary["source_unchanged"],
        "arms": [item["arm"]["id"] for item in results],
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
