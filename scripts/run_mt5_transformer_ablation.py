#!/usr/bin/env python3
"""Run a frozen paired MT5 OOS intervention for Five v3 temporal tokens."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
from typing import Any

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import shared_five_network
from self_growing_trader.self_growing_runner import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)


CANDIDATE_SCRIPT = ROOT / "scripts" / "run_edge_discovery_v193_candidate.py"
TEMPORAL_SCALE_KEY = "post_lstm_transformer.temporal_scale"


def _require_schema10_seed(source: Path, source_hash: str) -> dict[str, object]:
    from scripts.run_tensor_rule_v186_candidate import inspect_imitation_seed

    evidence = inspect_imitation_seed(source, checkpoint_sha256=source_hash)
    if not (
        int(evidence.get("report_schema_version", 0)) >= 10
        and evidence.get("effective_for_mt5") is True
        and evidence.get("validation_gate_enforced") is True
        and evidence.get("validation_gate_passed") is True
        and evidence.get("candidate_reload_passed") is True
    ):
        raise RuntimeError(
            "paired temporal MT5 requires schema-10 full-validation/reload PASS: "
            + json.dumps(evidence, ensure_ascii=False)
        )
    return evidence


def _validate_bounded_window(
    from_date: str,
    to_date: str,
    *,
    max_span_days: int,
    allow_long_window: bool,
) -> int:
    start = datetime.strptime(str(from_date), "%Y.%m.%d")
    end = datetime.strptime(str(to_date), "%Y.%m.%d")
    days = (end - start).days
    if days < 1:
        raise ValueError("to-date must be after from-date")
    if days > int(max_span_days) and not allow_long_window:
        raise ValueError(
            f"fail-fast window rejects {days} days > {max_span_days}; "
            "pass --allow-long-window only after a one-day pair completes"
        )
    return days


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _network_state_hashes(network: torch.nn.Module) -> dict[str, str]:
    return {name: _tensor_sha256(value) for name, value in network.state_dict().items()}


def _changed_state_keys(before: dict[str, str], after: dict[str, str]) -> list[str]:
    return sorted(
        key for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    )


def _derived_imitation_sidecar(source: Path, target: Path, intervention: str) -> None:
    source_sidecar = source.with_suffix(source.suffix + ".imitation.json")
    if not source_sidecar.is_file():
        return
    report = json.loads(source_sidecar.read_text(encoding="utf-8"))
    report["candidate_checkpoint_sha256"] = _sha256(target)
    report["evaluation_intervention"] = {
        "kind": intervention,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": _sha256(source),
        "training_performed": False,
    }
    target.with_suffix(target.suffix + ".imitation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def create_intervention_seeds(
    source: Path,
    config_path: Path,
    seed_dir: Path,
) -> tuple[dict[str, Path], dict[str, object]]:
    config = load_config(config_path)
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        source, config, policy_core="current", seed=0
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    before = _network_state_hashes(network)
    original_scale = float(network.post_lstm_transformer.temporal_scale.detach().cpu())
    if abs(original_scale) <= 1e-8:
        raise RuntimeError(
            "v3 temporal OOS requires a trained checkpoint with temporal_scale != 0"
        )
    on_path = seed_dir / "v3_temporal_on.pkl"
    off_path = seed_dir / "v2_temporal_scale_0.pkl"
    _atomic_pickle(system, on_path)
    with torch.no_grad():
        network.post_lstm_transformer.temporal_scale.zero_()
    after = _network_state_hashes(network)
    changed = _changed_state_keys(before, after)
    if changed != [TEMPORAL_SCALE_KEY]:
        raise RuntimeError(f"intervention changed unexpected tensors: {changed}")
    _atomic_pickle(system, off_path)

    reloaded_on = shared_five_network(
        tai_he_chuyen_gia_mot_context_mot_policy(
            on_path, config, policy_core="current", seed=0
        )
    )
    reloaded_off = shared_five_network(
        tai_he_chuyen_gia_mot_context_mot_policy(
            off_path, config, policy_core="current", seed=0
        )
    )
    reload_changed = _changed_state_keys(
        _network_state_hashes(reloaded_on), _network_state_hashes(reloaded_off)
    )
    off_scale = float(reloaded_off.post_lstm_transformer.temporal_scale.detach().cpu())
    if reload_changed != [TEMPORAL_SCALE_KEY] or off_scale != 0.0:
        raise RuntimeError(
            f"reload intervention mismatch: changed={reload_changed}, off_scale={off_scale}"
        )
    _derived_imitation_sidecar(source, on_path, "v3_temporal_on_canonical_clone")
    _derived_imitation_sidecar(source, off_path, "v2_temporal_scale_zero")
    paths = {"v3_temporal_on": on_path, "v2_temporal_off": off_path}
    return paths, {
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": _sha256(source),
        "original_temporal_scale": original_scale,
        "intervened_temporal_scale": off_scale,
        "changed_network_state_keys": changed,
        "reload_changed_network_state_keys": reload_changed,
        "arm_checkpoint_sha256": {
            name: _sha256(path) for name, path in paths.items()
        },
        "reload_validation_passed": True,
    }


def _arm_profile(base: dict[str, Any], arm: str) -> dict[str, Any]:
    profile = json.loads(json.dumps(base))
    experiment = dict(profile.get("experiment", {}) or {})
    experiment["id"] = f"transformer_ablation_{arm}"
    experiment["ablation_arm"] = arm
    experiment["architecture_ablation"] = True
    profile["experiment"] = experiment
    app = dict(profile.get("app_overrides", {}) or {})
    app["mt5_learning_enabled"] = False
    profile["app_overrides"] = app
    return profile


def _discard_arm_checkpoints(arm_dir: Path) -> list[str]:
    removed = []
    root = arm_dir.resolve()
    for path in arm_dir.glob("*.pkl"):
        if path.resolve().parent != root:
            raise RuntimeError(f"refusing checkpoint cleanup outside arm: {path}")
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def _load_arm_manifest(arm_dir: Path) -> dict[str, Any]:
    path = arm_dir / "manifest.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _arm_completion(exit_code: int, manifest: dict[str, Any]) -> tuple[bool, str]:
    if int(exit_code) != 0:
        return False, f"child_exit_code_{int(exit_code)}"
    mt5 = dict(manifest.get("mt5", {}) or {})
    metrics = dict(mt5.get("metrics", {}) or {})
    if not mt5:
        return False, "missing_mt5_result"
    if bool(mt5.get("interrupted", False)):
        return False, str(metrics.get("interrupted_reason", "mt5_interrupted"))
    if bool(metrics.get("incomplete", False)):
        return False, str(metrics.get("interrupted_reason", "mt5_incomplete"))
    if not str(mt5.get("policy", "") or ""):
        return False, "missing_completed_policy"
    return True, "completed_mt5_report"


def _metric(metrics: dict[str, Any], key: str) -> Any:
    value: Any = metrics
    for part in key.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _difference(left: Any, right: Any) -> float | None:
    try:
        return float(left) - float(right)
    except (TypeError, ValueError):
        return None


def _paired_oos_gate(
    on: dict[str, Any], off: dict[str, Any], *, minimum_lifecycles: int
) -> dict[str, object]:
    minimum_lifecycles = max(50, int(minimum_lifecycles))
    on_execution = dict(on.get("execution_summary", {}) or {})
    off_execution = dict(off.get("execution_summary", {}) or {})
    on_expectancy = dict(dict(on.get("expectancy_audit", {}) or {}).get(
        "policy_expectancy", {}
    ) or {})
    off_expectancy = dict(dict(off.get("expectancy_audit", {}) or {}).get(
        "policy_expectancy", {}
    ) or {})
    deltas = {
        "net_profit": _difference(on.get("net_profit"), off.get("net_profit")),
        "profit_factor": _difference(on.get("profit_factor"), off.get("profit_factor")),
        "policy_expectancy_mean_r": _difference(
            on_expectancy.get("mean_net_r"), off_expectancy.get("mean_net_r")
        ),
        "policy_expectancy_lcb_r": _difference(
            on_expectancy.get("expectancy_lcb"), off_expectancy.get("expectancy_lcb")
        ),
        "equity_drawdown_pct": _difference(
            on.get("equity_drawdown_pct"), off.get("equity_drawdown_pct")
        ),
    }
    support = {
        "v3_temporal_on": int(on_expectancy.get("sample_count", 0) or 0),
        "v2_temporal_off": int(off_expectancy.get("sample_count", 0) or 0),
    }
    checks = {
        "minimum_policy_lifecycles": min(support.values(), default=0) >= int(minimum_lifecycles),
        "both_feature_parity_valid": all(
            dict(arm.get("feature_audit", {}) or {}).get("valid") is True
            and dict(arm.get("feature_audit", {}) or {}).get("numerical_valid") is True
            and int(dict(arm.get("feature_audit", {}) or {}).get("checked_rows", 0)) > 0
            for arm in (on, off)
        ),
        "both_reconciled": bool(dict(on.get("reconciliation", {}) or {}).get("reconciled", False))
        and bool(dict(off.get("reconciliation", {}) or {}).get("reconciled", False)),
        "both_execution_ledgers_complete": bool(on_execution.get("ledger_integrity", 0.0) == 1.0)
        and bool(off_execution.get("ledger_integrity", 0.0) == 1.0),
        "v3_net_profit_strictly_better": deltas["net_profit"] is not None
        and float(deltas["net_profit"]) > 0.0,
        "v3_profit_factor_not_worse": deltas["profit_factor"] is not None
        and float(deltas["profit_factor"]) >= 0.0,
        "v3_policy_expectancy_strictly_better": deltas["policy_expectancy_mean_r"] is not None
        and float(deltas["policy_expectancy_mean_r"]) > 0.0,
        "v3_policy_expectancy_lcb_not_worse": deltas["policy_expectancy_lcb_r"] is not None
        and float(deltas["policy_expectancy_lcb_r"]) >= 0.0,
        "v3_drawdown_not_materially_worse": deltas["equity_drawdown_pct"] is not None
        and float(deltas["equity_drawdown_pct"]) <= 0.25,
    }
    return {
        "passed": all(checks.values()),
        "minimum_policy_lifecycles_per_arm": int(minimum_lifecycles),
        "support": support,
        "deltas_v3_minus_v2": deltas,
        "checks": checks,
        "candidate_only": True,
        "promotable": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--model", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-wall-sec", type=int, default=300)
    parser.add_argument("--max-decision-packets", type=int, default=256)
    parser.add_argument("--minimum-policy-lifecycles", type=int, default=50)
    parser.add_argument("--max-span-days", type=int, default=1)
    parser.add_argument("--allow-long-window", action="store_true")
    parser.add_argument("--mt5-visual", action="store_true")
    parser.add_argument("--keep-arm-checkpoints", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    span_days = _validate_bounded_window(
        args.from_date,
        args.to_date,
        max_span_days=max(1, int(args.max_span_days)),
        allow_long_window=bool(args.allow_long_window),
    )
    if int(args.max_wall_sec) <= 0:
        raise ValueError("fail-fast runner requires --max-wall-sec > 0")
    if int(args.max_decision_packets) <= 0:
        raise ValueError("paired runner requires --max-decision-packets > 0")

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"architecture ablation output must be new: {output}")
    source = args.checkpoint.resolve()
    source_hash = _sha256(source)
    validation_evidence = _require_schema10_seed(source, source_hash)
    output.mkdir(parents=True)
    seeds, intervention = create_intervention_seeds(
        source, args.config.resolve(), output / "arm_seeds"
    )
    base_profile = yaml.safe_load(args.profile.read_text(encoding="utf-8")) or {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.time(),
        "contract": {
            "arms": list(seeds),
            "architecture_ablation": True,
            "sequential": True,
            "policy_learning": "frozen",
            "same_dates": [args.from_date, args.to_date],
            "span_days": int(span_days),
            "fail_fast_max_span_days": int(args.max_span_days),
            "fail_fast_max_wall_sec_per_arm": int(args.max_wall_sec),
            "same_max_decision_packets": int(args.max_decision_packets),
            "same_model": int(args.model),
            "same_seed": int(args.seed),
            "single_changed_network_state_key": TEMPORAL_SCALE_KEY,
            "candidate_only": True,
            "promotable": False,
        },
        "intervention": intervention,
        "schema10_seed_preflight": validation_evidence,
        "arms": {},
        "prepared_only": bool(args.prepare_only),
    }
    summary_path = output / "transformer_ablation_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prepare_only:
        print(json.dumps({"event": "architecture_ablation_prepared", "summary": str(summary_path), **intervention}, ensure_ascii=False, indent=2))
        return 0

    for index, (arm, arm_seed) in enumerate(seeds.items(), start=1):
        arm_dir = output / arm
        profile_path = output / f"profile_{arm}.yaml"
        profile_path.write_text(
            yaml.safe_dump(_arm_profile(base_profile, arm), sort_keys=False),
            encoding="utf-8",
        )
        command = [
            sys.executable, str(CANDIDATE_SCRIPT), "--mode", "mt5-smoke",
            "--profile", str(profile_path), "--config", str(args.config.resolve()),
            "--seed-artifact", str(arm_seed),
            "--runner-source", str(args.runner_dir.resolve()),
            "--output-dir", str(arm_dir), "--from-date", args.from_date,
            "--to-date", args.to_date, "--model", str(args.model),
            "--seed", str(args.seed), "--timeout-sec", str(args.timeout_sec),
            "--max-wall-sec", str(args.max_wall_sec),
            "--max-decision-packets", str(args.max_decision_packets),
        ]
        if args.mt5_visual:
            command.append("--mt5-visual")
        print(json.dumps({
            "event": "architecture_ablation_arm_started",
            "arm": arm, "index": index, "total": len(seeds),
        }, ensure_ascii=False), flush=True)
        try:
            completed = subprocess.run(command, cwd=ROOT, check=False)
        except KeyboardInterrupt:
            report["interrupted"] = True
            report["interrupted_arm"] = arm
            report["complete"] = False
            report["source_checkpoint_unchanged"] = _sha256(source) == source_hash
            summary_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(json.dumps({
                "event": "architecture_ablation_interrupted",
                "arm": arm,
                "summary": str(summary_path),
            }, ensure_ascii=False), flush=True)
            return 130
        diagnostics_path = arm_dir / "mt5_diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8")) if diagnostics_path.is_file() else {}
        manifest = _load_arm_manifest(arm_dir)
        arm_complete, completion_reason = _arm_completion(completed.returncode, manifest)
        mt5 = dict(manifest.get("mt5", {}) or {})
        row = {
            "exit_code": int(completed.returncode),
            "complete": bool(arm_complete),
            "completion_reason": completion_reason,
            "seed_checkpoint": str(arm_seed),
            "seed_checkpoint_sha256": _sha256(arm_seed),
            "diagnostics": diagnostics,
            "mt5_metrics": dict(mt5.get("metrics", {}) or {}),
            "mt5_interrupted": bool(mt5.get("interrupted", False)),
            "manifest": str(arm_dir / "manifest.json"),
        }
        if not args.keep_arm_checkpoints:
            row["discarded_runtime_checkpoints"] = _discard_arm_checkpoints(arm_dir)
        report["arms"][arm] = row
        report["source_checkpoint_unchanged"] = _sha256(source) == source_hash
        summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        if not arm_complete:
            report["failure_arm"] = arm
            report["failure_reason"] = completion_reason
            break

    on = dict(dict(report["arms"].get("v3_temporal_on", {}) or {}).get("mt5_metrics", {}) or {})
    off = dict(dict(report["arms"].get("v2_temporal_off", {}) or {}).get("mt5_metrics", {}) or {})
    report["v3_temporal_on_minus_v2_temporal_off"] = {
        key: _difference(_metric(on, metric), _metric(off, metric))
        for key, metric in (
            ("net_profit", "net_profit"),
            ("profit_factor", "profit_factor"),
            ("expectancy", "execution_summary.expectancy"),
            ("win_rate", "win_rate"),
            ("equity_drawdown_pct", "equity_drawdown_pct"),
            ("total_trades", "total_trades"),
        )
    }
    report["complete"] = len(report["arms"]) == 2 and all(
        bool(row.get("complete", False)) for row in report["arms"].values()
    )
    report["paired_oos_gate"] = (
        _paired_oos_gate(
            on, off, minimum_lifecycles=max(1, int(args.minimum_policy_lifecycles))
        )
        if report["complete"] else {
            "passed": False,
            "reason": "paired_arms_incomplete",
        }
    )
    report["source_checkpoint_unchanged"] = _sha256(source) == source_hash
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "architecture_ablation_complete",
        "complete": report["complete"],
        "summary": str(summary_path),
        "v3_temporal_on_minus_v2_temporal_off": report[
            "v3_temporal_on_minus_v2_temporal_off"
        ],
        "paired_oos_gate": report["paired_oos_gate"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["complete"] and report["paired_oos_gate"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
