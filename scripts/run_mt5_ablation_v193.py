#!/usr/bin/env python3
"""Run matched, sequential MT5 ablations on isolated candidate clones.

Arms share the same immutable seed, dates, MT5 model and random seed.  The
script retains compact telemetry and the comparison report while optionally
discarding the very large per-arm checkpoints after their metrics are secured.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_SCRIPT = ROOT / "scripts" / "run_edge_discovery_v193_candidate.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_for_arm(base: dict[str, Any], arm: str) -> dict[str, Any]:
    profile = copy.deepcopy(base)
    experiment = dict(profile.get("experiment", {}) or {})
    experiment["id"] = f"ablation_{arm}"
    experiment["ablation_arm"] = arm
    profile["experiment"] = experiment
    app = dict(profile.get("app_overrides", {}) or {})
    learning = dict(profile.get("learning_overrides", {}) or {})
    if arm == "frozen_policy":
        app["mt5_learning_enabled"] = False
    elif arm == "full_online":
        # This arm is reached only after _frozen_positive_lcb_gate passes.
        app["mt5_learning_enabled"] = True
    elif arm == "reasoning_off":
        app["mt5_learning_enabled"] = True
        learning["five_reasoning_rl_enabled"] = False
        learning["five_reasoning_encoder_credit_scale"] = 0.0
        learning["five_reasoning_action_scale"] = 0.0
    else:
        raise ValueError(f"unknown ablation arm: {arm}")
    profile["app_overrides"] = app
    profile["learning_overrides"] = learning
    return profile


def _metric(summary: dict[str, Any], section: str, key: str) -> Any:
    return dict(summary.get(section, {}) or {}).get(key)


def _delta(value: Any, baseline: Any) -> float | None:
    try:
        return float(value) - float(baseline)
    except (TypeError, ValueError):
        return None


def _discard_large_checkpoints(arm_dir: Path) -> list[str]:
    removed: list[str] = []
    resolved_root = arm_dir.resolve()
    for path in arm_dir.glob("*.pkl"):
        resolved = path.resolve()
        if resolved.parent != resolved_root:
            raise RuntimeError(f"refusing to remove checkpoint outside arm: {resolved}")
        path.unlink(missing_ok=True)
        removed.append(path.name)
    return removed


def _frozen_positive_lcb_gate(
    diagnostics: dict[str, Any], *, minimum_positions: int = 200
) -> dict[str, Any]:
    performance = dict(diagnostics.get("performance", {}) or {})
    optimizer = dict(diagnostics.get("optimizer", {}) or {})
    positions = int(performance.get("final_position_closes", 0) or 0)
    lower_bound = performance.get("position_expectancy_lcb_95_one_sided")
    try:
        lower_bound_value = float(lower_bound)
    except (TypeError, ValueError):
        lower_bound_value = None
    checks = {
        "minimum_completed_positions": positions >= int(minimum_positions),
        "expectancy_lower_bound_available": lower_bound_value is not None,
        "expectancy_lower_bound_strictly_positive": (
            lower_bound_value is not None and lower_bound_value > 0.0
        ),
        "frozen_run_has_no_ppo_updates": int(optimizer.get("ppo_updates", 0) or 0) == 0,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_positions": int(minimum_positions),
        "observed_positions": positions,
        "position_expectancy_lcb_95_one_sided": lower_bound_value,
        "scope": "same checkpoint, date range, MT5 model and seed as online arms",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Matched MT5 ablation V193")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    parser.add_argument("--model", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument("--max-wall-sec", type=int, default=0)
    parser.add_argument("--mt5-visual", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--min-frozen-positions", type=int, default=200)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"ablation output must be new: {output}")
    output.mkdir(parents=True)
    seed = args.checkpoint.resolve()
    seed_hash = _sha256(seed)
    base_profile = yaml.safe_load(args.profile.read_text(encoding="utf-8")) or {}
    arms = ("frozen_policy", "full_online", "reasoning_off")
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": time.time(),
        "contract": {
            "arms": list(arms),
            "sequential": True,
            "same_checkpoint": str(seed),
            "same_checkpoint_sha256": seed_hash,
            "same_dates": [args.from_date, args.to_date],
            "same_model": args.model,
            "same_seed": args.seed,
            "reasoning_off_scope": "auxiliary reasoning learning disabled; action scale remains zero",
            "architecture_ablation": False,
        },
        "arms": {},
    }
    for index, arm in enumerate(arms, start=1):
        arm_dir = output / arm
        profile_path = output / f"profile_{arm}.yaml"
        profile_path.write_text(
            yaml.safe_dump(_profile_for_arm(base_profile, arm), sort_keys=False), encoding="utf-8"
        )
        command = [
            sys.executable, str(CANDIDATE_SCRIPT), "--mode", "mt5-smoke",
            "--profile", str(profile_path), "--config", str(args.config.resolve()),
            "--seed-artifact", str(seed), "--runner-source", str(args.runner_dir.resolve()),
            "--output-dir", str(arm_dir), "--from-date", args.from_date,
            "--to-date", args.to_date, "--model", str(args.model), "--seed", str(args.seed),
            "--timeout-sec", str(args.timeout_sec), "--max-wall-sec", str(args.max_wall_sec),
            # The seven-choice checkpoint is deliberately research-only.  The
            # frozen arm is the evidence gate; online arms are unreachable
            # until that exact frozen run has a positive lower bound.
            "--allow-unverified-imitation-seed",
        ]
        if args.mt5_visual:
            command.append("--mt5-visual")
        print(json.dumps({
            "event": "ablation_arm_started", "arm": arm, "index": index, "total": len(arms)
        }, ensure_ascii=False), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)
        diagnostics_path = arm_dir / "mt5_diagnostics.json"
        diagnostics = (
            json.loads(diagnostics_path.read_text(encoding="utf-8"))
            if diagnostics_path.is_file() else {}
        )
        row = {
            "exit_code": completed.returncode,
            "diagnostics": diagnostics,
            "source_checkpoint_sha256_after": _sha256(seed),
        }
        if row["source_checkpoint_sha256_after"] != seed_hash:
            raise RuntimeError(f"source checkpoint changed in arm {arm}")
        if not args.keep_checkpoints:
            row["discarded_checkpoints"] = _discard_large_checkpoints(arm_dir)
        report["arms"][arm] = row
        if arm == "frozen_policy":
            frozen_gate = _frozen_positive_lcb_gate(
                diagnostics, minimum_positions=args.min_frozen_positions
            )
            report["ppo_enable_gate"] = frozen_gate
        (output / "ablation_summary.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({
            "event": "ablation_arm_finished", "arm": arm,
            "exit_code": completed.returncode,
            "profit_factor": _metric(diagnostics, "performance", "profit_factor"),
            "expectancy": _metric(diagnostics, "performance", "expectancy"),
            "max_drawdown": _metric(diagnostics, "risk", "max_equity_drawdown_pct"),
            "ppo_updates": _metric(diagnostics, "optimizer", "ppo_updates"),
        }, ensure_ascii=False), flush=True)
        if completed.returncode != 0:
            report["complete"] = False
            report["failure_arm"] = arm
            break
        if arm == "frozen_policy" and not bool(report["ppo_enable_gate"]["passed"]):
            report["complete"] = False
            report["failure_arm"] = "ppo_enable_gate"
            report["blocked_arms"] = ["full_online", "reasoning_off"]
            print(json.dumps({
                "event": "ppo_blocked_by_frozen_lower_bound",
                **dict(report["ppo_enable_gate"]),
            }, ensure_ascii=False), flush=True)
            break

    full = dict(dict(report["arms"].get("full_online", {}) or {}).get("diagnostics", {}) or {})
    comparisons: dict[str, Any] = {}
    for arm, row in dict(report["arms"]).items():
        diagnostics = dict(dict(row or {}).get("diagnostics", {}) or {})
        comparisons[arm] = {
            "delta_profit_factor_vs_full": _delta(
                _metric(diagnostics, "performance", "profit_factor"),
                _metric(full, "performance", "profit_factor"),
            ),
            "delta_expectancy_vs_full": _delta(
                _metric(diagnostics, "performance", "expectancy"),
                _metric(full, "performance", "expectancy"),
            ),
            "delta_max_drawdown_vs_full": _delta(
                _metric(diagnostics, "risk", "max_equity_drawdown_pct"),
                _metric(full, "risk", "max_equity_drawdown_pct"),
            ),
        }
    report["comparisons"] = comparisons
    report["complete"] = len(report["arms"]) == len(arms) and all(
        int(dict(row or {}).get("exit_code", 1)) == 0 for row in report["arms"].values()
    )
    report["source_checkpoint_unchanged"] = _sha256(seed) == seed_hash
    summary_path = output / "ablation_summary.json"
    summary_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "ablation_complete", "complete": report["complete"], "summary": str(summary_path)
    }, ensure_ascii=False), flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
