#!/usr/bin/env python3
"""Train then frozen-OOS-test isolated context-expert capacity candidates.

Each arm starts from the same seed with a different shared Direct-RNN topology.
Train and OOS use separate portable MT5 clones, so neither writes to the live
runner or to an existing candidate.  This is deliberately a capacity-only
experiment: all learning, input, action, regularization, and execution
settings are inherited unchanged from the base configuration.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.chuyen_gia_policy import (  # noqa: E402
    evaluate_policy_mt5_oos,
    huan_luyen_he_chuyen_gia_mt5,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config  # noqa: E402


# Keep the capacity candidates explicit.  The three-layer arm is intentionally
# width-matched to 4x128 so a comparison changes depth only.
DEFAULT_ARMS = {
    "3x128": (3, 128),
    "4x128": (4, 128),
    "4x144": (4, 144),
    "6x128": (6, 128),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_runner(source: Path, target: Path) -> None:
    """Clone mutable MT5 state and hard-link immutable historical data."""
    if target.exists():
        raise FileExistsError(f"runner clone da ton tai: {target}")
    target.mkdir(parents=True)
    for folder in ("Common/Files", "MQL5/Files", "Reports", "Tester", "logs"):
        (target / folder).mkdir(parents=True, exist_ok=True)
    for name in ("terminal64.exe", "metatester64.exe"):
        os.link(source / name, target / name)
    experts = target / "MQL5" / "Experts"
    experts.mkdir(parents=True, exist_ok=True)
    for name in ("SelfGrowing_MT5_Trainer_EA.ex5", "SelfGrowing_MT5_Trainer_EA.mq5"):
        shutil.copy2(source / "MQL5" / "Experts" / name, experts / name)
    for root, _dirs, files in os.walk(source / "Bases"):
        relative = Path(root).relative_to(source / "Bases")
        destination = target / "Bases" / relative
        destination.mkdir(parents=True, exist_ok=True)
        for filename in files:
            os.link(Path(root) / filename, destination / filename)
    shutil.copytree(source / "Config", target / "Config")
    terminal_ini = target / "Config" / "terminal.ini"
    text = terminal_ini.read_text(encoding="utf-16")
    for key in ("LastBuildDataPath", "LastBuild", "LastBuildIDE", "LastBuildTester"):
        text = re.sub(rf"{key}=\d+", f"{key}=5833", text)
    terminal_ini.write_text(text, encoding="utf-16")


def copy_artifacts(metrics: dict[str, Any], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for label, key in {
        "report": "report_path",
        "execution_ledger": "execution_ledger_path",
        "bridge": "bridge_path",
        "feature_audit": "feature_audit_path",
    }.items():
        value = metrics.get(key)
        if not value:
            continue
        source = Path(str(value))
        if source.is_file():
            target = destination / source.name
            shutil.copy2(source, target)
            copied[label] = str(target.resolve())
    return copied


def build_config(base, *, hidden_layers: int, hidden_size: int):
    dropout = tuple(base.learning.hidden_dropout_rates[:hidden_layers])
    normalization = tuple(True for _ in range(hidden_layers))
    return replace(
        base,
        model=replace(base.model, hidden_layers=int(hidden_layers), hidden_size=int(hidden_size)),
        learning=replace(
            base.learning,
            hidden_dropout_enabled=True,
            hidden_dropout_rates=dropout,
            hidden_normalization_enabled=True,
            hidden_normalization_layers=normalization,
        ),
        observability=replace(base.observability, enabled=False),
    )


def parse_arms(raw: str) -> list[tuple[str, int, int]]:
    parsed: list[tuple[str, int, int]] = []
    for token in raw.split(","):
        name = token.strip()
        if name not in DEFAULT_ARMS:
            raise ValueError(f"arm khong hop le: {name}; chon {', '.join(DEFAULT_ARMS)}")
        layers, size = DEFAULT_ARMS[name]
        parsed.append((name, layers, size))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    parser.add_argument("--runner-source", type=Path, default=ROOT / "mt5_v2_runner")
    parser.add_argument("--arms", default="4x128,4x144,6x128")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--train-from", default="2025.01.01")
    parser.add_argument("--train-to", default="2025.01.02")
    parser.add_argument("--oos-from", default="2025.01.02")
    parser.add_argument("--oos-to", default="2025.01.03")
    parser.add_argument("--model", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--timeout-sec", type=int, default=300)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/he_chuyen_gia/experiments/capacity_ablation_v182",
    )
    args = parser.parse_args()

    source_runner = args.runner_source.resolve()
    out = args.output_dir.resolve()
    if not source_runner.is_dir():
        raise FileNotFoundError(source_runner)
    arms = parse_arms(args.arms)
    out.mkdir(parents=True, exist_ok=True)
    base_config = load_config(args.config)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "capacity_ablation_train_then_frozen_oos",
        "base_config": str(args.config.resolve()),
        "seed": int(args.seed),
        "train_window": {"from": args.train_from, "to": args.train_to},
        "oos_window": {"from": args.oos_from, "to": args.oos_to},
        "model": int(args.model),
        "arms": [],
    }
    for arm, layers, width in arms:
        arm_dir = out / arm
        candidate = arm_dir / "trained.pkl"
        config = build_config(base_config, hidden_layers=layers, hidden_size=width)
        train_runner = arm_dir / "mt5_train"
        oos_runner = arm_dir / "mt5_oos"
        row: dict[str, Any] = {
            "arm": arm,
            "hidden_layers": layers,
            "hidden_size": width,
            "dropout_rates": list(config.learning.hidden_dropout_rates),
            "normalization_layers": list(config.learning.hidden_normalization_layers),
            "candidate": str(candidate),
        }
        try:
            clone_runner(source_runner, train_runner)
            print(json.dumps({"phase": "train_start", "arm": arm}, ensure_ascii=False), flush=True)
            train_result = huan_luyen_he_chuyen_gia_mt5(
                config,
                output_path=candidate,
                runner_dir=train_runner,
                timeout_sec=int(args.timeout_sec),
                max_wall_sec=0,
                visual=False,
                from_date=args.train_from,
                to_date=args.train_to,
                backtest_days=1,
                model=int(args.model),
                seed=int(args.seed),
                resume_existing=False,
            )
            row["train"] = train_result
            row["candidate_sha256"] = sha256_file(candidate)
            clone_runner(source_runner, oos_runner)
            trained = tai_he_chuyen_gia_mot_context_mot_policy(candidate, config, seed=int(args.seed))
            print(json.dumps({"phase": "oos_start", "arm": arm}, ensure_ascii=False), flush=True)
            oos_result = evaluate_policy_mt5_oos(
                trained,
                config,
                runner_dir=oos_runner,
                timeout_sec=int(args.timeout_sec),
                visual=False,
                shutdown_terminal=True,
                from_date=args.oos_from,
                to_date=args.oos_to,
                model=int(args.model),
            )
            best = dict(oos_result.get("best", {}) or {})
            metrics = dict(best.get("metrics", {}) or {})
            row["oos"] = {
                "completed": True,
                "frozen": bool(oos_result.get("frozen", False)),
                "fitness": best.get("fitness"),
                "metrics": metrics,
                "source_policy_state_hash_before": oos_result.get("source_policy_state_hash_before"),
                "source_policy_state_hash_after": oos_result.get("source_policy_state_hash_after"),
                "artifacts": copy_artifacts(metrics, arm_dir / "oos_artifacts"),
            }
            row["candidate_sha256_after_oos"] = sha256_file(candidate)
            row["candidate_unchanged_during_oos"] = row["candidate_sha256"] == row["candidate_sha256_after_oos"]
        except Exception as exc:
            row["error_type"] = type(exc).__name__
            row["error"] = str(exc)
        summary["arms"].append(row)
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"phase": "arm_finished", "arm": arm, "error": row.get("error")}, ensure_ascii=False), flush=True)
    print(json.dumps({"phase": "summary", "path": str(out / "summary.json")}, ensure_ascii=False), flush=True)
    return 0 if all("error" not in row for row in summary["arms"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
