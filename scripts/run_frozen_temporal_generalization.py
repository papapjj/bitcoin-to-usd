#!/usr/bin/env python3
"""Evaluate one context-expert checkpoint on isolated, frozen temporal OOS windows.

The checkpoint is loaded read-only and evaluated with learning disabled.  Each
window uses a portable MT5 clone, so a currently open MT5 terminal and its
Strategy Tester session cannot be reused or interrupted.  The script keeps the
report, execution ledger, bridge, and feature-audit artifacts for every
window; it never promotes, overwrites, or serializes the candidate checkpoint.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import os
import re
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.chuyen_gia_policy import (  # noqa: E402
    evaluate_policy_mt5_oos,
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.mt5_runner import MT5BacktestConfig  # noqa: E402


SUPPORTED_MT5_TESTER_MODELS = (0, 1, 2, 4)
DEFAULT_MT5_TESTER_MODEL = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_windows(raw: str) -> list[tuple[str, str]]:
    windows: list[tuple[str, str]] = []
    for item in raw.split(","):
        start, separator, end = item.strip().partition(":")
        if not separator or not start or not end:
            raise ValueError(f"window khong hop le: {item!r}; dung YYYY.MM.DD:YYYY.MM.DD")
        windows.append((start, end))
    if not windows:
        raise ValueError("can it nhat mot cua so OOS")
    return windows


def apply_experiment_profile(config: Any, profile_path: Path | None) -> tuple[Any, dict[str, Any]]:
    """Apply the same immutable profile overrides used by candidate training.

    Frozen OOS must use the train-time action/selector contract.  Keeping this
    merge at the runner boundary avoids silently evaluating a candidate with
    the base config's veto and exploration defaults.
    """
    if profile_path is None:
        return config, {}
    resolved = profile_path.resolve()
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Experiment profile root must be a mapping: {resolved}")
    learning = replace(
        config.learning,
        **dict(raw.get("learning_overrides", {}) or {}),
    )
    expectancy = replace(
        config.expectancy,
        **dict(raw.get("expectancy_overrides", {}) or {}),
    )
    app_overrides = dict(raw.get("app_overrides", {}) or {})
    return replace(
        config,
        learning=learning,
        expectancy=expectancy,
        **app_overrides,
    ), raw


def clone_runner(source: Path, target: Path) -> None:
    """Create an independent portable runner while hard-linking immutable history."""
    target.mkdir(parents=True, exist_ok=True)
    for folder in ("Common/Files", "MQL5/Files", "MQL5/Profiles/Tester", "Reports", "Tester", "logs"):
        (target / folder).mkdir(parents=True, exist_ok=True)
    
    override_dir = source / "runtime_overrides"
    executable_source = source
    runtime_build = ""
    if override_dir.is_dir():
        required = tuple(override_dir / name for name in ("terminal64.exe", "metatester64.exe"))
        missing = [str(path) for path in required if not path.is_file()]
        build_file = override_dir / "build.txt"
        if not missing and build_file.is_file():
            runtime_build = build_file.read_text(encoding="utf-8").strip()
            executable_source = override_dir
            
    for name in ("terminal64.exe", "metatester64.exe", "MetaEditor64.exe"):
        src_bin = executable_source / name if (executable_source / name).is_file() else (source / name)
        if src_bin.is_file():
            try:
                os.link(src_bin, target / name)
            except OSError:
                shutil.copy2(src_bin, target / name)
    if runtime_build and (override_dir / "build.txt").is_file():
        shutil.copy2(override_dir / "build.txt", target / "build.txt")

    experts = target / "MQL5" / "Experts"
    experts.mkdir(parents=True, exist_ok=True)
    mq5_src = source / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    ex5_src = source / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.ex5"
    if mq5_src.exists():
        shutil.copy2(mq5_src, experts / "SelfGrowing_MT5_Trainer_EA.mq5")
    if ex5_src.exists():
        shutil.copy2(ex5_src, experts / "SelfGrowing_MT5_Trainer_EA.ex5")
        now = time.time() + 10.0
        os.utime(experts / "SelfGrowing_MT5_Trainer_EA.ex5", (now, now))

    include_source = source / "MQL5" / "Include"
    include_target = target / "MQL5" / "Include"
    if include_source.is_dir():
        shutil.copytree(include_source, include_target, copy_function=shutil.copy2)

    for root, _dirs, files in os.walk(source / "Bases"):
        relative = Path(root).relative_to(source / "Bases")
        destination = target / "Bases" / relative
        destination.mkdir(parents=True, exist_ok=True)
        for filename in files:
            try:
                os.link(Path(root) / filename, destination / filename)
            except OSError:
                shutil.copy2(Path(root) / filename, destination / filename)

    shutil.copytree(source / "Config", target / "Config")
    terminal_ini = target / "Config" / "terminal.ini"
    text = terminal_ini.read_text(encoding="utf-16")
    for key in ("LastBuildDataPath", "LastBuild", "LastBuildIDE", "LastBuildTester"):
        text = re.sub(rf"{key}=\d+", f"{key}=5833", text)
    terminal_ini.write_text(text, encoding="utf-16")


def copy_artifacts(metrics: dict[str, Any], destination: Path) -> dict[str, str]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    mapping = {
        "report": "report_path",
        "execution_ledger": "execution_ledger_path",
        "bridge": "bridge_path",
        "feature_audit": "feature_audit_path",
    }
    for label, key in mapping.items():
        source_value = metrics.get(key)
        if not source_value:
            continue
        source = Path(str(source_value))
        if not source.is_file():
            continue
        destination_file = destination / source.name
        shutil.copy2(source, destination_file)
        copied[label] = str(destination_file.resolve())
    return copied


def evaluate_window(
    *,
    index: int,
    from_date: str,
    to_date: str,
    candidate: Path,
    config: Any,
    source_runner: Path,
    clone_root: Path,
    artifacts_root: Path,
    timeout_sec: int,
    model: int,
    visual: bool = False,
) -> dict[str, Any]:
    """Evaluate one OOS window inside its own portable MT5 namespace.

    The function intentionally owns the complete mutable lifecycle for one
    window.  That makes it safe for the caller to execute several windows in
    parallel: no worker shares a runner, report directory, bridge file, or
    execution ledger with another worker.
    """
    clone = clone_root / f"window_{index:02d}"
    print(
        json.dumps(
            {"phase": "start", "window": index, "from": from_date, "to": to_date},
            ensure_ascii=False,
        ),
        flush=True,
    )
    try:
        clone_runner(source_runner, clone)
        # The loader/evaluator are allowed to keep per-policy recurrent state,
        # but the config object itself must not become a cross-window mutation
        # channel when workers run concurrently.
        worker_config = deepcopy(config)
        policy = tai_he_chuyen_gia_mot_context_mot_policy(
            candidate, worker_config, seed=int(worker_config.learning.seed)
        )
        result = evaluate_policy_mt5_oos(
            policy,
            worker_config,
            runner_dir=clone,
            timeout_sec=int(timeout_sec),
            visual=bool(visual),
            shutdown_terminal=True,
            from_date=from_date,
            to_date=to_date,
            model=int(model),
        )
        best = dict(result.get("best", {}) or {})
        metrics = dict(best.get("metrics", {}) or {})
        row: dict[str, Any] = {
            "window": index,
            "oos": {"from": from_date, "to": to_date},
            "completed": True,
            "frozen": bool(result.get("frozen", False)),
            "evaluation_only": bool(result.get("evaluation_only", False)),
            "fitness": best.get("fitness"),
            "metrics": metrics,
            "source_policy_state_hash_before": result.get("source_policy_state_hash_before"),
            "source_policy_state_hash_after": result.get("source_policy_state_hash_after"),
            "artifacts": copy_artifacts(metrics, artifacts_root / f"window_{index:02d}"),
            "runner_clone": str(clone),
        }
    except Exception as exc:  # preserve evidence for failed windows too
        row = {
            "window": index,
            "oos": {"from": from_date, "to": to_date},
            "completed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "runner_clone": str(clone),
        }
    print(
        json.dumps({"phase": "finished", **row}, ensure_ascii=False, default=str),
        flush=True,
    )
    return row


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "outputs/he_chuyen_gia/candidates/v182_smoke_online.pkl",
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="candidate experiment profile used during training; its overrides are applied before frozen OOS",
    )
    parser.add_argument("--runner-source", type=Path, default=ROOT / "mt5_v2_runner")
    parser.add_argument(
        "--windows",
        default="2025.01.02:2025.01.03,2025.01.10:2025.01.11,2025.01.20:2025.01.21",
        help="danh sach OOS start:end, cach nhau boi dau phay",
    )
    parser.add_argument(
        "--model",
        type=int,
        choices=SUPPORTED_MT5_TESTER_MODELS,
        default=DEFAULT_MT5_TESTER_MODEL,
        help="MT5 tester model; 4 = Every tick based on real ticks (promotion gate)",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="Evaluation timeout in seconds per window (defaults to 300s in headless mode, 600s in visual mode)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="so cua so OOS doc lap chay dong thoi; moi cua so co MT5 portable clone rieng",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/he_chuyen_gia/analysis/v182_frozen_temporal_oos.json",
    )
    parser.add_argument(
        "--mt5-visual",
        action="store_true",
        help="Launch MT5 Strategy Tester in visual GUI mode rendering charts and trades live",
    )
    return parser


def run_temporal_generalization(
    args: argparse.Namespace | None = None,
    *,
    candidate: Path | str | None = None,
    config: Path | str | Any | None = None,
    profile: Path | str | None = None,
    runner_source: Path | str | None = None,
    windows: str | list[tuple[str, str]] | None = None,
    model: int | None = None,
    timeout_sec: int | None = None,
    max_workers: int | None = None,
    output: Path | str | None = None,
    mt5_visual: bool | None = None,
) -> dict[str, Any]:
    if args is None:
        args = build_parser().parse_args([])
    else:
        args = deepcopy(args)

    if candidate is not None:
        args.candidate = Path(candidate)
    if config is not None:
        args.config = config if not isinstance(config, (str, Path)) else Path(config)
    if profile is not None:
        args.profile = Path(profile)
    if runner_source is not None:
        args.runner_source = Path(runner_source)
    if windows is not None:
        args.windows = windows
    if model is not None:
        args.model = int(model)
    if timeout_sec is not None:
        args.timeout_sec = int(timeout_sec)
    if max_workers is not None:
        args.max_workers = int(max_workers)
    if output is not None:
        args.output = Path(output)
    if mt5_visual is not None:
        args.mt5_visual = bool(mt5_visual)

    is_visual = bool(getattr(args, "mt5_visual", False))

    # Resolve timeout_sec default: 600s for visual mode, 300s for headless mode
    if getattr(args, "timeout_sec", None) is None:
        args.timeout_sec = 600 if is_visual else 300

    # Single-instance visual guard: enforce max_workers = 1 when visual is True
    if is_visual:
        args.max_workers = 1

    candidate_path = Path(args.candidate).resolve()
    source_runner = Path(args.runner_source).resolve()
    if not candidate_path.is_file():
        raise FileNotFoundError(candidate_path)
    if not source_runner.is_dir():
        raise FileNotFoundError(source_runner)

    if isinstance(args.windows, list):
        parsed_windows = args.windows
    else:
        parsed_windows = parse_windows(str(args.windows))

    if int(args.max_workers) < 1:
        raise ValueError("--max-workers phai >= 1")

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clone_root = output_path.parent / f"{output_path.stem}_{int(time.time())}_runners"
    artifacts_root = output_path.parent / f"{output_path.stem}_artifacts"
    clone_root.mkdir(parents=True, exist_ok=True)

    loaded_config = args.config if hasattr(args.config, "learning") else load_config(args.config)
    loaded_config, profile_payload = apply_experiment_profile(loaded_config, args.profile)
    loaded_config = replace(loaded_config, observability=replace(loaded_config.observability, enabled=False))
    candidate_sha_before = sha256_file(candidate_path)

    effective_max_workers = min(int(args.max_workers), len(parsed_windows))
    indexed_windows = [
        (index, from_date, to_date)
        for index, (from_date, to_date) in enumerate(parsed_windows, start=1)
    ]

    if effective_max_workers == 1:
        rows = [
            evaluate_window(
                index=index,
                from_date=from_date,
                to_date=to_date,
                candidate=candidate_path,
                config=loaded_config,
                source_runner=source_runner,
                clone_root=clone_root,
                artifacts_root=artifacts_root,
                timeout_sec=int(args.timeout_sec),
                model=int(args.model),
                visual=is_visual,
            )
            for index, from_date, to_date in indexed_windows
        ]
    else:
        completed_rows: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=effective_max_workers, thread_name_prefix="mt5-oos") as executor:
            futures = {
                executor.submit(
                    evaluate_window,
                    index=index,
                    from_date=from_date,
                    to_date=to_date,
                    candidate=candidate_path,
                    config=loaded_config,
                    source_runner=source_runner,
                    clone_root=clone_root,
                    artifacts_root=artifacts_root,
                    timeout_sec=int(args.timeout_sec),
                    model=int(args.model),
                    visual=is_visual,
                ): index
                for index, from_date, to_date in indexed_windows
            }
            for future in as_completed(futures):
                completed_rows[futures[future]] = future.result()
        rows = [completed_rows[index] for index, _, _ in indexed_windows]

    candidate_sha_after = sha256_file(candidate_path)
    exit_code = 0 if all(bool(row.get("completed")) for row in rows) else 2
    summary = {
        "schema_version": 1,
        "purpose": "frozen_temporal_generalization",
        "candidate": str(candidate_path),
        "candidate_sha256_before": candidate_sha_before,
        "candidate_sha256_after": candidate_sha_after,
        "candidate_unchanged": candidate_sha_before == candidate_sha_after,
        "profile": str(args.profile.resolve()) if args.profile else "",
        "profile_experiment": dict(profile_payload.get("experiment", {}) or {}),
        "model": int(args.model),
        "learning_enabled": False,
        "visual": is_visual,
        "timeout_sec": int(args.timeout_sec),
        "max_workers": effective_max_workers,
        "parallel_windows": bool(effective_max_workers > 1),
        "windows": rows,
        "exit_code": exit_code,
        "note": "Cac cua so la OOS theo thoi gian so voi checkpoint. Model 4 la dieu kien real-tick toi thieu cho promotion; ket qua van khong phai bang chung promotion neu chua co paired baseline, reconciliation, terminal-flat va pooled LCB.",
    }
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"phase": "summary", "output": str(output_path), "candidate_unchanged": summary["candidate_unchanged"]}, ensure_ascii=False), flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_temporal_generalization(args)
    return int(summary.get("exit_code", 0))


if __name__ == "__main__":
    raise SystemExit(main())
