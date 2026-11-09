#!/usr/bin/env python3
"""Fail-fast checklist for the local MT5 runner expected by this project."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_EXPERT = "SelfGrowing_MT5_Trainer_EA"


def check_mt5_runner(runner_dir: Path, expert: str) -> dict[str, object]:
    root = runner_dir.expanduser().resolve()
    expected = {
        "runner directory": root,
        "terminal64.exe": root / "terminal64.exe",
        "MQL5 directory": root / "MQL5",
        "MQL5/Files": root / "MQL5" / "Files",
        "MQL5/Profiles/Tester": root / "MQL5" / "Profiles" / "Tester",
        "compute_features.mqh": root / "MQL5" / "Include" / "compute_features.mqh",
    }
    checks: list[dict[str, object]] = []
    for label, path in expected.items():
        checks.append({"name": label, "path": str(path), "ok": path.exists()})

    expert_source = root / "MQL5" / "Experts" / f"{expert}.mq5"
    expert_binary = root / "MQL5" / "Experts" / f"{expert}.ex5"
    checks.append(
        {
            "name": "EA source or compiled binary",
            "path": f"{expert_source} OR {expert_binary}",
            "ok": expert_source.exists() or expert_binary.exists(),
        }
    )
    failed = [item for item in checks if not item["ok"]]
    return {
        "ready": not failed,
        "runner_dir": str(root),
        "expert": expert,
        "checks": checks,
        "failed": [item["name"] for item in failed],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the MT5 runner layout used by self-growing-trader.")
    parser.add_argument("--runner-dir", default="mt5_v2_runner")
    parser.add_argument("--expert", default=DEFAULT_EXPERT)
    args = parser.parse_args()
    result = check_mt5_runner(Path(args.runner_dir), args.expert)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
