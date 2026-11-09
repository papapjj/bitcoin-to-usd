#!/usr/bin/env python3
"""Preflight the graph-MT5 bridge without pretending to compile or trade.

This command is intentionally safe in CI and developer machines without MT5.
It validates that a portable runner contains the files required before a
MetaEditor compile / Strategy Tester invocation can be attempted.  It does not
launch terminal64.exe and it does not claim a successful MT5 smoke test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir", default="mt5_v2_runner_current", help="MT5 portable runner directory")
    parser.add_argument("--expert", default="SelfGrowing_MT5_Trainer_EA", help="Expert basename without .mq5")
    args = parser.parse_args(argv)

    runner = Path(args.runner_dir).expanduser().resolve()
    required = {
        "terminal64.exe": runner / "terminal64.exe",
        "metaeditor64.exe": runner / "metaeditor64.exe",
        f"MQL5/Experts/{args.expert}.mq5": runner / "MQL5" / "Experts" / f"{args.expert}.mq5",
    }
    failed = [name for name, path in required.items() if not path.is_file()]
    payload = {
        "ok": not failed,
        "phase": "preflight",
        "runner_dir": str(runner),
        "failed": failed,
        "note": (
            "Preflight only: no MetaEditor compile and no Strategy Tester run were performed."
            if not failed else
            "Required MT5 portable files are missing; no process was launched."
        ),
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
