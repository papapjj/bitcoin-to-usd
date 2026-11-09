"""Static integrity checks for V9 learner recovery / live telemetry.

This deliberately proves source-level contract only.  It cannot prove MQL5
compilation, tester tick ordering or broker execution.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "src/self_growing_trader/learner_recovery.py": [
        "class ActionRecoveryLearner",
        "HOLD_SHADOW_EVIDENCE_BUILDING",
        "REPLACED_HOLD_WITH_VERIFIED_ACTION",
        "negative SELL record does not lower BUY",
    ],
    "src/self_growing_trader/config.py": [
        "action_recovery_min_support",
        "action_recovery_shadow_weight",
        "action_recovery_max_cells_per_action",
    ],
    "src/self_growing_trader/self_growing_runner.py": [
        "_v9_policy_act",
        "observe_shadow_transition",
        "learner_recovery_global_hold_penalty",
        "emit_bridge_heartbeat",
    ],
    "src/self_growing_trader/mt5_runner.py": [
        "mt5_bridge_interval",
        "stage=\"request_seen\"",
        "stage=\"action_written\"",
    ],
    "src/self_growing_trader/live_observatory.py": [
        "LIVE_SCHEMA_VERSION = 4",
        "bridge_heartbeat",
        "last_event_at",
        "learner_recovery",
    ],
    "src/self_growing_trader/desktop_app.py": [
        "LIVE DATA STALE",
        "hold_reason",
        "bridge_sequence",
    ],
}


def main() -> int:
    failed: dict[str, list[str]] = {}
    for relative, markers in REQUIRED.items():
        path = ROOT / relative
        if not path.exists():
            failed[relative] = ["missing_file"]
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        missing = [marker for marker in markers if marker not in text]
        if missing:
            failed[relative] = missing
    payload = {
        "ok": not failed,
        "schema": 9,
        "checked_files": len(REQUIRED),
        "failed": failed,
        "note": "static source check only; compile MetaEditor and run real-tick MT5 smoke separately.",
    }
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
