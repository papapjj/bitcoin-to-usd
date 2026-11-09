"""Validate and append one human/expert RLHF feedback JSON object."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_growing_trader.rlhf_feedback import append_rlhf_feedback_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--row", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--preference-only", action="store_true")
    args = parser.parse_args()
    row = json.loads(args.row.read_text(encoding="utf-8"))
    if not isinstance(row, dict):
        raise SystemExit("--row must contain one JSON object")
    receipt = append_rlhf_feedback_row(
        args.output,
        row,
        reward_pair=not args.preference_only,
        require_human_feedback=True,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

