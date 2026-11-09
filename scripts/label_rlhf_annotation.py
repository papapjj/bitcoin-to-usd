"""Convert one explicit annotation-task decision into RLHF reward JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_growing_trader.rlhf_annotation import (
    build_rlhf_reward_pair_from_task,
    load_rlhf_annotation_tasks,
)
from self_growing_trader.rlhf_feedback import append_rlhf_feedback_row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=Path)
    parser.add_argument("--task-id", default="")
    parser.add_argument("--chosen-trace-id", required=True)
    parser.add_argument("--rejected-trace-id", required=True)
    parser.add_argument("--label-source", choices=("human", "expert"), required=True)
    parser.add_argument("--rater-id", required=True)
    parser.add_argument("--rater-version", required=True)
    parser.add_argument("--confidence", required=True, type=float)
    parser.add_argument("--split", default="train")
    parser.add_argument("--reason-code", action="append", default=[])
    parser.add_argument("--horizon-bars", type=int, default=48)
    parser.add_argument("--feedback-time", default=None)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    tasks = load_rlhf_annotation_tasks(args.task, task_id=args.task_id)
    if len(tasks) != 1:
        raise SystemExit("label one task at a time; pass --task-id when input contains multiple tasks")
    pair = build_rlhf_reward_pair_from_task(
        tasks[0],
        chosen_trace_id=args.chosen_trace_id,
        rejected_trace_id=args.rejected_trace_id,
        label_source=args.label_source,
        rater_id=args.rater_id,
        rater_version=args.rater_version,
        confidence=args.confidence,
        split=args.split,
        reason_codes=args.reason_code,
        horizon_bars=args.horizon_bars,
        feedback_time=args.feedback_time,
    )
    receipt = append_rlhf_feedback_row(
        args.output,
        pair.to_row(),
        reward_pair=True,
        require_human_feedback=True,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
