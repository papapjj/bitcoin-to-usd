#!/usr/bin/env python3
"""Build candidate-only RLAIF reward pairs from explicit teacher scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.rlhf_annotation import load_rlhf_annotation_tasks
from self_growing_trader.rlhf_rlaif import (
    build_rlaif_reward_pairs,
    load_rlaif_teacher_scores,
    write_rlaif_reward_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--teacher-scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-score-margin", type=float, default=0.0)
    args = parser.parse_args()

    tasks = load_rlhf_annotation_tasks(args.tasks)
    scores, invalid_lines = load_rlaif_teacher_scores(args.teacher_scores)
    if invalid_lines:
        raise ValueError(f"invalid RLAIF teacher score lines: {list(invalid_lines)}")
    pairs, stats = build_rlaif_reward_pairs(
        tasks,
        scores,
        minimum_score_margin=args.minimum_score_margin,
    )
    manifest = write_rlaif_reward_dataset(
        args.output,
        pairs,
        tasks_path=args.tasks,
        scores_path=args.teacher_scores,
        stats=stats,
    )
    print(json.dumps({
        "output": str(args.output),
        "pair_count": len(pairs),
        "manifest": manifest,
        "promotion_allowed": False,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
