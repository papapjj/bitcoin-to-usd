"""Validate a causal RLHF preference JSONL dataset for the trading policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_growing_trader.rlhf_preference import (
    load_rlhf_preference_jsonl,
    rlhf_dataset_gate,
    validate_rlhf_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-pairs", type=int, default=32)
    parser.add_argument("--min-raters", type=int, default=2)
    parser.add_argument("--require-human-feedback", action="store_true")
    parser.add_argument("--require-inter-rater-groups", action="store_true")
    parser.add_argument("--min-inter-rater-agreement", type=float)
    args = parser.parse_args()
    pairs, invalid_lines = load_rlhf_preference_jsonl(args.input)
    report = validate_rlhf_dataset(pairs)
    report["invalid_lines"] = list(invalid_lines)
    report["input"] = str(args.input)
    dataset_gate = rlhf_dataset_gate(
        report,
        min_pairs=args.min_pairs,
        min_raters=args.min_raters,
        require_human_feedback=args.require_human_feedback,
        require_inter_rater_groups=args.require_inter_rater_groups,
        min_inter_rater_agreement=args.min_inter_rater_agreement,
    )
    if invalid_lines:
        dataset_gate["ready_for_training"] = False
        dataset_gate["reasons"] = sorted({
            *[str(reason) for reason in dataset_gate.get("reasons", ())],
            "invalid_jsonl_lines",
        })
    report["dataset_gate"] = dataset_gate
    report["ready_for_training"] = bool(dataset_gate["ready_for_training"])
    report["promotion_allowed"] = False
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 0 if report["ready_for_training"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
