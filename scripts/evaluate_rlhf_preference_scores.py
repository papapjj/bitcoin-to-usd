"""Evaluate frozen-checkpoint scores against a causal RLHF pair dataset.

The score file is deliberately separate from the pair dataset.  It must be
produced by a frozen checkpoint runner, and this script only joins by the
immutable pair_id; it never turns an offline metric into an MT5 promotion.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from self_growing_trader.rlhf_preference import (
    load_rlhf_preference_jsonl,
    pairwise_preference_metrics,
    rlhf_dataset_gate,
    rlhf_preference_gate,
    validate_rlhf_dataset,
)


def _load_score_rows(path: Path) -> tuple[dict[str, tuple[float, float, str]], tuple[int, ...]]:
    scores: dict[str, tuple[float, float, str]] = {}
    invalid: list[int] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
                pair_id = str(row.get("pair_id", "") or "").strip()
                chosen = float(row.get("chosen_score"))
                rejected = float(row.get("rejected_score"))
                split = str(row.get("split", "") or "").strip().lower()
            except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
                invalid.append(line_number)
                continue
            if (
                not pair_id
                or not np.isfinite(chosen)
                or not np.isfinite(rejected)
                or pair_id in scores
            ):
                invalid.append(line_number)
                continue
            scores[pair_id] = (chosen, rejected, split)
    return scores, tuple(invalid)


def evaluate(
    pairs_path: Path,
    scores_path: Path,
    *,
    min_pairs: int = 32,
    min_raters: int = 2,
    require_human_feedback: bool = False,
    require_inter_rater_groups: bool = False,
    min_inter_rater_agreement: float | None = None,
    min_weighted_accuracy: float = 0.55,
    max_log_loss: float = 0.69,
    checkpoint_sha256: str = "",
) -> dict[str, object]:
    pairs, invalid_pair_lines = load_rlhf_preference_jsonl(pairs_path)
    dataset_summary = validate_rlhf_dataset(pairs)
    dataset_gate = rlhf_dataset_gate(
        dataset_summary,
        min_pairs=min_pairs,
        min_raters=min_raters,
        require_human_feedback=require_human_feedback,
        require_inter_rater_groups=require_inter_rater_groups,
        min_inter_rater_agreement=min_inter_rater_agreement,
    )
    scores, invalid_score_lines = _load_score_rows(scores_path)
    missing_pairs = [pair.pair_id for pair in pairs if pair.pair_id not in scores]
    pair_ids = {pair.pair_id for pair in pairs}
    unknown_score_pairs = sorted(pair_id for pair_id in scores if pair_id not in pair_ids)
    score_split_mismatches = sorted(
        pair.pair_id
        for pair in pairs
        if pair.pair_id in scores
        and scores[pair.pair_id][2]
        and scores[pair.pair_id][2] != pair.split
    )
    split_metrics: dict[str, object] = {}
    for split in sorted({pair.split for pair in pairs}):
        selected = [pair for pair in pairs if pair.split == split and pair.pair_id in scores]
        chosen_scores = [scores[pair.pair_id][0] for pair in selected]
        rejected_scores = [scores[pair.pair_id][1] for pair in selected]
        split_metrics[split] = pairwise_preference_metrics(
            selected, chosen_scores, rejected_scores
        )
    test_metrics = dict(split_metrics.get("test", {}))
    preference_gate = rlhf_preference_gate(
        test_metrics,
        min_pairs=min_pairs,
        min_weighted_accuracy=min_weighted_accuracy,
        max_log_loss=max_log_loss,
    ) if test_metrics else {
        "passed": False,
        "reasons": ["missing_test_metrics"],
        "promotion_allowed": False,
        "evidence_class": "offline_preference_only",
        "requires_mt5_frozen_oos_lcb": True,
    }
    gate_reasons = list(preference_gate.get("reasons", ()) or ())
    if invalid_pair_lines:
        gate_reasons.append("invalid_pair_lines")
    if invalid_score_lines:
        gate_reasons.append("invalid_score_lines")
    if missing_pairs:
        gate_reasons.append("missing_score_pairs")
    if unknown_score_pairs:
        gate_reasons.append("unknown_score_pairs")
    if score_split_mismatches:
        gate_reasons.append("score_split_mismatch")
    if gate_reasons:
        preference_gate["passed"] = False
        preference_gate["reasons"] = sorted(set(gate_reasons))
    return {
        "pairs_path": str(pairs_path),
        "scores_path": str(scores_path),
        "checkpoint_sha256": str(checkpoint_sha256 or ""),
        "invalid_pair_lines": list(invalid_pair_lines),
        "invalid_score_lines": list(invalid_score_lines),
        "missing_score_pair_ids": missing_pairs,
        "unknown_score_pair_ids": unknown_score_pairs,
        "score_split_mismatch_pair_ids": score_split_mismatches,
        "dataset": dataset_summary,
        "dataset_gate": dataset_gate,
        "split_metrics": split_metrics,
        "preference_gate": preference_gate,
        "promotion_allowed": False,
        "evidence_class": "offline_preference_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--min-raters", type=int, default=2)
    parser.add_argument("--require-human-feedback", action="store_true")
    parser.add_argument("--require-inter-rater-groups", action="store_true")
    parser.add_argument("--min-inter-rater-agreement", type=float)
    parser.add_argument("--min-weighted-accuracy", type=float, default=0.55)
    parser.add_argument("--max-log-loss", type=float, default=0.69)
    args = parser.parse_args()
    report = evaluate(
        args.pairs,
        args.scores,
        min_pairs=args.min_pairs,
        min_raters=args.min_raters,
        require_human_feedback=args.require_human_feedback,
        require_inter_rater_groups=args.require_inter_rater_groups,
        min_inter_rater_agreement=args.min_inter_rater_agreement,
        min_weighted_accuracy=args.min_weighted_accuracy,
        max_log_loss=args.max_log_loss,
        checkpoint_sha256=args.checkpoint_sha256,
    )
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "dataset_ready": report["dataset_gate"]["ready_for_training"],
        "test_gate_passed": report["preference_gate"]["passed"],
        "promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
    return 0 if report["dataset_gate"]["ready_for_training"] and report["preference_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
