"""Fit and gate the candidate-only RLHF reward model from JSONL pairs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_growing_trader.rlhf_preference import (
    rlhf_dataset_gate,
    validate_rlhf_dataset,
)
from self_growing_trader.rlhf_reward_model import (
    fit_rlhf_reward_model,
    load_rlhf_reward_jsonl,
    rlhf_reward_model_gate,
    score_rlhf_reward_pairs,
)


def _load_input_manifest(input_path: Path) -> dict[str, object]:
    manifest_path = input_path.with_suffix(input_path.suffix + ".manifest.json")
    if not manifest_path.is_file():
        return {}
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--feedback-mode", choices=("human", "rlaif"), default="human",
        help=(
            "human = human/expert RLHF voi calibration giua rater; "
            "rlaif = teacher/mt5_rater rieng, khong duoc goi la human RLHF."
        ),
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--min-pairs", type=int, default=50)
    parser.add_argument("--min-raters", type=int, default=3)
    parser.add_argument("--min-inter-rater-agreement", type=float, default=0.60)
    parser.add_argument("--min-weighted-accuracy", type=float, default=0.55)
    parser.add_argument("--max-log-loss", type=float, default=0.69)
    args = parser.parse_args()

    rows, invalid_lines = load_rlhf_reward_jsonl(args.input)
    input_manifest = _load_input_manifest(args.input)
    pair_summary = validate_rlhf_dataset([row.pair for row in rows])
    is_human_feedback = args.feedback_mode == "human"
    allowed_sources = {"human", "expert"} if is_human_feedback else {"teacher", "mt5_rater"}
    observed_sources = set(pair_summary.get("source_counts", {}) or {})
    dataset_gate = rlhf_dataset_gate(
        pair_summary,
        min_pairs=args.min_pairs,
        min_raters=(args.min_raters if is_human_feedback else 1),
        require_human_feedback=is_human_feedback,
        require_inter_rater_groups=is_human_feedback,
        min_inter_rater_agreement=(
            args.min_inter_rater_agreement if is_human_feedback else None
        ),
    )
    split_rows = {
        split: tuple(row for row in rows if row.pair.split == split)
        for split in ("train", "validation", "test")
    }
    dataset_reasons = list(pair_summary.get("errors", ()) or ())
    dataset_reasons.extend(dataset_gate.get("reasons", ()) or ())
    if not observed_sources:
        dataset_reasons.append("no_feedback_source")
    elif not observed_sources.issubset(allowed_sources):
        dataset_reasons.append(
            "feedback_mode_source_mismatch:" + ",".join(sorted(observed_sources - allowed_sources))
        )
    if invalid_lines:
        dataset_reasons.append("invalid_jsonl_lines")
    feature_schemas = {row.feature_schema for row in rows}
    feature_counts = {
        len(row.chosen_features)
        for row in rows
    } | {
        len(row.rejected_features)
        for row in rows
    }
    if len(feature_schemas) > 1:
        dataset_reasons.append("mixed_reward_feature_schema")
    if len(feature_counts) > 1:
        dataset_reasons.append("mixed_reward_feature_count")
    if len(split_rows["train"]) < args.min_pairs:
        dataset_reasons.append("insufficient_train_pairs")
    for split in ("validation", "test"):
        if not split_rows[split]:
            dataset_reasons.append(f"missing_split:{split}")

    report: dict[str, object] = {
        "input": str(args.input),
        "feedback_mode": args.feedback_mode,
        "invalid_lines": list(invalid_lines),
        "dataset": pair_summary,
        "dataset_gate": dataset_gate,
        "dataset_reasons": sorted(set(dataset_reasons)),
        "promotion_allowed": False,
        "evidence_class": (
            "offline_reward_model_only"
            if is_human_feedback else "offline_rlaif_teacher_only"
        ),
    }
    if not is_human_feedback:
        report["teacher_provenance"] = dict(
            input_manifest.get("teacher_provenance", {}) or {}
        )
    if not dataset_reasons:
        model, fit_summary = fit_rlhf_reward_model(
            split_rows["train"],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        report["model"] = model.to_row()
        report["fit"] = fit_summary
        split_reports: dict[str, object] = {}
        for split, split_data in split_rows.items():
            metrics = score_rlhf_reward_pairs(model, split_data)
            split_gate = dict(rlhf_reward_model_gate(
                metrics,
                min_pairs=args.min_pairs,
                min_weighted_accuracy=args.min_weighted_accuracy,
                max_log_loss=args.max_log_loss,
            ))
            # Keep the evidence class consistent at every report level.  A
            # teacher-labelled RLAIF report must not look like human/offline
            # feedback merely because the generic numeric gate was reused.
            split_gate["evidence_class"] = report["evidence_class"]
            split_reports[split] = {
                "metrics": metrics,
                "gate": split_gate,
            }
        report["splits"] = split_reports
        test_gate = dict(split_reports["test"]["gate"])
        test_gate["evidence_class"] = report["evidence_class"]
        report["test_gate"] = test_gate
    else:
        report["test_gate"] = {
            "passed": False,
            "reasons": sorted(set(dataset_reasons)),
            "promotion_allowed": False,
            "evidence_class": report["evidence_class"],
            "requires_rlhf_ppo_policy_gate": True,
            "requires_mt5_frozen_oos_lcb": True,
        }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "feedback_mode": args.feedback_mode,
        "reward_model_gate_passed": bool(report["test_gate"]["passed"]),
        "promotion_allowed": False,
    }, ensure_ascii=False, indent=2))
    return 0 if report["test_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
