#!/usr/bin/env python3
"""Verify the RLHF evidence contract of a champion-shadow MT5 summary.

This is an audit tool, not a promotion tool.  It checks that an MT5 result
actually loaded the expected RLAIF reward model, used a frozen reference rooted
at the immutable champion, and reported update/failure counters.  It never
turns an incomplete run or a failed OOS matrix into a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _resolve(root: Path, value: object) -> Path:
    path = Path(str(value or ""))
    return path if path.is_absolute() else root / path


def verify_summary(
    summary_path: str | Path,
    *,
    expected_feedback_mode: str = "rlaif",
    min_updates: int = 1,
    require_complete: bool = False,
) -> dict[str, object]:
    """Return a fail-closed RLHF evidence report for one matrix summary."""

    summary_file = Path(summary_path).resolve()
    root = summary_file.parent
    for candidate_root in summary_file.parents:
        if (candidate_root / "outputs").is_dir():
            root = candidate_root
            break
    payload = json.loads(summary_file.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("MT5 summary must be a JSON object")

    expected_mode = str(expected_feedback_mode).strip().lower()
    if expected_mode not in {"human", "rlaif"}:
        raise ValueError("expected_feedback_mode must be human or rlaif")
    if int(min_updates) < 0:
        raise ValueError("min_updates must be non-negative")

    champion_hash = str(payload.get("champion_sha256", "") or "").strip().lower()
    top_reasons: list[str] = []
    storage_preflight = _mapping(payload.get("storage_preflight"))
    storage_preflight_valid = True
    if storage_preflight:
        if storage_preflight.get("passed") is not True:
            storage_preflight_valid = False
            top_reasons.append("storage_preflight_not_passed")
        if storage_preflight.get("enabled") is True:
            try:
                free_bytes = int(storage_preflight.get("free_bytes", 0) or 0)
                required_bytes = int(storage_preflight.get("required_free_bytes", 0) or 0)
            except (TypeError, ValueError):
                free_bytes, required_bytes = -1, 0
            if free_bytes < required_bytes:
                storage_preflight_valid = False
                top_reasons.append("storage_preflight_below_floor")
    if payload.get("champion_immutable_verified") is not True:
        top_reasons.append("champion_immutable_not_verified")
    if not champion_hash:
        top_reasons.append("champion_sha256_missing")
    champion_evidence_valid = not top_reasons

    candidate_reports: list[dict[str, object]] = []
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        top_reasons.append("candidate_records_missing")
        candidates = []

    for index, candidate in enumerate(candidates):
        candidate_map = _mapping(candidate)
        train_result = _mapping(candidate_map.get("train_result"))
        metrics = _mapping(train_result.get("metrics"))
        phase = _mapping(metrics.get("rlhf_policy_phase"))
        reasons: list[str] = []
        if phase.get("enabled") is not True:
            reasons.append("rlhf_phase_not_enabled")
        if str(phase.get("feedback_mode", "") or "").strip().lower() != expected_mode:
            reasons.append("feedback_mode_mismatch")
        source_hash = str(phase.get("source_checkpoint_sha256", "") or "").strip().lower()
        if not source_hash:
            reasons.append("source_checkpoint_hash_missing")
        elif champion_hash and source_hash != champion_hash:
            reasons.append("source_checkpoint_hash_mismatch")
        if phase.get("reference_policy_frozen") is not True:
            reasons.append("reference_not_frozen")
        if not str(phase.get("reference_policy_sha256", "") or "").strip():
            reasons.append("reference_policy_hash_missing")
        teacher_hash = str(
            phase.get("teacher_checkpoint_sha256", "") or ""
        ).strip().lower()
        independent_teacher_required = bool(
            phase.get("independent_teacher_required", False)
        )
        if independent_teacher_required:
            if not teacher_hash:
                reasons.append("independent_teacher_hash_missing")
            elif teacher_hash == source_hash:
                reasons.append("independent_teacher_matches_source")
        try:
            updates = int(phase.get("updates", 0) or 0)
            failures = int(phase.get("failures", 0) or 0)
        except (TypeError, ValueError):
            updates, failures = -1, -1
            reasons.append("rlhf_counters_invalid")
        if updates < int(min_updates):
            reasons.append(f"updates_below_floor:{updates}<{int(min_updates)}")
        if failures != 0:
            reasons.append(f"rlhf_failures:{failures}")
        if phase.get("promotion_allowed") is not False:
            reasons.append("rlhf_phase_promotion_flag_not_false")
        # New candidate profiles may explicitly request removal of FiveAdam
        # moments after MT5 closes.  Validate that the persisted manifest
        # proves the operation happened, while keeping older summaries valid
        # when they predate this optional field.
        strip_requested = phase.get("optimizer_state_strip_requested")
        if strip_requested is True:
            if phase.get("optimizer_state_stripped") is not True:
                reasons.append("optimizer_state_strip_not_confirmed")
            try:
                stripped_count = int(phase.get("optimizer_state_stripped_count", 0) or 0)
            except (TypeError, ValueError):
                stripped_count = 0
            if stripped_count < 1:
                reasons.append("optimizer_state_strip_count_missing")
        elif strip_requested not in (None, False):
            reasons.append("optimizer_state_strip_requested_invalid")

        reward_model_path = _resolve(root, phase.get("reward_model_path"))
        reward_model_report: Mapping[str, object] = {}
        if not str(phase.get("reward_model_path", "") or "").strip():
            reasons.append("reward_model_path_missing")
        elif not reward_model_path.is_file():
            reasons.append("reward_model_file_missing")
        else:
            try:
                loaded = json.loads(reward_model_path.read_text(encoding="utf-8"))
                reward_model_report = _mapping(loaded)
            except (OSError, json.JSONDecodeError):
                reasons.append("reward_model_report_invalid")
            report_mode = str(reward_model_report.get("feedback_mode", "") or "").strip().lower()
            if report_mode != expected_mode:
                reasons.append("reward_model_feedback_mode_mismatch")
            if _mapping(reward_model_report.get("test_gate")).get("passed") is not True:
                reasons.append("reward_model_test_gate_not_passed")
            report_model_hash = str(_mapping(reward_model_report.get("model")).get("sha256", "") or "")
            phase_model_hash = str(phase.get("reward_model_sha256", "") or "")
            if not phase_model_hash or not report_model_hash or phase_model_hash != report_model_hash:
                reasons.append("reward_model_hash_mismatch")
            report_teacher = _mapping(reward_model_report.get("teacher_provenance"))
            report_teacher_hash = str(
                report_teacher.get("teacher_checkpoint_sha256", "") or ""
            ).strip().lower()
            if independent_teacher_required and not report_teacher_hash:
                reasons.append("reward_model_teacher_provenance_missing")
            elif teacher_hash and report_teacher_hash != teacher_hash:
                reasons.append("reward_model_teacher_hash_mismatch")

        incomplete = bool(
            candidate_map.get("interrupted")
            or train_result.get("interrupted")
            or metrics.get("incomplete")
        )
        if require_complete and incomplete:
            reasons.append("candidate_incomplete")
        candidate_reports.append({
            "index": index,
            "seed": candidate_map.get("seed"),
            "candidate_sha256": str(candidate_map.get("candidate_sha256", "") or ""),
            "incomplete": incomplete,
            "updates": updates,
            "failures": failures,
            "evidence_valid": not reasons,
            "reasons": reasons,
            "phase": dict(phase),
        })

    valid_candidates = bool(candidate_reports) and all(
        bool(row["evidence_valid"]) for row in candidate_reports
    )
    if not valid_candidates:
        top_reasons.append("candidate_rlhf_evidence_invalid")
    matrix_gate = _mapping(payload.get("matrix_gate"))
    matrix_eligible = matrix_gate.get("eligible") is True
    if not matrix_eligible:
        top_reasons.append("matrix_gate_not_eligible")
    complete = bool(valid_candidates and all(not bool(row["incomplete"]) for row in candidate_reports))
    promotion_ready = bool(
        valid_candidates
        and complete
        and matrix_eligible
        and storage_preflight_valid
        and payload.get("promoted") is True
    )
    if not promotion_ready:
        top_reasons.append("promotion_not_allowed")

    return {
        "schema_version": 1,
        "summary": str(summary_file),
        "expected_feedback_mode": expected_mode,
        "champion_sha256": champion_hash,
        "storage_preflight": dict(storage_preflight),
        "evidence_valid": bool(
            champion_evidence_valid and valid_candidates and storage_preflight_valid
        ),
        "complete": complete,
        "matrix_gate_eligible": matrix_eligible,
        "promotion_ready": promotion_ready,
        "promotion_allowed": False,
        "reasons": sorted(set(top_reasons)),
        "candidates": candidate_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--expected-feedback-mode", default="rlaif")
    parser.add_argument("--min-updates", type=int, default=1)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail the verification when any candidate is incomplete.",
    )
    args = parser.parse_args(argv)
    try:
        report = verify_summary(
            args.summary,
            expected_feedback_mode=args.expected_feedback_mode,
            min_updates=args.min_updates,
            require_complete=args.require_complete,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"evidence_valid": False, "error": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if bool(report["evidence_valid"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
