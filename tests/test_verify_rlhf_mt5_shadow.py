from __future__ import annotations

import json
from pathlib import Path

from scripts.verify_rlhf_mt5_shadow import verify_summary
from scripts.audit_rlhf_storage import classify_rlhf_artifact


_TEST_ROOT = Path("outputs/rlhf_verify_mt5_shadow_test")
_TEST_FILES = (
    _TEST_ROOT / "reward_model.json",
    _TEST_ROOT / "summary.json",
)


def _cleanup_test_files() -> None:
    for path in _TEST_FILES:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    try:
        _TEST_ROOT.rmdir()
    except OSError:
        pass


def _write_summary(
    *,
    strip_requested: object = None,
    stripped: object = None,
    count: int = 0,
    storage_preflight: dict[str, object] | None = None,
) -> Path:
    _TEST_ROOT.mkdir(parents=True, exist_ok=True)
    report = _TEST_ROOT / "reward_model.json"
    report.write_text(json.dumps({
        "feedback_mode": "rlaif",
        "test_gate": {"passed": True},
        "model": {"sha256": "reward-hash"},
        "teacher_provenance": {
            "teacher_checkpoint_sha256": "teacher-hash",
        },
    }), encoding="utf-8")
    phase = {
        "enabled": True,
        "feedback_mode": "rlaif",
        "source_checkpoint_sha256": "champion-hash",
        "reference_policy_frozen": True,
        "reference_policy_sha256": "reference-hash",
        "teacher_checkpoint_sha256": "teacher-hash",
        "independent_teacher_required": True,
        "updates": 2,
        "failures": 0,
        "promotion_allowed": False,
        "reward_model_path": str(report),
        "reward_model_sha256": "reward-hash",
    }
    if strip_requested is not None:
        phase["optimizer_state_strip_requested"] = strip_requested
    if stripped is not None:
        phase["optimizer_state_stripped"] = stripped
    if count:
        phase["optimizer_state_stripped_count"] = count
    summary = _TEST_ROOT / "summary.json"
    payload = {
        "champion_sha256": "champion-hash",
        "champion_immutable_verified": True,
        "promoted": True,
        "matrix_gate": {"eligible": True},
        "candidates": [{
            "seed": 1,
            "candidate_sha256": "candidate-hash",
            "interrupted": False,
            "train_result": {
                "interrupted": False,
                "metrics": {"rlhf_policy_phase": phase},
            },
        }],
    }
    if storage_preflight is not None:
        payload["storage_preflight"] = storage_preflight
    summary.write_text(json.dumps(payload), encoding="utf-8")
    return summary


def test_verifier_keeps_legacy_summary_compatible() -> None:
    try:
        report = verify_summary(_write_summary())
        assert report["evidence_valid"] is True
    finally:
        _cleanup_test_files()


def test_verifier_requires_proof_when_optimizer_strip_is_requested() -> None:
    try:
        summary = _write_summary(
            strip_requested=True,
            stripped=False,
            count=0,
        )
        report = verify_summary(summary)
        assert report["evidence_valid"] is False
        reasons = report["candidates"][0]["reasons"]
        assert "optimizer_state_strip_not_confirmed" in reasons
        assert "optimizer_state_strip_count_missing" in reasons
    finally:
        _cleanup_test_files()


def test_verifier_accepts_confirmed_optimizer_strip() -> None:
    try:
        report = verify_summary(_write_summary(
            strip_requested=True,
            stripped=True,
            count=1,
        ))
        assert report["evidence_valid"] is True
    finally:
        _cleanup_test_files()


def test_verifier_rejects_failed_storage_preflight() -> None:
    try:
        report = verify_summary(_write_summary(storage_preflight={
            "enabled": True,
            "passed": False,
            "free_bytes": 10,
            "required_free_bytes": 20,
        }))
        assert report["evidence_valid"] is False
        assert "storage_preflight_not_passed" in report["reasons"]
        assert "storage_preflight_below_floor" in report["reasons"]
    finally:
        _cleanup_test_files()


def test_rlhf_storage_classifier_keeps_forensic_files_review_only() -> None:
    assert classify_rlhf_artifact("seed/shadow.incomplete.pkl") == "incomplete_checkpoint_for_review"
    assert classify_rlhf_artifact("seed/execution_trace_observatory.jsonl") == "observatory_for_review"
    assert classify_rlhf_artifact("seed.runtime/bridge.csv") == "runtime_transport"
    assert classify_rlhf_artifact("seed/summary.json") == "matrix_manifest"
    assert classify_rlhf_artifact("seed/reward_model.json") == "evidence_or_report"
