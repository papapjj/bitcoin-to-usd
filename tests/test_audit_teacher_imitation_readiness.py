from __future__ import annotations

from pathlib import Path

from scripts.audit_teacher_imitation_readiness import audit_readiness


def test_readiness_audit_reports_missing_teacher_dataset_without_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "seed.pkl"
    checkpoint.write_bytes(b"seed")
    dataset = tmp_path / "missing.npz"

    result = audit_readiness(checkpoint, dataset)

    assert result["ready_for_action_imitation"] is False
    assert result["deployable"] is False
    assert result["gates"]["teacher_dataset"]["reason"] == "dataset_missing"
    assert not dataset.exists()
    assert not checkpoint.with_suffix(".pkl.imitation.json").exists()
