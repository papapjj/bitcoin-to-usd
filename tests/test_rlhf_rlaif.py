from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from self_growing_trader.rlhf_annotation import build_rlhf_annotation_tasks
from self_growing_trader.rlhf_rlaif import (
    RLAIF_TEACHER_SCHEMA_VERSION,
    build_rlaif_reward_pairs,
    load_rlaif_teacher_scores,
)
from scripts.score_rlaif_tasks_with_frozen_policy import (
    chronological_task_splits,
    score_tasks,
)


def _row(action: int, trace_id: str) -> dict[str, object]:
    return {
        "run_id": "rlaif-run",
        "created_at": 1780000000.0,
        "trace_id": trace_id,
        "request": {
            "event_id": "rlaif-event",
            "sequence": 1,
            "time": "2026.09.22 08:00:00",
        },
        "trace": {
            "raw_obs": np.linspace(-1.0, 1.0, 137).tolist(),
            "selected_idx": action,
            "action_details": {
                "five_agent_index": action - 1,
                "five_action_index": action,
                "five_decision_state_sha256": "state-rlaif",
                "five_risk_geometry_sha256": "geometry-rlaif",
            },
        },
    }


def _score(task_id: str, *, scores: list[float] | None = None) -> dict[str, object]:
    return {
        "schema_version": RLAIF_TEACHER_SCHEMA_VERSION,
        "task_id": task_id,
        "teacher_id": "local-teacher",
        "teacher_version": "teacher-v1",
        "label_source": "teacher",
        "action_scores": scores or [0.0, 0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
        "confidence": 0.9,
        "split": "train",
        "reason_codes": ["direction", "risk"],
        "horizon_bars": 48,
        "causal_input": True,
        "teacher_training_evidence": "offline_frozen_teacher",
    }


def test_rlaif_teacher_selects_only_between_task_candidates() -> None:
    tasks, _ = build_rlhf_annotation_tasks((_row(1, "left"), _row(2, "right")))
    pairs, stats = build_rlaif_reward_pairs(tasks, (_score(tasks[0]["task_id"]),))
    assert len(pairs) == 1
    assert stats["evidence_class"] == "offline_rlaif_teacher_only"
    assert pairs[0].pair.label_source == "teacher"
    assert pairs[0].pair.chosen_action == 1
    assert pairs[0].pair.rejected_action == 2
    assert pairs[0].pair.candidate_only is True
    assert pairs[0].pair.teacher_training_evidence == "offline_frozen_teacher"


def test_rlaif_skips_ambiguous_teacher_margin() -> None:
    tasks, _ = build_rlhf_annotation_tasks((_row(1, "left"), _row(2, "right")))
    scores = _score(tasks[0]["task_id"], scores=[0.0, 0.51, 0.50, 0.0, 0.0, 0.0, 0.0])
    pairs, stats = build_rlaif_reward_pairs(tasks, (scores,), minimum_score_margin=0.05)
    assert pairs == ()
    assert stats["ambiguous_margin"] == 1


def test_rlaif_rejects_future_outcome_payload_and_invalid_lines() -> None:
    path = Path("outputs/rlhf_test_rlaif_teacher_scores.jsonl")
    bad = _score("task")
    bad["net_r"] = 1.0
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(bad) + "\nnot-json\n", encoding="utf-8")
        rows, invalid = load_rlaif_teacher_scores(path)
        assert rows == ()
        assert invalid == (1, 2)
    finally:
        path.unlink(missing_ok=True)


def test_frozen_policy_teacher_scores_only_the_two_causal_candidates() -> None:
    tasks, _ = build_rlhf_annotation_tasks((_row(1, "left"), _row(2, "right")))

    class FakeNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(()))

        def initial_memories(self, *, device):
            return []

        def forward_five(self, observation, memories, *, deterministic):
            logits = torch.zeros((5, 12), dtype=observation.dtype, device=observation.device)
            logits[:, 1] = 0.20
            logits[:, 2] = 0.80
            return [], [], {"action_logits": logits}

    rows, report = score_tasks(
        [dict(tasks[0])],
        FakeNetwork(),
        teacher_id="fake-teacher",
        teacher_version="fake-v1",
        teacher_training_evidence="unit-test-frozen-checkpoint",
        confidence=0.8,
    )
    assert report["score_rows_emitted"] == 1
    assert rows[0]["action_scores"][1] == pytest.approx(0.20)
    assert rows[0]["action_scores"][2] == pytest.approx(0.80)
    assert all(rows[0]["action_scores"][action] == 0.0 for action in (0, 3, 4, 5, 6))


def test_chronological_teacher_split_keeps_each_context_together() -> None:
    first, _ = build_rlhf_annotation_tasks((_row(1, "a1"), _row(2, "a2")))
    second, _ = build_rlhf_annotation_tasks((_row(1, "b1"), _row(2, "b2")))
    tasks = [
        dict(first[0], decision_time="2026-01-01T00:00:00+00:00"),
        dict(second[0], decision_time="2026-02-01T00:00:00+00:00"),
    ]
    # Two contexts are insufficient by design; no accidental train/test split
    # is allowed when there is no held-out context.
    with pytest.raises(ValueError, match="at least three contexts"):
        chronological_task_splits(tasks)
