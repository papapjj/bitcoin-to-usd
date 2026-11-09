from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from self_growing_trader.rlhf_annotation import (
    RLHF_ANNOTATION_SCHEMA_VERSION,
    build_rlhf_annotation_tasks,
    build_rlhf_reward_pair_from_task,
    load_rlhf_annotation_tasks,
)
from self_growing_trader.rlhf_feedback import append_rlhf_feedback_row
from self_growing_trader.rlhf_reward_model import load_rlhf_reward_jsonl


def _observatory_row(*, action: int, trace_id: str, agent_index: int) -> dict[str, object]:
    details = {
        "five_agent_index": agent_index,
        "five_action_index": action,
        "five_decision_state_sha256": "state-a",
        "five_risk_geometry_sha256": "geometry-a",
        "action_name": "BUY_MARKET",
        "sl_atr": 1.5,
        "tp_atr": 3.0,
    }
    return {
        "schema_version": 1,
        "run_id": "run-a",
        "created_at": 1780000000.0,
        "trace_id": trace_id,
        "request": {
            "event_id": "event-a",
            "sequence": 7,
            "time": "2026.09.22 08:00:00",
        },
        "trace": {
            "trace_id": trace_id,
            "raw_obs": np.linspace(-1.0, 1.0, 137).tolist(),
            "selected_idx": action,
            "action_details": details,
        },
    }


def test_annotation_queue_groups_same_causal_context_without_choosing_winner() -> None:
    tasks, stats = build_rlhf_annotation_tasks((
        _observatory_row(action=1, trace_id="trace-a", agent_index=0),
        _observatory_row(action=2, trace_id="trace-b", agent_index=1),
        _observatory_row(action=7, trace_id="management", agent_index=2),
    ))
    assert stats["rows_usable"] == 2
    assert stats["rows_rejected"] == 1
    assert len(tasks) == 1
    task = tasks[0]
    assert task["schema_version"] == RLHF_ANNOTATION_SCHEMA_VERSION
    assert task["annotation"]["chosen_trace_id"] == ""
    assert {candidate["action"] for candidate in task["candidates"]} == {1, 2}
    assert len(task["candidates"][0]["features"]) == 144
    assert task["candidates"][0]["rater_view"]["sl_atr"] == 1.5


def test_annotation_queue_rejects_malformed_observation_without_crashing() -> None:
    row = _observatory_row(action=1, trace_id="bad", agent_index=0)
    row["trace"]["raw_obs"] = "not-a-vector"
    tasks, stats = build_rlhf_annotation_tasks((row,))
    assert tasks == ()
    assert stats["rows_rejected"] == 1


def test_annotation_decision_becomes_append_only_reward_pair() -> None:
    tasks, _ = build_rlhf_annotation_tasks((
        _observatory_row(action=1, trace_id="trace-a", agent_index=0),
        _observatory_row(action=2, trace_id="trace-b", agent_index=1),
    ))
    task_path = Path("outputs/rlhf_test_annotation_tasks.jsonl")
    dataset_path = Path("outputs/rlhf_test_annotation_pairs.jsonl")
    task_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        task_path.write_text(
            "\n".join(json.dumps(task) for task in tasks) + "\n",
            encoding="utf-8",
        )
        loaded = load_rlhf_annotation_tasks(task_path)
        assert len(loaded) == 1
        pair = build_rlhf_reward_pair_from_task(
            loaded[0],
            chosen_trace_id="trace-a",
            rejected_trace_id="trace-b",
            label_source="expert",
            rater_id="teacher-1",
            rater_version="rubric-v1",
            confidence=0.9,
            reason_codes=("timing", "risk"),
            feedback_time="2026-09-22T08:01:00+00:00",
        )
        receipt = append_rlhf_feedback_row(
            dataset_path,
            pair.to_row(),
            reward_pair=True,
            require_human_feedback=True,
        )
        assert receipt["row_count"] == 1
        rows, invalid = load_rlhf_reward_jsonl(dataset_path)
        assert not invalid
        assert len(rows) == 1
        assert rows[0].pair.label_source == "expert"
        assert rows[0].pair.chosen_action == 1
    finally:
        task_path.unlink(missing_ok=True)
        dataset_path.unlink(missing_ok=True)


def test_annotation_decision_rejects_tampered_candidate_features() -> None:
    tasks, _ = build_rlhf_annotation_tasks((
        _observatory_row(action=1, trace_id="trace-a", agent_index=0),
        _observatory_row(action=2, trace_id="trace-b", agent_index=1),
    ))
    tampered = json.loads(json.dumps(tasks[0]))
    tampered["candidates"][0]["features"][137 + 1] += 0.25
    try:
        build_rlhf_reward_pair_from_task(
            tampered,
            chosen_trace_id="trace-a",
            rejected_trace_id="trace-b",
            label_source="expert",
            rater_id="teacher-1",
            rater_version="rubric-v1",
            confidence=0.9,
            feedback_time="2026-09-22T08:01:00+00:00",
        )
    except ValueError as exc:
        assert "one-hot does not match" in str(exc)
    else:
        raise AssertionError("tampered annotation features must fail closed")
