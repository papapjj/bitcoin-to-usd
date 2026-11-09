from __future__ import annotations

import json

import scripts.watch_v13_learning_live as watch


def test_learning_watch_counts_direct_selector_updates(tmp_path, monkeypatch) -> None:
    event_path = tmp_path / "learning_events.jsonl"
    event_path.write_text(
        json.dumps({
            "event_type": "selector_update",
            "run_id": "run-selector",
            "payload": {"proposal_id": "expert-a", "realized_utility": -0.5},
        }) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(watch, "LEARNING_EVENTS", event_path)
    monkeypatch.setattr(watch, "RUN_ID", "run-selector")

    snapshot = watch._learning_event_snapshot()

    assert snapshot["selector_update_events"] == 1
    assert snapshot["lifecycle_source"] == "none"
