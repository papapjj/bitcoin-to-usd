"""Fail-closed append-only capture for human RLHF feedback pairs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

from .rlhf_preference import (
    load_rlhf_preference_jsonl,
    parse_rlhf_preference_row,
)
from .rlhf_reward_model import load_rlhf_reward_jsonl, parse_rlhf_reward_row


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def append_rlhf_feedback_row(
    path: str | Path,
    row: Mapping[str, object],
    *,
    reward_pair: bool = True,
    require_human_feedback: bool = True,
) -> dict[str, object]:
    """Validate and append one rater row without overwriting old feedback.

    This is intentionally a single-writer primitive.  A UI/service that uses
    multiple writers must put an external file lock around this call.
    Existing malformed lines or duplicate pair IDs abort the append.
    """

    output = Path(path)
    if reward_pair:
        parsed = parse_rlhf_reward_row(row)
        if parsed is None:
            raise ValueError("invalid causal RLHF reward pair")
        pair = parsed.pair
        canonical_row = parsed.to_row()
        existing, invalid_lines = load_rlhf_reward_jsonl(output) if output.exists() else ((), ())
    else:
        parsed_preference = parse_rlhf_preference_row(row)
        if parsed_preference is None:
            raise ValueError("invalid causal RLHF preference pair")
        parsed = parsed_preference
        pair = parsed_preference
        canonical_row = parsed_preference.to_row()
        existing, invalid_lines = (
            load_rlhf_preference_jsonl(output) if output.exists() else ((), ())
        )
    if invalid_lines:
        raise ValueError("cannot append to a dataset with invalid existing lines")
    if require_human_feedback and pair.label_source not in {"human", "expert"}:
        raise ValueError("human RLHF capture only accepts human or expert raters")
    if any(existing_pair.pair_id == pair.pair_id for existing_pair in existing):
        raise ValueError("duplicate RLHF pair_id")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(canonical_row, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "pair_id": pair.pair_id,
        "dataset_path": str(output),
        "dataset_sha256": _sha256_file(output),
        "row_count": len(existing) + 1,
        "label_source": pair.label_source,
        "rater_id": pair.rater_id,
        "reward_pair": bool(reward_pair),
        "evidence_class": "human_feedback_capture",
        "promotion_allowed": False,
    }
