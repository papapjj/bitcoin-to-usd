"""Causal teacher/RLAIF adapters for the RLHF preference pipeline.

This module is intentionally separate from human RLHF.  A teacher may be a
local frozen policy, an offline value teacher, or an MT5 audit model.  It may
produce candidate scores, but its labels are never counted as human/expert
feedback and never grant promotion by themselves.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .rlhf_annotation import (
    RLHF_ANNOTATION_SCHEMA_VERSION,
    build_rlhf_reward_pair_from_task,
)
from .rlhf_reward_model import RLHFRewardPair


RLAIF_TEACHER_SCHEMA_VERSION = "rlaif_teacher_score_v1"
RLAIF_LABEL_SOURCES = frozenset({"teacher", "mt5_rater"})


def _text(value: object) -> str:
    return str(value or "").strip()


def _finite_float(value: object, default: float = float("nan")) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) else float(default)


def _reason_codes(value: object) -> tuple[str, ...]:
    if isinstance(value, (tuple, list, set)):
        values = value
    else:
        values = _text(value).replace(",", "|").split("|")
    return tuple(sorted({_text(item).lower() for item in values if _text(item)}))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_teacher_score(row: Mapping[str, object]) -> dict[str, object]:
    """Validate one teacher score without accepting outcome fields as input.

    The teacher may have been trained offline on historical outcomes, but a
    score row must contain only a task id and a score over the seven actions.
    Future/PnL/lifecycle values in the score payload are rejected so they
    cannot silently become annotation features or a claimed live label.
    """

    if _text(row.get("schema_version")) != RLAIF_TEACHER_SCHEMA_VERSION:
        raise ValueError("unsupported RLAIF teacher score schema")
    task_id = _text(row.get("task_id"))
    teacher_id = _text(row.get("teacher_id"))
    teacher_version = _text(row.get("teacher_version"))
    source = _text(row.get("label_source", "teacher")).lower()
    if not task_id or not teacher_id or not teacher_version:
        raise ValueError("RLAIF score requires task_id, teacher_id and teacher_version")
    if source not in RLAIF_LABEL_SOURCES:
        raise ValueError("RLAIF score label_source must be teacher or mt5_rater")
    if row.get("causal_input") is not True:
        raise ValueError("RLAIF score must declare causal_input=true")
    scores = np.asarray(row.get("action_scores", ()), dtype=np.float64).reshape(-1)
    if scores.shape != (7,) or not bool(np.all(np.isfinite(scores))):
        raise ValueError("RLAIF action_scores must be finite seven-action data")
    confidence = _finite_float(row.get("confidence"), float("nan"))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("RLAIF teacher confidence must be in [0,1]")
    # These keys are never valid teacher input.  They indicate that a caller
    # attempted to smuggle an outcome audit into the annotation decision.
    forbidden = {
        "pnl", "net_r", "gross_r", "mfe_r", "mae_r", "profit", "close_row",
        "position_ticket", "exit_code", "future_bars", "future_path", "lifecycle",
    }
    present = sorted(key for key in forbidden if key in row)
    if present:
        raise ValueError("RLAIF score contains forbidden outcome fields: " + ",".join(present))
    teacher_training_evidence = _text(row.get("teacher_training_evidence"))
    if not teacher_training_evidence:
        raise ValueError("RLAIF score requires teacher_training_evidence")
    return {
        "schema_version": RLAIF_TEACHER_SCHEMA_VERSION,
        "task_id": task_id,
        "teacher_id": teacher_id,
        "teacher_version": teacher_version,
        "label_source": source,
        "causal_input": True,
        "action_scores": tuple(float(value) for value in scores),
        "confidence": float(confidence),
        "split": _text(row.get("split", "train")).lower(),
        "reason_codes": _reason_codes(row.get("reason_codes", ())),
        "horizon_bars": int(float(row.get("horizon_bars", 48) or 48)),
        "source_run_id": _text(row.get("source_run_id")),
        "teacher_training_evidence": teacher_training_evidence,
    }


def load_rlaif_teacher_scores(
    path: str | Path,
) -> tuple[tuple[dict[str, object], ...], tuple[int, ...]]:
    """Load score rows and return valid rows plus rejected line numbers."""

    rows: list[dict[str, object]] = []
    invalid: list[int] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
                if not isinstance(value, Mapping):
                    raise ValueError("row must be an object")
                rows.append(_validate_teacher_score(value))
            except (json.JSONDecodeError, TypeError, ValueError):
                invalid.append(line_number)
    return tuple(rows), tuple(invalid)


def build_rlaif_reward_pairs(
    tasks: Iterable[Mapping[str, object]],
    teacher_scores: Iterable[Mapping[str, object]],
    *,
    minimum_score_margin: float = 0.0,
) -> tuple[tuple[RLHFRewardPair, ...], dict[str, object]]:
    """Convert explicit teacher scores into candidate-only reward pairs.

    The task still supplies the two causal feature vectors.  The teacher only
    chooses between those two actions; it cannot invent an action that was not
    shown to the rater and it cannot read an outcome field from the task.
    Ambiguous scores are skipped instead of forcing a label.
    """

    margin_floor = float(minimum_score_margin)
    if not np.isfinite(margin_floor) or margin_floor < 0.0:
        raise ValueError("minimum_score_margin must be finite and non-negative")
    task_by_id = {
        _text(task.get("task_id")): task
        for task in tasks
        if _text(task.get("task_id"))
    }
    pairs: list[RLHFRewardPair] = []
    stats: Counter[str] = Counter()
    seen_task_ids: set[str] = set()
    seen_pair_ids: set[str] = set()
    for raw_score in teacher_scores:
        stats["score_rows_seen"] += 1
        score = _validate_teacher_score(raw_score)
        task_id = str(score["task_id"])
        task = task_by_id.get(task_id)
        if task is None:
            stats["task_missing"] += 1
            continue
        if task_id in seen_task_ids:
            stats["duplicate_task_score"] += 1
            continue
        if _text(task.get("schema_version")) != RLHF_ANNOTATION_SCHEMA_VERSION:
            stats["task_schema_invalid"] += 1
            continue
        candidates = task.get("candidates")
        if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
            stats["task_candidates_invalid"] += 1
            continue
        candidate_actions = {
            int(candidate.get("action"))
            for candidate in candidates
            if isinstance(candidate, Mapping)
        }
        if len(candidate_actions) != 2 or any(
            action < 0 or action >= 7 for action in candidate_actions
        ):
            stats["task_action_surface_invalid"] += 1
            continue
        scores = np.asarray(score["action_scores"], dtype=np.float64)
        ordered = sorted(candidate_actions, key=lambda action: (float(scores[action]), action), reverse=True)
        chosen_action, rejected_action = ordered
        score_margin = float(scores[chosen_action] - scores[rejected_action])
        if score_margin < margin_floor:
            stats["ambiguous_margin"] += 1
            continue
        candidate_by_action = {
            int(candidate.get("action")): candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
        }
        chosen = candidate_by_action[chosen_action]
        rejected = candidate_by_action[rejected_action]
        try:
            pair = build_rlhf_reward_pair_from_task(
                task,
                chosen_trace_id=_text(chosen.get("trace_id")),
                rejected_trace_id=_text(rejected.get("trace_id")),
                label_source=str(score["label_source"]),
                rater_id=str(score["teacher_id"]),
                rater_version=str(score["teacher_version"]),
                confidence=float(score["confidence"]),
                split=str(score["split"]),
                reason_codes=tuple(score["reason_codes"]) + ("rlaif_score_margin",),
                horizon_bars=int(score["horizon_bars"]),
                teacher_training_evidence=str(score["teacher_training_evidence"]),
            )
        except (TypeError, ValueError):
            stats["pair_validation_failed"] += 1
            continue
        # ``pair_id`` is generated by the shared builder from task/rater/traces.
        # Keep a deterministic duplicate guard and retain the score margin only
        # in the report; the causal feature vector remains untouched.
        if pair.pair_id in seen_pair_ids:
            stats["duplicate_pair"] += 1
            continue
        seen_task_ids.add(task_id)
        seen_pair_ids.add(pair.pair_id)
        pairs.append(pair)
        stats["pairs_emitted"] += 1
    stats.setdefault("score_rows_seen", 0)
    stats.setdefault("pairs_emitted", 0)
    return tuple(pairs), {
        **{key: int(value) for key, value in stats.items()},
        "minimum_score_margin": margin_floor,
        "teacher_source_counts": dict(Counter(pair.pair.label_source for pair in pairs)),
        "teacher_ids": sorted({pair.pair.rater_id for pair in pairs}),
        "teacher_training_evidence": sorted({
            pair.pair.teacher_training_evidence
            for pair in pairs
            if pair.pair.teacher_training_evidence
        }),
        "promotion_allowed": False,
        "evidence_class": "offline_rlaif_teacher_only",
        "requires_human_rlhf_gate": False,
        "requires_mt5_frozen_oos_lcb": True,
    }


def write_rlaif_reward_dataset(
    output: str | Path,
    pairs: Sequence[RLHFRewardPair],
    *,
    tasks_path: str | Path,
    scores_path: str | Path,
    stats: Mapping[str, object],
) -> dict[str, object]:
    """Write a new RLAIF JSONL dataset and an immutable provenance manifest."""

    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"RLAIF output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair.to_row(), ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "schema_version": RLAIF_TEACHER_SCHEMA_VERSION,
        "dataset": str(destination),
        "dataset_sha256": _sha256(destination),
        "tasks": str(Path(tasks_path)),
        "tasks_sha256": _sha256(Path(tasks_path)),
        "teacher_scores": str(Path(scores_path)),
        "teacher_scores_sha256": _sha256(Path(scores_path)),
        "pair_count": len(pairs),
        "stats": dict(stats),
        "promotion_allowed": False,
        "evidence_class": "offline_rlaif_teacher_only",
    }
    # Preserve the score worker's teacher identity so the downstream reward
    # model report can prove which frozen policy produced the labels.  A
    # hand-authored score file may not have a sidecar; in that case leave the
    # provenance absent and let any independent-teacher deployment gate fail
    # closed.
    score_manifest_path = Path(scores_path).with_suffix(
        Path(scores_path).suffix + ".manifest.json"
    )
    if score_manifest_path.is_file():
        try:
            score_manifest = json.loads(score_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            score_manifest = {}
        if isinstance(score_manifest, Mapping):
            manifest["teacher_provenance"] = {
                "teacher_id": str(score_manifest.get("teacher_id", "") or ""),
                "teacher_version": str(score_manifest.get("teacher_version", "") or ""),
                "teacher_checkpoint_sha256": str(
                    score_manifest.get("checkpoint_sha256", "") or ""
                ),
                "score_manifest": str(score_manifest_path),
                "score_manifest_sha256": _sha256(score_manifest_path),
            }
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest
