#!/usr/bin/env python3
"""Score causal RLHF tasks with a frozen local Five policy teacher.

The output is RLAIF supervision, not human feedback.  Only the two actions
already present in each annotation task receive teacher scores; no lifecycle,
PnL, or future field is read by this worker.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.self_growing_runner import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.seven_choice_teacher import shared_five_network
from self_growing_trader.rlhf_annotation import load_rlhf_annotation_tasks
from self_growing_trader.rlhf_rlaif import RLAIF_TEACHER_SCHEMA_VERSION


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _candidate_raw_obs(candidate: object) -> np.ndarray:
    if not isinstance(candidate, dict):
        raise ValueError("annotation candidate must be an object")
    values = np.asarray(candidate.get("features", ()), dtype=np.float32).reshape(-1)
    if values.shape != (144,) or not bool(np.all(np.isfinite(values))):
        raise ValueError("annotation candidate must contain finite 144 features")
    return values[:137]


def _candidate_action(candidate: object) -> int:
    if not isinstance(candidate, dict):
        raise ValueError("annotation candidate must be an object")
    try:
        action = int(candidate.get("action"))
    except (TypeError, ValueError) as exc:
        raise ValueError("annotation candidate action is invalid") from exc
    if not 0 <= action < 7:
        raise ValueError("annotation candidate action is outside entry surface")
    return action


def _candidate_agent(candidate: object) -> int:
    if not isinstance(candidate, dict):
        raise ValueError("annotation candidate must be an object")
    try:
        agent = int(candidate.get("agent_index"))
    except (TypeError, ValueError) as exc:
        raise ValueError("annotation candidate agent index is invalid") from exc
    if not 0 <= agent < 5:
        raise ValueError("annotation candidate agent index is outside Five")
    return agent


def chronological_task_splits(
    tasks: list[dict[str, object]],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[str, str]:
    """Assign whole causal contexts to chronological train/validation/test splits."""

    train_fraction = float(train_fraction)
    validation_fraction = float(validation_fraction)
    if (
        not np.isfinite(train_fraction)
        or not np.isfinite(validation_fraction)
        or not 0.0 < train_fraction < 1.0
        or not 0.0 <= validation_fraction < 1.0
        or train_fraction + validation_fraction >= 1.0
    ):
        raise ValueError("invalid chronological split fractions")
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for task in tasks:
        context = str(task.get("context_id", "")).strip()
        if not context:
            raise ValueError("task context_id is required for chronological split")
        groups[context].append(task)
    ordered = sorted(
        groups.items(),
        key=lambda item: (
            min(str(task.get("decision_time", "")) for task in item[1]),
            item[0],
        ),
    )
    if len(ordered) < 3:
        raise ValueError("chronological split requires at least three contexts")
    train_count = max(1, int(np.floor(len(ordered) * train_fraction)))
    validation_count = max(1, int(np.floor(len(ordered) * validation_fraction)))
    if train_count + validation_count >= len(ordered):
        validation_count = max(1, len(ordered) - train_count - 1)
    if train_count + validation_count >= len(ordered):
        train_count = max(1, len(ordered) - validation_count - 1)
    assignment: dict[str, str] = {}
    for index, (context, context_tasks) in enumerate(ordered):
        split = (
            "train" if index < train_count
            else "validation" if index < train_count + validation_count
            else "test"
        )
        for task in context_tasks:
            task_id = str(task.get("task_id", "")).strip()
            if not task_id:
                raise ValueError("task_id is required for chronological split")
            assignment[task_id] = split
    return assignment


def score_tasks(
    tasks: list[dict[str, object]],
    network,
    *,
    teacher_id: str,
    teacher_version: str,
    teacher_training_evidence: str,
    confidence: float,
    split_by_task_id: dict[str, str] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not teacher_id.strip() or not teacher_version.strip():
        raise ValueError("teacher_id and teacher_version are required")
    if not np.isfinite(float(confidence)) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    if not teacher_training_evidence.strip():
        raise ValueError("teacher_training_evidence is required")
    network.eval()
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for task in tasks:
            candidates = task.get("candidates")
            if not isinstance(candidates, list) or len(candidates) != 2:
                raise ValueError("RLAIF task must contain exactly two candidates")
            scores = np.zeros(7, dtype=np.float64)
            scored_actions: set[int] = set()
            agent_indices: list[int] = []
            for candidate in candidates:
                action = _candidate_action(candidate)
                agent = _candidate_agent(candidate)
                raw_obs = _candidate_raw_obs(candidate)
                observation = torch.as_tensor(
                    np.repeat(raw_obs[None, :], 5, axis=0),
                    dtype=dtype,
                    device=device,
                )
                memories = network.initial_memories(device=device)
                _, _, tensors = network.forward_five(
                    observation,
                    memories,
                    deterministic=True,
                )
                logits = tensors["action_logits"].detach().cpu().numpy()
                if logits.shape != (5, 12) or not np.all(np.isfinite(logits)):
                    raise ValueError("frozen teacher returned invalid action logits")
                scores[action] = float(logits[agent, action])
                scored_actions.add(action)
                agent_indices.append(agent)
            if scored_actions != {
                _candidate_action(candidates[0]),
                _candidate_action(candidates[1]),
            }:
                raise ValueError("teacher did not score both task actions")
            rows.append({
                "schema_version": RLAIF_TEACHER_SCHEMA_VERSION,
                "task_id": str(task.get("task_id", "")),
                "teacher_id": teacher_id,
                "teacher_version": teacher_version,
                "teacher_training_evidence": teacher_training_evidence,
                "label_source": "teacher",
                "action_scores": [float(value) for value in scores],
                "confidence": float(confidence),
                "split": str(
                    (split_by_task_id or {}).get(
                        str(task.get("task_id", "")),
                        (task.get("annotation") or {}).get("split", "train"),
                    )
                ),
                "source_run_id": str(task.get("source_run_id", "") or ""),
                "causal_input": True,
                "score_mode": "frozen_policy_action_logit_v1",
                "teacher_agent_indices": agent_indices,
            })
    return rows, {
        "tasks_seen": len(tasks),
        "score_rows_emitted": len(rows),
        "split_counts": {
            split: sum(1 for row in rows if row["split"] == split)
            for split in ("train", "validation", "test")
        },
        "teacher_id": teacher_id,
        "teacher_version": teacher_version,
        "teacher_training_evidence": teacher_training_evidence,
        "promotion_allowed": False,
        "evidence_class": "offline_rlaif_teacher_only",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--dataset-config", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--teacher-id", default="frozen-local-five")
    parser.add_argument("--teacher-version", default="frozen_policy_action_logit_v1")
    parser.add_argument("--confidence", type=float, default=0.75)
    parser.add_argument("--max-tasks", type=int, default=0)
    parser.add_argument(
        "--chronological-split", action="store_true",
        help="Assign whole contexts to chronological train/validation/test splits.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"teacher score output already exists: {args.output}")
    checkpoint_hash = _sha256(args.checkpoint)
    tasks = list(load_rlhf_annotation_tasks(args.tasks))
    if args.max_tasks:
        if args.max_tasks < 1:
            raise ValueError("max-tasks must be positive when supplied")
        tasks = tasks[:args.max_tasks]
    split_by_task_id = (
        chronological_task_splits(
            tasks,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
        )
        if args.chronological_split else None
    )
    config = load_config(args.dataset_config)
    system = tai_he_chuyen_gia_mot_context_mot_policy(args.checkpoint, config, seed=0)
    network = shared_five_network(system)
    network.require_current_architecture()
    rows, report = score_tasks(
        tasks,
        network,
        teacher_id=args.teacher_id,
        teacher_version=args.teacher_version,
        teacher_training_evidence=f"frozen_checkpoint_sha256:{checkpoint_hash}",
        confidence=args.confidence,
        split_by_task_id=split_by_task_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report.update({
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "tasks": str(args.tasks),
        "tasks_sha256": _sha256(args.tasks),
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    })
    report_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
