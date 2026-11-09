"""Build and label causal RLHF annotation tasks from MT5 observatory rows.

The observatory is an evidence source for the decision-time context, not a
label source.  This module only creates pairwise annotation tasks from entry
actions that were observed in the same immutable context.  A human or
qualified expert must still choose the preferred action before the row can
enter the reward-model dataset.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from .rlhf_policy_loop import causal_policy_feature_vector
from .rlhf_reward_model import RLHFRewardPair
from .rlhf_preference import (
    RLHFPreferencePair,
    RLHF_SPLITS,
    rlhf_dataset_balance_gate,
    rlhf_dataset_balance_validation_gate,
)


RLHF_ANNOTATION_SCHEMA_VERSION = "rlhf_annotation_task_v1"
TRIAD_ACTIONS: tuple[int, int, int] = (0, 1, 2)
TRIAD_ACTION_NAMES: dict[int, str] = {
    0: "HOLD",
    1: "MARKET_BUY",
    2: "MARKET_SELL",
}


def _text(value: object) -> str:
    return str(value or "").strip()


def _integer(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _iso_time(value: object, *, fallback_epoch: object = 0.0) -> str:
    """Return an ISO timestamp accepted by ``RLHFPreferencePair``."""

    raw = _text(value)
    if raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            parsed = None
        if parsed is None:
            for fmt in (
                "%Y.%m.%d %H:%M:%S",
                "%Y.%m.%d %H:%M",
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f",
            ):
                try:
                    parsed = datetime.strptime(raw, fmt).replace(tzinfo=timezone.utc)
                    break
                except ValueError:
                    continue
        if parsed is not None:
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
    try:
        epoch = float(fallback_epoch)
    except (TypeError, ValueError):
        epoch = 0.0
    if not np.isfinite(epoch) or epoch <= 0.0:
        epoch = 0.0
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def _canonical_hash(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_from_observatory_row(row: Mapping[str, object]) -> dict[str, object] | None:
    trace = row.get("trace")
    if not isinstance(trace, Mapping):
        return None
    details = trace.get("action_details")
    if not isinstance(details, Mapping):
        return None
    try:
        raw_obs = np.asarray(trace.get("raw_obs", ()), dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if raw_obs.size != 137 or not bool(np.all(np.isfinite(raw_obs))):
        return None
    action = _integer(details.get("five_action_index", trace.get("selected_idx")))
    if not 0 <= action < 7:
        return None
    state_hash = _text(details.get("five_decision_state_sha256"))
    geometry_hash = _text(details.get("five_risk_geometry_sha256"))
    if not state_hash or not geometry_hash:
        return None
    request = row.get("request")
    request_map = request if isinstance(request, Mapping) else {}
    event_id = _text(request_map.get("event_id"))
    sequence = _integer(request_map.get("sequence"), -1)
    if not event_id and sequence < 0:
        return None
    trace_id = _text(row.get("trace_id") or trace.get("trace_id"))
    if not trace_id:
        return None
    try:
        feature = causal_policy_feature_vector(raw_obs, action)
    except ValueError:
        return None
    rater_view: dict[str, object] = {}
    for name in (
        "action_name", "target_slot", "management_mode", "partial_close_pct",
        "trailing_stop_atr", "sl_atr", "tp_atr", "five_delay_steps",
        "five_delay_index", "five_risk_index", "five_entry_offset_index",
        "five_sl_index", "five_tp_index", "five_trailing_index",
        "five_manage_mode_index", "five_close_fraction_index",
    ):
        value = details.get(name)
        if isinstance(value, (str, int, float, bool)) or value is None:
            rater_view[name] = value
    probabilities = details.get("five_policy_action_probabilities")
    if isinstance(probabilities, Sequence) and not isinstance(probabilities, (str, bytes)):
        if action < len(probabilities):
            try:
                probability = float(probabilities[action])
            except (TypeError, ValueError):
                probability = None
            if probability is not None and np.isfinite(probability):
                rater_view["policy_action_probability"] = probability
    return {
        "trace_id": trace_id,
        "action": int(action),
        "features": [float(value) for value in feature],
        "state_hash": state_hash,
        "geometry_hash": geometry_hash,
        "agent_index": _integer(details.get("five_agent_index"), -1),
        "action_name": _text(details.get("action_name")),
        "decision_time": _iso_time(
            request_map.get("time"), fallback_epoch=row.get("created_at", 0.0)
        ),
        "source_run_id": _text(row.get("run_id")),
        "request_event_id": event_id,
        "request_sequence": int(sequence),
        "rater_view": rater_view,
    }


def generate_counterfactual_triad_candidates(
    base_candidate_or_row: Mapping[str, object],
    *,
    raw_obs: Sequence[float] | np.ndarray | None = None,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Generate counterfactual candidates for the triad {HOLD=0, BUY=1, SELL=2} at the same market state.

    Given a base candidate observation or MT5 observatory row, constructs three candidate
    dictionaries sharing the exact same state_hash, geometry_hash, decision_time, and 137-dim
    market state features, but with causal feature vectors corresponding to HOLD (0),
    MARKET_BUY (1), and MARKET_SELL (2).
    """
    if "trace" in base_candidate_or_row and isinstance(base_candidate_or_row["trace"], Mapping):
        base_cand = _candidate_from_observatory_row(base_candidate_or_row)
        if base_cand is None:
            raise ValueError("cannot generate triad from invalid observatory row")
    else:
        base_cand = dict(base_candidate_or_row)

    if raw_obs is not None:
        obs_137 = np.asarray(raw_obs, dtype=np.float32).reshape(-1)
    else:
        features = base_cand.get("features", ())
        arr = np.asarray(features, dtype=np.float32).reshape(-1)
        if arr.size >= 137:
            obs_137 = arr[:137]
        else:
            raise ValueError("base candidate does not contain at least 137 features")

    if obs_137.size != 137 or not bool(np.all(np.isfinite(obs_137))):
        raise ValueError("raw_obs must be a finite 137-dimensional state vector")

    base_trace_id = str(base_cand.get("trace_id", "") or "trace")
    if ":cf_" in base_trace_id:
        base_trace_id = base_trace_id.split(":cf_")[0]

    state_hash = str(base_cand.get("state_hash", "") or "")
    geometry_hash = str(base_cand.get("geometry_hash", "") or "")
    agent_index = _integer(base_cand.get("agent_index", -1), -1)
    decision_time = str(base_cand.get("decision_time", "") or "")
    source_run_id = str(base_cand.get("source_run_id", "") or "")
    request_event_id = str(base_cand.get("request_event_id", "") or "")
    request_sequence = _integer(base_cand.get("request_sequence", -1), -1)
    base_rater_view = dict(base_cand.get("rater_view", {}) or {})

    triad: list[dict[str, object]] = []
    for action in TRIAD_ACTIONS:
        action_name = TRIAD_ACTION_NAMES[action]
        feature_vector = causal_policy_feature_vector(obs_137, action)
        rater_view = dict(base_rater_view)
        rater_view["action_name"] = action_name
        rater_view["five_action_index"] = action
        rater_view["is_counterfactual_triad"] = True

        candidate = {
            "trace_id": f"{base_trace_id}:cf_{action_name.lower()}",
            "action": int(action),
            "features": [float(val) for val in feature_vector],
            "state_hash": state_hash,
            "geometry_hash": geometry_hash,
            "agent_index": int(agent_index),
            "action_name": action_name,
            "decision_time": decision_time,
            "source_run_id": source_run_id,
            "request_event_id": request_event_id,
            "request_sequence": int(request_sequence),
            "rater_view": rater_view,
        }
        triad.append(candidate)

    return (triad[0], triad[1], triad[2])


def build_counterfactual_triad_tasks(
    rows: Iterable[Mapping[str, object]],
    *,
    max_tasks: int = 0,
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    """Build pairwise annotation tasks covering all triad combinations (BUY vs HOLD, SELL vs HOLD, BUY vs SELL).

    For each unique decision context (event_key, state_hash, geometry_hash), generates
    the counterfactual triad {HOLD=0, BUY=1, SELL=2} and constructs all 3 pairwise tasks:
    (BUY, HOLD), (SELL, HOLD), (BUY, SELL).
    """
    task_limit = int(max_tasks)
    if task_limit < 0:
        raise ValueError("max_tasks must be non-negative")

    groups: dict[tuple[str, str, str], dict[str, object]] = {}
    stats = {
        "rows_seen": 0,
        "rows_usable": 0,
        "rows_rejected": 0,
        "contexts_with_triads": 0,
        "tasks_emitted": 0,
    }

    for row in rows:
        stats["rows_seen"] += 1
        candidate = _candidate_from_observatory_row(row) if ("trace" in row) else dict(row)
        if candidate is None or not candidate.get("state_hash"):
            stats["rows_rejected"] += 1
            continue
        stats["rows_usable"] += 1
        event_key = _text(candidate.get("request_event_id", ""))
        if not event_key:
            event_key = f"sequence:{int(candidate.get('request_sequence', -1))}"
        key = (
            event_key,
            str(candidate["state_hash"]),
            str(candidate["geometry_hash"]),
        )
        if key not in groups:
            groups[key] = candidate

    tasks: list[dict[str, object]] = []
    for (event_key, state_hash, geometry_hash), base_cand in sorted(groups.items()):
        try:
            triad = generate_counterfactual_triad_candidates(base_cand)
        except Exception:
            continue
        stats["contexts_with_triads"] += 1
        triad_pairs = [
            (triad[1], triad[0]),  # BUY (1) vs HOLD (0)
            (triad[2], triad[0]),  # SELL (2) vs HOLD (0)
            (triad[1], triad[2]),  # BUY (1) vs SELL (2)
        ]
        for left, right in triad_pairs:
            task_identity = {
                "schema_version": RLHF_ANNOTATION_SCHEMA_VERSION,
                "context_id": f"mt5:{event_key}:{state_hash}:{geometry_hash}",
                "left_trace_id": str(left["trace_id"]),
                "right_trace_id": str(right["trace_id"]),
                "left_action": int(left["action"]),
                "right_action": int(right["action"]),
            }
            task_id = _canonical_hash(task_identity)
            tasks.append({
                "schema_version": RLHF_ANNOTATION_SCHEMA_VERSION,
                "task_id": task_id,
                "context_id": task_identity["context_id"],
                "decision_time": min(
                    str(left["decision_time"]), str(right["decision_time"])
                ),
                "pre_decision_state_hash": state_hash,
                "risk_geometry_hash": geometry_hash,
                "source_run_id": str(left["source_run_id"] or right["source_run_id"]),
                "candidates": [left, right],
                "annotation": {
                    "chosen_trace_id": "",
                    "rejected_trace_id": "",
                    "label_source": "",
                    "rater_id": "",
                    "rater_version": "",
                    "confidence": 0.0,
                    "split": "train",
                    "reason_codes": [],
                },
            })
            if task_limit and len(tasks) >= task_limit:
                break
        if task_limit and len(tasks) >= task_limit:
            break

    stats["tasks_emitted"] = len(tasks)
    return tuple(tasks), stats


def build_rlhf_annotation_tasks(
    rows: Iterable[Mapping[str, object]],
    *,
    max_pairs_per_context: int = 3,
    max_tasks: int = 0,
    generate_counterfactual_triads: bool = False,
) -> tuple[tuple[dict[str, object], ...], dict[str, int]]:
    """Create unlabeled pair tasks from causal MT5 observatory rows.

    Rows are grouped by request event plus the immutable decision and risk
    hashes.  Duplicate observations of the same action are collapsed.  The
    function never chooses a winner and never reads lifecycle/PnL fields.
    If generate_counterfactual_triads is True, each context is augmented with
    the counterfactual triad {HOLD=0, BUY=1, SELL=2}.
    """

    pair_limit = int(max_pairs_per_context)
    if pair_limit < 1:
        raise ValueError("max_pairs_per_context must be positive")
    task_limit = int(max_tasks)
    if task_limit < 0:
        raise ValueError("max_tasks must be non-negative")
    groups: dict[tuple[str, str, str], dict[int, dict[str, object]]] = {}
    stats = {
        "rows_seen": 0,
        "rows_usable": 0,
        "rows_rejected": 0,
        "contexts_with_pairs": 0,
        "tasks_emitted": 0,
    }
    for row in rows:
        stats["rows_seen"] += 1
        candidate = _candidate_from_observatory_row(row)
        if candidate is None:
            stats["rows_rejected"] += 1
            continue
        stats["rows_usable"] += 1
        event_key = _text(candidate["request_event_id"])
        if not event_key:
            event_key = f"sequence:{int(candidate['request_sequence'])}"
        key = (
            event_key,
            str(candidate["state_hash"]),
            str(candidate["geometry_hash"]),
        )
        actions = groups.setdefault(key, {})
        action = int(candidate["action"])
        actions.setdefault(action, candidate)

    tasks: list[dict[str, object]] = []
    for (event_key, state_hash, geometry_hash), actions in sorted(groups.items()):
        if generate_counterfactual_triads and actions:
            first_cand = next(iter(actions.values()))
            try:
                triad = generate_counterfactual_triad_candidates(first_cand)
                for c in triad:
                    act = int(c["action"])
                    if act not in actions:
                        actions[act] = c
            except Exception:
                pass

        candidates = [actions[action] for action in sorted(actions)]
        pairs = list(itertools.combinations(candidates, 2))[:pair_limit]
        if not pairs:
            continue
        stats["contexts_with_pairs"] += 1
        for left, right in pairs:
            task_identity = {
                "schema_version": RLHF_ANNOTATION_SCHEMA_VERSION,
                "context_id": f"mt5:{event_key}:{state_hash}:{geometry_hash}",
                "left_trace_id": str(left["trace_id"]),
                "right_trace_id": str(right["trace_id"]),
                "left_action": int(left["action"]),
                "right_action": int(right["action"]),
            }
            task_id = _canonical_hash(task_identity)
            tasks.append({
                "schema_version": RLHF_ANNOTATION_SCHEMA_VERSION,
                "task_id": task_id,
                "context_id": task_identity["context_id"],
                "decision_time": min(
                    str(left["decision_time"]), str(right["decision_time"])
                ),
                "pre_decision_state_hash": state_hash,
                "risk_geometry_hash": geometry_hash,
                "source_run_id": str(left["source_run_id"] or right["source_run_id"]),
                "candidates": [left, right],
                "annotation": {
                    "chosen_trace_id": "",
                    "rejected_trace_id": "",
                    "label_source": "",
                    "rater_id": "",
                    "rater_version": "",
                    "confidence": 0.0,
                    "split": "train",
                    "reason_codes": [],
                },
            })
            if task_limit and len(tasks) >= task_limit:
                break
        if task_limit and len(tasks) >= task_limit:
            break
    stats["tasks_emitted"] = len(tasks)
    return tuple(tasks), stats


def load_rlhf_annotation_tasks(
    path: str | Path,
    *,
    task_id: str = "",
) -> tuple[dict[str, object], ...]:
    """Load valid task objects from JSONL and optionally select one task."""

    wanted = _text(task_id)
    tasks: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            text = raw.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid annotation task JSON at line {line_number}") from exc
            if not isinstance(row, Mapping):
                raise ValueError(f"annotation task at line {line_number} is not an object")
            if _text(row.get("schema_version")) != RLHF_ANNOTATION_SCHEMA_VERSION:
                raise ValueError(f"unsupported annotation task schema at line {line_number}")
            if wanted and _text(row.get("task_id")) != wanted:
                continue
            tasks.append(dict(row))
    if wanted and not tasks:
        raise ValueError(f"annotation task not found: {wanted}")
    return tuple(tasks)


def build_rlhf_reward_pair_from_task(
    task: Mapping[str, object],
    *,
    chosen_trace_id: str,
    rejected_trace_id: str,
    label_source: str,
    rater_id: str,
    rater_version: str,
    confidence: float,
    split: str = "train",
    reason_codes: Sequence[str] = (),
    horizon_bars: int = 48,
    feedback_time: str | None = None,
    teacher_training_evidence: str = "",
) -> RLHFRewardPair:
    """Turn one explicit rater choice into a validated reward pair."""

    if _text(task.get("schema_version")) != RLHF_ANNOTATION_SCHEMA_VERSION:
        raise ValueError("unsupported annotation task schema")
    candidates = task.get("candidates")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ValueError("annotation task candidates are missing")
    by_trace = {
        _text(candidate.get("trace_id")): candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and _text(candidate.get("trace_id"))
    }
    chosen_id = _text(chosen_trace_id)
    rejected_id = _text(rejected_trace_id)
    if chosen_id == rejected_id or chosen_id not in by_trace or rejected_id not in by_trace:
        raise ValueError("chosen/rejected trace IDs must name two distinct task candidates")
    chosen = by_trace[chosen_id]
    rejected = by_trace[rejected_id]
    expected_state_hash = _text(task.get("pre_decision_state_hash"))
    expected_geometry_hash = _text(task.get("risk_geometry_hash"))
    for candidate in (chosen, rejected):
        if _text(candidate.get("state_hash")) != expected_state_hash:
            raise ValueError("annotation candidate state hash does not match task")
        if _text(candidate.get("geometry_hash")) != expected_geometry_hash:
            raise ValueError("annotation candidate geometry hash does not match task")
        action = _integer(candidate.get("action"))
        try:
            features = np.asarray(candidate.get("features", ()), dtype=np.float32).reshape(-1)
        except (TypeError, ValueError) as exc:
            raise ValueError("annotation candidate features are invalid") from exc
        if features.size != 144 or not bool(np.all(np.isfinite(features))):
            raise ValueError("annotation candidate features must be finite 144-dimensional data")
        expected_action = np.zeros(7, dtype=np.float32)
        if not 0 <= action < 7:
            raise ValueError("annotation candidate action is outside the entry surface")
        expected_action[action] = 1.0
        if not bool(np.array_equal(features[137:], expected_action)):
            raise ValueError("annotation candidate action one-hot does not match action")
    decision_time = _iso_time(task.get("decision_time"))
    feedback = _iso_time(feedback_time, fallback_epoch=datetime.now(tz=timezone.utc).timestamp())
    split_name = _text(split).lower()
    if split_name not in RLHF_SPLITS:
        raise ValueError("unsupported RLHF annotation split")
    pair_id = "|".join((
        _text(task.get("task_id")), _text(rater_id), chosen_id, rejected_id,
    ))
    pair = RLHFPreferencePair(
        pair_id=pair_id,
        context_id=_text(task.get("context_id")),
        chosen_trace_id=chosen_id,
        rejected_trace_id=rejected_id,
        chosen_action=_integer(chosen.get("action")),
        rejected_action=_integer(rejected.get("action")),
        decision_time=decision_time,
        feedback_time=feedback,
        pre_decision_state_hash=_text(task.get("pre_decision_state_hash")),
        risk_geometry_hash=_text(task.get("risk_geometry_hash")),
        label_source=_text(label_source).lower(),
        rater_id=_text(rater_id),
        rater_version=_text(rater_version),
        confidence=float(confidence),
        split=split_name,
        horizon_bars=int(horizon_bars),
        reason_codes=tuple(_text(value).lower() for value in reason_codes if _text(value)),
        source_run_id=_text(task.get("source_run_id")),
        teacher_training_evidence=_text(teacher_training_evidence),
        feature_causal=True,
        candidate_only=True,
        outcome_audit_only=True,
    )
    return RLHFRewardPair(
        pair=pair,
        chosen_features=tuple(float(value) for value in chosen.get("features", ())),
        rejected_features=tuple(float(value) for value in rejected.get("features", ())),
        feature_schema="mt5_raw_obs137_action_onehot_v1",
    )
