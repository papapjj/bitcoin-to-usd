"""Frozen counterfactual input attribution for a context-expert checkpoint.

For each recorded entry decision, this tool keeps the recurrent state, action
legality, causal context-history tokens and all other normalized observations
fixed.  It replaces one *wire* input by the median of its (context, action)
cell, recomputes the entry-quality interactions, and measures:

    delta logit = logits(original) - logits(counterfactual)

This is decision-surface attribution, not a market counterfactual or proof
that removing a feature improves realised PnL.  The script deliberately loads
a checkpoint read-only, disables learning and TimingNet action overlays, then
restores the exact deterministic Full-RL dropout step before every paired
forward.  It writes only its JSON audit.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from self_growing_trader.input_normalization import normalize_input_vector
from self_growing_trader.nhat_ky_giai_thich import _feature_name
from self_growing_trader.protocol_contract import INPUT_COUNT


ENTRY_ACTIONS = frozenset(range(1, 7))
ENTRY_ACTIVE_INDICES = frozenset(range(0, 29)) | frozenset(range(77, 106)) | frozenset(range(117, 137))
MASKED_SENTINEL_INDICES = (29, 69, 106)


def _integer(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _parse_inputs(value: object) -> np.ndarray | None:
    try:
        values = np.asarray([float(item) for item in str(value or "").split("|")], dtype=float)
    except ValueError:
        return None
    if values.size != INPUT_COUNT or not np.isfinite(values).all():
        return None
    return values


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _snapshot_runtime(policy: object) -> dict[str, object]:
    """Keep only state that can influence the next Direct-RNN forward pass."""

    modules = list(getattr(policy, "modules", ()) or ())
    return {
        "semantic_runtime_memory": copy.deepcopy(getattr(policy, "_semantic_runtime_memory", {})),
        "semantic_deep_states": copy.deepcopy(getattr(policy, "_semantic_deep_states", {})),
        "semantic_rl_histories": copy.deepcopy(getattr(policy, "_semantic_rl_histories", {})),
        "step_count": int(getattr(policy, "_step_count", 0) or 0),
        "last_observation": np.asarray(getattr(policy, "_last_observation", np.zeros(0)), dtype=float).copy(),
        "module_deep_states": [
            tuple(np.asarray(item, dtype=float).copy() for item in (getattr(module, "deep_state", ()) or ()))
            for module in modules
        ],
        # Full-RL deliberately retains dropout in frozen inference so online
        # and frozen surfaces stay comparable.  It is therefore part of the
        # causal forward state: without restoring it, a no-op feature ablation
        # would consume a different mask and fabricate a logit delta.
        "module_hidden_dropout_steps": [
            int(getattr(module, "hidden_dropout_step", 0) or 0)
            for module in modules
        ],
    }


def _restore_runtime(policy: object, snapshot: dict[str, object]) -> None:
    policy._semantic_runtime_memory = copy.deepcopy(snapshot["semantic_runtime_memory"])
    policy._semantic_deep_states = copy.deepcopy(snapshot["semantic_deep_states"])
    policy._semantic_rl_histories = copy.deepcopy(snapshot["semantic_rl_histories"])
    policy._step_count = int(snapshot["step_count"])
    policy._last_observation = np.asarray(snapshot["last_observation"], dtype=float).copy()
    for module, deep_state, dropout_step in zip(
        getattr(policy, "modules", ()) or (),
        snapshot["module_deep_states"],
        snapshot["module_hidden_dropout_steps"],
    ):
        module.deep_state = tuple(np.asarray(item, dtype=float).copy() for item in deep_state)
        module.hidden_dropout_step = int(dropout_step)


def _configure_entry(policy: object, context: str, allowed_actions: object) -> None:
    allowed = allowed_actions if isinstance(allowed_actions, list) else list(range(8))
    if not any(_integer(item) in ENTRY_ACTIONS for item in allowed):
        allowed = list(range(8))
    policy._learning_enabled = False
    # Full-RL already suppresses its action delta.  Disable this auxiliary
    # recurrent branch as well so it cannot advance state while auditing a
    # Direct-RNN-only decision surface.
    policy.timing_network_enabled = False
    policy._context_expert_management_only = False
    policy._context_expert_owner_metadata = {
        "ma_context": context,
        "decision_policy_context": context,
        "management_only": False,
    }
    policy._decision_context = {"allowed_actions": list(allowed)}
    policy._account_context = {}
    policy._positions_pipe_context = ""


def _load_final_r_by_event(ledger_path: Path) -> dict[str, float]:
    entry_event_by_trace: dict[str, str] = {}
    final_r_by_trace: dict[str, float] = {}
    priority = {"ORDER_ADD": 0, "DEAL_ADD": 1, "REQUEST": 2}
    for row in _read_tsv(ledger_path):
        trace_id = str(row.get("trace_id", "") or "")
        event_id = str(row.get("event_id", "") or "")
        action = _integer(row.get("requested_action"))
        transaction = str(row.get("transaction_type", "") or "").upper()
        if trace_id and event_id and action in ENTRY_ACTIONS and transaction in priority:
            # First accepted intent is immutable entry provenance for a trace.
            entry_event_by_trace.setdefault(trace_id, event_id)
        if str(row.get("position_final_close", "")).strip().lower() not in {"1", "true", "yes"}:
            continue
        owner_trace = str(row.get("position_owner_trace_id", "") or "") or trace_id
        try:
            initial_risk = float(row.get("initial_risk", 0.0) or 0.0)
            net = (
                float(row.get("profit", 0.0) or 0.0)
                + float(row.get("commission", 0.0) or 0.0)
                + float(row.get("swap", 0.0) or 0.0)
            )
        except ValueError:
            continue
        if owner_trace and initial_risk > 1e-12:
            final_r_by_trace[owner_trace] = net / initial_risk
    return {
        event_id: final_r_by_trace[trace_id]
        for trace_id, event_id in entry_event_by_trace.items()
        if trace_id in final_r_by_trace
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_path = Path(args.checkpoint)
    feature_path = Path(args.feature_audit)
    metadata_path = Path(args.metadata)
    archive_path = Path(args.archive)
    ledger_path = Path(args.ledger)

    with checkpoint_path.open("rb") as handle:
        system = pickle.load(handle)

    raw_features: dict[str, np.ndarray] = {}
    for row in _read_tsv(feature_path):
        event_id = str(row.get("event_id", "") or "")
        vector = _parse_inputs(row.get("inputs"))
        if event_id and vector is not None:
            raw_features.setdefault(event_id, vector)

    metadata: list[dict[str, object]] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if str(row.get("event_id", "")) in raw_features:
                metadata.append(row)
    metadata.sort(key=lambda row: _integer(row.get("sequence"), 0))

    # Conditioned medians are calculated in the model's normalized wire space.
    # Cells below support are excluded rather than using the sample itself as
    # its "counterfactual" median.
    entry_samples: list[tuple[str, str, int]] = []
    cells: dict[tuple[str, int], list[np.ndarray]] = defaultdict(list)
    for row in metadata:
        context = str(row.get("decision_policy_context") or row.get("context_nhan_dien") or "")
        active_context = str(row.get("context_nhan_dien") or context)
        action = _integer(row.get("policy_action"))
        event_id = str(row.get("event_id", ""))
        if (
            str(row.get("decision_role", "")) == "CONTEXT_ENTRY_GENERALIST"
            and context == active_context
            and context in system.experts
            and action in ENTRY_ACTIONS
        ):
            vector = normalize_input_vector(raw_features[event_id], mode=args.normalization_mode, size=INPUT_COUNT)
            entry_samples.append((event_id, context, action))
            cells[(context, action)].append(vector)
    medians = {
        key: np.median(np.vstack(values), axis=0)
        for key, values in cells.items()
        if len(values) >= args.min_cell_support
    }

    archive = json.loads(archive_path.read_text(encoding="utf-8"))
    cohen_d = {
        int(row["index"]): float(row["cohen_d"])
        for row in archive.get("feature_statistics", [])
    }
    selected_indices = {
        index
        for index in ENTRY_ACTIVE_INDICES
        if abs(cohen_d.get(index, 0.0)) <= float(args.max_abs_cohen_d)
    }
    if args.include_masked_sentinels:
        selected_indices.update(MASKED_SENTINEL_INDICES)
    selected_indices = sorted(selected_indices)

    outcome_by_event = _load_final_r_by_event(ledger_path)
    for expert in system.experts.values():
        policy = expert.policy
        policy._learning_enabled = False
        policy.timing_network_enabled = False
        policy.reset_runtime_state()
        policy._semantic_runtime_memory = {}
        policy._semantic_deep_states = {}
        policy._semantic_rl_histories = {}
    system._context_history = []

    aggregates: dict[int, dict[str, list[float]]] = {index: defaultdict(list) for index in selected_indices}
    frozen_action_agreement = 0
    scored = 0
    skipped_small_cell = 0
    skipped_context_mismatch = 0
    for row in metadata:
        event_id = str(row.get("event_id", ""))
        active_context = str(row.get("context_nhan_dien") or row.get("decision_policy_context") or "")
        if active_context not in system.experts:
            continue
        policy = system.experts[active_context].policy
        _configure_entry(policy, active_context, row.get("context_allowed_actions"))
        normalized_wire = normalize_input_vector(raw_features[event_id], mode=args.normalization_mode, size=INPUT_COUNT)
        policy_observation, _ = system._policy_observation_with_context_history(normalized_wire)
        before = _snapshot_runtime(policy)
        base_trace = policy._act_direct_rnn(policy_observation)
        after = _snapshot_runtime(policy)

        selected_context = str(row.get("decision_policy_context") or active_context)
        selected_action = _integer(row.get("policy_action"))
        cell = (selected_context, selected_action)
        if event_id not in {item[0] for item in entry_samples}:
            system._record_current_context(active_context)
            continue
        if selected_context != active_context:
            skipped_context_mismatch += 1
            system._record_current_context(active_context)
            continue
        if cell not in medians:
            skipped_small_cell += 1
            system._record_current_context(active_context)
            continue

        base_logits = np.asarray(base_trace.action_details["direct_rnn_rl_logits"], dtype=float)
        selected = int(np.argmax(base_logits))
        frozen_action_agreement += int(selected == selected_action)
        other = base_logits.copy()
        other[selected] = -np.inf
        base_margin = float(base_logits[selected] - np.max(other))
        outcome = outcome_by_event.get(event_id)
        outcome_group = "loss" if outcome is not None and outcome < 0 else ("win" if outcome is not None and outcome > 0 else "unknown")
        median = medians[cell]
        for index in selected_indices:
            altered = normalized_wire.copy()
            altered[index] = median[index]
            counterfactual_observation, _ = system._policy_observation_with_context_history(altered)
            _restore_runtime(policy, before)
            counterfactual = policy._act_direct_rnn(counterfactual_observation)
            cf_logits = np.asarray(counterfactual.action_details["direct_rnn_rl_logits"], dtype=float)
            cf_other = cf_logits.copy()
            cf_other[selected] = -np.inf
            cf_margin = float(cf_logits[selected] - np.max(cf_other))
            values = aggregates[index]
            values["mean_abs_logit_delta"].append(float(np.mean(np.abs(base_logits - cf_logits))) )
            values["selected_logit_delta"].append(float(base_logits[selected] - cf_logits[selected]))
            margin_drop = float(base_margin - cf_margin)
            values["margin_drop"].append(margin_drop)
            flipped = float(int(np.argmax(cf_logits) != selected))
            values["action_flip"].append(flipped)
            values[f"action_flip_{outcome_group}"].append(flipped)
            values[f"margin_drop_{outcome_group}"].append(margin_drop)
        _restore_runtime(policy, after)
        scored += 1
        system._record_current_context(active_context)

    rows: list[dict[str, object]] = []
    for index in selected_indices:
        values = aggregates[index]
        rows.append({
            "index": index,
            "feature": _feature_name(index),
            "archived_cohen_d": cohen_d.get(index),
            "mean_abs_logit_delta": _mean(values["mean_abs_logit_delta"]),
            "mean_selected_logit_delta": _mean(values["selected_logit_delta"]),
            "mean_margin_drop": _mean(values["margin_drop"]),
            "action_flip_rate": _mean(values["action_flip"]),
            "loss_action_flip_rate": _mean(values["action_flip_loss"]),
            "win_action_flip_rate": _mean(values["action_flip_win"]),
            "loss_mean_margin_drop": _mean(values["margin_drop_loss"]),
            "win_mean_margin_drop": _mean(values["margin_drop_win"]),
            "loss_sample_count": len(values["action_flip_loss"]),
            "win_sample_count": len(values["action_flip_win"]),
            "entry_route_active": index in ENTRY_ACTIVE_INDICES,
        })
    rows.sort(key=lambda row: (float(row["action_flip_rate"] or 0.0), float(row["mean_abs_logit_delta"] or 0.0)), reverse=True)
    return {
        "kind": "frozen_conditional_median_logit_attribution",
        "interpretation": "decision-surface attribution only; market/PnL counterfactual requires MT5 replay",
        "checkpoint": str(checkpoint_path),
        "normalization_mode": args.normalization_mode,
        "feature_audit_rows": len(raw_features),
        "metadata_rows": len(metadata),
        "entry_samples_total": len(entry_samples),
        "entry_samples_scored": scored,
        "min_conditional_cell_support": args.min_cell_support,
        "eligible_condition_cells": {
            f"{context}|action_{action}": len(cells[(context, action)])
            for context, action in sorted(medians)
        },
        "skipped_context_mismatch": skipped_context_mismatch,
        "skipped_small_cell": skipped_small_cell,
        "frozen_action_agreement_with_recorded_action": frozen_action_agreement / max(1, scored),
        "known_final_outcome_count": sum(1 for event_id, _, _ in entry_samples if event_id in outcome_by_event),
        "features": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--feature-audit", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normalization-mode", default="structure_aware_v4")
    parser.add_argument("--min-cell-support", type=int, default=8)
    parser.add_argument("--max-abs-cohen-d", type=float, default=0.10)
    parser.add_argument("--include-masked-sentinels", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = run(arguments)
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "entry_samples_scored": result["entry_samples_scored"],
        "feature_count": len(result["features"]),
        "frozen_action_agreement_with_recorded_action": result["frozen_action_agreement_with_recorded_action"],
    }, ensure_ascii=False))
