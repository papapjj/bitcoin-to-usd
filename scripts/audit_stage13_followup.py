#!/usr/bin/env python3
"""Fail-fast Stage13 follow-up: production parity, calibration, ablation, retrain."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.audit_five_architecture import (
    _action_head_eval,
    _binary_metrics,
    _frozen_policy_hidden_rows,
    _load_network,
    _margin_metrics,
    _replay_raw_hidden_cube,
    _sample_split_windows,
    _sha256,
    _state_hash_excluding,
)
from self_growing_trader.config import load_config
from self_growing_trader.five_shared_policy import (
    FIVE_AGENT_COUNT,
    FIVE_REASONING_ACTION_COUNT,
    LEGACY_HEURISTIC_CLOSE_ACTIONS,
    LONG_ENTRY_ACTIONS,
    SHORT_ENTRY_ACTIONS,
    FiveAgentMemory,
    SharedFiveTradingPolicy,
)
from self_growing_trader.five_teacher_imitation import _role_logits, shared_five_network
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


AUTHORIZED_PREFIXES = ("heads.action_query.", "heads.entry_role_calibrator.")
ROLES = (
    ("agent_0_buy_role", 0, 1, LONG_ENTRY_ACTIONS),
    ("agent_1_sell_role", 1, 2, SHORT_ENTRY_ACTIONS),
)
CLASS_WEIGHT_MODES = ("unweighted", "sqrt_inverse", "capped_inverse_3", "balanced_inverse")


def _atomic_json(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_pickle(value: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _state_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _load_and_apply_action_state(
    network: SharedFiveTradingPolicy,
    source_checkpoint: Path,
    state_path: Path,
) -> dict[str, Any]:
    payload = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("action-head artifact khong co state_dict hop le")
    expected_source = str(payload.get("source_checkpoint_sha256", ""))
    actual_source = _sha256(source_checkpoint)
    if expected_source != actual_source:
        raise ValueError(
            f"source checkpoint hash mismatch: artifact={expected_source}, actual={actual_source}"
        )
    state = dict(payload["state_dict"])
    if not state:
        raise ValueError("action-head state_dict rong")
    current = network.state_dict()
    unexpected = sorted(
        name for name in state
        if not name.startswith(AUTHORIZED_PREFIXES) or name not in current
    )
    missing = sorted(
        name for name in current
        if name.startswith(AUTHORIZED_PREFIXES) and name not in state
    )
    mismatched = sorted(
        name for name, value in state.items()
        if name in current and tuple(value.shape) != tuple(current[name].shape)
    )
    if unexpected or missing or mismatched:
        raise ValueError(
            "action-head authorization fail: "
            f"unexpected={unexpected}, missing={missing}, shape_mismatch={mismatched}"
        )
    current.update(state)
    network.load_state_dict(current, strict=True)
    calibrator = network.heads.entry_role_calibrator
    calibrator.teacher_entry_scope = str(payload.get("teacher_entry_scope", ""))
    if calibrator.teacher_entry_scope != "all_entry_mass_v2":
        raise ValueError("Stage13 state phai dung all_entry_mass_v2")
    calibrator.teacher_imitation_trained = True
    network.eval()
    return {
        "source_checkpoint_sha256": actual_source,
        "state_sha256": _state_sha256(state),
        "state_keys": sorted(state),
        "candidate_only": bool(payload.get("candidate_only", True)),
        "teacher_entry_scope": calibrator.teacher_entry_scope,
    }


def _manual_runtime_logits(
    network: SharedFiveTradingPolicy, policy_hidden: torch.Tensor
) -> torch.Tensor:
    ids = torch.arange(FIVE_AGENT_COUNT, device=policy_hidden.device, dtype=torch.long)
    logits = network.heads.primary_action_logits(policy_hidden, ids)
    if bool(getattr(network, "reasoning_rl_enabled", False)):
        allowed = torch.ones(
            (FIVE_AGENT_COUNT, FIVE_REASONING_ACTION_COUNT),
            dtype=torch.bool,
            device=policy_hidden.device,
        )
        _, reasoning_logits = network.reasoning_head.greedy(
            policy_hidden, allowed, steps=network.reasoning_steps
        )
        scale = float(getattr(network, "reasoning_action_scale", 0.0))
        if scale > 0.0:
            centered = reasoning_logits - reasoning_logits.mean(dim=-1, keepdim=True)
            residual = torch.clamp(
                centered,
                -float(network.reasoning_action_residual_clip),
                float(network.reasoning_action_residual_clip),
            )
            logits = logits.clone()
            logits[:, :FIVE_REASONING_ACTION_COUNT] += scale * residual
    logits = logits.clone()
    for action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
        logits[:, action] = -30.0
    return logits


def _role_mass_margin(logits: np.ndarray, agent_id: int) -> np.ndarray:
    entries = LONG_ENTRY_ACTIONS if agent_id == 0 else SHORT_ENTRY_ACTIONS
    values = np.asarray(logits, dtype=np.float64)
    selected = values[..., list(entries)]
    maximum = np.max(selected, axis=-1, keepdims=True)
    log_mass = np.squeeze(maximum, axis=-1) + np.log(
        np.sum(np.exp(selected - maximum), axis=-1)
    )
    return log_mass - values[..., 0]


def run_production_stage13_parity(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    state_path: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    atol: float = 1e-5,
    rtol: float = 1e-6,
) -> Path:
    network = _load_network(checkpoint, config_path)
    source_before = _sha256(checkpoint)
    frozen_before = _state_hash_excluding(network, AUTHORIZED_PREFIXES)
    state_audit = _load_and_apply_action_state(network, checkpoint, state_path)
    sampled = _sample_split_windows(
        dataset, split="validation", sequence_length=sequence_length, window_count=windows
    )
    raw_cube = _replay_raw_hidden_cube(network, sampled)
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    production_hidden: list[np.ndarray] = []
    production_policy_hidden: list[np.ndarray] = []
    production_logits: list[np.ndarray] = []
    production_actions: list[np.ndarray] = []
    with torch.inference_mode():
        for window in range(sampled["observations"].shape[0]):
            memories = network.initial_memories()
            hidden_rows: list[np.ndarray] = []
            policy_rows: list[np.ndarray] = []
            logit_rows: list[np.ndarray] = []
            action_rows: list[np.ndarray] = []
            for step in range(sampled["observations"].shape[1]):
                observation = torch.as_tensor(
                    sampled["observations"][window, step], device=device, dtype=dtype
                ).expand(FIVE_AGENT_COUNT, -1)
                results, next_memories, tensors = network.forward_five(
                    observation, memories, deterministic=True
                )
                hidden_rows.append(tensors["hidden"].detach().cpu().numpy())
                policy_rows.append(tensors["policy_hidden"].detach().cpu().numpy())
                logit_rows.append(tensors["action_logits"].detach().cpu().numpy())
                action_rows.append(np.asarray([item.action_index for item in results], dtype=np.int64))
                teacher_actions = sampled["actions"][window, step]
                memories = [
                    FiveAgentMemory(
                        hidden=memory.hidden,
                        cell=memory.cell,
                        previous_action=torch.as_tensor(
                            int(teacher_actions[index]), device=device, dtype=torch.long
                        ),
                    )
                    for index, memory in enumerate(next_memories)
                ]
            production_hidden.append(np.stack(hidden_rows))
            production_policy_hidden.append(np.stack(policy_rows))
            production_logits.append(np.stack(logit_rows))
            production_actions.append(np.stack(action_rows))
    production_hidden_np = np.stack(production_hidden)
    production_policy_np = np.stack(production_policy_hidden)
    production_logits_np = np.stack(production_logits)
    production_actions_np = np.stack(production_actions)
    manual_policy_rows: list[np.ndarray] = []
    manual_logit_rows: list[np.ndarray] = []
    manual_action_rows: list[np.ndarray] = []
    with torch.inference_mode():
        for window in range(raw_cube.shape[0]):
            policy_steps: list[np.ndarray] = []
            logit_steps: list[np.ndarray] = []
            action_steps: list[np.ndarray] = []
            for step in range(raw_cube.shape[1]):
                hidden = torch.as_tensor(raw_cube[window, step], device=device, dtype=dtype)
                policy_hidden = network.post_lstm_transformer(hidden)
                logits = _manual_runtime_logits(network, policy_hidden)
                policy_steps.append(policy_hidden.detach().cpu().numpy())
                logit_steps.append(logits.detach().cpu().numpy())
                action_steps.append(torch.argmax(logits, dim=-1).detach().cpu().numpy())
            manual_policy_rows.append(np.stack(policy_steps))
            manual_logit_rows.append(np.stack(logit_steps))
            manual_action_rows.append(np.stack(action_steps))
    manual_policy_np = np.stack(manual_policy_rows)
    manual_logits_np = np.stack(manual_logit_rows)
    manual_actions_np = np.stack(manual_action_rows)

    def comparison(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
        drift = np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))
        return {
            "shape": list(left.shape),
            "max_abs_drift": float(np.max(drift)),
            "mean_abs_drift": float(np.mean(drift)),
            "allclose": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        }

    checks = {
        "raw_lstm_hidden": comparison(production_hidden_np, raw_cube),
        "post_transformer_policy_hidden": comparison(production_policy_np, manual_policy_np),
        "final_preselection_action_logits": comparison(production_logits_np, manual_logits_np),
        "deterministic_actions": {
            "shape": list(production_actions_np.shape),
            "mismatch_count": int(np.sum(production_actions_np != manual_actions_np)),
            "exact_match": bool(np.array_equal(production_actions_np, manual_actions_np)),
        },
    }
    role_margin_checks: dict[str, object] = {}
    for role_name, agent_id, _entry_action, _entries in ROLES:
        role_margin_checks[role_name] = comparison(
            _role_mass_margin(production_logits_np[:, :, agent_id], agent_id),
            _role_mass_margin(manual_logits_np[:, :, agent_id], agent_id),
        )
    passed = (
        all(bool(checks[name]["allclose"]) for name in (
            "raw_lstm_hidden", "post_transformer_policy_hidden", "final_preselection_action_logits"
        ))
        and bool(checks["deterministic_actions"]["exact_match"])
        and all(bool(value["allclose"]) for value in role_margin_checks.values())
        and frozen_before == _state_hash_excluding(network, AUTHORIZED_PREFIXES)
        and source_before == _sha256(checkpoint)
    )
    payload = {
        "stage": 14,
        "name": "production_vs_stage13_parity",
        "passed": bool(passed),
        "fail_fast": True,
        "checkpoint": str(checkpoint),
        "state_artifact": str(state_path),
        "dataset": str(dataset),
        "split": "validation",
        "sequence_length": int(sequence_length),
        "window_count": int(sampled["observations"].shape[0]),
        "teacher_forced_previous_action": True,
        "production_path": "SharedFiveTradingPolicy.forward_five",
        "diagnostic_path": "causal_replay_then_post_lstm_transformer_then_primary_action_logits",
        "atol": float(atol),
        "rtol": float(rtol),
        "state_audit": state_audit,
        "checks": checks,
        "role_entry_mass_margin_checks": role_margin_checks,
        "frozen_backbone_unchanged": True,
        "source_checkpoint_unchanged": True,
    }
    path = output_dir / "14_production_stage13_parity.json"
    _atomic_json(payload, path)
    if not passed:
        raise RuntimeError(f"Production-vs-Stage13 parity FAILED; xem {path}")
    print(json.dumps({"artifact": str(path), "passed": True, "checks": checks}, ensure_ascii=False, indent=2))
    return path


def _constrained_threshold_sweep(
    labels: np.ndarray,
    margins: np.ndarray,
    *,
    rate_ceiling_multiplier: float = 1.10,
    minimum_balanced_accuracy: float = 0.52,
    minimum_precision: float = 0.30,
) -> dict[str, Any]:
    y = np.asarray(labels, dtype=np.int64)
    score = np.asarray(margins, dtype=np.float64)
    if len(y) != len(score) or len(np.unique(y)) != 2 or not np.all(np.isfinite(score)):
        raise ValueError("constrained threshold sweep requires finite aligned two-class rows")
    order = np.argsort(score, kind="stable")
    sorted_score = score[order]
    sorted_labels = y[order]
    unique, first_indices, counts = np.unique(
        sorted_score, return_index=True, return_counts=True
    )
    middle = (unique[:-1] + unique[1:]) * 0.5 if len(unique) > 1 else np.empty(0)
    thresholds = np.r_[unique[0] - 1e-6, middle, unique[-1] + 1e-6]
    positives_per_value = np.add.reduceat(sorted_labels, first_indices).astype(np.float64)
    negatives_per_value = counts.astype(np.float64) - positives_per_value
    cumulative_positive = np.cumsum(positives_per_value)
    cumulative_negative = np.cumsum(negatives_per_value)
    low_positive = np.r_[0.0, cumulative_positive[:-1], cumulative_positive[-1]]
    low_negative = np.r_[0.0, cumulative_negative[:-1], cumulative_negative[-1]]
    total_positive = float(cumulative_positive[-1])
    total_negative = float(cumulative_negative[-1])
    tp = total_positive - low_positive
    fn = low_positive
    fp = total_negative - low_negative
    tn = low_negative
    entry_recall = tp / np.maximum(1.0, tp + fn)
    hold_recall = tn / np.maximum(1.0, tn + fp)
    precision = tp / np.maximum(1.0, tp + fp)
    predicted_rate = (tp + fp) / float(len(y))
    f1 = 2.0 * precision * entry_recall / np.maximum(1e-12, precision + entry_recall)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(
        tp * tn - fp * fn,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    balanced = 0.5 * (entry_recall + hold_recall)
    roc_auc = float(roc_auc_score(y, score))
    average_precision = float(average_precision_score(y, score))
    label_rate = float(np.mean(y))
    ceiling = float(rate_ceiling_multiplier * label_rate)
    precision_floor = float(max(minimum_precision, 1.20 * label_rate))
    rows: list[dict[str, Any]] = []
    for index, threshold in enumerate(thresholds):
        metrics = {
            "balanced_accuracy": float(balanced[index]),
            "mcc": float(mcc[index]),
            "entry_precision": float(precision[index]),
            "entry_recall": float(entry_recall[index]),
            "entry_f1": float(f1[index]),
            "label_entry_rate": label_rate,
            "predicted_entry_rate": float(predicted_rate[index]),
            "roc_auc": roc_auc,
            "average_precision": average_precision,
            "threshold": float(threshold),
        }
        eligible = bool(
            metrics["balanced_accuracy"] >= minimum_balanced_accuracy
            and metrics["entry_precision"] >= precision_floor
            and metrics["predicted_entry_rate"] <= ceiling
        )
        rows.append({
            **metrics,
            "eligible": eligible,
            "rate_ceiling": ceiling,
            "precision_floor": precision_floor,
        })
    eligible_rows = [row for row in rows if row["eligible"]]
    feasible_rows = [row for row in rows if row["predicted_entry_rate"] <= ceiling]
    pool = eligible_rows or feasible_rows or rows
    selected = max(pool, key=lambda row: (
        float(row["mcc"]),
        float(row["balanced_accuracy"]),
        float(row["entry_precision"]),
        float(row["entry_f1"]),
        -abs(float(row["predicted_entry_rate"]) - label_rate),
    ))
    return {
        "selection_scope": "validation_only",
        "selection_rule": "eligible_then_mcc_then_balanced_accuracy_then_precision_then_f1",
        "candidate_count": int(len(rows)),
        "eligible_candidate_count": int(len(eligible_rows)),
        "selected_threshold": float(selected["threshold"]),
        "selected_validation_metrics": selected,
        "curve": rows,
    }


def _split_role_scores(
    network: SharedFiveTradingPolicy,
    rows: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, dict[str, np.ndarray]]:
    hidden, agents, labels = rows
    with torch.inference_mode():
        query = network.heads.action_query(hidden)
        logits = _role_logits(query, agents, network)
        margins = (logits[:, 1] - logits[:, 0]).detach().cpu().numpy()
    agent_values = agents.detach().cpu().numpy()
    y = labels.detach().cpu().numpy()
    return {
        role_name: {"labels": y[agent_values == agent_id], "margins": margins[agent_values == agent_id]}
        for role_name, agent_id, _entry_action, _entries in ROLES
    }


def _operating_points(
    network: SharedFiveTradingPolicy,
    rows: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    scores = {split: _split_role_scores(network, values) for split, values in rows.items()}
    sweeps: dict[str, Any] = {}
    evaluations: dict[str, Any] = {split: {} for split in rows}
    for role_name, _agent_id, _entry_action, _entries in ROLES:
        sweep = _constrained_threshold_sweep(
            scores["validation"][role_name]["labels"],
            scores["validation"][role_name]["margins"],
        )
        threshold = float(sweep["selected_threshold"])
        sweeps[role_name] = sweep
        for split in rows:
            labels = scores[split][role_name]["labels"]
            margins = scores[split][role_name]["margins"]
            evaluations[split][role_name] = _margin_metrics(labels, margins, threshold)
    return sweeps, evaluations


def run_stage13_margin_sweep(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    state_path: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
) -> Path:
    parity_path = output_dir / "14_production_stage13_parity.json"
    if not parity_path.is_file() or not json.loads(parity_path.read_text(encoding="utf-8")).get("passed"):
        raise RuntimeError("Stage15 yeu cau Stage14 parity passed")
    network = _load_network(checkpoint, config_path)
    state_audit = _load_and_apply_action_state(network, checkpoint, state_path)
    sampled = {
        split: _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length, window_count=windows
        )
        for split in ("train", "validation", "test")
    }
    raw_logits: dict[str, np.ndarray] = {}
    rows: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for split in sampled:
        raw = _replay_raw_hidden_cube(network, sampled[split])
        shape = raw.shape
        device = next(network.parameters()).device
        dtype = next(network.parameters()).dtype
        with torch.inference_mode():
            hidden = torch.as_tensor(raw, device=device, dtype=dtype).reshape(-1, FIVE_AGENT_COUNT, shape[-1])
            policy_hidden = network.post_lstm_transformer(hidden)
            ids = torch.arange(FIVE_AGENT_COUNT, device=device).expand(policy_hidden.shape[0], -1)
            raw_logits[split] = network.heads.primary_action_logits(
                policy_hidden, ids
            ).detach().cpu().numpy().reshape(shape[0], shape[1], FIVE_AGENT_COUNT, -1)
        rows[split] = _frozen_policy_hidden_rows(network, sampled[split])
    sweeps, evaluations = _operating_points(network, rows)
    roles: dict[str, Any] = {}
    histogram: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role_name, agent_id, entry_action, _entries in ROLES:
        split_details: dict[str, Any] = {}
        for split in sampled:
            mask = sampled[split]["owner_masks"][:, :, agent_id]
            actions = sampled[split]["actions"][:, :, agent_id][mask]
            keep = np.isin(actions, (0, entry_action))
            logits = raw_logits[split][:, :, agent_id][mask][keep]
            labels = (actions[keep] == entry_action).astype(np.int64)
            market_margin = logits[:, entry_action] - logits[:, 0]
            mass_margin = _role_mass_margin(logits, agent_id)
            threshold = float(sweeps[role_name]["selected_threshold"])
            split_details[split] = {
                "samples": int(len(labels)),
                "market_action_margin_at_zero": _margin_metrics(labels, market_margin, 0.0),
                "entry_mass_margin_at_zero": _margin_metrics(labels, mass_margin, 0.0),
                "entry_mass_margin_at_validation_threshold": _margin_metrics(labels, mass_margin, threshold),
            }
            if split == "test":
                histogram[role_name] = (mass_margin[labels == 0], mass_margin[labels == 1])
        roles[role_name] = {
            "agent_id": int(agent_id),
            "market_entry_action": int(entry_action),
            "entry_mass_actions": list(_entries),
            "validation_threshold_sweep": sweeps[role_name],
            "evaluations": split_details,
        }
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "15_stage13_entry_mass_margin_histograms.png"
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for axis, (role_name, hold_entry) in zip(axes, histogram.items()):
        hold, entry = hold_entry
        axis.hist(hold, bins=40, alpha=0.65, label="HOLD", density=True)
        axis.hist(entry, bins=40, alpha=0.65, label="ENTRY", density=True)
        axis.axvline(float(sweeps[role_name]["selected_threshold"]), color="black", linestyle="--")
        axis.set_title(role_name)
        axis.set_xlabel("logsumexp(role entry logits) - HOLD logit")
        axis.legend()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    payload = {
        "stage": 15,
        "name": "stage13_logit_margin_and_validation_threshold_sweep",
        "checkpoint": str(checkpoint),
        "state_artifact": str(state_path),
        "state_audit": state_audit,
        "parity_artifact": str(parity_path),
        "chronological_split_contract": True,
        "threshold_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "roles": roles,
        "plot": str(plot_path),
    }
    path = output_dir / "15_stage13_logit_margin_threshold_sweep.json"
    _atomic_json(payload, path)
    print(json.dumps({
        "artifact": str(path),
        "plot": str(plot_path),
        "test": {name: value["evaluations"]["test"] for name, value in roles.items()},
    }, ensure_ascii=False, indent=2))
    return path


def _class_weights(labels: torch.Tensor, mode: str) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=2).to(dtype=torch.float32)
    if torch.any(counts <= 0):
        raise ValueError("class-weight ablation requires both classes")
    if mode == "unweighted":
        weights = torch.ones(2, dtype=torch.float32)
    elif mode == "sqrt_inverse":
        weights = torch.sqrt(counts.sum() / (2.0 * counts))
    elif mode == "capped_inverse_3":
        weights = torch.tensor([1.0, min(3.0, float(counts[0] / counts[1]))])
    elif mode == "balanced_inverse":
        weights = counts.sum() / (2.0 * counts)
    else:
        raise ValueError(f"unknown class-weight mode: {mode}")
    return weights / weights.mean()


def _validation_key(sweeps: dict[str, Any], loss: float) -> tuple[float, ...]:
    selected = [sweeps[name]["selected_validation_metrics"] for name, *_ in ROLES]
    return (
        float(sum(bool(item["eligible"]) for item in selected)),
        min(float(item["roc_auc"]) for item in selected),
        min(float(item["average_precision"]) for item in selected),
        min(float(item["mcc"]) for item in selected),
        min(float(item["balanced_accuracy"]) for item in selected),
        min(float(item["entry_precision"]) for item in selected),
        -float(loss),
    )


def _train_action_head(
    network: SharedFiveTradingPolicy,
    rows: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    *,
    class_weight_mode: str,
    epochs: int,
    patience: int,
    learning_rate: float,
    writer_dir: Path,
) -> dict[str, Any]:
    from torch.utils.tensorboard import SummaryWriter
    torch.manual_seed(0)
    np.random.seed(0)
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    for parameter in network.heads.action_query.parameters():
        parameter.requires_grad_(True)
    for parameter in network.heads.entry_role_calibrator.parameters():
        parameter.requires_grad_(True)
    trainable = [parameter for parameter in network.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=1e-4)
    train_hidden, train_agents, train_labels = rows["train"]
    weights = _class_weights(train_labels, class_weight_mode).to(
        device=train_hidden.device, dtype=train_hidden.dtype
    )
    initial_state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items() if name.startswith(AUTHORIZED_PREFIXES)
    }
    validation_rows = {"validation": rows["validation"]}
    initial_sweeps, initial_evaluations = _operating_points(network, validation_rows)
    initial_eval = _action_head_eval(network, *rows["validation"])
    best_key = _validation_key(initial_sweeps, float(initial_eval["overall"]["loss"]))
    best_state = initial_state
    best_epoch = 0
    best_sweeps = initial_sweeps
    best_evaluations = initial_evaluations
    stale = 0
    history: list[dict[str, Any]] = []
    writer = SummaryWriter(log_dir=str(writer_dir))
    try:
        for epoch in range(1, max(1, epochs) + 1):
            network.train()
            optimizer.zero_grad(set_to_none=True)
            logits = _role_logits(network.heads.action_query(train_hidden), train_agents, network)
            loss = torch.nn.functional.cross_entropy(logits, train_labels, weight=weights)
            loss.backward()
            gradient_max = 0.0
            for name, parameter in network.named_parameters():
                if not parameter.requires_grad:
                    if parameter.grad is not None:
                        raise RuntimeError(f"frozen parameter received gradient: {name}")
                    continue
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"missing/non-finite action-head gradient: {name}")
                gradient_max = max(gradient_max, float(torch.linalg.vector_norm(parameter.grad).item()))
            torch.nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()
            network.eval()
            sweeps, evaluations = _operating_points(network, validation_rows)
            validation_eval = _action_head_eval(network, *rows["validation"])
            key = _validation_key(sweeps, float(validation_eval["overall"]["loss"]))
            history.append({
                "epoch": epoch,
                "loss": float(loss.item()),
                "gradient_max_norm": gradient_max,
                "selection_key": list(key),
                "validation": evaluations["validation"],
            })
            writer.add_scalar("loss/train", float(loss.item()), epoch)
            writer.add_scalar("gradient/max_norm", gradient_max, epoch)
            for role_name in sweeps:
                metrics = sweeps[role_name]["selected_validation_metrics"]
                for metric in ("balanced_accuracy", "mcc", "entry_precision", "entry_recall", "entry_f1", "predicted_entry_rate"):
                    writer.add_scalar(f"validation/{role_name}/{metric}", float(metrics[metric]), epoch)
            if key > best_key:
                best_key = key
                best_epoch = epoch
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in network.state_dict().items() if name.startswith(AUTHORIZED_PREFIXES)
                }
                best_sweeps = sweeps
                best_evaluations = evaluations
                stale = 0
            else:
                stale += 1
            if stale >= max(1, patience):
                break
    finally:
        writer.close()
    current = network.state_dict()
    current.update(best_state)
    network.load_state_dict(current, strict=True)
    network.eval()
    final_sweeps, final_evaluations = _operating_points(network, rows)
    return {
        "class_weight_mode": class_weight_mode,
        "class_weights": weights.detach().cpu().tolist(),
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "selection_key": list(best_key),
        "state_dict": best_state,
        "state_sha256": _state_sha256(best_state),
        "validation_threshold_sweeps": final_sweeps,
        "evaluations": final_evaluations,
        "history": history,
        "tensorboard_log_dir": str(writer_dir),
        "initial_validation": initial_evaluations["validation"],
        "best_snapshot_validation": best_evaluations["validation"],
    }


def _prepare_neutral_action_head(network: SharedFiveTradingPolicy) -> dict[str, torch.Tensor]:
    network.heads.entry_role_calibrator.reset_teacher_mapping()
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    network.heads.entry_role_calibrator.teacher_imitation_trained = False
    return {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items() if name.startswith(AUTHORIZED_PREFIXES)
    }


def _restore_authorized(network: SharedFiveTradingPolicy, state: dict[str, torch.Tensor]) -> None:
    current = network.state_dict()
    current.update({name: value.clone() for name, value in state.items()})
    network.load_state_dict(current, strict=True)
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    network.heads.entry_role_calibrator.teacher_imitation_trained = False


def run_class_weight_ablation(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> Path:
    required = output_dir / "15_stage13_logit_margin_threshold_sweep.json"
    if not required.is_file():
        raise RuntimeError("Stage16 yeu cau Stage15 hoan thanh")
    network = _load_network(checkpoint, config_path)
    source_before = _sha256(checkpoint)
    frozen_before = _state_hash_excluding(network, AUTHORIZED_PREFIXES)
    sampled = {
        split: _sample_split_windows(dataset, split=split, sequence_length=sequence_length, window_count=windows)
        for split in ("train", "validation", "test")
    }
    rows = {split: _frozen_policy_hidden_rows(network, values) for split, values in sampled.items()}
    neutral_state = _prepare_neutral_action_head(network)
    arms: dict[str, Any] = {}
    winner_state: dict[str, torch.Tensor] | None = None
    winner_name = ""
    winner_key: tuple[float, ...] | None = None
    for mode in CLASS_WEIGHT_MODES:
        _restore_authorized(network, neutral_state)
        result = _train_action_head(
            network, rows,
            class_weight_mode=mode,
            epochs=epochs,
            patience=patience,
            learning_rate=learning_rate,
            writer_dir=output_dir / "16_class_weight_tensorboard_ranked" / mode,
        )
        state = result.pop("state_dict")
        key = tuple(float(value) for value in result["selection_key"])
        arms[mode] = result
        if winner_key is None or key > winner_key:
            winner_key = key
            winner_name = mode
            winner_state = state
    assert winner_state is not None and winner_key is not None
    if _state_hash_excluding(network, AUTHORIZED_PREFIXES) != frozen_before or _sha256(checkpoint) != source_before:
        raise RuntimeError("class-weight ablation changed frozen backbone or source checkpoint")
    state_path = output_dir / "16_class_weight_winner_state.pt"
    torch.save({
        "state_dict": winner_state,
        "teacher_entry_scope": "all_entry_mass_v2",
        "source_checkpoint_sha256": source_before,
        "candidate_only": True,
        "class_weight_mode": winner_name,
    }, state_path)
    payload = {
        "stage": 16,
        "name": "class_weight_ablation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_before,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "chronological_split_contract": True,
        "same_data_seed_epoch_budget": True,
        "seed": 0,
        "epochs_requested_per_arm": int(epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "winner_selected_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "candidate_only": True,
        "promotable": False,
        "winner": winner_name,
        "winner_selection_key": list(winner_key),
        "winner_state_artifact": str(state_path),
        "arms": arms,
        "frozen_backbone_unchanged": True,
        "source_checkpoint_unchanged": True,
    }
    path = output_dir / "16_class_weight_ablation.json"
    _atomic_json(payload, path)
    print(json.dumps({
        "artifact": str(path),
        "winner": winner_name,
        "selection_key": list(winner_key),
        "arms": {
            name: {
                "class_weights": value["class_weights"],
                "best_epoch": value["best_epoch"],
                "epochs_completed": value["epochs_completed"],
                "validation": value["evaluations"]["validation"],
                "test": value["evaluations"]["test"],
            } for name, value in arms.items()
        },
    }, ensure_ascii=False, indent=2))
    return path


def _record_validation_operating_points(
    network: SharedFiveTradingPolicy, sweeps: dict[str, Any]
) -> dict[str, float]:
    del network
    return {
        role_name: float(sweeps[role_name]["selected_threshold"])
        for role_name, _agent_id, _entry_action, _entries in ROLES
    }


def _exact_authorized_match(left: SharedFiveTradingPolicy, right: SharedFiveTradingPolicy) -> bool:
    left_state = left.state_dict()
    right_state = right.state_dict()
    return all(
        torch.equal(left_state[name].detach().cpu(), right_state[name].detach().cpu())
        for name in left_state if name.startswith(AUTHORIZED_PREFIXES)
    )


def run_head_only_retrain_v2(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output_dir: Path,
    *,
    sequence_length: int,
    windows: int,
    epochs: int,
    patience: int,
    learning_rate: float,
) -> Path:
    ablation_path = output_dir / "16_class_weight_ablation.json"
    if not ablation_path.is_file():
        raise RuntimeError("Stage17 yeu cau Stage16 hoan thanh")
    ablation = json.loads(ablation_path.read_text(encoding="utf-8"))
    mode = str(ablation["winner"])
    if mode not in CLASS_WEIGHT_MODES:
        raise RuntimeError(f"class-weight winner khong hop le: {mode}")
    network = _load_network(checkpoint, config_path)
    source_before = _sha256(checkpoint)
    frozen_before = _state_hash_excluding(network, AUTHORIZED_PREFIXES)
    sampled = {
        split: _sample_split_windows(dataset, split=split, sequence_length=sequence_length, window_count=windows)
        for split in ("train", "validation", "test")
    }
    rows = {split: _frozen_policy_hidden_rows(network, values) for split, values in sampled.items()}
    _prepare_neutral_action_head(network)
    result = _train_action_head(
        network, rows,
        class_weight_mode=mode,
        epochs=epochs,
        patience=patience,
        learning_rate=learning_rate,
        writer_dir=output_dir / "17_head_only_retrain_v2_tensorboard_ranked",
    )
    trained_state = result.pop("state_dict")
    _restore_authorized(network, trained_state)
    sweeps, pre_calibration = _operating_points(network, rows)
    audited_thresholds = _record_validation_operating_points(network, sweeps)
    network.heads.entry_role_calibrator.teacher_entry_scope = "all_entry_mass_v2"
    network.heads.entry_role_calibrator.teacher_imitation_trained = True
    post_calibration = {
        split: _action_head_eval(network, *rows[split])
        for split in ("train", "validation", "test")
    }
    final_state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items() if name.startswith(AUTHORIZED_PREFIXES)
    }
    state_path = output_dir / "17_head_only_retrain_v2_state.pt"
    torch.save({
        "state_dict": final_state,
        "teacher_entry_scope": "all_entry_mass_v2",
        "source_checkpoint_sha256": source_before,
        "candidate_only": True,
        "class_weight_mode": mode,
        "validation_operating_points_audited_not_applied": audited_thresholds,
    }, state_path)
    reload_network = _load_network(checkpoint, config_path)
    reload_audit = _load_and_apply_action_state(reload_network, checkpoint, state_path)
    if not _exact_authorized_match(network, reload_network):
        raise RuntimeError("head-only v2 state reload parity failed")
    if _state_hash_excluding(network, AUTHORIZED_PREFIXES) != frozen_before:
        raise RuntimeError("head-only v2 changed frozen backbone")
    candidate_path = output_dir / "17_head_only_retrain_v2_candidate.pkl"
    config = load_config(config_path)
    system = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint, config, policy_core="current", seed=0)
    candidate_network = shared_five_network(system)
    _load_and_apply_action_state(candidate_network, checkpoint, state_path)
    _atomic_pickle(system, candidate_path)
    candidate_reloaded = _load_network(candidate_path, config_path)
    candidate_reload_match = _exact_authorized_match(network, candidate_reloaded)
    if not candidate_reload_match:
        raise RuntimeError("full candidate checkpoint reload parity failed")
    validation_roles = post_calibration["validation"]["roles"]
    head_gate_passed = all(
        float(metrics["balanced_accuracy"]) >= 0.52
        and float(metrics["mcc"]) > 0.0
        and float(metrics["entry_precision"]) >= max(0.30, 1.20 * float(metrics["label_entry_rate"]))
        and float(metrics["predicted_entry_rate"]) <= 1.10 * float(metrics["label_entry_rate"])
        for metrics in validation_roles.values()
    )
    source_after = _sha256(checkpoint)
    if source_after != source_before:
        raise RuntimeError("head-only v2 changed source checkpoint")
    payload = {
        "stage": 17,
        "name": "head_only_retrain_v2_after_class_weight_ablation",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_before,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "class_weight_winner": mode,
        "sequence_length": int(sequence_length),
        "window_count_per_split": int(windows),
        "epochs_requested": int(epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "chronological_split_contract": True,
        "best_epoch_selected_on_validation_only": True,
        "operating_points_measured_on_validation_only": True,
        "test_used_for_final_evaluation_only": True,
        "validation_operating_points_audited_not_applied": audited_thresholds,
        "training": result,
        "pre_calibration_operating_point": pre_calibration,
        "post_calibration": post_calibration,
        "head_validation_gate_passed": bool(head_gate_passed),
        "candidate_only": True,
        "promotable": False,
        "promotion_blocker": "requires paired MT5/OOS evidence even when head gate passes",
        "state_artifact": str(state_path),
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": _sha256(candidate_path),
        "reload_state_exact": True,
        "full_candidate_reload_exact": bool(candidate_reload_match),
        "reload_audit": reload_audit,
        "frozen_backbone_unchanged": True,
        "source_checkpoint_unchanged": True,
    }
    path = output_dir / "17_head_only_retrain_v2.json"
    _atomic_json(payload, path)
    print(json.dumps({
        "artifact": str(path),
        "candidate_checkpoint": str(candidate_path),
        "candidate_checkpoint_sha256": payload["candidate_checkpoint_sha256"],
        "class_weight_winner": mode,
        "best_epoch": result["best_epoch"],
        "head_validation_gate_passed": head_gate_passed,
        "validation": post_calibration["validation"],
        "test": post_calibration["test"],
    }, ensure_ascii=False, indent=2))
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--stage13-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--windows", type=int, default=16)
    parser.add_argument("--ablation-epochs", type=int, default=40)
    parser.add_argument("--ablation-patience", type=int, default=8)
    parser.add_argument("--retrain-windows", type=int, default=32)
    parser.add_argument("--retrain-epochs", type=int, default=80)
    parser.add_argument("--retrain-patience", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--start-stage", type=int, choices=(14, 15, 16, 17), default=14)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    dataset = args.dataset.resolve()
    state = args.stage13_state.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.start_stage <= 14:
        run_production_stage13_parity(
            checkpoint, config, dataset, state, output_dir,
            sequence_length=max(2, args.sequence_length), windows=max(1, args.windows),
        )
    if args.start_stage <= 15:
        run_stage13_margin_sweep(
            checkpoint, config, dataset, state, output_dir,
            sequence_length=max(2, args.sequence_length), windows=max(1, args.windows),
        )
    if args.start_stage <= 16:
        run_class_weight_ablation(
            checkpoint, config, dataset, output_dir,
            sequence_length=max(2, args.sequence_length), windows=max(1, args.windows),
            epochs=max(1, args.ablation_epochs), patience=max(1, args.ablation_patience),
            learning_rate=float(args.learning_rate),
        )
    if args.start_stage <= 17:
        run_head_only_retrain_v2(
            checkpoint, config, dataset, output_dir,
            sequence_length=max(2, args.sequence_length), windows=max(1, args.retrain_windows),
            epochs=max(1, args.retrain_epochs), patience=max(1, args.retrain_patience),
            learning_rate=float(args.learning_rate),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
