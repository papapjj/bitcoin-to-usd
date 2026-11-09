#!/usr/bin/env python3
"""Train a frozen-backbone Five v3 temporal branch for controlled OOS ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.audit_five_architecture import (  # noqa: E402
    _replay_raw_hidden_cube,
    _sample_split_windows,
)
from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.five_shared_policy import FIVE_AGENT_COUNT  # noqa: E402
from self_growing_trader.five_teacher_imitation import (  # noqa: E402
    _role_logits,
    shared_five_network,
)
from self_growing_trader.self_growing_runner import (  # noqa: E402
    tai_he_chuyen_gia_mot_context_mot_policy,
)


TEMPORAL_ABLATION_TRAIN_SCHEMA_VERSION = 10


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _temporal_parameter(name: str) -> bool:
    return name == "post_lstm_transformer.temporal_scale" or name.startswith((
        "post_lstm_transformer.temporal_encoder_layer.",
        "post_lstm_transformer.temporal_output_projection.",
        "post_lstm_transformer.temporal_horizon_embedding.",
        "post_lstm_transformer.temporal_agent_embedding.",
    ))


def _trainable_parameter(name: str) -> bool:
    return (
        _temporal_parameter(name)
        or name.startswith("heads.action_query.")
        or name.startswith("heads.entry_role_calibrator.role_calibrators.")
    )


def _state_hash_excluding_trainable(network: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        if _trainable_parameter(name):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _forward_window_batch(
    network: torch.nn.Module,
    hidden: torch.Tensor,
    sampled: dict[str, np.ndarray],
    window_indices: np.ndarray,
    *,
    burn_in: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block = network.post_lstm_transformer
    device = next(network.parameters()).device
    dtype = next(network.parameters()).dtype
    batch_hidden = hidden[window_indices].to(device=device, dtype=dtype)
    batch = int(batch_hidden.shape[0])
    temporal_tokens, temporal_valid = block.empty_temporal_memory(
        batch, device=device, dtype=dtype
    )
    logits: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    roles: list[torch.Tensor] = []
    agent_grid = torch.arange(FIVE_AGENT_COUNT, device=device).expand(batch, -1)
    for step in range(int(batch_hidden.shape[1])):
        if step < int(burn_in):
            temporal_tokens, temporal_valid = block.update_temporal_memory(
                batch_hidden[:, step], temporal_tokens, temporal_valid
            )
            continue
        policy_hidden = block(
            batch_hidden[:, step], temporal_tokens, temporal_valid
        )
        temporal_tokens, temporal_valid = block.update_temporal_memory(
            batch_hidden[:, step], temporal_tokens, temporal_valid
        )
        action_query = network.heads.action_query(policy_hidden)
        action_query = action_query.reshape(batch * FIVE_AGENT_COUNT, -1)
        flat_agents = agent_grid.reshape(-1)
        role_logits = _role_logits(action_query, flat_agents, network).reshape(
            batch, FIVE_AGENT_COUNT, 2
        )
        actions = torch.as_tensor(
            sampled["actions"][window_indices, step], dtype=torch.long, device=device
        )
        owners = torch.as_tensor(
            sampled["owner_masks"][window_indices, step], dtype=torch.bool, device=device
        )
        role_mask = owners & (agent_grid < 2)
        if bool(role_mask.any().item()):
            selected_roles = agent_grid[role_mask]
            selected_actions = actions[role_mask]
            selected_labels = torch.where(
                selected_roles == 0,
                selected_actions == 1,
                selected_actions == 2,
            ).long()
            logits.append(role_logits[role_mask])
            labels.append(selected_labels)
            roles.append(selected_roles)
    return torch.cat(logits), torch.cat(labels), torch.cat(roles)


def _metrics(labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "entry_precision": float(precision_score(labels, predictions, zero_division=0)),
        "entry_recall": float(recall_score(labels, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "label_entry_rate": float(np.mean(labels)),
        "predicted_entry_rate": float(np.mean(predictions)),
        "samples": int(len(labels)),
        "entry_samples": int(np.sum(labels)),
    }


def _evaluate(
    network: torch.nn.Module,
    hidden: torch.Tensor,
    sampled: dict[str, np.ndarray],
    *,
    batch_size: int,
    burn_in: int = 0,
) -> dict[str, object]:
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_roles: list[torch.Tensor] = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, hidden.shape[0], batch_size):
            index = np.arange(start, min(start + batch_size, hidden.shape[0]))
            logits, labels, roles = _forward_window_batch(
                network, hidden, sampled, index, burn_in=burn_in
            )
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_roles.append(roles.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    roles = torch.cat(all_roles).numpy()
    scores = (logits[:, 1] - logits[:, 0]).numpy()
    predictions = (scores > 0.0).astype(np.int64)
    result: dict[str, object] = {
        "overall": _metrics(labels, predictions, scores),
        "roles": {},
    }
    for name, role in (("buy", 0), ("sell", 1)):
        mask = roles == role
        result["roles"][name] = _metrics(
            labels[mask], predictions[mask], scores[mask]
        )
    return result


def _select_train_boundary(
    labels: np.ndarray, scores: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Choose a role boundary from train rows without consulting validation.

    The returned value is folded into the corresponding role calibrator bias,
    so production still uses the native zero boundary.  Strict schema-10
    eligibility wins first; MCC and balanced accuracy break ties.
    """

    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if len(labels) != len(scores) or len(labels) == 0:
        raise ValueError("train boundary requires equally-sized non-empty rows")
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        raise ValueError("train boundary requires two finite-score classes")
    order = np.argsort(scores, kind="stable")
    ordered, counts = np.unique(scores[order], return_counts=True)
    epsilon = max(1e-7, float(np.ptp(scores)) * 1e-7)
    candidates = np.concatenate((
        [float(scores.min()) - epsilon],
        ordered,
    ))
    label_rate = float(np.mean(labels))
    # Exact cumulative counts at every distinct score; strict score > threshold
    # excludes the entire tied group. Avoid thousands of sklearn calls per sweep.
    cumulative_positive = np.cumsum(labels[order])[np.cumsum(counts) - 1]
    fn = np.r_[0, cumulative_positive].astype(np.float64)
    tn = np.r_[0, np.cumsum(counts) - cumulative_positive].astype(np.float64)
    tp, fp = float(labels.sum()) - fn, float((labels == 0).sum()) - tn
    recall = tp / (tp + fn)
    precision = tp / np.maximum(1.0, tp + fp)
    balanced = .5 * (recall + tn / (tn + fp))
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denominator, out=np.zeros_like(tp), where=denominator > 0)
    rate = (tp + fp) / len(labels)
    eligible = ((balanced >= .52) & (mcc >= .05) & (recall >= .20)
                & (precision >= .30) & (rate <= 1.20 * label_rate))
    selected = int(np.lexsort((
        -np.arange(len(candidates)), -np.abs(rate - label_rate),
        -np.maximum(0, rate - 1.20 * label_rate), recall, precision,
        balanced, mcc, eligible.astype(int),
    ))[-1])
    best_threshold = float(candidates[selected])
    best_metrics = _metrics(labels, (scores > best_threshold).astype(int), scores)
    best_metrics["eligible"] = bool(eligible[selected])
    return best_threshold, best_metrics


def _align_train_zero_boundaries(
    network: torch.nn.Module,
    hidden: torch.Tensor,
    sampled: dict[str, np.ndarray],
    *,
    batch_size: int,
    burn_in: int = 0,
) -> dict[str, object]:
    """Fold independent BUY/SELL train boundaries into calibrator biases."""

    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_roles: list[torch.Tensor] = []
    network.eval()
    with torch.inference_mode():
        for start in range(0, hidden.shape[0], batch_size):
            index = np.arange(start, min(start + batch_size, hidden.shape[0]))
            logits, labels, roles = _forward_window_batch(
                network, hidden, sampled, index, burn_in=burn_in
            )
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            all_roles.append(roles.cpu())
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    roles = torch.cat(all_roles).numpy()
    scores = (logits[:, 1] - logits[:, 0]).numpy()
    result: dict[str, object] = {}
    calibrator = network.heads.entry_role_calibrator
    for role_name, role in (("buy", 0), ("sell", 1)):
        selected = roles == role
        threshold, metrics = _select_train_boundary(labels[selected], scores[selected])
        applied = bool(metrics["eligible"])
        if applied:
            with torch.no_grad():
                calibrator.role_calibrators[role].output.bias[1].sub_(threshold)
        result[role_name] = {
            "folded_threshold": float(threshold) if applied else 0.0,
            "proposed_threshold": float(threshold),
            "applied": applied,
            "reason": "train_gate_passed" if applied else "no_eligible_train_boundary",
            "pre_fold_metrics": metrics,
        }
    return result


def _role_conditioned_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    roles: torch.Tensor,
    *,
    false_entry_penalty: float,
    ranking_margin: float,
    ranking_loss_weight: float,
    rate_calibration_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    margins = logits[:, 1] - logits[:, 0]
    role_losses: list[torch.Tensor] = []
    ranking_terms: list[torch.Tensor] = []
    rate_terms: list[torch.Tensor] = []
    for role in (0, 1):
        selected = roles == role
        if not bool(selected.any()):
            continue
        role_labels = labels[selected]
        role_margins = margins[selected]
        positive_count = torch.sum(role_labels == 1).to(dtype=role_margins.dtype)
        negative_count = torch.sum(role_labels == 0).to(dtype=role_margins.dtype)
        pos_weight = (negative_count / positive_count.clamp_min(1.0)).clamp(1.0, 8.0)
        role_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            role_margins,
            role_labels.to(dtype=role_margins.dtype),
            pos_weight=pos_weight,
        )
        entry_probability = torch.sigmoid(role_margins)
        if false_entry_penalty > 0.0:
            role_loss = role_loss + false_entry_penalty * torch.mean(
                (role_labels == 0).to(dtype=entry_probability.dtype) * entry_probability
            )
        role_losses.append(role_loss)
        positive = role_margins[role_labels == 1]
        negative = role_margins[role_labels == 0]
        if len(positive) and len(negative):
            pair_count = min(len(positive), len(negative), 256)
            ranking_terms.append(torch.nn.functional.softplus(
                ranking_margin - positive[:pair_count] + negative[:pair_count]
            ).mean())
        label_rate = role_labels.to(dtype=entry_probability.dtype).mean()
        rate_terms.append(torch.square(entry_probability.mean() - label_rate))
    if not role_losses:
        raise ValueError("role loss requires BUY or SELL owner rows")
    loss = torch.stack(role_losses).mean()
    if ranking_terms:
        loss = loss + ranking_loss_weight * torch.stack(ranking_terms).mean()
    if rate_terms:
        loss = loss + rate_calibration_weight * torch.stack(rate_terms).mean()
    return loss, {
        "classification": float(torch.stack(role_losses).mean().detach()),
        "ranking": float(torch.stack(ranking_terms).mean().detach()) if ranking_terms else 0.0,
        "rate_calibration": float(torch.stack(rate_terms).mean().detach()) if rate_terms else 0.0,
    }


def _selection_key(metrics: dict[str, object]) -> tuple[float, ...]:
    roles = list(dict(metrics["roles"]).values())
    eligible = [
        float(
            float(row["balanced_accuracy"]) >= 0.52
            and float(row["mcc"]) >= 0.05
            and float(row["entry_recall"]) >= 0.20
            and float(row["entry_precision"]) >= 0.30
            and float(row["predicted_entry_rate"])
            <= 1.20 * float(row["label_entry_rate"])
        )
        for row in roles
    ]
    return (
        sum(eligible),
        min(float(row["roc_auc"]) for row in roles),
        min(float(row["mcc"]) for row in roles),
        min(float(row["balanced_accuracy"]) for row in roles),
        min(float(row["entry_precision"]) for row in roles),
        min(float(row["entry_recall"]) for row in roles),
        -max(
            max(0.0, float(row["predicted_entry_rate"]) - 1.20 * float(row["label_entry_rate"]))
            for row in roles
        ),
        float(dict(metrics["overall"])["balanced_accuracy"]),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--burn-in", type=int, default=128)
    parser.add_argument("--windows-per-split", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--temporal-learning-rate", type=float, default=0.0002)
    parser.add_argument("--calibrator-learning-rate", type=float, default=0.002)
    parser.add_argument("--false-entry-penalty", type=float, default=0.50)
    parser.add_argument("--initial-temporal-scale", type=float, default=0.05)
    parser.add_argument("--ranking-loss-weight", type=float, default=0.50)
    parser.add_argument("--ranking-margin", type=float, default=2.0)
    parser.add_argument("--rate-calibration-weight", type=float, default=10.0)
    args = parser.parse_args()
    if args.burn_in < 0 or args.burn_in >= args.sequence_length:
        raise ValueError("burn-in must satisfy 0 <= burn-in < sequence-length")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.sequence_length < 128:
        raise ValueError("temporal v3 ablation training requires sequence-length >= 128")
    if args.windows_per_split < 4:
        raise ValueError("windows-per-split must be >= 4")
    torch.manual_seed(0)
    np.random.seed(0)
    source = args.checkpoint.resolve()
    source_hash = _sha256(source)
    config = load_config(args.config.resolve())
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        source, config, policy_core="current", seed=0
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    network.cpu().eval()
    if not np.isfinite(args.initial_temporal_scale) or args.initial_temporal_scale <= 0.0:
        raise ValueError("initial-temporal-scale must be finite and > 0")
    # A v2 checkpoint is intentionally migrated with scale zero for exact source
    # compatibility. Training from that zero gate starves the temporal encoder
    # of first-step gradients and lets the much smaller calibrator learn only a
    # role-wide offset. A candidate v3 must start from the same small non-zero
    # gate used by a freshly-created v3 network.
    with torch.no_grad():
        network.post_lstm_transformer.temporal_scale.fill_(
            float(args.initial_temporal_scale)
        )
    calibrator = network.heads.entry_role_calibrator
    calibrator.reset_teacher_mapping()
    calibrator.teacher_entry_scope = "all_entry_mass_v2"
    calibrator.teacher_imitation_trained = False
    frozen_before = _state_hash_excluding_trainable(network)

    sampled: dict[str, dict[str, np.ndarray]] = {}
    hidden: dict[str, torch.Tensor] = {}
    for split in ("train", "validation", "test"):
        sampled[split] = _sample_split_windows(
            args.dataset.resolve(), split=split,
            sequence_length=args.sequence_length,
            window_count=args.windows_per_split,
        )
        print(json.dumps({
            "event": "lstm_cache_started", "split": split,
            "windows": int(sampled[split]["observations"].shape[0]),
            "steps": int(args.sequence_length),
        }), flush=True)
        hidden[split] = torch.from_numpy(
            _replay_raw_hidden_cube(network, sampled[split])
        )
        print(json.dumps({
            "event": "lstm_cache_complete", "split": split,
            "shape": list(hidden[split].shape),
        }), flush=True)

    for parameter in network.parameters():
        parameter.requires_grad_(False)
    temporal_parameters = []
    action_query_parameters = []
    calibrator_parameters = []
    for name, parameter in network.named_parameters():
        if _temporal_parameter(name):
            parameter.requires_grad_(True)
            temporal_parameters.append(parameter)
        elif name.startswith("heads.action_query."):
            parameter.requires_grad_(True)
            action_query_parameters.append(parameter)
        elif name.startswith("heads.entry_role_calibrator.role_calibrators."):
            parameter.requires_grad_(True)
            calibrator_parameters.append(parameter)
    if not temporal_parameters or not action_query_parameters or not calibrator_parameters:
        raise RuntimeError("missing temporal/action-query/calibrator training surface")
    optimizer = torch.optim.AdamW([
        {"params": temporal_parameters, "lr": args.temporal_learning_rate},
        {"params": action_query_parameters, "lr": args.calibrator_learning_rate},
        {"params": calibrator_parameters, "lr": args.calibrator_learning_rate},
    ], weight_decay=1e-4)
    initial = {
        split: _evaluate(
            network, hidden[split], sampled[split], batch_size=args.batch_size,
            burn_in=args.burn_in,
        )
        for split in ("train", "validation")
    }
    best_state = {
        name: value.detach().cpu().clone()
        for name, value in network.state_dict().items()
        if _trainable_parameter(name)
    }
    best_key = _selection_key(initial["validation"])
    best_epoch = 0
    stale = 0
    history: list[dict[str, object]] = []
    gradient_seen: set[str] = set()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        order = np.random.permutation(hidden["train"].shape[0])
        losses = []
        network.eval()
        for start in range(0, len(order), args.batch_size):
            index = order[start:start + args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            logits, labels, roles = _forward_window_batch(
                network, hidden["train"], sampled["train"], index,
                burn_in=args.burn_in,
            )
            loss, _ = _role_conditioned_loss(
                logits,
                labels,
                roles,
                false_entry_penalty=args.false_entry_penalty,
                ranking_margin=args.ranking_margin,
                ranking_loss_weight=args.ranking_loss_weight,
                rate_calibration_weight=args.rate_calibration_weight,
            )
            loss.backward()
            for name, parameter in network.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    if not torch.isfinite(parameter.grad).all():
                        raise RuntimeError(f"non-finite gradient: {name}")
                    if float(parameter.grad.detach().abs().max().item()) > 0.0:
                        gradient_seen.add(name)
            torch.nn.utils.clip_grad_norm_(
                temporal_parameters + action_query_parameters + calibrator_parameters,
                max_norm=5.0,
            )
            optimizer.step()
            losses.append(float(loss.detach()))
        validation = _evaluate(
            network, hidden["validation"], sampled["validation"],
            batch_size=args.batch_size,
            burn_in=args.burn_in,
        )
        key = _selection_key(validation)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "validation": validation,
            "selection_key": list(key),
            "temporal_scale": float(
                network.post_lstm_transformer.temporal_scale.detach()
            ),
            "epoch_seconds": time.monotonic() - epoch_started,
        }
        history.append(row)
        print(json.dumps({"event": "temporal_epoch", **row}), flush=True)
        if key > best_key:
            best_key = key
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in network.state_dict().items()
                if _trainable_parameter(name)
            }
            stale = 0
        else:
            stale += 1
        # Persist bounded trainable tensors after every epoch so interruption
        # cannot discard the selected representation. This is research state,
        # not a production checkpoint or a schema-10 promotion certificate.
        _atomic_pickle({
            "status": "research_state_only",
            "source_checkpoint_sha256": source_hash,
            "dataset_sha256": _sha256(args.dataset.resolve()),
            "arguments": {name: str(value) if isinstance(value, Path) else value
                          for name, value in vars(args).items()},
            "best_state": best_state,
            "best_epoch": best_epoch,
            "history": history,
        }, args.output.with_suffix(args.output.suffix + ".best_state.pkl"))
        if stale >= args.patience:
            break

    state = network.state_dict()
    state.update(best_state)
    network.load_state_dict(state, strict=True)
    train_boundary_alignment = _align_train_zero_boundaries(
        network,
        hidden["train"],
        sampled["train"],
        batch_size=args.batch_size,
        burn_in=args.burn_in,
    )
    final = {
        split: _evaluate(
            network, hidden[split], sampled[split], batch_size=args.batch_size,
            burn_in=args.burn_in,
        )
        for split in ("train", "validation", "test")
    }
    validation_roles = list(dict(final["validation"]["roles"]).values())
    validation_gate = {
        "both_train_boundaries_eligible": all(
            bool(row["applied"]) for row in train_boundary_alignment.values()
        ),
        "minimum_role_samples": min(int(row["samples"]) for row in validation_roles) >= 50,
        "minimum_role_balanced_accuracy": min(
            float(row["balanced_accuracy"]) for row in validation_roles
        ) >= 0.52,
        "minimum_role_mcc": min(float(row["mcc"]) for row in validation_roles) >= 0.05,
        "minimum_role_recall": min(
            float(row["entry_recall"]) for row in validation_roles
        ) >= 0.20,
        "minimum_role_precision": min(
            float(row["entry_precision"]) for row in validation_roles
        ) >= 0.30,
        "maximum_role_entry_rate": all(
            float(row["predicted_entry_rate"])
            <= 1.20 * float(row["label_entry_rate"])
            for row in validation_roles
        ),
    }
    validation_gate["passed"] = all(validation_gate.values())
    temporal_gradient_seen = any(
        name.startswith("post_lstm_transformer.temporal_")
        for name in gradient_seen
    )
    calibrator_gradient_seen = all(any(
        f"role_calibrators.{role}." in name for name in gradient_seen
    ) for role in (0, 1))
    action_query_gradient_seen = any(
        name.startswith("heads.action_query.") for name in gradient_seen
    )
    frozen_after = _state_hash_excluding_trainable(network)
    if frozen_after != frozen_before or _sha256(source) != source_hash:
        raise RuntimeError("temporal ablation trainer changed frozen backbone or source")
    if not temporal_gradient_seen or not action_query_gradient_seen or not calibrator_gradient_seen:
        raise RuntimeError("missing temporal, action-query, or role-calibrator gradient")
    if abs(float(network.post_lstm_transformer.temporal_scale.detach())) <= 1e-8:
        raise RuntimeError("selected temporal candidate has zero temporal scale")
    calibrator.teacher_imitation_trained = True
    calibrator.teacher_imitation_schema_version = 7
    calibrator.teacher_imitation_projection = "temporal_v3_ablation_subset_128"
    _atomic_pickle(system, args.output.resolve())
    report = {
        "schema_version": TEMPORAL_ABLATION_TRAIN_SCHEMA_VERSION,
        "status": "evaluation_candidate_only",
        "promotable": False,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "source_checkpoint_unchanged": True,
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": _sha256(args.dataset.resolve()),
        "chronological_splits": True,
        "sequence_length": int(args.sequence_length),
        "burn_in": int(args.burn_in),
        "windows_per_split": int(args.windows_per_split),
        "initial_temporal_scale": float(args.initial_temporal_scale),
        "training_surface": [
            "post_lstm_transformer.temporal_*",
            "heads.action_query.*",
            "heads.entry_role_calibrator.role_calibrators.*",
        ],
        "frozen_surface_unchanged": True,
        "selection_scope": "validation_only",
        "test_used_for_selection": False,
        "best_epoch": int(best_epoch),
        "epochs_ran": len(history),
        "loss_contract": {
            "classification": "role_conditioned_balanced_bce",
            "positive_weight": "per_batch_per_role_clip_negative_over_positive_1_to_8",
            "ranking_margin": float(args.ranking_margin),
            "ranking_loss_weight": float(args.ranking_loss_weight),
            "rate_calibration_weight": float(args.rate_calibration_weight),
            "runtime_boundary": "role_specific_zero_train_bias_no_validation_threshold",
        },
        "gradient_nonzero_parameters": sorted(gradient_seen),
        "temporal_gradient_seen": temporal_gradient_seen,
        "action_query_gradient_seen": action_query_gradient_seen,
        "both_role_calibrator_gradients_seen": calibrator_gradient_seen,
        "initial": initial,
        "history": history,
        "train_boundary_alignment": train_boundary_alignment,
        "final": final,
        "validation_gate": validation_gate,
        "candidate_checkpoint": str(args.output.resolve()),
        "candidate_checkpoint_sha256": _sha256(args.output.resolve()),
        "temporal_scale": float(network.post_lstm_transformer.temporal_scale.detach()),
        "mt5_oos_proven": False,
    }
    report_path = args.output.with_suffix(args.output.suffix + ".temporal_ablation.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "temporal_v3_ablation_candidate_complete",
        "checkpoint": str(args.output.resolve()),
        "report": str(report_path.resolve()),
        "best_epoch": best_epoch,
        "validation_gate": validation_gate,
        "test": final["test"],
    }, ensure_ascii=False, indent=2))
    return 0 if bool(validation_gate["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
