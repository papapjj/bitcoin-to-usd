#!/usr/bin/env python3
"""Chronological OOS signal probe and 2/8/32/128 horizon ablation for Five v3."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for value in (ROOT, SRC):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from scripts.audit_five_architecture import _replay_raw_hidden_cube, _sample_split_windows  # noqa: E402
from self_growing_trader.config import load_config  # noqa: E402
from self_growing_trader.five_shared_policy import (  # noqa: E402
    FIVE_AGENT_COUNT,
    FIVE_TEMPORAL_TOKEN_HORIZONS,
)
from self_growing_trader.five_teacher_imitation import _role_logits, shared_five_network  # noqa: E402
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy  # noqa: E402


TEMPORAL_HORIZON_PROBE_SCHEMA_VERSION = 1


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predictions = (scores > 0.0).astype(np.int64)
    return {
        "samples": int(labels.size),
        "entry_samples": int(labels.sum()),
        "label_entry_rate": float(labels.mean()),
        "predicted_entry_rate": float(predictions.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def _horizon_masks() -> dict[str, torch.Tensor]:
    count = len(FIVE_TEMPORAL_TOKEN_HORIZONS)
    masks = {"all": torch.ones(count, dtype=torch.bool)}
    for index, horizon in enumerate(FIVE_TEMPORAL_TOKEN_HORIZONS):
        mask = torch.ones(count, dtype=torch.bool)
        mask[index] = False
        masks[f"drop_{horizon}"] = mask
    return masks


def _collect_split(
    network: torch.nn.Module,
    hidden: torch.Tensor,
    sampled: dict[str, np.ndarray],
    *,
    burn_in: int,
) -> dict[str, object]:
    block = network.post_lstm_transformer
    batch = int(hidden.shape[0])
    dtype = next(network.parameters()).dtype
    memory, valid = block.empty_temporal_memory(batch, device=torch.device("cpu"), dtype=dtype)
    agent_grid = torch.arange(FIVE_AGENT_COUNT).expand(batch, -1)
    masks = _horizon_masks()
    role_rows: dict[int, dict[str, list[np.ndarray] | dict[str, list[np.ndarray]]]] = {
        role: {
            "labels": [], "current": [],
            "tokens": {str(horizon): [] for horizon in FIVE_TEMPORAL_TOKEN_HORIZONS},
            "margins": {"off": [], **{name: [] for name in masks}},
        }
        for role in (0, 1)
    }
    network.eval()
    with torch.inference_mode():
        for step in range(int(hidden.shape[1])):
            current_hidden = hidden[:, step].to(dtype=dtype)
            projected = block.input_projection(current_hidden)
            attended = block.encoder_layer(projected)
            base_hidden = current_hidden + block.residual_scale.to(dtype=dtype) * block.output_projection(attended)
            policy_by_arm: dict[str, torch.Tensor] = {"off": base_hidden}
            for name, mask in masks.items():
                enabled = valid & mask.reshape(1, 1, -1)
                policy_by_arm[name] = block(current_hidden, memory, enabled)
            if step >= burn_in:
                actions = torch.as_tensor(sampled["actions"][:, step], dtype=torch.long)
                owners = torch.as_tensor(sampled["owner_masks"][:, step], dtype=torch.bool)
                margin_by_arm: dict[str, torch.Tensor] = {}
                for name, policy_hidden in policy_by_arm.items():
                    queries = network.heads.action_query(policy_hidden).reshape(batch * FIVE_AGENT_COUNT, -1)
                    logits = _role_logits(queries, agent_grid.reshape(-1), network).reshape(batch, FIVE_AGENT_COUNT, 2)
                    margin_by_arm[name] = logits[..., 1] - logits[..., 0]
                for role, entry_action in ((0, 1), (1, 2)):
                    selected = owners[:, role]
                    if not bool(selected.any()):
                        continue
                    rows = role_rows[role]
                    rows["labels"].append((actions[selected, role] == entry_action).numpy().astype(np.int64))
                    rows["current"].append(projected[selected, role].numpy().astype(np.float32))
                    for index, horizon in enumerate(FIVE_TEMPORAL_TOKEN_HORIZONS):
                        rows["tokens"][str(horizon)].append(memory[selected, role, index].numpy().astype(np.float32))
                    for name, margin in margin_by_arm.items():
                        rows["margins"][name].append(margin[selected, role].numpy().astype(np.float32))
            memory, valid = block.update_temporal_memory(current_hidden, memory, valid)
    result: dict[str, object] = {}
    for role in (0, 1):
        rows = role_rows[role]
        result[str(role)] = {
            "labels": np.concatenate(rows["labels"]),
            "current": np.concatenate(rows["current"]),
            "tokens": {name: np.concatenate(values) for name, values in rows["tokens"].items()},
            "margins": {name: np.concatenate(values) for name, values in rows["margins"].items()},
        }
    return result


def _fit_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    validation_x: np.ndarray,
    validation_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
) -> dict[str, object]:
    candidates: list[tuple[float, float, object]] = []
    for c_value in (0.01, 0.1, 1.0):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=c_value, class_weight="balanced", max_iter=1000,
                solver="liblinear", random_state=0,
            ),
        )
        model.fit(train_x, train_y)
        validation_score = model.decision_function(validation_x)
        candidates.append((float(roc_auc_score(validation_y, validation_score)), c_value, model))
    validation_auc, selected_c, selected = max(candidates, key=lambda row: (row[0], -row[1]))
    return {
        "selected_c_on_validation": float(selected_c),
        "validation": _metrics(validation_y, selected.decision_function(validation_x)),
        "test": _metrics(test_y, selected.decision_function(test_x)),
    }


def run_probe(
    checkpoint: Path,
    config_path: Path,
    dataset: Path,
    output: Path,
    *,
    sequence_length: int,
    burn_in: int,
    windows_per_split: int,
) -> dict[str, object]:
    if sequence_length <= burn_in or burn_in < 128:
        raise ValueError("require sequence_length > burn_in >= 128")
    source_hash = _sha256(checkpoint)
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        checkpoint, load_config(config_path), policy_core="current", seed=0
    )
    network = shared_five_network(system).cpu().eval()
    network.require_current_architecture()
    split_rows: dict[str, dict[str, object]] = {}
    for split in ("train", "validation", "test"):
        sampled = _sample_split_windows(
            dataset, split=split, sequence_length=sequence_length,
            window_count=windows_per_split,
        )
        print(json.dumps({"event": "temporal_probe_cache_started", "split": split}), flush=True)
        hidden = torch.from_numpy(_replay_raw_hidden_cube(network, sampled))
        split_rows[split] = _collect_split(network, hidden, sampled, burn_in=burn_in)
        print(json.dumps({"event": "temporal_probe_split_complete", "split": split}), flush=True)

    roles: dict[str, object] = {}
    signal_checks: list[bool] = []
    for role_name, role in (("buy", 0), ("sell", 1)):
        train = split_rows["train"][str(role)]
        validation = split_rows["validation"][str(role)]
        test = split_rows["test"][str(role)]
        baseline = _fit_probe(
            train["current"], train["labels"], validation["current"], validation["labels"],
            test["current"], test["labels"],
        )
        horizons: dict[str, object] = {}
        for horizon in FIVE_TEMPORAL_TOKEN_HORIZONS:
            key = str(horizon)
            token_probe = _fit_probe(
                train["tokens"][key], train["labels"], validation["tokens"][key], validation["labels"],
                test["tokens"][key], test["labels"],
            )
            combined_probe = _fit_probe(
                np.concatenate((train["current"], train["tokens"][key]), axis=1), train["labels"],
                np.concatenate((validation["current"], validation["tokens"][key]), axis=1), validation["labels"],
                np.concatenate((test["current"], test["tokens"][key]), axis=1), test["labels"],
            )
            horizons[key] = {"token_only": token_probe, "current_plus_token": combined_probe}
        selected_horizon = max(
            horizons,
            key=lambda key: float(horizons[key]["current_plus_token"]["validation"]["roc_auc"]),
        )
        selected_test = horizons[selected_horizon]["current_plus_token"]["test"]
        test_uplift = float(selected_test["roc_auc"] - baseline["test"]["roc_auc"])
        passed = bool(
            int(selected_test["samples"]) >= 50
            and int(selected_test["entry_samples"]) >= 50
            and float(horizons[selected_horizon]["token_only"]["test"]["roc_auc"]) >= 0.52
            and test_uplift >= 0.005
        )
        signal_checks.append(passed)
        fixed_head = {
            name: {
                "validation": _metrics(validation["labels"], validation["margins"][name]),
                "test": _metrics(test["labels"], test["margins"][name]),
            }
            for name in validation["margins"]
        }
        roles[role_name] = {
            "current_probe": baseline,
            "horizons": horizons,
            "selected_horizon_on_validation": int(selected_horizon),
            "selected_test_auc_uplift_vs_current": test_uplift,
            "temporal_signal_gate_passed": passed,
            "fixed_head_horizon_ablation": fixed_head,
        }
    report = {
        "schema_version": TEMPORAL_HORIZON_PROBE_SCHEMA_VERSION,
        "created_at": time.time(),
        "status": "chronological_oos_probe",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": source_hash,
        "source_checkpoint_unchanged": _sha256(checkpoint) == source_hash,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "sequence_length": int(sequence_length),
        "burn_in": int(burn_in),
        "windows_per_split": int(windows_per_split),
        "selection_scope": "validation_only",
        "test_used_for_selection": False,
        "horizons": list(FIVE_TEMPORAL_TOKEN_HORIZONS),
        "roles": roles,
        "temporal_signal_gate_passed": all(signal_checks),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--burn-in", type=int, default=128)
    parser.add_argument("--windows-per-split", type=int, default=8)
    args = parser.parse_args()
    report = run_probe(
        args.checkpoint.resolve(), args.config.resolve(), args.dataset.resolve(), args.output.resolve(),
        sequence_length=args.sequence_length, burn_in=args.burn_in,
        windows_per_split=args.windows_per_split,
    )
    print(json.dumps({
        "event": "temporal_horizon_probe_complete",
        "output": str(args.output.resolve()),
        "temporal_signal_gate_passed": report["temporal_signal_gate_passed"],
    }, ensure_ascii=False, indent=2))
    return 0 if bool(report["temporal_signal_gate_passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
