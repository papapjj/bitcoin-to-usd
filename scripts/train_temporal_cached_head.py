#!/usr/bin/env python3
"""Fit the existing action query/calibrators in standardized coordinates.

The train-only affine transform is folded into action_query before export.
No model architecture or inference input contract changes. Subset validation
is research evidence and does not certify full imitation schema 10.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.train_temporal_v3_ablation_candidate import _atomic_pickle, _metrics, _sha256, _select_train_boundary
from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import _role_logits, shared_five_network
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


def owned(name):
    return name.startswith(("heads.action_query.", "heads.entry_role_calibrator.role_calibrators."))


def frozen_hash(network):
    digest = hashlib.sha256()
    for name, value in sorted(network.state_dict().items()):
        if not owned(name):
            digest.update(name.encode())
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def standardize_linear(layer, mean, scale):
    with torch.no_grad():
        layer.bias.add_(layer.weight @ mean)
        layer.weight.mul_(scale.unsqueeze(0))


def fold_linear(layer, mean, scale):
    with torch.no_grad():
        layer.weight.div_(scale.unsqueeze(0))
        layer.bias.sub_(layer.weight @ mean)


def evaluate(network, features, labels, roles):
    with torch.inference_mode():
        query = network.heads.action_query(features)
        logits = _role_logits(query, roles, network)
        margin = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    result = {}
    for role, name in ((0, "buy"), (1, "sell")):
        mask = roles.numpy() == role
        row = _metrics(labels.numpy()[mask], (margin[mask] > 0).astype(int), margin[mask])
        row["eligible"] = bool(row["samples"] >= 50 and row["balanced_accuracy"] >= .52
            and row["mcc"] >= .05 and row["entry_recall"] >= .20
            and row["entry_precision"] >= .30
            and row["predicted_entry_rate"] <= 1.2 * row["label_entry_rate"])
        result[name] = row
    return result, margin


def selection(metrics):
    rows = list(metrics.values())
    return (sum(row["eligible"] for row in rows), min(row["roc_auc"] for row in rows), min(row["mcc"] for row in rows),
            min(row["balanced_accuracy"] for row in rows), min(row["entry_precision"] for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--feature-cache", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4096)
    state_mode = parser.add_mutually_exclusive_group()
    state_mode.add_argument("--finalize-best-state", type=Path)
    state_mode.add_argument("--warm-start-best-state", type=Path)
    parser.add_argument("--fit-train-boundary", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.manual_seed(23)
    np.random.seed(23)
    source_hash = _sha256(args.checkpoint)
    cache_hash = _sha256(args.feature_cache)
    audit = json.loads(args.audit_report.read_text(encoding="utf-8"))
    if audit["checkpoint_sha256"] != source_hash or audit["feature_cache_sha256"] != cache_hash:
        raise ValueError("checkpoint/cache does not match completed bottleneck audit")
    config_hash = _sha256(args.config)
    if audit.get("config_sha256", config_hash) != config_hash:
        raise ValueError("config differs from the feature extraction config")
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        args.checkpoint.resolve(), load_config(args.config.resolve()), policy_core="current", seed=0)
    network = shared_five_network(system).cpu().eval()
    before_hash = frozen_hash(network)
    for name, parameter in network.named_parameters():
        parameter.requires_grad_(owned(name))
    data = {}
    with np.load(args.feature_cache, allow_pickle=False) as archive:
        for split in ("train", "validation"):
            data[split] = (
                torch.from_numpy(np.concatenate([archive[f"{split}_{role}_temporal_on"] for role in (0, 1)])),
                torch.from_numpy(np.concatenate([archive[f"{split}_{role}_labels"] for role in (0, 1)])),
                torch.from_numpy(np.concatenate([np.full(len(archive[f"{split}_{role}_labels"]), role)
                                                for role in (0, 1)])),
            )
    mean = data["train"][0].mean(0)
    scale = data["train"][0].std(0, unbiased=False).clamp_min(1e-3)
    initial, original_margin = evaluate(network, *data["validation"])
    standardize_linear(network.heads.action_query, mean, scale)
    normalized = {split: ((values[0] - mean) / scale, values[1], values[2])
                  for split, values in data.items()}
    _, normalized_margin = evaluate(network, *normalized["validation"])
    initial_parity = float(np.max(np.abs(original_margin - normalized_margin)))
    if initial_parity > 1e-4:
        raise RuntimeError(f"standardization changed logits: {initial_parity}")
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=1e-3)
    best_key, best_epoch = selection(initial), 0
    best = {n: p.detach().clone() for n, p in network.state_dict().items() if owned(n)}
    history, stale = [], 0
    state_path = args.finalize_best_state or args.warm_start_best_state
    if state_path:
        with state_path.open("rb") as handle:
            saved = pickle.load(handle)
        if saved["source_checkpoint_sha256"] != source_hash or saved["cache_sha256"] != cache_hash:
            raise ValueError("best state has different checkpoint/cache provenance")
        torch.testing.assert_close(saved["mean"], mean, rtol=0, atol=0)
        torch.testing.assert_close(saved["scale"], scale, rtol=0, atol=0)
        best, best_epoch, history = saved["best_state"], saved["best_epoch"], saved["history"]
        network.load_state_dict({**network.state_dict(), **best})
        warm_metrics, _ = evaluate(network, *normalized["validation"])
        best_key = selection(warm_metrics)
    train_x, train_y, train_roles = normalized["train"]
    epoch_offset = len(history)
    for epoch in range(epoch_offset + 1, epoch_offset + (0 if args.finalize_best_state else args.epochs) + 1):
        losses = []
        for index in torch.randperm(len(train_y)).split(args.batch_size):
            logits = _role_logits(network.heads.action_query(train_x[index]), train_roles[index], network)
            # Natural class frequencies make the zero boundary a probability
            # decision, without balancing the prior then undoing it via bias.
            loss = torch.nn.functional.cross_entropy(logits, train_y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in network.parameters() if p.requires_grad], 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        validation, _ = evaluate(network, *normalized["validation"])
        key = selection(validation)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "validation": validation})
        if epoch == 1 or epoch % 10 == 0 or any(row["eligible"] for row in validation.values()):
            print(json.dumps({"event": "cached_head_epoch", **history[-1]}), flush=True)
        if key > best_key:
            best_key, best_epoch, stale = key, epoch, 0
            best = {n: p.detach().clone() for n, p in network.state_dict().items() if owned(n)}
        else:
            stale += 1
        _atomic_pickle({"status": "research_state_only", "best_state": best, "best_epoch": best_epoch,
                        "history": history, "mean": mean, "scale": scale,
                        "source_checkpoint_sha256": source_hash, "cache_sha256": cache_hash},
                       args.output.with_suffix(".best_state.pkl"))
        if stale >= args.patience:
            break
    network.load_state_dict({**network.state_dict(), **best})
    selected, expected_margin = evaluate(network, *normalized["validation"])
    fold_linear(network.heads.action_query, mean, scale)
    final, actual_margin = evaluate(network, *data["validation"])
    folding_parity = float(np.max(np.abs(expected_margin - actual_margin)))
    if folding_parity > 1e-4 or not np.array_equal(expected_margin > 0, actual_margin > 0):
        raise RuntimeError(f"folding changed runtime decisions: {folding_parity}")
    if frozen_hash(network) != before_hash or _sha256(args.checkpoint) != source_hash:
        raise RuntimeError("frozen backbone/source changed")
    boundary_audit = {}
    before_boundary = final
    if args.fit_train_boundary:
        _, train_margin = evaluate(network, *data["train"])
        for role, name in ((0, "buy"), (1, "sell")):
            mask = data["train"][2].numpy() == role
            threshold, metrics = _select_train_boundary(data["train"][1].numpy()[mask], train_margin[mask])
            applied = bool(metrics["eligible"])
            if applied:
                with torch.no_grad():
                    network.heads.entry_role_calibrator.role_calibrators[role].output.bias[1].sub_(threshold)
            boundary_audit[name] = {"applied": applied, "train": metrics, "threshold": threshold}
        final, _ = evaluate(network, *data["validation"])
    if (_sha256(args.config) != config_hash or _sha256(args.feature_cache) != cache_hash
            or _sha256(args.checkpoint) != source_hash or frozen_hash(network) != before_hash):
        raise RuntimeError("feature cache/config/source/frozen backbone changed during head fit")
    _atomic_pickle(system, args.output)
    report = {"schema_version": 1, "status": "research_candidate_only", "promotable": False,
              "test_read": False, "full_schema10_passed": False,
              "checkpoint": str(args.output), "checkpoint_sha256": _sha256(args.output),
              "source_checkpoint_sha256": source_hash, "source_unchanged": True, "frozen_backbone_unchanged": True,
              "cache_sha256": cache_hash, "normalizer_fit": "train_only_folded_into_existing_action_query",
              "feature_replay": {key: audit[key] for key in (
                  "sequence_length", "burn_in", "sampling", "representation", "splits"
              ) if key in audit},
              "config_sha256": config_hash,
              "learning_rate": args.learning_rate, "batch_size": args.batch_size,
              "warm_start_state": str(args.warm_start_best_state) if args.warm_start_best_state else None,
              "warm_start_optimizer_reset": bool(args.warm_start_best_state),
              "initial_parity_max_abs": initial_parity, "folding_parity_max_abs": folding_parity,
              "best_epoch": best_epoch, "history": history, "initial": initial, "validation": final,
              "validation_before_train_boundary": before_boundary, "train_boundary": boundary_audit,
              "subset_gate_passed": all(row["eligible"] for row in final.values())}
    args.output.with_suffix(".head_fit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "cached_head_complete", "best_epoch": best_epoch,
                      "subset_gate_passed": report["subset_gate_passed"], "validation": final}), flush=True)
    return 0 if report["subset_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
