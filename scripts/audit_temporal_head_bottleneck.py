#!/usr/bin/env python3
"""Locate temporal fusion/query/calibration signal loss on train/validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.audit_five_architecture import _sample_split_windows, _replay_raw_hidden_cube
from scripts.train_temporal_v3_ablation_candidate import _metrics, _select_train_boundary, _sha256
from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import _role_logits, shared_five_network
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


def collect(network, hidden, sampled):
    block = network.post_lstm_transformer
    batch = len(hidden)
    tokens, valid = block.empty_temporal_memory(batch, device=hidden.device, dtype=hidden.dtype)
    arrays = {role: {name: [] for name in (
        "raw_lstm", "temporal_off", "temporal_on", "action_query", "labels", "margin"
    )} for role in (0, 1)}
    agents = torch.arange(5).expand(batch, -1)
    with torch.inference_mode():
        for step in range(hidden.shape[1]):
            raw = hidden[:, step]
            if step >= 128:
                projected = block.input_projection(raw)
                off = raw + block.residual_scale * block.output_projection(block.encoder_layer(projected))
                on = block(raw, tokens, valid)
                query = network.heads.action_query(on)
                logits = _role_logits(query.reshape(batch * 5, -1), agents.reshape(-1), network)
                margins = (logits[:, 1] - logits[:, 0]).reshape(batch, 5)
                for role in (0, 1):
                    owner = sampled["owner_masks"][:, step, role]
                    arrays[role]["labels"].append(
                        (sampled["actions"][owner, step, role] == role + 1).astype(np.int64)
                    )
                    for name, values in (("raw_lstm", raw), ("temporal_off", off),
                                         ("temporal_on", on), ("action_query", query), ("margin", margins)):
                        arrays[role][name].append(values[owner, role].cpu().numpy())
            tokens, valid = block.update_temporal_memory(raw, tokens, valid)
    return {str(role): {name: np.concatenate(rows) for name, rows in group.items()}
            for role, group in arrays.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reuse-feature-cache", type=Path)
    parser.add_argument("--feature-cache-sha256")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    source_hash = _sha256(args.checkpoint)
    dataset_hash = _sha256(args.dataset)
    data = {}
    if args.reuse_feature_cache:
        if not args.feature_cache_sha256 or _sha256(args.reuse_feature_cache) != args.feature_cache_sha256.lower():
            raise ValueError("feature cache requires the exact recorded SHA-256")
        with np.load(args.reuse_feature_cache, allow_pickle=False) as archive:
            data = {split: {role: {name: archive[f"{split}_{role}_{name}"] for name in (
                "raw_lstm", "temporal_off", "temporal_on", "action_query", "labels", "margin"
            )} for role in ("0", "1")} for split in ("train", "validation")}
        for split in data:
            sampled = _sample_split_windows(args.dataset, split=split, sequence_length=256, window_count=8)
            for role in (0, 1):
                owners = sampled["owner_masks"][:, 128:, role].T
                expected = (sampled["actions"][:, 128:, role].T[owners] == role + 1).astype(int)
                if not np.array_equal(expected, data[split][str(role)]["labels"]):
                    raise ValueError("cache labels do not match the chronological dataset rows")
    else:
        network = shared_five_network(tai_he_chuyen_gia_mot_context_mot_policy(
            args.checkpoint.resolve(), load_config(args.config.resolve()), policy_core="current", seed=0
        )).cpu().eval()
        for split in ("train", "validation"):
            print(json.dumps({"event": "bottleneck_replay", "split": split}), flush=True)
            sampled = _sample_split_windows(args.dataset, split=split, sequence_length=256, window_count=8)
            data[split] = collect(network, torch.from_numpy(_replay_raw_hidden_cube(network, sampled)), sampled)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_feature_cache:
        np.savez_compressed(args.output.with_suffix(".features.npz"), **{
            f"{split}_{role}_{name}": value for split, groups in data.items()
            for role, group in groups.items() for name, value in group.items()
        })
    report = {"schema_version": 1, "scope": "train_validation_diagnostic_only", "test_read": False,
              "checkpoint_sha256": source_hash, "dataset_sha256": dataset_hash, "roles": {},
              "feature_cache_sha256": _sha256(args.reuse_feature_cache or args.output.with_suffix(".features.npz")),
              "feature_cache_reused": bool(args.reuse_feature_cache)}
    for role, role_name in (("0", "buy"), ("1", "sell")):
        train, val = data["train"][role], data["validation"][role]
        output = {}
        for name in ("raw_lstm", "temporal_off", "temporal_on", "action_query", "margin"):
            candidates = []
            for c in ((None,) if name == "margin" else (0.01, 0.1, 1.0)):
                if c is None:
                    train_score, val_score = train[name], val[name]
                else:
                    probe = make_pipeline(StandardScaler(), LogisticRegression(
                        C=c, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=0
                    ))
                    probe.fit(train[name], train["labels"])
                    train_score, val_score = probe.decision_function(train[name]), probe.decision_function(val[name])
                boundary, training = _select_train_boundary(train["labels"], train_score)
                validation = _metrics(val["labels"], (val_score > boundary).astype(int), val_score)
                validation["eligible"] = bool(
                    training["eligible"] and validation["balanced_accuracy"] >= .52
                    and validation["mcc"] >= .05 and validation["entry_recall"] >= .20
                    and validation["entry_precision"] >= .30
                    and validation["predicted_entry_rate"] <= 1.2 * validation["label_entry_rate"]
                )
                candidates.append({"C": c, "train_boundary": boundary, "train": training,
                                   "validation": validation})
            selected = max(candidates, key=lambda row: (
                row["validation"]["eligible"], row["validation"]["roc_auc"]
            ))
            output[name] = {"selected": selected, "candidates": candidates}
            print(json.dumps({"event": "bottleneck_probe", "role": role_name, "surface": name,
                              "selected": selected}), flush=True)
            report["roles"][role_name] = output
            args.output.with_suffix(".partial.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["roles"][role_name] = output
    report["source_unchanged"] = _sha256(args.checkpoint) == source_hash
    report["dataset_unchanged"] = _sha256(args.dataset) == dataset_hash
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
