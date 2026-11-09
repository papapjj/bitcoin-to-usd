#!/usr/bin/env python3
"""Build representative head-fit features at exact production 64-step boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.train_temporal_v3_ablation_candidate import _sha256
from scripts.validate_temporal_head_full import subset
from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import (
    load_teacher_imitation_data, shared_five_network, extract_owner_hidden_sequences,
)
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


def aligned_rows(data, count, length=64):
    """Uniform nonoverlapping full windows, aligned from the start of this split."""
    if count < 1:
        raise ValueError("window count must be positive")
    starts = np.flatnonzero(np.r_[True, data.sample_times[1:] != data.sample_times[:-1]])
    endpoints = np.r_[starts, len(data.actions)]
    available = len(starts) // length
    if not available:
        raise ValueError("split has no complete replay window")
    chosen = np.unique(np.linspace(0, available - 1, min(count, available), dtype=np.int64))
    rows = np.concatenate([np.arange(endpoints[index * length], endpoints[(index + 1) * length]) for index in chosen])
    return rows, chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "dataset", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--train-windows", type=int, default=128)
    parser.add_argument("--validation-windows", type=int, default=64)
    parser.add_argument("--batch-windows", type=int, default=32)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    hashes = {name + "_sha256": _sha256(getattr(args, name)) for name in ("checkpoint", "config", "dataset")}
    data = load_teacher_imitation_data(args.dataset)
    network = shared_five_network(tai_he_chuyen_gia_mot_context_mot_policy(
        args.checkpoint.resolve(), load_config(args.config), policy_core="current", seed=0)).cpu().eval()
    arrays, splits = {}, {}
    for split, count in (("train", args.train_windows), ("validation", args.validation_windows)):
        split_data = subset(data, np.flatnonzero(data.splits == split))
        rows, windows = aligned_rows(split_data, count)
        selected = subset(split_data, rows)
        print(json.dumps({"event": "production_cache_replay", "split": split, "windows": len(windows), "rows": len(rows)}), flush=True)
        hidden, sequence_audit = extract_owner_hidden_sequences(
            network, selected, batch_size=args.batch_windows, sequence_length=64, representation="hidden")
        splits[split] = {"rows": len(rows), "window_indices": windows.tolist(), "sequence_audit": sequence_audit}
        for role in (0, 1):
            mask = selected.agent_ids == role
            arrays[f"{split}_{role}_temporal_on"] = hidden.numpy()[mask]
            arrays[f"{split}_{role}_labels"] = (selected.actions[mask] != 0).astype(np.int64)
            arrays[f"{split}_{role}_sample_times"] = selected.sample_times[mask]
    for name in ("checkpoint", "config", "dataset"):
        if _sha256(getattr(args, name)) != hashes[name + "_sha256"]:
            raise ValueError(name + " changed during extraction")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    report = {"schema_version": 1, "status": "complete", **hashes,
              "feature_cache_sha256": _sha256(args.output), "sequence_length": 64, "burn_in": 0,
              "sampling": "uniform_complete_windows_aligned_to_production_split_origin",
              "representation": "production_policy_hidden", "splits": splits,
              "test_evaluated": False, "source_unchanged": True, "dataset_unchanged": True}
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"event": "production_cache_complete", "rows": {k: v["rows"] for k, v in splits.items()}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
