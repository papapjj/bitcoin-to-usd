#!/usr/bin/env python3
"""Separate validation sampling shift from recurrent replay shift, without tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.audit_five_architecture import _sample_split_windows
from scripts.train_temporal_v3_ablation_candidate import _metrics, _sha256
from scripts.validate_temporal_head_full import scores, validate_chunk
from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import shared_five_network
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


def match_rows(times, agents, sampled, role):
    """Cache collection is step-major, then window-major, restricted to owners."""
    lookup = {(str(time), int(agent)): row for row, (time, agent) in enumerate(zip(times, agents))}
    if len(lookup) != len(times):
        raise ValueError("duplicate timestamp/owner rows")
    mask = sampled["owner_masks"][:, 128:, role].T
    selected_times = sampled["sample_times"][:, 128:].T[mask]
    return np.asarray([lookup[(str(time), role)] for time in selected_times], dtype=np.int64)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "config", "dataset", "full-validation-dir", "feature-cache", "cache-audit", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    full = json.loads((args.full_validation_dir / "report.json").read_text(encoding="utf-8"))
    audit = json.loads(args.cache_audit.read_text(encoding="utf-8"))
    contract = full["contract"]
    if full["status"] != "full_validation_complete" or not full["checks"]["all_validation_rows"]:
        raise ValueError("complete full validation required")
    for name in ("checkpoint", "config", "dataset"):
        if _sha256(getattr(args, name)) != contract[name + "_sha256"]:
            raise ValueError(name + " hash mismatch")
    if (audit["checkpoint_sha256"] != contract["baseline_sha256"]
            or audit["dataset_sha256"] != contract["dataset_sha256"]
            or audit["feature_cache_sha256"] != _sha256(args.feature_cache)):
        raise ValueError("cache provenance mismatch")
    with np.load(args.dataset, allow_pickle=False) as archive:
        mask = archive["splits"] == "validation"
        times, agents, labels = archive["sample_times"][mask], archive["agent_ids"][mask], (archive["actions"][mask] != 0).astype(int)
    parts, stop = [], 0
    for path in sorted(args.full_validation_dir.glob("chunk_*.pkl")):
        with path.open("rb") as handle:
            chunk = pickle.load(handle)
        validate_chunk(chunk, stop, chunk["stop"])
        stop = chunk["stop"]
        parts.append(chunk["candidate"])
    if stop != len(labels):
        raise ValueError("incomplete validation score coverage")
    full_margin = np.concatenate(parts)
    network = shared_five_network(tai_he_chuyen_gia_mot_context_mot_policy(
        args.checkpoint.resolve(), load_config(args.config), policy_core="current", seed=0)).cpu().eval()
    sampled = _sample_split_windows(args.dataset, split="validation", sequence_length=256, window_count=8)
    result = {"contract": contract, "test_evaluated": False, "threshold_tuned": False,
              "scope": "development_validation_replay_and_sampling_diagnostic", "roles": {}}
    with np.load(args.feature_cache, allow_pickle=False) as archive:
        for role, name in ((0, "buy"), (1, "sell")):
            rows = match_rows(times, agents, sampled, role)
            y = archive[f"validation_{role}_labels"]
            if not np.array_equal(labels[rows], y):
                raise ValueError("matched cache labels differ")
            warm = scores(network, torch.from_numpy(archive[f"validation_{role}_temporal_on"]),
                          torch.full((len(rows),), role, dtype=torch.long))
            cold = full_margin[rows]
            blocks = []
            for indices in np.array_split(np.arange(len(labels)), 4):
                indices = indices[agents[indices] == role]
                blocks.append({"first_time": str(times[indices[0]]), "last_time": str(times[indices[-1]]),
                               "metrics": _metrics(labels[indices], (full_margin[indices] > 0).astype(int), full_margin[indices])})
            result["roles"][name] = {
                "same_rows_warm256_burn128": _metrics(y, (warm > 0).astype(int), warm),
                "same_rows_production_reset64": _metrics(y, (cold > 0).astype(int), cold),
                "decision_flip_rate": float(np.mean((warm > 0) != (cold > 0))),
                "margin_abs_difference_mean": float(np.mean(np.abs(warm - cold))),
                "margin_abs_difference_max": float(np.max(np.abs(warm - cold))),
                "chronological_quarters_reset64": blocks,
            }
    result["checkpoint_unchanged"] = _sha256(args.checkpoint) == contract["checkpoint_sha256"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
