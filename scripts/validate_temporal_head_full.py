#!/usr/bin/env python3
"""Resumable full validation using production imitation's 64-step replay.

Writes independent metric evidence; never fabricates full-BPTT training lineage
or an imitation.json certificate for a frozen-head research candidate.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import fields, replace
import json
from pathlib import Path
import pickle
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "src"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from scripts.train_temporal_v3_ablation_candidate import _atomic_pickle, _metrics, _sha256
from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import (
    load_teacher_imitation_data, shared_five_network, extract_owner_hidden_sequences, _role_logits,
)
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


def subset(data, index):
    return replace(data, **{field.name: getattr(data, field.name)[index]
        for field in fields(data) if isinstance(getattr(data, field.name), np.ndarray)})


def scores(network, hidden, agents):
    with torch.inference_mode():
        logits = _role_logits(hidden, agents, network)
    return (logits[:, 1] - logits[:, 0]).numpy()


def validate_chunk(chunk, start, stop):
    """Reject partial/corrupt resume evidence before computing any metric."""
    if chunk["start"] != start or chunk["stop"] != stop or stop <= start:
        raise ValueError("chunk coverage mismatch")
    for arm in ("candidate", "baseline"):
        values = np.asarray(chunk[arm])
        if values.shape != (stop - start,) or not np.isfinite(values).all():
            raise ValueError(f"invalid {arm} chunk scores")
    if chunk["audit"]["sequence_length"] != 64:
        raise ValueError("chunk replay sequence mismatch")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-windows", type=int, default=32)
    args = parser.parse_args()
    if args.batch_windows < 1:
        raise ValueError("batch-windows must be positive")
    contract = {"checkpoint_sha256": _sha256(args.checkpoint), "baseline_sha256": _sha256(args.baseline),
                "dataset_sha256": _sha256(args.dataset), "config_sha256": _sha256(args.config),
                "sequence_length": 64, "batch_windows": args.batch_windows,
                "split": "validation", "test_read": False,
                "replay": "production_extract_owner_hidden_sequences_teacher_forced_reset_64"}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    contract_path = args.output_dir / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("resume contract mismatch")
    else:
        contract_path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    data = load_teacher_imitation_data(args.dataset)
    data = subset(data, np.flatnonzero(data.splits == "validation"))
    if not len(data.actions) or not np.all(np.isin(data.agent_ids, [0, 1])):
        raise ValueError("validation must contain nonempty BUY/SELL owner rows only")
    for role in (0, 1):
        if len(np.unique(data.actions[data.agent_ids == role] != 0)) != 2:
            raise ValueError("validation needs both HOLD and ENTRY labels for each role")
    config = load_config(args.config)
    baseline_system = tai_he_chuyen_gia_mot_context_mot_policy(args.baseline.resolve(), config, policy_core="current", seed=0)
    baseline_network = shared_five_network(baseline_system).cpu().eval()
    baseline_heads = copy.deepcopy(baseline_network.heads)
    # The cached-head fit authorizes only action_query and entry calibrators.
    from scripts.train_temporal_cached_head import frozen_hash
    baseline_frozen_hash = frozen_hash(baseline_network)
    del baseline_network, baseline_system
    system = tai_he_chuyen_gia_mot_context_mot_policy(args.checkpoint.resolve(), config, policy_core="current", seed=0)
    network = shared_five_network(system).cpu().eval()
    if frozen_hash(network) != baseline_frozen_hash:
        raise ValueError("candidate and baseline do not share the frozen recurrent backbone")
    candidate_heads = network.heads
    starts = np.flatnonzero(np.r_[True, data.sample_times[1:] != data.sample_times[:-1]])
    endpoints = np.r_[starts, len(data.actions)]
    chunk_groups = 64 * args.batch_windows
    chunks = []
    begun = time.monotonic()
    for group_start in range(0, len(starts), chunk_groups):
        start = int(endpoints[group_start])
        stop = int(endpoints[min(group_start + chunk_groups, len(starts))])
        path = args.output_dir / f"chunk_{group_start:07d}.pkl"
        if path.exists():
            with path.open("rb") as handle:
                chunk = pickle.load(handle)
        else:
            segment = subset(data, slice(start, stop))
            hidden, audit = extract_owner_hidden_sequences(network, segment,
                batch_size=args.batch_windows, sequence_length=64, representation="hidden")
            agents = torch.as_tensor(segment.agent_ids, dtype=torch.long)
            network.heads = candidate_heads
            candidate_scores = scores(network, hidden, agents)
            network.heads = baseline_heads
            baseline_scores = scores(network, hidden, agents)
            network.heads = candidate_heads
            chunk = {"start": start, "stop": stop, "candidate": candidate_scores,
                     "baseline": baseline_scores, "audit": audit}
            validate_chunk(chunk, start, stop)
            _atomic_pickle(chunk, path)
        validate_chunk(chunk, start, stop)
        chunks.append(chunk)
        progress = {"status": "running", "rows_completed": stop, "total_rows": len(data.actions),
                    "chunks_completed": len(chunks), "elapsed_seconds_this_invocation": time.monotonic() - begun}
        (args.output_dir / "progress.json").write_text(json.dumps(progress, indent=2), encoding="utf-8")
        print(json.dumps({"event": "full_validation_progress", **progress}), flush=True)
    labels = (data.actions != 0).astype(int)
    metrics = {}
    for arm in ("candidate", "baseline"):
        margin = np.concatenate([chunk[arm] for chunk in chunks])
        metrics[arm] = {"overall": _metrics(labels, (margin > 0).astype(int), margin), "roles": {}}
        for role, name in ((0, "buy"), (1, "sell")):
            mask = data.agent_ids == role
            metrics[arm]["roles"][name] = _metrics(labels[mask], (margin[mask] > 0).astype(int), margin[mask])
    candidate = metrics["candidate"]
    overall = candidate["overall"]
    precision_floor = max(.30, 1.20 * overall["label_entry_rate"])
    rate_ceiling = min(.50, max(.03, 1.20 * overall["label_entry_rate"]))
    checks = {"all_validation_rows": len(labels) == sum(c["stop"] - c["start"] for c in chunks),
              "overall_ba": overall["balanced_accuracy"] >= .55,
              "ba_gain": overall["balanced_accuracy"] - metrics["baseline"]["overall"]["balanced_accuracy"] >= .02,
              "overall_mcc": overall["mcc"] >= .08,
              "overall_precision": overall["entry_precision"] >= precision_floor,
              "overall_entry_rate": overall["predicted_entry_rate"] <= rate_ceiling}
    for name, row in candidate["roles"].items():
        checks[name] = bool(row["samples"] >= 50 and row["balanced_accuracy"] >= .52
                            and row["mcc"] >= .05 and row["entry_recall"] >= .20
                            and row["entry_precision"] >= .30
                            and row["predicted_entry_rate"] <= 1.20 * row["label_entry_rate"])
    checks["checkpoint_unchanged"] = _sha256(args.checkpoint) == contract["checkpoint_sha256"]
    checks["baseline_unchanged"] = _sha256(args.baseline) == contract["baseline_sha256"]
    checks["dataset_unchanged"] = _sha256(args.dataset) == contract["dataset_sha256"]
    checks["config_unchanged"] = _sha256(args.config) == contract["config_sha256"]
    result = {"schema_version": 1, "contract": contract, "status": "full_validation_complete",
              "metrics": metrics, "checks": checks, "metric_gate_passed": all(checks.values()),
              "full_bptt_training_certified": False, "mt5_oos_proven": False, "promotable": False}
    (args.output_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.output_dir / "progress.json").write_text(json.dumps({"status": "complete", "rows_completed": len(labels),
        "total_rows": len(labels), "metric_gate_passed": result["metric_gate_passed"]}, indent=2), encoding="utf-8")
    print(json.dumps({"event": "full_validation_complete", **result}), flush=True)
    return 0 if result["metric_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
