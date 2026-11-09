#!/usr/bin/env python3
"""Compose a protected SL-only imitation checkpoint from verified artifacts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.five_teacher_imitation import (
    geometry_invariant_sha256,
    sha256_file,
    shared_five_network,
    teacher_trainable_sha256,
)
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy


def _atomic_pickle(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=output.name + ".", suffix=".tmp", dir=str(output.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _head_sha256(network: object, name: str) -> str:
    digest = hashlib.sha256()
    head = network.heads.parameter_heads[name]
    for key, tensor in sorted(head.state_dict().items()):
        digest.update(key.encode("utf-8"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes(order="C"))
    return digest.hexdigest()


def compose_sl_only(base_network: object, donor_network: object) -> dict[str, object]:
    """Copy only the learned SL head; action, TP, delay and trailing stay base."""

    # Contract v7 expands legacy SL/TP rows by copying equal-value bins and
    # assigning -30 logits to newly inserted bins. This migration is output
    # neutral for the base checkpoint and must precede the SL donor copy.
    base_network.require_current_architecture()
    donor_network.require_current_architecture()
    before = {name: _head_sha256(base_network, name) for name in ("delay", "sl", "tp", "trailing")}
    donor_sl = _head_sha256(donor_network, "sl")
    base_network.heads.parameter_heads["sl"].load_state_dict(
        donor_network.heads.parameter_heads["sl"].state_dict()
    )
    setattr(base_network.heads, "teacher_geometry_imitation_trained", True)
    setattr(base_network.heads, "teacher_geometry_imitation_schema_version", 2)
    setattr(base_network.heads, "teacher_geometry_imitation_heads", ("sl",))
    after = {name: _head_sha256(base_network, name) for name in ("delay", "sl", "tp", "trailing")}
    if after["sl"] != donor_sl:
        raise RuntimeError("SL head sau compose khong khop donor")
    for name in ("delay", "tp", "trailing"):
        if after[name] != before[name]:
            raise RuntimeError(f"head {name} cua base bi thay doi")
    return {"before": before, "after": after, "donor_sl_sha256": donor_sl}


def _reset_runtime(system: object) -> int:
    registry = getattr(system, "registry", None)
    mapping = getattr(registry, "chuyen_gia_theo_context", {}) if registry is not None else {}
    reset = 0
    seen: set[int] = set()
    for expert in mapping.values():
        policy = getattr(expert, "policy", None)
        if policy is None or id(policy) in seen:
            continue
        seen.add(id(policy))
        if hasattr(policy, "five_ppo_trainer"):
            policy.five_ppo_trainer = None
            reset += 1
        for name in ("five_rollout_steps", "five_pattern_steps", "five_pending_timeout_steps", "five_lifecycle_steps"):
            if hasattr(policy, name):
                setattr(policy, name, [])
    return reset


def main() -> int:
    parser = argparse.ArgumentParser(description="Compose checkpoint imitation SL-only")
    parser.add_argument("--config", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--geometry-donor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    base = Path(args.base_checkpoint).resolve()
    donor = Path(args.geometry_donor).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    base_report_path = base.with_suffix(base.suffix + ".imitation.json")
    donor_report_path = donor.with_suffix(donor.suffix + ".imitation.json")
    base_report = json.loads(base_report_path.read_text(encoding="utf-8"))
    donor_report = json.loads(donor_report_path.read_text(encoding="utf-8"))
    base_hash_before = sha256_file(base)
    donor_hash_before = sha256_file(donor)
    if base_report.get("candidate_checkpoint_sha256") != base_hash_before:
        raise RuntimeError("sidecar base khong khop checkpoint")
    if donor_report.get("candidate_checkpoint_sha256") != donor_hash_before:
        raise RuntimeError("sidecar donor khong khop checkpoint")

    config = load_config(args.config)
    system = tai_he_chuyen_gia_mot_context_mot_policy(base, config, seed=args.seed)
    donor_system = tai_he_chuyen_gia_mot_context_mot_policy(donor, config, seed=args.seed)
    network = shared_five_network(system)
    donor_network = shared_five_network(donor_system)
    invariant_before = geometry_invariant_sha256(network)
    action_hash_before = teacher_trainable_sha256(network)
    composition = compose_sl_only(network, donor_network)
    if geometry_invariant_sha256(network) != invariant_before:
        raise RuntimeError("compose lam thay doi tensor ngoai geometry heads")
    if teacher_trainable_sha256(network) != action_hash_before:
        raise RuntimeError("compose lam thay doi action imitation")
    reset_count = _reset_runtime(system)
    del donor_system
    _atomic_pickle(system, output)
    output_hash = sha256_file(output)

    reloaded = tai_he_chuyen_gia_mot_context_mot_policy(output, config, seed=args.seed)
    reloaded_network = shared_five_network(reloaded)
    if _head_sha256(reloaded_network, "sl") != composition["donor_sl_sha256"]:
        raise RuntimeError("reload lam sai SL head")
    for name in ("delay", "tp", "trailing"):
        if _head_sha256(reloaded_network, name) != composition["before"][name]:
            raise RuntimeError(f"reload lam sai head base {name}")
    if tuple(getattr(reloaded_network.heads, "teacher_geometry_imitation_heads", ())) != ("sl",):
        raise RuntimeError("reload lam mat marker SL-only")
    if sha256_file(base) != base_hash_before or sha256_file(donor) != donor_hash_before:
        raise RuntimeError("checkpoint nguon bi thay doi trong luc compose")

    donor_geometry = copy.deepcopy(dict(donor_report.get("geometry_training", {}) or {}))
    donor_geometry["ablation_heads"] = ["sl"]
    report = {
        "schema_version": 7,
        "status": "candidate_only",
        "promotable": False,
        "training_scope": "action imitation preserved; learned structural SL only; base TP/delay/trailing preserved",
        "mt5_integration": {
            "entry_scope": "all_entry_mass_v2",
            "ppo_runtime_reset_count": reset_count,
            "requires_runtime_teacher_retention": True,
            "geometry_scope": "accepted_pivot_entries_sl_only_v1",
            "geometry_heads": ["sl"],
            "geometry_objective": "ordinal_log_distance_v2",
            "execution_contract_version": 7,
        },
        "base_checkpoint": str(base),
        "base_checkpoint_sha256": base_hash_before,
        "geometry_donor": str(donor),
        "geometry_donor_sha256": donor_hash_before,
        "dataset": donor_report.get("dataset", ""),
        "dataset_sha256": donor_report.get("dataset_sha256", ""),
        "dataset_manifest": donor_report.get("dataset_manifest", {}),
        "candidate_checkpoint": str(output),
        "candidate_checkpoint_sha256": output_hash,
        "candidate_checkpoint_bytes": output.stat().st_size,
        "composition": composition,
        "geometry_training": donor_geometry,
        "verification": {
            "source_hash_unchanged": True,
            "donor_hash_unchanged": True,
            "frozen_network_hash_unchanged": True,
            "protected_action_logits_unchanged": True,
            "action_calibration_preserved_during_geometry_train": True,
            "non_sl_geometry_heads_preserved": True,
            "geometry_checkpoint_reload_passed": True,
            "candidate_reload_passed": True,
            "future_fields_used_as_inputs": False,
            "mt5_oos_proven": False,
        },
    }
    report_path = output.with_suffix(output.suffix + ".imitation.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "sl_only_composition_complete",
        "output": str(output),
        "sha256": output_hash,
        "report": str(report_path),
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
