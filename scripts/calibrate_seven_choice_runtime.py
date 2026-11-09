#!/usr/bin/env python3
"""Tạo bản nghiên cứu chỉ đổi ngưỡng và phần bù đầu bảy lựa chọn."""
from __future__ import annotations

import argparse
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
from self_growing_trader.self_growing_runner import (
    tai_he_chuyen_gia_mot_context_mot_policy,
)
from self_growing_trader.seven_choice_teacher import shared_five_network


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_pickle(value: object, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=output.name + ".", suffix=".tmp", dir=str(output.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _lookup_sweep(path: Path, scale: float, threshold: float) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    for item in report.get("residual_threshold_sweep", []):
        if abs(float(item.get("residual_scale", 0.0)) - scale) > 1e-9:
            continue
        if abs(float(item.get("threshold_r", 0.0)) - threshold) > 1e-9:
            continue
        return item
    raise ValueError("khong tim thay cap hinh trong bao cao quet")


def _assert_sweep_binding(
    report_path: Path,
    source: Path,
    source_hash: str,
    dataset: Path,
    config: Path,
) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    declared = Path(str(report.get("checkpoint", ""))).resolve()
    if not declared.is_file():
        raise FileNotFoundError(
            "bao cao quet khong tro toi checkpoint ton tai: " + str(declared)
        )
    declared_hash = _sha256(declared)
    if declared_hash != source_hash or str(report.get("checkpoint_sha256", "")) != source_hash:
        raise ValueError(
            "checkpoint nguon khong trung checkpoint da do trong bao cao quet: "
            f"source={source} report={declared}"
        )
    declared_dataset = Path(str(report.get("dataset", ""))).resolve()
    if declared_dataset != dataset.resolve() or not dataset.is_file():
        raise ValueError("tap du lieu khong trung bao cao quet")
    if str(report.get("dataset_sha256", "")) != _sha256(dataset):
        raise ValueError("hash tap du lieu khong trung bao cao quet")
    declared_config = Path(str(report.get("config", ""))).resolve()
    if declared_config != config.resolve() or not config.is_file():
        raise ValueError("config khong trung bao cao quet")
    if str(report.get("config_sha256", "")) != _sha256(config):
        raise ValueError("hash config khong trung bao cao quet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sweep-report", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--residual-scale", required=True, type=float)
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"checkpoint da ton tai: {args.output}")
    if args.residual_scale < 0.0 or args.minimum_validation_entries < 2:
        raise ValueError("cau hinh hieu chuan khong hop le")
    evidence = _lookup_sweep(
        args.sweep_report.resolve(), float(args.residual_scale), float(args.threshold)
    )
    validation_entries = int(evidence["validation"].get("predicted_entries", 0))
    source = args.source.resolve()
    source_hash = _sha256(source)
    dataset = args.dataset.resolve()
    config = args.config.resolve()
    _assert_sweep_binding(
        args.sweep_report.resolve(), source, source_hash, dataset, config
    )
    system = tai_he_chuyen_gia_mot_context_mot_policy(
        source, load_config(args.config.resolve()), seed=0
    )
    network = shared_five_network(system)
    network.require_current_architecture()
    heads = network.heads
    heads.seven_choice_expected_value_decision_enabled = True
    heads.seven_choice_expected_value_threshold_r = float(args.threshold)
    heads.seven_choice_expected_value_action_residual_scale = float(args.residual_scale)
    # Runtime optimizers must not keep references to the pre-calibration
    # tensors.  The calibrated artifact remains research-only.
    reset_count = 0
    for expert in getattr(getattr(system, "registry", None), "chuyen_gia_theo_context", {}).values():
        policy = getattr(expert, "policy", None)
        if policy is not None and getattr(policy, "five_trainer", None) is not None:
            policy.five_trainer = None
            reset_count += 1
    output = args.output.resolve()
    _atomic_pickle(system, output)
    output_hash = _sha256(output)
    reloaded = tai_he_chuyen_gia_mot_context_mot_policy(
        output, load_config(args.config.resolve()), seed=0
    )
    reloaded_heads = shared_five_network(reloaded).heads
    if abs(float(getattr(reloaded_heads, "seven_choice_expected_value_threshold_r", 0.0)) - args.threshold) > 1e-9:
        raise RuntimeError("tai lai mat nguong hieu chuan")
    if abs(float(getattr(reloaded_heads, "seven_choice_expected_value_action_residual_scale", 0.0)) - args.residual_scale) > 1e-9:
        raise RuntimeError("tai lai mat phan bu dau hanh dong")
    report = {
        "schema_version": 1,
        "status": "research_only",
        "promotable": False,
        "source_checkpoint": str(source),
        "source_checkpoint_sha256": source_hash,
        "dataset": str(dataset),
        "dataset_sha256": _sha256(dataset),
        "config": str(config),
        "config_sha256": _sha256(config),
        "candidate_checkpoint": str(output),
        "candidate_checkpoint_sha256": output_hash,
        "calibration": {
            "threshold_r": float(args.threshold),
            "action_residual_scale": float(args.residual_scale),
            "selection_uses": "validation only",
            "test_used_for_selection": False,
            "minimum_validation_entries": int(args.minimum_validation_entries),
            "validation_entries": validation_entries,
            "below_minimum_validation_entries": bool(
                validation_entries < args.minimum_validation_entries
            ),
            "sweep_report": str(args.sweep_report.resolve()),
            "sweep_evidence": evidence,
        },
        "verification": {
            "candidate_reload_passed": True,
            "runtime_optimizer_reset_count": reset_count,
            "future_fields_used_as_inputs": False,
            "mt5_oos_proven": False,
            "ppo_allowed": False,
        },
        "next_gate": "paired frozen MT5 on unseen dates with at least 200 closed positions and positive lower bound",
    }
    report_path = output.with_suffix(output.suffix + ".seven_choice.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"candidate": str(output), "report": str(report_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
