#!/usr/bin/env python3
"""Học đầu dự đoán lợi nhuận ròng cho sáu cách vào lệnh.

Thân mã hóa, LSTM, bộ nhớ thời gian và các đầu hành động được giữ nguyên.
Luồng này dùng khi tập dữ liệu lớn hơn khả năng lưu phát lại trạng thái ẩn;
đầu giá trị vẫn chỉ nhận gốc cây dự đoán từ quan sát nhân quả.
"""
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.seven_choice_teacher import (
    _fit_expected_value_teacher,
    load_seven_choice_dataset,
    shared_five_network,
    seven_choice_runtime_contract,
)
from self_growing_trader.execution_contract import (
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
)


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Hoc dau ky vong loi nhuan sau chi phi tren than Five co dinh."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    parser.add_argument(
        "--positive-outcome-weight", type=float, default=0.0,
        help="Tang trong so mau co loi nhuan duong trong dau gia tri (chi nghien cuu).",
    )
    parser.add_argument(
        "--validation-blocks", type=int, default=1,
        help="So doan lien tiep trong tap kiem dinh dung de chon nguong.",
    )
    parser.add_argument(
        "--selection-score-mode",
        choices=("value", "value_times_win", "filled_value", "win_probability"),
        default="value",
        help="Diem xep hang: gia tri, gia tri nhan xac suat thang, hoac xac suat thang.",
    )
    parser.add_argument(
        "--ranking-loss-weight", type=float, default=0.5,
        help="Trong so phan xep hang loi nhuan giua cac lua chon hop le va HOLD.",
    )
    parser.add_argument(
        "--win-target-threshold-r", type=float, default=0.0,
        help="Muc loi nhuan rong toi thieu de danh dau mot lenh thang.",
    )
    parser.add_argument(
        "--path-excursion-loss-weight", type=float, default=0.0,
        help="Trong so hoc bien do thuan/bat loi cua tung cach vao lenh.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-research-only", action="store_true",
        help="Luu dau da hoc de kiem tra khi can duoi am; khong cho MT5/PPO.",
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dataset_path = Path(args.dataset).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"checkpoint dau ra da ton tai: {output}")
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0.0:
        raise ValueError("cau hinh hoc dau ky vong khong hop le")
    if not np.isfinite(args.win_target_threshold_r):
        raise ValueError("win-target-threshold-r phai huu han")
    if not np.isfinite(args.path_excursion_loss_weight) or args.path_excursion_loss_weight < 0.0:
        raise ValueError("path-excursion-loss-weight phai huu han va >= 0")
    started = time.time()
    source_hash = _sha256(source)
    dataset_hash = _sha256(dataset_path)
    data = load_seven_choice_dataset(dataset_path)
    config = load_config(config_path)
    runtime_contract = seven_choice_runtime_contract(data, config)
    if not bool(runtime_contract["match"]) and not args.allow_research_only:
        raise RuntimeError(
            "bo nhan va cau hinh chay khong cung hop dong; "
            "chi duoc bo qua trong pham vi nghien cuu voi "
            "--allow-research-only: "
            + json.dumps(runtime_contract["mismatches"], ensure_ascii=False)
        )
    system = tai_he_chuyen_gia_mot_context_mot_policy(source, config, seed=args.seed)
    network = shared_five_network(system)
    invariant_before = _sha256(source)
    _, _, expected_report = _fit_expected_value_teacher(
        network,
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        minimum_validation_entries=args.minimum_validation_entries,
        seed=args.seed,
        positive_outcome_weight=args.positive_outcome_weight,
        validation_blocks=args.validation_blocks,
        selection_score_mode=args.selection_score_mode,
        ranking_loss_weight=args.ranking_loss_weight,
        win_target_threshold_r=args.win_target_threshold_r,
        path_excursion_loss_weight=args.path_excursion_loss_weight,
        allow_research_only=bool(args.allow_research_only),
        progress=lambda row: print(json.dumps({"event": "seven_choice_value_epoch", **row}, ensure_ascii=False), flush=True),
    )
    if not expected_report["validation_gate_passed"] and not args.allow_research_only:
        raise RuntimeError("dau ky vong khong dat can duoi tren tap kiem dinh")
    if not expected_report["test_gate_passed"] and not args.allow_research_only:
        raise RuntimeError("dau ky vong khong dat can duoi tren tap thu kin")
    # The value labels and the MT5 decoder must share one geometry.  A source
    # checkpoint can contain an older learned factorization, so declare the
    # fixed bins carried by the dataset and make the forward path emit the
    # fixed-geometry contract marker.
    geometry = data.manifest.get("config", {})
    fixed_sl = float(geometry.get("fixed_sl_atr", 1.5))
    fixed_tp_rr = float(geometry.get("fixed_tp_rr", 2.0))
    fixed_delay = int(geometry.get("fixed_delay_index", 0))
    fixed_offset_index = int(
        data.manifest.get("causal_selection_audit", {}).get("fixed_entry_offset_index", 7)
    )
    heads = network.heads
    heads.seven_choice_fixed_geometry_enabled = True
    heads.seven_choice_fixed_delay_index = fixed_delay
    heads.seven_choice_fixed_entry_offset_index = fixed_offset_index
    heads.seven_choice_fixed_sl_index = int(np.argmin(np.abs(np.asarray(SL_ATR_BINS) - fixed_sl)))
    heads.seven_choice_fixed_tp_index = int(np.argmin(np.abs(np.asarray(TP_RR_BINS) - fixed_tp_rr)))
    heads.seven_choice_fixed_trailing_index = int(TRAILING_ATR_BINS.index(0.0))
    heads.seven_choice_fixed_sl_atr = fixed_sl
    heads.seven_choice_fixed_tp_rr = fixed_tp_rr
    network.heads.seven_choice_imitation_trained = True
    network.heads.seven_choice_imitation_schema_version = int(
        data.manifest.get("schema_version", 14)
    )
    network.heads.seven_choice_expected_value_decision_enabled = True
    network.heads.seven_choice_action_decision_mode = "expected_value"
    network.heads.seven_choice_expected_value_threshold_r = float(
        expected_report["selected_threshold"]
    )
    network.heads.seven_choice_expected_value_win_probability_floor = float(
        expected_report.get("win_probability_floor", 0.0)
    )
    network.heads.seven_choice_expected_value_logit_scale = 10.0
    network.heads.seven_choice_expected_value_decision_mode = "direct_post_cost_net_r_v1"
    network.heads.seven_choice_expected_value_score_mode = args.selection_score_mode
    network.heads.seven_choice_expected_value_fill_probability_floor = float(
        expected_report.get("fill_probability_floor", 0.0) or 0.0
    )
    network.heads.seven_choice_expected_value_action_residual_scale = 0.0
    if _sha256(source) != source_hash:
        raise RuntimeError("checkpoint goc da bi thay doi")
    for expert in getattr(getattr(system, "registry", None), "chuyen_gia_theo_context", {}).values():
        policy = getattr(expert, "policy", None)
        if policy is not None and getattr(policy, "five_trainer", None) is not None:
            policy.five_trainer = None
    _atomic_pickle(system, output)
    output_hash = _sha256(output)
    reloaded = tai_he_chuyen_gia_mot_context_mot_policy(output, config, seed=args.seed)
    reloaded_network = shared_five_network(reloaded)
    reloaded_network.require_current_architecture()
    if not bool(getattr(reloaded_network.heads, "seven_choice_expected_value_decision_enabled", False)):
        raise RuntimeError("checkpoint tai lai mat co che quyet dinh theo ky vong")
    report = {
        "schema_version": 1,
        "status": "research_only" if expected_report.get("research_only") else "candidate_only",
        "promotable": False,
        "base_checkpoint": str(source),
        "base_checkpoint_sha256": source_hash,
        "dataset": str(dataset_path),
        "dataset_sha256": dataset_hash,
        "dataset_manifest": data.manifest,
        "runtime_contract": runtime_contract,
        "candidate_checkpoint": str(output),
        "candidate_checkpoint_sha256": output_hash,
        "elapsed_seconds": time.time() - started,
        "training": {
            "scope": "frozen encoder, LSTM, Transformer, action heads and value; action-path head plus root residual and bounded 137-input adapter",
            "expected_value_teacher": expected_report,
            "decision_mode": "direct_post_cost_net_r_v1",
            "selection_score_mode": args.selection_score_mode,
            "live_action_residual_scale": 0.0,
            "threshold_r": float(expected_report["selected_threshold"]),
            "fixed_geometry": {
                "enabled": True,
                "sl_atr": fixed_sl,
                "tp_rr": fixed_tp_rr,
                "delay_index": fixed_delay,
                "entry_offset_index": fixed_offset_index,
                "win_probability_floor": float(
                    expected_report.get("win_probability_floor", 0.0)
                ),
            },
        },
        "verification": {
            "source_hash_unchanged": True,
            "candidate_reload_passed": True,
            "runtime_decision_metadata_roundtrip": True,
            "future_fields_used_as_inputs": False,
            "mt5_oos_proven": False,
            "ppo_allowed": False,
            "research_only": bool(expected_report.get("research_only", False)),
        },
        "next_gate": "paired frozen MT5 on an unseen date range with positive position expectancy lower bound",
    }
    report_path = output.with_suffix(output.suffix + ".seven_choice.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "seven_choice_value_training_complete",
        "checkpoint": str(output),
        "report": str(report_path),
        "test": expected_report["results"]["test"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
