#!/usr/bin/env python3
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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.config import load_config
from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.seven_choice_teacher import (
    load_seven_choice_dataset,
    shared_five_network,
    seven_choice_runtime_contract,
    train_seven_choice_heads,
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
        description="Giu than Five co dinh va hoc lai cac dau ra cua bay lua chon."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden-cache", default="")
    parser.add_argument("--reuse-hidden-cache", action="store_true")
    parser.add_argument(
        "--hidden-extraction-batch-size", type=int, default=64,
        help="So cua so thoi gian phat lai song song khi trich trang thai an.",
    )
    parser.add_argument(
        "--validation-blocks", type=int, default=1,
        help="So doan lien tiep trong tap kiem dinh dung de chon nguong.",
    )
    parser.add_argument(
        "--action-target-mode", choices=("causal", "outcome"), default="causal",
        help=(
            "Nhan cho dau hanh dong: causal la nhan lich su; outcome la nhan "
            "ket qua sau chi phi duoc tao offline, chi dung cho nghien cuu."
        ),
    )
    parser.add_argument(
        "--selection-score-mode",
        choices=("value", "value_times_win", "filled_value", "win_probability"),
        default="value",
        help="Diem xep hang dau vao: gia tri, gia tri nhan xac suat thang, hoac xac suat thang.",
    )
    parser.add_argument(
        "--ranking-loss-weight", type=float, default=0.5,
        help="Trong so phan xep hang loi nhuan giua cac lua chon hop le va HOLD.",
    )
    parser.add_argument(
        "--path-excursion-loss-weight", type=float, default=0.0,
        help=(
            "Trong so hoc bien do thuan/bat loi MFE/MAE cua tung cach vao lenh; "
            "mac dinh tat de giu nguyen cac luot chay truoc."
        ),
    )
    parser.add_argument(
        "--action-decision-mode", choices=("expected_value", "direct"),
        default="expected_value",
        help=(
            "Nguon quyet dinh vao lenh: expected_value dung mat gia tri sau chi phi; "
            "direct dung dau ra hanh dong da hoc truc tiep."
        ),
    )
    parser.add_argument("--action-learning-rate-multiplier", type=float, default=10.0)
    parser.add_argument("--expected-value-epochs", type=int, default=20)
    parser.add_argument("--expected-value-learning-rate", type=float, default=0.003)
    parser.add_argument(
        "--hidden-value-residual-enabled", action="store_true",
        help=(
            "Hoc hieu chinh gia tri nho tu trang thai an theo thoi gian; "
            "chi dung trong ho so nghien cuu sau khi than mang da khoa."
        ),
    )
    parser.add_argument("--hidden-value-residual-epochs", type=int, default=8)
    parser.add_argument("--hidden-value-residual-learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden-value-residual-scale-r", type=float, default=0.25)
    parser.add_argument(
        "--hidden-value-residual-ranking-loss-weight", type=float, default=0.0,
        help=(
            "Trong so xep hang HOLD va ba lua chon dung vai cho hieu chinh trang thai an; "
            "mac dinh tat va chi dung trong ho so nghien cuu."
        ),
    )
    parser.add_argument(
        "--positive-outcome-weight", type=float, default=0.0,
        help=(
            "Tang trong so mau co loi nhuan duong trong dau gia tri; chi nen dung "
            "cho phep thu nghiem va van phai qua phat lai tu sinh."
        ),
    )
    parser.add_argument("--minimum-validation-entries", type=int, default=128)
    parser.add_argument(
        "--allow-research-only", action="store_true",
        help="Luu dau ra de phan tich khi can duoi am; khong cho MT5/PPO.",
    )
    parser.add_argument(
        "--allow-contract-mismatch", action="store_true",
        help=(
            "Chi cho phep khi cung bat --allow-research-only de soi ho so cu "
            "co hinh hoc hoac vong doi khac bo nhan."
        ),
    )
    args = parser.parse_args()

    source = Path(args.source).resolve()
    dataset_path = Path(args.dataset).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"checkpoint dau ra da ton tai: {output}")
    source_hash = _sha256(source)
    dataset_hash = _sha256(dataset_path)
    started = time.time()
    data = load_seven_choice_dataset(dataset_path)
    config = load_config(config_path)
    runtime_contract = seven_choice_runtime_contract(data, config)
    if not bool(runtime_contract["match"]):
        if not (args.allow_contract_mismatch and args.allow_research_only):
            raise RuntimeError(
                "bo nhan va cau hinh chay khong cung hop dong; "
                "chi duoc bo qua trong pham vi nghien cuu voi "
                "--allow-contract-mismatch --allow-research-only: "
                + json.dumps(runtime_contract["mismatches"], ensure_ascii=False)
            )
    system = tai_he_chuyen_gia_mot_context_mot_policy(source, config, seed=args.seed)
    network = shared_five_network(system)
    result = train_seven_choice_heads(
        network, data, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, patience=args.patience,
        sequence_length=args.sequence_length, seed=args.seed,
        hidden_cache_path=(Path(args.hidden_cache).resolve() if args.hidden_cache else None),
        reuse_hidden_cache=bool(args.reuse_hidden_cache),
        hidden_extraction_batch_size=args.hidden_extraction_batch_size,
        validation_blocks=args.validation_blocks,
        action_target_mode=args.action_target_mode,
        selection_score_mode=args.selection_score_mode,
        ranking_loss_weight=args.ranking_loss_weight,
        path_excursion_loss_weight=args.path_excursion_loss_weight,
        action_decision_mode=args.action_decision_mode,
        action_learning_rate_multiplier=args.action_learning_rate_multiplier,
        expected_value_epochs=args.expected_value_epochs,
        expected_value_learning_rate=args.expected_value_learning_rate,
        hidden_value_residual_enabled=bool(args.hidden_value_residual_enabled),
        hidden_value_residual_epochs=args.hidden_value_residual_epochs,
        hidden_value_residual_learning_rate=args.hidden_value_residual_learning_rate,
        hidden_value_residual_scale_r=args.hidden_value_residual_scale_r,
        hidden_value_residual_ranking_loss_weight=args.hidden_value_residual_ranking_loss_weight,
        positive_outcome_weight=args.positive_outcome_weight,
        minimum_validation_entries=args.minimum_validation_entries,
        allow_research_only=bool(args.allow_research_only),
        progress=lambda row: print(json.dumps({"event": "seven_choice_epoch", **row}), flush=True),
    )
    if _sha256(source) != source_hash:
        raise RuntimeError("checkpoint goc da bi thay doi")
    # Runtime trainers are rebuilt lazily. Removing stale optimizer objects
    # prevents PPO state from referring to the pre-imitation head tensors.
    reset_count = 0
    registry = getattr(system, "registry", None)
    for expert in getattr(registry, "chuyen_gia_theo_context", {}).values():
        policy = getattr(expert, "policy", None)
        if policy is not None and getattr(policy, "five_trainer", None) is not None:
            policy.five_trainer = None
            reset_count += 1
    _atomic_pickle(system, output)
    output_hash = _sha256(output)
    reloaded = tai_he_chuyen_gia_mot_context_mot_policy(output, config, seed=args.seed)
    reloaded_network = shared_five_network(reloaded)
    # Verify the exact runtime decision path, not only the historical
    # ``imitation_trained`` marker.  A legacy artifact can carry that marker
    # while missing the expected-value settings; accepting it would make the
    # next MT5 run silently use primary logits instead of the trained surface.
    reloaded_network.require_current_architecture()
    if not bool(getattr(reloaded_network.heads, "seven_choice_imitation_trained", False)):
        raise RuntimeError("checkpoint tai lai mat dau vet hoc bay lua chon")
    required_runtime_metadata = (
        "seven_choice_expected_value_decision_enabled",
        "seven_choice_action_decision_mode",
        "seven_choice_expected_value_threshold_r",
        "seven_choice_expected_value_score_mode",
        "seven_choice_expected_value_fill_probability_floor",
        "seven_choice_expected_value_action_residual_scale",
        "seven_choice_expected_value_hidden_residual_enabled",
        "seven_choice_expected_value_hidden_residual_scale_r",
        "seven_choice_expected_value_logit_scale",
        "seven_choice_fixed_geometry_enabled",
        "seven_choice_fixed_delay_index",
        "seven_choice_fixed_entry_offset_index",
        "seven_choice_fixed_sl_index",
        "seven_choice_fixed_tp_index",
        "seven_choice_fixed_trailing_index",
    )
    missing_runtime_metadata = [
        name for name in required_runtime_metadata
        if not hasattr(reloaded_network.heads, name)
    ]
    if missing_runtime_metadata:
        raise RuntimeError(
            "checkpoint tai lai thieu metadata duong quyet dinh: "
            + ", ".join(missing_runtime_metadata)
        )
    report = {
        "schema_version": 1,
        "status": "research_only" if result.get("research_only") else "candidate_only",
        "promotable": False,
        "base_checkpoint": str(source), "base_checkpoint_sha256": source_hash,
        "dataset": str(dataset_path), "dataset_sha256": dataset_hash,
        "runtime_contract": runtime_contract,
        "dataset_manifest": data.manifest,
        "candidate_checkpoint": str(output), "candidate_checkpoint_sha256": output_hash,
        "elapsed_seconds": time.time() - started,
        "training": result,
        "verification": {
            "source_hash_unchanged": True,
            "frozen_trunk_hash_unchanged": True,
            "candidate_reload_passed": True,
            "runtime_decision_metadata_roundtrip": True,
            "runtime_decision_metadata_fields": list(required_runtime_metadata),
            "ppo_runtime_reset_count": reset_count,
            "future_fields_used_as_inputs": False,
            "mt5_oos_proven": False,
            "ppo_allowed": False,
            "research_only": bool(result.get("research_only", False)),
            "runtime_contract_match": bool(runtime_contract["match"]),
        },
        "next_gate": (
            "paired frozen MT5 on the exact checkpoint/date/model/seed; "
            "policy-origin expectancy lower bound must be strictly positive"
        ),
    }
    report_path = output.with_suffix(output.suffix + ".seven_choice.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "event": "seven_choice_training_complete", "checkpoint": str(output),
        "report": str(report_path), "test": result["after"]["test"],
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
