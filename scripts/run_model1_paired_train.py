from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from self_growing_trader.chuyen_gia_policy import huan_luyen_he_chuyen_gia_mt5
from self_growing_trader.config import load_config
from self_growing_trader.learning_integrity import sha256_file
from self_growing_trader.model1_python_backend import train_context_expert_python_model1
from self_growing_trader.model1_replay import (
    Model1OracleDataset,
    Model1OracleTape,
    model1_training_state_hash,
    validate_model1_execution_contract,
)
from self_growing_trader.mt5_runner import MT5BacktestConfig, MT5StrategyTesterEvaluator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired MT5 model-1 and Python-tape training from one immutable seed"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--runner-dir", default="mt5_v2_runner")
    parser.add_argument("--from-date", default="2025.01.01")
    parser.add_argument("--to-date", default="2025.01.02")
    parser.add_argument("--max-wall-sec", type=int, default=45)
    parser.add_argument("--timeout-sec", type=int, default=60)
    parser.add_argument("--shutdown-terminal", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if not bool(config.environment.strict_model1_parity):
        raise ValueError("paired train requires environment.strict_model1_parity=true")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path = Path(args.seed)
    seed_hash_before = sha256_file(seed_path)
    mt5_output = output_dir / "mt5_trained.pkl"
    mt5_result = huan_luyen_he_chuyen_gia_mt5(
        replace(config, environment=replace(config.environment, backend="mt5")),
        output_path=mt5_output,
        seed_system_path=seed_path,
        runner_dir=args.runner_dir,
        timeout_sec=int(args.timeout_sec),
        max_wall_sec=int(args.max_wall_sec),
        visual=False,
        shutdown_terminal=bool(args.shutdown_terminal),
        from_date=args.from_date,
        to_date=args.to_date,
        model=1,
        seed=int(config.learning.seed),
        resume_existing=False,
    )
    seed_hash_after = sha256_file(seed_path)
    if seed_hash_before != seed_hash_after:
        raise RuntimeError("immutable seed changed during paired MT5 run")
    mt5_policy_path = Path(str(mt5_result.get("policy", "") or ""))
    if not mt5_policy_path.is_file():
        checkpoint = dict(mt5_result.get("checkpoint", {}) or {})
        mt5_policy_path = Path(str(checkpoint.get("incomplete_artifact", "") or ""))
    if not mt5_policy_path.is_file():
        raise RuntimeError("MT5 paired run produced no policy artifact")

    runtime = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=Path(args.runner_dir)))
    capture_names = {
        "bridge.csv": runtime.config.bridge_file,
        "feature_audit.csv": runtime.config.feature_audit_file,
        "execution.csv": runtime.config.execution_file,
        "action.csv": runtime.config.action_file,
    }
    for target_name, runtime_name in capture_names.items():
        source = runtime._latest_runtime_file(runtime_name)
        if source is None or not source.is_file():
            raise FileNotFoundError(f"missing MT5 runtime artifact: {runtime_name}")
        shutil.copy2(source, output_dir / target_name)

    oracle = Model1OracleDataset(
        output_dir / "bridge.csv",
        output_dir / "feature_audit.csv",
        output_dir / "execution.csv",
    )
    tape_path = oracle.compile(output_dir / "oracle_full.npz")
    decisions = oracle.decisions()
    reconstructed = oracle.reconstruct_inputs(
        max_positions=int(config.risk.max_positions),
        max_holding_bars=int(config.orders.max_holding_bars),
        min_daily_trades=int(config.orders.min_daily_trades),
        max_daily_trades=int(config.orders.max_daily_trades),
    )
    if any(
        not np.array_equal(decision.raw_inputs, actual)
        for decision, actual in zip(decisions, reconstructed, strict=True)
    ):
        raise RuntimeError("paired MT5 capture failed exact 137-input reconstruction")

    python_output = output_dir / "python_trained.pkl"
    python_config = replace(
        config,
        environment=replace(
            config.environment,
            backend="python_model1",
            model1_tape_path=str(tape_path),
        ),
    )
    python_result = train_context_expert_python_model1(
        python_config,
        output_path=python_output,
        seed_system_path=seed_path,
        runtime_dir=output_dir / "python_runtime",
    )
    tape = Model1OracleTape(tape_path)
    margins = np.asarray([float(item["margin"]) for item in tape.metadata], dtype=float)
    free_margins = np.asarray([float(item["free_margin"]) for item in tape.metadata], dtype=float)
    leverages = sorted({int(item["leverage"]) for item in tape.metadata})
    execution = validate_model1_execution_contract(tape)
    with mt5_policy_path.open("rb") as handle:
        mt5_policy = pickle.load(handle)
    with python_output.open("rb") as handle:
        python_policy = pickle.load(handle)
    mt5_state_hash = model1_training_state_hash(mt5_policy)
    python_state_hash = model1_training_state_hash(python_policy)
    reasons: list[str] = []
    if not execution.exact:
        reasons.append("execution_contract_failed")
    if mt5_state_hash != python_state_hash:
        reasons.append("training_state_sha256_mismatch")
    if not bool(python_result.get("exact_decision_parity", False)):
        reasons.append("decision_parity_failed")
    report = {
        "schema_version": 1,
        "exact": not reasons,
        "latency_fill_timing_exempt": True,
        "seed_path": str(seed_path),
        "seed_sha256": seed_hash_after,
        "mt5_policy": str(mt5_policy_path),
        "python_policy": str(python_output),
        "tape": str(tape_path),
        "tape_content_sha256": tape.content_sha256,
        "decision_wire_sha256": tape.decision_wire_sha256,
        "logical_state_sha256": tape.logical_state_sha256,
        "account_margin_state_sha256": tape.account_margin_state_sha256,
        "execution_ledger_sha256": tape.execution_ledger_sha256,
        "decision_count": len(tape),
        "input_value_count": int(tape.inputs.size),
        "output_value_count": int(tape.outputs.size),
        "execution_row_count": max(0, len(tape.execution_lines) - 1),
        "account_margin_snapshot_count": len(tape.metadata),
        "nonzero_margin_snapshot_count": int(np.count_nonzero(margins)),
        "margin_min": float(np.min(margins)),
        "margin_max": float(np.max(margins)),
        "free_margin_min": float(np.min(free_margins)),
        "free_margin_max": float(np.max(free_margins)),
        "leverage_values": leverages,
        "checked_entry_geometries": execution.checked_entry_geometries,
        "mt5_training_state_sha256": mt5_state_hash,
        "python_training_state_sha256": python_state_hash,
        "reasons": reasons,
    }
    report_path = output_dir / "paired_training_parity.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if reasons:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
