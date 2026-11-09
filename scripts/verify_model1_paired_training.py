from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

from self_growing_trader.model1_replay import (
    Model1OracleTape,
    model1_training_state_hash,
    validate_model1_execution_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Require zero-difference MT5/Python model-1 training state"
    )
    parser.add_argument("--mt5-policy", required=True)
    parser.add_argument("--python-policy", required=True)
    parser.add_argument("--tape", required=True)
    parser.add_argument("--python-report", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    with Path(args.mt5_policy).open("rb") as handle:
        mt5_policy = pickle.load(handle)
    with Path(args.python_policy).open("rb") as handle:
        python_policy = pickle.load(handle)
    tape = Model1OracleTape(args.tape)
    python_report = json.loads(Path(args.python_report).read_text(encoding="utf-8"))
    execution_report = validate_model1_execution_contract(tape)
    mt5_hash = model1_training_state_hash(mt5_policy)
    python_hash = model1_training_state_hash(python_policy)
    reasons: list[str] = []
    if mt5_hash != python_hash:
        reasons.append("training_state_sha256_mismatch")
    if not bool(python_report.get("exact_decision_parity", False)):
        reasons.append("decision_parity_failed")
    if int(python_report.get("decision_count", -1)) != len(tape):
        reasons.append("decision_count_mismatch")
    if str(python_report.get("tape_content_sha256", "")) != tape.content_sha256:
        reasons.append("tape_hash_mismatch")
    if not execution_report.exact:
        reasons.append("execution_contract_failed")
    report = {
        "schema_version": 1,
        "exact": not reasons,
        "latency_fill_timing_exempt": True,
        "tape_content_sha256": tape.content_sha256,
        "decision_count": len(tape),
        "input_value_count": int(tape.inputs.size),
        "output_value_count": int(tape.outputs.size),
        "execution_row_count": max(0, len(tape.execution_lines) - 1),
        "checked_entry_geometries": execution_report.checked_entry_geometries,
        "decision_wire_sha256": tape.decision_wire_sha256,
        "logical_state_sha256": tape.logical_state_sha256,
        "account_margin_state_sha256": tape.account_margin_state_sha256,
        "execution_ledger_sha256": tape.execution_ledger_sha256,
        "execution_contract_mismatches": list(execution_report.mismatches[:10]),
        "mt5_training_state_sha256": mt5_hash,
        "python_training_state_sha256": python_hash,
        "reasons": reasons,
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered)
    if reasons:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
