from __future__ import annotations

import argparse
import json

import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.model1_python_backend import Model1PythonBackend
from self_growing_trader.model1_replay import Model1OracleTape


def main() -> None:
    parser = argparse.ArgumentParser(description="Require exact model-1 policy wire parity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument("--tape", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    tape = Model1OracleTape(args.tape)
    decisions = Model1PythonBackend(config, args.seed).run_tape(tape)
    mismatches = []
    for index, decision in enumerate(decisions):
        expected_outputs = tape.outputs[index]
        if bool(config.environment.strict_model1_parity):
            expected_outputs = np.round(
                expected_outputs, int(config.environment.output_wire_decimals)
            )
        failing = np.flatnonzero(decision.outputs != expected_outputs)
        if failing.size or decision.action != int(tape.actions[index]):
            mismatches.append({
                "sequence": index + 1, "event_id": decision.event_id,
                "expected_action": int(tape.actions[index]), "actual_action": decision.action,
                "failing_output_indexes": failing.tolist(),
                "max_abs_error": float(np.max(np.abs(decision.outputs - expected_outputs))),
            })
    report = {
        "schema_version": 1, "exact": not mismatches, "rows": len(tape),
        "checked_output_values": int(tape.outputs.size), "mismatch_count": len(mismatches),
        "mismatch_preview": mismatches[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
