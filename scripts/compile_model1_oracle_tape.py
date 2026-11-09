from __future__ import annotations

import argparse
import json

from self_growing_trader.model1_replay import Model1OracleDataset, Model1OracleTape


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile an MT5 model-1 oracle into a CSV-free replay tape")
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--feature-audit", required=True)
    parser.add_argument("--execution")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    target = Model1OracleDataset(
        args.bridge, args.feature_audit, args.execution
    ).compile(args.output)
    tape = Model1OracleTape(target)
    print(json.dumps({
        "schema_version": 1,
        "path": str(target.resolve()),
        "rows": len(tape),
        "input_values": int(tape.inputs.size),
        "output_values": int(tape.outputs.size),
        "execution_rows": max(0, len(tape.execution_lines) - 1),
        "content_sha256": tape.content_sha256,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
