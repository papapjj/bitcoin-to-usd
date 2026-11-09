from __future__ import annotations

import argparse
import json

import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.model1_replay import Model1OracleDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Require zero-difference MT5 model-1 input parity")
    parser.add_argument("--config", required=True)
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--feature-audit", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = Model1OracleDataset(args.bridge, args.feature_audit)
    decisions = dataset.decisions()
    reconstructed = dataset.reconstruct_inputs(
        max_positions=int(config.risk.max_positions),
        max_holding_bars=int(config.orders.max_holding_bars),
        min_daily_trades=int(config.orders.min_daily_trades),
        max_daily_trades=int(config.orders.max_daily_trades),
    )
    mismatches: list[dict[str, object]] = []
    for decision, actual in zip(decisions, reconstructed, strict=True):
        indexes = np.flatnonzero(decision.raw_inputs != actual)
        if indexes.size:
            mismatches.append({
                "sequence": decision.sequence,
                "event_id": decision.event_id,
                "failing_indexes": indexes.tolist(),
                "max_abs_error": float(np.max(np.abs(decision.raw_inputs - actual))),
            })
    report = {
        "schema_version": 1,
        "exact": not mismatches,
        "checked_rows": len(decisions),
        "checked_values": len(decisions) * 137,
        "mismatch_count": len(mismatches),
        "max_abs_error": 0.0 if not mismatches else max(float(row["max_abs_error"]) for row in mismatches),
        "mismatch_preview": mismatches[:10],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
