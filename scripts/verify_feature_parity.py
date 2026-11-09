#!/usr/bin/env python3
"""Verify a real EA 137-feature audit against the Python reference.

This is intentionally an offline verifier.  It does not claim MT5 execution
was correct merely because a source-level static checker passed; it consumes an
actual audit CSV emitted by the compiled EA and exits non-zero on any mismatch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_growing_trader.config import load_config
from self_growing_trader.feature_contract import (
    compare_mt5_feature_audit_against_reference,
    feature_schema_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config used for the MT5 run")
    parser.add_argument("--audit", required=True, help="self_growing_feature_audit.csv exported by EA")
    parser.add_argument("--reference-csv", default="", help="Raw OHLCV CSV used by the tester; defaults to config data.csv_path")
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--atol", type=float, default=None)
    parser.add_argument("--rtol", type=float, default=None)
    parser.add_argument(
        "--audit-inputs-normalized",
        action="store_true",
        help=(
            "Dùng duy nhất cho audit bridge v8 trở xuống, nơi EA đã xuất "
            "vector policy-normalized. Bridge v9 mặc định xuất raw."
        ),
    )
    args = parser.parse_args()

    config = load_config(Path(args.config))
    schema = feature_schema_fingerprint(
        config.features,
        config.symbol,
        session_timezone=config.learning.session_timezone,
        spread_semantics=config.learning.spread_semantics,
        mt5_server_utc_offset_hours=config.learning.mt5_server_utc_offset_hours,
        input_normalization_mode=config.learning.input_normalization_mode,
    )
    result = compare_mt5_feature_audit_against_reference(
        args.audit,
        args.reference_csv or config.csv_path,
        config.features,
        config.symbol,
        expected_schema_hash=schema,
        datetime_column=config.datetime_column,
        initial_equity=float(config.risk.initial_equity),
        min_daily_trades=int(config.orders.min_daily_trades),
        max_daily_trades=int(config.orders.max_daily_trades),
        max_holding_bars=int(config.orders.max_holding_bars),
        atol=float(args.atol) if args.atol is not None else 1e-6,
        rtol=float(args.rtol) if args.rtol is not None else 1e-5,
        max_rows=max(1, int(args.max_rows)),
        require_broker_metadata=True,
        input_normalization_mode=config.learning.input_normalization_mode,
        audit_inputs_normalized=bool(args.audit_inputs_normalized),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=str))
    return 0 if bool(result.get("numerical_valid", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
