#!/usr/bin/env python3
"""Tạo snapshot 137 đầu vào có trạng thái nhân quả để nghiên cứu.

Tệp đầu vào là snapshot tick v2 do
``BottomTop_Counterfactual_Exporter_EA`` xuất.  Công cụ chạy một lịch hành vi
cố định qua các tick quá khứ, lấp các ô vị thế/tài khoản/phần thực thi trước
khi ghi từng dòng.  Không dùng MT5, nhãn tương lai hay cập nhật học.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.causal_state_replay import (  # noqa: E402
    CausalStateReplayConfig,
    V16_FIXED_ENTRY_OFFSET_ATR,
    augment_snapshot_file,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon-bars", type=int, default=48)
    parser.add_argument("--pending-expiry-bars", type=int, default=5)
    parser.add_argument("--cycle-bars", type=int, default=0)
    parser.add_argument("--fixed-entry-offset-atr", type=float, default=V16_FIXED_ENTRY_OFFSET_ATR)
    parser.add_argument("--fixed-sl-atr", type=float, default=1.50)
    parser.add_argument("--fixed-tp-rr", type=float, default=2.00)
    parser.add_argument("--risk-fraction", type=float, default=0.01)
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    parser.add_argument("--max-positions", type=int, default=5)
    parser.add_argument("--max-holding-bars", type=int, default=48)
    parser.add_argument("--min-daily-trades", type=int, default=0)
    parser.add_argument("--max-daily-trades", type=int, default=0)
    args = parser.parse_args()
    report = augment_snapshot_file(
        args.snapshots,
        args.output,
        config=CausalStateReplayConfig(
            horizon_bars=args.horizon_bars,
            pending_expiry_bars=args.pending_expiry_bars,
            cycle_bars=args.cycle_bars,
            fixed_entry_offset_atr=args.fixed_entry_offset_atr,
            fixed_sl_atr=args.fixed_sl_atr,
            fixed_tp_rr=args.fixed_tp_rr,
            risk_fraction=args.risk_fraction,
            initial_balance=args.initial_balance,
            max_positions=args.max_positions,
            max_holding_bars=args.max_holding_bars,
            min_daily_trades=args.min_daily_trades,
            max_daily_trades=args.max_daily_trades,
        ),
    )
    # Windows PowerShell may expose a cp1252 stdout.  Escaping the Vietnamese
    # report text keeps the command usable there while the sidecar itself stays
    # UTF-8 and retains the original text.
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
