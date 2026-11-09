#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from self_growing_trader.seven_choice_teacher import (
    SevenChoiceBuildConfig,
    build_seven_choice_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tao nhan phan thuc te cho HOLD va sau kieu vao lenh."
    )
    parser.add_argument("--snapshots", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizon-bars", type=int, default=48)
    parser.add_argument("--pending-expiry-bars", type=int, default=5)
    parser.add_argument(
        "--holding-horizon-from-fill", action="store_true",
        help="Dong lenh sau so nến tinh tu luc lenh cho khop, dung hop dong MT5.",
    )
    parser.add_argument(
        "--risk-denominator-from-fill", action="store_true",
        help="Chia R theo khoang tu gia khop den SL, dung so giao dich MT5.",
    )
    parser.add_argument(
        "--pending-fill-price-mode", choices=("order", "touch"), default="order",
        help="Gia khop lenh cho: order = gia dat theo MT5; touch = gia cham cua du lieu cu.",
    )
    parser.add_argument("--entry-offset-cap-atr", type=float, default=5.0)
    parser.add_argument("--fixed-entry-offset-atr", type=float, default=0.75)
    parser.add_argument("--fixed-sl-atr", type=float, default=1.50)
    parser.add_argument("--fixed-tp-rr", type=float, default=2.00)
    parser.add_argument("--fixed-delay-index", type=int, default=0)
    parser.add_argument("--action-history-min-count", type=int, default=48)
    parser.add_argument(
        "--action-history-stride-bars", type=int, default=1,
        help=(
            "Khoang cach cap nhat lich su nhan qua khu. Dat bang horizon-bars "
            "de tranh dem nhieu mau dung chung mot duong gia tuong lai."
        ),
    )
    parser.add_argument("--action-lcb-z", type=float, default=1.0)
    parser.add_argument("--action-lcb-min-r", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()
    report = build_seven_choice_dataset(
        args.snapshots,
        args.output,
        config=SevenChoiceBuildConfig(
            horizon_bars=args.horizon_bars,
            pending_expiry_bars=args.pending_expiry_bars,
            holding_horizon_from_fill=args.holding_horizon_from_fill,
            risk_denominator_from_fill=args.risk_denominator_from_fill,
            pending_fill_price_mode=args.pending_fill_price_mode,
            entry_offset_cap_atr=args.entry_offset_cap_atr,
            fixed_entry_offset_atr=args.fixed_entry_offset_atr,
            fixed_sl_atr=args.fixed_sl_atr,
            fixed_tp_rr=args.fixed_tp_rr,
            fixed_delay_index=args.fixed_delay_index,
            action_history_min_count=args.action_history_min_count,
            action_history_stride_bars=args.action_history_stride_bars,
            action_lcb_z=args.action_lcb_z,
            action_lcb_min_r=args.action_lcb_min_r,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            max_rows=args.max_rows,
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
