from __future__ import annotations

import argparse
import json

from self_growing_trader.teacher_imitation_dataset import (
    TeacherDatasetBuildConfig,
    build_teacher_imitation_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dung dataset imitation tu nhan dinh/day lich su cua Bottom/Top EA."
    )
    parser.add_argument("--labels", required=True, help="bottom_top_historical_labels.csv")
    parser.add_argument("--output", required=True, help="File .npz dau ra")
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument(
        "--confirmation-bars", type=int, default=3,
        help="So bar ben phai can co truoc khi gan nhan dinh/day (mac dinh: 3).",
    )
    parser.add_argument(
        "--max-confirmation-wait-bars", type=int, default=5,
        help="Cho toi da them bao nhieu phut de tim observation HOLD co san.",
    )
    args = parser.parse_args()
    result = build_teacher_imitation_dataset(
        args.labels,
        args.output,
        config=TeacherDatasetBuildConfig(
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            confirmation_bars=args.confirmation_bars,
            max_confirmation_wait_bars=args.max_confirmation_wait_bars,
        ),
    )
    print(json.dumps({
        "output": str(result.output_path),
        "manifest": str(result.manifest_path),
        "samples": result.sample_count,
        "splits": result.split_counts,
        "actions": result.action_counts,
        "skipped": result.skipped_counts,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
