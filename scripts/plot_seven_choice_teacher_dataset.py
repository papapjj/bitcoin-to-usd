#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACTION_NAMES = (
    "Giữ",
    "Mua thị trường",
    "Bán thị trường",
    "Mua giới hạn",
    "Bán giới hạn",
    "Mua dừng",
    "Bán dừng",
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Vẽ báo cáo tập dạy bảy lựa chọn.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with np.load(args.dataset.resolve(), allow_pickle=False) as archive:
        best = np.asarray(archive["best_actions"], dtype=np.int64)
        splits = np.asarray(archive["splits"])
        filled = np.asarray(archive["counterfactual_filled"], dtype=np.bool_)
        wait = np.asarray(archive["counterfactual_fill_wait_bars"], dtype=np.int64)
        net_r = np.asarray(archive["counterfactual_net_r"], dtype=np.float32)

    colors = ("#64748b", "#059669", "#dc2626", "#10b981", "#ef4444", "#14b8a6", "#f97316")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    x = np.arange(7)

    split_names = (("train", "Học"), ("validation", "Kiểm định"), ("test", "Thử"))
    width = 0.25
    for offset, (key, label) in enumerate(split_names):
        mask = splits == key
        counts = np.asarray([(best[mask] == action).sum() for action in range(7)])
        axes[0, 0].bar(x + (offset - 1) * width, counts, width=width, label=label)
    axes[0, 0].set_title("Số lần mỗi lựa chọn là nhãn tốt nhất")
    axes[0, 0].set_ylabel("Số mẫu")
    axes[0, 0].legend()

    fill_rates = filled.mean(axis=0) * 100.0
    axes[0, 1].bar(x, fill_rates, color=colors)
    axes[0, 1].set_title("Khả năng khớp mô phỏng")
    axes[0, 1].set_ylabel("Tỷ lệ (%)")
    axes[0, 1].set_ylim(0, 105)
    for i, value in enumerate(fill_rates):
        axes[0, 1].text(i, value + 1.0, f"{value:.1f}", ha="center", fontsize=8)

    means = net_r.mean(axis=0)
    axes[1, 0].bar(x, means, color=colors)
    axes[1, 0].axhline(0.0, color="#0f172a", linewidth=1)
    axes[1, 0].set_title("Kết quả ròng trung bình của nhãn mô phỏng")
    axes[1, 0].set_ylabel("Đơn vị rủi ro ban đầu")

    pending = (3, 4, 5, 6)
    wait_means = [float(wait[filled[:, action], action].mean()) for action in pending]
    axes[1, 1].bar(np.arange(len(pending)), wait_means, color=[colors[i] for i in pending])
    axes[1, 1].set_title("Thời gian chờ trung bình khi lệnh chờ khớp")
    axes[1, 1].set_ylabel("Số nến M1")
    axes[1, 1].set_xticks(np.arange(len(pending)), [ACTION_NAMES[i] for i in pending])

    for axis in axes.flat:
        if axis is not axes[1, 1]:
            axis.set_xticks(x, ACTION_NAMES)
        axis.tick_params(axis="x", rotation=28)
        axis.grid(axis="y", alpha=0.2)

    fig.suptitle(
        f"Tập dạy bảy lựa chọn — {len(best):,} mẫu\n"
        "Các kết quả dùng dữ liệu tương lai để tạo nhãn; không phải bằng chứng MT5 có lãi",
        fontsize=15,
    )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
