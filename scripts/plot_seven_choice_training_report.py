#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="Vẽ kết quả học các đầu ra bảy lựa chọn.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.resolve().read_text(encoding="utf-8"))
    training = dict(report["training"])
    before = dict(training["before"])
    after = dict(training["after"])
    history = list(training["history"])
    groups = ("validation", "test")
    group_labels = ("Kiểm định", "Thử chưa từng học")
    colors = ("#94a3b8", "#0f766e")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    x = np.arange(len(groups))
    width = 0.34

    for offset, (phase, label) in enumerate(((before, "Trước"), (after, "Sau"))):
        values = [100.0 * float(phase[group]["action_accuracy"]) for group in groups]
        axes[0, 0].bar(x + (offset - 0.5) * width, values, width, label=label, color=colors[offset])
    axes[0, 0].set_title("Chọn đúng lựa chọn tốt nhất")
    axes[0, 0].set_ylabel("Tỷ lệ (%)")
    axes[0, 0].set_xticks(x, group_labels)
    axes[0, 0].legend()

    for offset, (phase, label) in enumerate(((before, "Trước"), (after, "Sau"))):
        values = [float(phase[group]["mean_net_r_regret"]) for group in groups]
        axes[0, 1].bar(x + (offset - 0.5) * width, values, width, label=label, color=colors[offset])
    axes[0, 1].set_title("Phần kết quả ròng bỏ lỡ so với nhãn tốt nhất")
    axes[0, 1].set_ylabel("Đơn vị rủi ro ban đầu; thấp hơn tốt hơn")
    axes[0, 1].set_xticks(x, group_labels)
    axes[0, 1].legend()

    output_keys = ("delay_accuracy", "entry_offset_accuracy", "sl_accuracy", "tp_accuracy")
    output_labels = ("Chờ đặt", "Giá đặt", "Cắt lỗ", "Chốt lời")
    x_outputs = np.arange(len(output_keys))
    for offset, (phase, label) in enumerate(((before, "Trước"), (after, "Sau"))):
        values = [100.0 * float(phase["test"][key]) for key in output_keys]
        axes[1, 0].bar(
            x_outputs + (offset - 0.5) * width, values, width,
            label=label, color=colors[offset],
        )
    axes[1, 0].set_title("Độ đúng các đầu ra trên phần thử")
    axes[1, 0].set_ylabel("Tỷ lệ (%)")
    axes[1, 0].set_xticks(x_outputs, output_labels)
    axes[1, 0].legend()

    epochs = [int(row["epoch"]) for row in history]
    regret = [float(row["mean_net_r_regret"]) for row in history]
    accuracy = [100.0 * float(row["action_accuracy"]) for row in history]
    left = axes[1, 1]
    right = left.twinx()
    left.plot(epochs, regret, marker="o", color="#b91c1c", label="Phần bỏ lỡ")
    right.plot(epochs, accuracy, marker="s", color="#0f766e", label="Chọn đúng")
    left.set_title("Diễn biến trên phần kiểm định")
    left.set_xlabel("Vòng học")
    left.set_ylabel("Phần kết quả ròng bỏ lỡ", color="#b91c1c")
    right.set_ylabel("Chọn đúng (%)", color="#0f766e")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Học lại các đầu ra với thân mạng cố định\n"
        "Đây là độ khớp nhãn; lợi nhuận chỉ được xác nhận bằng phép thử MT5 ngoài mẫu",
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
