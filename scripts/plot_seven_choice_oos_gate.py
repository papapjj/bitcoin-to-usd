from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ACTION_ORDER = [
    "HOLD",
    "MARKET_BUY",
    "MARKET_SELL",
    "BUY_LIMIT",
    "SELL_LIMIT",
    "BUY_STOP",
    "SELL_STOP",
]

ACTION_LABELS = {
    "HOLD": "ĐỨNG NGOÀI",
    "MARKET_BUY": "MUA THỊ TRƯỜNG",
    "MARKET_SELL": "BÁN THỊ TRƯỜNG",
    "BUY_LIMIT": "MUA CHỜ THẤP",
    "SELL_LIMIT": "BÁN CHỜ CAO",
    "BUY_STOP": "MUA CHỜ VƯỢT",
    "SELL_STOP": "BÁN CHỜ PHÁ",
}


def _number(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _position_results(path: Path) -> list[float]:
    accumulated: dict[str, float] = {}
    completed: set[str] = set()
    completion_order: list[str] = []
    close_index = 0
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("closed") != "1":
                continue
            close_index += 1
            ticket = str(row.get("position_ticket", "")).strip()
            key = ticket or f"deal:{close_index}"
            pnl = _number(row.get("profit")) + _number(row.get("commission")) + _number(row.get("swap"))
            accumulated[key] = accumulated.get(key, 0.0) + pnl
            if row.get("position_final_close") == "1" and key not in completed:
                completed.add(key)
                completion_order.append(key)
    return [accumulated[key] for key in completion_order]


def _action_counts(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row.get("transaction_type") == "BATCH_RESOLUTION" and row.get("source") == "policy":
                counts[str(row.get("action_name", "UNKNOWN"))] += 1
    return counts


def _running_bounds(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    means: list[float] = []
    lower: list[float] = []
    for end in range(1, len(values) + 1):
        sample = np.asarray(values[:end], dtype=np.float64)
        mean = float(sample.mean())
        spread = float(sample.std(ddof=1)) if end > 1 else 0.0
        means.append(mean)
        lower.append(mean - 1.64 * spread / math.sqrt(end))
    return np.asarray(means), np.asarray(lower)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ve ket qua cong chan loi nhuan nhanh bay lua chon")
    parser.add_argument("--execution", required=True)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    execution = Path(args.execution).resolve()
    diagnostics_path = Path(args.diagnostics).resolve()
    output = Path(args.output).resolve()
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    positions = _position_results(execution)
    counts = _action_counts(execution)
    if not positions:
        raise RuntimeError("chua co vi the dong hoan tat de ve")

    pnl = np.asarray(positions, dtype=np.float64)
    cumulative = np.cumsum(pnl)
    mean, lower = _running_bounds(positions)
    performance = diagnostics.get("performance", {})
    optimizer = diagnostics.get("optimizer", {})
    risk = diagnostics.get("risk", {})

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10})
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), constrained_layout=True)
    fig.suptitle("Kiểm định nhánh bảy lựa chọn trên MT5", fontsize=17, fontweight="bold")

    x = np.arange(1, len(pnl) + 1)
    axes[0, 0].plot(x, cumulative, color="#b42318", linewidth=2)
    axes[0, 0].axhline(0.0, color="#475467", linewidth=1)
    axes[0, 0].fill_between(x, cumulative, 0.0, where=cumulative < 0.0, color="#fda29b", alpha=0.35)
    axes[0, 0].set_title("Lợi nhuận tích lũy theo vị thế")
    axes[0, 0].set_xlabel("Số vị thế đã đóng")
    axes[0, 0].set_ylabel("Lợi nhuận sau phí")
    axes[0, 0].grid(alpha=0.2)

    axes[0, 1].plot(x, mean, label="Trung bình", color="#175cd3", linewidth=2)
    axes[0, 1].plot(x, lower, label="Cận dưới một phía 95%", color="#b42318", linewidth=2)
    axes[0, 1].axhline(0.0, color="#475467", linewidth=1)
    axes[0, 1].set_title("Cận dưới lợi nhuận mỗi vị thế")
    axes[0, 1].set_xlabel("Số vị thế đã đóng")
    axes[0, 1].set_ylabel("Lợi nhuận mỗi vị thế")
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.2)

    labels = [ACTION_LABELS[name] for name in ACTION_ORDER]
    values = [counts[name] for name in ACTION_ORDER]
    bars = axes[1, 0].bar(labels, values, color="#7f56d9")
    axes[1, 0].set_title("Các lựa chọn do mạng phát ra")
    axes[1, 0].set_ylabel("Số lần")
    axes[1, 0].set_yscale("symlog", linthresh=1.0)
    axes[1, 0].bar_label(bars, labels=[f"{value:,}" for value in values], padding=3)
    axes[1, 0].tick_params(axis="x", rotation=28, labelsize=8)
    axes[1, 0].grid(axis="y", alpha=0.2)

    passed = bool(
        len(positions) >= 200
        and lower[-1] > 0.0
        and int(optimizer.get("ppo_updates", 0) or 0) == 0
    )
    verdict = "ĐẠT — được phép thử PPO" if passed else "KHÔNG ĐẠT — PPO tiếp tục bị khóa"
    details = [
        verdict,
        "",
        f"Vị thế đóng duy nhất: {len(positions):,}",
        f"Lợi nhuận sau phí: {float(cumulative[-1]):,.2f}",
        f"Trung bình mỗi vị thế: {float(mean[-1]):,.2f}",
        f"Cận dưới một phía 95%: {float(lower[-1]):,.2f}",
        f"Hệ số lợi nhuận: {_number(performance.get('profit_factor')):.3f}",
        f"Sụt giảm vốn lớn nhất: {100.0 * _number(risk.get('max_equity_drawdown_pct')):.2f}%",
        f"Số lần PPO cập nhật: {int(optimizer.get('ppo_updates', 0) or 0):,}",
    ]
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.03,
        0.95,
        "\n".join(details),
        va="top",
        fontsize=13,
        linespacing=1.45,
        color="#067647" if passed else "#b42318",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({
        "output": str(output),
        "positions": len(positions),
        "net_pnl": float(cumulative[-1]),
        "position_mean": float(mean[-1]),
        "position_lcb_95_one_sided": float(lower[-1]),
        "ppo_gate_passed": passed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
