"""Export one verified MT5 diagnosis from the actual tester ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from self_growing_trader.mt5_live_diagnostics import Mt5LiveDiagnostics


ACTION_LABELS = {
    "HOLD": "Đứng ngoài",
    "MARKET_BUY": "Mua ngay",
    "MARKET_SELL": "Bán ngay",
    "BUY_LIMIT": "Chờ mua thấp",
    "SELL_LIMIT": "Chờ bán cao",
    "BUY_STOP": "Chờ mua phá",
    "SELL_STOP": "Chờ bán phá",
    "CANCEL_ORDERS": "Hủy lệnh chờ",
    "CLOSE_ALL": "Đóng tất cả",
    "MANAGE_POSITION": "Quản lý vị thế",
}


def _read_json(path: Path) -> dict[str, Any]:
    for _ in range(3):
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (OSError, json.JSONDecodeError):
            time.sleep(0.05)
    return {}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "chưa có"
    return f"{float(value):,.{digits}f}"


def build_snapshot(output: Path, runner: Path) -> dict[str, Any]:
    published = _read_json(output / "mt5_diagnostics.json")
    direct = Mt5LiveDiagnostics(output, runner).refresh()
    direct["verified_sources"] = {
        "performance_risk_policy_execution": "tester_ledgers_reparsed_from_start",
        "optimizer_reward": "running_monitor_observatory_state",
        "published_monitor_was_missing_tester_ledgers": bool(
            int(dict(published.get("performance", {}) or {}).get("closed_trades", 0) or 0) == 0
            and int(dict(direct.get("performance", {}) or {}).get("closed_trades", 0) or 0) > 0
        ),
    }
    if published:
        direct["optimizer"] = dict(published.get("optimizer", {}) or direct["optimizer"])
        direct["reward"] = dict(published.get("reward", {}) or direct["reward"])
        published_policy = dict(published.get("policy", {}) or {})
        direct["policy"]["reasoning_action_scale"] = published_policy.get(
            "reasoning_action_scale", direct["policy"].get("reasoning_action_scale")
        )
        direct["policy"]["imitation_prior_enabled"] = published_policy.get(
            "imitation_prior_enabled", direct["policy"].get("imitation_prior_enabled")
        )
    return direct


def write_report(snapshot: dict[str, Any], path: Path) -> None:
    performance = dict(snapshot["performance"])
    risk = dict(snapshot["risk"])
    policy = dict(snapshot["policy"])
    execution = dict(snapshot["execution"])
    optimizer = dict(snapshot["optimizer"])
    reward = dict(snapshot["reward"])
    profit_factor = performance.get("profit_factor")
    expectancy = performance.get("expectancy")
    hold_share = policy.get("hold_share")
    timeout_rate = execution.get("unfilled_timeout_rate")
    credited = float(reward.get("closed_trades_credited", 0.0) or 0.0)
    final_closes = int(performance.get("final_position_closes", 0) or 0)
    final_close_rows = int(performance.get("final_close_rows", final_closes) or 0)
    duplicate_final_close_rows = int(performance.get("duplicate_final_close_rows", 0) or 0)
    close_reason_net_pnl = dict(performance.get("close_reason_net_pnl", {}) or {})
    lines = [
        "# Chẩn đoán MT5 từ sổ thực thi thật",
        "",
        f"- Thời gian thị trường gần nhất: `{snapshot.get('market_time', '')}`.",
        f"- Số lần chốt khớp: **{int(performance.get('closed_deals', 0) or 0)}**; vị thế độc lập đã đóng hết: **{final_closes}**.",
        f"- Lãi ròng: **{_fmt(performance.get('net_pnl'))}**; số dư: **{_fmt(risk.get('balance'))}**; mức giảm vốn lớn nhất: **{_fmt(100.0 * float(risk.get('max_equity_drawdown_pct', 0.0) or 0.0))}%**.",
        f"- Hệ số lợi nhuận: **{_fmt(profit_factor, 3)}**; kỳ vọng mỗi lần chốt: **{_fmt(expectancy, 3)}**.",
        f"- Tỷ lệ thắng: **{_fmt(100.0 * float(performance.get('win_rate', 0.0) or 0.0))}%**; tỷ lệ hòa vốn cần có: **{_fmt(100.0 * float(performance.get('breakeven_win_rate', 0.0) or 0.0))}%**.",
        f"- Đứng ngoài chiếm **{_fmt(100.0 * float(hold_share or 0.0))}%**; lệnh chờ hết hạn chưa khớp chiếm **{_fmt(100.0 * float(timeout_rate or 0.0))}%**.",
        f"- Số lượt cập nhật chính sách: **{int(optimizer.get('ppo_updates', 0) or 0)}**; số giao dịch được ghi công trong tín hiệu học: **{_fmt(credited, 0)}**.",
        f"- Giữ phần học bắt chước khi tự học: **{'bật' if policy.get('imitation_prior_enabled') else 'tắt'}**; tác động phần suy luận lên hành động: **{_fmt(policy.get('reasoning_action_scale'), 3)}**.",
        "",
        "## Kết luận",
        "",
    ]
    if duplicate_final_close_rows > 0:
        lines.insert(
            -3,
            f"- Có **{final_close_rows}** dòng mang cờ đóng cuối nhưng chỉ **{final_closes}** vị thế độc lập; "
            f"**{duplicate_final_close_rows}** dòng đóng cuối bị lặp theo mã vị thế.",
        )
    if close_reason_net_pnl:
        reason_labels = {"EXPERT": "tác nhân chủ động đóng", "SL": "cắt lỗ", "TP": "chốt lời"}
        reason_parts = [
            f"{reason_labels.get(reason, reason)} {float(pnl):+,.2f}"
            for reason, pnl in sorted(close_reason_net_pnl.items())
        ]
        lines.insert(-3, "- Lãi ròng theo nguyên nhân chốt: **" + "; ".join(reason_parts) + "**.")
    if profit_factor is not None and float(profit_factor) < 1.0 and expectancy is not None and float(expectancy) < 0.0:
        lines.append("Mô hình đang có lợi thế âm ngay trên dữ liệu MT5 đã thực thi; tiếp tục nhiều bước hơn không tự biến tín hiệu này thành có lợi nhuận.")
    if hold_share is not None and float(hold_share) > 0.80:
        lines.append("Phân bố hành động đang co mạnh về đứng ngoài, nhưng các lần vào lệnh còn lại vẫn gây lỗ lớn.")
    if timeout_rate is not None and float(timeout_rate) > 0.20:
        lines.append("Chất lượng lệnh chờ kém: nhiều lệnh hết hạn trước khi khớp, cho thấy kiểu vào lệnh chưa được học cùng nhãn đáy/đỉnh.")
    stop_loss_pnl = float(close_reason_net_pnl.get("SL", 0.0) or 0.0)
    expert_close_pnl = float(close_reason_net_pnl.get("EXPERT", 0.0) or 0.0)
    take_profit_pnl = float(close_reason_net_pnl.get("TP", 0.0) or 0.0)
    if stop_loss_pnl < 0.0 and expert_close_pnl >= 0.0 and take_profit_pnl >= 0.0:
        lines.append(
            "Khoản lỗ tập trung ở các lần cắt lỗ; tác nhân chủ động đóng và chốt lời đều có tổng lãi dương. "
            "Cần sửa chất lượng điểm vào cùng khoảng cắt lỗ/chốt lời trước khi tăng thời gian tự học."
        )
    if credited > max(1, final_closes) * 1.5:
        lines.append("Số lần ghi công trong tín hiệu học lớn hơn nhiều số vị thế hoàn tất; cần coi đây là lượt ghi công theo tác nhân/bước, không phải số giao dịch độc lập.")
    if not policy.get("imitation_prior_enabled"):
        lines.append("Lượt chạy đã bỏ điều kiện xác minh nên cơ chế giữ phần học bắt chước bị tắt; cập nhật trực tuyến có thể làm lệch nhanh ranh giới vào lệnh đã học.")
    lines.extend([
        "",
        "Nguồn số giao dịch, hành động và vốn được đọc lại trực tiếp từ các sổ của tác nhân thử nghiệm; số cập nhật và tín hiệu học lấy từ trạng thái được xuất gần nhất của bộ theo dõi.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_chart(snapshot: dict[str, Any], path: Path) -> None:
    performance = dict(snapshot["performance"])
    risk = dict(snapshot["risk"])
    policy = dict(snapshot["policy"])
    execution = dict(snapshot["execution"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    figure.suptitle("Chẩn đoán MT5 từ sổ thực thi thật", fontsize=15, fontweight="bold")

    action_counts = dict(policy.get("action_counts", {}) or {})
    ordered_actions = sorted(action_counts, key=action_counts.get, reverse=True)
    axes[0, 0].bar(
        [ACTION_LABELS.get(name, name) for name in ordered_actions],
        [action_counts[name] for name in ordered_actions],
        color="#315f9c",
    )
    axes[0, 0].set_title("Phân bố hành động")
    axes[0, 0].tick_params(axis="x", rotation=35)
    axes[0, 0].set_ylabel("Số lần")
    axes[0, 0].set_yscale("log")

    close_reasons = dict(performance.get("close_reasons", {}) or {})
    close_reason_net_pnl = dict(performance.get("close_reason_net_pnl", {}) or {})
    reason_labels = {"EXPERT": "Tác nhân đóng", "SL": "Cắt lỗ", "TP": "Chốt lời"}
    reason_names = list(close_reasons)
    reason_colors = [
        "#3d9b68" if float(close_reason_net_pnl.get(name, 0.0) or 0.0) >= 0.0 else "#c84f4f"
        for name in reason_names
    ]
    axes[0, 1].bar(
        [reason_labels.get(name, name) for name in reason_names],
        [close_reasons[name] for name in reason_names],
        color=reason_colors,
    )
    axes[0, 1].set_title("Nguyên nhân đóng")
    axes[0, 1].set_ylabel("Số lần chốt")
    for index, reason in enumerate(reason_names):
        pnl = float(close_reason_net_pnl.get(reason, 0.0) or 0.0)
        axes[0, 1].text(
            index,
            float(close_reasons[reason]),
            f"{pnl:+,.0f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    funnel = {
        "Yêu cầu": int(execution.get("requests", 0) or 0),
        "Được nhận": int(execution.get("accepted", 0) or 0),
        "Đã đặt": int(execution.get("placed", 0) or 0),
        "Đã khớp": int(execution.get("filled", 0) or 0),
        "Đóng hết": int(performance.get("final_position_closes", 0) or 0),
    }
    axes[1, 0].bar(funnel.keys(), funnel.values(), color="#d28d32")
    axes[1, 0].set_title("Đường đi của lệnh")
    axes[1, 0].tick_params(axis="x", rotation=25)
    axes[1, 0].set_ylabel("Số lượng")

    axes[1, 1].axis("off")
    key_text = (
        f"Lãi ròng                 {_fmt(performance.get('net_pnl'))}\n"
        f"Hệ số lợi nhuận          {_fmt(performance.get('profit_factor'), 3)}\n"
        f"Kỳ vọng/lần chốt         {_fmt(performance.get('expectancy'), 3)}\n"
        f"Tỷ lệ thắng              {_fmt(100.0 * float(performance.get('win_rate', 0.0) or 0.0))}%\n"
        f"Tỷ lệ thắng hòa vốn      {_fmt(100.0 * float(performance.get('breakeven_win_rate', 0.0) or 0.0))}%\n"
        f"Giảm vốn lớn nhất        {_fmt(100.0 * float(risk.get('max_equity_drawdown_pct', 0.0) or 0.0))}%\n"
        f"Đứng ngoài               {_fmt(100.0 * float(policy.get('hold_share', 0.0) or 0.0))}%\n"
        f"Lệnh chờ hết hạn         {_fmt(100.0 * float(execution.get('unfilled_timeout_rate', 0.0) or 0.0))}%"
    )
    axes[1, 1].text(
        0.04, 0.96, key_text, va="top", family="DejaVu Sans", fontsize=11,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#f4f6f8", "edgecolor": "#9aa5b1"},
    )
    axes[1, 1].set_title("Chỉ số chính")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runner-dir", type=Path, required=True)
    parser.add_argument("--name", default="chan_doan_hien_tai")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    runner = args.runner_dir.resolve()
    snapshot = build_snapshot(output, runner)
    json_path = output / f"{args.name}.json"
    report_path = output / f"{args.name}.vi.md"
    chart_path = output / f"{args.name}.png"
    json_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(snapshot, report_path)
    write_chart(snapshot, chart_path)
    print(json.dumps({
        "json": str(json_path), "report": str(report_path), "chart": str(chart_path),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
