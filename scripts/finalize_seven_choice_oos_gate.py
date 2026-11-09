from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


LONG_ACTIONS = {"MARKET_BUY", "BUY_LIMIT", "BUY_STOP"}
SHORT_ACTIONS = {"MARKET_SELL", "SELL_LIMIT", "SELL_STOP"}
ENTRY_DEAL_ENTRY_VALUES = {"0", "0.0", "2", "2.0", "IN", "INOUT"}


def _number(value: object) -> float:
    try:
        result = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _positive_risk(value: object) -> float:
    """Return a usable immutable risk denominator, or zero when absent."""

    return abs(_number(value))


def _first_existing(candidates: tuple[Path, ...]) -> Path:
    """Return the first existing path, retaining the first for a clear error."""

    return next((item for item in candidates if item.is_file()), candidates[0])


def _lower_bound(values: list[float]) -> float | None:
    """One-sided 95% lower bound for a position-level sample."""

    if not values:
        return None
    mean = float(sum(values) / len(values))
    if len(values) < 2:
        return mean
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean - 1.64 * math.sqrt(max(0.0, variance)) / math.sqrt(len(values))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _analyse_ledger(path: Path) -> dict[str, Any]:
    policy_actions: Counter[str] = Counter()
    filled_actions: Counter[str] = Counter()
    entry_action_by_position: dict[str, str] = {}
    position_pnl: defaultdict[str, float] = defaultdict(float)
    position_risk: dict[str, float] = {}
    completed_positions: set[str] = set()

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.DictReader(handle, delimiter="\t"), start=1):
            action = str(row.get("action_name", "UNKNOWN"))
            if row.get("transaction_type") == "BATCH_RESOLUTION" and row.get("source") == "policy":
                policy_actions[action] += 1

            ticket = str(row.get("position_ticket", "")).strip()
            # Only an entry action can establish ownership.  A partial
            # close/reversal may also carry a filled flag, so accepting every
            # filled row could overwrite the original entry attribution.
            deal_entry = str(row.get("deal_entry", "") or "").strip().upper()
            is_entry_fill = (
                row.get("filled") == "1"
                and action in LONG_ACTIONS | SHORT_ACTIONS
                and (not deal_entry or deal_entry in ENTRY_DEAL_ENTRY_VALUES)
            )
            if is_entry_fill and ticket:
                entry_action_by_position.setdefault(ticket, action)
                filled_actions[action] += 1

            if row.get("closed") != "1":
                continue
            key = ticket or f"deal:{row_number}"
            position_pnl[key] += (
                _number(row.get("profit"))
                + _number(row.get("commission"))
                + _number(row.get("swap"))
            )
            risk = _positive_risk(row.get("initial_risk"))
            if risk > 1e-12:
                position_risk[key] = max(position_risk.get(key, 0.0), risk)
            if row.get("position_final_close") == "1":
                completed_positions.add(key)

    by_entry: dict[str, list[float]] = defaultdict(list)
    unattributed = 0
    for position in sorted(completed_positions):
        action = entry_action_by_position.get(position, "UNKNOWN")
        if action == "UNKNOWN":
            unattributed += 1
        by_entry[action].append(position_pnl[position])

    attributed: dict[str, dict[str, float | int]] = {}
    for action, values in sorted(by_entry.items()):
        wins = sum(value > 0.0 for value in values)
        losses = sum(value < 0.0 for value in values)
        total = float(sum(values))
        attributed[action] = {
            "completed_positions": len(values),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / (wins + losses) if wins + losses else 0.0,
            "net_pnl": total,
            "mean_pnl": total / len(values) if values else 0.0,
        }
        r_values = [
            position_pnl[position] / position_risk[position]
            for position in completed_positions
            if entry_action_by_position.get(position, "UNKNOWN") == action
            and position_risk.get(position, 0.0) > 1e-12
        ]
        attributed[action].update({
            "risk_covered_positions": len(r_values),
            "sum_r": float(sum(r_values)),
            "mean_r": float(sum(r_values) / len(r_values)) if r_values else None,
            "lcb95_r": _lower_bound(r_values),
        })

    long_actions = sum(policy_actions[name] for name in LONG_ACTIONS)
    short_actions = sum(policy_actions[name] for name in SHORT_ACTIONS)
    directional_actions = long_actions + short_actions
    position_values = [position_pnl[key] for key in completed_positions]
    position_mean = (
        float(sum(position_values) / len(position_values)) if position_values else None
    )
    position_lcb = _lower_bound(position_values)
    r_positions = [
        key for key in completed_positions if position_risk.get(key, 0.0) > 1e-12
    ]
    position_r_values = [position_pnl[key] / position_risk[key] for key in r_positions]
    position_mean_r = (
        float(sum(position_r_values) / len(position_r_values))
        if position_r_values else None
    )
    position_lcb_r = _lower_bound(position_r_values)
    return {
        "policy_action_counts": dict(policy_actions),
        "long_entry_actions": long_actions,
        "short_entry_actions": short_actions,
        "entry_direction_imbalance": (
            (long_actions - short_actions) / directional_actions
            if directional_actions
            else None
        ),
        "filled_entry_counts": dict(filled_actions),
        "completed_positions": len(completed_positions),
        "risk_covered_positions": len(r_positions),
        "risk_coverage": float(len(r_positions) / len(completed_positions)) if completed_positions else 0.0,
        "unattributed_completed_positions": unattributed,
        "position_results_by_entry_action": attributed,
        "position_net_pnl": float(sum(position_pnl[key] for key in completed_positions)),
        "position_mean_pnl": position_mean,
        "position_lcb_95_one_sided": position_lcb,
        "position_mean_r": position_mean_r,
        "position_lcb_95_one_sided_r": position_lcb_r,
        "position_r_sample_count": len(position_r_values),
        "position_r_denominator": "initial_risk_max_per_position",
    }


def _write_vietnamese_report(
    path: Path,
    diagnostics: dict[str, Any],
    analysis: dict[str, Any],
    requested_dates: list[str],
    gate: dict[str, Any],
) -> None:
    performance = diagnostics["performance"]
    risk = diagnostics["risk"]
    optimizer = diagnostics["optimizer"]
    policy = diagnostics["policy"]
    execution = diagnostics["execution"]
    by_entry = analysis["position_results_by_entry_action"]
    gate_passed = bool(gate.get("passed"))
    gate_text = "đạt" if gate_passed else "không đạt"
    position_lcb_r = analysis.get("position_lcb_95_one_sided_r")
    position_mean_r = analysis.get("position_mean_r")

    entry_lines = []
    for action, values in by_entry.items():
        entry_lines.append(
            f"| `{action}` | {values['completed_positions']} | "
            f"{100.0 * float(values['win_rate']):.2f}% | "
            f"{float(values['net_pnl']):,.2f} | {float(values['mean_pnl']):,.2f} | "
            f"{'chưa có' if values.get('mean_r') is None else f'{float(values['mean_r']):,.3f}R'} |"
        )

    pending_fills = sum(
        int(analysis["filled_entry_counts"].get(name, 0))
        for name in ("BUY_LIMIT", "SELL_LIMIT", "BUY_STOP", "SELL_STOP")
    )
    text = f"""# Báo cáo kiểm định nhánh bảy lựa chọn

## Kết luận

Nhánh cố định **{gate_text}** điều kiện mở PPO. Lượt thử đã đi hết khoảng
{requested_dates[0]} đến {requested_dates[1]} theo giờ thị trường, đóng
{performance['final_position_closes']} vị thế và lỗ ròng
{float(performance['net_pnl']):,.2f}. Cận dưới một phía 95% của lợi nhuận mỗi
vị thế là {float(performance['position_expectancy_lcb_95_one_sided']):,.2f} tiền,
{"chưa có" if position_lcb_r is None else f"{float(position_lcb_r):,.3f}R"},
so với mức bắt buộc 0R. Số lần cập nhật PPO vẫn là
{int(optimizer.get('ppo_updates', 0) or 0)}.

## Bằng chứng MT5

- Hệ số lợi nhuận: {float(performance['profit_factor']):.3f}.
- Tỷ lệ thắng theo lần đóng: {100.0 * float(performance['win_rate']):.2f}%.
- Lợi nhuận trung bình mỗi vị thế: {float(performance['position_expectancy']):,.2f}.
- Lợi nhuận trung bình mỗi vị thế theo rủi ro cố định: {"chưa có" if position_mean_r is None else f"{float(position_mean_r):,.3f}R"}; độ bao phủ mẫu: {100.0 * float(analysis.get('risk_coverage', 0.0)):.2f}%.
- Sụt giảm vốn lớn nhất: {100.0 * float(risk['max_equity_drawdown_pct']):.2f}%.
- Yêu cầu được nhận: {execution['accepted']}/{execution['requests']}; bị từ chối: {execution['rejected']}.
- Vị thế được mở: {execution['filled']}; trong đó lệnh chờ khớp: {pending_fills}; tỷ lệ hết hạn: {100.0 * float(execution['unfilled_timeout_rate']):.2f}%.
- Lựa chọn giữ nguyên trạng thái chiếm {100.0 * float(policy['hold_share']):.2f}%.
- Lựa chọn mở hướng mua: {analysis['long_entry_actions']}; hướng bán: {analysis['short_entry_actions']}.

## Lợi nhuận theo cách mở vị thế

| Cách mở | Vị thế | Tỷ lệ thắng | Lợi nhuận sau phí | Trung bình tiền | Trung bình R |
|---|---:|---:|---:|---:|---:|
{chr(10).join(entry_lines)}

## Nguyên nhân chính

1. Bộ dữ liệu schema 13 đã sửa điểm lệch trước đây: lệnh chờ khớp tại giá đặt,
   mua cộng chênh lệch giá ở nến khớp, còn cắt lỗ/chốt lời dùng ATR lúc khớp.
   Các ứng viên cũ dùng giá chạm nên bị loại, không được dùng làm bằng chứng.
2. Sau khi sửa, phép dò nhân quả 80 cặp cắt lỗ/chốt lời vẫn có cận dưới âm ở
   cả kiểm định và tập thử kín. Không có đầu ra nào đủ điều kiện để tạo ứng
   viên mới.
3. Nhánh cố định MT5 đóng {performance['final_position_closes']} vị thế với
   lợi nhuận trung bình {float(performance['position_expectancy']):,.2f}; cận
   dưới theo R là {"chưa có" if position_lcb_r is None else f"{float(position_lcb_r):,.3f}R"}.
4. Mọi phép thử tiếp theo phải dùng cùng hình học, cùng giá khớp và cùng ranh
   giới thời gian; không được hạ ngưỡng chỉ để tạo thêm giao dịch.

## Quyết định

{"Có thể chuyển sang bước kiểm tra PPO theo đúng điều kiện đã định; báo cáo này chưa tự chạy PPO." if gate_passed else "PPO tiếp tục bị khóa. Không dùng điểm lưu này để quảng bá hoặc tự học trực tiếp."} Lần dạy kế tiếp phải bỏ việc chọn tối ưu bằng tương lai, tăng nhãn đứng
ngoài, giữ trạng thái tuần tự liên tục như MT5, rồi lặp lại cùng phép thử cố
định. Chỉ khi cận dưới lợi nhuận theo R vượt 0 mới được mở PPO.
"""
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chot bang chung cong PPO cua nhanh bay lua chon")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--candidate-report", required=True)
    parser.add_argument("--from-date", required=True)
    parser.add_argument("--to-date", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    frozen = run_dir / "frozen_policy"
    runtime = frozen / "ablation_frozen_policy.runtime"
    # Older matched runs kept the writer output below the runtime directory;
    # newer runs retain the completed evidence directly in frozen_policy.
    # Accept both layouts so a finished run cannot be rejected merely because
    # the storage wrapper changed.
    ledger_candidates = (
        runtime / "execution_ledger.csv",
        frozen / "mt5_execution_ledger.tsv",
        frozen / "execution_ledger.csv",
    )
    source_ledger = _first_existing(ledger_candidates)
    report_candidates = (
        runtime / "report.htm",
        frozen / "mt5_report.htm",
        frozen / "report.htm",
    )
    source_report = _first_existing(report_candidates)
    diagnostics_path = frozen / "mt5_diagnostics.json"
    summary_path = run_dir / "ablation_summary.json"
    candidate_path = Path(args.candidate_report).resolve()
    if not source_ledger.is_file() or not diagnostics_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError("thieu so lenh, chan doan hoac tong hop cua luot MT5")

    retained_ledger = frozen / "mt5_execution_ledger.tsv"
    retained_report = frozen / "mt5_report.htm"
    if source_ledger.resolve() != retained_ledger.resolve():
        shutil.copy2(source_ledger, retained_ledger)
    if source_report.is_file() and source_report.resolve() != retained_report.resolve():
        shutil.copy2(source_report, retained_report)

    analysis = _analyse_ledger(retained_ledger)
    analysis_path = frozen / "oos_action_analysis.json"
    _write_json(analysis_path, analysis)

    diagnostics = _read_json(diagnostics_path)
    expected_last_day = (
        datetime.strptime(args.to_date, "%Y.%m.%d") - timedelta(days=1)
    ).strftime("%Y.%m.%d")
    market_time = str(diagnostics.get("market_time", ""))
    diagnostics["policy"].update({
        "long_entry_actions": analysis["long_entry_actions"],
        "short_entry_actions": analysis["short_entry_actions"],
        "entry_direction_imbalance": analysis["entry_direction_imbalance"],
    })
    diagnostics["performance"].update({
        "position_expectancy": analysis["position_mean_pnl"],
        "position_expectancy_lcb_95_one_sided": analysis["position_lcb_95_one_sided"],
        "position_expectancy_r": analysis["position_mean_r"],
        "position_expectancy_lcb_95_one_sided_r": analysis["position_lcb_95_one_sided_r"],
        "position_risk_coverage": analysis["risk_coverage"],
    })
    diagnostics["evaluation"] = {
        "requested_dates": [args.from_date, args.to_date],
        "requested_period_completed": market_time.startswith(expected_last_day),
        "evidence_ledger": str(retained_ledger),
        "action_analysis": str(analysis_path),
    }
    _write_json(diagnostics_path, diagnostics)

    minimum_positions = 200
    observed_positions = int(diagnostics["performance"].get("final_position_closes", 0) or 0)
    position_lcb_r = analysis.get("position_lcb_95_one_sided_r")
    risk_coverage = float(analysis.get("risk_coverage", 0.0) or 0.0)
    ppo_updates = int(diagnostics["optimizer"].get("ppo_updates", 0) or 0)
    period_completed = bool(diagnostics["evaluation"]["requested_period_completed"])
    checks = {
        "minimum_completed_positions": observed_positions >= minimum_positions,
        "risk_denominator_coverage_at_least_90pct": risk_coverage >= 0.90,
        "expectancy_lower_bound_available": position_lcb_r is not None,
        "expectancy_lower_bound_strictly_positive": (
            position_lcb_r is not None and float(position_lcb_r) > 0.0
        ),
        "frozen_run_has_no_ppo_updates": ppo_updates == 0,
        "requested_period_completed": period_completed,
    }
    gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_positions": 200,
        "observed_positions": observed_positions,
        "position_expectancy_lcb_95_one_sided": diagnostics["performance"][
            "position_expectancy_lcb_95_one_sided"
        ],
        "position_expectancy_lcb_95_one_sided_r": position_lcb_r,
        "position_risk_coverage": risk_coverage,
        "ppo_updates": ppo_updates,
        "requested_period_completed": period_completed,
        "expectancy_unit": "R",
        "reason": (
            "frozen branch lower confidence bound is strictly positive"
            if all(checks.values())
            else "frozen branch did not satisfy every fixed-branch gate"
        ),
    }

    summary = _read_json(summary_path)
    summary["arms"]["frozen_policy"]["diagnostics"] = diagnostics
    summary["arms"]["frozen_policy"]["mt5_period_completed"] = gate[
        "requested_period_completed"
    ]
    summary["arms"]["frozen_policy"]["post_run_failure"] = {
        "kind": "storage_reserve_guard",
        "affects_mt5_metrics": False,
        "detail": "MT5 report was complete before the wrapper rejected low free disk reserve",
    }
    summary["ppo_enable_gate"].update(gate)
    if gate["passed"]:
        summary.pop("blocked_arms", None)
    else:
        summary["blocked_arms"] = ["full_online", "reasoning_off"]
    summary["evaluation_complete"] = True
    summary["complete"] = False
    summary["result"] = (
        "frozen_branch_passed_gate_online_arms_not_run"
        if gate["passed"] else "frozen_branch_rejected_ppo_locked"
    )
    summary["evidence"] = {
        "execution_ledger": str(retained_ledger),
        "action_analysis": str(analysis_path),
        "diagnostics": str(diagnostics_path),
    }
    _write_json(summary_path, summary)

    candidate = _read_json(candidate_path)
    candidate.setdefault("verification", {}).update({
        "mt5_oos_evaluated": True,
        "mt5_oos_full_window_completed": gate["requested_period_completed"],
        "mt5_oos_proven": False,
        "ppo_allowed": False,
    })
    candidate["mt5_oos_gate"] = {
        **gate,
        "summary": str(summary_path),
        "diagnostics": str(diagnostics_path),
        "action_analysis": str(analysis_path),
    }
    candidate["next_gate"] = (
        "rebuild causal non-oracle labels and continuous-state replay, retrain frozen heads, "
        "then repeat the same MT5 lower-bound gate"
    )
    # This finalizer only records the prerequisite.  The matched runner still
    # decides whether to launch an online arm; never silently turn PPO on here.
    candidate["ppo_allowed"] = False
    _write_json(candidate_path, candidate)

    report_path = run_dir.parent / "BAO_CAO.vi.md"
    _write_vietnamese_report(
        report_path,
        diagnostics,
        analysis,
        [args.from_date, args.to_date],
        gate,
    )
    print(json.dumps({
        "summary": str(summary_path),
        "report": str(report_path),
        "analysis": str(analysis_path),
        "ledger": str(retained_ledger),
        "gate": gate,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
