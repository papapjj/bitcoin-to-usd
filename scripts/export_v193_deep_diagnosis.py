"""Export a reproducible deep diagnosis for one V193 MT5 run.

The utility compares the teacher targets, completed MT5 position lifecycles,
the epoch snapshot, and the matched MT5 ablation.  It never loads or mutates a
model checkpoint.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ACTION_LABELS = {
    "BUY_LIMIT": "Chờ mua thấp",
    "BUY_STOP": "Chờ mua phá",
    "MARKET_BUY": "Mua ngay",
    "MARKET_SELL": "Bán ngay",
    "SELL_LIMIT": "Chờ bán cao",
    "SELL_STOP": "Chờ bán phá",
}
AGE_ORDER = ["<=2", "2-5", "5-10", "10-20", "20-40", ">40"]


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def _quantiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    points = np.quantile(finite, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)),
        "minimum": float(points[0]),
        "p05": float(points[1]),
        "p25": float(points[2]),
        "median": float(points[3]),
        "p75": float(points[4]),
        "p95": float(points[5]),
        "maximum": float(points[6]),
    }


def _profit_factor(values: np.ndarray) -> float | None:
    wins = float(values[values > 0.0].sum())
    losses = float(-values[values < 0.0].sum())
    return wins / losses if losses > 0.0 else None


def _wilson_interval(wins: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total <= 0:
        return None
    rate = wins / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    groups: np.ndarray | None = None,
    samples: int = 20_000,
    seed: int = 19_315,
) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    if groups is None:
        indices = rng.integers(0, len(values), size=(samples, len(values)))
        means = values[indices].mean(axis=1)
    else:
        group_values = np.asarray(groups).astype(str)
        unique = np.unique(group_values)
        sums = np.asarray([values[group_values == key].sum() for key in unique], dtype=np.float64)
        counts = np.asarray([(group_values == key).sum() for key in unique], dtype=np.float64)
        selected = rng.integers(0, len(unique), size=(samples, len(unique)))
        means = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _position_summary(rows: list[dict[str, Any]], *, seed: int) -> dict[str, Any]:
    pnl = np.asarray([_number(row.get("net_pnl")) for row in rows], dtype=np.float64)
    risk = np.asarray([_number(row.get("initial_risk")) for row in rows], dtype=np.float64)
    valid_risk = risk > 0.0
    realized_r = pnl[valid_risk] / risk[valid_risk]
    days = np.asarray([str(row.get("entry_day", "")) for row in rows], dtype=str)
    r_days = days[valid_risk]
    wins = int((pnl > 0.0).sum())
    return {
        "positions": int(len(rows)),
        "wins": wins,
        "losses": int((pnl < 0.0).sum()),
        "win_rate": float(wins / len(rows)) if rows else 0.0,
        "win_rate_95_interval": _wilson_interval(wins, len(rows)),
        "net_pnl": float(pnl.sum()),
        "expectancy_money": float(pnl.mean()) if len(pnl) else 0.0,
        "profit_factor": _profit_factor(pnl),
        "mean_realized_r": float(realized_r.mean()) if len(realized_r) else None,
        "mean_realized_r_position_bootstrap_95_interval": _bootstrap_mean_interval(
            realized_r, seed=seed
        ),
        "mean_realized_r_day_block_bootstrap_95_interval": _bootstrap_mean_interval(
            realized_r, groups=r_days, seed=seed + 1
        ),
        "observed_days": int(len(np.unique(days))),
    }


def _teacher_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        valid = np.asarray(data["geometry_target_valid"], dtype=bool)
        accepted = np.asarray(data["teacher_trade_accepted"], dtype=bool)
        actions = np.asarray(data["actions"], dtype=np.int64)
        splits = np.asarray(data["splits"]).astype(str)
        net_r = np.asarray(data["teacher_net_r"], dtype=np.float64)[valid]
        sl_atr = np.asarray(data["teacher_sl_atr"], dtype=np.float64)[valid]
        tp_rr = np.asarray(data["teacher_tp_rr"], dtype=np.float64)[valid]
        exit_offsets = np.asarray(data["exit_bar_offsets"], dtype=np.float64)[valid]
        confirmation = np.asarray(data["confirmation_delay_minutes"], dtype=np.float64)[valid]
        mfe = np.asarray(data["teacher_mfe_r"], dtype=np.float64)[valid]
        mae = np.asarray(data["teacher_mae_r"], dtype=np.float64)[valid]
        delay = np.asarray(data["teacher_delay_indices"], dtype=np.int64)[valid]
        trailing = np.asarray(data["teacher_trailing_indices"], dtype=np.int64)[valid]
        return {
            "dataset": str(path.resolve()),
            "schema_version": int(np.asarray(data["schema_version"]).reshape(-1)[0]),
            "rows": int(len(actions)),
            "accepted_rows": int(accepted.sum()),
            "geometry_valid_rows": int(valid.sum()),
            "action_counts": {str(key): int(value) for key, value in Counter(actions.tolist()).items()},
            "split_counts": {str(key): int(value) for key, value in Counter(splits.tolist()).items()},
            "positive_share": float((actions != 0).mean()),
            "net_r": _quantiles(net_r),
            "sl_atr": _quantiles(sl_atr),
            "tp_rr": _quantiles(tp_rr),
            "exit_bars": _quantiles(exit_offsets),
            "confirmation_delay_minutes": _quantiles(confirmation),
            "delay_index_counts": {str(key): int(value) for key, value in Counter(delay.tolist()).items()},
            "trailing_index_counts": {str(key): int(value) for key, value in Counter(trailing.tolist()).items()},
            "mfe_nonzero_rows": int(np.count_nonzero(mfe)),
            "mae_nonzero_rows": int(np.count_nonzero(mae)),
            "mfe": _quantiles(mfe),
            "mae": _quantiles(mae),
            "tp_rr_values": tp_rr.tolist(),
        }


def _checkpoint_head_summary(checkpoint: Path, config_path: Path) -> dict[str, Any]:
    from self_growing_trader.config import load_config
    from self_growing_trader.execution_contract import SL_ATR_BINS, TP_RR_BINS, TRAILING_ATR_BINS
    from self_growing_trader.five_teacher_imitation import shared_five_network
    from self_growing_trader.self_growing_runner import tai_he_chuyen_gia_mot_context_mot_policy

    config = load_config(config_path)
    system = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint, config, seed=0)
    network = shared_five_network(system)
    heads = network.heads
    value_bins: dict[str, tuple[float, ...]] = {
        "sl": tuple(float(value) for value in SL_ATR_BINS),
        "tp": tuple(float(value) for value in TP_RR_BINS),
        "trailing": tuple(float(value) for value in TRAILING_ATR_BINS),
    }
    geometry: dict[str, Any] = {}
    for name in ("delay", "sl", "tp", "trailing"):
        head = heads.parameter_heads[name]
        weight = head.weight.detach().cpu().numpy()
        bias = head.bias.detach().cpu().numpy()
        selected = int(np.argmax(bias))
        geometry[name] = {
            "weight_norm": float(np.linalg.norm(weight)),
            "weight_nonzero_count": int(np.count_nonzero(weight)),
            "bias_argmax_index": selected,
            "bias_argmax_value": float(bias[selected]),
            "selected_bin_value_if_constant": (
                value_bins[name][selected]
                if name in value_bins and not np.count_nonzero(weight)
                else None
            ),
            "constant_for_all_hidden_states": not bool(np.count_nonzero(weight)),
        }
    calibrator = heads.entry_role_calibrator
    return {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "teacher_entry_imitation_trained": bool(
            getattr(calibrator, "teacher_imitation_trained", False)
        ),
        "teacher_entry_scope": str(getattr(calibrator, "teacher_entry_scope", "")),
        "teacher_geometry_imitation_trained": bool(
            getattr(heads, "teacher_geometry_imitation_trained", False)
        ),
        "teacher_geometry_imitation_schema_version": int(
            getattr(heads, "teacher_geometry_imitation_schema_version", 0) or 0
        ),
        "reasoning_action_scale": float(getattr(network, "reasoning_action_scale", 0.0)),
        "geometry_heads": geometry,
    }


def build_analysis(
    teacher_path: Path,
    lifecycle_path: Path,
    ablation_path: Path,
    checkpoint_path: Path,
    live_path: Path,
    checkpoint_model_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    teacher = _teacher_summary(teacher_path)
    lifecycle = _read_json(lifecycle_path)
    ablation = _read_json(ablation_path)
    checkpoint = _read_json(checkpoint_path)
    live = _read_json(live_path)
    positions = list(lifecycle.get("positions", []))
    if not positions:
        raise RuntimeError("bao cao vong doi lenh khong co vi the")

    by_action: dict[str, Any] = {}
    for index, action in enumerate(ACTION_LABELS):
        rows = [row for row in positions if str(row.get("entry_action")) == action]
        if rows:
            by_action[action] = _position_summary(rows, seed=19_315 + 101 * index)

    overall = _position_summary(positions, seed=29_315)
    early = [row for row in positions if _number(row.get("age_bars")) <= 5.0]
    early_summary = _position_summary(early, seed=39_315)
    early_summary["share_of_total_loss"] = (
        abs(_number(early_summary.get("net_pnl"))) / abs(_number(overall.get("net_pnl")))
        if _number(overall.get("net_pnl")) < 0.0 and _number(early_summary.get("net_pnl")) < 0.0
        else 0.0
    )
    rr_two = [row for row in positions if math.isclose(_number(row.get("effective_rr")), 2.0)]
    sl_two_five = [row for row in positions if math.isclose(_number(row.get("effective_sl_atr")), 2.5)]

    performance = dict(live.get("performance", {}) or {})
    return {
        "schema_version": 1,
        "scope": {
            "market_start": min(str(row.get("entry_time", "")) for row in positions),
            "market_end": str(live.get("market_time", "")),
            "requested_period": "2025.01.01-2025.03.31",
            "run_completed": False,
            "temporal_oos": False,
        },
        "measurement_correction": {
            "closed_deal_rows": int(lifecycle.get("closed_deal_rows", 0) or 0),
            "final_close_rows": int(lifecycle.get("final_close_rows", 0) or 0),
            "unique_completed_positions": int(
                lifecycle.get("unique_completed_positions", overall["positions"]) or 0
            ),
            "duplicate_final_close_rows": int(
                lifecycle.get("duplicate_final_close_rows", 0) or 0
            ),
        },
        "teacher": teacher,
        "runtime": {
            "overall": overall,
            "lifecycle_overall": lifecycle.get("overall", {}),
            "early_at_most_5_bars": early_summary,
            "by_entry_action": by_action,
            "rr_2_0": _position_summary(rr_two, seed=49_315),
            "sl_atr_2_5": _position_summary(sl_two_five, seed=59_315),
            "by_age_bars": lifecycle.get("by_age_bars", {}),
            "by_rr": lifecycle.get("by_rr", {}),
            "geometry_modes": lifecycle.get("geometry_modes", {}),
            "pending_timeout_rate": _number(
                dict(live.get("execution", {}) or {}).get("unfilled_timeout_rate")
            ),
            "hold_share": _number(dict(live.get("policy", {}) or {}).get("hold_share")),
            "close_reason_deal_pnl": performance.get("close_reason_net_pnl", {}),
            "close_reason_deal_counts": performance.get("close_reasons", {}),
        },
        "checkpoint": checkpoint.get("best_state_audit", {}),
        "checkpoint_status": {
            "status": checkpoint.get("status"),
            "promotable": checkpoint.get("promotable"),
            "mt5_seed_eligible": checkpoint.get("mt5_seed_eligible"),
            "reason": checkpoint.get("reason"),
        },
        "checkpoint_heads": _checkpoint_head_summary(checkpoint_model_path, config_path),
        "matched_ablation": ablation.get("primary_comparison", []),
        "matched_ablation_verdict": ablation.get("verdict", {}),
    }


def _error(values: Iterable[float], intervals: Iterable[list[float] | None]) -> np.ndarray:
    pairs = []
    for value, interval in zip(values, intervals):
        if interval is None:
            pairs.append((0.0, 0.0))
        else:
            pairs.append((max(0.0, value - interval[0]), max(0.0, interval[1] - value)))
    return np.asarray(pairs, dtype=np.float64).T


def write_chart(analysis: dict[str, Any], path: Path) -> None:
    runtime = dict(analysis["runtime"])
    teacher = dict(analysis["teacher"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    figure.suptitle("V193: chuỗi từ dữ liệu dạy đến kết quả MT5", fontsize=16, fontweight="bold")

    by_action = dict(runtime["by_entry_action"])
    actions = list(by_action)
    means = [float(by_action[action]["mean_realized_r"]) for action in actions]
    intervals = [by_action[action]["mean_realized_r_day_block_bootstrap_95_interval"] for action in actions]
    colors = ["#3d9b68" if value >= 0.0 else "#c84f4f" for value in means]
    axes[0, 0].bar(
        [ACTION_LABELS.get(action, action) for action in actions], means,
        yerr=_error(means, intervals), color=colors, capsize=4,
    )
    axes[0, 0].axhline(0.0, color="#273444", linewidth=1)
    axes[0, 0].set_title("Kỳ vọng theo kiểu vào lệnh\n(cận 95% lấy mẫu lại theo ngày)")
    axes[0, 0].set_ylabel("Lợi nhuận / rủi ro ban đầu")
    axes[0, 0].tick_params(axis="x", rotation=28)
    for index, action in enumerate(actions):
        axes[0, 0].text(index, means[index], f"n={by_action[action]['positions']}", ha="center", va="bottom")

    age = dict(runtime.get("by_age_bars", {}))
    age_keys = [key for key in AGE_ORDER if key in age]
    age_pnl = [float(age[key].get("net_pnl", 0.0)) for key in age_keys]
    axes[0, 1].bar(age_keys, age_pnl, color=["#c84f4f" if value < 0.0 else "#3d9b68" for value in age_pnl])
    axes[0, 1].axhline(0.0, color="#273444", linewidth=1)
    axes[0, 1].set_title("Lãi lỗ theo tuổi vị thế")
    axes[0, 1].set_xlabel("Số thanh từ lúc vào đến lúc đóng")
    axes[0, 1].set_ylabel("Lãi ròng")
    for index, key in enumerate(age_keys):
        axes[0, 1].text(index, age_pnl[index], f"n={age[key].get('positions', 0)}", ha="center", va="top" if age_pnl[index] < 0 else "bottom")

    teacher_rr = np.asarray(teacher.pop("tp_rr_values"), dtype=np.float64)
    runtime_rr = np.asarray([
        float(key) for key, summary in dict(runtime.get("by_rr", {})).items()
        for _ in range(int(summary.get("positions", 0)))
    ], dtype=np.float64)
    bins = np.asarray([0.0, 0.375, 0.625, 0.875, 1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 5.5])
    axes[1, 0].hist(teacher_rr, bins=bins, weights=np.ones(len(teacher_rr)) / len(teacher_rr), alpha=0.65, label="Dữ liệu dạy", color="#315f9c")
    axes[1, 0].hist(runtime_rr, bins=bins, weights=np.ones(len(runtime_rr)) / len(runtime_rr), alpha=0.55, label="MT5 thực tế", color="#d28d32")
    axes[1, 0].axvline(float(analysis["teacher"]["tp_rr"]["median"]), color="#315f9c", linestyle="--")
    axes[1, 0].axvline(
        float(dict(runtime.get("lifecycle_overall", {})).get("median_rr", 0.0)),
        color="#d28d32", linestyle="--",
    )
    axes[1, 0].set_title("Mục tiêu lợi nhuận: dữ liệu dạy so với MT5")
    axes[1, 0].set_xlabel("Lợi nhuận mục tiêu / rủi ro ban đầu")
    axes[1, 0].set_ylabel("Tỷ trọng")
    axes[1, 0].legend()

    comparison = list(analysis.get("matched_ablation", []))
    labels = {
        "Frozen argmax": "Cố định",
        "Frozen + exploration": "Cố định + thăm dò",
        "Online PPO train pass": "Tự học PPO",
    }
    names = [labels.get(str(row.get("label")), str(row.get("label"))) for row in comparison]
    profits = [float(row.get("net_profit", 0.0)) for row in comparison]
    axes[1, 1].bar(names, profits, color=["#c84f4f" if value < 0.0 else "#3d9b68" for value in profits])
    axes[1, 1].axhline(0.0, color="#273444", linewidth=1)
    axes[1, 1].set_title("Thử nghiệm MT5 cùng điều kiện")
    axes[1, 1].set_ylabel("Lãi ròng")
    axes[1, 1].tick_params(axis="x", rotation=18)
    for index, row in enumerate(comparison):
        axes[1, 1].text(
            index, profits[index],
            f"{profits[index]:+.0f}\nHS={float(row.get('profit_factor', 0.0)):.2f}\nn={int(row.get('total_trades', 0))}",
            ha="center", va="top" if profits[index] < 0.0 else "bottom",
        )

    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-dataset", type=Path, required=True)
    parser.add_argument("--lifecycle-report", type=Path, required=True)
    parser.add_argument("--ablation-report", type=Path, required=True)
    parser.add_argument("--checkpoint-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--live-diagnosis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="suy_luan_sau_v193")
    args = parser.parse_args()

    analysis = build_analysis(
        args.teacher_dataset.resolve(), args.lifecycle_report.resolve(),
        args.ablation_report.resolve(), args.checkpoint_report.resolve(),
        args.live_diagnosis.resolve(),
        args.checkpoint.resolve(), args.config.resolve(),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{args.name}.json"
    chart_path = output_dir / f"{args.name}.png"
    write_chart(analysis, chart_path)
    analysis["teacher"].pop("tp_rr_values", None)
    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "chart": str(chart_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
