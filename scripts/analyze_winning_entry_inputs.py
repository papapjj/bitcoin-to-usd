"""Audit raw MT5 entry inputs for winning versus losing closed lifecycles.

The script joins only immutable, archived ``feature_audit.csv`` rows to the
corresponding final-close rows in ``execution_ledger.csv``.  It deliberately
does not inspect model weights or future bars.  Its output is descriptive:
candidate runs over the same market window are not independent OOS evidence.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

from self_growing_trader.nhat_ky_giai_thich import _feature_name
from self_growing_trader.protocol_contract import INPUT_COUNT


ENTRY_ACTIONS = frozenset(range(1, 7))


@dataclass(frozen=True)
class EntryOutcome:
    run: str
    trace_id: str
    context: str
    action: int
    net_r: float
    values: tuple[float, ...]


def _number(value: object, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _integer(value: object, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _is_final_close(row: dict[str, str]) -> bool:
    return str(row.get("position_final_close", "")).strip().lower() in {"1", "true", "yes"}


def _parse_inputs(raw: object) -> tuple[float, ...] | None:
    try:
        values = tuple(float(item) for item in str(raw or "").split("|"))
    except ValueError:
        return None
    if len(values) != INPUT_COUNT or not all(math.isfinite(value) for value in values):
        return None
    return values


def _read_tsv(path: Path) -> Iterable[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle, delimiter="\t")


def _entry_event_by_trace(
    ledger_path: Path,
) -> tuple[dict[str, tuple[str, int, str]], list[dict[str, str]]]:
    """Return immutable entry provenance plus final-close rows per owner trace.

    A policy-managed close has its own ``trace_id``.  The lifecycle must be
    joined through ``position_owner_trace_id`` to recover the entry-time
    feature vector; using the close trace would incorrectly yield zero joins.
    """

    entries: dict[str, tuple[int, str, int, str]] = {}
    final_rows: list[dict[str, str]] = []
    priority = {"ORDER_ADD": 0, "DEAL_ADD": 1, "REQUEST": 2}
    for row in _read_tsv(ledger_path):
        trace_id = str(row.get("trace_id", "") or "").strip()
        # Broker/system SL/TP rows may deliberately have no *exit* trace.
        # Keep them here: their position_owner_trace_id still names the entry
        # whose immutable input vector must be analysed.
        if _is_final_close(row):
            final_rows.append(row)
        if not trace_id:
            continue
        action = _integer(row.get("requested_action"))
        event_id = str(row.get("event_id", "") or "").strip()
        transaction = str(row.get("transaction_type", "") or "").strip().upper()
        if action not in ENTRY_ACTIONS or not event_id or transaction not in priority:
            continue
        candidate = (
            priority[transaction],
            event_id,
            action,
            str(row.get("intent_context", "") or "").strip() or "UNKNOWN",
        )
        if trace_id not in entries or candidate < entries[trace_id]:
            entries[trace_id] = candidate
    return {
        trace_id: (event_id, action, context)
        for trace_id, (_priority, event_id, action, context) in entries.items()
    }, final_rows


def _feature_rows_by_event(feature_path: Path) -> dict[str, tuple[float, ...]]:
    rows: dict[str, tuple[float, ...]] = {}
    for row in _read_tsv(feature_path):
        event_id = str(row.get("event_id", "") or "").strip()
        values = _parse_inputs(row.get("inputs"))
        if event_id and values is not None:
            rows.setdefault(event_id, values)
    return rows


def load_outcomes(candidates_dir: Path) -> tuple[list[EntryOutcome], dict[str, object]]:
    outcomes: list[EntryOutcome] = []
    source_rows: list[dict[str, object]] = []
    for runtime in sorted(candidates_dir.glob("*.runtime")):
        ledger_path = runtime / "execution_ledger.csv"
        feature_path = runtime / "feature_audit.csv"
        if not ledger_path.is_file() or not feature_path.is_file():
            continue
        entry_events, final_rows = _entry_event_by_trace(ledger_path)
        features = _feature_rows_by_event(feature_path)
        joined = 0
        no_entry = 0
        no_feature = 0
        for final in final_rows:
            trace_id = str(final.get("trace_id", "") or "").strip()
            owner_trace_id = str(final.get("position_owner_trace_id", "") or "").strip() or trace_id
            entry = entry_events.get(owner_trace_id)
            if entry is None:
                no_entry += 1
                continue
            event_id, action, context = entry
            values = features.get(event_id)
            if values is None:
                no_feature += 1
                continue
            risk = _number(final.get("initial_risk"))
            if risk <= 1e-12:
                continue
            net = (
                _number(final.get("profit"))
                + _number(final.get("commission"))
                + _number(final.get("swap"))
            )
            outcomes.append(EntryOutcome(
                run=runtime.name.removesuffix(".runtime"),
                trace_id=owner_trace_id,
                context=context,
                action=action,
                net_r=float(net / risk),
                values=values,
            ))
            joined += 1
        source_rows.append({
            "run": runtime.name.removesuffix(".runtime"),
            "final_close_rows": len(final_rows),
            "joined_rows": joined,
            "missing_entry_event": no_entry,
            "missing_feature_row": no_feature,
        })
    return outcomes, {"sources": source_rows}


def feature_statistics(outcomes: list[EntryOutcome]) -> tuple[list[dict[str, object]], np.ndarray, np.ndarray]:
    winners = np.asarray([row.values for row in outcomes if row.net_r > 0.0], dtype=float)
    losers = np.asarray([row.values for row in outcomes if row.net_r < 0.0], dtype=float)
    if winners.size == 0 or losers.size == 0:
        raise ValueError("Need at least one winning and one losing lifecycle after the immutable join")
    n_win, n_loss = winners.shape[0], losers.shape[0]
    win_mean = winners.mean(axis=0)
    loss_mean = losers.mean(axis=0)
    win_std = winners.std(axis=0, ddof=1) if n_win > 1 else np.zeros(INPUT_COUNT)
    loss_std = losers.std(axis=0, ddof=1) if n_loss > 1 else np.zeros(INPUT_COUNT)
    pooled = np.sqrt(np.maximum(
        ((n_win - 1) * win_std**2 + (n_loss - 1) * loss_std**2) / max(1, n_win + n_loss - 2),
        1.0e-12,
    ))
    effect = (win_mean - loss_mean) / pooled
    overall_mean = np.vstack((winners, losers)).mean(axis=0)
    overall_std = np.vstack((winners, losers)).std(axis=0, ddof=1)
    overall_std = np.maximum(overall_std, 1.0e-12)
    # Pooling actions/contexts can make an input look useful merely because a
    # profitable context chose that action more often.  Compute a second,
    # descriptive contrast only within identical run × context × entry-action
    # strata.  It still is not a causal/OOS estimate, but it detects this
    # obvious composition confound.
    strata: dict[tuple[str, str, int], list[EntryOutcome]] = defaultdict(list)
    for outcome in outcomes:
        if outcome.net_r != 0.0:
            strata[(outcome.run, outcome.context, outcome.action)].append(outcome)
    within_sum = np.zeros(INPUT_COUNT, dtype=float)
    within_weight = 0.0
    within_strata = 0
    for members in strata.values():
        group_winners = np.asarray([item.values for item in members if item.net_r > 0.0], dtype=float)
        group_losers = np.asarray([item.values for item in members if item.net_r < 0.0], dtype=float)
        if group_winners.size == 0 or group_losers.size == 0:
            continue
        n_group_win, n_group_loss = group_winners.shape[0], group_losers.shape[0]
        weight = float(n_group_win * n_group_loss / (n_group_win + n_group_loss))
        within_sum += weight * (group_winners.mean(axis=0) - group_losers.mean(axis=0))
        within_weight += weight
        within_strata += 1
    within_delta = within_sum / max(within_weight, 1.0e-12)
    within_effect = within_delta / pooled
    rows = []
    for index in range(INPUT_COUNT):
        rows.append({
            "index": index,
            "feature": _feature_name(index),
            "winner_mean": float(win_mean[index]),
            "loser_mean": float(loss_mean[index]),
            "winner_std": float(win_std[index]),
            "loser_std": float(loss_std[index]),
            "delta_win_minus_loss": float(win_mean[index] - loss_mean[index]),
            "cohen_d": float(effect[index]),
            "within_run_context_action_delta": float(within_delta[index]),
            "within_run_context_action_effect": float(within_effect[index]),
            "within_run_context_action_strata": int(within_strata),
            "winner_mean_z": float((win_mean[index] - overall_mean[index]) / overall_std[index]),
            "loser_mean_z": float((loss_mean[index] - overall_mean[index]) / overall_std[index]),
        })
    rows.sort(key=lambda row: abs(float(row["cohen_d"])), reverse=True)
    return rows, winners, losers


def winning_feature_ranges(outcomes: list[EntryOutcome]) -> list[dict[str, object]]:
    """Describe the actual input envelope of final net-R winning lifecycles.

    This is intentionally separate from the Win-minus-Loss contrast.  It lets
    an audit answer the different question: what ranges did this owner expert
    *actually see* when its entries ultimately won?  It must not be converted
    into a hard rule, gate, or feature mask without a separately controlled
    causal/OOS experiment.
    """

    winners = np.asarray([row.values for row in outcomes if row.net_r > 0.0], dtype=float)
    if winners.size == 0:
        raise ValueError("Need at least one winning lifecycle to calculate winning input ranges")
    count = int(winners.shape[0])
    minimum = winners.min(axis=0)
    p05, p25, median, p75, p95 = np.percentile(winners, [5, 25, 50, 75, 95], axis=0)
    maximum = winners.max(axis=0)
    mean = winners.mean(axis=0)
    std = winners.std(axis=0, ddof=1) if count > 1 else np.zeros(INPUT_COUNT)
    rows: list[dict[str, object]] = []
    for index in range(INPUT_COUNT):
        rows.append({
            "index": index,
            "feature": _feature_name(index),
            "winning_lifecycle_count": count,
            "winner_mean": float(mean[index]),
            "winner_std": float(std[index]),
            "winner_min": float(minimum[index]),
            "winner_p05": float(p05[index]),
            "winner_p25": float(p25[index]),
            "winner_median": float(median[index]),
            "winner_p75": float(p75[index]),
            "winner_p95": float(p95[index]),
            "winner_max": float(maximum[index]),
            "winner_iqr": float(p75[index] - p25[index]),
            "winner_range": float(maximum[index] - minimum[index]),
        })
    return rows


def _write_chart(stats: list[dict[str, object]], outcomes: list[EntryOutcome], output: Path) -> None:
    top = list(reversed(stats[:20]))
    labels = [f"{row['index']:03d} {row['feature']}" for row in top]
    effects = np.asarray([float(row["cohen_d"]) for row in top])
    colors = np.where(effects >= 0.0, "#159f8f", "#d05a5a")
    contexts = Counter((row.context, "win" if row.net_r > 0.0 else "loss") for row in outcomes if row.net_r != 0.0)
    context_names = sorted({context for context, _kind in contexts})
    wins = [contexts[(context, "win")] for context in context_names]
    losses = [contexts[(context, "loss")] for context in context_names]

    fig = plt.figure(figsize=(17, 14), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.35, 1.0))
    axis_effect = fig.add_subplot(grid[0, 0])
    axis_means = fig.add_subplot(grid[0, 1])
    axis_context = fig.add_subplot(grid[1, :])
    axis_effect.barh(labels, effects, color=colors)
    axis_effect.axvline(0.0, color="#334155", linewidth=1)
    axis_effect.set_title("Top 20 raw-entry inputs by standardized Win − Loss difference")
    axis_effect.set_xlabel("Cohen d (positive = higher among winning lifecycles)")
    axis_effect.tick_params(axis="y", labelsize=8)

    win_z = [float(row["winner_mean_z"]) for row in top]
    loss_z = [float(row["loser_mean_z"]) for row in top]
    y = np.arange(len(top))
    within_effect = [float(row["within_run_context_action_effect"]) for row in top]
    axis_means.scatter(within_effect, y, color="#4f46e5", label="Within run/context/action", zorder=3)
    axis_means.scatter(win_z, y, color="#159f8f", label="Winning mean z", zorder=3)
    axis_means.scatter(loss_z, y, color="#d05a5a", label="Losing mean z", zorder=3)
    axis_means.axvline(0.0, color="#334155", linewidth=1)
    axis_means.set_yticks(y, labels, fontsize=8)
    axis_means.set_title("Raw means plus within run/context/action contrast")
    axis_means.set_xlabel("z-score / within-stratum standardized contrast")
    axis_means.legend(loc="best")

    x = np.arange(len(context_names))
    axis_context.bar(x, wins, label="Winning final R > 0", color="#159f8f")
    axis_context.bar(x, losses, bottom=wins, label="Losing final R < 0", color="#d05a5a")
    axis_context.set_xticks(x, context_names, rotation=25, ha="right")
    axis_context.set_ylabel("Closed lifecycles joined to immutable entry input")
    axis_context.set_title("Joined outcome mix by entry context")
    axis_context.legend(loc="upper right")

    fig.suptitle(
        f"Archived MT5 entry-input audit — {len(outcomes)} joined lifecycles; "
        f"{sum(row.net_r > 0 for row in outcomes)} wins / {sum(row.net_r < 0 for row in outcomes)} losses\n"
        "Descriptive pooled in-sample evidence only; repeated candidate windows are not independent OOS samples.",
        fontsize=13,
    )
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_expert_chart(
    expert_stats: dict[str, list[dict[str, object]]],
    expert_outcomes: dict[str, list[EntryOutcome]],
    output: Path,
) -> None:
    """Small multiples make sparse expert-specific signals inspectable."""

    contexts = sorted(expert_stats)
    columns = 2
    rows = max(1, int(math.ceil(len(contexts) / columns)))
    figure, axes = plt.subplots(rows, columns, figsize=(18, 4.8 * rows), squeeze=False)
    all_effects = [
        abs(float(item["cohen_d"]))
        for values in expert_stats.values()
        for item in values[:8]
    ]
    limit = max(0.5, max(all_effects, default=0.0) * 1.12)
    for axis, context in zip(axes.ravel(), contexts):
        stats = expert_stats[context]
        top = list(reversed(stats[:8]))
        labels = [f"{row['index']:03d} {row['feature']}" for row in top]
        effects = np.asarray([float(row["cohen_d"]) for row in top])
        adjusted = np.asarray([float(row["within_run_context_action_effect"]) for row in top])
        y = np.arange(len(top))
        colors = np.where(effects >= 0.0, "#159f8f", "#d05a5a")
        axis.barh(y, effects, color=colors, alpha=0.88, label="Pooled d")
        axis.scatter(adjusted, y, color="#4f46e5", s=32, zorder=3, label="Within run/action")
        axis.axvline(0.0, color="#334155", linewidth=1)
        axis.set_xlim(-limit, limit)
        axis.set_yticks(y, labels, fontsize=8)
        items = expert_outcomes[context]
        win_count = sum(item.net_r > 0.0 for item in items)
        loss_count = sum(item.net_r < 0.0 for item in items)
        axis.set_title(f"{context}: {len(items)} lifecycle; {win_count} win / {loss_count} loss")
        axis.set_xlabel("Standardized Win − Loss contrast")
    for axis in axes.ravel()[len(contexts):]:
        axis.remove()
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.955))
    figure.suptitle(
        "Archived MT5 entry inputs by owner expert/context\n"
        "Bars are pooled Cohen d; dots hold run/context fixed and compare only entry action. Descriptive only.",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _write_expert_all_input_heatmap(
    expert_stats: dict[str, list[dict[str, object]]],
    expert_outcomes: dict[str, list[EntryOutcome]],
    output: Path,
) -> None:
    """Render all input rows without silently hiding weak or non-top features.

    The raw Cohen-d chart remains useful for a compact first read.  This
    heatmap is the exhaustive companion: every schema position has one row and
    every owner expert has one column.  Its values use the within-run/context/
    action contrast because the context split already removes the owner
    mixture, while action composition can still otherwise dominate a column.
    """

    contexts = sorted(expert_stats)
    matrix = np.zeros((INPUT_COUNT, len(contexts)), dtype=float)
    for column, context in enumerate(contexts):
        by_index = {int(row["index"]): row for row in expert_stats[context]}
        for index in range(INPUT_COUNT):
            matrix[index, column] = float(by_index[index]["within_run_context_action_effect"])
    limit = max(0.5, float(np.max(np.abs(matrix))) * 1.04)
    figure, axis = plt.subplots(figsize=(17, max(30, 0.255 * INPUT_COUNT)))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_yticks(
        np.arange(INPUT_COUNT),
        [f"{index:03d} {_feature_name(index)}" for index in range(INPUT_COUNT)],
        fontsize=5.5,
    )
    labels = []
    for context in contexts:
        items = expert_outcomes[context]
        labels.append(
            f"{context}\n{sum(item.net_r > 0.0 for item in items)}W / "
            f"{sum(item.net_r < 0.0 for item in items)}L"
        )
    axis.set_xticks(np.arange(len(contexts)), labels, fontsize=9)
    axis.set_xlabel("Owner expert/context — count is immutable joined lifecycle Win / Loss")
    axis.set_ylabel("All policy input-schema positions (0–136; none omitted)")
    axis.set_title(
        "All 137 entry inputs by owner expert\n"
        "Within run/context/action standardized Win − Loss contrast; descriptive archive only"
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.012)
    colorbar.set_label("Within-stratum standardized contrast (positive = higher among winners)")
    figure.tight_layout()
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _safe_chart_filename(context: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", context.lower()).strip("_") or "unknown"


def _write_expert_all_input_charts(
    expert_stats: dict[str, list[dict[str, object]]],
    expert_outcomes: dict[str, list[EntryOutcome]],
    output_dir: Path,
) -> dict[str, str]:
    """Write one readable, exhaustive 137-input plot for each owner expert."""

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for context in sorted(expert_stats):
        by_index = {int(row["index"]): row for row in expert_stats[context]}
        rows = [by_index[index] for index in range(INPUT_COUNT)]
        chunks = [rows[start:start + 46] for start in range(0, INPUT_COUNT, 46)]
        limit = max(
            0.5,
            max(
                abs(float(row["cohen_d"]))
                for row in rows
            ) * 1.12,
            max(
                abs(float(row["within_run_context_action_effect"]))
                for row in rows
            ) * 1.12,
        )
        figure, axes = plt.subplots(1, len(chunks), figsize=(28, 19), sharex=True)
        if len(chunks) == 1:
            axes = [axes]
        for axis, chunk in zip(axes, chunks):
            labels = [f"{int(row['index']):03d} {row['feature']}" for row in chunk]
            effects = np.asarray([float(row["cohen_d"]) for row in chunk])
            adjusted = np.asarray([float(row["within_run_context_action_effect"]) for row in chunk])
            y = np.arange(len(chunk))
            axis.barh(
                y,
                effects,
                color=np.where(effects >= 0.0, "#159f8f", "#d05a5a"),
                alpha=0.88,
                label="Pooled Cohen d",
            )
            axis.scatter(adjusted, y, color="#4f46e5", s=18, zorder=3, label="Within run/action")
            axis.axvline(0.0, color="#334155", linewidth=1)
            axis.set_xlim(-limit, limit)
            axis.set_yticks(y, labels, fontsize=6.6)
            axis.invert_yaxis()
            axis.set_xlabel("Standardized Win − Loss contrast")
        handles, labels = axes[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.965))
        items = expert_outcomes[context]
        figure.suptitle(
            f"{context} — all {INPUT_COUNT} entry inputs; {len(items)} lifecycle; "
            f"{sum(item.net_r > 0.0 for item in items)} win / "
            f"{sum(item.net_r < 0.0 for item in items)} loss\n"
            "Bars are pooled Cohen d; dots hold run/context fixed and compare only entry action. Descriptive only.",
            fontsize=14,
            y=0.995,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.93))
        filename = f"{_safe_chart_filename(context)}_all_{INPUT_COUNT}_inputs.png"
        figure.savefig(output_dir / filename, dpi=180, bbox_inches="tight")
        plt.close(figure)
        paths[context] = filename
    return paths


def _table(rows: Iterable[dict[str, object]]) -> str:
    columns = (
        "index", "feature", "winner_mean", "loser_mean", "delta_win_minus_loss", "cohen_d",
        "within_run_context_action_effect",
    )
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = []
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            values.append(html.escape(f"{value:.5f}" if isinstance(value, float) else str(value)))
        body.append("<tr>" + "".join(f"<td>{value}</td>" for value in values) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def _write_html(
    stats: list[dict[str, object]],
    outcomes: list[EntryOutcome],
    provenance: dict[str, object],
    chart_name: str,
    output: Path,
) -> None:
    wins = [row for row in outcomes if row.net_r > 0.0]
    losses = [row for row in outcomes if row.net_r < 0.0]
    source_rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key in (
            "run", "final_close_rows", "joined_rows", "missing_entry_event", "missing_feature_row"
        )) + "</tr>"
        for row in provenance["sources"]
    )
    output.write_text(f"""<!doctype html>
<html lang=\"vi\"><meta charset=\"utf-8\"><title>Input thắng/thua — MT5 archive</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1500px;margin:28px auto;color:#172033;padding:0 18px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #d8dee9;padding:7px;text-align:left}}th{{background:#f2f5f8}}.note{{background:#fff5d8;border-left:4px solid #e5a62b;padding:12px}}</style>
<h1>Input tại entry của lifecycle thắng và thua</h1>
<p class=\"note\">Scope: chỉ các artifact MT5 có đủ <code>feature_audit.csv</code> và <code>execution_ledger.csv</code>. Win là final net-R &gt; 0; loss là final net-R &lt; 0. <code>cohen_d</code> là chênh lệch pooled; <code>within_run_context_action_effect</code> chỉ so sánh win/loss trong cùng run, context và entry-action để phát hiện confound thành phần. Cả hai vẫn là mô tả in-sample, không phải nhân quả, feature selection hay bằng chứng OOS.</p>
<ul><li>Lifecycle join hợp lệ: <b>{len(outcomes)}</b></li><li>Win: <b>{len(wins)}</b>; Loss: <b>{len(losses)}</b></li><li>Biểu đồ: <a href=\"{html.escape(chart_name)}\">PNG</a></li></ul>
<img src=\"{html.escape(chart_name)}\" style=\"max-width:100%;height:auto\" alt=\"Input thắng thua\">
<h2>Top 20 khác biệt chuẩn hóa tuyệt đối</h2>{_table(stats[:20])}
<h2>Input có trung bình cao hơn ở lệnh thắng</h2>{_table([row for row in stats if float(row['cohen_d']) > 0][:20])}
<h2>Input có trung bình cao hơn ở lệnh thua</h2>{_table([row for row in stats if float(row['cohen_d']) < 0][:20])}
<h2>Đủ toàn bộ {INPUT_COUNT} input theo schema</h2>
<details open><summary>Hiện bảng đầy đủ — không lọc theo effect</summary>{_table(sorted(stats, key=lambda row: int(row['index'])))}</details>
<h2>Coverage theo artifact</h2><table><thead><tr><th>run</th><th>final close</th><th>join</th><th>thiếu entry event</th><th>thiếu feature</th></tr></thead><tbody>{source_rows}</tbody></table>
</html>""", encoding="utf-8")


def _write_expert_html(
    expert_stats: dict[str, list[dict[str, object]]],
    expert_outcomes: dict[str, list[EntryOutcome]],
    chart_name: str,
    all_input_heatmap_name: str,
    all_input_chart_dir_name: str,
    all_input_chart_names: dict[str, str],
    output: Path,
) -> None:
    sections: list[str] = []
    for context in sorted(expert_stats):
        outcomes = expert_outcomes[context]
        wins = sum(item.net_r > 0.0 for item in outcomes)
        losses = sum(item.net_r < 0.0 for item in outcomes)
        all_inputs = sorted(expert_stats[context], key=lambda row: int(row["index"]))
        chart_path = f"{all_input_chart_dir_name}/{all_input_chart_names[context]}"
        sections.append(
            f"<h2>{html.escape(context)} — {len(outcomes)} lifecycle; {wins} win / {losses} loss</h2>"
            + "<p><a href=\"" + html.escape(chart_path) + "\">Biểu đồ đầy đủ 137 input của specialist này (PNG)</a></p>"
            + "<h3>Top 12 để đọc nhanh</h3>"
            + _table(expert_stats[context][:12])
            + f"<h3>Đủ toàn bộ {INPUT_COUNT} input</h3>"
            + "<details><summary>Mở bảng đầy đủ — không lọc theo effect</summary>"
            + _table(all_inputs)
            + "</details>"
        )
    output.write_text(f"""<!doctype html>
<html lang=\"vi\"><meta charset=\"utf-8\"><title>Input theo chuyên gia — MT5 archive</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:1500px;margin:28px auto;color:#172033;padding:0 18px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{border-bottom:1px solid #d8dee9;padding:7px;text-align:left}}th{{background:#f2f5f8}}.note{{background:#fff5d8;border-left:4px solid #e5a62b;padding:12px}}</style>
<h1>Input entry thắng/thua theo chuyên gia owner</h1>
<p class=\"note\">Mỗi specialist/context chỉ dùng lifecycle được join qua immutable <code>position_owner_trace_id</code>. Bar là pooled Cohen d; cột <code>within_run_context_action_effect</code> giữ nguyên run/context và chỉ so sánh win/loss trong cùng action. Không dùng kết quả này làm mask, gate, trọng số cố định hay tuyên bố edge.</p>
<img src=\"{html.escape(chart_name)}\" style=\"max-width:100%;height:auto\" alt=\"Input theo chuyên gia\">
<h2>Heatmap đầy đủ 6 chuyên gia × {INPUT_COUNT} input</h2>
<p><a href=\"{html.escape(all_input_heatmap_name)}\">Mở PNG heatmap đầy đủ</a></p>
<img src=\"{html.escape(all_input_heatmap_name)}\" style=\"max-width:100%;height:auto\" alt=\"Tất cả input theo chuyên gia\">
{''.join(sections)}
</html>""", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates-dir", type=Path, default=Path("outputs/he_chuyen_gia/candidates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/he_chuyen_gia/analysis"))
    args = parser.parse_args()
    outcomes, provenance = load_outcomes(args.candidates_dir)
    stats, winners, losers = feature_statistics(outcomes)
    pooled_winning_ranges = winning_feature_ranges(outcomes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / "input_winner_analysis_archived_mt5"
    chart_path = prefix.with_suffix(".png")
    json_path = prefix.with_suffix(".json")
    html_path = prefix.with_suffix(".html")
    _write_chart(stats, outcomes, chart_path)
    payload = {
        "scope": "archived MT5 runtime artifacts with immutable feature audit plus final execution ledger",
        "input_schema_count": INPUT_COUNT,
        "joined_lifecycles": len(outcomes),
        "wins": int(winners.shape[0]),
        "losses": int(losers.shape[0]),
        "provenance": provenance,
        "feature_statistics": stats,
        "winning_feature_ranges": pooled_winning_ranges,
        "outcomes": [asdict(row) | {"values": None} for row in outcomes],
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_html(stats, outcomes, provenance, chart_path.name, html_path)
    expert_outcomes: dict[str, list[EntryOutcome]] = defaultdict(list)
    for outcome in outcomes:
        expert_outcomes[outcome.context].append(outcome)
    expert_stats: dict[str, list[dict[str, object]]] = {}
    expert_winning_ranges: dict[str, list[dict[str, object]]] = {}
    skipped_experts: dict[str, str] = {}
    for context, rows in sorted(expert_outcomes.items()):
        if any(item.net_r > 0.0 for item in rows):
            expert_winning_ranges[context] = winning_feature_ranges(rows)
        if not any(item.net_r > 0.0 for item in rows) or not any(item.net_r < 0.0 for item in rows):
            skipped_experts[context] = "missing_win_or_loss"
            continue
        expert_stats[context], _wins, _losses = feature_statistics(rows)
    expert_prefix = args.output_dir / "input_winner_analysis_by_expert_archived_mt5"
    expert_chart_path = expert_prefix.with_suffix(".png")
    expert_json_path = expert_prefix.with_suffix(".json")
    expert_html_path = expert_prefix.with_suffix(".html")
    expert_all_input_heatmap_path = args.output_dir / "input_winner_analysis_by_expert_all_inputs_heatmap.png"
    expert_all_input_chart_dir = args.output_dir / "input_winner_analysis_by_expert_all_inputs"
    _write_expert_chart(expert_stats, expert_outcomes, expert_chart_path)
    _write_expert_all_input_heatmap(
        expert_stats,
        expert_outcomes,
        expert_all_input_heatmap_path,
    )
    expert_all_input_chart_names = _write_expert_all_input_charts(
        expert_stats,
        expert_outcomes,
        expert_all_input_chart_dir,
    )
    _write_expert_html(
        expert_stats,
        expert_outcomes,
        expert_chart_path.name,
        expert_all_input_heatmap_path.name,
        expert_all_input_chart_dir.name,
        expert_all_input_chart_names,
        expert_html_path,
    )
    expert_payload = {
        "scope": "same immutable archive join, split by entry owner context",
        "all_input_heatmap": expert_all_input_heatmap_path.name,
        "all_input_charts": {
            context: f"{expert_all_input_chart_dir.name}/{filename}"
            for context, filename in sorted(expert_all_input_chart_names.items())
        },
        "included_experts": {
            context: {
                "lifecycle_count": len(rows),
                "wins": sum(item.net_r > 0.0 for item in rows),
                "losses": sum(item.net_r < 0.0 for item in rows),
                "feature_statistics": expert_stats.get(context, []),
                "winning_feature_ranges": expert_winning_ranges.get(context, []),
            }
            for context, rows in sorted(expert_outcomes.items())
        },
        "skipped_experts": skipped_experts,
    }
    expert_json_path.write_text(json.dumps(expert_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "chart": str(chart_path), "html": str(html_path), "json": str(json_path),
        "joined_lifecycles": len(outcomes), "wins": int(winners.shape[0]), "losses": int(losers.shape[0]),
        "top_features": stats[:10], "provenance": provenance,
        "expert_chart": str(expert_chart_path),
        "expert_all_input_heatmap": str(expert_all_input_heatmap_path),
        "expert_all_input_chart_dir": str(expert_all_input_chart_dir),
        "expert_html": str(expert_html_path),
        "expert_json": str(expert_json_path),
        "experts": {
            context: {
                "lifecycle_count": len(rows),
                "wins": sum(item.net_r > 0.0 for item in rows),
                "losses": sum(item.net_r < 0.0 for item in rows),
            }
            for context, rows in sorted(expert_outcomes.items())
        },
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
