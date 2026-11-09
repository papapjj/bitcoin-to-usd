#!/usr/bin/env python3
"""Render a compact comparison chart for completed MT5 short-gate runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _manifest(run: Path) -> dict:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    labels: list[str] = []
    profit: list[float] = []
    factor: list[float] = []
    lcb: list[float] = []
    samples: list[int] = []
    latest_paired: dict = {}
    for run in args.runs:
        manifest = _manifest(run)
        metrics = dict(manifest["mt5"]["metrics"])
        summary = dict(metrics.get("execution_summary", {}) or {})
        labels.append(run.name.split("_2026", 1)[0].replace("mt5_oos_short_", ""))
        profit.append(float(metrics.get("net_profit", 0.0) or 0.0))
        factor.append(float(metrics.get("profit_factor", 0.0) or 0.0))
        lcb.append(float(summary.get(
            "expectancy_lcb", metrics.get("fitness_policy_expectancy_lcb", 0.0)
        ) or 0.0))
        samples.append(int(summary.get(
            "lifecycle_count", metrics.get("fitness_policy_origin_lifecycle_count", 0)
        ) or 0))
        latest_paired = dict(metrics.get("paired_seed_validation", {}) or {})

    latest = args.runs[-1]
    journals = sorted(latest.glob("*.mt5_failure_replay.jsonl"))
    replay_profit: list[float] = []
    if journals:
        for line in journals[-1].read_text(encoding="utf-8").splitlines():
            if line.strip():
                replay_profit.append(float(json.loads(line).get("net_profit", 0.0) or 0.0))

    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle("MT5 short-gate diagnosis", fontsize=16, fontweight="bold")

    colors = ["#229954" if value > 0 else "#c0392b" for value in profit]
    axes[0, 0].bar(x, profit, color=colors)
    axes[0, 0].axhline(0, color="black", linewidth=0.8)
    axes[0, 0].set_title("Frozen net profit")
    axes[0, 0].set_xticks(x, labels)
    for i, value in enumerate(profit):
        axes[0, 0].text(i, value, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)

    axes[0, 1].plot(x, factor, marker="o", color="#2471a3", label="Profit factor")
    axes[0, 1].axhline(1.0, color="#229954", linestyle="--", linewidth=1, label="PF gate = 1")
    axes[0, 1].set_title("Profit factor")
    axes[0, 1].set_xticks(x, labels)
    axes[0, 1].legend(fontsize=8)

    axes[1, 0].plot(x, lcb, marker="o", color="#8e44ad", label="Policy expectancy LCB")
    axes[1, 0].axhline(0.0, color="#229954", linestyle="--", linewidth=1, label="Gate = 0")
    axes[1, 0].set_title("Current-window execution expectancy LCB")
    axes[1, 0].set_xticks(x, labels)
    for i, value in enumerate(samples):
        axes[1, 0].annotate(f"n={value}", (i, lcb[i]), xytext=(0, -15), textcoords="offset points", ha="center", fontsize=7)
    axes[1, 0].legend(fontsize=8)

    if bool(latest_paired.get("performed", False)):
        paired_labels = ["Candidate", "Protected seed"]
        paired_profit = [
            float(latest_paired.get("candidate_net_profit", 0.0) or 0.0),
            float(latest_paired.get("baseline_net_profit", 0.0) or 0.0),
        ]
        rx = np.arange(2)
        replay_colors = ["#229954" if value > 0 else "#c0392b" for value in paired_profit]
        axes[1, 1].bar(rx, paired_profit, color=replay_colors)
        axes[1, 1].set_xticks(rx, paired_labels)
        axes[1, 1].set_title(
            f"Paired frozen gate: {'PASS' if latest_paired.get('accepted') else 'ROLLBACK'}"
        )
        for i, value in enumerate(paired_profit):
            axes[1, 1].text(i, value, f"{value:+.2f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    else:
        rx = np.arange(len(replay_profit))
        replay_colors = ["#229954" if value > 0 else "#c0392b" for value in replay_profit]
        axes[1, 1].bar(rx, replay_profit, color=replay_colors)
        axes[1, 1].set_title(f"{labels[-1]} failure replay sessions")
        axes[1, 1].set_xlabel("Replay session")
    axes[1, 1].axhline(0, color="black", linewidth=0.8)
    axes[1, 1].set_ylabel("Net profit")

    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170)
    plt.close(fig)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
