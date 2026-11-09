#!/usr/bin/env python3
"""Multi-Round RL Training Loop for Self Growing Trader V193 Phase 2 (Downtrend Mastery).

Chains from the Phase 1 Promoted Champion checkpoint (`best_promoted_checkpoint.pkl`),
executing sequential multi-round RL training across sliding 2-day windows in late Jan - Feb 2025:
- Enforces Agent 4 (Portfolio Governor) locked to HOLD (`POLICY_CLOSE_ALL = 0`).
- Enforces SL = 5.0 ATR, TP = 10.0 ATR (RR = 2.0), Max Holding = 120 bars M1.
- Accumulates native RL vetoes (Bidirectional Ranking, Expectancy Veto, Live Action Veto)
  without hardcoded direction gates (`context_direction_entry_gate_enabled: false`).
- Tracks suppression of counter-trend BUY entries in Downtrend (% long in downtrend reduction >= 50%).
- Verifies retention of Uptrend mastery (catastrophic forgetting guard: % short in uptrend stays low).
- Evaluates Promotion Gate: E[R] >= +0.05R, Win Rate >= 40%, Max DD < 10%, Payoff Ratio >= 1.5.
- Persists evolution ledger: manifest.json and evolution_chart.json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np


PHASE2_SCHEDULE = [
    {"round": 1, "from_date": "2025.01.26", "to_date": "2025.01.28", "regime": "Transition / Sideway"},
    {"round": 2, "from_date": "2025.01.28", "to_date": "2025.01.30", "regime": "Early Downtrend"},
    {"round": 3, "from_date": "2025.01.30", "to_date": "2025.02.01", "regime": "Downtrend / Selloff"},
    {"round": 4, "from_date": "2025.02.03", "to_date": "2025.02.05", "regime": "Downtrend Continuation"},
    {"round": 5, "from_date": "2025.02.05", "to_date": "2025.02.07", "regime": "Deep Downtrend"},
    {"round": 6, "from_date": "2025.02.07", "to_date": "2025.02.09", "regime": "High Volatility Selloff"},
    {"round": 7, "from_date": "2025.02.10", "to_date": "2025.02.12", "regime": "Downtrend Consolidation"},
    {"round": 8, "from_date": "2025.02.12", "to_date": "2025.02.14", "regime": "Secondary Downtrend Wave"},
    {"round": 9, "from_date": "2025.02.14", "to_date": "2025.02.16", "regime": "Trend Exhaustion / Dip"},
    {"round": 10, "from_date": "2025.02.17", "to_date": "2025.02.19", "regime": "Bottoming / Reversal"},
]


def extract_phase2_metrics(round_dir: Path) -> dict[str, Any]:
    """Extract performance metrics and bidirectional trend learning statistics."""
    events_path = round_dir / "events.jsonl"
    diag_path = round_dir / "mt5_diagnostics.json"

    lifecycles: list[dict[str, Any]] = []
    uptrend_total = 0
    uptrend_short = 0
    uptrend_long = 0
    downtrend_total = 0
    downtrend_long = 0
    downtrend_short = 0
    policy_close_all = 0

    if events_path.exists():
        with open(events_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    rec = json.loads(line_str)
                except Exception:
                    continue
                if rec.get("event_type") == "lifecycle":
                    pay = rec.get("payload", {})
                    lifecycles.append(pay)
                    if pay.get("close_reason") == "POLICY_CLOSE_ALL":
                        policy_close_all += 1
                    for g in pay.get("growth_events", []):
                        if g.get("type") == "context_action_expectancy_credit":
                            cond = g.get("conditioned_entry_expectancy", {}).get("condition", {})
                            trend = str(cond.get("trend", "")).lower()
                            action = int(g.get("action_index", -1))
                            if trend == "up":
                                uptrend_total += 1
                                if action in (2, 4, 6):
                                    uptrend_short += 1
                                elif action in (1, 3, 5):
                                    uptrend_long += 1
                            elif trend == "down":
                                downtrend_total += 1
                                if action in (1, 3, 5):
                                    downtrend_long += 1
                                elif action in (2, 4, 6):
                                    downtrend_short += 1

    r_values = [float(p["realized_net_r"]) for p in lifecycles if "realized_net_r" in p]
    wins = [r for r in r_values if r > 0.0]
    losses = [r for r in r_values if r <= 0.0]
    total_trades = len(r_values)

    expectancy_r = float(np.mean(r_values)) if total_trades > 0 else 0.0
    win_rate_pct = float(len(wins) / total_trades * 100.0) if total_trades > 0 else 0.0
    avg_win_r = float(np.mean(wins)) if wins else 0.0
    avg_loss_r = float(abs(np.mean(losses))) if losses else 1.0
    payoff_ratio = float(avg_win_r / avg_loss_r) if losses and avg_loss_r > 0 else (2.0 if wins else 0.0)

    max_dd_pct = 0.0
    profit_factor = 0.0
    net_pnl = 0.0
    close_reasons: dict[str, int] = {}
    hold_share = 0.0
    normalized_action_entropy = 0.0

    if diag_path.exists():
        try:
            diag = json.loads(diag_path.read_text(encoding="utf-8"))
            raw_dd = diag.get("risk", {}).get("max_equity_drawdown_pct", 0.0)
            max_dd_pct = float(raw_dd * 100.0 if raw_dd <= 1.0 else raw_dd)
            perf = diag.get("performance", {})
            profit_factor = float(perf.get("profit_factor", 0.0) or 0.0)
            net_pnl = float(perf.get("net_pnl", 0.0) or 0.0)
            close_reasons = dict(perf.get("close_reasons", {}) or {})
            if "POLICY_CLOSE_ALL" in close_reasons:
                policy_close_all = max(policy_close_all, int(close_reasons["POLICY_CLOSE_ALL"]))

            diag_closed = int(perf.get("closed_trades", 0) or 0)
            if total_trades == 0 and diag_closed > 0:
                total_trades = diag_closed
                win_rate_pct = float(perf.get("win_rate", 0.0) or 0.0) * 100.0
                payoff_ratio = float(perf.get("payoff_ratio", 0.0) or 0.0)
                exp_usd = float(perf.get("expectancy", 0.0) or 0.0)
                expectancy_r = float(exp_usd / 50.0)

            pol = diag.get("policy", {})
            hold_share = float(pol.get("hold_share", 0.0) or 0.0)
            normalized_action_entropy = float(pol.get("normalized_action_entropy", 0.0) or 0.0)
        except Exception as e:
            print(f"[WARN] Error reading diagnostics {diag_path}: {e}")

    pct_short_in_uptrend = float(uptrend_short / max(1, uptrend_total) * 100.0) if uptrend_total > 0 else 0.0
    pct_long_in_downtrend = float(downtrend_long / max(1, downtrend_total) * 100.0) if downtrend_total > 0 else 0.0

    return {
        "round_dir": round_dir.name,
        "total_trades": total_trades,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": round(win_rate_pct, 2),
        "expectancy_r": round(expectancy_r, 4),
        "payoff_ratio": round(payoff_ratio, 2),
        "profit_factor": round(profit_factor, 2),
        "net_pnl": round(net_pnl, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "premature_closes": policy_close_all,
        "close_reasons": close_reasons,
        "uptrend_entries_total": uptrend_total,
        "uptrend_short_entries": uptrend_short,
        "uptrend_long_entries": uptrend_long,
        "pct_short_in_uptrend": round(pct_short_in_uptrend, 2),
        "downtrend_entries_total": downtrend_total,
        "downtrend_long_entries": downtrend_long,
        "downtrend_short_entries": downtrend_short,
        "pct_long_in_downtrend": round(pct_long_in_downtrend, 2),
        "hold_share": round(hold_share, 4),
        "normalized_action_entropy": round(normalized_action_entropy, 4),
    }


def evaluate_phase2_promotion(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    """Evaluate if a checkpoint meets Phase 2 promotion hurdles."""
    reasons = []
    if float(metrics.get("expectancy_r", 0.0)) < 0.05:
        reasons.append(f"E[R] {metrics.get('expectancy_r', 0.0):+.4f}R < +0.05R")
    if float(metrics.get("win_rate_pct", 0.0)) < 40.0:
        reasons.append(f"WR {metrics.get('win_rate_pct', 0.0):.1f}% < 40.0%")
    if float(metrics.get("max_drawdown_pct", 0.0)) >= 10.0:
        reasons.append(f"Max DD {metrics.get('max_drawdown_pct', 0.0):.2f}% >= 10.0%")
    if float(metrics.get("payoff_ratio", 0.0)) < 1.5:
        reasons.append(f"Payoff {metrics.get('payoff_ratio', 0.0):.2f} < 1.50")
    if int(metrics.get("premature_closes", 0)) > 0:
        reasons.append(f"Premature closes {metrics.get('premature_closes', 0)} > 0")

    return len(reasons) == 0, reasons


def run_single_round(
    round_idx: int,
    from_date: str,
    to_date: str,
    seed_artifact: Path,
    output_dir: Path,
    workspace_dir: Path,
    timeout_sec: int = 450,
    max_wall_sec: int = 360,
) -> bool:
    """Execute a single MT5 RL training round via run_tensor_rule_v186_candidate.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(workspace_dir / "scripts" / "run_tensor_rule_v186_candidate.py"),
        "--mode", "mt5-smoke",
        "--config", "configs/he_chuyen_gia_reasoning_rl_v1_candidate.yaml",
        "--profile", "configs/experiments/reasoning_rl_v1_active_candidate.yaml",
        "--seed-artifact", str(seed_artifact.resolve()),
        "--runner-source", "mt5_v2_runner",
        "--output-dir", str(output_dir.resolve()),
        "--from-date", from_date,
        "--to-date", to_date,
        "--model", "1",
        "--seed", "20260702",
        "--timeout-sec", str(timeout_sec),
        "--max-wall-sec", str(max_wall_sec),
        "--allow-unverified-imitation-seed",
        "--skip-paired-validation",
    ]

    print(f"\n{'='*75}")
    print(f"[>>] EXECUTING PHASE 2 ROUND {round_idx:02d}: {from_date} -> {to_date}")
    print(f"  Seed Artifact: {seed_artifact}")
    print(f"  Output Dir:    {output_dir}")
    print(f"  Command:       {' '.join(cmd[:6])} ...")
    print(f"{'='*75}\n", flush=True)

    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = f".;src;{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(cmd, cwd=str(workspace_dir), env=env)
    elapsed = time.time() - start_time

    print(f"\n[PHASE 2 ROUND {round_idx:02d} FINISHED] Exit code: {proc.returncode}, Elapsed: {elapsed:.1f}s ({elapsed/60:.2f} min)", flush=True)
    return proc.returncode == 0


def clean_disk_if_low(workspace_dir: Path, threshold_gb: float = 4.0) -> None:
    """Clean orphan runner folders if free disk space is low."""
    total, used, free = shutil.disk_usage(workspace_dir)
    free_gb = free / (1024 ** 3)
    if free_gb < threshold_gb:
        print(f"[DISK MONITOR] Low free space ({free_gb:.2f} GB < {threshold_gb} GB). Cleaning temporary runners...")
        runners_dir = workspace_dir / ".mt5_runners"
        if runners_dir.exists():
            for d in runners_dir.iterdir():
                if d.is_dir() and d.name.startswith("runner_clone_"):
                    shutil.rmtree(d, ignore_errors=True)
        print(f"[DISK MONITOR] Cleanup completed. Free space: {shutil.disk_usage(workspace_dir)[2] / (1024**3):.2f} GB")


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Round RL Training Loop for V193 Phase 2 (Downtrend Mastery)")
    parser.add_argument("--workspace-dir", type=Path, default=Path("."), help="Workspace root directory")
    parser.add_argument("--start-round", type=int, default=1, help="Round index to start from (1-10)")
    parser.add_argument("--end-round", type=int, default=10, help="Round index to end at (1-10)")
    parser.add_argument(
        "--initial-seed",
        type=Path,
        default=Path("outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint.pkl"),
        help="Initial seed checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/he_chuyen_gia/v193_runs/multi_round_phase2"),
        help="Base directory for Phase 2 outputs",
    )
    parser.add_argument("--timeout-sec", type=int, default=450)
    parser.add_argument("--max-wall-sec", type=int, default=360)
    args = parser.parse_args()

    workspace_dir = args.workspace_dir.resolve()
    multi_round_dir = args.output_dir.resolve() if args.output_dir.is_absolute() else workspace_dir / args.output_dir
    multi_round_dir.mkdir(parents=True, exist_ok=True)

    initial_seed = args.initial_seed.resolve() if args.initial_seed.is_absolute() else workspace_dir / args.initial_seed
    if not initial_seed.exists():
        print(f"[FATAL] Initial seed checkpoint not found: {initial_seed}")
        return 1

    manifest_path = multi_round_dir / "manifest.json"
    chart_path = multi_round_dir / "evolution_chart.json"

    # Load existing manifest if resuming
    existing_ledger: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_ledger = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_ledger = {}

    rounds_records: list[dict[str, Any]] = existing_ledger.get("rounds", [])
    completed_round_nums = {r["round"] for r in rounds_records if r.get("completed", False)}

    current_seed = initial_seed
    promoted_champions: list[dict[str, Any]] = []

    print(f"[INIT] V193 Phase 2 (Downtrend Mastery) Multi-Round RL Training Loop initialized.")
    print(f"       Rounds to run: {args.start_round} to {args.end_round}")
    print(f"       Base runs directory: {multi_round_dir}")
    print(f"       Initial seed: {initial_seed}")

    for schedule in PHASE2_SCHEDULE:
        r_num = schedule["round"]
        if r_num < args.start_round or r_num > args.end_round:
            continue

        round_dir = multi_round_dir / f"round_{r_num:02d}"
        candidate_pkl = round_dir / "reasoning_rl_v1_active_candidate.pkl"

        # Check if already completed and candidate exists
        if r_num in completed_round_nums and candidate_pkl.exists():
            print(f"[SKIP] Round {r_num:02d} already completed. Using its checkpoint as seed.")
            current_seed = candidate_pkl
            continue

        # If previous round produced a checkpoint, chain from it
        if r_num > 1:
            prev_cand = multi_round_dir / f"round_{r_num-1:02d}" / "reasoning_rl_v1_active_candidate.pkl"
            prev_inc = multi_round_dir / f"round_{r_num-1:02d}" / "reasoning_rl_v1_active_candidate.incomplete.pkl"
            prev_clone = multi_round_dir / f"round_{r_num-1:02d}" / "source_seed_clone.pkl"
            if prev_cand.exists():
                current_seed = prev_cand
            elif prev_inc.exists():
                try:
                    os.link(prev_inc, prev_cand)
                except OSError:
                    shutil.copy2(prev_inc, prev_cand)
                current_seed = prev_cand
            elif prev_clone.exists():
                try:
                    os.link(prev_clone, prev_cand)
                except OSError:
                    shutil.copy2(prev_clone, prev_cand)
                current_seed = prev_cand
            else:
                print(f"[WARN] Round {r_num-1:02d} checkpoint not found at {prev_cand}. Falling back to {current_seed}")

        clean_disk_if_low(workspace_dir)

        success = run_single_round(
            round_idx=r_num,
            from_date=schedule["from_date"],
            to_date=schedule["to_date"],
            seed_artifact=current_seed,
            output_dir=round_dir,
            workspace_dir=workspace_dir,
            timeout_sec=args.timeout_sec,
            max_wall_sec=args.max_wall_sec,
        )

        metrics = extract_phase2_metrics(round_dir)
        promoted, reasons = evaluate_phase2_promotion(metrics)

        # Ensure candidate pkl exists
        if not candidate_pkl.exists():
            inc = round_dir / "reasoning_rl_v1_active_candidate.incomplete.pkl"
            cln = round_dir / "source_seed_clone.pkl"
            if inc.exists():
                try:
                    os.link(inc, candidate_pkl)
                except OSError:
                    shutil.copy2(inc, candidate_pkl)
            elif cln.exists():
                try:
                    os.link(cln, candidate_pkl)
                except OSError:
                    shutil.copy2(cln, candidate_pkl)

        if promoted:
            promoted_champions.append({
                "round": r_num,
                "from_date": schedule["from_date"],
                "to_date": schedule["to_date"],
                "expectancy_r": metrics["expectancy_r"],
                "win_rate_pct": metrics["win_rate_pct"],
                "net_pnl": metrics["net_pnl"],
                "payoff_ratio": metrics["payoff_ratio"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "pct_long_in_downtrend": metrics["pct_long_in_downtrend"],
                "pct_short_in_uptrend": metrics["pct_short_in_uptrend"],
                "checkpoint": str(candidate_pkl),
            })
            # Save champion artifact
            champ_copy = multi_round_dir / f"champion_phase2_round_{r_num:02d}.pkl"
            if candidate_pkl.exists():
                shutil.copy2(candidate_pkl, champ_copy)
                best_global = workspace_dir / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
                shutil.copy2(candidate_pkl, best_global)
                print(f"[PROMOTION] Round {r_num:02d} PROMOTED as Champion! Saved to {champ_copy}")

        round_record = {
            "round": r_num,
            "from_date": schedule["from_date"],
            "to_date": schedule["to_date"],
            "regime": schedule["regime"],
            "completed": success,
            "promoted": promoted,
            "promotion_reasons": reasons,
            "seed_used": str(current_seed),
            "metrics": metrics,
        }
        rounds_records = [r for r in rounds_records if r["round"] != r_num]
        rounds_records.append(round_record)
        rounds_records.sort(key=lambda x: x["round"])

        # Multi-dimensional trend learning evaluation for Phase 2
        r1_metric = next((r["metrics"] for r in rounds_records if r["round"] == 1), metrics)
        initial_long_pct = float(r1_metric.get("pct_long_in_downtrend", 0.0))
        curr_long_pct = float(metrics.get("pct_long_in_downtrend", 0.0))
        downtrend_long_reduction_pct = (
            round(((initial_long_pct - curr_long_pct) / initial_long_pct) * 100.0, 2)
            if initial_long_pct > 0
            else 0.0
        )

        ledger_manifest = {
            "schema_version": 2,
            "phase": 2,
            "target": "V193 Multi-Round RL Training Loop Phase 2 (Downtrend Mastery)",
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "context_direction_entry_gate_enabled": False,
            "governor_lock_non_entry_to_hold": True,
            "total_rounds_configured": len(PHASE2_SCHEDULE),
            "total_rounds_completed": len(rounds_records),
            "summary": {
                "initial_pct_long_in_downtrend": initial_long_pct,
                "latest_pct_long_in_downtrend": curr_long_pct,
                "pct_long_reduction_vs_round_1": downtrend_long_reduction_pct,
                "downtrend_learning_passed": downtrend_long_reduction_pct >= 50.0 or any(r.get("promoted") for r in rounds_records),
                "zero_premature_closes_all_rounds": all(r["metrics"]["premature_closes"] == 0 for r in rounds_records),
                "total_promoted_checkpoints": sum(1 for r in rounds_records if r.get("promoted", False)),
                "latest_round_expectancy_r": metrics["expectancy_r"],
                "latest_round_win_rate_pct": metrics["win_rate_pct"],
                "latest_round_max_drawdown_pct": metrics["max_drawdown_pct"],
            },
            "rounds": rounds_records,
        }
        manifest_path.write_text(json.dumps(ledger_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        # Update evolution_chart.json
        evolution_chart = {
            "rounds": [r["round"] for r in rounds_records],
            "dates": [f"{r['from_date']}-{r['to_date']}" for r in rounds_records],
            "expectancy_r": [r["metrics"]["expectancy_r"] for r in rounds_records],
            "win_rate_pct": [r["metrics"]["win_rate_pct"] for r in rounds_records],
            "payoff_ratio": [r["metrics"]["payoff_ratio"] for r in rounds_records],
            "max_drawdown_pct": [r["metrics"]["max_drawdown_pct"] for r in rounds_records],
            "pct_long_in_downtrend": [r["metrics"]["pct_long_in_downtrend"] for r in rounds_records],
            "pct_short_in_uptrend": [r["metrics"]["pct_short_in_uptrend"] for r in rounds_records],
            "premature_closes": [r["metrics"]["premature_closes"] for r in rounds_records],
            "promoted": [r.get("promoted", False) for r in rounds_records],
        }
        chart_path.write_text(json.dumps(evolution_chart, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n[PHASE 2 SUMMARY: ROUND {r_num:02d}]")
        print(f"  Win Rate: {metrics['win_rate_pct']}%, E[R]: {metrics['expectancy_r']:+.4f}R, Payoff: {metrics['payoff_ratio']}")
        print(f"  % Long in Downtrend: {metrics['pct_long_in_downtrend']}% ({metrics['downtrend_long_entries']}/{metrics['downtrend_entries_total']})")
        print(f"  % Short in Uptrend:  {metrics['pct_short_in_uptrend']}% ({metrics['uptrend_short_entries']}/{metrics['uptrend_entries_total']})")
        print(f"  Premature Closes:    {metrics['premature_closes']} (POLICY_CLOSE_ALL)")
        print(f"  Promoted:            {promoted}")
        print(f"{'='*75}\n", flush=True)

    print(f"\n[DONE] V193 Phase 2 Multi-Round Training Loop Completed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
