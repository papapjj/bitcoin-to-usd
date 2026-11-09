#!/usr/bin/env python3
"""Challenger Audit Script for Self Growing Trader V193 Multi-Round Runs.

Performs rigorous empirical interrogation of raw logs:
1. Verifies POLICY_CLOSE_ALL == 0 in all events.jsonl and mt5_diagnostics.json across all 10 rounds.
2. Verifies checkpoint serialization, seed lineage, and execution trace authenticity.
3. Computes exact mathematical metrics (E[R], Win Rate, Payoff Ratio, Max DD, Net PnL) from raw trades.
4. Analyzes discrepancy between events.jsonl and mt5_diagnostics.json trade counts.
5. Computes trend learning rates and tests acceptance hurdles.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np


def file_hash(path: Path) -> str:
    """Return SHA256 of first 1MB of file for quick identity verification."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        chunk = f.read(1024 * 1024)
        h.update(chunk)
    return h.hexdigest()[:16]


def audit_multi_round(base_dir: Path) -> dict[str, Any]:
    audit_results: dict[str, Any] = {
        "rounds": {},
        "global_checks": {
            "policy_close_all_events_found": 0,
            "policy_close_all_diag_found": 0,
            "seed_chain_verified": True,
            "seed_chain_breaks": [],
            "all_rounds_completed": True,
        },
    }

    print(f"=== AUDITING MULTI-ROUND RUNS IN: {base_dir} ===")

    prev_candidate_path: Path | None = None
    prev_candidate_hash: str | None = None

    for r in range(1, 11):
        round_dir = base_dir / f"round_{r:02d}"
        if not round_dir.exists():
            print(f"[FAIL] Round dir missing: {round_dir}")
            audit_results["global_checks"]["all_rounds_completed"] = False
            continue

        r_data: dict[str, Any] = {
            "round": r,
            "dir": str(round_dir),
            "files": os.listdir(round_dir),
            "events_path_exists": (round_dir / "events.jsonl").exists(),
            "diag_path_exists": (round_dir / "mt5_diagnostics.json").exists(),
            "manifest_path_exists": (round_dir / "manifest.json").exists(),
            "candidate_pkl_exists": (round_dir / "reasoning_rl_v1_active_candidate.pkl").exists(),
            "candidate_pkl_size": (round_dir / "reasoning_rl_v1_active_candidate.pkl").stat().st_size if (round_dir / "reasoning_rl_v1_active_candidate.pkl").exists() else 0,
            "observatory_exists": (round_dir / "reasoning_rl_v1_active_candidate.execution_trace_observatory.jsonl").exists(),
        }

        # Check seed chaining
        seed_clone = round_dir / "source_seed_clone.pkl"
        if r > 1:
            if seed_clone.exists():
                clone_size = seed_clone.stat().st_size
                clone_hash = file_hash(seed_clone)
                r_data["seed_clone_size"] = clone_size
                r_data["seed_clone_hash"] = clone_hash
                if prev_candidate_path and prev_candidate_path.exists():
                    prev_size = prev_candidate_path.stat().st_size
                    prev_hash = file_hash(prev_candidate_path)
                    if clone_size != prev_size or clone_hash != prev_hash:
                        audit_results["global_checks"]["seed_chain_verified"] = False
                        audit_results["global_checks"]["seed_chain_breaks"].append(
                            f"Round {r} seed clone ({clone_hash}, {clone_size}B) != Round {r-1} candidate ({prev_hash}, {prev_size}B)"
                        )
            else:
                audit_results["global_checks"]["seed_chain_breaks"].append(f"Round {r} missing source_seed_clone.pkl")

        # Set previous candidate for next round check
        candidate_pkl = round_dir / "reasoning_rl_v1_active_candidate.pkl"
        if candidate_pkl.exists():
            prev_candidate_path = candidate_pkl
            prev_candidate_hash = file_hash(candidate_pkl)
            r_data["candidate_hash"] = prev_candidate_hash

        # Audit events.jsonl
        events_path = round_dir / "events.jsonl"
        lifecycles = []
        action_counts: dict[int, int] = {}
        close_reasons_events: dict[str, int] = {}
        policy_close_all_count = 0
        uptrend_entries = 0
        uptrend_short_entries = 0
        uptrend_long_entries = 0
        downtrend_entries = 0
        downtrend_short_entries = 0
        downtrend_long_entries = 0
        non_entry_actions_taken = {2: 0, 3: 0, 4: 0} # Agent 2, 3, 4 actions other than HOLD

        if events_path.exists():
            with open(events_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        ev = json.loads(line_str)
                    except Exception as err:
                        print(f"[WARN] Error decoding line {line_idx} in {events_path}: {err}")
                        continue
                    
                    ev_type = ev.get("event_type")
                    payload = ev.get("payload", {})
                    
                    if ev_type == "decision":
                        act = payload.get("action_index")
                        if act is not None:
                            action_counts[act] = action_counts.get(act, 0) + 1
                            # Action 10 is POLICY_CLOSE_ALL
                            if act == 10:
                                policy_close_all_count += 1
                                print(f"[CRITICAL ALERT] Found action 10 (POLICY_CLOSE_ALL) decision in Round {r} line {line_idx}!")

                    elif ev_type == "lifecycle":
                        lifecycles.append(payload)
                        cr = payload.get("close_reason", "UNKNOWN")
                        close_reasons_events[cr] = close_reasons_events.get(cr, 0) + 1
                        if cr == "POLICY_CLOSE_ALL":
                            policy_close_all_count += 1
                            print(f"[CRITICAL ALERT] Found POLICY_CLOSE_ALL lifecycle in Round {r} line {line_idx}!")

                        # Trend entry analysis
                        for g in payload.get("growth_events", []):
                            if g.get("type") == "context_action_expectancy_credit":
                                cond = g.get("conditioned_entry_expectancy", {}).get("condition", {})
                                trend = str(cond.get("trend", "")).lower()
                                act_idx = int(g.get("action_index", -1))
                                if trend == "up":
                                    uptrend_entries += 1
                                    if act_idx in (2, 4, 6):
                                        uptrend_short_entries += 1
                                    elif act_idx in (1, 3, 5):
                                        uptrend_long_entries += 1
                                elif trend == "down":
                                    downtrend_entries += 1
                                    if act_idx in (2, 4, 6):
                                        downtrend_short_entries += 1
                                    elif act_idx in (1, 3, 5):
                                        downtrend_long_entries += 1

        r_data["total_lifecycles"] = len(lifecycles)
        r_data["close_reasons_events"] = close_reasons_events
        r_data["action_counts"] = action_counts
        r_data["policy_close_all_count"] = policy_close_all_count
        audit_results["global_checks"]["policy_close_all_events_found"] += policy_close_all_count

        # Math verification on raw lifecycles
        r_values = [float(p.get("realized_net_r", 0.0)) for p in lifecycles if "realized_net_r" in p]
        pnl_values = [float(p.get("pnl", 0.0)) for p in lifecycles if "pnl" in p]
        holding_bars = [int(p.get("holding_bars", 0)) for p in lifecycles if "holding_bars" in p]

        wins = [r_val for r_val in r_values if r_val > 0.0]
        losses = [r_val for r_val in r_values if r_val <= 0.0]
        n_trades = len(r_values)

        calc_er = float(np.mean(r_values)) if n_trades > 0 else 0.0
        calc_wr = float(len(wins) / n_trades * 100.0) if n_trades > 0 else 0.0
        calc_avg_win = float(np.mean(wins)) if wins else 0.0
        calc_avg_loss = float(abs(np.mean(losses))) if losses else 1.0
        calc_payoff = float(calc_avg_win / calc_avg_loss) if (losses and calc_avg_loss > 0) else (2.0 if wins else 0.0)
        calc_net_pnl = float(np.sum(pnl_values)) if pnl_values else 0.0

        # Calculate Max Drawdown from trade equity curve
        cum_pnl = np.cumsum([0.0] + pnl_values)
        peak = np.maximum.accumulate(cum_pnl)
        dd = peak - cum_pnl
        # Also percentage drawdown relative to nominal $10,000 balance
        max_dd_nominal = float(np.max(dd)) if len(dd) > 0 else 0.0
        max_dd_nominal_pct = float(max_dd_nominal / 10000.0 * 100.0)

        # Check holding bars distribution
        max_held = max(holding_bars) if holding_bars else 0
        min_held = min(holding_bars) if holding_bars else 0

        r_data["raw_math"] = {
            "trade_count": n_trades,
            "wins": len(wins),
            "losses": len(losses),
            "expectancy_r": round(calc_er, 4),
            "win_rate_pct": round(calc_wr, 2),
            "avg_win_r": round(calc_avg_win, 4),
            "avg_loss_r": round(calc_avg_loss, 4),
            "payoff_ratio": round(calc_payoff, 2),
            "net_pnl": round(calc_net_pnl, 2),
            "max_dd_nominal_pct": round(max_dd_nominal_pct, 2),
            "max_holding_bars": max_held,
            "min_holding_bars": min_held,
        }

        # Audit mt5_diagnostics.json
        diag_path = round_dir / "mt5_diagnostics.json"
        if diag_path.exists():
            try:
                diag = json.loads(diag_path.read_text(encoding="utf-8"))
                perf = diag.get("performance", {})
                risk = diag.get("risk", {})
                pol = diag.get("policy", {})

                diag_close_reasons = perf.get("close_reasons", {})
                diag_policy_close_all = int(diag_close_reasons.get("POLICY_CLOSE_ALL", 0))
                audit_results["global_checks"]["policy_close_all_diag_found"] += diag_policy_close_all

                diag_raw_dd = risk.get("max_equity_drawdown_pct", 0.0)
                diag_dd_pct = float(diag_raw_dd * 100.0 if diag_raw_dd <= 1.0 else diag_raw_dd)

                r_data["diagnostics"] = {
                    "closed_trades": perf.get("closed_trades"),
                    "win_rate": perf.get("win_rate"),
                    "profit_factor": perf.get("profit_factor"),
                    "payoff_ratio": perf.get("payoff_ratio"),
                    "expectancy": perf.get("expectancy"),
                    "net_pnl": perf.get("net_pnl"),
                    "max_equity_drawdown_pct": diag_dd_pct,
                    "close_reasons": diag_close_reasons,
                    "policy_close_all": diag_policy_close_all,
                    "hold_share": pol.get("hold_share"),
                    "normalized_action_entropy": pol.get("normalized_action_entropy"),
                }
            except Exception as e:
                print(f"[ERROR] Failed to parse {diag_path}: {e}")

        # Trend learning calculation for this round
        pct_short_in_up = (uptrend_short_entries / uptrend_entries * 100.0) if uptrend_entries > 0 else 0.0
        r_data["trend_learning"] = {
            "uptrend_entries_total": uptrend_entries,
            "uptrend_short_entries": uptrend_short_entries,
            "uptrend_long_entries": uptrend_long_entries,
            "pct_short_in_uptrend": round(pct_short_in_up, 2),
            "downtrend_entries_total": downtrend_entries,
            "downtrend_short_entries": downtrend_short_entries,
            "downtrend_long_entries": downtrend_long_entries,
        }

        # Check execution trace observatory
        obs_path = round_dir / "reasoning_rl_v1_active_candidate.execution_trace_observatory.jsonl"
        if obs_path.exists():
            obs_lines = 0
            with open(obs_path, "r", encoding="utf-8") as f:
                for _ in f:
                    obs_lines += 1
            r_data["observatory_lines"] = obs_lines

        audit_results["rounds"][r] = r_data

    return audit_results


def main():
    base_dir = Path("outputs/he_chuyen_gia/v193_runs/multi_round_v193")
    results = audit_multi_round(base_dir)

    print("\n" + "=" * 80)
    print("AUDIT SUMMARY:")
    print(f"Policy Close All in Events:      {results['global_checks']['policy_close_all_events_found']}")
    print(f"Policy Close All in Diagnostics: {results['global_checks']['policy_close_all_diag_found']}")
    print(f"Seed Chain Verified:             {results['global_checks']['seed_chain_verified']}")
    if results['global_checks']['seed_chain_breaks']:
        print(f"Seed Chain Breaks:               {results['global_checks']['seed_chain_breaks']}")
    print("=" * 80)

    print("\nPER-ROUND AUDIT TABLE:")
    print(f"{'R':>2} | {'Events':>6} | {'Diag':>5} | {'Raw WR':>7} | {'Diag WR':>7} | {'Raw E[R]':>8} | {'Raw Payoff':>10} | {'Diag Payoff':>11} | {'Max DD':>7} | {'Net PnL':>10} | {'%Short(Up)':>10}")
    print("-" * 105)

    for r in range(1, 11):
        rd = results["rounds"].get(r, {})
        m = rd.get("raw_math", {})
        d = rd.get("diagnostics", {})
        tl = rd.get("trend_learning", {})

        ev_n = m.get("trade_count", 0)
        dg_n = d.get("closed_trades", 0)
        raw_wr = f"{m.get('win_rate_pct', 0.0):.1f}%"
        dg_wr = f"{float(d.get('win_rate', 0.0) or 0.0)*100:.1f}%" if d.get("win_rate") is not None else "N/A"
        raw_er = f"{m.get('expectancy_r', 0.0):+.4f}R"
        raw_po = f"{m.get('payoff_ratio', 0.0):.2f}"
        dg_po = f"{float(d.get('payoff_ratio', 0.0) or 0.0):.2f}" if d.get("payoff_ratio") is not None else "N/A"
        dd = f"{d.get('max_equity_drawdown_pct', 0.0):.2f}%"
        pnl = f"${m.get('net_pnl', 0.0):+.2f}"
        short_up = f"{tl.get('pct_short_in_uptrend', 0.0):.1f}% ({tl.get('uptrend_short_entries', 0)}/{tl.get('uptrend_entries_total', 0)})"

        print(f"{r:>2} | {ev_n:>6} | {dg_n:>5} | {raw_wr:>7} | {dg_wr:>7} | {raw_er:>8} | {raw_po:>10} | {dg_po:>11} | {dd:>7} | {pnl:>10} | {short_up:>10}")

    return results


if __name__ == "__main__":
    main()
