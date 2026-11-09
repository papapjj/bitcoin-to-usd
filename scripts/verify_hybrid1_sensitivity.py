#!/usr/bin/env python3
"""
scripts/verify_hybrid1_sensitivity.py

Deep-dive empirical verification & stress-testing for HYBRID 1 (Option E + Option B):
1. Parameter Sensitivity Sweep:
   - max_consecutive_sl in [1, 2, 3, 4]
   - cooldown_minutes in [30, 45, 60, 90, 120]
2. Day-by-Day PnL Breakdown (Daily Consistency)
3. Slippage & Cost Resilience ($0, $5, $10, $15 penalty per trade)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from simulate_candidate_benchmarks import (
    load_market_indicators,
    load_execution_ledger,
    attach_market_features,
    simulate_htf_macro_filter,
    simulate_circuit_breaker,
    compute_metrics,
    TradeRecord,
)

def main():
    m1_df = load_market_indicators()
    bear_w1_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase2/frozen_oos_phase2_champion_artifacts/window_01/self_growing_execution.csv"
    bear_w2_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase2/frozen_oos_phase2_champion_artifacts/window_02/self_growing_execution.csv"
    bull_w1_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/frozen_oos_round_01_artifacts/window_01/self_growing_execution.csv"
    bull_w2_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/frozen_oos_round_01_artifacts/window_02/self_growing_execution.csv"

    bear_w1 = attach_market_features(load_execution_ledger(bear_w1_path, "BEAR"), m1_df)
    bear_w2 = attach_market_features(load_execution_ledger(bear_w2_path, "BEAR"), m1_df)
    bull_w1 = attach_market_features(load_execution_ledger(bull_w1_path, "BULL"), m1_df)
    bull_w2 = attach_market_features(load_execution_ledger(bull_w2_path, "BULL"), m1_df)

    w1_all = sorted(bear_w1 + bull_w1, key=lambda x: x.fill_time)
    w2_all = sorted(bear_w2 + bull_w2, key=lambda x: x.fill_time)

    # Base HTF filtered trades
    w1_htf = simulate_htf_macro_filter(w1_all)
    w2_htf = simulate_htf_macro_filter(w2_all)

    print("\n" + "="*80)
    print(" 1. PARAMETER SENSITIVITY SWEEP FOR HYBRID 1 (Cooldown vs Consecutive SL)")
    print("="*80)
    print(f"{'SL Trigger':<12} | {'Cooldown (m)':<14} | {'W1 PnL ($)':<12} | {'W2 PnL ($)':<12} | {'Total PnL ($)':<14} | {'Max DD (%)':<10} | {'Win Rate':<10}")
    print("-" * 88)

    sensitivity_results = []
    for trigger_sl in [1, 2, 3, 4]:
        for cd_mins in [30, 45, 60, 90, 120]:
            w1_hyb = simulate_circuit_breaker(w1_htf, max_consecutive_sl=trigger_sl, cooldown_minutes=cd_mins)
            w2_hyb = simulate_circuit_breaker(w2_htf, max_consecutive_sl=trigger_sl, cooldown_minutes=cd_mins)
            m1 = compute_metrics(w1_hyb, "Hyb1", "W1")
            m2 = compute_metrics(w2_hyb, "Hyb1", "W2")
            tot_pnl = round(m1.net_pnl + m2.net_pnl, 2)
            max_dd = max(m1.max_dd_pct, m2.max_dd_pct)
            tot_trades = m1.total_trades + m2.total_trades
            tot_wins = (m1.win_rate/100.0 * m1.total_trades) + (m2.win_rate/100.0 * m2.total_trades)
            tot_wr = round(tot_wins / tot_trades * 100, 2) if tot_trades else 0.0

            sensitivity_results.append({
                "trigger_sl": trigger_sl,
                "cooldown_minutes": cd_mins,
                "w1_pnl": m1.net_pnl,
                "w2_pnl": m2.net_pnl,
                "total_pnl": tot_pnl,
                "max_dd": max_dd,
                "win_rate": tot_wr,
            })
            print(f"{trigger_sl:<12} | {cd_mins:<14} | ${m1.net_pnl:<11.2f} | ${m2.net_pnl:<11.2f} | ${tot_pnl:<13.2f} | {max_dd:<9.2f}% | {tot_wr:<9.2f}%")

    # 2. Daily PnL Consistency for Baseline vs Hybrid 1 (trigger=2, cd=60)
    w1_hyb_base = simulate_circuit_breaker(w1_htf, max_consecutive_sl=2, cooldown_minutes=60)
    w2_hyb_base = simulate_circuit_breaker(w2_htf, max_consecutive_sl=2, cooldown_minutes=60)
    all_hyb_trades = w1_hyb_base + w2_hyb_base

    df_trades = pd.DataFrame([
        {
            "fill_time": t.fill_time,
            "date": t.fill_time.strftime("%Y-%m-%d"),
            "profit": t.profit,
            "action": t.action,
            "model": t.model,
            "close_reason": t.close_reason,
        }
        for t in all_hyb_trades
    ])

    print("\n" + "="*80)
    print(" 2. DAILY PnL BREAKDOWN FOR HYBRID 1 (Trigger=2, Cooldown=60m)")
    print("="*80)
    daily = df_trades.groupby("date").agg(
        trades=("profit", "count"),
        net_pnl=("profit", "sum"),
        wins=("profit", lambda x: (x > 0).sum()),
        losses=("profit", lambda x: (x < 0).sum()),
    )
    daily["win_rate"] = (daily["wins"] / daily["trades"] * 100).round(2)
    daily["cum_pnl"] = daily["net_pnl"].cumsum().round(2)
    print(daily.to_string())

    # 3. Slippage / Friction Resilience Test
    print("\n" + "="*80)
    print(" 3. FRICTION & SLIPPAGE STRESS-TEST (Deducting $ penalty per trade)")
    print("="*80)
    print(f"{'Penalty / Trade':<18} | {'Total Trades':<14} | {'Net PnL ($)':<14} | {'Profit Factor':<14} | {'Net Return Degradation'}")
    print("-" * 80)
    for penalty in [0.0, 2.5, 5.0, 7.5, 10.0, 15.0]:
        adj_profits = [t.profit - penalty for t in all_hyb_trades]
        adj_net = sum(adj_profits)
        gross_win = sum(p for p in adj_profits if p > 0)
        gross_loss = abs(sum(p for p in adj_profits if p < 0))
        adj_pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else 999.0
        degradation = round((adj_net - daily['net_pnl'].sum()) / daily['net_pnl'].sum() * 100, 2)
        print(f"${penalty:<17.1f} | {len(all_hyb_trades):<14} | ${adj_net:<13.2f} | {adj_pf:<14.2f} | {degradation}%")

    out_json = ROOT / "outputs/candidate_benchmarks/hybrid1_deep_verification.json"
    with open(out_json, "w") as f:
        json.dump({
            "sensitivity": sensitivity_results,
            "daily": daily.reset_index().to_dict(orient="records"),
        }, f, indent=2)
    print(f"\n[Artifact] Saved detailed verification results to {out_json}")

if __name__ == "__main__":
    main()
