#!/usr/bin/env python3
"""
scripts/stress_test_hybrid1_sensitivity.py

Milestone 1 Adversarial Stress Test & Parameter Sensitivity Analysis for Hybrid 1 Champion.

Stress testing dimensions:
1. Cooldown duration perturbation: [30m, 45m, 60m, 90m, 120m]
2. SL trigger threshold perturbation: [1 SL, 2 SLs, 3 SLs]
3. H1 EMA parameters perturbation:
   - EMA12 / EMA26
   - EMA21 / EMA50 (baseline)
   - EMA9 / EMA21
4. Threshold margins: [0.00, 0.02, 0.05, 0.10]
   - Evaluated under both Hysteresis (regime persistence) and CHOP (defensive silence) deadband modes
   - Evaluated under ATR-normalized and relative price scale

Verification gates:
- Gate 1: Window 1 Net PnL > +$150.00 USD
- Gate 2: Window 2 Net PnL > +$2,000.00 USD
- Gate 3: Worst Max DD < 10.00%
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.regime_data import load_raw_btcusd_m1
from scripts.simulate_candidate_benchmarks import (
    TradeRecord,
    PerformanceMetrics,
    compute_metrics,
    load_execution_ledger,
)


def load_raw_m1_data(start_date: str = "2025-01-15", end_date: str = "2025-02-28") -> pd.DataFrame:
    """Loads raw M1 data."""
    print(f"[StressTest] Ingesting M1 dataset ({start_date} -> {end_date})...")
    m1_df = load_raw_btcusd_m1(start_date=start_date, end_date=end_date)
    return m1_df.sort_values("datetime").reset_index(drop=True)


def compute_h1_indicators(
    m1_df: pd.DataFrame,
    fast_span: int,
    slow_span: int,
) -> pd.DataFrame:
    """Computes H1 candles, EMAs, and H1 ATR (14)."""
    h1_df = (
        m1_df.set_index("datetime")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    h1_df["h1_fast"] = h1_df["close"].ewm(span=fast_span, adjust=False).mean()
    h1_df["h1_slow"] = h1_df["close"].ewm(span=slow_span, adjust=False).mean()
    h1_df["h1_diff"] = h1_df["h1_fast"] - h1_df["h1_slow"]
    h1_df["h1_diff_pct"] = (h1_df["h1_diff"] / h1_df["h1_slow"]) * 100.0

    # H1 ATR(14)
    tr1 = h1_df["high"] - h1_df["low"]
    tr2 = (h1_df["high"] - h1_df["close"].shift(1)).abs()
    tr3 = (h1_df["low"] - h1_df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    h1_df["h1_atr"] = tr.rolling(14, min_periods=1).mean().bfill()
    h1_df["h1_diff_atr"] = h1_df["h1_diff"] / h1_df["h1_atr"]
    return h1_df


def assign_h1_macro_regime(
    h1_df: pd.DataFrame,
    margin: float,
    scale_type: str = "atr",  # "atr" or "pct"
    deadband_mode: str = "hysteresis",  # "hysteresis", "chop", or "bear"
) -> pd.DataFrame:
    """Assigns H1 macro bull/bear flags given a threshold margin and deadband resolution."""
    df = h1_df.copy()
    diff_col = "h1_diff_atr" if scale_type == "atr" else "h1_diff_pct"

    bull_flags: List[bool] = []
    bear_flags: List[bool] = []

    last_regime = "BEAR"  # Default initial state

    for val in df[diff_col]:
        if val > margin:
            current = "BULL"
        elif val < -margin:
            current = "BEAR"
        else:
            # Inside deadband [-margin, +margin]
            if deadband_mode == "hysteresis":
                current = last_regime
            elif deadband_mode == "chop":
                current = "CHOP"
            elif deadband_mode == "bear":
                current = "BEAR"
            else:
                current = last_regime

        last_regime = current if current in ("BULL", "BEAR") else last_regime
        bull_flags.append(current == "BULL")
        bear_flags.append(current == "BEAR")

    df["h1_macro_bull"] = bull_flags
    df["h1_macro_bear"] = bear_flags
    return df


def align_macro_to_trades(
    trades: List[TradeRecord],
    m1_df: pd.DataFrame,
    h1_df: pd.DataFrame,
) -> List[TradeRecord]:
    """Attaches lookahead-free H1 macro flags to trade records based on fill_time."""
    if not trades:
        return []
    # Merge h1_macro flags into m1_df
    m1_aligned = pd.merge_asof(
        m1_df[["datetime"]],
        h1_df[["datetime", "h1_macro_bull", "h1_macro_bear"]],
        on="datetime",
        direction="backward",
    )

    tdf = pd.DataFrame([asdict(t) for t in trades])
    tdf["fill_time"] = pd.to_datetime(tdf["fill_time"])
    tdf = tdf.drop(columns=["h1_macro_bull", "h1_macro_bear"], errors="ignore")
    tdf = pd.merge_asof(
        tdf.sort_values("fill_time"),
        m1_aligned[["datetime", "h1_macro_bull", "h1_macro_bear"]],
        left_on="fill_time",
        right_on="datetime",
        direction="backward",
    )

    enriched: List[TradeRecord] = []
    for _, row in tdf.iterrows():
        enriched.append(
            TradeRecord(
                model=row["model"],
                ticket=int(row["ticket"]),
                action=row["action"],
                fill_time=row["fill_time"],
                close_time=row["close_time"],
                entry_price=row["entry_price"],
                close_price=row["close_price"],
                sl_price=row["sl_price"],
                tp_price=row["tp_price"],
                profit=row["profit"],
                close_reason=row["close_reason"],
                volume=row["volume"],
                h1_macro_bull=bool(row["h1_macro_bull"]),
                h1_macro_bear=bool(row["h1_macro_bear"]),
            )
        )
    return enriched


def filter_htf_macro(input_trades: List[TradeRecord]) -> List[TradeRecord]:
    """Filters trades by H1 macro direction (Option E)."""
    allowed: List[TradeRecord] = []
    for t in input_trades:
        # If Bull regime, allow only BULL model (suppress BEAR)
        if t.h1_macro_bull and t.model == "BEAR":
            continue
        # If Bear regime, allow only BEAR model (suppress BULL)
        if t.h1_macro_bear and t.model == "BULL":
            continue
        # If CHOP (neither bull nor bear), suppress both
        if not t.h1_macro_bull and not t.h1_macro_bear:
            continue
        allowed.append(t)
    return allowed


def simulate_circuit_breaker_stress(
    input_trades: List[TradeRecord],
    max_consecutive_sl: int = 2,
    cooldown_minutes: int = 60,
    sl_lookback_minutes: int = 60,
) -> List[TradeRecord]:
    """Parametric Circuit Breaker simulation."""
    events: List[Tuple[str, pd.Timestamp, TradeRecord]] = []
    for t in input_trades:
        events.append(("FILL", t.fill_time, t))
        events.append(("CLOSE", t.close_time, t))
    events.sort(key=lambda x: x[1])

    cooldowns: Dict[str, Optional[pd.Timestamp]] = {"BEAR": None, "BULL": None}
    sl_history: Dict[str, List[pd.Timestamp]] = {"BEAR": [], "BULL": []}
    allowed_trades: List[TradeRecord] = []

    for ev_type, ev_time, t in events:
        m = t.model
        if ev_type == "FILL":
            cd = cooldowns[m]
            if cd is not None and ev_time < cd:
                continue
            allowed_trades.append(t)
        elif ev_type == "CLOSE":
            if t in allowed_trades:
                if t.close_reason == "SL":
                    sl_history[m] = [
                        tm for tm in sl_history[m] if (ev_time - tm).total_seconds() <= sl_lookback_minutes * 60
                    ]
                    sl_history[m].append(ev_time)
                    if len(sl_history[m]) >= max_consecutive_sl:
                        cooldowns[m] = ev_time + pd.Timedelta(minutes=cooldown_minutes)
                        sl_history[m] = []
                elif t.close_reason == "TP" or t.profit > 0:
                    sl_history[m] = []
    return allowed_trades


@dataclass
class StressResult:
    cooldown_min: int
    sl_threshold: int
    ema_pair: str
    margin: float
    scale_type: str
    deadband_mode: str
    w1_pnl: float
    w1_wr: float
    w1_sl: int
    w1_dd: float
    w2_pnl: float
    w2_wr: float
    w2_sl: int
    w2_dd: float
    tot_pnl: float
    comb_pf: float
    worst_dd: float
    gate_w1_pass: bool
    gate_w2_pass: bool
    gate_dd_pass: bool
    all_gates_pass: bool


def run_stress_test_grid() -> Tuple[List[StressResult], pd.DataFrame]:
    m1_df = load_raw_m1_data()

    # Ingest ground-truth ledgers
    bear_w1_path = (
        ROOT
        / "outputs/he_chuyen_gia/v193_runs/multi_round_phase2/frozen_oos_phase2_champion_artifacts/window_01/self_growing_execution.csv"
    )
    bear_w2_path = (
        ROOT
        / "outputs/he_chuyen_gia/v193_runs/multi_round_phase2/frozen_oos_phase2_champion_artifacts/window_02/self_growing_execution.csv"
    )
    bull_w1_path = (
        ROOT
        / "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/frozen_oos_round_01_artifacts/window_01/self_growing_execution.csv"
    )
    bull_w2_path = (
        ROOT
        / "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/frozen_oos_round_01_artifacts/window_02/self_growing_execution.csv"
    )

    print("[StressTest] Ingesting ground-truth execution ledgers...")
    bear_w1_raw = load_execution_ledger(bear_w1_path, "BEAR")
    bear_w2_raw = load_execution_ledger(bear_w2_path, "BEAR")
    bull_w1_raw = load_execution_ledger(bull_w1_path, "BULL")
    bull_w2_raw = load_execution_ledger(bull_w2_path, "BULL")

    cooldowns = [30, 45, 60, 90, 120]
    sl_triggers = [1, 2, 3]
    ema_pairs = [
        ("EMA12/EMA26", 12, 26),
        ("EMA21/EMA50", 21, 50),
        ("EMA9/EMA21", 9, 21),
    ]
    margins = [0.00, 0.02, 0.05, 0.10]
    deadband_modes = ["hysteresis", "chop"]

    results: List[StressResult] = []

    print("[StressTest] Running parameter perturbation matrix across all dimensions...")

    for ema_label, f_span, s_span in ema_pairs:
        raw_h1 = compute_h1_indicators(m1_df, f_span, s_span)

        for margin in margins:
            for db_mode in deadband_modes:
                # Assign macro flags
                h1_assigned = assign_h1_macro_regime(raw_h1, margin=margin, scale_type="atr", deadband_mode=db_mode)

                # Attach to trades
                b_w1 = align_macro_to_trades(bear_w1_raw, m1_df, h1_assigned)
                b_w2 = align_macro_to_trades(bear_w2_raw, m1_df, h1_assigned)
                l_w1 = align_macro_to_trades(bull_w1_raw, m1_df, h1_assigned)
                l_w2 = align_macro_to_trades(bull_w2_raw, m1_df, h1_assigned)

                all_w1 = sorted(b_w1 + l_w1, key=lambda x: x.fill_time)
                all_w2 = sorted(b_w2 + l_w2, key=lambda x: x.fill_time)

                # Step 1: HTF Macro Filter
                filtered_w1 = filter_htf_macro(all_w1)
                filtered_w2 = filter_htf_macro(all_w2)

                for cd in cooldowns:
                    for sl_t in sl_triggers:
                        # Step 2: Circuit Breaker
                        cb_w1 = simulate_circuit_breaker_stress(
                            filtered_w1, max_consecutive_sl=sl_t, cooldown_minutes=cd
                        )
                        cb_w2 = simulate_circuit_breaker_stress(
                            filtered_w2, max_consecutive_sl=sl_t, cooldown_minutes=cd
                        )

                        m1 = compute_metrics(cb_w1, "Hybrid 1", "Window 1")
                        m2 = compute_metrics(cb_w2, "Hybrid 1", "Window 2")

                        tot_pnl = round(m1.net_pnl + m2.net_pnl, 2)
                        comb_profit = m1.gross_profit + m2.gross_profit
                        comb_loss = m1.gross_loss + m2.gross_loss
                        comb_pf = round(comb_profit / comb_loss, 2) if comb_loss > 0 else (999.0 if comb_profit > 0 else 0.0)
                        worst_dd = max(m1.max_dd_pct, m2.max_dd_pct)

                        g1 = m1.net_pnl > 150.0
                        g2 = m2.net_pnl > 2000.0
                        g3 = worst_dd < 10.0
                        all_pass = g1 and g2 and g3

                        results.append(
                            StressResult(
                                cooldown_min=cd,
                                sl_threshold=sl_t,
                                ema_pair=ema_label,
                                margin=margin,
                                scale_type="atr",
                                deadband_mode=db_mode,
                                w1_pnl=m1.net_pnl,
                                w1_wr=m1.win_rate,
                                w1_sl=m1.sl_count,
                                w1_dd=m1.max_dd_pct,
                                w2_pnl=m2.net_pnl,
                                w2_wr=m2.win_rate,
                                w2_sl=m2.sl_count,
                                w2_dd=m2.max_dd_pct,
                                tot_pnl=tot_pnl,
                                comb_pf=comb_pf,
                                worst_dd=worst_dd,
                                gate_w1_pass=g1,
                                gate_w2_pass=g2,
                                gate_dd_pass=g3,
                                all_gates_pass=all_pass,
                            )
                        )

    df_res = pd.DataFrame([asdict(r) for r in results])
    print(f"[StressTest] Completed {len(df_res)} parameter combinations.")
    return results, df_res


def print_summary_analyses(df_res: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print("  HYBRID 1 SENSITIVITY STRESS TEST ANALYSIS — SUMMARY BY DIMENSION")
    print("=" * 120)

    # 1. By Cooldown Duration (under baseline EMA21/EMA50, SL=2, Margin=0.00)
    print("\n--- 1. COOLDOWN DURATION PERTURBATION (EMA21/50, SL=2, Margin=0.00) ---")
    base_cd = df_res[
        (df_res["ema_pair"] == "EMA21/EMA50")
        & (df_res["sl_threshold"] == 2)
        & (df_res["margin"] == 0.00)
        & (df_res["deadband_mode"] == "hysteresis")
    ].sort_values("cooldown_min")
    print(
        base_cd[
            [
                "cooldown_min",
                "w1_pnl",
                "w1_wr",
                "w1_sl",
                "w2_pnl",
                "w2_wr",
                "w2_sl",
                "tot_pnl",
                "worst_dd",
                "all_gates_pass",
            ]
        ].to_string(index=False)
    )

    # 2. By SL Trigger Threshold (under baseline EMA21/EMA50, Cooldown=60, Margin=0.00)
    print("\n--- 2. SL TRIGGER THRESHOLD PERTURBATION (EMA21/50, CD=60m, Margin=0.00) ---")
    base_sl = df_res[
        (df_res["ema_pair"] == "EMA21/EMA50")
        & (df_res["cooldown_min"] == 60)
        & (df_res["margin"] == 0.00)
        & (df_res["deadband_mode"] == "hysteresis")
    ].sort_values("sl_threshold")
    print(
        base_sl[
            [
                "sl_threshold",
                "w1_pnl",
                "w1_wr",
                "w1_sl",
                "w2_pnl",
                "w2_wr",
                "w2_sl",
                "tot_pnl",
                "worst_dd",
                "all_gates_pass",
            ]
        ].to_string(index=False)
    )

    # 3. By H1 EMA Parameters (under baseline CD=60m, SL=2, Margin=0.00)
    print("\n--- 3. H1 EMA PARAMETER PERTURBATION (CD=60m, SL=2, Margin=0.00) ---")
    base_ema = df_res[
        (df_res["cooldown_min"] == 60)
        & (df_res["sl_threshold"] == 2)
        & (df_res["margin"] == 0.00)
        & (df_res["deadband_mode"] == "hysteresis")
    ].sort_values("ema_pair")
    print(
        base_ema[
            [
                "ema_pair",
                "w1_pnl",
                "w1_wr",
                "w1_sl",
                "w2_pnl",
                "w2_wr",
                "w2_sl",
                "tot_pnl",
                "worst_dd",
                "all_gates_pass",
            ]
        ].to_string(index=False)
    )

    # 4. By Threshold Margin (under baseline CD=60m, SL=2, EMA21/50)
    print("\n--- 4. THRESHOLD MARGIN PERTURBATION (CD=60m, SL=2, EMA21/50) ---")
    base_mar = df_res[
        (df_res["cooldown_min"] == 60)
        & (df_res["sl_threshold"] == 2)
        & (df_res["ema_pair"] == "EMA21/EMA50")
    ].sort_values(["deadband_mode", "margin"])
    print(
        base_mar[
            [
                "deadband_mode",
                "margin",
                "w1_pnl",
                "w1_wr",
                "w1_sl",
                "w2_pnl",
                "w2_wr",
                "w2_sl",
                "tot_pnl",
                "worst_dd",
                "all_gates_pass",
            ]
        ].to_string(index=False)
    )

    # Overall Global Pass Rate
    total_runs = len(df_res)
    passed_runs = df_res["all_gates_pass"].sum()
    print("\n" + "=" * 120)
    print(f"  GLOBAL ROBUSTNESS RATE: {passed_runs} / {total_runs} combinations passed ALL GATES ({passed_runs/total_runs*100:.1f}%)")
    print(f"  Gate 1 Pass (W1 > $150): {df_res['gate_w1_pass'].sum()} / {total_runs} ({df_res['gate_w1_pass'].mean()*100:.1f}%)")
    print(f"  Gate 2 Pass (W2 > $2000): {df_res['gate_w2_pass'].sum()} / {total_runs} ({df_res['gate_w2_pass'].mean()*100:.1f}%)")
    print(f"  Gate 3 Pass (Worst DD < 10%): {df_res['gate_dd_pass'].sum()} / {total_runs} ({df_res['gate_dd_pass'].mean()*100:.1f}%)")
    print("=" * 120 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test Hybrid 1 sensitivity across parameter dimensions.")
    parser.add_argument(
        "--output-csv",
        type=str,
        default=str(ROOT / "outputs/candidate_benchmarks/hybrid1_sensitivity_stress_test.csv"),
        help="Path to save output CSV of all stress test combinations",
    )
    args = parser.parse_args()

    results, df_res = run_stress_test_grid()
    print_summary_analyses(df_res)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_res.to_csv(out_path, index=False)
    print(f"[StressTest] Full results matrix saved to: {out_path}")


if __name__ == "__main__":
    main()
