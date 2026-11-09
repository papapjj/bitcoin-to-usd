#!/usr/bin/env python3
"""
scripts/simulate_candidate_benchmarks.py

Empirical Simulation Benchmark Matrix for MT5 Strategy Tester Candidates:
Evaluates Candidates A, B, C, D, E and Hybrids across:
- Window 1 (2025.02.20 - 2025.02.24): Bull Wave ($96k -> $98k)
- Window 2 (2025.02.24 - 2025.02.28): Bear Wave ($98k -> $93k)

Ingests ground-truth execution ledgers from:
- outputs/he_chuyen_gia/v193_runs/multi_round_phase2/frozen_oos_phase2_champion_artifacts/
- outputs/he_chuyen_gia/v193_runs/multi_round_phase3/frozen_oos_round_01_artifacts/
and market data from C:\\Users\\Admin\\Desktop\\BTCUSDm.csv.

Outputs:
- Formatted comparative matrix table to stdout
- Machine-readable JSON output
- Machine-readable CSV output
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

# Project root setup
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.regime_data import load_raw_btcusd_m1


@dataclass
class TradeRecord:
    model: str
    ticket: int
    action: str
    fill_time: pd.Timestamp
    close_time: pd.Timestamp
    entry_price: float
    close_price: float
    sl_price: float
    tp_price: float
    profit: float
    close_reason: str
    volume: float
    h1_macro_bull: bool = False
    h1_macro_bear: bool = False
    m5_mom_bull: bool = False
    m5_mom_bear: bool = False


@dataclass
class PerformanceMetrics:
    candidate_name: str
    window_name: str
    total_trades: int
    net_pnl: float
    win_rate: float
    profit_factor: float
    sl_count: int
    tp_count: int
    timeout_count: int
    max_dd_pct: float
    gross_profit: float
    gross_loss: float
    avg_win: float
    avg_loss: float
    payoff_ratio: float


def load_market_indicators(start_date: str = "2025-01-15", end_date: str = "2025-02-28") -> pd.DataFrame:
    """Loads raw M1 data and pre-computes lookahead-free H1 and M5 indicators."""
    print(f"[Data] Loading M1 bars from BTCUSDm.csv ({start_date} -> {end_date})...")
    m1_df = load_raw_btcusd_m1(start_date=start_date, end_date=end_date)
    m1_df = m1_df.sort_values("datetime").reset_index(drop=True)

    # Higher-Timeframe H1 Indicators (EMA21 vs EMA50)
    h1_df = (
        m1_df.set_index("datetime")
        .resample("1h")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    h1_df["h1_ema21"] = h1_df["close"].ewm(span=21, adjust=False).mean()
    h1_df["h1_ema50"] = h1_df["close"].ewm(span=50, adjust=False).mean()
    h1_df["h1_macro_bull"] = h1_df["h1_ema21"] > h1_df["h1_ema50"]
    h1_df["h1_macro_bear"] = h1_df["h1_ema21"] < h1_df["h1_ema50"]

    # Microstructure M5 Indicators (EMA9 vs EMA21 + 3-bar velocity)
    m5_df = (
        m1_df.set_index("datetime")
        .resample("5min")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    m5_df["m5_ema9"] = m5_df["close"].ewm(span=9, adjust=False).mean()
    m5_df["m5_ema21"] = m5_df["close"].ewm(span=21, adjust=False).mean()
    m5_df["m5_velocity"] = m5_df["close"].diff(3)
    m5_df["m5_mom_bull"] = (m5_df["m5_ema9"] > m5_df["m5_ema21"]) & (m5_df["m5_velocity"] > 0)
    m5_df["m5_mom_bear"] = (m5_df["m5_ema9"] < m5_df["m5_ema21"]) & (m5_df["m5_velocity"] < 0)

    # As-of backward merge to ensure lookahead-free indicator alignment
    m1_df = pd.merge_asof(
        m1_df,
        h1_df[["datetime", "h1_macro_bull", "h1_macro_bear"]],
        on="datetime",
        direction="backward",
    )
    m1_df = pd.merge_asof(
        m1_df,
        m5_df[["datetime", "m5_mom_bull", "m5_mom_bear"]],
        on="datetime",
        direction="backward",
    )
    print(f"[Data] Loaded {len(m1_df):,} M1 bars with aligned H1 and M5 indicators.")
    return m1_df


def load_execution_ledger(exec_path: Path, model_label: str) -> List[TradeRecord]:
    """Parses ground-truth execution ledger into TradeRecord instances."""
    if not exec_path.exists():
        raise FileNotFoundError(f"Execution ledger not found at {exec_path}")

    df = pd.read_csv(exec_path, sep="\t")
    closed = df[df["status"] == "CLOSED"].copy()
    filled = df[df["status"] == "FILLED"].copy()

    trades: List[TradeRecord] = []
    for _, f_row in filled.iterrows():
        ticket = int(f_row["position_ticket"])
        c_matches = closed[closed["position_ticket"] == ticket]
        if len(c_matches) == 0:
            continue
        c_row = c_matches.iloc[0]
        trades.append(
            TradeRecord(
                model=model_label,
                ticket=ticket,
                action=str(f_row["action_name"]),
                fill_time=pd.to_datetime(f_row["logged_at"]),
                close_time=pd.to_datetime(c_row["logged_at"]),
                entry_price=float(f_row["entry_price"]),
                close_price=float(c_row["price"]),
                sl_price=float(c_row["initial_sl_price"]),
                tp_price=float(c_row["initial_tp_price"]),
                profit=float(c_row["profit"]),
                close_reason=str(c_row["close_reason"]),
                volume=float(f_row["volume"]),
            )
        )
    return trades


def attach_market_features(trades: List[TradeRecord], m1_df: pd.DataFrame) -> List[TradeRecord]:
    """Attaches lookahead-free H1 and M5 indicators at the exact fill timestamp."""
    if not trades:
        return trades
    tdf = pd.DataFrame([asdict(t) for t in trades])
    tdf = tdf.drop(columns=["h1_macro_bull", "h1_macro_bear", "m5_mom_bull", "m5_mom_bear"], errors="ignore")
    tdf["fill_time"] = pd.to_datetime(tdf["fill_time"])
    tdf = pd.merge_asof(
        tdf.sort_values("fill_time"),
        m1_df[["datetime", "h1_macro_bull", "h1_macro_bear", "m5_mom_bull", "m5_mom_bear"]],
        left_on="fill_time",
        right_on="datetime",
        direction="backward",
    )
    enriched: List[TradeRecord] = []
    for _, row in tdf.iterrows():
        enriched.append(
            TradeRecord(
                model=row["model"],
                ticket=row["ticket"],
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
                m5_mom_bull=bool(row["m5_mom_bull"]),
                m5_mom_bear=bool(row["m5_mom_bear"]),
            )
        )
    return enriched


def compute_metrics(
    trades: List[TradeRecord], candidate_name: str, window_name: str, initial_balance: float = 10000.0
) -> PerformanceMetrics:
    """Calculates comprehensive trading performance and risk metrics."""
    if not trades:
        return PerformanceMetrics(
            candidate_name=candidate_name,
            window_name=window_name,
            total_trades=0,
            net_pnl=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            sl_count=0,
            tp_count=0,
            timeout_count=0,
            max_dd_pct=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            payoff_ratio=0.0,
        )

    trades_sorted = sorted(trades, key=lambda x: x.close_time)
    pnl_series = [t.profit for t in trades_sorted]
    net_pnl = sum(pnl_series)
    wins = [p for p in pnl_series if p > 0]
    losses = [p for p in pnl_series if p < 0]

    total_trades = len(trades_sorted)
    win_rate = (len(wins) / total_trades) * 100.0 if total_trades > 0 else 0.0

    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

    avg_win = (gross_profit / len(wins)) if wins else 0.0
    avg_loss = (gross_loss / len(losses)) if losses else 0.0
    payoff_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0.0

    sl_count = sum(1 for t in trades_sorted if t.close_reason == "SL")
    tp_count = sum(1 for t in trades_sorted if t.close_reason == "TP")
    timeout_count = sum(1 for t in trades_sorted if t.close_reason == "EXPERT")

    # Equity Curve & Max Drawdown
    equity = initial_balance
    peak = initial_balance
    max_dd = 0.0
    for p in pnl_series:
        equity += p
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100.0
        if dd > max_dd:
            max_dd = dd

    return PerformanceMetrics(
        candidate_name=candidate_name,
        window_name=window_name,
        total_trades=total_trades,
        net_pnl=round(net_pnl, 2),
        win_rate=round(win_rate, 2),
        profit_factor=round(profit_factor, 2),
        sl_count=sl_count,
        tp_count=tp_count,
        timeout_count=timeout_count,
        max_dd_pct=round(max_dd, 2),
        gross_profit=round(gross_profit, 2),
        gross_loss=round(gross_loss, 2),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        payoff_ratio=round(payoff_ratio, 2),
    )


def simulate_circuit_breaker(
    input_trades: List[TradeRecord], max_consecutive_sl: int = 2, cooldown_minutes: int = 60
) -> List[TradeRecord]:
    """Option B: Consecutive Loss Circuit Breaker.
    Silences a model for `cooldown_minutes` if it encounters `max_consecutive_sl` within 60 minutes.
    """
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
                continue  # Suppressed by circuit breaker cooldown
            allowed_trades.append(t)
        elif ev_type == "CLOSE":
            if t in allowed_trades:
                if t.close_reason == "SL":
                    sl_history[m] = [tm for tm in sl_history[m] if (ev_time - tm).total_seconds() <= 3600]
                    sl_history[m].append(ev_time)
                    if len(sl_history[m]) >= max_consecutive_sl:
                        cooldowns[m] = ev_time + pd.Timedelta(minutes=cooldown_minutes)
                        sl_history[m] = []  # Reset counter upon trip
                elif t.close_reason == "TP" or t.profit > 0:
                    sl_history[m] = []  # Win resets consecutive loss streak
    return allowed_trades


def simulate_momentum_gate(input_trades: List[TradeRecord]) -> List[TradeRecord]:
    """Option C: Microstructure Order-Type & Momentum Gate.
    Vetoes counter-trend limit orders placed into strong adverse momentum.
    """
    allowed: List[TradeRecord] = []
    for t in input_trades:
        if t.action == "SELL_LIMIT" and t.m5_mom_bull:
            continue
        if t.action == "BUY_LIMIT" and t.m5_mom_bear:
            continue
        allowed.append(t)
    return allowed


def simulate_bayesian_trailing_decay(
    input_trades: List[TradeRecord], lam: float = 0.0065, active_threshold: float = 0.40
) -> List[TradeRecord]:
    """Option D: Online Bayesian / Trailing Performance Decay.
    Tracks trailing 60-minute losses and applies exponential weight decay.
    """
    events: List[Tuple[str, pd.Timestamp, TradeRecord]] = []
    for t in input_trades:
        events.append(("FILL", t.fill_time, t))
        events.append(("CLOSE", t.close_time, t))
    events.sort(key=lambda x: x[1])

    trailing_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {"BEAR": [], "BULL": []}
    allowed: List[TradeRecord] = []

    for ev_type, ev_time, t in events:
        m = t.model
        # Prune transactions older than 60 minutes
        trailing_history["BEAR"] = [x for x in trailing_history["BEAR"] if (ev_time - x[0]).total_seconds() <= 3600]
        trailing_history["BULL"] = [x for x in trailing_history["BULL"] if (ev_time - x[0]).total_seconds() <= 3600]

        if ev_type == "FILL":
            bear_loss = sum(abs(x[1]) for x in trailing_history["BEAR"] if x[1] < 0)
            bull_loss = sum(abs(x[1]) for x in trailing_history["BULL"] if x[1] < 0)
            w_bear = float(np.exp(-lam * bear_loss))
            w_bull = float(np.exp(-lam * bull_loss))
            tot_w = w_bear + w_bull
            p_bear = w_bear / tot_w if tot_w > 0 else 0.5
            p_bull = w_bull / tot_w if tot_w > 0 else 0.5

            if m == "BEAR" and p_bear >= active_threshold:
                allowed.append(t)
            elif m == "BULL" and p_bull >= active_threshold:
                allowed.append(t)
        elif ev_type == "CLOSE":
            if t in allowed:
                trailing_history[m].append((ev_time, t.profit))
    return allowed


def simulate_htf_macro_filter(input_trades: List[TradeRecord]) -> List[TradeRecord]:
    """Option E: Higher-Timeframe Trend Invariance Filter (H1 EMA21 vs EMA50).
    Permits only Bull specialist when H1 is Bullish; permits only Bear specialist when H1 is Bearish.
    """
    allowed: List[TradeRecord] = []
    for t in input_trades:
        if t.h1_macro_bull and t.model == "BEAR":
            continue
        if t.h1_macro_bear and t.model == "BULL":
            continue
        allowed.append(t)
    return allowed


def run_benchmark_matrix() -> Tuple[Dict[str, Dict[str, PerformanceMetrics]], pd.DataFrame]:
    """Executes the full comparative empirical simulation matrix across all candidates and windows."""
    m1_df = load_market_indicators()

    # Ground-truth execution paths
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

    print("[Trace] Ingesting ground-truth execution ledgers...")
    bear_w1 = attach_market_features(load_execution_ledger(bear_w1_path, "BEAR"), m1_df)
    bear_w2 = attach_market_features(load_execution_ledger(bear_w2_path, "BEAR"), m1_df)
    bull_w1 = attach_market_features(load_execution_ledger(bull_w1_path, "BULL"), m1_df)
    bull_w2 = attach_market_features(load_execution_ledger(bull_w2_path, "BULL"), m1_df)

    windows = [
        ("Window 1 (Uptrend)", bear_w1, bull_w1),
        ("Window 2 (Downtrend)", bear_w2, bull_w2),
    ]

    candidate_names = [
        "Baseline Bear (Phase 2 Champion)",
        "Baseline Bull (Phase 3 Round 01)",
        "Option A: Dual-Independent Portfolio",
        "Option B: Consecutive Loss Circuit Breaker",
        "Option C: Microstructure Order-Type Gate",
        "Option D: Online Bayesian / Trailing Decay",
        "Option E: HTF Trend Invariance Filter",
        "Hybrid 1: Option E + Option B",
        "Hybrid 2: Option A + Option B + Option C",
        "Hybrid 3: Option E + Option C + Option B",
    ]

    matrix_results: Dict[str, Dict[str, PerformanceMetrics]] = {
        "Window 1 (Uptrend)": {},
        "Window 2 (Downtrend)": {},
    }

    summary_rows: List[Dict[str, Any]] = []

    for win_label, bear_trades, bull_trades in windows:
        all_trades = sorted(bear_trades + bull_trades, key=lambda x: x.fill_time)

        # 1. Baseline Bear
        m_bear = compute_metrics(bear_trades, candidate_names[0], win_label)

        # 2. Baseline Bull
        m_bull = compute_metrics(bull_trades, candidate_names[1], win_label)

        # 3. Option A (Dual Independent)
        m_opt_a = compute_metrics(all_trades, candidate_names[2], win_label)

        # 4. Option B (Circuit Breaker)
        opt_b_trades = simulate_circuit_breaker(all_trades, max_consecutive_sl=2, cooldown_minutes=60)
        m_opt_b = compute_metrics(opt_b_trades, candidate_names[3], win_label)

        # 5. Option C (Momentum Gate)
        opt_c_trades = simulate_momentum_gate(all_trades)
        m_opt_c = compute_metrics(opt_c_trades, candidate_names[4], win_label)

        # 6. Option D (Bayesian Trailing Decay)
        opt_d_trades = simulate_bayesian_trailing_decay(all_trades, lam=0.0065, active_threshold=0.40)
        m_opt_d = compute_metrics(opt_d_trades, candidate_names[5], win_label)

        # 7. Option E (HTF Trend Filter)
        opt_e_trades = simulate_htf_macro_filter(all_trades)
        m_opt_e = compute_metrics(opt_e_trades, candidate_names[6], win_label)

        # 8. Hybrid 1 (Option E + Option B: HTF Filter + Circuit Breaker)
        hybrid_1_trades = simulate_circuit_breaker(opt_e_trades, max_consecutive_sl=2, cooldown_minutes=60)
        m_hyb_1 = compute_metrics(hybrid_1_trades, candidate_names[7], win_label)

        # 9. Hybrid 2 (Option A + Option B + Option C: Dual Indep + Breaker + Micro Gate)
        hybrid_2_trades = simulate_circuit_breaker(opt_c_trades, max_consecutive_sl=2, cooldown_minutes=60)
        m_hyb_2 = compute_metrics(hybrid_2_trades, candidate_names[8], win_label)

        # 10. Hybrid 3 (Option E + Option C + Option B: Full-Stack Multi-Tier)
        opt_ec_trades = [
            t
            for t in opt_c_trades
            if not (t.h1_macro_bull and t.model == "BEAR") and not (t.h1_macro_bear and t.model == "BULL")
        ]
        hybrid_3_trades = simulate_circuit_breaker(opt_ec_trades, max_consecutive_sl=2, cooldown_minutes=60)
        m_hyb_3 = compute_metrics(hybrid_3_trades, candidate_names[9], win_label)

        res_dict = {
            candidate_names[0]: m_bear,
            candidate_names[1]: m_bull,
            candidate_names[2]: m_opt_a,
            candidate_names[3]: m_opt_b,
            candidate_names[4]: m_opt_c,
            candidate_names[5]: m_opt_d,
            candidate_names[6]: m_opt_e,
            candidate_names[7]: m_hyb_1,
            candidate_names[8]: m_hyb_2,
            candidate_names[9]: m_hyb_3,
        }
        matrix_results[win_label] = res_dict

    # Build Comparative Synthesis Rows
    w1 = matrix_results["Window 1 (Uptrend)"]
    w2 = matrix_results["Window 2 (Downtrend)"]

    for name in candidate_names:
        r1 = w1[name]
        r2 = w2[name]
        total_pnl = round(r1.net_pnl + r2.net_pnl, 2)
        total_sl = r1.sl_count + r2.sl_count
        total_tp = r1.tp_count + r2.tp_count
        total_trades = r1.total_trades + r2.total_trades
        overall_wr = round(((r1.tp_count + r2.tp_count + (r1.total_trades - r1.sl_count - r1.tp_count) * (r1.win_rate/100 if r1.total_trades else 0)) / total_trades) * 100, 2) if total_trades else 0.0
        # Precise combined WR
        combined_wins = (r1.win_rate / 100.0 * r1.total_trades) + (r2.win_rate / 100.0 * r2.total_trades)
        combined_wr = round((combined_wins / total_trades) * 100.0, 2) if total_trades else 0.0
        combined_loss = r1.gross_loss + r2.gross_loss
        combined_profit = r1.gross_profit + r2.gross_profit
        combined_pf = round(combined_profit / combined_loss, 2) if combined_loss > 0 else (999.0 if combined_profit > 0 else 0.0)
        worst_dd = max(r1.max_dd_pct, r2.max_dd_pct)

        summary_rows.append(
            {
                "Candidate Architecture": name,
                "Window 1 Net PnL ($)": r1.net_pnl,
                "Win 1 WR (%)": r1.win_rate,
                "Win 1 PF": r1.profit_factor,
                "Win 1 SL": r1.sl_count,
                "Win 1 TP": r1.tp_count,
                "Win 1 Timeouts": r1.timeout_count,
                "Win 1 Max DD (%)": r1.max_dd_pct,
                "Window 2 Net PnL ($)": r2.net_pnl,
                "Win 2 WR (%)": r2.win_rate,
                "Win 2 PF": r2.profit_factor,
                "Win 2 SL": r2.sl_count,
                "Win 2 TP": r2.tp_count,
                "Win 2 Timeouts": r2.timeout_count,
                "Win 2 Max DD (%)": r2.max_dd_pct,
                "Total Net PnL ($)": total_pnl,
                "Combined WR (%)": combined_wr,
                "Combined PF": combined_pf,
                "Total SL": total_sl,
                "Total TP": total_tp,
                "Worst Max DD (%)": worst_dd,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    return matrix_results, summary_df


def print_matrix_table(summary_df: pd.DataFrame) -> None:
    """Prints a formatted ASCII benchmark table."""
    print("\n" + "=" * 148)
    print(
        f"{'Candidate Architecture':<42} | {'Win 1 PnL':<11} | {'W1 WR%':<7} | {'W1 SL':<6} | {'Win 2 PnL':<11} | {'W2 WR%':<7} | {'W2 SL':<6} | {'Total PnL':<12} | {'Comb PF':<7} | {'Worst DD%':<9}"
    )
    print("=" * 148)
    for _, row in summary_df.iterrows():
        p1 = f"${row['Window 1 Net PnL ($)']:,.2f}"
        p2 = f"${row['Window 2 Net PnL ($)']:,.2f}"
        ptot = f"${row['Total Net PnL ($)']:,.2f}"
        print(
            f"{row['Candidate Architecture']:<42} | {p1:<11} | {row['Win 1 WR (%)']:<7.1f} | {row['Win 1 SL']:<6} | {p2:<11} | {row['Win 2 WR (%)']:<7.1f} | {row['Win 2 SL']:<6} | {ptot:<12} | {row['Combined PF']:<7.2f} | {row['Worst Max DD (%)']:<9.2f}%"
        )
    print("=" * 148 + "\n")


def save_artifacts(
    matrix_results: Dict[str, Dict[str, PerformanceMetrics]],
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Saves benchmark results to JSON and CSV formats."""
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "simulation_benchmark_matrix.json"
    csv_path = output_dir / "simulation_benchmark_matrix.csv"

    # Prepare JSON structure
    serializable: Dict[str, Any] = {
        "metadata": {
            "timestamp": pd.Timestamp.now().isoformat(),
            "symbol": "BTCUSD M1",
            "window_1": "2025.02.20 - 2025.02.24 (Bull Wave)",
            "window_2": "2025.02.24 - 2025.02.28 (Bear Wave)",
            "initial_balance": 10000.0,
            "risk_geometry": "SL = 5.0 ATR, TP = 10.0 ATR (RR = 1:2), Max Holding = 120 bars",
        },
        "windows": {},
        "summary": summary_df.to_dict(orient="records"),
    }

    for win_label, cands in matrix_results.items():
        serializable["windows"][win_label] = {c_name: asdict(metrics) for c_name, metrics in cands.items()}

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)
    print(f"[Artifact] Saved machine-readable benchmark JSON to {json_path}")

    summary_df.to_csv(csv_path, index=False)
    print(f"[Artifact] Saved benchmark synthesis CSV to {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate candidate benchmark matrix across Window 1 and Window 2.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(ROOT / "outputs/candidate_benchmarks"),
        help="Directory to save JSON and CSV artifacts",
    )
    args = parser.parse_args()

    print("================================================================================")
    print("  SELF GROWING TRADER V193 — CANDIDATE ARCHITECTURE SIMULATION BENCHMARK MATRIX")
    print("================================================================================")

    matrix_results, summary_df = run_benchmark_matrix()
    print_matrix_table(summary_df)
    save_artifacts(matrix_results, summary_df, Path(args.output_dir))


if __name__ == "__main__":
    main()
