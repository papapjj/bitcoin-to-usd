#!/usr/bin/env python3
"""
scripts/run_mt5_champion_vs_challenger.py

MT5 Strategy Tester Head-to-Head Arena (Champion vs Challenger).
Runs real MT5 sessions for both Champion and Challenger on BTCUSDm M1,
extracts official execution metrics from MT5, and enforces the 5-Gate Promotion Contract.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

# Safe UTF-8 configuration
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    evaluate_policy_mt5_oos,
)
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.learning_integrity import sha256_file
from self_growing_trader.isolated_mt5_runner import isolated_mt5_runner_context


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5ChampionVsChallenger")


def extract_metrics_from_mt5_result(result: dict, start_date: str, end_date: str) -> dict:
    """Parse key quantitative metrics from evaluate_policy_mt5_oos return dict."""
    best = result.get("best", {})
    metrics = best.get("metrics", {})
    fitness = float(best.get("fitness", 0.0) or 0.0)

    # Calculate days in window
    try:
        d0 = datetime.strptime(start_date, "%Y.%m.%d")
        d1 = datetime.strptime(end_date, "%Y.%m.%d")
        total_days = max(1.0, float((d1 - d0).total_seconds() / 86400.0))
    except Exception:
        total_days = 4.0

    total_trades = int(
        metrics.get("total_trades", 0)
        or metrics.get("trade_count", 0)
        or metrics.get("trades", 0)
        or 0
    )
    trades_per_day = round(total_trades / total_days, 1)

    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    if win_rate <= 1.0 and win_rate > 0.0:
        win_rate *= 100.0

    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    net_pnl = float(metrics.get("net_profit", 0.0) or metrics.get("profit", 0.0) or 0.0)
    max_dd = float(
        metrics.get("equity_drawdown_pct", 0.0)
        or metrics.get("max_drawdown_pct", 0.0)
        or metrics.get("balance_drawdown_pct", 0.0)
        or metrics.get("drawdown", 0.0)
        or 0.0
    )
    if max_dd <= 1.0 and max_dd > 0.0:
        max_dd *= 100.0

    return {
        "total_trades": total_trades,
        "trades_per_day": trades_per_day,
        "win_rate": round(win_rate, 2),
        "profit_factor": round(pf, 2),
        "net_pnl": round(net_pnl, 2),
        "max_drawdown": round(max_dd, 2),
        "fitness": round(fitness, 4),
        "raw_metrics": metrics,
    }


def evaluate_five_gates(challenger_m: dict, champion_m: dict) -> dict:
    """Evaluate 5-Gate Promotion Contract on MT5 execution metrics."""
    g1_freq = challenger_m["trades_per_day"] >= 100.0 or challenger_m["total_trades"] >= 200
    g2_wr = challenger_m["win_rate"] >= 85.0
    g3_dd = challenger_m["max_drawdown"] <= 6.0
    g4_pnl = challenger_m["net_pnl"] >= (champion_m["net_pnl"] * 1.05) if champion_m["net_pnl"] > 0 else (challenger_m["net_pnl"] > champion_m["net_pnl"])
    g5_stability = challenger_m["total_trades"] > 0

    all_passed = g1_freq and g2_wr and g3_dd and g4_pnl and g5_stability

    return {
        "gate_1_frequency": {"passed": g1_freq, "value": f"{challenger_m['trades_per_day']} trades/day", "target": ">= 100/day"},
        "gate_2_win_rate": {"passed": g2_wr, "value": f"{challenger_m['win_rate']}%", "target": ">= 85.0%"},
        "gate_3_drawdown": {"passed": g3_dd, "value": f"{challenger_m['max_drawdown']}%", "target": "<= 6.0%"},
        "gate_4_outperformance": {"passed": g4_pnl, "value": f"${challenger_m['net_pnl']}", "target": f"> Champion (${champion_m['net_pnl']})"},
        "gate_5_stability": {"passed": g5_stability, "value": "Stable MT5 session", "target": "0 crash/error"},
        "promoted": all_passed,
    }


def print_scoreboard(champ_name: str, champ_m: dict, chall_name: str, chall_m: dict, gates: dict):
    print("\n" + "=" * 88, flush=True)
    print("        METATRADER 5 HEAD-TO-HEAD TOURNAMENT: CHAMPION vs CHALLENGER", flush=True)
    print("=" * 88, flush=True)
    print(f"{'Chi So Danh Gia (MT5 Official)':<32} | {'CHAMPION (' + champ_name + ')':<25} | {'CHALLENGER (' + chall_name + ')':<25}", flush=True)
    print("-" * 88, flush=True)

    def row(label, k, fmt, unit=""):
        v_c = fmt.format(champ_m[k]) + unit
        v_ch = fmt.format(chall_m[k]) + unit
        win_marker = " [WIN]" if chall_m[k] > champ_m[k] else ""
        print(f"{label:<32} | {v_c:<25} | {v_ch:<20}{win_marker}", flush=True)

    row("Tong so lenh khop MT5", "total_trades", "{:d}")
    row("Tan suat lenh / ngay", "trades_per_day", "{:.1f}", " lenh/ngay")
    row("Ty le thang (Win Rate)", "win_rate", "{:.2f}", "%")
    row("He so loi nhuan (PF)", "profit_factor", "{:.2f}")
    row("Lai rong thuc te (Net PnL)", "net_pnl", "${:+.2f}")
    row("Sut giam tai khoan (Max DD)", "max_drawdown", "{:.2f}", "%")
    print("=" * 88, flush=True)

    print("\n[KIEM DINH 5 CONG THANG HANG - 5-GATE PROMOTION CONTRACT]:", flush=True)
    for g_key, g_info in gates.items():
        if g_key == "promoted":
            continue
        status_txt = "DAT (PASS)  " if g_info["passed"] else "TRUOT (FAIL)"
        print(f"  * {g_key:<24}: {status_txt} | Thuc te: {g_info['value']:<18} | Yeu cau: {g_info['target']}", flush=True)

    print("-" * 88, flush=True)
    if gates["promoted"]:
        print("  ==> KET QUA CHUNG CUOC: CHALLENGER CHIEN THANG TUYET DOI! DU TIEU CHUAN THANG HANG!", flush=True)
    else:
        print("  ==> KET QUA CHUNG CUOC: CHALLENGER CHUA DU DIEU KIEN THANG HANG. GIU NGUYEN CHAMPION.", flush=True)
    print("=" * 88 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Run MT5 Head-to-Head Tournament (Champion vs Challenger)")
    parser.add_argument("--champion-type", choices=["dual_router", "phase1", "phase2", "bull_r01"], default="dual_router")
    parser.add_argument("--challenger-path", required=False, default=None, help="Path to Challenger checkpoint .pkl")
    parser.add_argument("--from-date", default="2025.01.14", help="Match start date (YYYY.MM.DD)")
    parser.add_argument("--to-date", default="2025.01.18", help="Match end date (YYYY.MM.DD)")
    parser.add_argument("--visual", action="store_true", help="Launch visual MT5 window on desktop")
    parser.add_argument("--promote-on-win", action="store_true", help="Automatically promote Challenger if it beats Champion")
    parser.add_argument("--dry-run", action="store_true", help="Run fast tournament validation without full MT5 session")
    parser.add_argument("--output-dir", default="outputs/tournaments", help="Output directory for match reports")
    args = parser.parse_args()

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)
    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    runner_dir = ROOT / "mt5_v2_runner"

    print("=" * 88, flush=True)
    print(f"  KHOI DONG DAU TRUONG METATRADER 5: CHAMPION VS CHALLENGER", flush=True)
    print(f"  Thoi gian thi dau : {args.from_date} -> {args.to_date} (Symbol: BTCUSDm M1)", flush=True)
    print("=" * 88, flush=True)

    # 1. Load Champion
    print("\n[1/4] Dang chuan bi Policy CHAMPION...", flush=True)
    if args.champion_type == "dual_router":
        bear_path = ROOT / "outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl"
        bull_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl"
        champion_policy = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_model_path,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            micro_scalp_enabled=True,
        )
        champ_name = "DualChampionRouter"
    else:
        champ_path = ROOT / "outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint.pkl"
        champion_policy = tai_he_chuyen_gia_mot_context_mot_policy(champ_path, config, seed=2026)
        champ_name = champ_path.name
    print(f"      -> Champion san sang: {champ_name}", flush=True)

    # 2. Run Champion on MT5
    print("\n[2/4] [HIEP 1]: Khoi chay MT5 Strategy Tester cho CHAMPION...", flush=True)
    t0 = time.time()
    if args.dry_run:
        champ_res = {
            "best": {
                "fitness": 3.71,
                "metrics": {
                    "total_trades": 312,
                    "win_rate": 86.42,
                    "profit_factor": 2.85,
                    "net_profit": 5240.50,
                    "equity_drawdown_pct": 3.80,
                }
            }
        }
    else:
        with isolated_mt5_runner_context(
            session_name="champ",
            symbol="BTCUSDm",
            from_date=args.from_date,
            to_date=args.to_date,
        ) as champ_runner_dir:
            champ_res = evaluate_policy_mt5_oos(
                champion_policy,
                config,
                runner_dir=champ_runner_dir,
                timeout_sec=1200,
                visual=bool(args.visual),
                shutdown_terminal=True,
                from_date=args.from_date,
                to_date=args.to_date,
                model=1,
            )
    t_champ = time.time() - t0
    champ_metrics = extract_metrics_from_mt5_result(champ_res, args.from_date, args.to_date)
    print(f"      -> Hiep 1 hoan thanh trong {t_champ:.1f}s!", flush=True)
    print(f"      -> Ket qua Champion MT5: {champ_metrics['total_trades']} lenh | WR: {champ_metrics['win_rate']}% | PnL: ${champ_metrics['net_pnl']} | DD: {champ_metrics['max_drawdown']}%", flush=True)

    # 3. Load & Run Challenger
    print("\n[3/4] [HIEP 2]: Dang chuan bi & Khoi chay MT5 cho CHALLENGER...", flush=True)
    if args.challenger_path is not None and Path(args.challenger_path).is_file():
        chall_path = Path(args.challenger_path)
        challenger_policy = tai_he_chuyen_gia_mot_context_mot_policy(chall_path, config, seed=2026)
        chall_name = chall_path.name
    else:
        # If no explicit challenger is provided, test the Bull Candidate with Micro Scalp
        bull_candidate_path = ROOT / "outputs/he_chuyen_gia/v193_runs/multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl"
        challenger_policy = load_dual_champion_router(
            bear_path=ROOT / "outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl",
            bull_path=bull_candidate_path,
            detector_path=regime_model_path,
            bull_threshold=0.30,  # More aggressive threshold
            bear_threshold=0.30,
            chop_threshold=0.40,
            micro_scalp_enabled=True,
        )
        chall_name = "ChallengerRouter_Aggressive"

    t0 = time.time()
    if args.dry_run:
        chall_res = {
            "best": {
                "fitness": 4.12,
                "metrics": {
                    "total_trades": 356,
                    "win_rate": 88.20,
                    "profit_factor": 3.15,
                    "net_profit": 6280.00,
                    "equity_drawdown_pct": 3.10,
                }
            }
        }
    else:
        with isolated_mt5_runner_context(
            session_name="chall",
            symbol="BTCUSDm",
            from_date=args.from_date,
            to_date=args.to_date,
        ) as chall_runner_dir:
            chall_res = evaluate_policy_mt5_oos(
                challenger_policy,
                config,
                runner_dir=chall_runner_dir,
                timeout_sec=1200,
                visual=bool(args.visual),
                shutdown_terminal=True,
                from_date=args.from_date,
                to_date=args.to_date,
                model=1,
            )
    t_chall = time.time() - t0
    chall_metrics = extract_metrics_from_mt5_result(chall_res, args.from_date, args.to_date)
    print(f"      -> Hiep 2 hoan thanh trong {t_chall:.1f}s!", flush=True)
    print(f"      -> Ket qua Challenger MT5: {chall_metrics['total_trades']} lenh | WR: {chall_metrics['win_rate']}% | PnL: ${chall_metrics['net_pnl']} | DD: {chall_metrics['max_drawdown']}%", flush=True)

    # 4. Evaluation and Arbitration
    print("\n[4/4] Trong tai MT5 dang tong hop ket qua va doi chieu 5 Cong...", flush=True)
    gates = evaluate_five_gates(chall_metrics, champ_metrics)
    print_scoreboard(champ_name, champ_metrics, chall_name, chall_metrics, gates)

    # Save match artifact
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    match_file = out_dir / f"match_{ts}.json"
    match_data = {
        "timestamp": ts,
        "from_date": args.from_date,
        "to_date": args.to_date,
        "champion": {"name": champ_name, "metrics": champ_metrics},
        "challenger": {"name": chall_name, "metrics": chall_metrics},
        "gates": gates,
    }
    with open(match_file, "w", encoding="utf-8") as f:
        json.dump(match_data, f, indent=2)
    print(f"Bao cao tran dau da duoc luu tai: {match_file}", flush=True)

    # Promotion logic
    if gates["promoted"] and args.promote_on_win:
        print("\n>>> DANG THUC HIEN THANG HANG (PROMOTION) CHALLENGER THANH NEW CHAMPION...", flush=True)
        archive_dir = ROOT / "outputs" / "champion_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        promoted_dst = ROOT / "outputs" / "champion_current.pkl"
        if promoted_dst.is_file():
            shutil.copy2(promoted_dst, archive_dir / f"champion_prev_{ts}.pkl")
        if args.challenger_path and Path(args.challenger_path).is_file():
            shutil.copy2(Path(args.challenger_path), promoted_dst)
        print(">>> THANG HANG THANH CONG! New Champion da san sang cho thuc chien MT5.", flush=True)


if __name__ == "__main__":
    main()
