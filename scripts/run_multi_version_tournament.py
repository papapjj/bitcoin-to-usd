#!/usr/bin/env python3
"""
scripts/run_multi_version_tournament.py

Multi-Version MT5 Strategy Tester Arena & League Tournament.
Executes head-to-head evaluation across 5 distinct generations and archetypes
on BTCUSDm M1 under identical market conditions.
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

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    evaluate_policy_mt5_oos,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_execution_contract_kwargs,
)
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.mt5_runner import (
    MT5StrategyTesterEvaluator,
    MT5_MODEL_1MIN_OHLC,
)
from self_growing_trader.isolated_mt5_runner import isolated_mt5_runner_context
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiVersionTournament")


def extract_metrics(result: dict, start_date: str, end_date: str) -> dict:
    best = result.get("best", {})
    metrics = best.get("metrics", {})
    fitness = float(best.get("fitness", 0.0) or 0.0)

    try:
        d0 = datetime.strptime(start_date, "%Y.%m.%d")
        d1 = datetime.strptime(end_date, "%Y.%m.%d")
        total_days = max(1.0, float((d1 - d0).total_seconds() / 86400.0))
    except Exception:
        total_days = 2.0

    total_trades = int(metrics.get("trade_count", 0) or metrics.get("total_trades", 0) or 0)
    trades_per_day = round(total_trades / total_days, 1)

    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    if win_rate <= 1.0 and win_rate > 0.0:
        win_rate *= 100.0

    pf = float(metrics.get("profit_factor", 0.0) or 0.0)
    net_pnl = float(metrics.get("net_profit", 0.0) or metrics.get("profit", 0.0) or 0.0)
    max_dd = float(metrics.get("max_drawdown_pct", 0.0) or metrics.get("drawdown", 0.0) or 0.0)
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
        "report_path": metrics.get("report_path", ""),
    }


def print_league_table(results_table: list[dict]):
    print("\n" + "=" * 105, flush=True)
    print("                BANG TONG SAP GIAI DAU DA THE HE (MULTI-VERSION LEAGUE ARENA)", flush=True)
    print("=" * 105, flush=True)
    header = (
        f"{'Hang':<5} | {'The He / Phien Ban':<30} | {'Trades':<7} | {'Lenh/Ngay':<9} | "
        f"{'Win Rate':<9} | {'PF':<6} | {'Net PnL':<10} | {'Max DD':<8} | {'Diem Elo':<8}"
    )
    print(header, flush=True)
    print("-" * 105, flush=True)

    # Sort by Net PnL descending, then by Max DD ascending
    sorted_res = sorted(results_table, key=lambda x: (x["net_pnl"], -x["max_drawdown"]), reverse=True)

    for rank, entry in enumerate(sorted_res, 1):
        pnl_str = f"${entry['net_pnl']:+,.2f}"
        dd_str = f"{entry['max_drawdown']:.2f}%"
        wr_str = f"{entry['win_rate']:.1f}%"
        elo_score = round(1000.0 + entry["net_pnl"] * 2.0 + entry["win_rate"] * 5.0 - entry["max_drawdown"] * 15.0, 1)
        row = (
            f"{rank:<5} | {entry['name']:<30} | {entry['total_trades']:<7} | "
            f"{entry['trades_per_day']:<9} | {wr_str:<9} | {entry['profit_factor']:<6.2f} | "
            f"{pnl_str:<10} | {dd_str:<8} | {elo_score:<8.1f}"
        )
        print(row, flush=True)
    print("=" * 105 + "\n", flush=True)


def build_frontier_mythos_policy(config, checkpoint_path: Path, packets: int = 1440) -> MythosMT5Policy:
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
    _cau_hinh_policy_chuyen_gia_cho_mt5(base_policy, config)

    mythos_policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
    mythos_policy.max_packets = packets
    mythos_policy.conviction_threshold = 0.035
    mythos_policy.require_margin_alignment = True
    mythos_policy.enable_mtf_filter = True
    mythos_policy.counter_trap_sl_atr = 2.0
    mythos_policy.counter_trap_tp_rr = 2.50
    mythos_policy.sniper_sl_atr = 2.0
    mythos_policy.sniper_tp_rr = 2.50

    # Inject Triad Self-Play Champion weights if available, else MuZero Champion
    triad_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_selfplay_triad_champion.pt"
    champion_pt = triad_pt if triad_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt")
    if champion_pt.exists():
        ckpt = torch.load(champion_pt, map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" in ckpt:
            mythos_policy.pipeline.cot_engine.load_state_dict(
                ckpt["reasoning_engine_state_dict"], strict=True
            )
            print(f"      [Frontier Mythos] Injected {champion_pt.name} (Triad Self-Play verified).", flush=True)

    return mythos_policy


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Version League Tournament on MT5")
    parser.add_argument("--from-date", default="2025.01.06", help="Tournament start date")
    parser.add_argument("--to-date", default="2025.01.08", help="Tournament end date")
    parser.add_argument("--visual", action="store_true", help="Launch visual MT5 window")
    parser.add_argument("--packets", type=int, default=1440, help="Max packets per match")
    parser.add_argument("--versions", nargs="+", default=["gen1", "gen2", "challenger"],
                        choices=["gen1", "gen2", "gen3", "dual_router", "challenger"],
                        help="List of versions to enter into the tournament")
    args = parser.parse_args()

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    if not config_path.exists():
        config_path = ROOT / "configs" / "he_chuyen_gia_mot_context_mot_policy.yaml"
    config = load_config(config_path)

    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    v193_dir = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs"

    print("=" * 105, flush=True)
    print("               KHOI TRANH TAI GIAI DAU DA THE HE (METATRADER 5 LEAGUE ARENA)", flush=True)
    print(f"               Thoi gian: {args.from_date} -> {args.to_date} | Cap: BTCUSDm M1", flush=True)
    print("=" * 105, flush=True)

    league_results = []

    for v_key in args.versions:
        print(f"\n>>> [TRAN DAU]: Khoi dong phien ban '{v_key}'...", flush=True)
        session_id = f"tourn_{v_key}_{int(time.time())}"

        if v_key == "gen1":
            p_path = v193_dir / "best_promoted_checkpoint.pkl"
            if not p_path.exists():
                print(f"      Bo qua {v_key}: khong tim thay file {p_path.name}")
                continue
            pol = tai_he_chuyen_gia_mot_context_mot_policy(p_path, config, seed=2026)
            name = "Gen-1 (Phase 1 Defensive Champion)"

        elif v_key == "gen2":
            p_path = v193_dir / "best_promoted_checkpoint_phase2.pkl"
            if not p_path.exists():
                print(f"      Bo qua {v_key}: khong tim thay file {p_path.name}")
                continue
            pol = tai_he_chuyen_gia_mot_context_mot_policy(p_path, config, seed=2026)
            name = "Gen-2 (Phase 2 Bear Specialist)"

        elif v_key == "gen3":
            p_path = v193_dir / "champion_round_5.pkl"
            if not p_path.exists():
                print(f"      Bo qua {v_key}: khong tim thay file {p_path.name}")
                continue
            pol = tai_he_chuyen_gia_mot_context_mot_policy(p_path, config, seed=2026)
            name = "Gen-3 (Champion Round 5)"

        elif v_key == "dual_router":
            bear_path = v193_dir / "best_promoted_checkpoint_phase2.pkl"
            bull_path = v193_dir / "multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl"
            if not bear_path.exists() or not bull_path.exists():
                print(f"      Bo qua {v_key}: thieu checkpoint thanh phan")
                continue
            pol = load_dual_champion_router(
                bear_path=bear_path,
                bull_path=bull_path,
                detector_path=regime_model_path,
                bull_threshold=0.35,
                bear_threshold=0.35,
                chop_threshold=0.38,
                micro_scalp_enabled=True,
            )
            name = "DualChampionRouter (Bear/Bull Hybrid)"

        elif v_key == "challenger":
            base_cp = v193_dir / "best_promoted_checkpoint_phase2.pkl"
            pol = build_frontier_mythos_policy(config, base_cp, packets=args.packets)
            name = "Gen-5 (Frontier Mythos Challenger)"
        else:
            continue

        print(f"      Tuyen thu: {name}", flush=True)
        t0 = time.time()
        with isolated_mt5_runner_context(
            session_name=session_id,
            symbol="BTCUSDm",
            from_date=args.from_date,
            to_date=args.to_date,
        ) as runner_dir:
            res = evaluate_policy_mt5_oos(
                pol,
                config,
                runner_dir=runner_dir,
                timeout_sec=1200,
                visual=bool(args.visual),
                shutdown_terminal=True,
                from_date=args.from_date,
                to_date=args.to_date,
                model=MT5_MODEL_1MIN_OHLC,
            )
        elapsed = time.time() - t0
        metrics = extract_metrics(res, args.from_date, args.to_date)
        metrics["name"] = name
        metrics["elapsed_sec"] = round(elapsed, 1)
        league_results.append(metrics)
        print(f"      Ket qua: Trades: {metrics['total_trades']} | WR: {metrics['win_rate']}% | PnL: ${metrics['net_pnl']} | DD: {metrics['max_drawdown']}% (Xong trong {elapsed:.1f}s)", flush=True)

    # In bang tong sap
    print_league_table(league_results)

    # Luu ket qua tournament
    out_dir = ROOT / "outputs" / "tournaments"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"multi_version_league_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "from_date": args.from_date, "to_date": args.to_date, "results": league_results}, f, indent=2)
    print(f"[HOAN TAT] Da luu bien ban thi dau tai: {out_file}", flush=True)


if __name__ == "__main__":
    main()
