#!/usr/bin/env python3
"""
Automated MT5 Efficiency Observation & Optimization Loop.
Systematically benchmarks, diagnoses root causes, and tunes gating thresholds
(PRM Gate, MAD Trap Veto similarity, Circuit Breaker cooldown) across iterative MT5 runs.

Target Stopping Criteria:
  - Win Rate >= 80.0%
  - Profit Factor > 3.0
  - Max Drawdown <= 10.0%
  - Total Trades >= 15
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import pandas as pd
import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    evaluate_policy_mt5_oos,
)
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.cppi_risk_engine import CPPIConfig
from self_growing_trader.two_tier_runner import TwoTierConfig
from self_growing_trader.isolated_mt5_runner import isolated_mt5_runner_context
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MT5OptimizationLoop")


def parse_execution_ledger(ledger_path: Path) -> Dict[str, Any]:
    """Parse real executed trades from self_growing_execution.csv."""
    if not ledger_path.exists() or ledger_path.stat().st_size == 0:
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_profit": 0.0, "profit_factor": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "deals": []}

    try:
        df = pd.read_csv(ledger_path, sep="\t", header=0, low_memory=False)
        deals = df[(df["deal_ticket"] > 0) & (df["transaction_type"] == "DEAL_ADD")]
        closed_deals = deals[deals["status"] == "CLOSED"]
        if len(closed_deals) == 0:
            closed_deals = deals[(deals["profit"] != 0) | (deals["close_reason"].notna() & (deals["close_reason"] != ""))]

        total_trades = len(closed_deals)
        profits = [float(p) for p in closed_deals["profit"]] if "profit" in closed_deals else []
        wins = sum(1 for p in profits if p > 0)
        losses = sum(1 for p in profits if p < 0)
        gross_profit = sum(p for p in profits if p > 0)
        gross_loss = abs(sum(p for p in profits if p < 0))
        net_profit = sum(profits)
        win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (999.0 if gross_profit > 0 else 0.0)

        deals_summary = []
        for _, row in closed_deals.iterrows():
            deals_summary.append({
                "time": str(row.get("logged_at", "")),
                "action": str(row.get("action_name", "")),
                "volume": float(row.get("volume", 0)),
                "price": float(row.get("price", 0)),
                "profit": float(row.get("profit", 0)),
                "close_reason": str(row.get("close_reason", "")),
            })

        return {
            "total_trades": total_trades,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "deals": deals_summary,
        }
    except Exception as e:
        logger.warning(f"Error parsing execution ledger {ledger_path}: {e}")
        return {"total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "net_profit": 0.0, "profit_factor": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "deals": []}


def run_single_iteration(
    iter_name: str,
    policy: Any,
    config: Any,
    from_date: str = "2025.01.14",
    to_date: str = "2025.01.16",
    symbol: str = "BTCUSDm",
) -> Dict[str, Any]:
    """Execute a single MT5 Strategy Tester simulation and extract full metrics."""
    logger.info(f"=== Starting Iteration: {iter_name} ({from_date} -> {to_date}) ===")
    t_start = time.time()
    trace_p = ROOT / "outputs" / "mt5_optimization_loop" / "reasoning_trace.jsonl"
    if trace_p.exists():
        try:
            trace_p.unlink()
        except Exception:
            pass

    with isolated_mt5_runner_context(
        session_name=f"opt_{iter_name[:12]}",
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        shutdown_terminal=True,
        auto_cleanup=False,
    ) as runner_dir:
        result = evaluate_policy_mt5_oos(
            policy,
            config,
            runner_dir=runner_dir,
            timeout_sec=7200,
            visual=False,  # Headless for high-speed multi-round execution
            shutdown_terminal=True,
            from_date=from_date,
            to_date=to_date,
            model=1,
        )

        # Parse metrics from manifest
        best = result.get("best", {})
        metrics = best.get("metrics", {})

        # Parse real trades from execution ledger in runner
        exec_path = runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_execution.csv"
        if not exec_path.exists():
            exec_path = runner_dir / "MQL5" / "Files" / "self_growing_execution.csv"

        ledger_stats = parse_execution_ledger(exec_path)

    elapsed_sec = time.time() - t_start

    # Consolidate metrics
    total_trades = int(metrics.get("total_trades", ledger_stats["total_trades"]))
    net_profit = float(metrics.get("net_profit", ledger_stats["net_profit"]))
    profit_factor = float(metrics.get("profit_factor", ledger_stats["profit_factor"]))
    win_rate = float(metrics.get("win_rate", ledger_stats["win_rate"]))
    drawdown_pct = float(metrics.get("equity_drawdown_pct", 0.0))

    logger.info(
        f"Iteration [{iter_name}] Complete in {elapsed_sec:.1f}s | "
        f"Trades: {total_trades} | Net Profit: ${net_profit:.2f} | "
        f"PF: {profit_factor:.2f} | WR: {win_rate:.2f}% | Max DD: {drawdown_pct:.2f}%"
    )

    return {
        "iteration_name": iter_name,
        "elapsed_seconds": elapsed_sec,
        "total_trades": total_trades,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "win_rate": win_rate,
        "drawdown_pct": drawdown_pct,
        "ledger_stats": ledger_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="MT5 Efficiency Observation & Optimization Loop")
    parser.add_argument("--from-date", default="2025.01.10", help="Start date (YYYY.MM.DD) - 7 full days backtest")
    parser.add_argument("--to-date", default="2025.01.17", help="End date (YYYY.MM.DD)")
    parser.add_argument("--symbol", default="BTCUSDm", help="Market symbol")
    parser.add_argument("--iteration", type=int, default=24, help="Iteration step to run (24 for 7-Day Dual-Champion Convergence)")
    args = parser.parse_args()

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)

    bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    regime_detector = RegimeDetector.load(regime_model_path)

    out_dir = ROOT / "outputs" / "mt5_optimization_loop"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "loop_results.json"

    results: List[Dict[str, Any]] = []
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                results = json.load(f)
        except Exception:
            results = []

    if args.iteration == 3:
        logger.info("[Iter 3] Building Mythos Integrated Shield (Dual-Champion Signals + Kelly Sizing + ATR Trailing Stop)...")
        base_router_3 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
        )
        policy_3 = MythosMT5Policy(base_policy=base_router_3)
        policy_3.prm_threshold = 0.65
        policy_3.veto_similarity_threshold = 0.94
        if hasattr(policy_3, "mad"):
            policy_3.mad.veto_similarity_threshold = 0.94
        policy_3.loss_cooldown_bars = 4
        policy_3.max_consecutive_losses = 2
        policy_3.scalp_sl_atr = 2.0
        policy_3.scalp_tp_rr = 2.5
        policy_3.trailing_atr = 1.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_3.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)

        res_3 = run_single_iteration(
            iter_name="Iter_3_Mythos_Integrated_Shield",
            policy=policy_3,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )

        # Remove previous Iter_3 if present and append new
        results = [r for r in results if r.get("iteration_name") != "Iter_3_Mythos_Integrated_Shield"]
        results.append(res_3)

    elif args.iteration == 4:
        logger.info("[Iter 4] Running Dual-Champion Pure with Patched Safe Risk & ATR Trailing Stop...")
        safe_router = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
        )
        res_4 = run_single_iteration(
            iter_name="Iter_4_DualChampion_SafeShield",
            policy=safe_router,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_4_DualChampion_SafeShield"]
        results.append(res_4)

    elif args.iteration == 5:
        logger.info("[Iter 5] Running Dual-Champion with Fast Dynamic Cooldown (5m) & Safe Risk Shield...")
        safe_router_5 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=5.0,
            circuit_breaker_window_minutes=15.0,
        )
        res_5 = run_single_iteration(
            iter_name="Iter_5_DualChampion_OptimalCadence",
            policy=safe_router_5,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_5_DualChampion_OptimalCadence"]
        results.append(res_5)

    elif args.iteration == 6:
        logger.info("[Iter 6] Running Dual-Champion Full Flow Safe Shield...")
        safe_router_6 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            circuit_breaker_enabled=False,
        )
        res_6 = run_single_iteration(
            iter_name="Iter_6_DualChampion_FullSafeShield",
            policy=safe_router_6,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_6_DualChampion_FullSafeShield"]
        results.append(res_6)

    elif args.iteration == 7:
        logger.info("[Iter 7] Running Mythos with Clean Bar Cadence (No Wall-Clock Lock) & Safe Shield...")
        base_router_7 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            circuit_breaker_enabled=False,
        )
        policy_7 = MythosMT5Policy(base_policy=base_router_7)
        policy_7.prm_threshold = 0.70
        policy_7.veto_similarity_threshold = 0.93
        if hasattr(policy_7, "mad"):
            policy_7.mad.veto_similarity_threshold = 0.93
        policy_7.loss_cooldown_bars = 6
        policy_7.scalp_sl_atr = 2.0
        policy_7.scalp_tp_rr = 2.5
        policy_7.trailing_atr = 1.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_7.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)

        res_7 = run_single_iteration(
            iter_name="Iter_7_Mythos_TrueCadence_SafeShield",
            policy=policy_7,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_7_Mythos_TrueCadence_SafeShield"]
        results.append(res_7)

    elif args.iteration == 8:
        logger.info("[Iter 8] Running Pure Dual-Champion (Bull+Bear Specialists only, micro_scalp=False, safe CPPI + ATR trailing)...")
        safe_router_8 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
        )
        res_8 = run_single_iteration(
            iter_name="Iter_8_DualChampion_Pure_SafeShield",
            policy=safe_router_8,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_8_DualChampion_Pure_SafeShield"]
        results.append(res_8)

    elif args.iteration == 24:
        logger.info(
            f"[Iter 24] Running 7-Day Dual-Champion Full Convergence ({args.from_date} -> {args.to_date}) | "
            f"Max 5 Positions | SL 5.0 ATR | Trailing 1.0 ATR | Safe CPPI 0.015..."
        )
        safe_cppi_24 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        router_24 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=30.0,
            circuit_breaker_window_minutes=60.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            min_spacing_bars=3,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.0,
            cppi_config=safe_cppi_24,
        )
        res_24 = run_single_iteration(
            iter_name="Iter_24_7Days_TrueConvergence_DualChampion",
            policy=router_24,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_24_7Days_TrueConvergence_DualChampion"]
        results.append(res_24)

    elif args.iteration == 25:
        logger.info(
            f"[Iter 25] Running 7-Day Macro-Trend Shield + Directional Cluster Cap ({args.from_date} -> {args.to_date}) | "
            f"Max 5 Positions (Max 2/side) | Spacing 10 bars | Breakeven 2.0 ATR | Safe CPPI 0.015..."
        )
        safe_cppi_25 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_25 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=4.0,
            trailing_atr=1.5,
        )
        router_25 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=30.0,
            circuit_breaker_window_minutes=60.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=10,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.5,
            cppi_config=safe_cppi_25,
            two_tier_config=two_tier_25,
        )
        res_25 = run_single_iteration(
            iter_name="Iter_25_7Days_MacroTrend_ClusterShield",
            policy=router_25,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_25_7Days_MacroTrend_ClusterShield"]
        results.append(res_25)

    elif args.iteration == 26:
        logger.info(
            f"[Iter 26] Running 7-Day World-Class Mythos xLSTM 17.5M + CoT Reasoning + MuZero Champion ({args.from_date} -> {args.to_date}) | "
            f"PRM 0.70 | Microstructure Shield | Max 5 Positions (Max 2/side) | Spacing 10 bars | Safe CPPI 0.015..."
        )
        safe_cppi_26 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_26 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=4.0,
            trailing_atr=1.5,
        )
        base_router_26 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=30.0,
            circuit_breaker_window_minutes=60.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=10,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.0,
            cppi_config=safe_cppi_26,
            two_tier_config=two_tier_26,
        )
        policy_26 = MythosMT5Policy(base_policy=base_router_26)
        policy_26.prm_threshold = 0.70
        policy_26.veto_similarity_threshold = 0.93
        if hasattr(policy_26, "mad"):
            policy_26.mad.veto_similarity_threshold = 0.93
        policy_26.loss_cooldown_bars = 4
        policy_26.scalp_sl_atr = 2.0
        policy_26.scalp_tp_rr = 2.5
        policy_26.trailing_atr = 1.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_26.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 26] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_26.pipeline, "world_model"):
                policy_26.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 26] Successfully loaded Latent World Model from %s", champ_pt)

        res_26 = run_single_iteration(
            iter_name="Iter_26_7Days_Mythos_xLSTM_MuZero_CoT_Champion",
            policy=policy_26,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_26_7Days_Mythos_xLSTM_MuZero_CoT_Champion"]
        results.append(res_26)

    elif args.iteration == 27:
        logger.info(
            f"[Iter 27] Running 2-Day Bar-Cadence Mythos xLSTM + Macro Trend Lock + Anti-FOMO Shield ({args.from_date} -> {args.to_date}) | "
            f"PRM 0.70 | Max 5 Positions (Max 2/side) | Spacing 10 bars | Safe CPPI 0.015..."
        )
        safe_cppi_27 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_27 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=4.0,
            trailing_atr=1.5,
        )
        base_router_27 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=10,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.0,
            cppi_config=safe_cppi_27,
            two_tier_config=two_tier_27,
        )
        policy_27 = MythosMT5Policy(base_policy=base_router_27)
        policy_27.prm_threshold = 0.70
        policy_27.veto_similarity_threshold = 0.93
        if hasattr(policy_27, "mad"):
            policy_27.mad.veto_similarity_threshold = 0.93
        policy_27.loss_cooldown_bars = 10
        policy_27.scalp_sl_atr = 2.0
        policy_27.scalp_tp_rr = 2.5
        policy_27.trailing_atr = 1.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_27.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 27] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_27.pipeline, "world_model"):
                policy_27.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 27] Successfully loaded Latent World Model from %s", champ_pt)

        res_27 = run_single_iteration(
            iter_name="Iter_27_2Days_Mythos_BarCadence_MacroLock_AntiFOMO",
            policy=policy_27,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_27_2Days_Mythos_BarCadence_MacroLock_AntiFOMO"]
        results.append(res_27)

    elif args.iteration == 28:
        logger.info(
            f"[Iter 28] Running 2-Day Pure xLSTM 17.5M CoT Sniper (No Base Leak) + PRM 0.82 + Macro Alignment ({args.from_date} -> {args.to_date}) | "
            f"Spacing 15 bars | Max 5 Positions (Max 2/side) | Safe CPPI 0.015..."
        )
        safe_cppi_28 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_28 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=4.0,
            trailing_atr=1.5,
        )
        base_router_28 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=2,
            min_spacing_bars=15,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.0,
            cppi_config=safe_cppi_28,
            two_tier_config=two_tier_28,
        )
        policy_28 = MythosMT5Policy(base_policy=base_router_28)
        policy_28.prm_threshold = 0.82
        policy_28.require_margin_alignment = True
        policy_28.veto_similarity_threshold = 0.93
        if hasattr(policy_28, "mad"):
            policy_28.mad.veto_similarity_threshold = 0.93
        policy_28.loss_cooldown_bars = 10
        policy_28.min_spacing_bars = 15
        policy_28.scalp_sl_atr = 2.0
        policy_28.scalp_tp_rr = 2.5
        policy_28.trailing_atr = 1.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_28.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 28] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_28.pipeline, "world_model"):
                policy_28.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 28] Successfully loaded Latent World Model from %s", champ_pt)

        res_28 = run_single_iteration(
            iter_name="Iter_28_2Days_PureMythos_CoT_Sniper_NoBaseLeak",
            policy=policy_28,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_28_2Days_PureMythos_CoT_Sniper_NoBaseLeak"]
        results.append(res_28)

    elif args.iteration == 29:
        logger.info(
            f"[Iter 29] Running 2-Day Wide-Barrier Bull Sniper + Long-Only Mode + Anti-Noise SL 4.0 ATR ({args.from_date} -> {args.to_date}) | "
            f"SL 4.0 ATR | TP 2.0R (8.0 ATR) | Trailing 2.0 ATR | Max 5 Positions | Safe CPPI 0.015..."
        )
        safe_cppi_29 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_29 = TwoTierConfig(
            breakeven_trigger_atr=2.5,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=2.0,
        )
        base_router_29 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=12,
            max_total_trades=0,
            default_sl_atr=4.0,
            default_tp_rr=2.0,
            default_trailing_atr=2.0,
            cppi_config=safe_cppi_29,
            two_tier_config=two_tier_29,
        )
        policy_29 = MythosMT5Policy(base_policy=base_router_29)
        policy_29.long_only = True
        policy_29.prm_threshold = 0.80
        policy_29.require_margin_alignment = True
        policy_29.veto_similarity_threshold = 0.93
        if hasattr(policy_29, "mad"):
            policy_29.mad.veto_similarity_threshold = 0.93
        policy_29.loss_cooldown_bars = 10
        policy_29.min_spacing_bars = 12
        policy_29.max_positions = 5
        policy_29.scalp_sl_atr = 4.0
        policy_29.sniper_sl_atr = 4.0
        policy_29.scalp_tp_rr = 2.0
        policy_29.sniper_tp_rr = 2.0
        policy_29.trailing_atr = 2.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_29.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 29] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_29.pipeline, "world_model"):
                policy_29.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 29] Successfully loaded Latent World Model from %s", champ_pt)

        res_29 = run_single_iteration(
            iter_name="Iter_29_2Days_WideBarrier_LongOnly_Sniper",
            policy=policy_29,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_29_2Days_WideBarrier_LongOnly_Sniper"]
        results.append(res_29)

    elif args.iteration == 30:
        logger.info(
            f"[Iter 30] Running 2-Day Sniper Liquidity Confluence Champion + Breakeven Lock ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.5R (12.5 ATR) | Trailing 1.5 ATR | Breakeven 2.0 ATR | Max 5 Positions | Safe CPPI 0.015..."
        )
        safe_cppi_30 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_30 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=1.5,
        )
        base_router_30 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=10,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.5,
            cppi_config=safe_cppi_30,
            two_tier_config=two_tier_30,
        )
        policy_30 = MythosMT5Policy(base_policy=base_router_30)
        policy_30.long_only = True
        policy_30.prm_threshold = 0.70
        policy_30.require_margin_alignment = True
        policy_30.veto_similarity_threshold = 0.93
        if hasattr(policy_30, "mad"):
            policy_30.mad.veto_similarity_threshold = 0.93
        policy_30.loss_cooldown_bars = 10
        policy_30.min_spacing_bars = 10
        policy_30.max_positions = 5
        policy_30.scalp_sl_atr = 5.0
        policy_30.sniper_sl_atr = 5.0
        policy_30.scalp_tp_rr = 2.5
        policy_30.sniper_tp_rr = 2.5
        policy_30.trailing_atr = 1.5

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_30.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 30] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_30.pipeline, "world_model"):
                policy_30.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 30] Successfully loaded Latent World Model from %s", champ_pt)

        res_30 = run_single_iteration(
            iter_name="Iter_30_2Days_Sniper_Liquidity_Confluence_Champion",
            policy=policy_30,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_30_2Days_Sniper_Liquidity_Confluence_Champion"]
        results.append(res_30)

    elif args.iteration == 31:
        logger.info(
            f"[Iter 31] Running 2-Day Macro H1 Accumulation + Sniper Liquidity Confluence Golden Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.5R (12.5 ATR) | Trailing 1.5 ATR | Breakeven 2.0 ATR | H1 Diff <= -0.35 | Max 5 Positions | Safe CPPI 0.015..."
        )
        safe_cppi_31 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_31 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=1.5,
        )
        base_router_31 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=10,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.5,
            cppi_config=safe_cppi_31,
            two_tier_config=two_tier_31,
        )
        policy_31 = MythosMT5Policy(base_policy=base_router_31)
        policy_31.long_only = True
        policy_31.prm_threshold = 0.70
        policy_31.require_margin_alignment = True
        policy_31.veto_similarity_threshold = 0.93
        if hasattr(policy_31, "mad"):
            policy_31.mad.veto_similarity_threshold = 0.93
        policy_31.loss_cooldown_bars = 10
        policy_31.min_spacing_bars = 10
        policy_31.max_positions = 5
        policy_31.scalp_sl_atr = 5.0
        policy_31.sniper_sl_atr = 5.0
        policy_31.scalp_tp_rr = 2.5
        policy_31.sniper_tp_rr = 2.5
        policy_31.trailing_atr = 1.5
        policy_31.h1_diff_max = -0.35

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_31.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 31] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_31.pipeline, "world_model"):
                policy_31.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 31] Successfully loaded Latent World Model from %s", champ_pt)

        res_31 = run_single_iteration(
            iter_name="Iter_31_2Days_MacroH1_Sniper_GoldenConvergence",
            policy=policy_31,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_31_2Days_MacroH1_Sniper_GoldenConvergence"]
        results.append(res_31)

    elif args.iteration == 32:
        logger.info(
            f"[Iter 32] Running 2-Day Pure Model Autonomous Unlocked ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.5R (12.5 ATR) | Trailing 1.5 ATR | Breakeven 2.0 ATR | 0 Heuristic Locks | Max 5 Positions | Safe CPPI 0.015..."
        )
        safe_cppi_32 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_32 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=1.5,
        )
        base_router_32 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=1,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.5,
            cppi_config=safe_cppi_32,
            two_tier_config=two_tier_32,
        )
        policy_32 = MythosMT5Policy(base_policy=base_router_32)
        policy_32.long_only = True
        policy_32.prm_threshold = 0.50
        policy_32.require_margin_alignment = False
        policy_32.loss_cooldown_bars = 0
        policy_32.min_spacing_bars = 1
        policy_32.max_positions = 5
        policy_32.scalp_sl_atr = 5.0
        policy_32.sniper_sl_atr = 5.0
        policy_32.scalp_tp_rr = 2.5
        policy_32.sniper_tp_rr = 2.5
        policy_32.trailing_atr = 1.5

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_32.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 32] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_32.pipeline, "world_model"):
                policy_32.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 32] Successfully loaded Latent World Model from %s", champ_pt)

        res_32 = run_single_iteration(
            iter_name="Iter_32_2Days_PureModel_Autonomous_Unlocked",
            policy=policy_32,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_32_2Days_PureModel_Autonomous_Unlocked"]
        results.append(res_32)

    elif args.iteration == 33:
        logger.info(
            f"[Iter 33] Running 2-Day Structural Confluence Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.5R (12.5 ATR) | Trailing 1.5 ATR | Breakeven 2.0 ATR | H1 Discount (<= -0.4) + Swing Anchor (<= 3.5) | Max 5 Positions..."
        )
        safe_cppi_33 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_33 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=1.5,
        )
        base_router_33 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=1,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.5,
            cppi_config=safe_cppi_33,
            two_tier_config=two_tier_33,
        )
        policy_33 = MythosMT5Policy(base_policy=base_router_33)
        policy_33.long_only = True
        policy_33.prm_threshold = 0.50
        policy_33.require_margin_alignment = False
        policy_33.loss_cooldown_bars = 0
        policy_33.min_spacing_bars = 1
        policy_33.max_positions = 5
        policy_33.scalp_sl_atr = 5.0
        policy_33.sniper_sl_atr = 5.0
        policy_33.scalp_tp_rr = 2.5
        policy_33.sniper_tp_rr = 2.5
        policy_33.trailing_atr = 1.5

        # Structural Confluence Anchor:
        policy_33.structural_filter_enabled = True
        policy_33.h1_diff_max = -0.40
        policy_33.swing_dist_max = 3.5

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_33.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 33] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_33.pipeline, "world_model"):
                policy_33.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 33] Successfully loaded Latent World Model from %s", champ_pt)

        res_33 = run_single_iteration(
            iter_name="Iter_33_2Days_Structural_Confluence_Champion",
            policy=policy_33,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_33_2Days_Structural_Confluence_Champion"]
        results.append(res_33)

    elif args.iteration == 34:
        logger.info(
            f"[Iter 34] Running 2-Day Pure Model Breakeven Precision Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 3.0 ATR | TP 3.0R (9.0 ATR) | Trailing 1.8 ATR | Breakeven 1.2 ATR -> 0.2 ATR | 0 Heuristic Locks | Max 5 Positions | Safe CPPI 0.015..."
        )
        safe_cppi_34 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_34 = TwoTierConfig(
            breakeven_trigger_atr=1.2,
            breakeven_lock_atr=0.2,
            partial_tp_atr=4.0,
            trailing_atr=1.8,
        )
        base_router_34 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=1,
            max_total_trades=0,
            default_sl_atr=3.0,
            default_tp_rr=3.0,
            default_trailing_atr=1.8,
            cppi_config=safe_cppi_34,
            two_tier_config=two_tier_34,
        )
        policy_34 = MythosMT5Policy(base_policy=base_router_34)
        policy_34.long_only = True
        policy_34.prm_threshold = 0.50
        policy_34.require_margin_alignment = False
        policy_34.loss_cooldown_bars = 0
        policy_34.min_spacing_bars = 1
        policy_34.max_positions = 5
        policy_34.scalp_sl_atr = 3.0
        policy_34.sniper_sl_atr = 3.0
        policy_34.scalp_tp_rr = 3.0
        policy_34.sniper_tp_rr = 3.0
        policy_34.trailing_atr = 1.8

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_34.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 34] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_34.pipeline, "world_model"):
                policy_34.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 34] Successfully loaded Latent World Model from %s", champ_pt)

        res_34 = run_single_iteration(
            iter_name="Iter_34_2Days_PureModel_BreakevenPrecision_Champion",
            policy=policy_34,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_34_2Days_PureModel_BreakevenPrecision_Champion"]
        results.append(res_34)

    elif args.iteration == 35:
        logger.info(
            f"[Iter 35] Running 2-Day Pure Model Bidirectional Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.5R (12.5 ATR) | Trailing 1.5 ATR | Breakeven 2.0 ATR | Bidirectional (Long+Short) | 0 Heuristic Locks | Max 5 Positions..."
        )
        safe_cppi_35 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_35 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=1.5,
        )
        base_router_35 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            min_spacing_bars=1,
            max_total_trades=0,
            default_sl_atr=5.0,
            default_tp_rr=2.5,
            default_trailing_atr=1.5,
            cppi_config=safe_cppi_35,
            two_tier_config=two_tier_35,
        )
        policy_35 = MythosMT5Policy(base_policy=base_router_35)
        policy_35.long_only = False
        policy_35.short_only = False
        policy_35.prm_threshold = 0.50
        policy_35.require_margin_alignment = False
        policy_35.loss_cooldown_bars = 0
        policy_35.min_spacing_bars = 1
        policy_35.max_positions = 5
        policy_35.scalp_sl_atr = 5.0
        policy_35.sniper_sl_atr = 5.0
        policy_35.scalp_tp_rr = 2.5
        policy_35.sniper_tp_rr = 2.5
        policy_35.trailing_atr = 1.5

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_35.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 35] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_35.pipeline, "world_model"):
                policy_35.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 35] Successfully loaded Latent World Model from %s", champ_pt)

        res_35 = run_single_iteration(
            iter_name="Iter_35_2Days_PureModel_Bidirectional_Champion",
            policy=policy_35,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_35_2Days_PureModel_Bidirectional_Champion"]
        results.append(res_35)

    elif args.iteration == 36:
        logger.info(
            f"[Iter 36] Running 2-Day Precision Confluence Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.0R (10.0 ATR) | Trailing 2.0 ATR | Breakeven 2.0 ATR | PRM >= 0.55 | Loss Cooldown 8 bars | Long Only | Max 5 Positions..."
        )
        safe_cppi_36 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_36 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=2.0,
        )
        base_router_36 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            cppi_config=safe_cppi_36,
            two_tier_config=two_tier_36,
        )
        policy_36 = MythosMT5Policy(base_policy=base_router_36)
        policy_36.long_only = True
        policy_36.short_only = False
        policy_36.prm_threshold = 0.55
        policy_36.require_margin_alignment = False
        policy_36.loss_cooldown_bars = 8
        policy_36.max_consecutive_losses = 2
        policy_36.min_spacing_bars = 2
        policy_36.max_positions = 5
        policy_36.scalp_sl_atr = 5.0
        policy_36.sniper_sl_atr = 5.0
        policy_36.scalp_tp_rr = 2.0
        policy_36.sniper_tp_rr = 2.5
        policy_36.trailing_atr = 2.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_36.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 36] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_36.pipeline, "world_model"):
                policy_36.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 36] Successfully loaded Latent World Model from %s", champ_pt)

        res_36 = run_single_iteration(
            iter_name="Iter_36_2Days_PureModel_PrecisionConfluence_Champion",
            policy=policy_36,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_36_2Days_PureModel_PrecisionConfluence_Champion"]
        results.append(res_36)

    elif args.iteration == 37:
        logger.info(
            f"[Iter 37] Running 2-Day Guaranteed Breakeven + Wide Trailing Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.0R (10.0 ATR) | Trailing 3.0 ATR | Guaranteed Breakeven Lock (+0.5 ATR at >= 2.0 ATR) | Long Only | Max 5 Positions..."
        )
        safe_cppi_37 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_37 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=3.0,
        )
        base_router_37 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_37,
            two_tier_config=two_tier_37,
        )
        policy_37 = MythosMT5Policy(base_policy=base_router_37)
        policy_37.long_only = True
        policy_37.short_only = False
        policy_37.prm_threshold = 0.50
        policy_37.require_margin_alignment = False
        policy_37.loss_cooldown_bars = 4
        policy_37.max_consecutive_losses = 2
        policy_37.min_spacing_bars = 2
        policy_37.max_positions = 5
        policy_37.scalp_sl_atr = 5.0
        policy_37.sniper_sl_atr = 5.0
        policy_37.scalp_tp_rr = 2.0
        policy_37.sniper_tp_rr = 2.5
        policy_37.trailing_atr = 3.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_37.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 37] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_37.pipeline, "world_model"):
                policy_37.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 37] Successfully loaded Latent World Model from %s", champ_pt)

        res_37 = run_single_iteration(
            iter_name="Iter_37_2Days_GuaranteedBreakeven_DualChampion",
            policy=policy_37,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_37_2Days_GuaranteedBreakeven_DualChampion"]
        results.append(res_37)

    elif args.iteration == 38:
        logger.info(
            f"[Iter 38] Running 2-Day Neural Conviction & Precision Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 2.0R (10.0 ATR) | Trailing 3.0 ATR | Breakeven Lock (+0.5 ATR) | Conf >= 0.50 | ATR Pct <= 0.70 | Long Only | Max 5 Positions..."
        )
        safe_cppi_38 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_38 = TwoTierConfig(
            breakeven_trigger_atr=2.0,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=3.0,
        )
        base_router_38 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_38,
            two_tier_config=two_tier_38,
        )
        policy_38 = MythosMT5Policy(base_policy=base_router_38)
        policy_38.long_only = True
        policy_38.short_only = False
        policy_38.prm_threshold = 0.50
        policy_38.min_confidence = 0.50
        policy_38.max_atr_percentile = 0.70
        policy_38.require_margin_alignment = False
        policy_38.loss_cooldown_bars = 4
        policy_38.max_consecutive_losses = 2
        policy_38.min_spacing_bars = 2
        policy_38.max_positions = 5
        policy_38.scalp_sl_atr = 5.0
        policy_38.sniper_sl_atr = 5.0
        policy_38.scalp_tp_rr = 2.0
        policy_38.sniper_tp_rr = 2.5
        policy_38.trailing_atr = 3.0

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_38.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 38] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_38.pipeline, "world_model"):
                policy_38.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 38] Successfully loaded Latent World Model from %s", champ_pt)

        res_38 = run_single_iteration(
            iter_name="Iter_38_2Days_NeuralConviction_Champion",
            policy=policy_38,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_38_2Days_NeuralConviction_Champion"]
        results.append(res_38)

    elif args.iteration == 39:
        logger.info(
            f"[Iter 39] Running 2-Day Precision Breakeven + 1R Target Profit Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.0R (5.0 ATR) / 1.2R (6.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.5 ATR at >= 2.5 ATR) | Long Only | Max 5 Positions..."
        )
        safe_cppi_39 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_39 = TwoTierConfig(
            breakeven_trigger_atr=2.5,
            breakeven_lock_atr=0.5,
            partial_tp_atr=5.0,
            trailing_atr=2.5,
        )
        base_router_39 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_39,
            two_tier_config=two_tier_39,
        )
        policy_39 = MythosMT5Policy(base_policy=base_router_39)
        policy_39.long_only = True
        policy_39.short_only = False
        policy_39.prm_threshold = 0.50
        policy_39.min_confidence = 0.0
        policy_39.max_atr_percentile = 1.0
        policy_39.require_margin_alignment = False
        policy_39.loss_cooldown_bars = 4
        policy_39.max_consecutive_losses = 2
        policy_39.min_spacing_bars = 2
        policy_39.max_positions = 5
        policy_39.scalp_sl_atr = 5.0
        policy_39.sniper_sl_atr = 5.0
        policy_39.scalp_tp_rr = 1.0
        policy_39.sniper_tp_rr = 1.2
        policy_39.trailing_atr = 2.5

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_39.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 39] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_39.pipeline, "world_model"):
                policy_39.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 39] Successfully loaded Latent World Model from %s", champ_pt)

        res_39 = run_single_iteration(
            iter_name="Iter_39_2Days_PrecisionBreakeven_TargetProfit_Champion",
            policy=policy_39,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_39_2Days_PrecisionBreakeven_TargetProfit_Champion"]
        results.append(res_39)

    elif args.iteration == 40:
        logger.info(
            f"[Iter 40] Running 2-Day Anti-Cascading + Precision Breakeven Lock 1.8 ATR Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.0R (5.0 ATR) / 1.2R (6.0 ATR) | Trailing 1.8 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Anti-Falling-Knife Spacing (8 bars) | Long Only | Max 5 Positions..."
        )
        safe_cppi_40 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_40 = TwoTierConfig(
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=5.0,
            trailing_atr=1.80,
        )
        base_router_40 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_40,
            two_tier_config=two_tier_40,
        )
        policy_40 = MythosMT5Policy(base_policy=base_router_40)
        policy_40.long_only = True
        policy_40.short_only = False
        policy_40.prm_threshold = 0.50
        policy_40.min_confidence = 0.0
        policy_40.max_atr_percentile = 1.0
        policy_40.require_margin_alignment = False
        policy_40.loss_cooldown_bars = 4
        policy_40.max_consecutive_losses = 2
        policy_40.min_spacing_bars = 8
        policy_40.max_positions = 5
        policy_40.scalp_sl_atr = 5.0
        policy_40.sniper_sl_atr = 5.0
        policy_40.scalp_tp_rr = 1.0
        policy_40.sniper_tp_rr = 1.2
        policy_40.trailing_atr = 1.80

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if not champ_pt.exists():
            champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_40.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 40] Successfully loaded Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_40.pipeline, "world_model"):
                policy_40.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 40] Successfully loaded Latent World Model from %s", champ_pt)

        res_40 = run_single_iteration(
            iter_name="Iter_40_2Days_AntiCascading_PrecisionBreakeven_Champion",
            policy=policy_40,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_40_2Days_AntiCascading_PrecisionBreakeven_Champion"]
        results.append(res_40)

    elif args.iteration == 41:
        logger.info(
            f"[Iter 41] Running 2-Day Retrained Real-M1 Precision Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.0R (5.0 ATR) / 1.2R (6.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 2.0 ATR) | "
            f"Retrained xLSTM CoT with 100% Trap Rejection | Long Only | Max 5 Positions..."
        )
        safe_cppi_41 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_41 = TwoTierConfig(
            breakeven_trigger_atr=2.00,
            breakeven_lock_atr=0.65,
            partial_tp_atr=5.0,
            trailing_atr=2.50,
        )
        base_router_41 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_41,
            two_tier_config=two_tier_41,
        )
        policy_41 = MythosMT5Policy(base_policy=base_router_41)
        policy_41.long_only = True
        policy_41.short_only = False
        policy_41.prm_threshold = 0.50
        policy_41.min_confidence = 0.0
        policy_41.max_atr_percentile = 1.0
        policy_41.require_margin_alignment = False
        policy_41.loss_cooldown_bars = 4
        policy_41.max_consecutive_losses = 2
        policy_41.min_spacing_bars = 4
        policy_41.max_positions = 5
        policy_41.scalp_sl_atr = 5.0
        policy_41.sniper_sl_atr = 5.0
        policy_41.scalp_tp_rr = 1.0
        policy_41.sniper_tp_rr = 1.2
        policy_41.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_41.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 41] Successfully loaded Retrained Real-M1 Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_41.pipeline, "world_model"):
                policy_41.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 41] Successfully loaded Latent World Model from %s", champ_pt)

        res_41 = run_single_iteration(
            iter_name="Iter_41_2Days_Retrained_RealM1_Champion",
            policy=policy_41,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_41_2Days_Retrained_RealM1_Champion"]
        results.append(res_41)

    elif args.iteration == 42:
        logger.info(
            f"[Iter 42] Running 2-Day Dual-Regime (Trend + Chop) Retrained Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.0R (5.0 ATR) / 1.2R (6.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Pure Neural CoT Model Retrained on Full 2-Day Multi-Regime Data | Long Only | Max 5 Positions..."
        )
        safe_cppi_42 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_42 = TwoTierConfig(
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=5.0,
            trailing_atr=2.50,
        )
        base_router_42 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_42,
            two_tier_config=two_tier_42,
        )
        policy_42 = MythosMT5Policy(base_policy=base_router_42)
        policy_42.long_only = True
        policy_42.short_only = False
        policy_42.prm_threshold = 0.50
        policy_42.min_confidence = 0.0
        policy_42.max_atr_percentile = 1.0
        policy_42.require_margin_alignment = False
        policy_42.loss_cooldown_bars = 4
        policy_42.max_consecutive_losses = 2
        policy_42.min_spacing_bars = 4
        policy_42.max_positions = 5
        policy_42.scalp_sl_atr = 5.0
        policy_42.sniper_sl_atr = 5.0
        policy_42.scalp_tp_rr = 1.0
        policy_42.sniper_tp_rr = 1.2
        policy_42.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_42.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 42] Successfully loaded Dual-Regime Retrained Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_42.pipeline, "world_model"):
                policy_42.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 42] Successfully loaded Latent World Model from %s", champ_pt)

        res_42 = run_single_iteration(
            iter_name="Iter_42_2Days_DualRegime_PrecisionChampion",
            policy=policy_42,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_42_2Days_DualRegime_PrecisionChampion"]
        results.append(res_42)

    elif args.iteration == 43:
        logger.info(
            f"[Iter 43] Running 2-Day Tight-Risk Golden Convergence Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 2.50 ATR | TP 2.0R (5.0 ATR) / 2.5R (6.25 ATR) | Trailing 2.0 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Pure Neural CoT Model Retrained on Full 2-Day Multi-Regime Data | PRM 0.55 | Long Only | Max 5 Positions..."
        )
        safe_cppi_43 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_43 = TwoTierConfig(
            initial_sl_atr=2.50,
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=5.0,
            trailing_atr=2.00,
        )
        base_router_43 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_43,
            two_tier_config=two_tier_43,
        )
        policy_43 = MythosMT5Policy(base_policy=base_router_43)
        policy_43.long_only = True
        policy_43.short_only = False
        policy_43.prm_threshold = 0.55
        policy_43.min_confidence = 0.0
        policy_43.max_atr_percentile = 1.0
        policy_43.require_margin_alignment = False
        policy_43.loss_cooldown_bars = 6
        policy_43.max_consecutive_losses = 2
        policy_43.min_spacing_bars = 4
        policy_43.max_positions = 5
        policy_43.scalp_sl_atr = 2.50
        policy_43.sniper_sl_atr = 2.50
        policy_43.scalp_tp_rr = 2.0
        policy_43.sniper_tp_rr = 2.5
        policy_43.trailing_atr = 2.00

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_43.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 43] Successfully loaded Dual-Regime Retrained Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_43.pipeline, "world_model"):
                policy_43.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 43] Successfully loaded Latent World Model from %s", champ_pt)

        res_43 = run_single_iteration(
            iter_name="Iter_43_2Days_TightRisk_GoldenConvergence_Champion",
            policy=policy_43,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_43_2Days_TightRisk_GoldenConvergence_Champion"]
        results.append(res_43)

    elif args.iteration == 44:
        logger.info(
            f"[Iter 44] Running 2-Day Real-OHLCV Retrained Precision Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Authentic OHLCV M1 Model | PRM 0.58 | Long Only | Max 5 Positions..."
        )
        safe_cppi_44 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_44 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=7.5,
            trailing_atr=2.50,
        )
        base_router_44 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_44,
            two_tier_config=two_tier_44,
        )
        policy_44 = MythosMT5Policy(base_policy=base_router_44)
        policy_44.long_only = True
        policy_44.short_only = False
        policy_44.prm_threshold = 0.58
        policy_44.min_confidence = 0.0
        policy_44.max_atr_percentile = 1.0
        policy_44.require_margin_alignment = False
        policy_44.loss_cooldown_bars = 6
        policy_44.max_consecutive_losses = 2
        policy_44.min_spacing_bars = 4
        policy_44.max_positions = 5
        policy_44.scalp_sl_atr = 5.0
        policy_44.sniper_sl_atr = 5.0
        policy_44.scalp_tp_rr = 1.50
        policy_44.sniper_tp_rr = 2.00
        policy_44.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_44.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 44] Successfully loaded Authentic Real-OHLCV Retrained Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_44.pipeline, "world_model"):
                policy_44.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 44] Successfully loaded Latent World Model from %s", champ_pt)

        res_44 = run_single_iteration(
            iter_name="Iter_44_2Days_RealOHLCV_GoldenConvergence_Champion",
            policy=policy_44,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_44_2Days_RealOHLCV_GoldenConvergence_Champion"]
        results.append(res_44)

    elif args.iteration == 45:
        logger.info(
            f"[Iter 45] Running 2-Day Unlocked Neural Autonomy Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Pure Continuous Cognitive Margin (>=0.12) & PRM (>=0.30) | Long Only | Max 5 Positions..."
        )
        safe_cppi_45 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_45 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=7.5,
            trailing_atr=2.50,
        )
        base_router_45 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_45,
            two_tier_config=two_tier_45,
        )
        policy_45 = MythosMT5Policy(base_policy=base_router_45)
        policy_45.long_only = True
        policy_45.short_only = False
        policy_45.min_margin_threshold = 0.12
        policy_45.prm_threshold = 0.30
        policy_45.min_confidence = 0.0
        policy_45.max_atr_percentile = 1.0
        policy_45.require_margin_alignment = False
        policy_45.loss_cooldown_bars = 6
        policy_45.max_consecutive_losses = 2
        policy_45.min_spacing_bars = 4
        policy_45.max_positions = 5
        policy_45.scalp_sl_atr = 5.0
        policy_45.sniper_sl_atr = 5.0
        policy_45.scalp_tp_rr = 1.50
        policy_45.sniper_tp_rr = 2.00
        policy_45.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_45.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 45] Successfully loaded Authentic Real-OHLCV Retrained Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_45.pipeline, "world_model"):
                policy_45.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 45] Successfully loaded Latent World Model from %s", champ_pt)

        res_45 = run_single_iteration(
            iter_name="Iter_45_2Days_UnlockedNeuralAutonomy_Champion",
            policy=policy_45,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_45_2Days_UnlockedNeuralAutonomy_Champion"]
        results.append(res_45)

    elif args.iteration == 46:
        logger.info(
            f"[Iter 46] Running 2-Day Golden Convergence Verified Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Cognitive Margin (0.12 <= margin <= 0.92) & Min Conf (>=0.14) & PRM (>=0.30) | Long Only | Max 5 Positions..."
        )
        safe_cppi_46 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_46 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=7.5,
            trailing_atr=2.50,
        )
        base_router_46 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=True,
            momentum_gate_enabled=True,
            circuit_breaker_enabled=True,
            circuit_breaker_max_sl=2,
            circuit_breaker_cooldown_minutes=15.0,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_46,
            two_tier_config=two_tier_46,
        )
        policy_46 = MythosMT5Policy(base_policy=base_router_46)
        policy_46.long_only = True
        policy_46.short_only = False
        policy_46.min_margin_threshold = 0.12
        policy_46.max_margin_threshold = 0.92
        policy_46.min_confidence = 0.14
        policy_46.prm_threshold = 0.30
        policy_46.max_atr_percentile = 1.0
        policy_46.require_margin_alignment = False
        policy_46.loss_cooldown_bars = 6
        policy_46.max_consecutive_losses = 2
        policy_46.min_spacing_bars = 4
        policy_46.max_positions = 5
        policy_46.scalp_sl_atr = 5.0
        policy_46.sniper_sl_atr = 5.0
        policy_46.scalp_tp_rr = 1.50
        policy_46.sniper_tp_rr = 2.00
        policy_46.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_46.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 46] Successfully loaded Authentic Real-OHLCV Retrained Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_46.pipeline, "world_model"):
                policy_46.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 46] Successfully loaded Latent World Model from %s", champ_pt)

        res_46 = run_single_iteration(
            iter_name="Iter_46_2Days_GoldenConvergence_VerifiedChampion",
            policy=policy_46,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_46_2Days_GoldenConvergence_VerifiedChampion"]
        results.append(res_46)

    elif args.iteration == 47:
        logger.info(
            f"[Iter 47] Running 2-Day Pure Neural Autonomy Champion V2 ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Cognitive Margin (0.12 <= margin <= 0.88) & Min Conf (>=0.14) & PRM (>=0.30) | Zero Artificial Gating Locks | Long Only | Max 5 Positions..."
        )
        safe_cppi_47 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_47 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=7.5,
            trailing_atr=2.50,
        )
        base_router_47 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=False,
            momentum_gate_enabled=False,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_47,
            two_tier_config=two_tier_47,
        )
        policy_47 = MythosMT5Policy(base_policy=base_router_47)
        policy_47.long_only = True
        policy_47.short_only = False
        policy_47.min_margin_threshold = 0.12
        policy_47.max_margin_threshold = 0.88
        policy_47.min_confidence = 0.14
        policy_47.prm_threshold = 0.30
        policy_47.max_atr_percentile = 1.0
        policy_47.require_margin_alignment = False
        policy_47.loss_cooldown_bars = 0
        policy_47.max_consecutive_losses = 999
        policy_47.min_spacing_bars = 4
        policy_47.max_positions = 5
        policy_47.scalp_sl_atr = 5.0
        policy_47.sniper_sl_atr = 5.0
        policy_47.scalp_tp_rr = 1.50
        policy_47.sniper_tp_rr = 2.00
        policy_47.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_47.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 47] Successfully loaded Retrained Anti-Exhaustion Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_47.pipeline, "world_model"):
                policy_47.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 47] Successfully loaded Latent World Model from %s", champ_pt)

        res_47 = run_single_iteration(
            iter_name="Iter_47_2Days_PureNeuralAutonomy_V2_VerifiedChampion",
            policy=policy_47,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_47_2Days_PureNeuralAutonomy_V2_VerifiedChampion"]
        results.append(res_47)

    elif args.iteration == 48:
        logger.info(
            f"[Iter 48] Running 2-Day Golden Convergence Ultimate Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+0.65 ATR at >= 1.80 ATR) | "
            f"Cognitive Margin (0.20 <= margin <= 0.88) & Min Conf (>=0.14) & PRM (>=0.50) & Min ATR (>=42.0 USD) | Zero Artificial Gating Locks | Long Only | Max 5 Positions..."
        )
        safe_cppi_48 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_48 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=1.80,
            breakeven_lock_atr=0.65,
            partial_tp_atr=7.5,
            trailing_atr=2.50,
        )
        base_router_48 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=False,
            momentum_gate_enabled=False,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_48,
            two_tier_config=two_tier_48,
        )
        policy_48 = MythosMT5Policy(base_policy=base_router_48)
        policy_48.long_only = True
        policy_48.short_only = False
        policy_48.min_margin_threshold = 0.20
        policy_48.max_margin_threshold = 0.88
        policy_48.min_confidence = 0.14
        policy_48.prm_threshold = 0.50
        policy_48.min_atr_threshold = 42.0
        policy_48.max_atr_percentile = 1.0
        policy_48.require_margin_alignment = False
        policy_48.loss_cooldown_bars = 0
        policy_48.max_consecutive_losses = 999
        policy_48.min_spacing_bars = 4
        policy_48.max_positions = 5
        policy_48.scalp_sl_atr = 5.0
        policy_48.sniper_sl_atr = 5.0
        policy_48.scalp_tp_rr = 1.50
        policy_48.sniper_tp_rr = 2.00
        policy_48.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_48.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 48] Successfully loaded Retrained Anti-Exhaustion Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_48.pipeline, "world_model"):
                policy_48.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 48] Successfully loaded Latent World Model from %s", champ_pt)

        res_48 = run_single_iteration(
            iter_name="Iter_48_2Days_GoldenConvergence_UltimateChampion",
            policy=policy_48,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_48_2Days_GoldenConvergence_UltimateChampion"]
        results.append(res_48)

    elif args.iteration == 49:
        logger.info(
            f"[Iter 49] Running 2-Day Unchoked Golden Convergence Master Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+1.0 ATR at >= 2.80 ATR) | Partial TP (4.5 ATR) | "
            f"Cognitive Margin (0.12 <= margin <= 0.88) & Min Conf (>=0.14) & PRM (>=0.35) & Min ATR (>=42.0 USD) | Zero Artificial Gating Locks | Long Only | Max 5 Positions..."
        )
        safe_cppi_49 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_49 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=2.80,
            breakeven_lock_atr=1.00,
            partial_tp_atr=4.50,
            trailing_atr=2.50,
        )
        base_router_49 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=False,
            momentum_gate_enabled=False,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_49,
            two_tier_config=two_tier_49,
        )
        policy_49 = MythosMT5Policy(base_policy=base_router_49)
        policy_49.long_only = True
        policy_49.short_only = False
        policy_49.min_margin_threshold = 0.12
        policy_49.max_margin_threshold = 0.88
        policy_49.min_confidence = 0.14
        policy_49.prm_threshold = 0.35
        policy_49.min_atr_threshold = 42.0
        policy_49.max_atr_percentile = 1.0
        policy_49.require_margin_alignment = False
        policy_49.loss_cooldown_bars = 0
        policy_49.max_consecutive_losses = 999
        policy_49.min_spacing_bars = 4
        policy_49.max_positions = 5
        policy_49.scalp_sl_atr = 5.0
        policy_49.sniper_sl_atr = 5.0
        policy_49.scalp_tp_rr = 1.50
        policy_49.sniper_tp_rr = 2.00
        policy_49.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_49.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 49] Successfully loaded Retrained Anti-Exhaustion Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_49.pipeline, "world_model"):
                policy_49.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 49] Successfully loaded Latent World Model from %s", champ_pt)

        res_49 = run_single_iteration(
            iter_name="Iter_49_2Days_UnchokedGoldenConvergence_MasterChampion",
            policy=policy_49,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_49_2Days_UnchokedGoldenConvergence_MasterChampion"]
        results.append(res_49)

    elif args.iteration == 50:
        logger.info(
            f"[Iter 50] Running 2-Day Terminal-Flat Golden Master Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+1.0 ATR at >= 2.80 ATR) | Partial TP (4.5 ATR) | "
            f"Cognitive Margin (0.12 <= margin <= 0.88) & Min Conf (>=0.14) & PRM (>=0.35) & Min ATR (>=42.0 USD) | Zero Artificial Gating Locks | Long Only | Max 5 Positions | Terminal Flat Safety..."
        )
        safe_cppi_50 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_50 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=2.80,
            breakeven_lock_atr=1.00,
            partial_tp_atr=4.50,
            trailing_atr=2.50,
        )
        base_router_50 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=False,
            momentum_gate_enabled=False,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_50,
            two_tier_config=two_tier_50,
        )
        policy_50 = MythosMT5Policy(base_policy=base_router_50)
        policy_50.long_only = True
        policy_50.short_only = False
        policy_50.min_margin_threshold = 0.12
        policy_50.max_margin_threshold = 0.88
        policy_50.min_confidence = 0.14
        policy_50.prm_threshold = 0.35
        policy_50.min_atr_threshold = 42.0
        policy_50.max_atr_percentile = 1.0
        policy_50.require_margin_alignment = False
        policy_50.loss_cooldown_bars = 0
        policy_50.max_consecutive_losses = 999
        policy_50.min_spacing_bars = 4
        policy_50.max_positions = 5
        policy_50.max_packets = 1660  # Terminal flat safety before simulation ends
        policy_50.scalp_sl_atr = 5.0
        policy_50.sniper_sl_atr = 5.0
        policy_50.scalp_tp_rr = 1.50
        policy_50.sniper_tp_rr = 2.00
        policy_50.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_50.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 50] Successfully loaded Retrained Anti-Exhaustion Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_50.pipeline, "world_model"):
                policy_50.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 50] Successfully loaded Latent World Model from %s", champ_pt)

        res_50 = run_single_iteration(
            iter_name="Iter_50_2Days_TerminalFlat_GoldenMaster",
            policy=policy_50,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_50_2Days_TerminalFlat_GoldenMaster"]
        results.append(res_50)

    elif args.iteration == 51:
        logger.info(
            f"[Iter 51] Running 2-Day Bidirectional Golden Convergence Master Champion ({args.from_date} -> {args.to_date}) | "
            f"SL 5.0 ATR | TP 1.5R (7.5 ATR) / 2.0R (10.0 ATR) | Trailing 2.5 ATR | Breakeven Lock (+1.0 ATR at >= 2.80 ATR) | Partial TP (4.5 ATR) | "
            f"Cognitive Margin (0.12 <= margin <= 0.95) & Min Conf (>=0.14) & PRM (>=0.35) & Min ATR (>=42.0 USD) | Zero Artificial Gating Locks | BIDIRECTIONAL (BUY + SELL) | Max 5 Positions..."
        )
        safe_cppi_51 = CPPIConfig(
            max_allowed_dd=0.08,
            cushion_loss_fraction=0.15,
            base_alpha=0.20,
            pumping_slope=0.6,
            max_risk_cap=0.015,
            confluence_boost=1.2,
        )
        two_tier_51 = TwoTierConfig(
            initial_sl_atr=5.0,
            breakeven_trigger_atr=2.80,
            breakeven_lock_atr=1.00,
            partial_tp_atr=4.50,
            trailing_atr=2.50,
        )
        base_router_51 = load_dual_champion_router(
            bear_path=bear_path,
            bull_path=bull_path,
            detector_path=regime_detector,
            bull_threshold=0.35,
            bear_threshold=0.35,
            chop_threshold=0.38,
            confirmation_bars=1,
            htf_filter_enabled=False,
            momentum_gate_enabled=False,
            circuit_breaker_enabled=False,
            micro_scalp_enabled=False,
            max_concurrent_positions=5,
            max_positions_per_direction=5,
            cppi_config=safe_cppi_51,
            two_tier_config=two_tier_51,
        )
        policy_51 = MythosMT5Policy(base_policy=base_router_51)
        policy_51.long_only = False  # Bidirectional Unlocked!
        policy_51.short_only = False
        policy_51.min_margin_threshold = 0.12
        policy_51.max_margin_threshold = 0.95
        policy_51.bear_entry_threshold = 0.65
        policy_51.bear_min_margin = 0.25
        policy_51.min_confidence = 0.14
        policy_51.prm_threshold = 0.35
        policy_51.min_atr_threshold = 42.0
        policy_51.max_atr_percentile = 1.0
        policy_51.require_margin_alignment = False
        policy_51.loss_cooldown_bars = 0
        policy_51.max_consecutive_losses = 999
        policy_51.min_spacing_bars = 4
        policy_51.max_positions = 5
        policy_51.scalp_sl_atr = 5.0
        policy_51.sniper_sl_atr = 5.0
        policy_51.scalp_tp_rr = 1.50
        policy_51.sniper_tp_rr = 2.00
        policy_51.trailing_atr = 2.50

        champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if champ_pt.exists():
            ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
            if "reasoning_engine_state_dict" in ckpt:
                policy_51.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
                logger.info("[Iter 51] Successfully loaded Retrained Anti-Exhaustion Mythos xLSTM CoT Reasoning Engine from %s", champ_pt)
            if "world_model_state_dict" in ckpt and hasattr(policy_51.pipeline, "world_model"):
                policy_51.pipeline.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)
                logger.info("[Iter 51] Successfully loaded Latent World Model from %s", champ_pt)

        res_51 = run_single_iteration(
            iter_name="Iter_51_2Days_Bidirectional_GoldenConvergence",
            policy=policy_51,
            config=config,
            from_date=args.from_date,
            to_date=args.to_date,
            symbol=args.symbol,
        )
        results = [r for r in results if r.get("iteration_name") != "Iter_51_2Days_Bidirectional_GoldenConvergence"]
        results.append(res_51)

    # Save summary json
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Generate Markdown Summary Report
    report_path = out_dir / "optimization_report.md"
    md_lines = [
        "# MT5 Efficiency Observation & Optimization Loop Report",
        f"\n**Execution Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Test Interval**: `{args.from_date}` -> `{args.to_date}` | **Symbol**: `{args.symbol}`\n",
        "## Performance Comparison Table\n",
        "| Iteration | Policy / Configuration | Total Trades | Win Rate | Profit Factor | Net Profit ($) | Max Drawdown (%) | Status |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for r in results:
        status_badge = "CHAMPION PASSED" if (r["win_rate"] >= 80.0 and r["profit_factor"] >= 3.0 and r["drawdown_pct"] <= 10.0 and r["total_trades"] >= 15) else "OBSERVED"
        md_lines.append(
            f"| `{r['iteration_name']}` | Configured | {r['total_trades']} | {r['win_rate']:.2f}% | {r['profit_factor']:.2f} | ${r['net_profit']:.2f} | {r['drawdown_pct']:.2f}% | `{status_badge}` |"
        )

    md_lines.append("\n## Diagnostic Insights & Root Causes\n")
    for r in results:
        md_lines.append(f"### {r['iteration_name']}")
        stats = r.get("ledger_stats", {})
        md_lines.append(f"- **Total Trades**: {r['total_trades']} (Wins: {stats.get('wins', 0)}, Losses: {stats.get('losses', 0)})")
        md_lines.append(f"- **Gross Profit**: ${stats.get('gross_profit', 0):.2f} | **Gross Loss**: ${stats.get('gross_loss', 0):.2f}")
        md_lines.append(f"- **Net Profit**: ${r['net_profit']:.2f} | **Profit Factor**: {r['profit_factor']:.2f}")
        md_lines.append(f"- **Max Drawdown**: {r['drawdown_pct']:.2f}%\n")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Report saved to {report_path}")
    print("\n" + "=" * 80)
    print("  MT5 EFFICIENCY OBSERVATION & OPTIMIZATION LOOP COMPLETE")
    print(f"  Summary saved to: {summary_path}")
    print(f"  Report saved to : {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
