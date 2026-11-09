#!/usr/bin/env python3
"""
scripts/launch_visual_mt5_strategy_tester.py

Interactive Visual MT5 Strategy Tester Launcher.
Runs the Aggressive Alpha Pumping Champion Strategy in full graphical mode (Visual=1)
allowing real-time chart inspection of candlesticks, trades, SL/TP levels, and dynamic ATR Trailing Stops.

Aligned Parameters from /grill-me interview:
- Target: Aggressive Alpha Pumping (30-60 trades/2 days, Max DD <= 8-10%)
- Leverage: Dynamic CPPI Pumping Slope 2.2x, base_risk 1.0%, risk_cap 5.0%
- Geometry: SL 2.0 ATR, TP 2.5 RR, Trailing Stop 1.2 ATR
- Gating: Dual-Champion Core + Mythos PRM Gate 0.58 + Bar Spacing 3 bars
- Cooldown: 5 M1 bars on 2 consecutive SL + 50% Soft Deleveraging
"""

import argparse
import logging
import os
from pathlib import Path
import sys
import time
import torch

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import evaluate_policy_mt5_oos
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.cppi_risk_engine import CPPIConfig
from self_growing_trader.isolated_mt5_runner import isolated_mt5_runner_context
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("VisualMT5Tester")


def main():
    parser = argparse.ArgumentParser(description="Launch Visual MT5 Strategy Tester")
    parser.add_argument("--from-date", default="2025.02.20", help="Backtest start date (YYYY.MM.DD)")
    parser.add_argument("--to-date", default="2025.02.25", help="Backtest end date (YYYY.MM.DD)")
    parser.add_argument("--symbol", default="BTCUSDm", help="Trading symbol (default: BTCUSDm)")
    parser.add_argument("--model", type=int, default=1, help="Tester model (1: M1 OHLC, 0: Real ticks)")
    parser.add_argument("--prm-gate", type=float, default=0.58, help="PRM gating threshold (default: 0.58)")
    parser.add_argument("--max-risk", type=float, default=0.05, help="Max risk cap per trade (default: 0.05 = 5 percent)")
    parser.add_argument("--direct", action="store_true", default=True, help="Run direct active visual mode (default: True)")
    args = parser.parse_args()

    print("=" * 80)
    print("  LAUNCHING VISUAL MT5 STRATEGY TESTER (AGGRESSIVE ALPHA PUMPING CHAMPION)")
    print(f"  Symbol: {args.symbol} | Dates: {args.from_date} -> {args.to_date} | Model: {args.model}")
    print("=" * 80)

    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    config = load_config(config_path)

    bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    regime_detector = RegimeDetector.load(regime_model_path)

    # 1. Calibrate CPPI Risk Engine for Aggressive Alpha Pumping
    cppi_cfg = CPPIConfig(
        max_allowed_dd=0.08,             # Max allowed DD 8.0%
        cushion_loss_fraction=0.25,      # 25% of cushion
        base_alpha=0.5,                  # 0.5x base leverage contraction factor
        pumping_slope=2.2,               # Pumping slope 2.2x chosen in grill-me
        max_risk_cap=args.max_risk,      # Configurable risk cap
    )

    # 2. Build Dual-Champion Router Core
    logger.info("Initializing Dual-Champion Router with CPPI 2.2x & ATR Trailing Stop 1.2...")
    base_router = load_dual_champion_router(
        bear_path=bear_path,
        bull_path=bull_path,
        detector_path=regime_detector,
        bull_threshold=0.35,
        bear_threshold=0.35,
        chop_threshold=0.38,
        circuit_breaker_enabled=False,   # Handled by Mythos bar-based cooldown
        micro_scalp_enabled=False,       # Pure specialists core
        cppi_config=cppi_cfg,
    )

    # 3. Wrap with Mythos Metacognitive Reasoning Policy
    logger.info("Configuring Policy (direct=%s, prm_gate=%s)...", args.direct, args.prm_gate)
    policy = MythosMT5Policy(base_policy=base_router)
    if args.direct:
        logger.info("Applying DIRECT High-Frequency Visual Mode (Active M1 Execution)...")
        policy.direct_action = True
        policy.prm_threshold = float(args.prm_gate)
        policy.min_confidence = 0.35
        policy.min_margin_threshold = 0.04
        policy.bear_entry_threshold = 0.35
        policy.bear_min_margin = 0.04
        policy.min_spacing_bars = 3
        policy.post_exit_cooldown_bars = 3
        policy.loss_cooldown_bars = 5
        policy.max_consecutive_losses = 2
        policy.max_positions = 3
    else:
        policy.direct_action = True
        policy.prm_threshold = float(args.prm_gate)
        policy.loss_cooldown_bars = 5
        policy.max_consecutive_losses = 2
        policy.min_spacing_bars = 3
        policy.post_exit_cooldown_bars = 3
        policy.max_positions = 3

    policy.veto_similarity_threshold = 0.94
    if hasattr(policy, "mad"):
        policy.mad.veto_similarity_threshold = 0.94
    policy.scalp_sl_atr = 2.50           # Spread-compensated SL >= 2.50 ATR
    policy.scalp_tp_rr = 1.50            # TP 3.75 ATR
    policy.trailing_atr = 1.20           # Dynamic Trailing Stop 1.2 ATR
    policy.sniper_sl_atr = 3.50          # Spread-compensated SL 3.50 ATR
    policy.sniper_tp_rr = 1.50

    # Bypass microstructure shield and matrix correlation filters for strict zero-filter execution
    if hasattr(policy, "pipeline"):
        if hasattr(policy.pipeline, "microstructure_shield"):
            policy.pipeline.microstructure_shield.min_cost_ratio = 0.0
        if hasattr(policy.pipeline, "matrix"):
            policy.pipeline.matrix.max_correlation = 1.0

    # Injected Champion CoT weights - Priority: Pure Anti-MAE Sniper Champion
    candidates = [
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt",
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_sniper_anti_mae_champion.pt",
        ROOT / "outputs" / "checkpoints" / "tca_optimal_frontier.pt",
        ROOT / "outputs" / "checkpoints" / "openai_five_continuous_champion.pt",
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_asset_champion.pt",
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt",
    ]
    champ_pt = None
    for p in candidates:
        if p.exists():
            champ_pt = p
            break

    if champ_pt is not None:
        raw_dict = torch.load(champ_pt, map_location="cpu", weights_only=False)
        if "bull_cot_state_dict" in raw_dict and hasattr(policy.pipeline, "bilateral_engine"):
            policy.pipeline.bilateral_engine.load_checkpoint(champ_pt)
            policy.pipeline.bilateral_engine.config.prm_threshold = float(args.prm_gate)
            if hasattr(policy.pipeline.bilateral_engine, "arbiter"):
                policy.pipeline.bilateral_engine.arbiter.prm_threshold = float(args.prm_gate)
            logger.info("Loaded CoT weights via load_checkpoint into bilateral_engine from %s (prm_gate=%.2f)", champ_pt.name, float(args.prm_gate))
            print(f"  [OK] Injected Pure Anti-MAE Sniper Champion into Bilateral Engine: {champ_pt.name} (prm_threshold={float(args.prm_gate):.2f})")
        elif "reasoning_engine_state_dict" in raw_dict and hasattr(policy.pipeline, "cot_engine"):
            policy.pipeline.cot_engine.load_state_dict(raw_dict["reasoning_engine_state_dict"], strict=True)
            logger.info("Loaded CoT weights from %s", champ_pt.name)
            print(f"  [OK] Injected Supreme Champion Reasoning Weights: {champ_pt.name}")

    # 4. Launch in isolated visual context
    from_date = args.from_date
    to_date = args.to_date
    symbol = args.symbol

    logger.info("Preparing Visual MT5 Strategy Tester window (%s from %s to %s)...", symbol, from_date, to_date)
    with isolated_mt5_runner_context(
        session_name="visual_tester_agg",
        symbol=symbol,
        from_date=from_date,
        to_date=to_date,
        shutdown_terminal=False,         # Keep window open for user
        auto_cleanup=False,              # Keep sandbox folder
    ) as runner_dir:
        print("\n" + "=" * 80)
        print(f"  [OK] VISUAL RUNNER DIRECTORY: {runner_dir}")
        print("  Starting MetaTrader 5 GUI Window...")
        print("  (Ban co the quan sat bieu do nen M1, mui ten vao lenh va duong Trailing Stop!)")
        print("=" * 80 + "\n")

        result = evaluate_policy_mt5_oos(
            policy,
            config,
            runner_dir=runner_dir,
            timeout_sec=14400,
            visual=True,                 # Visual=1 in tester.ini
            shutdown_terminal=False,     # Keep terminal open
            from_date=from_date,
            to_date=to_date,
            model=args.model,
        )

        best = result.get("best", {})
        metrics = best.get("metrics", {})
        print("\n" + "=" * 80)
        print("  VISUAL MT5 STRATEGY TESTER RUN COMPLETE")
        print(f"  Metrics: {metrics}")
        print("=" * 80)


if __name__ == "__main__":
    main()
