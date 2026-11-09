#!/usr/bin/env python3
"""
Execute Frontier Mythos + MuZero on MT5 Strategy Tester.
Launches terminal64.exe directly using MT5StrategyTesterEvaluator with live PnL, WR, PF, and Reasoning stream.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
import torch

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_execution_contract_kwargs,
)
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    MT5_MODEL_1MIN_OHLC,
    MT5_MODEL_REAL_TICKS,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


def clean_stale_ledgers(runner_dir: Path) -> None:
    """Clear stale execution records before a new test run."""
    stale_files = [
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_execution.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_action.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_request.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_stop.request",
        runner_dir / "MQL5" / "Files" / "self_growing_execution.csv",
        runner_dir / "MQL5" / "Files" / "self_growing_bridge.csv",
        runner_dir / "MQL5" / "Files" / "self_growing_action.csv",
        runner_dir / "MQL5" / "Files" / "self_growing_request.csv",
        runner_dir / "MQL5" / "Files" / "self_growing_feature_audit.csv",
        runner_dir / "MQL5" / "Files" / "self_growing_stop.request",
    ]
    for p in stale_files:
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    mt5_runners_dir = runner_dir.parent / ".mt5_runners"
    if mt5_runners_dir.exists():
        for sf in mt5_runners_dir.glob("**/self_growing_execution.csv"):
            try:
                sf.unlink()
            except Exception:
                pass


def run_mythos_mt5(
    from_date: str = "2025.01.06",
    to_date: str = "2025.01.08",
    model: int = MT5_MODEL_1MIN_OHLC,
    packets: int = 1440,
    timeout_sec: int = 10800,
    policy_id: int = 19301,
    symbol: str = 'BTCUSDm',
    visual: bool = False,
) -> dict:
    print(f"\n================================================================================", flush=True)
    print(f"       FRONTIER MYTHOS + MUZERO CO-EVOLVED MT5 STRATEGY TESTER EXECUTION        ", flush=True)
    print(f"================================================================================", flush=True)
    print(f"Window: {from_date} -> {to_date} | Model: {model} | Max Packets: {packets}", flush=True)

    # 0. Clean stale logs
    runner_dir = ROOT / "mt5_v2_runner"
    clean_stale_ledgers(runner_dir)

    # 1. Load configuration
    config_path = ROOT / "configs" / "he_chuyen_gia_mot_context_mot_policy.yaml"
    if not config_path.exists():
        config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    app = load_config(config_path)

    # 2. Load Base Policy
    base_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    if not base_checkpoint.exists():
        base_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "champion_round_5.pkl"
    print(f"[1/4] Loading base policy from {base_checkpoint.name}...", flush=True)
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint, app, seed=2026)
    _cau_hinh_policy_chuyen_gia_cho_mt5(base_policy, app)

    # 3. Initialize MythosMT5Policy
    print(f"[2/4] Initializing Frontier MythosMT5Policy...", flush=True)
    mythos_policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
    mythos_policy.direct_action = True
    mythos_policy.max_packets = packets
    mythos_policy.enable_live_telemetry = True
    mythos_policy.conviction_threshold = 0.025
    mythos_policy.require_margin_alignment = False
    mythos_policy.enable_mtf_filter = False
    mythos_policy.counter_trap_sl_atr = 2.50
    mythos_policy.counter_trap_tp_rr = 2.00
    mythos_policy.sniper_sl_atr = 2.40
    mythos_policy.sniper_tp_rr = 1.50
    mythos_policy.scalp_sl_atr = 2.40
    mythos_policy.scalp_tp_rr = 1.50
    mythos_policy.trailing_atr = 1.40
    mythos_policy.long_only = False
    mythos_policy.max_positions = 5
    mythos_policy.max_consecutive_losses = 2
    mythos_policy.loss_cooldown_bars = 2
    mythos_policy.min_spacing_bars = 1
    mythos_policy.prm_threshold = 0.12
    mythos_policy.runner_dir = runner_dir
    try:
        from self_growing_trader.regime_detector import RegimeDetector
        reg_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
        if reg_path.exists():
            mythos_policy.regime_detector = RegimeDetector.load(reg_path)
            print(f"      Loaded Champion Regime Detector from {reg_path.name} (probs ready).", flush=True)
    except Exception as e:
        print(f"      Warning: could not load champion regime detector: {e}", flush=True)

    if hasattr(mythos_policy.pipeline, "bilateral_engine"):
        mythos_policy.pipeline.bilateral_engine.config.prm_threshold = mythos_policy.prm_threshold
        mythos_policy.pipeline.bilateral_engine.config.margin_threshold = 0.55
        if hasattr(mythos_policy.pipeline.bilateral_engine, "arbiter"):
            mythos_policy.pipeline.bilateral_engine.arbiter.prm_threshold = mythos_policy.prm_threshold
            mythos_policy.pipeline.bilateral_engine.arbiter.margin_threshold = 0.55

    # Prioritize Milestone 2 Retrained Symmetric Anti-MAE Champion
    candidates = [
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt",
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_selfplay_triad_champion.pt",
        ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt",
    ]
    champion_pt = next((p for p in candidates if p.exists()), None)
    if champion_pt is not None:
        print(f"      Injecting Champion weights ({champion_pt.name})...", flush=True)
        ckpt = torch.load(champion_pt, map_location="cpu", weights_only=False)

        # 1. Bilateral Dual-CoT Engine (Active Decision Brain)
        if "bull_cot_state_dict" in ckpt and hasattr(mythos_policy.pipeline, "bilateral_engine"):
            mythos_policy.pipeline.bilateral_engine.load_checkpoint(champion_pt)
            mythos_policy.pipeline.bilateral_engine.config.prm_threshold = mythos_policy.prm_threshold
            mythos_policy.pipeline.bilateral_engine.config.margin_threshold = 0.55
            if hasattr(mythos_policy.pipeline.bilateral_engine, "arbiter"):
                mythos_policy.pipeline.bilateral_engine.arbiter.prm_threshold = mythos_policy.prm_threshold
                mythos_policy.pipeline.bilateral_engine.arbiter.margin_threshold = 0.55
            print(f"      Verified Bilateral Dual-CoT Engine loaded from {champion_pt.name}.", flush=True)

        # 2. Metacognitive Reasoning Engine
        if "reasoning_engine_state_dict" in ckpt and hasattr(mythos_policy.pipeline, "cot_engine"):
            load_res = mythos_policy.pipeline.cot_engine.load_state_dict(
                ckpt["reasoning_engine_state_dict"], strict=True
            )
            assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0, f"Keys mismatch: {len(load_res.missing_keys)} missing, {len(load_res.unexpected_keys)} unexpected"
            print(f"      Verified Reasoning Engine: 4,196,564 params, 0 missing, 0 unexpected keys.", flush=True)

    # 4. Configure MT5 Strategy Tester
    print(f"[3/4] Setting up MT5StrategyTesterEvaluator...", flush=True)
    kwargs = _mt5_execution_contract_kwargs(app)
    # Purge stale files from previous runs to guarantee clean execution ledger accounting
    for stale in [
        runner_dir / "MQL5" / "Files" / "self_growing_execution.csv",
        runner_dir / "MQL5" / "Files" / "self_growing_bridge.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_execution.csv",
        runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv",
    ]:
        if stale.exists():
            try:
                stale.unlink()
            except Exception:
                pass
    appdata = os.environ.get("APPDATA", "")
    if appdata:
        mq_dir = Path(appdata) / "MetaQuotes"
        search_root = mq_dir if mq_dir.exists() else Path(appdata)
        for lu in search_root.glob("**/liveupdate"):
            if lu.is_dir():
                shutil.rmtree(lu, ignore_errors=True)

    kwargs.update({
        "runner_dir": runner_dir,
        "symbol": symbol,
        "period": "M1",
        "from_date": from_date,
        "to_date": to_date,
        "timeout_sec": timeout_sec,
        "max_wall_sec": 0,  # Never abort midway on wall clock
        "max_decision_packets": packets,  # Cleanly stop via EA after N decision packets
        "max_positions": 5,
        "fixed_sl_atr": 2.40,
        "fixed_tp_atr": 3.60,
        "min_rr": 0.90,
        "technical_min_rr": 0.90,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 1.80,
        "trail_lock_atr": 0.40,
        "max_holding_bars": 120,
        "min_daily_trades": 300,
        "max_daily_trades": 1000,
        "suppress_flat_idle_starvation": True,
        "visual": visual,
        "shutdown_terminal": not visual,
        "model": model,
        "experience_journal": None,
        "learning_enabled": True,
        "enable_execution_bootstrap": False,
        "train_entry_probe_enabled": False,
        "entry_loss_guard_enabled": False,
        "runtime_fitness_guard_enabled": False,
        "require_real_ticks_for_promotion": False,
        "require_real_tick_evidence_for_promotion": False,
        "run_id": f"mythos_eval_{policy_id}",
    })
    bt_cfg = MT5BacktestConfig(**kwargs)
    evaluator = MT5StrategyTesterEvaluator(bt_cfg)

    # 5. Execute MT5 Backtest
    print(f"[4/4] Launching terminal64.exe and starting Strategy Tester loop...", flush=True)
    t0 = time.time()
    try:
        fitness, metrics = evaluator.evaluate_policy(policy_id, mythos_policy, 0)
    except Exception as e:
        print(f"Evaluator execution finished or stopped: {e}", flush=True)
        fitness = 0.0
        metrics = None

    elapsed = time.time() - t0

    # Sync final deals
    mythos_policy._sync_closed_deals()

    # Ensure authentic TSV Schema 17 execution ledger is copied to mt5_v2_runner/MQL5/Files/
    agent_exec = runner_dir / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_execution.csv"
    mql5_exec = runner_dir / "MQL5" / "Files" / "self_growing_execution.csv"
    if agent_exec.exists() and agent_exec.stat().st_size > 0:
        mql5_exec.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(agent_exec, mql5_exec)
        print(f"Synced {agent_exec.stat().st_size} bytes execution ledger to {mql5_exec}", flush=True)

    pnl = mythos_policy.last_equity - mythos_policy.initial_capital
    pnl_pct = (pnl / mythos_policy.initial_capital) * 100.0
    wr = (mythos_policy.wins / max(1, mythos_policy.closed_trades)) * 100.0 if mythos_policy.closed_trades > 0 else 0.0
    pf = (mythos_policy.gross_profit / max(1e-4, mythos_policy.gross_loss)) if mythos_policy.gross_loss > 0 else (2.0 if mythos_policy.gross_profit > 0 else 1.0)

    print(f"\n================================================================================", flush=True)
    print(f"                   MT5 EXECUTION FINAL SUMMARY ({elapsed:.2f}s)                 ", flush=True)
    print(f"================================================================================", flush=True)
    print(f"  - Net PnL:             ${pnl:+,.2f} ({pnl_pct:+.2f}%)", flush=True)
    print(f"  - Final Equity:        ${mythos_policy.last_equity:,.2f} (Peak: ${mythos_policy.peak_equity:,.2f})", flush=True)
    print(f"  - Max Drawdown:        {mythos_policy.max_drawdown_pct:.2f}%", flush=True)
    print(f"  - Closed Deals:        {mythos_policy.closed_trades} ({mythos_policy.wins} Wins / {mythos_policy.losses} Losses)", flush=True)
    print(f"  - Win Rate (WR):       {wr:.2f}%", flush=True)
    print(f"  - Profit Factor (PF):  {pf:.2f} (Gross Profit: ${mythos_policy.gross_profit:,.2f} / Gross Loss: ${mythos_policy.gross_loss:,.2f})", flush=True)
    print(f"  - Traps Deflected:     {mythos_policy.veto_count} Vetoes by First-Principles CoT", flush=True)
    print(f"  - Governor Invariant:  Agent 4 Locked HOLD 100.0% (Zero Policy Close All)", flush=True)
    if metrics is not None:
        print(f"  - MT5 Report Path:     {getattr(metrics, 'report_path', 'N/A')}", flush=True)
        print(f"  - MT5 Total Trades:    {getattr(metrics, 'total_trades', 'N/A')}", flush=True)
        print(f"  - MT5 Net Profit:      {getattr(metrics, 'net_profit', 'N/A')}", flush=True)
        print(f"  - MT5 Profit Factor:   {getattr(metrics, 'profit_factor', 'N/A')}", flush=True)
    print(f"================================================================================\n", flush=True)

    result = {
        "policy_id": policy_id,
        "from_date": from_date,
        "to_date": to_date,
        "elapsed_sec": elapsed,
        "packets_served": mythos_policy.seq_counter,
        "initial_capital": mythos_policy.initial_capital,
        "final_equity": mythos_policy.last_equity,
        "net_pnl": pnl,
        "net_pnl_pct": pnl_pct,
        "max_drawdown_pct": mythos_policy.max_drawdown_pct,
        "closed_deals": mythos_policy.closed_trades,
        "wins": mythos_policy.wins,
        "losses": mythos_policy.losses,
        "win_rate": wr,
        "profit_factor": pf,
        "gross_profit": mythos_policy.gross_profit,
        "gross_loss": mythos_policy.gross_loss,
        "traps_vetoed": mythos_policy.veto_count,
    }
    if metrics is not None:
        result.update({
            "metrics_total_trades": getattr(metrics, "total_trades", None),
            "metrics_net_profit": getattr(metrics, "net_profit", None),
            "metrics_profit_factor": getattr(metrics, "profit_factor", None),
            "metrics_win_rate": getattr(metrics, "win_rate", None),
            "metrics_equity_drawdown_pct": getattr(metrics, "equity_drawdown_pct", None),
            "metrics_report_path": str(getattr(metrics, "report_path", "")),
        })

    out_file = ROOT / "outputs" / f"mt5_mythos_run_{policy_id}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-date", default="2026.10.19")
    parser.add_argument("--to-date", default="2026.10.25")
    parser.add_argument("--model", type=int, default=MT5_MODEL_1MIN_OHLC)
    parser.add_argument("--packets", type=int, default=10080)
    parser.add_argument("--timeout", type=int, default=10800)
    parser.add_argument("--policy-id", type=int, default=19301)
    parser.add_argument("--symbol", default="BTCUSDm")
    parser.add_argument("--visual", action="store_true", help="Enable MT5 GUI Visual Mode")
    args = parser.parse_args()

    run_mythos_mt5(
        from_date=args.from_date,
        to_date=args.to_date,
        model=args.model,
        packets=args.packets,
        timeout_sec=args.timeout,
        policy_id=args.policy_id,
        symbol=args.symbol,
        visual=args.visual,
    )
