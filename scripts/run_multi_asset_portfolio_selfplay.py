#!/usr/bin/env python3
"""
scripts/run_multi_asset_portfolio_selfplay.py

Multi-Asset Portfolio Self-Play In-The-Loop Runner
Focusing on high-liquidity non-currency assets:
- XAUUSDm (Gold)
- BTCUSDm (Bitcoin)
- ETHUSDm (Ethereum)
- BCHUSDm (Bitcoin Cash)

Features:
1. Online In-the-Loop Self-Play (SPIN + MAD Reflexion + STaR on MT5 real ticks).
2. Fully Autonomous Policy (neural network dynamically determines entry/exit/risk).
3. Concurrent position risk management: 0.25%-0.50% equity risk per trade.
4. Comprehensive portfolio manifest generation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import subprocess
import sys
import time

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from self_growing_trader.mt5_runner import parse_mt5_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiAssetSelfPlayPortfolio")

PORTFOLIO_UNIVERSE = ["XAUUSDm", "BTCUSDm", "ETHUSDm", "BCHUSDm"]


def run_symbol_evaluation(
    symbol: str,
    from_date: str,
    to_date: str,
    packets: int = 100,
    policy_id: int = 19701,
    timeout: int = 300,
) -> dict:
    """Run MT5 Strategy Tester for a single symbol with in-the-loop self-play."""
    print(f"\n" + "=" * 80)
    print(f" >>> RUNNING EVALUATION: {symbol} [{from_date} -> {to_date}] (Policy: {policy_id})")
    print(f"=" * 80)

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_mythos_mt5_live.py"),
        "--from-date", from_date,
        "--to-date", to_date,
        "--symbol", symbol,
        "--packets", str(packets),
        "--timeout", str(timeout),
        "--policy-id", str(policy_id),
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0

    print(f"Finished {symbol} in {elapsed:.1f}s (Exit Code: {proc.returncode})")

    # Locate report
    report_path = ROOT / "mt5_v2_runner" / "Reports" / f"he_chuyen_gia_mt5_round_0000_evaluation_{policy_id}.htm"
    metrics = {}
    if report_path.exists():
        try:
            m = parse_mt5_report(str(report_path))
            metrics = {
                "total_trades": m.total_trades,
                "net_profit": m.net_profit,
                "profit_factor": m.profit_factor,
                "win_rate": m.win_rate,
                "equity_drawdown_pct": m.equity_drawdown_pct,
                "avg_win": m.avg_win,
                "avg_loss": m.avg_loss,
            }
        except Exception as e:
            logger.warning("Could not parse MT5 report for %s: %s", symbol, e)

    # Read live json if present
    run_json = ROOT / "outputs" / f"mt5_mythos_run_{policy_id}.json"
    manifest_data = {}
    if run_json.exists():
        try:
            manifest_data = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "symbol": symbol,
        "policy_id": policy_id,
        "elapsed_sec": elapsed,
        "exit_code": proc.returncode,
        "metrics": metrics,
        "manifest": manifest_data,
        "report_htm": str(report_path) if report_path.exists() else "",
    }


def run_online_selfplay_cycle(epochs: int = 5) -> bool:
    """Execute Online In-the-Loop Self-Play Co-Evolution using latest trade evidence."""
    print(f"\n[ONLINE SELF-PLAY] Triggering Co-Evolution Fine-Tuning ({epochs} epochs)...")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_frontier_selfplay_triad.py"),
        "--epochs", str(epochs),
        "--ticks-per-epoch", "60",
        "--save-checkpoint",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        print("  -> Online Self-Play successfully completed and weights updated.")
        # Sync weights
        src = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_selfplay_triad_champion.pt"
        dst = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        if src.exists():
            import shutil
            shutil.copyfile(src, dst)
        return True
    else:
        print(f"  -> Online Self-Play encountered notice: {res.stderr[:200]}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Multi-Asset Non-Currency Portfolio Self-Play Runner")
    parser.add_argument("--from-date", type=str, default="2024.11.18", help="Start date (YYYY.MM.DD)")
    parser.add_argument("--to-date", type=str, default="2024.11.25", help="End date (YYYY.MM.DD)")
    parser.add_argument("--symbols", type=str, default="XAUUSDm,BTCUSDm,ETHUSDm,BCHUSDm", help="Comma-separated symbols")
    parser.add_argument("--packets", type=int, default=100, help="Packets per symbol evaluation")
    parser.add_argument("--selfplay-epochs", type=int, default=5, help="Self-play epochs between symbols")
    parser.add_argument("--base-policy-id", type=int, default=19701, help="Base policy ID")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_dir = ROOT / "outputs" / "multi_asset_portfolio_selfplay"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("      FRONTIER MULTI-ASSET PORTFOLIO ONLINE SELF-PLAY PIPELINE STARTING      ")
    print(f"      Universe: {symbols} (Non-Currency High Liquidity)")
    print(f"      Period:   {args.from_date} -> {args.to_date} | Packets: {args.packets}")
    print("=" * 90)

    portfolio_results = []
    total_trades = 0
    total_net_profit = 0.0
    wins = 0
    losses = 0

    for idx, sym in enumerate(symbols):
        policy_id = args.base_policy_id + idx * 10
        res = run_symbol_evaluation(
            symbol=sym,
            from_date=args.from_date,
            to_date=args.to_date,
            packets=args.packets,
            policy_id=policy_id,
        )
        portfolio_results.append(res)

        m = res.get("metrics", {})
        mf = res.get("manifest", {})

        sym_trades = m.get("total_trades", mf.get("closed_deals", 0))
        sym_pnl = m.get("net_profit", mf.get("net_pnl", 0.0))
        sym_wr = m.get("win_rate", mf.get("win_rate", 0.0))

        total_trades += sym_trades
        total_net_profit += sym_pnl

        print(f"  -> {sym}: Trades={sym_trades}, Net PnL={sym_pnl:+.2f} USD, WR={sym_wr:.1f}%")

        # Online in-the-loop self-play adaptation between symbols
        if args.selfplay_epochs > 0:
            run_online_selfplay_cycle(epochs=args.selfplay_epochs)

    # Calculate portfolio aggregates
    avg_wr = 0.0
    if total_trades > 0:
        weighted_wr = sum(
            r.get("metrics", {}).get("win_rate", 0.0) * r.get("metrics", {}).get("total_trades", 0)
            for r in portfolio_results
        )
        avg_wr = weighted_wr / total_trades

    # Estimate daily rate (assuming 5 trading days per week)
    trades_per_day = round(total_trades / 5.0, 1)

    portfolio_manifest = {
        "title": "Frontier Multi-Asset Non-Currency Portfolio Self-Play Benchmark",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "from_date": args.from_date,
        "to_date": args.to_date,
        "symbols": symbols,
        "summary": {
            "total_trades": total_trades,
            "estimated_daily_trades": trades_per_day,
            "total_net_profit": round(total_net_profit, 2),
            "portfolio_win_rate": round(avg_wr, 2),
        },
        "results": portfolio_results,
    }

    manifest_file = out_dir / "multi_asset_portfolio_master_manifest.json"
    manifest_file.write_text(json.dumps(portfolio_manifest, indent=2), encoding="utf-8")

    print("\n" + "=" * 90)
    print("      PORTFOLIO MULTI-ASSET BENCHMARK COMPLETED")
    print(f"      Total Trades:       {total_trades} (Est. {trades_per_day} trades/day)")
    print(f"      Total Net Profit:   {total_net_profit:+.2f} USD")
    print(f"      Portfolio Win Rate: {avg_wr:.2f}%")
    print(f"      Master Manifest:    {manifest_file}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    main()
