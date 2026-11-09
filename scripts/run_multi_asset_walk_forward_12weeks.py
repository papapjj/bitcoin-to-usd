#!/usr/bin/env python3
"""
12-Week Rolling Multi-Asset Walk-Forward Orchestrator with Bilateral Co-Evolution.

Period: 2024.10.01 -> 2025.01.01 (Covering US Presidential Election & Bitcoin ATH Surge)
Target Invariants:
  - Multi-Asset Frequency: > 300 trades/day across high-liquidity universe
  - Target Win Rate: > 80.0%
  - Target Profit Factor: > 5.0
  - Max Drawdown: <= 10.0%
  - Anti-Collusion Invariant: Minimax Zero-sum (R_adv = -R_c) + loss_inactivity penalty
  - Safety Invariant: 100.0% Governor HOLD (POLICY_CLOSE_ALL == 0.0)
  - Latency SLA: <= 50.0 ms/tick
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
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
sys.path.insert(0, str(ROOT / "src"))

WEEKS_SCHEDULE = [
    {"week": 1,  "from_date": "2024.10.01", "to_date": "2024.10.07", "label": "Q4 Early Consolidation"},
    {"week": 2,  "from_date": "2024.10.07", "to_date": "2024.10.14", "label": "Mid-October Accumulation"},
    {"week": 3,  "from_date": "2024.10.14", "to_date": "2024.10.21", "label": "Pre-Election Positioning"},
    {"week": 4,  "from_date": "2024.10.21", "to_date": "2024.10.28", "label": "Volatility Compression"},
    {"week": 5,  "from_date": "2024.10.28", "to_date": "2024.11.04", "label": "Pre-Election Tension"},
    {"week": 6,  "from_date": "2024.11.04", "to_date": "2024.11.11", "label": "US Presidential Election Regime"},
    {"week": 7,  "from_date": "2024.11.11", "to_date": "2024.11.18", "label": "Post-Election Bull Surge"},
    {"week": 8,  "from_date": "2024.11.18", "to_date": "2024.11.25", "label": "Euphoria Continuation"},
    {"week": 9,  "from_date": "2024.11.25", "to_date": "2024.12.02", "label": "Thanksgiving Liquidity Sweep"},
    {"week": 10, "from_date": "2024.12.02", "to_date": "2024.12.09", "label": "Bitcoin Historic ATH Test"},
    {"week": 11, "from_date": "2024.12.09", "to_date": "2024.12.16", "label": "High-Altitude Volatility"},
    {"week": 12, "from_date": "2024.12.16", "to_date": "2024.12.23", "label": "Year-End Institutional Rebalance"},
]


def run_week_evaluation(
    week_info: dict,
    symbol: str = "BTCUSDm",
    packets: int = 150,
    timeout: int = 300,
    policy_id: int = 19301,
    symbol_idx: int = 0,
) -> dict:
    """Run MT5 Strategy Tester for a specific weekly window."""
    w = week_info["week"]
    from_d = week_info["from_date"]
    to_d = week_info["to_date"]
    label = week_info["label"]
    eff_policy_id = policy_id + w + symbol_idx * 100

    print(f"\n================================================================================")
    print(f" >>> EXECUTING WEEK {w:02d}/12 [{from_d} -> {to_d}] :: {label} [{symbol}] (Policy: {eff_policy_id})")
    print(f"================================================================================")

    cmd = [
        sys.executable,
        "-u",
        str(ROOT / "scripts" / "run_mythos_mt5_live.py"),
        "--from-date", from_d,
        "--to-date", to_d,
        "--packets", str(packets),
        "--timeout", str(timeout),
        "--policy-id", str(eff_policy_id),
        "--symbol", symbol,
    ]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0

    # Parse stdout for MT5 results
    output = proc.stdout
    print(f"Week {w} [{symbol}] completed in {elapsed:.1f}s (Exit code: {proc.returncode})")

    # Read latest report if available
    report_dir = ROOT / "mt5_v2_runner" / "Reports"
    latest_htm = None
    if report_dir.exists():
        htms = sorted(report_dir.glob(f"*_{eff_policy_id}.htm"), key=lambda p: p.stat().st_mtime, reverse=True)
        if htms:
            latest_htm = htms[0]
        else:
            all_htms = sorted(report_dir.glob("*.htm"), key=lambda p: p.stat().st_mtime, reverse=True)
            if all_htms:
                latest_htm = all_htms[0]

    # Check for run metrics json
    run_json = ROOT / "outputs" / f"mt5_mythos_run_{eff_policy_id}.json"
    manifest_data = {}
    if run_json.exists():
        try:
            manifest_data = json.loads(run_json.read_text(encoding="utf-8"))
        except Exception:
            pass

    return {
        "week": w,
        "from_date": from_d,
        "to_date": to_d,
        "label": label,
        "symbol": symbol,
        "elapsed_sec": elapsed,
        "exit_code": proc.returncode,
        "manifest": manifest_data,
        "report_htm": str(latest_htm) if latest_htm else "",
    }


def run_coevolution_fine_tuning(epochs: int = 5) -> bool:
    """Run Frontier Self-Play Triad Co-Evolution training."""
    print(f"\n[CO-EVOLUTION] Fine-tuning Champion Triad ({epochs} epochs)...")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_frontier_selfplay_triad.py"),
        "--epochs", str(epochs),
        "--save-checkpoint",
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if res.returncode == 0:
        print("  -> Co-Evolution successfully completed and champion weights saved.")
        return True
    else:
        print(f"  -> Co-Evolution encountered warning/error: {res.stderr[:300]}")
        return False


def run_m4_verification() -> bool:
    """Verify M4 Invariants (Governor Safety, Latency, SHA-256)."""
    print(f"\n[AUDIT] Running M4 Invariants Verification...")
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "verify_m4_invariants.py"),
    ]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(res.stdout)
    return res.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="12-Week Multi-Asset Walk-Forward Orchestrator")
    parser.add_argument("--start-week", type=int, default=1, help="Starting week (1-12)")
    parser.add_argument("--end-week", type=int, default=12, help="Ending week (1-12)")
    parser.add_argument("--symbols", type=str, default="BTCUSDm", help="Comma-separated symbols")
    parser.add_argument("--packets", type=int, default=150, help="Packets per evaluation")
    parser.add_argument("--epochs", type=int, default=5, help="Co-evolution epochs per cycle")
    parser.add_argument("--skip-coevolution", action="store_true", help="Skip training between weeks")
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_dir = ROOT / "outputs" / "walk_forward_multi_asset_12w"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_records = []

    print("\n" + "=" * 80)
    print("      12-WEEK MULTI-ASSET ROLLING WALK-FORWARD PIPELINE STARTING      ")
    print(f"      Universe: {symbols} | Weeks: {args.start_week} -> {args.end_week}")
    print("=" * 80)

    for week_cfg in WEEKS_SCHEDULE:
        w_num = week_cfg["week"]
        if w_num < args.start_week or w_num > args.end_week:
            continue

        week_results = []
        for s_idx, sym in enumerate(symbols):
            res = run_week_evaluation(
                week_info=week_cfg,
                symbol=sym,
                packets=args.packets,
                symbol_idx=s_idx,
            )
            week_results.append(res)

        # Co-Evolution Adaptation
        if not args.skip_coevolution:
            train_ok = run_coevolution_fine_tuning(epochs=args.epochs)
        else:
            train_ok = True

        # Audit Invariants
        audit_ok = run_m4_verification()

        week_record = {
            "week": w_num,
            "config": week_cfg,
            "results": week_results,
            "coevolution_ok": train_ok,
            "audit_ok": audit_ok,
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }
        summary_records.append(week_record)

        # Write progressive ledger
        ledger_path = out_dir / f"walk_forward_progress_w{w_num:02d}.json"
        ledger_path.write_text(json.dumps(week_record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[CHECKPOINT] Week {w_num} progress committed to {ledger_path.name}")

    # Master Report
    master_path = out_dir / "walk_forward_12weeks_master_manifest.json"
    master_path.write_text(json.dumps({
        "status": "COMPLETED",
        "total_weeks": len(summary_records),
        "records": summary_records,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("      12-WEEK MULTI-ASSET ROLLING WALK-FORWARD PIPELINE FINISHED      ")
    print(f"      Master Manifest: {master_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
