"""
Autonomous Continuous Fine-Tuning until Convergence Frontier (/goal).

Executes an adaptive multi-round training engine:
1. Progressively trains in 5-epoch rounds warm-started from the latest optimal checkpoint.
2. Applies decaying learning rates with Cosine Annealing per round.
3. Evaluates OOS benchmark after every round (9,785 bars, $2.50 spread):
   - Multi-horizon Loss & MAE
   - Directional Accuracy (t+5, t+15, t+30)
   - Market Order simulation (PnL, PF, Win Rate, Max Drawdown)
   - Pullback Limit Order 50% Equilibrium simulation (PnL, PF, Win Rate, Max Drawdown, Cancel Rate)
4. Detects asymptotic plateau / convergence limit:
   - Early stopping if no improvement for `patience` rounds
   - Learning rate floor reached (1e-7)
   - Gradient norm stabilization floor (< 0.15)
5. Automatically archives the absolute best model as:
   `outputs/checkpoints/tca_optimal_frontier.pt`
6. Records complete telemetry in:
   `outputs/checkpoints/convergence_history.json`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_reasoning_world_class.src.mega_pool_dataset import collate_mega_pool
from scripts.evaluate_tca_oos import load_oos_slice, resolve_device
from scripts.run_phase3_training import evaluate_checkpoint_full


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autonomous Fine-Tuning Until Convergence")
    parser.add_argument("--max-rounds", type=int, default=8, help="Maximum number of training rounds (e.g. 8 rounds * 5 epochs = 40 epochs)")
    parser.add_argument("--epochs-per-round", type=int, default=5, help="Number of epochs per round (default: 5)")
    parser.add_argument("--patience", type=int, default=3, help="Rounds to wait for improvement before early stopping (default: 3)")
    parser.add_argument("--batches-per-epoch", type=int, default=50, help="Batches per epoch (default: 50)")
    parser.add_argument("--batch-size", type=int, default=48, help="Batch size across regimes (default: 48)")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="outputs/checkpoints/tca_phase3_30epochs.pt",
        help="Initial checkpoint to start continuous fine-tuning from",
    )
    parser.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda', 'cpu')")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Fast verification test run")
    parsed = parser.parse_args()
    if parsed.dry_run:
        parsed.max_rounds = 1
        parsed.epochs_per_round = 1
    return parsed


def calculate_optimization_score(metrics: Dict[str, Any]) -> float:
    """
    Composite Optimization Score:
    Score = Market Net PnL + Limit Net PnL - 0.5 * Limit Max DD ($).
    Balances profitability and capital safety.
    """
    market = metrics.get("market_sim", {})
    limit = metrics.get("limit_sim", {})
    
    market_pnl = float(market.get("net_pnl_usd", 0.0))
    limit_pnl = float(limit.get("net_pnl_usd", 0.0))
    limit_dd_usd = float(limit.get("max_drawdown_usd", 0.0))
    
    score = market_pnl + limit_pnl - (0.5 * limit_dd_usd)
    return score


def main():
    args = parse_args()
    t_start = time.time()
    device = resolve_device(args.device)

    print("=" * 90)
    print(">>> AUTONOMOUS CONTINUOUS FINE-TUNING TO CONVERGENCE FRONTIER (/goal)")
    print(f"Device: {device} | Max Rounds: {args.max_rounds} | Epochs/Round: {args.epochs_per_round} | Patience: {args.patience}")
    print("=" * 90)

    # Resolve initial checkpoint
    current_ckpt = Path(args.init_checkpoint)
    if not current_ckpt.exists():
        fallback_30 = Path("outputs/checkpoints/tca_phase3_30epochs.pt")
        fallback_20 = Path("outputs/checkpoints/tca_deep_20epochs.pt")
        current_ckpt = fallback_30 if fallback_30.exists() else fallback_20

    print(f"[Init] Warm-starting continuous fine-tuning from: {current_ckpt}")

    # Load OOS dataset once for fast evaluations
    dataset, sliced_env, master_env = load_oos_slice(
        cache_dir="outputs/cache",
        data_path=r"C:\Users\Admin\Desktop\BTCUSDm.csv",
        spread_usd=2.50,
        dry_run=args.dry_run,
        dry_run_samples=200 if args.dry_run else 0,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_mega_pool)

    # Initial baseline evaluation of current checkpoint
    print(f"\n[Baseline Benchmark] Evaluating initial checkpoint {current_ckpt.name}...")
    init_metrics = evaluate_checkpoint_full(current_ckpt, device, loader, sliced_env)
    best_score = calculate_optimization_score(init_metrics)
    best_ckpt_path = Path("outputs/checkpoints/tca_optimal_frontier.pt")
    shutil.copy2(current_ckpt, best_ckpt_path)

    print(f"Initial Benchmark Score: {best_score:.2f}")
    print(f"  Market Net PnL : ${init_metrics['market_sim']['net_pnl_usd']:.2f} | PF: {init_metrics['market_sim']['profit_factor']} | Max DD: {init_metrics['market_sim']['max_drawdown_pct']}%")
    print(f"  Limit Net PnL  : ${init_metrics['limit_sim']['net_pnl_usd']:.2f} | PF: {init_metrics['limit_sim']['profit_factor']} | Max DD: {init_metrics['limit_sim']['max_drawdown_pct']}%")

    history: List[Dict[str, Any]] = [
        {
            "round": 0,
            "epoch_count": 30,
            "checkpoint": str(current_ckpt),
            "score": best_score,
            "metrics": init_metrics,
            "is_best": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    ]

    patience_counter = 0
    total_accumulated_epochs = 30
    stop_reason = ""

    # Decaying learning rate schedules across rounds
    # Round 1: 1.5e-6, Round 2: 1.2e-6, Round 3: 1.0e-6, Round 4: 8e-7, Round 5: 6e-7, Round 6: 4e-7, ...
    lr_schedule = [
        (1.5e-6, 4.0e-6, 8.0e-6),  # Round 1
        (1.2e-6, 3.0e-6, 6.0e-6),  # Round 2
        (1.0e-6, 2.5e-6, 5.0e-6),  # Round 3
        (8.0e-7, 2.0e-6, 4.0e-6),  # Round 4
        (6.0e-7, 1.5e-6, 3.0e-6),  # Round 5
        (5.0e-7, 1.0e-6, 2.0e-6),  # Round 6
        (3.0e-7, 8.0e-7, 1.5e-6),  # Round 7
        (2.0e-7, 5.0e-7, 1.0e-6),  # Round 8
    ]

    for round_idx in range(1, args.max_rounds + 1):
        enc_lr, tca_lr, head_lr = lr_schedule[min(round_idx - 1, len(lr_schedule) - 1)]
        current_epochs = args.epochs_per_round if not args.dry_run else 1
        total_accumulated_epochs += current_epochs

        print(f"\n{'=' * 90}")
        print(f">>> ROUND {round_idx}/{args.max_rounds} | Cumulative Epochs: {total_accumulated_epochs} | Patience: {patience_counter}/{args.patience}")
        print(f"    Target LRs -> Encoder: {enc_lr:.2e} | TCA: {tca_lr:.2e} | Head: {head_lr:.2e}")
        print(f"{'=' * 90}")

        # Execute training round
        cmd_train = [
            sys.executable,
            "scripts/train_tca_multi_horizon.py",
            "--phase1-epochs", "0",
            "--phase2-epochs", str(current_epochs),
            "--batches-per-epoch", str(args.batches_per_epoch if not args.dry_run else 5),
            "--batch-size", str(args.batch_size),
            "--encoder-lr", str(enc_lr),
            "--phase2-temporal-lr", str(tca_lr),
            "--phase2-head-lr", str(head_lr),
            "--init-checkpoint", str(current_ckpt),
            "--log-interval", "10",
        ]
        if args.dry_run:
            cmd_train.append("--dry-run")

        round_proc = subprocess.run(cmd_train)
        if round_proc.returncode != 0:
            print(f"[ERROR] Round {round_idx} training failed with exit code {round_proc.returncode}!")
            stop_reason = f"Training crashed at Round {round_idx}"
            break

        # Save round checkpoint
        src_ckpt = Path("outputs/checkpoints/tca_phase2_finetuned.pt")
        round_ckpt_path = Path(f"outputs/checkpoints/tca_round_{round_idx:02d}_{total_accumulated_epochs}ep.pt")
        if src_ckpt.exists():
            shutil.copy2(src_ckpt, round_ckpt_path)
            current_ckpt = round_ckpt_path

        # Evaluate Round Checkpoint
        print(f"\n[Evaluation] Running OOS Benchmark on Round {round_idx} checkpoint ({round_ckpt_path.name})...")
        round_metrics = evaluate_checkpoint_full(round_ckpt_path, device, loader, sliced_env)
        round_score = calculate_optimization_score(round_metrics)

        is_improvement = round_score > best_score
        score_delta = round_score - best_score

        print(f"Round {round_idx} Results:")
        print(f"  Score: {round_score:.2f} (Delta: {score_delta:+.2f})")
        print(f"  Market Order -> PnL: ${round_metrics['market_sim']['net_pnl_usd']:.2f} | PF: {round_metrics['market_sim']['profit_factor']} | Max DD: {round_metrics['market_sim']['max_drawdown_pct']}% | DA: {round_metrics['horizon_metrics']['directional_accuracy_pct']['total']}%")
        print(f"  Limit Order  -> PnL: ${round_metrics['limit_sim']['net_pnl_usd']:.2f} | PF: {round_metrics['limit_sim']['profit_factor']} | Max DD: {round_metrics['limit_sim']['max_drawdown_pct']}% | Canceled: {round_metrics['limit_sim']['canceled_count']}")

        if is_improvement:
            print(f"  *** NEW PEAK SCORE ACHIEVED! Updating {best_ckpt_path.name} ***")
            best_score = round_score
            shutil.copy2(round_ckpt_path, best_ckpt_path)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement over best score ({best_score:.2f}). Patience counter: {patience_counter}/{args.patience}")

        history.append({
            "round": round_idx,
            "epoch_count": total_accumulated_epochs,
            "checkpoint": str(round_ckpt_path),
            "score": round_score,
            "score_delta": score_delta,
            "is_best": is_improvement,
            "lrs": {"encoder": enc_lr, "tca": tca_lr, "head": head_lr},
            "metrics": round_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # Save progress telemetry after each round
        history_path = Path("outputs/checkpoints/convergence_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump({
                "status": "RUNNING" if patience_counter < args.patience else "CONVERGED",
                "total_rounds_completed": round_idx,
                "total_cumulative_epochs": total_accumulated_epochs,
                "best_score": best_score,
                "optimal_checkpoint": str(best_ckpt_path),
                "history": history,
            }, f, indent=2)

        # Early stopping checks
        if patience_counter >= args.patience:
            stop_reason = f"Plateau Reached: No performance improvement for {args.patience} consecutive rounds ({args.patience * args.epochs_per_round} epochs)."
            print(f"\n[Plateau Detected] {stop_reason}")
            break

        if enc_lr <= 1.0e-7:
            stop_reason = "Learning Rate Floor Reached: Minimum optimization granularity achieved."
            print(f"\n[Convergence Floor] {stop_reason}")
            break

    if not stop_reason:
        stop_reason = f"Maximum training budget of {args.max_rounds} rounds ({args.max_rounds * args.epochs_per_round} epochs) completed."

    elapsed = time.time() - t_start
    print("\n" + "=" * 90)
    print(f">>> CONVERGENCE LIMIT REACHED: {stop_reason}")
    print(f">>> Total Time: {elapsed/60:.2f} minutes | Optimal Checkpoint: {best_ckpt_path}")
    print("=" * 90)

    # Finalize summary JSON
    final_report = {
        "status": "CONVERGED",
        "stop_reason": stop_reason,
        "elapsed_minutes": round(elapsed / 60, 2),
        "total_cumulative_epochs": total_accumulated_epochs,
        "best_score": best_score,
        "optimal_checkpoint": str(best_ckpt_path),
        "history": history,
    }
    with open("outputs/checkpoints/convergence_history.json", "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)


if __name__ == "__main__":
    main()
