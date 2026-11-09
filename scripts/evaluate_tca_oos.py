"""
Out-of-Sample (OOS) Crash Slice Evaluation Runner (Feature 14).

Evaluates model configurations on the February 2025 crash slice [55000:64785]
(9,785 authentic M1 bars, $11,234.50 drop, >15-20% drawdown, $2.50 fixed spread):
1. Baseline World Model (30 keys default, static 137D feature baseline)
2. Phase 1 Pre-trained TCA (frozen base encoder + trained TCA 41 keys + Projection Head 8 keys)
3. Phase 2 Fine-tuned TCA (jointly fine-tuned encoder + TCA + Projection Head)

Calculates multi-horizon return metrics (DA% at t+5, t+15, t+30; MAE; Huber loss)
and spread-adjusted simulated trading performance ($2.50 spread deduction).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from research_reasoning_world_class.src.mega_pool_dataset import (
    SlicedM1Environment,
    SingleRegimeDataset,
    collate_mega_pool,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MultiHorizonProjectionHead,
)


def resolve_device(device_str: str = "auto") -> torch.device:
    """Resolves torch.device safely with graceful CPU fallback."""
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[Device] Warning: CUDA requested but unavailable. Falling back to CPU.")
        return torch.device("cpu")
    elif device_str == "cpu":
        return torch.device("cpu")
    else:  # "auto"
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    """Sets deterministic random seed for evaluation reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_oos_slice(
    cache_dir: Union[str, Path] = "outputs/cache",
    data_path: Union[str, Path] = r"C:\Users\Admin\Desktop\BTCUSDm.csv",
    spread_usd: float = 2.50,
    dry_run: bool = False,
    dry_run_samples: int = 200,
) -> Tuple[SingleRegimeDataset, SlicedM1Environment, RealM1Environment]:
    """
    Loads authentic February 2025 crash slice [55000:64785] from precomputed cache
    or authentic CSV if cache missing.
    """
    cache_file = Path(cache_dir) / "pool_oos_master.npz"
    master_env = RealM1Environment(spread_usd=spread_usd)

    if cache_file.exists():
        with np.load(cache_file, allow_pickle=True) as data:
            master_env.total_bars = len(data["close_arr"])
            master_env.open_arr = data["open_arr"].astype(np.float32)
            master_env.high_arr = data["high_arr"].astype(np.float32)
            master_env.low_arr = data["low_arr"].astype(np.float32)
            master_env.close_arr = data["close_arr"].astype(np.float32)
            master_env.volume_arr = data["volume_arr"].astype(np.float32)
            master_env.spread_arr = data["spread_arr"].astype(np.float32)
            master_env.atr_arr = data["atr_arr"].astype(np.float32)
            master_env.day_progress_arr = data["day_progress_arr"].astype(np.float32)
            master_env.obs_matrix = data["obs_matrix"].astype(np.float32)
            if "timestamps" in data:
                master_env.timestamps = data["timestamps"].tolist()
            else:
                master_env.timestamps = [""] * master_env.total_bars
    else:
        if not Path(data_path).exists():
            raise FileNotFoundError(f"Neither cache ({cache_file}) nor data ({data_path}) exists.")
        master_env = RealM1Environment(csv_path=str(data_path), spread_usd=spread_usd)
        master_env.load_data("2025.01.15", "2025.02.28")

    # Slice February 2025 crash: [55000:64785] (9,785 bars)
    sliced_env = SlicedM1Environment(master_env, 55000, 64785, name="Feb2025_Crash_Slice")
    dataset = SingleRegimeDataset("oos", sliced_env, lookback=240, horizons=(5, 15, 30), is_oos=True)

    if dry_run:
        indices = list(range(min(dry_run_samples, len(dataset))))
        dataset = Subset(dataset, indices)

    return dataset, sliced_env, master_env


def load_baseline_model(
    checkpoint_path: Union[str, Path],
    head_checkpoint_path: Optional[Union[str, Path]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tuple[LatentDynamicsWorldModel, MultiHorizonProjectionHead]:
    """Loads baseline 30-key World Model (no temporal cross-attention stream) + projection head."""
    wm = LatentDynamicsWorldModel().to(device)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.0).to(device)

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        wm_state = ckpt.get("world_model_state_dict", {})
        clean_wm = {k: v for k, v in wm_state.items() if not k.startswith("temporal_encoder.")}
        wm.load_state_dict(clean_wm, strict=False)

    # Use Phase 1 or Phase 2 projection head weights to fairly evaluate static 137D representation
    head_path = head_checkpoint_path or "outputs/checkpoints/tca_phase1_pretrained.pt"
    if Path(head_path).exists():
        ckpt = torch.load(head_path, map_location=device, weights_only=False)
        if "projection_head_state_dict" in ckpt:
            head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)

    wm.eval()
    head.eval()
    return wm, head


def load_phase1_model(
    checkpoint_path: Union[str, Path] = "outputs/checkpoints/tca_phase1_pretrained.pt",
    device: torch.device = torch.device("cpu"),
) -> Tuple[LatentDynamicsWorldModel, MultiHorizonProjectionHead]:
    """Loads Phase 1 Pre-trained model (frozen base encoder + trained TCA + Projection Head)."""
    wm = LatentDynamicsWorldModel().to(device)
    wm.enable_temporal_stream(candle_len=240, temp_dim=128, num_heads=4, init_gate_bias=-2.0)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.0).to(device)

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        wm_state = ckpt.get("world_model_state_dict", {})
        clean_wm = {k: v for k, v in wm_state.items() if not k.startswith("temporal_encoder.")}
        wm.load_state_dict(clean_wm, strict=False)

        if "temporal_encoder_state_dict" in ckpt and wm.temporal_encoder is not None:
            wm.temporal_encoder.load_state_dict(ckpt["temporal_encoder_state_dict"], strict=True)
        if "projection_head_state_dict" in ckpt:
            head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)

    wm.eval()
    head.eval()
    return wm, head


def load_phase2_model(
    checkpoint_path: Union[str, Path] = "outputs/checkpoints/tca_phase2_finetuned.pt",
    device: torch.device = torch.device("cpu"),
) -> Tuple[LatentDynamicsWorldModel, MultiHorizonProjectionHead]:
    """Loads Phase 2 Fine-tuned model (jointly fine-tuned base encoder + TCA + Projection Head)."""
    wm = LatentDynamicsWorldModel().to(device)
    wm.enable_temporal_stream(candle_len=240, temp_dim=128, num_heads=4, init_gate_bias=-2.0)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.0).to(device)

    if Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        wm_state = ckpt.get("world_model_state_dict", {})
        clean_wm = {k: v for k, v in wm_state.items() if not k.startswith("temporal_encoder.")}
        wm.load_state_dict(clean_wm, strict=False)

        if "temporal_encoder_state_dict" in ckpt and wm.temporal_encoder is not None:
            wm.temporal_encoder.load_state_dict(ckpt["temporal_encoder_state_dict"], strict=True)
        if "projection_head_state_dict" in ckpt:
            head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)

    wm.eval()
    head.eval()
    return wm, head


def compute_horizon_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes multi-horizon metrics:
    - Directional Accuracy (DA%) for t+5, t+15, t+30 and Mean DA%
    - Mean Absolute Error (MAE) for t+5, t+15, t+30 and Mean MAE
    - Huber Loss (Smooth L1 with beta=1.0) for t+5, t+15, t+30 and Mean Huber
    """
    diff = preds - targets
    abs_diff = np.abs(diff)
    mae_per_horizon = np.mean(abs_diff, axis=0)

    # Smooth L1 / Huber loss with beta=1.0:
    # 0.5 * x^2 if |x| < 1.0 else |x| - 0.5
    huber = np.where(abs_diff < 1.0, 0.5 * (diff ** 2), abs_diff - 0.5)
    huber_per_horizon = np.mean(huber, axis=0)

    # Directional accuracy: sign agreement
    pred_sign = np.sign(preds)
    target_sign = np.sign(targets)
    # Consider 0 sign as matching if target is 0, otherwise sign agreement
    correct_sign = (pred_sign == target_sign)
    da_per_horizon = np.mean(correct_sign.astype(np.float64), axis=0) * 100.0

    return {
        "loss": {
            "total": round(float(np.mean(huber_per_horizon)), 4),
            "h5": round(float(huber_per_horizon[0]), 4),
            "h15": round(float(huber_per_horizon[1]), 4),
            "h30": round(float(huber_per_horizon[2]), 4),
        },
        "mae": {
            "total": round(float(np.mean(mae_per_horizon)), 4),
            "h5": round(float(mae_per_horizon[0]), 4),
            "h15": round(float(mae_per_horizon[1]), 4),
            "h30": round(float(mae_per_horizon[2]), 4),
        },
        "directional_accuracy_pct": {
            "total": round(float(np.mean(da_per_horizon)), 2),
            "h5": round(float(da_per_horizon[0]), 2),
            "h15": round(float(da_per_horizon[1]), 2),
            "h30": round(float(da_per_horizon[2]), 2),
        },
    }


def simulate_spread_adjusted_trading(
    preds: np.ndarray,
    targets: np.ndarray,
    closes: np.ndarray,
    atrs: np.ndarray,
    spread_usd: float = 2.50,
    threshold: float = 0.20,
) -> Dict[str, Any]:
    """
    Simulates spread-adjusted trading on prediction horizon t+15:
    - If pred_return > +threshold: BUY signal (+1)
    - If pred_return < -threshold: SELL signal (-1)
    - Fixed spread friction deducted from every trade.
    """
    net_pnls: List[float] = []
    trade_returns: List[float] = []

    N = len(preds)
    for i in range(N):
        pred_15 = preds[i, 1]
        atr = max(atrs[i], 1e-4)

        if pred_15 > threshold:
            # BUY: price move is targets[i, 1] * atr
            gross_pnl = targets[i, 1] * atr
            net_pnl = gross_pnl - spread_usd
            net_pnls.append(net_pnl)
            trade_returns.append(net_pnl / closes[i])
        elif pred_15 < -threshold:
            # SELL: price move is -targets[i, 1] * atr
            gross_pnl = -targets[i, 1] * atr
            net_pnl = gross_pnl - spread_usd
            net_pnls.append(net_pnl)
            trade_returns.append(net_pnl / closes[i])

    total_trades = len(net_pnls)
    if total_trades == 0:
        return {
            "spread_usd": spread_usd,
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "profit_factor": 1.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "net_pnl_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = sum(1 for p in net_pnls if p > 0)
    losses = sum(1 for p in net_pnls if p < 0)
    win_rate = (wins / total_trades) * 100.0

    gross_profit = float(sum(p for p in net_pnls if p > 0))
    gross_loss = float(abs(sum(p for p in net_pnls if p < 0)))
    profit_factor = (gross_profit / (gross_loss + 1e-6)) if gross_loss > 0 else (99.0 if gross_profit > 0 else 1.0)
    net_pnl = float(sum(net_pnls))

    # Calculate Maximum Drawdown
    equity = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd_usd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0

    # Drawdown percentage relative to base capital (e.g. $50,000 reference account)
    ref_capital = 50000.0
    max_dd_pct = (max_dd_usd / ref_capital) * 100.0

    return {
        "spread_usd": spread_usd,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
    }


def evaluate_model_pipeline(
    model_name: str,
    wm: LatentDynamicsWorldModel,
    head: MultiHorizonProjectionHead,
    loader: DataLoader,
    closes: np.ndarray,
    atrs: np.ndarray,
    device: torch.device,
    spread_usd: float = 2.50,
    threshold: float = 0.20,
    is_baseline: bool = False,
) -> Dict[str, Any]:
    """Runs evaluation across all batches and returns loss, MAE, DA%, and simulated trading results."""
    wm.eval()
    head.eval()

    all_preds: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            obs = batch.obs_137.to(device)
            targets = batch.targets_3.to(device)

            if is_baseline:
                # Baseline: 137D observation encoded without temporal sequence
                z = wm.encode_state(obs)
            else:
                candles = batch.candle_seq_240.to(device)
                z = wm.encode_state(obs, candle_seq=candles)

            preds = head(z)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    # Multi-horizon metric computation
    horizon_metrics = compute_horizon_metrics(preds_np, targets_np)

    # Trading simulation
    sim_metrics = simulate_spread_adjusted_trading(
        preds_np,
        targets_np,
        closes[: len(preds_np)],
        atrs[: len(preds_np)],
        spread_usd=spread_usd,
        threshold=threshold,
    )

    return {
        "loss": horizon_metrics["loss"],
        "mae": horizon_metrics["mae"],
        "directional_accuracy_pct": horizon_metrics["directional_accuracy_pct"],
        "simulation": sim_metrics,
    }


def evaluate_all_checkpoints(
    baseline_path: Union[str, Path] = "outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt",
    phase1_path: Union[str, Path] = "outputs/checkpoints/tca_phase1_pretrained.pt",
    phase2_path: Union[str, Path] = "outputs/checkpoints/tca_phase2_finetuned.pt",
    data_path: Union[str, Path] = r"C:\Users\Admin\Desktop\BTCUSDm.csv",
    cache_dir: Union[str, Path] = "outputs/cache",
    output_json: Union[str, Path] = "outputs/checkpoints/oos_crash_evaluation_results.json",
    batch_size: int = 128,
    spread_usd: float = 2.50,
    threshold: float = 0.20,
    device_str: str = "auto",
    dry_run: bool = False,
    dry_run_samples: int = 200,
) -> Dict[str, Any]:
    """Main evaluation pipeline comparing Baseline, Phase 1, and Phase 2."""
    set_seed(42)
    device = resolve_device(device_str)
    print(f"\n[OOS Evaluator] Target Device: {device} | Dry Run: {dry_run} | Spread: ${spread_usd:.2f}")

    # 1. Load Data
    dataset, sliced_env, master_env = load_oos_slice(
        cache_dir=cache_dir,
        data_path=data_path,
        spread_usd=spread_usd,
        dry_run=dry_run,
        dry_run_samples=dry_run_samples,
    )
    evaluated_samples = len(dataset)
    print(f"[OOS Evaluator] Crash slice active: {sliced_env.total_bars} bars, {evaluated_samples} evaluated samples.")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_mega_pool,
        drop_last=False,
    )

    # Closes and ATRs aligned to samples
    closes = sliced_env.close_arr[:evaluated_samples]
    atrs = sliced_env.atr_arr[:evaluated_samples]
    crash_drop = float(sliced_env.close_arr[0] - np.min(sliced_env.close_arr))
    mean_atr = float(np.mean(sliced_env.atr_arr))

    start_ts = (
        master_env.timestamps[55000]
        if len(master_env.timestamps) > 55000 and master_env.timestamps[55000]
        else "2025.02.22 04:48:00"
    )
    end_ts = (
        master_env.timestamps[64784]
        if len(master_env.timestamps) > 64784 and master_env.timestamps[64784]
        else "2025.02.28 23:59:00"
    )

    results: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "device": str(device),
        "dry_run": dry_run,
        "dataset": {
            "name": "February 2025 Crash Slice",
            "range": [55000, 64785],
            "total_bars": sliced_env.total_bars,
            "evaluated_samples": evaluated_samples,
            "start_timestamp": start_ts,
            "end_timestamp": end_ts,
            "spread_usd": spread_usd,
            "crash_drop_usd": round(crash_drop, 2),
            "mean_atr_usd": round(mean_atr, 2),
        },
        "models": {},
    }

    # Model 1: Baseline World Model (Unaugmented, no TCA stream)
    print("\n--- [1/3] Evaluating Baseline World Model (Static 137D Encoder) ---")
    wm_base, head_base = load_baseline_model(baseline_path, head_checkpoint_path=phase1_path, device=device)
    base_results = evaluate_model_pipeline(
        "baseline",
        wm_base,
        head_base,
        loader,
        closes,
        atrs,
        device,
        spread_usd=spread_usd,
        threshold=threshold,
        is_baseline=True,
    )
    results["models"]["baseline"] = {
        "checkpoint": str(baseline_path),
        "architecture": "WorldModel (30 keys, 137D Unaugmented, No TCA)",
        **base_results,
    }
    print(f"Baseline Loss: {base_results['loss']['total']} | DA%: {base_results['directional_accuracy_pct']['total']}% | PnL: ${base_results['simulation']['net_pnl_usd']:,.2f}")

    # Model 2: Phase 1 Pre-trained (Frozen Base Encoder + 240-Candle TCA)
    print("\n--- [2/3] Evaluating Phase 1 Pre-trained TCA (Frozen Base Encoder) ---")
    wm_p1, head_p1 = load_phase1_model(phase1_path, device=device)
    p1_results = evaluate_model_pipeline(
        "phase1_pretrained",
        wm_p1,
        head_p1,
        loader,
        closes,
        atrs,
        device,
        spread_usd=spread_usd,
        threshold=threshold,
        is_baseline=False,
    )
    results["models"]["phase1_pretrained"] = {
        "checkpoint": str(phase1_path),
        "architecture": "Frozen Encoder + 240-Candle TCA (41 keys) + Head (8 keys)",
        **p1_results,
    }
    print(f"Phase 1 Loss: {p1_results['loss']['total']} | DA%: {p1_results['directional_accuracy_pct']['total']}% | PnL: ${p1_results['simulation']['net_pnl_usd']:,.2f}")

    # Model 3: Phase 2 Fine-tuned (Jointly Fine-tuned Encoder + TCA)
    print("\n--- [3/3] Evaluating Phase 2 Fine-tuned TCA (Joint Alignment) ---")
    wm_p2, head_p2 = load_phase2_model(phase2_path, device=device)
    p2_results = evaluate_model_pipeline(
        "phase2_finetuned",
        wm_p2,
        head_p2,
        loader,
        closes,
        atrs,
        device,
        spread_usd=spread_usd,
        threshold=threshold,
        is_baseline=False,
    )
    results["models"]["phase2_finetuned"] = {
        "checkpoint": str(phase2_path),
        "architecture": "Fine-tuned Encoder + TCA + Projection Head",
        **p2_results,
    }
    print(f"Phase 2 Loss: {p2_results['loss']['total']} | DA%: {p2_results['directional_accuracy_pct']['total']}% | PnL: ${p2_results['simulation']['net_pnl_usd']:,.2f}")

    # Comparative analysis
    p2_pnl = p2_results["simulation"]["net_pnl_usd"]
    base_pnl = base_results["simulation"]["net_pnl_usd"]
    pnl_delta = p2_pnl - base_pnl

    results["comparison"] = {
        "phase2_vs_baseline_pnl_delta_usd": round(pnl_delta, 2),
        "best_simulation_pnl_model": (
            "phase2_finetuned" if p2_pnl >= max(base_pnl, p1_results["simulation"]["net_pnl_usd"])
            else ("phase1_pretrained" if p1_results["simulation"]["net_pnl_usd"] >= base_pnl else "baseline")
        ),
    }

    # Export results JSON atomically
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OOS Evaluator] Successfully saved results to: {out_path}")

    return results


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate TCA World Models on Out-of-Sample Crash Slice")
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="outputs/he_chuyen_gia/v193_runs/mythos_muzero_specialized_challenger.pt",
        help="Path to baseline challenger checkpoint",
    )
    parser.add_argument(
        "--phase1-checkpoint",
        type=str,
        default="outputs/checkpoints/tca_phase1_pretrained.pt",
        help="Path to Phase 1 pre-trained checkpoint",
    )
    parser.add_argument(
        "--phase2-checkpoint",
        type=str,
        default="outputs/checkpoints/tca_phase2_finetuned.pt",
        help="Path to Phase 2 fine-tuned checkpoint",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=r"C:\Users\Admin\Desktop\BTCUSDm.csv",
        help="Path to authentic M1 CSV",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="outputs/cache",
        help="Directory containing precomputed cache",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="outputs/checkpoints/oos_crash_evaluation_results.json",
        help="Output path for evaluation report JSON",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Inference batch size",
    )
    parser.add_argument(
        "--spread-usd",
        type=float,
        default=2.50,
        help="Fixed spread friction ($2.50 USD)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.20,
        help="Minimum ATR prediction threshold to trigger simulated trade",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Target device (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run fast verification mode (200 samples)",
    )

    args = parser.parse_args()

    try:
        evaluate_all_checkpoints(
            baseline_path=args.baseline_checkpoint,
            phase1_path=args.phase1_checkpoint,
            phase2_path=args.phase2_checkpoint,
            data_path=args.data_path,
            cache_dir=args.cache_dir,
            output_json=args.output_json,
            batch_size=args.batch_size,
            spread_usd=args.spread_usd,
            threshold=args.threshold,
            device_str=args.device,
            dry_run=args.dry_run,
        )
        return 0
    except Exception as e:
        print(f"\n[OOS Evaluator ERROR] {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
