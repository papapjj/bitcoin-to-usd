"""
Test script for verifying Pullback Limit Order Simulation from TCA Attended Candles.
Compares Market Order vs Pullback Limit Order (50% retracement of top-1 attended candle).
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research_reasoning_world_class.src.mega_pool_dataset import collate_mega_pool
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MultiHorizonProjectionHead,
)
from scripts.evaluate_tca_oos import load_oos_slice, resolve_device

def simulate_pullback_limit_trading(
    preds: np.ndarray,
    attns: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    spread_usd: float = 2.50,
    threshold: float = 0.20,
    ttl_bars: int = 10,
    hold_bars: int = 15,
):
    """
    Simulates Pullback Limit Order execution based on 50% retracement of the top-attended candle:
    - On BUY signal (pred > +threshold):
      Limit price = midpoint of the candle with highest attention weight in past 240 bars.
      Limit price placed <= current close.
      If next `ttl_bars` reaches Low <= Limit price -> Filled.
      Exits after `hold_bars` or at end of slice.
      If not filled within ttl -> Cancelled with 0 loss and 0 spread fee.
    - On SELL signal (pred < -threshold):
      Limit price = midpoint of top-attended candle >= current close.
      If next `ttl_bars` reaches High >= Limit price -> Filled.
    """
    N = len(preds)
    net_pnls = []
    trade_returns = []
    signals_count = 0
    filled_count = 0
    canceled_count = 0

    # Mean attention over heads: [N, 240]
    if attns.ndim == 3:
        mean_attns = attns.mean(axis=1)
    else:
        mean_attns = attns

    for i in range(N - max(ttl_bars, hold_bars)):
        pred_15 = preds[i, 1]
        atr = max(atrs[i], 1e-4)

        if abs(pred_15) <= threshold:
            continue

        signals_count += 1
        curr_price = closes[i]

        # Top-1 attended candle index in 0..239
        top_offset = int(np.argmax(mean_attns[i])) # 0 = oldest candle (239 bars ago), 239 = latest candle
        lag = 239 - top_offset
        top_bar = max(0, i - lag)
        
        top_high = highs[top_bar]
        top_low = lows[top_bar]
        p_50 = (top_high + top_low) / 2.0

        if pred_15 > threshold:
            # BUY SIGNAL
            # Desired limit price: 50% retracement level
            if p_50 < curr_price and (curr_price - p_50) <= 2.0 * atr:
                limit_price = p_50
            else:
                limit_price = curr_price - 0.2 * atr # slight pullback

            # Check fill in next ttl_bars
            filled = False
            fill_bar = -1
            for k in range(1, ttl_bars + 1):
                if i + k < N and lows[i + k] <= limit_price:
                    filled = True
                    fill_bar = i + k
                    break

            if filled:
                filled_count += 1
                exit_bar = min(fill_bar + hold_bars, N - 1)
                exit_price = closes[exit_bar]
                gross_pnl = exit_price - limit_price
                net_pnl = gross_pnl - spread_usd
                net_pnls.append(net_pnl)
                trade_returns.append(net_pnl / limit_price)
            else:
                canceled_count += 1

        elif pred_15 < -threshold:
            # SELL SIGNAL
            if p_50 > curr_price and (p_50 - curr_price) <= 2.0 * atr:
                limit_price = p_50
            else:
                limit_price = curr_price + 0.2 * atr

            filled = False
            fill_bar = -1
            for k in range(1, ttl_bars + 1):
                if i + k < N and highs[i + k] >= limit_price:
                    filled = True
                    fill_bar = i + k
                    break

            if filled:
                filled_count += 1
                exit_bar = min(fill_bar + hold_bars, N - 1)
                exit_price = closes[exit_bar]
                gross_pnl = limit_price - exit_price
                net_pnl = gross_pnl - spread_usd
                net_pnls.append(net_pnl)
                trade_returns.append(net_pnl / limit_price)
            else:
                canceled_count += 1

    total_trades = len(net_pnls)
    if total_trades == 0:
        return {
            "signals_count": signals_count,
            "filled_count": 0,
            "canceled_count": canceled_count,
            "fill_rate_pct": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 1.0,
            "net_pnl_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
        }

    wins = sum(1 for p in net_pnls if p > 0)
    win_rate = (wins / total_trades) * 100.0
    gross_profit = float(sum(p for p in net_pnls if p > 0))
    gross_loss = float(abs(sum(p for p in net_pnls if p < 0)))
    profit_factor = (gross_profit / (gross_loss + 1e-6)) if gross_loss > 0 else 99.0
    net_pnl = float(sum(net_pnls))

    equity = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_dd_usd = float(np.max(drawdown)) if len(drawdown) > 0 else 0.0
    max_dd_pct = (max_dd_usd / 50000.0) * 100.0

    return {
        "signals_count": signals_count,
        "filled_count": filled_count,
        "canceled_count": canceled_count,
        "fill_rate_pct": round((filled_count / max(1, signals_count)) * 100.0, 2),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "gross_profit_usd": round(gross_profit, 2),
        "gross_loss_usd": round(gross_loss, 2),
        "net_pnl_usd": round(net_pnl, 2),
        "max_drawdown_usd": round(max_dd_usd, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
    }

def main():
    device = resolve_device("auto")
    print(f"Device: {device}")

    # Load OOS dataset
    dataset, sliced_env, master_env = load_oos_slice(
        cache_dir="outputs/cache",
        data_path=r"C:\Users\Admin\Desktop\BTCUSDm.csv",
        spread_usd=2.50,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_mega_pool)

    # Load 20 epochs checkpoint
    ckpt_path = Path("outputs/checkpoints/tca_deep_20epochs.pt")
    if not ckpt_path.exists():
        ckpt_path = Path("outputs/checkpoints/tca_phase2_finetuned.pt")
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    wm = LatentDynamicsWorldModel().to(device)
    wm.load_state_dict(ckpt["world_model_state_dict"], strict=False)
    wm.enable_temporal_stream(candle_len=240)
    wm.temporal_encoder.load_state_dict(ckpt["temporal_encoder_state_dict"], strict=True)
    wm.eval()

    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.05).to(device)
    head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)
    head.eval()

    all_preds = []
    all_attns = []

    print("Running inference and extracting attention...")
    with torch.no_grad():
        for batch in loader:
            obs = batch.obs_137.to(device)
            candles = batch.candle_seq_240.to(device)
            z, attn = wm.encode_state(obs, candle_seq=candles, return_attention=True)
            preds = head(z)
            all_preds.append(preds.cpu().numpy())
            all_attns.append(attn.cpu().numpy())

    preds_np = np.concatenate(all_preds, axis=0)
    attns_np = np.concatenate(all_attns, axis=0)

    closes = sliced_env.close_arr[: len(preds_np)]
    highs = sliced_env.high_arr[: len(preds_np)]
    lows = sliced_env.low_arr[: len(preds_np)]
    atrs = sliced_env.atr_arr[: len(preds_np)]

    print(f"Evaluated {len(preds_np)} bars.")
    
    # Run Pullback Limit simulation
    res_limit = simulate_pullback_limit_trading(
        preds_np, attns_np, closes, highs, lows, atrs, spread_usd=2.50, threshold=0.20, ttl_bars=10, hold_bars=15
    )
    print("\n--- PULLBACK LIMIT ORDER RESULTS ---")
    for k, v in res_limit.items():
        print(f"  {k}: {v}")

if __name__ == "__main__":
    main()
