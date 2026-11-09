"""
Verification script for Institutional Risk-Managed Execution.
Tests our best champion (tca_optimal_frontier.pt) on the Feb 2025 Crash Slice
with institutional risk controls:
1. Strict Concurrency: Maximum 1 active position at a time (no stacking).
2. Hard Stop Loss (1.5 * ATR) and Take Profit (2.5 * ATR).
3. Risk-based position sizing (1% account equity risk per trade, starting at $50,000).
4. Pullback Limit Order (50% retracement of top-attended candle) vs Market Order.
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


def simulate_risk_managed_trading(
    preds: np.ndarray,
    attns: np.ndarray,
    closes: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    atrs: np.ndarray,
    spread_usd: float = 2.50,
    threshold: float = 0.25,
    initial_capital: float = 50000.0,
    risk_per_trade_pct: float = 1.0,  # Risk exactly 1% ($500) per trade
    sl_atr_mult: float = 1.5,
    tp_atr_mult: float = 2.5,
    use_pullback_limit: bool = True,
    ttl_bars: int = 10,
    max_hold_bars: int = 15,
):
    equity = initial_capital
    peak_equity = initial_capital
    max_drawdown_usd = 0.0
    max_drawdown_pct = 0.0

    equity_curve = [initial_capital]
    trades = []
    
    in_position = False
    pos_type = 0 # +1: BUY, -1: SELL
    entry_price = 0.0
    entry_bar = 0
    sl_price = 0.0
    tp_price = 0.0
    pos_size = 0.0 # in BTC

    pending_order = False
    pending_type = 0
    pending_price = 0.0
    pending_bar = 0
    pending_size = 0.0

    N = len(preds)
    mean_attns = attns.mean(axis=1) if attns.ndim == 3 else attns

    for i in range(N):
        curr_close = closes[i]
        curr_high = highs[i]
        curr_low = lows[i]
        curr_atr = max(atrs[i], 1e-4)

        # 1. Update active position (check SL / TP / Max Hold)
        if in_position:
            holding_bars = i - entry_bar
            closed = False
            exit_price = 0.0
            pnl_gross = 0.0

            if pos_type == 1: # BUY POSITION
                if curr_low <= sl_price:
                    # SL hit
                    exit_price = sl_price
                    closed = True
                elif curr_high >= tp_price:
                    # TP hit
                    exit_price = tp_price
                    closed = True
                elif holding_bars >= max_hold_bars:
                    # Time exit
                    exit_price = curr_close
                    closed = True

                if closed:
                    pnl_gross = (exit_price - entry_price) * pos_size
                    pnl_net = pnl_gross - (spread_usd * pos_size)
                    equity += pnl_net
                    trades.append({
                        "type": "BUY",
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl": pnl_net,
                        "bars": holding_bars,
                        "win": pnl_net > 0,
                    })
                    in_position = False

            elif pos_type == -1: # SELL POSITION
                if curr_high >= sl_price:
                    exit_price = sl_price
                    closed = True
                elif curr_low <= tp_price:
                    exit_price = tp_price
                    closed = True
                elif holding_bars >= max_hold_bars:
                    exit_price = curr_close
                    closed = True

                if closed:
                    pnl_gross = (entry_price - exit_price) * pos_size
                    pnl_net = pnl_gross - (spread_usd * pos_size)
                    equity += pnl_net
                    trades.append({
                        "type": "SELL",
                        "entry": entry_price,
                        "exit": exit_price,
                        "pnl": pnl_net,
                        "bars": holding_bars,
                        "win": pnl_net > 0,
                    })
                    in_position = False

        # Update Drawdown
        if equity > peak_equity:
            peak_equity = equity
        dd_usd = peak_equity - equity
        dd_pct = (dd_usd / peak_equity) * 100.0
        if dd_usd > max_drawdown_usd:
            max_drawdown_usd = dd_usd
        if dd_pct > max_drawdown_pct:
            max_drawdown_pct = dd_pct
        equity_curve.append(equity)

        # 2. Check pending limit order fill
        if pending_order:
            if (i - pending_bar) > ttl_bars:
                # Expired / Canceled
                pending_order = False
            else:
                if pending_type == 1 and curr_low <= pending_price:
                    # Buy limit filled
                    in_position = True
                    pos_type = 1
                    entry_price = pending_price
                    entry_bar = i
                    pos_size = pending_size
                    sl_price = entry_price - (sl_atr_mult * curr_atr)
                    tp_price = entry_price + (tp_atr_mult * curr_atr)
                    pending_order = False
                elif pending_type == -1 and curr_high >= pending_price:
                    # Sell limit filled
                    in_position = True
                    pos_type = -1
                    entry_price = pending_price
                    entry_bar = i
                    pos_size = pending_size
                    sl_price = entry_price + (sl_atr_mult * curr_atr)
                    tp_price = entry_price - (tp_atr_mult * curr_atr)
                    pending_order = False

        # 3. New signal generation (only if NOT in position and NOT pending order)
        if not in_position and not pending_order:
            pred_15 = preds[i, 1]
            if abs(pred_15) > threshold:
                # Risk budget = 1% of current equity
                risk_usd = equity * (risk_per_trade_pct / 100.0)
                sl_dist = sl_atr_mult * curr_atr
                calc_size = risk_usd / sl_dist # Size in BTC so max loss is exactly risk_usd

                # Extract top attended candle
                top_offset = int(np.argmax(mean_attns[i]))
                top_bar = max(0, i - (239 - top_offset))
                p_50 = (highs[top_bar] + lows[top_bar]) / 2.0

                if pred_15 > threshold:
                    # BUY SIGNAL
                    if use_pullback_limit:
                        target_limit = p_50 if (p_50 < curr_close and (curr_close - p_50) <= 2.0 * curr_atr) else (curr_close - 0.2 * curr_atr)
                        pending_order = True
                        pending_type = 1
                        pending_price = target_limit
                        pending_bar = i
                        pending_size = calc_size
                    else:
                        in_position = True
                        pos_type = 1
                        entry_price = curr_close
                        entry_bar = i
                        pos_size = calc_size
                        sl_price = entry_price - (sl_atr_mult * curr_atr)
                        tp_price = entry_price + (tp_atr_mult * curr_atr)

                elif pred_15 < -threshold:
                    # SELL SIGNAL
                    if use_pullback_limit:
                        target_limit = p_50 if (p_50 > curr_close and (p_50 - curr_close) <= 2.0 * curr_atr) else (curr_close + 0.2 * curr_atr)
                        pending_order = True
                        pending_type = -1
                        pending_price = target_limit
                        pending_bar = i
                        pending_size = calc_size
                    else:
                        in_position = True
                        pos_type = -1
                        entry_price = curr_close
                        entry_bar = i
                        pos_size = calc_size
                        sl_price = entry_price + (sl_atr_mult * curr_atr)
                        tp_price = entry_price - (tp_atr_mult * curr_atr)

    total_trades = len(trades)
    if total_trades == 0:
        return {"total_trades": 0, "win_rate": 0.0, "net_pnl": 0.0, "max_dd_pct": 0.0}

    wins = [t for t in trades if t["win"]]
    win_rate = (len(wins) / total_trades) * 100.0
    gross_win = sum(t["pnl"] for t in trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in trades if t["pnl"] < 0))
    profit_factor = gross_win / max(1e-6, gross_loss)
    net_profit = equity - initial_capital

    return {
        "initial_capital": initial_capital,
        "final_equity": round(equity, 2),
        "net_profit_usd": round(net_profit, 2),
        "return_pct": round((net_profit / initial_capital) * 100.0, 2),
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_usd": round(max_drawdown_usd, 2),
        "max_drawdown_pct": round(max_drawdown_pct, 2),
    }


def main():
    device = resolve_device("auto")
    print(f"Device: {device}")

    dataset, sliced_env, master_env = load_oos_slice(
        cache_dir="outputs/cache",
        data_path=r"C:\Users\Admin\Desktop\BTCUSDm.csv",
        spread_usd=2.50,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_mega_pool)

    ckpt_path = Path("outputs/checkpoints/tca_optimal_frontier.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    wm = LatentDynamicsWorldModel().to(device)
    wm.load_state_dict(ckpt["world_model_state_dict"], strict=False)
    wm.enable_temporal_stream(candle_len=240)
    wm.temporal_encoder.load_state_dict(ckpt["temporal_encoder_state_dict"], strict=True)
    wm.eval()

    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3, dropout=0.05).to(device)
    head.load_state_dict(ckpt["projection_head_state_dict"], strict=True)
    head.eval()

    all_preds, all_attns = [], []
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

    print(f"\n--- 1. INSTITUTIONAL RISK MANAGEMENT: MARKET ORDER ---")
    res_mkt = simulate_risk_managed_trading(
        preds_np, attns_np, closes, highs, lows, atrs,
        threshold=0.25, risk_per_trade_pct=1.0, sl_atr_mult=1.5, tp_atr_mult=2.5,
        use_pullback_limit=False
    )
    for k, v in res_mkt.items():
        print(f"  {k}: {v}")

    print(f"\n--- 2. INSTITUTIONAL RISK MANAGEMENT: PULLBACK LIMIT 50% ---")
    res_lim = simulate_risk_managed_trading(
        preds_np, attns_np, closes, highs, lows, atrs,
        threshold=0.25, risk_per_trade_pct=1.0, sl_atr_mult=1.5, tp_atr_mult=2.5,
        use_pullback_limit=True
    )
    for k, v in res_lim.items():
        print(f"  {k}: {v}")

    print(f"\n--- 3. HIGH SELECTIVITY (THRESHOLD=0.35, RISK=0.5%) ---")
    res_ultra = simulate_risk_managed_trading(
        preds_np, attns_np, closes, highs, lows, atrs,
        threshold=0.35, risk_per_trade_pct=0.5, sl_atr_mult=1.5, tp_atr_mult=3.0,
        use_pullback_limit=True
    )
    for k, v in res_ultra.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
