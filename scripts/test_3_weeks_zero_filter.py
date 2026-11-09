#!/usr/bin/env python3
"""
scripts/test_3_weeks_zero_filter.py

Comprehensive 3-Week Multi-Regime Evaluation with ZERO manual filters / heuristic gates.
Weeks tested:
- Week 1: 2026.10.05 to 2026.10.11 (10,070 bars)
- Week 2: 2026.10.12 to 2026.10.18 (10,080 bars)
- Week 3: 2026.10.19 to 2026.10.25 (10,080 bars)

Measures:
- Total Trades
- Win Rate (%)
- Profit Factor (PF)
- Net PnL ($)
- Max Drawdown ($)
- Average MAE (ATR)
- BUY vs SELL trade distribution
"""

import sys
from pathlib import Path
import time
import logging
import torch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ThreeWeeksTester")


def process_week_csv(csv_path: Path, lookback_bars: int = 60, horizon_bars: int = 20):
    logger.info("Processing week data: %s", csv_path.name)
    df = pd.read_csv(csv_path, sep="\t")
    df_renamed = df.rename(columns={
        '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
        '<TICKVOL>': 'Volume', '<SPREAD>': 'Spread',
    })
    df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']
    
    feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
    N = len(feat_df)
    
    m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
    slots = np.zeros((N, 40), dtype=np.float32)
    account = np.zeros((N, 8), dtype=np.float32)
    m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
    execution = np.zeros((N, 11), dtype=np.float32)
    timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
    wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
    base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

    closes = feat_df['Close'].values.astype(np.float64)
    highs = feat_df['High'].values.astype(np.float64)
    lows = feat_df['Low'].values.astype(np.float64)
    opens = feat_df['Open'].values.astype(np.float64)
    atrs = feat_df['atr'].values.astype(np.float64)

    valid_count = N - horizon_bars - lookback_bars
    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)

    for i, t in enumerate(range(lookback_bars, N - horizon_bars)):
        cur_p = closes[t]
        cur_atr = max(atrs[t], 1.0)
        m1_bars_60 = np.column_stack([
            opens[t-lookback_bars:t],
            highs[t-lookback_bars:t],
            lows[t-lookback_bars:t],
            closes[t-lookback_bars:t],
        ])
        ingress_matrix[i] = extract_canonical_bilateral_ingress_233(
            base_137[t], m1_bars_60, cur_p, cur_atr
        )

    return {
        "ingress": ingress_matrix,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "atrs": atrs,
        "lookback": lookback_bars,
        "horizon": horizon_bars,
        "valid_count": valid_count,
    }


def simulate_trading(engine, week_data, prm_threshold=0.55, sl_mult=1.5, tp_mult=2.25):
    """Simulates realistic broker execution without any external filters."""
    engine.eval()
    ingress = torch.from_numpy(week_data["ingress"]).float()
    with torch.no_grad():
        v = engine.forward_batch(ingress, batch_size=512)
        
    acts = v.actions.cpu().numpy()
    prms = v.prm_scores.cpu().numpy()
    
    closes = week_data["closes"]
    highs = week_data["highs"]
    lows = week_data["lows"]
    atrs = week_data["atrs"]
    lookback = week_data["lookback"]
    N = len(acts)
    
    balance = 10000.0
    peak_equity = 10000.0
    max_dd_usd = 0.0
    
    buy_wins, buy_losses = 0, 0
    sell_wins, sell_losses = 0, 0
    gross_profit = 0.0
    gross_loss = 0.0
    trade_maes = []
    
    in_pos = False
    pos_dir = 0
    pos_entry_p = 0.0
    pos_sl_p = 0.0
    pos_tp_p = 0.0
    pos_atr = 0.0
    pos_vol = 0.01  # 0.01 BTC lot
    worst_adv = 0.0
    
    for i in range(N):
        t = lookback + i
        cur_p = closes[t]
        cur_atr = atrs[t]
        cur_h = highs[t]
        cur_l = lows[t]
        
        # Position lifecycle
        if in_pos:
            if pos_dir == 1:
                adv = (pos_entry_p - cur_l) / pos_atr
                if adv > worst_adv: worst_adv = adv
                if cur_l <= pos_sl_p:
                    pnl = (pos_sl_p - pos_entry_p) * pos_vol
                    balance += pnl
                    gross_loss += abs(pnl)
                    buy_losses += 1
                    trade_maes.append(worst_adv)
                    in_pos = False
                elif cur_h >= pos_tp_p:
                    pnl = (pos_tp_p - pos_entry_p) * pos_vol
                    balance += pnl
                    gross_profit += pnl
                    buy_wins += 1
                    trade_maes.append(worst_adv)
                    in_pos = False
            elif pos_dir == -1:
                adv = (cur_h - pos_entry_p) / pos_atr
                if adv > worst_adv: worst_adv = adv
                if cur_h >= pos_sl_p:
                    pnl = (pos_entry_p - pos_sl_p) * pos_vol
                    balance += pnl
                    gross_loss += abs(pnl)
                    sell_losses += 1
                    trade_maes.append(worst_adv)
                    in_pos = False
                elif cur_l <= pos_tp_p:
                    pnl = (pos_entry_p - pos_tp_p) * pos_vol
                    balance += pnl
                    gross_profit += pnl
                    sell_wins += 1
                    trade_maes.append(worst_adv)
                    in_pos = False
                    
            if balance > peak_equity:
                peak_equity = balance
            dd = peak_equity - balance
            if dd > max_dd_usd:
                max_dd_usd = dd
                
        # New trade trigger (Pure neural: action != 0 and prm >= threshold)
        if not in_pos:
            act = acts[i]
            prm = prms[i]
            if prm >= prm_threshold and act != 0:
                in_pos = True
                pos_dir = act
                pos_entry_p = cur_p
                pos_atr = cur_atr
                worst_adv = 0.0
                if act == 1:
                    pos_sl_p = cur_p - sl_mult * cur_atr
                    pos_tp_p = cur_p + tp_mult * cur_atr
                else:
                    pos_sl_p = cur_p + sl_mult * cur_atr
                    pos_tp_p = cur_p - tp_mult * cur_atr

    tot_trades = buy_wins + buy_losses + sell_wins + sell_losses
    tot_wins = buy_wins + sell_wins
    wr = (tot_wins / max(1, tot_trades)) * 100.0
    b_wr = (buy_wins / max(1, buy_wins + buy_losses)) * 100.0
    s_wr = (sell_wins / max(1, sell_wins + sell_losses)) * 100.0
    pf = gross_profit / max(1e-4, gross_loss)
    net_pnl = balance - 10000.0
    avg_mae = float(np.mean(trade_maes)) if trade_maes else 0.0
    
    return {
        "trades": tot_trades,
        "wins": tot_wins,
        "wr": wr,
        "buy_trades": buy_wins + buy_losses,
        "buy_wr": b_wr,
        "sell_trades": sell_wins + sell_losses,
        "sell_wr": s_wr,
        "pf": pf,
        "net_pnl": net_pnl,
        "max_dd": max_dd_usd,
        "avg_mae": avg_mae,
    }


def evaluate_all_3_weeks(checkpoint_path: Path, prm_threshold: float = 0.58):
    cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
    engine = BilateralDualCoTEngine(cfg)
    logger.info("Loading Checkpoint: %s", checkpoint_path.name)
    engine.load_checkpoint(checkpoint_path)

    weeks = [
        ("Week 1 (Oct 05 - Oct 11, 2026)", ROOT / "outputs" / "btcusd_m1_week1_20261005.csv"),
        ("Week 2 (Oct 12 - Oct 18, 2026)", ROOT / "outputs" / "btcusd_m1_week2_20261012.csv"),
        ("Week 3 (Oct 19 - Oct 25, 2026)", ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"),
        ("Week 4 (Oct 26 - Nov 01, 2026)", ROOT / "outputs" / "btcusd_m1_week4_20261026.csv"),
    ]

    all_results = []
    print("\n" + "=" * 90)
    print(f"  3-WEEK OUT-OF-SAMPLE AUDIT (Checkpoint: {checkpoint_path.name})")
    print(f"  Zero Filters / Gating Heuristics | Pure Neural Dispatches | PRM Gate: {prm_threshold:.2f}")
    print("=" * 90)

    for name, path in weeks:
        w_data = process_week_csv(path)
        res = simulate_trading(engine, w_data, prm_threshold=prm_threshold, sl_mult=1.5, tp_mult=2.25)
        res["name"] = name
        all_results.append(res)
        print(
            f"{name:32s} | Trades={res['trades']:3d} | WR={res['wr']:5.1f}% (BUY: {res['buy_wr']:5.1f}% [{res['buy_trades']:2d}], "
            f"SELL: {res['sell_wr']:5.1f}% [{res['sell_trades']:2d}]) | PF={res['pf']:4.2f} | PnL=+${res['net_pnl']:6.2f} | DD=${res['max_dd']:5.2f} | MAE={res['avg_mae']:4.2f} ATR"
        )

    # Consolidated aggregate
    tot_tr = sum(r["trades"] for r in all_results)
    tot_w = sum(r["wins"] for r in all_results)
    tot_pnl = sum(r["net_pnl"] for r in all_results)
    max_dd = max(r["max_dd"] for r in all_results)
    avg_mae = np.mean([r["avg_mae"] for r in all_results])
    agg_wr = (tot_w / max(1, tot_tr)) * 100.0

    print("-" * 90)
    print(f"CONSOLIDATED 3-WEEK TOTAL: Trades={tot_tr:4d} | Win Rate={agg_wr:5.1f}% | Total Net PnL=+${tot_pnl:7.2f} | Max DD=${max_dd:5.2f} | Avg MAE={avg_mae:4.2f} ATR")
    print("=" * 90 + "\n")
    return all_results


if __name__ == "__main__":
    ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
    evaluate_all_3_weeks(ckpt, prm_threshold=0.58)
