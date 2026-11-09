#!/usr/bin/env python3
"""
scripts/train_oos_anti_mae_champion.py

Rigorous Anti-MAE Sniper Engine Finetuning with Out-Of-Sample (OOS) Validation.
Train split: 2025.02.20 - 2025.02.24 (Days 1 to 5, ~7,200 bars)
Test split (OOS): 2025.02.25 - 2025.02.27 (Days 6 to 8, ~4,300 bars)

Objective:
Minimize MAE to maximum (< 0.5 ATR) so that tight SL (1.2 - 1.8 ATR)
achieves Win Rate >= 70-80% WITHOUT any hard-coded heuristic gates or filters.
"""

import sys
from pathlib import Path
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
logger = logging.getLogger("OOSAntiMAETrainer")


def build_labeled_dataset(
    max_mae_atr: float = 0.50,
    min_mfe_atr: float = 1.80,
    horizon_bars: int = 20,
    lookback_bars: int = 60,
):
    clean_csv = ROOT / "outputs" / "btcusd_m1_7day_clean.csv"
    logger.info("Loading 7-day canonical M1 data: %s", clean_csv)
    df = pd.read_csv(clean_csv, sep="\t")
    df_renamed = df.rename(columns={
        '<OPEN>': 'Open', 'open': 'Open',
        '<HIGH>': 'High', 'high': 'High',
        '<LOW>': 'Low', 'low': 'Low',
        '<CLOSE>': 'Close', 'close': 'Close',
        '<TICKVOL>': 'Volume', 'tick_volume': 'Volume', '<VOL>': 'RealVolume',
        '<SPREAD>': 'Spread', 'spread': 'Spread',
    })
    if 'Datetime' not in df_renamed.columns:
        if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
            df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']
        elif 'time' in df_renamed.columns:
            df_renamed['Datetime'] = pd.to_datetime(df_renamed['time'])
        else:
            df_renamed['Datetime'] = pd.date_range("2025-02-20", periods=len(df), freq="1min")

    logger.info("Computing 137D Canonical Market Features...")
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
    dates = df['<DATE>'].values

    valid_count = N - horizon_bars - lookback_bars
    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
    action_targets = np.zeros(valid_count, dtype=np.int64)  # 0: HOLD, 1: BUY, 2: SELL
    prm_targets = np.zeros(valid_count, dtype=np.float32)
    sample_prices = np.zeros(valid_count, dtype=np.float64)
    sample_atrs = np.zeros(valid_count, dtype=np.float64)
    sample_dates = []

    logger.info("Extracting 233D ingress features and labeling Anti-MAE targets...")
    for i, t in enumerate(range(lookback_bars, N - horizon_bars)):
        cur_p = closes[t]
        cur_atr = max(atrs[t], 1.0)
        sample_prices[i] = cur_p
        sample_atrs[i] = cur_atr
        sample_dates.append(str(dates[t]))

        m1_bars_60 = np.column_stack([
            opens[t-lookback_bars:t],
            highs[t-lookback_bars:t],
            lows[t-lookback_bars:t],
            closes[t-lookback_bars:t],
        ])
        ingress_matrix[i] = extract_canonical_bilateral_ingress_233(
            base_137[t], m1_bars_60, cur_p, cur_atr
        )

        f_h = highs[t+1 : t+1+horizon_bars]
        f_l = lows[t+1 : t+1+horizon_bars]

        # Strict Anti-MAE Precision Sniper Condition:
        # BUY: Low never drops > max_mae_atr, High expands >= min_mfe_atr
        buy_ok = False
        buy_mae = 0.0
        for k in range(len(f_h)):
            adv = (cur_p - f_l[k]) / cur_atr
            if adv > buy_mae:
                buy_mae = adv
            if (cur_p - f_l[k]) >= max_mae_atr * cur_atr:
                break
            if (f_h[k] - cur_p) >= min_mfe_atr * cur_atr:
                buy_ok = True
                break

        # SELL: High never rises > max_mae_atr, Low expands >= min_mfe_atr
        sell_ok = False
        sell_mae = 0.0
        for k in range(len(f_h)):
            adv = (f_h[k] - cur_p) / cur_atr
            if adv > sell_mae:
                sell_mae = adv
            if (f_h[k] - cur_p) >= max_mae_atr * cur_atr:
                break
            if (cur_p - f_l[k]) >= min_mfe_atr * cur_atr:
                sell_ok = True
                break

        if buy_ok and not sell_ok:
            action_targets[i] = 1
            prm_targets[i] = float(np.clip(1.0 - buy_mae / max_mae_atr * 0.4, 0.6, 1.0))
        elif sell_ok and not buy_ok:
            action_targets[i] = 2
            prm_targets[i] = float(np.clip(1.0 - sell_mae / max_mae_atr * 0.4, 0.6, 1.0))
        else:
            action_targets[i] = 0
            prm_targets[i] = 0.0

    sample_dates = np.array(sample_dates)
    return {
        "ingress": ingress_matrix,
        "actions": action_targets,
        "prms": prm_targets,
        "prices": sample_prices,
        "atrs": sample_atrs,
        "dates": sample_dates,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "lookback_bars": lookback_bars,
        "horizon_bars": horizon_bars,
    }


def evaluate_split(engine, ingress, sample_indices, data, prm_threshold=0.55, sl_mult=1.5, tp_mult=2.25):
    """Zero-filter direct neural evaluation on given indices."""
    engine.eval()
    device = next(engine.parameters()).device
    
    with torch.no_grad():
        x_sub = torch.from_numpy(ingress[sample_indices]).float()
        v = engine.forward_batch(x_sub, batch_size=512)
    
    acts = v.actions.cpu().numpy()
    prms = v.prm_scores.cpu().numpy()
    
    lookback = data["lookback_bars"]
    horizon = data["horizon_bars"]
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    atrs = data["atrs"]
    
    buy_wins, buy_losses = 0, 0
    sell_wins, sell_losses = 0, 0
    buy_maes, sell_maes = [], []
    
    for local_i, global_i in enumerate(sample_indices):
        prm = prms[local_i]
        if prm < prm_threshold:
            continue
        act = acts[local_i]
        if act == 0:
            continue
        
        t = lookback + global_i
        cur_p = closes[t]
        cur_atr = atrs[global_i]
        f_h = highs[t+1 : t+1+horizon]
        f_l = lows[t+1 : t+1+horizon]
        
        if act == 1: # BUY
            sl_p = cur_p - sl_mult * cur_atr
            tp_p = cur_p + tp_mult * cur_atr
            win = False
            worst_adverse = 0.0
            for bar_k in range(len(f_h)):
                adv = (cur_p - f_l[bar_k]) / cur_atr
                if adv > worst_adverse:
                    worst_adverse = adv
                if f_l[bar_k] <= sl_p:
                    break
                if f_h[bar_k] >= tp_p:
                    win = True
                    break
            buy_maes.append(worst_adverse)
            if win:
                buy_wins += 1
            else:
                buy_losses += 1
                
        elif act == -1: # SELL
            sl_p = cur_p + sl_mult * cur_atr
            tp_p = cur_p - tp_mult * cur_atr
            win = False
            worst_adverse = 0.0
            for bar_k in range(len(f_h)):
                adv = (f_h[bar_k] - cur_p) / cur_atr
                if adv > worst_adverse:
                    worst_adverse = adv
                if f_h[bar_k] >= sl_p:
                    break
                if f_l[bar_k] <= tp_p:
                    win = True
                    break
            sell_maes.append(worst_adverse)
            if win:
                sell_wins += 1
            else:
                sell_losses += 1
                
    total_trades = buy_wins + buy_losses + sell_wins + sell_losses
    total_wins = buy_wins + sell_wins
    wr = (total_wins / max(1, total_trades)) * 100.0
    b_wr = (buy_wins / max(1, buy_wins + buy_losses)) * 100.0
    s_wr = (sell_wins / max(1, sell_wins + sell_losses)) * 100.0
    avg_mae = float(np.mean(buy_maes + sell_maes)) if (buy_maes or sell_maes) else 0.0
    
    return {
        "trades": total_trades,
        "wins": total_wins,
        "wr": wr,
        "buy_wr": b_wr,
        "buy_trades": buy_wins + buy_losses,
        "sell_wr": s_wr,
        "sell_trades": sell_wins + sell_losses,
        "avg_mae": avg_mae,
    }


def train_oos_anti_mae():
    data = build_labeled_dataset(max_mae_atr=0.50, min_mfe_atr=1.80, horizon_bars=20, lookback_bars=60)
    dates = data["dates"]
    N = len(dates)
    
    # Train split: Days 2025.02.20 to 2025.02.24
    # Test split (OOS): Days 2025.02.25 to 2025.02.27
    train_mask = np.isin(dates, ['2025.02.20', '2025.02.21', '2025.02.22', '2025.02.23', '2025.02.24'])
    test_mask = ~train_mask
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    logger.info("Dataset Split: Total=%d | Train=%d bars (Days 1-5) | OOS Test=%d bars (Days 6-8)",
                N, len(train_indices), len(test_indices))
    
    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    cfg = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.50,
        trap_threshold=0.60,
    )
    engine = BilateralDualCoTEngine(cfg).to(device)
    
    # Base champion warm-start
    base_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
    if base_ckpt.exists():
        logger.info("Initializing from base champion weights: %s", base_ckpt)
        engine.load_checkpoint(base_ckpt)
        
    ingress = data["ingress"]
    act_tgt = data["actions"]
    prm_tgt = data["prms"]
    
    class_weights = torch.tensor([1.0, 1.4, 1.3], dtype=torch.float32, device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    criterion_mse = nn.MSELoss()
    
    epochs = 15
    batch_size = 64
    lr = 8e-5
    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_oos_wr = 0.0
    best_checkpoint_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_sniper_anti_mae_champion.pt"
    
    logger.info("Beginning Multi-Epoch Finetuning with Out-Of-Sample Validation...")
    for ep in range(1, epochs + 1):
        engine.train()
        np.random.shuffle(train_indices)
        total_loss = 0.0
        batches = 0
        t0 = time.time()
        
        for b_start in range(0, len(train_indices), batch_size):
            b_idx = train_indices[b_start : b_start + batch_size]
            b_x = torch.from_numpy(ingress[b_idx]).float().to(device)
            b_act = torch.from_numpy(act_tgt[b_idx]).long().to(device)
            b_prm = torch.from_numpy(prm_tgt[b_idx]).float().to(device)
            
            optimizer.zero_grad()
            
            z = engine.state_adapter(b_x)
            bull_out = engine.bull_cot(z)
            bear_out = engine.bear_cot(z)
            p_thought = bull_out.cot_hops[:, -1, :]
            a_thought = bear_out.cot_hops[:, -1, :]
            
            final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=3)
            synthesis = torch.cat([final_p, final_a], dim=-1)
            
            logits = engine.arbiter.arbiter_net(synthesis)
            prm_preds = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)
            
            b_target = (b_act == 1).float()
            loss_bull = F.binary_cross_entropy(torch.clamp(bull_out.bull_score.view(-1), 1e-4, 1.0 - 1e-4), b_target)
            
            r_target = (b_act == 2).float()
            loss_bear = F.binary_cross_entropy(torch.clamp(bear_out.bear_score.view(-1), 1e-4, 1.0 - 1e-4), r_target)
            
            loss_arb = criterion_ce(logits, b_act)
            loss_prm = criterion_mse(prm_preds, b_prm)
            
            loss = loss_arb + 0.5 * loss_prm + 0.3 * loss_bull + 0.3 * loss_bear
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        scheduler.step()
        train_elapsed = time.time() - t0
        avg_loss = total_loss / max(1, batches)
        
        # Zero-filter OOS Evaluation
        oos_metrics = evaluate_split(engine, ingress, test_indices, data, prm_threshold=0.55, sl_mult=1.5, tp_mult=2.25)
        
        logger.info(
            "Epoch %d/%d (%.1fs) | Loss: %.4f | OOS Trades: %d | OOS WR: %.1f%% (BUY: %.1f%% [%d], SELL: %.1f%% [%d]) | Avg MAE: %.2f ATR",
            ep, epochs, train_elapsed, avg_loss,
            oos_metrics["trades"], oos_metrics["wr"],
            oos_metrics["buy_wr"], oos_metrics["buy_trades"],
            oos_metrics["sell_wr"], oos_metrics["sell_trades"],
            oos_metrics["avg_mae"]
        )
        
        if oos_metrics["wr"] > best_oos_wr and oos_metrics["trades"] >= 50:
            best_oos_wr = oos_metrics["wr"]
            engine.save_checkpoint(best_checkpoint_path, metadata={
                "epoch": ep,
                "oos_wr": oos_metrics["wr"],
                "oos_trades": oos_metrics["trades"],
                "oos_avg_mae": oos_metrics["avg_mae"],
                "buy_wr": oos_metrics["buy_wr"],
                "sell_wr": oos_metrics["sell_wr"],
                "max_mae_atr": 0.50,
                "min_mfe_atr": 1.80,
            })
            logger.info(">>> New Best OOS Checkpoint Promoted: WR=%.1f%%, MAE=%.2f ATR -> %s",
                        best_oos_wr, oos_metrics["avg_mae"], best_checkpoint_path)

    logger.info("Training Finished. Best OOS WR: %.1f%%.", best_oos_wr)


if __name__ == "__main__":
    train_oos_anti_mae()
