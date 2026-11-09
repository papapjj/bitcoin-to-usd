"""
train_high_precision_champion.py

High-Precision Bilateral Dual-CoT Fine-Tuning Engine
Eliminates False-Positive Entries (Anti-MAE / Anti-Premature SL)
Optimizes for:
- Win Rate >= 80%
- Profit Factor > 3.0
- Trades >= 700 across 7 days (> 100 trades/day)
- Max Drawdown < 10%
- Symmetrical Bull / Bear execution (~50% BUY, ~50% SELL)
"""

import sys, os
sys.path.insert(0, os.getcwd())
import time
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)
from scripts.train_7day_bilateral_dual_cot_selfplay import (
    extract_features_and_targets_from_m1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HighPrecisionTrainer")


def focal_binary_cross_entropy(preds, targets, gamma=2.0, alpha=0.5, pos_weight=2.0):
    """
    Focal Binary Cross Entropy:
    Heavily penalizes confident false positives, forcing high precision.
    """
    p = torch.clamp(preds, 1e-5, 1.0 - 1e-5)
    ce_loss = - (pos_weight * targets * torch.log(p) + (1.0 - targets) * torch.log(1.0 - p))
    p_t = p * targets + (1.0 - p) * (1.0 - targets)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * ce_loss).mean()


def focal_cross_entropy(logits, targets, gamma=2.0, weights=None):
    """
    Multi-class Focal Loss:
    Down-weights easy examples, focuses on hard transitions between HOLD and BUY/SELL.
    """
    ce_loss = F.cross_entropy(logits, targets, weight=weights, reduction='none')
    p = torch.exp(-ce_loss)
    focal_loss = ((1.0 - p) ** gamma) * ce_loss
    return focal_loss.mean()


def main():
    device = torch.device("cpu")
    torch.set_num_threads(4)
    
    csv_path = Path("outputs/btcusd_m1_7day_clean.csv")
    if not csv_path.exists():
        logger.error("Dataset not found at %s", csv_path)
        return

    logger.info("Loading 7-day clean M1 dataset...")
    df = pd.read_csv(csv_path, sep="\t")
    ingress, targets, traps, prices, atrs = extract_features_and_targets_from_m1(df)
    N = len(ingress)
    
    # Extract OHLC for simulation
    opens = df['<OPEN>'].values[-N:] if '<OPEN>' in df.columns else df['open'].values[-N:]
    highs = df['<HIGH>'].values[-N:] if '<HIGH>' in df.columns else df['high'].values[-N:]
    lows = df['<LOW>'].values[-N:] if '<LOW>' in df.columns else df['low'].values[-N:]
    closes = df['<CLOSE>'].values[-N:] if '<CLOSE>' in df.columns else df['close'].values[-N:]

    # Map targets: 0: HOLD, 1: BUY, 2: SELL
    y_cls = np.zeros(N, dtype=np.int64)
    y_cls[targets == 1] = 1
    y_cls[targets == -1] = 2

    # Split: 80% Train, 20% Out-of-Sample Validation
    split_idx = int(N * 0.80)
    train_x = ingress[:split_idx]
    train_y = y_cls[:split_idx]
    train_traps = traps[:split_idx]
    
    val_x = ingress[split_idx:]
    val_y = y_cls[split_idx:]
    val_traps = traps[split_idx:]
    val_prices = prices[split_idx:]
    val_atrs = atrs[split_idx:]
    val_opens = opens[split_idx:]
    val_highs = highs[split_idx:]
    val_lows = lows[split_idx:]
    val_closes = closes[split_idx:]

    logger.info("Train samples: %d | Val samples: %d", len(train_x), len(val_x))

    # Initialize Engine
    config = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.35,
        trap_threshold=0.60,
    )
    engine = BilateralDualCoTEngine(config).to(device)

    # Load starting checkpoint
    ckpt_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
    if ckpt_path.exists():
        logger.info("Loading baseline weights from %s...", ckpt_path.name)
        engine.load_checkpoint(str(ckpt_path))

    # Separate optimizers for Specialists and Arbiter
    lr = 8e-5
    opt_bull = torch.optim.AdamW(engine.bull_cot.parameters(), lr=lr, weight_decay=1e-4)
    opt_bear = torch.optim.AdamW(
        list(engine.bear_cot.parameters()) + list(engine.trap_detector.parameters()),
        lr=lr * 1.2, # Slightly higher lr to bring Bear specialist to parity
        weight_decay=1e-4,
    )
    opt_arbiter = torch.optim.AdamW(
        list(engine.cross_examination.parameters())
        + list(engine.arbiter.parameters())
        + list(engine.state_adapter.parameters()),
        lr=lr * 1.5,
        weight_decay=1e-4,
    )

    batch_size = 256
    epochs = 8
    indices = np.arange(len(train_x))
    
    arb_class_weights = torch.tensor([1.2, 2.0, 2.0], device=device)
    
    best_val_pf = 0.0
    best_val_wr = 0.0
    best_checkpoint = None

    logger.info("Starting High-Precision Fine-Tuning (%d Epochs)...", epochs)
    
    for epoch in range(1, epochs + 1):
        engine.train()
        np.random.shuffle(indices)
        
        t_start = time.perf_counter()
        total_bull_loss = 0.0
        total_bear_loss = 0.0
        total_arb_loss = 0.0
        batches = 0
        
        for b_start in range(0, len(indices), batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            if len(b_idx) < 8:
                continue
                
            x_b = torch.from_numpy(train_x[b_idx]).float().to(device)
            y_b = torch.from_numpy(train_y[b_idx]).long().to(device)
            trap_b = torch.from_numpy(train_traps[b_idx]).long().to(device)
            
            # Forward adapter & CoT specialists
            z_b = engine.state_adapter(x_b)
            bull_out = engine.bull_cot(z_b)
            bear_out = engine.bear_cot(z_b.detach())
            
            b_scores = bull_out.bull_score
            r_scores = bear_out.bear_score
            
            # Bull Specialist: Target=1 for BUY, 0 otherwise
            b_tgt = torch.where(y_b == 1, 1.0, 0.0).float()
            bull_loss = focal_binary_cross_entropy(b_scores, b_tgt, gamma=2.0, pos_weight=2.5)
            
            opt_bull.zero_grad()
            bull_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(engine.bull_cot.parameters(), 1.0)
            opt_bull.step()
            
            # Bear Specialist: Target=1 for SELL, 0 otherwise
            r_tgt = torch.where(y_b == 2, 1.0, 0.0).float()
            bear_loss = focal_binary_cross_entropy(r_scores, r_tgt, gamma=2.0, pos_weight=2.5)
            
            # Trap loss
            p_thought = bull_out.cot_hops[:, -1, :].detach()
            a_thought = bear_out.cot_hops[:, -1, :]
            joint_a_p = torch.cat([a_thought, p_thought], dim=-1)
            trap_logits = engine.trap_detector(joint_a_p)
            trap_weights = torch.tensor([1.0, 10.0, 10.0], device=device)
            trap_loss = F.cross_entropy(trap_logits, trap_b, weight=trap_weights)
            
            bear_total_loss = bear_loss + 1.0 * trap_loss
            opt_bear.zero_grad()
            bear_total_loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm_(list(engine.bear_cot.parameters()) + list(engine.trap_detector.parameters()), 1.0)
            opt_bear.step()
            
            # Cross-Examination & Arbiter
            opt_arbiter.zero_grad()
            final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought.detach(), a_thought.detach(), turns=3)
            synthesis = torch.cat([final_p, final_a], dim=-1)
            arb_logits = engine.arbiter.arbiter_net(synthesis)
            prm_preds = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)
            
            # Arbiter targets (with trap overrides)
            arb_targets = y_b.clone()
            arb_targets[trap_b == 1] = 2  # Bull Trap -> SELL
            arb_targets[trap_b == 2] = 1  # Bear Trap -> BUY
            
            arb_focal = focal_cross_entropy(arb_logits, arb_targets, gamma=2.0, weights=arb_class_weights)
            
            # PRM Loss
            arb_preds_idx = torch.argmax(arb_logits, dim=-1)
            prm_targets = torch.where(arb_preds_idx == arb_targets, 0.95, 0.10).float()
            prm_loss = F.mse_loss(prm_preds, prm_targets)
            
            arb_loss = arb_focal + 0.8 * prm_loss
            arb_loss.backward()
            nn.utils.clip_grad_norm_(
                list(engine.cross_examination.parameters())
                + list(engine.arbiter.parameters())
                + list(engine.state_adapter.parameters()),
                1.0,
            )
            opt_arbiter.step()
            
            total_bull_loss += bull_loss.item()
            total_bear_loss += bear_loss.item()
            total_arb_loss += arb_loss.item()
            batches += 1
            
        dt = time.perf_counter() - t_start
        logger.info(
            "Epoch %d/%d (%.1fs) | Bull Loss: %.4f | Bear Loss: %.4f | Arb Loss: %.4f",
            epoch, epochs, dt,
            total_bull_loss / batches,
            total_bear_loss / batches,
            total_arb_loss / batches,
        )
        
        # -------------------------------------------------------------
        # EVALUATION ON OUT-OF-SAMPLE VALIDATION DATA
        # -------------------------------------------------------------
        engine.eval()
        with torch.no_grad():
            val_verdict = engine.forward_batch(torch.from_numpy(val_x).float().to(device), batch_size=512)
            v_actions = val_verdict.actions.cpu().numpy()
            v_b_scores = val_verdict.bull_scores.cpu().numpy()
            v_r_scores = val_verdict.bear_scores.cpu().numpy()
            v_prms = val_verdict.prm_scores.cpu().numpy()
            
        # Simulate OOS Validation trades
        val_trades = []
        last_t = -999
        v_N = len(val_x)
        
        for t in range(50, v_N - 30):
            act = v_actions[t]
            if act == 0 or (t - last_t < 2):
                continue
                
            # Trend guard
            ma20 = np.mean(val_prices[t-20:t])
            ma50 = np.mean(val_prices[t-50:t])
            if ma20 > ma50 and val_prices[t] > ma50 and act == -1:
                continue
            if ma20 < ma50 and val_prices[t] < ma50 and act == 1:
                continue
                
            # Sniper timing confirmation
            if act == 1:
                if not (val_closes[t] >= val_opens[t] or (val_closes[t] - val_lows[t] >= 0.20 * val_atrs[t])):
                    continue
            else:
                if not (val_closes[t] <= val_opens[t] or (val_highs[t] - val_closes[t] >= 0.20 * val_atrs[t])):
                    continue
                    
            last_t = t
            px = val_prices[t]
            atr = val_atrs[t]
            fwd_h = val_highs[t+1 : t+31]
            fwd_l = val_lows[t+1 : t+31]
            fwd_c = val_closes[t+1 : t+31]
            
            sl_dist = 2.0 * atr
            be_trig = 0.8 * atr
            be_lock = 0.35 * atr
            tp_dist = 2.2 * atr
            trail = 0.8 * atr
            
            cur_sl = sl_dist
            peak_fav = 0.0
            pnl_r = 0.0
            done = False
            
            for k in range(len(fwd_c)):
                fav = (fwd_h[k] - px) if act == 1 else (px - fwd_l[k])
                adv = (px - fwd_l[k]) if act == 1 else (fwd_h[k] - px)
                c_fav = (fwd_c[k] - px) if act == 1 else (px - fwd_c[k])
                peak_fav = max(peak_fav, fav)
                if peak_fav >= be_trig:
                    cur_sl = -max(be_lock, peak_fav - trail)
                if adv >= cur_sl:
                    pnl_r = -cur_sl / sl_dist
                    done = True
                    break
                elif fav >= tp_dist:
                    pnl_r = tp_dist / sl_dist
                    done = True
                    break
            if not done:
                pnl_r = c_fav / sl_dist
            val_trades.append(pnl_r)
            
        if len(val_trades) > 0:
            vt = np.array(val_trades)
            wins = vt[vt > 0]
            losses = vt[vt <= 0]
            val_wr = len(wins) / len(vt) * 100.0
            val_pf = wins.sum() / max(0.001, abs(losses.sum()))
            cum = np.cumsum(vt)
            val_dd = (np.maximum.accumulate(cum) - cum).max() * 0.25
            buys = np.sum(v_actions == 1)
            sells = np.sum(v_actions == -1)
            logger.info(
                "  >>> VAL METRICS: Trades: %d (B:%d, S:%d) | Win Rate: %.2f%% | PF: %.2f | DD: %.2f%%",
                len(val_trades), buys, sells, val_wr, val_pf, val_dd
            )
            
            save_path = Path("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
            if val_pf > best_val_pf and len(val_trades) >= 30:
                best_val_pf = val_pf
                best_val_wr = val_wr
                engine.save_checkpoint(save_path, metadata={'best_val_pf': best_val_pf, 'best_val_wr': best_val_wr, 'epoch': epoch})
                logger.info("  *** New Best OOS Checkpoint Saved (PF: %.2f, WR: %.2f%%) ***", val_pf, val_wr)

    logger.info("Training complete. Best OOS Validation PF: %.2f, WR: %.2f%%", best_val_pf, best_val_wr)


if __name__ == "__main__":
    main()
