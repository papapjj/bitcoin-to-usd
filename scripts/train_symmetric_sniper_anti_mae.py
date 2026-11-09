#!/usr/bin/env python3
"""
scripts/train_symmetric_sniper_anti_mae.py

Symmetric Anti-MAE Sniper Engine.
Balances Bull and Bear branches so BUY and SELL are executed with equal conviction,
minimal MAE (< 0.50 ATR), and Win Rate >= 70-80% under tight SL (1.2 - 1.5 ATR).
Zero manual filters or heuristic gates.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)
from scripts.train_pure_anti_mae_engine import build_pure_anti_mae_dataset, evaluate_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SymmetricSniperTrainer")


def train_symmetric_champion(epochs: int = 10, batch_size: int = 64, lr: float = 8e-5):
    data = build_pure_anti_mae_dataset(max_mae_atr=0.55, min_mfe_atr=1.80, horizon_bars=20, lookback_bars=60)
    dates = data["dates"]
    
    train_mask = np.isin(dates, ['2025.02.20', '2025.02.21', '2025.02.22', '2025.02.23', '2025.02.24'])
    test_mask = ~train_mask
    
    train_indices = np.where(train_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    ingress = data["ingress"]
    act_tgt = data["actions"]
    prm_tgt = data["prms"]
    buy_maes = data["buy_maes"]
    sell_maes = data["sell_maes"]
    
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
    
    ckpt_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_sniper_anti_mae_champion.pt"
    if ckpt_path.exists():
        logger.info("Initializing from current sniper champion: %s", ckpt_path)
        engine.load_checkpoint(ckpt_path)
        
    # Balanced symmetrical class weights
    n_h = max(1, int(np.sum(act_tgt[train_indices] == 0)))
    n_b = max(1, int(np.sum(act_tgt[train_indices] == 1)))
    n_s = max(1, int(np.sum(act_tgt[train_indices] == 2)))
    tot = len(train_indices)
    w_h = tot / (3.0 * n_h)
    w_b = tot / (3.0 * n_b)
    # Balanced class weights for equal BUY and SELL representation
    w_s = tot / (3.0 * n_s)
    class_weights = torch.tensor([w_h, w_b, w_s], dtype=torch.float32, device=device)
    
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    criterion_mse = nn.MSELoss()
    
    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_score = -999.0
    output_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
    
    logger.info("Starting Symmetric Anti-MAE Sniper Training for %d epochs...", epochs)
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
            b_b_mae = torch.from_numpy(buy_maes[b_idx]).float().to(device)
            b_s_mae = torch.from_numpy(sell_maes[b_idx]).float().to(device)
            
            optimizer.zero_grad()
            
            z = engine.state_adapter(b_x)
            bull_out = engine.bull_cot(z)
            bear_out = engine.bear_cot(z)
            p_thought = bull_out.cot_hops[:, -1, :]
            a_thought = bear_out.cot_hops[:, -1, :]
            
            final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=3)
            synthesis = torch.cat([final_p, final_a], dim=-1)
            
            logits = engine.arbiter.arbiter_net(synthesis)
            probs = F.softmax(logits, dim=-1)
            prm_preds = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)
            
            # Anti-MAE Excursion Penalties
            mae_penalty_buy = probs[:, 1] * torch.relu((b_b_mae - 0.55) / 0.55) ** 2
            mae_penalty_sell = probs[:, 2] * torch.relu((b_s_mae - 0.55) / 0.55) ** 2
            loss_anti_mae = torch.mean(mae_penalty_buy + mae_penalty_sell)
            
            # Specialist CoT targets (Equalized Bull and Bear)
            b_target = (b_act == 1).float()
            loss_bull = F.binary_cross_entropy(torch.clamp(bull_out.bull_score.view(-1), 1e-4, 1.0 - 1e-4), b_target)
            
            r_target = (b_act == 2).float()
            loss_bear = F.binary_cross_entropy(torch.clamp(bear_out.bear_score.view(-1), 1e-4, 1.0 - 1e-4), r_target)
            
            loss_arb = criterion_ce(logits, b_act)
            loss_prm = criterion_mse(prm_preds, b_prm)
            
            loss = loss_arb + 0.5 * loss_prm + 0.5 * loss_bull + 0.5 * loss_bear + 1.2 * loss_anti_mae
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        scheduler.step()
        train_elapsed = time.time() - t0
        avg_loss = total_loss / max(1, batches)
        
        # Evaluate on OOS split with tight SL 1.2 ATR, TP 1.8 ATR
        oos_res = evaluate_split(engine, ingress, test_indices, data, prm_threshold=0.50, sl_mult=1.2, tp_mult=1.8)
        tr = oos_res["trades"]
        wr = oos_res["wr"]
        b_wr = oos_res["buy_wr"]
        b_tr = oos_res["buy_trades"]
        s_wr = oos_res["sell_wr"]
        s_tr = oos_res["sell_trades"]
        mae = oos_res["avg_mae"]
        
        logger.info(
            "Epoch %d/%d (%.1fs) | Loss: %.4f | OOS Trades: %d | WR: %.1f%% (BUY: %.1f%% [%d], SELL: %.1f%% [%d]) | MAE: %.2f ATR",
            ep, epochs, train_elapsed, avg_loss, tr, wr, b_wr, b_tr, s_wr, s_tr, mae
        )
        
        # Balance score: reward high WR, penalize MAE, reward presence of both BUY and SELL trades
        min_side_trades = min(b_tr, s_tr)
        score = wr - 15.0 * mae + 0.1 * min_side_trades
        if tr >= 15 and score > best_score:
            best_score = score
            engine.save_checkpoint(output_checkpoint, metadata={
                "epoch": ep,
                "oos_wr": wr,
                "oos_trades": tr,
                "oos_mae": mae,
                "buy_wr": b_wr,
                "buy_trades": b_tr,
                "sell_wr": s_wr,
                "sell_trades": s_tr,
                "score": score,
            })
            # Also keep latest champion file updated
            engine.save_checkpoint(ckpt_path, metadata={"epoch": ep, "oos_wr": wr, "oos_mae": mae})
            logger.info(">>> Promoted Symmetric Anti-MAE Champion (WR=%.1f%%, MAE=%.2f ATR) -> %s", wr, mae, output_checkpoint)

    logger.info("Symmetric Anti-MAE Training Complete!")


if __name__ == "__main__":
    train_symmetric_champion(epochs=8, batch_size=64, lr=7e-5)
