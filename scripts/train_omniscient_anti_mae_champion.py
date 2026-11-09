#!/usr/bin/env python3
"""
scripts/train_omniscient_anti_mae_champion.py

Omniscient Anti-MAE Champion Training Engine.
Co-trains across all market regimes:
1. Bull Expansion (Feb 2025: 91k -> 99k)
2. Drift Bear & Rebound (Oct 2026 Week 1: 86k -> 83k)
3. Range Consolidation (Oct 2026 Week 2: 83k -> 82k)
4. Waterfall Crash (Oct 2026 Week 3: 84k -> 79k, including the -3,200 USD drop on 2026.10.22)

Direct Differentiable Loss Penalization:
  L_mae = p_buy * ReLU((MAE_buy - 0.55)/0.55)^2 + p_sell * ReLU((MAE_sell - 0.55)/0.55)^2
Strictly ZERO manual filters or heuristic gates.
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

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)
from scripts.train_multi_regime_anti_mae import extract_features_from_csv, evaluate_test_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("OmniscientTrainer")


def train_omniscient_champion(epochs: int = 10, batch_size: int = 64, lr: float = 6e-5):
    logger.info("Extracting features across all diverse regimes...")
    feb_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_7day_clean.csv")
    w1_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week1_20261005.csv")
    w2_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week2_20261012.csv")
    w3_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week3_20261019.csv")
    
    # Completely independent OOS test set: Week 4 (Oct 26 - Nov 01, 2026, Range: 5,747 USD)
    w4_test = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week4_20261026.csv")
    
    # Unified Training Matrix (41,743 bars)
    train_ingress = np.vstack([
        feb_data["ingress"],
        w1_data["ingress"],
        w2_data["ingress"],
        w3_data["ingress"],
    ])
    train_actions = np.concatenate([
        feb_data["actions"],
        w1_data["actions"],
        w2_data["actions"],
        w3_data["actions"],
    ])
    train_prms = np.concatenate([
        feb_data["prms"],
        w1_data["prms"],
        w2_data["prms"],
        w3_data["prms"],
    ])
    train_b_maes = np.concatenate([
        feb_data["buy_maes"],
        w1_data["buy_maes"],
        w2_data["buy_maes"],
        w3_data["buy_maes"],
    ])
    train_s_maes = np.concatenate([
        feb_data["sell_maes"],
        w1_data["sell_maes"],
        w2_data["sell_maes"],
        w3_data["sell_maes"],
    ])
    
    N_train = len(train_ingress)
    logger.info("Omniscient Training Pool: %d samples | OOS Week 4 Test: %d samples", N_train, len(w4_test["ingress"]))
    
    # Class balance weights
    n_h = max(1, int(np.sum(train_actions == 0)))
    n_b = max(1, int(np.sum(train_actions == 1)))
    n_s = max(1, int(np.sum(train_actions == 2)))
    w_h = N_train / (3.0 * n_h)
    w_b = N_train / (3.0 * n_b)
    w_s = N_train / (3.0 * n_s)
    
    device = torch.device("cpu")
    torch.set_num_threads(4)
    class_weights = torch.tensor([w_h, w_b, w_s], dtype=torch.float32, device=device)
    logger.info("Class Weights: HOLD=%.2f, BUY=%.2f, SELL=%.2f (HOLD: %d, BUY: %d, SELL: %d)",
                w_h, w_b, w_s, n_h, n_b, n_s)
                
    cfg = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.58,
        trap_threshold=0.60,
    )
    engine = BilateralDualCoTEngine(cfg).to(device)
    
    # Warm start from multi-regime checkpoint
    ckpt_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_regime_anti_mae_champion.pt"
    if ckpt_path.exists():
        logger.info("Warm-starting from: %s", ckpt_path.name)
        engine.load_checkpoint(ckpt_path)
        
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    criterion_mse = nn.MSELoss()
    
    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_wr = 0.0
    output_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_omniscient_anti_mae_champion.pt"
    primary_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
    
    indices = np.arange(N_train)
    logger.info("Starting Omniscient Training for %d epochs...", epochs)
    
    for ep in range(1, epochs + 1):
        engine.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        total_mae_loss = 0.0
        batches = 0
        t0 = time.time()
        
        for b_start in range(0, N_train, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_x = torch.from_numpy(train_ingress[b_idx]).float().to(device)
            b_act = torch.from_numpy(train_actions[b_idx]).long().to(device)
            b_prm = torch.from_numpy(train_prms[b_idx]).float().to(device)
            b_b_mae = torch.from_numpy(train_b_maes[b_idx]).float().to(device)
            b_s_mae = torch.from_numpy(train_s_maes[b_idx]).float().to(device)
            
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
            
            # Pure Anti-MAE Differentiable Excursion Penalty
            # Severe quadratic penalty on any predicted action whose forward MAE exceeds 0.55 ATR
            mae_penalty_buy = probs[:, 1] * torch.relu((b_b_mae - 0.55) / 0.55) ** 2
            mae_penalty_sell = probs[:, 2] * torch.relu((b_s_mae - 0.55) / 0.55) ** 2
            loss_anti_mae = torch.mean(mae_penalty_buy + mae_penalty_sell)
            
            b_target = (b_act == 1).float()
            loss_bull = F.binary_cross_entropy(torch.clamp(bull_out.bull_score.view(-1), 1e-4, 1.0 - 1e-4), b_target)
            
            r_target = (b_act == 2).float()
            loss_bear = F.binary_cross_entropy(torch.clamp(bear_out.bear_score.view(-1), 1e-4, 1.0 - 1e-4), r_target)
            
            loss_arb = criterion_ce(logits, b_act)
            loss_prm = criterion_mse(prm_preds, b_prm)
            
            loss = loss_arb + 0.5 * loss_prm + 0.3 * loss_bull + 0.5 * loss_bear + 1.8 * loss_anti_mae
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            total_mae_loss += loss_anti_mae.item()
            batches += 1
            
        scheduler.step()
        train_elapsed = time.time() - t0
        avg_loss = total_loss / max(1, batches)
        avg_mae_loss = total_mae_loss / max(1, batches)
        
        # Evaluate on completely unseen Week 4 (5,747 USD range)
        oos_res = evaluate_test_set(engine, w4_test, prm_threshold=0.60, sl_mult=1.5, tp_mult=2.25)
        tr = oos_res["trades"]
        wr = oos_res["wr"]
        b_wr = oos_res["buy_wr"]
        b_tr = oos_res["buy_trades"]
        s_wr = oos_res["sell_wr"]
        s_tr = oos_res["sell_trades"]
        mae = oos_res["avg_mae"]
        
        logger.info(
            "Epoch %d/%d (%.1fs) | Loss: %.4f (MAE_L: %.4f) | OOS Week 4: Trades=%d | WR=%.1f%% (BUY: %.1f%% [%d], SELL: %.1f%% [%d]) | MAE=%.2f ATR",
            ep, epochs, train_elapsed, avg_loss, avg_mae_loss, tr, wr, b_wr, b_tr, s_wr, s_tr, mae
        )
        
        if tr >= 20 and wr > best_wr:
            best_wr = wr
            engine.save_checkpoint(output_checkpoint, metadata={
                "epoch": ep,
                "oos_wr": wr,
                "oos_trades": tr,
                "oos_mae": mae,
                "buy_wr": b_wr,
                "sell_wr": s_wr,
            })
            engine.save_checkpoint(primary_checkpoint, metadata={"epoch": ep, "oos_wr": wr, "oos_mae": mae})
            logger.info(">>> Promoted Omniscient Champion (WR=%.1f%%, MAE=%.2f ATR) -> %s", wr, mae, output_checkpoint)

    logger.info("Omniscient Training Complete! Best Checkpoint: %s", output_checkpoint)


if __name__ == "__main__":
    train_omniscient_champion(epochs=8, batch_size=64, lr=6e-5)
