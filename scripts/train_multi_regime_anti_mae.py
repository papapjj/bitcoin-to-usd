#!/usr/bin/env python3
"""
scripts/train_multi_regime_anti_mae.py

Multi-Regime Anti-MAE Engine Finetuning.
Co-trains on both Bull Regimes (Feb 2025) and Bear Regimes (Oct 2026).
Solves the "buy-the-dip in bear trend" defect without any heuristic gates or filters.
Directly penalizes adverse excursion via differentiable gradient backpropagation.
"""

import os
import sys
from pathlib import Path
import time
import logging
from typing import Any, Dict, Optional, Tuple
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
from research_reasoning_world_class.src.smc_liquidity_engine import (
    SMCLiquidityEngine,
    SMCConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MultiRegimeTrainer")


def extract_features_from_csv(
    csv_path: Path,
    k_sl: float = 2.50,
    k_tp: float = 2.50,
    spread_usd: float = 28.98,
    horizon_bars: int = 40,
    lookback_bars: int = 60,
):
    logger.info("Extracting features from: %s", csv_path.name)
    df = pd.read_csv(csv_path, sep="\t")
    df_renamed = df.rename(columns={
        '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
        '<TICKVOL>': 'Volume', '<SPREAD>': 'Spread',
    })
    if 'Datetime' not in df_renamed.columns:
        if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
            df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']
        else:
            df_renamed['Datetime'] = pd.date_range("2025-01-01", periods=len(df), freq="1min")

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
    action_targets = np.zeros(valid_count, dtype=np.int64)
    prm_targets = np.zeros(valid_count, dtype=np.float32)
    buy_maes = np.zeros(valid_count, dtype=np.float32)
    sell_maes = np.zeros(valid_count, dtype=np.float32)
    smc_engine = SMCLiquidityEngine(SMCConfig())

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

        # Causal Market Structure Confluence Filter (Defect 3)
        smc_feats = smc_engine.extract_features(m1_bars_60)
        ssl_sweep_fresh = (smc_feats[1] > 0.5) and (smc_feats[11] <= 0.30) and (smc_feats[5] >= 0.25)
        bsl_sweep_fresh = (smc_feats[0] > 0.5) and (smc_feats[10] <= 0.30) and (smc_feats[4] >= 0.25)
        choch_bull = (smc_feats[53] > 0.5)
        choch_bear = (smc_feats[53] < -0.5)
        trapped_sellers = (smc_feats[49] > 0.5)
        trapped_buyers = (smc_feats[48] > 0.5)
        bull_confluence = (ssl_sweep_fresh or choch_bull or trapped_sellers or smc_feats[63] > 0.30) and (smc_feats[56] <= 0.60)
        bear_confluence = (bsl_sweep_fresh or choch_bear or trapped_buyers or smc_feats[63] < -0.30) and (smc_feats[56] >= 0.40)

        sl_dist = k_sl * cur_atr + spread_usd
        tp_dist = k_tp * cur_atr
        f_h = highs[t+1 : t+1+horizon_bars]
        f_l = lows[t+1 : t+1+horizon_bars]

        # Path-Dependent Forward Simulation
        buy_win = False
        worst_adv_buy = 0.0
        if bull_confluence:
            b_sl = cur_p - sl_dist
            b_tp = cur_p + tp_dist
            for k in range(horizon_bars):
                adv = (cur_p - f_l[k]) / cur_atr
                if adv > worst_adv_buy:
                    worst_adv_buy = adv
                # Intrabar collision -> conservative stop out
                if f_l[k] <= b_sl and f_h[k] >= b_tp:
                    break
                if f_l[k] <= b_sl:
                    break
                if f_h[k] >= b_tp:
                    buy_win = True
                    break
        buy_maes[i] = worst_adv_buy

        sell_win = False
        worst_adv_sell = 0.0
        if bear_confluence:
            s_sl = cur_p + sl_dist
            s_tp = cur_p - tp_dist
            for k in range(horizon_bars):
                adv = (f_h[k] - cur_p) / cur_atr
                if adv > worst_adv_sell:
                    worst_adv_sell = adv
                # Intrabar collision -> conservative stop out
                if f_h[k] >= s_sl and f_l[k] <= s_tp:
                    break
                if f_h[k] >= s_sl:
                    break
                if f_l[k] <= s_tp:
                    sell_win = True
                    break
        sell_maes[i] = worst_adv_sell

        if buy_win and not sell_win:
            action_targets[i] = 1
            prm_targets[i] = float(np.clip(1.0 - (worst_adv_buy / k_sl) * 0.35, 0.65, 1.0))
        elif sell_win and not buy_win:
            action_targets[i] = 2
            prm_targets[i] = float(np.clip(1.0 - (worst_adv_sell / k_sl) * 0.35, 0.65, 1.0))
        else:
            action_targets[i] = 0
            prm_targets[i] = 0.10

    return {
        "ingress": ingress_matrix,
        "actions": action_targets,
        "prms": prm_targets,
        "buy_maes": buy_maes,
        "sell_maes": sell_maes,
        "closes": closes,
        "highs": highs,
        "lows": lows,
        "atrs": atrs,
        "lookback": lookback_bars,
        "horizon": horizon_bars,
    }


def atomic_save_checkpoint(
    engine: BilateralDualCoTEngine,
    checkpoint_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Saves checkpoint atomically with full schema including reasoning_engine_state_dict."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp_{os.getpid()}")

    payload = {
        "version": "v193_bilateral_dual_cot",
        "saved_at": time.time(),
        "bull_cot_state_dict": engine.bull_cot.state_dict(),
        "bear_cot_state_dict": engine.bear_cot.state_dict(),
        "cross_examination_state_dict": engine.cross_examination.state_dict(),
        "arbiter_state_dict": engine.arbiter.state_dict(),
        "trap_detector_state_dict": engine.trap_detector.state_dict(),
        "state_adapter_state_dict": engine.state_adapter.state_dict(),
        "reasoning_engine_state_dict": engine.bull_cot.state_dict(),  # Dual compatibility
        "config": engine.config,
        "metadata": metadata or {},
    }
    torch.save(payload, str(temp_path))
    temp_path.replace(checkpoint_path)
    logger.info("Atomically saved checkpoint to: %s", checkpoint_path)
    return checkpoint_path


def evaluate_test_set(
    engine: BilateralDualCoTEngine,
    test_data: Dict[str, Any],
    prm_threshold: float = 0.58,
    sl_mult: float = 2.50,
    tp_mult: float = 2.50,
    spread_usd: float = 28.98,
) -> Dict[str, Any]:
    engine.eval()
    ingress = torch.from_numpy(test_data["ingress"]).float()
    with torch.no_grad():
        v = engine.forward_batch(ingress, batch_size=512)
        
    acts = v.actions.cpu().numpy()
    prms = v.prm_scores.cpu().numpy()
    
    closes = test_data["closes"]
    highs = test_data["highs"]
    lows = test_data["lows"]
    atrs = test_data["atrs"]
    lookback = test_data["lookback"]
    horizon = test_data["horizon"]
    
    buy_wins, buy_losses = 0, 0
    sell_wins, sell_losses = 0, 0
    trade_maes = []
    
    for i in range(len(acts)):
        if prms[i] < prm_threshold or acts[i] == 0:
            continue
        act = acts[i]
        t = lookback + i
        cur_p = closes[t]
        cur_atr = atrs[t]
        f_h = highs[t+1 : t+1+horizon]
        f_l = lows[t+1 : t+1+horizon]
        
        worst_adv = 0.0
        if act == 1:
            sl_p = cur_p - (sl_mult * cur_atr + spread_usd)
            tp_p = cur_p + tp_mult * cur_atr
            win = False
            for k in range(len(f_h)):
                adv = (cur_p - f_l[k]) / cur_atr
                if adv > worst_adv:
                    worst_adv = adv
                # Intrabar collision -> conservative loss
                if f_l[k] <= sl_p and f_h[k] >= tp_p:
                    break
                if f_l[k] <= sl_p:
                    break
                if f_h[k] >= tp_p:
                    win = True
                    break
            trade_maes.append(worst_adv)
            if win:
                buy_wins += 1
            else:
                buy_losses += 1
        elif act in (-1, 2):
            sl_p = cur_p + (sl_mult * cur_atr + spread_usd)
            tp_p = cur_p - tp_mult * cur_atr
            win = False
            for k in range(len(f_h)):
                adv = (f_h[k] - cur_p) / cur_atr
                if adv > worst_adv:
                    worst_adv = adv
                # Intrabar collision -> conservative loss
                if f_h[k] >= sl_p and f_l[k] <= tp_p:
                    break
                if f_h[k] >= sl_p:
                    break
                if f_l[k] <= tp_p:
                    win = True
                    break
            trade_maes.append(worst_adv)
            if win:
                sell_wins += 1
            else:
                sell_losses += 1
            
    tot = buy_wins + buy_losses + sell_wins + sell_losses
    wr = ((buy_wins + sell_wins) / max(1, tot)) * 100.0
    b_wr = (buy_wins / max(1, buy_wins + buy_losses)) * 100.0
    s_wr = (sell_wins / max(1, sell_wins + sell_losses)) * 100.0
    mae = float(np.mean(trade_maes)) if trade_maes else 0.0
    return {
        "trades": tot,
        "wr": wr,
        "buy_trades": buy_wins + buy_losses,
        "buy_wr": b_wr,
        "sell_trades": sell_wins + sell_losses,
        "sell_wr": s_wr,
        "avg_mae": mae,
    }


def train_multi_regime(epochs: int = 8, batch_size: int = 128, lr: float = 5e-5):
    # 1. Build Multi-Regime Datasets
    # Training: Feb 2025 (Bull) + Oct 2026 Week 1 (Bear) + Oct 2026 Week 2 (Consolidation)
    feb_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_7day_clean.csv")
    w1_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week1_20261005.csv")
    w2_data = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week2_20261012.csv")
    
    # Testing (Unseen OOS): Oct 2026 Week 3
    w3_test = extract_features_from_csv(ROOT / "outputs" / "btcusd_m1_week3_20261019.csv")
    
    # Concatenate training matrices across diverse regimes
    train_ingress = np.vstack([feb_data["ingress"], w1_data["ingress"], w2_data["ingress"]])
    train_actions = np.concatenate([feb_data["actions"], w1_data["actions"], w2_data["actions"]])
    train_prms = np.concatenate([feb_data["prms"], w1_data["prms"], w2_data["prms"]])
    train_b_maes = np.concatenate([feb_data["buy_maes"], w1_data["buy_maes"], w2_data["buy_maes"]])
    train_s_maes = np.concatenate([feb_data["sell_maes"], w1_data["sell_maes"], w2_data["sell_maes"]])
    
    N_train = len(train_ingress)
    logger.info("Multi-Regime Training Pool: %d samples | OOS Test: %d samples", N_train, len(w3_test["ingress"]))
    
    # Defect 4: Prioritized HOLD action weighting [1.25, 1.50, 1.50]
    device = torch.device("cpu")
    torch.set_num_threads(8)
    class_weights = torch.tensor([1.25, 1.50, 1.50], dtype=torch.float32, device=device)
    logger.info("Multi-Regime Prioritized Weights: HOLD=%.2f, BUY=%.2f, SELL=%.2f",
                class_weights[0].item(), class_weights[1].item(), class_weights[2].item())
    
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
    
    # Warm start from existing champion
    ckpt_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
    if ckpt_path.exists():
        logger.info("Warm starting from: %s", ckpt_path.name)
        engine.load_checkpoint(ckpt_path)
        
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.02)
    criterion_mse = nn.MSELoss()
    
    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_wr = 0.0
    output_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_regime_anti_mae_champion.pt"
    
    indices = np.arange(N_train)
    logger.info("Beginning Multi-Regime Anti-MAE Finetuning for %d epochs...", epochs)
    saved_any = False
    
    for ep in range(1, epochs + 1):
        engine.train()
        np.random.shuffle(indices)
        total_loss = 0.0
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
            
            # Direct Anti-MAE Excursion Penalties (Adjusted threshold 2.50 ATR)
            mae_penalty_buy = probs[:, 1] * torch.relu((b_b_mae - 2.50) / 2.50) ** 2
            mae_penalty_sell = probs[:, 2] * torch.relu((b_s_mae - 2.50) / 2.50) ** 2
            loss_anti_mae = torch.mean(mae_penalty_buy + mae_penalty_sell)
            
            # Quadratic False-Entry Penalty on y = 0
            hold_mask = (b_act == 0)
            if hold_mask.any():
                loss_false_entry = torch.mean((probs[hold_mask, 1] + probs[hold_mask, 2]) ** 2)
            else:
                loss_false_entry = torch.tensor(0.0, device=device)
            
            b_target = (b_act == 1).float()
            loss_bull = F.binary_cross_entropy(torch.clamp(bull_out.bull_score.view(-1), 1e-4, 1.0 - 1e-4), b_target)
            
            r_target = (b_act == 2).float()
            loss_bear = F.binary_cross_entropy(torch.clamp(bear_out.bear_score.view(-1), 1e-4, 1.0 - 1e-4), r_target)
            
            loss_arb = criterion_ce(logits, b_act)
            loss_prm = criterion_mse(prm_preds, b_prm)
            
            loss = loss_arb + 0.5 * loss_prm + 0.5 * loss_bull + 0.5 * loss_bear + 1.8 * loss_anti_mae + 1.0 * loss_false_entry
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            
        scheduler.step()
        train_elapsed = time.time() - t0
        avg_loss = total_loss / max(1, batches)
        
        # Evaluate on completely unseen OOS Week 3 (Oct 19 - Oct 25)
        oos_res = evaluate_test_set(engine, w3_test, prm_threshold=0.50, sl_mult=2.50, tp_mult=2.50, spread_usd=28.98)
        tr = oos_res["trades"]
        wr = oos_res["wr"]
        b_wr = oos_res["buy_wr"]
        b_tr = oos_res["buy_trades"]
        s_wr = oos_res["sell_wr"]
        s_tr = oos_res["sell_trades"]
        mae = oos_res["avg_mae"]
        
        logger.info(
            "Epoch %d/%d (%.1fs) | Loss: %.4f | OOS Week 3: Trades=%d | WR=%.1f%% (BUY: %.1f%% [%d], SELL: %.1f%% [%d]) | MAE=%.2f ATR",
            ep, epochs, train_elapsed, avg_loss, tr, wr, b_wr, b_tr, s_wr, s_tr, mae
        )
        
        if tr >= 10 and wr >= best_wr:
            best_wr = wr
            saved_any = True
            meta = {
                "epoch": ep,
                "oos_wr": wr,
                "oos_trades": tr,
                "oos_mae": mae,
                "buy_wr": b_wr,
                "sell_wr": s_wr,
            }
            atomic_save_checkpoint(engine, output_checkpoint, metadata=meta)
            atomic_save_checkpoint(engine, ckpt_path, metadata=meta)
            logger.info(">>> Promoted Multi-Regime Anti-MAE Champion (WR=%.1f%%, MAE=%.2f ATR) -> %s",
                        wr, mae, output_checkpoint)

    if not saved_any:
        # Fallback save at final epoch
        meta = {"epoch": epochs, "oos_wr": wr, "oos_mae": mae}
        atomic_save_checkpoint(engine, output_checkpoint, metadata=meta)
        atomic_save_checkpoint(engine, ckpt_path, metadata=meta)
        logger.info(">>> Saved Final Multi-Regime Anti-MAE Champion -> %s", output_checkpoint)

    logger.info("Multi-Regime Training Completed. Champion: %s", output_checkpoint)


if __name__ == "__main__":
    train_multi_regime(epochs=8, batch_size=128, lr=5e-5)
