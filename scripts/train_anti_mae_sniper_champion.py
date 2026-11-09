#!/usr/bin/env python3
"""
scripts/train_anti_mae_sniper_champion.py

Trains the Bilateral Dual-CoT Engine with Pure Anti-MAE Sniper Targets.
Goal: Eliminate entry inaccuracy (High MAE / impulse bar chasing)
so the model achieves high Win Rate (65-75%+) with tight SL (1.0 - 1.5 ATR)
WITHOUT any external filters or hand-crafted gating.
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
logger = logging.getLogger("SniperAntiMAETrainer")


def prepare_sniper_dataset(
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

    valid_count = N - horizon_bars - lookback_bars
    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
    action_targets = np.zeros(valid_count, dtype=np.int64)  # 0: HOLD, 1: BUY, 2: SELL
    prm_targets = np.zeros(valid_count, dtype=np.float32)
    sample_prices = np.zeros(valid_count, dtype=np.float64)
    sample_atrs = np.zeros(valid_count, dtype=np.float64)

    logger.info("Extracting 233D ingress features and labeling Anti-MAE targets...")
    for i, t in enumerate(range(lookback_bars, N - horizon_bars)):
        cur_p = closes[t]
        cur_atr = max(atrs[t], 1.0)
        sample_prices[i] = cur_p
        sample_atrs[i] = cur_atr

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
        for k in range(len(f_h)):
            if (cur_p - f_l[k]) >= max_mae_atr * cur_atr:
                break
            if (f_h[k] - cur_p) >= min_mfe_atr * cur_atr:
                buy_ok = True
                break

        # SELL: High never rises > max_mae_atr, Low expands >= min_mfe_atr
        sell_ok = False
        for k in range(len(f_h)):
            if (f_h[k] - cur_p) >= max_mae_atr * cur_atr:
                break
            if (cur_p - f_l[k]) >= min_mfe_atr * cur_atr:
                sell_ok = True
                break

        if buy_ok and not sell_ok:
            action_targets[i] = 1
            prm_targets[i] = 1.0
        elif sell_ok and not buy_ok:
            action_targets[i] = 2
            prm_targets[i] = 1.0
        else:
            action_targets[i] = 0
            prm_targets[i] = 0.0

    b_cnt = int(np.sum(action_targets == 1))
    s_cnt = int(np.sum(action_targets == 2))
    h_cnt = int(np.sum(action_targets == 0))
    logger.info(
        "Sniper Dataset Ready: %d samples | BUY: %d (%.1f%%) | SELL: %d (%.1f%%) | HOLD: %d (%.1f%%)",
        valid_count, b_cnt, (b_cnt/valid_count)*100, s_cnt, (s_cnt/valid_count)*100, h_cnt, (h_cnt/valid_count)*100
    )
    return ingress_matrix, action_targets, prm_targets, sample_prices, sample_atrs, closes, highs, lows, atrs


def train_sniper_champion(
    epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1.2e-4,
    output_checkpoint: str = "outputs/he_chuyen_gia/v193_runs/mythos_sniper_anti_mae_champion.pt",
):
    ingress_matrix, action_targets, prm_targets, sample_prices, sample_atrs, closes, highs, lows, atrs = prepare_sniper_dataset()
    device = torch.device("cpu")
    torch.set_num_threads(4)

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

    # Load base champion weights to retain general market representation
    base_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
    if base_ckpt.exists():
        logger.info("Initializing from base champion weights: %s", base_ckpt)
        engine.load_checkpoint(base_ckpt)

    # Balanced class weights: HOLD is ~54%, BUY is ~22%, SELL is ~24%
    class_weights = torch.tensor([1.0, 1.3, 1.2], dtype=torch.float32, device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)
    criterion_mse = nn.MSELoss()

    # Unified optimizer for end-to-end co-evolution
    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    N = len(ingress_matrix)
    indices = np.arange(N)

    logger.info("Starting Anti-MAE Sniper Training for %d epochs (batch_size=%d, lr=%.2e)...", epochs, batch_size, lr)
    for ep in range(1, epochs + 1):
        engine.train()
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        correct_actions = 0
        total_evaluated = 0

        t0 = time.time()
        for b_start in range(0, N, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_x = torch.from_numpy(ingress_matrix[b_idx]).float().to(device)
            b_act = torch.from_numpy(action_targets[b_idx]).long().to(device)
            b_prm = torch.from_numpy(prm_targets[b_idx]).float().to(device)

            optimizer.zero_grad()

            # End-to-end forward pass
            z = engine.state_adapter(b_x)
            bull_out = engine.bull_cot(z)
            bear_out = engine.bear_cot(z)
            p_thought = bull_out.cot_hops[:, -1, :]
            a_thought = bear_out.cot_hops[:, -1, :]

            # Adversarial Dialectical Cross-Examination
            final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=3)
            synthesis = torch.cat([final_p, final_a], dim=-1)

            # Arbiter logits & PRM evaluator
            logits = engine.arbiter.arbiter_net(synthesis) # (B, 3) [HOLD, BUY, SELL]
            prm_preds = engine.arbiter.prm_evaluator(synthesis).squeeze(-1) # (B,)

            # Specialist sub-losses
            # Bull specialist: predict 1.0 when BUY, 0.0 otherwise
            b_target = (b_act == 1).float()
            loss_bull = F.binary_cross_entropy(torch.clamp(bull_out.bull_score, 1e-4, 1.0 - 1e-4), b_target)

            # Bear specialist: predict 1.0 when SELL, 0.0 otherwise
            r_target = (b_act == 2).float()
            loss_bear = F.binary_cross_entropy(torch.clamp(bear_out.bear_score, 1e-4, 1.0 - 1e-4), r_target)

            # Arbiter loss
            loss_arb = criterion_ce(logits, b_act)
            loss_prm = criterion_mse(prm_preds, b_prm)

            loss = loss_arb + 0.4 * loss_prm + 0.3 * loss_bull + 0.3 * loss_bear
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            preds = logits.argmax(dim=-1)
            correct_actions += (preds == b_act).sum().item()
            total_evaluated += len(b_act)

        scheduler.step()
        acc = (correct_actions / max(1, total_evaluated)) * 100.0
        avg_loss = total_loss / max(1, batches)
        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d (%.1fs) - Loss: %.4f | Action Accuracy: %.2f%% | LR: %.6f",
            ep, epochs, elapsed, avg_loss, acc, scheduler.get_last_lr()[0]
        )

    # Save model
    out_file = ROOT / output_checkpoint
    out_file.parent.mkdir(parents=True, exist_ok=True)
    engine.eval()
    engine.save_checkpoint(out_file, metadata={
        "trained_target": "pure_anti_mae_sniper",
        "max_mae_atr": 0.50,
        "min_mfe_atr": 1.80,
        "accuracy": acc,
        "epochs": epochs,
    })
    logger.info("Successfully saved Pure Anti-MAE Sniper Champion to: %s", out_file)

    # Run quick empirical verification
    logger.info("Running zero-filter validation on full 7-day canonical series...")
    with torch.no_grad():
        v = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
        acts = v.actions.numpy()

    b_signals = np.sum(acts == 1)
    s_signals = np.sum(acts == -1)
    h_signals = np.sum(acts == 0)
    logger.info("Validation Model Outputs: BUY=%d, SELL=%d, HOLD=%d", b_signals, s_signals, h_signals)


if __name__ == "__main__":
    train_sniper_champion(epochs=8, batch_size=64, lr=1.2e-4)
