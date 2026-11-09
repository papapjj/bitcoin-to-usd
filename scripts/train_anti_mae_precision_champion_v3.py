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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)
from research_reasoning_world_class.src.bilateral_dual_cot import (
    extract_canonical_bilateral_ingress_233,
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AntiMAE_Finetune_V3")


def prepare_dataset_v3():
    clean_csv = ROOT / "outputs" / "btcusd_m1_7day_with_warmup.csv"
    logger.info("Loading canonical M1 data: %s", clean_csv)
    df = pd.read_csv(clean_csv, sep="\t")
    df_renamed = df.rename(columns={
        '<OPEN>': 'Open', 'open': 'Open',
        '<HIGH>': 'High', 'high': 'High',
        '<LOW>': 'Low', 'low': 'Low',
        '<CLOSE>': 'Close', 'close': 'Close',
        '<TICKVOL>': 'Volume', 'tick_volume': 'Volume', '<VOL>': 'RealVolume',
        '<SPREAD>': 'Spread', 'spread': 'Spread',
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

    lookback_bars = 60
    horizon_bars = 30
    valid_count = N - horizon_bars - lookback_bars

    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
    dir_targets = np.zeros(valid_count, dtype=np.int64)
    prm_targets = np.zeros(valid_count, dtype=np.float32)
    sample_prices = np.zeros(valid_count, dtype=np.float64)
    sample_atrs = np.zeros(valid_count, dtype=np.float64)

    logger.info("Extracting 233D ingress features and labeling Ultra-Precision Anti-MAE V3 targets...")
    max_mae_atr = 0.38
    min_mfe_atr = 1.60

    for i, t in enumerate(range(lookback_bars, N - horizon_bars)):
        cur_p = closes[t]
        cur_atr = max(atrs[t], 10.0)
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

        lw = (min(opens[t], closes[t]) - lows[t]) / cur_atr
        uw = (highs[t] - max(opens[t], closes[t])) / cur_atr

        f_h = highs[t+1 : t+1+horizon_bars]
        f_l = lows[t+1 : t+1+horizon_bars]

        # Anti-MAE BUY Target: Low never drops > 0.38 ATR, High reaches >= 1.60 ATR
        buy_target = False
        for k in range(len(f_h)):
            if (cur_p - f_l[k]) >= max_mae_atr * cur_atr:
                break
            if (f_h[k] - cur_p) >= min_mfe_atr * cur_atr:
                buy_target = True
                break

        # Anti-MAE SELL Target: High never rises > 0.38 ATR, Low reaches >= 1.60 ATR
        sell_target = False
        for k in range(len(f_h)):
            if (f_h[k] - cur_p) >= max_mae_atr * cur_atr:
                break
            if (cur_p - f_l[k]) >= min_mfe_atr * cur_atr:
                sell_target = True
                break

        # Precision Microstructure Validation
        if buy_target and not sell_target and uw <= 0.40 and (lw >= 0.04 or closes[t] >= opens[t]):
            dir_targets[i] = 1 # BUY
            prm_targets[i] = 0.95
        elif sell_target and not buy_target and lw <= 0.40 and (uw >= 0.04 or closes[t] <= opens[t]):
            dir_targets[i] = -1 # SELL
            prm_targets[i] = 0.95
        else:
            dir_targets[i] = 0 # HOLD
            prm_targets[i] = 0.10

    logger.info(
        "Dataset Ready: %d samples | BUY targets: %d (%.1f%%) | SELL targets: %d (%.1f%%) | HOLD: %d (%.1f%%)",
        valid_count,
        int(np.sum(dir_targets == 1)),
        float(np.mean(dir_targets == 1) * 100),
        int(np.sum(dir_targets == -1)),
        float(np.mean(dir_targets == -1) * 100),
        int(np.sum(dir_targets == 0)),
        float(np.mean(dir_targets == 0) * 100),
    )
    return ingress_matrix, dir_targets, prm_targets, sample_prices, sample_atrs


def train_champions_v3(epochs: int = 10, batch_size: int = 64, lr: float = 1.0e-4):
    ingress_matrix, dir_targets, prm_targets, sample_prices, sample_atrs = prepare_dataset_v3()
    device = torch.device("cpu")

    config = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.35,
        trap_threshold=0.90, # Strict trap threshold so false counter-punches never fire
    )
    engine = BilateralDualCoTEngine(config).to(device)

    # Initialize from existing champion weights if present
    init_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    if init_ckpt.exists():
        logger.info("Initializing from champion weights: %s", init_ckpt.name)
        engine.load_checkpoint(init_ckpt)

    engine.train()

    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    N_samples = len(dir_targets)
    # Target mapping: HOLD=0, BUY=1, SELL=2
    mapped_targets = np.where(dir_targets == 1, 1, np.where(dir_targets == -1, 2, 0))

    # Balanced class weights to eliminate HOLD dominance
    count_0 = np.sum(mapped_targets == 0)
    count_1 = np.sum(mapped_targets == 1)
    count_2 = np.sum(mapped_targets == 2)
    logger.info("Class distribution: HOLD=%d, BUY=%d, SELL=%d", count_0, count_1, count_2)

    weight_tensor = torch.tensor([
        1.0,
        float(count_0 / max(count_1, 1)),
        float(count_0 / max(count_2, 1)),
    ], dtype=torch.float32, device=device)
    logger.info("Balanced loss weights: %s", weight_tensor.cpu().numpy().round(2))

    ce_loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)
    bce_loss_fn = nn.BCEWithLogitsLoss()

    X_tensor = torch.from_numpy(ingress_matrix).float()
    y_tensor = torch.from_numpy(mapped_targets).long()
    prm_tensor = torch.from_numpy(prm_targets).float()

    logger.info("Starting Anti-MAE Dual-Champion V3 Multi-Task Training (%d epochs)...", epochs)

    for epoch in range(1, epochs + 1):
        perm = np.random.permutation(N_samples)
        total_loss = 0.0
        correct_entries = 0
        total_non_hold = 0
        correct_all = 0
        batches = 0

        for b_start in range(0, N_samples, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            bx = X_tensor[b_idx].to(device)
            by = y_tensor[b_idx].to(device)
            bprm = prm_tensor[b_idx].to(device)

            optimizer.zero_grad()

            # Adapt state to latent_dim (768D)
            z = engine.state_adapter(bx)

            # Bull Specialist forward pass
            bull_out = engine.bull_cot(z)
            b_score = bull_out.bull_score.squeeze(-1) if bull_out.bull_score.dim() > 1 else bull_out.bull_score

            # Bear Specialist forward pass
            bear_out = engine.bear_cot(z)
            r_score = bear_out.bear_score.squeeze(-1) if bear_out.bear_score.dim() > 1 else bear_out.bear_score

            # Dialectical Cross-Examination
            p_thought = bull_out.cot_hops[:, -1, :]
            a_thought = bear_out.cot_hops[:, -1, :]
            final_p, final_a, _, _, _, _ = engine.cross_examination(p_thought, a_thought, turns=3)

            # Arbiter synthesis (256 + 256 = 512)
            synthesis = torch.cat([final_p, final_a], dim=-1)
            logits = engine.arbiter.arbiter_net(synthesis)
            prm_pred = engine.arbiter.prm_evaluator(synthesis).squeeze(-1)

            # Multi-Task Losses:
            # 1. Arbiter 3-way directional routing loss
            loss_arb = ce_loss_fn(logits, by)

            # 2. Bull Specialist dedicated loss: high when BUY target (1)
            target_bull = (by == 1).float()
            b_clamped = torch.clamp(b_score, 1e-6, 1.0 - 1e-6)
            # Weighted binary cross entropy for Bull Specialist
            pos_weight_bull = (len(by) - target_bull.sum()) / max(1.0, target_bull.sum())
            loss_bull = - (pos_weight_bull * target_bull * torch.log(b_clamped) + (1.0 - target_bull) * torch.log(1.0 - b_clamped)).mean()

            # 3. Bear Specialist dedicated loss: high when SELL target (2)
            target_bear = (by == 2).float()
            r_clamped = torch.clamp(r_score, 1e-6, 1.0 - 1e-6)
            # Weighted binary cross entropy for Bear Specialist
            pos_weight_bear = (len(by) - target_bear.sum()) / max(1.0, target_bear.sum())
            loss_bear = - (pos_weight_bear * target_bear * torch.log(r_clamped) + (1.0 - target_bear) * torch.log(1.0 - r_clamped)).mean()

            # 4. PRM consistency loss (prm_pred is already sigmoid output)
            loss_prm = F.mse_loss(prm_pred, bprm)

            loss = loss_arb + 1.2 * loss_bull + 1.2 * loss_bear + 0.8 * loss_prm
            loss.backward()

            torch.nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1

            preds = torch.argmax(logits, dim=-1)
            correct_all += (preds == by).sum().item()
            non_hold_mask = (by != 0)
            total_non_hold += non_hold_mask.sum().item()
            correct_entries += (preds[non_hold_mask] == by[non_hold_mask]).sum().item()

        scheduler.step()
        acc_all = correct_all / N_samples
        acc_entries = correct_entries / max(1, total_non_hold)
        logger.info(
            "Epoch %d/%d | Loss: %.4f | Overall Acc: %.2f%% | Entry Recall: %.2f%% (%d/%d)",
            epoch, epochs, total_loss / batches, acc_all * 100, acc_entries * 100, correct_entries, total_non_hold
        )

    # Save finalized V3 Champion Checkpoint
    out_dir = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    v3_pt = out_dir / "mythos_precision_champion_v3.pt"
    engine.save_checkpoint(v3_pt)
    logger.info("Saved Anti-MAE Precision Champion V3 Checkpoint to %s", v3_pt)

    # Also update master champion symlink/copy
    master_pt = out_dir / "mythos_precision_champion.pt"
    engine.save_checkpoint(master_pt)
    logger.info("Updated master checkpoint %s", master_pt)


if __name__ == "__main__":
    train_champions_v3(epochs=8, batch_size=64, lr=1.2e-4)
