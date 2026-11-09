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
logger = logging.getLogger("AntiMAE_Finetune_V2")


def prepare_dataset():
    clean_csv = ROOT / "outputs" / "btcusd_m1_7day_with_warmup.csv"
    if not clean_csv.exists():
        clean_csv = ROOT / "outputs" / "btcusd_m1_7day_clean.csv"
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
    if 'Datetime' not in df_renamed.columns:
        if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
            df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']
        elif 'time' in df_renamed.columns:
            df_renamed['Datetime'] = pd.to_datetime(df_renamed['time'])
        else:
            df_renamed['Datetime'] = pd.date_range("2025-02-20", periods=len(df), freq="1min")

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
    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values

    lookback_bars = 60
    horizon_bars = 30
    valid_count = N - horizon_bars - lookback_bars

    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
    dir_targets = np.zeros(valid_count, dtype=np.int64)
    trap_targets = np.zeros(valid_count, dtype=np.int64)
    sample_prices = np.zeros(valid_count, dtype=np.float64)
    sample_atrs = np.zeros(valid_count, dtype=np.float64)

    logger.info("Extracting 233D ingress features and labeling Ultra-Precision Anti-MAE targets...")
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

        rec_l_20 = np.min(lows[t-20:t])
        rec_h_20 = np.max(highs[t-20:t])
        rng_20 = max(1.0, rec_h_20 - rec_l_20)
        rel_pos_20 = (cur_p - rec_l_20) / rng_20

        lw = (min(opens[t], closes[t]) - lows[t]) / cur_atr
        uw = (highs[t] - max(opens[t], closes[t])) / cur_atr

        f_h = highs[t+1 : t+1+horizon_bars]
        f_l = lows[t+1 : t+1+horizon_bars]

        # Ultra-Strict Anti-MAE BUY Target: Low never drops > 0.45 ATR, High reaches >= 1.80 ATR
        buy_target = False
        for k in range(len(f_h)):
            if (cur_p - f_l[k]) >= 0.45 * cur_atr:
                break
            if (f_h[k] - cur_p) >= 1.80 * cur_atr:
                buy_target = True
                break

        # Ultra-Strict Anti-MAE SELL Target: High never rises > 0.45 ATR, Low reaches >= 1.80 ATR
        sell_target = False
        for k in range(len(f_h)):
            if (f_h[k] - cur_p) >= 0.45 * cur_atr:
                break
            if (cur_p - f_l[k]) >= 1.80 * cur_atr:
                sell_target = True
                break

        # Value zone and candle structure filter
        if buy_target and not sell_target and rel_pos_20 <= 0.70 and (lw >= 0.05 or closes[t] >= opens[t]) and cur_p >= ema50[t] - 0.8 * cur_atr:
            dir_targets[i] = 1
        elif sell_target and not buy_target and rel_pos_20 >= 0.30 and (uw >= 0.05 or closes[t] <= opens[t]) and cur_p <= ema50[t] + 0.8 * cur_atr:
            dir_targets[i] = -1
        else:
            dir_targets[i] = 0

        # Wyckoff sweep trap ground truth
        if highs[t] >= rec_h_20 and closes[t] <= rec_h_20:
            if (cur_p - np.min(lows[t+1:t+16])) >= 0.80 * cur_atr:
                trap_targets[i] = 1  # Bull trap
        elif lows[t] <= rec_l_20 and closes[t] >= rec_l_20:
            if (np.max(highs[t+1:t+16]) - cur_p) >= 0.80 * cur_atr:
                trap_targets[i] = 2  # Bear trap

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
    return ingress_matrix, dir_targets, trap_targets, sample_prices, sample_atrs


def train_champions_v2(epochs: int = 8, batch_size: int = 64, lr: float = 1.2e-4):
    ingress_matrix, dir_targets, trap_targets, sample_prices, sample_atrs = prepare_dataset()
    device = torch.device("cpu")

    config = BilateralDualCoTConfig(
        input_dim=233,
        latent_dim=768,
        hidden_dim=256,
        num_hops=4,
        num_debate_turns=3,
        prm_threshold=0.38,
        trap_threshold=0.60,
    )
    engine = BilateralDualCoTEngine(config).to(device)

    # Initialize from existing champion weights
    init_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    if not init_ckpt.exists():
        init_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
    if init_ckpt.exists():
        logger.info("Initializing from champion weights: %s", init_ckpt.name)
        engine.load_checkpoint(init_ckpt)

    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Class-weighted CrossEntropy to counter 87% HOLD imbalance
    class_weights = torch.tensor([1.0, 4.0, 4.0], dtype=torch.float32, device=device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.03)

    N = len(ingress_matrix)
    indices = np.arange(N)

    # Action target mapping: 0 -> HOLD (0), 1 -> BUY (1), -1 -> SELL (2)
    action_targets = np.zeros(N, dtype=np.int64)
    action_targets[dir_targets == 1] = 1
    action_targets[dir_targets == -1] = 2

    # Binary targets for Bull / Bear specialized heads
    is_buy_targets = (dir_targets == 1).astype(np.float32)
    is_sell_targets = (dir_targets == -1).astype(np.float32)
    prm_targets = (dir_targets != 0).astype(np.float32)

    logger.info("Beginning Anti-MAE Finetuning V2 Loop for %d epochs...", epochs)
    engine.train()
    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        correct_actions = 0
        buy_correct = 0
        buy_total = 0
        sell_correct = 0
        sell_total = 0
        total_evaluated = 0

        for b_start in range(0, N, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_x = torch.from_numpy(ingress_matrix[b_idx]).float().to(device)
            b_act = torch.from_numpy(action_targets[b_idx]).long().to(device)
            b_prm = torch.from_numpy(prm_targets[b_idx]).float().to(device)
            b_buy = torch.from_numpy(is_buy_targets[b_idx]).float().to(device)
            b_sell = torch.from_numpy(is_sell_targets[b_idx]).float().to(device)
            b_trap = torch.from_numpy(trap_targets[b_idx]).long().to(device)

            optimizer.zero_grad()
            
            # Forward pass through architecture
            z = engine.state_adapter(b_x)
            bull_out = engine.bull_cot(z)
            bear_out = engine.bear_cot(z)
            
            p_thought = bull_out.cot_hops[:, -1, :]
            a_thought = bear_out.cot_hops[:, -1, :]
            
            # Joint trap detection
            joint_state = torch.cat([a_thought, p_thought], dim=-1)
            trap_logits = engine.trap_detector(joint_state)
            
            # Dialectical cross-examination
            final_p, final_a, _, _, _, _ = engine.cross_examination(
                p_thought, a_thought, turns=engine.config.num_debate_turns
            )
            
            # Metacognitive Arbiter
            synthesis = torch.cat([final_p, final_a], dim=-1)
            arb_logits = engine.arbiter.arbiter_net(synthesis) # (B, 3) [HOLD, BUY, SELL]
            prm_scores = engine.arbiter.prm_evaluator(synthesis).squeeze(-1) # (B,)
            
            b_score = bull_out.bull_score.squeeze(-1)
            r_score = bear_out.bear_score.squeeze(-1)
            
            # Multi-Task Losses:
            loss_action = criterion_ce(arb_logits, b_act)
            loss_bull = F.binary_cross_entropy(b_score, b_buy)
            loss_bear = F.binary_cross_entropy(r_score, b_sell)
            loss_prm = F.binary_cross_entropy(prm_scores, b_prm)
            loss_trap = criterion_ce(trap_logits, b_trap)
            
            total_b_loss = loss_action + 0.6 * loss_bull + 0.6 * loss_bear + 0.6 * loss_prm + 0.3 * loss_trap
            total_b_loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += total_b_loss.item()
            batches += 1
            preds = arb_logits.argmax(dim=-1)
            correct_actions += (preds == b_act).sum().item()
            buy_mask = (b_act == 1)
            buy_total += buy_mask.sum().item()
            buy_correct += ((preds == 1) & buy_mask).sum().item()
            sell_mask = (b_act == 2)
            sell_total += sell_mask.sum().item()
            sell_correct += ((preds == 2) & sell_mask).sum().item()
            total_evaluated += len(b_act)

        scheduler.step()
        acc = (correct_actions / max(1, total_evaluated)) * 100.0
        b_acc = (buy_correct / max(1, buy_total)) * 100.0
        s_acc = (sell_correct / max(1, sell_total)) * 100.0
        avg_loss = total_loss / max(1, batches)
        logger.info(
            "Epoch %d/%d - Loss: %.4f | Overall Acc: %.2f%% | BUY Recall: %.1f%% (%d/%d) | SELL Recall: %.1f%% (%d/%d) | LR: %.6f",
            ep, epochs, avg_loss, acc, b_acc, buy_correct, buy_total, s_acc, sell_correct, sell_total, scheduler.get_last_lr()[0]
        )

    # Save finetuned precision champion checkpoint
    engine.eval()
    out_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(out_path, metadata={
        "finetuned_anti_mae_v2": True,
        "max_mae_threshold": 0.45,
        "min_mfe_threshold": 1.80,
        "overall_acc": acc,
        "buy_recall": b_acc,
        "sell_recall": s_acc,
        "epochs": epochs,
    })
    logger.info("Finetuned Precision Champion V2 successfully saved to: %s", out_path)


if __name__ == "__main__":
    train_champions_v2(epochs=8, batch_size=64, lr=1.2e-4)
