import sys
from pathlib import Path
import time
import logging
import torch
import torch.nn as nn
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
logger = logging.getLogger("AntiMAETrainer")


def prepare_dataset():
    clean_csv = ROOT / "outputs" / "btcusd_m1_7day_clean.csv"
    logger.info("Loading clean 7-day canonical M1 data: %s", clean_csv)
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
    horizon_bars = 25
    valid_count = N - horizon_bars - lookback_bars

    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
    dir_targets = np.zeros(valid_count, dtype=np.int64)
    trap_targets = np.zeros(valid_count, dtype=np.int64)
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

        rec_l_20 = np.min(lows[t-20:t])
        rec_h_20 = np.max(highs[t-20:t])
        rng_20 = max(1.0, rec_h_20 - rec_l_20)
        rel_pos_20 = (cur_p - rec_l_20) / rng_20

        lw = (min(opens[t], closes[t]) - lows[t]) / cur_atr
        uw = (highs[t] - max(opens[t], closes[t])) / cur_atr

        f_h = highs[t+1 : t+1+horizon_bars]
        f_l = lows[t+1 : t+1+horizon_bars]

        # BUY Pivot Target: Low never drops > 0.50 ATR, High reaches >= 1.80 ATR
        buy_target = False
        for k in range(len(f_h)):
            if (cur_p - f_l[k]) >= 0.50 * cur_atr:
                break
            if (f_h[k] - cur_p) >= 1.80 * cur_atr:
                buy_target = True
                break

        # SELL Pivot Target: High never rises > 0.50 ATR, Low reaches >= 1.80 ATR
        sell_target = False
        for k in range(len(f_h)):
            if (f_h[k] - cur_p) >= 0.50 * cur_atr:
                break
            if (cur_p - f_l[k]) >= 1.80 * cur_atr:
                sell_target = True
                break

        # Strict value zone and rejection filter
        if buy_target and not sell_target and rel_pos_20 <= 0.68 and (lw >= 0.05 or closes[t] > opens[t]) and cur_p >= ema50[t] - 0.7 * cur_atr:
            dir_targets[i] = 1
        elif sell_target and not buy_target and rel_pos_20 >= 0.32 and (uw >= 0.05 or closes[t] < opens[t]) and cur_p <= ema50[t] + 0.7 * cur_atr:
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


def train_champions(epochs: int = 6, batch_size: int = 64, lr: float = 1.5e-4):
    ingress_matrix, dir_targets, trap_targets, sample_prices, sample_atrs = prepare_dataset()
    device = torch.device("cpu")

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

    # Initialize from existing champion checkpoint
    init_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    if not init_ckpt.exists():
        init_ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
    if init_ckpt.exists():
        logger.info("Initializing from champion weights: %s", init_ckpt)
        engine.load_checkpoint(init_ckpt)

    optimizer = optim.AdamW(engine.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion_ce = nn.CrossEntropyLoss(label_smoothing=0.05)
    criterion_bce = nn.BCEWithLogitsLoss()

    N = len(ingress_matrix)
    indices = np.arange(N)

    # Action target mapping: 0 -> HOLD (0), 1 -> BUY (1), -1 -> SELL (2)
    action_targets = np.zeros(N, dtype=np.int64)
    action_targets[dir_targets == 1] = 1
    action_targets[dir_targets == -1] = 2

    # Binary PRM target: 1.0 for valid high-precision setups, 0.0 for chop/hold
    prm_targets = (dir_targets != 0).astype(np.float32)

    logger.info("Beginning Anti-MAE Finetuning Loop for %d epochs...", epochs)
    engine.train()
    for ep in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        batches = 0
        correct_actions = 0
        total_evaluated = 0

        for b_start in range(0, N, batch_size):
            b_idx = indices[b_start : b_start + batch_size]
            b_x = torch.from_numpy(ingress_matrix[b_idx]).float().to(device)
            b_act = torch.from_numpy(action_targets[b_idx]).long().to(device)
            b_prm = torch.from_numpy(prm_targets[b_idx]).float().to(device)
            b_trap = torch.from_numpy(trap_targets[b_idx]).long().to(device)

            optimizer.zero_grad()
            batch_res = engine.forward_batch(b_x, batch_size=len(b_x))

            # Action logits loss from arbiter
            arb_logits = engine.arbiter.action_head(batch_res.arbiter_hidden)
            loss_action = criterion_ce(arb_logits, b_act)

            # PRM score loss
            prm_logits = engine.arbiter.prm_head(batch_res.arbiter_hidden).squeeze(-1)
            loss_prm = criterion_bce(prm_logits, b_prm)

            # Trap detector loss
            trap_logits = engine.trap_detector(batch_res.bear_hidden)
            loss_trap = criterion_ce(trap_logits, b_trap)

            loss = loss_action + 0.6 * loss_prm + 0.4 * loss_trap
            loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            batches += 1
            preds = arb_logits.argmax(dim=-1)
            correct_actions += (preds == b_act).sum().item()
            total_evaluated += len(b_act)

        scheduler.step()
        acc = (correct_actions / max(1, total_evaluated)) * 100.0
        avg_loss = total_loss / max(1, batches)
        logger.info("Epoch %d/%d - Loss: %.4f | Action Accuracy: %.2f%% | LR: %.6f", ep, epochs, avg_loss, acc, scheduler.get_last_lr()[0])

    # Save finetuned precision champion checkpoint
    engine.eval()
    out_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    engine.save_checkpoint(out_path, metadata={
        "finetuned_anti_mae": True,
        "max_mae_threshold": 0.50,
        "min_mfe_threshold": 1.80,
        "accuracy": acc,
        "epochs": epochs,
    })
    logger.info("Finetuned Precision Champion successfully saved to: %s", out_path)


if __name__ == "__main__":
    train_champions(epochs=6, batch_size=64, lr=1.5e-4)
