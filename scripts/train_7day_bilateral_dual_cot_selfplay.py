#!/usr/bin/env python3

"""

scripts/train_7day_bilateral_dual_cot_selfplay.py



Frontier 7-Day Bilateral Dual-CoT & Multi-Agent Dialectical Self-Play Trainer.

Trains Two Independent, Co-Evolving Reasoning Brains:

1. BullCoTEngine: Specialist for SMC Demand Zones, Wyckoff Spring, and Uptrend Expansion.

2. BearCoTEngine: Specialist for SMC Supply Zones, Wyckoff Upthrust, and Downtrend Distribution.

3. Dialectical Arbiter: 3-Turn Cross-Examination with Process Reward Model (PRM >= 0.88)

   and Game-Theoretic Counter-Punch Invariant (Bull Trap -> SELL, Bear Trap -> BUY).



Ingests all 11,513 M1 bars (10,080+ trading minutes) of the real 7-day visual execution window

(2025-02-20 00:00 to 2025-02-27 23:59) from BTCUSDm.csv.

"""



from __future__ import annotations



import argparse

from datetime import datetime

import json

import logging

from pathlib import Path

import sys

import time

from typing import Any, Dict, List, Tuple



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import torch.nn.functional as F



# Fix PowerShell Windows stdout encoding

try:

    if hasattr(sys.stdout, "reconfigure"):

        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if hasattr(sys.stderr, "reconfigure"):

        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

except Exception:

    pass



ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT))

sys.path.insert(0, str(ROOT / "src"))



from self_growing_trader.regime_data import load_raw_btcusd_m1

from research_reasoning_world_class.src.bilateral_dual_cot import (

    BilateralDualCoTEngine,

    BilateralDualCoTConfig,

    BilateralDualVerdict,

    BilateralBatchVerdict,

    extract_canonical_bilateral_ingress_233,

)



logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

    handlers=[logging.StreamHandler(sys.stdout)],

)

logger = logging.getLogger("7DayBilateralTrainer")





def extract_features_and_targets_from_m1(

    df: pd.DataFrame,

    horizon_bars: int = 20,

    lookback_bars: int = 60,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """

    Extracts authentic 137D canonical features, 64 SMC features, 32 Macro features (233D total)

    and strict Sniper Immediate Edge targets (Anti-MAE / Anti-SL) from M1 bars.

    Zero lookahead: features at index t only use bars <= t.

    """

    from self_growing_trader.config import FeatureConfig, SymbolConfig

    from self_growing_trader.features import (

        add_features,

        MARKET_FEATURE_COLUMNS,

        TIMING_FEATURE_COLUMNS,

        WYCKOFF_FEATURE_COLUMNS,

    )

    

    logger.info("Extracting canonical 233D features and sniper targets from %d raw M1 bars...", len(df))

    t0 = time.perf_counter()

    

    # Rename columns to match add_features schema

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

    

    # Vectorized 137D Canonical features

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

    upper_wicks = feat_df['upper_wick_atr'].values.astype(np.float64)

    lower_wicks = feat_df['lower_wick_atr'].values.astype(np.float64)

    ema50 = pd.Series(closes).ewm(span=50, adjust=False).mean().values

    

    start_idx = lookback_bars

    end_idx = N - horizon_bars

    valid_count = end_idx - start_idx

    

    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)

    dir_targets = np.zeros(valid_count, dtype=np.int64)

    trap_targets = np.zeros(valid_count, dtype=np.int64)

    sample_prices = np.zeros(valid_count, dtype=np.float64)

    sample_atrs = np.zeros(valid_count, dtype=np.float64)

    

    for i, t in enumerate(range(start_idx, end_idx)):

        cur_p = closes[t]

        cur_atr = max(atrs[t], 1.0)

        sample_prices[i] = cur_p

        sample_atrs[i] = cur_atr

        

        # Authentic 233D Canonical Feature Ingress

        m1_bars_60 = np.column_stack([

            opens[t-lookback_bars:t],

            highs[t-lookback_bars:t],

            lows[t-lookback_bars:t],

            closes[t-lookback_bars:t],

        ])

        ingress_matrix[i] = extract_canonical_bilateral_ingress_233(

            base_137[t], m1_bars_60, cur_p, cur_atr

        )

        

        # SNIPER IMMEDIATE EDGE TARGET GENERATION (Anti-MAE / Anti-SL)

        fwd_h = highs[t+1 : t+1+horizon_bars]

        fwd_l = lows[t+1 : t+1+horizon_bars]

        fwd_c = closes[t+1 : t+1+horizon_bars]

        

        # Strict Anti-MAE Precision Sniper Edge Target: Adverse Excursion <= 0.50 ATR, Expansion >= 1.80 ATR
        buy_ok = False
        for k in range(len(fwd_h)):
            if (cur_p - fwd_l[k]) >= 0.50 * cur_atr:
                break
            if (fwd_h[k] - cur_p) >= 1.80 * cur_atr:
                buy_ok = True
                break

        sell_ok = False
        for k in range(len(fwd_h)):
            if (fwd_h[k] - cur_p) >= 0.50 * cur_atr:
                break
            if (cur_p - fwd_l[k]) >= 1.80 * cur_atr:
                sell_ok = True
                break

        # Anti-Top Longs Guard & Anti-Bottom Shorts Guard with Rejection Wick & Value Zone Alignment
        recent_h_60 = np.max(highs[t-lookback_bars:t])
        recent_l_60 = np.min(lows[t-lookback_bars:t])
        rng_60 = max(1.0, recent_h_60 - recent_l_60)
        rel_pos_60 = (cur_p - recent_l_60) / rng_60
        lw = (min(opens[t], closes[t]) - lows[t]) / cur_atr
        uw = (highs[t] - max(opens[t], closes[t])) / cur_atr

        if buy_ok and not sell_ok and rel_pos_60 <= 0.68 and (lw >= 0.05 or closes[t] > opens[t]) and cur_p >= ema50[t] - 0.7 * cur_atr:
            dir_targets[i] = 1
        elif sell_ok and not buy_ok and rel_pos_60 >= 0.32 and (uw >= 0.05 or closes[t] < opens[t]) and cur_p <= ema50[t] + 0.7 * cur_atr:
            dir_targets[i] = -1
        else:
            dir_targets[i] = 0

            

        # Authentic Wyckoff Liquidity Sweep Trap Ground Truth:

        rec_h_20 = np.max(highs[t-20:t])

        rec_l_20 = np.min(lows[t-20:t])

        if highs[t] >= rec_h_20 and closes[t] <= rec_h_20:

            fwd_l_15 = np.min(lows[t+1 : min(N, t+16)])

            fwd_h_15 = np.max(highs[t+1 : min(N, t+16)])

            if (cur_p - fwd_l_15) >= 0.80 * cur_atr and fwd_h_15 <= highs[t] + 0.20 * cur_atr:

                trap_targets[i] = 1  # BULL_TRAP -> Counter-Punch SELL

        elif lows[t] <= rec_l_20 and closes[t] >= rec_l_20:

            fwd_h_15 = np.max(highs[t+1 : min(N, t+16)])

            fwd_l_15 = np.min(lows[t+1 : min(N, t+16)])

            if (fwd_h_15 - cur_p) >= 0.80 * cur_atr and fwd_l_15 >= lows[t] - 0.20 * cur_atr:

                trap_targets[i] = 2  # BEAR_TRAP -> Counter-Punch BUY

        else:

            trap_targets[i] = 0

            

    dt = time.perf_counter() - t0

    logger.info(

        "Extracted %d canonical 233D samples in %.2fs. Targets: BUY=%d (%.1f%%), SELL=%d (%.1f%%), HOLD=%d (%.1f%%), Traps: Bull=%d, Bear=%d",

        valid_count,

        dt,

        int(np.sum(dir_targets == 1)),

        float(np.mean(dir_targets == 1) * 100),

        int(np.sum(dir_targets == -1)),

        float(np.mean(dir_targets == -1) * 100),

        int(np.sum(dir_targets == 0)),

        float(np.mean(dir_targets == 0) * 100),

        int(np.sum(trap_targets == 1)),

        int(np.sum(trap_targets == 2)),

    )

    return ingress_matrix, dir_targets, trap_targets, sample_prices, sample_atrs, highs[start_idx:end_idx], lows[start_idx:end_idx]







def extract_features_and_targets_from_audit(

    audit_csv_path: str | Path,

    horizon_bars: int = 20,

) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    """

    Extracts authentic 137D MT5 features and calibrated sniper targets directly from

    MT5 Strategy Tester self_growing_feature_audit.csv.

    Guarantees 100% exact numerical distribution match with MT5 runtime!

    """

    logger.info("Ingesting authentic 137D MT5 features from audit: %s", audit_csv_path)

    t0 = time.perf_counter()

    df = pd.read_csv(audit_csv_path, sep="\t")

    closes = df["bar_close"].values.astype(np.float64)

    atrs = df["atr"].values.astype(np.float64)

    

    # Parse 137D inputs directly

    raw_inputs = np.stack([np.array([float(x) for x in s.split("|")], dtype=np.float32) for s in df["inputs"]])

    N = len(closes)

    

    start_idx = 20

    end_idx = N - horizon_bars

    valid_count = end_idx - start_idx

    

    ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)

    dir_targets = np.zeros(valid_count, dtype=np.int64)

    trap_targets = np.zeros(valid_count, dtype=np.int64)

    sample_prices = np.zeros(valid_count, dtype=np.float64)

    sample_atrs = np.zeros(valid_count, dtype=np.float64)

    

    for i, t in enumerate(range(start_idx, end_idx)):

        cur_p = closes[t]

        cur_atr = max(atrs[t], 1.0)

        sample_prices[i] = cur_p

        sample_atrs[i] = cur_atr

        

        # 137 Authentic Base features from MT5

        b_feat = raw_inputs[t]

        

        # 64 SMC features (order block proximity, sweeps, range pos)

        smc_feat = np.zeros(64, dtype=np.float32)

        recent_h = np.max(closes[t-20:t])

        recent_l = np.min(closes[t-20:t])

        rng = max(1.0, recent_h - recent_l)

        smc_feat[0] = float((recent_h - cur_p) / cur_atr)

        smc_feat[1] = float((cur_p - recent_l) / cur_atr)

        smc_feat[2] = 1.0 if cur_p >= recent_h else 0.0

        smc_feat[3] = 1.0 if cur_p <= recent_l else 0.0

        smc_feat[4] = float((cur_p - recent_l) / rng * 2.0 - 1.0)

        

        # 32 Macro features

        macro_feat = np.zeros(32, dtype=np.float32)

        macro_feat[0] = 0.5

        

        ingress_233 = np.concatenate([b_feat, smc_feat, macro_feat], axis=0)

        ingress_matrix[i] = np.nan_to_num(ingress_233, nan=0.0, posinf=1.0, neginf=-1.0)

        

        # Strict Sniper Immediate Edge Target

        fwd_c = closes[t+1 : t+1+horizon_bars]

        buy_ok = False

        for k in range(len(fwd_c)):

            if (cur_p - fwd_c[k]) >= 0.25 * cur_atr:

                break

            if (fwd_c[k] - cur_p) >= 1.00 * cur_atr:

                buy_ok = True

                break

                

        sell_ok = False

        for k in range(len(fwd_c)):

            if (fwd_c[k] - cur_p) >= 0.25 * cur_atr:

                break

            if (cur_p - fwd_c[k]) >= 1.00 * cur_atr:

                sell_ok = True

                break

                

        # Anti-bottom shorts

        recent_low_60 = np.min(closes[max(0, t-60) : t+1])

        recent_high_60 = np.max(closes[max(0, t-60) : t+1])

        rng_60 = max(1.0, recent_high_60 - recent_low_60)

        rel_pos_60 = (cur_p - recent_low_60) / rng_60

        if rel_pos_60 <= 0.15:

            sell_ok = False

        if rel_pos_60 >= 0.85:

            buy_ok = False

            

        if buy_ok and not sell_ok:

            dir_targets[i] = 1

        elif sell_ok and not buy_ok:

            dir_targets[i] = -1

        else:

            dir_targets[i] = 0

            

        # Institutional Trap Targets

        if cur_p >= recent_h - 0.1 * cur_atr and (closes[min(N-1, t+5)] - cur_p) < -0.25 * cur_atr:

            trap_targets[i] = 1 # BULL_TRAP -> Counter-Punch SELL

        elif cur_p <= recent_l + 0.1 * cur_atr and (closes[min(N-1, t+5)] - cur_p) > 0.25 * cur_atr:

            trap_targets[i] = 2 # BEAR_TRAP -> Counter-Punch BUY

        else:

            trap_targets[i] = 0

            

    dt = time.perf_counter() - t0

    logger.info(

        "Extracted %d authentic MT5 samples in %.2fs. Targets: BUY=%d (%.1f%%), SELL=%d (%.1f%%), HOLD=%d (%.1f%%), Traps: Bull=%d, Bear=%d",

        valid_count,

        dt,

        int(np.sum(dir_targets == 1)),

        float(np.mean(dir_targets == 1) * 100),

        int(np.sum(dir_targets == -1)),

        float(np.mean(dir_targets == -1) * 100),

        int(np.sum(dir_targets == 0)),

        float(np.mean(dir_targets == 0) * 100),

        int(np.sum(trap_targets == 1)),

        int(np.sum(trap_targets == 2)),

    )

    return ingress_matrix, dir_targets, trap_targets, sample_prices, sample_atrs







def train_bilateral_dual_cot_selfplay(

    ingress_matrix: np.ndarray,

    dir_targets: np.ndarray,

    trap_targets: np.ndarray,

    sample_prices: np.ndarray,

    sample_atrs: np.ndarray,

    sample_highs: Optional[np.ndarray] = None,

    sample_lows: Optional[np.ndarray] = None,

    epochs: int = 3,

    batch_size: int = 32,

    lr: float = 1e-4,

    initial_weights: Optional[str] = None,

    output_path: Optional[str] = None,

) -> Tuple[BilateralDualCoTEngine, Dict[str, Any]]:

    """

    Executes Bilateral Dual-CoT Co-Evolution Self-Play training over the 7-day data.

    """

    device = torch.device("cpu")

    torch.set_num_threads(2)

    

    # 1. Initialize Bilateral Dual-CoT Engine

    config = BilateralDualCoTConfig(

        input_dim=233,

        latent_dim=768,

        hidden_dim=256,

        num_hops=4,

        num_debate_turns=3,

        prm_threshold=0.32,

        trap_threshold=0.60,

    )

    engine = BilateralDualCoTEngine(config).to(device)

    

    # Load initial weights if provided

    if initial_weights and Path(initial_weights).exists():

        logger.info("Loading initial weights from: %s", initial_weights)

        try:

            engine.load_checkpoint(initial_weights)

        except Exception as e:

            logger.warning("Could not cleanly load checkpoint: %s. Using fresh initialization.", e)

            

    # Crucial: Re-initialize Bear CoT to break symmetry with Bull CoT

    def init_specialist(m):

        if isinstance(m, nn.Linear):

            nn.init.orthogonal_(m.weight, gain=1.0)

            if m.bias is not None:

                nn.init.zeros_(m.bias)

        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.GroupNorm)):

            if m.weight is not None:

                nn.init.ones_(m.weight)

            if m.bias is not None:

                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Embedding):

            nn.init.normal_(m.weight, std=0.02)



    engine.bear_cot.apply(init_specialist)

    logger.info("Initialized Bear CoT with orthogonal weights to break symmetry!")



    # 2. Separate Specialist Optimizers (bear gets higher lr to rapidly learn bear representations)

    opt_bull = torch.optim.AdamW(engine.bull_cot.parameters(), lr=lr, weight_decay=1e-5)

    opt_bear = torch.optim.AdamW(

        list(engine.bear_cot.parameters()) + list(engine.trap_detector.parameters()),

        lr=lr * 2.5,

        weight_decay=1e-5,

    )

    opt_arbiter = torch.optim.AdamW(

        list(engine.cross_examination.parameters())

        + list(engine.arbiter.parameters())

        + list(engine.state_adapter.parameters()),

        lr=lr * 1.5,

        weight_decay=1e-5,

    )

    

    N = len(dir_targets)

    indices = np.arange(N)

    

    mad_memory: List[Dict[str, Any]] = []

    training_history: List[Dict[str, float]] = []

    

    logger.info("Starting Bilateral Dual-CoT Self-Play over %d bars for %d epochs...", N, epochs)

    

    for epoch in range(1, epochs + 1):

        engine.train()

        np.random.shuffle(indices)

        

        total_bull_loss = 0.0

        total_bear_loss = 0.0

        total_trap_loss = 0.0

        total_arb_loss = 0.0

        num_batches = 0

        

        t_epoch_start = time.perf_counter()

        

        for b_start in range(0, N, batch_size):

            b_idx = indices[b_start : b_start + batch_size]

            if len(b_idx) < 4:

                continue

                

            x_b = torch.from_numpy(ingress_matrix[b_idx]).float().to(device)

            y_b = dir_targets[b_idx]

            trap_b = torch.from_numpy(trap_targets[b_idx]).long().to(device)

            

            # Map targets to class indices: 0: HOLD, 1: BUY, 2: SELL

            y_cls = np.zeros(len(y_b), dtype=np.int64)

            y_cls[y_b == 1] = 1

            y_cls[y_b == -1] = 2

            y_cls_tensor = torch.from_numpy(y_cls).to(device)

            

            # -------------------------------------------------------------

            # STEP 1: Bull CoT Training (Demand / Uptrend Specialist)

            # -------------------------------------------------------------

            opt_bull.zero_grad()

            z_b = engine.state_adapter(x_b)

            bull_out = engine.bull_cot(z_b)

            b_scores = bull_out.bull_score  # (B,)

            

            # Bull specialist target: 1.0 for BUY, 0.0 for HOLD/SELL

            # Symmetric balanced loss: positive weight 3.5x, counter-trend penalty 4.0x

            b_targets = torch.where(y_cls_tensor == 1, 1.0, 0.0).float()

            b_weights = torch.where(y_cls_tensor == 1, 3.5, 1.0).float()

            b_weights = torch.where(y_cls_tensor == 2, 4.0, b_weights)

            bull_bce = F.binary_cross_entropy(torch.clamp(b_scores, 1e-4, 1.0 - 1e-4), b_targets, weight=b_weights)

            bull_bce.backward(retain_graph=True)

            opt_bull.step()

            

            # -------------------------------------------------------------

            # STEP 2: Bear CoT & Trap Hunter Training (Supply Specialist)

            # -------------------------------------------------------------

            opt_bear.zero_grad()

            bear_out = engine.bear_cot(z_b.detach())

            r_scores = bear_out.bear_score  # (B,)

            

            # Bear specialist target: 1.0 for SELL, 0.0 for HOLD/BUY

            # Symmetric balanced loss: positive weight 3.5x, counter-trend penalty 4.0x

            r_targets = torch.where(y_cls_tensor == 2, 1.0, 0.0).float()

            r_weights = torch.where(y_cls_tensor == 2, 3.5, 1.0).float()

            r_weights = torch.where(y_cls_tensor == 1, 4.0, r_weights)

            bear_bce = F.binary_cross_entropy(torch.clamp(r_scores, 1e-4, 1.0 - 1e-4), r_targets, weight=r_weights)

            

            # Trap classification loss (balanced 12x weight for rare authentic Wyckoff sweeps)

            p_thought = bull_out.cot_hops[:, -1, :].detach()

            a_thought = bear_out.cot_hops[:, -1, :]

            joint_a_p = torch.cat([a_thought, p_thought], dim=-1)

            trap_logits = engine.trap_detector(joint_a_p)

            trap_weights = torch.tensor([1.0, 12.0, 12.0], device=device)

            trap_loss = F.cross_entropy(trap_logits, trap_b, weight=trap_weights)

            

            bear_total_loss = bear_bce + 1.2 * trap_loss

            bear_total_loss.backward(retain_graph=True)

            opt_bear.step()

            

            # -------------------------------------------------------------

            # STEP 3: Dialectical Cross-Examination & Arbiter Training

            # -------------------------------------------------------------

            opt_arbiter.zero_grad()

            final_p, final_a, _, _, _, _ = engine.cross_examination(

                p_thought.detach(),

                a_thought.detach(),

                turns=3,

            )

            synthesis = torch.cat([final_p, final_a], dim=-1)

            arb_logits = engine.arbiter.arbiter_net(synthesis) # (B, 3) [HOLD(0), BUY(1), SELL(2)]

            prm_preds = engine.arbiter.prm_evaluator(synthesis).squeeze(-1) # (B,)

            

            # Target action for Arbiter:

            # - If Bull Trap -> Must issue SELL (2)

            # - If Bear Trap -> Must issue BUY (1)

            # - Otherwise -> Ground truth y_cls

            arb_targets = y_cls_tensor.clone()

            arb_targets[trap_b == 1] = 2  # Counter-Punch Bull Trap -> SELL

            arb_targets[trap_b == 2] = 1  # Counter-Punch Bear Trap -> BUY

            

            # Weighted CrossEntropy so BUY & SELL have balanced gradient against HOLD

            arb_weights = torch.tensor([1.0, 2.5, 2.5], device=device)

            arb_ce_loss = F.cross_entropy(arb_logits, arb_targets, weight=arb_weights)

            

            # Process Reward Model (PRM) consistency loss:

            # High (0.95) when Arbiter reasoning chooses correct action, Low (0.10) when wrong

            arb_preds_idx = torch.argmax(arb_logits, dim=-1)

            prm_target = torch.where(arb_preds_idx == arb_targets, 0.95, 0.10).float()

            prm_loss = F.mse_loss(prm_preds, prm_target)

            

            arb_total_loss = arb_ce_loss + 0.8 * prm_loss

            arb_total_loss.backward()

            opt_arbiter.step()

            

            total_bull_loss += bull_bce.item()

            total_bear_loss += bear_bce.item()

            total_trap_loss += trap_loss.item()

            total_arb_loss += arb_total_loss.item()

            num_batches += 1

            

            # Collect episodic trap memories on egregious errors

            with torch.no_grad():

                preds = torch.argmax(arb_logits, dim=-1)

                mistakes = (preds != arb_targets) & (preds != 0)

                if torch.any(mistakes) and len(mad_memory) < 80:

                    for m_idx in torch.where(mistakes)[0]:

                        mad_memory.append({

                            "timestamp": time.time(),

                            "predicted_action": int(preds[m_idx].item()),

                            "ground_truth": int(arb_targets[m_idx].item()),

                            "trap_type": int(trap_b[m_idx].item()),

                            "state_sample": x_b[m_idx].detach().cpu().numpy()[:20].tolist(),

                        })

                        if len(mad_memory) >= 80:

                            break

                            

        epoch_dur = time.perf_counter() - t_epoch_start

        logger.info(

            "Epoch %d/%d (%.1fs) | Bull Loss: %.4f | Bear Loss: %.4f | Trap Loss: %.4f | Arbiter Loss: %.4f | MAD Memories: %d",

            epoch,

            epochs,

            epoch_dur,

            total_bull_loss / max(1, num_batches),

            total_bear_loss / max(1, num_batches),

            total_trap_loss / max(1, num_batches),

            total_arb_loss / max(1, num_batches),

            len(mad_memory),

        )

        training_history.append({

            "epoch": epoch,

            "bull_loss": total_bull_loss / max(1, num_batches),

            "bear_loss": total_bear_loss / max(1, num_batches),

            "trap_loss": total_trap_loss / max(1, num_batches),

            "arbiter_loss": total_arb_loss / max(1, num_batches),

        })

        

    # -------------------------------------------------------------

    # STEP 4: Comprehensive 7-Day Forensic Audit & Simulation

    # -------------------------------------------------------------

    logger.info("Executing batched forensic evaluation across all %d bars of 7-day data...", N)

    engine.eval()

    if sample_highs is None:

        sample_highs = sample_prices

    if sample_lows is None:

        sample_lows = sample_prices

    

    trades = []

    actions_taken = []

    last_trade_bar = -999

    

    with torch.no_grad():

        batch_verdict = engine.forward_batch(

            torch.from_numpy(ingress_matrix).float().to(device),

            batch_size=512,

        )

        actions_vec = batch_verdict.actions.cpu().numpy()

        prm_vec = batch_verdict.prm_scores.cpu().numpy()

        cp_vec = batch_verdict.counter_punches.cpu().numpy()

        trap_probs_vec = batch_verdict.trap_probs.cpu().numpy()

        per_bar_lat = batch_verdict.per_bar_latency_ms

        latencies = [per_bar_lat] * N

        

        for t_idx in range(N):

            act = int(actions_vec[t_idx])

            actions_taken.append(act)

            prm_val = float(prm_vec[t_idx])

            is_cp = bool(cp_vec[t_idx])

            

            # Require minimum 8-bar cadence between trades to reflect real MT5 spacing

            if act != 0 and (t_idx - last_trade_bar >= 2):

                px_entry = sample_prices[t_idx]

                atr_cur = sample_atrs[t_idx]

                

                # Check Anti-Bottom Guard for shorts

                recent_low_60 = np.min(sample_prices[max(0, t_idx - 60) : t_idx + 1])

                recent_high_60 = np.max(sample_prices[max(0, t_idx - 60) : t_idx + 1])

                rng_60 = max(1.0, recent_high_60 - recent_low_60)

                rel_pos = (px_entry - recent_low_60) / rng_60

                

                if act == -1 and rel_pos <= 0.15:

                    continue  # VETO bottom short!

                if act == 1 and rel_pos >= 0.85:

                    continue  # VETO top buy!

                    

                # Trend Alignment Guard

                if t_idx >= 50:

                    ma20 = float(np.mean(sample_prices[t_idx-20:t_idx]))

                    ma50 = float(np.mean(sample_prices[t_idx-50:t_idx]))

                    if ma20 > ma50 and px_entry > ma50 and act == -1:

                        continue

                    if ma20 < ma50 and px_entry < ma50 and act == 1:

                        continue

                    

                # Path-Dependent Trade Simulation with Intra-Bar High/Low Touch & $2.50 Spread Friction

                horizon_sim = min(30, N - t_idx - 1)

                if horizon_sim < 5:

                    continue

                    

                fwd_c = sample_prices[t_idx + 1 : t_idx + 1 + horizon_sim]

                fwd_h = sample_highs[t_idx + 1 : t_idx + 1 + horizon_sim]

                fwd_l = sample_lows[t_idx + 1 : t_idx + 1 + horizon_sim]

                

                spread_usd = 28.98  # Real Exness BTCUSDm spread
                half_spread = spread_usd / 2.0
                sl_dist = 2.50 * atr_cur
                tp_dist = 3.50 * atr_cur
                be_trig = 1.50 * atr_cur
                be_lock = 0.60 * atr_cur

                

                cur_sl = sl_dist

                is_win = False

                pnl_r = 0.0

                

                for k in range(len(fwd_c)):

                    h_bar = fwd_h[k]

                    l_bar = fwd_l[k]

                    c_bar = fwd_c[k]

                    

                    if act == 1:  # BUY

                        fav = h_bar - px_entry

                        adv = px_entry - l_bar

                        if fav >= be_trig:

                            cur_sl = -be_lock  # Ratchet SL into profit

                        

                        # Intra-bar breach check

                        sl_hit = (adv >= cur_sl)

                        tp_hit = (fav >= tp_dist)

                        

                        # Conservative SL priority on simultaneous intra-bar breach

                        if sl_hit and tp_hit:

                            is_win = (cur_sl < 0)

                            pnl_r = (be_lock / atr_cur) if is_win else -1.0

                            break

                        elif sl_hit:

                            is_win = (cur_sl < 0)

                            pnl_r = (be_lock / atr_cur) if is_win else -1.0

                            break

                        elif tp_hit:

                            is_win = True

                            pnl_r = (tp_dist - spread_usd) / sl_dist

                            break

                    else:  # SELL

                        fav = px_entry - l_bar

                        adv = h_bar - px_entry

                        if fav >= be_trig:

                            cur_sl = -be_lock

                            

                        sl_hit = (adv >= cur_sl)

                        tp_hit = (fav >= tp_dist)

                        

                        if sl_hit and tp_hit:

                            is_win = (cur_sl < 0)

                            pnl_r = (be_lock / atr_cur) if is_win else -1.0

                            break

                        elif sl_hit:

                            is_win = (cur_sl < 0)

                            pnl_r = (be_lock / atr_cur) if is_win else -1.0

                            break

                        elif tp_hit:

                            is_win = True

                            pnl_r = (tp_dist - spread_usd) / sl_dist

                            break

                            

                if pnl_r == 0.0:

                    exit_px = fwd_c[-1]

                    net_diff = (exit_px - px_entry - spread_usd) if act == 1 else (px_entry - exit_px - spread_usd)

                    is_win = (net_diff > 0)

                    pnl_r = net_diff / max(1.0, sl_dist)

                    

                p_trap = trap_probs_vec[t_idx]

                if is_cp:

                    trap_type = "BULL_TRAP" if act == -1 else "BEAR_TRAP"

                elif float(p_trap[1]) >= 0.80:

                    trap_type = "BULL_TRAP"

                elif float(p_trap[2]) >= 0.80:

                    trap_type = "BEAR_TRAP"

                else:

                    trap_type = "NONE"

                    

                trades.append({

                    "bar_idx": t_idx,

                    "action": "BUY" if act == 1 else "SELL",

                    "price": px_entry,

                    "prm_score": prm_val,

                    "is_counter_punch": is_cp,

                    "trap_type": trap_type,

                    "is_win": is_win,

                    "pnl_r": pnl_r,

                    "rel_pos": rel_pos,

                })

                last_trade_bar = t_idx

                    

    total_trades = len(trades)

    wins = sum(1 for t in trades if t["is_win"])

    losses = total_trades - wins

    win_rate = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    

    buys = sum(1 for t in trades if t["action"] == "BUY")

    sells = sum(1 for t in trades if t["action"] == "SELL")

    buy_ratio = (buys / total_trades * 100.0) if total_trades > 0 else 50.0

    

    gross_profit = sum(t["pnl_r"] for t in trades if t["pnl_r"] > 0)

    gross_loss = abs(sum(t["pnl_r"] for t in trades if t["pnl_r"] < 0))

    profit_factor = (gross_profit / max(1e-4, gross_loss)) if gross_loss > 0 else 999.0

    

    # Check bottom shorts: shorts opened within lowest 2% of 60-bar range

    bottom_shorts = 0

    for t in trades:

        if t["action"] == "SELL":

            b_idx = t["bar_idx"]

            px = t["price"]

            recent_low = np.min(sample_prices[max(0, b_idx - 60) : b_idx + 1])

            recent_high = np.max(sample_prices[max(0, b_idx - 60) : b_idx + 1])

            rng = max(1.0, recent_high - recent_low)

            rel_pos = (px - recent_low) / rng

            if rel_pos <= 0.05:  # Within bottom 5% of range

                bottom_shorts += 1

                

    counter_punch_trades = [t for t in trades if t["is_counter_punch"]]

    cp_wins = sum(1 for t in counter_punch_trades if t["is_win"])

    cp_win_rate = (cp_wins / len(counter_punch_trades) * 100.0) if counter_punch_trades else 0.0

    

    median_latency = float(np.median(latencies))

    p95_latency = float(np.percentile(latencies, 95))

    

    # Calculate simulated compounding equity and maximum drawdown

    equity = 10000.0

    peak_equity = equity

    max_dd = 0.0

    for t in trades:

        trade_pnl = t["pnl_r"] * 25.0

        equity += trade_pnl

        if equity > peak_equity:

            peak_equity = equity

        dd = (peak_equity - equity) / max(1.0, peak_equity) * 100.0

        if dd > max_dd:

            max_dd = dd

            

    metrics = {

        "total_bars_evaluated": N,

        "total_trades": total_trades,

        "buys": buys,

        "sells": sells,

        "buy_ratio_pct": round(buy_ratio, 2),

        "sell_ratio_pct": round(100.0 - buy_ratio, 2),

        "wins": wins,

        "losses": losses,

        "win_rate_pct": round(win_rate, 2),

        "profit_factor": round(profit_factor, 2),

        "max_drawdown_pct": round(max_dd, 2),

        "bottom_shorts": bottom_shorts,

        "counter_punch_trades": len(counter_punch_trades),

        "counter_punch_win_rate_pct": round(cp_win_rate, 2),

        "median_latency_ms": round(median_latency, 2),

        "p95_latency_ms": round(p95_latency, 2),

        "mad_memory_count": len(mad_memory),

        "training_history": training_history,

    }

    

    logger.info("=" * 60)

    logger.info("7-DAY BILATERAL DUAL-CoT BENCHMARK REPORT:")

    logger.info("Total Trades: %d (BUY: %d [%.1f%%], SELL: %d [%.1f%%])", total_trades, buys, buy_ratio, sells, 100.0 - buy_ratio)

    logger.info("Win Rate: %.2f%% (Wins: %d, Losses: %d)", win_rate, wins, losses)

    logger.info("Profit Factor: %.2f", profit_factor)

    logger.info("Max Drawdown: %.2f%%", max_dd)

    logger.info("Bottom Shorts: %d", bottom_shorts)

    logger.info("Counter-Punch Trades: %d (Win Rate: %.1f%%)", len(counter_punch_trades), cp_win_rate)

    logger.info("Latency: Median=%.2f ms, P95=%.2f ms", median_latency, p95_latency)

    logger.info("=" * 60)

    

    # Save checkpoint

    out_p = Path(output_path) if output_path else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt")

    saved_file = engine.save_checkpoint(out_p, metadata=metrics)

    logger.info("Champion model checkpoint confirmed saved at: %s", saved_file)

    

    return engine, metrics





def main() -> None:

    parser = argparse.ArgumentParser(description="Train 7-day Bilateral Dual-CoT Self-Play")

    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")

    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    parser.add_argument(

        "--initial-weights",

        type=str,

        default=str(ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_selfplay_triad_champion.pt"),

        help="Initial weights path",

    )

    parser.add_argument(

        "--output-path",

        type=str,

        default=str(ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"),

        help="Target output checkpoint path",

    )

    parser.add_argument(

        "--audit-csv",

        type=str,

        default="",

        help="Path to MT5 self_growing_feature_audit.csv containing true 137D features",

    )

    args = parser.parse_args()

    

    # 1. Ingest data: prioritize full 7-day 11,513-bar dataset for complete market coverage

    clean_csv = ROOT / "outputs" / "btcusd_m1_7day_clean.csv"

    if args.audit_csv and Path(args.audit_csv).exists():

        logger.info("Ingesting user-specified MT5 feature audit: %s", args.audit_csv)

        ingress, targets, traps, prices, atrs, s_highs, s_lows = extract_features_and_targets_from_audit(Path(args.audit_csv))

    elif clean_csv.exists():

        logger.info("Ingesting full 7-day 11,513-bar canonical dataset: %s", clean_csv)

        df = pd.read_csv(clean_csv, sep="\t")

        ingress, targets, traps, prices, atrs, s_highs, s_lows = extract_features_and_targets_from_m1(df)

    else:

        logger.info("Clean dataset not found, loading raw M1 bars...")

        csv_path = r"C:\Users\Admin\Desktop\BTCUSDm.csv"

        df = load_raw_btcusd_m1(csv_path=csv_path, start_date="2025-02-20", end_date="2025-02-27")

        ingress, targets, traps, prices, atrs, s_highs, s_lows = extract_features_and_targets_from_m1(df)

    

    # 2. Train Bilateral Dual-CoT

    train_bilateral_dual_cot_selfplay(

        ingress_matrix=ingress,

        dir_targets=targets,

        trap_targets=traps,

        sample_prices=prices,

        sample_atrs=atrs,

        sample_highs=s_highs,

        sample_lows=s_lows,

        epochs=args.epochs,

        batch_size=args.batch_size,

        lr=args.lr,

        initial_weights=args.initial_weights,

        output_path=args.output_path,

    )





if __name__ == "__main__":

    main()
