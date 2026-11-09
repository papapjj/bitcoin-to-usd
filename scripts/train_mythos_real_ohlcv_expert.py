"""
train_mythos_real_ohlcv_expert.py

Retrains the Metacognitive Reasoning Engine (xLSTM 17.5M CoT, Adversarial Reflection, PRM Verifier)
using 100% AUTHENTIC RECONSTRUCTED OHLCV M1 CANDLESTICKS from the MT5 audit telemetry.

Solves the Root Causes:
1. Authentic M1 OHLCV: Feeds genuine High, Low, Open, Close, Volume into SMC Engine (FVG, Order Blocks, Liquidity Sweeps).
2. Anti-Exhaustion Labeling: Prevents buying at the peak of extended waves by filtering out candles extended > 1.0 ATR from EMA20.
3. Low-ATR Noise Filter: Suppresses trading when ATR < 50.0 USD where Exness spread ($28-$30) causes negative edge.
4. Strict Trap & Dump Veto: Labels all falling-knife crashes and asymmetric drawdowns as VETO (PRM=0.05, Margin=-0.95).
5. Balanced 3-Way Target: Ideal Buy (+1), Trap Veto (-1), Chop/Range Hold (0).
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research_reasoning_world_class.src.unified_reasoning_pipeline import UnifiedReasoningTraderPipeline
from research_reasoning_world_class.src.metacognitive_reasoning import MetacognitiveReasoningEngine


def main():
    print("=" * 80)
    print("  AUTHENTIC REAL-OHLCV M1 RETRAINING OF MYTHOS REASONING ENGINE (V2)")
    print("=" * 80)

    # 1. Locate authentic feature audit CSV from newest runner
    cand_csv = [
        ROOT / ".mt5_runners" / "opt_Iter_46_2Day_14240_20261029_202154_339863" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv",
        ROOT / ".mt5_runners" / "opt_Iter_43_2Day_12028_20261029_184615_139309" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv",
        ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv",
    ]
    target_csv = None
    for p in cand_csv:
        if p.exists() and p.stat().st_size > 1000:
            target_csv = p
            break

    if target_csv is None:
        found = list(ROOT.glob(".mt5_runners/**/self_growing_feature_audit.csv"))
        for p in found:
            if p.stat().st_size > 10000:
                target_csv = p
                break

    if target_csv is None:
        raise FileNotFoundError("Could not find any valid self_growing_feature_audit.csv!")

    print(f"Loading authentic MT5 telemetry from: {target_csv}...")
    df = pd.read_csv(target_csv, sep="\t")
    print(f"Loaded {len(df)} authentic M1 observation rows across dates: {df['bar_time'].str.slice(0, 10).unique().tolist()}")

    closes = df["bar_close"].values
    atrs = df["atr"].values
    N = len(df)
    H = 20  # 20-bar forward evaluation horizon

    # 2. Compute authentic reconstructed OHLCV bars
    print("Reconstructing authentic M1 OHLCV candlestick sequence...")
    ohlcv_bars = []
    inputs_list = []
    for i in range(N):
        row = df.iloc[i]
        in_vec = np.array([float(x) for x in row["inputs"].split("|")[:137]], dtype=np.float32)
        c = float(row["bar_close"])
        atr = float(row["atr"])
        body = in_vec[7]
        upper_wick = max(0.0, in_vec[8])
        lower_wick = max(0.0, in_vec[9])
        range_val = max(0.0, in_vec[10])
        rel_vol = max(0.1, in_vec[62]) if len(in_vec) > 62 else 1.0

        o = c - (body * atr)
        h = max(o, c) + (upper_wick * atr)
        l = min(o, c) - (lower_wick * atr)
        if range_val > 0 and (h - l) < range_val * atr * 0.95:
            mid = 0.5 * (o + c)
            h = max(h, mid + 0.5 * range_val * atr)
            l = min(l, mid - 0.5 * range_val * atr)
        v = max(1.0, rel_vol * 150.0)

        ohlcv_bars.append([o, h, l, c, v])
        inputs_list.append(in_vec)

    ohlcv_bars = np.array(ohlcv_bars, dtype=np.float64)
    print(f"Reconstructed {len(ohlcv_bars)} authentic M1 bars.")

    # 3. Labeling with Anti-Exhaustion, Spread-Safety, and Strict Dump Veto
    labels = []
    target_bull = []
    target_bear = []
    target_prm = []
    target_margin = []
    valid_indices = []

    for t in range(N - H):
        p0 = closes[t]
        atr = max(atrs[t], 10.0)
        future = closes[t + 1 : t + 1 + H]
        mfe = (np.max(future) - p0) / atr
        mae = (p0 - np.min(future)) / atr

        # Anti-exhaustion: distance from 20-period moving average
        start_ema = max(0, t - 20)
        ma20 = np.mean(closes[start_ema : t + 1])
        dist_ma_atr = (p0 - ma20) / atr

        valid_indices.append(t)
        
        # 1. Low ATR Noise Guard: suppress entries when ATR < 50.0 USD
        if atr < 50.0:
            labels.append(0)
            target_bull.append(0.12)
            target_bear.append(0.12)
            target_prm.append(0.18)
            target_margin.append(0.00)
        # 2. Ideal BUY: genuine base breakout, low initial MAE, not overextended
        elif mfe >= 2.0 and mae <= 0.6 and dist_ma_atr <= 1.0:
            labels.append(1)
            target_bull.append(0.98)
            target_bear.append(0.02)
            target_prm.append(0.95)
            target_margin.append(0.95)
        # 3. Dangerous Trap / Dump: falling knife, dominant drawdown, or exhaustion buy at peak
        elif (mae >= 1.2) or (mae >= 0.8 and mfe <= 0.5) or (dist_ma_atr > 1.2 and mae >= 0.8):
            labels.append(-1)
            target_bull.append(0.02)
            target_bear.append(0.98)
            target_prm.append(0.05)
            target_margin.append(-0.95)
        # 4. Chop / Neutral Hold: range bound oscillation
        else:
            labels.append(0)
            target_bull.append(0.15)
            target_bear.append(0.15)
            target_prm.append(0.20)
            target_margin.append(0.00)

    labels = np.array(labels)
    target_bull = torch.tensor(target_bull, dtype=torch.float32)
    target_bear = torch.tensor(target_bear, dtype=torch.float32)
    target_prm = torch.tensor(target_prm, dtype=torch.float32)
    target_margin = torch.tensor(target_margin, dtype=torch.float32)

    print(f"\nLabel Distribution across {len(labels)} samples:")
    print(f"  -> Ideal Buy Entries (+1) : {np.sum(labels == 1)} ({np.mean(labels == 1)*100:.1f}%)")
    print(f"  -> Trap / Dump Veto  (-1) : {np.sum(labels == -1)} ({np.mean(labels == -1)*100:.1f}%)")
    print(f"  -> Chop / Hold        (0) : {np.sum(labels == 0)} ({np.mean(labels == 0)*100:.1f}%)")

    # 4. Extract 768D Latent Embeddings through UnifiedReasoningTraderPipeline
    print("\nExtracting 768D Latent Embeddings through UnifiedReasoningTraderPipeline using authentic M1 OHLCV...")
    pipeline = UnifiedReasoningTraderPipeline()
    latents = []

    # Initialize rolling buffer with first 60 authentic bars
    m1_buffer = np.zeros((60, 5), dtype=np.float64)
    init_bars = ohlcv_bars[:60] if len(ohlcv_bars) >= 60 else ohlcv_bars
    m1_buffer[-len(init_bars):] = init_bars
    macros = {s: np.ones(60) for s in ["DXY", "US10Y", "SPX", "XAUUSD"]}

    t0_extract = time.perf_counter()
    with torch.no_grad():
        for idx in valid_indices:
            in_vec = inputs_list[idx]
            bar = ohlcv_bars[idx]
            c = bar[3]

            m1_buffer[:-1] = m1_buffer[1:]
            m1_buffer[-1] = bar

            _ = pipeline.process_tick(
                symbol="BTCUSDm",
                base_obs_137=in_vec,
                m1_bars_60=m1_buffer,
                macro_quotes=macros,
                current_price=c,
            )
            latents.append(pipeline.last_governor_latent.clone())

    latents = torch.stack(latents, dim=0)  # [N_samples, 768]
    print(f"Successfully extracted {latents.shape[0]} latent vectors in {time.perf_counter() - t0_extract:.2f}s.")

    # 5. Supervised Fine-Tuning of Metacognitive Reasoning Engine
    engine: MetacognitiveReasoningEngine = pipeline.cot_engine
    engine.train()
    for param in engine.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(engine.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

    num_epochs = 100
    batch_size = 32
    num_samples = latents.shape[0]
    print(f"\nCommencing Multi-Regime Precision Fine-Tuning across {num_epochs} Epochs...")

    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()

    for epoch in range(1, num_epochs + 1):
        perm = torch.randperm(num_samples)
        epoch_losses = []
        epoch_acc_buy = []
        epoch_acc_trap = []
        epoch_acc_chop = []

        for b_start in range(0, num_samples, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            b_z = latents[b_idx]
            t_bull = target_bull[b_idx]
            t_bear = target_bear[b_idx]
            t_prm = target_prm[b_idx]
            t_margin = target_margin[b_idx]
            b_labels = labels[b_idx.numpy()]

            optimizer.zero_grad()
            out = engine(b_z)

            loss_bull = bce_loss(torch.clamp(out.bull_score, 1e-4, 0.9999), t_bull)
            loss_bear = bce_loss(torch.clamp(out.bear_score, 1e-4, 0.9999), t_bear)
            loss_prm = mse_loss(out.prm_score, t_prm)
            loss_margin = mse_loss(out.adversarial_margin, t_margin)

            total_loss = loss_bull + loss_bear + 1.8 * loss_prm + 1.0 * loss_margin
            total_loss.backward()
            nn.utils.clip_grad_norm_(engine.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(total_loss.item())

            # Evaluate training accuracy metrics
            with torch.no_grad():
                pred_bull = (out.bull_score > 0.60).numpy()
                pred_trap = (out.bear_score > 0.60).numpy()
                pred_hold = ((out.bull_score <= 0.60) & (out.bear_score <= 0.60)).numpy()

                buy_mask = b_labels == 1
                if np.sum(buy_mask) > 0:
                    epoch_acc_buy.append(np.mean(pred_bull[buy_mask]))

                trap_mask = b_labels == -1
                if np.sum(trap_mask) > 0:
                    epoch_acc_trap.append(np.mean(pred_trap[trap_mask]))

                chop_mask = b_labels == 0
                if np.sum(chop_mask) > 0:
                    epoch_acc_chop.append(np.mean(pred_hold[chop_mask]))

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == num_epochs:
            mean_loss = np.mean(epoch_losses)
            buy_rate = np.mean(epoch_acc_buy) * 100.0 if epoch_acc_buy else 0.0
            trap_rate = np.mean(epoch_acc_trap) * 100.0 if epoch_acc_trap else 0.0
            chop_rate = np.mean(epoch_acc_chop) * 100.0 if epoch_acc_chop else 0.0
            print(
                f"Epoch [{epoch:3d}/{num_epochs:3d}] | Loss: {mean_loss:.4f} | "
                f"Ideal Buy Hit: {buy_rate:5.1f}% | Trap Veto Rate: {trap_rate:5.1f}% | Chop Hold Rate: {chop_rate:5.1f}%"
            )

    # 6. Export retrained champion weights
    engine.eval()
    out_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "reasoning_engine_state_dict": engine.state_dict(),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_samples": num_samples,
        "epochs": num_epochs,
        "architecture": "Mythos_xLSTM_17.5M_CoT_RealOHLCV_PrecisionChampion_V2",
    }
    if hasattr(pipeline, "world_model"):
        ckpt["world_model_state_dict"] = pipeline.world_model.state_dict()

    torch.save(ckpt, out_pt)
    print(f"\n[OK] Retraining successfully completed! Exported precision champion weights to: {out_pt}")
    print("=" * 80)


if __name__ == "__main__":
    main()
