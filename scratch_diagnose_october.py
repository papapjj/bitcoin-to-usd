import torch
import numpy as np
import pandas as pd
from pathlib import Path
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.test_3_weeks_zero_filter import process_week_csv

ROOT = Path(".")
ckpt_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(ckpt_path)
engine.eval()

w1_path = ROOT / "outputs" / "btcusd_m1_week1_20261005.csv"
w1 = process_week_csv(w1_path)
ingress = torch.from_numpy(w1["ingress"]).float()

with torch.no_grad():
    v = engine.forward_batch(ingress, batch_size=512)

acts = v.actions.cpu().numpy()
prms = v.prm_scores.cpu().numpy()
bull_scores = v.bull_scores.cpu().numpy()
bear_scores = v.bear_scores.cpu().numpy()

print(f"Total samples: {len(acts)}")
print(f"PRM stats: min={prms.min():.3f}, mean={prms.mean():.3f}, max={prms.max():.3f}")
print(f"Bull scores: mean={bull_scores.mean():.3f}, max={bull_scores.max():.3f}")
print(f"Bear scores: mean={bear_scores.mean():.3f}, max={bear_scores.max():.3f}")

# PRM distribution
for th in [0.50, 0.55, 0.58, 0.60, 0.65, 0.70, 0.75]:
    cnt = (prms >= th).sum()
    act_cnt = ((prms >= th) & (acts != 0)).sum()
    b_cnt = ((prms >= th) & (acts == 1)).sum()
    s_cnt = ((prms >= th) & (acts == -1)).sum()
    print(f"PRM >= {th:.2f}: {cnt:4d} bars | Signals: {act_cnt:4d} (BUY: {b_cnt:3d}, SELL: {s_cnt:3d})")

# Autopsy of first 10 LOSS trades
closes = w1["closes"]
highs = w1["highs"]
lows = w1["lows"]
opens = w1["opens"] if "opens" in w1 else closes
atrs = w1["atrs"]
lookback = w1["lookback"]
horizon = w1["horizon"]

print("\n--- AUTOPSY OF FIRST 10 LOSSES (PRM >= 0.58) ---")
loss_count = 0
for i in range(len(acts)):
    if prms[i] < 0.58 or acts[i] == 0:
        continue
    t = lookback + i
    cur_p = closes[t]
    cur_atr = atrs[t]
    act = acts[i]
    
    f_h = highs[t+1 : t+1+horizon]
    f_l = lows[t+1 : t+1+horizon]
    
    if act == 1: # BUY
        sl_p = cur_p - 1.5 * cur_atr
        tp_p = cur_p + 2.25 * cur_atr
        hit_sl = False
        bars_to_exit = 0
        worst_adv = 0.0
        for k in range(len(f_h)):
            adv = (cur_p - f_l[k]) / cur_atr
            if adv > worst_adv: worst_adv = adv
            if f_l[k] <= sl_p:
                hit_sl = True
                bars_to_exit = k + 1
                break
            if f_h[k] >= tp_p:
                break
        if hit_sl:
            loss_count += 1
            ret1 = (cur_p - closes[t-1]) / cur_atr
            print(f"Loss #{loss_count:2d} [BUY] at bar {t} | P={cur_p:.1f}, ATR={cur_atr:.1f} | MAE={worst_adv:.2f} ATR in {bars_to_exit} bars | Prev Bar Ret={ret1:+.2f} ATR | PRM={prms[i]:.2f}, Bull={bull_scores[i]:.2f}, Bear={bear_scores[i]:.2f}")
            if loss_count >= 10:
                break
    elif act == -1: # SELL
        sl_p = cur_p + 1.5 * cur_atr
        tp_p = cur_p - 2.25 * cur_atr
        hit_sl = False
        bars_to_exit = 0
        worst_adv = 0.0
        for k in range(len(f_h)):
            adv = (f_h[k] - cur_p) / cur_atr
            if adv > worst_adv: worst_adv = adv
            if f_h[k] >= sl_p:
                hit_sl = True
                bars_to_exit = k + 1
                break
            if f_l[k] <= tp_p:
                break
        if hit_sl:
            loss_count += 1
            ret1 = (cur_p - closes[t-1]) / cur_atr
            print(f"Loss #{loss_count:2d} [SELL] at bar {t} | P={cur_p:.1f}, ATR={cur_atr:.1f} | MAE={worst_adv:.2f} ATR in {bars_to_exit} bars | Prev Bar Ret={ret1:+.2f} ATR | PRM={prms[i]:.2f}, Bull={bull_scores[i]:.2f}, Bear={bear_scores[i]:.2f}")
            if loss_count >= 10:
                break
