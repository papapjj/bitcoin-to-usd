import torch
import numpy as np
import pandas as pd
from pathlib import Path
from research_reasoning_world_class.src.bilateral_dual_cot import BilateralDualCoTEngine, BilateralDualCoTConfig
from scripts.test_3_weeks_zero_filter import process_week_csv

ROOT = Path(".")
ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_multi_regime_anti_mae_champion.pt"
cfg = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, prm_threshold=0.35)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(ckpt)
engine.eval()

w3_data = process_week_csv(ROOT / "outputs" / "btcusd_m1_week3_20261019.csv")
ingress = torch.from_numpy(w3_data["ingress"]).float()
with torch.no_grad():
    v = engine.forward_batch(ingress, batch_size=512)

acts = v.actions.cpu().numpy()
prms = v.prm_scores.cpu().numpy()

closes = w3_data["closes"]
highs = w3_data["highs"]
lows = w3_data["lows"]
atrs = w3_data["atrs"]
lookback = w3_data["lookback"]
horizon = w3_data["horizon"]

df_raw = pd.read_csv("outputs/btcusd_m1_week3_20261019.csv", sep="\t")
dates = df_raw["<DATE>"].values
times = df_raw["<TIME>"].values

print("=== WEEK 3 LOSS TIMELINE AUDIT (PRM >= 0.60) ===")
loss_dates = []
win_dates = []

for i in range(len(acts)):
    if prms[i] < 0.60 or acts[i] == 0:
        continue
    act = acts[i]
    t = lookback + i
    cur_p = closes[t]
    cur_atr = atrs[t]
    f_h = highs[t+1 : t+1+horizon]
    f_l = lows[t+1 : t+1+horizon]
    
    dt_str = f"{dates[t]} {times[t]}"
    
    if act == 1:
        sl_p = cur_p - 1.5 * cur_atr
        tp_p = cur_p + 2.25 * cur_atr
        win = False
        for k in range(len(f_h)):
            if f_l[k] <= sl_p: break
            if f_h[k] >= tp_p:
                win = True
                break
        if win: win_dates.append((dt_str, "BUY", cur_p, cur_atr))
        else: loss_dates.append((dt_str, "BUY", cur_p, cur_atr))
    elif act == -1:
        sl_p = cur_p + 1.5 * cur_atr
        tp_p = cur_p - 2.25 * cur_atr
        win = False
        for k in range(len(f_h)):
            if f_h[k] >= sl_p: break
            if f_l[k] <= tp_p:
                win = True
                break
        if win: win_dates.append((dt_str, "SELL", cur_p, cur_atr))
        else: loss_dates.append((dt_str, "SELL", cur_p, cur_atr))

print(f"Total signals: {len(win_dates) + len(loss_dates)} | Wins: {len(win_dates)} | Losses: {len(loss_dates)}")

loss_df = pd.DataFrame(loss_dates, columns=["datetime", "dir", "price", "atr"])
loss_df["date"] = loss_df["datetime"].str[:10]
win_df = pd.DataFrame(win_dates, columns=["datetime", "dir", "price", "atr"])
win_df["date"] = win_df["datetime"].str[:10]

print("\nLosses by day in Week 3:")
print(loss_df.groupby(["date", "dir"]).size())

print("\nWins by day in Week 3:")
print(win_df.groupby(["date", "dir"]).size())
