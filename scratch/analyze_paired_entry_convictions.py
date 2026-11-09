import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

EXEC_CSV = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_execution.csv"
WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

import torch
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)

# 1. Parse execution ledger for filled entry deals and closed deals
df_exec = pd.read_csv(EXEC_CSV, sep="\t", low_memory=False)
print("Total rows in execution ledger:", len(df_exec))

# Positions tracking
# A position has an entry deal (deal_entry == 1 or transaction_type == 'DEAL_ADD' and deal_ticket != 0 and status == 'FILLED' or placed)
# Let's inspect rows with trans_type DEAL_ADD
deals = df_exec[df_exec["transaction_type"] == "DEAL_ADD"].copy()
print("Total DEAL_ADD rows:", len(deals))
print("Deal entry values:", deals["deal_entry"].value_counts())
print("Status values for DEAL_ADD:", deals["status"].value_counts())

# Separate entry deals (deal_entry == 0 or 1 in MT5: ENTRY_IN=0, ENTRY_OUT=1, ENTRY_INOUT=2)
# In MT5: 0 is ENTRY_IN (open position), 1 is ENTRY_OUT (close position)
entry_deals = deals[deals["deal_entry"] == 0].copy()
exit_deals = deals[deals["deal_entry"] == 1].copy()

print(f"Entry deals (ENTRY_IN): {len(entry_deals)}")
print(f"Exit deals (ENTRY_OUT): {len(exit_deals)}")

# Match by position_ticket
merged = pd.merge(
    entry_deals[["position_ticket", "logged_at", "action_name", "price", "volume", "initial_sl_price", "initial_tp_price"]],
    exit_deals[["position_ticket", "logged_at", "action_name", "price", "profit", "exit_kind", "age_bars"]],
    on="position_ticket",
    suffixes=("_open", "_close")
)
print(f"Successfully paired {len(merged)} complete round-trip positions!")

# Direction in entry deal:
# action_name_open: MARKET_BUY -> direction BUY; MARKET_SELL -> direction SELL
print("\nAction names for ENTRY_IN:")
print(merged["action_name_open"].value_counts())

print("\nAction names for ENTRY_OUT:")
print(merged["action_name_close"].value_counts())

# Now, map entry time to Week 3 M1 bars and evaluate model outputs AT ENTRY!
df_m1 = pd.read_csv(WEEK3_CSV, sep="\t")
col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
m1_times = (df_m1[col_date].astype(str) + " " + df_m1[col_time].astype(str).str[:5]).values
time_to_m1_idx = {str(t): i for i, t in enumerate(m1_times)}

data = extract_features_from_csv(WEEK3_CSV)
ingress = torch.from_numpy(data["ingress"]).float()

cfg = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256,
    num_hops=4, num_debate_turns=3, prm_threshold=0.32,
    trap_threshold=0.60, margin_threshold=0.07,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(ingress, batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
margins = (b_scores - 0.0934) - (r_scores - 0.1662)

# Analyze AT ENTRY TIME
entry_stats = []
for idx, row in merged.iterrows():
    open_time = str(row["logged_at_open"]).strip()[:16]
    m1_idx = time_to_m1_idx.get(open_time, -1)
    
    act_open = row["action_name_open"]
    dir_str = "BUY" if "BUY" in act_open else ("SELL" if "SELL" in act_open else "UNKNOWN")
    profit = row["profit"]
    is_win = profit > 0
    
    if m1_idx >= 0 and m1_idx < len(b_scores):
        bs = b_scores[m1_idx]
        rs = r_scores[m1_idx]
        prm = prm_scores[m1_idx]
        m = margins[m1_idx]
    else:
        bs = rs = prm = m = 0.0
        
    entry_stats.append({
        "position_ticket": row["position_ticket"],
        "open_time": open_time,
        "close_time": row["logged_at_close"],
        "direction": dir_str,
        "profit": profit,
        "is_win": is_win,
        "exit_kind": row["exit_kind"],
        "b_score": bs,
        "r_score": rs,
        "prm": prm,
        "margin": m,
        "age_bars": row["age_bars"],
    })

df_paired = pd.DataFrame(entry_stats)
print("\n" + "=" * 80)
print("REAL ENTRY-TIME CONVICTION STATISTICS (At exact moment position was OPENED)")
print("=" * 80)
for d in ["BUY", "SELL"]:
    sub = df_paired[df_paired["direction"] == d]
    print(f"\n{d} Positions (N={len(sub)}):")
    print(f"  Win rate: {sub['is_win'].mean()*100:.2f}% ({sub['is_win'].sum()} W / {len(sub)-sub['is_win'].sum()} L)")
    print(f"  b_score: mean={sub['b_score'].mean():.4f}, median={sub['b_score'].median():.4f}")
    print(f"  r_score: mean={sub['r_score'].mean():.4f}, median={sub['r_score'].median():.4f}")
    print(f"  margin:  mean={sub['margin'].mean():.4f}, median={sub['margin'].median():.4f}")
    print(f"  prm:     mean={sub['prm'].mean():.4f}, median={sub['prm'].median():.4f}")
    print(f"  age_bars: mean={sub['age_bars'].mean():.1f}, median={sub['age_bars'].median():.1f}")

print("\nDirectional Margin Compliance at Entry:")
print(f"BUY positions with margin >= +0.07 at entry: {(df_paired[df_paired['direction']=='BUY']['margin'] >= 0.07).sum()} / {len(df_paired[df_paired['direction']=='BUY'])}")
print(f"SELL positions with margin <= -0.07 at entry: {(df_paired[df_paired['direction']=='SELL']['margin'] <= -0.07).sum()} / {len(df_paired[df_paired['direction']=='SELL'])}")

print("\nMargin distribution for BUY at entry:")
print(df_paired[df_paired['direction']=='BUY']['margin'].describe())

print("\nMargin distribution for SELL at entry:")
print(df_paired[df_paired['direction']=='SELL']['margin'].describe())
