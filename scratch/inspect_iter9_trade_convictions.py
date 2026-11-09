import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)

TRADES_CSV = ROOT / "outputs" / "acceptance_artifacts" / "milestone_3" / "iteration_9" / "trades.csv"
WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

df_trades = pd.read_csv(TRADES_CSV)
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

print("=" * 80)
print("INSPECTING THE 244 EXECUTED TRADES IN ITERATION 9")
print("=" * 80)

trade_data = []
for idx, row in df_trades.iterrows():
    t_str = str(row["logged_at"]).strip()[:16]
    m1_idx = time_to_m1_idx.get(t_str, -1)
    if m1_idx >= 0 and m1_idx < len(b_scores):
        trade_data.append({
            "trade_id": idx,
            "logged_at": t_str,
            "direction": row["direction"],
            "profit": row["profit"],
            "is_win": row["profit"] > 0,
            "exit_kind": row["exit_kind"],
            "b_score": b_scores[m1_idx],
            "r_score": r_scores[m1_idx],
            "prm": prm_scores[m1_idx],
            "margin": margins[m1_idx],
            "b_med_centered": b_scores[m1_idx] - 0.0934,
            "r_med_centered": r_scores[m1_idx] - 0.1662,
        })

df_t = pd.DataFrame(trade_data)
print(f"Matched {len(df_t)} out of {len(df_trades)} trades to M1 bars.\n")

print("--- Conviction Metrics by Direction ---")
for d in ["BUY", "SELL"]:
    sub = df_t[df_t["direction"] == d]
    print(f"\n{d} Trades (N={len(sub)}):")
    print(f"  b_score: mean={sub['b_score'].mean():.4f}, median={sub['b_score'].median():.4f}, min={sub['b_score'].min():.4f}, max={sub['b_score'].max():.4f}")
    print(f"  r_score: mean={sub['r_score'].mean():.4f}, median={sub['r_score'].median():.4f}, min={sub['r_score'].min():.4f}, max={sub['r_score'].max():.4f}")
    print(f"  margin:  mean={sub['margin'].mean():.4f}, median={sub['margin'].median():.4f}, min={sub['margin'].min():.4f}, max={sub['margin'].max():.4f}")
    print(f"  prm:     mean={sub['prm'].mean():.4f}, median={sub['prm'].median():.4f}, min={sub['prm'].min():.4f}, max={sub['prm'].max():.4f}")

# Look at distribution of margin across all 201 SELL trades
print("\nSELL Trades margin distribution (should be <= -0.07):")
print(f"SELL with margin <= -0.07: {(df_t[df_t['direction'] == 'SELL']['margin'] <= -0.07).sum()} / {len(df_t[df_t['direction'] == 'SELL'])}")
print(f"SELL with margin > -0.07 (ANOMALOUS): {(df_t[df_t['direction'] == 'SELL']['margin'] > -0.07).sum()}")

# Look at distribution of margin across all 43 BUY trades
print("\nBUY Trades margin distribution (should be >= +0.07):")
print(f"BUY with margin >= +0.07: {(df_t[df_t['direction'] == 'BUY']['margin'] >= 0.07).sum()} / {len(df_t[df_t['direction'] == 'BUY'])}")
print(f"BUY with margin < +0.07 (ANOMALOUS): {(df_t[df_t['direction'] == 'BUY']['margin'] < 0.07).sum()}")

if (df_t[df_t['direction'] == 'SELL']['margin'] > -0.07).sum() > 0:
    print("\nSample anomalous SELL trades (margin > -0.07):")
    print(df_t[(df_t['direction'] == 'SELL') & (df_t['margin'] > -0.07)][['logged_at', 'direction', 'margin', 'prm', 'b_score', 'r_score']].head(10))

if (df_t[df_t['direction'] == 'BUY']['margin'] < 0.07).sum() > 0:
    print("\nSample anomalous BUY trades (margin < 0.07):")
    print(df_t[(df_t['direction'] == 'BUY') & (df_t['margin'] < 0.07)][['logged_at', 'direction', 'margin', 'prm', 'b_score', 'r_score']].head(10))
