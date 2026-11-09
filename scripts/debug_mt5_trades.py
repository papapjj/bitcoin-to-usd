import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from research_reasoning_world_class.src.bilateral_dual_cot import (
    extract_canonical_bilateral_ingress_233,
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
)
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)

csv_path = ROOT / "outputs" / "btcusd_m1_7day_with_warmup.csv"
df = pd.read_csv(csv_path, sep="\t")
df_renamed = df.rename(columns={
    "<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close",
    "<TICKVOL>": "Volume", "<VOL>": "RealVolume", "<SPREAD>": "Spread"
})
df_renamed["Datetime"] = df_renamed["<DATE>"] + " " + df_renamed["<TIME>"]

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

closes = feat_df["Close"].values.astype(np.float64)
highs = feat_df["High"].values.astype(np.float64)
lows = feat_df["Low"].values.astype(np.float64)
opens = feat_df["Open"].values.astype(np.float64)
atrs = feat_df["atr"].values.astype(np.float64)
datetimes = pd.to_datetime(feat_df["Datetime"]).dt.strftime("%Y.%m.%d %H:%M").values

lookback = 60
horizon = 45
valid_indices = list(range(lookback, N - horizon))

engine = BilateralDualCoTEngine(BilateralDualCoTConfig())
ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
engine.load_checkpoint(ckpt)
engine.eval()

all_ingress = []
for t in valid_indices:
    cur_p = closes[t]
    cur_atr = max(atrs[t], 10.0)
    m1_60 = np.column_stack([opens[t-lookback:t], highs[t-lookback:t], lows[t-lookback:t], closes[t-lookback:t]])
    ing = extract_canonical_bilateral_ingress_233(base_137[t], m1_60, cur_p, cur_atr)
    all_ingress.append(ing)

all_ingress_tensor = torch.from_numpy(np.array(all_ingress, dtype=np.float32))

bull_scores, bear_scores, prm_scores = [], [], []
batch_size = 512
with torch.no_grad():
    for i in range(0, len(all_ingress_tensor), batch_size):
        b_chunk = all_ingress_tensor[i:i+batch_size]
        out = engine.forward_batch(b_chunk)
        bull_scores.extend(out.bull_scores.cpu().numpy().tolist())
        bear_scores.extend(out.bear_scores.cpu().numpy().tolist())
        prm_scores.extend(out.prm_scores.cpu().numpy().tolist())

bull_scores = np.array(bull_scores)
bear_scores = np.array(bear_scores)
prm_scores = np.array(prm_scores)

csv_path = r"c:\Users\Admin\Desktop\new-chat\.mt5_runners\visual_7day_champion_9296_20261101_044403_974184\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv"
df_mt5 = pd.read_csv(csv_path, sep="\t")
fills = df_mt5[df_mt5["filled"] == 1].reset_index(drop=True)
closes_mt5 = df_mt5[df_mt5["position_final_close"] == 1].reset_index(drop=True)

dt_minute_to_idx = {datetimes[t]: (idx, t) for idx, t in enumerate(valid_indices)}

matches = []
for i in range(min(len(fills), len(closes_mt5))):
    f_t = str(fills.iloc[i]["logged_at"])
    f_minute = f_t[:16]
    profit = closes_mt5.iloc[i]["profit"]
    close_r = closes_mt5.iloc[i]["close_reason"]
    exit_k = closes_mt5.iloc[i]["exit_kind"]
    act = fills.iloc[i]["action_name"]
    
    if f_minute in dt_minute_to_idx:
        idx, t = dt_minute_to_idx[f_minute]
        bu = bull_scores[idx]
        be = bear_scores[idx]
        pr = prm_scores[idx]
        matches.append({
            "mt5_fill": f_t,
            "act": act,
            "bu": bu,
            "be": be,
            "diff": bu - be,
            "prm": pr,
            "profit": profit,
            "exit_k": exit_k
        })

df_matches = pd.DataFrame(matches)
print(f"Matched {len(df_matches)} of {len(fills)} trades.")
print(df_matches.head(15).to_string())

v_buy = (df_matches["act"] == "MARKET_BUY") & (df_matches["diff"] >= 0.12) & (df_matches["prm"] >= 0.35) & (df_matches["bu"] >= 0.40)
v_sell = (df_matches["act"] == "MARKET_SELL") & (-df_matches["diff"] >= 0.084) & (df_matches["prm"] >= 0.35) & (df_matches["be"] >= 0.35)
df_matches["valid_benchmark"] = v_buy | v_sell

print("\nBenchmark validation breakdown:")
print(df_matches["valid_benchmark"].value_counts())

print("\n--- TRADES MATCHING BENCHMARK SPEC ---")
bt = df_matches[df_matches["valid_benchmark"]]
if len(bt) > 0:
    w = int((bt["profit"] > 0).sum())
    l = int((bt["profit"] < 0).sum())
    print(f"Count: {len(bt)}, Wins: {w}, Losses: {l}, WR: {w/len(bt):.1%}, Profit: {bt['profit'].sum():.2f}")
    print(bt["exit_k"].value_counts())

print("\n--- NOISE TRADES (ENTRIES THAT VIOLATED BENCHMARK SPEC) ---")
nt = df_matches[~df_matches["valid_benchmark"]]
if len(nt) > 0:
    w = int((nt["profit"] > 0).sum())
    l = int((nt['profit'] < 0).sum())
    print(f"Count: {len(nt)}, Wins: {w}, Losses: {l}, WR: {w/len(nt):.1%}, Profit: {nt['profit'].sum():.2f}")
    print(nt["exit_k"].value_counts())
