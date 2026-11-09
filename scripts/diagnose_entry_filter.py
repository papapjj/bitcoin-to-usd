import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import pandas as pd
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

ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion_v3.pt"
engine = BilateralDualCoTEngine(BilateralDualCoTConfig())
engine.load_checkpoint(ckpt)
engine.eval()

lookback = 60
horizon = 35
spread = 28.98

valid_indices = list(range(lookback, N - horizon))
all_ingress = []
for t in valid_indices:
    cur_p = closes[t]
    cur_atr = max(atrs[t], 10.0)
    m1_60 = np.column_stack([opens[t-lookback:t], highs[t-lookback:t], lows[t-lookback:t], closes[t-lookback:t]])
    all_ingress.append(extract_canonical_bilateral_ingress_233(base_137[t], m1_60, cur_p, cur_atr))

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

# Best baseline: bu_th=0.42, be_th=0.42, diff_th=0.10, prm_th=0.32, sl=2.8, be_t=1.6, be_l=0.8, tp=3.5
sl_atr = 2.8
be_trig = 1.6
be_lock = 0.8
tp_atr = 3.5

trades = []
last_exit = -999
last_entry = -999

for idx, t in enumerate(valid_indices):
    if t - last_entry < 2: continue
    if t - last_exit < 1: continue

    bu = bull_scores[idx]
    be = bear_scores[idx]
    pr = prm_scores[idx]

    is_buy = (bu - be >= 0.10) and (pr >= 0.32) and (bu >= 0.42)
    is_sell = (be - bu >= 0.10) and (pr >= 0.32) and (be >= 0.42)

    if not is_buy and not is_sell: continue
    if is_buy and is_sell: continue

    cur_atr = max(atrs[t], 10.0)
    cp = closes[t]

    t_data = {
        "idx": idx, "t": t, "type": "BUY" if is_buy else "SELL",
        "bu": bu, "be": be, "pr": pr, "diff": abs(bu - be),
        "ema20_diff": base_137[t, 26], "ema50_diff": base_137[t, 27],
        "rel_pos_60": base_137[t, 13], "uw": base_137[t, 8], "lw": base_137[t, 9],
        "body": base_137[t, 7], "vol": base_137[t, 62] if base_137.shape[1] > 62 else 1.0,
    }

    if is_buy:
        entry_p = cp + spread * 0.5
        cur_sl = cp - sl_atr * cur_atr
        tp_p = cp + tp_atr * cur_atr
        be_trig_p = cp + be_trig * cur_atr
        be_lock_p = cp + be_lock * cur_atr
        be_active = False

        for k in range(t + 1, min(t + 1 + horizon, N)):
            if lows[k] <= cur_sl:
                pnl = cur_sl - entry_p
                last_exit = k
                break
            if highs[k] >= be_trig_p and not be_active:
                cur_sl = be_lock_p
                be_active = True
            if highs[k] >= tp_p:
                pnl = tp_p - entry_p
                last_exit = k
                break
        else:
            pnl = closes[min(t + horizon, N - 1)] - entry_p
            last_exit = min(t + horizon, N - 1)
    else:
        entry_p = cp - spread * 0.5
        cur_sl = cp + sl_atr * cur_atr
        tp_p = cp - tp_atr * cur_atr
        be_trig_p = cp - be_trig * cur_atr
        be_lock_p = cp - be_lock * cur_atr
        be_active = False

        for k in range(t + 1, min(t + 1 + horizon, N)):
            if highs[k] >= cur_sl:
                pnl = entry_p - cur_sl
                last_exit = k
                break
            if lows[k] <= be_trig_p and not be_active:
                cur_sl = be_lock_p
                be_active = True
            if lows[k] <= tp_p:
                pnl = entry_p - tp_p
                last_exit = k
                break
        else:
            pnl = entry_p - closes[min(t + horizon, N - 1)]
            last_exit = min(t + horizon, N - 1)

    last_entry = t
    t_data["pnl"] = pnl
    t_data["win"] = pnl > 0
    trades.append(t_data)

tdf = pd.DataFrame(trades)
print(f"Total trades: {len(tdf)} | WR: {tdf['win'].mean()*100:.1f}%")

print("\n--- BUY TRADES: WIN vs LOSS FEATURE COMPARISON ---")
buy_df = tdf[tdf["type"] == "BUY"]
print(buy_df.groupby("win")[["bu", "pr", "diff", "rel_pos_60", "ema20_diff", "uw", "lw", "body"]].mean())

print("\n--- SELL TRADES: WIN vs LOSS FEATURE COMPARISON ---")
sell_df = tdf[tdf["type"] == "SELL"]
print(sell_df.groupby("win")[["be", "pr", "diff", "rel_pos_60", "ema20_diff", "uw", "lw", "body"]].mean())
