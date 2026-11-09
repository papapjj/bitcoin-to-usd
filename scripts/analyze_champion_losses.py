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
if not ckpt.exists():
    ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
print(f"Loading {ckpt.name}...")
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

# Analyze trades with diff=0.15, prm=0.35, b_conf=0.25, sl=3.0, be_t=2.0, be_l=1.0, tp=4.0
sl_atr = 3.0
be_trig = 2.0
be_lock = 1.0
tp_atr = 4.0

trades = []
last_exit = -999
last_entry = -999

for idx, t in enumerate(valid_indices):
    if t - last_entry < 2: continue
    if t - last_exit < 1: continue

    bu = bull_scores[idx]
    be = bear_scores[idx]
    pr = prm_scores[idx]

    is_buy = (bu - be >= 0.15) and (pr >= 0.35) and (bu >= 0.40)
    is_sell = (be - bu >= 0.15 * 0.7) and (pr >= 0.35) and (be >= 0.25)

    if not is_buy and not is_sell: continue
    if is_buy and is_sell: continue

    cur_atr = max(atrs[t], 10.0)
    cp = closes[t]

    trade_info = {
        "t": t, "type": "BUY" if is_buy else "SELL", "cp": cp, "atr": cur_atr,
        "bu": bu, "be": be, "pr": pr
    }

    if is_buy:
        entry_p = cp + spread * 0.5
        cur_sl = cp - sl_atr * cur_atr
        tp_p = cp + tp_atr * cur_atr
        be_trig_p = cp + be_trig * cur_atr
        be_lock_p = cp + be_lock * cur_atr
        be_active = False

        max_mae = 0.0
        max_mfe = 0.0
        exit_reason = "HORIZON"

        for k in range(t + 1, min(t + 1 + horizon, N)):
            adverse = (entry_p - lows[k]) / cur_atr
            favorable = (highs[k] - entry_p) / cur_atr
            if adverse > max_mae: max_mae = adverse
            if favorable > max_mfe: max_mfe = favorable

            if lows[k] <= cur_sl:
                pnl = cur_sl - entry_p
                last_exit = k
                exit_reason = "BE_LOCK" if be_active else "INITIAL_SL"
                break
            if highs[k] >= be_trig_p and not be_active:
                cur_sl = be_lock_p
                be_active = True
            if highs[k] >= tp_p:
                pnl = tp_p - entry_p
                last_exit = k
                exit_reason = "TP"
                break
        else:
            pnl = closes[min(t + horizon, N - 1)] - entry_p
            last_exit = min(t + horizon, N - 1)
            exit_reason = "HORIZON"
    else:
        entry_p = cp - spread * 0.5
        cur_sl = cp + sl_atr * cur_atr
        tp_p = cp - tp_atr * cur_atr
        be_trig_p = cp - be_trig * cur_atr
        be_lock_p = cp - be_lock * cur_atr
        be_active = False

        max_mae = 0.0
        max_mfe = 0.0
        exit_reason = "HORIZON"

        for k in range(t + 1, min(t + 1 + horizon, N)):
            adverse = (highs[k] - entry_p) / cur_atr
            favorable = (entry_p - lows[k]) / cur_atr
            if adverse > max_mae: max_mae = adverse
            if favorable > max_mfe: max_mfe = favorable

            if highs[k] >= cur_sl:
                pnl = entry_p - cur_sl
                last_exit = k
                exit_reason = "BE_LOCK" if be_active else "INITIAL_SL"
                break
            if lows[k] <= be_trig_p and not be_active:
                cur_sl = be_lock_p
                be_active = True
            if lows[k] <= tp_p:
                pnl = entry_p - tp_p
                last_exit = k
                exit_reason = "TP"
                break
        else:
            pnl = entry_p - closes[min(t + horizon, N - 1)]
            last_exit = min(t + horizon, N - 1)
            exit_reason = "HORIZON"

    last_entry = t
    trade_info["pnl"] = pnl
    trade_info["win"] = pnl > 0
    trade_info["exit_reason"] = exit_reason
    trade_info["mae_atr"] = max_mae
    trade_info["mfe_atr"] = max_mfe
    trade_info["duration"] = last_exit - t
    trades.append(trade_info)

tdf = pd.DataFrame(trades)
print(f"Total trades: {len(tdf)} (Win Rate: {tdf['win'].mean()*100:.1f}%)")
print("\nExit Reason Breakdown:")
print(tdf.groupby(["type", "exit_reason"])["win"].agg(["count", "mean", lambda x: (x > 0).sum()]))

print("\nLosses Breakdown:")
losses = tdf[~tdf["win"]]
print(f"Total losses: {len(losses)}")
print(losses["exit_reason"].value_counts())
print(f"Losses mean MAE (ATR): {losses['mae_atr'].mean():.2f}")
print(f"Losses mean duration: {losses['duration'].mean():.1f} bars")

print("\nWins Breakdown:")
wins = tdf[tdf["win"]]
print(f"Total wins: {len(wins)}")
print(wins["exit_reason"].value_counts())
print(f"Wins mean MFE (ATR): {wins['mfe_atr'].mean():.2f}")
print(f"Wins mean MAE (ATR): {wins['mae_atr'].mean():.2f}")
