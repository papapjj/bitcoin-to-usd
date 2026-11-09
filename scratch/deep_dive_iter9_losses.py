import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from scripts.test_5_regimes_zero_filter import process_regime_csv
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)

TRADES_CSV = ROOT / "outputs" / "acceptance_artifacts" / "milestone_3" / "iteration_9" / "trades.csv"
WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

df_trades = pd.read_csv(TRADES_CSV)
df_m1 = pd.read_csv(WEEK3_CSV, sep="\t")
col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
col_o = "<OPEN>" if "<OPEN>" in df_m1.columns else "OPEN"
col_h = "<HIGH>" if "<HIGH>" in df_m1.columns else "HIGH"
col_l = "<LOW>" if "<LOW>" in df_m1.columns else "LOW"
col_c = "<CLOSE>" if "<CLOSE>" in df_m1.columns else "CLOSE"
col_v = "<TICKVOL>" if "<TICKVOL>" in df_m1.columns else ("VOL" if "VOL" in df_m1.columns else None)

m1_times = (df_m1[col_date].astype(str) + " " + df_m1[col_time].astype(str).str[:5]).values
time_to_m1_idx = {str(t): i for i, t in enumerate(m1_times)}

closes = df_m1[col_c].values.astype(np.float64)
highs = df_m1[col_h].values.astype(np.float64)
lows = df_m1[col_l].values.astype(np.float64)
opens = df_m1[col_o].values.astype(np.float64)
N = len(closes)

# Calculate EMAs for trend analysis
def compute_ema(series, span):
    return pd.Series(series).ewm(span=span, adjust=False).mean().values

ema20 = compute_ema(closes, 20)
ema50 = compute_ema(closes, 50)
ema200 = compute_ema(closes, 200)

# Calculate ATR
tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
tr = np.insert(tr, 0, highs[0] - lows[0])
atr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

# Forward pass on all 10,080 bars of Week 3
data = extract_features_from_csv(WEEK3_CSV)
ingress = torch.from_numpy(data["ingress"]).float()
N_feat = min(len(ingress), N)

cfg = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256,
    num_hops=4, num_debate_turns=3, prm_threshold=0.32,
    trap_threshold=0.60, margin_threshold=0.07,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(ingress[:N_feat], batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
margins = (b_scores - 0.0934) - (r_scores - 0.1662)

print("=" * 80)
print(f"WEEK 3 OVERALL MARKET CONTEXT ({N} M1 bars)")
print("=" * 80)
print(f"Start price: {closes[0]:.2f}, End price: {closes[-1]:.2f}, Min: {np.min(lows):.2f}, Max: {np.max(highs):.2f}")
print(f"Net change: {closes[-1] - closes[0]:+.2f} ({((closes[-1]/closes[0])-1)*100:+.2f}%)")

# Overall model score distributions on full week 3
print("\nModel distributions on ALL 10,080 bars:")
print(f"b_score: median={np.median(b_scores):.4f}, mean={np.mean(b_scores):.4f}")
print(f"r_score: median={np.median(r_scores):.4f}, mean={np.mean(r_scores):.4f}")
print(f"prm_score: median={np.median(prm_scores):.4f}, mean={np.mean(prm_scores):.4f}")
print(f"margin (b-0.0934)-(r-0.1662): median={np.median(margins):.4f}, mean={np.mean(margins):.4f}")
raw_diff = b_scores - r_scores
print(f"raw_diff (b - r): median={np.median(raw_diff):.4f}, mean={np.mean(raw_diff):.4f}")

# Signal counts on full week 3
for p_g in [0.32, 0.35, 0.40, 0.52]:
    mask = prm_scores >= p_g
    for th in [0.05, 0.07, 0.08, 0.09, 0.10, 0.12]:
        nb = int(np.sum(mask & (margins >= th)))
        ns = int(np.sum(mask & (margins <= -th)))
        print(f"Full Week3: PRM>={p_g:.2f}, Margin>={th:.2f} -> BUY={nb:<5d}, SELL={ns:<5d}, Ratio={nb/max(1,ns):.2f}")

print("\n" + "=" * 80)
print("ANALYZING ITERATION 9 REAL EXECUTED TRADES vs MARKET REGIME")
print("=" * 80)

# Map executed trades to M1 bar index
trade_records = []
for idx, row in df_trades.iterrows():
    t_str = str(row["logged_at"]).strip()[:16]
    m1_idx = time_to_m1_idx.get(t_str, -1)
    
    direction = row["direction"]
    profit = row["profit"]
    is_win = profit > 0
    exit_kind = row["exit_kind"]
    entry_p = row["entry_price"]
    exit_p = row["exit_price"]
    
    if m1_idx >= 0 and m1_idx < N_feat:
        b_s = b_scores[m1_idx]
        r_s = r_scores[m1_idx]
        prm = prm_scores[m1_idx]
        m = margins[m1_idx]
        cur_atr = atr14[m1_idx]
        e20 = ema20[m1_idx]
        e50 = ema50[m1_idx]
        e200 = ema200[m1_idx]
        trend_m1 = 1 if e20 > e50 else (-1 if e20 < e50 else 0)
        macro_trend = 1 if closes[m1_idx] > e200 else -1
        # Regime: Chop if |e20 - e50| < 0.25 * atr
        is_chop = abs(e20 - e50) < 0.25 * cur_atr
    else:
        b_s = r_s = prm = m = cur_atr = 0.0
        trend_m1 = macro_trend = 0
        is_chop = False
        
    trade_records.append({
        "trade_id": idx,
        "time": t_str,
        "m1_idx": m1_idx,
        "direction": direction,
        "profit": profit,
        "is_win": is_win,
        "exit_kind": exit_kind,
        "b_score": b_s,
        "r_score": r_s,
        "prm": prm,
        "margin": m,
        "atr": cur_atr,
        "trend_m1": trend_m1,
        "macro_trend": macro_trend,
        "is_chop": is_chop,
    })

df_tr = pd.DataFrame(trade_records)

print("\nTrade distribution by Market Condition at Entry:")
print("1. Chop vs Trending:")
print(pd.crosstab(df_tr["is_chop"], [df_tr["direction"], df_tr["is_win"]], margins=True))

print("\n2. Macro Trend (Price > EMA200 = Bullish 1, Bearish -1):")
print(pd.crosstab(df_tr["macro_trend"], [df_tr["direction"], df_tr["is_win"]], margins=True))

print("\n3. M1 Trend (EMA20 > EMA50 = Bullish 1, Bearish -1):")
print(pd.crosstab(df_tr["trend_m1"], [df_tr["direction"], df_tr["is_win"]], margins=True))

print("\n4. Model Conviction (margin magnitude) vs Win Rate:")
df_tr["abs_margin"] = df_tr["margin"].abs()
df_tr["margin_bucket"] = pd.cut(df_tr["abs_margin"], bins=[0.0, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 1.0])
print(df_tr.groupby("margin_bucket", observed=False)["is_win"].agg(["count", "sum", "mean"]))

print("\n5. PRM score vs Win Rate:")
df_tr["prm_bucket"] = pd.cut(df_tr["prm"], bins=[0.0, 0.32, 0.35, 0.40, 0.45, 0.50, 1.0])
print(df_tr.groupby("prm_bucket", observed=False)["is_win"].agg(["count", "sum", "mean"]))

print("\n6. Loss Analysis: Why did 119 trades lose?")
losses_df = df_tr[~df_tr["is_win"]]
print(f"Total losses: {len(losses_df)}")
print("Losses by exit kind:")
print(losses_df["exit_kind"].value_counts())
print("Losses by direction:")
print(losses_df["direction"].value_counts())
print(f"Losses in Chop: {losses_df['is_chop'].sum()} ({losses_df['is_chop'].mean()*100:.1f}%)")
print(f"Losses counter-macro-trend: {np.sum((losses_df['direction'] == 'SELL') & (losses_df['macro_trend'] == 1)) + np.sum((losses_df['direction'] == 'BUY') & (losses_df['macro_trend'] == -1))} out of {len(losses_df)}")
