import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)
from scripts.test_5_regimes_zero_filter import process_regime_csv

WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
BRIDGE_CSV = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

# Load M1 price dataset
w3_data = process_regime_csv(WEEK3_CSV)
closes = w3_data["closes"]
highs = w3_data["highs"]
lows = w3_data["lows"]
atrs = w3_data["atrs"]

# Load bridge csv
df_bridge = pd.read_csv(BRIDGE_CSV, sep="\t", low_memory=False)
df_agent0 = df_bridge[df_bridge["agent_id"] == 0].reset_index(drop=True)
N = len(df_agent0)

base_obs_list = []
times_list = df_agent0["time"].tolist()

for i in range(N):
    raw_str = str(df_agent0.iloc[i, 18])
    parts = [float(x) for x in raw_str.strip("|").split("|") if x]
    if len(parts) < 137:
        parts = parts + [0.0] * (137 - len(parts))
    base_obs_list.append(parts[:137])
base_obs_matrix = np.array(base_obs_list, dtype=np.float32)

ingress_matrix = np.array([
    extract_canonical_bilateral_ingress_233(base_obs_matrix[i], None, 0.0, 45.0)
    for i in range(N)
], dtype=np.float32)

cfg = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256,
    num_hops=4, num_debate_turns=3, prm_threshold=0.32,
    trap_threshold=0.60, margin_threshold=0.07,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
trap_probs = v.trap_probs.cpu().numpy()
margins = (b_scores - 0.0934) - (r_scores - 0.1662)

# Map bridge times to M1 bar index
df_m1 = pd.read_csv(WEEK3_CSV, sep="\t")
col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
m1_times = (df_m1[col_date].astype(str) + " " + df_m1[col_time].astype(str).str[:5]).values
time_to_m1_idx = {str(t): idx for idx, t in enumerate(m1_times)}
pkt_m1_idx = [time_to_m1_idx.get(t_str, -1) for t_str in times_list]

print("PRM score summary:")
print(pd.Series(prm_scores).describe(percentiles=[0.1, 0.25, 0.33, 0.5, 0.67, 0.75, 0.9]))

# Function to simulate single-trade outcome from packet i if entered with direction dir_val
def evaluate_entry_outcome(i, dir_val, sl_atr=2.6, tp_atr=3.0, trail_act=1.40, trail_lock=0.45, spread_usd=28.98):
    m1_start = pkt_m1_idx[i]
    if m1_start < 0 or m1_start >= len(closes) - 1:
        return None
    entry_p = closes[m1_start]
    entry_atr = max(1.0, atrs[m1_start])
    
    sl_dist = sl_atr * entry_atr
    tp_dist = tp_atr * entry_atr
    trail_act_dist = trail_act * entry_atr
    trail_lock_dist = trail_lock * entry_atr
    
    sl_price = entry_p - sl_dist if dir_val == 1 else entry_p + sl_dist
    tp_price = entry_p + tp_dist if dir_val == 1 else entry_p - tp_dist
    trail_activated = False
    cur_trail_sl = sl_price
    
    # Simulate up to 120 bars
    for b in range(m1_start + 1, min(len(closes), m1_start + 120)):
        h = highs[b]
        l = lows[b]
        c = closes[b]
        
        if dir_val == 1:
            # Trailing check
            fav = h - entry_p
            if fav >= trail_act_dist:
                trail_activated = True
                new_sl = entry_p + trail_lock_dist + (fav - trail_act_dist) * 0.5
                if new_sl > cur_trail_sl:
                    cur_trail_sl = new_sl
            # Check TP
            if h >= tp_price:
                pnl = (tp_price - entry_p) * 0.01 - spread_usd * 0.01
                return {"pnl": pnl, "kind": "TP", "bars": b - m1_start}
            # Check SL
            if l <= cur_trail_sl:
                exit_p = cur_trail_sl
                pnl = (exit_p - entry_p) * 0.01 - spread_usd * 0.01
                kind = "TRAILING_SL" if trail_activated and exit_p > entry_p else "INITIAL_SL"
                return {"pnl": pnl, "kind": kind, "bars": b - m1_start}
        else: # SELL
            fav = entry_p - l
            if fav >= trail_act_dist:
                trail_activated = True
                new_sl = entry_p - trail_lock_dist - (fav - trail_act_dist) * 0.5
                if new_sl < cur_trail_sl:
                    cur_trail_sl = new_sl
            # Check TP
            if l <= tp_price:
                pnl = (entry_p - tp_price) * 0.01 - spread_usd * 0.01
                return {"pnl": pnl, "kind": "TP", "bars": b - m1_start}
            # Check SL
            if h >= cur_trail_sl:
                exit_p = cur_trail_sl
                pnl = (entry_p - exit_p) * 0.01 - spread_usd * 0.01
                kind = "TRAILING_SL" if trail_activated and exit_p < entry_p else "INITIAL_SL"
                return {"pnl": pnl, "kind": kind, "bars": b - m1_start}
    
    # Time exit
    exit_p = closes[min(len(closes) - 1, m1_start + 120)]
    pnl = ((exit_p - entry_p) if dir_val == 1 else (entry_p - exit_p)) * 0.01 - spread_usd * 0.01
    return {"pnl": pnl, "kind": "TIME", "bars": 120}

# Collect outcomes for signals under margin_threshold = 0.07
entries = []
for i in range(N):
    m = margins[i]
    prm = prm_scores[i]
    if m >= 0.07:
        d = 1
    elif m <= -0.07:
        d = -1
    else:
        d = 0
    if d != 0:
        res = evaluate_entry_outcome(i, d)
        if res is not None:
            entries.append({
                "pkt": i,
                "dir": d,
                "prm": prm,
                "margin": m,
                "pnl": res["pnl"],
                "is_win": res["pnl"] > 0,
                "kind": res["kind"],
                "bars": res["bars"],
            })

df_ent = pd.DataFrame(entries)
print(f"\nTotal raw directional signals (|margin| >= 0.07): {len(df_ent)}")

# Group by PRM bins
df_ent["prm_bin"] = pd.qcut(df_ent["prm"], q=5, duplicates="drop")
print("\nPerformance by PRM Quantile Bins:")
for bin_val, sub in df_ent.groupby("prm_bin", observed=True):
    wr = sub["is_win"].mean() * 100
    gp = sub[sub["pnl"] > 0]["pnl"].sum()
    gl = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 0
    print(f"  PRM {bin_val}: N={len(sub):<3d}, WR={wr:.1f}%, GP=${gp:.2f}, GL=${gl:.2f}, PF={pf:.2f}, MeanPnL=${sub['pnl'].mean():.2f}")

# Sweep PRM thresholds
print("\nPerformance vs PRM Threshold (cumulative >= prm_thresh):")
for p_th in [0.25, 0.28, 0.30, 0.32, 0.34, 0.35, 0.36, 0.38, 0.40, 0.42, 0.45]:
    sub = df_ent[df_ent["prm"] >= p_th]
    if len(sub) == 0:
        continue
    wr = sub["is_win"].mean() * 100
    gp = sub[sub["pnl"] > 0]["pnl"].sum()
    gl = abs(sub[sub["pnl"] <= 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 0
    print(f"  PRM >= {p_th:.2f}: N={len(sub):<3d} ({len(sub)/6:.1f}/day), WR={wr:.2f}%, GP=${gp:.2f}, GL=${gl:.2f}, PF={pf:.2f}, Net=${sub['pnl'].sum():.2f}")
