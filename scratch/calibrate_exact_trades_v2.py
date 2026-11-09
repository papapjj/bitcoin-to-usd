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

BRIDGE_CSV = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

df_bridge = pd.read_csv(BRIDGE_CSV, sep="\t", low_memory=False)
df_agent0 = df_bridge[df_bridge["agent_id"] == 0].reset_index(drop=True)
N = len(df_agent0) # 1022

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
margins = (b_scores - 0.0934) - (r_scores - 0.1662)

SCALE = 1481.0 / N # 1.4491

def simulate(prm_thresh, margin_thresh, min_spacing, loss_cd, max_pos, avg_holding=5):
    last_entry_pkt = -999
    cooldown_until = 0
    consec_losses = 0
    open_positions = []
    executed = []
    
    np.random.seed(42)
    for t in range(N):
        still_open = []
        for p in open_positions:
            if (t - p["entry_pkt"]) >= avg_holding:
                prm_val = p["prm"]
                win_prob = 0.58 if prm_val >= 0.38 else (0.53 if prm_val >= 0.34 else 0.45)
                is_win = np.random.rand() < win_prob
                executed.append({"is_win": is_win, "prm": prm_val})
                if not is_win:
                    consec_losses += 1
                    if consec_losses >= 2:
                        cooldown_until = t + loss_cd
                else:
                    consec_losses = 0
            else:
                still_open.append(p)
        open_positions = still_open
        
        prm = prm_scores[t]
        m = margins[t]
        d = 1 if m >= margin_thresh else (-1 if m <= -margin_thresh else 0)
        if prm < prm_thresh or d == 0:
            continue
        if t < cooldown_until:
            continue
        if (t - last_entry_pkt) < min_spacing:
            continue
        if len(open_positions) >= max_pos:
            continue
        open_positions.append({"entry_pkt": t, "dir": d, "prm": prm})
        last_entry_pkt = t
        
    scaled_trades = len(executed) * SCALE
    return scaled_trades, np.mean([x["is_win"] for x in executed]) if executed else 0

results = []
for prm in [0.32, 0.33, 0.34, 0.35, 0.36, 0.38, 0.40]:
    for loss_cd in [2, 3, 4]:
        for min_sp in [2, 3]:
            for max_pos in [2, 3]:
                tr, wr = simulate(prm, 0.07, min_sp, loss_cd, max_pos)
                t_day = tr / 6.0
                results.append({
                    "prm": prm,
                    "loss_cd": loss_cd,
                    "min_sp": min_sp,
                    "max_pos": max_pos,
                    "trades": round(tr, 1),
                    "trades_per_day": round(t_day, 2),
                    "wr_est": round(wr * 100, 2),
                    "meets_40": t_day >= 40.0
                })

df_res = pd.DataFrame(results)
print(f"Total configs evaluated: {len(df_res)}")
print(f"Configs meeting >= 40 trades/day: {sum(df_res['meets_40'])} / {len(df_res)}")
print("\nTop 15 configs meeting >= 40 trades/day sorted by estimated Win Rate:")
print(df_res[df_res["meets_40"]].sort_values(by=["wr_est", "trades_per_day"], ascending=[False, False]).head(15).to_string())
