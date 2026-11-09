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
    num_hops=4, num_debate_turns=3, prm_threshold=0.35,
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

print("=" * 80)
print("PACING SIMULATION ACROSS BRIDGE PACKETS")
print("=" * 80)

# Simulate full pacing loop
def run_pacing_simulation(prm_thresh, margin_thresh, min_spacing, loss_cd, max_pos, avg_holding_pkts=5):
    last_entry_pkt = -999
    cooldown_until = 0
    consec_losses = 0
    open_positions = []
    executed_trades = []
    
    # Simple win model calibrated from Iteration 8 and 9
    # High PRM (>= 0.40) WR ~58%, Medium PRM (0.35-0.40) WR ~55%, Low PRM (<0.35) WR ~40%
    np.random.seed(42)
    
    for t in range(N):
        # Update open positions
        still_open = []
        for p in open_positions:
            if (t - p["entry_pkt"]) >= avg_holding_pkts:
                # trade closes
                # determine outcome based on PRM
                prm_val = p["prm"]
                if prm_val >= 0.45:
                    win_prob = 0.62
                elif prm_val >= 0.40:
                    win_prob = 0.58
                elif prm_val >= 0.35:
                    win_prob = 0.55
                else:
                    win_prob = 0.40
                
                is_win = np.random.rand() < win_prob
                profit = 16.0 if is_win else -16.0
                executed_trades.append({
                    "entry_pkt": p["entry_pkt"],
                    "exit_pkt": t,
                    "dir": p["dir"],
                    "prm": prm_val,
                    "is_win": is_win,
                    "profit": profit
                })
                if not is_win:
                    consec_losses += 1
                    if consec_losses >= 2:
                        cooldown_until = t + loss_cd
                else:
                    consec_losses = 0
            else:
                still_open.append(p)
        open_positions = still_open
        
        # Check signal
        prm = prm_scores[t]
        m = margins[t]
        d = 1 if m >= margin_thresh else (-1 if m <= -margin_thresh else 0)
        
        if prm < prm_thresh or d == 0:
            continue
            
        # Pacing checks
        if t < cooldown_until:
            continue
        if (t - last_entry_pkt) < min_spacing:
            continue
        if len(open_positions) >= max_pos:
            continue
            
        # Enter
        open_positions.append({"entry_pkt": t, "dir": d, "prm": prm})
        last_entry_pkt = t
        
    df_ex = pd.DataFrame(executed_trades)
    tot = len(df_ex)
    if tot == 0:
        return {"trades": 0, "daily": 0, "wr": 0}
    wr = df_ex["is_win"].mean() * 100
    gp = df_ex[df_ex["profit"] > 0]["profit"].sum()
    gl = abs(df_ex[df_ex["profit"] <= 0]["profit"].sum())
    pf = gp / gl if gl > 0 else 0
    return {
        "trades": tot,
        "daily": tot / 6.0,
        "wr": wr,
        "pf": pf,
        "gp": gp,
        "gl": gl,
        "net": gp - gl
    }

configs = [
    {"name": "Iter 9 Baseline", "prm": 0.32, "margin": 0.07, "spacing": 2, "loss_cd": 2, "max_pos": 3},
    {"name": "Iter 8 Baseline", "prm": 0.40, "margin": 0.12, "spacing": 2, "loss_cd": 4, "max_pos": 2},
    {"name": "Candidate A (prm=0.34, cd=3)", "prm": 0.34, "margin": 0.07, "spacing": 2, "loss_cd": 3, "max_pos": 3},
    {"name": "Candidate B (prm=0.35, cd=3)", "prm": 0.35, "margin": 0.07, "spacing": 2, "loss_cd": 3, "max_pos": 3},
    {"name": "Candidate C (prm=0.35, cd=3, margin=0.08)", "prm": 0.35, "margin": 0.08, "spacing": 2, "loss_cd": 3, "max_pos": 3},
    {"name": "Candidate D (prm=0.36, cd=3)", "prm": 0.36, "margin": 0.07, "spacing": 2, "loss_cd": 3, "max_pos": 3},
    {"name": "Candidate E (prm=0.38, cd=3)", "prm": 0.38, "margin": 0.07, "spacing": 2, "loss_cd": 3, "max_pos": 3},
]

for c in configs:
    res = run_pacing_simulation(c["prm"], c["margin"], c["spacing"], c["loss_cd"], c["max_pos"])
    print(f"{c['name']:<40s}: Trades={res['trades']:<3d} ({res['daily']:.2f}/d), WR={res['wr']:.1f}%, PF={res['pf']:.2f}, Net=${res['net']:.1f}")
