import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
import json

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
    trap_threshold=0.60, margin_threshold=0.12,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
margins = (b_scores - 0.0218) - (r_scores - 0.2275)

# Also load actual trade entries from trades.csv
TRADES_CSV = ROOT / "outputs" / "acceptance_artifacts" / "milestone_3" / "iteration_8" / "trades.csv"
df_trades = pd.read_csv(TRADES_CSV)
print(f"Iteration 8 actual trades: {len(df_trades)} (Buys: {sum(df_trades['direction'] == 'BUY')}, Sells: {sum(df_trades['direction'] == 'SELL')})")

# Let's inspect which of the 1,770 bars were entered in Iteration 8
actual_entry_actions = df_agent0["action"].values
actual_agent1_actions = df_bridge[df_bridge["agent_id"] == 1]["action"].values
print(f"Bridge recorded actions: Agent0 BUY={np.sum(actual_entry_actions == 1)}, Agent1 SELL={np.sum(actual_agent1_actions == 2)}")

# Now simulate pacing filter on these 1,770 packets:
def evaluate_pacing(
    margin_thresh: float,
    prm_gate: float,
    min_spacing: int = 2,
    post_exit_cd: int = 2,
    loss_cd: int = 2,
    max_pos: int = 3,
    avg_trade_bars: int = 6, # in 1,770-packet scale (each packet ~4.8 min, so 26 min ~ 5-6 packets!)
):
    entries = []
    open_positions = []
    last_entry_bar = -999
    last_exit_bar = -999
    cooldown_until = 0
    consec_losses = 0
    
    # Track reasons for rejection
    reject_counts = {
        "prm_low": 0,
        "margin_hold": 0,
        "loss_cooldown": 0,
        "post_exit_cooldown": 0,
        "min_spacing": 0,
        "max_positions": 0,
    }
    
    for t in range(N):
        # Update existing positions exit
        still_open = []
        for pos in open_positions:
            if (t - pos["entry_packet"]) >= avg_trade_bars:
                # trade exits
                last_exit_bar = t
                # estimate win based on historical WR of ~58% for BUY, 46% for SELL
                is_win = (np.random.RandomState(t).rand() < (0.58 if pos["dir"] == 1 else 0.46))
                if not is_win:
                    consec_losses += 1
                    if consec_losses >= 2:
                        cooldown_until = t + loss_cd
                else:
                    consec_losses = 0
            else:
                still_open.append(pos)
        open_positions = still_open
        
        # Check signal
        cur_prm = prm_scores[t]
        cur_margin = margins[t]
        
        if cur_prm < prm_gate:
            reject_counts["prm_low"] += 1
            continue
            
        act_dir = 0
        if cur_margin >= margin_thresh:
            act_dir = 1
        elif cur_margin <= -margin_thresh:
            act_dir = -1
        else:
            reject_counts["margin_hold"] += 1
            continue
            
        # Pacing guards
        if t < cooldown_until:
            reject_counts["loss_cooldown"] += 1
            continue
        if (t - last_exit_bar) < post_exit_cd:
            reject_counts["post_exit_cooldown"] += 1
            continue
        if (t - last_entry_bar) < min_spacing:
            reject_counts["min_spacing"] += 1
            continue
        if len(open_positions) >= max_pos:
            reject_counts["max_positions"] += 1
            continue
            
        # Entered!
        entries.append({
            "packet": t,
            "dir": act_dir,
            "prm": cur_prm,
            "margin": cur_margin,
        })
        open_positions.append({
            "entry_packet": t,
            "dir": act_dir,
        })
        last_entry_bar = t
        
    tot = len(entries)
    buys = sum(1 for e in entries if e["dir"] == 1)
    sells = sum(1 for e in entries if e["dir"] == -1)
    trades_per_day = tot / 6.0
    return {
        "total_trades": tot,
        "trades_per_day": trades_per_day,
        "buys": buys,
        "sells": sells,
        "reject_counts": reject_counts,
    }

print("\n" + "=" * 80)
print("PACING SIMULATION ON THE 1,770 DECISION BARS")
print("=" * 80)

# 1. Baseline check (Iter 8 params: margin=0.12, prm=0.40, min_spacing=2, post_exit=2, loss_cd=2, max_pos=3)
base = evaluate_pacing(0.12, 0.40, min_spacing=2, post_exit_cd=2, loss_cd=2, max_pos=3)
print(f"BASELINE (Margin 0.12, PRM 0.40): Trades={base['total_trades']}, Tr/Day={base['trades_per_day']:.2f} (Buys: {base['buys']}, Sells: {base['sells']})")
print(f"Rejections: {base['reject_counts']}")

print("\n" + "-" * 80)
print(f"{'Margin':<8} | {'PRM':<6} | {'Space':<6} | {'PostCD':<6} | {'MaxP':<6} | {'Trades':<7} | {'Tr/Day':<8} | {'Buys':<6} | {'Sells':<6} | {'B:S':<6}")
print("-" * 80)

for m_th in [0.06, 0.07, 0.08, 0.10, 0.12]:
    for p_g in [0.32, 0.35, 0.40]:
        for sp in [1, 2]:
            for pcd in [1, 2]:
                for mp in [2, 3, 4]:
                    res = evaluate_pacing(m_th, p_g, min_spacing=sp, post_exit_cd=pcd, loss_cd=2, max_pos=mp)
                    if res["trades_per_day"] >= 25.0 or (m_th == 0.12 and p_g == 0.40):
                        tag = " *** >=40/DAY ***" if res["trades_per_day"] >= 40.0 else ""
                        print(f"{m_th:<8.2f} | {p_g:<6.2f} | {sp:<6d} | {pcd:<6d} | {mp:<6d} | {res['total_trades']:<7d} | {res['trades_per_day']:<8.2f} | {res['buys']:<6d} | {res['sells']:<6d} | {res['buys'] / max(1, res['sells']):<6.2f}{tag}")
