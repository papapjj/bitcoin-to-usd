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

TRADES_CSV = ROOT / "outputs" / "acceptance_artifacts" / "milestone_3" / "iteration_9" / "trades.csv"
BRIDGE_CSV = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"
WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"

# 1. Analyze trades.csv from Iteration 9
df_trades = pd.read_csv(TRADES_CSV)
print("=" * 80)
print(f"ITERATION 9 TRADES ANALYSIS (Total closed trades: {len(df_trades)})")
print("=" * 80)
print("Direction breakdown:")
print(df_trades["direction"].value_counts())
print("\nWin/Loss by direction:")
df_trades["is_win"] = df_trades["profit"] > 0
print(pd.crosstab(df_trades["direction"], df_trades["is_win"], margins=True))

print("\nExit kind by direction:")
print(pd.crosstab(df_trades["direction"], df_trades["exit_kind"], margins=True))

print("\nProfit stats by direction:")
for d in ["BUY", "SELL"]:
    sub = df_trades[df_trades["direction"] == d]
    wins = sub[sub["is_win"]]
    losses = sub[~sub["is_win"]]
    gp = wins["profit"].sum()
    gl = abs(losses["profit"].sum())
    pf = gp / gl if gl > 0 else 0.0
    wr = len(wins) / len(sub) * 100
    avg_win = wins["profit"].mean() if len(wins) else 0
    avg_loss = abs(losses["profit"].mean()) if len(losses) else 0
    print(f"{d}: N={len(sub)}, WR={wr:.2f}%, GP=${gp:.2f}, GL=${gl:.2f}, PF={pf:.2f}, AvgWin=${avg_win:.2f}, AvgLoss=${avg_loss:.2f}, Payoff={avg_win/avg_loss if avg_loss else 0:.2f}")

# 2. Analyze the 1,770 decision packets from bridge CSV
print("\n" + "=" * 80)
print("ANALYZING 1,770 MT5 DECISION PACKETS & MODEL OUTPUT DISTRIBUTIONS")
print("=" * 80)
df_bridge = pd.read_csv(BRIDGE_CSV, sep="\t", low_memory=False)
df_agent0 = df_bridge[df_bridge["agent_id"] == 0].reset_index(drop=True)
N = len(df_agent0)
print(f"Loaded {N} decision bars from bridge.")

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

b_med = float(np.median(b_scores))
r_med = float(np.median(r_scores))
prm_med = float(np.median(prm_scores))

print(f"Empirical medians across 1,770 packets:")
print(f"  b_score median: {b_med:.4f} (mean: {np.mean(b_scores):.4f}, std: {np.std(b_scores):.4f})")
print(f"  r_score median: {r_med:.4f} (mean: {np.mean(r_scores):.4f}, std: {np.std(r_scores):.4f})")
print(f"  prm_score median: {prm_med:.4f} (mean: {np.mean(prm_scores):.4f}, std: {np.std(prm_scores):.4f})")

print("\nTrap probs percentiles [0, 50, 90, 99, 100]:")
for col_idx, col_name in enumerate(["No Trap", "Bull Trap", "Bear Trap"]):
    vals = trap_probs[:, col_idx]
    p = np.percentile(vals, [0, 50, 90, 95, 99, 100])
    print(f"  {col_name}: min={p[0]:.4f}, 50%={p[1]:.4f}, 90%={p[2]:.4f}, 95%={p[3]:.4f}, 99%={p[4]:.4f}, max={p[5]:.4f}")

# Check margins under different centerings:
margin_current = (b_scores - 0.0934) - (r_scores - 0.1662)
margin_empirical_pkt = (b_scores - b_med) - (r_scores - r_med)
margin_old = (b_scores - 0.0218) - (r_scores - 0.2275)

print("\n--- Margin centering comparison ---")
print(f"Using (0.0934, 0.1662): mean={np.mean(margin_current):.4f}, median={np.median(margin_current):.4f}, std={np.std(margin_current):.4f}")
print(f"Using empirical pkt medians ({b_med:.4f}, {r_med:.4f}): mean={np.mean(margin_empirical_pkt):.4f}, median={np.median(margin_empirical_pkt):.4f}")
print(f"Using old (0.0218, 0.2275): mean={np.mean(margin_old):.4f}, median={np.median(margin_old):.4f}")

# Signal counts across 1,770 packets for margin_current vs margin_empirical_pkt
for label, m_arr in [("Current (0.0934, 0.1662)", margin_current), ("Empirical Pkt Medians", margin_empirical_pkt), ("Old (0.0218, 0.2275)", margin_old)]:
    print(f"\nSignal counts with {label}:")
    for prm_gate in [0.35, 0.40, 0.52]:
        mask = prm_scores >= prm_gate
        for th in [0.05, 0.07, 0.08, 0.09, 0.10, 0.12]:
            nb = int(np.sum(mask & (m_arr >= th)))
            ns = int(np.sum(mask & (m_arr <= -th)))
            print(f"  PRM>={prm_gate:.2f}, Margin>={th:.2f} -> BUY={nb:<4d}, SELL={ns:<4d}, Tot={nb+ns:<4d}, Ratio={nb/max(1,ns):.2f}")
