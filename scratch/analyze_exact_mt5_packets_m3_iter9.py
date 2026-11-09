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

print("=" * 80)
print("EXTRACTING EXACT 1,770 MT5 STRATEGY TESTER DECISION BARS")
print("=" * 80)

df_bridge = pd.read_csv(BRIDGE_CSV, sep="\t", low_memory=False)
df_agent0 = df_bridge[df_bridge["agent_id"] == 0].reset_index(drop=True)
N = len(df_agent0)
print(f"Loaded {N} unique decision bars from bridge CSV.")

# Parse inputs from VectorToPipe string (pipe-separated floats)
col_inputs = df_agent0.columns[18]  # inputs pipe column
print(f"Parsing inputs from column: {col_inputs}")

base_obs_list = []
times_list = df_agent0["time"].tolist()
balances_list = df_agent0["balance"].astype(float).tolist()

for i in range(N):
    raw_str = str(df_agent0.iloc[i, 18])
    parts = [float(x) for x in raw_str.strip("|").split("|") if x]
    if len(parts) < 137:
        parts = parts + [0.0] * (137 - len(parts))
    base_obs_list.append(parts[:137])

base_obs_matrix = np.array(base_obs_list, dtype=np.float32)
print(f"Extracted base_obs matrix shape: {base_obs_matrix.shape}")

# Build 233D canonical ingress vector for each bar
ingress_list = []
for i in range(N):
    # In live policy: extract_canonical_bilateral_ingress_233
    ing = extract_canonical_bilateral_ingress_233(
        base_obs_137=base_obs_matrix[i],
        m1_bars_60=None,
        cur_p=0.0,
        cur_atr=45.0,
    )
    ingress_list.append(ing)

ingress_matrix = np.array(ingress_list, dtype=np.float32)
print(f"Ingress matrix shape: {ingress_matrix.shape}")

cfg = BilateralDualCoTConfig(
    input_dim=233,
    latent_dim=768,
    hidden_dim=256,
    num_hops=4,
    num_debate_turns=3,
    prm_threshold=0.32,
    trap_threshold=0.60,
    margin_threshold=0.12,
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

print("\n" + "=" * 80)
print("STATISTICS ACROSS THE EXACT 1,770 DECISION BARS")
print("=" * 80)

def print_stats(name, arr):
    p = np.percentile(arr, [0, 5, 10, 25, 50, 75, 90, 95, 100])
    print(f"--- {name} (N={len(arr)}) ---")
    print(f"Mean: {np.mean(arr):.4f} | Std: {np.std(arr):.4f}")
    print(f"Min: {p[0]:.4f} | 5%: {p[1]:.4f} | 10%: {p[2]:.4f} | 25%: {p[3]:.4f} | Median: {p[4]:.4f}")
    print(f"75%: {p[5]:.4f} | 90%: {p[6]:.4f} | 95%: {p[7]:.4f} | Max: {p[8]:.4f}")

print_stats("Bull Score (b_score)", b_scores)
print_stats("Bear Score (r_score)", r_scores)
print_stats("Dialectical Margin", margins)
print_stats("PRM Score", prm_scores)

print("\n" + "=" * 80)
print("SIGNAL COUNTS ON THE 1,770 DECISION BARS")
print("=" * 80)
print(f"{'PRM':<6} | {'Margin':<8} | {'BUY':<6} | {'SELL':<6} | {'Total':<6} | {'BUY%':<7} | {'SELL%':<7} | {'Total%':<7} | {'B:S Ratio':<9}")
print("-" * 80)

for p_g in [0.32, 0.35, 0.40]:
    prm_mask = prm_scores >= p_g
    for m_th in [0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.15]:
        n_buy = int(np.sum(prm_mask & (margins >= m_th)))
        n_sell = int(np.sum(prm_mask & (margins <= -m_th)))
        tot = n_buy + n_sell
        pct_buy = (n_buy / N) * 100.0
        pct_sell = (n_sell / N) * 100.0
        pct_tot = (tot / N) * 100.0
        ratio = (n_buy / max(1, n_sell))
        tag = " <-- Iter 8 baseline" if m_th == 0.12 and p_g == 0.40 else ""
        print(f"{p_g:<6.2f} | {m_th:<8.2f} | {n_buy:<6d} | {n_sell:<6d} | {tot:<6d} | {pct_buy:<7.1f}%| {pct_sell:<7.1f}%| {pct_tot:<7.1f}%| {ratio:<9.2f}{tag}")
    print("-" * 80)
