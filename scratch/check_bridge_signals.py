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
print(f"Total agent0 requests in bridge.csv: {N}")

# Check the date range
print("Earliest time in bridge:", df_agent0["time"].min())
print("Latest time in bridge:", df_agent0["time"].max())

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

# Compare two margins:
# Margin A: using (0.0934, 0.1662)
margin_A = (b_scores - 0.0934) - (r_scores - 0.1662)
# Margin B: raw (b_scores - r_scores)
margin_B = b_scores - r_scores
# Margin C: inverted?! (r_scores - 0.1662) - (b_scores - 0.0934)
margin_C = (r_scores - 0.1662) - (b_scores - 0.0934)

print("\nSignal counts across the bridge requests:")
print(f"Total packets: {N}")
mask = prm_scores >= 0.32
print(f"Packets with PRM >= 0.32: {mask.sum()} ({mask.mean()*100:.1f}%)")

print("\n--- Under Margin A: (b - 0.0934) - (r - 0.1662) ---")
buy_A = mask & (margin_A >= 0.07)
sell_A = mask & (margin_A <= -0.07)
print(f"BUY signals: {buy_A.sum()}")
print(f"SELL signals: {sell_A.sum()}")

print("\n--- Under Margin B: raw (b - r) ---")
buy_B = mask & (margin_B >= 0.07)
sell_B = mask & (margin_B <= -0.07)
print(f"BUY signals: {buy_B.sum()}")
print(f"SELL signals: {sell_B.sum()}")

print("\n--- Under Margin C: (r - 0.1662) - (b - 0.0934) ---")
buy_C = mask & (margin_C >= 0.07)
sell_C = mask & (margin_C <= -0.07)
print(f"BUY signals: {buy_C.sum()}")
print(f"SELL signals: {sell_C.sum()}")
