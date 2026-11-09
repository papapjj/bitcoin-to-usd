import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd
from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)

WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

df_m1 = pd.read_csv(WEEK3_CSV, sep="\t")
col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
col_o = "<OPEN>" if "<OPEN>" in df_m1.columns else "OPEN"
col_h = "<HIGH>" if "<HIGH>" in df_m1.columns else "HIGH"
col_l = "<LOW>" if "<LOW>" in df_m1.columns else "LOW"
col_c = "<CLOSE>" if "<CLOSE>" in df_m1.columns else "CLOSE"
col_v = "<TICKVOL>" if "<TICKVOL>" in df_m1.columns else ("VOL" if "VOL" in df_m1.columns else None)

v_vals = df_m1[col_v].values.astype(np.float64) if col_v else np.ones(len(df_m1), dtype=np.float64) * 100.0
bars = np.column_stack([
    df_m1[col_o].values.astype(np.float64),
    df_m1[col_h].values.astype(np.float64),
    df_m1[col_l].values.astype(np.float64),
    df_m1[col_c].values.astype(np.float64),
    v_vals,
])
closes = bars[:, 3]
N = len(bars)

# Compute ATR
tr = np.maximum(bars[1:, 1] - bars[1:, 2], np.maximum(np.abs(bars[1:, 1] - bars[:-1, 3]), np.abs(bars[1:, 2] - bars[:-1, 3])))
tr = np.insert(tr, 0, bars[0, 1] - bars[0, 2])
atrs = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

data = extract_features_from_csv(WEEK3_CSV)
base_obs = data["ingress"][:, :137]

N = min(len(bars), len(base_obs))

# Build ingress_with_smc
ingress_with_smc = []
for t in range(N):
    start_i = max(0, t - 59)
    slice_bars = bars[start_i : t + 1]
    if len(slice_bars) < 60:
        pad = np.repeat(slice_bars[:1], 60 - len(slice_bars), axis=0)
        m1_buf = np.vstack([pad, slice_bars])
    else:
        m1_buf = slice_bars
        
    ing = extract_canonical_bilateral_ingress_233(
        base_obs_137=base_obs[t],
        m1_bars_60=m1_buf,
        cur_p=closes[t],
        cur_atr=atrs[t],
    )
    ingress_with_smc.append(ing)

ingress_with_smc = np.array(ingress_with_smc, dtype=np.float32)

cfg = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256,
    num_hops=4, num_debate_turns=3, prm_threshold=0.32,
    trap_threshold=0.60, margin_threshold=0.07,
)
engine = BilateralDualCoTEngine(cfg)
engine.load_checkpoint(CHAMPION_PATH)
engine.eval()

with torch.no_grad():
    v = engine.forward_batch(torch.from_numpy(ingress_with_smc).float(), batch_size=256)

b_scores = v.bull_scores.cpu().numpy()
r_scores = v.bear_scores.cpu().numpy()
prm_scores = v.prm_scores.cpu().numpy()
margins = (b_scores - 0.0934) - (r_scores - 0.1662)

print("=" * 80)
print("TRUE MODEL OUTPUT WITH LIVE SMC STRUCTURE ON ALL 10,080 BARS")
print("=" * 80)
print(f"b_score: median={np.median(b_scores):.4f}, mean={np.mean(b_scores):.4f}")
print(f"r_score: median={np.median(r_scores):.4f}, mean={np.mean(r_scores):.4f}")
print(f"prm_score: median={np.median(prm_scores):.4f}, mean={np.mean(prm_scores):.4f}")
print(f"margin: median={np.median(margins):.4f}, mean={np.mean(margins):.4f}")

mask = prm_scores >= 0.32
n_buy = int(np.sum(mask & (margins >= 0.07)))
n_sell = int(np.sum(mask & (margins <= -0.07)))
print(f"\nSignal counts at PRM>=0.32, Margin>=0.07:")
print(f"  BUY signals: {n_buy}")
print(f"  SELL signals: {n_sell}")
print(f"  Total signals: {n_buy + n_sell}")
print(f"  BUY / SELL Ratio: {n_buy / max(1, n_sell):.2f}")
