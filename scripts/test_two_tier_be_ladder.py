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
engine = BilateralDualCoTEngine(BilateralDualCoTConfig())
engine.load_checkpoint(ckpt)
engine.eval()

lookback = 60
horizon = 35
spread = 28.98 # Real Exness BTCUSDm spread

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

print("\n" + "=" * 125)
print("  EVALUATING TWO-TIER BREAKEVEN LADDER & TRAILING STOP ON REAL M1 (7 DAYS)")
print("=" * 125)
print(f"{'Diff':<5} {'PRM':<5} {'Bu/Be':<6} {'SL':<4} {'BE1':<8} {'BE2':<8} {'TP':<4} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<6} {'B_W/L':<10} {'S_W/L':<10} {'DD':<6} {'Net PnL ($)'}")
print("-" * 125)

results = []

# Test grid of Two-Tier Breakeven Ladder
# be1_trig / be1_lock: early scratch protection (e.g. at +1.10 ATR, lock +0.10 ATR spread cover)
# be2_trig / be2_lock: profit protection (e.g. at +1.80 ATR, lock +0.80 ATR profit)
for bu_be in [0.38, 0.40, 0.42]:
    for diff_th in [0.10, 0.12, 0.15]:
        for prm_th in [0.30, 0.32, 0.35]:
            for sl_atr, be1_t, be1_l, be2_t, be2_l, tp_atr in [
                (2.8, 1.10, 0.15, 1.80, 0.90, 3.6),
                (2.8, 1.20, 0.20, 1.90, 1.00, 3.8),
                (3.0, 1.15, 0.15, 1.85, 0.90, 4.0),
                (3.0, 1.25, 0.20, 2.00, 1.00, 4.0),
                (2.6, 1.05, 0.10, 1.70, 0.80, 3.5),
                (2.8, 1.00, 0.10, 1.75, 0.85, 3.6),
            ]:
                trades = []
                last_exit = -999
                last_entry = -999
                b_wins, b_losses = 0, 0
                s_wins, s_losses = 0, 0

                for idx, t in enumerate(valid_indices):
                    if t - last_entry < 2: continue
                    if t - last_exit < 1: continue

                    bu = bull_scores[idx]
                    be = bear_scores[idx]
                    pr = prm_scores[idx]

                    is_buy = (bu - be >= diff_th) and (pr >= prm_th) and (bu >= bu_be)
                    is_sell = (be - bu >= diff_th) and (pr >= prm_th) and (be >= bu_be)

                    if not is_buy and not is_sell: continue
                    if is_buy and is_sell: continue

                    cur_atr = max(atrs[t], 10.0)
                    cp = closes[t]

                    if is_buy:
                        entry_p = cp + spread * 0.5
                        cur_sl = cp - sl_atr * cur_atr
                        tp_p = cp + tp_atr * cur_atr
                        be1_trig_p = cp + be1_t * cur_atr
                        be1_lock_p = cp + be1_l * cur_atr
                        be2_trig_p = cp + be2_t * cur_atr
                        be2_lock_p = cp + be2_l * cur_atr
                        ladder_state = 0 # 0=initial, 1=be1, 2=be2

                        for k in range(t + 1, min(t + 1 + horizon, N)):
                            # Check SL exit
                            if lows[k] <= cur_sl:
                                pnl = cur_sl - entry_p
                                last_exit = k
                                break
                            # Check ladder promotions
                            if highs[k] >= be2_trig_p and ladder_state < 2:
                                cur_sl = be2_lock_p
                                ladder_state = 2
                            elif highs[k] >= be1_trig_p and ladder_state < 1:
                                cur_sl = be1_lock_p
                                ladder_state = 1
                            # Check TP exit
                            if highs[k] >= tp_p:
                                pnl = tp_p - entry_p
                                last_exit = k
                                break
                        else:
                            pnl = closes[min(t + horizon, N - 1)] - entry_p
                            last_exit = min(t + horizon, N - 1)

                        if pnl > 0: b_wins += 1
                        else: b_losses += 1
                    else:
                        entry_p = cp - spread * 0.5
                        cur_sl = cp + sl_atr * cur_atr
                        tp_p = cp - tp_atr * cur_atr
                        be1_trig_p = cp - be1_t * cur_atr
                        be1_lock_p = cp - be1_l * cur_atr
                        be2_trig_p = cp - be2_t * cur_atr
                        be2_lock_p = cp - be2_l * cur_atr
                        ladder_state = 0

                        for k in range(t + 1, min(t + 1 + horizon, N)):
                            # Check SL exit
                            if highs[k] >= cur_sl:
                                pnl = entry_p - cur_sl
                                last_exit = k
                                break
                            # Check ladder promotions
                            if lows[k] <= be2_trig_p and ladder_state < 2:
                                cur_sl = be2_lock_p
                                ladder_state = 2
                            elif lows[k] <= be1_trig_p and ladder_state < 1:
                                cur_sl = be1_lock_p
                                ladder_state = 1
                            # Check TP exit
                            if lows[k] <= tp_p:
                                pnl = entry_p - tp_p
                                last_exit = k
                                break
                        else:
                            pnl = entry_p - closes[min(t + horizon, N - 1)]
                            last_exit = min(t + horizon, N - 1)

                        if pnl > 0: s_wins += 1
                        else: s_losses += 1

                    last_entry = t
                    trades.append(pnl)

                tr = np.array(trades)
                if len(tr) >= 100:
                    n = len(tr)
                    wr = np.sum(tr > 0) / n * 100.0
                    gp = np.sum(tr[tr > 0])
                    gl = abs(np.sum(tr[tr < 0]))
                    pf = gp / gl if gl > 0 else 999.0
                    tot_pnl = np.sum(tr)
                    cum = np.cumsum(tr)
                    dd = (np.maximum.accumulate(cum) - cum).max() / max(10000.0, cum.max() + 10000.0) * 100.0
                    results.append({
                        "bu_be": bu_be, "diff": diff_th, "prm": prm_th,
                        "sl": sl_atr, "be1": f"{be1_t}/{be1_l}", "be2": f"{be2_t}/{be2_l}", "tp": tp_atr,
                        "wr": wr, "pf": pf, "n": n, "t_day": n / 7.0,
                        "b_wl": f"{b_wins}/{b_losses}", "s_wl": f"{s_wins}/{s_losses}",
                        "dd": dd, "pnl": tot_pnl
                    })

rdf = pd.DataFrame(results)
rdf = rdf.sort_values(by=["wr", "pf"], ascending=False)
for _, r in rdf.head(25).iterrows():
    print(f"{r['diff']:<5.2f} {r['prm']:<5.2f} {r['bu_be']:<6.2f} {r['sl']:<4.1f} {r['be1']:<8} {r['be2']:<8} {r['tp']:<4.1f} | {r['wr']:5.1f}%  {r['pf']:5.2f}  {int(r['n']):<8} {r['t_day']:<6.1f} {r['b_wl']:<10} {r['s_wl']:<10} {r['dd']:4.1f}% ${r['pnl']:+,.0f}")
