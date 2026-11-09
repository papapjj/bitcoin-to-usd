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
horizon = 45
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

print("\n" + "=" * 135)
print("  REAL MT5 TWO-TIER SYSTEM: PARTIAL TP + BREAKEVEN LOCK + TRAILING STOP")
print("=" * 135)
print(f"{'BuTh':<5} {'BeTh':<5} {'Diff':<5} {'PRM':<5} {'SL':<4} {'BE_T':<5} {'BE_L':<5} {'PartTP':<7} {'Trail':<6} {'FullTP':<7} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<6} {'B_W/L':<10} {'S_W/L':<10} {'DD':<6} {'Net PnL ($)'}")
print("-" * 135)

configs = []
for bu_th, be_th in [(0.56, 0.50), (0.58, 0.52), (0.60, 0.54)]:
    for diff_th in [0.12, 0.15]:
        for prm_th in [0.35, 0.38]:
            for sl_atr, be_trig, be_lock, part_tp, trail_atr, full_tp in [
                (2.2, 1.35, 0.80, 2.40, 1.00, 4.50),
                (2.2, 1.40, 0.85, 2.50, 1.00, 4.80),
                (2.4, 1.40, 0.85, 2.60, 1.10, 5.00),
                (2.4, 1.50, 0.90, 2.80, 1.10, 5.20),
                (2.6, 1.50, 0.90, 2.80, 1.20, 5.50),
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

                    is_buy = (bu - be >= diff_th) and (pr >= prm_th) and (bu >= bu_th)
                    is_sell = (be - bu >= diff_th) and (pr >= prm_th) and (be >= be_th)

                    if not is_buy and not is_sell: continue
                    if is_buy and is_sell: continue

                    cur_atr = max(atrs[t], 10.0)
                    cp = closes[t]

                    if is_buy:
                        entry_p = cp + spread * 0.5
                        cur_sl = cp - sl_atr * cur_atr
                        be_trig_p = cp + be_trig * cur_atr
                        be_lock_p = cp + be_lock * cur_atr
                        part_tp_p = cp + part_tp * cur_atr
                        full_tp_p = cp + full_tp * cur_atr

                        remaining_pos = 1.0
                        total_pnl = 0.0
                        be_active = False
                        part_taken = False
                        highest_p = entry_p

                        for k in range(t + 1, min(t + 1 + horizon, N)):
                            if highs[k] > highest_p:
                                highest_p = highs[k]

                            # Breakeven lock
                            if highs[k] >= be_trig_p and not be_active:
                                cur_sl = max(cur_sl, be_lock_p)
                                be_active = True

                            # Trailing stop
                            if be_active:
                                trail_p = highest_p - trail_atr * cur_atr
                                if trail_p > cur_sl:
                                    cur_sl = trail_p

                            # Check Partial TP (50% position)
                            if highs[k] >= part_tp_p and not part_taken:
                                part_pnl = (part_tp_p - entry_p) * 0.5
                                total_pnl += part_pnl
                                remaining_pos = 0.5
                                part_taken = True
                                cur_sl = max(cur_sl, be_lock_p)

                            # Check SL exit for remaining
                            if lows[k] <= cur_sl:
                                exit_pnl = (cur_sl - entry_p) * remaining_pos
                                total_pnl += exit_pnl
                                last_exit = k
                                break

                            # Check Full TP
                            if highs[k] >= full_tp_p:
                                exit_pnl = (full_tp_p - entry_p) * remaining_pos
                                total_pnl += exit_pnl
                                last_exit = k
                                break
                        else:
                            c_exit = closes[min(t + horizon, N - 1)]
                            total_pnl += (c_exit - entry_p) * remaining_pos
                            last_exit = min(t + horizon, N - 1)

                        if total_pnl > 0: b_wins += 1
                        else: b_losses += 1
                    else:
                        entry_p = cp - spread * 0.5
                        cur_sl = cp + sl_atr * cur_atr
                        be_trig_p = cp - be_trig * cur_atr
                        be_lock_p = cp - be_lock * cur_atr
                        part_tp_p = cp - part_tp * cur_atr
                        full_tp_p = cp - full_tp * cur_atr

                        remaining_pos = 1.0
                        total_pnl = 0.0
                        be_active = False
                        part_taken = False
                        lowest_p = entry_p

                        for k in range(t + 1, min(t + 1 + horizon, N)):
                            if lows[k] < lowest_p:
                                lowest_p = lows[k]

                            # Breakeven lock
                            if lows[k] <= be_trig_p and not be_active:
                                cur_sl = min(cur_sl, be_lock_p)
                                be_active = True

                            # Trailing stop
                            if be_active:
                                trail_p = lowest_p + trail_atr * cur_atr
                                if trail_p < cur_sl:
                                    cur_sl = trail_p

                            # Check Partial TP
                            if lows[k] <= part_tp_p and not part_taken:
                                part_pnl = (entry_p - part_tp_p) * 0.5
                                total_pnl += part_pnl
                                remaining_pos = 0.5
                                part_taken = True
                                cur_sl = min(cur_sl, be_lock_p)

                            # Check SL exit
                            if highs[k] >= cur_sl:
                                exit_pnl = (entry_p - cur_sl) * remaining_pos
                                total_pnl += exit_pnl
                                last_exit = k
                                break

                            # Check Full TP
                            if lows[k] <= full_tp_p:
                                exit_pnl = (entry_p - full_tp_p) * remaining_pos
                                total_pnl += exit_pnl
                                last_exit = k
                                break
                        else:
                            c_exit = closes[min(t + horizon, N - 1)]
                            total_pnl += (entry_p - c_exit) * remaining_pos
                            last_exit = min(t + horizon, N - 1)

                        if total_pnl > 0: s_wins += 1
                        else: s_losses += 1

                    last_entry = t
                    trades.append(total_pnl)

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
                    configs.append({
                        "bu_th": bu_th, "be_th": be_th, "diff": diff_th, "prm": prm_th,
                        "sl": sl_atr, "be_t": be_trig, "be_l": be_lock, "part": part_tp, "trail": trail_atr, "full": full_tp,
                        "wr": wr, "pf": pf, "n": n, "t_day": n / 7.0,
                        "b_wl": f"{b_wins}/{b_losses}", "s_wl": f"{s_wins}/{s_losses}",
                        "dd": dd, "pnl": tot_pnl
                    })

cdf = pd.DataFrame(configs)
cdf = cdf.sort_values(by=["pf", "wr"], ascending=False)
for _, r in cdf.head(25).iterrows():
    print(f"{r['bu_th']:<5.2f} {r['be_th']:<5.2f} {r['diff']:<5.2f} {r['prm']:<5.2f} {r['sl']:<4.1f} {r['be_t']:<5.2f} {r['be_l']:<5.2f} {r['part']:<7.2f} {r['trail']:<6.1f} {r['full']:<7.2f} | {r['wr']:5.1f}%  {r['pf']:5.2f}  {int(r['n']):<8} {r['t_day']:<6.1f} {r['b_wl']:<10} {r['s_wl']:<10} {r['dd']:4.1f}% ${r['pnl']:+,.0f}")
