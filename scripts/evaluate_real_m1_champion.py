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

def run_evaluation():
    csv_path = ROOT / "outputs" / "btcusd_m1_7day_with_warmup.csv"
    print(f"Loading data from {csv_path}...")
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

    # Load Champion Neural Engine
    engine = BilateralDualCoTEngine(BilateralDualCoTConfig())
    ckpt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    print(f"Loading checkpoint {ckpt.name}...")
    engine.load_checkpoint(ckpt)
    engine.eval()

    # Pre-extract ingress features for all valid bars
    lookback = 60
    horizon = 45
    spread = 28.98 # Exness BTCUSDm real spread in USD

    print("Extracting 233D ingress features on real M1 bars...")
    batch_size = 512
    all_ingress = []
    valid_indices = list(range(lookback, N - horizon))

    for t in valid_indices:
        cur_p = closes[t]
        cur_atr = max(atrs[t], 10.0)
        m1_60 = np.column_stack([
            opens[t-lookback:t],
            highs[t-lookback:t],
            lows[t-lookback:t],
            closes[t-lookback:t],
        ])
        ing = extract_canonical_bilateral_ingress_233(base_137[t], m1_60, cur_p, cur_atr)
        all_ingress.append(ing)

    all_ingress_tensor = torch.from_numpy(np.array(all_ingress, dtype=np.float32))

    print("Running batch neural inference...")
    bull_scores = []
    bear_scores = []
    prm_scores = []

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

    print(f"Inference complete: Bull range [{bull_scores.min():.3f}, {bull_scores.max():.3f}], Bear range [{bear_scores.min():.3f}, {bear_scores.max():.3f}], PRM range [{prm_scores.min():.3f}, {prm_scores.max():.3f}]")

    print("\n--- GRID SEARCH: CALIBRATED BILATERAL EXECUTION ---")
    results = []

    for diff_thresh in [0.08, 0.12, 0.15]:
        for prm_thresh in [0.30, 0.35, 0.38]:
            for bear_conf_thresh in [0.25, 0.30, 0.35]:
                for sl_atr, be_trig, be_lock, tp_atr in [
                    (2.80, 2.00, 0.80, 4.00),
                    (3.00, 2.00, 1.00, 4.00),
                ]:
                    trades = []
                    last_exit = -999
                    last_entry = -999
                    b_wins, b_losses = 0, 0
                    s_wins, s_losses = 0, 0

                    for idx, t in enumerate(valid_indices):
                        if t - last_entry < 2:
                            continue
                        if t - last_exit < 1:
                            continue

                        bu = bull_scores[idx]
                        be = bear_scores[idx]
                        pr = prm_scores[idx]

                        # Calibrated threshold reflecting the empirical output distribution:
                        # Bull reaches ~0.90, Bear reaches ~0.47
                        is_buy = (bu - be >= diff_thresh) and (pr >= prm_thresh) and (bu >= 0.40)
                        is_sell = (be - bu >= diff_thresh * 0.7) and (pr >= prm_thresh) and (be >= bear_conf_thresh)

                        if not is_buy and not is_sell:
                            continue
                        if is_buy and is_sell:
                            continue

                        cur_atr = max(atrs[t], 10.0)
                        cp = closes[t]

                        pnl = 0.0

                        if is_buy:
                            entry_p = cp + spread * 0.5
                            cur_sl = cp - sl_atr * cur_atr
                            tp_p = cp + tp_atr * cur_atr
                            be_trig_p = cp + be_trig * cur_atr
                            be_lock_p = cp + be_lock * cur_atr
                            be_active = False

                            for k in range(t + 1, min(t + 1 + horizon, N)):
                                if lows[k] <= cur_sl:
                                    pnl = cur_sl - entry_p
                                    last_exit = k
                                    break
                                if highs[k] >= be_trig_p and not be_active:
                                    cur_sl = be_lock_p
                                    be_active = True
                                if highs[k] >= tp_p:
                                    pnl = tp_p - entry_p
                                    last_exit = k
                                    break
                            else:
                                pnl = closes[min(t + horizon, N - 1)] - entry_p
                                last_exit = min(t + horizon, N - 1)

                            if pnl > 0:
                                b_wins += 1
                            else:
                                b_losses += 1

                        else: # SELL
                            entry_p = cp - spread * 0.5
                            cur_sl = cp + sl_atr * cur_atr
                            tp_p = cp - tp_atr * cur_atr
                            be_trig_p = cp - be_trig * cur_atr
                            be_lock_p = cp - be_lock * cur_atr
                            be_active = False

                            for k in range(t + 1, min(t + 1 + horizon, N)):
                                if highs[k] + spread >= cur_sl:
                                    pnl = entry_p - cur_sl
                                    last_exit = k
                                    break
                                if lows[k] + spread <= be_trig_p and not be_active:
                                    cur_sl = be_lock_p
                                    be_active = True
                                if lows[k] + spread <= tp_p:
                                    pnl = entry_p - tp_p
                                    last_exit = k
                                    break
                            else:
                                pnl = entry_p - (closes[min(t + horizon, N - 1)] + spread)
                                last_exit = min(t + horizon, N - 1)

                            if pnl > 0:
                                s_wins += 1
                            else:
                                s_losses += 1

                        last_entry = t
                        trades.append(pnl)

                    tr = np.array(trades)
                    if len(tr) >= 30:
                        n = len(tr)
                        wr = np.sum(tr > 0) / n * 100.0
                        gp = np.sum(tr[tr > 0])
                        gl = abs(np.sum(tr[tr < 0]))
                        pf = gp / gl if gl > 0 else 999.0
                        tot_pnl = np.sum(tr)
                        cum = np.cumsum(tr)
                        dd = (np.maximum.accumulate(cum) - cum).max() / max(10000.0, cum.max() + 10000.0) * 100.0
                        results.append({
                            "diff": diff_thresh, "prm": prm_thresh, "b_conf": bear_conf_thresh,
                            "sl": sl_atr, "be_trig": be_trig, "be_lock": be_lock, "tp": tp_atr,
                            "wr": wr, "pf": pf, "n": n, "tr_per_day": n / 7.0,
                            "b_w": b_wins, "b_l": b_losses, "s_w": s_wins, "s_l": s_losses,
                            "dd": dd, "pnl": tot_pnl
                        })

    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        rdf = rdf.sort_values(by=["pf", "wr"], ascending=False)
        print(f"\n{'Diff':<6} {'PRM':<6} {'B_Conf':<7} {'SL':<4} {'BE_T':<5} {'BE_L':<5} {'TP':<4} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<6} {'B_W/L':<10} {'S_W/L':<10} {'DD':<6} {'Net PnL ($)'}")
        print("-" * 125)
        for _, r in rdf.head(30).iterrows():
            b_wl = f"{r['b_w']}/{r['b_l']}"
            s_wl = f"{r['s_w']}/{r['s_l']}"
            print(f"{r['diff']:<6.2f} {r['prm']:<6.2f} {r['b_conf']:<7.2f} {r['sl']:<4.1f} {r['be_trig']:<5.1f} {r['be_lock']:<5.1f} {r['tp']:<4.1f} | {r['wr']:5.1f}%  {r['pf']:5.2f}  {int(r['n']):<8} {r['tr_per_day']:<6.1f} {b_wl:<10} {s_wl:<10} {r['dd']:4.1f}% ${r['pnl']:+,.0f}")
    else:
        print("No configurations generated >= 30 trades.")

if __name__ == "__main__":
    run_evaluation()
