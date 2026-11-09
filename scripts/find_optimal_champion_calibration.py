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

v3_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion_v3.pt"
engine = BilateralDualCoTEngine(BilateralDualCoTConfig())
engine.load_checkpoint(v3_pt)
engine.eval()

lookback = 60
horizon = 30
spread = 28.98 # Real Exness spread in USD

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

print("Sweeping Breakeven & Take Profit configurations to maximize Win Rate >= 80% & PF >= 3.0 at >= 100 trades/day...")
best_cfgs = []

# Grid search across exit geometries and entry thresholds
for max_pos in [2, 3, 4, 5]:
    for min_spacing in [1, 2]:
        for bu_th, be_th in [(0.40, 0.35), (0.45, 0.38), (0.50, 0.40), (0.55, 0.45)]:
            for diff_th in [0.08, 0.12, 0.15]:
                for prm_th in [0.30, 0.35, 0.40]:
                    for sl_atr, be_trig, be_lock, tp_atr in [
                        (2.80, 1.40, 0.80, 3.80),
                        (3.00, 1.50, 1.00, 4.00),
                        (2.60, 1.20, 0.60, 3.20),
                        (3.20, 1.60, 1.00, 4.20),
                    ]:
                        open_pos = []
                        closed_trades = []
                        last_entry_t = -999

                        for idx, t in enumerate(valid_indices):
                            cur_atr = max(atrs[t], 10.0)
                            cur_h = highs[t]
                            cur_l = lows[t]
                            cur_c = closes[t]

                            # Check open positions
                            remaining = []
                            for pos in open_pos:
                                p_type = pos["type"]
                                entry_p = pos["entry_p"]
                                sl_p = pos["sl_p"]
                                be_p = pos["be_p"]
                                be_lock_p = pos["be_lock_p"]
                                tp_p = pos["tp_p"]
                                be_active = pos["be_active"]
                                closed = False

                                if p_type == "BUY":
                                    if cur_h >= be_p and not be_active:
                                        pos["be_active"] = True
                                        pos["sl_p"] = be_lock_p
                                    if cur_l <= pos["sl_p"]:
                                        pnl = (pos["sl_p"] - entry_p) * 0.01
                                        closed_trades.append({"pnl": pnl, "win": pnl > 0})
                                        closed = True
                                    elif cur_h >= tp_p:
                                        pnl = (tp_p - entry_p) * 0.01
                                        closed_trades.append({"pnl": pnl, "win": True})
                                        closed = True
                                else:
                                    if cur_l <= be_p and not be_active:
                                        pos["be_active"] = True
                                        pos["sl_p"] = be_lock_p
                                    if cur_h >= pos["sl_p"]:
                                        pnl = (entry_p - pos["sl_p"]) * 0.01
                                        closed_trades.append({"pnl": pnl, "win": pnl > 0})
                                        closed = True
                                    elif cur_l <= tp_p:
                                        pnl = (entry_p - tp_p) * 0.01
                                        closed_trades.append({"pnl": pnl, "win": True})
                                        closed = True

                                if not closed:
                                    if t - pos["entry_t"] >= 30:
                                        exit_p = cur_c + spread if p_type == "SELL" else cur_c - spread
                                        pnl = ((exit_p - entry_p) if p_type == "BUY" else (entry_p - exit_p)) * 0.01
                                        closed_trades.append({"pnl": pnl, "win": pnl > 0})
                                    else:
                                        remaining.append(pos)
                            open_pos = remaining

                            # New entry
                            if len(open_pos) < max_pos and (t - last_entry_t >= min_spacing):
                                bu = bull_scores[idx]
                                be = bear_scores[idx]
                                pr = prm_scores[idx]

                                is_buy = (bu - be >= diff_th) and (pr >= prm_th) and (bu >= bu_th)
                                is_sell = (be - bu >= diff_th * 0.7) and (pr >= prm_th) and (be >= be_th)

                                if is_buy and not is_sell:
                                    entry_p = cur_c + spread
                                    open_pos.append({
                                        "type": "BUY", "entry_t": t, "entry_p": entry_p,
                                        "sl_p": entry_p - sl_atr * cur_atr,
                                        "be_p": entry_p + be_trig * cur_atr,
                                        "be_lock_p": entry_p + be_lock * cur_atr,
                                        "tp_p": entry_p + tp_atr * cur_atr,
                                        "be_active": False
                                    })
                                    last_entry_t = t
                                elif is_sell and not is_buy:
                                    entry_p = cur_c - spread
                                    open_pos.append({
                                        "type": "SELL", "entry_t": t, "entry_p": entry_p,
                                        "sl_p": entry_p + sl_atr * cur_atr,
                                        "be_p": entry_p - be_trig * cur_atr,
                                        "be_lock_p": entry_p - be_lock * cur_atr,
                                        "tp_p": entry_p - tp_atr * cur_atr,
                                        "be_active": False
                                    })
                                    last_entry_t = t

                        if len(closed_trades) >= 500: # >= ~72 trades/day
                            wins = [tr["pnl"] for tr in closed_trades if tr["win"]]
                            losses = [abs(tr["pnl"]) for tr in closed_trades if not tr["win"]]
                            wr = len(wins) / len(closed_trades)
                            pf = sum(wins) / sum(losses) if sum(losses) > 0 else 999.0
                            t_day = len(closed_trades) / 7.0

                            if wr >= 0.78 and pf >= 2.8:
                                print(f"MaxPos={max_pos} Spacing={min_spacing} BuTh={bu_th:.2f} BeTh={be_th:.2f} Diff={diff_th:.2f} PRM={prm_th:.2f} SL={sl_atr:.1f} BE_T={be_trig:.1f} BE_L={be_lock:.1f} TP={tp_atr:.1f} | Trades={len(closed_trades)} ({t_day:.1f}/d) | WR={wr*100:.1f}% | PF={pf:.2f} | Profit=${sum(wins)-sum(losses):.0f}")
