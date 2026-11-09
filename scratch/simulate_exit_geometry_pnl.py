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
highs = bars[:, 1]
lows = bars[:, 2]
opens = bars[:, 0]
N = len(bars)

# Compute ATR
tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
tr = np.insert(tr, 0, highs[0] - lows[0])
atrs = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values

data = extract_features_from_csv(WEEK3_CSV)
base_obs = data["ingress"][:, :137]
N = min(len(bars), len(base_obs))

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
margins = (b_scores - 0.0989) - (r_scores - 0.1522)

# Now simulate exact trade execution with spread $28.98 and bar-by-bar MFE/MAE
BROKER_SPREAD = 28.98
LOT_SIZE = 0.01
POINT_VALUE = 1.0  # BTCUSD: 1 point = $1 per 1.0 lot, so 0.01 lot = $0.01 per $1 BTC move

def simulate_trades(prm_thresh, margin_thresh, sl_atr_mult, tp_atr_mult, trail_act_mult, trail_lock_mult, loss_cd=3, min_spacing=2, max_pos=3):
    last_entry_bar = -999
    cooldown_until = 0
    consec_losses = 0
    open_positions = []
    closed_trades = []
    
    for t in range(N):
        cur_atr = atrs[t]
        bar_h = highs[t]
        bar_l = lows[t]
        bar_c = closes[t]
        
        # 1. Update open positions
        still_open = []
        for p in open_positions:
            d = p["dir"] # 1: BUY, -1: SELL
            entry_p = p["entry_price"]
            sl_p = p["sl_price"]
            tp_p = p["tp_price"]
            has_trail = p["has_trail"]
            trail_p = p["trail_price"]
            entry_atr = p["entry_atr"]
            
            # Check price movement inside current bar
            closed = False
            exit_p = 0.0
            exit_reason = ""
            
            if d == 1: # BUY
                # Check SL first
                eff_sl = max(sl_p, trail_p) if has_trail else sl_p
                if bar_l <= eff_sl:
                    closed = True
                    exit_p = eff_sl
                    exit_reason = "TRAIL" if (has_trail and eff_sl > entry_p) else "SL"
                elif bar_h >= tp_p:
                    closed = True
                    exit_p = tp_p
                    exit_reason = "TP"
                else:
                    # Update trailing stop
                    fav_move = bar_h - entry_p
                    if fav_move >= trail_act_mult * entry_atr:
                        new_trail = entry_p + trail_lock_mult * entry_atr
                        if (not has_trail) or (new_trail > trail_p):
                            p["has_trail"] = True
                            p["trail_price"] = new_trail
            else: # SELL
                eff_sl = min(sl_p, trail_p) if has_trail else sl_p
                if bar_h >= eff_sl:
                    closed = True
                    exit_p = eff_sl
                    exit_reason = "TRAIL" if (has_trail and eff_sl < entry_p) else "SL"
                elif bar_l <= tp_p:
                    closed = True
                    exit_p = tp_p
                    exit_reason = "TP"
                else:
                    fav_move = entry_p - bar_l
                    if fav_move >= trail_act_mult * entry_atr:
                        new_trail = entry_p - trail_lock_mult * entry_atr
                        if (not has_trail) or (new_trail < trail_p):
                            p["has_trail"] = True
                            p["trail_price"] = new_trail
                            
            if closed:
                # Calculate PnL with spread
                if d == 1: # BUY
                    gross_diff = exit_p - entry_p
                else: # SELL
                    gross_diff = entry_p - exit_p
                # Subtract spread: for BUY entry paid ask, for SELL entry received bid, exit pays spread
                # Net USD pnl for 0.01 lot BTC
                net_pnl = (gross_diff - BROKER_SPREAD) * LOT_SIZE
                is_win = net_pnl > 0
                closed_trades.append({
                    "entry_bar": p["entry_bar"],
                    "exit_bar": t,
                    "dir": d,
                    "pnl": net_pnl,
                    "is_win": is_win,
                    "reason": exit_reason,
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
        
        # 2. Check signal for new entry
        prm = prm_scores[t]
        m = margins[t]
        if prm < prm_thresh:
            continue
            
        d = 1 if m >= margin_thresh else (-1 if m <= -margin_thresh else 0)
        if d == 0:
            continue
            
        if t < cooldown_until or (t - last_entry_bar) < min_spacing or len(open_positions) >= max_pos:
            continue
            
        # Place entry
        entry_p = bar_c
        sl_dist = sl_atr_mult * cur_atr
        tp_dist = tp_atr_mult * cur_atr
        if d == 1:
            sl_p = entry_p - sl_dist
            tp_p = entry_p + tp_dist
        else:
            sl_p = entry_p + sl_dist
            tp_p = entry_p - tp_dist
            
        open_positions.append({
            "entry_bar": t,
            "dir": d,
            "entry_price": entry_p,
            "sl_price": sl_p,
            "tp_price": tp_p,
            "has_trail": False,
            "trail_price": 0.0,
            "entry_atr": cur_atr,
        })
        last_entry_bar = t
        
    df_res = pd.DataFrame(closed_trades)
    tot = len(df_res)
    if tot == 0:
        return None
    wr = df_res["is_win"].mean() * 100
    gp = df_res[df_res["pnl"] > 0]["pnl"].sum()
    gl = abs(df_res[df_res["pnl"] <= 0]["pnl"].sum())
    pf = gp / gl if gl > 0 else 999.0
    net = gp - gl
    
    # Compute max drawdown
    cum_pnl = df_res["pnl"].cumsum()
    peak = cum_pnl.cummax()
    dd = peak - cum_pnl
    max_dd_usd = dd.max() if len(dd) > 0 else 0.0
    max_dd_pct = (max_dd_usd / 10000.0) * 100.0
    
    trades_per_day = tot / 6.0
    buy_wr = df_res[df_res["dir"] == 1]["is_win"].mean() * 100 if len(df_res[df_res["dir"] == 1]) > 0 else 0
    sell_wr = df_res[df_res["dir"] == -1]["is_win"].mean() * 100 if len(df_res[df_res["dir"] == -1]) > 0 else 0
    
    return {
        "trades": tot,
        "daily": trades_per_day,
        "wr": wr,
        "buy_wr": buy_wr,
        "sell_wr": sell_wr,
        "pf": pf,
        "net": net,
        "max_dd_pct": max_dd_pct,
    }

print("=" * 80)
print("GRID SEARCH ACROSS EXIT GEOMETRY AND THRESHOLDS")
print("=" * 80)

candidates = [
    # Baseline Iter 9
    {"name": "Iter 9 Baseline", "prm": 0.32, "m": 0.07, "sl": 2.6, "tp": 3.0, "t_act": 1.4, "t_lock": 0.45, "cd": 2, "sp": 2},
    # Micro-Scalp Higher WR: Quick TP 1.2 to 1.8 ATR, Tight Trailing Lock
    {"name": "Scalp A (tp=1.5, t_act=1.1, lock=0.4)", "prm": 0.35, "m": 0.07, "sl": 2.5, "tp": 1.5, "t_act": 1.1, "t_lock": 0.40, "cd": 3, "sp": 2},
    {"name": "Scalp B (tp=1.8, t_act=1.2, lock=0.5)", "prm": 0.35, "m": 0.07, "sl": 2.5, "tp": 1.8, "t_act": 1.2, "t_lock": 0.50, "cd": 3, "sp": 2},
    {"name": "Scalp C (tp=2.0, t_act=1.3, lock=0.6)", "prm": 0.35, "m": 0.07, "sl": 2.5, "tp": 2.0, "t_act": 1.3, "t_lock": 0.60, "cd": 3, "sp": 2},
    {"name": "Scalp D (tp=2.2, t_act=1.3, lock=0.7)", "prm": 0.35, "m": 0.07, "sl": 2.5, "tp": 2.2, "t_act": 1.3, "t_lock": 0.70, "cd": 3, "sp": 2},
    {"name": "Scalp E (tp=2.5, t_act=1.4, lock=0.8)", "prm": 0.35, "m": 0.07, "sl": 2.5, "tp": 2.5, "t_act": 1.4, "t_lock": 0.80, "cd": 3, "sp": 2},
    # Asymmetric RR: tp=1.5 ATR, sl=1.5 ATR
    {"name": "Symmetric RR (tp=1.5, sl=1.5)", "prm": 0.35, "m": 0.07, "sl": 1.5, "tp": 1.5, "t_act": 1.0, "t_lock": 0.40, "cd": 3, "sp": 2},
    # High Conviction margin=0.08
    {"name": "High Conv (tp=1.8, m=0.08)", "prm": 0.35, "m": 0.08, "sl": 2.5, "tp": 1.8, "t_act": 1.2, "t_lock": 0.50, "cd": 3, "sp": 2},
    {"name": "High Conv (tp=2.0, m=0.08)", "prm": 0.35, "m": 0.08, "sl": 2.5, "tp": 2.0, "t_act": 1.3, "t_lock": 0.60, "cd": 3, "sp": 2},
    {"name": "High Conv (tp=2.2, m=0.08)", "prm": 0.35, "m": 0.08, "sl": 2.5, "tp": 2.2, "t_act": 1.3, "t_lock": 0.70, "cd": 3, "sp": 2},
    # PRM 0.34
    {"name": "PRM 0.34 (tp=1.8, m=0.07)", "prm": 0.34, "m": 0.07, "sl": 2.5, "tp": 1.8, "t_act": 1.2, "t_lock": 0.50, "cd": 3, "sp": 2},
    {"name": "PRM 0.34 (tp=2.0, m=0.07)", "prm": 0.34, "m": 0.07, "sl": 2.5, "tp": 2.0, "t_act": 1.3, "t_lock": 0.60, "cd": 3, "sp": 2},
]

for c in candidates:
    res = simulate_trades(c["prm"], c["m"], c["sl"], c["tp"], c["t_act"], c["t_lock"], c["cd"], c["sp"])
    if res:
        print(f"{c['name']:<35s} | Trades={res['trades']:<3d} ({res['daily']:.1f}/d) | WR={res['wr']:.1f}% (B:{res['buy_wr']:.1f}%, S:{res['sell_wr']:.1f}%) | PF={res['pf']:.2f} | Net=${res['net']:.2f} | MaxDD={res['max_dd_pct']:.2f}%")
