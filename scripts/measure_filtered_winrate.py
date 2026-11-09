import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd
import torch

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    extract_canonical_bilateral_ingress_233,
)
from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import (
    add_features,
    MARKET_FEATURE_COLUMNS,
    TIMING_FEATURE_COLUMNS,
    WYCKOFF_FEATURE_COLUMNS,
)

# 1. Ingest Data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
    df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']

feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
N_total = len(feat_df)

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

# Canonical 137D features
m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
slots = np.zeros((N_total, 40), dtype=np.float32)
account = np.zeros((N_total, 8), dtype=np.float32)
m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
execution = np.zeros((N_total, 11), dtype=np.float32)
timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

lookback = 60
horizon = 25
start_idx = lookback
end_idx = N_total - horizon
valid_count = end_idx - start_idx

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 1.0)
    m1_bars_60 = np.column_stack([
        opens[t-lookback:t],
        highs[t-lookback:t],
        lows[t-lookback:t],
        closes[t-lookback:t],
    ])
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)

config = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256, num_hops=4, num_debate_turns=3,
    prm_threshold=0.32, trap_threshold=0.60
)
engine = BilateralDualCoTEngine(config)
engine.load_checkpoint("outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt")
engine.eval()

with torch.no_grad():
    verdict = engine.forward_batch(torch.from_numpy(ingress_matrix).float(), batch_size=512)
    actions = verdict.actions.numpy()
    b_scores = verdict.bull_scores.numpy()
    r_scores = verdict.bear_scores.numpy()
    prm_scores = verdict.prm_scores.numpy()

val_closes = closes[start_idx:end_idx]
val_highs = highs[start_idx:end_idx]
val_lows = lows[start_idx:end_idx]
val_opens = opens[start_idx:end_idx]
val_atrs = atrs[start_idx:end_idx]

val_ema_diff = base_137[start_idx:end_idx, 11]
val_p_e20 = base_137[start_idx:end_idx, 26]
val_p_e50 = base_137[start_idx:end_idx, 27]
val_rp = base_137[start_idx:end_idx, 13]
val_lw = base_137[start_idx:end_idx, 9]
val_uw = base_137[start_idx:end_idx, 8]

# H1 Macro Trend
s_c = pd.Series(closes)
h1_ema21 = s_c.ewm(span=21*60).mean().values
h1_ema50 = s_c.ewm(span=50*60).mean().values
val_h1_bull = (h1_ema21 > h1_ema50)[start_idx:end_idx]
val_h1_bear = (h1_ema21 < h1_ema50)[start_idx:end_idx]

REAL_SPREAD = 28.98 # Exness MT5 BTCUSDm spread

# Test filtering combinations
print("MEASURING PRECISION FILTERING ON MODEL PREDICTIONS:")
print(f"{'SL':<5} {'TP':<5} {'BE_t/l':<10} {'Filters':<25} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<6} {'B_W/L':<10} {'S_W/L':<10} {'DD':<6} {'Net PnL ($)'}")
print("-" * 120)

def test_config(sl_atr, tp_atr, be_t=None, be_l=0.0, use_h1=False, use_pullback=True, use_cb=True, max_hold=120):
    trades = []
    last_entry = -999
    last_exit = -999
    consec_sl = 0
    pause_until = 0
    b_w, b_l, s_w, s_l = 0, 0, 0, 0
    
    for t in range(valid_count - max_hold):
        if t < pause_until: continue
        if t - last_entry < 2: continue
        if t - last_exit < 1: continue
        
        act = int(actions[t])
        if act == 0: continue
        
        bs = b_scores[t]
        rs = r_scores[t]
        prm = prm_scores[t]
        diff = bs - rs
        
        if use_h1:
            if act == 1 and not val_h1_bull[t]: continue
            if act == -1 and not val_h1_bear[t]: continue
            
        if act == 1:
            if not (diff >= 0.05 and bs >= 0.48 and prm >= 0.33): continue
            if use_pullback:
                if not (val_ema_diff[t] > 0 and val_p_e50[t] > 0 and val_p_e20[t] <= 0.40 and val_rp[t] <= 0.80): continue
                if not (val_closes[t] >= val_opens[t] or val_lw[t] >= 0.20): continue
        elif act == -1:
            if not (-diff >= 0.05 and rs >= 0.48 and prm >= 0.33): continue
            if use_pullback:
                if not (val_ema_diff[t] < 0 and val_p_e50[t] < 0 and val_p_e20[t] >= -0.40 and val_rp[t] >= 0.20): continue
                if not (val_closes[t] <= val_opens[t] or val_uw[t] >= 0.20): continue
                
        last_entry = t
        cur_atr = max(val_atrs[t], 10.0)
        cp = val_closes[t]
        sl_dist = sl_atr * cur_atr
        tp_dist = tp_atr * cur_atr
        bet_dist = be_t * cur_atr if be_t else 999999.0
        bel_dist = be_l * cur_atr
        
        pnl = 0.0
        
        if act == 1:
            entry_p = cp + REAL_SPREAD * 0.5
            cur_sl = cp - sl_dist
            tp_p = cp + tp_dist
            best_bid = cp
            for k in range(t + 1, min(t + max_hold, valid_count)):
                best_bid = max(best_bid, val_highs[k])
                if best_bid >= cp + bet_dist:
                    cur_sl = max(cur_sl, cp + bel_dist)
                if val_lows[k] <= cur_sl:
                    pnl = cur_sl - entry_p
                    last_exit = k
                    break
                elif val_highs[k] >= tp_p:
                    pnl = tp_p - entry_p
                    last_exit = k
                    break
            else:
                pnl = val_closes[min(t + max_hold - 1, valid_count - 1)] - entry_p
                last_exit = min(t + max_hold - 1, valid_count - 1)
            if pnl > 0: b_w += 1
            else: b_l += 1
        else:
            entry_p = cp - REAL_SPREAD * 0.5
            cur_sl = cp + sl_dist
            tp_p = cp - tp_dist
            best_ask = cp
            for k in range(t + 1, min(t + max_hold, valid_count)):
                best_ask = min(best_ask, val_lows[k] + REAL_SPREAD)
                if best_ask <= cp - bet_dist:
                    cur_sl = min(cur_sl, cp - bel_dist)
                if val_highs[k] + REAL_SPREAD >= cur_sl:
                    pnl = entry_p - cur_sl
                    last_exit = k
                    break
                elif val_lows[k] + REAL_SPREAD <= tp_p:
                    pnl = entry_p - tp_p
                    last_exit = k
                    break
            else:
                pnl = entry_p - (val_closes[min(t + max_hold - 1, valid_count - 1)] + REAL_SPREAD)
                last_exit = min(t + max_hold - 1, valid_count - 1)
            if pnl > 0: s_w += 1
            else: s_l += 1
            
        trades.append(pnl)
        if use_cb:
            if pnl <= 0:
                consec_sl += 1
                if consec_sl >= 2:
                    pause_until = last_exit + 45
                    consec_sl = 0
            else:
                consec_sl = 0
                
    tr = np.array(trades)
    if len(tr) == 0: return
    n = len(tr)
    wr = np.sum(tr > 0) / n * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    cum = np.cumsum(tr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() / max(10000.0, peak.max() + 10000.0) * 100.0
    tot_pnl = np.sum(tr)
    
    b_str = f"{be_t}/{be_l}" if be_t else "None"
    f_str = f"H1={use_h1} PB={use_pullback} CB={use_cb}"
    b_wl = f"{b_w}/{b_l}"
    s_wl = f"{s_w}/{s_l}"
    print(f"{sl_atr:<5.1f} {tp_atr:<5.1f} {b_str:<10} {f_str:<25} | {wr:5.1f}%  {pf:5.2f}  {n:<8} {n/7.0:<6.1f} {b_wl:<10} {s_wl:<10} {dd:4.1f}% ${tot_pnl:+,.0f}")

for sl in [4.0, 5.0, 6.0]:
    for tp in [1.5, 2.0, 3.0]:
        for bet, bel in [(None, 0.0), (1.5, 0.8), (2.0, 1.2)]:
            for h1 in [False, True]:
                test_config(sl, tp, bet, bel, use_h1=h1)
