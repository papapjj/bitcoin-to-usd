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

# Load data
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

# 137D features
m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
slots = np.zeros((N_total, 40), dtype=np.float32)
account = np.zeros((N_total, 8), dtype=np.float32)
m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
execution = np.zeros((N_total, 11), dtype=np.float32)
timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

lookback = 60
horizon = 60
start_idx = lookback
end_idx = N_total - horizon
valid_count = end_idx - start_idx

REAL_SPREAD = 28.98

ingress_matrix = np.zeros((valid_count, 233), dtype=np.float32)
for i, t in enumerate(range(start_idx, end_idx)):
    cur_p = closes[t]
    cur_atr = max(atrs[t], 10.0)
    m1_bars_60 = np.column_stack([
        opens[t-lookback:t],
        highs[t-lookback:t],
        lows[t-lookback:t],
        closes[t-lookback:t],
    ])
    ingress_matrix[i] = extract_canonical_bilateral_ingress_233(base_137[t], m1_bars_60, cur_p, cur_atr)

config = BilateralDualCoTConfig(
    input_dim=233, latent_dim=768, hidden_dim=256, num_hops=4, num_debate_turns=3,
    prm_threshold=0.33, trap_threshold=0.60
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
val_base = base_137[start_idx:end_idx]

ema_diff = val_base[:, 11]
p_vs_e20 = val_base[:, 26]
p_vs_e50 = val_base[:, 27]
rel_pos_60 = val_base[:, 13]
body = val_base[:, 7]
uw = val_base[:, 8]
lw = val_base[:, 9]

s_c = pd.Series(closes)
h1_ema21 = s_c.ewm(span=21*60).mean().values
h1_ema50 = s_c.ewm(span=50*60).mean().values
val_h1_bull = (h1_ema21 > h1_ema50)[start_idx:end_idx]
val_h1_bear = (h1_ema21 < h1_ema50)[start_idx:end_idx]

def run_cadence_test(
    sl_atr, tp_atr, be_trig_atr, be_lock_atr, trail_atr,
    min_margin, min_score, prm_gate,
    min_spacing, post_exit_wait,
    cb_limit=2, cb_bars=45, max_hold=90
):
    trades = []
    last_entry = -999
    last_exit = -999
    consec_sl = 0
    pause_until = 0
    tp_h, be_h, sl_h, ti_h = 0, 0, 0, 0
    
    for t in range(valid_count - max_hold):
        if t < pause_until: continue
        if t - last_entry < min_spacing: continue
        if t - last_exit < post_exit_wait: continue
        
        act = int(actions[t])
        if act == 0: continue
        
        bs = b_scores[t]
        rs = r_scores[t]
        prm = prm_scores[t]
        diff = bs - rs
        
        if act == 1:
            if not val_h1_bull[t]: continue
            in_trend = ema_diff[t] > 0.0 and p_vs_e50[t] > 0.0
            in_val = p_vs_e20[t] <= 0.40
            not_top = rel_pos_60[t] <= 0.85
            rejection = body[t] >= 0.0 or lw[t] >= 0.15
            if not (in_trend and in_val and not_top and rejection): continue
            if not (diff >= min_margin and bs >= min_score and prm >= prm_gate): continue
        elif act == 2 or act == -1:
            if not val_h1_bear[t]: continue
            in_trend = ema_diff[t] < 0.0 and p_vs_e50[t] < 0.0
            in_val = p_vs_e20[t] >= -0.40
            not_bot = rel_pos_60[t] >= 0.15
            rejection = body[t] <= 0.0 or uw[t] >= 0.15
            if not (in_trend and in_val and not_bot and rejection): continue
            if not (-diff >= min_margin and rs >= min_score and prm >= prm_gate): continue
        else:
            continue
            
        last_entry = t
        cur_atr = max(val_atrs[t], 10.0)
        cp = val_closes[t]
        sl_dist = sl_atr * cur_atr
        tp_dist = tp_atr * cur_atr
        bet_dist = be_trig_atr * cur_atr if be_trig_atr else 999999.0
        bel_dist = be_lock_atr * cur_atr
        tr_dist = trail_atr * cur_atr if trail_atr > 0 else 999999.0
        
        pnl = 0.0
        reason = None
        
        if act == 1:
            entry_p = cp + REAL_SPREAD * 0.5
            cur_sl = cp - sl_dist
            tp_p = cp + tp_dist
            best_bid = cp
            be_active = False
            
            for k in range(t + 1, min(t + max_hold, valid_count)):
                k_high = val_highs[k]
                k_low = val_lows[k]
                best_bid = max(best_bid, k_high)
                
                if trail_atr > 0 and best_bid >= cp + bet_dist:
                    cur_sl = max(cur_sl, best_bid - tr_dist, cp + bel_dist)
                elif not be_active and k_high >= cp + bet_dist:
                    be_active = True
                    cur_sl = max(cur_sl, cp + bel_dist)
                    
                if k_low <= cur_sl:
                    pnl = cur_sl - entry_p
                    reason = 'BE' if cur_sl >= cp + bel_dist * 0.9 else 'SL'
                    last_exit = k
                    break
                elif k_high >= tp_p:
                    pnl = tp_p - entry_p
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = val_closes[min(t + max_hold - 1, valid_count - 1)] - entry_p
                reason = 'TIME'
                last_exit = min(t + max_hold - 1, valid_count - 1)
                
        else: # SELL
            entry_p = cp - REAL_SPREAD * 0.5
            cur_sl = cp + sl_dist
            tp_p = cp - tp_dist
            best_ask = cp
            be_active = False
            
            for k in range(t + 1, min(t + max_hold, valid_count)):
                k_high = val_highs[k] + REAL_SPREAD
                k_low = val_lows[k] + REAL_SPREAD
                best_ask = min(best_ask, k_low)
                
                if trail_atr > 0 and best_ask <= cp - bet_dist:
                    cur_sl = min(cur_sl, best_ask + tr_dist, cp - bel_dist)
                elif not be_active and k_low <= cp - bet_dist:
                    be_active = True
                    cur_sl = min(cur_sl, cp - bel_dist)
                    
                if k_high >= cur_sl:
                    pnl = entry_p - cur_sl
                    reason = 'BE' if cur_sl <= cp - bel_dist * 0.9 else 'SL'
                    last_exit = k
                    break
                elif k_low <= tp_p:
                    pnl = entry_p - tp_p
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = entry_p - (val_closes[min(t + max_hold - 1, valid_count - 1)] + REAL_SPREAD)
                reason = 'TIME'
                last_exit = min(t + max_hold - 1, valid_count - 1)
                
        if reason == 'TP': tp_h += 1
        elif reason == 'BE': be_h += 1
        elif reason == 'SL': sl_h += 1
        elif reason == 'TIME': ti_h += 1
        
        trades.append(pnl)
        
        if pnl <= 0:
            consec_sl += 1
            if consec_sl >= cb_limit:
                pause_until = last_exit + cb_bars
                consec_sl = 0
        else:
            consec_sl = 0
            
    tr = np.array(trades)
    if len(tr) == 0: return None
    n = len(tr)
    wr = np.sum(tr > 0) / n * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    cum = np.cumsum(tr)
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() / max(10000.0, peak.max() + 10000.0) * 100.0
    tot_pnl = np.sum(tr)
    
    return {
        'n': n, 'tr_d': n/7.0, 'wr': wr, 'pf': pf,
        'tp': tp_h, 'be': be_h, 'sl': sl_h, 'ti': ti_h,
        'dd': dd, 'pnl': tot_pnl
    }

print("\n--- Tuning for Maximum Cadence with WR >= 80% and PF >= 2.0 ---")
for spacing in [1, 2]:
    for exit_wait in [0, 1]:
        for mm in [0.02, 0.03, 0.04]:
            for ms in [0.45, 0.48]:
                for prm in [0.28, 0.32]:
                    res = run_cadence_test(
                        sl_atr=4.5, tp_atr=3.5, be_trig_atr=1.8, be_lock_atr=1.0, trail_atr=1.5,
                        min_margin=mm, min_score=ms, prm_gate=prm,
                        min_spacing=spacing, post_exit_wait=exit_wait
                    )
                    if res and res['wr'] >= 80.0 and res['pf'] >= 2.0:
                        hits = f"{res['tp']}/{res['be']}/{res['sl']}/{res['ti']}"
                        print(f"Sp={spacing} Ex={exit_wait} M={mm:.2f} S={ms:.2f} P={prm:.2f} | WR: {res['wr']:5.1f}% | PF: {res['pf']:5.2f} | N: {res['n']:3d} ({res['tr_d']:4.1f}/d) | Hits: {hits:<13} | DD: {res['dd']:4.1f}% | PnL: ${res['pnl']:+,.0f}")
