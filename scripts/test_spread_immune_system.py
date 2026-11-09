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

# Authoritative feature shortcuts from base_137
# col 11: ema_diff_atr = (ema20 - ema50) / atr
# col 26: price_vs_ema20_atr = (close - ema20) / atr
# col 27: price_vs_ema50_atr = (close - ema50) / atr
# col 13: range_position = (close - low60) / (high60 - low60)
# col 8: upper_wick_atr
# col 9: lower_wick_atr
# col 7: body_atr
ema_diff_atr = base_137[:, 11]
p_vs_ema20 = base_137[:, 26]
p_vs_ema50 = base_137[:, 27]
rel_pos_60 = base_137[:, 13]
upper_wicks = base_137[:, 8]
lower_wicks = base_137[:, 9]

# H1 Macro Trend (60-bar resampled EMA21 vs EMA50)
s_c = pd.Series(closes)
h1_ema21 = s_c.ewm(span=21*60).mean().values
h1_ema50 = s_c.ewm(span=50*60).mean().values
h1_macro_bull = h1_ema21 > h1_ema50
h1_macro_bear = h1_ema21 < h1_ema50

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
val_h1_bull = h1_macro_bull[start_idx:end_idx]
val_h1_bear = h1_macro_bear[start_idx:end_idx]
val_ema_diff = ema_diff_atr[start_idx:end_idx]
val_p_e20 = p_vs_ema20[start_idx:end_idx]
val_p_e50 = p_vs_ema50[start_idx:end_idx]
val_rp = rel_pos_60[start_idx:end_idx]
val_lw = lower_wicks[start_idx:end_idx]
val_uw = upper_wicks[start_idx:end_idx]

REAL_SPREAD = 28.98 # Exness MT5 BTCUSDm spread

def run_simulation(
    sl_atr=4.0,
    tp_atr=4.0,
    be_trig_atr=2.0,
    be_lock_atr=1.2,
    use_h1_filter=True,
    use_circuit_breaker=True,
    cb_consec_sl=2,
    cb_pause_bars=45,
    min_margin=0.05,
    min_score=0.48,
    prm_gate=0.33,
    max_hold=120,
):
    trades = []
    last_entry = -999
    last_exit = -999
    consec_sl = 0
    pause_until = 0
    
    tp_hits, be_hits, sl_hits, time_hits = 0, 0, 0, 0
    b_wins, b_losses, s_wins, s_losses = 0, 0, 0, 0
    
    for t in range(valid_count - max_hold):
        if t < pause_until: continue
        if t - last_entry < 2: continue
        if t - last_exit < 2: continue
        
        act = int(actions[t])
        if act == 0: continue
        
        bs = b_scores[t]
        rs = r_scores[t]
        prm = prm_scores[t]
        diff = bs - rs
        
        cp = val_closes[t]
        cur_atr = max(val_atrs[t], 10.0)
        
        # 1. H1 Macro Filter
        if use_h1_filter:
            if act == 1 and not val_h1_bull[t]: continue
            if act == -1 and not val_h1_bear[t]: continue
            
        # 2. Model Confidence Gating
        if act == 1:
            if not (diff >= min_margin and bs >= min_score and prm >= prm_gate): continue
            # Microstructure value zone check
            if not (val_ema_diff[t] > 0 and val_p_e50[t] > 0 and val_p_e20[t] <= 0.50 and val_rp[t] <= 0.85): continue
            # Confirmation
            if not (val_closes[t] >= val_opens[t] or val_lw[t] >= 0.20): continue
        elif act == -1:
            if not (-diff >= min_margin and rs >= min_score and prm >= prm_gate): continue
            if not (val_ema_diff[t] < 0 and val_p_e50[t] < 0 and val_p_e20[t] >= -0.50 and val_rp[t] >= 0.15): continue
            if not (val_closes[t] <= val_opens[t] or val_uw[t] >= 0.20): continue
            
        last_entry = t
        sl_dist = sl_atr * cur_atr
        tp_dist = tp_atr * cur_atr
        bet_dist = be_trig_atr * cur_atr if be_trig_atr is not None else 999999.0
        bel_dist = be_lock_atr * cur_atr
        
        pnl = 0.0
        reason = None
        
        if act == 1:
            entry_p = cp + REAL_SPREAD * 0.5
            cur_sl = cp - sl_dist
            tp_p = cp + tp_dist
            best_bid = cp
            
            for k in range(t + 1, min(t + max_hold, valid_count)):
                k_high = val_highs[k]
                k_low = val_lows[k]
                best_bid = max(best_bid, k_high)
                
                if best_bid >= cp + bet_dist:
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
                
            if pnl > 0: b_wins += 1
            else: b_losses += 1
            
        else: # SELL
            entry_p = cp - REAL_SPREAD * 0.5
            cur_sl = cp + sl_dist
            tp_p = cp - tp_dist
            best_ask = cp
            
            for k in range(t + 1, min(t + max_hold, valid_count)):
                k_high = val_highs[k] + REAL_SPREAD
                k_low = val_lows[k] + REAL_SPREAD
                best_ask = min(best_ask, k_low)
                
                if best_ask <= cp - bet_dist:
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
                
            if pnl > 0: s_wins += 1
            else: s_losses += 1
            
        if reason == 'TP': tp_hits += 1
        elif reason == 'BE': be_hits += 1
        elif reason == 'SL': sl_hits += 1
        elif reason == 'TIME': time_hits += 1
        
        trades.append(pnl)
        
        # Circuit Breaker Logic
        if use_circuit_breaker:
            if pnl <= 0:
                consec_sl += 1
                if consec_sl >= cb_consec_sl:
                    pause_until = last_exit + cb_pause_bars
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
        'n': n, 'tr_per_day': n / 7.0, 'wr': wr, 'pf': pf,
        'tp': tp_hits, 'be': be_hits, 'sl': sl_hits, 'ti': time_hits,
        'b_wins': b_wins, 'b_losses': b_losses, 's_wins': s_wins, 's_losses': s_losses,
        'dd': dd, 'pnl': tot_pnl
    }

print("=" * 110)
print(f"{'SL':<5} {'TP':<5} {'BE_t/l':<10} {'H1':<5} {'CB':<5} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<6} {'TP/BE/SL/TI':<16} {'DD':<6} {'Net PnL ($)'}")
print("=" * 110)

for sl in [3.0, 4.0, 5.0]:
    for tp in [2.0, 3.0, 4.0, 5.0]:
        for bet, bel in [(None, 0.0), (2.0, 1.2), (2.5, 1.5)]:
            for h1 in [True, False]:
                for cb in [True, False]:
                    res = run_simulation(
                        sl_atr=sl, tp_atr=tp, be_trig_atr=bet, be_lock_atr=bel,
                        use_h1_filter=h1, use_circuit_breaker=cb
                    )
                    if res and res['pf'] >= 1.20 and res['n'] >= 20:
                        be_s = f"{bet}/{bel}" if bet else "None"
                        hits = f"{res['tp']}/{res['be']}/{res['sl']}/{res['ti']}"
                        print(f"{sl:<5.1f} {tp:<5.1f} {be_s:<10} {str(h1):<5} {str(cb):<5} | {res['wr']:5.1f}%  {res['pf']:5.2f}  {res['n']:<8} {res['tr_per_day']:<6.1f} {hits:<16} {res['dd']:4.1f}% ${res['pnl']:+,.0f}")
