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

m_29 = feat_df[MARKET_FEATURE_COLUMNS[:29]].values.astype(np.float32)
slots = np.zeros((N_total, 40), dtype=np.float32)
account = np.zeros((N_total, 8), dtype=np.float32)
m_29_58 = feat_df[MARKET_FEATURE_COLUMNS[29:58]].values.astype(np.float32)
execution = np.zeros((N_total, 11), dtype=np.float32)
timing = feat_df[TIMING_FEATURE_COLUMNS].values.astype(np.float32)
wyckoff = feat_df[WYCKOFF_FEATURE_COLUMNS].values.astype(np.float32)
base_137 = np.concatenate([m_29, slots, account, m_29_58, execution, timing, wyckoff], axis=1)

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

# H1 and M5 Macro / Momentum indicators
s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values
ema200 = s_c.ewm(span=200).mean().values

s_h = pd.Series(highs)
s_l = pd.Series(lows)
rolling_h60 = s_h.rolling(60, min_periods=1).max().values
rolling_l60 = s_l.rolling(60, min_periods=1).min().values
rng60 = np.maximum(1.0, rolling_h60 - rolling_l60)
rel_pos_60 = (closes - rolling_l60) / rng60

# M5 momentum
m5_mom = pd.Series(closes).diff(5).values

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
    confidences = verdict.confidences.numpy()

val_closes = closes[start_idx:end_idx]
val_highs = highs[start_idx:end_idx]
val_lows = lows[start_idx:end_idx]
val_opens = opens[start_idx:end_idx]
val_atrs = atrs[start_idx:end_idx]
val_ema20 = ema20[start_idx:end_idx]
val_ema50 = ema50[start_idx:end_idx]
val_ema200 = ema200[start_idx:end_idx]
val_rp = rel_pos_60[start_idx:end_idx]
val_m5 = m5_mom[start_idx:end_idx]

def simulate_hybrid3(
    sl_atr=3.0,
    tp_atr=6.0,
    be_trig=2.0,
    be_lock=0.8,
    trail_atr=1.5,
    max_consec_sl=2,
    breaker_cooldown=45,
    max_bars=120,
):
    trades = []
    last_entry = -999
    last_exit = -999
    b_cnt, s_cnt = 0, 0
    tp_hits, be_hits, sl_hits, trail_hits, time_hits = 0, 0, 0, 0, 0
    
    consec_sl = {1: 0, -1: 0}
    cooldown_until = {1: 0, -1: 0}
    
    for t in range(valid_count - max_bars):
        if t - last_entry < 2: continue
        if t - last_exit < 3: continue
        
        act = int(actions[t])
        if act == 0: continue
        
        # Check circuit breaker
        if t < cooldown_until[act]: continue
        
        cp = val_closes[t]
        cur_atr = val_atrs[t]
        e20 = val_ema20[t]
        e50 = val_ema50[t]
        e200 = val_ema200[t]
        rp = val_rp[t]
        m5 = val_m5[t]
        
        # HTF & Value Zone & Momentum filters:
        if act == 1:
            if not (e20 > e50 and cp > e50): continue
            if not (val_lows[t] <= e20 + 0.35 * cur_atr and cp <= e20 + 0.50 * cur_atr and rp <= 0.85): continue
            if not (val_closes[t] >= val_opens[t] or (val_closes[t] - val_lows[t]) >= 0.20 * cur_atr): continue
            if m5 < 0: continue # Momentum confirmation
        elif act == -1:
            if not (e20 < e50 and cp < e50): continue
            if not (val_highs[t] >= e20 - 0.35 * cur_atr and cp >= e20 - 0.50 * cur_atr and rp >= 0.15): continue
            if not (val_closes[t] <= val_opens[t] or (val_highs[t] - val_closes[t]) >= 0.20 * cur_atr): continue
            if m5 > 0: continue # Momentum confirmation
            
        last_entry = t
        if act == 1: b_cnt += 1
        else: s_cnt += 1
        
        sl_d = sl_atr * cur_atr
        tp_d = tp_atr * cur_atr
        bet_d = be_trig * cur_atr if be_trig is not None else 999999.0
        bel_d = be_lock * cur_atr
        tr_d = trail_atr * cur_atr if trail_atr > 0 else 999999.0
        
        pnl = 0.0
        reason = None
        if act == 1:
            cur_sl = cp - sl_d
            tp_p = cp + tp_d
            be_active = False
            best_p = cp
            for k in range(t + 1, min(t + max_bars, valid_count)):
                best_p = max(best_p, val_highs[k])
                # Trailing stop update
                if trail_atr > 0 and best_p >= cp + bet_d:
                    cur_sl = max(cur_sl, best_p - tr_d, cp + bel_d)
                elif not be_active and val_highs[k] >= cp + bet_d:
                    be_active = True
                    cur_sl = max(cur_sl, cp + bel_d)
                    
                if val_lows[k] <= cur_sl:
                    pnl = cur_sl - cp
                    reason = 'BE' if (cur_sl >= cp + bel_d * 0.9) else 'SL'
                    last_exit = k
                    break
                elif val_highs[k] >= tp_p:
                    pnl = tp_d
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = val_closes[min(t + max_bars - 1, valid_count - 1)] - cp
                reason = 'TIME'
                last_exit = min(t + max_bars - 1, valid_count - 1)
        else:
            cur_sl = cp + sl_d
            tp_p = cp - tp_d
            be_active = False
            best_p = cp
            for k in range(t + 1, min(t + max_bars, valid_count)):
                best_p = min(best_p, val_lows[k])
                # Trailing stop update
                if trail_atr > 0 and best_p <= cp - bet_d:
                    cur_sl = min(cur_sl, best_p + tr_d, cp - bel_d)
                elif not be_active and val_lows[k] <= cp - bet_d:
                    be_active = True
                    cur_sl = min(cur_sl, cp - bel_d)
                    
                if val_highs[k] >= cur_sl:
                    pnl = cp - cur_sl
                    reason = 'BE' if (cur_sl <= cp - bel_d * 0.9) else 'SL'
                    last_exit = k
                    break
                elif val_lows[k] <= tp_p:
                    pnl = tp_d
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = cp - val_closes[min(t + max_bars - 1, valid_count - 1)]
                reason = 'TIME'
                last_exit = min(t + max_bars - 1, valid_count - 1)
                
        if reason == 'TP':
            tp_hits += 1
            consec_sl[act] = 0
        elif reason == 'BE':
            be_hits += 1
            consec_sl[act] = 0
        elif reason == 'SL':
            sl_hits += 1
            consec_sl[act] += 1
            if consec_sl[act] >= max_consec_sl:
                cooldown_until[act] = last_exit + breaker_cooldown
                consec_sl[act] = 0
        elif reason == 'TIME':
            time_hits += 1
            if pnl > 0: consec_sl[act] = 0
            else:
                consec_sl[act] += 1
                if consec_sl[act] >= max_consec_sl:
                    cooldown_until[act] = last_exit + breaker_cooldown
                    consec_sl[act] = 0
                    
        trades.append(pnl)
        
    tr = np.array(trades)
    if len(tr) == 0: return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    wins = np.sum(tr > 0)
    wr = wins / len(tr) * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    cum = np.cumsum(tr)
    dd = (np.maximum.accumulate(cum) - cum).max() / max(1.0, cum.max()) * 100.0 if cum.max() > 0 else 99.0
    net_atr = np.sum(tr)
    
    return wr, pf, len(tr), b_cnt, s_cnt, tp_hits, be_hits, sl_hits, time_hits, dd, net_atr

print("TESTING FULL-STACK HYBRID 3 GEOMETRY:")
print(f"{'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6} {'Trail':<6} | {'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL/TI':<18} {'DD':<6} {'Net ATR'}")
print("-" * 115)

for sl in [2.5, 3.0, 4.0, 5.0]:
    for tp in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for bet, bel in [(1.5, 0.5), (2.0, 0.8), (2.5, 1.0)]:
            for trail in [0.0, 1.2, 1.5, 2.0]:
                wr, pf, n, b, s, tp_h, be_h, sl_h, ti_h, dd, net_a = simulate_hybrid3(
                    sl_atr=sl, tp_atr=tp, be_trig=bet, be_lock=bel, trail_atr=trail
                )
                if pf >= 1.5 and n >= 100:
                    bs = f"{b}/{s}"
                    hits = f"{tp_h}/{be_h}/{sl_h}/{ti_h}"
                    print(f"{sl:<5.1f} {tp:<5.1f} {bet:<6.1f} {bel:<6.1f} {trail:<6.1f} | {wr:5.2f}%   {pf:5.2f}  {n:<8} {n/7.0:<8.1f} {bs:<12} {hits:<18} {dd:4.1f}% +{net_a:.1f}")
