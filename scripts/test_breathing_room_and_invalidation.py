import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np

# Load the exact 7-day dataset
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
opens = df['<OPEN>'].values if '<OPEN>' in df.columns else df['open'].values
highs = df['<HIGH>'].values if '<HIGH>' in df.columns else df['high'].values
lows = df['<LOW>'].values if '<LOW>' in df.columns else df['low'].values
closes = df['<CLOSE>'].values if '<CLOSE>' in df.columns else df['close'].values
N = len(closes)

tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
atr = np.zeros(N)
atr[0] = tr[0]
for i in range(1, len(tr)):
    atr[i] = 0.95 * atr[i-1] + 0.05 * tr[i]
atr = np.maximum(atr, 5.0)

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

def simulate_advanced(
    sl_atr=2.0,
    tp_atr=2.0,
    be_trig_atr=0.80,
    be_lock_atr=0.20,
    trail_trig_atr=1.20,
    trail_dist_atr=0.80,
    use_structural_stop=True,
    min_spacing_bars=3,
    max_active_positions=1,
    cooldown_bars=3,
    max_bars_in_trade=35,
    value_zone_dist=0.50,
    pb_zone_reach=0.35,
    anti_top_thresh=0.85,
    anti_bot_thresh=0.15,
):
    trades = []
    active_pos = []
    last_exit_t = -999
    last_entry_t = -999

    for t in range(60, N):
        cur_atr = atr[t]
        cp = closes[t]
        h_t = highs[t]
        l_t = lows[t]
        o_t = opens[t]
        e20 = ema20[t]
        e50 = ema50[t]
        
        # 1. Update active positions
        still_active = []
        for p in active_pos:
            p_dir = p['dir']
            p_entry = p['entry_p']
            cur_sl = p['cur_sl']
            tp_p = p['tp_p']
            be_active = p['be_active']
            trailing_active = p['trailing_active']
            entry_t = p['entry_t']
            p_atr = p['cur_atr']
            
            pnl = None
            exit_reason = None
            
            if p_dir == 1:
                # Structural trend invalidation (trend broken)
                if use_structural_stop and (cp < e50 - 0.20 * cur_atr or e20 < e50):
                    pnl = cp - p_entry
                    exit_reason = 'STRUCT_EXIT'
                else:
                    # Check Trailing activation
                    if h_t >= p_entry + trail_trig_atr * p_atr:
                        trailing_active = True
                        p['trailing_active'] = True
                    if trailing_active:
                        cur_sl = max(cur_sl, h_t - trail_dist_atr * p_atr)
                        p['cur_sl'] = cur_sl
                    elif not be_active and h_t >= p_entry + be_trig_atr * p_atr:
                        be_active = True
                        cur_sl = max(cur_sl, p_entry + be_lock_atr * p_atr)
                        p['cur_sl'] = cur_sl
                        p['be_active'] = True
                    
                    # Check TP
                    if h_t >= tp_p:
                        pnl = tp_p - p_entry
                        exit_reason = 'TP'
                    elif l_t <= cur_sl:
                        pnl = cur_sl - p_entry
                        exit_reason = 'TRAIL' if trailing_active else ('BE' if be_active else 'SL')
                    elif t - entry_t >= max_bars_in_trade:
                        pnl = cp - p_entry
                        exit_reason = 'TIME'
            else:
                # SELL
                if use_structural_stop and (cp > e50 + 0.20 * cur_atr or e20 > e50):
                    pnl = p_entry - cp
                    exit_reason = 'STRUCT_EXIT'
                else:
                    if l_t <= p_entry - trail_trig_atr * p_atr:
                        trailing_active = True
                        p['trailing_active'] = True
                    if trailing_active:
                        cur_sl = min(cur_sl, l_t + trail_dist_atr * p_atr)
                        p['cur_sl'] = cur_sl
                    elif not be_active and l_t <= p_entry - be_trig_atr * p_atr:
                        be_active = True
                        cur_sl = min(cur_sl, p_entry - be_lock_atr * p_atr)
                        p['cur_sl'] = cur_sl
                        p['be_active'] = True
                        
                    if l_t <= tp_p:
                        pnl = p_entry - tp_p
                        exit_reason = 'TP'
                    elif h_t >= cur_sl:
                        pnl = p_entry - cur_sl
                        exit_reason = 'TRAIL' if trailing_active else ('BE' if be_active else 'SL')
                    elif t - entry_t >= max_bars_in_trade:
                        pnl = p_entry - cp
                        exit_reason = 'TIME'
                        
            if pnl is not None:
                trades.append({
                    'entry_t': entry_t,
                    'exit_t': t,
                    'dir': p_dir,
                    'pnl': pnl,
                    'r_pnl': pnl / p_atr,
                    'reason': exit_reason,
                })
                last_exit_t = t
            else:
                still_active.append(p)
                
        active_pos = still_active
        
        # 2. Check spacing & cooldown
        if len(active_pos) >= max_active_positions:
            continue
        if t - last_entry_t < min_spacing_bars:
            continue
        if t - last_exit_t < cooldown_bars:
            continue
            
        # 3. Value Zone Entry Logic
        h60 = np.max(highs[t-60:t])
        l60 = np.min(lows[t-60:t])
        rng60 = max(1.0, h60 - l60)
        rel_pos = (cp - l60) / rng60
        
        up_trend = (e20 > e50) and (cp > e50)
        buy_value_zone = (l_t <= e20 + pb_zone_reach * cur_atr) and (cp <= e20 + value_zone_dist * cur_atr) and (cp >= e20 - 0.75 * cur_atr)
        buy_trigger = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= 0.20 * cur_atr)
        anti_top = (rel_pos <= anti_top_thresh)
        
        down_trend = (e20 < e50) and (cp < e50)
        sell_value_zone = (h_t >= e20 - pb_zone_reach * cur_atr) and (cp >= e20 - value_zone_dist * cur_atr) and (cp <= e20 + 0.75 * cur_atr)
        sell_trigger = (closes[t] <= opens[t]) or ((highs[t] - closes[t]) >= 0.20 * cur_atr)
        anti_bot = (rel_pos >= anti_bot_thresh)
        
        act = 0
        if up_trend and buy_value_zone and buy_trigger and anti_top:
            act = 1
        elif down_trend and sell_value_zone and sell_trigger and anti_bot:
            act = -1
            
        if act == 0:
            continue
            
        last_entry_t = t
        if act == 1:
            active_pos.append({
                'dir': 1,
                'entry_p': cp,
                'sl_p': cp - sl_atr * cur_atr,
                'tp_p': cp + tp_atr * cur_atr,
                'cur_sl': cp - sl_atr * cur_atr,
                'be_active': False,
                'trailing_active': False,
                'entry_t': t,
                'cur_atr': cur_atr,
            })
        else:
            active_pos.append({
                'dir': -1,
                'entry_p': cp,
                'sl_p': cp + sl_atr * cur_atr,
                'tp_p': cp - tp_atr * cur_atr,
                'cur_sl': cp + sl_atr * cur_atr,
                'be_active': False,
                'trailing_active': False,
                'entry_t': t,
                'cur_atr': cur_atr,
            })
            
    if len(trades) == 0:
        return None
    tdf = pd.DataFrame(trades)
    wins = (tdf['pnl'] > 0).sum()
    losses = (tdf['pnl'] < 0).sum()
    n = len(tdf)
    wr = wins / n * 100.0
    gp = tdf[tdf['pnl'] > 0]['pnl'].sum()
    gl = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
    pf = gp / gl if gl > 0 else 999.0
    cum = tdf['pnl'].cumsum().values
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() / max(1.0, peak.max()) * 100.0 if peak.max() > 0 else 99.0
    
    reasons = tdf['reason'].value_counts().to_dict()
    b_cnt = (tdf['dir'] == 1).sum()
    s_cnt = (tdf['dir'] == -1).sum()
    
    return {
        'wr': wr, 'pf': pf, 'n': n, 'tr_per_day': n / 7.0,
        'b_cnt': b_cnt, 's_cnt': s_cnt,
        'dd': dd, 'net_pnl': tdf['pnl'].sum(), 'r_net': tdf['r_pnl'].sum(),
        'reasons': reasons,
        'sl_atr': sl_atr, 'tp_atr': tp_atr, 'be_trig': be_trig_atr, 'be_lock': be_lock_atr,
        'trail_trig': trail_trig_atr, 'trail_dist': trail_dist_atr, 'struct': use_structural_stop,
    }

print("Running advanced execution geometry exploration...", flush=True)
results = []
for struct in [True, False]:
    for sl in [1.5, 2.0, 2.5]:
        for tp in [1.5, 2.0, 2.5, 3.0]:
            for bet in [0.70, 0.80, 1.00]:
                for bel in [0.15, 0.25]:
                    for tt in [1.20, 1.50]:
                        for td in [0.70, 0.90]:
                            r = simulate_advanced(
                                sl_atr=sl,
                                tp_atr=tp,
                                be_trig_atr=bet,
                                be_lock_atr=bel,
                                trail_trig_atr=tt,
                                trail_dist_atr=td,
                                use_structural_stop=struct,
                                cooldown_bars=3,
                                min_spacing_bars=3,
                            )
                            if r is not None and r['n'] >= 200:
                                results.append(r)

results.sort(key=lambda x: (x['wr'] >= 80.0, x['pf'] >= 3.0, x['pf']), reverse=True)

print(f"\nTop 25 Configurations:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'DD':<6} {'Net_R':<8} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6} {'TT':<5} {'TD':<5} {'Struct':<6}")
print("-" * 105)
for r in results[:25]:
    bs = f"{r['b_cnt']}/{r['s_cnt']}"
    print(f"{r['wr']:5.2f}%   {r['pf']:5.2f}  {r['n']:<8} {r['tr_per_day']:<8.1f} {bs:<12} {r['dd']:4.1f}% {r['r_net']:<8.1f} {r['sl_atr']:<5.1f} {r['tp_atr']:<5.1f} {r['be_trig']:<6.2f} {r['be_lock']:<6.2f} {r['trail_trig']:<5.2f} {r['trail_dist']:<5.2f} {str(r['struct']):<6}")
