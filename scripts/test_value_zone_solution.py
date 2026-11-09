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
dts = (df['<DATE>'] + ' ' + df['<TIME>']).values if '<DATE>' in df.columns else np.arange(len(closes))
N = len(closes)

# Calculate ATR
tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
atr = np.zeros(N)
atr[0] = tr[0]
for i in range(1, len(tr)):
    atr[i] = 0.95 * atr[i-1] + 0.05 * tr[i]
atr = np.maximum(atr, 5.0)

# Calculate EMA20 and EMA50
s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

print(f"Total bars: {N} ({N/1440.0:.2f} days)")

def simulate(
    sl_atr=2.0,
    tp_atr=1.8,
    be_trig_atr=0.60,
    be_lock_atr=0.40,
    min_spacing_bars=3,
    max_active_positions=1,
    cooldown_bars=3,
    max_bars_in_trade=35,
    value_zone_dist=0.50, # Max distance from EMA20 allowed (in ATR)
    pb_zone_reach=0.35,   # Low/High must have reached within EMA20 +/- this ATR
    anti_top_thresh=0.85,
    anti_bot_thresh=0.15,
):
    trades = []
    active_pos = [] # list of dicts: {'dir', 'entry_p', 'sl_p', 'tp_p', 'cur_sl', 'be_active', 'entry_t', 'cur_atr'}
    last_exit_t = -999
    last_entry_t = -999

    for t in range(60, N):
        cur_atr = atr[t]
        cp = closes[t]
        h_t = highs[t]
        l_t = lows[t]
        o_t = opens[t]
        
        # 1. Update active positions
        still_active = []
        for p in active_pos:
            p_dir = p['dir']
            p_entry = p['entry_p']
            cur_sl = p['cur_sl']
            tp_p = p['tp_p']
            be_active = p['be_active']
            entry_t = p['entry_t']
            
            pnl = None
            exit_reason = None
            
            if p_dir == 1:
                # Check BE activation
                if not be_active and h_t >= p_entry + be_trig_atr * p['cur_atr']:
                    be_active = True
                    cur_sl = max(cur_sl, p_entry + be_lock_atr * p['cur_atr'])
                    p['cur_sl'] = cur_sl
                    p['be_active'] = True
                
                # Check TP
                if h_t >= tp_p:
                    pnl = tp_p - p_entry
                    exit_reason = 'TP'
                # Check SL
                elif l_t <= cur_sl:
                    pnl = cur_sl - p_entry
                    exit_reason = 'BE' if be_active else 'SL'
                # Check timeout
                elif t - entry_t >= max_bars_in_trade:
                    pnl = cp - p_entry
                    exit_reason = 'TIME'
            else:
                # SELL
                # Check BE activation
                if not be_active and l_t <= p_entry - be_trig_atr * p['cur_atr']:
                    be_active = True
                    cur_sl = min(cur_sl, p_entry - be_lock_atr * p['cur_atr'])
                    p['cur_sl'] = cur_sl
                    p['be_active'] = True
                
                # Check TP
                if l_t <= tp_p:
                    pnl = p_entry - tp_p
                    exit_reason = 'TP'
                # Check SL
                elif h_t >= cur_sl:
                    pnl = p_entry - cur_sl
                    exit_reason = 'BE' if be_active else 'SL'
                # Check timeout
                elif t - entry_t >= max_bars_in_trade:
                    pnl = p_entry - cp
                    exit_reason = 'TIME'
                    
            if pnl is not None:
                trades.append({
                    'entry_t': entry_t,
                    'exit_t': t,
                    'dir': p_dir,
                    'entry_p': p_entry,
                    'exit_p': cp if exit_reason == 'TIME' else (tp_p if exit_reason == 'TP' else cur_sl),
                    'pnl': pnl,
                    'r_pnl': pnl / p['cur_atr'],
                    'reason': exit_reason,
                    'bars': t - entry_t,
                })
                last_exit_t = t
            else:
                still_active.append(p)
                
        active_pos = still_active
        
        # 2. Check if we can open new positions
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
        
        # UPTREND BUY:
        # a) Trend: ema20 > ema50 and cp > ema50
        # b) Pullback to Value Zone: low reached ema20 + pb_zone_reach * cur_atr and close <= ema20 + value_zone_dist * cur_atr
        # c) Invalidation floor: close >= ema20 - 0.75 * cur_atr
        # d) Trigger confirmation: green candle OR lower wick >= 0.20 * cur_atr
        # e) Anti-Top: rel_pos <= anti_top_thresh
        up_trend = (ema20[t] > ema50[t]) and (cp > ema50[t])
        buy_value_zone = (l_t <= ema20[t] + pb_zone_reach * cur_atr) and (cp <= ema20[t] + value_zone_dist * cur_atr) and (cp >= ema20[t] - 0.75 * cur_atr)
        buy_trigger = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= 0.20 * cur_atr)
        anti_top = (rel_pos <= anti_top_thresh)
        
        # DOWNTREND SELL:
        # a) Trend: ema20 < ema50 and cp < ema50
        # b) Rally to Value Zone: high reached ema20 - pb_zone_reach * cur_atr and close >= ema20 - value_zone_dist * cur_atr
        # c) Invalidation ceiling: close <= ema20 + 0.75 * cur_atr
        # d) Trigger confirmation: red candle OR upper wick >= 0.20 * cur_atr
        # e) Anti-Bottom: rel_pos >= anti_bot_thresh
        down_trend = (ema20[t] < ema50[t]) and (cp < ema50[t])
        sell_value_zone = (h_t >= ema20[t] - pb_zone_reach * cur_atr) and (cp >= ema20[t] - value_zone_dist * cur_atr) and (cp <= ema20[t] + 0.75 * cur_atr)
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
                'entry_t': t,
                'cur_atr': cur_atr,
            })
            
    # Compile statistics
    if len(trades) == 0:
        return None
    tdf = pd.DataFrame(trades)
    wins = (tdf['pnl'] > 0).sum()
    losses = (tdf['pnl'] < 0).sum()
    ties = (tdf['pnl'] == 0).sum()
    n = len(tdf)
    wr = wins / n * 100.0
    gp = tdf[tdf['pnl'] > 0]['pnl'].sum()
    gl = abs(tdf[tdf['pnl'] < 0]['pnl'].sum())
    pf = gp / gl if gl > 0 else 999.0
    net_pnl = tdf['pnl'].sum()
    r_net = tdf['r_pnl'].sum()
    
    cum = tdf['pnl'].cumsum().values
    peak = np.maximum.accumulate(cum)
    dd = (peak - cum).max() / max(1.0, peak.max()) * 100.0 if peak.max() > 0 else 99.0
    
    tp_hits = (tdf['reason'] == 'TP').sum()
    be_hits = (tdf['reason'] == 'BE').sum()
    sl_hits = (tdf['reason'] == 'SL').sum()
    time_hits = (tdf['reason'] == 'TIME').sum()
    
    b_cnt = (tdf['dir'] == 1).sum()
    s_cnt = (tdf['dir'] == -1).sum()
    
    return {
        'wr': wr, 'pf': pf, 'n': n, 'tr_per_day': n / 7.0,
        'b_cnt': b_cnt, 's_cnt': s_cnt,
        'tp_hits': tp_hits, 'be_hits': be_hits, 'sl_hits': sl_hits, 'time_hits': time_hits,
        'dd': dd, 'net_pnl': net_pnl, 'r_net': r_net,
        'sl_atr': sl_atr, 'tp_atr': tp_atr, 'be_trig': be_trig_atr, 'be_lock': be_lock_atr,
        'cooldown': cooldown_bars, 'max_bars': max_bars_in_trade,
    }

print("Running parameter grid scan over Value Zone Solution...")
res_list = []
for sl_a in [1.5, 2.0, 2.5, 3.0]:
    for tp_a in [1.2, 1.5, 1.8, 2.0, 2.5]:
        for bet_a in [0.50, 0.60, 0.70]:
            for bel_a in [0.30, 0.40, 0.50]:
                if bel_a >= bet_a: continue
                for cd in [2, 3, 5]:
                    r = simulate(
                        sl_atr=sl_a,
                        tp_atr=tp_a,
                        be_trig_atr=bet_a,
                        be_lock_atr=bel_a,
                        cooldown_bars=cd,
                        min_spacing_bars=cd,
                    )
                    if r is not None and r['n'] >= 200:
                        res_list.append(r)

res_list.sort(key=lambda x: (x['wr'] >= 80.0, x['pf'] >= 3.0, x['pf']), reverse=True)

print(f"\nTop 25 Configurations:")
print(f"{'WR':<8} {'PF':<6} {'Trades':<8} {'Tr/d':<8} {'B/S':<12} {'TP/BE/SL/TI':<18} {'DD':<6} {'SL':<5} {'TP':<5} {'BE_t':<6} {'BE_l':<6} {'CD':<4}")
print("-" * 95)
for r in res_list[:25]:
    bs = f"{r['b_cnt']}/{r['s_cnt']}"
    hits = f"{r['tp_hits']}/{r['be_hits']}/{r['sl_hits']}/{r['time_hits']}"
    print(f"{r['wr']:5.2f}%   {r['pf']:5.2f}  {r['n']:<8} {r['tr_per_day']:<8.1f} {bs:<12} {hits:<18} {r['dd']:4.1f}% {r['sl_atr']:<5.1f} {r['tp_atr']:<5.1f} {r['be_trig']:<6.2f} {r['be_lock']:<6.2f} {r['cooldown']:<4}")
