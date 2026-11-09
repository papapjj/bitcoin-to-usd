import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd

# Load clean 7-day M1 data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)
vols = df['<TICKVOL>'].values.astype(np.float64)
N = len(closes)

# ATR 14
tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
tr = np.insert(tr, 0, highs[0] - lows[0])
atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

# Indicators
s_c = pd.Series(closes)
ema5 = s_c.ewm(span=5).mean().values
ema9 = s_c.ewm(span=9).mean().values
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values
ema200 = s_c.ewm(span=200).mean().values

# Fast momentum & MACD
macd_line = ema9 - ema20
macd_signal = pd.Series(macd_line).ewm(span=9).mean().values
macd_hist = macd_line - macd_signal

# Stochastic Fast %K & %D
s_h = pd.Series(highs)
s_l = pd.Series(lows)
low14 = s_l.rolling(14, min_periods=1).min().values
high14 = s_h.rolling(14, min_periods=1).max().values
rng14 = np.maximum(1.0, high14 - low14)
stoch_k = (closes - low14) / rng14 * 100.0
stoch_d = pd.Series(stoch_k).rolling(3, min_periods=1).mean().values

# 60-bar range position
rolling_h60 = s_h.rolling(60, min_periods=1).max().values
rolling_l60 = s_l.rolling(60, min_periods=1).min().values
rng60 = np.maximum(1.0, rolling_h60 - rolling_l60)
rel_pos_60 = (closes - rolling_l60) / rng60

# Higher Timeframe H1 trend (60-bar rolling EMA)
h1_ema21 = s_c.ewm(span=21*60).mean().values
h1_ema50 = s_c.ewm(span=50*60).mean().values
h1_bull = h1_ema21 > h1_ema50
h1_bear = h1_ema21 < h1_ema50

# Candle features
body = np.abs(closes - opens)
upper_wick = highs - np.maximum(closes, opens)
lower_wick = np.minimum(closes, opens) - lows
bar_range = highs - lows

REAL_SPREAD = 28.98 # Exness MT5 BTCUSDm spread

print(f"Total M1 bars: {N}. Mean ATR: {np.mean(atr):.2f} USD.")

# Fast simulation engine
def eval_sniper_signals(buy_signals, sell_signals, sl_atr, tp_atr, be_trig_atr=None, be_lock_atr=0.0, max_hold=60, min_spacing=2):
    trades = []
    last_exit = -999
    last_entry = -999
    
    b_wins, b_losses = 0, 0
    s_wins, s_losses = 0, 0
    
    consec_losses = 0
    pause_until = 0
    
    for t in range(200, N - max_hold):
        if t < pause_until: continue
        if t - last_entry < min_spacing: continue
        if t - last_exit < 1: continue
        
        is_buy = buy_signals[t]
        is_sell = sell_signals[t]
        if not is_buy and not is_sell: continue
        if is_buy and is_sell: continue
        
        cur_atr = max(atr[t], 10.0)
        cp = closes[t]
        sl_dist = sl_atr * cur_atr
        tp_dist = tp_atr * cur_atr
        bet_dist = be_trig_atr * cur_atr if be_trig_atr is not None else 999999.0
        bel_dist = be_lock_atr * cur_atr
        
        pnl = 0.0
        
        if is_buy:
            entry_p = cp + REAL_SPREAD * 0.5
            cur_sl = cp - sl_dist
            tp_p = cp + tp_dist
            best_bid = cp
            
            for k in range(t + 1, min(t + max_hold, N)):
                k_high = highs[k]
                k_low = lows[k]
                best_bid = max(best_bid, k_high)
                
                if best_bid >= cp + bet_dist:
                    cur_sl = max(cur_sl, cp + bel_dist)
                    
                if k_low <= cur_sl:
                    pnl = cur_sl - entry_p
                    last_exit = k
                    break
                elif k_high >= tp_p:
                    pnl = tp_p - entry_p
                    last_exit = k
                    break
            else:
                pnl = closes[min(t + max_hold - 1, N - 1)] - entry_p
                last_exit = min(t + max_hold - 1, N - 1)
                
            if pnl > 0: b_wins += 1
            else: b_losses += 1
            
        else: # SELL
            entry_p = cp - REAL_SPREAD * 0.5
            cur_sl = cp + sl_dist
            tp_p = cp - tp_dist
            best_ask = cp
            
            for k in range(t + 1, min(t + max_hold, N)):
                k_high = highs[k] + REAL_SPREAD
                k_low = lows[k] + REAL_SPREAD
                best_ask = min(best_ask, k_low)
                
                if best_ask <= cp - bet_dist:
                    cur_sl = min(cur_sl, cp - bel_dist)
                    
                if k_high >= cur_sl:
                    pnl = entry_p - cur_sl
                    last_exit = k
                    break
                elif k_low <= tp_p:
                    pnl = entry_p - tp_p
                    last_exit = k
                    break
            else:
                pnl = entry_p - (closes[min(t + max_hold - 1, N - 1)] + REAL_SPREAD)
                last_exit = min(t + max_hold - 1, N - 1)
                
            if pnl > 0: s_wins += 1
            else: s_losses += 1
            
        last_entry = t
        trades.append(pnl)
        
        if pnl <= 0:
            consec_losses += 1
            if consec_losses >= 2:
                pause_until = last_exit + 30 # 30-minute circuit breaker
                consec_losses = 0
        else:
            consec_losses = 0
            
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
    tr_per_day = n / 7.0
    
    return {
        'n': n, 'tr_per_day': tr_per_day, 'wr': wr, 'pf': pf,
        'b_wins': b_wins, 'b_losses': b_losses, 's_wins': s_wins, 's_losses': s_losses,
        'dd': dd, 'pnl': tot_pnl
    }

# Search over signal combinations
print("\nSystematic Search for High Win Rate & Profit Factor...")

results = []

# Define Candidate Signal Generators
for trend_filter in ['h1', 'ema20_50', 'none']:
    for entry_type in ['pullback_wick', 'stoch_extreme', 'momentum_retest', 'bollinger_bounce']:
        buy_sig = np.zeros(N, dtype=bool)
        sell_sig = np.zeros(N, dtype=bool)
        
        for t in range(200, N):
            cur_atr = max(atr[t], 10.0)
            
            # 1. Trend Filter
            t_bull, t_bear = True, True
            if trend_filter == 'h1':
                t_bull = h1_bull[t]
                t_bear = h1_bear[t]
            elif trend_filter == 'ema20_50':
                t_bull = ema20[t] > ema50[t] and closes[t] > ema50[t]
                t_bear = ema20[t] < ema50[t] and closes[t] < ema50[t]
                
            # 2. Entry Types
            if entry_type == 'pullback_wick':
                # BUY: Low touched EMA9/EMA20, rejected with lower wick, closed green
                buy_cond = t_bull and (lows[t] <= ema20[t] + 0.3 * cur_atr) and (lower_wick[t] >= 0.25 * cur_atr) and (closes[t] >= opens[t]) and (rel_pos_60[t] < 0.85)
                sell_cond = t_bear and (highs[t] >= ema20[t] - 0.3 * cur_atr) and (upper_wick[t] >= 0.25 * cur_atr) and (closes[t] <= opens[t]) and (rel_pos_60[t] > 0.15)
            elif entry_type == 'stoch_extreme':
                # BUY: Stoch pulled back to oversold < 25, turning up, in trend
                buy_cond = t_bull and (stoch_k[t] < 30) and (stoch_k[t] > stoch_d[t]) and (closes[t] > opens[t]) and (rel_pos_60[t] < 0.80)
                sell_cond = t_bear and (stoch_k[t] > 70) and (stoch_k[t] < stoch_d[t]) and (closes[t] < opens[t]) and (rel_pos_60[t] > 0.20)
            elif entry_type == 'momentum_retest':
                # BUY: MACD Hist > 0, price pulls back to EMA9, green candle
                buy_cond = t_bull and (macd_hist[t] > 0) and (lows[t] <= ema9[t] + 0.2 * cur_atr) and (closes[t] > opens[t]) and (rel_pos_60[t] < 0.80)
                sell_cond = t_bear and (macd_hist[t] < 0) and (highs[t] >= ema9[t] - 0.2 * cur_atr) and (closes[t] < opens[t]) and (rel_pos_60[t] > 0.20)
            elif entry_type == 'bollinger_bounce':
                # 20-bar Bollinger Bands
                mid = ema20[t]
                dev = pd.Series(closes[max(0, t-20):t]).std() if t >= 20 else 20.0
                upper = mid + 2.0 * dev
                lower = mid - 2.0 * dev
                buy_cond = t_bull and (lows[t] <= lower + 0.3 * cur_atr) and (closes[t] > opens[t]) and (rel_pos_60[t] < 0.80)
                sell_cond = t_bear and (highs[t] >= upper - 0.3 * cur_atr) and (closes[t] < opens[t]) and (rel_pos_60[t] > 0.20)
                
            buy_sig[t] = buy_cond
            sell_sig[t] = sell_cond
            
        # Test across geometries
        for sl in [3.0, 4.0, 5.0, 6.0]:
            for tp in [1.5, 2.0, 2.5, 3.0, 4.0]:
                for be, lock in [(None, 0.0), (1.5, 1.0), (2.0, 1.2)]:
                    res = eval_sniper_signals(buy_sig, sell_sig, sl, tp, be, lock)
                    if res and res['n'] >= 50 and res['pf'] >= 1.50:
                        results.append({
                            'trend': trend_filter, 'entry': entry_type,
                            'sl': sl, 'tp': tp, 'be': f"{be}/{lock}" if be else "None",
                            **res
                        })

df_res = pd.DataFrame(results)
if len(df_res) > 0:
    df_res = df_res.sort_values(by=['pf', 'wr'], ascending=False)
    print(f"\nFound {len(df_res)} configurations with PF >= 1.50 and N >= 50:")
    print(f"{'Trend':<10} {'Entry':<18} {'SL':<4} {'TP':<4} {'BE':<10} | {'WR':<7} {'PF':<6} {'Trades':<8} {'Tr/d':<6} {'B_W/L':<10} {'S_W/L':<10} {'DD':<6} {'Net PnL ($)'}")
    print("-" * 115)
    for _, r in df_res.head(30).iterrows():
        b_wl = f"{r['b_wins']}/{r['b_losses']}"
        s_wl = f"{r['s_wins']}/{r['s_losses']}"
        print(f"{r['trend']:<10} {r['entry']:<18} {r['sl']:<4.1f} {r['tp']:<4.1f} {r['be']:<10} | {r['wr']:5.1f}%  {r['pf']:5.2f}  {int(r['n']):<8} {r['tr_per_day']:<6.1f} {b_wl:<10} {s_wl:<10} {r['dd']:4.1f}% ${r['pnl']:+,.0f}")
else:
    print("No configurations met the threshold. Checking best candidates...")
