import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd

# Load 7-day clean M1 data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)
vols = df['<TICKVOL>'].values.astype(np.float64)

# Calculate ATR
tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
tr = np.insert(tr, 0, highs[0] - lows[0])
atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

# Indicators
s_c = pd.Series(closes)
ema9 = s_c.ewm(span=9).mean().values
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values
ema200 = s_c.ewm(span=200).mean().values

# 60-bar range
s_h = pd.Series(highs)
s_l = pd.Series(lows)
rolling_h60 = s_h.rolling(60, min_periods=1).max().values
rolling_l60 = s_l.rolling(60, min_periods=1).min().values
rng60 = np.maximum(1.0, rolling_h60 - rolling_l60)
rel_pos_60 = (closes - rolling_l60) / rng60

# 5-bar range
rolling_h5 = s_h.rolling(5, min_periods=1).max().values
rolling_l5 = s_l.rolling(5, min_periods=1).min().values

# RSI 14
delta = s_c.diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-9)
rsi14 = 100 - (100 / (1 + rs)).values

REAL_SPREAD = 28.98 # Exness MT5 BTCUSDm spread
N = len(closes)

print(f"Data points: {N} M1 bars. Mean ATR: {np.mean(atr):.2f} USD. Spread: {REAL_SPREAD} USD.")

def test_strategy(name, entry_fn, sl_atr, tp_atr, be_trig_atr=None, be_lock_atr=0.0, max_bars=60):
    trades = []
    last_exit = -999
    last_entry = -999
    
    b_wins, b_losses = 0, 0
    s_wins, s_losses = 0, 0
    
    for t in range(200, N - max_bars):
        if t - last_entry < 2: continue
        if t - last_exit < 2: continue
        
        sig = entry_fn(t)
        if sig == 0: continue
        
        cur_atr = max(atr[t], 10.0)
        cp = closes[t]
        sl_dist = sl_atr * cur_atr
        tp_dist = tp_atr * cur_atr
        bet_dist = be_trig_atr * cur_atr if be_trig_atr is not None else 999999.0
        bel_dist = be_lock_atr * cur_atr
        
        pnl = 0.0
        reason = None
        
        if sig == 1: # BUY
            entry_p = cp + REAL_SPREAD * 0.5
            cur_sl = cp - sl_dist
            tp_p = cp + tp_dist
            best_bid = cp
            
            for k in range(t + 1, min(t + max_bars, N)):
                k_high = highs[k]
                k_low = lows[k]
                best_bid = max(best_bid, k_high)
                
                if best_bid >= cp + bet_dist:
                    cur_sl = max(cur_sl, cp + bel_dist)
                    
                if k_low <= cur_sl:
                    pnl = cur_sl - entry_p
                    reason = 'SL'
                    last_exit = k
                    break
                elif k_high >= tp_p:
                    pnl = tp_p - entry_p
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = closes[min(t + max_bars - 1, N - 1)] - entry_p
                reason = 'TIME'
                last_exit = min(t + max_bars - 1, N - 1)
                
            if pnl > 0: b_wins += 1
            else: b_losses += 1
            
        elif sig == -1: # SELL
            entry_p = cp - REAL_SPREAD * 0.5
            cur_sl = cp + sl_dist
            tp_p = cp - tp_dist
            best_ask = cp
            
            for k in range(t + 1, min(t + max_bars, N)):
                k_high = highs[k] + REAL_SPREAD
                k_low = lows[k] + REAL_SPREAD
                best_ask = min(best_ask, k_low)
                
                if best_ask <= cp - bet_dist:
                    cur_sl = min(cur_sl, cp - bel_dist)
                    
                if k_high >= cur_sl:
                    pnl = entry_p - cur_sl
                    reason = 'SL'
                    last_exit = k
                    break
                elif k_low <= tp_p:
                    pnl = entry_p - tp_p
                    reason = 'TP'
                    last_exit = k
                    break
            else:
                pnl = entry_p - (closes[min(t + max_bars - 1, N - 1)] + REAL_SPREAD)
                reason = 'TIME'
                last_exit = min(t + max_bars - 1, N - 1)
                
            if pnl > 0: s_wins += 1
            else: s_losses += 1
            
        last_entry = t
        trades.append(pnl)
        
    tr = np.array(trades)
    if len(tr) == 0: return
    n = len(tr)
    wr = np.sum(tr > 0) / n * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    cum = np.cumsum(tr)
    dd = (np.maximum.accumulate(cum) - cum).max() / max(1.0, cum.max()) * 100.0 if cum.max() > 0 else 99.0
    tot_pnl = np.sum(tr)
    tr_per_day = n / 7.0
    
    print(f"{name:<35} | WR: {wr:5.1f}% | PF: {pf:5.2f} | Trades: {n:4d} ({tr_per_day:5.1f}/d) | B: {b_wins}/{b_losses} | S: {s_wins}/{s_losses} | DD: {dd:4.1f}% | Net: ${tot_pnl:+,.0f}")

# Entry Filters to test:
def filter_trend_pullback_wick(t):
    # Trend: EMA20 vs EMA50
    # Value pullback: low touched EMA20
    # Rejection wick confirmation: closed back up
    cur_atr = atr[t]
    if ema20[t] > ema50[t] and closes[t] > ema50[t]: # Bull trend
        if lows[t] <= ema20[t] + 0.3 * cur_atr and closes[t] >= opens[t]:
            # Lower rejection wick
            lower_wick = closes[t] - lows[t]
            if lower_wick >= 0.25 * cur_atr and rel_pos_60[t] < 0.85:
                return 1
    elif ema20[t] < ema50[t] and closes[t] < ema50[t]: # Bear trend
        if highs[t] >= ema20[t] - 0.3 * cur_atr and closes[t] <= opens[t]:
            # Upper rejection wick
            upper_wick = highs[t] - closes[t]
            if upper_wick >= 0.25 * cur_atr and rel_pos_60[t] > 0.15:
                return -1
    return 0

def filter_micro_structure_retest(t):
    # Tests EMA9/EMA20 dynamic support/resistance with tight confirmation
    cur_atr = atr[t]
    if ema9[t] > ema20[t] and closes[t] > ema20[t]:
        # Pullback to EMA9 or EMA20
        pullback = (lows[t] <= ema9[t] + 0.2 * cur_atr) or (lows[t] <= ema20[t] + 0.2 * cur_atr)
        rejection = (closes[t] > opens[t]) and (closes[t] - lows[t] >= 0.20 * cur_atr)
        if pullback and rejection and rel_pos_60[t] <= 0.80:
            return 1
    elif ema9[t] < ema20[t] and closes[t] < ema20[t]:
        pullback = (highs[t] >= ema9[t] - 0.2 * cur_atr) or (highs[t] >= ema20[t] - 0.2 * cur_atr)
        rejection = (closes[t] < opens[t]) and (highs[t] - closes[t] >= 0.20 * cur_atr)
        if pullback and rejection and rel_pos_60[t] >= 0.20:
            return -1
    return 0

def filter_dynamic_channel_sniper(t):
    cur_atr = atr[t]
    # BUY: Trend is UP (EMA20 > EMA50), RSI pulled back below 50, now crossing up, candle is green
    if ema20[t] > ema50[t] and closes[t] > ema50[t]:
        if rsi14[t] < 55 and rsi14[t] > 35 and closes[t] > opens[t] and rel_pos_60[t] < 0.80:
            if lows[t] <= ema20[t] + 0.4 * cur_atr:
                return 1
    elif ema20[t] < ema50[t] and closes[t] < ema50[t]:
        if rsi14[t] > 45 and rsi14[t] < 65 and closes[t] < opens[t] and rel_pos_60[t] > 0.20:
            if highs[t] >= ema20[t] - 0.4 * cur_atr:
                return -1
    return 0

print("\n--- EXPERIMENT 1: Trend Pullback Wick ---")
for sl in [3.0, 4.0, 5.0, 6.0]:
    for tp in [1.5, 2.0, 3.0, 4.0]:
        for be, lock in [(None, 0.0), (1.5, 1.0), (2.0, 1.2), (2.5, 1.5)]:
            be_str = f"BE={be}/{lock}" if be else "No BE"
            test_strategy(f"SL={sl} TP={tp} {be_str}", filter_trend_pullback_wick, sl, tp, be, lock)

print("\n--- EXPERIMENT 2: Micro Structure Retest ---")
for sl in [3.0, 4.0, 5.0]:
    for tp in [1.5, 2.0, 3.0]:
        for be, lock in [(None, 0.0), (1.5, 1.0), (2.0, 1.2)]:
            be_str = f"BE={be}/{lock}" if be else "No BE"
            test_strategy(f"SL={sl} TP={tp} {be_str}", filter_micro_structure_retest, sl, tp, be, lock)

print("\n--- EXPERIMENT 3: Channel Sniper ---")
for sl in [3.0, 4.0, 5.0]:
    for tp in [1.5, 2.0, 3.0]:
        for be, lock in [(None, 0.0), (1.5, 1.0), (2.0, 1.2)]:
            be_str = f"BE={be}/{lock}" if be else "No BE"
            test_strategy(f"SL={sl} TP={tp} {be_str}", filter_dynamic_channel_sniper, sl, tp, be, lock)
