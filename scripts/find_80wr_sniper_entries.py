import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd

df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)
vols = df['<TICKVOL>'].values.astype(np.float64)
N = len(closes)

tr = np.maximum(highs[1:] - lows[1:], np.maximum(abs(highs[1:] - closes[:-1]), abs(lows[1:] - closes[:-1])))
tr = np.insert(tr, 0, highs[0] - lows[0])
atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

REAL_SPREAD = 28.98 # Exness MT5 BTCUSDm spread

# Forward Excursion Analysis across all bars
# For each bar t:
# If we enter BUY at close + SPREAD/2:
# What is the Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) over the next 15, 30, 60 bars?
# Does MFE reach +1.0 ATR before MAE drops -1.0, -1.5, -2.0, -3.0 ATR?

print("Analyzing Forward Excursions on 11,513 M1 Bars...")

for horizon in [15, 30, 60]:
    buy_mae_1_0 = []
    buy_mfe_reach_1_0 = []
    
    for t in range(50, N - horizon):
        cur_atr = max(atr[t], 10.0)
        cp = closes[t]
        entry_buy = cp + REAL_SPREAD * 0.5
        entry_sell = cp - REAL_SPREAD * 0.5
        
        fwd_h = highs[t+1 : t+1+horizon]
        fwd_l = lows[t+1 : t+1+horizon]
        
        # BUY: favorable is fwd_h - entry_buy, adverse is entry_buy - fwd_l
        buy_mfe = np.max(fwd_h) - entry_buy
        buy_mae = entry_buy - np.min(fwd_l)
        
        buy_mae_1_0.append(buy_mae / cur_atr)
        buy_mfe_reach_1_0.append(buy_mfe / cur_atr)
        
    print(f"Horizon {horizon}m: Mean BUY MAE = {np.mean(buy_mae_1_0):.2f} ATR (${np.mean(buy_mae_1_0)*np.mean(atr):.1f} USD) | Mean MFE = {np.mean(buy_mfe_reach_1_0):.2f} ATR")

# Now let's test specific high-probability patterns:
# Pattern A: Dip below EMA20 during Bull Trend, candle closes with rejection wick (Pin bar / Hammer)
# Pattern B: 3 consecutive down candles in Bull Trend, 4th candle turns green
# Pattern C: False breakout / Liquidity sweep: price sweeps the 10-bar low and instantly reclaims it
print("\nTesting Candlestick & Liquidity Sweep Patterns...")

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

s_h = pd.Series(highs)
s_l = pd.Series(lows)
low10 = s_l.rolling(10, min_periods=1).min().shift(1).values
high10 = s_h.rolling(10, min_periods=1).max().shift(1).values

lower_wick = np.minimum(closes, opens) - lows
upper_wick = highs - np.maximum(closes, opens)

# Pattern C: Liquidity Sweep Reclaim
# Low touched below 10-bar low, but Close finished ABOVE 10-bar low, with green body or strong lower wick
sweep_buy = (lows < low10) & (closes > low10) & ((closes >= opens) | (lower_wick >= 0.25 * atr))
sweep_sell = (highs > high10) & (closes < high10) & ((closes <= opens) | (upper_wick >= 0.25 * atr))

print(f"Sweep BUY events: {sweep_buy.sum()} | Sweep SELL events: {sweep_sell.sum()}")

def test_pattern_performance(buy_mask, sell_mask, sl_atr=3.0, tp_atr=2.0, max_hold=30):
    trades = []
    last_exit = -999
    last_entry = -999
    b_w, b_l, s_w, s_l = 0, 0, 0, 0
    
    for t in range(50, N - max_hold):
        if t - last_entry < 2: continue
        if t - last_exit < 1: continue
        
        is_b = buy_mask[t]
        is_s = sell_mask[t]
        if not is_b and not is_s: continue
        if is_b and is_s: continue
        
        cur_atr = max(atr[t], 10.0)
        cp = closes[t]
        sl_dist = sl_atr * cur_atr
        tp_dist = tp_atr * cur_atr
        
        pnl = 0.0
        if is_b:
            entry_p = cp + REAL_SPREAD * 0.5
            sl_p = cp - sl_dist
            tp_p = cp + tp_dist
            for k in range(t+1, min(t+max_hold, N)):
                if lows[k] <= sl_p:
                    pnl = sl_p - entry_p
                    last_exit = k
                    break
                elif highs[k] >= tp_p:
                    pnl = tp_p - entry_p
                    last_exit = k
                    break
            else:
                pnl = closes[min(t+max_hold-1, N-1)] - entry_p
                last_exit = min(t+max_hold-1, N-1)
            if pnl > 0: b_w += 1
            else: b_l += 1
        else:
            entry_p = cp - REAL_SPREAD * 0.5
            sl_p = cp + sl_dist
            tp_p = cp - tp_dist
            for k in range(t+1, min(t+max_hold, N)):
                if highs[k] + REAL_SPREAD >= sl_p:
                    pnl = entry_p - sl_p
                    last_exit = k
                    break
                elif lows[k] + REAL_SPREAD <= tp_p:
                    pnl = entry_p - tp_p
                    last_exit = k
                    break
            else:
                pnl = entry_p - (closes[min(t+max_hold-1, N-1)] + REAL_SPREAD)
                last_exit = min(t+max_hold-1, N-1)
            if pnl > 0: s_w += 1
            else: s_l += 1
            
        last_entry = t
        trades.append(pnl)
        
    tr = np.array(trades)
    if len(tr) == 0: return
    n = len(tr)
    wr = np.sum(tr > 0) / n * 100.0
    gp = np.sum(tr[tr > 0])
    gl = abs(np.sum(tr[tr < 0]))
    pf = gp / gl if gl > 0 else 999.0
    tot_pnl = np.sum(tr)
    print(f"SL={sl_atr} TP={tp_atr} | WR: {wr:5.1f}% | PF: {pf:5.2f} | N: {n:4d} ({n/7.0:4.1f}/d) | B: {b_w}/{b_l} | S: {s_w}/{s_l} | Net: ${tot_pnl:+,.0f}")

print("\n--- Testing Liquidity Sweep Pattern (All bars) ---")
for sl in [2.0, 3.0, 4.0, 5.0]:
    for tp in [1.5, 2.0, 3.0]:
        test_pattern_performance(sweep_buy, sweep_sell, sl, tp)

print("\n--- Testing Liquidity Sweep Pattern with Trend Filter (EMA20 vs EMA50) ---")
trend_sweep_buy = sweep_buy & (ema20 > ema50) & (closes > ema50)
trend_sweep_sell = sweep_sell & (ema20 < ema50) & (closes < ema50)
for sl in [2.0, 3.0, 4.0, 5.0]:
    for tp in [1.5, 2.0, 3.0]:
        test_pattern_performance(trend_sweep_buy, trend_sweep_sell, sl, tp)
