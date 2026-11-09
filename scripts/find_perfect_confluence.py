import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "src"))
import numpy as np
import pandas as pd

from self_growing_trader.config import FeatureConfig, SymbolConfig
from self_growing_trader.features import add_features

# Ingest Data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df_renamed = df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<TICKVOL>': 'Volume', '<VOL>': 'RealVolume', '<SPREAD>': 'Spread'
})
if '<DATE>' in df_renamed.columns and '<TIME>' in df_renamed.columns:
    df_renamed['Datetime'] = df_renamed['<DATE>'] + ' ' + df_renamed['<TIME>']

feat_df = add_features(df_renamed, FeatureConfig(), SymbolConfig())
N = len(feat_df)

closes = feat_df['Close'].values.astype(np.float64)
highs = feat_df['High'].values.astype(np.float64)
lows = feat_df['Low'].values.astype(np.float64)
opens = feat_df['Open'].values.astype(np.float64)
atrs = feat_df['atr'].values.astype(np.float64)

s_c = pd.Series(closes)
ema9 = s_c.ewm(span=9).mean().values
ema21 = s_c.ewm(span=21).mean().values
ema50 = s_c.ewm(span=50).mean().values

s_h = pd.Series(highs)
s_l = pd.Series(lows)
rolling_h30 = s_h.rolling(30, min_periods=1).max().values
rolling_l30 = s_l.rolling(30, min_periods=1).min().values

# Let's check ground truth outcomes for:
# BUY: does price reach +tp_atr before -sl_atr within horizon bars?
# SELL: does price reach -tp_atr before +sl_atr within horizon bars?

horizon = 30
sl_atr = 1.2
tp_atr = 1.0

buy_edge = np.zeros(N, dtype=bool)
sell_edge = np.zeros(N, dtype=bool)

for t in range(50, N - horizon):
    cp = closes[t]
    atr = max(atrs[t], 1.0)
    
    # Forward outcomes
    fwd_h = highs[t+1 : t+1+horizon]
    fwd_l = lows[t+1 : t+1+horizon]
    
    # BUY check
    b_ok = False
    for k in range(len(fwd_h)):
        if (cp - fwd_l[k]) >= sl_atr * atr: break
        if (fwd_h[k] - cp) >= tp_atr * atr: b_ok = True; break
    buy_edge[t] = b_ok
    
    # SELL check
    s_ok = False
    for k in range(len(fwd_l)):
        if (fwd_h[k] - cp) >= sl_atr * atr: break
        if (cp - fwd_l[k]) >= tp_atr * atr: s_ok = True; break
    sell_edge[t] = s_ok

print(f"Total bars: {N}")
print(f"Bars with BUY Edge (TP 1.0 before SL 1.2): {np.sum(buy_edge)} ({np.sum(buy_edge)/N*100:.1f}%)")
print(f"Bars with SELL Edge (TP 1.0 before SL 1.2): {np.sum(sell_edge)} ({np.sum(sell_edge)/N*100:.1f}%)")

# Now let's test confluences to find filters with WR >= 80% and >= 100 trades/day (700+ trades)
# Factors:
# 1. EMA alignment (EMA9 vs EMA21, EMA21 vs EMA50)
# 2. Pullback to EMA9 / EMA21
# 3. Candlestick wick / momentum confirmation
# 4. Multi-timeframe trend

results = []
for p_low in [0.10, 0.20, 0.30, 0.40, 0.50]:
    for wick_min in [0.0, 0.10, 0.20]:
        for sl_val, tp_val in [(1.0, 1.0), (1.2, 1.0), (1.5, 1.2), (1.5, 1.5)]:
            # Test sequential simulation
            trades = []
            last_entry = -999
            last_exit = -999
            b_cnt, s_cnt = 0, 0
            
            for t in range(50, N - horizon):
                if t - last_entry < 2: continue
                if t - last_exit < 2: continue
                
                cp = closes[t]
                atr = max(atrs[t], 1.0)
                e9 = ema9[t]
                e21 = ema21[t]
                e50 = ema50[t]
                
                act = 0
                # BUY: Trend UP, Pullback near EMA21, Candle rejection / green
                if e9 > e21 and cp > e50:
                    in_pullback = (lows[t] <= e21 + p_low * atr) and (cp <= e21 + 0.60 * atr)
                    trig = (closes[t] >= opens[t]) or ((closes[t] - lows[t]) >= wick_min * atr)
                    if in_pullback and trig:
                        act = 1
                # SELL: Trend DOWN, Pullback near EMA21, Candle rejection / red
                elif e9 < e21 and cp < e50:
                    in_pullback = (highs[t] >= e21 - p_low * atr) and (cp >= e21 - 0.60 * atr)
                    trig = (closes[t] <= opens[t]) or ((highs[t] - closes[t]) >= wick_min * atr)
                    if in_pullback and trig:
                        act = -1
                        
                if act == 0: continue
                last_entry = t
                if act == 1: b_cnt += 1
                else: s_cnt += 1
                
                # Check execution outcome
                fwd_h = highs[t+1 : t+1+horizon]
                fwd_l = lows[t+1 : t+1+horizon]
                sl_d = sl_val * atr
                tp_d = tp_val * atr
                
                pnl = 0.0
                if act == 1:
                    cur_sl = cp - sl_d
                    tp_p = cp + tp_d
                    for k in range(len(fwd_h)):
                        if fwd_l[k] <= cur_sl:
                            pnl = -sl_d
                            last_exit = t + 1 + k
                            break
                        elif fwd_h[k] >= tp_p:
                            pnl = tp_d
                            last_exit = t + 1 + k
                            break
                    else:
                        pnl = closes[t+horizon] - cp
                        last_exit = t + horizon
                else:
                    cur_sl = cp + sl_d
                    tp_p = cp - tp_d
                    for k in range(len(fwd_l)):
                        if fwd_h[k] >= cur_sl:
                            pnl = -sl_d
                            last_exit = t + 1 + k
                            break
                        elif fwd_l[k] <= tp_p:
                            pnl = tp_d
                            last_exit = t + 1 + k
                            break
                    else:
                        pnl = cp - closes[t+horizon]
                        last_exit = t + horizon
                        
                trades.append(pnl)
                
            tr = np.array(trades)
            if len(tr) < 100: continue
            wins = np.sum(tr > 0)
            wr = wins / len(tr) * 100.0
            gp = np.sum(tr[tr > 0])
            gl = abs(np.sum(tr[tr < 0]))
            pf = gp / gl if gl > 0 else 999.0
            cum = np.cumsum(tr)
            dd = (np.maximum.accumulate(cum) - cum).max() / max(1.0, cum.max()) * 100.0 if cum.max() > 0 else 99.0
            
            results.append({
                'wr': wr, 'pf': pf, 'n': len(tr), 'b': b_cnt, 's': s_cnt,
                'dd': dd, 'sl': sl_val, 'tp': tp_val, 'pb': p_low, 'wick': wick_min
            })

rdf = pd.DataFrame(results)
print("\nTOP CONFIGS SORTED BY PROFIT FACTOR:")
top_pf = rdf.sort_values(by='pf', ascending=False).head(15)
for _, r in top_pf.iterrows():
    print(f"WR: {r['wr']:5.2f}% | PF: {r['pf']:5.2f} | N: {int(r['n']):<4} ({r['n']/7.0:.1f}/d) | B/S: {int(r['b'])}/{int(r['s'])} | DD: {r['dd']:4.1f}% | SL: {r['sl']} | TP: {r['tp']} | PB: {r['pb']} | Wick: {r['wick']}")

print("\nTOP CONFIGS SORTED BY WIN RATE:")
top_wr = rdf.sort_values(by='wr', ascending=False).head(15)
for _, r in top_wr.iterrows():
    print(f"WR: {r['wr']:5.2f}% | PF: {r['pf']:5.2f} | N: {int(r['n']):<4} ({r['n']/7.0:.1f}/d) | B/S: {int(r['b'])}/{int(r['s'])} | DD: {r['dd']:4.1f}% | SL: {r['sl']} | TP: {r['tp']} | PB: {r['pb']} | Wick: {r['wick']}")
