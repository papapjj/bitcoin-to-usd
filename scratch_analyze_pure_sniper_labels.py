import numpy as np
import pandas as pd

df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
print(f"Total bars: {len(df)}")

closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)

# ATR14
tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
atr = np.zeros(len(df))
atr[:14] = np.mean(tr[:14])
for i in range(14, len(df)):
    atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14.0

horizon = 30
target_mfe_atr = 1.8
max_mae_atr = 1.2

buy_snipers = []
sell_snipers = []
holds = []

for i in range(len(df) - horizon):
    c0 = closes[i]
    a0 = atr[i]
    
    fut_h = highs[i+1 : i+horizon+1]
    fut_l = lows[i+1 : i+horizon+1]
    
    # Check BUY
    # Condition: price hits c0 + target_mfe_atr * a0 BEFORE hitting c0 - max_mae_atr * a0
    buy_sl = c0 - max_mae_atr * a0
    buy_tp = c0 + target_mfe_atr * a0
    
    buy_success = False
    for bar_h, bar_l in zip(fut_h, fut_l):
        if bar_l <= buy_sl:
            break
        if bar_h >= buy_tp:
            buy_success = True
            break
            
    # Check SELL
    sell_sl = c0 + max_mae_atr * a0
    sell_tp = c0 - target_mfe_atr * a0
    
    sell_success = False
    for bar_h, bar_l in zip(fut_h, fut_l):
        if bar_h >= sell_sl:
            break
        if bar_l <= sell_tp:
            sell_success = True
            break
            
    if buy_success and not sell_success:
        buy_snipers.append(i)
    elif sell_success and not buy_success:
        sell_snipers.append(i)
    else:
        holds.append(i)

n_buy = len(buy_snipers)
n_sell = len(sell_snipers)
n_hold = len(holds)
total = n_buy + n_sell + n_hold

print(f"Target: TP={target_mfe_atr} ATR, SL={max_mae_atr} ATR (RR = {target_mfe_atr/max_mae_atr:.2f}):")
print(f"BUY Snipers:  {n_buy:<5} ({n_buy/total*100:.1f}%)")
print(f"SELL Snipers: {n_sell:<5} ({n_sell/total*100:.1f}%)")
print(f"HOLD / Chop:  {n_hold:<5} ({n_hold/total*100:.1f}%)")
print(f"Total high-probability setups in 7 days: {n_buy + n_sell} (average {(n_buy + n_sell)/7:.1f} trades/day)")
