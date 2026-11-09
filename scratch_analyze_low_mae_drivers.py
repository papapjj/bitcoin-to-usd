import numpy as np
import pandas as pd
from pathlib import Path

df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
print(f"Total bars: {len(df)}")

closes = df['<CLOSE>'].values.astype(np.float64)
highs = df['<HIGH>'].values.astype(np.float64)
lows = df['<LOW>'].values.astype(np.float64)
opens = df['<OPEN>'].values.astype(np.float64)
vols = df['<TICKVOL>'].values.astype(np.float64)

# ATR14
tr = np.maximum(highs[1:] - lows[1:], np.maximum(np.abs(highs[1:] - closes[:-1]), np.abs(lows[1:] - closes[:-1])))
atr = np.zeros(len(df))
atr[:14] = np.mean(tr[:14])
for i in range(14, len(df)):
    atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14.0

# Calculate EMA20 and EMA60
def calc_ema(arr, span):
    res = np.zeros_like(arr)
    alpha = 2.0 / (span + 1.0)
    res[0] = arr[0]
    for i in range(1, len(arr)):
        res[i] = alpha * arr[i] + (1 - alpha) * res[i-1]
    return res

ema20 = calc_ema(closes, 20)
ema60 = calc_ema(closes, 60)

horizon = 30
data = []

for i in range(60, len(df) - horizon):
    cur_p = closes[i]
    cur_atr = max(1.0, atr[i])
    
    # Forward excursions
    fut_h = highs[i+1 : i+horizon+1]
    fut_l = lows[i+1 : i+horizon+1]
    
    b_mae = (cur_p - np.min(fut_l)) / cur_atr
    b_mfe = (np.max(fut_h) - cur_p) / cur_atr
    
    s_mae = (np.max(fut_h) - cur_p) / cur_atr
    s_mfe = (cur_p - np.min(fut_l)) / cur_atr
    
    # Features at bar i:
    # 1. Trend alignment: price relative to EMA20 and EMA60
    p_vs_e20 = (cur_p - ema20[i]) / cur_atr
    e20_vs_e60 = (ema20[i] - ema60[i]) / cur_atr
    
    # 2. Candle geometry: upper wick, lower wick, body
    body = (closes[i] - opens[i]) / cur_atr
    upper_wick = (highs[i] - max(opens[i], closes[i])) / cur_atr
    lower_wick = (min(opens[i], closes[i]) - lows[i]) / cur_atr
    
    # 3. Position in 20-bar range
    h20 = np.max(highs[i-19 : i+1])
    l20 = np.min(lows[i-19 : i+1])
    rng20 = max(1.0, h20 - l20)
    pos20 = (cur_p - l20) / rng20
    
    # 4. Momentum / Slope
    slope5 = (closes[i] - closes[i-5]) / (5 * cur_atr)
    
    data.append({
        'b_mae': b_mae,
        'b_mfe': b_mfe,
        's_mae': s_mae,
        's_mfe': s_mfe,
        'p_vs_e20': p_vs_e20,
        'e20_vs_e60': e20_vs_e60,
        'body': body,
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'pos20': pos20,
        'slope5': slope5,
    })

df_feat = pd.DataFrame(data)

# Analyze BUY: What conditions produce b_mae <= 0.6 AND b_mfe >= 1.8?
ideal_buys = df_feat[(df_feat['b_mae'] <= 0.6) & (df_feat['b_mfe'] >= 1.8)]
ideal_sells = df_feat[(df_feat['s_mae'] <= 0.6) & (df_feat['s_mfe'] >= 1.8)]

print(f"Total bars analyzed: {len(df_feat)}")
print(f"Ideal Low-MAE BUYs (MAE<=0.6, MFE>=1.8):  {len(ideal_buys)} ({len(ideal_buys)/len(df_feat)*100:.1f}%)")
print(f"Ideal Low-MAE SELLs (MAE<=0.6, MFE>=1.8): {len(ideal_sells)} ({len(ideal_sells)/len(df_feat)*100:.1f}%)")

print("\n--- Features for Ideal Low-MAE BUYs vs All ---")
print(f"{'Feature':<15} {'Ideal BUY Mean':<16} {'Overall Mean':<16}")
for col in ['p_vs_e20', 'e20_vs_e60', 'body', 'lower_wick', 'pos20', 'slope5']:
    print(f"{col:<15} {ideal_buys[col].mean():<16.3f} {df_feat[col].mean():<16.3f}")

print("\n--- Features for Ideal Low-MAE SELLs vs All ---")
print(f"{'Feature':<15} {'Ideal SELL Mean':<16} {'Overall Mean':<16}")
for col in ['p_vs_e20', 'e20_vs_e60', 'body', 'upper_wick', 'pos20', 'slope5']:
    print(f"{col:<15} {ideal_sells[col].mean():<16.3f} {df_feat[col].mean():<16.3f}")
