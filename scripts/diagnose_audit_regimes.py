import pandas as pd
import numpy as np

df = pd.read_csv(r'.mt5_runners\visual_7day_champion_6488_20261031_162132_907165\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_feature_audit.csv', sep='\t')
print("Total rows in audit:", len(df))

uptrend_count = 0
downtrend_count = 0
choppy_count = 0

for i in range(len(df)):
    inp_str = df['inputs'].iloc[i]
    vals = [float(x) for x in inp_str.split('|') if x.strip()]
    if len(vals) < 28: continue
    
    ema_diff = vals[11]
    p_vs_e20 = vals[26]
    p_vs_e50 = vals[27]
    rel_pos_60 = vals[13]
    body = vals[7]
    uw = vals[8]
    lw = vals[9]
    
    if ema_diff > 0.0 and p_vs_e50 > 0.0:
        uptrend_count += 1
    elif ema_diff < 0.0 and p_vs_e50 < 0.0:
        downtrend_count += 1
    else:
        choppy_count += 1

print(f"Uptrend bars: {uptrend_count} | Downtrend bars: {downtrend_count} | Choppy bars: {choppy_count}")
