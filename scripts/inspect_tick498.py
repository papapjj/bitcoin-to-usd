import pandas as pd
import numpy as np

# Load CSV
csv_path = r'.mt5_runners\visual_7day_champion_1944_20261031_152517_184180\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv'
df = pd.read_csv(csv_path, sep='\t')

# Let's inspect the clean data around 2025.02.20 18:46:00
data_df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
data_df = data_df.rename(columns={
    '<OPEN>': 'Open', '<HIGH>': 'High', '<LOW>': 'Low', '<CLOSE>': 'Close',
    '<DATE>': 'Date', '<TIME>': 'Time'
})
data_df['Datetime'] = data_df['Date'] + ' ' + data_df['Time']

# Find bar around 2025.02.20 18:46:00
idx = data_df[data_df['Datetime'].str.contains('2025.02.20 18:46')].index
if len(idx) > 0:
    t = idx[0]
    print(f"Bar index: {t}")
    sub = data_df.iloc[t-60:t+5]
    print(sub[['Datetime', 'Open', 'High', 'Low', 'Close']].tail(10))
    
    h60 = sub['High'].iloc[:60].max()
    l60 = sub['Low'].iloc[:60].min()
    cur_c = data_df['Close'].iloc[t]
    rp = (cur_c - l60) / (h60 - l60)
    print(f"Rolling 60 Max: {h60}, Min: {l60}, Cur Close: {cur_c}, Rel Pos: {rp:.4f}")
