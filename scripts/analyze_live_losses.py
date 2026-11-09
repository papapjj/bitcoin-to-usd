import pandas as pd
import numpy as np

# Load 7-day data
df = pd.read_csv('outputs/btcusd_m1_7day_clean.csv', sep='\t')
df['dt'] = df['<DATE>'] + ' ' + df['<TIME>']
closes = df['<CLOSE>'].values
highs = df['<HIGH>'].values
lows = df['<LOW>'].values
opens = df['<OPEN>'].values
dts = df['dt'].values

s_c = pd.Series(closes)
ema20 = s_c.ewm(span=20).mean().values
ema50 = s_c.ewm(span=50).mean().values

dt_map = {dt: idx for idx, dt in enumerate(dts)}

# Load execution csv
path = r'.mt5_runners\visual_7day_champion_3280_20261031_141851_985791\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv'
exec_df = pd.read_csv(path, sep='\t')

# Find entry deals (deal_entry == 0 or DEAL_ADD where closed == 0 and price > 0)
entries = exec_df[(exec_df['transaction_type'] == 'DEAL_ADD') & (exec_df['deal_entry'] == 0)].copy()
exits = exec_df[exec_df['closed'] == 1].copy()

print(f"Total entries logged: {len(entries)}")
print(f"Total exits logged: {len(exits)}")

# Match by position_ticket
merged = pd.merge(exits, entries, on='position_ticket', suffixes=('_exit', '_entry'))
print(f"Matched trades: {len(merged)}")

print("\nDetailed Trade Forensic:")
cols = [
    'logged_at_entry', 'action_name_entry', 'price_entry', 'entry_price_exit',
    'logged_at_exit', 'price_exit', 'profit_exit', 'age_bars_exit', 'close_reason_exit'
]
print(merged[cols].to_string())
