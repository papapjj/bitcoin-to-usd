import pandas as pd

df = pd.read_csv('mt5_v2_runner/MQL5/Files/self_growing_execution.csv', sep='\t')
# Filter rows between 16:30 and 22:00
df['time'] = pd.to_datetime(df['logged_at'])
sub = df[(df['time'] >= '2025-01-06 16:30:00') & (df['time'] <= '2025-01-06 22:00:00')]
deals = sub[sub['profit'] != 0]
print(f"Deals between 16:30 and 22:00: {len(deals)}")
for idx, r in deals.iterrows():
    print(f"Time: {r['logged_at']} | Pos: {r['position_ticket']} | Action: {r['action_name']} | Entry: {r['entry_price']} | Exit: {r['price']} | Profit: {r['profit']}")
