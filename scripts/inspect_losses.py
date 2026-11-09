import pandas as pd
p = 'mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(p, sep='\t')
losses = df[(df['closed'] == 1) & (df['profit'] < -20)]
print("Losing trades sample:")
for i, r in losses.head(10).iterrows():
    diff = r['price'] - r['entry_price']
    print(f"Time: {r['logged_at']} | Entry: {r['entry_price']:.2f} | Exit: {r['price']:.2f} | Diff: {diff:.2f} | Profit: {r['profit']:.2f} | Bars: {r['age_bars']}")
