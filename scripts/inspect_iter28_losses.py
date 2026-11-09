import pandas as pd

p = 'mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(p, sep='\t')
closed = df[df['closed'] == 1].copy()
recent = closed.tail(12)
print("Recent 12 deals in Iteration 28:")
for i, r in recent.iterrows():
    diff = r['price'] - r['entry_price']
    print(f"Time: {r['logged_at']} | PnL: ${r['profit']:.2f} | Exit: {r['exit_kind']} | Entry: {r['entry_price']:.2f} | Exit: {r['price']:.2f} | Diff: {diff:.2f}")

print("\nLosses breakdown:")
losses = closed[closed['profit'] < 0]
print(f"Total losses: {len(losses)} | Sum: ${losses['profit'].sum():.2f} | Mean: ${losses['profit'].mean():.2f}")
wins = closed[closed['profit'] > 0]
print(f"Total wins: {len(wins)} | Sum: ${wins['profit'].sum():.2f} | Mean: ${wins['profit'].mean():.2f}")
print(f"Win Rate: {len(wins)/len(closed)*100:.2f}% | PF: {wins['profit'].sum()/abs(losses['profit'].sum()):.2f}")
