import pandas as pd

p = 'mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(p, sep='\t')
closed = df[df['closed'] == 1].copy()
print(f"Closed deals: {len(closed)}")
wins = closed[closed['profit'] > 0]
losses = closed[closed['profit'] < 0]
be = closed[closed['profit'] == 0]
print(f"Wins: {len(wins)} | Losses: {len(losses)} | BE: {len(be)}")
if len(wins) > 0 and len(losses) > 0:
    print(f"Avg Win: ${wins['profit'].mean():.2f} | Avg Loss: ${losses['profit'].mean():.2f}")
    gp = wins['profit'].sum()
    gl = abs(losses['profit'].sum())
    print(f"Gross Profit: ${gp:.2f} | Gross Loss: ${gl:.2f} | PF: {gp/gl:.2f}")
print("\nBreakdown by exit_kind:")
print(closed.groupby('exit_kind')['profit'].agg(['count', 'sum', 'mean']))

print("\nRecent 10 closed deals:")
for i, r in closed.tail(10).iterrows():
    print(f"Time: {r['logged_at']} | Ticket: {r['position_ticket']} | Exit: {r['exit_kind']} | PnL: ${r['profit']:.2f}")
