import pandas as pd
import numpy as np

p = 'mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(p, sep='\t')
closed = df[df['closed'] == 1].copy()
closed['dt'] = pd.to_datetime(closed['logged_at'])
closed['day'] = closed['dt'].dt.date

print("Day-by-Day Performance in Iteration 28:")
daily = []
for day, g in closed.groupby('day'):
    wins = g[g['profit'] > 0]
    losses = g[g['profit'] < 0]
    gp = wins['profit'].sum()
    gl = abs(losses['profit'].sum())
    wr = len(wins) / len(g) * 100
    pf = gp / gl if gl > 0 else float('inf')
    net = gp - gl
    daily.append({
        'day': str(day),
        'trades': len(g),
        'wins': len(wins),
        'losses': len(losses),
        'WR%': f"{wr:.1f}%",
        'GrossProfit': f"${gp:.2f}",
        'GrossLoss': f"${gl:.2f}",
        'NetPnL': f"${net:+.2f}",
        'PF': f"{pf:.2f}",
        'price_start': f"{g['entry_price'].iloc[0]:.0f}",
        'price_end': f"{g['entry_price'].iloc[-1]:.0f}"
    })

res = pd.DataFrame(daily)
print(res.to_string(index=False))
