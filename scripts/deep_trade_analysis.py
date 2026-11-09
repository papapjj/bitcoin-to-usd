import pandas as pd
import numpy as np

p = 'outputs/acceptance_artifacts/milestone_3/iteration_26/self_growing_execution.csv'
df = pd.read_csv(p, sep='\t')
closed = df[df['closed'] == 1].copy()

print(f"Total Closed: {len(closed)}")
wins = closed[closed['profit'] > 0]
losses = closed[closed['profit'] < 0]

print(f"Wins: {len(wins)} ({len(wins)/len(closed)*100:.1f}%) | Losses: {len(losses)} ({len(losses)/len(closed)*100:.1f}%)")
print(f"Avg Win: ${wins['profit'].mean():.2f} | Avg Loss: ${losses['profit'].mean():.2f}")
print(f"Gross Profit: ${wins['profit'].sum():.2f} | Gross Loss: ${abs(losses['profit'].sum()):.2f}")
print(f"PF: {wins['profit'].sum() / abs(losses['profit'].sum()):.2f}")

# Look at age of wins vs age of losses
print("\nAge in bars:")
print(f"Wins age_bars: mean={wins['age_bars'].mean():.1f}, median={wins['age_bars'].median():.1f}")
print(f"Losses age_bars: mean={losses['age_bars'].mean():.1f}, median={losses['age_bars'].median():.1f}")

# Look at distribution of losses
print("\nLosses breakdown by close_reason & exit_kind:")
print(losses.groupby(['close_reason', 'exit_kind'])['profit'].agg(['count', 'mean', 'sum']))

# Look at time of day of losses
losses['hour'] = pd.to_datetime(losses['logged_at']).dt.hour
wins['hour'] = pd.to_datetime(wins['logged_at']).dt.hour
print("\nWin Rate by hour of day:")
hourly = pd.DataFrame({
    'wins': wins.groupby('hour')['profit'].count(),
    'losses': losses.groupby('hour')['profit'].count()
}).fillna(0)
hourly['total'] = hourly['wins'] + hourly['losses']
hourly['WR%'] = hourly['wins'] / hourly['total'] * 100
print(hourly.to_string())
