import pandas as pd
import numpy as np

df = pd.read_csv('mt5_v2_runner/MQL5/Files/self_growing_execution.csv', sep='\t')
print(f"Total rows: {len(df)}")

# Look at entries
entries = df[df['deal_entry'] == 1]
print(f"Entries count: {len(entries)}")

# Look at exits
exits = df[df['deal_entry'] == 2]
print(f"Exits count: {len(exits)}")
print("\n--- EXITS SUMMARY ---")
print(exits[['logged_at', 'action_name', 'deal_reason', 'close_reason', 'profit', 'volume', 'price']].to_string())

# Count wins and losses
wins = exits[exits['profit'] > 0]
losses = exits[exits['profit'] < 0]
print(f"\nWins: {len(wins)} | Total Profit: ${wins['profit'].sum():.2f} | Avg Win: ${wins['profit'].mean():.2f}")
print(f"Losses: {len(losses)} | Total Loss: ${losses['profit'].sum():.2f} | Avg Loss: ${losses['profit'].mean():.2f}")

# Group by intent_reason of entry
print("\n--- ENTRIES BY REASON ---")
if 'intent_reason' in entries.columns:
    print(entries['intent_reason'].value_counts())
