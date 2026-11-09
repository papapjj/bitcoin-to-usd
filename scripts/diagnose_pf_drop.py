import pandas as pd
import numpy as np

p = 'mt5_v2_runner/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(p, sep='\t')
print("Total rows:", len(df))

# In self_growing_execution.csv, each deal or exit has deal_entry == DEAL_ENTRY_OUT or closed == 1
# Let's see unique values of deal_entry and transaction_type
print("Unique transaction_type:", df['transaction_type'].unique())
print("Unique deal_entry:", df['deal_entry'].unique())

closed = df[df['closed'] == 1].copy()
# Alternatively, filter deals that have profit != 0 or exit
exit_deals = df[(df['closed'] == 1) | (df['deal_entry'] == 'DEAL_ENTRY_OUT') | (df['profit'] != 0.0)].copy()
# Remove duplicates by deal_ticket or event_id if any
print(f"Total exit rows: {len(exit_deals)}")

# Let's inspect unique deal_ticket with profit != 0
deal_pnl = df.groupby('position_ticket')['profit'].sum().reset_index()
# Filter completed positions
completed_pos = df[df['position_final_close'] == 1]['position_ticket'].unique()
print(f"Total completed positions: {len(completed_pos)}")

pos_df = []
for pos_id, group in df.groupby('position_ticket'):
    # total profit for this position
    net_p = group['profit'].sum()
    comm = group['commission'].sum() if 'commission' in group.columns else 0.0
    swp = group['swap'].sum() if 'swap' in group.columns else 0.0
    final_pnl = net_p + comm + swp
    is_closed = (group['position_final_close'] == 1).any() or (group['closed'] == 1).any()
    exit_row = group[group['closed'] == 1]
    close_reason = exit_row['close_reason'].iloc[-1] if not exit_row.empty else 'OPEN'
    exit_kind = exit_row['exit_kind'].iloc[-1] if not exit_row.empty else 'OPEN'
    
    pos_df.append({
        'position_ticket': pos_id,
        'is_closed': is_closed,
        'profit': net_p,
        'pnl': final_pnl,
        'close_reason': close_reason,
        'exit_kind': exit_kind,
        'first_time': group['logged_at'].iloc[0],
        'last_time': group['logged_at'].iloc[-1],
        'bars': group['age_bars'].max() if 'age_bars' in group.columns else len(group)
    })

res = pd.DataFrame(pos_df)
closed_res = res[res['is_closed'] & (res['position_ticket'] > 0)].copy()
print(f"Closed positions count: {len(closed_res)}")

wins = closed_res[closed_res['profit'] > 0]
losses = closed_res[closed_res['profit'] < 0]
breakeven = closed_res[closed_res['profit'] == 0]

gp = wins['profit'].sum()
gl = abs(losses['profit'].sum())
pf = gp / gl if gl > 0 else 0
wr = len(wins) / len(closed_res) * 100 if len(closed_res) > 0 else 0

print("\n" + "="*60)
print(f"CLOSED POSITIONS AUDIT:")
print(f"Total: {len(closed_res)} | Wins: {len(wins)} | Losses: {len(losses)} | BE: {len(breakeven)}")
print(f"Win Rate: {wr:.2f}% | Profit Factor: {pf:.2f} | Net PnL: ${gp - gl:.2f}")
print(f"Gross Profit: ${gp:.2f} | Gross Loss: ${gl:.2f}")
print(f"Avg Win: ${wins['profit'].mean():.2f} (Median: ${wins['profit'].median():.2f})")
print(f"Avg Loss: ${losses['profit'].mean():.2f} (Median: ${losses['profit'].median():.2f})")
print("="*60)

print("\nBreakdown by close_reason:")
print(closed_res.groupby('close_reason')['profit'].agg(['count', 'sum', 'mean', lambda x: (x > 0).mean() * 100]).rename(columns={'<lambda_0>': 'WR%'}))

print("\nBreakdown by exit_kind:")
print(closed_res.groupby('exit_kind')['profit'].agg(['count', 'sum', 'mean', lambda x: (x > 0).mean() * 100]).rename(columns={'<lambda_0>': 'WR%'}))

print("\nWorst 10 Losses:")
print(losses.sort_values('profit').head(10)[['position_ticket', 'profit', 'close_reason', 'exit_kind', 'first_time', 'last_time']])

print("\nBest 10 Wins:")
print(wins.sort_values('profit', ascending=False).head(10)[['position_ticket', 'profit', 'close_reason', 'exit_kind', 'first_time', 'last_time']])
