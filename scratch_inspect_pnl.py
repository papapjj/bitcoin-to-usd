import pandas as pd

path = '.mt5_runners/visual_tester_agg_17908_20261104_042201_514054/Tester/Agent-127.0.0.1-3000/MQL5/Files/self_growing_execution.csv'
df = pd.read_csv(path, sep='\t')
closed_deals = df[(df['closed'] == 1) | (df['position_final_close'] == 1)]
print(f"Total closed records: {len(closed_deals)}")
unique_closed = closed_deals.drop_duplicates(subset=['position_ticket'])
print(f"Unique closed positions: {len(unique_closed)}")

wins = 0
losses = 0
total_pnl = 0.0

for _, row in unique_closed.iterrows():
    profit = float(row.get('profit', 0.0))
    total_pnl += profit
    if profit > 0:
        wins += 1
    elif profit < 0:
        losses += 1
    t = row.get('logged_at')
    action = row.get('action_name')
    entry_p = row.get('entry_price')
    exit_p = row.get('price')
    sl_p = row.get('final_sl_price')
    tp_p = row.get('final_tp_price')
    reason = row.get('close_reason')
    print(f"Pos {row.get('position_ticket')}: {action} at {entry_p:.1f} -> Exit {exit_p:.1f} | PnL={profit:.2f}$ | Reason={reason} | SL={sl_p:.1f}")

wr = wins / max(1, (wins + losses)) * 100
print(f"\nSummary: Total={len(unique_closed)}, Wins={wins}, Losses={losses}, WinRate={wr:.1f}%, NetPnL={total_pnl:.2f}$")
