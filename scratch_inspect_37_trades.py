import pandas as pd

df = pd.read_csv('mt5_v2_runner/MQL5/Files/self_growing_execution.csv', sep='\t')
p = df[df['profit'] != 0]

for idx, r in p.iterrows():
    pos_id = int(r['position_ticket'])
    pos_rows = df[df['position_ticket'] == pos_id]
    entry_deal = pos_rows[(pos_rows['transaction_type'] == 'DEAL_ADD') & (pos_rows['profit'] == 0)]
    act = entry_deal['action_name'].iloc[0] if len(entry_deal) > 0 else 'UNKNOWN'
    ent_price = r['entry_price']
    exit_p = r['price']
    print(f"Pos {pos_id:2d}: {act:11s} @ {ent_price:.2f} -> Exit: {str(r['close_reason']):2s} @ {exit_p:.2f} | Profit: ${r['profit']:+7.2f} | DealTime: {r['logged_at']}")
