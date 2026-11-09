import json
import pandas as pd
import numpy as np

csv_file = r'.mt5_runners\opt_Iter_46_2Day_14240_20261029_202154_339863\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv'
df = pd.read_csv(csv_file, sep='\t')
closed = df[df['position_final_close'] == 1].copy()

# Read trace
buy_traces = []
with open('outputs/mt5_optimization_loop/reasoning_trace.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            rec = json.loads(line)
            if rec.get('action') == 'SCALP_BUY':
                buy_traces.append(rec)

print(f"Total Closed: {len(closed)}, Traces: {len(buy_traces)}")

closed_list = []
for idx, (_, row) in enumerate(closed.iterrows()):
    p = row['profit']
    bt = buy_traces[idx] if idx < len(buy_traces) else {}
    closed_list.append({
        'deal': idx + 1,
        'profit': p,
        'is_win': 1 if p > 0 else 0,
        'time': row['logged_at'],
        'entry_price': row['entry_price'],
        'exit_price': row['price'],
        'close_reason': row['close_reason'],
        'exit_kind': row['exit_kind'],
        'bull': bt.get('bull_score', 0),
        'bear': bt.get('bear_score', 0),
        'margin': bt.get('margin', 0),
        'prm': bt.get('prm_score', 0),
        'atr': bt.get('atr', 0),
    })

res_df = pd.DataFrame(closed_list)

print("\n--- SUMMARY OF WINS vs LOSSES ---")
wins = res_df[res_df['is_win'] == 1]
losses = res_df[res_df['is_win'] == 0]

print("WINS stats:")
print(wins[['margin', 'bull', 'bear', 'prm', 'atr']].describe())

print("\nLOSSES stats:")
print(losses[['margin', 'bull', 'bear', 'prm', 'atr']].describe())

print("\n--- DETAILED LOSSES ---")
print(losses[['deal', 'time', 'profit', 'entry_price', 'exit_price', 'close_reason', 'margin', 'bull', 'bear', 'prm', 'atr']].to_string())

print("\n--- DETAILED WINS ---")
print(wins[['deal', 'time', 'profit', 'entry_price', 'exit_price', 'close_reason', 'margin', 'bull', 'bear', 'prm', 'atr']].to_string())
