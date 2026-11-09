import json
import pandas as pd

csv_path = r'.mt5_runners\visual_7day_champion_1944_20261031_152517_184180\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv'
df = pd.read_csv(csv_path, sep='\t')

# Separate entries and exits
entries = df[df['deal_entry'] == 0]
exits = df[df['deal_entry'] == 1]

print(f"Total entries: {len(entries)} | Total exits: {len(exits)}")

# Read reasoning trace
trace_path = 'outputs/mt5_optimization_loop/reasoning_trace.jsonl'
traces = []
with open(trace_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            traces.append(json.loads(line.strip()))

print(f"Total reasoning traces: {len(traces)}")

# Match exits by position_ticket
print("\n" + "="*120)
print(f"{'#':<3} {'Type':<6} {'Entry Time':<20} {'Entry P':<10} {'Exit Time':<20} {'Exit P':<10} {'SL Price':<10} {'TP Price':<10} {'Bars':<5} {'Profit':<8} {'Reason':<6}")
print("="*120)

for i, (_, ex) in enumerate(exits.iterrows(), 1):
    pos_id = ex['position_ticket']
    ent = entries[entries['position_ticket'] == pos_id]
    if len(ent) > 0:
        e_row = ent.iloc[0]
        e_time = e_row['logged_at']
        e_act = e_row['action_name']
        e_p = e_row['price']
        sl_p = e_row['initial_sl_price']
        tp_p = e_row['initial_tp_price']
        x_time = ex['logged_at']
        x_p = ex['price']
        age = ex['age_bars']
        pnl = ex['profit']
        reason = ex['close_reason']
        print(f"{i:02d}. {e_act:<6} {e_time:<20} {e_p:<10.2f} {x_time:<20} {x_p:<10.2f} {sl_p:<10.2f} {tp_p:<10.2f} {age:<5} ${pnl:<+7.2f} {reason:<6}")
    else:
        print(f"{i:02d}. UNMATCHED EXIT: {ex['logged_at']} p={ex['price']} profit={ex['profit']}")
