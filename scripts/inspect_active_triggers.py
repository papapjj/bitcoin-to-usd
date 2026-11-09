import json
import pandas as pd

# Load execution CSV
csv_path = r'.mt5_runners\visual_7day_champion_1944_20261031_152517_184180\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_execution.csv'
df = pd.read_csv(csv_path, sep='\t')
entries = df[df['deal_entry'] == 0]

# Load reasoning trace
trace_path = 'outputs/mt5_optimization_loop/reasoning_trace.jsonl'
traces = []
with open(trace_path, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            traces.append(json.loads(line.strip()))

# Find active reasoning ticks (action != HOLD)
active_traces = [t for t in traces if t.get('action') != 'HOLD']
print(f"Total active reasoning triggers: {len(active_traces)}")

for i, at in enumerate(active_traces, 1):
    print(f"\nTrigger {i:02d}: Tick={at['tick']}, Price={at['price']}, ATR={at['atr']}, Action={at['action']}")
    print(f"  Bull={at.get('bull_score')}, Bear={at.get('bear_score')}, PRM={at.get('prm_score')}, Margin={at.get('margin')}")
    print(f"  Verdict: {at.get('verdict')}")
    print(f"  Scenarios: {at.get('scenarios')}")
