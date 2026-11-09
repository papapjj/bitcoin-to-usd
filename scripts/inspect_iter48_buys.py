import json

traces = []
with open('outputs/mt5_optimization_loop/reasoning_trace.jsonl', 'r') as f:
    for line in f:
        if line.strip():
            traces.append(json.loads(line))

buys = [t for t in traces if t.get('action') == 'SCALP_BUY']
print(f"Total BUY traces: {len(buys)}")
for i, b in enumerate(buys):
    print(f"#{i+1:02d}: Tick={b.get('tick')} Price={b.get('price')} ATR={b.get('atr')} Margin={b.get('margin')} Bull={b.get('bull_score')} PRM={b.get('prm_score')} Verdict={b.get('verdict')}")
