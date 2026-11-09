import json

p = 'outputs/mt5_optimization_loop/reasoning_trace.jsonl'
lines = []
with open(p, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip(): lines.append(json.loads(line.strip()))

high_bull = [l for l in lines if l.get('bull_score', 0) >= 0.48 or l.get('bear_score', 0) >= 0.48]
print(f'Ticks with conviction >= 0.48: {len(high_bull)}')
for l in high_bull[:10]:
    print(f"Tick {l['tick']}: Bull={l['bull_score']}, Bear={l['bear_score']}, PRM={l['prm_score']}, Action={l['action']}, Verdict={l['verdict']}")
