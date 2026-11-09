import json
p = 'outputs/mt5_optimization_loop/reasoning_trace.jsonl'
with open(p, 'r', encoding='utf-8') as f:
    lines = [json.loads(line.strip()) for line in f if line.strip()]

for l in lines[70:]:
    if l.get('bull_score', 0) >= 0.40 or l.get('bear_score', 0) >= 0.40 or l['action'] != 'HOLD':
        print(f"Tick {l['tick']}: Action={l['action']}, Bull={l['bull_score']}, Bear={l['bear_score']}, PRM={l['prm_score']}, Verdict={l['verdict']}")
