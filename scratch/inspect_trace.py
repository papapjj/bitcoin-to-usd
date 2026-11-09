import json
from pathlib import Path

p = Path("outputs/mt5_optimization_loop/reasoning_trace.jsonl")
starts = []
with open(p, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if '"tick": 1,' in line:
            d = json.loads(line)
            starts.append((i, d))

for idx, entry in starts:
    print(f"Line {idx}: price={entry.get('price')}, atr={entry.get('atr')}")

# Inspect the very last run from starts[-1][0] to end
last_start_line = starts[-1][0]
print(f"\nAnalyzing last run starting at line {last_start_line}...")
from collections import Counter
actions = Counter()
total = 0
margins = []
prms = []
b_scores = []
r_scores = []
veto_reasons = Counter()

with open(p, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if i >= last_start_line:
            total += 1
            d = json.loads(line)
            act = d.get("action")
            actions[act] += 1
            margins.append(d.get("margin", 0.0))
            prms.append(d.get("prm_score", 0.0))
            b_scores.append(d.get("bull_score", 0.0))
            r_scores.append(d.get("bear_score", 0.0))
            verdict = d.get("verdict", "")
            if "HOLD (" in verdict:
                reason = verdict.split("HOLD (")[1].split(",")[0]
                veto_reasons[reason] += 1

print(f"Total ticks in last run: {total}")
print("Action counts:", dict(actions))
print("Hold reasons:", dict(veto_reasons))
