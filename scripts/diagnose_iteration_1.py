import json
import pandas as pd

df = pd.read_csv("outputs/acceptance_artifacts/milestone_3/iteration_1/trades.csv")
print("Total trades:", len(df))
print("Win rate:", len(df[df['net_pnl'] > 0]) / len(df) * 100)
print("Gross profit:", df[df['net_pnl'] > 0]['net_pnl'].sum())
print("Gross loss:", df[df['net_pnl'] < 0]['net_pnl'].abs().sum())

# Inspect ticks with non-HOLD in reasoning_trace.jsonl
non_holds = []
with open("outputs/mt5_optimization_loop/reasoning_trace.jsonl") as f:
    for line in f:
        d = json.loads(line)
        if d["action"] != "HOLD":
            non_holds.append(d)

print(f"Total non-HOLD decisions in reasoning trace: {len(non_holds)}")
prms = [d["prm_score"] for d in non_holds]
print(f"Min PRM: {min(prms):.4f}, Mean PRM: {sum(prms)/len(prms):.4f}, Max PRM: {max(prms):.4f}")
print("PRM breakdown of non-HOLD decisions:")
print("  < 0.40:", sum(1 for p in prms if p < 0.40))
print("  0.40 - 0.50:", sum(1 for p in prms if 0.40 <= p < 0.50))
print("  0.50 - 0.58:", sum(1 for p in prms if 0.50 <= p < 0.58))
print("  0.58 - 0.65:", sum(1 for p in prms if 0.58 <= p < 0.65))
print("  >= 0.65:", sum(1 for p in prms if p >= 0.65))
