import json
from pathlib import Path
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/new-chat")
META_PATH = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_action.csv.metadata.jsonl"

records = []
with open(META_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

df = pd.DataFrame(records)
long_df = df[df["decision_role"] == "LONG_ENTRY_OWNER"]

print("Overall regime distribution across all 1,481 decision ticks:")
print(long_df["regime_name"].value_counts(normalize=True)*100)
print("\nCounts:")
print(long_df["regime_name"].value_counts())
print("\nMean probabilities:")
print(f"p_bull mean: {long_df['p_bull'].mean():.4f}, median: {long_df['p_bull'].median():.4f}")
print(f"p_bear mean: {long_df['p_bear'].mean():.4f}, median: {long_df['p_bear'].median():.4f}")
print(f"p_chop mean: {long_df['p_chop'].mean():.4f}, median: {long_df['p_chop'].median():.4f}")
