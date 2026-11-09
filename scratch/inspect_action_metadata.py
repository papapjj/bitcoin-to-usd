import json
from pathlib import Path
import pandas as pd

ROOT = Path("c:/Users/Admin/Desktop/new-chat")
META_PATH = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_action.csv.metadata.jsonl"

records = []
with open(META_PATH, "r", encoding="utf-8", errors="ignore") as f:
    for i, line in enumerate(f):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            records.append(d)
        except Exception:
            pass

print(f"Total lines parsed: {len(records)}")
df = pd.DataFrame(records)
print("Columns in metadata:", df.columns.tolist())

# Inspect actions returned to MT5
# Look at fields: action, verdict, emitted_actions, etc.
for col in df.columns:
    if "act" in col.lower() or "dir" in col.lower() or "verdict" in col.lower():
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts().head(10))

print("\nSample records:")
for r in records[:5]:
    print(r)
