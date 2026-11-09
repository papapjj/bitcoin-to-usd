import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import numpy as np

EXEC_CSV = ROOT / "mt5_v2_runner" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_execution.csv"
TRADES_CSV = ROOT / "outputs" / "acceptance_artifacts" / "milestone_3" / "iteration_9" / "trades.csv"

# Let's inspect self_growing_execution.csv to see all orders and deals
df_exec = pd.read_csv(EXEC_CSV, sep="\t", low_memory=False)
print("=" * 80)
print(f"SELF_GROWING_EXECUTION.CSV (Total rows: {len(df_exec)})")
print("=" * 80)
print("Columns:", df_exec.columns.tolist()[:15])

# Filter by DEAL_ADD or ORDER_SEND
if "trans_type" in df_exec.columns:
    print("\nTrans types:")
    print(df_exec["trans_type"].value_counts())

# Check deal types
if "deal_type" in df_exec.columns:
    print("\nDeal types:")
    print(df_exec["deal_type"].value_counts())

# Check action / direction in orders
for col in ["dir", "direction", "order_type", "type", "cmd"]:
    if col in df_exec.columns:
        print(f"\nValue counts for {col}:")
        print(df_exec[col].value_counts())

# Let's check the timing of BUY vs SELL trades across the week
df_trades = pd.read_csv(TRADES_CSV)
df_trades["date"] = df_trades["logged_at"].str[:10]
print("\nTrades per day by direction in Iteration 9:")
print(pd.crosstab(df_trades["date"], df_trades["direction"], margins=True))
