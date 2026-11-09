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

df = pd.DataFrame(records)
print(f"Total metadata records: {len(df)}")
# In Five-Agent Architecture, each decision tick has 5 rows:
# row 0: LONG_ENTRY_OWNER (Agent 0)
# row 1: SHORT_ENTRY_OWNER (Agent 1)
# row 2: POSITION_MANAGER (Agent 2)
# row 3: PENDING_MANAGER (Agent 3)
# row 4: PORTFOLIO_GOVERNOR (Agent 4)

# Group by sequence or event_id
grouped = df.groupby("event_id")
print("Total decision ticks (events):", len(grouped))

tick_actions = []
for ev_id, group in grouped:
    # Get rows by decision_role
    g_long = group[group["decision_role"] == "LONG_ENTRY_OWNER"]
    g_short = group[group["decision_role"] == "SHORT_ENTRY_OWNER"]
    
    act_long = g_long["emitted_action"].iloc[0] if len(g_long) else 0
    act_short = g_short["emitted_action"].iloc[0] if len(g_short) else 0
    
    name_long = g_long["action_name"].iloc[0] if len(g_long) else "HOLD"
    name_short = g_short["action_name"].iloc[0] if len(g_short) else "HOLD"
    
    regime = g_long["regime_name"].iloc[0] if len(g_long) else "UNKNOWN"
    p_bull = g_long["p_bull"].iloc[0] if len(g_long) else 0.0
    p_bear = g_long["p_bear"].iloc[0] if len(g_long) else 0.0
    p_chop = g_long["p_chop"].iloc[0] if len(g_long) else 0.0
    
    tick_actions.append({
        "event_id": ev_id,
        "seq": group["sequence"].iloc[0] if "sequence" in group else 0,
        "act_long": act_long,
        "act_short": act_short,
        "name_long": name_long,
        "name_short": name_short,
        "regime": regime,
        "p_bull": p_bull,
        "p_bear": p_bear,
        "p_chop": p_chop,
    })

df_ticks = pd.DataFrame(tick_actions)
print("\nTick-level actions:")
print("Long owner emitted actions:")
print(df_ticks["name_long"].value_counts())
print("\nShort owner emitted actions:")
print(df_ticks["name_short"].value_counts())

print("\nTicks where Long emitted BUY:", (df_ticks["act_long"] == 1).sum())
print("Ticks where Short emitted SELL:", (df_ticks["act_short"] == 2).sum())
print("Ticks where BOTH emitted entry (conflict):", ((df_ticks["act_long"] == 1) & (df_ticks["act_short"] == 2)).sum())

# Inspect the regime probabilities when BUY vs SELL was emitted
print("\nWhen Long emitted BUY:")
print(df_ticks[df_ticks["act_long"] == 1][["regime", "p_bull", "p_bear", "p_chop"]].describe())

print("\nWhen Short emitted SELL:")
print(df_ticks[df_ticks["act_short"] == 2][["regime", "p_bull", "p_bear", "p_chop"]].describe())

# Value counts of regime name when BUY vs SELL
print("\nRegime name when Long emitted BUY:")
print(df_ticks[df_ticks["act_long"] == 1]["regime"].value_counts())

print("\nRegime name when Short emitted SELL:")
print(df_ticks[df_ticks["act_short"] == 2]["regime"].value_counts())
