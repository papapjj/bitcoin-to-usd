import pandas as pd

df = pd.read_csv("outputs/acceptance_artifacts/milestone_3/iteration_9/trades.csv")
df["profit_num"] = pd.to_numeric(df["profit"], errors="coerce")
df["is_loss"] = df["profit_num"] <= 0

# Find streaks
streak = 0
streak_id = 0
streaks = []
for idx, row in df.iterrows():
    if row["is_loss"]:
        streak += 1
    else:
        if streak > 0:
            streaks.append((streak_id, streak, idx - streak, idx - 1))
            streak_id += 1
        streak = 0
if streak > 0:
    streaks.append((streak_id, streak, len(df) - streak, len(df) - 1))

streaks_sorted = sorted(streaks, key=lambda x: x[1], reverse=True)
print("Top 5 longest loss streaks in Iteration 9:")
for s_id, length, start_idx, end_idx in streaks_sorted[:5]:
    sub = df.iloc[start_idx:end_idx+1]
    print(f"\n--- Streak of length {length} (Trades #{start_idx} to #{end_idx}) ---")
    start_time = sub.iloc[0]["open_time"] if "open_time" in sub.columns else sub.iloc[0].get("time", "N/A")
    end_time = sub.iloc[-1]["close_time"] if "close_time" in sub.columns else sub.iloc[-1].get("time", "N/A")
    print(f"Time range: {start_time} to {end_time}")
    print(f"Directions: {sub['direction'].value_counts().to_dict()}")
    print(f"Exit kinds: {sub['exit_kind'].value_counts().to_dict()}")
    print(f"Total loss: ${sub['profit_num'].sum():.2f}")
    if "open_time" in sub.columns and "close_time" in sub.columns:
        print(sub[["ticket" if "ticket" in sub.columns else "deal_id" if "deal_id" in sub.columns else "direction", "direction", "open_time", "close_time", "profit_num", "exit_kind"]].to_string())
