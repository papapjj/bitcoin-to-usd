import json
import pandas as pd
import numpy as np

# 1. Load calibration sweep results
with open("scratch/calibration_sweep_results.json") as f:
    sweep = json.load(f)

df_sw = pd.DataFrame(sweep)
print(f"Loaded {len(df_sw)} sweep configurations.")
print("Columns:", df_sw.columns.tolist())

# Summary by prm_gate
print("\n=== Aggregation by prm_gate ===")
print(df_sw.groupby("prm_gate")[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct"]].agg(["mean", "min", "max"]).to_string())

# Summary by min_spacing
print("\n=== Aggregation by min_spacing ===")
print(df_sw.groupby("min_spacing")[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct"]].agg(["mean", "min", "max"]).to_string())

# Summary by loss_cd
print("\n=== Aggregation by loss_cd ===")
print(df_sw.groupby("loss_cd")[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct"]].agg(["mean", "min", "max"]).to_string())

# Summary by max_pos
print("\n=== Aggregation by max_pos ===")
print(df_sw.groupby("max_pos")[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct"]].agg(["mean", "min", "max"]).to_string())

# Summary by margin_thresh
print("\n=== Aggregation by margin_thresh ===")
print(df_sw.groupby("margin_thresh")[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct"]].agg(["mean", "min", "max"]).to_string())

# Configurations meeting trades_per_day >= 40.0
valid = df_sw[df_sw["trades_per_day"] >= 40.0]
print(f"\nTotal configurations with trades_per_day >= 40.0: {len(valid)} / {len(df_sw)}")
print("\nTop 10 configurations by Profit Factor with trades_per_day >= 40.0:")
top_pf = valid.sort_values(by="pf", ascending=False).head(10)
print(top_pf[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct", "prm_gate", "margin_thresh", "min_spacing", "max_pos", "loss_cd"]].to_string())

print("\nTop 10 configurations by Win Rate with trades_per_day >= 40.0:")
top_wr = valid.sort_values(by="wr", ascending=False).head(10)
print(top_wr[["trades_per_day", "wr", "pf", "net_pnl", "max_dd_pct", "prm_gate", "margin_thresh", "min_spacing", "max_pos", "loss_cd"]].to_string())
