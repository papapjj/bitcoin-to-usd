import numpy as np
import pandas as pd
import json

# Data points from the live MT5 Strategy Tester runs
live_runs = [
    {"iter": 8, "trades": 89, "freq": 14.83, "wr": 57.30, "pf": 1.38, "prm": 0.40, "spacing": 2, "loss_cd": 4, "max_pos": 2, "avg_win": 16.46, "avg_loss": 16.04},
    {"iter": 7, "trades": 161, "freq": 26.83, "wr": 55.28, "pf": 0.96, "prm": 0.52, "spacing": 2, "loss_cd": 4, "max_pos": 2, "avg_win": 12.35, "avg_loss": 15.84},
    {"iter": 9, "trades": 244, "freq": 40.67, "wr": 51.23, "pf": 0.91, "prm": 0.32, "spacing": 2, "loss_cd": 2, "max_pos": 3, "avg_win": 13.86, "avg_loss": 15.94},
]
df_live = pd.DataFrame(live_runs)

# Linear regression on live runs: WR vs Frequency
x = df_live["freq"].values
y_wr = df_live["wr"].values
y_pf = df_live["pf"].values

# Fit WR(F) = a * F + b
slope_wr, intercept_wr = np.polyfit(x, y_wr, 1)
# Fit PF(F) = c * F + d
slope_pf, intercept_pf = np.polyfit(x, y_pf, 1)

print("=== Linear Fit on MT5 Strategy Tester Live Runs ===")
print(f"WR(F) = {slope_wr:.4f} * F + {intercept_wr:.2f}")
print(f"PF(F) = {slope_pf:.4f} * F + {intercept_pf:.2f}")

# Quadratic fit
p2_wr = np.polyfit(x, y_wr, 2)
p2_pf = np.polyfit(x, y_pf, 2)
print(f"Quadratic WR(F) = {p2_wr[0]:.5f} * F^2 + {p2_wr[1]:.4f} * F + {p2_wr[2]:.2f}")
print(f"Quadratic PF(F) = {p2_pf[0]:.5f} * F^2 + {p2_pf[1]:.4f} * F + {p2_pf[2]:.2f}")

# Theoretical Model:
# Let quality distribution be Exponential/Pareto:
# WR(F) = WR_max - delta * (F / F_0)^gamma
# PF(F) = (WR / (1 - WR)) * R
print("\n=== Predictions at target frequencies ===")
for f_test in [15.0, 25.0, 35.0, 40.0, 42.0, 45.0, 50.0]:
    wr_pred = slope_wr * f_test + intercept_wr
    pf_pred = slope_pf * f_test + intercept_pf
    print(f"Freq = {f_test:4.1f} trades/day -> Pred WR = {wr_pred:.2f}%, Pred PF = {pf_pred:.2f}")
