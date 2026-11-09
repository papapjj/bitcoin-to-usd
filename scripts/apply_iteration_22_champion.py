from pathlib import Path

# 1. Update SelfGrowing_MT5_Trainer_EA.mq5
p_ea = Path("mt5_v2_runner/MQL5/Experts/SelfGrowing_MT5_Trainer_EA.mq5")
ea_text = p_ea.read_text(encoding="utf-8")

old_ea = """input double FixedSlAtr = 3.60;
input double FixedTpAtr = 3.60;
input double TrailActivationAtr = 2.20;
input double TrailLockAtr = 1.00;"""

new_ea = """input double FixedSlAtr = 3.80;
input double FixedTpAtr = 3.00;
input double TrailActivationAtr = 2.00;
input double TrailLockAtr = 0.80;"""

assert old_ea in ea_text, "old_ea not found"
ea_text = ea_text.replace(old_ea, new_ea, 1)
p_ea.write_text(ea_text, encoding="utf-8")
print("SelfGrowing_MT5_Trainer_EA.mq5 updated.")

# 2. Update scripts/run_mythos_mt5_live.py
p_run = Path("scripts/run_mythos_mt5_live.py")
run_text = p_run.read_text(encoding="utf-8")

old_run1 = """    mythos_policy.sniper_sl_atr = 3.60
    mythos_policy.sniper_tp_rr = 1.00
    mythos_policy.scalp_sl_atr = 3.60
    mythos_policy.scalp_tp_rr = 1.00
    mythos_policy.trailing_atr = 1.60"""

new_run1 = """    mythos_policy.sniper_sl_atr = 3.80
    mythos_policy.sniper_tp_rr = 0.789
    mythos_policy.scalp_sl_atr = 3.80
    mythos_policy.scalp_tp_rr = 0.789
    mythos_policy.trailing_atr = 1.60"""

assert old_run1 in run_text, "old_run1 not found"
run_text = run_text.replace(old_run1, new_run1, 1)

old_run2 = """"fixed_sl_atr": 3.60,
        "fixed_tp_atr": 3.60,
        "min_rr": 0.95,
        "technical_min_rr": 0.95,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 2.20,
        "trail_lock_atr": 1.00,"""

new_run2 = """"fixed_sl_atr": 3.80,
        "fixed_tp_atr": 3.00,
        "min_rr": 0.75,
        "technical_min_rr": 0.75,
        "max_spread_cost_r": 0.28,
        "trail_activation_atr": 2.00,
        "trail_lock_atr": 0.80,"""

assert old_run2 in run_text, "old_run2 not found"
run_text = run_text.replace(old_run2, new_run2, 1)
p_run.write_text(run_text, encoding="utf-8")
print("scripts/run_mythos_mt5_live.py updated.")

# 3. Update research_reasoning_world_class/src/mythos_mt5_policy.py
p_pol = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
pol_text = p_pol.read_text(encoding="utf-8")

old_macro = """        # Macro Trend Guard (Prevents counter-trend exhaustion entries against strong EMA50 momentum):
        if net_dir == -1 and p_vs_e50_val > 0.20 and -diff < 0.55:
            net_dir = 0  # Do not short in uptrend unless extreme bear conviction"""

new_macro = """        # Macro Trend Alignment Guard (Strict Trend-Following: never fight the macro trend)
        if net_dir == -1 and (ema_diff_val > 0.0 or p_vs_e50_val > 0.0):
            net_dir = 0  # Strictly forbid shorting when trend is bullish (EMA20 > EMA60 or Price > EMA50)
        elif net_dir == 1 and (ema_diff_val < 0.0 and p_vs_e50_val < -0.50):
            net_dir = 0  # Strictly forbid buying when trend is strongly bearish"""

assert old_macro in pol_text, "old_macro not found"
pol_text = pol_text.replace(old_macro, new_macro, 1)
p_pol.write_text(pol_text, encoding="utf-8")
print("research_reasoning_world_class/src/mythos_mt5_policy.py updated.")
