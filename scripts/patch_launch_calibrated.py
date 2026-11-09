from pathlib import Path

p = Path("scripts/launch_7day_visual_master_champion.py")
content = p.read_text(encoding="utf-8")

old_code = """    policy.min_margin_threshold = 0.20
    policy.max_margin_threshold = 0.95
    policy.bear_entry_threshold = 0.50
    policy.bear_min_margin = 0.20
    policy.min_confidence = 0.50
    policy.prm_threshold = 0.38
    policy.min_atr_threshold = 5.0
    policy.max_atr_percentile = 0.98
    policy.require_margin_alignment = True
    policy.loss_cooldown_bars = 0
    policy.max_consecutive_losses = 999
    policy.min_spacing_bars = 2
    policy.post_exit_cooldown_bars = 2
    policy.max_positions = 5
    policy.scalp_sl_atr = 2.80
    policy.sniper_sl_atr = 2.80
    policy.scalp_tp_rr = 1.43
    policy.sniper_tp_rr = 1.43
    policy.trailing_atr = 1.20
    policy.breakeven_trigger_atr = 2.00
    policy.breakeven_lock_atr = 0.80
    policy.prm_threshold = 0.38
    policy.min_margin_threshold = 0.20
    policy.min_confidence = 0.50"""

new_code = """    policy.min_margin_threshold = 0.12
    policy.max_margin_threshold = 0.95
    policy.bear_entry_threshold = 0.35
    policy.bear_min_margin = 0.08
    policy.min_confidence = 0.40
    policy.prm_threshold = 0.35
    policy.min_atr_threshold = 5.0
    policy.max_atr_percentile = 0.98
    policy.require_margin_alignment = True
    policy.loss_cooldown_bars = 0
    policy.max_consecutive_losses = 999
    policy.min_spacing_bars = 2
    policy.post_exit_cooldown_bars = 1
    policy.max_positions = 5
    policy.scalp_sl_atr = 2.80
    policy.sniper_sl_atr = 2.80
    policy.scalp_tp_rr = 1.43
    policy.sniper_tp_rr = 1.43
    policy.trailing_atr = 1.20
    policy.breakeven_trigger_atr = 2.00
    policy.breakeven_lock_atr = 0.80"""

old_norm = old_code.replace("\r\n", "\n")
c_norm = content.replace("\r\n", "\n")
assert old_norm in c_norm, "Old launch settings not found!"
c_norm = c_norm.replace(old_norm, new_code.replace("\r\n", "\n"), 1)

# Also update lines 316-317 threshold to 0.35
c_norm = c_norm.replace(
    "policy.pipeline.bilateral_engine.config.prm_threshold = 0.38\n        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.38",
    "policy.pipeline.bilateral_engine.config.prm_threshold = 0.35\n        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.35"
)
c_norm = c_norm.replace("(prm_threshold=0.38)", "(prm_threshold=0.35)")

p.write_text(c_norm, encoding="utf-8")
print("Successfully patched launch_7day_visual_master_champion.py!")
