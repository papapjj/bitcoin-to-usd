from pathlib import Path

launch_file = Path("scripts/launch_7day_visual_master_champion.py")
content = launch_file.read_text(encoding="utf-8")

old_params = """    policy.min_atr_threshold = 5.0
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

new_params = """    policy.min_atr_threshold = 5.0
    policy.max_atr_percentile = 0.98
    policy.require_margin_alignment = False
    policy.loss_cooldown_bars = 0
    policy.max_consecutive_losses = 999
    policy.min_spacing_bars = 2
    policy.post_exit_cooldown_bars = 1
    policy.max_positions = 5
    policy.prm_threshold = 0.30
    policy.min_margin_threshold = 0.08
    policy.min_confidence = 0.35
    policy.bear_entry_threshold = 0.28
    policy.bear_min_margin = 0.05
    policy.scalp_sl_atr = 2.60
    policy.sniper_sl_atr = 2.60
    policy.scalp_tp_rr = 1.38
    policy.sniper_tp_rr = 1.38
    policy.trailing_atr = 1.20
    policy.breakeven_trigger_atr = 1.60
    policy.breakeven_lock_atr = 0.80"""

old_load = """    # Load Frontier Bilateral Dual-CoT Precision Champion Weights with Calibrated Threshold
    precision_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    bilateral_pt = precision_pt if precision_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt")
    if bilateral_pt.exists():
        policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.35
        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.35
        policy.pipeline.use_bilateral_dual_cot = True
        logger.info("Loaded Frontier Bilateral Dual-CoT Precision Champion from %s (prm_threshold=0.35)", bilateral_pt.name)"""

new_load = """    # Load Frontier Bilateral Dual-CoT Precision Champion Weights with Calibrated Threshold
    v3_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion_v3.pt"
    precision_pt = v3_pt if v3_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt")
    bilateral_pt = precision_pt if precision_pt.exists() else (ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt")
    if bilateral_pt.exists():
        policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.30
        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.30
        policy.pipeline.bilateral_engine.arbiter.trap_threshold = 0.99
        policy.pipeline.use_bilateral_dual_cot = True
        logger.info("Loaded Frontier Bilateral Dual-CoT Precision Champion from %s (prm_threshold=0.30)", bilateral_pt.name)"""

norm_c = content.replace("\r\n", "\n")
norm_old_params = old_params.replace("\r\n", "\n")
norm_new_params = new_params.replace("\r\n", "\n")
norm_old_load = old_load.replace("\r\n", "\n")
norm_new_load = new_load.replace("\r\n", "\n")

if norm_old_params in norm_c:
    norm_c = norm_c.replace(norm_old_params, norm_new_params)
    print("Replaced launch params successfully!")

if norm_old_load in norm_c:
    norm_c = norm_c.replace(norm_old_load, norm_new_load)
    print("Replaced launch load successfully!")

launch_file.write_text(norm_c, encoding="utf-8")
print("Launch script patched successfully!")
