import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
launch_path = ROOT / "scripts" / "launch_7day_visual_master_champion.py"
content = launch_path.read_text(encoding="utf-8")
content_lf = content.replace("\r\n", "\n")

# Replace two_tier config
old_two_tier = """    two_tier = TwoTierConfig(
        initial_sl_atr=2.80,
        breakeven_trigger_atr=2.00,
        breakeven_lock_atr=0.80,
        partial_tp_atr=4.00,
        trailing_atr=1.20,
        trail_after_breakeven=True,
    )"""

new_two_tier = """    two_tier = TwoTierConfig(
        initial_sl_atr=2.80,
        breakeven_trigger_atr=1.50,
        breakeven_lock_atr=0.80,
        partial_tp_atr=4.50,
        trailing_atr=1.20,
        trail_after_breakeven=True,
    )"""

if old_two_tier in content_lf:
    content_lf = content_lf.replace(old_two_tier, new_two_tier)
    print("Replaced two_tier config.")

# Replace base_router thresholds
old_router = """        bull_threshold=0.35,
        bear_threshold=0.35,
        chop_threshold=0.38,
        confirmation_bars=1,
        htf_filter_enabled=True,
        momentum_gate_enabled=True,
        circuit_breaker_enabled=False,
        micro_scalp_enabled=False,
        max_concurrent_positions=5,
        max_positions_per_direction=5,"""

new_router = """        bull_threshold=0.58,
        bear_threshold=0.52,
        chop_threshold=0.38,
        confirmation_bars=1,
        htf_filter_enabled=True,
        momentum_gate_enabled=True,
        circuit_breaker_enabled=False,
        micro_scalp_enabled=False,
        max_concurrent_positions=3,
        max_positions_per_direction=3,"""

if old_router in content_lf:
    content_lf = content_lf.replace(old_router, new_router)
    print("Replaced base_router thresholds.")

# Replace policy config block
old_policy_block = """    policy = MythosMT5Policy(base_policy=base_router, initial_capital=args.initial_capital, symbol="BTC")
    policy.long_only = False  # Unlocked Bidirectional BUY & SELL
    policy.short_only = False
    policy.min_margin_threshold = 0.12
    policy.max_margin_threshold = 0.95
    policy.bear_entry_threshold = 0.35
    policy.bear_min_margin = 0.08
    policy.min_confidence = 0.40
    policy.prm_threshold = 0.35
    policy.min_atr_threshold = 5.0
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

new_policy_block = """    policy = MythosMT5Policy(base_policy=base_router, initial_capital=args.initial_capital, symbol="BTC")
    policy.long_only = False  # Unlocked Bidirectional BUY & SELL
    policy.short_only = False
    policy.min_confidence = 0.58       # Calibrated Bull Specialist Conviction
    policy.bear_entry_threshold = 0.52 # Calibrated Bear Specialist Conviction
    policy.min_margin_threshold = 0.15 # Bull - Bear Margin
    policy.bear_min_margin = 0.15      # Bear - Bull Margin
    policy.prm_threshold = 0.38        # Process Reward Model Gate
    policy.scalp_sl_atr = 2.80         # Strict Anti-MAE Stop Loss
    policy.sniper_sl_atr = 2.80
    policy.scalp_tp_rr = 1.25          # Target 3.50 ATR (2.80 * 1.25)
    policy.sniper_tp_rr = 1.25
    policy.breakeven_trigger_atr = 1.50 # Ratchet trigger at +1.50 ATR
    policy.breakeven_lock_atr = 0.80    # Lock +0.80 ATR into profit
    policy.trailing_atr = 1.20          # Chandelier trailing distance
    policy.max_positions = 3            # Concurrent position capacity
    policy.min_spacing_bars = 2         # Minimum bar spacing
    policy.post_exit_cooldown_bars = 1  # 1-bar anti-whip pause
    policy.min_atr_threshold = 5.0
    policy.max_atr_percentile = 0.98
    policy.require_margin_alignment = False
    policy.loss_cooldown_bars = 0
    policy.max_consecutive_losses = 999"""

if old_policy_block in content_lf:
    content_lf = content_lf.replace(old_policy_block, new_policy_block)
    print("Replaced policy config block.")

# Replace bilateral engine threshold block
old_bilateral = """    if bilateral_pt.exists():
        policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.30
        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.30
        policy.pipeline.bilateral_engine.arbiter.trap_threshold = 0.99
        policy.pipeline.use_bilateral_dual_cot = True
        logger.info("Loaded Frontier Bilateral Dual-CoT Precision Champion from %s (prm_threshold=0.30)", bilateral_pt.name)"""

new_bilateral = """    if bilateral_pt.exists():
        policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.38
        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.38
        policy.pipeline.bilateral_engine.arbiter.trap_threshold = 0.99
        policy.pipeline.use_bilateral_dual_cot = True
        logger.info("Loaded Frontier Bilateral Dual-CoT Precision Champion from %s (prm_threshold=0.38)", bilateral_pt.name)"""

if old_bilateral in content_lf:
    content_lf = content_lf.replace(old_bilateral, new_bilateral)
    print("Replaced bilateral engine block.")

launch_path.write_text(content_lf, encoding="utf-8")
print("Finished patching launch_7day_visual_master_champion.py!")
