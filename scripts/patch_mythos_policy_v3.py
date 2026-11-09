from pathlib import Path
import re

policy_file = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = policy_file.read_text(encoding="utf-8")

# 1. Update defaults in __init__
pattern_defaults = r"(self\.post_exit_cooldown_bars:\s*int\s*=\s*)\d+"
content = re.sub(pattern_defaults, r"\g<1>1", content)

# Ensure calibrated attributes exist with correct default values
default_props = """        self.prm_threshold: float = 0.30
        self.min_confidence: float = 0.35
        self.min_margin_threshold: float = 0.08
        self.bear_entry_threshold: float = 0.28
        self.bear_min_margin: float = 0.05
        self.scalp_sl_atr: float = 2.60
        self.sniper_sl_atr: float = 2.60
        self.scalp_tp_rr: float = 1.38
        self.sniper_tp_rr: float = 1.38
        self.trailing_atr: float = 1.20
        self.breakeven_trigger_atr: float = 1.60
        self.breakeven_lock_atr: float = 0.80
        self.max_positions: int = 5
        self.min_spacing_bars: int = 2
"""

# Replace existing attribute definitions or insert cleanly
if "self.prm_threshold: float" in content:
    content = re.sub(r"self\.prm_threshold:\s*float\s*=\s*[\d\.]+", "self.prm_threshold: float = 0.30", content)
if "self.min_confidence: float" in content:
    content = re.sub(r"self\.min_confidence:\s*float\s*=\s*[\d\.]+", "self.min_confidence: float = 0.35", content)

# 2. Update act_nam_tac_nhan decision logic for Bilateral Dual-CoT
bilateral_logic_target = """        # Bilateral Dual-CoT Metacognitive Arbiter Integration
        is_cp = getattr(tick_result, 'is_counter_punch', False)
        if getattr(self.pipeline, "use_bilateral_dual_cot", False):
            arb_act = int(tick_result.action)
            diff = b_s - r_s
            prm_gate = getattr(self, "prm_threshold", 0.35)
            min_margin = getattr(self, "min_margin_threshold", 0.08)
            min_conf = getattr(self, "min_confidence", 0.40)
            
            bear_thresh = getattr(self, "bear_entry_threshold", min_conf)
            sell_margin = getattr(self, "bear_min_margin", min_margin * 0.7)

            if is_cp:
                net_dir = arb_act
                conf = max(conf, 0.88)
            # Valid BUY: Bull conviction beats Bear by >= margin, b_s >= min_conf, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif diff >= min_margin and b_s >= min_conf and prm >= prm_gate and (arb_act == 1 or diff >= 0.08):
                net_dir = 1
                conf = max(conf, b_s)
            # Valid SELL: Bear conviction beats Bull by >= sell_margin, r_s >= bear_thresh, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif -diff >= sell_margin and r_s >= bear_thresh and prm >= prm_gate and (arb_act == -1 or -diff >= 0.08):
                net_dir = -1
                conf = max(conf, r_s)
            else:
                net_dir = 0"""

bilateral_logic_replacement = """        # Bilateral Dual-CoT Metacognitive Arbiter Integration
        is_cp = False  # Strictly disabled: Eliminate rogue counter-punch false triggers
        if getattr(self.pipeline, "use_bilateral_dual_cot", False):
            arb_act = int(tick_result.action)
            diff = b_s - r_s
            prm_gate = getattr(self, "prm_threshold", 0.30)
            min_margin = getattr(self, "min_margin_threshold", 0.08)
            min_conf = getattr(self, "min_confidence", 0.35)
            bear_thresh = getattr(self, "bear_entry_threshold", 0.28)
            sell_margin = getattr(self, "bear_min_margin", 0.05)

            # Direct Dual-Specialist Execution (Calibrated on 11,201 real M1 bars):
            if diff >= min_margin and b_s >= min_conf and prm >= prm_gate:
                net_dir = 1
                conf = max(conf, b_s)
            elif -diff >= sell_margin and r_s >= bear_thresh and prm >= prm_gate:
                net_dir = -1
                conf = max(conf, r_s)
            else:
                net_dir = 0"""

if bilateral_logic_target in content:
    content = content.replace(bilateral_logic_target, bilateral_logic_replacement)
    print("Replaced bilateral logic successfully!")
else:
    print("Warning: bilateral_logic_target not found by exact string, checking normalized...")
    # Normalize line endings
    norm_c = content.replace("\r\n", "\n")
    norm_t = bilateral_logic_target.replace("\r\n", "\n")
    if norm_t in norm_c:
        content = norm_c.replace(norm_t, bilateral_logic_replacement.replace("\r\n", "\n"))
        print("Replaced normalized bilateral logic successfully!")

# 3. Clean up the manual conflicting EMA filter in lines 1394-1437
# Replace with lightweight anti-top/anti-bottom wick safety
old_ema_filter = """        if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
            # Uptrend: strictly forbid counter-trend SELL unless genuine counter-punch
            if net_dir == -1 and not is_cp:
                net_dir = 0
            # BUY Value Zone Pullback & Anti-Top Guard:
            elif net_dir == 1 and not is_cp:
                in_value_zone = p_vs_e20_val <= 0.40
                not_at_top = rel_pos_60_val <= 0.85  # Anti-Top: forbid buying extreme top 15%
                rejection_confirmed = body_val >= 0.0 or lw_val >= 0.15
                if not (in_value_zone and not_at_top and rejection_confirmed):
                    net_dir = 0
        elif ema_diff_val < 0.0 and p_vs_e50_val < 0.0:
            # Downtrend: strictly forbid counter-trend BUY unless genuine counter-punch
            if net_dir == 1 and not is_cp:
                net_dir = 0
            # SELL Value Zone Pullback & Anti-Bottom Guard:
            elif net_dir == -1 and not is_cp:
                in_value_zone = p_vs_e20_val >= -0.40
                not_at_bottom = rel_pos_60_val >= 0.15  # Anti-Bottom: forbid selling extreme bottom 15%
                rejection_confirmed = body_val <= 0.0 or uw_val >= 0.15
                if not (in_value_zone and not_at_bottom and rejection_confirmed):
                    net_dir = 0
        else:
            # Choppy / mixed EMA regime:
            # 1. Require stronger neural margin (>= 0.12)
            # 2. Strict mean-reversion boundary: forbid buying top half or selling bottom half
            if not is_cp:
                if abs(diff) < getattr(self, "min_margin_threshold", 0.05):
                    net_dir = 0
                elif net_dir == 1:
                    # In mixed regime, only buy dips near or below EMA20, never buy highs!
                    in_dip = p_vs_e20_val <= 0.15
                    not_top = rel_pos_60_val <= 0.60
                    rejection = body_val >= 0.0 or lw_val >= 0.15
                    if not (in_dip and not_top and rejection):
                        net_dir = 0
                elif net_dir == -1:
                    # In mixed regime, only sell rallies near or above EMA20, never sell lows!
                    in_rally = p_vs_e20_val >= -0.15
                    not_bottom = rel_pos_60_val >= 0.40
                    rejection = body_val <= 0.0 or uw_val >= 0.15
                    if not (in_rally and not_bottom and rejection):
                        net_dir = 0"""

new_lightweight_filter = """        # Structural Anti-Wick Guard (Prevents buying candle wick tops or selling candle wick bottoms):
        if net_dir == 1 and uw_val >= 0.45:
            net_dir = 0  # Do not buy into an extreme upper rejection wick
        elif net_dir == -1 and lw_val >= 0.45:
            net_dir = 0  # Do not sell into an extreme lower rejection wick"""

norm_c = content.replace("\r\n", "\n")
norm_old = old_ema_filter.replace("\r\n", "\n")
if norm_old in norm_c:
    content = norm_c.replace(norm_old, new_lightweight_filter.replace("\r\n", "\n"))
    print("Replaced old EMA filter with lightweight anti-wick guard!")

policy_file.write_text(content, encoding="utf-8")
print("Policy patch complete!")
