from pathlib import Path

path = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = path.read_text(encoding="utf-8")

# 1. Update Value Zone parameters to match the proven 88% WR simulation
old_guards = """        if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
            # Uptrend: strictly forbid counter-trend SELL unless genuine counter-punch
            if net_dir == -1 and not is_cp:
                net_dir = 0
            # BUY Value Zone Pullback & Anti-Top Guard:
            elif net_dir == 1 and not is_cp:
                in_value_zone = p_vs_e20_val <= 0.40
                not_at_top = rel_pos_60_val <= 0.68  # Anti-Top Longs: forbid buying upper 32% of 60-bar range
                rejection_confirmed = body_val >= 0.0 or lw_val >= 0.20
                if not (in_value_zone and not_at_top and rejection_confirmed):
                    net_dir = 0
        elif ema_diff_val < 0.0 and p_vs_e50_val < 0.0:
            # Downtrend: strictly forbid counter-trend BUY unless genuine counter-punch
            if net_dir == 1 and not is_cp:
                net_dir = 0
            # SELL Value Zone Pullback & Anti-Bottom Guard:
            elif net_dir == -1 and not is_cp:
                in_value_zone = p_vs_e20_val >= -0.40
                not_at_bottom = rel_pos_60_val >= 0.32  # Anti-Bottom Shorts: forbid selling lower 32% of 60-bar range
                rejection_confirmed = body_val <= 0.0 or uw_val >= 0.20
                if not (in_value_zone and not_at_bottom and rejection_confirmed):
                    net_dir = 0
        else:
            # Choppy / mixed regime: forbid unanchored entries unless genuine counter-punch
            if not is_cp:
                net_dir = 0"""

new_guards = """        if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
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
            # Choppy / mixed EMA regime: permit entry only with strong neural margin (>= 0.08)
            if not is_cp and abs(diff) < 0.08:
                net_dir = 0"""

assert old_guards in content, "old_guards not found"
content = content.replace(old_guards, new_guards, 1)

path.write_text(content, encoding="utf-8")
print("Successfully patched mythos_mt5_policy.py")
