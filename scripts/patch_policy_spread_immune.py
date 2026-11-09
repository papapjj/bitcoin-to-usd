import re
from pathlib import Path

target_file = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = target_file.read_text(encoding="utf-8")

# 1. Patch _sync_feature_audit to re-center m1_buffer on first real bar_close
old_audit_sync = """                            if math.isfinite(bc) and bc > 100.0:
                                self.current_price = bc"""
new_audit_sync = """                            if math.isfinite(bc) and bc > 100.0:
                                if not getattr(self, "_m1_buffer_seeded", False):
                                    delta = bc - self.current_price
                                    self.m1_buffer[:, :4] += delta
                                    self._m1_buffer_seeded = True
                                self.current_price = bc"""

assert old_audit_sync in content, "old_audit_sync pattern not found"
content = content.replace(old_audit_sync, new_audit_sync, 1)

# 2. Patch lines 1286-1353: Trend, Value Zone Pullback, & Anti-Top/Bottom Guard from base_obs_137
old_guard_pat = re.compile(
    r"# Authentic M1 Multi-Bar Trend, Value Zone Pullback, & Anti-Top/Bottom Guard:.*?"
    r"# Neural Confidence and Volatility Regime Filter",
    re.DOTALL
)

new_guard_block = """# Authentic MT5 Trend, Value Zone Pullback, & Anti-Top/Bottom Guard from base_obs_137:
        cur_atr = max(1.0, float(self.current_atr))
        ema_diff_val = float(base_obs_137[11]) if base_obs_137.size > 11 else 0.0
        p_vs_e20_val = float(base_obs_137[26]) if base_obs_137.size > 26 else 0.0
        p_vs_e50_val = float(base_obs_137[27]) if base_obs_137.size > 27 else 0.0
        rel_pos_60_val = float(base_obs_137[13]) if base_obs_137.size > 13 else 0.50
        body_val = float(base_obs_137[7]) if base_obs_137.size > 7 else 0.0
        uw_val = float(base_obs_137[8]) if base_obs_137.size > 8 else 0.0
        lw_val = float(base_obs_137[9]) if base_obs_137.size > 9 else 0.0

        if ema_diff_val > 0.0 and p_vs_e50_val > 0.0:
            # Uptrend: strictly forbid counter-trend SELL unless genuine counter-punch
            if net_dir == -1 and not is_cp:
                net_dir = 0
            # BUY Value Zone Pullback & Anti-Top Guard:
            elif net_dir == 1 and not is_cp:
                in_value_zone = p_vs_e20_val <= 0.40
                not_at_top = rel_pos_60_val <= 0.85
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
                not_at_bottom = rel_pos_60_val >= 0.15
                rejection_confirmed = body_val <= 0.0 or uw_val >= 0.20
                if not (in_value_zone and not_at_bottom and rejection_confirmed):
                    net_dir = 0
        else:
            # Choppy / mixed regime: forbid unanchored entries unless genuine counter-punch
            if not is_cp:
                net_dir = 0

        # Post-Exit Cooldown (Anti-Chasing): Pause after any position closes to avoid chasing exhaustion wicks
        bars_since_exit = self.seq_counter - getattr(self, "_last_exit_bar", -999)
        if net_dir != 0 and bars_since_exit < getattr(self, "post_exit_cooldown_bars", 3):
            net_dir = 0

        # Neural Confidence and Volatility Regime Filter"""

m = old_guard_pat.search(content)
assert m is not None, "old_guard_pat not matched"
content = content[:m.start()] + new_guard_block + content[m.end() - len("# Neural Confidence and Volatility Regime Filter"):]

# 3. Patch scalp geometry defaults
content = content.replace("sl_atr=getattr(self, 'scalp_sl_atr', 2.0),", "sl_atr=getattr(self, 'scalp_sl_atr', 4.0),")
content = content.replace("tp_rr=getattr(self, 'scalp_tp_rr', 2.50),", "tp_rr=getattr(self, 'scalp_tp_rr', 0.85),")
content = content.replace("tp_rr=getattr(self, 'scalp_tp_rr', 1.00),", "tp_rr=getattr(self, 'scalp_tp_rr', 0.85),")
content = content.replace("trailing_atr=getattr(self, 'trailing_atr', 0.75),", "trailing_atr=getattr(self, 'trailing_atr', 1.50),")

target_file.write_text(content, encoding="utf-8")
print(f"Successfully patched {target_file}")
