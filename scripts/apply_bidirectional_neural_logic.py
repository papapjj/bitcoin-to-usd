with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'r', encoding='utf-8') as f:
    text = f.read()

target = '''                min_margin = getattr(self, 'min_margin_threshold', 0.12)
                max_margin = getattr(self, 'max_margin_threshold', 0.92)
                prm_gate = getattr(self, 'prm_threshold', 0.30)
                diff = b_s - r_s
                if min_margin <= diff <= max_margin and prm >= prm_gate:
                    net_dir = 1
                elif min_margin <= -diff <= max_margin and prm >= prm_gate:
                    net_dir = -1
                else:
                    net_dir = 0
            else:
                net_dir = 0

        # Macro & Regime Directional Alignment Guard
        if getattr(self, "long_only", False) and net_dir == -1:
            net_dir = 0
        if getattr(self, "short_only", False) and net_dir == 1:
            net_dir = 0

        # Neural Confidence and Volatility Regime Filter
        min_conf = getattr(self, "min_confidence", 0.0)
        max_atr_pct = getattr(self, "max_atr_percentile", 1.0)
        atr_pct = float(base_obs_137[14]) if base_obs_137.size > 14 else 0.5
        min_atr = float(getattr(self, "min_atr_threshold", 0.0))
        if conf < min_conf or atr_pct > max_atr_pct or self.current_atr < min_atr:
            net_dir = 0'''

replacement = '''                min_margin = getattr(self, 'min_margin_threshold', 0.12)
                max_margin = getattr(self, 'max_margin_threshold', 0.95)
                prm_gate = getattr(self, 'prm_threshold', 0.30)
                diff = b_s - r_s
                bear_thresh = getattr(self, 'bear_entry_threshold', 0.65)
                bear_min_margin = getattr(self, 'bear_min_margin', 0.25)
                if min_margin <= diff <= max_margin and prm >= prm_gate:
                    net_dir = 1
                elif (r_s - b_s) >= bear_min_margin and r_s >= bear_thresh:
                    net_dir = -1
                else:
                    net_dir = 0
            else:
                net_dir = 0

        # Macro & Regime Directional Alignment Guard
        if getattr(self, "long_only", False) and net_dir == -1:
            net_dir = 0
        if getattr(self, "short_only", False) and net_dir == 1:
            net_dir = 0

        # Neural Confidence and Volatility Regime Filter
        min_conf = getattr(self, "min_confidence", 0.0)
        max_atr_pct = getattr(self, "max_atr_percentile", 1.0)
        atr_pct = float(base_obs_137[14]) if base_obs_137.size > 14 else 0.5
        min_atr = float(getattr(self, "min_atr_threshold", 0.0))
        eff_conf = conf if net_dir == 1 else (r_s if net_dir == -1 else 0.0)
        if (net_dir != 0 and eff_conf < min_conf) or atr_pct > max_atr_pct or self.current_atr < min_atr:
            net_dir = 0'''

text_norm = text.replace('\r\n', '\n')
target_norm = target.replace('\r\n', '\n')
replacement_norm = replacement.replace('\r\n', '\n')

if target_norm in text_norm:
    text_norm = text_norm.replace(target_norm, replacement_norm, 1)
    # Also update sniper check for sell:
    target_sell_sniper = 'elif is_sniper or prm >= 0.90:\n                    entry_mode = "SNIPER_SELL"'
    repl_sell_sniper = 'elif is_sniper or r_s >= 0.85:\n                    entry_mode = "SNIPER_SELL"'
    text_norm = text_norm.replace(target_sell_sniper, repl_sell_sniper, 1)
    
    with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'w', encoding='utf-8') as f:
        f.write(text_norm)
    print("SUCCESS: Successfully updated mythos_mt5_policy.py with bidirectional neural logic.")
else:
    print("ERROR: target_norm not found.")
