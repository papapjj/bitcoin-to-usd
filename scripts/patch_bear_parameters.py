import re

file_path = 'research_reasoning_world_class/src/mythos_mt5_policy.py'
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

target = """        elif is_bear_entry:
            self.trade_count += 1
            self._last_entry_price = self.current_price
            self._last_entry_bar = self.seq_counter
            sn_sl = self.sniper_sl_atr if getattr(self, "sniper_sl_atr", None) is not None else 3.0
            sn_tp = self.sniper_tp_rr if getattr(self, "sniper_tp_rr", None) is not None else 2.50
            if entry_mode == "SNIPER_SELL":
                traces_list[1] = self._build_entry_trace(
                    traces_list[1],
                    action_idx=2,
                    dynamic_risk=kelly_risk,
                    sl_atr=sn_sl,
                    tp_rr=sn_tp,
                    trailing_atr=0.75,
                    prm_score=prm,
                    confidence=conf,
                )
                traces_list[0] = self._enforce_hold(traces_list[0], reason="SNIPER_SELL_EXCLUSIVE")
                mode_tag = "SNIPER_SELL [Neural Tier 2]"
                verdict_str = f"{mode_tag} (Kelly={kelly_risk:.2%}, PRM={prm:.2f}, Margin={margin:+.2f}, Conf={conf:.1%})"
            else:
                traces_list[1] = self._build_entry_trace(
                    traces_list[1],
                    action_idx=2,
                    dynamic_risk=kelly_risk,
                    sl_atr=getattr(self, 'scalp_sl_atr', 2.0),
                    tp_rr=getattr(self, 'scalp_tp_rr', 2.50),
                    trailing_atr=getattr(self, 'trailing_atr', 0.75),
                    prm_score=prm,
                    confidence=conf,
                )
                traces_list[0] = self._enforce_hold(traces_list[0], reason="SCALP_SELL_EXCLUSIVE")
                mode_tag = "SCALP_SELL [Neural Tier 1]"
                verdict_str = f"{mode_tag} (Kelly={kelly_risk:.2%}, PRM={prm:.2f}, Margin={margin:+.2f}, Conf={conf:.1%})" """

replacement = """        elif is_bear_entry:
            self.trade_count += 1
            self._last_entry_price = self.current_price
            self._last_entry_bar = self.seq_counter
            sn_sl = float(getattr(self, "sniper_sl_atr", 5.0))
            sn_tp = float(getattr(self, "sniper_tp_rr", 2.00))
            sn_trail = float(getattr(self, "trailing_atr", 2.50))
            if entry_mode == "SNIPER_SELL":
                traces_list[1] = self._build_entry_trace(
                    traces_list[1],
                    action_idx=2,
                    dynamic_risk=kelly_risk,
                    sl_atr=sn_sl,
                    tp_rr=sn_tp,
                    trailing_atr=sn_trail,
                    prm_score=prm,
                    confidence=conf,
                )
                traces_list[0] = self._enforce_hold(traces_list[0], reason="SNIPER_SELL_EXCLUSIVE")
                mode_tag = "SNIPER_SELL [Neural Tier 2]"
                verdict_str = f"{mode_tag} (Kelly={kelly_risk:.2%}, PRM={prm:.2f}, Margin={margin:+.2f}, Conf={conf:.1%})"
            else:
                traces_list[1] = self._build_entry_trace(
                    traces_list[1],
                    action_idx=2,
                    dynamic_risk=kelly_risk,
                    sl_atr=getattr(self, 'scalp_sl_atr', 5.0),
                    tp_rr=getattr(self, 'scalp_tp_rr', 1.50),
                    trailing_atr=getattr(self, 'trailing_atr', 2.50),
                    prm_score=prm,
                    confidence=conf,
                )
                traces_list[0] = self._enforce_hold(traces_list[0], reason="SCALP_SELL_EXCLUSIVE")
                mode_tag = "SCALP_SELL [Neural Tier 1]"
                verdict_str = f"{mode_tag} (Kelly={kelly_risk:.2%}, PRM={prm:.2f}, Margin={margin:+.2f}, Conf={conf:.1%})" """

target_clean = "\n".join(line.strip() for line in target.strip().splitlines())
lines = text.splitlines()

# Search block of lines
found_idx = -1
for i in range(len(lines) - 30):
    chunk = "\n".join(line.strip() for line in lines[i:i+34])
    if target_clean in chunk:
        found_idx = i
        break

if found_idx != -1:
    print(f"Found target block at line {found_idx + 1}")
    new_lines = lines[:found_idx] + replacement.strip().splitlines() + lines[found_idx+34:]
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(new_lines) + "\n")
    print("SUCCESS: Updated bear entry parameters.")
else:
    print("ERROR: Target block not found.")
