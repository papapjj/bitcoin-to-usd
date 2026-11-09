p = 'research_reasoning_world_class/src/mythos_mt5_policy.py'
with open(p, 'r', encoding='utf-8') as f:
    text = f.read()

old_block = """        if bars_since < min_spacing_bars:
            spacing_ok = False
        elif self._last_entry_price is not None and abs(self.current_price - self._last_entry_price) < 0.50 * self.current_atr:
        elif active_positions > 0 and self._last_entry_price is not None:
            # Anti-falling knife guard: do not open concurrent positions if price is moving against open position
            if net_dir == 1 and self.current_price < self._last_entry_price - 0.25 * self.current_atr:
                spacing_ok = False
            elif net_dir == -1 and self.current_price > self._last_entry_price + 0.25 * self.current_atr:
                spacing_ok = False
            spacing_ok = False"""

new_block = """        if bars_since < min_spacing_bars:
            spacing_ok = False
        elif self._last_entry_price is not None and abs(self.current_price - self._last_entry_price) < 0.50 * self.current_atr:
            spacing_ok = False
        elif active_positions > 0 and self._last_entry_price is not None:
            # Anti-falling knife guard: do not open concurrent positions if price is moving against open position
            if net_dir == 1 and self.current_price < self._last_entry_price - 0.25 * self.current_atr:
                spacing_ok = False
            elif net_dir == -1 and self.current_price > self._last_entry_price + 0.25 * self.current_atr:
                spacing_ok = False"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Fixed indentation in mythos_mt5_policy.py")
else:
    print("old_block not found")
