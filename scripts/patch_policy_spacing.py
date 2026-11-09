p = 'research_reasoning_world_class/src/mythos_mt5_policy.py'
with open(p, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
target_found = False
for line in lines:
    if 'min_spacing_bars = int(getattr(self, "min_spacing_bars", 3))' in line:
        new_lines.append('        min_spacing_bars = int(getattr(self, "min_spacing_bars", 8))\n')
        target_found = True
    elif 'elif self._last_entry_price is not None and abs(self.current_price - self._last_entry_price) < 0.50 * self.current_atr:' in line:
        new_lines.append(line)
        new_lines.append('        elif active_positions > 0 and self._last_entry_price is not None:\n')
        new_lines.append('            # Anti-falling knife guard: do not open concurrent positions if price is moving against open position\n')
        new_lines.append('            if net_dir == 1 and self.current_price < self._last_entry_price - 0.25 * self.current_atr:\n')
        new_lines.append('                spacing_ok = False\n')
        new_lines.append('            elif net_dir == -1 and self.current_price > self._last_entry_price + 0.25 * self.current_atr:\n')
        new_lines.append('                spacing_ok = False\n')
    else:
        new_lines.append(line)

if target_found:
    with open(p, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    print("Successfully patched policy spacing & anti-cascading guard!")
else:
    print("Target line not found!")
