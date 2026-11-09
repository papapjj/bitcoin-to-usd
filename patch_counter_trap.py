path = 'research_reasoning_world_class/src/mythos_mt5_policy.py'
with open(path, 'r', encoding='utf-8') as f:
    text = f.read()

old_ct = """        if active_positions == 0:
            # PRIORITY 1: Counter-Trap Sniper Inversion (Highest Expected Value & 3:1 R:R)
            if is_counter_trap_buy and bull_momentum_confirmed and not is_falling_knife:
                is_bull_entry = True
                entry_mode = "COUNTER_TRAP_BUY"
            elif is_counter_trap_sell and bear_momentum_confirmed and not is_rocket:
                is_bear_entry = True
                entry_mode = "COUNTER_TRAP_SELL\""""

new_ct = """        if active_positions == 0:
            # PRIORITY 1: Counter-Trap Sniper Inversion (Highest Expected Value & 3:1 R:R)
            if is_counter_trap_buy and bull_momentum_confirmed and not is_falling_knife and not is_macro_downtrend:
                is_bull_entry = True
                entry_mode = "COUNTER_TRAP_BUY"
            elif is_counter_trap_sell and bear_momentum_confirmed and not is_rocket and not is_macro_uptrend:
                is_bear_entry = True
                entry_mode = "COUNTER_TRAP_SELL\""""

if old_ct in text:
    text = text.replace(old_ct, new_ct)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)
    print("Macro trend alignment applied to Counter-Trap successfully!")
else:
    print("Target string not found!")
