path = 'research_reasoning_world_class/src/mythos_mt5_policy.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

import re
old_arb_regex = re.compile(r"# Tier 2 vs Tier 1 Selection criteria.*?entry_mode = \"SCALP_SELL\"", re.DOTALL)

new_arbitration = """# Tier 2 vs Tier 1 Selection criteria
        is_tier2_sniper_buy = bool(prm >= 0.85 and conf >= 0.40 and (is_sniper or margin >= 0.008) and not is_macro_downtrend)
        is_tier2_sniper_sell = bool(prm >= 0.85 and conf >= 0.40 and (is_sniper or margin <= -0.008) and not is_macro_uptrend)

        # PRIORITY ENTRY ARBITRATION LADDER (When Flat)
        is_bull_entry = False
        is_bear_entry = False
        entry_mode = "HOLD"

        if active_positions == 0:
            # PRIORITY 1: Counter-Trap Sniper Inversion (Highest Expected Value & 3:1 R:R)
            if is_counter_trap_buy and bull_momentum_confirmed and not is_falling_knife:
                is_bull_entry = True
                entry_mode = "COUNTER_TRAP_BUY"
            elif is_counter_trap_sell and bear_momentum_confirmed and not is_rocket:
                is_bear_entry = True
                entry_mode = "COUNTER_TRAP_SELL"

            # PRIORITY 2: Tier 2 Institutional Sniper Entry (Multi-Timeframe Trend Alignment)
            elif is_bull_direction and bull_momentum_confirmed and not is_falling_knife and is_tier2_sniper_buy:
                is_bull_entry = True
                entry_mode = "SNIPER_BUY"
            elif (not is_bull_direction) and bear_momentum_confirmed and not is_rocket and is_tier2_sniper_sell:
                is_bear_entry = True
                entry_mode = "SNIPER_SELL"

            # PRIORITY 3: High-Quality Trend Scalp (Strict Positive Margin & No Macro Contradiction)
            elif is_bull_direction and bull_momentum_confirmed and not is_falling_knife and margin >= 0.008 and conf >= 0.40 and not is_macro_downtrend:
                is_bull_entry = True
                entry_mode = "SCALP_BUY"
            elif (not is_bull_direction) and bear_momentum_confirmed and not is_rocket and margin <= -0.008 and conf >= 0.40 and not is_macro_uptrend:
                is_bear_entry = True
                entry_mode = "SCALP_SELL\""""

assert old_arb_regex.search(content) is not None, "Regex match failed"
content = old_arb_regex.sub(new_arbitration.strip(), content, count=1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Arbitration ladder calibrated successfully!")
