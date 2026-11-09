with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'r', encoding='utf-8') as f:
    code = f.read()

target = """            elif ma20 < ma50 and self.current_price < ma50:
                # Strong Downtrend: forbid BUY (unless price has reclaimed above ma20 on confirmed spring)
                if net_dir == 1 and (not getattr(tick_result, 'is_counter_punch', False) or self.current_price < ma20):
                    net_dir = 0

        # Neural Confidence and Volatility Regime Filter"""

sniper_patch = """            elif ma20 < ma50 and self.current_price < ma50:
                # Strong Downtrend: forbid BUY (unless price has reclaimed above ma20 on confirmed spring)
                if net_dir == 1 and (not getattr(tick_result, 'is_counter_punch', False) or self.current_price < ma20):
                    net_dir = 0

        # Sniper Timing Confirmation (Anti-MAE & Anti-Premature SL):
        # Prevents buying into free-falling red knives and shorting into vertical pumps
        if net_dir != 0 and len(valid_closes) >= 1 and not getattr(tick_result, 'is_counter_punch', False):
            cur_open = float(self.m1_buffer[-1, 0])
            cur_high = float(self.m1_buffer[-1, 1])
            cur_low = float(self.m1_buffer[-1, 2])
            cur_close = float(self.m1_buffer[-1, 3])
            cur_atr = max(1.0, float(self.current_atr))
            if cur_open > 0 and cur_high > 0 and cur_low > 0 and cur_close > 0:
                if net_dir == 1:
                    # BUY: requires candle close >= open (green bar) OR absorption wick (close - low >= 0.20 ATR)
                    candle_green = cur_close >= cur_open
                    lower_wick = cur_close - cur_low
                    if not (candle_green or lower_wick >= 0.20 * cur_atr):
                        net_dir = 0
                elif net_dir == -1:
                    # SELL: requires candle close <= open (red bar) OR rejection wick (high - close >= 0.20 ATR)
                    candle_red = cur_close <= cur_open
                    upper_wick = cur_high - cur_close
                    if not (candle_red or upper_wick >= 0.20 * cur_atr):
                        net_dir = 0

        # Neural Confidence and Volatility Regime Filter"""

if target in code:
    code = code.replace(target, sniper_patch)
    with open('research_reasoning_world_class/src/mythos_mt5_policy.py', 'w', encoding='utf-8') as f:
        f.write(code)
    print("Successfully patched mythos_mt5_policy.py with Sniper Timing Confirmation!")
else:
    print("Target block not found, check target string")
