import re
from pathlib import Path

policy_path = Path("research_reasoning_world_class/src/mythos_mt5_policy.py")
content = policy_path.read_text(encoding="utf-8")

# 1. Add _last_exit_bar and post_exit_cooldown_bars to __init__
pattern1 = r"(self\._last_entry_bar: int = -999\r?\n)"
replacement1 = r"\1        self._last_exit_bar: int = -999\n        self.post_exit_cooldown_bars: int = 3\n"
if "self._last_exit_bar" not in content:
    content = re.sub(pattern1, replacement1, content, count=1)
    print("1. Added _last_exit_bar to __init__")

# 2. Add _last_exit_bar to reset_runtime_state
pattern2 = r"(self\._direction_cooldown_until = \{1: 0, -1: 0\}\r?\n)"
replacement2 = r"\1        self._last_exit_bar = -999\n        self.post_exit_cooldown_bars = 3\n"
if "self._direction_cooldown_until = {1: 0, -1: 0}\n        self._last_exit_bar = -999" not in content and "self._direction_cooldown_until = {1: 0, -1: 0}\r\n        self._last_exit_bar = -999" not in content:
    content = re.sub(pattern2, replacement2, content, count=1)
    print("2. Added _last_exit_bar to reset_runtime_state")

# 3. Update set_account_context to detect position closures
target_acct = '''    def set_account_context(self, *args: Any, **kwargs: Any) -> None:
        if args:
            self._current_account_context = args[0]
            if len(args) > 1:
                pipe = args[1]
                self._current_positions_pipe = pipe
                if isinstance(pipe, str):
                    self._current_active_positions = len([p for p in pipe.strip().split("|") if p.strip()]) if pipe.strip() else 0
                elif isinstance(pipe, (list, tuple)):
                    self._current_active_positions = len(pipe)
        hook = getattr(self.base_policy, "set_account_context", None)
        if callable(hook):
            hook(*args, **kwargs)'''

new_acct = '''    def set_account_context(self, *args: Any, **kwargs: Any) -> None:
        if args:
            self._current_account_context = args[0]
            if len(args) > 1:
                pipe = args[1]
                self._current_positions_pipe = pipe
                prev_pos = getattr(self, "_current_active_positions", None)
                if isinstance(pipe, str):
                    new_pos = len([p for p in pipe.strip().split("|") if p.strip()]) if pipe.strip() else 0
                elif isinstance(pipe, (list, tuple)):
                    new_pos = len(pipe)
                else:
                    new_pos = 0
                if prev_pos is not None and prev_pos > new_pos:
                    self._last_exit_bar = getattr(self, "seq_counter", 0)
                self._current_active_positions = new_pos
        hook = getattr(self.base_policy, "set_account_context", None)
        if callable(hook):
            hook(*args, **kwargs)'''

# Normalize line endings for replacement
content = content.replace("\r\n", "\n")
if target_acct in content:
    content = content.replace(target_acct, new_acct)
    print("3. Updated set_account_context")

# 4. Update _sync_feature_audit to record _last_exit_bar
target_sync = "self.closed_trades += 1"
new_sync = "self.closed_trades += 1\n                                self._last_exit_bar = self.seq_counter"
if target_sync in content and "_last_exit_bar = self.seq_counter" not in content:
    content = content.replace(target_sync, new_sync, 1)
    print("4. Updated _sync_feature_audit")

# 5. Replace rolling Z-score logic with principled conviction margin & PRM
old_bilateral_block = '''        # Adaptive Online Rolling Z-Score Dialectical Arbiter
        is_cp = getattr(tick_result, 'is_counter_punch', False)
        if getattr(self.pipeline, "use_bilateral_dual_cot", False):
            if not hasattr(self, "_bull_score_hist"):
                from collections import deque
                self._bull_score_hist = deque(maxlen=120)
                self._bear_score_hist = deque(maxlen=120)
            self._bull_score_hist.append(b_s)
            self._bear_score_hist.append(r_s)
            
            if len(self._bull_score_hist) >= 20:
                mb = float(np.mean(self._bull_score_hist))
                sb = max(0.01, float(np.std(self._bull_score_hist)))
                mr = float(np.mean(self._bear_score_hist))
                sr = max(0.01, float(np.std(self._bear_score_hist)))
                b_z = (b_s - mb) / sb
                r_z = (r_s - mr) / sr
                
                if is_cp:
                    net_dir = int(tick_result.action)
                    conf = max(conf, 0.88)
                elif b_z >= 0.20 and (b_z - r_z) >= 0.15:
                    net_dir = 1
                    conf = max(conf, b_s, 0.75)
                elif r_z >= 0.20 and (r_z - b_z) >= 0.15:
                    net_dir = -1
                    conf = max(conf, r_s, 0.75)
                else:
                    net_dir = 0
            else:
                net_dir = int(tick_result.action) if is_cp else 0'''

new_bilateral_block = '''        # Bilateral Dual-CoT Metacognitive Arbiter Integration
        is_cp = getattr(tick_result, 'is_counter_punch', False)
        if getattr(self.pipeline, "use_bilateral_dual_cot", False):
            arb_act = int(tick_result.action)
            diff = b_s - r_s
            prm_gate = getattr(self, "prm_threshold", 0.33)
            min_margin = getattr(self, "min_margin_threshold", 0.05)
            
            if is_cp:
                net_dir = arb_act
                conf = max(conf, 0.88)
            # Valid BUY: Bull conviction beats Bear by >= margin, b_s >= 0.48, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif diff >= min_margin and b_s >= 0.48 and prm >= prm_gate and (arb_act == 1 or diff >= 0.08):
                net_dir = 1
                conf = max(conf, b_s)
            # Valid SELL: Bear conviction beats Bull by >= margin, r_s >= 0.48, PRM >= prm_gate, and Arbiter agrees or strong margin
            elif -diff >= min_margin and r_s >= 0.48 and prm >= prm_gate and (arb_act == -1 or -diff >= 0.08):
                net_dir = -1
                conf = max(conf, r_s)
            else:
                net_dir = 0'''

if old_bilateral_block in content:
    content = content.replace(old_bilateral_block, new_bilateral_block)
    print("5. Replaced rolling Z-score with calibrated conviction margin & PRM")

# 6. Trend, Value Zone Pullback, Anti-Top/Bottom, and Post-Exit Cooldown
old_trend_block = '''        # Authentic M1 Multi-Bar Trend Alignment Guard (from 60-bar rolling memory):
        # Universal invariant across all models (including Bilateral Dual-CoT):
        # In Uptrend (current_price > ma50 and ma20 > ma50), strictly forbid counter-trend shorts!
        # In Downtrend (current_price < ma50 and ma20 < ma50), strictly forbid catching falling knives!
        closes = self.m1_buffer[:, 3]
        valid_closes = closes[closes > 0]
        if len(valid_closes) >= 30:
            ma20 = float(np.mean(valid_closes[-20:]))
            ma50 = float(np.mean(valid_closes[-min(len(valid_closes), 50):]))
            if ma20 > ma50:
                # Uptrend: strictly forbid counter-trend SELL unless genuine counter-punch
                if net_dir == -1 and not is_cp:
                    net_dir = 0
            elif ma20 < ma50:
                # Downtrend: strictly forbid counter-trend BUY unless genuine counter-punch
                if net_dir == 1 and not is_cp:
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
                        net_dir = 0'''

new_trend_block = '''        # Authentic M1 Multi-Bar Trend, Value Zone Pullback, & Anti-Top/Bottom Guard:
        closes = self.m1_buffer[:, 3]
        valid_closes = closes[closes > 0]
        cur_atr = max(1.0, float(self.current_atr))
        cur_open = float(self.m1_buffer[-1, 0])
        cur_high = float(self.m1_buffer[-1, 1])
        cur_low = float(self.m1_buffer[-1, 2])
        cur_close = float(self.m1_buffer[-1, 3])

        if len(valid_closes) >= 30:
            ma20 = float(np.mean(valid_closes[-20:]))
            ma50 = float(np.mean(valid_closes[-min(len(valid_closes), 50):]))
            
            # Anti-Top & Anti-Bottom Guard from 60-bar rolling memory:
            valid_highs = self.m1_buffer[self.m1_buffer[:, 1] > 0, 1]
            valid_lows = self.m1_buffer[self.m1_buffer[:, 2] > 0, 2]
            if len(valid_highs) >= 15:
                max_h60 = float(np.max(valid_highs[-60:]))
                min_l60 = float(np.min(valid_lows[-60:]))
                rng60 = max(1.0, max_h60 - min_l60)
                rel_pos_60 = (self.current_price - min_l60) / rng60
            else:
                rel_pos_60 = 0.50

            if ma20 > ma50 and self.current_price > ma50:
                # Uptrend: strictly forbid counter-trend SELL unless genuine counter-punch
                if net_dir == -1 and not is_cp:
                    net_dir = 0
                # BUY Value Zone Pullback & Anti-Top Guard:
                elif net_dir == 1 and not is_cp:
                    in_value_zone = (cur_low <= ma20 + 0.35 * cur_atr) and (cur_close <= ma20 + 0.50 * cur_atr)
                    not_at_top = rel_pos_60 <= 0.85
                    if not (in_value_zone and not_at_top):
                        net_dir = 0
            elif ma20 < ma50 and self.current_price < ma50:
                # Downtrend: strictly forbid counter-trend BUY unless genuine counter-punch
                if net_dir == 1 and not is_cp:
                    net_dir = 0
                # SELL Value Zone Pullback & Anti-Bottom Guard:
                elif net_dir == -1 and not is_cp:
                    in_value_zone = (cur_high >= ma20 - 0.35 * cur_atr) and (cur_close >= ma20 - 0.50 * cur_atr)
                    not_at_bottom = rel_pos_60 >= 0.15
                    if not (in_value_zone and not_at_bottom):
                        net_dir = 0
            else:
                # Choppy/mixed regime: forbid unanchored entries unless genuine counter-punch
                if not is_cp:
                    net_dir = 0

        # Post-Exit Cooldown (Anti-Chasing): Pause after any position closes to avoid chasing exhaustion wicks
        bars_since_exit = self.seq_counter - getattr(self, "_last_exit_bar", -999)
        if net_dir != 0 and bars_since_exit < getattr(self, "post_exit_cooldown_bars", 3):
            net_dir = 0

        # Sniper Timing Confirmation (Anti-MAE & Anti-Premature SL):
        if net_dir != 0 and len(valid_closes) >= 1 and not getattr(tick_result, 'is_counter_punch', False):
            if cur_open > 0 and cur_high > 0 and cur_low > 0 and cur_close > 0:
                if net_dir == 1:
                    candle_green = cur_close >= cur_open
                    lower_wick = cur_close - cur_low
                    if not (candle_green or lower_wick >= 0.20 * cur_atr):
                        net_dir = 0
                elif net_dir == -1:
                    candle_red = cur_close <= cur_open
                    upper_wick = cur_high - cur_close
                    if not (candle_red or upper_wick >= 0.20 * cur_atr):
                        net_dir = 0'''

if old_trend_block in content:
    content = content.replace(old_trend_block, new_trend_block)
    print("6. Updated Trend, Value Zone Pullback, Anti-Top/Bottom, and Post-Exit Cooldown")

policy_path.write_text(content, encoding="utf-8")
print("Done patching mythos_mt5_policy.py!")
