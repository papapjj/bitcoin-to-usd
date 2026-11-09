import sys, os
from pathlib import Path

ROOT = Path.cwd()
policy_path = ROOT / "research_reasoning_world_class" / "src" / "mythos_mt5_policy.py"

with open(policy_path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add rolling deque initialization to __init__ if not already present
if "self._bull_score_hist = deque(maxlen=120)" not in text:
    old_init = "self._last_reasoning_z: Optional[torch.Tensor] = None"
    new_init = """self._last_reasoning_z: Optional[torch.Tensor] = None
        from collections import deque
        self._bull_score_hist = deque(maxlen=120)
        self._bear_score_hist = deque(maxlen=120)"""
    if old_init in text:
        text = text.replace(old_init, new_init, 1)
        print("Patched __init__ with rolling score deques!")
    else:
        print("Warning: old_init not found in text!")

# 2. Patch act_nam_tac_nhan signal resolution
old_block = """        net_dir = action
        # Pure Neural Autonomy: STRICTLY DISABLE LEAK TO DEGRADED BASE POLICY!
        # If bilateral dual-CoT is active, trust its dialectical arbiter verdict directly
        if getattr(self.pipeline, "use_bilateral_dual_cot", False):
            pass  # Bilateral Dual-CoT & Dialectical Arbiter verdict is authoritative"""

new_block = """        # Adaptive Online Rolling Z-Score Dialectical Arbiter
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
                elif b_z >= 0.20 and (b_z - r_z) >= 0.15:
                    net_dir = 1
                elif r_z >= 0.20 and (r_z - b_z) >= 0.15:
                    net_dir = -1
                else:
                    net_dir = 0
            else:
                net_dir = int(tick_result.action) if is_cp else 0"""

if old_block in text:
    text = text.replace(old_block, new_block, 1)
    print("Patched act_nam_tac_nhan signal resolution with rolling Z-scores!")
else:
    print("Warning: old_block not found in text!")

# 3. Patch trend alignment to strictly forbid counter-trend entries (unless counter-punch)
old_trend = """            if ma20 > ma50 and self.current_price > ma50:
                # Strong Uptrend: forbid SELL (unless price has broken below ma20 on confirmed sweep)
                if net_dir == -1 and (not getattr(tick_result, 'is_counter_punch', False) or self.current_price > ma20):
                    net_dir = 0
            elif ma20 < ma50 and self.current_price < ma50:
                # Strong Downtrend: forbid BUY (unless price has reclaimed above ma20 on confirmed spring)
                if net_dir == 1 and (not getattr(tick_result, 'is_counter_punch', False) or self.current_price < ma20):
                    net_dir = 0"""

new_trend = """            if ma20 > ma50:
                # Uptrend: strictly forbid counter-trend SELL unless genuine counter-punch
                if net_dir == -1 and not is_cp:
                    net_dir = 0
            elif ma20 < ma50:
                # Downtrend: strictly forbid counter-trend BUY unless genuine counter-punch
                if net_dir == 1 and not is_cp:
                    net_dir = 0"""

if old_trend in text:
    text = text.replace(old_trend, new_trend, 1)
    print("Patched trend alignment guard!")
else:
    print("Warning: old_trend not found in text!")

# 4. Patch default scalp SL and TP for SELL to match BUY symmetrically
old_sell = """                traces_list[1] = self._build_entry_trace(
                    traces_list[1],
                    action_idx=2,
                    dynamic_risk=kelly_risk,
                    sl_atr=getattr(self, 'scalp_sl_atr', 5.0),
                    tp_rr=getattr(self, 'scalp_tp_rr', 1.50),
                    trailing_atr=getattr(self, 'trailing_atr', 2.50),
                    prm_score=prm,
                    confidence=conf,
                )"""

new_sell = """                traces_list[1] = self._build_entry_trace(
                    traces_list[1],
                    action_idx=2,
                    dynamic_risk=kelly_risk,
                    sl_atr=getattr(self, 'scalp_sl_atr', 2.0),
                    tp_rr=getattr(self, 'scalp_tp_rr', 1.00),
                    trailing_atr=getattr(self, 'trailing_atr', 0.75),
                    prm_score=prm,
                    confidence=conf,
                )"""

if old_sell in text:
    text = text.replace(old_sell, new_sell, 1)
    print("Patched symmetrical scalp parameters for SELL!")
else:
    print("Warning: old_sell not found in text!")

with open(policy_path, "w", encoding="utf-8") as f:
    f.write(text)

print("Saved updated mythos_mt5_policy.py successfully!")
