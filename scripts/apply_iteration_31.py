import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

# 1. Update research_reasoning_world_class/src/mythos_mt5_policy.py
policy_path = ROOT / 'research_reasoning_world_class' / 'src' / 'mythos_mt5_policy.py'
with open(policy_path, 'r', encoding='utf-8') as f:
    pol_text = f.read()

# Clean up any duplicated CHOP guard block and install the upgraded Softmax Probabilities CHOP Guard
old_chop_block = '''        # Champion Market Regime / Sideway (CHOP) Guard (Milestone 2/3 Defense):
        detected_regime = None
        if getattr(self, "regime_detector", None) is not None and net_dir != 0:
            try:
                from self_growing_trader.regime_detector import MarketRegime
                reg_id, _ = self.regime_detector.predict_wire_features(base_obs_137)
                detected_regime = MarketRegime(reg_id)
            except Exception:
                detected_regime = None

        if detected_regime is not None and detected_regime.name == "CHOP":
            chop_gate = getattr(self, "chop_prm_threshold", 0.45)
            if prm < chop_gate:
                net_dir = 0  # Veto choppy breakout traps during sideways consolidation'''

new_chop_block = '''        # Champion Market Regime / Sideway (CHOP) Guard (Milestone 2/3 Softmax Defense):
        detected_regime = None
        regime_probs = None
        if getattr(self, "regime_detector", None) is not None and net_dir != 0:
            try:
                from self_growing_trader.regime_detector import MarketRegime
                reg_id, probs = self.regime_detector.predict_wire_features(base_obs_137)
                detected_regime = MarketRegime(reg_id)
                regime_probs = probs
            except Exception:
                detected_regime = None
                regime_probs = None

        # CHOP defense: if CHOP probability is elevated (>= 0.25) or argmax is CHOP:
        is_chop = (detected_regime is not None and detected_regime.name == "CHOP") or (regime_probs is not None and len(regime_probs) > 2 and regime_probs[2] >= 0.25)
        if is_chop and net_dir != 0:
            chop_gate = getattr(self, "chop_prm_threshold", 0.55)
            if prm < chop_gate:
                net_dir = 0  # VETO choppy breakout traps during sideways consolidation'''

# Replace all occurrences of old block
while old_chop_block in pol_text:
    pol_text = pol_text.replace(old_chop_block, new_chop_block)

# If not present via exact match, check and update
if 'regime_probs[2] >= 0.25' not in pol_text:
    # Anchor after Macro Trend Alignment Guard
    trend_guard = '''        # Macro Trend Alignment Guard (Strict Trend-Following: never fight the macro trend)
        if net_dir == -1 and (ema_diff_val > -0.05 or p_vs_e50_val > 0.10):
            net_dir = 0  # Strictly forbid shorting when trend is bullish (Bull Specialist owns market)
        elif net_dir == 1 and (ema_diff_val < 0.05 or p_vs_e50_val < -0.10):
            net_dir = 0  # Strictly forbid buying when trend is bearish (Bear Specialist owns market)'''
    assert trend_guard in pol_text, "trend_guard not found"
    pol_text = pol_text.replace(trend_guard, trend_guard + '\n\n' + new_chop_block)

# Upgrade concurrency spacing to allow pyramiding into winners
old_spacing = '''        if bars_since < min_spacing_bars:
            spacing_ok = False
        elif active_positions > 0 and self._last_entry_price is not None:
            if abs(self.current_price - self._last_entry_price) < 0.20 * self.current_atr:
                spacing_ok = False
            # Anti-falling knife guard: do not open concurrent positions if price is moving against open position
            elif net_dir == 1 and self.current_price < self._last_entry_price - 0.25 * self.current_atr:
                spacing_ok = False
            elif net_dir == -1 and self.current_price > self._last_entry_price + 0.25 * self.current_atr:
                spacing_ok = False'''

new_spacing = '''        if bars_since < min_spacing_bars:
            spacing_ok = False
        elif active_positions > 0 and self._last_entry_price is not None:
            # Allow pyramiding into winners when price moves with trend >= 0.15 ATR
            price_favorable = (net_dir == 1 and self.current_price >= self._last_entry_price + 0.15 * self.current_atr) or \
                              (net_dir == -1 and self.current_price <= self._last_entry_price - 0.15 * self.current_atr)
            if not price_favorable:
                spacing_ok = False'''

if old_spacing in pol_text:
    pol_text = pol_text.replace(old_spacing, new_spacing)

with open(policy_path, 'w', encoding='utf-8') as f:
    f.write(pol_text)
print("1. Successfully updated mythos_mt5_policy.py (Softmax CHOP Guard + Pyramiding concurrency)")

# 2. Update SelfGrowing_MT5_Trainer_EA.mq5
ea_path = ROOT / 'mt5_v2_runner' / 'MQL5' / 'Experts' / 'SelfGrowing_MT5_Trainer_EA.mq5'
with open(ea_path, 'r', encoding='utf-8') as f:
    ea_text = f.read()

ea_text = re.sub(r'input int MaxPositions\s*=\s*\d+;', 'input int MaxPositions = 4;', ea_text)
ea_text = re.sub(r'input double FixedSlAtr\s*=\s*[\d\.]+;', 'input double FixedSlAtr = 2.40;', ea_text)
ea_text = re.sub(r'input double FixedTpAtr\s*=\s*[\d\.]+;', 'input double FixedTpAtr = 3.20;', ea_text)
ea_text = re.sub(r'input double TrailActivationAtr\s*=\s*[\d\.]+;', 'input double TrailActivationAtr = 1.15;', ea_text)
ea_text = re.sub(r'input double TrailLockAtr\s*=\s*[\d\.]+;', 'input double TrailLockAtr = 0.95;', ea_text)

with open(ea_path, 'w', encoding='utf-8') as f:
    f.write(ea_text)
print('2. Successfully updated SelfGrowing_MT5_Trainer_EA.mq5 (MaxPositions=4, Sl=2.40, Tp=3.20, TrailAct=1.15, TrailLock=0.95)')

# Compile EA
cmd = [
    str(ROOT / 'mt5_v2_runner' / 'MetaEditor64.exe'),
    f"/compile:{ROOT / 'mt5_v2_runner' / 'MQL5' / 'Experts' / 'SelfGrowing_MT5_Trainer_EA.mq5'}",
    f"/log:{ROOT / 'mt5_v2_runner' / 'logs' / 'compile_SelfGrowing_MT5_Trainer_EA.log'}"
]
res = subprocess.run(cmd, capture_output=True, text=True)
log_path = ROOT / 'mt5_v2_runner' / 'logs' / 'compile_SelfGrowing_MT5_Trainer_EA.log'
log_content = log_path.read_text(encoding='utf-16', errors='ignore') if log_path.exists() else ''
print("   Compilation log tail:", [l for l in log_content.splitlines() if 'Result' in l or 'error' in l.lower()][-2:])

# 3. Update Tester .set file
set_path = ROOT / 'mt5_v2_runner' / 'MQL5' / 'Profiles' / 'Tester' / 'SelfGrowing_MT5_Trainer_EA.set'
try:
    with open(set_path, 'r', encoding='utf-16') as f:
        set_text = f.read()
except Exception:
    with open(set_path, 'r', encoding='utf-8', errors='ignore') as f:
        set_text = f.read()

set_text = re.sub(r'MaxPositions=\d+\|\|', 'MaxPositions=4||', set_text)
set_text = re.sub(r'FixedSlAtr=[\d\.]+\|\|[\d\.]+\|\|', 'FixedSlAtr=2.4||2.4||', set_text)
set_text = re.sub(r'FixedTpAtr=[\d\.]+\|\|[\d\.]+\|\|', 'FixedTpAtr=3.2||3.2||', set_text)
set_text = re.sub(r'TrailActivationAtr=[\d\.]+\|\|[\d\.]+\|\|', 'TrailActivationAtr=1.15||1.15||', set_text)
set_text = re.sub(r'TrailLockAtr=[\d\.]+\|\|[\d\.]+\|\|', 'TrailLockAtr=0.95||0.95||', set_text)

with open(set_path, 'w', encoding='utf-16') as f:
    f.write(set_text)
print('3. Successfully updated SelfGrowing_MT5_Trainer_EA.set')

# 4. Update run_mythos_mt5_live.py
py_path = ROOT / 'scripts' / 'run_mythos_mt5_live.py'
with open(py_path, 'r', encoding='utf-8') as f:
    py_text = f.read()

py_text = re.sub(r'mythos_policy\.max_positions\s*=\s*\d+', 'mythos_policy.max_positions = 4', py_text)
py_text = re.sub(r'mythos_policy\.min_spacing_bars\s*=\s*\d+', 'mythos_policy.min_spacing_bars = 1', py_text)
py_text = re.sub(r'mythos_policy\.prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.prm_threshold = 0.15', py_text)
py_text = re.sub(r'mythos_policy\.chop_prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.chop_prm_threshold = 0.55', py_text)
py_text = re.sub(r'mythos_policy\.sniper_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.sniper_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.sniper_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.sniper_tp_rr = 1.333', py_text)
py_text = re.sub(r'mythos_policy\.scalp_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.scalp_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.scalp_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.scalp_tp_rr = 1.333', py_text)
py_text = re.sub(r'mythos_policy\.trailing_atr\s*=\s*[\d\.]+', 'mythos_policy.trailing_atr = 0.95', py_text)

py_text = re.sub(r'"max_positions":\s*\d+', '"max_positions": 4', py_text)
py_text = re.sub(r'"fixed_sl_atr":\s*[\d\.]+', '"fixed_sl_atr": 2.40', py_text)
py_text = re.sub(r'"fixed_tp_atr":\s*[\d\.]+', '"fixed_tp_atr": 3.20', py_text)
py_text = re.sub(r'"trail_activation_atr":\s*[\d\.]+', '"trail_activation_atr": 1.15', py_text)
py_text = re.sub(r'"trail_lock_atr":\s*[\d\.]+', '"trail_lock_atr": 0.95', py_text)

with open(py_path, 'w', encoding='utf-8') as f:
    f.write(py_text)
print('4. Successfully updated run_mythos_mt5_live.py for Iteration 31')
