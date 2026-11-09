import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

# 1. Update research_reasoning_world_class/src/mythos_mt5_policy.py
policy_path = ROOT / 'research_reasoning_world_class' / 'src' / 'mythos_mt5_policy.py'
with open(policy_path, 'r', encoding='utf-8') as f:
    pol_text = f.read()

# Replace Macro Trend, HTF, Value Zone, CHOP, and Spacing blocks cleanly
old_block_pattern = r'# Higher Timeframe \(HTF\) Macro Alignment Guard.*?can_open = spacing_ok and \(max_positions <= 0 or active_positions < max_positions\)'
new_block = '''# Higher Timeframe (HTF) Macro Alignment Guard (Strict Trend-Following: never fight the macro trend)
        m15_diff = float(base_obs_137[36]) if base_obs_137.size > 36 else 0.0
        h1_diff = float(base_obs_137[40]) if base_obs_137.size > 40 else 0.0

        if net_dir == -1 and (h1_diff > 0.02 or m15_diff > 0.05 or ema_diff_val > 0.0 or p_vs_e50_val > 0.0):
            net_dir = 0  # Strictly forbid shorting when H1/M15/M1 is bullish
        elif net_dir == 1 and (h1_diff < -0.02 or m15_diff < -0.05 or ema_diff_val < 0.0 or p_vs_e50_val < 0.0):
            net_dir = 0  # Strictly forbid buying when H1/M15/M1 is bearish

        # Value Zone Pullback Guard (Anti-Chasing: do not buy tops or sell bottoms)
        if net_dir == 1 and p_vs_e20_val > 1.20:
            net_dir = 0  # Do not chase extended rallies stretched > 1.2 ATR above EMA20
        elif net_dir == -1 and p_vs_e20_val < -1.20:
            net_dir = 0  # Do not chase extended dumps stretched > 1.2 ATR below EMA20

        # Champion Market Regime / Sideway (CHOP) Guard (Definitive Milestone 3 Defense)
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

        # Robust CHOP defense: if CHOP probability is elevated (>= 0.30) or detector names CHOP:
        is_chop = (detected_regime is not None and detected_regime.name == "CHOP") or \\
                  (regime_probs is not None and len(regime_probs) > 2 and regime_probs[2] >= 0.30)
        if is_chop and net_dir != 0:
            chop_gate = getattr(self, "chop_prm_threshold", 0.70)
            if prm < chop_gate:
                net_dir = 0  # VETO choppy breakout traps during sideways consolidation

        if not getattr(self, "direct_action", True):
            # Neural Confidence and Volatility Regime Filter
            min_conf = getattr(self, "min_confidence", 0.0)
            max_atr_pct = getattr(self, "max_atr_percentile", 1.0)
            atr_pct = float(base_obs_137[14]) if base_obs_137.size > 14 else 0.5
            min_atr = float(getattr(self, "min_atr_threshold", 0.0))
            eff_conf = conf if net_dir != 0 else 0.0
            if (net_dir != 0 and eff_conf < min_conf) or atr_pct > max_atr_pct or self.current_atr < min_atr:
                net_dir = 0

        # Global Loss Cooldown Guard: after consecutive losses, pause all trading
        if net_dir != 0 and hasattr(self, "_global_cooldown_until"):
            if self.seq_counter < getattr(self, "_global_cooldown_until", 0):
                net_dir = 0

        # Directional Loss Cooldown Guard: after consecutive losses, pause that direction
        if net_dir != 0 and hasattr(self, "_direction_cooldown_until"):
            if self.seq_counter < self._direction_cooldown_until.get(net_dir, 0):
                net_dir = 0

        # Terminal flat safety: close-out cooldown in last 15 packets
        if getattr(self, "max_packets", 0) > 0 and self.seq_counter >= self.max_packets - 15:
            net_dir = 0

        # Multi-position laddering & concurrency guards:
        spacing_ok = True
        bars_since = self.seq_counter - self._last_entry_bar
        min_spacing_bars = max(1, int(getattr(self, "min_spacing_bars", 1)))
        if bars_since < min_spacing_bars:
            spacing_ok = False
        elif active_positions > 0 and self._last_entry_price is not None:
            # Allow pyramiding into winners as price advances >= 0.08 ATR in favor
            favorable = (net_dir == 1 and self.current_price >= self._last_entry_price + 0.08 * cur_atr) or \\
                        (net_dir == -1 and self.current_price <= self._last_entry_price - 0.08 * cur_atr)
            if not favorable:
                spacing_ok = False

        can_open = spacing_ok and (max_positions <= 0 or active_positions < max_positions)'''

pol_text = re.sub(old_block_pattern, new_block, pol_text, flags=re.DOTALL)

with open(policy_path, 'w', encoding='utf-8') as f:
    f.write(pol_text)
print("1. Successfully updated mythos_mt5_policy.py for Iteration 35")

# 2. Update SelfGrowing_MT5_Trainer_EA.mq5
ea_path = ROOT / 'mt5_v2_runner' / 'MQL5' / 'Experts' / 'SelfGrowing_MT5_Trainer_EA.mq5'
with open(ea_path, 'r', encoding='utf-8') as f:
    ea_text = f.read()

ea_text = re.sub(r'input int MaxPositions\s*=\s*\d+;', 'input int MaxPositions = 5;', ea_text)
ea_text = re.sub(r'input double FixedSlAtr\s*=\s*[\d\.]+;', 'input double FixedSlAtr = 2.40;', ea_text)
ea_text = re.sub(r'input double FixedTpAtr\s*=\s*[\d\.]+;', 'input double FixedTpAtr = 2.80;', ea_text)
ea_text = re.sub(r'input double TrailActivationAtr\s*=\s*[\d\.]+;', 'input double TrailActivationAtr = 1.80;', ea_text)
ea_text = re.sub(r'input double TrailLockAtr\s*=\s*[\d\.]+;', 'input double TrailLockAtr = 0.30;', ea_text)

with open(ea_path, 'w', encoding='utf-8') as f:
    f.write(ea_text)
print("2. Successfully verified SelfGrowing_MT5_Trainer_EA.mq5 (MaxPositions=5, Sl=2.40, Tp=2.80, TrailAct=1.80, TrailLock=0.30)")

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

set_text = re.sub(r'MaxPositions=\d+\|\|', 'MaxPositions=5||', set_text)
set_text = re.sub(r'FixedSlAtr=[\d\.]+\|\|[\d\.]+\|\|', 'FixedSlAtr=2.4||2.4||', set_text)
set_text = re.sub(r'FixedTpAtr=[\d\.]+\|\|[\d\.]+\|\|', 'FixedTpAtr=2.8||2.8||', set_text)
set_text = re.sub(r'TrailActivationAtr=[\d\.]+\|\|[\d\.]+\|\|', 'TrailActivationAtr=1.8||1.8||', set_text)
set_text = re.sub(r'TrailLockAtr=[\d\.]+\|\|[\d\.]+\|\|', 'TrailLockAtr=0.3||0.3||', set_text)

with open(set_path, 'w', encoding='utf-16') as f:
    f.write(set_text)
print("3. Successfully updated SelfGrowing_MT5_Trainer_EA.set")

# 4. Update run_mythos_mt5_live.py
py_path = ROOT / 'scripts' / 'run_mythos_mt5_live.py'
with open(py_path, 'r', encoding='utf-8') as f:
    py_text = f.read()

py_text = re.sub(r'mythos_policy\.max_positions\s*=\s*\d+', 'mythos_policy.max_positions = 5', py_text)
py_text = re.sub(r'mythos_policy\.min_spacing_bars\s*=\s*\d+', 'mythos_policy.min_spacing_bars = 1', py_text)
py_text = re.sub(r'mythos_policy\.prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.prm_threshold = 0.12', py_text)
py_text = re.sub(r'mythos_policy\.chop_prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.chop_prm_threshold = 0.70', py_text)
py_text = re.sub(r'mythos_policy\.loss_cooldown_bars\s*=\s*\d+', 'mythos_policy.loss_cooldown_bars = 2', py_text)
py_text = re.sub(r'mythos_policy\.sniper_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.sniper_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.sniper_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.sniper_tp_rr = 1.167', py_text)
py_text = re.sub(r'mythos_policy\.scalp_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.scalp_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.scalp_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.scalp_tp_rr = 1.167', py_text)
py_text = re.sub(r'mythos_policy\.trailing_atr\s*=\s*[\d\.]+', 'mythos_policy.trailing_atr = 1.40', py_text)

py_text = re.sub(r'"max_positions":\s*\d+', '"max_positions": 5', py_text)
py_text = re.sub(r'"fixed_sl_atr":\s*[\d\.]+', '"fixed_sl_atr": 2.40', py_text)
py_text = re.sub(r'"fixed_tp_atr":\s*[\d\.]+', '"fixed_tp_atr": 2.80', py_text)
py_text = re.sub(r'"trail_activation_atr":\s*[\d\.]+', '"trail_activation_atr": 1.80', py_text)
py_text = re.sub(r'"trail_lock_atr":\s*[\d\.]+', '"trail_lock_atr": 0.30', py_text)

with open(py_path, 'w', encoding='utf-8') as f:
    f.write(py_text)
print("4. Successfully updated run_mythos_mt5_live.py for Iteration 35")
