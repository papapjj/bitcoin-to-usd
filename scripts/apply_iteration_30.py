import re
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent

# 1. Update research_reasoning_world_class/src/mythos_mt5_policy.py
policy_path = ROOT / 'research_reasoning_world_class' / 'src' / 'mythos_mt5_policy.py'
with open(policy_path, 'r', encoding='utf-8') as f:
    pol_text = f.read()

# A. Update min_spacing_bars default and runtime clamping
pol_text = pol_text.replace(
    'self.min_spacing_bars: int = 2',
    'self.min_spacing_bars: int = 1'
)
pol_text = pol_text.replace(
    'min_spacing_bars = max(2, int(getattr(self, "min_spacing_bars", 2)))',
    'min_spacing_bars = max(1, int(getattr(self, "min_spacing_bars", 1)))'
)

# B. Add Champion Regime Detector initialization in __init__ if not already present
if 'self.regime_detector = None' not in pol_text:
    init_target = '        self.prm_threshold: float = 0.35'
    init_repl = '''        self.prm_threshold: float = 0.35
        self.chop_prm_threshold: float = 0.45
        self.regime_detector = None
        try:
            from self_growing_trader.regime_detector import RegimeDetector
            reg_path = Path("outputs") / "regime_models" / "champion_regime_detector.pkl"
            if reg_path.exists():
                self.regime_detector = RegimeDetector.load(reg_path)
        except Exception as e:
            logger.warning(f"Failed to load champion regime detector: {e}")'''
    assert init_target in pol_text, "Init target not found in mythos_mt5_policy.py"
    pol_text = pol_text.replace(init_target, init_repl)

# C. Add Sideway / CHOP Guard in predict
if 'CHOP Guard' not in pol_text:
    pred_target = '''        # Macro Trend Alignment Guard (Strict Trend-Following: never fight the macro trend)
        if net_dir == -1 and (ema_diff_val > -0.05 or p_vs_e50_val > 0.10):
            net_dir = 0  # Strictly forbid shorting when trend is bullish (Bull Specialist owns market)
        elif net_dir == 1 and (ema_diff_val < 0.05 or p_vs_e50_val < -0.10):
            net_dir = 0  # Strictly forbid buying when trend is bearish (Bear Specialist owns market)'''

    pred_repl = '''        # Macro Trend Alignment Guard (Strict Trend-Following: never fight the macro trend)
        if net_dir == -1 and (ema_diff_val > -0.05 or p_vs_e50_val > 0.10):
            net_dir = 0  # Strictly forbid shorting when trend is bullish (Bull Specialist owns market)
        elif net_dir == 1 and (ema_diff_val < 0.05 or p_vs_e50_val < -0.10):
            net_dir = 0  # Strictly forbid buying when trend is bearish (Bear Specialist owns market)

        # Champion Market Regime / Sideway (CHOP) Guard (Milestone 2/3 Defense):
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

    assert pred_target in pol_text, "Pred target not found in mythos_mt5_policy.py"
    pol_text = pol_text.replace(pred_target, pred_repl)

with open(policy_path, 'w', encoding='utf-8') as f:
    f.write(pol_text)
print("1. Successfully updated mythos_mt5_policy.py (min_spacing_bars=1, champion regime detector CHOP guard wired)")

# 2. Update SelfGrowing_MT5_Trainer_EA.mq5
ea_path = ROOT / 'mt5_v2_runner' / 'MQL5' / 'Experts' / 'SelfGrowing_MT5_Trainer_EA.mq5'
with open(ea_path, 'r', encoding='utf-8') as f:
    ea_text = f.read()

ea_text = re.sub(r'input int MaxPositions\s*=\s*\d+;', 'input int MaxPositions = 3;', ea_text)
ea_text = re.sub(r'input double FixedSlAtr\s*=\s*[\d\.]+;', 'input double FixedSlAtr = 2.40;', ea_text)
ea_text = re.sub(r'input double FixedTpAtr\s*=\s*[\d\.]+;', 'input double FixedTpAtr = 2.70;', ea_text)
ea_text = re.sub(r'input double TrailActivationAtr\s*=\s*[\d\.]+;', 'input double TrailActivationAtr = 0.85;', ea_text)
ea_text = re.sub(r'input double TrailLockAtr\s*=\s*[\d\.]+;', 'input double TrailLockAtr = 0.65;', ea_text)

with open(ea_path, 'w', encoding='utf-8') as f:
    f.write(ea_text)
print('2. Successfully updated SelfGrowing_MT5_Trainer_EA.mq5 (MaxPositions=3, Sl=2.20, Tp=2.70, TrailAct=0.85, TrailLock=0.65)')

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

set_text = re.sub(r'MaxPositions=\d+\|\|', 'MaxPositions=3||', set_text)
set_text = re.sub(r'FixedSlAtr=[\d\.]+\|\|[\d\.]+\|\|', 'FixedSlAtr=2.4||2.4||', set_text)
set_text = re.sub(r'FixedTpAtr=[\d\.]+\|\|[\d\.]+\|\|', 'FixedTpAtr=2.7||2.7||', set_text)
set_text = re.sub(r'TrailActivationAtr=[\d\.]+\|\|[\d\.]+\|\|', 'TrailActivationAtr=0.85||0.85||', set_text)
set_text = re.sub(r'TrailLockAtr=[\d\.]+\|\|[\d\.]+\|\|', 'TrailLockAtr=0.65||0.65||', set_text)

with open(set_path, 'w', encoding='utf-16') as f:
    f.write(set_text)
print('3. Successfully updated SelfGrowing_MT5_Trainer_EA.set')

# 4. Update run_mythos_mt5_live.py
py_path = ROOT / 'scripts' / 'run_mythos_mt5_live.py'
with open(py_path, 'r', encoding='utf-8') as f:
    py_text = f.read()

py_text = re.sub(r'mythos_policy\.max_positions\s*=\s*\d+', 'mythos_policy.max_positions = 3', py_text)
py_text = re.sub(r'mythos_policy\.min_spacing_bars\s*=\s*\d+', 'mythos_policy.min_spacing_bars = 1', py_text)
py_text = re.sub(r'mythos_policy\.prm_threshold\s*=\s*[\d\.]+', 'mythos_policy.prm_threshold = 0.15', py_text)
py_text = re.sub(r'mythos_policy\.sniper_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.sniper_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.sniper_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.sniper_tp_rr = 1.125', py_text)
py_text = re.sub(r'mythos_policy\.scalp_sl_atr\s*=\s*[\d\.]+', 'mythos_policy.scalp_sl_atr = 2.40', py_text)
py_text = re.sub(r'mythos_policy\.scalp_tp_rr\s*=\s*[\d\.]+', 'mythos_policy.scalp_tp_rr = 1.125', py_text)
py_text = re.sub(r'mythos_policy\.trailing_atr\s*=\s*[\d\.]+', 'mythos_policy.trailing_atr = 0.65', py_text)

py_text = re.sub(r'"max_positions":\s*\d+', '"max_positions": 3', py_text)
py_text = re.sub(r'"fixed_sl_atr":\s*[\d\.]+', '"fixed_sl_atr": 2.40', py_text)
py_text = re.sub(r'"fixed_tp_atr":\s*[\d\.]+', '"fixed_tp_atr": 2.70', py_text)
py_text = re.sub(r'"trail_activation_atr":\s*[\d\.]+', '"trail_activation_atr": 0.85', py_text)
py_text = re.sub(r'"trail_lock_atr":\s*[\d\.]+', '"trail_lock_atr": 0.65', py_text)

with open(py_path, 'w', encoding='utf-8') as f:
    f.write(py_text)
print('4. Successfully updated run_mythos_mt5_live.py for Iteration 30')
