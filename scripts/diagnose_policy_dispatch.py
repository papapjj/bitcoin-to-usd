import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
import pandas as pd
import numpy as np
import torch

from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from src.self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.two_tier_runner import TwoTierConfig

two_tier = TwoTierConfig(
    initial_sl_atr=4.5,
    breakeven_trigger_atr=1.80,
    breakeven_lock_atr=1.00,
    partial_tp_atr=3.50,
    trailing_atr=1.50,
)

base_router = load_dual_champion_router(
    bear_path='outputs/he_chuyen_gia/v193_runs/best_promoted_checkpoint_phase2.pkl',
    bull_path='outputs/he_chuyen_gia/v193_runs/multi_round_phase3/round_01/reasoning_rl_v1_active_candidate.pkl',
    detector_path='outputs/he_chuyen_gia/v193_runs/regime_detector_phase2.pkl',
    bull_threshold=0.35,
    bear_threshold=0.35,
    chop_threshold=0.38,
    confirmation_bars=1,
    htf_filter_enabled=True,
    momentum_gate_enabled=True,
    circuit_breaker_enabled=True,
    micro_scalp_enabled=False,
    max_concurrent_positions=2,
    max_positions_per_direction=2,
    two_tier_config=two_tier,
)

policy = MythosMT5Policy(base_policy=base_router, initial_capital=10000.0, symbol='BTC')
policy.long_only = False
policy.short_only = False
policy.min_margin_threshold = 0.05
policy.max_margin_threshold = 0.95
policy.bear_entry_threshold = 0.48
policy.bear_min_margin = 0.05
policy.min_confidence = 0.40
policy.prm_threshold = 0.30
policy.min_atr_threshold = 5.0
policy.max_atr_percentile = 0.98
policy.require_margin_alignment = True
policy.loss_cooldown_bars = 45
policy.max_consecutive_losses = 2
policy.min_spacing_bars = 2
policy.post_exit_cooldown_bars = 1
policy.max_positions = 2
policy.scalp_sl_atr = 4.5
policy.sniper_sl_atr = 4.5
policy.scalp_tp_rr = 0.75
policy.sniper_tp_rr = 0.75
policy.trailing_atr = 1.50
policy.breakeven_trigger_atr = 1.80
policy.breakeven_lock_atr = 1.00

bilateral_pt = 'outputs/he_chuyen_gia/v193_runs/mythos_bilateral_dual_cot_champion.pt'
policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
policy.pipeline.bilateral_engine.config.prm_threshold = 0.33
policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.33
policy.pipeline.use_bilateral_dual_cot = True

df = pd.read_csv(r'.mt5_runners\visual_7day_champion_6488_20261031_162132_907165\Tester\Agent-127.0.0.1-3000\MQL5\Files\self_growing_feature_audit.csv', sep='\t')
print("Testing policy.act_nam_tac_nhan on first 200 audit bars...")

# Let's mock a simple request object
class MockRequest:
    def __init__(self, inputs, equity=10000.0):
        self.inputs = inputs
        self.equity = equity
        self.positions = 0
        self.time = '2025.02.20 01:00'

dispatched_actions = []

for i in range(min(500, len(df))):
    inp_str = df['inputs'].iloc[i]
    vals = np.array([float(x) for x in inp_str.split('|') if x.strip()], dtype=np.float32)
    req = MockRequest(vals)
    traces = policy.act_nam_tac_nhan(vals, request=req)
    a0 = policy._extract_action_idx(traces[0])
    a1 = policy._extract_action_idx(traces[1])
    if a0 != 0 or a1 != 0:
        dispatched_actions.append((i, a0, a1))

print(f"Total dispatched trades in 500 bars: {len(dispatched_actions)}")
if dispatched_actions:
    print("Dispatched:", dispatched_actions[:10])
