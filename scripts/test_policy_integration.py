import sys, os
from pathlib import Path
ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import pandas as pd, numpy as np, torch
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.cppi_risk_engine import CPPIConfig
from self_growing_trader.two_tier_runner import TwoTierConfig

print("Testing MythosMT5Policy end-to-end integration...")
bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
regime_detector = RegimeDetector.load(regime_model_path)

safe_cppi = CPPIConfig()
two_tier = TwoTierConfig(
    initial_sl_atr=2.0,
    breakeven_trigger_atr=0.80,
    breakeven_lock_atr=0.35,
    partial_tp_atr=2.20,
    trailing_atr=0.75,
)

base_router = load_dual_champion_router(
    bear_path=bear_path,
    bull_path=bull_path,
    detector_path=regime_detector,
    bull_threshold=0.35,
    bear_threshold=0.35,
    chop_threshold=0.38,
    confirmation_bars=1,
    htf_filter_enabled=False,
    momentum_gate_enabled=False,
    circuit_breaker_enabled=False,
    micro_scalp_enabled=False,
    max_concurrent_positions=5,
    max_positions_per_direction=5,
    cppi_config=safe_cppi,
    two_tier_config=two_tier,
)

policy = MythosMT5Policy(base_policy=base_router, initial_capital=10000.0, symbol="BTC")
policy.long_only = False
policy.short_only = False
policy.scalp_sl_atr = 2.0
policy.sniper_sl_atr = 2.0
policy.scalp_tp_rr = 1.00
policy.sniper_tp_rr = 1.00
policy.trailing_atr = 0.75
policy.min_spacing_bars = 2

bilateral_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
if bilateral_pt.exists():
    policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
    policy.pipeline.use_bilateral_dual_cot = True
    print(f"Loaded bilateral champion weights: {bilateral_pt.name}")

df = pd.read_csv("outputs/btcusd_m1_7day_clean.csv", sep="\t")
print(f"Loaded {len(df)} M1 bars from btcusd_m1_7day_clean.csv")

# Run through 1000 consecutive bars
buy_entries = 0
sell_entries = 0
holds = 0

for i in range(100, 200):
    o = float(df['<OPEN>'].iloc[i])
    h = float(df['<HIGH>'].iloc[i])
    l = float(df['<LOW>'].iloc[i])
    c = float(df['<CLOSE>'].iloc[i])
    vol = float(df['<TICKVOL>'].iloc[i]) if '<TICKVOL>' in df.columns else 100.0
    
    # 137D dummy obs where obs[0] is close, obs[1] is ATR
    obs = np.zeros(137, dtype=np.float32)
    obs[0] = c
    obs[1] = 45.0
    obs[2] = (c - o) / 45.0  # body
    obs[3] = (h - max(o, c)) / 45.0  # upper wick
    obs[4] = (min(o, c) - l) / 45.0  # lower wick
    obs[5] = (h - l) / 45.0  # range
    obs[6] = 1.0  # rel vol
    
    traces = policy.act_nam_tac_nhan(obs)
    if i in [150, 151, 152, 200, 250, 300]:
        t0, t1 = traces[0], traces[1]
        act0 = getattr(t0, 'action_index', getattr(t0, 'selected_idx', -1))
        act1 = getattr(t1, 'action_index', getattr(t1, 'selected_idx', -1))
        print(f'Bar {i}: price={c:.1f}, act0={act0}, act1={act1}')
    t0 = traces[0]
    t1 = traces[1]
    
    act0 = t0.action if hasattr(t0, "action") else (t0.selected_idx if hasattr(t0, "selected_idx") else 0)
    act1 = t1.action if hasattr(t1, "action") else (t1.selected_idx if hasattr(t1, "selected_idx") else 0)
    
    # Check if entry
    out0 = getattr(t0, "output_vector", None)
    out1 = getattr(t1, "output_vector", None)
    
    if out0 is not None and out0[1] > 0.5:
        buy_entries += 1
    elif out1 is not None and out1[2] > 0.5:
        sell_entries += 1
    else:
        holds += 1

print("\n" + "=" * 60)
print("  MYTHOS POLICY INTEGRATION TEST (1,000 BARS)")
print("=" * 60)
print(f"Total Bars Evaluated : 1000")
print(f"BUY Entries (Agent 0): {buy_entries}")
print(f"SELL Entries (Agent 1): {sell_entries}")
print(f"HOLD / Vetoes        : {holds}")
print(f"Trade Ratio          : BUY={buy_entries/(buy_entries+sell_entries+1e-5):.1%}, SELL={sell_entries/(buy_entries+sell_entries+1e-5):.1%}")
print("=" * 60)
