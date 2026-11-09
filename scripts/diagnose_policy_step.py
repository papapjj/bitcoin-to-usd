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

print("Testing MythosMT5Policy diagnosis...")
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

df = pd.read_csv("outputs/btcusd_m1_7day_clean.csv", sep="\t")

for i in range(120, 160):
    c = float(df['<CLOSE>'].iloc[i])
    o = float(df['<OPEN>'].iloc[i])
    h = float(df['<HIGH>'].iloc[i])
    l = float(df['<LOW>'].iloc[i])
    
    obs = np.zeros(137, dtype=np.float32)
    obs[0] = c; obs[1] = 45.0
    obs[2] = (c - o) / 45.0
    obs[3] = (h - max(o, c)) / 45.0
    obs[4] = (min(o, c) - l) / 45.0
    obs[5] = (h - l) / 45.0
    obs[6] = 1.0
    
    traces = policy.act_nam_tac_nhan(obs)
    # Print diagnosis for bars 145 to 155
    if i >= 145:
        b_s = policy._bull_score_hist[-1] if len(policy._bull_score_hist) > 0 else 0
        r_s = policy._bear_score_hist[-1] if len(policy._bear_score_hist) > 0 else 0
        mb = float(np.mean(policy._bull_score_hist))
        sb = max(0.01, float(np.std(policy._bull_score_hist)))
        mr = float(np.mean(policy._bear_score_hist))
        sr = max(0.01, float(np.std(policy._bear_score_hist)))
        bz = (b_s - mb) / sb
        rz = (r_s - mr) / sr
        print(f"Bar {i}: price={c:.1f}, b_s={b_s:.4f}, r_s={r_s:.4f}, bz={bz:.2f}, rz={rz:.2f}, trade_count={policy.trade_count}")
