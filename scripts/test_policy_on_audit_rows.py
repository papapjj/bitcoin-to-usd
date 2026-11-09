import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from self_growing_trader.dual_champion_router import load_dual_champion_router
from self_growing_trader.regime_detector import RegimeDetector
from self_growing_trader.cppi_risk_engine import CPPIConfig
from self_growing_trader.two_tier_runner import TwoTierConfig

def test_live_audit_replay():
    audit_path = ROOT / ".mt5_runners" / "visual_7day_champion_6232_20261101_024544_115657" / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_feature_audit.csv"
    if not audit_path.exists():
        print("Audit path not found.")
        return

    print("Reading feature audit rows...")
    df_audit = pd.read_csv(audit_path, sep="\t")
    print(f"Total audit rows: {len(df_audit)}")

    bear_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    bull_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "multi_round_phase3" / "round_01" / "reasoning_rl_v1_active_candidate.pkl"
    regime_model_path = ROOT / "outputs" / "regime_models" / "champion_regime_detector.pkl"
    regime_detector = RegimeDetector.load(regime_model_path)

    safe_cppi = CPPIConfig(max_allowed_dd=0.08)
    two_tier = TwoTierConfig(
        initial_sl_atr=2.80,
        breakeven_trigger_atr=2.00,
        breakeven_lock_atr=0.80,
        partial_tp_atr=4.00,
        trailing_atr=1.20,
        trail_after_breakeven=True,
    )

    base_router = load_dual_champion_router(
        bear_path=bear_path,
        bull_path=bull_path,
        detector_path=regime_detector,
        cppi_config=safe_cppi,
        two_tier_config=two_tier,
    )

    policy = MythosMT5Policy(base_policy=base_router, initial_capital=10000.0, symbol="BTC")
    policy.long_only = False
    policy.short_only = False
    policy.min_margin_threshold = 0.12
    policy.bear_min_margin = 0.08
    policy.min_confidence = 0.40
    policy.bear_entry_threshold = 0.35
    policy.prm_threshold = 0.35
    policy.min_spacing_bars = 2
    policy.post_exit_cooldown_bars = 1
    policy.max_positions = 5
    policy.scalp_sl_atr = 2.80
    policy.sniper_sl_atr = 2.80
    policy.breakeven_trigger_atr = 2.00
    policy.breakeven_lock_atr = 0.80

    champ_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
    if champ_pt.exists():
        ckpt = torch.load(champ_pt, map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" in ckpt:
            policy.pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=False)
        if "world_model_state_dict" in ckpt:
            policy.world_model.load_state_dict(ckpt["world_model_state_dict"], strict=False)

    bilateral_pt = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_precision_champion.pt"
    if bilateral_pt.exists():
        policy.pipeline.bilateral_engine.load_checkpoint(bilateral_pt)
        policy.pipeline.bilateral_engine.config.prm_threshold = 0.35
        policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.35
        policy.pipeline.use_bilateral_dual_cot = True
        print(f"Loaded Bilateral Precision Champion from {bilateral_pt.name}")

    print("Replaying through audit rows...")
    actions = []
    class DummyReq:
        def __init__(self, time_str, inputs_arr):
            self.time = time_str
            self.inputs = inputs_arr
            self.equity = 10000.0
            self.positions = 0

    for idx in range(min(500, len(df_audit))):
        row = df_audit.iloc[idx]
        b_time = row["bar_time"]
        inputs = np.array([float(x) for x in row["inputs"].split("|") if x.strip()], dtype=np.float32)
        dummy_req = DummyReq(b_time, inputs)
        traces = policy.act_nam_tac_nhan(inputs, goal=None, request=dummy_req)
        act_0 = policy._extract_action_idx(traces[0])
        act_1 = policy._extract_action_idx(traces[1])
        actions.append((b_time, act_0, act_1))

    df_act = pd.DataFrame(actions, columns=["bar_time", "act_0", "act_1"])
    print("\nReplay Action 0 (Bull) counts:")
    print(df_act["act_0"].value_counts())
    print("\nReplay Action 1 (Bear) counts:")
    print(df_act["act_1"].value_counts())

if __name__ == "__main__":
    test_live_audit_replay()
