import sys
from pathlib import Path
ROOT = Path("c:/Users/Admin/Desktop/new-chat")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
import numpy as np
import pandas as pd

from scripts.train_multi_regime_anti_mae import extract_features_from_csv
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
)
from self_growing_trader.config import load_config

WEEK3_CSV = ROOT / "outputs" / "btcusd_m1_week3_20261019.csv"
CHAMPION_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_symmetric_anti_mae_champion.pt"

config_path = ROOT / "configs" / "he_chuyen_gia_mot_context_mot_policy.yaml"
if not config_path.exists():
    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
app = load_config(config_path)

base_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint, app, seed=2026)
_cau_hinh_policy_chuyen_gia_cho_mt5(base_policy, app)

mythos_policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
mythos_policy.direct_action = True
mythos_policy.conviction_threshold = 0.025
mythos_policy.require_margin_alignment = False
mythos_policy.enable_mtf_filter = False
mythos_policy.max_positions = 3
mythos_policy.max_consecutive_losses = 2
mythos_policy.loss_cooldown_bars = 2
mythos_policy.min_spacing_bars = 2
mythos_policy.prm_threshold = 0.32

# Load champion
ckpt = torch.load(CHAMPION_PATH, map_location="cpu", weights_only=False)
mythos_policy.pipeline.bilateral_engine.load_checkpoint(CHAMPION_PATH)
mythos_policy.pipeline.bilateral_engine.config.prm_threshold = 0.32
mythos_policy.pipeline.bilateral_engine.config.margin_threshold = 0.07
mythos_policy.pipeline.bilateral_engine.arbiter.prm_threshold = 0.32
mythos_policy.pipeline.bilateral_engine.arbiter.margin_threshold = 0.07

data = extract_features_from_csv(WEEK3_CSV)
obs_all = data["base_obs"]
N = len(obs_all)

print("=" * 80)
print(f"STEPPING MythosMT5Policy THROUGH {N} M1 BARS OF WEEK 3")
print("=" * 80)

# Track actions proposed and gated
dir_counts = {1: 0, -1: 0, 0: 0}
can_open_counts = {1: 0, -1: 0}
executed_counts = {1: 0, -1: 0}
veto_reasons = {}

# Mock request class
class MockReq:
    def __init__(self, t, price, atr):
        self.time = t
        self.current_price = price
        self.current_atr = atr
        self.positions = 0
        self.equity = 10000.0

df_m1 = pd.read_csv(WEEK3_CSV, sep="\t")
col_date = "<DATE>" if "<DATE>" in df_m1.columns else "DATE"
col_time = "<TIME>" if "<TIME>" in df_m1.columns else "TIME"
col_c = "<CLOSE>" if "<CLOSE>" in df_m1.columns else "CLOSE"
times = (df_m1[col_date].astype(str) + " " + df_m1[col_time].astype(str).str[:5]).values
closes = df_m1[col_c].values

# We simulate the first 1,500 bars to see the pacing and why SELL vs BUY happened
for t in range(min(1500, N)):
    req = MockReq(times[t], closes[t], 45.0)
    traces = mythos_policy.act_nam_tac_nhan(obs_all[t], request=req)
    a0 = mythos_policy._extract_action_idx(traces[0])
    a1 = mythos_policy._extract_action_idx(traces[1])
    
    act_name = "HOLD"
    if a0 == 1:
        act_name = "BUY"
        executed_counts[1] += 1
    elif a1 == 2:
        act_name = "SELL"
        executed_counts[-1] += 1

print(f"In first 1,500 bars: Executed BUY = {executed_counts[1]}, Executed SELL = {executed_counts[-1]}")
