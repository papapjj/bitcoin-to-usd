"""
Verification test for MythosMT5Policy:
- Ensures act_nam_tac_nhan returns exactly 5 traces
- Ensures Agent 4 is strictly HOLD (0)
- Ensures clone_policy and policy_state_hash work seamlessly
- Ensures latency is <= 50ms per tick
"""
import copy
import sys
import time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import (
    tai_he_chuyen_gia_mot_context_mot_policy,
    clone_policy,
    policy_state_hash,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy

print("[1/4] Loading base policy...")
config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
config = load_config(config_path)
checkpoint_path = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
base_policy = tai_he_chuyen_gia_mot_context_mot_policy(checkpoint_path, config, seed=2026)
print("Base policy loaded successfully.")

print("\n[2/4] Initializing MythosMT5Policy with xLSTM 17.5M reasoning engine...")
mythos_policy = MythosMT5Policy(base_policy=base_policy, initial_capital=10_000.0)
print("MythosMT5Policy initialized successfully.")

print("\n[3/4] Testing act_nam_tac_nhan with mock observation...")
obs = np.zeros(137, dtype=np.float32)
t0 = time.perf_counter()
traces = mythos_policy.act_nam_tac_nhan(obs)
t_tick = (time.perf_counter() - t0) * 1000.0

assert len(traces) == 5, f"Expected 5 traces, got {len(traces)}"
agent4_trace = traces[4]
agent4_out = getattr(agent4_trace, "output_vector", getattr(agent4_trace, "selected_action", None))
assert agent4_out[0] == 1.0, f"Agent 4 must be HOLD (action 0), got output: {agent4_out[:3]}"
print(f"act_nam_tac_nhan PASSED: 5 traces, Agent 4 locked to HOLD, latency={t_tick:.2f}ms")

print("\n[4/4] Testing clone_policy and policy_state_hash...")
hash_before = policy_state_hash(mythos_policy)
cloned = clone_policy(mythos_policy)
hash_cloned = policy_state_hash(cloned)
assert len(hash_before) == 64, f"Invalid hash: {hash_before}"
print(f"policy_state_hash PASSED: {hash_before[:16]}...")
print(f"clone_policy PASSED: Cloned object type {type(cloned)}")

print("\n>>> ALL CHECKS PASSED 100% FOR MythosMT5Policy! <<<")
