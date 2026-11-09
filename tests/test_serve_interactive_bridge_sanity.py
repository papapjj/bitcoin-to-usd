"""
tests/test_serve_interactive_bridge_sanity.py

Sanity verification for serve_interactive_visual_bridge.py with Adversarial Dialectical Reasoning.
"""

import json
from pathlib import Path
import time
import numpy as np
import pytest
import torch

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tai_he_chuyen_gia_mot_context_mot_policy
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.adversarial_dialectical_reasoning import (
    AdversarialDialecticalReasoningEngine,
    DialecticalConfig,
    DialecticalVerdict,
)

ROOT = Path(__file__).resolve().parents[1]


def test_bridge_dialectical_engine_integration():
    config_path = ROOT / "configs" / "he_chuyen_gia_reasoning_rl_v1_candidate.yaml"
    base_checkpoint = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "best_promoted_checkpoint_phase2.pkl"
    app_cfg = load_config(config_path)
    base_policy = tai_he_chuyen_gia_mot_context_mot_policy(base_checkpoint, app_cfg, seed=2026)
    policy = MythosMT5Policy(base_policy=base_policy)

    assert hasattr(policy.pipeline, "dialectical_engine")
    assert policy.pipeline.dialectical_engine is not None

    inputs_arr = np.random.randn(137).astype(np.float32)
    # Simulate bull trap condition: upper wick high, positive return fading
    inputs_arr[0] = 0.1
    inputs_arr[7] = 0.2
    inputs_arr[8] = 0.75 # Upper wick >= 0.65 -> Bull trap

    class BridgeReqMock:
        inputs = inputs_arr
        equity = 10000.0
        balance = 10000.0
        positions = 0
        time = "2026.10.27 06:00"
        sequence = 1

    traces = policy.act_nam_tac_nhan(inputs_arr, request=BridgeReqMock)
    assert len(traces) == 5

    # Check that dialectical engine executes properly
    dial_in = torch.from_numpy(inputs_arr).float()
    dial_verdict = policy.pipeline.dialectical_engine(dial_in)
    assert isinstance(dial_verdict, DialecticalVerdict)
    assert len(dial_verdict.debate_history) == 3
    assert dial_verdict.latency_ms < 50.0
