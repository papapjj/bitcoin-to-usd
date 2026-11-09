"""
Test Suite: Milestone M3 — Empirical Challenger 2 (Checkpoint Exactness & Live Replay Challenge)
File: research_reasoning_world_class/tests/test_challenger_m3_exactness_replay.py

Empirically validates:
1. Canonical checkpoint strict=True loadability, immutability, and parameter exactness (191 keys, 187 params, 4,196,564 weights).
2. Challenger checkpoint strict=True loadability, routing metadata (SAVED_CHALLENGER), and parameter exactness.
3. Zero drift (max_abs_diff == 0.0) between Canonical and Challenger reasoning engine weights (proving freezing invariants).
4. Genuine adaptation (diff > 0.0) in World Model weights between Canonical and Challenger.
5. Replay consensus action-diversity: active trading occurs during OOS replay (no freeze / action-collapse into 100% HOLD).
"""

import hashlib
import time
from pathlib import Path
import numpy as np
import pytest
import torch

from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.unified_reasoning_pipeline import (
    UnifiedReasoningTraderPipeline,
)
from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from scripts.train_mythos_muzero_coevolution import SlicedM1Environment
from scripts.train_mythos_muzero_specialized_massive import SpecializedMassiveTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT_ROOT / "outputs" / "he_chuyen_gia" / "v193_runs"
CANONICAL_CKPT = CKPT_DIR / "mythos_muzero_coevolved_champion.pt"
CHALLENGER_CKPT = CKPT_DIR / "mythos_muzero_specialized_challenger.pt"
CSV_PATH = Path(r"C:\Users\Admin\Desktop\BTCUSDm.csv")


@pytest.fixture(scope="module")
def cot_config():
    return ReasoningConfig(
        input_dim=768,
        hidden_dim=256,
        num_hops=4,
        max_hops=8,
        enable_cross_hop=True,
        enable_backtrack=True,
        enable_act=True,
        prm_threshold=0.85,
        adversarial_margin_threshold=0.20,
    )


class TestCheckpointExactnessEmpiricalChallenge:
    """Verifies strict=True loading, bitwise exactness, and parameter invariance."""

    def test_canonical_checkpoint_strict_and_immutability(self, cot_config):
        assert CANONICAL_CKPT.exists(), "Canonical checkpoint file missing!"
        ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
        assert "reasoning_engine_state_dict" in ckpt

        engine = MetacognitiveReasoningEngine(cot_config)
        res = engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
        assert len(res.missing_keys) == 0, f"Missing keys: {res.missing_keys}"
        assert len(res.unexpected_keys) == 0, f"Unexpected keys: {res.unexpected_keys}"

        named_params = dict(engine.named_parameters())
        assert len(named_params) == 187, f"Expected 187 named params, got {len(named_params)}"
        total_weights = sum(p.numel() for p in engine.parameters())
        assert total_weights == 4_196_564, f"Expected 4,196,564 weights, got {total_weights}"

        # Immutability check vs backup
        bak = CKPT_DIR / "mythos_muzero_coevolved_champion.pt.pre_finetune_bak"
        if bak.exists():
            h_canon = hashlib.sha256(CANONICAL_CKPT.read_bytes()).hexdigest()
            h_bak = hashlib.sha256(bak.read_bytes()).hexdigest()
            assert h_canon == h_bak, "Canonical checkpoint hash diverged from pre_finetune backup!"

    def test_challenger_checkpoint_strict_and_routing(self, cot_config):
        assert CHALLENGER_CKPT.exists(), "Challenger checkpoint file missing!"
        ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)
        assert "reasoning_engine_state_dict" in ckpt

        engine = MetacognitiveReasoningEngine(cot_config)
        res = engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
        assert len(res.missing_keys) == 0, f"Missing keys: {res.missing_keys}"
        assert len(res.unexpected_keys) == 0, f"Unexpected keys: {res.unexpected_keys}"

        named_params = dict(engine.named_parameters())
        assert len(named_params) == 187, f"Expected 187 named params, got {len(named_params)}"
        total_weights = sum(p.numel() for p in engine.parameters())
        assert total_weights == 4_196_564, f"Expected 4,196,564 weights, got {total_weights}"

        # Gate routing assertions
        assert "stats" in ckpt
        assert ckpt["stats"]["status"] == "SAVED_CHALLENGER"
        assert ckpt["stats"]["total_trades"] == 66
        assert round(ckpt["stats"]["win_rate"] * 100, 2) == 48.48

    def test_reasoning_engine_weight_diff_is_strictly_zero(self):
        c_ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
        ch_ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)

        c_sd = c_ckpt["reasoning_engine_state_dict"]
        ch_sd = ch_ckpt["reasoning_engine_state_dict"]

        assert len(c_sd) == 191
        assert len(ch_sd) == 191
        assert set(c_sd.keys()) == set(ch_sd.keys())

        max_abs_diff = 0.0
        for k in c_sd.keys():
            diff = torch.max(torch.abs(c_sd[k].float() - ch_sd[k].float())).item()
            if diff > max_abs_diff:
                max_abs_diff = diff

        assert max_abs_diff == 0.0, f"Max absolute diff {max_abs_diff} != 0.0"

    def test_world_model_weights_adaptation(self):
        c_ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
        ch_ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)

        for name in ["bull_world_model_state_dict", "bear_world_model_state_dict"]:
            assert name in c_ckpt and name in ch_ckpt
            max_d = 0.0
            for k in ch_ckpt[name]:
                if k in c_ckpt[name]:
                    d = torch.max(torch.abs(ch_ckpt[name][k].float() - c_ckpt[name][k].float())).item()
                    if d > max_d:
                        max_d = d
            assert max_d > 1e-4, f"World model {name} was not updated during training! max diff = {max_d}"


class TestLiveReplayConsensusActionDiversity:
    """Verifies that the Challenger model actively executes trades and avoids action freeze."""

    def test_live_replay_action_collapse_resolved(self):
        assert CSV_PATH.exists(), f"CSV data missing at {CSV_PATH}"
        env_master = RealM1Environment(csv_path=str(CSV_PATH), spread_usd=2.50)
        env_master.load_data("2025.01.15", "2025.02.28")
        oos_slice = SlicedM1Environment(env_master, 55000, 55200, name="OOS_Challenge_Sample")

        trainer = SpecializedMassiveTrainer(
            bull_env=oos_slice,
            bear_env=oos_slice,
            chop_env=oos_slice,
            lr=3e-5,
            mcts_simulations=4,
            beta_dyna=0.0,
        )

        ch_ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)
        trainer.reasoning_engine.load_state_dict(ch_ckpt["reasoning_engine_state_dict"], strict=True)
        trainer.bull_champion.reasoning_engine.load_state_dict(ch_ckpt["bull_cot_state_dict"], strict=True)
        trainer.bear_champion.reasoning_engine.load_state_dict(ch_ckpt["bear_cot_state_dict"], strict=True)
        trainer.world_model.load_state_dict(ch_ckpt["world_model_state_dict"], strict=True)
        trainer.bull_champion.world_model.load_state_dict(ch_ckpt["bull_world_model_state_dict"], strict=True)
        trainer.bear_champion.world_model.load_state_dict(ch_ckpt["bear_world_model_state_dict"], strict=True)

        actions = []
        with torch.inference_mode():
            for idx in range(100, 150):
                obs = oos_slice.get_observation(idx)
                dec = trainer.resolve_consensus(
                    obs_137=obs,
                    current_active_positions=0,
                    active_position_dir=0,
                    min_confidence=0.60,
                    conflict_margin=0.15,
                )
                actions.append(dec["action"])

        non_zero_count = sum(1 for a in actions if a != 0)
        active_ratio = non_zero_count / len(actions)
        print(f"\nEmpirical Replay Active Signal Ratio: {active_ratio:.2%} ({non_zero_count}/{len(actions)} bars)")
        assert non_zero_count > 0, "Complete action collapse detected: 0 active trades across 50 bars!"
        assert active_ratio >= 0.15, f"Active trading ratio {active_ratio:.2%} too low (< 15%)"
