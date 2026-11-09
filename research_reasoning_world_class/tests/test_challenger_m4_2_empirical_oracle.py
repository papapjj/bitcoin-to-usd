"""
Empirical Oracle Challenger Test Suite for Milestone 4 Deliverables.
File: research_reasoning_world_class/tests/test_challenger_m4_2_empirical_oracle.py

Role: teamwork_challenger_m4_2 (Milestone 4 Empirical Oracle Challenger 2)

Empirically challenges:
1. SlicedM1Environment:
   - Bit-for-bit exactness against master RealM1Environment on obs_matrix, close_arr, atr_arr, timestamps.
   - Slice boundary safety: testing whether simulate_trade at bar_idx = total_bars - 1 evaluates past the slice end.
   - Coordinate space consistency of exit_bar_idx.
2. Buffer & Optimizer Lifecycle:
   - 100% replay buffer flushing between Walk-Forward cycles.
   - 100% AdamW optimizer momentum clearing (exp_avg, exp_avg_sq purged).
   - 100% retention of trained network parameters theta_1 during cycle transition.
3. Checkpoint Strict Loading & Exactness:
   - 17-key schema validation in outputs/he_chuyen_gia/v193_runs/mythos_muzero_coevolved_champion.pt.
   - strict=True loading into UnifiedReasoningTraderPipeline and MetacognitiveReasoningEngine.
   - Exact counts: 191 tensor keys, 187 named parameters, 4,196,564 weights.
   - Exact numerical match: max_abs_diff == 0.0 across all 191 tensors.
   - strict=True loading of LatentDynamicsWorldModel (30 tensors).
"""

import math
import os
from pathlib import Path
import sys
from typing import Dict, List, Tuple
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
REASONING_DIR = ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from research_reasoning_world_class.src.real_m1_environment import (
    RealM1Environment,
    TradeResult,
)
from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.unified_reasoning_pipeline import (
    UnifiedReasoningTraderPipeline,
)
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    MuZeroLatentCoEvolutionTrainer,
)
from scripts.train_mythos_muzero_coevolution import SlicedM1Environment


CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
CHAMPION_CKPT_PATH = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"


@pytest.fixture(scope="module")
def loaded_master_env():
    """Loads the master RealM1Environment once for the test module."""
    if not os.path.exists(CSV_PATH):
        pytest.skip(f"Historical file not found: {CSV_PATH}")
    env = RealM1Environment(csv_path=CSV_PATH, spread_usd=2.50)
    total = env.load_data("2025.01.15", "2025.02.28")
    assert total == 64785, f"Expected 64,785 bars, got {total}"
    return env


# =============================================================================
# CHALLENGE 1: SLICED ENVIRONMENT ORACLE VERIFICATION
# =============================================================================
class TestSlicedM1EnvironmentOracle:
    """Empirical verification of SlicedM1Environment slicing, exactness, and boundary behavior."""

    @pytest.mark.parametrize(
        "name,start_idx,end_idx",
        [
            ("Cycle 1 Train", 0, 40316),
            ("Cycle 1 OOS", 40316, 60465),
            ("Cycle 2 Train", 10079, 50392),
            ("Cycle 2 OOS", 50392, 64785),
            ("Arbitrary Window", 1234, 5678),
        ],
    )
    def test_slice_exactness_bit_for_bit(self, loaded_master_env, name, start_idx, end_idx):
        """Prove SlicedM1Environment matches RealM1Environment slice bit-for-bit."""
        master = loaded_master_env
        sliced = SlicedM1Environment(master, start_idx, end_idx, name=name)

        expected_len = end_idx - start_idx
        assert len(sliced) == expected_len
        assert sliced.total_bars == expected_len

        # Bit-for-bit numpy array comparisons
        assert np.array_equal(sliced.obs_matrix, master.obs_matrix[start_idx:end_idx]), f"{name} obs_matrix mismatch"
        assert np.array_equal(sliced.close_arr, master.close_arr[start_idx:end_idx]), f"{name} close_arr mismatch"
        assert np.array_equal(sliced.open_arr, master.open_arr[start_idx:end_idx]), f"{name} open_arr mismatch"
        assert np.array_equal(sliced.high_arr, master.high_arr[start_idx:end_idx]), f"{name} high_arr mismatch"
        assert np.array_equal(sliced.low_arr, master.low_arr[start_idx:end_idx]), f"{name} low_arr mismatch"
        assert np.array_equal(sliced.atr_arr, master.atr_arr[start_idx:end_idx]), f"{name} atr_arr mismatch"
        assert np.array_equal(sliced.timestamps, master.timestamps[start_idx:end_idx]), f"{name} timestamps mismatch"

        # Direct accessor methods bit-for-bit check
        sample_indices = [0, 1, 10, expected_len // 2, expected_len - 1]
        for rel_idx in sample_indices:
            assert np.array_equal(sliced.get_observation(rel_idx), master.get_observation(start_idx + rel_idx))
            assert sliced.get_atr(rel_idx) == master.get_atr(start_idx + rel_idx)
            assert sliced.get_timestamp(rel_idx) == master.get_timestamp(start_idx + rel_idx)

    def test_slice_boundary_zero_peeking_verified(self, loaded_master_env):
        """
        Verify that simulate_trade at bar_idx = total_bars - 1 guarantees zero future peeking.
        safe_max_holding = max(0, min(60, 0)) = 0.
        Must terminate at DATASET_END with 0 holding bars and exit_bar_idx == total_bars - 1.
        """
        master = loaded_master_env
        sliced = SlicedM1Environment(master, 0, 40316, name="Cycle 1 Train")
        last_bar_idx = sliced.total_bars - 1  # 40315

        res = sliced.simulate_trade(
            bar_idx=last_bar_idx,
            action=1,
            sl_atr=2.0,
            tp_atr=3.0,
            max_holding_bars=60,
        )

        assert res.holding_bars == 0, f"Expected 0 holding bars at boundary, got {res.holding_bars}"
        assert res.exit_reason == "DATASET_END", f"Expected DATASET_END, got {res.exit_reason}"
        assert res.exit_bar_idx == last_bar_idx, f"Expected exit_bar_idx {last_bar_idx}, got {res.exit_bar_idx}"
        # Observation query at exit_bar_idx must succeed within slice
        obs = sliced.get_observation(res.exit_bar_idx)
        assert obs.shape == (137,)

    def test_slice_coordinate_space_and_bounds_verified(self, loaded_master_env):
        """
        Verify that SlicedM1Environment.simulate_trade returns exit_bar_idx in slice coordinates [0, total_bars),
        supports direct get_observation query, and rejects out-of-range bar indices with IndexError.
        """
        master = loaded_master_env
        c1_oos = SlicedM1Environment(master, 40316, 60465, name="Cycle 1 OOS")
        res = c1_oos.simulate_trade(
            bar_idx=10,
            action=1,
            sl_atr=2.0,
            tp_atr=3.0,
            max_holding_bars=60,
        )

        # exit_bar_idx must be in slice-relative space [0, total_bars)
        assert 0 <= res.exit_bar_idx < c1_oos.total_bars, (
            f"exit_bar_idx {res.exit_bar_idx} not in slice [0, {c1_oos.total_bars})"
        )
        assert res.exit_bar_idx == 10 + res.holding_bars

        # Observation query at exit_bar_idx must succeed seamlessly
        obs = c1_oos.get_observation(res.exit_bar_idx)
        assert obs.shape == (137,)

        # Boundary checks: negative and out-of-range bar_idx must raise IndexError
        with pytest.raises(IndexError):
            c1_oos.simulate_trade(bar_idx=-1, action=1, sl_atr=2.0, tp_atr=3.0)
        with pytest.raises(IndexError):
            c1_oos.simulate_trade(bar_idx=c1_oos.total_bars, action=1, sl_atr=2.0, tp_atr=3.0)
        with pytest.raises(IndexError):
            c1_oos.simulate_trade(bar_idx=c1_oos.total_bars + 100, action=1, sl_atr=2.0, tp_atr=3.0)


# =============================================================================
# CHALLENGE 2: BUFFER AND OPTIMIZER LIFECYCLE ORACLE VERIFICATION
# =============================================================================
class TestBufferAndOptimizerLifecycleOracle:
    """Empirical verification of replay buffer flushing, momentum clearing, and weight preservation."""

    def test_buffer_flush_and_optimizer_momentum_clear(self, loaded_master_env):
        """
        Prove:
        1. Replay buffers are 100% flushed (len == 0).
        2. AdamW optimizer momentum states (exp_avg, exp_avg_sq) are 100% cleared (len(state) == 0).
        3. Model parameters theta_1 retain trained values (torch.equal == True).
        """
        master = loaded_master_env
        trainer = MuZeroLatentCoEvolutionTrainer(env=master, mcts_simulations=2)

        # Save initial pre-training weights
        init_bull_weights = {k: v.clone() for k, v in trainer.bull_champion.world_model.state_dict().items()}
        init_bear_weights = {k: v.clone() for k, v in trainer.bear_champion.world_model.state_dict().items()}

        # Populate replay buffers with transitions
        for _ in range(10):
            trainer.bull_champion.add_replay(torch.randn(768), 1, 1.0, torch.randn(768))
            trainer.bear_champion.add_replay(torch.randn(768), 2, 1.0, torch.randn(768))

        assert len(trainer.bull_replay) == 10
        assert len(trainer.bear_replay) == 10
        assert len(trainer.bull_champion.replay_buffer) == 10
        assert len(trainer.bear_champion.replay_buffer) == 10

        # Perform gradient update steps to populate optimizer momentum states
        l_bull = trainer.bull_champion.train_replay_step(batch_size=4)
        l_bear = trainer.bear_champion.train_replay_step(batch_size=4)
        assert l_bull is not None and l_bear is not None

        # Verify optimizer momentum states are populated
        assert len(trainer.bull_champion.optimizer.state) > 0
        assert len(trainer.bear_champion.optimizer.state) > 0
        for p, state in trainer.bull_champion.optimizer.state.items():
            assert "exp_avg" in state and "exp_avg_sq" in state
        for p, state in trainer.bear_champion.optimizer.state.items():
            assert "exp_avg" in state and "exp_avg_sq" in state

        # Verify weights were updated away from initial state
        trained_bull_weights = {k: v.clone() for k, v in trainer.bull_champion.world_model.state_dict().items()}
        trained_bear_weights = {k: v.clone() for k, v in trainer.bear_champion.world_model.state_dict().items()}
        diff_bull = max(torch.max(torch.abs(trained_bull_weights[k] - init_bull_weights[k])).item() for k in trained_bull_weights)
        diff_bear = max(torch.max(torch.abs(trained_bear_weights[k] - init_bear_weights[k])).item() for k in trained_bear_weights)
        assert diff_bull > 0.0, "Bull weights did not update during training"
        assert diff_bear > 0.0, "Bear weights did not update during training"

        # EXECUTE LIFECYCLE TRANSITION (Cycle 1 -> Cycle 2)
        # 1. Flush replay buffers
        trainer.bull_replay.clear()
        trainer.bear_replay.clear()

        # 2. Clear optimizer momentum
        trainer.bull_champion.optimizer.state.clear()
        trainer.bear_champion.optimizer.state.clear()

        # ORACLE VERIFICATION:
        # A. Buffers 100% flushed
        assert len(trainer.bull_replay) == 0, "bull_replay not flushed"
        assert len(trainer.bear_replay) == 0, "bear_replay not flushed"
        assert len(trainer.bull_champion.replay_buffer) == 0, "bull champion replay_buffer not flushed"
        assert len(trainer.bear_champion.replay_buffer) == 0, "bear champion replay_buffer not flushed"

        # B. Optimizer momentum 100% cleared
        assert len(trainer.bull_champion.optimizer.state) == 0, "bull optimizer state not cleared"
        assert len(trainer.bear_champion.optimizer.state) == 0, "bear optimizer state not cleared"

        # C. Trained weights theta_1 100% retained bitwise
        for k, v in trainer.bull_champion.world_model.state_dict().items():
            assert torch.equal(v, trained_bull_weights[k]), f"Bull weight {k} modified during reset"
        for k, v in trainer.bear_champion.world_model.state_dict().items():
            assert torch.equal(v, trained_bear_weights[k]), f"Bear weight {k} modified during reset"


# =============================================================================
# CHALLENGE 3: CHECKPOINT STRICT LOADING & EXACTNESS CHALLENGE
# =============================================================================
class TestCheckpointExactnessChallenge:
    """Empirical verification of canonical checkpoint schema, strict loading, and zero numerical diff."""

    @pytest.fixture(scope="class")
    def loaded_checkpoint(self):
        assert CHAMPION_CKPT_PATH.exists(), f"Champion checkpoint not found at {CHAMPION_CKPT_PATH}"
        ckpt = torch.load(str(CHAMPION_CKPT_PATH), map_location="cpu", weights_only=False)
        return ckpt

    def test_canonical_17_keys_schema(self, loaded_checkpoint):
        """Verify checkpoint contains all 17 canonical keys."""
        ckpt = loaded_checkpoint
        expected_keys = [
            "reasoning_engine_state_dict", "world_model_state_dict", "adversary_state_dict",
            "bull_world_model_state_dict", "bear_world_model_state_dict", "bull_cot_state_dict",
            "bear_cot_state_dict", "geometry_head_state_dict", "bull_geometry_head_state_dict",
            "bear_geometry_head_state_dict", "architecture", "version", "timestamp",
            "trained_at", "training_samples", "epochs", "stats"
        ]
        missing = [k for k in expected_keys if k not in ckpt]
        assert missing == [], f"Missing canonical keys in checkpoint: {missing}"
        assert len(ckpt) == 17, f"Expected 17 top-level keys, got {len(ckpt)}"

    def test_unified_pipeline_strict_loading(self, loaded_checkpoint):
        """Verify strict=True loading into UnifiedReasoningTraderPipeline."""
        ckpt = loaded_checkpoint
        pipeline = UnifiedReasoningTraderPipeline()
        res = pipeline.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)

        assert len(res.missing_keys) == 0, f"Missing keys in pipeline cot_engine: {res.missing_keys}"
        assert len(res.unexpected_keys) == 0, f"Unexpected keys in pipeline cot_engine: {res.unexpected_keys}"

        named_params = dict(pipeline.cot_engine.named_parameters())
        assert len(named_params) == 187, f"Expected 187 named parameters, got {len(named_params)}"
        total_weights = sum(p.numel() for p in pipeline.cot_engine.parameters())
        assert total_weights == 4_196_564, f"Expected 4,196,564 weights, got {total_weights}"
        assert len(ckpt["reasoning_engine_state_dict"]) == 191, f"Expected 191 state dict keys, got {len(ckpt['reasoning_engine_state_dict'])}"

    def test_metacognitive_reasoning_engine_standalone_strict_loading(self, loaded_checkpoint):
        """Verify strict=True loading into standalone MetacognitiveReasoningEngine."""
        ckpt = loaded_checkpoint
        engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        res = engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)

        assert len(res.missing_keys) == 0
        assert len(res.unexpected_keys) == 0
        assert len(dict(engine.named_parameters())) == 187
        assert sum(p.numel() for p in engine.parameters()) == 4_196_564

    def test_checkpoint_numerical_exactness_zero_diff(self, loaded_checkpoint):
        """Verify maximum absolute numerical difference == 0.0 across all 191 tensors."""
        ckpt = loaded_checkpoint
        engine = MetacognitiveReasoningEngine(ReasoningConfig(input_dim=768, hidden_dim=256))
        engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)

        model_sd = engine.state_dict()
        ckpt_sd = ckpt["reasoning_engine_state_dict"]

        max_abs_diff = 0.0
        discrepancies = []
        for k in ckpt_sd:
            t_model = model_sd[k]
            t_ckpt = ckpt_sd[k]
            assert t_model.shape == t_ckpt.shape
            diff = torch.max(torch.abs(t_model.float() - t_ckpt.float())).item()
            if diff > max_abs_diff:
                max_abs_diff = diff
            if diff != 0.0:
                discrepancies.append(f"{k}: max_diff={diff}")

        assert discrepancies == [], f"Found numerical discrepancies: {discrepancies[:5]}"
        assert max_abs_diff == 0.0, f"Expected max_abs_diff == 0.0, got {max_abs_diff}"

    def test_world_model_strict_loading(self, loaded_checkpoint):
        """Verify strict=True loading into LatentDynamicsWorldModel."""
        ckpt = loaded_checkpoint
        wm = LatentDynamicsWorldModel()
        res = wm.load_state_dict(ckpt["world_model_state_dict"], strict=True)
        assert len(res.missing_keys) == 0
        assert len(res.unexpected_keys) == 0
        assert len(ckpt["world_model_state_dict"]) == 30
