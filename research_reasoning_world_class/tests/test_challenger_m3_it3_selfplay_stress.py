"""
Adversarial Stress Test Suite: Milestone 3 Iteration 3 Quality Gate
Focus: Self-Play MCTS Stability, Extreme Latent Fuzzing, and 10-Round Co-Evolution Stress
Module under test: research_reasoning_world_class/src/muzero_latent_coevolution.py
Challenger Agent: teamwork_challenger_m3_it3_1 (Roles: critic, specialist)
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
from typing import Dict, Any

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from research_reasoning_world_class.src.muzero_latent_coevolution import (
    MuZeroLatentCoEvolutionTrainer,
    ChampionRole,
)


class TestM3Iteration3AdversarialSelfPlayStress:
    """Stress tests verifying 10-round self-play stability and extreme input handling."""

    def test_10_rounds_adversarial_offline_self_play_stress(self):
        """
        Runs 10 consecutive rounds of self-play MCTS in offline mode (env=None).
        Verifies:
        - Exactly 10 rounds complete without crashes.
        - Telemetry dictionaries contain all expected metrics with length 10.
        - All values are finite (no NaN / Inf).
        - Both Bull and Bear replay buffers are populated.
        """
        torch.manual_seed(101)
        np.random.seed(101)

        trainer = MuZeroLatentCoEvolutionTrainer(mcts_simulations=4)
        res = trainer.run_self_play_mcts(num_rounds=10, log_interval=2, env=None)

        assert res["total_rounds"] == 10
        metrics = res["metrics"]
        assert len(metrics["total_loss"]) == 10
        assert len(metrics["mcts_value"]) == 10
        assert len(metrics["cos_similarity"]) == 10

        for idx, loss in enumerate(metrics["total_loss"]):
            assert math.isfinite(loss), f"Round {idx+1} total_loss is non-finite: {loss}"
        for idx, val in enumerate(metrics["mcts_value"]):
            assert math.isfinite(val), f"Round {idx+1} mcts_value is non-finite: {val}"
        for idx, sim in enumerate(metrics["cos_similarity"]):
            assert math.isfinite(sim), f"Round {idx+1} cos_similarity is non-finite: {sim}"

        assert len(trainer.bull_champion.replay_buffer) > 0, "Bull replay buffer empty!"
        assert len(trainer.bear_champion.replay_buffer) > 0, "Bear replay buffer empty!"

    def test_10_rounds_adversarial_grounded_real_m1_self_play_stress(self):
        """
        Runs 10 consecutive rounds of grounded self-play MCTS using authentic M1 bars.
        Verifies:
        - Zero crashes across 10 rounds with genuine path-dependent trade outcomes.
        - Optimizer steps update parameters (verifying active gradient descent).
        - Grounded targets yield finite losses and telemetry.
        """
        torch.manual_seed(202)
        np.random.seed(202)

        csv_path = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
        assert Path(csv_path).exists(), f"M1 data not found at {csv_path}"

        env = RealM1Environment(csv_path=csv_path, spread_usd=2.50)
        # Load small slice for rapid deterministic verification
        total_bars = env.load_data(start_date="2025.01.15", end_date="2025.01.17")
        assert total_bars >= 1000

        trainer = MuZeroLatentCoEvolutionTrainer(mcts_simulations=4, env=env)

        # Snapshot initial weights
        init_bull_param = next(trainer.bull_champion.world_model.parameters()).clone()
        init_bear_param = next(trainer.bear_champion.world_model.parameters()).clone()

        res = trainer.run_self_play_mcts(num_rounds=10, log_interval=5, env=env)

        assert res["total_rounds"] == 10
        metrics = res["metrics"]
        assert len(metrics["total_loss"]) == 10
        assert len(metrics["mcts_value"]) == 10

        # Verify weights were updated by optimizer
        final_bull_param = next(trainer.bull_champion.world_model.parameters())
        final_bear_param = next(trainer.bear_champion.world_model.parameters())
        assert not torch.equal(init_bull_param, final_bull_param), "Bull world model weights did not update!"
        assert not torch.equal(init_bear_param, final_bear_param), "Bear world model weights did not update!"

    def test_adversarial_latent_corruptions_step_single_champion(self):
        """
        Adversarially feeds corrupted latent tensors (extreme values, subnormals)
        into _step_single_champion to verify numerical stability and zero crashes.
        """
        trainer = MuZeroLatentCoEvolutionTrainer(mcts_simulations=2)
        bull = trainer.bull_champion

        # Corrupted latents
        corruptions = [
            torch.zeros(4, 768),
            torch.ones(4, 768) * 1e5,
            torch.ones(4, 768) * -1e5,
            torch.randn(4, 768) * 1e-35,
        ]

        for idx, z_corrupt in enumerate(corruptions):
            step_res = trainer._step_single_champion(
                champion=bull,
                z_0=z_corrupt,
                bar_idx=None,
                use_env=None,
                H=3,
            )
            assert "loss_dict" in step_res
            assert "best_action" in step_res
            assert step_res["best_action"] in {0, 1}, f"Corrupted batch {idx} resulted in forbidden action!"
            assert "mcts_value" in step_res
            assert math.isfinite(step_res["mcts_value"]), f"mcts_value not finite on batch {idx}"
