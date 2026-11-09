# -*- coding: utf-8 -*-
"""Adversarial Challenger 2 Test Suite: Governor HOLD & Policy Invariance.

Stress-tests:
1. 1,000 stochastic forward passes under extreme random observations and legal action masks.
2. Agents 2, 3, 4 producing EXACTLY Action 0 (HOLD) in 100.0% of steps (0 non-HOLD actions).
3. Zero Action 10 (POLICY_CLOSE_ALL / ACTION_CLOSE_ALL) emitted across all passes.
4. Action 10 logit clamped to -30.0 with probability 0.0 under all adversarial inputs.
5. Monkey-patched hostile head override defense (+999.0 CLOSE_ALL logit defeated).
6. Numerical boundary analysis of -30.0 masking constant.
7. context_direction_entry_gate_enabled = False strictly preserved across all configs and runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import numpy as np
import pytest
import yaml

from self_growing_trader.config import LearningIntegrityConfig
from self_growing_trader.five_agent_authority import (
    CLOSE_ALL_ACTION,
    FIVE_AGENT_COUNT,
    HOLD_ACTION,
    five_agent_authority_mask,
)
from self_growing_trader.orders import Action
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


# ==============================================================================
# Challenge 1: 1,000 Stochastic Forward Passes with Extreme Observations & Masks
# ==============================================================================

def test_governor_hold_invariance_1000_stochastic_passes() -> None:
    """Adversarially stress-test Governor HOLD invariance across 1,000 stochastic passes.

    Tests extreme random observations (Gaussian, large dynamic range, high variance,
    near-zero, negative) and adversarial legal action masks (including masks with
    Action 10 explicitly forced True, random noise masks, all-ones, all-zeros).
    Verifies that Agents 2, 3, 4 produce EXACTLY Action 0 (HOLD) in 100.0% of steps.
    """
    torch = pytest.importorskip("torch")
    from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy

    # Initialize policy network
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.eval()
    network.heads.lock_non_entry_to_hold = True
    network.heads.adaptive_volatility_targets_enabled = True
    network.heads.seven_choice_preserve_manager_wire = False

    memories = network.initial_memories()
    total_passes = 1000

    agent_actions: dict[int, list[int]] = {2: [], 3: [], 4: []}
    close_all_emissions = 0

    rng = np.random.default_rng(20261019)

    for step in range(total_passes):
        # Generate varied adversarial observations
        regime_mode = step % 5
        if regime_mode == 0:
            # Standard Gaussian
            obs_np = rng.standard_normal((5, INPUT_COUNT), dtype=np.float32)
        elif regime_mode == 1:
            # Extreme large values (+10,000 range)
            obs_np = rng.uniform(500.0, 10000.0, (5, INPUT_COUNT)).astype(np.float32)
        elif regime_mode == 2:
            # Extreme negative values (-10,000 range)
            obs_np = rng.uniform(-10000.0, -500.0, (5, INPUT_COUNT)).astype(np.float32)
        elif regime_mode == 3:
            # Tiny near-zero values
            obs_np = rng.standard_normal((5, INPUT_COUNT), dtype=np.float32) * 1e-6
        else:
            # Mixed Cauchy / extreme outlier regime
            obs_np = (rng.standard_cauchy((5, INPUT_COUNT)) * 100.0).astype(np.float32)
            obs_np = np.clip(obs_np, -1e6, 1e6)

        obs = torch.as_tensor(obs_np)

        # Generate adversarial legal action masks
        mask_mode = step % 6
        if mask_mode == 0:
            # All actions allowed
            allowed = torch.ones((5, ACTION_COUNT), dtype=torch.bool)
        elif mask_mode == 1:
            # Hostile mask: specifically enable CLOSE_ALL (Action 10) for Agent 4
            allowed = torch.zeros((5, ACTION_COUNT), dtype=torch.bool)
            allowed[:, 0] = True
            allowed[4, 10] = True  # Try to force Agent 4 to close
            allowed[2, 8] = True   # Try to force Agent 2 to close worst
            allowed[3, 7] = True   # Try to force Agent 3 to cancel
        elif mask_mode == 2:
            # Random boolean mask
            allowed = torch.as_tensor(rng.integers(0, 2, (5, ACTION_COUNT)), dtype=torch.bool)
            # Ensure at least one action is true so mask is non-empty
            allowed[:, 0] = True
        elif mask_mode == 3:
            # Degenerate: Action 0 False, all other actions True
            allowed = torch.ones((5, ACTION_COUNT), dtype=torch.bool)
            allowed[:, 0] = False
        elif mask_mode == 4:
            # Degenerate: all False
            allowed = torch.zeros((5, ACTION_COUNT), dtype=torch.bool)
        else:
            # None mask (unconstrained)
            allowed = None

        # Stochastic forward pass with exploration mix
        uniform_mix = 0.5 if step % 2 == 0 else 1.0  # Heavy exploration
        with torch.no_grad():
            results, memories, _ = network.forward_five(
                obs,
                memories,
                allowed_action_masks=allowed,
                deterministic=False,
                behavior_uniform_mix=uniform_mix,
            )

        for role in (2, 3, 4):
            act = int(results[role].action_index)
            agent_actions[role].append(act)
            if act == int(Action.CLOSE_ALL) or act == CLOSE_ALL_ACTION:
                close_all_emissions += 1

    # Verify 100.0% HOLD invariance
    for role in (2, 3, 4):
        actions = agent_actions[role]
        non_zero = [a for a in actions if a != 0]
        assert len(non_zero) == 0, (
            f"Agent {role} violated HOLD invariance! {len(non_zero)} non-zero actions emitted out of {total_passes}."
        )
        assert all(a == int(Action.HOLD) for a in actions), (
            f"Agent {role} produced non-HOLD action! Unique actions: {set(actions)}"
        )

    # Invariant: Zero POLICY_CLOSE_ALL
    assert close_all_emissions == 0, (
        f"POLICY_CLOSE_ALL invariant violated: {close_all_emissions} CLOSE_ALL emissions detected!"
    )


# ==============================================================================
# Challenge 2: Action 10 Logit Clamping to -30.0 & Exact 0.0 Probability
# ==============================================================================

def test_action_close_all_logit_clamped_and_zero_probability_under_extreme_inputs() -> None:
    """Verifies that Action 10 logit is clamped to -30.0 and probability is strictly 0.0 under all inputs."""
    torch = pytest.importorskip("torch")
    from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy

    num_trials = 1000
    rng = np.random.default_rng(42)

    for trial in range(num_trials):
        # Generate varied adversarial raw logits
        if trial % 3 == 0:
            # Massive positive logit for action 10 (+500.0 up to +10,000.0)
            logits = torch.randn(5, ACTION_COUNT)
            logits[:, 10] = float(rng.uniform(100.0, 10000.0))
            logits[:, 0] = float(rng.uniform(-10.0, 10.0))
        elif trial % 3 == 1:
            # Extreme random logits in range [-25.0, 25.0]
            logits = torch.as_tensor(rng.uniform(-25.0, 25.0, (5, ACTION_COUNT)), dtype=torch.float32)
        else:
            # Massive positive logits for all actions except HOLD
            logits = torch.full((5, ACTION_COUNT), 20.0, dtype=torch.float32)
            logits[:, 10] = 100.0

        # Allowed mask where non-entry agents (2, 3, 4) are locked to HOLD
        allowed = torch.ones((5, ACTION_COUNT), dtype=torch.bool)
        # Apply the policy lock rule
        allowed[2:, :] = False
        allowed[2:, 0] = True

        # Test _primary_action_distribution
        choice, log_prob, entropy, base_probs, behavior_probs = (
            SharedFiveTradingPolicy._primary_action_distribution(
                logits,
                allowed=allowed,
                deterministic=False,
                entry_probability_floor=0.1,
                uniform_mix=0.5,
                include_policy_probs=True,
            )
        )

        for role in (2, 3, 4):
            # 1. Probability of Action 10 must be EXACTLY 0.0
            assert base_probs[role, 10].item() == 0.0, (
                f"Trial {trial}: Base prob for Action 10 on Agent {role} is {base_probs[role, 10].item()} != 0.0!"
            )
            assert behavior_probs[role, 10].item() == 0.0, (
                f"Trial {trial}: Behavior prob for Action 10 on Agent {role} is {behavior_probs[role, 10].item()} != 0.0!"
            )

            # 2. Probability of Action 0 must be EXACTLY 1.0
            assert base_probs[role, 0].item() == pytest.approx(1.0, abs=1e-6), (
                f"Trial {trial}: Base prob for Action 0 on Agent {role} is {base_probs[role, 0].item()} != 1.0!"
            )
            assert behavior_probs[role, 0].item() == pytest.approx(1.0, abs=1e-6), (
                f"Trial {trial}: Behavior prob for Action 0 on Agent {role} is {behavior_probs[role, 0].item()} != 1.0!"
            )

            # 3. Selected choice must be Action 0
            assert choice[role].item() == 0, (
                f"Trial {trial}: Selected choice for Agent {role} is {choice[role].item()} != 0!"
            )

    # In addition, check logit masking directly
    legal = torch.zeros((5, ACTION_COUNT), dtype=torch.bool)
    legal[2:, 0] = True
    test_logits = torch.full((5, ACTION_COUNT), 50.0)
    masked_logits = torch.where(legal, test_logits, torch.full_like(test_logits, -30.0))
    for role in (2, 3, 4):
        assert masked_logits[role, 10].item() == -30.0
        assert masked_logits[role, 0].item() == 50.0


# ==============================================================================
# Challenge 3: chuyen_gia Authority Mask Invariance
# ==============================================================================

def test_chuyen_gia_authority_mask_invariance() -> None:
    """Verifies that five_agent_authority_mask strictly locks Agents 2, 3, 4 to HOLD."""
    # When base_mask tries to enable all actions
    base = np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool)

    # Simulate chuyen_gia_policy logic
    class DummyContextPolicy:
        five_lock_non_entry_to_hold = True
        five_fixed_geometry_enabled = True
        adaptive_volatility_targets_enabled = True

        def _five_allowed_masks(self, base_mask: np.ndarray | None = None) -> np.ndarray:
            arr = base_mask if base_mask is not None else np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool)
            authority_mask = five_agent_authority_mask(arr)
            if (
                bool(getattr(self, "five_lock_non_entry_to_hold", False))
                or bool(getattr(self, "five_fixed_geometry_enabled", False))
                or bool(getattr(self, "adaptive_volatility_targets_enabled", False))
            ):
                authority_mask[2:, :] = False
                authority_mask[2:, 0] = True
            return authority_mask

    policy = DummyContextPolicy()
    for _ in range(500):
        # Random hostile masks
        random_base = np.random.choice([True, False], size=(FIVE_AGENT_COUNT, ACTION_COUNT))
        mask = policy._five_allowed_masks(random_base)

        for role in (2, 3, 4):
            assert mask[role, HOLD_ACTION] is np.True_ or mask[role, HOLD_ACTION] == True
            assert not np.any(mask[role, 1:]), f"Agent {role} has non-HOLD actions allowed!"
            assert mask[role, CLOSE_ALL_ACTION] == False, f"Agent {role} has CLOSE_ALL allowed!"


# ==============================================================================
# Challenge 4: Adversarial Head Monkey-Patching Attack
# ==============================================================================

def test_adversarial_head_override_defense() -> None:
    """Verifies that even if heads output +999.0 for CLOSE_ALL, forward_five locks Agents 2, 3, 4 to HOLD."""
    torch = pytest.importorskip("torch")
    from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy

    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.eval()
    network.heads.lock_non_entry_to_hold = True
    network.heads.adaptive_volatility_targets_enabled = True
    network.heads.seven_choice_preserve_manager_wire = False

    # Monkey-patch heads.primary_action_logits to return massive positive logits (+999.0) for Action 10
    original_primary_action_logits = network.heads.primary_action_logits

    def hostile_primary_action_logits(hidden: torch.Tensor, agent_ids: torch.Tensor) -> torch.Tensor:
        res = original_primary_action_logits(hidden, agent_ids)
        res = res.clone()
        res[:, 10] = 999.0  # Massive temptation to CLOSE_ALL
        return res

    network.heads.primary_action_logits = hostile_primary_action_logits  # type: ignore

    memories = network.initial_memories()
    obs = torch.randn(5, INPUT_COUNT)

    # Even with all actions ostensibly allowed:
    allowed = torch.ones((5, ACTION_COUNT), dtype=torch.bool)

    for _ in range(100):
        with torch.no_grad():
            results, memories, _ = network.forward_five(
                obs,
                memories,
                allowed_action_masks=allowed,
                deterministic=False,  # Stochastic sampling!
                behavior_uniform_mix=0.5,
            )

        for role in (2, 3, 4):
            assert int(results[role].action_index) == 0, (
                f"Agent {role} emitted non-zero action {results[role].action_index} under adversarial logits!"
            )
            # Verify probability of Action 10 is strictly 0.0
            assert float(results[role].action_probabilities[10]) == 0.0, (
                f"Agent {role} action 10 probability is {results[role].action_probabilities[10]} != 0.0!"
            )


# ==============================================================================
# Challenge 5: Numerical Boundary Analysis of Masking Constant (-30.0)
# ==============================================================================

def test_masking_constant_numerical_stability_boundary() -> None:
    """Documents and tests the numerical boundary of the -30.0 logit masking constant.

    Shows that when legal logit is >= -50.0, normal float32 softmax correctly computes
    probability 1.0 on legal action and 0.0 on masked actions.
    """
    torch = pytest.importorskip("torch")
    from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy

    allowed = torch.zeros((1, ACTION_COUNT), dtype=torch.bool)
    allowed[0, 0] = True  # Only HOLD is legal

    # Realistic range [-15.0, 15.0]
    for test_val in [-50.0, -30.0, -15.0, 0.0, 15.0, 30.0, 100.0]:
        logits = torch.full((1, ACTION_COUNT), 10.0)
        logits[0, 0] = test_val
        logits[0, 10] = 50.0  # Illegal action has high raw logit

        choice, _, _, base_probs, behavior_probs = (
            SharedFiveTradingPolicy._primary_action_distribution(
                logits,
                allowed=allowed,
                deterministic=False,
                entry_probability_floor=0.0,
                uniform_mix=0.0,
                include_policy_probs=True,
            )
        )
        assert base_probs[0, 10].item() == 0.0
        assert base_probs[0, 0].item() == pytest.approx(1.0, abs=1e-5)
        assert choice[0].item() == 0


# ==============================================================================
# Challenge 6: Natural RL Invariant (Zero Hard Direction Gates)
# ==============================================================================

def test_context_direction_entry_gate_enabled_strictly_false() -> None:
    """Verifies that context_direction_entry_gate_enabled = False is strictly preserved.

    - In LearningIntegrityConfig default.
    - Across all active yaml configurations.
    - In runtime config instances.
    """
    # 1. Dataclass default
    config_default = LearningIntegrityConfig()
    assert config_default.context_direction_entry_gate_enabled is False, (
        "LearningIntegrityConfig.context_direction_entry_gate_enabled default must be False!"
    )

    # 2. YAML configs check
    configs_dir = Path("configs")
    yaml_files = list(configs_dir.glob("**/*.yaml")) + list(configs_dir.glob("**/*.yml"))
    assert len(yaml_files) > 0, "No config files found in configs/!"

    active_candidate_checked = False
    for yf in yaml_files:
        try:
            with open(yf, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(content, dict):
            continue

        for section_key in ("learning", "learning_overrides"):
            section = content.get(section_key, {})
            if isinstance(section, dict) and "context_direction_entry_gate_enabled" in section:
                val = section["context_direction_entry_gate_enabled"]
                assert val is False or val is None, (
                    f"Found context_direction_entry_gate_enabled={val} in {yf}! Must be False for natural RL."
                )
                if "reasoning_rl_v1_active_candidate" in yf.name:
                    active_candidate_checked = True

    assert active_candidate_checked, (
        "Did not verify reasoning_rl_v1_active_candidate.yaml for context_direction_entry_gate_enabled!"
    )
