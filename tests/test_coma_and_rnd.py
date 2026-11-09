"""Unit tests for COMA Centralized Critic, RND Exploration, and Hindsight Experience Replay (HER).

Verifies:
1. COMACentralizedCritic outputs and counterfactual advantage computation.
2. Counterfactual credit isolation: bad trader penalized, holding agents neutral.
3. RND intrinsic exploration reward calculation and Welford normalizer stability.
4. HER relabeler converts qualifying timeout trajectories into synthetic wins.
"""

import pytest
import torch

from src.self_growing_trader.coma_and_rnd import (
    COMACentralizedCritic,
    HERTrajectory,
    HindsightExperienceReplay,
    RNDExploration,
    WelfordNormalizer,
)


def test_welford_normalizer():
    normalizer = WelfordNormalizer()
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    for v in values:
        normalizer.update(v)

    assert abs(normalizer.mean - 3.0) < 1e-4
    assert normalizer.count == 5
    assert normalizer.std > 0.0

    # Test normalization
    norm_val = normalizer.normalize(3.0)
    assert abs(norm_val) < 1e-4


def test_coma_centralized_critic_shapes_and_advantage():
    B = 8
    state_dim = 137
    num_agents = 5
    action_count = 12

    critic = COMACentralizedCritic(state_dim=state_dim, num_agents=num_agents, action_count=action_count)
    state = torch.randn(B, state_dim)
    joint_actions = torch.randint(0, action_count, (B, num_agents))
    policy_probs = torch.softmax(torch.randn(B, num_agents, action_count), dim=-1)

    # Test Q-values output for Agent 0
    q_vals = critic(state, joint_actions, agent_id=0)
    assert q_vals.shape == (B, action_count)

    # Test counterfactual advantage
    adv, q_taken = critic.compute_counterfactual_advantage(state, joint_actions, policy_probs, agent_id=0)
    assert adv.shape == (B,)
    assert q_taken.shape == (B,)
    assert not torch.isnan(adv).any()


def test_coma_credit_assignment_isolation():
    """Verify that COMA isolates credit: an agent choosing a suboptimal action has negative advantage."""
    state_dim = 137
    critic = COMACentralizedCritic(state_dim=state_dim, num_agents=5, action_count=12)

    # Mock scenario: state s, actions where Agent 0 chose action 1 (BUY) while others chose 0 (HOLD)
    state = torch.randn(1, state_dim)
    joint_actions = torch.tensor([[1, 0, 0, 0, 0]])
    # Uniform policy for agent 0
    policy_probs = torch.full((1, 5, 12), 1.0 / 12.0)

    # We manually set Q-values so that action 0 (HOLD) has value 0.0, action 1 (BUY) has value -2.0, others 0.0
    with torch.no_grad():
        for p in critic.q_networks[0].parameters():
            p.zero_()
        # bias of final layer: set action 1 to -2.0
        critic.q_networks[0][-1].bias.fill_(0.0)
        critic.q_networks[0][-1].bias[1] = -2.0

    adv_0, q_0 = critic.compute_counterfactual_advantage(state, joint_actions, policy_probs, agent_id=0)
    # q_taken = -2.0, baseline = (-2.0 / 12) = -0.1667 => advantage = -2.0 - (-0.1667) = -1.8333 < 0!
    assert adv_0.item() < 0.0, f"Expected negative advantage for bad trade, got {adv_0.item()}"


def test_rnd_intrinsic_exploration():
    rnd = RNDExploration(state_dim=137, latent_dim=64, kappa=0.05)
    state = torch.randn(16, 137)

    reward, pred_loss = rnd.compute_intrinsic_reward(state)
    assert reward.shape == (16,)
    assert pred_loss.shape == ()
    assert not torch.isnan(reward).any()
    assert pred_loss.item() >= 0.0

    # Ensure target network weights do not receive gradients
    pred_loss.backward()
    for p in rnd.target_net.parameters():
        assert p.grad is None
    for p in rnd.predictor_net.parameters():
        assert p.grad is not None


def test_hindsight_experience_replay_relabeling():
    her = HindsightExperienceReplay(mfe_threshold=0.75, harvest_ratio=0.90)

    # Scenario 1: Timeout trade with high MFE (e.g. 1.20R) but closed at 0.10R with MAE 0.15R
    traj_qualifying = HERTrajectory(
        observations=[torch.randn(137)],
        actions=[torch.tensor(1)],
        rewards=[0.10],
        mfe=1.20,
        mae=0.15,
        holding_bars=120,
        is_timeout=True,
    )
    assert her.should_relabel(traj_qualifying)

    relabeled = her.relabel_trajectory(traj_qualifying)
    assert not relabeled.is_timeout
    expected_virtual_tp = 0.90 * 1.20 - 0.20 * 0.15  # 1.08 - 0.03 = 1.05R
    assert abs(relabeled.rewards[-1] - expected_virtual_tp) < 1e-4

    # Scenario 2: Timeout trade with low MFE (0.40R) -> should not relabel
    traj_low_mfe = HERTrajectory(
        observations=[torch.randn(137)],
        actions=[torch.tensor(1)],
        rewards=[-0.15],
        mfe=0.40,
        mae=0.50,
        holding_bars=120,
        is_timeout=True,
    )
    assert not her.should_relabel(traj_low_mfe)
    unmodified = her.relabel_trajectory(traj_low_mfe)
    assert unmodified.is_timeout
    assert unmodified.rewards[-1] == -0.15
