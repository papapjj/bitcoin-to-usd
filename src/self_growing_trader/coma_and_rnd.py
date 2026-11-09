"""COMA Centralized Critic, RND Intrinsic Exploration, and Hindsight Experience Replay (HER).

Production-grade implementation for Multi-Agent Reinforcement Learning in SGT-V193:
1. Counterfactual Multi-Agent Policy Gradients (COMA, Foerster et al., AAAI 2018):
   - Centralized Critic Q_phi(s, u) with counterfactual baseline.
   - Computes individual agent advantage: A^a_COMA(s, u) = Q_phi(s, u) - sum_u' pi^a(u'|s) Q_phi(s, (u^{-a}, u')).
   - Eliminates Lazy Agent problem and isolates individual credit/fault.

2. Random Network Distillation (RND, Burda et al., ICLR 2019):
   - Fixed random target network + trainable predictor network.
   - Welford online normalization of intrinsic curiosity rewards.
   - Prevents premature convergence to 100% HOLD policy.

3. Hindsight Experience Replay (HER, Andrychowicz et al., NeurIPS 2017):
   - Relabels near-miss timeout positions (MFE >= 0.75R) into synthetic winning trajectories.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

FIVE_AGENT_COUNT = 5
ACTION_COUNT = 12


class WelfordNormalizer:
    """Online running mean and variance computation via Welford's algorithm."""

    def __init__(self, eps: float = 1e-4) -> None:
        self.eps = float(eps)
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def update(self, val: float) -> None:
        self.count += 1
        delta = val - self.mean
        self.mean += delta / self.count
        delta2 = val - self.mean
        self.m2 += delta * delta2

    @property
    def var(self) -> float:
        if self.count < 2:
            return 1.0
        return max(self.m2 / (self.count - 1), self.eps)

    @property
    def std(self) -> float:
        return math.sqrt(self.var)

    def normalize(self, val: float) -> float:
        return (val - self.mean) / (self.std + self.eps)


class COMACentralizedCritic(nn.Module):
    """Centralized Critic for 5-agent team.
    
    Estimates Q_phi(s, u^{-a}, *) for all actions of agent a given other agents' actions.
    """

    def __init__(
        self,
        state_dim: int = 137,
        num_agents: int = FIVE_AGENT_COUNT,
        action_count: int = ACTION_COUNT,
        action_embed_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.num_agents = int(num_agents)
        self.action_count = int(action_count)
        self.action_embed_dim = int(action_embed_dim)
        self.hidden_dim = int(hidden_dim)

        self.action_embedding = nn.Embedding(self.action_count, self.action_embed_dim)

        # Joint input: state (137) + (num_agents - 1) * action_embed_dim
        other_actions_dim = (self.num_agents - 1) * self.action_embed_dim
        joint_in_dim = self.state_dim + other_actions_dim

        # Q-network per agent: outputs Q-values for all action_count actions
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(joint_in_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.action_count),
            )
            for _ in range(self.num_agents)
        ])

    def get_other_actions_embedding(self, joint_actions: Tensor, agent_id: int) -> Tensor:
        """Collects action embeddings for all agents except agent_id.
        
        Args:
            joint_actions: [B, num_agents] int64 tensor
            agent_id: index in 0..num_agents-1
        Returns:
            [B, (num_agents - 1) * action_embed_dim] tensor
        """
        other_indices = [i for i in range(self.num_agents) if i != agent_id]
        other_actions = joint_actions[:, other_indices]  # [B, num_agents - 1]
        embedded = self.action_embedding(other_actions)  # [B, num_agents - 1, embed_dim]
        return embedded.view(joint_actions.shape[0], -1)

    def forward(
        self,
        state: Tensor,           # [B, state_dim]
        joint_actions: Tensor,   # [B, num_agents]
        agent_id: int,
    ) -> Tensor:
        """Returns Q-values for all actions of agent_id given other agents' actions.
        
        Returns:
            [B, action_count] tensor of Q-values
        """
        other_embed = self.get_other_actions_embedding(joint_actions, agent_id)
        joint_input = torch.cat([state, other_embed], dim=-1)
        return self.q_networks[agent_id](joint_input)

    def compute_counterfactual_advantage(
        self,
        state: Tensor,           # [B, state_dim]
        joint_actions: Tensor,   # [B, num_agents]
        policy_probs: Tensor,    # [B, num_agents, action_count]
        agent_id: int,
    ) -> Tuple[Tensor, Tensor]:
        """Calculates counterfactual baseline and advantage for agent_id.
        
        Returns:
            advantage: [B] counterfactual advantage A^a_COMA
            q_taken: [B] Q-value of the actual action taken
        """
        q_all = self.forward(state, joint_actions, agent_id)  # [B, action_count]
        agent_action = joint_actions[:, agent_id].unsqueeze(-1)  # [B, 1]
        q_taken = q_all.gather(dim=-1, index=agent_action).squeeze(-1)  # [B]

        # Counterfactual baseline: expectation over policy distribution
        probs = policy_probs[:, agent_id, :]  # [B, action_count]
        baseline = torch.sum(probs * q_all, dim=-1)  # [B]

        advantage = q_taken - baseline
        return advantage, q_taken


class RNDExploration(nn.Module):
    """Random Network Distillation (RND) intrinsic curiosity module."""

    def __init__(
        self,
        state_dim: int = 137,
        latent_dim: int = 64,
        kappa: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.latent_dim = int(latent_dim)
        self.kappa = float(kappa)

        # Frozen Target Network
        self.target_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.latent_dim),
        )
        for p in self.target_net.parameters():
            p.requires_grad = False

        # Trainable Predictor Network
        self.predictor_net = nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, self.latent_dim),
        )

        self.normalizer = WelfordNormalizer()

    def compute_intrinsic_reward(self, state: Tensor) -> Tuple[Tensor, Tensor]:
        """Computes intrinsic reward bonus and predictor loss.
        
        Args:
            state: [B, state_dim]
        Returns:
            intrinsic_reward: [B] tensor
            predictor_loss: scalar tensor for training predictor
        """
        with torch.no_grad():
            target_feat = self.target_net(state)  # [B, 64]

        pred_feat = self.predictor_net(state)     # [B, 64]
        raw_error = torch.mean((pred_feat - target_feat) ** 2, dim=-1)  # [B]
        predictor_loss = torch.mean(raw_error)

        # Welford normalization
        rewards = []
        for err in raw_error.detach().cpu().numpy():
            self.normalizer.update(float(err))
            rewards.append(self.normalizer.normalize(float(err)))

        norm_reward = torch.tensor(rewards, dtype=state.dtype, device=state.device)
        # Scale by kappa and clip to prevent runaway curiosity
        intrinsic_reward = self.kappa * torch.clamp(norm_reward, min=-2.0, max=5.0)

        return intrinsic_reward, predictor_loss


@dataclass
class HERTrajectory:
    observations: List[Tensor]
    actions: List[Tensor]
    rewards: List[float]
    mfe: float
    mae: float
    holding_bars: int
    is_timeout: bool


class HindsightExperienceReplay:
    """Hindsight Experience Replay (HER) relabeler for financial lifecycle trading."""

    def __init__(self, mfe_threshold: float = 0.75, harvest_ratio: float = 0.90) -> None:
        self.mfe_threshold = float(mfe_threshold)
        self.harvest_ratio = float(harvest_ratio)

    def should_relabel(self, trajectory: HERTrajectory) -> bool:
        """Determines if a timeout or sub-optimal trade qualifies for hindsight relabeling."""
        return trajectory.is_timeout and (trajectory.mfe >= self.mfe_threshold)

    def relabel_trajectory(self, trajectory: HERTrajectory) -> HERTrajectory:
        """Relabels sub-optimal trajectory into a counterfactual win trajectory."""
        if not self.should_relabel(trajectory):
            return trajectory

        virtual_tp = self.harvest_ratio * trajectory.mfe
        # Synthetic boosted reward for capturing the wave before timeout
        relabeled_rewards = list(trajectory.rewards)
        if relabeled_rewards:
            relabeled_rewards[-1] = float(virtual_tp - 0.20 * trajectory.mae)

        return HERTrajectory(
            observations=trajectory.observations,
            actions=trajectory.actions,
            rewards=relabeled_rewards,
            mfe=trajectory.mfe,
            mae=trajectory.mae,
            holding_bars=trajectory.holding_bars,
            is_timeout=False,  # Converted to successful target achievement
        )
