"""PPO + frozen-reference KL objective for the RLHF policy stage.

The function here is an objective primitive, not a promotion mechanism.  It
expects rewards/advantages already produced by an independently frozen reward
model and keeps the reference logits detached.  The caller must still use the
single existing policy optimizer and the MT5/OOS gates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor


@dataclass(frozen=True)
class RLHFPPOConfig:
    clip_epsilon: float = 0.20
    kl_beta: float = 0.02
    value_loss_weight: float = 0.50
    entropy_weight: float = 0.001

    def __post_init__(self) -> None:
        if not 0.0 < float(self.clip_epsilon) < 1.0:
            raise ValueError("RLHF PPO clip_epsilon must be in (0,1)")
        if float(self.kl_beta) < 0.0:
            raise ValueError("RLHF PPO kl_beta must be non-negative")
        if float(self.value_loss_weight) < 0.0 or float(self.entropy_weight) < 0.0:
            raise ValueError("RLHF PPO loss weights must be non-negative")


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    selected = values[mask]
    if selected.numel() == 0:
        return values.sum() * 0.0
    return selected.mean()


def rlhf_ppo_loss_from_logits(
    *,
    new_logits: Tensor,
    old_logits: Tensor,
    reference_logits: Tensor,
    actions: Tensor,
    advantages: Tensor,
    values: Tensor,
    returns: Tensor,
    new_values: Tensor | None = None,
    allowed_action_mask: Tensor | None = None,
    sample_mask: Tensor | None = None,
    config: RLHFPPOConfig | None = None,
    ) -> tuple[Tensor, dict[str, object]]:
    """Compute one PPO update from reward-model advantages.

    ``old_logits`` are the rollout policy and ``reference_logits`` are a
    frozen anchor.  The exact categorical KL is computed over all actions;
    this avoids the unstable sampled-log-ratio approximation.  No optimizer
    step occurs here so the existing sole-writer policy optimizer remains the
    only parameter mutator.
    """

    cfg = config or RLHFPPOConfig()
    if new_logits.ndim != 2 or old_logits.shape != new_logits.shape:
        raise ValueError("new_logits and old_logits must have shape [B,A]")
    if reference_logits.shape != new_logits.shape:
        raise ValueError("reference_logits must have the same shape as logits")
    batch, action_count = new_logits.shape
    actions = actions.to(device=new_logits.device, dtype=torch.long).reshape(-1)
    advantages = advantages.to(device=new_logits.device, dtype=new_logits.dtype).reshape(-1)
    values = values.to(device=new_logits.device, dtype=new_logits.dtype).reshape(-1)
    current_values = (
        values
        if new_values is None
        else new_values.to(device=new_logits.device, dtype=new_logits.dtype).reshape(-1)
    )
    returns = returns.to(device=new_logits.device, dtype=new_logits.dtype).reshape(-1)
    if not all(value.numel() == batch for value in (
        actions, advantages, values, current_values, returns
    )):
        raise ValueError("RLHF PPO sample tensors must have batch length B")
    if bool(torch.any(actions < 0)) or bool(torch.any(actions >= action_count)):
        raise ValueError("RLHF PPO action is outside the policy surface")
    if not all(torch.isfinite(value).all() for value in (
        new_logits, old_logits, reference_logits, advantages, values,
        current_values, returns
    )):
        raise ValueError("RLHF PPO received a non-finite rollout tensor")
    action_mask = None
    if allowed_action_mask is not None:
        action_mask = allowed_action_mask.to(
            device=new_logits.device, dtype=torch.bool
        )
        if action_mask.shape != new_logits.shape:
            raise ValueError("allowed_action_mask must have shape [B,A]")
        if bool(torch.any(action_mask.sum(dim=-1) == 0)):
            raise ValueError("RLHF PPO has a row with no legal action")
        if bool(torch.any(~action_mask.gather(1, actions.unsqueeze(-1)).squeeze(-1))):
            raise ValueError("RLHF PPO rollout action is masked out")
        invalid_fill = torch.full_like(new_logits, -1e9)
        new_logits = torch.where(action_mask, new_logits, invalid_fill)
        old_logits = torch.where(action_mask, old_logits, invalid_fill)
        reference_logits = torch.where(action_mask, reference_logits, invalid_fill)
    mask = (
        torch.ones(batch, dtype=torch.bool, device=new_logits.device)
        if sample_mask is None
        else sample_mask.to(device=new_logits.device, dtype=torch.bool).reshape(-1)
    )
    if mask.numel() != batch:
        raise ValueError("RLHF PPO sample_mask must have batch length B")

    new_log_probs = torch.log_softmax(new_logits, dim=-1)
    old_log_probs = torch.log_softmax(old_logits.detach(), dim=-1)
    reference_log_probs = torch.log_softmax(reference_logits.detach(), dim=-1)
    selected = actions.unsqueeze(-1)
    new_selected = new_log_probs.gather(1, selected).squeeze(1)
    old_selected = old_log_probs.gather(1, selected).squeeze(1)
    ratios = torch.exp((new_selected - old_selected).clamp(-20.0, 20.0))
    unclipped = ratios * advantages
    clipped_ratios = torch.clamp(
        ratios, 1.0 - float(cfg.clip_epsilon), 1.0 + float(cfg.clip_epsilon)
    )
    clipped = clipped_ratios * advantages
    surrogate = torch.minimum(unclipped, clipped)
    actor_loss = -_masked_mean(surrogate, mask)

    value_loss = 0.5 * _masked_mean((current_values - returns) ** 2, mask)
    probabilities = torch.softmax(new_logits, dim=-1)
    exact_kl = torch.sum(
        probabilities * (new_log_probs - reference_log_probs), dim=-1
    )
    kl_loss = _masked_mean(exact_kl, mask)
    entropy = -torch.sum(probabilities * new_log_probs, dim=-1)
    entropy_mean = _masked_mean(entropy, mask)
    total_loss = (
        actor_loss
        + float(cfg.value_loss_weight) * value_loss
        + float(cfg.kl_beta) * kl_loss
        - float(cfg.entropy_weight) * entropy_mean
    )
    with torch.no_grad():
        selected_advantages = advantages[mask]
        stats = {
            "rlhf_ppo_samples": float(mask.sum().item()),
            "rlhf_ppo_actor_loss": float(actor_loss.detach().item()),
            "rlhf_ppo_value_loss": float(value_loss.detach().item()),
            "rlhf_ppo_reference_kl": float(kl_loss.detach().item()),
            "rlhf_ppo_entropy": float(entropy_mean.detach().item()),
            "rlhf_ppo_ratio_mean": float(_masked_mean(ratios, mask).detach().item()),
            "rlhf_ppo_clipped_fraction": float(
                _masked_mean((ratios != clipped_ratios).to(dtype=new_logits.dtype), mask)
                .detach().item()
            ),
            "rlhf_ppo_advantage_mean": float(
                selected_advantages.mean().item() if selected_advantages.numel() else 0.0
            ),
            "rlhf_ppo_total_loss": float(total_loss.detach().item()),
            "evidence_class": "rlhf_policy_candidate_only",
            "promotion_allowed": False,
        }
    return total_loss, stats
