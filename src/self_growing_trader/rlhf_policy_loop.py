"""Candidate-only rollout -> reward-model -> PPO policy loop.

The loop is intentionally explicit about what is frozen and what is learned:
the reward model scores causal rollout features, the reference logits remain
detached, and the existing FivePPOTrainer performs the only optimizer write.
It does not run MT5 or grant promotion; its output is a candidate manifest
that must still pass the frozen MT5/OOS gates.
"""
from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .rlhf_ppo import RLHFPPOConfig
from .rlhf_reward_model import RLHFRewardModel


RLHF_POLICY_ROLLOUT_SCHEMA_VERSION = "rlhf_policy_rollout_v1"
RLHF_POLICY_FEATURE_SCHEMA = "mt5_raw_obs137_action_onehot_v1"
# Codex action grading uses the full twelve-command execution contract.  The
# legacy human/RLHF entry scorer deliberately keeps the seven-action schema;
# changing it in-place would make old reward-model artifacts ambiguous.
CODEX_ACTION_FEATURE_SCHEMA = "mt5_raw_obs137_action12_onehot_v1"


def _finite_array(value: object) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
        return None
    if array.size == 0 or not bool(np.all(np.isfinite(array))):
        return None
    return array


def causal_policy_feature_vector(
    observation: Sequence[float] | np.ndarray,
    action: int,
    *,
    input_size: int = 137,
    action_count: int = 7,
) -> np.ndarray:
    """Build the shared causal feature schema for rater and policy rollout.

    Only the decision-time observation and the proposed entry action are
    included.  No reward, close row, future bar, trace ID or hindsight field is
    allowed into this vector.
    """

    values = np.asarray(observation, dtype=np.float32).reshape(-1)
    action = int(action)
    if values.size != int(input_size):
        raise ValueError("RLHF causal observation must have exactly 137 features")
    if not 0 <= action < int(action_count):
        raise ValueError("RLHF causal action is outside the configured action surface")
    if not bool(np.all(np.isfinite(values))):
        raise ValueError("RLHF causal observation contains NaN/Inf")
    one_hot = np.zeros(int(action_count), dtype=np.float32)
    one_hot[action] = 1.0
    return np.concatenate((values, one_hot), axis=0)


def freeze_reference_policy(
    network: torch.nn.Module,
) -> tuple[torch.nn.Module, str]:
    """Clone a policy checkpoint and return its immutable parameter hash."""

    reference = copy.deepcopy(network)
    reference.eval()
    digest = hashlib.sha256()
    for name, parameter in sorted(reference.state_dict().items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(str(parameter.dtype).encode("ascii"))
        digest.update(np.asarray(parameter.detach().cpu()).tobytes())
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference, digest.hexdigest()


@dataclass(frozen=True)
class RLHFPolicyRollout:
    """One causal policy rollout row that can be scored by the reward model."""

    rollout_id: str
    pair_id: str
    state_hash: str
    geometry_hash: str
    feature_schema: str
    features: tuple[float, ...]
    action: int
    old_logits: tuple[float, ...]
    reference_logits: tuple[float, ...]
    value: float
    allowed_action_mask: tuple[bool, ...]
    sample_mask: bool = True

    def __post_init__(self) -> None:
        errors: list[str] = []
        for name in (
            "rollout_id", "pair_id", "state_hash", "geometry_hash", "feature_schema"
        ):
            if not str(getattr(self, name, "") or "").strip():
                errors.append(f"missing_{name}")
        features = _finite_array(self.features)
        old = _finite_array(self.old_logits)
        reference = _finite_array(self.reference_logits)
        if features is None:
            errors.append("features_must_be_finite_nonempty")
        if old is None:
            errors.append("old_logits_must_be_finite_nonempty")
        if reference is None:
            errors.append("reference_logits_must_be_finite_nonempty")
        if old is not None and reference is not None and old.shape != reference.shape:
            errors.append("old_reference_logit_shape_mismatch")
        mask = tuple(bool(value) for value in self.allowed_action_mask)
        if old is not None and len(mask) != old.size:
            errors.append("allowed_action_mask_shape_mismatch")
        if mask and not any(mask):
            errors.append("no_legal_action")
        try:
            action = int(self.action)
        except (TypeError, ValueError):
            action = -1
        if old is not None and not 0 <= action < old.size:
            errors.append("action_out_of_policy_surface")
        if mask and 0 <= action < len(mask) and not mask[action]:
            errors.append("rollout_action_is_masked")
        try:
            value = float(self.value)
        except (TypeError, ValueError):
            value = float("nan")
        if not np.isfinite(value):
            errors.append("value_must_be_finite")
        if errors:
            raise ValueError("invalid RLHF policy rollout: " + "; ".join(errors))
        assert features is not None and old is not None and reference is not None
        object.__setattr__(self, "features", tuple(float(value) for value in features))
        object.__setattr__(self, "old_logits", tuple(float(value) for value in old))
        object.__setattr__(self, "reference_logits", tuple(float(value) for value in reference))
        object.__setattr__(self, "allowed_action_mask", mask)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "value", value)


def prepare_rlhf_rollout_batch(
    reward_model: RLHFRewardModel,
    rollouts: Sequence[RLHFPolicyRollout],
    *,
    device: torch.device | None = None,
) -> dict[str, object]:
    """Score a rollout batch and make one-step PPO returns/advantages.

    This V1 loop treats each mature feedback item as a terminal one-step
    rollout.  Multi-step trajectory returns can be added later without
    changing the reward-model or reference-policy contract.
    """

    rows = tuple(rollouts)
    if not rows:
        raise ValueError("RLHF policy rollout batch is empty")
    schemas = {row.feature_schema for row in rows}
    if schemas != {reward_model.feature_schema}:
        raise ValueError("rollout feature schema does not match reward model")
    rewards = np.asarray([reward_model.score(row.features) for row in rows], dtype=np.float32)
    values = np.asarray([row.value for row in rows], dtype=np.float32)
    returns = rewards.copy()
    direct_teacher_advantage = (
        getattr(reward_model, "advantage_mode", "") == "direct_teacher_reward"
    )
    raw_advantages = rewards.copy() if direct_teacher_advantage else returns - values
    if not direct_teacher_advantage and raw_advantages.size > 1 and float(raw_advantages.std()) > 1e-8:
        advantages = (raw_advantages - raw_advantages.mean()) / (
            raw_advantages.std() + 1e-8
        )
    else:
        advantages = raw_advantages.copy()
    old_logits = np.asarray([row.old_logits for row in rows], dtype=np.float32)
    reference_logits = np.asarray([row.reference_logits for row in rows], dtype=np.float32)
    actions = np.asarray([row.action for row in rows], dtype=np.int64)
    allowed_action_mask = np.asarray(
        [row.allowed_action_mask for row in rows], dtype=bool
    )
    sample_mask = np.asarray([row.sample_mask for row in rows], dtype=bool)
    tensor_device = device or torch.device("cpu")
    return {
        "old_logits": torch.as_tensor(old_logits, dtype=torch.float32, device=tensor_device),
        "reference_logits": torch.as_tensor(
            reference_logits, dtype=torch.float32, device=tensor_device
        ),
        "actions": torch.as_tensor(actions, dtype=torch.long, device=tensor_device),
        "advantages": torch.as_tensor(
            advantages, dtype=torch.float32, device=tensor_device
        ),
        "values": torch.as_tensor(values, dtype=torch.float32, device=tensor_device),
        "returns": torch.as_tensor(returns, dtype=torch.float32, device=tensor_device),
        "allowed_action_mask": torch.as_tensor(
            allowed_action_mask, dtype=torch.bool, device=tensor_device
        ),
        "sample_mask": torch.as_tensor(sample_mask, dtype=torch.bool, device=tensor_device),
        "raw_rewards": rewards,
        "raw_advantages": raw_advantages,
        "rollout_ids": tuple(row.rollout_id for row in rows),
        "pair_ids": tuple(row.pair_id for row in rows),
        "state_hashes": tuple(row.state_hash for row in rows),
        "geometry_hashes": tuple(row.geometry_hash for row in rows),
        "reward_model_sha256": reward_model.sha256,
        "feature_schema": reward_model.feature_schema,
    }


def optimize_rlhf_rollout_batch(
    trainer: Any,
    reward_model: RLHFRewardModel,
    rollouts: Sequence[RLHFPolicyRollout],
    *,
    new_logits: Tensor,
    new_values: Tensor | None = None,
    reference_policy_sha256: str,
    source_checkpoint_sha256: str,
    config: RLHFPPOConfig | None = None,
    auxiliary_loss: Tensor | None = None,
    auxiliary_stats: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Run one closed-loop candidate update through the existing Five trainer."""

    batch = prepare_rlhf_rollout_batch(
        reward_model, rollouts, device=new_logits.device
    )
    if new_logits.ndim != 2 or tuple(new_logits.shape) != tuple(
        batch["old_logits"].shape
    ):
        raise ValueError("new_logits shape does not match the rollout batch")
    result = trainer.optimize_rlhf_ppo_from_logits(
        new_logits=new_logits,
        old_logits=batch["old_logits"],
        reference_logits=batch["reference_logits"],
        actions=batch["actions"],
        advantages=batch["advantages"],
        values=batch["values"],
        returns=batch["returns"],
        new_values=new_values,
        allowed_action_mask=batch["allowed_action_mask"],
        sample_mask=batch["sample_mask"],
        reward_model_sha256=str(batch["reward_model_sha256"]),
        reference_policy_sha256=reference_policy_sha256,
        config=config,
        auxiliary_loss=auxiliary_loss,
        auxiliary_stats=auxiliary_stats,
    )
    raw_rewards = np.asarray(batch["raw_rewards"], dtype=np.float32)
    result.update({
        "rlhf_policy_rollout_schema": RLHF_POLICY_ROLLOUT_SCHEMA_VERSION,
        "rlhf_rollout_count": int(len(rollouts)),
        "rlhf_reward_mean": float(raw_rewards.mean()),
        "rlhf_reward_std": float(raw_rewards.std()),
        "rlhf_source_checkpoint_sha256": str(source_checkpoint_sha256),
        "rlhf_rollout_ids": list(batch["rollout_ids"]),
        "rlhf_pair_ids": list(batch["pair_ids"]),
        "rlhf_state_hashes": list(batch["state_hashes"]),
        "rlhf_geometry_hashes": list(batch["geometry_hashes"]),
        "promotion_allowed": False,
        "evidence_class": "rlhf_policy_candidate_only",
    })
    return result


def write_rlhf_candidate_manifest(
    path: str | Path,
    report: dict[str, object],
    *,
    checkpoint_sha256: str = "",
) -> None:
    """Persist auditable candidate telemetry; never writes a promotion flag."""

    payload = dict(report)
    payload["candidate_checkpoint_sha256"] = str(checkpoint_sha256 or "")
    payload["promotion_allowed"] = False
    payload["evidence_class"] = "rlhf_policy_candidate_only"
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
