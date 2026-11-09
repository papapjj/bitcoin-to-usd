"""Elastic Weight Consolidation (EWC) for Continuous Multi-Round RL Training.

Prevents catastrophic forgetting when transitioning between market regimes
(e.g., maintaining Uptrend policy knowledge during sustained Downtrend episodes).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# Core layers to protect in SharedFiveTradingPolicy:
DEFAULT_PROTECTED_PATTERNS: tuple[str, ...] = (
    "action_query",
    "action_embedding",
    "value_head",
    "heads.value.",
    "pre_lstm",
    "market_conv",
    "market_structure.conv",
    "market_fc",
    "market_structure.projection",
)


class ElasticWeightConsolidation:
    """Manages quadratic Fisher elastic penalty for policy weight preservation.

    Formula:
        L_EWC = (lambda_ewc / 2) * sum_{j in Critical} F_j * (theta_j - theta_j^*)^2
        where F_j = (1 / N) * sum (nabla_theta_j log pi(a | s))^2 + epsilon_fisher
        epsilon_fisher = 1e-4 * mean(F)
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        lambda_ewc: float = 250.0,
        protected_param_prefixes: Sequence[str] | None = None,
        sample_count_floor: int = 500,
    ) -> None:
        self.model = model
        self.lambda_ewc = float(lambda_ewc)
        self.sample_count_floor = int(sample_count_floor)
        self.protected_patterns = (
            tuple(protected_param_prefixes)
            if protected_param_prefixes is not None
            else DEFAULT_PROTECTED_PATTERNS
        )
        self.optimal_params: dict[str, Tensor] = {}
        self.fisher_diag: dict[str, Tensor] = {}

    @property
    def is_active(self) -> bool:
        """True when both optimal anchor parameters and Fisher diagonal are populated."""
        return bool(self.optimal_params and self.fisher_diag)

    def is_protected(self, param_name: str) -> bool:
        """Match parameter name against protected layer patterns."""
        # Specifically exclude recurrent LSTM core and transformer
        if param_name.startswith("lstm.") or ".lstm." in param_name:
            return False
        if "post_lstm_transformer" in param_name or "sparse_transformer" in param_name:
            return False
        # 1. Exact startswith prefix match
        if any(param_name.startswith(p) for p in self.protected_patterns):
            return True
        # 2. Substring pattern match (handles module nesting like heads.action_query.weight)
        if any(p in param_name for p in self.protected_patterns):
            return True
        return False

    def get_protected_parameters(self) -> dict[str, Tensor]:
        """Return dict of named parameters that match protected patterns."""
        return {
            name: param
            for name, param in self.model.named_parameters()
            if param.requires_grad and self.is_protected(name)
        }

    def register_optimal_checkpoint(
        self,
        eval_dataloader: Iterable[Mapping[str, Any] | Any],
        sample_count_floor: int | None = None,
    ) -> None:
        """Compute diagonal Fisher Information Matrix with damping floor and snapshot anchor weights.

        Args:
            eval_dataloader: Iterable yielding transition dicts or items containing
                             observations and action targets.
            sample_count_floor: Minimum transition threshold (default 500) for statistical stability.
        """
        floor = self.sample_count_floor if sample_count_floor is None else int(sample_count_floor)
        was_training = self.model.training
        self.model.eval()

        self.fisher_diag.clear()
        self.optimal_params.clear()

        # Snapshot optimal anchor weights
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.is_protected(name):
                self.optimal_params[name] = param.detach().clone()
                self.fisher_diag[name] = torch.zeros_like(param)

        if not self.optimal_params:
            if was_training:
                self.model.train()
            return

        items = list(eval_dataloader) if not isinstance(eval_dataloader, (list, tuple)) else eval_dataloader
        sample_count = len(items)
        if sample_count == 0:
            if was_training:
                self.model.train()
            return

        if sample_count < floor:
            warnings.warn(
                f"EWC sample_count={sample_count} < {floor} transitions; "
                f"sample size >= {floor} recommended to maintain >=84.6% top-parameter overlap.",
                UserWarning,
                stacklevel=2,
            )

        model_device = next(self.model.parameters()).device

        orig_grad_flags = {
            name: p.requires_grad for name, p in self.model.named_parameters()
        }
        for name, p in self.model.named_parameters():
            if not self.is_protected(name):
                p.requires_grad_(False)

        try:
            for item in items:
                self.model.zero_grad()

                obs: Any = None
                target_action: Any = None
                agent_ids: Any = None
                prev_actions: Any = None

                if isinstance(item, Mapping):
                    obs = item.get("obs") if item.get("obs") is not None else item.get("observation", item.get("observations"))
                    target_action = item.get("action") if item.get("action") is not None else item.get("actions", item.get("target_action"))
                    agent_ids = item.get("agent_ids")
                    prev_actions = item.get("prev_actions", item.get("previous_actions"))
                elif isinstance(item, (list, tuple)) and len(item) >= 2:
                    obs, target_action = item[0], item[1]
                elif hasattr(item, "step_dict"):
                    step_dict = getattr(item, "step_dict", {})
                    obs = step_dict.get("obs", step_dict.get("observation"))
                    target_action = getattr(item, "action_index", step_dict.get("action"))

                if obs is None or target_action is None:
                    continue

                # Tensor conversion and device placement
                if not isinstance(obs, Tensor):
                    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=model_device)
                else:
                    obs_t = obs.to(device=model_device, dtype=torch.float32)

                if not isinstance(target_action, Tensor):
                    action_t = torch.as_tensor(target_action, dtype=torch.long, device=model_device)
                else:
                    action_t = target_action.to(device=model_device, dtype=torch.long)

                # Model forward execution
                primary_logits: Tensor
                if (
                    hasattr(self.model, "observation_encoder")
                    and hasattr(self.model, "lstm")
                    and hasattr(self.model, "heads")
                    and obs_t.ndim >= 2
                    and obs_t.shape[-2] == 5
                ):
                    agent_t = torch.arange(5, dtype=torch.long, device=model_device) if agent_ids is None else torch.as_tensor(agent_ids, device=model_device)
                    prev_t = torch.zeros(5, dtype=torch.long, device=model_device) if prev_actions is None else torch.as_tensor(prev_actions, device=model_device)
                    enc_out = self.model.observation_encoder(obs_t, agent_t, prev_t, include_pattern_predictions=False)
                    encoded = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out
                    h_init = torch.zeros((5, self.model.hidden_size), device=model_device)
                    c_init = torch.zeros((5, self.model.hidden_size), device=model_device)
                    lstm_hidden, _ = self.model.lstm(encoded, (h_init, c_init))
                    if hasattr(self.model, "post_lstm_transformer"):
                        hidden = self.model.post_lstm_transformer(lstm_hidden)
                    else:
                        hidden = lstm_hidden
                    primary_logits = self.model.heads.primary_action_logits(hidden, agent_t)
                elif callable(self.model):
                    out = self.model(obs_t)
                    if isinstance(out, tuple):
                        primary_logits = out[0][:, :12] if out[0].shape[-1] >= 12 else out[0]
                    elif isinstance(out, Tensor):
                        primary_logits = out[:, :12] if out.shape[-1] >= 12 else out
                    else:
                        continue
                else:
                    continue

                # Negative log likelihood loss: -log pi(a | s)
                log_probs = F.log_softmax(primary_logits, dim=-1)
                if action_t.ndim == 0:
                    action_t = action_t.unsqueeze(0)
                if log_probs.ndim == 1:
                    log_probs = log_probs.unsqueeze(0)
                if action_t.shape[0] != log_probs.shape[0]:
                    action_t = action_t.expand(log_probs.shape[0])

                nll = F.nll_loss(log_probs, action_t)
                nll.backward()

                # Accumulate squared gradients: (grad)^2 / sample_count
                with torch.no_grad():
                    for name, param in self.model.named_parameters():
                        if param.grad is not None and name in self.fisher_diag:
                            self.fisher_diag[name].add_((param.grad ** 2) / float(sample_count))
        finally:
            for name, p in self.model.named_parameters():
                if name in orig_grad_flags:
                    p.requires_grad_(orig_grad_flags[name])

        # Apply Fisher Damping Floor: epsilon_fisher = 10^-4 * mean(F)
        # Guarantees non-zero elasticity for all critical neurons to prevent unconstrained drift.
        all_fisher_tensors = [f.view(-1) for f in self.fisher_diag.values()]
        if all_fisher_tensors:
            flat_fisher = torch.cat(all_fisher_tensors)
            mean_fisher = float(flat_fisher.mean().item())
            epsilon_fisher = 1e-4 * max(mean_fisher, 1e-8)
            for name in self.fisher_diag:
                self.fisher_diag[name].add_(epsilon_fisher)

        if was_training:
            self.model.train()

    def compute_ewc_loss(self) -> Tensor:
        """Compute quadratic elastic penalty:

        L_EWC = (lambda_ewc / 2) * sum_{j in Critical} F_j * (theta_j - theta_j^*)^2
        """
        if not self.is_active:
            device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device("cpu")
            return torch.tensor(0.0, device=device)

        device = next(self.model.parameters()).device
        total_penalty = torch.tensor(0.0, device=device)

        for name, param in self.model.named_parameters():
            if name in self.fisher_diag and name in self.optimal_params:
                F_j = self.fisher_diag[name].to(device=param.device, dtype=param.dtype)
                theta_star = self.optimal_params[name].to(device=param.device, dtype=param.dtype)
                penalty = torch.sum(F_j * (param - theta_star) ** 2)
                total_penalty = total_penalty + penalty

        return (self.lambda_ewc / 2.0) * total_penalty

    def state_dict(self) -> dict[str, Any]:
        """Serialize EWC state cleanly to dictionary."""
        return {
            "optimal_params": {k: v.detach().cpu().clone() for k, v in self.optimal_params.items()},
            "fisher_diag": {k: v.detach().cpu().clone() for k, v in self.fisher_diag.items()},
            "lambda_ewc": float(self.lambda_ewc),
            "sample_count_floor": int(self.sample_count_floor),
            "protected_patterns": list(self.protected_patterns),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        """Load EWC state with device matching."""
        self.lambda_ewc = float(state_dict.get("lambda_ewc", self.lambda_ewc))
        self.sample_count_floor = int(state_dict.get("sample_count_floor", self.sample_count_floor))
        if "protected_patterns" in state_dict:
            self.protected_patterns = tuple(state_dict["protected_patterns"])

        device = next(self.model.parameters()).device if list(self.model.parameters()) else torch.device("cpu")
        self.optimal_params = {
            k: torch.as_tensor(v, device=device).detach().clone()
            for k, v in state_dict.get("optimal_params", {}).items()
        }
        self.fisher_diag = {
            k: torch.as_tensor(v, device=device).detach().clone()
            for k, v in state_dict.get("fisher_diag", {}).items()
        }

    def save(self, filepath: str | Path) -> None:
        """Save EWC artifact to .pt file atomically."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), tmp_path)
        tmp_path.replace(path)

    def load(self, filepath: str | Path) -> None:
        """Load EWC artifact from .pt file."""
        path = Path(filepath)
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(path, map_location="cpu")
        self.load_state_dict(state)
