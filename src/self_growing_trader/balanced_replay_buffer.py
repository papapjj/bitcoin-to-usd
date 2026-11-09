"""Balanced Multi-Regime Replay Buffer for Reinforcement Learning (Phase 3 R1).

Maintains a balanced experience reservoir across Uptrend, Downtrend, and Sideway
market regimes, with internal win/loss stratification, to prevent catastrophic
forgetting across sequential training windows.
"""
from __future__ import annotations

import copy
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .five_shared_policy import FiveAgentMemory

_REGIME_TO_IDX: dict[str, int] = {"UPTREND": 0, "DOWNTREND": 1, "SIDEWAY": 2}
_IDX_TO_REGIME: dict[int, str] = {0: "UPTREND", 1: "DOWNTREND", 2: "SIDEWAY"}



def normalize_regime(regime: Any) -> str:
    """Normalize arbitrary regime strings or labels to canonical UPTREND, DOWNTREND, or SIDEWAY."""
    r = str(regime or "").strip().upper()
    if any(k in r for k in ("UP", "TANG", "BULL")):
        return "UPTREND"
    if any(k in r for k in ("DOWN", "GIAM", "BEAR")):
        return "DOWNTREND"
    return "SIDEWAY"


def _extract_regime_from_step(item: Mapping[str, Any]) -> str:
    """Extract and normalize regime key from a step dictionary."""
    if "regime" in item and item["regime"]:
        return normalize_regime(str(item["regime"]))
    if "lifecycle_stratum" in item:
        ls = item["lifecycle_stratum"]
        if isinstance(ls, (tuple, list)) and len(ls) >= 3:
            return normalize_regime(str(ls[2]))
    return "SIDEWAY"


def _sanitize_value_for_memory(v: Any) -> Any:
    """Recursively detach PyTorch tensors to CPU and copy numpy arrays."""
    if isinstance(v, FiveAgentMemory):
        return FiveAgentMemory(
            hidden=v.hidden.detach().cpu(),
            cell=v.cell.detach().cpu(),
            previous_action=v.previous_action.detach().cpu(),
            temporal_tokens=v.temporal_tokens.detach().cpu() if v.temporal_tokens is not None else None,
            temporal_valid=v.temporal_valid.detach().cpu() if v.temporal_valid is not None else None,
        )
    if isinstance(v, np.ndarray):
        return v.copy()
    if isinstance(v, torch.Tensor):
        return v.detach().cpu()
    if isinstance(v, tuple):
        return tuple(_sanitize_value_for_memory(item) for item in v)
    if isinstance(v, list):
        return [_sanitize_value_for_memory(item) for item in v]
    if isinstance(v, dict):
        return {k: _sanitize_value_for_memory(val) for k, val in v.items()}
    return v


def _sanitize_step_dict_for_memory(step_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Ensure numpy arrays are copied and PyTorch tensors are detached on CPU."""
    return {k: _sanitize_value_for_memory(v) for k, v in step_dict.items()}


def _serialize_value(v: Any) -> Any:
    """Convert arbitrary objects into JSON-serializable primitives."""
    if isinstance(v, FiveAgentMemory):
        prev_act = v.previous_action.detach().cpu()
        return {
            "__type__": "FiveAgentMemory",
            "hidden": v.hidden.detach().cpu().tolist(),
            "cell": v.cell.detach().cpu().tolist(),
            "previous_action": int(prev_act.item()) if prev_act.numel() == 1 else prev_act.tolist(),
            "temporal_tokens": v.temporal_tokens.detach().cpu().tolist() if v.temporal_tokens is not None else None,
            "temporal_valid": v.temporal_valid.detach().cpu().tolist() if v.temporal_valid is not None else None,
        }
    if isinstance(v, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "data": v.tolist(),
        }
    if isinstance(v, torch.Tensor):
        return {
            "__type__": "tensor",
            "dtype": str(v.dtype),
            "shape": list(v.shape),
            "data": v.detach().cpu().tolist(),
        }
    if isinstance(v, tuple):
        return {
            "__type__": "tuple",
            "items": [_serialize_value(item) for item in v],
        }
    if isinstance(v, list):
        return [_serialize_value(item) for item in v]
    if isinstance(v, dict):
        return {str(k): _serialize_value(val) for k, val in v.items()}
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else 0.0
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (bool, str)) or v is None:
        return v
    return str(v)


def _serialize_step_dict(step_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize step dictionary into pure JSON-safe primitives."""
    return {str(k): _serialize_value(v) for k, v in step_dict.items()}


def _deserialize_value(v: Any) -> Any:
    """Restore serialized primitives into numpy arrays and FiveAgentMemory."""
    if isinstance(v, dict):
        t = v.get("__type__")
        if t == "FiveAgentMemory":
            prev_act = v.get("previous_action", 0)
            if isinstance(prev_act, list):
                prev_tensor = torch.as_tensor(prev_act, dtype=torch.long, device="cpu")
            else:
                prev_tensor = torch.tensor(int(prev_act), dtype=torch.long, device="cpu")
            return FiveAgentMemory(
                hidden=torch.as_tensor(v["hidden"], dtype=torch.float32, device="cpu"),
                cell=torch.as_tensor(v["cell"], dtype=torch.float32, device="cpu"),
                previous_action=prev_tensor,
                temporal_tokens=(
                    torch.as_tensor(v["temporal_tokens"], dtype=torch.float32, device="cpu")
                    if v.get("temporal_tokens") is not None else None
                ),
                temporal_valid=(
                    torch.as_tensor(v["temporal_valid"], dtype=torch.bool, device="cpu")
                    if v.get("temporal_valid") is not None else None
                ),
            )
        if t == "ndarray":
            return np.asarray(v["data"], dtype=v["dtype"]).reshape(v["shape"])
        if t == "tensor":
            return torch.as_tensor(v["data"])
        if t == "tuple":
            return tuple(_deserialize_value(item) for item in v.get("items", []))
        return {k: _deserialize_value(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_deserialize_value(item) for item in v]
    return v


def _deserialize_step_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    """Restore serialized step dictionary into numpy arrays and FiveAgentMemory."""
    return {k: _deserialize_value(v) for k, v in data.items()}


@dataclass
class RegimeExperience:
    step_dict: dict[str, Any]
    regime: str  # "UPTREND", "DOWNTREND", "SIDEWAY"
    realized_r: float
    is_win: bool
    agent_index: int
    action_index: int
    timestamp_ns: int
    trace_id: str = ""
    td_error: float = 1.0

    def copy(self) -> "RegimeExperience":
        return RegimeExperience(
            step_dict=_sanitize_step_dict_for_memory(self.step_dict),
            regime=self.regime,
            realized_r=self.realized_r,
            is_win=self.is_win,
            agent_index=self.agent_index,
            action_index=self.action_index,
            timestamp_ns=self.timestamp_ns,
            trace_id=self.trace_id,
            td_error=self.td_error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "realized_r": self.realized_r,
            "is_win": self.is_win,
            "agent_index": self.agent_index,
            "action_index": self.action_index,
            "timestamp_ns": self.timestamp_ns,
            "trace_id": self.trace_id,
            "td_error": self.td_error,
            "step_dict": _serialize_step_dict(self.step_dict),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RegimeExperience":
        return cls(
            step_dict=_deserialize_step_dict(data.get("step_dict", {})),
            regime=str(data.get("regime", "SIDEWAY")),
            realized_r=float(data.get("realized_r", 0.0)),
            is_win=bool(data.get("is_win", False)),
            agent_index=int(data.get("agent_index", 0)),
            action_index=int(data.get("action_index", 0)),
            timestamp_ns=int(data.get("timestamp_ns", 0)),
            trace_id=str(data.get("trace_id", "")),
            td_error=float(data.get("td_error", 1.0)),
        )


@dataclass
class BalancedMultiRegimeReplayBuffer:
    capacity_per_regime: int = 128
    uptrend_pool: list[RegimeExperience] = field(default_factory=list)
    downtrend_pool: list[RegimeExperience] = field(default_factory=list)
    sideway_pool: list[RegimeExperience] = field(default_factory=list)
    schema_version: int = 2
    total_admitted: int = 0
    total_sampled: int = 0
    admitted_by_regime: dict[str, int] = field(
        default_factory=lambda: {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
    )
    sampled_by_regime: dict[str, int] = field(
        default_factory=lambda: {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
    )
    reservoir_sampling: bool = False
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_end: float = 1.0
    beta_steps: int = 1000
    step_count: int = 0
    beta: float = 0.4
    markov_counts: np.ndarray = field(
        default_factory=lambda: np.ones((3, 3), dtype=np.float64)
    )
    last_regime_idx: int = 2

    def record_regime_transition(self, current_regime: str) -> None:
        """Record Markov regime transition count with Laplace smoothing."""
        curr_idx = _REGIME_TO_IDX.get(normalize_regime(current_regime), 2)
        if (
            not hasattr(self, "markov_counts")
            or self.markov_counts is None
            or not isinstance(self.markov_counts, np.ndarray)
            or self.markov_counts.shape != (3, 3)
        ):
            self.markov_counts = np.ones((3, 3), dtype=np.float64)
        if not hasattr(self, "last_regime_idx"):
            self.last_regime_idx = 2
        self.markov_counts[self.last_regime_idx, curr_idx] += 1.0
        self.last_regime_idx = curr_idx

    def get_markov_probs(self) -> np.ndarray:
        """Returns row-normalized 3x3 Markov regime transition probability matrix."""
        if (
            not hasattr(self, "markov_counts")
            or self.markov_counts is None
            or not isinstance(self.markov_counts, np.ndarray)
            or self.markov_counts.shape != (3, 3)
        ):
            self.markov_counts = np.ones((3, 3), dtype=np.float64)
        row_sums = self.markov_counts.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0.0, 1.0, row_sums)
        return self.markov_counts / row_sums

    def ensure_compatible(self, capacity_per_regime: int = 128) -> None:
        """Ensure all fields and pools exist for legacy checkpoints."""
        if not hasattr(self, "uptrend_pool") or not isinstance(self.uptrend_pool, list):
            self.uptrend_pool = []
        if not hasattr(self, "downtrend_pool") or not isinstance(self.downtrend_pool, list):
            self.downtrend_pool = []
        if not hasattr(self, "sideway_pool") or not isinstance(self.sideway_pool, list):
            self.sideway_pool = []
        if not hasattr(self, "admitted_by_regime") or not isinstance(self.admitted_by_regime, dict):
            self.admitted_by_regime = {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
        if not hasattr(self, "sampled_by_regime") or not isinstance(self.sampled_by_regime, dict):
            self.sampled_by_regime = {"UPTREND": 0, "DOWNTREND": 0, "SIDEWAY": 0}
        if not hasattr(self, "reservoir_sampling"):
            self.reservoir_sampling = False
        if not hasattr(self, "alpha"):
            self.alpha = 0.6
        if not hasattr(self, "beta_start"):
            self.beta_start = 0.4
        if not hasattr(self, "beta_end"):
            self.beta_end = 1.0
        if not hasattr(self, "beta_steps"):
            self.beta_steps = 1000
        if not hasattr(self, "step_count"):
            self.step_count = 0
        if not hasattr(self, "beta"):
            self.beta = self.beta_start
        if not hasattr(self, "markov_counts") or self.markov_counts is None:
            self.markov_counts = np.ones((3, 3), dtype=np.float64)
        if not hasattr(self, "last_regime_idx"):
            self.last_regime_idx = 2
        self.capacity_per_regime = int(capacity_per_regime)

    def get_pool(self, regime: str) -> list[RegimeExperience]:
        r = normalize_regime(regime)
        if r == "UPTREND":
            return self.uptrend_pool
        if r == "DOWNTREND":
            return self.downtrend_pool
        return self.sideway_pool

    def add(
        self,
        step_dict: Mapping[str, Any],
        *,
        regime: str,
        realized_r: float = 0.0,
        td_error: float = 1.0,
        agent_index: int = 0,
        action_index: int = 0,
        trace_id: str = "",
        rng: Any = None,
    ) -> bool:
        if not math.isfinite(realized_r):
            return False
        if not math.isfinite(td_error):
            td_error = 1.0
        regime_key = normalize_regime(regime)
        is_win = float(realized_r) > 0.0
        safe_step_dict = _sanitize_step_dict_for_memory(step_dict)

        # Synchronize lifecycle_stratum regime component if present
        if "lifecycle_stratum" in safe_step_dict:
            ls = safe_step_dict["lifecycle_stratum"]
            if isinstance(ls, (tuple, list)) and len(ls) >= 3:
                if normalize_regime(str(ls[2])) != regime_key:
                    safe_step_dict["lifecycle_stratum"] = (ls[0], ls[1], regime_key)
        safe_step_dict["regime"] = regime_key

        exp = RegimeExperience(
            step_dict=safe_step_dict,
            regime=regime_key,
            realized_r=float(realized_r),
            is_win=is_win,
            agent_index=int(agent_index),
            action_index=int(action_index),
            timestamp_ns=time.time_ns(),
            trace_id=str(trace_id),
            td_error=float(td_error),
        )

        self.record_regime_transition(regime_key)
        pool = self.get_pool(regime_key)
        self.total_admitted += 1
        self.admitted_by_regime[regime_key] = self.admitted_by_regime.get(regime_key, 0) + 1
        k = self.admitted_by_regime[regime_key]

        if self.reservoir_sampling:
            # Genuine Algorithm R Reservoir Replacement:
            # Pool capacity C: when k <= C, append.
            # When k > C, choose random integer j in [0, k-1].
            # If j < C, replace pool[j] with exp.
            if len(pool) < self.capacity_per_regime:
                pool.append(exp)
            else:
                rand_source = rng if rng is not None else random
                j = rand_source.randint(0, k - 1)
                if j < self.capacity_per_regime:
                    pool[j] = exp
        else:
            # Legacy FIFO 50/50 Win/Loss Balance for strict backward compatibility
            max_sub = max(1, self.capacity_per_regime // 2)
            sub_items = [e for e in pool if e.is_win == is_win]
            if len(sub_items) >= max_sub:
                for idx, item in enumerate(pool):
                    if item.is_win == is_win:
                        pool.pop(idx)
                        break
            pool.append(exp)
        return True

    def update_priorities(
        self,
        regime: str,
        indices: Sequence[int],
        td_errors: Sequence[float],
    ) -> None:
        """Update TD-errors for specific indices in a regime pool."""
        pool = self.get_pool(regime)
        for idx, delta in zip(indices, td_errors):
            if 0 <= idx < len(pool) and math.isfinite(delta):
                pool[idx].td_error = float(abs(delta))

    def sample_balanced_batch(
        self,
        batch_size: int = 32,
        current_batch: list[dict[str, Any]] | None = None,
        return_is_weights: bool = False,
    ) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], torch.Tensor]:
        if batch_size <= 0:
            if return_is_weights:
                return [], torch.empty(0, dtype=torch.float32)
            return []

        regimes = ["UPTREND", "DOWNTREND", "SIDEWAY"]
        base = batch_size // 3
        rem = batch_size % 3
        quotas = {
            "UPTREND": base,
            "DOWNTREND": base,
            "SIDEWAY": base + rem,
        }
        collected: list[dict[str, Any]] = []
        is_weights_list: list[float] = []
        pool_needed = dict(quotas)
        leftover_items: dict[str, list[tuple[dict[str, Any], float]]] = {r: [] for r in regimes}

        # Step-count and Annealed Beta calculation
        self.step_count += 1
        fraction = min(1.0, self.step_count / float(max(1, self.beta_steps)))
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

        markov_P = self.get_markov_probs()
        next_regime_weights = markov_P[self.last_regime_idx]

        # 1. Hybridization: Consume current in-flight batch items first
        if current_batch:
            current_grouped: dict[str, list[dict[str, Any]]] = {r: [] for r in regimes}
            for item in current_batch:
                r = _extract_regime_from_step(item)
                current_grouped[r].append(item)
            for r in regimes:
                take_cnt = min(pool_needed[r], len(current_grouped[r]))
                if take_cnt > 0:
                    for item in current_grouped[r][:take_cnt]:
                        collected.append(copy.deepcopy(item))
                        is_weights_list.append(1.0)
                    pool_needed[r] -= take_cnt
                if len(current_grouped[r]) > take_cnt:
                    leftover_items[r].extend((item, 1.0) for item in current_grouped[r][take_cnt:])

        # 2. Sample from historical regime pools
        deficit = 0
        for r in regimes:
            needed = pool_needed[r]
            pool = self.get_pool(r)
            r_idx = _REGIME_TO_IDX.get(r, 2)
            N = len(pool)

            if N == 0:
                deficit += needed
                continue

            if not self.reservoir_sampling:
                # Legacy sampling with win/loss balance
                if N <= needed:
                    for exp in pool:
                        collected.append(copy.deepcopy(exp.step_dict))
                        is_weights_list.append(1.0)
                    deficit += (needed - N)
                else:
                    wins = [e for e in pool if e.is_win]
                    losses = [e for e in pool if not e.is_win]
                    tw = needed // 2
                    tl = needed - tw
                    if len(wins) < tw:
                        chosen_wins = list(wins)
                        chosen_losses = losses[: needed - len(chosen_wins)]
                        leftovers = losses[needed - len(chosen_wins) :]
                    elif len(losses) < tl:
                        chosen_losses = list(losses)
                        chosen_wins = wins[: needed - len(chosen_losses)]
                        leftovers = wins[needed - len(chosen_losses) :]
                    else:
                        chosen_wins = wins[:tw]
                        chosen_losses = losses[:tl]
                        leftovers = wins[tw:] + losses[tl:]
                    leftover_items[r].extend((e.step_dict, 1.0) for e in leftovers)
                    for exp in chosen_wins + chosen_losses:
                        collected.append(copy.deepcopy(exp.step_dict))
                        is_weights_list.append(1.0)
            else:
                # Dynamic Reservoir Prioritized Sampling (TD-Error & Markov Transition)
                markov_factor = 1.0 + 0.5 * float(next_regime_weights[r_idx])
                raw_prios = np.array(
                    [(abs(exp.td_error) + 1e-5) ** self.alpha for exp in pool],
                    dtype=np.float64,
                ) * markov_factor
                p_sum = raw_prios.sum()
                probs = raw_prios / p_sum if p_sum > 0.0 else np.ones(N, dtype=np.float64) / N

                sample_count = min(needed, N)
                if sample_count < needed:
                    deficit += (needed - sample_count)

                # Sample without replacement
                chosen_indices = np.random.choice(
                    N, size=sample_count, replace=False, p=probs
                )
                chosen_set = set(chosen_indices)
                unselected_indices = [idx for idx in range(N) if idx not in chosen_set]

                for idx in chosen_indices:
                    collected.append(copy.deepcopy(pool[idx].step_dict))
                    p_i = max(1e-12, float(probs[idx]))
                    w = (1.0 / (N * p_i)) ** self.beta
                    is_weights_list.append(w)

                for idx in unselected_indices:
                    p_i = max(1e-12, float(probs[idx]))
                    w = (1.0 / (N * p_i)) ** self.beta
                    leftover_items[r].append((pool[idx].step_dict, w))

        # 3. Deficit redistribution across populated pools
        if deficit > 0:
            active_leftovers = {r: list(items) for r, items in leftover_items.items() if items}
            while deficit > 0 and active_leftovers:
                for r in list(active_leftovers.keys()):
                    if deficit <= 0:
                        break
                    if active_leftovers[r]:
                        step, w = active_leftovers[r].pop(0)
                        collected.append(copy.deepcopy(step))
                        is_weights_list.append(w)
                        deficit -= 1
                    else:
                        del active_leftovers[r]

        # 4. Normalize Importance Sampling weights to max 1.0
        max_w = max(is_weights_list) if is_weights_list else 1.0
        max_w = max(1e-12, max_w)
        norm_weights = [float(w / max_w) for w in is_weights_list]

        # Embed IS weights into step dictionaries
        for idx, item in enumerate(collected):
            item["importance_sampling_weight"] = norm_weights[idx]
            item["is_weight"] = norm_weights[idx]

        # 5. Shuffle batch so regimes are balanced across update steps
        if len(collected) > 1:
            indices = np.random.permutation(len(collected))
            collected = [collected[i] for i in indices]
            norm_weights = [norm_weights[i] for i in indices]

        self.total_sampled += len(collected)
        for item in collected:
            r = _extract_regime_from_step(item)
            if r in self.sampled_by_regime:
                self.sampled_by_regime[r] += 1

        if return_is_weights:
            return collected, torch.tensor(norm_weights, dtype=torch.float32)
        return collected

    def sample_balanced_reservoir_batch(
        self,
        batch_size: int = 96,
        current_batch: list[dict[str, Any]] | None = None,
    ) -> tuple[list[dict[str, Any]], torch.Tensor]:
        """Convenience method returning both balanced batch and IS weights tensor."""
        return self.sample_balanced_batch(  # type: ignore[return-value]
            batch_size=batch_size,
            current_batch=current_batch,
            return_is_weights=True,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "capacity_per_regime": self.capacity_per_regime,
            "total_admitted": self.total_admitted,
            "total_sampled": self.total_sampled,
            "admitted_by_regime": dict(self.admitted_by_regime),
            "sampled_by_regime": dict(self.sampled_by_regime),
            "uptrend_pool": [e.to_dict() for e in self.uptrend_pool],
            "downtrend_pool": [e.to_dict() for e in self.downtrend_pool],
            "sideway_pool": [e.to_dict() for e in self.sideway_pool],
            "reservoir_sampling": self.reservoir_sampling,
            "alpha": self.alpha,
            "beta": self.beta,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "beta_steps": self.beta_steps,
            "step_count": self.step_count,
            "markov_counts": (
                self.markov_counts.tolist()
                if isinstance(self.markov_counts, np.ndarray)
                else [[1.0] * 3] * 3
            ),
            "last_regime_idx": self.last_regime_idx,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BalancedMultiRegimeReplayBuffer":
        buf = cls(
            capacity_per_regime=int(data.get("capacity_per_regime", 128)),
            schema_version=int(data.get("schema_version", 2)),
            total_admitted=int(data.get("total_admitted", 0)),
            total_sampled=int(data.get("total_sampled", 0)),
            admitted_by_regime=dict(data.get("admitted_by_regime", {})),
            sampled_by_regime=dict(data.get("sampled_by_regime", {})),
            reservoir_sampling=bool(data.get("reservoir_sampling", False)),
            alpha=float(data.get("alpha", 0.6)),
            beta_start=float(data.get("beta_start", 0.4)),
            beta_end=float(data.get("beta_end", 1.0)),
            beta_steps=int(data.get("beta_steps", 1000)),
        )
        buf.step_count = int(data.get("step_count", 0))
        buf.beta = float(data.get("beta", buf.beta_start))
        if "markov_counts" in data and data["markov_counts"] is not None:
            buf.markov_counts = np.asarray(data["markov_counts"], dtype=np.float64)
        else:
            buf.markov_counts = np.ones((3, 3), dtype=np.float64)
        buf.last_regime_idx = int(data.get("last_regime_idx", 2))
        buf.uptrend_pool = [RegimeExperience.from_dict(d) for d in data.get("uptrend_pool", [])]
        buf.downtrend_pool = [RegimeExperience.from_dict(d) for d in data.get("downtrend_pool", [])]
        buf.sideway_pool = [RegimeExperience.from_dict(d) for d in data.get("sideway_pool", [])]
        return buf

    def stats(self) -> dict[str, Any]:
        up_cnt = len(self.uptrend_pool)
        down_cnt = len(self.downtrend_pool)
        side_cnt = len(self.sideway_pool)
        tot = up_cnt + down_cnt + side_cnt
        up_wins = sum(1 for e in self.uptrend_pool if e.is_win)
        up_losses = up_cnt - up_wins
        down_wins = sum(1 for e in self.downtrend_pool if e.is_win)
        down_losses = down_cnt - down_wins
        side_wins = sum(1 for e in self.sideway_pool if e.is_win)
        side_losses = side_cnt - side_wins
        return {
            "capacity_per_regime": self.capacity_per_regime,
            "total_size": tot,
            "total_samples": tot,
            "uptrend_count": up_cnt,
            "uptrend_total": up_cnt,
            "downtrend_count": down_cnt,
            "downtrend_total": down_cnt,
            "sideway_count": side_cnt,
            "sideway_total": side_cnt,
            "uptrend_wins": up_wins,
            "uptrend_losses": up_losses,
            "downtrend_wins": down_wins,
            "downtrend_losses": down_losses,
            "sideway_wins": side_wins,
            "sideway_losses": side_losses,
            "total_admitted": self.total_admitted,
            "total_sampled": self.total_sampled,
        }


class BalancedMultiRegimeReservoirBuffer(BalancedMultiRegimeReplayBuffer):
    """Dynamic 3-Pool Reservoir Buffer with Reservoir Sampling enabled by default."""

    def __init__(
        self,
        capacity_per_regime: int = 128,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 1000,
        reservoir_sampling: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            capacity_per_regime=capacity_per_regime,
            reservoir_sampling=reservoir_sampling,
            alpha=alpha,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_steps=beta_steps,
            **kwargs,
        )

