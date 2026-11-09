from __future__ import annotations

"""Five-replica shared policy for the trading environment.

This module follows the parts of the 2019 OpenAI Five architecture that were
published in the paper, adapted only where the trading observation/action
semantics differ from Dota 2:

* five replicas share one parameter set and keep separate recurrent states;
* structured observation processing with continuous values, categorical
  embeddings and a two-layer unordered-set processor;
* a fully-connected pre-LSTM projection followed by cross-agent max pooling of
  the first 25 percent of the vector;
* one 4096-unit LSTM layer followed by one bottlenecked cross-agent
  Transformer layer;
* factorized action heads: primary action via action embeddings/dot product,
  target-object attention with an action-conditioned mask, independent heads
  for remaining action parameters, and a shared value-function projection;
* PPO/GAE training helpers and team-spirit reward mixing.

The public paper does not publish every internal width/activation/initializer.
Those trading-specific choices are explicit constants below and must not be
presented as recovered OpenAI implementation details.
"""

import copy as copylib
from dataclasses import dataclass, field
import math
from math import isfinite, log, sqrt
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as F
except Exception as exc:  # pragma: no cover - exercised only on missing dependency
    raise RuntimeError(
        "five_shared_policy requires PyTorch; install torch to use the five-agent profile"
    ) from exc

from .protocol_contract import (
    ACCOUNT_FEATURE_COUNT,
    ACCOUNT_OFFSET,
    ACTION_COUNT,
    BASE_MARKET_FEATURE_COUNT,
    EXECUTION_FEATURE_COUNT,
    EXECUTION_OFFSET,
    EXTRA_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    MAX_POSITION_SLOTS,
    POSITION_FEATURE_COUNT,
    POSITION_OFFSET,
    TIMING_FEATURE_COUNT,
    TIMING_OFFSET,
    WYCKOFF_FEATURE_COUNT,
    WYCKOFF_OFFSET,
)
from .execution_contract import (
    ACTION_DELAY_OUTPUT_INDEX,
    DEFAULT_SL_BIN_INDEX,
    DEFAULT_TP_BIN_INDEX,
    DEFAULT_TRAILING_BIN_INDEX,
    ENTRY_OFFSET_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    LEGACY_TP_RR_BINS,
    LEGACY_SL_ATR_BINS,
    MANAGEMENT_MODE_OUTPUT_INDEX,
    OUTPUT_COUNT,
    PARTIAL_CLOSE_OUTPUT_INDEX,
    RISK_OUTPUT_INDEX,
    SL_ATR_BINS,
    TARGET_SLOT_OUTPUT_INDEX,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    pack_geometry_choices,
)
from .five_agent_authority import (
    FIVE_AGENT_AUTHORITY_VERSION,
    FIVE_AGENT_AUTHORITIES,
)


FIVE_AGENT_COUNT = 5
FIVE_LSTM_SIZE = 4096
# OpenAI Five reported a 16-step recurrent unroll. The trading runtime uses a
# longer horizon so one PPO update can carry credit across the 48-bar pivot
# context used by the teacher dataset. Keep both values visible in audit data
# instead of presenting the trading override as a published Five parameter.
FIVE_PUBLISHED_LSTM_UNROLL = 16
FIVE_LSTM_UNROLL = 64
FIVE_CROSS_AGENT_POOL_FRACTION = 0.25
FIVE_POST_LSTM_TRANSFORMER_WIDTH = 512
FIVE_POST_LSTM_TRANSFORMER_HEADS = 8
FIVE_POST_LSTM_TRANSFORMER_FFN_SIZE = 2048
FIVE_POST_LSTM_TRANSFORMER_LAYERS = 1
FIVE_TEMPORAL_TOKEN_HORIZONS = (2, 8, 32, 128)
FIVE_TEMPORAL_TOKEN_COUNT = len(FIVE_TEMPORAL_TOKEN_HORIZONS)
FIVE_POLICY_ARCHITECTURE_VERSION = 4
FIVE_OBSERVATION_ARCHITECTURE_VERSION = 4
FIVE_REASONING_ACTION_COUNT = 7
FIVE_REASONING_SCHEMA_VERSION = 1
INTERNAL_POLICY_STATE_SCHEMA_VERSION = 1

# The paper publishes the processing topology but not every hidden width.
# These widths are trading-profile choices, deliberately centralized here.
POSITION_SET_HIDDEN = 64
POSITION_EMBED_SIZE = 64
MARKET_STRUCTURE_CHANNELS = 32
MARKET_STRUCTURE_EMBED_SIZE = 256
ACCOUNT_EMBED_SIZE = 64
EXECUTION_EMBED_SIZE = 64
AGENT_EMBED_SIZE = 32
PREVIOUS_ACTION_EMBED_SIZE = 32
PRE_LSTM_SIZE = 1024
PATTERN_LEAF_SIZE = 64
PATTERN_NODE_SIZE = 128
ACTION_EMBED_SIZE = 128
TARGET_QUERY_SIZE = 64

PARAMETER_BINS: dict[str, int] = {
    # OpenAI Five publicly used a four-way delay parameter: 0..3.
    "delay": 4,
    "risk": 16,
    "entry_offset": 17,
    "sl": len(SL_ATR_BINS),
    "tp": len(TP_RR_BINS),
    "trailing": len(TRAILING_ATR_BINS),
    # Trading-specific factorized management parameters.
    "manage_mode": 3,       # 0=no-op, 1=partial close, 2=full close
    "close_fraction": 4,    # 25%, 50%, 75%, 100%
}

GEOMETRY_HEAD_DEFAULTS = {
    "sl": DEFAULT_SL_BIN_INDEX,
    "tp": DEFAULT_TP_BIN_INDEX,
    "trailing": DEFAULT_TRAILING_BIN_INDEX,
}

ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})
LONG_ENTRY_ACTIONS = (1, 3, 5)
SHORT_ENTRY_ACTIONS = (2, 4, 6)
TARGETED_MANAGEMENT_ACTIONS = frozenset({11})
LEGACY_HEURISTIC_CLOSE_ACTIONS = frozenset({8, 9})
MANAGEMENT_ACTIONS = frozenset({8, 9, 10, 11})
FACTORIZED_ACTION_MARKER = FACTORIZED_ACTION_MARKER_VALUE


def discount_from_horizon(step_seconds: float, horizon_seconds: float) -> float:
    """Public Five horizon convention H=T/(1-gamma)."""
    step = max(float(step_seconds), 1e-9)
    horizon = max(float(horizon_seconds), step + 1e-9)
    return float(np.clip(1.0 - step / horizon, 0.0, 0.999999999))


MARKET_STRUCTURE_INPUT_COUNT = (
    BASE_MARKET_FEATURE_COUNT
    + EXTRA_MARKET_FEATURE_COUNT
    + TIMING_FEATURE_COUNT
    + WYCKOFF_FEATURE_COUNT
)


@dataclass(frozen=True)
class FivePublicProfile:
    """Published references plus explicit trading-runtime overrides."""

    agent_count: int = FIVE_AGENT_COUNT
    lstm_size: int = FIVE_LSTM_SIZE
    published_lstm_unroll: int = FIVE_PUBLISHED_LSTM_UNROLL
    lstm_unroll: int = FIVE_LSTM_UNROLL
    cross_agent_pool_fraction: float = FIVE_CROSS_AGENT_POOL_FRACTION
    post_lstm_transformer_width: int = FIVE_POST_LSTM_TRANSFORMER_WIDTH
    post_lstm_transformer_heads: int = FIVE_POST_LSTM_TRANSFORMER_HEADS
    post_lstm_transformer_ffn_size: int = FIVE_POST_LSTM_TRANSFORMER_FFN_SIZE
    post_lstm_transformer_layers: int = FIVE_POST_LSTM_TRANSFORMER_LAYERS
    temporal_token_horizons: tuple[int, ...] = FIVE_TEMPORAL_TOKEN_HORIZONS
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    team_spirit_start: float = 0.3
    team_spirit_end: float = 1.0
    gae_horizon_start_seconds: float = 60.0
    gae_horizon_end_seconds: float = 840.0
    entropy_start: float = 0.06
    entropy_end: float = 0.001
    learning_rate_start: float = 5e-5
    learning_rate_end: float = 1e-6
    past_opponent_fraction: float = 0.20
    # Trading-specific behavior floor, not a value published by OpenAI Five.
    # Five shared agents amplify exploration, so 0.5% per legal entry keeps
    # evidence alive without the 2% single-agent floor causing over-trading.
    behavior_entry_probability_floor: float = 0.005


@dataclass
class FiveTrainingSchedule:
    """Smooth schedule matching the public start/end values.

    ``progress`` is normalized to [0, 1]. The original project changed values
    over a long running experiment; trading does not have the same iteration
    clock, so callers choose how progress is mapped to their run.
    """

    progress: float = 0.0
    profile: FivePublicProfile = field(default_factory=FivePublicProfile)
    # Do not spend exploration merely because wall-clock training progressed.
    # The coefficient stays at the public-Five start value until the
    # lifecycle ledger contains enough statistically eligible positive
    # context/action scopes.  It then decays with additional verified edges.
    entropy_edge_gate_enabled: bool = False
    verified_edge_count: int = 0
    entropy_edge_release_count: int = 3
    entropy_edge_decay_count: int = 8
    flat_entry_mass_floor_start: float = 0.75
    flat_entry_mass_floor_end: float = 0.0
    flat_entry_mass_loss_weight: float = 1.0
    # Optional anti-saturation barrier. Keep zero by default for compatibility
    # with the existing probability-floor objective.
    flat_entry_mass_log_loss_weight: float = 0.0
    # Keep both entry owners represented on flat-account anchors. The total
    # six-entry floor alone can be satisfied by one direction only.
    flat_entry_direction_mass_floor_start: float = 0.0
    flat_entry_direction_mass_floor_end: float = 0.0
    flat_entry_direction_mass_loss_weight: float = 0.0
    # Decision-time exploration must react to *learned* economic evidence.
    # Positive-edge counts still release the original exploration curriculum,
    # while broad statistically eligible negative scope evidence may reduce
    # forced entry pressure before the account has to discover a positive edge.
    # A negative global aggregate without eligible scopes is deliberately
    # ignored to avoid a cold-start no-trade feedback loop. This is audit/state
    # only; no future label enters the observation or action graph.
    policy_expectancy_eligible: bool = False
    policy_expectancy_lcb: float = 0.0
    negative_critical_scope_count: int = 0
    # Sparse MT5 bridge decisions can be minutes apart.  Keep GAE expressed in
    # at least this many decision transitions so a 60..840 second public clock
    # cannot collapse economic PPO into one-step TD.
    gae_min_feedback_steps: int = 16
    # Small self-supervised pressure for the continuous in-network policy
    # state.  It prevents every causal context from mapping to one identical
    # vector without introducing strategy labels or an external router.
    internal_policy_state_diversity_weight: float = 1e-3
    internal_policy_state_variance_floor: float = 0.05
    # Dynamic Team-Spirit Cosine Decay schedule (Milestone 1 / Tier 1):
    # tau(t) = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * p))
    # where tau_max = 0.70, tau_min = 0.30, and p = clamp(t / T, 0.0, 1.0).
    team_spirit_cosine_decay_enabled: bool = False
    dynamic_team_spirit_cosine_enabled: bool = False
    team_spirit_max: float = 0.70
    team_spirit_min: float = 0.30

    def _lerp(self, start: float, end: float) -> float:
        p = float(np.clip(self.progress, 0.0, 1.0))
        return float(start + (end - start) * p)

    def get_team_spirit(
        self,
        step: int | float | None = None,
        total_steps: int | float | None = None,
        progress: float | None = None,
    ) -> float:
        """Dynamic Team-Spirit Cosine Decay schedule:
        tau(t) = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * p))
        where tau_max = 0.70, tau_min = 0.30, and p = clamp(t / T, 0.0, 1.0).
        """
        if progress is not None:
            p = float(progress)
        elif step is not None and total_steps is not None:
            total = float(total_steps)
            p = float(step) / total if (total > 0.0 and np.isfinite(total)) else 0.0
        elif step is not None and total_steps is None:
            p = float(step)
        else:
            p = float(getattr(self, "progress", 0.0))

        if not np.isfinite(p):
            p = 0.0
        p = float(np.clip(p, 0.0, 1.0))

        t_max = float(self.team_spirit_max)
        t_min = float(self.team_spirit_min)
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        decay = t_min + 0.5 * (t_max - t_min) * (1.0 + math.cos(math.pi * p))
        return float(np.clip(decay, t_min, t_max))

    def compute_team_spirit_cosine(
        self,
        progress: float | None = None,
        *,
        step: int | float | None = None,
        total_steps: int | float | None = None,
        tau_max: float | None = None,
        tau_min: float | None = None,
    ) -> float:
        """Compute Dynamic Team-Spirit Cosine Decay smoothly."""
        if progress is not None:
            p = float(progress)
        elif step is not None and total_steps is not None:
            total = float(total_steps)
            p = float(step) / total if (total > 0.0 and np.isfinite(total)) else 0.0
        else:
            p = float(getattr(self, "progress", 0.0))

        if not np.isfinite(p):
            p = 0.0
        p = float(np.clip(p, 0.0, 1.0))

        t_max = float(self.team_spirit_max if tau_max is None else tau_max)
        t_min = float(self.team_spirit_min if tau_min is None else tau_min)
        if t_min > t_max:
            t_min, t_max = t_max, t_min

        decay = t_min + 0.5 * (t_max - t_min) * (1.0 + math.cos(math.pi * p))
        return float(np.clip(decay, t_min, t_max))

    @property
    def team_spirit_decayed(self) -> float:
        """Query the cosine-decayed team spirit value regardless of gate enable flag."""
        return self.compute_team_spirit_cosine(self.progress)

    @property
    def team_spirit_cosine(self) -> float:
        """Query the cosine-decayed team spirit value regardless of gate enable flag."""
        return self.compute_team_spirit_cosine(self.progress)

    @property
    def team_spirit(self) -> float:
        if self.team_spirit_cosine_decay_enabled or self.dynamic_team_spirit_cosine_enabled:
            return self.compute_team_spirit_cosine(self.progress)
        return self._lerp(self.profile.team_spirit_start, self.profile.team_spirit_end)

    def set_progress(self, progress: float) -> None:
        p = float(progress) if np.isfinite(progress) else 0.0
        self.progress = float(np.clip(p, 0.0, 1.0))

    def set_step(self, step: int | float, total_steps: int | float) -> None:
        total = float(total_steps)
        if total <= 0.0 or not np.isfinite(total):
            self.progress = 0.0
        else:
            p = float(step) / total
            self.progress = float(np.clip(p if np.isfinite(p) else 0.0, 0.0, 1.0))

    @property
    def entropy_coefficient(self) -> float:
        base: float
        if self.entropy_edge_gate_enabled:
            release = max(1, int(self.entropy_edge_release_count))
            decay_end = max(release + 1, int(self.entropy_edge_decay_count))
            edges = max(0, int(self.verified_edge_count))
            if edges < release:
                base = float(self.profile.entropy_start)
            else:
                edge_progress = float(np.clip(
                    (edges - release) / float(decay_end - release), 0.0, 1.0
                ))
                base = float(
                    self.profile.entropy_start
                    + (self.profile.entropy_end - self.profile.entropy_start) * edge_progress
                )
        else:
            base = self._lerp(self.profile.entropy_start, self.profile.entropy_end)
        # Exploration is independent from reward/expectancy evidence.  A
        # losing global ledger must not lower entropy everywhere because doing
        # so also suppresses unseen neighbours of the few positive regions.
        # Only verified positive edges unlock the edge-count decay above.
        return base

    @property
    def negative_evidence_strength(self) -> float:
        """Bounded confidence that current policy evidence is economically bad.

        The LCB supplies direction/magnitude and the number of independently
        negative eligible scopes supplies breadth.  Either field alone is not
        enough to shut exploration off; the returned value only *attenuates*
        exploration and is always in [0,1].
        """
        if not bool(self.policy_expectancy_eligible):
            return 0.0
        lcb = float(self.policy_expectancy_lcb)
        if not np.isfinite(lcb) or lcb >= 0.0:
            return 0.0
        lcb_strength = float(np.clip((-lcb) / 0.20, 0.0, 1.0))
        breadth = float(np.clip(
            float(max(0, int(self.negative_critical_scope_count))) / 6.0,
            0.0,
            1.0,
        ))
        # A negative global mean is not enough to suppress discovery: it may
        # simply be a mixture of sparse, incompatible contexts.  Require at
        # least one eligible negative scope before attenuating the flat-entry
        # curriculum. Otherwise a losing cold-start policy enters a feedback
        # loop (negative aggregate -> fewer entries -> no new evidence).
        return float(np.clip(lcb_strength * breadth, 0.0, 1.0))

    @property
    def flat_entry_mass_floor(self) -> float:
        """Learned six-entry mass required on every flat-account state."""
        start = float(np.clip(self.flat_entry_mass_floor_start, 0.0, 1.0))
        end = float(np.clip(self.flat_entry_mass_floor_end, 0.0, start))
        if not self.entropy_edge_gate_enabled:
            base = float(start + (end - start) * np.clip(self.progress, 0.0, 1.0))
        else:
            release = max(1, int(self.entropy_edge_release_count))
            decay_end = max(release + 1, int(self.entropy_edge_decay_count))
            edges = max(0, int(self.verified_edge_count))
            if edges < release:
                base = start
            else:
                edge_progress = float(np.clip(
                    (edges - release) / float(decay_end - release), 0.0, 1.0
                ))
                base = float(start + (end - start) * edge_progress)

        # Statistically eligible negative lifecycle evidence may reduce forced
        # entry pressure. Entropy remains available to explore unseen states;
        # this only stops a losing policy from being trained toward entries.
        return float(base * (1.0 - self.negative_evidence_strength))

    @property
    def flat_entry_direction_mass_floor(self) -> float:
        """Minimum probability mass for each LONG/SHORT entry owner."""
        start = float(np.clip(self.flat_entry_direction_mass_floor_start, 0.0, 0.5))
        end = float(np.clip(self.flat_entry_direction_mass_floor_end, 0.0, start))
        if not self.entropy_edge_gate_enabled:
            base = float(start + (end - start) * np.clip(self.progress, 0.0, 1.0))
        else:
            release = max(1, int(self.entropy_edge_release_count))
            decay_end = max(release + 1, int(self.entropy_edge_decay_count))
            edges = max(0, int(self.verified_edge_count))
            if edges < release:
                base = start
            else:
                edge_progress = float(np.clip(
                    (edges - release) / float(decay_end - release), 0.0, 1.0
                ))
                base = float(start + (end - start) * edge_progress)
        return float(base * (1.0 - self.negative_evidence_strength))

    @property
    def learning_rate(self) -> float:
        return self._lerp(self.profile.learning_rate_start, self.profile.learning_rate_end)

    @property
    def gae_horizon_seconds(self) -> float:
        return self._lerp(
            self.profile.gae_horizon_start_seconds,
            self.profile.gae_horizon_end_seconds,
        )

    def effective_gae_horizon_seconds(self, step_seconds: float) -> float:
        return float(max(
            self.gae_horizon_seconds,
            max(float(step_seconds), 1e-9) * max(1, int(self.gae_min_feedback_steps)),
        ))


@dataclass
class FiveAgentMemory:
    hidden: Tensor
    cell: Tensor
    previous_action: Tensor
    temporal_tokens: Tensor | None = None
    temporal_valid: Tensor | None = None

    @classmethod
    def zeros(cls, *, device: torch.device, hidden_size: int = FIVE_LSTM_SIZE) -> "FiveAgentMemory":
        return cls(
            hidden=torch.zeros(hidden_size, dtype=torch.float32, device=device),
            cell=torch.zeros(hidden_size, dtype=torch.float32, device=device),
            previous_action=torch.zeros((), dtype=torch.long, device=device),
            temporal_tokens=None,
            temporal_valid=None,
        )

    def detached(self) -> "FiveAgentMemory":
        return FiveAgentMemory(
            hidden=self.hidden.detach(),
            cell=self.cell.detach(),
            previous_action=self.previous_action.detach(),
            temporal_tokens=(
                None if self.temporal_tokens is None else self.temporal_tokens.detach()
            ),
            temporal_valid=(
                None if self.temporal_valid is None else self.temporal_valid.detach()
            ),
        )


class FiveReasoningHead(nn.Module):
    """Small multi-step hypothesis policy trained from delayed verifiers.

    A reasoning token is one of ``HOLD + six entry actions``.  The head starts
    from the causal Five LSTM hidden state, proposes a hypothesis, feeds that
    hypothesis back through a recurrent cell, and may revise it on later
    steps.  Only the final token is scored by the immutable counterfactual
    path verifier.  Future labels are therefore training-only and never enter
    this forward graph at decision time.
    """

    def __init__(self, hidden_size: int, latent_size: int = 64) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_size = int(latent_size)
        self.context = nn.Linear(self.hidden_size, self.latent_size)
        self.hypothesis_embedding = nn.Embedding(
            FIVE_REASONING_ACTION_COUNT, self.latent_size
        )
        self.revision_cell = nn.GRUCell(self.latent_size, self.latent_size)
        self.hypothesis_logits = nn.Linear(
            self.latent_size, FIVE_REASONING_ACTION_COUNT
        )

    def initial_state(self, hidden: Tensor) -> Tensor:
        return torch.tanh(self.context(hidden))

    @staticmethod
    def _masked_logits(logits: Tensor, allowed: Tensor) -> Tensor:
        legal = allowed.to(device=logits.device, dtype=torch.bool)
        if legal.shape != logits.shape:
            raise ValueError("reasoning allowed mask has wrong shape")
        hold = torch.zeros_like(legal)
        hold[..., 0] = True
        legal = torch.where(legal.any(dim=-1, keepdim=True), legal, hold)
        return torch.where(legal, logits, torch.full_like(logits, -30.0))

    def sample(
        self,
        hidden: Tensor,
        allowed: Tensor,
        *,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        state = self.initial_state(hidden)
        choices: list[Tensor] = []
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        for _ in range(max(1, int(steps))):
            logits = self._masked_logits(self.hypothesis_logits(state), allowed)
            distribution = torch.distributions.Categorical(logits=logits)
            choice = torch.multinomial(
                distribution.probs,
                num_samples=1,
                replacement=True,
                generator=generator,
            ).squeeze(-1)
            choices.append(choice)
            log_probabilities.append(distribution.log_prob(choice))
            entropies.append(distribution.entropy())
            state = self.revision_cell(self.hypothesis_embedding(choice), state)
        return (
            torch.stack(choices, dim=-1),
            torch.stack(log_probabilities, dim=-1).sum(dim=-1),
            torch.stack(entropies, dim=-1).mean(dim=-1),
        )

    def evaluate(
        self,
        hidden: Tensor,
        allowed: Tensor,
        hypotheses: Tensor,
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        if hypotheses.ndim != 2 or hypotheses.shape[0] != hidden.shape[0]:
            raise ValueError("reasoning hypotheses must have shape [batch, steps]")
        state = self.initial_state(hidden)
        log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []
        logits_by_step: list[Tensor] = []
        for step in range(int(hypotheses.shape[1])):
            logits = self._masked_logits(self.hypothesis_logits(state), allowed)
            choice = hypotheses[:, step].to(device=hidden.device, dtype=torch.long)
            distribution = torch.distributions.Categorical(logits=logits)
            log_probabilities.append(distribution.log_prob(choice))
            entropies.append(distribution.entropy())
            logits_by_step.append(logits)
            state = self.revision_cell(self.hypothesis_embedding(choice), state)
        return (
            torch.stack(log_probabilities, dim=-1).sum(dim=-1),
            torch.stack(entropies, dim=-1).mean(dim=-1),
            logits_by_step,
        )

    def greedy(
        self,
        hidden: Tensor,
        allowed: Tensor,
        *,
        steps: int,
    ) -> tuple[Tensor, Tensor]:
        state = self.initial_state(hidden)
        choices: list[Tensor] = []
        final_logits = self.hypothesis_logits(state)
        for _ in range(max(1, int(steps))):
            final_logits = self._masked_logits(
                self.hypothesis_logits(state), allowed
            )
            choice = torch.argmax(final_logits, dim=-1)
            choices.append(choice)
            state = self.revision_cell(self.hypothesis_embedding(choice), state)
        return torch.stack(choices, dim=-1), final_logits


@dataclass(frozen=True)
class FiveActionResult:
    action_index: int
    target_index: int
    delay_index: int
    risk_index: int
    entry_offset_index: int
    sl_index: int
    tp_index: int
    trailing_index: int
    manage_mode_index: int
    close_fraction_index: int
    output_vector: np.ndarray
    log_probability: float
    entropy: float
    value: float
    action_probabilities: np.ndarray
    policy_action_probabilities: np.ndarray
    behavior_action_probabilities: np.ndarray
    policy_action_entropy: float
    behavior_action_entropy: float
    encoded_observation: np.ndarray
    hidden_before: np.ndarray
    hidden_after: np.ndarray
    cell_before: np.ndarray
    cell_after: np.ndarray
    head_log_probabilities: dict[str, float]
    head_entropies: dict[str, float]
    pattern_predictions: dict[str, float] = field(default_factory=dict)
    reasoning_trace: dict[str, object] = field(default_factory=dict)
    temporal_tokens_before: np.ndarray | None = None
    temporal_valid_before: np.ndarray | None = None
    # Retain the actual categorical policy logits separately from the
    # execution wire output so a later reference-policy comparison does not
    # reconstruct logits from probabilities.
    policy_action_logits: np.ndarray = field(
        default_factory=lambda: np.zeros(ACTION_COUNT, dtype=np.float32)
    )
    # Continuous in-network policy state.  This is not a strategy label or an
    # externally selected expert; it is the causal representation whose
    # bounded residual contributes to ``policy_action_logits``.
    internal_policy_state: np.ndarray = field(
        default_factory=lambda: np.zeros(0, dtype=np.float32)
    )
    codex_context_support: float = 0.0
    codex_context_similarity: float = 0.0
    codex_global_action_bias: float = 0.0
    codex_global_action_bias_scale: float = 0.0
    codex_global_action_bias_enabled: bool = False
    adaptive_applied: bool = False

    @property
    def action(self) -> Any:
        try:
            from .orders import Action
            return Action(self.action_index)
        except Exception:
            return self.action_index


class ProcessSet(nn.Module):
    """Two shared fully-connected layers plus max pooling for unordered sets."""

    def __init__(self, input_size: int, hidden_size: int, embedding_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)

    def forward(self, values: Tensor, valid_mask: Tensor) -> tuple[Tensor, Tensor]:
        # values: [batch, items, features]
        emb = torch.tanh(self.fc1(values))
        emb = torch.tanh(self.fc2(emb))
        mask = valid_mask.unsqueeze(-1)
        masked = torch.where(mask, emb, torch.full_like(emb, -torch.inf))
        pooled = masked.max(dim=1).values
        any_valid = valid_mask.any(dim=1, keepdim=True)
        pooled = torch.where(any_valid, pooled, torch.zeros_like(pooled))
        return pooled, emb


class DenseFeatureEncoder(nn.Module):
    """Two learned projections for a fixed-size semantic feature block."""

    def __init__(self, input_size: int, hidden_size: int, embedding_size: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.fc1 = nn.Linear(self.input_size, int(hidden_size))
        self.fc2 = nn.Linear(int(hidden_size), int(embedding_size))

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(f"expected dense feature block [N,{self.input_size}]")
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        return torch.tanh(self.fc2(torch.tanh(self.fc1(values))))


class ObservationPatternTree(nn.Module):
    """Hierarchical, action-independent representation of market opportunity.

    The leaves retain the semantic boundaries of the 137-value contract. Price
    structure and open positions meet in one node; account and execution state
    meet in another. The root is shared by the policy input and by supervised
    delayed-label heads, so dataset or mature MT5 path labels teach a reusable
    present-tense representation instead of entering the observation itself.
    """

    def __init__(self) -> None:
        super().__init__()
        self.market_leaf = nn.Linear(MARKET_STRUCTURE_EMBED_SIZE, PATTERN_LEAF_SIZE)
        self.position_leaf = nn.Linear(POSITION_EMBED_SIZE, PATTERN_LEAF_SIZE)
        self.account_leaf = nn.Linear(ACCOUNT_EMBED_SIZE, PATTERN_LEAF_SIZE)
        self.execution_leaf = nn.Linear(EXECUTION_EMBED_SIZE, PATTERN_LEAF_SIZE)
        self.price_node = nn.Linear(2 * PATTERN_LEAF_SIZE, PATTERN_NODE_SIZE)
        self.context_node = nn.Linear(2 * PATTERN_LEAF_SIZE, PATTERN_NODE_SIZE)
        self.root = nn.Linear(2 * PATTERN_NODE_SIZE, PATTERN_NODE_SIZE)
        self.utility_head = nn.Linear(PATTERN_NODE_SIZE, 2)
        self.excursion_head = nn.Linear(PATTERN_NODE_SIZE, 4)
        self.no_edge_head = nn.Linear(PATTERN_NODE_SIZE, 1)
        self._attach_rich_path_heads()

    def _attach_rich_path_heads(self) -> bool:
        """Gan cac head v4 trung tinh ma khong doi action/value cua checkpoint cu."""
        added = False
        reference = next(self.parameters())
        for name, size in (
            ("horizon_head", 5),
            ("path_dynamics_head", 10),
            ("first_hit_head", 6),
            ("action_path_head", 30),
            ("action_outcome_head", 24),
        ):
            if isinstance(getattr(self, name, None), nn.Module):
                continue
            head = nn.Linear(
                PATTERN_NODE_SIZE, size,
                device=reference.device, dtype=reference.dtype,
            )
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
            setattr(self, name, head)
            added = True
        if not isinstance(getattr(self, "action_path_residual", None), nn.Module):
            residual = nn.Sequential(
                nn.Linear(PATTERN_NODE_SIZE, 64),
                nn.Tanh(),
                nn.Linear(64, 30),
            ).to(device=reference.device, dtype=reference.dtype)
            # A legacy checkpoint keeps its exact action surface until this
            # new output branch is explicitly trained.
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
            setattr(self, "action_path_residual", residual)
            added = True
        if not isinstance(getattr(self, "action_path_observation_adapter", None), nn.Module):
            adapter = nn.Sequential(
                nn.Linear(INPUT_COUNT, 128),
                nn.Tanh(),
                nn.Linear(128, 128),
                nn.Tanh(),
                nn.Linear(128, 30),
            ).to(device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
            setattr(self, "action_path_observation_adapter", adapter)
            added = True
        return added

    def forward(
        self,
        *,
        market: Tensor,
        positions: Tensor,
        account: Tensor,
        execution: Tensor,
        raw_observations: Tensor | None = None,
    ) -> dict[str, Tensor]:
        market_leaf = torch.tanh(self.market_leaf(market))
        position_leaf = torch.tanh(self.position_leaf(positions))
        account_leaf = torch.tanh(self.account_leaf(account))
        execution_leaf = torch.tanh(self.execution_leaf(execution))
        price_node = torch.tanh(self.price_node(torch.cat((market_leaf, position_leaf), dim=-1)))
        context_node = torch.tanh(self.context_node(torch.cat((account_leaf, execution_leaf), dim=-1)))
        root = torch.tanh(self.root(torch.cat((price_node, context_node), dim=-1)))
        action_path_raw = self.action_path_head(root)
        residual = getattr(self, "action_path_residual", None)
        if isinstance(residual, nn.Module):
            action_path_raw = action_path_raw + residual(root)
        observation_adapter = getattr(self, "action_path_observation_adapter", None)
        if isinstance(observation_adapter, nn.Module) and raw_observations is not None:
            if raw_observations.ndim != 2 or raw_observations.shape[1] != INPUT_COUNT:
                raise ValueError(f"expected raw observations [N,{INPUT_COUNT}]")
            bounded = torch.tanh(torch.nan_to_num(
                raw_observations, nan=0.0, posinf=0.0, neginf=0.0
            ))
            action_path_raw = action_path_raw + observation_adapter(bounded)
        action_path_raw = action_path_raw.reshape(-1, 6, 5)
        action_path = torch.cat((
            torch.sigmoid(action_path_raw[..., :2]),
            torch.tanh(action_path_raw[..., 2:3]),
            torch.sigmoid(action_path_raw[..., 3:]),
        ), dim=-1)
        return {
            "pattern_market_leaf": market_leaf,
            "pattern_position_leaf": position_leaf,
            "pattern_account_leaf": account_leaf,
            "pattern_execution_leaf": execution_leaf,
            "pattern_price_node": price_node,
            "pattern_context_node": context_node,
            "pattern_root": root,
            # Utilities are normalized to [-1, 1]. Excursions are normalized
            # magnitudes in [0, 1]. no_edge remains a logit for stable BCE.
            "pattern_utility": torch.tanh(self.utility_head(root)),
            "pattern_excursion": torch.sigmoid(self.excursion_head(root)),
            "pattern_no_edge_logit": self.no_edge_head(root).squeeze(-1),
            "pattern_horizon_returns": torch.tanh(self.horizon_head(root)),
            "pattern_path_dynamics": torch.sigmoid(self.path_dynamics_head(root)),
            "pattern_first_hit_logits": self.first_hit_head(root).reshape(-1, 2, 3),
            "pattern_action_path": action_path,
            "pattern_action_outcome_logits": self.action_outcome_head(root).reshape(-1, 6, 4),
        }


class MarketStructureEncoder(nn.Module):
    """Learn local price/indicator structure before recurrent reasoning.

    The 137-column bridge is not an image.  Its market-related columns do,
    however, have a stable ordered contract: return horizons, candle geometry,
    indicators, closed higher-timeframe summaries, timing and Wyckoff signals.
    Two one-dimensional convolutions learn local interactions in that ordered
    structure.  Global average and maximum pooling preserve both broad regime
    information and sparse price/volume events without inventing a 2-D map.
    """

    def __init__(
        self,
        input_size: int = MARKET_STRUCTURE_INPUT_COUNT,
        channels: int = MARKET_STRUCTURE_CHANNELS,
        embedding_size: int = MARKET_STRUCTURE_EMBED_SIZE,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        channels = int(channels)
        self.conv1 = nn.Conv1d(1, channels // 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels // 2, channels, kernel_size=3, padding=1)
        self.projection = nn.Linear(2 * channels, int(embedding_size))

    def forward(self, values: Tensor) -> Tensor:
        if values.ndim != 2 or values.shape[1] != self.input_size:
            raise ValueError(f"expected market structure [N,{self.input_size}]")
        values = torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        hidden = torch.tanh(self.conv1(values.unsqueeze(1)))
        hidden = torch.tanh(self.conv2(hidden))
        pooled = torch.cat((hidden.mean(dim=-1), hidden.amax(dim=-1)), dim=-1)
        return torch.tanh(self.projection(pooled))


class FiveObservationEncoder(nn.Module):
    """Trading adaptation of the public observation-processing topology."""

    def __init__(self, pre_lstm_size: int = PRE_LSTM_SIZE) -> None:
        super().__init__()
        self.architecture_version = FIVE_OBSERVATION_ARCHITECTURE_VERSION
        self.position_set = ProcessSet(
            POSITION_FEATURE_COUNT,
            POSITION_SET_HIDDEN,
            POSITION_EMBED_SIZE,
        )
        self.market_structure = MarketStructureEncoder()
        self.account_encoder = DenseFeatureEncoder(
            ACCOUNT_FEATURE_COUNT, ACCOUNT_EMBED_SIZE, ACCOUNT_EMBED_SIZE,
        )
        self.execution_encoder = DenseFeatureEncoder(
            EXECUTION_FEATURE_COUNT, EXECUTION_EMBED_SIZE, EXECUTION_EMBED_SIZE,
        )
        self.pattern_tree = ObservationPatternTree()
        self.agent_embedding = nn.Embedding(FIVE_AGENT_COUNT, AGENT_EMBED_SIZE)
        self.previous_action_embedding = nn.Embedding(ACTION_COUNT, PREVIOUS_ACTION_EMBED_SIZE)
        flattened = (
            MARKET_STRUCTURE_EMBED_SIZE
            + ACCOUNT_EMBED_SIZE
            + EXECUTION_EMBED_SIZE
            + POSITION_EMBED_SIZE
            + AGENT_EMBED_SIZE
            + PREVIOUS_ACTION_EMBED_SIZE
        )
        self.pre_lstm = nn.Linear(flattened, pre_lstm_size)
        # Zero initialization is deliberate. A v2 checkpoint can gain the new
        # tree without changing one existing action/value output at migration
        # time; PPO later learns how much of the supervised root to consume.
        self.pattern_residual = nn.Linear(PATTERN_NODE_SIZE, pre_lstm_size, bias=False)
        nn.init.zeros_(self.pattern_residual.weight)
        self.pre_lstm_size = int(pre_lstm_size)

    def require_current_architecture(self) -> bool:
        """Validate v4 or migrate v2/v3 with output-neutral adapters.

        Returns true only when new modules were attached to a legacy pickle.
        Older, structurally ambiguous checkpoints are still rejected.
        """
        expected_input = (
            MARKET_STRUCTURE_EMBED_SIZE
            + ACCOUNT_EMBED_SIZE
            + EXECUTION_EMBED_SIZE
            + POSITION_EMBED_SIZE
            + AGENT_EMBED_SIZE
            + PREVIOUS_ACTION_EMBED_SIZE
        )
        observed_version = int(getattr(self, "architecture_version", 1) or 1)
        missing_base = [
            name
            for name in ("market_structure", "account_encoder", "execution_encoder")
            if not isinstance(getattr(self, name, None), nn.Module)
        ]
        observed_input = int(getattr(getattr(self, "pre_lstm", None), "in_features", -1))
        pattern_missing = [
            name for name in ("pattern_tree", "pattern_residual")
            if not isinstance(getattr(self, name, None), nn.Module)
        ]
        if (
            observed_version in {
                FIVE_OBSERVATION_ARCHITECTURE_VERSION - 2,
                FIVE_OBSERVATION_ARCHITECTURE_VERSION - 1,
            }
            and not missing_base
            and observed_input == expected_input
            and pattern_missing
        ):
            reference = next(self.parameters())
            self.pattern_tree = ObservationPatternTree().to(
                device=reference.device, dtype=reference.dtype
            )
            self.pattern_residual = nn.Linear(
                PATTERN_NODE_SIZE, self.pre_lstm_size, bias=False,
                device=reference.device, dtype=reference.dtype,
            )
            nn.init.zeros_(self.pattern_residual.weight)
            self.architecture_version = FIVE_OBSERVATION_ARCHITECTURE_VERSION
            return True
        if (
            observed_version == FIVE_OBSERVATION_ARCHITECTURE_VERSION - 1
            and not missing_base
            and observed_input == expected_input
            and not pattern_missing
        ):
            tree = getattr(self, "pattern_tree")
            if not isinstance(tree, ObservationPatternTree):
                raise ValueError("Five checkpoint v3 co pattern tree khong tuong thich")
            tree._attach_rich_path_heads()
            self.architecture_version = FIVE_OBSERVATION_ARCHITECTURE_VERSION
            return True
        if (
            observed_version == FIVE_OBSERVATION_ARCHITECTURE_VERSION
            and not missing_base
            and observed_input == expected_input
        ):
            tree = getattr(self, "pattern_tree")
            if not isinstance(tree, ObservationPatternTree):
                raise ValueError("Five checkpoint v4 co pattern tree khong tuong thich")
            return bool(tree._attach_rich_path_heads())
        missing = missing_base + pattern_missing
        if observed_version != FIVE_OBSERVATION_ARCHITECTURE_VERSION or missing or observed_input != expected_input:
            details = ", ".join(missing) if missing else "none"
            raise ValueError(
                "Five checkpoint observation architecture is incompatible "
                f"(expected v{FIVE_OBSERVATION_ARCHITECTURE_VERSION} with pre_lstm input "
                f"{expected_input}; observed v{observed_version}, input {observed_input}, "
                f"missing {details}). Use --fresh-system or a distinct --output; "
                "the old checkpoint must not be resized in place."
            )
        return False

    @staticmethod
    def _positions(observations: Tensor) -> tuple[Tensor, Tensor]:
        start = POSITION_OFFSET
        stop = POSITION_OFFSET + MAX_POSITION_SLOTS * POSITION_FEATURE_COUNT
        raw = observations[:, start:stop].reshape(
            observations.shape[0], MAX_POSITION_SLOTS, POSITION_FEATURE_COUNT
        )
        # Zero-filled slots are canonical empty slots in the bridge contract.
        valid = torch.any(torch.abs(raw) > 1e-12, dim=-1)
        return raw, valid

    @staticmethod
    def _market_structure(observations: Tensor) -> Tensor:
        return torch.cat(
            (
                observations[:, :POSITION_OFFSET],
                observations[:, EXTRA_MARKET_OFFSET:EXECUTION_OFFSET],
                observations[:, TIMING_OFFSET:WYCKOFF_OFFSET],
                observations[:, WYCKOFF_OFFSET:INPUT_COUNT],
            ),
            dim=-1,
        )

    def encode_components(
        self,
        observations: Tensor,
        agent_ids: Tensor,
        previous_actions: Tensor,
    ) -> dict[str, Tensor]:
        """Encode each semantic input family without flattening it first."""
        self.require_current_architecture()
        if observations.ndim != 2 or observations.shape[1] != INPUT_COUNT:
            raise ValueError(f"expected observations [N,{INPUT_COUNT}]")
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        positions, position_valid = self._positions(observations)
        pooled_positions, position_embeddings = self.position_set(positions, position_valid)
        market_embedding = self.market_structure(self._market_structure(observations))
        account_embedding = self.account_encoder(
            observations[:, ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET]
        )
        execution_embedding = self.execution_encoder(
            observations[:, EXECUTION_OFFSET:TIMING_OFFSET]
        )
        components = {
            "market": market_embedding,
            "account": account_embedding,
            "execution": execution_embedding,
            "positions": pooled_positions,
            "position_embeddings": position_embeddings,
            "position_valid": position_valid,
            "agent": self.agent_embedding(agent_ids),
            "previous_action": self.previous_action_embedding(
                previous_actions.clamp(0, ACTION_COUNT - 1)
            ),
        }
        components.update(self.pattern_tree(
            market=market_embedding,
            positions=pooled_positions,
            account=account_embedding,
            execution=execution_embedding,
            raw_observations=observations,
        ))
        return components

    def pattern_predictions(self, observations: Tensor) -> dict[str, Tensor]:
        """Predict opportunity labels using present observations only."""
        if observations.ndim != 2 or observations.shape[1] != INPUT_COUNT:
            raise ValueError(f"expected observations [N,{INPUT_COUNT}]")
        count = int(observations.shape[0])
        device = observations.device
        components = self.encode_components(
            observations,
            torch.zeros(count, dtype=torch.long, device=device),
            torch.zeros(count, dtype=torch.long, device=device),
        )
        return {
            "utility": components["pattern_utility"],
            "excursion": components["pattern_excursion"],
            "no_edge_logit": components["pattern_no_edge_logit"],
            "horizon_returns": components["pattern_horizon_returns"],
            "path_dynamics": components["pattern_path_dynamics"],
            "first_hit_logits": components["pattern_first_hit_logits"],
            "action_path": components["pattern_action_path"],
            "action_outcome_logits": components["pattern_action_outcome_logits"],
            "root": components["pattern_root"],
        }

    def forward(
        self,
        observations: Tensor,
        agent_ids: Tensor,
        previous_actions: Tensor,
        *,
        include_pattern_predictions: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, dict[str, Tensor]]:
        components = self.encode_components(observations, agent_ids, previous_actions)
        flattened = torch.cat(
            (
                components["market"],
                components["account"],
                components["execution"],
                components["positions"],
                components["agent"],
                components["previous_action"],
            ),
            dim=-1,
        )
        encoded = torch.tanh(
            self.pre_lstm(flattened) + self.pattern_residual(components["pattern_root"])
        )

        # Public Five architecture max-pools the first 25% across the five
        # replica vectors before the LSTM.  This operation is meaningful only
        # when all five replicas are processed together.
        if encoded.shape[0] == FIVE_AGENT_COUNT:
            width = max(1, int(round(encoded.shape[1] * FIVE_CROSS_AGENT_POOL_FRACTION)))
            shared_prefix = encoded[:, :width].max(dim=0, keepdim=True).values
            encoded = torch.cat(
                (shared_prefix.expand(FIVE_AGENT_COUNT, -1), encoded[:, width:]), dim=-1
            )
        basic = (encoded, components["position_embeddings"], components["position_valid"])
        if not include_pattern_predictions:
            return basic
        return basic + ({
            "utility": components["pattern_utility"],
            "excursion": components["pattern_excursion"],
            "no_edge_logit": components["pattern_no_edge_logit"],
            "horizon_returns": components["pattern_horizon_returns"],
            "path_dynamics": components["pattern_path_dynamics"],
            "first_hit_logits": components["pattern_first_hit_logits"],
            "action_path": components["pattern_action_path"],
            "action_outcome_logits": components["pattern_action_outcome_logits"],
            "root": components["pattern_root"],
        },)


class EntryQueryAdapter(nn.Module):
    """Neutral nonlinear mapping over the primary-action query.

    The final residual projection starts at exact zero, so attaching this module
    to an older checkpoint preserves every action logit.  During teacher
    imitation the adapter learns a role-conditioned nonlinear boundary in the
    compact action-query space. Full recurrent imitation also lets this loss
    flow through that query into the shared LSTM and observation encoder.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        self.normalization = nn.LayerNorm(self.width)
        self.hidden = nn.Linear(self.width, self.width)
        self.output = nn.Linear(self.width, self.width)
        self.role_embedding = nn.Embedding(2, self.width)
        self.reset_neutral()

    def reset_neutral(self) -> None:
        self.normalization.reset_parameters()
        self.hidden.reset_parameters()
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()
            self.role_embedding.weight.zero_()

    def forward(self, projected_query: Tensor, agent_ids: Tensor) -> Tensor:
        if agent_ids.shape != projected_query.shape[:-1]:
            raise ValueError("agent_ids must match entry-query leading dimensions")
        base = torch.tanh(projected_query)
        owner = (agent_ids == 0) | (agent_ids == 1)
        role_ids = agent_ids.clamp(min=0, max=1)
        role = self.role_embedding(role_ids)
        role = role * owner.unsqueeze(-1).to(dtype=role.dtype)
        delta = self.output(F.silu(self.hidden(self.normalization(base) + role)))
        return base + delta


class RoleSpecificEntryCalibrator(nn.Module):
    """Small independently-normalized BUY or SELL calibration branch."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.query_adapter = EntryQueryAdapter(int(width))
        self.output = nn.Linear(int(width), 2)
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def reset_neutral(self) -> None:
        self.query_adapter.reset_neutral()
        with torch.no_grad():
            self.output.weight.zero_()
            self.output.bias.zero_()

    def forward(self, projected_query: Tensor, role_id: int) -> Tensor:
        ids = torch.full(
            projected_query.shape[:-1], int(role_id), dtype=torch.long,
            device=projected_query.device,
        )
        return self.output(self.query_adapter(projected_query, ids))


class EntryRoleCalibrator(nn.Module):
    """Independent normalized entry margins for BUY and SELL.

    Each directional role owns a small nonlinear branch. The decision boundary
    is learned inside that role's output bias at training time; runtime no
    longer subtracts one validation-era absolute margin threshold. Roles 2..4
    always receive an exact zero residual.
    """

    def __init__(self, query_width: int = ACTION_EMBED_SIZE) -> None:
        super().__init__()
        self.query_width = int(query_width)
        self.role_calibrators = nn.ModuleList([
            RoleSpecificEntryCalibrator(self.query_width),
            RoleSpecificEntryCalibrator(self.query_width),
        ])
        self.teacher_entry_scope = "market_only_v1"
        self.teacher_imitation_trained = False
        self.teacher_calibration_mode = "role_specific_zero_boundary_v3"

    def require_current_mapping(self) -> bool:
        branches = getattr(self, "role_calibrators", None)
        if (
            isinstance(branches, nn.ModuleList)
            and len(branches) == 2
            and all(isinstance(item, RoleSpecificEntryCalibrator) for item in branches)
        ):
            if "hidden" in self._modules:
                del self._modules["hidden"]
                return True
            self.teacher_calibration_mode = str(getattr(
                self, "teacher_calibration_mode", "role_specific_zero_boundary_v3"
            ))
            return False

        legacy_output = getattr(self, "output", None)
        legacy_adapter = getattr(self, "query_adapter", None)
        legacy_bias = getattr(self, "role_entry_bias", None)
        width = int(
            getattr(legacy_output, "in_features", getattr(self, "query_width", ACTION_EMBED_SIZE))
        )
        reference = (
            legacy_output.weight if isinstance(legacy_output, nn.Linear)
            else next(self.parameters(), torch.zeros((), dtype=torch.float32))
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(73_000 + width)
            migrated = nn.ModuleList([
                RoleSpecificEntryCalibrator(width), RoleSpecificEntryCalibrator(width)
            ]).to(device=reference.device, dtype=reference.dtype)
        if isinstance(legacy_adapter, EntryQueryAdapter) and isinstance(legacy_output, nn.Linear):
            for role, branch in enumerate(migrated):
                branch.query_adapter.load_state_dict(legacy_adapter.state_dict())
                branch.output.load_state_dict(legacy_output.state_dict())
                if isinstance(legacy_bias, Tensor) and legacy_bias.numel() == 2:
                    with torch.no_grad():
                        branch.output.bias[1].add_(legacy_bias.reshape(-1)[role])
        self.role_calibrators = migrated
        self.query_width = width
        self.teacher_calibration_mode = "role_specific_zero_boundary_v3"
        for obsolete in ("query_adapter", "output", "hidden"):
            self._modules.pop(obsolete, None)
        self._parameters.pop("role_entry_bias", None)
        return True

    def reset_teacher_mapping(self) -> None:
        self.require_current_mapping()
        for branch in self.role_calibrators:
            branch.reset_neutral()
        self.teacher_calibration_mode = "role_specific_zero_boundary_v3"

    def forward_projected(self, projected_query: Tensor, agent_ids: Tensor) -> Tensor:
        self.require_current_mapping()
        if agent_ids.shape != projected_query.shape[:-1]:
            raise ValueError("agent_ids must match entry-query leading dimensions")
        zeros = projected_query.new_zeros((*projected_query.shape[:-1], 2))
        buy = self.role_calibrators[0](projected_query, 0)
        sell = self.role_calibrators[1](projected_query, 1)
        return torch.where(
            (agent_ids == 0).unsqueeze(-1), buy,
            torch.where((agent_ids == 1).unsqueeze(-1), sell, zeros),
        )


class InternalPolicyState(nn.Module):
    """Learned continuous policy state embedded in the shared Five network.

    The state is intentionally unnamed and continuous: the model is free to
    discover different decision circuits from causal history and reward,
    rather than being handed a fixed list of trading strategies.  For a
    five-agent batch the same state is derived from the mean post-LSTM
    representation and broadcast back to the agents.  Role authority remains
    in the existing action mask/calibrator; this module only changes the
    action surface that PPO trains.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        agent_count: int = FIVE_AGENT_COUNT,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.agent_count = int(agent_count)
        self.latent_size = int(np.clip(self.hidden_size // 2, 8, 32))
        self.projection_width = max(16, min(128, self.hidden_size))
        self.encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.latent_size),
        )
        self.action_residual = nn.Sequential(
            nn.Linear(self.latent_size, self.projection_width),
            nn.Tanh(),
            nn.Linear(self.projection_width, ACTION_COUNT),
        )
        # A migrated/untouched checkpoint must remain behavior-preserving.
        # PPO can still train the final projection immediately; after that
        # gradient reaches the latent encoder through the residual path.
        nn.init.zeros_(self.action_residual[-1].weight)
        nn.init.zeros_(self.action_residual[-1].bias)

    def forward(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("internal policy state hidden width mismatch")
        source = hidden
        # Agent identity is an authority concern, not a strategy label.  A
        # frame-level state avoids turning the five role replicas into five
        # hand-authored strategies while remaining fully causal.
        if hidden.ndim == 2 and int(hidden.shape[0]) == self.agent_count:
            shared = hidden.mean(dim=0, keepdim=True)
            source = shared.expand_as(hidden)
        elif hidden.ndim >= 3 and int(hidden.shape[-2]) == self.agent_count:
            shared = hidden.mean(dim=-2, keepdim=True)
            source = shared.expand_as(hidden)
        latent = self.encoder(source)
        residual = self.action_residual(latent)
        return latent, residual


class FactorizedActionHeads(nn.Module):
    """Factorized action surface matching the published topology."""

    def __init__(self, hidden_size: int = FIVE_LSTM_SIZE) -> None:
        super().__init__()
        # This is a continuous policy state learned *inside* the shared
        # network.  It is deliberately not a categorical strategy id and it
        # is not selected by an external router.  The state is derived from
        # the causal post-LSTM representation and its bounded residual is
        # added to the action surface below.  The final projection starts at
        # zero so loading an older checkpoint keeps its public behavior until
        # PPO has evidence with which to train the new internal state.
        # Do not perturb the caller's RNG stream while adding this
        # zero-output compatibility module; existing checkpoints/tests must
        # keep their legacy action-head initialization exactly.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(93_000 + int(hidden_size))
            self.internal_policy_state_net = InternalPolicyState(hidden_size)
        self.internal_policy_state_enabled = True
        self.internal_policy_state_scale = 0.25
        self.action_query = nn.Linear(hidden_size, ACTION_EMBED_SIZE)
        self.action_embedding = nn.Embedding(ACTION_COUNT, ACTION_EMBED_SIZE)
        self.entry_role_calibrator = EntryRoleCalibrator(ACTION_EMBED_SIZE)

        self.target_query = nn.Linear(hidden_size, TARGET_QUERY_SIZE)
        self.target_key = nn.Linear(POSITION_EMBED_SIZE, TARGET_QUERY_SIZE, bias=False)
        self.target_action_mask = nn.Embedding(ACTION_COUNT, POSITION_EMBED_SIZE)

        self.parameter_heads = nn.ModuleDict({
            name: nn.Linear(hidden_size, bins) for name, bins in PARAMETER_BINS.items()
        })
        for name, default_index in GEOMETRY_HEAD_DEFAULTS.items():
            self._initialize_geometry_head(self.parameter_heads[name], default_index)
        self.value = nn.Linear(hidden_size, 1)
        # Outcome labels are nonlinear in the frozen recurrent state.  Keep a
        # small residual head for the direct seven-choice experiment instead
        # of forcing the old bilinear action surface to represent that map.
        # Its last layer starts at zero, so an untouched checkpoint preserves
        # the legacy primary logits exactly.
        self.direct_action_head = nn.Sequential(
            nn.Linear(int(hidden_size), 128),
            nn.Tanh(),
            nn.Linear(128, 7),
        )
        nn.init.zeros_(self.direct_action_head[-1].weight)
        nn.init.zeros_(self.direct_action_head[-1].bias)
        self.direct_observation_head = nn.Sequential(
            nn.Linear(INPUT_COUNT, 128),
            nn.Tanh(),
            nn.Linear(128, 7),
        )
        nn.init.zeros_(self.direct_observation_head[-1].weight)
        nn.init.zeros_(self.direct_observation_head[-1].bias)
        # The expected-value surface is learned from the causal observation
        # root.  A small, separately gated residual lets a research run use
        # the recurrent/temporal state as a correction without changing the
        # frozen trunk or the seven-choice contract.  Zero initialization
        # keeps an untrained or legacy checkpoint bit-for-bit equivalent.
        self.expected_value_hidden_residual = nn.Sequential(
            nn.Linear(int(hidden_size), 128),
            nn.Tanh(),
            nn.Linear(128, 6),
        )
        nn.init.zeros_(self.expected_value_hidden_residual[-1].weight)
        nn.init.zeros_(self.expected_value_hidden_residual[-1].bias)
        self.seven_choice_expected_value_hidden_residual_enabled = False
        self.seven_choice_expected_value_hidden_residual_scale_r = 0.0
        # Codex feedback must be state-conditional.  Updating the shared
        # action query/embedding for one terminal packet changes every future
        # context, which can turn a small bounded packet shift into a large
        # out-of-sample policy drift.  This zero-initialized residual is an
        # opt-in candidate surface: legacy checkpoints keep identical logits
        # until the Codex learner explicitly enables and trains it.
        self.codex_action_residual = nn.Sequential(
            nn.Linear(int(hidden_size), 64),
            nn.Tanh(),
            nn.Linear(64, ACTION_COUNT),
        )
        nn.init.zeros_(self.codex_action_residual[-1].weight)
        nn.init.zeros_(self.codex_action_residual[-1].bias)
        self.codex_action_residual_enabled = False
        self.codex_action_residual_scale = 1.0
        # The contextual residual is intentionally gated by a hidden-state
        # prototype.  A verified terminal reward must nevertheless be able to
        # influence later, unseen contexts; otherwise a 0.995 cosine gate
        # turns every successful update into a one-state-only no-op.  This
        # separate zero-initialized action bias is a bounded candidate-only
        # global correction.  It never changes an untouched/legacy checkpoint
        # and is trained only by the Codex teacher optimizer path.
        self.codex_action_bias = nn.Parameter(torch.zeros(ACTION_COUNT))
        self.codex_action_bias_enabled = False
        self.codex_action_bias_scale = 0.50
        # A terminal receipt owns one action, not the whole entry class.  A
        # sequence of negative receipts for BUY/SELL variants must therefore
        # preserve their relative preference without creating an unbounded
        # common entry-vs-HOLD drift.  The flag is a runtime compatibility
        # switch; old checkpoints default to the safe centered behavior when
        # the attribute is absent after unpickling.
        self.codex_entry_bias_centering_enabled = True
        # The Python gate attributes above are useful for temporary runtime
        # probes, but Python attributes are not part of ``state_dict``.  Keep
        # a persistent one-bit marker so a candidate that learned a Codex
        # surface does not silently disable it after save/reload.
        self.register_buffer(
            "codex_action_surface_enabled",
            torch.tensor(False, dtype=torch.bool),
        )
        # A residual trained from a handful of terminal packets must not
        # generalize to every hidden state in the policy.  Store normalized
        # owner-state prototypes and gate the residual by nearest-prototype
        # support.  The empty buffer keeps legacy checkpoints behaviorally
        # neutral until the Codex learner records a verified context.
        self.register_buffer(
            "codex_context_prototypes",
            torch.empty((0, int(hidden_size)), dtype=torch.float32),
        )
        self.codex_context_prototype_capacity = 128
        self.codex_context_similarity_threshold = 0.90
        self.codex_context_support_floor = 0.50

    def set_codex_action_surface_enabled(self, enabled: bool = True) -> None:
        """Persist the Codex action-surface gate and mirror runtime gates.

        The optimizer uses the runtime attributes to temporarily disable the
        residual while probing the frozen trunk.  Therefore the persisted
        marker is updated only when the whole surface is explicitly enabled;
        a temporary residual probe must not turn it off.
        """
        value = bool(enabled)
        self.codex_action_residual_enabled = value
        self.codex_action_bias_enabled = value
        marker = getattr(self, "codex_action_surface_enabled", None)
        if isinstance(marker, Tensor) and marker.numel() == 1:
            marker.fill_(value)

    def _codex_action_surface_has_signal(self) -> bool:
        """Detect a learned legacy surface when its marker is absent."""
        bias = getattr(self, "codex_action_bias", None)
        if isinstance(bias, Tensor) and bias.numel() == ACTION_COUNT:
            if bool(torch.any(torch.abs(bias.detach()) > 1e-8)):
                return True
        residual = getattr(self, "codex_action_residual", None)
        if isinstance(residual, nn.Module):
            for parameter in residual.parameters():
                if bool(torch.any(torch.abs(parameter.detach()) > 1e-8)):
                    return True
        prototypes = getattr(self, "codex_context_prototypes", None)
        return isinstance(prototypes, Tensor) and prototypes.numel() > 0

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Restore the non-parameter Codex gate without breaking old states."""
        gate_key = f"{prefix}codex_action_surface_enabled"
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # Checkpoints created before the internal policy state was added do
        # not contain its tensors.  The module is zero-output initialized, so
        # keeping the freshly constructed tensors is the compatible migration
        # behavior; do not make strict loading reject those artifacts.
        latent_prefix = f"{prefix}internal_policy_state_net."
        missing_keys[:] = [
            key for key in missing_keys if not key.startswith(latent_prefix)
        ]
        if gate_key in state_dict:
            enabled = bool(self.codex_action_surface_enabled.detach().item())
            self.codex_action_residual_enabled = enabled
            self.codex_action_bias_enabled = enabled
        else:
            # Old state_dict artifacts predate the marker.  They should still
            # load strictly when their learned Codex tensors contain signal.
            while gate_key in missing_keys:
                missing_keys.remove(gate_key)
            enabled = bool(
                getattr(self, "codex_action_residual_enabled", False)
                or getattr(self, "codex_action_bias_enabled", False)
                or self._codex_action_surface_has_signal()
            )
            self.codex_action_residual_enabled = enabled
            self.codex_action_bias_enabled = enabled
            self.codex_action_surface_enabled.fill_(enabled)

    @staticmethod
    def _initialize_geometry_head(head: nn.Linear, default_index: int) -> None:
        """Start new geometry heads at the legacy 2.5/5.0/no-trail surface."""

        nn.init.zeros_(head.weight)
        nn.init.constant_(head.bias, -2.0)
        with torch.no_grad():
            head.bias[int(default_index)] = 2.0

    def require_current_architecture(self, hidden_size: int) -> bool:
        """Attach current geometry heads while preserving legacy RR behavior."""

        migrated = False
        reference = self.action_query.weight
        internal_state = getattr(self, "internal_policy_state_net", None)
        expected_latent_size = int(np.clip(int(hidden_size) // 2, 8, 32))
        expected_width = max(16, min(128, int(hidden_size)))
        if not (
            isinstance(internal_state, InternalPolicyState)
            and int(getattr(internal_state, "hidden_size", -1)) == int(hidden_size)
            and int(getattr(internal_state, "latent_size", -1)) == expected_latent_size
            and int(getattr(internal_state, "projection_width", -1)) == expected_width
        ):
            # Migration is deterministic and does not consume the caller's
            # RNG stream.  The residual output starts at zero, so an old
            # checkpoint keeps its action surface until PPO trains this new
            # internal representation.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(93_000 + int(hidden_size))
                internal_state = InternalPolicyState(int(hidden_size)).to(
                    device=reference.device, dtype=reference.dtype
                )
            self.internal_policy_state_net = internal_state
            migrated = True
        if not hasattr(self, "internal_policy_state_enabled"):
            self.internal_policy_state_enabled = True
            migrated = True
        if not hasattr(self, "internal_policy_state_scale"):
            self.internal_policy_state_scale = 0.25
            migrated = True
        self.internal_policy_state_scale = float(np.clip(
            float(getattr(self, "internal_policy_state_scale", 0.25) or 0.25),
            0.0,
            1.0,
        ))
        hidden_residual = getattr(self, "expected_value_hidden_residual", None)
        if not (
            isinstance(hidden_residual, nn.Sequential)
            and len(hidden_residual) == 3
            and isinstance(hidden_residual[0], nn.Linear)
            and isinstance(hidden_residual[1], nn.Tanh)
            and isinstance(hidden_residual[2], nn.Linear)
            and int(hidden_residual[0].in_features) == int(hidden_size)
            and int(hidden_residual[0].out_features) == 128
            and int(hidden_residual[2].in_features) == 128
            and int(hidden_residual[2].out_features) == 6
        ):
            hidden_residual = nn.Sequential(
                nn.Linear(int(hidden_size), 128),
                nn.Tanh(),
                nn.Linear(128, 6),
            ).to(device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(hidden_residual[-1].weight)
            nn.init.zeros_(hidden_residual[-1].bias)
            self.expected_value_hidden_residual = hidden_residual
            migrated = True
        if not hasattr(self, "seven_choice_expected_value_hidden_residual_enabled"):
            self.seven_choice_expected_value_hidden_residual_enabled = False
            migrated = True
        if not hasattr(self, "seven_choice_expected_value_hidden_residual_scale_r"):
            self.seven_choice_expected_value_hidden_residual_scale_r = 0.0
            migrated = True
        codex_residual = getattr(self, "codex_action_residual", None)
        if not (
            isinstance(codex_residual, nn.Sequential)
            and len(codex_residual) == 3
            and isinstance(codex_residual[0], nn.Linear)
            and isinstance(codex_residual[1], nn.Tanh)
            and isinstance(codex_residual[2], nn.Linear)
            and int(codex_residual[0].in_features) == int(hidden_size)
            and int(codex_residual[0].out_features) == 64
            and int(codex_residual[2].in_features) == 64
            and int(codex_residual[2].out_features) == ACTION_COUNT
        ):
            # Do not consume the caller's RNG while migrating an old
            # checkpoint.  The final layer is zeroed so adding the module is
            # behavior-preserving until a Codex update opts into the surface.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(87_000 + int(hidden_size))
                codex_residual = nn.Sequential(
                    nn.Linear(int(hidden_size), 64),
                    nn.Tanh(),
                    nn.Linear(64, ACTION_COUNT),
                ).to(device=reference.device, dtype=reference.dtype)
                nn.init.zeros_(codex_residual[-1].weight)
                nn.init.zeros_(codex_residual[-1].bias)
            self.codex_action_residual = codex_residual
            migrated = True
        if not hasattr(self, "codex_action_residual_enabled"):
            self.codex_action_residual_enabled = False
            migrated = True
        if not hasattr(self, "codex_action_residual_scale"):
            self.codex_action_residual_scale = 1.0
            migrated = True
        self.codex_action_residual_scale = float(np.clip(
            getattr(self, "codex_action_residual_scale", 1.0) or 1.0,
            0.0,
            2.0,
        ))
        codex_bias = getattr(self, "codex_action_bias", None)
        if not isinstance(codex_bias, Tensor) or codex_bias.numel() != ACTION_COUNT:
            replacement = reference.new_zeros((ACTION_COUNT,))
            if isinstance(codex_bias, Tensor) and codex_bias.numel() == ACTION_COUNT:
                replacement.copy_(codex_bias.detach().reshape(-1).to(
                    device=reference.device, dtype=reference.dtype
                ))
            self.codex_action_bias = nn.Parameter(replacement)
            migrated = True
        elif not isinstance(codex_bias, nn.Parameter):
            self.codex_action_bias = nn.Parameter(codex_bias.detach().clone())
            migrated = True
        if not hasattr(self, "codex_action_bias_enabled"):
            self.codex_action_bias_enabled = False
            migrated = True
        if not hasattr(self, "codex_action_bias_scale"):
            self.codex_action_bias_scale = 0.50
            migrated = True
        self.codex_action_bias_scale = float(np.clip(
            float(getattr(self, "codex_action_bias_scale", 0.50) or 0.50),
            0.0,
            0.50,
        ))
        gate_created = False
        gate_marker = getattr(self, "codex_action_surface_enabled", None)
        if not isinstance(gate_marker, Tensor) or gate_marker.numel() != 1:
            gate_marker = torch.zeros(
                (), dtype=torch.bool, device=reference.device,
            )
            if "codex_action_surface_enabled" in self._buffers:
                self._buffers["codex_action_surface_enabled"] = gate_marker
            else:
                self.register_buffer("codex_action_surface_enabled", gate_marker)
            gate_created = True
            migrated = True
        legacy_gate_enabled = bool(
            getattr(self, "codex_action_residual_enabled", False)
            or getattr(self, "codex_action_bias_enabled", False)
        )
        if legacy_gate_enabled or (
            gate_created and self._codex_action_surface_has_signal()
        ):
            gate_marker.fill_(True)
        prototypes = getattr(self, "codex_context_prototypes", None)
        if not (
            isinstance(prototypes, Tensor)
            and prototypes.ndim == 2
            and int(prototypes.shape[1]) == int(hidden_size)
        ):
            empty = reference.new_empty((0, int(hidden_size)))
            if "codex_context_prototypes" in self._buffers:
                self._buffers["codex_context_prototypes"] = empty
            else:
                self.register_buffer("codex_context_prototypes", empty)
            migrated = True
        if not hasattr(self, "codex_context_prototype_capacity"):
            self.codex_context_prototype_capacity = 128
            migrated = True
        self.codex_context_prototype_capacity = int(np.clip(
            getattr(self, "codex_context_prototype_capacity", 128) or 128,
            1,
            512,
        ))
        if not hasattr(self, "codex_context_similarity_threshold"):
            self.codex_context_similarity_threshold = 0.90
            migrated = True
        self.codex_context_similarity_threshold = float(np.clip(
            float(getattr(self, "codex_context_similarity_threshold", 0.90) or 0.90),
            0.90,
            0.9999,
        ))
        if not hasattr(self, "codex_context_support_floor"):
            self.codex_context_support_floor = 0.50
            migrated = True
        self.codex_context_support_floor = float(np.clip(
            float(getattr(self, "codex_context_support_floor", 0.50) or 0.50),
            0.0,
            1.0,
        ))
        direct = getattr(self, "direct_action_head", None)
        if not (
            isinstance(direct, nn.Sequential)
            and len(direct) == 3
            and isinstance(direct[0], nn.Linear)
            and isinstance(direct[2], nn.Linear)
            and int(direct[0].in_features) == int(hidden_size)
            and int(direct[0].out_features) == 128
            and int(direct[2].in_features) == 128
            and int(direct[2].out_features) == 7
        ):
            direct = nn.Sequential(
                nn.Linear(int(hidden_size), 128),
                nn.Tanh(),
                nn.Linear(128, 7),
            ).to(device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(direct[-1].weight)
            nn.init.zeros_(direct[-1].bias)
            self.direct_action_head = direct
            migrated = True
        observation = getattr(self, "direct_observation_head", None)
        if not (
            isinstance(observation, nn.Sequential)
            and len(observation) == 3
            and isinstance(observation[0], nn.Linear)
            and isinstance(observation[2], nn.Linear)
            and int(observation[0].in_features) == INPUT_COUNT
            and int(observation[0].out_features) == 128
            and int(observation[2].in_features) == 128
            and int(observation[2].out_features) == 7
        ):
            observation = nn.Sequential(
                nn.Linear(INPUT_COUNT, 128),
                nn.Tanh(),
                nn.Linear(128, 7),
            ).to(device=reference.device, dtype=reference.dtype)
            nn.init.zeros_(observation[-1].weight)
            nn.init.zeros_(observation[-1].bias)
            self.direct_observation_head = observation
            migrated = True
        calibrator = getattr(self, "entry_role_calibrator", None)
        if (
            not isinstance(calibrator, EntryRoleCalibrator)
        ):
            self.entry_role_calibrator = EntryRoleCalibrator(ACTION_EMBED_SIZE).to(
                device=reference.device, dtype=reference.dtype
            )
            migrated = True
        else:
            migrated = calibrator.require_current_mapping() or migrated
        for name, default_index in GEOMETRY_HEAD_DEFAULTS.items():
            expected_bins = int(PARAMETER_BINS[name])
            existing = self.parameter_heads[name] if name in self.parameter_heads else None
            if isinstance(existing, nn.Linear) and (
                int(existing.in_features), int(existing.out_features)
            ) == (int(hidden_size), expected_bins):
                continue
            head = nn.Linear(int(hidden_size), expected_bins).to(
                device=reference.device, dtype=reference.dtype
            )
            if (
                name in {"sl", "tp"}
                and isinstance(existing, nn.Linear)
                and int(existing.in_features) == int(hidden_size)
                and int(existing.out_features) == len(
                    LEGACY_SL_ATR_BINS if name == "sl" else LEGACY_TP_RR_BINS
                )
            ):
                # Contract v7 inserts 0.75-ATR SL and sub-2R TP choices. Copy
                # every marker-6 row onto the identical value and make new
                # rows neutral for old checkpoints.
                with torch.no_grad():
                    head.weight.zero_()
                    head.bias.fill_(-30.0)
                    old_bins = LEGACY_SL_ATR_BINS if name == "sl" else LEGACY_TP_RR_BINS
                    new_bins = SL_ATR_BINS if name == "sl" else TP_RR_BINS
                    for old_index, value in enumerate(old_bins):
                        new_index = new_bins.index(value)
                        head.weight[new_index].copy_(existing.weight[old_index])
                        head.bias[new_index].copy_(existing.bias[old_index])
            else:
                self._initialize_geometry_head(head, default_index)
            self.parameter_heads[name] = head
            migrated = True
        return migrated

    def primary_action_logits(self, hidden: Tensor, agent_ids: Tensor) -> Tensor:
        query = self.action_query(hidden)
        logits = torch.matmul(query, self.action_embedding.weight.t()) / sqrt(float(ACTION_EMBED_SIZE))
        if agent_ids.shape != hidden.shape[:-1]:
            raise ValueError("agent_ids must match primary-action hidden leading dimensions")
        calibrator = getattr(self, "entry_role_calibrator", None)
        if not isinstance(calibrator, EntryRoleCalibrator):
            return logits
        adjustment = calibrator.forward_projected(query, agent_ids)
        role_buy = (agent_ids == 0).to(dtype=logits.dtype)
        role_sell = (agent_ids == 1).to(dtype=logits.dtype)
        residual = torch.zeros_like(logits)
        residual[..., 0] = adjustment[..., 0] * (role_buy + role_sell)
        entry_adjustment = (
            adjustment[..., 1]
        )
        scope = str(getattr(calibrator, "teacher_entry_scope", "market_only_v1"))
        if scope == "all_entry_mass_v2":
            # The historical teacher labels direction/timing, not execution
            # style. Apply one shared prior to all three entry styles so the
            # existing policy still owns MARKET/LIMIT/STOP ranking.
            for action in LONG_ENTRY_ACTIONS:
                residual[..., action] = entry_adjustment * role_buy
            for action in SHORT_ENTRY_ACTIONS:
                residual[..., action] = entry_adjustment * role_sell
        else:
            residual[..., 1] = entry_adjustment * role_buy
            residual[..., 2] = entry_adjustment * role_sell
        return logits + residual

    def apply_internal_policy_state(
        self,
        action_logits: Tensor,
        hidden: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Apply the learned in-network policy state to the action surface.

        No strategy label or external routing decision is introduced here.
        The returned latent is only an auditable representation of the causal
        state that generated the residual; PPO receives gradients through the
        updated logits and therefore trains this same network component.
        """

        latent_net = getattr(self, "internal_policy_state_net", None)
        if not isinstance(latent_net, InternalPolicyState):
            zeros = action_logits.new_zeros(
                (*action_logits.shape[:-1], 0), dtype=action_logits.dtype
            )
            return action_logits, zeros
        if hidden.shape[:-1] != action_logits.shape[:-1]:
            raise ValueError("internal policy state hidden/action logits shape mismatch")
        latent, residual = latent_net(hidden)
        if not bool(getattr(self, "internal_policy_state_enabled", True)):
            return action_logits, latent
        scale = float(np.clip(
            float(getattr(self, "internal_policy_state_scale", 0.25) or 0.25),
            0.0,
            1.0,
        ))
        return action_logits + scale * torch.tanh(residual), latent

    def direct_action_logits(
        self,
        hidden: Tensor,
        agent_ids: Tensor,
        raw_observations: Tensor | None = None,
    ) -> Tensor:
        """Return the trainable direct seven-choice action surface.

        The residual is deliberately restricted to the two entry owners and
        the first seven contract actions.  Manager and exit actions retain the
        established primary logits, so this experiment cannot silently alter
        their authority or lifecycle behavior.
        """
        logits = self.primary_action_logits(hidden, agent_ids)
        direct = getattr(self, "direct_action_head", None)
        if not isinstance(direct, nn.Module):
            return logits
        if agent_ids.shape != hidden.shape[:-1]:
            raise ValueError("agent_ids must match direct-action hidden leading dimensions")
        delta = direct(hidden)
        observation = getattr(self, "direct_observation_head", None)
        if isinstance(observation, nn.Module) and raw_observations is not None:
            if raw_observations.shape != (*hidden.shape[:-1], INPUT_COUNT):
                raise ValueError(
                    "raw_observations phai cung hang an va co 137 gia tri"
                )
            bounded = torch.tanh(torch.nan_to_num(
                raw_observations, nan=0.0, posinf=0.0, neginf=0.0
            ))
            delta = delta + observation(bounded)
        owner = ((agent_ids == 0) | (agent_ids == 1)).unsqueeze(-1)
        updated = logits.clone()
        updated[..., :7] = torch.where(
            owner,
            updated[..., :7] + delta,
            updated[..., :7],
        )
        return updated

    def apply_codex_action_residual(
        self,
        action_logits: Tensor,
        hidden: Tensor,
    ) -> Tensor:
        """Apply the bounded, support-gated Codex candidate residual."""

        if not bool(getattr(self, "codex_action_residual_enabled", False)):
            return action_logits
        residual = getattr(self, "codex_action_residual", None)
        if not isinstance(residual, nn.Module):
            return action_logits
        if hidden.shape[:-1] != action_logits.shape[:-1]:
            raise ValueError("Codex residual hidden/action logits shape mismatch")
        scale = float(np.clip(
            getattr(self, "codex_action_residual_scale", 1.0) or 1.0,
            0.0,
            2.0,
        ))
        support = self.codex_context_support(hidden).unsqueeze(-1)
        updated = action_logits + scale * residual(hidden) * support
        bias = getattr(self, "codex_action_bias", None)
        if bool(getattr(self, "codex_action_bias_enabled", False)) and isinstance(
            bias, Tensor
        ) and bias.numel() == ACTION_COUNT:
            updated = updated + self.effective_codex_action_bias(
                device=updated.device,
                dtype=updated.dtype,
            )
        return updated

    def effective_codex_action_bias(
        self,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        """Return the bounded Codex bias actually added to action logits.

        Codex updates are action-specific, but the raw global vector is shared
        by every context.  If several entry actions receive negative terminal
        feedback, applying their raw values independently creates a common
        negative shift against HOLD and can make deterministic frozen replay
        choose HOLD forever.  Centering only the six entry actions preserves
        their relative ordering and the scored action's directional change,
        while preventing that shared class drift.  Management actions and the
        HOLD logit are intentionally left untouched.
        """

        target_device = device
        target_dtype = dtype
        bias = getattr(self, "codex_action_bias", None)
        if not isinstance(bias, Tensor) or bias.numel() != ACTION_COUNT:
            if target_device is None:
                target_device = bias.device if isinstance(bias, Tensor) else None
            return torch.zeros(
                (ACTION_COUNT,),
                device=target_device,
                dtype=target_dtype or (bias.dtype if isinstance(bias, Tensor) else torch.float32),
            )
        if target_device is None:
            target_device = bias.device
        if target_dtype is None:
            target_dtype = bias.dtype
        bias_scale = float(np.clip(
            float(getattr(self, "codex_action_bias_scale", 0.50) or 0.50),
            0.0,
            0.50,
        ))
        effective = bias_scale * torch.tanh(
            bias.to(device=target_device, dtype=target_dtype)
        )
        if bool(getattr(self, "codex_entry_bias_centering_enabled", True)):
            centered = effective.clone()
            centered[1:7] = centered[1:7] - centered[1:7].mean()
            effective = centered
        return effective

    def codex_context_support(self, hidden: Tensor) -> Tensor:
        """Return detached nearest-prototype support for each hidden state.

        Support is deliberately non-differentiable and is computed from the
        frozen recurrent/trunk state.  A state at a verified prototype gets
        support 1; states below the configured cosine threshold receive 0,
        and a configured support floor rejects weak interpolation tails.
        This prevents a small per-packet probability update from silently
        changing argmax actions in unrelated OOS contexts.
        """

        prototypes = getattr(self, "codex_context_prototypes", None)
        if not isinstance(prototypes, Tensor) or prototypes.ndim != 2:
            return hidden.new_zeros(hidden.shape[:-1])
        if int(prototypes.shape[0]) == 0:
            return hidden.new_zeros(hidden.shape[:-1])
        if int(prototypes.shape[1]) != int(hidden.shape[-1]):
            return hidden.new_zeros(hidden.shape[:-1])
        nearest = self.codex_context_nearest_similarity(hidden)
        threshold = float(np.clip(
            float(getattr(self, "codex_context_similarity_threshold", 0.90) or 0.90),
            0.90,
            0.9999,
        ))
        support = torch.clamp(
            (nearest - threshold) / max(1.0 - threshold, 1e-6),
            min=0.0,
            max=1.0,
        )
        # A captured prototype is its own verified context.  Normalization can
        # leave cosine similarity at 0.999999 rather than exactly 1.0, so make
        # that identity case deterministic for telemetry and gating.
        support = torch.where(
            nearest >= (1.0 - 1e-6),
            torch.ones_like(support),
            support,
        )
        support_floor = float(np.clip(
            float(getattr(self, "codex_context_support_floor", 0.50) or 0.50),
            0.0,
            1.0,
        ))
        return torch.where(support >= support_floor, support, torch.zeros_like(support))

    def codex_context_nearest_similarity(self, hidden: Tensor) -> Tensor:
        """Return raw nearest-prototype cosine similarity for audit telemetry."""

        prototypes = getattr(self, "codex_context_prototypes", None)
        if not isinstance(prototypes, Tensor) or prototypes.ndim != 2:
            return hidden.new_zeros(hidden.shape[:-1])
        if int(prototypes.shape[0]) == 0 or int(prototypes.shape[1]) != int(hidden.shape[-1]):
            return hidden.new_zeros(hidden.shape[:-1])
        normalized_hidden = F.normalize(hidden.detach(), dim=-1, eps=1e-8)
        normalized_prototypes = F.normalize(
            prototypes.detach().to(device=hidden.device, dtype=hidden.dtype),
            dim=-1,
            eps=1e-8,
        )
        similarity = torch.matmul(
            normalized_hidden.reshape(-1, hidden.shape[-1]),
            normalized_prototypes.transpose(0, 1),
        )
        return similarity.max(dim=-1).values.reshape(hidden.shape[:-1])

    @torch.no_grad()
    def capture_codex_context_prototype(self, hidden: Tensor) -> bool:
        """Record one verified owner hidden state for residual support."""

        vector = hidden.detach().reshape(-1)
        if vector.numel() != int(self.codex_context_prototypes.shape[-1]):
            raise ValueError("Codex context prototype hidden size mismatch")
        if not bool(torch.isfinite(vector).all().item()):
            return False
        norm = torch.linalg.vector_norm(vector)
        if not bool(torch.isfinite(norm).item()) or float(norm.item()) <= 1e-8:
            return False
        vector = (vector / norm).to(
            device=self.codex_context_prototypes.device,
            dtype=self.codex_context_prototypes.dtype,
        )
        existing = self.codex_context_prototypes
        if int(existing.shape[0]) > 0:
            threshold = float(np.clip(
                float(getattr(self, "codex_context_similarity_threshold", 0.90) or 0.90),
                0.90,
                0.9999,
            ))
            support_floor = float(np.clip(
                float(getattr(self, "codex_context_support_floor", 0.50) or 0.50),
                0.0,
                1.0,
            ))
            # A context may be reused only when it would actually pass the
            # residual support gate.  Otherwise capture a fresh prototype;
            # treating the weak interpolation tail as "already captured"
            # creates a successful-looking optimizer update with zero gradient.
            effective_threshold = threshold + support_floor * (1.0 - threshold)
            similarity = torch.mv(existing, vector)
            if float(torch.max(similarity).item()) >= effective_threshold:
                return False
        capacity = int(np.clip(
            getattr(self, "codex_context_prototype_capacity", 128) or 128,
            1,
            512,
        ))
        if int(existing.shape[0]) >= capacity:
            existing = existing[-(capacity - 1):] if capacity > 1 else existing[:0]
        self.codex_context_prototypes = torch.cat((existing, vector.unsqueeze(0)), dim=0)
        return True

    @torch.no_grad()
    def restore_codex_context_prototypes(self, snapshot: Tensor) -> None:
        """Restore the prototype buffer when a bounded update is rejected."""

        self.codex_context_prototypes = snapshot.detach().to(
            device=self.codex_context_prototypes.device,
            dtype=self.codex_context_prototypes.dtype,
        ).clone()

    def target_logits(
        self,
        hidden: Tensor,
        position_embeddings: Tensor,
        position_valid: Tensor,
        sampled_actions: Tensor,
    ) -> Tensor:
        query = self.target_query(hidden)
        action_mask = torch.sigmoid(self.target_action_mask(sampled_actions)).unsqueeze(1)
        masked_units = position_embeddings * action_mask
        keys = self.target_key(masked_units)
        logits = torch.sum(keys * query.unsqueeze(1), dim=-1) / sqrt(float(TARGET_QUERY_SIZE))
        return torch.where(position_valid, logits, torch.full_like(logits, -30.0))


class PostLstmCrossAgentTransformer(nn.Module):
    """Cross-agent attention plus causal multi-timescale temporal tokens.

    Attention over a single LSTM vector would be degenerate. The token axis
    here is therefore the fixed five-agent axis. Production compresses the
    4096-wide LSTM state to 512 for attention, expands it back to 4096, and
    adds a learnable residual correction. Small test profiles adapt the
    bottleneck and head count without changing the topology.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        model_width: int | None = None,
        head_count: int | None = None,
        ffn_size: int | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.model_width = int(
            min(FIVE_POST_LSTM_TRANSFORMER_WIDTH, self.hidden_size)
            if model_width is None else model_width
        )
        if self.hidden_size <= 0 or self.model_width <= 0:
            raise ValueError("hidden_size va transformer model_width phai duong")
        if head_count is None:
            self.head_count = next(
                candidate
                for candidate in (FIVE_POST_LSTM_TRANSFORMER_HEADS, 4, 2, 1)
                if candidate <= self.model_width and self.model_width % candidate == 0
            )
        else:
            self.head_count = int(head_count)
        if self.head_count <= 0 or self.model_width % self.head_count != 0:
            raise ValueError("transformer model_width phai chia het cho head_count")
        self.ffn_size = int(4 * self.model_width if ffn_size is None else ffn_size)
        if self.ffn_size <= 0:
            raise ValueError("transformer ffn_size phai duong")

        self.input_projection = nn.Linear(self.hidden_size, self.model_width)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=self.head_count,
            dim_feedforward=self.ffn_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.output_projection = nn.Linear(self.model_width, self.hidden_size)
        # Small but non-zero: stable at initialization, while every Transformer
        # tensor receives gradient from the first optimizer update.
        self.residual_scale = nn.Parameter(torch.tensor(0.10, dtype=torch.float32))
        self._attach_temporal_architecture(initial_scale=0.05)

    def _attach_temporal_architecture(self, *, initial_scale: float) -> None:
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_width,
            nhead=self.head_count,
            dim_feedforward=self.ffn_size,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_output_projection = nn.Linear(self.model_width, self.hidden_size)
        self.temporal_horizon_embedding = nn.Embedding(
            FIVE_TEMPORAL_TOKEN_COUNT + 1, self.model_width
        )
        self.temporal_agent_embedding = nn.Embedding(FIVE_AGENT_COUNT, self.model_width)
        self.temporal_scale = nn.Parameter(
            torch.tensor(float(initial_scale), dtype=torch.float32)
        )
        self.register_buffer(
            "temporal_decay",
            torch.as_tensor(
                [1.0 - 1.0 / float(horizon) for horizon in FIVE_TEMPORAL_TOKEN_HORIZONS],
                dtype=torch.float32,
            ),
        )

    def require_temporal_architecture(self) -> bool:
        required = (
            isinstance(getattr(self, "temporal_encoder_layer", None), nn.TransformerEncoderLayer)
            and isinstance(getattr(self, "temporal_output_projection", None), nn.Linear)
            and isinstance(getattr(self, "temporal_horizon_embedding", None), nn.Embedding)
            and isinstance(getattr(self, "temporal_agent_embedding", None), nn.Embedding)
            and isinstance(getattr(self, "temporal_scale", None), nn.Parameter)
            and isinstance(getattr(self, "temporal_decay", None), Tensor)
        )
        if required:
            return False
        reference = self.input_projection.weight
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(79_000 + self.hidden_size + self.model_width)
            self._attach_temporal_architecture(initial_scale=0.0)
        self.to(device=reference.device, dtype=reference.dtype)
        return True

    def empty_temporal_memory(
        self, *leading_shape: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros(
                (*leading_shape, FIVE_AGENT_COUNT, FIVE_TEMPORAL_TOKEN_COUNT, self.model_width),
                device=device,
                dtype=dtype,
            ),
            torch.zeros(
                (*leading_shape, FIVE_AGENT_COUNT, FIVE_TEMPORAL_TOKEN_COUNT),
                device=device,
                dtype=torch.bool,
            ),
        )

    def update_temporal_memory(
        self,
        hidden: Tensor,
        temporal_tokens: Tensor,
        temporal_valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Update causal EMA tokens after the current decision representation."""

        squeeze_batch = hidden.ndim == 2
        current_hidden = hidden.unsqueeze(0) if squeeze_batch else hidden
        current_tokens = temporal_tokens.unsqueeze(0) if squeeze_batch else temporal_tokens
        current_valid = temporal_valid.unsqueeze(0) if squeeze_batch else temporal_valid
        expected_tokens = (
            current_hidden.shape[0], FIVE_AGENT_COUNT,
            FIVE_TEMPORAL_TOKEN_COUNT, self.model_width,
        )
        if current_tokens.shape != expected_tokens:
            raise ValueError(f"temporal_tokens must have shape {expected_tokens}")
        if current_valid.shape != expected_tokens[:-1]:
            raise ValueError(f"temporal_valid must have shape {expected_tokens[:-1]}")
        projected = self.input_projection(current_hidden).unsqueeze(-2)
        decay = self.temporal_decay.to(
            device=current_hidden.device, dtype=current_hidden.dtype
        ).reshape(1, 1, FIVE_TEMPORAL_TOKEN_COUNT, 1)
        blended = decay * current_tokens + (1.0 - decay) * projected
        updated = torch.where(current_valid.unsqueeze(-1), blended, projected.expand_as(blended))
        valid = torch.ones_like(current_valid, dtype=torch.bool)
        if squeeze_batch:
            return updated.squeeze(0), valid.squeeze(0)
        return updated, valid

    def forward(
        self,
        hidden: Tensor,
        temporal_tokens: Tensor | None = None,
        temporal_valid: Tensor | None = None,
    ) -> Tensor:
        self.require_temporal_architecture()
        squeeze_batch = hidden.ndim == 2
        tokens = hidden.unsqueeze(0) if squeeze_batch else hidden
        if tokens.ndim != 3:
            raise ValueError("post-LSTM transformer expects [5,H] hoac [B,5,H]")
        if tokens.shape[-2:] != (FIVE_AGENT_COUNT, self.hidden_size):
            raise ValueError(
                f"post-LSTM transformer expects agent/hidden shape "
                f"({FIVE_AGENT_COUNT},{self.hidden_size})"
            )
        projected = self.input_projection(tokens)
        attended = self.encoder_layer(projected)
        correction = self.output_projection(attended)
        output = tokens + self.residual_scale.to(dtype=tokens.dtype) * correction
        batch = tokens.shape[0]
        if temporal_tokens is None or temporal_valid is None:
            memory, valid = self.empty_temporal_memory(
                batch, device=tokens.device, dtype=tokens.dtype
            )
        else:
            memory = temporal_tokens.unsqueeze(0) if squeeze_batch else temporal_tokens
            valid = temporal_valid.unsqueeze(0) if squeeze_batch else temporal_valid
        expected_memory = (
            batch, FIVE_AGENT_COUNT, FIVE_TEMPORAL_TOKEN_COUNT, self.model_width
        )
        if memory.shape != expected_memory or valid.shape != expected_memory[:-1]:
            raise ValueError("temporal memory has wrong shape")
        past = memory.permute(0, 2, 1, 3).reshape(
            batch, FIVE_TEMPORAL_TOKEN_COUNT * FIVE_AGENT_COUNT, self.model_width
        )
        all_tokens = torch.cat((past, projected), dim=1)
        horizons = torch.arange(
            FIVE_TEMPORAL_TOKEN_COUNT + 1, device=tokens.device, dtype=torch.long
        ).repeat_interleave(FIVE_AGENT_COUNT)
        agents = torch.arange(
            FIVE_AGENT_COUNT, device=tokens.device, dtype=torch.long
        ).repeat(FIVE_TEMPORAL_TOKEN_COUNT + 1)
        all_tokens = (
            all_tokens
            + self.temporal_horizon_embedding(horizons).unsqueeze(0)
            + self.temporal_agent_embedding(agents).unsqueeze(0)
        )
        padding = torch.cat((
            ~valid.permute(0, 2, 1).reshape(batch, -1),
            torch.zeros((batch, FIVE_AGENT_COUNT), dtype=torch.bool, device=tokens.device),
        ), dim=1)
        temporal_attended = self.temporal_encoder_layer(
            all_tokens, src_key_padding_mask=padding
        )[:, -FIVE_AGENT_COUNT:, :]
        temporal_correction = self.temporal_output_projection(temporal_attended)
        output = output + self.temporal_scale.to(dtype=tokens.dtype) * temporal_correction
        return output.squeeze(0) if squeeze_batch else output


class SharedFiveTradingPolicy(nn.Module):
    """One shared network executed as five recurrent replicas."""

    def __init__(
        self,
        *,
        pre_lstm_size: int = PRE_LSTM_SIZE,
        hidden_size: int = FIVE_LSTM_SIZE,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.pre_lstm_size = int(pre_lstm_size)
        self.observation_encoder = FiveObservationEncoder(pre_lstm_size=self.pre_lstm_size)
        self.lstm = nn.LSTMCell(self.pre_lstm_size, self.hidden_size)
        self.post_lstm_transformer = PostLstmCrossAgentTransformer(self.hidden_size)
        self.heads = FactorizedActionHeads(hidden_size=self.hidden_size)
        self.policy_architecture_version = FIVE_POLICY_ARCHITECTURE_VERSION
        self.reasoning_schema_version = FIVE_REASONING_SCHEMA_VERSION
        self.reasoning_rl_enabled = False
        self.reasoning_steps = 3
        self.reasoning_action_scale = 0.0
        self.reasoning_action_residual_clip = 0.25

    def forward(
        self,
        observations: Tensor,
        agent_ids: Tensor | None = None,
        previous_actions: Tensor | None = None,
        memories: Sequence[FiveAgentMemory] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """PyTorch-standard forward pass for SharedFiveTradingPolicy.

        Args:
            observations: [5, 137] float32 tensor
            agent_ids: [5] int64 tensor (defaults to 0..4)
            previous_actions: [5] int64 tensor (defaults to 0=HOLD)
            memories: optional sequence of FiveAgentMemory

        Returns:
            action_wire: [5, 20] float32 action wire tensor
            values: [5, 1] float32 state-value estimation tensor
            extra: dict containing intermediate predictions
        """
        self.require_current_architecture()
        device = observations.device
        if agent_ids is None:
            agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
        if previous_actions is None:
            previous_actions = torch.zeros(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
        if memories is None:
            memories = self.initial_memories(device=device)

        encoded, position_embeddings, position_valid, pattern_predictions = self.observation_encoder(
            observations, agent_ids, previous_actions, include_pattern_predictions=True
        )
        h_prev = torch.stack([item.hidden for item in memories]).to(device=device)
        c_prev = torch.stack([item.cell for item in memories]).to(device=device)
        temporal_tokens, temporal_valid = self._temporal_memory_from_memories(
            memories, device=device, dtype=h_prev.dtype
        )
        lstm_hidden, cell = self.lstm(encoded, (h_prev, c_prev))
        hidden = self.post_lstm_transformer(lstm_hidden, temporal_tokens, temporal_valid)

        action_logits = self.heads.primary_action_logits(hidden, agent_ids)
        values = self.heads.value(hidden)  # [5, 1]

        action_wire = torch.zeros((FIVE_AGENT_COUNT, OUTPUT_COUNT), device=device, dtype=observations.dtype)
        action_wire[:, :ACTION_COUNT] = action_logits

        extra = {
            "hidden": hidden,
            "encoded": encoded,
            "pattern_predictions": pattern_predictions,
            "action_logits": action_logits,
        }
        return action_wire, values, extra

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        """Load legacy/current checkpoints while migrating the latent state.

        The latent module is attached lazily by ``require_current_architecture``
        so an old state dict can be loaded before migration.  If a caller has
        already required the current architecture, its zero-output latent keys
        are the only permissible missing keys in a legacy artifact.
        """

        incompatible = super().load_state_dict(
            state_dict, strict=False, assign=assign
        )
        latent_prefix = "heads.internal_policy_state_net."
        incompatible.missing_keys[:] = [
            key for key in incompatible.missing_keys
            if not key.startswith(latent_prefix)
        ]
        if strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            details: list[str] = []
            if incompatible.missing_keys:
                details.append(
                    "Missing key(s) in state_dict: "
                    + ", ".join(f'\"{key}\"' for key in incompatible.missing_keys)
                    + "."
                )
            if incompatible.unexpected_keys:
                details.append(
                    "Unexpected key(s) in state_dict: "
                    + ", ".join(f'\"{key}\"' for key in incompatible.unexpected_keys)
                    + "."
                )
            raise RuntimeError(
                "Error(s) in loading state_dict for SharedFiveTradingPolicy:\n\t"
                + "\n\t".join(details)
            )
        return incompatible

    def _ensure_reasoning_architecture(self, *, force: bool = False) -> bool:
        """Attach the candidate head without changing legacy policy outputs."""
        migrated = False
        has_head = isinstance(getattr(self, "reasoning_head", None), FiveReasoningHead)
        has_reference = isinstance(
            getattr(self, "reasoning_reference_head", None), FiveReasoningHead
        )
        if not force and not bool(getattr(self, "reasoning_rl_enabled", False)):
            if not has_head and not has_reference:
                return False
        if not has_head:
            # Checkpoint migration must not consume the caller's RNG stream.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(51_000 + int(self.hidden_size))
                self.reasoning_head = FiveReasoningHead(self.hidden_size)
            migrated = True
        if not has_reference:
            self.reasoning_reference_head = copylib.deepcopy(self.reasoning_head)
            for parameter in self.reasoning_reference_head.parameters():
                parameter.requires_grad_(False)
            self.reasoning_reference_head.eval()
            migrated = True
        self.reasoning_schema_version = int(
            getattr(self, "reasoning_schema_version", FIVE_REASONING_SCHEMA_VERSION)
            or FIVE_REASONING_SCHEMA_VERSION
        )
        self.reasoning_rl_enabled = bool(
            getattr(self, "reasoning_rl_enabled", False)
        )
        self.reasoning_steps = int(np.clip(
            getattr(self, "reasoning_steps", 3) or 3, 1, 8
        ))
        self.reasoning_action_scale = float(np.clip(
            getattr(self, "reasoning_action_scale", 0.0) or 0.0, 0.0, 1.0
        ))
        self.reasoning_action_residual_clip = float(np.clip(
            getattr(self, "reasoning_action_residual_clip", 0.25) or 0.25,
            0.0,
            2.0,
        ))
        return migrated

    def refresh_reasoning_reference(self) -> None:
        """Explicitly promote the current reasoning head to the frozen KL anchor."""
        self._ensure_reasoning_architecture(force=True)
        self.reasoning_reference_head.load_state_dict(self.reasoning_head.state_dict())
        for parameter in self.reasoning_reference_head.parameters():
            parameter.requires_grad_(False)
        self.reasoning_reference_head.eval()

    def require_current_architecture(self) -> bool:
        observed_version = int(getattr(self, "policy_architecture_version", 1) or 1)
        if (
            observed_version not in {2, 3, FIVE_POLICY_ARCHITECTURE_VERSION}
            or not isinstance(
                getattr(self, "post_lstm_transformer", None),
                PostLstmCrossAgentTransformer,
            )
        ):
            raise RuntimeError(
                "checkpoint Five cu khong co post-LSTM Transformer bat buoc; "
                "hay tao checkpoint kien truc v3/v4 moi"
            )
        migrated = self.post_lstm_transformer.require_temporal_architecture()
        if observed_version != FIVE_POLICY_ARCHITECTURE_VERSION:
            self.policy_architecture_version = FIVE_POLICY_ARCHITECTURE_VERSION
            migrated = True
        migrated = self.observation_encoder.require_current_architecture() or migrated
        heads_migrated = self.heads.require_current_architecture(self.hidden_size)
        # A checkpoint marked as seven-choice trained must carry the complete
        # decision metadata.  Older artifacts only set
        # ``seven_choice_imitation_trained`` and then silently fell back to
        # the primary action logits because every runtime setting used its
        # default.  That makes an MT5 run look like it used the trained value
        # surface while it actually did not.  Fail closed until the candidate
        # is rebuilt and the settings round-trip through pickle.
        if bool(getattr(self.heads, "seven_choice_imitation_trained", False)):
            required = (
                "seven_choice_expected_value_decision_enabled",
                "seven_choice_action_decision_mode",
                "seven_choice_expected_value_threshold_r",
                "seven_choice_expected_value_score_mode",
                "seven_choice_expected_value_fill_probability_floor",
                "seven_choice_expected_value_action_residual_scale",
                "seven_choice_expected_value_hidden_residual_enabled",
                "seven_choice_expected_value_hidden_residual_scale_r",
                "seven_choice_expected_value_logit_scale",
                "seven_choice_fixed_geometry_enabled",
                "seven_choice_fixed_delay_index",
                "seven_choice_fixed_entry_offset_index",
                "seven_choice_fixed_sl_index",
                "seven_choice_fixed_tp_index",
                "seven_choice_fixed_trailing_index",
            )
            missing = [name for name in required if not hasattr(self.heads, name)]
            if missing:
                raise RuntimeError(
                    "checkpoint bay lua chon da danh dau hoc nhung thieu cau "
                    "hinh quyet dinh sau chi phi: " + ", ".join(missing)
                )
            mode = str(getattr(self.heads, "seven_choice_action_decision_mode", ""))
            if mode not in {"expected_value", "direct"}:
                raise RuntimeError(
                    "checkpoint bay lua chon co che do quyet dinh khong hop le: "
                    + mode
                )
            if mode == "expected_value" and not bool(
                getattr(self.heads, "seven_choice_expected_value_decision_enabled", False)
            ):
                raise RuntimeError(
                    "checkpoint bay lua chon khai bao expected_value nhung chua bat "
                    "mat gia tri sau chi phi"
                )
        reasoning_migrated = self._ensure_reasoning_architecture()
        return bool(migrated or heads_migrated or reasoning_migrated)

    def initial_memories(self, device: torch.device | None = None) -> list[FiveAgentMemory]:
        device = device or next(self.parameters()).device
        return [
            FiveAgentMemory.zeros(device=device, hidden_size=self.hidden_size)
            for _ in range(FIVE_AGENT_COUNT)
        ]

    def _apply_seven_choice_expected_value_decision(
        self,
        action_logits: Tensor,
        agent_ids: Tensor,
        pattern_action_path: Tensor,
        policy_hidden: Tensor | None = None,
    ) -> Tensor:
        """Use the trained post-cost value surface as the entry decision boundary.

        The seven-choice teacher predicts one normalized net-R value for each of
        the six entry choices from the causal observation root.  Distilling that
        sparse boundary through recurrent action logits proved brittle under a
        month boundary.  Current seven-choice checkpoints therefore make the
        value surface itself the entry logit source for the two entry owners.
        Other agents and all management actions remain untouched.
        """
        heads = self.heads
        if not bool(getattr(heads, "seven_choice_expected_value_decision_enabled", False)):
            return action_logits
        if pattern_action_path.shape != (*action_logits.shape[:-1], 6, 5):
            raise ValueError("pattern_action_path has wrong shape")
        threshold = float(getattr(heads, "seven_choice_expected_value_threshold_r", 0.0))
        scale = float(getattr(heads, "seven_choice_expected_value_logit_scale", 10.0))
        if not isfinite(threshold) or not isfinite(scale) or scale <= 0.0:
            raise ValueError("seven-choice expected-value decision config is invalid")
        expected_net_r = pattern_action_path[..., 2] * 5.0
        hidden_residual_enabled = bool(
            getattr(
                heads,
                "seven_choice_expected_value_hidden_residual_enabled",
                False,
            )
        )
        hidden_residual_scale = float(
            getattr(
                heads,
                "seven_choice_expected_value_hidden_residual_scale_r",
                0.0,
            )
        )
        if (
            not isfinite(hidden_residual_scale)
            or hidden_residual_scale < 0.0
            or hidden_residual_scale > 2.0
        ):
            raise ValueError("seven-choice hidden value residual scale is invalid")
        if hidden_residual_enabled and hidden_residual_scale > 0.0:
            hidden_head = getattr(heads, "expected_value_hidden_residual", None)
            if not isinstance(hidden_head, nn.Module):
                raise RuntimeError("thieu dau hieu chinh gia tri theo trang thai an")
            if policy_hidden is None:
                raise ValueError(
                    "bat hieu chinh gia tri theo trang thai an nhung thieu policy_hidden"
                )
            if policy_hidden.shape != (*action_logits.shape[:-1], self.hidden_size):
                raise ValueError("policy_hidden khong cung hang dan voi action_logits")
            hidden_delta = torch.tanh(hidden_head(policy_hidden))
            if hidden_delta.shape != (*action_logits.shape[:-1], 6):
                raise ValueError("hieu chinh gia tri theo trang thai an co kich thuoc sai")
            expected_net_r = expected_net_r + hidden_residual_scale * hidden_delta
        # For pending orders the first path component is the probability that
        # the order is actually filled.  The optional win floor is derived
        # from the fixed reward/risk ratio and is only enabled on new
        # seven-choice candidates whose win head was trained on filled rows.
        win_floor = float(getattr(heads, "seven_choice_expected_value_win_probability_floor", 0.0))
        if not isfinite(win_floor) or not 0.0 <= win_floor < 1.0:
            raise ValueError("seven-choice win-probability floor is invalid")
        win_probability = pattern_action_path[..., 1]
        win_eligible = torch.ones_like(expected_net_r, dtype=torch.bool)
        if win_floor > 0.0:
            win_eligible = win_probability >= win_floor
        score_mode = str(getattr(heads, "seven_choice_expected_value_score_mode", "value"))
        if score_mode == "value_times_win":
            expected_net_r = expected_net_r * pattern_action_path[..., 1]
        elif score_mode == "filled_value":
            # The value head is trained on rows that actually filled.  Keep
            # the decision score conditional on a fill; a separate fill head
            # supplies the eligibility floor so an almost-impossible pending
            # order cannot win.  The value is already E[net_R | filled]; do
            # not divide by the fill probability, which would magnify a
            # low-fill action and no longer represent profit per filled trade.
            fill_probability_floor = float(
                getattr(heads, "seven_choice_expected_value_fill_probability_floor", 0.10)
            )
            if not isfinite(fill_probability_floor) or not 0.0 < fill_probability_floor <= 1.0:
                raise ValueError("seven-choice fill-probability floor is invalid")
        elif score_mode == "win_probability":
            expected_net_r = pattern_action_path[..., 1]
        elif score_mode != "value":
            raise ValueError("seven-choice expected-value score mode is invalid")
        # Apply the floor after constructing the requested score.  Otherwise
        # ``win_probability`` mode would overwrite the -inf barrier and allow
        # a low-win action to beat HOLD.
        expected_net_r = torch.where(
            win_eligible,
            expected_net_r,
            torch.full_like(expected_net_r, -1e6),
        )
        if score_mode == "filled_value":
            fill_probability_floor = float(
                getattr(heads, "seven_choice_expected_value_fill_probability_floor", 0.10)
            )
            fill_eligible = pattern_action_path[..., 0] >= fill_probability_floor
            expected_net_r = torch.where(
                fill_eligible,
                expected_net_r,
                torch.full_like(expected_net_r, -1e6),
            )
        action_residual_scale = float(
            getattr(heads, "seven_choice_expected_value_action_residual_scale", 0.0)
        )
        if not isfinite(action_residual_scale) or action_residual_scale < 0.0:
            raise ValueError("seven-choice action residual scale is invalid")
        score_margin = float(
            getattr(heads, "seven_choice_expected_value_min_score_margin", 0.0)
        )
        if not isfinite(score_margin) or score_margin < 0.0:
            raise ValueError("seven-choice minimum score margin is invalid")
        value_eligible = expected_net_r > threshold
        adjusted = action_logits.clone()
        for role, actions in ((0, LONG_ENTRY_ACTIONS), (1, SHORT_ENTRY_ACTIONS)):
            owner = agent_ids == role
            if not bool(owner.any().item()):
                continue
            if score_margin > 0.0:
                role_scores = torch.stack(
                    [expected_net_r[..., action - 1] for action in actions], dim=-1
                )
                top_scores = torch.topk(role_scores, k=2, dim=-1).values
                margin_eligible = (top_scores[..., 0] - top_scores[..., 1]) >= score_margin
            else:
                margin_eligible = torch.ones_like(value_eligible[..., 0], dtype=torch.bool)
            adjusted[..., 0] = torch.where(
                owner, torch.zeros_like(adjusted[..., 0]), adjusted[..., 0]
            )
            for action in actions:
                value_logit = scale * (expected_net_r[..., action - 1] - threshold)
                if action_residual_scale > 0.0:
                    value_logit = value_logit + action_residual_scale * torch.tanh(
                        action_logits[..., action]
                    ) * value_eligible[..., action - 1].to(dtype=value_logit.dtype)
                value_logit = torch.where(
                    margin_eligible,
                    value_logit,
                    torch.full_like(value_logit, -1e6),
                )
                adjusted[..., action] = torch.where(
                    owner, value_logit, adjusted[..., action]
                )
        return adjusted

    def _temporal_memory_from_memories(
        self, memories: Sequence[FiveAgentMemory], *, device: torch.device, dtype: torch.dtype
    ) -> tuple[Tensor, Tensor]:
        transformer = self.post_lstm_transformer
        tokens, valid = transformer.empty_temporal_memory(device=device, dtype=dtype)
        for agent, memory in enumerate(memories):
            item_tokens = getattr(memory, "temporal_tokens", None)
            item_valid = getattr(memory, "temporal_valid", None)
            if item_tokens is None or item_valid is None:
                continue
            candidate_tokens = item_tokens.to(device=device, dtype=dtype)
            candidate_valid = item_valid.to(device=device, dtype=torch.bool)
            if candidate_tokens.shape != (
                FIVE_TEMPORAL_TOKEN_COUNT, transformer.model_width
            ) or candidate_valid.shape != (FIVE_TEMPORAL_TOKEN_COUNT,):
                raise ValueError("FiveAgentMemory temporal token shape is incompatible")
            tokens[agent] = candidate_tokens
            valid[agent] = candidate_valid
        return tokens, valid

    def reset_memories(
        self,
        memories: Sequence[FiveAgentMemory],
        done_mask: Tensor | np.ndarray | Sequence[bool] | None = None,
    ) -> list[FiveAgentMemory]:
        """Reset recurrent state only at a real sequence boundary.

        A normal decision step must never call this method.  ``done_mask`` is
        per agent; omitted means that all five recurrent streams start a new
        independent sequence.
        """
        if len(memories) != FIVE_AGENT_COUNT:
            raise ValueError("expected five recurrent memories")
        device = next(self.parameters()).device
        if done_mask is None:
            mask = torch.ones(FIVE_AGENT_COUNT, dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(done_mask, dtype=torch.bool, device=device).reshape(-1)
            if mask.numel() != FIVE_AGENT_COUNT:
                raise ValueError("done_mask must contain five values")
        fresh = self.initial_memories(device=device)
        out: list[FiveAgentMemory] = []
        for i, memory in enumerate(memories):
            if bool(mask[i].item()):
                out.append(fresh[i])
            else:
                out.append(memory.detached())
        return out

    def advance_history(
        self,
        observations: Tensor,
        memories: Sequence[FiveAgentMemory] | None = None,
        *,
        actions: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> list[FiveAgentMemory]:
        """Replay past observations through the recurrent state.

        ``observations`` has shape ``[time, five, 137]``.  ``actions[t]`` is
        the action actually taken after observation ``t`` and therefore
        becomes the previous-action input for step ``t+1``.  When historical
        actions are unavailable, HOLD (index 0) is kept as the previous
        action.  ``dones[t]`` marks a boundary after step ``t``.

        This method intentionally does not run the action/value heads: it is a
        state warm-up path for rebuilding recurrent memory after loading past
        market observations.
        """
        if observations.ndim != 3 or observations.shape[1:] != (FIVE_AGENT_COUNT, INPUT_COUNT):
            raise ValueError(f"expected observations [T,{FIVE_AGENT_COUNT},{INPUT_COUNT}]")
        steps = int(observations.shape[0])
        device = observations.device
        current = list(memories) if memories is not None else self.initial_memories(device=device)
        if len(current) != FIVE_AGENT_COUNT:
            raise ValueError("expected five recurrent memories")
        if actions is not None and actions.shape != (steps, FIVE_AGENT_COUNT):
            raise ValueError("actions must have shape [T,5]")
        if dones is not None and dones.shape != (steps, FIVE_AGENT_COUNT):
            raise ValueError("dones must have shape [T,5]")
        agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
        h = torch.stack([m.hidden for m in current]).to(device=device)
        c = torch.stack([m.cell for m in current]).to(device=device)
        previous_actions = torch.stack([m.previous_action for m in current]).to(device=device)
        temporal_tokens, temporal_valid = self._temporal_memory_from_memories(
            current, device=device, dtype=h.dtype
        )
        with torch.no_grad():
            for t in range(steps):
                encoded, _, _ = self.observation_encoder(
                    observations[t], agent_ids, previous_actions
                )
                h, c = self.lstm(encoded, (h, c))
                temporal_tokens, temporal_valid = self.post_lstm_transformer.update_temporal_memory(
                    h, temporal_tokens, temporal_valid
                )
                if actions is not None:
                    previous_actions = actions[t].to(device=device, dtype=torch.long).clamp(0, ACTION_COUNT - 1)
                if dones is not None:
                    ended = dones[t].to(device=device, dtype=torch.bool)
                    if bool(ended.any().item()):
                        h = torch.where(ended.unsqueeze(-1), torch.zeros_like(h), h)
                        c = torch.where(ended.unsqueeze(-1), torch.zeros_like(c), c)
                        temporal_tokens = torch.where(
                            ended[:, None, None], torch.zeros_like(temporal_tokens), temporal_tokens
                        )
                        temporal_valid = torch.where(
                            ended[:, None], torch.zeros_like(temporal_valid), temporal_valid
                        )
                        previous_actions = torch.where(
                            ended, torch.zeros_like(previous_actions), previous_actions
                        )
        return [
            FiveAgentMemory(
                hidden=h[i].detach(),
                cell=c[i].detach(),
                previous_action=previous_actions[i].detach(),
                temporal_tokens=temporal_tokens[i].detach(),
                temporal_valid=temporal_valid[i].detach(),
            )
            for i in range(FIVE_AGENT_COUNT)
        ]

    @staticmethod
    def _distribution(
        logits: Tensor,
        deterministic: bool,
        sampling_generator: torch.Generator | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        distribution = torch.distributions.Categorical(logits=logits)
        choice = (
            torch.argmax(logits, dim=-1)
            if deterministic
            else torch.multinomial(
                distribution.probs,
                num_samples=1,
                replacement=True,
                generator=sampling_generator,
            ).squeeze(-1)
        )
        return choice, distribution.log_prob(choice), distribution.entropy()

    @staticmethod
    def _primary_action_distribution(
        logits: Tensor,
        *,
        allowed: Tensor,
        deterministic: bool,
        entry_probability_floor: float,
        uniform_mix: float | Tensor = 0.0,
        include_policy_probs: bool = False,
        sampling_generator: torch.Generator | None = None,
    ) -> tuple[Tensor, ...]:
        """Tra ve rieng policy goc va policy hanh vi dung de lay mau.

        PPO dung log-probability cua policy hanh vi. Loss sua truc tiep action
        dung policy goc de alpha kham pha khong gia mao viec logits da hoc.
        """
        legal = allowed.to(device=logits.device, dtype=torch.bool)
        if legal.shape != logits.shape:
            raise ValueError("primary action allowed mask has wrong shape")
        hold = torch.zeros_like(legal)
        hold[:, 0] = True
        legal = torch.where(legal.any(dim=-1, keepdim=True), legal, hold)
        min_legal = torch.where(legal, logits, torch.full_like(logits, 1e4)).min(dim=-1, keepdim=True).values
        masked_logits = torch.where(legal, logits, min_legal - 50.0)
        base_probs = torch.softmax(masked_logits, dim=-1) * legal.to(dtype=logits.dtype)
        base_prob_sum = torch.nan_to_num(base_probs.sum(dim=-1, keepdim=True), nan=0.0)
        uniform_fallback = legal.to(dtype=logits.dtype) / legal.sum(dim=-1, keepdim=True).clamp_min(1.0)
        base_probs = torch.where(base_prob_sum > 1e-12, base_probs / base_prob_sum.clamp_min(1e-12), uniform_fallback)
        if deterministic or float(entry_probability_floor) <= 0.0:
            probs = base_probs
        else:
            entry_mask = torch.zeros_like(legal)
            entry_mask[:, 1:7] = True
            entry_mask &= legal
            floor = float(np.clip(entry_probability_floor, 0.0, 1.0 / max(1, len(ENTRY_ACTIONS))))
            reserved = entry_mask.sum(dim=-1, keepdim=True).to(dtype=logits.dtype) * floor
            probs = base_probs * (1.0 - reserved) + entry_mask.to(dtype=logits.dtype) * floor
            prob_sum = torch.nan_to_num(probs.sum(dim=-1, keepdim=True), nan=0.0)
            probs = torch.where(prob_sum > 1e-12, probs / prob_sum.clamp_min(1e-12), uniform_fallback)
        if not deterministic:
            alpha = torch.as_tensor(uniform_mix, dtype=logits.dtype, device=logits.device)
            if alpha.ndim == 0:
                alpha = alpha.expand(logits.shape[0])
            alpha = alpha.reshape(-1, 1).clamp(0.0, 1.0)
            if alpha.shape[0] != logits.shape[0]:
                raise ValueError("uniform_mix phai la scalar hoac mot gia tri moi agent")
            uniform = legal.to(dtype=logits.dtype)
            uniform = uniform / uniform.sum(dim=-1, keepdim=True).clamp_min(1.0)
            probs = (1.0 - alpha) * probs + alpha * uniform
            prob_sum = torch.nan_to_num(probs.sum(dim=-1, keepdim=True), nan=0.0)
            probs = torch.where(prob_sum > 1e-12, probs / prob_sum.clamp_min(1e-12), uniform_fallback)
        probs = torch.nan_to_num(probs, nan=0.0)
        distribution = torch.distributions.Categorical(probs=probs)
        choice = (
            torch.argmax(probs, dim=-1)
            if deterministic
            else torch.multinomial(
                probs,
                num_samples=1,
                replacement=True,
                generator=sampling_generator,
            ).squeeze(-1)
        )
        common = (choice, distribution.log_prob(choice), distribution.entropy())
        return common + ((base_probs, probs) if include_policy_probs else (probs,))

    @staticmethod
    def _bin_to_logit(index: int, bins: int) -> float:
        if bins <= 1:
            return 0.0
        p = (float(index) + 0.5) / float(bins)
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return float(log(p / (1.0 - p)))

    @staticmethod
    def _ordinal_median_choice(logits: Tensor) -> Tensor:
        """Choose the Bayes median bin for absolute ordinal log-distance."""

        probabilities = torch.softmax(logits, dim=-1)
        return (torch.cumsum(probabilities, dim=-1) < 0.5).sum(dim=-1).clamp_max(
            logits.shape[-1] - 1
        )

    def forward_five(
        self,
        observations: Tensor,
        memories: Sequence[FiveAgentMemory],
        *,
        allowed_action_masks: Tensor | None = None,
        deterministic: bool = False,
        behavior_uniform_mix: float = 0.0,
        behavior_entry_probability_floor: float | None = None,
        sampling_generator: torch.Generator | None = None,
    ) -> tuple[list[FiveActionResult], list[FiveAgentMemory], dict[str, Tensor]]:
        self.require_current_architecture()
        if observations.shape != (FIVE_AGENT_COUNT, INPUT_COUNT):
            raise ValueError(f"expected observation matrix ({FIVE_AGENT_COUNT},{INPUT_COUNT})")
        if len(memories) != FIVE_AGENT_COUNT:
            raise ValueError("expected five recurrent memories")
        device = observations.device
        agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
        previous_actions = torch.stack([item.previous_action for item in memories]).to(device=device)
        encoded, position_embeddings, position_valid, pattern_predictions = self.observation_encoder(
            observations, agent_ids, previous_actions, include_pattern_predictions=True
        )
        h_prev = torch.stack([item.hidden for item in memories]).to(device=device)
        c_prev = torch.stack([item.cell for item in memories]).to(device=device)
        temporal_tokens, temporal_valid = self._temporal_memory_from_memories(
            memories, device=device, dtype=h_prev.dtype
        )
        lstm_hidden, cell = self.lstm(encoded, (h_prev, c_prev))
        hidden = self.post_lstm_transformer(
            lstm_hidden, temporal_tokens, temporal_valid
        )
        next_temporal_tokens, next_temporal_valid = (
            self.post_lstm_transformer.update_temporal_memory(
                lstm_hidden, temporal_tokens, temporal_valid
            )
        )

        action_mode = str(
            getattr(self.heads, "seven_choice_action_decision_mode", "expected_value")
        )
        action_logits = (
            self.heads.direct_action_logits(hidden, agent_ids, observations)
            if action_mode == "direct"
            else self.heads.primary_action_logits(hidden, agent_ids)
        )
        # Codex feedback is an opt-in residual over the current hidden state.
        # It is zero/disabled for legacy and non-Codex paths, so ordinary
        # checkpoints preserve their action surface exactly.
        codex_context_similarity = self.heads.codex_context_nearest_similarity(hidden)
        codex_context_support = self.heads.codex_context_support(hidden)
        # The expected-value decision surface may replace the entry-owner
        # logits.  Apply the Codex residual *after* that replacement so a
        # verified teacher update reaches the logits that are actually used
        # for the final action, rather than only changing an intermediate
        # tensor that is discarded by the value boundary.
        action_logits = self._apply_seven_choice_expected_value_decision(
            action_logits,
            agent_ids,
            pattern_predictions["action_path"],
            policy_hidden=hidden,
        )
        action_logits, internal_policy_state = self.heads.apply_internal_policy_state(
            action_logits, hidden
        )
        action_logits = self.heads.apply_codex_action_residual(action_logits, hidden)
        codex_global_bias_tensor = action_logits.new_zeros((ACTION_COUNT,))
        codex_global_bias_enabled = bool(
            getattr(self.heads, "codex_action_residual_enabled", False)
            and getattr(self.heads, "codex_action_bias_enabled", False)
        )
        if codex_global_bias_enabled:
            bias = getattr(self.heads, "codex_action_bias", None)
            if isinstance(bias, Tensor) and bias.numel() == ACTION_COUNT:
                codex_global_bias_tensor = self.heads.effective_codex_action_bias(
                    device=action_logits.device,
                    dtype=action_logits.dtype,
                )
                codex_global_bias_scale = float(
                    getattr(self.heads, "codex_action_bias_scale", 0.50) or 0.50
                )
            else:
                codex_global_bias_enabled = False
        codex_global_bias_scale = float(
            getattr(self.heads, "codex_action_bias_scale", 0.0) or 0.0
        ) if codex_global_bias_enabled else 0.0
        reasoning_choices: Tensor | None = None
        reasoning_logits: Tensor | None = None
        if bool(getattr(self, "reasoning_rl_enabled", False)):
            reasoning_allowed = torch.ones(
                (FIVE_AGENT_COUNT, FIVE_REASONING_ACTION_COUNT),
                dtype=torch.bool,
                device=device,
            )
            if allowed_action_masks is not None:
                reasoning_allowed = allowed_action_masks[
                    :, :FIVE_REASONING_ACTION_COUNT
                ].to(device=device, dtype=torch.bool)
            reasoning_choices, reasoning_logits = self.reasoning_head.greedy(
                hidden,
                reasoning_allowed,
                steps=self.reasoning_steps,
            )
            action_scale = float(getattr(self, "reasoning_action_scale", 0.0))
            if action_scale > 0.0:
                centered = reasoning_logits - reasoning_logits.mean(
                    dim=-1, keepdim=True
                )
                residual = torch.clamp(
                    centered,
                    -float(self.reasoning_action_residual_clip),
                    float(self.reasoning_action_residual_clip),
                )
                action_logits = action_logits.clone()
                action_logits[:, :FIVE_REASONING_ACTION_COUNT] = (
                    action_logits[:, :FIVE_REASONING_ACTION_COUNT]
                    + action_scale * residual
                )
        if allowed_action_masks is not None:
            allowed = allowed_action_masks.to(device=device, dtype=torch.bool)
            if allowed.shape != action_logits.shape:
                raise ValueError("allowed_action_masks has wrong shape")
            hold = torch.zeros_like(allowed)
            hold[:, 0] = True
            allowed = torch.where(allowed.any(dim=-1, keepdim=True), allowed, hold)
            for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                action_logits[:, legacy_action] = torch.where(
                    allowed[:, legacy_action],
                    action_logits[:, legacy_action],
                    torch.full_like(action_logits[:, legacy_action], -30.0),
                )
        else:
            allowed = torch.ones_like(action_logits, dtype=torch.bool)
            for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                action_logits[:, legacy_action] = -30.0
                allowed[:, legacy_action] = False
        # The seven-choice teacher labels a complete entry life cycle with
        # fixed stop, target and holding horizon.  Its data does not contain
        # labels for discretionary position management or order cancellation.
        # Letting the old manager act here changes the measured result before
        # the fixed horizon and makes the live run incomparable with its
        # labels.  A fixed-geometry candidate therefore keeps the three
        # non-entry owners at HOLD until their missing outputs are trained.
        if (
            bool(getattr(self.heads, "seven_choice_fixed_geometry_enabled", False))
            or bool(getattr(self.heads, "lock_non_entry_to_hold", False))
            or bool(getattr(self.heads, "adaptive_volatility_targets_enabled", False))
        ) and not bool(getattr(self.heads, "seven_choice_preserve_manager_wire", False)):
            allowed[2:, :] = False
            allowed[2:, 0] = True
        actions, action_log_prob, action_entropy, policy_action_probs, behavior_action_probs = self._primary_action_distribution(
            action_logits,
            allowed=allowed,
            deterministic=deterministic,
            entry_probability_floor=(
                FivePublicProfile().behavior_entry_probability_floor
                if behavior_entry_probability_floor is None
                else float(behavior_entry_probability_floor)
            ),
            uniform_mix=behavior_uniform_mix,
            include_policy_probs=True,
            sampling_generator=sampling_generator,
        )

        target_logits = self.heads.target_logits(
            hidden, position_embeddings, position_valid, actions
        )
        targets, target_log_prob, target_entropy = self._distribution(
            target_logits, deterministic, sampling_generator
        )

        parameter_choices: dict[str, Tensor] = {}
        parameter_log_probs: dict[str, Tensor] = {}
        parameter_entropies: dict[str, Tensor] = {}
        parameter_logits: dict[str, Tensor] = {}
        for name, head in self.heads.parameter_heads.items():
            logits = head(hidden)
            choice, lp, ent = self._distribution(
                logits, deterministic, sampling_generator
            )
            if (
                deterministic
                and name in {"sl", "tp"}
                and bool(getattr(self.heads, "teacher_geometry_imitation_trained", False))
                and int(getattr(
                    self.heads, "teacher_geometry_imitation_schema_version", 0
                )) >= 2
                and name in set(getattr(
                    self.heads,
                    "teacher_geometry_imitation_heads",
                    ("sl", "tp"),
                ))
            ):
                choice = self._ordinal_median_choice(logits)
                distribution = torch.distributions.Categorical(logits=logits)
                lp = distribution.log_prob(choice)
                ent = distribution.entropy()
            parameter_logits[name] = logits
            parameter_choices[name] = choice
            parameter_log_probs[name] = lp
            parameter_entropies[name] = ent

        value = self.heads.value(hidden).squeeze(-1)
        results: list[FiveActionResult] = []
        next_memories: list[FiveAgentMemory] = []
        for i in range(FIVE_AGENT_COUNT):
            action = int(actions[i].item())
            target = int(targets[i].item()) if bool(position_valid[i].any().item()) else 0
            delay = int(parameter_choices["delay"][i].item())
            risk = int(parameter_choices["risk"][i].item())
            entry_offset = int(parameter_choices["entry_offset"][i].item())
            sl = int(parameter_choices["sl"][i].item())
            tp = int(parameter_choices["tp"][i].item())
            trailing = int(parameter_choices["trailing"][i].item())
            manage_mode = int(parameter_choices["manage_mode"][i].item())
            close_fraction = int(parameter_choices["close_fraction"][i].item())

            # Keep the execution surface identical to the counterfactual
            # labels for a seven-choice candidate.  The source checkpoint may
            # carry an older learned geometry (for example TP=0.5R); allowing
            # that stale choice to reach MT5 would invalidate the value head's
            # target.  Only the two entry owners are changed and management
            # actions keep their ordinary factorized parameters.
            is_fixed = bool(getattr(self.heads, "seven_choice_fixed_geometry_enabled", False))
            is_adaptive = bool(getattr(self.heads, "adaptive_volatility_targets_enabled", False))
            fixed_geometry = (is_fixed or is_adaptive) and i in (0, 1) and action in ENTRY_ACTIONS
            adaptive_applied = False
            if fixed_geometry:
                delay = int(getattr(self.heads, "seven_choice_fixed_delay_index", 0))
                entry_offset = int(
                    getattr(self.heads, "seven_choice_fixed_entry_offset_index", 7)
                )
                trailing = int(
                    getattr(self.heads, "seven_choice_fixed_trailing_index", DEFAULT_TRAILING_BIN_INDEX)
                )
                if is_adaptive:
                    chop_val = 0.5
                    m15_dir = 0.0
                    vol_val = 0.5
                    if observations.shape[-1] > 93 and (
                        abs(float(observations[i, 93])) > 1e-5
                        or abs(float(observations[i, 84])) > 1e-5
                        or abs(float(observations[i, 82])) > 1e-5
                    ):
                        chop_val = float(observations[i, 93])
                        m15_dir = float(observations[i, 84])
                        vol_val = float(observations[i, 82])
                    elif observations.shape[-1] > 44 and (
                        abs(float(observations[i, 44])) > 1e-5
                        or (observations.shape[-1] > 36 and abs(float(observations[i, 36])) > 1e-5)
                        or (observations.shape[-1] > 29 and abs(float(observations[i, 29])) > 1e-5)
                    ):
                        chop_val = float(observations[i, 44])
                        m15_dir = float(observations[i, 36]) if observations.shape[-1] > 36 else 0.0
                        vol_val = float(observations[i, 29]) if observations.shape[-1] > 29 else 0.5
                    elif observations.shape[-1] > 93:
                        chop_val = float(observations[i, 93])
                        m15_dir = float(observations[i, 84])
                        vol_val = float(observations[i, 82])

                    from .execution_contract import resolve_adaptive_geometry_indices
                    sl, tp, trailing, _ = resolve_adaptive_geometry_indices(
                        chop_index=chop_val,
                        m15_direction=m15_dir,
                        trend_strength=abs(m15_dir),
                        volatility=vol_val,
                    )
                    adaptive_applied = True
                else:
                    sl = int(getattr(self.heads, "seven_choice_fixed_sl_index", DEFAULT_SL_BIN_INDEX))
                    tp = int(getattr(self.heads, "seven_choice_fixed_tp_index", DEFAULT_TP_BIN_INDEX))

            # Only semantically active heads contribute to behavior log-prob.
            head_lps = {"action": float(action_log_prob[i].item())}
            head_ents = {"action": float(action_entropy[i].item())}
            total_lp = action_log_prob[i]
            total_entropy = action_entropy[i]
            if action in TARGETED_MANAGEMENT_ACTIONS and bool(position_valid[i].any().item()):
                total_lp = total_lp + target_log_prob[i]
                total_entropy = total_entropy + target_entropy[i]
                head_lps["target"] = float(target_log_prob[i].item())
                head_ents["target"] = float(target_entropy[i].item())
                total_lp = total_lp + parameter_log_probs["manage_mode"][i]
                total_entropy = total_entropy + parameter_entropies["manage_mode"][i]
                head_lps["manage_mode"] = float(parameter_log_probs["manage_mode"][i].item())
                head_ents["manage_mode"] = float(parameter_entropies["manage_mode"][i].item())
                if manage_mode == 1:
                    total_lp = total_lp + parameter_log_probs["close_fraction"][i]
                    total_entropy = total_entropy + parameter_entropies["close_fraction"][i]
                    head_lps["close_fraction"] = float(parameter_log_probs["close_fraction"][i].item())
                    head_ents["close_fraction"] = float(parameter_entropies["close_fraction"][i].item())
            # Delay is always a real action parameter in the public Five action
            # factorization. The trading runner stores it in action metadata.
            total_lp = total_lp + parameter_log_probs["delay"][i]
            total_entropy = total_entropy + parameter_entropies["delay"][i]
            head_lps["delay"] = float(parameter_log_probs["delay"][i].item())
            head_ents["delay"] = float(parameter_entropies["delay"][i].item())
            if action in ENTRY_ACTIONS:
                for name in ("risk", "entry_offset", "sl", "tp", "trailing"):
                    total_lp = total_lp + parameter_log_probs[name][i]
                    total_entropy = total_entropy + parameter_entropies[name][i]
                    head_lps[name] = float(parameter_log_probs[name][i].item())
                    head_ents[name] = float(parameter_entropies[name][i].item())

            output = np.zeros(OUTPUT_COUNT, dtype=np.float32)
            output[:ACTION_COUNT] = action_logits[i].detach().cpu().numpy().astype(np.float32)
            # ``action`` may be sampled while learning. The 20-output bridge is
            # the execution command, therefore its decoded argmax must be the
            # exact sampled action. PPO probabilities/log-probabilities remain
            # in FiveActionResult and are not replaced by this execution view.
            output[action] = np.float32(max(float(np.max(output[:ACTION_COUNT])), float(output[action])) + 1.0)
            output[RISK_OUTPUT_INDEX] = np.float32(self._bin_to_logit(risk, PARAMETER_BINS["risk"]))
            # Contract v6 uses the historic compatibility rows 13/14 for the
            # always-active Four-way delay and an explicit factorized marker.
            output[ACTION_DELAY_OUTPUT_INDEX] = np.float32(
                self._bin_to_logit(delay, PARAMETER_BINS["delay"])
            )
            # Fixed seven-choice bins already replaced the entry parameters
            # above. Keep the current wire marker so both decoders read those
            # bins and the delay, and management agents retain their target
            # and partial/full-close commands.
            output[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] = np.float32(FACTORIZED_ACTION_MARKER)
            output[ENTRY_OFFSET_OUTPUT_INDEX] = np.float32(
                self._bin_to_logit(entry_offset, PARAMETER_BINS["entry_offset"])
            )
            output[GEOMETRY_PACKED_OUTPUT_INDEX] = np.float32(
                pack_geometry_choices(sl, tp, trailing)
            )
            output[PARTIAL_CLOSE_OUTPUT_INDEX] = np.float32(
                self._bin_to_logit(close_fraction, PARAMETER_BINS["close_fraction"])
            )
            output[MANAGEMENT_MODE_OUTPUT_INDEX] = np.float32(
                self._bin_to_logit(manage_mode, PARAMETER_BINS["manage_mode"])
            )
            output[TARGET_SLOT_OUTPUT_INDEX] = np.float32(
                self._bin_to_logit(target, MAX_POSITION_SLOTS)
            )

            results.append(FiveActionResult(
                action_index=action,
                target_index=target,
                delay_index=delay,
                risk_index=risk,
                entry_offset_index=entry_offset,
                sl_index=sl,
                tp_index=tp,
                trailing_index=trailing,
                manage_mode_index=manage_mode,
                close_fraction_index=close_fraction,
                output_vector=output,
                log_probability=float(total_lp.item()),
                entropy=float(total_entropy.item()),
                value=float(value[i].item()),
                action_probabilities=behavior_action_probs[i].detach().cpu().numpy().astype(np.float32),
                policy_action_probabilities=policy_action_probs[i].detach().cpu().numpy().astype(np.float32),
                behavior_action_probabilities=behavior_action_probs[i].detach().cpu().numpy().astype(np.float32),
                policy_action_entropy=float(torch.distributions.Categorical(
                    probs=policy_action_probs[i]
                ).entropy().item()),
                behavior_action_entropy=float(action_entropy[i].item()),
                encoded_observation=encoded[i].detach().cpu().numpy().astype(np.float32),
                hidden_before=h_prev[i].detach().cpu().numpy().astype(np.float32),
                hidden_after=lstm_hidden[i].detach().cpu().numpy().astype(np.float32),
                cell_before=c_prev[i].detach().cpu().numpy().astype(np.float32),
                cell_after=cell[i].detach().cpu().numpy().astype(np.float32),
                head_log_probabilities=head_lps,
                head_entropies=head_ents,
                pattern_predictions={
                    "long_utility_norm": float(pattern_predictions["utility"][i, 0].item()),
                    "short_utility_norm": float(pattern_predictions["utility"][i, 1].item()),
                    "long_mfe_norm": float(pattern_predictions["excursion"][i, 0].item()),
                    "long_adverse_norm": float(pattern_predictions["excursion"][i, 1].item()),
                    "short_mfe_norm": float(pattern_predictions["excursion"][i, 2].item()),
                    "short_adverse_norm": float(pattern_predictions["excursion"][i, 3].item()),
                    "no_edge_probability": float(torch.sigmoid(
                        pattern_predictions["no_edge_logit"][i]
                    ).item()),
                    "h4_return_norm": float(pattern_predictions["horizon_returns"][i, 0].item()),
                    "h8_return_norm": float(pattern_predictions["horizon_returns"][i, 1].item()),
                    "h16_return_norm": float(pattern_predictions["horizon_returns"][i, 2].item()),
                    "h32_return_norm": float(pattern_predictions["horizon_returns"][i, 3].item()),
                    "h48_return_norm": float(pattern_predictions["horizon_returns"][i, 4].item()),
                    "long_underwater_probability": float(pattern_predictions["path_dynamics"][i, 2].item()),
                    "short_underwater_probability": float(pattern_predictions["path_dynamics"][i, 7].item()),
                    "pending_fill_probability": float(pattern_predictions["action_path"][i, 2:, 0].mean().item()),
                    **{
                        f"entry_{action}_expected_net_r": float(
                            pattern_predictions["action_path"][i, action - 1, 2].item() * 5.0
                        )
                        for action in range(1, 7)
                    },
                    **{
                        f"entry_{action}_win_probability": float(
                            pattern_predictions["action_path"][i, action - 1, 1].item()
                        )
                        for action in range(1, 7)
                    },
                    "seven_choice_expected_value_threshold_r": float(getattr(
                        self.heads, "seven_choice_expected_value_threshold_r", 0.0
                    )),
                    "seven_choice_expected_value_win_probability_floor": float(getattr(
                        self.heads, "seven_choice_expected_value_win_probability_floor", 0.0
                    )),
                    "seven_choice_expected_value_decision_enabled": bool(getattr(
                        self.heads, "seven_choice_expected_value_decision_enabled", False
                    )),
                    "long_sl_first_probability": float(torch.softmax(
                        pattern_predictions["first_hit_logits"][i, 0], dim=-1
                    )[1].item()),
                    "long_tp_first_probability": float(torch.softmax(
                        pattern_predictions["first_hit_logits"][i, 0], dim=-1
                    )[2].item()),
                    "short_sl_first_probability": float(torch.softmax(
                        pattern_predictions["first_hit_logits"][i, 1], dim=-1
                    )[1].item()),
                    "short_tp_first_probability": float(torch.softmax(
                        pattern_predictions["first_hit_logits"][i, 1], dim=-1
                    )[2].item()),
                },
                reasoning_trace=(
                    {
                        "schema_version": FIVE_REASONING_SCHEMA_VERSION,
                        "steps": [int(value) for value in reasoning_choices[i].tolist()],
                        "final_hypothesis": int(reasoning_choices[i, -1].item()),
                        "action_scale": float(self.reasoning_action_scale),
                        "action_residual_clip": float(
                            self.reasoning_action_residual_clip
                        ),
                        "training_labels_visible": False,
                    }
                    if reasoning_choices is not None
                    else {}
                ),
                temporal_tokens_before=temporal_tokens[i].detach().cpu().numpy().astype(np.float16),
                temporal_valid_before=temporal_valid[i].detach().cpu().numpy().astype(bool),
                policy_action_logits=action_logits[i].detach().cpu().numpy().astype(np.float32),
                internal_policy_state=internal_policy_state[i].detach().cpu().numpy().astype(np.float32),
                codex_context_support=float(codex_context_support[i].detach().item()),
                codex_context_similarity=float(codex_context_similarity[i].detach().item()),
                codex_global_action_bias=float(
                    codex_global_bias_tensor[int(action)].detach().item()
                ),
                codex_global_action_bias_scale=codex_global_bias_scale,
                codex_global_action_bias_enabled=codex_global_bias_enabled,
                adaptive_applied=adaptive_applied,
            ))
            next_memories.append(FiveAgentMemory(
                hidden=lstm_hidden[i].detach(),
                cell=cell[i].detach(),
                previous_action=actions[i].detach(),
                temporal_tokens=next_temporal_tokens[i].detach(),
                temporal_valid=next_temporal_valid[i].detach(),
            ))

        tensors = {
            "encoded": encoded,
            "position_embeddings": position_embeddings,
            "position_valid": position_valid,
            "hidden": lstm_hidden,
            "policy_hidden": hidden,
            "cell": cell,
            "action_logits": action_logits,
            "internal_policy_state": internal_policy_state,
            "policy_action_probabilities": policy_action_probs,
            "codex_context_support": codex_context_support,
            "codex_context_similarity": codex_context_similarity,
            "behavior_action_probabilities": behavior_action_probs,
            "actions": actions,
            "target_logits": target_logits,
            "targets": targets,
            "value": value,
            "pattern_utility": pattern_predictions["utility"],
            "pattern_excursion": pattern_predictions["excursion"],
            "pattern_no_edge_logit": pattern_predictions["no_edge_logit"],
            "pattern_horizon_returns": pattern_predictions["horizon_returns"],
            "pattern_path_dynamics": pattern_predictions["path_dynamics"],
            "pattern_first_hit_logits": pattern_predictions["first_hit_logits"],
            "pattern_action_path": pattern_predictions["action_path"],
            "pattern_action_outcome_logits": pattern_predictions["action_outcome_logits"],
            "reasoning_choices": (
                torch.empty(
                    (FIVE_AGENT_COUNT, 0), dtype=torch.long, device=device
                )
                if reasoning_choices is None else reasoning_choices
            ),
            "total_action_log_prob": torch.as_tensor(
                [item.log_probability for item in results], dtype=torch.float32, device=device
            ),
            "total_entropy": torch.as_tensor(
                [item.entropy for item in results], dtype=torch.float32, device=device
            ),
        }
        for name, logits in parameter_logits.items():
            tensors[f"{name}_logits"] = logits
            tensors[f"{name}_choice"] = parameter_choices[name]
        return results, next_memories, tensors

    def evaluate_sequence(
        self,
        observations: Tensor,
        initial_memories: Sequence[FiveAgentMemory],
        *,
        actions: Tensor,
        targets: Tensor,
        delay_indices: Tensor,
        risk_indices: Tensor,
        entry_offset_indices: Tensor,
        manage_mode_indices: Tensor,
        close_fraction_indices: Tensor,
        sl_indices: Tensor | None = None,
        tp_indices: Tensor | None = None,
        trailing_indices: Tensor | None = None,
        allowed_action_masks: Tensor | None = None,
        episode_dones: Tensor | None = None,
        behavior_uniform_mixes: Tensor | None = None,
        behavior_entry_probability_floor: float | None = None,
    ) -> dict[str, Tensor]:
        """Re-evaluate one truncated recurrent sample without resampling actions.

        Shapes use [time, five, ...]. Recurrent state is carried through the
        whole sample so gradients flow through exactly the unroll window. A
        terminal marker after step ``t`` resets only that agent before
        ``t+1`` so memory never leaks between independent episodes.
        """
        self.require_current_architecture()
        if observations.ndim != 3 or observations.shape[1:] != (FIVE_AGENT_COUNT, INPUT_COUNT):
            raise ValueError(f"expected observations [T,{FIVE_AGENT_COUNT},{INPUT_COUNT}]")
        steps = int(observations.shape[0])
        if steps <= 0 or steps > FIVE_LSTM_UNROLL:
            raise ValueError(f"sequence length must be in [1,{FIVE_LSTM_UNROLL}]")
        if len(initial_memories) != FIVE_AGENT_COUNT:
            raise ValueError("expected five initial memories")
        if episode_dones is not None and episode_dones.shape != (steps, FIVE_AGENT_COUNT):
            raise ValueError("episode_dones must have shape [T,5]")
        if behavior_uniform_mixes is not None and behavior_uniform_mixes.shape != (steps, FIVE_AGENT_COUNT):
            raise ValueError("behavior_uniform_mixes must have shape [T,5]")
        device = observations.device
        if sl_indices is None:
            sl_indices = torch.full((steps, FIVE_AGENT_COUNT), DEFAULT_SL_BIN_INDEX, dtype=torch.long, device=device)
        if tp_indices is None:
            tp_indices = torch.full((steps, FIVE_AGENT_COUNT), DEFAULT_TP_BIN_INDEX, dtype=torch.long, device=device)
        if trailing_indices is None:
            trailing_indices = torch.full((steps, FIVE_AGENT_COUNT), DEFAULT_TRAILING_BIN_INDEX, dtype=torch.long, device=device)
        h = torch.stack([m.hidden for m in initial_memories]).to(device=device)
        c = torch.stack([m.cell for m in initial_memories]).to(device=device)
        previous_actions = torch.stack([m.previous_action for m in initial_memories]).to(device=device)
        temporal_tokens, temporal_valid = self._temporal_memory_from_memories(
            initial_memories, device=device, dtype=h.dtype
        )
        agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)
        log_probs: list[Tensor] = []
        policy_log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        primary_action_log_probs: list[Tensor] = []
        policy_action_log_probs: list[Tensor] = []
        primary_action_entropies: list[Tensor] = []
        primary_entry_probability_mass: list[Tensor] = []
        primary_action_probabilities: list[Tensor] = []
        raw_action_logits_seq: list[Tensor] = []
        policy_action_entropies: list[Tensor] = []
        policy_action_probabilities: list[Tensor] = []
        values: list[Tensor] = []
        hidden_seq: list[Tensor] = []
        policy_hidden_seq: list[Tensor] = []
        internal_policy_state_seq: list[Tensor] = []
        cell_seq: list[Tensor] = []
        for t in range(steps):
            if t > 0 and episode_dones is not None:
                ended = episode_dones[t - 1].to(device=device, dtype=torch.bool)
                if bool(ended.any().item()):
                    # This is a hard recurrent boundary. torch.where also
                    # blocks gradient flow from the new episode into the old
                    # recurrent state for ended agents.
                    h = torch.where(ended.unsqueeze(-1), torch.zeros_like(h), h)
                    c = torch.where(ended.unsqueeze(-1), torch.zeros_like(c), c)
                    previous_actions = torch.where(
                        ended, torch.zeros_like(previous_actions), previous_actions
                    )
                    temporal_tokens = torch.where(
                        ended[:, None, None], torch.zeros_like(temporal_tokens), temporal_tokens
                    )
                    temporal_valid = torch.where(
                        ended[:, None], torch.zeros_like(temporal_valid), temporal_valid
                    )
            encoded, position_embeddings, position_valid, pattern_predictions = self.observation_encoder(
                observations[t], agent_ids, previous_actions,
                include_pattern_predictions=True,
            )
            h, c = self.lstm(encoded, (h, c))
            policy_hidden = self.post_lstm_transformer(h, temporal_tokens, temporal_valid)
            temporal_tokens, temporal_valid = self.post_lstm_transformer.update_temporal_memory(
                h, temporal_tokens, temporal_valid
            )
            action_mode = str(
                getattr(self.heads, "seven_choice_action_decision_mode", "expected_value")
            )
            action_logits = (
                self.heads.direct_action_logits(
                    policy_hidden, agent_ids, observations[t]
                )
                if action_mode == "direct"
                else self.heads.primary_action_logits(policy_hidden, agent_ids)
            )
            action_logits = self._apply_seven_choice_expected_value_decision(
                action_logits,
                agent_ids,
                pattern_predictions["action_path"],
                policy_hidden=policy_hidden,
            )
            action_logits, internal_policy_state = self.heads.apply_internal_policy_state(
                action_logits, policy_hidden
            )
            # Keep the ordering identical to ``forward_five``: expected-value
            # logits are the base decision surface, and the verified Codex
            # residual is the final state-conditional intervention.
            action_logits = self.heads.apply_codex_action_residual(
                action_logits, policy_hidden
            )
            if bool(getattr(self, "reasoning_rl_enabled", False)) and float(
                getattr(self, "reasoning_action_scale", 0.0)
            ) > 0.0:
                reasoning_allowed = torch.ones(
                    (FIVE_AGENT_COUNT, FIVE_REASONING_ACTION_COUNT),
                    dtype=torch.bool,
                    device=device,
                )
                if allowed_action_masks is not None:
                    reasoning_allowed = allowed_action_masks[
                        t, :, :FIVE_REASONING_ACTION_COUNT
                    ].to(device=device, dtype=torch.bool)
                _, reasoning_logits = self.reasoning_head.greedy(
                    policy_hidden, reasoning_allowed, steps=self.reasoning_steps
                )
                centered = reasoning_logits - reasoning_logits.mean(
                    dim=-1, keepdim=True
                )
                residual = torch.clamp(
                    centered,
                    -float(self.reasoning_action_residual_clip),
                    float(self.reasoning_action_residual_clip),
                )
                action_logits = action_logits.clone()
                action_logits[:, :FIVE_REASONING_ACTION_COUNT] = (
                    action_logits[:, :FIVE_REASONING_ACTION_COUNT]
                    + float(self.reasoning_action_scale) * residual
                )
            if allowed_action_masks is not None:
                allowed = allowed_action_masks[t].to(device=device, dtype=torch.bool)
                hold = torch.zeros_like(allowed)
                hold[:, 0] = True
                allowed = torch.where(allowed.any(dim=-1, keepdim=True), allowed, hold)
                for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                    action_logits[:, legacy_action] = torch.where(
                        allowed[:, legacy_action],
                        action_logits[:, legacy_action],
                        torch.full_like(action_logits[:, legacy_action], -30.0),
                    )
            else:
                allowed = torch.ones_like(action_logits, dtype=torch.bool)
                for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                    action_logits[:, legacy_action] = -30.0
                    allowed[:, legacy_action] = False
            # Candidate-only anti-collapse regularization must see the raw
            # policy logits. Runtime legality masks still apply below; a
            # temporary HOLD-only mask must not erase the recovery gradient
            # from the underlying policy.
            raw_action_logits_seq.append(action_logits)
            _, _, _, policy_probs, action_probs = self._primary_action_distribution(
                action_logits,
                allowed=allowed,
                deterministic=False,
                entry_probability_floor=(
                    FivePublicProfile().behavior_entry_probability_floor
                    if behavior_entry_probability_floor is None
                    else float(behavior_entry_probability_floor)
                ),
                uniform_mix=(
                    0.0 if behavior_uniform_mixes is None else behavior_uniform_mixes[t]
                ),
                include_policy_probs=True,
            )
            action_dist = torch.distributions.Categorical(probs=action_probs)
            policy_dist = torch.distributions.Categorical(probs=policy_probs)
            chosen_action = actions[t].to(device=device, dtype=torch.long)
            total_lp = action_dist.log_prob(chosen_action)
            policy_total_lp = policy_dist.log_prob(chosen_action)
            primary_action_log_probs.append(action_dist.log_prob(chosen_action))
            policy_action_log_probs.append(policy_dist.log_prob(chosen_action))
            primary_action_entropies.append(action_dist.entropy())
            primary_entry_probability_mass.append(action_dist.probs[:, 1:7].sum(dim=-1))
            primary_action_probabilities.append(action_dist.probs)
            # Beta entropy phai day policy goc rong ra; alpha chi thay doi
            # policy hanh vi dung de lay mau va tinh likelihood PPO.
            total_ent = policy_dist.entropy()
            policy_action_entropies.append(policy_dist.entropy())
            policy_action_probabilities.append(policy_probs)

            target_logits = self.heads.target_logits(
                policy_hidden, position_embeddings, position_valid, chosen_action
            )
            target_dist = torch.distributions.Categorical(logits=target_logits)
            chosen_target = targets[t].to(device=device, dtype=torch.long)
            target_active = torch.as_tensor(
                [int(a.item()) in TARGETED_MANAGEMENT_ACTIONS for a in chosen_action],
                dtype=torch.bool,
                device=device,
            ) & position_valid.any(dim=1)
            total_lp = total_lp + torch.where(
                target_active, target_dist.log_prob(chosen_target), torch.zeros_like(total_lp)
            )
            policy_total_lp = policy_total_lp + torch.where(
                target_active, target_dist.log_prob(chosen_target), torch.zeros_like(policy_total_lp)
            )
            total_ent = total_ent + torch.where(
                target_active, target_dist.entropy(), torch.zeros_like(total_ent)
            )

            selected_parameter_indices = {
                "delay": delay_indices[t],
                "risk": risk_indices[t],
                "entry_offset": entry_offset_indices[t],
                "sl": sl_indices[t],
                "tp": tp_indices[t],
                "trailing": trailing_indices[t],
                "manage_mode": manage_mode_indices[t],
                "close_fraction": close_fraction_indices[t],
            }
            for name, head in self.heads.parameter_heads.items():
                logits = head(policy_hidden)
                dist = torch.distributions.Categorical(logits=logits)
                choice = selected_parameter_indices[name].to(device=device, dtype=torch.long)
                if name == "delay":
                    active = torch.ones(FIVE_AGENT_COUNT, dtype=torch.bool, device=device)
                elif name in {"risk", "entry_offset", "sl", "tp", "trailing"}:
                    active = torch.as_tensor(
                        [int(a.item()) in ENTRY_ACTIONS for a in chosen_action],
                        dtype=torch.bool,
                        device=device,
                    )
                elif name == "manage_mode":
                    active = target_active
                else:  # close_fraction
                    manage_choice = manage_mode_indices[t].to(device=device, dtype=torch.long)
                    active = target_active & (manage_choice == 1)
                total_lp = total_lp + torch.where(active, dist.log_prob(choice), torch.zeros_like(total_lp))
                policy_total_lp = policy_total_lp + torch.where(
                    active, dist.log_prob(choice), torch.zeros_like(policy_total_lp)
                )
                total_ent = total_ent + torch.where(active, dist.entropy(), torch.zeros_like(total_ent))

            log_probs.append(total_lp)
            policy_log_probs.append(policy_total_lp)
            entropies.append(total_ent)
            values.append(self.heads.value(policy_hidden).squeeze(-1))
            hidden_seq.append(h)
            policy_hidden_seq.append(policy_hidden)
            internal_policy_state_seq.append(internal_policy_state)
            cell_seq.append(c)
            previous_actions = chosen_action
        return {
            "log_probability": torch.stack(log_probs, dim=0),
            # The stored rollout log-probability above belongs to the behavior
            # policy because it includes deliberate uniform exploration. PPO's
            # current-policy side must exclude that sampler mixture; otherwise
            # actor gradients learn the random exploration draws rather than
            # preserving the raw action signal that generated them.
            "policy_log_probability": torch.stack(policy_log_probs, dim=0),
            "entropy": torch.stack(entropies, dim=0),
            "primary_action_log_probability": torch.stack(primary_action_log_probs, dim=0),
            "policy_action_log_probability": torch.stack(policy_action_log_probs, dim=0),
            "primary_action_entropy": torch.stack(primary_action_entropies, dim=0),
            "primary_entry_probability_mass": torch.stack(primary_entry_probability_mass, dim=0),
            "primary_action_probabilities": torch.stack(primary_action_probabilities, dim=0),
            "raw_action_logits": torch.stack(raw_action_logits_seq, dim=0),
            "policy_action_entropy": torch.stack(policy_action_entropies, dim=0),
            "policy_action_probabilities": torch.stack(policy_action_probabilities, dim=0),
            "value": torch.stack(values, dim=0),
            "hidden": torch.stack(hidden_seq, dim=0),
            "policy_hidden": torch.stack(policy_hidden_seq, dim=0),
            "internal_policy_state": torch.stack(internal_policy_state_seq, dim=0),
            "cell": torch.stack(cell_seq, dim=0),
        }

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


@dataclass(frozen=True)
class FiveRolloutStep:
    observations: np.ndarray
    memories_hidden: np.ndarray
    memories_cell: np.ndarray
    previous_actions: np.ndarray
    actions: np.ndarray
    targets: np.ndarray
    delay_indices: np.ndarray
    risk_indices: np.ndarray
    entry_offset_indices: np.ndarray
    manage_mode_indices: np.ndarray
    close_fraction_indices: np.ndarray
    old_log_probabilities: np.ndarray
    values: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    allowed_action_masks: np.ndarray | None = None
    behavior_uniform_mixes: np.ndarray | None = None



@dataclass(frozen=True)
class FiveEconomicState:
    """Cumulative economic state used to build dense but objective-aligned rewards."""

    agent_mark_to_market_r: np.ndarray
    transaction_cost_r: np.ndarray
    agent_risk_ratio: np.ndarray
    conflict_cost_r: np.ndarray
    team_equity_r: float
    team_drawdown: float
    agent_closed_trade_count: np.ndarray = field(default_factory=lambda: np.zeros(FIVE_AGENT_COUNT, dtype=np.float32))
    agent_winning_trade_count: np.ndarray = field(default_factory=lambda: np.zeros(FIVE_AGENT_COUNT, dtype=np.float32))
    agent_expectancy_shortfall_r: np.ndarray = field(default_factory=lambda: np.zeros(FIVE_AGENT_COUNT, dtype=np.float32))
    agent_tail_loss_cost_r: np.ndarray = field(default_factory=lambda: np.zeros(FIVE_AGENT_COUNT, dtype=np.float32))
    agent_realized_trade_r: np.ndarray = field(default_factory=lambda: np.zeros(FIVE_AGENT_COUNT, dtype=np.float32))

    def __post_init__(self) -> None:
        for name in (
            "agent_mark_to_market_r", "transaction_cost_r",
            "agent_risk_ratio", "conflict_cost_r",
            "agent_closed_trade_count", "agent_winning_trade_count",
            "agent_expectancy_shortfall_r", "agent_tail_loss_cost_r",
            "agent_realized_trade_r",
        ):
            values = np.asarray(getattr(self, name), dtype=np.float32).reshape(-1)
            if values.size != FIVE_AGENT_COUNT:
                raise ValueError(f"{name} must contain five values")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} contains non-finite values")
            object.__setattr__(self, name, values)
        if not np.isfinite(float(self.team_equity_r)) or not np.isfinite(float(self.team_drawdown)):
            raise ValueError("team economic state must be finite")


@dataclass(frozen=True)
class FiveRewardBreakdown:
    raw_rewards: np.ndarray
    terminal_components: np.ndarray
    profit_delta_r: np.ndarray
    settled_trade_delta_r: np.ndarray
    profit_basis_r: np.ndarray
    position_improvement_reward_r: np.ndarray
    transaction_cost_r: np.ndarray
    drawdown_increase: float
    risk_excess_penalty: np.ndarray
    conflict_penalty: np.ndarray
    competitive_advantage: np.ndarray
    competitive_bonus: np.ndarray
    strategy_scores: np.ndarray
    terminal_rank_bonus: np.ndarray
    terminal_team_score: float
    closed_trade_delta: np.ndarray
    winning_trade_delta: np.ndarray
    wr_balance_bonus: np.ndarray
    expectancy_shortfall_penalty: np.ndarray
    tail_loss_penalty: np.ndarray
    expectancy_gate: np.ndarray


class FiveTradingRewardShaper:
    """Dense trading reward aligned to marked PnL, drawdown and risk.

    It deliberately does not reward opening, closing, trade count, indicator
    agreement or order acceptance.  This prevents easy reward-hacking paths.
    """

    def __init__(
        self,
        *,
        drawdown_weight: float = 0.30,
        risk_excess_weight: float = 0.10,
        conflict_weight: float = 0.05,
        competition_weight: float = 0.25,
        terminal_rank_weight: float = 0.50,
        terminal_equity_weight: float = 1.0,
        terminal_drawdown_weight: float = 1.0,
        expectancy_wr_v2_enabled: bool = False,
        wr_target: float = 0.55,
        wr_balance_weight: float = 0.05,
        expectancy_shortfall_weight: float = 0.10,
        tail_loss_weight: float = 0.10,
        expectancy_floor_r: float = 0.05,
        position_improvement_weight: float = 1.0,
        reward_mode: str = "dense_mtm",
        settlement_weight: float = 1.0,
        mark_to_market_weight: float = 1.0,
        settlement_clip_r: float = 3.0,
    ) -> None:
        self.drawdown_weight = float(drawdown_weight)
        self.risk_excess_weight = float(risk_excess_weight)
        self.conflict_weight = float(conflict_weight)
        self.competition_weight = float(competition_weight)
        self.terminal_rank_weight = float(terminal_rank_weight)
        # Compatibility-only legacy knobs. P0 removed the unbounded terminal
        # team score; keep accepting old constructor arguments/checkpoints but
        # force their effective contribution to zero.
        del terminal_equity_weight, terminal_drawdown_weight
        self.terminal_equity_weight = 0.0
        self.terminal_drawdown_weight = 0.0
        self.expectancy_wr_v2_enabled = bool(expectancy_wr_v2_enabled)
        self.wr_target = float(np.clip(wr_target, 0.0, 1.0))
        self.wr_balance_weight = float(max(0.0, wr_balance_weight))
        self.expectancy_shortfall_weight = float(max(0.0, expectancy_shortfall_weight))
        self.tail_loss_weight = float(max(0.0, tail_loss_weight))
        self.expectancy_floor_r = float(max(0.0, expectancy_floor_r))
        self.position_improvement_weight = float(max(0.0, position_improvement_weight))
        self.reward_mode = str(reward_mode or "dense_mtm").strip().lower()
        if self.reward_mode not in {"dense_mtm", "settlement_net_r_v1"}:
            raise ValueError(
                "reward_mode must be 'dense_mtm' or 'settlement_net_r_v1'"
            )
        self.settlement_weight = float(max(0.0, settlement_weight))
        self.mark_to_market_weight = float(max(0.0, mark_to_market_weight))
        self.settlement_clip_r = float(max(0.0, settlement_clip_r))
        self.previous: FiveEconomicState | None = None
        self.max_drawdown = 0.0

    def reset(self, initial: FiveEconomicState | None = None) -> None:
        self.previous = initial
        self.max_drawdown = max(0.0, float(initial.team_drawdown)) if initial is not None else 0.0

    def shape(self, state: FiveEconomicState, *, terminal: bool = False) -> FiveRewardBreakdown:
        previous = self.previous
        if previous is None:
            # Each environment episode starts from a known zero economic
            # potential. The former first-observation baseline silently
            # discarded any trade which opened and closed before feedback #1.
            profit_delta = np.asarray(state.agent_mark_to_market_r, dtype=np.float32).copy()
            drawdown_increase = max(0.0, float(state.team_drawdown))
            closed_trade_delta = np.asarray(state.agent_closed_trade_count, dtype=np.float32).copy()
            winning_trade_delta = np.asarray(state.agent_winning_trade_count, dtype=np.float32).copy()
            settled_trade_delta = np.asarray(state.agent_realized_trade_r, dtype=np.float32).copy()
            shortfall_delta = np.asarray(state.agent_expectancy_shortfall_r, dtype=np.float32).copy()
            tail_loss_delta = np.asarray(state.agent_tail_loss_cost_r, dtype=np.float32).copy()
        else:
            profit_delta = (
                np.asarray(state.agent_mark_to_market_r, dtype=np.float32)
                - np.asarray(previous.agent_mark_to_market_r, dtype=np.float32)
            )
            drawdown_increase = max(0.0, float(state.team_drawdown) - float(previous.team_drawdown))
            closed_trade_delta = np.maximum(
                np.asarray(state.agent_closed_trade_count, dtype=np.float32)
                - np.asarray(previous.agent_closed_trade_count, dtype=np.float32), 0.0
            )
            winning_trade_delta = np.maximum(
                np.asarray(state.agent_winning_trade_count, dtype=np.float32)
                - np.asarray(previous.agent_winning_trade_count, dtype=np.float32), 0.0
            )
            # Realized trade PnL is cumulative, but unlike the count deltas it
            # must retain its sign. A losing settlement is the primary
            # negative learning event in settlement_net_r_v1.
            settled_trade_delta = (
                np.asarray(state.agent_realized_trade_r, dtype=np.float32)
                - np.asarray(previous.agent_realized_trade_r, dtype=np.float32)
            )
            shortfall_delta = np.maximum(
                np.asarray(state.agent_expectancy_shortfall_r, dtype=np.float32)
                - np.asarray(previous.agent_expectancy_shortfall_r, dtype=np.float32), 0.0
            )
            tail_loss_delta = np.maximum(
                np.asarray(state.agent_tail_loss_cost_r, dtype=np.float32)
                - np.asarray(previous.agent_tail_loss_cost_r, dtype=np.float32), 0.0
            )
        if self.reward_mode == "settlement_net_r_v1":
            # The runtime MTM is cumulative account value, so a close can
            # leave MTM unchanged while realized PnL increases (open value is
            # simply transferred into the realized bucket). Use MTM only on
            # non-settlement steps; otherwise the close would be counted twice.
            unrealized_delta = np.where(
                np.abs(settled_trade_delta) > 1e-8, 0.0, profit_delta
            )
            profit_basis = (
                np.float32(self.settlement_weight) * settled_trade_delta
                + np.float32(self.mark_to_market_weight) * unrealized_delta
            )
            if self.settlement_clip_r > 0.0:
                profit_basis = np.clip(
                    profit_basis, -self.settlement_clip_r, self.settlement_clip_r
                )
            profit_basis = np.asarray(profit_basis, dtype=np.float32)
        else:
            profit_basis = profit_delta.copy()
        costs = np.maximum(np.asarray(state.transaction_cost_r, dtype=np.float32), 0.0)
        risk_excess = np.maximum(np.asarray(state.agent_risk_ratio, dtype=np.float32) - 1.0, 0.0)
        risk_penalty = self.risk_excess_weight * np.square(risk_excess)
        conflict_penalty = self.conflict_weight * np.maximum(
            np.asarray(state.conflict_cost_r, dtype=np.float32), 0.0
        )
        if self.expectancy_wr_v2_enabled:
            # Sum over trades is positive iff local WR exceeds the configured
            # target. Its scale is deliberately much smaller than one R.
            realized_expectancy = np.divide(
                np.asarray(state.agent_realized_trade_r, dtype=np.float32),
                np.maximum(np.asarray(state.agent_closed_trade_count, dtype=np.float32), 1.0),
            )
            expectancy_gate = (
                (np.asarray(state.agent_closed_trade_count, dtype=np.float32) > 0.0)
                & (realized_expectancy >= np.float32(self.expectancy_floor_r))
            ).astype(np.float32)
            wr_balance_bonus = expectancy_gate * self.wr_balance_weight * (
                winning_trade_delta - np.float32(self.wr_target) * closed_trade_delta
            )
            expectancy_shortfall_penalty = self.expectancy_shortfall_weight * shortfall_delta
            tail_loss_penalty = self.tail_loss_weight * tail_loss_delta
        else:
            wr_balance_bonus = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
            expectancy_shortfall_penalty = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
            tail_loss_penalty = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
            expectancy_gate = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
        position_improvement_reward = (
            np.float32(self.position_improvement_weight) * np.maximum(profit_basis, 0.0)
        ).astype(np.float32)
        rewarded_profit_delta = (
            np.minimum(profit_basis, 0.0) + position_improvement_reward
        ).astype(np.float32)
        economic_delta = (
            rewarded_profit_delta
            + wr_balance_bonus
            - costs
            - np.float32(self.drawdown_weight * drawdown_increase)
            - risk_penalty
            - conflict_penalty
            - expectancy_shortfall_penalty
            - tail_loss_penalty
        ).astype(np.float32)
        # Five's team-spirit term alone can make all replicas converge on the
        # same mediocre strategy.  Trading needs an adversarial signal as well:
        # each replica is rewarded only for *incremental* utility above the
        # five-agent mean.  Using the delta, rather than the cumulative lead,
        # prevents a winning agent from receiving the same bonus repeatedly
        # while the economic state is unchanged.  The vector sums to zero, so
        # it cannot manufacture portfolio profit.
        # Rank only newly created, agent-attributable utility.  Ongoing risk
        # remains a direct penalty for its owner, but must not hand the four
        # inactive opponents a repeated positive reward on every unchanged
        # tick.  Team drawdown is common to all agents and therefore excluded
        # from the relative tournament term as well.
        competitive_basis = (rewarded_profit_delta - costs - conflict_penalty).astype(np.float32)
        competitive_advantage = (
            competitive_basis - np.mean(competitive_basis, dtype=np.float32)
        ).astype(np.float32)
        competitive_bonus = (
            np.float32(self.competition_weight) * competitive_advantage
        ).astype(np.float32)
        raw = (economic_delta + competitive_bonus).astype(np.float32)
        self.max_drawdown = max(self.max_drawdown, max(0.0, float(state.team_drawdown)))
        terminal_score = 0.0
        terminal_component = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
        terminal_rank_bonus = np.zeros(FIVE_AGENT_COUNT, dtype=np.float32)
        # Strategy score deliberately remains profit-first.  Risk and conflict
        # already reduce the step reward; terminal ranking compares cumulative
        # marked-to-market R and uses tanh only to bound one episode's impact.
        strategy_scores = np.asarray(state.agent_mark_to_market_r, dtype=np.float32).copy()
        if terminal:
            # P0 reward fix: cumulative account equity must not be injected as
            # an episode-end reward. Dense marked-to-market deltas already
            # telescope to economic PnL, so adding team_equity_r here counts
            # the same outcome again and, across daily resets, can repeatedly
            # charge/reward the whole tester history. The old unbounded team
            # term also produced terminal shocks on the order of +/-100R.
            #
            # Keep only the bounded, zero-mean relative ranking signal. It is
            # capped by terminal_rank_weight and therefore cannot dominate the
            # economic reward scale. terminal_team_score intentionally stays 0.
            centered_scores = strategy_scores - np.mean(strategy_scores, dtype=np.float32)
            terminal_rank_bonus = (
                np.float32(self.terminal_rank_weight) * np.tanh(centered_scores)
            ).astype(np.float32)
            terminal_component = terminal_rank_bonus.copy()
            raw = raw + terminal_component
        self.previous = state
        return FiveRewardBreakdown(
            raw_rewards=raw.astype(np.float32),
            terminal_components=terminal_component,
            profit_delta_r=profit_delta.astype(np.float32),
            settled_trade_delta_r=settled_trade_delta.astype(np.float32),
            profit_basis_r=profit_basis.astype(np.float32),
            position_improvement_reward_r=position_improvement_reward,
            transaction_cost_r=costs.astype(np.float32),
            drawdown_increase=float(drawdown_increase),
            risk_excess_penalty=risk_penalty.astype(np.float32),
            conflict_penalty=conflict_penalty.astype(np.float32),
            competitive_advantage=competitive_advantage,
            competitive_bonus=competitive_bonus,
            strategy_scores=strategy_scores,
            terminal_rank_bonus=terminal_rank_bonus,
            terminal_team_score=float(terminal_score),
            closed_trade_delta=closed_trade_delta.astype(np.float32),
            winning_trade_delta=winning_trade_delta.astype(np.float32),
            wr_balance_bonus=wr_balance_bonus.astype(np.float32),
            expectancy_shortfall_penalty=expectancy_shortfall_penalty.astype(np.float32),
            tail_loss_penalty=tail_loss_penalty.astype(np.float32),
            expectancy_gate=expectancy_gate.astype(np.float32),
        )


def mix_team_spirit_rewards(
    raw_rewards: np.ndarray,
    team_spirit: float,
    sample_mask: np.ndarray | None = None,
) -> np.ndarray:
    rewards = np.asarray(raw_rewards, dtype=np.float32)
    if rewards.shape[-1] != FIVE_AGENT_COUNT:
        raise ValueError("last reward dimension must be five agents")
    tau = float(np.clip(team_spirit, 0.0, 1.0))
    if sample_mask is None:
        team_mean = np.mean(rewards, axis=-1, keepdims=True)
    else:
        mask = np.asarray(sample_mask, dtype=bool)
        if mask.shape != rewards.shape:
            raise ValueError("sample_mask must have the same shape as raw_rewards")
        active_count = np.sum(mask, axis=-1, keepdims=True, dtype=np.float32)
        active_sum = np.sum(
            np.where(mask, rewards, 0.0), axis=-1, keepdims=True, dtype=np.float32
        )
        team_mean = np.divide(
            active_sum,
            np.maximum(active_count, 1.0),
            out=np.zeros_like(active_sum, dtype=np.float32),
            where=active_count > 0.0,
        )
    return ((1.0 - tau) * rewards + tau * team_mean).astype(np.float32)


def compute_team_spirit_cosine(
    progress: float | Tensor,
    start: float = 0.70,
    end: float = 0.30,
) -> float | Tensor:
    """Computes dynamic team-spirit cosine decay schedule:
    tau(p) = end + 0.5 * (start - end) * [1 + cos(pi * p)]
    """
    if isinstance(progress, Tensor):
        p = torch.clamp(progress, 0.0, 1.0)
        return float(end) + 0.5 * float(start - end) * (1.0 + torch.cos(math.pi * p))
    p = float(np.clip(float(progress) if np.isfinite(progress) else 0.0, 0.0, 1.0))
    cos_term = math.cos(math.pi * p)
    return float(end + 0.5 * (start - end) * (1.0 + cos_term))


def dynamic_team_spirit_cosine_decay(
    progress: float | None = None,
    *,
    step: int | float | None = None,
    total_steps: int | float | None = None,
    tau_max: float = 0.70,
    tau_min: float = 0.30,
) -> float:
    """Dynamic Team-Spirit Cosine Decay schedule.

    tau(t) = tau_min + 0.5 * (tau_max - tau_min) * (1 + cos(pi * t / T))
    where tau_max = 0.70, tau_min = 0.30, t is current step/round progress,
    and T is total horizon/steps.
    """
    if progress is not None:
        p = float(progress)
    elif step is not None and total_steps is not None:
        total = float(total_steps)
        p = float(step) / total if (total > 0.0 and np.isfinite(total)) else 0.0
    elif step is not None and total_steps is None:
        p = float(step)
    else:
        p = 0.0

    if not np.isfinite(p):
        p = 0.0
    p = float(np.clip(p, 0.0, 1.0))

    t_max = float(tau_max)
    t_min = float(tau_min)
    if t_min > t_max:
        t_min, t_max = t_max, t_min

    decay = t_min + 0.5 * (t_max - t_min) * (1.0 + math.cos(math.pi * p))
    return float(np.clip(decay, t_min, t_max))


def time_weight_shaped_rewards(
    rewards: np.ndarray,
    elapsed_seconds: np.ndarray | float,
    *,
    terminal_component: np.ndarray | None = None,
) -> np.ndarray:
    """Return reward without wall-clock decay.

    P0 reward fix: economic credit must not depend on how late in the trading
    day it happens. PPO/GAE already performs temporal discounting; multiplying
    rewards by ``0.6 ** (elapsed_seconds / 600)`` applied a second discount
    tied to episode age and drove late-day PnL, costs and risk penalties toward
    zero. ``elapsed_seconds`` and ``terminal_component`` remain in the signature
    for checkpoint/caller compatibility, but no longer change reward magnitude.
    """
    del elapsed_seconds, terminal_component
    return np.asarray(rewards, dtype=np.float32).copy()


def generalized_advantage_estimation(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    lam: float = 0.95,
    bootstrap_value: np.ndarray | float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("rewards, values and dones must have matching shapes")
    bootstrap = np.asarray(bootstrap_value, dtype=np.float32)
    advantages = np.zeros_like(rewards)
    next_advantage = np.zeros(rewards.shape[1:], dtype=np.float32)
    next_value = np.broadcast_to(bootstrap, rewards.shape[1:]).astype(np.float32).copy()
    for t in range(rewards.shape[0] - 1, -1, -1):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + float(gamma) * next_value * nonterminal - values[t]
        next_advantage = delta + float(gamma) * float(lam) * nonterminal * next_advantage
        advantages[t] = next_advantage
        next_value = values[t]
    returns = advantages + values
    return advantages.astype(np.float32), returns.astype(np.float32)


class RunningRewardScale:
    def __init__(self, decay: float = 0.999, epsilon: float = 1e-8) -> None:
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.mean = 0.0
        self.second = 1.0
        self.count = 0

    def update_and_normalize(
        self,
        values: np.ndarray,
        sample_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        array = np.asarray(values, dtype=np.float32)
        mask = (
            np.ones(array.shape, dtype=bool)
            if sample_mask is None
            else np.asarray(sample_mask, dtype=bool)
        )
        if mask.shape != array.shape:
            raise ValueError("reward sample mask has wrong shape")
        for value in array[mask].reshape(-1):
            self.count += 1
            self.mean = self.decay * self.mean + (1.0 - self.decay) * float(value)
            self.second = self.decay * self.second + (1.0 - self.decay) * float(value * value)
        variance = max(self.second - self.mean * self.mean, self.epsilon)
        normalized = (array / sqrt(variance)).astype(np.float32)
        return np.where(mask, normalized, 0.0).astype(np.float32)


class FiveAdam:
    """Adam with Five's published per-parameter gradient clipping rule.

    The second moment is updated from the *unclipped* gradient. The gradient
    used by the first moment/update is clipped to +/-5*sqrt(v_hat).
    """

    def __init__(self, parameters: Iterable[Tensor], *, lr: float, beta1: float, beta2: float, eps: float = 1e-8) -> None:
        self.parameters = [p for p in parameters if p.requires_grad]
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.state: dict[int, dict[str, Tensor | int]] = {}

    def __setstate__(self, state: object) -> None:
        """Restore a checkpoint without leaving Tensor keys in ``state``.

        Older checkpoints serialized ``torch.nn.Parameter`` objects as keys
        in the optimizer state.  Waiting until the first optimizer step to
        migrate those keys still leaves a window where a loader, snapshot
        helper, or rollback path can perform a normal dict lookup and trigger
        Tensor equality (``Boolean value of Tensor ... is ambiguous``).
        Canonicalize immediately after unpickling so every subsequent caller
        sees the identity-keyed representation.
        """
        if not isinstance(state, dict):
            raise TypeError("FiveAdam checkpoint state must be a dictionary")
        self.__dict__.update(state)
        self._canonicalize_state_keys()

    def _canonicalize_state_keys(self) -> int:
        """Convert legacy Tensor/Parameter state keys to integer identities.

        Early checkpoints could pickle ``torch.nn.Parameter`` objects as the
        keys of ``state``.  The current optimizer uses ``id(parameter)``.  A
        normal ``setdefault(id(parameter), ...)`` is unsafe while a legacy
        key is still present: Python may compare that Parameter with the
        integer key, and a multi-element Tensor cannot be coerced to bool.
        Resolve every key by type/identity only, then rebuild the same mapping
        in place so callers holding ``optimizer.state`` keep seeing it.

        Integer keys take precedence when a mixed legacy/current checkpoint
        contains both representations for the same object.  No Tensor
        equality is evaluated anywhere in this migration path.
        """
        state = self.state
        if not isinstance(state, dict) or not state:
            return 0

        known_ids = {id(p) for p in self.parameters} if hasattr(self, "parameters") else set()
        canonical: dict[int, tuple[int, object]] = {}
        changed = 0
        for key, value in list(state.items()):
            if isinstance(key, (int, np.integer)):
                canonical_key = int(key)
                if known_ids and canonical_key not in known_ids:
                    # Integer address from a prior process/unpickled checkpoint that does not
                    # match any active parameter in self.parameters.
                    changed += 1
                    continue
                priority = 2
                already_canonical = isinstance(key, int)
            elif isinstance(key, Tensor):
                canonical_key = id(key)
                priority = 1
                already_canonical = False
            else:
                # FiveAdam state is parameter-indexed.  Preserve unknown
                # object-keyed legacy entries by identity rather than asking
                # Python to compare them with a Tensor-like key later.
                canonical_key = id(key)
                priority = 1
                already_canonical = False

            previous = canonical.get(canonical_key)
            if previous is None or priority > previous[0]:
                canonical[canonical_key] = (priority, value)
            if not already_canonical:
                changed += 1

        if changed or len(canonical) != len(state):
            state.clear()
            state.update({key: value for key, (_priority, value) in canonical.items()})
        return changed + max(0, len(state) - len(canonical))

    def zero_grad(self) -> None:
        for parameter in self.parameters:
            parameter.grad = None

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def ensure_parameters(self, parameters: Iterable[Tensor]) -> int:
        """Attach parameters introduced by a neutral checkpoint migration."""
        self._canonicalize_state_keys()
        known = {id(parameter) for parameter in self.parameters}
        added = 0
        for parameter in parameters:
            if parameter.requires_grad and id(parameter) not in known:
                self.parameters.append(parameter)
                known.add(id(parameter))
                added += 1
        return added

    @torch.no_grad()
    def step(self) -> None:
        # Normalize legacy object keys before the first ``setdefault``.  This
        # is deliberately inside the optimizer as well as the Codex-specific
        # snapshot path: generic PPO/teacher-imitation updates can encounter
        # the same pre-identity checkpoint state.
        self._canonicalize_state_keys()
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach()
            key = id(parameter)
            state = self.state.get(key)
            if (
                state is None
                or not isinstance(state, dict)
                or not isinstance(state.get("m"), Tensor)
                or state["m"].shape != parameter.shape
                or not isinstance(state.get("v"), Tensor)
                or state["v"].shape != parameter.shape
            ):
                state = {
                    "step": 0,
                    "m": torch.zeros_like(parameter),
                    "v": torch.zeros_like(parameter),
                }
                self.state[key] = state
            step = int(state["step"]) + 1
            m = state["m"]
            v = state["v"]
            assert isinstance(m, Tensor) and isinstance(v, Tensor)
            # v sees the unclipped gradient, as published.
            v.mul_(self.beta2).addcmul_(gradient, gradient, value=1.0 - self.beta2)
            v_hat = v / (1.0 - self.beta2 ** step)
            bound = 5.0 * torch.sqrt(v_hat.clamp_min(0.0))
            clipped = torch.maximum(torch.minimum(gradient, bound), -bound)
            m.mul_(self.beta1).add_(clipped, alpha=1.0 - self.beta1)
            m_hat = m / (1.0 - self.beta1 ** step)
            parameter.addcdiv_(m_hat, torch.sqrt(v_hat).add_(self.eps), value=-self.lr)
            state["step"] = step

from .elastic_weight_consolidation import (
    DEFAULT_PROTECTED_PATTERNS,
    ElasticWeightConsolidation,
)
from .representation_regularizer import (
    RepresentationRegularizer,
)



class FivePPOTrainer:
    """PPO/GAE trainer using the public Five optimization constants."""

    def __init__(
        self,
        policy: SharedFiveTradingPolicy,
        *,
        schedule: FiveTrainingSchedule | None = None,
        value_loss_weight: float = 1.0,
        device: torch.device | None = None,
    ) -> None:
        self.policy = policy
        self.schedule = schedule or FiveTrainingSchedule()
        self.profile = self.schedule.profile
        self.value_loss_weight = float(value_loss_weight)
        self.device = device or next(policy.parameters()).device
        self.reward_scale = RunningRewardScale()
        self.pattern_reward_scale = RunningRewardScale()
        self.pending_timeout_reward_scale = RunningRewardScale()
        self.optimizer = FiveAdam(
            self.policy.parameters(),
            lr=self.schedule.learning_rate,
            beta1=self.profile.adam_beta1,
            beta2=self.profile.adam_beta2,
        )
        self.teacher_imitation_enabled = False
        self.teacher_geometry_imitation_enabled = False
        self.teacher_imitation_freeze_updates = 0
        self.teacher_imitation_anchor_updates = 0
        self.teacher_imitation_anchor_weight = 0.0
        self.teacher_imitation_disable_flat_entry_floor_during_warmup = True
        self.teacher_imitation_optimizer_updates = 0
        self.teacher_imitation_anchor: dict[str, Tensor] = {}
        self.last_teacher_imitation_stats: dict[str, float] = {}
        self.ewc: ElasticWeightConsolidation | None = None
        self.regularizer: RepresentationRegularizer | None = None

    def attach_representation_regularizer(
        self,
        regularizer: RepresentationRegularizer | None = None,
        *,
        hidden_dim: int = 4096,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        rank_weight: float = 0.05,
        arc_weight: float = 0.20,
    ) -> RepresentationRegularizer:
        """Attach or initialize RepresentationRegularizer."""
        if regularizer is not None:
            self.regularizer = regularizer
        elif self.regularizer is None:
            self.regularizer = RepresentationRegularizer(
                hidden_dim=hidden_dim,
                var_weight=var_weight,
                cov_weight=cov_weight,
                rank_weight=rank_weight,
                arc_weight=arc_weight,
            ).to(device=self.device)
        self.optimizer.ensure_parameters(self.regularizer.parameters())
        return self.regularizer

    def attach_ewc(
        self,
        ewc: ElasticWeightConsolidation | None = None,
        lambda_ewc: float = 250.0,
    ) -> ElasticWeightConsolidation:
        """Attach or initialize ElasticWeightConsolidation manager."""
        if ewc is not None:
            self.ewc = ewc
        elif self.ewc is None:
            self.ewc = ElasticWeightConsolidation(self.policy, lambda_ewc=lambda_ewc)
        return self.ewc

    def _teacher_retained_parameters(self) -> dict[str, Tensor]:
        retained: dict[str, Tensor] = {}
        calibrator = getattr(self.policy.heads, "entry_role_calibrator", None)
        seven_choice = bool(getattr(
            self.policy.heads, "seven_choice_imitation_trained", False
        ))
        if seven_choice:
            retained.update({
                f"seven_choice.action_query.{name}": value
                for name, value in self.policy.heads.action_query.named_parameters()
            })
            retained.update({
                f"seven_choice.action_embedding.{name}": value
                for name, value in self.policy.heads.action_embedding.named_parameters()
            })
            direct = getattr(self.policy.heads, "direct_action_head", None)
            if isinstance(direct, nn.Module):
                retained.update({
                    f"seven_choice.direct_action_head.{name}": value
                    for name, value in direct.named_parameters()
                })
            direct_observation = getattr(
                self.policy.heads, "direct_observation_head", None
            )
            if isinstance(direct_observation, nn.Module):
                retained.update({
                    f"seven_choice.direct_observation_head.{name}": value
                    for name, value in direct_observation.named_parameters()
                })
        if isinstance(calibrator, EntryRoleCalibrator):
            retained.update({f"action.{name}": value for name, value in calibrator.named_parameters()})
        if bool(getattr(self, "teacher_geometry_imitation_enabled", False)) or seven_choice:
            geometry_heads = (
                ("delay", "entry_offset", "sl", "tp", "trailing")
                if seven_choice else ("delay", "sl", "tp", "trailing")
            )
            for head_name in geometry_heads:
                head = self.policy.heads.parameter_heads[head_name]
                retained.update({
                    f"geometry.{head_name}.{name}": value
                    for name, value in head.named_parameters()
                })
        if seven_choice:
            retained.update({
                f"seven_choice.action_path.{name}": value
                for name, value in self.policy.observation_encoder.pattern_tree.action_path_head.named_parameters()
            })
            residual = getattr(
                self.policy.observation_encoder.pattern_tree,
                "action_path_residual", None,
            )
            if isinstance(residual, nn.Module):
                retained.update({
                    f"seven_choice.action_path_residual.{name}": value
                    for name, value in residual.named_parameters()
                })
            observation_adapter = getattr(
                self.policy.observation_encoder.pattern_tree,
                "action_path_observation_adapter", None,
            )
            if isinstance(observation_adapter, nn.Module):
                retained.update({
                    f"seven_choice.action_path_observation_adapter.{name}": value
                    for name, value in observation_adapter.named_parameters()
                })
        return retained

    def configure_teacher_imitation_retention(
        self,
        *,
        freeze_updates: int,
        anchor_updates: int,
        anchor_weight: float,
        disable_flat_entry_floor_during_warmup: bool = True,
    ) -> None:
        calibrator = getattr(self.policy.heads, "entry_role_calibrator", None)
        enabled = bool(
            isinstance(calibrator, EntryRoleCalibrator)
            and getattr(calibrator, "teacher_imitation_trained", False)
            and str(getattr(calibrator, "teacher_entry_scope", "")) == "all_entry_mass_v2"
        )
        self.teacher_imitation_enabled = enabled
        self.teacher_geometry_imitation_enabled = bool(
            enabled and (
                getattr(self.policy.heads, "teacher_geometry_imitation_trained", False)
                or getattr(self.policy.heads, "seven_choice_imitation_trained", False)
            )
        )
        self.teacher_imitation_freeze_updates = max(0, int(freeze_updates))
        self.teacher_imitation_anchor_updates = max(
            self.teacher_imitation_freeze_updates, int(anchor_updates)
        )
        self.teacher_imitation_anchor_weight = max(0.0, float(anchor_weight))
        self.teacher_imitation_disable_flat_entry_floor_during_warmup = bool(
            disable_flat_entry_floor_during_warmup
        )
        if not enabled:
            self.teacher_imitation_anchor = {}
            return
        current = self._teacher_retained_parameters()
        stored = getattr(self, "teacher_imitation_anchor", {})
        if set(stored) != set(current) or any(
            tuple(stored[name].shape) != tuple(parameter.shape)
            for name, parameter in current.items()
        ):
            self.teacher_imitation_anchor = {
                name: parameter.detach().clone()
                for name, parameter in current.items()
            }
            self.teacher_imitation_optimizer_updates = 0

    @property
    def teacher_imitation_warmup_active(self) -> bool:
        return bool(
            self.teacher_imitation_enabled
            and self.teacher_imitation_optimizer_updates < self.teacher_imitation_freeze_updates
        )

    @property
    def teacher_imitation_anchor_active(self) -> bool:
        return bool(
            self.teacher_imitation_enabled
            and self.teacher_imitation_optimizer_updates < self.teacher_imitation_anchor_updates
            and self.teacher_imitation_anchor_weight > 0.0
        )

    @property
    def teacher_imitation_representation_frozen(self) -> bool:
        """Keep the recurrent teacher feature space stable during retention.

        The calibrator was fitted on frozen encoder/LSTM hidden states.  Letting
        those tensors drift while merely anchoring the calibrator preserves its
        weights but not its learned decision boundary.  The base representation
        and action-style surface therefore stay frozen for the whole retention
        interval; after the short warm-up the calibrator alone may adapt under
        its anchor penalty.
        """

        return bool(
            self.teacher_imitation_enabled
            and self.teacher_imitation_optimizer_updates < self.teacher_imitation_anchor_updates
        )

    def set_progress(self, progress: float) -> None:
        self.schedule.set_progress(progress)
        self.optimizer.set_lr(self.schedule.learning_rate)

    def set_step(self, step: int | float, total_steps: int | float) -> None:
        self.schedule.set_step(step, total_steps)
        self.optimizer.set_lr(self.schedule.learning_rate)

    def ensure_policy_parameters(self) -> int:
        return self.optimizer.ensure_parameters(self.policy.parameters())

    def _evaluate_failure_replay_case(self, case: Any) -> dict[str, Tensor]:
        """Re-forward one immutable failure anchor through the *current* policy.

        Failure replay is deliberately not PPO replay: no stale old log-prob or
        old advantage is used.  The stored observation/recurrent state is only
        a causal context anchor; gradients are computed from current weights.
        """
        observations = np.asarray(case.observations, dtype=np.float32)
        if observations.shape != (FIVE_AGENT_COUNT, INPUT_COUNT):
            raise ValueError("failure replay observations must have shape [5,137]")
        hidden = np.asarray(case.memories_hidden, dtype=np.float32)
        cell = np.asarray(case.memories_cell, dtype=np.float32)
        previous = np.asarray(case.previous_actions, dtype=np.int64).reshape(-1)
        hidden_size = int(getattr(self.policy, "hidden_size", FIVE_LSTM_SIZE))
        if hidden.shape != (FIVE_AGENT_COUNT, hidden_size):
            raise ValueError("failure replay hidden state has wrong shape")
        if cell.shape != (FIVE_AGENT_COUNT, hidden_size):
            raise ValueError("failure replay cell state has wrong shape")
        if previous.size != FIVE_AGENT_COUNT:
            raise ValueError("failure replay previous actions have wrong shape")
        temporal_tokens_raw = getattr(case, "memories_temporal_tokens", None)
        temporal_valid_raw = getattr(case, "memories_temporal_valid", None)
        temporal_tokens = None
        temporal_valid = None
        if temporal_tokens_raw is not None or temporal_valid_raw is not None:
            if temporal_tokens_raw is None or temporal_valid_raw is None:
                raise ValueError("failure replay temporal state must be complete")
            temporal_tokens = np.asarray(temporal_tokens_raw, dtype=np.float32)
            temporal_valid = np.asarray(temporal_valid_raw, dtype=bool)
            expected_tokens = (
                FIVE_AGENT_COUNT,
                FIVE_TEMPORAL_TOKEN_COUNT,
                int(self.policy.post_lstm_transformer.model_width),
            )
            if temporal_tokens.shape != expected_tokens:
                raise ValueError("failure replay temporal tokens have wrong shape")
            if temporal_valid.shape != expected_tokens[:-1]:
                raise ValueError("failure replay temporal validity has wrong shape")
        memories = [
            FiveAgentMemory(
                hidden=torch.as_tensor(hidden[index], dtype=torch.float32, device=self.device),
                cell=torch.as_tensor(cell[index], dtype=torch.float32, device=self.device),
                previous_action=torch.as_tensor(
                    int(previous[index]), dtype=torch.long, device=self.device
                ),
                temporal_tokens=(
                    None if temporal_tokens is None else torch.as_tensor(
                        temporal_tokens[index], dtype=torch.float32, device=self.device
                    )
                ),
                temporal_valid=(
                    None if temporal_valid is None else torch.as_tensor(
                        temporal_valid[index], dtype=torch.bool, device=self.device
                    )
                ),
            )
            for index in range(FIVE_AGENT_COUNT)
        ]
        def choice(name: str) -> Tensor:
            values = np.asarray(getattr(case, name), dtype=np.int64).reshape(1, FIVE_AGENT_COUNT)
            return torch.as_tensor(values, dtype=torch.long, device=self.device)

        masks = np.asarray(case.allowed_action_masks, dtype=bool)
        if masks.shape != (FIVE_AGENT_COUNT, ACTION_COUNT):
            raise ValueError("failure replay action mask has wrong shape")
        return self.policy.evaluate_sequence(
            torch.as_tensor(
                observations[None, ...], dtype=torch.float32, device=self.device
            ),
            memories,
            actions=choice("actions"),
            targets=choice("targets"),
            delay_indices=choice("delay_indices"),
            risk_indices=choice("risk_indices"),
            entry_offset_indices=choice("entry_offset_indices"),
            sl_indices=choice("sl_indices"),
            tp_indices=choice("tp_indices"),
            trailing_indices=choice("trailing_indices"),
            manage_mode_indices=choice("manage_mode_indices"),
            close_fraction_indices=choice("close_fraction_indices"),
            allowed_action_masks=torch.as_tensor(
                masks[None, ...], dtype=torch.bool, device=self.device
            ),
            episode_dones=torch.ones(
                (1, FIVE_AGENT_COUNT), dtype=torch.bool, device=self.device
            ),
            # Repair is judged on the neural policy itself, not on the
            # exploration mixture that may deliberately perturb execution.
            behavior_uniform_mixes=torch.zeros(
                (1, FIVE_AGENT_COUNT), dtype=torch.float32, device=self.device
            ),
            behavior_entry_probability_floor=0.0,
        )

    def evaluate_failure_replay_probabilities(
        self, cases: Sequence[Any],
    ) -> dict[str, float]:
        """Return current probability of repeating each stored failed action."""
        probabilities: dict[str, float] = {}
        with torch.no_grad():
            for case in cases:
                agent = int(getattr(case, "failed_agent_index", -1))
                action = int(getattr(case, "failed_action_index", -1))
                if not (0 <= agent < FIVE_AGENT_COUNT and 0 <= action < ACTION_COUNT):
                    continue
                evaluated = self._evaluate_failure_replay_case(case)
                current = evaluated["policy_action_probabilities"][0, agent, action]
                probabilities[str(getattr(case, "case_id", ""))] = float(
                    current.detach().cpu().item()
                )
        return probabilities

    def failure_replay_loss(
        self, cases: Sequence[Any],
    ) -> tuple[Tensor, dict[str, object]]:
        """Suppress only a still-repeated failed action on old causal contexts.

        The loss is ``-log(1-p_failed)`` under the current policy.  It has no
        stale PPO ratio and does not prescribe HOLD: probability can move to
        any other action that is legal in the stored context.
        """
        zero = next(self.policy.parameters()).sum() * 0.0
        losses: list[Tensor] = []
        probabilities: list[float] = []
        for case in cases:
            agent = int(getattr(case, "failed_agent_index", -1))
            action = int(getattr(case, "failed_action_index", -1))
            if not (0 <= agent < FIVE_AGENT_COUNT and 0 <= action < ACTION_COUNT):
                continue
            evaluated = self._evaluate_failure_replay_case(case)
            probability = evaluated["policy_action_probabilities"][0, agent, action]
            weight = float(np.clip(
                getattr(case, "training_weight", lambda: 1.0)(), 0.25, 2.0
            ))
            losses.append(
                -torch.log1p(-probability.clamp(max=1.0 - 1e-6)) * weight
            )
            probabilities.append(float(probability.detach().cpu().item()))
        if not losses:
            return zero, {
                "failure_replay_loss": 0.0,
                "failure_replay_samples": 0.0,
                "failure_replay_failed_action_probability_mean": 0.0,
                "failure_replay_failed_action_probability_max": 0.0,
            }
        loss = torch.stack(losses).mean()
        return loss, {
            "failure_replay_loss": float(loss.detach().cpu().item()),
            "failure_replay_samples": float(len(losses)),
            "failure_replay_failed_action_probability_mean": float(np.mean(probabilities)),
            "failure_replay_failed_action_probability_max": float(np.max(probabilities)),
        }

    @staticmethod
    def initial_sl_conditional_entry_loss(
        raw_action_logits: Tensor,
        actions: Tensor,
        masks: Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, int]:
        """Penalize failed entry actions by transferring probability mass directly to HOLD (action 0)."""
        if raw_action_logits.ndim != 3 or raw_action_logits.shape[-1] != ACTION_COUNT:
            raise ValueError("raw_action_logits phai co shape [B,5,12]")
        if actions.shape != raw_action_logits.shape[:2] or masks.shape != actions.shape:
            raise ValueError("actions/masks phai co shape [B,5]")
        if weights.shape != raw_action_logits.shape[:1]:
            raise ValueError("weights phai co shape [B]")
        chosen_entry = actions.to(dtype=torch.long)
        eligible = masks.to(dtype=torch.bool) & (chosen_entry >= 1) & (chosen_entry <= 6)
        if not bool(torch.any(eligible)):
            return raw_action_logits.sum() * 0.0, 0
        hold_logits = raw_action_logits[..., 0]
        chosen_logits = raw_action_logits.gather(
            2, chosen_entry.clamp(0, ACTION_COUNT - 1).unsqueeze(-1)
        ).squeeze(-1)
        per_sample = torch.nn.functional.softplus(chosen_logits - hold_logits)
        expanded_weight = weights.reshape(-1, 1).expand_as(per_sample)
        return (per_sample[eligible] * expanded_weight[eligible]).mean(), int(
            eligible.sum().detach().cpu().item()
        )

    @staticmethod
    def clipped_surrogate(
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantage: Tensor,
        clip: float,
    ) -> Tensor:
        ratio = torch.exp(new_log_prob - old_log_prob)
        unclipped = ratio * advantage
        clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
        return torch.minimum(unclipped, clipped)

    def rlhf_ppo_loss_from_logits(
        self,
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
        config: object | None = None,
    ) -> tuple[Tensor, dict[str, float]]:
        """Bridge the candidate RLHF PPO objective into the existing trainer.

        The bridge intentionally performs no optimizer step.  Callers retain
        FiveAdam as the sole writer, just like the existing PPO paths; the
        reward model and reference policy must be frozen by the caller.
        """

        from .rlhf_ppo import RLHFPPOConfig, rlhf_ppo_loss_from_logits

        if config is None:
            resolved_config = RLHFPPOConfig()
        elif isinstance(config, RLHFPPOConfig):
            resolved_config = config
        else:
            raise TypeError("RLHF PPO config must be RLHFPPOConfig or None")
        return rlhf_ppo_loss_from_logits(
            new_logits=new_logits,
            old_logits=old_logits,
            reference_logits=reference_logits,
            actions=actions,
            advantages=advantages,
            values=values,
            returns=returns,
            new_values=new_values,
            allowed_action_mask=allowed_action_mask,
            sample_mask=sample_mask,
            config=resolved_config,
        )

    def optimize_rlhf_ppo_from_logits(
        self,
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
        reward_model_sha256: str,
        reference_policy_sha256: str,
        reward_model_frozen: bool = True,
        reference_policy_frozen: bool = True,
        sample_mask: Tensor | None = None,
        config: object | None = None,
        auxiliary_loss: Tensor | None = None,
        auxiliary_stats: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Run one guarded candidate-only RLHF PPO update.

        The caller must provide the hashes from the frozen reward model and
        reference checkpoint.  This guard prevents an untracked scorer or a
        moving reference from silently becoming an RLHF training run.
        """

        if not str(reward_model_sha256 or "").strip():
            raise ValueError("RLHF PPO requires a reward model hash")
        if not str(reference_policy_sha256 or "").strip():
            raise ValueError("RLHF PPO requires a reference policy hash")
        if not reward_model_frozen:
            raise ValueError("RLHF reward model must be frozen during policy update")
        if not reference_policy_frozen:
            raise ValueError("RLHF reference policy must be frozen during policy update")
        self.ensure_policy_parameters()
        loss, stats = self.rlhf_ppo_loss_from_logits(
            new_logits=new_logits,
            old_logits=old_logits,
            reference_logits=reference_logits,
            actions=actions,
            advantages=advantages,
            values=values,
            returns=returns,
            new_values=new_values,
            allowed_action_mask=allowed_action_mask,
            sample_mask=sample_mask,
            config=config,
        )
        if auxiliary_loss is not None:
            if not isinstance(auxiliary_loss, Tensor):
                raise TypeError("RLHF auxiliary_loss must be a torch Tensor")
            if auxiliary_loss.ndim != 0 or not bool(torch.isfinite(auxiliary_loss).item()):
                raise ValueError("RLHF auxiliary_loss must be one finite scalar")
            if auxiliary_loss.device != loss.device:
                raise ValueError("RLHF auxiliary_loss must use the PPO loss device")
            loss = loss + auxiliary_loss
            stats["rlhf_auxiliary_loss"] = float(auxiliary_loss.detach().cpu().item())
        optimizer_stats = self.step_from_loss(loss)
        result: dict[str, object] = dict(stats)
        result.update(optimizer_stats)
        if auxiliary_stats:
            result.update(dict(auxiliary_stats))
        result.update({
            "rlhf_ppo_update": True,
            "rlhf_reward_model_sha256": str(reward_model_sha256),
            "rlhf_reference_policy_sha256": str(reference_policy_sha256),
            "rlhf_reward_model_frozen": True,
            "rlhf_reference_policy_frozen": True,
            "rlhf_policy_optimizer_writer": "FiveAdam",
            "evidence_class": "rlhf_policy_candidate_only",
            "promotion_allowed": False,
        })
        return result

    def loss_from_tensors(
        self,
        *,
        new_log_prob: Tensor,
        old_log_prob: Tensor,
        advantage: Tensor,
        value: Tensor,
        returns: Tensor,
        entropy: Tensor,
        policy_loss_mask: Tensor | None = None,
        sample_loss_mask: Tensor | None = None,
        value_loss_weight_override: float | None = None,
        center_advantages: bool = True,
    ) -> tuple[Tensor, dict[str, float]]:
        sample_mask = (
            torch.ones_like(advantage, dtype=torch.bool)
            if sample_loss_mask is None
            else sample_loss_mask.to(device=advantage.device, dtype=torch.bool)
        )
        if sample_mask.shape != advantage.shape:
            raise ValueError("sample_loss_mask has wrong shape")
        actor_mask = (
            torch.ones_like(advantage, dtype=torch.bool)
            if policy_loss_mask is None
            else policy_loss_mask.to(device=advantage.device, dtype=torch.bool)
        )
        if actor_mask.shape != advantage.shape:
            raise ValueError("policy_loss_mask has wrong shape")
        actor_mask = actor_mask & sample_mask
        eligible_advantage = advantage[actor_mask]
        normalized_advantage = torch.zeros_like(advantage)
        if eligible_advantage.numel() > 1 and center_advantages:
            normalized_advantage[actor_mask] = (
                (eligible_advantage - eligible_advantage.mean())
                / (eligible_advantage.std(unbiased=False) + 1e-8)
            )
        elif eligible_advantage.numel() >= 1:
            normalized_advantage[actor_mask] = eligible_advantage
        surrogate = self.clipped_surrogate(
            new_log_prob, old_log_prob, normalized_advantage, self.profile.ppo_clip
        )
        policy_loss = (
            -surrogate[actor_mask].mean()
            if bool(torch.any(actor_mask))
            else surrogate.sum() * 0.0
        )
        value_loss = (
            0.5 * F.mse_loss(value[sample_mask], returns[sample_mask])
            if bool(torch.any(sample_mask))
            else value.sum() * 0.0
        )
        entropy_bonus = (
            entropy[sample_mask].mean()
            if bool(torch.any(sample_mask))
            else entropy.sum() * 0.0
        )
        value_weight = (
            self.value_loss_weight
            if value_loss_weight_override is None
            else float(value_loss_weight_override)
        )
        total = (
            policy_loss
            + value_weight * value_loss
            - self.schedule.entropy_coefficient * entropy_bonus
        )
        return total, {
            "policy_loss": float(policy_loss.detach().cpu().item()),
            "value_loss": float(value_loss.detach().cpu().item()),
            "entropy": float(entropy_bonus.detach().cpu().item()),
            "total_loss": float(total.detach().cpu().item()),
            "actor_eligible_fraction": float(actor_mask.float().mean().detach().cpu().item()),
            "actor_eligible_samples": float(actor_mask.sum().detach().cpu().item()),
            "reward_sample_eligible_fraction": float(sample_mask.float().mean().detach().cpu().item()),
            "reward_sample_eligible_samples": float(sample_mask.sum().detach().cpu().item()),
            "advantage_centered": float(bool(center_advantages)),
            "raw_actor_advantage_mean": float(
                eligible_advantage.mean().detach().cpu().item()
                if eligible_advantage.numel() else 0.0
            ),
            "normalized_actor_advantage_mean": float(
                normalized_advantage[actor_mask].mean().detach().cpu().item()
                if bool(torch.any(actor_mask)) else 0.0
            ),
        }

    @staticmethod
    def _flat_entry_mass_regularizers(
        raw_entry_mass: Tensor,
        flat_account_mask: Tensor,
        entry_floor: float,
    ) -> tuple[Tensor, Tensor]:
        """Return the legacy square loss and optional log-barrier loss.

        The square term is useful near the floor but its gradient through a
        softmax becomes negligible when HOLD has already saturated. The log
        gap is only used by opt-in candidate profiles and is computed on the
        same flat-account rows, so it cannot create an entry target for an
        occupied account.
        """
        zero = raw_entry_mass.sum() * 0.0
        if not bool(torch.any(flat_account_mask)) or float(entry_floor) <= 0.0:
            return zero, zero
        flat_mass = raw_entry_mass[flat_account_mask]
        floor = torch.as_tensor(
            float(entry_floor), dtype=raw_entry_mass.dtype, device=raw_entry_mass.device
        )
        square_loss = torch.relu(floor - flat_mass).square().mean()
        log_gap = torch.relu(
            torch.log(floor.clamp_min(1e-8))
            - torch.log(flat_mass.clamp_min(1e-8))
        ).mean()
        return square_loss, log_gap

    @staticmethod
    def _internal_policy_state_diversity_loss(
        latent: Tensor,
        sample_mask: Tensor | None,
        *,
        variance_floor: float,
    ) -> tuple[Tensor, dict[str, float]]:
        """Keep the learned policy state from collapsing to one vector.

        This objective is intentionally label-free.  It only asks the causal
        representation to retain variance across observed contexts and to
        avoid copying the same latent dimension into every other dimension;
        reward/PPO still decides which latent differences are economically
        useful.  No strategy ids, fixed strategy names, or future labels are
        introduced.
        """

        if latent.ndim < 2 or latent.shape[-1] <= 0:
            zero = latent.sum() * 0.0
            return zero, {
                "internal_policy_state_samples": 0.0,
                "internal_policy_state_std_mean": 0.0,
                "internal_policy_state_diversity_loss": 0.0,
            }
        flat = latent.reshape(-1, latent.shape[-1])
        if sample_mask is not None:
            mask = sample_mask.to(device=latent.device, dtype=torch.bool).reshape(-1)
            if mask.numel() != flat.shape[0]:
                raise ValueError("internal policy state sample mask has wrong shape")
            flat = flat[mask]
        if flat.shape[0] < 2:
            zero = latent.sum() * 0.0
            return zero, {
                "internal_policy_state_samples": float(flat.shape[0]),
                "internal_policy_state_std_mean": 0.0,
                "internal_policy_state_diversity_loss": 0.0,
            }
        floor = float(max(0.0, variance_floor))
        centered = flat - flat.mean(dim=0, keepdim=True)
        std = torch.sqrt(centered.square().mean(dim=0) + 1e-8)
        variance_loss = torch.relu(
            torch.as_tensor(floor, dtype=latent.dtype, device=latent.device) - std
        ).mean()
        normalized = centered / std.clamp_min(max(floor, 1e-3))
        covariance = normalized.transpose(0, 1).matmul(normalized) / float(flat.shape[0])
        identity = torch.eye(
            covariance.shape[0], dtype=covariance.dtype, device=covariance.device
        )
        off_diagonal = covariance - identity
        off_diagonal.fill_diagonal_(0.0)
        decorrelation_loss = off_diagonal.square().mean()
        loss = variance_loss + 0.1 * decorrelation_loss
        return loss, {
            "internal_policy_state_samples": float(flat.shape[0]),
            "internal_policy_state_std_mean": float(std.mean().detach().cpu().item()),
            "internal_policy_state_diversity_loss": float(loss.detach().cpu().item()),
        }

    def reasoning_grpo_loss(
        self,
        *,
        hidden: Tensor,
        allowed_action_masks: Tensor,
        action_q_values_r: Tensor,
        anchor_mask: Tensor,
        group_size: int,
        reasoning_steps: int,
        ppo_clip: float,
        kl_beta: float,
        entropy_weight: float,
        loss_weight: float,
        encoder_credit_scale: float,
        min_group_std_r: float,
    ) -> tuple[Tensor, dict[str, float]]:
        """GRPO objective for multi-step trading hypotheses.

        Each causal state produces ``group_size`` hypothesis trajectories.  The
        immutable action-local counterfactual verifier scores only the final
        hypothesis; rewards are normalized inside that state/agent group.  The
        reference head is frozen and separate from PPO's rollout-old policy.
        """
        self.policy.require_current_architecture()
        zero = hidden.sum() * 0.0
        enabled = anchor_mask.to(device=hidden.device, dtype=torch.bool).reshape(-1)
        batch = int(hidden.shape[0])
        if hidden.ndim != 3 or hidden.shape[1] != FIVE_AGENT_COUNT:
            raise ValueError("reasoning hidden must have shape [B,5,H]")
        if enabled.numel() != batch:
            raise ValueError("reasoning anchor mask must have shape [B]")
        if allowed_action_masks.shape[:2] != hidden.shape[:2]:
            raise ValueError("reasoning allowed masks must have shape [B,5,A]")
        if action_q_values_r.shape != (batch, FIVE_REASONING_ACTION_COUNT):
            raise ValueError("reasoning verifier values must have shape [B,7]")
        if not bool(torch.any(enabled)) or float(loss_weight) <= 0.0:
            return zero, {
                "reasoning_rl_enabled": 1.0,
                "reasoning_group_eligible": 0.0,
                "reasoning_optimizer_eligible": 0.0,
                "reasoning_loss": 0.0,
                "reasoning_policy_loss": 0.0,
                "reasoning_kl": 0.0,
                "reasoning_entropy": 0.0,
                "reasoning_reward_mean_r": 0.0,
                "reasoning_reward_std_r": 0.0,
            }

        groups = max(2, int(group_size))
        steps = max(1, int(reasoning_steps))
        flat_hidden = hidden.reshape(batch * FIVE_AGENT_COUNT, hidden.shape[-1])
        scale = float(np.clip(encoder_credit_scale, 0.0, 1.0))
        flat_hidden = flat_hidden.detach() + scale * (
            flat_hidden - flat_hidden.detach()
        )
        legal = allowed_action_masks[
            ..., :FIVE_REASONING_ACTION_COUNT
        ].to(device=hidden.device, dtype=torch.bool).reshape(
            batch * FIVE_AGENT_COUNT, FIVE_REASONING_ACTION_COUNT
        )
        hold = torch.zeros_like(legal)
        hold[:, 0] = True
        legal = torch.where(legal.any(dim=-1, keepdim=True), legal, hold)
        expanded_hidden = flat_hidden.unsqueeze(1).expand(
            -1, groups, -1
        ).reshape(-1, flat_hidden.shape[-1])
        expanded_legal = legal.unsqueeze(1).expand(
            -1, groups, -1
        ).reshape(-1, FIVE_REASONING_ACTION_COUNT)

        with torch.no_grad():
            hypotheses, old_log_probability, _ = self.policy.reasoning_head.sample(
                expanded_hidden,
                expanded_legal,
                steps=steps,
            )
        new_log_probability, entropy, current_logits = (
            self.policy.reasoning_head.evaluate(
                expanded_hidden, expanded_legal, hypotheses
            )
        )
        with torch.no_grad():
            _, _, reference_logits = self.policy.reasoning_reference_head.evaluate(
                expanded_hidden.detach(), expanded_legal, hypotheses
            )

        kl_steps: list[Tensor] = []
        for current, reference in zip(current_logits, reference_logits):
            current_log = torch.log_softmax(current, dim=-1)
            reference_log = torch.log_softmax(reference, dim=-1)
            current_probability = torch.softmax(current, dim=-1)
            kl_steps.append(
                torch.sum(current_probability * (current_log - reference_log), dim=-1)
            )
        kl = torch.stack(kl_steps, dim=-1).mean(dim=-1)

        final_hypothesis = hypotheses[:, -1].reshape(
            batch, FIVE_AGENT_COUNT, groups
        )
        verifier = action_q_values_r.to(
            device=hidden.device, dtype=torch.float32
        ).unsqueeze(1).expand(-1, FIVE_AGENT_COUNT, -1)
        verifier = verifier.unsqueeze(2).expand(-1, -1, groups, -1)
        rewards = verifier.gather(
            -1, final_hypothesis.unsqueeze(-1)
        ).squeeze(-1)
        reward_mean = rewards.mean(dim=-1, keepdim=True)
        reward_std = rewards.std(dim=-1, unbiased=False, keepdim=True)
        valid_group = (
            enabled.unsqueeze(-1).expand(-1, FIVE_AGENT_COUNT)
            & (reward_std.squeeze(-1) >= float(max(0.0, min_group_std_r)))
        )
        advantage = (rewards - reward_mean) / reward_std.clamp_min(1e-6)
        advantage = advantage.reshape(-1)
        valid_sample = valid_group.unsqueeze(-1).expand(
            -1, -1, groups
        ).reshape(-1)
        ratio = torch.exp(new_log_probability - old_log_probability.detach())
        surrogate = torch.minimum(
            ratio * advantage,
            torch.clamp(
                ratio,
                1.0 - float(ppo_clip),
                1.0 + float(ppo_clip),
            ) * advantage,
        )
        if bool(torch.any(valid_sample)):
            policy_loss = -surrogate[valid_sample].mean()
            kl_loss = kl[valid_sample].mean()
            entropy_bonus = entropy[valid_sample].mean()
            objective = (
                policy_loss
                + float(kl_beta) * kl_loss
                - float(entropy_weight) * entropy_bonus
            )
        else:
            policy_loss = zero
            kl_loss = zero
            entropy_bonus = zero
            objective = zero
        weighted = float(loss_weight) * objective
        eligible_rewards = rewards[valid_group] if bool(torch.any(valid_group)) else None
        return weighted, {
            "reasoning_rl_enabled": 1.0,
            "reasoning_group_size": float(groups),
            "reasoning_steps": float(steps),
            "reasoning_group_eligible": float(valid_group.sum().detach().cpu().item()),
            "reasoning_optimizer_eligible": float(valid_sample.sum().detach().cpu().item()),
            "reasoning_loss": float(weighted.detach().cpu().item()),
            "reasoning_policy_loss": float(policy_loss.detach().cpu().item()),
            "reasoning_kl": float(kl_loss.detach().cpu().item()),
            "reasoning_entropy": float(entropy_bonus.detach().cpu().item()),
            "reasoning_reward_mean_r": float(
                eligible_rewards.mean().detach().cpu().item()
                if eligible_rewards is not None else 0.0
            ),
            "reasoning_reward_std_r": float(
                eligible_rewards.std(unbiased=False).detach().cpu().item()
                if eligible_rewards is not None and eligible_rewards.numel() > 1 else 0.0
            ),
            "reasoning_reference_frozen": 1.0,
            "reasoning_encoder_credit_scale": scale,
        }

    def optimize_sequence(
        self,
        *,
        observations: np.ndarray,
        initial_memories: Sequence[FiveAgentMemory],
        actions: np.ndarray,
        targets: np.ndarray,
        delay_indices: np.ndarray,
        risk_indices: np.ndarray,
        entry_offset_indices: np.ndarray,
        manage_mode_indices: np.ndarray,
        close_fraction_indices: np.ndarray,
        sl_indices: np.ndarray | None = None,
        tp_indices: np.ndarray | None = None,
        trailing_indices: np.ndarray | None = None,
        old_log_probabilities: np.ndarray,
        rewards: np.ndarray,
        dones: np.ndarray,
        step_seconds: float,
        horizon_seconds: float,
        bootstrap_value: np.ndarray | float = 0.0,
        allowed_action_masks: np.ndarray | None = None,
        policy_loss_masks: np.ndarray | None = None,
        sample_loss_masks: np.ndarray | None = None,
        behavior_uniform_mixes: np.ndarray | None = None,
        flat_account_masks: np.ndarray | None = None,
    ) -> dict[str, float]:
        self.ensure_policy_parameters()
        observations = np.asarray(observations, dtype=np.float32)
        if observations.shape[0] > FIVE_LSTM_UNROLL:
            raise ValueError(f"toi da {FIVE_LSTM_UNROLL} buoc moi lan cap nhat")
        rewards = self.reward_scale.update_and_normalize(
            np.asarray(rewards, dtype=np.float32), sample_mask=sample_loss_masks
        )
        gamma = discount_from_horizon(step_seconds, horizon_seconds)
        obs_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        target_t = torch.as_tensor(targets, dtype=torch.long, device=self.device)
        delay_t = torch.as_tensor(delay_indices, dtype=torch.long, device=self.device)
        risk_t = torch.as_tensor(risk_indices, dtype=torch.long, device=self.device)
        offset_t = torch.as_tensor(entry_offset_indices, dtype=torch.long, device=self.device)
        shape = np.asarray(actions).shape
        sl_t = torch.as_tensor(
            np.full(shape, DEFAULT_SL_BIN_INDEX) if sl_indices is None else sl_indices,
            dtype=torch.long, device=self.device,
        )
        tp_t = torch.as_tensor(
            np.full(shape, DEFAULT_TP_BIN_INDEX) if tp_indices is None else tp_indices,
            dtype=torch.long, device=self.device,
        )
        trailing_t = torch.as_tensor(
            np.full(shape, DEFAULT_TRAILING_BIN_INDEX) if trailing_indices is None else trailing_indices,
            dtype=torch.long, device=self.device,
        )
        manage_t = torch.as_tensor(manage_mode_indices, dtype=torch.long, device=self.device)
        close_t = torch.as_tensor(close_fraction_indices, dtype=torch.long, device=self.device)
        allowed_t = None if allowed_action_masks is None else torch.as_tensor(
            allowed_action_masks, dtype=torch.bool, device=self.device
        )
        evaluated = self.policy.evaluate_sequence(
            obs_t,
            initial_memories,
            actions=action_t,
            targets=target_t,
            delay_indices=delay_t,
            risk_indices=risk_t,
            entry_offset_indices=offset_t,
            sl_indices=sl_t,
            tp_indices=tp_t,
            trailing_indices=trailing_t,
            manage_mode_indices=manage_t,
            close_fraction_indices=close_t,
            allowed_action_masks=allowed_t,
            episode_dones=torch.as_tensor(dones, dtype=torch.bool, device=self.device),
            behavior_uniform_mixes=(
                None if behavior_uniform_mixes is None else torch.as_tensor(
                    behavior_uniform_mixes, dtype=torch.float32, device=self.device
                )
            ),
            behavior_entry_probability_floor=self.profile.behavior_entry_probability_floor,
        )
        values_np = evaluated["value"].detach().cpu().numpy().astype(np.float32)
        advantages_np, returns_np = generalized_advantage_estimation(
            rewards,
            values_np,
            np.asarray(dones, dtype=np.float32),
            gamma=gamma,
            lam=self.profile.gae_lambda,
            bootstrap_value=bootstrap_value,
        )
        old_lp = torch.as_tensor(old_log_probabilities, dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(advantages_np, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns_np, dtype=torch.float32, device=self.device)
        policy_mask_t = None if policy_loss_masks is None else torch.as_tensor(
            policy_loss_masks, dtype=torch.bool, device=self.device
        )
        sample_mask_t = None if sample_loss_masks is None else torch.as_tensor(
            sample_loss_masks, dtype=torch.bool, device=self.device
        )
        loss, stats = self.loss_from_tensors(
            new_log_prob=evaluated["policy_log_probability"],
            old_log_prob=old_lp,
            advantage=advantages,
            value=evaluated["value"],
            returns=returns,
            # Diversity must be measured and optimized on the raw primary
            # action policy. Parameter heads can stay uncertain while the
            # action head has already collapsed to HOLD; summing their entropy
            # hid exactly that failure in the 20260815 MT5 run.
            entropy=evaluated["policy_action_entropy"],
            policy_loss_mask=policy_mask_t,
            sample_loss_mask=sample_mask_t,
        )
        internal_state_loss, internal_state_stats = self._internal_policy_state_diversity_loss(
            evaluated["internal_policy_state"],
            sample_mask_t,
            variance_floor=float(self.schedule.internal_policy_state_variance_floor),
        )
        loss = loss + float(self.schedule.internal_policy_state_diversity_weight) * internal_state_loss
        if getattr(self, "regularizer", None) is not None:
            targets_3class = self.regularizer.derive_targets(action_t)
            reg_dict = self.regularizer(
                evaluated["policy_hidden"],
                targets=targets_3class,
                sample_mask=sample_mask_t,
            )
            loss = loss + reg_dict["loss_reg_total"]
            stats["loss_var"] = float(reg_dict["loss_var"].detach().cpu().item())
            stats["loss_cov"] = float(reg_dict["loss_cov"].detach().cpu().item())
            stats["loss_rank"] = float(reg_dict["loss_rank"].detach().cpu().item())
            stats["loss_arcface"] = float(reg_dict["loss_arcface"].detach().cpu().item())
            stats["loss_reg_total"] = float(reg_dict["loss_reg_total"].detach().cpu().item())
            stats["effective_rank"] = float(reg_dict["effective_rank"])
            stats["inter_class_cos"] = float(reg_dict["inter_class_cos"])
        flat_mask_t = (
            torch.zeros_like(action_t, dtype=torch.bool)
            if flat_account_masks is None
            else torch.as_tensor(flat_account_masks, dtype=torch.bool, device=self.device)
        )
        if flat_mask_t.shape != action_t.shape:
            raise ValueError("flat_account_masks has wrong shape")
        raw_entry_mass = torch.softmax(
            evaluated["raw_action_logits"], dim=-1
        )[..., 1:7].sum(dim=-1)
        entry_floor = (
            0.0 if (
                self.teacher_imitation_warmup_active
                and self.teacher_imitation_disable_flat_entry_floor_during_warmup
            )
            else float(self.schedule.flat_entry_mass_floor)
        )
        flat_entry_mass_loss, flat_entry_mass_log_loss = self._flat_entry_mass_regularizers(
            raw_entry_mass, flat_mask_t, entry_floor
        )
        loss = loss + float(self.schedule.flat_entry_mass_loss_weight) * flat_entry_mass_loss
        loss = loss + float(self.schedule.flat_entry_mass_log_loss_weight) * flat_entry_mass_log_loss
        if entry_floor > 0.0 and bool(torch.any(flat_mask_t)):
            stats["total_loss"] = float(loss.detach().cpu().item())
        stats.update(self.step_from_loss(loss))
        stats.update({
            "gamma": float(gamma),
            "horizon_seconds": float(horizon_seconds),
            "learning_rate": float(self.schedule.learning_rate),
            "entropy_coefficient": float(self.schedule.entropy_coefficient),
            "entropy_edge_gate_enabled": float(self.schedule.entropy_edge_gate_enabled),
            "entropy_verified_edge_count": float(self.schedule.verified_edge_count),
            "entropy_edge_release_count": float(self.schedule.entropy_edge_release_count),
            "entropy_edge_decay_count": float(self.schedule.entropy_edge_decay_count),
            "team_spirit": float(self.schedule.team_spirit),
            "primary_action_entropy": float(
                evaluated["primary_action_entropy"].mean().detach().cpu().item()
            ),
            "policy_action_entropy": float(
                evaluated["policy_action_entropy"].mean().detach().cpu().item()
            ),
            "behavior_uniform_mix": float(
                0.0 if behavior_uniform_mixes is None else np.mean(behavior_uniform_mixes)
            ),
            "flat_entry_mass_floor": entry_floor,
            "flat_entry_mass_loss_weight": float(self.schedule.flat_entry_mass_loss_weight),
            "flat_entry_mass_loss": float(flat_entry_mass_loss.detach().cpu().item()),
            "flat_entry_mass_log_loss_weight": float(
                self.schedule.flat_entry_mass_log_loss_weight
            ),
            "flat_entry_mass_log_loss": float(
                flat_entry_mass_log_loss.detach().cpu().item()
            ),
            "flat_entry_mass_samples": float(flat_mask_t.sum().detach().cpu().item()),
            "flat_entry_mass_mean": float(
                raw_entry_mass[flat_mask_t].mean().detach().cpu().item()
                if bool(torch.any(flat_mask_t)) else 0.0
            ),
            "flat_entry_mass_regularizer_source": "raw_unmasked_action_logits",
            "primary_entry_probability_mass": float(
                evaluated["primary_entry_probability_mass"].mean().detach().cpu().item()
            ),
            "internal_policy_state_diversity_weight": float(
                self.schedule.internal_policy_state_diversity_weight
            ),
            **internal_state_stats,
        })
        return stats

    def optimize_independent_pattern_steps(
        self,
        *,
        observations: np.ndarray,
        initial_memories: Sequence[Sequence[FiveAgentMemory]],
        actions: np.ndarray,
        targets: np.ndarray,
        delay_indices: np.ndarray,
        risk_indices: np.ndarray,
        entry_offset_indices: np.ndarray,
        manage_mode_indices: np.ndarray,
        close_fraction_indices: np.ndarray,
        sl_indices: np.ndarray | None = None,
        tp_indices: np.ndarray | None = None,
        trailing_indices: np.ndarray | None = None,
        old_log_probabilities: np.ndarray,
        old_values: np.ndarray,
        rewards: np.ndarray,
        allowed_action_masks: np.ndarray,
        policy_loss_masks: np.ndarray,
        pattern_tree_enabled: np.ndarray | None = None,
        utility_targets: np.ndarray | None = None,
        excursion_targets: np.ndarray | None = None,
        no_edge_targets: np.ndarray | None = None,
        auxiliary_loss_weights: np.ndarray | None = None,
        utility_loss_weights: np.ndarray | None = None,
        excursion_loss_weights: np.ndarray | None = None,
        no_edge_loss_weights: np.ndarray | None = None,
        direction_classes: np.ndarray | None = None,
        direction_loss_weights: np.ndarray | None = None,
        edge_hold_ceilings: np.ndarray | None = None,
        hold_ceiling_loss_weights: np.ndarray | None = None,
        horizon_targets: np.ndarray | None = None,
        horizon_masks: np.ndarray | None = None,
        path_dynamics_targets: np.ndarray | None = None,
        first_hit_classes: np.ndarray | None = None,
        action_path_targets: np.ndarray | None = None,
        action_outcome_classes: np.ndarray | None = None,
        horizon_loss_weights: np.ndarray | None = None,
        path_dynamics_loss_weights: np.ndarray | None = None,
        first_hit_loss_weights: np.ndarray | None = None,
        action_path_loss_weights: np.ndarray | None = None,
        action_outcome_loss_weights: np.ndarray | None = None,
        ranking_best_actions: np.ndarray | None = None,
        ranking_margins: np.ndarray | None = None,
        ranking_loss_masks: np.ndarray | None = None,
        ranking_loss_weights: np.ndarray | None = None,
        initial_sl_local_loss_masks: np.ndarray | None = None,
        initial_sl_local_loss_weights: np.ndarray | None = None,
        reasoning_rl_masks: np.ndarray | None = None,
        reasoning_action_q_values_r: np.ndarray | None = None,
        reasoning_rl_enabled: bool = False,
        reasoning_group_size: int = 8,
        reasoning_steps: int = 3,
        reasoning_ppo_clip: float = 0.20,
        reasoning_kl_beta: float = 0.02,
        reasoning_entropy_weight: float = 0.001,
        reasoning_loss_weight: float = 0.25,
        reasoning_encoder_credit_scale: float = 0.0,
        reasoning_min_group_std_r: float = 0.02,
        reward_scale: RunningRewardScale | None = None,
        preserve_reward_sign: bool = False,
        behavior_uniform_mixes: np.ndarray | None = None,
        flat_account_masks: np.ndarray | None = None,
        failure_replay_cases: Sequence[Any] | None = None,
        failure_replay_weight: float = 0.0,
    ) -> dict[str, float]:
        """One actor-only PPO update over independent terminal anchor states."""
        self.ensure_policy_parameters()
        batch = int(np.asarray(observations).shape[0])
        if batch <= 0 or len(initial_memories) != batch:
            raise ValueError("independent pattern batch is empty or mismatched")
        choice_shape = np.asarray(actions).shape
        if sl_indices is None:
            sl_indices = np.full(choice_shape, DEFAULT_SL_BIN_INDEX, dtype=np.int64)
        if tp_indices is None:
            tp_indices = np.full(choice_shape, DEFAULT_TP_BIN_INDEX, dtype=np.int64)
        if trailing_indices is None:
            trailing_indices = np.full(choice_shape, DEFAULT_TRAILING_BIN_INDEX, dtype=np.int64)
        if behavior_uniform_mixes is None:
            behavior_uniform_mixes = np.zeros(choice_shape, dtype=np.float32)
        behavior_uniform_mixes = np.asarray(behavior_uniform_mixes, dtype=np.float32)
        if behavior_uniform_mixes.shape != choice_shape:
            raise ValueError("behavior_uniform_mixes phai co shape [B,5]")
        evaluated_rows: list[dict[str, Tensor]] = []
        for index in range(batch):
            evaluated_rows.append(self.policy.evaluate_sequence(
                torch.as_tensor(observations[index:index + 1], dtype=torch.float32, device=self.device),
                initial_memories[index],
                actions=torch.as_tensor(actions[index:index + 1], dtype=torch.long, device=self.device),
                targets=torch.as_tensor(targets[index:index + 1], dtype=torch.long, device=self.device),
                delay_indices=torch.as_tensor(delay_indices[index:index + 1], dtype=torch.long, device=self.device),
                risk_indices=torch.as_tensor(risk_indices[index:index + 1], dtype=torch.long, device=self.device),
                entry_offset_indices=torch.as_tensor(entry_offset_indices[index:index + 1], dtype=torch.long, device=self.device),
                sl_indices=torch.as_tensor(sl_indices[index:index + 1], dtype=torch.long, device=self.device),
                tp_indices=torch.as_tensor(tp_indices[index:index + 1], dtype=torch.long, device=self.device),
                trailing_indices=torch.as_tensor(trailing_indices[index:index + 1], dtype=torch.long, device=self.device),
                manage_mode_indices=torch.as_tensor(manage_mode_indices[index:index + 1], dtype=torch.long, device=self.device),
                close_fraction_indices=torch.as_tensor(close_fraction_indices[index:index + 1], dtype=torch.long, device=self.device),
                allowed_action_masks=torch.as_tensor(
                    allowed_action_masks[index:index + 1], dtype=torch.bool, device=self.device
                ),
                episode_dones=torch.ones((1, FIVE_AGENT_COUNT), dtype=torch.bool, device=self.device),
                behavior_uniform_mixes=torch.as_tensor(
                    behavior_uniform_mixes[index:index + 1],
                    dtype=torch.float32,
                    device=self.device,
                ),
            ))
        new_log_prob = torch.cat([row["policy_log_probability"] for row in evaluated_rows], dim=0)
        # Delayed path/lifecycle batches use the same raw-action entropy
        # contract as online PPO. Auxiliary heads cannot pay the diversity
        # budget on behalf of a collapsed action distribution.
        entropy = torch.cat([row["policy_action_entropy"] for row in evaluated_rows], dim=0)
        value = torch.cat([row["value"] for row in evaluated_rows], dim=0)
        normalized_rewards = (reward_scale or self.pattern_reward_scale).update_and_normalize(
            np.asarray(rewards, dtype=np.float32)
        )
        returns = torch.as_tensor(normalized_rewards, dtype=torch.float32, device=self.device)
        baseline = torch.as_tensor(old_values, dtype=torch.float32, device=self.device)
        advantage = returns if preserve_reward_sign else returns - baseline
        actor_loss, stats = self.loss_from_tensors(
            new_log_prob=new_log_prob,
            old_log_prob=torch.as_tensor(old_log_probabilities, dtype=torch.float32, device=self.device),
            advantage=advantage,
            value=value,
            returns=returns,
            entropy=entropy,
            policy_loss_mask=torch.as_tensor(policy_loss_masks, dtype=torch.bool, device=self.device),
            value_loss_weight_override=0.0,
            center_advantages=not preserve_reward_sign,
        )
        internal_policy_state = torch.cat(
            [row["internal_policy_state"] for row in evaluated_rows], dim=0
        )
        internal_state_loss, internal_state_stats = self._internal_policy_state_diversity_loss(
            internal_policy_state,
            torch.as_tensor(policy_loss_masks, dtype=torch.bool, device=self.device),
            variance_floor=float(self.schedule.internal_policy_state_variance_floor),
        )
        actor_loss = actor_loss + float(
            self.schedule.internal_policy_state_diversity_weight
        ) * internal_state_loss
        stats.update({
            "internal_policy_state_diversity_weight": float(
                self.schedule.internal_policy_state_diversity_weight
            ),
            **internal_state_stats,
        })
        enabled_np = np.zeros(batch, dtype=bool) if pattern_tree_enabled is None else np.asarray(
            pattern_tree_enabled, dtype=bool
        ).reshape(-1)
        if enabled_np.size != batch:
            raise ValueError("pattern_tree_enabled must have one value per anchor")
        if direction_classes is None:
            utility_np = np.asarray(utility_targets, dtype=np.float32)
            no_edge_np = np.asarray(no_edge_targets, dtype=np.float32).reshape(-1)
            direction_classes = np.where(
                no_edge_np >= 0.5,
                0,
                np.where(utility_np[:, 0] >= utility_np[:, 1], 1, 2),
            ).astype(np.int64)
        if direction_loss_weights is None:
            direction_loss_weights = np.zeros(batch, dtype=np.float32)
        if edge_hold_ceilings is None:
            edge_hold_ceilings = np.full(batch, 0.20, dtype=np.float32)
        if hold_ceiling_loss_weights is None:
            hold_ceiling_loss_weights = np.zeros(batch, dtype=np.float32)
        if horizon_targets is None:
            horizon_targets = np.zeros((batch, 5), dtype=np.float32)
        if horizon_masks is None:
            horizon_masks = np.zeros((batch, 5), dtype=np.float32)
        if path_dynamics_targets is None:
            path_dynamics_targets = np.zeros((batch, 10), dtype=np.float32)
        if first_hit_classes is None:
            first_hit_classes = np.zeros((batch, 2), dtype=np.int64)
        if action_path_targets is None:
            action_path_targets = np.zeros((batch, 6, 5), dtype=np.float32)
        if action_outcome_classes is None:
            action_outcome_classes = np.zeros((batch, 6), dtype=np.int64)
        if horizon_loss_weights is None:
            horizon_loss_weights = np.zeros(batch, dtype=np.float32)
        if path_dynamics_loss_weights is None:
            path_dynamics_loss_weights = np.zeros(batch, dtype=np.float32)
        if first_hit_loss_weights is None:
            first_hit_loss_weights = np.zeros(batch, dtype=np.float32)
        if action_path_loss_weights is None:
            action_path_loss_weights = np.zeros(batch, dtype=np.float32)
        if action_outcome_loss_weights is None:
            action_outcome_loss_weights = np.zeros(batch, dtype=np.float32)
        pattern_obs = torch.as_tensor(
            np.asarray(observations, dtype=np.float32).reshape(batch * FIVE_AGENT_COUNT, INPUT_COUNT),
            dtype=torch.float32,
            device=self.device,
        )
        predicted = self.policy.observation_encoder.pattern_predictions(pattern_obs)
        utility_pred = predicted["utility"].reshape(batch, FIVE_AGENT_COUNT, 2).mean(dim=1)
        excursion_pred = predicted["excursion"].reshape(batch, FIVE_AGENT_COUNT, 4).mean(dim=1)
        no_edge_logit = predicted["no_edge_logit"].reshape(batch, FIVE_AGENT_COUNT).mean(dim=1)
        horizon_pred = predicted["horizon_returns"].reshape(batch, FIVE_AGENT_COUNT, 5).mean(dim=1)
        path_dynamics_pred = predicted["path_dynamics"].reshape(batch, FIVE_AGENT_COUNT, 10).mean(dim=1)
        first_hit_logits = predicted["first_hit_logits"].reshape(batch, FIVE_AGENT_COUNT, 2, 3).mean(dim=1)
        action_path_pred = predicted["action_path"].reshape(batch, FIVE_AGENT_COUNT, 6, 5).mean(dim=1)
        action_outcome_logits = predicted["action_outcome_logits"].reshape(batch, FIVE_AGENT_COUNT, 6, 4).mean(dim=1)
        evaluated_hidden = torch.cat(
            [row["policy_hidden"] for row in evaluated_rows], dim=0
        )
        raw_agent_ids = torch.arange(
            FIVE_AGENT_COUNT, dtype=torch.long, device=self.device
        ).expand(evaluated_hidden.shape[:-1])
        action_mode = str(
            getattr(
                self.policy.heads,
                "seven_choice_action_decision_mode",
                "expected_value",
            )
        )
        raw_observations = torch.as_tensor(
            np.asarray(observations, dtype=np.float32).reshape(
                batch, FIVE_AGENT_COUNT, INPUT_COUNT
            ),
            dtype=torch.float32,
            device=self.device,
        )
        raw_action_logits = (
            self.policy.heads.direct_action_logits(
                evaluated_hidden, raw_agent_ids, raw_observations
            )
            if action_mode == "direct"
            else self.policy.heads.primary_action_logits(
                evaluated_hidden, raw_agent_ids
            )
        )
        raw_action_logits, internal_policy_state = self.policy.heads.apply_internal_policy_state(
            raw_action_logits, evaluated_hidden
        )
        flat_mask_t = (
            torch.zeros((batch, FIVE_AGENT_COUNT), dtype=torch.bool, device=self.device)
            if flat_account_masks is None
            else torch.as_tensor(flat_account_masks, dtype=torch.bool, device=self.device)
        )
        if flat_mask_t.shape != (batch, FIVE_AGENT_COUNT):
            raise ValueError("flat_account_masks phai co shape [B,5]")
        raw_entry_mass = torch.softmax(raw_action_logits, dim=-1)[..., 1:7].sum(dim=-1)
        raw_action_probabilities = torch.cat(
            [row["policy_action_probabilities"] for row in evaluated_rows], dim=0
        )
        entry_floor = (
            0.0 if (
                self.teacher_imitation_warmup_active
                and self.teacher_imitation_disable_flat_entry_floor_during_warmup
            )
            else float(self.schedule.flat_entry_mass_floor)
        )
        flat_entry_mass_loss, flat_entry_mass_log_loss = self._flat_entry_mass_regularizers(
            raw_entry_mass, flat_mask_t, entry_floor
        )
        direction_floor = float(self.schedule.flat_entry_direction_mass_floor)
        flat_entry_direction_mass_loss = raw_entry_mass.sum() * 0.0
        direction_flat_mask = flat_mask_t[:, :2]
        if bool(torch.any(direction_flat_mask)) and direction_floor > 0.0:
            owner_entry_mass = torch.stack((
                raw_action_probabilities[:, 0, list(LONG_ENTRY_ACTIONS)].sum(dim=-1),
                raw_action_probabilities[:, 1, list(SHORT_ENTRY_ACTIONS)].sum(dim=-1),
            ), dim=1)
            flat_entry_direction_mass_loss = torch.relu(
                torch.as_tensor(direction_floor, dtype=owner_entry_mass.dtype, device=self.device)
                - owner_entry_mass[direction_flat_mask]
            ).square().mean()
        raw_allowed = torch.as_tensor(
            allowed_action_masks, dtype=torch.bool, device=self.device,
        )
        for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
            raw_allowed[..., legacy_action] = False
        raw_allowed[..., 0] = True
        masked_action_logits = torch.where(
            raw_allowed, raw_action_logits, torch.full_like(raw_action_logits, -30.0),
        )
        action_probabilities = torch.softmax(masked_action_logits, dim=-1)
        group_probabilities_by_agent = torch.stack((
            action_probabilities[..., 0],
            action_probabilities[..., list(LONG_ENTRY_ACTIONS)].sum(dim=-1),
            action_probabilities[..., list(SHORT_ENTRY_ACTIONS)].sum(dim=-1),
        ), dim=-1)
        group_probabilities = group_probabilities_by_agent.mean(dim=1)
        auxiliary_loss = utility_pred.sum() * 0.0
        utility_loss = utility_pred.sum() * 0.0
        excursion_loss = excursion_pred.sum() * 0.0
        no_edge_loss = no_edge_logit.sum() * 0.0
        direction_loss = group_probabilities.sum() * 0.0
        hold_ceiling_loss = group_probabilities.sum() * 0.0
        ranking_loss = raw_action_logits.sum() * 0.0
        initial_sl_local_loss = raw_action_logits.sum() * 0.0
        horizon_loss = horizon_pred.sum() * 0.0
        path_dynamics_loss = path_dynamics_pred.sum() * 0.0
        first_hit_loss = first_hit_logits.sum() * 0.0
        action_path_loss = action_path_pred.sum() * 0.0
        action_outcome_loss = action_outcome_logits.sum() * 0.0
        utility_mae = 0.0
        excursion_mae = 0.0
        no_edge_accuracy = 0.0
        ranking_pair_count = 0.0
        if ranking_best_actions is not None:
            best_action = torch.as_tensor(
                ranking_best_actions, dtype=torch.long, device=self.device,
            )
            chosen_action = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            ranking_margin = torch.as_tensor(
                ranking_margins, dtype=torch.float32, device=self.device,
            )
            ranking_mask = torch.as_tensor(
                ranking_loss_masks, dtype=torch.bool, device=self.device,
            )
            if any(item.shape != (batch, FIVE_AGENT_COUNT) for item in (
                best_action, chosen_action, ranking_margin, ranking_mask,
            )):
                raise ValueError("ranking target phai co shape [B,5]")
            if bool(torch.any((best_action < 0) | (best_action > 6))):
                raise ValueError("ranking best action phai nam trong HOLD + 6 entry")
            chosen_logit = raw_action_logits.gather(2, chosen_action.unsqueeze(-1)).squeeze(-1)
            best_logit = raw_action_logits.gather(2, best_action.unsqueeze(-1)).squeeze(-1)
            pair_loss = torch.relu(ranking_margin + chosen_logit - best_logit)
            if bool(torch.any(ranking_mask)):
                anchor_weight = torch.as_tensor(
                    ranking_loss_weights, dtype=torch.float32, device=self.device,
                ).reshape(batch, 1)
                ranking_loss = (
                    pair_loss[ranking_mask]
                    * anchor_weight.expand(-1, FIVE_AGENT_COUNT)[ranking_mask]
                ).mean()
                ranking_pair_count = float(ranking_mask.sum().detach().cpu().item())
        initial_sl_local_count = 0.0
        if initial_sl_local_loss_masks is not None:
            local_mask = torch.as_tensor(
                initial_sl_local_loss_masks, dtype=torch.bool, device=self.device,
            )
            if local_mask.shape != (batch, FIVE_AGENT_COUNT):
                raise ValueError("initial_sl_local_loss_masks phai co shape [B,5]")
            local_weight = (
                torch.ones(batch, dtype=torch.float32, device=self.device)
                if initial_sl_local_loss_weights is None
                else torch.as_tensor(
                    initial_sl_local_loss_weights, dtype=torch.float32, device=self.device
                ).reshape(-1)
            )
            initial_sl_local_loss, local_count = self.initial_sl_conditional_entry_loss(
                raw_action_logits,
                torch.as_tensor(actions, dtype=torch.long, device=self.device),
                local_mask,
                local_weight,
            )
            initial_sl_local_count = float(local_count)
        if bool(np.any(enabled_np)):
            utility_target = torch.as_tensor(utility_targets, dtype=torch.float32, device=self.device)
            excursion_target = torch.as_tensor(excursion_targets, dtype=torch.float32, device=self.device)
            no_edge_target = torch.as_tensor(no_edge_targets, dtype=torch.float32, device=self.device).reshape(-1)
            horizon_target = torch.as_tensor(horizon_targets, dtype=torch.float32, device=self.device)
            horizon_mask = torch.as_tensor(horizon_masks, dtype=torch.float32, device=self.device)
            path_dynamics_target = torch.as_tensor(path_dynamics_targets, dtype=torch.float32, device=self.device)
            first_hit_target = torch.as_tensor(first_hit_classes, dtype=torch.long, device=self.device)
            action_path_target = torch.as_tensor(action_path_targets, dtype=torch.float32, device=self.device)
            action_outcome_target = torch.as_tensor(action_outcome_classes, dtype=torch.long, device=self.device)
            if utility_target.shape != (batch, 2) or excursion_target.shape != (batch, 4) or no_edge_target.shape != (batch,):
                raise ValueError("pattern tree target shapes must be [B,2], [B,4], [B]")
            if (
                horizon_target.shape != (batch, 5) or horizon_mask.shape != (batch, 5)
                or path_dynamics_target.shape != (batch, 10)
                or first_hit_target.shape != (batch, 2)
                or action_path_target.shape != (batch, 6, 5)
                or action_outcome_target.shape != (batch, 6)
            ):
                raise ValueError("rich path targets phai co shape [B,5], [B,10], [B,2], [B,6,5], [B,6]")
            enabled = torch.as_tensor(enabled_np, dtype=torch.bool, device=self.device)
            point_utility = F.smooth_l1_loss(utility_pred, utility_target, reduction="none").mean(dim=-1)
            direction_utility = F.smooth_l1_loss(
                utility_pred[:, 0] - utility_pred[:, 1],
                utility_target[:, 0] - utility_target[:, 1],
                reduction="none",
            )
            utility_per = 0.5 * (point_utility + direction_utility)
            excursion_per = F.smooth_l1_loss(
                excursion_pred, excursion_target, reduction="none"
            ).mean(dim=-1)
            positive = float(no_edge_target[enabled].sum().detach().cpu().item())
            negative = float(enabled.sum().detach().cpu().item()) - positive
            pos_weight = torch.as_tensor(
                float(np.clip(negative / max(positive, 1.0), 1.0, 4.0)),
                dtype=torch.float32,
                device=self.device,
            )
            no_edge_per = F.binary_cross_entropy_with_logits(
                no_edge_logit, no_edge_target, reduction="none", pos_weight=pos_weight
            )
            horizon_point = F.smooth_l1_loss(horizon_pred, horizon_target, reduction="none")
            horizon_per = (horizon_point * horizon_mask).sum(dim=-1) / horizon_mask.sum(dim=-1).clamp_min(1.0)
            path_dynamics_per = F.smooth_l1_loss(
                path_dynamics_pred, path_dynamics_target, reduction="none"
            ).mean(dim=-1)
            first_hit_per = F.cross_entropy(
                first_hit_logits.reshape(batch * 2, 3), first_hit_target.reshape(-1), reduction="none"
            ).reshape(batch, 2).mean(dim=-1)
            action_path_per = F.smooth_l1_loss(
                action_path_pred, action_path_target, reduction="none"
            ).mean(dim=(1, 2))
            action_outcome_per = F.cross_entropy(
                action_outcome_logits.reshape(batch * 6, 4), action_outcome_target.reshape(-1), reduction="none"
            ).reshape(batch, 6).mean(dim=-1)
            overall_w = torch.as_tensor(auxiliary_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            utility_w = torch.as_tensor(utility_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            excursion_w = torch.as_tensor(excursion_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            no_edge_w = torch.as_tensor(no_edge_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            direction_w = torch.as_tensor(direction_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            hold_ceiling_w = torch.as_tensor(hold_ceiling_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            horizon_w = torch.as_tensor(horizon_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            path_dynamics_w = torch.as_tensor(path_dynamics_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            first_hit_w = torch.as_tensor(first_hit_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            action_path_w = torch.as_tensor(action_path_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            action_outcome_w = torch.as_tensor(action_outcome_loss_weights, dtype=torch.float32, device=self.device).reshape(-1)
            direction_target = torch.as_tensor(direction_classes, dtype=torch.long, device=self.device).reshape(-1)
            hold_ceiling = torch.as_tensor(edge_hold_ceilings, dtype=torch.float32, device=self.device).reshape(-1)
            if any(item.shape != (batch,) for item in (
                overall_w, utility_w, excursion_w, no_edge_w, direction_w,
                hold_ceiling_w, horizon_w, path_dynamics_w, first_hit_w,
                action_path_w, action_outcome_w, direction_target, hold_ceiling,
            )):
                raise ValueError("pattern tree loss weights must have one value per anchor")
            if bool(torch.any((direction_target < 0) | (direction_target > 2))):
                raise ValueError("direction classes must be HOLD=0, LONG=1 or SHORT=2")
            selected_group = group_probabilities_by_agent.gather(
                2,
                direction_target.reshape(batch, 1, 1).expand(-1, FIVE_AGENT_COUNT, 1),
            ).squeeze(-1)
            direction_per = -torch.log(selected_group.clamp_min(1e-8)).mean(dim=1)
            edge_mask = direction_target != 0
            hold_ceiling_per = torch.where(
                edge_mask,
                torch.relu(
                    group_probabilities_by_agent[:, :, 0]
                    - hold_ceiling.clamp(0.0, 1.0).unsqueeze(-1)
                ).square().mean(dim=1),
                torch.zeros_like(group_probabilities[:, 0]),
            )
            denominator = (
                utility_w + excursion_w + no_edge_w + direction_w + hold_ceiling_w
                + horizon_w + path_dynamics_w + first_hit_w + action_path_w + action_outcome_w
            ).clamp_min(1e-6)
            combined_per = (
                utility_w * utility_per
                + excursion_w * excursion_per
                + no_edge_w * no_edge_per
                + direction_w * direction_per
                + hold_ceiling_w * hold_ceiling_per
                + horizon_w * horizon_per
                + path_dynamics_w * path_dynamics_per
                + first_hit_w * first_hit_per
                + action_path_w * action_path_per
                + action_outcome_w * action_outcome_per
            ) / denominator
            auxiliary_loss = (overall_w[enabled] * combined_per[enabled]).mean()
            utility_loss = utility_per[enabled].mean()
            excursion_loss = excursion_per[enabled].mean()
            no_edge_loss = no_edge_per[enabled].mean()
            direction_loss = direction_per[enabled].mean()
            hold_ceiling_loss = hold_ceiling_per[enabled].mean()
            rich_enabled = enabled & (
                horizon_w + path_dynamics_w + first_hit_w + action_path_w + action_outcome_w > 0.0
            )
            if bool(torch.any(rich_enabled)):
                horizon_loss = horizon_per[rich_enabled].mean()
                path_dynamics_loss = path_dynamics_per[rich_enabled].mean()
                first_hit_loss = first_hit_per[rich_enabled].mean()
                action_path_loss = action_path_per[rich_enabled].mean()
                action_outcome_loss = action_outcome_per[rich_enabled].mean()
            utility_mae = float(torch.abs(utility_pred[enabled] - utility_target[enabled]).mean().detach().cpu().item())
            excursion_mae = float(torch.abs(excursion_pred[enabled] - excursion_target[enabled]).mean().detach().cpu().item())
            no_edge_accuracy = float(
                ((torch.sigmoid(no_edge_logit[enabled]) >= 0.5) == (no_edge_target[enabled] >= 0.5))
                .float().mean().detach().cpu().item()
            )
        replay_loss, replay_stats = self.failure_replay_loss(
            tuple(failure_replay_cases or ())
        )
        replay_weight = float(np.clip(failure_replay_weight, 0.0, 2.0))
        reasoning_loss = actor_loss.sum() * 0.0
        reasoning_stats = {
            "reasoning_rl_enabled": float(bool(reasoning_rl_enabled)),
            "reasoning_group_eligible": 0.0,
            "reasoning_optimizer_eligible": 0.0,
            "reasoning_loss": 0.0,
        }
        if bool(reasoning_rl_enabled):
            if reasoning_rl_masks is None or reasoning_action_q_values_r is None:
                raise ValueError(
                    "reasoning RL requires anchor masks and action-local verifier values"
                )
            reasoning_loss, reasoning_stats = self.reasoning_grpo_loss(
                hidden=evaluated_hidden,
                allowed_action_masks=torch.as_tensor(
                    allowed_action_masks, dtype=torch.bool, device=self.device
                ),
                action_q_values_r=torch.as_tensor(
                    reasoning_action_q_values_r,
                    dtype=torch.float32,
                    device=self.device,
                ),
                anchor_mask=torch.as_tensor(
                    reasoning_rl_masks, dtype=torch.bool, device=self.device
                ),
                group_size=reasoning_group_size,
                reasoning_steps=reasoning_steps,
                ppo_clip=reasoning_ppo_clip,
                kl_beta=reasoning_kl_beta,
                entropy_weight=reasoning_entropy_weight,
                loss_weight=reasoning_loss_weight,
                encoder_credit_scale=reasoning_encoder_credit_scale,
                min_group_std_r=reasoning_min_group_std_r,
            )
        total_loss = (
            actor_loss + auxiliary_loss + ranking_loss + initial_sl_local_loss
            + replay_weight * replay_loss + reasoning_loss
            + float(self.schedule.flat_entry_mass_loss_weight) * flat_entry_mass_loss
            + float(self.schedule.flat_entry_mass_log_loss_weight) * flat_entry_mass_log_loss
            + float(self.schedule.flat_entry_direction_mass_loss_weight)
            * flat_entry_direction_mass_loss
        )
        if getattr(self, "regularizer", None) is not None:
            actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            targets_3class = self.regularizer.derive_targets(actions_t)
            reg_dict = self.regularizer(
                evaluated_hidden,
                targets=targets_3class,
            )
            total_loss = total_loss + reg_dict["loss_reg_total"]
            stats["pattern_loss_var"] = float(reg_dict["loss_var"].detach().cpu().item())
            stats["pattern_loss_cov"] = float(reg_dict["loss_cov"].detach().cpu().item())
            stats["pattern_loss_rank"] = float(reg_dict["loss_rank"].detach().cpu().item())
            stats["pattern_loss_arcface"] = float(reg_dict["loss_arcface"].detach().cpu().item())
            stats["pattern_loss_reg_total"] = float(reg_dict["loss_reg_total"].detach().cpu().item())
            stats["pattern_effective_rank"] = float(reg_dict["effective_rank"])
        stats.update(self.step_from_loss(total_loss))
        stats["pattern_actor_total_loss"] = float(stats["total_loss"])
        stats["flat_entry_mass_floor"] = entry_floor
        stats["flat_entry_mass_loss_weight"] = float(self.schedule.flat_entry_mass_loss_weight)
        stats["flat_entry_mass_loss"] = float(flat_entry_mass_loss.detach().cpu().item())
        stats["flat_entry_mass_log_loss_weight"] = float(
            self.schedule.flat_entry_mass_log_loss_weight
        )
        stats["flat_entry_mass_log_loss"] = float(
            flat_entry_mass_log_loss.detach().cpu().item()
        )
        stats["flat_entry_mass_samples"] = float(flat_mask_t.sum().detach().cpu().item())
        stats["flat_entry_mass_mean"] = float(
            raw_entry_mass[flat_mask_t].mean().detach().cpu().item()
            if bool(torch.any(flat_mask_t)) else 0.0
        )
        stats["flat_entry_mass_regularizer_source"] = "raw_unmasked_action_logits"
        stats["flat_entry_direction_mass_floor"] = direction_floor
        stats["flat_entry_direction_mass_loss_weight"] = float(
            self.schedule.flat_entry_direction_mass_loss_weight
        )
        stats["flat_entry_direction_mass_loss"] = float(
            flat_entry_direction_mass_loss.detach().cpu().item()
        )
        stats["flat_entry_direction_mass_samples"] = float(
            direction_flat_mask.sum().detach().cpu().item()
        )
        stats["pattern_tree_auxiliary_loss"] = float(auxiliary_loss.detach().cpu().item())
        stats.update(reasoning_stats)
        stats["pattern_tree_utility_loss"] = float(utility_loss.detach().cpu().item())
        stats["pattern_tree_excursion_loss"] = float(excursion_loss.detach().cpu().item())
        stats["pattern_tree_no_edge_loss"] = float(no_edge_loss.detach().cpu().item())
        stats["pattern_tree_direction_loss"] = float(direction_loss.detach().cpu().item())
        stats["pattern_tree_hold_ceiling_loss"] = float(hold_ceiling_loss.detach().cpu().item())
        stats["pattern_action_ranking_loss"] = float(ranking_loss.detach().cpu().item())
        stats["initial_sl_local_loss"] = float(initial_sl_local_loss.detach().cpu().item())
        stats["initial_sl_local_samples"] = initial_sl_local_count
        stats.update(replay_stats)
        stats["failure_replay_weight"] = replay_weight
        stats["failure_replay_weighted_loss"] = float(
            (replay_weight * replay_loss).detach().cpu().item()
        )
        stats["pattern_action_ranking_pairs"] = float(ranking_pair_count)
        stats["pattern_tree_horizon_loss"] = float(horizon_loss.detach().cpu().item())
        stats["pattern_tree_path_dynamics_loss"] = float(path_dynamics_loss.detach().cpu().item())
        stats["pattern_tree_first_hit_loss"] = float(first_hit_loss.detach().cpu().item())
        stats["pattern_tree_action_path_loss"] = float(action_path_loss.detach().cpu().item())
        stats["pattern_tree_action_outcome_loss"] = float(action_outcome_loss.detach().cpu().item())
        stats["pattern_tree_hold_probability"] = float(
            group_probabilities[:, 0].mean().detach().cpu().item()
        )
        stats["pattern_tree_long_probability"] = float(
            group_probabilities[:, 1].mean().detach().cpu().item()
        )
        stats["pattern_tree_short_probability"] = float(
            group_probabilities[:, 2].mean().detach().cpu().item()
        )
        stats["pattern_tree_utility_mae"] = utility_mae
        stats["pattern_tree_excursion_mae"] = excursion_mae
        stats["pattern_tree_no_edge_accuracy"] = no_edge_accuracy
        stats["pattern_tree_supervised_anchors"] = float(np.sum(enabled_np))
        stats["total_loss"] = float(total_loss.detach().cpu().item())
        stats.update({
            "pattern_batch_size": float(batch),
            "preserve_reward_sign": float(bool(preserve_reward_sign)),
            "entropy_coefficient": float(self.schedule.entropy_coefficient),
            "behavior_uniform_mix": float(np.mean(behavior_uniform_mixes)),
            "entropy_verified_edge_count": float(self.schedule.verified_edge_count),
            "primary_action_entropy": float(torch.cat(
                [row["primary_action_entropy"] for row in evaluated_rows], dim=0
            ).mean().detach().cpu().item()),
            "primary_entry_probability_mass": float(torch.cat(
                [row["primary_entry_probability_mass"] for row in evaluated_rows], dim=0
            ).mean().detach().cpu().item()),
        })
        return stats

    def step_from_loss(self, loss: Tensor) -> dict[str, float]:
        calibrator = getattr(self.policy.heads, "entry_role_calibrator", None)
        anchor_penalty = loss.sum() * 0.0
        anchor_active = self.teacher_imitation_anchor_active
        retained_parameters = self._teacher_retained_parameters()
        if anchor_active and retained_parameters:
            terms = []
            for name, parameter in retained_parameters.items():
                anchor = self.teacher_imitation_anchor[name].to(
                    device=parameter.device, dtype=parameter.dtype
                )
                terms.append((parameter - anchor).square().mean())
            if terms:
                anchor_penalty = torch.stack(terms).mean()
                loss = loss + self.teacher_imitation_anchor_weight * anchor_penalty

        # --- ELASTIC WEIGHT CONSOLIDATION (EWC) PENALTY INJECTION ---
        ewc_loss = loss.sum() * 0.0
        ewc_active = bool(getattr(self, "ewc", None) is not None and getattr(self.ewc, "is_active", False))
        if ewc_active:
            ewc_loss = self.ewc.compute_ewc_loss()
            loss = loss + ewc_loss
        # -------------------------------------------------------------

        self.optimizer.zero_grad()
        loss.backward()
        warmup_frozen = self.teacher_imitation_warmup_active
        representation_frozen = self.teacher_imitation_representation_frozen
        if representation_frozen:
            # The teacher consumes causal observations through the encoder,
            # LSTM and primary-action projection.  Preserve that feature space
            # for the complete retention interval.  During the shorter warm-up
            # the calibrator is frozen too; afterwards it may adapt under the
            # anchor penalty. Geometry heads trained by the teacher follow the
            # same freeze/anchor schedule; unrelated execution heads stay live.
            teacher_modules = (
                self.policy.observation_encoder,
                self.policy.lstm,
                self.policy.post_lstm_transformer,
                self.policy.heads.action_query,
                self.policy.heads.action_embedding,
            )
            direct = getattr(self.policy.heads, "direct_action_head", None)
            if isinstance(direct, nn.Module):
                teacher_modules = teacher_modules + (direct,)
            direct_observation = getattr(
                self.policy.heads, "direct_observation_head", None
            )
            if isinstance(direct_observation, nn.Module):
                teacher_modules = teacher_modules + (direct_observation,)
            if warmup_frozen:
                teacher_modules = teacher_modules + (calibrator,)
                seven_choice = bool(getattr(
                    self.policy.heads, "seven_choice_imitation_trained", False
                ))
                if self.teacher_geometry_imitation_enabled or seven_choice:
                    geometry_names = (
                        ("delay", "entry_offset", "sl", "tp", "trailing")
                        if seven_choice else ("delay", "sl", "tp", "trailing")
                    )
                    teacher_modules = teacher_modules + tuple(
                        self.policy.heads.parameter_heads[name]
                        for name in geometry_names
                    )
                if seven_choice:
                    teacher_modules = teacher_modules + (
                        self.policy.observation_encoder.pattern_tree.action_path_head,
                    )
                    residual = getattr(
                        self.policy.observation_encoder.pattern_tree,
                        "action_path_residual", None,
                    )
                    if isinstance(residual, nn.Module):
                        teacher_modules = teacher_modules + (residual,)
                    observation_adapter = getattr(
                        self.policy.observation_encoder.pattern_tree,
                        "action_path_observation_adapter", None,
                    )
                    if isinstance(observation_adapter, nn.Module):
                        teacher_modules = teacher_modules + (observation_adapter,)
            seen_parameters: set[int] = set()
            for module in teacher_modules:
                if not isinstance(module, nn.Module):
                    continue
                for parameter in module.parameters():
                    if id(parameter) in seen_parameters:
                        continue
                    seen_parameters.add(id(parameter))
                    parameter.grad = None
        has_nan_grad = False
        for parameter in self.policy.parameters():
            if parameter.grad is not None:
                if torch.isnan(parameter.grad).any() or torch.isinf(parameter.grad).any():
                    has_nan_grad = True
                    break
        if has_nan_grad:
            self.optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
        if self.teacher_imitation_enabled:
            self.teacher_imitation_optimizer_updates += 1
        distance = 0.0
        if self.teacher_imitation_enabled and retained_parameters:
            with torch.no_grad():
                values = []
                for name, parameter in retained_parameters.items():
                    anchor = self.teacher_imitation_anchor[name].to(
                        device=parameter.device, dtype=parameter.dtype
                    )
                    values.append(float((parameter - anchor).square().mean().cpu().item()))
                distance = float(sum(values) / max(1, len(values)))
        self.last_teacher_imitation_stats = {
            "teacher_imitation_enabled": float(self.teacher_imitation_enabled),
            "teacher_imitation_geometry_enabled": float(self.teacher_geometry_imitation_enabled),
            "teacher_imitation_warmup_active": float(warmup_frozen),
            "teacher_imitation_pathway_frozen": float(warmup_frozen),
            "teacher_imitation_representation_frozen": float(representation_frozen),
            "teacher_imitation_anchor_active": float(anchor_active),
            "teacher_imitation_optimizer_updates": float(self.teacher_imitation_optimizer_updates),
            "teacher_imitation_anchor_penalty": float(anchor_penalty.detach().cpu().item()),
            "teacher_imitation_anchor_distance": distance,
            "ewc_loss": float(ewc_loss.detach().cpu().item()) if isinstance(ewc_loss, Tensor) else 0.0,
            "ewc_active": float(ewc_active),
        }
        return dict(self.last_teacher_imitation_stats)


def public_profile_audit() -> dict[str, object]:
    profile = FivePublicProfile()
    return {
        "agent_count": profile.agent_count,
        "lstm_size": profile.lstm_size,
        "published_lstm_unroll": profile.published_lstm_unroll,
        "lstm_unroll": profile.lstm_unroll,
        "cross_agent_pool_fraction": profile.cross_agent_pool_fraction,
        "post_lstm_transformer_width": profile.post_lstm_transformer_width,
        "post_lstm_transformer_heads": profile.post_lstm_transformer_heads,
        "post_lstm_transformer_ffn_size": profile.post_lstm_transformer_ffn_size,
        "post_lstm_transformer_layers": profile.post_lstm_transformer_layers,
        "temporal_token_horizons": list(profile.temporal_token_horizons),
        "temporal_token_count_per_agent": FIVE_TEMPORAL_TOKEN_COUNT,
        "policy_architecture_version": FIVE_POLICY_ARCHITECTURE_VERSION,
        "internal_policy_state": True,
        "internal_policy_state_schema_version": INTERNAL_POLICY_STATE_SCHEMA_VERSION,
        "internal_policy_state_is_continuous": True,
        "internal_policy_state_external_strategy_router": False,
        "agent_authority_version": FIVE_AGENT_AUTHORITY_VERSION,
        "agent_authorities": [
            {
                "agent_id": item.agent_id,
                "role": item.role,
                "allowed_actions": sorted(item.allowed_actions),
                "responsibility": item.responsibility,
            }
            for item in FIVE_AGENT_AUTHORITIES
        ],
        "ppo_clip": profile.ppo_clip,
        "gae_lambda": profile.gae_lambda,
        "adam_beta1": profile.adam_beta1,
        "adam_beta2": profile.adam_beta2,
        "team_spirit_start": profile.team_spirit_start,
        "team_spirit_end": profile.team_spirit_end,
        "gae_horizon_start_seconds": profile.gae_horizon_start_seconds,
        "gae_horizon_end_seconds": profile.gae_horizon_end_seconds,
        "entropy_start": profile.entropy_start,
        "entropy_end": profile.entropy_end,
        "learning_rate_start": profile.learning_rate_start,
        "learning_rate_end": profile.learning_rate_end,
        "past_opponent_fraction": profile.past_opponent_fraction,
        "behavior_entry_probability_floor": profile.behavior_entry_probability_floor,
        "internal_policy_state_diversity_weight": 1e-3,
        "internal_policy_state_variance_floor": 0.05,
        "observation_architecture_version": FIVE_OBSERVATION_ARCHITECTURE_VERSION,
        "observation_pattern_tree": True,
        "observation_pattern_root_size": PATTERN_NODE_SIZE,
        "observation_pattern_targets": (
            "long_utility", "short_utility", "long_mfe", "long_adverse",
            "short_mfe", "short_adverse", "no_edge",
            "returns_h4_h8_h16_h32_h48", "time_to_mae_mfe",
            "underwater_recovery_persistence", "sl_tp_first",
            "action_fill_delay", "action_postfill_mae_mfe", "action_outcome",
        ),
        "trading_pre_lstm_size": PRE_LSTM_SIZE,
        "trading_position_embedding_size": POSITION_EMBED_SIZE,
        "publicly_exact_internal_widths": False,
    }
