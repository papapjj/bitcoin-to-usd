from __future__ import annotations

import copy as copylib
import heapq
import hashlib
import json
import pickle
import shutil
import time
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np

from .config import AppConfig, FeatureConfig, SymbolConfig
from .live_observatory import LiveTelemetryWriter, topology_summary
from .data import split_time_ordered
from .environment import TradingEnvironment
from .fitness import daily_trade_metrics, edge_confidence_lcb, goal_progress_metrics, tight_risk_score
from .feature_contract import feature_schema_fingerprint
from .hidden_normalization import normalize_hidden
from .hidden_regularization import deterministic_dropout, layerwise_bools, layerwise_rates, mandatory_layerwise_rates
from .legacy_pickle_migration import load_pickle_with_removed_context_structural_migration
from .learning_integrity import clone_policy, policy_state_hash, promotion_stability_report, run_manifest, sha256_file, sha256_tree, stable_object_hash
from .growth import (
    MODULE_LIFECYCLE_ACTIVE,
    MODULE_LIFECYCLE_CANDIDATE,
    MODULE_LIFECYCLE_CONSOLIDATED,
    MODULE_LIFECYCLE_DORMANT,
    MODULE_LIFECYCLE_PRUNING,
    GrowthController,
    DevelopmentalOutput,
    ModuleSpec,
    ROLE_CRITIC_SPECIALIST,
    ROLE_FEATURE_SPECIALIST,
    ROLE_GENERAL_SPECIALIST,
    ROLE_HYPOTHESIS_SPECIALIST,
    ROLE_MEMORY_SPECIALIST,
    ROLE_PREDICTOR_SPECIALIST,
    ShadowBranch,
    cosine_gradient_similarity,
    diagnose_cluster,
    error_clustering,
    should_split_module,
)
from .mt5_runner import (
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    MT5_MODEL_REAL_TICKS,
    _date_span_days,
    _dict_expectancy,
    mt5_learning_signal,
)
from .m1_graph_module import M1GraphModule
from .partitioned_memory import PartitionedWorkingMemory
from .trading_world_model import TradingWorldModel, WORLD_MODEL_SCHEMA_VERSION
from .learner_recovery import ActionRecoveryLearner, LEARNER_RECOVERY_SCHEMA_VERSION, primary_action, trace_context
from .context_frame import (
    ACTION_COUNT as CONTEXT_ACTION_COUNT,
    CANONICAL_MEMORY_SCHEMA_VERSION,
    ContextFrame,
    MEMORY_KEY_SIZE as CONTEXT_MEMORY_KEY_SIZE,
    blockwise_cosine,
)
from .protocol_contract import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    POSITION_OFFSET,
    TIMING_OFFSET,
)
from .graph_plasticity import GraphCreditSignal, GraphPlasticityEngine, GraphPlasticityReport
from .graph_supernetwork_runtime import (
    GraphSupernetworkRuntime,
    summarize_graph_trial_history,
    write_graph_trial_history,
)
from .neural_graph import NeuralGraphBudget
from .reasoning_graph import ReasoningGraphSystem
from .causal_ledger import CausalLedger, CausalOutcomeReport, CreditPlan
from .causal_replay import CounterfactualReplayJournal
from .child_supernetwork_manager import ChildSupernetworkManager
from .mistake_memory import (
    MistakeEpisodeStore,
    episode_from_rejection,
    episode_from_trace,
)
from .mistake_replay import MistakeReplayEngine
from .replacement_learning import (
    NEGATIVE_CONTEXT_CREDIT_MULTIPLIER,
    action_scope,
    proposal_confidence,
    scope_matches,
    stable_identifier,
)
from .plasticity import (
    PlasticityCoefficients,
    STDPParams,
    ShortTermPlasticityState,
    compute_total_loss,
    homeostatic_rescale_weights,
    homeostatic_update_bias,
    compose_stable_plasticity_update,
    normalize_plasticity_vector,
    short_term_plasticity_effect,
    stdp_update_scalar,
)


# Direct-RNN hidden regularization is a hard training contract. Dropout is
# active for training forwards only; frozen inference is deterministic.
REQUIRED_HIDDEN_NORMALIZATION_MODE = "layernorm_v1"
REQUIRED_HIDDEN_DROPOUT_RATES = (0.05, 0.10, 0.10, 0.05, 0.10, 0.10)


def _mandatory_hidden_normalization_mode(value: object) -> str:
    mode = str(value or REQUIRED_HIDDEN_NORMALIZATION_MODE).strip().lower()
    return mode if mode in {"rms_v1", "layernorm_v1"} else REQUIRED_HIDDEN_NORMALIZATION_MODE


MEMORY_KEY_SCHEMA_VERSION = CANONICAL_MEMORY_SCHEMA_VERSION
MARKET_MEMORY_KEY_SIZE = 32
ACCOUNT_MEMORY_KEY_SIZE = 8
POSITION_MEMORY_KEY_SIZE = 6
MEMORY_KEY_SIZE = MARKET_MEMORY_KEY_SIZE + ACCOUNT_MEMORY_KEY_SIZE + POSITION_MEMORY_KEY_SIZE
ACCOUNT_CONTEXT_FIELDS = (
    "equity_to_balance",
    "floating_pnl_r",
    "current_equity_drawdown",
    "balance_drawdown",
    "open_risk_ratio",
    "equity_return_1_bar",
    "equity_return_5_bar",
    "equity_volatility_20",
    "losing_streak_norm",
    "time_under_water_norm",
    "position_count_norm",
    "day_trade_pressure",
)
# Bridge/action schema v4: 12 discrete commands (including MANAGE_POSITION)
# plus 8 parameters.  Legacy 19-output artifacts are readable but must be
# migrated/retrained before deployment to the v4 EA.
ACTION_COUNT = 12
PARAMETER_OFFSET = ACTION_COUNT
OUTPUT_COUNT = PARAMETER_OFFSET + 8
ENTRY_ACTION_INDICES = frozenset({1, 2, 3, 4, 5, 6})
M1_COMPONENTS = (
    "state_encoder",
    "working_memory",
    "hypothesis_bank",
    "predictor",
    "critic",
    "selector",
)


def _component_input_mask(input_size: int, sources: tuple[str, ...] | list[str]) -> np.ndarray:
    """Map module input sources to the exact v8 bridge feature groups.

    The old percentage-based bands silently split account/position/execution
    fields between unrelated modules whenever the schema had a full bridge input. For
    the production contract we select semantic field groups.  Small synthetic
    test inputs retain a deterministic fallback rather than pretending to use
    the v4 schema.
    """
    n = max(1, int(input_size))
    mask = np.zeros(n, dtype=float)
    if n >= INPUT_COUNT:
        market = np.r_[np.arange(0, POSITION_OFFSET), np.arange(EXTRA_MARKET_OFFSET, EXECUTION_OFFSET), np.arange(TIMING_OFFSET, min(n, INPUT_COUNT))]
        positions = np.arange(POSITION_OFFSET, ACCOUNT_OFFSET)
        account = np.arange(ACCOUNT_OFFSET, EXTRA_MARKET_OFFSET)
        execution = np.arange(EXECUTION_OFFSET, min(n, TIMING_OFFSET))
        all_features = np.arange(min(n, INPUT_COUNT))
        source_map = {
            "state_encoder": np.r_[market, execution],
            "working_memory": np.r_[market, positions, account],
            "predictor": np.r_[market, positions, execution],
            "critic": all_features,
            "selector": all_features,
        }
        for source in sources:
            indices = source_map.get(str(source))
            if indices is not None:
                mask[np.asarray(indices, dtype=int)] = 1.0
        if not np.any(mask):
            mask[np.r_[market, execution]] = 1.0
        return mask
    # Synthetic/small policies have no v4 field layout.  Let their modules see
    # every supplied input rather than slicing arbitrary index percentages.
    mask[:] = 1.0
    return mask


def _component_output_mask(output_size: int, targets: tuple[str, ...] | list[str]) -> np.ndarray:
    """Map a neural-development output spec to concrete action dimensions."""
    n = max(1, int(output_size))
    mask = np.zeros(n, dtype=float)
    target_map = {
        "state_encoder": slice(0, min(n, 8)),
        "working_memory": slice(0, min(n, 11)),
        "predictor": slice(1, min(n, 8)),
        "critic": slice(0, min(n, 11)),
        "selector": slice(0, min(n, 12)),
    }
    for target in targets:
        band = target_map.get(str(target))
        if band is not None:
            mask[band] = 1.0
    if not np.any(mask):
        mask[: min(n, 8)] = 1.0
    return mask



@dataclass(frozen=True)
class MT5CurriculumStage:
    name: str
    start_round: int
    from_date: str
    to_date: str
    max_positions: int
    max_holding_bars: int


@dataclass(frozen=True)
class GrowthSignal:
    error: float
    novelty: float
    conflict: float
    representation_need: float
    complexity: float

    @property
    def score(self) -> float:
        return self.error + self.novelty + self.conflict + self.representation_need - self.complexity


# ── M₁ Sub-Components ────────────────────────────────────────────────────────


class StateEncoder:
    __slots__ = ("w", "b", "activation_ema")

    def __init__(self, input_size: int, hidden_size: int, rng: np.random.Generator) -> None:
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        self.w = rng.normal(0.0, scale, size=(hidden_size, input_size))
        self.b = rng.normal(0.0, 0.05, size=hidden_size)
        self.activation_ema = 0.0

    def encode(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(self.w @ x + self.b)

    def copy(self) -> "StateEncoder":
        enc = StateEncoder.__new__(StateEncoder)
        enc.w = self.w.copy()
        enc.b = self.b.copy()
        enc.activation_ema = float(self.activation_ema)
        return enc


class WorkingMemory:
    __slots__ = ("size", "state", "g_keep_w", "g_keep_b", "g_write_w", "g_write_b",
                 "g_erase_w", "g_erase_b", "w_write", "decay")

    def __init__(self, size: int, input_size: int, rng: np.random.Generator) -> None:
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        self.size = size
        self.state = np.zeros(size, dtype=float)
        self.g_keep_w = rng.normal(0.0, scale, size=input_size)
        self.g_keep_b = 0.0
        self.g_write_w = rng.normal(0.0, scale, size=input_size)
        self.g_write_b = 0.0
        self.g_erase_w = rng.normal(0.0, scale, size=input_size)
        self.g_erase_b = 0.0
        self.w_write = rng.normal(0.0, 0.1, size=(size, input_size))
        self.decay = 0.85

    def step(self, x: np.ndarray, candidate: np.ndarray | None = None) -> np.ndarray:
        g_keep = 1.0 / (1.0 + np.exp(-np.clip(float(self.g_keep_w @ x + self.g_keep_b), -60, 60)))
        g_write = 1.0 / (1.0 + np.exp(-np.clip(float(self.g_write_w @ x + self.g_write_b), -60, 60)))
        g_erase = 1.0 / (1.0 + np.exp(-np.clip(float(self.g_erase_w @ x + self.g_erase_b), -60, 60)))
        if candidate is None:
            candidate = np.tanh(self.w_write @ x)
        self.state = self.decay * g_keep * self.state + g_write * candidate
        self.state = (1.0 - g_erase) * self.state
        return self.state

    def reset(self) -> None:
        self.state = np.zeros(self.size, dtype=float)

    def copy(self) -> "WorkingMemory":
        wm = WorkingMemory.__new__(WorkingMemory)
        wm.size = self.size
        wm.state = self.state.copy()
        wm.g_keep_w = self.g_keep_w.copy()
        wm.g_keep_b = float(self.g_keep_b)
        wm.g_write_w = self.g_write_w.copy()
        wm.g_write_b = float(self.g_write_b)
        wm.g_erase_w = self.g_erase_w.copy()
        wm.g_erase_b = float(self.g_erase_b)
        wm.w_write = self.w_write.copy()
        wm.decay = float(self.decay)
        return wm


class HypothesisBank:
    __slots__ = ("K", "w_gen", "b_gen", "w_decode", "b_decode", "lateral_w",
                 "scores", "selected_idx", "anti_hebb_strength")

    def __init__(self, K: int, state_size: int, output_size: int, rng: np.random.Generator) -> None:
        scale = 1.0 / max(1.0, np.sqrt(float(state_size)))
        self.K = K
        self.w_gen = rng.normal(0.0, scale, size=(K, state_size))
        self.b_gen = rng.normal(0.0, 0.05, size=K)
        self.w_decode = rng.normal(0.0, 0.25, size=(output_size, K))
        self.b_decode = rng.normal(0.0, 0.05, size=output_size)
        self.lateral_w = np.zeros((K, K), dtype=float)
        np.fill_diagonal(self.lateral_w, 0.0)
        self.scores = np.zeros(K, dtype=float)
        self.selected_idx = 0
        self.anti_hebb_strength = 0.01

    def generate(self, state: np.ndarray, memory: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        combined = np.concatenate([state, memory]) if memory.size > 0 else state
        if combined.size != self.w_gen.shape[1]:
            padded = np.zeros(self.w_gen.shape[1], dtype=float)
            padded[: min(combined.size, self.w_gen.shape[1])] = combined[: self.w_gen.shape[1]]
            combined = padded
        raw = self.w_gen @ combined + self.b_gen
        activations = np.tanh(raw)
        activations = activations - self.anti_hebb_strength * (self.lateral_w @ activations)
        self.scores = activations
        decoded = self.w_decode @ activations + self.b_decode
        return activations, decoded

    def select_lateral_inhibition(self, beta: float = 0.5) -> int:
        z = self.scores.copy()
        for _ in range(5):
            inhibition = self.lateral_w @ z
            z = np.tanh(self.scores - beta * inhibition)
        self.selected_idx = int(np.argmax(z))
        return self.selected_idx

    def update_lateral_weights(self, learning_rate: float = 0.001) -> None:
        for i in range(self.K):
            for j in range(self.K):
                if i != j:
                    self.lateral_w[i, j] += learning_rate * self.scores[i] * self.scores[j]
        np.fill_diagonal(self.lateral_w, 0.0)
        self.lateral_w = np.clip(self.lateral_w, -1.0, 1.0)

    def copy(self) -> "HypothesisBank":
        hb = HypothesisBank.__new__(HypothesisBank)
        hb.K = self.K
        hb.w_gen = self.w_gen.copy()
        hb.b_gen = self.b_gen.copy()
        hb.w_decode = self.w_decode.copy()
        hb.b_decode = self.b_decode.copy()
        hb.lateral_w = self.lateral_w.copy()
        hb.scores = self.scores.copy()
        hb.selected_idx = self.selected_idx
        hb.anti_hebb_strength = float(self.anti_hebb_strength)
        return hb


class Predictor:
    __slots__ = ("w_state", "w_memory", "w_action", "b", "D", "w_proj", "b_proj")

    def __init__(self, state_size: int, memory_size: int, action_size: int, rng: np.random.Generator) -> None:
        scale = 1.0 / max(1.0, np.sqrt(float(state_size + memory_size)))
        self.w_state = rng.normal(0.0, scale, size=(state_size, state_size))
        self.w_memory = rng.normal(0.0, scale, size=(state_size, memory_size))
        self.w_action = rng.normal(0.0, scale, size=(state_size, action_size))
        self.b = rng.normal(0.0, 0.05, size=state_size)
        self.D = 3
        self.w_proj = rng.normal(0.0, 0.1, size=(state_size, state_size))
        self.b_proj = rng.normal(0.0, 0.05, size=state_size)

    def predict(self, state: np.ndarray, memory: np.ndarray, action: np.ndarray) -> np.ndarray:
        s = state.copy()
        for _ in range(self.D):
            prev_s = s.copy()
            s = np.tanh(self.w_state @ s + self.w_memory @ memory + self.w_action @ action + self.b)
            if np.allclose(s, prev_s, atol=1e-6):
                break
        return np.tanh(self.w_proj @ s + self.b_proj)

    def predict_multistep(self, state: np.ndarray, memory: np.ndarray, action: np.ndarray, steps: int = 3) -> list[np.ndarray]:
        predictions = []
        s = state.copy()
        for _ in range(steps):
            prev_s = s.copy()
            s = np.tanh(self.w_state @ s + self.w_memory @ memory + self.w_action @ action + self.b)
            predictions.append(np.tanh(self.w_proj @ s + self.b_proj))
            if np.allclose(s, prev_s, atol=1e-6):
                break
        return predictions

    def copy(self) -> "Predictor":
        pred = Predictor.__new__(Predictor)
        pred.w_state = self.w_state.copy()
        pred.w_memory = self.w_memory.copy()
        pred.w_action = self.w_action.copy()
        pred.b = self.b.copy()
        pred.D = self.D
        pred.w_proj = self.w_proj.copy()
        pred.b_proj = self.b_proj.copy()
        return pred


class Critic:
    __slots__ = ("w_state", "w_goal", "b", "w_score", "b_score", "alpha_risk", "gamma_cost", "delta_info")

    def __init__(self, state_size: int, goal_size: int, rng: np.random.Generator) -> None:
        scale = 1.0 / max(1.0, np.sqrt(float(state_size)))
        self.w_state = rng.normal(0.0, scale, size=(state_size, state_size))
        self.w_goal = rng.normal(0.0, scale, size=(state_size, goal_size))
        self.b = rng.normal(0.0, 0.05, size=state_size)
        self.w_score = rng.normal(0.0, 0.1, size=state_size)
        self.b_score = 0.0
        self.alpha_risk = 0.1
        self.gamma_cost = 0.05
        self.delta_info = 0.2

    def evaluate(self, predicted_state: np.ndarray, goal: np.ndarray, risk: float = 0.0, cost: float = 0.0, info_gain: float = 0.0) -> float:
        goal_padded = np.zeros(self.w_goal.shape[1], dtype=float)
        goal_padded[: min(goal.size, goal_padded.size)] = goal[: goal_padded.size]
        h = np.tanh(self.w_state @ predicted_state + self.w_goal @ goal_padded + self.b)
        raw_score = float(self.w_score @ h + self.b_score)
        return raw_score - self.alpha_risk * risk - self.gamma_cost * cost + self.delta_info * info_gain

    def copy(self) -> "Critic":
        cr = Critic.__new__(Critic)
        cr.w_state = self.w_state.copy()
        cr.w_goal = self.w_goal.copy()
        cr.b = self.b.copy()
        cr.w_score = self.w_score.copy()
        cr.b_score = float(self.b_score)
        cr.alpha_risk = float(self.alpha_risk)
        cr.gamma_cost = float(self.gamma_cost)
        cr.delta_info = float(self.delta_info)
        return cr


class Selector:
    __slots__ = ("K", "lateral_w", "rho", "beta", "prev_z")

    def __init__(self, K: int, rho: float = 0.5, beta: float = 0.3) -> None:
        self.K = K
        self.lateral_w = np.zeros((K, K), dtype=float)
        self.rho = rho
        self.beta = beta
        self.prev_z = np.zeros(K, dtype=float)

    def select(self, scores: np.ndarray) -> tuple[int, np.ndarray]:
        z = scores.copy()
        for _ in range(10):
            inhibition = self.lateral_w @ z
            z = np.tanh(scores - self.beta * inhibition + self.rho * self.prev_z)
        self.prev_z = z.copy()
        selected = int(np.argmax(z))
        return selected, z

    def update_lateral(self, activations: np.ndarray, lr: float = 0.005) -> None:
        for i in range(self.K):
            for j in range(self.K):
                if i != j:
                    self.lateral_w[i, j] += lr * activations[i] * activations[j]
        np.fill_diagonal(self.lateral_w, 0.0)
        self.lateral_w = np.clip(self.lateral_w, -0.5, 0.5)

    def copy(self) -> "Selector":
        sel = Selector.__new__(Selector)
        sel.K = self.K
        sel.lateral_w = self.lateral_w.copy()
        sel.rho = float(self.rho)
        sel.beta = float(self.beta)
        sel.prev_z = self.prev_z.copy()
        return sel


# ── M₁ Module ────────────────────────────────────────────────────────────────




@dataclass(frozen=True)
class ModuleActionTrace:
    """Decision-time hidden state used for action-specific neuron attribution."""

    module_name: str
    gate: float
    hidden: np.ndarray
    raw_output: np.ndarray


@dataclass(frozen=True)
class StepTrace:
    trace_id: str
    timestamp: float
    raw_obs: np.ndarray
    encoded_state: np.ndarray
    memory_before: np.ndarray
    memory_after: np.ndarray
    selected_idx: int
    selected_action: np.ndarray
    candidate_actions: np.ndarray
    candidate_logits: np.ndarray
    candidate_scores: np.ndarray
    selected_action_log_prob: float
    predicted_next_state: np.ndarray
    critic_value: float
    confidence: float
    novelty: float
    conflict: float
    risk: float
    gate_values: tuple[float, ...]
    prediction_error: float
    td_delta: float
    action_details: dict[str, object] = field(default_factory=dict)
    positions_pipe: str = ""
    positions_pipe_after: str = ""
    # Populated by the MT5 bridge after it has observed whether the EA actually
    # accepted/executed the action.  Empty keeps direct simulator calls intact.
    execution_details: dict[str, object] = field(default_factory=dict)
    experience_starvation: dict[str, float] = field(default_factory=dict)
    account_context: dict[str, float] = field(default_factory=dict)
    account_context_after: dict[str, float] = field(default_factory=dict)
    # Captured at decision time.  Later lifecycle credit must never infer a
    # neuron's state from current runtime activation.
    module_action_traces: tuple[ModuleActionTrace, ...] = ()
    # Graph-M1 evidence is immutable attribution metadata.  IDs are preserved
    # across journal/replay but never used to mutate a baseline topology.
    active_neuron_ids: tuple[str, ...] = ()
    active_edge_ids: tuple[str, ...] = ()
    graph_edge_gates: dict[str, float] = field(default_factory=dict)
    graph_region_activity: dict[str, float] = field(default_factory=dict)
    # V18 keeps immutable activity evidence for *every* graph, not only the
    # primary M1 module.  Keys are telemetry graph IDs (for example
    # ``m1:m1_thinking`` and ``reasoning_graph``), while local IDs remain
    # unchanged inside each record for lifecycle credit.
    graph_traces: dict[str, dict[str, object]] = field(default_factory=dict)
    memory_match_quality: float = 0.0
    lifecycle_tags: tuple[str, ...] = ()
    module_need_reason: str = ""
    # Immutable decision context needed to keep temporal memory and bridge
    # credit aligned with the exact tick that selected the action.
    context_frame_id: str = ""
    context_schema_version: int = 0
    canonical_context_key: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float))
    allowed_action_mask: tuple[bool, ...] = ()
    goal_context: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    learning_drive_context: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    context_audit: dict[str, object] = field(default_factory=dict)


@dataclass
class CuriosityNeuronSystem:
    input_size: int = 64
    hidden_size: int = 32
    output_size: int = 32
    w_in: np.ndarray = field(default=None, repr=False)
    w_rec: np.ndarray = field(default=None, repr=False)
    b_hidden: np.ndarray = field(default=None, repr=False)
    w_out: np.ndarray = field(default=None, repr=False)
    b_out: np.ndarray = field(default=None, repr=False)
    state: np.ndarray = field(default=None, repr=False)
    plasticity: float = 0.05
    trust: float = 0.1
    age: int = 0

    @classmethod
    def seed(
        cls,
        input_size: int = 64,
        hidden_size: int = 32,
        output_size: int = 32,
        rng: np.random.Generator | None = None,
    ) -> "CuriosityNeuronSystem":
        rng = rng or np.random.default_rng()
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            w_in=rng.normal(0.0, scale, size=(hidden_size, input_size)),
            w_rec=rng.normal(0.0, 0.03, size=(hidden_size, hidden_size)),
            b_hidden=np.zeros(hidden_size, dtype=float),
            w_out=rng.normal(0.0, 0.08, size=(output_size, hidden_size)),
            b_out=np.zeros(output_size, dtype=float),
            state=np.zeros(hidden_size, dtype=float),
        )

    def ensure_compatible(self) -> None:
        if self.w_in is None or self.w_rec is None or self.w_out is None:
            seeded = CuriosityNeuronSystem.seed(self.input_size, self.hidden_size, self.output_size)
            self.w_in = seeded.w_in
            self.w_rec = seeded.w_rec
            self.b_hidden = seeded.b_hidden
            self.w_out = seeded.w_out
            self.b_out = seeded.b_out
            self.state = seeded.state
        if self.state is None or self.state.size != self.hidden_size:
            self.state = np.zeros(self.hidden_size, dtype=float)
        if not hasattr(self, "plasticity"):
            self.plasticity = 0.05
        if not hasattr(self, "trust"):
            self.trust = 0.1
        if not hasattr(self, "age"):
            self.age = 0

    def copy(self) -> "CuriosityNeuronSystem":
        self.ensure_compatible()
        return CuriosityNeuronSystem(
            input_size=int(self.input_size),
            hidden_size=int(self.hidden_size),
            output_size=int(self.output_size),
            w_in=self.w_in.copy(),
            w_rec=self.w_rec.copy(),
            b_hidden=self.b_hidden.copy(),
            w_out=self.w_out.copy(),
            b_out=self.b_out.copy(),
            state=self.state.copy(),
            plasticity=float(self.plasticity),
            trust=float(self.trust),
            age=int(self.age),
        )

    @staticmethod
    def _resize(values: np.ndarray, size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = np.zeros(size, dtype=float)
        if arr.size:
            out[: min(size, arr.size)] = arr[:size]
        return out

    def context_from(
        self,
        *,
        state: np.ndarray,
        action: np.ndarray | None = None,
        memory_query: dict[str, object] | None = None,
        causal_signals: list[dict[str, object]] | None = None,
        starvation: dict[str, float] | None = None,
    ) -> np.ndarray:
        self.ensure_compatible()
        memory_query = memory_query or {}
        causal_signals = causal_signals or []
        starvation = starvation or {}
        context = np.zeros(self.input_size, dtype=float)
        cursor = 0
        for block in (
            self._resize(state, 24),
            self._resize(action if action is not None else np.zeros(0), 12),
            self._resize(np.asarray(memory_query.get("action_bias", []), dtype=float), 12),
            self._resize(np.asarray(memory_query.get("memory_context", []), dtype=float), 8),
            self._resize(np.asarray([
                starvation.get("no_trade_streak", 0.0),
                starvation.get("unfilled_streak", 0.0),
                starvation.get("cadence_deficit", 0.0),
                starvation.get("starvation", 0.0),
                starvation.get("intrinsic_reward", 0.0),
            ], dtype=float), 5),
        ):
            kept = min(block.size, self.input_size - cursor)
            if kept <= 0:
                break
            context[cursor:cursor + kept] = block[:kept]
            cursor += kept
        if cursor < self.input_size:
            severities = [float(item.get("severity", 0.0)) for item in causal_signals[:4]]
            utilities = [float(item.get("utility", 0.0)) for item in causal_signals[:4]]
            tags = [1.0 if item.get("tags") else 0.0 for item in causal_signals[:4]]
            tail = np.asarray(severities + utilities + tags, dtype=float)
            kept = min(tail.size, self.input_size - cursor)
            context[cursor:cursor + kept] = tail[:kept]
        return np.clip(context, -5.0, 5.0)

    def forward(self, context: np.ndarray) -> dict[str, np.ndarray | float]:
        self.ensure_compatible()
        x = self._resize(context, self.input_size)
        self.state = np.tanh(self.w_in @ x + self.w_rec @ self.state + self.b_hidden)
        raw = self.w_out @ self.state + self.b_out
        out = np.tanh(raw)
        return {
            "raw": raw,
            "proposal": out,
            "action_delta": out[:12].copy(),
            "gate_delta": out[12:20].copy(),
            "recurrent_delta": out[20:28].copy(),
            "growth_delta": out[28:32].copy(),
            "drive": float((out[28] + 1.0) * 0.5) if out.size > 28 else 0.0,
            "risk_dampen": float((out[29] + 1.0) * 0.5) if out.size > 29 else 0.0,
            "memory_write": float((out[30] + 1.0) * 0.5) if out.size > 30 else 0.0,
        }

    def observe(self, context: np.ndarray, proposal: np.ndarray, reward: float, td_delta: float, prediction_error: float) -> None:
        self.ensure_compatible()
        x = self._resize(context, self.input_size)
        y = self._resize(proposal, self.output_size)
        signal = float(np.tanh(float(reward) + float(td_delta) - 0.1 * float(prediction_error)))
        lr = float(np.clip(self.plasticity * (0.25 + abs(signal)), 0.0001, 0.02))
        self.w_out += lr * signal * np.outer(y, self.state)
        self.b_out += lr * signal * y
        self.w_in += 0.25 * lr * signal * np.outer(self.state, x)
        self.w_rec += 0.1 * lr * signal * np.outer(self.state, self.state)
        self.trust = float(np.clip(0.995 * self.trust + 0.005 * max(0.0, signal), 0.0, 2.0))
        self.plasticity = float(np.clip(0.995 * self.plasticity + 0.005 * abs(signal), 0.005, 0.5))
        self.age += 1

    def propose_adjustments(self, context: np.ndarray) -> dict[str, np.ndarray | float]:
        return self.forward(context)


DEVELOPMENT_ACTIONS = ("hold", "grow", "prune", "split", "merge", "consolidate")
DEVELOPMENT_ROLE_LIBRARY = (
    (
        ROLE_PREDICTOR_SPECIALIST,
        ("state_encoder", "working_memory"),
        ("critic",),
    ),
    (
        ROLE_MEMORY_SPECIALIST,
        ("state_encoder",),
        ("predictor", "selector"),
    ),
    (
        ROLE_CRITIC_SPECIALIST,
        ("predictor",),
        ("selector",),
    ),
    (
        ROLE_HYPOTHESIS_SPECIALIST,
        ("state_encoder", "working_memory"),
        ("selector",),
    ),
    (
        ROLE_FEATURE_SPECIALIST,
        (),
        ("state_encoder",),
    ),
    (
        ROLE_GENERAL_SPECIALIST,
        ("state_encoder",),
        ("selector",),
    ),
)


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = arr - float(np.max(arr)) if arr.size else arr
    exp = np.exp(np.clip(arr, -40.0, 40.0))
    total = float(np.sum(exp))
    if total <= 0.0:
        return np.ones_like(exp) / max(1, exp.size)
    return exp / total


@dataclass
class NeuralDevelopmentOutput:
    action: str
    action_probs: np.ndarray
    role: str
    role_probs: np.ndarray
    input_sources: tuple[str, ...]
    output_targets: tuple[str, ...]
    initial_gate_bias: float
    initial_plasticity: float
    edge_open_strength: float
    residual_scale: float
    raw: np.ndarray = field(repr=False)

    def as_spec(self) -> ModuleSpec:
        return ModuleSpec(
            role=self.role,
            input_sources=list(self.input_sources),
            output_targets=list(self.output_targets),
            initial_gate_bias=float(self.initial_gate_bias),
            initial_plasticity=float(self.initial_plasticity),
        )

    def summary(self) -> dict[str, object]:
        return {
            "action": self.action,
            "action_confidence": float(np.max(self.action_probs)) if self.action_probs.size else 0.0,
            "role": self.role,
            "role_confidence": float(np.max(self.role_probs)) if self.role_probs.size else 0.0,
            "initial_gate_bias": float(self.initial_gate_bias),
            "initial_plasticity": float(self.initial_plasticity),
            "edge_open_strength": float(self.edge_open_strength),
            "residual_scale": float(self.residual_scale),
        }


@dataclass
class NeuralDevelopmentController:
    input_size: int = 32
    hidden_size: int = 24
    output_size: int = 18
    w_in: np.ndarray = field(default=None, repr=False)
    w_rec: np.ndarray = field(default=None, repr=False)
    b_hidden: np.ndarray = field(default=None, repr=False)
    w_out: np.ndarray = field(default=None, repr=False)
    b_out: np.ndarray = field(default=None, repr=False)
    state: np.ndarray = field(default=None, repr=False)
    plasticity: float = 0.01
    age: int = 0

    @classmethod
    def seed(
        cls,
        input_size: int = 32,
        hidden_size: int = 24,
        output_size: int = 18,
        rng: np.random.Generator | None = None,
    ) -> "NeuralDevelopmentController":
        rng = rng or np.random.default_rng()
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        controller = cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            w_in=rng.normal(0.0, scale, size=(hidden_size, input_size)),
            w_rec=rng.normal(0.0, 0.03, size=(hidden_size, hidden_size)),
            b_hidden=np.zeros(hidden_size, dtype=float),
            w_out=rng.normal(0.0, 0.08, size=(output_size, hidden_size)),
            b_out=np.zeros(output_size, dtype=float),
            state=np.zeros(hidden_size, dtype=float),
        )
        controller.b_out[0] = 0.35
        controller.b_out[1] = 0.05
        return controller

    def ensure_compatible(self) -> None:
        if self.w_in is None or self.w_rec is None or self.w_out is None:
            seeded = NeuralDevelopmentController.seed(self.input_size, self.hidden_size, self.output_size)
            self.w_in = seeded.w_in
            self.w_rec = seeded.w_rec
            self.b_hidden = seeded.b_hidden
            self.w_out = seeded.w_out
            self.b_out = seeded.b_out
            self.state = seeded.state
        if self.state is None or self.state.size != self.hidden_size:
            self.state = np.zeros(self.hidden_size, dtype=float)
        if not hasattr(self, "plasticity"):
            self.plasticity = 0.01
        if not hasattr(self, "age"):
            self.age = 0

    def copy(self) -> "NeuralDevelopmentController":
        self.ensure_compatible()
        return NeuralDevelopmentController(
            input_size=int(self.input_size),
            hidden_size=int(self.hidden_size),
            output_size=int(self.output_size),
            w_in=self.w_in.copy(),
            w_rec=self.w_rec.copy(),
            b_hidden=self.b_hidden.copy(),
            w_out=self.w_out.copy(),
            b_out=self.b_out.copy(),
            state=self.state.copy(),
            plasticity=float(self.plasticity),
            age=int(self.age),
        )

    @staticmethod
    def _resize(values: np.ndarray, size: int) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = np.zeros(size, dtype=float)
        out[: min(size, arr.size)] = arr[:size]
        return out

    def context_from(
        self,
        signal: GrowthSignal,
        modules: list["ModuleState"],
        memory_summary: dict[str, object] | None,
        reward_modulator: float,
    ) -> np.ndarray:
        memory_summary = memory_summary or {}
        scores = np.asarray([m.score for m in modules], dtype=float) if modules else np.zeros(1)
        utilities = np.asarray([m.utility for m in modules], dtype=float) if modules else np.zeros(1)
        gates = np.asarray([m.gate_ema for m in modules], dtype=float) if modules else np.zeros(1)
        plasticities = np.asarray([m.plasticity for m in modules], dtype=float) if modules else np.zeros(1)
        context = np.array([
            signal.error,
            signal.novelty,
            signal.conflict,
            signal.representation_need,
            signal.complexity,
            signal.score,
            np.tanh(float(reward_modulator)),
            len(modules) / 16.0,
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.mean(utilities)),
            float(np.min(utilities)),
            float(np.mean(gates)),
            float(np.mean(plasticities)),
            float(memory_summary.get("neurons", 0)) / 1000.0,
            float(memory_summary.get("clusters", 0)) / 256.0,
            float(memory_summary.get("strategies", 0)) / 96.0,
            float(memory_summary.get("meta_strategies", 0)) / 32.0,
            float(memory_summary.get("positive_neurons", 0)) / max(1.0, float(memory_summary.get("neurons", 1))),
            float(memory_summary.get("causal_nodes", 0)) / 1000.0,
        ], dtype=float)
        return np.clip(self._resize(context, self.input_size), -5.0, 5.0)

    def forward(self, context: np.ndarray) -> NeuralDevelopmentOutput:
        self.ensure_compatible()
        x = self._resize(context, self.input_size)
        self.state = np.tanh(self.w_in @ x + self.w_rec @ self.state + self.b_hidden)
        raw = self.w_out @ self.state + self.b_out
        action_probs = _softmax(raw[:len(DEVELOPMENT_ACTIONS)])
        role_start = len(DEVELOPMENT_ACTIONS)
        role_end = role_start + len(DEVELOPMENT_ROLE_LIBRARY)
        role_probs = _softmax(raw[role_start:role_end])
        action = DEVELOPMENT_ACTIONS[int(np.argmax(action_probs))]
        role_idx = int(np.argmax(role_probs))
        role, inputs, outputs = DEVELOPMENT_ROLE_LIBRARY[role_idx]
        gate_raw = raw[role_end] if raw.size > role_end else 0.0
        plasticity_raw = raw[role_end + 1] if raw.size > role_end + 1 else 0.0
        edge_raw = raw[role_end + 2] if raw.size > role_end + 2 else 0.0
        residual_raw = raw[role_end + 3] if raw.size > role_end + 3 else 0.0
        return NeuralDevelopmentOutput(
            action=action,
            action_probs=action_probs.copy(),
            role=role,
            role_probs=role_probs.copy(),
            input_sources=tuple(inputs),
            output_targets=tuple(outputs),
            initial_gate_bias=float(-2.5 + 2.0 / (1.0 + np.exp(-np.clip(gate_raw, -10.0, 10.0)))),
            initial_plasticity=float(0.5 + 2.0 / (1.0 + np.exp(-np.clip(plasticity_raw, -10.0, 10.0)))),
            edge_open_strength=float(2.0 / (1.0 + np.exp(-np.clip(edge_raw, -10.0, 10.0)))),
            residual_scale=float(0.02 / (1.0 + np.exp(-np.clip(residual_raw, -10.0, 10.0)))),
            raw=raw.copy(),
        )

    def observe(self, context: np.ndarray, output: NeuralDevelopmentOutput, structural_reward: float) -> None:
        self.ensure_compatible()
        signal = float(np.tanh(structural_reward))
        lr = float(np.clip(self.plasticity * (0.25 + abs(signal)), 0.00005, 0.01))
        action_target = -output.action_probs
        action_target[int(np.argmax(output.action_probs))] += 1.0
        role_target = -output.role_probs
        role_target[int(np.argmax(output.role_probs))] += 1.0
        grad = np.zeros(self.output_size, dtype=float)
        grad[:len(DEVELOPMENT_ACTIONS)] = signal * action_target
        role_start = len(DEVELOPMENT_ACTIONS)
        grad[role_start:role_start + len(DEVELOPMENT_ROLE_LIBRARY)] = signal * role_target
        self.w_out += lr * np.outer(grad, self.state)
        self.b_out += lr * grad
        self.w_in += 0.1 * lr * signal * np.outer(self.state, self._resize(context, self.input_size))
        self.w_rec += 0.05 * lr * signal * np.outer(self.state, self.state)
        self.plasticity = float(np.clip(0.995 * self.plasticity + 0.005 * abs(signal), 0.001, 0.2))
        self.age += 1


@dataclass
class NeuralScalarNetwork:
    input_size: int
    hidden_size: int
    output_size: int
    w_in: np.ndarray = field(default=None, repr=False)
    w_rec: np.ndarray = field(default=None, repr=False)
    b_hidden: np.ndarray = field(default=None, repr=False)
    w_out: np.ndarray = field(default=None, repr=False)
    b_out: np.ndarray = field(default=None, repr=False)
    state: np.ndarray = field(default=None, repr=False)
    plasticity: float = 0.01
    age: int = 0

    @classmethod
    def seed(
        cls,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator | None = None,
    ) -> "NeuralScalarNetwork":
        rng = rng or np.random.default_rng()
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        return cls(
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            w_in=rng.normal(0.0, scale, size=(hidden_size, input_size)),
            w_rec=rng.normal(0.0, 0.03, size=(hidden_size, hidden_size)),
            b_hidden=np.zeros(hidden_size, dtype=float),
            w_out=rng.normal(0.0, 0.08, size=(output_size, hidden_size)),
            b_out=np.zeros(output_size, dtype=float),
            state=np.zeros(hidden_size, dtype=float),
        )

    def ensure_compatible(self) -> None:
        if self.w_in is None or self.w_rec is None or self.w_out is None:
            seeded = NeuralScalarNetwork.seed(self.input_size, self.hidden_size, self.output_size)
            self.w_in = seeded.w_in
            self.w_rec = seeded.w_rec
            self.b_hidden = seeded.b_hidden
            self.w_out = seeded.w_out
            self.b_out = seeded.b_out
            self.state = seeded.state
        if self.state is None or self.state.size != self.hidden_size:
            self.state = np.zeros(self.hidden_size, dtype=float)
        if not hasattr(self, "plasticity"):
            self.plasticity = 0.01
        if not hasattr(self, "age"):
            self.age = 0

    def copy(self) -> "NeuralScalarNetwork":
        self.ensure_compatible()
        kwargs = dict(
            input_size=int(self.input_size),
            hidden_size=int(self.hidden_size),
            output_size=int(self.output_size),
            w_in=self.w_in.copy(),
            w_rec=self.w_rec.copy(),
            b_hidden=self.b_hidden.copy(),
            w_out=self.w_out.copy(),
            b_out=self.b_out.copy(),
            state=self.state.copy(),
            plasticity=float(self.plasticity),
            age=int(self.age),
        )
        if hasattr(self, "labels"):
            kwargs["labels"] = tuple(getattr(self, "labels"))
        return self.__class__(**kwargs)

    def _resize(self, values: np.ndarray) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = np.zeros(self.input_size, dtype=float)
        out[: min(arr.size, self.input_size)] = arr[:self.input_size]
        return np.clip(out, -8.0, 8.0)

    def forward(self, values: np.ndarray) -> np.ndarray:
        self.ensure_compatible()
        x = self._resize(values)
        self.state = np.tanh(self.w_in @ x + self.w_rec @ self.state + self.b_hidden)
        return self.w_out @ self.state + self.b_out

    def observe(self, values: np.ndarray, target: np.ndarray, strength: float = 1.0) -> None:
        self.ensure_compatible()
        x = self._resize(values)
        raw = self.forward(x)
        target_arr = np.asarray(target, dtype=float).reshape(-1)
        padded = np.zeros(self.output_size, dtype=float)
        padded[: min(target_arr.size, self.output_size)] = target_arr[:self.output_size]
        error = np.clip(padded - raw, -2.0, 2.0)
        lr = float(np.clip(self.plasticity * abs(float(strength)), 0.00005, 0.02))
        self.w_out += lr * np.outer(error, self.state)
        self.b_out += lr * error
        self.w_in += 0.05 * lr * np.outer(np.tanh(self.state), x)
        self.w_rec += 0.02 * lr * np.outer(self.state, self.state)
        self.plasticity = float(np.clip(0.995 * self.plasticity + 0.005 * float(np.mean(np.abs(error))), 0.001, 0.2))
        self.age += 1


@dataclass
class NeuralUtilityNetwork(NeuralScalarNetwork):
    @classmethod
    def seed(cls, rng: np.random.Generator | None = None) -> "NeuralUtilityNetwork":
        base = NeuralScalarNetwork.seed(24, 16, 2, rng)
        return cls(**base.__dict__)

    def evaluate_window(self, features: np.ndarray) -> tuple[float, float]:
        raw = self.forward(features)
        utility = float(np.tanh(raw[0]))
        trust = float(1.0 / (1.0 + np.exp(-np.clip(raw[1], -10.0, 10.0)))) if raw.size > 1 else 0.5
        return utility, trust


@dataclass
class NeuralRiskCostGate(NeuralScalarNetwork):
    @classmethod
    def seed(cls, rng: np.random.Generator | None = None) -> "NeuralRiskCostGate":
        base = NeuralScalarNetwork.seed(32, 16, 3, rng)
        gate = cls(**base.__dict__)
        gate.b_out[:] = np.array([0.0, 0.0, 0.0])
        return gate

    def gate(self, features: np.ndarray) -> dict[str, float]:
        raw = self.forward(features)
        risk = float(1.0 / (1.0 + np.exp(-np.clip(raw[0], -10.0, 10.0))))
        cost = float(1.0 / (1.0 + np.exp(-np.clip(raw[1], -10.0, 10.0))))
        patience = float(1.0 / (1.0 + np.exp(-np.clip(raw[2], -10.0, 10.0))))
        return {"risk": risk, "cost": cost, "patience": patience}


@dataclass
class NeuralActionParameterHead(NeuralScalarNetwork):
    @classmethod
    def seed(cls, rng: np.random.Generator | None = None) -> "NeuralActionParameterHead":
        base = NeuralScalarNetwork.seed(64, 24, 8, rng)
        head = cls(**base.__dict__)
        head.b_out[:] = 0.0
        return head

    def parameters(self, features: np.ndarray) -> np.ndarray:
        return np.tanh(self.forward(features))


@dataclass
class NeuralPositionOutcomeClassifier(NeuralScalarNetwork):
    labels: tuple[str, ...] = (
        "gave_back_profit",
        "near_tp_giveback",
        "near_tp_missed",
        "closed_after_near_tp_bad",
        "should_have_partial_or_trail",
    )

    @classmethod
    def seed(cls, rng: np.random.Generator | None = None) -> "NeuralPositionOutcomeClassifier":
        base = NeuralScalarNetwork.seed(16, 16, 5, rng)
        clf = cls(**base.__dict__)
        clf.w_out[:] = 0.0
        clf.b_out[:] = -2.0
        # Biological prior encoded as synaptic weights; it is trainable and can be overridden by experience.
        clf.w_in[:, :] *= 0.2
        return clf

    def features_from(
        self,
        target_position: dict[str, object],
        after_position: dict[str, object] | None,
        reward: float,
        td_delta: float,
        utility: float,
    ) -> np.ndarray:
        mfe_r = float(target_position.get("mfe_r", target_position.get("unrealized_r", 0.0)) or 0.0)
        unrealized = float(target_position.get("unrealized_r", 0.0) or 0.0)
        after_unrealized = float(after_position.get("unrealized_r", unrealized)) if after_position else unrealized
        giveback_r = float(target_position.get("giveback_r", max(0.0, mfe_r - unrealized)) or 0.0)
        if after_position is not None:
            giveback_r = max(giveback_r, mfe_r - after_unrealized)
        min_dist_tp = float(target_position.get("min_dist_tp_atr", 999.0) or 999.0)
        min_dist_sl = float(target_position.get("min_dist_sl_atr", 999.0) or 999.0)
        near_tp_flag = 1.0 if bool(target_position.get("near_tp", False)) else 0.0
        closed_after = 1.0 if after_position is None else 0.0
        return np.array([
            np.tanh(mfe_r),
            np.tanh(giveback_r),
            np.tanh(unrealized),
            np.tanh(after_unrealized),
            np.tanh(float(reward)),
            np.tanh(float(td_delta)),
            np.tanh(float(utility)),
            near_tp_flag,
            np.clip(1.0 / (1.0 + max(0.0, min_dist_tp)), 0.0, 1.0),
            np.clip(1.0 / (1.0 + max(0.0, min_dist_sl)), 0.0, 1.0),
            closed_after,
            np.tanh(float(target_position.get("age_bars", 0.0) or 0.0) / 100.0),
        ], dtype=float)

    def classify(self, features: np.ndarray) -> tuple[list[str], np.ndarray]:
        raw = self.forward(features)
        probs = 1.0 / (1.0 + np.exp(-np.clip(raw, -10.0, 10.0)))
        tags = [label for label, prob in zip(self.labels, probs) if float(prob) >= 0.55]
        return tags, probs


# ── ModuleState (Enhanced) ────────────────────────────────────────────────────


@dataclass
class ModuleState:
    name: str
    role: str
    input_size: int
    hidden_size: int
    output_size: int
    w_in: np.ndarray
    w_rec: np.ndarray
    b_hidden: np.ndarray
    w_out: np.ndarray
    b_out: np.ndarray
    gate_w: np.ndarray
    w_lat: np.ndarray = field(default=None, repr=False)
    deep_w: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    deep_rec: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    deep_b: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    deep_state: tuple[np.ndarray, ...] = field(default_factory=tuple, repr=False)
    # Routing masks are persisted with a specialist module.  They make the
    # neural-development input/output specification operational rather than
    # metadata-only.
    input_mask: np.ndarray = field(default=None, repr=False)
    output_mask: np.ndarray = field(default=None, repr=False)
    neuron_utility: np.ndarray = field(default=None, repr=False)
    neuron_advantage_ema: np.ndarray = field(default=None, repr=False)
    neuron_contribution_ema: np.ndarray = field(default=None, repr=False)
    neuron_activation_ema: np.ndarray = field(default=None, repr=False)
    neuron_redundancy_ema: np.ndarray = field(default=None, repr=False)
    neuron_samples: np.ndarray = field(default=None, repr=False)
    neuron_frozen: np.ndarray = field(default=None, repr=False)
    input_eligibility: np.ndarray = field(default=None, repr=False)
    credit_ema: float = 0.0
    gate_b: float = 0.0
    plasticity: float = 1.0
    age: int = 0
    score: float = 0.0
    utility: float = 0.0
    gate_ema: float = 0.0
    activation_ema: float = 0.0
    lifecycle: str = MODULE_LIFECYCLE_DORMANT
    memory_decay: float = 0.85

    p_stp: np.ndarray = field(default=None, repr=False)
    stp_rho: float = 0.95
    stp_alpha: float = 0.1
    hidden_normalization_mode: str = REQUIRED_HIDDEN_NORMALIZATION_MODE
    hidden_normalization_target_rms: float = 1.0
    hidden_normalization_clip: float = 3.0
    hidden_normalization_eps: float = 1e-5
    hidden_normalization_enabled: bool = True
    hidden_normalization_layers: tuple[bool, ...] = field(default_factory=tuple)
    hidden_dropout_enabled: bool = True
    hidden_dropout_rates: tuple[float, ...] = REQUIRED_HIDDEN_DROPOUT_RATES
    hidden_dropout_seed: int = 0
    hidden_dropout_train_only: bool = True
    hidden_dropout_step: int = field(default=0, repr=False)
    # The output path may be dropout-masked while recurrent memory remains
    # unmasked. Credit must consume the same vector that produced the output.
    last_output_hidden: np.ndarray | None = field(default=None, repr=False)
    plasticity_input_normalization_mode: str = "soft_clip_v1"
    plasticity_update_clip: float = 0.25
    plasticity_weight_norm_cap: float = 0.0
    plasticity_update_count: int = 0
    plasticity_update_ema: float = 0.0
    plasticity_update_var: float = 0.0
    plasticity_last_stats: dict[str, float] = field(default_factory=dict, repr=False)

    stdp_params: STDPParams = field(default=None, repr=False)
    plasticity_coeffs: PlasticityCoefficients = field(default=None, repr=False)
    last_spike_time: float = 0.0
    spike_history: list = field(default_factory=list, repr=False)

    m1_component: M1GraphModule | None = field(default=None, repr=False)

    @classmethod
    def create(
        cls,
        name: str,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rng: np.random.Generator,
        role: str = "specialist",
        hidden_layers: int = 1,
    ) -> "ModuleState":
        layer_count = max(1, int(hidden_layers))
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        w_in = rng.normal(0.0, scale, size=(hidden_size, input_size))
        w_rec = rng.normal(0.0, 0.05, size=(hidden_size, hidden_size))
        deep_scale = 1.0 / max(1.0, np.sqrt(float(hidden_size)))
        deep_w = tuple(
            rng.normal(0.0, deep_scale, size=(hidden_size, hidden_size))
            for _ in range(layer_count - 1)
        )
        deep_rec = tuple(
            rng.normal(0.0, 0.035, size=(hidden_size, hidden_size))
            for _ in range(layer_count - 1)
        )
        deep_b = tuple(
            rng.normal(0.0, 0.04, size=hidden_size)
            for _ in range(layer_count - 1)
        )
        return cls(
            name=name,
            role=role,
            input_size=input_size,
            hidden_size=hidden_size,
            output_size=output_size,
            w_in=w_in,
            w_rec=w_rec,
            w_lat=rng.normal(0.0, 0.02, size=(hidden_size, hidden_size)),
            deep_w=deep_w,
            deep_rec=deep_rec,
            deep_b=deep_b,
            deep_state=tuple(np.zeros(hidden_size, dtype=float) for _ in range(layer_count - 1)),
            hidden_normalization_layers=tuple(True for _ in range(layer_count)),
            hidden_dropout_rates=layerwise_rates(REQUIRED_HIDDEN_DROPOUT_RATES, layer_count),
            hidden_dropout_seed=int(rng.integers(0, 2**31 - 1)),
            input_mask=np.ones(input_size, dtype=float),
            output_mask=np.ones(output_size, dtype=float),
            neuron_utility=np.zeros(hidden_size, dtype=float),
            neuron_advantage_ema=np.zeros(hidden_size, dtype=float),
            neuron_contribution_ema=np.zeros(hidden_size, dtype=float),
            neuron_activation_ema=np.zeros(hidden_size, dtype=float),
            neuron_redundancy_ema=np.zeros(hidden_size, dtype=float),
            neuron_samples=np.zeros(hidden_size, dtype=float),
            neuron_frozen=np.zeros(hidden_size, dtype=bool),
            input_eligibility=np.zeros((hidden_size, input_size), dtype=float),
            b_hidden=rng.normal(0.0, 0.05, size=hidden_size),
            w_out=rng.normal(0.0, 0.25, size=(output_size, hidden_size)),
            b_out=rng.normal(0.0, 0.05, size=output_size),
            gate_w=rng.normal(0.0, scale, size=input_size),
            p_stp=np.ones((hidden_size, input_size), dtype=float),
            plasticity_weight_norm_cap=max(1.0, float(np.sqrt(hidden_size) * 1.5)),
            stdp_params=STDPParams(),
            plasticity_coeffs=PlasticityCoefficients(),
        )

    def copy(self) -> "ModuleState":
        return ModuleState(
            name=self.name,
            role=self.role,
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size,
            w_in=self.w_in.copy(),
            w_rec=self.w_rec.copy(),
            w_lat=self.w_lat.copy() if self.w_lat is not None else None,
            deep_w=tuple(layer.copy() for layer in getattr(self, "deep_w", ()) or ()),
            deep_rec=tuple(layer.copy() for layer in getattr(self, "deep_rec", ()) or ()),
            deep_b=tuple(layer.copy() for layer in getattr(self, "deep_b", ()) or ()),
            deep_state=tuple(layer.copy() for layer in getattr(self, "deep_state", ()) or ()),
            input_mask=self.input_mask.copy() if self.input_mask is not None else None,
            output_mask=self.output_mask.copy() if self.output_mask is not None else None,
            neuron_utility=self.neuron_utility.copy() if self.neuron_utility is not None else None,
            neuron_advantage_ema=self.neuron_advantage_ema.copy() if self.neuron_advantage_ema is not None else None,
            neuron_contribution_ema=self.neuron_contribution_ema.copy() if self.neuron_contribution_ema is not None else None,
            neuron_activation_ema=self.neuron_activation_ema.copy() if self.neuron_activation_ema is not None else None,
            neuron_redundancy_ema=self.neuron_redundancy_ema.copy() if self.neuron_redundancy_ema is not None else None,
            neuron_samples=self.neuron_samples.copy() if self.neuron_samples is not None else None,
            neuron_frozen=self.neuron_frozen.copy() if self.neuron_frozen is not None else None,
            input_eligibility=self.input_eligibility.copy() if self.input_eligibility is not None else None,
            credit_ema=float(self.credit_ema),
            b_hidden=self.b_hidden.copy(),
            w_out=self.w_out.copy(),
            b_out=self.b_out.copy(),
            gate_w=self.gate_w.copy(),
            gate_b=float(self.gate_b),
            plasticity=float(self.plasticity),
            age=int(self.age),
            score=float(self.score),
            utility=float(self.utility),
            gate_ema=float(self.gate_ema),
            activation_ema=float(self.activation_ema),
            lifecycle=self.lifecycle,
            memory_decay=float(self.memory_decay),
            p_stp=self.p_stp.copy() if self.p_stp is not None else None,
            stp_rho=float(self.stp_rho),
            stp_alpha=float(self.stp_alpha),
            hidden_normalization_mode=_mandatory_hidden_normalization_mode(
                getattr(self, "hidden_normalization_mode", REQUIRED_HIDDEN_NORMALIZATION_MODE)
            ),
            hidden_normalization_target_rms=float(getattr(self, "hidden_normalization_target_rms", 1.0) or 1.0),
            hidden_normalization_clip=float(getattr(self, "hidden_normalization_clip", 3.0) or 3.0),
            hidden_normalization_eps=float(getattr(self, "hidden_normalization_eps", 1e-5) or 1e-5),
            hidden_normalization_enabled=True,
            hidden_normalization_layers=tuple(True for _ in range(1 + len(getattr(self, "deep_w", ()) or ()))),
            hidden_dropout_enabled=True,
            hidden_dropout_rates=mandatory_layerwise_rates(
                getattr(self, "hidden_dropout_rates", REQUIRED_HIDDEN_DROPOUT_RATES),
                1 + len(getattr(self, "deep_w", ()) or ()),
            ),
            hidden_dropout_seed=max(0, int(getattr(self, "hidden_dropout_seed", 0) or 0)),
            hidden_dropout_train_only=True,
            hidden_dropout_step=max(0, int(getattr(self, "hidden_dropout_step", 0) or 0)),
            last_output_hidden=(
                np.asarray(getattr(self, "last_output_hidden", None), dtype=float).copy()
                if getattr(self, "last_output_hidden", None) is not None else None
            ),
            plasticity_input_normalization_mode=str(getattr(self, "plasticity_input_normalization_mode", "soft_clip_v1") or "soft_clip_v1"),
            plasticity_update_clip=float(getattr(self, "plasticity_update_clip", 0.25) or 0.25),
            plasticity_weight_norm_cap=float(getattr(self, "plasticity_weight_norm_cap", 0.0) or 0.0),
            plasticity_update_count=int(getattr(self, "plasticity_update_count", 0) or 0),
            plasticity_update_ema=float(getattr(self, "plasticity_update_ema", 0.0) or 0.0),
            plasticity_update_var=float(getattr(self, "plasticity_update_var", 0.0) or 0.0),
            plasticity_last_stats=dict(getattr(self, "plasticity_last_stats", {}) or {}),
            stdp_params=self.stdp_params,
            plasticity_coeffs=self.plasticity_coeffs,
            last_spike_time=float(self.last_spike_time),
            spike_history=list(self.spike_history),
            m1_component=self.m1_component.copy() if self.m1_component is not None else None,
        )

    def _normalize_hidden_layer(self, values: np.ndarray, layer_index: int, layer_count: int) -> np.ndarray:
        self.hidden_normalization_mode = _mandatory_hidden_normalization_mode(
            getattr(self, "hidden_normalization_mode", REQUIRED_HIDDEN_NORMALIZATION_MODE)
        )
        self.hidden_normalization_enabled = True
        self.hidden_normalization_layers = tuple(True for _ in range(max(1, layer_count)))
        return normalize_hidden(
            values,
            mode=self.hidden_normalization_mode,
            target_rms=float(getattr(self, "hidden_normalization_target_rms", 1.0) or 1.0),
            clip=float(getattr(self, "hidden_normalization_clip", 3.0) or 3.0),
            eps=float(getattr(self, "hidden_normalization_eps", 1e-5) or 1e-5),
        )

    def _dropout_hidden(
        self,
        values: np.ndarray,
        layer_index: int,
        layer_count: int,
        *,
        training: bool,
        dropout_step: int,
    ) -> np.ndarray:
        self.hidden_dropout_enabled = True
        self.hidden_dropout_train_only = True
        self.hidden_dropout_rates = mandatory_layerwise_rates(
            getattr(self, "hidden_dropout_rates", REQUIRED_HIDDEN_DROPOUT_RATES),
            max(1, layer_count),
        )
        if not training:
            return np.asarray(values, dtype=float).reshape(-1).copy()
        rates = layerwise_rates(getattr(self, "hidden_dropout_rates", ()), layer_count)
        rate = rates[min(max(0, int(layer_index)), layer_count - 1)]
        return deterministic_dropout(
            values,
            rate,
            seed=max(0, int(getattr(self, "hidden_dropout_seed", 0) or 0)),
            step=dropout_step,
            layer_index=layer_index,
        )

    def activate(
        self,
        observation: np.ndarray,
        previous_hidden: np.ndarray | None = None,
        *,
        training: bool = False,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        if previous_hidden is None or previous_hidden.size != self.hidden_size:
            previous_hidden = np.zeros(self.hidden_size, dtype=float)
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(obs.size, self.input_size)] = obs[:self.input_size]
            obs = padded
        if self.input_mask is None or self.input_mask.size != self.input_size:
            self.input_mask = np.ones(self.input_size, dtype=float)
        if self.output_mask is None or self.output_mask.size != self.output_size:
            self.output_mask = np.ones(self.output_size, dtype=float)
        masked_obs = obs * np.clip(self.input_mask, 0.0, 1.0)
        if self.p_stp is None:
            self.p_stp = np.ones_like(self.w_in)

        self.p_stp = np.clip(self.stp_rho * self.p_stp + self.stp_alpha * np.abs(masked_obs).reshape(1, -1), 0.01, 5.0)
        effective_w = self.w_in * self.p_stp

        recurrent = self.w_rec @ previous_hidden
        deep_w = tuple(getattr(self, "deep_w", ()) or ())
        layer_count = 1 + len(deep_w)
        hidden_preactivation = self._normalize_hidden_layer(
            effective_w @ masked_obs + recurrent + self.b_hidden,
            0,
            layer_count,
        )
        hidden = np.tanh(hidden_preactivation)
        hidden = self.memory_decay * previous_hidden + (1.0 - self.memory_decay) * hidden

        deep_rec = tuple(getattr(self, "deep_rec", ()) or ())
        deep_b = tuple(getattr(self, "deep_b", ()) or ())
        deep_state = tuple(getattr(self, "deep_state", ()) or ())
        if len(deep_rec) != len(deep_w):
            deep_rec = tuple(np.zeros((self.hidden_size, self.hidden_size), dtype=float) for _ in deep_w)
        if len(deep_b) != len(deep_w):
            deep_b = tuple(np.zeros(self.hidden_size, dtype=float) for _ in deep_w)
        if len(deep_state) != len(deep_w):
            deep_state = tuple(np.zeros(self.hidden_size, dtype=float) for _ in deep_w)

        # The regularization contract is always configured.  Only frozen
        # inference skips sampling a mask, preserving deterministic replay.
        dropout_active = bool(training)
        if dropout_active:
            self.hidden_dropout_step = max(0, int(getattr(self, "hidden_dropout_step", 0) or 0)) + 1
        dropout_step = max(0, int(getattr(self, "hidden_dropout_step", 0) or 0))
        next_deep_state: list[np.ndarray] = []
        layer_input = self._dropout_hidden(
            hidden,
            0,
            layer_count,
            training=bool(training),
            dropout_step=dropout_step,
        )
        final_hidden = hidden
        for deep_index, (layer_w, layer_rec, layer_b, layer_prev) in enumerate(zip(deep_w, deep_rec, deep_b, deep_state), start=1):
            if layer_w.shape != (self.hidden_size, self.hidden_size):
                continue
            if layer_rec.shape != (self.hidden_size, self.hidden_size):
                layer_rec = np.zeros((self.hidden_size, self.hidden_size), dtype=float)
            layer_prev = np.asarray(layer_prev, dtype=float).reshape(-1)
            if layer_prev.size != self.hidden_size:
                layer_prev = np.zeros(self.hidden_size, dtype=float)
            layer_b = np.asarray(layer_b, dtype=float).reshape(-1)
            if layer_b.size != self.hidden_size:
                layer_b = np.zeros(self.hidden_size, dtype=float)
            layer_preactivation = self._normalize_hidden_layer(
                layer_w @ layer_input + layer_rec @ layer_prev + layer_b,
                deep_index,
                layer_count,
            )
            layer_candidate = np.tanh(layer_preactivation)
            layer_hidden = self.memory_decay * layer_prev + (1.0 - self.memory_decay) * layer_candidate
            next_deep_state.append(layer_hidden)
            final_hidden = layer_hidden
            layer_input = self._dropout_hidden(
                layer_hidden,
                deep_index,
                layer_count,
                training=bool(training),
                dropout_step=dropout_step,
            )
        self.deep_state = tuple(next_deep_state)
        hidden = final_hidden

        raw_gate = float(self.gate_w @ masked_obs + self.gate_b)
        gate = 1.0 / (1.0 + np.exp(-np.clip(raw_gate, -60.0, 60.0)))
        output_hidden = layer_input if deep_w else self._dropout_hidden(
            hidden,
            0,
            layer_count,
            training=bool(training),
            dropout_step=dropout_step,
        )
        self.last_output_hidden = np.asarray(output_hidden, dtype=float).copy()
        output = (self.w_out @ output_hidden + self.b_out) * np.clip(self.output_mask, 0.0, 1.0)
        return gate, hidden, output

    def mutate(self, rng: np.random.Generator, sigma: float) -> None:
        strength = sigma * max(0.05, self.plasticity)
        self.w_in += rng.normal(0.0, strength, size=self.w_in.shape)
        self.w_rec += rng.normal(0.0, strength * 0.35, size=self.w_rec.shape)
        if self.w_lat is not None:
            self.w_lat += rng.normal(0.0, strength * 0.2, size=self.w_lat.shape)
            self.w_lat *= 0.995
        if getattr(self, "deep_w", None):
            self.deep_w = tuple(layer + rng.normal(0.0, strength * 0.45, size=layer.shape) for layer in self.deep_w)
            self.deep_rec = tuple(layer + rng.normal(0.0, strength * 0.2, size=layer.shape) for layer in self.deep_rec)
            self.deep_b = tuple(layer + rng.normal(0.0, strength * 0.45, size=layer.shape) for layer in self.deep_b)
        self.b_hidden += rng.normal(0.0, strength, size=self.b_hidden.shape)
        self.w_out += rng.normal(0.0, strength, size=self.w_out.shape)
        self.b_out += rng.normal(0.0, strength, size=self.b_out.shape)
        self.gate_w += rng.normal(0.0, strength * 0.5, size=self.gate_w.shape)
        self.gate_b += float(rng.normal(0.0, strength * 0.5))
        self.plasticity = float(np.clip(self.plasticity * rng.uniform(0.92, 1.08), 0.05, 2.5))
        self.memory_decay = float(np.clip(self.memory_decay + rng.normal(0.0, 0.02), 0.1, 0.98))
        self.w_rec *= 0.995

        if self.m1_component is not None:
            self.m1_component.mutate(rng, sigma)

        self.age += 1

    def apply_plasticity_rules(
        self,
        pre_activation: np.ndarray,
        post_activation: np.ndarray,
        grad_loss_w: np.ndarray | None = None,
        current_time: float = 0.0,
        reward_modulator: float = 0.0,
    ) -> None:
        if self.plasticity_coeffs is None:
            return
        coeffs = self.plasticity_coeffs

        modulator = np.clip(reward_modulator, -2.0, 2.0)
        if abs(modulator) < 0.01:
            return

        stdp_matrix_total = np.zeros_like(self.w_in)
        if self.stdp_params is not None and len(self.spike_history) > 0:
            post = np.tanh(np.nan_to_num(
                np.asarray(post_activation, dtype=float).flatten()[:self.hidden_size],
                nan=0.0,
                posinf=20.0,
                neginf=-20.0,
            ))
            for past_time, past_pre in self.spike_history[-5:]:
                dt = current_time - past_time
                stdp_delta = stdp_update_scalar(dt, self.stdp_params)
                if abs(stdp_delta) > 1e-6:
                    pre = normalize_plasticity_vector(
                        past_pre,
                        size=self.input_size,
                        mode=str(getattr(self, "plasticity_input_normalization_mode", "soft_clip_v1") or "soft_clip_v1"),
                    )
                    stdp_matrix = stdp_delta * np.outer(
                        np.pad(post, (0, max(0, self.hidden_size - post.size)))[:self.hidden_size],
                        np.pad(pre, (0, max(0, self.input_size - pre.size)))[:self.input_size],
                    )
                    if stdp_matrix.shape == self.w_in.shape:
                        stdp_matrix_total += stdp_matrix

        delta, normalized_pre, stats = compose_stable_plasticity_update(
            self.w_in,
            pre_activation,
            post_activation,
            grad_loss_w,
            coeffs,
            modulator,
            input_mode=str(getattr(self, "plasticity_input_normalization_mode", "soft_clip_v1") or "soft_clip_v1"),
            max_update_norm=float(getattr(self, "plasticity_update_clip", 0.25) or 0.25),
            stdp_delta=stdp_matrix_total,
        )
        update = delta * float(np.clip(self.plasticity, 0.05, 2.5))
        candidate_weights = self.w_in + update
        weight_cap = float(getattr(self, "plasticity_weight_norm_cap", 0.0) or 0.0)
        if weight_cap <= 0.0:
            weight_cap = max(1.0, float(np.sqrt(self.hidden_size) * 1.5))
            self.plasticity_weight_norm_cap = weight_cap
        candidate_norm = float(np.linalg.norm(candidate_weights))
        if candidate_norm > weight_cap:
            candidate_weights *= weight_cap / max(candidate_norm, 1e-12)
            update = candidate_weights - self.w_in
        self.w_in = candidate_weights
        self.w_in = np.nan_to_num(np.clip(self.w_in, -4.0, 4.0), nan=0.0, posinf=4.0, neginf=-4.0)
        update_norm = float(np.linalg.norm(update))
        count = int(getattr(self, "plasticity_update_count", 0) or 0)
        ema = float(getattr(self, "plasticity_update_ema", 0.0) or 0.0)
        variance = float(getattr(self, "plasticity_update_var", 0.0) or 0.0)
        alpha = 0.05
        diff = update_norm - ema
        ema = ema + alpha * diff
        variance = (1.0 - alpha) * (variance + alpha * diff * diff)
        self.plasticity_update_count = count + 1
        self.plasticity_update_ema = float(ema)
        self.plasticity_update_var = float(max(0.0, variance))
        stats["scaled_update_norm"] = update_norm
        stats["weight_norm"] = float(np.linalg.norm(self.w_in))
        stats["update_count"] = float(self.plasticity_update_count)
        stats["update_ema"] = self.plasticity_update_ema
        stats["update_var"] = self.plasticity_update_var
        self.plasticity_last_stats = {key: float(value) for key, value in stats.items()}
        self.last_spike_time = current_time
        self.spike_history.append((current_time, normalized_pre.copy()))
        if len(self.spike_history) > 20:
            self.spike_history = self.spike_history[-20:]

    def observe_usage(self, gate: float, hidden: np.ndarray, fitness: float | None = None) -> None:
        activation = float(np.mean(np.abs(hidden))) if hidden.size else 0.0
        self.gate_ema = 0.95 * self.gate_ema + 0.05 * float(gate)
        self.activation_ema = 0.95 * self.activation_ema + 0.05 * activation
        if fitness is not None:
            self.utility = 0.9 * self.utility + 0.1 * (float(gate) * float(fitness))
        if self.utility > 0.0 and self.gate_ema > 0.02:
            self.lifecycle = MODULE_LIFECYCLE_ACTIVE
        if self.utility > 0.0 and self.age >= 5 and self.plasticity < 0.35:
            self.lifecycle = MODULE_LIFECYCLE_CONSOLIDATED
        if self.gate_ema < 0.005 and self.age >= 5:
            self.lifecycle = MODULE_LIFECYCLE_PRUNING

    def apply_homeostatic_plasticity(self, target_activation: float = 0.35) -> None:
        activation_error = target_activation - self.activation_ema
        self.b_hidden += 0.01 * activation_error
        norm = float(np.linalg.norm(self.w_in))
        if norm > 0.0:
            self.w_in *= min(1.0, np.sqrt(self.w_in.size) / norm)
        rec_norm = float(np.linalg.norm(self.w_rec))
        if rec_norm > 0.0:
            self.w_rec *= min(1.0, np.sqrt(self.w_rec.size) * 0.25 / rec_norm)
        if self.gate_ema > 0.85 and self.activation_ema > target_activation * 1.8:
            self.gate_b -= 0.02
            self.plasticity = min(2.5, self.plasticity * 1.02)
        elif self.gate_ema < 0.02 and self.activation_ema < target_activation * 0.25:
            self.gate_b += 0.01


# ── Gating Functions ─────────────────────────────────────────────────────────


def sparsemax(logits: np.ndarray) -> np.ndarray:
    sorted_logits = np.sort(logits)[::-1]
    cumsum = np.cumsum(sorted_logits)
    k = np.arange(1, len(sorted_logits) + 1)
    threshold = (cumsum - 1.0) / k
    support = sorted_logits > threshold
    if not np.any(support):
        return np.ones_like(logits) / len(logits)
    k_star = int(np.max(np.where(support)))
    tau = threshold[k_star]
    return np.maximum(logits - tau, 0.0)


def top_k_gating(logits: np.ndarray, k: int = 2) -> np.ndarray:
    gates = np.zeros_like(logits)
    top_indices = np.argsort(logits)[-k:]
    top_values = logits[top_indices]
    exp_values = np.exp(top_values - np.max(top_values))
    softmax_values = exp_values / np.sum(exp_values)
    gates[top_indices] = softmax_values
    return gates


@dataclass
class MemoryNeuron:
    node_id: str
    trace_id: str
    obs_embedding: np.ndarray
    action_embedding: np.ndarray
    next_embedding: np.ndarray
    reward_ema: float
    td_ema: float
    prediction_error_ema: float
    strength: float = 1.0
    utility: float = 0.0
    forget_gate: float = 0.02
    plasticity: float = 0.1
    recurrent_state: float = 0.0
    age: int = 0
    support_count: int = 1
    memory_key_schema_version: int = MEMORY_KEY_SCHEMA_VERSION
    account_context: dict[str, float] = field(default_factory=dict)
    account_context_after: dict[str, float] = field(default_factory=dict)
    position_context: np.ndarray = field(default_factory=lambda: np.zeros(POSITION_MEMORY_KEY_SIZE, dtype=float), repr=False)
    # Multi-timescale context turns a raw snapshot into an episode signature.
    context_delta: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    short_context: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    regime_context: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    outcome_variance: float = 0.0
    uncertainty: float = 1.0
    last_seen_step: int = 0
    # Detailed memory upgrade: distinguish policy decisions from bootstrap,
    # risk overrides and forced management. Non-policy episodes remain useful
    # evidence, but they cannot directly train or promote the policy action path.
    action_origin: str = "policy"
    raw_utility: float = 0.0
    credit_weight: float = 1.0
    promotion_state: str = "tentative"
    support_policy_count: float = 0.0
    support_bootstrap_count: float = 0.0
    counterfactual_action_values: dict[str, float] = field(default_factory=dict)
    causal_lesson: dict[str, object] = field(default_factory=dict)

    def copy(self) -> "MemoryNeuron":
        return MemoryNeuron(
            node_id=self.node_id,
            trace_id=self.trace_id,
            obs_embedding=self.obs_embedding.copy(),
            action_embedding=self.action_embedding.copy(),
            next_embedding=self.next_embedding.copy(),
            reward_ema=float(self.reward_ema),
            td_ema=float(self.td_ema),
            prediction_error_ema=float(self.prediction_error_ema),
            strength=float(self.strength),
            utility=float(self.utility),
            forget_gate=float(self.forget_gate),
            plasticity=float(self.plasticity),
            recurrent_state=float(self.recurrent_state),
            age=int(self.age),
            support_count=int(getattr(self, "support_count", 1)),
            memory_key_schema_version=int(getattr(self, "memory_key_schema_version", 1)),
            account_context={str(k): float(v) for k, v in getattr(self, "account_context", {}).items()},
            account_context_after={str(k): float(v) for k, v in getattr(self, "account_context_after", {}).items()},
            position_context=np.asarray(getattr(self, "position_context", np.zeros(POSITION_MEMORY_KEY_SIZE)), dtype=float).copy(),
            context_delta=np.asarray(getattr(self, "context_delta", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            short_context=np.asarray(getattr(self, "short_context", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            regime_context=np.asarray(getattr(self, "regime_context", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            evidence_for=float(getattr(self, "evidence_for", max(0.0, getattr(self, "utility", 0.0)))),
            evidence_against=float(getattr(self, "evidence_against", max(0.0, -getattr(self, "utility", 0.0)))),
            outcome_variance=float(getattr(self, "outcome_variance", 0.0)),
            uncertainty=float(getattr(self, "uncertainty", 1.0)),
            last_seen_step=int(getattr(self, "last_seen_step", 0)),
            action_origin=str(getattr(self, "action_origin", "policy") or "policy"),
            raw_utility=float(getattr(self, "raw_utility", getattr(self, "utility", 0.0))),
            credit_weight=float(getattr(self, "credit_weight", 1.0)),
            promotion_state=str(getattr(self, "promotion_state", "tentative") or "tentative"),
            support_policy_count=float(getattr(self, "support_policy_count", 1.0 if str(getattr(self, "action_origin", "policy")) == "policy" else 0.0)),
            support_bootstrap_count=float(getattr(self, "support_bootstrap_count", 1.0 if str(getattr(self, "action_origin", "policy")) == "bootstrap" else 0.0)),
            counterfactual_action_values={str(k): float(v) for k, v in dict(getattr(self, "counterfactual_action_values", {}) or {}).items()},
            causal_lesson=copylib.deepcopy(dict(getattr(self, "causal_lesson", {}) or {})),
        )


@dataclass
class MemoryConnection:
    source: str
    target: str
    kind: str
    weight: float
    plasticity: float = 0.05
    utility: float = 0.0
    age: int = 0

    def copy(self) -> "MemoryConnection":
        return MemoryConnection(
            source=self.source,
            target=self.target,
            kind=self.kind,
            weight=float(self.weight),
            plasticity=float(self.plasticity),
            utility=float(self.utility),
            age=int(self.age),
        )


@dataclass
class NeuralMemoryCell:
    """Generic neural substrate cell for every memory function.

    Episode, cluster, concept, strategy, meta-strategy, causal and reflection
    memory are represented as neural populations connected by synapses.  The
    typed dataclasses above remain as compatibility views so existing training,
    MT5 and tests keep working, but the canonical substrate is a graph of these
    cells and ``NeuralMemorySynapse`` objects.
    """

    cell_id: str
    role: str
    vector: np.ndarray
    activation: float = 0.0
    bias: float = 0.0
    utility: float = 0.0
    plasticity: float = 0.05
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    uncertainty: float = 1.0
    source_kind: str = ""
    source_ref: str = ""
    meta: dict[str, object] = field(default_factory=dict)
    recurrent_state: float = 0.0
    age: int = 0

    def copy(self) -> "NeuralMemoryCell":
        return NeuralMemoryCell(
            cell_id=self.cell_id,
            role=self.role,
            vector=np.asarray(self.vector, dtype=float).copy(),
            activation=float(self.activation),
            bias=float(self.bias),
            utility=float(self.utility),
            plasticity=float(self.plasticity),
            evidence_for=float(self.evidence_for),
            evidence_against=float(self.evidence_against),
            uncertainty=float(self.uncertainty),
            source_kind=str(self.source_kind),
            source_ref=str(self.source_ref),
            meta=copylib.deepcopy(dict(self.meta or {})),
            recurrent_state=float(self.recurrent_state),
            age=int(self.age),
        )


@dataclass
class NeuralMemorySynapse:
    source: str
    target: str
    kind: str
    weight: float
    plasticity: float = 0.05
    utility: float = 0.0
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    enabled: bool = True
    meta: dict[str, object] = field(default_factory=dict)
    age: int = 0

    def copy(self) -> "NeuralMemorySynapse":
        return NeuralMemorySynapse(
            source=str(self.source),
            target=str(self.target),
            kind=str(self.kind),
            weight=float(self.weight),
            plasticity=float(self.plasticity),
            utility=float(self.utility),
            evidence_for=float(self.evidence_for),
            evidence_against=float(self.evidence_against),
            enabled=bool(self.enabled),
            meta=copylib.deepcopy(dict(self.meta or {})),
            age=int(self.age),
        )


@dataclass
class MemoryCluster:
    cluster_id: str
    member_ids: list[str]
    centroid: np.ndarray
    action_proto: np.ndarray
    utility: float
    strength: float = 1.0
    plasticity: float = 0.08
    forget_gate: float = 0.015
    age: int = 0
    context_delta_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    short_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    regime_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    uncertainty: float = 1.0
    policy_support: float = 0.0
    bootstrap_support: float = 0.0
    contradiction_rate: float = 0.0
    promotion_state: str = "shadow"

    def copy(self) -> "MemoryCluster":
        return MemoryCluster(
            cluster_id=self.cluster_id,
            member_ids=list(self.member_ids),
            centroid=self.centroid.copy(),
            action_proto=self.action_proto.copy(),
            utility=float(self.utility),
            strength=float(self.strength),
            plasticity=float(self.plasticity),
            forget_gate=float(self.forget_gate),
            age=int(self.age),
            context_delta_proto=np.asarray(getattr(self, "context_delta_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            short_context_proto=np.asarray(getattr(self, "short_context_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            regime_context_proto=np.asarray(getattr(self, "regime_context_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            evidence_for=float(getattr(self, "evidence_for", max(0.0, self.utility))),
            evidence_against=float(getattr(self, "evidence_against", max(0.0, -self.utility))),
            uncertainty=float(getattr(self, "uncertainty", 1.0)),
            policy_support=float(getattr(self, "policy_support", 0.0)),
            bootstrap_support=float(getattr(self, "bootstrap_support", 0.0)),
            contradiction_rate=float(getattr(self, "contradiction_rate", 0.0)),
            promotion_state=str(getattr(self, "promotion_state", "shadow") or "shadow"),
        )


@dataclass
class StrategyMemory:
    strategy_id: str
    cluster_ids: list[str]
    context_proto: np.ndarray
    action_proto: np.ndarray
    utility: float
    confidence: float = 0.1
    plasticity: float = 0.05
    recurrent_state: float = 0.0
    age: int = 0
    context_delta_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    short_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    regime_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    uncertainty: float = 1.0
    policy_support: float = 0.0
    bootstrap_support: float = 0.0
    contradiction_rate: float = 0.0
    promotion_state: str = "shadow"

    def copy(self) -> "StrategyMemory":
        return StrategyMemory(
            strategy_id=self.strategy_id,
            cluster_ids=list(self.cluster_ids),
            context_proto=self.context_proto.copy(),
            action_proto=self.action_proto.copy(),
            utility=float(self.utility),
            confidence=float(self.confidence),
            plasticity=float(self.plasticity),
            recurrent_state=float(self.recurrent_state),
            age=int(self.age),
            context_delta_proto=np.asarray(getattr(self, "context_delta_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            short_context_proto=np.asarray(getattr(self, "short_context_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            regime_context_proto=np.asarray(getattr(self, "regime_context_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            evidence_for=float(getattr(self, "evidence_for", max(0.0, self.utility))),
            evidence_against=float(getattr(self, "evidence_against", max(0.0, -self.utility))),
            uncertainty=float(getattr(self, "uncertainty", 1.0)),
            policy_support=float(getattr(self, "policy_support", 0.0)),
            bootstrap_support=float(getattr(self, "bootstrap_support", 0.0)),
            contradiction_rate=float(getattr(self, "contradiction_rate", 0.0)),
            promotion_state=str(getattr(self, "promotion_state", "shadow") or "shadow"),
        )


@dataclass
class MetaStrategyMemory:
    meta_id: str
    strategy_ids: list[str]
    context_proto: np.ndarray
    utility: float
    growth_bias: float = 0.0
    prune_bias: float = 0.0
    trust: float = 0.1
    age: int = 0
    context_delta_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    short_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    regime_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    uncertainty: float = 1.0

    def copy(self) -> "MetaStrategyMemory":
        return MetaStrategyMemory(
            meta_id=self.meta_id,
            strategy_ids=list(self.strategy_ids),
            context_proto=self.context_proto.copy(),
            utility=float(self.utility),
            growth_bias=float(self.growth_bias),
            prune_bias=float(self.prune_bias),
            trust=float(self.trust),
            age=int(self.age),
            context_delta_proto=np.asarray(getattr(self, "context_delta_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            short_context_proto=np.asarray(getattr(self, "short_context_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            regime_context_proto=np.asarray(getattr(self, "regime_context_proto", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
            evidence_for=float(getattr(self, "evidence_for", max(0.0, self.utility))),
            evidence_against=float(getattr(self, "evidence_against", max(0.0, -self.utility))),
            uncertainty=float(getattr(self, "uncertainty", 1.0)),
        )


@dataclass
class MemoryConcept:
    """Semantic memory node distilled from repeated episodic clusters.

    Concepts are not raw trades.  They are evidence-bearing abstractions such as
    ``policy_positive_low_uncertainty`` or ``loss_high_prediction_error`` that
    can drive growth only after policy-origin support and low contradiction.
    """

    concept_id: str
    name: str
    cluster_ids: list[str]
    context_proto: np.ndarray
    action_proto: np.ndarray
    utility: float = 0.0
    evidence_for: float = 0.0
    evidence_against: float = 0.0
    policy_support: float = 0.0
    bootstrap_support: float = 0.0
    contradiction_rate: float = 0.0
    uncertainty: float = 1.0
    confidence: float = 0.0
    promotion_state: str = "shadow"
    best_action: int = 0
    worst_action: int = 0
    causal_profile: dict[str, float] = field(default_factory=dict)
    counterfactual_values: dict[str, float] = field(default_factory=dict)
    related_neurons: list[str] = field(default_factory=list)
    related_strategies: list[str] = field(default_factory=list)
    context_delta_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    short_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    regime_context_proto: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    age: int = 0

    def copy(self) -> "MemoryConcept":
        return MemoryConcept(
            concept_id=self.concept_id,
            name=self.name,
            cluster_ids=list(self.cluster_ids),
            context_proto=self.context_proto.copy(),
            action_proto=self.action_proto.copy(),
            utility=float(self.utility),
            evidence_for=float(self.evidence_for),
            evidence_against=float(self.evidence_against),
            policy_support=float(self.policy_support),
            bootstrap_support=float(self.bootstrap_support),
            contradiction_rate=float(self.contradiction_rate),
            uncertainty=float(self.uncertainty),
            confidence=float(self.confidence),
            promotion_state=str(self.promotion_state),
            best_action=int(self.best_action),
            worst_action=int(self.worst_action),
            causal_profile={str(k): float(v) for k, v in dict(self.causal_profile or {}).items()},
            counterfactual_values={str(k): float(v) for k, v in dict(self.counterfactual_values or {}).items()},
            related_neurons=list(self.related_neurons),
            related_strategies=list(self.related_strategies),
            context_delta_proto=np.asarray(self.context_delta_proto, dtype=float).copy(),
            short_context_proto=np.asarray(self.short_context_proto, dtype=float).copy(),
            regime_context_proto=np.asarray(self.regime_context_proto, dtype=float).copy(),
            age=int(self.age),
        )


@dataclass
class ExperienceSuperNetwork:
    neurons: dict[str, MemoryNeuron] = field(default_factory=dict)
    clusters: dict[str, MemoryCluster] = field(default_factory=dict)
    concepts: dict[str, MemoryConcept] = field(default_factory=dict)
    strategies: dict[str, StrategyMemory] = field(default_factory=dict)
    meta_strategies: dict[str, MetaStrategyMemory] = field(default_factory=dict)
    causal_nodes: dict[str, dict[str, object]] = field(default_factory=dict)
    connections: list[MemoryConnection] = field(default_factory=list)
    neural_cells: dict[str, NeuralMemoryCell] = field(default_factory=dict)
    neural_synapses: list[NeuralMemorySynapse] = field(default_factory=list)
    neural_substrate_version: int = 5
    pure_neural_memory: bool = True
    # ``None`` means no fixed architecture cap. Evidence and contradiction,
    # not a count ceiling, determine consolidation and retirement.
    max_neurons: int | None = None
    max_clusters: int | None = None
    max_strategies: int | None = None
    max_meta_strategies: int | None = None
    max_connections: int | None = None
    duplicate_similarity: float = 0.992
    duplicate_action_similarity: float = 0.985
    novelty_utility_threshold: float = 0.08
    memory_key_schema_version: int = MEMORY_KEY_SCHEMA_VERSION
    memory_key_size: int = MEMORY_KEY_SIZE
    next_id: int = 0
    recent_ids: list[str] = field(default_factory=list)
    context_step: int = 0
    previous_context: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    short_context_state: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    regime_context_state: np.ndarray = field(default_factory=lambda: np.zeros(MEMORY_KEY_SIZE, dtype=float), repr=False)
    position_outcome_classifier: NeuralPositionOutcomeClassifier = field(default_factory=NeuralPositionOutcomeClassifier.seed, repr=False)
    reflection_log: list[dict[str, object]] = field(default_factory=list, repr=False)
    memory_origin_stats: dict[str, float] = field(default_factory=dict)

    def ensure_compatible(self) -> None:
        if not hasattr(self, "neurons"):
            self.neurons = {}
        if not hasattr(self, "clusters"):
            self.clusters = {}
        if not hasattr(self, "concepts") or self.concepts is None:
            self.concepts = {}
        if not hasattr(self, "strategies"):
            self.strategies = {}
        if not hasattr(self, "meta_strategies"):
            self.meta_strategies = {}
        if not hasattr(self, "causal_nodes"):
            self.causal_nodes = {}
        if not hasattr(self, "connections"):
            self.connections = []
        if not hasattr(self, "neural_cells") or self.neural_cells is None:
            self.neural_cells = {}
        if not hasattr(self, "neural_synapses") or self.neural_synapses is None:
            self.neural_synapses = []
        if not hasattr(self, "neural_substrate_version"):
            self.neural_substrate_version = 5
        if not hasattr(self, "pure_neural_memory"):
            self.pure_neural_memory = True
        self.pure_neural_memory = bool(getattr(self, "pure_neural_memory", True))
        if not hasattr(self, "max_neurons"):
            self.max_neurons = None
        if not hasattr(self, "max_clusters"):
            self.max_clusters = None
        if not hasattr(self, "max_strategies"):
            self.max_strategies = None
        if not hasattr(self, "max_meta_strategies"):
            self.max_meta_strategies = None
        if not hasattr(self, "max_connections"):
            self.max_connections = None
        # Old artifacts with fixed ceilings are upgraded to evidence-based,
        # open-ended retention rather than silently retaining obsolete caps.
        for capacity_name in ("max_neurons", "max_clusters", "max_strategies", "max_meta_strategies", "max_connections"):
            raw_capacity = getattr(self, capacity_name, None)
            setattr(self, capacity_name, None if raw_capacity is None or int(raw_capacity) <= 0 else None)
        if not hasattr(self, "duplicate_similarity"):
            self.duplicate_similarity = 0.992
        if not hasattr(self, "duplicate_action_similarity"):
            self.duplicate_action_similarity = 0.985
        if not hasattr(self, "novelty_utility_threshold"):
            self.novelty_utility_threshold = 0.08
        if not hasattr(self, "memory_key_schema_version"):
            self.memory_key_schema_version = MEMORY_KEY_SCHEMA_VERSION
        if not hasattr(self, "memory_key_size"):
            self.memory_key_size = MEMORY_KEY_SIZE
        if not hasattr(self, "next_id"):
            self.next_id = 0
        if not hasattr(self, "recent_ids"):
            self.recent_ids = []
        if not hasattr(self, "context_step"):
            self.context_step = 0
        for temporal_name in ("previous_context", "short_context_state", "regime_context_state"):
            value = np.asarray(getattr(self, temporal_name, np.zeros(MEMORY_KEY_SIZE)), dtype=float).reshape(-1)
            setattr(self, temporal_name, self._resize(value, MEMORY_KEY_SIZE))
        if not hasattr(self, "position_outcome_classifier") or self.position_outcome_classifier is None:
            self.position_outcome_classifier = NeuralPositionOutcomeClassifier.seed()
        self.position_outcome_classifier.ensure_compatible()
        if not hasattr(self, "reflection_log") or self.reflection_log is None:
            self.reflection_log = []
        if not hasattr(self, "memory_origin_stats") or self.memory_origin_stats is None:
            self.memory_origin_stats = {}
        for node in self.neurons.values():
            if not hasattr(node, "support_count"):
                node.support_count = 1
            if not hasattr(node, "memory_key_schema_version"):
                node.memory_key_schema_version = 1
            if not hasattr(node, "account_context"):
                node.account_context = {}
            if not hasattr(node, "account_context_after"):
                node.account_context_after = {}
            if not hasattr(node, "position_context") or np.asarray(node.position_context).size != POSITION_MEMORY_KEY_SIZE:
                node.position_context = np.zeros(POSITION_MEMORY_KEY_SIZE, dtype=float)
            for temporal_name in ("context_delta", "short_context", "regime_context"):
                value = np.asarray(getattr(node, temporal_name, np.zeros(MEMORY_KEY_SIZE)), dtype=float).reshape(-1)
                setattr(node, temporal_name, self._resize(value, MEMORY_KEY_SIZE))
            node.evidence_for = float(getattr(node, "evidence_for", max(0.0, node.utility)))
            node.evidence_against = float(getattr(node, "evidence_against", max(0.0, -node.utility)))
            node.outcome_variance = float(getattr(node, "outcome_variance", 0.0))
            node.uncertainty = float(getattr(node, "uncertainty", 1.0))
            node.last_seen_step = int(getattr(node, "last_seen_step", 0))
            node.action_origin = str(getattr(node, "action_origin", "policy") or "policy")
            node.raw_utility = float(getattr(node, "raw_utility", getattr(node, "utility", 0.0)))
            node.credit_weight = float(getattr(node, "credit_weight", 1.0 if node.action_origin == "policy" else 0.0))
            node.promotion_state = str(getattr(node, "promotion_state", "tentative") or "tentative")
            node.support_policy_count = float(getattr(node, "support_policy_count", 1.0 if node.action_origin == "policy" else 0.0))
            node.support_bootstrap_count = float(getattr(node, "support_bootstrap_count", 1.0 if node.action_origin == "bootstrap" else 0.0))
            node.counterfactual_action_values = {str(k): float(v) for k, v in dict(getattr(node, "counterfactual_action_values", {}) or {}).items()}
            node.causal_lesson = copylib.deepcopy(dict(getattr(node, "causal_lesson", {}) or {}))
        for concept in self.concepts.values():
            for temporal_name in ("context_delta_proto", "short_context_proto", "regime_context_proto"):
                value = np.asarray(getattr(concept, temporal_name, np.zeros(MEMORY_KEY_SIZE)), dtype=float).reshape(-1)
                setattr(concept, temporal_name, self._resize(value, MEMORY_KEY_SIZE))
            concept.evidence_for = float(getattr(concept, "evidence_for", max(0.0, concept.utility)))
            concept.evidence_against = float(getattr(concept, "evidence_against", max(0.0, -concept.utility)))
            concept.policy_support = float(getattr(concept, "policy_support", 0.0))
            concept.bootstrap_support = float(getattr(concept, "bootstrap_support", 0.0))
            concept.contradiction_rate = float(getattr(concept, "contradiction_rate", 0.0))
            concept.uncertainty = float(getattr(concept, "uncertainty", 1.0))
            concept.confidence = float(getattr(concept, "confidence", 0.0))
            concept.promotion_state = str(getattr(concept, "promotion_state", "shadow") or "shadow")
            concept.causal_profile = {str(k): float(v) for k, v in dict(getattr(concept, "causal_profile", {}) or {}).items()}
            concept.counterfactual_values = {str(k): float(v) for k, v in dict(getattr(concept, "counterfactual_values", {}) or {}).items()}
            concept.related_neurons = list(getattr(concept, "related_neurons", []) or [])
            concept.related_strategies = list(getattr(concept, "related_strategies", []) or [])
            concept.age = int(getattr(concept, "age", 0))
        for collection in (self.clusters.values(), self.strategies.values(), self.meta_strategies.values()):
            for item in collection:
                for temporal_name in ("context_delta_proto", "short_context_proto", "regime_context_proto"):
                    value = np.asarray(getattr(item, temporal_name, np.zeros(MEMORY_KEY_SIZE)), dtype=float).reshape(-1)
                    setattr(item, temporal_name, self._resize(value, MEMORY_KEY_SIZE))
                item.evidence_for = float(getattr(item, "evidence_for", max(0.0, item.utility)))
                item.evidence_against = float(getattr(item, "evidence_against", max(0.0, -item.utility)))
                item.uncertainty = float(getattr(item, "uncertainty", 1.0))
                item.policy_support = float(getattr(item, "policy_support", 0.0))
                item.bootstrap_support = float(getattr(item, "bootstrap_support", 0.0))
                item.contradiction_rate = float(getattr(item, "contradiction_rate", 0.0))
                item.promotion_state = str(getattr(item, "promotion_state", "shadow") or "shadow")
        self._synchronise_neural_memory_substrate()

    def copy(self) -> "ExperienceSuperNetwork":
        self.ensure_compatible()
        return ExperienceSuperNetwork(
            neurons={key: node.copy() for key, node in self.neurons.items()},
            clusters={key: cluster.copy() for key, cluster in self.clusters.items()},
            concepts={key: concept.copy() for key, concept in self.concepts.items()},
            strategies={key: strategy.copy() for key, strategy in self.strategies.items()},
            meta_strategies={key: meta.copy() for key, meta in self.meta_strategies.items()},
            causal_nodes={key: copylib.deepcopy(value) for key, value in self.causal_nodes.items()},
            connections=[conn.copy() for conn in self.connections],
            neural_cells={key: cell.copy() for key, cell in self.neural_cells.items()},
            neural_synapses=[syn.copy() for syn in self.neural_synapses],
            neural_substrate_version=int(getattr(self, "neural_substrate_version", 4)),
            max_neurons=self.max_neurons,
            max_clusters=self.max_clusters,
            max_strategies=self.max_strategies,
            max_meta_strategies=self.max_meta_strategies,
            max_connections=self.max_connections,
            duplicate_similarity=float(self.duplicate_similarity),
            duplicate_action_similarity=float(self.duplicate_action_similarity),
            novelty_utility_threshold=float(self.novelty_utility_threshold),
            memory_key_schema_version=int(self.memory_key_schema_version),
            memory_key_size=int(self.memory_key_size),
            next_id=int(self.next_id),
            recent_ids=list(self.recent_ids),
            context_step=int(self.context_step),
            previous_context=np.asarray(self.previous_context, dtype=float).copy(),
            short_context_state=np.asarray(self.short_context_state, dtype=float).copy(),
            regime_context_state=np.asarray(self.regime_context_state, dtype=float).copy(),
            position_outcome_classifier=self.position_outcome_classifier.copy() if hasattr(self, "position_outcome_classifier") and self.position_outcome_classifier is not None else NeuralPositionOutcomeClassifier.seed(),
            reflection_log=copylib.deepcopy(list(getattr(self, "reflection_log", []))),
            memory_origin_stats={str(k): float(v) for k, v in dict(getattr(self, "memory_origin_stats", {}) or {}).items()},
        )

    def _cell_id(self, kind: str, ref: str) -> str:
        return f"{kind}:{ref}"

    def _item_vector(self, item: object, *, fallback_size: int | None = None) -> np.ndarray:
        size = int(fallback_size or getattr(self, "memory_key_size", MEMORY_KEY_SIZE))
        for name in ("obs_embedding", "centroid", "context_proto"):
            if hasattr(item, name):
                return self._resize(np.asarray(getattr(item, name), dtype=float), size)
        if isinstance(item, dict):
            payload = item.get("payload", {}) if isinstance(item.get("payload", {}), dict) else {}
            values = []
            for key in sorted(payload)[:size]:
                try:
                    values.append(float(payload[key]))
                except (TypeError, ValueError):
                    values.append(float(len(str(payload[key])) % 17) / 17.0)
            return self._resize(np.asarray(values, dtype=float), size)
        return np.zeros(size, dtype=float)

    def _upsert_neural_cell(self, cell_id: str, role: str, vector: np.ndarray, item: object, *, source_kind: str, source_ref: str) -> None:
        vector = self._resize(np.asarray(vector, dtype=float), getattr(self, "memory_key_size", MEMORY_KEY_SIZE))
        evidence_for = float(getattr(item, "evidence_for", max(0.0, float(getattr(item, "utility", 0.0))))) if not isinstance(item, dict) else max(0.0, float(item.get("utility", 0.0)))
        evidence_against = float(getattr(item, "evidence_against", max(0.0, -float(getattr(item, "utility", 0.0))))) if not isinstance(item, dict) else max(0.0, -float(item.get("utility", 0.0)))
        uncertainty = float(getattr(item, "uncertainty", 1.0)) if not isinstance(item, dict) else float(item.get("uncertainty", 1.0))
        utility = float(getattr(item, "utility", 0.0)) if not isinstance(item, dict) else float(item.get("utility", 0.0))
        activation = float(np.tanh(utility + evidence_for - evidence_against))
        if isinstance(item, dict):
            action_vec = np.zeros(20, dtype=float)
            context_delta = np.zeros(MEMORY_KEY_SIZE, dtype=float)
            short_context = np.zeros(MEMORY_KEY_SIZE, dtype=float)
            regime_context = np.zeros(MEMORY_KEY_SIZE, dtype=float)
            support_strength = float(item.get("strength", item.get("confidence", item.get("trust", 1.0))) or 1.0)
        else:
            action_vec = np.asarray(getattr(item, "action_proto", getattr(item, "action_embedding", np.zeros(20))), dtype=float).reshape(-1)
            context_delta = np.asarray(getattr(item, "context_delta_proto", getattr(item, "context_delta", np.zeros(MEMORY_KEY_SIZE))), dtype=float).reshape(-1)
            short_context = np.asarray(getattr(item, "short_context_proto", getattr(item, "short_context", np.zeros(MEMORY_KEY_SIZE))), dtype=float).reshape(-1)
            regime_context = np.asarray(getattr(item, "regime_context_proto", getattr(item, "regime_context", np.zeros(MEMORY_KEY_SIZE))), dtype=float).reshape(-1)
            support_strength = float(getattr(item, "strength", getattr(item, "confidence", getattr(item, "trust", 1.0))))
        meta = {
            "source_kind": source_kind,
            "source_ref": source_ref,
            "promotion_state": str(getattr(item, "promotion_state", "") if not isinstance(item, dict) else item.get("promotion_state", "")),
            "policy_support": float(getattr(item, "policy_support", getattr(item, "support_policy_count", 0.0)) if not isinstance(item, dict) else item.get("policy_support", 0.0)),
            "bootstrap_support": float(getattr(item, "bootstrap_support", getattr(item, "support_bootstrap_count", 0.0)) if not isinstance(item, dict) else item.get("bootstrap_support", 0.0)),
            "action_vector": self._resize(action_vec, 20).tolist(),
            "context_delta": self._resize(context_delta, MEMORY_KEY_SIZE).tolist(),
            "short_context": self._resize(short_context, MEMORY_KEY_SIZE).tolist(),
            "regime_context": self._resize(regime_context, MEMORY_KEY_SIZE).tolist(),
            "support_strength": float(support_strength),
            "credit_weight": float(getattr(item, "credit_weight", 1.0) if not isinstance(item, dict) else item.get("credit_weight", 1.0)),
        }
        old = self.neural_cells.get(cell_id)
        if old is None:
            self.neural_cells[cell_id] = NeuralMemoryCell(
                cell_id=cell_id, role=role, vector=vector, activation=activation,
                utility=utility, evidence_for=evidence_for, evidence_against=evidence_against,
                uncertainty=uncertainty, source_kind=source_kind, source_ref=source_ref, meta=meta,
            )
        else:
            old.role = role
            old.vector = 0.85 * self._resize(old.vector, vector.size) + 0.15 * vector
            old.activation = activation
            old.utility = utility
            old.evidence_for = evidence_for
            old.evidence_against = evidence_against
            old.uncertainty = uncertainty
            old.source_kind = source_kind
            old.source_ref = source_ref
            old.meta.update(meta)
            old.age += 1

    def _make_neural_synapse(self, source: str, target: str, kind: str, *, weight: float = 1.0, utility: float = 0.0, meta: dict[str, object] | None = None) -> NeuralMemorySynapse:
        return NeuralMemorySynapse(
            source=source, target=target, kind=kind, weight=float(np.clip(weight, -8.0, 8.0)),
            utility=float(utility), evidence_for=max(0.0, float(utility)), evidence_against=max(0.0, -float(utility)),
            meta=dict(meta or {}),
        )

    def _synchronise_neural_memory_substrate(self) -> None:
        """Project all memory views onto a pure neuron/synapse substrate.

        No memory feature is removed: the old typed structures are maintained as
        views for compatibility, while the substrate encodes the same episodes,
        clusters, concepts, strategies, meta strategies, causal nodes and
        reflection states as cells connected by synapses.
        """
        if not hasattr(self, "neural_cells") or self.neural_cells is None:
            self.neural_cells = {}
        if not hasattr(self, "neural_synapses") or self.neural_synapses is None:
            self.neural_synapses = []
        active_ids: set[str] = set()
        synapses: list[NeuralMemorySynapse] = []

        for node_id, node in self.neurons.items():
            cid = self._cell_id("episode", node_id)
            active_ids.add(cid)
            self._upsert_neural_cell(cid, "episodic_memory", self._item_vector(node), node, source_kind="episode", source_ref=node_id)

        for cluster_id, cluster in self.clusters.items():
            cid = self._cell_id("cluster", cluster_id)
            active_ids.add(cid)
            self._upsert_neural_cell(cid, "context_attractor", self._item_vector(cluster), cluster, source_kind="cluster", source_ref=cluster_id)
            for member_id in list(getattr(cluster, "member_ids", []) or []):
                src = self._cell_id("episode", member_id)
                if src in active_ids or member_id in self.neurons:
                    synapses.append(self._make_neural_synapse(src, cid, "episode_to_cluster", weight=1.0, utility=float(getattr(cluster, "utility", 0.0)), meta={"cluster_id": cluster_id}))

        for concept_id, concept in self.concepts.items():
            cid = self._cell_id("concept", concept_id)
            active_ids.add(cid)
            self._upsert_neural_cell(cid, "semantic_concept", self._item_vector(concept), concept, source_kind="concept", source_ref=concept_id)
            for cluster_id in list(getattr(concept, "cluster_ids", []) or []):
                src = self._cell_id("cluster", cluster_id)
                synapses.append(self._make_neural_synapse(src, cid, "cluster_to_concept", weight=max(0.1, float(getattr(concept, "confidence", 0.0))), utility=float(getattr(concept, "utility", 0.0)), meta={"concept_id": concept_id}))
            for neuron_id in list(getattr(concept, "related_neurons", []) or []):
                src = self._cell_id("episode", neuron_id)
                synapses.append(self._make_neural_synapse(src, cid, "episode_to_concept_support", weight=0.35, utility=float(getattr(concept, "utility", 0.0)), meta={"concept_id": concept_id}))

        for strategy_id, strategy in self.strategies.items():
            cid = self._cell_id("strategy", strategy_id)
            active_ids.add(cid)
            self._upsert_neural_cell(cid, "procedural_strategy", self._item_vector(strategy), strategy, source_kind="strategy", source_ref=strategy_id)
            for cluster_id in list(getattr(strategy, "cluster_ids", []) or []):
                src = self._cell_id("cluster", cluster_id)
                synapses.append(self._make_neural_synapse(src, cid, "cluster_to_strategy", weight=max(0.1, float(getattr(strategy, "confidence", 0.0))), utility=float(getattr(strategy, "utility", 0.0)), meta={"strategy_id": strategy_id}))
            for concept in self.concepts.values():
                if strategy_id in list(getattr(concept, "related_strategies", []) or []):
                    src = self._cell_id("concept", concept.concept_id)
                    synapses.append(self._make_neural_synapse(src, cid, "concept_to_strategy", weight=max(0.1, float(getattr(concept, "confidence", 0.0))), utility=float(getattr(strategy, "utility", 0.0)), meta={"strategy_id": strategy_id}))

        for meta_id, meta in self.meta_strategies.items():
            cid = self._cell_id("meta", meta_id)
            active_ids.add(cid)
            self._upsert_neural_cell(cid, "meta_strategy", self._item_vector(meta), meta, source_kind="meta_strategy", source_ref=meta_id)
            for strategy_id in list(getattr(meta, "strategy_ids", []) or []):
                src = self._cell_id("strategy", strategy_id)
                synapses.append(self._make_neural_synapse(src, cid, "strategy_to_meta", weight=max(0.1, float(getattr(meta, "trust", 0.0))), utility=float(getattr(meta, "utility", 0.0)), meta={"meta_id": meta_id}))

        for causal_id, causal in self.causal_nodes.items():
            cid = self._cell_id("causal", causal_id)
            active_ids.add(cid)
            self._upsert_neural_cell(cid, f"causal_{str(causal.get('kind', 'node'))}", self._item_vector(causal), causal, source_kind="causal", source_ref=causal_id)

        for idx, reflection in enumerate(list(getattr(self, "reflection_log", []) or [])[-16:]):
            ref = str(reflection.get("reflection_id", f"recent-{idx}"))
            cid = self._cell_id("reflection", ref)
            active_ids.add(cid)
            utility = float(reflection.get("mean_policy_utility", 0.0) or 0.0)
            item = {"utility": utility, "payload": {"episodes": reflection.get("episodes", 0), "policy_origin": reflection.get("policy_origin", 0)}}
            self._upsert_neural_cell(cid, "reflection_loop", self._item_vector(item), item, source_kind="reflection", source_ref=ref)
            for concept_id in [c.get("concept_id") for c in list(reflection.get("top_concepts", []) or []) if isinstance(c, dict)][:8]:
                if concept_id:
                    synapses.append(self._make_neural_synapse(self._cell_id("concept", str(concept_id)), cid, "concept_to_reflection", weight=0.5, utility=utility, meta={"reflection": ref}))

        # Keep custom synapses that connect still-active cells, then append the
        # rebuilt structural synapses.
        preserved: list[NeuralMemorySynapse] = []
        for syn in list(self.neural_synapses or []):
            if syn.source in active_ids and syn.target in active_ids and syn.kind.startswith("learned_"):
                preserved.append(syn.copy())
        self.neural_cells = {cid: cell for cid, cell in self.neural_cells.items() if cid in active_ids}
        self.neural_synapses = preserved + synapses
        self.neural_substrate_version = 5

    @staticmethod
    def _resize(values: np.ndarray, size: int = 32) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        out = np.zeros(size, dtype=float)
        if arr.size:
            out[: min(size, arr.size)] = arr[:size]
        return out

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, dtype=float).reshape(-1)
        b = np.asarray(b, dtype=float).reshape(-1)
        if a.size != b.size:
            size = max(a.size, b.size)
            aa = np.zeros(size, dtype=float)
            bb = np.zeros(size, dtype=float)
            aa[:a.size] = a
            bb[:b.size] = b
            a, b = aa, bb
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 1e-12:
            return 0.0
        return float(np.clip((a @ b) / denom, -1.0, 1.0))

    @staticmethod
    def _account_context_vector(context: dict[str, object] | None) -> np.ndarray:
        data = dict(context or {})
        values = np.zeros(len(ACCOUNT_CONTEXT_FIELDS), dtype=float)
        for idx, field_name in enumerate(ACCOUNT_CONTEXT_FIELDS):
            try:
                values[idx] = float(data.get(field_name, 0.0) or 0.0)
            except (TypeError, ValueError):
                values[idx] = 0.0
        values = np.clip(values, -2.0, 2.0)
        encoded = np.array([
            values[0],
            values[1],
            0.5 * (values[2] + values[3]),
            values[4],
            values[5],
            values[6],
            values[7],
            0.5 * (values[8] + values[9]) + 0.25 * values[10] + 0.25 * values[11],
        ], dtype=float)
        return np.clip(encoded, -2.0, 2.0)

    def _position_context_vector(self, positions_pipe: str | None) -> np.ndarray:
        positions = self.parse_positions_pipe(positions_pipe)
        if not positions:
            return np.zeros(POSITION_MEMORY_KEY_SIZE, dtype=float)
        count = max(1, len(positions))
        unrealized = np.asarray([float(item.get("unrealized_r", 0.0) or 0.0) for item in positions], dtype=float)
        ages = np.asarray([float(item.get("age_bars", 0.0) or 0.0) for item in positions], dtype=float)
        sides = np.asarray([float(item.get("side", 0.0) or 0.0) for item in positions], dtype=float)
        near_tp = np.asarray([1.0 if bool(item.get("near_tp", False)) else 0.0 for item in positions], dtype=float)
        giveback = np.asarray([float(item.get("giveback_r", 0.0) or 0.0) for item in positions], dtype=float)
        return np.clip(np.array([
            count / 5.0,
            float(np.mean(unrealized)) / 3.0,
            float(np.min(unrealized)) / 3.0,
            float(np.mean(np.clip(ages / 60.0, 0.0, 2.0))),
            float(np.mean(sides)),
            float(np.mean(near_tp) + 0.25 * np.mean(np.clip(giveback, 0.0, 4.0))),
        ], dtype=float), -2.0, 2.0)

    def _memory_key(
        self,
        state_embedding: np.ndarray,
        account_context: dict[str, object] | None = None,
        positions_pipe: str | None = "",
    ) -> np.ndarray:
        market_key = self._resize(state_embedding, MARKET_MEMORY_KEY_SIZE)
        account_key = self._account_context_vector(account_context)
        position_key = self._position_context_vector(positions_pipe)
        has_position = bool(str(positions_pipe or "").strip())
        market_weight, account_weight, position_weight = (0.55, 0.25, 0.20) if has_position else (0.80, 0.20, 0.0)
        return np.concatenate([
            market_weight * market_key,
            account_weight * account_key,
            position_weight * position_key,
        ]).astype(float)

    def _trace_memory_key(self, trace: StepTrace) -> np.ndarray:
        return self._memory_key(
            trace.encoded_state,
            getattr(trace, "account_context", {}),
            getattr(trace, "positions_pipe", ""),
        )

    def _observe_context(self, context: np.ndarray, *, advance: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Represent the current situation at four timescales.

        A memory is not retrieved solely because its raw vector is close.  Its
        immediate transition, short rolling context and slow regime context
        must also be compatible.  This is what lets a neuron survive in a
        repeated *kind* of situation rather than an accidental snapshot.
        """
        current = self._resize(context, self.memory_key_size)
        previous = self._resize(self.previous_context, self.memory_key_size)
        short_before = self._resize(self.short_context_state, self.memory_key_size)
        regime_before = self._resize(self.regime_context_state, self.memory_key_size)
        delta = current - previous
        short = 0.74 * short_before + 0.26 * current
        regime = 0.975 * regime_before + 0.025 * current
        if advance:
            self.context_step += 1
            self.previous_context = current.copy()
            self.short_context_state = short.copy()
            self.regime_context_state = regime.copy()
        return current, delta, short, regime

    def _multi_scale_similarity(
        self,
        current: np.ndarray,
        delta: np.ndarray,
        short: np.ndarray,
        regime: np.ndarray,
        item: object,
        *,
        context_name: str = "centroid",
        delta_name: str = "context_delta_proto",
        short_name: str = "short_context_proto",
        regime_name: str = "regime_context_proto",
    ) -> float:
        base = self._resize(np.asarray(getattr(item, context_name, np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        delta_proto = self._resize(np.asarray(getattr(item, delta_name, np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        short_proto = self._resize(np.asarray(getattr(item, short_name, np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        regime_proto = self._resize(np.asarray(getattr(item, regime_name, np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        similarities = np.array([
            max(0.0, self._cosine(current, base)),
            max(0.0, self._cosine(delta, delta_proto)),
            max(0.0, self._cosine(short, short_proto)),
            max(0.0, self._cosine(regime, regime_proto)),
        ], dtype=float)
        # Empty historical temporal signals should not veto newly migrated memory.
        active = np.array([
            np.linalg.norm(base) > 1e-8,
            np.linalg.norm(delta_proto) > 1e-8,
            np.linalg.norm(short_proto) > 1e-8,
            np.linalg.norm(regime_proto) > 1e-8,
        ], dtype=bool)
        weights = np.array([0.45, 0.15, 0.20, 0.20], dtype=float)
        weights[~active] = 0.0
        if float(weights.sum()) <= 1e-12:
            return 0.0
        return float(np.clip(float(similarities @ weights) / float(weights.sum()), 0.0, 1.0))

    @staticmethod
    def _update_evidence(item: object, utility: float, *, lr: float = 0.10) -> None:
        evidence_for = float(getattr(item, "evidence_for", 0.0)) + max(0.0, utility)
        evidence_against = float(getattr(item, "evidence_against", 0.0)) + max(0.0, -utility)
        total = evidence_for + evidence_against
        prior_mean = (1.0 + evidence_for) / (2.0 + total)
        variance = float(getattr(item, "outcome_variance", 0.0))
        mean_utility = float(getattr(item, "utility", 0.0))
        variance = (1.0 - lr) * variance + lr * (utility - mean_utility) ** 2
        support_uncertainty = 1.0 / (1.0 + total)
        uncertainty = float(np.clip(0.55 * support_uncertainty + 0.45 * np.tanh(variance), 0.0, 1.0))
        setattr(item, "evidence_for", evidence_for)
        setattr(item, "evidence_against", evidence_against)
        setattr(item, "outcome_variance", variance)
        setattr(item, "uncertainty", uncertainty)
        # The posterior mean is available to callers without overwriting local
        # utility, which still encodes magnitude and risk-adjusted reward.
        setattr(item, "evidence_posterior", prior_mean)

    @staticmethod
    def _evidence_reliability(item: object) -> tuple[float, float, float]:
        pos = max(0.0, float(getattr(item, "evidence_for", 0.0)))
        neg = max(0.0, float(getattr(item, "evidence_against", 0.0)))
        total = pos + neg
        posterior = (1.0 + pos) / (2.0 + total)
        direction = 2.0 * posterior - 1.0
        support = 1.0 - np.exp(-total / 4.0)
        uncertainty = float(np.clip(max(float(getattr(item, "uncertainty", 1.0)), 1.0 / (1.0 + total)), 0.0, 1.0))
        reliability = float(np.clip(support * (1.0 - 0.65 * uncertainty), 0.0, 1.0))
        return reliability, direction, uncertainty

    def _new_id(self) -> str:
        self.next_id += 1
        return f"mem-{self.next_id}"

    def _typed_id(self, prefix: str) -> str:
        self.next_id += 1
        return f"{prefix}-{self.next_id}"

    @staticmethod
    def parse_positions_pipe(positions_pipe: str | None) -> list[dict[str, object]]:
        if not positions_pipe:
            return []
        snapshots: list[dict[str, object]] = []
        for slot, item in enumerate(str(positions_pipe).split("^")):
            if not item:
                continue
            parts = item.split(":")
            if len(parts) < 8:
                continue
            try:
                snapshots.append({
                    "slot": slot,
                    "ticket": parts[0],
                    "side": int(float(parts[1])),
                    "volume": float(parts[2]),
                    "entry": float(parts[3]),
                    "sl": float(parts[4]),
                    "tp": float(parts[5]),
                    "unrealized_r": float(parts[6]),
                    "age_bars": float(parts[7]),
                    "mfe_r": float(parts[8]) if len(parts) > 8 else float(parts[6]),
                    "mae_r": float(parts[9]) if len(parts) > 9 else float(parts[6]),
                    "dist_tp_atr": float(parts[10]) if len(parts) > 10 else 0.0,
                    "dist_sl_atr": float(parts[11]) if len(parts) > 11 else 0.0,
                    "min_dist_tp_atr": float(parts[12]) if len(parts) > 12 else 0.0,
                    "min_dist_sl_atr": float(parts[13]) if len(parts) > 13 else 0.0,
                    "bars_since_mfe": float(parts[14]) if len(parts) > 14 else 0.0,
                    "near_tp": bool(float(parts[15])) if len(parts) > 15 else False,
                    "giveback_r": float(parts[16]) if len(parts) > 16 else 0.0,
                    "position_owner_context": str(parts[17]) if len(parts) > 17 else "",
                    "position_owner_expert": str(parts[18]) if len(parts) > 18 else "",
                    "position_owner_policy": str(parts[19]) if len(parts) > 19 else "",
                    "owner_trace_id": str(parts[20]) if len(parts) > 20 else "",
                })
            except ValueError:
                continue
        return snapshots

    def _causal_id(self, prefix: str) -> str:
        return self._typed_id(prefix)

    def _add_causal_node(self, kind: str, payload: dict[str, object], utility: float, strength: float) -> str:
        node_id = self._causal_id(kind)
        self.causal_nodes[node_id] = {
            "kind": kind,
            "payload": copylib.deepcopy(payload),
            "utility": float(utility),
            "strength": float(strength),
            "age": 0,
        }
        return node_id

    def _absorb_causal_details(self, memory_node_id: str, trace: StepTrace, reward: float, cost: float, signal: dict[str, float], utility: float) -> None:
        action_details = dict(getattr(trace, "action_details", {}) or {})
        positions = self.parse_positions_pipe(getattr(trace, "positions_pipe", ""))
        positions_after = self.parse_positions_pipe(getattr(trace, "positions_pipe_after", ""))
        if not action_details and not positions:
            return
        td_delta = float(signal.get("td_delta", 0.0))
        action_payload = {
            "trace_id": str(trace.trace_id),
            "memory_key_schema_version": int(self.memory_key_schema_version),
            "account_context": {str(k): float(v) for k, v in getattr(trace, "account_context", {}).items()},
            "action_name": str(action_details.get("action_name", "")),
            "target_slot": int(action_details.get("target_slot", 0) or 0),
            "management_mode": str(action_details.get("management_mode", "")),
            "partial_close_pct": float(action_details.get("partial_close_pct", 0.0) or 0.0),
            "trailing_stop_atr": float(action_details.get("trailing_stop_atr", 0.0) or 0.0),
            "sl_atr": float(action_details.get("sl_atr", 0.0) or 0.0),
            "tp_atr": float(action_details.get("tp_atr", 0.0) or 0.0),
        }
        action_id = self._add_causal_node("action", action_payload, utility, 1.0 + max(0.0, utility))
        self.connections.append(MemoryConnection(memory_node_id, action_id, "causes_action", 1.0, plasticity=0.08, utility=utility))

        target_slot = int(action_payload["target_slot"])
        target_position = positions[target_slot] if 0 <= target_slot < len(positions) else (positions[0] if positions else {})
        after_by_ticket = {str(item.get("ticket", "")): item for item in positions_after}
        after_position = after_by_ticket.get(str(target_position.get("ticket", ""))) if target_position else None
        previous_id = action_id
        if target_position:
            position_payload = dict(target_position)
            position_payload["trace_id"] = str(trace.trace_id)
            position_id = self._add_causal_node("position", position_payload, utility, 1.0 + abs(float(position_payload.get("unrealized_r", 0.0))))
            self.connections.append(MemoryConnection(action_id, position_id, "targets_position", 1.0, plasticity=0.08, utility=utility))
            previous_id = position_id

        outcome_payload = {
            "trace_id": str(trace.trace_id),
            "memory_key_schema_version": int(self.memory_key_schema_version),
            "account_context_before": {str(k): float(v) for k, v in getattr(trace, "account_context", {}).items()},
            "account_context_after": {str(k): float(v) for k, v in getattr(trace, "account_context_after", {}).items()},
            "reward": float(reward),
            "cost": float(cost),
            "td_delta": td_delta,
            "prediction_error": float(signal.get("prediction_error", 0.0)),
            "utility": float(utility),
            "positive": bool(td_delta > 0.0 or reward > 0.0),
        }
        if target_position:
            classifier = self.position_outcome_classifier
            lifecycle_features = classifier.features_from(target_position, after_position, reward, td_delta, utility)
            neural_tags, tag_probs = classifier.classify(lifecycle_features)
            mfe_r = float(target_position.get("mfe_r", target_position.get("unrealized_r", 0.0)) or 0.0)
            giveback_r = float(target_position.get("giveback_r", max(0.0, mfe_r - float(target_position.get("unrealized_r", 0.0))) or 0.0))
            near_tp = bool(target_position.get("near_tp", False)) or float(target_position.get("min_dist_tp_atr", 999.0) or 999.0) <= 0.25
            closed_after = after_position is None and bool(positions_after or target_position)
            after_unrealized = float(after_position.get("unrealized_r", 0.0)) if after_position else None
            if after_position is not None:
                giveback_r = max(giveback_r, mfe_r - after_unrealized)
            audit_tags: list[str] = []
            if giveback_r >= 0.75 and mfe_r >= 1.0:
                audit_tags.append("gave_back_profit")
            if near_tp and giveback_r >= 0.5:
                audit_tags.append("near_tp_giveback")
            if near_tp and (utility < 0.0 or td_delta < 0.0 or reward < 0.0):
                audit_tags.append("near_tp_missed")
            if closed_after and near_tp and (utility < 0.0 or reward < 0.0):
                audit_tags.append("closed_after_near_tp_bad")
            if "near_tp_missed" in audit_tags or "gave_back_profit" in audit_tags:
                audit_tags.append("should_have_partial_or_trail")
            audit_vec = np.zeros(len(classifier.labels), dtype=float)
            for idx, label in enumerate(classifier.labels):
                audit_vec[idx] = 1.0 if label in audit_tags else 0.0
            classifier.observe(lifecycle_features, audit_vec, strength=1.0 + abs(utility) + abs(td_delta))
            tags = sorted(set(neural_tags) | set(audit_tags))
            outcome_payload.update({
                "target_ticket": str(target_position.get("ticket", "")),
                "target_slot": int(target_position.get("slot", target_slot)),
                "mfe_r": mfe_r,
                "mae_r": float(target_position.get("mae_r", target_position.get("unrealized_r", 0.0)) or 0.0),
                "giveback_r": float(max(0.0, giveback_r)),
                "near_tp": bool(near_tp),
                "closed_after": bool(closed_after),
                "after_unrealized_r": after_unrealized,
                "lifecycle_tags": tags,
                "neural_lifecycle_probs": {label: float(prob) for label, prob in zip(classifier.labels, tag_probs)},
                "audit_lifecycle_tags": sorted(set(audit_tags)),
            })
        outcome_id = self._add_causal_node("outcome", outcome_payload, utility, 1.0 + abs(td_delta) + max(0.0, reward))
        self.connections.append(MemoryConnection(previous_id, outcome_id, "leads_to_outcome", np.tanh(utility), plasticity=0.1, utility=utility))
        self.connections.append(MemoryConnection(outcome_id, memory_node_id, "credit", np.tanh(td_delta + reward - cost), plasticity=0.1, utility=utility))

    @staticmethod
    def _trace_action_signature(trace: StepTrace) -> tuple[str, str, int]:
        details = dict(getattr(trace, "action_details", {}) or {})
        try:
            target_slot = int(details.get("target_slot", 0) or 0)
        except (TypeError, ValueError):
            target_slot = 0
        return str(details.get("action_name", "")), str(details.get("management_mode", "")), target_slot

    def _memory_action_signature(self, node_id: str) -> tuple[str, str, int] | None:
        for conn in self.connections:
            if conn.source != node_id or conn.kind != "causes_action":
                continue
            action_node = self.causal_nodes.get(conn.target)
            if not action_node or action_node.get("kind") != "action":
                continue
            payload = dict(action_node.get("payload", {}) or {})
            try:
                target_slot = int(payload.get("target_slot", 0) or 0)
            except (TypeError, ValueError):
                target_slot = 0
            return str(payload.get("action_name", "")), str(payload.get("management_mode", "")), target_slot
        return None

    def _find_duplicate_memory(self, trace: StepTrace) -> tuple[str, MemoryNeuron] | None:
        if not dict(getattr(trace, "action_details", {}) or {}):
            return None
        obs = self._trace_memory_key(trace)
        _, delta, short, regime = self._observe_context(obs, advance=False)
        action = self._resize(trace.selected_action, 16)
        signature = self._trace_action_signature(trace)
        # No architecture cap: all episodic memories are eligible. In practice
        # a future ANN/ANN index can accelerate this without changing semantics.
        candidates = list(self.neurons)
        best: tuple[float, str, MemoryNeuron] | None = None
        for node_id in candidates:
            node = self.neurons.get(node_id)
            if node is None or self._memory_action_signature(node_id) != signature:
                continue
            if int(getattr(node, "memory_key_schema_version", 1)) != int(self.memory_key_schema_version):
                continue
            obs_sim = self._multi_scale_similarity(
                obs, delta, short, regime, node,
                context_name="obs_embedding", delta_name="context_delta",
                short_name="short_context", regime_name="regime_context",
            )
            action_sim = self._cosine(action, node.action_embedding)
            if obs_sim < self.duplicate_similarity or action_sim < self.duplicate_action_similarity:
                continue
            score = 0.7 * obs_sim + 0.3 * action_sim
            if best is None or score > best[0]:
                best = (score, node_id, node)
        return None if best is None else (best[1], best[2])

    def _update_duplicate_memory(
        self,
        node_id: str,
        node: MemoryNeuron,
        trace: StepTrace,
        next_obs: np.ndarray,
        reward: float,
        cost: float,
        signal: dict[str, float],
        utility: float,
    ) -> str:
        action_origin = self._action_origin_for_trace(trace)
        credit_weight = self._credit_weight_for_origin(action_origin)
        raw_utility = float(utility)
        effective_utility = raw_utility * credit_weight
        old_utility = float(node.utility)
        support = int(getattr(node, "support_count", 1))
        lr = float(np.clip(1.0 / max(2.0, support + 1.0), 0.02, 0.20))
        obs_key = self._trace_memory_key(trace)
        current, delta, short, regime = self._observe_context(obs_key, advance=True)
        after_key = self._memory_key(next_obs, getattr(trace, "account_context_after", {}), getattr(trace, "positions_pipe_after", ""))
        node.obs_embedding = (1.0 - lr) * self._resize(node.obs_embedding, obs_key.size) + lr * current
        node.context_delta = (1.0 - lr) * self._resize(node.context_delta, delta.size) + lr * delta
        node.short_context = (1.0 - lr) * self._resize(node.short_context, short.size) + lr * short
        node.regime_context = (1.0 - lr) * self._resize(node.regime_context, regime.size) + lr * regime
        node.action_embedding = (1.0 - lr) * node.action_embedding + lr * self._resize(trace.selected_action, 16)
        node.next_embedding = (1.0 - lr) * self._resize(node.next_embedding, after_key.size) + lr * after_key
        node.reward_ema = float((1.0 - lr) * node.reward_ema + lr * float(reward))
        node.td_ema = float((1.0 - lr) * node.td_ema + lr * float(signal.get("td_delta", 0.0)))
        node.prediction_error_ema = float((1.0 - lr) * node.prediction_error_ema + lr * float(signal.get("prediction_error", 0.0)))
        node.raw_utility = float((1.0 - lr) * float(getattr(node, "raw_utility", node.utility)) + lr * raw_utility)
        node.credit_weight = float((1.0 - lr) * float(getattr(node, "credit_weight", credit_weight)) + lr * credit_weight)
        node.utility = float((1.0 - lr) * node.utility + lr * effective_utility)
        node.strength = float(np.clip(node.strength + 0.03 * max(0.0, effective_utility) - 0.01 * max(0.0, -effective_utility), 0.0, 5.0))
        node.recurrent_state = float(np.tanh((1.0 - lr) * node.recurrent_state + lr * float(signal.get("td_delta", 0.0))))
        self._update_evidence(node, effective_utility, lr=lr)
        node.last_seen_step = int(self.context_step)
        node.action_origin = action_origin
        node.support_policy_count = float(getattr(node, "support_policy_count", 0.0)) + (1.0 if action_origin == "policy" else 0.0)
        node.support_bootstrap_count = float(getattr(node, "support_bootstrap_count", 0.0)) + (1.0 if action_origin == "bootstrap" else 0.0)
        node.counterfactual_action_values = self._counterfactual_values_for_trace(trace)
        node.causal_lesson = self._causal_lesson_for_transition(trace, reward, cost, signal, raw_utility, action_origin)
        self._update_promotion_state(node)
        self.memory_origin_stats[action_origin] = float(self.memory_origin_stats.get(action_origin, 0.0)) + 1.0
        setattr(node, "support_count", support + 1)
        node.memory_key_schema_version = int(self.memory_key_schema_version)
        node.account_context = {str(k): float(v) for k, v in getattr(trace, "account_context", {}).items()}
        node.account_context_after = {str(k): float(v) for k, v in getattr(trace, "account_context_after", {}).items()}
        node.position_context = self._position_context_vector(getattr(trace, "positions_pipe", ""))
        if node_id in self.recent_ids:
            self.recent_ids.remove(node_id)
        self.recent_ids.append(node_id)
        td_delta = float(signal.get("td_delta", 0.0))
        pred_error = float(signal.get("prediction_error", 0.0))
        meaningful = (
            abs(effective_utility - old_utility) > self.novelty_utility_threshold
            or (credit_weight > 0.0 and abs(td_delta) > self.novelty_utility_threshold)
            or (credit_weight > 0.0 and abs(float(reward)) > self.novelty_utility_threshold)
            or pred_error > 2.0
        )
        if meaningful:
            self._absorb_causal_details(node_id, trace, reward, cost, signal, effective_utility)
        if credit_weight > 0.0:
            self._grow_hierarchy(node)
        self.develop()
        return node_id

    def absorb_transition(
        self,
        trace: StepTrace,
        next_obs: np.ndarray,
        reward: float,
        cost: float,
        signal: dict[str, float],
    ) -> str:
        execution_details = dict(getattr(trace, "execution_details", {}) or {})
        # MT5 action messages are only requests.  A rejected/ignored request
        # must never allocate episodic/causal memory.  Direct simulator calls
        # carry no bridge marker and preserve their original behaviour.
        if bool(execution_details.get("bridge_observed", False)) and not bool(execution_details.get("memory_eligible", False)):
            return ""
        td_delta = float(signal.get("td_delta", 0.0))
        pred_error = float(signal.get("prediction_error", 0.0))
        raw_utility = td_delta + float(reward) - float(cost) - 0.1 * pred_error
        action_origin = self._action_origin_for_trace(trace)
        credit_weight = self._credit_weight_for_origin(action_origin)
        utility = raw_utility * credit_weight
        duplicate = self._find_duplicate_memory(trace)
        if duplicate is not None:
            return self._update_duplicate_memory(duplicate[0], duplicate[1], trace, next_obs, reward, cost, signal, raw_utility)
        node_id = self._new_id()
        current_context, context_delta, short_context, regime_context = self._observe_context(self._trace_memory_key(trace), advance=True)
        node = MemoryNeuron(
            node_id=node_id,
            trace_id=str(trace.trace_id),
            obs_embedding=current_context,
            action_embedding=self._resize(trace.selected_action, 16),
            next_embedding=self._memory_key(next_obs, getattr(trace, "account_context_after", {}), getattr(trace, "positions_pipe_after", "")),
            reward_ema=float(reward),
            td_ema=td_delta,
            prediction_error_ema=pred_error,
            strength=float(np.clip(0.5 + max(0.0, utility), 0.05, 3.0)),
            utility=float(utility),
            raw_utility=float(raw_utility),
            credit_weight=float(credit_weight),
            action_origin=str(action_origin),
            promotion_state="tentative" if action_origin == "policy" else "shadow",
            support_policy_count=1.0 if action_origin == "policy" else 0.0,
            support_bootstrap_count=1.0 if action_origin == "bootstrap" else 0.0,
            counterfactual_action_values=self._counterfactual_values_for_trace(trace),
            causal_lesson=self._causal_lesson_for_transition(trace, reward, cost, signal, raw_utility, action_origin),
            forget_gate=float(np.clip(0.03 + max(0.0, -utility) * 0.02, 0.005, 0.25)),
            plasticity=float(np.clip(0.05 + abs(td_delta) * 0.02 + max(0.0, pred_error) * 0.01, 0.01, 0.5)),
            recurrent_state=float(np.tanh(td_delta)),
            memory_key_schema_version=int(self.memory_key_schema_version),
            account_context={str(k): float(v) for k, v in getattr(trace, "account_context", {}).items()},
            account_context_after={str(k): float(v) for k, v in getattr(trace, "account_context_after", {}).items()},
            position_context=self._position_context_vector(getattr(trace, "positions_pipe", "")),
            context_delta=context_delta,
            short_context=short_context,
            regime_context=regime_context,
            evidence_for=max(0.0, utility),
            evidence_against=max(0.0, -utility),
            outcome_variance=0.0,
            uncertainty=0.5,
            last_seen_step=int(self.context_step),
        )
        self._update_evidence(node, utility, lr=1.0)
        self._update_promotion_state(node)
        self.memory_origin_stats[action_origin] = float(self.memory_origin_stats.get(action_origin, 0.0)) + 1.0
        setattr(node, "support_count", 1)
        self.neurons[node_id] = node
        self._absorb_causal_details(node_id, trace, reward, cost, signal, utility)
        for old_id in self.recent_ids[-12:]:
            old = self.neurons.get(old_id)
            if old is None:
                continue
            sim = self._cosine(old.obs_embedding, node.obs_embedding)
            if sim > 0.2:
                self.connections.append(MemoryConnection(old_id, node_id, "lateral", sim * node.strength, plasticity=node.plasticity, utility=utility))
            if old.td_ema > 0.0 and td_delta > 0.0:
                self.connections.append(MemoryConnection(old_id, node_id, "remember", 0.5 + 0.5 * sim, plasticity=node.plasticity, utility=utility))
            if old.utility < 0.0 or utility < 0.0:
                self.connections.append(MemoryConnection(old_id, node_id, "forget", -abs(0.2 + 0.3 * sim), plasticity=node.plasticity, utility=-abs(utility)))
            self.connections.append(MemoryConnection(old_id, node_id, "recurrent", 0.1 * np.tanh(old.td_ema + td_delta), plasticity=node.plasticity, utility=utility))
            self.connections.append(MemoryConnection(old_id, node_id, "plastic", sim * node.plasticity, plasticity=node.plasticity, utility=utility))
        self.recent_ids.append(node_id)
        if credit_weight > 0.0:
            self._grow_hierarchy(node)
        self.develop()
        return node_id

    def _grow_hierarchy(self, node: MemoryNeuron) -> None:
        if not self.clusters:
            self._create_cluster([node.node_id])
            return
        best = max(
            self.clusters.values(),
            key=lambda cluster: self._multi_scale_similarity(
                node.obs_embedding, node.context_delta, node.short_context, node.regime_context,
                cluster, context_name="centroid",
            ),
        )
        sim = self._multi_scale_similarity(
            node.obs_embedding, node.context_delta, node.short_context, node.regime_context,
            best, context_name="centroid",
        )
        if sim > 0.72:
            best.member_ids.append(node.node_id)
            lr = best.plasticity
            best.centroid = (1.0 - lr) * self._resize(best.centroid, self.memory_key_size) + lr * self._resize(node.obs_embedding, self.memory_key_size)
            best.context_delta_proto = (1.0 - lr) * self._resize(best.context_delta_proto, self.memory_key_size) + lr * self._resize(node.context_delta, self.memory_key_size)
            best.short_context_proto = (1.0 - lr) * self._resize(best.short_context_proto, self.memory_key_size) + lr * self._resize(node.short_context, self.memory_key_size)
            best.regime_context_proto = (1.0 - lr) * self._resize(best.regime_context_proto, self.memory_key_size) + lr * self._resize(node.regime_context, self.memory_key_size)
            best.action_proto = (1.0 - lr) * best.action_proto + lr * node.action_embedding
            best.utility = float(0.9 * best.utility + 0.1 * node.utility)
            self._update_evidence(best, node.utility, lr=lr)
            best.policy_support = float(getattr(best, "policy_support", 0.0)) + float(getattr(node, "support_policy_count", 0.0))
            best.bootstrap_support = float(getattr(best, "bootstrap_support", 0.0)) + float(getattr(node, "support_bootstrap_count", 0.0))
            self._update_promotion_state(best)
            best.strength = float(np.clip(best.strength + 0.05 * max(0.0, node.utility), 0.0, 5.0))
            self.connections.append(MemoryConnection(node.node_id, best.cluster_id, "abstract", sim, plasticity=node.plasticity, utility=node.utility))
            self.connections.append(MemoryConnection(best.cluster_id, node.node_id, "down", sim * best.strength, plasticity=best.plasticity, utility=best.utility))
            cluster = best
        else:
            cluster = self._create_cluster([node.node_id])
        concept = self._update_or_create_concept(cluster)
        if concept is not None and str(getattr(concept, "promotion_state", "shadow")) in {"candidate", "consolidated"}:
            # Strategies grow from concepts, not from a single low-match episode.
            # This makes structural growth evidence-driven without imposing a fixed cap.
            if len(cluster.member_ids) >= 3 and cluster.utility > 0.0:
                self._grow_strategy(cluster)

    def _create_cluster(self, member_ids: list[str]) -> MemoryCluster:
        members = [self.neurons[mid] for mid in member_ids if mid in self.neurons]
        centroid = np.mean([self._resize(m.obs_embedding, self.memory_key_size) for m in members], axis=0) if members else np.zeros(self.memory_key_size, dtype=float)
        action_proto = np.mean([m.action_embedding for m in members], axis=0) if members else np.zeros(16, dtype=float)
        utility = float(np.mean([m.utility for m in members])) if members else 0.0
        policy_support = float(sum(float(getattr(m, "support_policy_count", 0.0)) for m in members))
        bootstrap_support = float(sum(float(getattr(m, "support_bootstrap_count", 0.0)) for m in members))
        cluster_id = self._typed_id("cluster")
        cluster = MemoryCluster(
            cluster_id=cluster_id,
            member_ids=list(member_ids),
            centroid=centroid,
            action_proto=action_proto,
            utility=utility,
            strength=float(np.clip(0.5 + max(0.0, utility), 0.05, 4.0)),
            forget_gate=float(np.clip(0.02 + max(0.0, -utility) * 0.02, 0.005, 0.2)),
            context_delta_proto=(np.mean([self._resize(m.context_delta, self.memory_key_size) for m in members], axis=0) if members else np.zeros(self.memory_key_size, dtype=float)),
            short_context_proto=(np.mean([self._resize(m.short_context, self.memory_key_size) for m in members], axis=0) if members else np.zeros(self.memory_key_size, dtype=float)),
            regime_context_proto=(np.mean([self._resize(m.regime_context, self.memory_key_size) for m in members], axis=0) if members else np.zeros(self.memory_key_size, dtype=float)),
            evidence_for=float(sum(getattr(m, "evidence_for", max(0.0, m.utility)) for m in members)),
            evidence_against=float(sum(getattr(m, "evidence_against", max(0.0, -m.utility)) for m in members)),
            uncertainty=float(np.mean([getattr(m, "uncertainty", 1.0) for m in members])) if members else 1.0,
            policy_support=policy_support,
            bootstrap_support=bootstrap_support,
            promotion_state="shadow",
        )
        self._update_promotion_state(cluster)
        self._update_evidence(cluster, utility, lr=0.20)
        self.clusters[cluster_id] = cluster
        for member_id in member_ids:
            self.connections.append(MemoryConnection(member_id, cluster_id, "abstract", 1.0, plasticity=cluster.plasticity, utility=utility))
            self.connections.append(MemoryConnection(cluster_id, member_id, "down", cluster.strength, plasticity=cluster.plasticity, utility=utility))
        return cluster

    def _concept_name_for_cluster(self, cluster: MemoryCluster) -> str:
        reliability, direction, uncertainty = self._evidence_reliability(cluster)
        if float(getattr(cluster, "bootstrap_support", 0.0)) > float(getattr(cluster, "policy_support", 0.0)):
            origin = "bootstrap_shadow"
        elif direction >= 0.25:
            origin = "policy_positive"
        elif direction <= -0.25:
            origin = "policy_negative"
        else:
            origin = "mixed"
        if uncertainty > 0.65:
            confidence = "high_uncertainty"
        elif reliability > 0.55:
            confidence = "reliable"
        else:
            confidence = "low_support"
        if cluster.utility < -0.25:
            outcome = "loss"
        elif cluster.utility > 0.25:
            outcome = "edge"
        else:
            outcome = "neutral"
        return f"{origin}:{outcome}:{confidence}"

    def _causal_profile_for_cluster(self, cluster: MemoryCluster) -> dict[str, float]:
        profile: dict[str, float] = {}
        for member_id in cluster.member_ids[-128:]:
            node = self.neurons.get(member_id)
            if node is None:
                continue
            lesson = dict(getattr(node, "causal_lesson", {}) or {})
            for cause in lesson.get("suspected_causes", []) or []:
                key = str(cause)
                profile[key] = profile.get(key, 0.0) + 1.0
        total = sum(profile.values())
        if total > 0:
            profile = {key: float(value / total) for key, value in profile.items()}
        return profile

    def _counterfactual_profile_for_cluster(self, cluster: MemoryCluster) -> dict[str, float]:
        totals: dict[str, float] = {}
        counts: dict[str, float] = {}
        for member_id in cluster.member_ids[-128:]:
            node = self.neurons.get(member_id)
            if node is None:
                continue
            for action, value in dict(getattr(node, "counterfactual_action_values", {}) or {}).items():
                totals[str(action)] = totals.get(str(action), 0.0) + float(value)
                counts[str(action)] = counts.get(str(action), 0.0) + 1.0
        return {action: float(total / max(1.0, counts.get(action, 1.0))) for action, total in totals.items()}

    def _update_or_create_concept(self, cluster: MemoryCluster) -> MemoryConcept | None:
        """Create semantic memory only after repeated policy-origin evidence.

        Low memory match is not enough.  A concept requires repeated evidence,
        survives as shadow when bootstrap-dominated, and becomes a growth signal
        only after policy support, low uncertainty and low contradiction.
        """
        self.ensure_compatible()
        if float(getattr(cluster, "policy_support", 0.0)) < 2.0 and float(getattr(cluster, "bootstrap_support", 0.0)) < 3.0:
            return None
        name = self._concept_name_for_cluster(cluster)
        existing = [
            concept for concept in self.concepts.values()
            if concept.name == name and self._multi_scale_similarity(
                cluster.centroid, cluster.context_delta_proto, cluster.short_context_proto, cluster.regime_context_proto,
                concept, context_name="context_proto",
            ) > 0.62
        ]
        causal_profile = self._causal_profile_for_cluster(cluster)
        cf_profile = self._counterfactual_profile_for_cluster(cluster)
        if existing:
            concept = max(existing, key=lambda item: item.confidence)
            if cluster.cluster_id not in concept.cluster_ids:
                concept.cluster_ids.append(cluster.cluster_id)
            lr = 0.10
            concept.context_proto = (1.0 - lr) * self._resize(concept.context_proto, self.memory_key_size) + lr * self._resize(cluster.centroid, self.memory_key_size)
            concept.action_proto = (1.0 - lr) * concept.action_proto + lr * cluster.action_proto
            concept.context_delta_proto = (1.0 - lr) * self._resize(concept.context_delta_proto, self.memory_key_size) + lr * self._resize(cluster.context_delta_proto, self.memory_key_size)
            concept.short_context_proto = (1.0 - lr) * self._resize(concept.short_context_proto, self.memory_key_size) + lr * self._resize(cluster.short_context_proto, self.memory_key_size)
            concept.regime_context_proto = (1.0 - lr) * self._resize(concept.regime_context_proto, self.memory_key_size) + lr * self._resize(cluster.regime_context_proto, self.memory_key_size)
            concept.utility = float(0.90 * concept.utility + 0.10 * cluster.utility)
            for key, value in causal_profile.items():
                concept.causal_profile[key] = float(0.85 * concept.causal_profile.get(key, 0.0) + 0.15 * value)
            for key, value in cf_profile.items():
                concept.counterfactual_values[key] = float(0.85 * concept.counterfactual_values.get(key, 0.0) + 0.15 * value)
        else:
            concept_id = self._typed_id("concept")
            concept = MemoryConcept(
                concept_id=concept_id,
                name=name,
                cluster_ids=[cluster.cluster_id],
                context_proto=cluster.centroid.copy(),
                action_proto=cluster.action_proto.copy(),
                utility=float(cluster.utility),
                evidence_for=float(getattr(cluster, "evidence_for", max(0.0, cluster.utility))),
                evidence_against=float(getattr(cluster, "evidence_against", max(0.0, -cluster.utility))),
                policy_support=float(getattr(cluster, "policy_support", 0.0)),
                bootstrap_support=float(getattr(cluster, "bootstrap_support", 0.0)),
                uncertainty=float(getattr(cluster, "uncertainty", 1.0)),
                causal_profile=causal_profile,
                counterfactual_values=cf_profile,
                related_neurons=list(cluster.member_ids[-32:]),
                context_delta_proto=cluster.context_delta_proto.copy(),
                short_context_proto=cluster.short_context_proto.copy(),
                regime_context_proto=cluster.regime_context_proto.copy(),
            )
            self.concepts[concept_id] = concept
        concept.policy_support += float(getattr(cluster, "policy_support", 0.0))
        concept.bootstrap_support += float(getattr(cluster, "bootstrap_support", 0.0))
        self._update_evidence(concept, float(cluster.utility), lr=0.12)
        reliability, direction, uncertainty = self._evidence_reliability(concept)
        total = max(1e-9, float(concept.evidence_for) + float(concept.evidence_against))
        concept.contradiction_rate = float(concept.evidence_against / total)
        concept.uncertainty = float(uncertainty)
        concept.confidence = float(np.clip(reliability * max(0.0, direction), 0.0, 1.0))
        if concept.action_proto.size:
            concept.best_action = int(np.argmax(concept.action_proto))
            concept.worst_action = int(np.argmin(concept.action_proto))
        if concept.policy_support >= 8.0 and concept.confidence > 0.20 and concept.contradiction_rate < 0.40:
            concept.promotion_state = "consolidated"
        elif concept.policy_support >= 3.0 and concept.contradiction_rate < 0.55:
            concept.promotion_state = "candidate"
        elif concept.contradiction_rate >= 0.55 and concept.policy_support >= 2.0:
            concept.promotion_state = "contradicted"
        else:
            concept.promotion_state = "shadow"
        self.connections.append(MemoryConnection(cluster.cluster_id, concept.concept_id, "conceptualizes", concept.confidence, plasticity=0.04, utility=concept.utility))
        self.connections.append(MemoryConnection(concept.concept_id, cluster.cluster_id, "explains", concept.confidence, plasticity=0.04, utility=concept.utility))
        return concept

    def _grow_strategy(self, cluster: MemoryCluster) -> None:
        related = [
            other for other in self.clusters.values()
            if other.cluster_id != cluster.cluster_id and self._cosine(other.centroid, cluster.centroid) > 0.55 and other.utility > -0.25
        ]
        cluster_ids = [cluster.cluster_id] + [other.cluster_id for other in sorted(related, key=lambda item: item.utility, reverse=True)]
        existing = [
            strategy for strategy in self.strategies.values()
            if cluster.cluster_id in strategy.cluster_ids or self._cosine(strategy.context_proto, cluster.centroid) > 0.75
        ]
        if existing:
            strategy = max(existing, key=lambda item: item.utility)
            for cluster_id in cluster_ids:
                if cluster_id not in strategy.cluster_ids:
                    strategy.cluster_ids.append(cluster_id)
            lr = strategy.plasticity
            strategy.context_proto = (1.0 - lr) * self._resize(strategy.context_proto, self.memory_key_size) + lr * self._resize(cluster.centroid, self.memory_key_size)
            strategy.action_proto = (1.0 - lr) * strategy.action_proto + lr * cluster.action_proto
            strategy.context_delta_proto = (1.0 - lr) * self._resize(strategy.context_delta_proto, self.memory_key_size) + lr * self._resize(cluster.context_delta_proto, self.memory_key_size)
            strategy.short_context_proto = (1.0 - lr) * self._resize(strategy.short_context_proto, self.memory_key_size) + lr * self._resize(cluster.short_context_proto, self.memory_key_size)
            strategy.regime_context_proto = (1.0 - lr) * self._resize(strategy.regime_context_proto, self.memory_key_size) + lr * self._resize(cluster.regime_context_proto, self.memory_key_size)
            strategy.utility = float(0.9 * strategy.utility + 0.1 * cluster.utility)
            self._update_evidence(strategy, cluster.utility, lr=lr)
            strategy.policy_support = float(getattr(strategy, "policy_support", 0.0)) + float(getattr(cluster, "policy_support", 0.0))
            strategy.bootstrap_support = float(getattr(strategy, "bootstrap_support", 0.0)) + float(getattr(cluster, "bootstrap_support", 0.0))
            self._update_promotion_state(strategy)
            strategy.confidence = float(np.clip(strategy.confidence + 0.03 * max(0.0, cluster.utility), 0.0, 1.5))
        else:
            strategy_id = self._typed_id("strategy")
            strategy = StrategyMemory(
                strategy_id=strategy_id,
                cluster_ids=cluster_ids,
                context_proto=cluster.centroid.copy(),
                action_proto=cluster.action_proto.copy(),
                utility=float(cluster.utility),
                confidence=float(np.clip(0.2 + max(0.0, cluster.utility), 0.05, 1.5)),
                recurrent_state=float(np.tanh(cluster.utility)),
                context_delta_proto=cluster.context_delta_proto.copy(),
                short_context_proto=cluster.short_context_proto.copy(),
                regime_context_proto=cluster.regime_context_proto.copy(),
                evidence_for=float(getattr(cluster, "evidence_for", max(0.0, cluster.utility))),
                evidence_against=float(getattr(cluster, "evidence_against", max(0.0, -cluster.utility))),
                uncertainty=float(getattr(cluster, "uncertainty", 1.0)),
                policy_support=float(getattr(cluster, "policy_support", 0.0)),
                bootstrap_support=float(getattr(cluster, "bootstrap_support", 0.0)),
                promotion_state="shadow",
            )
            self._update_evidence(strategy, cluster.utility, lr=0.20)
            self._update_promotion_state(strategy)
            self.strategies[strategy_id] = strategy
        for cluster_id in cluster_ids:
            self.connections.append(MemoryConnection(cluster_id, strategy.strategy_id, "abstract", strategy.confidence, plasticity=strategy.plasticity, utility=strategy.utility))
            self.connections.append(MemoryConnection(strategy.strategy_id, cluster_id, "down", strategy.confidence, plasticity=strategy.plasticity, utility=strategy.utility))
        if len(strategy.cluster_ids) >= 2 and strategy.utility > 0.0:
            self._grow_meta_strategy(strategy)

    def _grow_meta_strategy(self, strategy: StrategyMemory) -> None:
        peers = [
            other for other in self.strategies.values()
            if other.strategy_id != strategy.strategy_id and self._cosine(other.context_proto, strategy.context_proto) > 0.5
        ]
        strategy_ids = [strategy.strategy_id] + [peer.strategy_id for peer in sorted(peers, key=lambda item: item.utility, reverse=True)]
        existing = [
            meta for meta in self.meta_strategies.values()
            if strategy.strategy_id in meta.strategy_ids or self._cosine(meta.context_proto, strategy.context_proto) > 0.7
        ]
        if existing:
            meta = max(existing, key=lambda item: item.utility)
            for strategy_id in strategy_ids:
                if strategy_id not in meta.strategy_ids:
                    meta.strategy_ids.append(strategy_id)
            meta.context_proto = 0.95 * self._resize(meta.context_proto, self.memory_key_size) + 0.05 * self._resize(strategy.context_proto, self.memory_key_size)
            meta.context_delta_proto = 0.95 * self._resize(meta.context_delta_proto, self.memory_key_size) + 0.05 * self._resize(strategy.context_delta_proto, self.memory_key_size)
            meta.short_context_proto = 0.95 * self._resize(meta.short_context_proto, self.memory_key_size) + 0.05 * self._resize(strategy.short_context_proto, self.memory_key_size)
            meta.regime_context_proto = 0.95 * self._resize(meta.regime_context_proto, self.memory_key_size) + 0.05 * self._resize(strategy.regime_context_proto, self.memory_key_size)
            meta.utility = float(0.92 * meta.utility + 0.08 * strategy.utility)
            self._update_evidence(meta, strategy.utility, lr=0.08)
        else:
            meta_id = self._typed_id("meta")
            meta = MetaStrategyMemory(
                meta_id=meta_id,
                strategy_ids=strategy_ids,
                context_proto=strategy.context_proto.copy(),
                utility=float(strategy.utility),
                context_delta_proto=strategy.context_delta_proto.copy(),
                short_context_proto=strategy.short_context_proto.copy(),
                regime_context_proto=strategy.regime_context_proto.copy(),
                evidence_for=float(getattr(strategy, "evidence_for", max(0.0, strategy.utility))),
                evidence_against=float(getattr(strategy, "evidence_against", max(0.0, -strategy.utility))),
                uncertainty=float(getattr(strategy, "uncertainty", 1.0)),
            )
            self._update_evidence(meta, strategy.utility, lr=0.15)
            self.meta_strategies[meta_id] = meta
        meta.growth_bias = float(np.clip(max(0.0, meta.utility), 0.0, 2.0))
        meta.prune_bias = float(np.clip(max(0.0, -meta.utility), 0.0, 2.0))
        meta.trust = float(np.clip(0.9 * meta.trust + 0.1 * strategy.confidence, 0.0, 2.0))
        for strategy_id in strategy_ids:
            self.connections.append(MemoryConnection(strategy_id, meta.meta_id, "abstract", meta.trust, plasticity=0.03, utility=meta.utility))
            self.connections.append(MemoryConnection(meta.meta_id, strategy_id, "down", meta.trust, plasticity=0.03, utility=meta.utility))

    def develop(self) -> None:
        for node in self.neurons.values():
            node.age += 1
            node.strength = float(np.clip(node.strength * (1.0 - node.forget_gate) + 0.03 * max(0.0, node.utility), 0.0, 5.0))
            node.plasticity = float(np.clip(0.99 * node.plasticity + 0.01 * abs(node.td_ema), 0.005, 0.75))
        for conn in self.connections:
            conn.age += 1
            src_strength, dst_utility = self._node_strength(conn.source), self._node_utility(conn.target)
            if src_strength is None or dst_utility is None:
                continue
            conn.weight = float(np.clip(conn.weight + conn.plasticity * 0.01 * src_strength * np.tanh(dst_utility), -3.0, 3.0))
            conn.utility = float(0.98 * conn.utility + 0.02 * dst_utility)
        for cluster in self.clusters.values():
            cluster.age += 1
            cluster.strength = float(np.clip(cluster.strength * (1.0 - cluster.forget_gate) + 0.02 * max(0.0, cluster.utility), 0.0, 5.0))
        for concept in self.concepts.values():
            concept.age += 1
            concept.confidence = float(np.clip(0.995 * concept.confidence + 0.005 * max(0.0, concept.utility), 0.0, 1.0))
        for strategy in self.strategies.values():
            strategy.age += 1
            strategy.confidence = float(np.clip(0.995 * strategy.confidence + 0.01 * max(0.0, strategy.utility), 0.0, 2.0))
            strategy.recurrent_state = float(np.tanh(0.95 * strategy.recurrent_state + 0.05 * strategy.utility))
        for meta in self.meta_strategies.values():
            meta.age += 1
            meta.trust = float(np.clip(0.995 * meta.trust + 0.005 * max(0.0, meta.utility), 0.0, 2.0))
        for node in self.causal_nodes.values():
            node["age"] = int(node.get("age", 0)) + 1
            node["strength"] = float(np.clip(float(node.get("strength", 1.0)) * 0.995 + 0.005 * max(0.0, float(node.get("utility", 0.0))), 0.0, 5.0))
        if self.max_neurons is not None and len(self.neurons) > self.max_neurons:
            removable = sorted(self.neurons.values(), key=lambda node: (node.utility + node.strength, -node.age))
            remove_ids = {node.node_id for node in removable[: len(self.neurons) - self.max_neurons]}
            self.neurons = {key: node for key, node in self.neurons.items() if key not in remove_ids}
            self.connections = [conn for conn in self.connections if conn.source not in remove_ids and conn.target not in remove_ids]
            self.recent_ids = [node_id for node_id in self.recent_ids if node_id not in remove_ids]
        if self.max_neurons is not None and len(self.causal_nodes) > self.max_neurons:
            removable = sorted(self.causal_nodes.items(), key=lambda item: (float(item[1].get("utility", 0.0)) + float(item[1].get("strength", 0.0)), -int(item[1].get("age", 0))))
            remove_ids = {key for key, _ in removable[: len(self.causal_nodes) - self.max_neurons]}
            self.causal_nodes = {key: node for key, node in self.causal_nodes.items() if key not in remove_ids}
            self.connections = [conn for conn in self.connections if conn.source not in remove_ids and conn.target not in remove_ids]
        self._prune_hierarchy()
        if self.max_connections is not None and len(self.connections) > self.max_connections:
            self.connections.sort(key=lambda conn: (abs(conn.utility) + abs(conn.weight), -conn.age), reverse=True)
            self.connections = self.connections[: self.max_connections]

    def _node_strength(self, node_id: str) -> float | None:
        if node_id in self.neurons:
            return self.neurons[node_id].strength
        if node_id in self.clusters:
            return self.clusters[node_id].strength
        if node_id in self.strategies:
            return self.strategies[node_id].confidence
        if node_id in self.meta_strategies:
            return self.meta_strategies[node_id].trust
        if node_id in self.causal_nodes:
            return float(self.causal_nodes[node_id].get("strength", 0.0))
        return None

    def _node_utility(self, node_id: str) -> float | None:
        if node_id in self.neurons:
            return self.neurons[node_id].utility
        if node_id in self.clusters:
            return self.clusters[node_id].utility
        if node_id in self.strategies:
            return self.strategies[node_id].utility
        if node_id in self.meta_strategies:
            return self.meta_strategies[node_id].utility
        if node_id in self.causal_nodes:
            return float(self.causal_nodes[node_id].get("utility", 0.0))
        return None

    def _prune_hierarchy(self) -> None:
        remove_ids: set[str] = set()
        if self.max_clusters is not None and len(self.clusters) > self.max_clusters:
            removable = sorted(self.clusters.values(), key=lambda item: (item.utility + item.strength, -item.age))
            remove_ids.update(item.cluster_id for item in removable[: len(self.clusters) - self.max_clusters])
        if self.max_strategies is not None and len(self.strategies) > self.max_strategies:
            removable = sorted(self.strategies.values(), key=lambda item: (item.utility + item.confidence, -item.age))
            remove_ids.update(item.strategy_id for item in removable[: len(self.strategies) - self.max_strategies])
        if self.max_meta_strategies is not None and len(self.meta_strategies) > self.max_meta_strategies:
            removable = sorted(self.meta_strategies.values(), key=lambda item: (item.utility + item.trust, -item.age))
            remove_ids.update(item.meta_id for item in removable[: len(self.meta_strategies) - self.max_meta_strategies])
        if not remove_ids:
            return
        self.clusters = {key: item for key, item in self.clusters.items() if key not in remove_ids}
        self.concepts = {key: item for key, item in self.concepts.items() if key not in remove_ids}
        self.strategies = {key: item for key, item in self.strategies.items() if key not in remove_ids}
        self.meta_strategies = {key: item for key, item in self.meta_strategies.items() if key not in remove_ids}
        self.causal_nodes = {key: item for key, item in self.causal_nodes.items() if key not in remove_ids}
        self.connections = [conn for conn in self.connections if conn.source not in remove_ids and conn.target not in remove_ids]

    @staticmethod
    def _action_origin_for_trace(trace: StepTrace) -> str:
        details = dict(getattr(trace, "action_details", {}) or {})
        execution = dict(getattr(trace, "execution_details", {}) or {})
        if bool(details.get("execution_bootstrap", False)):
            return "bootstrap"
        explicit = str(details.get("action_origin", "") or execution.get("action_origin", "")).strip().lower()
        if explicit:
            return explicit
        if bool(details.get("action_masked", False)):
            return "mask"
        if bool(execution.get("risk_override", False)):
            return "risk_override"
        if bool(execution.get("forced_exit", False)):
            return "forced_exit"
        return "policy"

    @staticmethod
    def _credit_weight_for_origin(origin: str) -> float:
        origin = str(origin or "policy").lower()
        if origin == "policy":
            return 1.0
        if origin == "risk_override":
            # Fixed-risk sizing does not originate in the risk output head;
            # never reinforce a head whose value was bypassed at execution.
            return 0.0
        if origin == "forced_exit":
            return 0.25
        if origin == "mask":
            return 0.10
        # Bootstrap is exploration evidence only: it may be stored and reflected
        # on, but it must not directly reinforce action-producing policy memory.
        if origin == "bootstrap":
            return 0.0
        return 0.5

    def _counterfactual_values_for_trace(self, trace: StepTrace) -> dict[str, float]:
        values: dict[str, float] = {}
        scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
        actions = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        for idx in range(min(actions.shape[0], scores.size if scores.size else actions.shape[0], 16)):
            try:
                action_idx = int(np.argmax(actions[idx])) if actions.shape[1] else idx
                score = float(scores[idx]) if idx < scores.size else float(np.max(actions[idx]))
                values[str(action_idx)] = score
            except Exception:
                continue
        selected = int(getattr(trace, "selected_idx", -1))
        if selected >= 0 and str(selected) not in values:
            values[str(selected)] = float(getattr(trace, "critic_value", 0.0))
        return values

    def _causal_lesson_for_transition(
        self,
        trace: StepTrace,
        reward: float,
        cost: float,
        signal: dict[str, float],
        raw_utility: float,
        action_origin: str,
    ) -> dict[str, object]:
        details = dict(getattr(trace, "action_details", {}) or {})
        account = dict(getattr(trace, "account_context", {}) or {})
        pred_error = float(signal.get("prediction_error", getattr(trace, "prediction_error", 0.0)))
        td_delta = float(signal.get("td_delta", getattr(trace, "td_delta", 0.0)))
        risk = float(getattr(trace, "risk", 0.0))
        lesson_type = "positive_skill" if raw_utility > 0.0 else "failure_analysis"
        suspected_causes: list[str] = []
        if action_origin != "policy":
            suspected_causes.append(f"non_policy_origin:{action_origin}")
        if pred_error > 1.0:
            suspected_causes.append("prediction_error_high")
        if td_delta < -0.05:
            suspected_causes.append("critic_overestimated_outcome")
        if risk > 0.65:
            suspected_causes.append("risk_state_high")
        if float(account.get("drawdown_pct", 0.0) or 0.0) > 5.0:
            suspected_causes.append("account_drawdown_context")
        if bool(details.get("execution_bootstrap", False)):
            suspected_causes.append("bootstrap_exploration_not_policy_skill")
        if float(reward) < 0.0 and not suspected_causes:
            suspected_causes.append("unexplained_negative_outcome")
        if float(reward) > 0.0 and not suspected_causes:
            suspected_causes.append("validated_positive_outcome")
        return {
            "lesson_type": lesson_type,
            "action_origin": str(action_origin),
            "suspected_causes": suspected_causes,
            "td_delta": float(td_delta),
            "prediction_error": float(pred_error),
            "reward": float(reward),
            "cost": float(cost),
            "raw_utility": float(raw_utility),
            "selected_idx": int(getattr(trace, "selected_idx", -1)),
            "action_name": str(details.get("action_name", "")),
            "policy_action": int(details.get("policy_action", -1) or -1),
            "bridge_action": int(details.get("bridge_action", -1) or -1),
        }

    @staticmethod
    def _eligible_for_action_memory(item: object) -> bool:
        origin = str(getattr(item, "action_origin", "policy") or "policy")
        credit_weight = float(getattr(item, "credit_weight", 1.0 if origin == "policy" else 0.0))
        return origin == "policy" and credit_weight >= 0.5

    def _update_promotion_state(self, item: object) -> None:
        pos = max(0.0, float(getattr(item, "evidence_for", 0.0)))
        neg = max(0.0, float(getattr(item, "evidence_against", 0.0)))
        policy_support = float(getattr(item, "policy_support", getattr(item, "support_policy_count", 0.0)))
        total = pos + neg
        contradiction_rate = float(neg / (total + 1e-9)) if total > 0.0 else 0.0
        setattr(item, "contradiction_rate", contradiction_rate)
        uncertainty = float(getattr(item, "uncertainty", 1.0))
        utility = float(getattr(item, "utility", 0.0))
        if policy_support >= 6.0 and utility > 0.05 and contradiction_rate < 0.40 and uncertainty < 0.55:
            state = "consolidated"
        elif policy_support >= 3.0 and utility > 0.0 and contradiction_rate < 0.55:
            state = "candidate"
        elif neg > pos and policy_support >= 2.0:
            state = "contradicted"
        else:
            state = "shadow"
        setattr(item, "promotion_state", state)

    def reflect_memory(self, *, recent_window: int = 128) -> dict[str, object]:
        self.ensure_compatible()
        recent = [self.neurons[node_id] for node_id in self.recent_ids[-recent_window:] if node_id in self.neurons]
        if not recent:
            reflection = {"episodes": 0, "recommendations": ["collect_policy_origin_episodes"]}
            self.reflection_log.append(reflection)
            return reflection
        origins: dict[str, int] = {}
        for node in recent:
            origins[node.action_origin] = origins.get(node.action_origin, 0) + 1
        policy_nodes = [node for node in recent if self._eligible_for_action_memory(node)]
        bootstrap_nodes = [node for node in recent if node.action_origin == "bootstrap"]
        contradicted_clusters = [c.cluster_id for c in self.clusters.values() if str(getattr(c, "promotion_state", "")) == "contradicted"][:10]
        concept_states: dict[str, int] = {}
        for concept in self.concepts.values():
            concept_states[str(getattr(concept, "promotion_state", "shadow"))] = concept_states.get(str(getattr(concept, "promotion_state", "shadow")), 0) + 1
        top_concepts = sorted(
            [
                {
                    "concept_id": c.concept_id,
                    "name": c.name,
                    "state": c.promotion_state,
                    "confidence": float(c.confidence),
                    "policy_support": float(c.policy_support),
                    "contradiction_rate": float(c.contradiction_rate),
                    "best_action": int(c.best_action),
                    "top_causes": sorted(c.causal_profile.items(), key=lambda item: item[1], reverse=True)[:3],
                }
                for c in self.concepts.values()
            ],
            key=lambda item: (float(item["confidence"]), float(item["policy_support"])),
            reverse=True,
        )[:8]
        recommendations: list[str] = []
        if len(policy_nodes) < max(3, len(recent) // 4):
            recommendations.append("reduce_bootstrap_credit_or_increase_policy_action_diversity")
        if bootstrap_nodes:
            recommendations.append("bootstrap_episodes_should_remain_shadow_evidence")
        if contradicted_clusters:
            recommendations.append("prune_or_shadow_contradicted_memory_clusters")
        if not any(state in concept_states for state in ("candidate", "consolidated")) and len(policy_nodes) >= 8:
            recommendations.append("build_semantic_concepts_before_structural_growth")
        if concept_states.get("contradicted", 0) > 0:
            recommendations.append("retire_or_ablate_contradicted_concepts")
        if not recommendations:
            recommendations.append("continue_consolidating_policy_origin_memory")
        reflection = {
            "episodes": len(recent),
            "policy_origin": len(policy_nodes),
            "bootstrap_origin": len(bootstrap_nodes),
            "origin_counts": origins,
            "mean_policy_utility": float(np.mean([n.raw_utility for n in policy_nodes])) if policy_nodes else 0.0,
            "mean_bootstrap_utility": float(np.mean([n.raw_utility for n in bootstrap_nodes])) if bootstrap_nodes else 0.0,
            "contradicted_clusters": contradicted_clusters,
            "concept_states": concept_states,
            "top_concepts": top_concepts,
            "recommendations": recommendations,
            "neural_substrate_cells": len(getattr(self, "neural_cells", {}) or {}),
            "neural_substrate_synapses": len(getattr(self, "neural_synapses", []) or []),
        }
        self.reflection_log.append(reflection)
        if len(self.reflection_log) > 256:
            self.reflection_log = self.reflection_log[-256:]
        return reflection




    def _cell_reliability(self, cell: NeuralMemoryCell) -> tuple[float, float, float]:
        pos = max(0.0, float(getattr(cell, "evidence_for", 0.0)))
        neg = max(0.0, float(getattr(cell, "evidence_against", 0.0)))
        total = pos + neg
        posterior = (1.0 + pos) / (2.0 + total)
        direction = 2.0 * posterior - 1.0
        support = 1.0 - np.exp(-total / 4.0)
        uncertainty = float(np.clip(max(float(getattr(cell, "uncertainty", 1.0)), 1.0 / (1.0 + total)), 0.0, 1.0))
        reliability = float(np.clip(support * (1.0 - 0.65 * uncertainty), 0.0, 1.0))
        return reliability, direction, uncertainty

    def _cell_multi_scale_similarity(self, current: np.ndarray, delta: np.ndarray, short: np.ndarray, regime: np.ndarray, cell: NeuralMemoryCell) -> float:
        meta = dict(getattr(cell, "meta", {}) or {})
        base = self._resize(np.asarray(getattr(cell, "vector", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        delta_proto = self._resize(np.asarray(meta.get("context_delta", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        short_proto = self._resize(np.asarray(meta.get("short_context", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        regime_proto = self._resize(np.asarray(meta.get("regime_context", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        sims = np.array([
            max(0.0, self._cosine(current, base)),
            max(0.0, self._cosine(delta, delta_proto)),
            max(0.0, self._cosine(short, short_proto)),
            max(0.0, self._cosine(regime, regime_proto)),
        ], dtype=float)
        active = np.array([
            np.linalg.norm(base) > 1e-8,
            np.linalg.norm(delta_proto) > 1e-8,
            np.linalg.norm(short_proto) > 1e-8,
            np.linalg.norm(regime_proto) > 1e-8,
        ], dtype=bool)
        weights = np.array([0.45, 0.15, 0.20, 0.20], dtype=float)
        weights[~active] = 0.0
        if float(weights.sum()) <= 1e-12:
            return 0.0
        return float(np.clip(float(sims @ weights) / float(weights.sum()), 0.0, 1.0))

    def _neural_action_gain(self, cell: NeuralMemoryCell) -> float:
        meta = dict(getattr(cell, "meta", {}) or {})
        role = str(getattr(cell, "role", ""))
        state = str(meta.get("promotion_state", ""))
        if role == "episodic_memory":
            return 1.0 if float(meta.get("credit_weight", 1.0) or 0.0) >= 0.99 else 0.0
        if role == "semantic_concept":
            return 1.0 if state in {"candidate", "consolidated"} else 0.0
        if role in {"procedural_strategy", "meta_strategy"}:
            return 1.0
        if role == "context_attractor":
            return 0.25
        return 0.0

    def _query_neural_substrate(
        self,
        state: np.ndarray,
        delta: np.ndarray,
        short: np.ndarray,
        regime: np.ndarray,
        *,
        output_size: int,
        module_count: int,
    ) -> dict[str, object]:
        action_bias = np.zeros(output_size, dtype=float)
        memory_context = np.zeros(32, dtype=float)
        critic_bias = np.zeros(32, dtype=float)
        gate_bias = np.zeros(max(1, module_count), dtype=float)
        matched: list[tuple[str, float, float]] = []
        total_weight = 0.0
        total_reliability = 0.0
        weighted_uncertainty = 0.0
        contradiction_strength = 0.0
        growth_bias = 0.0
        prune_bias = 0.0
        trust = 0.0
        role_scales = {
            "semantic_concept": 1.15,
            "procedural_strategy": 1.00,
            "meta_strategy": 0.90,
            "context_attractor": 0.40,
            "episodic_memory": 0.18,
            "causal_action": 0.20,
            "causal_position": 0.20,
            "causal_outcome": 0.25,
            "reflection_loop": 0.15,
        }
        for cell_id, cell in list((getattr(self, "neural_cells", {}) or {}).items()):
            role = str(getattr(cell, "role", ""))
            scale = float(role_scales.get(role, 0.0))
            if scale <= 0.0 or not np.all(np.isfinite(np.asarray(cell.vector, dtype=float))):
                continue
            sim = self._cell_multi_scale_similarity(state, delta, short, regime, cell)
            if sim <= 1e-8:
                continue
            reliability, evidence_direction, uncertainty = self._cell_reliability(cell)
            utility_direction = float(np.tanh(float(getattr(cell, "utility", 0.0))))
            signed_quality = 0.65 * evidence_direction + 0.35 * utility_direction
            meta = dict(getattr(cell, "meta", {}) or {})
            support_strength = max(0.05, float(meta.get("support_strength", abs(float(getattr(cell, "activation", 0.0))) + 1.0) or 1.0))
            magnitude = scale * sim * reliability * support_strength
            if magnitude <= 1e-8:
                continue
            action = self._resize(np.asarray(meta.get("action_vector", np.zeros(output_size)), dtype=float), output_size)
            context_vector = self._resize(np.asarray(getattr(cell, "vector", np.zeros(self.memory_key_size)), dtype=float), 32)
            action_bias[:] += self._neural_action_gain(cell) * magnitude * signed_quality * action
            memory_context[:] += magnitude * context_vector
            critic_bias[:] += magnitude * (-signed_quality) * context_vector
            total_weight += magnitude
            total_reliability += magnitude * reliability
            weighted_uncertainty += magnitude * uncertainty
            contradiction_strength += magnitude * max(0.0, -signed_quality)
            matched.append((cell_id, sim, magnitude * signed_quality))
            if role == "meta_strategy":
                trust += sim * reliability * max(0.0, evidence_direction) * max(0.0, float(getattr(cell, "activation", 0.0)))
                growth_bias += sim * reliability * max(0.0, float(getattr(cell, "utility", 0.0)))
                prune_bias += sim * (1.0 - reliability + max(0.0, -evidence_direction)) * max(0.0, -float(getattr(cell, "utility", 0.0)))
        if total_weight > 1e-9:
            strength = float(np.clip(total_reliability / (1.0 + total_weight), 0.0, 1.0))
            action_bias = 0.18 * strength * np.clip(action_bias / total_weight, -1.0, 1.0)
            memory_context = 0.25 * strength * np.clip(memory_context / total_weight, -1.0, 1.0)
            critic_bias = 0.35 * strength * np.clip(critic_bias / total_weight, -1.0, 1.0)
            gate_bias += 0.08 * strength
            match_strength = strength
            uncertainty = float(np.clip(weighted_uncertainty / total_weight, 0.0, 1.0))
        else:
            match_strength = 0.0
            strength = 0.0
            uncertainty = 1.0
        causal_matches = sorted(
            [
                (cid, str(cell.role), float(cell.utility), copylib.deepcopy(dict(cell.meta or {})))
                for cid, cell in getattr(self, "neural_cells", {}).items()
                if str(getattr(cell, "role", "")).startswith("causal_") and abs(float(getattr(cell, "utility", 0.0))) > 0.0
            ],
            key=lambda item: abs(float(item[2])),
            reverse=True,
        )[:5]
        return {
            "action_bias": action_bias,
            "memory_context": memory_context,
            "critic_bias": critic_bias,
            "gate_bias": np.clip(gate_bias, 0.0, 0.12),
            "growth_bias": float(np.clip(growth_bias, 0.0, 1.0)),
            "prune_bias": float(np.clip(prune_bias, 0.0, 1.0)),
            "trust": float(np.clip(trust + strength, 0.0, 2.0)),
            "match_strength": float(np.clip(match_strength, 0.0, 1.0)),
            "uncertainty": uncertainty,
            "contradiction_strength": float(np.clip(contradiction_strength / (1.0 + total_weight), 0.0, 1.0)),
            "matches": sorted(matched, key=lambda item: abs(item[2]), reverse=True)[:5],
            "causal_matches": causal_matches,
            "memory_key_schema_version": int(self.memory_key_schema_version),
            "memory_key_size": int(self.memory_key_size),
            "context_step": int(self.context_step),
            "pure_neural_memory": True,
        }

    def query(
        self,
        state_embedding: np.ndarray,
        output_size: int,
        module_count: int = 1,
        account_context: dict[str, object] | None = None,
        positions_pipe: str | None = "",
    ) -> dict[str, object]:
        """Retrieve context-conditioned evidence from episode, concept and skill memory.

        Retrieval is evidence-aware.  Positive support can strengthen a candidate
        action; contradictory evidence is preserved and enters the critic path,
        rather than being discarded just because it is inconvenient for the
        current policy.
        """
        self.ensure_compatible()
        raw_state = self._memory_key(state_embedding, account_context, positions_pipe)
        state, delta, short, regime = self._observe_context(raw_state, advance=True)
        # V5 strict neural runtime: retrieval reads the neural substrate, not
        # typed Episode/Cluster/Concept/Strategy objects.  Typed structures may
        # still exist as migration/debug views, but they no longer decide memory
        # output when pure_neural_memory=True.
        if bool(getattr(self, "pure_neural_memory", True)):
            self._synchronise_neural_memory_substrate()
            return self._query_neural_substrate(state, delta, short, regime, output_size=output_size, module_count=module_count)
        action_bias = np.zeros(output_size, dtype=float)
        memory_context = np.zeros(32, dtype=float)
        critic_bias = np.zeros(32, dtype=float)
        gate_bias = np.zeros(max(1, module_count), dtype=float)
        matched: list[tuple[str, float, float]] = []
        total_weight = 0.0
        total_reliability = 0.0
        weighted_uncertainty = 0.0
        contradiction_strength = 0.0

        def accumulate(item: object, item_id: str, *, context_name: str, action_name: str, scale: float) -> None:
            nonlocal total_weight, total_reliability, weighted_uncertainty, contradiction_strength
            temporal_names = (
                ("context_delta", "short_context", "regime_context")
                if context_name == "obs_embedding"
                else ("context_delta_proto", "short_context_proto", "regime_context_proto")
            )
            sim = self._multi_scale_similarity(
                state, delta, short, regime, item, context_name=context_name,
                delta_name=temporal_names[0], short_name=temporal_names[1], regime_name=temporal_names[2],
            )
            if sim <= 1e-8:
                return
            reliability, evidence_direction, uncertainty = self._evidence_reliability(item)
            utility_direction = float(np.tanh(float(getattr(item, "utility", 0.0))))
            signed_quality = 0.65 * evidence_direction + 0.35 * utility_direction
            magnitude = scale * sim * reliability * max(0.05, float(getattr(item, "strength", getattr(item, "confidence", getattr(item, "trust", 1.0)))))
            if magnitude <= 1e-8:
                return
            action = self._resize(np.asarray(getattr(item, action_name, np.zeros(output_size)), dtype=float), output_size)
            context_vector = self._resize(np.asarray(getattr(item, context_name, np.zeros(self.memory_key_size)), dtype=float), 32)
            # A negative/contradictory memory penalises matching action patterns
            # through the critic, while the action path remains deliberately
            # bounded so retrieval cannot bypass MT5 risk controls. Bootstrap or
            # forced episodes may inform context/critic, but cannot directly push
            # action logits as if the policy had chosen them.
            action_gain = 1.0 if self._eligible_for_action_memory(item) or not hasattr(item, "action_origin") else 0.0
            action_bias[:] += action_gain * magnitude * signed_quality * action
            memory_context[:] += magnitude * context_vector
            critic_bias[:] += magnitude * (-signed_quality) * context_vector
            total_weight += magnitude
            total_reliability += magnitude * reliability
            weighted_uncertainty += magnitude * uncertainty
            contradiction_strength += magnitude * max(0.0, -signed_quality)
            matched.append((item_id, sim, magnitude * signed_quality))

        for concept in self.concepts.values():
            if str(getattr(concept, "promotion_state", "shadow")) in {"candidate", "consolidated", "contradicted"}:
                accumulate(concept, concept.concept_id, context_name="context_proto", action_name="action_proto", scale=1.15)
        for strategy in self.strategies.values():
            accumulate(strategy, strategy.strategy_id, context_name="context_proto", action_name="action_proto", scale=1.0)
        for cluster in self.clusters.values():
            accumulate(cluster, cluster.cluster_id, context_name="centroid", action_name="action_proto", scale=0.40)
        # Episodic memory provides rare-event evidence even before a concept has
        # enough repetitions to consolidate into a cluster.
        for node in self.neurons.values():
            accumulate(node, node.node_id, context_name="obs_embedding", action_name="action_embedding", scale=0.18)

        if total_weight > 1e-9:
            strength = float(np.clip(total_reliability / (1.0 + total_weight), 0.0, 1.0))
            action_bias = 0.18 * strength * np.clip(action_bias / total_weight, -1.0, 1.0)
            memory_context = 0.25 * strength * np.clip(memory_context / total_weight, -1.0, 1.0)
            critic_bias = 0.35 * strength * np.clip(critic_bias / total_weight, -1.0, 1.0)
            gate_bias += 0.08 * strength
            match_strength = strength
            uncertainty = float(np.clip(weighted_uncertainty / total_weight, 0.0, 1.0))
        else:
            match_strength = 0.0
            strength = 0.0
            uncertainty = 1.0

        growth_bias = 0.0
        prune_bias = 0.0
        trust = 0.0
        for meta in self.meta_strategies.values():
            sim = self._multi_scale_similarity(state, delta, short, regime, meta, context_name="context_proto")
            if sim <= 1e-8:
                continue
            reliability, direction, meta_uncertainty = self._evidence_reliability(meta)
            trust += sim * reliability * max(0.0, direction) * max(0.0, meta.trust)
            growth_bias += sim * reliability * max(0.0, meta.growth_bias)
            prune_bias += sim * (1.0 - reliability + max(0.0, -direction)) * max(0.0, meta.prune_bias)
            uncertainty = float(np.clip(0.8 * uncertainty + 0.2 * meta_uncertainty, 0.0, 1.0))

        causal_matches = sorted(
            (
                (node_id, str(node.get("kind", "")), float(node.get("utility", 0.0)), copylib.deepcopy(node.get("payload", {})))
                for node_id, node in self.causal_nodes.items()
                if abs(float(node.get("utility", 0.0))) > 0.0
            ),
            key=lambda item: (abs(item[2]), float(self.causal_nodes[item[0]].get("strength", 0.0))),
            reverse=True,
        )[:5]
        return {
            "action_bias": action_bias,
            "memory_context": memory_context,
            "critic_bias": critic_bias,
            "gate_bias": np.clip(gate_bias, 0.0, 0.12),
            "growth_bias": float(np.clip(growth_bias, 0.0, 1.0)),
            "prune_bias": float(np.clip(prune_bias, 0.0, 1.0)),
            "trust": float(np.clip(trust + strength, 0.0, 2.0)),
            "match_strength": float(np.clip(match_strength, 0.0, 1.0)),
            "uncertainty": uncertainty,
            "contradiction_strength": float(np.clip(contradiction_strength / (1.0 + total_weight), 0.0, 1.0)),
            "matches": sorted(matched, key=lambda item: abs(item[2]), reverse=True)[:5],
            "causal_matches": causal_matches,
            "memory_key_schema_version": int(self.memory_key_schema_version),
            "memory_key_size": int(self.memory_key_size),
            "context_step": int(self.context_step),
        }

    def summary(self) -> dict[str, object]:
        kinds: dict[str, int] = {}
        for conn in self.connections:
            kinds[conn.kind] = kinds.get(conn.kind, 0) + 1
        causal_kinds: dict[str, int] = {}
        for node in self.causal_nodes.values():
            kind = str(node.get("kind", "unknown"))
            causal_kinds[kind] = causal_kinds.get(kind, 0) + 1
        positive = sum(1 for node in self.neurons.values() if node.td_ema > 0.0 or node.reward_ema > 0.0)
        support_total = sum(int(getattr(node, "support_count", 1)) for node in self.neurons.values())
        compressed = max(0, support_total - len(self.neurons))
        evidence_for = float(sum(getattr(node, "evidence_for", 0.0) for node in self.neurons.values()))
        evidence_against = float(sum(getattr(node, "evidence_against", 0.0) for node in self.neurons.values()))
        mean_uncertainty = float(np.mean([getattr(node, "uncertainty", 1.0) for node in self.neurons.values()])) if self.neurons else 1.0
        return {
            "memory_key_schema_version": int(getattr(self, "memory_key_schema_version", MEMORY_KEY_SCHEMA_VERSION)),
            "memory_key_size": int(getattr(self, "memory_key_size", MEMORY_KEY_SIZE)),
            "neurons": len(self.neurons),
            "clusters": len(self.clusters),
            "concepts": len(self.concepts),
            "strategies": len(self.strategies),
            "meta_strategies": len(self.meta_strategies),
            "causal_nodes": len(self.causal_nodes),
            "connections": len(self.connections),
            "positive_neurons": positive,
            "support_total": support_total,
            "compressed_transitions": compressed,
            "evidence_for": evidence_for,
            "evidence_against": evidence_against,
            "mean_uncertainty": mean_uncertainty,
            "memory_origin_stats": {str(k): float(v) for k, v in dict(getattr(self, "memory_origin_stats", {}) or {}).items()},
            "concept_states": {state: sum(1 for c in self.concepts.values() if str(getattr(c, "promotion_state", "shadow")) == state) for state in ("shadow", "candidate", "consolidated", "contradicted")},
            "top_concepts": [
                {"concept_id": c.concept_id, "name": c.name, "state": c.promotion_state, "confidence": float(c.confidence), "policy_support": float(c.policy_support)}
                for c in sorted(self.concepts.values(), key=lambda item: (float(item.confidence), float(item.policy_support)), reverse=True)[:5]
            ],
            "last_reflection": copylib.deepcopy(self.reflection_log[-1]) if getattr(self, "reflection_log", []) else {},
            "context_step": int(getattr(self, "context_step", 0)),
            "connection_kinds": kinds,
            "causal_kinds": causal_kinds,
            "neural_substrate_version": int(getattr(self, "neural_substrate_version", 4)),
            "neural_memory_cells": len(getattr(self, "neural_cells", {}) or {}),
            "neural_memory_synapses": len(getattr(self, "neural_synapses", []) or []),
            "neural_memory_roles": {role: sum(1 for cell in getattr(self, "neural_cells", {}).values() if cell.role == role) for role in sorted({cell.role for cell in getattr(self, "neural_cells", {}).values()})},
            "neural_synapse_kinds": {kind: sum(1 for syn in getattr(self, "neural_synapses", []) if syn.kind == kind) for kind in sorted({syn.kind for syn in getattr(self, "neural_synapses", [])})},
            "depth": 5 if self.meta_strategies else (4 if self.strategies else (3 if self.concepts else (2 if self.clusters else 1))),
        }

    def causal_plasticity_signals(
        self,
        max_items: int = 32,
        min_abs_utility: float = 0.05,
    ) -> list[dict[str, object]]:
        self.ensure_compatible()
        incoming: dict[str, list[MemoryConnection]] = {}
        outgoing: dict[str, list[MemoryConnection]] = {}
        for conn in self.connections:
            incoming.setdefault(conn.target, []).append(conn)
            outgoing.setdefault(conn.source, []).append(conn)

        def action_for_node(node_id: str) -> tuple[str | None, dict[str, object]]:
            node = self.causal_nodes.get(node_id)
            if node and node.get("kind") == "action":
                return node_id, dict(node.get("payload", {}) or {})
            for conn in incoming.get(node_id, []):
                source = self.causal_nodes.get(conn.source)
                if source and source.get("kind") == "action":
                    return conn.source, dict(source.get("payload", {}) or {})
                action_id, payload = action_for_node(conn.source)
                if action_id is not None:
                    return action_id, payload
            return None, {}

        signals: list[dict[str, object]] = []
        for outcome_id, outcome_node in self.causal_nodes.items():
            if outcome_node.get("kind") != "outcome":
                continue
            payload = dict(outcome_node.get("payload", {}) or {})
            tags = set(str(item) for item in payload.get("lifecycle_tags", []) or [])
            utility = float(payload.get("utility", outcome_node.get("utility", 0.0)) or 0.0)
            td_delta = float(payload.get("td_delta", 0.0) or 0.0)
            reward = float(payload.get("reward", 0.0) or 0.0)
            is_bad = utility < -min_abs_utility or td_delta < -min_abs_utility or reward < -min_abs_utility
            is_lifecycle_bad = bool(tags & {"near_tp_missed", "near_tp_giveback", "gave_back_profit", "closed_after_near_tp_bad"})
            if not (is_bad or is_lifecycle_bad):
                continue

            action_id, action_payload = action_for_node(outcome_id)
            if not action_payload:
                continue
            memory_sources = [
                conn.source for conn in incoming.get(action_id or "", [])
                if conn.source in self.neurons and conn.kind == "causes_action"
            ]
            memory_node = self.neurons.get(memory_sources[0]) if memory_sources else None
            obs_embedding = memory_node.obs_embedding.copy() if memory_node is not None else np.zeros(32, dtype=float)
            action_embedding = memory_node.action_embedding.copy() if memory_node is not None else np.zeros(16, dtype=float)
            severity = float(np.clip(abs(utility) + abs(td_delta) + abs(reward) + float(payload.get("giveback_r", 0.0)) * 0.25, 0.05, 5.0))
            signals.append({
                "outcome_id": outcome_id,
                "action_id": action_id,
                "action_name": str(action_payload.get("action_name", "")),
                "management_mode": str(action_payload.get("management_mode", "")),
                "target_slot": int(action_payload.get("target_slot", payload.get("target_slot", 0)) or 0),
                "tags": sorted(tags),
                "severity": severity,
                "utility": utility,
                "td_delta": td_delta,
                "reward": reward,
                "giveback_r": float(payload.get("giveback_r", 0.0) or 0.0),
                "near_tp": bool(payload.get("near_tp", False)),
                "obs_embedding": obs_embedding,
                "action_embedding": action_embedding,
            })
        signals.sort(key=lambda item: float(item["severity"]), reverse=True)
        return signals[:max_items]


# ── SelfGrowingPolicy (Enhanced) ─────────────────────────────────────────────


@dataclass
class SelfGrowingPolicy:
    input_size: int = INPUT_COUNT
    output_size: int = OUTPUT_COUNT
    modules: list[ModuleState] = field(default_factory=list)
    m1_components: tuple[str, ...] = M1_COMPONENTS
    m1_backend: str = "direct_rnn"
    decision_backend: str = "direct_rnn"
    artifact_version: int = 8
    revision: int = 0
    best_fitness: float = -float("inf")
    growth_events: list[dict[str, object]] = field(default_factory=list)
    _runtime_memory: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    # Directed matrix per (target, source) module. Shapes are
    # (hidden_target, hidden_source), so heterogeneous hidden widths are safe.
    lateral_edges: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    _last_observation: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float), init=False, repr=False)
    gating_mode: str = "sigmoid"
    growth_controller: GrowthController = field(default=None, repr=False)
    plasticity_coeffs: PlasticityCoefficients = field(default=None, repr=False)
    loss_history: list[float] = field(default_factory=list, repr=False)
    active_shadows: list[ShadowBranch] = field(default_factory=list, repr=False)
    experience_buffer: list[dict[str, object]] = field(default_factory=list, repr=False)
    # Bounded evidence retention keeps candidates comparable and prevents
    # historical winners/losers from growing unbounded across trials.
    experience_buffer_max: int = 4096
    rejection_memory: list[dict[str, object]] = field(default_factory=list, repr=False)
    rejection_memory_max: int = 2048
    memory_supernet: ExperienceSuperNetwork = field(default_factory=ExperienceSuperNetwork, repr=False)
    mistake_episode_store: MistakeEpisodeStore = field(default_factory=MistakeEpisodeStore, repr=False)
    curiosity_system: CuriosityNeuronSystem = field(default_factory=CuriosityNeuronSystem, repr=False)
    neural_development: NeuralDevelopmentController = field(default_factory=NeuralDevelopmentController, repr=False)
    neural_utility: NeuralUtilityNetwork = field(default_factory=NeuralUtilityNetwork.seed, repr=False)
    risk_cost_gate: NeuralRiskCostGate = field(default_factory=NeuralRiskCostGate.seed, repr=False)
    action_parameter_head: NeuralActionParameterHead = field(default_factory=NeuralActionParameterHead.seed, repr=False)
    reasoning_graph: ReasoningGraphSystem = field(default_factory=ReasoningGraphSystem.seed, repr=False)
    _step_count: int = field(default=0, init=False, repr=False)
    _reward_modulator: float = field(default=0.0, init=False, repr=False)
    _experience_starvation: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    _outcome_starvation: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    # Graph-only candidate orchestration.  It retains role priors/audit but
    # never owns a second production graph; candidate graphs stay isolated.
    graph_supernetwork: GraphSupernetworkRuntime | None = field(default=None, repr=False)
    graph_supernetwork_config: dict[str, object] = field(default_factory=dict, repr=False)
    # Set only on a copied training candidate. Trial micro-growth is never
    # permitted to change a baseline policy's topology in-place.
    graph_trial_id: str = field(default="", repr=False)
    graph_trial_mode: bool = field(default=False, repr=False)
    # Persisted replacement-policy controls. Candidate copies retain these
    # values, unlike the former late-bound v11 attributes.
    parameter_replacement_learning: bool = field(default=True, repr=False)
    parameter_replacement_min_samples: int = field(default=3, repr=False)
    parameter_replacement_min_confidence: float = field(default=0.55, repr=False)
    child_supernetwork_manager: ChildSupernetworkManager | None = field(default=None, repr=False)

    @classmethod
    def seed(
        cls,
        input_size: int = INPUT_COUNT,
        output_size: int = OUTPUT_COUNT,
        hidden_size: int = 24,
        seed: int | None = None,
        *,
        hidden_layers: int = 1,
        m1_backend: str | None = None,
    ) -> "SelfGrowingPolicy":
        """Create a policy with an explicit decision backend."""
        default_backend = "direct_rnn" if int(input_size) == INPUT_COUNT and int(output_size) == OUTPUT_COUNT else "m1_graph"
        decision_backend = _normalize_m1_backend(m1_backend or default_backend)
        rng = np.random.default_rng(seed)
        module = ModuleState.create(
            "m1_thinking",
            input_size,
            hidden_size,
            output_size,
            rng,
            role="m1_thinking_core",
            hidden_layers=hidden_layers,
        )
        module.lifecycle = MODULE_LIFECYCLE_ACTIVE
        if decision_backend == "m1_graph":
            module.m1_component = M1GraphModule.seed(
                input_size=input_size,
                output_size=output_size,
                state_size=hidden_size,
                K=4,
                seed=seed,
            )
        return cls(
            input_size=input_size,
            output_size=output_size,
            modules=[module],
            m1_components=M1_COMPONENTS,
            m1_backend=decision_backend,
            decision_backend=decision_backend,
            artifact_version=9,
            growth_controller=GrowthController(),
            plasticity_coeffs=PlasticityCoefficients(),
            curiosity_system=CuriosityNeuronSystem.seed(rng=rng),
            neural_development=NeuralDevelopmentController.seed(rng=rng),
            neural_utility=NeuralUtilityNetwork.seed(rng=rng),
            risk_cost_gate=NeuralRiskCostGate.seed(rng=rng),
            action_parameter_head=NeuralActionParameterHead.seed(rng=rng),
            reasoning_graph=ReasoningGraphSystem.seed(rng=rng),
            graph_supernetwork=GraphSupernetworkRuntime(),
        )

    def copy(self) -> "SelfGrowingPolicy":
        gc_copy = GrowthController(
            tau_grow=self.growth_controller.tau_grow,
            tau_prune=self.growth_controller.tau_prune,
            tau_module=self.growth_controller.tau_module,
            tau_split=self.growth_controller.tau_split,
            tau_merge=self.growth_controller.tau_merge,
            H=self.growth_controller.H,
            K_max=self.growth_controller.K_max,
            N_max=self.growth_controller.N_max,
            n_min=self.growth_controller.n_min,
            coeffs=self.growth_controller.coeffs,
        ) if self.growth_controller is not None else None
        copied = SelfGrowingPolicy(
            input_size=self.input_size,
            output_size=self.output_size,
            modules=[module.copy() for module in self.modules],
            lateral_edges={key: value.copy() for key, value in getattr(self, "lateral_edges", {}).items()},
            m1_components=tuple(self.m1_components),
            m1_backend=_normalize_m1_backend(getattr(self, "decision_backend", getattr(self, "m1_backend", "direct_rnn"))),
            decision_backend=_normalize_m1_backend(getattr(self, "decision_backend", getattr(self, "m1_backend", "direct_rnn"))),
            artifact_version=int(getattr(self, "artifact_version", 1)),
            revision=self.revision,
            best_fitness=float(self.best_fitness),
            growth_events=[dict(event) for event in self.growth_events],
            gating_mode=self.gating_mode,
            growth_controller=gc_copy,
            plasticity_coeffs=self.plasticity_coeffs,
            loss_history=list(self.loss_history),
            active_shadows=list(self.active_shadows),
            experience_buffer=[dict(item) for item in getattr(self, "experience_buffer", [])],
            experience_buffer_max=max(1, int(getattr(self, "experience_buffer_max", 4096) or 4096)),
            rejection_memory=[dict(item) for item in getattr(self, "rejection_memory", [])],
            rejection_memory_max=max(1, int(getattr(self, "rejection_memory_max", 2048) or 2048)),
            memory_supernet=self.memory_supernet.copy() if hasattr(self, "memory_supernet") and self.memory_supernet is not None else ExperienceSuperNetwork(),
            mistake_episode_store=(
                self.mistake_episode_store.copy()
                if hasattr(self, "mistake_episode_store") and self.mistake_episode_store is not None else MistakeEpisodeStore()
            ),
            curiosity_system=self.curiosity_system.copy() if hasattr(self, "curiosity_system") and self.curiosity_system is not None else CuriosityNeuronSystem.seed(),
            neural_development=self.neural_development.copy() if hasattr(self, "neural_development") and self.neural_development is not None else NeuralDevelopmentController.seed(),
            neural_utility=self.neural_utility.copy() if hasattr(self, "neural_utility") and self.neural_utility is not None else NeuralUtilityNetwork.seed(),
            risk_cost_gate=self.risk_cost_gate.copy() if hasattr(self, "risk_cost_gate") and self.risk_cost_gate is not None else NeuralRiskCostGate.seed(),
            action_parameter_head=self.action_parameter_head.copy() if hasattr(self, "action_parameter_head") and self.action_parameter_head is not None else NeuralActionParameterHead.seed(),
            reasoning_graph=self.reasoning_graph.copy() if hasattr(self, "reasoning_graph") and self.reasoning_graph is not None else ReasoningGraphSystem.seed(),
            graph_supernetwork=self.graph_supernetwork.copy() if getattr(self, "graph_supernetwork", None) is not None else None,
            graph_supernetwork_config=dict(getattr(self, "graph_supernetwork_config", {}) or {}),
            graph_trial_id=str(getattr(self, "graph_trial_id", "") or ""),
            graph_trial_mode=bool(getattr(self, "graph_trial_mode", False)),
            parameter_replacement_learning=bool(getattr(self, "parameter_replacement_learning", True)),
            parameter_replacement_min_samples=max(1, int(getattr(self, "parameter_replacement_min_samples", 3) or 3)),
            parameter_replacement_min_confidence=float(np.clip(getattr(self, "parameter_replacement_min_confidence", 0.55) or 0.55, 0.05, 0.95)),
            child_supernetwork_manager=(
                self.child_supernetwork_manager.copy()
                if getattr(self, "child_supernetwork_manager", None) is not None else None
            ),
        )
        copied._experience_starvation = dict(getattr(self, "_experience_starvation", {}) or {})
        copied._outcome_starvation = dict(getattr(self, "_outcome_starvation", {}) or {})
        return copied

    def reset_runtime_state(self) -> None:
        self._runtime_memory = {
            module.name: np.zeros(module.hidden_size, dtype=float)
            for module in self.modules
        }
        for module in self.modules:
            if self._uses_m1_graph() and isinstance(module.m1_component, M1GraphModule):
                module.m1_component.reset_state()

    def configure_learning_integrity(self, settings: object | None = None) -> None:
        """Install run-level bounded evidence controls on a policy artifact."""
        if settings is None:
            return
        hidden_normalization_mode = _mandatory_hidden_normalization_mode(
            getattr(settings, "hidden_normalization_mode", REQUIRED_HIDDEN_NORMALIZATION_MODE)
        )
        hidden_normalization_enabled = True
        hidden_normalization_target_rms = float(np.clip(getattr(settings, "hidden_normalization_target_rms", 1.0) or 1.0, 1e-6, 10.0))
        hidden_normalization_clip = float(np.clip(getattr(settings, "hidden_normalization_clip", 3.0) or 3.0, 1e-6, 20.0))
        hidden_normalization_eps = float(np.clip(getattr(settings, "hidden_normalization_eps", 1e-5) or 1e-5, 1e-9, 1.0))
        configured_layer_count = max(
            [1 + len(getattr(module, "deep_w", ()) or ()) for module in getattr(self, "modules", ()) or ()]
            or [1]
        )
        hidden_normalization_layers = tuple(True for _ in range(configured_layer_count))
        hidden_dropout_enabled = True
        hidden_dropout_rates = mandatory_layerwise_rates(
            getattr(settings, "hidden_dropout_rates", REQUIRED_HIDDEN_DROPOUT_RATES),
            configured_layer_count,
        )
        hidden_dropout_seed = max(0, int(getattr(settings, "hidden_dropout_seed", 0) or 0))
        hidden_dropout_train_only = True
        for module_index, module in enumerate(getattr(self, "modules", ()) or ()):
            layer_count = 1 + len(getattr(module, "deep_w", ()) or ())
            module.hidden_normalization_mode = hidden_normalization_mode
            module.hidden_normalization_enabled = hidden_normalization_enabled
            module.hidden_normalization_target_rms = hidden_normalization_target_rms
            module.hidden_normalization_clip = hidden_normalization_clip
            module.hidden_normalization_eps = hidden_normalization_eps
            module.hidden_normalization_layers = layerwise_bools(
                hidden_normalization_layers,
                layer_count,
                default=True,
            )
            module.hidden_dropout_enabled = hidden_dropout_enabled
            module.hidden_dropout_rates = mandatory_layerwise_rates(hidden_dropout_rates, layer_count)
            module.hidden_dropout_seed = (hidden_dropout_seed + module_index) & 0x7FFFFFFF
            module.hidden_dropout_train_only = hidden_dropout_train_only
        self.experience_buffer_max = max(1, int(getattr(settings, "experience_buffer_max", 4096) or 4096))
        self.rejection_memory_max = max(1, int(getattr(settings, "rejection_memory_max", 2048) or 2048))
        self._execution_trace_cache_max = max(1, int(getattr(settings, "execution_trace_cache_max", 8192) or 8192))
        memory_neurons_max = max(1, int(getattr(settings, "memory_neurons_max", 4096) or 4096))
        memory_connections_max = max(1, int(getattr(settings, "memory_connections_max", 16384) or 16384))
        mistake_episodes_max = max(1, int(getattr(settings, "mistake_episodes_max", 2048) or 2048))
        if len(self.experience_buffer) > self.experience_buffer_max:
            self.experience_buffer = list(self.experience_buffer[-self.experience_buffer_max:])
        if len(self.rejection_memory) > self.rejection_memory_max:
            self.rejection_memory = list(self.rejection_memory[-self.rejection_memory_max:])
        self.ensure_compatible(target_input_size=self.input_size)
        self.memory_supernet.max_neurons = memory_neurons_max
        self.memory_supernet.max_connections = memory_connections_max
        self.memory_supernet.ensure_compatible()
        self.mistake_episode_store.max_episodes = mistake_episodes_max
        self.mistake_episode_store.ensure_compatible()
        # Apply quotas immediately to migrated/long-running candidates instead
        # of waiting for a later development tick.
        self.memory_supernet.develop()

    def configure_graph_supernetwork(self, settings: object | None = None) -> None:
        """Apply bounded graph-growth controls to an isolated candidate."""
        if settings is None:
            return
        if not self._uses_m1_graph():
            return
        names = (
            "enable_child_supernet", "max_child_networks", "max_generation",
            "max_children_per_network", "diversity_threshold", "max_cross_module_edges",
            "max_m1_module_edges", "max_cross_module_recurrent_edges", "max_cross_module_lateral_edges",
            "recurrent_loop_limit", "max_graph_neurons", "max_graph_connections", "max_hidden_neurons_per_module",
            "max_hidden_layers_per_module", "hidden_depth_growth_threshold",
            "cross_module_enabled", "allowed_cross_module_edge_kinds",
            "allowed_module_to_m1_targets", "allowed_m1_to_module_targets", "allowed_specialist_roles",
            "flexible_cross_module_connectivity", "flexible_connection_fanout", "flexible_peer_module_fanout",
            "module_role_budget", "prediction_need_threshold", "memory_need_threshold",
            "risk_need_threshold", "position_need_threshold", "cadence_need_threshold", "role_cooldown",
            "require_mt5_promotion", "enable_trial_micro_growth", "micro_growth_max_neurons_per_trial",
            "micro_growth_learning_threshold", "micro_growth_initial_gate", "micro_growth_initial_weight",
            "micro_growth_survival_contexts", "micro_growth_survival_similar_outcomes",
            "micro_growth_context_similarity_threshold", "micro_growth_context_prototype_lr",
            "micro_growth_survival_positive_supports", "micro_growth_survival_utility",
            "micro_growth_consolidate_utility", "micro_growth_fade_rate",
            "micro_growth_prune_fade_threshold", "micro_growth_grace_outcomes",
            "micro_growth_max_modules_per_trial", "specialist_module_initial_gate",
            "specialist_module_initial_weight", "specialist_module_input_adapters",
            "specialist_module_core_neurons", "specialist_module_output_adapters",
            "specialist_module_enable_bypass_interfaces",
            "specialist_min_causal_confidence_to_activate",
            "specialist_require_causal_validation_to_consolidate",
        )
        values = {name: getattr(settings, name) for name in names if hasattr(settings, name)}
        self.graph_supernetwork_config.update(values)
        self.m1_backend = "m1_graph"
        self.decision_backend = "m1_graph"
        if self.graph_supernetwork is None:
            self.graph_supernetwork = GraphSupernetworkRuntime()
        runtime = self.graph_supernetwork
        runtime.role_budget = values.get("module_role_budget", None)
        if runtime.role_budget is not None and int(runtime.role_budget) <= 0:
            runtime.role_budget = None
        connector = runtime.connector
        connector.enabled = bool(values.get("cross_module_enabled", connector.enabled))
        for attr, key in (("allowed_edge_kinds", "allowed_cross_module_edge_kinds"), ("allowed_roles", "allowed_specialist_roles")):
            value = values.get(key)
            if value is not None:
                setattr(connector, attr, frozenset(str(item) for item in value))
        if "flexible_cross_module_connectivity" in values:
            connector.flexible_connectivity = bool(values["flexible_cross_module_connectivity"])
        # 0/null has the same open-ended meaning across all structural fields.
        for attr, key in (("flexible_max_edges", "flexible_connection_fanout"), ("flexible_peer_max_edges", "flexible_peer_module_fanout")):
            value = values.get(key)
            setattr(connector, attr, 0 if value is None else max(0, int(value)))
        thresholds = [float(values[key]) for key in (
            "prediction_need_threshold", "memory_need_threshold", "risk_need_threshold",
            "position_need_threshold", "cadence_need_threshold",
        ) if key in values]
        if thresholds:
            runtime.need_threshold = float(np.clip(np.mean(thresholds), 0.0, 1.0))
        trial_growth = runtime.trial_growth
        mapping = {
            "enabled": "enable_trial_micro_growth",
            "max_new_neurons_per_trial": "micro_growth_max_neurons_per_trial",
            "learning_threshold": "micro_growth_learning_threshold",
            "initial_gate": "micro_growth_initial_gate",
            "initial_weight": "micro_growth_initial_weight",
            "survival_contexts": "micro_growth_survival_contexts",
            "survival_similar_outcomes": "micro_growth_survival_similar_outcomes",
            "context_similarity_threshold": "micro_growth_context_similarity_threshold",
            "context_prototype_lr": "micro_growth_context_prototype_lr",
            "survival_positive_supports": "micro_growth_survival_positive_supports",
            "survival_utility": "micro_growth_survival_utility",
            "consolidate_utility": "micro_growth_consolidate_utility",
            "fade_rate": "micro_growth_fade_rate",
            "prune_fade_threshold": "micro_growth_prune_fade_threshold",
            "grace_outcomes": "micro_growth_grace_outcomes",
            "max_new_modules_per_trial": "micro_growth_max_modules_per_trial",
            "module_initial_gate": "specialist_module_initial_gate",
            "module_initial_weight": "specialist_module_initial_weight",
            "module_input_adapters": "specialist_module_input_adapters",
            "module_core_neurons": "specialist_module_core_neurons",
            "module_output_adapters": "specialist_module_output_adapters",
            "module_enable_bypass_interfaces": "specialist_module_enable_bypass_interfaces",
            "min_causal_confidence_to_activate": "specialist_min_causal_confidence_to_activate",
            "require_causal_validation_to_consolidate": "specialist_require_causal_validation_to_consolidate",
        }
        for attribute, config_name in mapping.items():
            if config_name in values:
                setattr(trial_growth, attribute, values[config_name])
        trial_growth.__post_init__()
        if bool(values.get("enable_child_supernet", False)) and self.child_supernetwork_manager is None:
            root_id = f"m1-graph-root-{self.revision}-{time.time_ns()}"
            self.child_supernetwork_manager = ChildSupernetworkManager.root(
                network_id=root_id,
                max_alive_networks=max(1, int(values.get("max_child_networks", 4))),
                max_children=max(1, int(values.get("max_child_networks", 4))),
                max_generation=values.get("max_generation"),
                max_children_per_network=values.get("max_children_per_network"),
                diversity_threshold=float(values.get("diversity_threshold", 0.05)),
            )
        for _module, m1 in self._graph_m1_modules():
            graph = m1.graph

            def _bounded_cap(raw: object, *, existing: int, fallback: int) -> int:
                """Never install a cap below the immutable seeded topology.

                A config cap controls *new* growth. Old artifacts and graph
                blueprints already contain recurrent/lateral wiring; lowering a
                cap below that count must not turn a valid policy invalid at
                load time. The effective cap is therefore at least the current
                audited count, and future additions are still rejected.
                """
                try:
                    requested = int(raw) if raw is not None else int(fallback)
                except (TypeError, ValueError):
                    requested = int(fallback)
                if requested <= 0:
                    requested = int(fallback)
                return max(int(existing), max(1, requested))

            existing_recurrent = graph._count_edges(kind="recurrent")
            existing_lateral = graph._count_edges(kind="lateral")
            existing_cross = graph._count_cross_module_edges()
            existing_m1_module = graph._count_m1_module_edges()
            existing_cross_recurrent = graph._count_cross_module_edges(kind="recurrent")
            existing_cross_lateral = graph._count_cross_module_edges(kind="lateral")
            cross_enabled = bool(connector.enabled)
            graph.budget = NeuralGraphBudget(
                max_neurons=_bounded_cap(values.get("max_graph_neurons", 512), existing=len(graph.neurons), fallback=512),
                max_connections=_bounded_cap(values.get("max_graph_connections", 16384), existing=len(graph.edges), fallback=16384),
                max_recurrent_edges=_bounded_cap(values.get("recurrent_loop_limit", 96), existing=existing_recurrent, fallback=96),
                max_lateral_edges=_bounded_cap(values.get("recurrent_loop_limit", 96), existing=existing_lateral, fallback=96),
                # A disabled connector blocks all *new* cross-module wiring;
                # preserve any pre-existing audited edges only so a migrated
                # artifact does not fail invariant checks on load.
                max_cross_module_edges=(existing_cross if not cross_enabled else _bounded_cap(values.get("max_cross_module_edges", 96), existing=existing_cross, fallback=96)),
                max_m1_module_edges=(existing_m1_module if not cross_enabled else _bounded_cap(values.get("max_m1_module_edges", 96), existing=existing_m1_module, fallback=96)),
                max_cross_module_recurrent_edges=(existing_cross_recurrent if not cross_enabled else _bounded_cap(values.get("max_cross_module_recurrent_edges", 64), existing=existing_cross_recurrent, fallback=64)),
                max_cross_module_lateral_edges=(existing_cross_lateral if not cross_enabled else _bounded_cap(values.get("max_cross_module_lateral_edges", 64), existing=existing_cross_lateral, fallback=64)),
            )
            module_to_m1 = values.get("allowed_module_to_m1_targets")
            if module_to_m1 is not None:
                m1.graph.allowed_module_m1_targets = frozenset(str(item) for item in module_to_m1)
            m1_to_module = values.get("allowed_m1_to_module_targets")
            if m1_to_module is not None:
                m1.graph.allowed_m1_module_targets = frozenset(str(item) for item in m1_to_module)
            m1.graph.assert_invariants()

    def convert_m1_to_graph_backend(self) -> None:
        """One-way compatibility migration into a fresh graph-native M1.

        Pre-graph artifacts cannot retain retired-core internals after graph-only migration.
        Their outer policy metadata and memories survive, while the M1 core is
        re-seeded as a graph and must pass normal candidate evaluation before
        promotion.
        """
        for module in self.modules:
            if not isinstance(getattr(module, "m1_component", None), M1GraphModule):
                module.m1_component = M1GraphModule.seed(
                    input_size=module.input_size,
                    output_size=module.output_size,
                    state_size=module.hidden_size,
                    K=4,
                    seed=(self.revision + len(module.name)) & 0x7FFFFFFF,
                )
                self.growth_events.append({
                    "round": self.revision,
                    "type": "graph_core_reinitialized",
                    "module": module.name,
                    "reason": "retired_pre_graph_artifact",
                })
        self.m1_backend = "m1_graph"
        self.decision_backend = "m1_graph"
        if self.graph_supernetwork is None:
            self.graph_supernetwork = GraphSupernetworkRuntime()

    @staticmethod
    def _resize_vector(values: np.ndarray, size: int, fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size == size:
            return arr.copy()
        resized = np.full(size, fill, dtype=float)
        resized[: min(arr.size, size)] = arr[:size]
        return resized

    @staticmethod
    def _resize_matrix_columns(values: np.ndarray, cols: int, fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Expected 2D matrix for column resize.")
        if arr.shape[1] == cols:
            return arr.copy()
        resized = np.full((arr.shape[0], cols), fill, dtype=float)
        kept = min(arr.shape[1], cols)
        resized[:, :kept] = arr[:, :kept]
        return resized

    @staticmethod
    def _resize_matrix_rows(values: np.ndarray, rows: int, fill: float = 0.0) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2:
            raise ValueError("Expected 2D matrix for row resize.")
        if arr.shape[0] == rows:
            return arr.copy()
        resized = np.full((rows, arr.shape[1]), fill, dtype=float)
        kept = min(arr.shape[0], rows)
        resized[:kept, :] = arr[:kept, :]
        return resized

    def _resize_module_input(self, module: ModuleState, target_input_size: int) -> None:
        if module.input_size == target_input_size and module.w_in.shape[1] == target_input_size:
            return
        old_input_size = int(module.input_size)
        module.w_in = self._resize_matrix_columns(module.w_in, target_input_size, fill=0.0)
        module.gate_w = self._resize_vector(module.gate_w, target_input_size, fill=0.0)
        if target_input_size > old_input_size:
            seed_material = f"{module.name}:{old_input_size}:{target_input_size}:feature-v10".encode("utf-8")
            seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "little", signed=False)
            rng = np.random.default_rng(seed)
            scale = 0.02 / np.sqrt(max(1, target_input_size))
            module.w_in[:, old_input_size:target_input_size] = rng.normal(
                0.0, scale, size=(module.hidden_size, target_input_size - old_input_size)
            )
            module.gate_w[old_input_size:target_input_size] = rng.normal(
                0.0, scale, size=target_input_size - old_input_size
            )
        module.p_stp = self._resize_matrix_columns(
            module.p_stp if module.p_stp is not None else np.ones((module.hidden_size, old_input_size), dtype=float),
            target_input_size,
            fill=1.0,
        )
        module.input_mask = self._resize_vector(
            module.input_mask if module.input_mask is not None else np.ones(old_input_size, dtype=float),
            target_input_size,
            fill=1.0,
        )
        module.input_size = target_input_size
        if module.m1_component is not None:
            m1 = module.m1_component
            if isinstance(m1, M1GraphModule):
                m1.resize_input(target_input_size)
                return
            m1.state_encoder.w = self._resize_matrix_columns(m1.state_encoder.w, target_input_size, fill=0.0)
            m1.working_memory.g_keep_w = self._resize_vector(m1.working_memory.g_keep_w, target_input_size, fill=0.0)
            m1.working_memory.g_write_w = self._resize_vector(m1.working_memory.g_write_w, target_input_size, fill=0.0)
            m1.working_memory.g_erase_w = self._resize_vector(m1.working_memory.g_erase_w, target_input_size, fill=0.0)
            m1.working_memory.w_write = self._resize_matrix_columns(m1.working_memory.w_write, target_input_size, fill=0.0)

    def _resize_module_output(self, module: ModuleState, target_output_size: int) -> None:
        if module.output_size == target_output_size and module.w_out.shape[0] == target_output_size:
            return
        old_output_size = int(module.output_size)
        module.w_out = self._resize_matrix_rows(module.w_out, target_output_size, fill=0.0)
        module.b_out = self._resize_vector(module.b_out, target_output_size, fill=0.0)
        module.output_mask = self._resize_vector(
            module.output_mask if module.output_mask is not None else np.ones(old_output_size, dtype=float),
            target_output_size,
            fill=1.0,
        )
        module.output_size = target_output_size
        if module.m1_component is not None:
            m1 = module.m1_component
            if isinstance(m1, M1GraphModule):
                m1.resize_output(target_output_size)
                return
            m1.hypothesis_bank.w_decode = self._resize_matrix_rows(m1.hypothesis_bank.w_decode, target_output_size, fill=0.0)
            m1.hypothesis_bank.b_decode = self._resize_vector(m1.hypothesis_bank.b_decode, target_output_size, fill=0.0)
            m1.critic.w_goal = self._resize_matrix_columns(m1.critic.w_goal, target_output_size, fill=0.0)

    @staticmethod
    def _lateral_key(target_name: str, source_name: str) -> str:
        return f"{target_name}<-{source_name}"

    @staticmethod
    def _resize_matrix_shape(values: np.ndarray | None, rows: int, cols: int) -> np.ndarray:
        out = np.zeros((rows, cols), dtype=float)
        if values is None:
            return out
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 2:
            out[: min(rows, arr.shape[0]), : min(cols, arr.shape[1])] = arr[:rows, :cols]
        return out

    def _ensure_lateral_edges(self) -> None:
        current = getattr(self, "lateral_edges", {}) or {}
        module_by_name = {module.name: module for module in self.modules}
        repaired: dict[str, np.ndarray] = {}
        for target in self.modules:
            for source in self.modules:
                if target.name == source.name:
                    continue
                key = self._lateral_key(target.name, source.name)
                old = current.get(key)
                # Backward migration: old models stored a same-width matrix on
                # the source module.  Use it only when the shape is unambiguous.
                if old is None and source.w_lat is not None and source.w_lat.shape == (target.hidden_size, source.hidden_size):
                    old = source.w_lat
                repaired[key] = self._resize_matrix_shape(old, target.hidden_size, source.hidden_size)
        self.lateral_edges = repaired

    def _masked_module_observation(self, module: ModuleState, observation: np.ndarray) -> np.ndarray:
        obs = self._resize_vector(np.asarray(observation, dtype=float), module.input_size)
        if module.input_mask is None or module.input_mask.size != module.input_size:
            module.input_mask = np.ones(module.input_size, dtype=float)
        return obs * np.clip(module.input_mask, 0.0, 1.0)

    def _integrate_lateral(
        self,
        hidden_states: list[np.ndarray],
        raw_outputs: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        if len(self.modules) <= 1:
            return hidden_states, raw_outputs
        self._ensure_lateral_edges()
        updated_hidden = [np.asarray(hidden, dtype=float).copy() for hidden in hidden_states]
        updated_raw = [np.asarray(raw, dtype=float).copy() for raw in raw_outputs]
        for target_idx, target in enumerate(self.modules):
            cross_input = np.zeros(target.hidden_size, dtype=float)
            for source_idx, source in enumerate(self.modules):
                if source_idx == target_idx:
                    continue
                edge = self.lateral_edges.get(self._lateral_key(target.name, source.name))
                if edge is not None:
                    cross_input += edge @ hidden_states[source_idx]
            lateral_delta = 0.10 * cross_input
            updated_hidden[target_idx] = np.clip(updated_hidden[target_idx] + lateral_delta, -3.0, 3.0)
            # Feed lateral integration into the current output as well as the
            # recurrent state for the next step.
            residual = target.w_out @ lateral_delta
            if target.output_mask is not None and target.output_mask.size == residual.size:
                residual = residual * target.output_mask
            updated_raw[target_idx] = updated_raw[target_idx] + residual
        return updated_hidden, updated_raw

    def structural_complexity(self) -> float:
        """Evidence-priced complexity across dense modules, graph and memory.

        This is deliberately a soft survival cost rather than a topology cap.
        A candidate can grow without a fixed ceiling, but each retained neuron,
        synapse, memory proposal and burst of structural churn must earn enough
        out-of-sample utility to offset its operational cost.
        """
        self.ensure_compatible()
        dense_edges = 0
        hidden = 0
        for module in self.modules:
            hidden_layers = 1 + len(getattr(module, "deep_w", ()) or ())
            hidden += int(module.hidden_size) * int(hidden_layers)
            dense_edges += int(np.count_nonzero(np.abs(module.w_in) > 0.005))
            dense_edges += int(np.count_nonzero(np.abs(module.w_rec) > 0.005))
            dense_edges += sum(int(np.count_nonzero(np.abs(layer) > 0.005)) for layer in (getattr(module, "deep_w", ()) or ()))
            dense_edges += sum(int(np.count_nonzero(np.abs(layer) > 0.005)) for layer in (getattr(module, "deep_rec", ()) or ()))
            dense_edges += int(np.count_nonzero(np.abs(module.w_out) > 0.005))
        self._ensure_lateral_edges()
        dense_edges += sum(int(np.count_nonzero(np.abs(edge) > 0.005)) for edge in self.lateral_edges.values())

        graph_nodes = graph_edges = disabled_graph_edges = 0
        for module in self.modules:
            m1 = getattr(module, "m1_component", None)
            graph = getattr(m1, "graph", None)
            if graph is None:
                continue
            graph_nodes += len(getattr(graph, "neurons", {}) or {})
            for edge in (getattr(graph, "edges", {}) or {}).values():
                if bool(getattr(edge, "enabled", True)):
                    graph_edges += 1
                else:
                    disabled_graph_edges += 1

        memory = getattr(self, "memory_supernet", None)
        memory_cells = len(getattr(memory, "neural_cells", {}) or {})
        memory_synapses = len(getattr(memory, "neural_synapses", []) or [])
        # Recently changing structures are expensive until they prove stable;
        # the event window makes oscillatory grow/prune behaviour visible to
        # model selection without permanently punishing old, consolidated work.
        churn_types = {
            "graph_trial_micro_growth", "graph_candidate_proposal", "expand_module",
            "merge_modules", "prune_module", "graph_compaction",
        }
        churn = sum(
            1 for event in list(getattr(self, "growth_events", []) or [])[-256:]
            if str(event.get("type", "")) in churn_types
        )
        return float(
            len(self.modules)
            + 0.02 * hidden
            + 0.00025 * dense_edges
            + 0.025 * graph_nodes
            + 0.0025 * graph_edges
            + 0.00010 * disabled_graph_edges
            + 0.010 * memory_cells
            + 0.0015 * memory_synapses
            + 0.020 * churn
        )

    def _apply_fitness_credit(self, module: ModuleState, credit: float, observation: np.ndarray) -> None:
        """Apply low-amplitude module structural credit only.

        Window fitness cannot identify which active neuron caused the selected
        action. Per-neuron utility is updated exclusively by execution/lifecycle
        attribution below.
        """
        hidden = self._runtime_memory.get(module.name, np.zeros(module.hidden_size, dtype=float))
        hidden = self._resize_vector(hidden, module.hidden_size)
        pre = self._masked_module_observation(module, observation)
        if module.input_eligibility is None or module.input_eligibility.shape != (module.hidden_size, module.input_size):
            module.input_eligibility = np.zeros((module.hidden_size, module.input_size), dtype=float)
        eligibility = np.outer(hidden, pre)
        module.input_eligibility = 0.95 * module.input_eligibility + 0.05 * eligibility
        module.credit_ema = 0.90 * float(getattr(module, "credit_ema", 0.0)) + 0.10 * float(credit)
        lr = float(np.clip(0.0004 * module.plasticity, 0.000005, 0.001))
        if abs(credit) > 1e-9:
            frozen = np.asarray(getattr(module, "neuron_frozen", np.zeros(module.hidden_size, dtype=bool)), dtype=bool)
            update_rows = ~frozen
            if np.any(update_rows):
                module.w_in[update_rows] += lr * credit * module.input_eligibility[update_rows]
                module.w_rec[update_rows] += 0.10 * lr * credit * np.outer(hidden, hidden)[update_rows]
        if module.input_mask is not None:
            module.w_in[:, module.input_mask < 0.5] = 0.0
        if module.output_mask is not None:
            module.w_out[module.output_mask < 0.5, :] = 0.0

    @staticmethod
    def _trace_action_index(trace: StepTrace, action_count: int) -> int:
        details = dict(getattr(trace, "action_details", {}) or {})
        for key in ("bridge_action", "requested_action", "action"):
            try:
                value = int(details.get(key, -1))
            except (TypeError, ValueError):
                value = -1
            if 0 <= value < action_count:
                return value
        values = np.asarray(trace.selected_action, dtype=float).reshape(-1)
        if values.size <= 0:
            return 0
        return int(np.argmax(values[: min(action_count, values.size)]))

    @staticmethod
    def _runner_up_action(trace: StepTrace, selected_action: int, action_count: int) -> int:
        values = np.asarray(trace.selected_action, dtype=float).reshape(-1)
        n = min(action_count, values.size)
        if n <= 1:
            return selected_action
        scores = values[:n].copy()
        if 0 <= selected_action < n:
            scores[selected_action] = -np.inf
        return int(np.argmax(scores))

    @staticmethod
    def _neuron_redundancy(module: ModuleState) -> np.ndarray:
        """Output-vector redundancy in [0,1]; high values signal duplicates."""
        h = module.hidden_size
        if h <= 1 or module.w_out.size == 0:
            return np.zeros(h, dtype=float)
        vectors = np.asarray(module.w_out, dtype=float).T
        norms = np.linalg.norm(vectors, axis=1)
        redundancy = np.zeros(h, dtype=float)
        for i in range(h):
            if norms[i] <= 1e-12:
                continue
            similarities = []
            for j in range(h):
                if i == j or norms[j] <= 1e-12:
                    continue
                similarities.append(abs(float(vectors[i] @ vectors[j] / (norms[i] * norms[j]))))
            redundancy[i] = max(similarities) if similarities else 0.0
        return np.clip(redundancy, 0.0, 1.0)

    def _cache_execution_trace(self, trace: StepTrace) -> None:
        trace_id = str(getattr(trace, "trace_id", "") or "")
        if not trace_id:
            return
        # Pickles/journals created before fitness-v5 do not contain these
        # ephemeral fields.  Recreate them lazily before reconciliation.
        if not hasattr(self, "_execution_trace_cache") or not isinstance(getattr(self, "_execution_trace_cache", None), dict):
            self._execution_trace_cache = {}
        if not hasattr(self, "_execution_trace_cache_max"):
            self._execution_trace_cache_max = max(1, int(getattr(self, "_execution_trace_cache_max", 8192) or 8192))
        self._execution_trace_cache[trace_id] = copylib.deepcopy(trace)
        max_items = getattr(self, "_execution_trace_cache_max", None)
        if max_items is not None and int(max_items) > 0 and len(self._execution_trace_cache) > int(max_items):
            for stale in list(self._execution_trace_cache)[:len(self._execution_trace_cache) - int(max_items)]:
                self._execution_trace_cache.pop(stale, None)

    def _record_mistake_episode_from_trace(
        self,
        trace: StepTrace,
        next_obs: np.ndarray | None,
        reward: float,
        cost: float,
        signal: dict[str, object],
        *,
        source: str = "journal",
        metrics: dict[str, object] | None = None,
        reject_reasons: list[str] | tuple[str, ...] = (),
    ) -> dict[str, object] | None:
        self.ensure_compatible(target_input_size=self.input_size)
        td_delta = float(signal.get("td_delta", 0.0) or 0.0)
        prediction_error = float(signal.get("prediction_error", 0.0) or 0.0)
        utility = td_delta + float(reward) - float(cost) - 0.1 * prediction_error
        lifecycle_tags = set(str(item) for item in tuple(getattr(trace, "lifecycle_tags", ()) or ()))
        lifecycle_tags.update(
            str(item)
            for item in dict(getattr(trace, "execution_details", {}) or {}).get("lifecycle_tags", []) or []
        )
        lifecycle_bad = bool(lifecycle_tags & {
            "near_tp_missed",
            "near_tp_giveback",
            "gave_back_profit",
            "closed_after_near_tp_bad",
            "should_have_partial_or_trail",
            "stress_entry_loss",
        })
        has_rejection = bool(reject_reasons)
        if utility >= -0.05 and float(reward) >= 0.0 and td_delta >= 0.0 and not lifecycle_bad and not has_rejection:
            return None
        episode = episode_from_trace(
            trace,
            next_obs,
            float(reward),
            float(cost),
            signal,
            source=source,
            metrics=metrics,
            reject_reasons=reject_reasons,
        )
        stored = self.mistake_episode_store.add(episode)
        event = {
            "round": self.revision,
            "type": "mistake_episode_recorded",
            "episode_id": str(stored.episode_id),
            "trace_id": str(stored.trace_id),
            "source": str(stored.source),
            "action": int(stored.action_index),
            "action_name": str(stored.action_name),
            "tags": list(stored.mistake_tags),
            "severity": float(stored.severity),
            "support_count": int(stored.support_count),
            "active_neurons": len(stored.active_neuron_ids),
            "active_edges": len(stored.active_edge_ids),
        }
        self.growth_events.append(event)
        return event

    def _cause_conditioned_replacement_action(
        self,
        trace: StepTrace,
        selected: int,
        action_count: int,
        *,
        stress: float = 0.0,
        runner_up: int | None = None,
    ) -> int:
        """Choose a replacement action for a bad outcome without suppressing the old action.

        This is intentionally conservative and cause-conditioned: high account
        stress creates wait/manage alternatives; wrong-direction-like losses
        create opposite-market alternatives; exit/position mistakes create
        management alternatives.  Memory v8 can later provide richer neural
        replacement biases, but this prevents the M1 credit path from collapsing
        into negative action suppression.
        """
        selected = int(selected)
        action_count = max(1, int(action_count))
        action_details = dict(getattr(trace, "action_details", {}) or {})
        raw_allowed = action_details.get("context_allowed_actions", ())
        try:
            scoped_allowed = {
                int(action) for action in raw_allowed
                if 0 <= int(action) < action_count
            }
        except (TypeError, ValueError):
            scoped_allowed = set()
        has_scoped_domain = bool(scoped_allowed)

        def replacement_is_reachable(action: int) -> bool:
            return not has_scoped_domain or int(action) in scoped_allowed

        account_context = dict(getattr(trace, "account_context", {}) or {})
        floating = float(account_context.get("floating_pnl_r", 0.0) or 0.0)
        open_risk = float(account_context.get("open_risk_ratio", 0.0) or 0.0)
        drawdown = float(account_context.get("current_equity_drawdown", 0.0) or 0.0)
        high_stress = bool(stress > 0.65 or open_risk > 0.35 or drawdown > 0.08)
        # Action map: 0 HOLD, 1 BUY, 2 SELL, 3 BUY_LIMIT, 4 SELL_LIMIT,
        # 5 BUY_STOP, 6 SELL_STOP, 7 CANCEL, 8 CLOSE_WORST, 9 CLOSE_BEST,
        # 10 CLOSE_ALL, 11 MANAGE.
        if action_count > 11 and selected in {8, 9, 10, 11} and replacement_is_reachable(11):
            return 11
        if high_stress and action_count > 11 and (
            replacement_is_reachable(8) or replacement_is_reachable(11)
        ):
            if floating < -0.25 and replacement_is_reachable(8):
                return 8
            if replacement_is_reachable(11):
                return 11
        existing_position = bool(str(getattr(trace, "positions_pipe", "") or "").strip()) or bool(
            float(account_context.get("position_count_norm", 0.0) or 0.0) > 0.0
        )
        if (
            selected in ENTRY_ACTION_INDICES
            and existing_position
            and action_count > 11
            and replacement_is_reachable(11)
        ):
            # A losing add-on entry should teach the policy to manage its
            # existing exposure. This is lifecycle credit, not a runtime mask:
            # every entry action remains legal on the next decision.
            return 11
        if selected == 1 and action_count > 2 and replacement_is_reachable(2):
            return 2
        if selected == 2 and action_count > 1 and replacement_is_reachable(1):
            return 1
        if (
            selected in {3, 4, 5, 6}
            and runner_up is not None
            and int(runner_up) in ENTRY_ACTION_INDICES
            and int(runner_up) != selected
            and 0 <= int(runner_up) < action_count
            and replacement_is_reachable(int(runner_up))
        ):
            # Prefer the decision-time second-best executable entry. It is the
            # closest context-local counterfactual and may change both side and
            # execution style when direction, rather than timing alone, failed.
            return int(runner_up)
        # A losing pending entry must teach a different executable entry
        # style, not increase HOLD.  Keep replacement side-local so the
        # no-entry-blocking profile learns limit <-> stop without a veto.
        pending_replacements = {3: 5, 5: 3, 4: 6, 6: 4}
        replacement = pending_replacements.get(selected)
        if replacement is not None and replacement < action_count and replacement_is_reachable(replacement):
            return replacement
        role_local_replacements = {1: 2, 2: 1, 3: 4, 4: 3, 5: 6, 6: 5}
        role_local = role_local_replacements.get(selected)
        if role_local is not None and role_local < action_count and replacement_is_reachable(role_local):
            return role_local
        if selected == 0 and action_count > 2 and not high_stress:
            # HOLD can be wrong through missed opportunity; create a balanced
            # small directional replacement rather than making HOLD forbidden.
            preferred = 1 if float(getattr(trace, "m1_critic_value", 0.0) or 0.0) >= 0.0 else 2
            if replacement_is_reachable(preferred):
                return preferred
            scoped_entries = sorted(scoped_allowed.intersection(ENTRY_ACTION_INDICES))
            if scoped_entries:
                return int(scoped_entries[0])
        return min(action_count - 1, max(0, selected))

    def _apply_neuron_action_credit(
        self,
        trace: StepTrace,
        advantage: float,
        *,
        strength: float = 1.0,
        provenance: str = "lifecycle",
    ) -> int:
        """Assign signed action-margin credit to decision-time neurons."""
        traces = tuple(getattr(trace, "module_action_traces", ()) or ())
        if not traces:
            return 0
        action_count = min(ACTION_COUNT, max(1, int(self.output_size)))
        selected = self._trace_action_index(trace, action_count)
        alternative = self._runner_up_action(trace, selected, action_count)
        denom_gates = max(1e-9, sum(max(0.0, float(item.gate)) for item in traces))
        credited = 0
        bounded_advantage = float(np.clip(advantage, -3.0, 3.0))
        account_context = dict(getattr(trace, "account_context", {}) or {})
        entry_add_on = bool(selected in ENTRY_ACTION_INDICES) and (
            bool(str(getattr(trace, "positions_pipe", "") or "").strip())
            or float(account_context.get("position_count_norm", 0.0) or 0.0) > 0.0
        )
        stress = float(np.clip(
            float(account_context.get("current_equity_drawdown", 0.0) or 0.0)
            + 0.5 * float(account_context.get("open_risk_ratio", 0.0) or 0.0)
            + 0.5 * max(0.0, -float(account_context.get("floating_pnl_r", 0.0) or 0.0))
            + 0.25 * float(account_context.get("equity_volatility_20", 0.0) or 0.0),
            0.0,
            3.0,
        ))
        if selected in ENTRY_ACTION_INDICES and bounded_advantage < 0.0:
            bounded_advantage = float(np.clip(bounded_advantage - 0.35 * stress, -3.0, 3.0))
        for item in traces:
            module = next((candidate for candidate in self.modules if candidate.name == item.module_name), None)
            if module is None or selected >= module.w_out.shape[0] or alternative >= module.w_out.shape[0]:
                continue
            hidden = self._resize_vector(np.asarray(item.hidden, dtype=float), module.hidden_size)
            gate_weight = max(0.0, float(item.gate)) / denom_gates
            action_margin_weights = module.w_out[selected] - module.w_out[alternative]
            signed = gate_weight * hidden * action_margin_weights
            normalizer = max(1e-9, float(np.sum(np.abs(signed))))
            contribution = signed / normalizer
            activation = np.tanh(np.abs(hidden))
            redundancy = self._neuron_redundancy(module)
            frozen = np.asarray(getattr(module, "neuron_frozen", np.zeros(module.hidden_size, dtype=bool)), dtype=bool)
            if module.neuron_utility is None or module.neuron_utility.size != module.hidden_size:
                module.neuron_utility = np.zeros(module.hidden_size, dtype=float)
            for attr in ("neuron_advantage_ema", "neuron_contribution_ema", "neuron_activation_ema", "neuron_redundancy_ema", "neuron_samples"):
                value = getattr(module, attr, None)
                if value is None or np.asarray(value).size != module.hidden_size:
                    setattr(module, attr, np.zeros(module.hidden_size, dtype=float))
            active = activation > 0.05
            local_credit = bounded_advantage * contribution * activation * float(strength)
            update_mask = active & ~frozen
            module.neuron_utility = 0.94 * module.neuron_utility + 0.06 * np.where(update_mask, local_credit, 0.0)
            module.neuron_advantage_ema = 0.94 * module.neuron_advantage_ema + 0.06 * np.where(active, bounded_advantage, 0.0)
            module.neuron_contribution_ema = 0.94 * module.neuron_contribution_ema + 0.06 * np.where(active, contribution, 0.0)
            module.neuron_activation_ema = 0.94 * module.neuron_activation_ema + 0.06 * activation
            module.neuron_redundancy_ema = 0.94 * module.neuron_redundancy_ema + 0.06 * redundancy
            module.neuron_samples = module.neuron_samples + active.astype(float)
            replacement_mode = bool(bounded_advantage < 0.0)
            replacement_action = alternative
            if replacement_mode:
                replacement_action = self._cause_conditioned_replacement_action(
                    trace,
                    selected,
                    action_count,
                    stress=stress,
                    runner_up=alternative,
                )
                if replacement_action >= module.w_out.shape[0]:
                    replacement_action = alternative
            if np.any(update_mask):
                credit_multiplier = NEGATIVE_CONTEXT_CREDIT_MULTIPLIER if replacement_mode else 1.0
                lr = float(np.clip(
                    0.001 * module.plasticity * abs(bounded_advantage) * strength * credit_multiplier,
                    0.00001,
                    0.016 if replacement_mode else 0.002,
                ))
                if replacement_mode:
                    # Pairwise context-local margin update. Only the hidden
                    # path that produced this decision is changed; unrelated
                    # contexts keep their selected-action weights.
                    replacement_gain = 3.0 if entry_add_on and replacement_action == 11 else 1.0
                    step = replacement_gain * lr * abs(bounded_advantage) * gate_weight * hidden[update_mask]
                    module.w_out[selected, update_mask] -= step
                    if replacement_action != selected:
                        module.w_out[replacement_action, update_mask] += step
                else:
                    module.w_out[selected, update_mask] += lr * bounded_advantage * gate_weight * hidden[update_mask]
                    module.w_out[alternative, update_mask] -= lr * bounded_advantage * gate_weight * hidden[update_mask]
                credited += int(np.count_nonzero(update_mask))
            module.credit_ema = 0.95 * module.credit_ema + 0.05 * (abs(bounded_advantage) if replacement_mode else bounded_advantage)
            self.growth_events.append({
                "round": self.revision,
                "type": "neuron_action_replacement_credit" if replacement_mode else "neuron_action_credit",
                "module": module.name,
                "action": int(selected),
                "alternative": int(alternative),
                "replacement_action": int(replacement_action),
                "negative_credit_multiplier": float(
                    NEGATIVE_CONTEXT_CREDIT_MULTIPLIER if replacement_mode else 1.0
                ),
                "replacement_gain": float(3.0 if entry_add_on and replacement_action == 11 else 1.0),
                "advantage": float(bounded_advantage),
                "negative_action_suppression_disabled": bool(replacement_mode),
                "account_stress": float(stress),
                "credited_neurons": int(np.count_nonzero(update_mask)),
                "provenance": provenance,
            })
        return credited

    def apply_execution_lifecycle_credit(
        self,
        trace_id: str,
        realized_r: float,
        *,
        close_reason: str = "",
        duration_bars: float | None = None,
        min_holding_bars: int = 3,
    ) -> int:
        """Backfill final position outcome to the entry trace recorded by MT5."""
        cache = getattr(self, "_execution_trace_cache", None)
        if not isinstance(cache, dict):
            return 0
        trace = cache.pop(str(trace_id), None)
        if not isinstance(trace, StepTrace):
            return 0
        baseline = float(getattr(self, "_lifecycle_outcome_ema", 0.0))
        outcome = float(np.clip(realized_r, -3.0, 3.0))
        advantage = outcome - baseline
        if duration_bars is not None and float(duration_bars) < max(1, int(min_holding_bars)):
            advantage -= 0.25
        if str(close_reason).upper() in {"STOP_OUT"}:
            advantage -= 0.25
        self._lifecycle_outcome_ema = 0.90 * baseline + 0.10 * outcome
        # Lifecycle settlement is the only reliable realised outcome for an
        # entry. Route it back to the exact graph decision trace, including
        # weak candidate specialist edges; do not rely on an intermediate
        # placement/acceptance acknowledgement.
        graph_edges = tuple(str(item) for item in getattr(trace, "active_edge_ids", ()) or ())
        graph_nodes = tuple(str(item) for item in getattr(trace, "active_neuron_ids", ()) or ())
        graph_credited = 0
        if graph_edges or graph_nodes:
            lifecycle_tags: list[str] = []
            reason = str(close_reason).upper()
            if outcome < 0.0 and reason in {"STOP_OUT", "SL", "STOP_LOSS"}:
                lifecycle_tags.append("stress_entry_loss")
            if duration_bars is not None and float(duration_bars) < max(1, int(min_holding_bars)):
                lifecycle_tags.append("closed_after_near_tp_bad")
            for module, m1 in self._graph_m1_modules():
                module_edges = tuple(edge_id for edge_id in graph_edges if edge_id in m1.graph.edges)
                module_nodes = tuple(node_id for node_id in graph_nodes if node_id in m1.graph.neurons)
                if not module_edges and not module_nodes:
                    continue
                report = GraphPlasticityEngine().apply_credit(
                    m1.graph,
                    GraphCreditSignal(
                        # M1 v10: bad outcomes must create replacement pressure,
                        # not negative suppression of the active action path.
                        reward=max(0.0, outcome),
                        td_delta=max(0.0, advantage),
                        lifecycle_tags=tuple(lifecycle_tags),
                        account_stress=float(np.clip(dict(getattr(trace, "account_context", {}) or {}).get("current_equity_drawdown", 0.0), 0.0, 1.0)),
                    ),
                    active_edge_ids=module_edges,
                    active_neuron_ids=module_nodes,
                    # Candidate edges are detected individually by the engine;
                    # baseline core edges only record negative attribution.
                    candidate_mode=False,
                    update_mode="online",
                )
                graph_credited += int(report.updated_edges + report.updated_neurons)
                self.growth_events.append({
                    "round": self.revision,
                    "type": "graph_final_lifecycle_credit",
                    "trace_id": trace.trace_id,
                    "close_reason": str(close_reason),
                    "realized_r": outcome,
                    "advantage": advantage,
                        "updated_edges": report.updated_edges,
                        "updated_neurons": report.updated_neurons,
                        "protected_baseline_edges": report.protected_baseline_edges,
                        "gradient_abs_sum": report.gradient_abs_sum,
                        "hebbian_abs_sum": report.hebbian_abs_sum,
                        "anti_hebbian_abs_sum": report.anti_hebbian_abs_sum,
                        "oja_drive_abs_sum": report.oja_drive_abs_sum,
                        "oja_stability_abs_sum": report.oja_stability_abs_sum,
                        "update_abs_sum": report.update_abs_sum,
                    })
        if graph_credited:
            return graph_credited
        return self._apply_neuron_action_credit(trace, advantage, strength=1.0, provenance="final_lifecycle")

    def freeze_high_utility_neurons(self, min_samples: int = 30, lcb_threshold: float = 0.05) -> int:
        frozen_count = 0
        for module in self.modules:
            samples = np.asarray(module.neuron_samples, dtype=float)
            utility = np.asarray(module.neuron_utility, dtype=float)
            variability = np.maximum(0.05, np.abs(np.asarray(module.neuron_advantage_ema, dtype=float)))
            lcb = utility - 1.64 * variability / np.sqrt(np.maximum(samples, 1.0))
            mask = (samples >= float(min_samples)) & (lcb >= float(lcb_threshold)) & ~np.asarray(module.neuron_frozen, dtype=bool)
            if np.any(mask):
                module.neuron_frozen[mask] = True
                frozen_count += int(np.count_nonzero(mask))
        if frozen_count:
            self.growth_events.append({"round": self.revision, "type": "freeze_high_utility_neurons", "count": frozen_count})
        return frozen_count

    def prune_low_utility_neurons(
        self,
        min_samples: int = 30,
        lcb_threshold: float = -0.05,
        redundancy_threshold: float = 0.95,
        max_per_module: int = 1,
    ) -> int:
        """Functionally prune proven weak/redundant neurons without reindexing RNNs."""
        pruned = 0
        for module in self.modules:
            if module.hidden_size <= 2:
                continue
            samples = np.asarray(module.neuron_samples, dtype=float)
            utility = np.asarray(module.neuron_utility, dtype=float)
            variability = np.maximum(0.05, np.abs(np.asarray(module.neuron_advantage_ema, dtype=float)))
            lcb = utility - 1.64 * variability / np.sqrt(np.maximum(samples, 1.0))
            redundancy = np.asarray(module.neuron_redundancy_ema, dtype=float)
            activation = np.asarray(module.neuron_activation_ema, dtype=float)
            frozen = np.asarray(module.neuron_frozen, dtype=bool)
            candidates = np.where(
                (samples >= float(min_samples)) & ~frozen & (lcb < float(lcb_threshold))
                & ((activation < 0.03) | (redundancy >= float(redundancy_threshold)))
            )[0]
            if candidates.size <= 0:
                continue
            candidates = candidates[np.argsort(lcb[candidates])[:max(1, int(max_per_module))]]
            for idx in candidates:
                module.w_out[:, int(idx)] = 0.0
                module.w_rec[int(idx), :] *= 0.10
                module.w_rec[:, int(idx)] *= 0.10
                pruned += 1
                self.growth_events.append({
                    "round": self.revision,
                    "type": "prune_neuron", "module": module.name, "neuron": int(idx),
                    "utility_lcb": float(lcb[int(idx)]), "redundancy": float(redundancy[int(idx)]),
                })
        return pruned

    def _apply_lateral_fitness_credit(self, credits: dict[str, float]) -> None:
        self._ensure_lateral_edges()
        for target in self.modules:
            target_hidden = self._resize_vector(self._runtime_memory.get(target.name, np.zeros(target.hidden_size)), target.hidden_size)
            for source in self.modules:
                if source.name == target.name:
                    continue
                source_hidden = self._resize_vector(self._runtime_memory.get(source.name, np.zeros(source.hidden_size)), source.hidden_size)
                key = self._lateral_key(target.name, source.name)
                credit = 0.5 * (credits.get(target.name, 0.0) + credits.get(source.name, 0.0))
                self.lateral_edges[key] += 0.0005 * credit * np.outer(target_hidden, source_hidden)
                self.lateral_edges[key] = np.clip(self.lateral_edges[key], -4.0, 4.0)

    def ensure_compatible(self, target_input_size: int | None = None) -> None:
        if not hasattr(self, "m1_components"):
            self.m1_components = M1_COMPONENTS
        if not hasattr(self, "decision_backend"):
            self.decision_backend = "direct_rnn"
        self.decision_backend = _normalize_m1_backend(getattr(self, "decision_backend", "direct_rnn"))
        self.m1_backend = self.decision_backend
        if not hasattr(self, "artifact_version"):
            self.artifact_version = 1
        self.artifact_version = max(9, int(self.artifact_version))
        if not hasattr(self, "_runtime_memory"):
            self._runtime_memory = {}
        if not hasattr(self, "gating_mode"):
            self.gating_mode = "sigmoid"
        if not hasattr(self, "growth_controller"):
            self.growth_controller = GrowthController()
        self.__dict__.pop("dev_controller", None)
        if not hasattr(self, "plasticity_coeffs"):
            self.plasticity_coeffs = PlasticityCoefficients()
        if not hasattr(self, "loss_history"):
            self.loss_history = []
        if not hasattr(self, "active_shadows"):
            self.active_shadows = []
        if not hasattr(self, "experience_buffer"):
            self.experience_buffer = []
        # V7 bounds every retained evidence class.  Old artifacts may contain
        # ``None`` from the open-ended era; normalize them to an explicit cap
        # before any candidate copies/replays the state.
        self.experience_buffer_max = max(1, int(getattr(self, "experience_buffer_max", 4096) or 4096))
        if not hasattr(self, "rejection_memory"):
            self.rejection_memory = []
        self.rejection_memory_max = max(1, int(getattr(self, "rejection_memory_max", 2048) or 2048))
        if len(self.experience_buffer) > self.experience_buffer_max:
            self.experience_buffer = list(self.experience_buffer[-self.experience_buffer_max:])
        if len(self.rejection_memory) > self.rejection_memory_max:
            self.rejection_memory = list(self.rejection_memory[-self.rejection_memory_max:])
        if not hasattr(self, "memory_supernet") or self.memory_supernet is None:
            self.memory_supernet = ExperienceSuperNetwork()
        self.memory_supernet.ensure_compatible()
        if not hasattr(self, "mistake_episode_store") or self.mistake_episode_store is None:
            self.mistake_episode_store = MistakeEpisodeStore()
        self.mistake_episode_store.ensure_compatible()
        if not hasattr(self, "curiosity_system") or self.curiosity_system is None:
            self.curiosity_system = CuriosityNeuronSystem.seed()
        self.curiosity_system.ensure_compatible()
        if not hasattr(self, "neural_development") or self.neural_development is None:
            self.neural_development = NeuralDevelopmentController.seed()
        self.neural_development.ensure_compatible()
        if not hasattr(self, "neural_utility") or self.neural_utility is None:
            self.neural_utility = NeuralUtilityNetwork.seed()
        self.neural_utility.ensure_compatible()
        if not hasattr(self, "risk_cost_gate") or self.risk_cost_gate is None:
            self.risk_cost_gate = NeuralRiskCostGate.seed()
        self.risk_cost_gate.ensure_compatible()
        if not hasattr(self, "action_parameter_head") or self.action_parameter_head is None:
            self.action_parameter_head = NeuralActionParameterHead.seed()
        self.action_parameter_head.ensure_compatible()
        if not hasattr(self, "reasoning_graph") or self.reasoning_graph is None:
            self.reasoning_graph = ReasoningGraphSystem.seed()
        self.reasoning_graph.ensure_compatible()
        if not hasattr(self, "_step_count"):
            self._step_count = 0
        if not hasattr(self, "_execution_trace_cache"):
            self._execution_trace_cache = {}
        self._execution_trace_cache_max = max(1, int(getattr(self, "_execution_trace_cache_max", 8192) or 8192))
        if not hasattr(self, "_lifecycle_outcome_ema"):
            self._lifecycle_outcome_ema = 0.0
        if not hasattr(self, "_reward_modulator"):
            self._reward_modulator = 0.0
        if not hasattr(self, "_experience_starvation"):
            self._experience_starvation = {}
        if not hasattr(self, "_outcome_starvation"):
            self._outcome_starvation = {}
        if not hasattr(self, "graph_supernetwork"):
            self.graph_supernetwork = None
        if not hasattr(self, "graph_supernetwork_config"):
            self.graph_supernetwork_config = {}
        if not hasattr(self, "graph_trial_id"):
            self.graph_trial_id = ""
        if not hasattr(self, "graph_trial_mode"):
            self.graph_trial_mode = False
        if not hasattr(self, "child_supernetwork_manager"):
            self.child_supernetwork_manager = None
        if self._uses_m1_graph() and self.graph_supernetwork is None:
            self.graph_supernetwork = GraphSupernetworkRuntime()
        if self._uses_m1_graph() and self.graph_supernetwork is not None and not hasattr(self.graph_supernetwork, "trial_growth"):
            from .trial_micro_growth import TrialMicroGrowthController
            self.graph_supernetwork.trial_growth = TrialMicroGrowthController()
        if not hasattr(self, "_account_context"):
            self._account_context = {}
        if not hasattr(self, "_positions_pipe_context"):
            self._positions_pipe_context = ""
        for module in self.modules:
            if not hasattr(module, "role"):
                module.role = "m1_thinking_core" if module.name == "m1_thinking" else "specialist"
            if not hasattr(module, "w_rec"):
                module.w_rec = np.zeros((module.hidden_size, module.hidden_size), dtype=float)
            if not hasattr(module, "w_lat") or module.w_lat is None:
                module.w_lat = np.zeros((module.hidden_size, module.hidden_size), dtype=float)
            if not hasattr(module, "deep_w") or module.deep_w is None:
                module.deep_w = ()
            if not hasattr(module, "deep_rec") or module.deep_rec is None:
                module.deep_rec = tuple(np.zeros((module.hidden_size, module.hidden_size), dtype=float) for _ in module.deep_w)
            if not hasattr(module, "deep_b") or module.deep_b is None:
                module.deep_b = tuple(np.zeros(module.hidden_size, dtype=float) for _ in module.deep_w)
            if not hasattr(module, "deep_state") or module.deep_state is None:
                module.deep_state = tuple(np.zeros(module.hidden_size, dtype=float) for _ in module.deep_w)
            if not hasattr(module, "input_mask") or module.input_mask is None or module.input_mask.size != module.input_size:
                module.input_mask = np.ones(module.input_size, dtype=float)
            if not hasattr(module, "output_mask") or module.output_mask is None or module.output_mask.size != module.output_size:
                module.output_mask = np.ones(module.output_size, dtype=float)
            if not hasattr(module, "neuron_utility") or module.neuron_utility is None or module.neuron_utility.size != module.hidden_size:
                module.neuron_utility = np.zeros(module.hidden_size, dtype=float)
            for attr, dtype in (
                ("neuron_advantage_ema", float),
                ("neuron_contribution_ema", float),
                ("neuron_activation_ema", float),
                ("neuron_redundancy_ema", float),
                ("neuron_samples", float),
                ("neuron_frozen", bool),
            ):
                value = getattr(module, attr, None)
                if value is None or np.asarray(value).size != module.hidden_size:
                    setattr(module, attr, np.zeros(module.hidden_size, dtype=dtype))
            if not hasattr(module, "input_eligibility") or module.input_eligibility is None or module.input_eligibility.shape != (module.hidden_size, module.input_size):
                module.input_eligibility = np.zeros((module.hidden_size, module.input_size), dtype=float)
            if not hasattr(module, "credit_ema"):
                module.credit_ema = 0.0
            if not hasattr(module, "utility"):
                module.utility = 0.0
            if not hasattr(module, "gate_ema"):
                module.gate_ema = 0.0
            if not hasattr(module, "activation_ema"):
                module.activation_ema = 0.0
            if not hasattr(module, "lifecycle"):
                module.lifecycle = MODULE_LIFECYCLE_ACTIVE if module.role == "m1_thinking_core" else MODULE_LIFECYCLE_DORMANT
            if not hasattr(module, "memory_decay"):
                module.memory_decay = 0.85
            legacy_alpha_gate = module.__dict__.pop("alpha_gate", None)
            if legacy_alpha_gate is not None:
                legacy_alpha = np.asarray(legacy_alpha_gate, dtype=float)
                if legacy_alpha.shape == module.w_in.shape:
                    module.w_in *= 1.0 / (1.0 + np.exp(-np.clip(legacy_alpha, -10.0, 10.0)))
            if not hasattr(module, "p_stp"):
                module.p_stp = np.ones((module.hidden_size, module.input_size), dtype=float)
            if not hasattr(module, "stp_rho"):
                module.stp_rho = 0.95
            if not hasattr(module, "stp_alpha"):
                module.stp_alpha = 0.1
            if not hasattr(module, "hidden_normalization_target_rms"):
                module.hidden_normalization_target_rms = 1.0
            if not hasattr(module, "hidden_normalization_clip"):
                module.hidden_normalization_clip = 3.0
            if not hasattr(module, "hidden_normalization_eps"):
                module.hidden_normalization_eps = 1e-5
            hidden_layer_count = 1 + len(getattr(module, "deep_w", ()) or ())
            module.hidden_normalization_mode = _mandatory_hidden_normalization_mode(
                getattr(module, "hidden_normalization_mode", REQUIRED_HIDDEN_NORMALIZATION_MODE)
            )
            module.hidden_normalization_enabled = True
            module.hidden_normalization_layers = tuple(True for _ in range(hidden_layer_count))
            module.hidden_dropout_enabled = True
            module.hidden_dropout_rates = mandatory_layerwise_rates(
                getattr(module, "hidden_dropout_rates", REQUIRED_HIDDEN_DROPOUT_RATES),
                hidden_layer_count,
            )
            if not hasattr(module, "hidden_dropout_seed"):
                module.hidden_dropout_seed = 0
            module.hidden_dropout_seed = max(0, int(getattr(module, "hidden_dropout_seed", 0) or 0))
            module.hidden_dropout_train_only = True
            if not hasattr(module, "hidden_dropout_step"):
                module.hidden_dropout_step = 0
            module.hidden_dropout_step = max(0, int(getattr(module, "hidden_dropout_step", 0) or 0))
            if not hasattr(module, "plasticity_input_normalization_mode"):
                module.plasticity_input_normalization_mode = "raw"
            if str(module.plasticity_input_normalization_mode).strip().lower() not in {"raw", "tanh_v1", "soft_clip_v1", "groupwise_v2", "groupwise_mtf_v3", "structure_aware_v4"}:
                module.plasticity_input_normalization_mode = "raw"
            if not hasattr(module, "plasticity_update_clip"):
                module.plasticity_update_clip = 0.25
            module.plasticity_update_clip = float(np.clip(getattr(module, "plasticity_update_clip", 0.25) or 0.25, 0.0, 10.0))
            if not hasattr(module, "plasticity_weight_norm_cap"):
                module.plasticity_weight_norm_cap = max(1.0, float(np.sqrt(module.hidden_size) * 1.5))
            module.plasticity_weight_norm_cap = float(np.clip(
                getattr(module, "plasticity_weight_norm_cap", 0.0) or max(1.0, float(np.sqrt(module.hidden_size) * 1.5)),
                1.0,
                1e4,
            ))
            if not hasattr(module, "plasticity_update_count"):
                module.plasticity_update_count = 0
            if not hasattr(module, "plasticity_update_ema"):
                module.plasticity_update_ema = 0.0
            if not hasattr(module, "plasticity_update_var"):
                module.plasticity_update_var = 0.0
            if not hasattr(module, "plasticity_last_stats") or not isinstance(module.plasticity_last_stats, dict):
                module.plasticity_last_stats = {}
            if not hasattr(module, "stdp_params"):
                module.stdp_params = STDPParams()
            if not hasattr(module, "plasticity_coeffs"):
                module.plasticity_coeffs = PlasticityCoefficients()
            if not hasattr(module, "last_spike_time"):
                module.last_spike_time = 0.0
            if not hasattr(module, "spike_history"):
                module.spike_history = []
            if self._uses_m1_graph() and not isinstance(getattr(module, "m1_component", None), M1GraphModule):
                module.m1_component = M1GraphModule.seed(
                    input_size=module.input_size,
                    output_size=module.output_size,
                    state_size=module.hidden_size,
                    K=4,
                    seed=(self.revision + len(module.name)) & 0x7FFFFFFF,
                )
                self.growth_events.append({
                    "round": self.revision,
                    "type": "graph_core_reinitialized",
                    "module": module.name,
                    "reason": "graph_only_compatibility",
                })
        self._ensure_lateral_edges()
        if target_input_size is not None and int(target_input_size) > 0 and (
            self.input_size != int(target_input_size)
            or any(module.input_size != int(target_input_size) or module.w_in.shape[1] != int(target_input_size) for module in self.modules)
        ):
            old_input_size = int(self.input_size)
            for module in self.modules:
                self._resize_module_input(module, int(target_input_size))
            self.input_size = int(target_input_size)
            self.growth_events.append({
                "round": self.revision,
                "type": "resize_input",
                "old_input": old_input_size,
                "new_input": int(target_input_size),
            })
        target_output_size = OUTPUT_COUNT if self.input_size >= INPUT_COUNT else int(self.output_size)
        if target_output_size > 0 and (
            self.output_size != target_output_size
            or any(module.output_size != target_output_size or module.w_out.shape[0] != target_output_size for module in self.modules)
        ):
            old_output_size = int(self.output_size)
            for module in self.modules:
                self._resize_module_output(module, target_output_size)
            self.output_size = target_output_size
            self.growth_events.append({
                "round": self.revision,
                "type": "resize_output",
                "old_output": old_output_size,
                "new_output": int(target_output_size),
            })
        # Make graph dimensions follow bridge schema changes without a predecessor runtime
        # fallback. New input/output neurons start inactive until evidence can
        # recruit/use them through normal graph development.
        for module in self.modules:
            m1 = module.m1_component
            if self._uses_m1_graph() and isinstance(m1, M1GraphModule):
                m1.resize_input(module.input_size)
                m1.resize_output(module.output_size)
        self.m1_backend = self.decision_backend

    def _uses_m1_graph(self) -> bool:
        return _normalize_m1_backend(getattr(self, "decision_backend", getattr(self, "m1_backend", "direct_rnn"))) == "m1_graph"

    def set_experience_starvation(self, starvation: dict[str, float] | None) -> None:
        runtime = {str(k): float(v) for k, v in (starvation or {}).items()}
        outcome = {str(k): float(v) for k, v in getattr(self, "_outcome_starvation", {}).items()}
        combined = dict(runtime)
        combined.update(outcome)
        previous_outcome_starvation = float(runtime.get("outcome_starvation", 0.0))
        runtime_starvation = max(0.0, float(runtime.get("starvation", 0.0)) - previous_outcome_starvation)
        outcome_starvation = float(outcome.get("outcome_starvation", 0.0))
        runtime_intrinsic = max(0.0, float(runtime.get("intrinsic_reward", 0.0)) - 0.02 * previous_outcome_starvation)
        # Keep *runtime* entry scarcity distinct from outcome-level failure.
        # Runtime scarcity is allowed to request a bounded neural exploration
        # probe on the next bridge ticks. Outcome scarcity instead belongs to
        # structural proposal/replay selection; feeding it directly into the
        # live action logits made a rejected historical trial keep perturbing
        # every future decision and contributed to HOLD/CLOSE collapse.
        combined["runtime_starvation"] = float(np.clip(runtime_starvation, 0.0, 3.0))
        combined["outcome_starvation"] = float(np.clip(outcome_starvation, 0.0, 3.0))
        combined["starvation"] = float(np.clip(runtime_starvation + outcome_starvation, 0.0, 5.0))
        combined["intrinsic_reward"] = float(np.clip(
            runtime_intrinsic + 0.02 * outcome_starvation,
            0.0,
            0.12,
        ))
        self._experience_starvation = combined

    def set_account_context(self, account_context: dict[str, float] | None, positions_pipe: str | None = "") -> None:
        self._account_context = {str(k): float(v) for k, v in (account_context or {}).items()}
        self._positions_pipe_context = str(positions_pipe or "")

    def _query_memory(self, state_embedding: np.ndarray) -> dict[str, object]:
        if not hasattr(self, "memory_supernet") or self.memory_supernet is None:
            return {}
        return self.memory_supernet.query(
            state_embedding,
            self.output_size,
            len(self.modules),
            account_context=getattr(self, "_account_context", {}),
            positions_pipe=getattr(self, "_positions_pipe_context", ""),
        )

    def apply_outcome_starvation(
        self,
        *,
        goal_hit: bool | None = None,
        improved: bool | None = None,
        fitness_delta: float = 0.0,
        reason: str = "",
    ) -> dict[str, float]:
        self.ensure_compatible()
        current = dict(getattr(self, "_outcome_starvation", {}) or {})
        failed_goal_streak = float(current.get("failed_goal_streak", 0.0))
        non_improve_streak = float(current.get("non_improve_streak", 0.0))

        if goal_hit is True:
            failed_goal_streak = max(0.0, failed_goal_streak - 2.0)
        elif goal_hit is False:
            failed_goal_streak += 1.0

        if improved is True:
            non_improve_streak = max(0.0, non_improve_streak - 1.0)
        elif improved is False:
            non_improve_streak += 1.0

        regression = max(0.0, -float(fitness_delta))
        goal_deficit = 0.25 * failed_goal_streak
        trial_regression = 0.20 * non_improve_streak + 0.05 * min(regression, 10.0)
        outcome_starvation = float(np.clip(goal_deficit + trial_regression, 0.0, 3.0))
        self._outcome_starvation = {
            "failed_goal_streak": failed_goal_streak,
            "non_improve_streak": non_improve_streak,
            "goal_deficit": float(goal_deficit),
            "trial_regression": float(trial_regression),
            "outcome_starvation": outcome_starvation,
            "last_fitness_delta": float(fitness_delta),
            "last_outcome_reason": 0.0,
        }
        self.growth_events.append({
            "round": self.revision,
            "type": "outcome_starvation",
            "goal_hit": goal_hit,
            "improved": improved,
            "fitness_delta": float(fitness_delta),
            "reason": reason,
            "outcome_starvation": outcome_starvation,
            "failed_goal_streak": failed_goal_streak,
            "non_improve_streak": non_improve_streak,
        })
        self.set_experience_starvation(self._experience_starvation)
        return dict(self._outcome_starvation)

    @staticmethod
    def _bad_rejection_traces(policy: "SelfGrowingPolicy", *, limit: int = 96) -> list[tuple[StepTrace, float]]:
        traces: list[tuple[StepTrace, float]] = []
        for item in getattr(policy, "experience_buffer", []) or []:
            if not isinstance(item, dict):
                continue
            trace = item.get("trace")
            if not isinstance(trace, StepTrace):
                continue
            reward = float(item.get("reward", 0.0) or 0.0)
            td_delta = float(item.get("td_delta", getattr(trace, "td_delta", 0.0)) or 0.0)
            score = min(reward, td_delta, float(getattr(trace, "td_delta", 0.0) or 0.0))
            if score <= 0.0:
                traces.append((trace, score))
        traces.sort(key=lambda pair: pair[1])
        return traces[:max(1, int(limit))]

    def _apply_rejection_attribution_to_active_paths(
        self,
        rejected_policy: "SelfGrowingPolicy",
        *,
        severity: float,
        no_trade: bool,
        loss_bad: bool,
        risk_bad: bool,
    ) -> dict[str, object]:
        traces = self._bad_rejection_traces(rejected_policy)
        if not traces:
            return {"bad_traces": 0, "graph_edges": 0, "graph_neurons": 0, "module_links": 0}

        module_by_name = {module.name: module for module in self.modules}
        graph_edges_updated = 0
        graph_neurons_updated = 0
        module_links_updated = 0
        penalty = float(np.clip(0.04 * severity, 0.03, 0.18))
        entry_relief = 0.5 if no_trade and not (loss_bad or risk_bad) else 1.0
        for trace, score in traces:
            action_idx = self._trace_action_index(trace, self.output_size)
            trace_scale = float(np.clip(1.0 + abs(score), 1.0, 3.0))
            # Legacy/recurrent module path: mutate the hidden->selected-action
            # links that were active when the bad action was selected.
            for module_trace in getattr(trace, "module_action_traces", ()) or ():
                module = module_by_name.get(module_trace.module_name)
                if module is None or action_idx >= module.w_out.shape[0]:
                    continue
                hidden = np.asarray(module_trace.hidden, dtype=float).reshape(-1)
                width = min(hidden.size, module.w_out.shape[1])
                if width <= 0:
                    continue
                active = np.tanh(hidden[:width])
                # V10 replacement-only learning: rejected/bad traces must not
                # reduce the old action path.  No-trade still strengthens entry;
                # loss/risk cases are handled by positive replacement/management
                # pathways below, not by suppressing action_idx.
                if no_trade and action_idx in ENTRY_ACTION_INDICES:
                    local = penalty * trace_scale * entry_relief * active
                    module.w_out[action_idx, :width] = np.clip(module.w_out[action_idx, :width] + local, -8.0, 8.0)
                module_links_updated += int(np.count_nonzero(np.abs(active) > 1e-6))

            active_edges = tuple(dict.fromkeys(str(item) for item in getattr(trace, "active_edge_ids", ()) or ()))
            active_nodes = tuple(dict.fromkeys(str(item) for item in getattr(trace, "active_neuron_ids", ()) or ()))
            if not active_edges and not active_nodes:
                continue
            # Graph M1 path: edge/node ids are stable for baseline graph
            # portions. Candidate-only ids simply do not exist on baseline and
            # are ignored, preserving rollback.
            for _module, m1 in self._graph_m1_modules():
                graph = m1.graph
                for edge_id in active_edges:
                    edge = graph.edges.get(edge_id)
                    if edge is None:
                        continue
                    # Do not suppress active edges on bad/rejected outcomes.
                    # Keep eligibility/plasticity so replacement neurons can use
                    # the trace as context for alternative action synthesis.
                    edge.eligibility = float(np.clip(edge.eligibility + 0.10 * trace_scale, 0.0, 100.0))
                    edge.plasticity = float(np.clip(edge.plasticity + 0.04 * severity, 0.05, 3.0))
                    graph_edges_updated += 1
                for node_id in active_nodes:
                    node = graph.neurons.get(node_id)
                    if node is None:
                        continue
                    # Do not reduce active/output node bias on bad outcomes.
                    node.eligibility = float(np.clip(node.eligibility + abs(node.activation), 0.0, 100.0))
                    node.plasticity = float(np.clip(node.plasticity + 0.03 * severity, 0.05, 3.0))
                    graph_neurons_updated += 1
                graph.assert_invariants(output_size=self.output_size)
        return {
            "bad_traces": len(traces),
            "graph_edges": graph_edges_updated,
            "graph_neurons": graph_neurons_updated,
            "module_links": module_links_updated,
        }

    def absorb_rejection_feedback(
        self,
        rejected_policy: "SelfGrowingPolicy",
        *,
        reject_reasons: list[str] | tuple[str, ...] = (),
        metrics: dict[str, object] | None = None,
        fitness_delta: float = 0.0,
    ) -> dict[str, object]:
        """Keep strong lessons from a rejected trial without copying its topology."""
        self.ensure_compatible(target_input_size=self.input_size)
        metrics = dict(metrics or {})
        reasons = [str(reason) for reason in (reject_reasons or ())]
        total_trades = float(metrics.get("total_trades", metrics.get("trades", 0.0)) or 0.0)
        net_profit = float(metrics.get("net_profit", 0.0) or 0.0)
        profit_factor = float(metrics.get("profit_factor", 0.0) or 0.0)
        drawdown = max(
            float(metrics.get("balance_drawdown_pct", 0.0) or 0.0),
            float(metrics.get("equity_drawdown_pct", 0.0) or 0.0),
        )
        reason_blob = " ".join(reasons).lower()
        no_trade = total_trades <= 0.0 or "no_trade" in reason_blob or "insufficient" in reason_blob
        execution_bad = "execution" in reason_blob or "reconciliation" in reason_blob
        loss_bad = net_profit <= 0.0 or profit_factor < 1.0 or "net_profit" in reason_blob or "profit_factor" in reason_blob
        risk_bad = drawdown > 0.0 or "drawdown" in reason_blob or execution_bad
        severity = float(np.clip(
            0.75
            + 0.20 * len(reasons)
            + 0.015 * max(0.0, -float(fitness_delta))
            + (1.25 if no_trade else 0.0)
            + (0.75 if loss_bad else 0.0)
            + (0.50 if risk_bad else 0.0),
            0.75,
            4.0,
        ))
        lesson = {
            "round": int(getattr(self, "revision", 0)),
            "reasons": reasons,
            "fitness_delta": float(fitness_delta),
            "severity": severity,
            "total_trades": total_trades,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "drawdown_pct": drawdown,
            "no_trade": bool(no_trade),
            "loss_bad": bool(loss_bad),
            "risk_bad": bool(risk_bad),
            "candidate_backend": str(getattr(rejected_policy, "m1_backend", "")),
        }
        self.rejection_memory.append(lesson)
        rejection_cap = getattr(self, "rejection_memory_max", None)
        if rejection_cap is not None and int(rejection_cap) > 0:
            self.rejection_memory = self.rejection_memory[-int(rejection_cap):]
        rejection_episode = episode_from_rejection(
            reject_reasons=reasons,
            metrics=metrics,
            fitness_delta=float(fitness_delta),
            source="rejection",
        )
        stored_rejection_episode = self.mistake_episode_store.add(rejection_episode)
        lesson["mistake_episode_id"] = str(stored_rejection_episode.episode_id)
        lesson["mistake_tags"] = list(stored_rejection_episode.mistake_tags)

        starvation = dict(getattr(self, "_outcome_starvation", {}) or {})
        starvation["outcome_starvation"] = float(np.clip(float(starvation.get("outcome_starvation", 0.0)) + 0.35 * severity, 0.0, 5.0))
        if no_trade:
            starvation["no_trade_streak"] = float(np.clip(float(starvation.get("no_trade_streak", 0.0)) + 3.0 * severity, 0.0, 200.0))
            starvation["cadence_deficit"] = float(np.clip(float(starvation.get("cadence_deficit", 0.0)) + 0.35 * severity, 0.0, 5.0))
        if risk_bad:
            starvation["trial_regression"] = float(np.clip(float(starvation.get("trial_regression", 0.0)) + 0.25 * severity, 0.0, 5.0))
        self._outcome_starvation = starvation
        self.set_experience_starvation(getattr(self, "_experience_starvation", {}))
        attribution = self._apply_rejection_attribution_to_active_paths(
            rejected_policy,
            severity=severity,
            no_trade=no_trade,
            loss_bad=loss_bad,
            risk_bad=risk_bad,
        )

        # V10 replacement-only learning: do not lower HOLD/ENTRY biases as a
        # penalty.  Bad outcomes strengthen replacement/management/patience
        # channels; no-trade strengthens entry channels.
        hold_delta = 0.0 if no_trade else 0.08 * severity
        entry_delta = 0.16 * severity if no_trade else 0.0
        manage_delta = 0.18 * severity if (loss_bad or risk_bad) else 0.06 * severity
        close_indices = [idx for idx in (8, 9, 10, 11) if idx < self.output_size]
        parameter_indices = [idx for idx in (PARAMETER_OFFSET + 3, PARAMETER_OFFSET + 4, PARAMETER_OFFSET + 5, PARAMETER_OFFSET + 6) if idx < self.output_size]
        for module in self.modules:
            if module.b_out.size:
                module.b_out[0] += hold_delta
                for idx in ENTRY_ACTION_INDICES:
                    if idx < module.b_out.size:
                        module.b_out[idx] += entry_delta
                for idx in close_indices + parameter_indices:
                    if idx < module.b_out.size:
                        module.b_out[idx] += manage_delta
                module.b_out = np.clip(module.b_out, -8.0, 8.0)
            module.utility = float(np.clip(0.85 * float(getattr(module, "utility", 0.0)) - 0.15 * severity, -5.0, 5.0))
            module.plasticity = float(np.clip(float(getattr(module, "plasticity", 1.0)) + 0.08 * severity, 0.05, 3.0))
            if isinstance(getattr(module, "m1_component", None), M1GraphModule):
                graph = module.m1_component.graph
                for node in graph.neurons.values():
                    if node.region != "output":
                        continue
                    try:
                        idx = int(str(node.id).rsplit(":", 1)[-1])
                    except ValueError:
                        continue
                    if idx == 0:
                        node.bias = float(np.clip(node.bias + hold_delta, -8.0, 8.0))
                    elif idx in ENTRY_ACTION_INDICES:
                        node.bias = float(np.clip(node.bias + entry_delta, -8.0, 8.0))
                    elif idx in close_indices + parameter_indices:
                        node.bias = float(np.clip(node.bias + manage_delta, -8.0, 8.0))
                graph.assert_invariants(output_size=self.output_size)

        self.risk_cost_gate.ensure_compatible()
        if no_trade:
            # Encourage exploration/entry replacement without suppressing HOLD.
            self.risk_cost_gate.b_out[:3] += np.array([0.0, 0.0, 0.0], dtype=float) * severity
            self.curiosity_system.b_out[:min(7, self.curiosity_system.b_out.size)] += 0.10 * severity
        else:
            # Bad trades increase risk awareness and replacement search, not
            # negative action suppression.
            self.risk_cost_gate.b_out[:3] += np.array([0.30, 0.25, 0.20], dtype=float) * severity
            self.curiosity_system.b_out[:min(7, self.curiosity_system.b_out.size)] += 0.04 * severity
        self.risk_cost_gate.b_out = np.clip(self.risk_cost_gate.b_out, -8.0, 8.0)
        self.curiosity_system.b_out = np.clip(self.curiosity_system.b_out, -8.0, 8.0)
        self.neural_development.ensure_compatible()
        self.neural_development.b_out[0] -= 0.20 * severity
        if len(DEVELOPMENT_ACTIONS) > 1:
            self.neural_development.b_out[1] += 0.30 * severity
        self.neural_development.b_out = np.clip(self.neural_development.b_out, -8.0, 8.0)
        self.reasoning_graph.influence = float(np.clip(self.reasoning_graph.influence + 0.015 * severity, 0.0, 0.25))
        self.reasoning_graph.plasticity = float(np.clip(self.reasoning_graph.plasticity + 0.015 * severity, 0.0001, 0.20))
        lesson_context = self._resize_vector(np.asarray([
            severity,
            float(fitness_delta),
            total_trades,
            net_profit,
            profit_factor,
            drawdown,
            float(no_trade),
            float(loss_bad),
            float(risk_bad),
            float(execution_bad),
            float(len(reasons)),
            float(len(self.rejection_memory)),
        ], dtype=float), self.reasoning_graph.input_size)
        for controller in ("development", "curiosity", "risk", "action_head", "utility"):
            self._reasoning_context(controller, lesson_context, self.reasoning_graph.input_size)
            reward = -severity
            if no_trade and controller in {"curiosity", "development"}:
                reward = severity
            self._credit_reasoning(controller, reward, strength=2.0 + severity)
        event = {
            "round": self.revision,
            "type": "rejection_feedback_applied",
            **lesson,
            "mistake_episode_summary": self.mistake_episode_store.summary(),
            "hold_delta": float(hold_delta),
            "entry_delta": float(entry_delta),
            "manage_delta": float(manage_delta),
            "direct_attribution": attribution,
        }
        self.growth_events.append(event)
        return event

    def _compute_gates(self, obs: np.ndarray) -> np.ndarray:
        raw_logits = np.array([
            float(module.gate_w @ obs + module.gate_b) for module in self.modules
        ])
        memory_query = self._query_memory(obs)
        gate_bias = np.asarray(memory_query.get("gate_bias", np.zeros(len(self.modules))), dtype=float)
        if self.gating_mode == "sparsemax":
            gates = sparsemax(raw_logits)
        elif self.gating_mode == "top_k":
            k = min(2, len(self.modules))
            gates = top_k_gating(raw_logits, k=k)
        else:
            gates = np.array([
                1.0 / (1.0 + np.exp(-np.clip(v, -60.0, 60.0))) for v in raw_logits
            ])
        if gate_bias.size == gates.size:
            gates = np.clip(gates + gate_bias, 0.0, 1.5)
        return gates

    def _reasoning_context(self, controller: str, context: np.ndarray, target_size: int) -> np.ndarray:
        if not hasattr(self, "reasoning_graph") or self.reasoning_graph is None:
            self.reasoning_graph = ReasoningGraphSystem.seed()
        return self.reasoning_graph.enrich_context(controller, context, target_size=target_size)

    def _credit_reasoning(self, controller: str, reward: float, *, strength: float = 1.0) -> None:
        if not hasattr(self, "reasoning_graph") or self.reasoning_graph is None:
            self.reasoning_graph = ReasoningGraphSystem.seed()
        self.reasoning_graph.observe(controller, reward, strength=strength)

    def _action_head_context(
        self,
        state: np.ndarray,
        output: np.ndarray,
        memory_query: dict[str, object],
        gates: np.ndarray,
    ) -> np.ndarray:
        blocks = [
            self._resize_vector(np.asarray(state, dtype=float), 24),
            self._resize_vector(np.asarray(output, dtype=float), self.output_size),
            self._resize_vector(np.asarray(memory_query.get("replacement_action_bias", memory_query.get("success_action_bias", np.maximum(0.0, np.asarray(memory_query.get("action_bias", []), dtype=float)))), dtype=float), 12),
            self._resize_vector(np.asarray(memory_query.get("memory_context", []), dtype=float), 6),
            self._resize_vector(np.asarray(gates, dtype=float), 3),
            self._resize_vector(np.asarray([
                self._experience_starvation.get("no_trade_streak", 0.0),
                self._experience_starvation.get("unfilled_streak", 0.0),
                self._experience_starvation.get("cadence_deficit", 0.0),
                self._experience_starvation.get("starvation", 0.0),
                self._experience_starvation.get("intrinsic_reward", 0.0),
            ], dtype=float), 5),
        ]
        context = np.concatenate(blocks)
        return self._reasoning_context("action_head", context, self.action_parameter_head.input_size)

    def _risk_gate_context(
        self,
        obs: np.ndarray,
        output: np.ndarray,
        gates: np.ndarray,
        memory_query: dict[str, object],
    ) -> np.ndarray:
        memory_summary = self.memory_supernet.summary() if hasattr(self, "memory_supernet") else {}
        causal_bad = self.memory_supernet.causal_plasticity_signals(max_items=4) if hasattr(self, "memory_supernet") else []
        causal_severity = float(np.mean([float(item.get("severity", 0.0)) for item in causal_bad])) if causal_bad else 0.0
        base = np.array([
            float(np.mean(obs)) if obs.size else 0.0,
            float(np.std(obs)) if obs.size else 0.0,
            float(np.max(np.abs(output))) if output.size else 0.0,
            float(np.mean(gates)) if gates.size else 0.0,
            float(np.max(gates)) if gates.size else 0.0,
            float(memory_query.get("trust", 0.0)),
            float(memory_summary.get("positive_neurons", 0)) / max(1.0, float(memory_summary.get("neurons", 1))),
            causal_severity,
            self._reward_modulator,
            self._experience_starvation.get("starvation", 0.0),
            self._experience_starvation.get("cadence_deficit", 0.0),
            self._experience_starvation.get("unfilled_streak", 0.0),
        ], dtype=float)
        context = self._resize_vector(base, self.risk_cost_gate.input_size)
        return self._reasoning_context("risk", context, self.risk_cost_gate.input_size)

    def _apply_biological_action_heads(
        self,
        output: np.ndarray,
        state: np.ndarray,
        obs: np.ndarray,
        gates: np.ndarray,
        memory_query: dict[str, object],
    ) -> tuple[np.ndarray, dict[str, float]]:
        adjusted = np.asarray(output, dtype=float).copy()
        if adjusted.size != self.output_size:
            adjusted = self._resize_vector(adjusted, self.output_size)
        head_context = self._action_head_context(state, adjusted, memory_query, gates)
        params = self.action_parameter_head.parameters(head_context)
        parameter_offset = PARAMETER_OFFSET if adjusted.size >= OUTPUT_COUNT else 11
        param_count = min(8, max(0, adjusted.size - parameter_offset), params.size)
        if param_count > 0:
            adjusted[parameter_offset:parameter_offset + param_count] = np.tanh(
                0.35 * adjusted[parameter_offset:parameter_offset + param_count] + 0.65 * params[:param_count]
            )
        risk_info = self.risk_cost_gate.gate(self._risk_gate_context(obs, adjusted, gates, memory_query))
        dampen = float(np.clip(0.5 * risk_info["risk"] + 0.5 * risk_info["cost"], 0.0, 1.0))
        action_count = ACTION_COUNT if adjusted.size >= OUTPUT_COUNT else 11
        if adjusted.size >= action_count:
            adjusted[1:7] *= 1.0 - 0.35 * dampen
            adjusted[0] += 0.15 * risk_info["patience"]
            adjusted[8:11] += 0.10 * dampen
        return adjusted, risk_info

    def _utility_context(self, fitness: float) -> np.ndarray:
        memory_summary = self.memory_supernet.summary() if hasattr(self, "memory_supernet") else {}
        scores = np.asarray([m.score for m in self.modules], dtype=float) if self.modules else np.zeros(1)
        utilities = np.asarray([m.utility for m in self.modules], dtype=float) if self.modules else np.zeros(1)
        gates = np.asarray([m.gate_ema for m in self.modules], dtype=float) if self.modules else np.zeros(1)
        plasticities = np.asarray([m.plasticity for m in self.modules], dtype=float) if self.modules else np.zeros(1)
        base = np.array([
            np.tanh(float(fitness)),
            np.tanh(float(self._reward_modulator)),
            float(len(self.modules)) / 16.0,
            float(np.mean(scores)),
            float(np.std(scores)),
            float(np.mean(utilities)),
            float(np.min(utilities)),
            float(np.mean(gates)),
            float(np.max(gates)),
            float(np.mean(plasticities)),
            float(memory_summary.get("neurons", 0)) / 1000.0,
            float(memory_summary.get("positive_neurons", 0)) / max(1.0, float(memory_summary.get("neurons", 1))),
            float(memory_summary.get("strategies", 0)) / 96.0,
            float(memory_summary.get("meta_strategies", 0)) / 32.0,
            float(memory_summary.get("causal_nodes", 0)) / 1000.0,
        ], dtype=float)
        context = self._resize_vector(base, self.neural_utility.input_size)
        return self._reasoning_context("utility", context, self.neural_utility.input_size)

    def activate(self, observation: np.ndarray) -> np.ndarray:
        self.ensure_compatible()
        obs = np.asarray(observation, dtype=float)
        if obs.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(obs.size, self.input_size)] = obs[: self.input_size]
            obs = padded

        self._step_count += 1
        self._last_observation = obs.copy()
        gates = self._compute_gates(obs)

        hidden_states: list[np.ndarray] = []
        raw_outputs: list[np.ndarray] = []
        use_m1_graph = self._uses_m1_graph()
        for idx, module in enumerate(self.modules):
            previous_hidden = self._runtime_memory.get(module.name)
            gate, hidden, raw = module.activate(
                obs,
                previous_hidden,
                training=bool(getattr(self, "_learning_enabled", True)),
            )
            if use_m1_graph and module.m1_component is not None:
                module_obs = self._masked_module_observation(module, obs)
                if isinstance(module.m1_component, M1GraphModule):
                    m1_context = self._query_memory(module_obs)
                else:
                    prelim_state = module.m1_component.state_encoder.encode(module_obs)
                    m1_context = self._query_memory(prelim_state)
                m1_output, m1_error, m1_novelty, m1_conflict, m1_confidence = module.m1_component.forward(module_obs, context=m1_context)
                if m1_output.size == raw.size:
                    raw = m1_output * np.clip(module.output_mask, 0.0, 1.0)
            hidden_states.append(hidden)
            raw_outputs.append(raw)

        hidden_states, raw_outputs = self._integrate_lateral(hidden_states, raw_outputs)

        gated_outputs = []
        for idx, module in enumerate(self.modules):
            gate = gates[idx]
            hidden = hidden_states[idx]
            self._runtime_memory[module.name] = hidden
            module.observe_usage(gate, hidden)
            gated_outputs.append(gate * raw_outputs[idx])
            if module.plasticity_coeffs is not None and self._step_count % 10 == 0:
                module.apply_plasticity_rules(
                    obs[:module.input_size],
                    hidden,
                    current_time=float(self._step_count),
                    reward_modulator=self._reward_modulator,
                )

        gate_sum = max(float(np.sum(gates)), 1e-9)
        output = np.sum(gated_outputs, axis=0) / gate_sum
        memory_query = self._query_memory(obs)
        raw_action_bias = np.asarray(memory_query.get("action_bias", np.zeros(self.output_size)), dtype=float)
        action_bias = np.asarray(memory_query.get("replacement_action_bias", memory_query.get("success_action_bias", np.maximum(0.0, raw_action_bias))), dtype=float)
        curiosity_context = self.curiosity_system.context_from(
            state=obs,
            action=output,
            memory_query=memory_query,
            causal_signals=self.memory_supernet.causal_plasticity_signals(max_items=4) if hasattr(self, "memory_supernet") else [],
            starvation=self._experience_starvation,
        )
        curiosity_context = self._reasoning_context("curiosity", curiosity_context, self.curiosity_system.input_size)
        curiosity = self.curiosity_system.forward(curiosity_context)
        curiosity_action = self._resize_vector(np.asarray(curiosity.get("action_delta", []), dtype=float), self.output_size)
        if action_bias.size == output.size:
            output = output + action_bias
        # V8: raw curiosity may generate replay/shadow work but never live
        # entry logits. This removes uncertainty-driven BUY/SELL probing.
        self._v8_curiosity_diagnostic = {
            "mode": str(getattr(self, "live_curiosity_mode", "shadow_only") or "shadow_only"),
            "raw_delta_norm": float(np.linalg.norm(curiosity_action)),
            "live_action_delta_norm": 0.0,
            "shadow_replay_requested": bool(float(curiosity.get("novelty", 0.0) or 0.0) > 0.0),
        }
        output, _risk_info = self._apply_biological_action_heads(output, obs, obs, gates, memory_query)
        return np.tanh(output)

    @staticmethod
    def _action_index_from_name(action_name: str) -> int:
        mapping = {
            "HOLD": 0,
            "MARKET_BUY": 1,
            "MARKET_SELL": 2,
            "BUY_LIMIT": 3,
            "SELL_LIMIT": 4,
            "BUY_STOP": 5,
            "SELL_STOP": 6,
            "CANCEL_ORDERS": 7,
            "CLOSE_WORST": 8,
            "CLOSE_BEST": 9,
            "CLOSE_ALL": 10,
            "MANAGE_POSITION": 11,
        }
        return mapping.get(str(action_name).upper(), 0)

    def apply_causal_memory_plasticity(
        self,
        *,
        max_signals: int = 16,
        lr: float = 0.0015,
        min_abs_utility: float = 0.05,
    ) -> dict[str, object]:
        self.ensure_compatible()
        signals = self.memory_supernet.causal_plasticity_signals(max_items=max_signals, min_abs_utility=min_abs_utility)
        if not signals:
            return {"applied": 0, "signals": 0, "max_severity": 0.0}

        applied = 0
        max_severity = 0.0
        for signal in signals:
            severity = float(signal.get("severity", 0.0))
            if severity <= 0.0:
                continue
            max_severity = max(max_severity, severity)
            obs = np.asarray(signal.get("obs_embedding", np.zeros(32)), dtype=float)
            tags = set(signal.get("tags", []) or [])
            bad_action_idx = min(self.output_size - 1, max(0, self._action_index_from_name(str(signal.get("action_name", "")))))
            lifecycle_bad = bool(tags & {"near_tp_missed", "near_tp_giveback", "gave_back_profit", "closed_after_near_tp_bad", "should_have_partial_or_trail"})
            delta = float(np.clip(lr * severity, 0.00005, 0.015))
            safe_action_idx = 0
            parameter_offset = PARAMETER_OFFSET if self.output_size >= OUTPUT_COUNT else 11
            management_rows = [idx for idx in (parameter_offset + 4, parameter_offset + 5, parameter_offset + 6) if idx < self.output_size]

            for module in self.modules:
                context = self._resize_vector(obs, module.input_size)
                hidden_context = np.tanh(module.w_in @ context + module.b_hidden)
                hidden_norm = hidden_context / max(1.0, float(np.linalg.norm(hidden_context)))
                input_norm = context / max(1.0, float(np.linalg.norm(context)))
                scale = delta / max(1, len(self.modules))

                # V10: causal bad signals do not weaken bad_action_idx.
                # They strengthen the cause-conditioned safe/replacement path.
                if safe_action_idx < module.w_out.shape[0]:
                    module.w_out[safe_action_idx, :] += 0.5 * scale * hidden_norm
                    module.b_out[safe_action_idx] += 0.1 * scale
                if lifecycle_bad:
                    for row in management_rows:
                        module.w_out[row, :] += scale * hidden_norm
                        module.b_out[row] += 0.2 * scale

                gate_direction = -1.0 if float(signal.get("utility", 0.0)) < 0.0 else 0.5
                module.gate_w += gate_direction * scale * input_norm
                module.gate_b += gate_direction * 0.1 * scale
                if module.w_rec is not None and hidden_norm.size == module.w_rec.shape[0]:
                    module.w_rec -= 0.1 * scale * np.outer(hidden_norm, hidden_norm)
                if module.w_lat is not None and hidden_norm.size == module.w_lat.shape[0]:
                    module.w_lat -= 0.05 * scale * np.outer(hidden_norm, hidden_norm)

                if self._uses_m1_graph() and module.m1_component is not None:
                    m1 = module.m1_component
                    if isinstance(m1, M1GraphModule):
                        # The trial graph has no hypothesis-bank matrix.  Keep
                        # causal learning bounded to output bias/weak incoming
                        # edges on the *candidate* graph; production remains
                        # untouched until the enclosing MT5 trial is promoted.
                        output_ids = list(m1.graph.regions.get("output").neuron_ids)
                        def adjust_output(index: int, amount: float) -> None:
                            if 0 <= index < len(output_ids):
                                node = m1.graph.neurons[output_ids[index]]
                                node.bias = float(np.clip(node.bias + amount, -2.0, 2.0))
                                for edge in m1.graph.edges.values():
                                    if edge.dst == node.id and edge.enabled:
                                        edge.weight = float(np.clip(edge.weight + 0.10 * amount, -2.0, 2.0))
                                        edge.gate = float(np.clip(edge.gate, -8.0, 2.0))
                        # Do not suppress bad_action_idx; strengthen replacement.
                        adjust_output(safe_action_idx, 0.5 * scale)
                        if lifecycle_bad:
                            for row in management_rows:
                                adjust_output(row, scale)
                    else:
                        if safe_action_idx < m1.hypothesis_bank.w_decode.shape[0]:
                            m1.hypothesis_bank.w_decode[safe_action_idx, :] += 0.5 * scale
                        for row in management_rows:
                            if row < m1.hypothesis_bank.w_decode.shape[0]:
                                m1.hypothesis_bank.w_decode[row, :] += scale
                applied += 1

        for module in self.modules:
            module.w_out = np.clip(module.w_out, -8.0, 8.0)
            module.b_out = np.clip(module.b_out, -8.0, 8.0)
            module.gate_w = np.clip(module.gate_w, -8.0, 8.0)
            module.gate_b = float(np.clip(module.gate_b, -8.0, 8.0))
            if module.w_rec is not None:
                module.w_rec = np.clip(module.w_rec, -4.0, 4.0)
            if module.w_lat is not None:
                module.w_lat = np.clip(module.w_lat, -4.0, 4.0)
        if applied:
            self.growth_events.append({
                "round": self.revision,
                "type": "causal_memory_plasticity",
                "applied": int(applied),
                "signals": int(len(signals)),
                "max_severity": float(max_severity),
            })
        return {"applied": int(applied), "signals": int(len(signals)), "max_severity": float(max_severity)}

    def _curiosity_context(self) -> tuple[np.ndarray, list[dict[str, object]], dict[str, object]]:
        self.ensure_compatible()
        state = np.zeros(32, dtype=float)
        if self.memory_supernet.neurons:
            recent = [self.memory_supernet.neurons[node_id].obs_embedding for node_id in self.memory_supernet.recent_ids if node_id in self.memory_supernet.neurons]
            if recent:
                state = np.mean(np.vstack(recent[-8:]), axis=0)
        action = np.zeros(self.output_size, dtype=float)
        if self.memory_supernet.neurons:
            recent_actions = [self.memory_supernet.neurons[node_id].action_embedding for node_id in self.memory_supernet.recent_ids if node_id in self.memory_supernet.neurons]
            if recent_actions:
                action = self._resize_vector(np.mean(np.vstack(recent_actions[-8:]), axis=0), self.output_size)
        memory_query = self._query_memory(state)
        causal_signals = self.memory_supernet.causal_plasticity_signals(max_items=8)
        context = self.curiosity_system.context_from(
            state=state,
            action=action,
            memory_query=memory_query,
            causal_signals=causal_signals,
            starvation=self._experience_starvation,
        )
        context = self._reasoning_context("curiosity", context, self.curiosity_system.input_size)
        return context, causal_signals, memory_query

    def _loss_action_mutation_evidence(self, *, limit: int = 128) -> tuple[np.ndarray, dict[str, object]]:
        """Rows and connections eligible for sigma mutation from realised bad outcomes."""
        self.ensure_compatible()
        mask = np.zeros(self.output_size, dtype=bool)
        module_hidden: dict[int, dict[str, np.ndarray]] = {}
        graph_edges: dict[int, set[str]] = {}
        graph_neurons: dict[int, set[str]] = {}

        def mark_trace(trace: StepTrace) -> None:
            idx = self._trace_action_index(trace, self.output_size)
            if 0 <= idx < mask.size:
                mask[idx] = True
            details = dict(getattr(trace, "action_details", {}) or {})
            for key in ("bridge_action", "requested_action", "action"):
                try:
                    extra = int(details.get(key, -1))
                except (TypeError, ValueError):
                    extra = -1
                if 0 <= extra < mask.size:
                    mask[extra] = True
            for module_trace in tuple(getattr(trace, "module_action_traces", ()) or ()):
                action_hidden = module_hidden.setdefault(idx, {})
                hidden = np.asarray(getattr(module_trace, "hidden", []), dtype=float).reshape(-1)
                if hidden.size <= 0:
                    continue
                current = action_hidden.get(str(module_trace.module_name))
                if current is None or current.size != hidden.size:
                    current = np.zeros(hidden.size, dtype=float)
                # Use absolute activation as an eligibility mask; action_delta
                # still decides the update direction.
                action_hidden[str(module_trace.module_name)] = current + np.tanh(np.abs(hidden))
            active_edges = {str(item) for item in tuple(getattr(trace, "active_edge_ids", ()) or ()) if str(item)}
            active_neurons = {str(item) for item in tuple(getattr(trace, "active_neuron_ids", ()) or ()) if str(item)}
            if active_edges:
                graph_edges.setdefault(idx, set()).update(active_edges)
            if active_neurons:
                graph_neurons.setdefault(idx, set()).update(active_neurons)

        for item in list(getattr(self, "experience_buffer", []) or [])[-max(1, int(limit)):]:
            if not isinstance(item, dict):
                continue
            trace = item.get("trace")
            if not isinstance(trace, StepTrace):
                continue
            reward = float(item.get("reward", 0.0) or 0.0)
            td_delta = float(item.get("td_delta", getattr(trace, "td_delta", 0.0)) or 0.0)
            trace_td = float(getattr(trace, "td_delta", 0.0) or 0.0)
            if min(reward, td_delta, trace_td) < 0.0:
                mark_trace(trace)
        if hasattr(self, "memory_supernet"):
            for signal in self.memory_supernet.causal_plasticity_signals(max_items=limit):
                action_name = str(signal.get("action_name", "") or "")
                if action_name:
                    idx = self._action_index_from_name(action_name)
                    if 0 <= idx < mask.size:
                        mask[idx] = True
                embedding = np.asarray(signal.get("action_embedding", []), dtype=float).reshape(-1)
                if embedding.size:
                    idx = int(np.argmax(embedding[:min(mask.size, embedding.size)]))
                    if 0 <= idx < mask.size:
                        mask[idx] = True
        evidence: dict[str, object] = {
            "module_hidden": module_hidden,
            "graph_edges": graph_edges,
            "graph_neurons": graph_neurons,
        }
        return mask, evidence

    def _loss_action_mask(self, *, limit: int = 128) -> np.ndarray:
        """Rows eligible for sigma mutation, based only on realised bad outcomes."""
        mask, _evidence = self._loss_action_mutation_evidence(limit=limit)
        return mask

    def apply_curiosity_proposal(
        self,
        *,
        strength: float = 0.02,
        residual_noise: float = 0.0,
        action_mask: np.ndarray | None = None,
        action_evidence: dict[str, object] | None = None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        self.ensure_compatible()
        rng = rng or np.random.default_rng()
        context, causal_signals, memory_query = self._curiosity_context()
        proposal = self.curiosity_system.propose_adjustments(context)
        action_delta = np.asarray(proposal.get("action_delta", []), dtype=float)
        gate_delta = np.asarray(proposal.get("gate_delta", []), dtype=float)
        recurrent_delta = np.asarray(proposal.get("recurrent_delta", []), dtype=float)
        drive = float(proposal.get("drive", 0.0))
        risk_dampen = float(proposal.get("risk_dampen", 0.0))
        scale = float(np.clip(strength * (0.25 + drive) * (1.0 - 0.5 * risk_dampen), 0.0001, 0.05))
        mask = self._resize_vector(np.asarray(action_mask, dtype=float), self.output_size) > 0.5 if action_mask is not None else np.ones(self.output_size, dtype=bool)
        evidence = action_evidence or {}
        module_hidden = evidence.get("module_hidden", {}) if isinstance(evidence, dict) else {}
        graph_edges = evidence.get("graph_edges", {}) if isinstance(evidence, dict) else {}
        graph_neurons = evidence.get("graph_neurons", {}) if isinstance(evidence, dict) else {}
        mutated_rows: set[int] = set()
        mutated_connections = 0

        for module_index, module in enumerate(self.modules):
            if action_delta.size:
                action_vec = self._resize_vector(action_delta, module.output_size)
                local_mask = self._resize_vector(mask.astype(float), module.output_size) > 0.5
                action_vec = np.where(local_mask, action_vec, 0.0)
                hidden_basis = np.ones(module.hidden_size, dtype=float) / max(1.0, np.sqrt(float(module.hidden_size)))
                for row in np.flatnonzero(local_mask & (np.abs(action_vec) > 1e-12)):
                    row_evidence = module_hidden.get(int(row), {}) if isinstance(module_hidden, dict) else {}
                    row_hidden = row_evidence.get(module.name) if isinstance(row_evidence, dict) else None
                    basis = hidden_basis
                    if row_hidden is not None:
                        basis = self._resize_vector(np.asarray(row_hidden, dtype=float), module.hidden_size)
                        active = basis > 0.01
                        if np.any(active):
                            basis = np.where(active, basis, 0.0)
                            basis = basis / max(1.0, float(np.linalg.norm(basis)))
                    module.w_out[int(row), :] += scale * float(action_vec[int(row)]) * basis
                    module.b_out[int(row)] += 0.25 * scale * float(action_vec[int(row)])
                    mutated_rows.add(int(row))
                    mutated_connections += int(np.count_nonzero(np.abs(basis) > 1e-12))
            if gate_delta.size and action_mask is None:
                g = float(gate_delta[module_index % gate_delta.size])
                input_basis = np.ones(module.input_size, dtype=float) / max(1.0, np.sqrt(float(module.input_size)))
                module.gate_w += 0.5 * scale * g * input_basis
                module.gate_b += 0.1 * scale * g
            if recurrent_delta.size and module.w_rec is not None and action_mask is None:
                r = self._resize_vector(recurrent_delta, module.hidden_size)
                r = r / max(1.0, float(np.linalg.norm(r)))
                module.w_rec += 0.25 * scale * np.outer(r, r)
            if self._uses_m1_graph() and module.m1_component is not None and action_delta.size:
                m1 = module.m1_component
                if isinstance(m1, M1GraphModule):
                    output_ids = list(m1.graph.regions.get("output").neuron_ids)
                    action_vec = self._resize_vector(action_delta, len(output_ids))
                    for index, node_id in enumerate(output_ids):
                        if index >= mask.size or not bool(mask[index]):
                            continue
                        delta = float(np.clip(0.20 * scale * action_vec[index], -0.02, 0.02))
                        node = m1.graph.neurons[node_id]
                        node.bias = float(np.clip(node.bias + delta, -2.0, 2.0))
                        active_edges = graph_edges.get(index, set()) if isinstance(graph_edges, dict) else set()
                        active_neurons = graph_neurons.get(index, set()) if isinstance(graph_neurons, dict) else set()
                        for edge in m1.graph.edges.values():
                            if edge.dst == node_id and edge.enabled:
                                has_active_evidence = bool(active_edges or active_neurons)
                                if has_active_evidence and edge.id not in active_edges and edge.src not in active_neurons:
                                    continue
                                edge.weight = float(np.clip(edge.weight + 0.05 * delta, -2.0, 2.0))
                                mutated_connections += 1
            if residual_noise > 0.0:
                local_mask = self._resize_vector(mask.astype(float), module.output_size) > 0.5
                if np.any(local_mask):
                    noise = rng.normal(0.0, residual_noise, size=module.w_out.shape)
                    for row in np.flatnonzero(local_mask):
                        row_evidence = module_hidden.get(int(row), {}) if isinstance(module_hidden, dict) else {}
                        row_hidden = row_evidence.get(module.name) if isinstance(row_evidence, dict) else None
                        if row_hidden is None:
                            module.w_out[int(row), :] += noise[int(row), :]
                            mutated_connections += int(module.w_out.shape[1])
                        else:
                            hidden_gate = self._resize_vector(np.asarray(row_hidden, dtype=float), module.hidden_size) > 0.01
                            if np.any(hidden_gate):
                                module.w_out[int(row), hidden_gate] += noise[int(row), hidden_gate]
                                mutated_connections += int(np.count_nonzero(hidden_gate))
                        mutated_rows.add(int(row))

        self.growth_events.append({
            "round": self.revision,
            "type": "curiosity_neural_proposal",
            "drive": drive,
            "risk_dampen": risk_dampen,
            "strength": scale,
            "causal_signals": len(causal_signals),
            "loss_action_rows": sorted(mutated_rows),
            "loss_action_connections": int(mutated_connections),
            "memory_matches": len(memory_query.get("matches", [])) if isinstance(memory_query, dict) else 0,
        })
        return {
            "drive": drive,
            "risk_dampen": risk_dampen,
            "strength": scale,
            "causal_signals": len(causal_signals),
            "loss_action_rows": sorted(mutated_rows),
            "loss_action_connections": int(mutated_connections),
        }

    def _graph_m1_modules(self) -> list[tuple[ModuleState, M1GraphModule]]:
        if not self._uses_m1_graph():
            return []
        return [
            (module, module.m1_component)
            for module in self.modules
            if isinstance(getattr(module, "m1_component", None), M1GraphModule)
        ]

    def graph_observability_metrics(self) -> dict[str, float]:
        """Summarize graph decision evidence without changing policy state.

        These values are intentionally derived from immutable decision traces
        so trial-history reporting can show whether a structural candidate is
        reducing prediction error, using memory productively and oscillating
        between selector hypotheses.
        """
        traces: list[StepTrace] = []
        for item in getattr(self, "experience_buffer", [])[-64:]:
            trace = item.get("trace") if isinstance(item, dict) else None
            if isinstance(trace, StepTrace):
                traces.append(trace)
        if not traces:
            return {
                "prediction_error_mean": 0.0,
                "prediction_error_trend": 0.0,
                "memory_usefulness": 0.0,
                "selector_switch_rate": 0.0,
                "credit_active_neurons": 0.0,
                "credit_active_edges": 0.0,
            }
        errors = np.asarray([float(trace.prediction_error) for trace in traces], dtype=float)
        matches = np.asarray([float(trace.memory_match_quality) for trace in traces], dtype=float)
        rewards = np.asarray([float(item.get("reward", 0.0)) for item in getattr(self, "experience_buffer", [])[-len(traces):] if isinstance(item, dict)], dtype=float)
        selected = [int(trace.selected_idx) for trace in traces]
        switches = sum(1 for before, after in zip(selected, selected[1:]) if before != after)
        split = max(1, errors.size // 2)
        trend = float(np.mean(errors[split:]) - np.mean(errors[:split])) if errors.size > 1 else 0.0
        reward_scale = float(np.tanh(np.mean(rewards))) if rewards.size else 0.0
        usefulness = float(np.clip(np.mean(matches) * (0.5 + 0.5 * max(0.0, reward_scale)), 0.0, 1.0))
        return {
            "prediction_error_mean": float(np.mean(errors)),
            "prediction_error_trend": trend,
            "memory_usefulness": usefulness,
            "selector_switch_rate": float(switches / max(1, len(selected) - 1)),
            "credit_active_neurons": float(np.mean([len(trace.active_neuron_ids) for trace in traces])),
            "credit_active_edges": float(np.mean([len(trace.active_edge_ids) for trace in traces])),
        }

    def _graph_candidate_evidence(self, signal: GrowthSignal) -> tuple[object | None, dict[str, float], list[object]]:
        """Build need evidence from real decision traces, not hard win-rate rules."""
        trace = None
        history: list[object] = []
        for item in getattr(self, "experience_buffer", [])[-32:]:
            candidate = item.get("trace") if isinstance(item, dict) else None
            if isinstance(candidate, StepTrace):
                history.append(candidate)
                trace = candidate
        learning = {
            "prediction_error": float(signal.error),
            "td_delta": float(self._reward_modulator),
            "reward": float(self._reward_modulator),
            "expectancy": float(self._reward_modulator),
        }
        if trace is not None:
            learning["prediction_error"] = max(learning["prediction_error"], float(trace.prediction_error))
            return trace, learning, history
        # A fresh candidate may not have a journal yet.  Use the continuous
        # neural growth signal as trace-shaped teacher evidence, not a fixed
        # win-rate rule, so the role controller can still make a bounded first
        # proposal.
        return {
            "trace_id": f"graph-bootstrap-{self.revision}",
            "regime": "bootstrap",
            "prediction_error": float(signal.error),
            "novelty": float(signal.novelty),
            "conflict": float(signal.conflict),
            "memory_match_quality": float(np.clip(1.0 - signal.representation_need, 0.0, 1.0)),
            "lifecycle_tags": (),
            "account_context": dict(getattr(self, "_account_context", {}) or {}),
            "experience_starvation": dict(getattr(self, "_experience_starvation", {}) or {}),
        }, learning, history

    def _propose_graph_supernetwork_candidate(
        self,
        *,
        rng: np.random.Generator,
        mutation_sigma: float,
    ) -> dict[str, object]:
        """Apply graph topology work only after this policy is already a copy.

        ``SelfGrowingPolicy.propose`` calls this method on a trial copy.  The
        baseline's M1 graph, weights, memory state and topology are therefore
        never mutated by a structural proposal.
        """
        self.ensure_compatible()
        if self.graph_supernetwork is None:
            self.graph_supernetwork = GraphSupernetworkRuntime()
        signal = self.growth_signal()
        trace, learning, history = self._graph_candidate_evidence(signal)
        summaries: list[dict[str, object]] = []
        for index, (module, m1) in enumerate(self._graph_m1_modules()):
            candidate_id = f"graph-{self.revision}-{index}-{len(self.growth_events)}"
            proposal = self.graph_supernetwork.propose(
                m1.graph,
                candidate_id=candidate_id,
                trace=trace,
                learning_signal=learning,
                history=history,
                account_context=getattr(self, "_account_context", {}),
                starvation=getattr(self, "_experience_starvation", {}),
                rng=rng,
            )
            proposal_dict = proposal.to_dict()
            proposal_dict["module"] = module.name
            proposal_dict["mutation_sigma"] = float(mutation_sigma)
            summaries.append(proposal_dict)
            self.growth_events.append({
                "round": self.revision,
                "type": "graph_candidate_proposal",
                **proposal_dict,
            })
            m1.graph.assert_invariants()
        return {
            "proposals": summaries,
            "module_need_metrics": self.graph_supernetwork.metrics(
                self._graph_m1_modules()[0][1].graph if self._graph_m1_modules() else None
            ),
        }

    def _maybe_spawn_graph_child(self) -> dict[str, object] | None:
        manager = self.child_supernetwork_manager
        if manager is None or not bool(self.graph_supernetwork_config.get("enable_child_supernet", False)):
            return None
        graph_modules = self._graph_m1_modules()
        if not graph_modules:
            return None
        _module, m1 = graph_modules[0]
        record = manager.spawn_candidate(
            parent_graph=m1.graph,
            input_size=self.input_size,
            output_size=self.output_size,
            graph_summary={"graph_neurons": float(len(m1.graph.neurons)), "graph_connections": float(len(m1.graph.edges))},
            memory_summary=self.memory_supernet.summary() if hasattr(self, "memory_supernet") else {},
            winning_causal_patterns=self.memory_supernet.causal_plasticity_signals(max_items=8) if hasattr(self, "memory_supernet") else (),
            seed=int(time.time_ns() % (2**31 - 1)),
            journal_path=self.graph_supernetwork_config.get("experience_journal_path"),
        )
        if record is None:
            return None
        event = {
            "round": self.revision,
            "type": "child_supernetwork_spawned",
            "network_id": record.network_id,
            "blueprint_id": record.blueprint.blueprint_id,
            "role": record.blueprint.role,
            "warmup_rows": record.warmup_rows,
            "status": record.status,
        }
        self.growth_events.append(event)
        return event

    def validate_graph_child(
        self,
        network_id: str,
        *,
        replay_validator,
        mt5_validator,
        trial_history_path: str | Path | None = None,
    ) -> bool:
        if self.child_supernetwork_manager is None:
            raise ValueError("child supernetwork is not enabled")
        promoted = self.child_supernetwork_manager.validate_child(
            network_id,
            replay_validator=replay_validator,
            mt5_validator=mt5_validator,
            trial_history_path=trial_history_path,
        )
        self.growth_events.append({
            "round": self.revision,
            "type": "child_supernetwork_validation",
            "network_id": network_id,
            "promoted": bool(promoted),
        })
        return promoted

    def select_graph_policy(
        self,
        *,
        baseline_confidence: float,
        baseline_utility: float,
        child_scores: dict[str, tuple[float, float]] | None = None,
    ) -> dict[str, object]:
        """Return an auditable selector decision; safety/action decode stays in Python/EA."""
        if self.child_supernetwork_manager is None:
            return {"selected_policy_id": "m1", "fallback_policy_id": "m1", "reason": "child_supernetwork_disabled", "used_fallback": True}
        account = dict(getattr(self, "_account_context", {}) or {})
        trace = self.child_supernetwork_manager.select(
            memory_match=float(np.clip(1.0 - getattr(self, "_reward_modulator", 0.0) ** 2, 0.0, 1.0)),
            account_stress=float(np.clip(account.get("current_equity_drawdown", 0.0), 0.0, 1.0)),
            drawdown=float(np.clip(account.get("balance_drawdown", 0.0), 0.0, 1.0)),
            open_positions=int(max(0, account.get("position_count_norm", 0.0) * 10)),
            baseline_confidence=float(baseline_confidence),
            baseline_utility=float(baseline_utility),
            child_scores=child_scores,
        )
        event = {
            "round": self.revision,
            "type": "policy_ensemble_selection",
            "selected_policy_id": trace.selected_policy_id,
            "fallback_policy_id": trace.fallback_policy_id,
            "reason": trace.reason,
            "used_fallback": trace.used_fallback,
            "scores": dict(trace.scores),
        }
        self.growth_events.append(event)
        return event

    def record_graph_candidate_outcome(self, *, promoted: bool, source_policy: "SelfGrowingPolicy" | None = None) -> None:
        """Feed MT5 promote/reject back into the neural role prior.

        Reject feedback belongs to the surviving baseline too, so callers may
        pass the evaluated trial as ``source_policy`` when the baseline itself
        remains unchanged.
        """
        if self.graph_supernetwork is None:
            return
        origin = source_policy if source_policy is not None else self
        proposal = next(
            (
                event for event in reversed(getattr(origin, "growth_events", []))
                if event.get("type") == "graph_candidate_proposal"
            ),
            None,
        )
        if proposal is None:
            return
        self.graph_supernetwork.record_outcome(proposal, promoted=promoted)
        compaction = {"skipped": True, "reason": "not_promoted"}
        replacement_audit = {"promoted": False, "validated": 0, "consolidated": 0, "shadow": 0, "rejected": 0}
        if bool(promoted) and bool(getattr(self, "graph_trial_mode", False)):
            replacement_audit = self.finalize_parameter_replacement_promotion()
            compaction = self.compact_graph_structures()
        if self.child_supernetwork_manager is not None:
            if promoted:
                self.child_supernetwork_manager.on_root_promotion()
                self._maybe_spawn_graph_child()
            else:
                self.child_supernetwork_manager.on_root_rejection()
        self.growth_events.append({
            "round": self.revision,
            "type": "graph_candidate_outcome",
            "candidate_id": proposal.get("candidate_id", ""),
            "role": proposal.get("role", ""),
            "reason": proposal.get("reason", ""),
            "promoted": bool(promoted),
            "compaction": compaction,
            "parameter_replacement_audit": replacement_audit,
        })

    def _apply_mistake_replay_to_candidate(self, *, max_items: int = 64) -> dict[str, object]:
        self.ensure_compatible(target_input_size=self.input_size)
        episodes = self.mistake_episode_store.query_recent_bad(limit=max_items, min_severity=0.05)
        if not episodes:
            return {"episodes": 0, "repeated_mistakes": 0, "safer_alternatives": 0}
        summary = MistakeReplayEngine(max_items=max_items).replay(self, episodes, mutate=True)
        event = {
            "round": self.revision,
            "type": "mistake_replay_candidate",
            "episodes": int(summary.get("episodes", 0)),
            "repeated_mistakes": int(summary.get("repeated_mistakes", 0)),
            "safer_alternatives": int(summary.get("safer_alternatives", 0)),
            "mean_score": float(summary.get("mean_score", 0.0)),
            "mutated_edges": int(summary.get("mutated_edges", 0)),
            "mutated_neurons": int(summary.get("mutated_neurons", 0)),
            "mutated_module_links": int(summary.get("mutated_module_links", 0)),
            "items": list(summary.get("items", []))[:8],
        }
        self.growth_events.append(event)
        return event

    def propose(self, rng: np.random.Generator, mutation_sigma: float, max_modules: int | None = None) -> "SelfGrowingPolicy":
        """Create one isolated, complexity-bounded graph candidate."""
        configured_cap = int(dict(getattr(self, "graph_supernetwork_config", {}) or {}).get("max_modules", 4) or 4)
        module_cap = max(1, int(max_modules if max_modules is not None else configured_cap))
        trial = self.copy()
        trial.ensure_compatible(target_input_size=trial.input_size)
        trial.revision += 1
        trial.graph_trial_mode = True
        trial.graph_trial_id = f"graph-trial-{trial.revision}-{time.time_ns()}"
        if trial.graph_supernetwork is None:
            trial.graph_supernetwork = GraphSupernetworkRuntime()
        trial.graph_supernetwork.begin_trial(trial.graph_trial_id)
        trial.growth_events.append({
            "round": trial.revision,
            "type": "graph_trial_started",
            "trial_id": trial.graph_trial_id,
        })
        plasticity_summary = trial.apply_causal_memory_plasticity()
        loss_action_mask, loss_action_evidence = trial._loss_action_mutation_evidence()
        curiosity_summary = trial.apply_curiosity_proposal(
            strength=max(0.001, mutation_sigma * 0.5),
            residual_noise=0.0,
            action_mask=loss_action_mask,
            action_evidence=loss_action_evidence,
            rng=rng,
        )
        graph_summary = trial._propose_graph_supernetwork_candidate(rng=rng, mutation_sigma=mutation_sigma)
        width_growth_summary = trial._maybe_expand_hidden_width(rng=rng)
        depth_growth_summary = trial._maybe_expand_hidden_depth(rng=rng)
        # A candidate may not materialize more module-level complexity than
        # the declared trial budget.  Ignore/reject excess novelty instead of
        # letting candidate order change available capacity.
        if len(trial.modules) > module_cap:
            trial.modules = trial.modules[:module_cap]
            trial._ensure_lateral_edges()
            trial.growth_events.append({"round": trial.revision, "type": "module_budget_trim", "cap": module_cap})
        mistake_replay_summary = trial._apply_mistake_replay_to_candidate()
        if not trial._check_stability():
            rejected = self.copy()
            rejected.growth_events.append({
                "round": trial.revision,
                "type": "reject_unstable_graph_candidate",
                "causal_plasticity": plasticity_summary,
                "curiosity_proposal": curiosity_summary,
                "graph_proposal": graph_summary,
                "hidden_width_growth": width_growth_summary,
                "hidden_depth_growth": depth_growth_summary,
                "mistake_replay": mistake_replay_summary,
            })
            return rejected
        return trial

    def _neural_development_output(self, signal: GrowthSignal) -> NeuralDevelopmentOutput:
        self.ensure_compatible(target_input_size=self.input_size)
        memory_summary = self.memory_supernet.summary() if hasattr(self, "memory_supernet") else {}
        context = self.neural_development.context_from(
            signal=signal,
            modules=self.modules,
            memory_summary=memory_summary,
            reward_modulator=self._reward_modulator,
        )
        context = self._reasoning_context("development", context, self.neural_development.input_size)
        return self.neural_development.forward(context)

    def _check_stability(self) -> bool:
        """Reject numerically unsafe candidates before they can be promoted.

        The graph branch verifies topology and every mutable scalar so a
        candidate cannot be promoted merely because it avoided graph checks.
        """
        for module in self.modules:
            w_norm = float(np.linalg.norm(module.w_in))
            if w_norm > 100.0:
                return False
            if module.w_rec is not None:
                rec_norm = float(np.linalg.norm(module.w_rec))
                if rec_norm > 50.0:
                    return False
            for layer in (getattr(module, "deep_w", ()) or ()):
                if not np.all(np.isfinite(layer)) or float(np.linalg.norm(layer)) > 120.0:
                    return False
            for layer in (getattr(module, "deep_rec", ()) or ()):
                if not np.all(np.isfinite(layer)) or float(np.linalg.norm(layer)) > 80.0:
                    return False
            for layer in (getattr(module, "deep_b", ()) or ()):
                if not np.all(np.isfinite(layer)):
                    return False
            if np.any(np.isnan(module.w_in)) or np.any(np.isinf(module.w_in)):
                return False
            if module.activation_ema > 5.0:
                return False

            m1 = getattr(module, "m1_component", None)
            if not isinstance(m1, M1GraphModule):
                continue
            try:
                m1.graph.assert_invariants(output_size=m1.output_size)
            except (TypeError, ValueError, KeyError):
                return False
            for node in m1.graph.neurons.values():
                values = (
                    node.bias, node.state, node.utility, node.plasticity,
                    node.eligibility, getattr(node, "forget_gate", 0.0),
                )
                if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                    return False
            for edge in m1.graph.edges.values():
                values = (
                    edge.weight, edge.gate, edge.utility, edge.eligibility,
                    edge.plasticity, getattr(edge, "short_term_state", 0.0),
                    getattr(edge, "consolidation", 0.0),
                )
                if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                    return False
        return True

    def expand_module(self, module_name: str, k_new: int, rng: np.random.Generator | None = None) -> bool:
        """Add hidden neurons without invalidating M1 or lateral topology."""
        if k_new <= 0:
            return False
        rng = rng or np.random.default_rng()
        target = next((module for module in self.modules if module.name == module_name), None)
        if target is None:
            return False
        old_h = int(target.hidden_size)
        hidden_cap = max(2, int(dict(getattr(self, "graph_supernetwork_config", {}) or {}).get("max_hidden_neurons_per_module", 128) or 128))
        k_new = min(int(k_new), max(0, hidden_cap - old_h))
        if k_new <= 0:
            return False
        new_h = old_h + int(k_new)
        utility = np.asarray(getattr(target, "neuron_utility", np.zeros(old_h)), dtype=float)
        samples = np.asarray(getattr(target, "neuron_samples", np.zeros(old_h)), dtype=float)
        eligible = np.where(samples >= 10.0)[0]
        ranked = eligible[np.argsort(utility[eligible])[::-1]] if eligible.size else np.array([], dtype=int)
        source_indices = [int(ranked[offset % ranked.size]) if ranked.size else int(rng.integers(old_h)) for offset in range(int(k_new))]

        new_w_in = np.zeros((new_h, target.input_size), dtype=float)
        new_w_in[:old_h] = target.w_in
        new_w_rec = np.zeros((new_h, new_h), dtype=float)
        new_w_rec[:old_h, :old_h] = target.w_rec
        new_b_hidden = np.zeros(new_h, dtype=float)
        new_b_hidden[:old_h] = target.b_hidden
        new_w_out = np.zeros((target.output_size, new_h), dtype=float)
        new_w_out[:, :old_h] = target.w_out
        new_stp = np.ones((new_h, target.input_size), dtype=float)
        if target.p_stp is not None:
            new_stp[:old_h] = target.p_stp
        new_utility = np.zeros(new_h, dtype=float)
        if target.neuron_utility is not None:
            new_utility[:old_h] = target.neuron_utility
        new_neuron_arrays: dict[str, np.ndarray] = {}
        for attr, dtype in (
            ("neuron_advantage_ema", float),
            ("neuron_contribution_ema", float),
            ("neuron_activation_ema", float),
            ("neuron_redundancy_ema", float),
            ("neuron_samples", float),
            ("neuron_frozen", bool),
        ):
            old_value = getattr(target, attr, None)
            arr = np.zeros(new_h, dtype=dtype)
            if old_value is not None and np.asarray(old_value).size == old_h:
                arr[:old_h] = np.asarray(old_value, dtype=dtype)
            new_neuron_arrays[attr] = arr
        new_eligibility = np.zeros((new_h, target.input_size), dtype=float)
        if target.input_eligibility is not None:
            new_eligibility[:old_h] = target.input_eligibility

        for offset, source_idx in enumerate(source_indices):
            row = old_h + offset
            new_w_in[row] = target.w_in[source_idx] + rng.normal(0.0, 0.01, target.input_size)
            new_w_rec[row, :old_h] = target.w_rec[source_idx, :]
            new_w_rec[:old_h, row] = target.w_rec[:, source_idx]
            new_b_hidden[row] = target.b_hidden[source_idx] + float(rng.normal(0.0, 0.01))
            new_w_out[:, row] = 0.5 * target.w_out[:, source_idx]
            if target.p_stp is not None:
                new_stp[row] = target.p_stp[source_idx]
            if target.neuron_utility is not None:
                new_utility[row] = target.neuron_utility[source_idx]
            for attr, arr in new_neuron_arrays.items():
                old_value = getattr(target, attr, None)
                if old_value is not None and np.asarray(old_value).size == old_h:
                    arr[row] = np.asarray(old_value)[source_idx]
            if target.input_eligibility is not None:
                new_eligibility[row] = target.input_eligibility[source_idx]

        target.w_in = new_w_in
        target.w_rec = new_w_rec
        target.b_hidden = new_b_hidden
        target.w_out = new_w_out
        target.p_stp = new_stp
        target.neuron_utility = new_utility
        for attr, arr in new_neuron_arrays.items():
            setattr(target, attr, arr)
        target.input_eligibility = new_eligibility
        target.hidden_size = new_h
        if target.w_lat is not None:
            target.w_lat = self._resize_matrix_shape(target.w_lat, new_h, new_h)
        target.deep_w = tuple(self._resize_matrix_shape(layer, new_h, new_h) for layer in getattr(target, "deep_w", ()) or ())
        target.deep_rec = tuple(self._resize_matrix_shape(layer, new_h, new_h) for layer in getattr(target, "deep_rec", ()) or ())
        target.deep_b = tuple(self._resize_vector(layer, new_h) for layer in getattr(target, "deep_b", ()) or ())
        target.deep_state = tuple(self._resize_vector(layer, new_h) for layer in getattr(target, "deep_state", ()) or ())
        if target.m1_component is not None:
            target.m1_component.resize_hidden(new_h, rng)
        self._runtime_memory[target.name] = self._resize_vector(
            self._runtime_memory.get(target.name, np.zeros(old_h, dtype=float)), new_h
        )
        self._ensure_lateral_edges()
        if hasattr(self, "world") and getattr(self, "world", None) is not None:
            self.world.ensure_for_modules(self.modules, rng)
        self.growth_events.append({
            "round": self.revision,
            "type": "expand_module",
            "module": module_name,
            "old_hidden": old_h,
            "new_hidden": new_h,
            "k_new": int(k_new),
        })
        return True

    def expand_module_depth(self, module_name: str, layers_new: int = 1, rng: np.random.Generator | None = None) -> bool:
        """Add hidden layers to a candidate module without changing bridge shape."""
        if layers_new <= 0:
            return False
        rng = rng or np.random.default_rng()
        target = next((module for module in self.modules if module.name == module_name), None)
        if target is None:
            return False
        config = dict(getattr(self, "graph_supernetwork_config", {}) or {})
        layer_cap = max(1, int(config.get("max_hidden_layers_per_module", 12) or 12))
        old_layers = 1 + len(getattr(target, "deep_w", ()) or ())
        add_count = min(int(layers_new), max(0, layer_cap - old_layers))
        if add_count <= 0:
            return False
        h = int(target.hidden_size)
        base_scale = 1.0 / max(1.0, np.sqrt(float(h)))
        deep_w = list(getattr(target, "deep_w", ()) or ())
        deep_rec = list(getattr(target, "deep_rec", ()) or ())
        deep_b = list(getattr(target, "deep_b", ()) or ())
        deep_state = list(getattr(target, "deep_state", ()) or ())
        for _ in range(add_count):
            identity = np.eye(h, dtype=float) * 0.85
            deep_w.append(identity + rng.normal(0.0, base_scale * 0.08, size=(h, h)))
            deep_rec.append(rng.normal(0.0, 0.01, size=(h, h)))
            deep_b.append(rng.normal(0.0, 0.02, size=h))
            deep_state.append(np.zeros(h, dtype=float))
        target.deep_w = tuple(deep_w)
        target.deep_rec = tuple(deep_rec)
        target.deep_b = tuple(deep_b)
        target.deep_state = tuple(deep_state)
        self.growth_events.append({
            "round": self.revision,
            "type": "expand_module_depth",
            "module": module_name,
            "old_hidden_layers": old_layers,
            "new_hidden_layers": 1 + len(target.deep_w),
            "layers_new": add_count,
        })
        return True

    def _maybe_expand_hidden_width(self, rng: np.random.Generator | None = None) -> dict[str, object]:
        signal = self.growth_signal()
        config = dict(getattr(self, "graph_supernetwork_config", {}) or {})
        threshold = float(config.get("hidden_depth_growth_threshold", 0.55) or 0.55)
        if float(signal.representation_need) < threshold:
            return {"expanded": 0, "representation_need": float(signal.representation_need), "threshold": threshold}
        cap = max(2, int(config.get("max_hidden_neurons_per_module", 512) or 512))
        expanded: list[dict[str, object]] = []
        candidates = sorted(
            self.modules,
            key=lambda module: (float(getattr(module, "plasticity", 0.0)), -float(getattr(module, "utility", 0.0))),
            reverse=True,
        )
        for module in candidates[: max(1, min(2, len(candidates)))]:
            room = max(0, cap - int(module.hidden_size))
            if room <= 0:
                continue
            k_new = max(1, min(room, max(4, int(np.ceil(module.hidden_size * 0.125)))))
            old_hidden = int(module.hidden_size)
            if self.expand_module(module.name, k_new, rng=rng):
                expanded.append({"module": module.name, "old_hidden": old_hidden, "new_hidden": int(module.hidden_size), "k_new": int(module.hidden_size) - old_hidden})
        return {
            "expanded": len(expanded),
            "modules": expanded,
            "representation_need": float(signal.representation_need),
            "threshold": threshold,
            "cap": cap,
        }

    def _maybe_expand_hidden_depth(self, rng: np.random.Generator | None = None) -> dict[str, object]:
        signal = self.growth_signal()
        config = dict(getattr(self, "graph_supernetwork_config", {}) or {})
        threshold = float(config.get("hidden_depth_growth_threshold", 0.55) or 0.55)
        if float(signal.representation_need) < threshold:
            return {"expanded": 0, "representation_need": float(signal.representation_need), "threshold": threshold}
        expanded: list[str] = []
        candidates = sorted(
            self.modules,
            key=lambda module: (float(getattr(module, "plasticity", 0.0)), -float(getattr(module, "utility", 0.0))),
            reverse=True,
        )
        for module in candidates[: max(1, min(2, len(candidates)))]:
            if self.expand_module_depth(module.name, 1, rng=rng):
                expanded.append(module.name)
        return {
            "expanded": len(expanded),
            "modules": expanded,
            "representation_need": float(signal.representation_need),
            "threshold": threshold,
        }

    def act(self, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
        self.ensure_compatible()
        if not self._uses_m1_graph():
            return self._act_direct_rnn(observation, goal)
        obs = np.asarray(observation, dtype=float)
        if obs.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(obs.size, self.input_size)] = obs[: self.input_size]
            obs = padded
        self._step_count += 1
        self._last_observation = obs.copy()
        gates = self._compute_gates(obs)

        m1_state = np.zeros(1)
        m1_memory_before = np.zeros(1)
        m1_memory_after = np.zeros(1)
        m1_selected_idx = 0
        m1_predicted_next = np.zeros(self.output_size)
        m1_critic_value = 0.0
        m1_confidence = 0.0
        m1_novelty = 0.0
        m1_conflict = 0.0
        m1_prediction_error = 0.0
        m1_candidate_actions = np.zeros((1, self.output_size))
        m1_candidate_scores = np.zeros(1, dtype=float)
        m1_candidate_logits = np.zeros(1, dtype=float)
        m1_selected_log_prob = 0.0
        m1_selected_action = np.zeros(self.output_size)
        graph_active_neuron_ids: tuple[str, ...] = ()
        graph_active_edge_ids: tuple[str, ...] = ()
        graph_edge_gates: dict[str, float] = {}
        graph_region_activity: dict[str, float] = {}
        graph_memory_match_quality = 0.0

        hidden_states = []
        raw_outputs = []
        for idx, module in enumerate(self.modules):
            previous_hidden = self._runtime_memory.get(module.name)
            gate, hidden, raw = module.activate(
                obs,
                previous_hidden,
                training=bool(getattr(self, "_learning_enabled", True)),
            )

            if module.m1_component is not None:
                m1 = module.m1_component
                module_obs = self._masked_module_observation(module, obs)
                if isinstance(m1, M1GraphModule):
                    m1_context = dict(self._query_memory(module_obs))
                    account = dict(getattr(self, "_account_context", {}) or {})
                    m1_context["account_stress"] = float(np.clip(max(
                        float(account.get("current_equity_drawdown", 0.0) or 0.0),
                        float(account.get("open_risk_ratio", 0.0) or 0.0),
                        float(account.get("equity_volatility", 0.0) or 0.0),
                    ), 0.0, 1.0))
                    m1_context["drawdown"] = float(np.clip(account.get("current_equity_drawdown", 0.0) or 0.0, 0.0, 1.0))
                    m1_context["open_risk_ratio"] = float(np.clip(account.get("open_risk_ratio", 0.0) or 0.0, 0.0, 1.0))
                    # The graph needs to see *current opportunity scarcity*
                    # before it chooses an action.  Previously this signal
                    # reached only the external bootstrap after M1 had already
                    # emitted HOLD/CLOSE, so all training entries were forced
                    # bridge actions instead of policy decisions.
                    starvation = dict(getattr(self, "_experience_starvation", {}) or {})
                    no_trade_streak = float(max(0.0, starvation.get("no_trade_streak", 0.0) or 0.0))
                    cadence_deficit = float(np.clip(starvation.get("cadence_deficit", 0.0) or 0.0, 0.0, 1.0))
                    runtime_starvation = float(np.clip(
                        starvation.get("runtime_starvation", starvation.get("starvation", 0.0)) or 0.0,
                        0.0,
                        3.0,
                    ))
                    # A streak of roughly six decision ticks is sufficient to
                    # activate a *small neural probe*.  This is not an economic
                    # reward, and account stress below remains the hard safety
                    # override via M1's entry gate.
                    entry_pressure = float(np.clip(max(
                        runtime_starvation / 1.5,
                        cadence_deficit,
                        no_trade_streak / 6.0,
                    ), 0.0, 1.0))
                    m1_context["runtime_starvation"] = runtime_starvation
                    m1_context["no_trade_streak"] = no_trade_streak
                    m1_context["cadence_deficit"] = cadence_deficit
                    # Scarcity is a learning-drive signal, not market evidence.
                    # It may request representation/exploration work, but it may
                    # not manufacture BUY/SELL opportunity or overwrite the
                    # uncertainty estimated from world/memory context.
                    m1_context["policy_entry_pressure"] = entry_pressure
                    m1_context["learning_drive"] = np.asarray([
                        min(1.0, no_trade_streak / 12.0),
                        min(1.0, cadence_deficit),
                        min(1.0, runtime_starvation / 3.0),
                    ], dtype=float)
                    m1_context["market_opportunity"] = float(np.clip(
                        m1_context.get("market_opportunity", 0.0) or 0.0,
                        0.0,
                        1.0,
                    ))
                    m1_context["uncertainty"] = float(np.clip(
                        float(m1_context.get("uncertainty", 0.0) or 0.0),
                        0.0,
                        1.0,
                    ))
                    # Capture graph memory state around activation.  This is
                    # the graph working-memory path and keeps
                    # StepTrace/replay semantics meaningful for memory-aware
                    # policy analysis.
                    memory_ids = tuple(m1.graph.regions.get("memory").neuron_ids)
                    encoder_ids = tuple(m1.graph.regions.get("encoder").neuron_ids)
                    m1_memory_before = self._resize_vector(
                        np.asarray([m1.graph.neurons[node_id].state for node_id in memory_ids], dtype=float),
                        self.output_size,
                    )
                    m1_output, m1_err, m1_nov, m1_con, m1_conf = m1.forward(module_obs, goal, context=m1_context)
                    m1_memory_after = self._resize_vector(
                        np.asarray([m1.graph.neurons[node_id].state for node_id in memory_ids], dtype=float),
                        self.output_size,
                    )
                    m1_state = self._resize_vector(
                        np.asarray([m1.graph.neurons[node_id].state for node_id in encoder_ids], dtype=float),
                        self.output_size,
                    )
                    # The graph path queries memory after encoded state
                    # was available.  Preserve that causal ordering in graph
                    # mode: a context match discovered from the encoded state
                    # becomes a small bounded feedback signal for memory and
                    # the candidate action score, never a replacement output.
                    encoded_memory = self._query_memory(m1_state)
                    encoded_trust = float(np.clip(max(
                        encoded_memory.get("trust", 0.0),
                        encoded_memory.get("match_strength", 0.0),
                    ), 0.0, 1.0)) if isinstance(encoded_memory, dict) else 0.0
                    if encoded_trust > 0.0:
                        memory_vector = np.asarray(encoded_memory.get("memory_context", []), dtype=float).reshape(-1)
                        memory_vector = self._resize_vector(memory_vector, len(memory_ids))
                        for mem_index, node_id in enumerate(memory_ids):
                            node = m1.graph.neurons[node_id]
                            node.state = float(np.clip(node.state + (0.05 + 0.15 * encoded_trust) * memory_vector[mem_index], -1.0, 1.0))
                        encoded_bias = self._resize_vector(np.asarray(encoded_memory.get("action_bias", []), dtype=float), m1_output.size)
                        if encoded_bias.size == m1_output.size:
                            bounded_bias = np.clip(encoded_bias, -1.0, 1.0)
                            # Re-score concrete graph hypotheses after the
                            # encoded-state memory query.  This preserves
                            # M1 where causal memory affects selector scores,
                            # not only the final action vector.
                            candidate_actions = np.asarray(getattr(m1, "last_candidate_actions", []), dtype=float)
                            candidate_scores = np.asarray(getattr(m1, "last_candidate_scores", []), dtype=float)
                            if candidate_actions.ndim == 2 and candidate_actions.shape[1] == bounded_bias.size and candidate_scores.size == candidate_actions.shape[0]:
                                candidate_scores = candidate_scores + 0.10 * encoded_trust * (candidate_actions @ bounded_bias)
                                selected = int(np.argmax(candidate_scores))
                                probs = np.exp(candidate_scores - np.max(candidate_scores))
                                probs = probs / (float(np.sum(probs)) + 1e-12)
                                m1.last_candidate_scores = candidate_scores
                                m1.last_candidate_logits = candidate_scores.copy()
                                m1.last_selected_idx = selected
                                m1.last_selected_log_prob = float(np.log(max(float(probs[selected]), 1e-12)))
                                m1.last_critic_value = float(candidate_scores[selected])
                                m1_output = np.tanh(candidate_actions[selected] + 0.20 * encoded_trust * bounded_bias)
                            else:
                                m1_output = np.tanh(m1_output + 0.15 * encoded_trust * bounded_bias)
                        m1_memory_after = self._resize_vector(
                            np.asarray([m1.graph.neurons[node_id].state for node_id in memory_ids], dtype=float),
                            self.output_size,
                        )
                    graph_trace = m1.last_trace
                    if graph_trace is not None:
                        graph_active_neuron_ids = tuple(graph_trace.active_neuron_ids)
                        graph_active_edge_ids = tuple(graph_trace.active_edge_ids)
                        graph_edge_gates = dict(graph_trace.edge_gates)
                        graph_region_activity = dict(graph_trace.region_activity)
                    graph_memory_match_quality = max(
                        float(np.clip(max(
                            m1_context.get("trust", 0.0),
                            m1_context.get("match_strength", 0.0),
                        ), 0.0, 1.0)) if isinstance(m1_context, dict) else 0.0,
                        encoded_trust,
                    )
                    m1_prediction_error = m1_err
                    m1_novelty = m1_nov
                    m1_conflict = m1_con
                    m1_confidence = m1_conf
                    if m1_output.size == raw.size:
                        raw = m1_output * np.clip(module.output_mask, 0.0, 1.0)
                    # M1GraphModule keeps the same public return tuple as
                    # the former monolithic M1, while exposing hypothesis/critic evidence
                    # through immutable runtime fields.  Preserve it in the
                    # StepTrace so policy-gradient credit targets the action
                    # actually selected rather than a synthetic one-candidate
                    # wrapper.
                    m1_selected_idx = int(getattr(m1, "last_selected_idx", 0))
                    m1_candidate_actions = np.asarray(
                        getattr(m1, "last_candidate_actions", np.asarray([m1_output], dtype=float)),
                        dtype=float,
                    ).copy()
                    m1_candidate_scores = np.asarray(
                        getattr(m1, "last_candidate_scores", np.asarray([m1_conf], dtype=float)),
                        dtype=float,
                    ).copy()
                    m1_candidate_logits = np.asarray(
                        getattr(m1, "last_candidate_logits", m1_candidate_scores),
                        dtype=float,
                    ).copy()
                    m1_predicted_next = self._resize_vector(
                        np.asarray(getattr(m1, "last_predicted_state", m1_output), dtype=float),
                        self.output_size,
                    )
                    m1_critic_value = float(getattr(m1, "last_critic_value", m1_conf))
                    m1_selected_log_prob = float(getattr(m1, "last_selected_log_prob", 0.0))
                    m1_selected_action = m1_output.copy()
                else:
                    m1_memory_before = m1.working_memory.state.copy()
                    prelim_state = m1.state_encoder.encode(module_obs)
                    m1_context = self._query_memory(prelim_state)
                    m1_output, m1_err, m1_nov, m1_con, m1_conf = m1.forward(module_obs, goal, context=m1_context)
                    m1_memory_after = m1.working_memory.state.copy()
                    m1_state = m1.state_encoder.encode(module_obs)
                    m1_prediction_error = m1_err
                    m1_novelty = m1_nov
                    m1_conflict = m1_con
                    m1_confidence = m1_conf
                    if m1_output.size == raw.size:
                        raw = m1_output * np.clip(module.output_mask, 0.0, 1.0)
                    scores = np.zeros(m1.K, dtype=float)
                    predictions = []
                    for k in range(m1.K):
                        action_vec = np.zeros(m1.K, dtype=float)
                        action_vec[k] = 1.0
                        predicted = m1.predictor.predict(m1_state, m1_memory_after, action_vec)
                        predictions.append(predicted)
                        g = goal if goal is not None else np.zeros(m1.critic.w_goal.shape[1])
                        scores[k] = m1.critic.evaluate(predicted, g)
                        memory_context = np.asarray(m1_context.get("memory_context", []), dtype=float).reshape(-1) if isinstance(m1_context, dict) else np.zeros(0)
                        trust = float(m1_context.get("trust", 0.0)) if isinstance(m1_context, dict) else 0.0
                        if memory_context.size:
                            scores[k] += 0.03 * trust * float(predicted @ memory_context[:predicted.size])
                    action_bias_for_scores = np.asarray(m1_context.get("replacement_action_bias", m1_context.get("success_action_bias", np.maximum(0.0, np.asarray(m1_context.get("action_bias", []), dtype=float)))), dtype=float).reshape(-1) if isinstance(m1_context, dict) else np.zeros(0)
                    hyp_acts_for_scores, _ = m1.hypothesis_bank.generate(m1_state, m1_memory_after)
                    if action_bias_for_scores.size == self.output_size:
                        for k in range(m1.K):
                            mask = np.zeros(m1.K)
                            mask[k] = 1.0
                            candidate_action = m1.hypothesis_bank.w_decode @ (hyp_acts_for_scores * mask) + m1.hypothesis_bank.b_decode
                            scores[k] += 0.08 * max(0.0, float(candidate_action @ action_bias_for_scores))
                    sel_idx, z = m1.selector.select(scores)
                    m1_selected_idx = sel_idx
                    m1_candidate_scores = scores.copy()
                    m1_candidate_logits = scores.copy()
                    m1_predicted_next = predictions[sel_idx]
                    m1_critic_value = float(scores[sel_idx])
                    exp_s = np.exp(scores - np.max(scores))
                    p = exp_s / (np.sum(exp_s) + 1e-10)
                    m1_selected_log_prob = float(np.log(max(p[sel_idx], 1e-12)))
                    m1_selected_action = m1.hypothesis_bank.w_decode @ (
                        m1.hypothesis_bank.generate(m1_state, m1_memory_after)[0] * np.eye(m1.K)[sel_idx]
                    ) + m1.hypothesis_bank.b_decode
                    m1_candidate_actions = np.zeros((m1.K, self.output_size))
                    hyp_acts, _ = m1.hypothesis_bank.generate(m1_state, m1_memory_after)
                    for k in range(m1.K):
                        mask = np.zeros(m1.K)
                        mask[k] = 1.0
                        m1_candidate_actions[k] = m1.hypothesis_bank.w_decode @ (hyp_acts * mask) + m1.hypothesis_bank.b_decode

            hidden_states.append(hidden)
            raw_outputs.append(raw)

        hidden_states, raw_outputs = self._integrate_lateral(hidden_states, raw_outputs)

        gated_outputs = []
        for idx, module in enumerate(self.modules):
            gate = gates[idx]
            hidden = hidden_states[idx]
            self._runtime_memory[module.name] = hidden
            module.observe_usage(gate, hidden)
            gated_outputs.append(gate * raw_outputs[idx])

        gate_sum = max(float(np.sum(gates)), 1e-9)
        output = np.sum(gated_outputs, axis=0) / gate_sum
        memory_query = self._query_memory(m1_state)
        raw_action_bias = np.asarray(memory_query.get("action_bias", np.zeros(self.output_size)), dtype=float)
        action_bias = np.asarray(memory_query.get("replacement_action_bias", memory_query.get("success_action_bias", np.maximum(0.0, raw_action_bias))), dtype=float)
        curiosity_context = self.curiosity_system.context_from(
            state=m1_state,
            action=output,
            memory_query=memory_query,
            causal_signals=self.memory_supernet.causal_plasticity_signals(max_items=4) if hasattr(self, "memory_supernet") else [],
            starvation=self._experience_starvation,
        )
        curiosity_context = self._reasoning_context("curiosity", curiosity_context, self.curiosity_system.input_size)
        curiosity = self.curiosity_system.forward(curiosity_context)
        curiosity_action = self._resize_vector(np.asarray(curiosity.get("action_delta", []), dtype=float), self.output_size)
        if action_bias.size == output.size:
            output = output + action_bias
            if m1_selected_action.size == action_bias.size:
                m1_selected_action = np.tanh(m1_selected_action + action_bias)
                m1_candidate_actions = np.tanh(m1_candidate_actions + action_bias.reshape(1, -1))
        # V8: even a high-novelty M1 state is not a trading signal.  Curiosity
        # can request shadow/replay work only; all of its live action deltas
        # are zeroed here. Verified contextual expectancy is handled by the
        # M1 evidence path independently of this curiosity system.
        curiosity_action = np.zeros_like(curiosity_action)
        curiosity_gain = 0.0
        market_opportunity = float(np.clip(
            memory_query.get("market_opportunity", 0.0) if isinstance(memory_query, dict) else 0.0,
            0.0,
            1.0,
        ))
        exploration_floor = 0.0
        self._v8_curiosity_diagnostic = {
            "mode": str(getattr(self, "live_curiosity_mode", "shadow_only") or "shadow_only"),
            "raw_delta_norm": float(np.linalg.norm(np.asarray(curiosity.get("action_delta", []), dtype=float))),
            "live_action_delta_norm": 0.0,
            "shadow_replay_requested": bool(float(curiosity.get("novelty", 0.0) or 0.0) > 0.0),
        }
        output, risk_info = self._apply_biological_action_heads(output, m1_state, obs, gates, memory_query)
        # V8 context-owner action contract:
        #
        # `m1_selected_idx` selects a concrete hypothesis.  Earlier versions
        # accidentally overwrote its semantic action head with the aggregate
        # multi-module output below, so thinking could select BUY_STOP/CLOSE_ALL
        # while the bridge decoded MARKET_SELL.  Modules may still shape the
        # *parameter* heads (risk/SL/TP/trailing), but the discrete action must
        # come from the exact candidate selected by the M1 selector.
        aggregate_parameter_output = np.tanh(output)
        parameter_offset = PARAMETER_OFFSET if aggregate_parameter_output.size >= OUTPUT_COUNT else 11
        param_count = min(8, max(0, aggregate_parameter_output.size - parameter_offset))
        candidates_valid = (
            m1_candidate_actions.ndim == 2
            and m1_candidate_actions.shape[0] > 0
            and m1_candidate_actions.shape[1] == aggregate_parameter_output.size
        )
        if candidates_valid:
            candidate_index = int(np.clip(m1_selected_idx, 0, m1_candidate_actions.shape[0] - 1))
            candidate_output = np.asarray(m1_candidate_actions[candidate_index], dtype=float).reshape(-1).copy()
            # Candidate action logits can be raw in legacy backends or bounded
            # in graph mode.  `tanh` is idempotent enough for bounded values and
            # keeps the bridge contract in [-1, 1].
            candidate_output[:min(ACTION_COUNT, candidate_output.size)] = np.tanh(
                candidate_output[:min(ACTION_COUNT, candidate_output.size)]
            )
            if param_count > 0:
                candidate_output[parameter_offset:parameter_offset + param_count] = (
                    aggregate_parameter_output[parameter_offset:parameter_offset + param_count]
                )
                # Explanatory candidate rows must carry the same decoded
                # parameters as the final action, while retaining their own
                # discrete action proposal.
                m1_candidate_actions[:, parameter_offset:parameter_offset + param_count] = (
                    aggregate_parameter_output[parameter_offset:parameter_offset + param_count].reshape(1, -1)
                )
            output = candidate_output
        else:
            output = aggregate_parameter_output
        if m1_selected_action.size == output.size:
            m1_selected_action = output.copy()

        module_action_traces = tuple(
            ModuleActionTrace(
                module_name=module.name,
                gate=float(gates[idx]) if idx < len(gates) else 0.0,
                hidden=np.asarray(
                    getattr(module, "last_output_hidden", hidden_states[idx]),
                    dtype=float,
                ).copy(),
                raw_output=np.asarray(raw_outputs[idx], dtype=float).copy(),
            )
            for idx, module in enumerate(self.modules)
        )

        selected_action_index = int(np.argmax(output[:min(ACTION_COUNT, output.size)])) if output.size else 0
        replacement_scope = _m1v11_scope_from_output(self, output)
        action_details = {
            "action_name": str(replacement_scope.get("action_name", "")),
            "model_hypothesis_index": int(m1_selected_idx),
            "selected_candidate_action_index": int(selected_action_index),
            "policy_output_action_index": int(selected_action_index),
            "semantic_action": int(selected_action_index),
            "semantic_action_source": "selected_hypothesis_candidate",
            "final_action_output": output.astype(float).tolist(),
            "parameter_replacement_proposal_ids": list(risk_info.get("parameter_replacement_proposal_ids", []) or []),
            "parameter_replacement_mode": str(risk_info.get("parameter_replacement_mode", "none")),
            "parameter_replacement_scope": dict(risk_info.get("parameter_replacement_scope", replacement_scope) or replacement_scope),
            # Live/continuous diagnostics: distinguish a policy HOLD from a
            # HOLD created by the risk/cost gate.  Keep this compact and scalar
            # so it can be serialized to JSONL without leaking arrays.
            "risk_gate": {
                str(key): float(value)
                for key, value in dict(risk_info or {}).items()
                if isinstance(value, (int, float, np.integer, np.floating))
            },
            "market_opportunity": float(market_opportunity),
            "exploration_floor": float(exploration_floor),
        }

        return StepTrace(
            trace_id=f"trace-{self.revision}-{self._step_count}-{time.time_ns()}",
            timestamp=time.time(),
            raw_obs=obs.copy(),
            encoded_state=m1_state.copy(),
            memory_before=m1_memory_before.copy(),
            memory_after=m1_memory_after.copy(),
            selected_idx=m1_selected_idx,
            selected_action=m1_selected_action.copy(),
            candidate_actions=m1_candidate_actions.copy(),
            candidate_logits=m1_candidate_logits.copy(),
            candidate_scores=m1_candidate_scores.copy(),
            selected_action_log_prob=m1_selected_log_prob,
            predicted_next_state=m1_predicted_next.copy(),
            critic_value=m1_critic_value,
            confidence=m1_confidence,
            novelty=m1_novelty,
            conflict=m1_conflict,
            risk=float(0.5 * risk_info.get("risk", 0.0) + 0.5 * risk_info.get("cost", 0.0)),
            action_details=action_details,
            positions_pipe=str(getattr(self, "_positions_pipe_context", "") or ""),
            gate_values=tuple(float(g) for g in gates),
            prediction_error=m1_prediction_error,
            td_delta=0.0,
            account_context=dict(getattr(self, "_account_context", {}) or {}),
            experience_starvation=dict(self._experience_starvation),
            module_action_traces=module_action_traces,
            active_neuron_ids=graph_active_neuron_ids,
            active_edge_ids=graph_active_edge_ids,
            graph_edge_gates=graph_edge_gates,
            graph_region_activity=graph_region_activity,
            memory_match_quality=graph_memory_match_quality,
        )

    def _act_direct_rnn(self, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
        obs = np.asarray(observation, dtype=float).reshape(-1)
        if obs.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(obs.size, self.input_size)] = obs[: self.input_size]
            obs = padded
        self._step_count += 1
        self._last_observation = obs.copy()
        gates = self._compute_gates(obs)

        hidden_states: list[np.ndarray] = []
        raw_outputs: list[np.ndarray] = []
        for module in self.modules:
            previous_hidden = self._runtime_memory.get(module.name)
            _gate, hidden, raw = module.activate(
                obs,
                previous_hidden,
                training=bool(getattr(self, "_learning_enabled", True)),
            )
            hidden_states.append(hidden)
            raw_outputs.append(raw)

        hidden_states, raw_outputs = self._integrate_lateral(hidden_states, raw_outputs)

        gated_outputs: list[np.ndarray] = []
        for idx, module in enumerate(self.modules):
            gate = float(gates[idx]) if idx < len(gates) else 0.0
            hidden = hidden_states[idx]
            self._runtime_memory[module.name] = hidden
            module.observe_usage(gate, hidden)
            gated_outputs.append(gate * raw_outputs[idx])

        gate_sum = max(float(np.sum(gates)), 1e-9)
        output = np.sum(gated_outputs, axis=0) / gate_sum if gated_outputs else np.zeros(self.output_size, dtype=float)
        encoded_state = self._resize_vector(
            np.mean(np.vstack([self._resize_vector(state, self.output_size) for state in hidden_states]), axis=0)
            if hidden_states else np.zeros(self.output_size, dtype=float),
            self.output_size,
        )
        memory_query = self._query_memory(obs)
        raw_action_bias = np.asarray(memory_query.get("action_bias", np.zeros(self.output_size)), dtype=float)
        action_bias = np.asarray(memory_query.get("replacement_action_bias", memory_query.get("success_action_bias", np.maximum(0.0, raw_action_bias))), dtype=float)
        if action_bias.size == output.size:
            parameter_offset_for_bias = PARAMETER_OFFSET if output.size >= OUTPUT_COUNT else min(ACTION_COUNT, output.size)
            if parameter_offset_for_bias < output.size:
                output[parameter_offset_for_bias:] += action_bias[parameter_offset_for_bias:]
        curiosity_context = self.curiosity_system.context_from(
            state=obs,
            action=output,
            memory_query=memory_query,
            causal_signals=self.memory_supernet.causal_plasticity_signals(max_items=4) if hasattr(self, "memory_supernet") else [],
            starvation=self._experience_starvation,
        )
        curiosity_context = self._reasoning_context("curiosity", curiosity_context, self.curiosity_system.input_size)
        curiosity = self.curiosity_system.forward(curiosity_context)
        self._v8_curiosity_diagnostic = {
            "mode": str(getattr(self, "live_curiosity_mode", "shadow_only") or "shadow_only"),
            "raw_delta_norm": float(np.linalg.norm(np.asarray(curiosity.get("action_delta", []), dtype=float))),
            "live_action_delta_norm": 0.0,
            "shadow_replay_requested": bool(float(curiosity.get("novelty", 0.0) or 0.0) > 0.0),
        }
        output, risk_info = self._apply_biological_action_heads(output, obs, obs, gates, memory_query)
        output = np.tanh(output)
        action_count = min(ACTION_COUNT, output.size)
        raw_allowed_actions = dict(getattr(self, "_decision_context", {}) or {}).get("allowed_actions", None)
        if raw_allowed_actions is not None and action_count:
            context_action_mask = np.ones(action_count, dtype=bool)
            context_action_mask[:] = False
            if isinstance(raw_allowed_actions, Mapping):
                for raw_index, is_allowed in raw_allowed_actions.items():
                    try:
                        action_index = int(raw_index)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= action_index < action_count and bool(is_allowed):
                        context_action_mask[action_index] = True
            else:
                try:
                    raw_mask = np.asarray(raw_allowed_actions, dtype=bool).reshape(-1)
                except (TypeError, ValueError):
                    raw_mask = np.zeros(0, dtype=bool)
                if raw_mask.size:
                    context_action_mask[:min(action_count, raw_mask.size)] = raw_mask[:action_count]
            if not np.any(context_action_mask):
                context_action_mask[0] = True
            output[:action_count] = np.where(context_action_mask, output[:action_count], -1.0)

        module_action_traces = tuple(
            ModuleActionTrace(
                module_name=module.name,
                gate=float(gates[idx]) if idx < len(gates) else 0.0,
                hidden=np.asarray(
                    getattr(module, "last_output_hidden", hidden_states[idx]),
                    dtype=float,
                ).copy(),
                raw_output=np.asarray(raw_outputs[idx], dtype=float).copy(),
            )
            for idx, module in enumerate(self.modules)
        )
        selected_action_index = int(np.argmax(output[:action_count])) if action_count else 0
        replacement_scope = _m1v11_scope_from_output(self, output)
        action_details = {
            "action_name": str(replacement_scope.get("action_name", "")),
            "model_hypothesis_index": 0,
            "selected_candidate_action_index": int(selected_action_index),
            "selected_hypothesis_action_index": int(selected_action_index),
            "policy_output_action_index": int(selected_action_index),
            "semantic_action": int(selected_action_index),
            "semantic_action_source": "direct_rnn",
            "semantic_matches_selected_hypothesis": True,
            "decision_backend": "direct_rnn",
            "final_action_output": output.astype(float).tolist(),
            "parameter_replacement_proposal_ids": list(risk_info.get("parameter_replacement_proposal_ids", []) or []),
            "parameter_replacement_mode": str(risk_info.get("parameter_replacement_mode", "none")),
            "parameter_replacement_scope": dict(risk_info.get("parameter_replacement_scope", replacement_scope) or replacement_scope),
            "risk_gate": {
                str(key): float(value)
                for key, value in dict(risk_info or {}).items()
                if isinstance(value, (int, float, np.integer, np.floating))
            },
        }
        candidates = output.reshape(1, -1)
        score = float(output[selected_action_index]) if output.size else 0.0
        return StepTrace(
            trace_id=f"trace-{self.revision}-{self._step_count}-{time.time_ns()}",
            timestamp=time.time(),
            raw_obs=obs.copy(),
            encoded_state=encoded_state.copy(),
            memory_before=np.zeros(self.output_size, dtype=float),
            memory_after=encoded_state.copy(),
            selected_idx=0,
            selected_action=output.copy(),
            candidate_actions=candidates.copy(),
            candidate_logits=np.asarray([score], dtype=float),
            candidate_scores=np.asarray([score], dtype=float),
            selected_action_log_prob=0.0,
            predicted_next_state=encoded_state.copy(),
            critic_value=score,
            confidence=float(np.clip(abs(score), 0.0, 1.0)),
            novelty=0.0,
            conflict=0.0,
            risk=float(0.5 * risk_info.get("risk", 0.0) + 0.5 * risk_info.get("cost", 0.0)),
            action_details=action_details,
            positions_pipe=str(getattr(self, "_positions_pipe_context", "") or ""),
            gate_values=tuple(float(g) for g in gates),
            prediction_error=0.0,
            td_delta=0.0,
            account_context=dict(getattr(self, "_account_context", {}) or {}),
            experience_starvation=dict(self._experience_starvation),
            module_action_traces=module_action_traces,
            active_neuron_ids=(),
            active_edge_ids=(),
            graph_edge_gates={},
            graph_region_activity={},
            memory_match_quality=0.0,
        )

    @staticmethod
    def _memory_eligible_for_transition(trace: StepTrace) -> bool:
        """Allow memory only for a filled/closed action with bridge evidence."""
        execution_details = dict(getattr(trace, "execution_details", {}) or {})
        if not bool(execution_details.get("bridge_observed", False)):
            return True
        return bool(execution_details.get("memory_eligible", False))

    @staticmethod
    def _action_feedback_eligible_for_transition(trace: StepTrace) -> bool:
        """Allow immediate TD/plasticity only for realised execution outcomes.

        A fill is cached under its trace id and receives final lifecycle credit
        later. It must not receive a synthetic interim reward merely because an
        order was accepted or a position briefly appeared.
        """
        execution_details = dict(getattr(trace, "execution_details", {}) or {})
        if not bool(execution_details.get("bridge_observed", False)):
            return True
        if "action_feedback_eligible" in execution_details:
            return bool(execution_details.get("action_feedback_eligible", False))
        # Compatibility for v4 traces produced before this distinction.
        return bool(execution_details.get("memory_eligible", False))

    def observe_transition(
        self,
        trace: StepTrace,
        next_obs: np.ndarray,
        reward: float,
        cost: float,
        done: bool,
        gamma: float = 0.95,
        record_experience: bool = True,
    ) -> dict[str, float]:
        next_obs_arr = np.asarray(next_obs, dtype=float)
        if next_obs_arr.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(next_obs_arr.size, self.input_size)] = next_obs_arr[: self.input_size]
            next_obs_arr = padded
        total_pred_error = 0.0
        total_td_delta = 0.0
        memory_eligible = bool(record_experience and self._memory_eligible_for_transition(trace))
        # Passive bridge requests update state prediction/working memory only;
        # they must not receive TD credit for an action the EA never executed.
        action_feedback_eligible = self._action_feedback_eligible_for_transition(trace)

        for module in self.modules:
            if module.m1_component is not None:
                m1 = module.m1_component
                if isinstance(m1, M1GraphModule):
                    # Graph M1 has no matrix-only state_encoder/predictor/critic
                    # members.  Attribution uses the immutable decision trace
                    # and updates a candidate graph only for destructive paths.
                    actual_next_state = np.tanh(self._resize_vector(next_obs_arr, self.output_size))
                    predicted = self._resize_vector(trace.predicted_next_state, self.output_size)
                    pred_error = float(np.linalg.norm(actual_next_state - predicted))
                    total_pred_error = pred_error
                    raw_td_delta = float(reward - cost - trace.critic_value)
                    td_delta = raw_td_delta if action_feedback_eligible else 0.0
                    total_td_delta = td_delta
                    account_context = dict(getattr(trace, "account_context", {}) or {})
                    stress = float(np.clip(
                        float(account_context.get("current_equity_drawdown", 0.0) or 0.0)
                        + 0.5 * float(account_context.get("open_risk_ratio", 0.0) or 0.0)
                        + 0.25 * float(account_context.get("equity_volatility_20", 0.0) or 0.0),
                        0.0,
                        1.0,
                    ))
                    lifecycle_tags = tuple(dict.fromkeys(
                        list(getattr(trace, "lifecycle_tags", ()) or ())
                        + [str(tag) for tag in dict(getattr(trace, "execution_details", {}) or {}).get("lifecycle_tags", []) or []]
                    ))
                    if action_feedback_eligible:
                        graph_report = GraphPlasticityEngine().apply_credit(
                            m1.graph,
                            GraphCreditSignal(
                                reward=float(reward),
                                cost=float(cost),
                                td_delta=float(td_delta),
                                prediction_error=pred_error,
                                account_stress=stress,
                                lifecycle_tags=lifecycle_tags,
                                memory_match_quality=float(getattr(trace, "memory_match_quality", 0.0)),
                            ),
                            active_edge_ids=getattr(trace, "active_edge_ids", ()),
                            active_neuron_ids=getattr(trace, "active_neuron_ids", ()),
                            candidate_mode=bool(getattr(module, "lifecycle", "") == MODULE_LIFECYCLE_CANDIDATE),
                            update_mode="candidate" if getattr(module, "lifecycle", "") == MODULE_LIFECYCLE_CANDIDATE else "baseline",
                        )
                    else:
                        # Filled-but-open positions are recorded by trace id for
                        # final lifecycle credit; they do not mutate utilities,
                        # gates or weights on an interim prediction error.
                        graph_report = GraphPlasticityReport(credit=0.0, update_mode="deferred_lifecycle")
                    # The graph may grow during a training trial, but only on
                    # the copied candidate policy and only after a realised
                    # outcome. Baselines, order placement and open fills do
                    # not receive structural mutations here.
                    micro_growth_events: list[dict[str, object]] = []
                    if (
                        action_feedback_eligible
                        and record_experience
                        and bool(getattr(self, "graph_trial_mode", False))
                        and bool(getattr(self, "graph_trial_id", ""))
                        and self.graph_supernetwork is not None
                    ):
                        micro_growth_events = self.graph_supernetwork.learn_in_trial(
                            m1.graph,
                            trial_id=str(self.graph_trial_id),
                            trace=trace,
                            learning_signal={
                                "reward": float(reward),
                                "cost": float(cost),
                                "td_delta": float(td_delta),
                                "prediction_error": float(pred_error),
                            },
                            account_context=account_context,
                        )
                        for event in micro_growth_events:
                            self.growth_events.append({
                                "round": self.revision,
                                "type": "graph_trial_micro_growth",
                                "trace_id": trace.trace_id,
                                **event,
                            })
                    if action_feedback_eligible:
                        m1.memory_bank.append(actual_next_state.copy())
                        if m1.memory_bank_max is not None and len(m1.memory_bank) > m1.memory_bank_max:
                            m1.memory_bank = m1.memory_bank[-m1.memory_bank_max:]
                    self.growth_events.append({
                        "round": self.revision,
                        "type": "graph_credit_attribution",
                        "trace_id": trace.trace_id,
                        "updated_edges": graph_report.updated_edges,
                        "updated_neurons": graph_report.updated_neurons,
                        "disabled_edges": graph_report.disabled_edges,
                        "micro_growth_events": len(micro_growth_events),
                        "module_need_reason": getattr(trace, "module_need_reason", ""),
                    })
                    continue
                actual_next_state = m1.state_encoder.encode(next_obs_arr)
                pred_error = float(np.linalg.norm(actual_next_state - trace.predicted_next_state[:actual_next_state.size]))
                total_pred_error = pred_error

                g = np.zeros(m1.critic.w_goal.shape[1])
                next_value = float(m1.critic.evaluate(actual_next_state, g))
                td_target = reward - cost + gamma * (1.0 - float(done)) * next_value
                raw_td_delta = td_target - trace.critic_value
                td_delta = raw_td_delta if action_feedback_eligible else 0.0
                total_td_delta = td_delta
                advantage = np.clip(td_delta, -2.0, 2.0)

                pred_lr = 0.01 * min(1.0, pred_error)
                pred_grad = actual_next_state - trace.predicted_next_state[:actual_next_state.size]
                if pred_grad.size > 0:
                    m1.predictor.w_proj += pred_lr * np.outer(
                        pred_grad[:m1.predictor.w_proj.shape[0]],
                        trace.encoded_state[:m1.predictor.w_proj.shape[1]],
                    )

                if action_feedback_eligible:
                    crit_lr = 0.005 * advantage
                    m1.critic.w_score += crit_lr * trace.encoded_state[:m1.critic.w_score.size]

                    policy_lr = 0.002 * advantage
                    sel = trace.selected_idx
                    if sel < m1.K:
                        hyp_acts = np.zeros(m1.K, dtype=float)
                        hyp_acts[sel] = 1.0
                        m1.hypothesis_bank.w_gen += policy_lr * np.outer(
                            hyp_acts,
                            np.pad(trace.encoded_state, (0, max(0, m1.hypothesis_bank.w_gen.shape[1] - trace.encoded_state.size)))[:m1.hypothesis_bank.w_gen.shape[1]],
                        )
                        m1.hypothesis_bank.w_decode += policy_lr * np.outer(
                            np.pad(trace.selected_action, (0, max(0, m1.hypothesis_bank.w_decode.shape[0] - trace.selected_action.size)))[:m1.hypothesis_bank.w_decode.shape[0]],
                            hyp_acts,
                        )
                        m1.selector.update_lateral(hyp_acts, lr=0.001 * abs(advantage))

                trace_memory_after = self._resize_vector(trace.memory_after, m1.working_memory.state.size)
                m1.working_memory.state = gamma * m1.working_memory.state + (1 - gamma) * trace_memory_after

                if hasattr(m1, 'memory_bank') and len(m1.memory_bank) > 0:
                    m1.memory_bank.append(actual_next_state.copy())
                    if m1.memory_bank_max is not None and len(m1.memory_bank) > m1.memory_bank_max:
                        m1.memory_bank = m1.memory_bank[-m1.memory_bank_max:]

        self._reward_modulator = float(np.clip(total_td_delta, -2.0, 2.0))
        signal = {
            "prediction_error": total_pred_error,
            "td_delta": total_td_delta,
            "reward": reward,
            "cost": cost,
            "memory_eligible": float(memory_eligible),
            "execution_confirmed": float(action_feedback_eligible),
        }
        if memory_eligible:
            # Filled-entry traces are retained for final lifecycle attribution.
            self._cache_execution_trace(trace)
        if action_feedback_eligible:
            provisional_advantage = float(np.clip(total_td_delta + float(reward) - float(cost), -2.0, 2.0))
            signal["neuron_credit_count"] = float(self._apply_neuron_action_credit(
                trace, provisional_advantage, strength=0.20, provenance="immediate_execution"
            ))
        else:
            signal["neuron_credit_count"] = 0.0
        memory_query = self._query_memory(trace.encoded_state)
        gates_arr = np.asarray(trace.gate_values, dtype=float)
        if action_feedback_eligible:
            action_context = self._action_head_context(trace.encoded_state, trace.selected_action, memory_query, gates_arr)
            param_target = np.zeros(8, dtype=float)
            parameter_offset = PARAMETER_OFFSET if trace.selected_action.size >= OUTPUT_COUNT else 11
            if trace.selected_action.size > parameter_offset:
                kept = min(8, trace.selected_action.size - parameter_offset)
                param_target[:kept] = trace.selected_action[parameter_offset:parameter_offset + kept]
            advantage_sign = 1.0 if (total_td_delta + reward - cost) >= 0.0 else -0.35
            self.action_parameter_head.observe(action_context, advantage_sign * param_target, strength=1.0 + abs(total_td_delta) + abs(reward))
            self._credit_reasoning("action_head", total_td_delta + reward - cost, strength=1.0 + abs(total_td_delta) + abs(reward))
            risk_context = self._risk_gate_context(trace.raw_obs, trace.selected_action, gates_arr, memory_query)
            bad = max(0.0, -float(total_td_delta)) + max(0.0, float(cost))
            risk_target = np.array([
                np.clip(bad, 0.0, 1.0),
                np.clip(float(cost) * 5.0, 0.0, 1.0),
                np.clip(max(0.0, -float(total_td_delta)), 0.0, 1.0),
            ], dtype=float)
            self.risk_cost_gate.observe(risk_context, risk_target, strength=1.0 + bad)
            self._credit_reasoning("risk", -bad, strength=1.0 + bad)
        if memory_eligible:
            # Keep verified positive, negative, and neutral transitions.  A
            # positive-only buffer produces survivorship bias during replay.
            self.experience_buffer.append({
                "trace": copylib.deepcopy(trace),
                "next_obs": next_obs_arr.copy(),
                "reward": float(reward),
                "cost": float(cost),
                "done": bool(done),
                "td_delta": float(total_td_delta),
                "prediction_error": float(total_pred_error),
            })
            experience_cap = getattr(self, "experience_buffer_max", None)
            if experience_cap is not None and int(experience_cap) > 0 and len(self.experience_buffer) > int(experience_cap):
                self.experience_buffer = self.experience_buffer[-int(experience_cap):]
        if memory_eligible:
            self.memory_supernet.absorb_transition(trace, next_obs_arr, reward, cost, signal)
            mistake_event = self._record_mistake_episode_from_trace(
                trace,
                next_obs_arr,
                reward,
                cost,
                signal,
                source="mt5_ledger" if dict(getattr(trace, "execution_details", {}) or {}).get("bridge_observed", False) else "journal",
            )
            signal["mistake_episode_recorded"] = 1.0 if mistake_event is not None else 0.0
            curiosity_memory_query = self._query_memory(trace.encoded_state)
            curiosity_context = self.curiosity_system.context_from(
                state=trace.encoded_state,
                action=trace.selected_action,
                memory_query=curiosity_memory_query,
                causal_signals=self.memory_supernet.causal_plasticity_signals(max_items=4),
                starvation=getattr(trace, "experience_starvation", {}),
            )
            curiosity_context = self._reasoning_context("curiosity", curiosity_context, self.curiosity_system.input_size)
            curiosity_output = self.curiosity_system.forward(curiosity_context)
            self.curiosity_system.observe(
                curiosity_context,
                np.asarray(curiosity_output.get("proposal", []), dtype=float),
                reward,
                total_td_delta,
                total_pred_error,
            )
            self._credit_reasoning("curiosity", reward + total_td_delta - cost, strength=1.0 + abs(total_pred_error))
            signal["memory_recorded"] = 1.0
        else:
            signal["mistake_episode_recorded"] = 0.0
            signal["memory_recorded"] = 0.0
        return signal

    def replay_stratified_experiences_from(
        self,
        source: "SelfGrowingPolicy",
        max_items: int = 200,
        *,
        seed: int = 0,
    ) -> int:
        """Replay balanced verified evidence from another candidate.

        Samples are partitioned by outcome sign first, then ranked by signal
        magnitude within each partition.  This prevents the former
        positive-only replay from reinforcing only winners while preserving a
        deterministic bounded replay order.
        """
        self.ensure_compatible(target_input_size=self.input_size)
        source.ensure_compatible(target_input_size=self.input_size)
        buckets: dict[str, list[tuple[float, int, dict[str, object]]]] = {"positive": [], "negative": [], "neutral": []}
        for sequence, item in enumerate(getattr(source, "experience_buffer", [])):
            td_delta = float(item.get("td_delta", 0.0) or 0.0)
            reward = float(item.get("reward", 0.0) or 0.0)
            priority = abs(td_delta) + abs(reward) + 0.25 * abs(float(item.get("cost", 0.0) or 0.0))
            bucket = "positive" if (td_delta > 1e-12 or reward > 1e-12) else ("negative" if (td_delta < -1e-12 or reward < -1e-12) else "neutral")
            buckets[bucket].append((priority, sequence, item))
        per_bucket = max(1, int(max_items) // 3)
        selected: list[tuple[float, int, dict[str, object]]] = []
        for key in ("positive", "negative", "neutral"):
            selected.extend(sorted(buckets[key], key=lambda row: (row[0], row[1]), reverse=True)[:per_bucket])
        if len(selected) < int(max_items):
            remaining = [row for values in buckets.values() for row in values if row not in selected]
            selected.extend(sorted(remaining, key=lambda row: (row[0], row[1]), reverse=True)[: int(max_items) - len(selected)])
        if selected:
            order = np.random.default_rng(int(seed)).permutation(len(selected))
            selected = [selected[int(index)] for index in order]
        replayed = 0
        for _, _, item in selected[:int(max_items)]:
            trace = item.get("trace")
            next_obs = item.get("next_obs")
            if not isinstance(trace, StepTrace) or next_obs is None:
                continue
            reward = float(item.get("reward", 0.0) or 0.0)
            cost = float(item.get("cost", 0.0) or 0.0)
            signal = self.observe_transition(
                copylib.deepcopy(trace),
                np.asarray(next_obs, dtype=float),
                reward,
                cost,
                bool(item.get("done", False)),
                record_experience=False,
            )
            # ``record_experience=False`` prevents recursively re-adding the
            # replay sample; it must not suppress memory learning itself.
            self.memory_supernet.absorb_transition(copylib.deepcopy(trace), np.asarray(next_obs, dtype=float), reward, cost, signal)
            replayed += 1
        return replayed

    def replay_positive_experiences_from(
        self,
        source: "SelfGrowingPolicy",
        max_items: int = 200,
        min_td_delta: float = 0.0,
        min_reward: float = 0.0,
    ) -> int:
        """Compatibility alias retained for callers from V6.

        Threshold arguments are intentionally ignored: V7 replay is
        stratified by outcome so negative/flat evidence cannot disappear.
        """
        del min_td_delta, min_reward
        return self.replay_stratified_experiences_from(source, max_items=max_items, seed=0)

    def _grow_module(self, signal: GrowthSignal, neural_dev: NeuralDevelopmentOutput, rng: np.random.Generator, max_modules: int) -> None:
        parent = max(self.modules, key=lambda m: m.score)
        spec = neural_dev.as_spec()

        gc = self.growth_controller
        if gc is not None:
            new_module = parent.copy()
            branch = ShadowBranch(original_name=parent.name, shadow_name="")
            new_module.name = f"{spec.role}_{len(self.modules) + 1}"
            new_module.role = spec.role
            new_module.lifecycle = MODULE_LIFECYCLE_CANDIDATE
            new_module.gate_b = spec.initial_gate_bias
            new_module.plasticity = spec.initial_plasticity
            new_module.utility = 0.0
            new_module.gate_ema = 0.0
            new_module.age = 0
            new_module.w_in = 0.98 * new_module.w_in
            new_module.w_rec = 0.98 * new_module.w_rec
            new_module.input_mask = _component_input_mask(new_module.input_size, spec.input_sources)
            new_module.output_mask = _component_output_mask(new_module.output_size, spec.output_targets)
            new_module.w_in[:, new_module.input_mask < 0.5] = 0.0
            new_module.w_out[new_module.output_mask < 0.5, :] = 0.0
            branch.shadow_name = new_module.name
            self.active_shadows.append(branch)
        else:
            new_module = parent.copy()
            new_module.name = f"{spec.role}_{len(self.modules) + 1}"
            new_module.role = spec.role
            new_module.lifecycle = MODULE_LIFECYCLE_CANDIDATE
            new_module.gate_b = spec.initial_gate_bias
            new_module.plasticity = spec.initial_plasticity
            new_module.utility = 0.0
            new_module.gate_ema = 0.0
            new_module.age = 0
            new_module.w_in = 0.98 * new_module.w_in
            new_module.w_rec = 0.98 * new_module.w_rec
            new_module.input_mask = _component_input_mask(new_module.input_size, spec.input_sources)
            new_module.output_mask = _component_output_mask(new_module.output_size, spec.output_targets)
            new_module.w_in[:, new_module.input_mask < 0.5] = 0.0
            new_module.w_out[new_module.output_mask < 0.5, :] = 0.0
            new_module.lifecycle = MODULE_LIFECYCLE_CANDIDATE
        self.modules.append(new_module)
        self._ensure_lateral_edges()
        self.growth_events.append({
            "round": self.revision,
            "type": "neural_spec_growth",
            "role": spec.role,
            "parent": parent.name,
            "module": new_module.name,
            "input_sources": spec.input_sources,
            "output_targets": spec.output_targets,
            "neural_action_confidence": float(np.max(neural_dev.action_probs)),
            "neural_role_confidence": float(np.max(neural_dev.role_probs)),
            "growth_score": signal.score,
            "error": signal.error,
            "novelty": signal.novelty,
            "conflict": signal.conflict,
            "representation_need": signal.representation_need,
            "complexity": signal.complexity,
        })

    def commit_shadow(self, shadow_name: str, val_score: float) -> bool:
        gc = self.growth_controller
        for branch in self.active_shadows:
            if branch.shadow_name == shadow_name:
                if gc is not None:
                    gc.finalize_shadow(branch, val_score, delta_keep=0.01)
                else:
                    branch.validation_score = val_score
                    branch.accepted = val_score > 0.01
                if branch.accepted:
                    self.growth_events.append({
                        "round": self.revision,
                        "type": "shadow_committed",
                        "module": shadow_name,
                        "val_score": val_score,
                    })
                return branch.accepted
        return False

    def rollback_shadow(self, shadow_name: str) -> None:
        self.modules = [m for m in self.modules if m.name != shadow_name]
        self._ensure_lateral_edges()
        self.active_shadows = [b for b in self.active_shadows if b.shadow_name != shadow_name]
        self._runtime_memory.pop(shadow_name, None)
        self.growth_events.append({
            "round": self.revision,
            "type": "shadow_rolled_back",
            "module": shadow_name,
        })

    def _try_module_split(self, max_modules: int, force: bool = False) -> None:
        if len(self.modules) >= max_modules:
            return
        best_pair: tuple[float, int, int] | None = None
        for i, mod_a in enumerate(self.modules):
            for j, mod_b in enumerate(self.modules):
                if j <= i:
                    continue
                if mod_a.w_in.shape != mod_b.w_in.shape:
                    continue
                grad_a = mod_a.w_in.flatten()
                grad_b = mod_b.w_in.flatten()
                sim = cosine_gradient_similarity(grad_a, grad_b)
                if best_pair is None or sim < best_pair[0]:
                    best_pair = (sim, i, j)
        if best_pair is None:
            return
        sim, i, j = best_pair
        mod_a = self.modules[i]
        mod_b = self.modules[j]
        if not force and not should_split_module(mod_a.w_in.flatten(), mod_b.w_in.flatten(), tau_split=-0.2):
            return
        new_module = mod_a.copy()
        new_module.name = f"{mod_a.name}_split_{len(self.modules) + 1}"
        new_module.role = mod_a.role
        new_module.lifecycle = MODULE_LIFECYCLE_CANDIDATE
        new_module.gate_b -= 1.0
        new_module.plasticity = min(2.0, new_module.plasticity * 1.2)
        new_module.w_in = 0.5 * (mod_a.w_in - mod_b.w_in)
        new_module.w_out = 0.5 * (mod_a.w_out + mod_b.w_out)
        self.modules.append(new_module)
        self._ensure_lateral_edges()
        self.growth_events.append({
            "round": self.revision,
            "type": "neural_split_module",
            "parent_a": mod_a.name,
            "parent_b": mod_b.name,
            "module": new_module.name,
            "cosine_sim": sim,
        })

    def _try_module_merge(self) -> None:
        from .growth import compute_module_similarity
        if len(self.modules) <= 2:
            return
        merge_candidates = []
        for i, mod_a in enumerate(self.modules):
            for j, mod_b in enumerate(self.modules):
                if j <= i:
                    continue
                if mod_a.role == "m1_thinking_core" or mod_b.role == "m1_thinking_core":
                    continue
                if mod_a.w_in.shape != mod_b.w_in.shape:
                    continue
                gate_a = np.array([mod_a.gate_ema])
                gate_b = np.array([mod_b.gate_ema])
                hidden_a = self._runtime_memory.get(mod_a.name, np.zeros(mod_a.hidden_size))
                hidden_b = self._runtime_memory.get(mod_b.name, np.zeros(mod_b.hidden_size))
                out_a = mod_a.w_out.flatten()[:min(10, mod_a.w_out.size)]
                out_b = mod_b.w_out.flatten()[:min(10, mod_b.w_out.size)]
                sim = compute_module_similarity(gate_a, gate_b, hidden_a, hidden_b, out_a, out_b)
                if sim > 0.8 and mod_a.w_in.shape == mod_b.w_in.shape:
                    merge_candidates.append((sim, i, j))

        if merge_candidates:
            merge_candidates.sort(reverse=True)
            _, i, j = merge_candidates[0]
            kept = self.modules[i]
            removed = self.modules[j]
            alpha = 0.5
            kept.w_in = alpha * kept.w_in + (1 - alpha) * removed.w_in
            kept.w_out = alpha * kept.w_out + (1 - alpha) * removed.w_out
            kept.b_hidden = alpha * kept.b_hidden + (1 - alpha) * removed.b_hidden
            kept.b_out = alpha * kept.b_out + (1 - alpha) * removed.b_out
            self.modules.pop(j)
            self._ensure_lateral_edges()
            self.growth_events.append({
                "round": self.revision,
                "type": "merge_modules",
                "kept": kept.name,
                "removed": removed.name,
            })

    def growth_signal(self) -> GrowthSignal:
        if not self.modules:
            return GrowthSignal(1.0, 1.0, 0.0, 1.0, 0.0)
        scores = np.asarray([module.score for module in self.modules], dtype=float)
        utilities = np.asarray([module.utility for module in self.modules], dtype=float)
        gates = np.asarray([module.gate_ema for module in self.modules], dtype=float)
        plasticities = np.asarray([module.plasticity for module in self.modules], dtype=float)
        stagnant = self.best_fitness == -float("inf") or float(np.max(scores)) <= 0.0
        error = 1.0 if stagnant else max(0.0, -float(np.mean(scores)) / (abs(float(np.max(scores))) + 1.0))
        novelty = float(np.mean(gates < 0.02))
        conflict = float(np.std(scores) / (abs(float(np.mean(scores))) + 1.0))
        representation_need = float(np.mean((plasticities > 1.4) | (utilities <= 0.0)))
        complexity = self.structural_complexity() / 10.0
        memory_summary = self.memory_supernet.summary() if hasattr(self, "memory_supernet") else {}
        if int(memory_summary.get("strategies", 0)) > len(self.modules):
            representation_need = min(2.0, representation_need + 0.15)
        if int(memory_summary.get("meta_strategies", 0)) > 0:
            error = max(0.0, error - 0.05)
        return GrowthSignal(error, novelty, conflict, representation_need, complexity)

    def prune_low_utility(self, force: bool = False) -> None:
        if len(self.modules) <= 1:
            return
        retained: list[ModuleState] = []
        pruned: list[str] = []
        for module in self.modules:
            if module.role == "m1_thinking_core":
                retained.append(module)
                continue
            if module.lifecycle != MODULE_LIFECYCLE_PRUNING or module.utility > -1.0:
                retained.append(module)
            else:
                pruned.append(module.name)
        if pruned:
            self.modules = retained
            self._ensure_lateral_edges()
            self.growth_events.append({"round": self.revision, "type": "prune_low_utility", "modules": pruned})

    def consolidate(self, fitness: float, baseline_fitness: float = 0.0) -> None:
        """Use raw rank fitness for selection and a relative signal for neural change."""
        rank_fitness = float(fitness)
        learning_signal = mt5_learning_signal(rank_fitness, baseline_fitness=float(baseline_fitness))
        self.best_fitness = max(float(self.best_fitness), rank_fitness)
        previous_signal = float(self.loss_history[-1]) if self.loss_history else 0.0
        self._reward_modulator = float(np.clip(learning_signal - previous_signal, -2.0, 2.0))
        utility_context = self._utility_context(learning_signal)
        neural_window_utility, neural_trust = self.neural_utility.evaluate_window(utility_context)
        fitness_target = np.array([learning_signal, 1.0 if learning_signal > 0.0 else 0.0], dtype=float)
        self.neural_utility.observe(utility_context, fitness_target, strength=1.0 + abs(self._reward_modulator))
        self._credit_reasoning("utility", learning_signal, strength=1.0 + abs(self._reward_modulator))
        structural_signal = float(
            (1.0 - neural_trust) * learning_signal + neural_trust * neural_window_utility
        )

        usages = np.asarray([
            max(0.02, module.gate_ema) * (0.25 + 0.75 * min(1.0, module.activation_ema))
            for module in self.modules
        ], dtype=float)
        usage_total = max(1e-9, float(np.sum(usages)))
        credits: dict[str, float] = {}

        for idx, module in enumerate(self.modules):
            module_credit = float(structural_signal * usages[idx] / usage_total)
            credits[module.name] = module_credit
            module.score = 0.8 * module.score + 0.2 * learning_signal
            module.utility = 0.85 * module.utility + 0.15 * module_credit

            if module.score > 0.0:
                module.plasticity = max(0.05, module.plasticity * 0.98)

            self._apply_fitness_credit(module, module_credit, self._last_observation)
            hidden = self._runtime_memory.get(module.name, np.zeros(module.hidden_size, dtype=float))
            module.observe_usage(module.gate_ema, hidden, learning_signal)

        self._apply_lateral_fitness_credit(credits)
        self.loss_history.append(learning_signal)
        if hasattr(self, "neural_development") and self.neural_development is not None:
            signal = self.growth_signal()
            memory_summary = self.memory_supernet.summary() if hasattr(self, "memory_supernet") else {}
            context = self.neural_development.context_from(
                signal=signal,
                modules=self.modules,
                memory_summary=memory_summary,
                reward_modulator=structural_signal,
            )
            dev_out = self.neural_development.forward(context)
            self.neural_development.observe(context, dev_out, structural_signal)
            self._credit_reasoning("development", structural_signal, strength=1.0 + abs(structural_signal))
        if self.growth_controller is not None and len(self.loss_history) >= 2:
            self.growth_controller.update_thresholds(self.loss_history[-1] - self.loss_history[-2])

    def apply_bridge_feedback(
        self,
        gate_usage: tuple[float, ...],
        hidden_norms: tuple[float, ...],
        fitness: float,
        baseline_fitness: float = 0.0,
    ) -> None:
        learning_signal = mt5_learning_signal(float(fitness), baseline_fitness=float(baseline_fitness))
        for idx, module in enumerate(self.modules):
            gate = float(gate_usage[idx]) if idx < len(gate_usage) else module.gate_ema
            hidden_norm = float(hidden_norms[idx]) if idx < len(hidden_norms) else module.activation_ema
            module.gate_ema = 0.8 * module.gate_ema + 0.2 * gate
            module.activation_ema = 0.8 * module.activation_ema + 0.2 * hidden_norm
            module.utility = 0.8 * module.utility + 0.2 * gate * learning_signal
            if hidden_norm > 0.95:
                module.plasticity = min(2.5, module.plasticity * 1.05)
            elif hidden_norm < 0.02 and module.age >= 3:
                module.gate_b -= 0.05
            module.apply_homeostatic_plasticity()
            if module.utility <= 0.0 and module.gate_ema < 0.01 and module.age >= 5:
                module.lifecycle = MODULE_LIFECYCLE_PRUNING
            elif module.utility > 0.0 and module.gate_ema >= 0.01:
                module.lifecycle = MODULE_LIFECYCLE_ACTIVE

    def run_development_cycle(
        self,
        train_fn,
        analyze_fn,
        validation_data=None,
        rng: np.random.Generator | None = None,
    ) -> dict[str, object]:
        if rng is None:
            rng = np.random.default_rng()

        report: dict[str, object] = {"steps": []}

        train_metrics = train_fn(self)
        report["steps"].append({"phase": "train", "metrics": train_metrics})

        analysis = analyze_fn(self) if analyze_fn else {}
        report["steps"].append({"phase": "analyze", "analysis": analysis})

        diagnosis = self._diagnose(analysis)
        report["steps"].append({"phase": "diagnose", "diagnosis": diagnosis})

        developments = self._develop(diagnosis, rng)
        report["steps"].append({"phase": "develop", "developments": developments})

        self._stabilize()
        report["steps"].append({"phase": "stabilize"})

        if validation_data is not None:
            eval_result = self._evaluate(validation_data)
            report["steps"].append({"phase": "evaluate", "result": eval_result})
            self._keep_or_prune(eval_result)
            report["steps"].append({"phase": "keep_or_prune"})

        return report

    def _diagnose(self, analysis: dict) -> list[str]:
        issues = []
        for module in self.modules:
            if module.gate_ema < 0.01 and module.age >= 3:
                issues.append(f"{module.name}:unused")
            if module.activation_ema > 0.8:
                issues.append(f"{module.name}:saturated")
            if module.utility < 0.0:
                issues.append(f"{module.name}:negative_utility")
            if module.plasticity > 1.5:
                issues.append(f"{module.name}:high_plasticity")
        return issues

    def _develop(self, issues: list[str], rng: np.random.Generator) -> list[str]:
        actions = []
        for issue in issues:
            parts = issue.split(":")
            if len(parts) != 2:
                continue
            module_name, issue_type = parts
            if issue_type == "unused":
                actions.append(f"prune:{module_name}")
            elif issue_type == "saturated":
                actions.append(f"expand:{module_name}")
            elif issue_type == "negative_utility":
                actions.append(f"reduce_gate:{module_name}")
            elif issue_type == "high_plasticity":
                actions.append(f"consolidate:{module_name}")
        return actions

    def _stabilize(self) -> None:
        for module in self.modules:
            module.apply_homeostatic_plasticity()
            if module.w_rec is not None:
                rec_norm = float(np.linalg.norm(module.w_rec))
                if rec_norm > 0:
                    module.w_rec *= min(1.0, np.sqrt(module.w_rec.size) * 0.25 / rec_norm)

    def _evaluate(self, validation_data) -> dict[str, float]:
        return {"validation_fitness": float(self.best_fitness)}

    def _keep_or_prune(self, eval_result: dict) -> None:
        gc = self.growth_controller
        val_fitness = eval_result.get("validation_fitness", 0.0)
        for module in self.modules:
            if module.lifecycle == MODULE_LIFECYCLE_CANDIDATE:
                if gc is not None:
                    baseline_loss = -self.best_fitness if self.best_fitness > -float("inf") else 0.0
                    new_loss = -val_fitness
                    accepted, reasons = gc.validate_change(baseline_loss, new_loss, delta_keep=0.01)
                    if accepted and module.utility > 0.0 and module.gate_ema > 0.01:
                        module.lifecycle = MODULE_LIFECYCLE_ACTIVE
                    elif module.age >= 10:
                        module.lifecycle = MODULE_LIFECYCLE_PRUNING
                else:
                    if module.utility > 0.0 and module.gate_ema > 0.01:
                        module.lifecycle = MODULE_LIFECYCLE_ACTIVE
                    elif module.age >= 10:
                        module.lifecycle = MODULE_LIFECYCLE_PRUNING


@dataclass
class NeuronNode:
    node_id: str
    module_name: str
    index: int
    pos3: np.ndarray
    hpos: np.ndarray
    energy: float = 1.0
    knowledge: float = 0.0
    trust: float = 0.5
    activation_ema: float = 0.0

    def copy(self) -> "NeuronNode":
        return NeuronNode(
            node_id=self.node_id,
            module_name=self.module_name,
            index=int(self.index),
            pos3=self.pos3.copy(),
            hpos=self.hpos.copy(),
            energy=float(self.energy),
            knowledge=float(self.knowledge),
            trust=float(self.trust),
            activation_ema=float(self.activation_ema),
        )


@dataclass
class EdgeRoute:
    route_id: str
    src: str
    dst: str
    distance: float
    bandwidth: float = 1.0
    trust: float = 0.5
    utility: float = 0.0
    alpha: float = 0.0

    def copy(self) -> "EdgeRoute":
        return EdgeRoute(
            route_id=self.route_id,
            src=self.src,
            dst=self.dst,
            distance=float(self.distance),
            bandwidth=float(self.bandwidth),
            trust=float(self.trust),
            utility=float(self.utility),
            alpha=float(self.alpha),
        )


@dataclass
class ResourceLedger:
    energy_flow: dict[str, float] = field(default_factory=dict)
    knowledge_flow: dict[str, float] = field(default_factory=dict)
    costs: dict[str, float] = field(default_factory=dict)

    def copy(self) -> "ResourceLedger":
        return ResourceLedger(
            energy_flow=dict(self.energy_flow),
            knowledge_flow=dict(self.knowledge_flow),
            costs=dict(self.costs),
        )


@dataclass
class KnowledgeStore:
    vectors: dict[str, np.ndarray] = field(default_factory=dict)
    utility: dict[str, float] = field(default_factory=dict)
    # No fixed world-memory ceiling.  Explicit positive limits remain opt-in.
    max_items: int | None = None

    def copy(self) -> "KnowledgeStore":
        return KnowledgeStore(
            vectors={key: value.copy() for key, value in self.vectors.items()},
            utility=dict(self.utility),
            max_items=self.max_items,
        )

    def upsert(self, key: str, vector: np.ndarray, utility: float) -> None:
        self.vectors[key] = np.asarray(vector, dtype=float).copy()
        self.utility[key] = float(utility)
        if self.max_items is not None and int(self.max_items) > 0 and len(self.vectors) > int(self.max_items):
            ranked = sorted(self.utility.items(), key=lambda item: item[1])
            for drop_key, _ in ranked[: len(self.vectors) - int(self.max_items)]:
                self.vectors.pop(drop_key, None)
                self.utility.pop(drop_key, None)


@dataclass
class NationState:
    name: str
    pos3: np.ndarray
    hpos: np.ndarray
    energy: float = 1.0
    knowledge: float = 0.0
    trust: float = 0.5
    utility: float = 0.0
    neurons: list[NeuronNode] = field(default_factory=list)
    edge_routes: list[EdgeRoute] = field(default_factory=list)

    def copy(self) -> "NationState":
        return NationState(
            name=self.name,
            pos3=self.pos3.copy(),
            hpos=self.hpos.copy(),
            energy=float(self.energy),
            knowledge=float(self.knowledge),
            trust=float(self.trust),
            utility=float(self.utility),
            neurons=[node.copy() for node in self.neurons],
            edge_routes=[route.copy() for route in self.edge_routes],
        )

    def influence(self) -> float:
        energy_factor = 1.0 / (1.0 + np.exp(-np.clip(self.energy - 0.5, -10.0, 10.0)))
        knowledge_factor = 1.0 / (1.0 + np.exp(-np.clip(self.knowledge, -10.0, 10.0)))
        trust_factor = np.clip(self.trust, 0.05, 1.5)
        return float(np.clip(0.25 + energy_factor * knowledge_factor * trust_factor, 0.05, 2.0))


@dataclass
class WorldState:
    nations: dict[str, NationState] = field(default_factory=dict)
    ledger: ResourceLedger = field(default_factory=ResourceLedger)
    knowledge_store: KnowledgeStore = field(default_factory=KnowledgeStore)
    revision: int = 0

    def copy(self) -> "WorldState":
        return WorldState(
            nations={name: nation.copy() for name, nation in self.nations.items()},
            ledger=self.ledger.copy(),
            knowledge_store=self.knowledge_store.copy(),
            revision=int(self.revision),
        )

    @staticmethod
    def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))

    @staticmethod
    def hyperbolic_similarity(a: np.ndarray, b: np.ndarray) -> float:
        av = np.asarray(a, dtype=float)
        bv = np.asarray(b, dtype=float)
        a_norm = min(float(np.linalg.norm(av)), 0.999)
        b_norm = min(float(np.linalg.norm(bv)), 0.999)
        diff = float(np.linalg.norm(av - bv))
        denom = max(1e-9, (1.0 - a_norm * a_norm) * (1.0 - b_norm * b_norm))
        arg = 1.0 + 2.0 * diff * diff / denom
        distance = float(np.arccosh(max(1.0, arg)))
        return float(1.0 / (1.0 + distance))

    @classmethod
    def seed_for_modules(cls, modules: list[ModuleState], rng: np.random.Generator) -> "WorldState":
        world = cls()
        for idx, module in enumerate(modules):
            angle = 2.0 * np.pi * idx / max(1, len(modules))
            pos3 = np.array([np.cos(angle), np.sin(angle), 0.15 * idx], dtype=float)
            hpos = np.clip(rng.normal(0.0, 0.08, size=3), -0.25, 0.25)
            neurons = []
            for n_idx in range(module.hidden_size):
                offset = rng.normal(0.0, 0.05, size=3)
                neurons.append(NeuronNode(
                    node_id=f"{module.name}:n{n_idx}",
                    module_name=module.name,
                    index=n_idx,
                    pos3=pos3 + offset,
                    hpos=np.clip(hpos + rng.normal(0.0, 0.02, size=3), -0.5, 0.5),
                ))
            world.nations[module.name] = NationState(module.name, pos3, hpos, neurons=neurons)
        world.rebuild_routes()
        return world

    def ensure_for_modules(self, modules: list[ModuleState], rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        existing = dict(self.nations)
        self.nations = {}
        for idx, module in enumerate(modules):
            nation = existing.get(module.name)
            if nation is None:
                angle = 2.0 * np.pi * idx / max(1, len(modules))
                pos3 = np.array([np.cos(angle), np.sin(angle), 0.15 * idx], dtype=float)
                hpos = np.clip(rng.normal(0.0, 0.08, size=3), -0.25, 0.25)
                nation = NationState(module.name, pos3, hpos)
            nation.name = module.name
            nation.neurons = [
                node for node in nation.neurons
                if node.module_name == module.name and node.index < module.hidden_size
            ]
            known = {node.index for node in nation.neurons}
            for n_idx in range(module.hidden_size):
                if n_idx not in known:
                    nation.neurons.append(NeuronNode(
                        node_id=f"{module.name}:n{n_idx}",
                        module_name=module.name,
                        index=n_idx,
                        pos3=nation.pos3 + rng.normal(0.0, 0.05, size=3),
                        hpos=np.clip(nation.hpos + rng.normal(0.0, 0.02, size=3), -0.5, 0.5),
                    ))
            self.nations[module.name] = nation
        self.rebuild_routes()

    def rebuild_routes(self) -> None:
        names = list(self.nations)
        for name, nation in self.nations.items():
            old = {route.dst: route for route in nation.edge_routes}
            routes = []
            for other_name in names:
                if other_name == name:
                    continue
                other = self.nations[other_name]
                dist = self.euclidean_distance(nation.pos3, other.pos3)
                route = old.get(other_name) or EdgeRoute(
                    route_id=f"{name}->{other_name}",
                    src=name,
                    dst=other_name,
                    distance=dist,
                )
                route.distance = dist
                route.bandwidth = float(np.clip(1.0 / (1.0 + dist), 0.05, 1.0))
                routes.append(route)
            nation.edge_routes = routes

    def message_weight(self, src: str, dst: str) -> float:
        if src not in self.nations or dst not in self.nations or src == dst:
            return 0.0
        a = self.nations[src]
        b = self.nations[dst]
        dist = self.euclidean_distance(a.pos3, b.pos3)
        spatial = 1.0 / (1.0 + dist)
        semantic = self.hyperbolic_similarity(a.hpos, b.hpos)
        resource = np.clip(a.energy, 0.0, 2.0) * np.clip(a.trust, 0.0, 2.0)
        return float(np.clip(spatial * semantic * resource, 0.0, 2.0))

    def influences(self) -> dict[str, float]:
        values: dict[str, float] = {}
        for name, nation in self.nations.items():
            inbound = sum(self.message_weight(other, name) for other in self.nations if other != name)
            values[name] = float(np.clip(nation.influence() + 0.1 * inbound, 0.05, 2.0))
        return values

    def update_after_transition(self, trace: StepTrace, learning_signal: dict[str, float], modules: list[ModuleState]) -> None:
        td_delta = float(learning_signal.get("td_delta", 0.0))
        pred_error = float(learning_signal.get("prediction_error", 0.0))
        reward = float(learning_signal.get("reward", 0.0))
        gate_values = list(trace.gate_values)
        for idx, module in enumerate(modules):
            nation = self.nations.get(module.name)
            if nation is None:
                continue
            gate = float(gate_values[idx]) if idx < len(gate_values) else module.gate_ema
            positive = max(0.0, td_delta) + max(0.0, reward)
            negative = max(0.0, -td_delta) + max(0.0, pred_error - 1.0)
            nation.energy = float(np.clip(0.985 * nation.energy + 0.03 * positive - 0.01 * gate - 0.02 * negative, 0.01, 3.0))
            nation.knowledge = float(np.clip(0.99 * nation.knowledge + 0.05 * positive + 0.01 * max(0.0, 1.0 - pred_error), -3.0, 3.0))
            nation.trust = float(np.clip(0.98 * nation.trust + 0.03 * np.tanh(td_delta) - 0.01 * negative, 0.01, 1.5))
            nation.utility = float(0.9 * nation.utility + 0.1 * (td_delta - pred_error))
            self.ledger.energy_flow[module.name] = nation.energy
            self.ledger.knowledge_flow[module.name] = nation.knowledge
            self.ledger.costs[module.name] = negative
            self.knowledge_store.upsert(module.name, np.array([nation.energy, nation.knowledge, nation.trust], dtype=float), nation.utility)
        self.revision += 1


@dataclass
class SpatialWorldPolicy(SelfGrowingPolicy):
    world: WorldState = field(default_factory=WorldState)
    world_core: str = "mixed_geometry_v1"

    @classmethod
    def seed(cls, input_size: int = INPUT_COUNT, output_size: int = OUTPUT_COUNT, hidden_size: int = 24, seed: int | None = None) -> "SpatialWorldPolicy":
        base = SelfGrowingPolicy.seed(input_size=input_size, output_size=output_size, hidden_size=hidden_size, seed=seed)
        rng = np.random.default_rng(seed)
        return cls.from_policy(base, rng=rng)

    @classmethod
    def from_policy(cls, policy: SelfGrowingPolicy, rng: np.random.Generator | None = None) -> "SpatialWorldPolicy":
        policy.ensure_compatible()
        rng = rng or np.random.default_rng(0)
        world = getattr(policy, "world", None)
        world = world.copy() if isinstance(world, WorldState) else WorldState.seed_for_modules(policy.modules, rng)
        world.ensure_for_modules(policy.modules, rng)
        gc = policy.growth_controller
        gc_copy = GrowthController(
            tau_grow=gc.tau_grow,
            tau_prune=gc.tau_prune,
            tau_module=gc.tau_module,
            tau_split=gc.tau_split,
            tau_merge=gc.tau_merge,
            H=gc.H,
            K_max=gc.K_max,
            N_max=gc.N_max,
            n_min=gc.n_min,
            coeffs=gc.coeffs,
        ) if gc is not None else None
        converted = cls(
            input_size=policy.input_size,
            output_size=policy.output_size,
            modules=[module.copy() for module in policy.modules],
            lateral_edges={key: value.copy() for key, value in getattr(policy, "lateral_edges", {}).items()},
            m1_components=tuple(policy.m1_components),
            revision=int(policy.revision),
            best_fitness=float(policy.best_fitness),
            growth_events=[dict(event) for event in policy.growth_events],
            gating_mode=policy.gating_mode,
            growth_controller=gc_copy,
            plasticity_coeffs=policy.plasticity_coeffs,
            loss_history=list(policy.loss_history),
            active_shadows=list(policy.active_shadows),
            experience_buffer=[dict(item) for item in getattr(policy, "experience_buffer", [])],
            experience_buffer_max=max(1, int(getattr(policy, "experience_buffer_max", 4096) or 4096)),
            rejection_memory=[dict(item) for item in getattr(policy, "rejection_memory", [])],
            rejection_memory_max=max(1, int(getattr(policy, "rejection_memory_max", 2048) or 2048)),
            memory_supernet=policy.memory_supernet.copy() if hasattr(policy, "memory_supernet") and policy.memory_supernet is not None else ExperienceSuperNetwork(),
            mistake_episode_store=(
                policy.mistake_episode_store.copy()
                if hasattr(policy, "mistake_episode_store") and policy.mistake_episode_store is not None else MistakeEpisodeStore()
            ),
            curiosity_system=policy.curiosity_system.copy() if hasattr(policy, "curiosity_system") and policy.curiosity_system is not None else CuriosityNeuronSystem.seed(),
            neural_development=policy.neural_development.copy() if hasattr(policy, "neural_development") and policy.neural_development is not None else NeuralDevelopmentController.seed(),
            neural_utility=policy.neural_utility.copy() if hasattr(policy, "neural_utility") and policy.neural_utility is not None else NeuralUtilityNetwork.seed(),
            risk_cost_gate=policy.risk_cost_gate.copy() if hasattr(policy, "risk_cost_gate") and policy.risk_cost_gate is not None else NeuralRiskCostGate.seed(),
            action_parameter_head=policy.action_parameter_head.copy() if hasattr(policy, "action_parameter_head") and policy.action_parameter_head is not None else NeuralActionParameterHead.seed(),
            world=world,
        )
        converted.live_observatory_config = dict(getattr(policy, "live_observatory_config", {}) or {})
        writer = getattr(policy, "_live_observer", None)
        converted._live_observer = writer.copy() if isinstance(writer, LiveTelemetryWriter) else None
        converted._live_observer_started = bool(getattr(policy, "_live_observer_started", False))
        return converted

    def copy(self) -> "SpatialWorldPolicy":
        copied = SpatialWorldPolicy.from_policy(self)
        copied.world = self.world.copy()
        copied.world_core = self.world_core
        return copied

    def ensure_compatible(self, target_input_size: int | None = None) -> None:
        super().ensure_compatible(target_input_size=target_input_size)
        if not hasattr(self, "world") or not isinstance(self.world, WorldState):
            self.world = WorldState.seed_for_modules(self.modules, np.random.default_rng(0))
        if not hasattr(self, "world_core"):
            self.world_core = "mixed_geometry_v1"
        self.world.ensure_for_modules(self.modules, np.random.default_rng(0))

    def _world_influences(self) -> dict[str, float]:
        self.ensure_compatible()
        return self.world.influences()

    def _compute_gates(self, obs: np.ndarray) -> np.ndarray:
        gates = super()._compute_gates(obs)
        influences = self.world.influences()
        factors = np.asarray([influences.get(module.name, 1.0) for module in self.modules], dtype=float)
        gated = gates * factors
        if self.gating_mode in {"sparsemax", "top_k"}:
            total = float(np.sum(gated))
            return gated / total if total > 1e-9 else gates
        return np.clip(gated, 0.0, 1.5)

    def _with_spatial_hidden_bias(self, fn):
        self.ensure_compatible()
        influences = self.world.influences()
        old_biases = []
        for module in self.modules:
            old_biases.append(module.b_hidden.copy())
            factor = influences.get(module.name, 1.0)
            module.b_hidden = module.b_hidden + np.full(module.hidden_size, 0.05 * (factor - 1.0), dtype=float)
        try:
            return fn()
        finally:
            for module, old_bias in zip(self.modules, old_biases):
                module.b_hidden = old_bias

    def activate(self, observation: np.ndarray) -> np.ndarray:
        return self._with_spatial_hidden_bias(lambda: super(SpatialWorldPolicy, self).activate(observation))

    def act(self, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
        trace = self._with_spatial_hidden_bias(lambda: super(SpatialWorldPolicy, self).act(observation, goal))
        influences = self.world.influences()
        if influences:
            scale = float(np.clip(np.mean(list(influences.values())), 0.25, 1.75))
            trace = replace(
                trace,
                selected_action=np.tanh(trace.selected_action * scale),
                candidate_actions=np.tanh(trace.candidate_actions * scale),
            )
        return trace

    def observe_transition(
        self,
        trace: StepTrace,
        next_obs: np.ndarray,
        reward: float,
        cost: float,
        done: bool,
        gamma: float = 0.95,
        record_experience: bool = True,
    ) -> dict[str, float]:
        signal = super().observe_transition(trace, next_obs, reward, cost, done, gamma=gamma, record_experience=record_experience)
        self.world.update_after_transition(trace, signal, self.modules)
        return signal

    def propose(self, rng: np.random.Generator, mutation_sigma: float, max_modules: int) -> "SpatialWorldPolicy":
        trial = SpatialWorldPolicy.from_policy(super().propose(rng, mutation_sigma, max_modules), rng=rng)
        trial.world = self.world.copy()
        trial.world.ensure_for_modules(trial.modules, rng)
        for nation in trial.world.nations.values():
            nation.pos3 = nation.pos3 + rng.normal(0.0, mutation_sigma * 0.1, size=3)
            nation.hpos = np.clip(nation.hpos + rng.normal(0.0, mutation_sigma * 0.05, size=3), -0.95, 0.95)
        trial.world.rebuild_routes()
        return trial

    def consolidate(self, fitness: float, baseline_fitness: float = 0.0) -> None:
        super().consolidate(fitness, baseline_fitness=baseline_fitness)
        learning_signal = mt5_learning_signal(float(fitness), baseline_fitness=float(baseline_fitness))
        for module in self.modules:
            nation = self.world.nations.get(module.name)
            if nation is None:
                continue
            nation.utility = 0.9 * nation.utility + 0.1 * learning_signal
            nation.trust = float(np.clip(nation.trust + 0.01 * learning_signal, 0.01, 1.5))
            if learning_signal > 0.0:
                nation.knowledge = float(np.clip(nation.knowledge + 0.01, -3.0, 3.0))


# ── Training Functions ────────────────────────────────────────────────────────


def build_mt5_curriculum_stages(config: AppConfig, rounds: int) -> list[MT5CurriculumStage]:
    max_positions = int(config.risk.max_positions)
    max_holding_bars = int(config.orders.max_holding_bars)
    if max_positions == 0:
        return [MT5CurriculumStage("full", 0, "2025.01.01", "2025.02.01", 0, max_holding_bars)]
    if rounds < 3:
        return [MT5CurriculumStage("full", 0, "2025.01.01", "2025.02.01", max_positions, max_holding_bars)]
    medium_start = max(1, int(np.ceil(rounds / 3.0)))
    full_start = max(medium_start + 1, int(np.ceil(rounds * 2.0 / 3.0)))
    return [
        MT5CurriculumStage("easy-week", 0, "2025.01.01", "2025.01.08", max(1, min(max_positions, 2)), min(max_holding_bars, 30)),
        MT5CurriculumStage("medium-half-month", medium_start, "2025.01.01", "2025.01.16", max(1, min(max_positions, 3)), min(max_holding_bars, 45)),
        MT5CurriculumStage("full-month", full_start, "2025.01.01", "2025.02.01", max_positions, max_holding_bars),
    ]


def mt5_curriculum_stage_for_round(stages: list[MT5CurriculumStage], round_index: int) -> tuple[int, MT5CurriculumStage]:
    active_index = 0
    for idx, stage in enumerate(stages):
        if round_index >= stage.start_round:
            active_index = idx
    return active_index, stages[active_index]


def constant_hold_policy(_observation: np.ndarray) -> np.ndarray:
    outputs = np.zeros(OUTPUT_COUNT, dtype=float)
    outputs[0] = 1.0
    return outputs


def _normalize_policy_core(policy_core: str | None = None) -> str:
    core = (policy_core or "current").strip().lower()
    if core in {"current", "self-growing", "self_growing", "default"}:
        return "current"
    if core in {"world", "spatial", "spatial-world", "spatial_world"}:
        return "world"
    raise ValueError(f"Unsupported policy core: {policy_core}")


def _seed_policy_for_core(policy_core: str | None = None) -> SelfGrowingPolicy:
    core = _normalize_policy_core(policy_core)
    if core == "world":
        return SpatialWorldPolicy.seed()
    return SelfGrowingPolicy.seed()


def _tao_factory_policy_doc_lap_context(
    config: AppConfig,
    *,
    policy_core: str = "current",
    seed: int | None = None,
):
    """Factory tạo graph độc lập cho từng context owner.

    Factory này chỉ tạo *cơ thể neural* cùng schema. Nó không copy winner,
    memory, reward history hay bias BUY/SELL giữa các chuyên gia.
    """
    rng = np.random.default_rng(seed)
    hidden_size = int(getattr(getattr(config, "model", None), "hidden_size", 24) or 24)
    hidden_layers = int(getattr(getattr(config, "model", None), "hidden_layers", 1) or 1)
    recurrent_core = str(
        getattr(getattr(config, "model", None), "recurrent_core", "legacy_rnn") or "legacy_rnn"
    ).strip().lower()
    core = _normalize_policy_core(policy_core)

    def tao_policy_doc_lap():
        policy_seed = int(rng.integers(0, 2**31 - 1))
        if core == "world":
            return SpatialWorldPolicy.seed(
                input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
                hidden_size=hidden_size, seed=policy_seed,
            )
        if recurrent_core == "openai_five_lstm_v1":
            from .chuyen_gia_policy import ChuyenGiaPolicy
            return ChuyenGiaPolicy.seed(
                input_size=INPUT_COUNT,
                output_size=OUTPUT_COUNT,
                hidden_size=hidden_size,
                hidden_layers=hidden_layers,
                recurrent_core=recurrent_core,
                seed=policy_seed,
            )
        return SelfGrowingPolicy.seed(
            input_size=INPUT_COUNT, output_size=OUTPUT_COUNT,
            hidden_size=hidden_size, hidden_layers=hidden_layers, seed=policy_seed,
        )

    return tao_policy_doc_lap


def tao_he_chuyen_gia_mot_context_mot_policy(
    config: AppConfig,
    *,
    policy_core: str = "current",
    seed: int | None = None,
):
    """Tạo năm policy gốc độc lập, mỗi policy sở hữu một context nhiệm vụ.

    Hàm này không clone một ``winner`` sang các expert. Mỗi expert seed bằng
    seed riêng nên graph, weight, memory và growth state độc lập ngay từ lúc
    khởi tạo. Nó là điểm vào duy nhất của kiến trúc một-context-một-policy.
    """
    from .he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia, HeChuyenGiaMotContextMotPolicy

    cau_hinh = getattr(config, "he_chuyen_gia", CauHinhHeChuyenGia())
    if not isinstance(cau_hinh, CauHinhHeChuyenGia):
        cau_hinh = CauHinhHeChuyenGia()
    if not bool(cau_hinh.bat):
        raise ValueError("he_chuyen_gia.bat phai = true de tao he mot-context-mot-policy")
    return HeChuyenGiaMotContextMotPolicy(
        cau_hinh=cau_hinh,
        tao_policy=_tao_factory_policy_doc_lap_context(config, policy_core=policy_core, seed=seed),
    )


def tai_he_chuyen_gia_mot_context_mot_policy(
    path: str | Path,
    config: AppConfig,
    *,
    policy_core: str = "current",
    seed: int | None = None,
):
    """Đọc artifact context-expert và gắn lại factory để có thể tách context sau load."""
    from .he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia, HeChuyenGiaMotContextMotPolicy

    with Path(path).open("rb") as handle:
        system = load_pickle_with_removed_context_structural_migration(handle)
    if not isinstance(system, HeChuyenGiaMotContextMotPolicy):
        raise TypeError("artifact khong phai he mot-context-mot-policy")
    # The artifact carries the profile it was trained with, but continuation
    # must use the caller's current profile for non-learned runtime controls
    # (selector penalties/probe budget, routing thresholds, and promotion
    # guardrails).  Learned weights/evidence remain in the loaded system.
    current_expert_config = getattr(config, "he_chuyen_gia", None)
    if isinstance(current_expert_config, CauHinhHeChuyenGia):
        system.cau_hinh = current_expert_config
    system.gan_factory_tao_policy(
        _tao_factory_policy_doc_lap_context(config, policy_core=policy_core, seed=seed)
    )
    system.ensure_compatible()
    return system


def backup_context_expert_artifacts(output_path: str | Path) -> dict[str, str]:
    """Copy existing context-expert artifact/registry before an overwrite."""
    output = Path(output_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backups: dict[str, str] = {}
    for label, path in (
        ("policy", output),
        ("registry", output.with_suffix(".registry.json")),
    ):
        if not path.exists() or not path.is_file():
            continue
        backup = path.with_name(f"{path.stem}.rollback.{timestamp}{path.suffix}")
        shutil.copy2(path, backup)
        backups[label] = str(backup)
    return backups


def _cau_hinh_policy_chuyen_gia_cho_mt5(policy, config: AppConfig, *, policy_core: str) -> None:
    """Cấu hình từng policy owner; tuyệt đối không tạo graph/global memory chung."""
    from .he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia

    config_expert = getattr(config, "he_chuyen_gia", CauHinhHeChuyenGia())
    if not isinstance(config_expert, CauHinhHeChuyenGia):
        config_expert = CauHinhHeChuyenGia()
    # The facade owns context routing and writes these values into each local
    # policy during proposal construction.  Rebind it for loaded artifacts so
    # a candidate cannot silently keep the seed's direction-prior/topology
    # profile during an MT5 A/B run.
    for component in (
        policy,
        getattr(policy, "nhan_dien", None),
        getattr(policy, "registry", None),
    ):
        if component is not None and hasattr(component, "cau_hinh"):
            component.cau_hinh = config_expert
    timing_configurator = getattr(getattr(policy, "nhan_dien", None), "configure_timing_network", None)
    if callable(timing_configurator):
        timing_configurator(getattr(config, "learning", None))
    configure_risk_caps = getattr(policy, "configure_risk_caps", None)
    if callable(configure_risk_caps):
        configure_risk_caps(getattr(config, "risk", None))
    learning_cfg = getattr(config, "learning", None)
    plasticity_input_normalization_mode = str(
        getattr(learning_cfg, "plasticity_input_normalization_mode", "raw") or "raw"
    ).strip().lower()
    if plasticity_input_normalization_mode not in {"raw", "tanh_v1", "soft_clip_v1", "groupwise_v2", "groupwise_mtf_v3", "structure_aware_v4"}:
        plasticity_input_normalization_mode = "raw"
    plasticity_update_clip = float(np.clip(
        float(getattr(learning_cfg, "plasticity_update_clip", 0.25) or 0.25),
        0.0,
        10.0,
    ))
    for expert in policy.experts.values():
        local = expert.policy
        local.ensure_compatible()
        setattr(local, "plasticity_input_normalization_mode", plasticity_input_normalization_mode)
        setattr(local, "plasticity_update_clip", plasticity_update_clip)
        for module in getattr(local, "modules", ()) or ():
            module.plasticity_input_normalization_mode = plasticity_input_normalization_mode
            module.plasticity_update_clip = plasticity_update_clip
        local.graph_trial_mode = False
        local.graph_trial_id = ""
        local.continuous_mode = True
        if getattr(local, "m1_backend", "direct_rnn") == "m1_graph":
            local.configure_graph_supernetwork(getattr(config, "supernetwork", None))
            local.graph_supernetwork_config["max_modules"] = max(
                1, int(getattr(getattr(config, "model", None), "max_modules", 4) or 4)
            )
        # Learning-integrity settings belong to every policy owner, including
        # the production direct-RNN path.  Keeping this call inside the graph
        # branch would let a loaded direct-RNN candidate retain stale
        # action-recovery thresholds from its pickle during MT5 A/B runs.
        local.configure_learning_integrity(getattr(config, "learning", None))
        expectancy_cfg = getattr(config, "expectancy", None)
        if expectancy_cfg is not None:
            local.expectancy_learning = bool(getattr(expectancy_cfg, "enabled", True))
            local.expectancy_min_samples = int(getattr(expectancy_cfg, "min_samples", 6))

    observability_cfg = getattr(config, "observability", None)
    if bool(getattr(observability_cfg, "enabled", False)):
        enable = getattr(policy, "enable_live_observatory", None)
        if callable(enable):
            enable(
                observability_cfg,
                run_id=f"context-expert-{int(time.time())}",
                source="huan_luyen_he_chuyen_gia_mt5",
                metadata={"che_do": "mot_context_mot_chuyen_gia"},
            )


def huan_luyen_he_chuyen_gia_mt5(
    config: AppConfig,
    *,
    output_path: str | Path,
    seed_system_path: str | Path | None = None,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    from_date: str | None = None,
    to_date: str | None = None,
    backtest_days: int | None = None,
    model: int = 0,
    policy_core: str = "current",
    seed: int | None = None,
) -> dict[str, object]:
    """Huấn luyện một phiên MT5 cho kho policy context-owner.

    Hàm không gọi ``ContinuousSessionRuntime`` cũ vì runtime đó duy trì
    registry specialist toàn cục. Toàn bộ trace, lifecycle và ``develop`` ở
    đây được facade chuyển về đúng policy owner. Đây là điều kiện để tránh
    global policy quay trở lại qua đường continuous runtime.
    """
    from .he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia

    config_expert = getattr(config, "he_chuyen_gia", CauHinhHeChuyenGia())
    if not bool(getattr(config_expert, "bat", False)):
        raise ValueError("can dung config he_chuyen_gia.bat=true")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rollback = backup_context_expert_artifacts(output)
    core = _normalize_policy_core(policy_core)
    if seed_system_path:
        system = tai_he_chuyen_gia_mot_context_mot_policy(
            seed_system_path, config, policy_core=core, seed=seed,
        )
    else:
        system = tao_he_chuyen_gia_mot_context_mot_policy(config, policy_core=core, seed=seed)
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config, policy_core=core)
    reset_runtime = getattr(system, "reset_runtime_state", None)
    runtime_state_reset = callable(reset_runtime)
    if runtime_state_reset:
        reset_runtime()

    start = str(from_date or "2025.01.01")
    end = str(to_date or "")
    if not end:
        days = max(1, int(backtest_days or 31))
        try:
            end = (datetime.strptime(start, "%Y.%m.%d") + timedelta(days=days)).strftime("%Y.%m.%d")
        except ValueError:
            end = "2025.02.01"
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=Path(runner_dir),
            from_date=start,
            to_date=end,
            timeout_sec=int(timeout_sec),
            max_positions=int(config.risk.max_positions),
            max_holding_bars=int(config.orders.max_holding_bars),
            min_daily_trades=int(config.orders.min_daily_trades),
            max_daily_trades=int(config.orders.max_daily_trades),
            **_mt5_execution_contract_kwargs(config),
            visual=bool(visual),
            shutdown_terminal=bool(shutdown_terminal),
            model=int(model),
            # Không cắm continuous runtime cũ vì nó có state specialist toàn cục.
            continuous_runtime=None,
        )
    )
    fitness, metrics = evaluator.evaluate_policy(1, system, 0)
    with output.open("wb") as handle:
        pickle.dump(system, handle, protocol=pickle.HIGHEST_PROTOCOL)
    registry_path = system.luu_so_dang_ky(output.with_suffix(".registry.json"))
    return {
        "mode": "mot_context_mot_chuyen_gia_mt5",
        "policy": str(output),
        "so_dang_ky": str(registry_path),
        "fitness": float(fitness),
        "metrics": _mt5_metrics_to_dict(metrics),
        "rollback": rollback,
        "runtime_state_reset_before_mt5": bool(runtime_state_reset),
        "date_range": {"from": start, "to": end},
        "ràng_buộc": {
            "mot_context_mot_owner": True,
            "policy_chung_ra_lenh": False,
            "tron_policy": False,
            "hoc_tu_flat_idle": False,
        },
    }


def _coerce_policy_core(policy: SelfGrowingPolicy, policy_core: str | None = None) -> SelfGrowingPolicy:
    core = _normalize_policy_core(policy_core)
    policy.ensure_compatible(target_input_size=INPUT_COUNT)
    if core == "world" and not isinstance(policy, SpatialWorldPolicy):
        return SpatialWorldPolicy.from_policy(policy)
    if core == "current" and isinstance(policy, SpatialWorldPolicy):
        base = SelfGrowingPolicy(
            input_size=policy.input_size,
            output_size=policy.output_size,
            modules=[module.copy() for module in policy.modules],
            m1_components=tuple(policy.m1_components),
            revision=int(policy.revision),
            best_fitness=float(policy.best_fitness),
            growth_events=[dict(event) for event in policy.growth_events],
            gating_mode=policy.gating_mode,
            growth_controller=policy.growth_controller,
            plasticity_coeffs=policy.plasticity_coeffs,
            loss_history=list(policy.loss_history),
            active_shadows=list(policy.active_shadows),
            experience_buffer=[dict(item) for item in getattr(policy, "experience_buffer", [])],
            experience_buffer_max=max(1, int(getattr(policy, "experience_buffer_max", 4096) or 4096)),
            rejection_memory=[dict(item) for item in getattr(policy, "rejection_memory", [])],
            rejection_memory_max=max(1, int(getattr(policy, "rejection_memory_max", 2048) or 2048)),
            memory_supernet=policy.memory_supernet.copy() if hasattr(policy, "memory_supernet") and policy.memory_supernet is not None else ExperienceSuperNetwork(),
            mistake_episode_store=(
                policy.mistake_episode_store.copy()
                if hasattr(policy, "mistake_episode_store") and policy.mistake_episode_store is not None else MistakeEpisodeStore()
            ),
            curiosity_system=policy.curiosity_system.copy() if hasattr(policy, "curiosity_system") and policy.curiosity_system is not None else CuriosityNeuronSystem.seed(),
            neural_development=policy.neural_development.copy() if hasattr(policy, "neural_development") and policy.neural_development is not None else NeuralDevelopmentController.seed(),
            neural_utility=policy.neural_utility.copy() if hasattr(policy, "neural_utility") and policy.neural_utility is not None else NeuralUtilityNetwork.seed(),
            risk_cost_gate=policy.risk_cost_gate.copy() if hasattr(policy, "risk_cost_gate") and policy.risk_cost_gate is not None else NeuralRiskCostGate.seed(),
            action_parameter_head=policy.action_parameter_head.copy() if hasattr(policy, "action_parameter_head") and policy.action_parameter_head is not None else NeuralActionParameterHead.seed(),
        )
        base.ensure_compatible(target_input_size=INPUT_COUNT)
        return base
    return policy


def _normalize_m1_backend(m1_backend: str | None = None) -> str:
    backend = str(m1_backend or "direct_rnn").strip().lower()
    aliases = {
        "rnn": "direct_rnn",
        "direct": "direct_rnn",
        "direct_head": "direct_rnn",
        "direct_rnn": "direct_rnn",
        "graph": "m1_graph",
        "m1": "m1_graph",
        "m1_graph": "m1_graph",
    }
    if backend not in aliases:
        raise ValueError("decision backend must be 'direct_rnn' or 'm1_graph'.")
    return aliases[backend]

def _apply_m1_backend(policy: SelfGrowingPolicy, m1_backend: str | None = None) -> SelfGrowingPolicy:
    policy.decision_backend = _normalize_m1_backend(m1_backend)
    policy.m1_backend = policy.decision_backend
    policy.ensure_compatible(target_input_size=policy.input_size)
    return policy

def load_policy(path: str | Path, policy_core: str | None = None) -> SelfGrowingPolicy:
    with Path(path).open("rb") as handle:
        policy = load_pickle_with_removed_context_structural_migration(handle)
    if not isinstance(policy, SelfGrowingPolicy):
        raise TypeError(f"Expected SelfGrowingPolicy artifact: {path}")
    policy = _coerce_policy_core(policy, policy_core)
    policy.ensure_compatible(target_input_size=getattr(policy, "input_size", INPUT_COUNT))
    return policy


def _array_from_journal(value, fallback_size: int = 0) -> np.ndarray:
    target_size = max(0, int(fallback_size))
    if value is None:
        return np.zeros(target_size, dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if target_size > 0:
        out = np.zeros(target_size, dtype=float)
        out[: min(target_size, arr.size)] = arr[:target_size]
        return out
    return arr


def _policy_trace_state_size(policy: SelfGrowingPolicy) -> int:
    for module in getattr(policy, "modules", []):
        m1 = getattr(module, "m1_component", None)
        if m1 is not None:
            return int(getattr(m1, "state_size", getattr(module, "hidden_size", 0)) or 0)
    return int(getattr(policy.modules[0], "hidden_size", 0)) if getattr(policy, "modules", []) else 0


def _step_trace_from_journal(entry: dict[str, object], policy: SelfGrowingPolicy) -> StepTrace | None:
    trace_data = entry.get("trace")
    if not isinstance(trace_data, dict):
        return None
    state_size = _policy_trace_state_size(policy)
    m1_k = 1
    for module in getattr(policy, "modules", []):
        m1 = getattr(module, "m1_component", None)
        if m1 is not None:
            m1_k = int(getattr(m1, "K", m1_k) or m1_k)
            break
    raw_obs = _array_from_journal(trace_data.get("raw_obs"), policy.input_size)
    encoded_state = _array_from_journal(trace_data.get("encoded_state"), state_size)
    selected_action = _array_from_journal(trace_data.get("selected_action"), policy.output_size)
    raw_module_traces = trace_data.get("module_action_traces", [])
    module_action_traces: tuple[ModuleActionTrace, ...] = tuple(
        ModuleActionTrace(
            module_name=str(item.get("module_name", "")),
            gate=float(item.get("gate", 0.0)),
            hidden=_array_from_journal(item.get("hidden")),
            raw_output=_array_from_journal(item.get("raw_output")),
        )
        for item in raw_module_traces if isinstance(item, dict)
    )
    return StepTrace(
        trace_id=str(entry.get("trace_id") or trace_data.get("trace_id") or "journal-trace"),
        timestamp=float(trace_data.get("timestamp", entry.get("created_at", time.time()))),
        raw_obs=raw_obs,
        encoded_state=encoded_state,
        memory_before=_array_from_journal(trace_data.get("memory_before"), state_size),
        memory_after=_array_from_journal(trace_data.get("memory_after"), state_size),
        selected_idx=int(trace_data.get("selected_idx", 0)),
        selected_action=selected_action,
        candidate_actions=_array_from_journal(trace_data.get("candidate_actions"), m1_k * policy.output_size).reshape(m1_k, policy.output_size),
        candidate_logits=_array_from_journal(trace_data.get("candidate_logits"), m1_k),
        candidate_scores=_array_from_journal(trace_data.get("candidate_scores"), m1_k),
        selected_action_log_prob=float(trace_data.get("selected_action_log_prob", 0.0)),
        predicted_next_state=_array_from_journal(trace_data.get("predicted_next_state"), state_size),
        critic_value=float(trace_data.get("critic_value", 0.0)),
        confidence=float(trace_data.get("confidence", entry.get("confidence", 0.0))),
        novelty=float(trace_data.get("novelty", 0.0)),
        conflict=float(trace_data.get("conflict", 0.0)),
        risk=float(trace_data.get("risk", 0.0)),
        gate_values=tuple(float(item) for item in trace_data.get("gate_values", [])),
        prediction_error=float(trace_data.get("prediction_error", entry.get("prediction_error", 0.0))),
        td_delta=float(trace_data.get("td_delta", entry.get("td_delta", 0.0))),
        action_details=dict(entry.get("action_details", trace_data.get("action_details", {})) or {}),
        positions_pipe=str(entry.get("positions_pipe", trace_data.get("positions_pipe", "")) or ""),
        positions_pipe_after=str(entry.get("positions_pipe_after", trace_data.get("positions_pipe_after", "")) or ""),
        execution_details=dict(entry.get("execution_details", trace_data.get("execution_details", {})) or {}),
        account_context=dict(entry.get("account_context", trace_data.get("account_context", {})) or {}),
        account_context_after=dict(entry.get("account_context_after", trace_data.get("account_context_after", {})) or {}),
        experience_starvation=dict(trace_data.get("experience_starvation", entry.get("experience_starvation", {})) or {}),
        module_action_traces=module_action_traces,
        active_neuron_ids=tuple(str(item) for item in trace_data.get("active_neuron_ids", []) or []),
        active_edge_ids=tuple(str(item) for item in trace_data.get("active_edge_ids", []) or []),
        graph_edge_gates={str(key): float(value) for key, value in dict(trace_data.get("graph_edge_gates", {}) or {}).items()},
        graph_region_activity={str(key): float(value) for key, value in dict(trace_data.get("graph_region_activity", {}) or {}).items()},
        graph_traces={
            str(graph_id): dict(record or {})
            for graph_id, record in dict(trace_data.get("graph_traces", {}) or {}).items()
            if isinstance(record, dict)
        },
        memory_match_quality=float(trace_data.get("memory_match_quality", 0.0) or 0.0),
        lifecycle_tags=tuple(str(item) for item in trace_data.get("lifecycle_tags", []) or []),
        module_need_reason=str(trace_data.get("module_need_reason", "") or ""),
    )


def _tail_jsonl_lines(path: Path, max_lines: int, max_bytes: int) -> list[bytes]:
    """Read only a bounded complete-line tail from a potentially large journal."""
    line_limit = max(1, int(max_lines))
    byte_limit = max(1024, int(max_bytes))
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            end = handle.tell()
            start = max(0, end - byte_limit)
            handle.seek(start)
            if start > 0:
                handle.readline()  # discard a partial JSON record
            data = handle.read()
    except OSError:
        return []
    return data.splitlines()[-line_limit:]


def replay_positive_experience_journal(
    policy: SelfGrowingPolicy,
    journal_path: str | Path | None,
    max_items: int = 200,
    min_td_delta: float = 0.0,
    min_reward: float = 0.0,
    scan_max_lines: int = 20_000,
    scan_max_bytes: int = 8 * 1024 * 1024,
) -> int:
    """Replay a bounded top-k tail; never parse an unbounded journal in RAM."""
    if not journal_path or max_items <= 0:
        return 0
    path = Path(journal_path)
    if not path.exists():
        return 0
    heap: list[tuple[float, float, int, dict[str, object]]] = []
    for sequence, raw_line in enumerate(_tail_jsonl_lines(path, scan_max_lines, scan_max_bytes)):
        try:
            entry = json.loads(raw_line.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        if str(entry.get("source", "")) == "mt5_online":
            trace_data = entry.get("trace", {}) if isinstance(entry.get("trace", {}), dict) else {}
            try:
                memory_schema = int(entry.get("memory_key_schema", trace_data.get("memory_key_schema", 0)) or 0)
            except (TypeError, ValueError):
                memory_schema = 0
            if memory_schema != MEMORY_KEY_SCHEMA_VERSION:
                continue
        # v1 MT5 journal rows lack execution evidence and can be made entirely
        # of ignored EA requests.  Do not replay them as trading experience.
        execution_details = dict(entry.get("execution_details", {}) or {})
        if str(entry.get("source", "")) == "mt5_online" and not bool(execution_details.get("memory_eligible", False)):
            continue
        try:
            td_delta = float(entry.get("td_delta", 0.0))
            reward = float(entry.get("reward", 0.0))
        except (TypeError, ValueError):
            continue
        if td_delta <= min_td_delta and reward <= min_reward:
            continue
        item = (td_delta, reward, sequence, entry)
        if len(heap) < max_items:
            heapq.heappush(heap, item)
        elif item[:3] > heap[0][:3]:
            heapq.heapreplace(heap, item)
    candidates = [item[3] for item in sorted(heap, key=lambda item: item[:3], reverse=True)]
    replayed = 0
    for entry in candidates:
        trace = _step_trace_from_journal(entry, policy)
        if trace is None:
            continue
        next_obs = _array_from_journal(entry.get("next_obs"), policy.input_size)
        signal = policy.observe_transition(
            trace,
            next_obs,
            float(entry.get("reward", 0.0)),
            float(entry.get("cost", 0.0)),
            bool(entry.get("done", False)),
            record_experience=False,
        )
        policy.memory_supernet.absorb_transition(
            trace,
            next_obs,
            float(entry.get("reward", 0.0)),
            float(entry.get("cost", 0.0)),
            signal,
        )
        replayed += 1
    return replayed

def replay_stratified_experience_journal(
    policy: SelfGrowingPolicy,
    journal_path: str | Path | None,
    max_items: int = 200,
    *,
    seed: int = 0,
    scan_max_lines: int = 20_000,
    scan_max_bytes: int = 8 * 1024 * 1024,
    plastic_max_items: int = 8,
) -> int:
    """Replay verified positive *and* negative execution evidence.

    Positive-only replay creates survivorship bias: the policy remembers what
    paid but not which state/action combinations should be avoided. This keeps
    a deterministic, balanced tail and still ignores unverified flat-idle rows.
    """
    if not journal_path or max_items <= 0:
        return 0
    path = Path(journal_path)
    if not path.exists():
        return 0
    positive: list[tuple[float, int, dict[str, object]]] = []
    negative: list[tuple[float, int, dict[str, object]]] = []
    neutral: list[tuple[float, int, dict[str, object]]] = []
    for sequence, raw_line in enumerate(_tail_jsonl_lines(path, scan_max_lines, scan_max_bytes)):
        try:
            entry = json.loads(raw_line.decode("utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        evidence = dict(entry.get("execution_details", {}) or {})
        if str(entry.get("source", "")) == "mt5_online" and not bool(evidence.get("memory_eligible", False)):
            continue
        try:
            reward = float(entry.get("reward", 0.0))
            td_delta = float(entry.get("td_delta", 0.0))
        except (TypeError, ValueError):
            continue
        priority = abs(reward) + 0.25 * abs(td_delta)
        item = (priority, sequence, entry)
        if reward > 1e-12 or td_delta > 1e-12:
            positive.append(item)
        elif reward < -1e-12 or td_delta < -1e-12:
            negative.append(item)
        else:
            neutral.append(item)
    half = max_items // 2
    selected = (
        sorted(positive, key=lambda item: (item[0], item[1]), reverse=True)[:half]
        + sorted(negative, key=lambda item: (item[0], item[1]), reverse=True)[:max_items - half]
    )
    if len(selected) < max_items:
        selected += sorted(neutral, key=lambda item: (item[0], item[1]), reverse=True)[:max_items - len(selected)]
    rng = np.random.default_rng(int(seed))
    if selected:
        order = rng.permutation(len(selected))
        selected = [selected[int(i)] for i in order]
    replayed = 0
    plastic_budget = max(0, min(int(max_items), int(plastic_max_items)))
    for _, _, entry in selected:
        trace = _step_trace_from_journal(entry, policy)
        if trace is None:
            continue
        next_obs = _array_from_journal(entry.get("next_obs"), policy.input_size)
        if replayed < plastic_budget:
            signal = policy.observe_transition(
                trace, next_obs, float(entry.get("reward", 0.0)),
                float(entry.get("cost", 0.0)), bool(entry.get("done", False)),
                record_experience=False,
            )
        else:
            signal = {
                "prediction_error": float(entry.get("prediction_error", 0.0) or 0.0),
                "td_delta": float(entry.get("td_delta", 0.0) or 0.0),
                "reward": float(entry.get("reward", 0.0) or 0.0),
                "cost": float(entry.get("cost", 0.0) or 0.0),
                "memory_eligible": 1.0,
                "execution_confirmed": float(bool(dict(getattr(trace, "execution_details", {}) or {}).get("memory_eligible", False))),
                "journal_replay_memory_only": 1.0,
            }
        policy.memory_supernet.absorb_transition(
            trace, next_obs, float(entry.get("reward", 0.0)),
            float(entry.get("cost", 0.0)), signal,
        )
        replayed += 1
    return replayed


def load_policy_callable(path: str | Path):
    policy = load_policy(path)
    return policy.activate


def _frozen_evaluation_copy(policy):
    # Never fall back to the caller's object: evaluation must have a distinct
    # mutable runtime even for policies without a bespoke copy() method.
    evaluated = clone_policy(policy)
    reset = getattr(evaluated, "reset_runtime_state", None)
    if callable(reset):
        reset()
    return evaluated


def evaluate_policy(data, config: AppConfig, policy) -> dict[str, float]:
    # Evaluation must not modify weights, memory or topology.
    evaluated = _frozen_evaluation_copy(policy)
    result = TradingEnvironment(data, config).run(evaluated, learning_enabled=False)
    daily_trade_counts = np.asarray(result.daily_trade_counts, dtype=float)
    metrics = {
        "fitness": result.fitness,
        "equity": result.equity,
        "trades": float(len(result.trades)),
        "max_daily_trades": float(np.max(daily_trade_counts)) if daily_trade_counts.size else 0.0,
        "avg_daily_trades": float(np.mean(daily_trade_counts)) if daily_trade_counts.size else 0.0,
        "total_r": result.total_r,
        "expectancy": result.expectancy,
        "winrate": result.winrate,
        "max_drawdown_pct": result.max_drawdown_pct,
    }
    metrics.update(goal_progress_metrics(result))
    return metrics


def audit_policy(data, config: AppConfig, policy) -> dict[str, object]:
    """Frozen edge audit; return targets/trade quotas are never selection terms."""
    train_df, validation_df, test_df = split_time_ordered(data, config.splits.train, config.splits.validation)
    splits = {"train": train_df, "validation": validation_df, "test": test_df, "all": data}
    metrics = {name: evaluate_policy(split_df, config, policy) for name, split_df in splits.items()}
    def _audit_metric(row: object, key: str, fallback: str | None = None, default: float = 0.0) -> float:
        if not isinstance(row, dict):
            return default
        raw = row.get(key, row.get(fallback, default) if fallback else default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return default
        return value if np.isfinite(value) else default

    # Keep audit compatibility with compact/legacy evaluators while retaining
    # the canonical frozen fitness metric for production runs.
    edge_scores = [_audit_metric(metrics[name], "fitness", "stable_goal_score", -float("inf")) for name in ("train", "validation", "test")]
    drawdowns = [_audit_metric(metrics[name], "max_drawdown_pct", "equity_drawdown_pct", float("inf")) for name in ("train", "validation", "test")]
    expectancies = [_audit_metric(metrics[name], "expectancy", "fitness_expectancy_lcb", 0.0) for name in ("train", "validation", "test")]
    metrics["summary"] = {
        "robust_edge_score": float(np.mean(edge_scores) - 0.35 * np.std(edge_scores) - 100.0 * max(drawdowns)),
        "mean_expectancy": float(np.mean(expectancies)),
        "worst_drawdown_pct": float(max(drawdowns)),
        "total_r_sum": float(sum(_audit_metric(metrics[name], "total_r", default=0.0) for name in ("train", "validation", "test"))),
        "all_splits_goal_hit": bool(all(bool(metrics[name].get("goal_hit", False)) if isinstance(metrics[name], dict) else False for name in ("train", "validation", "test"))),
        "frozen": True,
    }
    return metrics


def training_windows(data, window_bars: int, max_windows: int):
    if max_windows <= 1 or window_bars <= 0 or len(data) <= window_bars:
        return [data.reset_index(drop=True)]
    starts = np.linspace(0, len(data) - window_bars, max_windows, dtype=int)
    unique_starts = list(dict.fromkeys(int(start) for start in starts))
    return [data.iloc[start : start + window_bars].reset_index(drop=True) for start in unique_starts]


def robust_training_fitness(results) -> float:
    """Selection objective: confidence-adjusted edge, drawdown and stability.

    The legacy 5% daily target and daily-trade quota remain reporting metrics
    only. They cannot manufacture an incentive to trade a weak regime.
    """
    if not results:
        return -float("inf")
    scores = np.asarray([tight_risk_score(result) for result in results], dtype=float)
    lcbs = np.asarray([edge_confidence_lcb(result) for result in results], dtype=float)
    drawdowns = np.asarray([float(result.max_drawdown_pct) for result in results], dtype=float)
    return float(
        np.mean(scores)
        - 0.50 * np.std(scores)
        + 40.0 * np.mean(lcbs)
        - 20.0 * np.std(lcbs)
        - 450.0 * np.max(drawdowns)
    )


def validation_selection_score(train_results, validation_result, policy=None) -> float:
    train_edge = float(np.mean([edge_confidence_lcb(result) for result in train_results])) if train_results else -1.0
    val_edge = edge_confidence_lcb(validation_result)
    complexity_fn = getattr(policy, "structural_complexity", None) if policy is not None else None
    complexity = float(complexity_fn()) if callable(complexity_fn) else float(len(getattr(policy, "modules", [])) if policy is not None else 0.0)
    # This is an evidence/compression penalty, not a global topology cap.
    complexity_penalty = 0.25 * complexity
    return float(
        0.35 * robust_training_fitness(train_results)
        + 95.0 * val_edge
        + 0.20 * tight_risk_score(validation_result)
        - 450.0 * validation_result.max_drawdown_pct
        - validation_result.risk_abuse_penalty
        - complexity_penalty
        - (100.0 if validation_result.ruin else 0.0)
    )


def _mt5_execution_contract_kwargs(config: AppConfig) -> dict[str, object]:
    """Mirror the Python execution envelope into every MT5 evaluator.

    Compatibility defaults keep lightweight test/migration configs usable,
    while production ``AppConfig`` always supplies every field explicitly.
    """
    risk = getattr(config, "risk")
    orders = getattr(config, "orders")
    raw_features = getattr(config, "features", None)
    raw_continuous = getattr(config, "continuous", None)
    # Lightweight tests/migration shims sometimes pass a SimpleNamespace
    # instead of AppConfig. Convert it into the canonical dataclasses before
    # hashing; an absent optional field must not crash the evaluator path.
    feature_defaults = FeatureConfig()
    features = FeatureConfig(
        atr_period=int(getattr(raw_features, "atr_period", feature_defaults.atr_period)),
        ema_fast=int(getattr(raw_features, "ema_fast", feature_defaults.ema_fast)),
        ema_slow=int(getattr(raw_features, "ema_slow", feature_defaults.ema_slow)),
        range_window=int(getattr(raw_features, "range_window", feature_defaults.range_window)),
        regime_window=int(getattr(raw_features, "regime_window", feature_defaults.regime_window)),
        atr_percentile_window=int(getattr(raw_features, "atr_percentile_window", feature_defaults.atr_percentile_window)),
        max_spread_atr=float(getattr(raw_features, "max_spread_atr", feature_defaults.max_spread_atr)),
        shock_range_atr=float(getattr(raw_features, "shock_range_atr", feature_defaults.shock_range_atr)),
        wyckoff_range_window=int(getattr(raw_features, "wyckoff_range_window", feature_defaults.wyckoff_range_window)),
        wyckoff_volume_short_window=int(getattr(raw_features, "wyckoff_volume_short_window", feature_defaults.wyckoff_volume_short_window)),
        wyckoff_volume_long_window=int(getattr(raw_features, "wyckoff_volume_long_window", feature_defaults.wyckoff_volume_long_window)),
        wyckoff_event_horizon=int(getattr(raw_features, "wyckoff_event_horizon", feature_defaults.wyckoff_event_horizon)),
        mtf_scaling_mode=str(getattr(raw_features, "mtf_scaling_mode", feature_defaults.mtf_scaling_mode) or feature_defaults.mtf_scaling_mode),
    )
    raw_symbol = getattr(config, "symbol", None)
    symbol_defaults = SymbolConfig()
    symbol = SymbolConfig(
        price_digits=int(getattr(raw_symbol, "price_digits", symbol_defaults.price_digits)),
        point_size=float(getattr(raw_symbol, "point_size", symbol_defaults.point_size)),
        spread_in_points=bool(getattr(raw_symbol, "spread_in_points", symbol_defaults.spread_in_points)),
    )
    entry_offset = float(getattr(orders, "entry_offset_atr", 5.0))
    initial_equity = float(getattr(risk, "initial_equity", 10_000.0))
    min_risk = float(getattr(risk, "min_risk_pct", 0.001))
    max_risk = float(getattr(risk, "max_risk_pct", 0.02))
    max_open = float(getattr(risk, "max_open_risk_pct", max(0.05, max_risk)))
    ruin = float(getattr(risk, "ruin_equity_pct", 0.50))
    fixed = float(getattr(risk, "fixed_risk_pct", 0.0))
    return {
        "period": str(getattr(orders, "timeframe", "M1") or "M1").strip().upper(),
        "entry_offset_atr": entry_offset,
        "technical_min_rr": float(getattr(orders, "technical_min_rr", 0.05)),
        "fixed_sl_atr": float(getattr(orders, "fixed_sl_atr", 5.0)),
        "fixed_tp_atr": float(getattr(orders, "fixed_tp_atr", 10.0)),
        "initial_equity": initial_equity,
        "min_risk_pct": min_risk,
        "max_risk_pct": max_risk,
        "max_open_risk_pct": max_open,
        "ruin_equity_pct": ruin,
        "enforce_entry_limits": bool(getattr(risk, "enforce_entry_limits", False)),
        "fixed_risk_pct": fixed if fixed > 0.0 else None,
        "pending_expiry_bars": int(getattr(orders, "pending_expiry_bars", 60)),
        "pending_cancel_after_decisions": int(getattr(raw_continuous, "pending_cancel_after_decisions", 12)),
        "pending_cancel_interval_decisions": int(getattr(raw_continuous, "pending_cancel_interval_decisions", 4)),
        "max_spread_atr": float(features.max_spread_atr),
        "max_spread_cost_r": float(getattr(orders, "max_spread_cost_r", 0.1)),
        "shock_range_atr": float(features.shock_range_atr),
        "reset_execution_ledger_on_init": False,
        "require_hedging_account": True,
        "feature_atr_period": int(features.atr_period),
        "feature_ema_fast": int(features.ema_fast),
        "feature_ema_slow": int(features.ema_slow),
        "feature_range_window": int(features.range_window),
        "feature_regime_window": int(features.regime_window),
        "feature_atr_percentile_window": int(features.atr_percentile_window),
        "feature_wyckoff_range_window": int(features.wyckoff_range_window),
        "feature_wyckoff_volume_short_window": int(features.wyckoff_volume_short_window),
        "feature_wyckoff_volume_long_window": int(features.wyckoff_volume_long_window),
        "feature_wyckoff_event_horizon": int(features.wyckoff_event_horizon),
        "feature_mtf_scaling_mode": str(getattr(features, "mtf_scaling_mode", "legacy_m1_atr_v1") or "legacy_m1_atr_v1"),
        "feature_schema_hash": feature_schema_fingerprint(
            features,
            symbol,
            session_timezone=str(getattr(getattr(config, "learning", None), "session_timezone", "UTC")),
            spread_semantics=str(getattr(getattr(config, "learning", None), "spread_semantics", "ask_minus_bid_price")),
            mt5_server_utc_offset_hours=int(getattr(getattr(config, "learning", None), "mt5_server_utc_offset_hours", 0)),
            input_normalization_mode=str(getattr(getattr(config, "learning", None), "input_normalization_mode", "raw")),
        ),
        "input_normalization_mode": str(getattr(getattr(config, "learning", None), "input_normalization_mode", "raw")),
        "feature_reference_csv": str(getattr(config, "csv_path", "") or ""),
        "feature_server_utc_offset_hours": int(getattr(getattr(config, "learning", None), "mt5_server_utc_offset_hours", 0)),
        "feature_reference_datetime_column": str(getattr(config, "datetime_column", "Datetime") or "Datetime"),
        "symbol_point_size": float(symbol.point_size),
        "symbol_spread_in_points": bool(symbol.spread_in_points),
    }

def train_self_growing_mt5(
    config: AppConfig,
    rounds: int,
    output_path: str,
    trials: int = 10,
    seed_policy_path: str | Path | None = None,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    curriculum_learning: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    mutation_sigma: float = 0.08,
    max_modules: int | None = None,
    include_seed_trial: bool = True,
    model: int = 0,
    seed: int | None = None,
    policy_core: str = "current",
    m1_backend: str = "direct_rnn",
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None = "outputs/mt5_experience_journal.jsonl",
    replay_journal_max: int = 64,
    enforce_promotion_gate: bool = True,
) -> dict[str, object]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if mt5_close_scope not in {"trial", "round", "never"}:
        raise ValueError("mt5_close_scope must be one of: trial, round, never")
    if replay_journal_max < 0:
        raise ValueError("replay_journal_max must be >= 0")
    if bool(getattr(
        getattr(config, "learning", None), "five_rlhf_policy_update_enabled", False
    )):
        raise ValueError(
            "Five RLHF candidate updates must use "
            "huan-luyen-he-chuyen-gia-mt5; legacy train_self_growing_mt5 "
            "cannot enforce the RLHF candidate-only promotion contract"
        )

    # A shared default journal lets one training run silently replay another
    # run's transitions.  That contaminates candidate evaluation and can make
    # test or A/B latency grow with stale history.  Keep an explicitly chosen
    # journal untouched, but map the historical CLI default to an output-local
    # sidecar journal for isolation.
    if str(experience_journal_path or "") == "outputs/mt5_experience_journal.jsonl":
        experience_journal_path = str(Path(output_path).with_suffix(".journal.jsonl"))

    core = _normalize_policy_core(policy_core)
    effective_shutdown_terminal = bool(shutdown_terminal)
    close_scope = "never" if not effective_shutdown_terminal else mt5_close_scope
    # Visual Strategy Tester sessions cannot reliably reuse a portable terminal:
    # closing the Visual window may leave terminal64.exe alive, and the next
    # /config launch is then absorbed by that old instance and runs in the
    # background.  Force a clean terminal per trial for all multi-pass Visual
    # training lifecycles, including ``--keep-mt5-open``/``never``; otherwise
    # later /config launches can exit without producing a report.
    if visual and close_scope in {"round", "never"}:
        print(
            f"[MT5] visual mode overrides mt5_close_scope={close_scope} -> trial so each subsequent trial reopens Visual.",
            flush=True,
        )
        close_scope = "trial"
        effective_shutdown_terminal = True
    seed_path = Path(seed_policy_path) if seed_policy_path else None
    if seed_path is not None and not seed_path.exists():
        print(f"[Train] seed policy not found: {seed_path}; starting from a fresh {core} policy.", flush=True)
        seed_path = None
    base_policy = load_policy(seed_path, policy_core=core) if seed_path else _seed_policy_for_core(core)
    base_policy = _apply_m1_backend(base_policy, m1_backend)
    causal_cfg = getattr(config, "causal", None)
    if causal_cfg is not None and bool(getattr(causal_cfg, "enabled", True)):
        base_policy.causal_replay_requests_path = str(getattr(causal_cfg, "replay_request_path", "outputs/causal_replay_requests.jsonl"))
        base_policy.causal_replay_results_path = str(getattr(causal_cfg, "replay_result_path", "outputs/causal_replay_results.jsonl"))
        base_policy.ensure_compatible()
        causal_ledger = getattr(base_policy, "causal_ledger", None)
        if isinstance(causal_ledger, CausalLedger):
            causal_ledger.max_decisions = int(getattr(causal_cfg, "max_decisions", causal_ledger.max_decisions))
            causal_ledger.max_reports = int(getattr(causal_cfg, "max_reports", causal_ledger.max_reports))
    observability_cfg = getattr(config, "observability", None)
    if bool(getattr(observability_cfg, "enabled", False)):
        base_policy.enable_live_observatory(
            observability_cfg,
            run_id=f"mt5-{int(time.time())}-{Path(output_path).stem}",
            source="train_self_growing_mt5",
            metadata={
                "output_path": str(output_path),
                "runner_dir": str(runner_dir),
                "from_date": str(from_date or ""),
                "to_date": str(to_date or ""),
            },
        )
    # V13: robust expectancy is policy state, not a CLI-only score. Apply the
    # canonical config to the seed before replay so journal evidence and every
    # candidate copy share the same statistical contract.
    expectancy_cfg = getattr(config, "expectancy", None)
    if expectancy_cfg is not None:
        base_policy.expectancy_learning = bool(getattr(expectancy_cfg, "enabled", True))
        base_policy.expectancy_min_samples = int(getattr(expectancy_cfg, "min_samples", 6))
        base_policy.ensure_compatible()
        expectancy_ledger = getattr(getattr(base_policy, "memory_supernet", None), "contextual_expectancy", None)
        if expectancy_ledger is not None:
            expectancy_ledger.min_samples = int(getattr(expectancy_cfg, "min_samples", 6))
            expectancy_ledger.merge_similarity = float(getattr(expectancy_cfg, "merge_similarity", 0.90))
            expectancy_ledger.max_samples_per_record = int(getattr(expectancy_cfg, "max_samples_per_context", 256))
    if getattr(base_policy, "m1_backend", "direct_rnn") == "m1_graph":
        base_policy.configure_graph_supernetwork(getattr(config, "supernetwork", None))
        base_policy.configure_learning_integrity(getattr(config, "learning", None))
        base_policy.graph_supernetwork_config["max_modules"] = max(1, int(getattr(getattr(config, "model", None), "max_modules", 4) or 4))
        base_policy.graph_supernetwork_config["experience_journal_path"] = str(experience_journal_path) if experience_journal_path else ""
    effective_seed = int(getattr(getattr(config, "learning", None), "seed", 20260629) if seed is None else seed)
    journal_replayed = replay_stratified_experience_journal(
        base_policy, experience_journal_path, max_items=replay_journal_max, seed=effective_seed,
    )
    if journal_replayed:
        print(f"[Train] replayed {journal_replayed} stratified journal transitions from {experience_journal_path}", flush=True)
    rng = np.random.default_rng(effective_seed)
    stages = build_mt5_curriculum_stages(config, rounds) if curriculum_learning else [
        MT5CurriculumStage(
            "selected",
            0,
            from_date or "2025.01.01",
            to_date or "2025.02.01",
            int(config.risk.max_positions),
            int(config.orders.max_holding_bars),
        )
    ]

    def stage_backtest_config(stage: MT5CurriculumStage) -> MT5BacktestConfig:
        return MT5BacktestConfig(
            runner_dir=Path(runner_dir),
            from_date=from_date or stage.from_date,
            to_date=to_date or stage.to_date,
            timeout_sec=timeout_sec,
            max_positions=stage.max_positions,
            max_holding_bars=stage.max_holding_bars,
            min_daily_trades=config.orders.min_daily_trades,
            max_daily_trades=config.orders.max_daily_trades,
            **_mt5_execution_contract_kwargs(config),
            visual=visual,
            shutdown_terminal=effective_shutdown_terminal and close_scope == "trial",
            model=model,
            experience_journal=str(experience_journal_path) if experience_journal_path else None,
            learning_enabled=True,
            rlhf_policy_update_enabled=bool(getattr(
                getattr(config, "learning", None), "five_rlhf_policy_update_enabled", False
            )),
            rlhf_reward_model_path=str(getattr(
                getattr(config, "learning", None), "five_rlhf_reward_model_path", ""
            ) or ""),
            rlhf_reward_feedback_mode=str(getattr(
                getattr(config, "learning", None), "five_rlhf_reward_feedback_mode", "human"
            ) or "human"),
            rlhf_teacher_checkpoint_sha256=str(getattr(
                getattr(config, "learning", None), "five_rlhf_teacher_checkpoint_sha256", ""
            ) or ""),
            rlhf_require_independent_teacher=bool(getattr(
                getattr(config, "learning", None), "five_rlhf_require_independent_teacher", False
            )),
            rlhf_source_checkpoint_sha256=str(getattr(
                getattr(config, "learning", None), "five_rlhf_source_checkpoint_sha256", ""
            ) or ""),
            rlhf_update_interval_packets=int(getattr(
                getattr(config, "learning", None), "five_rlhf_update_interval_packets", 1
            ) or 1),
            rlhf_max_updates_per_run=int(getattr(
                getattr(config, "learning", None), "five_rlhf_max_updates_per_run", 0
            ) or 0),
            entry_guard_single_position=bool(getattr(getattr(config, "learning", None), "entry_guard_single_position", False)),
            entry_loss_guard_enabled=bool(getattr(getattr(config, "learning", None), "entry_loss_guard_enabled", False)),
            entry_loss_guard_min_closed=int(getattr(getattr(config, "learning", None), "entry_loss_guard_min_closed", 6) or 6),
            entry_loss_guard_max_closed_loss_r=float(getattr(getattr(config, "learning", None), "entry_loss_guard_max_closed_loss_r", 0.25) or 0.25),
            entry_loss_guard_consecutive_losses=int(getattr(getattr(config, "learning", None), "entry_loss_guard_consecutive_losses", 4) or 4),
            train_entry_probe_enabled=bool(getattr(getattr(config, "learning", None), "train_entry_probe_enabled", False)),
            train_entry_probe_after_decisions=int(getattr(getattr(config, "learning", None), "train_entry_probe_after_decisions", 12) or 12),
            train_entry_probe_interval_decisions=int(getattr(getattr(config, "learning", None), "train_entry_probe_interval_decisions", 6) or 6),
            train_entry_probe_max_per_day=int(getattr(getattr(config, "learning", None), "train_entry_probe_max_per_day", 2) or 0),
            train_entry_probe_max_per_run=int(getattr(getattr(config, "learning", None), "train_entry_probe_max_per_run", 6) or 0),
            train_entry_probe_min_direction_quality=float(getattr(getattr(config, "learning", None), "train_entry_probe_min_direction_quality", 0.75)),
            runtime_fitness_guard_enabled=bool(getattr(getattr(config, "learning", None), "runtime_fitness_guard_enabled", True)),
            runtime_fitness_rolling_window=int(getattr(getattr(config, "learning", None), "runtime_fitness_rolling_window", 50) or 50),
            runtime_fitness_min_closed=int(getattr(getattr(config, "learning", None), "runtime_fitness_min_closed", 20) or 20),
            runtime_fitness_min_score=float(getattr(getattr(config, "learning", None), "runtime_fitness_min_score", -0.30)),
            runtime_fitness_max_immediate_exit_ratio=float(getattr(getattr(config, "learning", None), "runtime_fitness_max_immediate_exit_ratio", 0.60)),
            runtime_fitness_max_daily_loss_r=float(getattr(getattr(config, "learning", None), "runtime_fitness_max_daily_loss_r", 3.0)),
            runtime_fitness_max_run_loss_r=float(getattr(getattr(config, "learning", None), "runtime_fitness_max_run_loss_r", 8.0)),
            runtime_fitness_max_daily_trades=int(getattr(getattr(config, "learning", None), "runtime_fitness_max_daily_trades", 300) or 0),
            runtime_fitness_max_drawdown_pct=float(getattr(getattr(config, "learning", None), "runtime_fitness_max_drawdown_pct", 20.0)),
            require_real_ticks_for_promotion=bool(getattr(getattr(config, "learning", None), "require_real_ticks_for_promotion", True)),
            require_real_tick_evidence_for_promotion=bool(getattr(getattr(config, "learning", None), "require_real_tick_evidence_for_promotion", True)),
            require_terminal_flat_for_promotion=bool(getattr(getattr(config, "learning", None), "require_terminal_flat_for_promotion", True)),
            run_id=f"train-{effective_seed}-{stage.name}",
        )

    def evaluator_for_stage(stage: MT5CurriculumStage) -> MT5StrategyTesterEvaluator:
        return MT5StrategyTesterEvaluator(stage_backtest_config(stage))

    best_policy = clone_policy(base_policy)
    best: dict[str, object] = {"fitness": -float("inf"), "metrics": {}, "round": 0, "trial_id": 0, "stage": stages[0].name}
    history: list[dict[str, object]] = []
    graph_trial_history_path = (
        str(getattr(getattr(config, "supernetwork", None), "trial_history_path", "") or "")
        if getattr(base_policy, "m1_backend", "direct_rnn") == "m1_graph" else ""
    )
    # Selection baseline is the current champion on the same MT5 window,
    # evaluated frozen. Trial order must never decide growth credit.
    # Production graph candidates are never accepted merely because their
    # scalar score beats a different historical trial. They must pass a same-
    # window baseline comparison and all execution/reconciliation gates.
    strict_graph_promotion = bool(
        enforce_promotion_gate
        and
        getattr(getattr(config, "supernetwork", None), "require_mt5_promotion", False)
        and getattr(base_policy, "m1_backend", "direct_rnn") == "m1_graph"
    )

    for round_index in range(rounds):
        stage_index, stage = mt5_curriculum_stage_for_round(stages, round_index)
        evaluator = evaluator_for_stage(stage)
        # Build frozen config independently from the evaluator instance: unit
        # tests and external runners may provide a lightweight evaluator proxy
        # without a public ``config`` attribute.
        frozen_evaluator = MT5StrategyTesterEvaluator(replace(
            stage_backtest_config(stage), learning_enabled=False,
            rlhf_policy_update_enabled=False,
            experience_journal=None, continuous_runtime=None, shutdown_terminal=effective_shutdown_terminal and close_scope == "trial",
        ))
        try:
            for trial_id in range(1, trials + 1):
                if round_index == 0 and trial_id == 1 and seed_path and include_seed_trial:
                    trial = clone_policy(best_policy)
                else:
                    trial = best_policy.propose(rng, mutation_sigma, max_modules)
                try:
                    train_fitness, train_metrics = evaluator.evaluate_policy(trial_id, trial, round_index)
                    clear_rlhf_reference = getattr(trial, "clear_five_rlhf_reference", None)
                    if callable(clear_rlhf_reference):
                        clear_rlhf_reference()
                    # Candidate selection is always frozen after any permitted
                    # training interaction. No fake empty module telemetry is
                    # passed into plasticity.
                    fitness, metrics = frozen_evaluator.evaluate_policy(trial_id + 500_000, clone_policy(trial), round_index)
                    if np.isfinite(float(best["fitness"])):
                        fitness_baseline, baseline_metrics = frozen_evaluator.evaluate_policy(
                            trial_id + 600_000, clone_policy(best_policy), round_index
                        )
                        promotion_baseline = {"best": {"fitness": float(fitness_baseline), "metrics": _mt5_metrics_to_dict(baseline_metrics)}}
                    else:
                        fitness_baseline = 0.0
                        promotion_baseline = {"best": {"fitness": -float("inf"), "metrics": {}}}
                    # No ``apply_bridge_feedback((), ())``: absent verified
                    # module activity is no evidence and must stay neutral.
                    trial.consolidate(float(fitness), baseline_fitness=float(fitness_baseline))
                    fitness_delta = float(fitness) - float(fitness_baseline)
                    improved_vs_champion = (not np.isfinite(float(best["fitness"]))) or float(fitness) > float(fitness_baseline)

                    shadow_names = [b.shadow_name for b in trial.active_shadows]
                    if shadow_names:
                        baseline = trial.copy()
                        for sn in shadow_names:
                            baseline.modules = [m for m in baseline.modules if m.name != sn]
                            baseline._runtime_memory.pop(sn, None)
                        baseline.active_shadows = []
                        shadow_baseline_fitness, _ = frozen_evaluator.evaluate_policy(trial_id + 700_000, clone_policy(baseline), round_index)
                        for sn in shadow_names:
                            accepted = trial.commit_shadow(sn, fitness - shadow_baseline_fitness)
                            if not accepted:
                                trial.rollback_shadow(sn)
                        trial.active_shadows = []

                    metrics_dict = _mt5_metrics_to_dict(metrics)
                    graph_candidates = [
                        event for event in getattr(trial, "growth_events", [])
                        if event.get("type") == "graph_candidate_proposal"
                    ]
                    graph_outcomes = [
                        event for event in getattr(trial, "growth_events", [])
                        if event.get("type") == "graph_candidate_outcome"
                    ]
                    item = {
                        "round": round_index,
                        "trial_id": trial_id,
                        "fitness": fitness,
                        "training_fitness": float(train_fitness),
                        "training_metrics": _mt5_metrics_to_dict(train_metrics),
                        "paired_champion_fitness": float(fitness_baseline),
                        "metrics": metrics_dict,
                        "modules": [module.name for module in trial.modules],
                        "stage": stage.name,
                        "stage_index": stage_index,
                        "shadow_ab": shadow_names,
                        "causal_plasticity": [
                            event for event in getattr(trial, "growth_events", [])
                            if event.get("type") == "causal_memory_plasticity"
                        ][-3:],
                        "curiosity_proposal": [
                            event for event in getattr(trial, "growth_events", [])
                            if event.get("type") == "curiosity_neural_proposal"
                        ][-3:],
                        "graph_candidates": graph_candidates[-3:],
                        "graph_candidate_outcomes": graph_outcomes[-3:],
                        "mistake_replay": [
                            event for event in getattr(trial, "growth_events", [])
                            if event.get("type") == "mistake_replay_candidate"
                        ][-3:],
                        "module_need_metrics": (
                            trial.graph_supernetwork.metrics(
                                trial._graph_m1_modules()[0][1].graph if trial._graph_m1_modules() else None
                            )
                            if getattr(trial, "graph_supernetwork", None) is not None else {}
                        ),
                        "graph_observability": trial.graph_observability_metrics(),
                        "mistake_episodes": trial.mistake_episode_store.summary(),
                        "child_supernetwork_metrics": (
                            trial.child_supernetwork_manager.metrics()
                            if getattr(trial, "child_supernetwork_manager", None) is not None else {}
                        ),
                    }
                    history.append(item)
                    outcome_starvation = trial.apply_outcome_starvation(
                        improved=improved_vs_champion,
                        fitness_delta=fitness_delta,
                        reason="trial_improved_vs_champion" if improved_vs_champion else "trial_not_improved_vs_champion",
                    )
                    item["outcome_starvation"] = outcome_starvation
                    promotion_reasons: list[str] = []
                    if "fitness_eligible" in metrics_dict and not bool(metrics_dict.get("fitness_eligible", False)):
                        promotion_reasons.append("fitness_not_eligible")
                    elif float(fitness) <= -99.999:
                        promotion_reasons.append("fitness_rank_reject")
                    has_promotion_baseline = np.isfinite(float(best["fitness"]))
                    if strict_graph_promotion:
                        if has_promotion_baseline:
                            promotion_ratio, promotion_delta = _expectancy_promotion_controls(config)
                            promotion_reasons = mt5_promotion_rejection_reasons(
                                promotion_baseline,
                                {"best": {"fitness": float(fitness), "metrics": metrics_dict}},
                                # Quota is deliberately disabled in V7 selection.
                                min_daily_trades=0,
                                backtest_days=_date_span_days(stage.from_date, stage.to_date),
                                min_policy_origin_ratio=promotion_ratio,
                                promotion_lcb_min_delta=promotion_delta,
                            )
                            if "fitness_eligible" in metrics_dict and not bool(metrics_dict.get("fitness_eligible", False)):
                                promotion_reasons.insert(0, "fitness_not_eligible")
                            elif float(fitness) <= -99.999:
                                promotion_reasons.insert(0, "fitness_rank_reject")
                        promoted = bool(np.isfinite(float(fitness)) and not promotion_reasons and improved_vs_champion)
                        if not promoted and not promotion_reasons:
                            promotion_reasons = ["paired_champion_not_improved"]
                    else:
                        promoted = bool(np.isfinite(float(fitness)) and not promotion_reasons and improved_vs_champion)
                        if not promoted and not promotion_reasons:
                            promotion_reasons = ["paired_champion_not_improved"]
                    item["promotion_gate"] = {
                        "strict_graph_promotion": strict_graph_promotion,
                        "baseline": promotion_baseline,
                        "reject_reasons": promotion_reasons,
                    }
                    if promoted:
                        trial.record_graph_candidate_outcome(promoted=True)
                        item["promoted"] = True
                        item["graph_candidate_outcomes"] = [
                            event for event in getattr(trial, "growth_events", [])
                            if event.get("type") == "graph_candidate_outcome"
                        ][-3:]
                        best_policy = clone_policy(trial)
                        best = item
                    else:
                        # MT5 rejection is fed back to the persistent baseline
                        # prior without copying any rejected graph state.
                        best_policy.record_graph_candidate_outcome(promoted=False, source_policy=trial)
                        item["promoted"] = False
                        best_policy.apply_outcome_starvation(
                            improved=improved_vs_champion,
                            fitness_delta=fitness_delta,
                            reason="candidate_not_promoted:" + ",".join(promotion_reasons[:3]),
                        )
                        feedback_event = best_policy.absorb_rejection_feedback(
                            trial,
                            reject_reasons=promotion_reasons,
                            metrics=metrics_dict,
                            fitness_delta=fitness_delta,
                        )
                        item["rejection_feedback"] = feedback_event
                    if graph_trial_history_path and item.get("graph_candidates"):
                        write_graph_trial_history(graph_trial_history_path, item)
                except Exception as exc:
                    history.append({"round": round_index, "trial_id": trial_id, "error": str(exc), "stage": stage.name})
                    best_policy.apply_outcome_starvation(
                        improved=False,
                        fitness_delta=-1.0,
                        reason="trial_error",
                    )
        finally:
            if close_scope == "round":
                evaluator.close_open_terminals()

    if float(best["fitness"]) == -float("inf"):
        if history:
            last = history[-1]
            detail = last.get("error") or (
                f"last_fitness={last.get('fitness')!r} "
                f"reject_reasons={last.get('promotion_gate', {}).get('reject_reasons', [])!r}"
            )
        else:
            detail = "no evaluator history"
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        candidate_path = out.with_name(f"{out.stem}.candidate_unpromoted{out.suffix}")
        with candidate_path.open("wb") as handle:
            pickle.dump(best_policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
        failure_manifest = {
            "schema_version": 1,
            "status": "unpromoted_candidate_only",
            "reason": str(detail),
            "requested_output": str(out),
            "candidate_output": str(candidate_path),
            "rounds_requested": int(rounds),
            "history_count": len(history),
            "history_tail": history[-5:],
            "note": "Winner output is intentionally not written because no MT5 fitness passed promotion/reconciliation gates.",
        }
        candidate_path.with_suffix(candidate_path.suffix + ".manifest.json").write_text(
            json.dumps(failure_manifest, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        raise RuntimeError(f"MT5 training did not produce a valid policy fitness: {detail}")

    final_stage = stages[int(best.get("stage_index", 0))]
    bridge_runtime = evaluator_for_stage(final_stage).publish_policy(best_policy)
    stop_live_observatory = getattr(best_policy, "stop_live_observatory", None)
    if callable(stop_live_observatory):
        # Telemetry is runtime-only. Keep the completed journal but do not make
        # a saved winner silently append to an old training stream when it is
        # later loaded for inference/backtest without --live-observatory.
        stop_live_observatory(reason="train_completed")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(best_policy, handle)
    manifest = run_manifest(
        seed=effective_seed, data_path=getattr(config, "csv_path", None), protocol_version=7,
        extra={
            "output": str(out), "model": int(model), "from_date": str(from_date or ""),
            "to_date": str(to_date or ""), "policy_state_hash": policy_state_hash(best_policy),
            "feature_schema_hash": _mt5_execution_contract_kwargs(config).get("feature_schema_hash", ""),
            "effective_config_sha256": stable_object_hash(config),
            "python_source_sha256": sha256_tree(Path(__file__).resolve().parent),
            "ea_source_sha256": sha256_file(Path(runner_dir) / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"),
            "ea_binary_sha256": sha256_file(Path(runner_dir) / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.ex5"),
            "terminal_binary_sha256": sha256_file(Path(runner_dir) / "terminal64.exe"),
            "runner_dir": str(Path(runner_dir).resolve()),
            "tester_fidelity": "every_tick_real_ticks" if int(model) == 0 else "non_promotable",
        },
    )
    manifest_path = out.with_suffix(out.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {
        "winner": str(out),
        "trials": int(trials),
        "rounds": int(rounds),
        "best": best,
        "curriculum_learning": bool(curriculum_learning),
        "curriculum_stages": [stage.__dict__ for stage in stages],
        "bridge_runtime": bridge_runtime,
        "history": history,
        "include_seed_trial": bool(include_seed_trial),
        "promotion_gate_enabled": bool(enforce_promotion_gate),
        "policy_core": core,
        "m1_backend": getattr(base_policy, "m1_backend", "direct_rnn"),
        "mt5_close_scope": close_scope,
        "experience_journal": str(experience_journal_path) if experience_journal_path else None,
        "journal_replayed": int(journal_replayed),
        "seed": int(effective_seed),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "memory_supernet": best_policy.memory_supernet.summary() if hasattr(best_policy, "memory_supernet") else {},
        "mistake_episodes": best_policy.mistake_episode_store.summary() if hasattr(best_policy, "mistake_episode_store") else {},
        "supernetwork_trial_history": graph_trial_history_path or None,
        "supernetwork_summary": summarize_graph_trial_history(history),
        "child_supernetwork": (
            best_policy.child_supernetwork_manager.metrics()
            if getattr(best_policy, "child_supernetwork_manager", None) is not None else {}
        ),
    }


def _mt5_best_metrics(result: dict[str, object]) -> dict[str, object]:
    best = result.get("best", {})
    return best.get("metrics", {}) if isinstance(best, dict) and isinstance(best.get("metrics", {}), dict) else {}


def _mt5_metrics_to_dict(metrics: object) -> dict[str, object]:
    if isinstance(metrics, dict):
        out = metrics.copy()
    elif hasattr(metrics, "__dict__"):
        out = metrics.__dict__.copy()
    else:
        out = {}
    report_path = out.get("report_path")
    if report_path is not None:
        out["report_path"] = str(report_path)
    ledger_path = out.get("execution_ledger_path")
    if ledger_path is not None:
        out["execution_ledger_path"] = str(ledger_path)
    reconciliation = out.get("reconciliation")
    if reconciliation is not None and hasattr(reconciliation, "__dict__"):
        reconciliation_dict = reconciliation.__dict__.copy()
        out["reconciliation"] = reconciliation_dict
        out["fitness_reconciled"] = bool(reconciliation_dict.get("reconciled", False))
        out["fitness_reconciliation_reasons"] = list(reconciliation_dict.get("reasons", ()) or ())
    summary = out.get("execution_summary")
    if summary is not None and hasattr(summary, "__dict__"):
        summary_dict = summary.__dict__.copy()
        lifecycles = tuple(summary_dict.pop("lifecycles", ()) or ())
        summary_dict["ledger_path"] = str(summary_dict.get("ledger_path") or "")
        summary_dict["lifecycle_preview"] = [
            {
                "position_ticket": str(getattr(item, "position_ticket", "")),
                "trace_id": str(getattr(item, "trace_id", "")),
                "realized_r": float(getattr(item, "realized_r", 0.0) or 0.0),
                "duration_bars": getattr(item, "duration_bars", None),
                "close_reason": str(getattr(item, "close_reason", "")),
            }
            for item in lifecycles[:10]
        ]
        out["execution_summary"] = summary_dict
        for key in (
            "lifecycle_count", "ledger_integrity", "risk_coverage",
            "trace_coverage", "execution_failure_rate",
            "immediate_exit_ratio", "expectancy_lcb", "cvar_5pct",
            "sample_unit", "schema_version",
        ):
            if key in summary_dict:
                out[f"fitness_{key}"] = summary_dict[key]
    feature_audit = out.get("feature_audit")
    if isinstance(feature_audit, dict):
        # Keep a compact, serialisable audit verdict in trial history and
        # promotion gates.  Full mismatch rows remain in the artifact file.
        out["feature_audit_available"] = bool(feature_audit.get("available", False))
        out["feature_audit_valid"] = bool(feature_audit.get("valid", False))
        out["feature_parity_numerical_checked"] = bool(feature_audit.get("numerical_checked", False))
        out["feature_parity_numerical_valid"] = bool(feature_audit.get("numerical_valid", False))
        out["feature_parity_checked_rows"] = int(feature_audit.get("checked_rows", 0) or 0)
        out["feature_parity_mismatch_count"] = int(feature_audit.get("mismatch_count", 0) or 0)
        out["feature_parity_reason"] = str(feature_audit.get("reason", "") or "")
    # Preserve fidelity fields only when the evaluator actually supplied them.
    # Synthetic/unit-test metrics must remain "unknown", not be converted to
    # tester_model=-1 and then rejected as non-real-tick evidence. Production
    # MT5 evaluator metrics always supply both fields.
    if "tester_model" in out:
        out["tester_model"] = int(out.get("tester_model", -1) or -1)
    if "require_real_ticks_for_promotion" in out:
        out["require_real_ticks_for_promotion"] = bool(out.get("require_real_ticks_for_promotion", True))
    if "require_real_tick_evidence_for_promotion" in out:
        out["require_real_tick_evidence_for_promotion"] = bool(out.get("require_real_tick_evidence_for_promotion", True))
    if "require_terminal_flat_for_promotion" in out:
        out["require_terminal_flat_for_promotion"] = bool(out.get("require_terminal_flat_for_promotion", True))
    reconciliation = out.get("reconciliation")
    if isinstance(reconciliation, dict) and "terminal_open_positions" in reconciliation:
        out["fitness_terminal_open_positions"] = int(reconciliation.get("terminal_open_positions", 0) or 0)
    real_tick_evidence = out.get("real_tick_evidence")
    if isinstance(real_tick_evidence, dict):
        out["real_tick_evidence_available"] = bool(real_tick_evidence.get("available", False))
        out["real_tick_evidence_verified"] = bool(real_tick_evidence.get("verified", False))
        out["real_tick_evidence_reason"] = str(real_tick_evidence.get("reason", "") or "")
    if "feature_schema_hash" in out:
        out["feature_schema_hash"] = str(out.get("feature_schema_hash", "") or "")
    expectancy_audit = out.get("expectancy_audit")
    if isinstance(expectancy_audit, dict):
        policy_exp = dict(expectancy_audit.get("policy_expectancy", {}) or {})
        all_exp = dict(expectancy_audit.get("all_expectancy", {}) or {})
        out["fitness_expectancy_schema_version"] = int(expectancy_audit.get("schema_version", 0) or 0)
        out["fitness_policy_origin_lifecycle_count"] = int(expectancy_audit.get("policy_origin_lifecycle_count", 0) or 0)
        out["fitness_all_origin_lifecycle_count"] = int(expectancy_audit.get("all_origin_lifecycle_count", 0) or 0)
        out["fitness_policy_origin_ratio"] = float(expectancy_audit.get("policy_origin_ratio", 0.0) or 0.0)
        out["fitness_policy_expectancy_lcb"] = float(policy_exp.get("expectancy_lcb", 0.0) or 0.0)
        out["fitness_policy_expectancy_mean"] = float(policy_exp.get("mean_net_r", 0.0) or 0.0)
        out["fitness_policy_expectancy_samples"] = int(policy_exp.get("sample_count", 0) or 0)
        out["fitness_policy_expectancy_eligible"] = bool(policy_exp.get("eligible", False))
        out["fitness_context_negative_critical"] = int(expectancy_audit.get("negative_critical_scope_count", 0) or 0)
        out["fitness_context_positive_eligible"] = int(expectancy_audit.get("positive_eligible_scope_count", 0) or 0)
        out["fitness_all_expectancy_lcb"] = float(all_exp.get("expectancy_lcb", 0.0) or 0.0)
    return out


def _mt5_best_fitness(result: dict[str, object]) -> float:
    best = result.get("best", {})
    try:
        return float(best.get("fitness", -float("inf"))) if isinstance(best, dict) else -float("inf")
    except (TypeError, ValueError):
        return -float("inf")


def _mt5_train_to_goal_progress_score(result: dict[str, object]) -> float:
    """Loose train-to-goal progress score used when hard promotion gates are disabled."""
    metrics = _mt5_best_metrics(result)
    if not metrics:
        return _mt5_best_fitness(result)
    fitness = _mt5_best_fitness(result)
    net_profit = _mt5_metric(metrics, "net_profit")
    profit_factor = _mt5_metric(metrics, "profit_factor")
    expectancy = _mt5_metric(metrics, "fitness_expectancy_lcb", _dict_expectancy(metrics))
    drawdown = max(
        _mt5_metric(metrics, "equity_drawdown_pct"),
        _mt5_metric(metrics, "balance_drawdown_pct"),
    )
    win_rate = _mt5_metric(metrics, "win_rate")
    # Lifecycle count is an evidence gate elsewhere, never a reward.  Adding
    # it to a soft progress score was a hidden pressure to overtrade.
    ledger_integrity = _mt5_metric(metrics, "fitness_ledger_integrity", 1.0)
    execution_failure = _mt5_metric(metrics, "fitness_execution_failure_rate", 0.0)
    immediate_exit = _mt5_metric(metrics, "fitness_immediate_exit_ratio", 0.0)
    return float(
        fitness
        + 0.03 * net_profit
        + 30.0 * np.tanh(profit_factor - 1.0)
        + 12.0 * expectancy
        + 0.25 * win_rate
        + 15.0 * ledger_integrity
        - 4.0 * drawdown
        - 30.0 * execution_failure
        - 20.0 * immediate_exit
    )


def _mt5_metric(metrics: dict[str, object], key: str, default: float = 0.0) -> float:
    try:
        return float(metrics.get(key, default))
    except (TypeError, ValueError):
        return default


def _mt5_goal_reached(
    metrics: dict[str, object],
    max_drawdown_pct: float,
    min_profit_factor: float,
    min_winrate: float,
    min_expectancy: float,
    min_daily_trades: int,
    backtest_days: int,
) -> bool:
    lifecycle_count = _mt5_metric(metrics, "fitness_lifecycle_count", _mt5_metric(metrics, "total_trades"))
    # Keep a sample floor, not a cadence target.  The former min_daily_trades
    # argument only contributes a bounded coverage hint for compatibility.
    cadence_hint = max(0, int(min_daily_trades)) * max(1, int(backtest_days))
    required = max(1, max(8, min(40, int(np.ceil(0.5 * cadence_hint)))) if cadence_hint else 8)
    expectancy = _mt5_metric(metrics, "fitness_expectancy_lcb", _dict_expectancy(metrics))
    ledger_integrity = _mt5_metric(metrics, "fitness_ledger_integrity", 0.0)
    has_ledger = "fitness_lifecycle_count" in metrics
    return (
        _mt5_metric(metrics, "net_profit") > 0.0
        and _mt5_metric(metrics, "profit_factor") >= min_profit_factor
        and _mt5_metric(metrics, "equity_drawdown_pct") <= max_drawdown_pct
        and lifecycle_count >= required
        and _mt5_metric(metrics, "win_rate") >= min_winrate
        and expectancy >= min_expectancy
        and (not has_ledger or ledger_integrity >= 0.70)
    )



def mt5_minimum_evidence_rejection_reasons(
    trial_result: dict[str, object],
    *,
    min_daily_trades: int = 0,
    backtest_days: int = 31,
) -> list[str]:
    """Reject candidates that do not contain enough verified execution evidence.

    This is intentionally narrower than the full promotion gate. Exploratory
    callers may still compare a soft progress score, but they can never replace
    a seed with a policy evaluated on a handful of forced or incomplete
    lifecycles. It prevents a collapse in trade coverage from masquerading as
    progress.
    """
    metrics = _mt5_best_metrics(trial_result)
    lifecycle_count = _mt5_metric(metrics, "fitness_lifecycle_count", _mt5_metric(metrics, "total_trades"))
    cadence_hint = max(0, int(min_daily_trades)) * max(1, int(backtest_days))
    required = max(8, min(40, int(np.ceil(0.5 * cadence_hint)))) if cadence_hint else 8
    reasons: list[str] = []
    if lifecycle_count <= 0:
        reasons.append("no_finalised_position_lifecycles")
    elif lifecycle_count < required:
        reasons.append(f"insufficient_position_samples:{lifecycle_count:g}<{required:g}")
    if not bool(metrics.get("fitness_reconciled", True)):
        details = metrics.get("fitness_reconciliation_reasons", [])
        rendered = ";".join(str(item) for item in details) if isinstance(details, (list, tuple)) else str(details)
        reasons.append(f"report_execution_mismatch:{rendered or 'unknown'}")
    if "fitness_lifecycle_count" not in metrics:
        reasons.append("execution_ledger_missing")
    if _mt5_metric(metrics, "fitness_ledger_integrity", 0.0) < 0.70:
        reasons.append("ledger_integrity_below_0.70")
    # Concrete evaluator metrics must prove real-tick/fidelity parity.  Legacy
    # synthetic test fixtures that do not carry these fields are not treated as
    # evidence either way; production metrics always include both keys.
    if "tester_model" in metrics and bool(metrics.get("require_real_ticks_for_promotion", True)) and int(metrics.get("tester_model", -1)) != MT5_MODEL_REAL_TICKS:
        reasons.append("tester_model_not_real_ticks")
    if "real_tick_evidence_verified" in metrics and bool(metrics.get("require_real_tick_evidence_for_promotion", True)) and not bool(metrics.get("real_tick_evidence_verified", False)):
        reasons.append(f"real_tick_evidence_not_proven:{metrics.get('real_tick_evidence_reason', 'unknown')}")
    if "require_terminal_flat_for_promotion" in metrics and bool(metrics.get("require_terminal_flat_for_promotion", True)) and int(metrics.get("fitness_terminal_open_positions", 0) or 0) > 0:
        reasons.append(f"terminal_open_positions_nonzero:{int(metrics.get('fitness_terminal_open_positions', 0) or 0)}")
    if "feature_parity_numerical_valid" in metrics and not bool(metrics.get("feature_parity_numerical_valid", False)):
        reasons.append("numerical_feature_parity_not_proven")
    return reasons


def _expectancy_promotion_controls(config: object) -> tuple[float, float]:
    """Read V13 promotion thresholds without breaking legacy test/config shims."""
    section = getattr(config, "expectancy", None)
    ratio = getattr(section, "min_policy_origin_ratio", 0.60) if section is not None else 0.60
    delta = getattr(section, "promotion_lcb_min_delta", 0.01) if section is not None else 0.01
    try:
        ratio_value = float(ratio)
    except (TypeError, ValueError):
        ratio_value = 0.60
    try:
        delta_value = float(delta)
    except (TypeError, ValueError):
        delta_value = 0.01
    return float(np.clip(ratio_value, 0.0, 1.0)), max(0.0, delta_value)


def mt5_promotion_rejection_reasons(
    baseline_result: dict[str, object],
    trial_result: dict[str, object],
    max_drawdown_pct: float = 15.0,
    min_daily_trades: int = 0,
    backtest_days: int = 31,
    min_policy_origin_ratio: float = 0.60,
    promotion_lcb_min_delta: float = 0.01,
) -> list[str]:
    min_policy_origin_ratio = float(np.clip(min_policy_origin_ratio, 0.0, 1.0))
    promotion_lcb_min_delta = max(0.0, float(promotion_lcb_min_delta))
    reasons: list[str] = []
    baseline_fitness = _mt5_best_fitness(baseline_result)
    trial_fitness = _mt5_best_fitness(trial_result)
    baseline_metrics = _mt5_best_metrics(baseline_result)
    trial_metrics = _mt5_best_metrics(trial_result)
    if trial_fitness <= baseline_fitness:
        reasons.append(f"fitness_not_improved:{trial_fitness:.6g}<={baseline_fitness:.6g}")
    lifecycle_count = _mt5_metric(trial_metrics, "fitness_lifecycle_count", _mt5_metric(trial_metrics, "total_trades"))
    cadence_hint = max(0, int(min_daily_trades)) * max(1, int(backtest_days))
    required = max(8, min(40, int(np.ceil(0.5 * cadence_hint)))) if cadence_hint else 8
    if lifecycle_count <= 0:
        reasons.append("no_finalised_position_lifecycles")
    elif lifecycle_count < required:
        reasons.append(f"insufficient_position_samples:{lifecycle_count:g}<{required:g}")
    if not bool(trial_metrics.get("fitness_reconciled", True)):
        details = trial_metrics.get("fitness_reconciliation_reasons", [])
        rendered = ";".join(str(item) for item in details) if isinstance(details, (list, tuple)) else str(details)
        reasons.append(f"report_execution_mismatch:{rendered or 'unknown'}")
    if "fitness_lifecycle_count" not in trial_metrics:
        reasons.append("execution_ledger_missing")
    if _mt5_metric(trial_metrics, "fitness_ledger_integrity", 0.0) < 0.70:
        reasons.append("ledger_integrity_below_0.70")
    if "tester_model" in trial_metrics and bool(trial_metrics.get("require_real_ticks_for_promotion", True)) and int(trial_metrics.get("tester_model", -1)) != MT5_MODEL_REAL_TICKS:
        reasons.append("tester_model_not_real_ticks")
    if "real_tick_evidence_verified" in trial_metrics and bool(trial_metrics.get("require_real_tick_evidence_for_promotion", True)) and not bool(trial_metrics.get("real_tick_evidence_verified", False)):
        reasons.append(f"real_tick_evidence_not_proven:{trial_metrics.get('real_tick_evidence_reason', 'unknown')}")
    if "require_terminal_flat_for_promotion" in trial_metrics and bool(trial_metrics.get("require_terminal_flat_for_promotion", True)) and int(trial_metrics.get("fitness_terminal_open_positions", 0) or 0) > 0:
        reasons.append(f"terminal_open_positions_nonzero:{int(trial_metrics.get('fitness_terminal_open_positions', 0) or 0)}")
    if "feature_parity_numerical_valid" in trial_metrics and not bool(trial_metrics.get("feature_parity_numerical_valid", False)):
        reasons.append("numerical_feature_parity_not_proven")
    if _mt5_metric(trial_metrics, "fitness_immediate_exit_ratio", 0.0) > 0.90:
        reasons.append("immediate_exit_ratio_above_0.90")
    if _mt5_metric(trial_metrics, "fitness_execution_failure_rate", 0.0) > 0.20:
        reasons.append("execution_failure_rate_above_0.20")
    if _mt5_metric(trial_metrics, "profit_factor") < 1.0:
        reasons.append("profit_factor_below_1")
    if _mt5_metric(trial_metrics, "net_profit") <= 0.0:
        reasons.append("net_profit_not_positive")
    if _mt5_metric(trial_metrics, "equity_drawdown_pct") > max_drawdown_pct:
        reasons.append(f"drawdown_above_{max_drawdown_pct:g}pct")
    if baseline_metrics and _mt5_metric(trial_metrics, "profit_factor") < _mt5_metric(baseline_metrics, "profit_factor"):
        reasons.append("profit_factor_worse_than_baseline")
    # Graph candidates must improve execution quality too, not just exploit a
    # noisy scalar fitness. Small tolerance permits normal MT5 rounding.
    if baseline_metrics:
        if _mt5_metric(trial_metrics, "fitness_immediate_exit_ratio", 0.0) > _mt5_metric(baseline_metrics, "fitness_immediate_exit_ratio", 0.0) + 0.05:
            reasons.append("immediate_exit_ratio_worse_than_baseline")
        if _mt5_metric(trial_metrics, "fitness_expectancy_lcb", -float("inf")) < _mt5_metric(baseline_metrics, "fitness_expectancy_lcb", -float("inf")) - 0.02:
            reasons.append("expectancy_lcb_worse_than_baseline")
        if _mt5_metric(trial_metrics, "fitness_execution_failure_rate", 0.0) > _mt5_metric(baseline_metrics, "fitness_execution_failure_rate", 0.0) + 0.03:
            reasons.append("execution_failure_rate_worse_than_baseline")

    # V13 hard evidence gate: a global score or bootstrap-derived profit cannot
    # promote a graph.  Once the evaluator supplies expectancy audit fields,
    # candidate promotion requires robust *policy-origin* net-R edge and must
    # not create an additional statistically negative context/action scope.
    audit_available = "fitness_expectancy_schema_version" in trial_metrics
    if audit_available:
        min_policy_samples = max(3, int(np.ceil(min_policy_origin_ratio * required)))
        policy_samples = _mt5_metric(trial_metrics, "fitness_policy_origin_lifecycle_count", 0.0)
        policy_ratio = _mt5_metric(trial_metrics, "fitness_policy_origin_ratio", 0.0)
        policy_lcb = _mt5_metric(trial_metrics, "fitness_policy_expectancy_lcb", -float("inf"))
        if policy_samples < min_policy_samples:
            reasons.append(f"policy_origin_samples_insufficient:{policy_samples:g}<{min_policy_samples:g}")
        if policy_ratio < 0.0:
            reasons.append(f"policy_origin_ratio_below_{min_policy_origin_ratio:.2f}:{policy_ratio:.3f}")
        if policy_lcb <= -10.0:
            reasons.append(f"policy_origin_expectancy_lcb_nonpositive:{policy_lcb:.6g}")
        if _mt5_metric(trial_metrics, "fitness_context_positive_eligible", 0.0) <= 0.0:
            reasons.append("no_positive_context_action_expectancy_scope")
        baseline_negative = _mt5_metric(baseline_metrics, "fitness_context_negative_critical", 0.0) if baseline_metrics else 0.0
        trial_negative = _mt5_metric(trial_metrics, "fitness_context_negative_critical", 0.0)
        if trial_negative > baseline_negative:
            reasons.append("negative_context_action_scope_increased")
        if baseline_metrics and "fitness_expectancy_schema_version" in baseline_metrics:
            baseline_policy_lcb = _mt5_metric(baseline_metrics, "fitness_policy_expectancy_lcb", -float("inf"))
            # Require a material conservative edge improvement, not just a
            # floating-point tie on a global trial score.
            if policy_lcb <= baseline_policy_lcb + promotion_lcb_min_delta:
                reasons.append("policy_origin_expectancy_lcb_not_materially_improved")
    return reasons

def validate_replay_candidate_mt5(
    candidate: SelfGrowingPolicy,
    config: AppConfig,
    baseline_result: dict[str, object],
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    max_drawdown_pct: float = 50.0,
    from_date: str | None = None,
    to_date: str | None = None,
    model: int = 0,
    backtest_days: int = 31,
    experience_journal_path: str | Path | None = None,
) -> dict[str, object]:
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=Path(runner_dir),
            from_date=from_date or "2025.01.01",
            to_date=to_date or "2025.02.01",
            timeout_sec=timeout_sec,
            max_positions=int(config.risk.max_positions),
            max_holding_bars=int(config.orders.max_holding_bars),
            min_daily_trades=config.orders.min_daily_trades,
            max_daily_trades=config.orders.max_daily_trades,
            **_mt5_execution_contract_kwargs(config),
            visual=visual,
            shutdown_terminal=shutdown_terminal,
            model=model,
            # Replay validation is a measurement only.  It must never mutate
            # memory/topology, synthesize bootstrap orders, or emit a journal
            # that a later candidate can replay as training evidence.
            learning_enabled=False,
            experience_journal=None,
        )
    )
    # Validation must be a deep frozen clone.  Never let a replay candidate
    # mutate through a shallow ``copy`` or write adaptive journal feedback.
    frozen_candidate = clone_policy(candidate)
    fitness, metrics = evaluator.evaluate_policy(9001, frozen_candidate, 0)
    metrics_dict = _mt5_metrics_to_dict(metrics)
    result = {"best": {"fitness": fitness, "metrics": metrics_dict}}
    promotion_ratio, promotion_delta = _expectancy_promotion_controls(config)
    reject_reasons = mt5_promotion_rejection_reasons(
        baseline_result,
        result,
        max_drawdown_pct=max_drawdown_pct,
        min_daily_trades=config.orders.min_daily_trades,
        backtest_days=backtest_days,
        min_policy_origin_ratio=promotion_ratio,
        promotion_lcb_min_delta=promotion_delta,
    )
    return {
        "promoted": not reject_reasons,
        "reject_reasons": reject_reasons,
        "score": fitness,
        "best": result["best"],
        "result": result,
    }


def derived_compare_runner_dirs(base_runner_dir: str | Path) -> dict[str, Path]:
    base = Path(base_runner_dir)
    return {
        "current": base.with_name(f"{base.name}_current"),
        "world": base.with_name(f"{base.name}_world"),
    }


def _runner_copy_ignore(_dir: str, names: list[str]) -> set[str]:
    ignored = set()
    transient_dirs = {"Reports", "logs", "Logs", "cache", "__pycache__", "self_growing_mt5_train"}
    transient_suffixes = (".log", ".tst", ".htm", ".html", ".tmp")
    transient_files = {
        "self_growing_request.csv",
        "self_growing_action.csv",
        "self_growing_bridge.csv",
    }
    for name in names:
        if name in transient_dirs or name in transient_files or name.endswith(transient_suffixes):
            ignored.add(name)
    return ignored


def ensure_compare_runner_dirs(
    base_runner_dir: str | Path,
    runner_dir_current: str | Path | None = None,
    runner_dir_world: str | Path | None = None,
) -> dict[str, Path]:
    base = Path(base_runner_dir)
    derived = derived_compare_runner_dirs(base)
    runners = {
        "current": Path(runner_dir_current) if runner_dir_current else derived["current"],
        "world": Path(runner_dir_world) if runner_dir_world else derived["world"],
    }
    if runners["current"].resolve() == runners["world"].resolve():
        raise ValueError("compare-cores requires distinct runner dirs for current and world cores.")
    for core, path in runners.items():
        if path.exists():
            continue
        if not base.exists():
            raise FileNotFoundError(f"Cannot prepare {core} runner; base MT5 runner not found: {base}")
        shutil.copytree(base, path, ignore=_runner_copy_ignore)
    return runners


def _result_entry(core: str, path: Path, result: dict[str, object] | None = None, error: Exception | None = None) -> dict[str, object]:
    if error is not None:
        return {
            "core": core,
            "output": str(path),
            "ok": False,
            "error": str(error),
            "fitness": None,
            "best": None,
            "result": None,
        }
    assert result is not None
    return {
        "core": core,
        "output": str(path),
        "ok": True,
        "error": None,
        "fitness": _mt5_best_fitness(result),
        "best": result.get("best"),
        "result": result,
    }


def _exchange_positive_replay_between_cores(
    config: AppConfig,
    target_core: str,
    target_path: Path,
    source_core: str,
    source_path: Path,
    baseline_result: dict[str, object],
    output_path: Path,
    runner_dir: Path,
    exchange_replay_max: int,
    timeout_sec: int,
    visual: bool,
    shutdown_terminal: bool,
    max_drawdown_pct: float,
    from_date: str | None,
    to_date: str | None,
    model: int,
) -> dict[str, object]:
    if not target_path.exists() or not source_path.exists():
        return {
            "target_core": target_core,
            "source_core": source_core,
            "promoted": False,
            "replayed_experiences": 0,
            "reject_reasons": ["missing_policy_artifact"],
            "score": None,
            "output": str(output_path),
            "validation": None,
        }
    target = load_policy(target_path, policy_core=target_core)
    source = load_policy(source_path, policy_core=source_core)
    before_modules = [module.name for module in target.modules]
    replayed = target.replay_stratified_experiences_from(source, max_items=exchange_replay_max, seed=int(getattr(getattr(config, "learning", None), "seed", 0)))
    if [module.name for module in target.modules] != before_modules:
        return {
            "target_core": target_core,
            "source_core": source_core,
            "promoted": False,
            "replayed_experiences": replayed,
            "reject_reasons": ["structure_changed_during_exchange"],
            "score": None,
            "output": str(output_path),
            "validation": None,
        }
    if replayed <= 0:
        return {
            "target_core": target_core,
            "source_core": source_core,
            "promoted": False,
            "replayed_experiences": 0,
            "reject_reasons": ["no_positive_experiences"],
            "score": None,
            "output": str(output_path),
            "validation": None,
        }
    if from_date and to_date:
        backtest_days = max(1, (datetime.strptime(to_date, "%Y.%m.%d") - datetime.strptime(from_date, "%Y.%m.%d")).days)
    else:
        backtest_days = 31
    try:
        validation = validate_replay_candidate_mt5(
            target,
            config,
            baseline_result,
            runner_dir=runner_dir,
            timeout_sec=timeout_sec,
            visual=visual,
            shutdown_terminal=shutdown_terminal,
            max_drawdown_pct=max_drawdown_pct,
            from_date=from_date,
            to_date=to_date,
            model=model,
            backtest_days=backtest_days,
        )
    except Exception as exc:
        return {
            "target_core": target_core,
            "source_core": source_core,
            "promoted": False,
            "replayed_experiences": replayed,
            "reject_reasons": [f"exchange_validation_error:{exc}"],
            "score": None,
            "output": str(output_path),
            "validation": None,
        }
    if bool(validation["promoted"]):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as handle:
            pickle.dump(target, handle)
    return {
        "target_core": target_core,
        "source_core": source_core,
        "promoted": bool(validation["promoted"]),
        "replayed_experiences": replayed,
        "reject_reasons": list(validation["reject_reasons"]),
        "score": validation["score"],
        "output": str(output_path),
        "validation": validation,
    }


def compare_policy_cores_mt5(
    config: AppConfig,
    rounds: int,
    output_dir: str | Path = "outputs/compare_cores",
    trials: int = 10,
    runner_dir: str | Path = "mt5_v2_runner",
    runner_dir_current: str | Path | None = None,
    runner_dir_world: str | Path | None = None,
    current_seed: str | Path | None = None,
    world_seed: str | Path | None = None,
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    curriculum_learning: bool = False,
    from_date: str | None = None,
    to_date: str | None = None,
    mutation_sigma: float = 0.08,
    max_modules: int = 4,
    model: int = 0,
    exchange_replay_max: int = 200,
    skip_exchange: bool = False,
    max_drawdown_pct: float = 50.0,
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None = "outputs/mt5_experience_journal.jsonl",
    replay_journal_max: int = 64,
) -> dict[str, object]:
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if exchange_replay_max < 0:
        raise ValueError("exchange_replay_max must be >= 0")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runners = ensure_compare_runner_dirs(runner_dir, runner_dir_current, runner_dir_world)
    outputs = {
        "current": out_dir / "current.pkl",
        "world": out_dir / "world.pkl",
    }
    seeds = {
        "current": current_seed,
        "world": world_seed,
    }
    # A compare run must not let the two policy cores learn from the same
    # append-only journal.  They may exchange only the explicitly validated
    # positive replay path below.
    if experience_journal_path:
        requested_journal = Path(experience_journal_path)
        suffix = requested_journal.suffix or ".jsonl"
        journal_dir = out_dir / "journals"
        journal_dir.mkdir(parents=True, exist_ok=True)
        journals = {
            core: journal_dir / f"{requested_journal.stem}.{core}{suffix}"
            for core in ("current", "world")
        }
    else:
        journals = {"current": None, "world": None}

    def run_core(core: str) -> dict[str, object]:
        return train_self_growing_mt5(
            config,
            rounds,
            str(outputs[core]),
            trials=trials,
            seed_policy_path=seeds[core],
            runner_dir=runners[core],
            timeout_sec=timeout_sec,
            visual=visual,
            shutdown_terminal=shutdown_terminal,
            curriculum_learning=curriculum_learning,
            from_date=from_date,
            to_date=to_date,
            mutation_sigma=mutation_sigma,
            max_modules=max_modules,
            model=model,
            policy_core=core,
            mt5_close_scope=mt5_close_scope,
            experience_journal_path=journals[core],
            replay_journal_max=replay_journal_max,
        )

    core_results: dict[str, dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(run_core, core): core for core in ("current", "world")}
        for future in as_completed(futures):
            core = futures[future]
            try:
                core_results[core] = _result_entry(core, outputs[core], future.result())
            except Exception as exc:
                core_results[core] = _result_entry(core, outputs[core], error=exc)

    exchange: dict[str, object] = {"skipped": bool(skip_exchange), "items": []}
    candidate_entries: list[dict[str, object]] = [
        entry for entry in core_results.values() if bool(entry.get("ok"))
    ]
    if not skip_exchange and all(bool(core_results.get(core, {}).get("ok")) for core in ("current", "world")):
        exchange_outputs = {
            "current": out_dir / "current.exchanged.pkl",
            "world": out_dir / "world.exchanged.pkl",
        }
        exchange_items = [
            _exchange_positive_replay_between_cores(
                config,
                "current",
                outputs["current"],
                "world",
                outputs["world"],
                core_results["current"]["result"],
                exchange_outputs["current"],
                runners["current"],
                exchange_replay_max,
                timeout_sec,
                visual,
                shutdown_terminal,
                max_drawdown_pct,
                from_date,
                to_date,
                model,
            ),
            _exchange_positive_replay_between_cores(
                config,
                "world",
                outputs["world"],
                "current",
                outputs["current"],
                core_results["world"]["result"],
                exchange_outputs["world"],
                runners["world"],
                exchange_replay_max,
                timeout_sec,
                visual,
                shutdown_terminal,
                max_drawdown_pct,
                from_date,
                to_date,
                model,
            ),
        ]
        exchange["items"] = exchange_items
        for item in exchange_items:
            if bool(item.get("promoted")):
                candidate_entries.append({
                    "core": item["target_core"],
                    "output": item["output"],
                    "ok": True,
                    "error": None,
                    "fitness": float(item["score"]),
                    "best": item.get("validation", {}).get("best") if isinstance(item.get("validation"), dict) else None,
                    "result": item.get("validation", {}).get("result") if isinstance(item.get("validation"), dict) else None,
                    "source": "exchange",
                })
    elif not skip_exchange:
        exchange["skipped"] = True
        exchange["reason"] = "requires_both_cores_successful"

    winner = max(candidate_entries, key=lambda entry: float(entry.get("fitness", -float("inf"))), default=None)
    best_output = out_dir / "best.pkl"
    if winner and Path(str(winner["output"])).exists():
        shutil.copy2(str(winner["output"]), best_output)
    summary = {
        "command": "compare-cores",
        "rounds": int(rounds),
        "trials": int(trials),
        "runner_dirs": {core: str(path) for core, path in runners.items()},
        "experience_journals": {core: (str(path) if path is not None else None) for core, path in journals.items()},
        "outputs": {core: str(path) for core, path in outputs.items()} | {"best": str(best_output)},
        "current": core_results.get("current"),
        "world": core_results.get("world"),
        "exchange": exchange,
        "winner": winner,
        "best_output": str(best_output) if winner else None,
    }
    summary_path = out_dir / "compare_cores.json"
    summary["summary_path"] = str(summary_path)
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, default=str)
    return summary


def improve_self_growing(
    config: AppConfig,
    rounds: int,
    seed_policy_path: str,
    output_path: str,
    trial_path: str | None = None,
    attempts: int = 1,
    trials: int = 10,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    curriculum_learning: bool = False,
    max_drawdown_pct: float = 50.0,
    from_date: str | None = None,
    to_date: str | None = None,
    model: int = 0,
    policy_core: str = "current",
    m1_backend: str = "direct_rnn",
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None = "outputs/mt5_experience_journal.jsonl",
    replay_journal_max: int = 64,
    mutation_sigma: float = 0.08,
    max_modules: int | None = None,
    enforce_promotion_gate: bool = True,
) -> dict[str, object]:
    core = _normalize_policy_core(policy_core)
    seed_path = Path(seed_policy_path)
    if not seed_path.exists():
        raise FileNotFoundError(f"Seed policy not found: {seed_path}")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_eval_path = output.with_name(f"{output.stem}.baseline.mt5.pkl")
    baseline_result = train_self_growing_mt5(
        config, 1, str(baseline_eval_path), 1, seed_path, runner_dir, timeout_sec, visual, shutdown_terminal, False,
        from_date=from_date, to_date=to_date, model=model, policy_core=core, m1_backend=m1_backend, mt5_close_scope=mt5_close_scope,
        experience_journal_path=experience_journal_path, replay_journal_max=replay_journal_max,
        mutation_sigma=mutation_sigma, max_modules=max_modules,
        enforce_promotion_gate=enforce_promotion_gate,
    )
    if baseline_eval_path.exists():
        baseline_eval_path.unlink()
    best_trial: Path | None = None
    best_result: dict[str, object] | None = None
    attempts_summary: list[dict[str, object]] = []
    for attempt in range(1, attempts + 1):
        trial = Path(trial_path) if trial_path and attempts == 1 else output.with_name(f"{output.stem}.trial.{attempt}.pkl")
        trial_result = train_self_growing_mt5(
            config, rounds, str(trial), trials, seed_path, runner_dir, timeout_sec, visual, shutdown_terminal, curriculum_learning,
            from_date=from_date, to_date=to_date, include_seed_trial=False, model=model, policy_core=core, m1_backend=m1_backend, mt5_close_scope=mt5_close_scope,
            experience_journal_path=experience_journal_path, replay_journal_max=replay_journal_max,
            mutation_sigma=mutation_sigma, max_modules=max_modules,
            enforce_promotion_gate=enforce_promotion_gate,
        )
        trial_score = _mt5_best_fitness(trial_result)
        attempts_summary.append({"attempt": attempt, "trial": str(trial), "score": trial_score, "best": trial_result.get("best")})
        if best_result is None or trial_score > _mt5_best_fitness(best_result):
            if best_trial is not None and best_trial != trial and best_trial.exists():
                best_trial.unlink()
            best_trial = trial
            best_result = trial_result
        elif trial != output and trial.exists():
            trial.unlink()
    if best_trial is None or best_result is None:
        raise RuntimeError("Improve did not produce a trial.")
    if from_date and to_date:
        backtest_days = max(1, (datetime.strptime(to_date, "%Y.%m.%d") - datetime.strptime(from_date, "%Y.%m.%d")).days)
    else:
        backtest_days = 31
    if enforce_promotion_gate:
        promotion_ratio, promotion_delta = _expectancy_promotion_controls(config)
        reject_reasons = mt5_promotion_rejection_reasons(
            baseline_result,
            best_result,
            max_drawdown_pct=max_drawdown_pct,
            min_daily_trades=config.orders.min_daily_trades,
            backtest_days=backtest_days,
            min_policy_origin_ratio=promotion_ratio,
            promotion_lcb_min_delta=promotion_delta,
        )
        promoted = not reject_reasons
    else:
        # Disabling the full gate is allowed only for exploratory selection.
        # Minimum verified execution evidence remains mandatory; otherwise a
        # low-trade candidate can win a soft score by simply avoiding exposure.
        reject_reasons = mt5_minimum_evidence_rejection_reasons(
            best_result,
            min_daily_trades=config.orders.min_daily_trades,
            backtest_days=backtest_days,
        )
        baseline_progress = _mt5_train_to_goal_progress_score(baseline_result)
        trial_progress = _mt5_train_to_goal_progress_score(best_result)
        promoted = not reject_reasons and bool(trial_progress > baseline_progress + 1e-9)
        if not reject_reasons and not promoted:
            reject_reasons.append(f"goal_progress_not_improved:{trial_progress:.6g}<={baseline_progress:.6g}")
    rejected_path: Path | None = None
    replayed_experiences = 0
    replay_promoted = False
    replay_reject_reasons: list[str] = []
    replay_score: float | None = None
    replay_validation: dict[str, object] | None = None
    if promoted:
        shutil.copy2(best_trial, output)
    else:
        rejected_path = output.with_name(f"{output.stem}.rejected_best.pkl")
        shutil.copy2(best_trial, rejected_path)
        replay_target = load_policy(seed_path, policy_core=core)
        replay_source = load_policy(best_trial, policy_core=core)
        replayed_experiences = replay_target.replay_stratified_experiences_from(replay_source, seed=int(getattr(getattr(config, "learning", None), "seed", 0)))
        if replayed_experiences > 0:
            try:
                replay_validation = validate_replay_candidate_mt5(
                    replay_target,
                    config,
                    baseline_result,
                    runner_dir=runner_dir,
                    timeout_sec=timeout_sec,
                    visual=visual,
                    shutdown_terminal=shutdown_terminal,
                    max_drawdown_pct=max_drawdown_pct,
                    from_date=from_date,
                    to_date=to_date,
                    model=model,
                    backtest_days=backtest_days,
                )
            except Exception as exc:
                replay_validation = {
                    "promoted": False,
                    "reject_reasons": [f"replay_validation_error:{exc}"],
                    "score": None,
                    "best": None,
                    "result": None,
                }
            replay_promoted = bool(replay_validation["promoted"])
            replay_reject_reasons = list(replay_validation["reject_reasons"])
            replay_score = float(replay_validation["score"]) if replay_validation["score"] is not None else None
            if replay_promoted:
                with output.open("wb") as handle:
                    pickle.dump(replay_target, handle)
            elif seed_path != output:
                shutil.copy2(seed_path, output)
        else:
            if seed_path != output:
                shutil.copy2(seed_path, output)
    if best_trial != output and best_trial.exists():
        best_trial.unlink()
    baseline_score = _mt5_best_fitness(baseline_result)
    trial_score = _mt5_best_fitness(best_result)
    baseline_progress_score = _mt5_train_to_goal_progress_score(baseline_result)
    trial_progress_score = _mt5_train_to_goal_progress_score(best_result)
    effective_score = replay_score if replay_promoted and replay_score is not None else trial_score
    effective_improved = bool(promoted or replay_promoted)
    outcome_starvation = apply_outcome_starvation_to_policy_file(
        output,
        policy_core=core,
        goal_hit=None,
        improved=effective_improved,
        fitness_delta=float(effective_score) - float(baseline_score),
        reason="improve_promoted" if effective_improved else "improve_not_promoted",
        mutate_policy=bool(effective_improved),
    )
    return {
        "promoted": promoted,
        "promotion_gate_enabled": bool(enforce_promotion_gate),
        "reject_reasons": reject_reasons,
        "rejected_trial": str(rejected_path) if rejected_path else None,
        "seed_policy": str(seed_path),
        "trial": str(best_trial),
        "output": str(output),
        "baseline_score": baseline_score,
        "trial_score": trial_score,
        "baseline_progress_score": baseline_progress_score,
        "trial_progress_score": trial_progress_score,
        "replayed_experiences": replayed_experiences,
        "replay_promoted": replay_promoted,
        "replay_reject_reasons": replay_reject_reasons,
        "replay_score": replay_score,
        "outcome_starvation": outcome_starvation,
        "replay_validation": replay_validation,
        "replay_source": str(rejected_path) if rejected_path else None,
        "attempts": attempts_summary,
        "baseline": baseline_result.get("best"),
        "trial_summary": best_result.get("best"),
        "baseline_result": baseline_result,
        "trial_result": best_result,
        "policy_core": core,
    }


def ensure_mt5_seed_policy(
    config: AppConfig,
    seed_policy_path: str | Path,
    output_path: str | Path,
    *,
    rounds: int,
    trials: int,
    runner_dir: str | Path,
    timeout_sec: int,
    visual: bool,
    shutdown_terminal: bool,
    curriculum_learning: bool,
    from_date: str | None,
    to_date: str | None,
    model: int,
    policy_core: str,
    m1_backend: str = "direct_rnn",
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None,
    replay_journal_max: int,
) -> tuple[str, dict[str, object] | None]:
    seed_path = Path(seed_policy_path)
    if seed_path.exists():
        return str(seed_path), None
    bootstrap_output = Path(output_path)
    bootstrap_output.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Train] seed policy not found for goal workflow: {seed_path}; bootstrapping {bootstrap_output}", flush=True)
    result = train_self_growing_mt5(
        config,
        max(1, rounds),
        str(bootstrap_output),
        trials=max(1, trials),
        seed_policy_path=None,
        runner_dir=runner_dir,
        timeout_sec=timeout_sec,
        visual=visual,
        shutdown_terminal=shutdown_terminal,
        curriculum_learning=curriculum_learning,
        from_date=from_date,
        to_date=to_date,
        model=model,
        policy_core=policy_core,
        m1_backend=m1_backend,
        mt5_close_scope=mt5_close_scope,
        experience_journal_path=experience_journal_path,
        replay_journal_max=replay_journal_max,
    )
    return str(bootstrap_output), {"bootstrap_seed": str(seed_path), "output": str(bootstrap_output), "result": result}


def apply_outcome_starvation_to_policy_file(
    policy_path: str | Path,
    *,
    policy_core: str = "current",
    goal_hit: bool | None = None,
    improved: bool | None = None,
    fitness_delta: float = 0.0,
    reason: str = "",
    mutate_policy: bool = False,
) -> dict[str, float]:
    """Apply or preview outcome-starvation feedback without poisoning a seed.

    Failed/rejected trials may be useful diagnostics but they must not mutate a
    production seed that will be reused by the next round. ``mutate_policy`` is
    therefore explicit and defaults to ``False``; callers may persist this
    feedback only after a real promotion/goal acceptance.
    """
    path = Path(policy_path)
    if not path.exists():
        return {}
    policy = load_policy(path, policy_core=policy_core)
    if not hasattr(policy, "apply_outcome_starvation"):
        return {}
    target = policy if bool(mutate_policy) else clone_policy(policy)
    starvation = target.apply_outcome_starvation(
        goal_hit=goal_hit,
        improved=improved,
        fitness_delta=fitness_delta,
        reason=reason,
    )
    result = {str(key): float(value) for key, value in dict(starvation or {}).items()}
    result["persisted"] = 1.0 if bool(mutate_policy) else 0.0
    if mutate_policy:
        with path.open("wb") as handle:
            pickle.dump(target, handle)
    return result


def auto_improve_self_growing(
    config: AppConfig,
    train_rounds: int,
    seed_policy_path: str,
    output_path: str,
    improve_rounds: int,
    attempts: int = 1,
    log_path: str | None = None,
    trials: int = 10,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    curriculum_learning: bool = False,
    max_drawdown_pct: float = 50.0,
    from_date: str | None = None,
    to_date: str | None = None,
    model: int = 0,
    policy_core: str = "current",
    m1_backend: str = "direct_rnn",
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None = "outputs/mt5_experience_journal.jsonl",
    replay_journal_max: int = 64,
) -> dict[str, object]:
    log_file = Path(log_path) if log_path else Path(output_path).with_suffix(".auto_improve.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    current_seed, bootstrap = ensure_mt5_seed_policy(
        config,
        seed_policy_path,
        output_path,
        rounds=train_rounds,
        trials=trials,
        runner_dir=runner_dir,
        timeout_sec=timeout_sec,
        visual=visual,
        shutdown_terminal=shutdown_terminal,
        curriculum_learning=curriculum_learning,
        from_date=from_date,
        to_date=to_date,
        model=model,
        policy_core=policy_core,
        m1_backend=m1_backend,
        mt5_close_scope=mt5_close_scope,
        experience_journal_path=experience_journal_path,
        replay_journal_max=replay_journal_max,
    )
    history = []
    if bootstrap is not None:
        bootstrap["round"] = 0
        history.append(bootstrap)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bootstrap, ensure_ascii=False, default=str) + "\n")
    promoted_count = 0
    for round_index in range(1, improve_rounds + 1):
        started = time.time()
        result = improve_self_growing(
            config, train_rounds, current_seed, output_path, attempts=attempts, trials=trials, runner_dir=runner_dir,
            timeout_sec=timeout_sec, visual=visual, shutdown_terminal=shutdown_terminal, curriculum_learning=curriculum_learning,
            max_drawdown_pct=max_drawdown_pct, from_date=from_date, to_date=to_date, model=model,
            policy_core=policy_core, m1_backend=m1_backend, mt5_close_scope=mt5_close_scope,
            experience_journal_path=experience_journal_path, replay_journal_max=replay_journal_max,
        )
        result["round"] = round_index
        result["elapsed_sec"] = float(time.time() - started)
        history.append(result)
        if bool(result["promoted"]):
            promoted_count += 1
        current_seed = output_path
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    return {"rounds_requested": improve_rounds, "rounds_completed": len(history), "promoted_count": promoted_count, "output": output_path, "log_path": str(log_file), "history": history, "last": history[-1], "bootstrap": bootstrap}


def train_to_goal(
    config: AppConfig,
    train_rounds: int,
    seed_policy_path: str,
    output_path: str,
    max_rounds: int = 100,
    attempts: int = 1,
    log_path: str | None = None,
    max_drawdown_pct: float = 50.0,
    min_profit_factor: float = 1.2,
    trials: int = 10,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    curriculum_learning: bool = False,
    min_winrate: float = 0.0,
    min_expectancy: float = 0.0,
    min_daily_trades: int | None = None,
    backtest_days: int = 31,
    from_date: str | None = None,
    to_date: str | None = None,
    model: int = 0,
    policy_core: str = "current",
    m1_backend: str = "direct_rnn",
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None = "outputs/mt5_experience_journal.jsonl",
    replay_journal_max: int = 64,
    mutation_sigma: float = 0.08,
    max_modules: int | None = None,
) -> dict[str, object]:
    log_file = Path(log_path) if log_path else Path(output_path).with_suffix(".train_to_goal.jsonl")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # A goal cannot be validated on a trivial handful of trades.  The regular
    # execution cadence is the default unless an explicit stricter value is
    # supplied by the caller.
    effective_min_daily_trades = int(config.orders.min_daily_trades) if min_daily_trades is None else max(0, int(min_daily_trades))
    current_seed, bootstrap = ensure_mt5_seed_policy(
        config,
        seed_policy_path,
        output_path,
        rounds=train_rounds,
        trials=trials,
        runner_dir=runner_dir,
        timeout_sec=timeout_sec,
        visual=visual,
        shutdown_terminal=shutdown_terminal,
        curriculum_learning=curriculum_learning,
        from_date=from_date,
        to_date=to_date,
        model=model,
        policy_core=policy_core,
        m1_backend=m1_backend,
        mt5_close_scope=mt5_close_scope,
        experience_journal_path=experience_journal_path,
        replay_journal_max=replay_journal_max,
    )
    history = []
    if bootstrap is not None:
        bootstrap["round"] = 0
        history.append(bootstrap)
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(bootstrap, ensure_ascii=False, default=str) + "\n")
    promoted_count = 0
    goal_reached = False
    for round_index in range(1, max_rounds + 1):
        result = improve_self_growing(
            config, train_rounds, current_seed, output_path, attempts=attempts, trials=trials, runner_dir=runner_dir,
            timeout_sec=timeout_sec, visual=visual, shutdown_terminal=shutdown_terminal, curriculum_learning=curriculum_learning,
            max_drawdown_pct=max_drawdown_pct, from_date=from_date, to_date=to_date, model=model,
            policy_core=policy_core, m1_backend=m1_backend, mt5_close_scope=mt5_close_scope,
            experience_journal_path=experience_journal_path, replay_journal_max=replay_journal_max,
            mutation_sigma=mutation_sigma, max_modules=max_modules,
            # Train-to-goal is a promotion workflow, not an unconstrained
            # explorer.  It must never advance the seed with a low-sample
            # candidate merely because a soft score looks less bad.
            enforce_promotion_gate=True,
        )
        result["promotion_gate_enabled"] = True
        result["round"] = round_index
        history.append(result)
        if bool(result["promoted"]):
            promoted_count += 1
            current_seed = output_path
        baseline_metrics = _mt5_best_metrics(result.get("baseline_result", {}))
        trial_metrics = _mt5_best_metrics(result.get("trial_result", {}))
        baseline_goal = _mt5_goal_reached(
            baseline_metrics,
            max_drawdown_pct,
            min_profit_factor,
            min_winrate,
            min_expectancy,
            effective_min_daily_trades,
            backtest_days,
        )
        trial_goal = bool(result.get("promoted")) and _mt5_goal_reached(
            trial_metrics,
            max_drawdown_pct,
            min_profit_factor,
            min_winrate,
            min_expectancy,
            effective_min_daily_trades,
            backtest_days,
        )
        goal_reached = baseline_goal or trial_goal
        result["goal_source"] = "baseline" if baseline_goal else ("trial_promoted" if trial_goal else None)
        result["goal_min_daily_trades"] = effective_min_daily_trades
        if baseline_goal and Path(current_seed) != Path(output_path):
            shutil.copy2(current_seed, output_path)
        baseline_score = float(result.get("baseline_score", -float("inf")))
        trial_score = float(result.get("trial_score", -float("inf")))
        goal_starvation = apply_outcome_starvation_to_policy_file(
            output_path,
            policy_core=policy_core,
            goal_hit=goal_reached,
            improved=trial_score > baseline_score,
            fitness_delta=trial_score - baseline_score,
            reason="goal_reached" if goal_reached else "goal_not_reached",
            mutate_policy=bool(trial_goal),
        )
        result["goal_starvation"] = goal_starvation
        current_seed = output_path
        with log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        if goal_reached:
            break
    return {"goal_reached": goal_reached, "max_rounds": max_rounds, "rounds_completed": len(history), "promoted_count": promoted_count, "output": output_path, "log_path": str(log_file), "history": history, "last": history[-1], "bootstrap": bootstrap}


def evaluate_policy_mt5_oos(
    candidate: SelfGrowingPolicy,
    config: AppConfig,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    from_date: str | None = None,
    to_date: str | None = None,
    model: int = 0,
    policy_id: int = 9102,
) -> dict[str, object]:
    """Evaluate a copy of a policy without training, replay, or journaling.

    Walk-forward validation must remain out-of-sample.  In particular this
    helper never calls ``train_self_growing_mt5`` and passes no journal to the
    evaluator, so an OOS result cannot mutate policy weights through replay.
    """
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=Path(runner_dir),
            from_date=from_date or "2025.01.01",
            to_date=to_date or "2025.02.01",
            timeout_sec=timeout_sec,
            max_positions=int(config.risk.max_positions),
            max_holding_bars=int(config.orders.max_holding_bars),
            min_daily_trades=int(config.orders.min_daily_trades),
            max_daily_trades=int(config.orders.max_daily_trades),
            **_mt5_execution_contract_kwargs(config),
            visual=visual,
            shutdown_terminal=shutdown_terminal,
            model=model,
            experience_journal=None,
            learning_enabled=False,
            require_real_tick_evidence_for_promotion=bool(getattr(getattr(config, "learning", None), "require_real_tick_evidence_for_promotion", True)),
            require_terminal_flat_for_promotion=bool(getattr(getattr(config, "learning", None), "require_terminal_flat_for_promotion", True)),
            continuous_runtime=None,
        )
    )
    before = policy_state_hash(candidate)
    evaluation_policy = clone_policy(candidate)
    reset_runtime = getattr(evaluation_policy, "reset_runtime_state", None)
    if callable(reset_runtime):
        reset_runtime()
    fitness, metrics = evaluator.evaluate_policy(int(policy_id), evaluation_policy, 0)
    after = policy_state_hash(candidate)
    if before != after:
        raise RuntimeError("frozen OOS evaluation mutated the source policy")
    return {
        "best": {"fitness": float(fitness), "metrics": _mt5_metrics_to_dict(metrics)},
        "evaluation_only": True,
        "frozen": True,
        "source_policy_state_hash_before": before,
        "source_policy_state_hash_after": after,
        "journal_replayed": 0,
    }


def walk_forward_windows(
    from_date: str,
    to_date: str,
    train_window_days: int,
    test_window_days: int | None = None,
    step_days: int | None = None,
) -> list[dict[str, str]]:
    """Tạo các cửa sổ walk-forward theo đúng thứ tự ``train → OOS test``.

    Phiên bản cũ chỉ sinh một cửa sổ ``train_window`` rồi cắt 80/20 nội bộ;
    ``test_window`` thực tế chỉ bị dùng làm bước nhảy. Điều đó vừa gây hiểu
    nhầm CLI vừa làm OOS ngắn hơn cấu hình người dùng. Hàm này tách hai đoạn
    thời gian rõ ràng và không cho OOS lọt vào pha train của chính cửa sổ đó.
    """
    try:
        start_dt = datetime.strptime(str(from_date), "%Y.%m.%d")
        end_dt = datetime.strptime(str(to_date), "%Y.%m.%d")
    except ValueError as exc:
        raise ValueError("--from-date va --to-date phai co dang YYYY.MM.DD") from exc

    train_days = int(train_window_days)
    test_days = int(test_window_days if test_window_days is not None else train_days)
    stride_days = int(step_days if step_days is not None else test_days)
    if train_days < 1:
        raise ValueError("--train-window phai >= 1 ngay")
    if test_days < 1:
        raise ValueError("--test-window phai >= 1 ngay")
    if stride_days < 1:
        raise ValueError("--step phai >= 1 ngay")
    if end_dt <= start_dt:
        raise ValueError("--to-date phai lon hon --from-date")

    windows: list[dict[str, str]] = []
    current = start_dt
    required_end = current + timedelta(days=train_days + test_days)
    while required_end <= end_dt:
        train_end = current + timedelta(days=train_days)
        test_end = train_end + timedelta(days=test_days)
        windows.append({
            "from_date": current.strftime("%Y.%m.%d"),
            "to_date": test_end.strftime("%Y.%m.%d"),
            "train_from": current.strftime("%Y.%m.%d"),
            "train_to": train_end.strftime("%Y.%m.%d"),
            "test_from": train_end.strftime("%Y.%m.%d"),
            "test_to": test_end.strftime("%Y.%m.%d"),
        })
        current += timedelta(days=stride_days)
        required_end = current + timedelta(days=train_days + test_days)
    return windows


def _kiem_tra_cua_so_walk_forward(
    *,
    from_date: str,
    to_date: str,
    train_window_days: int,
    test_window_days: int,
    step_days: int | None,
) -> list[dict[str, str]]:
    windows = walk_forward_windows(
        from_date,
        to_date,
        train_window_days,
        test_window_days,
        step_days,
    )
    if windows:
        return windows
    try:
        available_days = (datetime.strptime(to_date, "%Y.%m.%d") - datetime.strptime(from_date, "%Y.%m.%d")).days
    except ValueError:
        available_days = -1
    required_days = int(train_window_days) + int(test_window_days)
    raise ValueError(
        "Khong tao duoc cua so walk-forward: khoang ngay co "
        f"{available_days} ngay, can toi thieu {required_days} ngay "
        f"({int(train_window_days)} ngay train + {int(test_window_days)} ngay OOS test)."
    )


def _duong_dan_nhat_ky_walk_forward(output_path: Path, window_index: int, seed_index: int) -> str:
    """Không tái sử dụng journal giữa các ô OOS × seed."""
    return str(output_path.with_suffix(f".window_{window_index:03d}.seed_{seed_index:03d}.journal.jsonl"))


def _tong_ket_walk_forward(
    *,
    results: list[dict[str, object]],
    out_dir: Path,
    learning_cfg: object,
    artifact_key: str = "output",
) -> dict[str, object]:
    valid_oos = [float(row["val_fitness"]) for row in results if np.isfinite(float(row.get("val_fitness", -float("inf"))))]
    oos_arr = np.asarray(valid_oos, dtype=float) if valid_oos else np.array([], dtype=float)
    stability = promotion_stability_report(
        results,
        min_windows=int(getattr(learning_cfg, "promotion_min_oos_windows", 3)),
        min_seeds=int(getattr(learning_cfg, "promotion_min_seeds", 3)),
    )
    best_oos_row: dict[str, object] | None = None
    for row in results:
        if not bool(row.get("oos_valid", False)):
            continue
        if best_oos_row is None or float(row["val_fitness"]) > float(best_oos_row["val_fitness"]):
            best_oos_row = dict(row)

    candidate_output = str(out_dir / "walk_forward_candidate_unpromoted.pkl")
    source = Path(str(best_oos_row.get(artifact_key, ""))) if best_oos_row is not None else None
    if source is not None and source.exists():
        shutil.copy2(source, candidate_output)
    else:
        candidate_output = ""

    final_output = ""
    if stability["eligible"] and candidate_output:
        final_output = str(out_dir / "walk_forward_best.pkl")
        shutil.copy2(candidate_output, final_output)

    return {
        "windows": len({str(row.get("validation_window", "")) for row in results}),
        "evaluations": len(results),
        "mean_oos_fitness": float(np.mean(oos_arr)) if oos_arr.size else -float("inf"),
        "median_oos_fitness": float(np.median(oos_arr)) if oos_arr.size else -float("inf"),
        "worst_oos_fitness": float(np.min(oos_arr)) if oos_arr.size else -float("inf"),
        "std_oos_fitness": float(np.std(oos_arr)) if oos_arr.size else float("inf"),
        "selected_window": int(best_oos_row["window"]) if best_oos_row is not None else None,
        "selected_seed": int(best_oos_row["seed"]) if best_oos_row is not None else None,
        "candidate_output": candidate_output,
        "final_output": final_output,
        "promotion_evidence": stability,
        "promotion_eligible": bool(stability["eligible"] and final_output),
    }


def walk_forward_train(
    config: AppConfig,
    rounds: int,
    output_dir: str,
    train_window_bars: int = 7,
    test_window_bars: int = 7,
    step_bars: int | None = None,
    log_path: str | None = None,
    trials: int = 10,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    curriculum_learning: bool = False,
    from_date: str = "2025.01.01",
    to_date: str = "2025.02.01",
    val_ratio: float | None = None,
    model: int = 0,
    policy_core: str = "current",
    m1_backend: str = "direct_rnn",
    mt5_close_scope: str = "trial",
    experience_journal_path: str | Path | None = "outputs/mt5_experience_journal.jsonl",
    replay_journal_max: int = 64,
    seed_count: int = 1,
) -> dict[str, object]:
    """Walk-forward cho policy graph chung cũ, với train/OOS tách tuyệt đối.

    ``val_ratio`` còn được giữ trong signature để không phá caller cũ nhưng
    không còn được dùng. OOS luôn có đúng ``test_window_bars`` ngày.
    """
    del val_ratio, experience_journal_path  # Journal từng ô được cô lập chủ động.
    if int(seed_count) < 1:
        raise ValueError("--seed-count phai >= 1")
    windows = _kiem_tra_cua_so_walk_forward(
        from_date=from_date,
        to_date=to_date,
        train_window_days=int(train_window_bars),
        test_window_days=int(test_window_bars),
        step_days=step_bars,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path) if log_path else out_dir / "walk_forward.jsonl"
    if log_file.exists():
        log_file.unlink()
    results: list[dict[str, object]] = []
    base_seed = int(getattr(getattr(config, "learning", None), "seed", 20260629))

    for i, window in enumerate(windows, start=1):
        for seed_index in range(int(seed_count)):
            training_seed = base_seed + (i - 1) * 100_003 + seed_index
            train_output = out_dir / f"window_{i:03d}_seed_{seed_index:03d}.pkl"
            train_result = train_self_growing_mt5(
                config,
                rounds,
                str(train_output),
                trials,
                None,  # Không carry policy từ cửa sổ trước: mỗi ô matrix độc lập.
                runner_dir,
                timeout_sec,
                visual,
                shutdown_terminal,
                curriculum_learning,
                from_date=window["train_from"],
                to_date=window["train_to"],
                model=model,
                policy_core=policy_core,
                m1_backend=m1_backend,
                mt5_close_scope=mt5_close_scope,
                experience_journal_path=_duong_dan_nhat_ky_walk_forward(train_output, i, seed_index),
                replay_journal_max=replay_journal_max,
                seed=training_seed,
            )
            train_fitness = _mt5_best_fitness(train_result)
            train_metrics = _mt5_best_metrics(train_result)
            try:
                validation_policy = load_policy(train_output, policy_core=policy_core)
                val_result = evaluate_policy_mt5_oos(
                    validation_policy,
                    config,
                    runner_dir=runner_dir,
                    timeout_sec=timeout_sec,
                    visual=visual,
                    shutdown_terminal=shutdown_terminal,
                    from_date=window["test_from"],
                    to_date=window["test_to"],
                    model=model,
                )
                val_fitness = _mt5_best_fitness(val_result)
                val_metrics = _mt5_best_metrics(val_result)
            except Exception as exc:
                val_fitness = -float("inf")
                val_metrics = {"evaluation_error": True, "error": f"{type(exc).__name__}: {exc}"}
            row = {
                "window": i,
                "validation_window": f"{window['test_from']}:{window['test_to']}",
                "seed": training_seed,
                "seed_index": seed_index,
                "train_from": window["train_from"],
                "train_to": window["train_to"],
                "test_from": window["test_from"],
                "test_to": window["test_to"],
                # Tên val_* được giữ để tương thích integrity/promotion report.
                "val_from": window["test_from"],
                "val_to": window["test_to"],
                "train_fitness": train_fitness,
                "val_fitness": val_fitness,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "output": str(train_output),
                "journal": _duong_dan_nhat_ky_walk_forward(train_output, i, seed_index),
                "mode": "policy_chung_legacy",
                "oos_valid": bool(np.isfinite(float(val_fitness))),
            }
            results.append(row)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = _tong_ket_walk_forward(
        results=results,
        out_dir=out_dir,
        learning_cfg=getattr(config, "learning", None),
    )
    summary.update({
        "mode": "policy_chung_legacy",
        "train_window_days": int(train_window_bars),
        "test_window_days": int(test_window_bars),
        "step_days": int(step_bars if step_bars is not None else test_window_bars),
        "seed_count": int(seed_count),
        "oos_frozen": True,
        "journal_cach_ly_theo_o": True,
    })
    return {"summary": summary, "windows": results}


def walk_forward_he_chuyen_gia_mt5(
    config: AppConfig,
    *,
    output_dir: str,
    train_window_days: int = 7,
    test_window_days: int = 7,
    step_days: int | None = None,
    log_path: str | None = None,
    seed_count: int = 1,
    seed_system_path: str | Path | None = None,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    model: int = 0,
    policy_core: str = "current",
    from_date: str = "2025.01.01",
    to_date: str = "2025.02.01",
    seed: int = 0,
) -> dict[str, object]:
    """Walk-forward riêng cho kiến trúc một-context–một-chuyên-gia.

    Tuyệt đối không gọi ``train_self_growing_mt5``. Mỗi ô OOS × seed khởi tạo
    một kho chuyên gia độc lập, train chỉ ở đoạn train, rồi frozen OOS ở đoạn
    test. Vì vậy command này không thể âm thầm quay về policy chung.
    """
    if not bool(getattr(getattr(config, "he_chuyen_gia", None), "bat", False)):
        raise ValueError("walk-forward-he-chuyen-gia-mt5 can config he_chuyen_gia.bat=true")
    if int(seed_count) < 1:
        raise ValueError("--seed-count phai >= 1")
    windows = _kiem_tra_cua_so_walk_forward(
        from_date=from_date,
        to_date=to_date,
        train_window_days=int(train_window_days),
        test_window_days=int(test_window_days),
        step_days=step_days,
    )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = Path(log_path) if log_path else out_dir / "walk_forward_he_chuyen_gia.jsonl"
    if log_file.exists():
        log_file.unlink()
    results: list[dict[str, object]] = []
    base_seed = int(seed)

    for i, window in enumerate(windows, start=1):
        for seed_index in range(int(seed_count)):
            training_seed = base_seed + (i - 1) * 100_003 + seed_index
            train_output = out_dir / f"context_expert_window_{i:03d}_seed_{seed_index:03d}.pkl"
            train_result = huan_luyen_he_chuyen_gia_mt5(
                config,
                output_path=train_output,
                seed_system_path=seed_system_path,
                runner_dir=runner_dir,
                timeout_sec=timeout_sec,
                visual=visual,
                shutdown_terminal=shutdown_terminal,
                from_date=window["train_from"],
                to_date=window["train_to"],
                model=model,
                policy_core=policy_core,
                seed=training_seed,
            )
            train_fitness = float(train_result.get("fitness", -float("inf")))
            train_metrics = dict(train_result.get("metrics", {}) or {})
            try:
                with train_output.open("rb") as handle:
                    validation_system = load_pickle_with_removed_context_structural_migration(handle)
                ensure = getattr(validation_system, "ensure_compatible", None)
                if callable(ensure):
                    ensure()
                val_result = evaluate_policy_mt5_oos(
                    validation_system,
                    config,
                    runner_dir=runner_dir,
                    timeout_sec=timeout_sec,
                    visual=visual,
                    shutdown_terminal=shutdown_terminal,
                    from_date=window["test_from"],
                    to_date=window["test_to"],
                    model=model,
                )
                val_fitness = _mt5_best_fitness(val_result)
                val_metrics = _mt5_best_metrics(val_result)
            except Exception as exc:
                val_fitness = -float("inf")
                val_metrics = {"evaluation_error": True, "error": f"{type(exc).__name__}: {exc}"}
            row = {
                "window": i,
                "validation_window": f"{window['test_from']}:{window['test_to']}",
                "seed": training_seed,
                "seed_index": seed_index,
                "train_from": window["train_from"],
                "train_to": window["train_to"],
                "test_from": window["test_from"],
                "test_to": window["test_to"],
                "val_from": window["test_from"],
                "val_to": window["test_to"],
                "train_fitness": train_fitness,
                "val_fitness": val_fitness,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "output": str(train_output),
                "mode": "mot_context_mot_chuyen_gia",
                "oos_valid": bool(np.isfinite(float(val_fitness))),
            }
            results.append(row)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    summary = _tong_ket_walk_forward(
        results=results,
        out_dir=out_dir,
        learning_cfg=getattr(config, "learning", None),
    )
    summary.update({
        "mode": "mot_context_mot_chuyen_gia",
        "train_window_days": int(train_window_days),
        "test_window_days": int(test_window_days),
        "step_days": int(step_days if step_days is not None else test_window_days),
        "seed_count": int(seed_count),
        "oos_frozen": True,
        "mot_context_mot_owner": True,
        "policy_chung_ra_lenh": False,
        "tron_policy": False,
    })
    return {"summary": summary, "windows": results}

# ---------------------------------------------------------------------------
# Memory v6: strict pure-neural runtime
# ---------------------------------------------------------------------------
# The dataclasses above are retained only so legacy pickle files can still be
# deserialised by Python.  From v6 onward ExperienceSuperNetwork runtime writes,
# reads, develops, reflects and queries exclusively through NeuralMemoryCell and
# NeuralMemorySynapse.  Legacy dictionaries are emptied after one-way migration.

def _v6_nm_action_origin(self, trace: StepTrace) -> str:
    details = dict(getattr(trace, "execution_details", {}) or {})
    if bool(details.get("risk_override", False)):
        return "risk_override"
    if bool(details.get("masked", False)):
        return "mask"
    if bool(details.get("forced_exit", False)):
        return "forced_exit"
    if bool(details.get("bootstrap", False)):
        return "bootstrap"
    action_details = dict(getattr(trace, "action_details", {}) or {})
    if bool(action_details.get("bootstrap", False)):
        return "bootstrap"
    return "policy"


def _v6_nm_credit_weight(origin: str) -> float:
    origin = str(origin or "policy")
    if origin == "policy":
        return 1.0
    if origin == "risk_override":
        # Execution used a configured fixed risk rather than this output head.
        return 0.0
    if origin in {"forced_exit", "mask"}:
        return 0.25
    return 0.0


def _v6_cell_prefix(role: str) -> str:
    mapping = {
        "episodic_memory": "episode",
        "context_attractor": "cluster",
        "semantic_concept": "concept",
        "procedural_strategy": "strategy",
        "meta_strategy": "meta",
        "reflection_loop": "reflection",
    }
    if role.startswith("causal_"):
        return "causal"
    return mapping.get(role, role)


def _v6_cell_id(self, role: str) -> str:
    return f"{_v6_cell_prefix(role)}:{self._new_id()}"


def _v6_upsert_cell(
    self,
    cell_id: str,
    role: str,
    vector: np.ndarray,
    *,
    utility: float = 0.0,
    evidence_for: float | None = None,
    evidence_against: float | None = None,
    uncertainty: float = 1.0,
    activation: float = 0.0,
    meta: dict[str, object] | None = None,
) -> NeuralMemoryCell:
    vector = self._resize(np.asarray(vector, dtype=float), self.memory_key_size)
    pos = max(0.0, float(utility)) if evidence_for is None else max(0.0, float(evidence_for))
    neg = max(0.0, -float(utility)) if evidence_against is None else max(0.0, float(evidence_against))
    cell = self.neural_cells.get(cell_id)
    if cell is None:
        cell = NeuralMemoryCell(
            cell_id=cell_id,
            role=role,
            vector=vector,
            activation=float(activation),
            utility=float(utility),
            evidence_for=pos,
            evidence_against=neg,
            uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
            source_kind="neural_runtime",
            source_ref=cell_id,
            meta=copylib.deepcopy(dict(meta or {})),
        )
        self.neural_cells[cell_id] = cell
        return cell
    lr = float(np.clip(float(cell.plasticity), 0.01, 0.25))
    cell.vector = (1.0 - lr) * self._resize(cell.vector, self.memory_key_size) + lr * vector
    cell.activation = float(activation)
    cell.utility = float((1.0 - lr) * cell.utility + lr * float(utility))
    cell.evidence_for = float((1.0 - lr) * cell.evidence_for + lr * pos)
    cell.evidence_against = float((1.0 - lr) * cell.evidence_against + lr * neg)
    cell.uncertainty = float(np.clip((1.0 - lr) * cell.uncertainty + lr * float(uncertainty), 0.0, 1.0))
    cell.meta.update(copylib.deepcopy(dict(meta or {})))
    cell.age += 1
    return cell


def _v6_synapse(self, source: str, target: str, kind: str, *, weight: float = 1.0, utility: float = 0.0, meta: dict[str, object] | None = None) -> NeuralMemorySynapse:
    return NeuralMemorySynapse(
        source=str(source),
        target=str(target),
        kind=str(kind),
        weight=float(np.clip(weight, -8.0, 8.0)),
        utility=float(utility),
        evidence_for=max(0.0, float(utility)),
        evidence_against=max(0.0, -float(utility)),
        meta=copylib.deepcopy(dict(meta or {})),
    )


def _v6_migrate_legacy_views(self) -> None:
    """One-way migration from old typed memory views into neural cells."""
    if not getattr(self, "neural_cells", None):
        self.neural_cells = {}
    if not getattr(self, "neural_synapses", None):
        self.neural_synapses = []
    migrated_any = False
    for node_id, node in (getattr(self, "neurons", {}) or {}).items():
        cid = f"episode:{node_id}"
        action_origin = str(getattr(node, "action_origin", "policy") or "policy")
        meta = {
            "legacy_ref": node_id,
            "action_origin": action_origin,
            "credit_weight": float(getattr(node, "credit_weight", _v6_nm_credit_weight(action_origin))),
            "action_vector": self._resize(getattr(node, "action_embedding", np.zeros(16)), 20).tolist(),
            "context_delta": self._resize(getattr(node, "context_delta", np.zeros(self.memory_key_size)), self.memory_key_size).tolist(),
            "short_context": self._resize(getattr(node, "short_context", np.zeros(self.memory_key_size)), self.memory_key_size).tolist(),
            "regime_context": self._resize(getattr(node, "regime_context", np.zeros(self.memory_key_size)), self.memory_key_size).tolist(),
            "promotion_state": str(getattr(node, "promotion_state", "shadow")),
            "counterfactual_action_values": copylib.deepcopy(dict(getattr(node, "counterfactual_action_values", {}) or {})),
            "causal_lesson": copylib.deepcopy(dict(getattr(node, "causal_lesson", {}) or {})),
        }
        _v6_upsert_cell(self, cid, "episodic_memory", getattr(node, "obs_embedding", np.zeros(self.memory_key_size)), utility=float(getattr(node, "utility", 0.0)), evidence_for=float(getattr(node, "evidence_for", max(0.0, getattr(node, "utility", 0.0)))), evidence_against=float(getattr(node, "evidence_against", max(0.0, -getattr(node, "utility", 0.0)))), uncertainty=float(getattr(node, "uncertainty", 1.0)), activation=float(getattr(node, "strength", 1.0)), meta=meta)
        migrated_any = True

    # Preserve every legacy semantic layer as a neural mirror.  These typed
    # records are compatibility views only; retrieval remains neural, but a
    # checkpoint must never lose concepts/strategies/causes during migration.
    def _legacy_cell_id(kind: str, ref: str) -> str:
        return f"legacy:{kind}:{ref}"

    def _legacy_add_cell(kind: str, ref: str, role: str, item: object) -> str:
        cell_id = _legacy_cell_id(kind, ref)
        try:
            vector = self._item_vector(item)
        except Exception:
            vector = np.zeros(self.memory_key_size, dtype=float)
        self._upsert_neural_cell(
            cell_id,
            role,
            vector,
            item,
            source_kind=f"legacy_{kind}",
            source_ref=str(ref),
        )
        cell = self.neural_cells[cell_id]
        meta = dict(getattr(cell, "meta", {}) or {})
        if isinstance(item, dict):
            payload = dict(item.get("payload", {}) or {})
            meta.update({"payload": copylib.deepcopy(payload)})
        else:
            meta.update({
                "causal_profile": copylib.deepcopy(dict(getattr(item, "causal_profile", {}) or {})),
                "counterfactual_values": copylib.deepcopy(dict(getattr(item, "counterfactual_values", getattr(item, "counterfactual_action_values", {})) or {})),
                "causal_lesson": copylib.deepcopy(dict(getattr(item, "causal_lesson", {}) or {})),
                "related_neurons": list(getattr(item, "related_neurons", []) or []),
                "related_strategies": list(getattr(item, "related_strategies", []) or []),
            })
        cell.meta = meta
        return cell_id

    legacy_cluster_ids: dict[str, str] = {}
    legacy_concept_ids: dict[str, str] = {}
    legacy_strategy_ids: dict[str, str] = {}
    for cluster_id, cluster in (getattr(self, "clusters", {}) or {}).items():
        legacy_cluster_ids[str(cluster_id)] = _legacy_add_cell("cluster", str(cluster_id), "context_attractor", cluster)
        migrated_any = True
    for concept_id, concept in (getattr(self, "concepts", {}) or {}).items():
        legacy_concept_ids[str(concept_id)] = _legacy_add_cell("concept", str(concept_id), "semantic_concept", concept)
        migrated_any = True
    for strategy_id, strategy in (getattr(self, "strategies", {}) or {}).items():
        legacy_strategy_ids[str(strategy_id)] = _legacy_add_cell("strategy", str(strategy_id), "procedural_strategy", strategy)
        migrated_any = True
    for meta_id, meta in (getattr(self, "meta_strategies", {}) or {}).items():
        _legacy_add_cell("meta", str(meta_id), "meta_strategy", meta)
        migrated_any = True
    for causal_id, causal in (getattr(self, "causal_nodes", {}) or {}).items():
        kind = str(dict(causal or {}).get("kind", "node"))
        _legacy_add_cell("causal", str(causal_id), f"causal_{kind}", dict(causal or {}))
        migrated_any = True

    existing_synapses = {
        (str(syn.source), str(syn.target), str(syn.kind))
        for syn in list(getattr(self, "neural_synapses", []) or [])
    }
    def _legacy_link(source: str, target: str, kind: str, *, weight: float = 1.0, utility: float = 0.0, meta: dict[str, object] | None = None) -> None:
        key = (str(source), str(target), str(kind))
        if source in self.neural_cells and target in self.neural_cells and key not in existing_synapses:
            self.neural_synapses.append(_v6_synapse(self, source, target, kind, weight=weight, utility=utility, meta=meta))
            existing_synapses.add(key)

    for cluster_id, cluster in (getattr(self, "clusters", {}) or {}).items():
        target = legacy_cluster_ids.get(str(cluster_id), "")
        for member_id in list(getattr(cluster, "member_ids", []) or []):
            _legacy_link(f"episode:{member_id}", target, "episode_to_cluster", utility=float(getattr(cluster, "utility", 0.0)), meta={"legacy_cluster": str(cluster_id)})
    for concept_id, concept in (getattr(self, "concepts", {}) or {}).items():
        target = legacy_concept_ids.get(str(concept_id), "")
        for cluster_id in list(getattr(concept, "cluster_ids", []) or []):
            _legacy_link(legacy_cluster_ids.get(str(cluster_id), ""), target, "cluster_to_concept", weight=max(0.1, float(getattr(concept, "confidence", 0.0))), utility=float(getattr(concept, "utility", 0.0)), meta={"legacy_concept": str(concept_id)})
        for member_id in list(getattr(concept, "related_neurons", []) or []):
            _legacy_link(f"episode:{member_id}", target, "episode_to_concept_support", weight=0.35, utility=float(getattr(concept, "utility", 0.0)), meta={"legacy_concept": str(concept_id)})
    for strategy_id, strategy in (getattr(self, "strategies", {}) or {}).items():
        target = legacy_strategy_ids.get(str(strategy_id), "")
        for cluster_id in list(getattr(strategy, "cluster_ids", []) or []):
            _legacy_link(legacy_cluster_ids.get(str(cluster_id), ""), target, "cluster_to_strategy", weight=max(0.1, float(getattr(strategy, "confidence", 0.0))), utility=float(getattr(strategy, "utility", 0.0)), meta={"legacy_strategy": str(strategy_id)})
        for concept_id, concept in (getattr(self, "concepts", {}) or {}).items():
            if str(strategy_id) in {str(v) for v in list(getattr(concept, "related_strategies", []) or [])}:
                _legacy_link(legacy_concept_ids.get(str(concept_id), ""), target, "concept_to_strategy", weight=max(0.1, float(getattr(concept, "confidence", 0.0))), utility=float(getattr(strategy, "utility", 0.0)), meta={"legacy_strategy": str(strategy_id)})
    for meta_id, meta in (getattr(self, "meta_strategies", {}) or {}).items():
        target = _legacy_cell_id("meta", str(meta_id))
        for strategy_id in list(getattr(meta, "strategy_ids", []) or []):
            _legacy_link(legacy_strategy_ids.get(str(strategy_id), ""), target, "strategy_to_meta", weight=max(0.1, float(getattr(meta, "trust", 0.0))), utility=float(getattr(meta, "utility", 0.0)), meta={"legacy_meta": str(meta_id)})

    for conn in list(getattr(self, "connections", []) or []):
        src = f"episode:{conn.source}" if str(conn.source) in getattr(self, "neurons", {}) else str(conn.source)
        dst = f"episode:{conn.target}" if str(conn.target) in getattr(self, "neurons", {}) else str(conn.target)
        kind = f"legacy_{conn.kind}"
        key = (src, dst, kind)
        if src in self.neural_cells and dst in self.neural_cells and key not in existing_synapses:
            self.neural_synapses.append(_v6_synapse(self, src, dst, kind, weight=float(conn.weight), utility=float(conn.utility)))
            existing_synapses.add(key)
    # Keep old typed collections as *read-only migration views*.  Runtime query,
    # development and credit use only ``neural_cells`` / ``neural_synapses``, but
    # deleting the source views here used to erase semantic/causal evidence that
    # was added by an older checkpoint after its first compatibility pass.  They
    # are retained solely so future migration is lossless and inspectable.
    self.recent_ids = [cid for cid in getattr(self, "recent_ids", []) if cid in self.neural_cells]
    if migrated_any:
        self._v6_develop_neural_hierarchy()


def _v6_ensure_compatible(self) -> None:
    if not hasattr(self, "neural_cells") or self.neural_cells is None:
        self.neural_cells = {}
    if not hasattr(self, "neural_synapses") or self.neural_synapses is None:
        self.neural_synapses = []
    if not hasattr(self, "memory_origin_stats") or self.memory_origin_stats is None:
        self.memory_origin_stats = {}
    if not hasattr(self, "reflection_log") or self.reflection_log is None:
        self.reflection_log = []
    if not hasattr(self, "next_id"):
        self.next_id = 0
    if not hasattr(self, "recent_ids") or self.recent_ids is None:
        self.recent_ids = []
    if not hasattr(self, "context_step"):
        self.context_step = 0
    if not hasattr(self, "memory_key_schema_version"):
        self.memory_key_schema_version = MEMORY_KEY_SCHEMA_VERSION
    if not hasattr(self, "memory_key_size"):
        self.memory_key_size = MEMORY_KEY_SIZE
    for temporal_name in ("previous_context", "short_context_state", "regime_context_state"):
        value = np.asarray(getattr(self, temporal_name, np.zeros(MEMORY_KEY_SIZE)), dtype=float).reshape(-1)
        setattr(self, temporal_name, self._resize(value, MEMORY_KEY_SIZE))
    if not hasattr(self, "position_outcome_classifier") or self.position_outcome_classifier is None:
        self.position_outcome_classifier = NeuralPositionOutcomeClassifier.seed()
    self.position_outcome_classifier.ensure_compatible()
    self.pure_neural_memory = True
    self.neural_substrate_version = 6
    # V7 accepts explicit positive retention budgets. Old open-ended
    # checkpoints remain compatible until the policy-level integrity config
    # installs deployment/training limits.
    for _capacity_name in ("max_neurons", "max_clusters", "max_strategies", "max_meta_strategies", "max_connections"):
        _raw_capacity = getattr(self, _capacity_name, None)
        try:
            _capacity = int(_raw_capacity) if _raw_capacity is not None else 0
        except (TypeError, ValueError):
            _capacity = 0
        setattr(self, _capacity_name, _capacity if _capacity > 0 else None)
    _v6_migrate_legacy_views(self)


def _v6_copy(self) -> ExperienceSuperNetwork:
    self.ensure_compatible()
    return ExperienceSuperNetwork(
        neurons={key: node.copy() for key, node in getattr(self, "neurons", {}).items()},
        clusters={key: cluster.copy() for key, cluster in getattr(self, "clusters", {}).items()},
        concepts={key: concept.copy() for key, concept in getattr(self, "concepts", {}).items()},
        strategies={key: strategy.copy() for key, strategy in getattr(self, "strategies", {}).items()},
        meta_strategies={key: meta.copy() for key, meta in getattr(self, "meta_strategies", {}).items()},
        causal_nodes={key: copylib.deepcopy(value) for key, value in getattr(self, "causal_nodes", {}).items()},
        connections=[conn.copy() for conn in getattr(self, "connections", [])],
        neural_cells={key: cell.copy() for key, cell in self.neural_cells.items()},
        neural_synapses=[syn.copy() for syn in self.neural_synapses],
        neural_substrate_version=6,
        pure_neural_memory=True,
        max_neurons=getattr(self, "max_neurons", None),
        max_clusters=getattr(self, "max_clusters", None),
        max_strategies=getattr(self, "max_strategies", None),
        max_meta_strategies=getattr(self, "max_meta_strategies", None),
        max_connections=getattr(self, "max_connections", None),
        duplicate_similarity=float(getattr(self, "duplicate_similarity", 0.992)),
        duplicate_action_similarity=float(getattr(self, "duplicate_action_similarity", 0.985)),
        novelty_utility_threshold=float(getattr(self, "novelty_utility_threshold", 0.08)),
        memory_key_schema_version=int(getattr(self, "memory_key_schema_version", MEMORY_KEY_SCHEMA_VERSION)),
        memory_key_size=int(getattr(self, "memory_key_size", MEMORY_KEY_SIZE)),
        next_id=int(getattr(self, "next_id", 0)),
        recent_ids=list(getattr(self, "recent_ids", []) or []),
        context_step=int(getattr(self, "context_step", 0)),
        previous_context=np.asarray(getattr(self, "previous_context", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
        short_context_state=np.asarray(getattr(self, "short_context_state", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
        regime_context_state=np.asarray(getattr(self, "regime_context_state", np.zeros(MEMORY_KEY_SIZE)), dtype=float).copy(),
        position_outcome_classifier=self.position_outcome_classifier.copy(),
        reflection_log=copylib.deepcopy(list(getattr(self, "reflection_log", []) or [])),
        memory_origin_stats={str(k): float(v) for k, v in dict(getattr(self, "memory_origin_stats", {}) or {}).items()},
    )


def _v6_find_duplicate_cell(self, obs_key: np.ndarray, action_vec: np.ndarray, action_origin: str, delta: np.ndarray, short: np.ndarray, regime: np.ndarray) -> str | None:
    best: tuple[float, str] | None = None
    for cell_id, cell in self.neural_cells.items():
        if cell.role != "episodic_memory":
            continue
        meta = dict(cell.meta or {})
        if str(meta.get("action_origin", "policy")) != str(action_origin):
            continue
        sim_obs = self._cell_multi_scale_similarity(obs_key, delta, short, regime, cell)
        sim_action = self._cosine(action_vec, np.asarray(meta.get("action_vector", np.zeros(action_vec.size)), dtype=float))
        if sim_obs >= float(getattr(self, "duplicate_similarity", 0.992)) and sim_action >= float(getattr(self, "duplicate_action_similarity", 0.985)):
            score = 0.7 * sim_obs + 0.3 * sim_action
            if best is None or score > best[0]:
                best = (score, cell_id)
    return None if best is None else best[1]


def _v6_update_cell_evidence(self, cell: NeuralMemoryCell, utility: float, lr: float = 0.10) -> None:
    pos = max(0.0, float(utility))
    neg = max(0.0, -float(utility))
    cell.evidence_for = float((1.0 - lr) * float(cell.evidence_for) + lr * pos)
    cell.evidence_against = float((1.0 - lr) * float(cell.evidence_against) + lr * neg)
    total = cell.evidence_for + cell.evidence_against
    cell.uncertainty = float(np.clip(1.0 / (1.0 + total), 0.0, 1.0))
    cell.utility = float((1.0 - lr) * float(cell.utility) + lr * float(utility))
    cell.activation = float(np.tanh(float(cell.activation) + utility))
    cell.meta["promotion_state"] = _v6_promotion_state(cell)


def _v6_promotion_state(cell: NeuralMemoryCell) -> str:
    pos = max(0.0, float(cell.evidence_for))
    neg = max(0.0, float(cell.evidence_against))
    total = pos + neg
    contradiction = neg / max(1e-9, total)
    policy_support = float(dict(cell.meta or {}).get("policy_support", 0.0) or 0.0)
    if policy_support >= 8.0 and pos > neg and contradiction < 0.40:
        return "consolidated"
    if policy_support >= 3.0 and contradiction < 0.55:
        return "candidate"
    if policy_support >= 2.0 and contradiction >= 0.55:
        return "contradicted"
    return "shadow"


def _v6_absorb_transition(self, trace: StepTrace, next_obs: np.ndarray, reward: float, cost: float, signal: dict[str, float]) -> str:
    self.ensure_compatible()
    execution_details = dict(getattr(trace, "execution_details", {}) or {})
    if bool(execution_details.get("bridge_observed", False)) and not bool(execution_details.get("memory_eligible", False)):
        return ""
    td_delta = float(signal.get("td_delta", 0.0))
    pred_error = float(signal.get("prediction_error", 0.0))
    raw_utility = td_delta + float(reward) - float(cost) - 0.1 * pred_error
    action_origin = _v6_nm_action_origin(self, trace)
    credit_weight = _v6_nm_credit_weight(action_origin)
    utility = raw_utility * credit_weight
    obs_key = self._trace_memory_key(trace)
    current, delta, short, regime = self._observe_context(obs_key, advance=True)
    action_vec = self._resize(getattr(trace, "selected_action", np.zeros(20)), 20)
    cell_id = _v6_find_duplicate_cell(self, current, action_vec, action_origin, delta, short, regime)
    if cell_id is None:
        cell_id = _v6_cell_id(self, "episodic_memory")
    meta = {
        "trace_id": str(getattr(trace, "trace_id", "")),
        "action_origin": action_origin,
        "credit_weight": float(credit_weight),
        "raw_utility": float(raw_utility),
        "reward": float(reward),
        "td_delta": td_delta,
        "prediction_error": pred_error,
        "action_vector": action_vec.tolist(),
        "next_vector": self._resize(self._memory_key(next_obs, getattr(trace, "account_context_after", {}), getattr(trace, "positions_pipe_after", "")), self.memory_key_size).tolist(),
        "context_delta": delta.tolist(),
        "short_context": short.tolist(),
        "regime_context": regime.tolist(),
        "position_context": self._position_context_vector(getattr(trace, "positions_pipe", "")).tolist(),
        "account_context": {str(k): float(v) for k, v in getattr(trace, "account_context", {}).items()},
        "account_context_after": {str(k): float(v) for k, v in getattr(trace, "account_context_after", {}).items()},
        "counterfactual_action_values": self._counterfactual_values_for_trace(trace),
        "causal_lesson": self._causal_lesson_for_transition(trace, reward, cost, signal, raw_utility, action_origin),
        "policy_support": float(1.0 if action_origin == "policy" else 0.0),
        "bootstrap_support": float(1.0 if action_origin == "bootstrap" else 0.0),
        "support_strength": 1.0 + abs(float(raw_utility)),
    }
    cell = _v6_upsert_cell(self, cell_id, "episodic_memory", current, utility=utility, evidence_for=max(0.0, utility), evidence_against=max(0.0, -utility), uncertainty=0.35 if action_origin == "policy" else 0.85, activation=float(np.tanh(raw_utility)), meta=meta)
    # Increment support directly so repeated duplicate episodes accumulate.
    cell.meta["policy_support"] = float(cell.meta.get("policy_support", 0.0) or 0.0) + (1.0 if action_origin == "policy" else 0.0)
    cell.meta["bootstrap_support"] = float(cell.meta.get("bootstrap_support", 0.0) or 0.0) + (1.0 if action_origin == "bootstrap" else 0.0)
    _v6_update_cell_evidence(self, cell, utility, lr=0.20)
    self.memory_origin_stats[action_origin] = float(self.memory_origin_stats.get(action_origin, 0.0)) + 1.0
    # Pure neural causal/counterfactual cells.
    if abs(raw_utility) > self.novelty_utility_threshold or pred_error > 1.0:
        cause_id = _v6_cell_id(self, "causal_outcome")
        cause_meta = copylib.deepcopy(dict(meta.get("causal_lesson", {}) or {}))
        cause_meta.update({"episode_cell": cell_id, "action_origin": action_origin})
        _v6_upsert_cell(self, cause_id, "causal_outcome", current, utility=utility, evidence_for=max(0.0, utility), evidence_against=max(0.0, -utility), uncertainty=0.5, activation=float(np.tanh(raw_utility)), meta=cause_meta)
        self.neural_synapses.append(_v6_synapse(self, cell_id, cause_id, "episodic_to_causal", weight=float(np.tanh(raw_utility)), utility=utility))
        for key, value in dict(meta.get("counterfactual_action_values", {}) or {}).items():
            cf_id = _v6_cell_id(self, "causal_counterfactual")
            _v6_upsert_cell(self, cf_id, "causal_counterfactual", current, utility=float(value), evidence_for=max(0.0, float(value)), evidence_against=max(0.0, -float(value)), uncertainty=0.65, activation=float(np.tanh(value)), meta={"episode_cell": cell_id, "counterfactual_action": str(key), "action_origin": action_origin, "action_vector": action_vec.tolist()})
            self.neural_synapses.append(_v6_synapse(self, cell_id, cf_id, "episodic_to_counterfactual", weight=float(np.tanh(value)), utility=float(value)))
    for old_id in list(getattr(self, "recent_ids", []) or [])[-32:]:
        old = self.neural_cells.get(old_id)
        if old is None:
            continue
        sim = self._cosine(old.vector, cell.vector)
        if sim > 0.20:
            self.neural_synapses.append(_v6_synapse(self, old_id, cell_id, "learned_lateral", weight=sim, utility=utility))
        if float(old.utility) * utility > 0.0:
            self.neural_synapses.append(_v6_synapse(self, old_id, cell_id, "learned_recurrent_support", weight=0.5 + 0.5 * sim, utility=utility))
        elif float(old.utility) < 0.0 or utility < 0.0:
            self.neural_synapses.append(_v6_synapse(self, old_id, cell_id, "learned_inhibitory_contrast", weight=-abs(0.2 + 0.3 * sim), utility=-abs(utility)))
    if cell_id in self.recent_ids:
        self.recent_ids.remove(cell_id)
    self.recent_ids.append(cell_id)
    self.recent_ids = self.recent_ids[-512:]
    self._v6_develop_neural_hierarchy()
    self.develop()
    return cell_id


def _v6_role_cells(self, role: str) -> list[tuple[str, NeuralMemoryCell]]:
    return [(cid, cell) for cid, cell in self.neural_cells.items() if str(cell.role) == role]


def _v6_develop_neural_hierarchy(self) -> None:
    """Create all higher memory functions as cells/synapses, not objects."""
    self.ensure_compatible() if not getattr(self, "pure_neural_memory", False) else None
    episodes = _v6_role_cells(self, "episodic_memory")
    if not episodes:
        return
    existing_attractors = _v6_role_cells(self, "context_attractor")
    for ep_id, ep in episodes[-64:]:
        meta = dict(ep.meta or {})
        if str(meta.get("action_origin", "policy")) != "policy":
            # Bootstrap/shadow episodes can support context but not create growth.
            continue
        best: tuple[float, str, NeuralMemoryCell] | None = None
        for cid, attractor in existing_attractors:
            sim = self._cosine(ep.vector, attractor.vector)
            if best is None or sim > best[0]:
                best = (sim, cid, attractor)
        if best is None or best[0] < 0.72:
            cid = _v6_cell_id(self, "context_attractor")
            attractor = _v6_upsert_cell(self, cid, "context_attractor", ep.vector, utility=float(ep.utility), evidence_for=ep.evidence_for, evidence_against=ep.evidence_against, uncertainty=ep.uncertainty, activation=float(ep.activation), meta={"members": [ep_id], "action_vector": meta.get("action_vector", []), "context_delta": meta.get("context_delta", []), "short_context": meta.get("short_context", []), "regime_context": meta.get("regime_context", []), "policy_support": 1.0, "bootstrap_support": 0.0, "support_strength": 1.0})
            existing_attractors.append((cid, attractor))
            self.neural_synapses.append(_v6_synapse(self, ep_id, cid, "episode_to_attractor", weight=1.0, utility=float(ep.utility)))
        else:
            _, cid, attractor = best
            members = list(dict(attractor.meta or {}).get("members", []) or [])
            if ep_id not in members:
                members.append(ep_id)
            attractor.meta["members"] = members[-256:]
            attractor.meta["policy_support"] = float(attractor.meta.get("policy_support", 0.0) or 0.0) + 1.0
            _v6_upsert_cell(self, cid, "context_attractor", ep.vector, utility=float(ep.utility), evidence_for=ep.evidence_for, evidence_against=ep.evidence_against, uncertainty=ep.uncertainty, activation=float(ep.activation), meta=attractor.meta)
            self.neural_synapses.append(_v6_synapse(self, ep_id, cid, "episode_to_attractor", weight=max(0.1, float(best[0])), utility=float(ep.utility)))
    # Concepts grow from repeated context attractors.
    for attr_id, attr in list(_v6_role_cells(self, "context_attractor")):
        policy_support = float(dict(attr.meta or {}).get("policy_support", 0.0) or 0.0)
        if policy_support < 3.0:
            continue
        concept_key = stable_identifier("concept", tuple(np.round(self._resize(attr.vector, 8), 2).tolist()), length=16)
        meta = dict(attr.meta or {})
        meta.update({"source_attractor": attr_id, "promotion_state": _v6_promotion_state(attr), "support_strength": max(1.0, policy_support)})
        concept = _v6_upsert_cell(self, concept_key, "semantic_concept", attr.vector, utility=float(attr.utility), evidence_for=attr.evidence_for, evidence_against=attr.evidence_against, uncertainty=attr.uncertainty, activation=float(np.tanh(attr.utility)), meta=meta)
        concept.meta["policy_support"] = max(float(concept.meta.get("policy_support", 0.0) or 0.0), policy_support)
        concept.meta["promotion_state"] = _v6_promotion_state(concept)
        self.neural_synapses.append(_v6_synapse(self, attr_id, concept_key, "attractor_to_concept", weight=max(0.1, 1.0 - concept.uncertainty), utility=float(concept.utility)))
        # Strategy is now a procedural cell derived from concept activation.
        if str(concept.meta.get("promotion_state", "shadow")) in {"candidate", "consolidated"}:
            strategy_key = f"strategy:{concept_key.split(':', 1)[1]}"
            strategy = _v6_upsert_cell(self, strategy_key, "procedural_strategy", concept.vector, utility=float(concept.utility), evidence_for=concept.evidence_for, evidence_against=concept.evidence_against, uncertainty=concept.uncertainty, activation=float(concept.activation), meta=copylib.deepcopy(dict(concept.meta or {})))
            strategy.meta["source_concept"] = concept_key
            self.neural_synapses.append(_v6_synapse(self, concept_key, strategy_key, "concept_to_strategy", weight=max(0.1, 1.0 - strategy.uncertainty), utility=float(strategy.utility)))
            if str(strategy.meta.get("promotion_state", "shadow")) == "consolidated":
                meta_key = f"meta:{concept_key.split(':', 1)[1]}"
                meta_cell = _v6_upsert_cell(self, meta_key, "meta_strategy", strategy.vector, utility=float(strategy.utility), evidence_for=strategy.evidence_for, evidence_against=strategy.evidence_against, uncertainty=strategy.uncertainty, activation=float(strategy.activation), meta=copylib.deepcopy(dict(strategy.meta or {})))
                self.neural_synapses.append(_v6_synapse(self, strategy_key, meta_key, "strategy_to_meta", weight=max(0.1, 1.0 - meta_cell.uncertainty), utility=float(meta_cell.utility)))


def _v6_develop(self) -> None:
    self.neural_substrate_version = 6
    self.pure_neural_memory = True
    # No legacy hierarchy development. All plasticity is on cells/synapses.
    for cell in list(self.neural_cells.values()):
        cell.age += 1
        rel, direction, uncertainty = self._cell_reliability(cell)
        cell.activation = float(np.tanh(0.96 * float(cell.activation) + 0.04 * direction * rel))
        cell.plasticity = float(np.clip(0.99 * float(cell.plasticity) + 0.01 * (uncertainty + abs(float(cell.utility))), 0.005, 0.75))
        cell.meta["promotion_state"] = _v6_promotion_state(cell)
    kept: list[NeuralMemorySynapse] = []
    for syn in list(self.neural_synapses or []):
        if not bool(getattr(syn, "enabled", True)):
            continue
        if syn.source not in self.neural_cells or syn.target not in self.neural_cells:
            continue
        src = self.neural_cells[syn.source]
        dst = self.neural_cells[syn.target]
        syn.age += 1
        hebb = float(src.activation * dst.activation)
        syn.weight = float(np.clip(0.995 * float(syn.weight) + float(syn.plasticity) * hebb + 0.002 * float(dst.utility), -8.0, 8.0))
        syn.utility = float(0.98 * float(syn.utility) + 0.02 * (float(src.utility) + float(dst.utility)) / 2.0)
        if syn.age < 5000 or abs(syn.weight) > 1e-4 or abs(syn.utility) > 1e-4:
            kept.append(syn)
    self.neural_synapses = kept
    self._v6_develop_neural_hierarchy()
    # Historical typed collections are retained as read-only migration views.
    # Pure-neural query/development never reads them, but clearing them here
    # would make an old checkpoint lose semantic/causal evidence after one
    # development cycle and would make migration non-idempotent.
    self.legacy_views_read_only = True


def _v6_query(
    self,
    state: np.ndarray,
    *legacy_args: object,
    output_size: int = 20,
    module_count: int = 0,
    account_context: dict[str, object] | None = None,
    positions_pipe: str | None = "",
    **extra_context: object,
) -> dict[str, object]:
    """Strict pure-neural memory query with the legacy policy-call signature.

    The graph policy still calls memory as query(state, output_size, module_count,
    account_context=..., positions_pipe=...).  Earlier pure-neural v6 accepted
    only keyword-only output_size/module_count, causing train-to-goal to crash
    with positional arguments.  Accept both forms, keep runtime pure-neural, and
    fold every external context field into the neural memory key instead of
    consulting legacy Episode/Concept/Strategy runtime objects.
    """
    if len(legacy_args) >= 1 and legacy_args[0] is not None:
        output_size = int(legacy_args[0])
    if len(legacy_args) >= 2 and legacy_args[1] is not None:
        module_count = int(legacy_args[1])
    if len(legacy_args) >= 3 and account_context is None:
        maybe_account = legacy_args[2]
        if isinstance(maybe_account, dict):
            account_context = maybe_account
    if len(legacy_args) >= 4 and not positions_pipe:
        positions_pipe = str(legacy_args[3] or "")

    self.ensure_compatible()
    # Convert caller context into the same expanded neural key used by the
    # transition writer.  Nothing is read from legacy Episode/Concept objects.
    current, delta, short, regime = self._observe_context(
        self._memory_key(state, account_context, positions_pipe),
        advance=False,
    )
    result = self._query_neural_substrate(current, delta, short, regime, output_size=output_size, module_count=module_count)
    result["runtime"] = "pure_neural_v6"
    result["legacy_runtime_objects"] = 0
    result["legacy_positional_args"] = len(legacy_args)
    result["query_context_fields"] = sorted([str(k) for k in extra_context.keys()])
    return result


def _v6_reflect_memory(self, recent_window: int = 128) -> dict[str, object]:
    self.ensure_compatible()
    cells = list(self.neural_cells.items())
    episodes = [(cid, c) for cid, c in cells if c.role == "episodic_memory"][-recent_window:]
    concepts = [(cid, c) for cid, c in cells if c.role == "semantic_concept"]
    policy_origin = sum(1 for _, c in episodes if str(dict(c.meta or {}).get("action_origin", "")) == "policy")
    bootstrap_origin = sum(1 for _, c in episodes if str(dict(c.meta or {}).get("action_origin", "")) == "bootstrap")
    top_concepts = []
    for cid, cell in sorted(concepts, key=lambda item: (float(item[1].evidence_for) - float(item[1].evidence_against), -float(item[1].uncertainty)), reverse=True)[:8]:
        top_concepts.append({"cell_id": cid, "state": str(dict(cell.meta or {}).get("promotion_state", "shadow")), "utility": float(cell.utility), "evidence_for": float(cell.evidence_for), "evidence_against": float(cell.evidence_against), "uncertainty": float(cell.uncertainty)})
    recommendations: list[str] = []
    if bootstrap_origin > policy_origin:
        recommendations.append("reduce_bootstrap_credit_and_expand_policy_origin_sampling")
    if not top_concepts:
        recommendations.append("collect_more_policy_origin_episode_cells_before_growth")
    if any(item["state"] == "contradicted" for item in top_concepts):
        recommendations.append("ablate_contradicted_semantic_cells")
    reflection = {
        "reflection_id": f"reflection:{int(getattr(self, 'context_step', 0))}:{len(getattr(self, 'reflection_log', []) or [])}",
        "pure_neural_memory": True,
        "neural_substrate_version": 6,
        "episodes": len(episodes),
        "policy_origin": policy_origin,
        "bootstrap_origin": bootstrap_origin,
        "mean_policy_utility": float(np.mean([c.utility for _, c in episodes if str(dict(c.meta or {}).get("action_origin", "")) == "policy"])) if policy_origin else 0.0,
        "top_concepts": top_concepts,
        "recommendations": recommendations,
    }
    self.reflection_log.append(reflection)
    self.reflection_log = self.reflection_log[-256:]
    ref_id = reflection["reflection_id"]
    _v6_upsert_cell(self, ref_id, "reflection_loop", np.zeros(self.memory_key_size), utility=float(reflection["mean_policy_utility"]), evidence_for=max(0.0, float(reflection["mean_policy_utility"])), evidence_against=max(0.0, -float(reflection["mean_policy_utility"])), uncertainty=0.5, activation=float(np.tanh(float(reflection["mean_policy_utility"]))), meta=copylib.deepcopy(reflection))
    for concept in top_concepts:
        self.neural_synapses.append(_v6_synapse(self, str(concept["cell_id"]), ref_id, "concept_to_reflection", weight=0.5, utility=float(concept["utility"])))
    return reflection


def _v6_summary(self) -> dict[str, object]:
    self.ensure_compatible()
    role_counts: dict[str, int] = {}
    for cell in self.neural_cells.values():
        role_counts[str(cell.role)] = role_counts.get(str(cell.role), 0) + 1
    syn_counts: dict[str, int] = {}
    for syn in self.neural_synapses:
        syn_counts[str(syn.kind)] = syn_counts.get(str(syn.kind), 0) + 1
    evidence_for = float(sum(max(0.0, float(c.evidence_for)) for c in self.neural_cells.values()))
    evidence_against = float(sum(max(0.0, float(c.evidence_against)) for c in self.neural_cells.values()))
    mean_uncertainty = float(np.mean([float(c.uncertainty) for c in self.neural_cells.values()])) if self.neural_cells else 1.0
    concept_states: dict[str, int] = {"shadow": 0, "candidate": 0, "consolidated": 0, "contradicted": 0}
    for cell in self.neural_cells.values():
        if cell.role == "semantic_concept":
            state = str(dict(cell.meta or {}).get("promotion_state", "shadow"))
            concept_states[state] = concept_states.get(state, 0) + 1
    return {
        "runtime": "pure_neural_v6",
        "pure_neural_memory": True,
        "legacy_runtime_objects": 0,
        "memory_key_schema_version": int(getattr(self, "memory_key_schema_version", MEMORY_KEY_SCHEMA_VERSION)),
        "memory_key_size": int(getattr(self, "memory_key_size", MEMORY_KEY_SIZE)),
        "neurons": int(role_counts.get("episodic_memory", 0)),
        "clusters": int(role_counts.get("context_attractor", 0)),
        "concepts": int(role_counts.get("semantic_concept", 0)),
        "strategies": int(role_counts.get("procedural_strategy", 0)),
        "meta_strategies": int(role_counts.get("meta_strategy", 0)),
        "causal_nodes": int(sum(v for k, v in role_counts.items() if k.startswith("causal_"))),
        "connections": len(self.neural_synapses),
        "evidence_for": evidence_for,
        "evidence_against": evidence_against,
        "mean_uncertainty": mean_uncertainty,
        "memory_origin_stats": {str(k): float(v) for k, v in dict(getattr(self, "memory_origin_stats", {}) or {}).items()},
        "concept_states": concept_states,
        "last_reflection": copylib.deepcopy(self.reflection_log[-1]) if getattr(self, "reflection_log", []) else {},
        "context_step": int(getattr(self, "context_step", 0)),
        "neural_substrate_version": 6,
        "neural_memory_cells": len(self.neural_cells),
        "neural_memory_synapses": len(self.neural_synapses),
        "neural_memory_roles": role_counts,
        "neural_synapse_kinds": syn_counts,
        "depth": 5 if role_counts.get("meta_strategy", 0) else (4 if role_counts.get("procedural_strategy", 0) else (3 if role_counts.get("semantic_concept", 0) else (2 if role_counts.get("context_attractor", 0) else 1))),
    }


ExperienceSuperNetwork.ensure_compatible = _v6_ensure_compatible
ExperienceSuperNetwork.copy = _v6_copy
ExperienceSuperNetwork.absorb_transition = _v6_absorb_transition
ExperienceSuperNetwork.develop = _v6_develop
ExperienceSuperNetwork.query = _v6_query
ExperienceSuperNetwork.reflect_memory = _v6_reflect_memory
ExperienceSuperNetwork.summary = _v6_summary
ExperienceSuperNetwork._v6_develop_neural_hierarchy = _v6_develop_neural_hierarchy

# ---------------------------------------------------------------------------
# Memory v7: pure-neural mutation/growth and causal-diagnosis substrate
# ---------------------------------------------------------------------------
# v6 made memory retrieval strict-neural.  v7 moves the remaining memory growth
# and causal diagnosis mechanisms onto the same cell/synapse substrate.  Runtime
# no longer requires a symbolic causal-label object or a low-memory-match rule to
# create growth pressure: evidence is represented by neural signal cells,
# attractors and proposal cells.  Debug metadata may expose scalar traces, but it
# is not read as the decision source.

_V6_ABSORB_TRANSITION = ExperienceSuperNetwork.absorb_transition
_V6_DEVELOP = ExperienceSuperNetwork.develop
_V6_SUMMARY = ExperienceSuperNetwork.summary
_V6_REFLECT_MEMORY = ExperienceSuperNetwork.reflect_memory


def _v7_safe_float(value: object, default: float = 0.0) -> float:
    try:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else float(default)
    except Exception:
        return float(default)


def _v7_signal_cell_id(name: str) -> str:
    return f"control:{name}"


def _v7_resize(self, values: object, size: int | None = None) -> np.ndarray:
    return self._resize(np.asarray(values, dtype=float).reshape(-1), int(size or self.memory_key_size))


def _v7_causal_vector(self, episode_cell: NeuralMemoryCell) -> np.ndarray:
    """Encode causal evidence as a numeric neural pattern, not as labels."""
    meta = dict(getattr(episode_cell, "meta", {}) or {})
    lesson = dict(meta.get("causal_lesson", {}) or {})
    cf = dict(meta.get("counterfactual_action_values", {}) or {})
    action_vec = _v7_resize(self, meta.get("action_vector", []), 20)
    context_delta = _v7_resize(self, meta.get("context_delta", []), self.memory_key_size)
    short_context = _v7_resize(self, meta.get("short_context", []), self.memory_key_size)
    values = [
        _v7_safe_float(meta.get("raw_utility", episode_cell.utility)),
        _v7_safe_float(meta.get("reward", 0.0)),
        _v7_safe_float(meta.get("td_delta", 0.0)),
        _v7_safe_float(meta.get("prediction_error", 0.0)),
        _v7_safe_float(meta.get("credit_weight", 0.0)),
        _v7_safe_float(lesson.get("direction_error", 0.0)),
        _v7_safe_float(lesson.get("entry_quality_error", 0.0)),
        _v7_safe_float(lesson.get("exit_quality_error", 0.0)),
        _v7_safe_float(lesson.get("risk_error", 0.0)),
        _v7_safe_float(lesson.get("regime_error", 0.0)),
        _v7_safe_float(lesson.get("execution_error", 0.0)),
    ]
    cf_values = [_v7_safe_float(v) for v in cf.values()]
    values.extend([
        float(np.mean(cf_values)) if cf_values else 0.0,
        float(np.max(cf_values)) if cf_values else 0.0,
        float(np.min(cf_values)) if cf_values else 0.0,
        float(np.std(cf_values)) if len(cf_values) > 1 else 0.0,
        float(np.mean(np.abs(action_vec))) if action_vec.size else 0.0,
        float(np.mean(np.abs(context_delta))) if context_delta.size else 0.0,
        float(np.mean(np.abs(short_context))) if short_context.size else 0.0,
    ])
    return self._resize(np.asarray(values, dtype=float), self.memory_key_size)


def _v7_update_neural_signal(self, name: str, value: float, *, utility: float = 0.0, uncertainty: float = 0.5) -> NeuralMemoryCell:
    value = float(np.clip(value, -8.0, 8.0))
    vector = np.zeros(self.memory_key_size, dtype=float)
    vector[0] = value
    vector[1] = float(np.tanh(utility))
    vector[2] = float(np.clip(uncertainty, 0.0, 1.0))
    cid = _v7_signal_cell_id(name)
    return _v6_upsert_cell(
        self,
        cid,
        "neural_growth_signal",
        vector,
        utility=float(utility),
        evidence_for=max(0.0, value),
        evidence_against=max(0.0, -value),
        uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        activation=float(np.tanh(value)),
        meta={"signal_name": name, "value": value, "runtime": "pure_neural_v7"},
    )


def _v7_record_causal_attractor(self, episode_id: str) -> None:
    ep = self.neural_cells.get(episode_id)
    if ep is None or str(ep.role) != "episodic_memory":
        return
    meta = dict(ep.meta or {})
    if str(meta.get("action_origin", "policy")) != "policy":
        # Bootstrap can be stored as shadow experience but must not form causal
        # credit attractors for policy/growth.
        return
    causal_vec = _v7_causal_vector(self, ep)
    existing = [(cid, c) for cid, c in self.neural_cells.items() if c.role == "causal_attractor"]
    best: tuple[float, str, NeuralMemoryCell] | None = None
    for cid, cell in existing:
        sim = self._cosine(causal_vec, cell.vector)
        if best is None or sim > best[0]:
            best = (sim, cid, cell)
    utility = float(ep.utility)
    if best is None or best[0] < 0.78:
        cid = stable_identifier("causal_attr", tuple(np.round(causal_vec[:16], 2).tolist()), length=20)
        attr = _v6_upsert_cell(
            self,
            cid,
            "causal_attractor",
            causal_vec,
            utility=utility,
            evidence_for=max(0.0, utility),
            evidence_against=max(0.0, -utility),
            uncertainty=float(ep.uncertainty),
            activation=float(np.tanh(abs(utility) + _v7_safe_float(meta.get("prediction_error", 0.0)))),
            meta={"episodes": [episode_id], "runtime": "pure_neural_v7", "origin": "policy"},
        )
        self.neural_synapses.append(_v6_synapse(self, episode_id, cid, "episode_to_causal_attractor", weight=1.0, utility=utility))
    else:
        _, cid, attr = best
        episodes = list(dict(attr.meta or {}).get("episodes", []) or [])
        if episode_id not in episodes:
            episodes.append(episode_id)
        attr.meta["episodes"] = episodes[-512:]
        _v6_upsert_cell(
            self,
            cid,
            "causal_attractor",
            causal_vec,
            utility=utility,
            evidence_for=max(0.0, utility),
            evidence_against=max(0.0, -utility),
            uncertainty=float(ep.uncertainty),
            activation=float(np.tanh(abs(utility))),
            meta=attr.meta,
        )
        self.neural_synapses.append(_v6_synapse(self, episode_id, cid, "episode_to_causal_attractor", weight=max(0.1, float(best[0])), utility=utility))


def _v7_neural_growth_step(self) -> dict[str, object]:
    """Let growth emerge from neural signal cells and attractor evidence."""
    cells = list(self.neural_cells.items())
    episodes = [(cid, c) for cid, c in cells if c.role == "episodic_memory"]
    concepts = [(cid, c) for cid, c in cells if c.role == "semantic_concept"]
    causal_attrs = [(cid, c) for cid, c in cells if c.role == "causal_attractor"]
    recent = episodes[-128:]
    policy_recent = [c for _, c in recent if str(dict(c.meta or {}).get("action_origin", "")) == "policy"]
    utilities = np.asarray([float(c.utility) for c in policy_recent], dtype=float) if policy_recent else np.zeros(0)
    pred_errors = np.asarray([_v7_safe_float(dict(c.meta or {}).get("prediction_error", 0.0)) for c in policy_recent], dtype=float) if policy_recent else np.zeros(0)
    uncertainty = float(np.mean([float(c.uncertainty) for _, c in concepts])) if concepts else 1.0
    contradiction = float(np.mean([max(0.0, float(c.evidence_against) - float(c.evidence_for)) for _, c in causal_attrs])) if causal_attrs else 0.0
    novelty = float(np.mean([float(c.uncertainty) for c in policy_recent])) if policy_recent else 1.0
    error = float(np.mean(pred_errors)) if pred_errors.size else 0.0
    utility_mean = float(np.mean(utilities)) if utilities.size else 0.0
    # Signals are represented as persistent cells.  This replaces a symbolic
    # if/else growth diagnosis with competing neural activations.
    sig_novelty = _v7_update_neural_signal(self, "novelty", novelty, utility=utility_mean, uncertainty=uncertainty)
    sig_error = _v7_update_neural_signal(self, "prediction_error", error, utility=-abs(error), uncertainty=uncertainty)
    sig_contra = _v7_update_neural_signal(self, "contradiction", contradiction, utility=-contradiction, uncertainty=uncertainty)
    sig_utility = _v7_update_neural_signal(self, "utility", utility_mean, utility=utility_mean, uncertainty=uncertainty)
    support = float(len(policy_recent))
    drive = float(np.tanh(0.75 * novelty + 0.55 * error + 0.40 * contradiction - 0.25 * max(0.0, utility_mean)))
    inhibit = float(np.tanh(max(0.0, -utility_mean) + contradiction))
    proposal_kind = "hold"
    target_role = "context_attractor"
    if support >= 3.0 and drive > 0.45 and inhibit < 0.95:
        proposal_kind = "grow"
        if error >= max(novelty, contradiction):
            target_role = "prediction_repair_cell"
        elif contradiction >= max(novelty, error):
            target_role = "inhibitory_causal_guard"
        else:
            target_role = "context_attractor"
    elif support >= 6.0 and contradiction > 0.75:
        proposal_kind = "prune_or_inhibit"
        target_role = "contradicted_substrate"
    proposal_id = f"growth:{int(getattr(self, 'context_step', 0))}:{len([1 for c in self.neural_cells.values() if c.role == 'growth_proposal'])}"
    proposal_vec = self._resize(np.asarray([drive, inhibit, novelty, error, contradiction, utility_mean, support], dtype=float), self.memory_key_size)
    proposal = _v6_upsert_cell(
        self,
        proposal_id,
        "growth_proposal",
        proposal_vec,
        utility=utility_mean - 0.25 * contradiction,
        evidence_for=max(0.0, drive - inhibit),
        evidence_against=max(0.0, inhibit - drive),
        uncertainty=float(np.clip(uncertainty, 0.0, 1.0)),
        activation=drive,
        meta={"proposal_kind": proposal_kind, "target_role": target_role, "runtime": "pure_neural_v7"},
    )
    for signal in (sig_novelty, sig_error, sig_contra, sig_utility):
        self.neural_synapses.append(_v6_synapse(self, signal.cell_id, proposal_id, "signal_to_growth_proposal", weight=float(signal.activation), utility=float(proposal.utility)))
    if proposal_kind == "grow":
        target_id = stable_identifier("growth_cell", target_role, tuple(np.round(proposal_vec[:12], 2).tolist()), length=20)
        target = _v6_upsert_cell(
            self,
            target_id,
            target_role,
            proposal_vec,
            utility=float(proposal.utility),
            evidence_for=float(proposal.evidence_for),
            evidence_against=float(proposal.evidence_against),
            uncertainty=float(proposal.uncertainty),
            activation=float(proposal.activation),
            meta={"created_by": proposal_id, "runtime": "pure_neural_v7", "proposal_kind": proposal_kind},
        )
        self.neural_synapses.append(_v6_synapse(self, proposal_id, target_id, "growth_proposal_to_cell", weight=max(0.1, drive - inhibit), utility=float(target.utility)))
    elif proposal_kind == "prune_or_inhibit":
        for cid, cell in causal_attrs[-16:]:
            if float(cell.evidence_against) > float(cell.evidence_for):
                self.neural_synapses.append(_v6_synapse(self, proposal_id, cid, "growth_inhibits_contradicted", weight=-abs(drive), utility=-abs(float(cell.evidence_against))))
    return {
        "runtime": "pure_neural_v7",
        "proposal_kind": proposal_kind,
        "target_role": target_role,
        "drive": drive,
        "inhibit": inhibit,
        "novelty": novelty,
        "prediction_error": error,
        "contradiction": contradiction,
        "utility_mean": utility_mean,
        "policy_support": support,
    }


def _v7_absorb_transition(self, trace: StepTrace, next_obs: np.ndarray, reward: float, cost: float, signal: dict[str, float]) -> str:
    cell_id = _V6_ABSORB_TRANSITION(self, trace, next_obs, reward, cost, signal)
    if cell_id:
        _v7_record_causal_attractor(self, cell_id)
        self._v7_last_growth = _v7_neural_growth_step(self)
    return cell_id


def _v7_develop(self) -> None:
    _V6_DEVELOP(self)
    self._v7_last_growth = _v7_neural_growth_step(self)
    self.neural_substrate_version = 7
    self.pure_neural_memory = True


def _v7_summary(self) -> dict[str, object]:
    result = _V6_SUMMARY(self)
    role_counts: dict[str, int] = {}
    for cell in self.neural_cells.values():
        role_counts[str(cell.role)] = role_counts.get(str(cell.role), 0) + 1
    result.update({
        "runtime": "pure_neural_v7",
        "neural_substrate_version": 7,
        "pure_neural_growth": True,
        "pure_neural_causal_diagnosis": True,
        "causal_attractors": int(role_counts.get("causal_attractor", 0)),
        "growth_signal_cells": int(role_counts.get("neural_growth_signal", 0)),
        "growth_proposal_cells": int(role_counts.get("growth_proposal", 0)),
        "growth_runtime_cells": int(sum(v for k, v in role_counts.items() if k in {"prediction_repair_cell", "inhibitory_causal_guard", "contradicted_substrate"} or k.startswith("growth_cell"))),
        "last_neural_growth": copylib.deepcopy(getattr(self, "_v7_last_growth", {})),
    })
    return result


def _v7_reflect_memory(self, recent_window: int = 128) -> dict[str, object]:
    reflection = _V6_REFLECT_MEMORY(self, recent_window=recent_window)
    growth = _v7_neural_growth_step(self)
    reflection["neural_growth"] = copylib.deepcopy(growth)
    reflection["pure_neural_causal_diagnosis"] = True
    reflection["causal_attractors"] = sum(1 for c in self.neural_cells.values() if c.role == "causal_attractor")
    if self.reflection_log:
        self.reflection_log[-1] = copylib.deepcopy(reflection)
    return reflection


ExperienceSuperNetwork.absorb_transition = _v7_absorb_transition
ExperienceSuperNetwork.develop = _v7_develop
ExperienceSuperNetwork.summary = _v7_summary
ExperienceSuperNetwork.reflect_memory = _v7_reflect_memory

# ---------------------------------------------------------------------------
# Memory v8: replacement-action learning, not action suppression
# ---------------------------------------------------------------------------
# v7 could still express negative evidence as a negative action_bias through
# signed-quality retrieval.  v8 changes that runtime contract: a bad policy
# outcome never directly weakens the old action channel.  It is converted into
# a causal replacement signal, then one or more replacement_action_proposal
# cells.  Query reads positive replacement proposals in similar contexts.  The
# failed action is preserved as critic/context evidence only.

_V7_ABSORB_TRANSITION = ExperienceSuperNetwork.absorb_transition
_V7_QUERY = ExperienceSuperNetwork.query
_V7_SUMMARY = ExperienceSuperNetwork.summary
_V7_REFLECT_MEMORY = ExperienceSuperNetwork.reflect_memory
_V7_DEVELOP = ExperienceSuperNetwork.develop


def _v8_action_index_from_vector(action_vec: np.ndarray, action_count: int = 12) -> int:
    values = np.asarray(action_vec, dtype=float).reshape(-1)
    n = min(max(1, int(action_count)), values.size)
    if n <= 0:
        return 0
    if not np.any(np.isfinite(values[:n])):
        return 0
    return int(np.nanargmax(values[:n]))


def _v8_action_vector(self, output_size: int, action_index: int, strength: float = 1.0) -> np.ndarray:
    vec = np.zeros(max(1, int(output_size)), dtype=float)
    idx = int(np.clip(int(action_index), 0, vec.size - 1))
    vec[idx] = float(np.clip(strength, -1.0, 1.0))
    return vec


def _v8_opposite_direction(action_index: int) -> int:
    if int(action_index) == 1:
        return 2
    if int(action_index) == 2:
        return 1
    if int(action_index) == 3:
        return 4
    if int(action_index) == 4:
        return 3
    if int(action_index) == 5:
        return 6
    if int(action_index) == 6:
        return 5
    return 0


def _v8_parse_counterfactual_action(key: object) -> int | None:
    text = str(key).strip().lower()
    aliases = {
        "hold": 0,
        "wait": 0,
        "market_buy": 1,
        "buy": 1,
        "long": 1,
        "market_sell": 2,
        "sell": 2,
        "short": 2,
        "buy_limit": 3,
        "sell_limit": 4,
        "buy_stop": 5,
        "sell_stop": 6,
        "cancel": 7,
        "cancel_orders": 7,
        "close_worst": 8,
        "close_best": 9,
        "close_all": 10,
        "manage": 11,
        "manage_position": 11,
        "reduce": 11,
    }
    if text in aliases:
        return aliases[text]
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return None
    if 0 <= value <= 11:
        return value
    return None


def _v8_replacement_actions_from_causal(self, episode_cell: NeuralMemoryCell, output_size: int) -> list[dict[str, object]]:
    meta = dict(episode_cell.meta or {})
    action_vec = np.asarray(meta.get("action_vector", np.zeros(output_size)), dtype=float)
    old_action = _v8_action_index_from_vector(action_vec, action_count=min(12, output_size))
    lesson = dict(meta.get("causal_lesson", {}) or {})
    causes = [str(c).lower() for c in (lesson.get("suspected_causes", []) or [])]
    cf_values = dict(meta.get("counterfactual_action_values", {}) or {})
    proposals: list[dict[str, object]] = []

    # 1) Counterfactual positive alternatives win first.  These are still stored
    # as cells/synapses; this code only seeds the action-proposal population.
    for key, value in sorted(cf_values.items(), key=lambda item: float(item[1]) if isinstance(item[1], (int, float)) else -1e9, reverse=True):
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue
        if val <= 0.0:
            continue
        idx = _v8_parse_counterfactual_action(key)
        if idx is None or idx == old_action:
            continue
        proposals.append({"action_index": idx, "cause": "counterfactual_positive", "strength": float(np.clip(0.25 + val, 0.15, 1.0))})
        if len(proposals) >= 3:
            break

    cause_text = "|".join(causes)
    def add(idx: int, cause: str, strength: float) -> None:
        if len(proposals) >= 5:
            return
        idx = int(np.clip(idx, 0, max(0, min(12, output_size) - 1)))
        # The replacement may be HOLD, MANAGE or opposite direction, but it is
        # never represented as a negative old-action weight.
        proposals.append({"action_index": idx, "cause": cause, "strength": float(np.clip(strength, 0.05, 1.0))})

    # 2) Cause-conditioned neural priors.  These are intentionally replacement
    # priors, not suppression rules.  They seed proposal cells that must still
    # compete by evidence/reliability in query.
    if any(k in cause_text for k in ("direction", "wrong_direction", "adverse_direction", "trend")):
        add(_v8_opposite_direction(old_action), "direction_replacement", 0.75)
        add(0, "direction_wait_confirmation", 0.35)
    if any(k in cause_text for k in ("entry", "late", "early", "stress_entry", "overtrading")):
        add(0, "entry_timing_wait", 0.55)
        if old_action in {1, 2}:
            add(3 if old_action == 1 else 4, "entry_timing_limit_probe", 0.35)
    if any(k in cause_text for k in ("exit", "trailing", "missed_trailing", "hold", "duration")):
        add(11, "position_management_replacement", 0.65)
        add(8, "exit_worst_or_reduce", 0.35)
    if any(k in cause_text for k in ("risk", "drawdown", "size", "exposure", "stress")):
        add(11, "risk_reduce_manage", 0.65)
        add(0, "risk_wait", 0.45)
    if any(k in cause_text for k in ("spread", "slippage", "execution", "latency")):
        add(0, "execution_wait", 0.70)
        add(7, "execution_cancel_orders", 0.40)
    if any(k in cause_text for k in ("regime", "novelty", "memory", "uncertain", "unexplained")):
        add(0, "regime_wait_probe", 0.45)
        # probe in both directions with small strength if no clear direction.
        if old_action not in {1, 2}:
            add(1, "regime_buy_probe", 0.18)
            add(2, "regime_sell_probe", 0.18)

    if not proposals:
        # Unknown failure: generate a low-risk alternative rather than reducing
        # the old action.  This keeps learning generative.
        add(0, "unknown_cause_wait", 0.35)
        if old_action in {1, 2}:
            add(_v8_opposite_direction(old_action), "unknown_cause_opposite_probe", 0.18)

    # Merge duplicate action indexes by taking max strength and concatenating causes.
    merged: dict[int, dict[str, object]] = {}
    for prop in proposals:
        idx = int(prop["action_index"])
        if idx not in merged:
            merged[idx] = dict(prop)
        else:
            merged[idx]["strength"] = max(float(merged[idx].get("strength", 0.0)), float(prop.get("strength", 0.0)))
            merged[idx]["cause"] = str(merged[idx].get("cause", "")) + "+" + str(prop.get("cause", ""))
    return list(merged.values())[:5]


def _v8_record_replacement_learning(self, episode_id: str) -> None:
    episode = self.neural_cells.get(str(episode_id))
    if episode is None or str(episode.role) != "episodic_memory":
        return
    meta = dict(episode.meta or {})
    if str(meta.get("action_origin", "")) != "policy":
        return
    raw_utility = float(meta.get("raw_utility", episode.utility) or 0.0)
    pred_error = float(meta.get("prediction_error", 0.0) or 0.0)
    if raw_utility >= 0.0 and pred_error < 1.0:
        # Success is handled as imprinting, not replacement.
        _v8_record_success_imprint(self, episode_id)
        return
    proposals = _v8_replacement_actions_from_causal(self, episode, output_size=20)
    if not proposals:
        return
    action_vec = np.asarray(meta.get("action_vector", np.zeros(20)), dtype=float)
    old_action = _v8_action_index_from_vector(action_vec, action_count=12)
    lesson = copylib.deepcopy(dict(meta.get("causal_lesson", {}) or {}))
    signal_id = stable_identifier("replacement_signal", int(getattr(self, "context_step", 0)), str(episode_id), length=20)
    signal_vec = self._resize(np.concatenate([
        self._resize(episode.vector, self.memory_key_size)[: max(1, self.memory_key_size - 8)],
        np.asarray([raw_utility, pred_error, float(old_action), float(len(proposals)), float(episode.evidence_against), float(episode.uncertainty), 1.0, 0.0], dtype=float),
    ]), self.memory_key_size)
    signal = _v6_upsert_cell(
        self,
        signal_id,
        "causal_replacement_signal",
        signal_vec,
        utility=abs(raw_utility) + 0.15 * pred_error,
        evidence_for=max(0.0, abs(raw_utility)),
        evidence_against=0.0,
        uncertainty=float(np.clip(episode.uncertainty, 0.0, 1.0)),
        activation=float(np.tanh(abs(raw_utility) + pred_error)),
        meta={"episode_cell": episode_id, "old_action_index": old_action, "causal_lesson": lesson, "runtime": "pure_neural_v8", "replacement_not_suppression": True},
    )
    self.neural_synapses.append(_v6_synapse(self, episode_id, signal_id, "episode_to_replacement_signal", weight=1.0, utility=float(signal.utility)))
    for prop in proposals:
        action_index = int(prop["action_index"])
        strength = float(prop.get("strength", 0.25))
        vec = self._resize(episode.vector + 0.05 * _v8_action_vector(self, self.memory_key_size, action_index, strength), self.memory_key_size)
        prop_id = stable_identifier("replacement_action", signal_id, action_index, str(prop.get("cause", "")), length=20)
        prop_cell = _v6_upsert_cell(
            self,
            prop_id,
            "replacement_action_proposal",
            vec,
            utility=strength * (abs(raw_utility) + 0.1 * pred_error),
            evidence_for=max(0.0, strength * abs(raw_utility)),
            evidence_against=0.0,
            uncertainty=float(np.clip(0.35 + 0.35 * episode.uncertainty, 0.0, 1.0)),
            activation=strength,
            meta={
                "episode_cell": episode_id,
                "source_signal": signal_id,
                "old_action_index": old_action,
                "replacement_action_index": action_index,
                "replacement_action_vector": _v8_action_vector(self, 20, action_index, strength).tolist(),
                "cause": str(prop.get("cause", "unknown")),
                "strength": strength,
                "runtime": "pure_neural_v8",
                "replacement_not_suppression": True,
            },
        )
        self.neural_synapses.append(_v6_synapse(self, signal_id, prop_id, "replacement_signal_to_action_proposal", weight=strength, utility=float(prop_cell.utility)))


def _v8_record_success_imprint(self, episode_id: str) -> None:
    episode = self.neural_cells.get(str(episode_id))
    if episode is None or str(episode.role) != "episodic_memory":
        return
    meta = dict(episode.meta or {})
    if str(meta.get("action_origin", "")) != "policy":
        return
    raw_utility = float(meta.get("raw_utility", episode.utility) or 0.0)
    if raw_utility <= 0.0:
        return
    action_vec = self._resize(np.asarray(meta.get("action_vector", np.zeros(20)), dtype=float), 20)
    action_index = _v8_action_index_from_vector(action_vec, action_count=12)
    imprint_id = stable_identifier("success_imprint", action_index, str(episode_id), length=20)
    imprint = _v6_upsert_cell(
        self,
        imprint_id,
        "success_action_imprint",
        episode.vector,
        utility=raw_utility,
        evidence_for=max(0.0, raw_utility),
        evidence_against=0.0,
        uncertainty=max(0.05, float(episode.uncertainty) * 0.7),
        activation=float(np.tanh(raw_utility)),
        meta={
            "episode_cell": episode_id,
            "action_index": action_index,
            "action_vector": action_vec.tolist(),
            "runtime": "pure_neural_v8",
            "success_imprinting": True,
        },
    )
    self.neural_synapses.append(_v6_synapse(self, episode_id, imprint_id, "episode_to_success_imprint", weight=1.0, utility=float(imprint.utility)))


def _v8_pure_neural_lifecycle_payload(
    self,
    trace: StepTrace,
    reward: float,
    cost: float,
    raw_utility: float,
) -> dict[str, object]:
    """Encode position/lifecycle evidence directly in neural-cell metadata."""
    details = dict(getattr(trace, "action_details", {}) or {})
    positions = self.parse_positions_pipe(getattr(trace, "positions_pipe", ""))
    positions_after = self.parse_positions_pipe(getattr(trace, "positions_pipe_after", ""))
    if not positions:
        return {"lifecycle_tags": []}
    try:
        target_slot = int(details.get("target_slot", 0) or 0)
    except (TypeError, ValueError):
        target_slot = 0
    target = next((item for item in positions if int(item.get("slot", -1)) == target_slot), positions[0])
    after = next((item for item in positions_after if str(item.get("ticket", "")) == str(target.get("ticket", ""))), None)
    classifier = self.position_outcome_classifier
    classifier.ensure_compatible()
    td_delta = float(getattr(trace, "td_delta", 0.0) or 0.0)
    features = classifier.features_from(target, after, float(reward), td_delta, float(raw_utility))
    neural_tags, tag_probs = classifier.classify(features)
    mfe_r = float(target.get("mfe_r", target.get("unrealized_r", 0.0)) or 0.0)
    after_r = float(after.get("unrealized_r", 0.0) or 0.0) if after is not None else None
    giveback = float(target.get("giveback_r", max(0.0, mfe_r - float(target.get("unrealized_r", 0.0))) or 0.0))
    if after_r is not None:
        giveback = max(giveback, mfe_r - after_r)
    near_tp = bool(target.get("near_tp", False)) or float(target.get("min_dist_tp_atr", 999.0) or 999.0) <= 0.25
    audit: list[str] = []
    if giveback >= 0.75 and mfe_r >= 1.0:
        audit.append("gave_back_profit")
    if near_tp and giveback >= 0.5:
        audit.append("near_tp_giveback")
    if near_tp and (raw_utility < 0.0 or reward < 0.0 or td_delta < 0.0):
        audit.append("near_tp_missed")
    if after is None and near_tp and (raw_utility < 0.0 or reward < 0.0):
        audit.append("closed_after_near_tp_bad")
    if "near_tp_missed" in audit or "gave_back_profit" in audit:
        audit.append("should_have_partial_or_trail")
    target_vec = np.zeros(len(classifier.labels), dtype=float)
    for idx, label in enumerate(classifier.labels):
        target_vec[idx] = 1.0 if label in audit else 0.0
    classifier.observe(features, target_vec, strength=1.0 + abs(raw_utility) + abs(td_delta))
    return {
        "target_ticket": str(target.get("ticket", "")),
        "target_slot": int(target.get("slot", target_slot)),
        "position_side": int(target.get("side", 0) or 0),
        "mfe_r": mfe_r,
        "mae_r": float(target.get("mae_r", target.get("unrealized_r", 0.0)) or 0.0),
        "giveback_r": float(max(0.0, giveback)),
        "near_tp": bool(near_tp),
        "closed_after": bool(after is None),
        "after_unrealized_r": after_r,
        "lifecycle_tags": sorted(set(neural_tags) | set(audit)),
        "audit_lifecycle_tags": sorted(set(audit)),
        "neural_lifecycle_probs": {label: float(prob) for label, prob in zip(classifier.labels, tag_probs)},
    }


def _v8_enrich_pure_neural_causal_cells(
    self,
    episode_id: str,
    trace: StepTrace,
    reward: float,
    cost: float,
    signal: dict[str, float],
) -> None:
    episode = self.neural_cells.get(str(episode_id))
    if episode is None:
        return
    ep_meta = dict(episode.meta or {})
    raw_utility = float(ep_meta.get("raw_utility", episode.utility) or 0.0)
    payload = _v8_pure_neural_lifecycle_payload(self, trace, reward, cost, raw_utility)
    lesson = dict(ep_meta.get("causal_lesson", {}) or {})
    lesson.update(payload)
    lesson["action_name"] = str(dict(getattr(trace, "action_details", {}) or {}).get("action_name", lesson.get("action_name", "")))
    lesson["reward"] = float(reward)
    lesson["cost"] = float(cost)
    lesson["td_delta"] = float(signal.get("td_delta", lesson.get("td_delta", 0.0)) or 0.0)
    lesson["raw_utility"] = raw_utility
    # Causal outcome cells must retain the complete context schema used to
    # create their episode, otherwise account-dependent lessons cannot be
    # audited or safely replayed after serialization.
    lesson["memory_key_schema_version"] = int(getattr(self, "memory_key_schema_version", MEMORY_KEY_SCHEMA_VERSION))
    lesson["account_context_before"] = {
        str(k): float(v) for k, v in dict(getattr(trace, "account_context", {}) or {}).items()
    }
    lesson["account_context_after"] = {
        str(k): float(v) for k, v in dict(getattr(trace, "account_context_after", {}) or {}).items()
    }
    ep_meta["causal_lesson"] = lesson
    ep_meta["lifecycle_tags"] = list(payload.get("lifecycle_tags", []) or [])
    episode.meta = ep_meta

    causal_cells = [
        cell for cell in self.neural_cells.values()
        if str(cell.role) == "causal_outcome" and str(dict(cell.meta or {}).get("episode_cell", "")) == str(episode_id)
    ]
    force_causal = bool(payload.get("lifecycle_tags", []))
    if not causal_cells and force_causal:
        causal_id = stable_identifier("causal_outcome", episode_id, tuple(payload.get("lifecycle_tags", []) or []), length=20)
        causal = _v6_upsert_cell(
            self,
            causal_id,
            "causal_outcome",
            episode.vector,
            utility=float(episode.utility),
            evidence_for=max(0.0, float(episode.utility)),
            evidence_against=max(0.0, -float(episode.utility)),
            uncertainty=float(episode.uncertainty),
            activation=float(np.tanh(raw_utility)),
            meta={"episode_cell": str(episode_id), "runtime": "pure_neural_v8", **lesson},
        )
        self.neural_synapses.append(_v6_synapse(self, str(episode_id), str(causal.cell_id), "episodic_to_causal", weight=float(np.tanh(raw_utility)), utility=float(episode.utility)))
        causal_cells = [causal]
    for causal in causal_cells:
        meta = dict(causal.meta or {})
        meta.update(lesson)
        causal.meta = meta


def _v8_causal_plasticity_signals(self, max_items: int = 16, min_abs_utility: float = 0.05) -> list[dict[str, object]]:
    """Expose causal evidence from the active neural substrate only."""
    self.ensure_compatible()
    signals: list[dict[str, object]] = []
    for cell_id, causal in self.neural_cells.items():
        if str(causal.role) != "causal_outcome":
            continue
        meta = dict(causal.meta or {})
        tags = {str(item) for item in list(meta.get("lifecycle_tags", []) or [])}
        utility = float(meta.get("raw_utility", causal.utility) or causal.utility)
        td_delta = float(meta.get("td_delta", 0.0) or 0.0)
        reward = float(meta.get("reward", 0.0) or 0.0)
        is_bad = utility < -float(min_abs_utility) or td_delta < -float(min_abs_utility) or reward < -float(min_abs_utility)
        lifecycle_bad = bool(tags & {"near_tp_missed", "near_tp_giveback", "gave_back_profit", "closed_after_near_tp_bad", "should_have_partial_or_trail"})
        if not (is_bad or lifecycle_bad):
            continue
        episode = self.neural_cells.get(str(meta.get("episode_cell", "")))
        episode_meta = dict(getattr(episode, "meta", {}) or {}) if episode is not None else {}
        action_vec = self._resize(np.asarray(episode_meta.get("action_vector", np.zeros(20)), dtype=float), 20)
        action_index = _v8_action_index_from_vector(action_vec, action_count=12)
        action_name = str(meta.get("action_name", "") or episode_meta.get("action_name", ""))
        severity = float(np.clip(abs(utility) + abs(td_delta) + abs(reward) + 0.25 * float(meta.get("giveback_r", 0.0) or 0.0), 0.05, 5.0))
        signals.append({
            "outcome_id": str(cell_id),
            "episode_id": str(meta.get("episode_cell", "")),
            "action_name": action_name,
            "action_index": action_index,
            "management_mode": str(meta.get("management_mode", "")),
            "target_slot": int(meta.get("target_slot", 0) or 0),
            "tags": sorted(tags),
            "severity": severity,
            "utility": utility,
            "td_delta": td_delta,
            "reward": reward,
            "giveback_r": float(meta.get("giveback_r", 0.0) or 0.0),
            "near_tp": bool(meta.get("near_tp", False)),
            "obs_embedding": np.asarray(episode.vector if episode is not None else causal.vector, dtype=float).copy(),
            "action_embedding": action_vec.copy(),
        })
    signals.sort(key=lambda item: float(item["severity"]), reverse=True)
    return signals[:max(0, int(max_items))]


def _v8_absorb_transition(self, trace: StepTrace, next_obs: np.ndarray, reward: float, cost: float, signal: dict[str, float]) -> str:
    cell_id = _V7_ABSORB_TRANSITION(self, trace, next_obs, reward, cost, signal)
    if cell_id:
        _v8_enrich_pure_neural_causal_cells(self, cell_id, trace, reward, cost, signal)
        _v8_record_replacement_learning(self, cell_id)
    return cell_id


def _v8_query(
    self,
    state: np.ndarray,
    *legacy_args: object,
    output_size: int = 20,
    module_count: int = 0,
    account_context: dict[str, object] | None = None,
    positions_pipe: str | None = "",
    **extra_context: object,
) -> dict[str, object]:
    # Start from v7 result to preserve all memory_context/critic/growth fields,
    # then remove negative action suppression and add replacement/success action
    # synthesis from pure neural proposal cells.
    result = _V7_QUERY(self, state, *legacy_args, output_size=output_size, module_count=module_count, account_context=account_context, positions_pipe=positions_pipe, **extra_context)
    if len(legacy_args) >= 1 and legacy_args[0] is not None:
        output_size = int(legacy_args[0])
    if len(legacy_args) >= 2 and legacy_args[1] is not None:
        module_count = int(legacy_args[1])
    self.ensure_compatible()
    current, delta, short, regime = self._observe_context(self._memory_key(state, account_context, positions_pipe), advance=False)
    # Recompute a generative action bias from positive proposal cells only.
    action_bias = np.zeros(output_size, dtype=float)
    replacement_matches: list[dict[str, object]] = []
    success_matches: list[dict[str, object]] = []
    total_weight = 0.0
    for cell_id, cell in list(self.neural_cells.items()):
        role = str(cell.role)
        if role not in {"replacement_action_proposal", "success_action_imprint"}:
            continue
        sim = self._cell_multi_scale_similarity(current, delta, short, regime, cell)
        if sim <= 1e-8:
            continue
        reliability, evidence_direction, uncertainty = self._cell_reliability(cell)
        if evidence_direction <= 0.0 and float(cell.utility) <= 0.0:
            continue
        meta = dict(cell.meta or {})
        if role == "replacement_action_proposal":
            proposal_vec = self._resize(np.asarray(meta.get("replacement_action_vector", np.zeros(output_size)), dtype=float), output_size)
            strength = float(meta.get("strength", abs(float(cell.activation)) + 0.1) or 0.1)
            scale = 0.32
        else:
            proposal_vec = self._resize(np.asarray(meta.get("action_vector", np.zeros(output_size)), dtype=float), output_size)
            strength = max(0.05, abs(float(cell.activation)) + max(0.0, float(cell.utility)))
            scale = 0.26
        weight = float(scale * sim * max(0.0, reliability) * max(0.05, strength) * (1.0 - 0.45 * uncertainty))
        if weight <= 1e-9:
            continue
        action_bias += weight * proposal_vec
        total_weight += weight
        item = {"cell_id": cell_id, "role": role, "similarity": float(sim), "weight": weight, "utility": float(cell.utility), "cause": str(meta.get("cause", "")), "replacement_action_index": int(meta.get("replacement_action_index", meta.get("action_index", 0)) or 0)}
        if role == "replacement_action_proposal":
            replacement_matches.append(item)
        else:
            success_matches.append(item)
    if total_weight > 1e-9:
        action_bias = 0.22 * np.clip(action_bias / total_weight, -1.0, 1.0)
    # Override v7 action_bias so negative memories cannot push an old action
    # downward.  Causal/loss cells still affect critic_bias, memory_context and
    # reflection via v7 result.
    result["action_bias"] = action_bias
    result["replacement_action_bias"] = action_bias.copy()
    result["negative_action_suppression_disabled"] = True
    result["replacement_learning"] = True
    result["replacement_matches"] = sorted(replacement_matches, key=lambda x: float(x["weight"]), reverse=True)[:5]
    result["success_imprint_matches"] = sorted(success_matches, key=lambda x: float(x["weight"]), reverse=True)[:5]
    result["runtime"] = "pure_neural_v8_replacement_learning"
    result["neural_substrate_version"] = 8
    return result


def _v7_enforce_neural_memory_budgets(self) -> None:
    """Trim retained neural memory after growth without losing live references."""
    cell_cap = getattr(self, "max_neurons", None)
    if cell_cap is not None and int(cell_cap) > 0 and len(self.neural_cells) > int(cell_cap):
        protected_roles = {"episodic_memory", "causal_attractor", "replacement_action_proposal"}
        ranked = sorted(
            self.neural_cells.items(),
            key=lambda item: (
                1 if str(item[1].role) in protected_roles else 0,
                float(getattr(item[1], "evidence_for", 0.0)) - float(getattr(item[1], "evidence_against", 0.0))
                + 0.2 * abs(float(getattr(item[1], "utility", 0.0))),
                int(getattr(item[1], "age", 0)),
                str(item[0]),
            ),
            reverse=True,
        )
        keep_ids = {cell_id for cell_id, _ in ranked[: int(cell_cap)]}
        self.neural_cells = {cell_id: cell for cell_id, cell in self.neural_cells.items() if cell_id in keep_ids}
        self.neural_synapses = [syn for syn in self.neural_synapses if syn.source in keep_ids and syn.target in keep_ids]
        self.recent_ids = [cell_id for cell_id in self.recent_ids if cell_id in keep_ids]
    connection_cap = getattr(self, "max_connections", None)
    if connection_cap is not None and int(connection_cap) > 0 and len(self.neural_synapses) > int(connection_cap):
        self.neural_synapses = sorted(
            self.neural_synapses,
            key=lambda syn: (abs(float(getattr(syn, "utility", 0.0))) + abs(float(getattr(syn, "weight", 0.0))), -int(getattr(syn, "age", 0))),
            reverse=True,
        )[: int(connection_cap)]


def _v8_develop(self) -> None:
    _V7_DEVELOP(self)
    # Hebbian support between replacement signals and proposals; no negative
    # old-action weakening is performed here.
    for syn in list(getattr(self, "neural_synapses", []) or []):
        if str(syn.kind) == "replacement_signal_to_action_proposal":
            src = self.neural_cells.get(syn.source)
            dst = self.neural_cells.get(syn.target)
            if src is None or dst is None:
                continue
            syn.weight = float(np.clip(0.997 * float(syn.weight) + 0.01 * max(0.0, float(src.activation)) * max(0.0, float(dst.activation)), -8.0, 8.0))
            syn.utility = float(0.98 * float(syn.utility) + 0.02 * max(0.0, float(dst.utility)))
    _v7_enforce_neural_memory_budgets(self)
    self.neural_substrate_version = 8
    self.pure_neural_memory = True


def _v8_reflect_memory(self, recent_window: int = 128) -> dict[str, object]:
    reflection = _V7_REFLECT_MEMORY(self, recent_window=recent_window)
    role_counts: dict[str, int] = {}
    for c in self.neural_cells.values():
        role_counts[str(c.role)] = role_counts.get(str(c.role), 0) + 1
    reflection["replacement_learning"] = True
    reflection["negative_action_suppression_disabled"] = True
    reflection["replacement_action_proposals"] = int(role_counts.get("replacement_action_proposal", 0))
    reflection["success_action_imprints"] = int(role_counts.get("success_action_imprint", 0))
    reflection["recommendations"] = list(reflection.get("recommendations", []) or [])
    if role_counts.get("replacement_action_proposal", 0) <= 0:
        reflection["recommendations"].append("collect_policy_origin_negative_outcomes_to_seed_replacement_actions")
    if getattr(self, "reflection_log", None):
        self.reflection_log[-1] = copylib.deepcopy(reflection)
    return reflection


def _v8_summary(self) -> dict[str, object]:
    result = _V7_SUMMARY(self)
    role_counts: dict[str, int] = {}
    for c in self.neural_cells.values():
        role_counts[str(c.role)] = role_counts.get(str(c.role), 0) + 1
    result.update({
        "runtime": "pure_neural_v8_replacement_learning",
        "neural_substrate_version": 8,
        "replacement_learning": True,
        "negative_action_suppression_disabled": True,
        "replacement_action_proposals": int(role_counts.get("replacement_action_proposal", 0)),
        "causal_replacement_signals": int(role_counts.get("causal_replacement_signal", 0)),
        "success_action_imprints": int(role_counts.get("success_action_imprint", 0)),
        # Compatibility diagnostics: aliases are views of neural synapses/cells,
        # never a second symbolic runtime.
        "connection_kinds": dict(result.get("neural_synapse_kinds", {}) or {}),
        "causal_kinds": {
            str(role).removeprefix("causal_"): int(count)
            for role, count in role_counts.items() if str(role).startswith("causal_")
        },
        "replacement_principle": "bad_outcome_generates_cause_conditioned_alternatives_not_negative_old_action_bias",
    })
    return result


ExperienceSuperNetwork.absorb_transition = _v8_absorb_transition
ExperienceSuperNetwork.query = _v8_query
ExperienceSuperNetwork.develop = _v8_develop
ExperienceSuperNetwork.reflect_memory = _v8_reflect_memory
ExperienceSuperNetwork.summary = _v8_summary
ExperienceSuperNetwork.causal_plasticity_signals = _v8_causal_plasticity_signals

# ---------------------------------------------------------------------------
# M1 v11: Replacement learning for trade parameters (SL/TP/risk/trailing/etc.)
# ---------------------------------------------------------------------------
# Principle: a bad lifecycle must not globally suppress the selected action.
# It must search for a cause-conditioned replacement in the parameter vector:
# risk_pct, SL distance, TP distance, entry offset, trailing, partial close,
# management mode and target slot.  The old action remains available; its
# parameter pathway is replaced only for similar contexts.

_M1V11_ORIGINAL_APPLY_EXECUTION_LIFECYCLE_CREDIT = SelfGrowingPolicy.apply_execution_lifecycle_credit
_M1V11_ORIGINAL_APPLY_BIOLOGICAL_ACTION_HEADS = SelfGrowingPolicy._apply_biological_action_heads
_M1V11_ORIGINAL_ENSURE_COMPATIBLE = SelfGrowingPolicy.ensure_compatible


def _m1v11_parameter_indices(output_size: int) -> tuple[int, int]:
    parameter_offset = PARAMETER_OFFSET if int(output_size) >= OUTPUT_COUNT else 11
    return parameter_offset, max(0, int(output_size) - parameter_offset)


def _m1v11_extract_parameter_vector(policy: SelfGrowingPolicy, action: np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=float).reshape(-1)
    parameter_offset, param_count = _m1v11_parameter_indices(arr.size)
    out = np.zeros(8, dtype=float)
    if param_count > 0:
        kept = min(8, param_count)
        out[:kept] = arr[parameter_offset:parameter_offset + kept]
    return np.clip(out, -1.0, 1.0)


def _m1v11_action_index_from_trace(policy: SelfGrowingPolicy, trace: StepTrace) -> int:
    action_count = min(ACTION_COUNT, max(1, int(getattr(policy, "output_size", OUTPUT_COUNT))))
    try:
        return int(policy._trace_action_index(trace, action_count))
    except Exception:
        arr = np.asarray(getattr(trace, "selected_action", np.zeros(action_count)), dtype=float).reshape(-1)
        return int(np.argmax(arr[:min(action_count, arr.size)])) if arr.size else 0


def _m1v11_parameter_replacement_target(
    policy: SelfGrowingPolicy,
    trace: StepTrace,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return a replacement parameter target and a neural cause summary.

    Parameter dims follow orders.py decode schema:
      0 risk_pct raw      higher -> larger risk
      1 sl_atr raw        higher -> wider SL
      2 tp_atr raw        higher -> wider TP
      3 entry_offset raw  changes entry waiting/price offset
      4 trailing raw      higher -> wider/active trailing range
      5 partial raw       higher -> stronger partial-close preference
      6 management raw    higher -> breakeven/modify/partial mode
      7 position_scale    temporarily disabled target-slot selector
    """
    base = _m1v11_extract_parameter_vector(policy, np.asarray(getattr(trace, "selected_action", np.zeros(policy.output_size)), dtype=float))
    target = base.copy()
    cause: list[str] = []
    reason = str(close_reason or "").upper()
    outcome = float(realized_r)
    duration = float(duration_bars) if duration_bars is not None else float("nan")
    short_hold = bool(duration_bars is not None and duration < max(1, int(min_holding_bars)) * 2.0)
    long_hold = bool(duration_bars is not None and duration > max(8, int(min_holding_bars) * 6))
    account_context = dict(getattr(trace, "account_context", {}) or {})
    floating = float(account_context.get("floating_pnl_r", 0.0) or 0.0)
    open_risk = float(account_context.get("open_risk_ratio", 0.0) or 0.0)
    drawdown = float(account_context.get("current_equity_drawdown", 0.0) or 0.0)
    stress = float(np.clip(drawdown + 0.5 * open_risk + 0.25 * max(0.0, -floating), 0.0, 2.0))

    # Loss causes: replace parameters, never suppress selected action.
    if outcome < 0.0:
        # Universal bad-outcome safety: risk size must be learned separately from direction.
        target[0] = np.clip(target[0] - 0.35 - 0.25 * stress, -1.0, 1.0)
        cause.append("risk_size_replacement")
        if reason in {"SL", "STOP_LOSS", "STOP_OUT"} or "SL" in reason:
            if short_hold:
                # Stop hit too quickly: likely stop too tight or entry too early.
                target[1] = np.clip(target[1] + 0.45, -1.0, 1.0)  # wider SL
                target[3] = np.clip(target[3] * 0.25, -1.0, 1.0)  # less aggressive offset
                target[6] = np.clip(target[6] + 0.25, -1.0, 1.0)  # allow manage/breakeven mode
                cause.extend(["sl_too_tight", "entry_timing_replacement"])
            else:
                # Stop hit after enough time: direction/regime may be wrong, but parameter fix is smaller size.
                target[0] = np.clip(target[0] - 0.25, -1.0, 1.0)
                target[4] = np.clip(target[4] + 0.20, -1.0, 1.0)
                cause.extend(["late_sl_regime_or_direction", "trail_or_reduce_risk"])
        elif reason in {"EXPERT", "CLOSE", "MANUAL"} or "EXPERT" in reason:
            # Expert exit at loss: management decision poor; strengthen manage/partial/trailing.
            target[4] = np.clip(target[4] + 0.35, -1.0, 1.0)
            target[5] = np.clip(target[5] + 0.30, -1.0, 1.0)
            target[6] = np.clip(target[6] + 0.45, -1.0, 1.0)
            cause.extend(["bad_management_exit", "partial_or_trailing_replacement"])
        else:
            # Unknown loss: reduce size, require more management, leave action untouched.
            target[0] = np.clip(target[0] - 0.25, -1.0, 1.0)
            target[6] = np.clip(target[6] + 0.20, -1.0, 1.0)
            cause.append("unknown_loss_parameter_probe")

    # Positive causes: imprint successful parameters rather than changing action.
    elif outcome > 0.0:
        cause.append("successful_parameter_imprint")
        if reason in {"TP", "TAKE_PROFIT"} or "TP" in reason:
            # TP hit: keep current structure; gently strengthen RR/trailing pattern.
            target[2] = np.clip(target[2] + 0.12, -1.0, 1.0)
            target[4] = np.clip(target[4] + 0.08, -1.0, 1.0)
            cause.append("tp_structure_success")
        elif reason in {"EXPERT", "CLOSE", "MANUAL"} or "EXPERT" in reason:
            # Expert profit: successful management; imprint management mode.
            target[4] = np.clip(target[4] + 0.15, -1.0, 1.0)
            target[5] = np.clip(target[5] + 0.15, -1.0, 1.0)
            target[6] = np.clip(target[6] + 0.20, -1.0, 1.0)
            cause.append("management_success_imprint")
        if long_hold and outcome < 0.75:
            # Profit but inefficient: prefer earlier partial/trailing next time.
            target[5] = np.clip(target[5] + 0.25, -1.0, 1.0)
            target[4] = np.clip(target[4] + 0.20, -1.0, 1.0)
            cause.append("slow_profit_partial_replacement")

    # Near-flat / noisy outcome: convert to probe-sized parameters.
    else:
        target[0] = np.clip(target[0] - 0.20, -1.0, 1.0)
        target[1] = np.clip(target[1] + 0.10, -1.0, 1.0)
        target[6] = np.clip(target[6] + 0.10, -1.0, 1.0)
        cause.append("flat_outcome_probe_parameters")

    # High stress always favours risk/position-management replacements.
    if stress > 0.35:
        target[0] = np.clip(target[0] - 0.20 * stress, -1.0, 1.0)
        target[5] = np.clip(target[5] + 0.12 * stress, -1.0, 1.0)
        target[6] = np.clip(target[6] + 0.18 * stress, -1.0, 1.0)
        cause.append("stress_parameter_replacement")

    action_index = _m1v11_action_index_from_trace(policy, trace)
    summary = {
        "runtime": "m1_v11_parameter_replacement",
        "action_index": int(action_index),
        "realized_r": float(outcome),
        "close_reason": str(close_reason),
        "duration_bars": None if duration_bars is None else float(duration_bars),
        "stress": float(stress),
        "causes": sorted(set(cause)),
        "old_parameters": base.tolist(),
        "replacement_parameters": target.tolist(),
        "replacement_not_action_suppression": True,
    }
    return target, summary


def _m1v11_parameter_replacement_context(
    policy: SelfGrowingPolicy,
    trace: StepTrace,
    target: np.ndarray,
    cause_summary: dict[str, object],
) -> np.ndarray:
    memory_query = policy._query_memory(getattr(trace, "encoded_state", np.zeros(policy.input_size)))
    gates_arr = np.asarray(getattr(trace, "gate_values", np.zeros(len(policy.modules))), dtype=float)
    return policy._action_head_context(
        np.asarray(getattr(trace, "encoded_state", np.zeros(policy.input_size)), dtype=float),
        np.asarray(getattr(trace, "selected_action", np.zeros(policy.output_size)), dtype=float),
        memory_query,
        gates_arr,
    )


def _m1v11_scope_from_trace(policy: SelfGrowingPolicy, trace: StepTrace, action_index: int | None = None) -> dict[str, object]:
    idx = _m1v11_action_index_from_trace(policy, trace) if action_index is None else int(action_index)
    details = dict(getattr(trace, "action_details", {}) or {})
    positions_pipe = str(getattr(trace, "positions_pipe", "") or getattr(policy, "_positions_pipe_context", "") or "")
    return action_scope(idx, action_details=details, positions_pipe=positions_pipe)


def _m1v11_scope_from_output(policy: SelfGrowingPolicy, action: np.ndarray) -> dict[str, object]:
    arr = np.asarray(action, dtype=float).reshape(-1)
    action_count = min(ACTION_COUNT, arr.size)
    idx = int(np.argmax(arr[:action_count])) if action_count else 0
    return action_scope(idx, positions_pipe=str(getattr(policy, "_positions_pipe_context", "") or ""))


def _m1v11_record_parameter_replacement_cell(
    policy: SelfGrowingPolicy,
    trace: StepTrace,
    target: np.ndarray,
    cause_summary: dict[str, object],
) -> str:
    """Store a *shadow* replacement hypothesis, never an immediate global bias."""
    mem = getattr(policy, "memory_supernet", None)
    if mem is None or not hasattr(mem, "neural_cells"):
        return ""
    try:
        mem.ensure_compatible()
        state = np.asarray(getattr(trace, "encoded_state", np.zeros(policy.input_size)), dtype=float)
        positions_pipe = str(getattr(trace, "positions_pipe", "") or getattr(policy, "_positions_pipe_context", "") or "")
        trace_schema = int(getattr(trace, "context_schema_version", 0) or 0)
        if trace_schema == int(MEMORY_KEY_SCHEMA_VERSION):
            # v8 context keys describe the world frame.  Parameter proposals
            # retain their parameters in metadata; replacing key dimensions
            # with those parameters would corrupt similarity retrieval.
            key = mem._trace_memory_key(trace)
            vec = mem._resize(key, mem.memory_key_size)
        else:
            key = mem._memory_key(state, dict(getattr(trace, "account_context", {}) or {}), positions_pipe)
            vec = mem._resize(np.concatenate([mem._resize(key, mem.memory_key_size - 8), np.asarray(target, dtype=float)]), mem.memory_key_size)
        action_index = int(cause_summary.get("action_index", 0) or 0)
        scope = _m1v11_scope_from_trace(policy, trace, action_index)
        trace_id = str(getattr(trace, "trace_id", ""))
        causes = sorted(str(cause) for cause in list(cause_summary.get("causes", []) or []))
        cell_id = stable_identifier(
            "parameter_replacement",
            trace_id,
            scope,
            tuple(np.round(np.asarray(target, dtype=float), 4).tolist()),
            causes,
        )
        baseline_r = float(cause_summary.get("realized_r", 0.0) or 0.0)
        cell = _v6_upsert_cell(
            mem,
            cell_id,
            "parameter_replacement_proposal",
            vec,
            # The original losing outcome is a hypothesis source, not positive
            # evidence for the proposed replacement.  Its paired evidence
            # begins at zero and is earned only when a trial executes it.
            utility=0.0,
            evidence_for=0.0,
            evidence_against=0.0,
            uncertainty=1.0,
            activation=0.05,
            meta={
                "runtime": "m1_v11_parameter_replacement",
                "trace_id": trace_id,
                "action_index": action_index,
                "scope": scope,
                "replacement_parameters": np.asarray(target, dtype=float).tolist(),
                "causes": causes,
                "baseline_realized_r": baseline_r,
                "validation_attempts": 0,
                "evidence_for": 0.0,
                "evidence_against": 0.0,
                "proposal_state": "shadow",
                "replacement_not_action_suppression": True,
                "context_schema_version": int(trace_schema),
                "memory_key_schema_version": int(trace_schema),
            },
        )
        signal_id = stable_identifier("parameter_replacement_signal", trace_id, scope, causes)
        signal = _v6_upsert_cell(
            mem,
            signal_id,
            "parameter_replacement_signal",
            vec,
            utility=max(0.05, abs(baseline_r)),
            evidence_for=max(0.0, -baseline_r),
            evidence_against=0.0,
            uncertainty=0.65,
            activation=float(np.tanh(abs(baseline_r))),
            meta={
                "runtime": "m1_v11_parameter_replacement",
                "source_trace": trace_id,
                "scope": scope,
            },
        )
        existing = {
            (str(syn.source), str(syn.target), str(syn.kind))
            for syn in list(getattr(mem, "neural_synapses", []) or [])
        }
        link = (signal.cell_id, cell.cell_id, "parameter_signal_to_replacement")
        if link not in existing:
            mem.neural_synapses.append(_v6_synapse(
                mem,
                signal.cell_id,
                cell.cell_id,
                "parameter_signal_to_replacement",
                weight=max(0.10, abs(baseline_r)),
                utility=0.0,
            ))
        return str(cell.cell_id)
    except Exception:
        return ""


def _m1v11_validate_parameter_replacements(
    policy: SelfGrowingPolicy,
    trace: StepTrace,
    realized_r: float,
) -> list[dict[str, object]]:
    """Update paired evidence after an executed replacement proposal.

    A replacement is promoted from ``shadow`` only after it has beaten the
    outcome that created it often enough.  It is rejected when repeated trials
    underperform that baseline.  This keeps real-market feedback causal and
    prevents a single loss from becoming a permanent parameter rule.
    """
    details = dict(getattr(trace, "action_details", {}) or {})
    proposal_ids = [str(item) for item in list(details.get("parameter_replacement_proposal_ids", []) or []) if item]
    if not proposal_ids:
        return []
    mem = getattr(policy, "memory_supernet", None)
    if mem is None or not hasattr(mem, "neural_cells"):
        return []
    current_scope = _m1v11_scope_from_trace(policy, trace)
    min_samples = max(1, int(getattr(policy, "parameter_replacement_min_samples", 3) or 3))
    min_confidence = float(np.clip(getattr(policy, "parameter_replacement_min_confidence", 0.55) or 0.55, 0.05, 0.95))
    outcomes: list[dict[str, object]] = []
    for cell_id in proposal_ids:
        cell = mem.neural_cells.get(cell_id)
        if cell is None or str(getattr(cell, "role", "")) != "parameter_replacement_proposal":
            continue
        meta = dict(getattr(cell, "meta", {}) or {})
        if not scope_matches(meta.get("scope"), current_scope):
            continue
        baseline_r = float(meta.get("baseline_realized_r", 0.0) or 0.0)
        delta = float(realized_r) - baseline_r
        positive = max(0.0, float(meta.get("evidence_for", 0.0) or 0.0))
        negative = max(0.0, float(meta.get("evidence_against", 0.0) or 0.0))
        if delta > 0.05:
            positive += min(3.0, delta)
        else:
            negative += min(3.0, abs(delta) + 0.05)
        attempts = max(0, int(meta.get("validation_attempts", 0) or 0)) + 1
        meta["evidence_for"] = positive
        meta["evidence_against"] = negative
        meta["validation_attempts"] = attempts
        meta["last_validation_r"] = float(realized_r)
        meta["last_validation_delta_r"] = float(delta)
        confidence = proposal_confidence(meta)
        state = str(meta.get("proposal_state", "shadow") or "shadow")
        if attempts >= min_samples and confidence >= min_confidence and positive > negative:
            state = "validated"
        elif attempts >= min_samples and negative > positive * 1.20:
            state = "rejected"
        meta["proposal_state"] = state
        cell.meta = meta
        cell.evidence_for = positive
        cell.evidence_against = negative
        cell.uncertainty = float(np.clip(1.0 / (1.0 + attempts), 0.05, 1.0))
        cell.utility = float(np.clip((positive - negative) / (1.0 + positive + negative), -1.0, 1.0))
        cell.activation = float(np.tanh(max(0.0, cell.utility)))
        outcomes.append({
            "cell_id": cell_id,
            "proposal_state": state,
            "validation_attempts": attempts,
            "confidence": confidence,
            "delta_r": float(delta),
        })
    return outcomes


def _m1v11_apply_parameter_replacement_credit(
    self: SelfGrowingPolicy,
    trace: StepTrace,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
    strength: float = 1.0,
) -> dict[str, object]:
    self.ensure_compatible()
    if not bool(getattr(self, "parameter_replacement_learning", True)):
        event = {
            "round": int(getattr(self, "revision", 0)),
            "type": "m1_parameter_replacement_credit_skipped",
            "path": "parameter_replacement_credit",
            "trace_id": str(getattr(trace, "trace_id", "")),
            "reason": "parameter_replacement_learning_disabled",
        }
        self.growth_events.append(event)
        return {
            "parameter_head_updated": False,
            "skipped_disabled": True,
        }
    target, summary = _m1v11_parameter_replacement_target(
        self,
        trace,
        float(realized_r),
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
    )
    magnitude = abs(float(realized_r)) + 0.25
    if float(realized_r) > 0.0:
        magnitude = 0.5 + min(2.0, float(realized_r))
    # Candidate-only adaptation.  A baseline may retain a neutral shadow
    # hypothesis, but its global parameter head is not modified by a single
    # lifecycle event.  This is the hard isolation boundary for self-growth.
    credit_strength = float(np.clip(strength * magnitude, 0.25, 3.0))
    parameter_head = getattr(self, "action_parameter_head", None)
    if bool(getattr(self, "graph_trial_mode", False)) and parameter_head is not None:
        context = _m1v11_parameter_replacement_context(self, trace, target, summary)
        parameter_head.observe(context, target, strength=credit_strength)
        summary["parameter_head_updated"] = True
    else:
        summary["parameter_head_updated"] = False
    proposal_id = _m1v11_record_parameter_replacement_cell(self, trace, target, summary)
    event = {
        "round": int(getattr(self, "revision", 0)),
        "type": "m1_parameter_replacement_credit",
        **summary,
        "proposal_id": proposal_id,
        "credit_strength": credit_strength if bool(getattr(self, "graph_trial_mode", False)) else 0.0,
        "candidate_only": True,
    }
    self.growth_events.append(event)
    return event


def _m1v11_query_parameter_replacement_bias(
    self: SelfGrowingPolicy,
    state: np.ndarray,
    memory_query: dict[str, object],
    output_size: int,
    *,
    current_scope: dict[str, object] | None = None,
    return_metadata: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    mem = getattr(self, "memory_supernet", None)
    out = np.zeros(8, dtype=float)
    metadata: dict[str, object] = {"proposal_ids": [], "mode": "none", "matched": 0}
    if mem is None or not hasattr(mem, "neural_cells"):
        return (out, metadata) if return_metadata else out
    try:
        mem.ensure_compatible()
        scope = dict(current_scope or _m1v11_scope_from_output(self, np.zeros(max(ACTION_COUNT, int(output_size)))))
        active_frame = getattr(self, "_active_context_frame", None)
        if isinstance(active_frame, ContextFrame):
            active_frame = mem.advance_context_frame(active_frame)
            current = np.asarray(active_frame.temporal_current, dtype=float)
            delta = np.asarray(active_frame.temporal_delta, dtype=float)
            short = np.asarray(active_frame.temporal_short, dtype=float)
            regime = np.asarray(active_frame.temporal_regime, dtype=float)
        else:
            current, delta, short, regime = mem._observe_context(
                mem._memory_key(state, dict(getattr(self, "_account_context", {}) or {}), getattr(self, "_positions_pipe_context", "")),
                advance=False,
            )
        total = 0.0
        proposals: list[tuple[str, object, float, str]] = []
        for cell_id, cell in list(mem.neural_cells.items()):
            if str(cell.role) != "parameter_replacement_proposal":
                continue
            meta = dict(cell.meta or {})
            if not scope_matches(meta.get("scope"), scope):
                continue
            proposal_state = str(meta.get("proposal_state", "shadow") or "shadow")
            if proposal_state == "rejected":
                continue
            attempts = max(0, int(meta.get("validation_attempts", 0) or 0))
            confidence = proposal_confidence(meta)
            is_validated = proposal_state in {"validated", "consolidated"}
            # An unvalidated proposal can be explored only inside an isolated
            # candidate and with a tightly bounded influence.
            if not is_validated and not bool(getattr(self, "graph_trial_mode", False)):
                continue
            sim = mem._cell_multi_scale_similarity(current, delta, short, regime, cell)
            if sim <= 1e-8:
                continue
            reliability, evidence_direction, uncertainty = mem._cell_reliability(cell)
            if is_validated:
                evidence_scale = max(0.15, confidence, reliability)
                mode = "validated"
                base_scale = 0.30
            else:
                evidence_scale = 1.0 / (1.0 + attempts)
                mode = "shadow_probe"
                base_scale = 0.08
            params = np.asarray(meta.get("replacement_parameters", np.zeros(8)), dtype=float).reshape(-1)
            padded = np.zeros(8, dtype=float)
            padded[:min(8, params.size)] = params[:8]
            w = float(base_scale * sim * evidence_scale * (1.0 - 0.40 * uncertainty))
            if w <= 0.0:
                continue
            out += w * np.clip(padded, -1.0, 1.0)
            total += w
            proposals.append((str(cell_id), cell, w, mode))
        if total > 1e-9:
            out = np.clip(out / total, -1.0, 1.0)
            proposals.sort(key=lambda item: float(item[2]), reverse=True)
            metadata = {
                "proposal_ids": [item[0] for item in proposals[:4]],
                "mode": "validated" if any(item[3] == "validated" for item in proposals) else "shadow_probe",
                "matched": len(proposals),
                "scope": scope,
            }
    except Exception:
        out = np.zeros(8, dtype=float)
        metadata = {"proposal_ids": [], "mode": "error", "matched": 0}
    return (out, metadata) if return_metadata else out


def _m1v11_apply_biological_action_heads(
    self: SelfGrowingPolicy,
    output: np.ndarray,
    state: np.ndarray,
    obs: np.ndarray,
    gates: np.ndarray,
    memory_query: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    adjusted, risk_info = _M1V11_ORIGINAL_APPLY_BIOLOGICAL_ACTION_HEADS(self, output, state, obs, gates, memory_query)
    parameter_offset, param_count = _m1v11_parameter_indices(adjusted.size)
    scope = _m1v11_scope_from_output(self, adjusted)
    if param_count > 0:
        replacement_params, replacement_meta = _m1v11_query_parameter_replacement_bias(
            self,
            state,
            memory_query,
            adjusted.size,
            current_scope=scope,
            return_metadata=True,
        )
        kept = min(8, param_count, replacement_params.size)
        if kept > 0 and float(np.max(np.abs(replacement_params[:kept]))) > 1e-9:
            adjusted[parameter_offset:parameter_offset + kept] = np.tanh(
                0.92 * adjusted[parameter_offset:parameter_offset + kept]
                + 0.08 * replacement_params[:kept]
            )
            risk_info = dict(risk_info)
            risk_info["parameter_replacement_applied"] = 1.0
        else:
            risk_info = dict(risk_info)
            risk_info["parameter_replacement_applied"] = 0.0
        risk_info["parameter_replacement_proposal_ids"] = list(replacement_meta.get("proposal_ids", []) or [])
        risk_info["parameter_replacement_mode"] = str(replacement_meta.get("mode", "none"))
        risk_info["parameter_replacement_scope"] = scope
    return adjusted, risk_info


def _m1v11_apply_execution_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
) -> int:
    cache = getattr(self, "_execution_trace_cache", None)
    trace = None
    if isinstance(cache, dict):
        trace = cache.get(str(trace_id))
    credited = _M1V11_ORIGINAL_APPLY_EXECUTION_LIFECYCLE_CREDIT(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
    )
    if isinstance(trace, StepTrace):
        try:
            validation = _m1v11_validate_parameter_replacements(self, trace, float(realized_r))
            if validation:
                self.growth_events.append({
                    "round": int(getattr(self, "revision", 0)),
                    "type": "m1_parameter_replacement_validation",
                    "trace_id": str(trace_id),
                    "outcomes": validation,
                })
            self._apply_parameter_replacement_credit(
                trace,
                float(realized_r),
                close_reason=close_reason,
                duration_bars=duration_bars,
                min_holding_bars=min_holding_bars,
                strength=1.0,
            )
            credited += 1
        except Exception as exc:
            self.growth_events.append({
                "round": int(getattr(self, "revision", 0)),
                "type": "m1_parameter_replacement_credit_error",
                "trace_id": str(trace_id),
                "error": str(exc),
            })
    return credited


def _m1v11_finalize_parameter_replacement_promotion(self: SelfGrowingPolicy) -> dict[str, object]:
    """Consolidate only evidence-backed parameter hypotheses on MT5 promotion."""
    audit = {"promoted": True, "validated": 0, "consolidated": 0, "shadow": 0, "rejected": 0}
    mem = getattr(self, "memory_supernet", None)
    if mem is None or not hasattr(mem, "neural_cells"):
        return audit
    for cell in mem.neural_cells.values():
        if str(getattr(cell, "role", "")) != "parameter_replacement_proposal":
            continue
        meta = dict(getattr(cell, "meta", {}) or {})
        state = str(meta.get("proposal_state", "shadow") or "shadow")
        audit[state] = int(audit.get(state, 0)) + 1
        if state == "validated":
            meta["proposal_state"] = "consolidated"
            meta["promoted_revision"] = int(getattr(self, "revision", 0))
            meta["promoted_trial_id"] = str(getattr(self, "graph_trial_id", "") or "")
            cell.meta = meta
            audit["consolidated"] = int(audit.get("consolidated", 0)) + 1
    self.growth_events.append({
        "round": int(getattr(self, "revision", 0)),
        "type": "parameter_replacement_promotion_audit",
        **audit,
    })
    return audit


def _m1v11_compact_graph_structures(
    self: SelfGrowingPolicy,
    *,
    min_disabled_edge_age: int = 3,
    min_pruning_node_age: int = 4,
    min_rejected_proposal_attempts: int = 6,
) -> dict[str, object]:
    """Physically compact only a promoted trial's already-pruning structure.

    Runtime and baseline calls are explicitly no-ops.  The method removes only
    disabled/pruning, low-utility candidate edges and isolated pruning nodes;
    it never deletes an active path or a core input/output region member.
    """
    if not bool(getattr(self, "graph_trial_mode", False)):
        return {"skipped": True, "reason": "baseline_protected", "edges_removed": 0, "neurons_removed": 0, "memory_cells_removed": 0}
    removed_edges = 0
    removed_nodes = 0
    for module in self.modules:
        graph = getattr(getattr(module, "m1_component", None), "graph", None)
        if graph is None:
            continue
        edge_ids: list[str] = []
        for edge_id, edge in list(getattr(graph, "edges", {}).items()):
            stale_disabled = (not bool(getattr(edge, "enabled", True))) and int(getattr(edge, "age", 0)) >= int(min_disabled_edge_age)
            faded_candidate = (
                str(getattr(edge, "lifecycle", "")) == "pruning"
                and int(getattr(edge, "age", 0)) >= int(min_disabled_edge_age)
                and float(getattr(edge, "fade", 1.0)) <= 0.08
                and abs(float(getattr(edge, "utility", 0.0))) <= 0.10
            )
            if stale_disabled or faded_candidate:
                edge_ids.append(str(edge_id))
        for edge_id in edge_ids:
            graph.remove_connection(edge_id)
            removed_edges += 1

        connected: set[str] = set()
        for edge in graph.edges.values():
            connected.add(str(edge.src))
            connected.add(str(edge.dst))
        node_ids: list[str] = []
        for node_id, node in list(graph.neurons.items()):
            region = str(getattr(node, "region", ""))
            # Input/output members define the graph I/O contract and must never
            # be compacted by a generic maintenance pass.
            if region in {"input", "output"}:
                continue
            region_members = list(getattr(graph.regions.get(region), "neuron_ids", []) or [])
            candidate_pruning = (
                str(getattr(node, "lifecycle", "")) == "pruning"
                and int(getattr(node, "age", 0)) >= int(min_pruning_node_age)
                and (float(getattr(node, "fade", 1.0)) <= 0.08 or abs(float(getattr(node, "utility", 0.0))) <= 0.02)
            )
            if candidate_pruning and str(node_id) not in connected and len(region_members) > 1:
                node_ids.append(str(node_id))
        for node_id in node_ids:
            graph.remove_neuron(node_id)
            removed_nodes += 1
        graph.assert_invariants(output_size=self.output_size)

    # Rejected proposals retain their journal event, but after repeated failed
    # trials they should no longer consume neural memory.  Do not remove a
    # fresh rejection: it may be revisited by audit/replay first.
    memory_removed = 0
    mem = getattr(self, "memory_supernet", None)
    if mem is not None and hasattr(mem, "neural_cells"):
        discard = []
        for cell_id, cell in list(mem.neural_cells.items()):
            if str(getattr(cell, "role", "")) != "parameter_replacement_proposal":
                continue
            meta = dict(getattr(cell, "meta", {}) or {})
            if (
                str(meta.get("proposal_state", "")) == "rejected"
                and int(meta.get("validation_attempts", 0) or 0) >= int(min_rejected_proposal_attempts)
            ):
                discard.append(str(cell_id))
        if discard:
            drop = set(discard)
            for cell_id in discard:
                mem.neural_cells.pop(cell_id, None)
            mem.neural_synapses = [
                syn for syn in list(getattr(mem, "neural_synapses", []) or [])
                if str(getattr(syn, "source", "")) not in drop and str(getattr(syn, "target", "")) not in drop
            ]
            memory_removed = len(discard)
    result = {
        "skipped": False,
        "edges_removed": removed_edges,
        "neurons_removed": removed_nodes,
        "memory_cells_removed": memory_removed,
        "trial_id": str(getattr(self, "graph_trial_id", "") or ""),
    }
    self.growth_events.append({"round": int(getattr(self, "revision", 0)), "type": "graph_compaction", **result})
    return result


def _m1v11_ensure_compatible(
    self: SelfGrowingPolicy,
    target_input_size: int | None = None,
) -> None:
    """Keep the public compatibility contract intact after the v11 patch.

    The previous wrapper silently dropped ``target_input_size``.  That broke
    candidate creation, bridge-schema migration, loading old checkpoints, and
    transition feedback because those paths resize graph I/O through the
    compatibility entrypoint.  Forward every argument to the version it wraps.
    """
    _M1V11_ORIGINAL_ENSURE_COMPATIBLE(self, target_input_size=target_input_size)
    if not hasattr(self, "parameter_replacement_learning"):
        self.parameter_replacement_learning = True
    if not hasattr(self, "parameter_replacement_min_samples"):
        self.parameter_replacement_min_samples = 3
    if not hasattr(self, "parameter_replacement_min_confidence"):
        self.parameter_replacement_min_confidence = 0.55


SelfGrowingPolicy._apply_parameter_replacement_credit = _m1v11_apply_parameter_replacement_credit
SelfGrowingPolicy._apply_biological_action_heads = _m1v11_apply_biological_action_heads
SelfGrowingPolicy.apply_execution_lifecycle_credit = _m1v11_apply_execution_lifecycle_credit
SelfGrowingPolicy.compact_graph_structures = _m1v11_compact_graph_structures
SelfGrowingPolicy.finalize_parameter_replacement_promotion = _m1v11_finalize_parameter_replacement_promotion
SelfGrowingPolicy.ensure_compatible = _m1v11_ensure_compatible

# ---------------------------------------------------------------------------
# Context v12: immutable decision frames and semantic context partitioning
# ---------------------------------------------------------------------------
# A policy now owns exactly one ContextFrame per decision.  The frame feeds
# memory, M1 graph regions, legality selection and feedback attribution.  The
# wrappers remain backward compatible with legacy callers that only supply raw
# observations/account context.

_CONTEXT_V12_MEMORY_ENSURE = ExperienceSuperNetwork.ensure_compatible
_CONTEXT_V12_MEMORY_QUERY = ExperienceSuperNetwork.query
_CONTEXT_V12_MEMORY_ABSORB = ExperienceSuperNetwork.absorb_transition
_CONTEXT_V12_MEMORY_OBSERVE = ExperienceSuperNetwork._observe_context
_CONTEXT_V12_MEMORY_TRACE_KEY = ExperienceSuperNetwork._trace_memory_key
_CONTEXT_V12_MEMORY_CELL_SIM = ExperienceSuperNetwork._cell_multi_scale_similarity
_CONTEXT_V12_POLICY_ENSURE = SelfGrowingPolicy.ensure_compatible
_CONTEXT_V12_POLICY_ACT = SelfGrowingPolicy.act
_CONTEXT_V12_POLICY_QUERY_MEMORY = SelfGrowingPolicy._query_memory
_CONTEXT_V12_POLICY_SET_ACCOUNT_CONTEXT = SelfGrowingPolicy.set_account_context


def _context_v12_cell_schema(cell: object) -> int:
    meta = dict(getattr(cell, "meta", {}) or {})
    raw = meta.get("context_schema_version", meta.get("memory_key_schema_version", 0))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 0


def _context_v12_memory_ensure(self: ExperienceSuperNetwork) -> None:
    old_schema = int(getattr(self, "memory_key_schema_version", MEMORY_KEY_SCHEMA_VERSION) or MEMORY_KEY_SCHEMA_VERSION)
    _CONTEXT_V12_MEMORY_ENSURE(self)
    # v8 changes the meaning of the 46 dimensions while retaining their size:
    # 24 pooled market + 8 execution + 8 account + 6 position.  Old cells are
    # retained for audit but excluded from v8 retrieval, never compared in a
    # false shared coordinate system.
    if old_schema != int(MEMORY_KEY_SCHEMA_VERSION):
        migrated = set(getattr(self, "legacy_context_schema_versions", set()) or set())
        migrated.add(old_schema)
        self.legacy_context_schema_versions = migrated
    self.memory_key_schema_version = int(MEMORY_KEY_SCHEMA_VERSION)
    self.memory_key_size = int(MEMORY_KEY_SIZE)
    for temporal_name in ("previous_context", "short_context_state", "regime_context_state"):
        value = np.asarray(getattr(self, temporal_name, np.zeros(MEMORY_KEY_SIZE)), dtype=float).reshape(-1)
        setattr(self, temporal_name, self._resize(value, MEMORY_KEY_SIZE))
    if not hasattr(self, "_context_clock_frame_id"):
        self._context_clock_frame_id = ""
    if not hasattr(self, "_context_clock_snapshot"):
        self._context_clock_snapshot = None
    if not hasattr(self, "_context_clock_permit"):
        self._context_clock_permit = False
    if not hasattr(self, "_active_context_frame"):
        self._active_context_frame = None


def _context_v12_observe_context(
    self: ExperienceSuperNetwork,
    context: np.ndarray,
    *,
    advance: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Outcome writers call the historical method with advance=True.  Once a
    # decision frame is active, temporal time may advance only through
    # advance_context_frame(), never through an execution/lifecycle outcome.
    snapshot = getattr(self, "_context_clock_snapshot", None)
    if advance and snapshot is not None and not bool(getattr(self, "_context_clock_permit", False)):
        return tuple(np.asarray(value, dtype=float).copy() for value in snapshot)  # type: ignore[return-value]
    return _CONTEXT_V12_MEMORY_OBSERVE(self, context, advance=advance)


def _context_v12_advance_context_frame(self: ExperienceSuperNetwork, frame: ContextFrame) -> ContextFrame:
    self.ensure_compatible()
    if not isinstance(frame, ContextFrame):
        raise TypeError("advance_context_frame requires ContextFrame")
    frame_id = str(frame.frame_id)
    existing = getattr(self, "_context_clock_snapshot", None)
    if frame_id and frame_id == str(getattr(self, "_context_clock_frame_id", "")) and existing is not None:
        advanced = frame.with_temporal(*existing)
        self._active_context_frame = advanced
        return advanced
    self._context_clock_permit = True
    try:
        current, delta, short, regime = _CONTEXT_V12_MEMORY_OBSERVE(
            self,
            np.asarray(frame.canonical_memory_key, dtype=float),
            advance=True,
        )
    finally:
        self._context_clock_permit = False
    snapshot = (
        np.asarray(current, dtype=float).copy(),
        np.asarray(delta, dtype=float).copy(),
        np.asarray(short, dtype=float).copy(),
        np.asarray(regime, dtype=float).copy(),
    )
    self._context_clock_frame_id = frame_id
    self._context_clock_snapshot = snapshot
    advanced = frame.with_temporal(*snapshot)
    self._active_context_frame = advanced
    return advanced


def _context_v12_trace_memory_key(self: ExperienceSuperNetwork, trace: StepTrace) -> np.ndarray:
    schema = int(getattr(trace, "context_schema_version", 0) or 0)
    vector = np.asarray(getattr(trace, "canonical_context_key", np.zeros(0)), dtype=float).reshape(-1)
    if schema == int(MEMORY_KEY_SCHEMA_VERSION) and vector.size:
        return self._resize(vector, self.memory_key_size)
    return _CONTEXT_V12_MEMORY_TRACE_KEY(self, trace)


def _context_v12_cell_similarity(
    self: ExperienceSuperNetwork,
    current: np.ndarray,
    delta: np.ndarray,
    short: np.ndarray,
    regime: np.ndarray,
    cell: object,
) -> float:
    if isinstance(getattr(self, "_active_context_frame", None), ContextFrame):
        if _context_v12_cell_schema(cell) != int(MEMORY_KEY_SCHEMA_VERSION):
            return 0.0
        meta = dict(getattr(cell, "meta", {}) or {})
        base = self._resize(np.asarray(getattr(cell, "vector", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        delta_proto = self._resize(np.asarray(meta.get("context_delta", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        short_proto = self._resize(np.asarray(meta.get("short_context", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        regime_proto = self._resize(np.asarray(meta.get("regime_context", np.zeros(self.memory_key_size)), dtype=float), self.memory_key_size)
        sims = np.asarray([
            max(0.0, blockwise_cosine(current, base)),
            max(0.0, blockwise_cosine(delta, delta_proto)),
            max(0.0, blockwise_cosine(short, short_proto)),
            max(0.0, blockwise_cosine(regime, regime_proto)),
        ], dtype=float)
        active = np.asarray([
            np.linalg.norm(base) > 1e-8,
            np.linalg.norm(delta_proto) > 1e-8,
            np.linalg.norm(short_proto) > 1e-8,
            np.linalg.norm(regime_proto) > 1e-8,
        ], dtype=bool)
        weights = np.asarray([0.45, 0.15, 0.20, 0.20], dtype=float)
        weights[~active] = 0.0
        return float(np.clip(float(sims @ weights) / float(weights.sum()), 0.0, 1.0)) if float(weights.sum()) > 1e-12 else 0.0
    return _CONTEXT_V12_MEMORY_CELL_SIM(self, current, delta, short, regime, cell)


def _context_v12_query_with_frame(
    self: ExperienceSuperNetwork,
    frame: ContextFrame,
    *,
    output_size: int,
    module_count: int,
) -> dict[str, object]:
    frame = self.advance_context_frame(frame)
    current = np.asarray(frame.temporal_current, dtype=float)
    delta = np.asarray(frame.temporal_delta, dtype=float)
    short = np.asarray(frame.temporal_short, dtype=float)
    regime = np.asarray(frame.temporal_regime, dtype=float)
    result = self._query_neural_substrate(
        current,
        delta,
        short,
        regime,
        output_size=max(1, int(output_size)),
        module_count=max(0, int(module_count)),
    )
    # Keep v8's generative replacement principle while querying only the
    # canonical frame coordinate system.
    action_bias = np.zeros(max(1, int(output_size)), dtype=float)
    replacement_matches: list[dict[str, object]] = []
    success_matches: list[dict[str, object]] = []
    total_weight = 0.0
    for cell_id, cell in list(getattr(self, "neural_cells", {}).items()):
        role = str(getattr(cell, "role", ""))
        if role not in {"replacement_action_proposal", "success_action_imprint"}:
            continue
        if _context_v12_cell_schema(cell) != int(MEMORY_KEY_SCHEMA_VERSION):
            continue
        sim = self._cell_multi_scale_similarity(current, delta, short, regime, cell)
        if sim <= 1e-8:
            continue
        reliability, evidence_direction, uncertainty = self._cell_reliability(cell)
        if evidence_direction <= 0.0 and float(getattr(cell, "utility", 0.0)) <= 0.0:
            continue
        meta = dict(getattr(cell, "meta", {}) or {})
        if role == "replacement_action_proposal":
            proposal = self._resize(np.asarray(meta.get("replacement_action_vector", np.zeros(output_size)), dtype=float), output_size)
            strength = float(meta.get("strength", abs(float(getattr(cell, "activation", 0.0))) + 0.1) or 0.1)
            scale = 0.32
        else:
            proposal = self._resize(np.asarray(meta.get("action_vector", np.zeros(output_size)), dtype=float), output_size)
            strength = max(0.05, abs(float(getattr(cell, "activation", 0.0))) + max(0.0, float(getattr(cell, "utility", 0.0))))
            scale = 0.26
        weight = float(scale * sim * max(0.0, reliability) * max(0.05, strength) * (1.0 - 0.45 * uncertainty))
        if weight <= 1e-9:
            continue
        action_bias += weight * proposal
        total_weight += weight
        item = {
            "cell_id": str(cell_id), "role": role, "similarity": float(sim), "weight": weight,
            "utility": float(getattr(cell, "utility", 0.0)), "cause": str(meta.get("cause", "")),
            "replacement_action_index": int(meta.get("replacement_action_index", meta.get("action_index", 0)) or 0),
        }
        (replacement_matches if role == "replacement_action_proposal" else success_matches).append(item)
    if total_weight > 1e-9:
        action_bias = 0.22 * np.clip(action_bias / total_weight, -1.0, 1.0)
    result.update({
        "action_bias": action_bias,
        "replacement_action_bias": action_bias.copy(),
        "negative_action_suppression_disabled": True,
        "replacement_learning": True,
        "replacement_matches": sorted(replacement_matches, key=lambda item: float(item["weight"]), reverse=True)[:5],
        "success_imprint_matches": sorted(success_matches, key=lambda item: float(item["weight"]), reverse=True)[:5],
        "runtime": "pure_neural_v12_context_frame",
        "neural_substrate_version": 12,
        "context_frame_id": frame.frame_id,
        "context_key_source": "canonical_context_frame",
        "context_schema_version": int(MEMORY_KEY_SCHEMA_VERSION),
        "context_step": int(self.context_step),
    })
    return result


def _context_v12_memory_query(
    self: ExperienceSuperNetwork,
    state: np.ndarray,
    *legacy_args: object,
    output_size: int = 20,
    module_count: int = 0,
    account_context: dict[str, object] | None = None,
    positions_pipe: str | None = "",
    **extra_context: object,
) -> dict[str, object]:
    frame = extra_context.pop("context_frame", None)
    if isinstance(frame, ContextFrame):
        if len(legacy_args) >= 1 and legacy_args[0] is not None:
            output_size = int(legacy_args[0])
        if len(legacy_args) >= 2 and legacy_args[1] is not None:
            module_count = int(legacy_args[1])
        self.ensure_compatible()
        return _context_v12_query_with_frame(self, frame, output_size=output_size, module_count=module_count)
    return _CONTEXT_V12_MEMORY_QUERY(
        self, state, *legacy_args, output_size=output_size, module_count=module_count,
        account_context=account_context, positions_pipe=positions_pipe, **extra_context,
    )


def _context_v12_absorb_transition(
    self: ExperienceSuperNetwork,
    trace: StepTrace,
    next_obs: np.ndarray,
    reward: float,
    cost: float,
    signal: dict[str, float],
) -> str:
    cell_id = _CONTEXT_V12_MEMORY_ABSORB(self, trace, next_obs, reward, cost, signal)
    schema = int(getattr(trace, "context_schema_version", 0) or 0)
    if not cell_id or schema != int(MEMORY_KEY_SCHEMA_VERSION):
        return cell_id
    try:
        next_frame = ContextFrame.from_observation(
            next_obs,
            frame_id=f"{getattr(trace, 'context_frame_id', '')}:next",
            sequence_id=int(getattr(self, "context_step", 0)) + 1,
            timestamp=float(time.time()),
            account_context=getattr(trace, "account_context_after", {}),
        )
        for cid, cell in list(getattr(self, "neural_cells", {}).items()):
            meta = dict(getattr(cell, "meta", {}) or {})
            if str(cid) != str(cell_id) and str(meta.get("episode_cell", "")) != str(cell_id):
                continue
            meta["context_schema_version"] = int(MEMORY_KEY_SCHEMA_VERSION)
            meta["memory_key_schema_version"] = int(MEMORY_KEY_SCHEMA_VERSION)
            meta["context_frame_id"] = str(getattr(trace, "context_frame_id", ""))
            meta["next_vector"] = np.asarray(next_frame.canonical_memory_key, dtype=float).tolist()
            cell.meta = meta
    except Exception:
        # Memory absorption must not fail a confirmed MT5 lifecycle because an
        # optional context audit field cannot be serialised.
        pass
    return cell_id


def _context_v12_policy_ensure(self: SelfGrowingPolicy, target_input_size: int | None = None) -> None:
    _CONTEXT_V12_POLICY_ENSURE(self, target_input_size=target_input_size)
    if not hasattr(self, "_decision_context") or self._decision_context is None:
        self._decision_context = {}
    if not hasattr(self, "_active_context_frame"):
        self._active_context_frame = None
    if not hasattr(self, "context_schema_version"):
        self.context_schema_version = int(MEMORY_KEY_SCHEMA_VERSION)


def _context_v12_set_decision_context(
    self: SelfGrowingPolicy,
    *,
    account_context: dict[str, float] | None = None,
    positions_pipe: str | None = "",
    allowed_actions: object | None = None,
    execution_context: dict[str, float] | None = None,
    goal: object | None = None,
    sequence_id: int | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
) -> None:
    self.ensure_compatible()
    _CONTEXT_V12_POLICY_SET_ACCOUNT_CONTEXT(self, account_context, positions_pipe)
    self._decision_context = {
        "allowed_actions": allowed_actions,
        "execution_context": dict(execution_context or {}),
        "goal": None if goal is None else np.asarray(goal, dtype=float).reshape(-1),
        "sequence_id": int(sequence_id if sequence_id is not None else getattr(self, "_step_count", 0) + 1),
        "frame_id": str(frame_id or ""),
        "timestamp": float(timestamp if timestamp is not None else time.time()),
    }


def _context_v12_build_frame(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None) -> ContextFrame:
    self.ensure_compatible()
    payload = dict(getattr(self, "_decision_context", {}) or {})
    sequence = int(payload.get("sequence_id", getattr(self, "_step_count", 0) + 1) or 0)
    frame_id = str(payload.get("frame_id", "") or f"policy:{self.revision}:{sequence}:{getattr(self, '_step_count', 0) + 1}")
    effective_goal = goal if goal is not None else payload.get("goal", None)
    frame = ContextFrame.from_observation(
        observation,
        frame_id=frame_id,
        sequence_id=sequence,
        timestamp=float(payload.get("timestamp", time.time()) or time.time()),
        account_context=dict(getattr(self, "_account_context", {}) or {}),
        execution_context=dict(payload.get("execution_context", {}) or {}),
        starvation=dict(getattr(self, "_experience_starvation", {}) or {}),
        goal=effective_goal,
        allowed_action_mask=payload.get("allowed_actions", None),
    )
    frame = self.memory_supernet.advance_context_frame(frame)
    self._active_context_frame = frame
    return frame


def _context_v12_policy_query_memory(self: SelfGrowingPolicy, state_embedding: np.ndarray) -> dict[str, object]:
    frame = getattr(self, "_active_context_frame", None)
    if isinstance(frame, ContextFrame):
        result = self.memory_supernet.query(
            np.asarray(frame.canonical_memory_key, dtype=float),
            self.output_size,
            len(self.modules),
            context_frame=frame,
        )
        return frame.as_m1_context(result)
    return _CONTEXT_V12_POLICY_QUERY_MEMORY(self, state_embedding)


def _context_v12_mask_policy_output(output: np.ndarray, mask: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(output, dtype=float).reshape(-1).copy()
    count = min(adjusted.size, mask.size, ACTION_COUNT)
    if count <= 0:
        return adjusted
    allowed = np.asarray(mask[:count], dtype=bool)
    if not np.any(allowed):
        allowed[0] = True
    legal_values = adjusted[:count][allowed]
    floor = float(np.min(legal_values) - 2.0) if legal_values.size else -2.0
    for index in range(count):
        if not bool(allowed[index]):
            adjusted[index] = floor
    return np.tanh(adjusted)


def _context_v12_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    self.ensure_compatible()
    frame = _context_v12_build_frame(self, np.asarray(observation, dtype=float), goal)
    trace = _CONTEXT_V12_POLICY_ACT(self, observation, goal)
    output = _context_v12_mask_policy_output(np.asarray(trace.selected_action, dtype=float), frame.allowed_action_mask)
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "context_frame_id": frame.frame_id,
        "context_schema_version": int(MEMORY_KEY_SCHEMA_VERSION),
        "allowed_actions": [int(index) for index, allowed in enumerate(frame.allowed_action_mask) if bool(allowed)],
        "context_key_source": "canonical_context_frame",
        "policy_output_action_index": int(np.argmax(output[:min(ACTION_COUNT, output.size)])) if output.size else 0,
        "final_action_output": output.astype(float).tolist(),
    })
    return replace(
        trace,
        selected_action=output,
        action_details=details,
        context_frame_id=frame.frame_id,
        context_schema_version=int(MEMORY_KEY_SCHEMA_VERSION),
        canonical_context_key=np.asarray(frame.canonical_memory_key, dtype=float).copy(),
        allowed_action_mask=tuple(bool(item) for item in frame.allowed_action_mask),
        goal_context=np.asarray(frame.goal_vector, dtype=float).copy(),
        learning_drive_context=np.asarray(frame.learning_drive, dtype=float).copy(),
        context_audit=frame.audit_summary(),
    )


ExperienceSuperNetwork.ensure_compatible = _context_v12_memory_ensure
ExperienceSuperNetwork._observe_context = _context_v12_observe_context
ExperienceSuperNetwork.advance_context_frame = _context_v12_advance_context_frame
ExperienceSuperNetwork._trace_memory_key = _context_v12_trace_memory_key
ExperienceSuperNetwork._cell_multi_scale_similarity = _context_v12_cell_similarity
ExperienceSuperNetwork.query = _context_v12_memory_query
ExperienceSuperNetwork.absorb_transition = _context_v12_absorb_transition
SelfGrowingPolicy.ensure_compatible = _context_v12_policy_ensure
SelfGrowingPolicy.set_decision_context = _context_v12_set_decision_context
SelfGrowingPolicy._query_memory = _context_v12_policy_query_memory
SelfGrowingPolicy.act = _context_v12_policy_act

# ---------------------------------------------------------------------------
# Expectancy v13: robust net-R value learning by context × action × lifecycle
# ---------------------------------------------------------------------------
# ``expectancy`` used to be primarily a global post-trial scalar.  V13 makes it
# a decision-time value signal.  Only finalised lifecycles update this ledger;
# only policy-origin evidence can bias M1 action selection; and every value is
# scoped by the canonical ContextFrame plus the exact action/lifecycle scope.

from .contextual_expectancy import (
    EXPECTANCY_SCHEMA_VERSION,
    ContextualExpectancyLedger,
    action_scope as _v13_action_scope,
    stable_identifier as _v13_stable_identifier,
)

_EXPECTANCY_V13_MEMORY_ENSURE = ExperienceSuperNetwork.ensure_compatible
_EXPECTANCY_V13_MEMORY_COPY = ExperienceSuperNetwork.copy
_EXPECTANCY_V13_MEMORY_SUMMARY = ExperienceSuperNetwork.summary
_EXPECTANCY_V13_POLICY_ENSURE = SelfGrowingPolicy.ensure_compatible
_EXPECTANCY_V13_POLICY_COPY = SelfGrowingPolicy.copy
_EXPECTANCY_V13_POLICY_SET_CONTEXT = SelfGrowingPolicy.set_decision_context
_EXPECTANCY_V13_POLICY_QUERY_MEMORY = SelfGrowingPolicy._query_memory
_EXPECTANCY_V13_POLICY_ACT = SelfGrowingPolicy.act
_EXPECTANCY_V13_POLICY_LIFECYCLE_CREDIT = SelfGrowingPolicy.apply_execution_lifecycle_credit


def _expectancy_v13_memory_ensure(self: ExperienceSuperNetwork) -> None:
    _EXPECTANCY_V13_MEMORY_ENSURE(self)
    ledger = getattr(self, "contextual_expectancy", None)
    if not isinstance(ledger, ContextualExpectancyLedger):
        ledger = ContextualExpectancyLedger()
        self.contextual_expectancy = ledger
    ledger.ensure_compatible()
    self.expectancy_schema_version = int(EXPECTANCY_SCHEMA_VERSION)


def _expectancy_v13_memory_copy(self: ExperienceSuperNetwork) -> ExperienceSuperNetwork:
    self.ensure_compatible()
    copied = _EXPECTANCY_V13_MEMORY_COPY(self)
    source = getattr(self, "contextual_expectancy", None)
    copied.contextual_expectancy = source.copy() if isinstance(source, ContextualExpectancyLedger) else ContextualExpectancyLedger()
    copied.expectancy_schema_version = int(EXPECTANCY_SCHEMA_VERSION)
    return copied


def _expectancy_v13_memory_summary(self: ExperienceSuperNetwork) -> dict[str, object]:
    result = dict(_EXPECTANCY_V13_MEMORY_SUMMARY(self) or {})
    self.ensure_compatible()
    result["contextual_expectancy"] = self.contextual_expectancy.report()
    result["expectancy_schema_version"] = int(EXPECTANCY_SCHEMA_VERSION)
    return result


def _expectancy_v13_policy_ensure(self: SelfGrowingPolicy, target_input_size: int | None = None) -> None:
    _EXPECTANCY_V13_POLICY_ENSURE(self, target_input_size=target_input_size)
    self.memory_supernet.ensure_compatible()
    self.expectancy_schema_version = int(EXPECTANCY_SCHEMA_VERSION)
    self.expectancy_learning = bool(getattr(self, "expectancy_learning", True))
    self.expectancy_min_samples = max(1, int(getattr(self, "expectancy_min_samples", 6) or 6))
    self.memory_supernet.contextual_expectancy.min_samples = int(self.expectancy_min_samples)
    if not hasattr(self, "_expectancy_positions_pipe"):
        self._expectancy_positions_pipe = ""
    if not hasattr(self, "_expectancy_credited_lifecycles") or self._expectancy_credited_lifecycles is None:
        self._expectancy_credited_lifecycles = set()


def _expectancy_v13_policy_copy(self: SelfGrowingPolicy) -> SelfGrowingPolicy:
    copied = _EXPECTANCY_V13_POLICY_COPY(self)
    copied.expectancy_schema_version = int(EXPECTANCY_SCHEMA_VERSION)
    copied.expectancy_learning = bool(getattr(self, "expectancy_learning", True))
    copied.expectancy_min_samples = max(1, int(getattr(self, "expectancy_min_samples", 6) or 6))
    copied._expectancy_positions_pipe = str(getattr(self, "_expectancy_positions_pipe", "") or "")
    copied._expectancy_credited_lifecycles = set(getattr(self, "_expectancy_credited_lifecycles", set()) or set())
    return copied


def _expectancy_v13_policy_set_context(
    self: SelfGrowingPolicy,
    *,
    account_context: dict[str, float] | None = None,
    positions_pipe: str | None = "",
    allowed_actions: object | None = None,
    allowed_action_mask: object | None = None,
    execution_context: dict[str, float] | None = None,
    goal: object | None = None,
    sequence_id: int | None = None,
    frame_id: str | None = None,
    timestamp: float | None = None,
) -> None:
    effective_allowed = allowed_action_mask if allowed_action_mask is not None else allowed_actions
    _EXPECTANCY_V13_POLICY_SET_CONTEXT(
        self,
        account_context=account_context,
        positions_pipe=positions_pipe,
        allowed_actions=effective_allowed,
        execution_context=execution_context,
        goal=goal,
        sequence_id=sequence_id,
        frame_id=frame_id,
        timestamp=timestamp,
    )
    self._decision_context["allowed_action_mask"] = effective_allowed
    self._expectancy_positions_pipe = str(positions_pipe or "")


def _expectancy_v13_current_scope(self: SelfGrowingPolicy, action_index: int, action_details: dict[str, object] | None = None) -> dict[str, object]:
    return _v13_action_scope(
        int(action_index),
        action_details=dict(action_details or {}),
        positions_pipe=str(getattr(self, "_expectancy_positions_pipe", "") or ""),
    )


def _expectancy_v13_policy_query_memory(self: SelfGrowingPolicy, state_embedding: np.ndarray) -> dict[str, object]:
    result = dict(_EXPECTANCY_V13_POLICY_QUERY_MEMORY(self, state_embedding) or {})
    self.ensure_compatible()
    if not bool(getattr(self, "expectancy_learning", True)):
        return result
    frame = getattr(self, "_active_context_frame", None)
    if not isinstance(frame, ContextFrame):
        return result
    ledger = self.memory_supernet.contextual_expectancy
    expectation = ledger.query(
        frame.canonical_memory_key,
        output_size=int(self.output_size),
        positions_pipe=str(getattr(self, "_expectancy_positions_pipe", "") or ""),
        allowed_action_mask=frame.allowed_action_mask,
    )
    old_bias = np.asarray(result.get("action_bias", np.zeros(self.output_size)), dtype=float).reshape(-1)
    if old_bias.size != int(self.output_size):
        old_bias = np.resize(old_bias, int(self.output_size)) if old_bias.size else np.zeros(self.output_size, dtype=float)
    exp_bias = np.asarray(expectation.get("expectancy_action_bias", np.zeros(self.output_size)), dtype=float).reshape(-1)
    if exp_bias.size != int(self.output_size):
        exp_bias = np.resize(exp_bias, int(self.output_size)) if exp_bias.size else np.zeros(self.output_size, dtype=float)
    # Existing replacement/success evidence remains valid.  Robust positive
    # expectancy adds a bounded value contribution; a negative LCB is exposed
    # to the critic/audit but never converted into blind global suppression.
    combined_bias = np.clip(old_bias + 0.85 * exp_bias, -1.0, 1.0)
    old_success = np.asarray(result.get("success_action_bias", np.zeros(self.output_size)), dtype=float).reshape(-1)
    if old_success.size != int(self.output_size):
        old_success = np.resize(old_success, int(self.output_size)) if old_success.size else np.zeros(self.output_size, dtype=float)
    # V8: legacy ``opportunity_cost`` is not market evidence. A live entry
    # opportunity exists only when the same scoped expectancy query has
    # verified support, confidence and a positive lower confidence bound.
    opportunity = float(expectation.get("expectancy_market_opportunity", 0.0) or 0.0)
    if not bool(expectation.get("expectancy_entry_evidence_verified", False)):
        opportunity = 0.0
    lcb = np.asarray(expectation.get("expectancy_action_lcb", np.zeros(self.output_size)), dtype=float).reshape(-1)
    confidence = np.asarray(expectation.get("expectancy_action_confidence", np.zeros(self.output_size)), dtype=float).reshape(-1)
    result.update(expectation)
    result.update({
        "action_bias": combined_bias,
        "success_action_bias": np.clip(old_success + 0.65 * exp_bias, -1.0, 1.0),
        "market_opportunity": float(np.clip(opportunity, 0.0, 1.0)),
        "expectancy_critic_bias": np.concatenate([lcb, confidence]),
        "expectancy_value_source": "policy_origin_net_r_lcb",
        "negative_action_suppression_mode": "action_local_only",
    })
    return result


def _expectancy_v13_trace_action_index(policy: SelfGrowingPolicy, trace: StepTrace) -> int:
    action_count = min(ACTION_COUNT, max(1, int(getattr(policy, "output_size", OUTPUT_COUNT))))
    try:
        return int(policy._trace_action_index(trace, action_count))
    except Exception:
        action = np.asarray(getattr(trace, "selected_action", np.zeros(action_count)), dtype=float).reshape(-1)
        return int(np.argmax(action[:action_count])) if action.size else 0


def _expectancy_v13_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    trace = _EXPECTANCY_V13_POLICY_ACT(self, observation, goal)
    self.ensure_compatible()
    frame = getattr(self, "_active_context_frame", None)
    if not isinstance(frame, ContextFrame) or not bool(getattr(self, "expectancy_learning", True)):
        return trace
    memory = self._query_memory(np.asarray(getattr(trace, "encoded_state", observation), dtype=float))
    action_index = _expectancy_v13_trace_action_index(self, trace)
    lcb = np.asarray(memory.get("expectancy_action_lcb", np.zeros(self.output_size)), dtype=float).reshape(-1)
    means = np.asarray(memory.get("expectancy_action_mean", np.zeros(self.output_size)), dtype=float).reshape(-1)
    samples = np.asarray(memory.get("expectancy_action_samples", np.zeros(self.output_size)), dtype=float).reshape(-1)
    confidence = np.asarray(memory.get("expectancy_action_confidence", np.zeros(self.output_size)), dtype=float).reshape(-1)
    selected_lcb = float(lcb[action_index]) if 0 <= action_index < lcb.size else 0.0
    selected_mean = float(means[action_index]) if 0 <= action_index < means.size else 0.0
    selected_samples = float(samples[action_index]) if 0 <= action_index < samples.size else 0.0
    selected_confidence = float(confidence[action_index]) if 0 <= action_index < confidence.size else 0.0
    scope = _expectancy_v13_current_scope(self, action_index, dict(getattr(trace, "action_details", {}) or {}))
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "expectancy_schema_version": int(EXPECTANCY_SCHEMA_VERSION),
        "expectancy_action_index": int(action_index),
        "expectancy_scope": dict(scope),
        "expectancy_predicted_mean_net_r": selected_mean,
        "expectancy_predicted_lcb": selected_lcb,
        "expectancy_predicted_samples": selected_samples,
        "expectancy_predicted_confidence": selected_confidence,
        "expectancy_regret": float(memory.get("expectancy_regret", 0.0) or 0.0),
        "expectancy_best_action": int(memory.get("expectancy_best_action", 0) or 0),
        "expectancy_best_lcb": float(memory.get("expectancy_best_lcb", 0.0) or 0.0),
        "expectancy_value_source": "policy_origin_net_r_lcb",
    })
    audit = dict(getattr(trace, "context_audit", {}) or {})
    audit.update({
        "expectancy_predicted_lcb": selected_lcb,
        "expectancy_predicted_samples": selected_samples,
        "expectancy_record_count": int(memory.get("expectancy_record_count", 0) or 0),
    })
    return replace(trace, action_details=details, context_audit=audit)


def _expectancy_v13_origin(trace: StepTrace) -> str:
    details = dict(getattr(trace, "execution_details", {}) or {})
    action = dict(getattr(trace, "action_details", {}) or {})
    return str(details.get("action_origin", action.get("action_origin", "policy")) or "policy").strip().lower() or "policy"


def _expectancy_v13_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
) -> int:
    # The underlying lifecycle-credit path consumes/populates its trace cache.
    # Capture immutable decision evidence first so expectancy attribution never
    # disappears merely because neuron credit performed its cleanup correctly.
    cache = getattr(self, "_execution_trace_cache", {})
    trace = cache.get(str(trace_id)) if isinstance(cache, dict) else None
    credited = _EXPECTANCY_V13_POLICY_LIFECYCLE_CREDIT(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
    )
    self.ensure_compatible()
    if not isinstance(trace, StepTrace):
        return credited
    origin = _expectancy_v13_origin(trace)
    action_index = _expectancy_v13_trace_action_index(self, trace)
    scope = _v13_action_scope(
        action_index,
        action_details=dict(getattr(trace, "action_details", {}) or {}),
        positions_pipe=str(getattr(trace, "positions_pipe", "") or ""),
    )
    context_key = np.asarray(getattr(trace, "canonical_context_key", np.zeros(0)), dtype=float).reshape(-1)
    if context_key.size != MEMORY_KEY_SIZE:
        context_key = self.memory_supernet._trace_memory_key(trace)
    identity = _v13_stable_identifier(
        "expectancy_lifecycle",
        str(trace_id),
        int(action_index),
        str(close_reason),
        None if duration_bars is None else round(float(duration_bars), 5),
        round(float(realized_r), 8),
        length=32,
    )
    credited_ids = getattr(self, "_expectancy_credited_lifecycles", set())
    if identity in credited_ids:
        return credited
    self.memory_supernet.contextual_expectancy.observe(
        context_key,
        scope,
        float(realized_r),
        origin=origin,
        trace_id=str(trace_id),
        frame_id=str(getattr(trace, "context_frame_id", "") or ""),
        lifecycle_tags=tuple(getattr(trace, "lifecycle_tags", ()) or ()),
    )
    credited_ids.add(identity)
    self._expectancy_credited_lifecycles = credited_ids
    report = self.memory_supernet.contextual_expectancy.report()
    expected_regret = float(dict(getattr(trace, "action_details", {}) or {}).get("expectancy_regret", 0.0) or 0.0)
    # Expected-regret is a structural-growth signal only after a real policy
    # lifecycle has confirmed that the selected action was poor in a repeated
    # context.  It never grows a module just because one single trade lost.
    if origin == "policy" and float(realized_r) < 0.0 and expected_regret >= 0.10:
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "context_expectancy_regret",
            "trace_id": str(trace_id),
            "scope": dict(scope),
            "realized_r": float(realized_r),
            "expected_regret": expected_regret,
            "context_frame_id": str(getattr(trace, "context_frame_id", "") or ""),
            "requires_repeated_context_evidence": True,
        })
    self.growth_events.append({
        "round": int(getattr(self, "revision", 0)),
        "type": "context_action_expectancy_credit",
        "trace_id": str(trace_id),
        "action_index": int(action_index),
        "scope": dict(scope),
        "origin": origin,
        "realized_net_r": float(realized_r),
        "close_reason": str(close_reason),
        "policy_expectancy_lcb": float(dict(report.get("policy_expectancy", {}) or {}).get("expectancy_lcb", 0.0) or 0.0),
        "policy_origin_ratio": float(report.get("policy_origin_ratio", 0.0) or 0.0),
        "expectancy_schema_version": int(EXPECTANCY_SCHEMA_VERSION),
    })
    return credited + 1


def _expectancy_v13_policy_report(self: SelfGrowingPolicy) -> dict[str, object]:
    self.ensure_compatible()
    return self.memory_supernet.contextual_expectancy.report()


def _expectancy_v13_expected_action_values(
    self: SelfGrowingPolicy,
    *,
    positions_pipe: str | None = None,
    allowed_action_mask: object | None = None,
) -> dict[str, object]:
    self.ensure_compatible()
    frame = getattr(self, "_active_context_frame", None)
    if not isinstance(frame, ContextFrame):
        return {"expectancy_schema_version": EXPECTANCY_SCHEMA_VERSION, "available": False}
    return self.memory_supernet.contextual_expectancy.query(
        frame.canonical_memory_key,
        output_size=self.output_size,
        positions_pipe=str(positions_pipe if positions_pipe is not None else getattr(self, "_expectancy_positions_pipe", "") or ""),
        allowed_action_mask=allowed_action_mask if allowed_action_mask is not None else frame.allowed_action_mask,
    )


ExperienceSuperNetwork.ensure_compatible = _expectancy_v13_memory_ensure
ExperienceSuperNetwork.copy = _expectancy_v13_memory_copy
ExperienceSuperNetwork.summary = _expectancy_v13_memory_summary
SelfGrowingPolicy.ensure_compatible = _expectancy_v13_policy_ensure
SelfGrowingPolicy.copy = _expectancy_v13_policy_copy
SelfGrowingPolicy.set_decision_context = _expectancy_v13_policy_set_context
SelfGrowingPolicy._query_memory = _expectancy_v13_policy_query_memory
SelfGrowingPolicy.act = _expectancy_v13_policy_act
SelfGrowingPolicy.apply_execution_lifecycle_credit = _expectancy_v13_lifecycle_credit
SelfGrowingPolicy.contextual_expectancy_report = _expectancy_v13_policy_report
SelfGrowingPolicy.expected_action_values = _expectancy_v13_expected_action_values


# ---------------------------------------------------------------------------
# Live Neural Observatory V14: read-only event stream for direct training view
# ---------------------------------------------------------------------------
# This layer is intentionally appended after Context V12 and Expectancy V13.
# It observes the final immutable trace / final lifecycle credit rather than
# changing action selection, graph plasticity, structural growth or MT5 I/O.

_LIVE_OBSERVATORY_POLICY_ENSURE = SelfGrowingPolicy.ensure_compatible
_LIVE_OBSERVATORY_POLICY_COPY = SelfGrowingPolicy.copy
_LIVE_OBSERVATORY_POLICY_ACT = SelfGrowingPolicy.act
_LIVE_OBSERVATORY_POLICY_OBSERVE = SelfGrowingPolicy.observe_transition
_LIVE_OBSERVATORY_POLICY_LIFECYCLE = SelfGrowingPolicy.apply_execution_lifecycle_credit


def _live_observatory_ensure(self: SelfGrowingPolicy, target_input_size: int | None = None) -> None:
    _LIVE_OBSERVATORY_POLICY_ENSURE(self, target_input_size=target_input_size)
    config = getattr(self, "live_observatory_config", None)
    if not isinstance(config, dict):
        config = {"enabled": False}
        self.live_observatory_config = config
    writer = getattr(self, "_live_observer", None)
    if writer is None and bool(config.get("enabled", False)):
        self._live_observer = LiveTelemetryWriter(
            path=str(config.get("event_path", "outputs/neural_live.jsonl") or "outputs/neural_live.jsonl"),
            enabled=True,
            decision_sample_every=max(1, int(config.get("decision_sample_every", 1) or 1)),
            max_event_bytes=max(1024 * 1024, int(config.get("max_event_bytes", 128 * 1024 * 1024) or 128 * 1024 * 1024)),
            max_active_nodes=max(0, int(config.get("max_active_nodes", 256) if config.get("max_active_nodes", None) is not None else 256)),
            max_active_edges=max(0, int(config.get("max_active_edges", 0) if config.get("max_active_edges", None) is not None else 0)),
            max_state_nodes=max(8, int(config.get("max_state_nodes", 512) or 512)),
            max_state_edges=max(8, int(config.get("max_state_edges", 1024) or 1024)),
            state_change_epsilon=max(0.0, float(config.get("state_change_epsilon", 1e-6) or 0.0)),
            decision_profile_path=str(config.get("decision_profile_path", "") or ""),
            structural_event_path=str(config.get("structural_event_path", "") or ""),
            max_responsible_neurons=max(1, int(config.get("max_responsible_neurons", 32) or 32)),
            max_responsible_edges=max(1, int(config.get("max_responsible_edges", 48) or 48)),
            run_id=str(config.get("run_id", "") or ""),
            source=str(config.get("source", "training") or "training"),
        )
    if not hasattr(self, "_live_observer_started"):
        self._live_observer_started = False


def _live_observatory_copy(self: SelfGrowingPolicy) -> SelfGrowingPolicy:
    copied = _LIVE_OBSERVATORY_POLICY_COPY(self)
    copied.live_observatory_config = dict(getattr(self, "live_observatory_config", {}) or {})
    writer = getattr(self, "_live_observer", None)
    copied._live_observer = writer.copy() if isinstance(writer, LiveTelemetryWriter) else None
    copied._live_observer_started = bool(getattr(self, "_live_observer_started", False))
    return copied


def _live_observatory_writer(self: SelfGrowingPolicy) -> LiveTelemetryWriter | None:
    writer = getattr(self, "_live_observer", None)
    return writer if isinstance(writer, LiveTelemetryWriter) and bool(writer.enabled) else None


def _live_observatory_enable(
    self: SelfGrowingPolicy,
    settings: object,
    *,
    run_id: str = "",
    source: str = "training",
    metadata: dict[str, object] | None = None,
) -> None:
    config = {
        "enabled": bool(getattr(settings, "enabled", True)),
        "event_path": str(getattr(settings, "event_path", "outputs/neural_live.jsonl") or "outputs/neural_live.jsonl"),
        "decision_sample_every": max(1, int(getattr(settings, "decision_sample_every", 1) or 1)),
        "max_event_bytes": max(1024 * 1024, int(getattr(settings, "max_event_bytes", 128 * 1024 * 1024) or 128 * 1024 * 1024)),
        "max_active_nodes": max(0, int(getattr(settings, "max_active_nodes", 256) if getattr(settings, "max_active_nodes", None) is not None else 256)),
        "max_active_edges": max(0, int(getattr(settings, "max_active_edges", 0) if getattr(settings, "max_active_edges", None) is not None else 0)),
        "max_state_nodes": max(8, int(getattr(settings, "max_state_nodes", 512) or 512)),
        "max_state_edges": max(8, int(getattr(settings, "max_state_edges", 1024) or 1024)),
        "state_change_epsilon": max(0.0, float(getattr(settings, "state_change_epsilon", 1e-6) or 0.0)),
        "decision_profile_path": str(getattr(settings, "decision_profile_path", "") or ""),
        "structural_event_path": str(getattr(settings, "structural_event_path", "") or ""),
        "max_responsible_neurons": max(1, int(getattr(settings, "max_responsible_neurons", 32) or 32)),
        "max_responsible_edges": max(1, int(getattr(settings, "max_responsible_edges", 48) or 48)),
        "run_id": str(run_id or f"run-{int(time.time())}"),
        "source": str(source),
    }
    self.live_observatory_config = config
    path = Path(config["event_path"])
    # Reset only once per Python process.  train-to-goal may invoke nested MT5
    # training rounds; wiping the stream between rounds would destroy the live
    # growth history the dashboard is meant to display.
    if bool(getattr(settings, "reset_on_process_start", False)):
        marker = str(path.resolve())
        global _LIVE_OBSERVATORY_RESET_PATHS
        try:
            seen = _LIVE_OBSERVATORY_RESET_PATHS
        except NameError:
            seen = set()
            _LIVE_OBSERVATORY_RESET_PATHS = seen
        if marker not in seen:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            seen.add(marker)
    writer = LiveTelemetryWriter.from_config(settings, run_id=config["run_id"], source=config["source"])
    self._live_observer = writer
    self._live_observer_started = True
    writer.start(self, metadata=dict(metadata or {}))


def _live_observatory_set_trial(self: SelfGrowingPolicy, *, policy_id: int, round_index: int, phase: str = "mt5") -> None:
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.trial(self, policy_id=int(policy_id), round_index=int(round_index), phase=str(phase))


def _live_observatory_trial_finished(
    self: SelfGrowingPolicy,
    *,
    policy_id: int,
    round_index: int,
    fitness: float,
    metrics: dict[str, object] | None = None,
) -> None:
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.trial_finished(
            self,
            policy_id=int(policy_id),
            round_index=int(round_index),
            fitness=float(fitness),
            metrics=metrics,
        )


def _live_observatory_stop(self: SelfGrowingPolicy, *, reason: str = "completed") -> None:
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.emit("run_finished", {
            "reason": str(reason),
            "revision": int(getattr(self, "revision", 0)),
            "growth_event_count": len(list(getattr(self, "growth_events", []) or [])),
        })
        writer.enabled = False


def _live_observatory_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    trace = _LIVE_OBSERVATORY_POLICY_ACT(self, observation, goal)
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.decision(self, trace)
    return trace


def _live_observatory_policy_observe(
    self: SelfGrowingPolicy,
    trace: StepTrace,
    next_obs: np.ndarray,
    reward: float,
    cost: float,
    done: bool,
    gamma: float = 0.95,
    record_experience: bool = True,
) -> dict[str, float]:
    signal = _LIVE_OBSERVATORY_POLICY_OBSERVE(
        self, trace, next_obs, reward, cost, done, gamma=gamma, record_experience=record_experience
    )
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.feedback(
            self,
            trace,
            next_obs=next_obs,
            reward=float(reward),
            cost=float(cost),
            done=bool(done),
            signal=signal if isinstance(signal, dict) else {},
        )
    return signal



def _live_observatory_execution_micro_growth(
    self: SelfGrowingPolicy,
    trace: StepTrace | None,
    realized_r: float,
    *,
    close_reason: str,
    duration_bars: float | None,
    min_holding_bars: int,
    lifecycle_baseline: float,
) -> list[dict[str, object]]:
    """Run candidate-only structural learning at the real MT5 settlement point.

    ``observe_transition`` already calls ``learn_in_trial`` for synthetic or
    bridge-confirmed transitions.  MT5 outcomes, however, arrive after the
    position closes through the execution ledger.  Without this hook, the graph
    receives plasticity credit but never recruits/fades a micro-neuron at the
    moment the evidence actually becomes trustworthy.

    The baseline policy is deliberately protected: only an isolated candidate
    carrying an active graph trial ID may create, retain, fade, or prune a
    specialist.  A negative realised outcome becomes replacement pressure
    through ``cost`` and signed TD surprise; it does not globally suppress the
    old action path.
    """
    if not isinstance(trace, StepTrace):
        return []
    runtime = getattr(self, "graph_supernetwork", None)
    trial_id = str(getattr(self, "graph_trial_id", "") or "")
    if (
        runtime is None
        or not bool(getattr(self, "graph_trial_mode", False))
        or not trial_id
    ):
        return []

    outcome = float(np.clip(float(realized_r), -3.0, 3.0))
    advantage = float(outcome - float(lifecycle_baseline))
    close_key = str(close_reason or "").upper()
    short_hold = duration_bars is not None and float(duration_bars) < max(1, int(min_holding_bars))
    stopped_out = close_key in {"STOP_OUT", "SL", "STOP_LOSS"}
    if short_hold:
        advantage -= 0.25
    if stopped_out:
        advantage -= 0.25

    # There is no final price-state vector after a live MT5 lifecycle, so the
    # realised surprise is the most honest prediction-error proxy.  Preserve a
    # stronger model-side prediction error if the decision already carried one.
    trace_prediction_error = abs(float(getattr(trace, "prediction_error", 0.0) or 0.0))
    realised_surprise = abs(float(advantage))
    learning_signal = {
        "reward": max(0.0, outcome),
        "cost": max(0.0, -outcome) + (0.25 if short_hold else 0.0) + (0.25 if stopped_out else 0.0),
        "td_delta": float(advantage),
        "prediction_error": max(trace_prediction_error, realised_surprise),
    }
    # V21: the V19 causal ledger settles before this hook.  Carry only
    # calibrated evidence into module survival: a one-off SL/TP may create an
    # exploratory circuit, but it cannot consolidate a durable specialist.
    # This preserves the distinction between attribution and verified cause.
    causal_report = dict(getattr(self, "_causal_reports", {}) or {}).get(str(getattr(trace, "trace_id", "")))
    if causal_report is not None:
        hypotheses = list(getattr(causal_report, "hypotheses", ()) or ())
        top = hypotheses[0] if hypotheses else None
        confidence = float(np.clip(float(getattr(top, "confidence", 0.0) or 0.0), 0.0, 1.0)) if top is not None else 0.0
        coverage = dict(getattr(causal_report, "coverage", {}) or {})
        validated = bool(coverage.get("counterfactual_replay_validated", False))
        if top is not None and str(getattr(top, "status", "")) in {"replay_validated", "validated"}:
            validated = True
        cause_id = str(getattr(top, "cause_id", "") or "") if top is not None else ""
        learning_signal.update({
            "causal_confidence": confidence,
            "causal_validated": validated,
            "causal_pattern_key": cause_id or str(close_reason or "lifecycle"),
        })
    account_context = dict(getattr(trace, "account_context", {}) or {})
    emitted: list[dict[str, object]] = []
    for _module, m1 in self._graph_m1_modules():
        graph = getattr(m1, "graph", None)
        if graph is None:
            continue
        try:
            rows = runtime.learn_in_trial(
                graph,
                trial_id=trial_id,
                trace=trace,
                learning_signal=learning_signal,
                account_context=account_context,
            )
        except Exception as exc:
            self.growth_events.append({
                "round": int(getattr(self, "revision", 0)),
                "type": "graph_lifecycle_micro_growth_error",
                "trace_id": str(getattr(trace, "trace_id", "")),
                "trial_id": trial_id,
                "close_reason": close_key,
                "error": str(exc),
            })
            continue
        for row in rows:
            event = dict(row)
            event.update({
                "round": int(getattr(self, "revision", 0)),
                "type": str(event.get("kind", "graph_trial_micro_growth")),
                "trace_id": str(getattr(trace, "trace_id", "")),
                "source": "mt5_execution_lifecycle",
                "realized_r": outcome,
                "advantage": advantage,
                "close_reason": close_key,
            })
            self.growth_events.append(event)
            emitted.append(event)
    # Emit a non-structural audit event as well.  It proves that the live MT5
    # lifecycle reached the growth controller even when a bounded trial budget
    # or learning threshold correctly decides not to add a neuron.
    self.growth_events.append({
        "round": int(getattr(self, "revision", 0)),
        "type": "graph_lifecycle_micro_growth_evaluated",
        "trace_id": str(getattr(trace, "trace_id", "")),
        "trial_id": trial_id,
        "source": "mt5_execution_lifecycle",
        "realized_r": outcome,
        "advantage": advantage,
        "close_reason": close_key,
        "duration_bars": None if duration_bars is None else float(duration_bars),
        "learning_signal": dict(learning_signal),
        "growth_event_count": len(emitted),
    })
    return emitted


def _live_observatory_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
) -> int:
    cache = getattr(self, "_execution_trace_cache", None)
    trace = cache.get(str(trace_id)) if isinstance(cache, dict) else None
    lifecycle_baseline = float(getattr(self, "_lifecycle_outcome_ema", 0.0) or 0.0)
    credited = _LIVE_OBSERVATORY_POLICY_LIFECYCLE(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
    )
    _live_observatory_execution_micro_growth(
        self,
        trace if isinstance(trace, StepTrace) else None,
        float(realized_r),
        close_reason=str(close_reason),
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
        lifecycle_baseline=lifecycle_baseline,
    )
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.lifecycle(
            self,
            trace_id=str(trace_id),
            realized_r=float(realized_r),
            close_reason=str(close_reason),
            duration_bars=duration_bars,
            credited=int(credited),
        )
    return credited


SelfGrowingPolicy.ensure_compatible = _live_observatory_ensure
SelfGrowingPolicy.copy = _live_observatory_copy
SelfGrowingPolicy.enable_live_observatory = _live_observatory_enable
SelfGrowingPolicy.set_live_observatory_trial = _live_observatory_set_trial
SelfGrowingPolicy.stop_live_observatory = _live_observatory_stop
SelfGrowingPolicy.live_observatory_trial_finished = _live_observatory_trial_finished
SelfGrowingPolicy.act = _live_observatory_policy_act
SelfGrowingPolicy.observe_transition = _live_observatory_policy_observe
SelfGrowingPolicy.apply_execution_lifecycle_credit = _live_observatory_lifecycle_credit

# ---------------------------------------------------------------------------
# Reasoning V18: namespaced multi-graph evidence, lifecycle credit, and
# bounded pre-execution re-ranking.
# ---------------------------------------------------------------------------
# Earlier versions instantiated a separate reasoning graph but exposed only M1
# trace data.  Worse, MT5 settlement credited M1 after a final close while the
# reasoning graph kept its seed utility.  This layer keeps an immutable trace
# for both graphs, credits exactly the controller paths used at order entry,
# and lets action-head reasoning re-rank valid M1 candidates before the output
# leaves the policy.

_REASONING_V18_POLICY_ACT = _LIVE_OBSERVATORY_POLICY_ACT
_REASONING_V18_POLICY_LIFECYCLE = _LIVE_OBSERVATORY_POLICY_LIFECYCLE


def _reasoning_v18_action_name(index: int) -> str:
    names = (
        "HOLD", "MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT",
        "BUY_STOP", "SELL_STOP", "CANCEL_ORDERS", "CLOSE_WORST",
        "CLOSE_BEST", "CLOSE_ALL", "MANAGE_POSITION",
    )
    return names[int(np.clip(int(index), 0, len(names) - 1))]


def _reasoning_v18_m1_trace_record(
    module_name: str,
    graph: object,
    graph_trace: object,
    raw_obs: np.ndarray,
) -> dict[str, object]:
    """Build immutable M1 telemetry evidence without relying on UI IDs."""
    neurons = dict(getattr(graph, "neurons", {}) or {})
    edges = dict(getattr(graph, "edges", {}) or {})
    active_nodes = tuple(str(item) for item in getattr(graph_trace, "active_neuron_ids", ()) or ())
    active_edges = tuple(str(item) for item in getattr(graph_trace, "active_edge_ids", ()) or ())
    node_values = {
        node_id: {
            "activation": float(getattr(neurons.get(node_id), "activation", 0.0)),
            "state": float(getattr(neurons.get(node_id), "state", 0.0)),
        }
        for node_id in neurons
    }
    input_ids = list(getattr(getattr(graph, "regions", {}).get("input"), "neuron_ids", []) or [])
    observed = np.asarray(raw_obs, dtype=float).reshape(-1)
    raw_inputs = {
        str(node_id): float(observed[index]) if index < observed.size and np.isfinite(observed[index]) else 0.0
        for index, node_id in enumerate(input_ids)
    }
    return {
        "graph_id": f"m1:{module_name}",
        "graph_label": str(module_name),
        "active_neuron_ids": active_nodes,
        "active_edge_ids": active_edges,
        "edge_gates": {
            str(key): float(value)
            for key, value in dict(getattr(graph_trace, "edge_gates", {}) or {}).items()
        },
        "edge_contributions": {
            str(key): float(value)
            for key, value in dict(getattr(graph_trace, "edge_contributions", {}) or {}).items()
        },
        "region_activity": {
            str(key): float(value)
            for key, value in dict(getattr(graph_trace, "region_activity", {}) or {}).items()
        },
        "node_values": node_values,
        "raw_inputs": raw_inputs,
        "controllers": {},
        "active_node_count": int(len(active_nodes)),
        "active_edge_count": int(getattr(graph_trace, "active_edge_count", len(active_edges)) or len(active_edges)),
        "traced_edge_count": int(getattr(graph_trace, "traced_edge_count", len(active_edges)) or len(active_edges)),
    }


def _reasoning_v18_graph_traces(self: SelfGrowingPolicy, trace: StepTrace) -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for index, module in enumerate(list(getattr(self, "modules", []) or [])):
        m1 = getattr(module, "m1_component", None)
        graph = getattr(m1, "graph", None)
        graph_trace = getattr(m1, "last_trace", None)
        if graph is None or graph_trace is None:
            continue
        name = str(getattr(module, "name", f"module_{index}") or f"module_{index}")
        records[f"m1:{name}"] = _reasoning_v18_m1_trace_record(name, graph, graph_trace, trace.raw_obs)
    reasoning = getattr(self, "reasoning_graph", None)
    if reasoning is not None and hasattr(reasoning, "telemetry_trace"):
        try:
            record = dict(reasoning.telemetry_trace() or {})
            records["reasoning_graph"] = record
        except Exception as exc:
            records["reasoning_graph"] = {
                "graph_id": "reasoning_graph",
                "graph_label": "reasoning_graph",
                "active_neuron_ids": (),
                "active_edge_ids": (),
                "controllers": {},
                "telemetry_error": str(exc),
            }
    return records


def _reasoning_v18_rerank(
    self: SelfGrowingPolicy,
    trace: StepTrace,
) -> StepTrace:
    """Use reasoning action-head output to re-rank safe M1 candidates.

    It is intentionally bounded: the M1 candidate generator remains the
    source of hypotheses, while reasoning can select among those hypotheses
    before execution.  This makes the reasoning module causally relevant to
    BUY/SELL/HOLD without granting it authority to manufacture an illegal
    action outside the M1/MT5 contract.
    """
    reasoning = getattr(self, "reasoning_graph", None)
    candidates = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
    scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
    if reasoning is None or candidates.ndim != 2 or candidates.shape[0] == 0 or scores.size == 0:
        return trace
    count = min(candidates.shape[0], scores.size)
    if count <= 0:
        return trace
    candidates = candidates[:count]
    scores = scores[:count].copy()
    selected_before = int(np.clip(int(getattr(trace, "selected_idx", 0)), 0, count - 1))
    action_count = min(ACTION_COUNT, candidates.shape[1], int(getattr(self, "output_size", OUTPUT_COUNT)))
    if action_count <= 0:
        return trace

    # The full controller context is projected inside ReasoningGraphSystem;
    # no tail is silently discarded when it exceeds the graph input width.
    selection_context = np.concatenate([
        np.asarray(trace.raw_obs, dtype=float).reshape(-1),
        np.asarray(trace.encoded_state, dtype=float).reshape(-1),
        np.asarray(trace.memory_after, dtype=float).reshape(-1),
        np.asarray(trace.candidate_logits, dtype=float).reshape(-1),
        scores.reshape(-1),
        np.asarray(trace.selected_action, dtype=float).reshape(-1),
        np.asarray(trace.goal_context, dtype=float).reshape(-1),
        np.asarray(trace.learning_drive_context, dtype=float).reshape(-1),
        np.asarray(list(getattr(trace, "gate_values", ()) or ()), dtype=float).reshape(-1),
    ])
    try:
        reasoning.enrich_context("action_head", selection_context, target_size=max(1, selection_context.size))
        raw_delta = np.asarray(reasoning.controller_output("action_head", size=count), dtype=float).reshape(-1)
    except Exception as exc:
        details = dict(getattr(trace, "action_details", {}) or {})
        details["reasoning_rerank_error"] = str(exc)
        return replace(trace, action_details=details)
    if raw_delta.size != count:
        raw_delta = np.resize(raw_delta, count) if raw_delta.size else np.zeros(count, dtype=float)
    if not np.all(np.isfinite(raw_delta)):
        raw_delta = np.nan_to_num(raw_delta, nan=0.0, posinf=0.0, neginf=0.0)
    # Centre it so the layer can express relative preference rather than merely
    # shift every score.  Gain stays conservative until lifecycle evidence has
    # had a chance to update utility/gates.
    delta = np.clip(raw_delta - float(np.mean(raw_delta)), -1.0, 1.0)
    controller_credit = dict(getattr(reasoning, "last_controller_credit", {}) or {}).get("action_head", {})
    maturity = min(1.0, abs(float(dict(controller_credit or {}).get("credit", 0.0) or 0.0)) / 1.5)
    gain = float(np.clip(0.035 + 0.55 * float(getattr(reasoning, "influence", 0.08)) * (0.35 + 0.65 * maturity), 0.035, 0.18))
    adjusted_scores = scores + gain * delta

    allowed = np.asarray(getattr(trace, "allowed_action_mask", ()) or (), dtype=bool).reshape(-1)
    if allowed.size < action_count:
        allowed = np.pad(allowed, (0, action_count - allowed.size), constant_values=True)
    if allowed.size and not np.any(allowed[:action_count]):
        allowed[0] = True
    legality = np.ones(count, dtype=bool)
    if allowed.size:
        for row in range(count):
            candidate_index = int(np.argmax(candidates[row, :action_count]))
            legality[row] = bool(allowed[candidate_index])
        adjusted_scores[~legality] = -np.inf
    if not np.any(np.isfinite(adjusted_scores)):
        adjusted_scores = scores.copy()
        adjusted_scores[0] = max(adjusted_scores[0], float(np.max(scores)))
    selected_after = int(np.nanargmax(adjusted_scores))

    # Re-selection replaces action logits only; the original risk/position
    # parameter heads remain in force.  This protects SL/TP/scale controls
    # while allowing the selected discrete command to change.
    selected = np.asarray(trace.selected_action, dtype=float).reshape(-1).copy()
    if selected.size < candidates.shape[1]:
        selected = np.pad(selected, (0, candidates.shape[1] - selected.size))
    selected[:action_count] = candidates[selected_after, :action_count]
    selected = _context_v12_mask_policy_output(selected, allowed if allowed.size else np.ones(action_count, dtype=bool))
    logit = adjusted_scores - np.max(adjusted_scores[np.isfinite(adjusted_scores)])
    exp = np.exp(np.where(np.isfinite(logit), np.clip(logit, -40.0, 0.0), -40.0))
    probability = exp / max(float(np.sum(exp)), 1e-12)
    final_action_index = int(np.argmax(selected[:action_count])) if selected.size else 0
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "action_name": _reasoning_v18_action_name(final_action_index),
        "reasoning_schema_version": 18,
        "reasoning_selection_stage": "pre_execution_candidate_rerank",
        "reasoning_selected_candidate_before": selected_before,
        "reasoning_selected_candidate_after": selected_after,
        "reasoning_changed_candidate": bool(selected_before != selected_after),
        "reasoning_action_score_gain": gain,
        "reasoning_action_score_delta": [float(item) for item in delta.tolist()],
        "reasoning_candidate_scores_before": [float(item) for item in scores.tolist()],
        "reasoning_candidate_scores_after": [float(item) if np.isfinite(item) else -1e9 for item in adjusted_scores.tolist()],
        "reasoning_legal_candidate_count": int(np.sum(legality)),
        "reasoning_final_action_index": final_action_index,
    })
    return replace(
        trace,
        selected_idx=selected_after,
        selected_action=selected,
        candidate_scores=adjusted_scores,
        selected_action_log_prob=float(np.log(max(float(probability[selected_after]), 1e-12))),
        action_details=details,
    )


def _reasoning_v18_prepare_selector_bias(
    self: SelfGrowingPolicy,
    observation: np.ndarray,
    goal: np.ndarray | None,
) -> dict[str, dict[str, object]]:
    """Prepare a bounded reasoning score bias before M1 selection.

    The bias is stored only for the duration of one call to ``act``.  M1GraphModule
    reads it inside ``forward`` immediately before legality masking and selector
    choice; the final trace can therefore prove whether reasoning affected the
    candidate score path rather than only the already-selected output.
    """
    reasoning = getattr(self, "reasoning_graph", None)
    if reasoning is None:
        return {}
    obs = np.asarray(observation, dtype=float).reshape(-1)
    account = dict(getattr(self, "_account_context", {}) or {})
    decision = dict(getattr(self, "_decision_context", {}) or {})
    execution = dict(decision.get("execution_context", {}) or {})
    starvation = dict(getattr(self, "_experience_starvation", {}) or {})
    # Sort keys so semantically identical dictionaries create the same context
    # order across processes/replays.
    scalar_parts = [
        obs,
        np.asarray(goal if goal is not None else [], dtype=float).reshape(-1),
        np.asarray([float(account[key]) for key in sorted(account) if isinstance(account.get(key), (int, float, np.number))], dtype=float),
        np.asarray([float(execution[key]) for key in sorted(execution) if isinstance(execution.get(key), (int, float, np.number))], dtype=float),
        np.asarray([float(starvation[key]) for key in sorted(starvation) if isinstance(starvation.get(key), (int, float, np.number))], dtype=float),
    ]
    selector_context = np.concatenate([part for part in scalar_parts if part.size]) if any(part.size for part in scalar_parts) else np.zeros(1, dtype=float)
    try:
        reasoning.enrich_context("utility", selector_context, target_size=max(1, selector_context.size))
    except Exception as exc:
        return {"_error": {"error": str(exc)}}
    result: dict[str, dict[str, object]] = {}
    for index, module in enumerate(list(getattr(self, "modules", []) or [])):
        m1 = getattr(module, "m1_component", None)
        if not isinstance(m1, M1GraphModule):
            continue
        count = max(1, int(getattr(m1, "K", 1) or 1))
        bias = np.asarray(reasoning.controller_output("utility", size=count), dtype=float).reshape(-1)
        if bias.size != count:
            bias = np.resize(bias, count) if bias.size else np.zeros(count, dtype=float)
        # A non-zero seed gain makes the path measurable but remains below the
        # graph selector's own score scale. Lifecycle utility raises influence
        # only through learned gates/weights, not a sudden global multiplier.
        utility_credit = dict(getattr(reasoning, "last_controller_credit", {}) or {}).get("utility", {})
        maturity = min(1.0, abs(float(dict(utility_credit or {}).get("credit", 0.0) or 0.0)) / 1.5)
        gain = float(np.clip(0.020 + 0.45 * float(getattr(reasoning, "influence", 0.08)) * (0.25 + 0.75 * maturity), 0.020, 0.14))
        m1._runtime_reasoning_candidate_bias = bias.copy()
        m1._runtime_reasoning_candidate_gain = gain
        name = str(getattr(module, "name", f"module_{index}") or f"module_{index}")
        result[f"m1:{name}"] = {
            "gain": gain,
            "raw_bias": [float(value) for value in bias.tolist()],
            "controller": "utility",
            "selection_stage": "inside_m1_before_legality_and_selector",
        }
    return result


def _reasoning_v18_clear_selector_bias(self: SelfGrowingPolicy) -> None:
    for module in list(getattr(self, "modules", []) or []):
        m1 = getattr(module, "m1_component", None)
        if not isinstance(m1, M1GraphModule):
            continue
        for attr in ("_runtime_reasoning_candidate_bias", "_runtime_reasoning_candidate_gain"):
            try:
                delattr(m1, attr)
            except AttributeError:
                pass


def _reasoning_v18_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    self.ensure_compatible()
    reasoning = getattr(self, "reasoning_graph", None)
    if reasoning is not None and hasattr(reasoning, "begin_decision"):
        reasoning.begin_decision()
    selector_bias = _reasoning_v18_prepare_selector_bias(self, np.asarray(observation, dtype=float), goal)
    # Call the V13 final policy path directly; V14 only writes telemetry and
    # would otherwise publish an incomplete, pre-rerank decision event.
    try:
        trace = _REASONING_V18_POLICY_ACT(self, observation, goal)
    finally:
        _reasoning_v18_clear_selector_bias(self)
    trace = _reasoning_v18_rerank(self, trace)
    traces = _reasoning_v18_graph_traces(self, trace)
    # Keep legacy single-M1 fields for old replay / learning paths while making
    # per-graph records authoritative for live UI and lifecycle attribution.
    selector_l1: dict[str, float] = {}
    for index, module in enumerate(list(getattr(self, "modules", []) or [])):
        m1 = getattr(module, "m1_component", None)
        if not isinstance(m1, M1GraphModule):
            continue
        name = str(getattr(module, "name", f"module_{index}") or f"module_{index}")
        selector_l1[f"m1:{name}"] = float(np.sum(np.abs(np.asarray(getattr(m1, "last_reasoning_score_bias", np.zeros(0)), dtype=float))))
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "graph_trace_schema_version": 18,
        "graph_trace_ids": sorted(traces),
        "reasoning_active_node_count": int(dict(traces.get("reasoning_graph", {}) or {}).get("active_node_count", 0) or 0),
        "reasoning_active_edge_count": int(dict(traces.get("reasoning_graph", {}) or {}).get("active_edge_count", 0) or 0),
        "reasoning_selector_bias": selector_bias,
        "reasoning_selector_stage": "inside_m1_before_legality_and_selector",
        "reasoning_selector_score_bias_l1": selector_l1,
    })
    trace = replace(trace, graph_traces=traces, action_details=details)
    # Context-owner metadata được facade gắn trước policy.act. Phải nhập vào
    # trace trước writer.decision để hồ sơ log không mất context/chuyên gia.
    owner_metadata = getattr(self, "_context_expert_owner_metadata", None)
    if isinstance(owner_metadata, Mapping) and owner_metadata:
        audit = dict(getattr(trace, "context_audit", {}) or {})
        audit.update({str(key): value for key, value in dict(owner_metadata).items()})
        trace = replace(trace, context_audit=audit)
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.decision(self, trace)
    return trace


def _reasoning_v18_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
) -> int:
    cache = getattr(self, "_execution_trace_cache", None)
    trace = cache.get(str(trace_id)) if isinstance(cache, dict) else None
    lifecycle_baseline = float(getattr(self, "_lifecycle_outcome_ema", 0.0) or 0.0)
    # Same V13/M1 lifecycle path as before, minus the V14 write so V18 can emit
    # one complete event only after both M1 and reasoning have received credit.
    credited = _REASONING_V18_POLICY_LIFECYCLE(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
    )
    _live_observatory_execution_micro_growth(
        self,
        trace if isinstance(trace, StepTrace) else None,
        float(realized_r),
        close_reason=str(close_reason),
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
        lifecycle_baseline=lifecycle_baseline,
    )
    report: dict[str, object] = {}
    if isinstance(trace, StepTrace) and getattr(self, "reasoning_graph", None) is not None:
        try:
            advantage = float(np.clip(float(realized_r) - lifecycle_baseline, -3.0, 3.0))
            record = dict(getattr(trace, "graph_traces", {}) or {}).get("reasoning_graph", {})
            controller_traces = dict(record.get("controllers", {}) or {}) if isinstance(record, dict) else {}
            report = self.reasoning_graph.observe_lifecycle(
                realized_r=float(realized_r),
                advantage=advantage,
                close_reason=str(close_reason),
                strength=1.0,
                controller_traces=controller_traces,
            )
            self.growth_events.append({
                "round": int(getattr(self, "revision", 0)),
                "type": "reasoning_final_lifecycle_credit",
                "trace_id": str(trace_id),
                "source": "mt5_execution_lifecycle",
                "realized_r": float(realized_r),
                "advantage": advantage,
                "close_reason": str(close_reason),
                "duration_bars": None if duration_bars is None else float(duration_bars),
                "updated_nodes": int(report.get("updated_nodes", 0) or 0),
                "updated_edges": int(report.get("updated_edges", 0) or 0),
                "controllers": dict(report.get("controllers", {}) or {}),
                "decision_trace_source": str(report.get("decision_trace_source", "")),
                "replacement_preserves_action_head": bool(report.get("replacement_preserves_action_head", False)),
            })
            credited += 1
        except Exception as exc:
            self.growth_events.append({
                "round": int(getattr(self, "revision", 0)),
                "type": "reasoning_final_lifecycle_credit_error",
                "trace_id": str(trace_id),
                "source": "mt5_execution_lifecycle",
                "error": str(exc),
            })
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.lifecycle(
            self,
            trace_id=str(trace_id),
            realized_r=float(realized_r),
            close_reason=str(close_reason),
            duration_bars=duration_bars,
            credited=int(credited),
        )
    return int(credited)


SelfGrowingPolicy.act = _reasoning_v18_policy_act
SelfGrowingPolicy.apply_execution_lifecycle_credit = _reasoning_v18_lifecycle_credit


# ---------------------------------------------------------------------------
# Causal lifecycle ledger V19
# ---------------------------------------------------------------------------
# V19 deliberately composes over the V18 live/reasoning patch instead of
# replacing its safety gates. It freezes evidence at entry, settles only on a
# final MT5 lifecycle, and records causal explanations as hypotheses rather
# than treating a single SL/TP as proof of a rule.

_CAUSAL_V19_POLICY_ENSURE = SelfGrowingPolicy.ensure_compatible
_CAUSAL_V19_POLICY_COPY = SelfGrowingPolicy.copy
_CAUSAL_V19_POLICY_ACT = SelfGrowingPolicy.act
_CAUSAL_V19_POLICY_LIFECYCLE = SelfGrowingPolicy.apply_execution_lifecycle_credit
_CAUSAL_V19_POLICY_QUERY_MEMORY = SelfGrowingPolicy._query_memory
_REASONING_V19_OBSERVE_LIFECYCLE = ReasoningGraphSystem.observe_lifecycle


def _causal_v19_reasoning_observe_lifecycle(self: ReasoningGraphSystem, *args, **kwargs):
    """Inject a settled causal plan into V18's immutable-trace update safely."""
    if kwargs.get("controller_rewards") is None:
        pending = getattr(self, "_causal_pending_controller_rewards", None)
        if isinstance(pending, dict) and pending:
            kwargs["controller_rewards"] = dict(pending)
    return _REASONING_V19_OBSERVE_LIFECYCLE(self, *args, **kwargs)


ReasoningGraphSystem.observe_lifecycle = _causal_v19_reasoning_observe_lifecycle


def _causal_v19_ensure(self: SelfGrowingPolicy, target_input_size: int | None = None) -> None:
    _CAUSAL_V19_POLICY_ENSURE(self, target_input_size=target_input_size)
    ledger = getattr(self, "causal_ledger", None)
    if not isinstance(ledger, CausalLedger):
        self.causal_ledger = CausalLedger()
    if not isinstance(getattr(self, "_causal_reports", None), dict):
        self._causal_reports = {}
    if not isinstance(getattr(self, "_causal_trace_archive", None), dict):
        self._causal_trace_archive = {}
    if not isinstance(getattr(self, "_causal_replay_credit_signatures", None), dict):
        self._causal_replay_credit_signatures = {}
    # Append-only replay exchange. The EA/offline worker may consume the
    # request journal and write verified results later; absence of results
    # never turns a hypothesis into a fact.
    journal = getattr(self, "causal_replay_journal", None)
    if not isinstance(journal, CounterfactualReplayJournal):
        request_path = getattr(self, "causal_replay_requests_path", "outputs/causal_replay_requests.jsonl")
        result_path = getattr(self, "causal_replay_results_path", "outputs/causal_replay_results.jsonl")
        self.causal_replay_journal = CounterfactualReplayJournal(request_path, result_path)
    if not isinstance(getattr(self, "_causal_memory_queries_current", None), list):
        self._causal_memory_queries_current = []
    if not isinstance(getattr(self, "_causal_last_memory_query", None), dict):
        self._causal_last_memory_query = {}


def _causal_v19_copy(self: SelfGrowingPolicy) -> SelfGrowingPolicy:
    copied = _CAUSAL_V19_POLICY_COPY(self)
    ledger = getattr(self, "causal_ledger", None)
    copied.causal_ledger = ledger.copy() if isinstance(ledger, CausalLedger) else CausalLedger()
    copied._causal_reports = copylib.deepcopy(dict(getattr(self, "_causal_reports", {}) or {}))
    copied._causal_last_memory_query = copylib.deepcopy(dict(getattr(self, "_causal_last_memory_query", {}) or {}))
    copied._causal_memory_queries_current = []
    copied._causal_trace_archive = copylib.deepcopy(dict(getattr(self, "_causal_trace_archive", {}) or {}))
    copied._causal_replay_credit_signatures = dict(getattr(self, "_causal_replay_credit_signatures", {}) or {})
    copied.causal_replay_requests_path = str(getattr(self, "causal_replay_requests_path", "outputs/causal_replay_requests.jsonl"))
    copied.causal_replay_results_path = str(getattr(self, "causal_replay_results_path", "outputs/causal_replay_results.jsonl"))
    copied.causal_replay_journal = CounterfactualReplayJournal(
        copied.causal_replay_requests_path,
        copied.causal_replay_results_path,
    )
    return copied


def _causal_v19_query_memory(self: SelfGrowingPolicy, state_embedding: np.ndarray) -> dict[str, object]:
    result = _CAUSAL_V19_POLICY_QUERY_MEMORY(self, state_embedding)
    if isinstance(result, dict):
        frozen = copylib.deepcopy(result)
        if not isinstance(getattr(self, "_causal_memory_queries_current", None), list):
            self._causal_memory_queries_current = []
        self._causal_memory_queries_current.append(frozen)
        # The query immediately preceding the final action/parameter heads is
        # the best available causal memory evidence in the current architecture.
        self._causal_last_memory_query = frozen
    return result


def _causal_v19_emit_snapshot(self: SelfGrowingPolicy, trace: StepTrace) -> None:
    writer = _live_observatory_writer(self)
    ledger = getattr(self, "causal_ledger", None)
    if writer is None or not isinstance(ledger, CausalLedger):
        return
    snapshot = ledger.snapshot_for(str(getattr(trace, "trace_id", "")))
    if snapshot is None:
        return
    writer.emit("causal_snapshot", {
        "trace_id": str(trace.trace_id),
        "frame_id": str(getattr(trace, "context_frame_id", "") or ""),
        "snapshot": snapshot.to_dict(),
        "ledger": ledger.summary(),
    })


def _causal_v19_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    self.ensure_compatible()
    _causal_v19_process_replay_results(self)
    self._causal_memory_queries_current = []
    # V18 writes its decision event before this wrapper can freeze the
    # snapshot. Temporarily mute that one write, then publish an identical
    # decision only after the causal evidence exists. Telemetry stays read-only.
    writer = _live_observatory_writer(self)
    writer_was_enabled = bool(writer.enabled) if writer is not None else False
    if writer is not None and writer_was_enabled:
        writer.enabled = False
    try:
        trace = _CAUSAL_V19_POLICY_ACT(self, observation, goal)
    finally:
        if writer is not None:
            writer.enabled = writer_was_enabled
    ledger = getattr(self, "causal_ledger", None)
    # V26 continuous mode cannot freeze a causal snapshot until the bridge has
    # determined that an actual MT5 command exists. Otherwise an EA wire no-op
    # (action=0) is archived as a neural HOLD decision before action masking.
    if isinstance(ledger, CausalLedger) and not bool(getattr(self, "_continuous_session_active", False)):
        ledger.register_decision(self, trace)
        archive = getattr(self, "_causal_trace_archive", None)
        if isinstance(archive, dict):
            archive[str(trace.trace_id)] = copylib.deepcopy(trace)
            limit = int(getattr(ledger, "max_decisions", 4096) or 4096)
            while len(archive) > limit:
                archive.pop(next(iter(archive)), None)
    if writer is not None and writer_was_enabled:
        writer.decision(self, trace)
        _causal_v19_emit_snapshot(self, trace)
    return trace


def _causal_v19_plan_signature(plan: CreditPlan) -> str:
    """Stable signature so a verified replay can add incremental credit once."""
    payload = {
        "reasoning": {str(k): round(float(v), 8) for k, v in dict(plan.reasoning_controller_credit or {}).items()},
        "memory": {str(k): round(float(v), 8) for k, v in dict(plan.memory_credit or {}).items()},
        "m1": {str(k): round(float(v), 8) for k, v in dict(plan.m1_credit or {}).items()},
        "confidence": round(float(plan.confidence), 8),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _causal_v19_plan_delta(before: CreditPlan, after: CreditPlan) -> CreditPlan:
    def delta_map(left: dict[str, float], right: dict[str, float]) -> dict[str, float]:
        keys = set(left).union(right)
        return {
            str(key): float(np.clip(float(right.get(key, 0.0)) - float(left.get(key, 0.0)), -0.50, 0.50))
            for key in keys
            if abs(float(right.get(key, 0.0)) - float(left.get(key, 0.0))) > 1e-9
        }
    return CreditPlan(
        reasoning_controller_credit=delta_map(
            dict(before.reasoning_controller_credit or {}), dict(after.reasoning_controller_credit or {})
        ),
        memory_credit=delta_map(dict(before.memory_credit or {}), dict(after.memory_credit or {})),
        m1_credit=delta_map(dict(before.m1_credit or {}), dict(after.m1_credit or {})),
        confidence=max(0.0, float(after.confidence) - float(before.confidence)),
        notes=("incremental_credit_from_verified_counterfactual_replay",),
    )


def _causal_v19_process_replay_results(self: SelfGrowingPolicy) -> dict[str, int]:
    """Consume verified V19 replay results and apply *incremental* credit.

    It is invoked before decisions and is separately callable by tools.  The
    original MT5 lifecycle is never reopened; only the difference made by a
    later verified replay is attributed through the archived entry trace.
    """
    self.ensure_compatible()
    ledger = getattr(self, "causal_ledger", None)
    journal = getattr(self, "causal_replay_journal", None)
    if not isinstance(ledger, CausalLedger) or not isinstance(journal, CounterfactualReplayJournal):
        return {"reports": 0, "validated": 0, "memory_updates": 0, "m1_updates": 0, "reasoning_updates": 0}
    summary = {"reports": 0, "validated": 0, "memory_updates": 0, "m1_updates": 0, "reasoning_updates": 0}
    archive = getattr(self, "_causal_trace_archive", {})
    signatures = getattr(self, "_causal_replay_credit_signatures", {})
    if not isinstance(archive, dict) or not isinstance(signatures, dict):
        return summary
    for trace_id in list(getattr(ledger, "report_order", []) or []):
        previous = ledger.report_for(str(trace_id))
        if previous is None:
            continue
        updated = journal.apply_to_ledger(ledger, str(trace_id))
        if updated is None:
            continue
        summary["reports"] += 1
        if not bool(updated.coverage.get("counterfactual_replay_validated", False)):
            continue
        signature = _causal_v19_plan_signature(updated.credit_plan)
        if signatures.get(str(trace_id)) == signature:
            continue
        delta = _causal_v19_plan_delta(previous.credit_plan, updated.credit_plan)
        if not (delta.reasoning_controller_credit or delta.memory_credit or delta.m1_credit):
            signatures[str(trace_id)] = signature
            continue
        delta_report = replace(updated, credit_plan=delta)
        summary["memory_updates"] += int(_causal_v19_apply_memory_credit(self, delta_report))
        trace = archive.get(str(trace_id))
        summary["m1_updates"] += int(_causal_v19_apply_m1_credit(self, trace if isinstance(trace, StepTrace) else None, delta_report))
        reasoning = getattr(self, "reasoning_graph", None)
        if isinstance(trace, StepTrace) and reasoning is not None:
            record = dict(getattr(trace, "graph_traces", {}) or {}).get("reasoning_graph", {})
            controllers = dict(record.get("controllers", {}) or {}) if isinstance(record, dict) else {}
            if controllers and delta.reasoning_controller_credit:
                try:
                    all_rewards = {str(name): 0.0 for name in getattr(reasoning, "controllers", ())}
                    all_rewards.update(dict(delta.reasoning_controller_credit))
                    reply = reasoning.observe_lifecycle(
                        realized_r=0.0,
                        advantage=0.0,
                        close_reason="CAUSAL_REPLAY_VALIDATED",
                        strength=0.35,
                        controller_traces=controllers,
                        controller_rewards=all_rewards,
                    )
                    summary["reasoning_updates"] += int(reply.get("updated_nodes", 0) or 0) + int(reply.get("updated_edges", 0) or 0)
                except Exception as exc:
                    self.growth_events.append({
                        "round": int(getattr(self, "revision", 0)),
                        "type": "causal_replay_reasoning_credit_error",
                        "trace_id": str(trace_id),
                        "error": str(exc),
                    })
        signatures[str(trace_id)] = signature
        self._causal_reports[str(trace_id)] = updated
        summary["validated"] += 1
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "causal_replay_validated_credit",
            "trace_id": str(trace_id),
            "coverage": dict(updated.coverage),
            "top_causes": [
                {"cause": item.cause_id, "confidence": float(item.confidence), "status": item.status}
                for item in updated.hypotheses[:4]
            ],
            "memory_updates": int(summary["memory_updates"]),
            "m1_updates": int(summary["m1_updates"]),
            "reasoning_updates": int(summary["reasoning_updates"]),
        })
    self._causal_replay_credit_signatures = signatures
    return summary


def _causal_v19_apply_memory_credit(self: SelfGrowingPolicy, report: CausalOutcomeReport) -> int:
    """Apply bounded, context-scoped evidence to retrieved memory cells.

    A one-off loss only adjusts utility slightly. It never deletes a cell or
    changes unrelated memories, and it explicitly records the evidence
    confidence for later fade/consolidation logic.
    """
    memory = getattr(self, "memory_supernet", None)
    cells = getattr(memory, "neural_cells", None)
    if not isinstance(cells, dict):
        return 0
    plan = report.credit_plan
    changed = 0
    for cell_id, credit in dict(plan.memory_credit or {}).items():
        cell = cells.get(str(cell_id))
        if cell is None:
            continue
        old = float(getattr(cell, "utility", 0.0) or 0.0)
        bounded = float(np.clip(float(credit), -0.20, 0.20))
        cell.utility = float(np.clip(0.985 * old + 0.015 * bounded, -5.0, 5.0))
        meta = dict(getattr(cell, "meta", {}) or {})
        meta["causal_schema_version"] = 19
        meta["causal_last_trace_id"] = str(report.trace_id)
        meta["causal_last_credit"] = bounded
        meta["causal_last_confidence"] = float(plan.confidence)
        meta["causal_last_exit"] = str(report.outcome.exit_mechanism)
        meta["causal_last_causes"] = [item.cause_id for item in report.hypotheses[:4]]
        cell.meta = meta
        changed += 1
    return changed


def _causal_v19_apply_m1_credit(self: SelfGrowingPolicy, trace: StepTrace | None, report: CausalOutcomeReport) -> int:
    """Add a small cause-specific region attribution on top of safe V18 credit.

    V18 already logs negative baseline utility without destructive suppression.
    This pass targets only the graph regions implicated by a hypothesis and
    remains no-op until causal confidence is meaningful.
    """
    if not isinstance(trace, StepTrace) or float(report.credit_plan.confidence) < 0.45:
        return 0
    region_for = {
        "selector": {"selector"},
        "critic": {"critic"},
        "hypothesis": {"hypothesis"},
    }
    graph_records = dict(getattr(trace, "graph_traces", {}) or {})
    total = 0
    for component, credit in dict(report.credit_plan.m1_credit or {}).items():
        target_regions = region_for.get(str(component))
        if not target_regions or abs(float(credit)) <= 1e-9:
            continue
        for module, m1 in self._graph_m1_modules():
            graph_id = f"m1:{str(getattr(module, 'name', 'm1_thinking') or 'm1_thinking')}"
            record = dict(graph_records.get(graph_id, {}) or {})
            edge_ids = [str(item) for item in tuple(record.get("active_edge_ids", ()) or ())]
            node_ids = [str(item) for item in tuple(record.get("active_neuron_ids", ()) or ())]
            selected_edges = []
            for edge_id in edge_ids:
                edge = m1.graph.edges.get(edge_id)
                if edge is None:
                    continue
                src = m1.graph.neurons.get(edge.src)
                dst = m1.graph.neurons.get(edge.dst)
                if (src is not None and str(getattr(src, "region", "")) in target_regions) or (dst is not None and str(getattr(dst, "region", "")) in target_regions):
                    selected_edges.append(edge_id)
            selected_nodes = [node_id for node_id in node_ids if node_id in m1.graph.neurons and str(getattr(m1.graph.neurons[node_id], "region", "")) in target_regions]
            if not selected_edges and not selected_nodes:
                continue
            # Preserve baseline safety: negative signals record utility but do
            # not destructively prune/disable production pathways.
            result = GraphPlasticityEngine().apply_credit(
                m1.graph,
                GraphCreditSignal(
                    reward=max(0.0, float(credit)),
                    cost=max(0.0, -float(credit)),
                    lifecycle_tags=("causal_region_credit",),
                    account_stress=float(np.clip(dict(getattr(trace, "account_context", {}) or {}).get("current_equity_drawdown", 0.0), 0.0, 1.0)),
                ),
                active_edge_ids=selected_edges,
                active_neuron_ids=selected_nodes,
                candidate_mode=False,
                update_mode="online",
            )
            total += int(result.updated_edges + result.updated_neurons)
    return total


def _causal_v19_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
    lifecycle: object | None = None,
) -> int:
    self.ensure_compatible()
    cache = getattr(self, "_execution_trace_cache", None)
    trace = cache.get(str(trace_id)) if isinstance(cache, dict) else None
    # Preserve the original lifecycle invariant: final credit is exactly once
    # per entry trace.  Do not re-run causal inference, replay queuing or
    # selective updates after the underlying V13/V18 path has consumed it.
    if not isinstance(trace, StepTrace):
        return 0
    ledger = getattr(self, "causal_ledger", None)
    report: CausalOutcomeReport | None = None
    if isinstance(ledger, CausalLedger):
        # The bridge cache contains decoded order parameters/execution context
        # absent from the first raw policy tick, so it refreshes the immutable
        # snapshot immediately before settlement.
        if isinstance(trace, StepTrace):
            ledger.register_decision(self, trace)
        report = ledger.settle(
            trace_id=str(trace_id), lifecycle=lifecycle, realized_r=float(realized_r),
            close_reason=str(close_reason), duration_bars=duration_bars,
        )
        replay_requests = 0
        replay_results = 0
        journal = getattr(self, "causal_replay_journal", None)
        if isinstance(journal, CounterfactualReplayJournal):
            try:
                updated = journal.apply_to_ledger(ledger, str(trace_id))
                if updated is not None:
                    report = updated
                    replay_results = int(dict(report.coverage).get("counterfactual_replay_results", 0) or 0)
                replay_requests = len(journal.enqueue(report))
            except Exception as exc:
                # Diagnostic only: causal learning continues with explicitly
                # unvalidated hypotheses rather than fabricating a replay.
                self.growth_events.append({
                    "round": int(getattr(self, "revision", 0)),
                    "type": "causal_replay_journal_error",
                    "trace_id": str(trace_id),
                    "error": str(exc),
                })
        self._causal_reports[str(trace_id)] = report
        setattr(self, "_causal_replay_requests_last", int(replay_requests))
        setattr(self, "_causal_replay_results_last", int(replay_results))
        reasoning = getattr(self, "reasoning_graph", None)
        if reasoning is not None:
            reasoning._causal_pending_controller_rewards = dict(report.credit_plan.reasoning_controller_credit)

    # Publish only after all V18 learning and V19 causal attribution are
    # complete. Otherwise the UI would show a lifecycle before its causes.
    writer = _live_observatory_writer(self)
    writer_was_enabled = bool(writer.enabled) if writer is not None else False
    if writer is not None and writer_was_enabled:
        writer.enabled = False
    try:
        credited = _CAUSAL_V19_POLICY_LIFECYCLE(
            self, trace_id, realized_r, close_reason=close_reason,
            duration_bars=duration_bars, min_holding_bars=min_holding_bars,
        )
    finally:
        reasoning = getattr(self, "reasoning_graph", None)
        if reasoning is not None and hasattr(reasoning, "_causal_pending_controller_rewards"):
            delattr(reasoning, "_causal_pending_controller_rewards")
        if writer is not None:
            writer.enabled = writer_was_enabled

    if report is not None:
        memory_updates = _causal_v19_apply_memory_credit(self, report)
        m1_updates = _causal_v19_apply_m1_credit(self, trace if isinstance(trace, StepTrace) else None, report)
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "causal_lifecycle_settled",
            "trace_id": str(trace_id),
            "exit_mechanism": str(report.outcome.exit_mechanism),
            "coverage": dict(report.coverage),
            "top_causes": [
                {"cause": item.cause_id, "confidence": float(item.confidence), "status": item.status}
                for item in report.hypotheses[:4]
            ],
            "memory_updates": int(memory_updates),
            "m1_region_updates": int(m1_updates),
            "counterfactual_replay_validated": bool(report.coverage.get("counterfactual_replay_validated", False)),
            "counterfactual_replay_requests": int(getattr(self, "_causal_replay_requests_last", 0) or 0),
            "counterfactual_replay_results": int(getattr(self, "_causal_replay_results_last", 0) or 0),
        })
        credited += int(memory_updates + m1_updates)
    if writer is not None and writer_was_enabled:
        writer.lifecycle(
            self, trace_id=str(trace_id), realized_r=float(realized_r),
            close_reason=str(close_reason), duration_bars=duration_bars, credited=int(credited),
        )
    return int(credited)


def _causal_v19_report(self: SelfGrowingPolicy, trace_id: str) -> dict[str, object]:
    self.ensure_compatible()
    ledger = getattr(self, "causal_ledger", None)
    if not isinstance(ledger, CausalLedger):
        return {}
    report = ledger.report_for(str(trace_id))
    return {} if report is None else report.to_dict()


SelfGrowingPolicy.ensure_compatible = _causal_v19_ensure
SelfGrowingPolicy.copy = _causal_v19_copy
SelfGrowingPolicy._query_memory = _causal_v19_query_memory
SelfGrowingPolicy.act = _causal_v19_policy_act
SelfGrowingPolicy.apply_execution_lifecycle_credit = _causal_v19_lifecycle_credit
SelfGrowingPolicy.causal_lifecycle_report = _causal_v19_report
SelfGrowingPolicy.process_causal_replay_results = _causal_v19_process_replay_results

# ---------------------------------------------------------------------------
# V24 continuous MT5 session
# ---------------------------------------------------------------------------
def continuous_train_mt5(
    config: AppConfig,
    *,
    output_path: str | Path,
    seed_policy_path: str | Path | None = None,
    runner_dir: str | Path = "mt5_v2_runner",
    timeout_sec: int = 300,
    visual: bool = False,
    shutdown_terminal: bool = True,
    from_date: str | None = None,
    to_date: str | None = None,
    backtest_days: int | None = None,
    model: int = 0,
    policy_core: str = "current",
    m1_backend: str = "direct_rnn",
    session_id: str | None = None,
) -> dict[str, object]:
    """Run one persistent, event-driven MT5 learning session.

    Unlike ``train_self_growing_mt5`` this function does not clone a policy,
    does not create candidate trials and does not promote/reject a whole policy.
    The same policy instance serves every bridge request. Final lifecycle credit
    is tailed from the execution ledger and applied at a request safe-point
    after the configured post-exit wait.

    The function intentionally still performs one frozen Strategy Tester pass
    over a bounded date range. A date range is an execution session boundary,
    not a training *trial*: adaptive state, evidence and specialist modules are
    serialized into ``output_path`` and continue in the next session.
    """
    from .continuous_session import ContinuousRuntimeConfig, ContinuousSessionRuntime

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    core = _normalize_policy_core(policy_core)
    seed_path = Path(seed_policy_path) if seed_policy_path else None
    if seed_path is not None and not seed_path.exists():
        raise FileNotFoundError(f"continuous seed policy not found: {seed_path}")

    policy = load_policy(seed_path, policy_core=core) if seed_path else _seed_policy_for_core(core)
    policy = _apply_m1_backend(policy, m1_backend)
    policy.ensure_compatible()
    policy.graph_trial_mode = False
    policy.graph_trial_id = ""
    policy.continuous_mode = True

    causal_cfg = getattr(config, "causal", None)
    if causal_cfg is not None and bool(getattr(causal_cfg, "enabled", True)):
        policy.causal_replay_requests_path = str(getattr(causal_cfg, "replay_request_path", "outputs/causal_replay_requests.jsonl"))
        policy.causal_replay_results_path = str(getattr(causal_cfg, "replay_result_path", "outputs/causal_replay_results.jsonl"))
        policy.causal_replay_journal = None
        policy.ensure_compatible()
        ledger = getattr(policy, "causal_ledger", None)
        if isinstance(ledger, CausalLedger):
            ledger.max_decisions = int(getattr(causal_cfg, "max_decisions", ledger.max_decisions))
            ledger.max_reports = int(getattr(causal_cfg, "max_reports", ledger.max_reports))

    expectancy_cfg = getattr(config, "expectancy", None)
    if expectancy_cfg is not None:
        policy.expectancy_learning = bool(getattr(expectancy_cfg, "enabled", True))
        policy.expectancy_min_samples = int(getattr(expectancy_cfg, "min_samples", 6))

    if getattr(policy, "m1_backend", "direct_rnn") == "m1_graph":
        policy.configure_graph_supernetwork(getattr(config, "supernetwork", None))
        policy.configure_learning_integrity(getattr(config, "learning", None))
        policy.graph_supernetwork_config["max_modules"] = max(1, int(getattr(getattr(config, "model", None), "max_modules", 4) or 4))

    observability_cfg = getattr(config, "observability", None)
    if bool(getattr(observability_cfg, "enabled", False)):
        policy.enable_live_observatory(
            observability_cfg,
            run_id=f"continuous-{int(time.time())}-{output.stem}",
            source="continuous_train_mt5_v25",
            metadata={"output_path": str(output), "runner_dir": str(runner_dir)},
        )

    continuous_cfg = getattr(config, "continuous", ContinuousRuntimeConfig())
    if not isinstance(continuous_cfg, ContinuousRuntimeConfig):
        continuous_cfg = ContinuousRuntimeConfig()
    if not continuous_cfg.enabled:
        continuous_cfg = replace(continuous_cfg, enabled=True)
    actual_session_id = session_id or f"continuous_{int(time.time())}_{output.stem}"
    runtime = ContinuousSessionRuntime(
        continuous_cfg,
        session_id=actual_session_id,
        policy_output_path=output,
    )

    start = str(from_date or "2025.01.01")
    end = str(to_date or "")
    if not end:
        days = max(1, int(backtest_days or 31))
        try:
            end = (datetime.strptime(start, "%Y.%m.%d") + timedelta(days=days)).strftime("%Y.%m.%d")
        except ValueError:
            end = "2025.02.01"
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=Path(runner_dir),
            from_date=start,
            to_date=end,
            timeout_sec=int(timeout_sec),
            max_positions=int(config.risk.max_positions),
            max_holding_bars=int(config.orders.max_holding_bars),
            min_daily_trades=int(config.orders.min_daily_trades),
            max_daily_trades=int(config.orders.max_daily_trades),
            **_mt5_execution_contract_kwargs(config),
            visual=bool(visual),
            shutdown_terminal=bool(shutdown_terminal),
            model=int(model),
            # The normal journal remains available for diagnostics, but V24
            # causal/event state is the persistent source of continuous learning.
            experience_journal=None,
            continuous_runtime=runtime,
        )
    )
    fitness, metrics = evaluator.evaluate_policy(1, policy, 0)
    # The runtime may already have forced a final checkpoint. Saving the policy
    # explicitly is the portable session handoff used by the next invocation.
    with output.open("wb") as handle:
        pickle.dump(policy, handle, protocol=pickle.HIGHEST_PROTOCOL)
    runtime_summary = dict(getattr(evaluator, "last_continuous_summary", {}) or {})
    runtime.store.emit(
        "continuous_policy_saved",
        event_id=f"continuous_policy_saved:{actual_session_id}",
        sequence=int(runtime.current_sequence),
        payload={"output_path": str(output), "fitness": float(fitness)},
    )
    return {
        "mode": "continuous_event_driven_v25",
        "session_id": actual_session_id,
        "policy": str(output),
        "fitness": float(fitness),
        "metrics": _mt5_metrics_to_dict(metrics),
        "continuous": runtime_summary,
        "date_range": {"from": start, "to": end},
        "requirements": {
            "explicit_final_close": "EA execution ledger must emit position_final_close=1 for online lifecycle credit.",
            "post_exit_path": "Trajectory/post-exit causal conclusions require V24 EA trajectory fields; absent fields are marked incomplete.",
            "replay": "Only externally verified replay results can consolidate a continuous specialist module.",
        },
    }


# ---------------------------------------------------------------------------
# V24 continuous trade-activity safeguards
# ---------------------------------------------------------------------------
# The V11 action-head wrapper inherited the original neutral gate behaviour:
# sigmoid(0)=0.5 caused every neutral decision to dampen entries and boost
# HOLD.  In a persistent continuous session that creates a one-way
# conservative ratchet.  V24 preserves safety response only for *excess over
# neutral*, while keeping the existing parameter-replacement overlay.
_V24_PRIOR_ACTION_HEADS = SelfGrowingPolicy._apply_biological_action_heads
_V24_PRIOR_LIFECYCLE_CREDIT = SelfGrowingPolicy.apply_execution_lifecycle_credit


def _v24_apply_biological_action_heads(
    self: SelfGrowingPolicy,
    output: np.ndarray,
    state: np.ndarray,
    obs: np.ndarray,
    gates: np.ndarray,
    memory_query: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    adjusted = np.asarray(output, dtype=float).copy()
    if adjusted.size != self.output_size:
        adjusted = self._resize_vector(adjusted, self.output_size)

    head_context = self._action_head_context(state, adjusted, memory_query, gates)
    params = self.action_parameter_head.parameters(head_context)
    parameter_offset = PARAMETER_OFFSET if adjusted.size >= OUTPUT_COUNT else 11
    param_count = min(8, max(0, adjusted.size - parameter_offset), params.size)
    if param_count > 0:
        adjusted[parameter_offset:parameter_offset + param_count] = np.tanh(
            0.35 * adjusted[parameter_offset:parameter_offset + param_count]
            + 0.65 * params[:param_count]
        )

    risk_info: dict[str, object] = dict(
        self.risk_cost_gate.gate(self._risk_gate_context(obs, adjusted, gates, memory_query))
    )
    # A sigmoid gate has neutral prior 0.5.  Only the excess above that
    # neutral prior is allowed to suppress entries; otherwise a fresh policy
    # is conservative before it has observed any risk.
    risk_excess = float(np.clip((float(risk_info.get("risk", 0.5)) - 0.5) / 0.5, 0.0, 1.0))
    cost_excess = float(np.clip((float(risk_info.get("cost", 0.5)) - 0.5) / 0.5, 0.0, 1.0))
    patience_excess = float(np.clip((float(risk_info.get("patience", 0.5)) - 0.5) / 0.5, 0.0, 1.0))
    dampen = float(np.clip(0.5 * risk_excess + 0.5 * cost_excess, 0.0, 1.0))
    action_count = ACTION_COUNT if adjusted.size >= OUTPUT_COUNT else 11
    if adjusted.size >= action_count:
        adjusted[1:7] *= 1.0 - 0.22 * dampen
        adjusted[0] += 0.08 * patience_excess
        adjusted[8:11] += 0.08 * dampen

    # Preserve V11 parameter-replacement behaviour without reusing the old
    # global HOLD-biased action gate.
    replacement_offset, replacement_count = _m1v11_parameter_indices(adjusted.size)
    scope = _m1v11_scope_from_output(self, adjusted)
    if replacement_count > 0:
        replacement_params, replacement_meta = _m1v11_query_parameter_replacement_bias(
            self,
            state,
            memory_query,
            adjusted.size,
            current_scope=scope,
            return_metadata=True,
        )
        kept = min(8, replacement_count, replacement_params.size)
        if kept > 0 and float(np.max(np.abs(replacement_params[:kept]))) > 1e-9:
            adjusted[replacement_offset:replacement_offset + kept] = np.tanh(
                0.92 * adjusted[replacement_offset:replacement_offset + kept]
                + 0.08 * replacement_params[:kept]
            )
            risk_info["parameter_replacement_applied"] = 1.0
        else:
            risk_info["parameter_replacement_applied"] = 0.0
        risk_info["parameter_replacement_proposal_ids"] = list(replacement_meta.get("proposal_ids", []) or [])
        risk_info["parameter_replacement_mode"] = str(replacement_meta.get("mode", "none"))
        risk_info["parameter_replacement_scope"] = scope

    risk_info.update({
        "neutral_prior": 0.5,
        "risk_excess": risk_excess,
        "cost_excess": cost_excess,
        "patience_excess": patience_excess,
        "action_dampen": dampen,
        "trade_activity_guard_v24": 1.0,
    })
    return adjusted, risk_info


def _v24_continuous_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: float | None = None,
    lifecycle: object | None = None,
) -> int:
    """Apply normal V19 causal credit, then one bounded risk-gate update.

    Continuous bridge mode deliberately skips immediate economic
    ``observe_transition`` feedback.  This avoids double punishment at close
    time, but it also means the risk gate needs one final, idempotent update
    after a *verified* lifecycle settlement.
    """
    trace = self._trace_cache.get(str(trace_id)) if hasattr(self, "_trace_cache") else None
    credited = _V24_PRIOR_LIFECYCLE_CREDIT(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=3 if min_holding_bars is None else min_holding_bars,
        lifecycle=lifecycle,
    )
    if not bool(getattr(self, "_continuous_session_active", False)):
        return int(credited)
    if not isinstance(trace, StepTrace):
        return int(credited)

    signature = _event_id(
        "continuous_final_risk_gate",
        trace_id,
        str(close_reason),
        round(float(realized_r), 8),
        getattr(lifecycle, "position_ticket", ""),
    )
    seen = getattr(self, "_continuous_final_risk_gate_signatures", None)
    if not isinstance(seen, set):
        seen = set()
        setattr(self, "_continuous_final_risk_gate_signatures", seen)
    if signature in seen:
        return int(credited)
    seen.add(signature)

    try:
        gates = np.asarray(getattr(trace, "gate_values", np.zeros(len(getattr(self, "modules", [])))), dtype=float)
        state = np.asarray(getattr(trace, "encoded_state", np.zeros(self.input_size)), dtype=float)
        memory_query = self._query_memory(state)
        selected = np.asarray(getattr(trace, "selected_action", np.zeros(self.output_size)), dtype=float)
        context = self._risk_gate_context(np.asarray(trace.raw_obs, dtype=float), selected, gates, memory_query)
        outcome = float(np.clip(realized_r, -3.0, 3.0))
        loss_signal = float(np.clip(max(0.0, -outcome) / 2.0, 0.0, 1.0))
        slip = float(abs(getattr(lifecycle, "slippage_r", 0.0) or 0.0))
        cost_signal = float(np.clip(slip / 0.5, 0.0, 1.0))
        # Patience is not a direct loss proxy.  Keep it lower than risk so a
        # few normal losses do not convert the policy into permanent HOLD.
        target = np.asarray([loss_signal, cost_signal, 0.35 * loss_signal], dtype=float)
        strength = float(np.clip(0.08 + 0.12 * abs(outcome), 0.08, 0.30))
        self.risk_cost_gate.observe(context, target, learning_rate=strength)
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "continuous_final_risk_gate_credit_v24",
            "trace_id": str(trace_id),
            "realized_r": float(realized_r),
            "risk_target": target.tolist(),
            "learning_rate": strength,
            "close_reason": str(close_reason),
        })
        credited += 1
    except Exception as exc:
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "continuous_final_risk_gate_credit_error_v24",
            "trace_id": str(trace_id),
            "error": type(exc).__name__,
        })
    return int(credited)


SelfGrowingPolicy._apply_biological_action_heads = _v24_apply_biological_action_heads
SelfGrowingPolicy.apply_execution_lifecycle_credit = _v24_continuous_lifecycle_credit

# ---------------------------------------------------------------------------
# V25: anti-HOLD-ratchet hardening for persistent continuous sessions
# ---------------------------------------------------------------------------
# V24 made the risk gate neutral at exactly 0.5, but old checkpoints could
# still carry global rejection bias in gate logits.  More critically, the
# final-lifecycle update read a non-existent cache and passed the wrong keyword
# to ``observe``.  V25 keeps risk learning but confines entry suppression to
# observable account/execution pressure; an elevated recurrent gate by itself
# cannot silently turn every future market regime into HOLD.

_V25_PRIOR_ACTION_HEADS = SelfGrowingPolicy._apply_biological_action_heads
_V25_PRIOR_LIFECYCLE_CREDIT = SelfGrowingPolicy.apply_execution_lifecycle_credit
_V25_PRIOR_REJECTION_FEEDBACK = SelfGrowingPolicy.absorb_rejection_feedback


def _v25_gate_observe_probability(
    self: NeuralRiskCostGate,
    values: np.ndarray,
    probabilities: np.ndarray,
    *,
    strength: float = 1.0,
) -> None:
    """Learn gate *probabilities* through logits, not raw sigmoid targets.

    ``NeuralScalarNetwork.observe`` learns raw pre-sigmoid outputs.  Passing
    0.5 as though it were a neutral probability actually targets a sigmoid of
    about 0.62, creating a systematic risk/HOLD drift.  This helper converts
    bounded probabilities to the output space that ``gate()`` later sigmoids.
    """
    probs = np.asarray(probabilities, dtype=float).reshape(-1)
    probs = np.clip(probs, 0.05, 0.95)
    logits = np.log(probs / np.maximum(1e-9, 1.0 - probs))
    NeuralScalarNetwork.observe(self, values, logits, strength=float(strength))


NeuralRiskCostGate.observe_probability = _v25_gate_observe_probability


def _v25_contextual_entry_pressure(policy: SelfGrowingPolicy) -> float:
    """Return hard current-session pressure independent of learned gate bias."""
    account = dict(getattr(policy, "_account_context", {}) or {})
    decision = dict(getattr(policy, "_decision_context", {}) or {})
    execution = dict(decision.get("execution_context", {}) or {})
    drawdown = float(account.get("current_equity_drawdown", account.get("balance_drawdown", 0.0)) or 0.0)
    open_risk = float(account.get("open_risk_ratio", account.get("position_count_norm", 0.0)) or 0.0)
    day_pressure = float(execution.get("day_trade_pressure", account.get("day_trade_pressure", 0.0)) or 0.0)
    spread = max(0.0, float(execution.get("spread_points", 0.0) or 0.0))
    # These are only *hard pressure* guards, not a learnt execution model.
    # Thresholds keep ordinary BTC M1 spreads/drawdown from globally blocking
    # entry and leave fine-grained execution learning to modules/replay.
    drawdown_pressure = float(np.clip((drawdown - 0.05) / 0.15, 0.0, 1.0))
    exposure_pressure = float(np.clip((open_risk - 0.75) / 0.25, 0.0, 1.0))
    day_pressure = float(np.clip((day_pressure - 1.00) / 0.50, 0.0, 1.0))
    spread_pressure = float(np.clip((spread - 80.0) / 160.0, 0.0, 1.0))
    return float(max(drawdown_pressure, exposure_pressure, day_pressure, spread_pressure))


def _v25_apply_biological_action_heads(
    self: SelfGrowingPolicy,
    output: np.ndarray,
    state: np.ndarray,
    obs: np.ndarray,
    gates: np.ndarray,
    memory_query: dict[str, object],
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply V25 bounded, context-gated entry caution.

    V24's output is used for the action parameter/replacement path, then its
    broad learned-gate action modifications are reversed and replaced with a
    hard-pressure-gated effect.  This preserves backward-compatible parameter
    learning without allowing a historical global risk state to suppress entry
    indefinitely.
    """
    adjusted, info_any = _V25_PRIOR_ACTION_HEADS(self, output, state, obs, gates, memory_query)
    adjusted = np.asarray(adjusted, dtype=float).copy()
    info: dict[str, object] = dict(info_any or {})
    continuous = bool(getattr(self, "_continuous_session_active", False))
    if not continuous:
        return adjusted, info

    cfg = getattr(self, "_continuous_runtime_config", None)
    max_dampen = float(getattr(cfg, "risk_gate_entry_dampen_max", 0.08) or 0.08)
    inactivity_after = int(getattr(cfg, "inactivity_recalibration_after_decisions", 48) or 48)
    inactivity_cap = float(getattr(cfg, "inactivity_risk_gate_dampen_cap", 0.02) or 0.02)

    # Undo V24's direct global risk action transform before applying V25.
    old_dampen = float(info.get("action_dampen", 0.0) or 0.0)
    old_patience = float(info.get("patience_excess", 0.0) or 0.0)
    if adjusted.size >= ACTION_COUNT:
        factor = 1.0 - 0.22 * old_dampen
        if factor > 1e-6:
            adjusted[1:7] /= factor
        adjusted[0] -= 0.08 * old_patience
        adjusted[8:11] -= 0.08 * old_dampen

    risk_excess = float(np.clip(float(info.get("risk_excess", 0.0) or 0.0), 0.0, 1.0))
    cost_excess = float(np.clip(float(info.get("cost_excess", 0.0) or 0.0), 0.0, 1.0))
    patience_excess = float(np.clip(float(info.get("patience_excess", 0.0) or 0.0), 0.0, 1.0))
    hard_pressure = _v25_contextual_entry_pressure(self)
    gate_pressure = float(np.clip(0.5 * risk_excess + 0.5 * cost_excess, 0.0, 1.0))
    dampen = float(np.clip(max_dampen * gate_pressure * hard_pressure, 0.0, max_dampen))

    starvation = dict(getattr(self, "_experience_starvation", {}) or {})
    no_trade = float(starvation.get("no_trade_streak", 0.0) or 0.0)
    recalibrating = no_trade >= max(1, inactivity_after) and hard_pressure < 0.25
    if recalibrating:
        dampen = min(dampen, inactivity_cap)
        # Recurrent state represents short-term risk, not a permanent session
        # verdict.  Decay it only during a normal-execution inactivity window.
        try:
            self.risk_cost_gate.state = 0.90 * np.asarray(self.risk_cost_gate.state, dtype=float)
        except Exception:
            pass

    raw_bias = np.asarray(memory_query.get("replacement_action_bias", memory_query.get("success_action_bias", memory_query.get("action_bias", np.zeros(self.output_size)))), dtype=float).reshape(-1)
    memory_hold_bias = float(max(0.0, raw_bias[0])) if raw_bias.size else 0.0
    memory_entry_bias = float(np.max(np.maximum(0.0, raw_bias[1:7]))) if raw_bias.size >= 7 else 0.0
    memory_hold_recenter = 0.0
    # A long inactivity streak in a normal execution context is not evidence
    # that HOLD memories are correct.  Recenter only the unsupported excess;
    # do not force an entry or alter verified local direction overlays.
    if recalibrating and memory_hold_bias > memory_entry_bias and hard_pressure < 0.25:
        memory_hold_recenter = float(np.clip(memory_hold_bias - memory_entry_bias, 0.0, 0.06))
        if adjusted.size > 0:
            adjusted[0] -= memory_hold_recenter

    if adjusted.size >= ACTION_COUNT:
        adjusted[1:7] *= 1.0 - dampen
        adjusted[0] += 0.50 * dampen * patience_excess
        adjusted[8:11] += 0.35 * dampen

    info.update({
        "trade_activity_guard_v25": 1.0,
        "action_dampen": float(dampen),
        "risk_gate_hard_pressure": float(hard_pressure),
        "risk_gate_recalibrating": float(recalibrating),
        "memory_hold_bias": float(memory_hold_bias),
        "memory_entry_bias": float(memory_entry_bias),
        "memory_hold_recenter": float(memory_hold_recenter),
        "no_trade_streak": float(no_trade),
    })
    return adjusted, info


def _v25_continuous_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: float | None = None,
    lifecycle: object | None = None,
) -> int:
    """Settle normal causal credit, then apply one small calibrated gate update."""
    # Preserve the immutable entry trace before V19 settlement. Settlement
    # consumes the execution cache deliberately, so looking it up afterward
    # would make V26 continuous sessions lose the calibrated gate update.
    cache_before = getattr(self, "_execution_trace_cache", None)
    trace_before = cache_before.get(str(trace_id)) if isinstance(cache_before, dict) else None
    # Call the V19 causal settlement directly.  Calling the V24 wrapper here
    # can invoke its obsolete gate-credit branch as well, which reintroduces
    # duplicate economic learning when an old _trace_cache happens to exist.
    credited = _V24_PRIOR_LIFECYCLE_CREDIT(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=3 if min_holding_bars is None else min_holding_bars,
        lifecycle=lifecycle,
    )
    if not bool(getattr(self, "_continuous_session_active", False)):
        return int(credited)

    trace = trace_before
    cache = getattr(self, "_execution_trace_cache", None)
    if trace is None:
        trace = cache.get(str(trace_id)) if isinstance(cache, dict) else None
    if trace is None:
        archive = getattr(self, "_causal_trace_archive", None)
        trace = archive.get(str(trace_id)) if isinstance(archive, dict) else None
    if not isinstance(trace, StepTrace):
        return int(credited)

    signature_payload = "|".join([
        "continuous_final_risk_gate_v25",
        str(trace_id),
        str(close_reason),
        str(round(float(realized_r), 8)),
        str(getattr(lifecycle, "position_ticket", "")),
    ])
    signature = "continuous_final_risk_gate_v25:" + hashlib.sha1(signature_payload.encode("utf-8")).hexdigest()[:20]
    seen = getattr(self, "_continuous_final_risk_gate_signatures_v25", None)
    if not isinstance(seen, set):
        seen = set()
        setattr(self, "_continuous_final_risk_gate_signatures_v25", seen)
    if signature in seen:
        return int(credited)
    seen.add(signature)

    try:
        gates = np.asarray(getattr(trace, "gate_values", np.zeros(len(getattr(self, "modules", [])))), dtype=float)
        state = np.asarray(getattr(trace, "encoded_state", np.zeros(self.input_size)), dtype=float)
        memory_query = self._query_memory(state)
        selected = np.asarray(getattr(trace, "selected_action", np.zeros(self.output_size)), dtype=float)
        context = self._risk_gate_context(np.asarray(trace.raw_obs, dtype=float), selected, gates, memory_query)
        outcome = float(np.clip(realized_r, -3.0, 3.0))
        loss_signal = float(np.clip(max(0.0, -outcome) / 2.0, 0.0, 1.0))
        slip = float(abs(getattr(lifecycle, "slippage_r", 0.0) or 0.0))
        cost_signal = float(np.clip(slip / 0.5, 0.0, 1.0))
        # Probabilities are deliberately close to neutral.  A single normal SL
        # raises caution slightly; it cannot create a global anti-entry state.
        target_prob = np.asarray([
            0.50 + 0.08 * loss_signal,
            0.50 + 0.08 * cost_signal,
            0.50 + 0.03 * loss_signal,
        ], dtype=float)
        origin = str(dict(getattr(trace, "action_details", {}) or {}).get("action_origin", "policy"))
        origin_scale = 0.50 if origin == "bootstrap" else 1.0
        strength = float(np.clip((0.025 + 0.050 * max(loss_signal, cost_signal)) * origin_scale, 0.02, 0.08))
        self.risk_cost_gate.observe_probability(context, target_prob, strength=strength)
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "continuous_final_risk_gate_credit_v25",
            "trace_id": str(trace_id),
            "realized_r": float(realized_r),
            "target_probability": target_prob.tolist(),
            "strength": strength,
            "close_reason": str(close_reason),
            "action_origin": origin,
        })
        credited += 1
    except Exception as exc:
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "continuous_final_risk_gate_credit_error_v25",
            "trace_id": str(trace_id),
            "error": type(exc).__name__,
        })
    return int(credited)


def _v25_absorb_rejection_feedback(
    self: SelfGrowingPolicy,
    rejected_policy: "SelfGrowingPolicy",
    *,
    reject_reasons: list[str] | tuple[str, ...] = (),
    metrics: dict[str, object] | None = None,
    fitness_delta: float = 0.0,
) -> dict[str, object]:
    """Block legacy global rejection bias while a continuous session is active."""
    if bool(getattr(self, "_continuous_session_active", False)):
        event = {
            "round": int(getattr(self, "revision", 0)),
            "type": "continuous_legacy_rejection_feedback_blocked_v25",
            "reasons": [str(item) for item in reject_reasons],
            "fitness_delta": float(fitness_delta),
        }
        try:
            self.growth_events.append(event)
        except Exception:
            pass
        return {"blocked": True, "reason": "continuous_mode_uses_local_causal_updates", **event}
    return _V25_PRIOR_REJECTION_FEEDBACK(
        self,
        rejected_policy,
        reject_reasons=reject_reasons,
        metrics=metrics,
        fitness_delta=fitness_delta,
    )


# V25 also sanitises legacy replacement-memory HOLD proposals at the source
# after a sustained, normal-execution inactivity window.  Older journals may
# contain broad HOLD/WAIT action vectors for timing, spread or shock events.
# They remain available as causal/parameter evidence but may no longer become
# an unbounded action logit that starves policy-origin trade evidence.
_V25_PRIOR_QUERY_MEMORY = SelfGrowingPolicy._query_memory


def _v25_sanitize_continuous_memory_bias(
    policy: SelfGrowingPolicy,
    result: dict[str, object],
) -> dict[str, object]:
    if not bool(getattr(policy, "_continuous_session_active", False)):
        return result
    cfg = getattr(policy, "_continuous_runtime_config", None)
    inactivity_after = int(getattr(cfg, "inactivity_recalibration_after_decisions", 48) or 48)
    starvation = dict(getattr(policy, "_experience_starvation", {}) or {})
    no_trade = float(starvation.get("no_trade_streak", 0.0) or 0.0)
    if no_trade < max(1, inactivity_after) or _v25_contextual_entry_pressure(policy) >= 0.25:
        return result
    key = "replacement_action_bias" if "replacement_action_bias" in result else ("success_action_bias" if "success_action_bias" in result else "action_bias")
    raw = np.asarray(result.get(key, np.zeros(getattr(policy, "output_size", OUTPUT_COUNT))), dtype=float).reshape(-1)
    if raw.size < 7:
        return result
    bias = raw.copy()
    hold = max(0.0, float(bias[0]))
    entry_peak = float(np.max(np.maximum(0.0, bias[1:7])))
    # Permit a modest HOLD preference, but only enough to break close ties.
    # A persistent excess is legacy conservatism, not fresh evidence, after a
    # long normal-context no-trade streak.
    cap = entry_peak + 0.035
    suppressed = max(0.0, hold - cap)
    if suppressed <= 1e-12:
        return result
    bias[0] = min(float(bias[0]), cap)
    out = dict(result)
    out["raw_replacement_action_bias_v25"] = raw.copy()
    out["replacement_action_bias"] = bias.copy()
    out["action_bias"] = bias.copy()
    if "success_action_bias" in out:
        out["success_action_bias"] = bias.copy()
    out["trade_activity_memory_hold_suppressed_v25"] = float(suppressed)
    out["trade_activity_memory_hold_suppression_reason_v25"] = "normal_context_sustained_inactivity"
    return out


def _v25_query_memory(self: SelfGrowingPolicy, state: np.ndarray, *args: object, **kwargs: object) -> dict[str, object]:
    result = _V25_PRIOR_QUERY_MEMORY(self, state, *args, **kwargs)
    return _v25_sanitize_continuous_memory_bias(self, dict(result or {}))


SelfGrowingPolicy._query_memory = _v25_query_memory
SelfGrowingPolicy._apply_biological_action_heads = _v25_apply_biological_action_heads
SelfGrowingPolicy.apply_execution_lifecycle_credit = _v25_continuous_lifecycle_credit
SelfGrowingPolicy.absorb_rejection_feedback = _v25_absorb_rejection_feedback


# ---------------------------------------------------------------------------
# V26: no-order is NO_COMMAND, never a neural HOLD feedback target
# ---------------------------------------------------------------------------
_V26_PRIOR_TRACE_ACTION_INDEX = SelfGrowingPolicy._trace_action_index


def _v26_trace_action_index(trace: StepTrace, action_count: int) -> int:
    details = dict(getattr(trace, "action_details", {}) or {})
    if bool(details.get("no_neural_command", False)) or str(details.get("action_attribution_source", "")).lower() == "none":
        return -1
    selected = _V26_PRIOR_TRACE_ACTION_INDEX(trace, action_count)
    # In a continuous session the HOLD wire action has no neural attribution.
    if bool(getattr(trace, "continuous_no_hold_feedback", False)) and int(selected) == 0:
        return -1
    return int(selected)


SelfGrowingPolicy._trace_action_index = staticmethod(_v26_trace_action_index)


# ---------------------------------------------------------------------------
# V27: FLAT_IDLE execution-state firebreak
# ---------------------------------------------------------------------------
# MT5 being flat is never an action selected by the policy.  Keep explicit
# neural HOLD as a legal output, but make every flat/no-execution trace
# ineligible for action credit even if a legacy path reaches _trace_action_index.
_V27_PRIOR_TRACE_ACTION_INDEX = SelfGrowingPolicy._trace_action_index


def _v27_trace_action_index(trace: StepTrace, action_count: int) -> int:
    details = dict(getattr(trace, "action_details", {}) or {})
    if bool(details.get("flat_idle_non_feedback", False)):
        return -1
    if str(details.get("action_attribution_source", "")).lower() in {
        "none", "flat_idle", "hold_no_execution",
    }:
        return -1
    selected = int(_V27_PRIOR_TRACE_ACTION_INDEX(trace, action_count))
    # In a continuous session, HOLD is allowed as an output but has no executed
    # position lifecycle.  It therefore cannot become a feedback/credit target.
    if bool(getattr(trace, "continuous_flat_idle_firebreak", False)) and selected == 0:
        return -1
    return selected


SelfGrowingPolicy._trace_action_index = staticmethod(_v27_trace_action_index)


# ---------------------------------------------------------------------------
# V8 mechanism integrity: bounded trading world model, partitioned working
# memory, and shadow-only curiosity. This final layer preserves the V7 bridge
# wire schema while ensuring all policy APIs share the new mechanism contract.
# ---------------------------------------------------------------------------
_V8_PRIOR_POLICY_ENSURE = SelfGrowingPolicy.ensure_compatible
_V8_PRIOR_POLICY_COPY = SelfGrowingPolicy.copy
_V8_PRIOR_POLICY_RESET_RUNTIME = SelfGrowingPolicy.reset_runtime_state
_V8_PRIOR_POLICY_CONFIGURE_INTEGRITY = SelfGrowingPolicy.configure_learning_integrity
_V8_PRIOR_POLICY_ACT = SelfGrowingPolicy.act
_V8_PRIOR_POLICY_OBSERVE = SelfGrowingPolicy.observe_transition
_V8_PRIOR_POLICY_QUERY_MEMORY = SelfGrowingPolicy._query_memory


def _v8_policy_ensure(self: SelfGrowingPolicy, target_input_size: int | None = None) -> None:
    _V8_PRIOR_POLICY_ENSURE(self, target_input_size=target_input_size)
    size = int(getattr(self, "input_size", INPUT_COUNT) or INPUT_COUNT)
    actions = int(getattr(self, "output_size", OUTPUT_COUNT) or OUTPUT_COUNT)
    world = getattr(self, "trading_world_model", None)
    if not isinstance(world, TradingWorldModel):
        world = TradingWorldModel.seed(observation_size=size, action_size=actions, seed=20260629)
        self.trading_world_model = world
    world.ensure_compatible(observation_size=size, action_size=actions)
    partitioned = getattr(self, "partitioned_memory", None)
    if not isinstance(partitioned, PartitionedWorkingMemory):
        partitioned = PartitionedWorkingMemory()
        self.partitioned_memory = partitioned
    partitioned.ensure_compatible()
    mode = str(getattr(self, "live_curiosity_mode", "shadow_only") or "shadow_only").strip().lower()
    self.live_curiosity_mode = mode if mode in {"shadow_only", "evidence_only"} else "shadow_only"
    self.artifact_version = max(int(getattr(self, "artifact_version", 0) or 0), 10)


def _v8_policy_copy(self: SelfGrowingPolicy) -> SelfGrowingPolicy:
    copied = _V8_PRIOR_POLICY_COPY(self)
    _v8_policy_ensure(self)
    copied.trading_world_model = self.trading_world_model.copy()
    copied.partitioned_memory = self.partitioned_memory.copy()
    copied.live_curiosity_mode = str(self.live_curiosity_mode)
    copied.artifact_version = max(int(getattr(copied, "artifact_version", 0) or 0), 10)
    return copied


def _v8_policy_reset_runtime_state(self: SelfGrowingPolicy) -> None:
    _V8_PRIOR_POLICY_RESET_RUNTIME(self)
    _v8_policy_ensure(self)
    self.trading_world_model.reset_runtime_state()
    self.partitioned_memory.reset()


def _v8_policy_configure_learning_integrity(self: SelfGrowingPolicy, settings: object | None = None) -> None:
    _V8_PRIOR_POLICY_CONFIGURE_INTEGRITY(self, settings)
    mode = str(getattr(settings, "live_curiosity_mode", "shadow_only") or "shadow_only").strip().lower()
    self.live_curiosity_mode = mode if mode in {"shadow_only", "evidence_only"} else "shadow_only"
    _v8_policy_ensure(self)


def _v8_policy_query_memory(self: SelfGrowingPolicy, observation: np.ndarray) -> dict[str, object]:
    result = dict(_V8_PRIOR_POLICY_QUERY_MEMORY(self, observation) or {})
    partition = getattr(self, "_v8_partition_context", None)
    world = getattr(self, "_v8_world_context", None)
    if isinstance(partition, dict):
        result.update({key: value for key, value in partition.items() if key.endswith("_memory_context")})
    if isinstance(world, dict):
        result["world_model"] = dict(world)
        result["world_model_prediction_error"] = float(world.get("world_model_prediction_error", 0.0) or 0.0)
        result["world_model_uncertainty"] = float(world.get("world_model_uncertainty", 1.0) or 1.0)
        result["world_model_shadow_replay_requested"] = bool(world.get("world_model_shadow_replay_requested", False))
    verified = bool(result.get("expectancy_entry_evidence_verified", False))
    opportunity = float(result.get("expectancy_market_opportunity", 0.0) or 0.0)
    result["market_opportunity"] = float(np.clip(opportunity, 0.0, 1.0)) if verified else 0.0
    result["opportunity_cost"] = 0.0
    return result


def _v8_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    _v8_policy_ensure(self)
    obs = np.asarray(observation, dtype=float).reshape(-1)
    if obs.size != int(self.input_size):
        padded = np.zeros(int(self.input_size), dtype=float)
        padded[:min(padded.size, obs.size)] = obs[:min(padded.size, obs.size)]
        obs = padded
    partition = self.partitioned_memory.step(obs)
    world = self.trading_world_model.context(obs)
    self._v8_partition_context = partition
    self._v8_world_context = world
    try:
        trace = _V8_PRIOR_POLICY_ACT(self, obs, goal)
    finally:
        self._v8_partition_context = None
        self._v8_world_context = None
    details = dict(getattr(trace, "action_details", {}) or {})
    details["mechanism_schema_version"] = 8
    details["world_model"] = {
        "schema_version": int(world.get("world_model_schema_version", WORLD_MODEL_SCHEMA_VERSION) or WORLD_MODEL_SCHEMA_VERSION),
        "uncertainty": float(world.get("world_model_uncertainty", 1.0) or 1.0),
        "prediction_error": float(world.get("world_model_prediction_error", 0.0) or 0.0),
        "shadow_replay_requested": bool(world.get("world_model_shadow_replay_requested", False)),
    }
    details["partitioned_memory"] = {
        "market_norm": float(partition.get("market_memory_norm", 0.0) or 0.0),
        "position_norm": float(partition.get("position_lifecycle_memory_norm", 0.0) or 0.0),
        "execution_norm": float(partition.get("execution_memory_norm", 0.0) or 0.0),
    }
    curiosity = dict(getattr(self, "_v8_curiosity_diagnostic", {}) or {})
    details["curiosity"] = curiosity
    details["curiosity_live_applied"] = False
    details["curiosity_shadow_replay_requested"] = bool(
        curiosity.get("shadow_replay_requested", False)
        or world.get("world_model_shadow_replay_requested", False)
    )
    return replace(trace, action_details=details)


def _v8_policy_observe_transition(
    self: SelfGrowingPolicy,
    trace: StepTrace,
    next_obs: np.ndarray,
    reward: float,
    cost: float,
    done: bool,
    gamma: float = 0.95,
    record_experience: bool = True,
) -> dict[str, float]:
    _v8_policy_ensure(self)
    signal = _V8_PRIOR_POLICY_OBSERVE(
        self, trace, next_obs, reward, cost, done, gamma=gamma, record_experience=record_experience,
    )
    try:
        action = np.asarray(getattr(trace, "selected_action", np.zeros(self.output_size)), dtype=float)
        execution_details = dict(getattr(trace, "execution_details", {}) or {})
        lifecycle_tags = tuple(getattr(trace, "lifecycle_tags", ()) or ())
        raw_observation = np.asarray(
            getattr(trace, "raw_obs", getattr(self, "_last_observation", np.zeros(self.input_size))),
            dtype=float,
        )
        # Execution heads must learn only from broker/lifecycle evidence.  Market
        # horizons remain causal and may be updated from the observed bar path.
        execution_feedback_eligible = bool(execution_details) and bool(
            execution_details.get("execution_confirmed", False)
            or execution_details.get("position_final_close", False)
            or execution_details.get(
                "confirmed",
                execution_details.get(
                    "filled",
                    execution_details.get("accepted", execution_details.get("placed", False)),
                ),
            )
        )
        self._v8_last_world_update = self.trading_world_model.observe_transition(
            raw_observation,
            action,
            np.asarray(next_obs, dtype=float),
            reward=float(reward),
            cost=float(cost),
            execution_details=execution_details,
            lifecycle_tags=lifecycle_tags,
            execution_feedback_eligible=execution_feedback_eligible,
        )
        if bool(done):
            # No future transition in this episode may satisfy delayed labels.
            self.trading_world_model.pending = []
    except Exception as exc:
        # Do not break the trading policy if an optional diagnostic model cannot
        # update, but retain a finite diagnostic that makes the fault visible.
        self._v8_last_world_update = {"updated": False, "error": type(exc).__name__}
    return signal


def _v8_mechanism_summary(self: SelfGrowingPolicy) -> dict[str, object]:
    _v8_policy_ensure(self)
    return {
        "schema_version": 8,
        "live_curiosity_mode": str(self.live_curiosity_mode),
        "world_model": self.trading_world_model.summary(),
        "partitioned_memory": self.partitioned_memory.summary(),
    }


SelfGrowingPolicy.ensure_compatible = _v8_policy_ensure
SelfGrowingPolicy.copy = _v8_policy_copy
SelfGrowingPolicy.reset_runtime_state = _v8_policy_reset_runtime_state
SelfGrowingPolicy.configure_learning_integrity = _v8_policy_configure_learning_integrity
SelfGrowingPolicy._query_memory = _v8_policy_query_memory
SelfGrowingPolicy.act = _v8_policy_act
SelfGrowingPolicy.observe_transition = _v8_policy_observe_transition
SelfGrowingPolicy.mechanism_summary = _v8_mechanism_summary

# ---------------------------------------------------------------------------
# V9 learner recovery: action-local counterfactual evidence, replacement-only
# reranking, and bridge-visible shadow learning.
#
# The older stack contains several safety layers around M1.  V9 does not
# subtract a failed action from a global entry/HOLD head.  It instead learns
# BUY and SELL counterfactuals at every request, including long HOLD periods,
# and can replace a chosen action only when an already-legal candidate has a
# positive, context-local lower confidence bound.
# ---------------------------------------------------------------------------
_V9_PRIOR_POLICY_ENSURE = SelfGrowingPolicy.ensure_compatible
_V9_PRIOR_POLICY_COPY = SelfGrowingPolicy.copy
_V9_PRIOR_POLICY_RESET_RUNTIME = SelfGrowingPolicy.reset_runtime_state
_V9_PRIOR_POLICY_ACT = SelfGrowingPolicy.act
_V9_PRIOR_POLICY_OBSERVE = SelfGrowingPolicy.observe_transition
_V9_PRIOR_POLICY_LIFECYCLE = SelfGrowingPolicy.apply_execution_lifecycle_credit
_V9_PRIOR_POLICY_CONFIGURE_INTEGRITY = SelfGrowingPolicy.configure_learning_integrity
_V9_PRIOR_MECHANISM_SUMMARY = getattr(SelfGrowingPolicy, "mechanism_summary", None)


def _v9_action_name(action: int) -> str:
    names = (
        "HOLD", "MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT",
        "BUY_STOP", "SELL_STOP", "CANCEL_ORDERS", "CLOSE_WORST",
        "CLOSE_BEST", "CLOSE_ALL", "MANAGE_POSITION",
    )
    return names[int(np.clip(int(action), 0, len(names) - 1))]


def _v9_context_size(trace: StepTrace | None = None) -> int:
    if isinstance(trace, StepTrace):
        vector = np.asarray(getattr(trace, "canonical_context_key", np.zeros(0)), dtype=float).reshape(-1)
        if vector.size:
            return max(8, int(vector.size))
    return int(MEMORY_KEY_SIZE)


def _v9_policy_ensure(self: SelfGrowingPolicy, target_input_size: int | None = None) -> None:
    _V9_PRIOR_POLICY_ENSURE(self, target_input_size=target_input_size)
    learner = getattr(self, "action_recovery_learner", None)
    if not isinstance(learner, ActionRecoveryLearner):
        learner = ActionRecoveryLearner(context_size=int(MEMORY_KEY_SIZE))
        self.action_recovery_learner = learner
    learner.ensure_compatible(context_size=int(MEMORY_KEY_SIZE))
    self.action_recovery_enabled = bool(getattr(self, "action_recovery_enabled", True))
    self.action_recovery_learning_enabled = bool(getattr(self, "action_recovery_learning_enabled", True))
    self.artifact_version = max(int(getattr(self, "artifact_version", 0) or 0), 11)


def _v9_policy_configure_integrity(self: SelfGrowingPolicy, settings: object | None = None) -> None:
    """Apply V9 recovery thresholds from strict ``learning`` config."""
    _V9_PRIOR_POLICY_CONFIGURE_INTEGRITY(self, settings)
    _v9_policy_ensure(self)
    learner = self.action_recovery_learner
    self.action_recovery_enabled = bool(getattr(settings, "action_recovery_enabled", True))
    self.action_recovery_learning_enabled = bool(
        getattr(settings, "action_recovery_learning_enabled", True)
    )
    learner.min_replacement_support = float(getattr(settings, "action_recovery_min_support", learner.min_replacement_support) or learner.min_replacement_support)
    learner.min_replacement_lcb = float(getattr(settings, "action_recovery_min_lcb", learner.min_replacement_lcb) or learner.min_replacement_lcb)
    learner.merge_similarity = float(getattr(settings, "action_recovery_merge_similarity", learner.merge_similarity) or learner.merge_similarity)
    learner.shadow_weight = float(getattr(settings, "action_recovery_shadow_weight", learner.shadow_weight) or learner.shadow_weight)
    learner.max_cells_per_action = int(getattr(settings, "action_recovery_max_cells_per_action", learner.max_cells_per_action) or learner.max_cells_per_action)
    learner.negative_veto_enabled = bool(getattr(settings, "action_recovery_negative_veto_enabled", learner.negative_veto_enabled))
    learner.negative_veto_min_real_support = float(
        getattr(settings, "action_recovery_negative_veto_min_real_support", learner.negative_veto_min_real_support)
        or learner.negative_veto_min_real_support
    )
    raw_negative_veto_lcb = getattr(settings, "action_recovery_negative_veto_lcb", learner.negative_veto_lcb)
    if raw_negative_veto_lcb is not None:
        learner.negative_veto_lcb = float(raw_negative_veto_lcb)
        learner.negative_lcb_threshold = float(raw_negative_veto_lcb)
    learner.ensure_compatible(context_size=int(MEMORY_KEY_SIZE))


def _v9_policy_copy(self: SelfGrowingPolicy) -> SelfGrowingPolicy:
    _v9_policy_ensure(self)
    copied = _V9_PRIOR_POLICY_COPY(self)
    copied.action_recovery_learner = self.action_recovery_learner.copy()
    copied.action_recovery_enabled = bool(getattr(self, "action_recovery_enabled", True))
    copied.action_recovery_learning_enabled = bool(
        getattr(self, "action_recovery_learning_enabled", True)
    )
    copied.artifact_version = max(int(getattr(copied, "artifact_version", 0) or 0), 11)
    return copied


def _v9_policy_reset_runtime_state(self: SelfGrowingPolicy) -> None:
    _V9_PRIOR_POLICY_RESET_RUNTIME(self)
    _v9_policy_ensure(self)
    # Evidence persists across a continuous-session restart.  Only duplicate
    # guards are runtime state; resetting them cannot create a global HOLD bias.
    self.action_recovery_learner.reset_runtime_state()


def _v9_allowed_actions(trace: StepTrace) -> set[int]:
    mask = tuple(bool(item) for item in tuple(getattr(trace, "allowed_action_mask", ()) or ()))
    if mask:
        return {index for index, allowed in enumerate(mask) if allowed}
    candidates = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
    if candidates.ndim == 2 and candidates.size:
        return {primary_action(row) for row in candidates}
    return {0, 1, 2}


def _v9_replacement_trace(trace: StepTrace, recovery: dict[str, object]) -> StepTrace:
    """Apply an evidence-backed replacement without modifying base logits."""
    if bool(recovery.get("negative_veto_to_hold", False)):
        details = dict(getattr(trace, "action_details", {}) or {})
        details.update({
            "learner_recovery": dict(recovery),
            "hold_reason": str(recovery.get("hold_reason", "HOLD_LOCAL_REAL_NEGATIVE_VETO")),
            "learner_recovery_negative_veto_applied": True,
            "learner_recovery_negative_veto_from": int(recovery.get("base_action", 0) or 0),
            "learner_recovery_selector_mode": "local_real_negative_hold_veto",
            "action_name": "HOLD",
            "policy_output_action_index": 0,
        })
        candidates = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
        scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
        hold_indices = (
            [idx for idx, row in enumerate(candidates) if primary_action(row) == 0]
            if candidates.ndim == 2 else []
        )
        if hold_indices:
            hold_index = max(
                hold_indices,
                key=lambda idx: float(scores[idx]) if idx < scores.size else -1e9,
            )
            output = np.asarray(candidates[hold_index], dtype=float).reshape(-1).copy()
            selected_idx = int(hold_index)
        else:
            output = np.asarray(getattr(trace, "selected_action", np.zeros(0)), dtype=float).reshape(-1).copy()
            if output.size:
                output[:min(ACTION_COUNT, output.size)] = -1.0
                output[0] = 1.0
            selected_idx = int(getattr(trace, "selected_idx", 0) or 0)
        details["final_action_output"] = output.astype(float).tolist()
        return replace(trace, selected_idx=selected_idx, selected_action=output, action_details=details)

    index = recovery.get("replacement_candidate_index")
    if not bool(recovery.get("replacement_applied", False)) or index is None:
        details = dict(getattr(trace, "action_details", {}) or {})
        details["learner_recovery"] = dict(recovery)
        details["hold_reason"] = str(recovery.get("hold_reason", ""))
        penalty = float(np.clip(float(recovery.get("soft_action_penalty", 0.0) or 0.0), 0.0, 0.75))
        candidates = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
        scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
        base_action = int(recovery.get("base_action", primary_action(getattr(trace, "selected_action", np.zeros(0)))) or 0)
        if penalty > 0.0 and candidates.ndim == 2 and candidates.shape[0] > 0 and scores.size == candidates.shape[0]:
            revised = scores.copy()
            primary = [primary_action(row) for row in candidates]
            for candidate_index, candidate_action in enumerate(primary):
                if int(candidate_action) == base_action:
                    revised[candidate_index] -= penalty
            selected_index = int(np.argmax(revised))
            if 0 <= selected_index < candidates.shape[0] and selected_index != int(getattr(trace, "selected_idx", 0) or 0):
                output = np.asarray(candidates[selected_index], dtype=float).reshape(-1)
                details.update({
                    "learner_recovery_soft_penalty_applied": True,
                    "learner_recovery_soft_penalty": penalty,
                    "learner_recovery_soft_from": base_action,
                    "learner_recovery_soft_to": int(primary_action(output)),
                    "learner_recovery_selector_mode": "soft_local_action_penalty",
                })
                return replace(
                    trace,
                    selected_idx=selected_index,
                    selected_action=output,
                    candidate_scores=revised,
                    action_details=details,
                )
        details["learner_recovery_soft_penalty_applied"] = False
        details["learner_recovery_soft_penalty"] = penalty
        return replace(trace, action_details=details)
    candidates = np.asarray(getattr(trace, "candidate_actions", np.zeros((0, 0))), dtype=float)
    scores = np.asarray(getattr(trace, "candidate_scores", np.zeros(0)), dtype=float).reshape(-1)
    candidate_index = int(index)
    if candidates.ndim != 2 or not (0 <= candidate_index < candidates.shape[0]):
        details = dict(getattr(trace, "action_details", {}) or {})
        recovery = dict(recovery)
        recovery["replacement_applied"] = False
        recovery["replacement_rejected"] = "candidate_missing"
        details["learner_recovery"] = recovery
        details["hold_reason"] = str(recovery.get("hold_reason", ""))
        return replace(trace, action_details=details)
    output = np.asarray(candidates[candidate_index], dtype=float).reshape(-1)
    revised_scores = scores.copy()
    if revised_scores.size != candidates.shape[0]:
        revised_scores = np.zeros(candidates.shape[0], dtype=float)
    # Do not fake a learned logit or penalise the original candidate.  The tiny
    # deterministic tie-break simply makes the selected candidate explicit in
    # downstream trace readers while retaining the original ranking evidence.
    revised_scores[candidate_index] = max(float(revised_scores[candidate_index]), float(np.max(revised_scores)) + 1e-9)
    shifted = revised_scores - float(np.max(revised_scores))
    probabilities = np.exp(np.clip(shifted, -40.0, 0.0))
    probabilities /= max(float(np.sum(probabilities)), 1e-12)
    action = primary_action(output)
    details = dict(getattr(trace, "action_details", {}) or {})
    details.update({
        "learner_recovery": dict(recovery),
        "hold_reason": str(recovery.get("hold_reason", "")),
        "action_name": _v9_action_name(action),
        "policy_output_action_index": int(action),
        "final_action_output": output.astype(float).tolist(),
        "learner_recovery_replacement_from": int(recovery.get("base_action", 0) or 0),
        "learner_recovery_replacement_to": int(action),
        "learner_recovery_original_candidate_scores": [float(item) for item in scores.tolist()],
        "learner_recovery_selector_mode": "positive_local_lcb_replacement",
    })
    return replace(
        trace,
        selected_idx=int(candidate_index),
        selected_action=output.copy(),
        candidate_scores=revised_scores,
        selected_action_log_prob=float(np.log(max(float(probabilities[candidate_index]), 1e-12))),
        critic_value=float(revised_scores[candidate_index]),
        confidence=float(probabilities[candidate_index]),
        action_details=details,
    )


def _v9_policy_act(self: SelfGrowingPolicy, observation: np.ndarray, goal: np.ndarray | None = None) -> StepTrace:
    _v9_policy_ensure(self)
    # V8/V19 publish an intermediate decision before V9 has attached the
    # replacement/hypothesis audit.  Mute that intermediate write and publish
    # exactly one final immutable decision event below.
    writer = _live_observatory_writer(self)
    writer_was_enabled = bool(writer.enabled) if writer is not None else False
    if writer is not None and writer_was_enabled:
        writer.enabled = False
    try:
        trace = _V9_PRIOR_POLICY_ACT(self, observation, goal)
    finally:
        if writer is not None:
            writer.enabled = writer_was_enabled
    allowed = _v9_allowed_actions(trace)
    if bool(getattr(self, "action_recovery_enabled", True)):
        recovery = self.action_recovery_learner.rerank(trace, allowed_actions=allowed)
    else:
        recovery = {
            "learner_recovery_schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
            "replacement_applied": False,
            "hold_reason": "RECOVERY_DISABLED",
            "shadow_learning_active": bool(
                getattr(self, "action_recovery_learning_enabled", True)
            ),
            "learner_recovery_learning_enabled": bool(
                getattr(self, "action_recovery_learning_enabled", True)
            ),
            "action_values": {},
        }
    trace = _v9_replacement_trace(trace, recovery)
    final_output = np.asarray(getattr(trace, "selected_action", np.zeros(self.output_size)), dtype=float).reshape(-1)
    details = dict(getattr(trace, "action_details", {}) or {})
    base_action = int(recovery.get("base_action", 0) or 0)
    action_values = recovery.get("action_values", {})
    base_value = {}
    if isinstance(action_values, dict):
        candidate = action_values.get(str(base_action), action_values.get(base_action, {}))
        if isinstance(candidate, dict):
            base_value = candidate
    details.update({
        "learner_recovery_schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
        "learner_recovery_shadow_only": True,
        "learner_recovery_learning_enabled": bool(
            getattr(self, "action_recovery_learning_enabled", True)
        ),
        "shadow_learning_active": bool(recovery.get("shadow_learning_active", False)),
        "learner_recovery_global_hold_penalty": 0.0,
        "learner_recovery_negative_action_suppression": 1.0 if bool(recovery.get("negative_veto_to_hold", False)) else 0.0,
        "learner_recovery_base_action": base_action,
        "learner_recovery_base_action_local_negative": bool(recovery.get("base_action_local_negative", False)),
        "learner_recovery_base_available": float(base_value.get("available", 0.0) or 0.0),
        "learner_recovery_base_similarity": float(base_value.get("similarity", 0.0) or 0.0),
        "learner_recovery_base_support": float(base_value.get("support", 0.0) or 0.0),
        "learner_recovery_base_real_support": float(base_value.get("real_support", 0.0) or 0.0),
        "learner_recovery_base_shadow_support": float(base_value.get("shadow_support", 0.0) or 0.0),
        "learner_recovery_base_mean_net_r": float(base_value.get("mean_net_r", 0.0) or 0.0),
        "learner_recovery_base_lcb": float(base_value.get("lcb", 0.0) or 0.0),
        "learner_recovery_base_confidence": float(base_value.get("confidence", 0.0) or 0.0),
        "learner_recovery_negative_support_threshold": float(recovery.get("negative_support_threshold", 0.0) or 0.0),
        "learner_recovery_negative_veto_to_hold": bool(recovery.get("negative_veto_to_hold", False)),
        "learner_recovery_allowed_actions": sorted(int(item) for item in allowed),
        "learner_recovery_decision_count": int(self.action_recovery_learner.decision_count),
        "policy_output_action_index": int(np.argmax(final_output[:min(ACTION_COUNT, final_output.size)])) if final_output.size else 0,
        "final_action_output": final_output.astype(float).tolist(),
    })
    trace = replace(trace, action_details=details)
    if writer is not None and writer_was_enabled:
        writer.decision(self, trace)
    return trace


def _v9_shadow_transition(
    self: SelfGrowingPolicy,
    trace: StepTrace,
    next_obs: np.ndarray,
    *,
    sequence: int = 0,
    source: str = "mt5_bridge",
) -> dict[str, object]:
    """Learn market-direction counterfactuals without economic/action credit."""
    _v9_policy_ensure(self)
    if bool(getattr(self, "action_recovery_learning_enabled", True)):
        result = self.action_recovery_learner.observe_shadow_transition(trace, next_obs, sequence=int(sequence))
    else:
        result = {"updated": False, "reason": "recovery_disabled"}
    world_update: dict[str, object] = {}
    if bool(result.get("updated", False)):
        try:
            action = np.asarray(getattr(trace, "selected_action", np.zeros(self.output_size)), dtype=float)
            world_update = self.trading_world_model.observe_transition(
                np.asarray(getattr(trace, "raw_obs", np.zeros(self.input_size)), dtype=float),
                action,
                np.asarray(next_obs, dtype=float),
                reward=0.0,
                cost=0.0,
                execution_details=None,
                lifecycle_tags=(),
                execution_feedback_eligible=False,
            )
        except Exception as exc:
            world_update = {"updated": False, "error": type(exc).__name__}
    payload = {
        "schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
        "source": str(source),
        "sequence": int(sequence),
        "trace_id": str(getattr(trace, "trace_id", "")),
        "shadow": dict(result),
        "world_market_update": dict(world_update),
    }
    self._v9_last_shadow_update = payload
    writer = _live_observatory_writer(self)
    if writer is not None:
        writer.emit("shadow_feedback", payload)
    return payload


def _v9_policy_observe_transition(
    self: SelfGrowingPolicy,
    trace: StepTrace,
    next_obs: np.ndarray,
    reward: float,
    cost: float,
    done: bool,
    gamma: float = 0.95,
    record_experience: bool = True,
) -> dict[str, float]:
    _v9_policy_ensure(self)
    signal = _V9_PRIOR_POLICY_OBSERVE(
        self, trace, next_obs, reward, cost, done, gamma=gamma, record_experience=record_experience,
    )
    # Simulator/non-continuous paths use the same shadow learner.  Continuous
    # MT5 paths call ``observe_shadow_transition`` directly so no base learner
    # side effect is invoked merely because the account remained flat.
    try:
        self._v9_last_shadow_update = (
            self.action_recovery_learner.observe_shadow_transition(
                trace, next_obs, sequence=int(getattr(self, "_step_count", 0) or 0),
            )
            if bool(getattr(self, "action_recovery_learning_enabled", True))
            else {"updated": False, "reason": "recovery_disabled"}
        )
    except Exception as exc:
        self._v9_last_shadow_update = {"updated": False, "error": type(exc).__name__}
    execution_details = dict(getattr(trace, "execution_details", {}) or {})
    lifecycle_feedback = bool(
        execution_details.get("closed", False)
        and (
            execution_details.get("action_feedback_eligible", False)
            or execution_details.get("memory_eligible", False)
            or execution_details.get("outcome_eligible", False)
        )
    )
    if lifecycle_feedback and bool(getattr(self, "action_recovery_learning_enabled", True)):
        try:
            profit = float(execution_details.get("profit", 0.0) or 0.0)
        except (TypeError, ValueError):
            profit = 0.0
        try:
            commission = float(execution_details.get("commission", 0.0) or 0.0)
        except (TypeError, ValueError):
            commission = 0.0
        try:
            swap = float(execution_details.get("swap", 0.0) or 0.0)
        except (TypeError, ValueError):
            swap = 0.0
        try:
            initial_risk = abs(float(execution_details.get("initial_risk", 0.0) or 0.0))
        except (TypeError, ValueError):
            initial_risk = 0.0
        if initial_risk > 1e-12:
            realised_r = float((profit + commission + swap) / initial_risk)
        else:
            # MT5 online reward is 0.1 * realised-R when risk is available.
            # This fallback keeps older journal/bridge events directionally
            # useful without inventing a stronger signal than the learner saw.
            realised_r = float(reward) / 0.1 if abs(float(reward)) > 1e-12 else 0.0
        close_reason = str(execution_details.get("close_reason", ""))
        signature = "|".join([
            "v9_observe_close",
            str(getattr(trace, "trace_id", "")),
            str(execution_details.get("ack_sequence", "")),
            str(execution_details.get("deal_ticket", "")),
            str(execution_details.get("position_ticket", "")),
            str(close_reason),
            str(round(float(realised_r), 8)),
        ])
        try:
            recovery = self.action_recovery_learner.observe_lifecycle(
                trace,
                float(realised_r),
                close_reason=close_reason,
                sequence=int(getattr(self, "_step_count", 0) or 0),
                signature=signature,
                lifecycle_identity=str(
                    execution_details.get("position_ticket", "")
                    or execution_details.get("deal_ticket", "")
                    or execution_details.get("order_ticket", "")
                    or ""
                ),
            )
            self._v9_last_lifecycle_recovery = dict(recovery)
            self.growth_events.append({
                "round": int(getattr(self, "revision", 0)),
                "type": "learner_recovery_lifecycle_evidence",
                "trace_id": str(getattr(trace, "trace_id", "")),
                "realized_r": float(realised_r),
                "close_reason": close_reason,
                "recovery": dict(recovery),
                "source": "observe_transition_close",
            })
            writer = _live_observatory_writer(self)
            if writer is not None:
                writer.emit("learner_recovery_lifecycle", {
                    "trace_id": str(getattr(trace, "trace_id", "")),
                    "realized_r": float(realised_r),
                    "close_reason": close_reason,
                    "recovery": dict(recovery),
                    "source": "observe_transition_close",
                })
        except Exception as exc:
            self._v9_last_lifecycle_recovery = {"updated": False, "error": type(exc).__name__}
    return signal


def _v9_policy_lifecycle_credit(
    self: SelfGrowingPolicy,
    trace_id: str,
    realized_r: float,
    *,
    close_reason: str = "",
    duration_bars: float | None = None,
    min_holding_bars: int = 3,
    lifecycle: object | None = None,
    override_action: int | None = None,
) -> int:
    _v9_policy_ensure(self)
    cache = getattr(self, "_execution_trace_cache", None)
    trace = cache.get(str(trace_id)) if isinstance(cache, dict) else None
    if not isinstance(trace, StepTrace):
        archive = getattr(self, "_causal_trace_archive", None)
        trace = archive.get(str(trace_id)) if isinstance(archive, dict) else None
    credited = _V9_PRIOR_POLICY_LIFECYCLE(
        self,
        trace_id,
        realized_r,
        close_reason=close_reason,
        duration_bars=duration_bars,
        min_holding_bars=min_holding_bars,
        lifecycle=lifecycle,
    )
    if isinstance(trace, StepTrace):
        credited_ids = getattr(self, "_v9_lifecycle_memory_credited", None)
        if not isinstance(credited_ids, set):
            credited_ids = set()
        memory_identity = "|".join([
            "v9_lifecycle_memory",
            str(trace_id),
            str(close_reason),
            str(round(float(realized_r), 8)),
            str(getattr(lifecycle, "position_ticket", "")),
        ])
        if memory_identity not in credited_ids:
            try:
                lifecycle_details = dict(getattr(trace, "execution_details", {}) or {})
                lifecycle_details.update({
                    "bridge_observed": True,
                    "execution_confirmed": True,
                    "outcome_eligible": True,
                    "action_feedback_eligible": True,
                    "memory_eligible": True,
                    "closed": True,
                    "position_final_close": True,
                    "close_reason": str(close_reason),
                    "realized_r": float(realized_r),
                    "source": str(lifecycle_details.get("source", "lifecycle_credit") or "lifecycle_credit"),
                })
                if override_action is not None and int(override_action) > 0:
                    lifecycle_details["requested_action"] = int(override_action)
                if duration_bars is not None:
                    lifecycle_details["duration_bars"] = float(duration_bars)
                    lifecycle_details["age_bars"] = float(duration_bars)
                if lifecycle is not None:
                    for attr in (
                        "position_ticket", "net_pnl", "gross_profit", "commission", "swap",
                        "initial_risk", "mfe_r", "mae_r", "near_tp_ratio",
                    ):
                        value = getattr(lifecycle, attr, None)
                        if value is not None:
                            lifecycle_details[attr] = value
                    tags = tuple(getattr(lifecycle, "lifecycle_tags", ()) or ())
                    if tags:
                        lifecycle_details["lifecycle_tags"] = [str(item) for item in tags]
                lifecycle_trace = replace(
                    trace,
                    execution_details=lifecycle_details,
                    lifecycle_tags=tuple(str(item) for item in lifecycle_details.get("lifecycle_tags", ()) or ()),
                )
                next_obs = np.asarray(getattr(trace, "raw_obs", np.zeros(getattr(self, "input_size", INPUT_COUNT))), dtype=float)
                if next_obs.size != int(getattr(self, "input_size", INPUT_COUNT)):
                    padded = np.zeros(int(getattr(self, "input_size", INPUT_COUNT)), dtype=float)
                    padded[:min(padded.size, next_obs.size)] = next_obs[:padded.size]
                    next_obs = padded
                memory_signal = {
                    "td_delta": float(realized_r),
                    "prediction_error": 0.0,
                    "reward": float(realized_r),
                    "cost": 0.0,
                    "memory_eligible": 1.0,
                    "execution_confirmed": 1.0,
                    "source": "lifecycle_credit",
                }
                node_id = self.memory_supernet.absorb_transition(
                    lifecycle_trace,
                    next_obs,
                    float(realized_r),
                    0.0,
                    memory_signal,
                )
                if node_id:
                    self.growth_events.append({
                        "round": int(getattr(self, "revision", 0)),
                        "type": "lifecycle_memory_absorbed",
                        "trace_id": str(trace_id),
                        "memory_node_id": str(node_id),
                        "realized_r": float(realized_r),
                        "close_reason": str(close_reason),
                    })
                    credited += 1
                credited_ids.add(memory_identity)
                self._v9_lifecycle_memory_credited = credited_ids
            except Exception as exc:
                self.growth_events.append({
                    "round": int(getattr(self, "revision", 0)),
                    "type": "lifecycle_memory_absorb_error",
                    "trace_id": str(trace_id),
                    "error": type(exc).__name__,
                })
    if isinstance(trace, StepTrace) and bool(getattr(self, "action_recovery_learning_enabled", True)):
        signature = "|".join([
            "v9_lifecycle", str(trace_id), str(close_reason),
            str(round(float(realized_r), 8)), str(getattr(lifecycle, "position_ticket", "")),
        ])
        recovery = self.action_recovery_learner.observe_lifecycle(
            trace,
            float(realized_r),
            close_reason=str(close_reason),
            sequence=int(getattr(self, "_step_count", 0) or 0),
            signature=signature,
            override_action=override_action,
            lifecycle_identity=str(
                getattr(lifecycle, "position_ticket", "")
                or getattr(lifecycle, "deal_ticket", "")
                or getattr(lifecycle, "order_ticket", "")
                or ""
            ),
        )
        self._v9_last_lifecycle_recovery = dict(recovery)
        self.growth_events.append({
            "round": int(getattr(self, "revision", 0)),
            "type": "learner_recovery_lifecycle_evidence",
            "trace_id": str(trace_id),
            "realized_r": float(realized_r),
            "close_reason": str(close_reason),
            "recovery": dict(recovery),
        })
        writer = _live_observatory_writer(self)
        if writer is not None:
            writer.emit("learner_recovery_lifecycle", {
                "trace_id": str(trace_id),
                "realized_r": float(realized_r),
                "close_reason": str(close_reason),
                "recovery": dict(recovery),
            })
        credited += int(bool(recovery.get("updated", False)))
    return int(credited)


def _v9_emit_bridge_heartbeat(
    self: SelfGrowingPolicy,
    *,
    stage: str,
    sequence: int,
    request_event_id: str = "",
    trace_id: str = "",
    action: int | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Emit a bridge heartbeat even when the policy output is HOLD/no-op."""
    writer = _live_observatory_writer(self)
    if writer is None:
        return
    enriched_metadata = dict(metadata or {})
    if action is not None:
        action_index = int(action)

        def _metadata_action(name: str) -> int:
            value = enriched_metadata.get(name, action_index)
            if value is None or value == "":
                return action_index
            try:
                return int(value)
            except (TypeError, ValueError):
                return action_index

        wire_action = _metadata_action("wire_action")
        policy_action = _metadata_action("policy_action")
        legal_action = _metadata_action("legal_action")
        emitted_action = _metadata_action("emitted_action")
        enriched_metadata.setdefault("wire_action_name", _v9_action_name(wire_action))
        enriched_metadata.setdefault("policy_action_name", _v9_action_name(policy_action))
        enriched_metadata.setdefault("legal_action_name", _v9_action_name(legal_action))
        enriched_metadata.setdefault("emitted_action_name", _v9_action_name(emitted_action))
    payload = {
        "schema_version": LEARNER_RECOVERY_SCHEMA_VERSION,
        "stage": str(stage),
        "sequence": int(sequence),
        "request_event_id": str(request_event_id),
        "trace_id": str(trace_id),
        "action": None if action is None else int(action),
        "metadata": enriched_metadata,
    }
    if action is not None:
        payload["action_name"] = _v9_action_name(int(action))
        payload["execution_action_event"] = True
        payload["topology_summary"] = topology_summary(self)
    writer.emit("bridge_heartbeat", payload)


def _v9_mechanism_summary(self: SelfGrowingPolicy) -> dict[str, object]:
    _v9_policy_ensure(self)
    base: dict[str, object] = {}
    if callable(_V9_PRIOR_MECHANISM_SUMMARY):
        try:
            base = dict(_V9_PRIOR_MECHANISM_SUMMARY(self) or {})
        except Exception:
            base = {}
    # Keep the V8 top-level schema stable for existing dashboards/checkpoints;
    # V9 is an additive learner contract.
    base["learner_recovery_schema_version"] = LEARNER_RECOVERY_SCHEMA_VERSION
    base["learner_recovery_enabled"] = bool(getattr(self, "action_recovery_enabled", True))
    base["learner_recovery_learning_enabled"] = bool(
        getattr(self, "action_recovery_learning_enabled", True)
    )
    base["action_recovery"] = self.action_recovery_learner.summary()
    return base


SelfGrowingPolicy.ensure_compatible = _v9_policy_ensure
SelfGrowingPolicy.configure_learning_integrity = _v9_policy_configure_integrity
SelfGrowingPolicy.copy = _v9_policy_copy
SelfGrowingPolicy.reset_runtime_state = _v9_policy_reset_runtime_state
SelfGrowingPolicy.act = _v9_policy_act
SelfGrowingPolicy.observe_transition = _v9_policy_observe_transition
SelfGrowingPolicy.observe_shadow_transition = _v9_shadow_transition
SelfGrowingPolicy.apply_execution_lifecycle_credit = _v9_policy_lifecycle_credit
SelfGrowingPolicy.emit_bridge_heartbeat = _v9_emit_bridge_heartbeat
SelfGrowingPolicy.mechanism_summary = _v9_mechanism_summary
