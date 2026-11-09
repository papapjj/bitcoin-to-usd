"""Per-expert causal Transformer and soft tensor-rule abstraction bank.

The bank owns no optimizer.  It exposes parameter arrays and ascent
directions to the canonical Direct-RNN AdamW writer.  Runtime inputs contain
only closed M15 rows captured at decision time; future paths are accepted only
as delayed training targets.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
from typing import Iterable, Mapping

import numpy as np

from .auxiliary_objectives import (
    ACTION_AUXILIARY_NAMES,
    AUXILIARY_OUTPUT_COUNT,
    GLOBAL_AUXILIARY_NAMES,
    RETURN_HORIZONS_MINUTES,
    AuxiliaryTargets,
    auxiliary_loss_and_ascent,
)


TENSOR_RULE_SCHEMA_VERSION = 7
TENSOR_RULE_ARTIFACT_VERSION = 19
TENSOR_RULE_AUTHORITIES = frozenset({"telemetry", "shadow", "residual", "primary"})
TENSOR_RULE_SEQUENCE_LENGTH = 32
TENSOR_RULE_INPUT_SIZE = 78
TENSOR_RULE_ENTRY_ACTIONS = 6
# Distinct closed-M15 sequence observations: 8/16/24/32 bars are
# H2/H4/H6/H8.  These are delayed labels and never inference inputs.
TENSOR_RULE_REGIME_HORIZONS = (8, 16, 24, 32)
TENSOR_RULE_AUXILIARY_COUNT = AUXILIARY_OUTPUT_COUNT + len(TENSOR_RULE_REGIME_HORIZONS)
TENSOR_RULE_PARAMETER_PREFIX = "tensor_rule."


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    shifted = np.clip(array - maximum, -60.0, 60.0)
    weights = np.exp(shifted)
    return weights / np.maximum(np.sum(weights, axis=axis, keepdims=True), 1e-12)


def _layer_norm_forward(
    values: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    epsilon: float = 1e-5,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    mean = np.mean(values, axis=-1, keepdims=True)
    centered = values - mean
    variance = np.mean(centered * centered, axis=-1, keepdims=True)
    inverse = 1.0 / np.sqrt(variance + epsilon)
    normalized = centered * inverse
    return normalized * gamma + beta, (normalized, inverse, gamma, centered)


def _layer_norm_backward(
    direction: np.ndarray,
    cache: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized, inverse, gamma, _centered = cache
    width = max(1, int(direction.shape[-1]))
    dnormalized = direction * gamma
    dx = (
        dnormalized
        - np.mean(dnormalized, axis=-1, keepdims=True)
        - normalized * np.mean(dnormalized * normalized, axis=-1, keepdims=True)
    ) * inverse
    dgamma = np.sum(direction * normalized, axis=0)
    dbeta = np.sum(direction, axis=0)
    return dx, dgamma, dbeta


def _gelu(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coefficient = np.sqrt(2.0 / np.pi)
    inner = coefficient * (values + 0.044715 * values ** 3)
    tanh_inner = np.tanh(inner)
    output = 0.5 * values * (1.0 + tanh_inner)
    derivative = 0.5 * (1.0 + tanh_inner) + 0.5 * values * (
        1.0 - tanh_inner * tanh_inner
    ) * coefficient * (1.0 + 3.0 * 0.044715 * values * values)
    return output, derivative


@dataclass
class TensorRuleTargets:
    """Masked delayed targets; future values are never inference inputs."""

    auxiliary_values: np.ndarray = field(
        default_factory=lambda: np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=float)
    )
    auxiliary_mask: np.ndarray = field(
        default_factory=lambda: np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=bool)
    )
    entry_utilities: np.ndarray | None = None
    selected_action: int | None = None
    realized_r: float | None = None
    temporal_fold: int = 0

    @classmethod
    def from_item(cls, item: Mapping[str, object]) -> "TensorRuleTargets":
        values = np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=float)
        mask = np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=bool)
        raw_values = np.asarray(item.get("auxiliary_targets", ()), dtype=float).reshape(-1)
        raw_mask = np.asarray(item.get("auxiliary_target_mask", ()), dtype=bool).reshape(-1)
        count = min(AUXILIARY_OUTPUT_COUNT, raw_values.size, raw_mask.size)
        if count:
            values[:count] = raw_values[:count]
            mask[:count] = raw_mask[:count]
        regime_values = np.asarray(item.get("tensor_rule_regime_targets", ()), dtype=float).reshape(-1)
        regime_mask = np.asarray(item.get("tensor_rule_regime_mask", ()), dtype=bool).reshape(-1)
        regime_count = min(len(TENSOR_RULE_REGIME_HORIZONS), regime_values.size, regime_mask.size)
        if regime_count:
            start = AUXILIARY_OUTPUT_COUNT
            values[start:start + regime_count] = regime_values[:regime_count]
            mask[start:start + regime_count] = regime_mask[:regime_count]
        utilities_raw = np.asarray(item.get("entry_preference_utilities", ()), dtype=float).reshape(-1)
        utilities = utilities_raw.copy() if utilities_raw.size == 7 else None
        trace = item.get("trace")
        details = dict(getattr(trace, "action_details", {}) or {})
        try:
            selected = int(details.get("direct_rnn_rl_policy_action_index", -1))
        except (TypeError, ValueError):
            selected = -1
        realized = None
        if not bool(item.get("entry_preference_only", False)) and 1 <= selected <= 6:
            try:
                realized = float(item.get("outcome", 0.0))
            except (TypeError, ValueError):
                realized = None
        try:
            raw_fold = item.get("tensor_rule_temporal_fold", None)
            if raw_fold is None:
                frame_id = str(getattr(trace, "context_frame_id", "") or getattr(trace, "trace_id", ""))
                fold = int.from_bytes(hashlib.sha256(frame_id.encode("utf-8")).digest()[:2], "little") % 3
            else:
                fold = int(raw_fold) % 3
        except (TypeError, ValueError):
            fold = 0
        return cls(values, mask, utilities, selected, realized, fold)


@dataclass(frozen=True)
class TensorRuleForward:
    schema_version: int
    bank_version: int
    input_fingerprint: str
    valid_length: int
    summary: np.ndarray
    memberships: np.ndarray
    rule_id: int
    unknown: bool
    unknown_reason: str
    action_mean_r: np.ndarray
    action_std_r: np.ndarray
    action_lcb_r: np.ndarray
    auxiliary_logits: np.ndarray
    reconstruction: np.ndarray
    cache: dict[str, object] = field(default_factory=dict, repr=False)
    component_norms: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=float))
    action_latent_norms: np.ndarray = field(
        default_factory=lambda: np.zeros(TENSOR_RULE_ENTRY_ACTIONS, dtype=float)
    )

    def payload(self) -> dict[str, object]:
        active_memberships = np.asarray(self.memberships, dtype=float)
        positive = active_memberships[active_memberships > 1e-12]
        entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
        return {
            "schema_version": int(self.schema_version),
            "bank_version": int(self.bank_version),
            "input_fingerprint": str(self.input_fingerprint),
            "valid_length": int(self.valid_length),
            "memberships": [float(value) for value in active_memberships.tolist()],
            "membership_entropy": entropy,
            "rule_id": int(self.rule_id),
            "unknown": bool(self.unknown),
            "unknown_reason": str(self.unknown_reason),
            "action_mean_r": [0.0, *[float(value) for value in self.action_mean_r.tolist()]],
            "action_std_r": [0.0, *[float(value) for value in self.action_std_r.tolist()]],
            "action_lcb_r": [0.0, *[float(value) for value in self.action_lcb_r.tolist()]],
            "component_norms": [float(value) for value in self.component_norms.tolist()],
            "action_latent_norms": [0.0, *[float(value) for value in self.action_latent_norms.tolist()]],
        }


@dataclass
class TensorRuleBankState:
    """Independent fixed-capacity abstraction bank for exactly one expert."""

    schema_version: int = TENSOR_RULE_SCHEMA_VERSION
    bank_version: int = 1
    sequence_length: int = TENSOR_RULE_SEQUENCE_LENGTH
    input_size: int = TENSOR_RULE_INPUT_SIZE
    hidden_size: int = 64
    encoder_layers: int = 2
    attention_heads: int = 4
    ffn_size: int = 128
    max_rules: int = 16
    initial_rules: int = 2
    min_valid_length: int = 16
    unknown_membership: float = 0.55
    membership_temperature: float = 0.10
    motif_distance_weight: float = 0.75
    utility_assignment_temperature: float = 0.15
    authority: str = "telemetry"
    actor_residual_clip: float = 0.25
    trade_margin_r: float = 0.10
    effective_batch_size: int = 64
    ready_gradient_norm_budget: float = 0.50
    component_size: int = 16
    action_latent_size: int = 32
    counterfactual_encoder_credit_scale: float = 0.25
    lifecycle_encoder_credit_scale: float = 0.0
    utility_contrastive_margin: float = 0.20
    input_schema_fingerprint: str = "timing_m15_32x78_v1"
    seed_fingerprint: str = ""
    input_w: np.ndarray | None = field(default=None, repr=False)
    input_b: np.ndarray | None = field(default=None, repr=False)
    positional: np.ndarray | None = field(default=None, repr=False)
    blocks: list[dict[str, np.ndarray]] = field(default_factory=list, repr=False)
    prototype: np.ndarray | None = field(default=None, repr=False)
    time_importance_logits: np.ndarray | None = field(default=None, repr=False)
    feature_log_scale: np.ndarray | None = field(default=None, repr=False)
    rule_action_mean: np.ndarray | None = field(default=None, repr=False)
    rule_action_log_var: np.ndarray | None = field(default=None, repr=False)
    auxiliary_w: np.ndarray | None = field(default=None, repr=False)
    auxiliary_b: np.ndarray | None = field(default=None, repr=False)
    reconstruction_b: np.ndarray | None = field(default=None, repr=False)
    structure_w: np.ndarray | None = field(default=None, repr=False)
    structure_b: np.ndarray | None = field(default=None, repr=False)
    timing_w: np.ndarray | None = field(default=None, repr=False)
    timing_b: np.ndarray | None = field(default=None, repr=False)
    regime_w: np.ndarray | None = field(default=None, repr=False)
    regime_b: np.ndarray | None = field(default=None, repr=False)
    risk_w: np.ndarray | None = field(default=None, repr=False)
    risk_b: np.ndarray | None = field(default=None, repr=False)
    action_embedding: np.ndarray | None = field(default=None, repr=False)
    action_adapter_w: np.ndarray | None = field(default=None, repr=False)
    action_adapter_b: np.ndarray | None = field(default=None, repr=False)
    action_value_w: np.ndarray | None = field(default=None, repr=False)
    action_value_b: np.ndarray | None = field(default=None, repr=False)
    utility_prototype: np.ndarray | None = field(default=None, repr=False)
    active_mask: np.ndarray | None = field(default=None, repr=False)
    rule_support: np.ndarray | None = field(default=None, repr=False)
    rule_action_support: np.ndarray | None = field(default=None, repr=False)
    temporal_fold_support: np.ndarray | None = field(default=None, repr=False)
    fold_action_return_sum: np.ndarray | None = field(default=None, repr=False)
    fold_action_return_sq_sum: np.ndarray | None = field(default=None, repr=False)
    fold_action_return_support: np.ndarray | None = field(default=None, repr=False)
    calibration_error: np.ndarray | None = field(default=None, repr=False)
    low_support_rounds: np.ndarray | None = field(default=None, repr=False)
    gradient_accumulator: dict[str, np.ndarray] = field(default_factory=dict, repr=False)
    gradient_sample_count: int = 0
    last_ready_gradient_raw_norm: float = 0.0
    last_ready_gradient_clip_factor: float = 1.0
    pending_regime: list[dict[str, object]] = field(default_factory=list, repr=False)
    ready_regime: list[dict[str, object]] = field(default_factory=list, repr=False)
    update_count: int = 0
    calibration_rounds: int = 0
    unknown_count: int = 0
    decision_count: int = 0
    last_topology_proposal: dict[str, object] = field(default_factory=dict, repr=False)

    @classmethod
    def seed(
        cls,
        *,
        seed: int,
        sequence_length: int = 32,
        input_size: int = 78,
        hidden_size: int = 64,
        encoder_layers: int = 2,
        attention_heads: int = 4,
        ffn_size: int = 128,
        max_rules: int = 16,
        initial_rules: int = 2,
        min_valid_length: int = 16,
        unknown_membership: float = 0.55,
        membership_temperature: float = 0.10,
        motif_distance_weight: float = 0.75,
        utility_assignment_temperature: float = 0.15,
        authority: str = "telemetry",
        actor_residual_clip: float = 0.25,
        effective_batch_size: int = 64,
        ready_gradient_norm_budget: float = 0.50,
        component_size: int = 16,
        action_latent_size: int = 32,
        counterfactual_encoder_credit_scale: float = 0.25,
        lifecycle_encoder_credit_scale: float = 0.0,
        utility_contrastive_margin: float = 0.20,
        seed_fingerprint: str = "",
    ) -> "TensorRuleBankState":
        if hidden_size % attention_heads != 0:
            raise ValueError("tensor rule hidden_size must be divisible by attention_heads")
        rng = np.random.default_rng(int(seed))
        bank = cls(
            sequence_length=int(sequence_length), input_size=int(input_size),
            hidden_size=int(hidden_size), encoder_layers=int(encoder_layers),
            attention_heads=int(attention_heads), ffn_size=int(ffn_size),
            max_rules=int(max_rules), initial_rules=int(initial_rules),
            min_valid_length=int(min_valid_length), unknown_membership=float(unknown_membership),
            membership_temperature=float(membership_temperature),
            motif_distance_weight=float(motif_distance_weight),
            utility_assignment_temperature=float(utility_assignment_temperature),
            authority=str(authority), actor_residual_clip=float(actor_residual_clip),
            effective_batch_size=int(effective_batch_size),
            ready_gradient_norm_budget=float(ready_gradient_norm_budget),
            component_size=int(component_size), action_latent_size=int(action_latent_size),
            counterfactual_encoder_credit_scale=float(counterfactual_encoder_credit_scale),
            lifecycle_encoder_credit_scale=float(lifecycle_encoder_credit_scale),
            utility_contrastive_margin=float(utility_contrastive_margin),
            seed_fingerprint=str(seed_fingerprint or ""),
        )
        input_scale = 1.0 / np.sqrt(max(1, bank.input_size))
        hidden_scale = 1.0 / np.sqrt(max(1, bank.hidden_size))
        bank.input_w = rng.normal(0.0, input_scale, size=(bank.input_size, bank.hidden_size))
        bank.input_b = np.zeros(bank.hidden_size, dtype=float)
        bank.positional = rng.normal(0.0, 0.01, size=(bank.sequence_length, bank.hidden_size))
        bank.blocks = []
        for _ in range(bank.encoder_layers):
            bank.blocks.append({
                "ln1_gamma": np.ones(bank.hidden_size, dtype=float),
                "ln1_beta": np.zeros(bank.hidden_size, dtype=float),
                "q_w": rng.normal(0.0, hidden_scale, size=(bank.hidden_size, bank.hidden_size)),
                "q_b": np.zeros(bank.hidden_size, dtype=float),
                "k_w": rng.normal(0.0, hidden_scale, size=(bank.hidden_size, bank.hidden_size)),
                "k_b": np.zeros(bank.hidden_size, dtype=float),
                "v_w": rng.normal(0.0, hidden_scale, size=(bank.hidden_size, bank.hidden_size)),
                "v_b": np.zeros(bank.hidden_size, dtype=float),
                "o_w": rng.normal(0.0, hidden_scale, size=(bank.hidden_size, bank.hidden_size)),
                "o_b": np.zeros(bank.hidden_size, dtype=float),
                "ln2_gamma": np.ones(bank.hidden_size, dtype=float),
                "ln2_beta": np.zeros(bank.hidden_size, dtype=float),
                "ff1_w": rng.normal(0.0, hidden_scale, size=(bank.hidden_size, bank.ffn_size)),
                "ff1_b": np.zeros(bank.ffn_size, dtype=float),
                "ff2_w": rng.normal(0.0, 1.0 / np.sqrt(max(1, bank.ffn_size)), size=(bank.ffn_size, bank.hidden_size)),
                "ff2_b": np.zeros(bank.hidden_size, dtype=float),
            })
        bank.prototype = rng.normal(
            0.0, 0.08, size=(bank.max_rules, bank.sequence_length, bank.hidden_size)
        )
        bank.time_importance_logits = np.zeros((bank.max_rules, bank.sequence_length), dtype=float)
        bank.feature_log_scale = np.zeros((bank.max_rules, bank.hidden_size), dtype=float)
        bank.rule_action_mean = np.zeros((bank.max_rules, TENSOR_RULE_ENTRY_ACTIONS), dtype=float)
        bank.rule_action_log_var = np.zeros((bank.max_rules, TENSOR_RULE_ENTRY_ACTIONS), dtype=float)
        bank.auxiliary_w = rng.normal(
            0.0, 0.02, size=(TENSOR_RULE_AUXILIARY_COUNT, 4 * bank.component_size)
        )
        bank.auxiliary_b = np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=float)
        bank.reconstruction_b = np.zeros(bank.input_size, dtype=float)
        component_scale = 1.0 / np.sqrt(max(1, bank.hidden_size))
        action_input_size = 4 * bank.component_size + bank.component_size
        bank.structure_w = rng.normal(0.0, component_scale, size=(bank.hidden_size, bank.component_size))
        bank.structure_b = np.zeros(bank.component_size, dtype=float)
        bank.timing_w = rng.normal(0.0, component_scale, size=(bank.hidden_size, bank.component_size))
        bank.timing_b = np.zeros(bank.component_size, dtype=float)
        bank.regime_w = rng.normal(0.0, component_scale, size=(bank.hidden_size, bank.component_size))
        bank.regime_b = np.zeros(bank.component_size, dtype=float)
        bank.risk_w = rng.normal(0.0, component_scale, size=(bank.hidden_size, bank.component_size))
        bank.risk_b = np.zeros(bank.component_size, dtype=float)
        bank.action_embedding = rng.normal(
            0.0, 0.05, size=(TENSOR_RULE_ENTRY_ACTIONS, bank.component_size)
        )
        bank.action_adapter_w = rng.normal(
            0.0, 1.0 / np.sqrt(action_input_size),
            size=(TENSOR_RULE_ENTRY_ACTIONS, action_input_size, bank.action_latent_size),
        )
        bank.action_adapter_b = np.zeros(
            (TENSOR_RULE_ENTRY_ACTIONS, bank.action_latent_size), dtype=float
        )
        # Zero value rows preserve old rule-posterior output at migration. The
        # first utility batch opens only the relevant action row.
        bank.action_value_w = np.zeros(
            (TENSOR_RULE_ENTRY_ACTIONS, bank.action_latent_size), dtype=float
        )
        bank.action_value_b = np.zeros(TENSOR_RULE_ENTRY_ACTIONS, dtype=float)
        utility = rng.normal(
            0.0, 1.0, size=(bank.max_rules, TENSOR_RULE_ENTRY_ACTIONS + 1)
        )
        utility -= np.mean(utility, axis=1, keepdims=True)
        utility /= np.maximum(np.linalg.norm(utility, axis=1, keepdims=True), 1e-9)
        if bank.max_rules >= 2:
            # Antipodal cold-start anchors remove the symmetric 50/50 fixed
            # point. They are learned tensors, not semantic long/short rules.
            utility[1] = -utility[0]
        bank.utility_prototype = utility
        bank.active_mask = np.zeros(bank.max_rules, dtype=bool)
        bank.active_mask[:max(1, min(bank.initial_rules, bank.max_rules))] = True
        bank.rule_support = np.zeros(bank.max_rules, dtype=float)
        bank.rule_action_support = np.zeros((bank.max_rules, TENSOR_RULE_ENTRY_ACTIONS), dtype=float)
        bank.temporal_fold_support = np.zeros((bank.max_rules, 3), dtype=float)
        evidence_shape = (bank.max_rules, 3, TENSOR_RULE_ENTRY_ACTIONS)
        bank.fold_action_return_sum = np.zeros(evidence_shape, dtype=float)
        bank.fold_action_return_sq_sum = np.zeros(evidence_shape, dtype=float)
        bank.fold_action_return_support = np.zeros(evidence_shape, dtype=float)
        bank.calibration_error = np.full(TENSOR_RULE_ENTRY_ACTIONS, 0.10, dtype=float)
        bank.low_support_rounds = np.zeros(bank.max_rules, dtype=int)
        return bank

    def copy(self) -> "TensorRuleBankState":
        return copy.deepcopy(self)

    def ensure_compatible(self) -> None:
        prior_schema_version = int(getattr(self, "schema_version", 1) or 1)
        self.schema_version = TENSOR_RULE_SCHEMA_VERSION
        self.authority = str(getattr(self, "authority", "telemetry") or "telemetry").lower()
        if self.authority not in TENSOR_RULE_AUTHORITIES:
            self.authority = "telemetry"
        self.sequence_length = max(1, int(getattr(self, "sequence_length", 32) or 32))
        self.input_size = max(1, int(getattr(self, "input_size", 78) or 78))
        self.hidden_size = max(4, int(getattr(self, "hidden_size", 64) or 64))
        self.encoder_layers = max(1, int(getattr(self, "encoder_layers", 2) or 2))
        self.attention_heads = max(1, int(getattr(self, "attention_heads", 4) or 4))
        if self.hidden_size % self.attention_heads:
            raise ValueError("incompatible tensor-rule attention geometry")
        self.ffn_size = max(self.hidden_size, int(getattr(self, "ffn_size", 128) or 128))
        self.max_rules = max(2, int(getattr(self, "max_rules", 16) or 16))
        self.initial_rules = int(np.clip(getattr(self, "initial_rules", 2), 1, self.max_rules))
        self.min_valid_length = int(np.clip(getattr(self, "min_valid_length", 16), 1, self.sequence_length))
        self.unknown_membership = float(np.clip(getattr(self, "unknown_membership", 0.55), 0.0, 1.0))
        self.membership_temperature = float(np.clip(
            getattr(self, "membership_temperature", 0.10), 0.02, 2.0
        ))
        self.motif_distance_weight = float(np.clip(
            getattr(self, "motif_distance_weight", 0.75), 0.0, 1.0
        ))
        self.utility_assignment_temperature = float(np.clip(
            getattr(self, "utility_assignment_temperature", 0.15), 0.02, 2.0
        ))
        self.effective_batch_size = max(1, int(getattr(self, "effective_batch_size", 64) or 64))
        self.ready_gradient_norm_budget = float(np.clip(
            getattr(self, "ready_gradient_norm_budget", 0.50), 0.01, 10.0
        ))
        self.last_ready_gradient_raw_norm = max(
            0.0, float(getattr(self, "last_ready_gradient_raw_norm", 0.0) or 0.0)
        )
        self.last_ready_gradient_clip_factor = float(np.clip(
            getattr(self, "last_ready_gradient_clip_factor", 1.0), 0.0, 1.0
        ))
        self.component_size = max(4, int(getattr(self, "component_size", 16) or 16))
        self.action_latent_size = max(4, int(getattr(self, "action_latent_size", 32) or 32))
        self.counterfactual_encoder_credit_scale = float(np.clip(
            getattr(self, "counterfactual_encoder_credit_scale", 0.25), 0.0, 1.0
        ))
        self.lifecycle_encoder_credit_scale = float(np.clip(
            getattr(self, "lifecycle_encoder_credit_scale", 0.0), 0.0, 1.0
        ))
        self.utility_contrastive_margin = float(np.clip(
            getattr(self, "utility_contrastive_margin", 0.20), -0.5, 0.95
        ))
        expected = {
            "input_w": (self.input_size, self.hidden_size), "input_b": (self.hidden_size,),
            "positional": (self.sequence_length, self.hidden_size),
            "prototype": (self.max_rules, self.sequence_length, self.hidden_size),
            "time_importance_logits": (self.max_rules, self.sequence_length),
            "feature_log_scale": (self.max_rules, self.hidden_size),
            "rule_action_mean": (self.max_rules, TENSOR_RULE_ENTRY_ACTIONS),
            "rule_action_log_var": (self.max_rules, TENSOR_RULE_ENTRY_ACTIONS),
            "auxiliary_w": (TENSOR_RULE_AUXILIARY_COUNT, 4 * self.component_size),
            "auxiliary_b": (TENSOR_RULE_AUXILIARY_COUNT,),
            "reconstruction_b": (self.input_size,),
        }
        # Auxiliary target schema grew in V191. Preserve overlapping rows from
        # older artifacts and deterministically initialize only new horizons.
        if not isinstance(getattr(self, "auxiliary_w", None), np.ndarray) or self.auxiliary_w.shape != expected["auxiliary_w"]:
            old = np.asarray(
                getattr(self, "auxiliary_w", np.zeros((0, 4 * self.component_size))),
                dtype=float,
            )
            rng = np.random.default_rng(int.from_bytes(hashlib.sha256(
                (str(self.seed_fingerprint) + "|aux-v194").encode("utf-8")
            ).digest()[:8], "little"))
            migrated = rng.normal(0.0, 0.02, size=expected["auxiliary_w"])
            # Các schema 5/6/7 từng có 8 hoặc 11 mục tiêu cho mỗi hành động,
            # trong khi schema hiện tại có 17. Không được copy nguyên theo số
            # hàng vì từ hành động thứ hai trở đi stride đã đổi. Bốn hàng regime
            # cũ cũng phải chuyển về cuối thay vì rơi vào vùng timing mới.
            if prior_schema_version >= 5 and old.ndim == 2 and old.shape[1] == migrated.shape[1]:
                global_rows = 25
                new_stride = 17
                old_layout = {77: 8, 95: 11, 131: 17}.get(int(old.shape[0]))
                if old_layout is not None:
                    migrated[:global_rows] = old[:global_rows]
                    for action_offset in range(TENSOR_RULE_ENTRY_ACTIONS):
                        old_start = global_rows + action_offset * old_layout
                        new_start = global_rows + action_offset * new_stride
                        migrated[new_start:new_start + old_layout] = old[old_start:old_start + old_layout]
                    old_regime_start = global_rows + TENSOR_RULE_ENTRY_ACTIONS * old_layout
                    if old.shape[0] >= old_regime_start + len(TENSOR_RULE_REGIME_HORIZONS):
                        migrated[AUXILIARY_OUTPUT_COUNT:TENSOR_RULE_AUXILIARY_COUNT] = old[
                            old_regime_start:old_regime_start + len(TENSOR_RULE_REGIME_HORIZONS)
                        ]
            self.auxiliary_w = migrated
        if not isinstance(getattr(self, "auxiliary_b", None), np.ndarray) or self.auxiliary_b.shape != expected["auxiliary_b"]:
            old = np.asarray(getattr(self, "auxiliary_b", np.zeros(0)), dtype=float).reshape(-1)
            migrated = np.zeros(expected["auxiliary_b"], dtype=float)
            if prior_schema_version >= 5:
                global_rows = 25
                new_stride = 17
                old_layout = {77: 8, 95: 11, 131: 17}.get(int(old.size))
                if old_layout is not None:
                    migrated[:global_rows] = old[:global_rows]
                    for action_offset in range(TENSOR_RULE_ENTRY_ACTIONS):
                        old_start = global_rows + action_offset * old_layout
                        new_start = global_rows + action_offset * new_stride
                        migrated[new_start:new_start + old_layout] = old[old_start:old_start + old_layout]
                    old_regime_start = global_rows + TENSOR_RULE_ENTRY_ACTIONS * old_layout
                    if old.size >= old_regime_start + len(TENSOR_RULE_REGIME_HORIZONS):
                        migrated[AUXILIARY_OUTPUT_COUNT:TENSOR_RULE_AUXILIARY_COUNT] = old[
                            old_regime_start:old_regime_start + len(TENSOR_RULE_REGIME_HORIZONS)
                        ]
            self.auxiliary_b = migrated
        for name, shape in expected.items():
            value = getattr(self, name, None)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                raise ValueError(f"incompatible tensor-rule parameter: {name}")
        action_input_size = 5 * self.component_size
        extended = {
            "structure_w": (self.hidden_size, self.component_size),
            "structure_b": (self.component_size,),
            "timing_w": (self.hidden_size, self.component_size),
            "timing_b": (self.component_size,),
            "regime_w": (self.hidden_size, self.component_size),
            "regime_b": (self.component_size,),
            "risk_w": (self.hidden_size, self.component_size),
            "risk_b": (self.component_size,),
            "action_embedding": (TENSOR_RULE_ENTRY_ACTIONS, self.component_size),
            "action_adapter_w": (
                TENSOR_RULE_ENTRY_ACTIONS, action_input_size, self.action_latent_size
            ),
            "action_adapter_b": (TENSOR_RULE_ENTRY_ACTIONS, self.action_latent_size),
            "action_value_w": (TENSOR_RULE_ENTRY_ACTIONS, self.action_latent_size),
            "action_value_b": (TENSOR_RULE_ENTRY_ACTIONS,),
            "utility_prototype": (self.max_rules, TENSOR_RULE_ENTRY_ACTIONS + 1),
        }
        migration_seed = int.from_bytes(hashlib.sha256(
            (str(self.seed_fingerprint) + f"|{self.hidden_size}|{self.component_size}").encode("utf-8")
        ).digest()[:8], "little")
        migration = TensorRuleBankState.seed(
            seed=migration_seed, sequence_length=self.sequence_length, input_size=self.input_size,
            hidden_size=self.hidden_size, encoder_layers=self.encoder_layers,
            attention_heads=self.attention_heads, ffn_size=self.ffn_size,
            max_rules=self.max_rules, initial_rules=self.initial_rules,
            min_valid_length=self.min_valid_length, unknown_membership=self.unknown_membership,
            authority=self.authority, actor_residual_clip=self.actor_residual_clip,
            effective_batch_size=self.effective_batch_size, seed_fingerprint=self.seed_fingerprint,
            component_size=self.component_size, action_latent_size=self.action_latent_size,
            counterfactual_encoder_credit_scale=self.counterfactual_encoder_credit_scale,
            lifecycle_encoder_credit_scale=self.lifecycle_encoder_credit_scale,
            utility_contrastive_margin=self.utility_contrastive_margin,
            membership_temperature=self.membership_temperature,
            motif_distance_weight=self.motif_distance_weight,
            utility_assignment_temperature=self.utility_assignment_temperature,
        ) if any(
            not isinstance(getattr(self, name, None), np.ndarray)
            or np.asarray(getattr(self, name, None)).shape != shape
            for name, shape in extended.items()
        ) else None
        for name, shape in extended.items():
            value = getattr(self, name, None)
            if not isinstance(value, np.ndarray) or value.shape != shape:
                setattr(self, name, np.asarray(getattr(migration, name), dtype=float).copy())
        if not isinstance(self.blocks, list) or len(self.blocks) != self.encoder_layers:
            raise ValueError("incompatible tensor-rule transformer depth")
        for name, shape, dtype, fill in (
            ("active_mask", (self.max_rules,), bool, False),
            ("rule_support", (self.max_rules,), float, 0.0),
            ("rule_action_support", (self.max_rules, TENSOR_RULE_ENTRY_ACTIONS), float, 0.0),
            ("temporal_fold_support", (self.max_rules, 3), float, 0.0),
            ("calibration_error", (TENSOR_RULE_ENTRY_ACTIONS,), float, 0.10),
            ("low_support_rounds", (self.max_rules,), int, 0),
        ):
            value = np.asarray(getattr(self, name, np.zeros(0)), dtype=dtype)
            if value.shape != shape:
                value = np.full(shape, fill, dtype=dtype)
            setattr(self, name, value)
        if not np.any(self.active_mask):
            self.active_mask[:self.initial_rules] = True
        if not isinstance(getattr(self, "gradient_accumulator", None), dict):
            self.gradient_accumulator = {}
        self.gradient_sample_count = max(0, int(getattr(self, "gradient_sample_count", 0) or 0))
        self.calibration_rounds = max(0, int(getattr(self, "calibration_rounds", 0) or 0))
        self.pending_regime = list(getattr(self, "pending_regime", []) or [])[-128:]
        self.ready_regime = list(getattr(self, "ready_regime", []) or [])[-128:]
        evidence_shape = (self.max_rules, 3, TENSOR_RULE_ENTRY_ACTIONS)
        for name in (
            "fold_action_return_sum", "fold_action_return_sq_sum",
            "fold_action_return_support",
        ):
            value = np.asarray(getattr(self, name, np.zeros(0)), dtype=float)
            if value.shape != evidence_shape:
                value = np.zeros(evidence_shape, dtype=float)
            setattr(self, name, value)
        self.seed_fingerprint = str(getattr(self, "seed_fingerprint", "") or "")
        # V186 pre-release candidates briefly carried a separate decoder
        # matrix.  The final schema ties reconstruction to input_w.T, keeping
        # the bank within its parameter budget and avoiding two incompatible
        # feature bases.
        if hasattr(self, "reconstruction_w"):
            delattr(self, "reconstruction_w")
        if not isinstance(getattr(self, "last_topology_proposal", None), dict):
            self.last_topology_proposal = {}
        # V186/V191 used a first-observation dictionary that permanently tied
        # many unrelated utility signatures to two random slots.  V192 fully
        # replaces it with learned utility prototypes; keep no dead assignment
        # state in migrated artifacts.
        if hasattr(self, "utility_rule_assignments"):
            delattr(self, "utility_rule_assignments")
        self.sanitize()

    def parameters(self, *, prefix: str = TENSOR_RULE_PARAMETER_PREFIX) -> dict[str, np.ndarray]:
        values = {
            "input_w": self.input_w, "input_b": self.input_b, "positional": self.positional,
            "prototype": self.prototype, "time_importance_logits": self.time_importance_logits,
            "feature_log_scale": self.feature_log_scale,
            "rule_action_mean": self.rule_action_mean,
            "rule_action_log_var": self.rule_action_log_var,
            "auxiliary_w": self.auxiliary_w, "auxiliary_b": self.auxiliary_b,
            "reconstruction_b": self.reconstruction_b,
            "structure_w": self.structure_w, "structure_b": self.structure_b,
            "timing_w": self.timing_w, "timing_b": self.timing_b,
            "regime_w": self.regime_w, "regime_b": self.regime_b,
            "risk_w": self.risk_w, "risk_b": self.risk_b,
            "action_embedding": self.action_embedding,
            "action_adapter_w": self.action_adapter_w,
            "action_adapter_b": self.action_adapter_b,
            "action_value_w": self.action_value_w,
            "action_value_b": self.action_value_b,
            "utility_prototype": self.utility_prototype,
        }
        for index, block in enumerate(self.blocks):
            for name, value in block.items():
                values[f"block_{index}.{name}"] = value
        return {f"{prefix}{name}": np.asarray(value, dtype=float) for name, value in values.items()}

    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.parameters().values()))

    def sanitize(self) -> None:
        for value in self.parameters().values():
            np.nan_to_num(value, copy=False, nan=0.0, posinf=8.0, neginf=-8.0)
        np.clip(self.feature_log_scale, -4.0, 4.0, out=self.feature_log_scale)
        np.clip(self.rule_action_mean, -5.0, 5.0, out=self.rule_action_mean)
        np.clip(self.rule_action_log_var, -6.0, 4.0, out=self.rule_action_log_var)
        np.clip(self.time_importance_logits, -8.0, 8.0, out=self.time_importance_logits)
        np.clip(self.action_embedding, -4.0, 4.0, out=self.action_embedding)
        np.clip(self.action_value_w, -5.0, 5.0, out=self.action_value_w)
        np.clip(self.action_value_b, -5.0, 5.0, out=self.action_value_b)
        utility_norm = np.maximum(
            np.linalg.norm(self.utility_prototype, axis=1, keepdims=True), 1e-9
        )
        self.utility_prototype[:] = self.utility_prototype / utility_norm

    @staticmethod
    def _fingerprint(sequence: np.ndarray, valid_length: int) -> str:
        digest = hashlib.sha256()
        values = np.ascontiguousarray(np.asarray(sequence, dtype=np.float32))
        valid = int(np.clip(valid_length, 0, values.shape[0] if values.ndim else 0))
        causal_values = np.ascontiguousarray(values[-valid:]) if valid else values[:0]
        digest.update(repr(values.shape).encode("ascii"))
        digest.update(str(valid).encode("ascii"))
        digest.update(causal_values.tobytes())
        return digest.hexdigest()[:20]

    def _transformer_forward(
        self, sequence: np.ndarray, valid_length: int,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]], np.ndarray]:
        normalized_input = np.tanh(np.asarray(sequence, dtype=float))
        mask = np.arange(self.sequence_length) >= self.sequence_length - int(valid_length)
        x = normalized_input @ self.input_w + self.input_b + self.positional
        x = np.where(mask[:, None], x, 0.0)
        caches: list[dict[str, object]] = []
        head_width = self.hidden_size // self.attention_heads
        causal = np.tril(np.ones((self.sequence_length, self.sequence_length), dtype=bool))
        attention_mask = causal & mask[None, :]
        for block in self.blocks:
            norm1, norm1_cache = _layer_norm_forward(x, block["ln1_gamma"], block["ln1_beta"])
            q = norm1 @ block["q_w"] + block["q_b"]
            k = norm1 @ block["k_w"] + block["k_b"]
            v = norm1 @ block["v_w"] + block["v_b"]
            qh = q.reshape(self.sequence_length, self.attention_heads, head_width).transpose(1, 0, 2)
            kh = k.reshape(self.sequence_length, self.attention_heads, head_width).transpose(1, 0, 2)
            vh = v.reshape(self.sequence_length, self.attention_heads, head_width).transpose(1, 0, 2)
            scores = np.matmul(qh, kh.transpose(0, 2, 1)) / np.sqrt(float(head_width))
            scores = np.where(attention_mask[None, :, :], scores, -1e9)
            weights = _softmax(scores, axis=-1)
            weights = np.where(mask[None, :, None], weights, 0.0)
            attended_heads = np.matmul(weights, vh)
            attended = attended_heads.transpose(1, 0, 2).reshape(self.sequence_length, self.hidden_size)
            projected = attended @ block["o_w"] + block["o_b"]
            y = x + np.where(mask[:, None], projected, 0.0)
            norm2, norm2_cache = _layer_norm_forward(y, block["ln2_gamma"], block["ln2_beta"])
            ff_pre = norm2 @ block["ff1_w"] + block["ff1_b"]
            ff_hidden, gelu_derivative = _gelu(ff_pre)
            ff_output = ff_hidden @ block["ff2_w"] + block["ff2_b"]
            output = y + np.where(mask[:, None], ff_output, 0.0)
            output = np.where(mask[:, None], output, 0.0)
            caches.append({
                "x": x, "norm1": norm1, "norm1_cache": norm1_cache,
                "q": q, "k": k, "v": v, "qh": qh, "kh": kh, "vh": vh,
                "weights": weights, "attended": attended, "y": y,
                "norm2": norm2, "norm2_cache": norm2_cache,
                "ff_pre": ff_pre, "ff_hidden": ff_hidden,
                "gelu_derivative": gelu_derivative, "mask": mask,
            })
            x = output
        valid_indices = np.flatnonzero(mask)
        if valid_indices.size:
            mean = np.mean(x[valid_indices], axis=0)
            summary = 0.5 * mean + 0.5 * x[valid_indices[-1]]
        else:
            summary = np.zeros(self.hidden_size, dtype=float)
        return x, summary, caches, normalized_input

    def forward(
        self,
        sequence: np.ndarray,
        valid_length: int,
        *,
        training: bool = False,
        input_schema_fingerprint: str | None = None,
    ) -> TensorRuleForward:
        self.ensure_compatible()
        tensor = np.asarray(sequence, dtype=float)
        if tensor.shape != (self.sequence_length, self.input_size):
            empty = np.zeros(TENSOR_RULE_ENTRY_ACTIONS, dtype=float)
            return TensorRuleForward(
                self.schema_version, self.bank_version, "invalid", int(valid_length),
                np.zeros(self.hidden_size), np.zeros(self.max_rules), -1, True,
                "schema_mismatch", empty, empty, empty,
                np.zeros(TENSOR_RULE_AUXILIARY_COUNT),
                np.zeros((self.sequence_length, self.input_size)), {},
            )
        if (
            input_schema_fingerprint is not None
            and str(input_schema_fingerprint) != str(self.input_schema_fingerprint)
        ):
            empty = np.zeros(TENSOR_RULE_ENTRY_ACTIONS, dtype=float)
            return TensorRuleForward(
                self.schema_version, self.bank_version, "schema-mismatch", int(valid_length),
                np.zeros(self.hidden_size), np.zeros(self.max_rules), -1, True,
                "input_fingerprint_mismatch", empty, empty, empty,
                np.zeros(TENSOR_RULE_AUXILIARY_COUNT),
                np.zeros((self.sequence_length, self.input_size)), {},
            )
        valid = int(np.clip(valid_length, 0, self.sequence_length))
        encoded, summary, transformer_caches, normalized_input = self._transformer_forward(tensor, valid)
        valid_mask = np.arange(self.sequence_length) >= self.sequence_length - valid
        active = np.asarray(self.active_mask, dtype=bool)
        active_indices = np.flatnonzero(active)
        memberships = np.zeros(self.max_rules, dtype=float)
        match_cache: dict[str, object] = {}
        if active_indices.size and valid:
            prototype = self.prototype[active_indices]
            precision = np.exp(np.clip(self.feature_log_scale[active_indices], -4.0, 4.0))
            diff = encoded[None, :, :] - prototype
            contribution = np.mean(diff * diff * precision[:, None, :], axis=2)
            importance_logits = np.where(
                valid_mask[None, :], self.time_importance_logits[active_indices], -1e9
            )
            time_weights = _softmax(importance_logits, axis=1)
            sequence_distances = np.sum(time_weights * contribution, axis=1)
            valid_indices = np.flatnonzero(valid_mask)
            motif_weight = float(self.motif_distance_weight) if valid_indices.size > 1 else 0.0
            if motif_weight > 0.0:
                reference_indices = valid_indices[:-1]
                last_index = int(valid_indices[-1])
                encoded_motif = encoded[last_index] - np.mean(
                    encoded[reference_indices], axis=0
                )
                prototype_motif = prototype[:, last_index, :] - np.mean(
                    prototype[:, reference_indices, :], axis=1
                )
                motif_diff = encoded_motif[None, :] - prototype_motif
                motif_distances = np.mean(
                    motif_diff * motif_diff * precision, axis=1
                )
            else:
                reference_indices = np.zeros(0, dtype=int)
                last_index = -1
                encoded_motif = np.zeros(self.hidden_size, dtype=float)
                prototype_motif = np.zeros(
                    (active_indices.size, self.hidden_size), dtype=float
                )
                motif_diff = np.zeros_like(prototype_motif)
                motif_distances = np.zeros(active_indices.size, dtype=float)
            distances = (
                (1.0 - motif_weight) * sequence_distances
                + motif_weight * motif_distances
            )
            # Contribution already averages over the feature dimension.  A
            # second sqrt(hidden) division makes every rule nearly 0.5 and
            # prevents UNKNOWN from ever resolving; use an explicit bounded
            # metric-learning temperature instead.
            local_memberships = _softmax(-distances / self.membership_temperature)
            memberships[active_indices] = local_memberships
            match_cache = {
                "active_indices": active_indices, "diff": diff, "precision": precision,
                "contribution": contribution, "time_weights": time_weights,
                "local_memberships": local_memberships, "valid_mask": valid_mask,
                "sequence_distances": sequence_distances,
                "motif_distances": motif_distances,
                "motif_weight": motif_weight,
                "motif_diff": motif_diff,
                "reference_indices": reference_indices,
                "last_index": last_index,
            }
        component_names = ("structure", "timing", "regime", "risk")
        components: dict[str, np.ndarray] = {}
        component_pre: dict[str, np.ndarray] = {}
        for name in component_names:
            pre = summary @ getattr(self, f"{name}_w") + getattr(self, f"{name}_b")
            component_pre[name] = pre
            components[name] = np.tanh(pre)
        component_concat = np.concatenate([components[name] for name in component_names])
        action_inputs = np.concatenate((
            np.repeat(component_concat[None, :], TENSOR_RULE_ENTRY_ACTIONS, axis=0),
            self.action_embedding,
        ), axis=1)
        action_pre = np.einsum("ai,aij->aj", action_inputs, self.action_adapter_w) + self.action_adapter_b
        action_latents = np.tanh(action_pre)
        action_residual_mean = np.sum(action_latents * self.action_value_w, axis=1) + self.action_value_b
        mean = memberships @ self.rule_action_mean + action_residual_mean
        rule_variance = np.exp(np.clip(self.rule_action_log_var, -6.0, 4.0))
        second = memberships @ (rule_variance + self.rule_action_mean * self.rule_action_mean)
        variance = np.maximum(1e-6, second - mean * mean)
        effective_support = np.maximum(1.0, memberships @ self.rule_action_support)
        standard = np.sqrt(variance)
        standard_error = np.sqrt(variance / effective_support + self.calibration_error ** 2)
        lcb = mean - 1.645 * standard_error
        top_rule = int(np.argmax(memberships)) if np.any(memberships) else -1
        top_membership = float(memberships[top_rule]) if top_rule >= 0 else 0.0
        support_ready = bool(top_rule >= 0 and self.rule_support[top_rule] >= self.min_valid_length)
        if valid < self.min_valid_length:
            unknown, reason = True, "insufficient_history"
        elif top_rule < 0:
            unknown, reason = True, "no_active_rule"
        elif top_membership < self.unknown_membership:
            unknown, reason = True, "low_membership"
        elif not support_ready:
            unknown, reason = True, "insufficient_support"
        else:
            unknown, reason = False, ""
        auxiliary_component_mask = self._auxiliary_component_mask()
        masked_auxiliary_w = self.auxiliary_w * auxiliary_component_mask
        auxiliary = masked_auxiliary_w @ component_concat + self.auxiliary_b
        reconstruction_raw = encoded @ self.input_w.T + self.reconstruction_b
        reconstruction = np.tanh(reconstruction_raw)
        self.decision_count += int(not training)
        self.unknown_count += int(not training and unknown)
        cache = {
            "sequence": tensor.copy(), "valid_length": valid, "valid_mask": valid_mask,
            "encoded": encoded, "summary": summary, "transformer_caches": transformer_caches,
            "normalized_input": normalized_input, "match": match_cache,
            "memberships": memberships, "mean": mean, "variance": variance,
            "reconstruction_raw": reconstruction_raw, "reconstruction": reconstruction,
            "component_names": component_names, "component_pre": component_pre,
            "components": components, "component_concat": component_concat,
            "auxiliary_component_mask": auxiliary_component_mask,
            "action_inputs": action_inputs, "action_pre": action_pre,
            "action_latents": action_latents,
            "action_residual_mean": action_residual_mean,
        } if training else {}
        return TensorRuleForward(
            self.schema_version, self.bank_version, self._fingerprint(tensor, valid), valid,
            summary.copy(), memberships.copy(), top_rule, unknown, reason,
            mean.copy(), standard.copy(), lcb.copy(), auxiliary.copy(), reconstruction.copy(), cache,
            np.asarray([np.linalg.norm(components[name]) for name in component_names], dtype=float),
            np.linalg.norm(action_latents, axis=1),
        )

    def _auxiliary_component_mask(self) -> np.ndarray:
        """Hard-route label families, never market states, to latent factors.

        This mask prevents an outcome such as SL probability from freely
        rewriting the timing factor, while every rule and action remains a
        learned tensor.  It is an optimization contract, not an inference
        rule and therefore cannot directly choose an action.
        """
        width = 4 * self.component_size
        mask = np.zeros((TENSOR_RULE_AUXILIARY_COUNT, width), dtype=float)
        structure = slice(0, self.component_size)
        timing = slice(self.component_size, 2 * self.component_size)
        regime = slice(2 * self.component_size, 3 * self.component_size)
        risk = slice(3 * self.component_size, 4 * self.component_size)
        horizon_count = len(RETURN_HORIZONS_MINUTES)
        # Direction and quantiles describe the price structure conditional on
        # regime; volatility describes regime/risk instead.
        mask[:horizon_count, structure] = 1.0
        mask[:horizon_count, regime] = 1.0
        quantile_end = horizon_count + 3 * horizon_count
        mask[horizon_count:quantile_end, structure] = 1.0
        mask[horizon_count:quantile_end, regime] = 1.0
        mask[quantile_end:len(GLOBAL_AUXILIARY_NAMES), regime] = 1.0
        mask[quantile_end:len(GLOBAL_AUXILIARY_NAMES), risk] = 1.0
        action_start = len(GLOBAL_AUXILIARY_NAMES)
        action_width = len(ACTION_AUXILIARY_NAMES)
        for action in range(TENSOR_RULE_ENTRY_ACTIONS):
            base = action_start + action * action_width
            # All explicitly temporal labels update only the timing factor.
            for local in tuple(
                ACTION_AUXILIARY_NAMES.index(name)
                for name in (
                    "fill_probability", "time_to_positive", "timing_quality",
                    "enter_now_probability", "timing_advantage_r",
                    "optimal_delay_fraction",
                    "optimal_delay_0", "optimal_delay_1", "optimal_delay_2",
                    "optimal_delay_4", "optimal_delay_8", "optimal_delay_12",
                )
            ):
                mask[base + local, timing] = 1.0
            # expected R may depend on every factor.
            mask[base + 2, :] = 1.0
            # SL/MAE are downside-risk conditioned by regime.
            for local in tuple(
                ACTION_AUXILIARY_NAMES.index(name)
                for name in ("sl_probability", "mae_r")
            ):
                mask[base + local, regime] = 1.0
                mask[base + local, risk] = 1.0
            # TP/MFE are structural opportunity conditioned by regime.
            for local in tuple(
                ACTION_AUXILIARY_NAMES.index(name)
                for name in ("tp_probability", "mfe_r")
            ):
                mask[base + local, structure] = 1.0
                mask[base + local, regime] = 1.0
        # Delayed H2/H4/H6/H8 transition targets update regime only.
        mask[AUXILIARY_OUTPUT_COUNT:, regime] = 1.0
        return mask

    def _transformer_backward(
        self,
        direction_encoded: np.ndarray,
        direction_summary: np.ndarray,
        forward: TensorRuleForward,
    ) -> dict[str, np.ndarray]:
        cache = forward.cache
        encoded = np.asarray(cache["encoded"], dtype=float)
        valid_mask = np.asarray(cache["valid_mask"], dtype=bool)
        direction = np.asarray(direction_encoded, dtype=float).copy()
        valid_indices = np.flatnonzero(valid_mask)
        if valid_indices.size:
            direction[valid_indices] += 0.5 * direction_summary / float(valid_indices.size)
            direction[valid_indices[-1]] += 0.5 * direction_summary
        gradients: dict[str, np.ndarray] = {}
        head_width = self.hidden_size // self.attention_heads
        for index in range(self.encoder_layers - 1, -1, -1):
            block = self.blocks[index]
            block_cache = cache["transformer_caches"][index]
            mask = np.asarray(block_cache["mask"], dtype=bool)
            direction = np.where(mask[:, None], direction, 0.0)
            dy = direction.copy()
            dff_output = np.where(mask[:, None], direction, 0.0)
            ff_hidden = np.asarray(block_cache["ff_hidden"], dtype=float)
            norm2 = np.asarray(block_cache["norm2"], dtype=float)
            gradients[f"block_{index}.ff2_w"] = ff_hidden.T @ dff_output
            gradients[f"block_{index}.ff2_b"] = np.sum(dff_output, axis=0)
            dff_hidden = dff_output @ block["ff2_w"].T
            dff_pre = dff_hidden * np.asarray(block_cache["gelu_derivative"], dtype=float)
            gradients[f"block_{index}.ff1_w"] = norm2.T @ dff_pre
            gradients[f"block_{index}.ff1_b"] = np.sum(dff_pre, axis=0)
            dnorm2 = dff_pre @ block["ff1_w"].T
            dnorm2_input, dgamma2, dbeta2 = _layer_norm_backward(
                dnorm2, block_cache["norm2_cache"]
            )
            gradients[f"block_{index}.ln2_gamma"] = dgamma2
            gradients[f"block_{index}.ln2_beta"] = dbeta2
            dy += dnorm2_input
            dx = dy.copy()
            dprojected = np.where(mask[:, None], dy, 0.0)
            attended = np.asarray(block_cache["attended"], dtype=float)
            gradients[f"block_{index}.o_w"] = attended.T @ dprojected
            gradients[f"block_{index}.o_b"] = np.sum(dprojected, axis=0)
            dattended = dprojected @ block["o_w"].T
            dattended_heads = dattended.reshape(
                self.sequence_length, self.attention_heads, head_width
            ).transpose(1, 0, 2)
            weights = np.asarray(block_cache["weights"], dtype=float)
            qh = np.asarray(block_cache["qh"], dtype=float)
            kh = np.asarray(block_cache["kh"], dtype=float)
            vh = np.asarray(block_cache["vh"], dtype=float)
            dweights = np.matmul(dattended_heads, vh.transpose(0, 2, 1))
            dvh = np.matmul(weights.transpose(0, 2, 1), dattended_heads)
            dscores = weights * (
                dweights - np.sum(dweights * weights, axis=-1, keepdims=True)
            )
            dqh = np.matmul(dscores, kh) / np.sqrt(float(head_width))
            dkh = np.matmul(dscores.transpose(0, 2, 1), qh) / np.sqrt(float(head_width))
            dq = dqh.transpose(1, 0, 2).reshape(self.sequence_length, self.hidden_size)
            dk = dkh.transpose(1, 0, 2).reshape(self.sequence_length, self.hidden_size)
            dv = dvh.transpose(1, 0, 2).reshape(self.sequence_length, self.hidden_size)
            norm1 = np.asarray(block_cache["norm1"], dtype=float)
            for letter, local in (("q", dq), ("k", dk), ("v", dv)):
                gradients[f"block_{index}.{letter}_w"] = norm1.T @ local
                gradients[f"block_{index}.{letter}_b"] = np.sum(local, axis=0)
            dnorm1 = dq @ block["q_w"].T + dk @ block["k_w"].T + dv @ block["v_w"].T
            dnorm1_input, dgamma1, dbeta1 = _layer_norm_backward(
                dnorm1, block_cache["norm1_cache"]
            )
            gradients[f"block_{index}.ln1_gamma"] = dgamma1
            gradients[f"block_{index}.ln1_beta"] = dbeta1
            direction = dx + dnorm1_input
        normalized_input = np.asarray(cache["normalized_input"], dtype=float)
        gradients["input_w"] = normalized_input.T @ direction
        gradients["input_b"] = np.sum(direction, axis=0)
        gradients["positional"] = direction
        return gradients

    def sample_gradients(
        self,
        forward: TensorRuleForward,
        targets: TensorRuleTargets,
        *,
        summary_direction: np.ndarray | None = None,
        membership_logit_direction: np.ndarray | None = None,
        prototype_target_rule: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        if not forward.cache:
            return {}, {"eligible": 0.0}
        memberships = np.asarray(forward.memberships, dtype=float)
        direction_membership = np.zeros(self.max_rules, dtype=float)
        direction_summary = np.zeros(self.hidden_size, dtype=float)
        if summary_direction is not None:
            direction_summary += np.asarray(summary_direction, dtype=float)
        direction_encoded = np.zeros((self.sequence_length, self.hidden_size), dtype=float)
        gradients: dict[str, np.ndarray] = {
            "prototype": np.zeros_like(self.prototype),
            "time_importance_logits": np.zeros_like(self.time_importance_logits),
            "feature_log_scale": np.zeros_like(self.feature_log_scale),
            "rule_action_mean": np.zeros_like(self.rule_action_mean),
            "rule_action_log_var": np.zeros_like(self.rule_action_log_var),
            "auxiliary_w": np.zeros_like(self.auxiliary_w),
            "auxiliary_b": np.zeros_like(self.auxiliary_b),
            "reconstruction_b": np.zeros_like(self.reconstruction_b),
            "structure_w": np.zeros_like(self.structure_w),
            "structure_b": np.zeros_like(self.structure_b),
            "timing_w": np.zeros_like(self.timing_w),
            "timing_b": np.zeros_like(self.timing_b),
            "regime_w": np.zeros_like(self.regime_w),
            "regime_b": np.zeros_like(self.regime_b),
            "risk_w": np.zeros_like(self.risk_w),
            "risk_b": np.zeros_like(self.risk_b),
            "action_embedding": np.zeros_like(self.action_embedding),
            "action_adapter_w": np.zeros_like(self.action_adapter_w),
            "action_adapter_b": np.zeros_like(self.action_adapter_b),
            "action_value_w": np.zeros_like(self.action_value_w),
            "action_value_b": np.zeros_like(self.action_value_b),
        }
        component_direction = np.zeros(4 * self.component_size, dtype=float)
        action_loss = auxiliary_loss = reconstruction_loss = 0.0
        action_targets = None
        if targets.entry_utilities is not None:
            action_targets = np.clip(np.asarray(targets.entry_utilities[1:7], dtype=float), -3.0, 5.0)
        elif targets.realized_r is not None and targets.selected_action is not None and 1 <= targets.selected_action <= 6:
            action_targets = np.asarray(forward.action_mean_r, dtype=float).copy()
            action_targets[targets.selected_action - 1] = float(np.clip(targets.realized_r, -3.0, 5.0))
        if action_targets is not None:
            counterfactual = targets.entry_utilities is not None
            residual = action_targets - np.asarray(forward.action_mean_r, dtype=float)
            clipped = np.clip(residual, -1.0, 1.0) / float(TENSOR_RULE_ENTRY_ACTIONS)
            action_loss = float(np.mean(0.5 * np.minimum(residual * residual, 1.0) + np.maximum(np.abs(residual) - 1.0, 0.0)))
            if counterfactual:
                target_rank = _softmax(action_targets / 0.50)
                predicted_rank = _softmax(np.asarray(forward.action_mean_r) / 0.50)
                direction_mean = clipped + 0.25 * (target_rank - predicted_rank)
                eligible_actions = range(TENSOR_RULE_ENTRY_ACTIONS)
                encoder_credit_scale = self.counterfactual_encoder_credit_scale
            else:
                direction_mean = np.zeros(TENSOR_RULE_ENTRY_ACTIONS, dtype=float)
                selected_index = int(targets.selected_action) - 1
                direction_mean[selected_index] = float(clipped[selected_index])
                eligible_actions = (selected_index,)
                encoder_credit_scale = self.lifecycle_encoder_credit_scale
            gradients["rule_action_mean"] += np.outer(memberships, direction_mean)
            direction_membership += encoder_credit_scale * (self.rule_action_mean @ direction_mean)
            target_log_var = np.log(np.maximum(0.05, residual * residual + 0.05))
            for action in eligible_actions:
                gradients["rule_action_log_var"][:, action] += memberships * (
                    target_log_var[action] - self.rule_action_log_var[:, action]
                ) * 0.10
                local_direction = float(direction_mean[action])
                action_latent = np.asarray(forward.cache["action_latents"], dtype=float)[action]
                action_input = np.asarray(forward.cache["action_inputs"], dtype=float)[action]
                gradients["action_value_w"][action] += local_direction * action_latent
                gradients["action_value_b"][action] += local_direction
                latent_direction = local_direction * self.action_value_w[action]
                pre_direction = latent_direction * (1.0 - action_latent * action_latent)
                gradients["action_adapter_w"][action] += np.outer(action_input, pre_direction)
                gradients["action_adapter_b"][action] += pre_direction
                input_direction = self.action_adapter_w[action] @ pre_direction
                gradients["action_embedding"][action] += input_direction[-self.component_size:]
                component_direction += encoder_credit_scale * input_direction[:-self.component_size]
        aux_mask = np.asarray(targets.auxiliary_mask, dtype=bool)
        # A snapshot can expose dozens of auxiliary labels at once.  Summing
        # every head made the encoder direction grow with the number of labels,
        # so AdamW's single global clip mostly discarded the much smaller rule
        # membership signal.  Average the dense task, just as reconstruction
        # and action ranking are averaged, so adding a label cannot silently
        # change the optimizer scale of the whole Transformer.
        auxiliary_target_count = max(
            1, int(np.count_nonzero(aux_mask[:TENSOR_RULE_AUXILIARY_COUNT]))
        )
        auxiliary_gradient_scale = 1.0 / float(auxiliary_target_count)
        if np.any(aux_mask[:AUXILIARY_OUTPUT_COUNT]):
            legacy_targets = AuxiliaryTargets(
                np.asarray(targets.auxiliary_values[:AUXILIARY_OUTPUT_COUNT], dtype=float),
                aux_mask[:AUXILIARY_OUTPUT_COUNT],
            )
            local_loss, local_direction, _ = auxiliary_loss_and_ascent(
                np.asarray(forward.auxiliary_logits[:AUXILIARY_OUTPUT_COUNT], dtype=float),
                legacy_targets,
            )
            auxiliary_loss += float(local_loss)
            direction = np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=float)
            direction[:AUXILIARY_OUTPUT_COUNT] = (
                local_direction * auxiliary_gradient_scale
            )
            component_concat = np.asarray(forward.cache["component_concat"], dtype=float)
            auxiliary_component_mask = np.asarray(
                forward.cache["auxiliary_component_mask"], dtype=float
            )
            gradients["auxiliary_w"] += (
                np.outer(direction, component_concat) * auxiliary_component_mask
            )
            gradients["auxiliary_b"] += direction
            component_direction += (
                self.auxiliary_w * auxiliary_component_mask
            ).T @ direction
        regime_start = AUXILIARY_OUTPUT_COUNT
        for index in range(len(TENSOR_RULE_REGIME_HORIZONS)):
            target_index = regime_start + index
            if not bool(aux_mask[target_index]):
                continue
            raw = float(np.clip(forward.auxiliary_logits[target_index], -30.0, 30.0))
            target = float(targets.auxiliary_values[target_index])
            probability = 1.0 / (1.0 + np.exp(-raw))
            local = (target - probability) * auxiliary_gradient_scale
            auxiliary_loss += float(np.logaddexp(0.0, raw) - target * raw)
            component_concat = np.asarray(forward.cache["component_concat"], dtype=float)
            auxiliary_component_mask = np.asarray(
                forward.cache["auxiliary_component_mask"], dtype=float
            )
            gradients["auxiliary_w"][target_index] += (
                local * component_concat * auxiliary_component_mask[target_index]
            )
            gradients["auxiliary_b"][target_index] += local
            component_direction += (
                local * self.auxiliary_w[target_index]
                * auxiliary_component_mask[target_index]
            )

        # Component heads are the only bridge from dense/counterfactual
        # objectives into the shared Transformer summary. Realized lifecycle
        # credit uses encoder_credit_scale=0 by default and therefore remains
        # local to the executed action adapter and rule posterior.
        component_names = tuple(forward.cache["component_names"])
        components = dict(forward.cache["components"])
        for component_index, name in enumerate(component_names):
            start = component_index * self.component_size
            local = component_direction[start:start + self.component_size]
            activated = np.asarray(components[name], dtype=float)
            pre_direction = local * (1.0 - activated * activated)
            gradients[f"{name}_w"] += np.outer(forward.summary, pre_direction)
            gradients[f"{name}_b"] += pre_direction
            direction_summary += getattr(self, f"{name}_w") @ pre_direction
        valid_mask = np.asarray(forward.cache["valid_mask"], dtype=bool)
        if (
            prototype_target_rule is not None
            and 0 <= int(prototype_target_rule) < self.max_rules
            and bool(self.active_mask[int(prototype_target_rule)])
            and np.any(valid_mask)
        ):
            # Supervised prototype-centre objective.  The target rule comes
            # from a soft utility-signature assignment, never from an
            # inference-time if/else rule.  Its direction is still only an
            # AdamW input; no prototype is mutated here.
            rule = int(prototype_target_rule)
            encoded = np.asarray(forward.cache["encoded"], dtype=float)
            delta = encoded[valid_mask] - self.prototype[rule, valid_mask]
            strength = 0.10 / float(max(1, np.count_nonzero(valid_mask)))
            gradients["prototype"][rule, valid_mask] += strength * delta
            direction_encoded[valid_mask] -= strength * delta
        # A settled lifecycle may update only the executed action branch.  It
        # must not obtain an accidental encoder gradient merely because the
        # immutable decision sequence is available.  Reconstruction belongs
        # to dense counterfactual/auxiliary training, not sparse realized PnL.
        dense_encoder_target = (
            targets.entry_utilities is not None or np.any(aux_mask)
        )
        if np.any(valid_mask) and dense_encoder_target:
            reconstruction = np.asarray(forward.reconstruction, dtype=float)
            target_sequence = np.asarray(forward.cache["normalized_input"], dtype=float)
            residual = target_sequence - reconstruction
            local = np.where(valid_mask[:, None], residual * (1.0 - reconstruction * reconstruction), 0.0)
            local /= float(max(1, np.count_nonzero(valid_mask) * self.input_size))
            reconstruction_loss = float(0.5 * np.mean(residual[valid_mask] ** 2))
            encoded = np.asarray(forward.cache["encoded"], dtype=float)
            gradients["reconstruction_b"] += np.sum(local, axis=0)
            direction_encoded += local @ self.input_w
            decoder_input_w_direction = local.T @ encoded
        else:
            decoder_input_w_direction = np.zeros_like(self.input_w)
        match = dict(forward.cache.get("match", {}) or {})
        active_indices = np.asarray(match.get("active_indices", ()), dtype=int)
        if active_indices.size:
            local_memberships = np.asarray(match["local_memberships"], dtype=float)
            dm = direction_membership[active_indices]
            dlogit = local_memberships * (dm - float(np.sum(dm * local_memberships)))
            if membership_logit_direction is not None:
                requested = np.asarray(membership_logit_direction, dtype=float).reshape(-1)
                if requested.size == self.max_rules:
                    dlogit += requested[active_indices]
            ddistance = -dlogit / self.membership_temperature
            diff = np.asarray(match["diff"], dtype=float)
            precision = np.asarray(match["precision"], dtype=float)
            time_weights = np.asarray(match["time_weights"], dtype=float)
            contribution = np.asarray(match["contribution"], dtype=float)
            motif_weight = float(match.get("motif_weight", 0.0) or 0.0)
            sequence_weight = 1.0 - motif_weight
            motif_diff = np.asarray(
                match.get("motif_diff", np.zeros((active_indices.size, self.hidden_size))),
                dtype=float,
            )
            reference_indices = np.asarray(
                match.get("reference_indices", ()), dtype=int
            )
            last_index = int(match.get("last_index", -1) or -1)
            for local_index, rule_index in enumerate(active_indices):
                dcontribution = (
                    ddistance[local_index] * sequence_weight
                    * time_weights[local_index]
                )
                scaled = 2.0 * diff[local_index] * precision[local_index][None, :]
                scaled *= dcontribution[:, None] / float(self.hidden_size)
                direction_encoded += scaled
                gradients["prototype"][rule_index] -= scaled
                gradients["feature_log_scale"][rule_index] += np.sum(
                    diff[local_index] ** 2 * precision[local_index][None, :]
                    * dcontribution[:, None] / float(self.hidden_size), axis=0
                )
                dtime = (
                    ddistance[local_index] * sequence_weight
                    * contribution[local_index]
                )
                weights = time_weights[local_index]
                gradients["time_importance_logits"][rule_index] += weights * (
                    dtime - float(np.sum(dtime * weights))
                )
                if motif_weight > 0.0 and last_index >= 0 and reference_indices.size:
                    motif_direction = (
                        2.0 * motif_diff[local_index] * precision[local_index]
                        * ddistance[local_index] * motif_weight
                        / float(self.hidden_size)
                    )
                    direction_encoded[last_index] += motif_direction
                    direction_encoded[reference_indices] -= (
                        motif_direction / float(reference_indices.size)
                    )
                    gradients["prototype"][rule_index, last_index] -= motif_direction
                    gradients["prototype"][rule_index, reference_indices] += (
                        motif_direction / float(reference_indices.size)
                    )
                    gradients["feature_log_scale"][rule_index] += (
                        motif_diff[local_index] ** 2 * precision[local_index]
                        * ddistance[local_index] * motif_weight
                        / float(self.hidden_size)
                    )
        transformer_gradients = self._transformer_backward(
            direction_encoded, direction_summary, forward
        )
        transformer_gradients["input_w"] += decoder_input_w_direction
        gradients.update(transformer_gradients)
        prefixed = {f"{TENSOR_RULE_PARAMETER_PREFIX}{key}": value for key, value in gradients.items()}
        return prefixed, {
            "eligible": 1.0, "action_loss": action_loss,
            "auxiliary_loss": auxiliary_loss / float(auxiliary_target_count),
            "auxiliary_target_count": float(auxiliary_target_count),
            "reconstruction_loss": reconstruction_loss,
        }

    def _contrastive_directions(
        self, forwards: list[TensorRuleForward], targets: list[TensorRuleTargets],
    ) -> tuple[list[np.ndarray], float, int, int, float]:
        directions = [np.zeros_like(item.summary) for item in forwards]
        direction_pair_counts = np.zeros(len(forwards), dtype=int)
        loss = 0.0
        pairs = 0
        hard_negatives = 0

        def cosine_and_directions(left: np.ndarray, right: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
            left_norm = max(float(np.linalg.norm(left)), 1e-9)
            right_norm = max(float(np.linalg.norm(right)), 1e-9)
            cosine = float(np.clip((left @ right) / (left_norm * right_norm), -1.0, 1.0))
            dleft = right / (left_norm * right_norm) - cosine * left / (left_norm * left_norm)
            dright = left / (left_norm * right_norm) - cosine * right / (right_norm * right_norm)
            return cosine, dleft, dright

        for left in range(len(forwards)):
            left_utility = targets[left].entry_utilities
            if left_utility is None:
                continue
            for right in range(left + 1, len(forwards)):
                right_utility = targets[right].entry_utilities
                if right_utility is None or targets[left].temporal_fold == targets[right].temporal_fold:
                    continue
                left_values = np.asarray(left_utility[1:7], dtype=float)
                right_values = np.asarray(right_utility[1:7], dtype=float)
                left_signature = left_values - float(np.mean(left_values))
                right_signature = right_values - float(np.mean(right_values))
                utility_cosine, _, _ = cosine_and_directions(left_signature, right_signature)
                latent_cosine, dleft, dright = cosine_and_directions(
                    forwards[left].summary, forwards[right].summary
                )
                same_utility = (
                    int(np.argmax(left_utility)) == int(np.argmax(right_utility))
                    and utility_cosine >= 0.80
                )
                if same_utility:
                    strength = 0.05 * (1.0 - latent_cosine)
                    directions[left] += strength * dleft
                    directions[right] += strength * dright
                    direction_pair_counts[left] += 1
                    direction_pair_counts[right] += 1
                    loss += 0.5 * (1.0 - latent_cosine) ** 2
                    pairs += 1
                elif latent_cosine > self.utility_contrastive_margin:
                    strength = 0.05 * (latent_cosine - self.utility_contrastive_margin)
                    directions[left] -= strength * dleft
                    directions[right] -= strength * dright
                    direction_pair_counts[left] += 1
                    direction_pair_counts[right] += 1
                    loss += 0.5 * (latent_cosine - self.utility_contrastive_margin) ** 2
                    pairs += 1
                    hard_negatives += 1

        # Pair count depends on the accidental composition of an online batch.
        # Normalize before adding VICReg so a sample with nineteen neighbours
        # cannot drown the supervised prototype assignment of a sample with
        # one neighbour.
        for index, count in enumerate(direction_pair_counts):
            if count > 1:
                directions[index] /= float(count)
        if pairs:
            loss /= float(pairs)

        # VICReg-style anti-collapse pressure. It is batch-local, has no
        # inference branch and merely prevents every sequence from occupying
        # one low-rank direction.
        variance_loss = 0.0
        if len(forwards) >= 4:
            matrix = np.stack([item.summary for item in forwards], axis=0)
            centered = matrix - np.mean(matrix, axis=0, keepdims=True)
            std = np.sqrt(np.mean(centered * centered, axis=0) + 1e-6)
            deficit = np.maximum(0.0, 0.20 - std)
            variance_loss += float(np.mean(deficit * deficit))
            variance_direction = centered * (deficit / np.maximum(std, 1e-6))[None, :]
            variance_direction /= float(max(1, len(forwards) * self.hidden_size))
            covariance = centered.T @ centered / float(max(1, len(forwards) - 1))
            covariance -= np.diag(np.diag(covariance))
            variance_loss += float(0.01 * np.mean(covariance * covariance))
            covariance_direction = -0.02 * (centered @ covariance)
            covariance_direction /= float(max(1, (len(forwards) - 1) * self.hidden_size))
            for index in range(len(forwards)):
                directions[index] += variance_direction[index] + covariance_direction[index]
        return directions, float(loss + variance_loss), pairs, hard_negatives, float(variance_loss)

    @staticmethod
    def _cosine_directions(
        left: np.ndarray,
        right: np.ndarray,
    ) -> tuple[float, np.ndarray, np.ndarray]:
        left_norm = max(float(np.linalg.norm(left)), 1e-9)
        right_norm = max(float(np.linalg.norm(right)), 1e-9)
        cosine = float(np.clip(
            (left @ right) / (left_norm * right_norm), -1.0, 1.0
        ))
        dleft = right / (left_norm * right_norm) - cosine * left / (left_norm * left_norm)
        dright = left / (left_norm * right_norm) - cosine * right / (right_norm * right_norm)
        return cosine, dleft, dright

    @staticmethod
    def _utility_feature(values: np.ndarray) -> np.ndarray:
        """Continuous action-ranking signature used only as a delayed target."""
        utility = np.asarray(values, dtype=float).reshape(-1)
        advantage = utility - float(utility[0])
        advantage -= float(np.mean(advantage))
        return advantage / max(float(np.linalg.norm(advantage)), 1e-9)

    def _balanced_utility_assignments(
        self,
        features: np.ndarray,
        active: np.ndarray,
    ) -> np.ndarray:
        """Soft tensor assignments with batch balancing, never inference rules.

        The former first-seen signature dictionary mixed unrelated outcomes in
        the same slot forever.  Here the target space is continuous and its
        prototypes are updated by the same AdamW writer as the Transformer.
        A few Sinkhorn-style normalizations prevent both active rules from
        remaining at the symmetric 50/50 fixed point.
        """
        if not len(features) or not active.size:
            return np.zeros((len(features), active.size), dtype=float)
        prototypes = np.asarray(self.utility_prototype[active], dtype=float)
        prototypes /= np.maximum(
            np.linalg.norm(prototypes, axis=1, keepdims=True), 1e-9
        )
        logits = features @ prototypes.T / float(self.utility_assignment_temperature)
        assignments = _softmax(logits, axis=1)
        if len(features) >= active.size:
            for _ in range(3):
                assignments /= np.maximum(
                    np.sum(assignments, axis=0, keepdims=True), 1e-9
                )
                assignments *= float(len(features)) / float(active.size)
                assignments /= np.maximum(
                    np.sum(assignments, axis=1, keepdims=True), 1e-9
                )
        assignments = assignments * assignments
        assignments /= np.maximum(
            np.sum(assignments, axis=1, keepdims=True), 1e-9
        )
        return assignments

    def batch_gradients(
        self,
        samples: Iterable[tuple[np.ndarray, int, TensorRuleTargets]],
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        prepared = list(samples)
        forwards = [self.forward(sequence, valid, training=True) for sequence, valid, _ in prepared]
        targets = [target for _, _, target in prepared]
        contrastive, contrastive_loss, pairs, hard_negatives, variance_loss = self._contrastive_directions(forwards, targets)
        membership_directions = [np.zeros(self.max_rules, dtype=float) for _ in forwards]
        prototype_targets: list[int | None] = [None for _ in forwards]
        membership_loss = 0.0
        active = np.flatnonzero(self.active_mask)
        eligible_indices: list[int] = []
        utility_features: list[np.ndarray] = []
        for index, target in enumerate(targets):
            values = np.asarray(
                target.entry_utilities if target.entry_utilities is not None else (),
                dtype=float,
            ).reshape(-1)
            if active.size and values.size == TENSOR_RULE_ENTRY_ACTIONS + 1:
                eligible_indices.append(index)
                utility_features.append(self._utility_feature(values))
        feature_matrix = np.asarray(utility_features, dtype=float)
        assignments = self._balanced_utility_assignments(feature_matrix, active)
        utility_direction = np.zeros_like(self.utility_prototype)
        assignment_entropy = 0.0
        for local_index, index in enumerate(eligible_indices):
            forward = forwards[index]
            local = np.asarray(forward.memberships, dtype=float)
            target_membership = np.zeros(self.max_rules, dtype=float)
            target_membership[active] = assignments[local_index]
            membership_directions[index] = 0.75 * (target_membership - local)
            assigned = int(active[int(np.argmax(assignments[local_index]))])
            prototype_targets[index] = assigned
            membership_loss += float(-np.sum(
                target_membership[active]
                * np.log(np.maximum(local[active], 1e-12))
            ))
            positive = assignments[local_index][assignments[local_index] > 1e-12]
            assignment_entropy += float(-np.sum(positive * np.log(positive)))
            for column, rule in enumerate(active):
                weight = float(assignments[local_index, column])
                utility_direction[rule] += 0.20 * weight * (
                    feature_matrix[local_index] - self.utility_prototype[rule]
                )
        # Keep learned utility anchors distinct even when a short online batch
        # happens to contain only one dominant outcome family.
        for left in range(active.size):
            for right in range(left + 1, active.size):
                li, ri = int(active[left]), int(active[right])
                lv = self.utility_prototype[li]
                rv = self.utility_prototype[ri]
                cosine, dleft, dright = self._cosine_directions(lv, rv)
                if cosine > 0.0:
                    utility_direction[li] -= 0.05 * cosine * dleft
                    utility_direction[ri] -= 0.05 * cosine * dright
        total: dict[str, np.ndarray] = {}
        metrics = {
            "samples": 0.0, "action_loss": 0.0, "auxiliary_loss": 0.0,
            "auxiliary_target_count": 0.0, "reconstruction_loss": 0.0,
        }
        for forward, target, direction, membership_direction, prototype_target in zip(
            forwards, targets, contrastive, membership_directions, prototype_targets
        ):
            gradients, local = self.sample_gradients(
                forward, target, summary_direction=direction,
                membership_logit_direction=membership_direction,
                prototype_target_rule=prototype_target,
            )
            if not gradients:
                continue
            metrics["samples"] += 1.0
            for name in (
                "action_loss", "auxiliary_loss", "auxiliary_target_count",
                "reconstruction_loss",
            ):
                metrics[name] += float(local.get(name, 0.0))
            for key, value in gradients.items():
                total[key] = total.get(key, np.zeros_like(value)) + value
            memberships = np.asarray(forward.memberships, dtype=float)
            self.rule_support += memberships
            fold = int(np.clip(target.temporal_fold, 0, 2))
            self.temporal_fold_support[:, fold] += memberships
            if target.entry_utilities is not None:
                returns = np.clip(
                    np.asarray(target.entry_utilities[1:7], dtype=float), -5.0, 5.0
                )
                for action in range(TENSOR_RULE_ENTRY_ACTIONS):
                    self.rule_action_support[:, action] += memberships
                    weighted = memberships
                    self.fold_action_return_support[:, fold, action] += weighted
                    self.fold_action_return_sum[:, fold, action] += weighted * returns[action]
                    self.fold_action_return_sq_sum[:, fold, action] += weighted * returns[action] ** 2
            elif (
                target.realized_r is not None
                and target.selected_action is not None
                and 1 <= int(target.selected_action) <= TENSOR_RULE_ENTRY_ACTIONS
            ):
                action = int(target.selected_action) - 1
                outcome = float(np.clip(target.realized_r, -5.0, 5.0))
                self.rule_action_support[:, action] += memberships
                self.fold_action_return_support[:, fold, action] += memberships
                self.fold_action_return_sum[:, fold, action] += memberships * outcome
                self.fold_action_return_sq_sum[:, fold, action] += memberships * outcome * outcome
        if np.any(utility_direction):
            total[f"{TENSOR_RULE_PARAMETER_PREFIX}utility_prototype"] = utility_direction
        metrics["contrastive_loss"] = contrastive_loss
        metrics["contrastive_pairs"] = float(pairs)
        metrics["contrastive_hard_negatives"] = float(hard_negatives)
        metrics["anti_collapse_loss"] = float(variance_loss)
        metrics["membership_supervision_loss"] = float(
            membership_loss / max(1, sum(np.any(value) for value in membership_directions))
        )
        metrics["utility_assignment_entropy"] = float(
            assignment_entropy / max(1, len(eligible_indices))
        )
        if assignments.size:
            usage = np.mean(assignments, axis=0)
            metrics["utility_assignment_min_usage"] = float(np.min(usage))
            metrics["utility_assignment_max_usage"] = float(np.max(usage))
        else:
            metrics["utility_assignment_min_usage"] = 0.0
            metrics["utility_assignment_max_usage"] = 0.0
        return total, metrics

    def accumulate_gradients(
        self,
        gradients: Mapping[str, np.ndarray],
        *,
        sample_count: int,
    ) -> dict[str, np.ndarray]:
        count = max(0, int(sample_count))
        if count <= 0:
            return {}
        for key, value in gradients.items():
            array = np.asarray(value, dtype=float)
            self.gradient_accumulator[key] = self.gradient_accumulator.get(
                key, np.zeros_like(array)
            ) + array
        self.gradient_sample_count += count
        if self.gradient_sample_count < self.effective_batch_size:
            return {}
        divisor = float(max(1, self.gradient_sample_count))
        ready = {key: value / divisor for key, value in self.gradient_accumulator.items()}
        raw_norm = float(np.sqrt(sum(
            float(np.sum(np.asarray(value, dtype=float) ** 2))
            for value in ready.values()
        )))
        clip_factor = min(
            1.0, float(self.ready_gradient_norm_budget) / max(raw_norm, 1e-12)
        )
        if clip_factor < 1.0:
            ready = {key: value * clip_factor for key, value in ready.items()}
        self.last_ready_gradient_raw_norm = raw_norm
        self.last_ready_gradient_clip_factor = clip_factor
        self.gradient_accumulator = {}
        self.gradient_sample_count = 0
        self.update_count += 1
        return ready

    def observe_regime_decision(
        self,
        sequence: np.ndarray,
        valid_length: int,
        current_regime: str,
        *,
        sequence_id: int,
    ) -> int:
        """Mature H2/H4/H6/H8 transition labels without inference leakage."""
        regime = str(current_regime or "UNKNOWN")
        matured = 0
        remaining: list[dict[str, object]] = []
        for item in self.pending_regime:
            ages = int(sequence_id) - int(item.get("sequence_id", sequence_id))
            labels = dict(item.get("labels", {}) or {})
            for horizon in TENSOR_RULE_REGIME_HORIZONS:
                if horizon not in labels and ages >= horizon:
                    labels[horizon] = float(regime != str(item.get("origin_regime", "UNKNOWN")))
            item["labels"] = labels
            if len(labels) == len(TENSOR_RULE_REGIME_HORIZONS):
                values = np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=float)
                mask = np.zeros(TENSOR_RULE_AUXILIARY_COUNT, dtype=bool)
                for offset, horizon in enumerate(TENSOR_RULE_REGIME_HORIZONS):
                    values[AUXILIARY_OUTPUT_COUNT + offset] = float(labels[horizon])
                    mask[AUXILIARY_OUTPUT_COUNT + offset] = True
                self.ready_regime.append({
                    "sequence": np.asarray(item["sequence"], dtype=np.float32).copy(),
                    "valid_length": int(item["valid_length"]),
                    "targets": TensorRuleTargets(values, mask, temporal_fold=int(sequence_id) % 3),
                })
                matured += 1
            else:
                remaining.append(item)
        self.pending_regime = remaining[-128:]
        tensor = np.asarray(sequence, dtype=np.float32)
        if tensor.shape == (self.sequence_length, self.input_size):
            self.pending_regime.append({
                "sequence": tensor.copy(), "valid_length": int(valid_length),
                "origin_regime": regime, "sequence_id": int(sequence_id), "labels": {},
            })
        self.ready_regime = self.ready_regime[-128:]
        return matured

    def pop_ready_regime(self, limit: int = 16) -> list[tuple[np.ndarray, int, TensorRuleTargets]]:
        count = max(0, min(int(limit), len(self.ready_regime)))
        selected = self.ready_regime[:count]
        self.ready_regime = self.ready_regime[count:]
        return [
            (np.asarray(item["sequence"], dtype=float), int(item["valid_length"]), item["targets"])
            for item in selected
        ]

    def calibrate_uncertainty(self, *, min_cell_support: float = 20.0) -> dict[str, object]:
        """Calibrate one-sided LCB error from held-out temporal cells."""
        active = np.flatnonzero(self.active_mask)
        updated: list[int] = []
        for action in range(TENSOR_RULE_ENTRY_ACTIONS):
            residuals: list[float] = []
            weights: list[float] = []
            for rule in active:
                for fold in range(3):
                    support = float(self.fold_action_return_support[rule, fold, action])
                    if support < float(min_cell_support):
                        continue
                    observed = float(
                        self.fold_action_return_sum[rule, fold, action] / max(support, 1e-12)
                    )
                    residuals.append(abs(observed - float(self.rule_action_mean[rule, action])))
                    weights.append(support)
            if not residuals:
                continue
            order = np.argsort(np.asarray(residuals, dtype=float))
            sorted_residuals = np.asarray(residuals, dtype=float)[order]
            sorted_weights = np.asarray(weights, dtype=float)[order]
            cumulative = np.cumsum(sorted_weights)
            threshold = 0.95 * float(cumulative[-1])
            quantile_index = min(
                len(sorted_residuals) - 1, int(np.searchsorted(cumulative, threshold, side="left"))
            )
            self.calibration_error[action] = float(np.clip(
                sorted_residuals[quantile_index], 0.01, 3.0
            ))
            updated.append(action)
        if updated:
            self.calibration_rounds += 1
        return {
            "updated": bool(updated),
            "actions": [int(action + 1) for action in updated],
            "calibration_rounds": int(self.calibration_rounds),
            "calibration_error_r": [float(value) for value in self.calibration_error],
        }

    def evolve_at_round_boundary(
        self,
        validation: Mapping[str, object],
        *,
        round_boundary: bool,
    ) -> dict[str, object]:
        """Apply at most one deterministic split/merge/retire operation."""
        def finish(payload: Mapping[str, object]) -> dict[str, object]:
            result = dict(payload)
            self.last_topology_proposal = result
            return result

        if not bool(round_boundary):
            return finish({"changed": False, "reason": "topology_frozen_inside_round"})
        self.ensure_compatible()
        calibration = self.calibrate_uncertainty(
            min_cell_support=float(validation.get("calibration_min_cell_support", 20.0) or 20.0)
        )
        active = np.flatnonzero(self.active_mask)
        dormant = np.flatnonzero(~self.active_mask)
        split_improvement = float(validation.get("split_loss_improvement", 0.0) or 0.0)
        child_support = float(validation.get("split_child_min_support", 0.0) or 0.0)
        split_lcb_delta = float(validation.get("split_lcb_delta", -np.inf))
        if (
            dormant.size and active.size and split_improvement >= 0.05
            and child_support >= 32 and split_lcb_delta >= 0.0
        ):
            source = int(active[np.argmax(self.rule_support[active])])
            if self.rule_support[source] >= 128:
                target = int(dormant[0])
                self.prototype[target] = self.prototype[source]
                self.prototype[target, :, 0] += 0.01
                self.time_importance_logits[target] = self.time_importance_logits[source]
                self.feature_log_scale[target] = self.feature_log_scale[source]
                self.rule_action_mean[target] = self.rule_action_mean[source]
                self.rule_action_log_var[target] = self.rule_action_log_var[source]
                self.utility_prototype[target] = self.utility_prototype[source]
                self.utility_prototype[target, target % self.utility_prototype.shape[1]] += 0.05
                self.rule_support[source] *= 0.5
                self.rule_support[target] = self.rule_support[source]
                self.rule_action_support[target] = 0.5 * self.rule_action_support[source]
                self.rule_action_support[source] *= 0.5
                for evidence in (
                    self.fold_action_return_sum, self.fold_action_return_sq_sum,
                    self.fold_action_return_support,
                ):
                    evidence[target] = 0.5 * evidence[source]
                    evidence[source] *= 0.5
                self.active_mask[target] = True
                self.bank_version += 1
                return finish({
                    "changed": True, "operation": "split", "source": source,
                    "target": target, "held_out_loss_improvement": split_improvement,
                    "held_out_lcb_delta": split_lcb_delta,
                    "calibration": calibration,
                })
        merge_loss_worsening = float(validation.get("merge_loss_worsening", 1.0) or 1.0)
        if active.size >= 2 and merge_loss_worsening <= 0.01:
            for left_index, left in enumerate(active[:-1]):
                for right in active[left_index + 1:]:
                    p_left = self.prototype[left].reshape(-1)
                    p_right = self.prototype[right].reshape(-1)
                    cosine = float(np.dot(p_left, p_right) / max(np.linalg.norm(p_left) * np.linalg.norm(p_right), 1e-12))
                    a = _softmax(self.rule_action_mean[left])
                    b = _softmax(self.rule_action_mean[right])
                    middle = 0.5 * (a + b)
                    js = 0.5 * float(np.sum(a * np.log(np.maximum(a, 1e-12) / np.maximum(middle, 1e-12))))
                    js += 0.5 * float(np.sum(b * np.log(np.maximum(b, 1e-12) / np.maximum(middle, 1e-12))))
                    if cosine >= 0.95 and js <= 0.05:
                        weight = self.rule_support[left] / max(self.rule_support[left] + self.rule_support[right], 1e-12)
                        self.prototype[left] = weight * self.prototype[left] + (1.0 - weight) * self.prototype[right]
                        self.rule_action_mean[left] = weight * self.rule_action_mean[left] + (1.0 - weight) * self.rule_action_mean[right]
                        self.utility_prototype[left] = (
                            weight * self.utility_prototype[left]
                            + (1.0 - weight) * self.utility_prototype[right]
                        )
                        self.rule_support[left] += self.rule_support[right]
                        self.rule_action_support[left] += self.rule_action_support[right]
                        for evidence in (
                            self.fold_action_return_sum, self.fold_action_return_sq_sum,
                            self.fold_action_return_support,
                        ):
                            evidence[left] += evidence[right]
                            evidence[right] = 0.0
                        self.active_mask[right] = False
                        self.rule_support[right] = 0.0
                        self.rule_action_support[right] = 0.0
                        self.bank_version += 1
                        return finish({"changed": True, "operation": "merge", "source": int(right), "target": int(left), "cosine": cosine, "js": js, "calibration": calibration})
        unique = dict(validation.get("unique_validation_contribution", {}) or {})
        for rule in active:
            if self.rule_support[rule] < 16 or float(unique.get(str(int(rule)), 1.0) or 0.0) <= 0.0:
                self.low_support_rounds[rule] += 1
            else:
                self.low_support_rounds[rule] = 0
            if self.low_support_rounds[rule] >= 3 and active.size > 1:
                self.active_mask[rule] = False
                self.bank_version += 1
                return finish({"changed": True, "operation": "retire", "source": int(rule), "calibration": calibration})
        return finish({"changed": False, "reason": "no_validated_structure_change", "calibration": calibration})

    def reset_slot_optimizer_state(
        self,
        moment: dict[str, np.ndarray],
        variance: dict[str, np.ndarray],
        slot: int,
    ) -> None:
        index = int(slot)
        for name in (
            "prototype", "time_importance_logits", "feature_log_scale",
            "rule_action_mean", "rule_action_log_var", "utility_prototype",
        ):
            key = f"{TENSOR_RULE_PARAMETER_PREFIX}{name}"
            for state in (moment, variance):
                value = state.get(key)
                if isinstance(value, np.ndarray) and value.shape[0] == self.max_rules:
                    value[index] = 0.0

    def summary(self) -> dict[str, object]:
        active = np.flatnonzero(self.active_mask)
        temporal_lcb: dict[str, object] = {}
        rule_posteriors: dict[str, object] = {}
        for rule in active:
            local: list[list[float | None]] = []
            for fold in range(3):
                count = self.fold_action_return_support[rule, fold]
                mean = np.divide(
                    self.fold_action_return_sum[rule, fold], np.maximum(count, 1e-12)
                )
                second = np.divide(
                    self.fold_action_return_sq_sum[rule, fold], np.maximum(count, 1e-12)
                )
                variance = np.maximum(0.0, second - mean * mean)
                lcb = mean - 1.645 * np.sqrt(
                    variance / np.maximum(count, 1.0) + self.calibration_error ** 2
                )
                local.append([
                    float(lcb[action]) if count[action] >= 2.0 else None
                    for action in range(TENSOR_RULE_ENTRY_ACTIONS)
                ])
            temporal_lcb[str(int(rule))] = local
            posterior_variance = np.exp(np.clip(self.rule_action_log_var[rule], -6.0, 4.0))
            posterior_error = np.sqrt(
                posterior_variance / np.maximum(self.rule_action_support[rule], 1.0)
                + self.calibration_error ** 2
            )
            rule_posteriors[str(int(rule))] = {
                "mean_r": [float(value) for value in self.rule_action_mean[rule]],
                "std_r": [float(value) for value in np.sqrt(posterior_variance)],
                "lcb95_r": [
                    float(value) for value in self.rule_action_mean[rule] - 1.645 * posterior_error
                ],
            }
        return {
            "schema_version": int(self.schema_version), "bank_version": int(self.bank_version),
            "authority": str(self.authority), "parameter_count": self.parameter_count(),
            "input_schema_fingerprint": str(self.input_schema_fingerprint),
            "seed_fingerprint": str(self.seed_fingerprint),
            "active_rules": [int(value) for value in active.tolist()],
            "membership_temperature": float(self.membership_temperature),
            "motif_distance_weight": float(self.motif_distance_weight),
            "utility_assignment_temperature": float(self.utility_assignment_temperature),
            "utility_prototype_cosine": (
                float(np.dot(self.utility_prototype[active[0]], self.utility_prototype[active[1]]))
                if active.size >= 2 else None
            ),
            "rule_support": [float(self.rule_support[value]) for value in active],
            "rule_posteriors": rule_posteriors,
            "temporal_fold_lcb95_r": temporal_lcb,
            "unknown_rate": float(self.unknown_count / max(1, self.decision_count)),
            "calibration_rounds": int(self.calibration_rounds),
            "calibration_error_r": [float(value) for value in self.calibration_error],
            "gradient_sample_count": int(self.gradient_sample_count),
            "ready_gradient_norm_budget": float(self.ready_gradient_norm_budget),
            "last_ready_gradient_raw_norm": float(self.last_ready_gradient_raw_norm),
            "last_ready_gradient_clip_factor": float(self.last_ready_gradient_clip_factor),
            "update_count": int(self.update_count),
            "pending_regime": int(len(self.pending_regime)),
            "ready_regime": int(len(self.ready_regime)),
            "last_topology_proposal": dict(self.last_topology_proposal),
        }


__all__ = [
    "TENSOR_RULE_ARTIFACT_VERSION", "TENSOR_RULE_AUTHORITIES",
    "TENSOR_RULE_AUXILIARY_COUNT", "TENSOR_RULE_PARAMETER_PREFIX",
    "TENSOR_RULE_REGIME_HORIZONS", "TENSOR_RULE_SCHEMA_VERSION",
    "TensorRuleBankState", "TensorRuleForward", "TensorRuleTargets",
]
