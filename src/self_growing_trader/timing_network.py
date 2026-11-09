"""Dedicated, bounded timing network for context-expert policies.

TimingNet is the sole producer of learned timing quality and direction.  It
receives only market, timing and Wyckoff feature groups, emits timing heads
plus a small action correction, and accepts delayed lifecycle credit through a
decision-local trace.  It has no access to future bars or account/PnL fields.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping

import numpy as np

from .input_normalization import normalize_input_vector
from .protocol_contract import (
    BASE_MARKET_FEATURE_COUNT,
    EXTRA_MARKET_OFFSET,
    EXTRA_MARKET_FEATURE_COUNT,
    TIMING_OFFSET,
    TAIL_FEATURE_COUNT,
)


TIMING_NETWORK_MARKET_COUNT = BASE_MARKET_FEATURE_COUNT + EXTRA_MARKET_FEATURE_COUNT
TIMING_NETWORK_INPUT_SIZE = TIMING_NETWORK_MARKET_COUNT + TAIL_FEATURE_COUNT
TIMING_NETWORK_ACTION_COUNT = 12
# Version 4 adds the self-supervised deceleration head.  Its label is observed
# only after a later decision, so it never leaks a future bar into the action
# that made the prediction.  Old traces remain readable through the one-step
# fallback in ``update_lifecycle_credit``.
# Version 6 makes the M15 history an explicit fixed-width tensor.  The
# encoder consumes the most recent closed 32 M15 observations rather than
# relying on an unbounded hidden state whose effective horizon is opaque.
# The 137-column EA bridge remains unchanged: each tensor slice is still the
# same extracted 78 market/timing/Wyckoff inputs.
TIMING_NETWORK_SCHEMA_VERSION = 6

# Keep the label namespace explicit and stable.  The three volatile children
# are separate classes because collapsing them into CTX_VOLATILE would make a
# correct SPIKE->CHOP prediction look like a false negative transition.
REGIME_TRANSITION_LABELS = (
    "CTX_TANG_MANH", "CTX_TANG_YEU", "CTX_GIAM_MANH", "CTX_GIAM_YEU",
    "CTX_RANGING", "CTX_VOLATILE", "CTX_VOLATILE_SPIKE",
    "CTX_VOLATILE_EXPANSION", "CTX_VOLATILE_CHOP",
)
_REGIME_LABEL_INDEX = {name: index for index, name in enumerate(REGIME_TRANSITION_LABELS)}

# The optimizer is deliberately owned by TimingNet.  Policy/M1/graph keep
# their existing bounded online updates; these names are only used to flatten
# the TimingNet parameters into one deterministic vector for shadow Adam and
# SWATS projection.
_OPTIMIZER_PARAMETER_NAMES = (
    "w_quality", "w_direction", "w_entry", "w_urgency", "w_uncertainty", "w_deceleration",
    "w_action", "b_action", "w_embed", "b_embed", "b_quality",
    "b_direction", "b_entry", "b_urgency", "b_uncertainty",
    "w_regime_transition", "b_regime_transition",
    "w_in", "w_rec", "b_hidden",
)
_SLOW_PARAMETER_NAMES = frozenset({"w_rec", "w_embed", "b_embed"})


def extract_timing_features(observation: Iterable[float] | np.ndarray) -> np.ndarray:
    """Extract the versioned market/timing/Wyckoff groups from an observation.

    The bridge's 137-input layout interleaves positions/account/execution
    between market and tail features.  Older 117/121-input artifacts contain
    fewer tail slots; missing values are padded with zero and never inferred
    from a future row.
    """

    raw = np.asarray(observation, dtype=float).reshape(-1)
    raw = np.nan_to_num(raw, nan=0.0, posinf=20.0, neginf=-20.0)
    if raw.size == TIMING_NETWORK_INPUT_SIZE:
        return raw.copy()
    market = np.zeros(TIMING_NETWORK_MARKET_COUNT, dtype=float)
    if raw.size >= TIMING_OFFSET:
        base = raw[:BASE_MARKET_FEATURE_COUNT]
        extra = raw[EXTRA_MARKET_OFFSET : EXTRA_MARKET_OFFSET + EXTRA_MARKET_FEATURE_COUNT]
        market[: min(BASE_MARKET_FEATURE_COUNT, base.size)] = base[:BASE_MARKET_FEATURE_COUNT]
        market[BASE_MARKET_FEATURE_COUNT : BASE_MARKET_FEATURE_COUNT + min(EXTRA_MARKET_FEATURE_COUNT, extra.size)] = extra[:EXTRA_MARKET_FEATURE_COUNT]
    else:
        market_end = min(raw.size, TIMING_NETWORK_MARKET_COUNT)
        if market_end:
            market[:market_end] = raw[:market_end]
    tail = np.zeros(TAIL_FEATURE_COUNT, dtype=float)
    if raw.size > TIMING_OFFSET:
        tail_values = raw[TIMING_OFFSET : TIMING_OFFSET + TAIL_FEATURE_COUNT]
        tail[: min(tail.size, tail_values.size)] = tail_values[: tail.size]
    return np.concatenate([market, tail])


def timing_trend_score(
    observation: Iterable[float] | np.ndarray,
    *,
    mtf_scaling_mode: str = "legacy_m1_atr_v1",
) -> float:
    """Return the live regime router's directional score without look-ahead.

    Keeping this projection shared with ``MarketRegimeClassifier`` makes the
    deceleration label mean exactly "trend score moves toward zero", rather
    than an unrelated latent feature becoming smaller.  Importing locally
    avoids a module-level dependency cycle with the context facade.
    """

    market = extract_timing_features(observation)[:TIMING_NETWORK_MARKET_COUNT]
    from .market_regime_classifier import MarketRegimeClassifier

    return float(np.clip(
        MarketRegimeClassifier.timing_directional_score(
            market,
            mtf_scaling_mode=mtf_scaling_mode,
        ),
        -1.0,
        1.0,
    ))


@dataclass(frozen=True)
class TimingNetworkOutput:
    """Decision-time output exposed to policy telemetry."""

    quality: float
    direction: float
    entry_probability: float
    urgency: float
    uncertainty: float
    deceleration_probability: float
    trend_score: float
    deceleration_update_count: int
    regime_transition_probability: float
    regime_transition_probabilities: np.ndarray
    regime_transition_update_count: int
    current_regime: str
    embedding: np.ndarray
    action_logits: np.ndarray
    features: np.ndarray
    sequence_tensor: np.ndarray
    sequence_valid_length: int
    state_before: np.ndarray
    candidate: np.ndarray
    hidden: np.ndarray
    eligibility_in: np.ndarray
    eligibility_rec: np.ndarray
    eligibility_bias: np.ndarray
    eligibility_steps: int

    def trace_payload(self) -> dict[str, object]:
        """Return the bounded decision-local payload used at lifecycle close."""

        return {
            "schema_version": TIMING_NETWORK_SCHEMA_VERSION,
            "quality": float(self.quality),
            "direction": float(self.direction),
            "entry_probability": float(self.entry_probability),
            "urgency": float(self.urgency),
            "uncertainty": float(self.uncertainty),
            "deceleration_probability": float(self.deceleration_probability),
            "trend_score": float(self.trend_score),
            "regime_transition_probability": float(self.regime_transition_probability),
            "regime_transition_probabilities": np.asarray(self.regime_transition_probabilities, dtype=float).copy(),
            "regime_transition_update_count": int(self.regime_transition_update_count),
            "current_regime": str(self.current_regime),
            "features": np.asarray(self.features, dtype=float).copy(),
            # This is a decision-local causal tensor: rows are ordered from
            # oldest to newest and only closed M15 rows are present.  It is
            # kept float32 in the lifecycle trace just like eligibility.
            "sequence_tensor": np.asarray(self.sequence_tensor, dtype=np.float32).copy(),
            "sequence_valid_length": int(self.sequence_valid_length),
            "state_before": np.asarray(self.state_before, dtype=float).copy(),
            "candidate": np.asarray(self.candidate, dtype=float).copy(),
            "hidden": np.asarray(self.hidden, dtype=float).copy(),
            "embedding": np.asarray(self.embedding, dtype=float).copy(),
            "action_logits": np.asarray(self.action_logits, dtype=float).copy(),
            # Eligibility dominates trace size; float32 is sufficient because
            # gradients are clipped before applying and keeps execution-cache
            # memory bounded during long MT5 runs.
            "eligibility_in": np.asarray(self.eligibility_in, dtype=np.float32).copy(),
            "eligibility_rec": np.asarray(self.eligibility_rec, dtype=np.float32).copy(),
            "eligibility_bias": np.asarray(self.eligibility_bias, dtype=np.float32).copy(),
            "eligibility_steps": int(self.eligibility_steps),
        }

    def signal_payload(self, *, direction_threshold: float = 0.10) -> dict[str, object]:
        """Convert network heads into the canonical context timing signal.

        ``quality`` is deliberately gated by entry probability and uncertainty
        so a high auxiliary quality head cannot approve an entry by itself.
        Direction is ternarized only at the final context boundary; the raw
        continuous head remains available for audit and learning.
        """

        quality = float(np.clip(
            float(self.quality)
            * float(self.entry_probability)
            * float(1.0 - self.uncertainty),
            0.0,
            1.0,
        ))
        threshold = float(np.clip(direction_threshold, 0.0, 1.0))
        raw_direction = float(np.clip(self.direction, -1.0, 1.0))
        direction = 1 if raw_direction >= threshold else -1 if raw_direction <= -threshold else 0
        return {
            "timing_quality": quality,
            "timing_direction": int(direction),
            "timing_network_quality": float(np.clip(self.quality, 0.0, 1.0)),
            "timing_network_direction": raw_direction,
            "timing_network_entry_probability": float(np.clip(self.entry_probability, 0.0, 1.0)),
            "timing_network_uncertainty": float(np.clip(self.uncertainty, 0.0, 1.0)),
            "timing_network_deceleration_probability": float(np.clip(self.deceleration_probability, 0.0, 1.0)),
            "timing_network_trend_score": float(np.clip(self.trend_score, -1.0, 1.0)),
            "timing_network_deceleration_update_count": int(self.deceleration_update_count),
            "timing_network_regime_transition_probability": float(np.clip(self.regime_transition_probability, 0.0, 1.0)),
            "timing_network_regime_transition_probabilities": [float(value) for value in self.regime_transition_probabilities],
            "timing_network_regime_transition_update_count": int(self.regime_transition_update_count),
            "timing_network_current_regime": str(self.current_regime),
            "timing_network_direction_threshold": threshold,
        }


@dataclass
class TimingNetwork:
    """Small recurrent timing encoder with delayed, bounded lifecycle credit."""

    input_size: int = TIMING_NETWORK_INPUT_SIZE
    hidden_size: int = 32
    embedding_size: int = 16
    action_count: int = TIMING_NETWORK_ACTION_COUNT
    memory_decay: float = 0.85
    learning_rate: float = 0.0005
    # Candidate-only recurrent credit controls.  The eligibility matrices are
    # an online truncated-BPTT cache: they retain at most this many effective
    # steps and are included in the entry trace for delayed MT5 settlement.
    bptt_window: int = 24
    # The live TimingNet sees a materialized rolling M15 tensor.  This is
    # intentionally distinct from ``bptt_window``: sequence_length defines
    # the decision representation, while bptt_window bounds delayed-credit
    # eligibility.
    sequence_length: int = 32
    eligibility_decay: float = 0.90
    gradient_clip: float = 1.0
    # Must match the MTF feature contract used to produce ``observation``.
    # Legacy remains the default so protected checkpoints retain their prior
    # interpretation until the active runtime explicitly binds a profile.
    mtf_scaling_mode: str = "legacy_m1_atr_v1"
    state: np.ndarray = field(default=None, repr=False)
    sequence_buffer: np.ndarray = field(default=None, repr=False)
    sequence_valid_length: int = 0
    w_in: np.ndarray = field(default=None, repr=False)
    w_rec: np.ndarray = field(default=None, repr=False)
    b_hidden: np.ndarray = field(default=None, repr=False)
    w_embed: np.ndarray = field(default=None, repr=False)
    b_embed: np.ndarray = field(default=None, repr=False)
    w_quality: np.ndarray = field(default=None, repr=False)
    b_quality: float = 0.0
    w_direction: np.ndarray = field(default=None, repr=False)
    b_direction: float = 0.0
    w_entry: np.ndarray = field(default=None, repr=False)
    b_entry: float = 0.0
    w_urgency: np.ndarray = field(default=None, repr=False)
    b_urgency: float = 0.0
    w_uncertainty: np.ndarray = field(default=None, repr=False)
    b_uncertainty: float = 0.0
    # Self-supervised head: probability that the absolute rule-consistent
    # directional score will move materially toward zero after a bounded
    # number of later observations.
    w_deceleration: np.ndarray = field(default=None, repr=False)
    b_deceleration: float = 0.0
    w_action: np.ndarray = field(default=None, repr=False)
    b_action: np.ndarray = field(default=None, repr=False)
    eligibility_in: np.ndarray = field(default=None, repr=False)
    eligibility_rec: np.ndarray = field(default=None, repr=False)
    eligibility_bias: np.ndarray = field(default=None, repr=False)
    eligibility_steps: int = 0
    update_count: int = 0
    last_update: dict[str, object] = field(default_factory=dict, repr=False)
    deceleration_enabled: bool = False
    deceleration_horizon: int = 3
    deceleration_min_trend_strength: float = 0.18
    deceleration_min_drop: float = 0.10
    deceleration_update_scale: float = 0.25
    deceleration_learning_rate: float = 0.01
    deceleration_encoder_scale: float = 0.10
    deceleration_l2: float = 1e-4
    deceleration_pending: list[dict[str, object]] = field(default_factory=list, repr=False)
    deceleration_observation_count: int = 0
    deceleration_update_count: int = 0
    deceleration_positive_count: int = 0
    deceleration_last_target: float = 0.0
    deceleration_last_error: float = 0.0
    # Multi-horizon causal context-transition prediction.  It is an auxiliary
    # supervised head: labels arrive only after H later decisions and are
    # never fed back into the original action's input vector.
    regime_transition_enabled: bool = False
    regime_transition_horizons: tuple[int, ...] = (3, 6, 12)
    regime_transition_update_scale: float = 0.25
    regime_transition_learning_rate: float = 0.01
    regime_transition_encoder_scale: float = 0.08
    regime_transition_l2: float = 1e-4
    w_regime_transition: np.ndarray = field(default=None, repr=False)
    b_regime_transition: np.ndarray = field(default=None, repr=False)
    regime_transition_pending: list[dict[str, object]] = field(default_factory=list, repr=False)
    regime_transition_observation_count: int = 0
    regime_transition_update_count: int = 0
    regime_transition_last_target: str = ""
    regime_transition_last_error: float = 0.0
    # Calibration is kept separately from the training loss.  The entry
    # supervisor must be able to prove that a high transition probability was
    # historically right before it is allowed to suppress a real MT5 entry.
    # The bounded event list stores the prediction made at t and the delayed
    # binary outcome observed at t + shortest_horizon.
    regime_transition_calibration_events: list[tuple[float, int]] = field(default_factory=list, repr=False)
    regime_transition_calibration_count: int = 0
    regime_transition_brier_ema: float = 0.0
    # Candidate-only PID-SWATS state.  ``legacy`` keeps the protected v13
    # update path byte-compatible; enabling SWATS initializes the phase at
    # ADAM while retaining m/v after the SGD switch for alignment telemetry.
    swats_enabled: bool = False
    pid_enabled: bool = False
    optimizer_phase: str = "legacy"
    shadow_adam_m: np.ndarray = field(default=None, repr=False)
    shadow_adam_v: np.ndarray = field(default=None, repr=False)
    shadow_adam_step: int = 0
    adam_beta1: float = 0.90
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    # AdamW and cosine are deliberately restricted to the pre-switch Adam
    # phase.  Once SWATS has moved to SGD, PID is the single learning-rate
    # actuator; a second cosine controller would invalidate its safety logic.
    adamw_enabled: bool = False
    adamw_weight_decay: float = 0.0
    cosine_annealing_enabled: bool = False
    cosine_min_factor: float = 1.0
    cosine_cycle_steps: int = 128
    projection_beta: float = 0.90
    projection_tolerance: float = 0.10
    switch_consecutive: int = 6
    min_projection_steps: int = 12
    target_alignment: float = 0.90
    eta_sgd: float = 0.0
    eta_min: float = 1e-6
    eta_max: float = 0.004
    eta_init_factor: float = 0.90
    projection_ema: float = 0.0
    projection_count: int = 0
    projection_stable_count: int = 0
    switch_count: int = 0
    pid_integral: float = 0.0
    pid_previous_error: float = 0.0
    pid_derivative: float = 0.0
    pid_alignment_ema: float = 0.0
    pid_r_ema: float = 0.0
    pid_abs_r_ema: float = 0.0
    pid_signal_count: int = 0
    pid_good_steps: int = 0
    pid_bad_steps: int = 0
    pid_cooldown: int = 0
    pid_kp: float = 0.08
    pid_ki: float = 0.01
    pid_kd: float = 0.02
    pid_integral_limit: float = 2.0
    pid_derivative_ema: float = 0.80
    pid_deadband: float = 0.02
    pid_r_target: float = 0.0
    pid_r_scale_floor: float = 0.25
    pid_alignment_weight: float = 0.70
    pid_r_weight: float = 0.30
    pid_bad_steps_to_freeze: int = 4
    pid_recovery_steps: int = 3
    pid_cooldown_steps: int = 8
    pid_fallback_to_adam: bool = False

    @classmethod
    def seed(
        cls,
        *,
        input_size: int = TIMING_NETWORK_INPUT_SIZE,
        hidden_size: int = 32,
        embedding_size: int = 16,
        action_count: int = TIMING_NETWORK_ACTION_COUNT,
        seed: int = 0,
        learning_rate: float = 0.0005,
        bptt_window: int = 24,
        sequence_length: int = 32,
        eligibility_decay: float = 0.90,
        gradient_clip: float = 1.0,
        mtf_scaling_mode: str = "legacy_m1_atr_v1",
    ) -> "TimingNetwork":
        rng = np.random.default_rng(int(seed) & 0x7FFFFFFF)
        input_scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        hidden_scale = 1.0 / max(1.0, np.sqrt(float(hidden_size)))
        network = cls(
            input_size=int(input_size),
            hidden_size=int(hidden_size),
            embedding_size=int(embedding_size),
            action_count=int(action_count),
            learning_rate=float(np.clip(learning_rate, 1e-6, 0.02)),
            bptt_window=int(np.clip(int(bptt_window), 1, 128)),
            sequence_length=int(np.clip(int(sequence_length), 1, 128)),
            eligibility_decay=float(np.clip(eligibility_decay, 0.5, 0.999)),
            gradient_clip=float(np.clip(gradient_clip, 0.05, 10.0)),
            mtf_scaling_mode=str(mtf_scaling_mode or "legacy_m1_atr_v1"),
            state=np.zeros(int(hidden_size), dtype=float),
            sequence_buffer=np.zeros((int(np.clip(int(sequence_length), 1, 128)), int(input_size)), dtype=float),
            w_in=rng.normal(0.0, input_scale, size=(int(hidden_size), int(input_size))),
            w_rec=rng.normal(0.0, 0.03, size=(int(hidden_size), int(hidden_size))),
            b_hidden=np.zeros(int(hidden_size), dtype=float),
            w_embed=rng.normal(0.0, hidden_scale * 0.35, size=(int(embedding_size), int(hidden_size))),
            b_embed=np.zeros(int(embedding_size), dtype=float),
            # Zero heads preserve the legacy policy until a candidate receives
            # explicit timing lifecycle evidence.
            w_quality=np.zeros((1, int(embedding_size)), dtype=float),
            w_direction=np.zeros((1, int(embedding_size)), dtype=float),
            w_entry=np.zeros((1, int(embedding_size)), dtype=float),
            w_urgency=np.zeros((1, int(embedding_size)), dtype=float),
            w_uncertainty=np.zeros((1, int(embedding_size)), dtype=float),
            w_deceleration=np.zeros((1, int(embedding_size)), dtype=float),
            w_action=np.zeros((int(action_count), int(embedding_size)), dtype=float),
            b_action=np.zeros(int(action_count), dtype=float),
        )
        network.ensure_compatible()
        return network

    def configure_optimizer(self, config: Mapping[str, object] | None = None) -> None:
        """Apply the candidate TimingNet optimizer contract.

        The method is idempotent and intentionally does not reset weights or
        shadow state.  This is important when a checkpoint is resumed: a
        profile can tighten PID limits without silently restarting Adam's
        moment estimates.  Legacy artifacts remain in ``legacy`` when the
        feature is not explicitly enabled.
        """

        values = dict(config or {})
        self.ensure_compatible()
        self.swats_enabled = bool(values.get("timing_network_swats_enabled", values.get("swats_enabled", self.swats_enabled)))
        self.pid_enabled = bool(values.get("timing_network_pid_enabled", values.get("pid_enabled", self.pid_enabled)))
        if self.swats_enabled:
            if self.optimizer_phase not in {"adam", "sgd", "frozen"}:
                self.optimizer_phase = "adam"
        elif self.optimizer_phase not in {"legacy"}:
            # Turning the candidate switch off must not leave a frozen network
            # in an otherwise legacy run.  The m/v state is retained for audit.
            self.optimizer_phase = "legacy"
        self.adam_beta1 = float(np.clip(values.get("timing_network_adam_beta1", values.get("adam_beta1", self.adam_beta1)), 0.0, 0.9999))
        self.adam_beta2 = float(np.clip(values.get("timing_network_adam_beta2", values.get("adam_beta2", self.adam_beta2)), 0.0, 0.999999))
        self.adam_epsilon = float(np.clip(values.get("timing_network_adam_epsilon", values.get("adam_epsilon", self.adam_epsilon)), 1e-12, 1e-3))
        self.adamw_enabled = bool(values.get("timing_network_adamw_enabled", values.get("adamw_enabled", self.adamw_enabled)))
        self.adamw_weight_decay = float(np.clip(values.get("timing_network_adamw_weight_decay", values.get("adamw_weight_decay", self.adamw_weight_decay)), 0.0, 0.10))
        self.cosine_annealing_enabled = bool(values.get("timing_network_cosine_annealing_enabled", values.get("cosine_annealing_enabled", self.cosine_annealing_enabled)))
        self.cosine_min_factor = float(np.clip(values.get("timing_network_cosine_min_factor", values.get("cosine_min_factor", self.cosine_min_factor)), 0.01, 1.0))
        self.cosine_cycle_steps = int(np.clip(int(values.get("timing_network_cosine_cycle_steps", values.get("cosine_cycle_steps", self.cosine_cycle_steps))), 1, 4096))
        self.projection_beta = float(np.clip(values.get("timing_network_projection_beta", values.get("projection_beta", self.projection_beta)), 0.50, 0.9999))
        self.projection_tolerance = float(np.clip(values.get("timing_network_projection_tolerance", values.get("projection_tolerance", self.projection_tolerance)), 0.005, 1.0))
        self.switch_consecutive = int(np.clip(int(values.get("timing_network_switch_consecutive", values.get("switch_consecutive", self.switch_consecutive))), 1, 128))
        self.min_projection_steps = int(np.clip(int(values.get("timing_network_min_projection_steps", values.get("min_projection_steps", self.min_projection_steps))), 1, 512))
        self.target_alignment = float(np.clip(values.get("timing_network_target_alignment", values.get("target_alignment", self.target_alignment)), -1.0, 1.0))
        self.eta_min = float(np.clip(values.get("timing_network_sgd_eta_min", values.get("eta_min", self.eta_min)), 1e-8, 0.02))
        self.eta_max = float(np.clip(values.get("timing_network_sgd_eta_max", values.get("eta_max", self.eta_max)), self.eta_min, 0.02))
        self.eta_init_factor = float(np.clip(values.get("timing_network_sgd_eta_init_factor", values.get("eta_init_factor", self.eta_init_factor)), 0.05, 1.5))
        self.pid_kp = float(np.clip(values.get("timing_network_pid_kp", values.get("pid_kp", self.pid_kp)), 0.0, 2.0))
        self.pid_ki = float(np.clip(values.get("timing_network_pid_ki", values.get("pid_ki", self.pid_ki)), 0.0, 1.0))
        self.pid_kd = float(np.clip(values.get("timing_network_pid_kd", values.get("pid_kd", self.pid_kd)), 0.0, 1.0))
        self.pid_integral_limit = float(np.clip(values.get("timing_network_pid_integral_limit", values.get("pid_integral_limit", self.pid_integral_limit)), 0.05, 20.0))
        self.pid_derivative_ema = float(np.clip(values.get("timing_network_pid_derivative_ema", values.get("pid_derivative_ema", self.pid_derivative_ema)), 0.0, 0.99))
        self.pid_deadband = float(np.clip(values.get("timing_network_pid_deadband", values.get("pid_deadband", self.pid_deadband)), 0.0, 1.0))
        self.pid_r_target = float(np.clip(values.get("timing_network_pid_r_target", values.get("pid_r_target", self.pid_r_target)), -3.0, 3.0))
        self.pid_r_scale_floor = float(np.clip(values.get("timing_network_pid_r_scale_floor", values.get("pid_r_scale_floor", self.pid_r_scale_floor)), 1e-4, 3.0))
        self.pid_alignment_weight = float(np.clip(values.get("timing_network_pid_alignment_weight", values.get("pid_alignment_weight", self.pid_alignment_weight)), 0.0, 1.0))
        self.pid_r_weight = float(np.clip(values.get("timing_network_pid_r_weight", values.get("pid_r_weight", self.pid_r_weight)), 0.0, 1.0))
        weight_sum = self.pid_alignment_weight + self.pid_r_weight
        if weight_sum <= 1e-9:
            self.pid_alignment_weight, self.pid_r_weight = 0.70, 0.30
        else:
            self.pid_alignment_weight /= weight_sum
            self.pid_r_weight /= weight_sum
        self.pid_bad_steps_to_freeze = int(np.clip(int(values.get("timing_network_pid_bad_steps_to_freeze", values.get("pid_bad_steps_to_freeze", self.pid_bad_steps_to_freeze))), 1, 128))
        self.pid_recovery_steps = int(np.clip(int(values.get("timing_network_pid_recovery_steps", values.get("pid_recovery_steps", self.pid_recovery_steps))), 1, 128))
        self.pid_cooldown_steps = int(np.clip(int(values.get("timing_network_pid_cooldown_steps", values.get("pid_cooldown_steps", self.pid_cooldown_steps))), 0, 512))
        self.pid_fallback_to_adam = bool(values.get("timing_network_pid_fallback_to_adam", values.get("pid_fallback_to_adam", self.pid_fallback_to_adam)))
        self._ensure_optimizer_state()

    def configure_feature_contract(self, *, mtf_scaling_mode: str) -> None:
        """Bind the MTF encoding used by the causal timing labels.

        This is deliberately separate from the optimizer configuration: a
        TimingNet candidate may be trained offline against the native feature
        contract while the protected legacy checkpoint remains readable.
        """

        mode = str(mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        if mode not in {"legacy_m1_atr_v1", "native_tf_atr_v2"}:
            raise ValueError("TimingNetwork mtf_scaling_mode khong hop le")
        self.mtf_scaling_mode = mode

    def configure_sequence(self, config: Mapping[str, object] | None = None) -> None:
        """Set the causal M15 tensor width without altering the EA wire.

        Resizing retains the newest observations and never invents a missing
        bar.  A checkpoint without this field migrates to the default 32-step
        tensor through :meth:`ensure_compatible`.
        """

        values = dict(config or {})
        self.sequence_length = int(np.clip(int(values.get(
            "timing_network_sequence_length",
            values.get("sequence_length", self.sequence_length),
        )), 1, 128))
        self.ensure_compatible()

    def configure_deceleration(self, config: Mapping[str, object] | None = None) -> None:
        """Configure the causal, self-supervised trend-deceleration task.

        Unlike lifecycle credit, this task is labelled only when a later
        decision arrives.  It therefore remains valid during inference-only
        replays and does not inject realised P/L into a regime prediction.
        """

        values = dict(config or {})
        self.ensure_compatible()
        self.deceleration_enabled = bool(values.get(
            "timing_network_deceleration_enabled",
            values.get("deceleration_enabled", self.deceleration_enabled),
        ))
        self.deceleration_horizon = int(np.clip(int(values.get(
            "timing_network_deceleration_horizon",
            values.get("deceleration_horizon", self.deceleration_horizon),
        )), 1, 12))
        self.deceleration_min_trend_strength = float(np.clip(float(values.get(
            "timing_network_deceleration_min_trend_strength",
            values.get("deceleration_min_trend_strength", self.deceleration_min_trend_strength),
        )), 0.01, 1.0))
        self.deceleration_min_drop = float(np.clip(float(values.get(
            "timing_network_deceleration_min_drop",
            values.get("deceleration_min_drop", self.deceleration_min_drop),
        )), 0.005, 1.0))
        self.deceleration_update_scale = float(np.clip(float(values.get(
            "timing_network_deceleration_update_scale",
            values.get("deceleration_update_scale", self.deceleration_update_scale),
        )), 0.01, 1.0))
        self.deceleration_learning_rate = float(np.clip(float(values.get(
            "timing_network_deceleration_learning_rate",
            values.get("deceleration_learning_rate", self.deceleration_learning_rate),
        )), 1e-5, 0.05))
        self.deceleration_encoder_scale = float(np.clip(float(values.get(
            "timing_network_deceleration_encoder_scale",
            values.get("deceleration_encoder_scale", self.deceleration_encoder_scale),
        )), 0.0, 1.0))
        self.deceleration_l2 = float(np.clip(float(values.get(
            "timing_network_deceleration_l2",
            values.get("deceleration_l2", self.deceleration_l2),
        )), 0.0, 0.1))

    def configure_regime_transition(self, config: Mapping[str, object] | None = None) -> None:
        """Configure delayed multi-horizon regime labels without PnL leakage."""

        values = dict(config or {})
        self.ensure_compatible()
        self.regime_transition_enabled = bool(values.get(
            "timing_network_regime_transition_enabled",
            values.get("regime_transition_enabled", self.regime_transition_enabled),
        ))
        raw_horizons = values.get(
            "timing_network_regime_transition_horizons",
            values.get("regime_transition_horizons", self.regime_transition_horizons),
        )
        if isinstance(raw_horizons, str):
            raw_horizons = tuple(part.strip() for part in raw_horizons.split(",") if part.strip())
        try:
            horizons = tuple(sorted({int(value) for value in raw_horizons}))
        except TypeError:
            horizons = (3, 6, 12)
        self.regime_transition_horizons = tuple(value for value in horizons if 1 <= value <= 48) or (3,)
        self.regime_transition_update_scale = float(np.clip(float(values.get(
            "timing_network_regime_transition_update_scale",
            values.get("regime_transition_update_scale", self.regime_transition_update_scale),
        )), 0.0, 1.0))
        self.regime_transition_learning_rate = float(np.clip(float(values.get(
            "timing_network_regime_transition_learning_rate",
            values.get("regime_transition_learning_rate", self.regime_transition_learning_rate),
        )), 1e-5, 0.05))
        self.regime_transition_encoder_scale = float(np.clip(float(values.get(
            "timing_network_regime_transition_encoder_scale",
            values.get("regime_transition_encoder_scale", self.regime_transition_encoder_scale),
        )), 0.0, 1.0))
        self.regime_transition_l2 = float(np.clip(float(values.get(
            "timing_network_regime_transition_l2",
            values.get("regime_transition_l2", self.regime_transition_l2),
        )), 0.0, 0.1))
        self.ensure_compatible()

    def _parameter_arrays(self) -> list[tuple[str, np.ndarray]]:
        """Return parameters in the stable flattening order used in checkpoints."""

        return [
            ("w_quality", self.w_quality),
            ("w_direction", self.w_direction),
            ("w_entry", self.w_entry),
            ("w_urgency", self.w_urgency),
            ("w_uncertainty", self.w_uncertainty),
            ("w_deceleration", self.w_deceleration),
            ("w_action", self.w_action),
            ("b_action", self.b_action),
            ("w_embed", self.w_embed),
            ("b_embed", self.b_embed),
            ("b_quality", np.asarray([self.b_quality], dtype=float)),
            ("b_direction", np.asarray([self.b_direction], dtype=float)),
            ("b_entry", np.asarray([self.b_entry], dtype=float)),
            ("b_urgency", np.asarray([self.b_urgency], dtype=float)),
            ("b_uncertainty", np.asarray([self.b_uncertainty], dtype=float)),
            ("b_deceleration", np.asarray([self.b_deceleration], dtype=float)),
            ("w_regime_transition", self.w_regime_transition),
            ("b_regime_transition", self.b_regime_transition),
            ("w_in", self.w_in),
            ("w_rec", self.w_rec),
            ("b_hidden", self.b_hidden),
        ]

    def _ensure_optimizer_state(self) -> None:
        total = sum(int(value.size) for _, value in self._parameter_arrays())
        for name in ("shadow_adam_m", "shadow_adam_v"):
            value = np.asarray(getattr(self, name, None), dtype=float).reshape(-1) if getattr(self, name, None) is not None else np.zeros(0, dtype=float)
            if value.size != total or not np.isfinite(value).all():
                value = np.zeros(total, dtype=float)
            setattr(self, name, value)
        self.shadow_adam_step = max(0, int(getattr(self, "shadow_adam_step", 0) or 0))
        self.projection_count = max(0, int(getattr(self, "projection_count", 0) or 0))
        self.projection_stable_count = max(0, int(getattr(self, "projection_stable_count", 0) or 0))
        self.switch_count = max(0, int(getattr(self, "switch_count", 0) or 0))
        self.pid_signal_count = max(0, int(getattr(self, "pid_signal_count", 0) or 0))
        self.pid_good_steps = max(0, int(getattr(self, "pid_good_steps", 0) or 0))
        self.pid_bad_steps = max(0, int(getattr(self, "pid_bad_steps", 0) or 0))
        self.pid_cooldown = max(0, int(getattr(self, "pid_cooldown", 0) or 0))
        self.eta_sgd = float(np.clip(float(getattr(self, "eta_sgd", 0.0) or 0.0), 0.0, self.eta_max))
        self.optimizer_phase = str(getattr(self, "optimizer_phase", "legacy") or "legacy").lower()
        if self.optimizer_phase not in {"legacy", "adam", "sgd", "frozen"}:
            self.optimizer_phase = "legacy" if not self.swats_enabled else "adam"

    @staticmethod
    def _flatten_named(values: Mapping[str, np.ndarray], arrays: list[tuple[str, np.ndarray]]) -> np.ndarray:
        chunks: list[np.ndarray] = []
        for name, value in arrays:
            item = np.asarray(values[name], dtype=float).reshape(-1)
            if item.size != value.size:
                raise ValueError(f"optimizer gradient shape mismatch for {name}")
            chunks.append(item)
        return np.concatenate(chunks) if chunks else np.zeros(0, dtype=float)

    def _apply_flat_update(self, delta: np.ndarray) -> None:
        """Apply a flattened update with the historical per-block damping."""

        offset = 0
        for name, current in self._parameter_arrays():
            size = int(current.size)
            update = np.asarray(delta[offset : offset + size], dtype=float).reshape(current.shape)
            scale = 0.35 if name in _SLOW_PARAMETER_NAMES else 1.0
            new_value = np.asarray(current, dtype=float) + scale * update
            if name == "w_rec":
                new_value = np.clip(new_value, -2.0, 2.0)
            else:
                new_value = np.clip(new_value, -4.0, 4.0)
            if name.startswith("b_") and name not in {"b_action", "b_hidden", "b_embed", "b_regime_transition"}:
                setattr(self, name, float(new_value.reshape(-1)[0]))
            else:
                setattr(self, name, new_value)
            offset += size

    def _adam_cosine_factor(self) -> float:
        """Return the pre-switch Adam multiplier indexed by lifecycle updates.

        The schedule advances only when a settled lifecycle supplies a valid
        TimingNet gradient, never per MT5 tick.  This keeps its timebase
        compatible with SWATS projection statistics and sparse realised-R
        feedback.
        """

        if not self.cosine_annealing_enabled:
            return 1.0
        completed = max(0, int(self.shadow_adam_step) - 1)
        progress = min(1.0, completed / float(max(1, self.cosine_cycle_steps)))
        cosine = 0.5 * (1.0 + float(np.cos(np.pi * progress)))
        return float(self.cosine_min_factor + (1.0 - self.cosine_min_factor) * cosine)

    def _apply_adamw_decay(self, learning_rate: float) -> float:
        """Apply decoupled weight decay without contaminating SWATS projection.

        Projection keeps using the adaptive Adam direction only.  The decay
        term is applied separately to learned representation/policy weights,
        excluding the causal deceleration head whose BCE optimizer is
        intentionally independent from lifecycle realised-R credit.
        """

        if not self.adamw_enabled or self.adamw_weight_decay <= 0.0:
            return 0.0
        rate = max(0.0, float(learning_rate))
        if rate <= 0.0:
            return 0.0
        factor = float(np.clip(1.0 - rate * self.adamw_weight_decay, 0.0, 1.0))
        decayed = 0
        for name, current in self._parameter_arrays():
            if not name.startswith("w_") or name == "w_deceleration":
                continue
            new_value = np.clip(np.asarray(current, dtype=float) * factor, -4.0, 4.0)
            setattr(self, name, new_value)
            decayed += int(np.asarray(current).size)
        return float(decayed)

    def _pid_step(self, *, alignment: float, realized_r: float, gamma_hat: float) -> dict[str, float]:
        """Update the single composite PID actuator in log learning-rate space."""

        r = float(np.clip(realized_r, -3.0, 3.0))
        if self.pid_signal_count <= 0:
            self.pid_alignment_ema = alignment
            self.pid_r_ema = r
            self.pid_abs_r_ema = abs(r)
        else:
            beta = 0.90
            self.pid_alignment_ema = beta * self.pid_alignment_ema + (1.0 - beta) * alignment
            self.pid_r_ema = beta * self.pid_r_ema + (1.0 - beta) * r
            self.pid_abs_r_ema = beta * self.pid_abs_r_ema + (1.0 - beta) * abs(r - self.pid_r_ema)
        self.pid_signal_count += 1
        r_scale = max(self.pid_abs_r_ema, self.pid_r_scale_floor)
        alignment_error = float(np.clip(self.target_alignment - self.pid_alignment_ema, -1.0, 1.0))
        # Positive R is never allowed to inflate eta: it is useful as a
        # validation signal, while negative/insufficient R is a brake.
        r_error = float(np.clip(max(0.0, self.pid_r_target - self.pid_r_ema) / r_scale, 0.0, 1.0))
        error = float(np.clip(
            self.pid_alignment_weight * alignment_error + self.pid_r_weight * r_error,
            -1.0,
            1.0,
        ))
        if abs(error) < self.pid_deadband:
            error = 0.0
        previous_integral = self.pid_integral
        candidate_integral = float(np.clip(previous_integral + error, -self.pid_integral_limit, self.pid_integral_limit))
        raw_derivative = error - self.pid_previous_error
        self.pid_derivative = self.pid_derivative_ema * self.pid_derivative + (1.0 - self.pid_derivative_ema) * raw_derivative
        control = self.pid_kp * error + self.pid_ki * candidate_integral + self.pid_kd * self.pid_derivative
        eta_before = float(np.clip(self.eta_sgd or gamma_hat, self.eta_min, self.eta_max))
        log_before = float(np.log(max(eta_before, self.eta_min)))
        log_floor = float(np.log(self.eta_min))
        log_ceiling = float(np.log(self.eta_max))
        log_unclipped = log_before - control
        log_after = float(np.clip(log_unclipped, log_floor, log_ceiling))
        # Anti-windup: if the output is pinned and the new integral pushes it
        # farther into the pin, retain the previous integral instead.
        pushing_floor = log_after <= log_floor + 1e-12 and control > 0.0
        pushing_ceiling = log_after >= log_ceiling - 1e-12 and control < 0.0
        self.pid_integral = previous_integral if (pushing_floor or pushing_ceiling) else candidate_integral
        self.pid_previous_error = error
        self.eta_sgd = float(np.clip(np.exp(log_after), self.eta_min, self.eta_max))
        bad = alignment < (self.target_alignment * 0.80) or self.pid_r_ema < self.pid_r_target
        if bad:
            self.pid_bad_steps += 1
            self.pid_good_steps = 0
        else:
            self.pid_good_steps += 1
            self.pid_bad_steps = 0
        if self.pid_cooldown > 0:
            self.pid_cooldown -= 1
        if self.optimizer_phase == "sgd" and self.pid_bad_steps >= self.pid_bad_steps_to_freeze and self.pid_cooldown <= 0:
            self.optimizer_phase = "adam" if self.pid_fallback_to_adam else "frozen"
            self.eta_sgd = self.eta_min if self.optimizer_phase == "frozen" else float(np.clip(gamma_hat, self.eta_min, self.eta_max))
            self.pid_integral = 0.0
            self.projection_stable_count = 0
            self.pid_cooldown = self.pid_cooldown_steps
        elif self.optimizer_phase == "frozen" and self.pid_good_steps >= self.pid_recovery_steps:
            self.optimizer_phase = "sgd"
            self.eta_sgd = float(np.clip(self.eta_init_factor * max(gamma_hat, self.eta_min), self.eta_min, self.eta_max))
            self.pid_integral = 0.0
            self.pid_bad_steps = 0
            self.pid_cooldown = self.pid_cooldown_steps
        return {
            "pid_error": error,
            "pid_alignment_ema": float(self.pid_alignment_ema),
            "pid_r_ema": float(self.pid_r_ema),
            "pid_r_scale": float(r_scale),
            "pid_integral": float(self.pid_integral),
            "pid_derivative": float(self.pid_derivative),
            "pid_control": float(control),
            "eta_sgd": float(self.eta_sgd),
        }

    def _optimizer_update(
        self,
        gradients: Mapping[str, np.ndarray],
        *,
        learning_rate: float,
        realized_r: float,
        auxiliary: bool = False,
    ) -> dict[str, float]:
        """Apply either legacy SGD or the candidate PID-SWATS update.

        Existing gradients are *already* an ascent/update direction
        (``target - prediction``).  The canonical loss gradient is therefore
        ``g = -gradients``.  Keeping that sign explicit makes the SWATS
        projection formula match the paper while preserving the old target
        semantics exactly.
        """

        if not self.swats_enabled:
            arrays = self._parameter_arrays()
            flat = self._flatten_named(gradients, arrays)
            self._apply_flat_update(float(learning_rate) * flat)
            return {
                "optimizer_phase": 0.0,
                "optimizer_adam_step": float(self.shadow_adam_step),
                "optimizer_alignment": 0.0,
                "optimizer_gamma": 0.0,
                "optimizer_gamma_hat": 0.0,
                "optimizer_eta_sgd": 0.0,
                "optimizer_pid_error": 0.0,
                "optimizer_adam_learning_rate": float(learning_rate),
                "optimizer_adam_cosine_factor": 1.0,
                "optimizer_effective_learning_rate": float(learning_rate),
                "optimizer_weight_decay": 0.0,
                "optimizer_decayed_parameters": 0.0,
                "optimizer_auxiliary": 1.0 if auxiliary else 0.0,
            }

        self._ensure_optimizer_state()
        arrays = self._parameter_arrays()
        flat_update_direction = self._flatten_named(gradients, arrays)
        canonical_grad = -flat_update_direction
        self.shadow_adam_step += 1
        self.shadow_adam_m = self.adam_beta1 * self.shadow_adam_m + (1.0 - self.adam_beta1) * canonical_grad
        self.shadow_adam_v = self.adam_beta2 * self.shadow_adam_v + (1.0 - self.adam_beta2) * np.square(canonical_grad)
        m_hat = self.shadow_adam_m / max(1.0 - self.adam_beta1 ** self.shadow_adam_step, 1e-12)
        v_hat = self.shadow_adam_v / max(1.0 - self.adam_beta2 ** self.shadow_adam_step, 1e-12)
        adam_cosine_factor = self._adam_cosine_factor()
        adam_learning_rate = float(learning_rate) * adam_cosine_factor
        p_adam = -adam_learning_rate * m_hat / (np.sqrt(np.maximum(v_hat, 0.0)) + self.adam_epsilon)
        adam_norm = float(np.linalg.norm(p_adam))
        grad_norm = float(np.linalg.norm(canonical_grad))
        denominator = float(-np.dot(p_adam, canonical_grad))
        alignment = float(np.clip(
            np.dot(p_adam, -canonical_grad) / max(adam_norm * grad_norm, 1e-12),
            -1.0,
            1.0,
        )) if adam_norm > 1e-12 and grad_norm > 1e-12 else 0.0
        if auxiliary:
            # Dense self-supervised labels share the same optimizer state, so
            # they cannot silently become a hand-written weight writer.  They
            # deliberately do not advance SWATS projection/PID state: those
            # controls are calibrated from realised lifecycle R only.
            if self.optimizer_phase not in {"adam", "sgd", "frozen"}:
                self.optimizer_phase = "adam"
            if self.optimizer_phase == "frozen":
                actual_update = np.zeros_like(canonical_grad)
                effective_learning_rate = 0.0
            elif self.optimizer_phase == "sgd":
                eta = float(np.clip(self.eta_sgd or learning_rate, self.eta_min, self.eta_max))
                actual_update = -eta * canonical_grad
                effective_learning_rate = eta
            else:
                actual_update = p_adam
                effective_learning_rate = adam_learning_rate
            self._apply_flat_update(actual_update)
            decayed_parameters = self._apply_adamw_decay(effective_learning_rate)
            phase_code = {"legacy": 0.0, "adam": 1.0, "sgd": 2.0, "frozen": 3.0}.get(self.optimizer_phase, 0.0)
            return {
                "optimizer_phase": phase_code,
                "optimizer_adam_step": float(self.shadow_adam_step),
                "optimizer_alignment": float(alignment),
                "optimizer_gamma": 0.0,
                "optimizer_gamma_hat": float(self.projection_ema),
                "optimizer_projection_valid": 0.0,
                "optimizer_projection_stable_count": float(self.projection_stable_count),
                "optimizer_switch_count": float(self.switch_count),
                "optimizer_eta_sgd": float(self.eta_sgd),
                "optimizer_pid_error": 0.0,
                "optimizer_pid_alignment_ema": float(self.pid_alignment_ema),
                "optimizer_pid_r_ema": float(self.pid_r_ema),
                "optimizer_pid_r_scale": float(max(self.pid_abs_r_ema, self.pid_r_scale_floor)),
                "optimizer_pid_integral": float(self.pid_integral),
                "optimizer_pid_derivative": float(self.pid_derivative),
                "optimizer_pid_control": 0.0,
                "optimizer_adam_learning_rate": float(adam_learning_rate),
                "optimizer_adam_cosine_factor": float(adam_cosine_factor),
                "optimizer_effective_learning_rate": float(effective_learning_rate),
                "optimizer_weight_decay": float(self.adamw_weight_decay if self.adamw_enabled else 0.0),
                "optimizer_decayed_parameters": float(decayed_parameters),
                "optimizer_auxiliary": 1.0,
            }
        gamma = 0.0
        gamma_hat = float(self.projection_ema)
        projection_valid = denominator > self.adam_epsilon and adam_norm > self.adam_epsilon
        if projection_valid:
            gamma = float(np.clip((adam_norm * adam_norm) / denominator, self.eta_min, self.eta_max * 100.0))
            self.projection_count += 1
            beta = self.projection_beta
            self.projection_ema = beta * self.projection_ema + (1.0 - beta) * gamma
            gamma_hat = float(self.projection_ema / max(1.0 - beta ** self.projection_count, 1e-12))
            stable = (
                self.projection_count >= self.min_projection_steps
                and abs(gamma - gamma_hat) <= self.projection_tolerance * max(abs(gamma_hat), self.eta_min)
                and alignment >= self.target_alignment
            )
            self.projection_stable_count = self.projection_stable_count + 1 if stable else 0
        else:
            self.projection_stable_count = 0
        if self.optimizer_phase not in {"adam", "sgd", "frozen"}:
            self.optimizer_phase = "adam"
        if self.optimizer_phase == "adam" and self.projection_stable_count >= self.switch_consecutive:
            self.optimizer_phase = "sgd"
            self.switch_count += 1
            self.eta_sgd = float(np.clip(self.eta_init_factor * max(gamma_hat, self.eta_min), self.eta_min, self.eta_max))
            self.pid_integral = 0.0
            self.pid_previous_error = 0.0
            self.pid_derivative = 0.0
            self.pid_cooldown = self.pid_cooldown_steps

        pid_info: dict[str, float] = {}
        effective_learning_rate = adam_learning_rate
        if self.optimizer_phase == "sgd":
            if self.eta_sgd <= 0.0:
                self.eta_sgd = float(np.clip(self.eta_init_factor * max(gamma_hat, learning_rate), self.eta_min, self.eta_max))
            if self.pid_enabled and projection_valid:
                pid_info = self._pid_step(alignment=alignment, realized_r=realized_r, gamma_hat=max(gamma_hat, learning_rate))
            actual_update = p_adam if self.optimizer_phase == "adam" else -self.eta_sgd * canonical_grad
            effective_learning_rate = adam_learning_rate if self.optimizer_phase == "adam" else float(self.eta_sgd)
        elif self.optimizer_phase == "frozen":
            # Frozen is an intentional safe state, not a failed lifecycle.
            # Shadow Adam still advances above so recovery is data-driven.
            actual_update = np.zeros_like(canonical_grad)
            effective_learning_rate = 0.0
            if self.pid_enabled and projection_valid:
                pid_info = self._pid_step(alignment=alignment, realized_r=realized_r, gamma_hat=max(gamma_hat, learning_rate))
        else:
            actual_update = p_adam
        self._apply_flat_update(actual_update)
        decayed_parameters = self._apply_adamw_decay(effective_learning_rate)
        phase_code = {"legacy": 0.0, "adam": 1.0, "sgd": 2.0, "frozen": 3.0}.get(self.optimizer_phase, 0.0)
        return {
            "optimizer_phase": phase_code,
            "optimizer_adam_step": float(self.shadow_adam_step),
            "optimizer_alignment": float(alignment),
            "optimizer_gamma": float(gamma),
            "optimizer_gamma_hat": float(gamma_hat),
            "optimizer_projection_valid": 1.0 if projection_valid else 0.0,
            "optimizer_projection_stable_count": float(self.projection_stable_count),
            "optimizer_switch_count": float(self.switch_count),
            "optimizer_eta_sgd": float(self.eta_sgd),
            "optimizer_pid_error": float(pid_info.get("pid_error", 0.0)),
            "optimizer_pid_alignment_ema": float(pid_info.get("pid_alignment_ema", self.pid_alignment_ema)),
            "optimizer_pid_r_ema": float(pid_info.get("pid_r_ema", self.pid_r_ema)),
            "optimizer_pid_r_scale": float(pid_info.get("pid_r_scale", max(self.pid_abs_r_ema, self.pid_r_scale_floor))),
            "optimizer_pid_integral": float(pid_info.get("pid_integral", self.pid_integral)),
            "optimizer_pid_derivative": float(pid_info.get("pid_derivative", self.pid_derivative)),
            "optimizer_pid_control": float(pid_info.get("pid_control", 0.0)),
            "optimizer_adam_learning_rate": float(adam_learning_rate),
            "optimizer_adam_cosine_factor": float(adam_cosine_factor),
            "optimizer_effective_learning_rate": float(effective_learning_rate),
            "optimizer_weight_decay": float(self.adamw_weight_decay if self.adamw_enabled else 0.0),
            "optimizer_auxiliary": 0.0,
            "optimizer_decayed_parameters": float(decayed_parameters),
        }

    def ensure_compatible(self) -> None:
        """Migrate old/missing arrays without changing the bridge contract."""

        self.input_size = max(1, int(getattr(self, "input_size", TIMING_NETWORK_INPUT_SIZE) or TIMING_NETWORK_INPUT_SIZE))
        self.hidden_size = max(4, int(getattr(self, "hidden_size", 32) or 32))
        self.embedding_size = max(4, int(getattr(self, "embedding_size", 16) or 16))
        self.action_count = max(1, int(getattr(self, "action_count", TIMING_NETWORK_ACTION_COUNT) or TIMING_NETWORK_ACTION_COUNT))
        self.learning_rate = float(np.clip(float(getattr(self, "learning_rate", 0.0005) or 0.0005), 1e-6, 0.02))
        self.memory_decay = float(np.clip(float(getattr(self, "memory_decay", 0.85) or 0.85), 0.1, 0.98))
        self.bptt_window = int(np.clip(int(getattr(self, "bptt_window", 24) or 24), 1, 128))
        self.sequence_length = int(np.clip(
            int(getattr(self, "sequence_length", 32) or 32), 1, 128
        ))
        self.eligibility_decay = float(np.clip(float(getattr(self, "eligibility_decay", 0.90) or 0.90), 0.5, 0.999))
        self.gradient_clip = float(np.clip(float(getattr(self, "gradient_clip", 1.0) or 1.0), 0.05, 10.0))
        mode = str(getattr(self, "mtf_scaling_mode", "legacy_m1_atr_v1") or "legacy_m1_atr_v1").strip().lower()
        self.mtf_scaling_mode = mode if mode in {"legacy_m1_atr_v1", "native_tf_atr_v2"} else "legacy_m1_atr_v1"

        def matrix(name: str, shape: tuple[int, ...], scale: float = 0.0) -> np.ndarray:
            current = np.asarray(getattr(self, name, None), dtype=float) if getattr(self, name, None) is not None else np.zeros((0,), dtype=float)
            if current.shape == shape and np.isfinite(current).all():
                return current
            result = np.zeros(shape, dtype=float)
            if current.size:
                slices = tuple(slice(0, min(old, new)) for old, new in zip(current.shape, shape))
                if current.ndim == len(shape) and len(slices) == len(current.shape):
                    result[slices] = current[slices]
            if scale and not current.size:
                rng = np.random.default_rng((self.hidden_size * 1009 + self.embedding_size * 917 + self.input_size) & 0x7FFFFFFF)
                result = rng.normal(0.0, scale, size=shape)
            return result

        self.w_in = matrix("w_in", (self.hidden_size, self.input_size), 1.0 / np.sqrt(float(self.input_size)))
        self.w_rec = matrix("w_rec", (self.hidden_size, self.hidden_size), 0.03)
        self.b_hidden = matrix("b_hidden", (self.hidden_size,))
        self.w_embed = matrix("w_embed", (self.embedding_size, self.hidden_size), 0.05)
        self.b_embed = matrix("b_embed", (self.embedding_size,))
        for name in ("w_quality", "w_direction", "w_entry", "w_urgency", "w_uncertainty", "w_deceleration"):
            setattr(self, name, matrix(name, (1, self.embedding_size)))
        self.w_action = matrix("w_action", (self.action_count, self.embedding_size))
        self.b_action = matrix("b_action", (self.action_count,))
        self.b_deceleration = float(np.clip(
            float(getattr(self, "b_deceleration", 0.0) or 0.0), -4.0, 4.0
        ))
        self.eligibility_in = matrix("eligibility_in", (self.hidden_size, self.hidden_size, self.input_size))
        self.eligibility_rec = matrix("eligibility_rec", (self.hidden_size, self.hidden_size, self.hidden_size))
        self.eligibility_bias = matrix("eligibility_bias", (self.hidden_size, self.hidden_size))
        self.eligibility_steps = int(np.clip(int(getattr(self, "eligibility_steps", 0) or 0), 0, self.bptt_window))
        state = np.asarray(getattr(self, "state", None), dtype=float).reshape(-1) if getattr(self, "state", None) is not None else np.zeros(0, dtype=float)
        self.state = np.zeros(self.hidden_size, dtype=float)
        if state.size:
            self.state[: min(state.size, self.hidden_size)] = state[: self.hidden_size]
        raw_sequence = np.asarray(
            getattr(self, "sequence_buffer", None), dtype=float
        ) if getattr(self, "sequence_buffer", None) is not None else np.zeros((0, 0), dtype=float)
        sequence = np.zeros((self.sequence_length, self.input_size), dtype=float)
        if raw_sequence.ndim == 2 and raw_sequence.size:
            rows = min(raw_sequence.shape[0], self.sequence_length)
            cols = min(raw_sequence.shape[1], self.input_size)
            sequence[-rows:, :cols] = raw_sequence[-rows:, :cols]
        self.sequence_buffer = np.nan_to_num(sequence, nan=0.0, posinf=2.0, neginf=-2.0)
        inferred_valid = int(np.count_nonzero(np.any(np.abs(self.sequence_buffer) > 0.0, axis=1)))
        self.sequence_valid_length = int(np.clip(
            int(getattr(self, "sequence_valid_length", inferred_valid) or inferred_valid),
            0,
            self.sequence_length,
        ))
        self.update_count = max(0, int(getattr(self, "update_count", 0) or 0))
        if not isinstance(getattr(self, "last_update", None), dict):
            self.last_update = {}
        self.deceleration_enabled = bool(getattr(self, "deceleration_enabled", False))
        self.deceleration_horizon = int(np.clip(int(getattr(self, "deceleration_horizon", 3) or 3), 1, 12))
        self.deceleration_min_trend_strength = float(np.clip(
            float(getattr(self, "deceleration_min_trend_strength", 0.18) or 0.18), 0.01, 1.0
        ))
        self.deceleration_min_drop = float(np.clip(
            float(getattr(self, "deceleration_min_drop", 0.10) or 0.10), 0.005, 1.0
        ))
        self.deceleration_update_scale = float(np.clip(
            float(getattr(self, "deceleration_update_scale", 0.25) or 0.25), 0.01, 1.0
        ))
        self.deceleration_learning_rate = float(np.clip(
            float(getattr(self, "deceleration_learning_rate", 0.01) or 0.01), 1e-5, 0.05
        ))
        self.deceleration_encoder_scale = float(np.clip(
            float(getattr(self, "deceleration_encoder_scale", 0.10) or 0.10), 0.0, 1.0
        ))
        self.deceleration_l2 = float(np.clip(
            float(getattr(self, "deceleration_l2", 1e-4) or 0.0), 0.0, 0.1
        ))
        pending = getattr(self, "deceleration_pending", [])
        self.deceleration_pending = [
            dict(item) for item in pending
            if isinstance(item, Mapping)
        ][-max(1, self.deceleration_horizon):]
        self.deceleration_observation_count = max(0, int(getattr(self, "deceleration_observation_count", 0) or 0))
        self.deceleration_update_count = max(0, int(getattr(self, "deceleration_update_count", 0) or 0))
        self.deceleration_positive_count = max(0, int(getattr(self, "deceleration_positive_count", 0) or 0))
        self.deceleration_last_target = float(np.clip(float(getattr(self, "deceleration_last_target", 0.0) or 0.0), 0.0, 1.0))
        self.deceleration_last_error = float(np.clip(float(getattr(self, "deceleration_last_error", 0.0) or 0.0), -1.0, 1.0))
        self.regime_transition_enabled = bool(getattr(self, "regime_transition_enabled", False))
        raw_horizons = getattr(self, "regime_transition_horizons", (3, 6, 12)) or (3, 6, 12)
        try:
            self.regime_transition_horizons = tuple(sorted({int(value) for value in raw_horizons if 1 <= int(value) <= 48})) or (3,)
        except (TypeError, ValueError):
            self.regime_transition_horizons = (3, 6, 12)
        self.regime_transition_update_scale = float(np.clip(float(getattr(self, "regime_transition_update_scale", 0.25) or 0.25), 0.0, 1.0))
        self.regime_transition_learning_rate = float(np.clip(float(getattr(self, "regime_transition_learning_rate", 0.01) or 0.01), 1e-5, 0.05))
        self.regime_transition_encoder_scale = float(np.clip(float(getattr(self, "regime_transition_encoder_scale", 0.08) or 0.08), 0.0, 1.0))
        self.regime_transition_l2 = float(np.clip(float(getattr(self, "regime_transition_l2", 1e-4) or 0.0), 0.0, 0.1))
        horizon_count = len(self.regime_transition_horizons)
        class_count = len(REGIME_TRANSITION_LABELS)
        self.w_regime_transition = matrix("w_regime_transition", (horizon_count, class_count, self.embedding_size))
        self.b_regime_transition = matrix("b_regime_transition", (horizon_count, class_count))
        pending_transition = getattr(self, "regime_transition_pending", [])
        self.regime_transition_pending = [dict(item) for item in pending_transition if isinstance(item, Mapping)][-max(1, max(self.regime_transition_horizons) * horizon_count):]
        self.regime_transition_observation_count = max(0, int(getattr(self, "regime_transition_observation_count", 0) or 0))
        self.regime_transition_update_count = max(0, int(getattr(self, "regime_transition_update_count", 0) or 0))
        self.regime_transition_last_target = str(getattr(self, "regime_transition_last_target", "") or "")
        self.regime_transition_last_error = float(np.clip(float(getattr(self, "regime_transition_last_error", 0.0) or 0.0), -1.0, 1.0))
        raw_calibration = getattr(self, "regime_transition_calibration_events", [])
        calibration: list[tuple[float, int]] = []
        if isinstance(raw_calibration, (list, tuple)):
            for item in raw_calibration[-4096:]:
                try:
                    probability, target = item
                    probability = float(np.clip(float(probability), 0.0, 1.0))
                    target = 1 if int(target) else 0
                except (TypeError, ValueError):
                    continue
                calibration.append((probability, target))
        self.regime_transition_calibration_events = calibration
        self.regime_transition_calibration_count = max(
            len(calibration), int(getattr(self, "regime_transition_calibration_count", 0) or 0)
        )
        self.regime_transition_brier_ema = float(np.clip(
            float(getattr(self, "regime_transition_brier_ema", 0.0) or 0.0), 0.0, 1.0
        ))
        self.swats_enabled = bool(getattr(self, "swats_enabled", False))
        self.pid_enabled = bool(getattr(self, "pid_enabled", False))
        self.optimizer_phase = str(getattr(self, "optimizer_phase", "legacy") or "legacy").lower()
        if self.optimizer_phase not in {"legacy", "adam", "sgd", "frozen"}:
            self.optimizer_phase = "adam" if self.swats_enabled else "legacy"
        self.adam_beta1 = float(np.clip(float(getattr(self, "adam_beta1", 0.90) or 0.90), 0.0, 0.9999))
        self.adam_beta2 = float(np.clip(float(getattr(self, "adam_beta2", 0.999) or 0.999), 0.0, 0.999999))
        self.adam_epsilon = float(np.clip(float(getattr(self, "adam_epsilon", 1e-8) or 1e-8), 1e-12, 1e-3))
        self.adamw_enabled = bool(getattr(self, "adamw_enabled", False))
        self.adamw_weight_decay = float(np.clip(float(getattr(self, "adamw_weight_decay", 0.0) or 0.0), 0.0, 0.10))
        self.cosine_annealing_enabled = bool(getattr(self, "cosine_annealing_enabled", False))
        self.cosine_min_factor = float(np.clip(float(getattr(self, "cosine_min_factor", 1.0) or 1.0), 0.01, 1.0))
        self.cosine_cycle_steps = int(np.clip(int(getattr(self, "cosine_cycle_steps", 128) or 128), 1, 4096))
        self.projection_beta = float(np.clip(float(getattr(self, "projection_beta", 0.90) or 0.90), 0.50, 0.9999))
        self.projection_tolerance = float(np.clip(float(getattr(self, "projection_tolerance", 0.10) or 0.10), 0.005, 1.0))
        self.switch_consecutive = int(np.clip(int(getattr(self, "switch_consecutive", 6) or 6), 1, 128))
        self.min_projection_steps = int(np.clip(int(getattr(self, "min_projection_steps", 12) or 12), 1, 512))
        self.target_alignment = float(np.clip(float(getattr(self, "target_alignment", 0.90) or 0.90), -1.0, 1.0))
        self.eta_min = float(np.clip(float(getattr(self, "eta_min", 1e-6) or 1e-6), 1e-8, 0.02))
        self.eta_max = float(np.clip(float(getattr(self, "eta_max", 0.004) or 0.004), self.eta_min, 0.02))
        self.eta_init_factor = float(np.clip(float(getattr(self, "eta_init_factor", 0.90) or 0.90), 0.05, 1.5))
        self.pid_kp = float(np.clip(float(getattr(self, "pid_kp", 0.08) or 0.08), 0.0, 2.0))
        self.pid_ki = float(np.clip(float(getattr(self, "pid_ki", 0.01) or 0.01), 0.0, 1.0))
        self.pid_kd = float(np.clip(float(getattr(self, "pid_kd", 0.02) or 0.02), 0.0, 1.0))
        self.pid_integral_limit = float(np.clip(float(getattr(self, "pid_integral_limit", 2.0) or 2.0), 0.05, 20.0))
        self.pid_derivative_ema = float(np.clip(float(getattr(self, "pid_derivative_ema", 0.80) or 0.80), 0.0, 0.99))
        self.pid_deadband = float(np.clip(float(getattr(self, "pid_deadband", 0.02) or 0.02), 0.0, 1.0))
        self.pid_r_target = float(np.clip(float(getattr(self, "pid_r_target", 0.0) or 0.0), -3.0, 3.0))
        self.pid_r_scale_floor = float(np.clip(float(getattr(self, "pid_r_scale_floor", 0.25) or 0.25), 1e-4, 3.0))
        self.pid_alignment_weight = float(np.clip(float(getattr(self, "pid_alignment_weight", 0.70) or 0.70), 0.0, 1.0))
        self.pid_r_weight = float(np.clip(float(getattr(self, "pid_r_weight", 0.30) or 0.30), 0.0, 1.0))
        self.pid_bad_steps_to_freeze = int(np.clip(int(getattr(self, "pid_bad_steps_to_freeze", 4) or 4), 1, 128))
        self.pid_recovery_steps = int(np.clip(int(getattr(self, "pid_recovery_steps", 3) or 3), 1, 128))
        self.pid_cooldown_steps = int(np.clip(int(getattr(self, "pid_cooldown_steps", 8) or 8), 0, 512))
        self.pid_fallback_to_adam = bool(getattr(self, "pid_fallback_to_adam", False))
        self.shadow_adam_step = max(0, int(getattr(self, "shadow_adam_step", 0) or 0))
        self.projection_ema = float(getattr(self, "projection_ema", 0.0) or 0.0)
        self.projection_count = max(0, int(getattr(self, "projection_count", 0) or 0))
        self.projection_stable_count = max(0, int(getattr(self, "projection_stable_count", 0) or 0))
        self.switch_count = max(0, int(getattr(self, "switch_count", 0) or 0))
        self.pid_integral = float(getattr(self, "pid_integral", 0.0) or 0.0)
        self.pid_previous_error = float(getattr(self, "pid_previous_error", 0.0) or 0.0)
        self.pid_derivative = float(getattr(self, "pid_derivative", 0.0) or 0.0)
        self.pid_alignment_ema = float(getattr(self, "pid_alignment_ema", 0.0) or 0.0)
        self.pid_r_ema = float(getattr(self, "pid_r_ema", 0.0) or 0.0)
        self.pid_abs_r_ema = float(getattr(self, "pid_abs_r_ema", 0.0) or 0.0)
        self.pid_signal_count = max(0, int(getattr(self, "pid_signal_count", 0) or 0))
        self.pid_good_steps = max(0, int(getattr(self, "pid_good_steps", 0) or 0))
        self.pid_bad_steps = max(0, int(getattr(self, "pid_bad_steps", 0) or 0))
        self.pid_cooldown = max(0, int(getattr(self, "pid_cooldown", 0) or 0))
        self.eta_sgd = float(np.clip(float(getattr(self, "eta_sgd", 0.0) or 0.0), 0.0, self.eta_max))
        self._ensure_optimizer_state()

    def copy(self) -> "TimingNetwork":
        self.ensure_compatible()
        copied = TimingNetwork(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            embedding_size=self.embedding_size,
            action_count=self.action_count,
            memory_decay=self.memory_decay,
            learning_rate=self.learning_rate,
            bptt_window=self.bptt_window,
            sequence_length=self.sequence_length,
            eligibility_decay=self.eligibility_decay,
            gradient_clip=self.gradient_clip,
            mtf_scaling_mode=self.mtf_scaling_mode,
            state=self.state.copy(),
            sequence_buffer=self.sequence_buffer.copy(),
            sequence_valid_length=int(self.sequence_valid_length),
            w_in=self.w_in.copy(),
            w_rec=self.w_rec.copy(),
            b_hidden=self.b_hidden.copy(),
            w_embed=self.w_embed.copy(),
            b_embed=self.b_embed.copy(),
            w_quality=self.w_quality.copy(),
            b_quality=float(self.b_quality),
            w_direction=self.w_direction.copy(),
            b_direction=float(self.b_direction),
            w_entry=self.w_entry.copy(),
            b_entry=float(self.b_entry),
            w_urgency=self.w_urgency.copy(),
            b_urgency=float(self.b_urgency),
            w_uncertainty=self.w_uncertainty.copy(),
            b_uncertainty=float(self.b_uncertainty),
            w_deceleration=self.w_deceleration.copy(),
            b_deceleration=float(self.b_deceleration),
            w_action=self.w_action.copy(),
            b_action=self.b_action.copy(),
            eligibility_in=self.eligibility_in.copy(),
            eligibility_rec=self.eligibility_rec.copy(),
            eligibility_bias=self.eligibility_bias.copy(),
            eligibility_steps=int(self.eligibility_steps),
            update_count=int(self.update_count),
            last_update=dict(self.last_update),
            deceleration_enabled=bool(self.deceleration_enabled),
            deceleration_horizon=int(self.deceleration_horizon),
            deceleration_min_trend_strength=float(self.deceleration_min_trend_strength),
            deceleration_min_drop=float(self.deceleration_min_drop),
            deceleration_update_scale=float(self.deceleration_update_scale),
            deceleration_learning_rate=float(self.deceleration_learning_rate),
            deceleration_encoder_scale=float(self.deceleration_encoder_scale),
            deceleration_l2=float(self.deceleration_l2),
            deceleration_pending=[
                {
                    str(key): value.copy() if isinstance(value, np.ndarray) else value
                    for key, value in dict(item).items()
                }
                for item in self.deceleration_pending
            ],
            deceleration_observation_count=int(self.deceleration_observation_count),
            deceleration_update_count=int(self.deceleration_update_count),
            deceleration_positive_count=int(self.deceleration_positive_count),
            deceleration_last_target=float(self.deceleration_last_target),
            deceleration_last_error=float(self.deceleration_last_error),
            swats_enabled=bool(self.swats_enabled),
            pid_enabled=bool(self.pid_enabled),
            optimizer_phase=str(self.optimizer_phase),
            shadow_adam_m=self.shadow_adam_m.copy(),
            shadow_adam_v=self.shadow_adam_v.copy(),
            shadow_adam_step=int(self.shadow_adam_step),
            adam_beta1=float(self.adam_beta1),
            adam_beta2=float(self.adam_beta2),
            adam_epsilon=float(self.adam_epsilon),
            adamw_enabled=bool(self.adamw_enabled),
            adamw_weight_decay=float(self.adamw_weight_decay),
            cosine_annealing_enabled=bool(self.cosine_annealing_enabled),
            cosine_min_factor=float(self.cosine_min_factor),
            cosine_cycle_steps=int(self.cosine_cycle_steps),
            projection_beta=float(self.projection_beta),
            projection_tolerance=float(self.projection_tolerance),
            switch_consecutive=int(self.switch_consecutive),
            min_projection_steps=int(self.min_projection_steps),
            target_alignment=float(self.target_alignment),
            eta_sgd=float(self.eta_sgd),
            eta_min=float(self.eta_min),
            eta_max=float(self.eta_max),
            eta_init_factor=float(self.eta_init_factor),
            projection_ema=float(self.projection_ema),
            projection_count=int(self.projection_count),
            projection_stable_count=int(self.projection_stable_count),
            switch_count=int(self.switch_count),
            pid_integral=float(self.pid_integral),
            pid_previous_error=float(self.pid_previous_error),
            pid_derivative=float(self.pid_derivative),
            pid_alignment_ema=float(self.pid_alignment_ema),
            pid_r_ema=float(self.pid_r_ema),
            pid_abs_r_ema=float(self.pid_abs_r_ema),
            pid_signal_count=int(self.pid_signal_count),
            pid_good_steps=int(self.pid_good_steps),
            pid_bad_steps=int(self.pid_bad_steps),
            pid_cooldown=int(self.pid_cooldown),
            pid_kp=float(self.pid_kp),
            pid_ki=float(self.pid_ki),
            pid_kd=float(self.pid_kd),
            pid_integral_limit=float(self.pid_integral_limit),
            pid_derivative_ema=float(self.pid_derivative_ema),
            pid_deadband=float(self.pid_deadband),
            pid_r_target=float(self.pid_r_target),
            pid_r_scale_floor=float(self.pid_r_scale_floor),
            pid_alignment_weight=float(self.pid_alignment_weight),
            pid_r_weight=float(self.pid_r_weight),
            pid_bad_steps_to_freeze=int(self.pid_bad_steps_to_freeze),
            pid_recovery_steps=int(self.pid_recovery_steps),
            pid_cooldown_steps=int(self.pid_cooldown_steps),
            pid_fallback_to_adam=bool(self.pid_fallback_to_adam),
        )
        # Transition fields were introduced after the original constructor
        # contract.  Assigning them here keeps old pickle copies compatible
        # while preserving a trained candidate's delayed-label queue.
        copied.regime_transition_enabled = bool(self.regime_transition_enabled)
        copied.regime_transition_horizons = tuple(self.regime_transition_horizons)
        copied.regime_transition_update_scale = float(self.regime_transition_update_scale)
        copied.regime_transition_learning_rate = float(self.regime_transition_learning_rate)
        copied.regime_transition_encoder_scale = float(self.regime_transition_encoder_scale)
        copied.regime_transition_l2 = float(self.regime_transition_l2)
        copied.w_regime_transition = self.w_regime_transition.copy()
        copied.b_regime_transition = self.b_regime_transition.copy()
        copied.regime_transition_pending = [
            {str(key): value.copy() if isinstance(value, np.ndarray) else value for key, value in dict(item).items()}
            for item in self.regime_transition_pending
        ]
        copied.regime_transition_observation_count = int(self.regime_transition_observation_count)
        copied.regime_transition_update_count = int(self.regime_transition_update_count)
        copied.regime_transition_last_target = str(self.regime_transition_last_target)
        copied.regime_transition_last_error = float(self.regime_transition_last_error)
        copied.regime_transition_calibration_events = list(self.regime_transition_calibration_events)
        copied.regime_transition_calibration_count = int(self.regime_transition_calibration_count)
        copied.regime_transition_brier_ema = float(self.regime_transition_brier_ema)
        copied.ensure_compatible()
        return copied

    @staticmethod
    def _sigmoid(value: float) -> float:
        return float(1.0 / (1.0 + np.exp(-np.clip(float(value), -30.0, 30.0))))

    def _deceleration_update_from_future_score(self, future_trend_score: float) -> dict[str, float]:
        """Settle eligible self-supervised predictions with a later observation.

        The pending trace was captured at time ``t`` and this method runs only
        when the feature vector at ``t + horizon`` is available.  No trade
        outcome, lifecycle state, or future value is read by the original
        decision.  The auxiliary update uses the same AdamW/SWATS parameter
        writer as TimingNet, while keeping its PID/SWATS transition counters
        lifecycle-only.
        """

        if not self.deceleration_enabled:
            return {"updated": 0.0}
        current_step = int(self.deceleration_observation_count)
        ready: list[dict[str, object]] = []
        retained: list[dict[str, object]] = []
        for item in self.deceleration_pending:
            try:
                age = current_step - int(item.get("observation_step", current_step))
            except (AttributeError, TypeError, ValueError):
                continue
            if age >= self.deceleration_horizon:
                ready.append(dict(item))
            else:
                retained.append(dict(item))
        self.deceleration_pending = retained[-max(1, self.deceleration_horizon):]
        result: dict[str, float] = {"updated": 0.0, "settled": float(len(ready))}
        for trace in ready:
            try:
                start_score = float(np.clip(float(trace.get("trend_score", 0.0)), -1.0, 1.0))
                embedding = np.asarray(trace.get("embedding", []), dtype=float).reshape(-1)
                hidden = np.asarray(trace.get("hidden", []), dtype=float).reshape(-1)
                candidate = np.asarray(trace.get("candidate", []), dtype=float).reshape(-1)
                features = np.asarray(trace.get("features", []), dtype=float).reshape(-1)
                state_before = np.asarray(trace.get("state_before", []), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                continue
            if (
                abs(start_score) < self.deceleration_min_trend_strength
                or embedding.size != self.embedding_size
                or hidden.size != self.hidden_size
                or candidate.size != self.hidden_size
                or features.size != self.input_size
                or state_before.size != self.hidden_size
                or not all(np.isfinite(value).all() for value in (embedding, hidden, candidate, features, state_before))
            ):
                continue

            magnitude_drop = abs(start_score) - abs(float(future_trend_score))
            target = 1.0 if magnitude_drop >= self.deceleration_min_drop else 0.0
            prediction = self._sigmoid(float(np.asarray(self.w_deceleration @ embedding + self.b_deceleration).reshape(-1)[0]))
            error = float(np.clip(target - prediction, -1.0, 1.0))
            # Binary cross-entropy is the calibrated objective for a
            # probability head: d logit = target - prediction.  The old
            # squared-sigmoid derivative multiplied this by p(1-p), and its
            # tiny shared lifecycle LR left the head numerically fixed at 0.5
            # after hundreds of causal labels.
            d_deceleration = error
            encoder_scale = float(self.deceleration_encoder_scale)
            d_embedding = encoder_scale * d_deceleration * self.w_deceleration.reshape(-1)
            d_embedding_pre = d_embedding * (1.0 - np.square(np.clip(embedding, -1.0, 1.0)))
            d_hidden = self.w_embed.T @ d_embedding_pre

            arrays = self._parameter_arrays()
            gradients = {name: np.zeros_like(value, dtype=float) for name, value in arrays}
            gradients["w_deceleration"] = (
                d_deceleration * embedding.reshape(1, -1)
                - float(self.deceleration_l2) * self.w_deceleration
            )
            gradients["b_deceleration"] = np.asarray([d_deceleration], dtype=float)
            gradients["w_embed"] = np.outer(d_embedding_pre, hidden)
            gradients["b_embed"] = d_embedding_pre

            try:
                eligibility_in = np.asarray(trace.get("eligibility_in", []), dtype=float)
                eligibility_rec = np.asarray(trace.get("eligibility_rec", []), dtype=float)
                eligibility_bias = np.asarray(trace.get("eligibility_bias", []), dtype=float)
            except (TypeError, ValueError):
                eligibility_in = np.zeros(0, dtype=float)
                eligibility_rec = np.zeros(0, dtype=float)
                eligibility_bias = np.zeros(0, dtype=float)
            expected_in = (self.hidden_size, self.hidden_size, self.input_size)
            expected_rec = (self.hidden_size, self.hidden_size, self.hidden_size)
            expected_bias = (self.hidden_size, self.hidden_size)
            if (
                eligibility_in.shape != expected_in
                or eligibility_rec.shape != expected_rec
                or eligibility_bias.shape != expected_bias
                or not all(np.isfinite(value).all() for value in (eligibility_in, eligibility_rec, eligibility_bias))
            ):
                local_factor = (1.0 - self.memory_decay) * (1.0 - np.square(np.clip(candidate, -1.0, 1.0)))
                eligibility_in = np.zeros(expected_in, dtype=float)
                eligibility_rec = np.zeros(expected_rec, dtype=float)
                eligibility_bias = np.zeros(expected_bias, dtype=float)
                row_index = np.arange(self.hidden_size)
                eligibility_in[row_index, row_index, :] = local_factor[:, None] * features[None, :]
                eligibility_rec[row_index, row_index, :] = local_factor[:, None] * state_before[None, :]
                eligibility_bias[row_index, row_index] = local_factor
            gradients["w_in"] = np.einsum("k,kji->ji", d_hidden, eligibility_in)
            gradients["w_rec"] = np.einsum("k,kji->ji", d_hidden, eligibility_rec)
            gradients["b_hidden"] = np.einsum("k,kj->j", d_hidden, eligibility_bias)

            gradient_norm = float(np.sqrt(sum(float(np.sum(np.square(value))) for value in gradients.values())))
            clip_factor = float(min(1.0, self.gradient_clip / max(gradient_norm, 1e-12)))
            if clip_factor < 1.0:
                gradients = {name: value * clip_factor for name, value in gradients.items()}
            learning_rate = float(np.clip(
                self.deceleration_learning_rate * self.deceleration_update_scale,
                1e-5,
                0.05,
            ))
            optimizer = self._optimizer_update(
                gradients,
                learning_rate=learning_rate,
                realized_r=float(self.pid_r_ema),
                auxiliary=True,
            )
            self.deceleration_update_count += 1
            self.deceleration_positive_count += int(target > 0.5)
            self.deceleration_last_target = float(target)
            self.deceleration_last_error = float(error)
            result = {
                "updated": 1.0,
                "settled": float(len(ready)),
                "target": float(target),
                "prediction": float(prediction),
                "error": float(error),
                "magnitude_drop": float(magnitude_drop),
                "gradient_norm": gradient_norm,
                "gradient_clip_factor": clip_factor,
                **{str(key): float(value) for key, value in optimizer.items()},
            }
        return result

    def _queue_deceleration_trace(self, output: TimingNetworkOutput) -> None:
        if not self.deceleration_enabled:
            return
        self.deceleration_pending.append({
            "observation_step": int(self.deceleration_observation_count),
            "trend_score": float(output.trend_score),
            "embedding": output.embedding.copy(),
            "hidden": output.hidden.copy(),
            "candidate": output.candidate.copy(),
            "features": output.features.copy(),
            "state_before": output.state_before.copy(),
            "eligibility_in": output.eligibility_in.astype(np.float32, copy=True),
            "eligibility_rec": output.eligibility_rec.astype(np.float32, copy=True),
            "eligibility_bias": output.eligibility_bias.astype(np.float32, copy=True),
        })
        self.deceleration_pending = self.deceleration_pending[-max(1, self.deceleration_horizon):]

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        shifted = np.asarray(values, dtype=float) - float(np.max(values))
        exp = np.exp(np.clip(shifted, -30.0, 30.0))
        return exp / max(float(np.sum(exp)), 1e-12)

    def _regime_index(self, value: object) -> int | None:
        return _REGIME_LABEL_INDEX.get(str(value or "").strip().upper())

    def _regime_transition_probabilities(self, embedding: np.ndarray, current_regime: str) -> np.ndarray:
        current_index = self._regime_index(current_regime)
        if current_index is None or self.w_regime_transition.size == 0:
            return np.zeros(len(self.regime_transition_horizons), dtype=float)
        values = []
        for head, _horizon in enumerate(self.regime_transition_horizons):
            probs = self._softmax(self.w_regime_transition[head] @ embedding + self.b_regime_transition[head])
            values.append(float(np.clip(1.0 - probs[current_index], 0.0, 1.0)))
        return np.asarray(values, dtype=float)

    def regime_transition_calibration(self, threshold: float) -> dict[str, float]:
        """Return delayed-label evidence for an entry-block threshold.

        This is deliberately evaluated against the probability captured before
        the future label was visible.  A low aggregate Brier score alone is
        insufficient for an entry block in an imbalanced market, therefore the
        supervisor also requires high-probability precision and support.
        """

        cutoff = float(np.clip(threshold, 0.0, 1.0))
        events = list(getattr(self, "regime_transition_calibration_events", []) or [])
        total = len(events)
        brier = (
            float(np.mean([(probability - target) ** 2 for probability, target in events]))
            if events else 1.0
        )
        selected = [item for item in events if float(item[0]) >= cutoff]
        positive = int(sum(int(target) for _probability, target in selected))
        support = len(selected)
        return {
            "total": float(total),
            "brier": float(np.clip(brier, 0.0, 1.0)),
            "high_confidence_support": float(support),
            "high_confidence_precision": float(positive / support) if support else 0.0,
        }

    def _settle_regime_transition(self, current_regime: str) -> dict[str, float]:
        """Settle only predictions whose future regime label is now known."""

        if not self.regime_transition_enabled:
            return {"updated": 0.0}
        target_index = self._regime_index(current_regime)
        if target_index is None:
            return {"updated": 0.0, "unknown_target": 1.0}
        step = int(self.regime_transition_observation_count)
        retained: list[dict[str, object]] = []
        ready: list[dict[str, object]] = []
        for item in self.regime_transition_pending:
            try:
                age = step - int(item.get("observation_step", step))
                horizon = int(item.get("horizon", 0))
            except (AttributeError, TypeError, ValueError):
                continue
            (ready if age >= horizon else retained).append(dict(item))
        maximum = max(self.regime_transition_horizons) * len(self.regime_transition_horizons)
        self.regime_transition_pending = retained[-max(1, maximum):]
        updated = 0
        last_error = 0.0
        for trace in ready:
            try:
                head = self.regime_transition_horizons.index(int(trace["horizon"]))
                embedding = np.asarray(trace["embedding"], dtype=float).reshape(-1)
                hidden = np.asarray(trace["hidden"], dtype=float).reshape(-1)
                features = np.asarray(trace["features"], dtype=float).reshape(-1)
            except (KeyError, TypeError, ValueError):
                continue
            if embedding.size != self.embedding_size or hidden.size != self.hidden_size or features.size != self.input_size:
                continue
            if not all(np.isfinite(value).all() for value in (embedding, hidden, features)):
                continue
            # Only the nearest horizon is used by the no-entry supervisor.  It
            # is the actionable forecast ("about to change"), whereas longer
            # horizons remain useful auxiliary representation targets.
            if int(trace["horizon"]) == int(self.regime_transition_horizons[0]):
                source_regime = str(trace.get("source_regime", "") or "")
                predicted = float(np.clip(trace.get("transition_probability", 0.0), 0.0, 1.0))
                target_changed = int(source_regime != str(current_regime)) if source_regime else 0
                events = self.regime_transition_calibration_events
                events.append((predicted, target_changed))
                self.regime_transition_calibration_events = events[-4096:]
                self.regime_transition_calibration_count += 1
                squared_error = (predicted - target_changed) ** 2
                if self.regime_transition_calibration_count <= 1:
                    self.regime_transition_brier_ema = float(squared_error)
                else:
                    self.regime_transition_brier_ema = float(
                        0.95 * self.regime_transition_brier_ema + 0.05 * squared_error
                    )
            probs = self._softmax(self.w_regime_transition[head] @ embedding + self.b_regime_transition[head])
            error = -probs
            error[target_index] += 1.0
            # Cross-entropy gradient ascent.  A small encoder path shares the
            # causal regime representation without allowing dense labels to
            # overwrite the outcome-trained timing heads.
            d_embedding = self.regime_transition_encoder_scale * (self.w_regime_transition[head].T @ error)
            d_embedding_pre = d_embedding * (1.0 - np.square(np.clip(embedding, -1.0, 1.0)))
            d_hidden = self.w_embed.T @ d_embedding_pre
            update_w = np.outer(error, embedding) - self.regime_transition_l2 * self.w_regime_transition[head]
            update_b = error
            update_embed = np.outer(d_embedding_pre, hidden)
            update_in = np.outer(d_hidden * (1.0 - np.square(np.clip(hidden, -1.0, 1.0))), features)
            arrays = self._parameter_arrays()
            gradients = {name: np.zeros_like(value, dtype=float) for name, value in arrays}
            gradients["w_regime_transition"][head] = update_w
            gradients["b_regime_transition"][head] = update_b
            gradients["w_embed"] = update_embed
            # ``w_in`` has no historical slow-block damping, retain the
            # deliberate small encoder share from the former update rule.
            gradients["w_in"] = 0.35 * update_in
            norm = float(np.sqrt(sum(float(np.sum(np.square(value))) for value in gradients.values())))
            clip = min(1.0, self.gradient_clip / max(norm, 1e-12))
            if clip < 1.0:
                gradients = {name: value * clip for name, value in gradients.items()}
            lr = float(np.clip(
                self.regime_transition_learning_rate * self.regime_transition_update_scale,
                1e-5,
                0.05,
            ))
            self._optimizer_update(
                gradients,
                learning_rate=lr,
                realized_r=float(self.pid_r_ema),
                auxiliary=True,
            )
            updated += 1
            last_error = float(error[target_index])
        if updated:
            self.regime_transition_update_count += updated
            self.regime_transition_last_target = str(current_regime)
            self.regime_transition_last_error = float(last_error)
        return {"updated": float(updated), "settled": float(len(ready)), "target_index": float(target_index), "error": float(last_error)}

    def _queue_regime_transition(self, output: TimingNetworkOutput) -> None:
        if not self.regime_transition_enabled or self._regime_index(output.current_regime) is None:
            return
        for head, horizon in enumerate(self.regime_transition_horizons):
            self.regime_transition_pending.append({
                "observation_step": int(self.regime_transition_observation_count),
                "horizon": int(horizon),
                "source_regime": str(output.current_regime),
                "transition_probability": float(np.clip(
                    output.regime_transition_probabilities[head]
                    if head < output.regime_transition_probabilities.size else 0.0,
                    0.0,
                    1.0,
                )),
                "embedding": output.embedding.copy(),
                "hidden": output.hidden.copy(),
                "features": output.features.copy(),
            })
        maximum = max(self.regime_transition_horizons) * len(self.regime_transition_horizons)
        self.regime_transition_pending = self.regime_transition_pending[-max(1, maximum):]

    def _append_sequence_feature(self, features: np.ndarray) -> tuple[np.ndarray, int]:
        """Append one closed-M15 vector and return the fixed causal tensor.

        The left side is zero padding during warm-up.  The recurrent encoder
        only unrolls the valid suffix, so padding is never interpreted as a
        real flat market bar.  This method is intentionally called only by
        ``forward``; the context facade's M15 signature cache guarantees that
        repeated M1 requests do not append duplicate rows.
        """

        self.ensure_compatible()
        row = np.asarray(features, dtype=float).reshape(-1)
        if row.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(row.size, self.input_size)] = row[: self.input_size]
            row = padded
        self.sequence_buffer[:-1] = self.sequence_buffer[1:]
        self.sequence_buffer[-1] = np.nan_to_num(row, nan=0.0, posinf=2.0, neginf=-2.0)
        self.sequence_valid_length = min(self.sequence_length, int(self.sequence_valid_length) + 1)
        return self.sequence_buffer.copy(), int(self.sequence_valid_length)

    def _encode_sequence_tensor(
        self,
        sequence_tensor: np.ndarray,
        valid_length: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run the TimingNet encoder over the valid suffix of one M15 tensor."""

        tensor = np.asarray(sequence_tensor, dtype=float)
        valid = int(np.clip(valid_length, 0, self.sequence_length))
        if tensor.shape != (self.sequence_length, self.input_size):
            repaired = np.zeros((self.sequence_length, self.input_size), dtype=float)
            rows = min(tensor.shape[0] if tensor.ndim == 2 else 0, self.sequence_length)
            cols = min(tensor.shape[1] if tensor.ndim == 2 else 0, self.input_size)
            if rows and cols:
                repaired[-rows:, :cols] = tensor[-rows:, :cols]
            tensor = repaired
        state = np.zeros(self.hidden_size, dtype=float)
        state_before = state.copy()
        candidate = np.zeros(self.hidden_size, dtype=float)
        update_mix = 1.0 - self.memory_decay
        for values in (tensor[-valid:] if valid else ()):
            state_before = state.copy()
            candidate = np.tanh(self.w_in @ values + self.w_rec @ state_before + self.b_hidden)
            state = self.memory_decay * state_before + update_mix * candidate
        return state_before, candidate, state

    def forward(
        self,
        observation: Iterable[float] | np.ndarray,
        *,
        learn_deceleration: bool = False,
        regime_label: str = "",
        learn_regime_transition: bool = False,
    ) -> TimingNetworkOutput:
        self.ensure_compatible()
        supplied = np.asarray(observation, dtype=float).reshape(-1)
        # An isolated candidate may attach additional *causal* temporal
        # features after the canonical 78 TimingNet inputs.  Treat an exact
        # custom-size vector as already prepared; normal MT5 bridge vectors
        # (137) and legacy 78 vectors continue through canonical extraction.
        if self.input_size != TIMING_NETWORK_INPUT_SIZE and supplied.size == self.input_size:
            raw_timing_features = np.nan_to_num(supplied, nan=0.0, posinf=20.0, neginf=-20.0)
        else:
            raw_timing_features = extract_timing_features(supplied)
        trend_score = timing_trend_score(
            raw_timing_features[:TIMING_NETWORK_INPUT_SIZE],
            mtf_scaling_mode=self.mtf_scaling_mode,
        )
        if self.deceleration_enabled and bool(learn_deceleration):
            self.deceleration_observation_count += 1
            self._deceleration_update_from_future_score(trend_score)
        current_regime = str(regime_label or "").strip().upper()
        if self.regime_transition_enabled and bool(learn_regime_transition):
            self.regime_transition_observation_count += 1
            self._settle_regime_transition(current_regime)
        features = normalize_input_vector(raw_timing_features, mode="soft_clip_v1", size=self.input_size)
        if features.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[: min(features.size, self.input_size)] = features[: self.input_size]
            features = padded
        sequence_tensor, sequence_valid_length = self._append_sequence_feature(features)
        state_before, candidate, hidden = self._encode_sequence_tensor(
            sequence_tensor,
            sequence_valid_length,
        )
        update_mix = 1.0 - self.memory_decay
        self.state = hidden.copy()
        embedding = np.tanh(self.w_embed @ hidden + self.b_embed)
        quality = self._sigmoid(float(np.asarray(self.w_quality @ embedding + self.b_quality).reshape(-1)[0]))
        direction = float(np.tanh(float(np.asarray(self.w_direction @ embedding + self.b_direction).reshape(-1)[0])))
        entry_probability = self._sigmoid(float(np.asarray(self.w_entry @ embedding + self.b_entry).reshape(-1)[0]))
        urgency = self._sigmoid(float(np.asarray(self.w_urgency @ embedding + self.b_urgency).reshape(-1)[0]))
        uncertainty = self._sigmoid(float(np.asarray(self.w_uncertainty @ embedding + self.b_uncertainty).reshape(-1)[0]))
        deceleration_probability = self._sigmoid(float(np.asarray(
            self.w_deceleration @ embedding + self.b_deceleration
        ).reshape(-1)[0]))
        regime_transition_probabilities = self._regime_transition_probabilities(embedding, current_regime)
        action_logits = np.tanh(self.w_action @ embedding + self.b_action)
        # Timing may influence HOLD and entry execution style only. Management
        # and close decisions keep their dedicated policy/risk heads.
        if action_logits.size > 7:
            action_logits[7:] = 0.0

        # Exact online eligibility for the recurrent block.  ``J`` is the
        # derivative of the current hidden state with respect to the previous
        # hidden state.  Keeping the parameter axis explicit (k, j, input)
        # means delayed credit can contract the true chain-rule gradient rather
        # than pretending each recurrent row is independent.
        candidate_derivative = 1.0 - np.square(candidate)
        transition = (
            self.memory_decay * np.eye(self.hidden_size, dtype=float)
            + update_mix * candidate_derivative[:, None] * self.w_rec
        )
        direct_in = np.zeros_like(self.eligibility_in)
        direct_rec = np.zeros_like(self.eligibility_rec)
        direct_bias = np.zeros_like(self.eligibility_bias)
        row_index = np.arange(self.hidden_size)
        direct_in[row_index, row_index, :] = update_mix * candidate_derivative[:, None] * features[None, :]
        direct_rec[row_index, row_index, :] = update_mix * candidate_derivative[:, None] * state_before[None, :]
        direct_bias[row_index, row_index] = update_mix * candidate_derivative
        old_decay = float(self.eligibility_decay)
        self.eligibility_in = old_decay * (
            np.einsum("ko,oji->kji", transition, self.eligibility_in) + direct_in
        )
        self.eligibility_rec = old_decay * (
            np.einsum("ko,oji->kji", transition, self.eligibility_rec) + direct_rec
        )
        self.eligibility_bias = old_decay * (
            np.einsum("ko,oj->kj", transition, self.eligibility_bias) + direct_bias
        )
        self.eligibility_steps = min(self.bptt_window, int(self.eligibility_steps) + 1)
        output = TimingNetworkOutput(
            quality=quality,
            direction=direction,
            entry_probability=entry_probability,
            urgency=urgency,
            uncertainty=uncertainty,
            deceleration_probability=deceleration_probability,
            trend_score=trend_score,
            deceleration_update_count=int(self.deceleration_update_count),
            # The first configured horizon is the actionable forecast used by
            # the entry supervisor.  Longer horizons remain in the vector for
            # diagnostics/training but must not dilute an imminent transition.
            regime_transition_probability=float(regime_transition_probabilities[0]) if regime_transition_probabilities.size else 0.0,
            regime_transition_probabilities=regime_transition_probabilities.copy(),
            regime_transition_update_count=int(self.regime_transition_update_count),
            current_regime=current_regime,
            embedding=embedding.copy(),
            action_logits=action_logits.copy(),
            features=features.copy(),
            sequence_tensor=sequence_tensor.copy(),
            sequence_valid_length=int(sequence_valid_length),
            state_before=state_before,
            candidate=candidate.copy(),
            hidden=hidden.copy(),
            eligibility_in=self.eligibility_in.copy(),
            eligibility_rec=self.eligibility_rec.copy(),
            eligibility_bias=self.eligibility_bias.copy(),
            eligibility_steps=int(self.eligibility_steps),
        )
        if self.deceleration_enabled and bool(learn_deceleration):
            self._queue_deceleration_trace(output)
        if self.regime_transition_enabled and bool(learn_regime_transition):
            self._queue_regime_transition(output)
        return output

    def reset_runtime_state(self) -> None:
        """Reset recurrent state and online eligibility without changing weights."""

        self.ensure_compatible()
        self.state.fill(0.0)
        self.sequence_buffer.fill(0.0)
        self.sequence_valid_length = 0
        self.eligibility_in.fill(0.0)
        self.eligibility_rec.fill(0.0)
        self.eligibility_bias.fill(0.0)
        self.eligibility_steps = 0
        self.deceleration_pending.clear()
        self.deceleration_observation_count = 0
        self.regime_transition_pending.clear()
        self.regime_transition_observation_count = 0

    def update_lifecycle_credit(
        self,
        trace: Mapping[str, object] | None,
        realized_r: float,
        *,
        selected_action: int = 0,
        duration_bars: float | None = None,
        close_reason: str = "",
        update_scale: float = 1.0,
    ) -> dict[str, float]:
        """Apply bounded delayed credit with real activation derivatives.

        The update is a one-sample actor/auxiliary-head loss followed by an
        online truncated-BPTT contraction.  Each forward step updates an
        eligibility tensor for ``w_in``/``w_rec``/``b_hidden``; the entry trace
        freezes that tensor so a later MT5 close credits the bars that led to
        the entry, not the market state at settlement time.
        """

        self.ensure_compatible()
        if not isinstance(trace, Mapping):
            return {"updated": 0.0}
        try:
            features = np.asarray(trace.get("features", []), dtype=float).reshape(-1)
            state_before = np.asarray(trace.get("state_before", []), dtype=float).reshape(-1)
            hidden = np.asarray(trace.get("hidden", []), dtype=float).reshape(-1)
            embedding = np.asarray(trace.get("embedding", []), dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return {"updated": 0.0}
        if features.size != self.input_size or state_before.size != self.hidden_size or hidden.size != self.hidden_size or embedding.size != self.embedding_size:
            return {"updated": 0.0}
        if not all(np.isfinite(item).all() for item in (features, state_before, hidden, embedding)):
            return {"updated": 0.0}
        scale = float(np.clip(update_scale, 0.0, 1.0))
        if scale <= 0.0:
            return {"updated": 0.0}

        def trace_vector(name: str, size: int) -> np.ndarray | None:
            try:
                value = np.asarray(trace.get(name, []), dtype=float).reshape(-1)
            except (TypeError, ValueError):
                return None
            return value if value.size == size and np.isfinite(value).all() else None

        def trace_tensor(name: str, shape: tuple[int, ...]) -> np.ndarray | None:
            try:
                value = np.asarray(trace.get(name, []), dtype=float)
            except (TypeError, ValueError):
                return None
            return value.copy() if value.shape == shape and np.isfinite(value).all() else None

        update_mix = 1.0 - self.memory_decay
        candidate = trace_vector("candidate", self.hidden_size)
        if candidate is None:
            candidate = np.clip((hidden - self.memory_decay * state_before) / max(update_mix, 1e-6), -0.999999, 0.999999)
        action_logits = np.asarray(trace.get("action_logits", np.zeros(self.action_count)), dtype=float).reshape(-1)
        if action_logits.size != self.action_count or not np.isfinite(action_logits).all():
            action_logits = np.zeros(self.action_count, dtype=float)

        outcome = float(np.clip(realized_r, -3.0, 3.0))
        signed_target = float(np.tanh(outcome))
        quality_target = float(np.clip(0.5 + 0.45 * signed_target, 0.05, 0.95))
        entry_target = quality_target
        if duration_bars is not None and float(duration_bars) < 3.0:
            quality_target = float(np.clip(quality_target - 0.08, 0.05, 0.95))
        reason = str(close_reason or "").upper()
        if reason in {"STOP_OUT", "SL", "STOP_LOSS"}:
            quality_target = float(np.clip(quality_target - 0.08, 0.05, 0.95))
        q_pred = self._sigmoid(float(np.asarray(self.w_quality @ embedding + self.b_quality).reshape(-1)[0]))
        d_pred = float(np.tanh(float(np.asarray(self.w_direction @ embedding + self.b_direction).reshape(-1)[0])))
        e_pred = self._sigmoid(float(np.asarray(self.w_entry @ embedding + self.b_entry).reshape(-1)[0]))
        u_pred = self._sigmoid(float(np.asarray(self.w_urgency @ embedding + self.b_urgency).reshape(-1)[0]))
        uncertainty_pred = self._sigmoid(float(np.asarray(self.w_uncertainty @ embedding + self.b_uncertainty).reshape(-1)[0]))
        action_index = int(np.clip(int(selected_action), 0, self.action_count - 1))
        q_err = float(np.clip(quality_target - q_pred, -1.0, 1.0))
        # Direction is a market-direction target, not merely the sign of the
        # account return.  A profitable SELL means the market moved down; a
        # losing SELL means it moved up.  The old target used ``signed_target``
        # directly, so profitable long and profitable short trades taught the
        # same direction and left the head semantically unidentifiable.
        side_sign = 1.0 if action_index in {1, 3, 5} else -1.0 if action_index in {2, 4, 6} else 0.0
        direction_target = float(np.clip(signed_target * side_sign, -1.0, 1.0)) if side_sign else 0.0
        d_err = float(np.clip(direction_target - d_pred, -1.0, 1.0))
        e_err = float(np.clip(entry_target - e_pred, -1.0, 1.0))
        urgency_target = float(np.clip(1.0 - (float(duration_bars) / 120.0), 0.05, 0.95)) if duration_bars is not None else 0.5
        u_err = float(np.clip(urgency_target - u_pred, -1.0, 1.0))
        uncertainty_target = float(np.clip(0.8 - 0.35 * abs(signed_target), 0.1, 0.9))
        uncertainty_err = float(np.clip(uncertainty_target - uncertainty_pred, -1.0, 1.0))
        lr = float(np.clip(self.learning_rate * (0.25 + 0.75 * abs(signed_target)) * scale, 1e-6, 0.004))

        # d(target - prediction) / d(pre-activation).  Sigmoid and tanh
        # derivatives are applied here, instead of treating head errors as
        # weight gradients.  This avoids over-updating saturated outputs.
        d_quality = q_err * q_pred * (1.0 - q_pred)
        d_direction = d_err * (1.0 - d_pred * d_pred)
        d_entry = e_err * e_pred * (1.0 - e_pred)
        d_urgency = u_err * u_pred * (1.0 - u_pred)
        d_uncertainty = uncertainty_err * uncertainty_pred * (1.0 - uncertainty_pred)

        action_prob = np.exp(np.clip(action_logits, -8.0, 8.0) - np.max(action_logits))
        action_prob /= max(float(np.sum(action_prob)), 1e-9)
        action_error = -action_prob
        if action_index <= 6:
            action_error[action_index] += 1.0
            action_error *= signed_target
        else:
            action_error[:] = 0.0
        d_action = 0.50 * action_error * (1.0 - np.square(np.clip(action_logits, -1.0, 1.0)))
        if d_action.size > 7:
            d_action[7:] = 0.0

        d_embedding = (
            d_quality * self.w_quality.reshape(-1)
            + d_direction * self.w_direction.reshape(-1)
            + d_entry * self.w_entry.reshape(-1)
            + d_urgency * self.w_urgency.reshape(-1)
            + d_uncertainty * self.w_uncertainty.reshape(-1)
            + self.w_action.T @ d_action
        )
        embedding_derivative = 1.0 - np.square(np.clip(embedding, -1.0, 1.0))
        d_embedding_pre = d_embedding * embedding_derivative
        d_hidden = self.w_embed.T @ d_embedding_pre
        # The heads are intentionally zero-initialized so the candidate does
        # not alter v13 on its first decision.  Do not let that safety start
        # also make the recurrent representation permanently untrainable: a
        # small bounded outcome-proportional signal reaches the timing hidden
        # state from the first realised lifecycle.  This is a warm-start term;
        # all subsequent updates are driven by the chain-rule signal above.
        representation_signal = float(np.clip(
            0.25 * (q_err + d_err + e_err + u_err + uncertainty_err),
            -1.0,
            1.0,
        ))
        d_hidden += 0.25 * representation_signal * np.tanh(hidden) * (1.0 - np.square(np.clip(hidden, -1.0, 1.0)))

        gradients: dict[str, np.ndarray] = {
            "w_quality": d_quality * embedding.reshape(1, -1),
            "w_direction": d_direction * embedding.reshape(1, -1),
            "w_entry": d_entry * embedding.reshape(1, -1),
            "w_urgency": d_urgency * embedding.reshape(1, -1),
            "w_uncertainty": d_uncertainty * embedding.reshape(1, -1),
            "w_deceleration": np.zeros_like(self.w_deceleration),
            "w_action": np.outer(d_action, embedding),
            "b_action": d_action,
            "w_embed": np.outer(d_embedding_pre, hidden),
            "b_embed": d_embedding_pre,
            "b_quality": np.asarray([d_quality], dtype=float),
            "b_direction": np.asarray([d_direction], dtype=float),
            "b_entry": np.asarray([d_entry], dtype=float),
            "b_urgency": np.asarray([d_urgency], dtype=float),
            "b_uncertainty": np.asarray([d_uncertainty], dtype=float),
            "b_deceleration": np.zeros(1, dtype=float),
            "w_regime_transition": np.zeros_like(self.w_regime_transition),
            "b_regime_transition": np.zeros_like(self.b_regime_transition),
        }
        # Prefer frozen eligibility from the decision trace.  For old v13
        # traces, reconstruct a one-step diagonal eligibility so migration is
        # safe and still uses the correct tanh derivative.
        elig_in = trace_tensor("eligibility_in", (self.hidden_size, self.hidden_size, self.input_size))
        elig_rec = trace_tensor("eligibility_rec", (self.hidden_size, self.hidden_size, self.hidden_size))
        elig_bias = trace_tensor("eligibility_bias", (self.hidden_size, self.hidden_size))
        if elig_in is None or elig_rec is None or elig_bias is None:
            local_factor = update_mix * (1.0 - np.square(np.clip(candidate, -1.0, 1.0)))
            elig_in = np.zeros((self.hidden_size, self.hidden_size, self.input_size), dtype=float)
            elig_rec = np.zeros((self.hidden_size, self.hidden_size, self.hidden_size), dtype=float)
            elig_bias = np.zeros((self.hidden_size, self.hidden_size), dtype=float)
            row_index = np.arange(self.hidden_size)
            elig_in[row_index, row_index, :] = local_factor[:, None] * features[None, :]
            elig_rec[row_index, row_index, :] = local_factor[:, None] * state_before[None, :]
            elig_bias[row_index, row_index] = local_factor
        gradients["w_in"] = np.einsum("k,kji->ji", d_hidden, elig_in)
        gradients["w_rec"] = np.einsum("k,kji->ji", d_hidden, elig_rec)
        gradients["b_hidden"] = np.einsum("k,kj->j", d_hidden, elig_bias)

        gradient_norm = float(np.sqrt(sum(float(np.sum(np.square(value))) for value in gradients.values())))
        clip_factor = float(min(1.0, self.gradient_clip / max(gradient_norm, 1e-12)))
        if clip_factor < 1.0:
            gradients = {name: value * clip_factor for name, value in gradients.items()}
        optimizer_info = self._optimizer_update(
            gradients,
            learning_rate=lr,
            realized_r=outcome,
        )
        self.update_count += 1
        self.last_update = {
            "realized_r": outcome,
            "quality_error": q_err,
            "direction_error": d_err,
            "direction_target": direction_target,
            "side_sign": side_sign,
            "entry_error": e_err,
            "action_index": float(action_index),
            "learning_rate": lr,
            "gradient_norm": gradient_norm,
            "gradient_clip_factor": clip_factor,
            "bptt_steps": float(trace.get("eligibility_steps", 1) or 1),
            "w_in_grad_norm": float(np.linalg.norm(gradients["w_in"])),
            "w_rec_grad_norm": float(np.linalg.norm(gradients["w_rec"])),
            **optimizer_info,
            "optimizer_phase_name": str(self.optimizer_phase),
        }
        return {
            "updated": 1.0,
            "learning_rate": lr,
            "quality_error": q_err,
            "direction_error": d_err,
            "entry_error": e_err,
            "action_index": float(action_index),
            "gradient_norm": gradient_norm,
            "gradient_clip_factor": clip_factor,
            "bptt_steps": float(trace.get("eligibility_steps", 1) or 1),
            "w_in_grad_norm": float(np.linalg.norm(gradients["w_in"])),
            "w_rec_grad_norm": float(np.linalg.norm(gradients["w_rec"])),
            **optimizer_info,
        }


__all__ = [
    "TIMING_NETWORK_ACTION_COUNT",
    "TIMING_NETWORK_INPUT_SIZE",
    "TIMING_NETWORK_SCHEMA_VERSION",
    "TimingNetwork",
    "TimingNetworkOutput",
    "extract_timing_features",
    "timing_trend_score",
]
