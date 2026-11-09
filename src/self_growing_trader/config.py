from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

import yaml

from .live_observatory import LiveObservabilityConfig
from .continuous_session import ContinuousRuntimeConfig
from .he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia
from .protocol_contract import BRIDGE_SCHEMA_VERSION, INPUT_COUNT, OUTPUT_COUNT, require_supported_position_limit
from .hidden_regularization import (
    MAX_DROPOUT_RATE,
    MAX_LAYERWISE_VALUES,
    layerwise_bools,
    mandatory_layerwise_rates,
)


GRAPH_CONNECTABLE_REGIONS: tuple[str, ...] = (
    "encoder",
    "memory",
    "hypothesis",
    "predictor",
    "critic",
    "selector",
    "meta",
    "output",
)


def _unbounded(value: object | None) -> int | None:
    """Normalise capacity values: ``0``/``null`` means no architectural cap.

    This is intentionally limited to neural/memory topology capacities. It does
    not weaken execution-risk, data-integrity, promotion, or numeric-safety
    checks.
    """
    if value is None:
        return None
    number = int(value)
    return None if number <= 0 else number


@dataclass(frozen=True)
class SymbolConfig:
    price_digits: int = 2
    point_size: float = 0.01
    spread_in_points: bool = True

    def __post_init__(self) -> None:
        if int(self.price_digits) < 0 or float(self.point_size) <= 0.0:
            raise ValueError("symbol.price_digits/point_size are invalid")


@dataclass(frozen=True)
class FeatureConfig:
    atr_period: int = 14
    ema_fast: int = 20
    ema_slow: int = 60
    range_window: int = 60
    regime_window: int = 90
    atr_percentile_window: int = 240
    max_spread_atr: float = 8.0
    shock_range_atr: float = 12.0
    wyckoff_range_window: int = 120
    wyckoff_volume_short_window: int = 20
    wyckoff_volume_long_window: int = 60
    wyckoff_event_horizon: int = 60
    # Higher-timeframe values used to be divided by the M1 ATR.  That makes an
    # H1 EMA spread/ATR ratio enormous on a one-minute chart, after which the
    # boundary soft-clip turns it into an almost constant feature.  Keep that
    # historical contract only for protected artifacts; candidates measure MTF
    # direction in its own ATR and encode relative volatility symmetrically.
    mtf_scaling_mode: str = "legacy_m1_atr_v1"

    def __post_init__(self) -> None:
        if min(
            int(self.atr_period), int(self.ema_fast), int(self.ema_slow),
            int(self.range_window), int(self.regime_window), int(self.atr_percentile_window),
            int(self.wyckoff_range_window), int(self.wyckoff_volume_short_window),
            int(self.wyckoff_volume_long_window), int(self.wyckoff_event_horizon),
        ) < 1:
            raise ValueError("feature periods must be positive")
        if int(self.ema_fast) >= int(self.ema_slow):
            raise ValueError("features.ema_fast must be lower than ema_slow")
        if float(self.max_spread_atr) <= 0.0 or float(self.shock_range_atr) <= 0.0:
            raise ValueError("feature safety thresholds must be positive")
        if int(self.wyckoff_volume_short_window) >= int(self.wyckoff_volume_long_window):
            raise ValueError("features.wyckoff_volume_short_window must be lower than wyckoff_volume_long_window")
        if str(self.mtf_scaling_mode).strip().lower() not in {"legacy_m1_atr_v1", "native_tf_atr_v2"}:
            raise ValueError("features.mtf_scaling_mode must be legacy_m1_atr_v1 or native_tf_atr_v2")


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 10000.0
    min_risk_pct: float = 0.001
    max_risk_pct: float = 0.02
    max_open_risk_pct: float = 0.05
    # Zero means no software quota; broker validity and margin remain active.
    # Positive legacy values are bounded by the five detailed observation slots.
    max_positions: int = 5
    ruin_equity_pct: float = 0.5
    # Entry frequency caps are disabled by default. Hard broker/risk validity
    # checks still apply, but day-trade min/max must not stop MT5 training.
    enforce_entry_limits: bool = False
    fixed_risk_pct: float = 0.0
    slippage_atr_multiple: float = 0.0
    # Compatibility-only alias. Its unit is now an ATR multiple, never % price.
    slippage_atr_pct: float = 0.0
    spread_slippage_points: float = 0.0

    def __post_init__(self) -> None:
        if float(self.initial_equity) <= 0.0 or not 0.0 < float(self.min_risk_pct) <= float(self.max_risk_pct) <= 1.0:
            raise ValueError("risk bounds are invalid")
        if not float(self.max_risk_pct) <= float(self.max_open_risk_pct) <= 1.0:
            raise ValueError("max_open_risk_pct must be within [max_risk_pct, 1]")
        require_supported_position_limit(self.max_positions)
        if not 0.0 < float(self.ruin_equity_pct) <= 1.0 or not 0.0 <= float(self.fixed_risk_pct) <= float(self.max_risk_pct):
            raise ValueError("risk ruin/fixed values are invalid")
        if min(float(self.slippage_atr_multiple), float(self.slippage_atr_pct), float(self.spread_slippage_points)) < 0.0:
            raise ValueError("slippage must be non-negative")
        if float(self.slippage_atr_multiple) and float(self.slippage_atr_pct):
            raise ValueError("use only slippage_atr_multiple; slippage_atr_pct is compatibility alias")

    @property
    def effective_slippage_atr_multiple(self) -> float:
        return float(self.slippage_atr_multiple or self.slippage_atr_pct)


@dataclass(frozen=True)
class OrderConfig:
    # Zero disables EA-side forced close by holding age. Fixed SL/TP and
    # explicit policy CLOSE_* actions remain active.
    max_holding_bars: int = 60
    # Number of M1 bars used only after a policy close to score the exit
    # against a mark-to-market "would have held" counterfactual.
    exit_counterfactual_horizon_bars: int = 12
    pending_expiry_bars: int = 60
    # Zero means no cadence quota/cap. Daily trade counts are telemetry only.
    min_daily_trades: int = 0
    max_daily_trades: int = 0
    # Execution reward/risk floor mirrored into MT5.  This is intentionally
    # above 1R so long MT5 runs cannot learn a high-winrate negative-expectancy
    # pattern where each TP is much smaller than each SL.
    technical_min_rr: float = 1.2
    # Immutable entry geometry for all market and pending entries.  ATR is M1
    # ATR measured by the EA at the proposed entry; no policy output can change
    # either distance.
    fixed_sl_atr: float = 5.0
    fixed_tp_atr: float = 10.0
    max_spread_cost_r: float = 0.15
    entry_offset_atr: float = 5.0
    timeframe: str = "M1"

    def __post_init__(self) -> None:
        tf = str(self.timeframe).strip().upper()
        if tf not in {"M1", "M5", "M15", "M30", "H1"}:
            raise ValueError(f"timeframe '{self.timeframe}' is invalid; must be one of M1, M5, M15, M30, H1")
        object.__setattr__(self, "timeframe", tf)
        if int(self.max_holding_bars) < 0 or int(self.exit_counterfactual_horizon_bars) < 0 or int(self.pending_expiry_bars) < 1:
            raise ValueError("max holding/counterfactual horizon must be non-negative and pending expiry must be positive")
        if int(self.min_daily_trades) < 0 or int(self.max_daily_trades) < 0:
            raise ValueError("daily trade values must be non-negative")
        if int(self.max_daily_trades) > 0 and int(self.min_daily_trades) > int(self.max_daily_trades):
            raise ValueError("min_daily_trades cannot exceed max_daily_trades when cadence telemetry is configured")
        if (
            float(self.technical_min_rr) <= 0.0
            or float(self.fixed_sl_atr) <= 0.0
            or float(self.fixed_tp_atr) <= 0.0
            or float(self.fixed_tp_atr) / float(self.fixed_sl_atr) < float(self.technical_min_rr)
            or float(self.max_spread_cost_r) <= 0.0
            or float(self.entry_offset_atr) < 0.0
        ):
            raise ValueError("order risk values are invalid")
        if tf == "M1":
            empirical_spread_atr = 0.649373
            spread_sl_ratio = empirical_spread_atr / float(self.fixed_sl_atr)
            if spread_sl_ratio > float(self.max_spread_cost_r):
                raise ValueError(
                    f"On M1, fixed_sl_atr={self.fixed_sl_atr} violates spread/SL ratio <= {self.max_spread_cost_r:.2f} "
                    f"(requires fixed_sl_atr >= {empirical_spread_atr / self.max_spread_cost_r:.2f} ATR, got ratio {spread_sl_ratio:.3f})."
                )




@dataclass(frozen=True)
class SplitConfig:
    train: float = 0.70
    validation: float = 0.15
    test: float = 0.15

    def __post_init__(self) -> None:
        values = (float(self.train), float(self.validation), float(self.test))
        if any(value <= 0.0 for value in values) or abs(sum(values) - 1.0) > 1e-9:
            raise ValueError("splits must be positive and sum to 1.0")


@dataclass(frozen=True)
class ModelConfig:
    type: str = "self_growing_graph_cognitive_policy"
    input_count: int = INPUT_COUNT
    output_count: int = OUTPUT_COUNT
    initial_modules: int = 1
    hidden_size: int = 128
    hidden_layers: int = 6
    recurrent_core: str = "legacy_rnn"
    # Architectural invariant: every hidden block owns one learnable
    # feature-attention gate immediately before its recurrent transform.
    attention_before_each_hidden: bool = True
    # 0 means no fixed numerical ceiling. Structural survival is still governed
    # by evidence, owner isolation, MT5 lifecycle confirmation and resource limits.
    max_modules: int = 0
    mutation_sigma: float = 0.08

    def __post_init__(self) -> None:
        if int(self.input_count) != INPUT_COUNT or int(self.output_count) != OUTPUT_COUNT:
            raise ValueError(f"model shape must be {INPUT_COUNT}x{OUTPUT_COUNT} for bridge schema v{BRIDGE_SCHEMA_VERSION}")
        if int(self.initial_modules) != 1 or int(self.hidden_size) < 2:
            raise ValueError("model initial_modules must be 1 and hidden_size >= 2")
        core = str(self.recurrent_core or "legacy_rnn").strip().lower()
        if core not in {"legacy_rnn", "openai_five_lstm_v1", "sgt_compact_xlstm"}:
            raise ValueError("model.recurrent_core is not supported")
        if core == "openai_five_lstm_v1":
            if int(self.hidden_size) != 4096 or int(self.hidden_layers) != 1:
                raise ValueError("OpenAI Five recurrent core requires hidden_size=4096 and hidden_layers=1")
            if bool(self.attention_before_each_hidden):
                raise ValueError("OpenAI Five recurrent core does not use the project feature-attention block")
        elif core == "sgt_compact_xlstm":
            if int(self.hidden_size) != 768 or int(self.hidden_layers) != 2:
                raise ValueError("SGT-Compact xLSTM core requires hidden_size=768 and hidden_layers=2")
            if bool(self.attention_before_each_hidden):
                raise ValueError("SGT-Compact recurrent core does not use the legacy feature-attention block")
        else:
            if not 1 <= int(self.hidden_layers) <= MAX_LAYERWISE_VALUES:
                raise ValueError(
                    f"model.hidden_layers must be within [1, {MAX_LAYERWISE_VALUES}]"
                )
            if not bool(self.attention_before_each_hidden):
                raise ValueError("model.attention_before_each_hidden must be true")
        if int(self.max_modules) < 0:
            raise ValueError("model.max_modules must be >= 0 (0 = no fixed cap)")
        if not 0.0 < float(self.mutation_sigma) <= 1.0:
            raise ValueError("model.mutation_sigma must be in (0,1]")

@dataclass(frozen=True)
class ExpectancyConfig:
    """Controls for robust net-R value learning and promotion evidence."""

    enabled: bool = True
    min_samples: int = 6
    merge_similarity: float = 0.90
    max_samples_per_context: int = 256
    min_policy_origin_ratio: float = 0.60
    promotion_lcb_min_delta: float = 0.01

    def __post_init__(self) -> None:
        if self.min_samples < 1:
            raise ValueError("expectancy.min_samples must be at least 1")
        if not 0.50 <= self.merge_similarity < 1.0:
            raise ValueError("expectancy.merge_similarity must be within [0.50, 1.0)")
        if self.max_samples_per_context < 8:
            raise ValueError("expectancy.max_samples_per_context must be at least 8")
        if not 0.0 <= self.min_policy_origin_ratio <= 1.0:
            raise ValueError("expectancy.min_policy_origin_ratio must be within [0, 1]")
        if self.promotion_lcb_min_delta < 0.0:
            raise ValueError("expectancy.promotion_lcb_min_delta must be non-negative")


@dataclass(frozen=True)
class SupernetworkConfig:
    """Bounded graph-growth budget used for comparable MT5 candidates.

    V7 replaces open-ended structural state with explicit ceilings.  The
    ceilings are deliberately generous relative to the starter graph, but they
    prevent a lucky/noisy trial from winning only because it consumed more
    neurons, edges, child graphs, or replay memory than its comparator.
    """

    m1_backend: str = "direct_rnn"
    enable_child_supernet: bool = False
    max_child_networks: int = 0
    max_generation: int = 0
    max_children_per_network: int = 0
    diversity_threshold: float = 0.05
    cross_module_enabled: bool = True
    max_cross_module_edges: int = 0
    max_m1_module_edges: int = 0
    max_cross_module_recurrent_edges: int = 0
    max_cross_module_lateral_edges: int = 0
    recurrent_loop_limit: int = 0
    max_graph_neurons: int = 0
    max_graph_connections: int = 0
    max_hidden_neurons_per_module: int = 512
    max_hidden_layers_per_module: int = 24
    hidden_depth_growth_threshold: float = 0.55
    allowed_cross_module_edge_kinds: tuple[str, ...] = ()
    allowed_module_to_m1_targets: tuple[str, ...] = GRAPH_CONNECTABLE_REGIONS
    allowed_m1_to_module_targets: tuple[str, ...] = GRAPH_CONNECTABLE_REGIONS
    allowed_specialist_roles: tuple[str, ...] = ()
    flexible_cross_module_connectivity: bool = True
    flexible_connection_fanout: int = 0
    flexible_peer_module_fanout: int = 0
    module_role_budget: int = 0
    prediction_need_threshold: float = 0.35
    memory_need_threshold: float = 0.35
    risk_need_threshold: float = 0.35
    position_need_threshold: float = 0.35
    cadence_need_threshold: float = 0.35
    role_cooldown: int = 2
    require_mt5_promotion: bool = True
    trial_history_path: str = "outputs/supernetwork_trial_history.jsonl"
    enable_trial_micro_growth: bool = True
    micro_growth_max_neurons_per_trial: int = 0
    micro_growth_learning_threshold: float = 0.35
    micro_growth_initial_gate: float = -4.0
    micro_growth_initial_weight: float = 0.025
    micro_growth_survival_contexts: int = 3
    micro_growth_survival_similar_outcomes: int = 3
    micro_growth_context_similarity_threshold: float = 0.82
    micro_growth_context_prototype_lr: float = 0.08
    micro_growth_survival_positive_supports: int = 3
    micro_growth_survival_utility: float = 0.03
    micro_growth_consolidate_utility: float = 0.20
    micro_growth_fade_rate: float = 0.92
    micro_growth_prune_fade_threshold: float = 0.22
    micro_growth_grace_outcomes: int = 5
    micro_growth_max_modules_per_trial: int = 0
    specialist_module_initial_gate: float = -2.15
    specialist_module_initial_weight: float = 0.055
    specialist_module_input_adapters: int = 3
    specialist_module_core_neurons: int = 4
    specialist_module_output_adapters: int = 2
    specialist_module_enable_bypass_interfaces: bool = True
    specialist_min_causal_confidence_to_activate: float = 0.55
    specialist_require_causal_validation_to_consolidate: bool = True

    def __post_init__(self) -> None:
        backend = str(self.m1_backend).strip().lower()
        if backend == "graph":
            object.__setattr__(self, "m1_backend", "m1_graph")
            backend = "m1_graph"
        if backend not in {"direct_rnn", "m1_graph"}:
            raise ValueError("supernetwork.m1_backend must be direct_rnn or m1_graph.")
        for name in (
            "max_child_networks", "max_generation", "max_children_per_network",
            "max_cross_module_edges", "max_m1_module_edges",
            "max_cross_module_recurrent_edges", "max_cross_module_lateral_edges",
            "recurrent_loop_limit", "max_graph_neurons", "max_graph_connections",
            "max_hidden_neurons_per_module", "max_hidden_layers_per_module",
            "flexible_connection_fanout",
            "flexible_peer_module_fanout", "module_role_budget",
            "micro_growth_max_neurons_per_trial", "micro_growth_max_modules_per_trial",
        ):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"supernetwork.{name} must be >= 0 (0 = no fixed cap)")
        if not 0.0 <= self.diversity_threshold <= 1.0:
            raise ValueError("supernetwork.diversity_threshold must be within [0, 1]")
        if self.micro_growth_learning_threshold < 0.0:
            raise ValueError("supernetwork.micro_growth_learning_threshold must be non-negative")
        if not -6.0 <= self.specialist_module_initial_gate <= -0.10:
            raise ValueError("supernetwork.specialist_module_initial_gate must be within [-6, -0.10]")
        if not 0.005 <= self.specialist_module_initial_weight <= 0.20:
            raise ValueError("supernetwork.specialist_module_initial_weight must be within [0.005, 0.20]")
        if self.specialist_module_input_adapters < 1 or self.specialist_module_output_adapters < 1:
            raise ValueError("supernetwork specialist module adapters must be at least 1")
        if self.specialist_module_core_neurons < 2:
            raise ValueError("supernetwork.specialist_module_core_neurons must be at least 2")
        if not 0.0 <= self.specialist_min_causal_confidence_to_activate <= 1.0:
            raise ValueError("supernetwork.specialist_min_causal_confidence_to_activate must be within [0, 1]")
        if self.micro_growth_survival_similar_outcomes < 1:
            raise ValueError("supernetwork.micro_growth_survival_similar_outcomes must be at least 1")
        if not 0.50 <= self.micro_growth_context_similarity_threshold < 1.0:
            raise ValueError("supernetwork.micro_growth_context_similarity_threshold must be within [0.50, 1.0)")
        if not 0.0 < self.micro_growth_context_prototype_lr <= 0.50:
            raise ValueError("supernetwork.micro_growth_context_prototype_lr must be within (0, 0.50]")
        if not 0.0 < self.micro_growth_fade_rate <= 1.0:
            raise ValueError("supernetwork.micro_growth_fade_rate must be within (0, 1]")
        if not 0.0 < self.micro_growth_prune_fade_threshold < 1.0:
            raise ValueError("supernetwork.micro_growth_prune_fade_threshold must be within (0, 1)")


@dataclass(frozen=True)
class LearningIntegrityConfig:
    """Reproducibility and anti-leakage contract for training/promotion.

    These are not topology caps. They make every candidate comparable, keep
    out-of-sample evaluation frozen, and prevent low-fidelity OHLC results
    from being promoted as if they were tick-accurate evidence.
    """

    seed: int = 20260629
    require_real_ticks_for_promotion: bool = True
    # ``model=4`` (Every tick based on real ticks) is only a requested tester
    # setting. Promotion additionally requires fresh report evidence that the
    # run identifies real ticks.
    require_real_tick_evidence_for_promotion: bool = True
    # A reconciled report may still contain mark-to-market PnL from positions
    # that were open when the tester stopped. Such a window is useful for
    # training diagnostics, but it is not a closed lifecycle validation
    # window and must not be promoted as one.
    require_terminal_flat_for_promotion: bool = True
    frozen_oos: bool = True
    # Curiosity may request replay/shadow candidates, but it must not turn
    # uncertainty or missing data into a live BUY/SELL proposal.  ``evidence_only``
    # is retained for controlled research and still requires contextual
    # expectancy evidence before a non-HOLD action can be influenced.
    live_curiosity_mode: str = "shadow_only"
    # V9 action-recovery learner. Positive local evidence may replace an entry;
    # sufficiently repeated *real* negative lifecycle evidence may veto only
    # that action in the matching context to HOLD. This is never a global HOLD
    # penalty and never creates a forced entry.
    action_recovery_enabled: bool = True
    # Keep counterfactual/shadow and lifecycle evidence learning active even
    # when action-recovery reranking is disabled for an anti-overfit run.
    action_recovery_learning_enabled: bool = True
    action_recovery_min_support: float = 3.0
    action_recovery_min_lcb: float = 0.08
    action_recovery_merge_similarity: float = 0.90
    action_recovery_shadow_weight: float = 0.50
    action_recovery_max_cells_per_action: int = 128
    action_recovery_negative_veto_enabled: bool = False
    action_recovery_negative_veto_min_real_support: float = 3.0
    action_recovery_negative_veto_lcb: float = -0.04
    # MT5 training bridge guards.  These throttle new entries only; they do
    # not block HOLD, cancellation, management, or forced closes.
    entry_guard_single_position: bool = False
    # Candidate-only safety gate: volatile contexts must have directional
    # evidence before emitting a fresh entry.  False preserves seed/runtime
    # compatibility; promotion experiments enable it explicitly.
    context_direction_entry_gate_enabled: bool = False
    context_direction_entry_gate_min_timing_quality: float = 0.25
    # Candidate-only directional-side gate for confirmed trend contexts. It
    # removes only the opposite BUY/SELL side; HOLD and management remain legal.
    context_direction_side_entry_gate_enabled: bool = False
    # Candidate-only anti-overfit gate: a fresh entry may be vetoed only when
    # the loaded contextual-expectancy ledger has reliable negative LCB,
    # sufficient policy support, and the exact context/action/position scope
    # matches. False preserves seed/runtime compatibility.
    context_expectancy_entry_gate_enabled: bool = False
    context_expectancy_entry_gate_min_support: float = 12.0
    context_expectancy_entry_gate_max_lcb: float = -0.15
    # ``None`` preserves the legacy all-context behavior when a profile
    # deliberately enables the gate.  A non-empty tuple restricts hard LCB
    # vetoes to those owner contexts; all other owners keep expectancy
    # telemetry and the bounded soft action delta, but no action is masked.
    context_expectancy_entry_gate_allowed_contexts: tuple[str, ...] | None = None
    # Candidate-only fallback: when exact prototype retrieval has no reliable
    # negative evidence, consult the same owner's aggregate action/side/
    # lifecycle ledger without mixing semantic actions or management traces.
    context_expectancy_entry_gate_scope_fallback_enabled: bool = False
    entry_loss_guard_enabled: bool = False
    entry_loss_guard_min_closed: int = 6
    entry_loss_guard_max_closed_loss_r: float = 0.25
    entry_loss_guard_consecutive_losses: int = 4
    train_entry_probe_enabled: bool = False
    train_entry_probe_after_decisions: int = 12
    train_entry_probe_interval_decisions: int = 6
    train_entry_probe_max_per_day: int = 2
    train_entry_probe_max_per_run: int = 6
    # Probes are evidence collection, not random forced trades.  Require a
    # direction confirmed by the owner context/timing before emitting one.
    train_entry_probe_min_direction_quality: float = 0.75
    runtime_fitness_guard_enabled: bool = True
    runtime_fitness_rolling_window: int = 50
    runtime_fitness_min_closed: int = 20
    runtime_fitness_min_score: float = -0.30
    runtime_fitness_max_immediate_exit_ratio: float = 0.60
    runtime_fitness_max_daily_loss_r: float = 3.0
    runtime_fitness_max_run_loss_r: float = 8.0
    runtime_fitness_max_daily_trades: int = 300
    runtime_fitness_max_drawdown_pct: float = 20.0
    # Public-Five schedule plus trading-specific competitive reward.  The
    # tournament bonus is zero-sum across five agents; absolute portfolio
    # profit and drawdown remain the promotion objective.
    five_training_schedule_steps: int = 10_000
    five_reward_drawdown_weight: float = 0.30
    five_reward_risk_excess_weight: float = 0.10
    five_reward_conflict_weight: float = 0.05
    five_reward_competition_weight: float = 0.25
    five_reward_terminal_rank_weight: float = 0.50
    # ``dense_mtm`` preserves the existing marked-to-market objective. The
    # candidate settlement mode assigns most credit at lifecycle close using
    # net-R and keeps only a small open-position MTM signal.
    five_reward_mode: str = "dense_mtm"
    five_reward_settlement_weight: float = 1.0
    five_reward_mark_to_market_weight: float = 1.0
    five_reward_settlement_clip_r: float = 3.0
    # Candidate Reward V2. Net marked-R remains the primary objective. These
    # bounded closure-only terms make WR/expectancy learnable without paying a
    # reward merely for opening/closing or allowing tiny wins to hide fat-tail
    # losses.
    five_reward_expectancy_wr_v2_enabled: bool = False
    five_reward_wr_target: float = 0.55
    five_reward_wr_balance_weight: float = 0.05
    five_reward_expectancy_floor_r: float = 0.05
    five_reward_expectancy_shortfall_weight: float = 0.10
    five_reward_tail_loss_threshold_r: float = 1.0
    five_reward_tail_loss_weight: float = 0.10
    five_reward_tail_loss_cap_r: float = 3.0
    # Lifecycle Reward V4. Moi thanh phan co dau, duoc tinh dung mot lan cho
    # entry owner va tong reward dinh hinh bi chan trong [-clip, +clip].
    five_lifecycle_reward_enabled: bool = True
    five_lifecycle_shape_clip_r: float = 0.75
    five_lifecycle_min_sl_atr: float = 1.5
    five_lifecycle_min_tp_atr: float = 3.0
    five_lifecycle_geometry_penalty_r: float = 0.30
    five_lifecycle_geometry_bonus_r: float = 0.05
    # Optional candidate channel: route signed final net-R directly to the
    # immutable entry owner. Disabled by default so the legacy V4 profile is
    # unchanged; candidate profiles can combine this with the shaped V4 term.
    five_lifecycle_net_r_credit_enabled: bool = False
    five_lifecycle_net_r_credit_weight: float = 0.0
    five_lifecycle_net_r_credit_clip_r: float = 3.0
    # INITIAL_SL is not a scalar reward. It trains a trace-local conditional
    # loss among the six entry actions so probability cannot leak to HOLD or
    # MANAGE_POSITION.
    five_lifecycle_initial_sl_local_loss_weight: float = 1.0
    five_lifecycle_mfe_penalty_zero_r: float = 0.40
    five_lifecycle_mfe_penalty_010_r: float = 0.30
    five_lifecycle_mfe_penalty_025_r: float = 0.15
    five_lifecycle_mfe_penalty_neutral_at_r: float = 0.50
    five_lifecycle_excursion_bonus_at_r: float = 1.0
    five_lifecycle_excursion_good_mae_r: float = 0.25
    five_lifecycle_excursion_bad_mae_r: float = 1.0
    five_lifecycle_excursion_min_bonus_r: float = 0.15
    five_lifecycle_excursion_max_bonus_r: float = 0.30
    five_lifecycle_trailing_activation_mfe_r: float = 0.50
    five_lifecycle_trailing_capture_min_ratio: float = 0.50
    five_lifecycle_trailing_capture_target_ratio: float = 0.80
    five_lifecycle_trailing_missed_penalty_r: float = 0.20
    five_lifecycle_trailing_small_capture_penalty_r: float = 0.10
    five_lifecycle_trailing_min_bonus_r: float = 0.10
    five_lifecycle_trailing_max_bonus_r: float = 0.25
    five_lifecycle_exit_capture_min_ratio: float = 0.25
    five_lifecycle_exit_capture_target_ratio: float = 0.80
    five_lifecycle_exit_capture_min_bonus_r: float = 0.10
    five_lifecycle_exit_capture_max_bonus_r: float = 0.30
    five_lifecycle_recovery_min_bonus_r: float = 0.05
    five_lifecycle_recovery_max_bonus_r: float = 0.15
    five_lifecycle_pending_fill_max_bars: int = 5
    five_lifecycle_pending_fill_min_bonus_r: float = 0.05
    five_lifecycle_pending_fill_max_bonus_r: float = 0.10
    five_lifecycle_batch_size: int = 32
    five_lifecycle_batch_max_wait: int = 64
    five_lifecycle_batch_neutral_band_r: float = 0.05
    # Current-policy failure replay. Settled losing entry contexts are kept as
    # immutable anchors and re-forwarded on later lifecycle optimizer batches
    # until the current policy has reduced the failed action probability for
    # several audits. Dormant anchors periodically audit for regression.
    five_failure_replay_enabled: bool = True
    five_failure_replay_capacity: int = 256
    five_failure_replay_items_per_batch: int = 8
    # Do not turn a singleton failure into a global gradient.  The anchor is
    # still retained and audited; this threshold only controls replay training.
    five_failure_replay_min_bucket_support: int = 1
    five_failure_replay_audit_items: int = 32
    five_failure_replay_weight: float = 0.35
    five_failure_replay_admit_threshold_r: float = -0.25
    five_failure_replay_repair_probability: float = 0.08
    five_failure_replay_reopen_probability: float = 0.14
    five_failure_replay_repair_streak: int = 3
    five_failure_replay_audit_interval: int = 8
    # Strategy-Tester replay backend. When enabled, historical losing contexts
    # are revisited by MT5 and the current policy generates fresh on-policy
    # trajectories. The old auxiliary anchor loss is disabled for production
    # Five runs to avoid double-punishing the same failure.
    five_mt5_failure_replay_enabled: bool = False
    five_mt5_failure_replay_sessions_per_run: int = 12
    five_mt5_failure_replay_max_attempts_per_case: int = 6
    five_mt5_failure_replay_warmup_minutes: int = 120
    five_mt5_failure_replay_horizon_minutes: int = 180
    five_mt5_failure_replay_resolved_audit_items: int = 16
    # Evaluation-only causal A/B: keep policy parameters frozen while drawing
    # actions/parameters from the seeded behavior distribution. This must not
    # create a trainer or permit an optimizer update.
    five_frozen_behavior_sampling_enabled: bool = False
    five_frozen_behavior_sampling_seed: int = 20260702
    five_position_improvement_reward_weight: float = 1.0
    # Hinh phat hop dong hanh dong lon nhat: moi action quan ly/dong vi the khi
    # snapshot truoc quyet dinh khong co vi the. MANAGE_POSITION bao phu doi
    # SL/TP, trailing, dong mot phan va dong toan phan. Khoan phat gan rieng
    # voi action policy da chon va nam ngoai team-spirit.
    five_invalid_position_action_no_position_penalty_r: float = 1.0
    # CANCEL_ORDERS khi snapshot truoc quyet dinh khong co pending order la
    # mot vi pham truc tiep cua action contract. Phat dung agent phat lenh va
    # de khoan phat ngoai team-spirit de no khong the huong reward cua agent khac.
    five_invalid_cancel_no_pending_penalty_r: float = 2.0
    # A LIMIT/STOP that survives five bars without a fill receives a modest
    # timing cost. A deliberate early cancellation receives no reward credit.
    five_unfilled_pending_timeout_penalty_r: float = 0.25
    # Keep the public-Five entropy coefficient at its initial value until the
    # immutable expectancy ledger proves enough positive context/action
    # scopes. After release it decays by edge count, never by elapsed time.
    five_entropy_edge_gate_enabled: bool = False
    five_entropy_edge_release_count: int = 3
    five_entropy_edge_decay_count: int = 8
    five_entropy_start: float = 0.06
    # Tron hanh vi va entropy la hai co che kham pha doc lap voi reward.
    five_behavior_uniform_mix: float = 0.15
    # Candidate-only selector for the public Five entry decision surface.
    # ``checkpoint`` preserves the mode declared by the loaded checkpoint;
    # ``expected_value`` and ``direct`` are explicit research arms.
    five_action_decision_mode: str = "checkpoint"
    # Lower bound for temporal credit measured in actual bridge decisions.
    five_gae_min_feedback_steps: int = 16
    # A validated offline imitation checkpoint carries an entry-timing prior.
    # Keep that small head fixed during the first MT5 optimizer updates, then
    # anchor it while the recurrent representation adapts to economic reward.
    five_imitation_freeze_updates: int = 32
    five_imitation_anchor_updates: int = 256
    five_imitation_anchor_weight: float = 0.10
    five_imitation_warmup_uniform_mix_max: float = 0.05
    five_imitation_disable_flat_entry_floor_during_warmup: bool = True
    # Candidate runs may use a shorter delayed path-credit unroll so a
    # bounded MT5 smoke window still produces several optimizer updates. The
    # runtime default remains the shared trading Five unroll length (64).
    five_pattern_unroll: int = 64
    # Candidate-only behavior sampling floor per legal entry. This is not an
    # action mask: every legal action remains available, but flat-account
    # exploration cannot be consumed entirely by MANAGE/CLOSE.
    five_behavior_entry_probability_floor: float = 0.005
    # While the immutable ledger has not proved enough positive edges, every
    # flat-account state keeps a minimum learned probability mass on the six
    # entry actions. This affects the raw policy, not only behavior sampling.
    five_flat_entry_mass_floor_start: float = 0.75
    five_flat_entry_mass_floor_end: float = 0.0
    five_flat_entry_mass_loss_weight: float = 1.0
    # Candidate-only recovery term. A squared probability-floor loss has a
    # vanishing softmax gradient after HOLD saturation; the log barrier keeps
    # a bounded, non-zero push toward the legal entry mass. Default remains
    # disabled so existing production profiles are unchanged.
    five_flat_entry_mass_log_loss_weight: float = 0.0
    # Candidate-only owner balance floor. Unlike the total entry floor, this
    # keeps LONG and SHORT entry owners from satisfying exploration with one
    # direction only.
    five_flat_entry_direction_mass_floor_start: float = 0.0
    five_flat_entry_direction_mass_floor_end: float = 0.0
    five_flat_entry_direction_mass_loss_weight: float = 0.0
    # Candidate-only causal guard. When enabled, public Five may veto only an
    # exact entry action with enough context-matched negative ledger evidence.
    # HOLD remains mandatory and the default keeps the protected path intact.
    five_expectancy_action_veto_enabled: bool = False
    five_expectancy_action_veto_min_samples: int = 12
    five_expectancy_action_veto_min_confidence: float = 0.20
    five_expectancy_action_veto_min_lcb_r: float = -0.15
    five_expectancy_action_veto_hold_owner_on_negative: bool = False
    # Candidate-only run-local action circuit breaker. It uses only final,
    # owner-reconciled lifecycles from the current window, not checkpoint or
    # cross-window evidence. This is intentionally separate from the
    # context-retrieval veto above because context/position-state fragments can
    # leave an action-level negative edge under-supported at decision time.
    five_live_action_veto_enabled: bool = False
    five_live_action_veto_min_samples: int = 12
    five_live_action_veto_min_confidence: float = 0.20
    five_live_action_veto_min_lcb_r: float = -0.15
    five_live_action_veto_hold_owner_on_negative: bool = True
    # Candidate-only recovery probe. When every entry of an owner is being
    # held back by current-window evidence, keep at most one legal entry
    # action available so the selector can collect new immutable evidence.
    five_live_action_veto_probe_enabled: bool = False
    five_live_action_veto_probe_interval_decisions: int = 64
    # Candidate-only transition probe. When an owner first reaches the
    # configured negative-evidence threshold, allow one bounded recovery
    # probe immediately instead of waiting for the periodic cadence.
    five_live_action_veto_probe_on_negative_transition: bool = False
    # Candidate-only adaptive exploration.  The sampler may reduce random
    # entry draws for one entry owner after enough *current-run* final
    # lifecycle outcomes prove that owner's side is negative.  The other
    # owner and the learned policy remain untouched.
    five_live_owner_exploration_enabled: bool = False
    five_live_owner_exploration_min_samples: int = 12
    five_live_owner_exploration_min_confidence: float = 0.20
    five_live_owner_exploration_min_lcb_r: float = -0.15
    five_live_owner_exploration_min_mix: float = 0.02
    # Candidate-only bootstrap lease for an entry owner with no immutable
    # close sample in the current run. It keeps HOLD legal, is bounded by
    # cadence/run count, and exists only to obtain causal lifecycle evidence;
    # it is not a strategy override or a promotion signal.
    five_live_owner_bootstrap_probe_enabled: bool = False
    five_live_owner_bootstrap_probe_interval_decisions: int = 16
    five_live_owner_bootstrap_probe_max_per_run: int = 2
    # Research-only option: during the bounded zero-sample lease, remove
    # HOLD so the owner produces one market lifecycle sample. Legacy and OOS
    # defaults keep HOLD mandatory; the action remains trace/PPO-visible.
    five_live_owner_bootstrap_probe_force_market: bool = False
    # Delayed causal reward. Future outcome labels supervise only a cached
    # anchor after the dataset sample or MT5 closed-bar horizon has completed;
    # they never enter the decision observation or frozen/live inference.
    five_pattern_reward_enabled: bool = False
    # Candidate-only selector supervision for the public Five entry surface.
    # This is deliberately separate from ``five_pattern_reward_enabled`` so
    # disabling V4 does not silently discard delayed HOLD-vs-entry learning.
    five_entry_preference_supervision_enabled: bool = False
    # Candidate-only multi-step hypothesis policy.  It is trained by GRPO on
    # mature action-local counterfactual paths.  ``action_scale=0`` keeps it
    # telemetry-only until paired frozen-OOS promotion explicitly authorizes
    # a bounded residual on HOLD + six entry logits.
    five_reasoning_rl_enabled: bool = False
    # Candidate-only RLHF bridge. When enabled for an MT5 training run, the
    # public Five owner snapshots a frozen reference policy and stores its
    # decision-time logits in the execution observatory. It does not enable
    # reward-model scoring, PPO updates, or promotion by itself.
    five_rlhf_reference_capture_enabled: bool = False
    # Candidate-only MT5 RLHF policy phase. The worker loads an already-frozen
    # reward-model artifact and applies bounded PPO updates through FiveAdam;
    # it never trains the reward model or promotes the resulting checkpoint.
    five_rlhf_policy_update_enabled: bool = False
    # Candidate-only Codex teacher: every immutable confirmed MT5 close is
    # graded asynchronously while the pass is still running.  ``0`` means no
    # per-run case cap; a positive value is retained for bounded smoke runs.
    five_codex_teacher_enabled: bool = False
    five_codex_teacher_model: str = "gpt-5.6-luna"
    five_codex_teacher_reasoning_effort: str = "medium"
    five_codex_teacher_max_cases_per_run: int = 0
    five_codex_teacher_timeout_sec: int = 120
    # Online action barrier: after a finalized close is submitted, wait for
    # at least one verified Codex update before sampling the next action.
    # Zero disables the barrier explicitly; the default keeps connected
    # learning feedback causal without waiting forever on a failed CLI.
    # A Codex receipt is an online causal boundary: the next action should
    # not be sampled before the verified learner update unless the bounded
    # wait expires.  The CLI grade latency is commonly tens of seconds, so
    # keep this aligned with the 120-second subprocess timeout instead of
    # letting a 30-second barrier routinely release stale policy state.
    five_codex_teacher_online_wait_sec: float = 120.0
    # Bounded parallel grading keeps feedback close to the action stream while
    # avoiding an unbounded Codex subprocess fan-out. Updates are still
    # applied serially on the MT5 bridge thread.
    five_codex_teacher_max_workers: int = 2
    five_codex_teacher_reward_scale: float = 0.25
    five_codex_teacher_min_confidence: float = 0.55
    five_codex_teacher_max_probability_shift: float = 0.05
    # Codex feedback uses a separate, bounded learning rate. The normal Five
    # schedule starts at 5e-5, which is appropriate for broad PPO training but
    # makes a single verified terminal receipt effectively invisible on the
    # contextual residual. This rate is used only for the Codex-scoped update.
    # Use the full bounded Codex step by default.  The packet probability
    # guard and backtracking still cap every individual update at the
    # configured maximum shift, while 5e-3 was too small to carry a verified
    # terminal reward into an unseen live context.
    five_codex_teacher_learning_rate: float = 0.0003
    # The prototype residual must generalize across the recurrent-state drift
    # of one market regime.  With support_floor=0.50 this produces an
    # effective capture/reuse threshold of 0.95; the per-packet probability
    # guard still bounds every intervention.  A separate bounded global action
    # bias carries verified feedback into genuinely unseen contexts.
    five_codex_context_similarity_threshold: float = 0.90
    five_codex_context_support_floor: float = 0.50
    five_codex_global_action_bias_scale: float = 0.50
    # Codex must register a falsifiable hypothesis and a locally checkable
    # verification before its bounded intervention can update the policy.
    five_codex_teacher_hypothesis_required: bool = True
    # MT5 learning attempts the same Codex bridge for every learnable context
    # configuration when the local Codex CLI and a checkpoint are available.
    # The runtime remains audit-only when provenance, frozen reference, or
    # optimizer hooks are absent, and frozen/OOS passes explicitly disable the
    # bridge. This makes Codex behavior feedback a capability of the learning
    # path rather than a profile-specific opt-in.
    five_codex_teacher_auto_connect: bool = True
    five_rlhf_reward_model_path: str = ""
    # The MT5 worker must know whether the frozen scorer came from human
    # feedback or the separate teacher/RLAIF branch.
    five_rlhf_reward_feedback_mode: str = "human"
    # Optional frozen teacher provenance.  Independent-teacher mode requires
    # this hash to be present in the reward-model report and different from
    # the student/source champion hash.
    five_rlhf_teacher_checkpoint_sha256: str = ""
    five_rlhf_require_independent_teacher: bool = False
    five_rlhf_source_checkpoint_sha256: str = ""
    # Candidate/OOS-only storage control. When enabled, the final forensic
    # artifact drops FiveAdam's per-parameter moments after MT5 closes. It
    # must not be used for a continuation-training source or a champion.
    five_rlhf_strip_optimizer_state_for_oos: bool = False
    # Explicit research-only escape hatch for resuming policy weights from an
    # OOS artifact whose FiveAdam moments were stripped. Reinitialization is
    # lazy and is never treated as continuation from the original state.
    five_rlhf_allow_optimizer_state_reinitialization_for_research: bool = False
    # Research-only crash recovery. After this many accepted Codex teacher
    # updates the MT5 learner may write one atomic recovery snapshot beside
    # the final candidate. Zero keeps the legacy final-only behavior.
    five_rlhf_recovery_snapshot_interval_updates: int = 0
    # Matrix preflight only. Zero keeps the generic profile backward
    # compatible; the RLAIF matrix profile sets an explicit byte floor before
    # it starts any MT5 writer.
    five_rlhf_min_free_disk_bytes: int = 0
    five_rlhf_update_interval_packets: int = 1
    five_rlhf_max_updates_per_run: int = 0
    five_reasoning_steps: int = 3
    five_reasoning_group_size: int = 8
    five_reasoning_ppo_clip: float = 0.20
    five_reasoning_kl_beta: float = 0.02
    five_reasoning_entropy_weight: float = 0.001
    five_reasoning_loss_weight: float = 0.25
    five_reasoning_encoder_credit_scale: float = 0.0
    five_reasoning_min_group_std_r: float = 0.02
    five_reasoning_action_scale: float = 0.0
    five_reasoning_action_residual_clip: float = 0.25
    five_lock_non_entry_to_hold: bool = True
    five_fixed_geometry_enabled: bool = True
    adaptive_volatility_targets_enabled: bool = False
    # HOLD tai khoan trong chi nhan credit tren path tuong lai bat bien. Phat
    # bo lo edge lon hon rat nhieu thuong ne mot tap entry deu am.
    five_flat_hold_missed_edge_penalty_r: float = 0.25
    five_flat_hold_confirmed_no_edge_bonus_r: float = 0.01
    five_flat_hold_edge_threshold_r: float = 0.10
    five_flat_hold_no_edge_threshold_r: float = -0.10
    # Observation Pattern Tree auxiliary supervision. The tree is part of the
    # v4 encoder; these weights control immutable dataset labels and mature
    # MT5 path labels. They are ignored by frozen/live inference.
    five_pattern_tree_auxiliary_loss_weight: float = 0.25
    five_pattern_tree_min_edge_atr: float = 0.10
    five_pattern_tree_mfe_weight: float = 0.15
    five_pattern_tree_mae_weight: float = 0.25
    five_pattern_tree_utility_clip_atr: float = 3.0
    five_pattern_tree_utility_loss_weight: float = 1.0
    five_pattern_tree_excursion_loss_weight: float = 0.50
    five_pattern_tree_no_edge_loss_weight: float = 0.50
    # Directional group supervision makes HOLD difficult whenever the mined
    # future proves a tradable edge, without forcing entries on true no-edge
    # anchors. It supervises LONG/SHORT groups, never MARKET/LIMIT/STOP type.
    five_pattern_tree_direction_loss_weight: float = 1.0
    five_pattern_tree_edge_hold_ceiling: float = 0.20
    five_pattern_tree_hold_ceiling_loss_weight: float = 1.0
    five_pattern_tree_horizon_loss_weight: float = 0.50
    five_pattern_tree_path_dynamics_loss_weight: float = 0.50
    five_pattern_tree_first_hit_loss_weight: float = 0.50
    five_pattern_tree_action_path_loss_weight: float = 0.50
    five_pattern_tree_action_outcome_loss_weight: float = 0.50
    # Candidate-only signed shaping from the immutable action-local path
    # quality. It is kept separate from lifecycle economic reward and defaults
    # to zero in the public configuration to avoid double counting.
    five_pattern_action_quality_reward_scale_r: float = 0.0
    five_action_ranking_delta_r: float = 0.10
    # Counterfactual ranking is a hindsight teacher, so it must prove a
    # meaningful economic edge before it can move the shared action head.
    five_action_ranking_min_q_r: float = 0.10
    five_action_ranking_min_hold_delta_r: float = 0.10
    five_action_ranking_min_qualified_entries: int = 2
    five_action_ranking_margin_scale: float = 0.50
    five_action_ranking_margin_min: float = 0.05
    five_action_ranking_margin_max: float = 1.00
    five_action_ranking_loss_weight: float = 0.50
    # Bidirectional immutable path teacher.  When enabled, a flat entry-agent
    # HOLD can receive the causal missed-edge/no-edge credit, and an entry
    # chosen on a clearly no-edge path can be ranked back below HOLD.  The
    # default stays off so legacy selector-only profiles remain comparable.
    five_action_ranking_bidirectional_enabled: bool = False
    # Scale weight/plasticity updates during MT5 online training.  Evidence
    # stores and lifecycle accounting remain full-fidelity; only parameter
    # movement is damped so one short window cannot overwrite a stable seed.
    mt5_online_update_scale: float = 1.0
    # Negative lifecycle credit is intentionally configurable per candidate.
    # The protected v13 profile keeps the historical 8x replacement pressure,
    # while anti-overfit candidates can lower it so one loss does not flip an
    # action margin faster than the evidence supports.
    negative_context_credit_multiplier: float = 8.0
    # Extra gain when a losing entry is redirected to management.  This is
    # separate from the negative multiplier because it controls only the
    # replacement branch, not every negative neuron update.
    negative_entry_replacement_gain: float = 3.0
    # Input normalization is opt-in so legacy v13 artifacts keep their raw
    # observation contract.  Normalized candidates must be evaluated in a
    # separate artifact/config and may not overwrite the protected seed.
    input_normalization_mode: str = "raw"
    # ``role_isolated_v1`` is a hard semantic boundary on the unchanged
    # 137-feature wire vector.  Entry Direct-RNN sees market/timing only;
    # management sees position/execution only; account state sizes risk and
    # may veto an entry but never creates a BUY/SELL direction. ``legacy`` is
    # the exact protected-v13 behaviour.
    semantic_input_routing_mode: str = "legacy"
    semantic_duplicate_entry_block_enabled: bool = True
    semantic_risk_base_fraction: float = 0.50
    semantic_risk_min_fraction: float = 0.30
    semantic_risk_drawdown_soft_limit: float = 0.06
    semantic_risk_drawdown_hard_limit: float = 0.12
    # The current MT5 bridge supplies this as position-capacity exposure
    # (open positions / max positions), not monetary risk percent.  Keep the
    # conservative pressure point at 60% capacity and the hard veto near a
    # full book; a 3%/5% money-risk threshold here would veto after one trade.
    semantic_risk_open_risk_soft_limit: float = 0.60
    semantic_risk_open_risk_hard_limit: float = 0.90
    # Separate contract for long-term Hebbian/Oja updates. Raw is retained
    # only for legacy seed compatibility; new anti-overfit candidates should
    # explicitly use soft_clip_v1.
    plasticity_input_normalization_mode: str = "raw"
    plasticity_update_clip: float = 0.25
    # Direct-RNN lifecycle representation credit. It remains delayed and
    # entry-only, but every accepted gradient is applied by the persistent
    # AdamW/cosine optimizer below rather than a hand-written weight step.
    input_lifecycle_encoder_learning_enabled: bool = False
    input_lifecycle_encoder_batch_size: int = 6
    input_lifecycle_encoder_min_abs_advantage: float = 0.15
    input_lifecycle_encoder_min_sign_coherence: float = 0.55
    # Keep disabled by default so protected v13 remains a frozen baseline.
    # Candidate configs explicitly opt into optimizer-owned Direct-RNN learning.
    direct_rnn_adamw_enabled: bool = False
    direct_rnn_adamw_learning_rate: float = 0.0005
    direct_rnn_adamw_weight_decay: float = 1.0e-4
    direct_rnn_adam_beta1: float = 0.90
    direct_rnn_adam_beta2: float = 0.999
    direct_rnn_adam_epsilon: float = 1.0e-8
    direct_rnn_gradient_clip: float = 1.0
    direct_rnn_cosine_min_factor: float = 0.20
    direct_rnn_cosine_cycle_steps: int = 256
    # Candidate-only full RL lifecycle learner. It stores an immutable forward
    # cache, batches final MT5 outcomes and backpropagates through every depth
    # block before one AdamW step. Protected v13 remains disabled.
    direct_rnn_full_rl_enabled: bool = False
    # Per-expert causal Q/K/V working memory. It reads only observations that
    # preceded the current decision and is trained by the same Full-RL AdamW.
    direct_rnn_state_memory_enabled: bool = False
    direct_rnn_state_memory_capacity: int = 32
    direct_rnn_state_memory_size: int = 32
    direct_rnn_state_memory_min_history: int = 4
    direct_rnn_state_memory_temperature: float = 1.0
    direct_rnn_state_memory_injection_scale: float = 0.25
    direct_rnn_state_memory_path_delta_scale: float = 0.75
    direct_rnn_rl_batch_size: int = 8
    direct_rnn_rl_min_abs_mean_advantage: float = 0.05
    direct_rnn_rl_value_loss_weight: float = 0.25
    direct_rnn_rl_entropy_weight: float = 0.01
    # Candidate-only executed-action exploration. During learning, entry
    # experts use a bounded epsilon mixture over HOLD plus six legal entry
    # actions. Frozen/OOS and management routes remain deterministic.
    direct_rnn_rl_exploration_enabled: bool = False
    direct_rnn_rl_exploration_rate: float = 0.10
    direct_rnn_rl_exploration_min_action_probability: float = 0.02
    direct_rnn_rl_exploration_importance_clip: float = 2.0
    direct_rnn_rl_exploration_seed: int = 20260803
    # Reconcile delayed lifecycle/preference samples against the current
    # policy before AdamW. Legacy artifacts keep this disabled.
    direct_rnn_rl_delayed_reconciliation_enabled: bool = False
    direct_rnn_rl_delayed_importance_clip: float = 1.0
    direct_rnn_rl_delayed_policy_kl_temperature: float = 0.25
    direct_rnn_rl_delayed_max_policy_js: float = 0.50
    direct_rnn_rl_delayed_min_effective_weight: float = 0.02
    direct_rnn_rl_delayed_lifecycle_encoder_credit_scale: float = 0.0
    direct_rnn_rl_delayed_counterfactual_encoder_credit_scale: float = 0.25
    direct_rnn_rl_delayed_preference_target_mode: str = "on_policy_v2"
    # Delayed counterfactual supervision of the direct [HOLD, entry] logits.
    # It is consumed only inside the shared Full-RL AdamW batch.
    direct_rnn_rl_entry_preference_enabled: bool = False
    direct_rnn_rl_entry_preference_loss_weight: float = 0.25
    direct_rnn_rl_entry_preference_temperature: float = 0.50
    # Candidate-only pairwise human/expert/teacher feedback. This is disabled
    # by default until its held-out preference and MT5 frozen-OOS gates pass.
    direct_rnn_rl_rlhf_enabled: bool = False
    direct_rnn_rl_rlhf_loss_weight: float = 0.0
    direct_rnn_rl_rlhf_temperature: float = 0.25
    direct_rnn_rl_rlhf_min_confidence: float = 0.70
    direct_rnn_rl_entry_preference_horizon_bars: int = 48
    # Sparse long-horizon labels are observed without extending the full
    # counterfactual execution path. Legacy profiles keep 48; V191 uses 600.
    direct_rnn_rl_entry_abstraction_horizon_bars: int = 48
    # Observe HOLD decisions as well as selected entries. Without this, a
    # policy that learns to wait removes its own decision region from delayed
    # supervision and can drift back toward TRADE.
    direct_rnn_rl_entry_preference_include_hold_snapshots: bool = False
    # terminal_r_v1 preserves legacy labels. path_quality_v1 separates
    # direction/timing/geometry/cost on the immutable post-entry path.
    direct_rnn_rl_entry_utility_mode: str = "terminal_r_v1"
    # Timing-first curriculum: compare each action with delayed versions of
    # that same action and train only the ENTER-NOW/WAIT group boundary.
    direct_rnn_rl_timing_first_enabled: bool = False
    direct_rnn_rl_timing_delay_bars: tuple[int, ...] = (1, 2, 4, 8, 12)
    direct_rnn_rl_timing_gate_loss_weight: float = 0.25
    direct_rnn_rl_timing_margin_r: float = 0.05
    direct_rnn_rl_timing_temperature_r: float = 0.25
    direct_rnn_rl_timing_consensus_actions: int = 2
    direct_rnn_rl_timing_freeze_entry_lifecycle_actor: bool = True
    # V194 fail-closed staged curriculum.  ``curriculum_stage`` is an
    # explicit research authority; train loss can never advance it.  A stage
    # may be promoted only from paired chronological OOS evidence.
    curriculum_enabled: bool = False
    curriculum_stage: str = "representation"
    curriculum_auto_promote: bool = False
    curriculum_required_source_hash: str = ""
    curriculum_lifecycle_actor_scale: float = 0.05
    curriculum_tensor_rule_authority: str = "residual"
    curriculum_timing_probability_threshold: float = 0.65
    curriculum_timing_advantage_lcb_r: float = 0.05
    curriculum_timing_uncertainty_z: float = 1.28
    curriculum_timing_calibration_weight: float = 0.10
    curriculum_timing_consistency_weight: float = 0.10
    curriculum_timing_consistency_feature_distance: float = 0.10
    curriculum_reconstruction_mask_fraction: float = 0.15
    # Candidate retention contract. Seven existing output rows are trained as
    # TRADE/HOLD from on-policy selected-action utility plus action-given-TRADE;
    # no hindsight-best alternative may relabel a losing selected trade. The
    # replay store never writes parameters and feeds only shared Full-RL AdamW.
    direct_rnn_rl_trade_hold_separation_enabled: bool = False
    direct_rnn_rl_trade_hold_margin_r: float = 0.10
    direct_rnn_rl_trade_hold_temperature: float = 0.25
    direct_rnn_rl_trade_threshold: float = 0.55
    direct_rnn_rl_retention_enabled: bool = False
    direct_rnn_rl_retention_recent_capacity: int = 64
    direct_rnn_rl_retention_diverse_capacity: int = 128
    direct_rnn_rl_retention_anchor_capacity: int = 64
    direct_rnn_rl_retention_replay_items_per_batch: int = 8
    direct_rnn_rl_retention_replay_weight: float = 0.50
    direct_rnn_rl_retention_distillation_weight: float = 0.10
    direct_rnn_rl_retention_anchor_min_r: float = 0.25
    direct_rnn_rl_retention_audit_min_anchors: int = 2
    direct_rnn_rl_retention_max_anchor_kl_increase: float = 0.02
    direct_rnn_rl_retention_max_update_norm: float = 1.0
    # Dense delayed supervision derived from the same immutable entry path.
    # Heads are training-only and share the existing Full-RL AdamW writer.
    direct_rnn_rl_auxiliary_enabled: bool = False
    direct_rnn_rl_auxiliary_loss_weight: float = 0.10
    direct_rnn_rl_reconstruction_loss_weight: float = 0.02
    # V186 per-expert soft tensor rules.  Disabled/telemetry defaults keep
    # every legacy artifact and production action surface unchanged.
    tensor_rule_enabled: bool = False
    tensor_rule_authority: str = "telemetry"
    tensor_rule_sequence_length: int = 32
    tensor_rule_input_size: int = 78
    tensor_rule_hidden_size: int = 64
    tensor_rule_encoder_layers: int = 2
    tensor_rule_attention_heads: int = 4
    tensor_rule_ffn_size: int = 128
    tensor_rule_max_rules: int = 16
    tensor_rule_initial_rules: int = 2
    tensor_rule_min_valid_length: int = 16
    tensor_rule_unknown_membership: float = 0.55
    tensor_rule_membership_temperature: float = 0.10
    tensor_rule_motif_distance_weight: float = 0.75
    tensor_rule_utility_assignment_temperature: float = 0.15
    tensor_rule_actor_residual_clip: float = 0.25
    tensor_rule_trade_margin_r: float = 0.10
    tensor_rule_entry_lcb_gate_enabled: bool = False
    tensor_rule_effective_batch_size: int = 64
    tensor_rule_ready_gradient_norm_budget: float = 0.50
    tensor_rule_component_size: int = 16
    tensor_rule_action_latent_size: int = 32
    tensor_rule_counterfactual_encoder_credit_scale: float = 0.25
    tensor_rule_lifecycle_encoder_credit_scale: float = 0.0
    tensor_rule_utility_contrastive_margin: float = 0.20
    # Conditional representation learning labels a settled *entry* as quality
    # or non-quality within its immutable (context, action, side) bucket. The
    # auxiliary head is training-only: it has no inference gate and AdamW
    # Full-RL remains its sole writer. A PCGrad guard protects the shared RNN
    # encoder if this auxiliary gradient opposes actor evidence in the batch.
    direct_rnn_rl_conditioned_representation_enabled: bool = False
    direct_rnn_rl_conditioned_representation_loss_weight: float = 0.05
    direct_rnn_rl_conditioned_representation_min_abs_r: float = 0.10
    direct_rnn_rl_conditioned_representation_positive_weight: float = 1.25
    direct_rnn_rl_conditioned_representation_requires_actor_evidence: bool = True
    direct_rnn_rl_conditioned_representation_conflict_mode: str = "pcgrad_v1"
    # Counterfactual entry paths already label every executable action after
    # the fact.  Reuse that evidence to train the same `(context, action,
    # side)` representation head for alternatives that were not selected. It
    # remains an AdamW-batch auxiliary only: no score, mask or runtime gate
    # reads this head during inference.
    direct_rnn_rl_counterfactual_representation_enabled: bool = False
    direct_rnn_rl_counterfactual_representation_loss_weight: float = 0.08
    direct_rnn_rl_counterfactual_representation_temperature: float = 0.50
    # Candidate-only actor advantage contract. ``ema_v1`` preserves the
    # legacy lifecycle signal; ``critic_asymmetric_v1`` uses the cached
    # state-value prediction as a gradually trusted baseline, protects net
    # wins from a negative actor update, and can normalize the residual.
    direct_rnn_rl_advantage_mode: str = "ema_v1"
    direct_rnn_rl_critic_warmup_lifecycles: int = 32
    direct_rnn_rl_critic_max_weight: float = 0.85
    direct_rnn_rl_profit_floor: float = 0.05
    direct_rnn_rl_loss_floor: float = 0.05
    direct_rnn_rl_advantage_normalization_enabled: bool = False
    direct_rnn_rl_advantage_norm_decay: float = 0.95
    direct_rnn_rl_advantage_norm_scale_floor: float = 0.35
    direct_rnn_rl_advantage_norm_clip: float = 2.5
    # Number of compact base-RNN transitions retained with an entry trace for
    # lifecycle TBPTT.  One is the historical local-credit behaviour.
    direct_rnn_rl_temporal_bptt_steps: int = 8
    # Zero keeps the complete execution trace for a bounded diagnostic run.
    # The normal default remains a rolling 64 MiB journal.
    execution_trace_observatory_max_bytes: int = 64 * 1024 * 1024
    # Keep full recurrent tensors for deep diagnostics by default. Causal A/B
    # can disable them while retaining action/receipt/lifecycle ledger fields.
    execution_trace_observatory_include_recurrent_state: bool = True
    # Bound the short-term-depression (STP) drive before it multiplies the
    # incoming weights.  ``50`` is the legacy-equivalent setting (the STP
    # state is already capped at 5); smaller candidate values prevent extreme
    # raw feature tails from turning into hidden-activation saturation.
    stp_input_clip: float = 50.0
    # Every Direct-RNN hidden layer is regularized.  This is a runtime safety
    # contract, not a candidate switch: configuration may tune the strength
    # but may not disable either hidden normalization or deterministic dropout.
    hidden_normalization_mode: str = "layernorm_v1"
    hidden_normalization_target_rms: float = 1.0
    hidden_normalization_clip: float = 3.0
    hidden_normalization_eps: float = 1e-5
    hidden_normalization_enabled: bool = True
    hidden_normalization_layers: tuple[bool, ...] = (True, True, True, True, True, True)
    # Layer-wise slots are resolved against ``model.hidden_layers``. Shorter
    # lists are expanded by repeating the final value; this keeps a deeper
    # configured stack regularized instead of silently disabling dropout.
    hidden_dropout_enabled: bool = True
    hidden_dropout_rates: tuple[float, ...] = (0.05, 0.10, 0.10, 0.05, 0.10, 0.10)
    hidden_dropout_seed: int = 0
    hidden_dropout_train_only: bool = True
    # Candidate-only output calibration. Raw is the exact v13 behavior.
    output_normalization_mode: str = "raw"
    output_normalization_target_rms: float = 0.70
    output_normalization_clip: float = 3.0
    # Parameter activation remains a compatibility setting for risk and
    # pending-entry offset. Fixed V13 SL/TP never decode policy logits.
    execution_parameter_activation_mode: str = "legacy_tanh_v1"
    # Candidate-only dedicated timing network. Legacy v13 keeps this disabled
    # so loading the protected seed does not alter its action surface.
    timing_network_enabled: bool = False
    timing_network_action_scale: float = 0.10
    timing_network_update_scale: float = 0.25
    # Optional candidate-only lifecycle scale for TimingNet. Zero inherits
    # mt5_online_update_scale; a positive value decouples timing credit from
    # the broader graph/policy plasticity envelope.
    timing_network_lifecycle_scale: float = 0.0
    timing_network_learning_rate: float = 0.0005
    timing_network_hidden_size: int = 32
    timing_network_embedding_size: int = 16
    timing_network_bptt_window: int = 24
    # Number of closed M15 rows materialized as one causal TimingNet tensor.
    # It is independent from the delayed lifecycle-credit BPTT horizon.
    timing_network_sequence_length: int = 32
    timing_network_eligibility_decay: float = 0.90
    timing_network_gradient_clip: float = 1.0
    # Causal self-supervised target: predict whether the live deterministic
    # trend score will materially decay toward zero after later decisions.
    # This is candidate-only and independent from realised lifecycle credit.
    timing_network_deceleration_enabled: bool = False
    timing_network_deceleration_horizon: int = 3
    timing_network_deceleration_min_trend_strength: float = 0.18
    timing_network_deceleration_min_drop: float = 0.10
    timing_network_deceleration_update_scale: float = 0.25
    # The auxiliary classifier starts from a zero head.  It therefore needs a
    # separate, materially faster BCE step than the lifecycle/R optimizer;
    # the encoder stays deliberately slow so the dense label cannot overfit
    # the trading representation.
    timing_network_deceleration_learning_rate: float = 0.01
    timing_network_deceleration_encoder_scale: float = 0.10
    timing_network_deceleration_l2: float = 1e-4
    # Causal regime-transition auxiliary task. TimingNet still receives only
    # market/timing/Wyckoff inputs; a confirmed context label arrives only H
    # decisions later. The head never writes policy logits directly; the
    # context facade may block an entry only after independent calibration.
    timing_network_regime_transition_enabled: bool = False
    timing_network_regime_transition_horizons: tuple[int, ...] = (3, 6, 12)
    timing_network_regime_transition_update_scale: float = 0.25
    timing_network_regime_transition_learning_rate: float = 0.01
    timing_network_regime_transition_encoder_scale: float = 0.08
    timing_network_regime_transition_l2: float = 1e-4
    # Candidate-only SWATS/PID controller for TimingNet. Disabled by default
    # so the protected seed and old timing candidates keep their optimizer
    # contract until a separate artifact explicitly opts in.
    timing_network_swats_enabled: bool = False
    timing_network_pid_enabled: bool = False
    timing_network_adam_beta1: float = 0.90
    timing_network_adam_beta2: float = 0.999
    timing_network_adam_epsilon: float = 1e-8
    # AdamW/cosine act only during the Adam phase before SWATS switches to
    # PID-controlled SGD.  Defaults preserve legacy and protected v13 paths.
    timing_network_adamw_enabled: bool = False
    timing_network_adamw_weight_decay: float = 0.0
    timing_network_cosine_annealing_enabled: bool = False
    timing_network_cosine_min_factor: float = 1.0
    timing_network_cosine_cycle_steps: int = 128
    timing_network_projection_beta: float = 0.90
    timing_network_projection_tolerance: float = 0.10
    timing_network_switch_consecutive: int = 6
    timing_network_min_projection_steps: int = 12
    timing_network_target_alignment: float = 0.90
    timing_network_sgd_eta_min: float = 1e-6
    timing_network_sgd_eta_max: float = 0.004
    timing_network_sgd_eta_init_factor: float = 0.90
    timing_network_pid_kp: float = 0.08
    timing_network_pid_ki: float = 0.01
    timing_network_pid_kd: float = 0.02
    timing_network_pid_integral_limit: float = 2.0
    timing_network_pid_derivative_ema: float = 0.80
    timing_network_pid_deadband: float = 0.02
    timing_network_pid_r_target: float = 0.0
    timing_network_pid_r_scale_floor: float = 0.25
    timing_network_pid_alignment_weight: float = 0.70
    timing_network_pid_r_weight: float = 0.30
    timing_network_pid_bad_steps_to_freeze: int = 4
    timing_network_pid_recovery_steps: int = 3
    timing_network_pid_cooldown_steps: int = 8
    timing_network_pid_fallback_to_adam: bool = False
    # Canonical feature timestamps are UTC-naive.  The EA converts server bar
    # timestamps with this explicit offset before feature/session encoding and
    # audit export; no host-local timezone is ever inferred.
    session_timezone: str = "UTC"
    mt5_server_utc_offset_hours: int = 0
    spread_semantics: str = "ask_minus_bid_price"
    # Bounded, stratified evidence store.  These are learning-memory budgets,
    # not trading-entry limits.
    experience_buffer_max: int = 4096
    rejection_memory_max: int = 2048
    execution_trace_cache_max: int = 8192
    memory_neurons_max: int = 4096
    memory_connections_max: int = 16384
    mistake_episodes_max: int = 2048
    # Every independently trained candidate hash must pass every OOS window.
    promotion_min_oos_windows: int = 3
    promotion_min_seeds: int = 3
    # Strict frozen-OOS gates. These only control promotion; lifecycle
    # learning, replay and selector updates continue before the gate.
    promotion_require_fitness_eligible: bool = True
    promotion_require_reconciled: bool = True
    promotion_min_oos_trades: int = 30
    promotion_max_oos_drawdown_pct: float = 10.0
    promotion_min_policy_expectancy_lcb: float = 0.0
    promotion_max_negative_context_scopes: int = 0
    promotion_require_feature_parity: bool = True
    promotion_require_real_tick_evidence: bool = True
    tier1_vicreg_regularizer_enabled: bool = False
    tier1_regularizer_enabled: bool = False
    tier1_reservoir_buffer_enabled: bool = False
    tier1_ewc_enabled: bool = False
    tier2_compact_core_enabled: bool = False
    tier2_coma_enabled: bool = False
    tier2_rnd_enabled: bool = False

    def __post_init__(self) -> None:
        if int(self.seed) < 0:
            raise ValueError("learning.seed must be non-negative")
        if str(self.session_timezone).upper() != "UTC":
            raise ValueError("learning.session_timezone must be UTC for the MT5 feature contract")
        if not -23 <= int(self.mt5_server_utc_offset_hours) <= 23:
            raise ValueError("learning.mt5_server_utc_offset_hours must be within [-23, 23]")
        if str(self.spread_semantics) != "ask_minus_bid_price":
            raise ValueError("learning.spread_semantics must be ask_minus_bid_price")
        if str(self.live_curiosity_mode).strip().lower() not in {"shadow_only", "evidence_only"}:
            raise ValueError("learning.live_curiosity_mode must be shadow_only or evidence_only")
        if not 1.0 <= float(self.action_recovery_min_support) <= 256.0:
            raise ValueError("learning.action_recovery_min_support must be within [1, 256]")
        if not -1.0 <= float(self.action_recovery_min_lcb) <= 2.0:
            raise ValueError("learning.action_recovery_min_lcb must be within [-1, 2]")
        if not 0.50 <= float(self.action_recovery_merge_similarity) < 1.0:
            raise ValueError("learning.action_recovery_merge_similarity must be within [0.50, 1.0)")
        if not 0.01 <= float(self.action_recovery_shadow_weight) <= 1.0:
            raise ValueError("learning.action_recovery_shadow_weight must be within [0.01, 1]")
        if int(self.action_recovery_max_cells_per_action) < 4:
            raise ValueError("learning.action_recovery_max_cells_per_action must be at least 4")
        if not 1.0 <= float(self.action_recovery_negative_veto_min_real_support) <= 256.0:
            raise ValueError("learning.action_recovery_negative_veto_min_real_support must be within [1, 256]")
        if not -2.0 <= float(self.action_recovery_negative_veto_lcb) <= 0.0:
            raise ValueError("learning.action_recovery_negative_veto_lcb must be within [-2, 0]")
        if int(self.entry_loss_guard_min_closed) < 1:
            raise ValueError("learning.entry_loss_guard_min_closed must be at least 1")
        if float(self.entry_loss_guard_max_closed_loss_r) < 0.0:
            raise ValueError("learning.entry_loss_guard_max_closed_loss_r must be non-negative")
        if int(self.entry_loss_guard_consecutive_losses) < 1:
            raise ValueError("learning.entry_loss_guard_consecutive_losses must be at least 1")
        if int(self.train_entry_probe_after_decisions) < 1:
            raise ValueError("learning.train_entry_probe_after_decisions must be at least 1")
        if int(self.train_entry_probe_interval_decisions) < 1:
            raise ValueError("learning.train_entry_probe_interval_decisions must be at least 1")
        if int(self.train_entry_probe_max_per_day) < 0:
            raise ValueError("learning.train_entry_probe_max_per_day must be non-negative")
        if int(self.train_entry_probe_max_per_run) < 0:
            raise ValueError("learning.train_entry_probe_max_per_run must be non-negative")
        if not 0.0 <= float(self.train_entry_probe_min_direction_quality) <= 1.0:
            raise ValueError("learning.train_entry_probe_min_direction_quality must be within [0, 1]")
        if not 0.0 <= float(self.context_direction_entry_gate_min_timing_quality) <= 1.0:
            raise ValueError("learning.context_direction_entry_gate_min_timing_quality must be within [0, 1]")
        if int(self.runtime_fitness_rolling_window) < 1:
            raise ValueError("learning.runtime_fitness_rolling_window must be at least 1")
        if int(self.runtime_fitness_min_closed) < 1:
            raise ValueError("learning.runtime_fitness_min_closed must be at least 1")
        if not -5.0 <= float(self.runtime_fitness_min_score) <= 5.0:
            raise ValueError("learning.runtime_fitness_min_score must be within [-5, 5]")
        if not 0.0 <= float(self.runtime_fitness_max_immediate_exit_ratio) <= 1.0:
            raise ValueError("learning.runtime_fitness_max_immediate_exit_ratio must be within [0, 1]")
        if float(self.runtime_fitness_max_daily_loss_r) < 0.0:
            raise ValueError("learning.runtime_fitness_max_daily_loss_r must be non-negative")
        if float(self.runtime_fitness_max_run_loss_r) < 0.0:
            raise ValueError("learning.runtime_fitness_max_run_loss_r must be non-negative")
        if int(self.runtime_fitness_max_daily_trades) < 0:
            raise ValueError("learning.runtime_fitness_max_daily_trades must be non-negative")
        if not 0.0 < float(self.runtime_fitness_max_drawdown_pct) <= 100.0:
            raise ValueError("learning.runtime_fitness_max_drawdown_pct must be within (0, 100]")
        if int(self.five_training_schedule_steps) < 64:
            raise ValueError("learning.five_training_schedule_steps must be at least 64")
        for name in (
            "five_reward_drawdown_weight", "five_reward_risk_excess_weight",
            "five_reward_conflict_weight", "five_reward_competition_weight",
            "five_reward_terminal_rank_weight", "five_reward_settlement_weight",
            "five_reward_mark_to_market_weight", "five_reward_settlement_clip_r",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 5.0:
                raise ValueError(f"learning.{name} must be within [0, 5]")
        if str(self.five_reward_mode).strip().lower() not in {
            "dense_mtm", "settlement_net_r_v1"
        }:
            raise ValueError(
                "learning.five_reward_mode must be dense_mtm or settlement_net_r_v1"
            )
        if not 0.0 <= float(self.five_pattern_tree_edge_hold_ceiling) <= 1.0:
            raise ValueError("learning.five_pattern_tree_edge_hold_ceiling must be within [0, 1]")
        if not 0.0 <= float(self.five_reward_wr_target) <= 1.0:
            raise ValueError("learning.five_reward_wr_target must be within [0, 1]")
        for name in (
            "five_reward_wr_balance_weight", "five_reward_expectancy_floor_r",
            "five_reward_expectancy_shortfall_weight", "five_reward_tail_loss_threshold_r",
            "five_reward_tail_loss_weight", "five_reward_tail_loss_cap_r",
            "five_lifecycle_shape_clip_r", "five_lifecycle_min_sl_atr",
            "five_lifecycle_min_tp_atr", "five_lifecycle_geometry_penalty_r",
            "five_lifecycle_geometry_bonus_r", "five_lifecycle_initial_sl_local_loss_weight",
            "five_lifecycle_net_r_credit_weight", "five_lifecycle_net_r_credit_clip_r",
            "five_lifecycle_mfe_penalty_zero_r", "five_lifecycle_mfe_penalty_010_r",
            "five_lifecycle_mfe_penalty_025_r", "five_lifecycle_mfe_penalty_neutral_at_r",
            "five_lifecycle_excursion_bonus_at_r", "five_lifecycle_excursion_good_mae_r",
            "five_lifecycle_excursion_bad_mae_r", "five_lifecycle_excursion_min_bonus_r",
            "five_lifecycle_excursion_max_bonus_r", "five_lifecycle_trailing_activation_mfe_r",
            "five_lifecycle_trailing_capture_min_ratio", "five_lifecycle_trailing_capture_target_ratio",
            "five_lifecycle_trailing_missed_penalty_r", "five_lifecycle_trailing_small_capture_penalty_r",
            "five_lifecycle_trailing_min_bonus_r", "five_lifecycle_trailing_max_bonus_r",
            "five_lifecycle_exit_capture_min_ratio", "five_lifecycle_exit_capture_target_ratio",
            "five_lifecycle_exit_capture_min_bonus_r", "five_lifecycle_exit_capture_max_bonus_r",
            "five_lifecycle_recovery_min_bonus_r", "five_lifecycle_recovery_max_bonus_r",
            "five_lifecycle_pending_fill_min_bonus_r", "five_lifecycle_pending_fill_max_bonus_r",
            "five_lifecycle_batch_neutral_band_r", "five_position_improvement_reward_weight",
            "five_flat_hold_missed_edge_penalty_r", "five_flat_hold_confirmed_no_edge_bonus_r",
            "five_flat_hold_edge_threshold_r",
            "five_pattern_tree_auxiliary_loss_weight", "five_pattern_tree_utility_loss_weight",
            "five_pattern_tree_min_edge_atr", "five_pattern_tree_mfe_weight",
            "five_pattern_tree_mae_weight", "five_pattern_tree_utility_clip_atr",
            "five_pattern_tree_excursion_loss_weight", "five_pattern_tree_no_edge_loss_weight",
            "five_pattern_tree_direction_loss_weight", "five_pattern_tree_edge_hold_ceiling",
            "five_pattern_tree_hold_ceiling_loss_weight",
            "five_pattern_tree_horizon_loss_weight", "five_pattern_tree_path_dynamics_loss_weight",
            "five_pattern_tree_first_hit_loss_weight", "five_pattern_tree_action_path_loss_weight",
            "five_pattern_tree_action_outcome_loss_weight",
            "five_pattern_action_quality_reward_scale_r",
            "five_action_ranking_min_q_r", "five_action_ranking_min_hold_delta_r",
            "five_invalid_position_action_no_position_penalty_r",
            "five_invalid_cancel_no_pending_penalty_r",
            "five_unfilled_pending_timeout_penalty_r",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 5.0:
                raise ValueError(f"learning.{name} must be within [0, 5]")
        if float(self.five_lifecycle_excursion_bad_mae_r) <= float(self.five_lifecycle_excursion_good_mae_r):
            raise ValueError("learning five lifecycle bad MAE must exceed good MAE")
        if float(self.five_lifecycle_excursion_max_bonus_r) < float(self.five_lifecycle_excursion_min_bonus_r):
            raise ValueError("learning five lifecycle excursion max bonus must be >= min bonus")
        for prefix in ("trailing_capture", "exit_capture"):
            low = float(getattr(self, f"five_lifecycle_{prefix}_min_ratio"))
            high = float(getattr(self, f"five_lifecycle_{prefix}_target_ratio"))
            if not 0.0 <= low < high <= 1.0:
                raise ValueError(f"learning five lifecycle {prefix} ratios must satisfy 0 <= min < target <= 1")
        if float(self.five_lifecycle_trailing_max_bonus_r) < float(self.five_lifecycle_trailing_min_bonus_r):
            raise ValueError("learning five lifecycle trailing max bonus must be >= min bonus")
        if float(self.five_lifecycle_exit_capture_max_bonus_r) < float(self.five_lifecycle_exit_capture_min_bonus_r):
            raise ValueError("learning five lifecycle exit-capture max bonus must be >= min bonus")
        if float(self.five_lifecycle_pending_fill_max_bonus_r) < float(self.five_lifecycle_pending_fill_min_bonus_r):
            raise ValueError("learning five lifecycle pending-fill max bonus must be >= min bonus")
        if int(self.five_lifecycle_pending_fill_max_bars) < 1:
            raise ValueError("learning five lifecycle pending-fill max bars must be >= 1")
        if int(self.five_lifecycle_batch_size) < 16:
            raise ValueError("learning five lifecycle batch size must be at least 16")
        if int(self.five_lifecycle_batch_max_wait) < int(self.five_lifecycle_batch_size):
            raise ValueError("learning five lifecycle batch max wait must be >= batch size")
        if int(self.five_failure_replay_capacity) < 8:
            raise ValueError("learning.five_failure_replay_capacity must be >= 8")
        if not 0 <= int(self.five_failure_replay_items_per_batch) <= 64:
            raise ValueError("learning.five_failure_replay_items_per_batch must be within [0, 64]")
        if not 1 <= int(self.five_failure_replay_min_bucket_support) <= 64:
            raise ValueError("learning.five_failure_replay_min_bucket_support must be within [1, 64]")
        if not 0 <= int(self.five_failure_replay_audit_items) <= 256:
            raise ValueError("learning.five_failure_replay_audit_items must be within [0, 256]")
        if not 0.0 <= float(self.five_failure_replay_weight) <= 2.0:
            raise ValueError("learning.five_failure_replay_weight must be within [0, 2]")
        if not -3.0 <= float(self.five_failure_replay_admit_threshold_r) <= -0.01:
            raise ValueError("learning.five_failure_replay_admit_threshold_r must be within [-3, -0.01]")
        if not 0.0 <= float(self.five_failure_replay_repair_probability) <= 0.50:
            raise ValueError("learning.five_failure_replay_repair_probability must be within [0, 0.5]")
        if not float(self.five_failure_replay_repair_probability) <= float(self.five_failure_replay_reopen_probability) <= 0.75:
            raise ValueError("learning failure replay reopen probability must be >= repair probability and <= 0.75")
        if not 1 <= int(self.five_failure_replay_repair_streak) <= 32:
            raise ValueError("learning.five_failure_replay_repair_streak must be within [1, 32]")
        if not 1 <= int(self.five_failure_replay_audit_interval) <= 256:
            raise ValueError("learning.five_failure_replay_audit_interval must be within [1, 256]")
        if not 0 <= int(self.five_mt5_failure_replay_sessions_per_run) <= 256:
            raise ValueError("learning.five_mt5_failure_replay_sessions_per_run must be within [0, 256]")
        if not 1 <= int(self.five_mt5_failure_replay_max_attempts_per_case) <= 64:
            raise ValueError("learning.five_mt5_failure_replay_max_attempts_per_case must be within [1, 64]")
        if not 1 <= int(self.five_mt5_failure_replay_warmup_minutes) <= 1440:
            raise ValueError("learning.five_mt5_failure_replay_warmup_minutes must be within [1, 1440]")
        if not 1 <= int(self.five_mt5_failure_replay_horizon_minutes) <= 2880:
            raise ValueError("learning.five_mt5_failure_replay_horizon_minutes must be within [1, 2880]")
        if not 0 <= int(self.five_mt5_failure_replay_resolved_audit_items) <= 256:
            raise ValueError("learning.five_mt5_failure_replay_resolved_audit_items must be within [0, 256]")
        if not 0.0 <= float(self.five_behavior_uniform_mix) <= 1.0:
            raise ValueError("learning.five_behavior_uniform_mix must be within [0, 1]")
        if str(self.five_action_decision_mode).strip().lower() not in {
            "checkpoint", "expected_value", "direct"
        }:
            raise ValueError(
                "learning.five_action_decision_mode must be checkpoint, "
                "expected_value, or direct"
            )
        if not 2 <= int(self.five_gae_min_feedback_steps) <= 256:
            raise ValueError("learning.five_gae_min_feedback_steps must be within [2, 256]")
        if not 0 <= int(self.five_imitation_freeze_updates) <= 4096:
            raise ValueError("learning.five_imitation_freeze_updates must be within [0, 4096]")
        if not int(self.five_imitation_freeze_updates) <= int(self.five_imitation_anchor_updates) <= 65536:
            raise ValueError(
                "learning.five_imitation_anchor_updates must be >= freeze updates and <= 65536"
            )
        if not 0.0 <= float(self.five_imitation_anchor_weight) <= 10.0:
            raise ValueError("learning.five_imitation_anchor_weight must be within [0, 10]")
        if not 0.0 <= float(self.five_imitation_warmup_uniform_mix_max) <= 1.0:
            raise ValueError(
                "learning.five_imitation_warmup_uniform_mix_max must be within [0, 1]"
            )
        if not 1 <= int(self.five_pattern_unroll) <= 64:
            raise ValueError("learning.five_pattern_unroll must be within [1, 64]")
        if not 0.0 <= float(self.five_action_ranking_min_q_r) <= 5.0:
            raise ValueError("learning.five_action_ranking_min_q_r must be within [0, 5]")
        if not 0.0 <= float(self.five_action_ranking_min_hold_delta_r) <= 5.0:
            raise ValueError("learning.five_action_ranking_min_hold_delta_r must be within [0, 5]")
        if not 1 <= int(self.five_action_ranking_min_qualified_entries) <= 6:
            raise ValueError("learning.five_action_ranking_min_qualified_entries must be within [1, 6]")
        if not 1 <= int(self.five_reasoning_steps) <= 8:
            raise ValueError("learning.five_reasoning_steps must be within [1, 8]")
        if not 2 <= int(self.five_reasoning_group_size) <= 32:
            raise ValueError("learning.five_reasoning_group_size must be within [2, 32]")
        if not 0.0 < float(self.five_reasoning_ppo_clip) <= 1.0:
            raise ValueError("learning.five_reasoning_ppo_clip must be within (0, 1]")
        for name in (
            "five_reasoning_kl_beta", "five_reasoning_entropy_weight",
            "five_reasoning_loss_weight", "five_reasoning_min_group_std_r",
            "five_reasoning_action_residual_clip",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 10.0:
                raise ValueError(f"learning.{name} must be within [0, 10]")
        for name in (
            "five_reasoning_encoder_credit_scale", "five_reasoning_action_scale",
        ):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"learning.{name} must be within [0, 1]")
        if not 0.0 <= float(self.five_behavior_entry_probability_floor) <= (1.0 / 6.0):
            raise ValueError(
                "learning.five_behavior_entry_probability_floor must be within [0, 1/6]"
            )
        if not 0.0 <= float(self.five_flat_entry_mass_floor_end) <= float(self.five_flat_entry_mass_floor_start) <= 1.0:
            raise ValueError(
                "learning five flat-entry mass floor must satisfy 0 <= end <= start <= 1"
            )
        if float(self.five_flat_entry_mass_loss_weight) < 0.0:
            raise ValueError("learning.five_flat_entry_mass_loss_weight must be non-negative")
        if float(self.five_flat_entry_mass_log_loss_weight) < 0.0:
            raise ValueError("learning.five_flat_entry_mass_log_loss_weight must be non-negative")
        if not 0.0 <= float(self.five_flat_entry_direction_mass_floor_end) <= float(self.five_flat_entry_direction_mass_floor_start) <= 0.5:
            raise ValueError(
                "learning five directional entry floor must satisfy 0 <= end <= start <= 0.5"
            )
        if float(self.five_flat_entry_direction_mass_loss_weight) < 0.0:
            raise ValueError(
                "learning.five_flat_entry_direction_mass_loss_weight must be non-negative"
            )
        if int(self.five_live_action_veto_min_samples) < 1:
            raise ValueError("learning.five_live_action_veto_min_samples must be at least 1")
        if not 0.0 <= float(self.five_live_action_veto_min_confidence) <= 1.0:
            raise ValueError("learning.five_live_action_veto_min_confidence must be within [0, 1]")
        if not -12.0 <= float(self.five_live_action_veto_min_lcb_r) <= 0.0:
            raise ValueError("learning.five_live_action_veto_min_lcb_r must be within [-12, 0]")
        if not isinstance(self.five_live_action_veto_probe_enabled, bool):
            raise ValueError("learning.five_live_action_veto_probe_enabled must be boolean")
        if int(self.five_live_action_veto_probe_interval_decisions) < 1:
            raise ValueError("learning.five_live_action_veto_probe_interval_decisions must be at least 1")
        if not isinstance(self.five_live_action_veto_probe_on_negative_transition, bool):
            raise ValueError(
                "learning.five_live_action_veto_probe_on_negative_transition must be boolean"
            )
        if int(self.five_live_owner_exploration_min_samples) < 1:
            raise ValueError("learning.five_live_owner_exploration_min_samples must be at least 1")
        if not 0.0 <= float(self.five_live_owner_exploration_min_confidence) <= 1.0:
            raise ValueError("learning.five_live_owner_exploration_min_confidence must be within [0, 1]")
        if not -12.0 <= float(self.five_live_owner_exploration_min_lcb_r) <= 0.0:
            raise ValueError("learning.five_live_owner_exploration_min_lcb_r must be within [-12, 0]")
        if not 0.0 <= float(self.five_live_owner_exploration_min_mix) <= float(self.five_behavior_uniform_mix):
            raise ValueError("learning.five_live_owner_exploration_min_mix must be within [0, five_behavior_uniform_mix]")
        if not isinstance(self.five_live_owner_bootstrap_probe_enabled, bool):
            raise ValueError("learning.five_live_owner_bootstrap_probe_enabled must be boolean")
        if int(self.five_live_owner_bootstrap_probe_interval_decisions) < 1:
            raise ValueError("learning.five_live_owner_bootstrap_probe_interval_decisions must be at least 1")
        if int(self.five_live_owner_bootstrap_probe_max_per_run) < 0:
            raise ValueError("learning.five_live_owner_bootstrap_probe_max_per_run must be non-negative")
        if not isinstance(self.five_live_owner_bootstrap_probe_force_market, bool):
            raise ValueError("learning.five_live_owner_bootstrap_probe_force_market must be boolean")
        if not -5.0 <= float(self.five_flat_hold_no_edge_threshold_r) <= 0.0:
            raise ValueError("learning.five_flat_hold_no_edge_threshold_r must be within [-5, 0]")
        if int(self.five_entropy_edge_release_count) < 1:
            raise ValueError("learning.five_entropy_edge_release_count must be at least 1")
        if int(self.five_entropy_edge_decay_count) <= int(self.five_entropy_edge_release_count):
            raise ValueError(
                "learning.five_entropy_edge_decay_count must exceed five_entropy_edge_release_count"
            )
        if float(self.five_entropy_start) < 0.0:
            raise ValueError("learning.five_entropy_start must be non-negative")
        if float(self.five_action_ranking_margin_max) < float(self.five_action_ranking_margin_min):
            raise ValueError("learning.five_action_ranking_margin_max must not be below margin_min")
        if not 0.0 <= float(self.mt5_online_update_scale) <= 1.0:
            raise ValueError("learning.mt5_online_update_scale must be within [0, 1]")
        if int(self.five_codex_teacher_max_cases_per_run) < 0:
            raise ValueError("learning.five_codex_teacher_max_cases_per_run must be non-negative")
        if int(self.five_rlhf_recovery_snapshot_interval_updates) < 0:
            raise ValueError(
                "learning.five_rlhf_recovery_snapshot_interval_updates must be non-negative"
            )
        if int(self.five_codex_teacher_timeout_sec) < 10:
            raise ValueError("learning.five_codex_teacher_timeout_sec must be at least 10")
        if not 0.0 <= float(self.five_codex_teacher_online_wait_sec) <= 300.0:
            raise ValueError(
                "learning.five_codex_teacher_online_wait_sec must be within [0, 300]"
            )
        if not 1 <= int(self.five_codex_teacher_max_workers) <= 8:
            raise ValueError(
                "learning.five_codex_teacher_max_workers must be within [1, 8]"
            )
        if not 0.0 <= float(self.five_codex_teacher_min_confidence) <= 1.0:
            raise ValueError("learning.five_codex_teacher_min_confidence must be within [0, 1]")
        if not 0.0 < float(self.five_codex_teacher_reward_scale) <= 0.5:
            raise ValueError("learning.five_codex_teacher_reward_scale must be within (0, 0.5]")
        if not 0.0 < float(self.five_codex_teacher_max_probability_shift) <= 0.25:
            raise ValueError(
                "learning.five_codex_teacher_max_probability_shift must be within (0, 0.25]"
            )
        if not 0.0 < float(self.five_codex_teacher_learning_rate) <= 0.001:
            raise ValueError(
                "learning.five_codex_teacher_learning_rate must be within (0, 0.001]"
            )
        if not 0.90 <= float(self.five_codex_context_similarity_threshold) < 1.0:
            raise ValueError(
                "learning.five_codex_context_similarity_threshold must be within [0.90, 1)"
            )
        if not 0.0 <= float(self.five_codex_context_support_floor) <= 1.0:
            raise ValueError(
                "learning.five_codex_context_support_floor must be within [0, 1]"
            )
        if not 0.0 <= float(self.five_codex_global_action_bias_scale) <= 0.5:
            raise ValueError(
                "learning.five_codex_global_action_bias_scale must be within [0, 0.5]"
            )
        if not isinstance(self.five_codex_teacher_hypothesis_required, bool):
            raise ValueError("learning.five_codex_teacher_hypothesis_required must be boolean")
        if not isinstance(self.five_codex_teacher_auto_connect, bool):
            raise ValueError("learning.five_codex_teacher_auto_connect must be boolean")
        if not 0.0 <= float(self.negative_context_credit_multiplier) <= 16.0:
            raise ValueError("learning.negative_context_credit_multiplier must be within [0, 16]")
        if not 0.0 <= float(self.negative_entry_replacement_gain) <= 8.0:
            raise ValueError("learning.negative_entry_replacement_gain must be within [0, 8]")
        normalization_modes = {"raw", "tanh_v1", "soft_clip_v1", "groupwise_v2", "groupwise_mtf_v3", "structure_aware_v4"}
        if str(self.input_normalization_mode).strip().lower() not in normalization_modes:
            raise ValueError("learning.input_normalization_mode must be raw, tanh_v1, soft_clip_v1, groupwise_v2, groupwise_mtf_v3, or structure_aware_v4")
        if str(self.semantic_input_routing_mode).strip().lower() not in {"legacy", "role_isolated_v1"}:
            raise ValueError("learning.semantic_input_routing_mode must be legacy or role_isolated_v1")
        if not 0.27 <= float(self.semantic_risk_min_fraction) <= 0.73:
            raise ValueError("learning.semantic_risk_min_fraction must be within [0.27, .73]")
        if not float(self.semantic_risk_min_fraction) <= float(self.semantic_risk_base_fraction) <= 0.73:
            raise ValueError("learning.semantic_risk_base_fraction must be within [semantic_risk_min_fraction, .73]")
        if not 0.0 < float(self.semantic_risk_drawdown_soft_limit) <= float(self.semantic_risk_drawdown_hard_limit) <= 1.0:
            raise ValueError("semantic drawdown soft/hard limits must satisfy 0 < soft <= hard <= 1")
        if not 0.0 < float(self.semantic_risk_open_risk_soft_limit) <= float(self.semantic_risk_open_risk_hard_limit) <= 2.0:
            raise ValueError("semantic open-risk soft/hard limits must satisfy 0 < soft <= hard <= 2")
        if str(self.plasticity_input_normalization_mode).strip().lower() not in normalization_modes:
            raise ValueError("learning.plasticity_input_normalization_mode must be raw, tanh_v1, soft_clip_v1, groupwise_v2, groupwise_mtf_v3, or structure_aware_v4")
        if not 0.0 <= float(self.plasticity_update_clip) <= 10.0:
            raise ValueError("learning.plasticity_update_clip must be within [0, 10]")
        if not 2 <= int(self.input_lifecycle_encoder_batch_size) <= 64:
            raise ValueError("learning.input_lifecycle_encoder_batch_size must be within [2, 64]")
        if not 0.0 <= float(self.input_lifecycle_encoder_min_abs_advantage) <= 3.0:
            raise ValueError("learning.input_lifecycle_encoder_min_abs_advantage must be within [0, 3]")
        if not 0.0 <= float(self.input_lifecycle_encoder_min_sign_coherence) <= 1.0:
            raise ValueError("learning.input_lifecycle_encoder_min_sign_coherence must be within [0, 1]")
        if not 1e-6 <= float(self.direct_rnn_adamw_learning_rate) <= 0.02:
            raise ValueError("learning.direct_rnn_adamw_learning_rate must be within [1e-6, .02]")
        if not 0.0 <= float(self.direct_rnn_adamw_weight_decay) <= 0.10:
            raise ValueError("learning.direct_rnn_adamw_weight_decay must be within [0, .10]")
        if not 0.0 <= float(self.direct_rnn_adam_beta1) < 1.0:
            raise ValueError("learning.direct_rnn_adam_beta1 must be within [0, 1)")
        if not 0.0 <= float(self.direct_rnn_adam_beta2) < 1.0:
            raise ValueError("learning.direct_rnn_adam_beta2 must be within [0, 1)")
        if not 1e-12 <= float(self.direct_rnn_adam_epsilon) <= 1e-3:
            raise ValueError("learning.direct_rnn_adam_epsilon must be within [1e-12, 1e-3]")
        if not 1e-6 <= float(self.direct_rnn_gradient_clip) <= 100.0:
            raise ValueError("learning.direct_rnn_gradient_clip must be within [1e-6, 100]")
        if not 0.0 <= float(self.direct_rnn_cosine_min_factor) <= 1.0:
            raise ValueError("learning.direct_rnn_cosine_min_factor must be within [0, 1]")
        if not 1 <= int(self.direct_rnn_cosine_cycle_steps) <= 100_000:
            raise ValueError("learning.direct_rnn_cosine_cycle_steps must be within [1, 100000]")
        if not 2 <= int(self.direct_rnn_rl_batch_size) <= 64:
            raise ValueError("learning.direct_rnn_rl_batch_size must be within [2, 64]")
        if not 0.0 <= float(self.direct_rnn_rl_min_abs_mean_advantage) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_min_abs_mean_advantage must be within [0, 3]")
        if not 0.0 <= float(self.direct_rnn_rl_value_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_value_loss_weight must be within [0, 10]")
        if not 0.0 <= float(self.direct_rnn_rl_entropy_weight) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_entropy_weight must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_exploration_rate) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_exploration_rate must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_exploration_min_action_probability) <= 1.0:
            raise ValueError(
                "learning.direct_rnn_rl_exploration_min_action_probability must be within [0, 1]"
            )
        if not 1.0 <= float(self.direct_rnn_rl_exploration_importance_clip) <= 20.0:
            raise ValueError(
                "learning.direct_rnn_rl_exploration_importance_clip must be within [1, 20]"
            )
        if int(self.direct_rnn_rl_exploration_seed) < 0:
            raise ValueError("learning.direct_rnn_rl_exploration_seed must be non-negative")
        if not 0.01 <= float(self.direct_rnn_rl_delayed_importance_clip) <= 20.0:
            raise ValueError("learning.direct_rnn_rl_delayed_importance_clip must be within [0.01, 20]")
        if not 1e-4 <= float(self.direct_rnn_rl_delayed_policy_kl_temperature) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_delayed_policy_kl_temperature must be within [1e-4, 10]")
        if not 0.0 <= float(self.direct_rnn_rl_delayed_max_policy_js) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_delayed_max_policy_js must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_delayed_min_effective_weight) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_delayed_min_effective_weight must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_delayed_lifecycle_encoder_credit_scale) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_delayed_lifecycle_encoder_credit_scale must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_delayed_counterfactual_encoder_credit_scale) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_delayed_counterfactual_encoder_credit_scale must be within [0, 1]")
        if str(self.direct_rnn_rl_delayed_preference_target_mode).strip().lower() not in {
            "on_policy_v2", "counterfactual_consensus_v3"
        }:
            raise ValueError(
                "learning.direct_rnn_rl_delayed_preference_target_mode must be "
                "on_policy_v2 or counterfactual_consensus_v3"
            )
        if not 2 <= int(self.direct_rnn_state_memory_capacity) <= 256:
            raise ValueError("learning.direct_rnn_state_memory_capacity must be within [2, 256]")
        if not 4 <= int(self.direct_rnn_state_memory_size) <= 256:
            raise ValueError("learning.direct_rnn_state_memory_size must be within [4, 256]")
        if not 1 <= int(self.direct_rnn_state_memory_min_history) <= int(self.direct_rnn_state_memory_capacity):
            raise ValueError("learning.direct_rnn_state_memory_min_history must be within [1, capacity]")
        if not 0.05 <= float(self.direct_rnn_state_memory_temperature) <= 10.0:
            raise ValueError("learning.direct_rnn_state_memory_temperature must be within [0.05, 10]")
        if not 0.0 <= float(self.direct_rnn_state_memory_injection_scale) <= 2.0:
            raise ValueError("learning.direct_rnn_state_memory_injection_scale must be within [0, 2]")
        if not 0.0 <= float(self.direct_rnn_state_memory_path_delta_scale) <= 4.0:
            raise ValueError("learning.direct_rnn_state_memory_path_delta_scale must be within [0, 4]")
        if not 0.0 <= float(self.direct_rnn_rl_entry_preference_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_entry_preference_loss_weight must be within [0, 10]")
        if not 0.05 <= float(self.direct_rnn_rl_entry_preference_temperature) <= 5.0:
            raise ValueError("learning.direct_rnn_rl_entry_preference_temperature must be within [0.05, 5]")
        if not 0.0 <= float(self.direct_rnn_rl_rlhf_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_rlhf_loss_weight must be within [0, 10]")
        if not 0.05 <= float(self.direct_rnn_rl_rlhf_temperature) <= 5.0:
            raise ValueError("learning.direct_rnn_rl_rlhf_temperature must be within [0.05, 5]")
        if not 0.0 <= float(self.direct_rnn_rl_rlhf_min_confidence) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_rlhf_min_confidence must be within [0, 1]")
        if not 1 <= int(self.direct_rnn_rl_entry_preference_horizon_bars) <= 64:
            raise ValueError("learning.direct_rnn_rl_entry_preference_horizon_bars must be within [1, 64]")
        if not int(self.direct_rnn_rl_entry_preference_horizon_bars) <= int(self.direct_rnn_rl_entry_abstraction_horizon_bars) <= 600:
            raise ValueError(
                "learning.direct_rnn_rl_entry_abstraction_horizon_bars must be within [entry path horizon, 600]"
            )
        if str(self.direct_rnn_rl_entry_utility_mode).strip().lower() not in {"terminal_r_v1", "path_quality_v1"}:
            raise ValueError("learning.direct_rnn_rl_entry_utility_mode must be terminal_r_v1 or path_quality_v1")
        raw_timing_delays = self.direct_rnn_rl_timing_delay_bars
        if isinstance(raw_timing_delays, str):
            timing_delay_values = tuple(
                value.strip() for value in raw_timing_delays.split(",") if value.strip()
            )
        elif isinstance(raw_timing_delays, (tuple, list)):
            timing_delay_values = tuple(raw_timing_delays)
        else:
            timing_delay_values = (raw_timing_delays,)
        timing_delays = tuple(int(value) for value in timing_delay_values)
        if not timing_delays or len(timing_delays) > 12 or any(value < 1 or value > 47 for value in timing_delays):
            raise ValueError("learning.direct_rnn_rl_timing_delay_bars must contain 1..12 values within [1, 47]")
        if tuple(sorted(set(timing_delays))) != timing_delays:
            raise ValueError("learning.direct_rnn_rl_timing_delay_bars must be unique and increasing")
        object.__setattr__(self, "direct_rnn_rl_timing_delay_bars", timing_delays)
        if not 0.0 <= float(self.direct_rnn_rl_timing_gate_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_timing_gate_loss_weight must be within [0, 10]")
        if not -1.0 <= float(self.direct_rnn_rl_timing_margin_r) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_timing_margin_r must be within [-1, 3]")
        if not 0.05 <= float(self.direct_rnn_rl_timing_temperature_r) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_timing_temperature_r must be within [0.05, 3]")
        if not 1 <= int(self.direct_rnn_rl_timing_consensus_actions) <= 6:
            raise ValueError("learning.direct_rnn_rl_timing_consensus_actions must be within [1, 6]")
        from .curriculum_learning import normalize_curriculum_stage
        object.__setattr__(self, "curriculum_stage", normalize_curriculum_stage(self.curriculum_stage))
        if not 0.0 <= float(self.curriculum_lifecycle_actor_scale) <= 1.0:
            raise ValueError("learning.curriculum_lifecycle_actor_scale must be within [0, 1]")
        if str(self.curriculum_tensor_rule_authority).strip().lower() not in {
            "telemetry", "shadow", "residual", "primary"
        }:
            raise ValueError("learning.curriculum_tensor_rule_authority is invalid")
        if not 0.05 <= float(self.curriculum_timing_probability_threshold) <= 0.95:
            raise ValueError("learning.curriculum_timing_probability_threshold must be within [0.05, .95]")
        if not -1.0 <= float(self.curriculum_timing_advantage_lcb_r) <= 3.0:
            raise ValueError("learning.curriculum_timing_advantage_lcb_r must be within [-1, 3]")
        if not 0.0 <= float(self.curriculum_timing_uncertainty_z) <= 3.0:
            raise ValueError("learning.curriculum_timing_uncertainty_z must be within [0, 3]")
        for name in ("curriculum_timing_calibration_weight", "curriculum_timing_consistency_weight"):
            if not 0.0 <= float(getattr(self, name)) <= 10.0:
                raise ValueError(f"learning.{name} must be within [0, 10]")
        if not 0.0 <= float(self.curriculum_timing_consistency_feature_distance) <= 2.0:
            raise ValueError("learning.curriculum_timing_consistency_feature_distance must be within [0, 2]")
        if not 0.10 <= float(self.curriculum_reconstruction_mask_fraction) <= 0.20:
            raise ValueError("learning.curriculum_reconstruction_mask_fraction must be within [0.10, 0.20]")
        if not -3.0 <= float(self.direct_rnn_rl_trade_hold_margin_r) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_trade_hold_margin_r must be within [-3, 3]")
        if not 0.05 <= float(self.direct_rnn_rl_trade_hold_temperature) <= 5.0:
            raise ValueError("learning.direct_rnn_rl_trade_hold_temperature must be within [0.05, 5]")
        if not 0.05 <= float(self.direct_rnn_rl_trade_threshold) <= 0.95:
            raise ValueError("learning.direct_rnn_rl_trade_threshold must be within [0.05, .95]")
        if not 1 <= int(self.direct_rnn_rl_retention_recent_capacity) <= 4096:
            raise ValueError("learning.direct_rnn_rl_retention_recent_capacity must be within [1, 4096]")
        if not 1 <= int(self.direct_rnn_rl_retention_diverse_capacity) <= 8192:
            raise ValueError("learning.direct_rnn_rl_retention_diverse_capacity must be within [1, 8192]")
        if not 1 <= int(self.direct_rnn_rl_retention_anchor_capacity) <= 4096:
            raise ValueError("learning.direct_rnn_rl_retention_anchor_capacity must be within [1, 4096]")
        if not 0 <= int(self.direct_rnn_rl_retention_replay_items_per_batch) <= 64:
            raise ValueError("learning.direct_rnn_rl_retention_replay_items_per_batch must be within [0, 64]")
        if not 0.0 <= float(self.direct_rnn_rl_retention_replay_weight) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_retention_replay_weight must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_retention_distillation_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_retention_distillation_weight must be within [0, 10]")
        if not 0.0 <= float(self.direct_rnn_rl_retention_anchor_min_r) <= 5.0:
            raise ValueError("learning.direct_rnn_rl_retention_anchor_min_r must be within [0, 5]")
        if not 1 <= int(self.direct_rnn_rl_retention_audit_min_anchors) <= 64:
            raise ValueError("learning.direct_rnn_rl_retention_audit_min_anchors must be within [1, 64]")
        if not 0.0 <= float(self.direct_rnn_rl_retention_max_anchor_kl_increase) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_retention_max_anchor_kl_increase must be within [0, 10]")
        if not 1e-6 <= float(self.direct_rnn_rl_retention_max_update_norm) <= 100.0:
            raise ValueError("learning.direct_rnn_rl_retention_max_update_norm must be within [1e-6, 100]")
        if not 0.0 <= float(self.direct_rnn_rl_auxiliary_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_auxiliary_loss_weight must be within [0, 10]")
        if not 0.0 <= float(self.direct_rnn_rl_reconstruction_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_reconstruction_loss_weight must be within [0, 10]")
        if str(self.tensor_rule_authority).strip().lower() not in {"telemetry", "shadow", "residual", "primary"}:
            raise ValueError("learning.tensor_rule_authority must be telemetry, shadow, residual or primary")
        if int(self.tensor_rule_sequence_length) != 32 or int(self.tensor_rule_input_size) != 78:
            raise ValueError("learning tensor-rule input contract must remain 32x78")
        if not 16 <= int(self.tensor_rule_hidden_size) <= 256:
            raise ValueError("learning.tensor_rule_hidden_size must be within [16, 256]")
        if not 1 <= int(self.tensor_rule_encoder_layers) <= 8:
            raise ValueError("learning.tensor_rule_encoder_layers must be within [1, 8]")
        if not 1 <= int(self.tensor_rule_attention_heads) <= 16:
            raise ValueError("learning.tensor_rule_attention_heads must be within [1, 16]")
        if int(self.tensor_rule_hidden_size) % int(self.tensor_rule_attention_heads) != 0:
            raise ValueError("learning.tensor_rule_hidden_size must be divisible by attention_heads")
        if int(self.tensor_rule_ffn_size) < int(self.tensor_rule_hidden_size):
            raise ValueError("learning.tensor_rule_ffn_size must be >= hidden_size")
        if not 2 <= int(self.tensor_rule_max_rules) <= 64:
            raise ValueError("learning.tensor_rule_max_rules must be within [2, 64]")
        if not 1 <= int(self.tensor_rule_initial_rules) <= int(self.tensor_rule_max_rules):
            raise ValueError("learning.tensor_rule_initial_rules must be within [1, max_rules]")
        if not 1 <= int(self.tensor_rule_min_valid_length) <= int(self.tensor_rule_sequence_length):
            raise ValueError("learning.tensor_rule_min_valid_length must be within sequence length")
        if not 0.0 <= float(self.tensor_rule_unknown_membership) <= 1.0:
            raise ValueError("learning.tensor_rule_unknown_membership must be within [0, 1]")
        if not 0.02 <= float(self.tensor_rule_membership_temperature) <= 2.0:
            raise ValueError("learning.tensor_rule_membership_temperature must be within [0.02, 2]")
        if not 0.0 <= float(self.tensor_rule_motif_distance_weight) <= 1.0:
            raise ValueError("learning.tensor_rule_motif_distance_weight must be within [0, 1]")
        if not 0.02 <= float(self.tensor_rule_utility_assignment_temperature) <= 2.0:
            raise ValueError("learning.tensor_rule_utility_assignment_temperature must be within [0.02, 2]")
        if not 0.0 <= float(self.tensor_rule_actor_residual_clip) <= 2.0:
            raise ValueError("learning.tensor_rule_actor_residual_clip must be within [0, 2]")
        if not -3.0 <= float(self.tensor_rule_trade_margin_r) <= 3.0:
            raise ValueError("learning.tensor_rule_trade_margin_r must be within [-3, 3]")
        if not 1 <= int(self.tensor_rule_effective_batch_size) <= 1024:
            raise ValueError("learning.tensor_rule_effective_batch_size must be within [1, 1024]")
        if not 0.01 <= float(self.tensor_rule_ready_gradient_norm_budget) <= 10.0:
            raise ValueError("learning.tensor_rule_ready_gradient_norm_budget must be within [0.01, 10]")
        if not 4 <= int(self.tensor_rule_component_size) <= 64:
            raise ValueError("learning.tensor_rule_component_size must be within [4, 64]")
        if not 4 <= int(self.tensor_rule_action_latent_size) <= 128:
            raise ValueError("learning.tensor_rule_action_latent_size must be within [4, 128]")
        if not 0.0 <= float(self.tensor_rule_counterfactual_encoder_credit_scale) <= 1.0:
            raise ValueError("learning.tensor_rule_counterfactual_encoder_credit_scale must be within [0, 1]")
        if not 0.0 <= float(self.tensor_rule_lifecycle_encoder_credit_scale) <= 1.0:
            raise ValueError("learning.tensor_rule_lifecycle_encoder_credit_scale must be within [0, 1]")
        if not -0.5 <= float(self.tensor_rule_utility_contrastive_margin) <= 0.95:
            raise ValueError("learning.tensor_rule_utility_contrastive_margin must be within [-0.5, 0.95]")
        if not 0.0 <= float(self.direct_rnn_rl_conditioned_representation_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_conditioned_representation_loss_weight must be within [0, 10]")
        if not 0.0 <= float(self.direct_rnn_rl_conditioned_representation_min_abs_r) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_conditioned_representation_min_abs_r must be within [0, 3]")
        if not 0.0 < float(self.direct_rnn_rl_conditioned_representation_positive_weight) <= 20.0:
            raise ValueError("learning.direct_rnn_rl_conditioned_representation_positive_weight must be within (0, 20]")
        if str(self.direct_rnn_rl_conditioned_representation_conflict_mode).strip().lower() not in {"off", "pcgrad_v1"}:
            raise ValueError("learning.direct_rnn_rl_conditioned_representation_conflict_mode must be off or pcgrad_v1")
        if not 0.0 <= float(self.direct_rnn_rl_counterfactual_representation_loss_weight) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_counterfactual_representation_loss_weight must be within [0, 10]")
        if not 0.05 <= float(self.direct_rnn_rl_counterfactual_representation_temperature) <= 5.0:
            raise ValueError("learning.direct_rnn_rl_counterfactual_representation_temperature must be within [0.05, 5]")
        if str(self.direct_rnn_rl_advantage_mode).strip().lower() not in {"ema_v1", "critic_asymmetric_v1"}:
            raise ValueError("learning.direct_rnn_rl_advantage_mode must be ema_v1 or critic_asymmetric_v1")
        if not 1 <= int(self.direct_rnn_rl_critic_warmup_lifecycles) <= 4096:
            raise ValueError("learning.direct_rnn_rl_critic_warmup_lifecycles must be within [1, 4096]")
        if not 0.0 <= float(self.direct_rnn_rl_critic_max_weight) <= 1.0:
            raise ValueError("learning.direct_rnn_rl_critic_max_weight must be within [0, 1]")
        if not 0.0 <= float(self.direct_rnn_rl_profit_floor) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_profit_floor must be within [0, 3]")
        if not 0.0 <= float(self.direct_rnn_rl_loss_floor) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_loss_floor must be within [0, 3]")
        if not 0.50 <= float(self.direct_rnn_rl_advantage_norm_decay) < 1.0:
            raise ValueError("learning.direct_rnn_rl_advantage_norm_decay must be within [0.50, 1)")
        if not 1e-4 <= float(self.direct_rnn_rl_advantage_norm_scale_floor) <= 3.0:
            raise ValueError("learning.direct_rnn_rl_advantage_norm_scale_floor must be within [1e-4, 3]")
        if not 0.25 <= float(self.direct_rnn_rl_advantage_norm_clip) <= 10.0:
            raise ValueError("learning.direct_rnn_rl_advantage_norm_clip must be within [0.25, 10]")
        if not 1 <= int(self.direct_rnn_rl_temporal_bptt_steps) <= 32:
            raise ValueError("learning.direct_rnn_rl_temporal_bptt_steps must be within [1, 32]")
        if int(self.execution_trace_observatory_max_bytes) < 0:
            raise ValueError("learning.execution_trace_observatory_max_bytes must be non-negative")
        if not 0.1 <= float(self.stp_input_clip) <= 50.0:
            raise ValueError("learning.stp_input_clip must be within [0.1, 50]")
        if str(self.hidden_normalization_mode).strip().lower() not in {"rms_v1", "layernorm_v1"}:
            raise ValueError("learning.hidden_normalization_mode must be rms_v1 or layernorm_v1; hidden normalization is mandatory")
        if not bool(self.hidden_normalization_enabled):
            raise ValueError("learning.hidden_normalization_enabled must remain true")
        if not 1e-6 <= float(self.hidden_normalization_target_rms) <= 10.0:
            raise ValueError("learning.hidden_normalization_target_rms must be within (0, 10]")
        if not 1e-6 <= float(self.hidden_normalization_clip) <= 20.0:
            raise ValueError("learning.hidden_normalization_clip must be within (0, 20]")
        if not 1e-9 <= float(self.hidden_normalization_eps) <= 1.0:
            raise ValueError("learning.hidden_normalization_eps must be within [1e-9, 1]")
        def _layer_values(value: object) -> tuple[object, ...]:
            if value is None:
                return ()
            if isinstance(value, str):
                text = value.strip()
                return tuple(part.strip() for part in text.split(",") if part.strip())
            try:
                return tuple(value)  # type: ignore[arg-type]
            except TypeError:
                return (value,)

        normalization_layers = _layer_values(self.hidden_normalization_layers)
        if not normalization_layers or len(normalization_layers) > MAX_LAYERWISE_VALUES:
            raise ValueError(
                f"learning.hidden_normalization_layers must contain 1..{MAX_LAYERWISE_VALUES} values"
            )
        normalized_switches: list[bool] = []
        for item in normalization_layers:
            if isinstance(item, str):
                normalized_switches.append(item.strip().lower() in {"1", "true", "yes", "on"})
            else:
                normalized_switches.append(bool(item))
        object.__setattr__(self, "hidden_normalization_layers", tuple(normalized_switches))
        if not all(normalized_switches):
            raise ValueError("learning.hidden_normalization_layers must enable every hidden layer")
        if not bool(self.hidden_dropout_enabled):
            raise ValueError("learning.hidden_dropout_enabled must remain true")
        if not bool(self.hidden_dropout_train_only):
            raise ValueError("learning.hidden_dropout_train_only must remain true for the deterministic Full-RL dropout contract")
        dropout_rates = _layer_values(self.hidden_dropout_rates)
        if not dropout_rates or len(dropout_rates) > MAX_LAYERWISE_VALUES:
            raise ValueError(
                f"learning.hidden_dropout_rates must contain 1..{MAX_LAYERWISE_VALUES} values"
            )
        normalized_rates: list[float] = []
        for item in dropout_rates:
            try:
                rate = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("learning.hidden_dropout_rates must be numeric") from exc
            if not 0.0 < rate <= MAX_DROPOUT_RATE:
                raise ValueError(
                    f"learning.hidden_dropout_rates must be within (0, {MAX_DROPOUT_RATE}]"
                )
            normalized_rates.append(rate)
        object.__setattr__(self, "hidden_dropout_rates", tuple(normalized_rates))
        if int(self.hidden_dropout_seed) < 0:
            raise ValueError("learning.hidden_dropout_seed must be non-negative")
        if str(self.output_normalization_mode).strip().lower() not in {"raw", "rms_v1", "layernorm_v1"}:
            raise ValueError("learning.output_normalization_mode must be raw, rms_v1, or layernorm_v1")
        if not 1e-6 <= float(self.output_normalization_target_rms) <= 10.0:
            raise ValueError("learning.output_normalization_target_rms must be within (0, 10]")
        if not 1e-6 <= float(self.output_normalization_clip) <= 20.0:
            raise ValueError("learning.output_normalization_clip must be within (0, 20]")
        if str(self.execution_parameter_activation_mode).strip().lower() not in {"legacy_tanh_v1", "raw_logits_v2"}:
            raise ValueError(
                "learning.execution_parameter_activation_mode must be legacy_tanh_v1 or raw_logits_v2"
            )
        if not 0.0 <= float(self.timing_network_action_scale) <= 0.50:
            raise ValueError("learning.timing_network_action_scale must be within [0, 0.50]")
        if not 0.0 <= float(self.timing_network_update_scale) <= 1.0:
            raise ValueError("learning.timing_network_update_scale must be within [0, 1]")
        if not 0.0 <= float(self.timing_network_lifecycle_scale) <= 1.0:
            raise ValueError("learning.timing_network_lifecycle_scale must be within [0, 1]")
        if not 1e-6 <= float(self.timing_network_learning_rate) <= 0.02:
            raise ValueError("learning.timing_network_learning_rate must be within [1e-6, 0.02]")
        if int(self.timing_network_hidden_size) < 4 or int(self.timing_network_hidden_size) > 256:
            raise ValueError("learning.timing_network_hidden_size must be within [4, 256]")
        if int(self.timing_network_embedding_size) < 4 or int(self.timing_network_embedding_size) > 128:
            raise ValueError("learning.timing_network_embedding_size must be within [4, 128]")
        if int(self.timing_network_bptt_window) < 1 or int(self.timing_network_bptt_window) > 128:
            raise ValueError("learning.timing_network_bptt_window must be within [1, 128]")
        if int(self.timing_network_sequence_length) < 1 or int(self.timing_network_sequence_length) > 128:
            raise ValueError("learning.timing_network_sequence_length must be within [1, 128]")
        if not 0.5 <= float(self.timing_network_eligibility_decay) <= 0.999:
            raise ValueError("learning.timing_network_eligibility_decay must be within [0.5, 0.999]")
        if not 0.05 <= float(self.timing_network_gradient_clip) <= 10.0:
            raise ValueError("learning.timing_network_gradient_clip must be within [0.05, 10]")
        if not 1e-5 <= float(self.timing_network_deceleration_learning_rate) <= 0.05:
            raise ValueError("learning.timing_network_deceleration_learning_rate must be within [1e-5, 0.05]")
        if not 0.0 <= float(self.timing_network_deceleration_encoder_scale) <= 1.0:
            raise ValueError("learning.timing_network_deceleration_encoder_scale must be within [0, 1]")
        if not 0.0 <= float(self.timing_network_deceleration_l2) <= 0.1:
            raise ValueError("learning.timing_network_deceleration_l2 must be within [0, 0.1]")
        raw_transition_horizons = _layer_values(self.timing_network_regime_transition_horizons)
        if not raw_transition_horizons or len(raw_transition_horizons) > 6:
            raise ValueError("learning.timing_network_regime_transition_horizons must contain 1..6 values")
        try:
            transition_horizons = tuple(sorted({int(value) for value in raw_transition_horizons}))
        except (TypeError, ValueError) as exc:
            raise ValueError("learning.timing_network_regime_transition_horizons must be integer values") from exc
        if not all(1 <= value <= 48 for value in transition_horizons):
            raise ValueError("learning.timing_network_regime_transition_horizons must be within [1, 48]")
        object.__setattr__(self, "timing_network_regime_transition_horizons", transition_horizons)
        if not 1e-5 <= float(self.timing_network_regime_transition_learning_rate) <= 0.05:
            raise ValueError("learning.timing_network_regime_transition_learning_rate must be within [1e-5, .05]")
        if not 0.0 <= float(self.timing_network_regime_transition_update_scale) <= 1.0:
            raise ValueError("learning.timing_network_regime_transition_update_scale must be within [0, 1]")
        if not 0.0 <= float(self.timing_network_regime_transition_encoder_scale) <= 1.0:
            raise ValueError("learning.timing_network_regime_transition_encoder_scale must be within [0, 1]")
        if not 0.0 <= float(self.timing_network_regime_transition_l2) <= 0.1:
            raise ValueError("learning.timing_network_regime_transition_l2 must be within [0, .1]")
        if not 0.0 <= float(self.timing_network_adam_beta1) < 1.0:
            raise ValueError("learning.timing_network_adam_beta1 must be within [0, 1)")
        if not 0.0 <= float(self.timing_network_adam_beta2) < 1.0:
            raise ValueError("learning.timing_network_adam_beta2 must be within [0, 1)")
        if not 1e-12 <= float(self.timing_network_adam_epsilon) <= 1e-3:
            raise ValueError("learning.timing_network_adam_epsilon must be within [1e-12, 1e-3]")
        if not 0.0 <= float(self.timing_network_adamw_weight_decay) <= 0.10:
            raise ValueError("learning.timing_network_adamw_weight_decay must be within [0, 0.10]")
        if not 0.01 <= float(self.timing_network_cosine_min_factor) <= 1.0:
            raise ValueError("learning.timing_network_cosine_min_factor must be within [0.01, 1]")
        if int(self.timing_network_cosine_cycle_steps) < 1 or int(self.timing_network_cosine_cycle_steps) > 4096:
            raise ValueError("learning.timing_network_cosine_cycle_steps must be within [1, 4096]")
        if (bool(self.timing_network_adamw_enabled) or bool(self.timing_network_cosine_annealing_enabled)) and not bool(self.timing_network_swats_enabled):
            raise ValueError("learning AdamW/cosine requires timing_network_swats_enabled so the schedule has a defined Adam phase")
        if not 0.50 <= float(self.timing_network_projection_beta) < 1.0:
            raise ValueError("learning.timing_network_projection_beta must be within [0.50, 1)")
        if not 0.005 <= float(self.timing_network_projection_tolerance) <= 1.0:
            raise ValueError("learning.timing_network_projection_tolerance must be within [0.005, 1]")
        if int(self.timing_network_switch_consecutive) < 1 or int(self.timing_network_min_projection_steps) < 1:
            raise ValueError("learning.timing_network SWATS step counts must be >= 1")
        if not -1.0 <= float(self.timing_network_target_alignment) <= 1.0:
            raise ValueError("learning.timing_network_target_alignment must be within [-1, 1]")
        if not 1e-8 <= float(self.timing_network_sgd_eta_min) <= float(self.timing_network_sgd_eta_max) <= 0.02:
            raise ValueError("learning.timing_network_sgd_eta_min/max must be ordered within [1e-8, 0.02]")
        if not 0.05 <= float(self.timing_network_sgd_eta_init_factor) <= 1.5:
            raise ValueError("learning.timing_network_sgd_eta_init_factor must be within [0.05, 1.5]")
        if float(self.timing_network_pid_kp) < 0.0 or float(self.timing_network_pid_ki) < 0.0 or float(self.timing_network_pid_kd) < 0.0:
            raise ValueError("learning.timing_network PID gains must be non-negative")
        if not 0.05 <= float(self.timing_network_pid_integral_limit) <= 20.0:
            raise ValueError("learning.timing_network_pid_integral_limit must be within [0.05, 20]")
        if not 0.0 <= float(self.timing_network_pid_derivative_ema) <= 0.99:
            raise ValueError("learning.timing_network_pid_derivative_ema must be within [0, 0.99]")
        if not 0.0 <= float(self.timing_network_pid_deadband) <= 1.0:
            raise ValueError("learning.timing_network_pid_deadband must be within [0, 1]")
        if not -3.0 <= float(self.timing_network_pid_r_target) <= 3.0:
            raise ValueError("learning.timing_network_pid_r_target must be within [-3, 3]")
        if not 1e-4 <= float(self.timing_network_pid_r_scale_floor) <= 3.0:
            raise ValueError("learning.timing_network_pid_r_scale_floor must be within [1e-4, 3]")
        if not 0.0 <= float(self.timing_network_pid_alignment_weight) <= 1.0 or not 0.0 <= float(self.timing_network_pid_r_weight) <= 1.0:
            raise ValueError("learning.timing_network PID weights must be within [0, 1]")
        if float(self.timing_network_pid_alignment_weight) + float(self.timing_network_pid_r_weight) <= 0.0:
            raise ValueError("learning.timing_network PID weights cannot both be zero")
        if int(self.timing_network_pid_bad_steps_to_freeze) < 1 or int(self.timing_network_pid_recovery_steps) < 1 or int(self.timing_network_pid_cooldown_steps) < 0:
            raise ValueError("learning.timing_network PID safety counts are invalid")
        for name in ("experience_buffer_max", "rejection_memory_max", "execution_trace_cache_max", "memory_neurons_max", "memory_connections_max", "mistake_episodes_max", "promotion_min_oos_windows", "promotion_min_seeds", "promotion_min_oos_trades"):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"learning.{name} must be >= 1")
        if float(self.promotion_max_oos_drawdown_pct) <= 0.0 or float(self.promotion_max_oos_drawdown_pct) > 100.0:
            raise ValueError("learning.promotion_max_oos_drawdown_pct must be within (0, 100]")
        if not -2.0 <= float(self.promotion_min_policy_expectancy_lcb) <= 2.0:
            raise ValueError("learning.promotion_min_policy_expectancy_lcb must be within [-2, 2]")
        if int(self.promotion_max_negative_context_scopes) < 0:
            raise ValueError("learning.promotion_max_negative_context_scopes must be >= 0")


@dataclass(frozen=True)
class CausalConfig:
    """Evidence contract for SL/TP causal learning.

    The paths are append-only exchange files.  A missing result file simply
    leaves counterfactual proposals as ``needs_replay``; it never validates a
    cause by PnL sign alone.
    """

    enabled: bool = True
    replay_request_path: str = "outputs/causal_replay_requests.jsonl"
    replay_result_path: str = "outputs/causal_replay_results.jsonl"
    max_decisions: int = 4096
    max_reports: int = 4096
    require_verified_replay: bool = True

    def __post_init__(self) -> None:
        if self.max_decisions < 128:
            raise ValueError("causal.max_decisions must be at least 128")
        if self.max_reports < 128:
            raise ValueError("causal.max_reports must be at least 128")
        if not self.replay_request_path.strip() or not self.replay_result_path.strip():
            raise ValueError("causal replay paths must not be empty")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Selectable market runtime; MT5 remains the compatibility default."""

    backend: str = "mt5"
    model1_tape_path: str = ""
    strict_model1_parity: bool = False
    output_wire_decimals: int = 5
    offline_preflight: bool = False

    def __post_init__(self) -> None:
        backend = str(self.backend).strip().lower()
        if backend not in {"mt5", "python_model1"}:
            raise ValueError("environment.backend must be mt5 or python_model1")
        object.__setattr__(self, "backend", backend)
        if not 3 <= int(self.output_wire_decimals) <= 10:
            raise ValueError("environment.output_wire_decimals must be within [3, 10]")
        if backend == "python_model1" and not str(self.model1_tape_path).strip():
            raise ValueError("environment.model1_tape_path is required for python_model1")


@dataclass(frozen=True)
class AppConfig:
    # Runtime contract for context-owner structural learning.  The MT5
    # evaluator has a separate ``learning_enabled`` flag, but the policy
    # configuration also needs to know whether lazy specialist recruitment is
    # allowed; frozen OOS sets this to False so inference cannot mutate the
    # registry.
    mt5_learning_enabled: bool = True
    csv_path: str = r"C:\Users\Admin\Desktop\BTCUSDM1.csv"
    datetime_column: str = "Datetime"
    symbol: SymbolConfig = SymbolConfig()
    features: FeatureConfig = FeatureConfig()
    risk: RiskConfig = RiskConfig()
    orders: OrderConfig = OrderConfig()
    splits: SplitConfig = SplitConfig()
    expectancy: ExpectancyConfig = ExpectancyConfig()
    learning: LearningIntegrityConfig = LearningIntegrityConfig()
    observability: LiveObservabilityConfig = LiveObservabilityConfig()
    causal: CausalConfig = CausalConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    he_chuyen_gia: CauHinhHeChuyenGia = CauHinhHeChuyenGia()
    model: ModelConfig = ModelConfig()
    model_config_path: str = "configs/self_growing.yaml"
    top_k_validation: int = 5
    validation_interval: int = 5
    train_windows: int = 3
    train_window_bars: int = 2500
    supernetwork: SupernetworkConfig = SupernetworkConfig()
    # Persistent event-driven lifecycle runtime.  It is independent from the
    # context-owner registry; callers must still scope credit by trace owner.
    continuous: ContinuousRuntimeConfig = ContinuousRuntimeConfig()

    def __post_init__(self) -> None:
        if int(self.top_k_validation) < 1 or int(self.validation_interval) < 1 or int(self.train_windows) < 1 or int(self.train_window_bars) < 1:
            raise ValueError("model runtime controls must be positive")
        # ``model.hidden_layers`` is the one authoritative depth declaration.
        # A one-value dropout schedule intentionally broadcasts to every hidden
        # block, while a longer schedule is trimmed when depth is reduced.  This
        # prevents a config change from leaving a stale six-layer regularizer in
        # a newly created 3/4-layer Direct-RNN.
        layer_count = int(self.model.hidden_layers)
        normalized_layers = layerwise_bools(
            self.learning.hidden_normalization_layers,
            layer_count,
            default=True,
        )
        if not normalized_layers or not all(normalized_layers):
            raise ValueError(
                "learning.hidden_normalization_layers must enable every configured hidden layer"
            )
        dropout_rates = mandatory_layerwise_rates(
            self.learning.hidden_dropout_rates,
            layer_count,
        )
        object.__setattr__(
            self,
            "learning",
            replace(
                self.learning,
                hidden_normalization_layers=normalized_layers,
                hidden_dropout_rates=dropout_rates,
            ),
        )


def _strict_values(values: dict[str, Any], cls, section: str) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"{section} must be a mapping")
    unknown = sorted(set(values) - set(cls.__dataclass_fields__))
    if unknown:
        raise ValueError(f"Unknown {section} fields: {', '.join(unknown)}")
    return dict(values)


def _section(data: dict[str, Any], key: str) -> dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(dict(merged[key]), value)
        else:
            merged[key] = value
    return merged


def _load_config_mapping(config_path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML mapping and recursively apply its optional ``extends`` base.

    ``extends`` is resolved relative to the file declaring it.  It exists so a
    small, versioned runtime profile can inherit the complete production
    contract without copying hundreds of unrelated fields.  The child mapping
    always wins and the metadata key is removed before strict section parsing.
    """
    resolved = config_path.resolve()
    if resolved in seen:
        chain = " -> ".join(str(path) for path in (*seen, resolved))
        raise ValueError(f"Config extends cycle: {chain}")
    if not resolved.exists():
        return {}
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    parent_value = raw.pop("extends", None)
    if parent_value is None:
        return raw
    if not isinstance(parent_value, str) or not parent_value.strip():
        raise ValueError(f"Config extends must be a non-empty path: {config_path}")
    parent = Path(parent_value.strip())
    if not _path_is_absolute_cross_platform(str(parent)):
        parent = resolved.parent / parent
    if not parent.is_file():
        raise FileNotFoundError(f"Extended config not found: {parent}")
    return _merge(_load_config_mapping(parent, (*seen, resolved)), raw)


def _path_is_absolute_cross_platform(raw_path: str) -> bool:
    """Recognise POSIX, UNC and Windows drive paths regardless of host OS."""
    text = str(raw_path).strip()
    return Path(text).is_absolute() or bool(__import__("re").match(r"^[A-Za-z]:[\\/]", text)) or text.startswith("\\")


def _resolve_data_path(primary: Path, raw_path: str) -> str:
    """Resolve only explicit relative CSV paths against the config file."""
    if not str(raw_path).strip() or _path_is_absolute_cross_platform(str(raw_path)):
        return str(raw_path)
    return str((primary.parent / raw_path).resolve())


def _resolve_aux_config(primary: Path, raw_path: str) -> Path | None:
    raw = Path(raw_path)
    candidate = raw if raw.is_absolute() else primary.parent / raw
    if candidate.exists() and candidate.resolve() != primary.resolve():
        return candidate.resolve()
    return None


def _supernetwork_config(values: dict[str, Any]) -> SupernetworkConfig:
    prepared = {k: v for k, v in values.items() if k in SupernetworkConfig.__dataclass_fields__}
    backend = str(prepared.get("m1_backend", "direct_rnn") or "direct_rnn").strip().lower()
    prepared["m1_backend"] = "m1_graph" if backend == "graph" else backend
    for key in (
        "allowed_cross_module_edge_kinds",
        "allowed_module_to_m1_targets",
        "allowed_m1_to_module_targets",
        "allowed_specialist_roles",
    ):
        value = prepared.get(key)
        if isinstance(value, list):
            prepared[key] = tuple(str(item) for item in value)
        elif isinstance(value, str):
            prepared[key] = tuple(part.strip() for part in value.split(",") if part.strip())
    # V7: zero means no fixed numerical cap. Keep it as zero instead of
    # silently restoring legacy 512/16384/module quotas. `None` becomes zero.
    for key in (
        "max_child_networks", "max_generation", "max_children_per_network",
        "max_cross_module_edges", "max_m1_module_edges",
        "max_cross_module_recurrent_edges", "max_cross_module_lateral_edges",
        "recurrent_loop_limit", "max_graph_neurons", "max_graph_connections",
        "max_hidden_neurons_per_module", "max_hidden_layers_per_module",
        "flexible_connection_fanout",
        "flexible_peer_module_fanout", "module_role_budget",
        "micro_growth_max_neurons_per_trial", "micro_growth_max_modules_per_trial",
    ):
        if key in prepared and prepared[key] is None:
            prepared[key] = 0
    return SupernetworkConfig(**prepared)


def _model_values(values: dict[str, Any]) -> dict[str, Any]:
    meta = {"config_path", "top_k_validation", "validation_interval", "train_windows", "train_window_bars"}
    unknown = sorted(set(values) - set(ModelConfig.__dataclass_fields__) - meta)
    if unknown:
        raise ValueError(f"Unknown model fields: {', '.join(unknown)}")
    return {key: value for key, value in values.items() if key in ModelConfig.__dataclass_fields__}


def load_config(path: str | Path = "configs/default.yaml") -> AppConfig:
    config_path = Path(path)
    raw = _load_config_mapping(config_path)

    model = _section(raw, "model")
    # Only an explicit model.config_path imports an auxiliary config.  A small
    # standalone config must not silently merge the repository default.
    configured_model_path = model.get("config_path")
    model_path = str(configured_model_path or "configs/self_growing.yaml")
    aux_path = _resolve_aux_config(config_path, str(configured_model_path)) if configured_model_path else None
    if aux_path is not None:
        aux = yaml.safe_load(aux_path.read_text(encoding="utf-8")) or {}
        if not isinstance(aux, dict):
            raise ValueError(f"Model config root must be a mapping: {aux_path}")
        raw = _merge(raw, aux)

    data = _section(raw, "data")
    model = _section(raw, "model")
    return AppConfig(
        csv_path=_resolve_data_path(config_path, str(data.get("csv_path", AppConfig.csv_path))),
        datetime_column=str(data.get("datetime_column", AppConfig.datetime_column)),
        symbol=SymbolConfig(**_strict_values(_section(raw, "symbol"), SymbolConfig, "symbol")),
        features=FeatureConfig(**_strict_values(_section(raw, "features"), FeatureConfig, "features")),
        risk=RiskConfig(**_strict_values(_section(raw, "risk"), RiskConfig, "risk")),
        orders=OrderConfig(**_strict_values(_section(raw, "orders"), OrderConfig, "orders")),
        splits=SplitConfig(**_strict_values(_section(raw, "splits"), SplitConfig, "splits")),
        expectancy=ExpectancyConfig(**_strict_values(_section(raw, "expectancy"), ExpectancyConfig, "expectancy")),
        learning=LearningIntegrityConfig(**_strict_values(_section(raw, "learning"), LearningIntegrityConfig, "learning")),
        observability=LiveObservabilityConfig(**_strict_values(_section(raw, "observability"), LiveObservabilityConfig, "observability")),
        causal=CausalConfig(**_strict_values(_section(raw, "causal"), CausalConfig, "causal")),
        environment=EnvironmentConfig(**_strict_values(_section(raw, "environment"), EnvironmentConfig, "environment")),
        he_chuyen_gia=CauHinhHeChuyenGia(**_strict_values(_section(raw, "he_chuyen_gia"), CauHinhHeChuyenGia, "he_chuyen_gia")),
        model=ModelConfig(**{**_model_values(model), **({"max_modules": 0} if _model_values(model).get("max_modules") is None else {})}),
        model_config_path=model_path,
        top_k_validation=int(model.get("top_k_validation", 5)),
        validation_interval=int(model.get("validation_interval", 5)),
        train_windows=int(model.get("train_windows", 3)),
        train_window_bars=int(model.get("train_window_bars", 2500)),
        supernetwork=_supernetwork_config(_section(raw, "supernetwork")),
        continuous=ContinuousRuntimeConfig(**_strict_values(_section(raw, "continuous"), ContinuousRuntimeConfig, "continuous")),
    )
