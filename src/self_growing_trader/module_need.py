from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict


ROLE_PREDICTOR_SPECIALIST = "predictor_specialist"
ROLE_MEMORY_SPECIALIST = "memory_specialist"
ROLE_CRITIC_SPECIALIST = "critic_specialist"
ROLE_SELECTOR_SPECIALIST = "selector_specialist"
ROLE_POSITION_MANAGEMENT_SPECIALIST = "position_management_specialist"
ROLE_RISK_GATE_SPECIALIST = "risk_gate_specialist"
ROLE_CADENCE_EXPLORATION_SPECIALIST = "cadence_exploration_specialist"
ROLE_ENTRY_QUALITY_SPECIALIST = "entry_quality_specialist"
ROLE_EXIT_QUALITY_SPECIALIST = "exit_quality_specialist"
ROLE_MEMORY_CRITIC_SPECIALIST = "memory_critic_specialist"

VALID_MODULE_ROLES = frozenset({
    ROLE_PREDICTOR_SPECIALIST,
    ROLE_MEMORY_SPECIALIST,
    ROLE_CRITIC_SPECIALIST,
    ROLE_SELECTOR_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ROLE_RISK_GATE_SPECIALIST,
    ROLE_CADENCE_EXPLORATION_SPECIALIST,
    ROLE_ENTRY_QUALITY_SPECIALIST,
    ROLE_EXIT_QUALITY_SPECIALIST,
    ROLE_MEMORY_CRITIC_SPECIALIST,
})


@dataclass
class ModuleNeedSignal:
    prediction_error: float = 0.0
    future_state_uncertainty: float = 0.0
    novelty: float = 0.0
    memory_match_quality: float = 1.0
    repeated_unknown_regime: float = 0.0
    conflict: float = 0.0
    selector_uncertainty: float = 0.0
    critic_miscalibration: float = 0.0
    critic_over_penalty: float = 0.0
    risk_stress_loss: float = 0.0
    drawdown_entry_loss: float = 0.0
    exposure_bad_outcome: float = 0.0
    position_giveback: float = 0.0
    near_tp_missed: float = 0.0
    late_close_loss: float = 0.0
    cadence_deficit: float = 0.0
    no_trade_streak: float = 0.0
    unfilled_streak: float = 0.0
    low_expectancy_with_many_trades: float = 0.0
    early_reversal_loss: float = 0.0
    loss_held_too_long: float = 0.0
    memory_used_without_uplift: float = 0.0
    memory_bad_bias: float = 0.0

    def scores(self) -> dict[str, float]:
        low_memory_match = 1.0 - _clamp01(self.memory_match_quality)
        return {
            ROLE_PREDICTOR_SPECIALIST: _mean(
                self.prediction_error,
                self.future_state_uncertainty,
                self.critic_miscalibration,
            ),
            ROLE_MEMORY_SPECIALIST: _mean(
                self.novelty,
                low_memory_match,
                self.repeated_unknown_regime,
                self.prediction_error,
            ),
            ROLE_CRITIC_SPECIALIST: _mean(
                self.critic_miscalibration,
                self.critic_over_penalty,
                self.risk_stress_loss,
            ),
            ROLE_SELECTOR_SPECIALIST: _mean(
                self.conflict,
                self.selector_uncertainty,
                self.critic_miscalibration,
            ),
            ROLE_POSITION_MANAGEMENT_SPECIALIST: _mean(
                self.near_tp_missed,
                self.position_giveback,
                self.late_close_loss,
            ),
            ROLE_RISK_GATE_SPECIALIST: _mean(
                self.risk_stress_loss,
                self.drawdown_entry_loss,
                self.exposure_bad_outcome,
            ),
            ROLE_CADENCE_EXPLORATION_SPECIALIST: _mean(
                self.cadence_deficit,
                self.no_trade_streak,
                self.unfilled_streak,
            ),
            ROLE_ENTRY_QUALITY_SPECIALIST: _mean(
                self.low_expectancy_with_many_trades,
                self.early_reversal_loss,
                self.critic_miscalibration,
            ),
            ROLE_EXIT_QUALITY_SPECIALIST: _mean(
                self.position_giveback,
                self.loss_held_too_long,
                self.late_close_loss,
            ),
            ROLE_MEMORY_CRITIC_SPECIALIST: _mean(
                self.memory_used_without_uplift,
                self.memory_bad_bias,
                low_memory_match,
            ),
        }


@dataclass(frozen=True)
class ModuleErrorTrace:
    trace_id: str
    regime: str = "unknown"
    lifecycle_tags: tuple[str, ...] = ()
    prediction_error: float = 0.0
    novelty: float = 0.0
    conflict: float = 0.0
    memory_match_quality: float = 1.0
    reward: float = 0.0
    td_delta: float = 0.0
    account_stress: float = 0.0
    drawdown_entry_loss: float = 0.0
    cadence_deficit: float = 0.0
    expectancy: float = 0.0


@dataclass(frozen=True)
class ModuleErrorCluster:
    cluster_id: str
    traces: tuple[ModuleErrorTrace, ...]
    signal: ModuleNeedSignal


def cluster_module_errors(traces: list[ModuleErrorTrace], *, min_cluster_size: int = 1) -> list[ModuleErrorCluster]:
    buckets: dict[str, list[ModuleErrorTrace]] = defaultdict(list)
    for trace in traces:
        key = _cluster_key(trace)
        buckets[key].append(trace)
    clusters: list[ModuleErrorCluster] = []
    for key, items in buckets.items():
        if len(items) < min_cluster_size:
            continue
        clusters.append(ModuleErrorCluster(cluster_id=key, traces=tuple(items), signal=_signal_from_traces(items)))
    clusters.sort(key=lambda item: len(item.traces), reverse=True)
    return clusters


def _cluster_key(trace: ModuleErrorTrace) -> str:
    tags = set(trace.lifecycle_tags)
    if "near_tp_missed" in tags or "near_tp_giveback" in tags or "gave_back_profit" in tags:
        return f"{trace.regime}:position_giveback"
    if "stress_entry_loss" in tags or trace.drawdown_entry_loss > 0.5 or trace.account_stress > 0.7:
        return f"{trace.regime}:stress_entry_loss"
    if trace.memory_match_quality < 0.25 or trace.novelty > 0.7:
        return f"{trace.regime}:low_memory_match"
    if trace.prediction_error > 0.7:
        return f"{trace.regime}:high_prediction_error"
    if trace.conflict > 0.7:
        return f"{trace.regime}:hypothesis_conflict"
    if trace.cadence_deficit > 0.7:
        return f"{trace.regime}:cadence_deficit"
    return f"{trace.regime}:general_error"


def _signal_from_traces(traces: list[ModuleErrorTrace]) -> ModuleNeedSignal:
    tags = {tag for trace in traces for tag in trace.lifecycle_tags}
    return ModuleNeedSignal(
        prediction_error=_avg(trace.prediction_error for trace in traces),
        novelty=_avg(trace.novelty for trace in traces),
        conflict=_avg(trace.conflict for trace in traces),
        memory_match_quality=_avg(trace.memory_match_quality for trace in traces),
        critic_miscalibration=_avg(max(0.0, -trace.reward - trace.td_delta) for trace in traces),
        risk_stress_loss=_avg(trace.account_stress for trace in traces if trace.reward < 0.0),
        drawdown_entry_loss=_avg(trace.drawdown_entry_loss for trace in traces),
        position_giveback=1.0 if "gave_back_profit" in tags or "near_tp_giveback" in tags else 0.0,
        near_tp_missed=1.0 if "near_tp_missed" in tags else 0.0,
        late_close_loss=1.0 if "closed_after_near_tp_bad" in tags else 0.0,
        cadence_deficit=_avg(trace.cadence_deficit for trace in traces),
        low_expectancy_with_many_trades=1.0 if _raw_avg(trace.expectancy for trace in traces) < 0.0 and len(traces) >= 3 else 0.0,
    )


@dataclass
class ModuleSpec:
    role: str
    reason: str
    input_sources: list[str] = field(default_factory=list)
    output_targets: list[str] = field(default_factory=list)
    initial_gate_bias: float = -1.5
    initial_plasticity: float = 1.5
    budget: dict[str, int] = field(default_factory=dict)
    validation_goal: dict[str, bool | float | int | str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.role not in VALID_MODULE_ROLES:
            raise ValueError(f"unknown module role: {self.role}")
        if "ea_bridge_schema" in self.output_targets or "hard_risk_limit" in self.output_targets:
            raise ValueError("ModuleSpec cannot target bridge schema or hard risk limits")


class ModuleRolePrior:
    def __init__(self, *, min_prior: float = 0.2, max_prior: float = 3.0) -> None:
        self.min_prior = float(min_prior)
        self.max_prior = float(max_prior)
        self._prior = {role: 1.0 for role in VALID_MODULE_ROLES}
        self._cooldown: dict[tuple[str, str], int] = {}

    def choose_role(self, scores: dict[str, float], *, cluster_id: str | None = None) -> str:
        best_role = None
        best_score = -float("inf")
        for role in VALID_MODULE_ROLES:
            cooldown = self._cooldown.get((cluster_id or "", role), 0)
            penalty = 0.25 * cooldown
            score = float(scores.get(role, 0.0)) * self._prior[role] - penalty
            if score > best_score:
                best_score = score
                best_role = role
        if best_role is None:
            raise ValueError("no module roles available")
        return best_role

    def record_outcome(self, role: str, *, promoted: bool, cluster_id: str | None = None) -> None:
        if role not in VALID_MODULE_ROLES:
            raise ValueError(f"unknown module role: {role}")
        if promoted:
            self._prior[role] = min(self.max_prior, self._prior[role] * 1.15)
            self._cooldown.pop((cluster_id or "", role), None)
        else:
            self._prior[role] = max(self.min_prior, self._prior[role] * 0.85)
            key = (cluster_id or "", role)
            self._cooldown[key] = min(10, self._cooldown.get(key, 0) + 1)

    def prior(self, role: str) -> float:
        return self._prior[role]

    def cooldown(self, role: str, *, cluster_id: str | None = None) -> int:
        return self._cooldown.get((cluster_id or "", role), 0)


def module_spec_for_role(role: str, *, reason: str) -> ModuleSpec:
    if role == ROLE_POSITION_MANAGEMENT_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["position_state", "account_context", "causal_memory", "lifecycle_tags", "critic_state"],
            output_targets=["trailing_gate", "partial_close_ratio", "management_mode", "target_slot", "close_bias"],
            initial_gate_bias=-1.2,
            initial_plasticity=2.0,
            validation_goal={"reduce_giveback": True, "improve_expectancy": True, "max_drawdown_not_worse": True},
        )
    if role == ROLE_RISK_GATE_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["drawdown", "open_risk_ratio", "floating_pnl_r", "spread", "losing_streak", "equity_volatility"],
            output_targets=["entry_gate", "hold_bias", "close_bias", "reduce_risk_bias"],
            initial_gate_bias=-0.8,
            initial_plasticity=1.5,
            validation_goal={"reduce_stress_losses": True, "max_drawdown_not_worse": True},
        )
    if role == ROLE_MEMORY_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["market_key", "account_key", "position_key", "causal_memory_summary"],
            output_targets=["memory_context", "action_bias", "critic_bias"],
            initial_gate_bias=-1.5,
            initial_plasticity=2.2,
            validation_goal={"improve_memory_usefulness": True},
        )
    if role == ROLE_PREDICTOR_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["encoder_state", "working_memory", "chosen_action", "recent_outcomes"],
            output_targets=["predicted_state", "outcome_uncertainty"],
            initial_gate_bias=-1.4,
            initial_plasticity=2.0,
            validation_goal={"reduce_prediction_error": True},
        )
    if role == ROLE_CRITIC_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["predicted_state", "memory_context", "account_stress", "position_lifecycle"],
            output_targets=["action_utility", "risk_score", "cost_score"],
            initial_gate_bias=-1.1,
            initial_plasticity=1.8,
            validation_goal={"improve_selection_quality": True},
        )
    if role == ROLE_SELECTOR_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["hypothesis_activations", "critic_scores", "conflict", "confidence", "child_policy_scores"],
            output_targets=["selection_gate", "fallback_choice"],
            initial_gate_bias=-1.3,
            initial_plasticity=1.8,
            validation_goal={"reduce_bad_selection": True},
        )
    if role == ROLE_CADENCE_EXPLORATION_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["cadence_deficit", "no_trade_streak", "unfilled_streak", "account_context", "risk_gate_state"],
            output_targets=["exploration_bias", "candidate_gate", "risk_aware_entry_bias"],
            initial_gate_bias=-1.8,
            initial_plasticity=1.6,
            validation_goal={"increase_valid_cadence": True, "max_drawdown_not_worse": True},
        )
    if role == ROLE_ENTRY_QUALITY_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["encoder_state", "memory_context", "spread_cost", "trend_features", "range_features"],
            output_targets=["entry_logit_bias", "critic_context"],
            initial_gate_bias=-1.9,
            initial_plasticity=1.7,
            validation_goal={"improve_entry_expectancy": True, "max_drawdown_not_worse": True},
        )
    if role == ROLE_EXIT_QUALITY_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["position_lifecycle", "mfe", "mae", "distance_tp_sl", "bars_since_mfe", "giveback"],
            output_targets=["close_bias", "partial_close_ratio", "trailing_gate", "management_mode", "target_slot"],
            initial_gate_bias=-1.2,
            initial_plasticity=1.9,
            validation_goal={"reduce_giveback": True, "reduce_held_loss": True},
        )
    if role == ROLE_MEMORY_CRITIC_SPECIALIST:
        return ModuleSpec(
            role=role,
            reason=reason,
            input_sources=["memory_matches", "causal_match_utility", "support_count", "outcome_quality"],
            output_targets=["memory_gate", "memory_prune_pressure", "critic_bias"],
            initial_gate_bias=-1.6,
            initial_plasticity=1.8,
            validation_goal={"improve_memory_usefulness": True, "reduce_memory_bad_bias": True},
        )
    return ModuleSpec(role=role, reason=reason, initial_gate_bias=-1.5, initial_plasticity=1.5)


def _mean(*values: float) -> float:
    clipped = [_clamp01(value) for value in values]
    return float(sum(clipped) / max(1, len(clipped)))


def _avg(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return _clamp01(sum(items) / len(items))


def _raw_avg(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))
