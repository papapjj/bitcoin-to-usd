"""Causal decision ledger for MT5 position lifecycles.

The ledger deliberately separates facts from inference:

* mechanical facts are supplied by the execution ledger (fill/close/SL/TP);
* immutable decision evidence is frozen at entry time;
* causal explanations are hypotheses with explicit evidence and validation;
* counterfactual proposals are *not* promoted as truth without a replay/path
  evidence source.

It is intentionally dependency-light so it can be used by the policy, the MT5
parser, tests and offline analysis without importing the trainer runtime.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Iterable, Mapping

import numpy as np


CAUSAL_SCHEMA_VERSION = 25


class ExitMechanism(str, Enum):
    HARD_STOP_LOSS = "HARD_STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TRAILING_STOP = "TRAILING_STOP"
    BREAK_EVEN_STOP = "BREAK_EVEN_STOP"
    PARTIAL_TAKE_PROFIT = "PARTIAL_TAKE_PROFIT"
    POLICY_CLOSE = "POLICY_CLOSE"
    TIME_STOP = "TIME_STOP"
    RISK_EMERGENCY_CLOSE = "RISK_EMERGENCY_CLOSE"
    MANUAL_CLOSE = "MANUAL_CLOSE"
    STOP_OUT = "STOP_OUT"
    BROKER_CLOSE = "BROKER_CLOSE"
    MIXED_LIFECYCLE = "MIXED_LIFECYCLE"
    UNKNOWN = "UNKNOWN"


CAUSE_TYPES: tuple[str, ...] = (
    "stop_loss_too_tight",
    "take_profit_too_far",
    "take_profit_too_close",
    "entry_too_early",
    "wrong_direction_or_regime",
    "false_breakout",
    "volatility_shock",
    "spread_or_slippage_damage",
    "management_failure",
    "risk_size_too_large",
    "memory_mismatch",
    "reasoning_override_error",
)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _vector(values: Any, limit: int = 64) -> list[float]:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    return [_finite(value) for value in array[: max(0, int(limit))].tolist()]


def _mapping(values: Any, *, depth: int = 0) -> Any:
    """Make bounded JSON-safe evidence without converting numbers to strings."""
    if depth > 4:
        return "<depth-limited>"
    if values is None or isinstance(values, (str, bool, int)):
        return values
    if isinstance(values, (float, np.floating)):
        return _finite(values)
    if isinstance(values, np.integer):
        return int(values)
    if isinstance(values, np.ndarray):
        return _vector(values, 96)
    if is_dataclass(values):
        return _mapping(asdict(values), depth=depth + 1)
    if isinstance(values, Mapping):
        return {str(key): _mapping(value, depth=depth + 1) for key, value in list(values.items())[:96]}
    if isinstance(values, (list, tuple, set)):
        return [_mapping(value, depth=depth + 1) for value in list(values)[:96]]
    return str(values)


def _first_float(values: Mapping[str, Any] | None, *names: str, default: float = 0.0) -> float:
    if not isinstance(values, Mapping):
        return float(default)
    for name in names:
        if name in values and values[name] not in {None, ""}:
            return _finite(values[name], default)
    return float(default)


def _first_text(values: Mapping[str, Any] | None, *names: str, default: str = "") -> str:
    if not isinstance(values, Mapping):
        return str(default)
    for name in names:
        value = values.get(name)
        if value is not None and str(value).strip():
            return str(value).strip()
    return str(default)


def _bounded_dict(values: Mapping[str, Any] | None, limit: int = 32) -> dict[str, float]:
    if not isinstance(values, Mapping):
        return {}
    return {str(key): _finite(value) for key, value in list(values.items())[: max(0, int(limit))]}


def classify_exit_mechanism(close_reason: str, *, realized_r: float = 0.0, partial_close_count: int = 0) -> ExitMechanism:
    """Map broker/EA reason text to a taxonomy without inferring from PnL.

    Positive PnL never means TP by itself; the broker/EA close reason remains
    authoritative.  Ambiguous records remain ``UNKNOWN`` so learning does not
    treat a manual close as a validated TP/SL example.
    """
    reason = str(close_reason or "").strip().upper().replace("-", "_").replace(" ", "_")
    if partial_close_count > 0 and any(token in reason for token in ("TP", "TAKE_PROFIT", "TRAIL", "STOP")):
        return ExitMechanism.MIXED_LIFECYCLE
    if "STOP_OUT" in reason:
        return ExitMechanism.STOP_OUT
    if "TRAIL" in reason:
        return ExitMechanism.TRAILING_STOP
    if reason in {"BREAKEVEN_SL", "BREAK_EVEN_SL"}:
        return ExitMechanism.BREAK_EVEN_STOP
    if "BREAK" in reason and "EVEN" in reason:
        return ExitMechanism.BREAK_EVEN_STOP
    if reason in {"SL", "STOP_LOSS", "DEAL_REASON_SL", "HARD_STOP_LOSS", "INITIAL_SL"} or reason.endswith("_SL"):
        return ExitMechanism.HARD_STOP_LOSS
    if reason in {"TP", "TAKE_PROFIT", "DEAL_REASON_TP"} or reason.endswith("_TP"):
        return ExitMechanism.TAKE_PROFIT
    if "PARTIAL" in reason and ("TP" in reason or "PROFIT" in reason):
        return ExitMechanism.PARTIAL_TAKE_PROFIT
    if "TIME" in reason or "MAX_HOLD" in reason:
        return ExitMechanism.TIME_STOP
    if any(token in reason for token in ("RISK", "EMERGENCY", "KILL_SWITCH", "DAILY_LOSS")):
        return ExitMechanism.RISK_EMERGENCY_CLOSE
    if any(token in reason for token in ("EXPERT", "POLICY", "ALGO")):
        return ExitMechanism.POLICY_CLOSE
    if "MANUAL" in reason or "CLIENT" in reason:
        return ExitMechanism.MANUAL_CLOSE
    if any(token in reason for token in ("BROKER", "SERVER", "MARKET_CLOSE")):
        return ExitMechanism.BROKER_CLOSE
    # Do not turn a profit/loss sign into TP/SL. It is only kept as a note
    # inside the outcome evidence.
    _ = realized_r
    return ExitMechanism.UNKNOWN


@dataclass(frozen=True)
class MemoryEvidence:
    memory_id: str
    role: str = ""
    similarity: float = 0.0
    retrieval_weight: float = 0.0
    utility: float = 0.0
    cause: str = ""
    action_index: int = -1
    action_bias: list[float] = field(default_factory=list)
    contribution: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _mapping(asdict(self))


@dataclass(frozen=True)
class ReasoningEffect:
    controller: str
    active_node_count: int = 0
    active_edge_count: int = 0
    output: list[float] = field(default_factory=list)
    score_delta: list[float] = field(default_factory=list)
    score_gain: float = 0.0
    selected_before: int = -1
    selected_after: int = -1
    changed_candidate: bool = False
    raw_input_head: dict[str, float] = field(default_factory=dict)
    edge_contribution_l1: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return _mapping(asdict(self))


@dataclass(frozen=True)
class DecisionSnapshot:
    trace_id: str
    timestamp: float
    policy_revision: int
    context_frame_id: str
    action_name: str
    selected_action_index: int
    selected_hypothesis: int
    confidence: float
    candidate_scores: list[float]
    candidate_actions: list[list[float]]
    selected_action: list[float]
    action_details: dict[str, Any]
    account_context: dict[str, float]
    execution_context: dict[str, Any]
    market_context: dict[str, float]
    canonical_context_key: list[float]
    memory_evidence: tuple[MemoryEvidence, ...]
    reasoning_effects: tuple[ReasoningEffect, ...]
    graph_trace_summary: dict[str, Any]
    allowed_action_mask: tuple[bool, ...]
    # V25 separates the M1 hypothesis selector from the action that was
    # actually emitted by the MT5 bridge.  ``selected_action_index`` remains
    # the compatibility field consumed by legacy causal code, but now denotes
    # the best verified execution-intent action whenever that information is
    # available.  The original M1 hypothesis remains separately auditable.
    policy_action_index: int = -1
    legal_action_index: int = -1
    emitted_action_index: int = -1
    action_attribution_source: str = ""
    model_hypothesis_index: int = -1
    schema_version: int = CAUSAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["memory_evidence"] = [item.to_dict() for item in self.memory_evidence]
        payload["reasoning_effects"] = [item.to_dict() for item in self.reasoning_effects]
        # V10: tuyệt đối không phát hai sentinel legacy -1. Internal fields
        # vẫn tồn tại để đọc artifact cũ, nhưng audit chỉ hiển thị hypothesis,
        # semantic/wire action do contract mới xác nhận.
        policy_details = dict(self.action_details or {})
        semantic = _finite(policy_details.get("semantic_action", policy_details.get("emitted_action", -1)), -1.0)
        emitted = _finite(policy_details.get("emitted_action", -1), -1.0)
        wire = _finite(policy_details.get("wire_action", policy_details.get("emitted_action", -1)), -1.0)
        command_contract = (
            semantic >= 0 and emitted >= 0 and wire >= 0
            and int(semantic) == int(emitted) == int(wire)
        )
        no_command_wire_noop = bool(
            semantic < 0 and emitted < 0 and int(wire) == 0
            and (bool(policy_details.get("wire_noop", False)) or bool(policy_details.get("no_neural_command", False)))
        )
        for key in ("policy_action_index", "legal_action_index", "emitted_action_index", "model_hypothesis_index"):
            payload.pop(key, None)
        payload["selected_hypothesis_index"] = int(self.selected_hypothesis)
        payload["semantic_action_index"] = int(semantic) if command_contract else None
        payload["emitted_action_index"] = int(emitted) if command_contract else None
        payload["wire_action_index"] = int(wire) if (command_contract or no_command_wire_noop) else None
        payload["action_contract_complete"] = bool(command_contract or no_command_wire_noop)
        payload["action_contract_state"] = (
            "COMMAND" if command_contract else
            "NO_COMMAND_WIRE_NOOP" if no_command_wire_noop else
            "INCOMPLETE_OR_MISMATCH"
        )
        return _mapping(payload)


@dataclass(frozen=True)
class TrajectoryPoint:
    timestamp: float = 0.0
    age_bars: float = -1.0
    price: float = 0.0
    unrealized_r: float = 0.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    distance_to_sl_atr: float = 0.0
    distance_to_tp_atr: float = 0.0
    spread_atr_ratio: float = 0.0
    atr: float = 0.0
    volatility_shock: float = 0.0
    regime: str = ""
    management_event: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _mapping(asdict(self))


@dataclass(frozen=True)
class PositionOutcome:
    trace_id: str
    position_ticket: str
    entry_time: float = 0.0
    exit_time: float = 0.0
    entry_price: float = 0.0
    exit_price: float = 0.0
    initial_sl_price: float = 0.0
    final_sl_price: float = 0.0
    initial_tp_price: float = 0.0
    final_tp_price: float = 0.0
    exit_mechanism: str = ExitMechanism.UNKNOWN.value
    close_reason: str = ""
    realized_r: float = 0.0
    net_pnl: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    slippage_r: float = 0.0
    duration_bars: float = -1.0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    giveback_r: float = 0.0
    near_tp_ratio: float = 0.0
    post_exit_mfe_r: float | None = None
    post_exit_mae_r: float | None = None
    returned_to_entry: bool | None = None
    original_tp_hit_after_exit: bool | None = None
    trajectory: tuple[TrajectoryPoint, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CAUSAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["trajectory"] = [item.to_dict() for item in self.trajectory]
        return _mapping(payload)


@dataclass(frozen=True)
class CounterfactualPlan:
    name: str
    family: str
    parameters: dict[str, float | str]
    status: str = "needs_replay"
    result_r: float | None = None
    evidence_source: str = ""

    def to_dict(self) -> dict[str, Any]:
        return _mapping(asdict(self))


@dataclass(frozen=True)
class CauseHypothesis:
    cause_id: str
    confidence: float
    status: str
    evidence_for: tuple[str, ...] = ()
    evidence_against: tuple[str, ...] = ()
    context_scope: dict[str, Any] = field(default_factory=dict)
    counterfactuals: tuple[CounterfactualPlan, ...] = ()
    affected_targets: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["counterfactuals"] = [item.to_dict() for item in self.counterfactuals]
        return _mapping(payload)


@dataclass(frozen=True)
class CreditPlan:
    """Selective outcome attribution. Values are small bounded adjustments."""

    reasoning_controller_credit: dict[str, float] = field(default_factory=dict)
    memory_credit: dict[str, float] = field(default_factory=dict)
    m1_credit: dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return _mapping(asdict(self))


@dataclass(frozen=True)
class CausalOutcomeReport:
    trace_id: str
    snapshot: DecisionSnapshot | None
    outcome: PositionOutcome
    hypotheses: tuple[CauseHypothesis, ...]
    ablation: dict[str, Any]
    credit_plan: CreditPlan
    coverage: dict[str, Any]
    schema_version: int = CAUSAL_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": int(self.schema_version),
            "trace_id": self.trace_id,
            "snapshot": None if self.snapshot is None else self.snapshot.to_dict(),
            "outcome": self.outcome.to_dict(),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "ablation": _mapping(self.ablation),
            "credit_plan": self.credit_plan.to_dict(),
            "coverage": _mapping(self.coverage),
        }


def _timestamp(value: Any) -> float:
    if isinstance(value, datetime):
        return float(value.timestamp())
    return _finite(value)


def _memory_evidence_from_query(memory_query: Mapping[str, Any] | None) -> tuple[MemoryEvidence, ...]:
    if not isinstance(memory_query, Mapping):
        return ()
    items: list[MemoryEvidence] = []
    for field_name in ("replacement_matches", "success_imprint_matches", "matches"):
        raw = memory_query.get(field_name, [])
        if not isinstance(raw, Iterable) or isinstance(raw, (str, bytes, Mapping)):
            continue
        for raw_item in list(raw)[:8]:
            if not isinstance(raw_item, Mapping):
                continue
            memory_id = _first_text(raw_item, "cell_id", "memory_id", "id")
            if not memory_id:
                continue
            weight = _first_float(raw_item, "weight", "retrieval_weight")
            bias = raw_item.get("action_bias", raw_item.get("proposal", []))
            items.append(MemoryEvidence(
                memory_id=memory_id,
                role=_first_text(raw_item, "role", "memory_type"),
                similarity=_first_float(raw_item, "similarity", "context_similarity"),
                retrieval_weight=weight,
                utility=_first_float(raw_item, "utility"),
                cause=_first_text(raw_item, "cause"),
                action_index=int(_first_float(raw_item, "replacement_action_index", "action_index", default=-1.0)),
                action_bias=_vector(bias, 16),
                contribution=weight,
            ))
    # Deduplicate by the most informative retrieval evidence.
    dedup: dict[str, MemoryEvidence] = {}
    for item in items:
        current = dedup.get(item.memory_id)
        if current is None or abs(item.contribution) > abs(current.contribution):
            dedup[item.memory_id] = item
    return tuple(sorted(dedup.values(), key=lambda item: abs(item.contribution), reverse=True)[:12])


def _reasoning_effects_from_trace(trace: Any) -> tuple[ReasoningEffect, ...]:
    graphs = getattr(trace, "graph_traces", {})
    record = dict(graphs.get("reasoning_graph", {}) or {}) if isinstance(graphs, Mapping) else {}
    controllers = dict(record.get("controllers", {}) or {})
    details = dict(getattr(trace, "action_details", {}) or {})
    score_delta = _vector(details.get("reasoning_action_score_delta", []), 16)
    effects: list[ReasoningEffect] = []
    for controller, item in list(controllers.items())[:16]:
        if not isinstance(item, Mapping):
            continue
        contrib = _bounded_dict(item.get("edge_contributions", {}), 512)
        effects.append(ReasoningEffect(
            controller=str(controller),
            active_node_count=len(tuple(item.get("active_neuron_ids", ()) or ())),
            active_edge_count=len(tuple(item.get("active_edge_ids", ()) or ())),
            output=_vector(item.get("output", []), 32),
            score_delta=score_delta if str(controller) in {"utility", "action_head"} else [],
            score_gain=_finite(details.get("reasoning_action_score_gain", 0.0)),
            selected_before=int(_finite(details.get("reasoning_selected_candidate_before", -1), -1.0)),
            selected_after=int(_finite(details.get("reasoning_selected_candidate_after", -1), -1.0)),
            changed_candidate=bool(details.get("reasoning_changed_candidate", False)),
            raw_input_head=_bounded_dict(item.get("raw_inputs", {}), 16),
            edge_contribution_l1=float(sum(abs(value) for value in contrib.values())),
        ))
    return tuple(effects)


def _execution_action_from_details(action_details: Mapping[str, Any] | None, fallback: int = -1) -> tuple[int, str]:
    """Return the actual bridge intent action, never a hypothesis index.

    M1 ``selected_idx`` indexes a hypothesis candidate, not the MT5 action
    vector.  Treating it as action 0/1/2 caused V24 local inhibition to block
    BUY/SELL/HOLD incorrectly.  The bridge writes the fields below only after
    masks/bootstrap have selected the serialized action.
    """
    if isinstance(action_details, Mapping):
        # V27 flat-idle firebreak.  An MT5 flat snapshot/no-op is execution
        # state, not an action.  Even an explicit neural HOLD has no lifecycle
        # attribution while nothing was sent/filled; never let it enter the
        # causal action channel.
        if (
            bool(action_details.get("no_neural_command", False))
            or bool(action_details.get("wire_noop", False))
            or bool(action_details.get("flat_idle_non_feedback", False))
            or str(action_details.get("action_attribution_source", "")).lower() in {"none", "flat_idle", "hold_no_execution"}
        ):
            return -1, "none"
        for key, source in (
            ("filled_action", "filled"),
            ("emitted_action", "emitted"),
            ("bridge_action", "bridge"),
            ("requested_action", "requested"),
            ("legal_action", "legal"),
            ("policy_action", "policy"),
        ):
            value = action_details.get(key)
            try:
                index = int(float(value))
            except (TypeError, ValueError):
                continue
            if 0 <= index < 12:
                return index, source
    return int(fallback), "fallback" if fallback >= 0 else "unknown"


def snapshot_from_trace(policy: Any, trace: Any) -> DecisionSnapshot:
    """Freeze decision-time evidence without querying mutable current state."""
    raw_action_details = dict(getattr(trace, "action_details", {}) or {})
    # ``_mapping`` bounds large telemetry maps to 96 keys.  Contract fields
    # must survive that bound because causal settlement uses the action that
    # actually crossed the MT5 bridge, not the pre-bridge hypothesis.  Keep
    # those fields first while preserving the original order for the rest of
    # the diagnostic payload.
    action_contract_keys = (
        "action_name", "semantic_action", "emitted_action", "wire_action",
        "policy_action", "legal_action", "bridge_action", "wire_noop",
        "no_neural_command", "action_attribution_source", "model_hypothesis_index",
    )
    ordered_action_details = {
        key: raw_action_details[key]
        for key in action_contract_keys
        if key in raw_action_details
    }
    ordered_action_details.update({
        key: value for key, value in raw_action_details.items()
        if key not in ordered_action_details
    })
    action_details = _mapping(ordered_action_details)
    execution_details = _mapping(dict(getattr(trace, "execution_details", {}) or {}))
    context_audit = dict(getattr(trace, "context_audit", {}) or {})
    decision_context = dict(getattr(policy, "_decision_context", {}) or {})
    memory_query = getattr(policy, "_causal_last_memory_query", None)
    graph_traces = dict(getattr(trace, "graph_traces", {}) or {})
    graph_summary: dict[str, Any] = {}
    for graph_id, record in list(graph_traces.items())[:12]:
        if not isinstance(record, Mapping):
            continue
        graph_summary[str(graph_id)] = {
            "active_node_count": int(record.get("active_node_count", len(tuple(record.get("active_neuron_ids", ()) or ()))) or 0),
            "active_edge_count": int(record.get("active_edge_count", len(tuple(record.get("active_edge_ids", ()) or ()))) or 0),
            "active_neuron_ids": [str(item) for item in tuple(record.get("active_neuron_ids", ()) or ())[:128]],
            "active_edge_ids": [str(item) for item in tuple(record.get("active_edge_ids", ()) or ())[:192]],
            "region_activity": _bounded_dict(record.get("region_activity", {}), 24),
        }
    candidate_actions = np.asarray(getattr(trace, "candidate_actions", []), dtype=float)
    if candidate_actions.ndim == 1:
        candidate_actions = candidate_actions.reshape(1, -1)
    model_hypothesis = int(_finite(getattr(trace, "selected_idx", -1), -1.0))
    policy_action, _ = _execution_action_from_details(action_details, fallback=-1)
    legal_action = int(_finite(action_details.get("legal_action", -1), -1.0)) if isinstance(action_details, Mapping) else -1
    emitted_action, attribution_source = _execution_action_from_details(action_details, fallback=policy_action)
    return DecisionSnapshot(
        trace_id=str(getattr(trace, "trace_id", "") or ""),
        timestamp=_finite(getattr(trace, "timestamp", time.time()), time.time()),
        policy_revision=int(_finite(getattr(policy, "revision", 0), 0.0)),
        context_frame_id=str(getattr(trace, "context_frame_id", "") or ""),
        action_name=_first_text(action_details, "action_name", default=""),
        selected_action_index=int(emitted_action),
        selected_hypothesis=int(model_hypothesis),
        confidence=_finite(getattr(trace, "confidence", 0.0)),
        candidate_scores=_vector(getattr(trace, "candidate_scores", []), 32),
        candidate_actions=[_vector(row, 32) for row in candidate_actions[:16]],
        selected_action=_vector(getattr(trace, "selected_action", []), 32),
        action_details=dict(action_details) if isinstance(action_details, Mapping) else {},
        account_context=_bounded_dict(getattr(trace, "account_context", {}), 48),
        execution_context=_mapping(decision_context.get("execution_context", execution_details)),
        market_context={
            "novelty": _finite(getattr(trace, "novelty", 0.0)),
            "conflict": _finite(getattr(trace, "conflict", 0.0)),
            "prediction_error": _finite(getattr(trace, "prediction_error", 0.0)),
            "risk": _finite(getattr(trace, "risk", 0.0)),
            "memory_match_quality": _finite(getattr(trace, "memory_match_quality", 0.0)),
            **_bounded_dict(context_audit, 24),
        },
        canonical_context_key=_vector(getattr(trace, "canonical_context_key", []), 64),
        memory_evidence=_memory_evidence_from_query(memory_query if isinstance(memory_query, Mapping) else {}),
        reasoning_effects=_reasoning_effects_from_trace(trace),
        graph_trace_summary=graph_summary,
        allowed_action_mask=tuple(bool(item) for item in tuple(getattr(trace, "allowed_action_mask", ()) or ())),
        policy_action_index=int(_finite(action_details.get("policy_action", -1), -1.0)) if isinstance(action_details, Mapping) else -1,
        legal_action_index=int(legal_action),
        emitted_action_index=int(emitted_action),
        action_attribution_source=str(attribution_source),
        model_hypothesis_index=int(model_hypothesis),
    )


def outcome_from_lifecycle(lifecycle: Any, *, trace_id: str = "", realized_r: float | None = None,
                           close_reason: str = "", duration_bars: float | None = None) -> PositionOutcome:
    """Adapt an MT5 lifecycle or compatible object to immutable causal facts."""
    metadata = dict(getattr(lifecycle, "metadata", {}) or {}) if lifecycle is not None else {}
    trace = str(getattr(lifecycle, "trace_id", "") or trace_id or "")
    reason = str(getattr(lifecycle, "close_reason", "") or close_reason or "")
    net_r = _finite(getattr(lifecycle, "realized_r", realized_r if realized_r is not None else 0.0), 0.0)
    partial_count = int(_finite(getattr(lifecycle, "partial_close_count", 0), 0.0))
    mechanism = getattr(lifecycle, "exit_mechanism", "") if lifecycle is not None else ""
    if isinstance(mechanism, ExitMechanism):
        mechanism = mechanism.value
    if not mechanism:
        mechanism = classify_exit_mechanism(reason, realized_r=net_r, partial_close_count=partial_count).value
    trajectory_raw = list(getattr(lifecycle, "trajectory", ()) or []) if lifecycle is not None else []
    trajectory: list[TrajectoryPoint] = []
    for point in trajectory_raw[:512]:
        if isinstance(point, TrajectoryPoint):
            trajectory.append(point)
            continue
        if not isinstance(point, Mapping):
            continue
        trajectory.append(TrajectoryPoint(
            timestamp=_timestamp(point.get("timestamp", point.get("time", 0.0))),
            age_bars=_first_float(point, "age_bars", "bar_index", default=-1.0),
            price=_first_float(point, "price", "last_price", "bid", "close"),
            unrealized_r=_first_float(point, "unrealized_r", "floating_r"),
            mfe_r=_first_float(point, "mfe_r"),
            mae_r=_first_float(point, "mae_r"),
            distance_to_sl_atr=_first_float(point, "distance_to_sl_atr", "sl_distance_atr"),
            distance_to_tp_atr=_first_float(point, "distance_to_tp_atr", "tp_distance_atr"),
            spread_atr_ratio=_first_float(point, "spread_atr_ratio"),
            atr=_first_float(point, "atr"),
            volatility_shock=_first_float(point, "volatility_shock", "volatility_shock_score"),
            regime=_first_text(point, "regime", "market_regime"),
            management_event=_first_text(point, "management_event", "event"),
        ))
    entry_time = getattr(lifecycle, "entry_time", None) if lifecycle is not None else None
    exit_time = getattr(lifecycle, "exit_time", None) if lifecycle is not None else None
    mfe = _finite(getattr(lifecycle, "mfe_r", 0.0), 0.0)
    mae = _finite(getattr(lifecycle, "mae_r", 0.0), 0.0)
    if trajectory:
        mfe = max([mfe] + [point.mfe_r for point in trajectory])
        mae = min([mae] + [point.mae_r for point in trajectory])
    giveback = _finite(getattr(lifecycle, "giveback_r", 0.0), 0.0)
    if abs(giveback) <= 1e-12 and mfe > 0.0:
        giveback = max(0.0, mfe - net_r)
    target_r = _first_float(metadata, "initial_tp_r", "target_r", default=0.0)
    near_tp = _finite(getattr(lifecycle, "near_tp_ratio", 0.0), 0.0)
    if near_tp <= 0.0 and target_r > 1e-9:
        near_tp = max(0.0, mfe / target_r)
    return PositionOutcome(
        trace_id=trace,
        position_ticket=str(getattr(lifecycle, "position_ticket", "") or ""),
        entry_time=_timestamp(entry_time),
        exit_time=_timestamp(exit_time),
        entry_price=_finite(getattr(lifecycle, "entry_price", _first_float(metadata, "entry_price"))),
        exit_price=_finite(getattr(lifecycle, "exit_price", _first_float(metadata, "exit_price"))),
        initial_sl_price=_finite(getattr(lifecycle, "initial_sl_price", _first_float(metadata, "initial_sl_price", "initial_sl"))),
        final_sl_price=_finite(getattr(lifecycle, "final_sl_price", _first_float(metadata, "final_sl_price", "final_sl", "sl"))),
        initial_tp_price=_finite(getattr(lifecycle, "initial_tp_price", _first_float(metadata, "initial_tp_price", "initial_tp"))),
        final_tp_price=_finite(getattr(lifecycle, "final_tp_price", _first_float(metadata, "final_tp_price", "final_tp", "tp"))),
        exit_mechanism=str(mechanism),
        close_reason=reason,
        realized_r=net_r,
        net_pnl=_finite(getattr(lifecycle, "net_pnl", 0.0)),
        commission=_finite(getattr(lifecycle, "commission", 0.0)),
        swap=_finite(getattr(lifecycle, "swap", 0.0)),
        slippage_r=_finite(getattr(lifecycle, "slippage_r", _first_float(metadata, "slippage_r"))),
        duration_bars=_finite(getattr(lifecycle, "duration_bars", duration_bars if duration_bars is not None else -1.0), -1.0),
        mfe_r=mfe,
        mae_r=mae,
        giveback_r=giveback,
        near_tp_ratio=near_tp,
        post_exit_mfe_r=(None if getattr(lifecycle, "post_exit_mfe_r", None) is None else _finite(getattr(lifecycle, "post_exit_mfe_r"))),
        post_exit_mae_r=(None if getattr(lifecycle, "post_exit_mae_r", None) is None else _finite(getattr(lifecycle, "post_exit_mae_r"))),
        returned_to_entry=getattr(lifecycle, "returned_to_entry", None),
        original_tp_hit_after_exit=getattr(lifecycle, "original_tp_hit_after_exit", None),
        trajectory=tuple(trajectory),
        metadata=_mapping(metadata),
    )


def _plan(name: str, family: str, **parameters: float | str) -> CounterfactualPlan:
    return CounterfactualPlan(name=name, family=family, parameters=parameters)


def _hypothesis(cause_id: str, confidence: float, *, evidence_for: Iterable[str] = (), evidence_against: Iterable[str] = (),
                scope: Mapping[str, Any] | None = None, plans: Iterable[CounterfactualPlan] = (), targets: Iterable[str] = ()) -> CauseHypothesis:
    confidence = float(np.clip(confidence, 0.0, 0.98))
    status = "validated" if confidence >= 0.80 and any(plan.result_r is not None for plan in plans) else ("evidence_only" if confidence >= 0.45 else "shadow")
    return CauseHypothesis(
        cause_id=cause_id,
        confidence=confidence,
        status=status,
        evidence_for=tuple(str(item) for item in evidence_for),
        evidence_against=tuple(str(item) for item in evidence_against),
        context_scope=_mapping(scope or {}),
        counterfactuals=tuple(plans),
        affected_targets=tuple(str(item) for item in targets),
    )


def infer_hypotheses(snapshot: DecisionSnapshot | None, outcome: PositionOutcome) -> tuple[CauseHypothesis, ...]:
    """Create calibrated explanations. Facts and unverified hypotheses remain separate."""
    hypotheses: list[CauseHypothesis] = []
    scope = {
        "context_frame_id": snapshot.context_frame_id if snapshot else "",
        "action": snapshot.action_name if snapshot else "",
        "selected_hypothesis": snapshot.selected_hypothesis if snapshot else -1,
        "exit_mechanism": outcome.exit_mechanism,
    }
    exit_kind = str(outcome.exit_mechanism)
    is_loss = outcome.realized_r < 0.0
    post_return = bool(outcome.returned_to_entry) or bool(outcome.original_tp_hit_after_exit)
    post_mfe = outcome.post_exit_mfe_r
    sl_plans = (
        _plan("sl_1_25x", "sl_width", multiplier=1.25),
        _plan("sl_1_50x", "sl_width", multiplier=1.50),
        _plan("sl_2_00x", "sl_width", multiplier=2.00),
    )
    if exit_kind == ExitMechanism.HARD_STOP_LOSS.value and is_loss:
        evidence_for = ["broker/EA recorded hard stop loss"]
        score = 0.18
        if post_return:
            score += 0.52
            evidence_for.append("price returned to entry or original TP after stop")
        if post_mfe is not None and post_mfe > 0.75:
            score += min(0.20, 0.10 * float(post_mfe))
            evidence_for.append(f"post-exit favourable excursion={post_mfe:.2f}R")
        if outcome.duration_bars >= 0.0 and outcome.duration_bars <= 3.0:
            score += 0.06
            evidence_for.append("very short holding duration")
        if score >= 0.28:
            hypotheses.append(_hypothesis("stop_loss_too_tight", score, evidence_for=evidence_for,
                                           scope=scope, plans=sl_plans, targets=("action_head", "risk", "memory:sl_parameter")))

        wrong_evidence = ["broker/EA recorded hard stop loss"]
        wrong_score = 0.12
        if post_mfe is not None and post_mfe < 0.15:
            wrong_score += 0.28
            wrong_evidence.append("little favourable recovery after exit")
        if outcome.mae_r <= -1.25:
            wrong_score += 0.14
            wrong_evidence.append(f"adverse excursion={outcome.mae_r:.2f}R")
        shock = max([point.volatility_shock for point in outcome.trajectory], default=0.0)
        if shock >= 0.55:
            wrong_score += 0.05
        if wrong_score >= 0.25:
            hypotheses.append(_hypothesis("wrong_direction_or_regime", wrong_score, evidence_for=wrong_evidence,
                                           evidence_against=("not validated without opposite-direction replay",), scope=scope,
                                           plans=(_plan("opposite_direction", "direction", action="opposite"), _plan("hold", "direction", action="hold")),
                                           targets=("m1:selector", "m1:critic", "reasoning:utility", "memory:action_bias")))

        if outcome.duration_bars >= 0.0 and outcome.duration_bars <= 3.0:
            hypotheses.append(_hypothesis("entry_too_early", 0.32 + (0.10 if post_return else 0.0),
                                           evidence_for=("stop occurred within first three bars",), scope=scope,
                                           plans=(_plan("delay_1_bar", "timing", delay_bars=1.0), _plan("delay_2_bars", "timing", delay_bars=2.0)),
                                           targets=("m1:hypothesis", "m1:selector", "memory:entry_timing")))

    if is_loss and outcome.giveback_r >= 0.80 and outcome.mfe_r >= 0.65:
        hypotheses.append(_hypothesis("management_failure", min(0.86, 0.30 + 0.24 * outcome.giveback_r),
                                       evidence_for=(f"MFE={outcome.mfe_r:.2f}R before final result={outcome.realized_r:.2f}R", f"giveback={outcome.giveback_r:.2f}R"),
                                       scope=scope,
                                       plans=(_plan("partial_at_1r", "management", partial_close_r=1.0, fraction=0.5),
                                              _plan("break_even_after_0_8r", "management", break_even_trigger_r=0.8),
                                              _plan("looser_trailing", "management", trailing="loose")),
                                       targets=("management", "action_head", "risk")))

    if outcome.near_tp_ratio >= 0.80 and is_loss:
        hypotheses.append(_hypothesis("take_profit_too_far", min(0.80, 0.22 + 0.38 * outcome.near_tp_ratio),
                                       evidence_for=(f"near TP ratio={outcome.near_tp_ratio:.2f}", "trade finished non-positive after approaching target"),
                                       scope=scope,
                                       plans=(_plan("tp_0_75x", "tp_distance", multiplier=0.75), _plan("tp_0_85x", "tp_distance", multiplier=0.85)),
                                       targets=("action_head", "management", "memory:tp_parameter")))

    shock = max([point.volatility_shock for point in outcome.trajectory], default=0.0)
    if shock >= 0.55:
        hypotheses.append(_hypothesis("volatility_shock", min(0.85, 0.18 + 0.70 * shock),
                                       evidence_for=(f"max volatility shock={shock:.2f}",), scope=scope,
                                       plans=(_plan("skip_shock", "risk_guard", threshold=0.55), _plan("wider_sl_under_shock", "sl_width", multiplier=1.35)),
                                       targets=("risk", "m1:critic", "memory:regime")))

    spread = max([point.spread_atr_ratio for point in outcome.trajectory], default=0.0)
    if spread >= 0.18 or abs(outcome.slippage_r) >= 0.12:
        hypotheses.append(_hypothesis("spread_or_slippage_damage", min(0.82, 0.15 + 1.40 * max(spread - 0.10, 0.0) + 0.8 * abs(outcome.slippage_r)),
                                       evidence_for=(f"max spread/ATR={spread:.3f}", f"slippage={outcome.slippage_r:.2f}R"), scope=scope,
                                       plans=(_plan("skip_high_spread", "execution_guard", max_spread_atr=0.12), _plan("limit_entry", "execution_guard", order_type="limit")),
                                       targets=("execution_guard", "risk")))

    if snapshot is not None:
        memory_support = sum(max(0.0, item.contribution) for item in snapshot.memory_evidence)
        if is_loss and memory_support >= 0.08:
            hypotheses.append(_hypothesis("memory_mismatch", min(0.65, 0.16 + memory_support),
                                           evidence_for=(f"retrieved memory support={memory_support:.3f}", "outcome contradicted current action"),
                                           evidence_against=("requires repeated similar-context evidence",), scope=scope,
                                           plans=(_plan("memory_off", "ablation", component="memory"),),
                                           targets=tuple(f"memory:{item.memory_id}" for item in snapshot.memory_evidence[:4])))
        reasoning_l1 = sum(item.edge_contribution_l1 for item in snapshot.reasoning_effects)
        changed = any(item.changed_candidate for item in snapshot.reasoning_effects)
        if is_loss and (changed or reasoning_l1 >= 0.15):
            hypotheses.append(_hypothesis("reasoning_override_error", min(0.65, 0.12 + 0.15 * float(changed) + 0.05 * reasoning_l1),
                                           evidence_for=(("reasoning changed candidate" if changed else "reasoning controller had non-trivial active path"),),
                                           evidence_against=("requires ablation/replay before promotion",), scope=scope,
                                           plans=(_plan("reasoning_off", "ablation", component="reasoning"), _plan("action_head_off", "ablation", component="action_head")),
                                           targets=tuple(f"reasoning:{item.controller}" for item in snapshot.reasoning_effects)))

    # A TP is a success mechanically, but its opportunity cost remains a
    # separate proposal rather than a penalty.
    if exit_kind == ExitMechanism.TAKE_PROFIT.value and outcome.post_exit_mfe_r is not None and outcome.post_exit_mfe_r > max(0.75, abs(outcome.realized_r) * 0.75):
        hypotheses.append(_hypothesis("take_profit_too_close", min(0.78, 0.20 + 0.20 * outcome.post_exit_mfe_r),
                                       evidence_for=(f"post-TP extension={outcome.post_exit_mfe_r:.2f}R",), scope=scope,
                                       plans=(_plan("tp_1_25x", "tp_distance", multiplier=1.25), _plan("partial_plus_trail", "management", partial_close_r=1.0)),
                                       targets=("action_head", "management")))

    return tuple(sorted(hypotheses, key=lambda item: item.confidence, reverse=True))


def ablation_from_snapshot(snapshot: DecisionSnapshot | None) -> dict[str, Any]:
    """Decision-time ablation ledger, clearly labelled as attribution not replay.

    Exact re-execution needs a deterministic historical execution feed. This
    lightweight record preserves the deltas necessary to schedule that replay
    and prevents the UI from pretending a static subtraction is a validated
    counterfactual.
    """
    if snapshot is None:
        return {"available": False, "reason": "entry_snapshot_missing"}
    selected = int(snapshot.selected_action_index)
    base_scores = list(snapshot.candidate_scores)
    effects = [item.to_dict() for item in snapshot.reasoning_effects]
    total_reasoning_delta = np.zeros(len(base_scores), dtype=float)
    for effect in snapshot.reasoning_effects:
        values = np.asarray(effect.score_delta, dtype=float).reshape(-1)
        if values.size:
            total_reasoning_delta[: min(total_reasoning_delta.size, values.size)] += values[: min(total_reasoning_delta.size, values.size)]
    reasoning_off = np.asarray(base_scores, dtype=float) - total_reasoning_delta
    memory_total = float(sum(item.contribution for item in snapshot.memory_evidence))
    return {
        "available": True,
        "kind": "decision_time_attribution_requires_market_replay_for_validation",
        "full": {"selected": selected, "scores": base_scores},
        "reasoning_off_estimate": {"scores": _vector(reasoning_off, 32), "selected": int(np.argmax(reasoning_off)) if reasoning_off.size else -1},
        "memory_off_estimate": {"memory_total_contribution": memory_total, "selected": selected},
        "reasoning_effects": effects,
        "validated": False,
    }


def build_credit_plan(snapshot: DecisionSnapshot | None, outcome: PositionOutcome, hypotheses: Iterable[CauseHypothesis]) -> CreditPlan:
    """Convert evidence into small selective credits. No one-off global suppression."""
    hypothesis_list = list(hypotheses)
    best = hypothesis_list[0] if hypothesis_list else None
    outcome_credit = float(np.clip(outcome.realized_r, -2.0, 2.0))
    base = 0.15 * outcome_credit
    reasoning: dict[str, float] = {"utility": base, "risk": -0.10 * max(0.0, -outcome_credit)}
    memory: dict[str, float] = {}
    m1: dict[str, float] = {"selector": base, "critic": base}
    notes: list[str] = ["mechanical outcome is separated from causal hypothesis confidence"]
    confidence = 0.0
    if best is not None:
        confidence = best.confidence
        cause = best.cause_id
        if cause == "stop_loss_too_tight":
            reasoning["action_head"] = -0.35 * best.confidence
            reasoning["risk"] = -0.18 * best.confidence
            m1["selector"] = 0.02 * best.confidence  # Direction remains mostly protected.
            notes.append("penalize stop/risk parameter pathway; do not globally punish direction")
        elif cause == "wrong_direction_or_regime":
            reasoning["utility"] = -0.30 * best.confidence
            m1["selector"] = -0.32 * best.confidence
            m1["critic"] = -0.28 * best.confidence
            notes.append("direction/regime evidence targets selector and critic")
        elif cause == "entry_too_early":
            m1["hypothesis"] = -0.26 * best.confidence
            m1["selector"] = -0.14 * best.confidence
            notes.append("timing evidence does not widen SL by default")
        elif cause == "management_failure":
            reasoning["action_head"] = -0.28 * best.confidence
            reasoning["risk"] = -0.20 * best.confidence
            notes.append("management failure protects entry direction credit")
        elif cause == "take_profit_too_far":
            reasoning["action_head"] = -0.22 * best.confidence
            notes.append("target-distance evidence is parameter-scoped")
        elif cause == "volatility_shock":
            reasoning["risk"] = -0.30 * best.confidence
            m1["critic"] = -0.16 * best.confidence
            notes.append("shock evidence targets risk/critic")
        elif cause == "spread_or_slippage_damage":
            reasoning["risk"] = -0.25 * best.confidence
            notes.append("execution-quality evidence does not relabel market direction")
        elif cause == "memory_mismatch":
            notes.append("memory credit is context-scoped and decays, not immediate deletion")
        elif cause == "reasoning_override_error":
            for effect in snapshot.reasoning_effects if snapshot else ():
                reasoning[effect.controller] = reasoning.get(effect.controller, 0.0) - 0.20 * best.confidence
            notes.append("reasoning path receives shadow penalty until replay validates")
    if snapshot is not None:
        for item in snapshot.memory_evidence:
            # Only causal evidence above a minimal retrieval contribution
            # receives a record. One loss never deletes a memory.
            if abs(item.contribution) > 1e-5:
                multiplier = -0.20 if outcome_credit < 0.0 else 0.15
                memory[item.memory_id] = float(np.clip(multiplier * abs(item.contribution) * max(0.20, confidence or 0.35), -0.20, 0.20))
    return CreditPlan(
        reasoning_controller_credit={key: float(np.clip(value, -1.0, 1.0)) for key, value in reasoning.items()},
        memory_credit=memory,
        m1_credit={key: float(np.clip(value, -1.0, 1.0)) for key, value in m1.items()},
        confidence=confidence,
        notes=tuple(notes),
    )


class CausalLedger:
    """In-memory, serialisable causal ledger attached to a policy candidate."""

    def __init__(self, *, max_decisions: int = 4096, max_reports: int = 4096) -> None:
        self.max_decisions = max(128, int(max_decisions))
        self.max_reports = max(128, int(max_reports))
        self.decisions: dict[str, DecisionSnapshot] = {}
        self.position_to_trace: dict[str, str] = {}
        self.reports: dict[str, CausalOutcomeReport] = {}
        self.report_order: list[str] = []
        self.schema_version = CAUSAL_SCHEMA_VERSION

    def copy(self) -> "CausalLedger":
        return copy.deepcopy(self)

    def register_decision(self, policy: Any, trace: Any) -> DecisionSnapshot:
        snapshot = snapshot_from_trace(policy, trace)
        # V27 only blocks explicitly tagged bridge flat-idle/no-execution
        # traces.  The generic ledger is also used by non-MT5 unit paths and
        # offline tools whose traces do not yet carry bridge action metadata.
        details = dict(getattr(trace, "action_details", {}) or {})
        flat_non_feedback = bool(details.get("flat_idle_non_feedback", False)) or str(
            details.get("action_attribution_source", "")
        ).lower() in {"flat_idle", "hold_no_execution"}
        if not snapshot.trace_id or flat_non_feedback:
            return snapshot
        self.decisions[snapshot.trace_id] = snapshot
        overflow = len(self.decisions) - self.max_decisions
        if overflow > 0:
            for trace_id in list(self.decisions)[:overflow]:
                self.decisions.pop(trace_id, None)
        return snapshot

    def snapshot_for(self, trace_id: str) -> DecisionSnapshot | None:
        return self.decisions.get(str(trace_id))

    def settle(self, *, trace_id: str, lifecycle: Any = None, realized_r: float = 0.0,
               close_reason: str = "", duration_bars: float | None = None) -> CausalOutcomeReport:
        outcome = outcome_from_lifecycle(lifecycle, trace_id=str(trace_id), realized_r=realized_r,
                                         close_reason=close_reason, duration_bars=duration_bars)
        trace = str(outcome.trace_id or trace_id)
        snapshot = self.decisions.get(trace)
        hypotheses = infer_hypotheses(snapshot, outcome)
        ablation = ablation_from_snapshot(snapshot)
        credit_plan = build_credit_plan(snapshot, outcome, hypotheses)
        coverage = {
            "entry_snapshot": snapshot is not None,
            "trace_id": bool(trace),
            "position_ticket": bool(outcome.position_ticket),
            "trajectory_points": len(outcome.trajectory),
            "post_exit_path": bool(outcome.post_exit_mfe_r is not None or outcome.returned_to_entry is not None or outcome.original_tp_hit_after_exit is not None),
            "exit_mechanism_confirmed": outcome.exit_mechanism != ExitMechanism.UNKNOWN.value,
            "counterfactual_replay_validated": any(
                plan.result_r is not None
                for hypothesis in hypotheses
                for plan in hypothesis.counterfactuals
            ),
        }
        report = CausalOutcomeReport(
            trace_id=trace,
            snapshot=snapshot,
            outcome=outcome,
            hypotheses=hypotheses,
            ablation=ablation,
            credit_plan=credit_plan,
            coverage=coverage,
        )
        self.reports[trace] = report
        self.report_order.append(trace)
        if outcome.position_ticket:
            self.position_to_trace[outcome.position_ticket] = trace
        while len(self.report_order) > self.max_reports:
            stale = self.report_order.pop(0)
            self.reports.pop(stale, None)
        return report

    def report_for(self, trace_id: str) -> CausalOutcomeReport | None:
        return self.reports.get(str(trace_id))

    def decision_payload(self, trace_id: str) -> dict[str, Any]:
        item = self.snapshot_for(trace_id)
        return {} if item is None else item.to_dict()

    def summary(self) -> dict[str, Any]:
        validated = 0
        evidence_only = 0
        for report in self.reports.values():
            for hypothesis in report.hypotheses:
                if hypothesis.status == "validated":
                    validated += 1
                elif hypothesis.status == "evidence_only":
                    evidence_only += 1
        return {
            "schema_version": self.schema_version,
            "decision_count": len(self.decisions),
            "outcome_count": len(self.reports),
            "validated_hypotheses": validated,
            "evidence_only_hypotheses": evidence_only,
        }
