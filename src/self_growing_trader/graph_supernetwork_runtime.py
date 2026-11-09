from __future__ import annotations

"""Candidate-only orchestration for graph-backed M1 supernetworks.

This module deliberately has no MT5/EA dependency.  It turns decision traces
into ModuleNeedSignal values, applies structural work to an isolated graph
copy, wires specialist neurons through the graph whitelist, and produces an
audit record which the MT5 training loop can promote or reject as a normal
policy candidate.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Mapping
import copy

import numpy as np

from .cross_module import CrossModuleConnector
from .graph_development import GraphDevelopmentController, GraphStructuralAction, GraphDevelopmentEvent
from .graph_metrics import graph_metrics, module_need_metrics
from .module_need import (
    ModuleErrorTrace,
    ModuleNeedSignal,
    ModuleRolePrior,
    cluster_module_errors,
    module_spec_for_role,
)
from .trial_micro_growth import TrialMicroGrowthController
from .specialist_module_factory import SpecialistModuleFactory
from .neural_graph import (
    NEURON_LIFECYCLE_CANDIDATE,
    REGION_CRITIC,
    REGION_MEMORY,
    REGION_META,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    NeuralGraph,
)


ROLE_REGION: dict[str, str] = {
    "predictor_specialist": REGION_PREDICTOR,
    "memory_specialist": REGION_MEMORY,
    "critic_specialist": REGION_CRITIC,
    "selector_specialist": REGION_SELECTOR,
    "position_management_specialist": REGION_SELECTOR,
    "risk_gate_specialist": REGION_CRITIC,
    "cadence_exploration_specialist": REGION_META,
    "entry_quality_specialist": REGION_SELECTOR,
    "exit_quality_specialist": REGION_SELECTOR,
    "memory_critic_specialist": REGION_CRITIC,
}


@dataclass(frozen=True)
class GraphSupernetworkProposal:
    candidate_id: str
    action: str
    role: str
    reason: str
    module_need_scores: dict[str, float]
    structural_action: GraphStructuralAction
    specialist_module_id: str | None
    specialist_node_ids: tuple[str, ...]
    cross_module_edge_ids: tuple[str, ...]
    specialist_module_summary: dict[str, object] = field(default_factory=dict)
    before_metrics: dict[str, float] = field(default_factory=dict)
    after_metrics: dict[str, float] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "action": self.action,
            "role": self.role,
            "reason": self.reason,
            "module_need_scores": dict(self.module_need_scores),
            "structural_action": {
                "action": self.structural_action.action,
                "target": self.structural_action.target,
                "reason": self.structural_action.reason,
                "details": dict(self.structural_action.details),
            },
            "specialist_module_id": self.specialist_module_id,
            "specialist_node_ids": list(self.specialist_node_ids),
            "cross_module_edge_ids": list(self.cross_module_edge_ids),
            "specialist_module_summary": dict(self.specialist_module_summary),
            "before_metrics": dict(self.before_metrics),
            "after_metrics": dict(self.after_metrics),
            "created_at": self.created_at,
        }


@dataclass
class GraphSupernetworkRuntime:
    """Persistent priors and audit history attached to a graph M1 policy.

    The runtime is intentionally lightweight: it owns no separate mutable
    production graph.  A caller gives it a graph copy and only assigns that
    copy back to the policy after the existing MT5 promotion flow accepts the
    parent policy candidate.
    """

    role_prior: ModuleRolePrior = field(default_factory=ModuleRolePrior)
    controller: GraphDevelopmentController | None = None
    connector: CrossModuleConnector = field(default_factory=CrossModuleConnector)
    role_budget: int | None = None
    need_threshold: float = 0.35
    proposal_history: list[dict[str, object]] = field(default_factory=list)
    max_history: int | None = None
    # Immediate micro-growth runs inside an already-isolated policy trial. It
    # never mutates the baseline; promoted candidates retain their survival
    # evidence while rejected copies disappear with their topology.
    trial_growth: TrialMicroGrowthController = field(default_factory=TrialMicroGrowthController)

    def __post_init__(self) -> None:
        if self.controller is None:
            self.controller = GraphDevelopmentController(role_prior=self.role_prior)
        elif self.controller.role_prior is not self.role_prior:
            # Keep prior feedback and neural controller selection in the same
            # object graph after pickle/copy/migration.
            self.controller.role_prior = self.role_prior
        self.role_budget = None if self.role_budget is None or int(self.role_budget) <= 0 else int(self.role_budget)
        self.max_history = None if self.max_history is None or int(self.max_history) <= 0 else int(self.max_history)
        self.need_threshold = float(np.clip(self.need_threshold, 0.0, 1.0))

    def copy(self) -> "GraphSupernetworkRuntime":
        return copy.deepcopy(self)

    def infer_need(
        self,
        *,
        trace: object | None = None,
        learning_signal: Mapping[str, float] | None = None,
        history: Iterable[object] = (),
        account_context: Mapping[str, float] | None = None,
        starvation: Mapping[str, float] | None = None,
    ) -> tuple[str, ModuleNeedSignal]:
        """Cluster recent trace evidence; never use a hard win-rate rule."""
        traces: list[ModuleErrorTrace] = []
        for item in list(history)[-16:]:
            converted = _as_error_trace(item)
            if converted is not None:
                traces.append(converted)
        converted = _as_error_trace(
            trace,
            learning_signal=learning_signal,
            account_context=account_context,
            starvation=starvation,
        )
        if converted is not None:
            traces.append(converted)
        if not traces:
            return "runtime:general_error", ModuleNeedSignal()
        clusters = cluster_module_errors(traces)
        cluster = clusters[0]
        signal = cluster.signal
        # Starvation must be carried explicitly because it can be maintained
        # outside the trace journal while no fills exist.
        starvation = starvation or {}
        signal.cadence_deficit = max(signal.cadence_deficit, _clamp01(starvation.get("cadence_deficit", 0.0)))
        signal.no_trade_streak = max(signal.no_trade_streak, _clamp01(starvation.get("no_trade_streak", 0.0) / 20.0))
        signal.unfilled_streak = max(signal.unfilled_streak, _clamp01(starvation.get("unfilled_streak", 0.0) / 20.0))
        return cluster.cluster_id, signal

    def propose(
        self,
        graph: NeuralGraph,
        *,
        candidate_id: str,
        trace: object | None = None,
        learning_signal: Mapping[str, float] | None = None,
        history: Iterable[object] = (),
        account_context: Mapping[str, float] | None = None,
        starvation: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> GraphSupernetworkProposal:
        """Mutate only ``graph`` supplied by the candidate caller."""
        rng = rng or np.random.default_rng()
        cluster_id, signal = self.infer_need(
            trace=trace,
            learning_signal=learning_signal,
            history=history,
            account_context=account_context,
            starvation=starvation,
        )
        scores = signal.scores()
        before = graph_metrics(graph)
        assert self.controller is not None
        structural = self.controller.choose_action(signal, cluster_id=cluster_id)
        role = structural.role or self.role_prior.choose_role(scores, cluster_id=cluster_id)
        strongest_need = max(scores.values(), default=0.0)
        if strongest_need < self.need_threshold:
            structural = GraphStructuralAction("hold", role=role, reason="need_below_threshold")

        specialist_module_id: str | None = None
        specialist_nodes: list[str] = []
        cross_edges: list[str] = []
        specialist_summary: dict[str, object] = {}
        before_nodes = len(graph.neurons)
        before_edges = len(graph.edges)
        if structural.action != "hold" and strongest_need >= self.need_threshold and (self.role_budget is None or self._role_count(graph, role) < self.role_budget):
            specialist_module_id = f"specialist:{role}:{candidate_id}"
            try:
                build = SpecialistModuleFactory().create(
                    graph,
                    module_id=specialist_module_id,
                    role=role,
                    reason=cluster_id,
                    candidate_id=candidate_id,
                    trial_id=candidate_id,
                    context_key=cluster_id,
                    context_vector=_trace_context_vector(trace),
                    rng=rng,
                    causal_confidence=float(np.clip(_finite_signal(learning_signal, "causal_confidence"), 0.0, 1.0)),
                    causal_validated=bool((learning_signal or {}).get("causal_validated", False)),
                    causal_pattern_key=str((learning_signal or {}).get("causal_pattern_key", cluster_id) or cluster_id),
                )
                specialist_nodes = list(build.node_ids)
                cross_edges = list(build.interface_edge_ids)
                specialist_summary = build.to_dict()
                event = GraphDevelopmentEvent(
                    action="grow_specialist_module",
                    reason=cluster_id,
                    target=specialist_module_id,
                    before_neurons=before_nodes,
                    after_neurons=len(graph.neurons),
                    before_edges=before_edges,
                    after_edges=len(graph.edges),
                    details={
                        "role": role,
                        "candidate_id": candidate_id,
                        "module_need_scores": {str(key): float(value) for key, value in scores.items()},
                        "structural_action": {
                            "action": structural.action,
                            "reason": structural.reason,
                            "score": float(structural.score),
                            "region": structural.region,
                            "edge_kind": structural.edge_kind,
                            "metadata": dict(structural.metadata or {}),
                        },
                        "created_neurons": list(specialist_nodes),
                        "created_edges": list(cross_edges),
                        **specialist_summary,
                    },
                )
            except (ValueError, KeyError):
                specialist_module_id = None
                event = self.controller.apply_action(graph, GraphStructuralAction("hold", role=role, reason="specialist_factory_rejected"))
        else:
            event = self.controller.apply_action(graph, structural)
        graph.assert_invariants()
        proposal = GraphSupernetworkProposal(
            candidate_id=candidate_id,
            action=event.action,
            role=role,
            reason=cluster_id,
            module_need_scores=scores,
            structural_action=event,
            specialist_module_id=specialist_module_id,
            specialist_node_ids=tuple(specialist_nodes),
            cross_module_edge_ids=tuple(cross_edges),
            specialist_module_summary=dict(specialist_summary),
            before_metrics=before,
            after_metrics=graph_metrics(graph),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self.proposal_history.append(proposal.to_dict())
        if self.max_history is not None and len(self.proposal_history) > self.max_history:
            self.proposal_history = self.proposal_history[-self.max_history:]
        return proposal

    def begin_trial(self, trial_id: str) -> None:
        self.trial_growth.begin_trial(trial_id)

    def learn_in_trial(
        self,
        graph: NeuralGraph,
        *,
        trial_id: str,
        trace: object,
        learning_signal: Mapping[str, float],
        account_context: Mapping[str, float] | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[dict[str, object]]:
        """Grow/fade weak graph neurons immediately after a realised outcome.

        The caller must pass a graph belonging to an already-created trial
        policy. This preserves the baseline/candidate separation while making
        structural learning visible during the trial rather than only between
        MT5 rounds.
        """
        if not self.trial_growth.enabled or not trial_id:
            return []
        cluster_id, signal = self.infer_need(
            trace=trace,
            learning_signal=learning_signal,
            history=(),
            account_context=account_context,
            starvation=None,
        )
        scores = signal.scores()
        role = self.role_prior.choose_role(scores, cluster_id=cluster_id)
        region = ROLE_REGION.get(role, REGION_META)
        events = self.trial_growth.on_outcome(
            graph,
            trial_id=trial_id,
            role=role,
            region=region,
            reason=cluster_id,
            trace=trace,
            learning_signal=learning_signal,
            connector=self.connector,
            rng=rng,
        )
        rows = [event.to_dict() for event in events]
        for row in rows:
            row["module_need_scores"] = dict(scores)
            self.proposal_history.append(dict(row))
        if self.max_history is not None and len(self.proposal_history) > self.max_history:
            self.proposal_history = self.proposal_history[-self.max_history:]
        return rows

    def record_outcome(self, proposal: GraphSupernetworkProposal | Mapping[str, object], *, promoted: bool) -> None:
        role = str(proposal.role if isinstance(proposal, GraphSupernetworkProposal) else proposal.get("role", ""))
        reason = str(proposal.reason if isinstance(proposal, GraphSupernetworkProposal) else proposal.get("reason", ""))
        if role:
            self.role_prior.record_outcome(role, promoted=promoted, cluster_id=reason)
        record = proposal.to_dict() if isinstance(proposal, GraphSupernetworkProposal) else dict(proposal)
        record["promoted"] = bool(promoted)
        record["outcome_at"] = datetime.now(timezone.utc).isoformat()
        self.proposal_history.append(record)
        if self.max_history is not None and len(self.proposal_history) > self.max_history:
            self.proposal_history = self.proposal_history[-self.max_history:]

    def metrics(self, graph: NeuralGraph | None = None) -> dict[str, float]:
        metrics = module_need_metrics_from_history(self.proposal_history)
        metrics.update(self.trial_growth.metrics(graph))
        return metrics

    @staticmethod
    def _role_count(graph: NeuralGraph, role: str) -> int:
        registry = getattr(graph, "specialist_modules", {})
        if isinstance(registry, Mapping):
            return sum(1 for item in registry.values() if isinstance(item, Mapping) and str(item.get("role", "")) == str(role) and str(item.get("status", "")) != "pruning")
        return sum(1 for node in graph.neurons.values() if str(node.module_id or "").startswith(f"specialist:{role}:"))


def module_need_metrics_from_history(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    records = list(rows)
    if not records:
        return {"module_need_proposals": 0.0, "module_role_accept_rate": 0.0}
    promoted = [row for row in records if bool(row.get("promoted", False))]
    scores: list[float] = []
    for row in records:
        values = row.get("module_need_scores", {})
        if isinstance(values, Mapping):
            scores.extend(float(value) for value in values.values() if np.isfinite(float(value)))
    return {
        "module_need_proposals": float(len(records)),
        "module_role_accept_rate": float(len(promoted) / max(1, len(records))),
        "module_need_score_mean": float(np.mean(scores)) if scores else 0.0,
    }


def _finite_signal(values: Mapping[str, float] | None, name: str) -> float:
    try:
        value = float((values or {}).get(name, 0.0))
    except (TypeError, ValueError):
        return 0.0
    return value if np.isfinite(value) else 0.0


def _trace_context_vector(trace: object | None, size: int = 24) -> np.ndarray:
    raw = getattr(trace, "encoded_state", None) if trace is not None else None
    if raw is None and trace is not None:
        raw = getattr(trace, "raw_obs", None)
    try:
        vector = np.asarray(raw, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        vector = np.zeros(0, dtype=float)
    out = np.zeros(max(1, int(size)), dtype=float)
    if vector.size:
        out[: min(out.size, vector.size)] = vector[: min(out.size, vector.size)]
    norm = float(np.linalg.norm(out))
    return out / norm if norm > 1e-12 else out


def _as_error_trace(
    value: object,
    *,
    learning_signal: Mapping[str, float] | None = None,
    account_context: Mapping[str, float] | None = None,
    starvation: Mapping[str, float] | None = None,
) -> ModuleErrorTrace | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        data = value
    else:
        data = getattr(value, "__dict__", None)
        if not isinstance(data, Mapping):
            return None
    learning_signal = learning_signal or {}
    account_context = account_context or data.get("account_context", {}) or {}
    starvation = starvation or data.get("experience_starvation", {}) or {}
    tags = tuple(str(x) for x in data.get("lifecycle_tags", ()) or ())
    return ModuleErrorTrace(
        trace_id=str(data.get("trace_id", "graph-runtime")),
        regime=str(data.get("regime", "runtime")),
        lifecycle_tags=tags,
        prediction_error=_finite(data.get("prediction_error", learning_signal.get("prediction_error", 0.0))),
        novelty=_finite(data.get("novelty", 0.0)),
        conflict=_finite(data.get("conflict", 0.0)),
        memory_match_quality=_clamp01(data.get("memory_match_quality", 1.0)),
        reward=_finite(learning_signal.get("reward", data.get("reward", 0.0))),
        td_delta=_finite(learning_signal.get("td_delta", data.get("td_delta", 0.0))),
        account_stress=_clamp01(account_context.get("current_equity_drawdown", account_context.get("account_stress", 0.0))),
        drawdown_entry_loss=_clamp01(account_context.get("open_risk_ratio", 0.0)),
        cadence_deficit=_clamp01(starvation.get("cadence_deficit", 0.0)),
        expectancy=_finite(learning_signal.get("expectancy", 0.0)),
    )


def _finite(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if np.isfinite(parsed) else 0.0


def _clamp01(value: object) -> float:
    return float(np.clip(_finite(value), 0.0, 1.0))


def write_graph_trial_history(path: str, row: Mapping[str, object]) -> None:
    """Append an auditable JSONL trial record without requiring MT5 imports."""
    from pathlib import Path
    import json

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")


def summarize_graph_trial_history(rows: Iterable[Mapping[str, object]]) -> dict[str, float]:
    """Summarize candidate outcome and graph trace-health metrics."""
    records = list(rows)
    graph_records = [row for row in records if row.get("graph_candidates")]
    proposals = [proposal for row in graph_records for proposal in (row.get("graph_candidates") or []) if isinstance(proposal, Mapping)]
    accepted = 0
    roles: dict[str, list[bool]] = {}
    clusters: dict[str, list[bool]] = {}
    observability: dict[str, list[float]] = {
        "prediction_error_mean": [],
        "prediction_error_trend": [],
        "memory_usefulness": [],
        "selector_switch_rate": [],
        "credit_active_neurons": [],
        "credit_active_edges": [],
    }
    for row in graph_records:
        selected = bool(row.get("promoted", False))
        raw_observability = row.get("graph_observability", {})
        if isinstance(raw_observability, Mapping):
            for key, values in observability.items():
                value = _finite(raw_observability.get(key, 0.0))
                if np.isfinite(value):
                    values.append(value)
        for proposal in row.get("graph_candidates") or []:
            if not isinstance(proposal, Mapping):
                continue
            accepted += int(selected)
            role = str(proposal.get("role", "unknown"))
            reason = str(proposal.get("reason", "unknown"))
            roles.setdefault(role, []).append(selected)
            clusters.setdefault(reason, []).append(selected)
    out: dict[str, float] = {
        "growth_candidates_tested": float(len(proposals)),
        "growth_accept_rate": float(accepted / max(1, len(proposals))),
        "child_candidates": 0.0,
        "child_survival_rate": 0.0,
    }
    for key, values in observability.items():
        out[key] = float(np.mean(values)) if values else 0.0
    for role, values in roles.items():
        out[f"module_role_accept_rate:{role}"] = float(sum(values) / max(1, len(values)))
    for reason, values in clusters.items():
        out[f"cluster_to_module_accept_rate:{reason}"] = float(sum(values) / max(1, len(values)))
    return out
