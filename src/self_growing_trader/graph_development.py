from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import uuid

import numpy as np

from .graph_plasticity import GraphPlasticityEngine
from .module_need import (
    ROLE_CRITIC_SPECIALIST,
    ROLE_CADENCE_EXPLORATION_SPECIALIST,
    ROLE_ENTRY_QUALITY_SPECIALIST,
    ROLE_EXIT_QUALITY_SPECIALIST,
    ROLE_MEMORY_CRITIC_SPECIALIST,
    ROLE_MEMORY_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ROLE_PREDICTOR_SPECIALIST,
    ROLE_RISK_GATE_SPECIALIST,
    ROLE_SELECTOR_SPECIALIST,
    ModuleNeedSignal,
    ModuleRolePrior,
    VALID_MODULE_ROLES,
    module_spec_for_role,
)
from .neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_MEMORY,
    EDGE_PREDICTIVE,
    EDGE_SELECTOR,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_MEMORY,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    NeuralGraph,
)


ACTION_HOLD = "hold"
ACTION_GROW_NEURON = "grow_neuron"
ACTION_GROW_EDGE = "grow_edge"
ACTION_PRUNE_EDGE = "prune_edge"
ACTION_CONSOLIDATE = "consolidate"
ACTION_REWIRE_EDGE = "rewire_edge"
ACTION_SPLIT_REGION = "split_region"
ACTION_MERGE_REGION = "merge_region"


ROLE_REGION = {
    ROLE_PREDICTOR_SPECIALIST: REGION_PREDICTOR,
    ROLE_MEMORY_SPECIALIST: REGION_MEMORY,
    ROLE_CRITIC_SPECIALIST: REGION_CRITIC,
    ROLE_SELECTOR_SPECIALIST: REGION_SELECTOR,
    ROLE_POSITION_MANAGEMENT_SPECIALIST: REGION_MEMORY,
    ROLE_RISK_GATE_SPECIALIST: REGION_CRITIC,
    ROLE_CADENCE_EXPLORATION_SPECIALIST: REGION_SELECTOR,
    ROLE_ENTRY_QUALITY_SPECIALIST: REGION_ENCODER,
    ROLE_EXIT_QUALITY_SPECIALIST: REGION_SELECTOR,
    ROLE_MEMORY_CRITIC_SPECIALIST: REGION_CRITIC,
}

ROLE_EDGE_KIND = {
    ROLE_PREDICTOR_SPECIALIST: EDGE_PREDICTIVE,
    ROLE_MEMORY_SPECIALIST: EDGE_MEMORY,
    ROLE_CRITIC_SPECIALIST: EDGE_CRITIC,
    ROLE_SELECTOR_SPECIALIST: EDGE_SELECTOR,
    ROLE_POSITION_MANAGEMENT_SPECIALIST: EDGE_GATING,
    ROLE_RISK_GATE_SPECIALIST: EDGE_GATING,
    ROLE_CADENCE_EXPLORATION_SPECIALIST: EDGE_SELECTOR,
    ROLE_ENTRY_QUALITY_SPECIALIST: EDGE_FORWARD,
    ROLE_EXIT_QUALITY_SPECIALIST: EDGE_GATING,
    ROLE_MEMORY_CRITIC_SPECIALIST: EDGE_CRITIC,
}


@dataclass(frozen=True)
class GraphStructuralAction:
    action: str
    role: str | None = None
    region: str | None = None
    edge_kind: str | None = None
    reason: str = ""
    score: float = 0.0
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class GraphDevelopmentEvent:
    action: str
    reason: str
    target: str | None = None
    before_neurons: int = 0
    after_neurons: int = 0
    before_edges: int = 0
    after_edges: int = 0
    # ID/thời điểm giúp nối một event tạo cấu trúc với telemetry/lifecycle sau đó.
    event_id: str = field(default_factory=lambda: f"gde-{uuid.uuid4().hex[:16]}")
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: dict[str, object] = field(default_factory=dict)


class NeuralRoleController:
    """Small bounded role head over need scores; priors remain auditable."""

    def __init__(self, *, rng: np.random.Generator | None = None) -> None:
        self.roles = tuple(sorted(VALID_MODULE_ROLES))
        self.rng = rng or np.random.default_rng(0)
        self.weights = np.eye(len(self.roles), dtype=float)
        self.bias = np.zeros(len(self.roles), dtype=float)

    def logits(self, signal: ModuleNeedSignal) -> dict[str, float]:
        scores = signal.scores()
        vector = np.asarray([float(scores.get(role, 0.0)) for role in self.roles], dtype=float)
        values = np.tanh(self.weights @ vector + self.bias)
        return {role: float(value) for role, value in zip(self.roles, values)}

    def choose(self, signal: ModuleNeedSignal, prior: ModuleRolePrior, *, cluster_id: str | None = None) -> tuple[str, float, dict[str, float]]:
        logits = self.logits(signal)
        adjusted = {
            role: value * prior.prior(role) - 0.25 * prior.cooldown(role, cluster_id=cluster_id)
            for role, value in logits.items()
        }
        role = max(self.roles, key=lambda item: adjusted[item])
        return role, float(adjusted[role]), adjusted


class GraphDevelopmentController:
    def __init__(
        self,
        *,
        role_prior: ModuleRolePrior | None = None,
        grow_threshold: float = 0.35,
        prune_threshold: float = -0.5,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.role_prior = role_prior or ModuleRolePrior()
        self.role_controller = NeuralRoleController(rng=rng)
        self.grow_threshold = float(grow_threshold)
        self.prune_threshold = float(prune_threshold)
        self.rng = rng or np.random.default_rng()
        self.events: list[GraphDevelopmentEvent] = []

    def choose_action(self, signal: ModuleNeedSignal, *, cluster_id: str | None = None) -> GraphStructuralAction:
        role, neural_score, neural_logits = self.role_controller.choose(signal, self.role_prior, cluster_id=cluster_id)
        raw_score = float(signal.scores().get(role, 0.0))
        score = max(raw_score, neural_score)
        if score < self.grow_threshold:
            return GraphStructuralAction(ACTION_HOLD, role=role, reason=cluster_id or role, score=score, metadata={"neural_logits": neural_logits})
        return GraphStructuralAction(
            ACTION_GROW_NEURON,
            role=role,
            region=ROLE_REGION[role],
            edge_kind=ROLE_EDGE_KIND[role],
            reason=cluster_id or role,
            score=score,
            metadata={"neural_logits": neural_logits},
        )

    def apply_action(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        if action.action == ACTION_GROW_NEURON:
            return self._grow_neuron(graph, action)
        if action.action == ACTION_GROW_EDGE:
            return self._grow_edge(graph, action)
        if action.action == ACTION_PRUNE_EDGE:
            return self._prune_low_utility_edge(graph, action)
        if action.action == ACTION_CONSOLIDATE:
            return self._consolidate(graph, action)
        if action.action == ACTION_REWIRE_EDGE:
            return self._rewire_edge(graph, action)
        if action.action == ACTION_SPLIT_REGION:
            return self._split_region(graph, action)
        if action.action == ACTION_MERGE_REGION:
            return self._merge_region(graph, action)
        event = self._event(graph, action.action, action.reason)
        self.events.append(event)
        return event

    def propose_and_apply(
        self,
        graph: NeuralGraph,
        signal: ModuleNeedSignal,
        *,
        cluster_id: str | None = None,
    ) -> GraphDevelopmentEvent:
        return self.apply_action(graph, self.choose_action(signal, cluster_id=cluster_id))

    def _grow_neuron(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        region = action.region or REGION_MEMORY
        role = action.role or "graph_specialist"
        before_neurons = len(graph.neurons)
        before_edges = len(graph.edges)
        spec = module_spec_for_role(role, reason=action.reason)
        neuron_id = graph.add_neuron(
            region,
            neuron_id=f"{role}:{region}:{before_neurons}",
            module_id=role,
            lifecycle="candidate",
            bias=float(self.rng.normal(0.0, 0.01)),
            plasticity=spec.initial_plasticity,
        )
        new_neuron = graph.neurons[neuron_id]
        src = self._best_source(graph, preferred_regions=(REGION_ENCODER, REGION_MEMORY), dst=neuron_id)
        skipped_edges: list[str] = []
        created_edges: list[dict[str, object]] = []
        if src is not None:
            edge_id = self._safe_add_connection(
                graph,
                source=src,
                target=neuron_id,
                weight=float(self.rng.normal(0.0, 0.05)),
                gate=spec.initial_gate_bias,
                kind=action.edge_kind or EDGE_FORWARD,
                reason=action.reason,
                role=role,
            )
            if edge_id is None:
                skipped_edges.append("source_to_candidate")
            else:
                edge = graph.edges[edge_id]
                created_edges.append({"id": edge_id, "src": edge.src, "dst": edge.dst, "kind": edge.kind, "weight": float(edge.weight), "gate": float(edge.gate)})
        dst = self._best_destination(graph, preferred_regions=(REGION_CRITIC, REGION_SELECTOR, REGION_OUTPUT), src=neuron_id)
        if dst is not None and dst != neuron_id:
            edge_id = self._safe_add_connection(
                graph,
                source=neuron_id,
                target=dst,
                weight=float(self.rng.normal(0.0, 0.02)),
                gate=spec.initial_gate_bias,
                kind=action.edge_kind or EDGE_FORWARD,
                reason=action.reason,
                role=role,
            )
            if edge_id is None:
                skipped_edges.append("candidate_to_destination")
            else:
                edge = graph.edges[edge_id]
                created_edges.append({"id": edge_id, "src": edge.src, "dst": edge.dst, "kind": edge.kind, "weight": float(edge.weight), "gate": float(edge.gate)})
        graph.assert_invariants()
        event = GraphDevelopmentEvent(
            action=ACTION_GROW_NEURON,
            reason=action.reason,
            target=neuron_id,
            before_neurons=before_neurons,
            after_neurons=len(graph.neurons),
            before_edges=before_edges,
            after_edges=len(graph.edges),
            details={
                "role": role, "region": region, "edge_kind": action.edge_kind,
                "score": float(action.score), "neural_logits": dict(action.metadata.get("neural_logits", {}) or {}),
                "module_need_scores": dict(action.metadata.get("module_need_scores", {}) or {}),
                "created_neurons": [neuron_id], "created_edges": created_edges,
                "new_neuron": {"id": neuron_id, "region": new_neuron.region, "module_id": new_neuron.module_id, "bias": float(new_neuron.bias), "plasticity": float(new_neuron.plasticity), "lifecycle": new_neuron.lifecycle},
                "source_candidate": src, "destination_candidate": dst,
                "skipped_edges": skipped_edges,
            },
        )
        self.events.append(event)
        return event

    def _grow_edge(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        before_edges = len(graph.edges)
        src = self._best_source(graph)
        dst = self._best_destination(graph, src=src)
        edge_id = None
        created_edge: dict[str, object] = {}
        if src is not None and dst is not None and src != dst:
            edge_id = self._safe_add_connection(
                graph,
                source=src,
                target=dst,
                weight=float(self.rng.normal(0.0, 0.03)),
                gate=-2.0,
                kind=action.edge_kind or EDGE_FORWARD,
                reason=action.reason,
                role=action.role,
            )
            if edge_id is not None:
                edge = graph.edges[edge_id]
                created_edge = {"id": edge_id, "src": edge.src, "dst": edge.dst, "kind": edge.kind, "weight": float(edge.weight), "gate": float(edge.gate), "lifecycle": edge.lifecycle}
        graph.assert_invariants()
        event = self._event(
            graph,
            ACTION_GROW_EDGE,
            action.reason,
            target=edge_id,
            before_edges=before_edges,
            details={
                "role": action.role or "", "region": action.region or "", "edge_kind": action.edge_kind or EDGE_FORWARD,
                "score": float(action.score), "neural_logits": dict(action.metadata.get("neural_logits", {}) or {}),
                "module_need_scores": dict(action.metadata.get("module_need_scores", {}) or {}),
                "created_edges": [created_edge] if created_edge else [],
                "source_candidate": src, "destination_candidate": dst,
                "skipped_edges": [] if edge_id is not None else ["budget_or_whitelist_rejected"],
            },
        )
        self.events.append(event)
        return event

    @staticmethod
    def _safe_add_connection(
        graph: NeuralGraph,
        *,
        source: str,
        target: str,
        weight: float,
        gate: float,
        kind: str,
        reason: str,
        role: str | None,
    ) -> str | None:
        try:
            return graph.add_connection(
                source,
                target,
                weight=weight,
                gate=gate,
                kind=kind,
                created_by="graph_development",
                need_reason=reason,
                candidate_id=role,
            )
        except ValueError:
            return None

    def _prune_low_utility_edge(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        before_edges = len(graph.edges)
        candidates = [edge for edge in graph.edges.values() if edge.enabled and edge.utility <= self.prune_threshold]
        target = min(candidates, key=lambda edge: edge.utility).id if candidates else None
        before_snapshot = {}
        if target is not None:
            edge = graph.edges[target]
            before_snapshot = {"id": edge.id, "src": edge.src, "dst": edge.dst, "kind": edge.kind, "utility": float(edge.utility), "weight": float(edge.weight), "gate": float(edge.gate), "lifecycle": edge.lifecycle}
            graph.disable_connection(target)
        event = self._event(graph, ACTION_PRUNE_EDGE, action.reason, target=target, before_edges=before_edges, details={"pruned_edge_before": before_snapshot, "score": float(action.score), "role": action.role or ""})
        self.events.append(event)
        return event

    def _consolidate(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        before_edges = len(graph.edges)
        count = GraphPlasticityEngine().consolidate_high_utility(graph)
        event = self._event(graph, ACTION_CONSOLIDATE, action.reason, before_edges=before_edges, details={"consolidated_edges": count})
        self.events.append(event)
        return event

    def _rewire_edge(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        before_edges = len(graph.edges)
        candidates = [edge for edge in graph.edges.values() if edge.enabled]
        target = min(candidates, key=lambda edge: edge.utility).id if candidates else None
        before_snapshot = {}
        after_snapshot = {}
        if target is not None:
            edge = graph.edges[target]
            before_snapshot = {"id": edge.id, "src": edge.src, "dst": edge.dst, "kind": edge.kind, "weight": float(edge.weight), "gate": float(edge.gate), "utility": float(edge.utility)}
            dst = self._best_destination(graph, src=edge.src)
            if dst is not None and dst != edge.src:
                graph.rewire_connection(target, dst=dst, reason=action.reason, candidate_id=action.metadata.get("candidate_id") if action.metadata else None)
                edge = graph.edges[target]
                after_snapshot = {"id": edge.id, "src": edge.src, "dst": edge.dst, "kind": edge.kind, "weight": float(edge.weight), "gate": float(edge.gate), "utility": float(edge.utility)}
        event = self._event(graph, ACTION_REWIRE_EDGE, action.reason, target=target, before_edges=before_edges, details={"edge_before": before_snapshot, "edge_after": after_snapshot, "score": float(action.score), "role": action.role or ""})
        self.events.append(event)
        return event

    def _split_region(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        region = action.region or REGION_MEMORY
        source = str(action.metadata.get("source_module_id", "m1"))
        child = str(action.metadata.get("child_module_id", f"{action.role or 'specialist'}_split"))
        moved = graph.split_region(
            region, source_module_id=source, child_module_id=child,
            similarity=float(action.metadata.get("similarity", 0.5)),
            gradient_conflict=float(action.metadata.get("gradient_conflict", 0.5)),
        )
        event = self._event(graph, ACTION_SPLIT_REGION, action.reason, target=child, details={"moved_neurons": moved})
        self.events.append(event)
        return event

    def _merge_region(self, graph: NeuralGraph, action: GraphStructuralAction) -> GraphDevelopmentEvent:
        region = action.region or REGION_MEMORY
        count = graph.merge_region(
            region,
            source_module_id=str(action.metadata.get("source_module_id", "candidate")),
            target_module_id=str(action.metadata.get("target_module_id", "m1")),
            similarity=float(action.metadata.get("similarity", 0.8)),
        )
        event = self._event(graph, ACTION_MERGE_REGION, action.reason, target=str(action.metadata.get("target_module_id", "m1")), details={"merged_neurons": count})
        self.events.append(event)
        return event

    def _event(
        self,
        graph: NeuralGraph,
        action: str,
        reason: str,
        *,
        target: str | None = None,
        before_edges: int | None = None,
        details: dict[str, object] | None = None,
    ) -> GraphDevelopmentEvent:
        return GraphDevelopmentEvent(
            action=action,
            reason=reason,
            target=target,
            before_neurons=len(graph.neurons),
            after_neurons=len(graph.neurons),
            before_edges=len(graph.edges) if before_edges is None else before_edges,
            after_edges=len(graph.edges),
            details={
                "score": float(getattr(action, "score", 0.0)),
                "role": str(getattr(action, "role", "") or ""),
                "region": str(getattr(action, "region", "") or ""),
                "edge_kind": str(getattr(action, "edge_kind", "") or ""),
                "neural_logits": dict(getattr(action, "metadata", {}).get("neural_logits", {}) or {}),
                **({} if details is None else details),
            },
        )

    @staticmethod
    def _endpoint_pair_allowed(graph: NeuralGraph, src_id: str, dst_id: str) -> bool:
        src = graph.neurons[src_id]
        dst = graph.neurons[dst_id]
        if src.module_id == "m1" and dst.module_id != "m1" and dst.region not in graph.allowed_m1_module_targets:
            return False
        if dst.module_id == "m1" and src.module_id != "m1" and dst.region not in graph.allowed_module_m1_targets:
            return False
        return True

    @classmethod
    def _best_source(
        cls,
        graph: NeuralGraph,
        preferred_regions: tuple[str, ...] = (REGION_ENCODER, REGION_MEMORY, REGION_PREDICTOR),
        *,
        dst: str | None = None,
    ) -> str | None:
        def allowed(neuron_id: str) -> bool:
            return dst is None or cls._endpoint_pair_allowed(graph, neuron_id, dst)

        for region in preferred_regions:
            ids = [neuron_id for neuron_id in graph.regions[region].neuron_ids if allowed(neuron_id)]
            if ids:
                return max(ids, key=lambda neuron_id: graph.neurons[neuron_id].utility)
        return next((neuron_id for neuron_id in graph.neurons if allowed(neuron_id)), None)

    @classmethod
    def _best_destination(
        cls,
        graph: NeuralGraph,
        preferred_regions: tuple[str, ...] = (REGION_CRITIC, REGION_SELECTOR, REGION_OUTPUT),
        *,
        src: str | None = None,
    ) -> str | None:
        def allowed(neuron_id: str) -> bool:
            return src is None or cls._endpoint_pair_allowed(graph, src, neuron_id)

        for region in preferred_regions:
            ids = [neuron_id for neuron_id in graph.regions[region].neuron_ids if allowed(neuron_id)]
            if ids:
                return max(ids, key=lambda neuron_id: graph.neurons[neuron_id].utility)
        return next((neuron_id for neuron_id in graph.neurons if allowed(neuron_id)), None)
