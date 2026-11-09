from __future__ import annotations

"""Evidence-scoped specialist *module* growth for graph M1 candidates.

V20 called a single weak candidate neuron a module.  V21 keeps candidate-only
safety but recruits a bounded circuit: input adapters, a recurrent/lateral core
and output adapters.  The whole circuit shares one home context and lifecycle.
A one-off surprise may create a visible candidate; only repeated nearby
outcomes and causal evidence can activate/consolidate it.
"""

from dataclasses import dataclass, field
from typing import Mapping
import zlib

import numpy as np

from .neural_graph import (
    NEURON_LIFECYCLE_ACTIVE,
    NEURON_LIFECYCLE_CANDIDATE,
    NEURON_LIFECYCLE_CONSOLIDATED,
    NEURON_LIFECYCLE_PRUNING,
    NeuralGraph,
)
from .specialist_module_factory import SpecialistModuleFactory


@dataclass(frozen=True)
class TrialMicroGrowthEvent:
    kind: str
    trial_id: str
    neuron_id: str | None
    role: str
    reason: str
    context_key: str
    learning_strength: float
    credit: float
    module_id: str | None = None
    details: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "trial_id": self.trial_id,
            "neuron_id": self.neuron_id,
            "module_id": self.module_id,
            "role": self.role,
            "reason": self.reason,
            "context_key": self.context_key,
            "learning_strength": self.learning_strength,
            "credit": self.credit,
            "details": dict(self.details),
        }


@dataclass
class TrialMicroGrowthController:
    """Recruit, retain and prune *circuits* by local-context recurrence.

    A baseline policy is never modified.  This controller only receives an
    already-isolated candidate graph.  ``survival_contexts`` remains a
    compatibility alias but now counts nearby outcome support for one module.
    """

    enabled: bool = True
    max_new_neurons_per_trial: int | None = None
    max_new_modules_per_trial: int | None = 2
    learning_threshold: float = 0.35
    # Kept for config/pickle compatibility. Module factory uses the less
    # suppressed gate below so a newborn circuit can be observed and credited.
    initial_gate: float = -4.0
    initial_weight: float = 0.025
    module_initial_gate: float = -2.15
    module_initial_weight: float = 0.055
    module_input_adapters: int = 3
    module_core_neurons: int = 4
    module_output_adapters: int = 2
    module_enable_bypass_interfaces: bool = True
    survival_contexts: int = 3
    survival_similar_outcomes: int | None = None
    survival_positive_supports: int = 3
    survival_utility: float = 0.03
    consolidate_utility: float = 0.20
    context_similarity_threshold: float = 0.82
    context_prototype_lr: float = 0.08
    fade_rate: float = 0.92
    gate_fade: float = 0.12
    prune_fade_threshold: float = 0.22
    grace_outcomes: int = 5
    # An exploratory module may become active through repeated evidence; it
    # cannot consolidate into a durable specialist until a V19 replay/causal
    # report validates the cause or carries strong calibrated confidence.
    min_causal_confidence_to_activate: float = 0.55
    require_causal_validation_to_consolidate: bool = True
    current_trial_id: str = ""
    outcome_step: int = 0
    grown_this_trial: int = 0
    events: list[dict[str, object]] = field(default_factory=list)
    max_events: int | None = None

    def __post_init__(self) -> None:
        self.max_new_neurons_per_trial = None if self.max_new_neurons_per_trial is None or int(self.max_new_neurons_per_trial) <= 0 else int(self.max_new_neurons_per_trial)
        # Old config used max_new_neurons_per_trial. Map it to a conservative
        # module budget when an explicit module budget is absent.
        if self.max_new_modules_per_trial is None or int(self.max_new_modules_per_trial) <= 0:
            self.max_new_modules_per_trial = self.max_new_neurons_per_trial
        else:
            self.max_new_modules_per_trial = int(self.max_new_modules_per_trial)
        self.learning_threshold = float(max(0.0, self.learning_threshold))
        self.initial_gate = float(np.clip(self.initial_gate, -8.0, -0.25))
        self.initial_weight = float(np.clip(abs(self.initial_weight), 0.0, 0.15))
        self.module_initial_gate = float(np.clip(self.module_initial_gate, -6.0, -0.15))
        self.module_initial_weight = float(np.clip(abs(self.module_initial_weight), 0.005, 0.20))
        self.module_input_adapters = max(1, int(self.module_input_adapters))
        self.module_core_neurons = max(2, int(self.module_core_neurons))
        self.module_output_adapters = max(1, int(self.module_output_adapters))
        fallback_threshold = max(1, int(self.survival_contexts))
        self.survival_similar_outcomes = max(1, int(self.survival_similar_outcomes if self.survival_similar_outcomes is not None else fallback_threshold))
        self.survival_contexts = int(self.survival_similar_outcomes)
        self.survival_positive_supports = max(1, int(self.survival_positive_supports))
        self.survival_utility = float(self.survival_utility)
        self.consolidate_utility = float(max(self.survival_utility, self.consolidate_utility))
        self.context_similarity_threshold = float(np.clip(self.context_similarity_threshold, 0.50, 0.999))
        self.context_prototype_lr = float(np.clip(self.context_prototype_lr, 0.001, 0.50))
        self.fade_rate = float(np.clip(self.fade_rate, 0.50, 0.999))
        self.gate_fade = float(np.clip(self.gate_fade, 0.001, 1.0))
        self.prune_fade_threshold = float(np.clip(self.prune_fade_threshold, 0.01, 0.95))
        self.grace_outcomes = max(0, int(self.grace_outcomes))
        self.min_causal_confidence_to_activate = float(np.clip(self.min_causal_confidence_to_activate, 0.0, 1.0))
        self.max_events = None if self.max_events is None or int(self.max_events) <= 0 else int(self.max_events)

    def begin_trial(self, trial_id: str) -> None:
        trial_id = str(trial_id or "")
        if trial_id == self.current_trial_id:
            return
        self.current_trial_id = trial_id
        self.outcome_step = 0
        self.grown_this_trial = 0

    def on_outcome(
        self,
        graph: NeuralGraph,
        *,
        trial_id: str,
        role: str,
        region: str,
        reason: str,
        trace: object,
        learning_signal: Mapping[str, float],
        connector: object | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[TrialMicroGrowthEvent]:
        """Update existing modules and create one bounded candidate circuit.

        ``region`` and ``connector`` remain accepted for call compatibility;
        the factory selects role-safe source/core/target regions itself.
        """
        del region, connector
        if not self.enabled or not trial_id:
            return []
        self.begin_trial(trial_id)
        self.outcome_step += 1
        rng = rng or np.random.default_rng()
        credit = _credit(learning_signal)
        learning_strength = _learning_strength(learning_signal)
        context_key = _context_key(trace)
        context_vector = _context_vector(trace)
        causal_confidence = float(np.clip(_finite(learning_signal.get("causal_confidence", 0.0)), 0.0, 1.0))
        causal_validated = bool(learning_signal.get("causal_validated", False))
        causal_pattern_key = str(learning_signal.get("causal_pattern_key", reason) or reason)
        emitted = self._update_survival(
            graph,
            trial_id=trial_id,
            role=role,
            reason=reason,
            trace=trace,
            credit=credit,
            context_key=context_key,
            context_vector=context_vector,
            learning_strength=learning_strength,
            causal_confidence=causal_confidence,
            causal_validated=causal_validated,
        )

        budget_open = self.max_new_modules_per_trial is None or self.grown_this_trial < self.max_new_modules_per_trial
        if learning_strength >= self.learning_threshold and budget_open:
            event = self._recruit_module(
                graph,
                trial_id=trial_id,
                role=role,
                reason=reason,
                context_key=context_key,
                context_vector=context_vector,
                learning_strength=learning_strength,
                credit=credit,
                causal_confidence=causal_confidence,
                causal_validated=causal_validated,
                causal_pattern_key=causal_pattern_key,
                rng=rng,
            )
            if event is not None:
                emitted.append(event)
                self.grown_this_trial += 1

        for event in emitted:
            self._record(event)
        graph.assert_invariants()
        return emitted

    def metrics(self, graph: NeuralGraph | None = None) -> dict[str, float]:
        rows = list(self.events)
        metrics = {
            "trial_micro_growth_events": float(len(rows)),
            "specialist_modules_created": float(sum(1 for row in rows if row.get("kind") == "specialist_module_candidate_created")),
            "specialist_modules_active": float(sum(1 for row in rows if row.get("kind") == "specialist_module_activated")),
            "specialist_modules_consolidated": float(sum(1 for row in rows if row.get("kind") == "specialist_module_consolidated")),
            "specialist_modules_fading": float(sum(1 for row in rows if row.get("kind") == "specialist_module_fading")),
            "specialist_modules_pruned": float(sum(1 for row in rows if row.get("kind") == "specialist_module_pruned")),
        }
        if graph is not None:
            registry = _module_registry(graph)
            items = [raw for raw in registry.values() if str(raw.get("trial_id", "")) == str(self.current_trial_id)]
            metrics.update({
                "specialist_module_count": float(len(items)),
                "specialist_module_candidate_count": float(sum(str(item.get("status", "")) == "candidate" for item in items)),
                "specialist_module_active_count": float(sum(str(item.get("status", "")) == "active" for item in items)),
                "specialist_module_consolidated_count": float(sum(str(item.get("status", "")) == "consolidated" for item in items)),
                "specialist_module_pruning_count": float(sum(str(item.get("status", "")) == "pruning" for item in items)),
                "specialist_module_neuron_count": float(sum(len(tuple(item.get("node_ids", ()) or ())) for item in items)),
                "specialist_module_edge_count": float(sum(len(tuple(item.get("edge_ids", ()) or ())) for item in items)),
            })
        return metrics

    def _recruit_module(
        self,
        graph: NeuralGraph,
        *,
        trial_id: str,
        role: str,
        reason: str,
        context_key: str,
        context_vector: np.ndarray,
        learning_strength: float,
        credit: float,
        causal_confidence: float,
        causal_validated: bool,
        causal_pattern_key: str,
        rng: np.random.Generator,
    ) -> TrialMicroGrowthEvent | None:
        module_id = f"specialist:{role}:{trial_id}:{self.grown_this_trial}"
        try:
            build = SpecialistModuleFactory(
                input_adapters=self.module_input_adapters,
                core_neurons=self.module_core_neurons,
                output_adapters=self.module_output_adapters,
                initial_gate=self.module_initial_gate,
                initial_weight=self.module_initial_weight,
                enable_bypass_interfaces=self.module_enable_bypass_interfaces,
            ).create(
                graph,
                module_id=module_id,
                role=role,
                reason=reason,
                candidate_id=trial_id,
                trial_id=trial_id,
                context_key=context_key,
                context_vector=context_vector,
                rng=rng,
                causal_confidence=causal_confidence,
                causal_validated=causal_validated,
                causal_pattern_key=causal_pattern_key,
            )
        except (ValueError, KeyError):
            return None
        record = _module_registry(graph).get(module_id, {})
        record["born_outcome_step"] = int(self.outcome_step)
        return TrialMicroGrowthEvent(
            kind="specialist_module_candidate_created",
            trial_id=trial_id,
            neuron_id=build.core_ids[0] if build.core_ids else None,
            module_id=module_id,
            role=role,
            reason=reason,
            context_key=context_key,
            learning_strength=learning_strength,
            credit=credit,
            details={
                **build.to_dict(),
                "initial_gate": self.module_initial_gate,
                "initial_weight": self.module_initial_weight,
                "context_similarity_threshold": self.context_similarity_threshold,
                "causal_confidence": causal_confidence,
                "causal_validated": causal_validated,
                "causal_pattern_key": causal_pattern_key,
                "evidence_tier": str(record.get("evidence_tier", "exploratory")),
            },
        )

    def _update_survival(
        self,
        graph: NeuralGraph,
        *,
        trial_id: str,
        role: str,
        reason: str,
        trace: object,
        credit: float,
        context_key: str,
        context_vector: np.ndarray,
        learning_strength: float,
        causal_confidence: float,
        causal_validated: bool,
    ) -> list[TrialMicroGrowthEvent]:
        active_nodes = set(str(item) for item in (getattr(trace, "active_neuron_ids", ()) or ()))
        active_edges = set(str(item) for item in (getattr(trace, "active_edge_ids", ()) or ()))
        events: list[TrialMicroGrowthEvent] = []
        registry = _module_registry(graph)
        for module_id, record in list(registry.items()):
            if str(record.get("trial_id", "")) != str(trial_id):
                continue
            node_ids = [node_id for node_id in tuple(record.get("node_ids", ()) or ()) if node_id in graph.neurons]
            edge_ids = [edge_id for edge_id in tuple(record.get("edge_ids", ()) or ()) if edge_id in graph.edges]
            if not node_ids:
                record["status"] = "pruning"
                continue
            similarity = _context_similarity(record.get("context_prototype"), context_vector)
            record["last_context_similarity"] = similarity
            if similarity < self.context_similarity_threshold:
                continue
            record["similar_context_observations"] = int(record.get("similar_context_observations", 0) or 0) + 1
            # The circuit can earn activity evidence either from sparse trace
            # IDs or from its core activation, avoiding an all-or-nothing
            # failure when a telemetry trace cap hides weak candidate edges.
            used = bool(active_nodes.intersection(node_ids) or active_edges.intersection(edge_ids))
            if not used:
                used = any(abs(float(getattr(graph.neurons[node_id], "activation", 0.0))) > 1e-7 for node_id in node_ids)
            module_utility = _module_utility(graph, node_ids, edge_ids)
            record["utility"] = module_utility
            if causal_validated:
                record["causal_validated"] = True
                record["causal_confidence"] = max(float(record.get("causal_confidence", 0.0) or 0.0), causal_confidence)
                record["evidence_tier"] = "causal_validated"
            elif causal_confidence > float(record.get("causal_confidence", 0.0) or 0.0):
                record["causal_confidence"] = causal_confidence
                if causal_confidence >= 0.45:
                    record["evidence_tier"] = "causal_supported"

            if used and credit > 0.0:
                record["positive_supports"] = int(record.get("positive_supports", 0) or 0) + 1
                record["similar_context_supports"] = int(record.get("similar_context_supports", 0) or 0) + 1
                record["fade"] = float(np.clip(float(record.get("fade", 1.0) or 1.0) + 0.08, 0.0, 1.0))
                record["context_prototype"] = _blend_context(record.get("context_prototype"), context_vector, self.context_prototype_lr)
                self._support_module(graph, node_ids, edge_ids, record)
            elif used or int(record.get("similar_context_observations", 0) or 0) > self.grace_outcomes:
                self._fade_module(graph, node_ids, edge_ids, record)

            supports = int(record.get("similar_context_supports", 0) or 0)
            positives = int(record.get("positive_supports", 0) or 0)
            causal_ready = bool(record.get("causal_validated", False)) or float(record.get("causal_confidence", 0.0) or 0.0) >= self.min_causal_confidence_to_activate
            retained = supports >= int(self.survival_similar_outcomes or 1) and positives >= self.survival_positive_supports and module_utility >= self.survival_utility
            details = self._module_details(record, similarity=similarity, node_ids=node_ids, edge_ids=edge_ids, module_utility=module_utility)
            status = str(record.get("status", "candidate"))
            if retained and causal_ready and status == "candidate":
                self._set_module_lifecycle(graph, node_ids, edge_ids, record, NEURON_LIFECYCLE_ACTIVE)
                record["status"] = "active"
                events.append(self._event("specialist_module_activated", trial_id, module_id, role, reason, context_key, learning_strength, credit, details))
                status = "active"
            elif retained and status == "candidate" and not causal_ready:
                events.append(self._event("specialist_module_waiting_causal_evidence", trial_id, module_id, role, reason, context_key, learning_strength, credit, details))

            can_consolidate = retained and module_utility >= self.consolidate_utility and supports >= int(self.survival_similar_outcomes or 1) + 1
            if self.require_causal_validation_to_consolidate:
                can_consolidate = can_consolidate and bool(record.get("causal_validated", False))
            if can_consolidate and str(record.get("status", "")) in {"candidate", "active"}:
                self._set_module_lifecycle(graph, node_ids, edge_ids, record, NEURON_LIFECYCLE_CONSOLIDATED)
                record["status"] = "consolidated"
                events.append(self._event("specialist_module_consolidated", trial_id, module_id, role, reason, context_key, learning_strength, credit, details))

            if str(record.get("status", "")) == "candidate" and int(record.get("similar_context_observations", 0) or 0) > self.grace_outcomes and float(record.get("fade", 1.0) or 1.0) <= self.prune_fade_threshold:
                self._set_module_lifecycle(graph, node_ids, edge_ids, record, NEURON_LIFECYCLE_PRUNING)
                record["status"] = "pruning"
                for edge_id in edge_ids:
                    if graph.edges[edge_id].enabled:
                        graph.disable_connection(edge_id)
                events.append(self._event("specialist_module_pruned", trial_id, module_id, role, reason, context_key, learning_strength, credit, {**details, "fade": record.get("fade", 0.0)}))
            elif str(record.get("status", "")) == "candidate" and int(record.get("similar_context_observations", 0) or 0) > self.grace_outcomes and (not used or credit <= 0.0):
                events.append(self._event("specialist_module_fading", trial_id, module_id, role, reason, context_key, learning_strength, credit, {**details, "fade": record.get("fade", 1.0)}))
        return events

    def _support_module(self, graph: NeuralGraph, node_ids: list[str], edge_ids: list[str], record: dict[str, object]) -> None:
        for node_id in node_ids:
            node = graph.neurons[node_id]
            node.positive_supports = int(getattr(node, "positive_supports", 0)) + 1
            node.similar_context_supports = int(getattr(node, "similar_context_supports", 0)) + 1
            node.similar_context_observations = int(getattr(node, "similar_context_observations", 0)) + 1
            node.fade = float(np.clip(float(getattr(node, "fade", 1.0)) + 0.08, 0.0, 1.0))
            node.context_prototype = _blend_context(getattr(node, "context_prototype", None), record.get("context_prototype"), self.context_prototype_lr)
        for edge_id in edge_ids:
            edge = graph.edges[edge_id]
            edge.positive_supports = int(getattr(edge, "positive_supports", 0)) + 1
            edge.similar_context_supports = int(getattr(edge, "similar_context_supports", 0)) + 1
            edge.similar_context_observations = int(getattr(edge, "similar_context_observations", 0)) + 1
            edge.fade = float(np.clip(float(getattr(edge, "fade", 1.0)) + 0.08, 0.0, 1.0))
            # Let a useful candidate earn a small amount of authority instead
            # of remaining at a gate where it can never be credited again.
            edge.gate = float(np.clip(float(edge.gate) + 0.10, -8.0, 1.5))
            edge.context_prototype = _blend_context(getattr(edge, "context_prototype", None), record.get("context_prototype"), self.context_prototype_lr)

    def _fade_module(self, graph: NeuralGraph, node_ids: list[str], edge_ids: list[str], record: dict[str, object]) -> None:
        record["fade"] = float(np.clip(float(record.get("fade", 1.0) or 1.0) * self.fade_rate, 0.0, 1.0))
        for node_id in node_ids:
            node = graph.neurons[node_id]
            node.fade = float(np.clip(float(getattr(node, "fade", 1.0)) * self.fade_rate, 0.0, 1.0))
            node.plasticity = float(np.clip(float(getattr(node, "plasticity", 1.0)) * 1.01, 0.05, 3.0))
        for edge_id in edge_ids:
            edge = graph.edges[edge_id]
            if not edge.enabled:
                continue
            edge.fade = float(np.clip(float(getattr(edge, "fade", 1.0)) * self.fade_rate, 0.0, 1.0))
            edge.weight = float(np.clip(float(edge.weight) * self.fade_rate, -10.0, 10.0))
            edge.gate = float(np.clip(float(edge.gate) - self.gate_fade, -8.0, 8.0))
            edge.plasticity = float(np.clip(float(edge.plasticity) * 1.01, 0.05, 3.0))

    @staticmethod
    def _set_module_lifecycle(graph: NeuralGraph, node_ids: list[str], edge_ids: list[str], record: dict[str, object], lifecycle: str) -> None:
        for node_id in node_ids:
            graph.neurons[node_id].lifecycle = lifecycle
        for edge_id in edge_ids:
            edge = graph.edges[edge_id]
            if edge.enabled:
                edge.lifecycle = lifecycle
                if lifecycle == NEURON_LIFECYCLE_ACTIVE:
                    edge.gate = float(np.clip(edge.gate + 0.05, -8.0, 2.0))
                elif lifecycle == NEURON_LIFECYCLE_CONSOLIDATED:
                    edge.plasticity = float(np.clip(edge.plasticity * 0.96, 0.05, 3.0))
        record["lifecycle"] = lifecycle

    def _event(self, kind: str, trial_id: str, module_id: str, role: str, reason: str, context_key: str, learning_strength: float, credit: float, details: dict[str, object]) -> TrialMicroGrowthEvent:
        return TrialMicroGrowthEvent(
            kind=kind,
            trial_id=trial_id,
            neuron_id=str(details.get("core_ids", [""])[0]) if details.get("core_ids") else None,
            module_id=module_id,
            role=role,
            reason=reason,
            context_key=context_key,
            learning_strength=learning_strength,
            credit=credit,
            details=details,
        )

    def _module_details(self, record: Mapping[str, object], *, similarity: float, node_ids: list[str], edge_ids: list[str], module_utility: float) -> dict[str, object]:
        return {
            "node_ids": list(node_ids),
            "edge_ids": list(edge_ids),
            "core_ids": list(tuple(record.get("core_ids", ()) or ())),
            "input_adapter_ids": list(tuple(record.get("input_adapter_ids", ()) or ())),
            "output_adapter_ids": list(tuple(record.get("output_adapter_ids", ()) or ())),
            "similarity": similarity,
            "similarity_threshold": self.context_similarity_threshold,
            "similar_context_observations": int(record.get("similar_context_observations", 0) or 0),
            "similar_context_supports": int(record.get("similar_context_supports", 0) or 0),
            "positive_supports": int(record.get("positive_supports", 0) or 0),
            "utility": module_utility,
            "fade": float(record.get("fade", 1.0) or 1.0),
            "status": str(record.get("status", "candidate")),
            "evidence_tier": str(record.get("evidence_tier", "exploratory")),
            "causal_confidence": float(record.get("causal_confidence", 0.0) or 0.0),
            "causal_validated": bool(record.get("causal_validated", False)),
            "causal_pattern_key": str(record.get("causal_pattern_key", "")),
            "home_context_key": str(record.get("home_context_key", "")),
        }

    def _record(self, event: TrialMicroGrowthEvent) -> None:
        self.events.append(event.to_dict())
        if self.max_events is not None and len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]


def _module_registry(graph: NeuralGraph) -> dict[str, dict[str, object]]:
    registry = getattr(graph, "specialist_modules", None)
    if not isinstance(registry, dict):
        registry = {}
        graph.specialist_modules = registry
    return registry


def _module_utility(graph: NeuralGraph, node_ids: list[str], edge_ids: list[str]) -> float:
    values: list[float] = []
    values.extend(float(getattr(graph.neurons[node_id], "utility", 0.0)) for node_id in node_ids)
    values.extend(float(getattr(graph.edges[edge_id], "utility", 0.0)) for edge_id in edge_ids if graph.edges[edge_id].enabled)
    return float(np.mean(values)) if values else 0.0


def _credit(signal: Mapping[str, float]) -> float:
    return float(np.clip(
        _finite(signal.get("reward", 0.0))
        - _finite(signal.get("cost", 0.0))
        + 0.5 * _finite(signal.get("td_delta", 0.0))
        - 0.10 * abs(_finite(signal.get("prediction_error", 0.0))),
        -3.0,
        3.0,
    ))


def _learning_strength(signal: Mapping[str, float]) -> float:
    return float(np.clip(
        abs(_finite(signal.get("td_delta", 0.0)))
        + 0.50 * min(1.0, abs(_finite(signal.get("prediction_error", 0.0))))
        + min(1.0, abs(_finite(signal.get("reward", 0.0)) - _finite(signal.get("cost", 0.0)))),
        0.0,
        3.0,
    ))


def _finite(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if np.isfinite(parsed) else 0.0


def _context_key(trace: object) -> str:
    account = dict(getattr(trace, "account_context", {}) or {})
    tags = ",".join(sorted(str(tag) for tag in (getattr(trace, "lifecycle_tags", ()) or ()))) or "none"
    risk = _bucket(getattr(trace, "risk", 0.0), 4)
    novelty = _bucket(getattr(trace, "novelty", 0.0), 4)
    conflict = _bucket(getattr(trace, "conflict", 0.0), 4)
    drawdown = _bucket(account.get("current_equity_drawdown", 0.0), 4)
    memory = _bucket(getattr(trace, "memory_match_quality", 0.0), 4)
    position = "open" if str(getattr(trace, "positions_pipe", "") or "") else "flat"
    return f"r{risk}|n{novelty}|c{conflict}|d{drawdown}|m{memory}|p{position}|t{tags}"


def _context_vector(trace: object, state_size: int = 16) -> np.ndarray:
    raw = getattr(trace, "encoded_state", None)
    if raw is None:
        raw = getattr(trace, "raw_obs", None)
    state = _resize_vector(raw, state_size)
    state = np.tanh(np.nan_to_num(state, nan=0.0, posinf=0.0, neginf=0.0))
    state = _unit(state)
    account = dict(getattr(trace, "account_context", {}) or {})
    scalars = np.asarray([
        _bounded(getattr(trace, "risk", 0.0)),
        _bounded(getattr(trace, "novelty", 0.0)),
        _bounded(getattr(trace, "conflict", 0.0)),
        _bounded(getattr(trace, "memory_match_quality", 0.0)),
        _bounded(account.get("current_equity_drawdown", 0.0)),
        _bounded(account.get("open_risk_ratio", 0.0)),
        float(bool(str(getattr(trace, "positions_pipe", "") or ""))),
        float(np.clip((_finite(account.get("floating_pnl_r", 0.0)) + 2.0) / 4.0, 0.0, 1.0)),
    ], dtype=float)
    tag_features = np.zeros(4, dtype=float)
    for tag in (getattr(trace, "lifecycle_tags", ()) or ()):
        idx = zlib.crc32(str(tag).encode("utf-8")) % tag_features.size
        tag_features[idx] = 1.0
    return _unit(np.concatenate((0.80 * state, 0.35 * scalars, 0.20 * tag_features)))


def _context_similarity(prototype: object, current: np.ndarray) -> float:
    if prototype is None:
        return 1.0
    base = _resize_vector(prototype, current.size)
    base = _unit(base)
    current = _unit(current)
    if base.size == 0 or current.size == 0:
        return 1.0
    return float(np.clip(np.dot(base, current), -1.0, 1.0))


def _blend_context(prototype: object, current: object, learning_rate: float) -> np.ndarray:
    if current is None:
        return _resize_vector(prototype, 0)
    current_array = np.asarray(current, dtype=float).reshape(-1)
    if prototype is None:
        return _unit(current_array).copy()
    base = _resize_vector(prototype, current_array.size)
    return _unit((1.0 - learning_rate) * base + learning_rate * current_array)


def _unit(values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector.copy()


def _bounded(value: object) -> float:
    return float(np.clip(_finite(value), -1.0, 1.0))


def _bucket(value: object, count: int) -> int:
    bounded = _bounded(value)
    return int(np.clip(np.floor((bounded + 1.0) * 0.5 * max(1, count)), 0, max(0, count - 1)))


def _resize_vector(value: object, size: int) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        vector = np.zeros(0, dtype=float)
    out = np.zeros(max(0, int(size)), dtype=float)
    if vector.size:
        out[: min(out.size, vector.size)] = vector[: min(out.size, vector.size)]
    return out
