from __future__ import annotations

"""True, candidate-only specialist modules for graph M1.

A specialist is a *small circuit*, not a renamed single neuron.  The factory
creates a bounded adapter -> recurrent/lateral core -> adapter structure,
registers one module-level lifecycle record on ``NeuralGraph``, and connects the
module to the existing M1 graph through weak, auditable interfaces.

The implementation intentionally keeps every node inside the existing semantic
regions so the graph primitive, MT5 action schema, and safety/legality gates do
not need a new execution contract.  ``module_id`` and ``module_stage`` carry
module identity and make the circuit visible in telemetry/UI.
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .module_need import (
    ROLE_CADENCE_EXPLORATION_SPECIALIST,
    ROLE_CRITIC_SPECIALIST,
    ROLE_ENTRY_QUALITY_SPECIALIST,
    ROLE_EXIT_QUALITY_SPECIALIST,
    ROLE_MEMORY_CRITIC_SPECIALIST,
    ROLE_MEMORY_SPECIALIST,
    ROLE_POSITION_MANAGEMENT_SPECIALIST,
    ROLE_PREDICTOR_SPECIALIST,
    ROLE_RISK_GATE_SPECIALIST,
    ROLE_SELECTOR_SPECIALIST,
    module_spec_for_role,
)
from .neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_LATERAL,
    EDGE_MEMORY,
    EDGE_META,
    EDGE_PREDICTIVE,
    EDGE_RECURRENT,
    EDGE_SELECTOR,
    NEURON_LIFECYCLE_CANDIDATE,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_HYPOTHESIS,
    REGION_MEMORY,
    REGION_META,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    NeuralGraph,
)


@dataclass(frozen=True)
class SpecialistModuleBuild:
    module_id: str
    role: str
    reason: str
    context_key: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    input_adapter_ids: tuple[str, ...]
    core_ids: tuple[str, ...]
    output_adapter_ids: tuple[str, ...]
    interface_edge_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "module_id": self.module_id,
            "role": self.role,
            "reason": self.reason,
            "context_key": self.context_key,
            "node_ids": list(self.node_ids),
            "edge_ids": list(self.edge_ids),
            "input_adapter_ids": list(self.input_adapter_ids),
            "core_ids": list(self.core_ids),
            "output_adapter_ids": list(self.output_adapter_ids),
            "interface_edge_ids": list(self.interface_edge_ids),
            "node_count": len(self.node_ids),
            "edge_count": len(self.edge_ids),
        }


@dataclass(frozen=True)
class SpecialistModuleLayout:
    source_regions: tuple[str, ...]
    core_region: str
    target_regions: tuple[str, ...]
    inbound_kind: str
    outbound_kind: str


_ROLE_LAYOUTS: dict[str, SpecialistModuleLayout] = {
    ROLE_PREDICTOR_SPECIALIST: SpecialistModuleLayout(
        (REGION_ENCODER, REGION_MEMORY, REGION_HYPOTHESIS), REGION_PREDICTOR,
        (REGION_CRITIC,), EDGE_PREDICTIVE, EDGE_CRITIC,
    ),
    ROLE_MEMORY_SPECIALIST: SpecialistModuleLayout(
        (REGION_ENCODER, REGION_MEMORY), REGION_MEMORY,
        (REGION_PREDICTOR, REGION_CRITIC, REGION_HYPOTHESIS), EDGE_MEMORY, EDGE_MEMORY,
    ),
    ROLE_CRITIC_SPECIALIST: SpecialistModuleLayout(
        (REGION_PREDICTOR, REGION_MEMORY), REGION_CRITIC,
        (REGION_SELECTOR, REGION_OUTPUT), EDGE_CRITIC, EDGE_SELECTOR,
    ),
    ROLE_SELECTOR_SPECIALIST: SpecialistModuleLayout(
        (REGION_HYPOTHESIS, REGION_CRITIC), REGION_SELECTOR,
        (REGION_OUTPUT,), EDGE_SELECTOR, EDGE_SELECTOR,
    ),
    ROLE_POSITION_MANAGEMENT_SPECIALIST: SpecialistModuleLayout(
        (REGION_MEMORY, REGION_SELECTOR, REGION_CRITIC), REGION_SELECTOR,
        (REGION_OUTPUT,), EDGE_MEMORY, EDGE_GATING,
    ),
    ROLE_RISK_GATE_SPECIALIST: SpecialistModuleLayout(
        (REGION_ENCODER, REGION_CRITIC, REGION_MEMORY), REGION_CRITIC,
        (REGION_SELECTOR, REGION_OUTPUT), EDGE_CRITIC, EDGE_GATING,
    ),
    ROLE_CADENCE_EXPLORATION_SPECIALIST: SpecialistModuleLayout(
        (REGION_ENCODER, REGION_MEMORY, REGION_SELECTOR), REGION_META,
        (REGION_SELECTOR, REGION_OUTPUT), EDGE_META, EDGE_GATING,
    ),
    ROLE_ENTRY_QUALITY_SPECIALIST: SpecialistModuleLayout(
        (REGION_ENCODER, REGION_MEMORY, REGION_HYPOTHESIS), REGION_SELECTOR,
        (REGION_CRITIC, REGION_OUTPUT), EDGE_FORWARD, EDGE_SELECTOR,
    ),
    ROLE_EXIT_QUALITY_SPECIALIST: SpecialistModuleLayout(
        (REGION_MEMORY, REGION_SELECTOR, REGION_CRITIC), REGION_SELECTOR,
        (REGION_OUTPUT,), EDGE_MEMORY, EDGE_GATING,
    ),
    ROLE_MEMORY_CRITIC_SPECIALIST: SpecialistModuleLayout(
        (REGION_MEMORY, REGION_PREDICTOR), REGION_CRITIC,
        (REGION_MEMORY, REGION_SELECTOR), EDGE_MEMORY, EDGE_CRITIC,
    ),
}


def layout_for_role(role: str) -> SpecialistModuleLayout:
    return _ROLE_LAYOUTS.get(
        str(role),
        SpecialistModuleLayout((REGION_ENCODER, REGION_MEMORY), REGION_META, (REGION_SELECTOR,), EDGE_FORWARD, EDGE_META),
    )


class SpecialistModuleFactory:
    """Build a small auditable circuit inside an existing ``NeuralGraph``."""

    def __init__(
        self,
        *,
        input_adapters: int = 3,
        core_neurons: int = 4,
        output_adapters: int = 2,
        initial_gate: float = -2.15,
        initial_weight: float = 0.055,
        enable_bypass_interfaces: bool = True,
    ) -> None:
        self.input_adapters = max(1, int(input_adapters))
        self.core_neurons = max(2, int(core_neurons))
        self.output_adapters = max(1, int(output_adapters))
        self.initial_gate = float(np.clip(initial_gate, -6.0, -0.15))
        self.initial_weight = float(np.clip(abs(initial_weight), 0.005, 0.20))
        self.enable_bypass_interfaces = bool(enable_bypass_interfaces)

    def create(
        self,
        graph: NeuralGraph,
        *,
        module_id: str,
        role: str,
        reason: str,
        candidate_id: str,
        trial_id: str,
        context_key: str,
        context_vector: np.ndarray | None,
        rng: np.random.Generator | None = None,
        causal_confidence: float = 0.0,
        causal_validated: bool = False,
        causal_pattern_key: str = "",
    ) -> SpecialistModuleBuild:
        rng = rng or np.random.default_rng()
        if module_id in getattr(graph, "specialist_modules", {}):
            raise ValueError(f"duplicate specialist module: {module_id}")
        layout = layout_for_role(role)
        spec = module_spec_for_role(role, reason=reason)
        node_ids: list[str] = []
        internal_edges: list[str] = []
        interface_edges: list[str] = []

        input_nodes: list[str] = []
        for idx in range(self.input_adapters):
            region = layout.source_regions[idx % len(layout.source_regions)]
            node_id = f"{module_id}:in:{idx}"
            graph.add_neuron(
                region, neuron_id=node_id, module_id=module_id,
                lifecycle=NEURON_LIFECYCLE_CANDIDATE,
                plasticity=float(spec.initial_plasticity),
                bias=float(np.clip(rng.normal(0.0, 0.015), -0.04, 0.04)),
            )
            node = graph.neurons[node_id]
            node.module_stage = "input_adapter"
            node.module_role = str(role)
            node.birth_trial_id = str(trial_id)
            node.home_context_key = str(context_key)
            node.context_prototype = _unit(context_vector)
            input_nodes.append(node_id)
            node_ids.append(node_id)

        core_nodes: list[str] = []
        for idx in range(self.core_neurons):
            node_id = f"{module_id}:core:{idx}"
            graph.add_neuron(
                layout.core_region, neuron_id=node_id, module_id=module_id,
                lifecycle=NEURON_LIFECYCLE_CANDIDATE,
                plasticity=float(spec.initial_plasticity),
                forget_gate=-1.25 if layout.core_region in {REGION_MEMORY, REGION_PREDICTOR} else 0.0,
                bias=float(np.clip(rng.normal(0.0, 0.02), -0.05, 0.05)),
            )
            node = graph.neurons[node_id]
            node.module_stage = "core"
            node.module_role = str(role)
            node.birth_trial_id = str(trial_id)
            node.home_context_key = str(context_key)
            node.context_prototype = _unit(context_vector)
            core_nodes.append(node_id)
            node_ids.append(node_id)

        output_nodes: list[str] = []
        for idx in range(self.output_adapters):
            # The M1 output region has a fixed action schema (20 channels).
            # A specialist must never append a new output neuron there, or the
            # decoder/action contract breaks.  Its output adapters stay in the
            # role core region and connect weakly into canonical M1 outputs.
            requested_region = layout.target_regions[idx % len(layout.target_regions)]
            region = layout.core_region if requested_region == REGION_OUTPUT else requested_region
            node_id = f"{module_id}:out:{idx}"
            graph.add_neuron(
                region, neuron_id=node_id, module_id=module_id,
                lifecycle=NEURON_LIFECYCLE_CANDIDATE,
                plasticity=float(spec.initial_plasticity),
                bias=float(np.clip(rng.normal(0.0, 0.015), -0.04, 0.04)),
            )
            node = graph.neurons[node_id]
            node.module_stage = "output_adapter"
            node.module_role = str(role)
            node.birth_trial_id = str(trial_id)
            node.home_context_key = str(context_key)
            node.context_prototype = _unit(context_vector)
            output_nodes.append(node_id)
            node_ids.append(node_id)

        # Internal adapter -> core fan-in.  The small circuit can represent a
        # local context before it gains any authority over M1.
        for src in input_nodes:
            for dst in core_nodes:
                edge_id = self._edge(
                    graph, src, dst, kind=layout.inbound_kind, weight_scale=1.0,
                    gate=self.initial_gate, reason=reason, candidate_id=candidate_id,
                    role=role, trial_id=trial_id, context_key=context_key,
                    context_vector=context_vector, rng=rng,
                )
                if edge_id:
                    internal_edges.append(edge_id)

        # Recurrent ring + sparse lateral cross-talk. The ring is deliberately
        # bounded so a newborn has state but cannot dominate runtime cost.
        for idx, src in enumerate(core_nodes):
            dst = core_nodes[(idx + 1) % len(core_nodes)]
            edge_id = self._edge(
                graph, src, dst, kind=EDGE_RECURRENT, weight_scale=0.70,
                gate=self.initial_gate - 0.20, reason=reason, candidate_id=candidate_id,
                role=role, trial_id=trial_id, context_key=context_key,
                context_vector=context_vector, rng=rng,
            )
            if edge_id:
                internal_edges.append(edge_id)
        for idx, src in enumerate(core_nodes):
            for jdx, dst in enumerate(core_nodes):
                if idx == jdx or (idx + jdx) % 2:
                    continue
                edge_id = self._edge(
                    graph, src, dst, kind=EDGE_LATERAL, weight_scale=0.50,
                    gate=self.initial_gate - 0.35, reason=reason, candidate_id=candidate_id,
                    role=role, trial_id=trial_id, context_key=context_key,
                    context_vector=context_vector, rng=rng,
                )
                if edge_id:
                    internal_edges.append(edge_id)

        for src in core_nodes:
            for dst in output_nodes:
                edge_id = self._edge(
                    graph, src, dst, kind=layout.outbound_kind, weight_scale=0.85,
                    gate=self.initial_gate, reason=reason, candidate_id=candidate_id,
                    role=role, trial_id=trial_id, context_key=context_key,
                    context_vector=context_vector, rng=rng,
                )
                if edge_id:
                    internal_edges.append(edge_id)

        # Interfaces preserve the path M1 -> module -> M1.  Direct source ->
        # core and core -> sink bypasses let a 3-step M1 activation expose a
        # newborn circuit in the same decision, while adapters still provide a
        # visible, learnable module boundary.
        source_pool = self._m1_nodes(graph, layout.source_regions)
        target_pool = self._m1_nodes(graph, layout.target_regions)
        for idx, adapter in enumerate(input_nodes):
            source = source_pool[idx % len(source_pool)] if source_pool else None
            if source:
                edge_id = self._edge(
                    graph, source, adapter, kind=layout.inbound_kind, weight_scale=0.90,
                    gate=self.initial_gate, reason=reason, candidate_id=candidate_id,
                    role=role, trial_id=trial_id, context_key=context_key,
                    context_vector=context_vector, rng=rng,
                )
                if edge_id:
                    interface_edges.append(edge_id)
        for idx, adapter in enumerate(output_nodes):
            target = target_pool[idx % len(target_pool)] if target_pool else None
            if target:
                edge_id = self._edge(
                    graph, adapter, target, kind=layout.outbound_kind, weight_scale=0.75,
                    gate=self.initial_gate, reason=reason, candidate_id=candidate_id,
                    role=role, trial_id=trial_id, context_key=context_key,
                    context_vector=context_vector, rng=rng,
                )
                if edge_id:
                    interface_edges.append(edge_id)
        if self.enable_bypass_interfaces:
            for idx, core in enumerate(core_nodes):
                source = source_pool[idx % len(source_pool)] if source_pool else None
                if source:
                    edge_id = self._edge(
                        graph, source, core, kind=layout.inbound_kind, weight_scale=0.55,
                        gate=self.initial_gate + 0.35, reason=reason, candidate_id=candidate_id,
                        role=role, trial_id=trial_id, context_key=context_key,
                        context_vector=context_vector, rng=rng,
                    )
                    if edge_id:
                        interface_edges.append(edge_id)
            for idx, core in enumerate(core_nodes):
                target = target_pool[idx % len(target_pool)] if target_pool else None
                if target:
                    edge_id = self._edge(
                        graph, core, target, kind=layout.outbound_kind, weight_scale=0.45,
                        gate=self.initial_gate + 0.15, reason=reason, candidate_id=candidate_id,
                        role=role, trial_id=trial_id, context_key=context_key,
                        context_vector=context_vector, rng=rng,
                    )
                    if edge_id:
                        interface_edges.append(edge_id)

        all_edges = tuple(internal_edges + interface_edges)
        graph.specialist_modules[module_id] = {
            "module_id": module_id,
            "role": str(role),
            "reason": str(reason),
            "trial_id": str(trial_id),
            "candidate_id": str(candidate_id),
            "home_context_key": str(context_key),
            "context_prototype": _unit(context_vector),
            "node_ids": tuple(node_ids),
            "edge_ids": all_edges,
            "input_adapter_ids": tuple(input_nodes),
            "core_ids": tuple(core_nodes),
            "output_adapter_ids": tuple(output_nodes),
            "interface_edge_ids": tuple(interface_edges),
            "status": "candidate",
            "evidence_tier": "causal_validated" if causal_validated else ("causal_supported" if causal_confidence >= 0.45 else "exploratory"),
            "causal_confidence": float(np.clip(causal_confidence, 0.0, 1.0)),
            "causal_validated": bool(causal_validated),
            "causal_pattern_key": str(causal_pattern_key or reason),
            "similar_context_observations": 0,
            "similar_context_supports": 0,
            "positive_supports": 0,
            "utility": 0.0,
            "fade": 1.0,
            "born_outcome_step": 0,
            "last_context_similarity": 1.0,
        }
        graph.assert_invariants()
        return SpecialistModuleBuild(
            module_id=module_id,
            role=str(role),
            reason=str(reason),
            context_key=str(context_key),
            node_ids=tuple(node_ids),
            edge_ids=all_edges,
            input_adapter_ids=tuple(input_nodes),
            core_ids=tuple(core_nodes),
            output_adapter_ids=tuple(output_nodes),
            interface_edge_ids=tuple(interface_edges),
        )

    def _edge(
        self,
        graph: NeuralGraph,
        src: str,
        dst: str,
        *,
        kind: str,
        weight_scale: float,
        gate: float,
        reason: str,
        candidate_id: str,
        role: str,
        trial_id: str,
        context_key: str,
        context_vector: np.ndarray | None,
        rng: np.random.Generator,
    ) -> str | None:
        if src == dst and kind != EDGE_RECURRENT:
            return None
        if any(edge.enabled and edge.src == src and edge.dst == dst and edge.kind == kind for edge in graph.edges.values()):
            return None
        weight = float(np.clip(rng.normal(0.0, self.initial_weight * weight_scale), -self.initial_weight * 1.8, self.initial_weight * 1.8))
        if abs(weight) < self.initial_weight * 0.15:
            weight = self.initial_weight * 0.25 * (1.0 if rng.random() >= 0.5 else -1.0)
        try:
            edge_id = graph.add_connection(
                src, dst, weight=weight, gate=float(np.clip(gate, -8.0, 2.0)), kind=kind,
                created_by="specialist_module_factory", need_reason=str(reason),
                candidate_id=str(candidate_id), lifecycle=NEURON_LIFECYCLE_CANDIDATE,
            )
        except ValueError:
            return None
        edge = graph.edges[edge_id]
        edge.birth_trial_id = str(trial_id)
        edge.home_context_key = str(context_key)
        edge.context_prototype = _unit(context_vector)
        edge.module_role = str(role)
        return edge_id

    @staticmethod
    def _m1_nodes(graph: NeuralGraph, regions: Iterable[str]) -> list[str]:
        ordered: list[str] = []
        for region in regions:
            values = [
                node_id for node_id in graph.regions.get(region, ()).neuron_ids
                if str(getattr(graph.neurons[node_id], "module_id", "")) == "m1"
            ] if region in graph.regions else []
            values.sort(key=lambda node_id: (graph.neurons[node_id].utility, graph.neurons[node_id].activation, node_id), reverse=True)
            ordered.extend(values[:4])
        return ordered


def _unit(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    try:
        vector = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size == 0:
        return None
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else vector.copy()
