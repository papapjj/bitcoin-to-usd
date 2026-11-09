from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_LATERAL,
    EDGE_META,
    EDGE_PLASTIC,
    EDGE_RECURRENT,
    NEURON_LIFECYCLE_ACTIVE,
    NeuralGraph,
    NeuralGraphBudget,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_INPUT,
    REGION_MEMORY,
    REGION_META,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
)


REASONING_CONTROLLERS: tuple[str, ...] = (
    "curiosity",
    "development",
    "risk",
    "action_head",
    "utility",
)

CONTROLLER_REGIONS: dict[str, str] = {
    "curiosity": REGION_PREDICTOR,
    "development": REGION_META,
    "risk": REGION_CRITIC,
    "action_head": REGION_SELECTOR,
    "utility": REGION_MEMORY,
}

REASONING_SCHEMA_VERSION = 2


def _finite_vector(values: Any) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=float)
    return np.nan_to_num(array, nan=0.0, posinf=8.0, neginf=-8.0)


def _project_context(values: Any, size: int) -> np.ndarray:
    """Project an arbitrary controller context into a dense fixed-size vector.

    The former implementation copied only the first ``size`` values and filled
    remaining slots with zero.  That made a 12-value risk context leave 20 of
    32 reasoning inputs permanently silent and silently discarded the tail of
    longer action/head contexts.  This deterministic projection uses the whole
    context on every call: interpolated slots retain order, while tail slots
    carry stable distributional summaries.  It is a feature projection, not a
    learned policy update, therefore it cannot mutate model behaviour merely by
    being observed.
    """
    target = max(1, int(size))
    source = np.clip(_finite_vector(values), -8.0, 8.0)
    if source.size == 0:
        return np.zeros(target, dtype=float)

    stat_slots = min(8, max(0, target // 4))
    signal_slots = max(1, target - stat_slots)
    if source.size == 1:
        signal = np.full(signal_slots, float(source[0]), dtype=float)
    else:
        source_index = np.arange(source.size, dtype=float)
        target_index = np.linspace(0.0, float(source.size - 1), signal_slots)
        signal = np.interp(target_index, source_index, source)

    if stat_slots <= 0:
        return np.clip(signal[:target], -8.0, 8.0)

    quantiles = np.quantile(source, np.linspace(0.0, 1.0, min(3, stat_slots)))
    base_stats = np.asarray([
        float(np.mean(source)),
        float(np.std(source)),
        float(np.mean(np.abs(source))),
        float(np.min(source)),
        float(np.max(source)),
        float(source[0]),
        float(source[-1]),
        float(np.median(source)),
    ], dtype=float)
    if quantiles.size:
        base_stats[:quantiles.size] = quantiles
    stats = np.resize(base_stats, stat_slots)
    return np.clip(np.concatenate([signal, stats])[:target], -8.0, 8.0)


def _kind_for(src_region: str, dst_region: str) -> str:
    if src_region == dst_region:
        return EDGE_RECURRENT
    if dst_region == REGION_CRITIC:
        return EDGE_CRITIC
    if dst_region == REGION_SELECTOR:
        return EDGE_GATING
    if dst_region == REGION_MEMORY:
        return EDGE_PLASTIC
    if dst_region == REGION_PREDICTOR:
        return EDGE_FORWARD
    if dst_region == REGION_META:
        return EDGE_META
    return EDGE_FORWARD


@dataclass
class ReasoningGraphSystem:
    """Shared recurrent reasoning workspace for policy-level controllers.

    V18 preserves the graph as a separate, bounded-influence module while
    fixing three old failure modes: controller identity is now supplied as an
    explicit one-hot route, every context is densely projected instead of
    truncate/zero-pad, and immutable per-controller activation evidence is kept
    for lifecycle credit and live telemetry.
    """

    graph: NeuralGraph
    # ``input_size`` remains the number of semantic controller-context slots for
    # compatibility with older policy pickles.  Five controller identity inputs
    # are appended internally and are not part of this public context size.
    input_size: int = 32
    output_size: int = 16
    controllers: tuple[str, ...] = REASONING_CONTROLLERS
    influence: float = 0.08
    plasticity: float = 0.01
    age: int = 0
    last_controller: str = ""
    last_active_edge_ids: tuple[str, ...] = field(default_factory=tuple)
    last_active_neuron_ids: tuple[str, ...] = field(default_factory=tuple)
    last_controller_traces: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    last_controller_credit: dict[str, dict[str, float]] = field(default_factory=dict, repr=False)
    schema_version: int = REASONING_SCHEMA_VERSION

    @classmethod
    def seed(
        cls,
        *,
        input_size: int = 32,
        output_size: int = 16,
        controllers: Iterable[str] = REASONING_CONTROLLERS,
        rng: np.random.Generator | None = None,
    ) -> "ReasoningGraphSystem":
        rng = rng or np.random.default_rng()
        controller_names = tuple(dict.fromkeys(str(item) for item in controllers if str(item))) or REASONING_CONTROLLERS
        graph = NeuralGraph(
            budget=NeuralGraphBudget(
                max_neurons=0,
                max_connections=0,
                max_recurrent_edges=0,
                max_lateral_edges=0,
                max_cross_module_edges=0,
                max_m1_module_edges=0,
                max_cross_module_recurrent_edges=0,
                max_cross_module_lateral_edges=0,
            ),
            max_cross_module_initial_weight=0.20,
            trace_max_edges=128,
            trace_candidate_edges=32,
            trace_min_edges_per_route=8,
        )
        inputs = [
            graph.add_neuron(
                REGION_INPUT,
                neuron_id=f"reasoning:input:{i}",
                module_id="reasoning:context",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
            )
            for i in range(max(1, int(input_size)))
        ]
        controller_inputs = {
            controller: graph.add_neuron(
                REGION_INPUT,
                neuron_id=f"reasoning:controller:{controller}",
                module_id="reasoning:controller",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
            )
            for controller in controller_names
        }
        encoders = [
            graph.add_neuron(
                REGION_ENCODER,
                neuron_id=f"reasoning:encoder:{i}",
                module_id="reasoning:encoder",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
            )
            for i in range(12)
        ]
        outputs = [
            graph.add_neuron(
                REGION_OUTPUT,
                neuron_id=f"reasoning:output:{i}",
                module_id="reasoning:output",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
            )
            for i in range(max(1, int(output_size)))
        ]
        controller_nodes: dict[str, str] = {}
        for controller in controller_names:
            region = CONTROLLER_REGIONS.get(controller, REGION_META)
            controller_nodes[controller] = graph.add_neuron(
                region,
                neuron_id=f"reasoning:{controller}:core",
                module_id=f"reasoning:{controller}",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
                plasticity=1.2,
            )

        scale = 1.0 / max(1.0, np.sqrt(float(max(1, int(input_size)))))
        for src in inputs:
            for dst in encoders:
                graph.add_connection(
                    src,
                    dst,
                    weight=float(rng.normal(0.0, scale * 0.10)),
                    gate=-1.20,
                    kind=EDGE_FORWARD,
                    created_by="reasoning_seed_v18",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
        for controller, node_id in controller_nodes.items():
            region = graph.neurons[node_id].region
            for enc in encoders:
                graph.add_connection(
                    enc,
                    node_id,
                    weight=float(rng.normal(0.0, 0.05)),
                    gate=-2.10,
                    kind=_kind_for(REGION_ENCODER, region),
                    created_by="reasoning_seed_v18",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
            # Controller identity is a real neural input.  It gives the selected
            # core a strong route while weak encoder/lateral links preserve
            # shared workspace without cross-controller domination.
            graph.add_connection(
                controller_inputs[controller],
                node_id,
                weight=float(rng.normal(0.62, 0.04)),
                gate=0.35,
                kind=EDGE_GATING,
                created_by="reasoning_controller_route_v18",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
            )
            graph.add_connection(
                node_id,
                node_id,
                weight=float(rng.normal(0.0, 0.025)),
                gate=-2.8,
                kind=EDGE_RECURRENT,
                created_by="reasoning_seed_v18",
                lifecycle=NEURON_LIFECYCLE_ACTIVE,
            )
            for out in outputs:
                graph.add_connection(
                    node_id,
                    out,
                    weight=float(rng.normal(0.0, 0.060)),
                    gate=-1.15,
                    kind=EDGE_GATING,
                    created_by="reasoning_seed_v18",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
        node_items = list(controller_nodes.items())
        for index, (_left_name, left) in enumerate(node_items):
            for _right_name, right in node_items[index + 1:]:
                graph.add_connection(
                    left,
                    right,
                    weight=float(rng.normal(0.0, 0.012)),
                    gate=-4.5,
                    kind=EDGE_LATERAL,
                    created_by="reasoning_seed_v18",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
                graph.add_connection(
                    right,
                    left,
                    weight=float(rng.normal(0.0, 0.012)),
                    gate=-4.5,
                    kind=EDGE_LATERAL,
                    created_by="reasoning_seed_v18",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
        graph.assert_invariants(output_size=max(1, int(output_size)))
        return cls(
            graph=graph,
            input_size=max(1, int(input_size)),
            output_size=max(1, int(output_size)),
            controllers=controller_names,
            schema_version=REASONING_SCHEMA_VERSION,
        )

    @property
    def controller_input_size(self) -> int:
        return len(self.controllers)

    @property
    def total_input_size(self) -> int:
        return int(self.input_size) + self.controller_input_size

    def _controller_input_id(self, controller: str) -> str:
        return f"reasoning:controller:{controller}"

    def _controller_core_id(self, controller: str) -> str:
        return f"reasoning:{controller}:core"

    def _context_input_ids(self) -> list[str]:
        return [f"reasoning:input:{index}" for index in range(int(self.input_size))]

    def _ensure_controller_routing(self) -> None:
        """Migrate V15/V17 reasoning graphs in-place without losing learning."""
        self.controllers = tuple(dict.fromkeys(str(item) for item in (self.controllers or REASONING_CONTROLLERS) if str(item))) or REASONING_CONTROLLERS
        graph = self.graph
        for controller in self.controllers:
            input_id = self._controller_input_id(controller)
            if input_id not in graph.neurons:
                graph.add_neuron(
                    REGION_INPUT,
                    neuron_id=input_id,
                    module_id="reasoning:controller",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
            core_id = self._controller_core_id(controller)
            if core_id not in graph.neurons:
                region = CONTROLLER_REGIONS.get(controller, REGION_META)
                graph.add_neuron(
                    region,
                    neuron_id=core_id,
                    module_id=f"reasoning:{controller}",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                    plasticity=1.2,
                )
                for output_id in list(graph.regions[REGION_OUTPUT].neuron_ids):
                    graph.add_connection(
                        core_id,
                        output_id,
                        weight=0.015,
                        gate=-1.5,
                        kind=EDGE_GATING,
                        created_by="reasoning_migration_v18",
                        lifecycle=NEURON_LIFECYCLE_ACTIVE,
                    )
            route_exists = any(
                edge.src == input_id and edge.dst == core_id
                for edge in graph.edges.values()
            )
            if not route_exists:
                graph.add_connection(
                    input_id,
                    core_id,
                    weight=0.60,
                    gate=0.35,
                    kind=EDGE_GATING,
                    created_by="reasoning_migration_v18",
                    lifecycle=NEURON_LIFECYCLE_ACTIVE,
                )
        # Old saved systems can have input_size absent or malformed.  Preserve
        # existing semantic context width where possible and never count newly
        # added controller identity nodes as context dimensions.
        self.schema_version = REASONING_SCHEMA_VERSION

    def ensure_compatible(self) -> None:
        graph = getattr(self, "graph", None)
        if isinstance(graph, NeuralGraph):
            regions = getattr(graph, "regions", {}) or {}
            signature = (
                REASONING_SCHEMA_VERSION,
                int(getattr(self, "input_size", 32) or 32),
                int(getattr(self, "output_size", 16) or 16),
                tuple(str(item) for item in getattr(self, "controllers", REASONING_CONTROLLERS) if str(item)),
                len(getattr(graph, "neurons", {}) or {}),
                len(getattr(graph, "edges", {}) or {}),
                tuple(
                    (str(name), len(getattr(region, "neuron_ids", ()) or ()))
                    for name, region in sorted(regions.items(), key=lambda item: str(item[0]))
                ),
            )
            if getattr(self, "_compatibility_signature", None) == signature:
                return
        if not isinstance(getattr(self, "graph", None), NeuralGraph):
            seeded = self.seed(
                input_size=int(getattr(self, "input_size", 32) or 32),
                output_size=int(getattr(self, "output_size", 16) or 16),
            )
            self.graph = seeded.graph
            self.controllers = seeded.controllers
        self.input_size = max(1, int(getattr(self, "input_size", 32) or 32))
        self.output_size = max(1, int(getattr(self, "output_size", 16) or 16))
        self.controllers = tuple(dict.fromkeys(str(item) for item in getattr(self, "controllers", REASONING_CONTROLLERS) if str(item))) or REASONING_CONTROLLERS
        self.influence = float(np.clip(getattr(self, "influence", 0.08), 0.0, 0.35))
        self.plasticity = float(np.clip(getattr(self, "plasticity", 0.01), 0.0001, 0.25))
        self.age = int(getattr(self, "age", 0))
        self.last_controller = str(getattr(self, "last_controller", "") or "")
        self.last_active_edge_ids = tuple(getattr(self, "last_active_edge_ids", ()) or ())
        self.last_active_neuron_ids = tuple(getattr(self, "last_active_neuron_ids", ()) or ())
        traces = getattr(self, "last_controller_traces", None)
        self.last_controller_traces = dict(traces) if isinstance(traces, Mapping) else {}
        credits = getattr(self, "last_controller_credit", None)
        self.last_controller_credit = dict(credits) if isinstance(credits, Mapping) else {}
        self._ensure_controller_routing()
        self.graph.assert_invariants(output_size=self.output_size)
        regions = getattr(self.graph, "regions", {}) or {}
        self._compatibility_signature = (
            REASONING_SCHEMA_VERSION,
            int(self.input_size),
            int(self.output_size),
            tuple(str(item) for item in self.controllers),
            len(getattr(self.graph, "neurons", {}) or {}),
            len(getattr(self.graph, "edges", {}) or {}),
            tuple(
                (str(name), len(getattr(region, "neuron_ids", ()) or ()))
                for name, region in sorted(regions.items(), key=lambda item: str(item[0]))
            ),
        )

    def copy(self) -> "ReasoningGraphSystem":
        self.ensure_compatible()
        return copy.deepcopy(self)

    def begin_decision(self) -> None:
        """Discard stale controller traces before a new policy decision."""
        self.ensure_compatible()
        self.last_controller_traces = {}
        self.last_controller_credit = {}
        self.last_controller = ""
        self.last_active_edge_ids = ()
        self.last_active_neuron_ids = ()

    def _controller_input_vector(self, controller: str, context: Any) -> tuple[np.ndarray, np.ndarray]:
        name = str(controller)
        if name not in self.controllers:
            name = "utility" if "utility" in self.controllers else self.controllers[0]
        semantic = _project_context(context, self.input_size)
        role = np.zeros(self.controller_input_size, dtype=float)
        role[self.controllers.index(name)] = 1.0
        return np.concatenate([semantic, role]), semantic

    def _capture_controller_trace(
        self,
        controller: str,
        *,
        source_context: np.ndarray,
        projected_context: np.ndarray,
        input_vector: np.ndarray,
        output: np.ndarray,
    ) -> None:
        trace = self.graph.last_activation_trace
        if trace is None:
            return
        node_values = {
            node_id: {
                "activation": float(getattr(self.graph.neurons[node_id], "activation", 0.0)),
                "state": float(getattr(self.graph.neurons[node_id], "state", 0.0)),
            }
            for node_id in self.graph.neurons
        }
        raw_inputs = {
            node_id: float(input_vector[index])
            for index, node_id in enumerate(list(self.graph.regions[REGION_INPUT].neuron_ids))
            if index < input_vector.size
        }
        self.last_controller_traces[str(controller)] = {
            "controller": str(controller),
            "source_context_size": int(source_context.size),
            "projected_context": [float(item) for item in projected_context.tolist()],
            "raw_inputs": raw_inputs,
            "output": [float(item) for item in np.asarray(output, dtype=float).reshape(-1).tolist()],
            "active_neuron_ids": tuple(str(item) for item in trace.active_neuron_ids),
            "active_edge_ids": tuple(str(item) for item in trace.active_edge_ids),
            "edge_gates": {str(key): float(value) for key, value in trace.edge_gates.items()},
            "edge_contributions": {str(key): float(value) for key, value in trace.edge_contributions.items()},
            "region_activity": {str(key): float(value) for key, value in trace.region_activity.items()},
            "node_values": node_values,
            "active_edge_count": int(trace.active_edge_count),
            "traced_edge_count": int(trace.traced_edge_count),
        }

    def enrich_context(self, controller: str, context: np.ndarray, *, target_size: int | None = None) -> np.ndarray:
        self.ensure_compatible()
        source = _finite_vector(context)
        graph_input, projected = self._controller_input_vector(str(controller), source)
        output = np.asarray(self.graph.activate(graph_input, steps=2, return_trace=False), dtype=float).reshape(-1)
        trace = self.graph.last_activation_trace
        if trace is not None:
            self.last_active_edge_ids = tuple(trace.active_edge_ids)
            self.last_active_neuron_ids = tuple(trace.active_neuron_ids)
        self.last_controller = str(controller)
        self.age += 1
        self._capture_controller_trace(
            str(controller),
            source_context=source,
            projected_context=projected,
            input_vector=graph_input,
            output=output,
        )
        target = int(target_size or source.size or self.input_size)
        base = _resample_to_size(source, target)
        correction = _resample_to_size(output, target)
        enriched = base + self.influence * correction
        return np.clip(enriched, -8.0, 8.0)

    def controller_output(self, controller: str, *, size: int | None = None) -> np.ndarray:
        self.ensure_compatible()
        record = self.last_controller_traces.get(str(controller), {})
        values = _finite_vector(record.get("output", [])) if isinstance(record, Mapping) else np.zeros(0, dtype=float)
        target = int(size or values.size or self.output_size)
        return _resample_to_size(values, target)

    def controller_trace(self, controller: str) -> dict[str, Any]:
        self.ensure_compatible()
        record = self.last_controller_traces.get(str(controller), {})
        return copy.deepcopy(record) if isinstance(record, Mapping) else {}

    def telemetry_trace(self) -> dict[str, Any]:
        """Return immutable aggregate evidence for one decision's reasoning calls."""
        self.ensure_compatible()
        controller_records = {
            str(name): copy.deepcopy(record)
            for name, record in self.last_controller_traces.items()
            if isinstance(record, Mapping)
        }
        active_nodes: set[str] = set()
        active_edges: set[str] = set()
        edge_gates: dict[str, float] = {}
        edge_contrib: dict[str, float] = {}
        region_values: dict[str, list[float]] = {}
        node_values: dict[str, dict[str, float]] = {}
        input_values: dict[str, dict[str, float]] = {}
        for controller, record in controller_records.items():
            active_nodes.update(str(item) for item in record.get("active_neuron_ids", ()) or ())
            active_edges.update(str(item) for item in record.get("active_edge_ids", ()) or ())
            edge_gates.update({str(key): float(value) for key, value in dict(record.get("edge_gates", {}) or {}).items()})
            for edge_id, value in dict(record.get("edge_contributions", {}) or {}).items():
                edge_contrib[str(edge_id)] = edge_contrib.get(str(edge_id), 0.0) + float(value)
            for region, value in dict(record.get("region_activity", {}) or {}).items():
                region_values.setdefault(str(region), []).append(float(value))
            for node_id, values in dict(record.get("node_values", {}) or {}).items():
                raw = dict(values or {})
                existing = node_values.get(str(node_id), {"activation": 0.0, "state": 0.0})
                if abs(float(raw.get("activation", 0.0))) >= abs(float(existing.get("activation", 0.0))):
                    node_values[str(node_id)] = {
                        "activation": float(raw.get("activation", 0.0)),
                        "state": float(raw.get("state", 0.0)),
                    }
            for node_id, value in dict(record.get("raw_inputs", {}) or {}).items():
                input_values.setdefault(str(node_id), {})[controller] = float(value)
        return {
            "graph_id": "reasoning_graph",
            "graph_label": "reasoning_graph",
            "active_neuron_ids": tuple(sorted(active_nodes)),
            "active_edge_ids": tuple(sorted(active_edges)),
            "edge_gates": edge_gates,
            "edge_contributions": edge_contrib,
            "region_activity": {
                region: float(np.mean(values)) if values else 0.0
                for region, values in region_values.items()
            },
            "node_values": node_values,
            "input_values": input_values,
            "controllers": controller_records,
            "controller_credit": copy.deepcopy(self.last_controller_credit),
            "active_node_count": len(active_nodes),
            "active_edge_count": len(active_edges),
            "schema_version": REASONING_SCHEMA_VERSION,
        }

    def _credit_trace(self, controller: str, credit: float, *, strength: float) -> dict[str, float]:
        record = self.last_controller_traces.get(str(controller), {})
        if not isinstance(record, Mapping):
            return {"updated_nodes": 0.0, "updated_edges": 0.0, "credit": 0.0}
        bounded_credit = float(np.clip(credit, -2.0, 2.0))
        lr = float(np.clip(self.plasticity * max(0.1, abs(strength)), 0.00005, 0.025))
        active_nodes = set(str(item) for item in record.get("active_neuron_ids", ()) or ())
        active_edges = set(str(item) for item in record.get("active_edge_ids", ()) or ())
        core_id = self._controller_core_id(str(controller))
        active_nodes.add(core_id)
        node_values = dict(record.get("node_values", {}) or {})
        edge_contrib = dict(record.get("edge_contributions", {}) or {})
        updated_nodes = 0
        updated_edges = 0
        for node_id in active_nodes:
            node = self.graph.neurons.get(node_id)
            if node is None:
                continue
            activation = abs(float(dict(node_values.get(node_id, {}) or {}).get("activation", getattr(node, "activation", 0.0))))
            local = bounded_credit * max(0.10, activation)
            node.utility = float(np.clip(0.97 * node.utility + 0.03 * local, -5.0, 5.0))
            node.plasticity = float(np.clip(0.995 * node.plasticity + 0.005 * (0.25 + abs(local)), 0.05, 3.0))
            updated_nodes += 1
        for edge_id in active_edges:
            edge = self.graph.edges.get(edge_id)
            if edge is None:
                continue
            contribution = float(edge_contrib.get(edge_id, 0.0))
            evidence = max(0.05, min(1.0, abs(contribution)))
            signed_path = 1.0 if contribution >= 0.0 else -1.0
            local = bounded_credit * evidence
            edge.utility = float(np.clip(0.97 * edge.utility + 0.03 * local, -5.0, 5.0))
            # Positive controller credit reinforces the exact active path.  A
            # negative outcome is recorded in utility, but only risk/utility
            # controller paths are allowed to reduce their gate; action-head
            # paths stay available for replacement learning rather than being
            # globally suppressed after one losing trade.
            gate_credit = local
            if bounded_credit < 0.0 and str(controller) == "action_head":
                gate_credit = 0.0
            edge.gate = float(np.clip(edge.gate + lr * gate_credit, -8.0, 2.0))
            edge.weight = float(np.clip(edge.weight + 0.20 * lr * local * signed_path, -2.5, 2.5))
            edge.plasticity = float(np.clip(0.995 * edge.plasticity + 0.005 * (0.25 + abs(local)), 0.05, 3.0))
            updated_edges += 1
        self.last_controller_credit[str(controller)] = {
            "credit": bounded_credit,
            "strength": float(strength),
            "updated_nodes": float(updated_nodes),
            "updated_edges": float(updated_edges),
        }
        return dict(self.last_controller_credit[str(controller)])

    def observe(self, controller: str, reward: float, *, strength: float = 1.0) -> None:
        self.ensure_compatible()
        self._credit_trace(str(controller), float(reward), strength=float(strength))
        bounded = float(np.clip(reward, -2.0, 2.0))
        self.plasticity = float(np.clip(0.995 * self.plasticity + 0.005 * abs(bounded), 0.0001, 0.25))
        self.graph.assert_invariants(output_size=self.output_size)

    def observe_lifecycle(
        self,
        *,
        realized_r: float,
        advantage: float,
        close_reason: str = "",
        strength: float = 1.0,
        controller_traces: Mapping[str, Mapping[str, Any]] | None = None,
        controller_rewards: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Route an MT5 final outcome to the exact controller paths used at entry.

        ``controller_traces`` is deliberately accepted as immutable decision-time
        evidence.  A position may remain open through many later decisions; using
        ``last_controller_traces`` at settlement would otherwise credit whatever
        reasoning activity happened most recently rather than the graph pathway
        that selected the settled order.

        A losing entry never blindly suppresses the action-head pathway.  It
        instead increases credit for the risk path and records negative utility
        for action/value evidence, so a later similar context can prefer a
        replacement rather than simply becoming less active everywhere.
        """
        self.ensure_compatible()
        supplied = controller_traces if isinstance(controller_traces, Mapping) else None
        previous = self.last_controller_traces
        if supplied is not None:
            self.last_controller_traces = {
                str(key): copy.deepcopy(dict(value or {}))
                for key, value in supplied.items()
                if isinstance(value, Mapping)
            }
        try:
            if not self.last_controller_traces:
                return {"updated_nodes": 0, "updated_edges": 0, "controllers": {}, "close_reason": str(close_reason)}
            outcome = float(np.clip(realized_r, -3.0, 3.0))
            adv = float(np.clip(advantage, -3.0, 3.0))
            loss_pressure = max(0.0, -adv)
            rewards = {
                "action_head": max(0.0, adv),
                "risk": loss_pressure if loss_pressure > 0.0 else -0.20 * max(0.0, adv),
                "utility": adv,
                "curiosity": 0.20 * max(0.0, adv) - 0.05 * loss_pressure,
                "development": 0.25 * abs(adv),
            }
            # V19 causal lifecycle credit can refine controller attribution
            # only when an entry snapshot and outcome evidence exist.  Missing
            # keys retain the conservative V18 defaults; this keeps older
            # journals/replays backward-compatible.
            if isinstance(controller_rewards, Mapping):
                for name, value in controller_rewards.items():
                    if str(name) in self.controllers:
                        rewards[str(name)] = float(np.clip(value, -2.0, 2.0))
            reports: dict[str, dict[str, float]] = {}
            total_nodes = 0
            total_edges = 0
            for controller in self.controllers:
                if controller not in self.last_controller_traces:
                    continue
                report = self._credit_trace(controller, rewards.get(controller, adv), strength=strength)
                reports[controller] = report
                total_nodes += int(report.get("updated_nodes", 0.0))
                total_edges += int(report.get("updated_edges", 0.0))
            self.plasticity = float(np.clip(0.992 * self.plasticity + 0.008 * abs(adv), 0.0001, 0.25))
            self.graph.assert_invariants(output_size=self.output_size)
            return {
                "updated_nodes": int(total_nodes),
                "updated_edges": int(total_edges),
                "controllers": reports,
                "realized_r": outcome,
                "advantage": adv,
                "close_reason": str(close_reason),
                "decision_trace_source": "immutable_entry_trace" if supplied is not None else "latest_runtime_trace",
                "replacement_preserves_action_head": bool(loss_pressure > 0.0 and not (isinstance(controller_rewards, Mapping) and "action_head" in controller_rewards)),
                "controller_rewards_source": "causal_credit_plan" if isinstance(controller_rewards, Mapping) else "default_lifecycle_rule",
            }
        finally:
            if supplied is not None:
                self.last_controller_traces = previous

    def summary(self) -> dict[str, float | str]:
        self.ensure_compatible()
        return {
            "neurons": float(len(self.graph.neurons)),
            "edges": float(len(self.graph.edges)),
            "age": float(self.age),
            "plasticity": float(self.plasticity),
            "influence": float(self.influence),
            "last_controller": self.last_controller,
            "schema_version": float(REASONING_SCHEMA_VERSION),
        }


def _resample_to_size(values: Any, size: int) -> np.ndarray:
    target = max(1, int(size))
    source = _finite_vector(values)
    if source.size == 0:
        return np.zeros(target, dtype=float)
    if source.size == target:
        return np.clip(source.copy(), -8.0, 8.0)
    if source.size == 1:
        return np.full(target, float(source[0]), dtype=float)
    return np.clip(
        np.interp(np.linspace(0.0, float(source.size - 1), target), np.arange(source.size, dtype=float), source),
        -8.0,
        8.0,
    )
