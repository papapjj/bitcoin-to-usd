from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


NEURON_LIFECYCLE_CANDIDATE = "candidate"
NEURON_LIFECYCLE_ACTIVE = "active"
NEURON_LIFECYCLE_CONSOLIDATED = "consolidated"
NEURON_LIFECYCLE_PRUNING = "pruning"

VALID_NEURON_LIFECYCLES = frozenset({
    NEURON_LIFECYCLE_CANDIDATE,
    NEURON_LIFECYCLE_ACTIVE,
    NEURON_LIFECYCLE_CONSOLIDATED,
    NEURON_LIFECYCLE_PRUNING,
})

REGION_INPUT = "input"
REGION_ENCODER = "encoder"
REGION_MEMORY = "memory"
REGION_HYPOTHESIS = "hypothesis"
REGION_PREDICTOR = "predictor"
REGION_CRITIC = "critic"
REGION_SELECTOR = "selector"
REGION_META = "meta"
REGION_OUTPUT = "output"

VALID_REGIONS = frozenset({
    REGION_INPUT,
    REGION_ENCODER,
    REGION_MEMORY,
    REGION_HYPOTHESIS,
    REGION_PREDICTOR,
    REGION_CRITIC,
    REGION_SELECTOR,
    REGION_META,
    REGION_OUTPUT,
})

EDGE_FORWARD = "forward"
EDGE_RECURRENT = "recurrent"
EDGE_LATERAL = "lateral"
EDGE_MEMORY = "memory"
EDGE_PREDICTIVE = "predictive"
EDGE_CRITIC = "critic"
EDGE_SELECTOR = "selector"
EDGE_GATING = "gating"
EDGE_PLASTIC = "plastic"
EDGE_META = "meta"

VALID_EDGE_KINDS = frozenset({
    EDGE_FORWARD,
    EDGE_RECURRENT,
    EDGE_LATERAL,
    EDGE_MEMORY,
    EDGE_PREDICTIVE,
    EDGE_CRITIC,
    EDGE_SELECTOR,
    EDGE_GATING,
    EDGE_PLASTIC,
    EDGE_META,
})

DEFAULT_ALLOWED_M1_MODULE_TARGETS = frozenset({
    REGION_ENCODER,
    REGION_MEMORY,
    REGION_HYPOTHESIS,
    REGION_PREDICTOR,
    REGION_CRITIC,
    REGION_SELECTOR,
    REGION_META,
    REGION_OUTPUT,
})

DEFAULT_ALLOWED_MODULE_M1_TARGETS = frozenset({
    REGION_ENCODER,
    REGION_MEMORY,
    REGION_HYPOTHESIS,
    REGION_PREDICTOR,
    REGION_CRITIC,
    REGION_SELECTOR,
    REGION_OUTPUT,
    REGION_META,
})


@dataclass
class NeuronNode:
    id: str
    region: str
    bias: float = 0.0
    activation: float = 0.0
    state: float = 0.0
    utility: float = 0.0
    plasticity: float = 1.0
    eligibility: float = 0.0
    age: int = 0
    lifecycle: str = NEURON_LIFECYCLE_CANDIDATE
    module_id: str | None = None
    # Stage/role fields make an actual specialist circuit inspectable instead
    # of disguising a single candidate neuron as a module.
    module_stage: str = ""
    module_role: str = ""
    # A bounded, explicit forgetting control lets memory/recurrent regions use
    # short-term plasticity without retaining an unconstrained hidden state.
    forget_gate: float = 0.0
    # Trial-scoped micro-growth records compact local-regime evidence. A
    # newborn survives by succeeding repeatedly in the same or a nearby market
    # context; unrelated regimes neither prove nor disprove the specialization.
    birth_trial_id: str | None = None
    survival_contexts: set[str] = field(default_factory=set)
    positive_supports: int = 0
    outcome_observations: int = 0
    fade: float = 1.0
    born_outcome_step: int = 0
    last_active_outcome: int = 0
    home_context_key: str = ""
    context_prototype: np.ndarray | None = field(default=None, repr=False)
    similar_context_observations: int = 0
    similar_context_supports: int = 0
    last_context_similarity: float = 0.0
    context_similarity_ema: float = 0.0

    def __post_init__(self) -> None:
        if self.region not in VALID_REGIONS:
            raise ValueError(f"unknown neuron region: {self.region}")
        if self.lifecycle not in VALID_NEURON_LIFECYCLES:
            raise ValueError(f"unknown neuron lifecycle: {self.lifecycle}")


@dataclass
class ConnectionEdge:
    id: str
    src: str
    dst: str
    weight: float
    gate: float = -2.0
    kind: str = EDGE_FORWARD
    utility: float = 0.0
    eligibility: float = 0.0
    plasticity: float = 1.0
    age: int = 0
    enabled: bool = True
    src_module_id: str | None = None
    dst_module_id: str | None = None
    created_by: str | None = None
    need_reason: str | None = None
    candidate_id: str | None = None
    # Role of the specialist circuit that created this edge, if any.
    module_role: str = ""
    lifecycle: str = NEURON_LIFECYCLE_CANDIDATE
    # Fast plasticity uses this bounded state only for memory/recurrent edges.
    short_term_state: float = 0.0
    consolidation: float = 0.0
    birth_trial_id: str | None = None
    survival_contexts: set[str] = field(default_factory=set)
    positive_supports: int = 0
    fade: float = 1.0
    born_outcome_step: int = 0
    home_context_key: str = ""
    context_prototype: np.ndarray | None = field(default=None, repr=False)
    similar_context_observations: int = 0
    similar_context_supports: int = 0
    last_context_similarity: float = 0.0
    context_similarity_ema: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in VALID_EDGE_KINDS:
            raise ValueError(f"unknown edge kind: {self.kind}")
        if self.lifecycle not in VALID_NEURON_LIFECYCLES:
            raise ValueError(f"unknown edge lifecycle: {self.lifecycle}")

    @property
    def edge_kind(self) -> str:
        return self.kind


@dataclass
class NeuralRegion:
    name: str
    neuron_ids: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.name not in VALID_REGIONS:
            raise ValueError(f"unknown neural region: {self.name}")


@dataclass(frozen=True)
class NeuralGraphBudget:
    # V7 candidate budgets are hard integrity limits.  A value of zero is
    # retained only for loading legacy graphs before their run profile assigns
    # an explicit bounded budget.
    max_neurons: int = 0
    # Zero remains a legacy compatibility sentinel; production V7 profiles
    # install explicit positive values before candidate growth begins.
    max_connections: int = 0
    max_recurrent_edges: int = 0
    max_lateral_edges: int = 0
    max_cross_module_edges: int = 0
    max_m1_module_edges: int = 0
    max_cross_module_recurrent_edges: int = 0
    max_cross_module_lateral_edges: int = 0


@dataclass
class NeuralGraphInvariantReport:
    ok: bool
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphActivationTrace:
    """Decision-time attribution evidence for a graph activation.

    The trace holds identifiers and scalar summaries only.  It deliberately
    does not retain mutable neuron/edge objects, so it remains safe to journal,
    replay, pickle, and compare after a candidate graph is rolled back.
    """

    sequence: int
    active_neuron_ids: tuple[str, ...]
    active_edge_ids: tuple[str, ...]
    edge_gates: dict[str, float]
    edge_contributions: dict[str, float]
    region_activity: dict[str, float]
    context_regions: tuple[str, ...] = ()
    # The graph may contain thousands of active edges.  Decision traces retain
    # the strongest causal paths plus candidate edges, not every dense edge.
    # Counts preserve observability without forcing full-edge journaling.
    active_edge_count: int = 0
    traced_edge_count: int = 0


class NeuralGraph:
    """Bounded graph of neurons and connections used by the graph M1 backend.

    Structural operations are intentionally explicit.  Runtime activation
    never changes topology, and callers can retain ``last_activation_trace``
    for candidate-only credit assignment.
    """

    def __init__(
        self,
        *,
        budget: NeuralGraphBudget | None = None,
        allowed_m1_module_targets: Iterable[str] | None = None,
        allowed_module_m1_targets: Iterable[str] | None = None,
        recurrent_decay: float = 0.85,
        activation_clip: float = 20.0,
        activity_epsilon: float = 1e-8,
        max_cross_module_initial_weight: float = 0.25,
        max_action_bias_weight: float = 0.15,
        max_action_bias_gate: float = -0.25,
        trace_max_edges: int = 384,
        trace_candidate_edges: int = 32,
        trace_min_edges_per_route: int = 12,
    ) -> None:
        self.neurons: dict[str, NeuronNode] = {}
        self.edges: dict[str, ConnectionEdge] = {}
        # Module-level lifecycle registry. Records are plain dictionaries to
        # keep persisted policy artifacts forwards/backwards compatible.
        self.specialist_modules: dict[str, dict[str, Any]] = {}
        self.regions: dict[str, NeuralRegion] = {name: NeuralRegion(name) for name in VALID_REGIONS}
        self.budget = budget or NeuralGraphBudget()
        self.allowed_m1_module_targets = frozenset(allowed_m1_module_targets or DEFAULT_ALLOWED_M1_MODULE_TARGETS)
        self.allowed_module_m1_targets = frozenset(allowed_module_m1_targets or DEFAULT_ALLOWED_MODULE_M1_TARGETS)
        self.recurrent_decay = float(np.clip(recurrent_decay, 0.0, 1.0))
        self.activation_clip = float(max(1.0, activation_clip))
        self.activity_epsilon = float(max(0.0, activity_epsilon))
        self.max_cross_module_initial_weight = float(max(0.0, max_cross_module_initial_weight))
        self.max_action_bias_weight = float(max(0.0, max_action_bias_weight))
        self.max_action_bias_gate = float(max(-8.0, min(8.0, max_action_bias_gate)))
        # Sparse decision traces cap attribution I/O.  A route-aware reserve
        # prevents the dense sensory fan-in (input -> encoder) from consuming
        # the whole trace and hiding memory, predictor, critic, selector and
        # output paths in the live viewer. Candidate-only paths retain their
        # own small reserve so weak warm-up specialist edges can receive credit.
        self.trace_max_edges = max(1, int(trace_max_edges))
        self.trace_candidate_edges = max(0, int(trace_candidate_edges))
        self.trace_min_edges_per_route = max(0, int(trace_min_edges_per_route))
        self._compiled_topology: dict[str, object] | None = None
        self._next_neuron_index = 0
        self._next_edge_index = 0
        self._activation_sequence = 0
        self.last_activation_trace: GraphActivationTrace | None = None

    def __setstate__(self, state: dict[str, object]) -> None:
        """Load older pickles without turning an outdated artifact into a crash."""
        self.__dict__.update(state)
        self.budget = getattr(self, "budget", NeuralGraphBudget())
        self.allowed_m1_module_targets = frozenset(getattr(self, "allowed_m1_module_targets", DEFAULT_ALLOWED_M1_MODULE_TARGETS))
        self.allowed_module_m1_targets = frozenset(getattr(self, "allowed_module_m1_targets", DEFAULT_ALLOWED_MODULE_M1_TARGETS))
        self.recurrent_decay = float(np.clip(getattr(self, "recurrent_decay", 0.85), 0.0, 1.0))
        self.activation_clip = float(max(1.0, getattr(self, "activation_clip", 20.0)))
        self.activity_epsilon = float(max(0.0, getattr(self, "activity_epsilon", 1e-8)))
        self.max_cross_module_initial_weight = float(max(0.0, getattr(self, "max_cross_module_initial_weight", 0.25)))
        self.max_action_bias_weight = float(max(0.0, getattr(self, "max_action_bias_weight", 0.15)))
        self.max_action_bias_gate = float(getattr(self, "max_action_bias_gate", -0.25))
        self.trace_max_edges = max(1, int(getattr(self, "trace_max_edges", 384)))
        self.trace_candidate_edges = max(0, int(getattr(self, "trace_candidate_edges", 32)))
        self.trace_min_edges_per_route = max(0, int(getattr(self, "trace_min_edges_per_route", 12)))
        # Compiled index maps are a runtime cache; rebuild after unpickling so
        # they cannot retain stale references from older artifacts.
        self._compiled_topology = None
        self._activation_sequence = int(getattr(self, "_activation_sequence", 0))
        self.last_activation_trace = getattr(self, "last_activation_trace", None)
        raw_modules = getattr(self, "specialist_modules", {})
        self.specialist_modules = dict(raw_modules) if isinstance(raw_modules, Mapping) else {}
        for node in getattr(self, "neurons", {}).values():
            if not hasattr(node, "forget_gate"):
                node.forget_gate = 0.0
            if not hasattr(node, "birth_trial_id"):
                node.birth_trial_id = None
            if not hasattr(node, "survival_contexts") or not isinstance(node.survival_contexts, set):
                node.survival_contexts = set(getattr(node, "survival_contexts", ()) or ())
            for field_name, default in (
                ("positive_supports", 0), ("outcome_observations", 0), ("fade", 1.0),
                ("born_outcome_step", 0), ("last_active_outcome", 0),
                ("home_context_key", ""), ("similar_context_observations", 0),
                ("similar_context_supports", 0), ("last_context_similarity", 0.0),
                ("context_similarity_ema", 0.0), ("module_stage", ""), ("module_role", ""),
            ):
                if not hasattr(node, field_name):
                    setattr(node, field_name, default)
            prototype = getattr(node, "context_prototype", None)
            if prototype is None:
                node.context_prototype = None
            else:
                node.context_prototype = np.asarray(prototype, dtype=float).reshape(-1).copy()
        for edge in getattr(self, "edges", {}).values():
            if not hasattr(edge, "lifecycle"):
                edge.lifecycle = NEURON_LIFECYCLE_CANDIDATE
            if not hasattr(edge, "short_term_state"):
                edge.short_term_state = 0.0
            if not hasattr(edge, "consolidation"):
                edge.consolidation = 0.0
            if not hasattr(edge, "birth_trial_id"):
                edge.birth_trial_id = None
            if not hasattr(edge, "survival_contexts") or not isinstance(edge.survival_contexts, set):
                edge.survival_contexts = set(getattr(edge, "survival_contexts", ()) or ())
            for field_name, default in (
                ("positive_supports", 0), ("fade", 1.0), ("born_outcome_step", 0),
                ("home_context_key", ""), ("similar_context_observations", 0),
                ("similar_context_supports", 0), ("last_context_similarity", 0.0),
                ("context_similarity_ema", 0.0), ("module_role", ""),
            ):
                if not hasattr(edge, field_name):
                    setattr(edge, field_name, default)
            prototype = getattr(edge, "context_prototype", None)
            if prototype is None:
                edge.context_prototype = None
            else:
                edge.context_prototype = np.asarray(prototype, dtype=float).reshape(-1).copy()

    def copy(self) -> "NeuralGraph":
        cloned = copy.deepcopy(self)
        cloned._compiled_topology = None
        return cloned

    def _invalidate_topology_cache(self) -> None:
        self._compiled_topology = None

    def _compiled(self) -> dict[str, object]:
        """Return immutable index maps for vectorised sparse activation."""
        cached = self._compiled_topology
        if isinstance(cached, dict):
            return cached
        node_ids = tuple(self.neurons)
        index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        regions = {name: tuple(index[node_id] for node_id in self._region_ids(name) if node_id in index) for name in VALID_REGIONS}
        edge_ids = tuple(
            edge_id for edge_id, edge in self.edges.items()
            if edge.enabled and edge.src in index and edge.dst in index and self.neurons[edge.dst].region != REGION_INPUT
        )
        compiled = {
            "node_ids": node_ids,
            "index": index,
            "input_indices": np.asarray(regions[REGION_INPUT], dtype=int),
            "output_indices": np.asarray(regions[REGION_OUTPUT], dtype=int),
            "noninput_indices": np.asarray(tuple(idx for idx, node_id in enumerate(node_ids) if self.neurons[node_id].region != REGION_INPUT), dtype=int),
            "region_indices": {name: np.asarray(values, dtype=int) for name, values in regions.items()},
            "edge_ids": edge_ids,
            "edge_src": np.asarray([index[self.edges[edge_id].src] for edge_id in edge_ids], dtype=int),
            "edge_dst": np.asarray([index[self.edges[edge_id].dst] for edge_id in edge_ids], dtype=int),
        }
        self._compiled_topology = compiled
        return compiled

    def add_neuron(
        self,
        region: str,
        *,
        neuron_id: str | None = None,
        bias: float = 0.0,
        module_id: str | None = None,
        lifecycle: str = NEURON_LIFECYCLE_CANDIDATE,
        plasticity: float = 1.0,
        forget_gate: float = 0.0,
    ) -> str:
        if self.budget.max_neurons > 0 and len(self.neurons) >= self.budget.max_neurons:
            raise ValueError("neuron budget exceeded")
        if neuron_id is None:
            neuron_id = f"n{self._next_neuron_index}"
            self._next_neuron_index += 1
        if neuron_id in self.neurons:
            raise ValueError(f"duplicate neuron id: {neuron_id}")
        neuron = NeuronNode(
            id=neuron_id,
            region=region,
            bias=float(bias),
            module_id=module_id,
            lifecycle=lifecycle,
            plasticity=float(plasticity),
            forget_gate=float(np.clip(forget_gate, -8.0, 8.0)),
        )
        self.neurons[neuron_id] = neuron
        self.regions[region].neuron_ids.append(neuron_id)
        self._invalidate_topology_cache()
        return neuron_id

    def add_connection(
        self,
        src: str,
        dst: str,
        *,
        edge_id: str | None = None,
        weight: float = 0.0,
        gate: float = -2.0,
        kind: str = EDGE_FORWARD,
        created_by: str | None = None,
        need_reason: str | None = None,
        candidate_id: str | None = None,
        lifecycle: str = NEURON_LIFECYCLE_CANDIDATE,
    ) -> str:
        if self.budget.max_connections > 0 and len(self.edges) >= self.budget.max_connections:
            raise ValueError("connection budget exceeded")
        if src not in self.neurons:
            raise ValueError(f"missing source neuron: {src}")
        if dst not in self.neurons:
            raise ValueError(f"missing destination neuron: {dst}")
        if edge_id is None:
            edge_id = f"e{self._next_edge_index}"
            self._next_edge_index += 1
        if edge_id in self.edges:
            raise ValueError(f"duplicate edge id: {edge_id}")
        src_neuron = self.neurons[src]
        dst_neuron = self.neurons[dst]
        edge = ConnectionEdge(
            id=edge_id,
            src=src,
            dst=dst,
            weight=float(weight),
            gate=float(gate),
            kind=kind,
            src_module_id=src_neuron.module_id,
            dst_module_id=dst_neuron.module_id,
            created_by=created_by,
            need_reason=need_reason,
            candidate_id=candidate_id,
            lifecycle=lifecycle,
        )
        self._validate_new_edge(edge)
        self.edges[edge_id] = edge
        self._invalidate_topology_cache()
        return edge_id

    def rewire_connection(
        self,
        edge_id: str,
        *,
        src: str | None = None,
        dst: str | None = None,
        reason: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        """Safely change endpoints while retaining audit metadata and budgets."""
        if edge_id not in self.edges:
            raise KeyError(edge_id)
        previous = self.edges[edge_id]
        new_src = src or previous.src
        new_dst = dst or previous.dst
        if new_src not in self.neurons or new_dst not in self.neurons:
            raise ValueError("rewire endpoint missing from graph")
        replacement = copy.deepcopy(previous)
        replacement.src = new_src
        replacement.dst = new_dst
        replacement.src_module_id = self.neurons[new_src].module_id
        replacement.dst_module_id = self.neurons[new_dst].module_id
        replacement.need_reason = reason or previous.need_reason
        replacement.candidate_id = candidate_id or previous.candidate_id
        self.edges.pop(edge_id)
        try:
            self._validate_new_edge(replacement)
            self.edges[edge_id] = replacement
            self._invalidate_topology_cache()
        except Exception:
            self.edges[edge_id] = previous
            raise

    def split_region(
        self,
        region: str,
        *,
        source_module_id: str,
        child_module_id: str,
        similarity: float,
        gradient_conflict: float,
        max_similarity: float = 0.92,
        min_conflict: float = 0.15,
    ) -> list[str]:
        """Split ownership inside one role region only with a diversity guard.

        Regions are semantic roles, not arbitrary topology labels.  Splitting
        therefore re-homes low-utility candidate neurons into a new module id
        rather than silently adding an unregistered region name.
        """
        if region not in VALID_REGIONS:
            raise ValueError(f"unknown neural region: {region}")
        if similarity > max_similarity and gradient_conflict < min_conflict:
            raise ValueError("split guard rejected near-duplicate region")
        candidates = [
            node for node in self.neurons.values()
            if node.region == region and node.module_id == source_module_id
        ]
        if len(candidates) < 2:
            raise ValueError("not enough neurons to split")
        candidates.sort(key=lambda node: (node.utility, node.id))
        moved = candidates[len(candidates) // 2:]
        for node in moved:
            node.module_id = child_module_id
            node.lifecycle = NEURON_LIFECYCLE_CANDIDATE
        for edge in self.edges.values():
            edge.src_module_id = self.neurons[edge.src].module_id
            edge.dst_module_id = self.neurons[edge.dst].module_id
        self.assert_invariants()
        return [node.id for node in moved]

    def merge_region(
        self,
        region: str,
        *,
        source_module_id: str,
        target_module_id: str,
        similarity: float,
        min_similarity: float = 0.70,
    ) -> int:
        if region not in VALID_REGIONS:
            raise ValueError(f"unknown neural region: {region}")
        if similarity < min_similarity:
            raise ValueError("merge guard rejected dissimilar region")
        count = 0
        for node in self.neurons.values():
            if node.region == region and node.module_id == source_module_id:
                node.module_id = target_module_id
                count += 1
        for edge in self.edges.values():
            edge.src_module_id = self.neurons[edge.src].module_id
            edge.dst_module_id = self.neurons[edge.dst].module_id
        self.assert_invariants()
        return count

    def disable_connection(self, edge_id: str) -> None:
        self.edges[edge_id].enabled = False
        self.edges[edge_id].lifecycle = NEURON_LIFECYCLE_PRUNING
        self._invalidate_topology_cache()

    def remove_connection(self, edge_id: str) -> None:
        self.edges.pop(edge_id)
        self._invalidate_topology_cache()

    def remove_neuron(self, neuron_id: str) -> None:
        neuron = self.neurons.pop(neuron_id)
        self.regions[neuron.region].neuron_ids = [
            existing for existing in self.regions[neuron.region].neuron_ids if existing != neuron_id
        ]
        for edge_id in [edge.id for edge in self.edges.values() if edge.src == neuron_id or edge.dst == neuron_id]:
            self.remove_connection(edge_id)
        self._invalidate_topology_cache()

    def prune_disabled(self) -> int:
        disabled = [edge_id for edge_id, edge in self.edges.items() if not edge.enabled]
        for edge_id in disabled:
            self.remove_connection(edge_id)
        return len(disabled)

    def reset_state(self) -> None:
        for neuron in self.neurons.values():
            neuron.activation = 0.0
            neuron.state = 0.0
            neuron.eligibility = 0.0
        for edge in self.edges.values():
            edge.eligibility = 0.0
            edge.short_term_state = 0.0
        self.last_activation_trace = None

    def activate(
        self,
        inputs: np.ndarray,
        *,
        steps: int = 1,
        context_by_region: Mapping[str, Sequence[float] | np.ndarray | float] | None = None,
        return_trace: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, GraphActivationTrace]:
        """Activate the graph through vectorised sparse edge aggregation.

        The old implementation executed nested Python loops for every edge on
        every recurrence step and journaled all active edges.  This keeps the
        same bounded neuron/edge semantics while doing a single NumPy scatter
        accumulation per step and retaining only attributable paths in trace.
        """
        x = np.asarray(inputs, dtype=float).reshape(-1)
        compiled = self._compiled()
        node_ids = tuple(compiled["node_ids"])
        input_indices = np.asarray(compiled["input_indices"], dtype=int)
        output_indices = np.asarray(compiled["output_indices"], dtype=int)
        noninput_indices = np.asarray(compiled["noninput_indices"], dtype=int)
        region_indices = compiled["region_indices"]
        edge_ids = tuple(compiled["edge_ids"])
        src_idx = np.asarray(compiled["edge_src"], dtype=int)
        dst_idx = np.asarray(compiled["edge_dst"], dtype=int)
        if input_indices.size != x.size:
            raise ValueError(f"expected {input_indices.size} inputs, got {x.size}")

        node_count = len(node_ids)
        activations = np.asarray([self.neurons[node_id].activation for node_id in node_ids], dtype=float)
        states = np.asarray([self.neurons[node_id].state for node_id in node_ids], dtype=float)
        biases = np.asarray([self.neurons[node_id].bias for node_id in node_ids], dtype=float)
        node_eligibility = np.asarray([self.neurons[node_id].eligibility for node_id in node_ids], dtype=float)
        node_age = np.asarray([self.neurons[node_id].age for node_id in node_ids], dtype=int)
        forget = np.zeros(node_count, dtype=float)
        for region in (REGION_MEMORY, REGION_PREDICTOR):
            indices = np.asarray(region_indices.get(region, ()), dtype=int)
            if indices.size:
                forget[indices] = np.asarray([_sigmoid(getattr(self.neurons[node_ids[idx]], "forget_gate", 0.0)) for idx in indices], dtype=float)

        clipped_x = np.clip(np.nan_to_num(x, nan=0.0, posinf=self.activation_clip, neginf=-self.activation_clip), -self.activation_clip, self.activation_clip)
        activations[input_indices] = clipped_x
        states[input_indices] = clipped_x
        normalized_context = self._normalize_context(context_by_region)
        context_vector = np.zeros(node_count, dtype=float)
        for region, values in normalized_context.items():
            indices = np.asarray(region_indices.get(region, ()), dtype=int)
            if indices.size:
                context_vector[indices] += np.asarray(values, dtype=float)[:indices.size]

        edge_count = len(edge_ids)
        edge_refs = [self.edges[edge_id] for edge_id in edge_ids]
        if edge_count:
            weights = np.asarray([edge.weight for edge in edge_refs], dtype=float)
            gates = np.asarray([edge.gate for edge in edge_refs], dtype=float)
            edge_eligibility = np.asarray([edge.eligibility for edge in edge_refs], dtype=float)
            edge_age = np.asarray([edge.age for edge in edge_refs], dtype=int)
            short_term = np.asarray([edge.short_term_state for edge in edge_refs], dtype=float)
            recurrent_mask = np.asarray([edge.kind == EDGE_RECURRENT for edge in edge_refs], dtype=bool)
            short_term_mask = np.asarray([edge.kind in {EDGE_RECURRENT, EDGE_MEMORY} for edge in edge_refs], dtype=bool)
            lateral_mask = np.asarray([edge.kind == EDGE_LATERAL for edge in edge_refs], dtype=bool)
            gating_mask = np.asarray([edge.kind == EDGE_GATING for edge in edge_refs], dtype=bool)
            predictive_mask = np.asarray([edge.kind == EDGE_PREDICTIVE for edge in edge_refs], dtype=bool)
            critic_mask = np.asarray([edge.kind == EDGE_CRITIC for edge in edge_refs], dtype=bool)
            selector_mask = np.asarray([edge.kind == EDGE_SELECTOR for edge in edge_refs], dtype=bool)
            plastic_mask = np.asarray([edge.kind == EDGE_PLASTIC for edge in edge_refs], dtype=bool)
            meta_mask = np.asarray([edge.kind == EDGE_META for edge in edge_refs], dtype=bool)
        else:
            weights = gates = edge_eligibility = edge_age = short_term = np.zeros(0, dtype=float)
            recurrent_mask = short_term_mask = np.zeros(0, dtype=bool)
            lateral_mask = gating_mask = predictive_mask = critic_mask = np.zeros(0, dtype=bool)
            selector_mask = plastic_mask = meta_mask = np.zeros(0, dtype=bool)

        active_nodes = set(
            node_ids[idx] for idx in input_indices
            if abs(activations[idx]) > self.activity_epsilon
        )
        last_contribution = np.zeros(edge_count, dtype=float)
        last_gate_values = np.zeros(edge_count, dtype=float)
        steps = max(1, int(steps))
        for _ in range(steps):
            previous = activations.copy()
            accum = biases.copy()
            if noninput_indices.size:
                accum[noninput_indices] += context_vector[noninput_indices]
            if edge_count:
                src_activation = previous[src_idx].copy()
                if np.any(recurrent_mask):
                    src_activation[recurrent_mask] = (
                        self.recurrent_decay * states[src_idx[recurrent_mask]]
                        + (1.0 - self.recurrent_decay) * previous[src_idx[recurrent_mask]]
                    )
                if np.any(short_term_mask):
                    mask = short_term_mask
                    short_term[mask] = np.clip(0.92 * short_term[mask] + 0.08 * src_activation[mask], -1.0, 1.0)
                    src_activation[mask] *= 1.0 + 0.10 * short_term[mask]
                if np.any(lateral_mask):
                    src_activation[lateral_mask] = (
                        src_activation[lateral_mask] - previous[dst_idx[lateral_mask]]
                    )
                last_gate_values = 1.0 / (1.0 + np.exp(-np.clip(gates, -60.0, 60.0)))
                kind_effect = np.ones(edge_count, dtype=float)
                if np.any(gating_mask):
                    kind_effect[gating_mask] *= (
                        0.50 + 1.0 / (1.0 + np.exp(-2.0 * np.clip(src_activation[gating_mask], -30.0, 30.0)))
                    )
                if np.any(predictive_mask):
                    kind_effect[predictive_mask] *= (
                        1.0 + 0.25 * np.clip(np.abs(previous[dst_idx[predictive_mask]]), 0.0, 2.0)
                    )
                if np.any(critic_mask):
                    kind_effect[critic_mask] *= (
                        1.0 + 0.35 * np.clip(
                            np.abs(src_activation[critic_mask]) + np.abs(previous[dst_idx[critic_mask]]),
                            0.0,
                            2.0,
                        )
                    )
                if np.any(selector_mask):
                    kind_effect[selector_mask] *= (
                        0.75 + 0.50 * np.clip(
                            np.maximum(src_activation[selector_mask], 0.0)
                            + np.abs(previous[dst_idx[selector_mask]]),
                            0.0,
                            1.0,
                        )
                    )
                if np.any(plastic_mask):
                    kind_effect[plastic_mask] *= (
                        1.0 + 0.20 * np.clip(edge_eligibility[plastic_mask], 0.0, 5.0)
                    )
                if np.any(meta_mask):
                    kind_effect[meta_mask] *= (
                        1.0 + 0.15 * np.clip(np.abs(context_vector[dst_idx[meta_mask]]), 0.0, 2.0)
                    )
                last_contribution = last_gate_values * weights * src_activation * kind_effect
                np.add.at(accum, dst_idx, last_contribution)
                edge_eligibility = 0.95 * edge_eligibility + np.abs(src_activation)
                edge_age = edge_age + 1
            if noninput_indices.size:
                raw = accum[noninput_indices]
                blended = (1.0 - forget[noninput_indices]) * raw + forget[noninput_indices] * states[noninput_indices]
                updated = np.tanh(np.clip(blended, -self.activation_clip, self.activation_clip))
                activations[noninput_indices] = updated
                states[noninput_indices] = updated
                node_eligibility[noninput_indices] = 0.95 * node_eligibility[noninput_indices] + np.abs(updated)
                node_age[noninput_indices] += 1
                for idx in noninput_indices[np.abs(updated) > self.activity_epsilon]:
                    active_nodes.add(node_ids[int(idx)])

        # Sync mutable scalar evidence once, not once per recurrent step.  Do
        # array-level sanitisation first; scalar np.nan_to_num calls are very
        # expensive for an 8k-edge graph in CPython.
        safe_activations = np.clip(np.nan_to_num(activations, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        safe_states = np.clip(np.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        safe_node_eligibility = np.nan_to_num(node_eligibility, nan=0.0, posinf=0.0, neginf=0.0)
        safe_edge_eligibility = np.nan_to_num(edge_eligibility, nan=0.0, posinf=0.0, neginf=0.0)
        safe_short_term = np.clip(np.nan_to_num(short_term, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
        for idx, node_id in enumerate(node_ids):
            node = self.neurons[node_id]
            node.activation = float(safe_activations[idx])
            node.state = float(safe_states[idx])
            node.eligibility = float(safe_node_eligibility[idx])
            node.age = int(node_age[idx])
        for idx, edge in enumerate(edge_refs):
            edge.eligibility = float(safe_edge_eligibility[idx])
            edge.short_term_state = float(safe_short_term[idx])
            edge.age = int(edge_age[idx])

        active_mask = np.abs(last_contribution) > self.activity_epsilon if edge_count else np.zeros(0, dtype=bool)
        active_count = int(np.count_nonzero(active_mask))
        selected_indices: list[int] = []
        if active_count:
            active = np.flatnonzero(active_mask)
            active_ranked = sorted((int(idx) for idx in active), key=lambda idx: abs(float(last_contribution[idx])), reverse=True)

            # The old global top-K always preferred the 2,856 dense
            # input->encoder synapses.  It was mathematically valid but made
            # the live graph falsely look like a two-layer network.  Reserve a
            # small, causal trace quota for every active semantic route, then
            # spend the remaining budget on the globally strongest edges.
            route_buckets: dict[tuple[str, str, str], list[int]] = {}
            for idx in active_ranked:
                edge = edge_refs[idx]
                route = (self.neurons[edge.src].region, self.neurons[edge.dst].region, edge.kind)
                route_buckets.setdefault(route, []).append(idx)
            per_route = max(0, int(self.trace_min_edges_per_route))
            selected_set: set[int] = set()
            if per_route:
                # Round-robin allocation keeps the bound hard even after a
                # future graph grows many additional semantic routes.
                ranked_routes = sorted(
                    route_buckets.values(),
                    key=lambda indices: abs(float(last_contribution[indices[0]])),
                    reverse=True,
                )
                for rank in range(per_route):
                    for indices in ranked_routes:
                        if len(selected_set) >= self.trace_max_edges:
                            break
                        if rank < len(indices):
                            idx = indices[rank]
                            if idx not in selected_set:
                                selected_set.add(idx)
                                selected_indices.append(idx)
                    if len(selected_set) >= self.trace_max_edges:
                        break
            for idx in active_ranked:
                if len(selected_set) >= self.trace_max_edges:
                    break
                if idx not in selected_set:
                    selected_set.add(idx)
                    selected_indices.append(idx)

            # Reserve trace slots for genuine candidate specialist paths even
            # when their warm-up gate is too small to reach the route/global top-K.
            candidate = [
                idx for idx in active_ranked
                if edge_refs[idx].candidate_id not in {None, "", "graph_seed"}
                and edge_refs[idx].lifecycle == NEURON_LIFECYCLE_CANDIDATE
            ]
            if candidate:
                selected_indices.extend(candidate[:self.trace_candidate_edges])
        selected_indices = sorted(set(selected_indices), key=lambda idx: abs(float(last_contribution[idx])), reverse=True)
        edge_gates = {edge_ids[idx]: float(last_gate_values[idx]) for idx in selected_indices}
        edge_contributions = {edge_ids[idx]: float(last_contribution[idx]) for idx in selected_indices}
        active_edge_ids = tuple(edge_ids[idx] for idx in selected_indices)
        output = activations[output_indices].astype(float, copy=True)
        self._activation_sequence += 1
        trace = GraphActivationTrace(
            sequence=self._activation_sequence,
            active_neuron_ids=tuple(sorted(active_nodes)),
            active_edge_ids=active_edge_ids,
            edge_gates=edge_gates,
            edge_contributions=edge_contributions,
            region_activity={
                region: float(np.mean(np.abs(activations[np.asarray(indices, dtype=int)]))) if len(indices) else 0.0
                for region, indices in region_indices.items()
            },
            context_regions=tuple(sorted(normalized_context)),
            active_edge_count=active_count,
            traced_edge_count=len(active_edge_ids),
        )
        self.last_activation_trace = trace
        if return_trace:
            return output, trace
        return output

    def invariant_report(self, *, output_size: int | None = None) -> NeuralGraphInvariantReport:
        errors: list[str] = []
        region_members: set[str] = set()
        for region_name, region in self.regions.items():
            for neuron_id in region.neuron_ids:
                if neuron_id not in self.neurons:
                    errors.append(f"region {region_name} references missing neuron {neuron_id}")
                elif self.neurons[neuron_id].region != region_name:
                    errors.append(f"neuron {neuron_id} stored in wrong region {region_name}")
                if neuron_id in region_members:
                    errors.append(f"neuron {neuron_id} appears in multiple regions")
                region_members.add(neuron_id)
        for neuron_id in self.neurons:
            if neuron_id not in region_members:
                errors.append(f"neuron {neuron_id} missing from region registry")
        for edge in self.edges.values():
            if edge.src not in self.neurons:
                errors.append(f"edge {edge.id} missing source {edge.src}")
            if edge.dst not in self.neurons:
                errors.append(f"edge {edge.id} missing destination {edge.dst}")
            if edge.kind not in VALID_EDGE_KINDS:
                errors.append(f"edge {edge.id} has invalid kind {edge.kind}")
            if edge.lifecycle not in VALID_NEURON_LIFECYCLES:
                errors.append(f"edge {edge.id} has invalid lifecycle {edge.lifecycle}")
            for value in (edge.weight, edge.gate, edge.utility, edge.eligibility, edge.plasticity, edge.short_term_state):
                if not np.isfinite(value):
                    errors.append(f"edge {edge.id} contains non-finite value")
                    break
        for node in self.neurons.values():
            for value in (node.bias, node.activation, node.state, node.utility, node.plasticity, node.eligibility, getattr(node, "forget_gate", 0.0)):
                if not np.isfinite(value):
                    errors.append(f"neuron {node.id} contains non-finite value")
                    break
        # V21 module registry must never silently point at missing topology.
        # This catches a partial prune/rollback before telemetry or a candidate
        # promotion can report a fictional specialist module.
        registry = getattr(self, "specialist_modules", {})
        if isinstance(registry, Mapping):
            for module_id, raw in registry.items():
                if not isinstance(raw, Mapping):
                    errors.append(f"specialist module {module_id} has invalid registry record")
                    continue
                node_ids = tuple(raw.get("node_ids", ()) or ())
                edge_ids = tuple(raw.get("edge_ids", ()) or ())
                if not node_ids:
                    errors.append(f"specialist module {module_id} has no nodes")
                for node_id in node_ids:
                    node = self.neurons.get(str(node_id))
                    if node is None:
                        errors.append(f"specialist module {module_id} references missing node {node_id}")
                    elif str(getattr(node, "module_id", "") or "") != str(module_id):
                        errors.append(f"specialist module {module_id} owns mismatched node {node_id}")
                for edge_id in edge_ids:
                    if str(edge_id) not in self.edges:
                        errors.append(f"specialist module {module_id} references missing edge {edge_id}")
        if self.budget.max_neurons > 0 and len(self.neurons) > self.budget.max_neurons:
            errors.append("neuron budget exceeded")
        if self.budget.max_connections > 0 and len(self.edges) > self.budget.max_connections:
            errors.append("connection budget exceeded")
        if self.budget.max_recurrent_edges > 0 and self._count_edges(kind=EDGE_RECURRENT) > self.budget.max_recurrent_edges:
            errors.append("recurrent edge budget exceeded")
        if self.budget.max_lateral_edges > 0 and self._count_edges(kind=EDGE_LATERAL) > self.budget.max_lateral_edges:
            errors.append("lateral edge budget exceeded")
        if self.budget.max_cross_module_edges > 0 and self._count_cross_module_edges() > self.budget.max_cross_module_edges:
            errors.append("cross-module edge budget exceeded")
        if self.budget.max_m1_module_edges > 0 and self._count_m1_module_edges() > self.budget.max_m1_module_edges:
            errors.append("M1-module edge budget exceeded")
        if self.budget.max_cross_module_recurrent_edges > 0 and self._count_cross_module_edges(kind=EDGE_RECURRENT) > self.budget.max_cross_module_recurrent_edges:
            errors.append("cross-module recurrent edge budget exceeded")
        if self.budget.max_cross_module_lateral_edges > 0 and self._count_cross_module_edges(kind=EDGE_LATERAL) > self.budget.max_cross_module_lateral_edges:
            errors.append("cross-module lateral edge budget exceeded")
        if output_size is not None and len(self._region_ids(REGION_OUTPUT)) != int(output_size):
            errors.append(f"expected {output_size} output neurons")
        return NeuralGraphInvariantReport(ok=len(errors) == 0, errors=errors)

    def assert_invariants(self, *, output_size: int | None = None) -> None:
        report = self.invariant_report(output_size=output_size)
        if not report.ok:
            raise ValueError("; ".join(report.errors))

    def _normalize_context(
        self,
        context_by_region: Mapping[str, Sequence[float] | np.ndarray | float] | None,
    ) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        if not context_by_region:
            return result
        for region, raw in context_by_region.items():
            if region not in VALID_REGIONS or region == REGION_INPUT:
                continue
            count = len(self._region_ids(region))
            if count == 0:
                continue
            values = np.asarray(raw, dtype=float).reshape(-1)
            if values.size == 0:
                continue
            if values.size == 1:
                values = np.full(count, float(values[0]), dtype=float)
            elif values.size != count:
                resized = np.zeros(count, dtype=float)
                resized[: min(count, values.size)] = values[:count]
                values = resized
            result[region] = np.clip(values, -self.activation_clip, self.activation_clip)
        return result

    def _region_ids(self, region: str) -> list[str]:
        return list(self.regions[region].neuron_ids)

    def _validate_new_edge(self, edge: ConnectionEdge) -> None:
        src = self.neurons[edge.src]
        dst = self.neurons[edge.dst]
        if self.budget.max_recurrent_edges > 0 and edge.kind == EDGE_RECURRENT and self._count_edges(kind=EDGE_RECURRENT) >= self.budget.max_recurrent_edges:
            raise ValueError("recurrent edge budget exceeded")
        if self.budget.max_lateral_edges > 0 and edge.kind == EDGE_LATERAL and self._count_edges(kind=EDGE_LATERAL) >= self.budget.max_lateral_edges:
            raise ValueError("lateral edge budget exceeded")
        if _is_cross_module(src, dst):
            if self.budget.max_cross_module_edges > 0 and self._count_cross_module_edges() >= self.budget.max_cross_module_edges:
                raise ValueError("cross-module edge budget exceeded")
            if self.budget.max_cross_module_recurrent_edges > 0 and edge.kind == EDGE_RECURRENT and self._count_cross_module_edges(kind=EDGE_RECURRENT) >= self.budget.max_cross_module_recurrent_edges:
                raise ValueError("cross-module recurrent edge budget exceeded")
            if self.budget.max_cross_module_lateral_edges > 0 and edge.kind == EDGE_LATERAL and self._count_cross_module_edges(kind=EDGE_LATERAL) >= self.budget.max_cross_module_lateral_edges:
                raise ValueError("cross-module lateral edge budget exceeded")
            # Runtime development uses ``max_cross_module_initial_weight``.
            # The graph primitive also accepts audited migrations/tests that
            # reconstruct a pre-existing connection with a larger weight.
            if edge.created_by in {"graph_development", "adaptive_supernetwork"} and abs(edge.weight) > self.max_cross_module_initial_weight:
                raise ValueError("cross-module initial weight exceeds safe candidate limit")
        if _is_m1_module_edge(src, dst):
            if self.budget.max_m1_module_edges > 0 and self._count_m1_module_edges() >= self.budget.max_m1_module_edges:
                raise ValueError("M1-module edge budget exceeded")
            if src.module_id == "m1" and dst.region not in self.allowed_m1_module_targets:
                raise ValueError(f"M1 is not allowed to connect to {dst.region}")
            if dst.module_id == "m1" and dst.region not in self.allowed_module_m1_targets:
                raise ValueError(f"module is not allowed to connect to M1 {dst.region}")
            if dst.module_id == "m1" and dst.region == REGION_OUTPUT:
                if edge.kind not in {EDGE_GATING, EDGE_SELECTOR, EDGE_CRITIC}:
                    raise ValueError("module->M1 output must be a bounded bias/gate edge")
                if abs(edge.weight) > self.max_action_bias_weight or edge.gate > self.max_action_bias_gate:
                    raise ValueError("module->M1 output edge exceeds bounded bias/gate limits")

    def _count_edges(self, *, kind: str) -> int:
        return sum(1 for edge in self.edges.values() if edge.kind == kind and edge.enabled)

    def _count_cross_module_edges(self, *, kind: str | None = None) -> int:
        return sum(
            1
            for edge in self.edges.values()
            if edge.enabled
            and (kind is None or edge.kind == kind)
            and _is_cross_module(self.neurons[edge.src], self.neurons[edge.dst])
        )

    def _count_m1_module_edges(self) -> int:
        return sum(
            1
            for edge in self.edges.values()
            if edge.enabled and _is_m1_module_edge(self.neurons[edge.src], self.neurons[edge.dst])
        )


def _sigmoid(value: float) -> float:
    clipped = float(np.clip(value, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-clipped)))


def _is_cross_module(src: NeuronNode, dst: NeuronNode) -> bool:
    return src.module_id is not None and dst.module_id is not None and src.module_id != dst.module_id


def _is_m1_module_edge(src: NeuronNode, dst: NeuronNode) -> bool:
    return _is_cross_module(src, dst) and (src.module_id == "m1" or dst.module_id == "m1")
