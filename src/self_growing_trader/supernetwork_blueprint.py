from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .module_need import ModuleErrorCluster, ModuleNeedSignal, ModuleRolePrior, ModuleSpec, module_spec_for_role
from .neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_MEMORY,
    EDGE_PREDICTIVE,
    EDGE_RECURRENT,
    EDGE_SELECTOR,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_INPUT,
    REGION_MEMORY,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    NeuralGraph,
    NeuralGraphBudget,
)


@dataclass(frozen=True)
class RegionBlueprint:
    name: str
    neurons: int
    module_id: str | None = None


@dataclass(frozen=True)
class ConnectionRule:
    src_region: str
    dst_region: str
    kind: str = EDGE_FORWARD
    density: float = 1.0
    gate: float = -2.0
    weight_scale: float = 0.1


@dataclass(frozen=True)
class SuperNetworkBlueprint:
    blueprint_id: str
    parent_id: str | None
    role: str
    regions: tuple[RegionBlueprint, ...]
    connection_rules: tuple[ConnectionRule, ...]
    plasticity_profile: str = "fast_initial_then_consolidate"
    growth_policy: dict[str, int | float | bool | str | None] = field(default_factory=dict)
    risk_policy: dict[str, int | float | bool | str | None] = field(default_factory=dict)
    memory_schema: int = 6
    objective: str = ""

    def assert_no_parent_state(self) -> None:
        text = repr(self)
        blocked = ("weight=", "memory_state", "recurrent_state", "neuron_id_map", "connection_id_map")
        for token in blocked:
            if token in text:
                raise ValueError(f"blueprint appears to contain parent state token: {token}")




class BlueprintGenerator:
    """Generate design priors from failures, not parent weights or memory."""

    def __init__(self, role_prior: ModuleRolePrior | None = None) -> None:
        self.role_prior = role_prior or ModuleRolePrior()

    def generate(
        self,
        *,
        blueprint_id: str,
        parent_id: str | None,
        input_size: int,
        output_size: int,
        graph_summary: dict[str, float] | None = None,
        memory_summary: dict[str, float] | None = None,
        failure_clusters: Iterable[ModuleErrorCluster] = (),
        winning_causal_patterns: Iterable[dict[str, object]] = (),
        hidden_size: int = 12,
    ) -> SuperNetworkBlueprint:
        clusters = list(failure_clusters)
        signal = clusters[0].signal if clusters else ModuleNeedSignal()
        scores = signal.scores()
        role = self.role_prior.choose_role(scores, cluster_id=clusters[0].cluster_id if clusters else None)
        graph_summary = graph_summary or {}
        memory_summary = memory_summary or {}
        causal_count = sum(1 for _ in winning_causal_patterns)
        adjusted_hidden = max(4, int(hidden_size + min(8, graph_summary.get("graph_neurons", 0.0) // 64) + min(4, causal_count)))
        reason = clusters[0].cluster_id if clusters else f"generated:{role}"
        spec = module_spec_for_role(role, reason=reason)
        blueprint = blueprint_from_module_spec(
            spec,
            blueprint_id=blueprint_id,
            parent_id=parent_id,
            input_size=input_size,
            output_size=output_size,
            hidden_size=adjusted_hidden,
        )
        # Store only compressed counts/thresholds as a design prior.
        growth_policy = dict(blueprint.growth_policy)
        growth_policy.update({
            "memory_depth_hint": int(memory_summary.get("memory_depth", 0.0)),
            "causal_pattern_count": causal_count,
            "source_graph_neurons": int(graph_summary.get("graph_neurons", 0.0)),
        })
        result = SuperNetworkBlueprint(
            blueprint_id=blueprint.blueprint_id,
            parent_id=blueprint.parent_id,
            role=blueprint.role,
            regions=blueprint.regions,
            connection_rules=blueprint.connection_rules,
            plasticity_profile=blueprint.plasticity_profile,
            growth_policy=growth_policy,
            risk_policy=dict(blueprint.risk_policy),
            memory_schema=blueprint.memory_schema,
            objective=blueprint.objective,
        )
        result.assert_no_parent_state()
        return result


def blueprint_from_module_spec(
    spec: ModuleSpec,
    *,
    blueprint_id: str,
    parent_id: str | None,
    input_size: int,
    output_size: int,
    hidden_size: int = 12,
) -> SuperNetworkBlueprint:
    module_id = spec.role
    hidden_size = max(4, int(hidden_size))
    regions = (
        RegionBlueprint(REGION_INPUT, max(1, int(input_size)), module_id="input"),
        RegionBlueprint(REGION_ENCODER, hidden_size, module_id=module_id),
        RegionBlueprint(REGION_MEMORY, max(4, hidden_size // 2), module_id=module_id),
        RegionBlueprint(REGION_PREDICTOR, max(4, hidden_size // 2), module_id=module_id),
        RegionBlueprint(REGION_CRITIC, max(4, hidden_size // 3), module_id=module_id),
        RegionBlueprint(REGION_SELECTOR, max(2, hidden_size // 4), module_id=module_id),
        RegionBlueprint(REGION_OUTPUT, max(1, int(output_size)), module_id="output"),
    )
    rules = (
        ConnectionRule(REGION_INPUT, REGION_ENCODER, EDGE_FORWARD, density=0.35, gate=spec.initial_gate_bias, weight_scale=0.2),
        ConnectionRule(REGION_ENCODER, REGION_MEMORY, EDGE_MEMORY, density=0.25, gate=spec.initial_gate_bias, weight_scale=0.15),
        ConnectionRule(REGION_MEMORY, REGION_MEMORY, EDGE_RECURRENT, density=0.15, gate=spec.initial_gate_bias, weight_scale=0.08),
        ConnectionRule(REGION_ENCODER, REGION_PREDICTOR, EDGE_PREDICTIVE, density=0.25, gate=spec.initial_gate_bias, weight_scale=0.12),
        ConnectionRule(REGION_MEMORY, REGION_PREDICTOR, EDGE_MEMORY, density=0.25, gate=spec.initial_gate_bias, weight_scale=0.12),
        ConnectionRule(REGION_PREDICTOR, REGION_CRITIC, EDGE_CRITIC, density=0.40, gate=spec.initial_gate_bias, weight_scale=0.10),
        ConnectionRule(REGION_CRITIC, REGION_SELECTOR, EDGE_SELECTOR, density=0.60, gate=spec.initial_gate_bias, weight_scale=0.10),
        ConnectionRule(REGION_SELECTOR, REGION_OUTPUT, EDGE_GATING, density=0.50, gate=spec.initial_gate_bias, weight_scale=0.05),
    )
    return SuperNetworkBlueprint(
        blueprint_id=blueprint_id,
        parent_id=parent_id,
        role=spec.role,
        regions=regions,
        connection_rules=rules,
        growth_policy={"max_neurons": None, "max_connections": None, "capacity_mode": "evidence_governed_open"},
        risk_policy={"start_gated": True, "mt5_promotion_required": True},
        memory_schema=6,
        objective=spec.reason,
    )


def spawn_supernetwork_from_blueprint(
    blueprint: SuperNetworkBlueprint,
    *,
    seed: int,
    budget: NeuralGraphBudget | None = None,
) -> NeuralGraph:
    blueprint.assert_no_parent_state()
    rng = np.random.default_rng(seed)
    graph = NeuralGraph(budget=budget or _budget_from_blueprint(blueprint))
    region_ids: dict[str, list[str]] = {}
    for region in blueprint.regions:
        ids = []
        for index in range(region.neurons):
            neuron_id = f"{blueprint.blueprint_id}:{region.name}:{index}"
            ids.append(graph.add_neuron(region.name, neuron_id=neuron_id, module_id=region.module_id, lifecycle="candidate"))
        region_ids[region.name] = ids
    for rule in blueprint.connection_rules:
        _connect_by_rule(graph, region_ids.get(rule.src_region, ()), region_ids.get(rule.dst_region, ()), rule, rng, blueprint)
    graph.assert_invariants(output_size=len(region_ids.get(REGION_OUTPUT, ())))
    return graph


def diversity_score(a: NeuralGraph, b: NeuralGraph) -> float:
    a_stats = _graph_stats(a)
    b_stats = _graph_stats(b)
    keys = sorted(set(a_stats) | set(b_stats))
    diffs = []
    for key in keys:
        av = a_stats.get(key, 0.0)
        bv = b_stats.get(key, 0.0)
        denom = max(abs(av), abs(bv), 1.0)
        diffs.append(abs(av - bv) / denom)
    return float(np.mean(diffs)) if diffs else 0.0


def reject_if_clone_like(parent: NeuralGraph, child: NeuralGraph, *, min_diversity: float = 0.05) -> None:
    score = diversity_score(parent, child)
    if score < min_diversity:
        raise ValueError(f"child graph diversity too low: {score:.4f}")


def _budget_from_blueprint(blueprint: SuperNetworkBlueprint) -> NeuralGraphBudget:
    """Build an open-ended budget unless a caller explicitly supplies one.

    Blueprint generation may carry growth hints, but never installs a hidden
    global neuron or connection ceiling.  Zero is the NeuralGraph convention
    for an unbounded structural dimension.
    """
    return NeuralGraphBudget(
        max_neurons=0,
        max_connections=0,
        max_recurrent_edges=0,
        max_lateral_edges=0,
        max_cross_module_edges=0,
        max_m1_module_edges=0,
        max_cross_module_recurrent_edges=0,
        max_cross_module_lateral_edges=0,
    )


def _connect_by_rule(
    graph: NeuralGraph,
    src_ids: Iterable[str],
    dst_ids: Iterable[str],
    rule: ConnectionRule,
    rng: np.random.Generator,
    blueprint: SuperNetworkBlueprint,
) -> None:
    src_list = list(src_ids)
    dst_list = list(dst_ids)
    if not src_list or not dst_list:
        return
    density = float(np.clip(rule.density, 0.0, 1.0))
    for src in src_list:
        for dst in dst_list:
            if rule.kind == EDGE_RECURRENT and src == dst and len(src_list) > 1:
                pass
            if rng.random() > density:
                continue
            graph.add_connection(
                src,
                dst,
                weight=float(rng.normal(0.0, rule.weight_scale)),
                gate=rule.gate,
                kind=rule.kind,
                created_by="blueprint",
                need_reason=blueprint.objective,
                candidate_id=blueprint.blueprint_id,
            )


def _graph_stats(graph: NeuralGraph) -> dict[str, float]:
    stats: dict[str, float] = {"neurons": float(len(graph.neurons)), "edges": float(len(graph.edges))}
    for neuron in graph.neurons.values():
        stats[f"region:{neuron.region}"] = stats.get(f"region:{neuron.region}", 0.0) + 1.0
        if neuron.module_id:
            stats[f"module:{neuron.module_id}"] = stats.get(f"module:{neuron.module_id}", 0.0) + 1.0
    for edge in graph.edges.values():
        stats[f"edge:{edge.kind}"] = stats.get(f"edge:{edge.kind}", 0.0) + 1.0
    return stats
