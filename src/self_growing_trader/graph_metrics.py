from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from .module_need import ModuleNeedSignal
from .neural_graph import EDGE_RECURRENT, NeuronNode, NeuralGraph


def graph_metrics(graph: NeuralGraph) -> dict[str, float]:
    enabled_edges = [edge for edge in graph.edges.values() if edge.enabled]
    recurrent_edges = [edge for edge in enabled_edges if edge.kind == EDGE_RECURRENT]
    cross_module_edges = [
        edge for edge in enabled_edges
        if edge.src_module_id is not None and edge.dst_module_id is not None and edge.src_module_id != edge.dst_module_id
    ]
    m1_module_edges = [
        edge for edge in cross_module_edges
        if edge.src_module_id == "m1" or edge.dst_module_id == "m1"
    ]
    metrics = {
        "graph_neurons": float(len(graph.neurons)),
        "graph_edges": float(len(graph.edges)),
        "graph_enabled_edges": float(len(enabled_edges)),
        "graph_disabled_edges": float(len(graph.edges) - len(enabled_edges)),
        "graph_recurrent_edges": float(len(recurrent_edges)),
        "graph_lateral_edges": float(sum(1 for edge in enabled_edges if edge.kind == "lateral")),
        "cross_module_edges": float(len(cross_module_edges)),
        "m1_module_edges": float(len(m1_module_edges)),
        "m1_to_module_edges": float(sum(1 for edge in m1_module_edges if edge.src_module_id == "m1")),
        "module_to_m1_edges": float(sum(1 for edge in m1_module_edges if edge.dst_module_id == "m1")),
        "cross_module_edge_utility": _avg(edge.utility for edge in cross_module_edges),
        "edge_utility_mean": _avg(edge.utility for edge in enabled_edges),
        "edge_gate_mean": _avg(edge.gate for edge in enabled_edges),
        "edge_gate_min": _min(edge.gate for edge in enabled_edges),
        "edge_gate_max": _max(edge.gate for edge in enabled_edges),
        "edge_plasticity_mean": _avg(edge.plasticity for edge in enabled_edges),
        "edge_plasticity_min": _min(edge.plasticity for edge in enabled_edges),
        "edge_plasticity_max": _max(edge.plasticity for edge in enabled_edges),
        "node_utility_mean": _avg(neuron.utility for neuron in graph.neurons.values()),
        "recurrent_loop_count": float(len(recurrent_edges)),
        "recurrent_max_gain": max((abs(edge.weight) * _sigmoid(edge.gate) for edge in recurrent_edges), default=0.0),
        "nan_or_inf_count": float(_nan_or_inf_count(graph)),
    }
    for region, count in _region_counts(graph).items():
        metrics[f"region_{region}_neurons"] = float(count)
    for kind, edges in _group_edges_by_kind(cross_module_edges).items():
        metrics[f"cross_module_{kind}_edges"] = float(len(edges))
        metrics[f"cross_module_{kind}_utility"] = _avg(edge.utility for edge in edges)
    return metrics


def module_need_metrics(signal: ModuleNeedSignal) -> dict[str, float]:
    return {f"module_need_{role}": float(score) for role, score in signal.scores().items()}


def credit_attribution_metrics(
    *,
    active_neuron_ids: Iterable[str] = (),
    active_edge_ids: Iterable[str] = (),
    credited_neurons: int = 0,
    credited_edges: int = 0,
) -> dict[str, float]:
    return {
        "credit_active_neurons": float(len(tuple(active_neuron_ids))),
        "credit_active_edges": float(len(tuple(active_edge_ids))),
        "credit_updated_neurons": float(credited_neurons),
        "credit_updated_edges": float(credited_edges),
    }


def trial_history_metrics(records: Iterable[object]) -> dict[str, float]:
    rows = list(records)
    total = len(rows)
    promoted = 0
    children = 0
    child_promoted = 0
    by_role: dict[str, list[bool]] = defaultdict(list)
    cluster_outcomes: dict[str, list[bool]] = defaultdict(list)
    for record in rows:
        validation = getattr(record, "validation", None)
        is_promoted = bool(getattr(validation, "promoted", False))
        promoted += int(is_promoted)
        diff = dict(getattr(record, "structural_diff", {}) or {})
        role = str(dict(diff.get("details", {}) or {}).get("role", "unknown"))
        by_role[role].append(is_promoted)
        reason = str(getattr(record, "module_need_reason", "") or "unknown")
        cluster_outcomes[reason].append(is_promoted)
        if "blueprint" in diff or "spawn" in str(diff.get("action", "")):
            children += 1
            child_promoted += int(is_promoted)
    metrics = {
        "growth_candidates_tested": float(total),
        "growth_candidates_promoted": float(promoted),
        "growth_accept_rate": float(promoted / total) if total else 0.0,
        "child_spawned": float(children),
        "child_promoted": float(child_promoted),
        "child_survival_rate": float(child_promoted / children) if children else 0.0,
    }
    for role, values in by_role.items():
        metrics[f"module_role_accept_rate_{role}"] = float(sum(values) / len(values)) if values else 0.0
    for cluster, values in cluster_outcomes.items():
        metrics[f"cluster_to_module_outcome_{cluster}"] = float(sum(values) / len(values)) if values else 0.0
    return metrics


def _group_edges_by_kind(edges):
    grouped = defaultdict(list)
    for edge in edges:
        grouped[edge.kind].append(edge)
    return grouped


def _region_counts(graph: NeuralGraph) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for neuron in graph.neurons.values():
        counts[neuron.region] += 1
    return dict(counts)


def _avg(values) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return float(sum(items) / len(items))


def _min(values) -> float:
    items = [float(value) for value in values]
    return min(items) if items else 0.0


def _max(values) -> float:
    items = [float(value) for value in values]
    return max(items) if items else 0.0


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0))))


def _nan_or_inf_count(graph: NeuralGraph) -> int:
    count = 0
    for edge in graph.edges.values():
        values = (edge.weight, edge.gate, edge.utility, edge.eligibility, edge.plasticity, getattr(edge, "short_term_state", 0.0))
        count += sum(1 for value in values if not np.isfinite(value))
    for neuron in graph.neurons.values():
        count += _bad_neuron_values(neuron)
    return count


def _bad_neuron_values(neuron: NeuronNode) -> int:
    values = (neuron.bias, neuron.activation, neuron.state, neuron.utility, neuron.plasticity, neuron.eligibility, getattr(neuron, "forget_gate", 0.0))
    return sum(1 for value in values if not np.isfinite(value))
