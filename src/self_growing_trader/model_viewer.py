from __future__ import annotations

"""Static, inspectable 3D snapshot viewer for saved graph policies.

This module intentionally does *not* mutate a policy.  It exports an HTML
snapshot with graph-local identifiers namespaced before serialisation, module
circuit boundaries, child-supernetwork graphs, and balanced edge selection.
The browser renderer is event-driven rather than repainting continuously so a
saved policy with thousands of synapses remains inspectable.
"""

from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .m1_graph_module import M1GraphModule
from .self_growing_runner import load_policy


REGION_ORDER = {
    "input": 0,
    "encoder": 1,
    "memory": 2,
    "hypothesis": 3,
    "predictor": 4,
    "critic": 5,
    "selector": 6,
    "meta": 7,
    "output": 8,
    "curiosity": 3,
    "development": 3,
    "risk": 4,
    "action_head": 5,
    "utility": 6,
}


@dataclass(frozen=True)
class GraphDescriptor:
    graph_id: str
    label: str
    graph: Any
    graph_kind: str
    status: str = "active"
    selectable: bool = True
    graph_index: int = 0


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _qualified_id(graph_id: str, local_id: str) -> str:
    return f"{graph_id}::{local_id}"


def _safe_json(value: Any) -> str:
    # Prevent a model/journal string containing </script> from closing the
    # payload element in the generated standalone HTML file.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def _graph_descriptors(policy: Any) -> list[GraphDescriptor]:
    descriptors: list[GraphDescriptor] = []
    used: set[str] = set()

    def unique(value: str) -> str:
        base = str(value or "graph")
        candidate = base
        index = 2
        while candidate in used:
            candidate = f"{base}:{index}"
            index += 1
        used.add(candidate)
        return candidate

    for module_index, module in enumerate(getattr(policy, "modules", []) or []):
        m1 = getattr(module, "m1_component", None)
        if not isinstance(m1, M1GraphModule):
            continue
        graph = getattr(m1, "graph", None)
        if graph is None or not hasattr(graph, "neurons") or not hasattr(graph, "edges"):
            continue
        label = str(getattr(module, "name", f"module_{module_index}"))
        descriptors.append(GraphDescriptor(
            graph_id=unique(f"m1:{label}"),
            label=label,
            graph=graph,
            graph_kind="m1",
            status="active",
            selectable=True,
            graph_index=len(descriptors),
        ))

    reasoning = getattr(getattr(policy, "reasoning_graph", None), "graph", None)
    if reasoning is not None and hasattr(reasoning, "neurons") and hasattr(reasoning, "edges"):
        descriptors.append(GraphDescriptor(
            graph_id=unique("reasoning_graph"),
            label="reasoning_graph",
            graph=reasoning,
            graph_kind="reasoning",
            status="active",
            selectable=True,
            graph_index=len(descriptors),
        ))

    manager = getattr(policy, "child_supernetwork_manager", None)
    children = getattr(manager, "children", {}) if manager is not None else {}
    if isinstance(children, Mapping):
        for network_id, record in sorted(children.items(), key=lambda item: str(item[0])):
            graph = getattr(getattr(record, "network", None), "graph", None)
            if graph is None or not hasattr(graph, "neurons") or not hasattr(graph, "edges"):
                continue
            status = str(getattr(record, "status", "candidate") or "candidate")
            selectable = bool(getattr(record, "selectable", status == "promoted"))
            descriptors.append(GraphDescriptor(
                graph_id=unique(f"child:{network_id}"),
                label=f"child:{network_id}",
                graph=graph,
                graph_kind="child",
                status=status,
                selectable=selectable,
                graph_index=len(descriptors),
            ))
    return descriptors


def _normalise_ids(values: Any, graph_id: str) -> set[str]:
    output: set[str] = set()
    for value in values or ():
        text = str(value or "")
        if not text:
            continue
        if text.startswith(f"{graph_id}::"):
            output.add(text[len(graph_id) + 2:])
        elif "::" not in text:
            output.add(text)
    return output


def _read_latest_activity(journal_path: str | None) -> tuple[set[str], set[str], dict[str, dict[str, set[str]]], dict[str, Any]]:
    """Read the newest usable trace without guessing cross-graph raw IDs.

    Older journals may expose one raw root-graph trace.  Newer V18+ records
    can expose graph_traces keyed by graph ID.  The caller only applies a raw
    global trace to the first M1 graph, avoiding M1/reasoning ``e0`` collision.
    """
    if not journal_path:
        return set(), set(), {}, {}
    path = Path(journal_path)
    if not path.exists() or not path.is_file():
        return set(), set(), {}, {}
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return set(), set(), {}, {}

    for line in reversed(lines[-5000:]):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict):
            continue
        trace = item.get("trace") if isinstance(item.get("trace"), dict) else item
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        raw_nodes = {str(value) for value in (trace.get("active_neuron_ids", []) or []) if str(value)}
        raw_edges = {str(value) for value in (trace.get("active_edge_ids", []) or []) if str(value)}
        per_graph: dict[str, dict[str, set[str]]] = {}
        candidates: list[Any] = [trace.get("graph_traces"), payload.get("graph_traces"), payload.get("graphs")]
        for source in candidates:
            if not isinstance(source, Mapping):
                continue
            for graph_id, raw in source.items():
                if not isinstance(raw, Mapping):
                    continue
                per_graph[str(graph_id)] = {
                    "nodes": _normalise_ids(raw.get("active_neuron_ids", raw.get("active_nodes", ())), str(graph_id)),
                    "edges": _normalise_ids(raw.get("active_edge_ids", raw.get("active_edges", ())), str(graph_id)),
                }
        if raw_nodes or raw_edges or per_graph:
            return raw_nodes, raw_edges, per_graph, item
    return set(), set(), {}, {}


def _trace_activity_from_graph(graph: Any) -> tuple[set[str], set[str]]:
    trace = getattr(graph, "last_activation_trace", None)
    if trace is None:
        return set(), set()
    nodes = {str(value) for value in tuple(getattr(trace, "active_neuron_ids", ()) or ()) if str(value)}
    edges = {str(value) for value in tuple(getattr(trace, "active_edge_ids", ()) or ()) if str(value)}
    return nodes, edges


def _node_position(
    region: str,
    index: int,
    count: int,
    *,
    graph_index: int,
    graph_count: int,
    group_index: int,
    group_count: int,
) -> dict[str, float]:
    layer = REGION_ORDER.get(str(region), 4)
    count = max(1, int(count))
    angle = 2.0 * math.pi * (index / count)
    radius = 26.0 + 7.0 * math.sqrt(count)
    graph_offset = (float(graph_index) - (float(graph_count) - 1.0) / 2.0) * 620.0
    group_offset = (float(group_index) - (float(group_count) - 1.0) / 2.0) * 150.0
    return {
        "x": float((layer - 4.0) * 126.0),
        "y": float(graph_offset + group_offset + math.sin(angle) * radius),
        "z": float(math.cos(angle) * radius),
    }


def _group_record(
    *,
    graph: Any,
    descriptor: GraphDescriptor,
    local_module_id: str,
    node_ids: list[str],
    group_type: str,
) -> dict[str, Any]:
    registry = getattr(graph, "specialist_modules", {})
    raw = registry.get(local_module_id, {}) if isinstance(registry, Mapping) else {}
    raw = raw if isinstance(raw, Mapping) else {}
    is_specialist = bool(local_module_id and local_module_id != "m1" and (group_type == "specialist" or raw))
    label = local_module_id if is_specialist else (
        "Reasoning core" if descriptor.graph_kind == "reasoning" else (
            "Child graph core" if descriptor.graph_kind == "child" else "M1 core"
        )
    )
    return {
        "id": _qualified_id(descriptor.graph_id, local_module_id or "__core__"),
        "graph_id": descriptor.graph_id,
        "graph_label": descriptor.label,
        "graph_kind": descriptor.graph_kind,
        "local_module_id": str(local_module_id or ""),
        "label": label,
        "group_type": "specialist" if is_specialist else group_type,
        "is_specialist": bool(is_specialist),
        "role": str(raw.get("role", "") or ""),
        "status": str(raw.get("status", descriptor.status) or descriptor.status),
        "evidence_tier": str(raw.get("evidence_tier", "") or ""),
        "causal_confidence": _finite_float(raw.get("causal_confidence", 0.0)),
        "causal_validated": bool(raw.get("causal_validated", False)),
        "home_context_key": str(raw.get("home_context_key", "") or ""),
        "utility": _finite_float(raw.get("utility", 0.0)),
        "fade": _finite_float(raw.get("fade", 1.0), 1.0),
        "node_ids": list(node_ids),
        "edge_ids": [],
    }


def _append_graph(
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    groups: dict[str, dict[str, Any]],
    descriptor: GraphDescriptor,
    *,
    graph_count: int,
    active_nodes: set[str],
    active_edges: set[str],
) -> None:
    graph = descriptor.graph
    neuron_rows = list(getattr(graph, "neurons", {}).values())
    local_groups: dict[str, list[Any]] = {}
    for neuron in neuron_rows:
        local_module_id = str(getattr(neuron, "module_id", "") or "")
        if not local_module_id or local_module_id == "m1":
            local_module_id = "__core__"
        local_groups.setdefault(local_module_id, []).append(neuron)

    ordered_local_groups = sorted(local_groups, key=lambda item: (item != "__core__", item))
    group_index = {group_id: idx for idx, group_id in enumerate(ordered_local_groups)}
    region_counts: dict[tuple[str, str], int] = {}
    region_index: dict[tuple[str, str], int] = {}
    local_to_group: dict[str, str] = {}

    for local_module_id, members in local_groups.items():
        qualified_members = [_qualified_id(descriptor.graph_id, str(getattr(item, "id", ""))) for item in members if str(getattr(item, "id", ""))]
        raw_registry = getattr(graph, "specialist_modules", {})
        is_specialist = isinstance(raw_registry, Mapping) and local_module_id in raw_registry
        group_type = "specialist" if is_specialist else ("reasoning" if descriptor.graph_kind == "reasoning" else ("child" if descriptor.graph_kind == "child" else "core"))
        record = _group_record(
            graph=graph,
            descriptor=descriptor,
            local_module_id="" if local_module_id == "__core__" else local_module_id,
            node_ids=qualified_members,
            group_type=group_type,
        )
        groups[record["id"]] = record
        for neuron in members:
            local_to_group[str(getattr(neuron, "id", ""))] = str(record["id"])

    for neuron in neuron_rows:
        local_id = str(getattr(neuron, "id", ""))
        if not local_id:
            continue
        local_module_id = str(getattr(neuron, "module_id", "") or "")
        local_group = local_module_id if local_module_id and local_module_id != "m1" else "__core__"
        group_id = local_to_group[local_id]
        region = str(getattr(neuron, "region", "unknown"))
        key = (local_group, region)
        idx = region_index.get(key, 0)
        region_index[key] = idx + 1
        region_counts[key] = region_counts.get(key, 0) + 1

        # Count must be known before placement.  It is safe to derive from the
        # static group now because every neuron is already present.
        count = sum(1 for candidate in local_groups[local_group] if str(getattr(candidate, "region", "unknown")) == region)
        pos = _node_position(
            region,
            idx,
            count,
            graph_index=descriptor.graph_index,
            graph_count=max(1, graph_count),
            group_index=group_index[local_group],
            group_count=max(1, len(ordered_local_groups)),
        )
        activation = _finite_float(getattr(neuron, "activation", 0.0))
        state = _finite_float(getattr(neuron, "state", 0.0))
        is_active = local_id in active_nodes or abs(activation) > 1e-6 or abs(state) > 1e-6
        nodes.append({
            "id": _qualified_id(descriptor.graph_id, local_id),
            "local_id": local_id,
            "label": local_id,
            "graph_id": descriptor.graph_id,
            "graph_label": descriptor.label,
            "graph_kind": descriptor.graph_kind,
            "graph_status": descriptor.status,
            "module_group": group_id,
            "module_id": "" if local_group == "__core__" else local_module_id,
            "module_stage": str(getattr(neuron, "module_stage", "") or ""),
            "module_role": str(getattr(neuron, "module_role", "") or ""),
            "home_context_key": str(getattr(neuron, "home_context_key", "") or ""),
            "region": region,
            "kind": "neuron",
            "active": bool(is_active),
            "activation": activation,
            "state": state,
            "bias": _finite_float(getattr(neuron, "bias", 0.0)),
            "utility": _finite_float(getattr(neuron, "utility", 0.0)),
            "plasticity": _finite_float(getattr(neuron, "plasticity", 0.0)),
            "fade": _finite_float(getattr(neuron, "fade", 1.0), 1.0),
            "lifecycle": str(getattr(neuron, "lifecycle", "")),
            **pos,
        })

    for edge in list(getattr(graph, "edges", {}).values()):
        local_edge_id = str(getattr(edge, "id", ""))
        src_local = str(getattr(edge, "src", ""))
        dst_local = str(getattr(edge, "dst", ""))
        if not local_edge_id or not src_local or not dst_local or src_local not in local_to_group or dst_local not in local_to_group:
            continue
        src_group = local_to_group[src_local]
        dst_group = local_to_group[dst_local]
        src_neuron = getattr(graph, "neurons", {}).get(src_local)
        dst_neuron = getattr(graph, "neurons", {}).get(dst_local)
        src_region = str(getattr(src_neuron, "region", "unknown"))
        dst_region = str(getattr(dst_neuron, "region", "unknown"))
        internal_specialist = src_group == dst_group and bool(groups.get(src_group, {}).get("is_specialist"))
        interface_specialist = src_group != dst_group and (
            bool(groups.get(src_group, {}).get("is_specialist")) or bool(groups.get(dst_group, {}).get("is_specialist"))
        )
        qualified_edge = _qualified_id(descriptor.graph_id, local_edge_id)
        edges.append({
            "id": qualified_edge,
            "local_id": local_edge_id,
            "src": _qualified_id(descriptor.graph_id, src_local),
            "dst": _qualified_id(descriptor.graph_id, dst_local),
            "graph_id": descriptor.graph_id,
            "graph_label": descriptor.label,
            "graph_kind": descriptor.graph_kind,
            "src_group": src_group,
            "dst_group": dst_group,
            "src_module_id": str(getattr(edge, "src_module_id", "") or ""),
            "dst_module_id": str(getattr(edge, "dst_module_id", "") or ""),
            "module_role": str(getattr(edge, "module_role", "") or ""),
            "kind": str(getattr(edge, "kind", "forward")),
            "route": f"{src_region}→{dst_region}",
            "active": bool(local_edge_id in active_edges),
            "enabled": bool(getattr(edge, "enabled", True)),
            "weight": _finite_float(getattr(edge, "weight", 0.0)),
            "gate": _finite_float(getattr(edge, "gate", 0.0)),
            "utility": _finite_float(getattr(edge, "utility", 0.0)),
            "plasticity": _finite_float(getattr(edge, "plasticity", 0.0)),
            "fade": _finite_float(getattr(edge, "fade", 1.0), 1.0),
            "lifecycle": str(getattr(edge, "lifecycle", "")),
            "created_by": str(getattr(edge, "created_by", "") or ""),
            "need_reason": str(getattr(edge, "need_reason", "") or ""),
            "candidate_id": str(getattr(edge, "candidate_id", "") or ""),
            "is_module_internal": bool(internal_specialist),
            "is_module_interface": bool(interface_specialist),
        })
        if src_group == dst_group and src_group in groups:
            groups[src_group]["edge_ids"].append(qualified_edge)


def _update_group_bounds(groups: dict[str, dict[str, Any]], nodes: list[dict[str, Any]]) -> None:
    positions = {str(node["id"]): node for node in nodes}
    for group in groups.values():
        members = [positions[node_id] for node_id in group.get("node_ids", []) if node_id in positions]
        if not members:
            group["bounds"] = {}
            continue
        padding = 32.0 if group.get("is_specialist") else 18.0
        group["bounds"] = {
            "min_x": min(_finite_float(item.get("x")) for item in members) - padding,
            "max_x": max(_finite_float(item.get("x")) for item in members) + padding,
            "min_y": min(_finite_float(item.get("y")) for item in members) - padding,
            "max_y": max(_finite_float(item.get("y")) for item in members) + padding,
            "min_z": min(_finite_float(item.get("z")) for item in members) - padding,
            "max_z": max(_finite_float(item.get("z")) for item in members) + padding,
        }


def _annotate_connectivity(nodes: list[dict[str, Any]], edges: list[dict[str, Any]]) -> None:
    node_map = {str(node.get("id", "")): node for node in nodes}
    for node in nodes:
        node["degree_in"] = 0
        node["degree_out"] = 0
        node["active_degree_in"] = 0
        node["active_degree_out"] = 0
        node["_active_connections"] = []
    for edge in edges:
        src = str(edge.get("src", ""))
        dst = str(edge.get("dst", ""))
        kind = str(edge.get("kind", "edge"))
        edge_id = str(edge.get("id", ""))
        active = bool(edge.get("active"))
        if src in node_map:
            node_map[src]["degree_out"] += 1
            if active:
                node_map[src]["active_degree_out"] += 1
                node_map[src]["_active_connections"].append(f"out {edge_id} → {dst} ({kind})")
        if dst in node_map:
            node_map[dst]["degree_in"] += 1
            if active:
                node_map[dst]["active_degree_in"] += 1
                node_map[dst]["_active_connections"].append(f"in {edge_id} ← {src} ({kind})")
    for node in nodes:
        active_connections = list(node.pop("_active_connections", []))
        node["active_connections"] = active_connections[:24]
        if len(active_connections) > 24:
            node["active_connections_more"] = len(active_connections) - 24


def _edge_priority(edge: Mapping[str, Any]) -> tuple[float, float, float, str]:
    structural = 0.0
    if edge.get("active"):
        structural += 10_000.0
    if edge.get("is_module_internal"):
        structural += 8_000.0
    if edge.get("is_module_interface"):
        structural += 6_000.0
    if edge.get("enabled", True):
        structural += 10.0
    return (
        structural,
        abs(_finite_float(edge.get("weight", 0.0))) + abs(_finite_float(edge.get("utility", 0.0))) * 0.4,
        abs(_finite_float(edge.get("gate", 0.0))),
        str(edge.get("id", "")),
    )


def _balanced_edge_subset(edges: list[dict[str, Any]], max_edges: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    total = len(edges)
    if max_edges <= 0 or total <= max_edges:
        return list(edges), {"full_edges": total, "exported_edges": total, "truncated_edges": 0, "protected_edges": total}

    protected = [edge for edge in edges if edge.get("active") or edge.get("is_module_internal") or edge.get("is_module_interface")]
    selected: dict[str, dict[str, Any]] = {str(edge["id"]): edge for edge in protected}
    remaining = [edge for edge in edges if str(edge["id"]) not in selected]

    # Keep one or more representative synapses for every functional route so
    # dense input fan-in cannot consume the entire static snapshot.
    routes: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for edge in remaining:
        route_key = (str(edge.get("graph_id", "")), str(edge.get("route", "")), str(edge.get("kind", "")))
        routes.setdefault(route_key, []).append(edge)
    remaining_budget = max(0, max_edges - len(selected))
    route_keys = sorted(routes)
    per_route = max(1, remaining_budget // max(1, len(route_keys)))
    for route in route_keys:
        for edge in sorted(routes[route], key=_edge_priority, reverse=True)[:per_route]:
            if len(selected) >= max_edges:
                break
            selected[str(edge["id"])] = edge
        if len(selected) >= max_edges:
            break

    if len(selected) < max_edges:
        for edge in sorted(remaining, key=_edge_priority, reverse=True):
            if len(selected) >= max_edges:
                break
            selected.setdefault(str(edge["id"]), edge)

    ordered = sorted(selected.values(), key=_edge_priority, reverse=True)
    return ordered, {
        "full_edges": total,
        "exported_edges": len(ordered),
        "truncated_edges": max(0, total - len(ordered)),
        "protected_edges": len(protected),
    }


def _node_subset(nodes: list[dict[str, Any]], max_nodes: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    total = len(nodes)
    if max_nodes <= 0 or total <= max_nodes:
        return list(nodes), {"full_nodes": total, "exported_nodes": total, "truncated_nodes": 0}
    protected = [node for node in nodes if node.get("active") or node.get("module_id")]
    chosen: dict[str, dict[str, Any]] = {str(node["id"]): node for node in protected}
    for node in nodes:
        if len(chosen) >= max_nodes:
            break
        chosen.setdefault(str(node["id"]), node)
    selected = list(chosen.values())
    return selected, {
        "full_nodes": total,
        "exported_nodes": len(selected),
        "truncated_nodes": max(0, total - len(selected)),
    }


def export_model_view_data(
    policy_path: str,
    *,
    journal_path: str | None = None,
    policy_core: str = "current",
    max_nodes: int = 2500,
    max_edges: int = 12000,
) -> dict[str, Any]:
    policy = load_policy(policy_path, policy_core=policy_core)
    descriptors = _graph_descriptors(policy)
    journal_nodes, journal_edges, journal_by_graph, latest_journal = _read_latest_activity(journal_path)
    nodes: list[dict[str, Any]] = []
    edges_all: list[dict[str, Any]] = []
    groups: dict[str, dict[str, Any]] = {}

    first_m1 = next((descriptor.graph_id for descriptor in descriptors if descriptor.graph_kind == "m1"), "")
    for descriptor in descriptors:
        graph_nodes, graph_edges = _trace_activity_from_graph(descriptor.graph)
        journal_local = journal_by_graph.get(descriptor.graph_id, {})
        graph_journal_nodes = set(journal_local.get("nodes", set()))
        graph_journal_edges = set(journal_local.get("edges", set()))
        # Legacy journals contain one unqualified root trace.  Only apply it
        # to the first M1 graph; applying e0/e1 to every graph reintroduces
        # the collision that V22 is designed to remove.
        if descriptor.graph_id == first_m1:
            graph_journal_nodes |= _normalise_ids(journal_nodes, descriptor.graph_id)
            graph_journal_edges |= _normalise_ids(journal_edges, descriptor.graph_id)
        _append_graph(
            nodes,
            edges_all,
            groups,
            descriptor,
            graph_count=max(1, len(descriptors)),
            active_nodes=graph_nodes | graph_journal_nodes,
            active_edges=graph_edges | graph_journal_edges,
        )

    nodes, node_stats = _node_subset(nodes, int(max_nodes))
    node_ids = {str(node["id"]) for node in nodes}
    edges_all = [edge for edge in edges_all if str(edge.get("src", "")) in node_ids and str(edge.get("dst", "")) in node_ids]
    edges, edge_stats = _balanced_edge_subset(edges_all, int(max_edges))
    _update_group_bounds(groups, nodes)
    _annotate_connectivity(nodes, edges)

    edge_kind_counts: dict[str, int] = {}
    active_edge_kind_counts: dict[str, int] = {}
    for edge in edges:
        kind = str(edge.get("kind", "edge"))
        edge_kind_counts[kind] = edge_kind_counts.get(kind, 0) + 1
        if edge.get("active"):
            active_edge_kind_counts[kind] = active_edge_kind_counts.get(kind, 0) + 1

    graph_summaries: list[dict[str, Any]] = []
    for descriptor in descriptors:
        graph_nodes = [node for node in nodes if node.get("graph_id") == descriptor.graph_id]
        graph_edges = [edge for edge in edges if edge.get("graph_id") == descriptor.graph_id]
        graph_summaries.append({
            "graph_id": descriptor.graph_id,
            "label": descriptor.label,
            "kind": descriptor.graph_kind,
            "status": descriptor.status,
            "selectable": descriptor.selectable,
            "node_count": len(graph_nodes),
            "edge_count": len(graph_edges),
            "active_nodes": sum(1 for node in graph_nodes if node.get("active")),
            "active_edges": sum(1 for edge in graph_edges if edge.get("active")),
            "specialist_modules": sum(1 for group in groups.values() if group.get("graph_id") == descriptor.graph_id and group.get("is_specialist")),
        })

    group_list = sorted(groups.values(), key=lambda item: (str(item.get("graph_id", "")), not bool(item.get("is_specialist")), str(item.get("label", ""))))
    specialist_groups = [group for group in group_list if group.get("is_specialist")]
    return {
        "schema_version": 2,
        "policy_path": str(policy_path),
        "policy_core": str(policy_core),
        "m1_backend": str(getattr(policy, "m1_backend", "direct_rnn")),
        "revision": int(getattr(policy, "revision", 0) or 0),
        "journal_path": str(journal_path or ""),
        "latest_journal": latest_journal,
        "namespaced_ids": True,
        "summary": {
            "nodes": len(nodes),
            "edges": len(edges),
            "active_nodes": sum(1 for node in nodes if node.get("active")),
            "active_edges": sum(1 for edge in edges if edge.get("active")),
            "graphs": len(graph_summaries),
            "specialist_modules": len(specialist_groups),
            "child_graphs": sum(1 for item in graph_summaries if item.get("kind") == "child"),
            "edge_kinds": edge_kind_counts,
            "active_edge_kinds": active_edge_kind_counts,
            **node_stats,
            **edge_stats,
        },
        "graphs": graph_summaries,
        "module_groups": group_list,
        "specialist_modules": specialist_groups,
        "nodes": nodes,
        "edges": edges,
    }


def _viewer_html(data: dict[str, Any]) -> str:
    payload = _safe_json(data)
    title = html.escape(Path(str(data.get("policy_path", "model"))).name)
    template = r'''<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Open 3D Model V22 - __TITLE__</title>
<style>
:root{color-scheme:dark;--bg:#090b10;--panel:#111720;--line:#2d3a4e;--text:#e8edf3;--muted:#90a0b5;--accent:#60a5fa;--active:#34d399;--special:#f59e0b;--warn:#f97316;}
*{box-sizing:border-box}body{margin:0;overflow:hidden;background:var(--bg);color:var(--text);font:13px/1.45 "Segoe UI",Inter,Arial,sans-serif}
#canvas{position:fixed;inset:0;width:100vw;height:100vh;display:block;cursor:grab}#canvas:active{cursor:grabbing}
.top{position:fixed;left:18px;top:14px;right:390px;display:flex;gap:18px;align-items:flex-start;pointer-events:none}.brand{font-size:16px;font-weight:700;letter-spacing:.2px}.meta{color:var(--muted);margin-top:3px}.stats{display:flex;gap:10px;flex-wrap:wrap;color:var(--muted)}.stat{border-left:1px solid var(--line);padding-left:10px;min-width:72px}.stat b{display:block;color:var(--text);font-size:15px}
.side{position:fixed;top:0;right:0;width:372px;height:100vh;background:rgba(12,17,24,.95);border-left:1px solid var(--line);backdrop-filter:blur(10px);padding:14px;overflow:auto}.section{border-top:1px solid var(--line);padding-top:12px;margin-top:12px}.label{color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}.row{display:flex;gap:8px;align-items:center;margin:8px 0}.row>*{flex:1}button,select,input{background:#0b111a;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px 8px}button{cursor:pointer}button.active{border-color:var(--active);color:var(--active)}select{max-width:100%}input[type=range]{padding:0}.list{max-height:210px;overflow:auto;border:1px solid var(--line);border-radius:6px}.item{padding:7px 8px;border-bottom:1px solid rgba(45,58,78,.60);cursor:pointer}.item:hover{background:#182132}.item.active{color:var(--active)}.item.special{border-left:3px solid var(--special)}.kv{display:grid;grid-template-columns:116px 1fr;gap:5px 8px;color:var(--muted);word-break:break-word}.kv b{color:var(--text);font-weight:500}.hint{position:fixed;left:18px;bottom:14px;right:390px;color:var(--muted);background:rgba(12,17,24,.74);border:1px solid var(--line);border-radius:7px;padding:8px 10px}.tiny{color:var(--muted);font-size:11px}.pill{display:inline-block;border:1px solid var(--line);border-radius:10px;padding:1px 6px;margin-right:4px;font-size:11px}.warning{color:#fdba74}.check{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:12px}.check input{flex:0}
</style>
</head>
<body>
<canvas id="canvas"></canvas>
<div class="top"><div><div class="brand">Self-Growing Trader Model 3D · V22</div><div class="meta" id="meta"></div></div><div class="stats" id="stats"></div></div>
<aside class="side">
  <div class="label">Snapshot controls</div>
  <div class="row"><button id="reset">Reset view</button><button id="fit">Fit selection</button></div>
  <div class="row"><select id="viewMode"><option value="overview">Tổng quan</option><option value="module">Focus module</option><option value="active">Chỉ causal trace</option></select></div>
  <div class="row"><select id="graphFilter"><option value="">Tất cả graph</option></select></div>
  <div class="row"><select id="moduleFilter"><option value="">Tất cả module</option></select></div>
  <div class="row"><select id="kindFilter"><option value="">Tất cả loại liên kết</option></select></div>
  <div class="row"><label class="tiny" for="edgeBudget">Liên kết render: <b id="edgeBudgetValue"></b></label><input id="edgeBudget" type="range" min="200" max="1000" step="100"></div>
  <div class="row"><label class="check"><input id="showBoxes" type="checkbox" checked> Khung module/circuit</label><label class="check"><input id="showLabels" type="checkbox" checked> Nhãn specialist</label></div>
  <div class="section"><div class="label">Selection</div><div id="selection" class="kv"><span>Click</span><b>một node, edge hoặc module</b></div></div>
  <div class="section"><div class="label">Graphs</div><div id="graphs" class="list"></div></div>
  <div class="section"><div class="label">Specialist circuits</div><div id="specialists" class="list"></div></div>
  <div class="section"><div class="label">Active trace</div><div id="activeTrace" class="list"></div></div>
  <div class="section"><div class="label">Export status</div><div id="exportStatus" class="tiny"></div></div>
</aside>
<div class="hint">Kéo để xoay · lăn chuột để zoom · chọn specialist để xem circuit và interface edge · viewer chỉ đọc snapshot, không can thiệp train</div>
<script id="model-data" type="application/json">__PAYLOAD__</script>
<script>
const DATA=JSON.parse(document.getElementById('model-data').textContent);
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
const nodes=DATA.nodes||[],edges=DATA.edges||[],graphs=DATA.graphs||[],groups=DATA.module_groups||[];
const byId=new Map(nodes.map(n=>[n.id,n])),groupById=new Map(groups.map(g=>[g.id,g]));
let W=1,H=1,rx=-0.48,ry=0.72,zoom=1.0,cam={x:0,y:0,z:0},selected=null,dragging=false,lastX=0,lastY=0,drawQueued=false;
let viewMode='overview',graphFilter='',moduleFilter='',kindFilter='',showBoxes=true,showLabels=true;
const maxEdge=Math.max(200,Math.min(Math.max(200,edges.length),12000));const edgeBudget=document.getElementById('edgeBudget');edgeBudget.max=String(maxEdge);edgeBudget.value=String(Math.min(maxEdge,Math.max(1100,Math.min(1800,edges.length))));
const groupLabel=g=>g.is_specialist?`${g.role||'specialist'} · ${g.status||'candidate'}`:g.label;
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));}
function resize(){W=canvas.width=innerWidth*devicePixelRatio;H=canvas.height=innerHeight*devicePixelRatio;canvas.style.width=innerWidth+'px';canvas.style.height=innerHeight+'px';requestDraw();}addEventListener('resize',resize);resize();
document.getElementById('meta').textContent=`${DATA.policy_path} · backend=${DATA.m1_backend} · revision=${DATA.revision} · namespace=${DATA.namespaced_ids?'on':'off'}`;
document.getElementById('stats').innerHTML=Object.entries(DATA.summary||{}).filter(([k,v])=>typeof v!=='object'&&['nodes','edges','active_nodes','active_edges','graphs','specialist_modules','child_graphs'].includes(k)).map(([k,v])=>`<span class="stat"><b>${esc(v)}</b>${esc(k.replaceAll('_',' '))}</span>`).join('');
function fillSelects(){const gs=document.getElementById('graphFilter'),ms=document.getElementById('moduleFilter'),ks=document.getElementById('kindFilter');for(const g of graphs){const o=document.createElement('option');o.value=g.graph_id;o.textContent=`${g.label} · ${g.kind} · ${g.node_count}n`;gs.appendChild(o);}for(const g of groups){const o=document.createElement('option');o.value=g.id;o.textContent=`${g.is_specialist?'◆ ':''}${groupLabel(g)} · ${g.node_ids.length} node`;ms.appendChild(o);}for(const k of [...new Set(edges.map(e=>e.kind||'edge'))].sort()){const o=document.createElement('option');o.value=k;o.textContent=k;ks.appendChild(o);}}fillSelects();
function requestDraw(){if(drawQueued)return;drawQueued=true;requestAnimationFrame(()=>{drawQueued=false;draw();});}
function project(p){const sx=Math.sin(rx),cx=Math.cos(rx),sy=Math.sin(ry),cy=Math.cos(ry);let x=(p.x||0)-cam.x,y=(p.y||0)-cam.y,z=(p.z||0)-cam.z;let x1=x*cy-z*sy,z1=x*sy+z*cy;let y1=y*cx-z1*sx,z2=y*sx+z1*cx;const scale=(650*zoom)/(650+z2);return{x:W*.42+x1*scale*devicePixelRatio,y:H*.52+y1*scale*devicePixelRatio,z:z2,s:scale};}
function focusGroup(){return viewMode==='module'&&moduleFilter?moduleFilter:'';}
function baseEdges(){const focus=focusGroup();return edges.filter(e=>(!graphFilter||e.graph_id===graphFilter)&&(!kindFilter||e.kind===kindFilter)&&(!focus||e.src_group===focus||e.dst_group===focus)&&(!viewMode||viewMode!=='active'||e.active)&&byId.has(e.src)&&byId.has(e.dst));}
function currentEdges(){const source=baseEdges(),budget=Number(edgeBudget.value);const priority=e=>(e.active?10000:0)+(e.is_module_internal?8000:0)+(e.is_module_interface?6000:0)+Math.abs(e.weight||0)*20+Math.abs(e.utility||0)*5;return source.slice().sort((a,b)=>priority(b)-priority(a)).slice(0,budget);}
function currentNodes(){const list=nodes.filter(n=>(!graphFilter||n.graph_id===graphFilter));const focus=focusGroup();const linked=new Set();for(const e of currentEdges()){linked.add(e.src);linked.add(e.dst);}if(focus)return list.filter(n=>n.module_group===focus||linked.has(n.id));if(viewMode==='active')return list.filter(n=>n.active||linked.has(n.id));return list;}
function regionColor(r){return({input:'#94a3b8',encoder:'#60a5fa',memory:'#a78bfa',hypothesis:'#f472b6',predictor:'#22d3ee',critic:'#f59e0b',selector:'#fb7185',meta:'#c084fc',output:'#f8fafc'})[r]||'#7dd3fc';}
function edgeColor(e){if(e.active)return'rgba(52,211,153,.95)';if(e.is_module_internal)return'rgba(245,158,11,.65)';if(e.is_module_interface)return'rgba(251,146,60,.56)';if(!e.enabled)return'rgba(95,105,120,.14)';if((e.weight||0)<0)return'rgba(248,113,113,.32)';return'rgba(96,165,250,.28)';}
function cubeCorners(b){return[{x:b.min_x,y:b.min_y,z:b.min_z},{x:b.max_x,y:b.min_y,z:b.min_z},{x:b.max_x,y:b.max_y,z:b.min_z},{x:b.min_x,y:b.max_y,z:b.min_z},{x:b.min_x,y:b.min_y,z:b.max_z},{x:b.max_x,y:b.min_y,z:b.max_z},{x:b.max_x,y:b.max_y,z:b.max_z},{x:b.min_x,y:b.max_y,z:b.max_z}];}
function drawBoxes(){if(!showBoxes)return;const focus=focusGroup();for(const g of groups){if(!g.bounds||!Object.keys(g.bounds).length)continue;if(graphFilter&&g.graph_id!==graphFilter)continue;if(focus&&g.id!==focus)continue;if(!g.is_specialist&&g.graph_kind!=='child')continue;const pts=cubeCorners(g.bounds).map(project);const pairs=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];ctx.strokeStyle=g.is_specialist?'rgba(245,158,11,.72)':'rgba(96,165,250,.42)';ctx.lineWidth=1.2*devicePixelRatio;for(const [a,b] of pairs){ctx.beginPath();ctx.moveTo(pts[a].x,pts[a].y);ctx.lineTo(pts[b].x,pts[b].y);ctx.stroke();}if(showLabels){const c=project({x:(g.bounds.min_x+g.bounds.max_x)/2,y:(g.bounds.min_y+g.bounds.max_y)/2,z:(g.bounds.min_z+g.bounds.max_z)/2});ctx.font=`${11*devicePixelRatio}px Segoe UI`;ctx.fillStyle=g.is_specialist?'rgba(253,230,138,.96)':'rgba(191,219,254,.90)';ctx.fillText(`${g.is_specialist?'◆ ':''}${groupLabel(g)} (${g.node_ids.length})`,c.x+6*devicePixelRatio,c.y-6*devicePixelRatio);}}}
function draw(){ctx.clearRect(0,0,W,H);const es=currentEdges().map(e=>[e,project(byId.get(e.src)),project(byId.get(e.dst))]).sort((a,b)=>a[1].z+a[2].z-b[1].z-b[2].z);ctx.lineCap='round';for(const [e,a,b] of es){const width=(e.active?2.5:(e.is_module_internal?1.5:.55+Math.min(1.6,Math.abs(e.weight||0)*1.2)))*devicePixelRatio;ctx.strokeStyle=edgeColor(e);ctx.lineWidth=width;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}drawBoxes();const ns=currentNodes().map(n=>[n,project(n)]).sort((a,b)=>a[1].z-b[1].z);for(const [n,p] of ns){const specialist=groupById.get(n.module_group)?.is_specialist;const r=(n.active?5.5:(specialist?4.7:3.2))*Math.max(.52,p.s)*devicePixelRatio;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=n.active?'rgba(52,211,153,.95)':(specialist?'rgba(245,158,11,.92)':regionColor(n.region));ctx.fill();if(selected&&selected.id===n.id){ctx.strokeStyle='white';ctx.lineWidth=2*devicePixelRatio;ctx.stroke();}}}
function formatValue(v){if(Array.isArray(v))return v.length?v.map(x=>`<div>${esc(x)}</div>`).join(''):'';if(v&&typeof v==='object')return Object.entries(v).map(([k,val])=>`${esc(k)}=${esc(val)}`).join(', ');return esc(v);}
function showSelection(item){selected=item;document.getElementById('selection').innerHTML=Object.entries(item).filter(([k])=>!['x','y','z','bounds','node_ids','edge_ids'].includes(k)).map(([k,v])=>`<span>${esc(k)}</span><b>${formatValue(v)}</b>`).join('');requestDraw();}
function setFocus(id){moduleFilter=id||'';document.getElementById('moduleFilter').value=moduleFilter;viewMode=moduleFilter?'module':'overview';document.getElementById('viewMode').value=viewMode;const g=groupById.get(moduleFilter);if(g&&g.bounds){cam={x:(g.bounds.min_x+g.bounds.max_x)/2,y:(g.bounds.min_y+g.bounds.max_y)/2,z:(g.bounds.min_z+g.bounds.max_z)/2};zoom=1.22;showSelection(g);}requestDraw();}
function lineDist(px,py,x1,y1,x2,y2){const A=px-x1,B=py-y1,C=x2-x1,D=y2-y1,dot=A*C+B*D,len=C*C+D*D,t=Math.max(0,Math.min(1,dot/(len||1)));return Math.hypot(px-(x1+t*C),py-(y1+t*D));}
canvas.addEventListener('pointerdown',e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointermove',e=>{if(!dragging)return;ry+=(e.clientX-lastX)*.006;rx+=(e.clientY-lastY)*.006;lastX=e.clientX;lastY=e.clientY;requestDraw();});canvas.addEventListener('pointerup',()=>dragging=false);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=Math.max(.25,Math.min(4.5,zoom*(e.deltaY>0?.92:1.08)));requestDraw();},{passive:false});
canvas.addEventListener('click',e=>{const mx=e.clientX*devicePixelRatio,my=e.clientY*devicePixelRatio;let best=null,bd=Infinity;for(const n of currentNodes()){const p=project(n),d=Math.hypot(mx-p.x,my-p.y);if(d<bd&&d<15*devicePixelRatio){best=n;bd=d;}}if(!best){for(const ed of currentEdges()){const a=project(byId.get(ed.src)),b=project(byId.get(ed.dst)),d=lineDist(mx,my,a.x,a.y,b.x,b.y);if(d<bd&&d<8*devicePixelRatio){best=ed;bd=d;}}}if(best)showSelection(best);});
document.getElementById('reset').onclick=()=>{rx=-.48;ry=.72;zoom=1;cam={x:0,y:0,z:0};requestDraw();};document.getElementById('fit').onclick=()=>{if(moduleFilter)setFocus(moduleFilter);else{zoom=1;cam={x:0,y:0,z:0};requestDraw();}};
document.getElementById('viewMode').onchange=e=>{viewMode=e.target.value;requestDraw();};document.getElementById('graphFilter').onchange=e=>{graphFilter=e.target.value;requestDraw();};document.getElementById('moduleFilter').onchange=e=>setFocus(e.target.value);document.getElementById('kindFilter').onchange=e=>{kindFilter=e.target.value;requestDraw();};edgeBudget.oninput=()=>{document.getElementById('edgeBudgetValue').textContent=edgeBudget.value;requestDraw();};document.getElementById('edgeBudgetValue').textContent=edgeBudget.value;document.getElementById('showBoxes').onchange=e=>{showBoxes=e.target.checked;requestDraw();};document.getElementById('showLabels').onchange=e=>{showLabels=e.target.checked;requestDraw();};
function fillLists(){const ge=document.getElementById('graphs');ge.innerHTML=graphs.map(g=>`<div class="item" data-graph="${esc(g.graph_id)}"><b>${esc(g.label)}</b><br><small>${esc(g.kind)} · ${g.node_count} node · ${g.edge_count} edge · ${g.specialist_modules||0} specialist · ${esc(g.status||'')}</small></div>`).join('')||'<div class="item">Không có graph</div>';ge.querySelectorAll('[data-graph]').forEach(el=>el.onclick=()=>{graphFilter=el.dataset.graph;document.getElementById('graphFilter').value=graphFilter;requestDraw();});const se=document.getElementById('specialists');const specs=(DATA.specialist_modules||[]);se.innerHTML=specs.map(g=>`<div class="item special" data-module="${esc(g.id)}"><b>${esc(g.role||g.label)}</b><br><small>${esc(g.status)} · ${g.node_ids.length} node · ${g.edge_ids.length} internal edge · utility=${Number(g.utility||0).toFixed(3)} · causal=${Number(g.causal_confidence||0).toFixed(2)}</small></div>`).join('')||'<div class="item">Chưa có specialist module đã lưu trong policy này.</div>';se.querySelectorAll('[data-module]').forEach(el=>el.onclick=()=>setFocus(el.dataset.module));const activeNodes=nodes.filter(n=>n.active).slice(0,80),activeEdges=edges.filter(e=>e.active).slice(0,80);const ae=document.getElementById('activeTrace');ae.innerHTML=[...activeNodes.map(n=>`<div class="item" data-node="${esc(n.id)}">${esc(n.label)}<br><small>${esc(n.graph_id)} · ${esc(n.region)} · ${esc(n.module_stage||'')}</small></div>`),...activeEdges.map(e=>`<div class="item" data-edge="${esc(e.id)}">${esc(e.local_id)} · ${esc(e.kind)}<br><small>${esc(e.src)} → ${esc(e.dst)}</small></div>`)].join('')||'<div class="item">Không có trace hoạt hóa trong snapshot/journal.</div>';ae.querySelectorAll('[data-node]').forEach(el=>el.onclick=()=>showSelection(byId.get(el.dataset.node)));ae.querySelectorAll('[data-edge]').forEach(el=>el.onclick=()=>showSelection(edges.find(x=>x.id===el.dataset.edge)));const s=DATA.summary||{};document.getElementById('exportStatus').innerHTML=`Xuất ${s.exported_nodes||s.nodes}/${s.full_nodes||s.nodes} node và ${s.exported_edges||s.edges}/${s.full_edges||s.edges} edge.${(s.truncated_edges||0)>0?` <span class="warning">Đã route-balance và cắt ${s.truncated_edges} edge không ưu tiên.</span>`:' Không cắt edge.'}<br>Specialist internal/interface edge luôn được ưu tiên giữ khi có giới hạn.`;}fillLists();requestDraw();
</script>
</body>
</html>'''
    return template.replace("__PAYLOAD__", payload).replace("__TITLE__", title)


def export_model_viewer_html(
    policy_path: str,
    output_path: str,
    *,
    journal_path: str | None = None,
    policy_core: str = "current",
    max_nodes: int = 2500,
    max_edges: int = 12000,
) -> dict[str, Any]:
    data = export_model_view_data(
        policy_path,
        journal_path=journal_path,
        policy_core=policy_core,
        max_nodes=max_nodes,
        max_edges=max_edges,
    )
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_viewer_html(data), encoding="utf-8")
    return {
        "output": str(out),
        "policy": str(policy_path),
        "journal": str(journal_path or ""),
        "summary": data["summary"],
    }
