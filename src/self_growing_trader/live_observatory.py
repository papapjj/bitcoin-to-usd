"""Live, read-only neural telemetry and local browser observatory.

The trainer writes append-only JSONL events.  A separate local HTTP server reads
those events and renders the graph in a browser.  The observer has no callback
into policy selection, plasticity, growth, or MT5 execution; it is deliberately
one-way so enabling visualization cannot change a candidate's decisions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass, field, is_dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterable, Mapping

from .nhat_ky_giai_thich import xay_ho_so_quyet_dinh, xay_ho_so_phat_trien_cau_truc
from urllib.parse import urlparse

import numpy as np

# Schema 2 adds bounded synapse-state deltas.  Topology still remains the
# canonical full snapshot; state deltas let a dashboard show live
# weight/gate/utility/fade changes without serialising the whole graph after
# every realised outcome.
LIVE_SCHEMA_VERSION = 10
DEFAULT_EVENT_PATH = "outputs/neural_live.jsonl"

# V9 audit mode: 0 means no writer quota.  Full topology/activity/state data
# is preserved.  Physical RAM, disk and JVM capacity are the only boundaries.
TOPOLOGY_JSON_ARRAY_LIMIT = 0
EVENT_JSON_ARRAY_LIMIT = 0


@dataclass(frozen=True)
class LiveObservabilityConfig:
    """Runtime-only telemetry controls.

    This config controls observation bandwidth, not graph capacity or learning.
    ``decision_sample_every=1`` records every decision request; larger values
    reduce event volume without changing the network.
    """

    enabled: bool = False
    event_path: str = DEFAULT_EVENT_PATH
    decision_sample_every: int = 1
    # Full network snapshots are expensive but lifecycle/structural audit must
    # stay complete.  This limits only feedback-phase snapshots; the compact
    # feedback events themselves, lifecycle rows and growth rows are retained.
    network_development_feedback_sample_every: int = 1
    # 0 = không giới hạn/cắt telemetry. V9 ưu tiên audit đầy đủ; người dùng
    # tự chọn tốc độ render ở UI, không được mất dữ liệu ở writer.
    max_event_bytes: int = 0
    max_active_nodes: int = 0
    max_active_edges: int = 0
    max_state_nodes: int = 0
    max_state_edges: int = 0
    state_change_epsilon: float = 1e-6
    # Nhật ký audit tách khỏi telemetry topology để có thể đọc bằng công cụ/AI
    # mà không làm viewer phải nạp cả hồ sơ giải thích lớn ở mỗi tick.
    decision_profile_path: str = ""
    structural_event_path: str = ""
    network_development_path: str = ""
    max_responsible_neurons: int = 0
    max_responsible_edges: int = 0
    reset_on_process_start: bool = False
    # V10 tách log theo root_run_id. UI đọc active_run.json nên người dùng vẫn
    # chọn một đường dẫn gốc, nhưng telemetry của các run không trộn vào nhau.
    scope_paths_by_run: bool = False

    def __post_init__(self) -> None:
        if int(self.decision_sample_every) < 1:
            raise ValueError("observability.decision_sample_every must be at least 1")
        if int(self.network_development_feedback_sample_every) < 1:
            raise ValueError("observability.network_development_feedback_sample_every must be at least 1")
        if int(self.max_event_bytes) < 0:
            raise ValueError("observability.max_event_bytes must be >= 0 (0 = không compact log)")
        if int(self.max_active_nodes) < 0:
            raise ValueError("observability.max_active_nodes must be >= 0 (0 = full)")
        if int(self.max_active_edges) < 0:
            raise ValueError("observability.max_active_edges must be >= 0 (0 = full)")
        if int(self.max_state_nodes) < 0:
            raise ValueError("observability.max_state_nodes must be >= 0 (0 = full)")
        if int(self.max_state_edges) < 0:
            raise ValueError("observability.max_state_edges must be >= 0 (0 = full)")
        if float(self.state_change_epsilon) < 0.0:
            raise ValueError("observability.state_change_epsilon must be non-negative")
        if int(self.max_responsible_neurons) < 0:
            raise ValueError("observability.max_responsible_neurons must be >= 0 (0 = full)")
        if int(self.max_responsible_edges) < 0:
            raise ValueError("observability.max_responsible_edges must be >= 0 (0 = full)")


_PROCESS_RESET_PATHS: set[str] = set()
_PROCESS_RESET_LOCK = threading.Lock()
_WRITE_LOCKS: dict[str, threading.Lock] = {}
_WRITE_LOCKS_GUARD = threading.Lock()
# V11: nhiều policy owner có writer riêng nhưng cùng ghi một root run. Registry
# process-wide ngăn full topology bị lặp ở mọi decision/copy; key có policy
# instance + topology id nên không trộn topology giữa các bộ não độc lập.
_DEVELOPMENT_SNAPSHOT_REGISTRY: dict[str, tuple[str, int]] = {}
_DEVELOPMENT_SNAPSHOT_LOCK = threading.Lock()


def _safe_run_component(value: object) -> str:
    raw = str(value or "run")
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in raw)
    return safe[:128] or "run"


def _write_lock(path: Path) -> threading.Lock:
    key = str(path.resolve())
    with _WRITE_LOCKS_GUARD:
        lock = _WRITE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WRITE_LOCKS[key] = lock
        return lock


def _finite(value: object, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(out) if math.isfinite(out) else float(default)


def _json_safe(value: Any, *, max_array: int = 0, depth: int = 0) -> Any:
    """Chuyển dữ liệu telemetry sang JSON mà không cắt khi ``max_array=0``.

    V9 dùng ``0`` như chế độ audit đầy đủ. Không có giới hạn hiển thị ẩn ở
    writer; UI có thể render chậm với graph rất lớn nhưng vẫn nhận đủ dữ liệu.
    """
    # Scalar evidence remains meaningful even inside a deeply nested live
    # event (for example ``activity -> node -> controller_inputs -> risk``).
    # Apply the structural depth guard only after primitive conversion; the old
    # order turned those real values into ``<depth-limited>`` strings.
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return _finite(value)
    if isinstance(value, np.generic):
        return _json_safe(value.item(), max_array=max_array, depth=depth + 1)
    # Không cắt theo độ sâu trong chế độ audit đầy đủ. Mọi object do telemetry
    # phát ra đã là dict/list không tuần hoàn từ policy.
    if isinstance(value, np.ndarray):
        flat = np.asarray(value, dtype=float).reshape(-1)
        truncated = flat if int(max_array) <= 0 else flat[:int(max_array)]
        return {
            "values": [_finite(item) for item in truncated.tolist()],
            "size": int(flat.size),
            "truncated": bool(int(max_array) > 0 and flat.size > int(max_array)),
        }
    if is_dataclass(value):
        return _json_safe(asdict(value), max_array=max_array, depth=depth + 1)
    if isinstance(value, Mapping):
        # Những action legacy chỉ còn phục vụ tương thích nội bộ. Chúng từng
        # xuất hiện như sentinel -1 trong UI/log và gây hiểu nhầm; telemetry
        # V10 chỉ được công bố semantic/emitted/wire action.
        legacy_action_keys = {
            # Numeric legacy fields can carry sentinel -1 and must never leak
            # into V10 audit/UI. Text compatibility names are retained only
            # for old bridge-heartbeat readers; decision profiles never use
            # them as a decision source.
            "policy_action", "legal_action", "policy_action_index", "legal_action_index",
            "hanh_dong_policy", "hanh_dong_hop_le",
        }
        items = [(key, item) for key, item in value.items() if str(key) not in legacy_action_keys]
        if int(max_array) > 0:
            items = items[:int(max_array)]
        return {
            str(key): _json_safe(item, max_array=max_array, depth=depth + 1)
            for key, item in items
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        values = list(value)
        if int(max_array) > 0:
            values = values[:int(max_array)]
        return [_json_safe(item, max_array=max_array, depth=depth + 1) for item in values]
    return str(value)


def _vector(value: Any, limit: int = 0) -> list[float]:
    try:
        array = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return []
    selected = array if int(limit) <= 0 else array[:max(0, int(limit))]
    return [_finite(item) for item in selected.tolist()]


def _action_name(index: object, fallback: str = "") -> str:
    names = (
        "HOLD", "MARKET_BUY", "MARKET_SELL", "BUY_LIMIT", "SELL_LIMIT",
        "BUY_STOP", "SELL_STOP", "CANCEL_ORDERS", "CLOSE_WORST",
        "CLOSE_BEST", "CLOSE_ALL", "MANAGE_POSITION",
    )
    try:
        value = int(float(index))
    except (TypeError, ValueError):
        return str(fallback or "")
    if 0 <= value < len(names):
        return names[value]
    return str(fallback or value)


def _main_graphs(policy: Any) -> list[tuple[str, Any]]:
    """Return every graph that can exist in a policy, including child networks.

    Earlier live telemetry only exposed the root M1 and reasoning graph.  A
    true child supernetwork could therefore be spawned but remain invisible in
    the viewer.  Child candidates are still read-only here; selectability is
    governed by their own promotion gate.
    """
    graphs: list[tuple[str, Any]] = []
    for index, module in enumerate(getattr(policy, "modules", []) or []):
        m1 = getattr(module, "m1_component", None)
        graph = getattr(m1, "graph", None)
        if graph is not None and hasattr(graph, "neurons") and hasattr(graph, "edges"):
            graphs.append((str(getattr(module, "name", f"module_{index}")), graph))
    reasoning = getattr(getattr(policy, "reasoning_graph", None), "graph", None)
    if reasoning is not None and hasattr(reasoning, "neurons") and hasattr(reasoning, "edges"):
        graphs.append(("reasoning_graph", reasoning))
    manager = getattr(policy, "child_supernetwork_manager", None)
    children = getattr(manager, "children", {}) if manager is not None else {}
    if isinstance(children, Mapping):
        for network_id, record in sorted(children.items(), key=lambda item: str(item[0])):
            graph = getattr(getattr(record, "network", None), "graph", None)
            if graph is not None and hasattr(graph, "neurons") and hasattr(graph, "edges"):
                graphs.append((f"child:{network_id}", graph))
    return graphs


def _graph_id(module_name: str) -> str:
    """Stable telemetry namespace for graphs that reuse local e0/e1 IDs."""
    name = str(module_name or "graph")
    if name == "reasoning_graph":
        return "reasoning_graph"
    if name.startswith("child:"):
        return name
    return f"m1:{name}"


def _qualified_id(graph_id: str, local_id: str) -> str:
    return f"{graph_id}::{local_id}"


def _graph_registry(policy: Any) -> dict[str, tuple[str, Any]]:
    return {
        _graph_id(module_name): (str(module_name), graph)
        for module_name, graph in _main_graphs(policy)
    }


def graph_topology(policy: Any) -> dict[str, Any]:
    """A full structural snapshot emitted only at start/topology changes.

    ``NeuralGraph`` edge identifiers are local to a graph.  M1 and the
    reasoning graph both commonly contain ``e0`` … ``eN``.  Every telemetry
    identifier is therefore namespaced as ``graph_id::local_id`` before it
    leaves Python.  The local ID is retained for readable inspection.
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    graph_counts: dict[str, dict[str, Any]] = {}
    for module_name, graph in _main_graphs(policy):
        graph_id = _graph_id(module_name)
        graph_counts[graph_id] = {
            "graph_id": graph_id,
            "label": str(module_name),
            "node_count": int(len(getattr(graph, "neurons", {}) or {})),
            "edge_count": int(len(getattr(graph, "edges", {}) or {})),
        }
        for neuron in getattr(graph, "neurons", {}).values():
            local_id = str(getattr(neuron, "id", ""))
            if not local_id:
                continue
            node_id = _qualified_id(graph_id, local_id)
            nodes.append({
                "id": node_id,
                "local_id": local_id,
                "graph_id": graph_id,
                "module": str(module_name),
                "region": str(getattr(neuron, "region", "unknown")),
                "lifecycle": str(getattr(neuron, "lifecycle", "")),
                "bias": _finite(getattr(neuron, "bias", 0.0)),
                "utility": _finite(getattr(neuron, "utility", 0.0)),
                "plasticity": _finite(getattr(neuron, "plasticity", 0.0)),
                "fade": _finite(getattr(neuron, "fade", 1.0), 1.0),
                "age": int(_finite(getattr(neuron, "age", 0), 0.0)),
                "module_id": str(getattr(neuron, "module_id", "") or ""),
                "module_stage": str(getattr(neuron, "module_stage", "") or ""),
                "module_role": str(getattr(neuron, "module_role", "") or ""),
                "home_context_key": str(getattr(neuron, "home_context_key", "") or ""),
            })
        for edge in getattr(graph, "edges", {}).values():
            local_id = str(getattr(edge, "id", ""))
            src_local = str(getattr(edge, "src", ""))
            dst_local = str(getattr(edge, "dst", ""))
            if not local_id or not src_local or not dst_local:
                continue
            edge_id = _qualified_id(graph_id, local_id)
            edges.append({
                "id": edge_id,
                "local_id": local_id,
                "graph_id": graph_id,
                "module": str(module_name),
                "src": _qualified_id(graph_id, src_local),
                "dst": _qualified_id(graph_id, dst_local),
                "kind": str(getattr(edge, "kind", "forward")),
                "enabled": bool(getattr(edge, "enabled", True)),
                "weight": _finite(getattr(edge, "weight", 0.0)),
                "gate": _finite(getattr(edge, "gate", 0.0)),
                "utility": _finite(getattr(edge, "utility", 0.0)),
                "plasticity": _finite(getattr(edge, "plasticity", 0.0)),
                "fade": _finite(getattr(edge, "fade", 1.0), 1.0),
                "lifecycle": str(getattr(edge, "lifecycle", "")),
                "created_by": str(getattr(edge, "created_by", "") or ""),
                "need_reason": str(getattr(edge, "need_reason", "") or ""),
                "candidate_id": str(getattr(edge, "candidate_id", "") or ""),
                "module_role": str(getattr(edge, "module_role", "") or ""),
            })
    specialist_modules: dict[str, dict[str, Any]] = {}
    for module_name, graph in _main_graphs(policy):
        graph_id = _graph_id(module_name)
        registry = getattr(graph, "specialist_modules", {})
        if not isinstance(registry, Mapping):
            continue
        for module_id, raw in registry.items():
            if not isinstance(raw, Mapping):
                continue
            key = f"{graph_id}::{module_id}"
            specialist_modules[key] = {
                "id": key,
                "graph_id": graph_id,
                "module_id": str(module_id),
                "role": str(raw.get("role", "")),
                "reason": str(raw.get("reason", "")),
                "status": str(raw.get("status", "candidate")),
                "evidence_tier": str(raw.get("evidence_tier", "")),
                "causal_confidence": _finite(raw.get("causal_confidence", 0.0)),
                "causal_validated": bool(raw.get("causal_validated", False)),
                "home_context_key": str(raw.get("home_context_key", "")),
                "node_count": len(tuple(raw.get("node_ids", ()) or ())),
                "edge_count": len(tuple(raw.get("edge_ids", ()) or ())),
                "similar_context_observations": int(_finite(raw.get("similar_context_observations", 0), 0.0)),
                "similar_context_supports": int(_finite(raw.get("similar_context_supports", 0), 0.0)),
                "positive_supports": int(_finite(raw.get("positive_supports", 0), 0.0)),
                "utility": _finite(raw.get("utility", 0.0)),
                "fade": _finite(raw.get("fade", 1.0), 1.0),
            }
    digest = hashlib.sha256(
        json.dumps(
            {
                "nodes": [(item["id"], item["region"], item["lifecycle"]) for item in nodes],
                "edges": [(item["id"], item["src"], item["dst"], item["enabled"], item["lifecycle"]) for item in edges],
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]
    region_counts: dict[str, int] = {}
    for node in nodes:
        region = str(node["region"])
        region_counts[region] = region_counts.get(region, 0) + 1
    return {
        "topology_id": digest,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "region_counts": region_counts,
        "graph_counts": graph_counts,
        "namespaced_ids": True,
        "specialist_modules": specialist_modules,
        "nodes": nodes,
        "edges": edges,
    }


def topology_summary(policy: Any) -> dict[str, Any]:
    """Compact structural counters for high-frequency live events."""
    topology = graph_topology(policy)
    specialists = dict(topology.get("specialist_modules", {}) or {})
    return {
        "topology_id": str(topology.get("topology_id", "")),
        "node_count": int(topology.get("node_count", 0) or 0),
        "edge_count": int(topology.get("edge_count", 0) or 0),
        "region_counts": dict(topology.get("region_counts", {}) or {}),
        "graph_counts": dict(topology.get("graph_counts", {}) or {}),
        "specialist_module_count": int(len(specialists)),
        "specialist_modules": {
            str(module_id): {
                "role": str(item.get("role", "")),
                "status": str(item.get("status", "")),
                "evidence_tier": str(item.get("evidence_tier", "")),
                "node_count": int(item.get("node_count", 0) or 0),
                "edge_count": int(item.get("edge_count", 0) or 0),
                "utility": _finite(item.get("utility", 0.0)),
                "fade": _finite(item.get("fade", 1.0), 1.0),
            }
            for module_id, item in specialists.items()
        },
    }


_NODE_STATE_FIELDS: tuple[str, ...] = (
    "bias",
    "utility",
    "plasticity",
    "fade",
    "lifecycle",
)
_EDGE_STATE_FIELDS: tuple[str, ...] = (
    "weight",
    "gate",
    "utility",
    "plasticity",
    "fade",
    "enabled",
    "lifecycle",
)


def _state_snapshot_from_topology(topology: Mapping[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """Return the mutable state fields used to compute bounded live deltas."""
    nodes = {
        str(item.get("id", "")): {key: item.get(key) for key in _NODE_STATE_FIELDS}
        for item in list(topology.get("nodes", []) or [])
        if isinstance(item, Mapping) and str(item.get("id", ""))
    }
    edges = {
        str(item.get("id", "")): {key: item.get(key) for key in _EDGE_STATE_FIELDS}
        for item in list(topology.get("edges", []) or [])
        if isinstance(item, Mapping) and str(item.get("id", ""))
    }
    return {"nodes": nodes, "edges": edges}


def _state_value_changed(before: object, after: object, *, epsilon: float) -> bool:
    if isinstance(before, bool) or isinstance(after, bool):
        return bool(before) != bool(after)
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        return abs(_finite(before) - _finite(after)) > max(0.0, float(epsilon))
    return before != after


def _state_delta(
    previous: Mapping[str, Mapping[str, Mapping[str, Any]]],
    topology: Mapping[str, Any],
    *,
    epsilon: float,
    max_nodes: int,
    max_edges: int,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, Any]]]]:
    """Compare a topology/state snapshot without treating every plastic update as topology.

    New nodes and edges are represented in full, while existing objects carry
    only changed fields.  The returned snapshot is always complete even when
    the emitted delta is capped, preventing a long burst from being re-emitted
    on every following lifecycle.
    """
    current = _state_snapshot_from_topology(topology)
    before_nodes = dict(previous.get("nodes", {}) or {})
    before_edges = dict(previous.get("edges", {}) or {})
    node_changes: list[dict[str, Any]] = []
    edge_changes: list[dict[str, Any]] = []

    for node_id, after in current["nodes"].items():
        prior = before_nodes.get(node_id)
        if prior is None:
            node_changes.append({"id": node_id, **after, "created": True})
            continue
        changed = {key: value for key, value in after.items() if _state_value_changed(prior.get(key), value, epsilon=epsilon)}
        if changed:
            node_changes.append({"id": node_id, **changed})

    for edge_id, after in current["edges"].items():
        prior = before_edges.get(edge_id)
        if prior is None:
            edge_changes.append({"id": edge_id, **after, "created": True})
            continue
        changed = {key: value for key, value in after.items() if _state_value_changed(prior.get(key), value, epsilon=epsilon)}
        if changed:
            edge_changes.append({"id": edge_id, **changed})

    node_removed = sorted(set(before_nodes).difference(current["nodes"]))
    edge_removed = sorted(set(before_edges).difference(current["edges"]))
    node_total = len(node_changes)
    edge_total = len(edge_changes)
    node_limit = int(max_nodes)
    edge_limit = int(max_edges)
    visible_nodes = node_changes if node_limit <= 0 else node_changes[:node_limit]
    visible_edges = edge_changes if edge_limit <= 0 else edge_changes[:edge_limit]
    visible_removed_nodes = node_removed if node_limit <= 0 else node_removed[:node_limit]
    visible_removed_edges = edge_removed if edge_limit <= 0 else edge_removed[:edge_limit]
    return (
        {
            "topology_id": str(topology.get("topology_id", "")),
            "nodes": visible_nodes,
            "edges": visible_edges,
            "removed_node_ids": visible_removed_nodes,
            "removed_edge_ids": visible_removed_edges,
            "node_change_count": int(node_total),
            "edge_change_count": int(edge_total),
            "removed_node_count": int(len(node_removed)),
            "removed_edge_count": int(len(edge_removed)),
            "truncated": bool(
                (node_limit > 0 and (node_total > node_limit or len(node_removed) > node_limit))
                or (edge_limit > 0 and (edge_total > edge_limit or len(edge_removed) > edge_limit))
            ),
        },
        current,
    )


def apply_synapse_state_delta(topology: Mapping[str, Any] | None, delta: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge a telemetry delta into a topology snapshot for non-Java readers.

    This preserves compatibility with the browser and Tk readers: they keep a
    full graph snapshot but still see changing synapse state when no node or
    edge is created.
    """
    source = dict(topology or {})
    if not isinstance(delta, Mapping):
        return source
    nodes = {
        str(item.get("id", "")): dict(item)
        for item in list(source.get("nodes", []) or [])
        if isinstance(item, Mapping) and str(item.get("id", ""))
    }
    edges = {
        str(item.get("id", "")): dict(item)
        for item in list(source.get("edges", []) or [])
        if isinstance(item, Mapping) and str(item.get("id", ""))
    }
    for node_id in list(delta.get("removed_node_ids", []) or []):
        nodes.pop(str(node_id), None)
    for edge_id in list(delta.get("removed_edge_ids", []) or []):
        edges.pop(str(edge_id), None)
    for item in list(delta.get("nodes", []) or []):
        if not isinstance(item, Mapping):
            continue
        node_id = str(item.get("id", ""))
        if not node_id:
            continue
        nodes[node_id] = {**nodes.get(node_id, {}), **dict(item)}
    for item in list(delta.get("edges", []) or []):
        if not isinstance(item, Mapping):
            continue
        edge_id = str(item.get("id", ""))
        if not edge_id:
            continue
        edges[edge_id] = {**edges.get(edge_id, {}), **dict(item)}
    source["nodes"] = list(nodes.values())
    source["edges"] = list(edges.values())
    source["node_count"] = len(nodes)
    source["edge_count"] = len(edges)
    if delta.get("topology_id"):
        source["topology_id"] = str(delta["topology_id"])
    return source


def _trace_graphs(policy: Any, trace: Any) -> dict[str, dict[str, Any]]:
    """V6 chỉ hiển thị graph trace do policy owner phát trực tiếp.

    Không dựng attribution từ trace thiếu graph_traces vì dữ liệu đó không đủ
    để gắn chính xác context → chuyên gia → policy.
    """
    raw = getattr(trace, "graph_traces", None)
    if not isinstance(raw, Mapping) or not raw:
        return {}
    return {
        str(graph_id): dict(record)
        for graph_id, record in raw.items()
        if isinstance(record, Mapping)
    }


def _controller_input_summary(value: object) -> tuple[float, dict[str, float]]:
    if isinstance(value, Mapping):
        details = {str(key): _finite(item) for key, item in value.items()}
        if not details:
            return 0.0, {}
        # Use the largest-magnitude controller signal as the compact visual
        # value; the complete per-controller evidence remains in the payload.
        selected = max(details.values(), key=lambda item: abs(item))
        return float(selected), details
    return _finite(value), {}


def graph_activity(policy: Any, trace: Any, *, max_nodes: int, max_edges: int) -> dict[str, Any]:
    """Render-aware activity for every participating graph.

    Activity IDs are namespaced exactly like topology IDs.  Per-graph quotas
    prevent dense M1 traces from hiding the reasoning trace, while counts still
    report the complete causal activity rather than the display sample.
    """
    registry = _graph_registry(policy)
    evidence = _trace_graphs(policy, trace)
    graph_ids = [graph_id for graph_id in evidence if graph_id in registry]
    if not graph_ids:
        graph_ids = list(registry)
    node_budget = int(max_nodes)
    edge_budget = int(max_edges)
    per_graph_nodes = 0 if node_budget <= 0 else max(8, node_budget // max(1, len(graph_ids)))
    per_graph_edges = 0 if edge_budget <= 0 else max(8, edge_budget // max(1, len(graph_ids)))
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []
    graph_summaries: dict[str, dict[str, Any]] = {}
    total_active_nodes = 0
    total_active_edges = 0
    combined_regions: dict[str, float] = {}

    for graph_id in graph_ids:
        module_name, graph = registry[graph_id]
        item = dict(evidence.get(graph_id, {}) or {})
        active_node_ids = tuple(dict.fromkeys(str(value) for value in item.get("active_neuron_ids", ()) or () if str(value)))
        active_edge_ids = tuple(dict.fromkeys(str(value) for value in item.get("active_edge_ids", ()) or () if str(value)))
        total_active_nodes += len(active_node_ids)
        total_active_edges += len(active_edge_ids)
        node_values = dict(item.get("node_values", {}) or {})
        # V18 M1 traces expose one raw value per input; reasoning traces retain
        # a per-controller map for the same input node.  The viewer accepts
        # both without inventing a zero that would look like a dead feature.
        input_values = dict(item.get("input_values", item.get("raw_inputs", {})) or {})
        edge_gates = dict(item.get("edge_gates", {}) or {})
        edge_contrib = dict(item.get("edge_contributions", {}) or {})
        controllers = dict(item.get("controllers", {}) or {})
        region_activity = {str(key): _finite(value) for key, value in dict(item.get("region_activity", {}) or {}).items()}
        for region, value in region_activity.items():
            combined_regions[f"{graph_id}:{region}"] = value

        displayed_nodes = 0
        for local_id in (active_node_ids if per_graph_nodes <= 0 else active_node_ids[:per_graph_nodes]):
            node = getattr(graph, "neurons", {}).get(local_id)
            if node is None:
                continue
            details = dict(node_values.get(local_id, {}) or {})
            raw_value, controller_inputs = _controller_input_summary(input_values.get(local_id, {}))
            node_id = _qualified_id(graph_id, local_id)
            nodes.append({
                "id": node_id,
                "local_id": local_id,
                "graph_id": graph_id,
                "module": str(module_name),
                "activation": _finite(details.get("activation", getattr(node, "activation", 0.0))),
                "state": _finite(details.get("state", getattr(node, "state", 0.0))),
                "raw_input": raw_value,
                "controller_inputs": controller_inputs,
                "utility": _finite(getattr(node, "utility", 0.0)),
                "plasticity": _finite(getattr(node, "plasticity", 0.0)),
                "fade": _finite(getattr(node, "fade", 1.0), 1.0),
                "lifecycle": str(getattr(node, "lifecycle", "")),
            })
            displayed_nodes += 1

        displayed_edges = 0
        for local_id in (active_edge_ids if per_graph_edges <= 0 else active_edge_ids[:per_graph_edges]):
            edge = getattr(graph, "edges", {}).get(local_id)
            if edge is None:
                continue
            edge_id = _qualified_id(graph_id, local_id)
            edges.append({
                "id": edge_id,
                "local_id": local_id,
                "graph_id": graph_id,
                "module": str(module_name),
                "weight": _finite(getattr(edge, "weight", 0.0)),
                "gate": _finite(edge_gates.get(local_id, getattr(edge, "gate", 0.0))),
                "contribution": _finite(edge_contrib.get(local_id, 0.0)),
                "utility": _finite(getattr(edge, "utility", 0.0)),
                "plasticity": _finite(getattr(edge, "plasticity", 0.0)),
                "enabled": bool(getattr(edge, "enabled", True)),
                "lifecycle": str(getattr(edge, "lifecycle", "")),
            })
            displayed_edges += 1

        graph_summaries[graph_id] = {
            "graph_id": graph_id,
            "label": str(module_name),
            "active_node_count": len(active_node_ids),
            "active_edge_count": len(active_edge_ids),
            "displayed_node_count": displayed_nodes,
            "displayed_edge_count": displayed_edges,
            "controllers": controllers,
            "controller_credit": dict(item.get("controller_credit", {}) or {}),
            "region_activity": region_activity,
        }

    return {
        "active_node_count": int(total_active_nodes),
        "active_edge_count": int(total_active_edges),
        "displayed_node_count": int(len(nodes)),
        "displayed_edge_count": int(len(edges)),
        "nodes": nodes,
        "edges": edges,
        "graphs": graph_summaries,
        "region_activity": combined_regions,
        "namespaced_ids": True,
    }


@dataclass
class LiveTelemetryWriter:
    """Append-only writer used by a policy while it trains.

    It intentionally keeps no file handle open.  MT5 and the browser can read
    independently on Windows without fighting a long-lived shared handle.
    """

    path: str = DEFAULT_EVENT_PATH
    enabled: bool = False
    decision_sample_every: int = 1
    network_development_feedback_sample_every: int = 1
    # 0 = không giới hạn/cắt telemetry. V9 ưu tiên audit đầy đủ; người dùng
    # tự chọn tốc độ render ở UI, không được mất dữ liệu ở writer.
    max_event_bytes: int = 0
    max_active_nodes: int = 0
    max_active_edges: int = 0
    max_state_nodes: int = 0
    max_state_edges: int = 0
    state_change_epsilon: float = 1e-6
    decision_profile_path: str = ""
    structural_event_path: str = ""
    network_development_path: str = ""
    max_responsible_neurons: int = 0
    max_responsible_edges: int = 0
    run_id: str = ""
    source: str = "training"
    configured_path: str = ""
    scope_paths_by_run: bool = False
    root_run_id: str = ""
    _decision_count: int = field(default=0, repr=False)
    _feedback_count: int = field(default=0, repr=False)
    _last_growth_index: int = field(default=0, repr=False)
    _topology_id: str = field(default="", repr=False)
    _development_topology_id: str = field(default="", repr=False)
    # Mỗi decision đầu tiên của một topology có baseline đầy đủ riêng; những
    # decision sau chỉ tham chiếu để UI vẫn audit được mà không lặp 8k edge.
    _development_last_decision_topology_id: str = field(default="", repr=False)
    _development_snapshot_sequence: int = field(default=0, repr=False)
    _state_snapshot: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict, repr=False)
    _event_sequence: int = field(default=0, repr=False)
    _last_event_at: float = field(default=0.0, repr=False)
    _last_error: str = field(default="", repr=False)

    @classmethod
    def from_config(cls, config: object, *, run_id: str = "", source: str = "training") -> "LiveTelemetryWriter":
        configured_path = str(getattr(config, "event_path", DEFAULT_EVENT_PATH) or DEFAULT_EVENT_PATH)
        effective_run_id = str(run_id or f"run-{int(time.time())}-{os.getpid()}")
        scope_paths_by_run = bool(getattr(config, "scope_paths_by_run", False))
        if scope_paths_by_run:
            raw = Path(configured_path)
            event_path = str(raw.parent / "runs" / _safe_run_component(effective_run_id) / raw.name)
        else:
            event_path = configured_path
        return cls(
            path=event_path,
            configured_path=configured_path,
            scope_paths_by_run=scope_paths_by_run,
            root_run_id=effective_run_id,
            enabled=bool(getattr(config, "enabled", False)),
            decision_sample_every=max(1, int(getattr(config, "decision_sample_every", 1) or 1)),
            network_development_feedback_sample_every=max(1, int(getattr(config, "network_development_feedback_sample_every", 1) or 1)),
            max_event_bytes=max(0, int(getattr(config, "max_event_bytes", 0) if getattr(config, "max_event_bytes", None) is not None else 0)),
            max_active_nodes=max(0, int(getattr(config, "max_active_nodes", 0) if getattr(config, "max_active_nodes", None) is not None else 0)),
            max_active_edges=max(0, int(getattr(config, "max_active_edges", 0) if getattr(config, "max_active_edges", None) is not None else 0)),
            max_state_nodes=max(0, int(getattr(config, "max_state_nodes", 0) if getattr(config, "max_state_nodes", None) is not None else 0)),
            max_state_edges=max(0, int(getattr(config, "max_state_edges", 0) if getattr(config, "max_state_edges", None) is not None else 0)),
            state_change_epsilon=max(0.0, float(getattr(config, "state_change_epsilon", 1e-6) or 0.0)),
            decision_profile_path=str(getattr(config, "decision_profile_path", "") or ""),
            structural_event_path=str(getattr(config, "structural_event_path", "") or ""),
            network_development_path=str(getattr(config, "network_development_path", "") or ""),
            max_responsible_neurons=max(0, int(getattr(config, "max_responsible_neurons", 0) if getattr(config, "max_responsible_neurons", None) is not None else 0)),
            max_responsible_edges=max(0, int(getattr(config, "max_responsible_edges", 0) if getattr(config, "max_responsible_edges", None) is not None else 0)),
            run_id=effective_run_id,
            source=str(source),
        )

    def copy(self) -> "LiveTelemetryWriter":
        return LiveTelemetryWriter(
            path=self.path,
            enabled=bool(self.enabled),
            decision_sample_every=int(self.decision_sample_every),
            network_development_feedback_sample_every=int(self.network_development_feedback_sample_every),
            max_event_bytes=int(self.max_event_bytes),
            max_active_nodes=int(self.max_active_nodes),
            max_active_edges=int(self.max_active_edges),
            max_state_nodes=int(self.max_state_nodes),
            max_state_edges=int(self.max_state_edges),
            state_change_epsilon=float(self.state_change_epsilon),
            decision_profile_path=str(self.decision_profile_path),
            structural_event_path=str(self.structural_event_path),
            network_development_path=str(self.network_development_path),
            max_responsible_neurons=int(self.max_responsible_neurons),
            max_responsible_edges=int(self.max_responsible_edges),
            run_id=str(self.run_id),
            source=str(self.source),
            configured_path=str(self.configured_path),
            scope_paths_by_run=bool(self.scope_paths_by_run),
            root_run_id=str(self.root_run_id),
            _decision_count=int(self._decision_count),
            _feedback_count=int(self._feedback_count),
            _last_growth_index=int(self._last_growth_index),
            _topology_id=str(self._topology_id),
            _development_topology_id=str(self._development_topology_id),
            _development_last_decision_topology_id=str(self._development_last_decision_topology_id),
            _development_snapshot_sequence=int(self._development_snapshot_sequence),
            _state_snapshot={
                str(group): {str(item_id): dict(values) for item_id, values in dict(items).items()}
                for group, items in dict(self._state_snapshot or {}).items()
            },
            _event_sequence=int(self._event_sequence),
            _last_event_at=float(self._last_event_at),
            _last_error=str(self._last_error),
        )

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), default=str), encoding="utf-8")
            os.replace(temporary, path)
        except OSError:
            return

    def _active_run_payload(self, event_path: Path) -> dict[str, Any]:
        return {
            "schema_version": LIVE_SCHEMA_VERSION,
            "root_run_id": str(self.root_run_id or self.run_id),
            "run_id": str(self.run_id),
            "root_run_id": str(self.root_run_id or self.run_id),
            "source": str(self.source),
            "event_path": str(event_path),
            "decision_profile_path": str(self._aux_path("quyet_dinh")),
            "structural_event_path": str(self._aux_path("cau_truc")),
            "network_development_path": str(self._aux_path("phat_trien_mang")),
            "configured_event_path": str(self.configured_path or event_path),
            "status_path": str(self._status_path(event_path)),
            "scope_paths_by_run": bool(self.scope_paths_by_run),
            "started_at": time.time(),
        }

    def _publish_active_run(self, event_path: Path) -> None:
        payload = self._active_run_payload(event_path)
        self._atomic_json(event_path.with_name("active_run.json"), payload)
        configured = Path(self.configured_path) if str(self.configured_path).strip() else event_path
        if configured.parent != event_path.parent:
            self._atomic_json(configured.parent / "active_run.json", payload)

    def _status_path(self, event_path: Path) -> Path:
        return event_path.with_suffix(event_path.suffix + ".status.json")

    def _write_status(self, event_path: Path, *, event_type: str = "") -> None:
        self._atomic_json(self._status_path(event_path), {
            "schema_version": LIVE_SCHEMA_VERSION,
            "run_id": str(self.run_id),
            "root_run_id": str(self.root_run_id or self.run_id),
            "source": str(self.source),
            "event_path": str(event_path),
            "event_sequence": int(self._event_sequence),
            "last_event_at": float(self._last_event_at),
            "last_event_type": str(event_type),
            "last_error": str(self._last_error),
        })

    def _compact(self, path: Path) -> None:
        limit = int(self.max_event_bytes)
        if limit <= 0:
            return
        try:
            if not path.exists() or path.stat().st_size <= limit:
                return
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                start = max(0, handle.tell() - limit)
                handle.seek(start)
                if start:
                    handle.readline()
                data = handle.read()
            tmp = path.with_suffix(path.suffix + ".compact.tmp")
            with tmp.open("wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except OSError:
            return

    def emit(
        self,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        max_array: int = 0,
    ) -> bool:
        """Append one JSONL event and publish a durable writer heartbeat.

        The former writer swallowed I/O errors and left the UI rendering an old
        graph as though it were live.  V9 preserves non-blocking telemetry but
        makes success/failure observable through an adjacent status sidecar.
        """
        if not self.enabled:
            return False
        path = Path(self.path)
        self._event_sequence += 1
        now = time.time()
        event = {
            "schema_version": LIVE_SCHEMA_VERSION,
            "event_type": str(event_type),
            "created_at": now,
            "run_id": str(self.run_id),
            "root_run_id": str(self.root_run_id or self.run_id),
            "source": str(self.source),
            "event_sequence": int(self._event_sequence),
            "payload": _json_safe(dict(payload or {}), max_array=int(max_array)),
        }
        text = json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = _write_lock(path)
        with lock:
            self._compact(path)
            try:
                with path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                    handle.flush()
                self._last_event_at = now
                self._last_error = ""
                self._write_status(path, event_type=str(event_type))
                return True
            except OSError as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._write_status(path, event_type=str(event_type))
                return False

    def _aux_path(self, kind: str) -> Path:
        primary = Path(self.path)
        # Run-scoped mode must keep every companion JSONL beside its active
        # event stream. Explicit config paths are only root anchors for UI,
        # never a place to mix decision/structural logs across runs.
        if bool(self.scope_paths_by_run):
            if kind == "quyet_dinh":
                return primary.with_name(primary.stem + ".ho_so_quyet_dinh.jsonl")
            if kind == "phat_trien_mang":
                return primary.with_name(primary.stem + ".theo_doi_phat_trien_mang.jsonl")
            return primary.with_name(primary.stem + ".phat_trien_cau_truc.jsonl")
        if kind == "quyet_dinh":
            configured = str(self.decision_profile_path or "").strip()
            return Path(configured) if configured else primary.with_name(primary.stem + ".ho_so_quyet_dinh.jsonl")
        if kind == "phat_trien_mang":
            configured = str(self.network_development_path or "").strip()
            return Path(configured) if configured else primary.with_name(primary.stem + ".theo_doi_phat_trien_mang.jsonl")
        configured = str(self.structural_event_path or "").strip()
        return Path(configured) if configured else primary.with_name(primary.stem + ".phat_trien_cau_truc.jsonl")

    def _append_aux_event(self, path: Path, payload: Mapping[str, Any]) -> bool:
        """Ghi JSONL audit riêng; lỗi logging không được thay đổi policy/MT5."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps(_json_safe(dict(payload), max_array=0), ensure_ascii=False, separators=(",", ":")) + "\n"
            with _write_lock(path):
                self._compact(path)
                with path.open("a", encoding="utf-8", newline="") as handle:
                    handle.write(text)
                    handle.flush()
                self._compact(path)
            return True
        except OSError:
            return False

    def ho_so_quyet_dinh(self, policy: Any, trace: Any) -> dict[str, Any]:
        profile = xay_ho_so_quyet_dinh(
            policy, trace,
            max_neurons=int(self.max_responsible_neurons),
            max_edges=int(self.max_responsible_edges),
        )
        profile.update({"run_id": str(self.run_id), "root_run_id": str(self.root_run_id or self.run_id), "source": str(self.source)})
        self._append_aux_event(self._aux_path("quyet_dinh"), profile)
        return profile

    def theo_doi_phat_trien_mang(
        self,
        policy: Any,
        trace: Any | None,
        *,
        giai_doan: str,
        synapse_state: Mapping[str, Any] | None = None,
        growth_events: Iterable[Any] | None = None,
        activity: Mapping[str, Any] | None = None,
        bo_sung: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Ghi nhật ký phát triển mạng đầy đủ, chỉ đọc và không cắt dữ liệu.

        Log này khác hồ sơ quyết định: nó theo dõi topology, trạng thái neuron/
        cạnh thay đổi, plasticity, fade, lifecycle và mọi growth event trong
        chính policy-owner của context. Không dùng để tác động policy/MT5.
        """
        trace = trace if trace is not None else object()
        # At initialization there is no decision trace yet. Preserve the
        # policy owner provenance so the first full snapshot is attributable
        # to the right context/expert instead of appearing as ownerless.
        provenance = dict(getattr(policy, "_context_expert_runtime_provenance", {}) or {})
        audit = {**provenance, **dict(getattr(trace, "context_audit", {}) or {})}
        details = dict(getattr(trace, "action_details", {}) or {})
        topology = graph_topology(policy)
        topology_id = str(topology.get("topology_id", ""))
        policy_instance = str(provenance.get("policy_instance_uuid", "") or getattr(policy, "_context_expert_policy_instance_uuid", "") or id(policy))
        snapshot_key = "|".join((str(Path(self.path).resolve()), str(self.root_run_id or self.run_id), policy_instance, topology_id))
        with _DEVELOPMENT_SNAPSHOT_LOCK:
            known = _DEVELOPMENT_SNAPSHOT_REGISTRY.get(snapshot_key)
            if known is None:
                self._development_snapshot_sequence += 1
                topology_snapshot_id = f"{policy_instance}:{topology_id}:{self._development_snapshot_sequence}"
                _DEVELOPMENT_SNAPSHOT_REGISTRY[snapshot_key] = (topology_snapshot_id, int(self._development_snapshot_sequence))
                full_snapshot = True
            else:
                topology_snapshot_id = str(known[0])
                full_snapshot = False
        if full_snapshot:
            self._development_topology_id = topology_id
            self._development_last_decision_topology_id = topology_id
        network_activity = dict(activity or graph_activity(
            policy, trace, max_nodes=int(self.max_active_nodes), max_edges=int(self.max_active_edges)
        ))
        growth_review = dict(details.get("growth_review_lifecycle", details.get("growth_review", audit.get("growth_review", {}))) or {})
        payload = {
            "loai_ban_ghi": "THEO_DOI_PHAT_TRIEN_MANG",
            "schema_version": 1,
            "tao_luc": time.time(),
            "run_id": str(self.run_id),
            "root_run_id": str(self.root_run_id or self.run_id),
            "source": str(self.source),
            "giai_doan": str(giai_doan),
            "trace_id": str(getattr(trace, "trace_id", "") or ""),
            "frame_id": str(getattr(trace, "context_frame_id", "") or ""),
            "context": {
                "ma_context": str(audit.get("ma_context", "") or ""),
                "ma_chuyen_gia": str(audit.get("ma_chuyen_gia", "") or ""),
                "ma_policy": str(audit.get("ma_policy", "") or ""),
                "context_on_dinh": bool(audit.get("context_on_dinh", False)),
                "policy_instance_uuid": str(audit.get("policy_instance_uuid", "") or ""),
                "graph_instance_uuid": str(audit.get("graph_instance_uuid", "") or ""),
                "weight_hash": str(audit.get("weight_hash", "") or ""),
                "memory_hash": str(audit.get("memory_hash", "") or ""),
                "memory_instance_uuid": str(audit.get("memory_instance_uuid", "") or ""),
                "memory_object_identity": str(audit.get("memory_object_identity", "") or ""),
                "memory_entry_count": int(_finite(audit.get("memory_entry_count", 0), 0.0)),
                "policy_checkpoint_hash": str(audit.get("policy_checkpoint_hash", "") or ""),
                "root_run_id": str(audit.get("root_run_id", self.root_run_id or self.run_id) or ""),
                "expert_run_id": str(audit.get("expert_run_id", "") or ""),
            },
            "quyet_dinh": {
                "hanh_dong_nghia_vu": _action_name(details.get("semantic_action", details.get("emitted_action", "")), str(details.get("action_name", ""))),
                "hanh_dong_wire_mt5": _action_name(details.get("wire_action", details.get("emitted_action", "")), str(details.get("action_name", ""))),
                "wire_matches_semantic_action": bool(details.get("wire_matches_semantic_action", True)),
            },
            # Every row is self-contained. Browser/live-tail clients can begin
            # at any offset, so a reference-only row would otherwise render an
            # empty or stale topology until it happened to find a prior snapshot.
            # ``topology_moi`` retains the delta signal without sacrificing audit.
            "topology_day_du": topology,
            "topology_tham_chieu": {
                "topology_id": topology_id,
                "snapshot_id": topology_snapshot_id,
                "snapshot_day_du_co_san": True,
                "topology_moi": bool(full_snapshot),
            },
            "tong_quan_topology": topology_summary(policy),
            "hoat_dong_day_du": network_activity,
            "thay_doi_trang_thai_neuron_lien_ket": _json_safe(dict(synapse_state or {}), max_array=0),
            "su_kien_phat_trien": [_json_safe(item, max_array=0) for item in list(growth_events or [])],
            "growth_review": _json_safe(growth_review, max_array=0),
            "bo_sung": _json_safe(dict(bo_sung or {}), max_array=0),
        }
        self._append_aux_event(self._aux_path("phat_trien_mang"), payload)
        return payload

    def nhat_ky_cau_truc(self, policy: Any, trace: Any, events: Iterable[Any], *, giai_doan: str) -> list[dict[str, Any]]:
        """Append only genuine topology changes to the structural journal.

        Credit attribution remains visible in the normal telemetry event, but it
        must never be presented as neuron/edge growth. This keeps the UI log
        honest: every row in `nhat_ky_phat_trien_neuron_lien_ket.jsonl` means
        at least one node/edge/module was actually created, pruned, faded or
        rewired.
        """
        rows: list[dict[str, Any]] = []
        for event in list(events or []):
            profile = xay_ho_so_phat_trien_cau_truc(policy, trace, event, giai_doan=str(giai_doan))
            if not bool(profile.get("su_kien", {}).get("la_moc_cau_truc", False)):
                continue
            profile.update({"run_id": str(self.run_id), "root_run_id": str(self.root_run_id or self.run_id), "source": str(self.source)})
            self._append_aux_event(self._aux_path("cau_truc"), profile)
            rows.append(profile)
        return rows

    def heartbeat(self, *, stage: str, sequence: int, payload: Mapping[str, Any] | None = None) -> bool:
        """Emit a bridge/policy heartbeat even when no trade is placed."""
        body = {"stage": str(stage), "sequence": int(sequence), **dict(payload or {})}
        return self.emit("bridge_heartbeat", body)

    def emit_topology(self, policy: Any, *, reason: str, force: bool = False) -> bool:
        if not self.enabled:
            return False
        topology = graph_topology(policy)
        current_id = str(topology.get("topology_id", ""))
        if not force and current_id == self._topology_id:
            return False
        self._topology_id = current_id
        self.emit(
            "topology",
            {
                "reason": str(reason),
                "revision": int(getattr(policy, "revision", 0)),
                "topology": topology,
                "telemetry_complete": True,
                "telemetry_topology_node_count": int(topology.get("node_count", 0)),
                "telemetry_topology_edge_count": int(topology.get("edge_count", 0)),
            },
            max_array=TOPOLOGY_JSON_ARRAY_LIMIT,
        )
        self._state_snapshot = _state_snapshot_from_topology(topology)
        return True

    def synapse_state(self, policy: Any) -> dict[str, Any]:
        """Return changed node/edge state since the last live state observation."""
        if not self.enabled:
            return {"nodes": [], "edges": [], "removed_node_ids": [], "removed_edge_ids": []}
        topology = graph_topology(policy)
        delta, current = _state_delta(
            self._state_snapshot,
            topology,
            epsilon=float(self.state_change_epsilon),
            max_nodes=int(self.max_state_nodes),
            max_edges=int(self.max_state_edges),
        )
        self._state_snapshot = current
        return delta

    def _new_growth_events(self, policy: Any) -> list[Any]:
        growth_events = list(getattr(policy, "growth_events", []) or [])
        new_growth = growth_events[self._last_growth_index:]
        self._last_growth_index = len(growth_events)
        return list(new_growth)

    def start(self, policy: Any, *, metadata: Mapping[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        self._last_growth_index = len(list(getattr(policy, "growth_events", []) or []))
        event_path = Path(self.path)
        self._publish_active_run(event_path)
        self.emit("run_started", {
            "revision": int(getattr(policy, "revision", 0)),
            "artifact_version": int(getattr(policy, "artifact_version", 0)),
            "m1_backend": str(getattr(policy, "m1_backend", "direct_rnn")),
            "metadata": dict(metadata or {}),
        })
        self.emit_topology(policy, reason="run_started", force=True)
        # Mốc gốc đầy đủ: về sau UI so sánh topology/state với snapshot này.
        self.theo_doi_phat_trien_mang(
            policy, None, giai_doan="khoi_tao",
            synapse_state=self.synapse_state(policy),
            growth_events=[],
            bo_sung={"metadata": dict(metadata or {}), "moc": "TOPOLOGY_BAN_DAU"},
        )

    def trial(self, policy: Any, *, policy_id: int, round_index: int, phase: str = "mt5") -> None:
        if not self.enabled:
            return
        self.emit("trial_started", {
            "policy_id": int(policy_id),
            "round": int(round_index),
            "phase": str(phase),
            "revision": int(getattr(policy, "revision", 0)),
            "trial_mode": bool(getattr(policy, "graph_trial_mode", False)),
            "trial_id": str(getattr(policy, "graph_trial_id", "") or ""),
        })
        self.emit_topology(policy, reason="trial_started", force=False)
        self.theo_doi_phat_trien_mang(
            policy, None, giai_doan="bat_dau_thu_nghiem",
            synapse_state=self.synapse_state(policy),
            growth_events=[],
            bo_sung={"policy_id": int(policy_id), "round": int(round_index), "phase": str(phase)},
        )

    def decision(self, policy: Any, trace: Any) -> None:
        if not self.enabled:
            return
        self._decision_count += 1
        if self._decision_count % max(1, int(self.decision_sample_every)) != 0:
            return
        audit_gate = dict(getattr(trace, "context_audit", {}) or {})
        no_owner_policy = bool(getattr(trace, "ly_do_khong_hoc", False)) or bool(audit_gate.get("khong_co_policy_ra_lenh", False))
        if no_owner_policy:
            profile = self.ho_so_quyet_dinh(policy, trace)
            self.emit("context_unstable", {
                "trace_id": str(getattr(trace, "trace_id", "") or ""),
                "frame_id": str(getattr(trace, "context_frame_id", "") or ""),
                "giai_doan": str(audit_gate.get("giai_doan_telemetry", "CONTEXT_CHUA_ON_DINH")),
                "context": audit_gate,
                "ly_do": str(audit_gate.get("ly_do_khong_giao_dich", "CONTEXT_CHUA_ON_DINH")),
                "ho_so_context": {
                    "loai_ban_ghi": str(profile.get("loai_ban_ghi", "CONTEXT_CHUA_ON_DINH")),
                    "tom_tat_tieng_viet": str(profile.get("tom_tat_tieng_viet", "")),
                },
            })
            return
        details = dict(getattr(trace, "action_details", {}) or {})
        five_reasoning_trace = dict(details.get("five_reasoning_trace", {}) or {})
        audit = dict(getattr(trace, "context_audit", {}) or {})
        causal_ledger = getattr(policy, "causal_ledger", None)
        causal_snapshot = {}
        if causal_ledger is not None and hasattr(causal_ledger, "decision_payload"):
            try:
                causal_snapshot = dict(causal_ledger.decision_payload(str(getattr(trace, "trace_id", ""))) or {})
            except Exception:
                causal_snapshot = {}
        semantic_raw = int(_finite(details.get("semantic_action", details.get("emitted_action", -1)), -1.0))
        emitted_raw = int(_finite(details.get("emitted_action", -1), -1.0))
        wire_raw = int(_finite(details.get("wire_action", details.get("emitted_action", -1)), -1.0))
        command_contract = (
            semantic_raw >= 0 and emitted_raw >= 0 and wire_raw >= 0
            and semantic_raw == emitted_raw == wire_raw
        )
        # ``NO_NEURAL_COMMAND`` is deliberately encoded as wire HOLD (0) so
        # the EA can transport a no-op. It is complete transport evidence, not
        # an incomplete BUY/SELL command and not an explicit neural HOLD.
        no_command_wire_noop = bool(
            semantic_raw < 0 and emitted_raw < 0 and wire_raw == 0
            and (bool(details.get("wire_noop", False)) or bool(details.get("no_neural_command", False)))
        )
        action_contract_complete = bool(command_contract or no_command_wire_noop)
        action_contract_state = (
            "COMMAND" if command_contract else
            "NO_COMMAND_WIRE_NOOP" if no_command_wire_noop else
            "INCOMPLETE_OR_MISMATCH"
        )
        semantic_display = _action_name(semantic_raw, "NO_COMMAND") if command_contract else "NO_COMMAND"
        emitted_display = _action_name(emitted_raw, "NO_COMMAND") if command_contract else "NO_COMMAND"
        wire_display = _action_name(wire_raw, "KHÔNG_GỬI_LENH") if action_contract_complete else "KHÔNG_GỬI_LENH"
        payload = {
            "revision": int(getattr(policy, "revision", 0)),
            "step": int(getattr(policy, "_step_count", 0)),
            "trace_id": str(getattr(trace, "trace_id", "")),
            "frame_id": str(getattr(trace, "context_frame_id", "")),
            "context_schema_version": int(getattr(trace, "context_schema_version", 0) or 0),
            "action": {
                # Chỉ tên semantic/wire được phép audit. Không phát legacy
                # policy_action/legal_action, cũng không phát sentinel -1.
                "name": emitted_display,
                "semantic_name": semantic_display,
                "wire_name": wire_display,
                "emitted_name": emitted_display,
                "selected_index": int(getattr(trace, "selected_idx", 0)),
                "semantic_action": semantic_raw if command_contract else None,
                "emitted_action": emitted_raw if command_contract else None,
                "wire_action": wire_raw if action_contract_complete else None,
                "action_contract_complete": bool(action_contract_complete),
                "action_contract_state": action_contract_state,
                "wire_noop": bool(no_command_wire_noop),
                "execution_bootstrap": bool(details.get("execution_bootstrap", False)),
                "action_masked": bool(details.get("action_masked", False)),
                "selected_action": _vector(getattr(trace, "selected_action", []), 0),
                "candidate_scores": _vector(getattr(trace, "candidate_scores", []), 0),
                "candidate_logits": _vector(getattr(trace, "candidate_logits", []), 0),
                "candidate_actions": _json_safe(np.asarray(getattr(trace, "candidate_actions", []), dtype=float), max_array=0),
                "allowed_actions": [index for index, value in enumerate(tuple(getattr(trace, "allowed_action_mask", ()) or ())) if bool(value)],
                "confidence": _finite(getattr(trace, "confidence", 0.0)),
                "critic_value": _finite(getattr(trace, "critic_value", 0.0)),
            },
            "context": {
                "audit": audit,
                "account": dict(getattr(trace, "account_context", {}) or {}),
                "learning_drive": _vector(getattr(trace, "learning_drive_context", []), 0),
                "canonical_context_key": _vector(getattr(trace, "canonical_context_key", []), 0),
            },
            "expectancy": {
                "predicted_lcb": _finite(details.get("expectancy_predicted_lcb", 0.0)),
                "predicted_mean_net_r": _finite(details.get("expectancy_predicted_mean_net_r", 0.0)),
                "samples": _finite(details.get("expectancy_predicted_samples", 0.0)),
                "confidence": _finite(details.get("expectancy_predicted_confidence", 0.0)),
                "best_action": int(_finite(details.get("expectancy_best_action", 0.0))),
                "best_lcb": _finite(details.get("expectancy_best_lcb", 0.0)),
                "regret": _finite(details.get("expectancy_regret", 0.0)),
                "scope": dict(details.get("expectancy_scope", {}) or {}),
            },
            "reasoning": {
                "schema_version": int(_finite(details.get("reasoning_schema_version", 0), 0.0)),
                "selection_stage": str(details.get("reasoning_selection_stage", "")),
                "selected_candidate_before": int(_finite(details.get("reasoning_selected_candidate_before", -1), -1.0)),
                "selected_candidate_after": int(_finite(details.get("reasoning_selected_candidate_after", -1), -1.0)),
                "changed_candidate": bool(details.get("reasoning_changed_candidate", False)),
                "action_score_gain": _finite(details.get("reasoning_action_score_gain", 0.0)),
                "action_score_delta": _vector(details.get("reasoning_action_score_delta", []), 0),
                "legal_candidate_count": int(_finite(details.get("reasoning_legal_candidate_count", 0), 0.0)),
                "active_node_count": int(_finite(details.get("reasoning_active_node_count", 0), 0.0)),
                "active_edge_count": int(_finite(details.get("reasoning_active_edge_count", 0), 0.0)),
                "selector_stage": str(details.get("reasoning_selector_stage", "")),
                "selector_score_bias_l1": {
                    str(key): _finite(value)
                    for key, value in dict(details.get("reasoning_selector_score_bias_l1", {}) or {}).items()
                },
            },
            "five_reasoning": {
                "enabled": bool(details.get("five_reasoning_rl_enabled", False)),
                "schema_version": int(_finite(five_reasoning_trace.get("schema_version", 0), 0.0)),
                "steps": [
                    int(_finite(value, 0.0))
                    for value in list(five_reasoning_trace.get("steps", []) or [])
                ],
                "final_hypothesis": int(_finite(
                    five_reasoning_trace.get("final_hypothesis", -1), -1.0
                )),
                "action_scale": _finite(details.get(
                    "five_reasoning_action_scale",
                    five_reasoning_trace.get("action_scale", 0.0),
                )),
                "action_residual_clip": _finite(five_reasoning_trace.get(
                    "action_residual_clip", 0.0
                )),
                "training_labels_visible": bool(five_reasoning_trace.get(
                    "training_labels_visible", False
                )),
            },
            "learner_recovery": {
                "schema_version": int(_finite(details.get("learner_recovery_schema_version", 0), 0.0)),
                "hold_reason": str(details.get("hold_reason", "") or ""),
                "replacement_applied": bool(dict(details.get("learner_recovery", {}) or {}).get("replacement_applied", False)),
                "replacement_from": int(_finite(details.get("learner_recovery_replacement_from", 0), 0.0)),
                "replacement_to": int(_finite(details.get("learner_recovery_replacement_to", 0), 0.0)),
                "global_hold_penalty": _finite(details.get("learner_recovery_global_hold_penalty", 0.0)),
                "negative_action_suppression": _finite(details.get("learner_recovery_negative_action_suppression", 0.0)),
                "detail": _json_safe(dict(details.get("learner_recovery", {}) or {}), max_array=0),
            },
            "causal": {
                "schema_version": int(causal_snapshot.get("schema_version", 0) or 0),
                "entry_snapshot": _json_safe(causal_snapshot, max_array=0),
            },
            "network": {
                "novelty": _finite(getattr(trace, "novelty", 0.0)),
                "conflict": _finite(getattr(trace, "conflict", 0.0)),
                "prediction_error": _finite(getattr(trace, "prediction_error", 0.0)),
                "risk": _finite(getattr(trace, "risk", 0.0)),
                "memory_match_quality": _finite(getattr(trace, "memory_match_quality", 0.0)),
                "regions": {str(k): _finite(v) for k, v in dict(getattr(trace, "graph_region_activity", {}) or {}).items()},
                "activity": graph_activity(policy, trace, max_nodes=self.max_active_nodes, max_edges=self.max_active_edges),
            },
            "topology_summary": topology_summary(policy),
        }
        ho_so = self.ho_so_quyet_dinh(policy, trace)
        payload["ho_so_quyet_dinh_tom_tat"] = {
            "tom_tat_tieng_viet": str(ho_so.get("tom_tat_tieng_viet", "")),
            "ma_context": str(dict(ho_so.get("context", {}) or {}).get("ma_context", "")),
            "ma_chuyen_gia": str(dict(ho_so.get("context", {}) or {}).get("ma_chuyen_gia", "")),
            "so_neuron_quy_trach": len(list(ho_so.get("neuron_chiu_trach_nhiem", []) or [])),
            "so_lien_ket_quy_trach": len(list(ho_so.get("lien_ket_chiu_trach_nhiem", []) or [])),
            "regularization": dict(ho_so.get("regularization", {}) or {}),
        }
        # Hồ sơ đầy đủ đã được ghi append-only vào decision_profile_path; chỉ
        # giữ tóm tắt trong telemetry chính để không làm UI live bị nặng và để
        # event cuối cùng của một request vẫn là `decision` cho các reader cũ.
        self.emit("decision", payload)
        # Snapshot phát triển đầy đủ cho từng request: activity, topology,
        # trạng thái synapse và owner không bị cắt ở UI/journal.
        self.theo_doi_phat_trien_mang(
            policy, trace, giai_doan="quyet_dinh",
            synapse_state=self.synapse_state(policy),
            activity=payload["network"]["activity"],
            bo_sung={"ho_so_quyet_dinh": payload["ho_so_quyet_dinh_tom_tat"]},
        )

    def feedback(
        self,
        policy: Any,
        trace: Any,
        *,
        next_obs: Any,
        reward: float,
        cost: float,
        done: bool,
        signal: Mapping[str, Any] | None,
    ) -> None:
        if not self.enabled:
            return
        new_growth = self._new_growth_events(policy)
        synapse_state = self.synapse_state(policy)
        payload = {
            "revision": int(getattr(policy, "revision", 0)),
            "trace_id": str(getattr(trace, "trace_id", "")),
            "frame_id": str(getattr(trace, "context_frame_id", "")),
            "reward": _finite(reward),
            "cost": _finite(cost),
            "done": bool(done),
            "signal": dict(signal or {}),
            "execution": dict(getattr(trace, "execution_details", {}) or {}),
            "action": dict(getattr(trace, "action_details", {}) or {}),
            "growth_events": [_json_safe(event, max_array=0) for event in new_growth],
            "synapse_state": _json_safe(
                synapse_state,
                max_array=0,
            ),
            "topology_summary": topology_summary(policy),
            "activity": graph_activity(policy, trace, max_nodes=self.max_active_nodes, max_edges=self.max_active_edges),
            "next_obs_head": _vector(next_obs, 0),
        }
        if new_growth:
            payload["nhat_ky_cau_truc"] = self.nhat_ky_cau_truc(policy, trace, new_growth, giai_doan="feedback")
        self.emit("feedback", payload)
        self._feedback_count += 1
        # A complete feedback event is already append-only in the primary
        # stream.  Avoid serializing a second self-contained topology/state
        # snapshot for every broker acknowledgement; lifecycle and structural
        # snapshots below remain unconditional evidence of actual learning.
        snapshot_due = (
            bool(new_growth)
            or bool(done)
            or self._feedback_count % max(1, int(self.network_development_feedback_sample_every)) == 0
        )
        if snapshot_due:
            self.theo_doi_phat_trien_mang(
                policy, trace, giai_doan="feedback", synapse_state=synapse_state,
                growth_events=new_growth, activity=payload["activity"],
                bo_sung={"reward": _finite(reward), "cost": _finite(cost), "done": bool(done), "signal": dict(signal or {})},
            )
        if new_growth:
            self.emit_topology(policy, reason="growth_or_development", force=False)

    def development(self, policy: Any, trace: Any, *, actions: Any) -> None:
        if not self.enabled:
            return
        new_growth = self._new_growth_events(policy)
        synapse_state = self.synapse_state(policy)
        structural_logs = self.nhat_ky_cau_truc(policy, trace, new_growth, giai_doan="development") if new_growth else []
        development_payload = {
            "revision": int(getattr(policy, "revision", 0)),
            "trace_id": str(getattr(trace, "trace_id", "")),
            "actions": _json_safe(actions, max_array=0),
            "growth_events": [_json_safe(event, max_array=0) for event in new_growth],
            "nhat_ky_cau_truc": structural_logs,
            "synapse_state": _json_safe(synapse_state, max_array=0),
            "topology_summary": topology_summary(policy),
        }
        self.emit("development", development_payload)
        self.theo_doi_phat_trien_mang(
            policy, trace, giai_doan="phat_trien", synapse_state=synapse_state,
            growth_events=new_growth, bo_sung={"actions": _json_safe(actions, max_array=0), "nhat_ky_cau_truc": structural_logs},
        )
        if new_growth:
            self.emit_topology(policy, reason="development", force=False)

    def lifecycle(self, policy: Any, *, trace_id: str, realized_r: float, close_reason: str, duration_bars: float | None, credited: int) -> None:
        if not self.enabled:
            return
        report_fn = getattr(policy, "contextual_expectancy_report", None)
        report = report_fn() if callable(report_fn) else {}
        new_growth = self._new_growth_events(policy)
        synapse_state = self.synapse_state(policy)
        # Structural births/pruning are sent first so a UI can merge the
        # following synapse delta into known nodes and edges.
        if new_growth:
            self.emit_topology(policy, reason="lifecycle_growth", force=False)
        causal_report = {}
        causal_fn = getattr(policy, "causal_lifecycle_report", None)
        if callable(causal_fn):
            try:
                causal_report = dict(causal_fn(str(trace_id)) or {})
            except Exception:
                causal_report = {}
        structural_logs = self.nhat_ky_cau_truc(policy, getattr(policy, "_execution_trace_cache", {}).get(str(trace_id)) if isinstance(getattr(policy, "_execution_trace_cache", None), dict) else None, new_growth, giai_doan="lifecycle") if new_growth else []
        lifecycle_trace = getattr(policy, "_execution_trace_cache", {}).get(str(trace_id)) if isinstance(getattr(policy, "_execution_trace_cache", None), dict) else None
        lifecycle_payload = {
            "revision": int(getattr(policy, "revision", 0)),
            "trace_id": str(trace_id),
            "realized_net_r": _finite(realized_r),
            "close_reason": str(close_reason),
            "duration_bars": None if duration_bars is None else _finite(duration_bars),
            "credited": int(credited),
            "expectancy": _json_safe(report, max_array=0),
            "causal": _json_safe(causal_report, max_array=0),
            "growth_events": [_json_safe(event, max_array=0) for event in new_growth],
            "nhat_ky_cau_truc": structural_logs,
            "synapse_state": _json_safe(synapse_state, max_array=0),
            "topology_summary": topology_summary(policy),
        }
        self.emit("lifecycle", lifecycle_payload)
        self.theo_doi_phat_trien_mang(
            policy, lifecycle_trace, giai_doan="vong_doi_lenh", synapse_state=synapse_state,
            growth_events=new_growth, bo_sung={
                "realized_net_r": _finite(realized_r), "close_reason": str(close_reason),
                "duration_bars": None if duration_bars is None else _finite(duration_bars),
                "credited": int(credited), "expectancy": _json_safe(report, max_array=0),
                "causal": _json_safe(causal_report, max_array=0),
            },
        )
        self.emit_topology(policy, reason="lifecycle_credit", force=False)

    def trial_finished(self, policy: Any, *, policy_id: int, round_index: int, fitness: float, metrics: Mapping[str, Any] | None) -> None:
        if not self.enabled:
            return
        new_growth = self._new_growth_events(policy)
        synapse_state = self.synapse_state(policy)
        structural_logs = self.nhat_ky_cau_truc(policy, None, new_growth, giai_doan="ket_thuc_thu_nghiem") if new_growth else []
        self.emit("trial_finished", {
            "policy_id": int(policy_id),
            "round": int(round_index),
            "fitness": _finite(fitness),
            "metrics": _json_safe(dict(metrics or {}), max_array=0),
            "revision": int(getattr(policy, "revision", 0)),
            "growth_event_count": len(new_growth),
            "growth_events": [_json_safe(item, max_array=0) for item in new_growth],
            "nhat_ky_cau_truc": structural_logs,
            "synapse_state": _json_safe(synapse_state, max_array=0),
            "topology_summary": topology_summary(policy),
        })
        self.emit_topology(policy, reason="trial_finished", force=False)
        self.theo_doi_phat_trien_mang(
            policy, None, giai_doan="ket_thuc_thu_nghiem",
            synapse_state=synapse_state,
            growth_events=new_growth,
            bo_sung={
                "policy_id": int(policy_id), "round": int(round_index),
                "fitness": _finite(fitness), "metrics": _json_safe(dict(metrics or {}), max_array=0),
                "nhat_ky_cau_truc": structural_logs,
            },
        )


class LiveEventReader:
    """Incremental reader so browser polling never reparses the whole journal."""

    def __init__(self, path: str | Path, *, max_events: int = 0) -> None:
        self.path = Path(path)
        self.offset = 0
        # 0 = giữ toàn bộ event trong reader, đồng bộ hợp đồng V9 không cắt UI/audit.
        self.events: deque[dict[str, Any]] = deque() if int(max_events) <= 0 else deque(maxlen=max(100, int(max_events)))
        self.topology: dict[str, Any] | None = None
        self.latest: dict[str, dict[str, Any]] = {}
        self.last_error = ""
        self.writer_status: dict[str, Any] = {}

    def _status_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".status.json")

    def _refresh_writer_status(self) -> None:
        status_path = self._status_path()
        if not status_path.exists():
            self.writer_status = {}
            return
        try:
            parsed = json.loads(status_path.read_text(encoding="utf-8", errors="ignore"))
            self.writer_status = dict(parsed) if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            # A status sidecar can be atomically replaced while the UI reads.
            # Keep the last known good state instead of pretending the writer
            # has no status.
            return

    def _reset(self) -> None:
        self.offset = 0
        self.events.clear()
        self.topology = None
        self.latest = {}

    def refresh(self) -> None:
        self._refresh_writer_status()
        if not self.path.exists():
            return
        try:
            size = self.path.stat().st_size
            if size < self.offset:
                self._reset()
            with self.path.open("r", encoding="utf-8", errors="ignore") as handle:
                handle.seek(self.offset)
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    self.events.append(event)
                    kind = str(event.get("event_type", ""))
                    payload = event.get("payload", {})
                    if kind == "topology" and isinstance(payload, dict) and isinstance(payload.get("topology"), dict):
                        self.topology = payload["topology"]
                    elif isinstance(payload, dict) and isinstance(payload.get("synapse_state"), dict):
                        self.topology = apply_synapse_state_delta(self.topology, payload.get("synapse_state"))
                    if kind:
                        self.latest[kind] = event
                self.offset = handle.tell()
            self.last_error = ""
        except OSError as exc:
            self.last_error = str(exc)

    def state(self, *, limit: int = 180) -> dict[str, Any]:
        self.refresh()
        events = list(self.events)[-max(1, int(limit)):]
        event_counts: dict[str, int] = {}
        for item in events:
            key = str(item.get("event_type", "unknown"))
            event_counts[key] = event_counts.get(key, 0) + 1
        return {
            "schema_version": LIVE_SCHEMA_VERSION,
            "path": str(self.path),
            "exists": self.path.exists(),
            "last_error": self.last_error,
            "topology": self.topology or {"nodes": [], "edges": [], "node_count": 0, "edge_count": 0},
            "latest": self.latest,
            "events": events,
            "event_counts": event_counts,
            "writer_status": dict(self.writer_status or {}),
            "server_time": time.time(),
        }


_DASHBOARD_HTML = r"""<!doctype html>
<html lang="vi">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Self-Growing Trader — Live Neural Observatory</title>
<style>
:root{--bg:#0b0e14;--panel:#111722;--line:#273246;--text:#e9eef8;--muted:#94a3b8;--active:#45e0a8;--warn:#ffbd59;--bad:#ff6b6b;--blue:#6aa8ff}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font:13px/1.4 Segoe UI,Arial,sans-serif;overflow:hidden}
#graph{position:fixed;left:0;top:0;bottom:0;right:390px;width:calc(100vw - 390px);height:100vh;background:radial-gradient(circle at 45% 42%,#132035 0,#0b0e14 60%)}
#panel{position:fixed;top:0;right:0;width:390px;bottom:0;overflow:auto;background:rgba(15,21,32,.97);border-left:1px solid var(--line);padding:14px}
h1{font-size:16px;margin:0 0 3px}.muted{color:var(--muted)}.row{display:flex;gap:8px;align-items:center;margin:8px 0}.row>*{flex:1}button,select{background:#0b111a;color:var(--text);border:1px solid var(--line);border-radius:6px;padding:7px;cursor:pointer}button.on{border-color:var(--active);color:var(--active)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.card{background:#0b111a;border:1px solid var(--line);border-radius:7px;padding:8px}.card b{display:block;font-size:16px}.section{border-top:1px solid var(--line);margin-top:12px;padding-top:12px}.label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}.kv{display:grid;grid-template-columns:130px 1fr;gap:4px 8px;word-break:break-word}.kv span{color:var(--muted)}.list{max-height:210px;overflow:auto;border:1px solid var(--line);border-radius:6px}.item{padding:7px 8px;border-bottom:1px solid rgba(39,50,70,.6);cursor:pointer}.item:hover{background:#172237}.good{color:var(--active)}.warn{color:var(--warn)}.bad{color:var(--bad)}.tiny{font-size:11px}.legend{position:fixed;left:14px;bottom:14px;background:rgba(11,17,26,.82);border:1px solid var(--line);padding:7px 9px;border-radius:6px;color:var(--muted)}
</style></head>
<body><canvas id="graph"></canvas>
<div id="panel"><h1>Live Neural Observatory</h1><div id="status" class="muted">Đang chờ telemetry…</div>
<div class="row"><button id="pause">Tạm dừng khung hình</button><button id="active">Chỉ đường hoạt hóa</button></div>
<div class="grid" id="stats"></div>
<div class="section"><div class="label">Quyết định hiện tại</div><div id="decision" class="kv"></div></div>
<div class="section"><div class="label">Context & EXP</div><div id="context" class="kv"></div></div>
<div class="section"><div class="label">Node / Edge được chọn</div><div id="selection" class="kv"><span>Click</span><b>một node hoặc edge</b></div></div>
<div class="section"><div class="label">Sự kiện phát triển</div><div id="events" class="list"></div></div>
<div class="section"><div class="label">Hoạt hóa theo region</div><div id="regions" class="list"></div></div>
</div><div class="legend">Node sáng = đang hoạt hóa · đường xanh = trace hiện tại · vàng = candidate · đỏ = lifecycle pruning</div>
<script>
const canvas=document.getElementById('graph'),ctx=canvas.getContext('2d');let W=1,H=1,state=null,paused=false,activeOnly=false,selected=null,positions=new Map(),hover=null;
const regionOrder={input:0,encoder:1,memory:2,hypothesis:3,predictor:4,critic:5,selector:6,meta:7,output:8};
function resize(){W=canvas.width=canvas.clientWidth*devicePixelRatio;H=canvas.height=canvas.clientHeight*devicePixelRatio}addEventListener('resize',resize);resize();
function fmt(v,d=3){return Number.isFinite(+v)?(+v).toFixed(d):String(v??'')};
function esc(v){return String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function norm(a){let m=0;for(const x of a||[])m=Math.max(m,Math.abs(+x||0));return m||1}
function layout(top){const groups={};for(const n of top.nodes||[]){(groups[n.region]??=[]).push(n)};positions=new Map();for(const [region,items] of Object.entries(groups)){const layer=regionOrder[region]??4;items.sort((a,b)=>a.id.localeCompare(b.id));items.forEach((n,i)=>{const y=(i+1)/(items.length+1);positions.set(n.id,{x:(.08+.84*(layer/8))*W,y:y*H})})}}
function currentActivity(){return state?.latest?.decision?.payload?.network?.activity||state?.latest?.feedback?.payload?.activity||{nodes:[],edges:[],region_activity:{}}}
function draw(){ctx.clearRect(0,0,W,H);if(!state||!state.topology)return requestAnimationFrame(draw);const top=state.topology,act=currentActivity(),an=new Map((act.nodes||[]).map(x=>[x.id,x])),ae=new Map((act.edges||[]).map(x=>[x.id,x]));
for(const e of top.edges||[]){const a=positions.get(e.src),b=positions.get(e.dst);if(!a||!b)continue;const live=ae.get(e.id),isActive=!!live; if(activeOnly&&!isActive)continue;let weight=Math.abs(+e.weight||0),w=(isActive?2.5:Math.min(1.6,.35+weight))*devicePixelRatio;ctx.strokeStyle=isActive?'rgba(69,224,168,.92)':(!e.enabled?'rgba(110,120,140,.14)':(+e.weight<0?'rgba(255,107,107,.33)':'rgba(106,168,255,.28)'));ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
for(const n of top.nodes||[]){const p=positions.get(n.id);if(!p)continue;const live=an.get(n.id),isActive=!!live;if(activeOnly&&!isActive)continue;const activation=Math.abs(+(live?.activation??0));const r=(isActive?4.8+6*Math.min(1,activation):3.1)*devicePixelRatio;const c=regionColor(n.region);ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fillStyle=isActive?'rgba(69,224,168,.98)':c;ctx.fill();if(selected&&selected.id===n.id){ctx.strokeStyle='white';ctx.lineWidth=1.8*devicePixelRatio;ctx.stroke()}}
requestAnimationFrame(draw)}
function regionColor(r){return({input:'#94a3b8',encoder:'#60a5fa',memory:'#a78bfa',hypothesis:'#f472b6',predictor:'#22d3ee',critic:'#ffbd59',selector:'#fb7185',output:'#f8fafc'})[r]||'#7dd3fc'}draw();
canvas.addEventListener('click',e=>{if(!state)return;const x=e.offsetX*devicePixelRatio,y=e.offsetY*devicePixelRatio;let best=null,bd=1e9;for(const n of state.topology.nodes||[]){const p=positions.get(n.id);if(!p)continue;const d=Math.hypot(x-p.x,y-p.y);if(d<bd&&d<15*devicePixelRatio){best=n;bd=d}}if(best){selected=best;renderSelection(best)}});
document.getElementById('pause').onclick=e=>{paused=!paused;e.currentTarget.classList.toggle('on',paused);e.currentTarget.textContent=paused?'Tiếp tục cập nhật':'Tạm dừng khung hình'};document.getElementById('active').onclick=e=>{activeOnly=!activeOnly;e.currentTarget.classList.toggle('on',activeOnly)};
function renderSelection(x){const live=currentActivity(),a=(live.nodes||[]).find(n=>n.id===x.id)||(live.edges||[]).find(n=>n.id===x.id)||{};document.getElementById('selection').innerHTML=Object.entries({...x,...a}).filter(([k])=>!['src','dst'].includes(k)||true).map(([k,v])=>`<span>${esc(k)}</span><b>${esc(typeof v==='object'?JSON.stringify(v):v)}</b>`).join('')}
function kv(id,obj){document.getElementById(id).innerHTML=Object.entries(obj).map(([k,v])=>`<span>${esc(k)}</span><b>${esc(typeof v==='object'?JSON.stringify(v):v)}</b>`).join('')}
function render(s){if(paused)return;state=s;layout(s.topology||{nodes:[]});const top=s.topology||{},d=s.latest?.decision?.payload||{},a=d.action||{},c=d.context||{},ex=d.expectancy||{},n=d.network||{},fr=d.five_reasoning||{},activity=n.activity||currentActivity();document.getElementById('status').textContent=s.exists?`${s.path} · ${top.node_count||0} neuron · ${top.edge_count||0} liên kết`:'Chưa có event file. Hãy chạy train với --live-observatory.';
const stats=[['Neuron',top.node_count||0],['Connections',top.edge_count||0],['Active node',activity.active_node_count||0],['Active edge',activity.active_edge_count||0],['EXP LCB',fmt(ex.predicted_lcb,4)],['Confidence',fmt(a.confidence,3)]];document.getElementById('stats').innerHTML=stats.map(([k,v])=>`<div class="card"><b>${esc(v)}</b><span class="muted tiny">${esc(k)}</span></div>`).join('');
const reasoningSteps=(fr.steps||[]).length||0;const reasoningScale=+fr.action_scale||0;const reasoningLabel=reasoningScale===0?`${reasoningSteps}-step telemetry · action scale 0`:`${reasoningSteps}-step active · action scale ${fmt(reasoningScale,3)}`;
kv('decision',{action:a.name||'—',selected_hypothesis:a.selected_index??'—',candidate_scores:(a.candidate_scores||[]).map(v=>fmt(v,3)).join(' · '),reasoning_rl:fr.enabled?reasoningLabel:'tắt',reasoning_steps:(fr.steps||[]).join(' → '),reasoning_final:fr.final_hypothesis??'—',critic:fmt(a.critic_value),novelty:fmt(n.novelty),conflict:fmt(n.conflict),prediction_error:fmt(n.prediction_error),memory_match:fmt(n.memory_match_quality)});
kv('context',{frame:d.frame_id||'—',allowed_actions:(a.allowed_actions||[]).join(', '),EXP_LCB:fmt(ex.predicted_lcb,4),EXP_mean_R:fmt(ex.predicted_mean_net_r,4),EXP_samples:fmt(ex.samples,0),EXP_regret:fmt(ex.regret,4),best_action:ex.best_action??'—',learning_drive:(c.learning_drive||[]).map(v=>fmt(v,2)).join(' · ')});
const events=(s.events||[]).filter(x=>['feedback','development','lifecycle','trial_started','trial_finished','topology'].includes(x.event_type)).slice(-80).reverse();document.getElementById('events').innerHTML=events.map(e=>{const p=e.payload||{},g=(p.growth_events||[]),d=p.synapse_state||{},changes=(+d.node_change_count||0)+(+d.edge_change_count||0)+(+d.removed_node_count||0)+(+d.removed_edge_count||0);const growthName=g.map(x=>x.type||x.kind||x.action||'event').join(', ');const title=g.length?`${e.event_type} · ${growthName}`:(changes?`${e.event_type} · synapse Δ${changes}`:e.event_type);return `<div class="item"><b class="${g.length||changes?'warn':''}">${esc(title)}</b><br><span class="muted tiny">${new Date((e.created_at||0)*1000).toLocaleTimeString()} · ${esc(p.trace_id||p.policy_id||'')}</span></div>`}).join('')||'<div class="item muted">Chưa có feedback/growth event</div>';
const regions=activity.region_activity||{};document.getElementById('regions').innerHTML=Object.entries(regions).sort().map(([k,v])=>`<div class="item"><b>${esc(k)}</b><span class="muted"> · ${fmt(v,4)}</span></div>`).join('')||'<div class="item muted">Chưa có trace</div>'}
async function poll(){try{const r=await fetch('/api/state?limit=220',{cache:'no-store'});if(r.ok)render(await r.json())}catch(e){}setTimeout(poll,750)}poll();
</script></body></html>"""


def serve_live_observatory(
    event_path: str | Path = DEFAULT_EVENT_PATH,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = True,
) -> None:
    reader = LiveEventReader(event_path)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:  # silence polling noise
            return

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", _DASHBOARD_HTML.encode("utf-8"))
                return
            if path == "/api/state":
                body = json.dumps(reader.state(), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                self._send(HTTPStatus.OK, "application/json; charset=utf-8", body)
                return
            self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found")

    server = ThreadingHTTPServer((str(host), int(port)), Handler)
    url = f"http://{host}:{int(port)}/"
    print(f"[Observatory] reading {Path(event_path)}", flush=True)
    print(f"[Observatory] open {url}", flush=True)
    if open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


__all__ = [
    "DEFAULT_EVENT_PATH",
    "LIVE_SCHEMA_VERSION",
    "LiveEventReader",
    "LiveObservabilityConfig",
    "apply_synapse_state_delta",
    "LiveTelemetryWriter",
    "graph_activity",
    "graph_topology",
    "serve_live_observatory",
]
