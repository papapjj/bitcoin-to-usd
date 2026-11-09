from __future__ import annotations

import copy
import json
import pickle
from types import SimpleNamespace

from self_growing_trader.model_viewer import export_model_view_data, export_model_viewer_html
from self_growing_trader.module_need import ROLE_CRITIC_SPECIALIST
from self_growing_trader.self_growing_runner import SelfGrowingPolicy
from self_growing_trader.specialist_module_factory import SpecialistModuleFactory


def _seed_policy(tmp_path):
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=4, hidden_size=6, seed=91, m1_backend="graph")
    trace = policy.act([0.1, -0.2, 0.3, -0.4, 0.5])
    policy_path = tmp_path / "policy.pkl"
    with policy_path.open("wb") as handle:
        pickle.dump(policy, handle)
    journal_path = tmp_path / "journal.jsonl"
    journal_path.write_text(json.dumps({
        "trace": {
            "active_neuron_ids": list(trace.active_neuron_ids[:3]),
            "active_edge_ids": list(trace.active_edge_ids[:3]),
        }
    }) + "\n", encoding="utf-8")
    return policy, policy_path, journal_path


def test_model_viewer_exports_graph_nodes_edges_and_html(tmp_path) -> None:
    _, policy_path, journal_path = _seed_policy(tmp_path)
    data = export_model_view_data(str(policy_path), journal_path=str(journal_path), max_nodes=200, max_edges=300)

    assert data["summary"]["nodes"] > 0
    assert data["summary"]["edges"] > 0
    assert data["summary"]["active_nodes"] > 0
    assert data["summary"]["active_edges"] > 0
    assert data["summary"]["edge_kinds"]
    assert data["summary"]["active_edge_kinds"]
    assert data["namespaced_ids"] is True
    assert all("::" in node["id"] for node in data["nodes"])
    assert all("::" in edge["id"] and "::" in edge["src"] and "::" in edge["dst"] for edge in data["edges"])
    assert any("degree_in" in node and "active_connections" in node for node in data["nodes"])
    assert any(edge["active"] and edge["src"] and edge["dst"] and edge["kind"] for edge in data["edges"])

    html_path = tmp_path / "viewer.html"
    result = export_model_viewer_html(str(policy_path), str(html_path), journal_path=str(journal_path), max_nodes=200, max_edges=300)
    html = html_path.read_text(encoding="utf-8")

    assert result["output"] == str(html_path)
    assert "Self-Growing Trader Model 3D" in html
    assert "model-data" in html
    assert "Specialist circuits" in html
    assert "module_groups" in html
    assert "requestAnimationFrame(draw)" not in html


def test_model_viewer_exports_specialist_circuit_and_child_graph(tmp_path) -> None:
    policy, _, journal_path = _seed_policy(tmp_path)
    graph = policy.modules[0].m1_component.graph
    build = SpecialistModuleFactory().create(
        graph,
        module_id="specialist:critic:demo",
        role=ROLE_CRITIC_SPECIALIST,
        reason="demo",
        candidate_id="candidate-demo",
        trial_id="trial-demo",
        context_key="high_vol_breakout",
        context_vector=None,
        causal_confidence=0.72,
        causal_validated=True,
    )
    # A child graph deliberately reuses local e0/e1 IDs. V22 must namespace it
    # independently rather than overwriting root/reasoning edge identities.
    child_graph = copy.deepcopy(graph)
    policy.child_supernetwork_manager = SimpleNamespace(children={
        "demo": SimpleNamespace(network=SimpleNamespace(graph=child_graph), status="candidate", selectable=False),
    })
    policy_path = tmp_path / "policy_with_modules.pkl"
    with policy_path.open("wb") as handle:
        pickle.dump(policy, handle)

    data = export_model_view_data(str(policy_path), journal_path=str(journal_path), max_nodes=2500, max_edges=12000)
    groups = {group["id"]: group for group in data["module_groups"]}
    specialist = next(group for group in data["specialist_modules"] if group["local_module_id"] == build.module_id)

    assert specialist["is_specialist"] is True
    assert specialist["node_ids"]
    assert len(specialist["node_ids"]) == 9
    assert specialist["role"] == ROLE_CRITIC_SPECIALIST
    assert specialist["causal_validated"] is True
    assert specialist["bounds"]
    assert all(node_id in {node["id"] for node in data["nodes"]} for node_id in specialist["node_ids"])
    assert any(edge["is_module_internal"] for edge in data["edges"])
    assert any(graph["kind"] == "child" for graph in data["graphs"])
    assert len({edge["id"] for edge in data["edges"]}) == len(data["edges"])
    assert groups[specialist["id"]]["edge_ids"]
