from __future__ import annotations

from dataclasses import replace

from self_growing_trader.config import AppConfig
from self_growing_trader.desktop_app import project_live_graph, summarize_live_state, with_ui_observability


def _state() -> dict:
    return {
        "exists": True,
        "topology": {
            "node_count": 4,
            "edge_count": 4,
            "nodes": [
                {"id": "n1", "region": "input", "utility": 0.1},
                {"id": "n2", "region": "memory", "utility": 0.2},
                {"id": "n3", "region": "selector", "utility": 0.3},
                {"id": "n4", "region": "output", "utility": 0.4},
            ],
            "edges": [
                {"id": "e1", "src": "n1", "dst": "n2", "weight": 0.2, "enabled": True},
                {"id": "e2", "src": "n2", "dst": "n3", "weight": 0.3, "enabled": True},
                {"id": "e3", "src": "n3", "dst": "n4", "weight": 0.4, "enabled": True},
                {"id": "e4", "src": "n1", "dst": "n4", "weight": -0.5, "enabled": True},
            ],
        },
        "latest": {
            "decision": {
                "payload": {
                    "action": {"name": "MARKET_BUY", "confidence": 0.75},
                    "expectancy": {"predicted_lcb": 0.13, "samples": 9},
                    "network": {
                        "activity": {
                            "active_node_count": 2,
                            "active_edge_count": 1,
                            "nodes": [{"id": "n2", "activation": 0.9}, {"id": "n4", "activation": 0.3}],
                            "edges": [{"id": "e2", "contribution": 0.7}],
                        }
                    },
                }
            }
        },
        "events": [
            {"event_type": "feedback", "payload": {"growth_events": [{"type": "add_neuron"}]}},
        ],
    }


def test_embedded_ui_projection_keeps_active_trace_under_display_cap() -> None:
    state = _state()
    activity = state["latest"]["decision"]["payload"]["network"]["activity"]
    nodes, edges, active_nodes, active_edges = project_live_graph(
        state["topology"], activity, max_nodes=2, max_edges=2
    )
    assert {node["id"] for node in nodes} == {"n2", "n4"}
    assert active_nodes["n2"]["activation"] == 0.9
    assert active_edges["e2"]["contribution"] == 0.7
    assert edges == []  # e2 has n3 outside the hard node display cap.


def test_embedded_ui_summary_uses_actual_policy_trace_and_exp() -> None:
    summary = summarize_live_state(_state())
    assert summary["nodes"] == 4
    assert summary["action"] == "MARKET_BUY"
    assert summary["exp_lcb"] == 0.13
    assert summary["active_nodes"] == 2
    assert summary["growth_events"] == 1


def test_ui_observability_is_read_only_config_override() -> None:
    config = AppConfig()
    updated = with_ui_observability(
        config,
        enabled=True,
        event_path="outputs/ui_live.jsonl",
        decision_sample_every=3,
    )
    assert updated.observability.enabled is True
    assert updated.observability.event_path.endswith("ui_live.jsonl")
    assert updated.observability.decision_sample_every == 3
    assert updated.observability.reset_on_process_start is False
    assert updated.orders == config.orders


def test_embedded_ui_never_drops_active_nodes_to_satisfy_passive_display_cap() -> None:
    state = _state()
    activity = state["latest"]["decision"]["payload"]["network"]["activity"]
    activity = {**activity, "nodes": [
        {"id": "n1", "activation": 0.1},
        {"id": "n2", "activation": 0.9},
        {"id": "n4", "activation": 0.3},
    ]}
    nodes, _edges, active_nodes, _active_edges = project_live_graph(
        state["topology"], activity, max_nodes=1, max_edges=1
    )
    assert {node["id"] for node in nodes} == {"n1", "n2", "n4"}
    assert set(active_nodes) == {"n1", "n2", "n4"}


def test_live_view_toggle_labels_and_status_do_not_imply_train_control() -> None:
    from self_growing_trader.desktop_app import live_view_button_label, live_view_status_text

    assert live_view_button_label(False) == "Bật xem Live"
    assert live_view_button_label(True) == "Tắt xem Live"
    assert "train" in live_view_status_text(
        enabled=False,
        training=True,
        telemetry_available=True,
    ).lower()
    assert "trace thật" in live_view_status_text(
        enabled=True,
        training=False,
        telemetry_available=True,
    ).lower()
