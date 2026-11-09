from __future__ import annotations

import numpy as np

from self_growing_trader.graph_plasticity import GraphCreditSignal, GraphPlasticityEngine
from self_growing_trader.neural_graph import EDGE_FORWARD, EDGE_GATING, EDGE_PLASTIC, REGION_CRITIC, REGION_INPUT, NeuralGraph


def _credit_graph() -> tuple[NeuralGraph, str, str, str]:
    graph = NeuralGraph()
    src = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    out = graph.add_neuron(REGION_CRITIC, module_id="risk_gate_specialist", lifecycle="candidate")
    good = graph.add_connection(
        src,
        out,
        weight=0.5,
        gate=-2.0,
        kind=EDGE_FORWARD,
        created_by="test",
        need_reason="stress_entry_loss",
        candidate_id="cand",
    )
    idle = graph.add_connection(
        src,
        out,
        weight=-0.5,
        gate=-2.0,
        kind=EDGE_FORWARD,
        created_by="test",
        need_reason="idle",
        candidate_id="cand",
    )
    graph.activate(np.array([1.0]))
    return graph, src, good, idle


def test_graph_plasticity_updates_only_active_edges_and_neurons() -> None:
    graph, src, active_edge, idle_edge = _credit_graph()
    engine = GraphPlasticityEngine()

    report = engine.apply_credit(
        graph,
        GraphCreditSignal(reward=1.0, td_delta=0.5),
        active_edge_ids=[active_edge],
        active_neuron_ids=[src],
    )

    assert report.updated_edges == 1
    assert report.updated_neurons == 1
    assert graph.edges[active_edge].utility > 0.0
    assert graph.edges[active_edge].gate > -2.0
    assert graph.edges[idle_edge].utility == 0.0
    assert graph.neurons[src].utility > 0.0


def test_graph_plasticity_disables_bad_candidate_edge() -> None:
    graph, _src, active_edge, _idle_edge = _credit_graph()
    engine = GraphPlasticityEngine(min_utility=-0.01)

    report = engine.apply_credit(
        graph,
        GraphCreditSignal(reward=-5.0, cost=1.0, lifecycle_penalty=2.0, account_stress=1.0),
        active_edge_ids=[active_edge],
        candidate_mode=True,
    )

    assert report.disabled_edges == 1
    assert graph.edges[active_edge].enabled is False


def test_graph_plasticity_remains_finite_after_many_updates() -> None:
    graph, src, active_edge, _idle_edge = _credit_graph()
    engine = GraphPlasticityEngine(gate_lr=0.5, weight_lr=0.5)

    for idx in range(100):
        signal = GraphCreditSignal(reward=1.0 if idx % 2 == 0 else -1.0, td_delta=0.3, prediction_error=0.2)
        engine.apply_credit(graph, signal, active_edge_ids=[active_edge], active_neuron_ids=[src], candidate_mode=False)

    edge = graph.edges[active_edge]
    assert np.isfinite(edge.weight)
    assert np.isfinite(edge.gate)
    assert -engine.weight_clip <= edge.weight <= engine.weight_clip
    assert -engine.gate_clip <= edge.gate <= engine.gate_clip


def test_graph_plasticity_consolidates_high_utility_edges() -> None:
    graph, _src, active_edge, _idle_edge = _credit_graph()
    graph.edges[active_edge].utility = 1.0
    before = graph.edges[active_edge].plasticity

    count = GraphPlasticityEngine().consolidate_high_utility(graph, threshold=0.5)

    assert count == 1
    assert graph.edges[active_edge].plasticity < before


def test_graph_plasticity_reuses_homeostatic_bias_without_touching_inactive_nodes() -> None:
    graph, inactive_node, active_edge, _idle_edge = _credit_graph()
    active_dst = graph.edges[active_edge].dst
    active_before = graph.neurons[active_dst].bias
    inactive_before = graph.neurons[inactive_node].bias

    report = GraphPlasticityEngine().apply_credit(
        graph,
        GraphCreditSignal(reward=0.8, td_delta=0.2),
        active_edge_ids=(active_edge,),
        active_neuron_ids=(active_dst,),
        candidate_mode=True,
    )

    assert report.updated_neurons == 1
    assert graph.neurons[active_dst].bias != active_before
    assert graph.neurons[inactive_node].bias == inactive_before


def test_edge_kinds_have_distinct_plasticity_response() -> None:
    graph = NeuralGraph()
    src = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    dst = graph.add_neuron(REGION_CRITIC, module_id="risk_gate_specialist", lifecycle="candidate")
    forward = graph.add_connection(src, dst, weight=0.5, gate=-2.0, kind=EDGE_FORWARD, candidate_id="cand")
    gating = graph.add_connection(src, dst, weight=0.5, gate=-2.0, kind=EDGE_GATING, candidate_id="cand")
    plastic = graph.add_connection(src, dst, weight=0.5, gate=-2.0, kind=EDGE_PLASTIC, candidate_id="cand")
    graph.activate(np.array([1.0]))

    GraphPlasticityEngine(gate_lr=0.1, weight_lr=0.1).apply_credit(
        graph,
        GraphCreditSignal(reward=1.0, td_delta=0.5),
        active_edge_ids=(forward, gating, plastic),
        active_neuron_ids=(src, dst),
        candidate_mode=True,
    )

    forward_gate_delta = graph.edges[forward].gate - (-2.0)
    gating_gate_delta = graph.edges[gating].gate - (-2.0)
    forward_weight_delta = abs(graph.edges[forward].weight - 0.5)
    plastic_weight_delta = abs(graph.edges[plastic].weight - 0.5)
    assert gating_gate_delta > forward_gate_delta
    assert plastic_weight_delta > forward_weight_delta
