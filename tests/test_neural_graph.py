from __future__ import annotations

import pickle

import numpy as np
import pytest

from self_growing_trader.neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_LATERAL,
    EDGE_META,
    EDGE_PLASTIC,
    EDGE_PREDICTIVE,
    EDGE_RECURRENT,
    EDGE_SELECTOR,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_INPUT,
    REGION_MEMORY,
    REGION_OUTPUT,
    REGION_SELECTOR,
    NeuralGraph,
    NeuralGraphBudget,
)


def test_neural_graph_activate_and_validate_output_shape() -> None:
    graph = NeuralGraph()
    i0 = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    i1 = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    enc = graph.add_neuron(REGION_ENCODER, module_id="m1", lifecycle="active")
    out = graph.add_neuron(REGION_OUTPUT, module_id="m1", lifecycle="active")

    graph.add_connection(i0, enc, weight=0.5, gate=3.0, kind=EDGE_FORWARD)
    graph.add_connection(i1, enc, weight=-0.25, gate=3.0, kind=EDGE_FORWARD)
    graph.add_connection(enc, out, weight=1.2, gate=3.0, kind=EDGE_FORWARD)

    output = graph.activate(np.array([1.0, -1.0]), steps=2)

    assert output.shape == (1,)
    assert np.isfinite(output).all()
    assert output[0] > 0.0
    graph.assert_invariants(output_size=1)


def test_neural_graph_copy_is_deep_and_pickleable() -> None:
    graph = NeuralGraph()
    src = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    dst = graph.add_neuron(REGION_OUTPUT, module_id="m1", lifecycle="active")
    edge = graph.add_connection(src, dst, weight=0.7, gate=2.0, kind=EDGE_FORWARD)

    clone = graph.copy()
    clone.edges[edge].weight = -2.0
    clone.neurons[dst].bias = 3.0

    assert graph.edges[edge].weight == 0.7
    assert graph.neurons[dst].bias == 0.0

    loaded = pickle.loads(pickle.dumps(graph))
    loaded.assert_invariants(output_size=1)
    assert loaded.edges[edge].weight == 0.7


def test_neural_graph_allows_flexible_module_to_m1_targets_but_rejects_raw_input() -> None:
    graph = NeuralGraph()
    specialist = graph.add_neuron(REGION_MEMORY, module_id="memory_specialist", lifecycle="candidate")
    m1_input = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    m1_encoder = graph.add_neuron(REGION_ENCODER, module_id="m1", lifecycle="active")
    m1_critic = graph.add_neuron(REGION_CRITIC, module_id="m1", lifecycle="active")

    graph.add_connection(specialist, m1_encoder, weight=0.05, gate=-3.0, kind=EDGE_FORWARD)

    edge = graph.add_connection(
        specialist,
        m1_critic,
        weight=0.05,
        gate=-3.0,
        kind=EDGE_CRITIC,
        created_by="module_need",
        need_reason="low_memory_match",
        candidate_id="cand-1",
    )

    assert graph.edges[edge].src_module_id == "memory_specialist"
    assert graph.edges[edge].dst_module_id == "m1"
    assert graph.edges[edge].edge_kind == EDGE_CRITIC
    assert graph.edges[edge].need_reason == "low_memory_match"

    with pytest.raises(ValueError, match="not allowed"):
        graph.add_connection(specialist, m1_input, weight=0.05, gate=-3.0, kind=EDGE_FORWARD)
    graph.assert_invariants()


def test_neural_graph_allows_m1_to_flexible_module_regions() -> None:
    graph = NeuralGraph()
    m1_encoder = graph.add_neuron(REGION_ENCODER, module_id="m1", lifecycle="active")
    module_memory = graph.add_neuron(REGION_MEMORY, module_id="position_manager", lifecycle="candidate")
    module_output = graph.add_neuron(REGION_OUTPUT, module_id="position_manager", lifecycle="candidate")

    graph.add_connection(m1_encoder, module_memory, weight=0.1, gate=-2.0, kind=EDGE_FORWARD)
    graph.add_connection(m1_encoder, module_output, weight=0.1, gate=-2.0, kind=EDGE_FORWARD)
    graph.assert_invariants()


def test_neural_graph_cross_module_budget_and_recurrent_edges_are_clamped() -> None:
    graph = NeuralGraph(
        budget=NeuralGraphBudget(
            max_neurons=8,
            max_connections=8,
            max_recurrent_edges=1,
            max_lateral_edges=2,
            max_cross_module_edges=1,
            max_m1_module_edges=1,
        )
    )
    inp = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    mem = graph.add_neuron(REGION_MEMORY, module_id="m1", lifecycle="active")
    sel = graph.add_neuron(REGION_SELECTOR, module_id="selector_specialist", lifecycle="candidate")
    out = graph.add_neuron(REGION_OUTPUT, module_id="m1", lifecycle="active")

    graph.add_connection(mem, mem, weight=50.0, gate=10.0, kind=EDGE_RECURRENT)
    graph.add_connection(sel, out, weight=0.1, gate=-3.0, kind=EDGE_GATING)

    with pytest.raises(ValueError, match="recurrent edge budget"):
        graph.add_connection(out, out, weight=1.0, kind=EDGE_RECURRENT)

    with pytest.raises(ValueError, match="cross-module edge budget|M1-module edge budget"):
        graph.add_connection(inp, sel, weight=0.1, kind=EDGE_FORWARD)

    for _ in range(10):
        output = graph.activate(np.array([1.0]), steps=3)
        assert np.isfinite(output).all()


def test_neural_graph_enforces_total_neuron_budget() -> None:
    graph = NeuralGraph(
        budget=NeuralGraphBudget(
            max_neurons=1,
            max_recurrent_edges=8,
            max_lateral_edges=8,
            max_cross_module_edges=8,
            max_m1_module_edges=8,
        )
    )

    graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    with pytest.raises(ValueError, match="neuron budget exceeded"):
        graph.add_neuron(REGION_OUTPUT, module_id="m1", lifecycle="active")


def test_neural_graph_bounds_trace_but_preserves_total_activity_count() -> None:
    graph = NeuralGraph(trace_max_edges=3, trace_candidate_edges=1)
    source = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    outputs = [graph.add_neuron(REGION_OUTPUT, module_id="m1", lifecycle="active") for _ in range(8)]
    for index, destination in enumerate(outputs):
        graph.add_connection(source, destination, weight=1.0 + 0.1 * index, gate=4.0, kind=EDGE_FORWARD)

    _output, trace = graph.activate(np.array([1.0]), return_trace=True)

    assert trace.active_edge_count == 8
    assert trace.traced_edge_count == 3
    assert len(trace.active_edge_ids) == 3
    assert len(trace.edge_contributions) == 3


def test_m1_trace_reserves_edges_for_later_cognitive_routes() -> None:
    """A dense sensory fan-in must not hide the rest of M1 from live telemetry."""
    from self_growing_trader.m1_graph_factory import seed_m1_graph

    graph = seed_m1_graph(
        input_size=119,
        output_size=20,
        state_size=24,
        hypothesis_count=4,
        seed=7,
    )
    _output, trace = graph.activate(
        np.linspace(-1.0, 1.0, 119, dtype=float),
        steps=3,
        return_trace=True,
    )
    routes = {
        (graph.neurons[graph.edges[edge_id].src].region,
         graph.neurons[graph.edges[edge_id].dst].region,
         graph.edges[edge_id].kind)
        for edge_id in trace.active_edge_ids
    }

    assert len(trace.active_edge_ids) <= graph.trace_max_edges + graph.trace_candidate_edges
    assert ("input", "encoder", EDGE_FORWARD) in routes
    assert ("input", "memory", "memory") in routes
    assert ("encoder", "hypothesis", EDGE_FORWARD) in routes
    assert ("memory", "hypothesis", "memory") in routes
    assert ("hypothesis", "output", EDGE_GATING) in routes
    assert ("predictor", "critic", EDGE_CRITIC) in routes
    assert ("critic", "selector", "selector") in routes


def _single_edge_contribution(kind: str, *, context: dict[str, list[float]] | None = None) -> float:
    graph = NeuralGraph()
    src = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    dst = graph.add_neuron(REGION_OUTPUT, bias=0.25, module_id="m1", lifecycle="active")
    edge = graph.add_connection(src, dst, weight=0.5, gate=2.0, kind=kind)

    _output, trace = graph.activate(np.array([1.0]), steps=2, context_by_region=context, return_trace=True)

    return trace.edge_contributions[edge]


def test_edge_kinds_have_distinct_runtime_semantics() -> None:
    forward = _single_edge_contribution(EDGE_FORWARD)

    assert _single_edge_contribution(EDGE_GATING) > forward
    assert _single_edge_contribution(EDGE_PREDICTIVE) > forward
    assert _single_edge_contribution(EDGE_SELECTOR) > forward
    assert _single_edge_contribution(EDGE_PLASTIC) > forward
    assert _single_edge_contribution(EDGE_META, context={REGION_OUTPUT: [1.0]}) > forward
    assert _single_edge_contribution(EDGE_LATERAL) < forward
