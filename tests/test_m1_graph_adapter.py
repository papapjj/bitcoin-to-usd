from __future__ import annotations

import numpy as np

from self_growing_trader.m1_graph_adapter import m1_to_neural_graph
from self_growing_trader.neural_graph import EDGE_FORWARD, REGION_ENCODER, REGION_INPUT, REGION_OUTPUT
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_m1_to_neural_graph_clones_graph_native_core_regions() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)
    m1 = policy.modules[0].m1_component

    graph = m1_to_neural_graph(m1)

    assert graph is not m1.graph
    assert len(graph.regions[REGION_INPUT].neuron_ids) == 4
    assert len(graph.regions[REGION_ENCODER].neuron_ids) == 5
    assert len(graph.regions[REGION_OUTPUT].neuron_ids) == 3
    graph.assert_invariants(output_size=3)
    assert graph.edges
    edge = next(iter(graph.edges.values()))
    assert edge.kind == EDGE_FORWARD

    edge.weight += 1.0
    assert not np.isclose(edge.weight, m1.graph.edges[edge.id].weight)

def test_m1_graph_adapter_activation_is_finite_and_copy_safe() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=17)
    graph = m1_to_neural_graph(policy.modules[0].m1_component)
    output = graph.activate(np.array([0.2, -0.1, 0.3, -0.4]), steps=5)

    assert output.shape == (3,)
    assert np.isfinite(output).all()

    clone = graph.copy()
    edge_id = next(iter(clone.edges))
    clone.edges[edge_id].weight += 10.0
    assert not np.isclose(clone.edges[edge_id].weight, graph.edges[edge_id].weight)


def test_m1_graph_adapter_handles_mt5_sized_policy() -> None:
    policy = SelfGrowingPolicy.seed(seed=3)
    policy.convert_m1_to_graph_backend()
    graph = m1_to_neural_graph(policy.modules[0].m1_component)

    output = graph.activate(np.zeros(policy.input_size), steps=5)

    assert output.shape == (policy.output_size,)
    assert np.isfinite(output).all()
    graph.assert_invariants(output_size=policy.output_size)
