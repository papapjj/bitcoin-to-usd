"""Direct graph-native M1 construction.

This module deliberately has no dependency on the retired pre-graph core
implementation.  Every seeded policy starts as a typed neural graph whose
regions can later specialize, grow, fade and consolidate independently.
"""
from __future__ import annotations

import numpy as np

from .neural_graph import (
    EDGE_CRITIC,
    EDGE_FORWARD,
    EDGE_GATING,
    EDGE_LATERAL,
    EDGE_MEMORY,
    EDGE_PREDICTIVE,
    EDGE_RECURRENT,
    EDGE_SELECTOR,
    REGION_CRITIC,
    REGION_ENCODER,
    REGION_HYPOTHESIS,
    REGION_INPUT,
    REGION_MEMORY,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    NeuralGraph,
    NeuralGraphBudget,
)


def seed_m1_graph(
    *,
    input_size: int,
    output_size: int,
    state_size: int,
    hypothesis_count: int,
    seed: int | None = None,
    module_id: str = "m1",
) -> NeuralGraph:
    """Create the graph M1 directly, without an adapter or predecessor runtime.

    `0` in ``NeuralGraphBudget`` means open-ended.  The starter graph is small
    enough to evaluate efficiently, while later growth is governed by evidence
    from contextual memory rather than a fixed topology ceiling.
    """
    rng = np.random.default_rng(seed)
    input_size = max(1, int(input_size))
    output_size = max(1, int(output_size))
    state_size = max(2, int(state_size))
    hypothesis_count = max(2, int(hypothesis_count))
    graph = NeuralGraph(budget=NeuralGraphBudget())

    input_ids = _nodes(graph, REGION_INPUT, input_size, module_id)
    encoder_ids = _nodes(graph, REGION_ENCODER, state_size, module_id, rng)
    memory_ids = _nodes(graph, REGION_MEMORY, state_size, module_id, rng)
    hypothesis_ids = _nodes(graph, REGION_HYPOTHESIS, hypothesis_count, module_id, rng)
    predictor_ids = _nodes(graph, REGION_PREDICTOR, state_size, module_id, rng)
    critic_ids = _nodes(graph, REGION_CRITIC, state_size, module_id, rng)
    selector_ids = _nodes(graph, REGION_SELECTOR, hypothesis_count, module_id, rng)
    output_ids = _nodes(graph, REGION_OUTPUT, output_size, module_id, rng)

    # Direct sensory encoding and a context-sensitive working-memory loop.
    _random_dense(graph, input_ids, encoder_ids, rng, EDGE_FORWARD, "graph_seed_encoder", scale=0.20)
    _random_dense(graph, input_ids, memory_ids, rng, EDGE_MEMORY, "graph_seed_memory_input", scale=0.12)
    _diagonal(graph, encoder_ids, memory_ids, EDGE_MEMORY, "graph_seed_memory_write", weight=0.55, gate=-1.2)
    _diagonal(graph, memory_ids, memory_ids, EDGE_RECURRENT, "graph_seed_memory_recur", weight=0.42, gate=-1.1)

    # Competing hypotheses integrate encoded state plus contextual memory.
    _random_dense(graph, encoder_ids, hypothesis_ids, rng, EDGE_FORWARD, "graph_seed_hyp_state", scale=0.24)
    _random_dense(graph, memory_ids, hypothesis_ids, rng, EDGE_MEMORY, "graph_seed_hyp_memory", scale=0.20)
    _small_lateral(graph, hypothesis_ids, rng, "graph_seed_hyp_lateral")
    _random_dense(graph, hypothesis_ids, output_ids, rng, EDGE_GATING, "graph_seed_decode", scale=0.16, gate=-0.55)

    # Prediction, value evaluation and selection remain graph-local paths.
    _random_dense(graph, encoder_ids, predictor_ids, rng, EDGE_PREDICTIVE, "graph_seed_predict_state", scale=0.18)
    _random_dense(graph, memory_ids, predictor_ids, rng, EDGE_MEMORY, "graph_seed_predict_memory", scale=0.16)
    _random_dense(graph, hypothesis_ids, predictor_ids, rng, EDGE_PREDICTIVE, "graph_seed_predict_hyp", scale=0.14, gate=-0.3)
    _diagonal(graph, predictor_ids, predictor_ids, EDGE_RECURRENT, "graph_seed_predict_recur", weight=0.35, gate=-1.2)
    _random_dense(graph, predictor_ids, critic_ids, rng, EDGE_CRITIC, "graph_seed_critic", scale=0.18)
    _random_dense(graph, critic_ids, selector_ids, rng, EDGE_SELECTOR, "graph_seed_selector_value", scale=0.20, gate=-0.6)
    _diagonal(graph, hypothesis_ids, selector_ids, EDGE_SELECTOR, "graph_seed_selector_hyp", weight=0.70, gate=0.55)
    _small_lateral(graph, selector_ids, rng, "graph_seed_selector_lateral")

    graph.assert_invariants(output_size=output_size)
    return graph


def _nodes(graph: NeuralGraph, region: str, count: int, module_id: str, rng: np.random.Generator | None = None) -> list[str]:
    result: list[str] = []
    for idx in range(count):
        bias = 0.0 if rng is None else float(rng.normal(0.0, 0.02))
        result.append(graph.add_neuron(region, neuron_id=f"{module_id}:{region}:{idx}", module_id=module_id, lifecycle="active", bias=bias))
    return result


def _random_dense(graph: NeuralGraph, src: list[str], dst: list[str], rng: np.random.Generator, kind: str, created_by: str, *, scale: float, gate: float = 0.0) -> None:
    if not src or not dst:
        return
    weights = rng.normal(0.0, scale / max(1.0, np.sqrt(len(src))), size=(len(dst), len(src)))
    for row, dst_id in enumerate(dst):
        for col, src_id in enumerate(src):
            graph.add_connection(src_id, dst_id, weight=float(weights[row, col]), gate=gate, kind=kind, created_by=created_by, need_reason="graph_native_seed", candidate_id="graph_seed")


def _diagonal(graph: NeuralGraph, src: list[str], dst: list[str], kind: str, created_by: str, *, weight: float, gate: float) -> None:
    for idx in range(min(len(src), len(dst))):
        graph.add_connection(src[idx], dst[idx], weight=float(weight), gate=float(gate), kind=kind, created_by=created_by, need_reason="graph_native_seed", candidate_id="graph_seed")


def _small_lateral(graph: NeuralGraph, nodes: list[str], rng: np.random.Generator, created_by: str) -> None:
    for dst_idx, dst_id in enumerate(nodes):
        for src_idx, src_id in enumerate(nodes):
            if src_idx == dst_idx:
                continue
            graph.add_connection(src_id, dst_id, weight=float(rng.normal(0.0, 0.035)), gate=-1.7, kind=EDGE_LATERAL, created_by=created_by, need_reason="graph_native_seed", candidate_id="graph_seed")
