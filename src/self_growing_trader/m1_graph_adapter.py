from __future__ import annotations

from typing import Any

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


def m1_to_neural_graph(m1: Any, *, module_id: str = "m1") -> NeuralGraph:
    """Convert a legacy M1, or defensively clone an already-graph M1.

    Callers that inspect/train a freshly seeded policy should not need to know
    whether the policy has already crossed the graph-default migration line.
    Returning a deep graph copy preserves candidate isolation in both cases.
    """
    existing_graph = getattr(m1, "graph", None)
    if isinstance(existing_graph, NeuralGraph):
        return existing_graph.copy()
    input_size = int(m1.state_encoder.w.shape[1])
    hidden_size = int(m1.state_size)
    hypothesis_count = int(m1.K)
    output_size = int(m1.hypothesis_bank.w_decode.shape[0])
    estimated_neurons = input_size + hidden_size * 4 + hypothesis_count * 2 + output_size
    estimated_edges = (
        input_size * hidden_size * 2
        + hidden_size * hypothesis_count * 2
        + hypothesis_count * output_size
        + hidden_size * hidden_size * 4
        + hypothesis_count * hypothesis_count
        + output_size
    )
    graph = NeuralGraph(
        budget=NeuralGraphBudget(
            max_neurons=max(512, estimated_neurons + 64),
            max_connections=max(4096, estimated_edges + 512),
            max_recurrent_edges=max(512, hidden_size * hidden_size * 2),
            max_lateral_edges=max(512, hypothesis_count * hypothesis_count * 2),
            max_cross_module_edges=max(128, hidden_size * output_size),
            max_m1_module_edges=max(64, hidden_size * output_size),
        )
    )
    input_ids = [
        graph.add_neuron(REGION_INPUT, neuron_id=f"{module_id}:input:{i}", module_id=module_id, lifecycle="active")
        for i in range(input_size)
    ]
    encoder_ids = [
        graph.add_neuron(
            REGION_ENCODER,
            neuron_id=f"{module_id}:encoder:{i}",
            module_id=module_id,
            lifecycle="active",
            bias=float(m1.state_encoder.b[i]),
        )
        for i in range(hidden_size)
    ]
    memory_ids = [
        graph.add_neuron(REGION_MEMORY, neuron_id=f"{module_id}:memory:{i}", module_id=module_id, lifecycle="active")
        for i in range(hidden_size)
    ]
    hypothesis_ids = [
        graph.add_neuron(
            REGION_HYPOTHESIS,
            neuron_id=f"{module_id}:hypothesis:{i}",
            module_id=module_id,
            lifecycle="active",
            bias=float(m1.hypothesis_bank.b_gen[i]),
        )
        for i in range(hypothesis_count)
    ]
    predictor_ids = [
        graph.add_neuron(
            REGION_PREDICTOR,
            neuron_id=f"{module_id}:predictor:{i}",
            module_id=module_id,
            lifecycle="active",
            bias=float(m1.predictor.b[i]),
        )
        for i in range(hidden_size)
    ]
    critic_ids = [
        graph.add_neuron(
            REGION_CRITIC,
            neuron_id=f"{module_id}:critic:{i}",
            module_id=module_id,
            lifecycle="active",
            bias=float(m1.critic.b[i]),
        )
        for i in range(hidden_size)
    ]
    selector_ids = [
        graph.add_neuron(REGION_SELECTOR, neuron_id=f"{module_id}:selector:{i}", module_id=module_id, lifecycle="active")
        for i in range(hypothesis_count)
    ]
    output_ids = [
        graph.add_neuron(
            REGION_OUTPUT,
            neuron_id=f"{module_id}:output:{i}",
            module_id=module_id,
            lifecycle="active",
            bias=float(m1.hypothesis_bank.b_decode[i]),
        )
        for i in range(output_size)
    ]

    _dense_connect(graph, input_ids, encoder_ids, m1.state_encoder.w, EDGE_FORWARD, "m1_state_encoder")
    _dense_connect(graph, input_ids, memory_ids, m1.working_memory.w_write, EDGE_MEMORY, "m1_working_memory_write")
    _dense_connect(graph, encoder_ids, memory_ids, np.eye(hidden_size), EDGE_MEMORY, "m1_memory_candidate", gate=-1.5, weight_scale=1.0)
    _dense_connect(graph, memory_ids, memory_ids, np.eye(hidden_size), EDGE_RECURRENT, "m1_memory_state", gate=-1.0, weight_scale=float(m1.working_memory.decay))

    w_gen = np.asarray(m1.hypothesis_bank.w_gen, dtype=float)
    _dense_connect(graph, encoder_ids, hypothesis_ids, w_gen[:, :hidden_size], EDGE_FORWARD, "m1_hypothesis_state")
    _dense_connect(graph, memory_ids, hypothesis_ids, w_gen[:, hidden_size:hidden_size * 2], EDGE_MEMORY, "m1_hypothesis_memory")
    _dense_connect(graph, hypothesis_ids, hypothesis_ids, m1.hypothesis_bank.lateral_w, EDGE_LATERAL, "m1_hypothesis_lateral", gate=-2.0)
    _dense_connect(graph, hypothesis_ids, output_ids, m1.hypothesis_bank.w_decode, EDGE_GATING, "m1_hypothesis_decode", gate=-1.0)

    _dense_connect(graph, encoder_ids, predictor_ids, m1.predictor.w_state, EDGE_PREDICTIVE, "m1_predictor_state")
    _dense_connect(graph, memory_ids, predictor_ids, m1.predictor.w_memory, EDGE_MEMORY, "m1_predictor_memory")
    # In legacy M1 the predictor receives an explicit one-hot hypothesis/action
    # input.  Preserve that causal path in graph form; without it every
    # hypothesis receives the same critic state and selection is arbitrary.
    _dense_connect(graph, hypothesis_ids, predictor_ids, m1.predictor.w_action, EDGE_PREDICTIVE, "m1_predictor_hypothesis", gate=-0.5)
    _dense_connect(graph, predictor_ids, predictor_ids, m1.predictor.w_proj, EDGE_RECURRENT, "m1_predictor_projection", gate=-1.5)

    _dense_connect(graph, predictor_ids, critic_ids, m1.critic.w_state, EDGE_CRITIC, "m1_critic_state")
    score_matrix = np.tile(np.asarray(m1.critic.w_score, dtype=float), (hypothesis_count, 1))
    _dense_connect(graph, critic_ids, selector_ids, score_matrix, EDGE_SELECTOR, "m1_selector_score", gate=-1.0)
    # A diagonal hypothesis-to-selector route preserves per-hypothesis evidence
    # instead of allowing only a shared critic scalar into all selector units.
    _dense_connect(graph, hypothesis_ids, selector_ids, np.eye(hypothesis_count), EDGE_SELECTOR, "m1_selector_hypothesis", gate=1.0)
    _dense_connect(graph, selector_ids, selector_ids, m1.selector.lateral_w, EDGE_LATERAL, "m1_selector_lateral", gate=-2.0)
    graph.assert_invariants(output_size=output_size)
    return graph


def _dense_connect(
    graph: NeuralGraph,
    src_ids: list[str],
    dst_ids: list[str],
    matrix: np.ndarray,
    kind: str,
    created_by: str,
    *,
    gate: float = 0.0,
    weight_scale: float = 1.0,
) -> None:
    values = np.asarray(matrix, dtype=float)
    rows = min(len(dst_ids), values.shape[0])
    cols = min(len(src_ids), values.shape[1])
    for row in range(rows):
        for col in range(cols):
            weight = float(values[row, col]) * weight_scale
            if abs(weight) < 1e-12:
                continue
            graph.add_connection(
                src_ids[col],
                dst_ids[row],
                weight=weight,
                gate=gate,
                kind=kind,
                created_by=created_by,
                need_reason="legacy_m1_migration",
                candidate_id="legacy_m1_graph",
            )
