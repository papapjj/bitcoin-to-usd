from __future__ import annotations

import numpy as np

from self_growing_trader.live_observatory import LiveEventReader, LiveObservabilityConfig
from self_growing_trader.reasoning_graph import ReasoningGraphSystem
from self_growing_trader.self_growing_runner import (
    INPUT_COUNT,
    SelfGrowingPolicy,
    _REASONING_V18_POLICY_ACT,
    _reasoning_v18_rerank,
)


def _context(policy: SelfGrowingPolicy, frame_id: str = "v18:1") -> None:
    policy.set_decision_context(
        account_context={"current_equity_drawdown": 0.02, "open_risk_ratio": 0.0},
        execution_context={"spread_norm": 0.1, "margin_available_ratio": 1.0},
        allowed_actions={index: True for index in range(12)},
        frame_id=frame_id,
        sequence_id=1,
        timestamp=1_700_000_000.0,
    )


def test_reasoning_v18_projection_fills_semantic_slots_and_marks_controller() -> None:
    graph = ReasoningGraphSystem.seed(input_size=32, output_size=16, rng=np.random.default_rng(7))
    # Old risk path would carry 12 values then zero-pad input:12..31.  V18
    # projects the complete semantic context across all 32 slots instead.
    graph.enrich_context("risk", np.arange(1.0, 13.0), target_size=12)
    record = graph.controller_trace("risk")
    raw = record["raw_inputs"]
    assert abs(float(raw["reasoning:input:29"])) > 1e-8
    assert float(raw["reasoning:controller:risk"]) == 1.0
    assert float(raw["reasoning:controller:action_head"]) == 0.0


def test_reasoning_v18_namespaces_topology_and_exports_activity(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=11, input_size=INPUT_COUNT, m1_backend="m1_graph")
    path = tmp_path / "neural_live.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(path), reset_on_process_start=True),
        run_id="reasoning-v18-topology",
    )
    _context(policy)
    trace = policy.act(np.linspace(-0.5, 0.5, policy.input_size))
    assert "m1:m1_thinking" in trace.graph_traces
    assert "reasoning_graph" in trace.graph_traces

    state = LiveEventReader(path).state()
    topology = state["topology"]
    node_ids = [str(item["id"]) for item in topology["nodes"]]
    edge_ids = [str(item["id"]) for item in topology["edges"]]
    assert len(node_ids) == len(set(node_ids))
    assert len(edge_ids) == len(set(edge_ids))
    assert any(item.startswith("m1:m1_thinking::") for item in edge_ids)
    assert any(item.startswith("reasoning_graph::") for item in edge_ids)

    activity = state["latest"]["decision"]["payload"]["network"]["activity"]
    reasoning = activity["graphs"]["reasoning_graph"]
    assert int(reasoning["active_node_count"]) > 0
    assert int(reasoning["active_edge_count"]) > 0
    controller_nodes = [
        item for item in activity["nodes"]
        if str(item.get("local_id", "")).startswith("reasoning:controller:")
    ]
    assert controller_nodes
    assert isinstance(controller_nodes[0].get("controller_inputs"), dict)


def test_reasoning_v18_lifecycle_credits_immutable_entry_trace() -> None:
    policy = SelfGrowingPolicy.seed(seed=13, input_size=INPUT_COUNT, m1_backend="m1_graph")
    _context(policy, "entry:1")
    entry = policy.act(np.linspace(-0.25, 0.75, policy.input_size))
    entry_evidence = entry.graph_traces["reasoning_graph"]["controllers"]
    assert "action_head" in entry_evidence
    policy._execution_trace_cache = {entry.trace_id: entry}

    # A later decision overwrites the runtime graph state. Settlement must still
    # credit the first decision's immutable controller evidence.
    _context(policy, "later:2")
    policy.act(np.linspace(0.9, -0.9, policy.input_size))
    before = {key: (float(node.utility), float(node.plasticity)) for key, node in policy.reasoning_graph.graph.neurons.items()}
    credited = policy.apply_execution_lifecycle_credit(entry.trace_id, -0.8, close_reason="SL", duration_bars=4)
    assert credited > 0
    changed = sum(
        abs(float(node.utility) - before[key][0]) > 1e-12
        or abs(float(node.plasticity) - before[key][1]) > 1e-12
        for key, node in policy.reasoning_graph.graph.neurons.items()
    )
    assert changed > 0
    event = next(item for item in reversed(policy.growth_events) if item.get("type") == "reasoning_final_lifecycle_credit")
    assert event["decision_trace_source"] == "immutable_entry_trace"
    assert int(event["updated_nodes"]) > 0
    assert int(event["updated_edges"]) > 0


def test_reasoning_v18_can_change_the_selected_m1_candidate_before_execution() -> None:
    policy = SelfGrowingPolicy.seed(seed=17, input_size=INPUT_COUNT, m1_backend="m1_graph")
    _context(policy, "rerank:1")
    policy.reasoning_graph.begin_decision()
    base = _REASONING_V18_POLICY_ACT(policy, np.linspace(-1.0, 1.0, policy.input_size))
    count = int(base.candidate_scores.size)
    assert count >= 2
    target = (int(base.selected_idx) + 1) % count
    original = policy.reasoning_graph.controller_output

    def forced_output(controller: str, *, size: int | None = None) -> np.ndarray:
        width = int(size or count)
        values = np.full(width, -10.0, dtype=float)
        values[target] = 10.0
        return values

    policy.reasoning_graph.controller_output = forced_output  # type: ignore[method-assign]
    try:
        reranked = _reasoning_v18_rerank(policy, base)
    finally:
        policy.reasoning_graph.controller_output = original  # type: ignore[method-assign]
    assert reranked.selected_idx == target
    assert bool(reranked.action_details["reasoning_changed_candidate"])
    assert reranked.action_details["reasoning_selection_stage"] == "pre_execution_candidate_rerank"


def test_reasoning_v18_enters_m1_score_path_before_selector() -> None:
    policy = SelfGrowingPolicy.seed(seed=23, input_size=INPUT_COUNT, m1_backend="m1_graph")
    _context(policy, "internal-selector:1")
    original = policy.reasoning_graph.controller_output

    def forced_output(controller: str, *, size: int | None = None) -> np.ndarray:
        width = int(size or 1)
        if controller == "utility":
            values = np.full(width, -8.0, dtype=float)
            values[-1] = 8.0
            return values
        return original(controller, size=size)

    policy.reasoning_graph.controller_output = forced_output  # type: ignore[method-assign]
    try:
        trace = policy.act(np.linspace(-0.8, 0.8, policy.input_size))
    finally:
        policy.reasoning_graph.controller_output = original  # type: ignore[method-assign]
    m1 = policy.modules[0].m1_component
    assert m1 is not None
    assert float(np.sum(np.abs(m1.last_reasoning_score_bias))) > 0.0
    selector = dict(trace.action_details.get("reasoning_selector_bias", {}) or {})
    assert selector["m1:m1_thinking"]["selection_stage"] == "inside_m1_before_legality_and_selector"


def test_reasoning_v18_migrates_old_graph_without_controller_identity_nodes() -> None:
    graph = ReasoningGraphSystem.seed(input_size=16, output_size=8, rng=np.random.default_rng(29))
    old_node_ids = [node_id for node_id in graph.graph.neurons if node_id.startswith("reasoning:controller:")]
    old_core_ids = [node_id for node_id in graph.graph.neurons if node_id.startswith("reasoning:") and node_id.endswith(":core")]
    # Simulate a saved pre-V18 reasoning artifact: its learned context/encoder
    # structure remains, but controller identity inputs/routes did not exist.
    for node_id in old_node_ids + old_core_ids:
        graph.graph.remove_neuron(node_id)
    assert "reasoning:controller:risk" not in graph.graph.neurons
    graph.ensure_compatible()
    assert "reasoning:controller:risk" in graph.graph.neurons
    assert "reasoning:risk:core" in graph.graph.neurons
    assert any(
        edge.src == "reasoning:controller:risk" and edge.dst == "reasoning:risk:core"
        for edge in graph.graph.edges.values()
    )


def test_reasoning_v18_compatibility_cache_invalidates_on_topology_change() -> None:
    graph = ReasoningGraphSystem.seed(input_size=16, output_size=8, rng=np.random.default_rng(31))
    graph.ensure_compatible()
    first_signature = getattr(graph, "_compatibility_signature", None)
    graph.ensure_compatible()
    assert getattr(graph, "_compatibility_signature", None) == first_signature

    graph.graph.remove_neuron("reasoning:controller:risk")
    assert "reasoning:controller:risk" not in graph.graph.neurons
    graph.ensure_compatible()

    assert "reasoning:controller:risk" in graph.graph.neurons
    assert getattr(graph, "_compatibility_signature", None) == first_signature
