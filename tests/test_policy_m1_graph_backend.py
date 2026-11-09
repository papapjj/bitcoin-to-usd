from __future__ import annotations

import pickle

import numpy as np

from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.self_growing_runner import SelfGrowingPolicy, load_policy


def test_policy_can_convert_m1_to_graph_backend_and_activate() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=7)

    policy.convert_m1_to_graph_backend()
    output = policy.activate(np.ones(4))

    assert policy.m1_backend == "m1_graph"
    assert isinstance(policy.modules[0].m1_component, M1GraphModule)
    assert output.shape == (3,)
    assert np.isfinite(output).all()


def test_policy_graph_backend_act_returns_valid_trace() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=8)
    policy.convert_m1_to_graph_backend()

    trace = policy.act(np.array([0.1, -0.2, 0.3, -0.4]))

    assert trace.selected_action.shape == (3,)
    assert trace.candidate_actions.shape[1] == 3
    assert np.isfinite(trace.selected_action).all()
    assert trace.confidence >= 0.0


def test_policy_graph_backend_copy_and_load_compatibility(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=9)
    policy.convert_m1_to_graph_backend()
    copied = policy.copy()

    assert copied.m1_backend == "m1_graph"
    assert isinstance(copied.modules[0].m1_component, M1GraphModule)

    path = tmp_path / "graph_policy.pkl"
    with path.open("wb") as handle:
        pickle.dump(policy, handle)

    loaded = load_policy(path)

    assert loaded.m1_backend == "m1_graph"
    assert isinstance(loaded.modules[0].m1_component, M1GraphModule)


def test_graph_fill_caches_trace_but_defers_td_credit_to_final_lifecycle() -> None:
    from dataclasses import replace

    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=908, m1_backend="graph")
    trace = policy.act(np.array([0.1, -0.2, 0.3, -0.4]))
    filled = replace(
        trace,
        execution_details={
            "bridge_observed": True,
            "execution_confirmed": True,
            "outcome_eligible": True,
            "memory_eligible": True,
            "action_feedback_eligible": False,
            "filled": True,
            "closed": False,
        },
    )
    before = {edge_id: edge.utility for edge_id, edge in policy.modules[0].m1_component.graph.edges.items()}

    signal = policy.observe_transition(filled, np.zeros(4), reward=0.0, cost=0.0, done=False)

    assert signal["memory_eligible"] == 1.0
    assert signal["execution_confirmed"] == 0.0
    assert filled.trace_id in policy._execution_trace_cache
    after = {edge_id: edge.utility for edge_id, edge in policy.modules[0].m1_component.graph.edges.items()}
    assert after == before
