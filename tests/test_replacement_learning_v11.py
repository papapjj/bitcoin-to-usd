from __future__ import annotations

from dataclasses import replace

import numpy as np

from self_growing_trader.replacement_learning import action_scope, scope_matches, stable_identifier
from self_growing_trader.self_growing_runner import (
    SelfGrowingPolicy,
    _m1v11_query_parameter_replacement_bias,
    _m1v11_record_parameter_replacement_cell,
    _m1v11_validate_parameter_replacements,
)


def _entry_trace(policy: SelfGrowingPolicy, action_name: str = "MARKET_BUY"):
    trace = policy.act(np.array([0.2, -0.1, 0.3, -0.4], dtype=float))
    action = np.zeros(policy.output_size, dtype=float)
    action[1 if action_name.endswith("BUY") else 2] = 1.0
    return replace(
        trace,
        selected_action=action,
        action_details={"action_name": action_name, "target_slot": 0},
        positions_pipe="",
    )


def test_replacement_scope_is_action_side_and_lifecycle_safe() -> None:
    buy = action_scope(1, action_details={"action_name": "MARKET_BUY"})
    sell = action_scope(2, action_details={"action_name": "MARKET_SELL"})
    manage = action_scope(11, action_details={"action_name": "MANAGE_POSITION"}, positions_pipe="1:1:0.01:1:0:2:0:1")

    assert not scope_matches(buy, sell)
    assert not scope_matches(buy, manage)
    assert scope_matches(buy, dict(buy))
    assert stable_identifier("replacement", buy, [0.1, 0.2]) == stable_identifier("replacement", buy, [0.1, 0.2])


def test_parameter_replacement_is_shadow_scoped_then_validated() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=121)
    policy.graph_trial_mode = True
    trace = _entry_trace(policy, "MARKET_BUY")
    target = np.array([-0.3, 0.25, 0.15, 0.0, 0.2, 0.1, 0.0, 0.0], dtype=float)
    summary = {"action_index": 1, "realized_r": -1.0, "causes": ["sl_loss"], "replacement_not_action_suppression": True}

    proposal_id = _m1v11_record_parameter_replacement_cell(policy, trace, target, summary)
    assert proposal_id

    buy_scope = action_scope(1, action_details={"action_name": "MARKET_BUY"})
    buy_params, buy_meta = _m1v11_query_parameter_replacement_bias(
        policy, trace.encoded_state, {}, policy.output_size, current_scope=buy_scope, return_metadata=True,
    )
    assert buy_meta["proposal_ids"] == [proposal_id]
    assert np.max(np.abs(buy_params)) > 0.0

    sell_scope = action_scope(2, action_details={"action_name": "MARKET_SELL"})
    sell_params, sell_meta = _m1v11_query_parameter_replacement_bias(
        policy, trace.encoded_state, {}, policy.output_size, current_scope=sell_scope, return_metadata=True,
    )
    assert sell_meta["proposal_ids"] == []
    assert np.allclose(sell_params, 0.0)

    executed = replace(trace, action_details={
        "action_name": "MARKET_BUY",
        "target_slot": 0,
        "parameter_replacement_proposal_ids": [proposal_id],
    })
    for _ in range(3):
        outcomes = _m1v11_validate_parameter_replacements(policy, executed, realized_r=1.0)
    assert outcomes[-1]["proposal_state"] == "validated"

    # A validated proposal may survive candidate promotion and be reused by its
    # matching action scope, but remains invisible to the opposite action.
    policy.graph_trial_mode = False
    buy_params, buy_meta = _m1v11_query_parameter_replacement_bias(
        policy, trace.encoded_state, {}, policy.output_size, current_scope=buy_scope, return_metadata=True,
    )
    assert buy_meta["mode"] == "validated"
    assert np.max(np.abs(buy_params)) > 0.0


def test_parameter_head_credit_is_candidate_only() -> None:
    baseline = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=122)
    trace = _entry_trace(baseline)
    before = baseline.action_parameter_head.w_out.copy()
    event = baseline._apply_parameter_replacement_credit(trace, -1.0, close_reason="SL")
    assert event["candidate_only"] is True
    assert event["parameter_head_updated"] is False
    assert np.allclose(baseline.action_parameter_head.w_out, before)

    candidate = baseline.copy()
    candidate.graph_trial_mode = True
    candidate_trace = _entry_trace(candidate)
    candidate._apply_parameter_replacement_credit(candidate_trace, -1.0, close_reason="SL")
    assert not np.allclose(candidate.action_parameter_head.w_out, before)


def test_parameter_replacement_credit_is_cleanly_skipped_when_disabled() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=124)
    trace = _entry_trace(policy)
    policy.parameter_replacement_learning = False
    policy.action_parameter_head = None

    event = policy._apply_parameter_replacement_credit(trace, -1.0, close_reason="SL")

    assert event == {"parameter_head_updated": False, "skipped_disabled": True}
    assert policy.growth_events[-1]["type"] == "m1_parameter_replacement_credit_skipped"
    assert policy.growth_events[-1]["reason"] == "parameter_replacement_learning_disabled"


def test_compaction_removes_only_promoted_trial_pruning_edges() -> None:
    baseline = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=123)
    graph = baseline.modules[0].m1_component.graph
    edge_id = next(iter(graph.edges))
    graph.edges[edge_id].enabled = False
    graph.edges[edge_id].age = 10
    assert baseline.compact_graph_structures()["reason"] == "baseline_protected"
    assert edge_id in graph.edges

    trial = baseline.copy()
    trial.graph_trial_mode = True
    trial.graph_trial_id = "test-trial"
    result = trial.compact_graph_structures()
    assert result["edges_removed"] >= 1
    assert edge_id not in trial.modules[0].m1_component.graph.edges
    trial.modules[0].m1_component.graph.assert_invariants(output_size=trial.output_size)
