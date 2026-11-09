from __future__ import annotations

import numpy as np

from self_growing_trader.causal_ledger import _execution_action_from_details
from self_growing_trader.mt5_runner import BridgeRequest, MT5StrategyTesterEvaluator, NO_NEURAL_COMMAND


def test_v26_hold_channel_is_excluded_from_continuous_command_choice() -> None:
    outputs = np.zeros(20, dtype=float)
    outputs[0] = 0.99  # old HOLD winner must be ignored
    outputs[1] = 0.11
    outputs[2] = 0.08
    action, score = MT5StrategyTesterEvaluator._choose_non_hold_command(
        outputs,
        {0, 1, 2},
        min_score=0.0,
    )
    assert action == 1
    assert score == 0.11


def test_v26_no_command_is_internal_sentinel_not_hold() -> None:
    outputs = np.zeros(20, dtype=float)
    outputs[0] = 0.9
    outputs[1:7] = -0.1
    action, _ = MT5StrategyTesterEvaluator._choose_non_hold_command(
        outputs,
        {0, 1, 2, 3, 4, 5, 6},
        min_score=0.0,
    )
    assert action == NO_NEURAL_COMMAND


def test_v26_wire_noop_is_never_attributed_as_hold() -> None:
    action, source = _execution_action_from_details({
        "emitted_action": -1,
        "bridge_action": -1,
        "wire_action": 0,
        "wire_noop": True,
        "no_neural_command": True,
    })
    assert action == -1
    assert source == "none"


def test_v26_no_command_execution_has_no_credit_even_if_positions_change() -> None:
    before = BridgeRequest(sequence=1, inputs=np.zeros(1), positions=0, day_trades=0)
    after = BridgeRequest(sequence=2, inputs=np.zeros(1), positions=1, day_trades=1)
    details = MT5StrategyTesterEvaluator._execution_details(before, after, NO_NEURAL_COMMAND)
    assert details["no_neural_command"] is True
    assert details["memory_eligible"] is False
    assert details["action_feedback_eligible"] is False


def test_v26_continuous_policy_act_does_not_create_provisional_hold_snapshot() -> None:
    from self_growing_trader.self_growing_runner import SelfGrowingPolicy

    policy = SelfGrowingPolicy.seed(input_size=4, output_size=20, hidden_size=5, seed=2601, m1_backend="graph")
    policy._continuous_session_active = True
    trace = policy.act(np.zeros(4, dtype=float))
    assert policy.causal_ledger.snapshot_for(trace.trace_id) is None
