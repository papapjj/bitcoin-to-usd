from __future__ import annotations

from dataclasses import replace

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, ModuleActionTrace, StepTrace
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT


def test_negative_close_all_credit_reduces_close_and_strengthens_manage_on_same_hidden_path() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        seed=20260711,
    )
    module = policy.modules[0]
    hidden = np.linspace(0.2, 0.9, module.hidden_size)
    module.w_out[10] = 0.5
    module.w_out[11] = 0.1
    selected_action = np.zeros(OUTPUT_COUNT, dtype=float)
    selected_action[10] = 1.0
    trace = StepTrace(
        trace_id="exit-close-all",
        timestamp=0.0,
        raw_obs=np.zeros(INPUT_COUNT),
        encoded_state=np.zeros(module.hidden_size),
        memory_before=np.zeros(module.hidden_size),
        memory_after=np.zeros(module.hidden_size),
        selected_idx=0,
        selected_action=selected_action,
        candidate_actions=np.zeros((1, OUTPUT_COUNT)),
        candidate_logits=np.zeros(1),
        candidate_scores=np.zeros(1),
        selected_action_log_prob=0.0,
        predicted_next_state=np.zeros(module.hidden_size),
        critic_value=0.0,
        confidence=0.5,
        novelty=0.0,
        conflict=0.0,
        risk=0.0,
        gate_values=(1.0,),
        prediction_error=0.0,
        td_delta=0.0,
        action_details={"emitted_action": 10, "action_name": "CLOSE_ALL"},
        module_action_traces=(
            ModuleActionTrace(
                module_name=module.name,
                gate=1.0,
                hidden=hidden,
                raw_output=selected_action.copy(),
            ),
        ),
    )
    before_close = float(module.w_out[10] @ hidden)
    before_manage = float(module.w_out[11] @ hidden)

    credited = policy._apply_neuron_action_credit(trace, -1.0, strength=1.0, provenance="test")

    after_close = float(module.w_out[10] @ hidden)
    after_manage = float(module.w_out[11] @ hidden)
    assert credited == module.hidden_size
    assert after_close < before_close
    assert after_manage > before_manage
    event = policy.growth_events[-1]
    assert event["action"] == 10
    assert event["replacement_action"] == 11
    assert event["negative_credit_multiplier"] == 8.0
    assert event["context_local_negative_margin"] is True


def test_pending_entry_loss_replacement_never_strengthens_hold() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        seed=20260714,
    )
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))

    assert policy._cause_conditioned_replacement_action(trace, 3, 12) == 5
    assert policy._cause_conditioned_replacement_action(trace, 5, 12) == 3
    assert policy._cause_conditioned_replacement_action(trace, 4, 12) == 6
    assert policy._cause_conditioned_replacement_action(trace, 6, 12) == 4
    assert policy._cause_conditioned_replacement_action(trace, 5, 12, runner_up=4) == 4


def test_role_scoped_pending_loss_uses_reachable_same_style_direction() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        seed=20260714,
    )
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    limit_trace = replace(trace, action_details={"context_allowed_actions": [0, 3, 4]})
    stop_trace = replace(trace, action_details={"context_allowed_actions": [0, 5, 6]})

    assert policy._cause_conditioned_replacement_action(limit_trace, 3, 12, runner_up=5) == 4
    assert policy._cause_conditioned_replacement_action(stop_trace, 6, 12, runner_up=3) == 5


def test_role_scoped_add_on_loss_does_not_credit_unreachable_manager_action() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        seed=20260714,
    )
    trace = replace(
        policy.act(np.zeros(INPUT_COUNT, dtype=float)),
        account_context={"position_count_norm": 0.1, "open_risk_ratio": 0.1},
        positions_pipe="123:1:0.1:100:99:102:0.0:1",
    )
    scoped = replace(trace, action_details={"context_allowed_actions": [0, 5, 6]})

    assert policy._cause_conditioned_replacement_action(scoped, 5, 12) == 6


def test_losing_add_on_entry_learns_manage_without_runtime_entry_mask() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        seed=20260714,
    )
    trace = replace(
        policy.act(np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=float)),
        account_context={"position_count_norm": 0.1, "open_risk_ratio": 0.1},
        positions_pipe="123:1:0.1:100:99:102:0.0:1",
    )

    for entry_action in (1, 2, 3, 4, 5, 6):
        assert policy._cause_conditioned_replacement_action(trace, entry_action, 12) == 11

    credited = policy._apply_neuron_action_credit(
        replace(trace, action_details={"bridge_action": 5, "action_name": "BUY_STOP"}),
        -1.0,
        provenance="test_add_on_loss",
    )

    assert credited > 0
    assert policy.growth_events[-1]["replacement_action"] == 11
    assert policy.growth_events[-1]["replacement_gain"] == 3.0


def test_candidate_negative_credit_controls_do_not_change_v13_defaults() -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        seed=20260715,
    )
    policy._negative_context_credit_multiplier = 1.0
    policy._negative_entry_replacement_gain = 1.0
    trace = replace(
        policy.act(np.linspace(-1.0, 1.0, INPUT_COUNT, dtype=float)),
        account_context={"position_count_norm": 0.1, "open_risk_ratio": 0.1},
        positions_pipe="123:1:0.1:100:99:102:0.0:1",
        action_details={"bridge_action": 5, "action_name": "BUY_STOP"},
    )

    assert policy._apply_neuron_action_credit(trace, -1.0, provenance="test_candidate_controls") > 0
    event = policy.growth_events[-1]
    assert event["negative_credit_multiplier"] == 1.0
    assert event["replacement_gain"] == 1.0
