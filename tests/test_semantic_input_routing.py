from __future__ import annotations

import numpy as np
import pytest

from self_growing_trader.chuyen_gia_policy import (
    ACTION_COUNT,
    INPUT_COUNT,
    OUTPUT_COUNT,
    ChuyenGiaPolicy,
)
from self_growing_trader.semantic_input_routing import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    EXTRA_MARKET_OFFSET,
    POSITION_OFFSET,
    TIMING_OFFSET,
    build_semantic_input_route,
)
from self_growing_trader.context_history import POLICY_CONTEXT_HISTORY_INPUT_SIZE


def _allowed(*indexes: int) -> dict[int, bool]:
    active = set(indexes)
    return {index: index in active for index in range(ACTION_COUNT)}


def _policy(seed: int = 701) -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=10,
        hidden_layers=2,
        seed=seed,
    )
    policy.semantic_input_routing_mode = "role_isolated_v1"
    # Full-RL mode makes legacy action/risk overlays unavailable. This is the
    # runtime contract used by the candidate, not a test-only shortcut.
    policy.direct_rnn_adamw_enabled = True
    policy.direct_rnn_full_rl_enabled = True
    policy.ensure_compatible()
    return policy


def _entry_context(policy: ChuyenGiaPolicy, *, account: dict[str, float] | None = None, pending: float = 0.0, positions: str = "") -> None:
    policy.set_decision_context(
        account_context=account or {
            "current_equity_drawdown": 0.0,
            "open_risk_ratio": 0.0,
            "position_count_norm": 0.0,
        },
        positions_pipe=positions,
        allowed_actions=_allowed(0, 1, 2, 3, 4, 5, 6),
        execution_context={"pending_orders": pending},
    )


def test_entry_route_exposes_only_market_timing_and_wyckoff() -> None:
    observation = np.linspace(-0.8, 0.8, INPUT_COUNT)
    route = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 1, 2, 3, 4, 5, 6),
        input_size=INPUT_COUNT,
    )

    assert route.role == "entry"
    assert np.all(route.observation[POSITION_OFFSET:EXTRA_MARKET_OFFSET] == 0.0)
    assert np.all(route.observation[EXECUTION_OFFSET:TIMING_OFFSET] == 0.0)
    assert np.array_equal(route.observation[:POSITION_OFFSET], observation[:POSITION_OFFSET])
    assert np.array_equal(route.observation[EXTRA_MARKET_OFFSET:EXECUTION_OFFSET], observation[EXTRA_MARKET_OFFSET:EXECUTION_OFFSET])
    assert np.array_equal(route.observation[TIMING_OFFSET:], observation[TIMING_OFFSET:])


def test_management_route_cannot_read_market_or_account() -> None:
    observation = np.linspace(-0.8, 0.8, INPUT_COUNT)
    route = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 7, 8, 9, 10, 11),
        positions_pipe="17:1:0.1:1:0:0:0:4",
        input_size=INPUT_COUNT,
    )

    assert route.role == "management"
    assert np.all(route.observation[:POSITION_OFFSET] == 0.0)
    assert np.all(route.observation[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET] == 0.0)
    assert np.all(route.observation[TIMING_OFFSET:] == 0.0)
    assert np.array_equal(route.observation[POSITION_OFFSET:ACCOUNT_OFFSET], observation[POSITION_OFFSET:ACCOUNT_OFFSET])
    assert np.array_equal(route.observation[EXECUTION_OFFSET:TIMING_OFFSET], observation[EXECUTION_OFFSET:TIMING_OFFSET])


def test_context_history_tail_is_available_to_both_authority_routes() -> None:
    """Two Python-local causal tokens are not mistaken for unknown wire data."""

    observation = np.linspace(-0.8, 0.8, POLICY_CONTEXT_HISTORY_INPUT_SIZE)
    entry = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 1, 2, 3, 4, 5, 6),
        input_size=POLICY_CONTEXT_HISTORY_INPUT_SIZE,
    )
    management = build_semantic_input_route(
        observation,
        mode="role_isolated_v1",
        allowed_actions=_allowed(0, 7, 8, 9, 10, 11),
        positions_pipe="17:1:0.1:1:0:0:0:4",
        input_size=POLICY_CONTEXT_HISTORY_INPUT_SIZE,
    )

    assert np.array_equal(entry.observation[INPUT_COUNT:], observation[INPUT_COUNT:])
    assert np.array_equal(management.observation[INPUT_COUNT:], observation[INPUT_COUNT:])
    assert entry.audit()["semantic_route_context_history_active"] == 2
    assert management.audit()["semantic_route_context_history_active"] == 2


def test_entry_logits_are_invariant_to_stateful_wire_features() -> None:
    market = np.linspace(-0.5, 0.5, INPUT_COUNT)
    changed_state = market.copy()
    changed_state[POSITION_OFFSET:ACCOUNT_OFFSET] = 0.95
    changed_state[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET] = -0.85
    changed_state[EXECUTION_OFFSET:TIMING_OFFSET] = 0.75

    left = _policy()
    right = _policy()
    _entry_context(left)
    _entry_context(right)
    left_trace = left.act(market)
    right_trace = right.act(changed_state)

    assert np.allclose(
        left_trace.action_details["direct_rnn_rl_logits"],
        right_trace.action_details["direct_rnn_rl_logits"],
    )
    forward = right_trace.module_action_traces[0].forward_trace
    assert forward is not None
    assert np.all(forward.masked_observation[POSITION_OFFSET:EXTRA_MARKET_OFFSET] == 0.0)
    assert np.all(forward.masked_observation[EXECUTION_OFFSET:TIMING_OFFSET] == 0.0)


def test_full_rl_encoder_gradient_cannot_write_entry_forbidden_groups() -> None:
    policy = _policy()
    _entry_context(policy)
    trace = policy.act(np.linspace(-0.5, 0.5, INPUT_COUNT))
    module = policy.modules[0]
    module_trace = trace.module_action_traces[0]
    forward = module_trace.forward_trace
    assert forward is not None
    action_direction = np.zeros(OUTPUT_COUNT, dtype=float)
    action_direction[1] = 1.0
    directions = policy._direct_rnn_full_rl_module_directions(
        module,
        forward,
        action_direction,
        value_direction=0.1,
        gate_direction=0.2,
        gate_logit=module_trace.gate_logit,
        temporal_forward_traces=module_trace.temporal_forward_traces,
    )
    assert np.all(directions["w_in"][:, POSITION_OFFSET:EXTRA_MARKET_OFFSET] == 0.0)
    assert np.all(directions["w_in"][:, EXECUTION_OFFSET:TIMING_OFFSET] == 0.0)
    assert np.all(directions["gate_w"][POSITION_OFFSET:EXTRA_MARKET_OFFSET] == 0.0)
    assert np.all(directions["gate_w"][EXECUTION_OFFSET:TIMING_OFFSET] == 0.0)


def test_management_hidden_state_cannot_leak_into_later_entry() -> None:
    observation = np.linspace(-0.5, 0.5, INPUT_COUNT)
    management_observation = observation.copy()
    management_observation[POSITION_OFFSET:ACCOUNT_OFFSET] = 0.9
    management_observation[EXECUTION_OFFSET:TIMING_OFFSET] = -0.9

    contaminated = _policy()
    clean = _policy()
    # This is a state-isolation assertion, not a dropout-phase assertion.
    # Frozen Full-RL now deliberately retains the same deterministic dropout
    # surface as online execution, so synchronize only the phase consumed by
    # the prior management forward before comparing the entry routes.
    contaminated._learning_enabled = False
    clean._learning_enabled = False
    contaminated.set_decision_context(
        account_context={"position_count_norm": 0.2},
        positions_pipe="17:1:0.1:1:0:0:0:4",
        allowed_actions=_allowed(0, 7, 8, 9, 10, 11),
        execution_context={"pending_orders": 0.0},
    )
    contaminated.act(management_observation)
    for contaminated_module, clean_module in zip(contaminated.modules, clean.modules, strict=True):
        clean_module.hidden_dropout_step = contaminated_module.hidden_dropout_step
    _entry_context(contaminated)
    _entry_context(clean)

    after_management = contaminated.act(observation)
    fresh_entry = clean.act(observation)
    assert np.allclose(
        after_management.action_details["direct_rnn_rl_logits"],
        fresh_entry.action_details["direct_rnn_rl_logits"],
    )


def test_management_specialist_keeps_management_support_outside_entry_actor() -> None:
    policy = _policy()
    module = policy.modules[0]
    module.w_out[:] = 0.0
    module.b_out[:] = 0.0
    module.b_out[11] = 2.0  # MANAGE_POSITION
    policy._context_expert_management_only = True
    policy._context_expert_owner_metadata = {"management_only": True}
    policy.set_decision_context(
        account_context={"position_count_norm": 0.2},
        positions_pipe="17:1:0.1:1:0:0:0:4",
        allowed_actions=_allowed(0, 7, 8, 9, 10, 11),
        execution_context={"pending_orders": 0.0},
    )

    trace = policy.act(np.linspace(-0.5, 0.5, INPUT_COUNT))
    forward = trace.module_action_traces[0].forward_trace
    assert forward is not None
    assert trace.action_details["semantic_route"] == "management"
    assert trace.action_details["direct_rnn_management_control_only"] is True
    assert trace.action_details["direct_rnn_rl_policy_eligible"] is False
    assert trace.action_details["direct_rnn_rl_action_mask"][11] is True
    assert all(trace.action_details["direct_rnn_rl_action_mask"][index] is False for index in range(1, 7))
    assert int(np.argmax(trace.selected_action[:ACTION_COUNT])) == 11
    assert np.all(forward.masked_observation[:POSITION_OFFSET] == 0.0)
    assert np.all(forward.masked_observation[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET] == 0.0)
    assert np.all(forward.masked_observation[TIMING_OFFSET:] == 0.0)


def test_account_hard_safety_veto_is_not_actor_credit() -> None:
    policy = _policy()
    module = policy.modules[0]
    module.w_out[:] = 0.0
    module.b_out[:] = 0.0
    module.b_out[1] = 2.0  # Direct-RNN itself requests MARKET_BUY.
    _entry_context(policy, account={
        "current_equity_drawdown": 0.20,
        "open_risk_ratio": 0.0,
        "position_count_norm": 0.0,
    })

    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    assert trace.action_details["semantic_risk_hard_veto"] is True
    assert trace.action_details["semantic_safety_veto_kind"] == "account_risk_hard"
    assert int(np.argmax(trace.selected_action[:ACTION_COUNT])) == 0
    assert trace.action_details["direct_rnn_rl_policy_eligible"] is False


def test_small_monetary_open_risk_does_not_trigger_hard_veto() -> None:

    policy = _policy()
    module = policy.modules[0]
    module.w_out[:] = 0.0
    module.b_out[:] = 0.0
    module.b_out[1] = 2.0
    _entry_context(policy, account={
        "current_equity_drawdown": 0.0,
        "open_risk_ratio": 0.20,
        "position_count_norm": 0.20,
    })

    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    assert trace.action_details["semantic_open_risk_measure"] == "money_at_initial_sl_ratio"
    assert trace.action_details["semantic_risk_hard_veto"] is False
    assert int(np.argmax(trace.selected_action[:ACTION_COUNT])) == 1


def test_account_risk_fraction_survives_context_wire_tanh_once() -> None:
    policy = _policy()
    module = policy.modules[0]
    module.w_out[:] = 0.0
    module.b_out[:] = 0.0
    module.b_out[1] = 2.0
    # 80% money-at-SL risk reduces budget but stays below the 90% hard veto. Leave
    # positions_pipe empty to isolate account sizing from duplicate blocking.
    _entry_context(policy, account={
        "current_equity_drawdown": 0.0,
        "open_risk_ratio": 0.80,
        "position_count_norm": 0.80,
    })

    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    actual_fraction = 1.0 / (1.0 + np.exp(-float(trace.selected_action[ACTION_COUNT])))
    assert trace.action_details["semantic_risk_hard_veto"] is False
    assert actual_fraction == pytest.approx(trace.action_details["semantic_risk_fraction"], abs=1e-7)


def test_duplicate_pending_entry_is_vetoed_without_changing_direction_weights() -> None:
    policy = _policy()
    module = policy.modules[0]
    module.w_out[:] = 0.0
    module.b_out[:] = 0.0
    module.b_out[1] = 2.0
    weights_before = module.w_in.copy()
    _entry_context(policy, pending=1.0)

    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    assert trace.action_details["semantic_duplicate_blocked_actions"] == [1, 2, 3, 4, 5, 6]
    assert trace.action_details["semantic_safety_veto_kind"] == "duplicate_entry"
    assert int(np.argmax(trace.selected_action[:ACTION_COUNT])) == 0
    assert trace.action_details["direct_rnn_rl_policy_eligible"] is False
    assert np.array_equal(module.w_in, weights_before)


def test_disabled_duplicate_mask_cannot_flip_sell_to_opposite_buy() -> None:
    policy = _policy()
    policy.semantic_duplicate_entry_block_enabled = False
    module = policy.modules[0]
    module.w_out[:] = 0.0
    module.b_out[:] = 0.0
    module.b_out[2] = 2.0  # Direct-RNN requests MARKET_SELL.
    _entry_context(
        policy,
        positions="17:-1:0.1:1:0:0:0:4",  # Existing SELL.
    )

    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    assert trace.action_details["semantic_duplicate_blocked_actions"] == []
    assert trace.action_details["semantic_safety_veto_kind"] == "none"
    assert int(np.argmax(trace.selected_action[:ACTION_COUNT])) == 2
    assert trace.action_details["direct_rnn_rl_policy_eligible"] is True
