from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy, INPUT_COUNT, OUTPUT_COUNT


def _entry_trace(policy: ChuyenGiaPolicy, trace_id: str):
    policy.set_decision_context(
        account_context={"equity_to_balance": 1.0},
        positions_pipe="",
        allowed_actions=[True] * 12,
        execution_context={"spread_points": 12.0},
        sequence_id=1,
        frame_id=f"canonical:{trace_id}",
    )
    base = policy.act(np.linspace(-0.4, 0.4, policy.input_size))
    outputs = np.zeros(OUTPUT_COUNT, dtype=float)
    outputs[1] = 1.0
    return replace(
        base,
        trace_id=trace_id,
        selected_action=outputs,
        action_details={**base.action_details, "emitted_action": 1, "semantic_action": 1},
        execution_details={"position_ticket": f"ticket:{trace_id}"},
    )


def test_canonical_settlement_is_receipt_idempotent_and_action_local() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, seed=901)
    trace = _entry_trace(policy, "canonical-entry")
    policy._cache_execution_trace(trace)
    lifecycle = SimpleNamespace(position_ticket="ticket:canonical-entry")

    credited = policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        -1.0,
        close_reason="SL",
        duration_bars=4.0,
        lifecycle=lifecycle,
        trace=trace,
        override_action=1,
    )

    assert credited > 0
    ack = policy._last_lifecycle_credit_ack
    assert ack["settled"] is True
    assert ack["action_index"] == 1
    assert ack["components"]["expectancy"] == 1
    records = list(policy.memory_supernet.contextual_expectancy.records.values())
    assert any(int(record.scope["action_index"]) == 1 and record.values[-1] < 0.0 for record in records)
    assert policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        -1.0,
        close_reason="SL",
        duration_bars=4.0,
        lifecycle=lifecycle,
        trace=trace,
        override_action=1,
    ) == 0
    assert policy._last_lifecycle_credit_ack["reason"] == "duplicate_receipt"


def test_canonical_settlement_does_not_fall_back_to_an_unrelated_action() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, seed=902)
    trace = _entry_trace(policy, "canonical-side")
    policy._cache_execution_trace(trace)

    assert policy.apply_execution_lifecycle_credit(
        trace.trace_id,
        -0.8,
        close_reason="SL",
        duration_bars=5.0,
        trace=trace,
        override_action=1,
    ) > 0
    scopes = [dict(record.scope) for record in policy.memory_supernet.contextual_expectancy.records.values()]
    assert scopes
    assert all(int(scope["action_index"]) == 1 for scope in scopes)
