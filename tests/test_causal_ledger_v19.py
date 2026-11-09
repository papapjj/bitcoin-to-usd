from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from self_growing_trader.causal_ledger import ExitMechanism, classify_exit_mechanism
from self_growing_trader.live_observatory import LiveEventReader, LiveObservabilityConfig
from self_growing_trader.mt5_runner import parse_mt5_execution_ledger
from self_growing_trader.self_growing_runner import INPUT_COUNT, SelfGrowingPolicy


def _context(policy: SelfGrowingPolicy, frame_id: str = "causal:1") -> None:
    policy.set_decision_context(
        account_context={"current_equity_drawdown": 0.02, "open_risk_ratio": 0.01},
        execution_context={"spread_norm": 0.08, "margin_available_ratio": 1.0},
        allowed_actions={index: True for index in range(12)},
        frame_id=frame_id,
        sequence_id=1,
        timestamp=1_700_000_000.0,
    )


def test_exit_taxonomy_never_infers_tp_from_positive_pnl() -> None:
    assert classify_exit_mechanism("", realized_r=2.0) is ExitMechanism.UNKNOWN
    assert classify_exit_mechanism("SL", realized_r=-1.0) is ExitMechanism.HARD_STOP_LOSS
    assert classify_exit_mechanism("DEAL_REASON_TP", realized_r=1.0) is ExitMechanism.TAKE_PROFIT
    assert classify_exit_mechanism("EXPERT_CLOSE", realized_r=-0.3) is ExitMechanism.POLICY_CLOSE


def test_v19_ledger_parser_preserves_trajectory_and_exit_evidence(tmp_path) -> None:
    ledger = tmp_path / "execution.csv"
    ledger.write_text(
        "schema_version\tsource\tack_sequence\ttrace_id\tposition_ticket\tfilled\tclosed\tposition_final_close\tlogged_at\tinitial_risk\tprofit\tcommission\tswap\tclose_reason\tage_bars\tentry_price\texit_price\tinitial_sl_price\tfinal_sl_price\tinitial_tp_price\tfinal_tp_price\tmfe_r\tmae_r\tnear_tp_ratio\tmarket_price\tunrealized_r\tspread_atr_ratio\tvolatility_shock\tregime\tpost_exit_mfe_r\treturned_to_entry\toriginal_tp_hit_after_exit\n"
        "19\tpolicy\t1\ttrace-x\t77\t1\t0\t0\t2026.06.27 10:00:00\t100\t0\t0\t0\t\t0\t100\t0\t90\t90\t120\t120\t0\t0\t0\t100\t0\t0.08\t0.10\ttrend\t\t\t\n"
        "19\tpolicy\t1\ttrace-x\t77\t0\t0\t0\t2026.06.27 10:03:00\t100\t0\t0\t0\t\t3\t100\t0\t90\t90\t120\t120\t0.72\t-0.20\t0.60\t112\t0.72\t0.07\t0.20\ttrend\t\t\t\n"
        "19\tpolicy\t1\ttrace-x\t77\t0\t1\t1\t2026.06.27 10:04:00\t100\t-100\t-2\t0\tSL\t4\t100\t90\t90\t90\t120\t120\t0.72\t-1.00\t0.60\t90\t-1.00\t0.11\t0.30\ttrend\t1.40\t1\t1\n",
        encoding="utf-8",
    )
    parsed = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)
    assert parsed.lifecycle_count == 1
    item = parsed.lifecycles[0]
    assert item.exit_mechanism == ExitMechanism.HARD_STOP_LOSS.value
    assert item.entry_price == 100.0
    assert item.exit_price == 90.0
    assert item.post_exit_mfe_r == 1.4
    assert item.returned_to_entry is True
    assert item.original_tp_hit_after_exit is True
    assert len(item.trajectory) >= 3


def test_causal_lifecycle_uses_entry_snapshot_and_marks_hypotheses(tmp_path) -> None:
    policy = SelfGrowingPolicy.seed(seed=311, input_size=INPUT_COUNT)
    live_path = tmp_path / "neural_live.jsonl"
    policy.enable_live_observatory(
        LiveObservabilityConfig(enabled=True, event_path=str(live_path), reset_on_process_start=True),
        run_id="causal-v19",
    )
    _context(policy)
    trace = policy.act(np.linspace(-0.5, 0.5, policy.input_size))
    policy._execution_trace_cache = {trace.trace_id: trace}
    lifecycle = SimpleNamespace(
        trace_id=trace.trace_id,
        position_ticket="pos-1",
        realized_r=-1.0,
        close_reason="SL",
        duration_bars=2.0,
        net_pnl=-100.0,
        commission=-1.0,
        swap=0.0,
        entry_time=None,
        exit_time=None,
        partial_close_count=0,
        trajectory=(
            {"age_bars": 0, "mfe_r": 0.0, "mae_r": 0.0, "volatility_shock": 0.1},
            {"age_bars": 2, "mfe_r": 0.10, "mae_r": -1.0, "volatility_shock": 0.1},
        ),
        post_exit_mfe_r=1.5,
        returned_to_entry=True,
        original_tp_hit_after_exit=True,
        metadata={},
    )
    credited = policy.apply_execution_lifecycle_credit(
        trace.trace_id, -1.0, close_reason="SL", duration_bars=2.0, lifecycle=lifecycle,
    )
    assert credited > 0
    report = policy.causal_lifecycle_report(trace.trace_id)
    assert report["coverage"]["entry_snapshot"] is True
    assert report["outcome"]["exit_mechanism"] == ExitMechanism.HARD_STOP_LOSS.value
    names = {item["cause_id"] for item in report["hypotheses"]}
    assert "stop_loss_too_tight" in names
    assert "entry_too_early" in names
    assert report["ablation"]["available"] is True

    state = LiveEventReader(live_path).state()
    lifecycle_event = state["latest"]["lifecycle"]["payload"]
    assert lifecycle_event["causal"]["outcome"]["exit_mechanism"] == ExitMechanism.HARD_STOP_LOSS.value
    assert lifecycle_event["causal"]["coverage"]["entry_snapshot"] is True


def test_replay_result_requires_verified_execution_and_can_validate_stop_width(tmp_path):
    from self_growing_trader.causal_ledger import CausalLedger
    from self_growing_trader.causal_replay import (
        CounterfactualReplayJournal,
        CounterfactualReplayResult,
        apply_replay_results,
        replay_requests,
    )
    policy = SelfGrowingPolicy.seed(seed=99, input_size=INPUT_COUNT)
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    ledger = CausalLedger()
    ledger.register_decision(policy, trace)
    lifecycle = SimpleNamespace(
        trace_id=trace.trace_id,
        position_ticket="991",
        close_reason="SL",
        realized_r=-1.0,
        net_pnl=-100.0,
        duration_bars=2,
        returned_to_entry=True,
        original_tp_hit_after_exit=True,
        post_exit_mfe_r=1.4,
        trajectory=(),
        metadata={},
    )
    report = ledger.settle(trace_id=trace.trace_id, lifecycle=lifecycle, realized_r=-1.0, close_reason="SL", duration_bars=2)
    requests = replay_requests(report)
    tight = next(item for item in requests if item.cause_id == "stop_loss_too_tight" and item.plan_name == "sl_1_25x")
    unverified = apply_replay_results(report, [CounterfactualReplayResult(
        request_id=tight.request_id, trace_id=trace.trace_id, plan_name="sl_1_25x", result_r=1.0,
        verified=False, execution_model="", data_provenance="",
    )])
    assert not unverified.coverage["counterfactual_replay_validated"]
    verified = apply_replay_results(report, [CounterfactualReplayResult(
        request_id=tight.request_id, trace_id=trace.trace_id, plan_name="sl_1_25x", result_r=1.0,
        verified=True, execution_model="mt5_strategy_tester", data_provenance="BTCUSD/M1 deterministic feed",
    )])
    assert verified.coverage["counterfactual_replay_validated"]
    assert any(item.cause_id == "stop_loss_too_tight" and item.status == "replay_validated" for item in verified.hypotheses)
    journal = CounterfactualReplayJournal(tmp_path / "requests.jsonl", tmp_path / "results.jsonl")
    journal.enqueue(report)
    assert (tmp_path / "requests.jsonl").exists()


def test_policy_consumes_later_verified_replay_without_reopening_lifecycle(tmp_path):
    import json
    policy = SelfGrowingPolicy.seed(seed=222, input_size=INPUT_COUNT)
    policy.causal_replay_requests_path = str(tmp_path / "requests.jsonl")
    policy.causal_replay_results_path = str(tmp_path / "results.jsonl")
    policy.causal_replay_journal = None
    policy.ensure_compatible()
    trace = policy.act(np.zeros(INPUT_COUNT, dtype=float))
    policy._execution_trace_cache = {trace.trace_id: trace}
    lifecycle = SimpleNamespace(
        trace_id=trace.trace_id,
        position_ticket="P-222",
        close_reason="SL",
        realized_r=-1.0,
        net_pnl=-100.0,
        duration_bars=2,
        returned_to_entry=True,
        original_tp_hit_after_exit=True,
        post_exit_mfe_r=1.5,
        trajectory=(),
        metadata={},
    )
    assert policy.apply_execution_lifecycle_credit(trace.trace_id, -1.0, close_reason="SL", duration_bars=2, lifecycle=lifecycle) > 0
    requests = [json.loads(line) for line in (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    request = next(item for item in requests if item["cause_id"] == "stop_loss_too_tight" and item["plan_name"] == "sl_1_25x")
    (tmp_path / "results.jsonl").write_text(json.dumps({
        "schema_version": 19,
        "request_id": request["request_id"],
        "trace_id": trace.trace_id,
        "plan_name": "sl_1_25x",
        "result_r": 1.1,
        "verified": True,
        "execution_model": "mt5_strategy_tester",
        "data_provenance": "deterministic same-feed",
    }) + "\n", encoding="utf-8")
    outcome = policy.process_causal_replay_results()
    assert outcome["validated"] >= 1
    report = policy.causal_lifecycle_report(trace.trace_id)
    assert report["coverage"]["counterfactual_replay_validated"]
