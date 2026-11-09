from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from self_growing_trader import cli
from self_growing_trader.continuous_session import (
    ContinuousModuleRegistry,
    ContinuousRuntimeConfig,
    ContinuousSessionRuntime,
    ExecutionLedgerTailer,
)


def _header() -> str:
    return (
        "schema_version,event_seq,event_time,event_type,trace_id,position_ticket,"
        "filled,closed,position_final_close,profit,commission,swap,initial_risk,"
        "close_reason,age_bars,market_price,unrealized_r,post_exit_mfe_r,"
        "post_exit_mae_r,returned_to_entry,original_tp_hit_after_exit\n"
    )


def _append(path: Path, *rows: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.rstrip("\n") + "\n")


def test_execution_ledger_tailer_requires_final_close_and_keeps_post_exit_evidence(tmp_path: Path) -> None:
    ledger = tmp_path / "self_growing_execution.csv"
    ledger.write_text(_header(), encoding="utf-8")
    _append(
        ledger,
        "23,10,2026.06.28 10:00:00,FILL,trace-1,9001,1,0,0,0,0,0,100,,0,100.0,0,,,,",
    )
    tailer = ExecutionLedgerTailer(initial_equity=10_000.0)
    assert tailer.poll(ledger) == []

    _append(
        ledger,
        "23,13,2026.06.28 10:03:00,FINAL_CLOSE,trace-1,9001,0,1,1,-100,-2,0,100,SL,3,99.0,-1.02,,,,",
    )
    lifecycles = tailer.poll(ledger)
    assert len(lifecycles) == 1
    lifecycle = lifecycles[0]
    assert lifecycle.trace_id == "trace-1"
    assert lifecycle.position_ticket == "9001"
    assert lifecycle.close_reason == "SL"
    assert lifecycle.realized_r < -1.0

    # A later POST_EXIT row must update the same lifecycle object; it must not
    # emit/credit a duplicate final lifecycle.
    _append(
        ledger,
        "23,18,2026.06.28 10:08:00,POST_EXIT_OBSERVATION,trace-1,9001,0,0,0,0,0,0,100,,8,102.5,0,1.40,-0.10,1,1",
    )
    assert tailer.poll(ledger) == []
    assert lifecycle.post_exit_mfe_r == 1.40
    assert lifecycle.post_exit_mae_r == -0.10
    assert lifecycle.returned_to_entry is True
    assert lifecycle.original_tp_hit_after_exit is True


def test_continuous_runtime_waits_for_post_exit_safe_point_before_credit(tmp_path: Path) -> None:
    ledger = tmp_path / "execution.csv"
    ledger.write_text(_header(), encoding="utf-8")
    _append(
        ledger,
        "23,1,2026.06.28 10:00:00,FILL,trace-2,9002,1,0,0,0,0,0,100,,0,100.0,0,,,,",
        "23,3,2026.06.28 10:03:00,FINAL_CLOSE,trace-2,9002,0,1,1,-100,0,0,100,SL,3,99.0,-1.0,,,,",
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.risk = SimpleNamespace(initial_equity=10_000.0)
            self.causal_ledger = SimpleNamespace(reports={})
            self.calls: list[tuple[str, float]] = []

        def apply_execution_lifecycle_credit(self, trace_id, realized_r, **kwargs):
            self.calls.append((str(trace_id), float(realized_r)))
            return 1

        def process_causal_replay_results(self):
            return {}

    policy = FakePolicy()
    config = ContinuousRuntimeConfig(
        enabled=True,
        session_root=str(tmp_path / "continuous"),
        post_exit_bars=2,
        checkpoint_every_closed_positions=5,
        checkpoint_every_minutes=60,
    )
    runtime = ContinuousSessionRuntime(config, session_id="safe-point")

    runtime.before_decision(policy, SimpleNamespace(sequence=3), execution_path=ledger)
    assert policy.calls == []
    runtime.before_decision(policy, SimpleNamespace(sequence=4), execution_path=ledger)
    assert policy.calls == []
    runtime.before_decision(policy, SimpleNamespace(sequence=5), execution_path=ledger)
    assert policy.calls == [("trace-2", -1.0)]
    assert runtime.closed_positions == 1


def test_local_inhibition_is_context_scoped_and_changes_only_action_logits() -> None:
    config = ContinuousRuntimeConfig(enabled=True, max_total_action_bias=0.20)
    registry = ContinuousModuleRegistry(config)
    context = np.linspace(-0.5, 0.5, 16)
    policy = SimpleNamespace(
        continuous_adaptive_state={
            "local_inhibitions": [
                {
                    "context_key": "ctx:test",
                    "context_vector": context.tolist(),
                    "action": 1,
                    "replacement": 2,
                    "strength": 0.18,
                    "confidence": 1.0,
                    "created_sequence": 1,
                    "last_sequence": 1,
                    "supports": 2,
                }
            ]
        }
    )
    trace = SimpleNamespace(
        canonical_context_key=context,
        selected_action=np.zeros(20, dtype=float),
        action_details={},
    )

    updated = registry.overlay_trace(policy, trace, sequence=2)
    assert updated.selected_action[1] < 0.0
    assert updated.selected_action[2] > 0.0
    assert np.allclose(updated.selected_action[12:], 0.0)
    assert "continuous_overlay" in updated.action_details


def test_continuous_cli_has_no_round_or_trial_arguments(monkeypatch, capsys) -> None:
    captured = {}

    def fake_continuous(config, **kwargs):
        captured["continuous"] = config.continuous
        captured.update(kwargs)
        return {"mode": "continuous_event_driven_v23", "policy": kwargs["output_path"]}

    monkeypatch.setattr(cli, "continuous_train_mt5", fake_continuous)
    cli.main(
        [
            "continuous-train",
            "--output", "outputs/continuous_test.pkl",
            "--seed-policy", "outputs/winner.pkl",
            "--backtest-days", "45",
            "--post-exit-bars", "12",
            "--checkpoint-every-closed-positions", "7",
            "--continuous-root", "outputs/continuous_test",
            "--disable-local-inhibition",
        ]
    )

    assert captured["output_path"] == "outputs/continuous_test.pkl"
    assert captured["backtest_days"] == 45
    assert captured["continuous"].enabled is True
    assert captured["continuous"].post_exit_bars == 12
    assert captured["continuous"].checkpoint_every_closed_positions == 7
    assert captured["continuous"].session_root == "outputs/continuous_test"
    assert captured["continuous"].enable_local_inhibition is False
    assert "continuous_event_driven_v23" in capsys.readouterr().out
