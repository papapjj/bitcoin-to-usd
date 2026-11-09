from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from scripts import run_v118_isolated_oos as runner


def _metrics(path: str) -> dict[str, object]:
    return {
        "execution_ledger_path": path,
        "fitness_reconciled": True,
        "fitness_terminal_open_positions": 0,
        "feature_audit_valid": True,
        "real_tick_evidence_verified": True,
    }


def _config() -> SimpleNamespace:
    return SimpleNamespace(risk=SimpleNamespace(initial_equity=10000.0))


def test_pooled_policy_evidence_counts_unique_owner_lifecycles(monkeypatch) -> None:
    lifecycles = tuple(
        SimpleNamespace(
            final_close_confirmed=True,
            metadata={"position_owner_trace_id": f"trace-{index}"},
            position_ticket=f"ticket-{index}",
            trace_id="",
            realized_r=0.50,
            exit_time=1735689600.0 + float(index % 3) * 86400.0,
        )
        for index in range(30)
    )
    monkeypatch.setattr(
        runner,
        "parse_mt5_execution_ledger",
        lambda *_args, **_kwargs: SimpleNamespace(lifecycles=lifecycles),
    )
    rows = [{"window": 1, "candidate": {"metrics": _metrics("one.csv")}}]

    result = runner._pooled_policy_evidence(
        rows,
        arm="candidate",
        config=_config(),
        min_samples=30,
        min_lcb=0.0,
    )

    assert result["eligible"] is True
    assert result["policy_lifecycle_samples"] == 30
    assert result["unique_position_tickets"] == 30
    assert result["duplicate_position_tickets"] == []
    assert result["day_bootstrap"]["block_count"] == 3


def test_pooled_policy_evidence_rejects_duplicate_position_ticket(monkeypatch) -> None:
    lifecycles = tuple(
        SimpleNamespace(
            final_close_confirmed=True,
            metadata={"position_owner_trace_id": f"trace-{index}"},
            position_ticket=f"ticket-{index}",
            trace_id="",
            realized_r=0.50,
        )
        for index in range(30)
    )
    monkeypatch.setattr(
        runner,
        "parse_mt5_execution_ledger",
        lambda *_args, **_kwargs: SimpleNamespace(lifecycles=lifecycles),
    )
    rows = [
        {"window": 1, "candidate": {"metrics": _metrics("one.csv")}},
        {"window": 1, "candidate": {"metrics": _metrics("two.csv")}},
    ]

    result = runner._pooled_policy_evidence(
        rows,
        arm="candidate",
        config=_config(),
        min_samples=30,
        min_lcb=0.0,
    )

    assert result["eligible"] is False
    assert len(result["duplicate_position_tickets"]) == 30
    assert any("duplicate_position_ticket" in reason for reason in result["reasons"])


def test_pooled_policy_evidence_scopes_ticket_identity_to_window(monkeypatch) -> None:
    lifecycles = tuple(
        SimpleNamespace(
            final_close_confirmed=True,
            metadata={"position_owner_trace_id": f"trace-{index}"},
            position_ticket=f"ticket-{index}",
            trace_id="",
            realized_r=0.50,
        )
        for index in range(30)
    )
    monkeypatch.setattr(
        runner,
        "parse_mt5_execution_ledger",
        lambda *_args, **_kwargs: SimpleNamespace(lifecycles=lifecycles),
    )
    rows = [
        {"window": 1, "candidate": {"metrics": _metrics("one.csv")}},
        {"window": 2, "candidate": {"metrics": _metrics("two.csv")}},
    ]

    result = runner._pooled_policy_evidence(
        rows,
        arm="candidate",
        config=_config(),
        min_samples=30,
        min_lcb=None,
    )

    assert result["duplicate_position_tickets"] == []
    assert result["unique_position_tickets"] == 60


def test_pooled_paired_gate_rejects_candidate_lcb_regression() -> None:
    result = runner._pooled_paired_gate(
        {"policy_expectancy_lcb_r": -0.03, "policy_expectancy_mean_r": 0.02},
        {"policy_expectancy_lcb_r": 0.01, "policy_expectancy_mean_r": 0.01},
    )

    assert result["eligible"] is False
    assert any("pooled_expectancy_lcb_regression" in reason for reason in result["reasons"])


def test_block_bootstrap_lcb_requires_multiple_independent_days() -> None:
    result = runner._block_bootstrap_lcb(
        {"2025-01-01": [0.5, 0.5], "2025-01-02": [0.5]},
        min_blocks=3,
    )

    assert result["eligible"] is False
    assert result["block_lcb_r"] is None
    assert any("block_samples_insufficient" in reason for reason in result["reasons"])


def test_lifecycle_day_reads_native_datetime_and_utc_timestamp() -> None:
    naive = SimpleNamespace(exit_time=datetime(2025, 1, 3, 23, 30, 0))
    aware = SimpleNamespace(
        exit_time=datetime(2025, 1, 3, 23, 30, 0, tzinfo=timezone.utc)
    )

    assert runner._lifecycle_day(naive, window=1) == "2025-01-03"
    assert runner._lifecycle_day(aware, window=1) == "2025-01-03"
    assert runner._lifecycle_day(
        SimpleNamespace(exit_time="2025-01-04T00:30:00+01:00"),
        window=1,
    ) == "2025-01-03"


def test_pooled_evidence_groups_native_datetime_by_independent_day(monkeypatch) -> None:
    lifecycles = tuple(
        SimpleNamespace(
            final_close_confirmed=True,
            metadata={"position_owner_trace_id": f"trace-{index}"},
            position_ticket=f"ticket-{index}",
            trace_id="",
            realized_r=0.25,
            exit_time=datetime(2025, 1, 1 + (index % 3), 12, 0, 0),
        )
        for index in range(30)
    )
    monkeypatch.setattr(
        runner,
        "parse_mt5_execution_ledger",
        lambda *_args, **_kwargs: SimpleNamespace(lifecycles=lifecycles),
    )
    result = runner._pooled_policy_evidence(
        [{"window": 1, "candidate": {"metrics": _metrics("one.csv")}}],
        arm="candidate",
        config=_config(),
        min_samples=30,
        min_lcb=None,
        min_block_lcb=None,
    )

    assert result["day_bootstrap"]["block_count"] == 3
    assert result["day_bootstrap"]["block_lcb_r"] is not None


def test_pooled_paired_gate_rejects_day_bootstrap_regression() -> None:
    result = runner._pooled_paired_gate(
        {
            "policy_expectancy_lcb_r": 0.10,
            "policy_expectancy_mean_r": 0.20,
            "day_bootstrap": {"block_lcb_r": 0.00},
        },
        {
            "policy_expectancy_lcb_r": 0.00,
            "policy_expectancy_mean_r": 0.10,
            "day_bootstrap": {"block_lcb_r": 0.10},
        },
    )

    assert result["eligible"] is False
    assert any("pooled_day_bootstrap_lcb_regression" in reason for reason in result["reasons"])
