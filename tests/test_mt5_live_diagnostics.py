from __future__ import annotations

import json

import pytest

from self_growing_trader.mt5_live_diagnostics import Mt5DiagnosticsMonitor, Mt5LiveDiagnostics


def _write(path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_live_monitor_retries_when_windows_reader_temporarily_locks_status(tmp_path, monkeypatch) -> None:
    output = tmp_path / "candidate"
    output.mkdir(parents=True)
    monitor = Mt5DiagnosticsMonitor(output, tmp_path / "runtime")
    original_replace = type(monitor.status_path).replace
    attempts = {"count": 0}

    def flaky_replace(source, target):
        attempts["count"] += 1
        if attempts["count"] < 3:
            raise PermissionError(5, "reader temporarily holds status")
        return original_replace(source, target)

    monkeypatch.setattr(type(monitor.status_path), "replace", flaky_replace)
    monitor._publish({"schema_version": 3, "status": {"monitor": "ok"}})

    assert attempts["count"] == 3
    assert (tmp_path / "candidate" / "mt5_diagnostics.json").is_file()


def test_live_diagnostics_separates_execution_reward_and_optimizer(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_1"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    execution = files / "self_growing_execution.csv"
    bridge = files / "self_growing_bridge.csv"
    observatory = output / "edge.execution_trace_observatory.jsonl"
    _write(execution, [
        "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\tfilled\tclosed\tprofit\tcommission\tswap\tclose_reason",
        "BATCH_RESOLUTION\tNOOP_HOLD\tpolicy\tHOLD\t1\t0\t0\t0\t0\t0\t0\t0\tNOOP_HOLD",
        "BATCH_RESOLUTION\tSCHEDULED\tpolicy\tMARKET_BUY\t1\t0\t0\t0\t0\t0\t0\t0\tSCHEDULED",
        "REQUEST\tACCEPTED\tpolicy\tMARKET_BUY\t1\t0\t1\t1\t0\t0\t0\t0\tNO_EFFECT",
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t0\t0\t0\t1\t20\t-1\t0\tTP",
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t0\t0\t0\t1\t-10\t-1\t0\tSL",
    ])
    _write(bridge, [
        "event_id\ttime\tbalance\tequity",
        "e1\t2025.01.01 00:01\t10000\t10000",
        "e1\t2025.01.01 00:01\t10000\t10000",
        "e2\t2025.01.01 00:02\t10008\t9000",
    ])
    output.mkdir(parents=True, exist_ok=True)
    _write(observatory, [
        json.dumps({
            "stage": "five_competitive_reward",
            "routing": {
                "updated": 1.0,
                "rollout_steps": 64.0,
                "policy_loss": 0.2,
                "value_loss": 0.4,
                "total_loss": 0.6,
                "entropy": 0.8,
                "actor_eligible_fraction": 0.75,
                "reward_sample_eligible_fraction": 0.6,
                "reward_nonzero_count": 3,
                "reward_closed_trades": 2,
            },
        }),
        json.dumps({
            "stage": "five_intent_registered",
            "trace": {"action_details": {
                "five_reasoning_action_scale": 0.0,
                "five_imitation_prior_enabled": False,
            }},
        }),
        json.dumps({
            "stage": "five_pattern_path_credit_accepted",
            "routing": {
                "updates": 1.0,
                "reasoning_group_eligible": 5.0,
                "reasoning_optimizer_eligible": 32.0,
                "reasoning_optimizer_updated": 1.0,
            },
        }),
        json.dumps({
            "stage": "five_pattern_path_credit_accepted",
            "routing": {
                "updates": 0.0,
                "reasoning_group_eligible": 0.0,
                "reasoning_optimizer_eligible": 0.0,
                "reasoning_optimizer_updated": 0.0,
            },
        }),
        json.dumps({
            "stage": "five_pattern_path_credit_accepted",
            "routing": {
                "updates": 1.0,
                "reasoning_group_eligible": 7.0,
                "reasoning_optimizer_eligible": 48.0,
                "reasoning_optimizer_updated": 1.0,
            },
        }),
    ])

    diagnostics = Mt5LiveDiagnostics(output, runner)
    first = diagnostics.refresh()
    second = diagnostics.refresh()  # append reader must not double count

    assert first["performance"] == second["performance"]
    assert first["optimizer"] == second["optimizer"]
    assert first["performance"]["closed_trades"] == 2
    assert first["performance"]["net_pnl"] == pytest.approx(8.0)
    assert first["performance"]["profit_factor"] == pytest.approx(19.0 / 11.0)
    assert first["performance"]["close_reason_net_pnl"] == pytest.approx({
        "TP": 19.0,
        "SL": -11.0,
    })
    assert first["performance"]["payoff_ratio"] == pytest.approx(19.0 / 11.0)
    assert first["performance"]["breakeven_win_rate"] == pytest.approx(11.0 / 30.0)
    assert first["risk"]["max_equity_drawdown_pct"] == pytest.approx(0.10)
    assert first["optimizer"]["ppo_updates"] == 1
    assert first["optimizer"]["pattern_updates"] == 2
    assert first["optimizer"]["reasoning_optimizer_updates"] == 2
    assert first["optimizer"]["optimizer_update_count"] == 3
    assert first["optimizer"]["reasoning_group_eligible"] == pytest.approx(12.0)
    assert first["optimizer"]["reasoning_optimizer_eligible"] == pytest.approx(80.0)
    assert first["optimizer"]["rollout_steps"] == 64
    assert first["reward"]["nonzero_density"] == pytest.approx(0.6)
    assert first["policy"]["reasoning_action_scale"] == 0.0
    assert first["policy"]["imitation_prior_enabled"] is False
    assert first["execution"]["requests"] == 1
    assert first["status"]["optimizer"] == "warming_up"
    assert first["status"]["overfit_live"] == "not_measurable_without_oos"


def test_live_diagnostics_exposes_codex_teacher_counters_separately(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_codex"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(files / "self_growing_execution.csv", [])
    _write(files / "self_growing_bridge.csv", ["event_id\ttime\tbalance\tequity"])
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [])
    _write(output / "run.codex_teacher_receipts.jsonl.status.json", [
        json.dumps({
            "enabled": True,
            "collected": 4,
            "graded": 4,
            "hypotheses": 4,
            "hypotheses_confirmed": 4,
            "codex_optimizer_updates": 3,
            "interventions_applied": 3,
            "failures": 0,
            "pending": 0,
            "rejected_updates": 1,
            "account_impact_records": 4,
            "account_impact_confirmed": 4,
            "account_impact_partial": 0,
            "account_cashflow_confirmed": 3,
            "account_mark_confirmed": 1,
            "account_impact_reward_sum": -1.25,
            "rlhf_parameter_changed": 3,
            "rlhf_probability_shift_count": 3,
            "rlhf_max_probability_shift": 0.048,
            "model": "gpt-5.6-luna",
            "reasoning_effort": "xhigh",
            "executable_path": "C:/Codex/codex.exe",
            "cli_version": "codex-cli test",
            "preflight": {"status": "passed"},
        }),
    ])

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["status"]["codex_teacher"] == "active"
    assert result["codex_teacher"]["submitted"] == 4
    assert result["codex_teacher"]["graded"] == 4
    assert result["codex_teacher"]["hypotheses_confirmed"] == 4
    assert result["codex_teacher"]["updates"] == 3
    assert result["codex_teacher"]["interventions_applied"] == 3
    assert result["codex_teacher"]["rejected_updates"] == 1
    assert result["codex_teacher"]["account_cashflow_confirmed"] == 3
    assert result["codex_teacher"]["account_mark_confirmed"] == 1
    assert result["codex_teacher"]["parameter_changed"] == 3
    assert result["codex_teacher"]["max_probability_shift"] == pytest.approx(0.048)
    assert result["optimizer"]["codex_teacher_updates"] == 3
    # Generic PPO/pattern counters remain semantically separate.
    assert result["optimizer"]["optimizer_update_count"] == 0


def test_live_diagnostics_counts_cashflow_scope_from_append_only_receipt(tmp_path) -> None:
    """Receipt evidence remains visible when the status sidecar is stale."""
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_codex_cashflow"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(files / "self_growing_execution.csv", [])
    _write(files / "self_growing_bridge.csv", ["event_id\ttime\tbalance\tequity"])
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [])
    _write(output / "cashflow.codex_teacher_receipts.jsonl.status.json", [
        json.dumps({
            "enabled": True,
            "submitted": 1,
            "graded": 1,
            "hypotheses": 1,
            "hypotheses_confirmed": 1,
            "updates": 1,
            "failures": 0,
            "pending": 0,
        }),
    ])
    _write(output / "cashflow.codex_teacher_receipts.jsonl", [
        json.dumps({
            "record_type": "codex_teacher_receipt",
            "packet_sha256": "a" * 64,
            "trace_id": "trace-cashflow",
            "position_ticket": "deal-position-1",
            "teacher_reward": -0.4,
            "hypothesis": "deal cashflow is the immutable account signal",
            "verification_status": "confirmed",
            "account_impact": {
                "status": "confirmed",
                "evidence_scope": "immutable_lifecycle_plus_unique_deal_cashflow",
            },
        }),
    ])

    diagnostics = Mt5LiveDiagnostics(output, runner)
    first = diagnostics.refresh()
    second = diagnostics.refresh()

    assert first["codex_teacher"]["account_impact_confirmed"] == 1
    assert first["codex_teacher"]["account_cashflow_confirmed"] == 1
    assert first["codex_teacher"]["account_mark_confirmed"] == 0
    assert first["codex_teacher"]["account_impact_partial"] == 0
    assert second["codex_teacher"] == first["codex_teacher"]


def test_live_diagnostics_flags_negative_edge_and_action_collapse(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_2"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    header = "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\tfilled\tclosed\tprofit\tcommission\tswap\tclose_reason"
    rows = [header]
    rows.extend(
        "BATCH_RESOLUTION\tNOOP_HOLD\tpolicy\tHOLD\t1\t0\t0\t0\t0\t0\t0\t0\tNOOP_HOLD"
        for _ in range(100)
    )
    rows.extend(
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t0\t0\t0\t1\t-1\t0\t0\tSL"
        for _ in range(100)
    )
    _write(files / "self_growing_execution.csv", rows)
    _write(files / "self_growing_bridge.csv", [
        "event_id\ttime\tbalance\tequity",
        "e1\t2025.01.01 00:01\t10000\t10000",
        "e2\t2025.01.01 00:02\t9900\t7900",
    ])
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [])

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["status"]["edge"] == "negative"
    assert result["status"]["risk"] == "critical"
    assert result["status"]["policy"] == "collapsed"
    assert result["status"]["capacity_diagnosis"] == "insufficient_optimization"


def test_live_diagnostics_uses_economic_actor_scope_over_wire_holds(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_economic_scope"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    header = (
        "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\t"
        "filled\tclosed\tprofit\tcommission\tswap\tclose_reason"
    )
    wire_rows = [header]
    wire_rows.extend(
        "BATCH_RESOLUTION\tNOOP_HOLD\tpolicy\tHOLD\t1\t0\t0\t0\t0\t0\t0\t0\tNOOP_HOLD"
        for _ in range(9)
    )
    wire_rows.append(
        "BATCH_RESOLUTION\tSCHEDULED\tpolicy\tBUY_LIMIT\t1\t0\t0\t0\t0\t0\t0\t0\tSCHEDULED"
    )
    _write(files / "self_growing_execution.csv", wire_rows)
    _write(files / "self_growing_bridge.csv", ["event_id\ttime\tbalance\tequity"])
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [
        json.dumps({
            "stage": "five_execution_receipt",
            "trace_id": "eligible-buy",
            "trace": {"trace_id": "eligible-buy", "action_details": {
                "five_economic_actor_eligible": True,
                "action_name": "BUY_LIMIT",
            }},
        }),
        json.dumps({
            "stage": "five_execution_receipt",
            "trace_id": "eligible-sell",
            "trace": {"trace_id": "eligible-sell", "action_details": {
                "five_economic_actor_eligible": True,
                "action_name": "SELL_STOP",
            }},
        }),
    ])

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["policy"]["scope"] == "economic_actor"
    assert result["policy"]["wire_actions"] == 10
    assert result["policy"]["wire_hold_share"] == pytest.approx(0.9)
    assert result["policy"]["action_counts"] == {
        "BUY_LIMIT": 1,
        "SELL_STOP": 1,
    }
    assert result["policy"]["economic_actor"]["total"] == 2


def test_live_diagnostics_warns_when_wire_collapses_but_economic_scope_diversifies(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_wire_collapse"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    header = (
        "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\t"
        "filled\tclosed\tprofit\tcommission\tswap\tclose_reason"
    )
    wire_rows = [header]
    wire_rows.extend(
        "BATCH_RESOLUTION\tNOOP_HOLD\tpolicy\tHOLD\t1\t0\t0\t0\t0\t0\t0\t0\tNOOP_HOLD"
        for _ in range(30)
    )
    _write(files / "self_growing_execution.csv", wire_rows)
    _write(files / "self_growing_bridge.csv", ["event_id\ttime\tbalance\tequity"])
    output.mkdir(parents=True, exist_ok=True)
    economic_rows = []
    for index in range(25):
        action = "BUY_LIMIT" if index % 2 else "SELL_STOP"
        economic_rows.append({
            "stage": "five_execution_receipt",
            "trace_id": f"economic-{index}",
            "trace": {
                "trace_id": f"economic-{index}",
                "action_details": {
                    "five_economic_actor_eligible": True,
                    "action_name": action,
                },
            },
        })
    _write(output / "edge.execution_trace_observatory.jsonl", [
        json.dumps(row) for row in economic_rows
    ])

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["policy"]["scope"] == "economic_actor"
    assert result["policy"]["wire_collapse_detected"] is True
    assert result["policy"]["economic_actor_collapse_detected"] is False
    assert result["policy"]["wire_collapse_warning"] is True


def test_live_diagnostics_detects_entry_role_collapse_separately_from_manager_holds(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_entry_role_collapse"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(files / "self_growing_execution.csv", [
        "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\tfilled\tclosed\tprofit\tcommission\tswap\tclose_reason",
    ])
    _write(files / "self_growing_bridge.csv", [
        "event_id\ttime\tbalance\tequity",
        "e1\t2025.01.01 00:01\t10000\t10000",
    ])
    observatory = output / "edge.execution_trace_observatory.jsonl"
    rows = []
    for index in range(8):
        rows.append(json.dumps({
            "stage": "five_execution_receipt",
            "trace": {
                "trace_id": f"long-{index}",
                "action_details": {
                    "action_name": "HOLD",
                    "five_agent_authority_role": "LONG_ENTRY_OWNER",
                    "five_economic_actor_eligible": False,
                },
            },
        }))
    for index in range(8):
        rows.append(json.dumps({
            "stage": "five_execution_receipt",
            "trace": {
                "trace_id": f"short-{index}",
                "action_details": {
                    "action_name": "SELL_LIMIT",
                    "five_agent_authority_role": "SHORT_ENTRY_OWNER",
                    "five_economic_actor_eligible": True,
                },
            },
        }))
    for index in range(40):
        rows.append(json.dumps({
            "stage": "five_execution_receipt",
            "trace": {
                "trace_id": f"manager-{index}",
                "action_details": {
                    "action_name": "HOLD",
                    "five_agent_authority_role": "POSITION_MANAGER",
                    "five_economic_actor_eligible": False,
                },
            },
        }))
    _write(observatory, rows)

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["status"]["policy"] == "collapsed"
    assert result["policy"]["entry_role_collapse_detected"] is True
    assert result["policy"]["entry_role_distributions"]["LONG_ENTRY_OWNER"]["hold_share"] == pytest.approx(1.0)
    assert result["policy"]["entry_role_distributions"]["SHORT_ENTRY_OWNER"]["hold_share"] == pytest.approx(0.0)


def test_live_diagnostics_rebinds_from_clone_placeholders_to_late_agent_ledgers(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_3"
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [])

    # Runner clones contain empty files before the Strategy Tester agent starts.
    placeholder_files = runner / "MQL5" / "Files"
    _write(placeholder_files / "self_growing_execution.csv", [])
    _write(placeholder_files / "self_growing_bridge.csv", [])
    diagnostics = Mt5LiveDiagnostics(output, runner)

    before_agent = diagnostics.refresh()
    assert before_agent["performance"]["closed_trades"] == 0
    assert diagnostics.execution.header == []
    assert diagnostics.bridge.header == []

    agent_files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(agent_files / "self_growing_execution.csv", [
        "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\tfilled\tclosed\tprofit\tcommission\tswap\tclose_reason",
        "BATCH_RESOLUTION\tSCHEDULED\tpolicy\tMARKET_BUY\t1\t0\t0\t0\t0\t0\t0\t0\tSCHEDULED",
        "REQUEST\tACCEPTED\tpolicy\tMARKET_BUY\t1\t0\t1\t1\t0\t0\t0\t0\tNO_EFFECT",
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t0\t0\t0\t1\t12\t-1\t0\tTP",
    ])
    _write(agent_files / "self_growing_bridge.csv", [
        "event_id\ttime\tbalance\tequity",
        "e1\t2025.01.01 00:01\t10000\t10000",
        "e2\t2025.01.01 00:02\t10011\t10011",
    ])

    after_agent = diagnostics.refresh()
    repeated = diagnostics.refresh()

    assert diagnostics.execution.reader.path == agent_files / "self_growing_execution.csv"
    assert diagnostics.bridge.reader.path == agent_files / "self_growing_bridge.csv"
    assert after_agent["performance"]["closed_trades"] == 1
    assert after_agent["performance"]["net_pnl"] == pytest.approx(11.0)
    assert after_agent["policy"]["actions"] == 1
    assert after_agent["policy"]["long_entry_actions"] == 1
    assert after_agent["policy"]["short_entry_actions"] == 0
    assert after_agent["policy"]["entry_direction_imbalance"] == pytest.approx(1.0)
    assert after_agent["execution"]["requests"] == 1
    assert after_agent["risk"]["balance"] == pytest.approx(10011.0)
    assert repeated["performance"] == after_agent["performance"]
    assert repeated["execution"] == after_agent["execution"]


def test_live_diagnostics_counts_unique_final_positions_with_duplicate_close_rows(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_final_positions"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(files / "self_growing_execution.csv", [
        (
            "transaction_type\tstatus\tsource\taction_name\tclosed\tprofit\tcommission\t"
            "swap\tclose_reason\tposition_final_close\tposition_ticket"
        ),
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t6\t0\t0\tEXPERT\t1\t42",
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t4\t0\t0\tTP\t1\t42",
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t-5\t0\t0\tSL\t1\t43",
    ])
    _write(files / "self_growing_bridge.csv", ["event_id\ttime\tbalance\tequity"])
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [])

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["schema_version"] == 3
    assert result["performance"]["closed_deals"] == 3
    assert result["performance"]["final_close_rows"] == 3
    assert result["performance"]["final_position_closes"] == 2
    assert result["performance"]["duplicate_final_close_rows"] == 1
    assert result["performance"]["position_expectancy"] == pytest.approx(2.5)
    assert result["performance"]["position_expectancy_lcb_95_one_sided"] < 0.0


def test_live_diagnostics_detects_late_hold_only_tail(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_tail"
    files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(files / "self_growing_execution.csv", [])
    _write(files / "self_growing_bridge.csv", ["event_id\ttime\tbalance\tequity"])
    output.mkdir(parents=True, exist_ok=True)
    early = [
        {
            "stage": "five_execution_receipt",
            "trace_id": f"entry-{index}",
            "trace": {
                "trace_id": f"entry-{index}",
                "action_details": {
                    "action_name": "MARKET_BUY",
                    "five_economic_actor_eligible": True,
                },
            },
        }
        for index in range(5)
    ]
    tail = [
        {
            "stage": "five_execution_receipt",
            "trace_id": f"tail-{index}",
            "trace": {
                "trace_id": f"tail-{index}",
                "action_details": {
                    "action_name": "HOLD",
                    "five_economic_actor_eligible": True,
                },
            },
        }
        for index in range(25)
    ]
    _write(
        output / "edge.execution_trace_observatory.jsonl",
        [json.dumps(row) for row in early + tail],
    )

    result = Mt5LiveDiagnostics(output, runner).refresh()

    assert result["policy"]["tail_window_size"] == 25
    assert result["policy"]["tail_collapse_detected"] is True
    assert result["policy"]["tail"]["hold_share"] == pytest.approx(1.0)


def test_live_diagnostics_follows_a_newer_tester_agent_without_losing_prior_totals(tmp_path) -> None:
    output = tmp_path / "candidate"
    runner = tmp_path / "runtime" / "mt5_v186_4"
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "edge.execution_trace_observatory.jsonl", [])
    header = (
        "transaction_type\tstatus\tsource\taction_name\taccepted\trejected\tplaced\tfilled\t"
        "closed\tprofit\tcommission\tswap\tclose_reason\tposition_final_close\tposition_ticket"
    )
    bridge_header = "event_id\ttime\tbalance\tequity"
    first_agent = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    _write(first_agent / "self_growing_execution.csv", [
        header,
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t0\t0\t0\t1\t10\t0\t0\tTP\t1\t42",
    ])
    _write(first_agent / "self_growing_bridge.csv", [
        bridge_header,
        "a1\t2025.01.01 00:01\t10010\t10010",
    ])
    diagnostics = Mt5LiveDiagnostics(output, runner)
    first = diagnostics.refresh()
    assert first["performance"]["closed_trades"] == 1

    second_agent = runner / "Tester" / "Agent-127.0.0.1-3001" / "MQL5" / "Files"
    _write(second_agent / "self_growing_execution.csv", [
        header,
        "DEAL_ADD\tCLOSED\tunknown\tSYSTEM\t1\t0\t0\t0\t1\t-4\t0\t0\tSL\t1\t42",
    ])
    _write(second_agent / "self_growing_bridge.csv", [
        bridge_header,
        "a2\t2025.01.01 00:02\t10006\t10006",
    ])

    combined = diagnostics.refresh()

    assert diagnostics.execution.reader.path == second_agent / "self_growing_execution.csv"
    assert diagnostics.bridge.reader.path == second_agent / "self_growing_bridge.csv"
    assert combined["performance"]["closed_trades"] == 2
    assert combined["performance"]["final_close_rows"] == 2
    assert combined["performance"]["final_position_closes"] == 2
    assert combined["performance"]["duplicate_final_close_rows"] == 0
    assert combined["performance"]["net_pnl"] == pytest.approx(6.0)
    assert combined["performance"]["close_reason_net_pnl"] == pytest.approx({
        "TP": 10.0,
        "SL": -4.0,
    })
    assert combined["risk"]["balance"] == pytest.approx(10006.0)
