from __future__ import annotations

import json
import csv
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from dataclasses import fields, replace
from datetime import datetime

import numpy as np
import pytest

from self_growing_trader.mt5_runner import (
    BridgeAgentAction,
    BridgeRequest,
    ExperienceStarvationState,
    MT5BacktestConfig,
    MT5BacktestMetrics,
    MT5ExecutionSummary,
    MT5PositionLifecycle,
    RuntimeFitnessGuard,
    MT5StrategyTesterGlobalLock,
    MT5StrategyTesterEvaluator,
    compute_mt5_expectancy,
    count_bridge_rows,
    mt5_fitness,
    mt5_fitness_breakdown,
    mt5_structural_j,
    reconcile_mt5_report_and_ledger,
    parse_mt5_execution_ledger,
    parse_mt5_report,
    parse_mt5_report_lifecycles,
    parse_bridge_terminal_mark,
    execution_ledger_owner_expectancy_audit,
    select_mt5_evaluation_expectancy_audit,
    settle_terminal_open_position_lifecycles,
    lifecycle_credit_candidates,
    merge_report_lifecycles,
    replace_execution_summary_lifecycles,
    write_tester_ini,
    write_tester_set,
    _closed_execution_reward,
    _execution_record_close_details,
    _merge_final_close_receipt,
    _detect_mt5_startup_failure,
    verify_offline_cache,
    _legacy_ack_sequence_for_close_row,
    _register_legacy_open_trace,
    _pending_transition_feedback_allowed,
    _direct_rnn_actor_bridge_audit,
    _report_intent_sequence,
    enrich_step_trace,
)


def test_detect_mt5_startup_failure_reports_auth_rejection_without_raw_login(tmp_path) -> None:
    runner = tmp_path / "runner"
    log_dir = runner / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "20261009.log").write_text(
        "GS 0 Startup MetaTrader x64 build 6182 started\n"
        "GS 2 Network '411899577': authorization on Exness-MT5Real8 failed (Invalid account)\n"
        "IN 2 Tester terminal is not synchronized with the trade server before start automatic testing [0]\n",
        encoding="utf-8",
    )

    result = _detect_mt5_startup_failure(runner)

    assert result is not None
    assert result["failure_type"] == "tester_not_synchronized"
    assert result["server"] == "Exness-MT5Real8"
    assert result["authorization_reason"] == "Invalid account"
    assert "411899577" not in str(result)


def test_detect_mt5_startup_failure_accepts_later_success_marker(tmp_path) -> None:
    runner = tmp_path / "runner"
    log_dir = runner / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "20261009.log").write_text(
        "GS 2 Network authorization on Exness-MT5Real8 failed (Invalid account)\n"
        "JR 0 Network terminal synchronized with Exness Technologies Ltd\n"
        "GG 0 Tester automatic testing started\n",
        encoding="utf-8",
    )

    assert _detect_mt5_startup_failure(runner) is None


def test_detect_mt5_startup_failure_offline_preflight_bypasses_auth_failure(tmp_path) -> None:
    runner = tmp_path / "runner"
    log_dir = runner / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "20261009.log").write_text(
        "GS 0 Startup MetaTrader x64 build 6182 started\n"
        "GS 2 Network '411899577': authorization on Exness-MT5Real8 failed (Invalid account)\n",
        encoding="utf-8",
    )

    result = _detect_mt5_startup_failure(runner, offline_preflight=True)
    assert result is None


def test_detect_mt5_startup_failure_offline_preflight_bypasses_tester_not_synchronized(tmp_path) -> None:
    runner = tmp_path / "runner"
    log_dir = runner / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "20261009.log").write_text(
        "GS 2 Network '411899577': authorization on Exness-MT5Real8 failed (Invalid account)\n"
        "IN 2 Tester terminal is not synchronized with the trade server before start automatic testing [0]\n",
        encoding="utf-8",
    )

    result = _detect_mt5_startup_failure(runner, offline_preflight=True)
    assert result is None


def test_detect_mt5_startup_failure_recognizes_testing_started_on(tmp_path) -> None:
    runner = tmp_path / "runner"
    log_dir = runner / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "20261009.log").write_text(
        "GS 2 Network authorization on Exness-MT5Real8 failed (Invalid account)\n"
        "GG 0 Tester testing started on 2026.01.01\n",
        encoding="utf-8",
    )

    assert _detect_mt5_startup_failure(runner, offline_preflight=False) is None


def test_detect_mt5_startup_failure_multi_file_global_line_counter_preserves_new_crash(tmp_path) -> None:
    runner = tmp_path / "runner"
    log_dir = runner / "logs"
    log_dir.mkdir(parents=True)

    # Older log file with 100 lines and a success marker at the end
    f1 = log_dir / "20261001.log"
    f1_lines = ["line\n"] * 100 + ["Tester automatic testing started\n"]
    f1.write_text("".join(f1_lines), encoding="utf-8")

    # Newer log file with a crash at line 5
    f2 = log_dir / "20261002.log"
    f2_lines = ["line\n"] * 5 + ["terminal crashed\n"]
    f2.write_text("".join(f2_lines), encoding="utf-8")

    os.utime(f1, (1000.0, 1000.0))
    os.utime(f2, (2000.0, 2000.0))

    res = _detect_mt5_startup_failure(runner)
    assert res is not None, "Terminal crash in newer log file must be detected"
    assert res.get("failure_type") == "terminal_crashed"
    assert "terminal crashed" in str(res.get("reason", "")).lower()


def test_verify_offline_cache_local_data() -> None:
    runner_dir = Path("mt5_v2_runner")
    res = verify_offline_cache(runner_dir, "BTCUSDm")
    assert res.get("available") is True
    assert int(res.get("hcc_files", 0)) > 0 or int(res.get("tkc_files", 0)) > 0


def test_verify_offline_cache_missing_bases(tmp_path) -> None:
    res = verify_offline_cache(tmp_path, "BTCUSDm")
    assert res.get("available") is False
    assert res.get("reason") == "missing_bases"
from self_growing_trader.self_growing_runner import SelfGrowingPolicy
from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    ExperienceSuperNetwork,
    StepTrace,
    _apply_entry_timing_extrema_bias,
    _cau_hinh_policy_chuyen_gia_cho_mt5,
    _mt5_execution_contract_kwargs,
    _five_action_distribution_requires_champion_recovery,
)
from self_growing_trader.config import AppConfig, FeatureConfig, LearningIntegrityConfig, load_config
from self_growing_trader.he_chuyen_gia_ngu_canh import CauHinhHeChuyenGia, HeChuyenGiaMotContextMotPolicy
from self_growing_trader.learning_integrity import sha256_file
from self_growing_trader.orders import Action, PositionManagementMode
from self_growing_trader.input_normalization import normalize_input_vector
from self_growing_trader.protocol_contract import BRIDGE_SCHEMA_VERSION, EXECUTION_LEDGER_SCHEMA_VERSION, INPUT_COUNT
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
from self_growing_trader.five_agent_authority import five_agent_authority
from self_growing_trader.entry_preference import EntryPathBar, EntryPreferenceRecord
from self_growing_trader.rlhf_policy_loop import RLHF_POLICY_FEATURE_SCHEMA
from self_growing_trader.rlhf_reward_model import RLHFRewardModel


def test_champion_recovery_uses_frozen_continuation_thresholds() -> None:
    assert _five_action_distribution_requires_champion_recovery({
        "collapsed": False,
        "hold_share": 0.94,
        "effective_action_count": 1.40,
        "normalized_action_entropy": 0.17,
    }) is True
    assert _five_action_distribution_requires_champion_recovery({
        "collapsed": False,
        "hold_share": 0.80,
        "effective_action_count": 2.0,
        "normalized_action_entropy": 0.30,
    }) is True
    assert _five_action_distribution_requires_champion_recovery({
        "collapsed": False,
        "hold_share": 0.60,
        "effective_action_count": 2.5,
        "normalized_action_entropy": 0.55,
    }) is False


def test_champion_recovery_uses_economic_actor_scope_when_wire_holds_dominate() -> None:
    assert _five_action_distribution_requires_champion_recovery({
        "collapsed": True,
        "hold_share": 0.94,
        "effective_action_count": 1.40,
        "normalized_action_entropy": 0.17,
        "economic_actor": {
            "available": True,
            "total": 20,
            "collapsed": False,
            "hold_share": 0.20,
            "effective_action_count": 4.0,
            "normalized_action_entropy": 0.72,
        },
    }) is False


def test_five_action_distribution_snapshot_separates_wire_and_economic_actor() -> None:
    evaluator = MT5StrategyTesterEvaluator.__new__(MT5StrategyTesterEvaluator)
    evaluator._five_action_by_trace = {
        "inactive-hold": int(Action.HOLD),
        "eligible-buy": int(Action.BUY_LIMIT),
        "eligible-sell": int(Action.SELL_STOP),
    }
    evaluator._five_policy_action_by_trace = {
        "eligible-buy": int(Action.BUY_LIMIT),
        "eligible-sell": int(Action.SELL_STOP),
    }
    evaluator._five_economic_actor_eligible_by_trace = {
        "eligible-buy": True,
        "eligible-sell": True,
    }
    evaluator._five_agent_by_trace = {
        "inactive-hold": 2,
        "eligible-buy": 0,
        "eligible-sell": 1,
    }

    snapshot = evaluator._five_action_distribution_snapshot()

    assert snapshot["collapse_scope"] == "economic_actor"
    assert snapshot["wire"]["hold_share"] == pytest.approx(1.0 / 3.0)
    assert snapshot["economic_actor"]["total"] == 2
    assert snapshot["economic_actor"]["hold_share"] == pytest.approx(0.0)
    assert snapshot["entry_role_distributions"]["LONG_ENTRY_OWNER"]["total"] == 1
    assert snapshot["entry_role_distributions"]["SHORT_ENTRY_OWNER"]["total"] == 1
    assert snapshot["entry_role_collapse_detected"] is False


def test_five_action_distribution_detects_entry_owner_collapse_without_manager_holds() -> None:
    evaluator = MT5StrategyTesterEvaluator.__new__(MT5StrategyTesterEvaluator)
    actions = {
        **{f"long-{index}": int(Action.HOLD) for index in range(8)},
        **{f"short-{index}": int(Action.SELL_LIMIT) for index in range(8)},
        **{f"manager-{index}": int(Action.HOLD) for index in range(40)},
    }
    evaluator._five_action_by_trace = actions
    evaluator._five_policy_action_by_trace = dict(actions)
    evaluator._five_economic_actor_eligible_by_trace = {
        trace_id: trace_id.startswith("long-") or trace_id.startswith("short-")
        for trace_id in actions
    }
    evaluator._five_agent_by_trace = {
        **{f"long-{index}": 0 for index in range(8)},
        **{f"short-{index}": 1 for index in range(8)},
        **{f"manager-{index}": 2 for index in range(40)},
    }

    snapshot = evaluator._five_action_distribution_snapshot()

    assert snapshot["entry_role_distributions"]["LONG_ENTRY_OWNER"]["hold_share"] == pytest.approx(1.0)
    assert snapshot["entry_role_distributions"]["SHORT_ENTRY_OWNER"]["hold_share"] == pytest.approx(0.0)
    assert snapshot["entry_role_collapse_detected"] is True
    assert _five_action_distribution_requires_champion_recovery(snapshot) is True


def test_five_action_distribution_ignores_manager_hold_collapse_for_entry_guard() -> None:
    evaluator = MT5StrategyTesterEvaluator.__new__(MT5StrategyTesterEvaluator)
    actions = {f"manager-{index}": int(Action.HOLD) for index in range(40)}
    evaluator._five_action_by_trace = actions
    evaluator._five_policy_action_by_trace = dict(actions)
    evaluator._five_economic_actor_eligible_by_trace = {
        trace_id: False for trace_id in actions
    }
    evaluator._five_agent_by_trace = {
        trace_id: 2 for trace_id in actions
    }

    snapshot = evaluator._five_action_distribution_snapshot()

    assert snapshot["entry_role_distributions"] == {}
    assert snapshot["entry_role_collapse_detected"] is False


def test_five_action_distribution_snapshot_detects_late_tail_hold_collapse() -> None:
    evaluator = MT5StrategyTesterEvaluator.__new__(MT5StrategyTesterEvaluator)
    actions: dict[str, int] = {
        **{f"entry-{index}": int(Action.BUY_LIMIT) for index in range(6)},
        **{f"tail-{index}": int(Action.HOLD) for index in range(25)},
    }
    evaluator._five_action_by_trace = actions
    evaluator._five_policy_action_by_trace = dict(actions)
    evaluator._five_economic_actor_eligible_by_trace = {
        trace_id: True for trace_id in actions
    }

    snapshot = evaluator._five_action_distribution_snapshot()

    assert snapshot["collapsed"] is False
    assert snapshot["tail_window_size"] == 25
    assert snapshot["tail_collapse_detected"] is True
    assert snapshot["tail"]["hold_share"] == pytest.approx(1.0)
    assert _five_action_distribution_requires_champion_recovery(snapshot) is True


class _BridgeTestPolicy:
    input_size = INPUT_COUNT
    output_size = 20
    modules = []

    def __init__(self) -> None:
        self.acts = 0
        self.observes = 0
        self.develops = 0
        self.next_obs_seen = None
        self.trace_ids_seen = []

    def ensure_compatible(self):
        pass

    def reset_runtime_state(self):
        pass

    def act(self, observation, goal=None):
        self.acts += 1
        output = np.zeros(20, dtype=float)
        output[int(Action.MARKET_BUY)] = 1.0
        return SimpleNamespace(
            trace_id=f"trace-{self.acts}",
            selected_action=output,
            selected_idx=int(Action.MARKET_BUY),
            confidence=0.5,
            context_audit={},
        )

    def observe_transition(self, trace, next_obs, reward, cost, done=False, **kwargs):
        self.observes += 1
        self.next_obs_seen = np.asarray(next_obs, dtype=float)
        self.trace_ids_seen.append(trace.trace_id)
        return {"prediction_error": 0.1, "td_delta": float(reward) - float(cost)}

    def develop(self, trace, signal):
        self.develops += 1
        return []


def test_mt5_global_lock_serializes_sibling_processes(tmp_path) -> None:
    lock_path = tmp_path / "mt5.global.lock"
    child = (
        "import json,sys,time; "
        "from self_growing_trader.mt5_runner import MT5StrategyTesterGlobalLock; "
        "lock=MT5StrategyTesterGlobalLock(sys.argv[1]); "
        "lock.__enter__(); "
        "print(json.dumps({'wait': lock.waited_seconds}), flush=True); "
        "time.sleep(float(sys.argv[2])); "
        "lock.__exit__(None,None,None)"
    )
    first = subprocess.Popen(
        [sys.executable, "-c", child, str(lock_path), "8.0"],
        cwd=str(Path.cwd()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        first_line = first.stdout.readline() if first.stdout is not None else ""
        assert json.loads(first_line)["wait"] < 0.5
        assert first.poll() is None
        second = subprocess.Popen(
            [sys.executable, "-c", child, str(lock_path), "0"],
            cwd=str(Path.cwd()),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        second_line, second_err = second.communicate(timeout=20)
        assert second.returncode == 0, second_err
        assert json.loads(second_line)["wait"] >= 0.25, (first_line, second_line)
    finally:
        first.wait(timeout=10)


def test_phat_moi_action_dong_quan_ly_dua_tren_snapshot_truoc_quyet_dinh(tmp_path) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        offline_replay=True,
        five_invalid_position_action_no_position_penalty_r=1.25,
    ))
    invalid_actions = (
        Action.CLOSE_WORST,
        Action.CLOSE_BEST,
        Action.CLOSE_ALL,
        Action.MANAGE_POSITION,
        Action.HOLD,
    )
    pending = tuple(
        {
            "agent_id": agent,
            "action": int(action),
            "trace": SimpleNamespace(action_details={
                "policy_action": int(action),
                "five_action_index": int(action),
            }),
        }
        for agent, action in enumerate(invalid_actions)
    )
    penalties, positions_before = evaluator._five_invalid_position_action_no_position_penalties(
        BridgeRequest(sequence=1, inputs=np.zeros(INPUT_COUNT), positions=0, positions_pipe=""),
        pending,
    )
    assert positions_before == 0
    np.testing.assert_allclose(penalties, [1.25, 1.25, 1.25, 1.25, 0])

    no_penalties, positions_before = evaluator._five_invalid_position_action_no_position_penalties(
        BridgeRequest(
            sequence=2,
            inputs=np.zeros(INPUT_COUNT),
            positions=1,
            positions_pipe="7:1:0.1:100:99:102:0.2:3",
        ),
        pending,
    )
    assert positions_before == 1
    np.testing.assert_allclose(no_penalties, 0.0)


@pytest.mark.parametrize("manage_mode", tuple(PositionManagementMode))
def test_phat_manage_khi_trong_khong_phu_thuoc_mode(tmp_path, manage_mode) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        offline_replay=True,
        five_invalid_position_action_no_position_penalty_r=1.0,
    ))
    pending = ({
        "agent_id": 0,
        "action": int(Action.MANAGE_POSITION),
        "trace": SimpleNamespace(action_details={
            "policy_action": int(Action.MANAGE_POSITION),
            "five_action_index": int(Action.MANAGE_POSITION),
            "five_manage_mode_index": int(manage_mode),
            "trailing_stop_atr": 1.5,
            "partial_close_pct": 0.5,
        }),
    },)
    penalties, positions_before = evaluator._five_invalid_position_action_no_position_penalties(
        BridgeRequest(sequence=3, inputs=np.zeros(INPUT_COUNT), positions=0, positions_pipe=""),
        pending,
    )
    assert positions_before == 0
    np.testing.assert_allclose(penalties, [1.0, 0, 0, 0, 0])


def test_phat_nang_cancel_khi_snapshot_khong_co_lenh_cho(tmp_path) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        offline_replay=True,
        five_invalid_cancel_no_pending_penalty_r=2.0,
    ))
    pending = tuple(
        {
            "agent_id": agent,
            "action": int(action),
            "trace": SimpleNamespace(action_details={
                "policy_action": int(action),
                "five_action_index": int(action),
            }),
        }
        for agent, action in enumerate((Action.CANCEL_ORDERS, Action.MARKET_BUY))
    )

    penalties, pending_before = evaluator._five_invalid_cancel_no_pending_penalties(
        BridgeRequest(sequence=4, inputs=np.zeros(INPUT_COUNT), orders_pipe=""),
        pending,
    )
    assert pending_before == 0
    np.testing.assert_allclose(penalties, [2.0, 0, 0, 0, 0])

    no_penalties, pending_before = evaluator._five_invalid_cancel_no_pending_penalties(
        BridgeRequest(
            sequence=5,
            inputs=np.zeros(INPUT_COUNT),
            orders_pipe="81:4:0.10:2000.0",
        ),
        pending,
    )
    assert pending_before == 1
    np.testing.assert_allclose(no_penalties, 0.0)


def test_ledger_chi_phat_timeout_va_bo_qua_policy_huy_som(tmp_path, monkeypatch) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        offline_replay=True,
        five_unfilled_pending_timeout_penalty_r=0.25,
        pending_expiry_bars=5,
    ))
    ledger = tmp_path / "execution.csv"
    fields = [
        "source", "transaction_type", "status", "action_name", "cancelled",
        "trace_id", "order_ticket", "age_bars",
    ]
    with ledger.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow({
            "source": "policy", "transaction_type": "ORDER_DELETE",
            "status": "CANCELLED", "action_name": "SELL_LIMIT", "cancelled": "1",
            "trace_id": "five:owner:2", "order_ticket": "77",
            "age_bars": "2",
        })
        writer.writerow({
            "source": "policy", "transaction_type": "ORDER_DELETE",
            "status": "UNFILLED_TIMEOUT", "action_name": "BUY_STOP", "cancelled": "1",
            "trace_id": "five:timeout:1", "order_ticket": "78", "age_bars": "5",
        })
    monkeypatch.setattr(evaluator, "_latest_runtime_file", lambda _name: ledger)

    calls = []

    class Policy:
        def apply_five_unfilled_pending_timeout_penalty(self, trace_id, penalty_r):
            calls.append((trace_id, penalty_r))
            return {"accepted": True, "updated": 0, "reason": "queued"}

    first = evaluator._refresh_five_pending_order_timeouts(Policy())
    second = evaluator._refresh_five_pending_order_timeouts(Policy())
    assert first == {
        "observed": 1, "accepted": 1, "updated": 0, "cache_miss": 0,
    }
    assert second == {
        "observed": 0, "accepted": 0, "updated": 0, "cache_miss": 0,
    }
    assert calls == [
        ("five:timeout:1", pytest.approx(0.25)),
    ]


def _context_policy() -> HeChuyenGiaMotContextMotPolicy:
    return HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=_BridgeTestPolicy,
    )


def _write_fake_ea_artifacts(runner: Path) -> None:
    """Create a deliberately minimal source/binary pair for isolated runner tests.

    Production must use a compiled, hash-checked EA. These fixtures only let
    Python bridge tests reach their fake terminal process after the explicit
    fail-close EA integrity gate.
    """
    expert_dir = runner / "MQL5" / "Experts"
    expert_dir.mkdir(parents=True, exist_ok=True)
    source = expert_dir / "SelfGrowing_MT5_Trainer_EA.mq5"
    binary = expert_dir / "SelfGrowing_MT5_Trainer_EA.ex5"
    source.write_text("// test fixture EA source\n", encoding="utf-8")
    binary.write_bytes(b"test-fixture-ex5")
    # Keep EX5 newer than source so the stale build guard remains active.
    os.utime(binary, None)


def test_publish_policy_describes_python_action_bridge(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    policy = _context_policy()
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    summary = evaluator.publish_policy(policy)

    assert summary == {
        "bridge_mode": "python_action",
        "policy_runtime": "mot_context_mot_chuyen_gia",
        "so_chuyen_gia": len(policy.experts),
        "policy_chung_ra_lenh": False,
        "tron_policy": False,
    }


def test_five_competition_ledger_attributes_profit_cost_and_open_r_by_trace(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=runner, initial_equity=10_000.0, five_training_schedule_steps=160,
    ))
    evaluator._register_five_competitor_trace("trace-a0", 0)
    evaluator._register_five_competitor_trace("trace-a1", 1)
    lifecycle = MT5PositionLifecycle(
        position_ticket="p0", entry_time=None, exit_time=None,
        net_pnl=80.0, gross_profit=82.0, commission=-2.0, swap=0.0,
        initial_risk=100.0, realized_r=0.8, trace_id="trace-a0",
        close_reason="TP", final_close_confirmed=True,
    )
    assert evaluator._record_five_competition_lifecycles([lifecycle]) == 1
    assert evaluator._record_five_competition_lifecycles([lifecycle]) == 0

    position = ":".join([
        "p1", "1", "0.1", "100", "90", "120", "0.4", "3",
        "0.5", "-0.1", "0", "0", "0", "0", "1", "0", "0",
        "CTX_CHUNG", "BO_NAO_CHUNG", "BO_NAO_CHUNG", "trace-a1",
    ])
    request = BridgeRequest(
        sequence=2, inputs=np.zeros(INPUT_COUNT), time="2025.01.01 00:01:00",
        balance=10_080.0, equity=10_100.0, drawdown_pct=0.5,
        positions=1, positions_pipe=position,
    )
    economic = evaluator._five_economic_inputs(
        request, np.asarray([0, 0, 1, 0, 0], dtype=np.float32)
    )
    np.testing.assert_allclose(
        economic["agent_mark_to_market_r"], [0.82, 0.4, 0.0, 0.0, 0.0], atol=1e-7
    )
    np.testing.assert_allclose(economic["transaction_cost_r"], [0.02, 0, 0, 0, 0], atol=1e-7)
    np.testing.assert_allclose(economic["agent_risk_ratio"], [0, 1, 0, 0, 0])
    np.testing.assert_allclose(economic["conflict_cost_r"], [0, 0, 1, 0, 0])
    assert economic["team_equity_r"] == pytest.approx(1.0)
    assert economic["team_drawdown"] == pytest.approx(0.5)
    assert economic["training_progress"] == pytest.approx(1 / 160)

    # Transaction cost is consumed exactly once; cumulative gross PnL stays.
    later = evaluator._five_economic_inputs(replace(request, sequence=3), np.zeros(5))
    np.testing.assert_allclose(later["transaction_cost_r"], np.zeros(5))
    assert later["agent_mark_to_market_r"][0] == pytest.approx(0.82)


def test_five_competition_ledger_feeds_reward_v2_once_per_owner(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=runner,
        initial_equity=10_000.0,
        five_reward_expectancy_floor_r=0.05,
        five_reward_tail_loss_threshold_r=1.0,
        five_reward_tail_loss_cap_r=3.0,
    ))
    evaluator._register_five_competitor_trace("trace-win", 0)
    evaluator._register_five_competitor_trace("trace-loss", 4)
    win = MT5PositionLifecycle(
        position_ticket="p-win", entry_time=None, exit_time=None,
        net_pnl=80.0, gross_profit=82.0, commission=-2.0, swap=0.0,
        initial_risk=100.0, realized_r=0.8, trace_id="trace-win",
        close_reason="TP", final_close_confirmed=True,
    )
    loss = MT5PositionLifecycle(
        position_ticket="p-loss", entry_time=None, exit_time=None,
        net_pnl=-150.0, gross_profit=-150.0, commission=0.0, swap=0.0,
        initial_risk=100.0, realized_r=-1.5, trace_id="trace-loss",
        close_reason="SL", final_close_confirmed=True,
    )

    assert evaluator._record_five_competition_lifecycles([win, loss]) == 2
    assert evaluator._record_five_competition_lifecycles([win, loss]) == 0
    request = BridgeRequest(
        sequence=1, inputs=np.zeros(INPUT_COUNT), time="2025.01.01 00:01:00",
        balance=9_930.0, equity=9_930.0, drawdown_pct=0.7,
        positions=0, positions_pipe="",
    )
    economic = evaluator._five_economic_inputs(request, np.zeros(5))

    np.testing.assert_allclose(economic["agent_closed_trade_count"], [1, 0, 0, 0, 1])
    np.testing.assert_allclose(economic["agent_winning_trade_count"], [1, 0, 0, 0, 0])
    np.testing.assert_allclose(economic["agent_realized_trade_r"], [0.8, 0, 0, 0, -1.5])
    np.testing.assert_allclose(
        economic["agent_expectancy_shortfall_r"], [0, 0, 0, 0, 1.55], atol=1e-7
    )
    np.testing.assert_allclose(
        economic["agent_tail_loss_cost_r"], [0, 0, 0, 0, 0.25], atol=1e-7
    )

    # A later bridge request sees the same cumulative ledger facts. The reward
    # shaper consumes their delta, so re-reading the ledger cannot credit them.
    later = evaluator._five_economic_inputs(replace(request, sequence=2), np.zeros(5))
    for key in (
        "agent_closed_trade_count", "agent_winning_trade_count",
        "agent_realized_trade_r", "agent_expectancy_shortfall_r",
        "agent_tail_loss_cost_r",
    ):
        np.testing.assert_allclose(later[key], economic[key])


def test_five_trace_cannot_change_competitor_and_schedule_must_cover_rollout(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    evaluator._register_five_competitor_trace("immutable-trace", 0)
    with pytest.raises(RuntimeError, match="cannot move"):
        evaluator._register_five_competitor_trace("immutable-trace", 1)
    with pytest.raises(ValueError, match="at least one 64-step rollout"):
        MT5BacktestConfig(runner_dir=runner, five_training_schedule_steps=63)
    with pytest.raises(ValueError, match="thresholds must be non-negative"):
        MT5BacktestConfig(runner_dir=runner, five_reward_tail_loss_threshold_r=-0.1)


def test_five_reward_rejects_lifecycle_owned_by_hold_trace(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    evaluator._register_five_competitor_trace("hold-trace", 3, int(Action.HOLD))
    lifecycle = MT5PositionLifecycle(
        position_ticket="bad-owner", entry_time=None, exit_time=None,
        net_pnl=-100.0, gross_profit=-100.0, commission=0.0, swap=0.0,
        initial_risk=100.0, realized_r=-1.0, trace_id="hold-trace",
        close_reason="SL", final_close_confirmed=True,
    )

    with pytest.raises(RuntimeError, match="not an entry action"):
        evaluator._record_five_competition_lifecycles([lifecycle])


def test_read_versioned_bridge_request(tmp_path) -> None:
    request = tmp_path / "self_growing_request.csv"
    request.write_text(
        "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\n"
        "2\tevt-7\t7\t2025.01.01\t10000\t9990\t0.1\t12\t1\t3\t0.1|0.2|0.3\n",
        encoding="utf-8",
    )

    parsed = MT5StrategyTesterEvaluator._read_request(request)

    assert parsed is not None
    assert parsed.schema_version == 2
    assert parsed.event_id == "evt-7"
    assert parsed.sequence == 7
    assert parsed.equity == 9990.0
    assert parsed.drawdown_pct == 0.1
    assert parsed.spread_points == 12.0
    assert parsed.positions == 1
    assert parsed.day_trades == 3
    assert parsed.inputs.tolist() == [0.1, 0.2, 0.3]


def test_mt5_ea_contract_matches_python_protocol_versions() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    text = source.read_text(encoding="utf-8", errors="strict")

    bridge = re.search(r"#define\s+BRIDGE_SCHEMA_VERSION\s+(\d+)", text)
    ledger = re.search(r"#define\s+EXECUTION_LEDGER_SCHEMA_VERSION\s+(\d+)", text)
    assert bridge is not None
    assert ledger is not None
    assert int(bridge.group(1)) == BRIDGE_SCHEMA_VERSION
    assert int(ledger.group(1)) == EXECUTION_LEDGER_SCHEMA_VERSION
    assert "trace_id" in text
    assert "position_final_close" in text
    assert "position_owner_trace_id" in text


def test_mt5_five_ea_khong_mask_goi_hanh_dong_va_timeout_pending_sau_nam_bar() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (
        root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8", errors="strict")
    batch = text[
        text.index("void ResolveAndExecuteActionBatch("):
        text.index("void OnTick()")
    ]
    assert "ResolvePolicyRow(" in batch
    assert "BATCH_DIRECTION_CONFLICT" not in batch
    assert "BATCH_CAPACITY_BLOCKED" not in batch
    assert "BATCH_CONTROL_PRIORITY" not in batch
    assert "CancelExpiredPendingOrders();" in text
    assert '"UNFILLED_TIMEOUT"' in text
    assert "MaxPositions > 0 && CountPositions() + CountPolicyOrders() >= MaxPositions" in text
    assert "input int MaxPositions = 0;" in text


def test_mt5_config_accepts_zero_as_unlimited_position_quota(tmp_path) -> None:
    config = MT5BacktestConfig(runner_dir=tmp_path, max_positions=0)
    assert config.max_positions == 0


def test_mt5_unlimited_quota_does_not_mask_entries_with_many_positions(tmp_path) -> None:
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=tmp_path, max_positions=0, offline_replay=True)
    )
    request = BridgeRequest(
        sequence=1,
        inputs=np.zeros(INPUT_COUNT),
        positions=20,
        positions_pipe="^".join(
            f"{1000 + i}:1:0.01:100:99:102:0:3" for i in range(20)
        ),
    )
    allowed = evaluator._allowed_actions_for_request(request)
    assert {1, 2, 3, 4, 5, 6} <= allowed
    assert evaluator._execution_context_for_request(request, allowed)["position_capacity_remaining"] == 1.0


def test_mt5_ea_execution_contract_uses_fixed_atr_geometry_and_large_lifecycle_ring() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    text = source.read_text(encoding="utf-8", errors="strict")

    lifecycle = re.search(r"#define\s+POSITION_LIFECYCLE_MAX\s+(\d+)", text)
    fixed_sl_atr = re.search(r"input\s+double\s+FixedSlAtr\s*=\s*([0-9.]+)\s*;", text)
    fixed_tp_atr = re.search(r"input\s+double\s+FixedTpAtr\s*=\s*([0-9.]+)\s*;", text)
    assert lifecycle is not None
    assert fixed_sl_atr is not None
    assert fixed_tp_atr is not None
    assert int(lifecycle.group(1)) >= 4096
    assert float(fixed_sl_atr.group(1)) == 2.5
    assert float(fixed_tp_atr.group(1)) == 5.0
    assert "g_nextLifecycleSlot" in text
    assert "ResetLifecycleSlot" in text


def test_mt5_ea_hides_tester_indicators_before_creating_feature_handles() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    text = source.read_text(encoding="utf-8", errors="strict")

    on_init = text[text.index("int OnInit()"):text.index("void OnDeinit(")]
    assert "TesterHideIndicators(true);" in on_init
    assert on_init.index("TesterHideIndicators(true);") < on_init.index("ResetBridge();")
    assert "RemoveVisualIndicators" not in text
    # Feature extraction stays internal: cleaning the chart must not remove
    # the computation handles declared by the included feature module.
    features = (root / "mt5_v2_runner" / "MQL5" / "Include" / "compute_features.mqh").read_text(
        encoding="utf-8", errors="strict"
    )
    assert "g_rsi14Handle = iRSI" in features
    assert "g_macdHandle = iMACD" in features


def test_mt5_ea_prewarms_feature_buffers_before_headless_ticks_start() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "mt5_v2_runner"
        / "MQL5"
        / "Experts"
        / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8", errors="strict")
    features = (root / "mt5_v2_runner" / "MQL5" / "Include" / "compute_features.mqh").read_text(
        encoding="utf-8", errors="strict"
    )

    on_init = source[source.index("int OnInit()"):source.index("void OnDeinit(")]
    assert "if(!WaitForFeatureBuffers())" in on_init
    assert "bool WaitForFeatureBuffers(int timeoutMs = 5000)" in features
    warmup = features[features.index("bool WaitForFeatureBuffers("):features.index("double GetBufAt(")]
    assert "if(FeatureBuffersReady())" in warmup
    assert "Sleep(10);" in warmup
    assert "return FeatureBuffersReady();" in warmup


def test_mt5_ea_uses_m1_timestamp_not_preloaded_history_count_for_decision_cadence() -> None:
    """Headless tester preloads history, so iBars cannot be the M1 clock."""
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "mt5_v2_runner"
        / "MQL5"
        / "Experts"
        / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8", errors="strict")

    on_tick = source[source.index("void OnTick()"):source.index("void WriteRequest(")]
    assert "long tickEpoch = (long)TimeCurrent();" in on_tick
    assert "datetime decisionBarTime = (datetime)(tickEpoch - (tickEpoch % 60));" in on_tick
    assert "if(decisionBarTime == g_lastDecisionBarTime)" in on_tick
    assert "g_lastDecisionBarTime = decisionBarTime;" in on_tick
    assert "if(bars == currentBar)" not in on_tick
    assert "if(bars < RequiredFeatureBars())" not in on_tick
    assert "if(MaxDecisionPackets > 0 && sequence >= MaxDecisionPackets" in on_tick
    assert "TesterStop();" in on_tick
    bridge_wait = source[source.index("string RequestPythonActionBatch("):source.index("void AppendBridge(")]
    assert "IsStopped()" not in bridge_wait


def test_execution_credit_v16_separates_owner_geometry_exit_and_excursion_audit() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    text = source.read_text(encoding="utf-8", errors="strict")

    assert f"#define EXECUTION_LEDGER_SCHEMA_VERSION {EXECUTION_LEDGER_SCHEMA_VERSION}" in text
    assert '"intent_context", "intent_expert", "intent_policy"' in text
    assert '"position_owner_trace_id", "position_owner_context"' in text
    resolve = text[text.index("int ResolveIntent("):text.index("void MapIntentTickets(")]
    assert resolve.index("g_activeIntentIndex") < resolve.index("FindIntentByOrder")
    assert resolve.index("FindIntentByOrder") < resolve.index("FindIntentByDeal")
    assert resolve.index("FindIntentByDeal") < resolve.index("FindIntentByPosition")
    assert resolve.index("FindIntentByPosition") < resolve.index("FindIntentBySequenceAndAgent")
    assert resolve.index("FindIntentBySequenceAndAgent") < resolve.index("FindUniqueIntentBySequence")
    unique = text[text.index("int FindUniqueIntentBySequence("):text.index("int IntentSequenceFromComment(")]
    assert "if(match >= 0)" in unique
    assert "return -1;" in unique
    assert 'return "SG5:" +' in text
    assert "IntentAgentFromComment" in text
    assert "commentSequence, commentAgent" in text
    assert "DEAL_COMMENT" in text
    assert "ORDER_COMMENT" in text
    assert "dealEntry == DEAL_ENTRY_IN || dealEntry == DEAL_ENTRY_INOUT" in text
    assert '"old_sl_price", "old_tp_price", "new_sl_price", "new_tp_price"' in text
    assert '"exit_kind", "exit_controller", "exit_trace_id"' in text
    assert "LifecycleInitialRiskDistancePrice" in text
    assert "POSITION_TRAJECTORY" in text
    assert "FixedSlAtr" in text
    assert "FixedTpAtr" in text
    assert '"geometry_entry_pipe", "geometry_path_pipe"' in text
    assert "requestedEntryPrice" in text
    assert "RecordIntentRequestedGeometry" in text
    assert "proposedTrailingAtr" in text
    assert "five_trailing_head" in text
    assert "pathUnrealizedR" in text
    assert "pathMfeR" in text
    assert "pathMaeR" in text
    assert "g_lifecycleMfeR[pathIdx]" in text
    assert "g_lifecycleMaeR[pathIdx]" in text
    assert 'string geometryPathPipe = "";' in text
    assert 'if(ev.transactionType == "POSITION_TRAJECTORY")' in text
    assert "TRADE_ACTION_SLTP" in text
    assert "double accountBalanceAfter;" in text
    assert "double accountEquityAfter;" in text
    assert "if(closed && positionFinalClose)" in text
    assert "AccountInfoDouble(ACCOUNT_BALANCE)" in text
    assert "DoubleToString(ev.accountBalanceAfter, 12)" in text


def test_mt5_management_actions_are_scoped_to_position_owner() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    text = source.read_text(encoding="utf-8", errors="strict")

    assert "bool PositionMatchesOwner(string ownerExpert)" in text
    assert "bool SelectFirstPositionByOwner(string ownerExpert)" in text
    # The current EA also receives the immutable reference close/ATR used to
    # validate learned geometry.  Match the stable owner/action prefix while
    # tolerating that multiline extension instead of pinning the old literal
    # four-argument signature.
    assert re.search(
        r"void\s+ExecuteAction\s*\(\s*int\s+action,\s*"
        r"double\s+&outputs\[\],\s*double\s+riskPct,\s*"
        r"string\s+ownerExpert(?:\s*,\s*double\s+referenceClose,\s*"
        r"double\s+referenceAtr)?\s*\)",
        text,
    )
    assert "ManageSelectedPosition(outputs, ownerExpert)" in text
    assert "CloseRankedPosition(false, ownerExpert)" in text
    assert "CloseRankedPosition(true, ownerExpert)" in text
    assert "CloseAll(ownerExpert)" in text
    assert "if(!PositionMatchesOwner(ownerExpert))" in text
    assert re.search(
        r"ExecuteAction\(row\.action,\s*row\.outputs,\s*resolvedRiskPct,\s*"
        r"row\.expertId(?:,\s*referenceClose,\s*referenceAtr)?\)",
        text,
    )


def test_execution_credit_v10_does_not_attribute_broker_exit_to_active_intent() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    text = source.read_text(encoding="utf-8", errors="strict")

    helper = text[text.index("bool IsBrokerManagedExit("):text.index("bool IsAcceptedRetcode(")]
    transaction = text[text.index("void OnTradeTransaction("):text.index("int currentBar")]
    assert "DEAL_REASON_SL" in helper
    assert "DEAL_REASON_TP" in helper
    assert "DEAL_REASON_SO" in helper
    assert "bool brokerManagedExit" in transaction
    assert "? -1" in transaction
    assert transaction.index("bool brokerManagedExit") < transaction.index("ResolveIntent(")
    assert 'traceId == "" && g_executionEvents[slot].ownerTraceId' not in text


def test_pending_transition_feedback_is_live_for_non_flat_observations() -> None:
    assert _pending_transition_feedback_allowed(
        flat_idle=False,
        learning_enabled=True,
        has_observer=True,
        trace_learning_blocked=False,
    ) is True
    assert _pending_transition_feedback_allowed(
        flat_idle=True,
        learning_enabled=True,
        has_observer=True,
        trace_learning_blocked=False,
    ) is False


def test_direct_rnn_actor_bridge_firebreak_requires_emitted_action_match() -> None:
    sampled_entry = {
        "direct_rnn_rl_policy_eligible": True,
        "direct_rnn_rl_policy_action_index": int(Action.BUY_STOP),
    }

    matched = _direct_rnn_actor_bridge_audit(sampled_entry, int(Action.BUY_STOP))
    assert matched["direct_rnn_rl_policy_eligible"] is True
    assert matched["direct_rnn_rl_emitted_action_match"] is True
    assert matched["direct_rnn_rl_policy_eligibility_reason"] == "emitted_action_match"

    facade_management = _direct_rnn_actor_bridge_audit(sampled_entry, int(Action.CANCEL_ORDERS))
    assert facade_management["direct_rnn_rl_policy_eligible"] is False
    assert facade_management["direct_rnn_rl_emitted_action_match"] is False
    assert facade_management["direct_rnn_rl_policy_eligibility_reason"] == "emitted_action_changed"

    locally_blocked = _direct_rnn_actor_bridge_audit(
        {
            "direct_rnn_rl_policy_eligible": False,
            "direct_rnn_rl_policy_action_index": int(Action.BUY_STOP),
        },
        int(Action.BUY_STOP),
    )
    assert locally_blocked["direct_rnn_rl_policy_eligible"] is False
    assert locally_blocked["direct_rnn_rl_policy_eligibility_reason"] == "local_policy_ineligible"
    assert _direct_rnn_actor_bridge_audit({}, int(Action.HOLD)) == {}


def test_close_record_routes_entry_owner_credit_without_losing_exit_trace() -> None:
    details = _execution_record_close_details(
        {
            "ack_sequence": "41",
            "event_id": "exit-event",
            "trace_id": "exit-trace",
            "position_owner_trace_id": "entry-trace",
            "position_ticket": "700",
            "accepted": "1",
            "closed": "1",
            "close_reason": "EXPERT",
            "exit_kind": "POLICY_CLOSE_BEST",
            "exit_controller": "policy",
            "stop_controller": "policy_manage",
            "stop_reason": "POLICY_MODIFY_SLTP",
            "initial_sl_price": "91.5",
            "final_sl_price": "104.0",
            "profit": "-10",
            "initial_risk": "100",
        },
        requested_action=int(Action.MARKET_BUY),
        owner_ack_sequence=12,
    )

    assert details["ack_sequence"] == "12"
    assert details["ledger_ack_sequence"] == "41"
    assert details["intent_trace_id"] == "exit-trace"
    assert details["position_owner_trace_id"] == "entry-trace"
    assert details["action_feedback_eligible"] is True
    assert details["exit_kind"] == "POLICY_CLOSE_BEST"
    assert details["exit_controller"] == "policy"
    assert details["stop_reason"] == "POLICY_MODIFY_SLTP"
    assert details["initial_sl_price"] == "91.5"


def test_execution_ledger_v9_keeps_entry_trace_as_lifecycle_owner(tmp_path) -> None:
    ledger = tmp_path / "execution.tsv"
    header = (
        "schema_version\tlogged_at\tack_sequence\tevent_id\ttrace_id\trequested_action\tsource\t"
        "transaction_type\tstatus\taccepted\tfilled\tclosed\tposition_ticket\t"
        "intent_context\tintent_expert\tposition_owner_trace_id\tposition_owner_context\t"
        "position_owner_expert\tdeal_entry\tclose_reason\tprofit\tcommission\tswap\t"
        "position_final_close\tinitial_risk\tage_bars\n"
    )
    entry = (
        "9\t2025.04.01 00:01:00\t12\tentry-event\tentry-trace\t1\tpolicy\tDEAL_ADD\tFILLED\t"
        "1\t1\t0\t700\tCTX_TANG_MANH\tCG_TANG\tentry-trace\tCTX_TANG_MANH\tCG_TANG\t0\t\t0\t0\t0\t0\t100\t0\n"
    )
    close = (
        "9\t2025.04.01 00:15:00\t41\texit-event\texit-trace\t9\tpolicy\tDEAL_ADD\tCLOSED\t"
        "1\t0\t1\t700\tCTX_VOLATILE\tCG_VOL\tentry-trace\tCTX_TANG_MANH\tCG_TANG\t1\tEXPERT\t-10\t0\t0\t1\t100\t14\n"
    )
    ledger.write_text(header + entry + close, encoding="utf-8")

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    assert summary.schema_version == 9
    assert summary.lifecycle_count == 1
    assert summary.lifecycles[0].trace_id == "entry-trace"
    assert summary.lifecycles[0].metadata["position_owner_trace_id"] == "entry-trace"
    assert summary.lifecycles[0].metadata["intent_context"] == "CTX_VOLATILE"


def test_entry_timing_extrema_memory_is_soft_evidence_without_action_override() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=6, seed=123)
    obs = np.linspace(-1.0, 1.0, INPUT_COUNT)
    policy.position_context_extrema_memory = [{
        "position_ticket": "777",
        "side": 1,
        "timing_confirmed": True,
        "recovery_after_low_r": 0.4,
        "low": {
            "unrealized_r": -0.8,
            "observation": obs.tolist(),
        },
    }]
    action = np.zeros(20)
    action[1] = 1.0
    trace = StepTrace(
        trace_id="trace-entry-timing",
        timestamp=0.0,
        raw_obs=obs.copy(),
        encoded_state=obs.copy(),
        memory_before=np.zeros(1),
        memory_after=np.zeros(1),
        selected_idx=0,
        selected_action=action,
        candidate_actions=np.zeros((1, 20)),
        candidate_logits=np.zeros(1),
        candidate_scores=np.zeros(1),
        selected_action_log_prob=0.0,
        predicted_next_state=obs.copy(),
        critic_value=0.0,
        confidence=0.5,
        novelty=0.0,
        conflict=0.0,
        risk=0.0,
        gate_values=(),
        prediction_error=0.0,
        td_delta=0.0,
    )

    adjusted = _apply_entry_timing_extrema_bias(policy, trace)

    assert int(np.argmax(adjusted.selected_action[:12])) == 1
    assert np.array_equal(adjusted.selected_action, action)
    assert adjusted.action_details["entry_timing_extrema_evidence_available"] is True
    assert adjusted.action_details["entry_timing_action_override_applied"] is False
    assert adjusted.action_details["entry_timing_original_action"] == 1
    assert adjusted.action_details["entry_timing_preferred_action"] == 3


def test_entry_timing_keeps_all_entry_directions_legal_in_context_contract() -> None:
    all_entries = [0, 1, 2, 3, 4, 5, 6, 11]
    up = SimpleNamespace(context_audit={"context_allowed_actions": all_entries})
    down = SimpleNamespace(context_audit={"context_allowed_actions": all_entries})
    executable = {int(action) for action in Action}

    up_actions = MT5StrategyTesterEvaluator._effective_context_actions(up, executable)
    down_actions = MT5StrategyTesterEvaluator._effective_context_actions(down, executable)

    assert int(Action.BUY_LIMIT) in up_actions
    assert int(Action.SELL_LIMIT) in down_actions
    assert int(Action.MARKET_SELL) in up_actions
    assert int(Action.MARKET_BUY) in down_actions


def test_context_facade_applies_owner_entry_timing_extrema_bias() -> None:
    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(bat=True),
        tao_policy=lambda: ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=6, seed=125),
    )
    expert = system.registry.chuyen_gia_theo_context["CTX_TANG_MANH"]
    obs = np.linspace(1.0, -1.0, INPUT_COUNT)
    expert.policy.position_context_extrema_memory = [{
        "position_ticket": "888",
        "side": -1,
        "timing_confirmed": True,
        "recovery_after_low_r": 0.5,
        "low": {
            "unrealized_r": -0.9,
            "observation": obs.tolist(),
        },
    }]
    action = np.zeros(20)
    action[2] = 1.0
    trace = StepTrace(
        trace_id="trace-facade-entry-timing",
        timestamp=0.0,
        raw_obs=obs.copy(),
        encoded_state=obs.copy(),
        memory_before=np.zeros(1),
        memory_after=np.zeros(1),
        selected_idx=0,
        selected_action=action,
        candidate_actions=np.zeros((1, 20)),
        candidate_logits=np.zeros(1),
        candidate_scores=np.zeros(1),
        selected_action_log_prob=0.0,
        predicted_next_state=obs.copy(),
        critic_value=0.0,
        confidence=0.5,
        novelty=0.0,
        conflict=0.0,
        risk=0.0,
        gate_values=(),
        prediction_error=0.0,
        td_delta=0.0,
        context_audit={
            "ma_context": "CTX_TANG_MANH",
            "ma_chuyen_gia": expert.ma_chuyen_gia,
            "ma_policy": expert.ma_chuyen_gia,
        },
    )

    adjusted = system.apply_entry_timing_extrema_bias(trace)

    assert int(np.argmax(adjusted.selected_action[:12])) == int(Action.MARKET_SELL)
    assert adjusted.action_details["entry_timing_extrema_evidence_available"] is True
    assert adjusted.action_details["entry_timing_action_override_applied"] is False
    assert adjusted.action_details["entry_timing_original_action"] == int(Action.MARKET_SELL)
    assert adjusted.action_details["entry_timing_preferred_action"] == int(Action.SELL_LIMIT)


def test_intraday_reset_clears_short_term_timing_state_without_deleting_policy() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=6, seed=124)
    timing_memory = {"position_ticket": "1", "side": 1, "low": {"unrealized_r": -0.6}}
    policy.position_context_extrema_memory = [timing_memory]
    policy._decision_memory_query_cache = {"key": "old", "value": {"action_bias": []}}
    policy._account_context = {"day_trade_pressure": 0.8}
    policy._positions_pipe_context = "1:1:0.1:100:99:103:0:1"

    policy.reset_intraday_state("2025.04.02")

    assert policy.position_context_extrema_memory == [timing_memory]
    assert policy._decision_memory_query_cache is None
    assert policy._account_context == {}
    assert policy._positions_pipe_context == ""
    assert policy._last_intraday_reset_day == "2025.04.02"


def test_latest_runtime_file_ignores_stale_agent_bridge_files(tmp_path) -> None:
    runner = tmp_path / "runner"
    stale_dir = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    fresh_dir = runner / "MQL5" / "Files"
    stale_dir.mkdir(parents=True)
    fresh_dir.mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    stale = stale_dir / "self_growing_request.csv"
    fresh = fresh_dir / "self_growing_request.csv"
    stale.write_text("stale", encoding="utf-8")
    fresh.write_text("fresh", encoding="utf-8")
    os.utime(stale, ns=(1_000, 1_000))
    os.utime(fresh, ns=(3_000, 3_000))
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    evaluator._runtime_file_min_mtime_ns = 2_000

    assert evaluator._latest_runtime_file("self_growing_request.csv") == fresh


def test_mt5_runner_fails_closed_without_ea_artifacts(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    with pytest.raises(RuntimeError, match="ea_source_missing,ea_binary_missing"):
        evaluator._verify_ea_build_integrity()


def test_request_normalization_preserves_raw_regime_input_view(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            input_normalization_mode="structure_aware_v4",
        )
    )
    raw = np.zeros(137, dtype=float)
    raw[84], raw[85], raw[86], raw[87], raw[90] = 2.50, 1.00, 3.90, 0.0, 7.80

    normalized = evaluator._normalize_request_inputs(
        BridgeRequest(sequence=12, inputs=raw)
    )

    assert normalized.raw_inputs is not None
    assert np.array_equal(normalized.raw_inputs, raw)
    assert not np.shares_memory(normalized.raw_inputs, normalized.inputs)
    assert not np.array_equal(normalized.inputs, raw)
    assert np.allclose(
        normalized.inputs,
        normalize_input_vector(raw, mode="structure_aware_v4", size=INPUT_COUNT),
    )


def test_action_metadata_sidecar_records_context_owner_for_audit(tmp_path) -> None:
    action = tmp_path / "self_growing_action.csv"
    request = BridgeRequest(sequence=12, event_id="evt-12", inputs=np.zeros(2))
    trace = SimpleNamespace(
        trace_id="trace-owner",
        action_details={
            "hold_reason": "HOLD_LOCAL_REAL_NEGATIVE_VETO",
            "expectancy_action_delta_applied": [0.0, 0.0, -0.4],
            "expectancy_action_delta_norm": 0.4,
            "learner_recovery_negative_veto_applied": True,
            "learner_recovery_negative_veto_from": 5,
            "learner_recovery_base_action": 5,
            "learner_recovery_base_real_support": 3.0,
            "learner_recovery_base_lcb": -0.25,
            "training_entry_probe_source": "not_emitted_policy_veto",
        },
        context_audit={
            "ma_context": "CTX_GOC_VAO_LENH_KHONG_CO_LENH",
            "context_nhan_dien": "CTX_GOC_VAO_LENH_KHONG_CO_LENH",
            "ma_chuyen_gia": "CG_GOC_ENTRY",
            "ma_policy": "CG_GOC_ENTRY",
            "phien_ban_policy": "V001",
            "trang_thai_chuyen_gia": "HOAT_DONG_HAN_CHE",
            "ly_do_context": "stable",
            "context_action_contract": "OWNER_MANAGEMENT_ONLY_V11",
            "management_only": True,
            "owner_management_guard_reason": "NO_PROFIT_FOR_CLOSE_BEST",
        },
    )

    MT5StrategyTesterEvaluator._write_action_metadata(action, request, trace)

    assert not action.exists()
    sidecar = action.with_name(action.name + ".metadata.jsonl")
    rows = [json.loads(line) for line in sidecar.read_text(encoding="utf-8").splitlines()]
    assert rows == [{
        "schema_version": BRIDGE_SCHEMA_VERSION,
        "event_id": "evt-12",
        "sequence": 12,
        "trace_id": "trace-owner",
        "ma_context": "CTX_GOC_VAO_LENH_KHONG_CO_LENH",
        "context_nhan_dien": "CTX_GOC_VAO_LENH_KHONG_CO_LENH",
        "ma_chuyen_gia": "CG_GOC_ENTRY",
        "ma_policy": "CG_GOC_ENTRY",
        "phien_ban_policy": "V001",
        "trang_thai_chuyen_gia": "HOAT_DONG_HAN_CHE",
        "ly_do_context": "stable",
        "khong_co_policy_ra_lenh": False,
        "ly_do_khong_giao_dich": "",
        "context_action_contract": "OWNER_MANAGEMENT_ONLY_V11",
        "management_only": True,
        "owner_management_guard_reason": "NO_PROFIT_FOR_CLOSE_BEST",
        "hold_reason": "HOLD_LOCAL_REAL_NEGATIVE_VETO",
        "expectancy_action_delta_applied": [0.0, 0.0, -0.4],
        "expectancy_action_delta_norm": 0.4,
        "learner_recovery_negative_veto_applied": True,
        "learner_recovery_negative_veto_from": 5,
        "learner_recovery_base_action": 5,
        "learner_recovery_base_real_support": 3.0,
        "learner_recovery_base_lcb": -0.25,
        "training_entry_probe_source": "not_emitted_policy_veto",
    }]


def test_no_policy_context_trace_is_blocked_from_learning_and_journal(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, experience_journal=journal))
    request = BridgeRequest(sequence=1, event_id="evt-no-policy", inputs=np.zeros(2))
    trace = SimpleNamespace(
        trace_id="khong-policy-1",
        ly_do_khong_hoc=True,
        context_audit={"khong_co_policy_ra_lenh": True, "ma_context": "CTX_GOC_VAO_LENH_KHONG_CO_LENH"},
        execution_details={"bridge_observed": True, "memory_eligible": True},
        selected_action=np.zeros(20),
        action_details={"action_attribution_source": "none"},
    )

    assert evaluator._trace_learning_blocked(trace) is True
    evaluator._append_experience_journal(
        round_index=0,
        policy_id=1,
        request=request,
        trace=trace,
        next_obs=np.zeros(2),
        reward=1.0,
        cost=0.0,
        done=False,
        signal={"td_delta": 1.0, "memory_eligible": 1.0},
        action=0,
        score=0.0,
        confidence=0.0,
    )

    assert not journal.exists()


def test_experience_journal_serializes_nested_numpy_action_metadata(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    journal = tmp_path / "journal.jsonl"
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, experience_journal=journal))
    request = BridgeRequest(sequence=1, event_id="evt-array", inputs=np.zeros(2))
    trace = SimpleNamespace(
        trace_id="trace-array",
        ly_do_khong_hoc=False,
        context_audit={"khong_co_policy_ra_lenh": False},
        execution_details={"bridge_observed": True, "memory_eligible": True},
        selected_action=np.zeros(20),
        action_details={"five_cell_before": np.asarray([[1.0, 2.0]], dtype=np.float32)},
    )

    evaluator._append_experience_journal(
        round_index=0, policy_id=1, request=request, trace=trace,
        next_obs=np.zeros(2), reward=1.0, cost=0.0, done=False,
        signal={"td_delta": 1.0, "memory_eligible": 1.0},
        action=1, score=1.0, confidence=1.0,
    )

    row = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert row["action_details"]["five_cell_before"] == [[1.0, 2.0]]


def test_read_versioned_bridge_request_keeps_position_snapshot(tmp_path) -> None:
    request = tmp_path / "self_growing_request.csv"
    request.write_text(
        "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\tpositions_pipe\n"
        "2\tevt-8\t8\t2025.01.01\t10000\t10005\t0.1\t10\t1\t4\t0.4|0.5\t123:1:0.01:100:99:103:1.2:5\n",
        encoding="utf-8",
    )

    parsed = MT5StrategyTesterEvaluator._read_request(request)

    assert parsed is not None
    assert parsed.inputs.tolist() == [0.4, 0.5]
    assert parsed.positions_pipe.startswith("123:1")


def test_experience_starvation_tracks_no_trade_and_unfilled_pressure_after_execution_confirmation() -> None:
    inputs = np.zeros(119, dtype=float)
    inputs[56 + 5 * 8 + 6] = 0.5
    request = BridgeRequest(sequence=1, inputs=inputs, day_trades=0, positions=0)
    state = ExperienceStarvationState()

    first = state.preview(request, min_daily_trades=6)
    # V25: emitting an action is not evidence that it failed to fill. The
    # snapshot is read-only; update starvation only after the next MT5 request
    # can confirm the outcome of the preceding intent.
    preview = state.snapshot(request, action=5, trades_before=0, trades_after=0, min_daily_trades=6)
    assert first["cadence_deficit"] == 0.5
    assert preview["no_trade_streak"] == 0.0
    assert preview["unfilled_streak"] == 0.0

    after = BridgeRequest(sequence=2, inputs=inputs, day_trades=0, positions=0)
    assert state.confirm_execution(request, after, emitted_action=5) is False
    confirmed = state.preview(after, min_daily_trades=6)
    assert confirmed["no_trade_streak"] == 1.0
    assert confirmed["unfilled_streak"] == 1.0
    assert confirmed["starvation"] > first["starvation"]
    assert confirmed["intrinsic_reward"] == 0.0
    assert confirmed["exploration_pressure"] > first["exploration_pressure"]


def test_parse_mt5_report_reads_strategy_tester_metrics(tmp_path) -> None:
    report = tmp_path / "report.htm"
    report.write_text(
        "Total Net Profit: 505.96 Profit Factor: 1.41 Total Trades: 1046 "
        "Balance Drawdown Maximal: 379.00 (3.79%) "
        "Equity Drawdown Maximal: 421.00 (4.21%) "
        "Profit Trades (% of total): 840 (80.31%) "
        "Average profit trade: 2.10 "
        "Average loss trade: -1.00",
        encoding="utf-16",
    )

    metrics = parse_mt5_report(report)

    assert metrics.net_profit == 505.96
    assert metrics.profit_factor == 1.41
    assert metrics.total_trades == 1046
    assert metrics.balance_drawdown_pct == 3.79
    assert metrics.equity_drawdown_pct == 4.21
    assert metrics.win_rate == 80.31
    assert metrics.avg_win == 2.10
    assert metrics.avg_loss == -1.00
    assert mt5_fitness(metrics) > 0.0


def test_parse_mt5_report_preserves_no_loss_metrics_but_fitness_requires_sample_size(tmp_path) -> None:
    report = tmp_path / "report.htm"
    report.write_text(
        "Total Net Profit: 15.42 Profit Factor: 0.00 Expected Payoff: 5.14 Total Trades: 3 "
        "Balance Drawdown Maximal: 0.00 (0.00%) "
        "Equity Drawdown Maximal: 15.45 (0.15%) "
        "Profit Trades (% of total): 3 (100.00%) "
        "Loss Trades (% of total): 0 (0.00%) "
        "Average profit trade: 5.14 "
        "Average loss trade: 0.00",
        encoding="utf-16",
    )

    metrics = parse_mt5_report(report)

    assert metrics.profit_factor == 999.0
    assert metrics.win_rate == 100.0
    assert compute_mt5_expectancy(metrics) == 5.14
    assert mt5_fitness(metrics) < 0.0
    assert mt5_fitness(metrics, backtest_days=1, min_daily_trades=0) < 0.0


def test_mt5_fitness_penalizes_low_sample_no_loss_runs(tmp_path) -> None:
    report = tmp_path / "report.htm"
    report.write_text(
        "Total Net Profit: 2.43 Profit Factor: 0.00 Total Trades: 1 "
        "Balance Drawdown Maximal: 0.00 (0.00%) "
        "Equity Drawdown Maximal: 2.00 (0.02%) "
        "Profit Trades (% of total): 1 (100.00%) "
        "Loss Trades (% of total): 0 (0.00%) "
        "Average profit trade: 2.43 "
        "Average loss trade: 0.00",
        encoding="utf-16",
    )

    metrics = parse_mt5_report(report)

    assert metrics.profit_factor == 999.0
    assert mt5_fitness(metrics, backtest_days=2, min_daily_trades=7, max_daily_trades=20) < 0.0


def test_mt5_structural_j_rewards_real_window_quality(tmp_path) -> None:
    report = tmp_path / "report.htm"
    report.write_text(
        "Total Net Profit: 70.83 Profit Factor: 1.46 Total Trades: 179 "
        "Balance Drawdown Maximal: 48.00 (0.48%) "
        "Equity Drawdown Maximal: 67.00 (0.67%) "
        "Profit Trades (% of total): 76 (42.46%) "
        "Average profit trade: 2.96 "
        "Average loss trade: -1.50",
        encoding="utf-16",
    )

    metrics = parse_mt5_report(report)

    assert mt5_structural_j(metrics, backtest_days=2, min_daily_trades=7, max_daily_trades=100) > 0.0


def test_write_tester_ini_uses_unique_report_name(tmp_path) -> None:
    ini = write_tester_ini(MT5BacktestConfig(), "self_growing_mt5_train_round_0001_policy_2", tmp_path / "run.ini")

    text = ini.read_text(encoding="utf-8")
    assert "Expert=SelfGrowing_MT5_Trainer_EA" in text
    assert "ExpertParameters=SelfGrowing_MT5_Trainer_EA.set" in text
    assert "Symbol=BTCUSDm" in text
    assert "FromDate=2025.01.01" in text
    assert "Report=Reports\\self_growing_mt5_train_round_0001_policy_2" in text
    assert "ShutdownTerminal=1" in text
    assert "Visual=0" in text
    assert "Model=0" in text
    assert "ExecutionMode=0" in text


def test_materialize_report_copies_known_mt5_report_variant(tmp_path) -> None:
    runner = tmp_path / "runner"
    (runner / "reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    source = runner / "reports" / "self_growing_mt5_train_round_0001_policy_2.html"
    source.write_text(
        "Total Net Profit: 1.00 Profit Factor: 1.00 Total Trades: 1 "
        "Balance Drawdown Maximal: 0.00 (0.00%) Equity Drawdown Maximal: 0.00 (0.00%) "
        "Profit Trades (% of total): 1 (100.00%) Average profit trade: 1.00 Average loss trade: 0.00",
        encoding="utf-16",
    )
    expected = runner / "Reports" / "self_growing_mt5_train_round_0001_policy_2.htm"
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    materialized = evaluator._materialize_report("self_growing_mt5_train_round_0001_policy_2", expected)

    assert materialized == expected
    assert expected.exists()
    assert parse_mt5_report(expected).total_trades == 1


def test_materialize_report_ignores_reports_older_than_run_gate(tmp_path) -> None:
    runner = tmp_path / "runner"
    (runner / "reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    source = runner / "reports" / "he_chuyen_gia_mt5_round_0000_evaluation_1.html"
    source.write_text("old report", encoding="utf-16")
    old_ns = source.stat().st_mtime_ns
    expected = runner / "Reports" / "he_chuyen_gia_mt5_round_0000_evaluation_1.htm"
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    evaluator._report_min_mtime_ns = old_ns + 1_000_000

    assert evaluator._materialize_report("he_chuyen_gia_mt5_round_0000_evaluation_1", expected) is None

    source.write_text("new report", encoding="utf-16")
    evaluator._report_min_mtime_ns = source.stat().st_mtime_ns - 1

    assert evaluator._materialize_report("he_chuyen_gia_mt5_round_0000_evaluation_1", expected) == expected


def test_strategy_tester_evaluator_writes_session_manifest_hashes(tmp_path) -> None:
    runner = tmp_path / "runner"
    (runner / "MQL5" / "Experts").mkdir(parents=True)
    (runner / "MQL5" / "Profiles" / "Tester").mkdir(parents=True)
    (runner / "MQL5" / "Files" / "self_growing_mt5_train").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    terminal = runner / "terminal64.exe"
    source = runner / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    binary = runner / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.ex5"
    set_path = runner / "MQL5" / "Profiles" / "Tester" / "SelfGrowing_MT5_Trainer_EA.set"
    ini_path = runner / "MQL5" / "Files" / "self_growing_mt5_train" / "round_0000_policy_1.ini"
    report_path = runner / "Reports" / "self_growing_mt5_train_round_0000_policy_1.htm"
    terminal.write_text("terminal", encoding="utf-8")
    source.write_text("// ea source", encoding="utf-8")
    binary.write_text("compiled ex5", encoding="utf-8")
    set_path.write_text("set file", encoding="utf-8")
    ini_path.write_text("ini file", encoding="utf-8")
    report_path.write_text("report", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, feature_schema_hash="feature-hash")
    )
    policy = SimpleNamespace(name="policy", weights=[1.0, 2.0])

    manifest_path = evaluator._write_session_manifest(
        report_name="self_growing_mt5_train_round_0000_policy_1",
        policy_id=1,
        round_index=0,
        ini_path=ini_path,
        set_path=set_path,
        report_path=report_path,
        policy=policy,
        stage="finished",
        extra={"served_actions": 3},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_path == runner / "Reports" / "self_growing_mt5_train_round_0000_policy_1.manifest.json"
    assert manifest["schema_version"] == 1
    assert manifest["stage"] == "finished"
    assert manifest["feature_schema_hash"] == "feature-hash"
    assert manifest["ea_source_sha256"] == sha256_file(source)
    assert manifest["ea_binary_sha256"] == sha256_file(binary)
    assert manifest["terminal_sha256"] == sha256_file(terminal)
    assert manifest["tester_ini_sha256"] == sha256_file(ini_path)
    assert manifest["tester_set_sha256"] == sha256_file(set_path)
    assert manifest["report_sha256"] == sha256_file(report_path)
    assert isinstance(manifest["python_tree_sha256"], str)
    assert isinstance(manifest["protocol_contract_sha256"], str)
    assert isinstance(manifest["policy_state_sha256"], str)
    assert manifest["extra"]["served_actions"] == 3


def test_strategy_tester_evaluator_fail_closes_when_ea_artifacts_are_missing(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir(parents=True)
    (runner / "terminal64.exe").write_text("terminal", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    with pytest.raises(RuntimeError, match="ea_source_missing,ea_binary_missing"):
        evaluator._verify_ea_build_integrity()


def test_strategy_tester_evaluator_fail_closes_on_stale_or_mismatched_ea_build(tmp_path, monkeypatch) -> None:
    runner = tmp_path / "runner"
    experts = runner / "MQL5" / "Experts"
    experts.mkdir(parents=True)
    terminal = runner / "terminal64.exe"
    source = experts / "SelfGrowing_MT5_Trainer_EA.mq5"
    binary = experts / "SelfGrowing_MT5_Trainer_EA.ex5"
    terminal.write_text("terminal", encoding="utf-8")
    source.write_text("// source v2", encoding="utf-8")
    binary.write_text("compiled old", encoding="utf-8")
    os.utime(binary, (1_700_000_000, 1_700_000_000))
    os.utime(source, (1_700_000_100, 1_700_000_100))
    monkeypatch.setattr("self_growing_trader.mt5_runner.compile_mql5_ea", lambda *args, **kwargs: None)
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    with pytest.raises(RuntimeError, match="ea_binary_older_than_source"):
        evaluator._verify_ea_build_integrity()

    os.utime(binary, (1_700_000_200, 1_700_000_200))
    expected_source_hash = sha256_file(source)
    expected_binary_hash = sha256_file(binary)
    pinned = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=runner,
        expected_ea_source_sha256=expected_source_hash or "",
        expected_ea_binary_sha256=expected_binary_hash or "",
    ))
    assert pinned._verify_ea_build_integrity()["binary_sha256"] == expected_binary_hash

    mismatched = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=runner,
        expected_ea_source_sha256="0" * 64,
        expected_ea_binary_sha256=expected_binary_hash or "",
    ))
    with pytest.raises(RuntimeError, match="ea_source_hash_mismatch"):
        mismatched._verify_ea_build_integrity()


def test_write_tester_set_matches_current_ea_inputs(tmp_path) -> None:
    config = MT5BacktestConfig(
        min_daily_trades=12, max_daily_trades=60, max_decision_packets=256
    )
    set_file = write_tester_set(config, tmp_path / "SelfGrowing_MT5_Trainer_EA.set")

    text = set_file.read_text(encoding="utf-8")
    assert "BridgeFile=self_growing_bridge.csv" in text
    assert "RequestFile=self_growing_request.csv" in text
    assert "ActionFile=self_growing_action.csv" in text
    assert "FeatureSchemaHash=" in text
    assert "FeatureMtfScalingMode=" in text
    assert "FeatureAtrPeriod=14||14||1||140||N" in text
    assert "FeatureEmaFast=20||20||1||200||N" in text
    assert "FeatureEmaSlow=60||60||1||600||N" in text
    assert "MinDailyTrades=0||0||0||200||N" in text
    assert "MaxDailyTrades=0||0||0||500||N" in text
    assert "MaxDecisionPackets=256||256||0||1000000||N" in text
    for removed_input in (
        "RunId=",
        "StrategyMinRR=",
        "MinManualCloseAgeBars=",
        "InitialEquity=",
        "MinRiskPct=",
        "MaxRiskPct=",
        "MaxOpenRiskPct=",
        "RuinEquityPct=",
        "EnforceEntryLimits=",
        "FixedRiskPct=",
        "MaxSpreadAtr=",
        "MaxSpreadCostR=",
        "ResetExecutionLedgerOnInit=",
        "RequireHedgingAccount=",
        "LearningEnabled=",
    ):
        assert removed_input not in text

    root = Path(__file__).resolve().parents[1]
    ea_source = (
        root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8")
    ea_source += "\n" + (
        root / "mt5_v2_runner" / "MQL5" / "Include" / "compute_features.mqh"
    ).read_text(encoding="utf-8")
    declared_inputs = set(
        re.findall(r"(?m)^\s*input\s+\w+\s+(\w+)\s*=", ea_source)
    )
    serialized_inputs = {
        line.split("=", 1)[0]
        for line in text.splitlines()
        if "=" in line and line.strip() and not line.lstrip().startswith("#")
    }
    assert serialized_inputs <= declared_inputs


def test_mt5_config_rejects_negative_decision_packet_cap() -> None:
    with pytest.raises(ValueError, match="max_decision_packets must be non-negative"):
        MT5BacktestConfig(max_decision_packets=-1)


def test_ea_bridge_v10_exports_raw_features_without_mql_normalizer() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root
        / "mt5_v2_runner"
        / "MQL5"
        / "Experts"
        / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8", errors="strict")

    assert f"#define BRIDGE_SCHEMA_VERSION {BRIDGE_SCHEMA_VERSION}" in source
    for removed_symbol in (
        "InputNormalizationMode",
        "NormalizeInputVector",
        "SgBoundedTanh",
        "SgSoftClip",
        "SgAsinh",
        "SgPositiveAtrDistance",
        "SgSignedAtr",
        "SgVolatilityRatio",
    ):
        assert removed_symbol not in source


def test_strategy_tester_evaluator_forces_terminal_close_after_trial() -> None:
    class HangingProcess:
        def __init__(self):
            self.waits = 0
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired("terminal64.exe", timeout)
            if self.waits == 2:
                raise subprocess.TimeoutExpired("terminal64.exe", timeout)
            return 0

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    process = HangingProcess()

    MT5StrategyTesterEvaluator._close_terminal_after_trial(process)

    assert process.terminated is True
    assert process.killed is True
    assert process.waits == 3


def test_strategy_tester_evaluator_cleans_runner_processes_on_bridge_exception(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    (runner / "MQL5" / "Files").mkdir(parents=True)
    (runner / "MQL5" / "Profiles" / "Tester").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")

    class FakeProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

        def kill(self):
            raise AssertionError("terminate path should be enough for this test")

    fake_process = FakeProcess()
    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", lambda *args, **kwargs: fake_process)
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, timeout_sec=1, shutdown_terminal=True, experience_journal=None)
    )
    cleaned = {"called": False}
    monkeypatch.setattr(evaluator, "_serve_policy_actions", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("wall guard")))
    monkeypatch.setattr(evaluator, "close_runner_processes", lambda: cleaned.update(called=True))

    with pytest.raises(TimeoutError, match="wall guard"):
        evaluator.evaluate_policy(7, _context_policy(), round=0)

    assert fake_process.terminated is True
    assert cleaned["called"] is True


def test_keep_open_bridge_timeout_stops_tester_worker_without_closing_terminal(monkeypatch, tmp_path) -> None:
    """A bounded keep-open run must not leak a metatester worker into the next arm."""
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    (runner / "MQL5" / "Files").mkdir(parents=True)
    (runner / "MQL5" / "Profiles" / "Tester").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")

    class FakeProcess:
        terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            return 0

    fake_process = FakeProcess()
    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", lambda *args, **kwargs: fake_process)
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, timeout_sec=1, shutdown_terminal=False, experience_journal=None)
    )
    cleaned = {"tester": False, "terminal": False}
    monkeypatch.setattr(evaluator, "_serve_policy_actions", lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("wall guard")))
    monkeypatch.setattr(evaluator, "_close_stale_tester_agents", lambda: cleaned.update(tester=True))
    monkeypatch.setattr(evaluator, "close_runner_processes", lambda: cleaned.update(terminal=True))

    with pytest.raises(TimeoutError, match="wall guard"):
        evaluator.evaluate_policy(7, _context_policy(), round=0)

    assert fake_process.terminated is True
    assert cleaned == {"tester": True, "terminal": False}


def test_count_bridge_rows_ignores_header(tmp_path) -> None:
    bridge = tmp_path / "bridge.csv"
    bridge.write_text("time,bar\n2025.01.01,1\n2025.01.01,2\n", encoding="utf-8")

    assert count_bridge_rows(bridge) == 2


def test_strategy_tester_evaluator_serves_python_actions_and_requires_report(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    (runner / "Common" / "Files").mkdir(parents=True)
    (runner / "MQL5" / "Files").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    (runner / "Tester" / "cache").mkdir(parents=True)
    stale_cache = runner / "Tester" / "cache" / "SelfGrowing_MT5_Trainer_EA.BTCUSDm.M1.20250101_20250103.0.STALE.tst"
    stale_cache.write_text("stale", encoding="utf-8")
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    calls = []
    agent_files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    agent_files.mkdir(parents=True)

    class FakeProcess:
        def __init__(self, command, cwd):
            calls.append((command, cwd))
            self.done = False
            request = agent_files / "self_growing_request.csv"
            request.write_text("sequence,time,inputs_pipe\n1,2025.01.01,0.1|0.2\n", encoding="utf-8")

        def poll(self):
            action = agent_files / "self_growing_action.csv"
            if not self.done and action.exists() and "1\t" in action.read_text(encoding="utf-8"):
                report = runner / "Reports" / "he_chuyen_gia_mt5_round_0000_evaluation_7.htm"
                report.write_text(
                    "Total Net Profit: 10.00 Profit Factor: 1.20 Total Trades: 3 "
                    "Balance Drawdown Maximal: 1.00 (0.10%) Equity Drawdown Maximal: 2.00 (0.20%)",
                    encoding="utf-16",
                )
                bridge = agent_files / "self_growing_bridge.csv"
                bridge.write_text("time,bar,action\n2025.01.01,1,1\n", encoding="utf-8")
                self.done = True
            return 0 if self.done else None

        def wait(self, timeout=None):
            self.done = True
            return 0

    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", FakeProcess)
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, timeout_sec=12, experience_journal=None))
    monkeypatch.setattr(evaluator, "close_runner_processes", lambda: None)
    policy = _context_policy()
    finalized = []
    original_finalize = policy.finalize_bridge_decision

    def record_final_bridge_trace(trace):
        finalized.append(trace)
        return original_finalize(trace)

    monkeypatch.setattr(policy, "finalize_bridge_decision", record_final_bridge_trace)

    fitness, metrics = evaluator.evaluate_policy(7, policy, round=0)

    assert fitness == -100.0
    assert metrics.execution_summary is not None
    assert metrics.execution_summary.ledger_available is True
    assert metrics.execution_summary.lifecycle_count == 0
    assert metrics.total_trades == 3
    assert metrics.bridge_rows == 1
    assert metrics.bridge_path == runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files" / "self_growing_bridge.csv"
    assert (agent_files / "self_growing_action.csv").exists()
    assert (runner / "MQL5" / "Profiles" / "Tester" / "SelfGrowing_MT5_Trainer_EA.set").exists()
    assert not stale_cache.exists()
    assert calls[0][0][1] == "/portable"
    # The runner must defer facade telemetry until it has attached the
    # immutable semantic/emitted/wire action contract.
    assert policy._defer_live_decision_until_bridge is True
    assert len(finalized) == 1
    final_details = dict(getattr(finalized[0], "action_details", {}) or {})
    assert final_details["emitted_action"] == int(Action.MARKET_BUY)
    assert final_details["wire_action"] == int(Action.MARKET_BUY)


def test_action_bridge_timeout_allows_active_request_progress(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    (runner / "MQL5" / "Files").mkdir(parents=True)
    (runner / "MQL5" / "Profiles" / "Tester").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    agent_files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    agent_files.mkdir(parents=True)
    request = agent_files / "self_growing_request.csv"
    action = agent_files / "self_growing_action.csv"
    report = runner / "Reports" / "he_chuyen_gia_mt5_round_0000_evaluation_8.htm"

    clock = {"now": 0.0}

    def fake_monotonic():
        return clock["now"]

    def fake_sleep(seconds):
        clock["now"] += max(float(seconds), 0.6)

    def write_request(sequence: int) -> None:
        request.write_text(
            f"sequence,time,inputs_pipe\n{sequence},2025.01.01,0.1|0.2\n",
            encoding="utf-8",
        )

    class FakeProcess:
        def __init__(self, command, cwd):
            self.done = False
            self.last_action_text = ""
            self.sequence = 1
            write_request(self.sequence)

        def poll(self):
            if self.done:
                return 0
            if action.exists():
                action_text = action.read_text(encoding="utf-8")
                if action_text != self.last_action_text:
                    self.last_action_text = action_text
                    if self.sequence < 4:
                        self.sequence += 1
                        write_request(self.sequence)
                    else:
                        report.write_text(
                            "Total Net Profit: 10.00 Profit Factor: 1.20 Total Trades: 3 "
                            "Balance Drawdown Maximal: 1.00 (0.10%) Equity Drawdown Maximal: 2.00 (0.20%)",
                            encoding="utf-16",
                        )
                        self.done = True
                        return 0
            return None

        def wait(self, timeout=None):
            self.done = True
            return 0

    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", FakeProcess)
    monkeypatch.setattr("self_growing_trader.mt5_runner.time.monotonic", fake_monotonic)
    monkeypatch.setattr("self_growing_trader.mt5_runner.time.sleep", fake_sleep)
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, timeout_sec=1, experience_journal=None))
    monkeypatch.setattr(evaluator, "close_runner_processes", lambda: None)
    policy = _context_policy()

    _, metrics = evaluator.evaluate_policy(8, policy, round=0)

    assert metrics.total_trades == 3
    assert clock["now"] > 1.0


def test_strategy_tester_evaluator_runs_online_development_loop(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    (runner / "MQL5" / "Files").mkdir(parents=True)
    (runner / "MQL5" / "Profiles" / "Tester").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    agent_files = runner / "Tester" / "Agent-127.0.0.1-3000" / "MQL5" / "Files"
    agent_files.mkdir(parents=True)

    class FakeProcess:
        def __init__(self, command, cwd):
            self.poll_count = 0
            (agent_files / "self_growing_request.csv").write_text(
                "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\tpositions_pipe\n"
                "2\tevt-1\t1\t2025.01.01\t10000\t10000\t0\t10\t0\t0\t0.1|0.2\t\n",
                encoding="utf-8",
            )

        def poll(self):
            self.poll_count += 1
            action = agent_files / "self_growing_action.csv"
            if self.poll_count == 2:
                (agent_files / "self_growing_request.csv").write_text(
                    "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\tpositions_pipe\n"
                    "2\tevt-2\t2\t2025.01.01\t10000\t10010\t0\t10\t1\t1\t0.3|0.4\t123:1:0.01:100:99:103:1.2:5\n",
                    encoding="utf-8",
                )
            if self.poll_count > 3 and action.exists():
                (runner / "Reports" / "he_chuyen_gia_mt5_round_0000_evaluation_3.htm").write_text(
                    "Total Net Profit: 10.00 Profit Factor: 1.20 Total Trades: 3 "
                    "Balance Drawdown Maximal: 1.00 (0.10%) Equity Drawdown Maximal: 2.00 (0.20%)",
                    encoding="utf-16",
                )
                (agent_files / "self_growing_bridge.csv").write_text("time\tbar\taction\n2025.01.01\t1\t1\n", encoding="utf-8")
                return 0
            return None

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", FakeProcess)
    policy = _context_policy()
    journal = tmp_path / "experience.jsonl"
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, timeout_sec=12, experience_journal=str(journal)))
    monkeypatch.setattr(evaluator, "close_runner_processes", lambda: None)

    evaluator.evaluate_policy(3, policy, round=0)

    expert_policy = next(expert.policy for expert in policy.experts.values() if expert.policy.acts)
    assert expert_policy.acts >= 2
    assert expert_policy.observes >= 1
    assert expert_policy.trace_ids_seen[0].endswith("~trace-1")
    assert expert_policy.next_obs_seen is not None
    assert expert_policy.next_obs_seen.tolist() == [0.3, 0.4]

    action_text = (agent_files / "self_growing_action.csv").read_text(encoding="utf-8")
    assert "schema_version\tevent_id\ttrace_id\tsequence\tagent_id\tagent_count\taction\tscore\tconfidence\trisk_pct\toutputs_pipe" in action_text
    action_rows = [line.split("\t") for line in action_text.splitlines()[1:] if line.strip()]
    assert len(action_rows) == 1
    assert action_rows[-1][3] == "2"
    assert action_rows[-1][4:7] == ["0", "1", "1"]
    assert "~trace-" in action_rows[-1][2]
    # A bridge action without a finalised lifecycle must not create a synthetic
    # positive journal row.
    journal_rows = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert journal_rows
    assert all(float(row["reward"]) == 0.0 for row in journal_rows)
    assert all(row["is_positive"] is False for row in journal_rows)



def test_utf16_report_fallback_recovers_trace_owned_lifecycles(tmp_path) -> None:
    report = tmp_path / "tester.htm"
    report.write_text(
        """<html><body>
        <table><tr><th>Open Time</th><th>Order</th><th>Type</th><th>Volume</th><th>Price</th><th>S / L</th><th>T / P</th><th>Comment</th></tr>
        <tr><td>2025.01.01 00:00:00</td><td>101</td><td>buy limit</td><td>1.00 / 1.00</td><td>100.0</td><td>90.0</td><td>120.0</td><td>SG5:7:4</td></tr></table>
        <table><tr><th>Time</th><th>Deal</th><th>Type</th><th>Direction</th><th>Volume</th><th>Price</th><th>Order</th><th>Commission</th><th>Swap</th><th>Profit</th><th>Comment</th></tr>
        <tr><td>2025.01.01 00:01:00</td><td>201</td><td>buy</td><td>in</td><td>1.00</td><td>100.0</td><td>101</td><td>0</td><td>0</td><td>0</td><td>SG5:7:4</td></tr>
        <tr><td>2025.01.01 00:05:00</td><td>202</td><td>sell</td><td>out</td><td>1.00</td><td>110.0</td><td>202</td><td>-1</td><td>0</td><td>10</td><td>tp 120.0</td></tr>
        </table></body></html>""",
        encoding="utf-16",
    )

    lifecycles = parse_mt5_report_lifecycles(
        report, trace_by_sequence={7: "trace-owner-7"}, initial_equity=10_000.0, bar_seconds=60.0
    )

    assert len(lifecycles) == 1
    lifecycle = lifecycles[0]
    assert lifecycle.trace_id == "trace-owner-7"
    assert lifecycle.entry_sequence == 7
    assert lifecycle.net_pnl == 9.0
    assert lifecycle.final_close_confirmed is True
    assert lifecycle.close_reason == "TP"
    assert lifecycle.metadata["join_source"] == "strategy_tester_report_intent_comment"


def test_report_fallback_uses_current_mt5_deal_schema_and_entry_owner(tmp_path) -> None:
    report = tmp_path / "current-mt5-tester.htm"
    report.write_text(
        """<html><body>
        <table>
          <tr><th>Open Time</th><th>Order</th><th>Symbol</th><th>Type</th><th>Volume</th><th>Price</th><th>S / L</th><th>T / P</th><th>Time</th><th>State</th><th>Comment</th></tr>
          <tr><td>2025.01.01 00:00:00</td><td>101</td><td>BTCUSDm</td><td>sell</td><td>1.00 / 1.00</td><td>100.0</td><td>110.0</td><td>80.0</td><td>2025.01.01 00:00:00</td><td>filled</td><td>SG5:7:4</td></tr>
          <tr><td>2025.01.01 00:05:00</td><td>202</td><td>BTCUSDm</td><td>buy</td><td>1.00 / 1.00</td><td>110.0</td><td></td><td></td><td>2025.01.01 00:05:00</td><td>filled</td><td>SG5:8:2</td></tr>
        </table>
        <table>
          <tr><th>Time</th><th>Deal</th><th>Symbol</th><th>Type</th><th>Direction</th><th>Volume</th><th>Price</th><th>Order</th><th>Commission</th><th>Swap</th><th>Profit</th><th>Balance</th><th>Comment</th></tr>
          <tr><td>2025.01.01 00:00:00</td><td>301</td><td>BTCUSDm</td><td>sell</td><td>in</td><td>1.00</td><td>100.0</td><td>101</td><td>0.00</td><td>0.00</td><td>0.00</td><td>10000.00</td><td>SG5:7:4</td></tr>
          <tr><td>2025.01.01 00:05:00</td><td>302</td><td>BTCUSDm</td><td>buy</td><td>out</td><td>1.00</td><td>110.0</td><td>202</td><td>-1.00</td><td>0.00</td><td>10.00</td><td>10009.00</td><td>SG5:8:2</td></tr>
        </table>
        </body></html>""",
        encoding="utf-16",
    )

    lifecycles = parse_mt5_report_lifecycles(
        report, trace_by_sequence={7: "trace-entry-7", 8: "trace-exit-8"}, initial_equity=10_000.0
    )

    assert len(lifecycles) == 1
    lifecycle = lifecycles[0]
    assert lifecycle.net_pnl == 9.0
    assert lifecycle.entry_sequence == 7
    assert lifecycle.trace_id == "trace-entry-7"


def test_report_intent_sequence_accepts_sg5_and_unambiguous_legacy_sg4() -> None:
    assert _report_intent_sequence("SG5:91:3") == 91
    assert _report_intent_sequence("SG4:17") == 17
    assert _report_intent_sequence("SG5") is None


def test_lifecycle_credit_candidates_prefers_native_ledger_per_trace() -> None:
    native = MT5PositionLifecycle(
        position_ticket="native:1", entry_time=None, exit_time=None, net_pnl=-10.0, gross_profit=-10.0,
        commission=0.0, swap=0.0, initial_risk=10.0, realized_r=-1.0, duration_bars=5.0,
        close_reason="SL", source="execution_ledger", trace_id="trace-7", entry_sequence=7,
        entry_count=1, close_count=1, final_close_confirmed=True, exit_mechanism="SL",
    )
    report = MT5PositionLifecycle(
        position_ticket="report:1", entry_time=None, exit_time=None, net_pnl=-9.0, gross_profit=-9.0,
        commission=0.0, swap=0.0, initial_risk=10.0, realized_r=-0.9, duration_bars=5.0,
        close_reason="SL", source="strategy_tester_report_tag_fallback", trace_id="trace-7", entry_sequence=7,
        entry_count=1, close_count=1, final_close_confirmed=True, exit_mechanism="SL",
    )

    candidates = lifecycle_credit_candidates((native,), (report,))

    assert candidates == (native,)


def test_report_lifecycle_merge_deduplicates_native_and_resolves_fallback_owner() -> None:
    native = MT5PositionLifecycle(
        position_ticket="native:1", entry_time=None, exit_time=None, net_pnl=10.0, gross_profit=10.0,
        commission=0.0, swap=0.0, initial_risk=10.0, realized_r=1.0, duration_bars=5.0,
        close_reason="TP", source="execution_ledger", trace_id="entry-trace", entry_sequence=7,
        entry_count=1, close_count=1, final_close_confirmed=True, exit_mechanism="TP",
    )
    report_match = replace(native, position_ticket="report:1", source="strategy_tester_report_tag_fallback")
    report_fallback = replace(
        native,
        position_ticket="report:2",
        net_pnl=-5.0,
        gross_profit=-5.0,
        realized_r=-0.5,
        trace_id="",
        entry_sequence=9,
        source="strategy_tester_report_tag_fallback",
    )

    merged = merge_report_lifecycles(
        (native,),
        (report_match, report_fallback),
        trace_by_sequence={9: "fallback-entry-trace"},
    )

    assert len(merged) == 2
    assert merged[0].position_ticket == "native:1"
    assert merged[0].trace_id == "entry-trace"
    assert merged[0].metadata["report_lifecycle_join"] is True
    assert merged[1].trace_id == "fallback-entry-trace"
    assert merged[1].metadata["report_fallback"] is True
    assert merged[1].metadata["report_fallback_owner_resolved"] is True


def test_report_lifecycle_summary_recomputes_effective_samples() -> None:
    first = MT5PositionLifecycle(
        position_ticket="native:1", entry_time=None, exit_time=None, net_pnl=10.0, gross_profit=10.0,
        commission=0.0, swap=0.0, initial_risk=10.0, realized_r=1.0, duration_bars=5.0,
        close_reason="TP", source="execution_ledger", trace_id="entry-trace", entry_sequence=7,
        entry_count=1, close_count=1, final_close_confirmed=True, exit_mechanism="TP",
    )
    second = replace(first, position_ticket="report:2", net_pnl=-5.0, gross_profit=-5.0,
                     realized_r=-0.5, trace_id="fallback-entry-trace")
    summary = replace(MT5ExecutionSummary(ledger_available=True, ledger_integrity=1.0), lifecycles=(first,))

    effective = replace_execution_summary_lifecycles(
        summary, (first, second), initial_equity=10_000.0, min_holding_bars=2
    )

    assert effective.lifecycle_count == 2
    assert effective.expectancy == pytest.approx(0.25)
    assert effective.lifecycles == (first, second)
    assert effective.ledger_integrity == pytest.approx(1.0)


def test_report_lifecycle_merge_uses_unique_pnl_when_report_time_drifts() -> None:
    native = MT5PositionLifecycle(
        position_ticket="native:time-drift", entry_time=None,
        exit_time=datetime(2025, 1, 1, 0, 12, 16), net_pnl=15.99, gross_profit=15.99,
        commission=0.0, swap=0.0, initial_risk=15.99, realized_r=1.0, duration_bars=5.0,
        close_reason="EXPERT", source="execution_ledger", trace_id="entry-trace",
        entry_sequence=6, entry_count=1, close_count=1, final_close_confirmed=True,
        exit_mechanism="EXPERT",
    )
    report = replace(
        native,
        position_ticket="report:time-drift",
        exit_time=datetime(2025, 1, 1, 0, 11, 15),
        source="strategy_tester_report_tag_fallback",
    )

    merged = merge_report_lifecycles((native,), (report,))

    assert len(merged) == 1
    assert merged[0].position_ticket == "native:time-drift"
    assert merged[0].trace_id == "entry-trace"
    assert merged[0].metadata["report_lifecycle_join"] is True


def _fitness_metrics(
    *,
    profit_factor: float,
    win_rate: float,
    total_trades: int,
    net_profit: float,
    drawdown: float = 2.0,
    avg_win: float = 1.0,
    avg_loss: float = -1.0,
) -> MT5BacktestMetrics:
    return MT5BacktestMetrics(
        net_profit=net_profit,
        profit_factor=profit_factor,
        total_trades=total_trades,
        balance_drawdown_pct=drawdown,
        equity_drawdown_pct=drawdown,
        report_path=Path("/tmp/fitness.htm"),
        win_rate=win_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
    )


def test_mt5_fitness_preserves_losing_and_profitable_candidate_order() -> None:
    """A zero-win strategy must never be promoted by a zero multiplicative gate."""
    zero_win = _fitness_metrics(profit_factor=0.0, win_rate=0.0, total_trades=186, net_profit=-100.0)
    pf_half = _fitness_metrics(profit_factor=0.5, win_rate=45.0, total_trades=186, net_profit=-100.0)
    pf_ninety = _fitness_metrics(profit_factor=0.9, win_rate=47.0, total_trades=186, net_profit=-10.0)
    profitable = _fitness_metrics(
        profit_factor=1.25,
        win_rate=55.0,
        total_trades=186,
        net_profit=100.0,
        avg_win=1.25,
        avg_loss=-1.0,
    )

    assert mt5_fitness(zero_win) < mt5_fitness(pf_half) < mt5_fitness(pf_ninety) < mt5_fitness(profitable)
    assert mt5_structural_j(pf_half) < mt5_structural_j(pf_ninety)


def test_mt5_structural_objective_is_lot_scale_invariant() -> None:
    base = _fitness_metrics(
        profit_factor=1.4,
        win_rate=58.0,
        total_trades=186,
        net_profit=120.0,
        avg_win=1.4,
        avg_loss=-1.0,
    )
    scaled = _fitness_metrics(
        profit_factor=1.4,
        win_rate=58.0,
        total_trades=186,
        net_profit=1200.0,
        avg_win=14.0,
        avg_loss=-10.0,
    )

    assert mt5_structural_j(base) == mt5_structural_j(scaled)
    assert mt5_fitness(base) == mt5_fitness(scaled)


def test_mt5_fitness_complexity_penalty_does_not_floor_profitable_deep_policy() -> None:
    metrics = _fitness_metrics(
        profit_factor=1.32,
        win_rate=62.24,
        total_trades=143,
        net_profit=646.51,
        drawdown=4.02,
        avg_win=1.32,
        avg_loss=-1.0,
    )

    base = mt5_fitness_breakdown(metrics, backtest_days=1, complexity=0.0)
    deep = mt5_fitness_breakdown(metrics, backtest_days=1, complexity=1_727.0)

    assert deep.rank_score < base.rank_score
    assert deep.rank_score > -100.0
    assert deep.eligible is True


def test_mt5_fitness_bounds_negative_scope_rank_penalty_without_relaxing_gate() -> None:
    summary = MT5ExecutionSummary(
        ledger_available=True,
        lifecycle_count=8,
        complete_lifecycle_ratio=1.0,
        risk_coverage=1.0,
        trace_coverage=1.0,
        ledger_integrity=1.0,
        execution_failure_rate=0.0,
        immediate_exit_ratio=0.0,
        expectancy_lcb=0.5,
        profit_factor_shrunk=1.3,
        cvar_5pct=0.2,
    )
    metrics = replace(
        _fitness_metrics(
            profit_factor=1.3,
            win_rate=60.0,
            total_trades=8,
            net_profit=100.0,
            drawdown=2.0,
        ),
        execution_summary=summary,
        expectancy_audit={
            "schema_version": 13,
            "policy_origin_lifecycle_count": 8,
            "all_origin_lifecycle_count": 8,
            "policy_origin_ratio": 1.0,
            "policy_expectancy": {"expectancy_lcb": 0.5, "sample_count": 8},
            "negative_critical_scope_count": 100,
        },
    )

    breakdown = mt5_fitness_breakdown(metrics, execution_summary=summary, min_position_samples=8)

    assert breakdown.rank_score > -100.0
    assert breakdown.eligible is False
    assert "negative_context_action_scopes:100" in breakdown.eligibility_reasons


def test_execution_ledger_owner_audit_counts_policy_owned_terminal_closes() -> None:
    """A terminal SL close keeps its entry owner even when source is unknown."""
    policy_owned = MT5PositionLifecycle(
        position_ticket="1", entry_time=None, exit_time=None, net_pnl=-10.0,
        gross_profit=-10.0, commission=0.0, swap=0.0, initial_risk=10.0,
        realized_r=-1.0, close_reason="INITIAL_SL", source="unknown",
        trace_id="entry-trace-1", entry_count=1, close_count=1,
        final_close_confirmed=True,
        metadata={"position_owner_trace_id": "entry-trace-1", "position_owner_policy": "CTX_TANG_MANH"},
    )
    unowned = MT5PositionLifecycle(
        position_ticket="2", entry_time=None, exit_time=None, net_pnl=5.0,
        gross_profit=5.0, commission=0.0, swap=0.0, initial_risk=10.0,
        realized_r=0.5, close_reason="TP", source="unknown", trace_id="",
        entry_count=1, close_count=1, final_close_confirmed=True,
    )
    summary = MT5ExecutionSummary(lifecycle_count=2, lifecycles=(policy_owned, unowned))

    audit = execution_ledger_owner_expectancy_audit(summary, min_samples=1)

    assert audit["audit_source"] == "execution_ledger_immutable_owner_v1"
    assert audit["read_only"] is True
    assert audit["all_origin_lifecycle_count"] == 2
    assert audit["policy_origin_lifecycle_count"] == 1
    assert audit["policy_origin_ratio"] == pytest.approx(0.5)
    assert audit["policy_expectancy"]["sample_count"] == 1


def test_frozen_expectancy_uses_only_current_window_and_retains_history_as_diagnostic() -> None:
    checkpoint = {
        "policy_origin_lifecycle_count": 330,
        "policy_expectancy": {"sample_count": 330, "expectancy_lcb": -0.25},
        "negative_critical_scope_count": 6,
    }
    current = {
        "policy_origin_lifecycle_count": 16,
        "policy_expectancy": {"sample_count": 16, "expectancy_lcb": 0.04},
        "negative_critical_scope_count": 0,
    }

    frozen = select_mt5_evaluation_expectancy_audit(
        checkpoint, current, learning_enabled=False,
    )
    online = select_mt5_evaluation_expectancy_audit(
        checkpoint, current, learning_enabled=True,
    )

    assert frozen["policy_origin_lifecycle_count"] == 16
    assert frozen["policy_expectancy"]["expectancy_lcb"] == pytest.approx(0.04)
    assert frozen["checkpoint_expectancy_audit"] == checkpoint
    assert online["policy_origin_lifecycle_count"] == 330
    assert online["execution_ledger_owner_audit"] == current


def test_experience_journal_compaction_keeps_complete_bounded_jsonl_tail(tmp_path) -> None:
    journal = tmp_path / "experience.jsonl"
    journal.write_text(
        "".join(json.dumps({"sequence": index, "payload": "x" * 80}) + "\n" for index in range(24)),
        encoding="utf-8",
    )
    (tmp_path / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=tmp_path, experience_journal_max_bytes=450)
    )

    evaluator._compact_experience_journal(journal)

    retained = journal.read_text(encoding="utf-8").splitlines()
    assert journal.stat().st_size <= 450
    assert retained
    assert all(isinstance(json.loads(line), dict) for line in retained)
    assert json.loads(retained[-1])["sequence"] == 23


def test_bridge_execution_evidence_requires_mt5_ack_or_account_order_change() -> None:
    before = BridgeRequest(sequence=10, inputs=np.zeros(2), positions=0, day_trades=0)
    ignored = BridgeRequest(sequence=11, inputs=np.zeros(2), positions=0, day_trades=0)
    confirmed = BridgeRequest(
        sequence=11,
        inputs=np.zeros(2),
        positions=0,
        day_trades=0,
        execution_pipe="ack_sequence=10|accepted=1|placed=1|retcode=10008|ticket=123",
    )

    ignored_details = MT5StrategyTesterEvaluator._execution_details(before, ignored, requested_action=3)
    confirmed_details = MT5StrategyTesterEvaluator._execution_details(before, confirmed, requested_action=3)

    assert ignored_details["execution_confirmed"] is False
    assert ignored_details["memory_eligible"] is False
    assert confirmed_details["execution_confirmed"] is True
    assert confirmed_details["outcome_eligible"] is False
    assert confirmed_details["action_feedback_eligible"] is False
    assert confirmed_details["memory_eligible"] is False
    assert confirmed_details["ticket"] == "123"


def test_unconfirmed_bridge_trace_is_not_written_to_experience_journal(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    journal = tmp_path / "experience.jsonl"
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, experience_journal=str(journal))
    )
    policy = SelfGrowingPolicy.seed(input_size=2, output_size=19, hidden_size=3, seed=83)
    trace = policy.act(np.array([0.1, -0.2]))
    trace = replace(
        trace,
        execution_details={
            "bridge_observed": True,
            "execution_confirmed": False,
            "memory_eligible": False,
        },
    )
    request = BridgeRequest(sequence=1, inputs=np.array([0.1, -0.2]))

    evaluator._append_experience_journal(
        round_index=0,
        policy_id=1,
        request=request,
        trace=trace,
        next_obs=np.array([0.2, -0.1]),
        reward=1.0,
        cost=0.0,
        done=False,
        signal={"td_delta": 1.0, "prediction_error": 0.1},
        action=3,
        score=1.0,
        confidence=0.5,
    )

    assert not journal.exists()


def test_account_context_encoder_uses_normalized_stress_features(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, max_positions=5, max_holding_bars=30, max_daily_trades=60)
    )
    request = BridgeRequest(
        sequence=9,
        inputs=np.concatenate((np.zeros(70), np.asarray([0.08]), np.zeros(66))),
        balance=10000.0,
        equity=9850.0,
        drawdown_pct=3.2,
        positions=4,
        day_trades=12,
        positions_pipe="111:1:0.01:100:99:103:-1.0:15^222:-1:0.01:101:102:98:0.4:4",
    )

    context = evaluator._account_context_for_request(
        request,
        [10000.0, 9990.0, 9975.0, 9950.0, 9900.0, 9870.0],
    )

    assert context["schema_version"] == 7.0
    assert context["equity_to_balance"] == pytest.approx(-0.015)
    assert context["floating_pnl_r"] == pytest.approx(-0.12)
    assert context["current_equity_drawdown"] == pytest.approx(0.032)
    assert context["open_risk_ratio"] == pytest.approx(0.08)
    assert context["position_count_norm"] == pytest.approx(0.8)
    assert context["losing_streak_norm"] == pytest.approx(0.2)
    assert context["time_under_water_norm"] == pytest.approx(0.5)
    assert context["day_trade_pressure"] == pytest.approx(0.0)
    assert "balance" not in context
    assert "equity" not in context


def test_experience_journal_writes_account_context_schema_without_raw_balances(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    journal = tmp_path / "experience.jsonl"
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, experience_journal=str(journal))
    )
    policy = SelfGrowingPolicy.seed(input_size=2, output_size=20, hidden_size=3, seed=84)
    account_context = {
        "equity_to_balance": -0.02,
        "floating_pnl_r": -0.3,
        "current_equity_drawdown": 0.04,
        "open_risk_ratio": 0.6,
    }
    trace = replace(
        policy.act(np.array([0.1, -0.2])),
        execution_details={
            "bridge_observed": True,
            "execution_confirmed": True,
            "memory_eligible": True,
        },
        account_context=account_context,
        account_context_after={**account_context, "floating_pnl_r": -0.1},
    )
    request = BridgeRequest(sequence=1, inputs=np.array([0.1, -0.2]), balance=10000.0, equity=9800.0)

    evaluator._append_experience_journal(
        round_index=0,
        policy_id=1,
        request=request,
        trace=trace,
        next_obs=np.array([0.2, -0.1]),
        reward=1.0,
        cost=0.0,
        done=False,
        signal={"td_delta": 1.0, "prediction_error": 0.1},
        action=1,
        score=1.0,
        confidence=0.5,
    )

    item = json.loads(journal.read_text(encoding="utf-8").splitlines()[0])
    assert item["schema_version"] == 3
    assert item["memory_key_schema"] == 6
    assert item["account_context"] == account_context
    assert item["trace"]["account_context"] == account_context
    assert "balance" not in item["account_context"]
    assert "equity" not in item["account_context"]



def test_write_action_emits_current_wire_schema_and_20_outputs(tmp_path) -> None:
    request = BridgeRequest(sequence=7, inputs=np.zeros(2), event_id="evt-7", schema_version=3)
    path = tmp_path / "self_growing_action.csv"

    MT5StrategyTesterEvaluator._write_action(
        path, request, "trace-7", action=3, score=0.25, confidence=0.5, outputs=np.asarray([0.0, 1.0])
    )

    row = path.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert row[0] == str(BRIDGE_SCHEMA_VERSION)
    assert row[4:6] == ["0", "1"]
    assert float(row[9]) > 0.0
    assert len(row[10].split("|")) == 20


def test_write_action_embeds_context_owner_display_metadata_for_ea(tmp_path) -> None:
    request = BridgeRequest(sequence=7, inputs=np.zeros(2), event_id="evt-7", schema_version=7)
    path = tmp_path / "self_growing_action.csv"
    trace = SimpleNamespace(
        trace_id="trace-owner",
        context_audit={
            "context_nhan_dien": "CTX_TANG_MANH",
            "ma_chuyen_gia": "CG_TANG_MANH",
            "ma_policy": "CG_TANG_MANH",
            "trang_thai_chuyen_gia": "HOAT_DONG",
            "ly_do_context": "regime=TANG_MANH_conf=0.82",
        },
    )

    MT5StrategyTesterEvaluator._write_action(
        path,
        request,
        "trace-owner",
        action=1,
        score=0.25,
        confidence=0.5,
        outputs=np.asarray([0.0, 1.0]),
        trace=trace,
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    row = lines[1].split("\t")
    assert header[11:] == ["market_context", "expert_id", "policy_id", "expert_state", "context_reason", "no_trade_reason"]
    assert row[11:16] == ["CTX_TANG_MANH", "CG_TANG_MANH", "CG_TANG_MANH", "HOAT_DONG", "regime=TANG_MANH_conf=0.82"]


def test_write_action_batch_emits_five_atomic_agent_rows(tmp_path) -> None:
    request = BridgeRequest(sequence=11, inputs=np.zeros(INPUT_COUNT), event_id="evt-11")
    path = tmp_path / "self_growing_action.csv"
    actions = [
        BridgeAgentAction(
            agent_id=index,
            trace_id=f"trace-{index}",
            action=index % 3,
            score=0.1 * index,
            confidence=0.5,
            outputs=np.linspace(-0.2, 0.2, 20),
        )
        for index in range(5)
    ]

    MT5StrategyTesterEvaluator._write_action_batch(path, request, actions)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 6
    header = lines[0].split("\t")
    assert header[4:6] == ["agent_id", "agent_count"]
    rows = [line.split("\t") for line in lines[1:]]
    assert [int(row[4]) for row in rows] == list(range(5))
    assert {int(row[5]) for row in rows} == {5}
    assert len({row[2] for row in rows}) == 5
    assert all(len(row[10].split("|")) == 20 for row in rows)


def test_write_action_batch_rejects_incomplete_or_duplicate_agents(tmp_path) -> None:
    request = BridgeRequest(sequence=12, inputs=np.zeros(INPUT_COUNT), event_id="evt-12")
    row = BridgeAgentAction(0, "trace-0", 0, 0.0, 0.0, np.zeros(20))
    with pytest.raises(ValueError, match="1 or 5"):
        MT5StrategyTesterEvaluator._write_action_batch(tmp_path / "a.csv", request, [row, row])
    duplicate = [replace(row, trace_id=f"trace-{index}") for index in range(5)]
    with pytest.raises(ValueError, match="agent ids"):
        MT5StrategyTesterEvaluator._write_action_batch(tmp_path / "b.csv", request, duplicate)


def test_five_bridge_packet_commits_all_five_agent_memories_together() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner._learning_enabled = False
        return inner

    config = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=config, tao_policy=factory)
    request = BridgeRequest(
        sequence=21,
        event_id="evt-21",
        inputs=np.linspace(-0.1, 0.1, INPUT_COUNT),
    )
    traces = system.act_nam_tac_nhan(request.inputs)
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig())

    packet, enriched = evaluator._five_bridge_action_packet(
        system,
        request,
        traces,
    )

    assert [row.agent_id for row in packet] == list(range(5))
    assert len({row.trace_id for row in packet}) == 5
    assert all(item.action_details["five_agent_bridge_simultaneous"] for item in enriched)
    assert all(not item.action_details["five_agent_bridge_serialized"] for item in enriched)
    assert all(
        row.action in five_agent_authority(row.agent_id).allowed_actions
        for row in packet
    )
    assert all(
        bool(item.action_details["action_masked"])
        == (
            int(item.action_details["policy_action"])
            != int(item.action_details["wire_action"])
        )
        for item in enriched
    )
    policy = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    assert policy.five_agent_history_steps == [1, 1, 1, 1, 1]


def test_five_rlhf_reference_capture_survives_mt5_observatory_snapshot() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner._learning_enabled = False
        return inner

    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
            so_tac_nhan=5,
            so_chieu_nhan_dang_tac_nhan=5,
            entry_style_specialists_bat=False,
            position_management_specialist_bat=False,
            volatile_subexperts_bat=False,
        ),
        tao_policy=factory,
    )
    installed = system.install_five_rlhf_reference()
    assert installed["installed"] is True
    traces = system.act_nam_tac_nhan(np.zeros(INPUT_COUNT, dtype=np.float32))
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig())
    snapshot = evaluator._execution_trace_snapshot(traces[0])
    details = snapshot["action_details"]
    assert details["five_reference_policy_frozen"] is True
    assert details["five_reference_policy_sha256"] == installed["reference_policy_sha256"]
    assert len(details["five_reference_policy_action_logits"]) == len(
        details["five_policy_action_logits"]
    )
    assert len(snapshot["allowed_action_mask"]) == len(
        details["five_policy_action_logits"]
    )
    cleared = system.clear_five_rlhf_reference()
    assert cleared["cleared"] >= 1


def test_mt5_rlhf_policy_phase_requires_frozen_artifact_contract() -> None:
    with pytest.raises(ValueError, match="reward model path"):
        MT5BacktestConfig(rlhf_policy_update_enabled=True)
    candidate = MT5BacktestConfig(
        rlhf_policy_update_enabled=True,
        rlhf_reward_model_path="outputs/rlhf/reward_model_v1.json",
        rlhf_source_checkpoint_sha256="source-checkpoint-hash",
        rlhf_update_interval_packets=4,
        rlhf_max_updates_per_run=12,
    )
    assert candidate.rlhf_policy_update_enabled is True
    assert candidate.rlhf_update_interval_packets == 4
    assert candidate.rlhf_max_updates_per_run == 12


def test_mt5_rlhf_policy_phase_loads_reward_model_and_installs_reference() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner._learning_enabled = False
        return inner

    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
            so_tac_nhan=5,
            so_chieu_nhan_dang_tac_nhan=5,
            entry_style_specialists_bat=False,
            position_management_specialist_bat=False,
            volatile_subexperts_bat=False,
        ),
        tao_policy=factory,
    )
    model = RLHFRewardModel(
        feature_schema=RLHF_POLICY_FEATURE_SCHEMA,
        mean=(0.0,) * (INPUT_COUNT + 7),
        scale=(1.0,) * (INPUT_COUNT + 7),
        weights=(0.0,) * (INPUT_COUNT + 7),
        bias=0.0,
    )
    artifact = Path("outputs/rlhf_test_reward_model_mt5.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    try:
        artifact.write_text(json.dumps({
            "model": model.to_row(),
            "feedback_mode": "human",
            "test_gate": {"passed": True, "reasons": []},
        }), encoding="utf-8")
        evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
            offline_replay=True,
            rlhf_policy_update_enabled=True,
            rlhf_reward_model_path=str(artifact),
            rlhf_reward_feedback_mode="human",
            rlhf_source_checkpoint_sha256="source-checkpoint-hash",
        ))
        loaded, manifest = evaluator._prepare_mt5_rlhf_policy_phase(system)
        assert loaded.sha256 == model.sha256
        assert manifest["reference_policy_frozen"] is True
        assert manifest["reference_policy_sha256"]
    finally:
        system.clear_five_rlhf_reference()
        artifact.unlink(missing_ok=True)


def test_mt5_rlhf_policy_phase_requires_independent_teacher_provenance() -> None:
    model = RLHFRewardModel(
        feature_schema=RLHF_POLICY_FEATURE_SCHEMA,
        mean=(0.0,) * (INPUT_COUNT + 7),
        scale=(1.0,) * (INPUT_COUNT + 7),
        weights=(0.0,) * (INPUT_COUNT + 7),
        bias=0.0,
    )
    artifact = Path("outputs/rlhf_test_reward_model_independent_teacher.json")
    artifact.parent.mkdir(parents=True, exist_ok=True)

    class FakePolicy:
        def install_five_rlhf_reference(self):
            return {
                "installed": True,
                "reference_policy_frozen": True,
                "reference_policy_sha256": "reference-hash",
            }

    teacher_hash = "teacher-checkpoint-hash"
    try:
        artifact.write_text(json.dumps({
            "model": model.to_row(),
            "feedback_mode": "rlaif",
            "teacher_provenance": {
                "teacher_checkpoint_sha256": teacher_hash,
                "teacher_id": "independent-teacher",
            },
            "test_gate": {"passed": True, "reasons": []},
        }), encoding="utf-8")
        evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
            offline_replay=True,
            rlhf_policy_update_enabled=True,
            rlhf_reward_model_path=str(artifact),
            rlhf_reward_feedback_mode="rlaif",
            rlhf_teacher_checkpoint_sha256=teacher_hash,
            rlhf_require_independent_teacher=True,
            rlhf_source_checkpoint_sha256="student-checkpoint-hash",
        ))
        loaded, manifest = evaluator._prepare_mt5_rlhf_policy_phase(FakePolicy())
        assert loaded.sha256 == model.sha256
        assert manifest["teacher_checkpoint_sha256"] == teacher_hash
        assert manifest["independent_teacher_required"] is True

        same_source = MT5StrategyTesterEvaluator(MT5BacktestConfig(
            offline_replay=True,
            rlhf_policy_update_enabled=True,
            rlhf_reward_model_path=str(artifact),
            rlhf_reward_feedback_mode="rlaif",
            rlhf_teacher_checkpoint_sha256=teacher_hash,
            rlhf_require_independent_teacher=True,
            rlhf_source_checkpoint_sha256=teacher_hash,
        ))
        with pytest.raises(ValueError, match="must differ from the source champion"):
            same_source._prepare_mt5_rlhf_policy_phase(FakePolicy())
    finally:
        artifact.unlink(missing_ok=True)


def test_five_bridge_packet_blocks_lower_probability_opposite_entry() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner._learning_enabled = False
        return inner

    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
            so_tac_nhan=5,
            so_chieu_nhan_dang_tac_nhan=5,
            entry_style_specialists_bat=False,
            position_management_specialist_bat=False,
            volatile_subexperts_bat=False,
        ),
        tao_policy=factory,
    )
    request = BridgeRequest(
        sequence=22, event_id="opposite-22", inputs=np.zeros(INPUT_COUNT),
    )
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig())
    traces = list(system.act_nam_tac_nhan(request.inputs))
    forced = ((0, int(Action.MARKET_BUY), 0.40), (1, int(Action.MARKET_SELL), 0.70))
    for agent, action, probability in forced:
        outputs = evaluator._force_action_output(
            np.asarray(traces[agent].selected_action, dtype=float), action
        )
        details = dict(traces[agent].action_details)
        details["five_action_index"] = action
        details["five_selected_action_probability"] = probability
        traces[agent] = replace(
            traces[agent], selected_idx=action, selected_action=outputs,
            confidence=probability, action_details=details,
        )

    packet, enriched = evaluator._five_bridge_action_packet(
        system, request, tuple(traces)
    )

    assert packet[0].action == int(Action.HOLD)
    assert packet[1].action == int(Action.MARKET_SELL)
    assert enriched[0].action_details["five_pre_wire_arbitration_blocked"] is True
    assert enriched[0].action_details["action_origin"] == "opposite_entry_arbitration"
    assert enriched[1].action_details["five_opposite_entry_winner_agent"] == 1


def test_mt5_five_pattern_path_routes_once_and_masks_unexecuted_actor_credit() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner.five_pattern_reward_enabled = True
        return inner

    config = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=config, tao_policy=factory)
    rows = list(system.act_nam_tac_nhan(np.zeros(INPUT_COUNT, dtype=np.float32)))
    actions = (1, 2, 1, 2, 0)
    for index, row in enumerate(rows):
        details = dict(row.action_details)
        details["five_action_index"] = actions[index]
        details["action_masked"] = index == 1
        details["five_arbitration_blocked"] = index == 2
        rows[index] = replace(
            row,
            action_details=details,
            execution_details={"rejected": index == 3},
        )
    registered = system.register_five_pattern_anchor(rows, sequence=42)
    assert registered["registered"] is True
    bound = system.bind_five_pattern_anchor_receipts(rows, sequence=42)
    assert bound["bound"] is True

    local = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    captured = {}
    original_observe = local.observe_five_pattern_transition

    def observe(traces, rewards, *, auxiliary_targets=None):
        captured["traces"] = tuple(traces)
        captured["rewards"] = np.asarray(rewards)
        captured["targets"] = dict(auxiliary_targets or {})
        return original_observe(traces, rewards, auxiliary_targets=auxiliary_targets)

    local.observe_five_pattern_transition = observe
    record = EntryPreferenceRecord(
        trace_id=rows[0].trace_id,
        sequence=42,
        decision_time="2025.01.01 00:00",
        selected_action=1,
        bid=100.0,
        ask=100.2,
        atr=1.0,
        sl_atr=1.0,
        rr=2.0,
        entry_prices=(100.2, 100.0, 99.5, 100.7, 100.7, 99.5),
        bars=(EntryPathBar(102.0, 99.8, 101.5, 0.2),),
        observed_bars=48,
    )
    result = system.apply_five_pattern_path_credit_detailed(rows[0].trace_id, record)
    assert result["accepted"] is True
    eligibility = [
        bool(trace.action_details["five_pattern_actor_eligible"])
        for trace in captured["traces"]
    ]
    assert eligibility == [True, False, False, False, True]
    ranking_eligibility = [
        bool(trace.action_details["five_pattern_ranking_eligible"])
        for trace in captured["traces"]
    ]
    assert ranking_eligibility == [True, False, False, False, False]
    np.testing.assert_allclose(captured["rewards"][:4], 0.0)
    assert captured["rewards"][4] < 0.0
    assert captured["targets"]["direction_class"] == 1
    # Tác nhân 4 chỉ quản trị danh mục, nên không nhận gradient xếp hạng
    # counterfactual cho sáu hành động entry.
    assert bool(local.five_pattern_steps[-1]["ranking_loss_masks"][4]) is False

    duplicate = system.apply_five_pattern_path_credit_detailed(
        rows[1].trace_id,
        replace(record, trace_id=rows[1].trace_id),
    )
    assert duplicate["accepted"] is True
    assert duplicate["queued"] is False
    assert duplicate["reason"] == "duplicate_sequence"

    registered = system.register_five_pattern_anchor(rows, sequence=43)
    assert registered["registered"] is True
    no_edge = replace(
        record,
        sequence=43,
        bars=(EntryPathBar(100.2, 100.0, 100.1, 0.2),),
    )
    result = system.apply_five_pattern_path_credit_detailed(rows[0].trace_id, no_edge)
    assert result["accepted"] is True
    assert captured["targets"]["direction_class"] == 0
    assert not bool(np.any(local.five_pattern_steps[-1]["ranking_loss_masks"]))

    # A flat HOLD snapshot must still become a ranking anchor when its causal
    # path proves a legal entry beats HOLD. Management/governor agents remain
    # excluded by the legal-entry mask.
    hold_rows = list(rows)
    hold_details = dict(hold_rows[0].action_details)
    hold_details["five_action_index"] = 0
    hold_rows[0] = replace(
        hold_rows[0], selected_idx=0, action_details=hold_details,
    )
    # Simulate the management-only row explicitly: HOLD plus a management
    # action, but no legal entry. It must not become a counterfactual entry
    # ranking anchor even though HOLD is a valid entry-preference action.
    management_mask = [False] * 12
    management_mask[0] = True
    management_mask[10] = True
    hold_rows[4] = replace(
        hold_rows[4], allowed_action_mask=tuple(management_mask),
    )
    registered = system.register_five_pattern_anchor(hold_rows, sequence=44)
    assert registered["registered"] is True
    bound = system.bind_five_pattern_anchor_receipts(hold_rows, sequence=44)
    assert bound["bound"] is True
    hold_record = replace(
        record,
        trace_id=hold_rows[0].trace_id,
        sequence=44,
        selected_action=0,
    )
    result = system.apply_five_pattern_path_credit_detailed(
        hold_rows[0].trace_id, hold_record,
    )
    assert result["accepted"] is True
    hold_ranking = [
        bool(trace.action_details["five_pattern_ranking_eligible"])
        for trace in captured["traces"]
    ]
    assert hold_ranking[0] is True
    assert hold_ranking[4] is False

    # A malformed mask may retain the legacy PPO fallback, but it must never
    # authorize a counterfactual ranking target.
    malformed_rows = list(hold_rows)
    malformed_rows[0] = replace(malformed_rows[0], allowed_action_mask=())
    registered = system.register_five_pattern_anchor(malformed_rows, sequence=45)
    assert registered["registered"] is True
    bound = system.bind_five_pattern_anchor_receipts(malformed_rows, sequence=45)
    assert bound["bound"] is True
    malformed_record = replace(
        hold_record,
        trace_id=malformed_rows[0].trace_id,
        sequence=45,
    )
    result = system.apply_five_pattern_path_credit_detailed(
        malformed_rows[0].trace_id, malformed_record,
    )
    assert result["accepted"] is True
    malformed_ranking = [
        bool(trace.action_details["five_pattern_ranking_eligible"])
        for trace in captured["traces"]
    ]
    assert malformed_ranking[0] is False


def test_mt5_delayed_five_backend_can_use_selector_without_v4_pattern_reward() -> None:
    evaluator = MT5StrategyTesterEvaluator()

    class LegacyFive:
        five_entry_preference_supervision_enabled = False

    class SelectorFive:
        five_entry_preference_supervision_enabled = True

    assert evaluator._entry_preference_credit_backend(
        LegacyFive(), is_five_pattern=True
    ) == "five_pattern_path"
    assert evaluator._entry_preference_credit_backend(
        SelectorFive(), is_five_pattern=True
    ) == "entry_preference"
    assert evaluator._entry_preference_credit_backend(
        LegacyFive(), is_five_pattern=False
    ) == "entry_preference"


def test_mt5_five_selector_credit_keeps_ranking_but_zeroes_v4_reward() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner.five_pattern_reward_enabled = False
        inner.five_entry_preference_supervision_enabled = True
        return inner

    config = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=config, tao_policy=factory)
    rows = list(system.act_nam_tac_nhan(np.zeros(INPUT_COUNT, dtype=np.float32)))
    for row_index, row in enumerate(rows):
        details = dict(row.action_details)
        details.update({
            "five_action_index": 1,
            "action_masked": False,
            "five_arbitration_blocked": False,
            "five_agent_index": row_index,
        })
        rows[row_index] = replace(
            row,
            action_details=details,
            execution_details={"rejected": False},
        )
    assert system.register_five_pattern_anchor(rows, sequence=99)["registered"] is True
    assert system.bind_five_pattern_anchor_receipts(rows, sequence=99)["bound"] is True

    local = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    captured: dict[str, object] = {}
    original_observe = local.observe_five_pattern_transition

    def observe(traces, rewards, *, auxiliary_targets=None, credit_channel="pattern"):
        captured["rewards"] = np.asarray(rewards)
        captured["targets"] = dict(auxiliary_targets or {})
        captured["channel"] = credit_channel
        return original_observe(
            traces,
            rewards,
            auxiliary_targets=auxiliary_targets,
            credit_channel=credit_channel,
        )

    local.observe_five_pattern_transition = observe
    record = EntryPreferenceRecord(
        trace_id=rows[0].trace_id,
        sequence=99,
        decision_time="2025.01.01 00:00",
        selected_action=1,
        bid=100.0,
        ask=100.2,
        atr=1.0,
        sl_atr=1.0,
        rr=2.0,
        entry_prices=(100.2, 100.0, 99.5, 100.7, 100.7, 99.5),
        bars=(EntryPathBar(102.0, 99.8, 101.5, 0.2),),
        observed_bars=48,
    )
    result = system.apply_five_entry_preference_credit_detailed(
        rows[0].trace_id, record
    )
    assert result["accepted"] is True
    np.testing.assert_allclose(captured["rewards"], 0.0)
    assert captured["channel"] == "selector"
    assert captured["targets"]["ranking_loss_weight"] > 0.0
    assert captured["targets"]["auxiliary_loss_weight"] == 0.0


def test_mt5_five_bidirectional_selector_credits_flat_hold_owner() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner.five_pattern_reward_enabled = False
        inner.five_entry_preference_supervision_enabled = True
        inner.five_action_ranking_bidirectional_enabled = True
        return inner

    config = CauHinhHeChuyenGia(
        bat=True,
        che_do_quyet_dinh="nam_tac_nhan_dung_chung",
        cho_phep_policy_chung_ra_lenh=True,
        so_tac_nhan=5,
        so_chieu_nhan_dang_tac_nhan=5,
        entry_style_specialists_bat=False,
        position_management_specialist_bat=False,
        volatile_subexperts_bat=False,
    )
    system = HeChuyenGiaMotContextMotPolicy(cau_hinh=config, tao_policy=factory)
    rows = list(system.act_nam_tac_nhan(np.zeros(INPUT_COUNT, dtype=np.float32)))
    for row_index, row in enumerate(rows):
        details = dict(row.action_details)
        details.update({
            "five_action_index": 0 if row_index == 0 else 1,
            "action_masked": False,
            "five_arbitration_blocked": False,
            "five_agent_index": row_index,
        })
        rows[row_index] = replace(
            row,
            selected_idx=details["five_action_index"],
            action_details=details,
            execution_details={"rejected": False},
        )
    assert system.register_five_pattern_anchor(rows, sequence=100)["registered"] is True
    assert system.bind_five_pattern_anchor_receipts(rows, sequence=100)["bound"] is True

    local = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    captured: dict[str, object] = {}
    original_observe = local.observe_five_pattern_transition

    def observe(traces, rewards, *, auxiliary_targets=None, credit_channel="pattern"):
        captured["traces"] = tuple(traces)
        captured["rewards"] = np.asarray(rewards)
        captured["channel"] = credit_channel
        return original_observe(
            traces,
            rewards,
            auxiliary_targets=auxiliary_targets,
            credit_channel=credit_channel,
        )

    local.observe_five_pattern_transition = observe
    record = EntryPreferenceRecord(
        trace_id=rows[0].trace_id,
        sequence=100,
        decision_time="2025.01.01 00:00",
        selected_action=0,
        bid=100.0,
        ask=100.2,
        atr=1.0,
        sl_atr=1.0,
        rr=2.0,
        entry_prices=(100.2, 100.0, 99.5, 100.7, 100.7, 99.5),
        bars=(EntryPathBar(102.0, 99.8, 101.5, 0.2),),
        observed_bars=48,
    )
    result = system.apply_five_entry_preference_credit_detailed(
        rows[0].trace_id, record
    )
    assert result["accepted"] is True
    assert captured["channel"] == "selector"
    rewards = np.asarray(captured["rewards"])
    assert rewards[0] < 0.0
    np.testing.assert_allclose(rewards[1:], 0.0)
    assert captured["traces"][0].action_details[
        "five_counterfactual_hold_credit_eligible"
    ] is True
    assert captured["traces"][0].action_details[
        "five_reward_sample_eligible"
    ] is True


def test_mt5_five_owner_receives_active_pattern_reward_config() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        return inner

    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
            so_tac_nhan=5,
            so_chieu_nhan_dang_tac_nhan=5,
            entry_style_specialists_bat=False,
            position_management_specialist_bat=False,
            volatile_subexperts_bat=False,
        ),
        tao_policy=factory,
    )
    config = load_config("configs/he_chuyen_gia_latest_ui_v193.yaml")
    _cau_hinh_policy_chuyen_gia_cho_mt5(system, config)
    owner = system.registry.chuyen_gia_theo_context["CTX_CHUNG"].policy
    assert owner.five_pattern_reward_enabled is True
    assert owner.five_flat_hold_missed_edge_penalty_r == pytest.approx(0.25)
    assert owner.five_flat_hold_confirmed_no_edge_bonus_r == pytest.approx(0.01)
    assert owner.five_behavior_uniform_mix == pytest.approx(0.15)
    assert owner.five_entropy_start == pytest.approx(0.06)
    assert owner.five_flat_entry_mass_floor_start == pytest.approx(0.75)
    assert owner.five_flat_entry_mass_floor_end == pytest.approx(0.0)
    assert owner.five_flat_entry_mass_loss_weight == pytest.approx(1.0)
    assert owner.five_lifecycle_initial_sl_local_loss_weight == pytest.approx(1.0)
    trainer = owner._ensure_five_trainer()
    assert trainer.schedule.flat_entry_mass_floor == pytest.approx(0.75)
    assert owner.five_lifecycle_batch_size == 32
    assert owner.five_lifecycle_batch_max_wait == 64
    assert owner.five_position_improvement_reward_weight == pytest.approx(1.0)
    assert owner.five_pattern_tree_min_edge_atr == pytest.approx(0.10)
    assert owner.five_pattern_tree_direction_loss_weight == pytest.approx(1.0)
    assert owner.five_pattern_tree_edge_hold_ceiling == pytest.approx(0.20)
    assert owner.five_invalid_cancel_no_pending_penalty_r == pytest.approx(2.0)
    assert owner.five_unfilled_pending_timeout_penalty_r == pytest.approx(0.25)
    assert owner.five_entropy_edge_gate_enabled is True
    assert owner.five_entropy_edge_release_count == 3
    assert owner.five_entropy_edge_decay_count == 8


def test_mt5_five_capacity_mask_applies_before_sampling_and_at_wire() -> None:
    def factory() -> ChuyenGiaPolicy:
        inner = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
        inner.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
        inner.five_agent_memories = inner.five_shared_network.initial_memories()
        inner._learning_enabled = False
        return inner

    system = HeChuyenGiaMotContextMotPolicy(
        cau_hinh=CauHinhHeChuyenGia(
            bat=True,
            che_do_quyet_dinh="nam_tac_nhan_dung_chung",
            cho_phep_policy_chung_ra_lenh=True,
            so_tac_nhan=5,
            so_chieu_nhan_dang_tac_nhan=5,
            entry_style_specialists_bat=False,
            position_management_specialist_bat=False,
            volatile_subexperts_bat=False,
        ),
        tao_policy=factory,
    )
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        offline_replay=True,
        max_positions=5,
    ))
    request = BridgeRequest(
        sequence=77,
        event_id="capacity-77",
        inputs=np.zeros(INPUT_COUNT, dtype=float),
        positions=5,
    )
    allowed = evaluator._allowed_actions_for_request(request)
    assert not (allowed & evaluator._entry_actions())
    system.set_decision_context(
        allowed_actions={int(action): True for action in allowed},
        account_context={"open_risk_ratio": 1.0},
        positions_pipe="",
        execution_context={"position_capacity_remaining": 0.0},
    )
    traces = tuple(system.act_nam_tac_nhan(request.inputs))
    assert all(int(trace.selected_idx) not in evaluator._entry_actions() for trace in traces)

    packet, enriched = evaluator._five_bridge_action_packet(system, request, traces)
    assert all(int(row.action) not in evaluator._entry_actions() for row in packet)
    assert all(bool(trace.action_details["context_contract_enforced_at_wire"]) for trace in enriched)

    # Defensive wire check: even a stale/foreign trace claiming an entry while
    # its immutable decision mask forbids entries is converted to HOLD and is
    # marked masked, so it cannot become actor/path credit.
    stale = replace(
        traces[0],
        selected_idx=int(Action.MARKET_BUY),
        action_details={
            **dict(traces[0].action_details),
            "five_action_index": int(Action.MARKET_BUY),
        },
    )
    stale_rows = (stale,) + traces[1:]
    packet, enriched = evaluator._five_bridge_action_packet(system, request, stale_rows)
    assert packet[0].action == int(Action.HOLD)
    assert enriched[0].action_details["action_masked"] is True


def test_execution_feedback_is_partitioned_by_trace_and_agent() -> None:
    before = BridgeRequest(sequence=31, inputs=np.zeros(INPUT_COUNT), positions=0)
    records = (
        "ack_sequence=31|agent_id=0|intent_trace_id=trace-0|requested_action=1|accepted=1|filled=1"
        "^ack_sequence=31|agent_id=1|intent_trace_id=trace-1|requested_action=2|rejected=1|status=CONFLICT"
    )
    after = BridgeRequest(sequence=32, inputs=np.zeros(INPUT_COUNT), positions=1, execution_pipe=records)

    first = MT5StrategyTesterEvaluator._execution_details(
        before, after, 1, expected_trace_id="trace-0", expected_agent_id=0,
    )
    second = MT5StrategyTesterEvaluator._execution_details(
        before, after, 2, expected_trace_id="trace-1", expected_agent_id=1,
    )

    assert first["filled"] is True
    assert first["rejected"] is False
    assert first["agent_id"] == 0
    assert second["filled"] is False
    assert second["rejected"] is True
    assert second["agent_id"] == 1


def test_visual_trial_closes_stranded_terminal_before_new_launch(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    runner.mkdir(exist_ok=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, visual=True))
    calls = {"close": 0, "close_runner": 0}

    def fake_close_open_terminals():
        calls["close"] += 1

    def fake_close_runner_processes():
        calls["close_runner"] += 1

    monkeypatch.setattr(evaluator, "close_open_terminals", fake_close_open_terminals)
    monkeypatch.setattr(evaluator, "close_runner_processes", fake_close_runner_processes)
    monkeypatch.setattr(evaluator, "_clear_bridge_files", lambda: None)
    monkeypatch.setattr(evaluator, "_clear_tester_cache", lambda: None)
    monkeypatch.setattr("self_growing_trader.mt5_runner.write_tester_set", lambda *args, **kwargs: tmp_path / "test.set")
    monkeypatch.setattr("self_growing_trader.mt5_runner.write_tester_ini", lambda *args, **kwargs: tmp_path / "test.ini")
    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stop-after-preflight")))

    with pytest.raises(RuntimeError, match="stop-after-preflight"):
        evaluator.evaluate_policy(1, object(), round=0)

    assert calls["close"] == 1
    assert calls["close_runner"] == 1


def test_evaluator_keeps_launcher_ini_outside_liveupdate_mql5_tree(
    monkeypatch, tmp_path,
) -> None:
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    runner.mkdir(exist_ok=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    captured: dict[str, object] = {}

    monkeypatch.setattr(evaluator, "_close_stale_tester_agents", lambda: None)
    monkeypatch.setattr(evaluator, "_clear_bridge_files", lambda: None)
    monkeypatch.setattr(evaluator, "_clear_tester_cache", lambda: None)
    monkeypatch.setattr(evaluator, "_clear_report_artifacts", lambda *_args: None)
    monkeypatch.setattr(evaluator, "_write_session_manifest", lambda **_kwargs: tmp_path / "manifest.json")

    def stop_after_command(command, *, cwd):
        captured["command"] = command
        captured["cwd"] = cwd
        raise RuntimeError("stop-after-command")

    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", stop_after_command)

    with pytest.raises(RuntimeError, match="stop-after-command"):
        evaluator.evaluate_policy(820_000, object(), round=0)

    ini = runner / "codex_run_configs" / "round_0000_evaluation_820000.ini"
    assert ini.is_file()
    assert runner / "MQL5" not in ini.parents
    assert f"/config:{ini}" in captured["command"]


def test_visual_bridge_does_not_accept_early_report_before_process_exit(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    report = tmp_path / "early.htm"
    report.write_text("early report placeholder", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, visual=True, timeout_sec=1, experience_journal=None)
    )
    calls = {"parse": 0}

    class RunningProcess:
        def poll(self):
            return None

    def fake_parse(_path):
        calls["parse"] += 1
        raise AssertionError("visual report must not be parsed before process exit")

    monkeypatch.setattr("self_growing_trader.mt5_runner.parse_mt5_report", fake_parse)

    with pytest.raises(TimeoutError, match="Timed out serving MT5 action bridge"):
        evaluator._serve_policy_actions(RunningProcess(), report, _context_policy(), policy_id=1, round_index=0)

    assert calls["parse"] == 0


def test_nonvisual_bridge_finishes_when_a_parseable_report_appears(monkeypatch, tmp_path) -> None:
    """Keep the terminal open without delaying checkpoint persistence."""
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    report = tmp_path / "completed.htm"
    report.write_text("completed report placeholder", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, visual=False, timeout_sec=60, experience_journal=None)
    )

    class RunningProcess:
        def poll(self):
            return None

    monkeypatch.setattr(evaluator, "_materialize_report", lambda *_args: report)
    monkeypatch.setattr("self_growing_trader.mt5_runner.parse_mt5_report", lambda _path: object())

    served = evaluator._serve_policy_actions(
        RunningProcess(), report, _context_policy(), policy_id=1, round_index=0
    )

    assert served == 0


def test_evaluator_rejects_completed_report_without_any_policy_bridge_action(monkeypatch, tmp_path) -> None:
    """A no-op report must never be persisted as a candidate evaluation."""
    runner = tmp_path / "runner"
    _write_fake_ea_artifacts(runner)
    (runner / "MQL5" / "Files").mkdir(parents=True)
    (runner / "MQL5" / "Profiles" / "Tester").mkdir(parents=True)
    (runner / "Reports").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")

    class CompletedWithoutBridge:
        def __init__(self, _command, cwd):
            report = runner / "Reports" / "he_chuyen_gia_mt5_round_0000_evaluation_1.htm"
            report.write_text(
                "Total Net Profit: 0.00 Profit Factor: 0.00 Total Trades: 0 "
                "Balance Drawdown Maximal: 0.00 (0.00%) Equity Drawdown Maximal: 0.00 (0.00%)",
                encoding="utf-16",
            )

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.Popen", CompletedWithoutBridge)
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=runner, timeout_sec=12, shutdown_terminal=False, experience_journal=None,
    ))

    with pytest.raises(RuntimeError, match="zero policy actions"):
        evaluator.evaluate_policy(1, _context_policy(), round=0)


def test_execution_trace_observatory_binds_fixed_geometry_before_final_credit(tmp_path) -> None:
    trace_audit = tmp_path / "trace-observatory.jsonl"
    (tmp_path / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        experience_journal=None,
        execution_trace_observatory=str(trace_audit),
    ))
    lifecycle = MT5PositionLifecycle(
        position_ticket="P-8", entry_time=None, exit_time=None, net_pnl=10.0,
        gross_profit=10.0, commission=0.0, swap=0.0, realized_r=0.5,
        duration_bars=6.0, close_reason="SL", trace_id="owner~trace-8",
        final_close_confirmed=True, exit_kind="INITIAL_SL", exit_controller="broker",
        entry_price=100.0, initial_sl_price=90.0, initial_tp_price=120.0,
        final_sl_price=100.0, final_tp_price=120.0,
        metadata={
            "geometry_mode": "fixed_atr_v1", "fixed_sl_atr": 2.5,
            "fixed_tp_atr": 5.0, "fixed_rr": 2.0,
        },
    )

    class Policy:
        def bind_execution_receipt(self, trace_id, receipt):
            self.bound = (trace_id, dict(receipt))
            return {"bound": True, "owner": "owner"}

        def apply_execution_lifecycle_credit(self, trace_id, realized_r, **kwargs):
            self.credited = (trace_id, realized_r, kwargs)
            return 1

    policy = Policy()
    assert evaluator._apply_execution_lifecycle_credit(
        policy=policy, lifecycle=lifecycle, round_index=3, policy_id=7,
        source="unit_test",
    ) == 1
    assert policy.bound[0] == "owner~trace-8"
    assert policy.bound[1]["closed"] is True
    assert policy.bound[1]["close_reason"] == "INITIAL_SL"
    assert policy.bound[1]["broker_close_reason"] == "SL"
    assert policy.bound[1]["geometry_mode"] == "fixed_atr_v1"
    assert policy.bound[1]["fixed_sl_atr"] == 2.5
    assert policy.bound[1]["fixed_tp_atr"] == 5.0
    assert policy.bound[1]["fixed_rr"] == 2.0
    assert policy.credited[0] == "owner~trace-8"
    assert policy.credited[2]["close_reason"] == "INITIAL_SL"
    event = json.loads(trace_audit.read_text(encoding="utf-8").strip())
    assert event["stage"] == "lifecycle_credit"
    assert event["routing"]["bound"] is True
    assert event["credit"] == 1
    assert event["source"] == "mt5_execution_trace_observatory"


def test_execution_trace_preserves_compaction_opt_out_without_debug_override(tmp_path) -> None:
    (tmp_path / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        experience_journal=None,
        execution_trace_observatory_max_bytes=0,
    ))
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=20, hidden_size=8, hidden_layers=6, seed=716)
    policy.ensure_compatible()

    trace = policy.act(np.linspace(-0.2, 0.2, INPUT_COUNT, dtype=float))
    snapshot = evaluator._execution_trace_snapshot(trace)
    module = snapshot["module_action_traces"][0]

    assert len(module["hidden"]) == 8
    assert "direct_rnn_forward_debug" not in module

    audit = tmp_path / "trace.jsonl"
    audit.write_bytes(b'{"old":true}\n' + b"x" * 256)
    before = audit.read_bytes()
    evaluator._compact_execution_trace_observatory(audit)
    assert audit.read_bytes() == before


def test_execution_trace_compaction_keeps_hysteresis_below_limit(tmp_path) -> None:
    (tmp_path / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        experience_journal=None,
        execution_trace_observatory_max_bytes=1_000,
    ))
    audit = tmp_path / "trace.jsonl"
    audit.write_bytes(b"".join(
        json.dumps({"index": index, "payload": "x" * 40}).encode("utf-8") + b"\n"
        for index in range(40)
    ))

    evaluator._compact_execution_trace_observatory(audit)

    retained = audit.read_bytes()
    assert len(retained) <= 750
    assert retained.endswith(b"\n")
    assert all(json.loads(line) for line in retained.splitlines())


def test_execution_trace_can_omit_recurrent_state_but_keep_action_audit(tmp_path) -> None:
    (tmp_path / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(
        runner_dir=tmp_path,
        experience_journal=None,
        execution_trace_observatory_include_recurrent_state=False,
    ))
    trace = SimpleNamespace(
        trace_id="trace-compact",
        selected_idx=1,
        selected_action=np.asarray([0.0, 1.0]),
        action_details={
            "five_action_index": 1,
            "five_sl_index": 2,
            "five_tp_index": 3,
            "five_cell_before": np.ones(4096),
            "five_cell_after": np.ones(4096),
            "five_temporal_tokens_before": np.ones((4, 512)),
            "five_temporal_valid_before": np.ones(4, dtype=bool),
        },
        execution_details={"accepted": True},
        context_audit={},
        module_action_traces=(),
    )

    snapshot = evaluator._execution_trace_snapshot(trace)

    assert snapshot["action_details"]["five_action_index"] == 1
    assert snapshot["action_details"]["five_sl_index"] == 2
    assert snapshot["action_details"]["five_tp_index"] == 3
    assert "five_cell_before" not in snapshot["action_details"]
    assert "five_cell_after" not in snapshot["action_details"]
    assert "five_temporal_tokens_before" not in snapshot["action_details"]
    assert "five_temporal_valid_before" not in snapshot["action_details"]


def test_wall_guard_settles_closed_lifecycles_before_timeout(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, max_wall_sec=1, experience_journal=None)
    )

    lifecycle = MT5PositionLifecycle(
        position_ticket="42",
        entry_time=None,
        exit_time=None,
        net_pnl=12.0,
        gross_profit=12.0,
        commission=0.0,
        swap=0.0,
        realized_r=0.6,
        duration_bars=8.0,
        close_reason="TP",
        trace_id="owner~trace-42",
        final_close_confirmed=True,
    )
    monkeypatch.setattr(
        "self_growing_trader.mt5_runner.parse_mt5_execution_ledger",
        lambda *args, **kwargs: MT5ExecutionSummary(
            ledger_available=True,
            lifecycle_count=1,
            lifecycles=(lifecycle,),
        ),
    )
    preference_path = tmp_path / "self_growing_entry_preference.csv"
    preference_path.write_text(
        "schema_version\ttrace_id\tsequence\tdecision_time\tselected_action\tsample_scope\tpositions_before\torders_before\tbid\task\tatr\tsl_atr\trr\tentry_prices_pipe\tpath_pipe\n"
        "2\towner~trace-pref\t3\t2025.01.01 00:00\t1\tentry_marginal_v2\t2\t1\t100\t101\t1\t1\t2\t101|100|100|102|102|99\t100.5|98.5|99.5|1\n",
        encoding="utf-8",
    )
    def latest_runtime_file(filename):
        if str(filename) == evaluator.config.entry_preference_file:
            return preference_path
        return tmp_path / "ledger.tsv"
    monkeypatch.setattr(evaluator, "_latest_runtime_file", latest_runtime_file)
    ticks = iter((100.0, 102.0))
    monkeypatch.setattr("self_growing_trader.mt5_runner.time.monotonic", lambda: next(ticks))

    class TimeoutPolicy:
        experts = {}
        registry = object()
        nhan_dien = object()

        def reset_runtime_state(self):
            pass

        def apply_execution_lifecycle_credit(self, trace_id, realized_r, **kwargs):
            self.receipt = (trace_id, realized_r, kwargs)
            return 3

        def apply_entry_preference_credit(self, trace_id, record):
            self.preference_receipt = (trace_id, record.sample_scope, record.positions_before)
            return 5

    policy = TimeoutPolicy()

    class RunningProcess:
        def poll(self):
            return None

    with pytest.raises(TimeoutError, match="MT5 max wall clock exceeded"):
        evaluator._serve_policy_actions(RunningProcess(), tmp_path / "report.htm", policy, policy_id=1, round_index=0)

    assert policy.receipt[0] == "owner~trace-42"
    assert policy.receipt[1] == pytest.approx(0.6)
    assert policy.preference_receipt == ("owner~trace-pref", "entry_marginal_v2", 2)
    assert evaluator._last_live_entry_preference_records == 1
    assert evaluator._last_live_entry_preference_updates == 5


def test_five_preference_csv_routes_only_to_pattern_consumer(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, max_wall_sec=1, experience_journal=None)
    )
    preference_path = tmp_path / "self_growing_entry_preference.csv"
    preference_path.write_text(
        "schema_version\ttrace_id\tsequence\tdecision_time\tselected_action\tsample_scope\tpositions_before\torders_before\tbid\task\tatr\tsl_atr\trr\tentry_prices_pipe\tpath_pipe\n"
        "2\tfive:packet:0\t42\t2025.01.01 00:00\t1\tflat_v1\t0\t0\t100\t100.2\t1\t1\t2\t100.2|100|99.5|100.7|100.7|99.5\t102|99.8|101.5|0.2\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        evaluator,
        "_latest_runtime_file",
        lambda filename: preference_path if str(filename) == evaluator.config.entry_preference_file else tmp_path / "missing.tsv",
    )
    ticks = iter((100.0, 102.0))
    monkeypatch.setattr("self_growing_trader.mt5_runner.time.monotonic", lambda: next(ticks))

    class FivePolicy:
        experts = {}
        registry = object()
        nhan_dien = object()

        def reset_runtime_state(self):
            pass

        def apply_five_pattern_path_credit_detailed(self, trace_id, record):
            self.pattern_receipt = (trace_id, record.sequence)
            return {
                "accepted": True, "queued": True, "updates": 1,
                "retryable": False, "reason": "unit_test",
            }

        def apply_entry_preference_credit_detailed(self, trace_id, record):
            raise AssertionError("Five trace must not enter Direct-RNN consumer")

    policy = FivePolicy()

    class RunningProcess:
        def poll(self):
            return None

    with pytest.raises(TimeoutError, match="MT5 max wall clock exceeded"):
        evaluator._serve_policy_actions(
            RunningProcess(), tmp_path / "report.htm", policy, policy_id=1, round_index=0
        )

    assert policy.pattern_receipt == ("five:packet:0", 42)
    assert evaluator._last_live_entry_preference_updates == 1


def test_incremental_csv_tail_giu_lai_dong_dang_ghi_do(tmp_path) -> None:
    from self_growing_trader.mt5_runner import _IncrementalCsvTail

    path = tmp_path / "tail.tsv"
    path.write_text("a\tb\n1\t", encoding="utf-8")
    reader = _IncrementalCsvTail()

    assert reader.read_rows(path) == []
    with path.open("a", encoding="utf-8") as handle:
        handle.write("2\n3\t4\n")

    assert reader.read_rows(path) == [
        {"a": "1", "b": "2"},
        {"a": "3", "b": "4"},
    ]
    assert reader.read_rows(path) == []


def test_bridge_masks_close_and_cancel_actions_when_account_is_flat(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, max_positions=5))
    request = BridgeRequest(sequence=4, inputs=np.zeros(2), positions=0, orders_pipe="")
    outputs = np.zeros(19, dtype=float)
    outputs[7] = 9.0   # CANCEL_ORDERS: invalid with no pending order.
    outputs[10] = 8.0  # CLOSE_ALL: invalid with no position.
    outputs[2] = 1.0   # MARKET_SELL: best valid action.

    policy_action, _ = evaluator._choose_action(outputs)
    bridge_action, _ = evaluator._choose_action(outputs, evaluator._allowed_actions_for_request(request))

    assert policy_action == 7
    assert bridge_action == 2


def test_bridge_normalizes_pending_order_offset_to_valid_direction(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    outputs = np.zeros(19, dtype=float)

    buy_limit = evaluator._normalize_pending_offset(outputs, int(Action.BUY_LIMIT))
    buy_stop = evaluator._normalize_pending_offset(outputs, int(Action.BUY_STOP))
    sell_limit = evaluator._normalize_pending_offset(outputs, int(Action.SELL_LIMIT))
    sell_stop = evaluator._normalize_pending_offset(outputs, int(Action.SELL_STOP))

    # V7 stores a direction-neutral magnitude; the shared decoder applies the
    # action-specific price direction immediately before execution.
    assert buy_limit[15] >= 0.0
    assert sell_stop[15] >= 0.0
    assert buy_stop[15] >= 0.0
    assert sell_limit[15] >= 0.0


def test_bridge_resolves_action_recovery_learner_from_context_owner_expert() -> None:
    learner = SimpleNamespace(cells={})
    expert = SimpleNamespace(policy=SimpleNamespace(action_recovery_learner=learner))

    class Facade:
        def _expert_from_trace(self, trace):
            assert trace.trace_id == "trace-owner"
            return expert

    trace = SimpleNamespace(trace_id="trace-owner", action_details={}, context_audit={})

    assert MT5StrategyTesterEvaluator._owner_action_recovery_learner(Facade(), trace) is learner


def test_negative_action_cell_blocks_learned_bad_entry_context() -> None:
    cell = SimpleNamespace(
        prototype=np.asarray([1.0, 0.0, 0.5], dtype=float),
        mean=-0.18,
        lcb=-0.86,
        support=3.8,
    )
    learner = SimpleNamespace(cells={int(Action.MARKET_SELL): [cell]})

    decision = MT5StrategyTesterEvaluator._negative_action_cell_decision(
        learner,
        int(Action.MARKET_SELL),
        np.asarray([1.0, 0.0, 0.5], dtype=float),
    )

    assert decision["blocked"] is True
    assert decision["support"] == pytest.approx(3.8)
    assert decision["lcb"] == pytest.approx(-0.86)


def test_bridge_uses_trace_canonical_context_for_action_recovery_filter() -> None:
    trace = SimpleNamespace(
        canonical_context_key=np.asarray([0.25, -0.5, 1.0], dtype=float),
        context_frame=np.asarray([9.0, 9.0, 9.0], dtype=float),
    )
    request = BridgeRequest(
        sequence=1,
        inputs=np.asarray([999999999.0, -999999999.0, 0.0], dtype=float),
    )

    vector = MT5StrategyTesterEvaluator._trace_recovery_context_vector(trace, request)

    assert vector.tolist() == [0.25, -0.5, 1.0]


def test_bridge_records_position_extrema_when_market_context_changes(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    class Facade:
        def __init__(self) -> None:
            self.events = []

        def record_position_context_extrema(self, payload):
            self.events.append(payload)

    policy = Facade()

    def request(sequence: int, unrealized_r: float) -> BridgeRequest:
        position = ":".join(
            [
                "777",
                "1",
                "0.10",
                "100.0",
                "99.0",
                "103.0",
                str(unrealized_r),
                str(sequence),
                str(max(unrealized_r, 0.0)),
                str(min(unrealized_r, 0.0)),
                "1.0",
                "1.0",
                "0.5",
                "0.5",
                "0.0",
                "0",
                "0.0",
                "CTX_TANG_MANH",
                "CG_GOC_CTX_TANG_MANH",
                "CG_GOC_CTX_TANG_MANH",
                "trace-entry",
            ]
        )
        return BridgeRequest(
            sequence=sequence,
            event_id=f"evt-{sequence}",
            time=f"2025.04.01 00:0{sequence}:00",
            inputs=np.ones(INPUT_COUNT, dtype=float) * sequence,
            positions=1,
            positions_pipe=position,
        )

    trace_a = SimpleNamespace(context_audit={"context_nhan_dien": "CTX_TANG_MANH"})
    trace_b = SimpleNamespace(context_audit={"context_nhan_dien": "CTX_RANGING"})

    evaluator._update_position_context_extrema(policy=policy, request=request(1, -0.30), trace=trace_a)
    evaluator._update_position_context_extrema(policy=policy, request=request(2, 0.80), trace=trace_a)
    assert policy.events == []

    evaluator._update_position_context_extrema(policy=policy, request=request(3, 0.20), trace=trace_b)

    assert len(policy.events) == 1
    event = policy.events[0]
    assert event["position_ticket"] == "777"
    assert event["position_owner_expert"] == "CG_GOC_CTX_TANG_MANH"
    assert event["segment_context"] == "CTX_TANG_MANH"
    assert event["new_context"] == "CTX_RANGING"
    assert event["high"]["unrealized_r"] == pytest.approx(0.80)
    assert event["high"]["sequence"] == 2
    assert event["low"]["unrealized_r"] == pytest.approx(-0.30)
    assert event["low"]["sequence"] == 1
    assert len(event["high"]["observation"]) == INPUT_COUNT


def test_bridge_flushes_position_extrema_when_position_disappears_without_context_change(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    class Facade:
        def __init__(self) -> None:
            self.events = []

        def record_position_context_extrema(self, payload):
            self.events.append(payload)

    policy = Facade()

    def open_request(sequence: int, unrealized_r: float) -> BridgeRequest:
        position = ":".join(
            [
                "888", "1", "0.10", "100.0", "99.0", "103.0",
                str(unrealized_r), str(sequence), str(max(unrealized_r, 0.0)),
                str(min(unrealized_r, 0.0)), "1.0", "1.0", "0.5", "0.5",
                "0.0", "0", "0.0", "CTX_TANG_MANH", "CG_GOC_CTX_TANG_MANH",
                "CG_GOC_CTX_TANG_MANH", "trace-entry",
            ]
        )
        return BridgeRequest(
            sequence=sequence,
            event_id=f"evt-{sequence}",
            time=f"2025.04.01 00:0{sequence}:00",
            inputs=np.ones(INPUT_COUNT, dtype=float) * sequence,
            positions=1,
            positions_pipe=position,
        )

    trace = SimpleNamespace(context_audit={"context_nhan_dien": "CTX_TANG_MANH"})
    evaluator._update_position_context_extrema(policy=policy, request=open_request(1, -0.40), trace=trace)
    evaluator._update_position_context_extrema(policy=policy, request=open_request(2, 0.90), trace=trace)
    close_request = BridgeRequest(
        sequence=3,
        event_id="evt-3",
        time="2025.04.01 00:03:00",
        inputs=np.ones(INPUT_COUNT, dtype=float) * 3,
        positions=0,
        positions_pipe="",
    )
    evaluator._update_position_context_extrema(policy=policy, request=close_request, trace=trace)

    assert len(policy.events) == 1
    event = policy.events[0]
    assert event["reason"] == "position_closed_or_missing"
    assert event["position_ticket"] == "888"
    assert event["recovery_after_low_r"] == pytest.approx(1.30)
    assert event["bars_from_segment_start_to_low"] == 0
    assert event["bars_from_segment_start_to_high"] == 1


def test_bridge_does_not_bootstrap_market_execution_after_persistent_no_trade(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            min_daily_trades=6,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)
    allowed = evaluator._allowed_actions_for_request(request)

    action, applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {"no_trade_streak": 8.0, "unfilled_streak": 0.0},
        allowed,
    )

    assert applied is False
    assert action == int(Action.HOLD)


def test_bridge_never_overrides_a_valid_policy_market_entry_when_starved(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            min_daily_trades=6,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)

    action, applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.MARKET_SELL),
        {"no_trade_streak": 20.0, "unfilled_streak": 0.0},
        evaluator._allowed_actions_for_request(request),
    )

    assert action == int(Action.MARKET_SELL)
    assert applied is False


def test_bridge_bootstrap_never_probes_market_without_policy_entry(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            min_daily_trades=0,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)

    action, applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {"no_trade_streak": 8.0, "unfilled_streak": 0.0},
        evaluator._allowed_actions_for_request(request),
    )

    assert action == int(Action.HOLD)
    assert applied is False


def test_train_entry_probe_can_seed_flat_mt5_training_when_explicitly_enabled(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            train_entry_probe_enabled=True,
            train_entry_probe_after_decisions=3,
            train_entry_probe_interval_decisions=2,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)
    allowed = evaluator._allowed_actions_for_request(request)
    direction = {
        "timing_direction": 1,
        "timing_quality": 0.80,
        "context_direction_prior": 1,
        "context_direction_prior_quality": 0.80,
        "context_direction_prior_applied": True,
    }

    first, first_applied = evaluator._bootstrap_execution_action(request, int(Action.HOLD), {}, allowed, probe_details=direction)
    second, second_applied = evaluator._bootstrap_execution_action(request, int(Action.HOLD), {}, allowed, probe_details=direction)
    third, third_applied = evaluator._bootstrap_execution_action(request, int(Action.HOLD), {}, allowed, probe_details=direction)
    fourth, fourth_applied = evaluator._bootstrap_execution_action(request, int(Action.HOLD), {}, allowed, probe_details=direction)
    fifth, fifth_applied = evaluator._bootstrap_execution_action(request, int(Action.HOLD), {}, allowed, probe_details=direction)

    assert first == int(Action.HOLD)
    assert first_applied is False
    assert second == int(Action.HOLD)
    assert second_applied is False
    assert third == int(Action.MARKET_BUY)
    assert third_applied is True
    assert fourth == int(Action.HOLD)
    assert fourth_applied is False
    assert fifth == int(Action.MARKET_BUY)
    assert fifth_applied is True
    assert evaluator._train_entry_probe_last_source == "directional_buy:timing+context_prior"


def test_train_entry_probe_skips_when_direction_is_missing_or_conflicting(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            train_entry_probe_enabled=True,
            train_entry_probe_after_decisions=1,
            train_entry_probe_interval_decisions=1,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)
    allowed = evaluator._allowed_actions_for_request(request)

    missing, missing_applied = evaluator._bootstrap_execution_action(
        request, int(Action.HOLD), {}, allowed, probe_details={}
    )
    conflicting, conflicting_applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {},
        allowed,
        probe_details={
            "timing_direction": 1,
            "timing_quality": 0.80,
            "context_direction_prior": -1,
            "context_direction_prior_quality": 0.80,
            "context_direction_prior_applied": True,
        },
    )

    assert missing == int(Action.HOLD)
    assert missing_applied is False
    assert conflicting == int(Action.HOLD)
    assert conflicting_applied is False
    assert evaluator._train_entry_probe_count == 0
    assert evaluator._train_entry_probe_last_source == "not_emitted_direction_conflict"


def test_train_entry_probe_cannot_override_local_recovery_veto(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            train_entry_probe_enabled=True,
            train_entry_probe_after_decisions=1,
            train_entry_probe_interval_decisions=1,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)

    action, applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {},
        evaluator._allowed_actions_for_request(request),
        policy_vetoed=True,
    )

    assert action == int(Action.HOLD)
    assert applied is False
    assert evaluator._train_entry_probe_count == 0
    assert evaluator._train_entry_probe_flat_count == 0
    assert evaluator._train_entry_probe_last_source == "not_emitted_policy_veto_without_action"


def test_train_entry_probe_can_rehabilitate_with_directional_action_other_than_veto(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            train_entry_probe_enabled=True,
            train_entry_probe_after_decisions=1,
            train_entry_probe_interval_decisions=1,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)
    allowed = evaluator._allowed_actions_for_request(request)
    buy_direction = {"timing_direction": 1, "timing_quality": 0.80}

    alternative, alternative_applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {},
        allowed,
        policy_vetoed=True,
        policy_vetoed_action=int(Action.BUY_STOP),
        probe_details=buy_direction,
    )

    assert alternative == int(Action.MARKET_BUY)
    assert alternative_applied is True
    assert evaluator._train_entry_probe_last_source == "directional_buy:timing:alternative_to_veto_5"


def test_train_entry_probe_never_reopens_exact_vetoed_action(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            train_entry_probe_enabled=True,
            train_entry_probe_after_decisions=1,
            train_entry_probe_interval_decisions=1,
        )
    )
    request = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0)
    allowed = evaluator._allowed_actions_for_request(request)

    action, applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {},
        allowed,
        policy_vetoed=True,
        policy_vetoed_action=int(Action.MARKET_BUY),
        probe_details={"timing_direction": 1, "timing_quality": 0.80},
    )

    assert action == int(Action.HOLD)
    assert applied is False
    assert evaluator._train_entry_probe_count == 0
    assert evaluator._train_entry_probe_last_source == "not_emitted_same_action_veto"


def test_entry_probe_lifecycle_is_neural_exploration_not_policy_expectancy() -> None:
    trace = SimpleNamespace(
        action_details={
            "execution_bootstrap": True,
            "training_entry_probe": True,
        },
        execution_details={},
    )

    assert ExperienceSuperNetwork._action_origin_for_trace(trace) == "policy_exploration"
    assert ExperienceSuperNetwork._credit_weight_for_origin("policy_exploration") == pytest.approx(0.5)
    assert ExperienceSuperNetwork._credit_weight_for_origin("bootstrap") == pytest.approx(0.0)


def test_enrich_step_trace_merges_policy_evidence_with_bridge_details() -> None:
    trace = SimpleNamespace(
        action_details={
            "entry_timing_extrema_evidence_available": True,
            "entry_timing_memory_score": 0.42,
            "learner_recovery": {"replacement_applied": True},
            "action_name": "POLICY_ACTION",
        },
        execution_details={},
        positions_pipe_after="",
        account_context={},
        account_context_after={},
    )

    enriched = enrich_step_trace(
        trace,
        {
            "action_name": "MARKET_BUY",
            "emitted_action": int(Action.MARKET_BUY),
        },
        positions_pipe="",
    )

    assert enriched.action_details["action_name"] == "MARKET_BUY"
    assert enriched.action_details["emitted_action"] == int(Action.MARKET_BUY)
    assert enriched.action_details["entry_timing_extrema_evidence_available"] is True
    assert enriched.action_details["entry_timing_memory_score"] == pytest.approx(0.42)
    assert enriched.action_details["learner_recovery"]["replacement_applied"] is True


def test_train_entry_probe_resets_daily_quota_from_mt5_request_time(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            train_entry_probe_enabled=True,
            train_entry_probe_after_decisions=1,
            train_entry_probe_interval_decisions=1,
            train_entry_probe_max_per_day=1,
            train_entry_probe_max_per_run=4,
        )
    )
    allowed = evaluator._allowed_actions_for_request(BridgeRequest(sequence=1, inputs=np.zeros(2), positions=0))
    day1 = BridgeRequest(sequence=1, inputs=np.zeros(2), positions=0, time="2025.04.01 00:00")
    day2 = BridgeRequest(sequence=2, inputs=np.zeros(2), positions=0, time="2025.04.02 00:00")
    buy = {"timing_direction": 1, "timing_quality": 0.8}
    sell = {"timing_direction": -1, "timing_quality": 0.8}

    first, first_applied = evaluator._bootstrap_execution_action(day1, int(Action.HOLD), {}, allowed, probe_details=buy)
    second, second_applied = evaluator._bootstrap_execution_action(day1, int(Action.HOLD), {}, allowed, probe_details=buy)
    third, third_applied = evaluator._bootstrap_execution_action(day2, int(Action.HOLD), {}, allowed, probe_details=sell)

    assert first == int(Action.MARKET_BUY)
    assert first_applied is True
    assert second == int(Action.HOLD)
    assert second_applied is False
    assert third == int(Action.MARKET_SELL)
    assert third_applied is True


def test_bridge_cancels_same_pending_order_after_configured_age(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            pending_cancel_after_decisions=3,
            pending_cancel_interval_decisions=2,
        )
    )
    allowed = {int(Action.HOLD), int(Action.CANCEL_ORDERS)}
    pending = BridgeRequest(sequence=96, inputs=np.zeros(2), positions=0, orders_pipe="9:4:0.38:94130.34")

    first, first_applied = evaluator._bootstrap_execution_action(
        pending,
        int(Action.HOLD),
        {},
        allowed,
    )
    second, second_applied = evaluator._bootstrap_execution_action(
        pending,
        int(Action.HOLD),
        {},
        allowed,
    )
    third, third_applied = evaluator._bootstrap_execution_action(
        pending,
        int(Action.HOLD),
        {},
        allowed,
    )
    fourth, fourth_applied = evaluator._bootstrap_execution_action(
        pending,
        int(Action.HOLD),
        {},
        allowed,
    )
    fifth, fifth_applied = evaluator._bootstrap_execution_action(
        pending,
        int(Action.HOLD),
        {},
        allowed,
    )

    assert first == int(Action.HOLD)
    assert first_applied is False
    assert second == int(Action.HOLD)
    assert second_applied is False
    assert third == int(Action.CANCEL_ORDERS)
    assert third_applied is True
    assert fourth == int(Action.HOLD)
    assert fourth_applied is False
    assert fifth == int(Action.CANCEL_ORDERS)
    assert fifth_applied is True


def test_bridge_pending_cancel_age_resets_when_pending_order_changes(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            pending_cancel_after_decisions=3,
            pending_cancel_interval_decisions=2,
        )
    )
    allowed = {int(Action.HOLD), int(Action.CANCEL_ORDERS)}

    old_pending = BridgeRequest(sequence=1, inputs=np.zeros(2), positions=0, orders_pipe="9:4:0.38:94130.34")
    new_pending = BridgeRequest(sequence=2, inputs=np.zeros(2), positions=0, orders_pipe="10:4:0.40:94200.00")
    evaluator._bootstrap_execution_action(old_pending, int(Action.HOLD), {}, allowed)
    evaluator._bootstrap_execution_action(old_pending, int(Action.HOLD), {}, allowed)
    action, applied = evaluator._bootstrap_execution_action(new_pending, int(Action.HOLD), {}, allowed)

    assert action == int(Action.HOLD)
    assert applied is False


def test_entry_single_position_guard_blocks_scale_in_entries(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            # Lifecycle isolation is independent from optional daily quotas.
            enforce_entry_limits=False,
            entry_guard_single_position=True,
            max_positions=5,
        )
    )

    allowed_with_position = evaluator._allowed_actions_for_request(
        BridgeRequest(sequence=2, inputs=np.zeros(2), positions=1, positions_pipe="1:1:0.1:1:0:2:-0.1:3")
    )
    allowed_with_pending = evaluator._allowed_actions_for_request(
        BridgeRequest(sequence=3, inputs=np.zeros(2), positions=0, orders_pipe="ticket=123")
    )

    assert not (allowed_with_position & evaluator._entry_actions())
    assert not (allowed_with_pending & evaluator._entry_actions())
    assert int(Action.HOLD) in allowed_with_position
    assert int(Action.HOLD) in allowed_with_pending


def test_context_expert_mt5_contract_forwards_single_position_guard() -> None:
    config = replace(
        AppConfig(),
        learning=LearningIntegrityConfig(entry_guard_single_position=True),
    )

    contract = _mt5_execution_contract_kwargs(config)

    assert contract["entry_guard_single_position"] is True


def test_context_expert_mt5_contract_preserves_native_mtf_feature_mode() -> None:
    config = replace(
        AppConfig(),
        features=FeatureConfig(mtf_scaling_mode="native_tf_atr_v2"),
    )

    contract = _mt5_execution_contract_kwargs(config)

    assert contract["feature_mtf_scaling_mode"] == "native_tf_atr_v2"


def test_context_expert_mt5_contract_contains_only_backtest_config_fields() -> None:
    contract = _mt5_execution_contract_kwargs(AppConfig())
    allowed = {item.name for item in fields(MT5BacktestConfig)}

    assert set(contract).issubset(allowed)


def test_visual_keep_open_finalizes_when_report_exists_and_tester_worker_ended(
    tmp_path, monkeypatch
) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, visual=True, shutdown_terminal=False)
    )
    report = runner / "Reports" / "done.htm"
    monkeypatch.setattr(evaluator, "_runner_tester_is_alive", lambda: False)

    assert evaluator._tester_report_ready_for_finalize(
        report,
        process_poll_status=None,
        now=10.1,
        last_request_seen=8.0,
    ) is True

    monkeypatch.setattr(evaluator, "_runner_tester_is_alive", lambda: True)
    assert evaluator._tester_report_ready_for_finalize(
        report,
        process_poll_status=None,
        now=10.1,
        last_request_seen=8.0,
    ) is False


def test_entry_loss_guard_blocks_entries_after_bad_day(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            enforce_entry_limits=True,
            entry_loss_guard_enabled=True,
            entry_loss_guard_min_closed=6,
            entry_loss_guard_max_closed_loss_r=0.25,
            entry_loss_guard_consecutive_losses=4,
        )
    )

    assert evaluator._entry_loss_guard_blocks_entries(
        closed_count=6,
        closed_net_r=-0.26,
        consecutive_losses=1,
    )
    assert evaluator._entry_loss_guard_blocks_entries(
        closed_count=3,
        closed_net_r=-0.10,
        consecutive_losses=4,
    )
    assert not evaluator._entry_loss_guard_blocks_entries(
        closed_count=5,
        closed_net_r=-0.50,
        consecutive_losses=2,
    )


def test_runtime_fitness_guard_blocks_immediate_exit_storm() -> None:
    guard = RuntimeFitnessGuard(
        enabled=True,
        rolling_window=10,
        min_closed=4,
        min_score=-0.30,
        max_immediate_exit_ratio=0.60,
        max_daily_loss_r=3.0,
        max_daily_trades=300,
        max_drawdown_pct=20.0,
        initial_equity=10000.0,
        min_holding_bars=3,
    )

    for index in range(4):
        guard.record_execution_row(
            {
                "closed": "1",
                "position_final_close": "1",
                "position_ticket": str(index),
                "deal_ticket": str(index + 100),
                "net_profit": "-1.0",
                "initial_risk": "10.0",
                "age_bars": "1",
                "close_reason": "EXPERT",
                "logged_at": "2025.04.01 00:00:00",
            },
            drawdown_pct=2.0,
        )

    assert guard.blocks_entries(drawdown_pct=2.0)
    assert "rolling_fitness" in guard.active_reason or "immediate_exit_ratio" in guard.active_reason
    snapshot = guard.snapshot()
    assert snapshot["active"] is True
    assert snapshot["rolling_count"] == 4
    assert snapshot["immediate_exit_ratio"] == 1.0


def test_runtime_fitness_guard_hard_blocks_on_drawdown_before_min_closed() -> None:
    guard = RuntimeFitnessGuard(
        enabled=True,
        rolling_window=10,
        min_closed=20,
        max_drawdown_pct=20.0,
    )

    assert guard.blocks_entries(drawdown_pct=20.0)
    assert "drawdown_pct" in guard.active_reason


def test_runtime_fitness_guard_blocks_on_total_run_loss() -> None:
    guard = RuntimeFitnessGuard(
        enabled=True,
        rolling_window=10,
        min_closed=20,
        max_daily_loss_r=99.0,
        max_run_loss_r=2.0,
        max_drawdown_pct=20.0,
    )

    for index in range(3):
        guard.record_execution_row(
            {
                "closed": "1",
                "position_final_close": "1",
                "position_ticket": str(index),
                "deal_ticket": str(index + 100),
                "net_profit": "-100.0",
                "initial_risk": "100.0",
                "age_bars": "5",
                "close_reason": "SL",
                "logged_at": f"2025.04.0{index + 1} 00:00:00",
            },
            drawdown_pct=2.0,
        )

    assert guard.blocks_entries(drawdown_pct=2.0)
    assert "run_loss_r" in guard.active_reason
    assert guard.snapshot()["run_net_r"] == pytest.approx(-3.0)


def test_flat_idle_observation_is_recovery_context_not_execution_confirmation() -> None:
    state = ExperienceStarvationState()

    state.observe_flat_idle()
    state.observe_flat_idle()
    preview = state.preview(BridgeRequest(sequence=2, inputs=np.zeros(120), positions=0), min_daily_trades=0)

    assert preview["flat_idle_streak"] == 2.0
    assert preview["no_trade_streak"] == 2.0
    assert preview["unfilled_streak"] == 0.0
    assert preview["intrinsic_reward"] == 0.0
    assert preview["opened_or_filled"] == 0.0


def test_bridge_bootstrap_does_not_scale_in_without_policy_entry(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(
            runner_dir=runner,
            max_positions=5,
            min_daily_trades=6,
        )
    )
    request = BridgeRequest(sequence=16, inputs=np.zeros(2), positions=1, orders_pipe="")

    action, applied = evaluator._bootstrap_execution_action(
        request,
        int(Action.HOLD),
        {"no_trade_streak": 8.0, "unfilled_streak": 0.0},
        evaluator._allowed_actions_for_request(request),
    )

    assert applied is False
    assert action == int(Action.HOLD)


def test_bridge_bootstrap_stops_only_at_max_positions_or_pending_order(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, max_positions=5, min_daily_trades=6)
    )
    starved = {"no_trade_streak": 32.0, "unfilled_streak": 32.0}

    full = BridgeRequest(sequence=32, inputs=np.zeros(2), positions=5, orders_pipe="")
    full_action, full_applied = evaluator._bootstrap_execution_action(
        full, int(Action.HOLD), starved, evaluator._allowed_actions_for_request(full)
    )
    assert full_action == int(Action.HOLD)
    assert full_applied is False

    pending = BridgeRequest(sequence=32, inputs=np.zeros(2), positions=1, orders_pipe="ticket=123")
    pending_action, pending_applied = evaluator._bootstrap_execution_action(
        pending, int(Action.HOLD), starved, evaluator._allowed_actions_for_request(pending)
    )
    assert pending_action == int(Action.HOLD)
    assert pending_applied is False


def test_execution_confirmation_resets_scale_in_cooldown_before_next_action() -> None:
    state = ExperienceStarvationState(no_trade_streak=8, unfilled_streak=4, last_day_trades=0)
    before = BridgeRequest(sequence=8, inputs=np.zeros(2), positions=0, day_trades=0)
    after = BridgeRequest(sequence=9, inputs=np.zeros(2), positions=1, day_trades=1)

    assert state.confirm_execution(before, after) is True
    assert state.no_trade_streak == 0
    assert state.unfilled_streak == 0
    assert state.last_day_trades == 1



def test_bridge_v4_does_not_credit_hold_from_an_unmatched_server_close() -> None:
    before = BridgeRequest(
        sequence=10,
        inputs=np.zeros(2),
        schema_version=4,
        positions=1,
        positions_pipe="101:1:0.01:100:99:103:1.2:5",
    )
    after = BridgeRequest(
        sequence=11,
        inputs=np.zeros(2),
        schema_version=4,
        positions=0,
        execution_pipe="ack_sequence=9|accepted=1|closed=1|close_reason=SL|position_ticket=101",
    )

    details = MT5StrategyTesterEvaluator._execution_details(before, after, requested_action=int(Action.HOLD))

    assert details["ledger_required"] is True
    assert details["positions_changed"] is True
    assert details["acknowledged"] is False
    assert details["execution_confirmed"] is False
    assert details["memory_eligible"] is False


def test_bridge_v4_credits_only_a_matching_transaction_acknowledgement() -> None:
    before = BridgeRequest(sequence=10, inputs=np.zeros(2), schema_version=4, positions=0)
    after = BridgeRequest(
        sequence=11,
        inputs=np.zeros(2),
        schema_version=4,
        positions=1,
        execution_pipe=(
            "ack_sequence=10|intent_event_id=evt-10|intent_trace_id=trace-10|"
            "requested_action=1|accepted=1|filled=1|retcode=10009|"
            "order_ticket=501|deal_ticket=601|position_ticket=701|"
            "transaction_type=DEAL_ADD|status=FILLED|source=policy"
        ),
    )

    details = MT5StrategyTesterEvaluator._execution_details(before, after, requested_action=int(Action.MARKET_BUY))

    assert details["ack_sequence"] == "10"
    assert details["execution_confirmed"] is True
    assert details["memory_eligible"] is True
    assert details["accepted"] is True
    assert details["filled"] is True
    assert details["placed"] is False
    assert details["cancelled"] is False
    assert details["modified"] is False
    assert details["closed"] is False
    assert details["order_ticket"] == "501"
    assert details["deal_ticket"] == "601"
    assert details["position_ticket"] == "701"
    assert details["intent_trace_id"] == "trace-10"


def test_closed_execution_reward_prefers_trade_pnl_over_account_equity_drift() -> None:
    stop_loss = {
        "profit": "-111.92",
        "commission": "0.0",
        "swap": "0.0",
        "initial_risk": "112.0",
        "close_reason": "SL",
    }
    take_profit = {
        "profit": "216.77",
        "commission": "0.0",
        "swap": "0.0",
        "initial_risk": "112.0",
        "close_reason": "TP",
    }

    sl_reward = _closed_execution_reward(stop_loss, initial_equity=10_000.0, entry_equity=10_000.0, exit_equity=10_500.0)
    tp_reward = _closed_execution_reward(take_profit, initial_equity=10_000.0, entry_equity=10_500.0, exit_equity=10_300.0)

    assert sl_reward < 0.0
    assert tp_reward > 0.0
    assert sl_reward == pytest.approx(-0.9992857143)
    assert tp_reward == pytest.approx(1.9354464286)


def test_closed_execution_reward_infers_sl_risk_when_pending_ledger_has_zero_initial_risk() -> None:
    stop_loss = {
        "profit": "-97.93",
        "commission": "0.0",
        "swap": "0.0",
        "initial_risk": "0.0",
        "close_reason": "SL",
    }

    reward = _closed_execution_reward(stop_loss, initial_equity=10_000.0, entry_equity=10_000.0, exit_equity=9_902.07)

    assert reward == pytest.approx(-1.0)


def test_closed_execution_reward_infers_tp_risk_when_pending_ledger_has_zero_initial_risk() -> None:
    take_profit = {
        "profit": "132.0",
        "commission": "0.0",
        "swap": "0.0",
        "initial_risk": "0.0",
        "close_reason": "TP",
    }

    reward = _closed_execution_reward(
        take_profit,
        initial_equity=10_000.0,
        technical_min_rr=1.2,
    )

    assert reward == pytest.approx(1.2)


def test_early_expert_close_ack_merges_final_deal_risk_before_learning() -> None:
    early_ack = {
        "closed": True,
        "position_ticket": "701",
        "profit": "-83.57",
        "commission": "0.0",
        "swap": "0.0",
        "initial_risk": "",
        "close_reason": "EXPERT",
    }
    rows = [
        {
            "closed": "1",
            "position_final_close": "1",
            "position_ticket": "701",
            "transaction_type": "DEAL_ADD",
            "profit": "-83.57",
            "commission": "0.0",
            "swap": "0.0",
            "initial_risk": "103.125",
            "close_reason": "EXPERT",
            "age_bars": "63",
            "exit_kind": "POLICY_CLOSE_WORST",
            "exit_controller": "policy",
            "initial_sl_price": "90.0",
            "final_sl_price": "95.0",
        },
        {
            "closed": "1",
            "position_final_close": "1",
            "position_ticket": "999",
            "transaction_type": "DEAL_ADD",
            "profit": "1000.0",
            "initial_risk": "1.0",
        },
    ]

    merged = _merge_final_close_receipt(early_ack, rows)
    reward = _closed_execution_reward(merged, initial_equity=10_000.0)

    assert merged["final_close_receipt_merged"] is True
    assert merged["initial_risk"] == "103.125"
    assert merged["exit_kind"] == "POLICY_CLOSE_WORST"
    assert merged["exit_controller"] == "policy"
    assert merged["initial_sl_price"] == "90.0"
    assert merged["final_sl_price"] == "95.0"
    assert reward == pytest.approx(-83.57 / 103.125)


def test_legacy_close_row_falls_back_to_position_ticket_when_ack_is_system() -> None:
    open_traces = {
        46: {"position_ticket": "10", "action": int(Action.MARKET_BUY)},
        51: {"position_ticket": "12", "action": int(Action.MARKET_SELL)},
    }
    close_row = {
        "ack_sequence": "0",
        "position_ticket": "10",
        "source": "max_holding",
        "closed": "1",
        "profit": "236.09",
    }

    assert _legacy_ack_sequence_for_close_row(close_row, open_traces) == 46
    assert _legacy_ack_sequence_for_close_row({"ack_sequence": "51", "position_ticket": "12"}, open_traces) == 51
    assert _legacy_ack_sequence_for_close_row({"ack_sequence": "0", "position_ticket": "999"}, open_traces) is None


def test_system_close_uses_owner_trace_registered_before_delayed_fill() -> None:
    open_traces: dict[int, dict[str, object]] = {}
    request = SimpleNamespace(sequence=46)
    trace = SimpleNamespace(trace_id="entry-trace-46")

    sequence = _register_legacy_open_trace(
        open_traces,
        request=request,
        trace=trace,
        action=int(Action.BUY_STOP),
        execution_details={"accepted": True, "placed": True, "filled": False},
        score=0.7,
        confidence=0.8,
        equity=10_000.0,
        drawdown=0.0,
        spread_points=12.0,
    )

    assert sequence == 46
    assert open_traces[46]["trace_id"] == "entry-trace-46"
    assert _legacy_ack_sequence_for_close_row(
        {
            "ack_sequence": "0",
            "position_ticket": "",
            "trace_id": "",
            "position_owner_trace_id": "entry-trace-46",
            "action_name": "SYSTEM",
            "closed": "1",
        },
        open_traces,
    ) == 46


def test_bridge_exposes_manage_position_only_when_a_position_exists(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    outputs = np.zeros(20, dtype=float)
    outputs[int(Action.MANAGE_POSITION)] = 10.0
    outputs[int(Action.HOLD)] = 1.0

    flat_action, _ = evaluator._choose_action(
        outputs,
        evaluator._allowed_actions_for_request(BridgeRequest(sequence=1, inputs=np.zeros(2), positions=0)),
    )
    positioned_action, _ = evaluator._choose_action(
        outputs,
        evaluator._allowed_actions_for_request(
            BridgeRequest(
                sequence=2, inputs=np.zeros(2), positions=1,
                positions_pipe="701:1:0.01:100:99:103:0.1:2",
            )
        ),
    )

    assert flat_action == int(Action.HOLD)
    assert positioned_action == int(Action.MANAGE_POSITION)


def test_bridge_v7_pending_offset_uses_parameter_index_15(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    outputs = np.zeros(20, dtype=float)

    buy_limit = evaluator._normalize_pending_offset(outputs, int(Action.BUY_LIMIT))
    buy_stop = evaluator._normalize_pending_offset(outputs, int(Action.BUY_STOP))

    assert buy_limit[15] >= 0.0
    assert buy_stop[15] >= 0.0
    assert buy_limit[14] == 0.0


def test_mt5_credit_learned_v15_contract_requires_trace_complete_final_close_path() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").read_text(encoding="utf-8")
    integration = (root / "docs" / "MT5_EXECUTION_CONTRACT_LEARNED_GEOMETRY_V15.vi.md").read_text(encoding="utf-8")

    assert "void OnTradeTransaction(" in source
    assert "ownerTraceId" in source
    assert "positionFinalClose" in source
    assert "initialRisk" in source
    assert "intent_context" in source
    assert "position_owner_trace_id" in source
    assert "initialRiskDistancePrice" in source
    assert "POST_EXIT_OBSERVATION" in source
    assert "exit_trace_id" in source
    assert "POSITION_TRAJECTORY" in source
    assert "FixedSlAtr" in source
    assert "FixedTpAtr" in source
    assert "learned_factorized_atr_v1" in source
    assert "TRADE_ACTION_SLTP" in source
    assert "five_trailing_head" in source
    assert "trailingAtr * entryAtr" in source
    assert "g_lifecycleInitialAtr[lifecycleIdx]" in source
    assert "OnTradeTransaction" in source
    assert "checkpoint" in integration.lower()
    assert "proposed_trailing_atr" in integration
    assert "learned_factorized_atr_v1" in integration


def test_mql5_pending_orders_are_normalized_against_broker_stop_level() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").read_text(encoding="utf-8")

    assert "double PendingMinDistance()" in source
    assert "SYMBOL_TRADE_STOPS_LEVEL" in source
    assert "double NormalizePendingEntryPrice(" in source
    assert "entry = NormalizePendingEntryPrice(type, entry);" in source


def test_mql5_context_owner_packet_does_not_use_five_long_entry_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").read_text(encoding="utf-8")

    assert "bool FiveAuthorityAllowsAction(int agentId, int action, int agentCount = FIVE_AGENT_COUNT)" in source
    assert "if(agentCount == 1)" in source
    assert "return action >= ACTION_HOLD && action <= ACTION_MANAGE_POSITION;" in source
    assert "FiveAuthorityAllowsAction(row.agentId, row.action, row.agentCount)" in source
    assert "FiveAuthorityAllowsAction(rows[i].agentId, rows[i].action, rows[i].agentCount)" in source


def test_runner_terminal_probe_is_false_for_unstarted_isolated_runner(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))

    assert evaluator._runner_terminal_is_alive() is False


def test_close_runner_processes_waits_for_stable_shutdown(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    captured: dict[str, object] = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["timeout"] = kwargs.get("timeout")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("self_growing_trader.mt5_runner.subprocess.run", fake_run)

    evaluator.close_runner_processes()

    script = str(captured["command"][-1])
    assert "$emptyChecks -ge 20" in script
    assert "metatester64.exe" in script
    assert "AddSeconds(12)" in script
    assert captured["timeout"] == 16


def test_close_runner_processes_fails_if_runner_survives(monkeypatch, tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner))
    monkeypatch.setattr(
        "self_growing_trader.mt5_runner.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=1),
    )

    with pytest.raises(RuntimeError, match="van con terminal/tester"):
        evaluator.close_runner_processes()


def _write_execution_ledger_v5(path: Path, rows: list[dict[str, object]]) -> None:
    header = [
        "schema_version", "logged_at", "ack_sequence", "event_id", "trace_id", "requested_action", "action_name", "source",
        "transaction_type", "status", "accepted", "placed", "filled", "cancelled", "modified", "closed", "rejected", "retcode",
        "order_ticket", "deal_ticket", "position_ticket", "position_owner_context", "position_owner_expert", "position_owner_policy", "position_owner_state", "position_owner_reason",
        "deal_entry", "deal_reason", "close_reason", "volume", "price", "profit", "commission", "swap",
        "position_final_close", "initial_risk", "age_bars",
        "exit_kind", "exit_controller", "exit_trace_id", "exit_requested_action",
        "post_exit_horizon_bars", "post_exit_hold_r", "geometry_entry_pipe", "geometry_path_pipe",
    ]
    lines = ["\t".join(header)]
    for row in rows:
        lines.append("\t".join(str(row.get(key, "")) for key in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _v5_event(**overrides: object) -> dict[str, object]:
    event: dict[str, object] = {
        "schema_version": 5,
        "logged_at": "2025.01.01 00:00:00",
        "ack_sequence": 1,
        "event_id": "evt-1",
        "trace_id": "trace-entry-1",
        "requested_action": 1,
        "action_name": "MARKET_BUY",
        "source": "policy",
        "transaction_type": "DEAL_ADD",
        "status": "FILLED",
        "accepted": 1,
        "placed": 0,
        "filled": 1,
        "cancelled": 0,
        "modified": 0,
        "closed": 0,
        "rejected": 0,
        "retcode": 10009,
        "order_ticket": 1001,
        "deal_ticket": 2001,
        "position_ticket": 3001,
        "deal_entry": 0,
        "deal_reason": 0,
        "close_reason": "",
        "volume": 0.01,
        "price": 100.0,
        "profit": 0.0,
        "commission": 0.0,
        "swap": 0.0,
        "position_final_close": 0,
        "initial_risk": 10.0,
        "age_bars": 0.0,
    }
    event.update(overrides)
    return event


def test_execution_ledger_reads_final_deal_account_mark_from_geometry_pipe(tmp_path) -> None:
    ledger = tmp_path / "self_growing_execution_account_mark.csv"
    entry_geometry = "0.4|1|2.5|2|5|100.0|99.0|102.0|0.4|fixed_atr_v1|2.5|5.0|2.0|0.0|0"
    final_geometry = entry_geometry + "|9975.0|9974.5"
    rows = [
        _v5_event(geometry_entry_pipe=entry_geometry),
        _v5_event(
            logged_at="2025.01.01 00:05:00",
            filled=0,
            closed=1,
            deal_entry=1,
            profit=-25.0,
            position_final_close=1,
            close_reason="SL",
            age_bars=5.0,
            geometry_entry_pipe=final_geometry,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    lifecycle = summary.lifecycles[0]
    assert lifecycle.final_close_confirmed is True
    assert lifecycle.account_context_after_close == {
        "balance": pytest.approx(9975.0),
        "equity": pytest.approx(9974.5),
        "sequence": pytest.approx(1.0),
    }


def test_execution_ledger_marks_reused_same_batch_account_marks_ambiguous(tmp_path) -> None:
    """A shared terminal snapshot must not be attributed to two positions."""

    ledger = tmp_path / "self_growing_execution_ambiguous_account_mark.csv"
    geometry = "0.4|1|2.5|2|5|100.0|99.0|102.0|0.4|fixed_atr_v1|2.5|5.0|2.0|0.0|0|9975.0|9975.0"
    rows = [
        _v5_event(position_ticket=3001, geometry_entry_pipe=geometry),
        _v5_event(
            logged_at="2025.01.01 00:05:00", position_ticket=3001,
            deal_ticket=2101, filled=0, closed=1, deal_entry=1,
            profit=-10.0, position_final_close=1, close_reason="SL",
            geometry_entry_pipe=geometry,
        ),
        _v5_event(position_ticket=3002, order_ticket=1002, deal_ticket=2002,
                  geometry_entry_pipe=geometry),
        _v5_event(
            logged_at="2025.01.01 00:05:00", position_ticket=3002,
            order_ticket=1002, deal_ticket=2102, filled=0, closed=1,
            deal_entry=1, profit=-12.0, position_final_close=1,
            close_reason="SL", geometry_entry_pipe=geometry,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    assert summary.lifecycle_count == 2
    assert {
        str(item.metadata["account_mark_attribution"])
        for item in summary.lifecycles
    } == {"ambiguous_same_close_batch"}


def test_execution_ledger_keeps_unique_final_deal_cashflow_without_snapshot(tmp_path) -> None:
    """Deal cashflow remains attributable when account snapshot is unavailable."""

    ledger = tmp_path / "self_growing_execution_unique_cashflow.csv"
    entry_geometry = "0.4|1|2.5|2|5|100.0|99.0|102.0|0.4|fixed_atr_v1|2.5|5.0|2.0|0.0|0"
    rows = [
        _v5_event(position_ticket=3001, geometry_entry_pipe=entry_geometry),
        _v5_event(
            logged_at="2025.01.01 00:05:00",
            position_ticket=3001,
            deal_ticket=2101,
            filled=0,
            closed=1,
            deal_entry=1,
            profit=-25.0,
            commission=0.0,
            swap=0.0,
            position_final_close=1,
            close_reason="SL",
            geometry_entry_pipe=entry_geometry,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    lifecycle = summary.lifecycles[0]
    assert lifecycle.account_context_after_close == {}
    assert lifecycle.account_cashflow_delta == pytest.approx(-25.0)
    assert lifecycle.account_cashflow_deal_ticket == "2101"
    assert lifecycle.metadata["account_cashflow_attribution"] == "unique_final_deal"


def test_execution_ledger_v5_aggregates_partial_close_by_position_and_keeps_trace(tmp_path) -> None:
    ledger = tmp_path / "self_growing_execution.csv"
    rows = [
        _v5_event(),
        _v5_event(
            logged_at="2025.01.01 00:03:00", deal_ticket=2002, filled=0, closed=1,
            deal_entry=1, profit=4.0, commission=-0.20, swap=-0.10, age_bars=3.0,
            close_reason="EXPERT", position_final_close=0,
        ),
        _v5_event(
            logged_at="2025.01.01 00:05:00", deal_ticket=2003, filled=0, closed=1,
            deal_entry=1, profit=8.0, commission=-0.20, swap=-0.10, age_bars=5.0,
            close_reason="TP", position_final_close=1,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0, min_holding_bars=3)

    assert summary.schema_version == 5
    assert summary.lifecycle_count == 1
    lifecycle = summary.lifecycles[0]
    assert lifecycle.close_count == 2
    assert lifecycle.trace_id == "trace-entry-1"
    assert lifecycle.entry_sequence == 1
    assert lifecycle.final_close_confirmed is True
    assert lifecycle.net_pnl == pytest.approx(11.4)
    assert lifecycle.realized_r == pytest.approx(1.14)
    assert summary.risk_coverage == 1.0
    assert summary.immediate_exit_ratio == 0.0


def test_execution_ledger_uses_horizon_receipt_age_when_schema_field_is_zero(tmp_path) -> None:
    """Schema-12 HORIZON rows use age_bars until the EA field is repaired."""

    ledger = tmp_path / "self_growing_execution.csv"
    rows = [
        _v5_event(),
        _v5_event(
            logged_at="2025.01.01 00:04:00", filled=0, closed=1,
            deal_entry=1, profit=4.0, position_final_close=1,
            close_reason="EXPERT", age_bars=4.0,
            exit_kind="POLICY_CLOSE_BEST", exit_controller="policy",
            exit_trace_id="trace-manager-1", exit_requested_action=9,
        ),
        _v5_event(
            logged_at="2025.01.01 00:16:00", trace_id="", requested_action=-1,
            action_name="SYSTEM", source="unknown", transaction_type="POST_EXIT_OBSERVATION",
            status="HORIZON", filled=0, closed=0, accepted=1,
            position_final_close=0, close_reason="POST_EXIT_HORIZON", age_bars=12.0,
            exit_kind="POLICY_CLOSE_BEST", exit_controller="policy",
            exit_trace_id="trace-manager-1", exit_requested_action=9,
            post_exit_horizon_bars=0, post_exit_hold_r=0.35,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    lifecycle = summary.lifecycles[0]
    assert lifecycle.exit_trace_id == "trace-manager-1"
    assert lifecycle.post_exit_horizon_bars == 12
    assert lifecycle.post_exit_hold_r == pytest.approx(0.35)


def test_execution_ledger_keeps_broker_exit_counterfactual_without_exit_trace(tmp_path) -> None:
    """A broker SL still yields hold-to-horizon evidence for the entry owner."""

    ledger = tmp_path / "self_growing_execution_broker_counterfactual.csv"
    rows = [
        _v5_event(),
        _v5_event(
            logged_at="2025.01.01 00:04:00", filled=0, closed=1,
            deal_entry=1, profit=-10.0, position_final_close=1,
            close_reason="SL", age_bars=4.0,
            exit_kind="INITIAL_SL", exit_controller="broker",
            exit_trace_id="", exit_requested_action=-1,
        ),
        _v5_event(
            logged_at="2025.01.01 00:16:00", trace_id="", requested_action=-1,
            action_name="SYSTEM", source="unknown", transaction_type="POST_EXIT_OBSERVATION",
            status="HORIZON", filled=0, closed=0, accepted=1,
            position_final_close=0, close_reason="POST_EXIT_HORIZON", age_bars=12.0,
            exit_kind="INITIAL_SL", exit_controller="broker",
            exit_trace_id="", exit_requested_action=-1,
            post_exit_horizon_bars=12, post_exit_hold_r=0.65,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    lifecycle = summary.lifecycles[0]
    assert lifecycle.exit_kind == "INITIAL_SL"
    assert lifecycle.exit_trace_id == ""
    assert lifecycle.post_exit_horizon_bars == 12
    assert lifecycle.post_exit_hold_r == pytest.approx(0.65)


def test_execution_ledger_recovers_same_tick_sl_and_tp_risk_with_audit_metadata(tmp_path) -> None:
    ledger = tmp_path / "self_growing_execution.csv"
    rows = [
        _v5_event(position_ticket=3001, initial_risk=0.0),
        _v5_event(
            position_ticket=3001, filled=0, closed=1, deal_entry=1,
            profit=-119.15, initial_risk=0.0, age_bars=0.0,
            close_reason="SL", position_final_close=1,
        ),
        _v5_event(position_ticket=3002, order_ticket=1002, deal_ticket=2002, initial_risk=0.0),
        _v5_event(
            position_ticket=3002, order_ticket=1002, deal_ticket=2003,
            filled=0, closed=1, deal_entry=1, profit=132.0,
            initial_risk=0.0, age_bars=0.0, close_reason="TP",
            position_final_close=1,
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(
        ledger,
        initial_equity=10_000.0,
        technical_min_rr=1.2,
    )

    assert summary.lifecycle_count == 2
    assert summary.risk_coverage == 1.0
    assert summary.ledger_integrity == pytest.approx(0.835)
    by_ticket = {item.position_ticket: item for item in summary.lifecycles}
    assert by_ticket["3001"].realized_r == pytest.approx(-1.0)
    assert by_ticket["3001"].metadata["initial_risk_inference_method"] == "same_tick_sl_terminal_pnl"
    assert by_ticket["3002"].initial_risk == pytest.approx(110.0)
    assert by_ticket["3002"].realized_r == pytest.approx(1.2)
    assert by_ticket["3002"].metadata["initial_risk_inferred"] is True


def test_execution_ledger_keeps_position_owner_metadata_on_close(tmp_path) -> None:
    ledger = tmp_path / "self_growing_execution.csv"
    rows = [
        _v5_event(
            position_owner_context="CTX_TANG_MANH",
            position_owner_expert="CG_GOC_CTX_TANG_MANH",
            position_owner_policy="CG_GOC_CTX_TANG_MANH",
            position_owner_state="HOAT_DONG_HAN_CHE",
            position_owner_reason="regime=TANG_MANH_conf=0.88",
        ),
        _v5_event(
            logged_at="2025.01.01 00:05:00",
            deal_ticket=2003,
            filled=0,
            closed=1,
            deal_entry=1,
            profit=8.0,
            close_reason="TP",
            position_final_close=1,
            age_bars=5.0,
            position_owner_context="CTX_TANG_MANH",
            position_owner_expert="CG_GOC_CTX_TANG_MANH",
            position_owner_policy="CG_GOC_CTX_TANG_MANH",
            position_owner_state="HOAT_DONG_HAN_CHE",
            position_owner_reason="regime=TANG_MANH_conf=0.88",
        ),
    ]
    _write_execution_ledger_v5(ledger, rows)

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0, min_holding_bars=3)

    lifecycle = summary.lifecycles[0]
    assert lifecycle.trace_id == "trace-entry-1"
    assert lifecycle.metadata["position_owner_context"] == "CTX_TANG_MANH"
    assert lifecycle.metadata["position_owner_expert"] == "CG_GOC_CTX_TANG_MANH"


def test_fitness_v5_penalizes_immediate_exit_and_requires_ledger_when_requested(tmp_path) -> None:
    metrics = _fitness_metrics(
        profit_factor=1.40, win_rate=58.0, total_trades=16, net_profit=100.0,
        drawdown=2.0, avg_win=1.4, avg_loss=-1.0,
    )
    good = []
    churn = []
    for index in range(8):
        ticket = 3000 + index
        common = {"position_ticket": ticket, "order_ticket": 1000 + index, "deal_ticket": 2000 + index, "trace_id": f"trace-{index}", "ack_sequence": index + 1}
        good.extend([
            _v5_event(**common),
            _v5_event(**{**common, "deal_ticket": 4000 + index, "filled": 0, "closed": 1, "deal_entry": 1, "profit": 10.0, "commission": -0.1, "position_final_close": 1, "age_bars": 5.0, "close_reason": "TP"}),
        ])
        churn.extend([
            _v5_event(**common),
            _v5_event(**{**common, "deal_ticket": 5000 + index, "filled": 0, "closed": 1, "deal_entry": 1, "profit": 10.0, "commission": -0.1, "position_final_close": 1, "age_bars": 1.0, "close_reason": "EXPERT"}),
        ])
    good_path, churn_path = tmp_path / "good.csv", tmp_path / "churn.csv"
    _write_execution_ledger_v5(good_path, good)
    _write_execution_ledger_v5(churn_path, churn)
    good_summary = parse_mt5_execution_ledger(good_path, initial_equity=10_000.0, min_holding_bars=3)
    churn_summary = parse_mt5_execution_ledger(churn_path, initial_equity=10_000.0, min_holding_bars=3)

    good_score = mt5_fitness(metrics, execution_summary=good_summary, require_execution_ledger=True, min_position_samples=8)
    churn_breakdown = mt5_fitness_breakdown(metrics, execution_summary=churn_summary, require_execution_ledger=True, min_position_samples=8)
    missing = mt5_fitness_breakdown(metrics, require_execution_ledger=True, min_position_samples=8)

    assert good_score > churn_breakdown.rank_score
    assert churn_summary.immediate_exit_ratio == 1.0
    assert "immediate_exit_ratio_above_0.99:1.000" in churn_breakdown.eligibility_reasons
    assert missing.rank_score == -100.0
    assert "execution_ledger_missing_or_no_finalised_lifecycle" in missing.eligibility_reasons


def test_v5_fitness_uses_position_lifecycles_not_partial_deal_count(tmp_path) -> None:
    ledger = tmp_path / "ledger.csv"
    rows = [_v5_event()]
    for index in range(8):
        rows.append(_v5_event(
            logged_at=f"2025.01.01 00:0{index + 1}:00", deal_ticket=2100 + index,
            filled=0, closed=1, deal_entry=1, profit=1.0, commission=-0.01,
            position_final_close=1 if index == 7 else 0, age_bars=float(index + 1),
            close_reason="EXPERT",
        ))
    _write_execution_ledger_v5(ledger, rows)
    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0, min_holding_bars=3)

    assert summary.lifecycle_count == 1
    assert summary.lifecycles[0].close_count == 8



def test_report_ledger_mismatch_is_an_invalid_run_not_false_negative_pnl() -> None:
    metrics = _fitness_metrics(
        profit_factor=1.5, win_rate=60.0, total_trades=0, net_profit=0.0,
        drawdown=1.0, avg_win=1.2, avg_loss=-1.0,
    )
    lifecycle = MT5PositionLifecycle(
        position_ticket="301", entry_time=None, exit_time=None,
        net_pnl=12.0, gross_profit=12.0, commission=0.0, swap=0.0,
        initial_risk=10.0, realized_r=1.2, duration_bars=5.0,
        close_reason="TP", final_close_confirmed=True,
    )
    summary = MT5ExecutionSummary(
        ledger_available=True, schema_version=5, lifecycle_count=1,
        ledger_integrity=1.0, lifecycles=(lifecycle,),
    )
    reconciliation = reconcile_mt5_report_and_ledger(metrics, summary)
    checked = replace(metrics, execution_summary=summary, reconciliation=reconciliation)

    breakdown = mt5_fitness_breakdown(
        checked, execution_summary=summary, require_execution_ledger=True,
        min_position_samples=1,
    )

    assert reconciliation.reconciled is False
    assert reconciliation.trade_count_consistent is False
    assert breakdown.rank_score == -100.0
    assert any(reason.startswith("report_execution_mismatch:") for reason in breakdown.eligibility_reasons)
    assert "net_profit_not_positive" not in breakdown.eligibility_reasons


def test_parsed_report_lifecycle_mismatch_is_not_reconciled() -> None:
    metrics = _fitness_metrics(
        profit_factor=0.0, win_rate=0.0, total_trades=5, net_profit=-144.87,
        drawdown=1.45, avg_win=0.0, avg_loss=-28.974,
    )
    lifecycles = tuple(
        MT5PositionLifecycle(
            position_ticket=str(index), entry_time=None, exit_time=None,
            net_pnl=value, gross_profit=value, commission=0.0, swap=0.0,
            initial_risk=100.0, realized_r=value / 100.0, duration_bars=12.0,
            close_reason="SL", final_close_confirmed=True,
        )
        for index, value in enumerate((-50.0, -40.0, -31.59), start=1)
    )
    summary = MT5ExecutionSummary(
        ledger_available=True, schema_version=17, lifecycle_count=3,
        ledger_integrity=1.0, lifecycles=lifecycles,
    )

    reconciliation = reconcile_mt5_report_and_ledger(
        metrics,
        summary,
        report_lifecycle_count=5,
        report_lifecycle_net_profit=-144.87,
    )

    assert reconciliation.reconciled is False
    assert reconciliation.trade_count_consistent is False
    assert any(
        reason == "report_lifecycle_count_mismatch:report=5,ledger=3"
        for reason in reconciliation.reasons
    )


def test_terminal_bridge_mark_reconciles_tester_forced_close(tmp_path) -> None:
    bridge = tmp_path / "bridge.csv"
    bridge.write_text(
        "time\tbalance\tequity\tpositions\n"
        "2025.04.01 23:59\t10028.58\t9996.22\t1\n",
        encoding="utf-8",
    )
    mark = parse_bridge_terminal_mark(bridge)
    metrics = _fitness_metrics(
        profit_factor=0.99, win_rate=55.56, total_trades=9, net_profit=-3.78,
        drawdown=3.73, avg_win=73.58, avg_loss=-92.93,
    )
    lifecycle = MT5PositionLifecycle(
        position_ticket="301", entry_time=None, exit_time=None,
        net_pnl=28.58, gross_profit=28.58, commission=0.0, swap=0.0,
        initial_risk=100.0, realized_r=0.2858, duration_bars=5.0,
        close_reason="SL", final_close_confirmed=True,
    )
    summary = MT5ExecutionSummary(
        ledger_available=True, schema_version=8, lifecycle_count=8,
        ledger_integrity=1.0, lifecycles=(lifecycle,),
    )

    reconciliation = reconcile_mt5_report_and_ledger(
        metrics,
        summary,
        terminal_open_positions=int(mark["open_positions"]),
        terminal_open_pnl=float(mark["open_pnl"]),
    )

    assert mark["open_pnl"] == pytest.approx(-32.36)
    assert reconciliation.ledger_marked_net_profit == pytest.approx(-3.78)
    assert reconciliation.reconciled is True

    # Reconciliation is intentionally weaker than a closed validation window:
    # the report can match ledger + mark-to-market while a position is still
    # open at tester shutdown. Promotion must reject that state explicitly.
    checked = replace(metrics, execution_summary=summary, reconciliation=reconciliation)
    strict = mt5_fitness_breakdown(
        checked,
        execution_summary=summary,
        require_execution_ledger=True,
        min_position_samples=1,
        require_terminal_flat=True,
    )
    assert strict.eligible is False
    assert strict.rank_score == -100.0
    assert "terminal_open_positions_nonzero:1" in strict.eligibility_reasons


def test_terminal_open_positions_receive_mark_to_market_lifecycle_credit(tmp_path) -> None:
    trace_id = "CG_ROLE_ENTRY_LIMIT__CTX_VOLATILE_EXPANSION~trace-301"
    positions_pipe = (
        "301:1:0.10:100:90:120:-0.5:12:0.8:-0.7:0:0:0:0:5:0:0.3:"
        "CTX_VOLATILE_EXPANSION:CG_GOC_CTX_VOLATILE_EXPANSION:"
        f"CG_GOC_CTX_VOLATILE_EXPANSION:{trace_id}"
    )
    bridge = tmp_path / "bridge.csv"
    bridge.write_text(
        "time\tbalance\tequity\tpositions\tpositions_pipe\n"
        f"2025.04.01 23:59\t10000\t9950\t1\t{positions_pipe}\n",
        encoding="utf-8",
    )
    ledger = tmp_path / "execution.csv"
    ledger.write_text(
        "schema_version\tlogged_at\tack_sequence\ttrace_id\tsource\tfilled\tclosed\t"
        "position_ticket\tinitial_risk\tposition_owner_trace_id\tposition_owner_context\t"
        "position_owner_expert\tposition_owner_policy\n"
        f"9\t2025.04.01 20:00:00\t7\t{trace_id}\tpolicy\t1\t0\t301\t100\t{trace_id}\t"
        "CTX_VOLATILE_EXPANSION\tCG_GOC_CTX_VOLATILE_EXPANSION\t"
        "CG_GOC_CTX_VOLATILE_EXPANSION\n",
        encoding="utf-8",
    )

    mark = parse_bridge_terminal_mark(bridge)
    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)
    settled = settle_terminal_open_position_lifecycles(
        summary,
        ledger,
        mark,
        initial_equity=10_000.0,
    )

    assert mark["open_position_marks"][0]["owner_trace_id"] == trace_id
    assert settled.lifecycle_count == 1
    lifecycle = settled.lifecycles[0]
    assert lifecycle.trace_id == trace_id
    assert lifecycle.realized_r == pytest.approx(-0.5)
    assert lifecycle.net_pnl == pytest.approx(-50.0)
    assert lifecycle.close_reason == "TESTER_TERMINAL_MARK"
    assert lifecycle.source == "mt5_terminal_mark"
    assert lifecycle.metadata["terminal_mark_settlement"] is True


def test_bridge_does_not_credit_placed_order_but_caches_a_fill_for_lifecycle() -> None:
    before = BridgeRequest(sequence=10, inputs=np.zeros(2), schema_version=4)
    placed = BridgeRequest(
        sequence=11, inputs=np.zeros(2), schema_version=4,
        execution_pipe="ack_sequence=10|accepted=1|placed=1|status=PLACED",
    )
    filled = BridgeRequest(
        sequence=11, inputs=np.zeros(2), schema_version=4, positions=1,
        execution_pipe="ack_sequence=10|accepted=1|filled=1|status=FILLED|position_ticket=7",
    )

    placed_details = MT5StrategyTesterEvaluator._execution_details(before, placed, int(Action.BUY_LIMIT))
    filled_details = MT5StrategyTesterEvaluator._execution_details(before, filled, int(Action.MARKET_BUY))

    assert placed_details["memory_eligible"] is False
    assert placed_details["action_feedback_eligible"] is False
    assert filled_details["memory_eligible"] is True
    assert filled_details["action_feedback_eligible"] is False


def test_close_actions_are_suppressed_until_position_is_mature_unless_emergency(tmp_path) -> None:
    runner = tmp_path / "runner"
    runner.mkdir()
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=runner, min_manual_close_age_bars=2))

    young = evaluator._allowed_actions_for_request(
        BridgeRequest(sequence=1, inputs=np.zeros(2), positions=1, positions_pipe="7:1:0.01:100:99:103:-0.1:1")
    )
    mature = evaluator._allowed_actions_for_request(
        BridgeRequest(sequence=2, inputs=np.zeros(2), positions=1, positions_pipe="7:1:0.01:100:99:103:-0.1:2")
    )
    emergency = evaluator._allowed_actions_for_request(
        BridgeRequest(sequence=3, inputs=np.zeros(2), positions=1, positions_pipe="7:1:0.01:100:99:103:-0.9:0")
    )

    assert int(Action.CLOSE_WORST) not in young
    assert int(Action.MANAGE_POSITION) in young
    assert int(Action.CLOSE_WORST) in mature
    assert int(Action.MANAGE_POSITION) in mature
    assert int(Action.CLOSE_WORST) in emergency
