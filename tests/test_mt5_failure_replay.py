from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.five_failure_replay import FiveFailureReplayBank
from self_growing_trader.five_shared_policy import FIVE_AGENT_COUNT, SharedFiveTradingPolicy
from self_growing_trader.mt5_failure_replay import (
    MT5FailureReplayCoordinator,
    build_mt5_failure_replay_window,
)
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    _mt5_replay_anchor_reached,
    write_tester_set,
)
from self_growing_trader.protocol_contract import ACTION_COUNT, INPUT_COUNT


def _case(bank: FiveFailureReplayBank, *, when: str = "2025.02.05 10:35") -> str:
    return bank.add(
        trace_id="trace-mt5-failure",
        condition_key="CTX|BUY_LIMIT|down|mid|expanding",
        failed_agent_index=0,
        failed_action_index=3,
        realized_r=-1.25,
        observations=np.zeros((FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32),
        memories_hidden=np.zeros((FIVE_AGENT_COUNT, 8), dtype=np.float32),
        memories_cell=np.zeros((FIVE_AGENT_COUNT, 8), dtype=np.float32),
        previous_actions=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        actions=np.asarray([3, 0, 0, 0, 0], dtype=np.int64),
        targets=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        delay_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        risk_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        entry_offset_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        sl_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        tp_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        trailing_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        manage_mode_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        close_fraction_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        allowed_action_masks=np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool),
        market_time=when,
        symbol="BTCUSDm",
        period="M1",
        source_event_id="evt-100",
        source_sequence=100,
    )


def test_failure_anchor_builds_precise_intraday_mt5_window() -> None:
    bank = FiveFailureReplayBank(capacity=32)
    case_id = _case(bank)
    window = build_mt5_failure_replay_window(
        bank.cases[case_id], warmup_minutes=120, horizon_minutes=180
    )
    assert window.anchor.strftime("%Y.%m.%d %H:%M") == "2025.02.05 10:35"
    assert window.start.strftime("%Y.%m.%d %H:%M") == "2025.02.05 08:35"
    assert window.end.strftime("%Y.%m.%d %H:%M") == "2025.02.05 13:35"
    assert window.from_date == "2025.02.05"
    assert window.to_date == "2025.02.06"


def test_tester_set_serializes_mt5_failure_replay_contract(tmp_path: Path) -> None:
    config = MT5BacktestConfig(
        offline_replay=True,
        mt5_failure_replay_mode=True,
        mt5_failure_replay_case_id="case-42",
        mt5_failure_replay_anchor_time="2025.02.05 10:35",
        mt5_failure_replay_warmup_minutes=90,
        mt5_failure_replay_horizon_minutes=150,
    )
    text = write_tester_set(config, tmp_path / "trainer.set").read_text(encoding="utf-8")
    assert "ReplayMode=true" in text
    assert "ReplayCaseId=case-42" in text
    assert "ReplayAnchorTime=2025.02.05 10:35" in text
    assert "ReplayWarmupMinutes=90" in text
    assert "ReplayHorizonMinutes=150" in text


def test_replay_anchor_accepts_first_request_after_missing_tick_minute() -> None:
    anchor = "2025.01.11 16:29"
    assert _mt5_replay_anchor_reached(anchor, "2025.01.11 16:28:00") is False
    assert _mt5_replay_anchor_reached(anchor, "2025.01.11 16:29:00") is True
    assert _mt5_replay_anchor_reached(anchor, "2025.01.11 16:30:00") is True
    assert _mt5_replay_anchor_reached(anchor, "") is False


def test_mt5_replay_coordinator_repeats_unresolved_case_until_repaired(tmp_path: Path) -> None:
    bank = FiveFailureReplayBank(capacity=32)
    case_id = _case(bank)

    class FakePolicy:
        def __init__(self) -> None:
            self.five_failure_replay_bank = bank
            self.five_failure_replay_repair_probability = 0.08
            self.five_failure_replay_reopen_probability = 0.14
            self.five_failure_replay_repair_streak = 3
            self.five_failure_replay_audit_interval = 8
            self.five_failure_replay_admission_suppressed = False
            self.resets = 0

        def reset_runtime_state(self) -> None:
            self.resets += 1

        def audit_five_failure_replay_cases(self, case_ids=None):
            ids = list(case_ids or [])
            before = bank.resolved_count
            bank.observe_probabilities(
                {item: 0.05 for item in ids},
                repair_probability=0.08,
                reopen_probability=0.14,
                repair_streak_required=3,
            )
            return {
                "audited": len(ids),
                "probabilities": {item: 0.05 for item in ids},
                "newly_resolved": bank.resolved_count - before,
                "newly_reopened": 0,
                "bank": bank.stats(),
            }

    policy = FakePolicy()
    seen_configs: list[MT5BacktestConfig] = []
    close_calls = {"count": 0}

    class FakeEvaluator:
        def __init__(self, config: MT5BacktestConfig) -> None:
            seen_configs.append(config)
            self.config = config
            self._mt5_failure_replay_anchor_seen = False

        def evaluate_policy(self, policy_id: int, current_policy, round_index: int):
            assert current_policy is policy
            assert self.config.learning_enabled is True
            assert self.config.mt5_failure_replay_mode is True
            assert self.config.mt5_failure_replay_case_id == case_id
            assert self.config.shutdown_terminal is False
            # Represents MT5 reaching the exact historical anchor. Fresh policy
            # transitions would be generated by the real evaluator here.
            self._mt5_failure_replay_anchor_seen = True
            return 0.5, SimpleNamespace(net_profit=-0.2, total_trades=2)

        def close_runner_processes(self) -> None:
            close_calls["count"] += 1

    coordinator = MT5FailureReplayCoordinator(
        MT5BacktestConfig(offline_replay=True, shutdown_terminal=False),
        evaluator_factory=FakeEvaluator,
        journal_path=tmp_path / "replay.jsonl",
    )
    summary = coordinator.run(
        policy,
        sessions=8,
        max_attempts_per_case=6,
        warmup_minutes=120,
        horizon_minutes=180,
    )

    # The same historical weakness is put back into the MT5 training
    # distribution repeatedly, not consumed once and forgotten.
    assert summary.completed_sessions == 3
    assert summary.replayed_case_ids == [case_id, case_id, case_id]
    assert bank.cases[case_id].resolved is True
    assert bank.cases[case_id].mt5_replay_count == 3
    assert policy.resets == 3
    assert len(seen_configs) == 3
    assert close_calls["count"] == 6
    assert policy.five_failure_replay_admission_suppressed is False
    assert (tmp_path / "replay.jsonl").read_text(encoding="utf-8").count("mt5_failure_replay_session") == 3


def test_lifecycle_failure_persists_mt5_market_anchor() -> None:
    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    policy._learning_enabled = True
    rows = list(policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)))
    target_id = str(rows[0].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details.update({
            "five_action_index": 1 if index == 0 else 0,
            "five_economic_actor_eligible": bool(index == 0),
            "five_execution_filled": bool(index == 0),
            "mt5_request_time": "2025.02.05 10:35",
            "mt5_request_event_id": "evt-anchor",
            "mt5_request_sequence": 123,
            "mt5_symbol": "BTCUSDm",
            "mt5_period": "M1",
        })
        enriched.append(replace(trace, action_details=details))
    policy.register_five_entry_lifecycle_packet(tuple(enriched))
    routed = policy.apply_five_entry_lifecycle_reward(
        target_id,
        lifecycle_reward_r=-0.30,
        realized_r=-1.0,
        components={"initial_sl": True, "total_r": -0.30},
    )
    stored = policy.five_failure_replay_bank.cases[routed["failure_replay_case_id"]]
    assert stored.market_time == "2025.02.05 10:35"
    assert stored.source_event_id == "evt-anchor"
    assert stored.source_sequence == 123
    assert stored.symbol == "BTCUSDm"
    assert stored.period == "M1"


def test_ea_replay_contract_is_intraday_and_stops_tester() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5"
    ).read_text(encoding="utf-8")
    assert "input bool ReplayMode = false;" in source
    assert "ConfigureReplayWindow()" in source
    assert "decisionBarTime < g_replayWindowStart" in source
    assert "decisionBarTime > g_replayWindowEnd" in source
    assert "TesterStop();" in source
    # Replay boundaries must guard delayed policy execution as well as bridge
    # requests; otherwise a delayed entry scheduled at the horizon can leak
    # into ticks after the replay episode.
    assert source.index("decisionBarTime > g_replayWindowEnd") < source.index("AdvanceDelayedPolicyActions();")
    assert 'eventId = "replay-" + ReplayCaseId' in source


def test_mt5_replay_scheduler_round_robins_unresolved_cases(tmp_path: Path) -> None:
    bank = FiveFailureReplayBank(capacity=32)
    first = _case(bank, when="2025.02.05 10:35")
    second = bank.add(
        trace_id="trace-mt5-failure-2",
        condition_key="CTX|SELL_LIMIT|up|mid|expanding",
        failed_agent_index=1,
        failed_action_index=4,
        realized_r=-1.10,
        observations=np.ones((FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32),
        memories_hidden=np.zeros((FIVE_AGENT_COUNT, 8), dtype=np.float32),
        memories_cell=np.zeros((FIVE_AGENT_COUNT, 8), dtype=np.float32),
        previous_actions=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        actions=np.asarray([0, 4, 0, 0, 0], dtype=np.int64),
        targets=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        delay_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        risk_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        entry_offset_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        sl_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        tp_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        trailing_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        manage_mode_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        close_fraction_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        allowed_action_masks=np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool),
        market_time="2025.02.06 11:20",
        symbol="BTCUSDm",
        period="M1",
    )

    class FakePolicy:
        def __init__(self) -> None:
            self.five_failure_replay_bank = bank
            self.five_failure_replay_audit_interval = 8
            self.five_failure_replay_admission_suppressed = False

        def reset_runtime_state(self) -> None:
            return None

        def audit_five_failure_replay_cases(self, case_ids=None):
            # Keep every case unresolved for this scheduling test.
            return {"audited": len(list(case_ids or [])), "newly_resolved": 0,
                    "newly_reopened": 0, "bank": bank.stats()}

    seen: list[str] = []

    class FakeEvaluator:
        def __init__(self, config: MT5BacktestConfig) -> None:
            self.config = config
            self._mt5_failure_replay_anchor_seen = False

        def evaluate_policy(self, policy_id: int, current_policy, round_index: int):
            seen.append(self.config.mt5_failure_replay_case_id)
            self._mt5_failure_replay_anchor_seen = True
            return 0.0, SimpleNamespace(net_profit=-0.1, total_trades=1)

        def close_runner_processes(self) -> None:
            return None

    coordinator = MT5FailureReplayCoordinator(
        MT5BacktestConfig(offline_replay=True), evaluator_factory=FakeEvaluator,
        journal_path=tmp_path / "round_robin.jsonl",
    )
    summary = coordinator.run(
        FakePolicy(), sessions=4, max_attempts_per_case=4,
        warmup_minutes=60, horizon_minutes=60,
    )
    assert summary.completed_sessions == 4
    assert seen[0] != seen[1]
    assert set(seen[:2]) == {first, second}
    assert seen[2] != seen[3]
    assert set(seen[2:4]) == {first, second}


def test_mt5_replay_child_failures_are_deduplicated_by_parent_context_action() -> None:
    bank = FiveFailureReplayBank(capacity=32)
    parent = _case(bank)
    common = dict(
        condition_key="CTX|MARKET_SELL|down|mid|expanding",
        failed_agent_index=0,
        failed_action_index=2,
        realized_r=-0.8,
        memories_hidden=np.zeros((FIVE_AGENT_COUNT, 8), dtype=np.float32),
        memories_cell=np.zeros((FIVE_AGENT_COUNT, 8), dtype=np.float32),
        previous_actions=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        actions=np.asarray([2, 0, 0, 0, 0], dtype=np.int64),
        targets=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        delay_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        risk_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        entry_offset_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        sl_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        tp_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        trailing_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        manage_mode_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        close_fraction_indices=np.zeros(FIVE_AGENT_COUNT, dtype=np.int64),
        allowed_action_masks=np.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool),
        market_time="2025.02.05 10:42",
        symbol="BTCUSDm",
        period="M1",
        origin="mt5_replay",
        parent_case_id=parent,
        dedupe_key=f"mt5-replay-child|{parent}|CTX|MARKET_SELL|down|mid|expanding|2",
    )
    first = bank.add(
        trace_id="replay-attempt-1",
        observations=np.ones((FIVE_AGENT_COUNT, INPUT_COUNT), dtype=np.float32),
        **common,
    )
    second = bank.add(
        trace_id="replay-attempt-2",
        observations=np.full((FIVE_AGENT_COUNT, INPUT_COUNT), 2.0, dtype=np.float32),
        realized_r=-1.2,
        **{k: v for k, v in common.items() if k != "realized_r"},
    )
    assert first == second
    assert first != parent
    child = bank.cases[first]
    assert child.origin == "mt5_replay"
    assert child.parent_case_id == parent
    assert child.realized_r == -1.2
    assert len(bank.cases) == 2
