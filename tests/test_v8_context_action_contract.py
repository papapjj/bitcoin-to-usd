from __future__ import annotations

import numpy as np

from self_growing_trader.config import load_config
from self_growing_trader.chuyen_gia_policy import tao_he_chuyen_gia_mot_context_mot_policy
from self_growing_trader.he_chuyen_gia_ngu_canh import NhiemVuNguCanh
from self_growing_trader.live_observatory import LiveTelemetryWriter


def _system(seed: int = 17):
    return tao_he_chuyen_gia_mot_context_mot_policy(
        load_config("configs/he_chuyen_gia.yaml"),
        seed=seed,
    )


def _stable_trace(system, observation: np.ndarray, *, execution_context: dict[str, float] | None = None):
    if execution_context is not None:
        system.set_decision_context(execution_context=execution_context)
    trace = None
    for _ in range(4):
        trace = system.act(observation)
    assert trace is not None
    return trace


def test_selected_hypothesis_is_the_final_discrete_policy_action():
    system = _system()
    trace = _stable_trace(system, np.zeros(119, dtype=float))

    candidates = np.asarray(trace.candidate_actions, dtype=float)
    selected_hypothesis = int(trace.selected_idx)
    assert candidates.ndim == 2
    assert 0 <= selected_hypothesis < candidates.shape[0]

    expected_action = int(np.argmax(candidates[selected_hypothesis, :12]))
    final_action = int(np.argmax(np.asarray(trace.selected_action, dtype=float)[:12]))
    details = dict(trace.action_details)

    assert final_action == expected_action
    allowed = set(trace.context_audit["context_allowed_actions"])
    assert expected_action in allowed
    assert int(details["selected_hypothesis_action_index"]) in allowed
    assert int(details["semantic_action"]) == final_action
    assert int(details["policy_output_action_index"]) == final_action
    assert int(details["selected_candidate_action_index"]) == final_action


def test_market_regime_contract_keeps_all_entry_directions_available():
    pending = _system()
    pending_trace = _stable_trace(
        pending,
        np.zeros(119, dtype=float),
        execution_context={"pending_orders": 1.0},
    )
    pending_allowed = set().union(*(
        set(item["allowed_actions"])
        for item in pending_trace.context_audit["selector_proposals"]
        if "ENTRY" in item["role"]
    ))
    assert set(range(1, 7)).issubset(pending_allowed)
    # The entry owner must retain CANCEL_ORDERS while pending orders are
    # present; otherwise the bridge mask intersects its capability scope down
    # to HOLD and the training stream starves.
    assert 7 in pending_allowed

    managing_buy = _system()
    observation = np.zeros(119, dtype=float)
    observation[34] = 0.8
    buy_trace = _stable_trace(managing_buy, observation)
    buy_allowed = set().union(*(
        set(item["allowed_actions"])
        for item in buy_trace.context_audit["selector_proposals"]
        if "ENTRY" in item["role"]
    ))
    assert set(range(1, 7)).issubset(buy_allowed)


def test_owner_provenance_proves_distinct_policy_graph_and_memory_instances():
    system = _system()
    forced = iter([
        ("CTX_RANGING", "RANGING"),
        ("CTX_VOLATILE", "VOLATILE"),
    ])

    def candidate(observation, execution_context=None):
        context_id, regime = next(forced)
        task = NhiemVuNguCanh[regime]
        return (
            context_id,
            context_id,
            task,
            (0.0, 0.0, 0.0),
            f"regime={regime}_conf=0.95_timing_q=0.60_dir=0",
        )

    system.nhan_dien._ung_vien = candidate
    entry = system.act(np.zeros(119, dtype=float))
    pending = system.act(np.zeros(119, dtype=float))
    entry_audit = dict(entry.context_audit)
    pending_audit = dict(pending.context_audit)

    assert entry_audit["ma_chuyen_gia"] != pending_audit["ma_chuyen_gia"]
    assert entry_audit["policy_instance_uuid"] != pending_audit["policy_instance_uuid"]
    assert entry_audit["graph_instance_uuid"] != pending_audit["graph_instance_uuid"]
    assert entry_audit["weight_hash"]
    assert pending_audit["weight_hash"]
    assert entry_audit["memory_hash"]
    assert pending_audit["memory_hash"]
    assert entry_audit["decision_policy_identity"]["policy_instance_uuid"]
    assert pending_audit["decision_policy_identity"]["policy_instance_uuid"]


def test_structural_journal_excludes_credit_only_events(tmp_path):
    path = tmp_path / "neural_live.jsonl"
    structural = tmp_path / "structural.jsonl"
    writer = LiveTelemetryWriter(
        path=str(path),
        structural_event_path=str(structural),
        enabled=True,
        run_id="unit",
    )
    rows = writer.nhat_ky_cau_truc(
        object(),
        None,
        [{
            "type": "graph_credit_attribution",
            "trace_id": "credit-only",
            "updated_edges": 2,
            "updated_neurons": 3,
            "disabled_edges": 0,
        }],
        giai_doan="lifecycle",
    )
    assert rows == []
    assert not structural.exists()


def test_mt5_bridge_blocks_unstable_owner_without_unboundlocalerror(tmp_path):
    """An unstable/missing owner must serialize transport HOLD and never crash."""
    import json
    from types import SimpleNamespace

    from self_growing_trader.mt5_runner import MT5BacktestConfig, MT5StrategyTesterEvaluator

    class _FinishedProcess:
        def poll(self):
            return 0

    class _UnstableOwnerPolicy:
        input_size = 119
        output_size = 20
        experts = {}
        registry = {}
        nhan_dien = object()

        def reset_runtime_state(self):
            return None

        def act(self, observation, goal=None):
            output = np.zeros(20, dtype=float)
            output[1] = 1.0
            return SimpleNamespace(
                trace_id="trace-unstable-owner",
                selected_action=output,
                selected_idx=1,
                confidence=0.0,
                action_details={},
                context_audit={"context_on_dinh": False},
            )

    runner = tmp_path / "runner"
    (runner / "Common" / "Files").mkdir(parents=True)
    (runner / "terminal64.exe").write_text("", encoding="utf-8")
    request_path = runner / "Common" / "Files" / "self_growing_request.csv"
    inputs = "|".join("0" for _ in range(119))
    request_path.write_text(
        "schema_version\tevent_id\tsequence\ttime\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions\tday_trades\tinputs_pipe\n"
        f"7\tevt-unstable\t1\t2025.01.01 00:00\t10000\t10000\t0\t1\t0\t0\t{inputs}\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.htm"
    report.write_text("done", encoding="utf-8")
    evaluator = MT5StrategyTesterEvaluator(
        MT5BacktestConfig(runner_dir=runner, timeout_sec=1, learning_enabled=False)
    )

    served = evaluator._serve_policy_actions(
        _FinishedProcess(), report, _UnstableOwnerPolicy(), policy_id=1, round_index=0
    )

    assert served == 1
    action_path = request_path.with_name("self_growing_action.csv")
    row = action_path.read_text(encoding="utf-8").splitlines()[1].split("\t")
    assert int(row[4]) == 0  # transport HOLD only; no semantic command emitted
    assert not action_path.with_name(action_path.name + ".metadata.jsonl").exists()
