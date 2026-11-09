from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from self_growing_trader.model1_replay import (
    CanonicalExecutionState,
    CanonicalOrderRequest,
    Model1OracleDataset,
    canonical_market_vector,
    compose_canonical_inputs,
    ReplayPositionSnapshot,
    build_dynamic_input_blocks,
    Model1TickDataset,
    Model1OracleTape,
    compare_execution_state,
    compare_order_request,
    require_decision_parity,
    model1_training_state_hash,
    validate_model1_execution_contract,
    sha256_file,
)


def _capture(tmp_path):
    ticks = tmp_path / "ticks.tsv"
    ticks.write_text(
        "time_msc\tbid\task\tlast\tvolume\tflags\n"
        "1000\t100.0\t100.2\t0.0\t1\t6\n"
        "1001\t100.1\t100.3\t0.0\t2\t6\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "tester_model": 1,
        "symbol": "BTCUSDm",
        "server": "test",
        "from_date": "2025.01.01",
        "to_date": "2025.01.02",
        "tick_count": 2,
        "tick_sha256": sha256_file(ticks),
        "point": 0.01,
        "digits": 2,
        "tick_size": 0.01,
        "tick_value": 0.01,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
        "contract_size": 1.0,
    }
    ticks.with_suffix(".tsv.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return ticks


def test_model1_dataset_requires_pinned_ordered_capture(tmp_path) -> None:
    ticks = _capture(tmp_path)
    dataset = Model1TickDataset(ticks)
    replay = list(dataset)
    assert [tick.time_msc for tick in replay] == [1000, 1001]
    assert replay[0].ask == 100.2
    bars = dataset.minute_bars()
    assert len(bars) == 1
    assert bars.iloc[0]["Open"] == 100.0
    assert bars.iloc[0]["Close"] == 100.1
    assert bars.iloc[0]["TickVol"] == 2


def test_model1_dataset_fails_closed_after_tick_mutation(tmp_path) -> None:
    ticks = _capture(tmp_path)
    ticks.write_text(ticks.read_text(encoding="utf-8") + "1002\t99\t100\t0\t1\t6\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        Model1TickDataset(ticks)


def test_model1_dataset_rejects_other_tester_model(tmp_path) -> None:
    ticks = _capture(tmp_path)
    path = ticks.with_suffix(".tsv.manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tester_model"] = 4
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="model-1"):
        Model1TickDataset(ticks)


def test_decision_parity_checks_inputs_outputs_and_final_action() -> None:
    report = require_decision_parity(
        sequence=7,
        mt5_raw_inputs=[1.0, 2.0], python_raw_inputs=[1.0, 2.0],
        mt5_normalized_inputs=[0.5, 0.75], python_normalized_inputs=[0.5, 0.75],
        mt5_outputs=[0.1, 0.9], python_outputs=[0.1, 0.9],
        mt5_action=1, python_action=1,
    )
    assert report.exact


def test_decision_parity_fails_closed_on_tiny_contract_drift() -> None:
    with pytest.raises(ValueError, match=r"raw_inputs\[1\]"):
        require_decision_parity(
            sequence=8,
            mt5_raw_inputs=[1.0, 2.0], python_raw_inputs=[1.0, 2.000001],
            mt5_normalized_inputs=[0.5], python_normalized_inputs=[0.5],
            mt5_outputs=[0.1], python_outputs=[0.1],
            mt5_action=0, python_action=0,
        )


def test_order_parity_is_zero_tick_by_default() -> None:
    mt5 = CanonicalOrderRequest(1, 1, "MARKET", 1, 0.1, 100.0, 95.0, 110.0, 0, 0)
    python = CanonicalOrderRequest(1, 1, "MARKET", 1, 0.1, 100.01, 95.0, 110.0, 0, 0)
    assert compare_order_request(mt5, python, tick_size=0.01)[0].field == "price"


def test_execution_parity_checks_serialized_state_and_net_components() -> None:
    mt5 = CanonicalExecutionState(1, True, 10009, "1", "2", "3", "p", "", 10001, 10002, -1, 0, 2, 0.2)
    python = CanonicalExecutionState(1, True, 10009, "1", "2", "3", "p", "", 10001, 10002, -1, 0, 2, 0.2)
    assert compare_execution_state(mt5, python) == ()


def test_oracle_requires_identical_feature_audit_and_bridge_inputs(tmp_path) -> None:
    bridge = tmp_path / "bridge.csv"
    audit = tmp_path / "audit.csv"
    bridge.write_text(
        "event_id\tinputs_pipe\toutputs_pipe\taction\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions_pipe\torders_pipe\tresponse_status\texecution_pipe\n"
        "e1\t1|2\t0.1|0.2\t1\t10000\t10000\t0\t10\t\t\tEXECUTED\t\n",
        encoding="utf-8",
    )
    audit.write_text("event_id\tinputs\ne1\t1|2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="feature oracle missing columns"):
        Model1OracleDataset(bridge, audit).decisions()

    header = "event_id\tinputs\tlogged_at\tbar_time\tbar_close\tatr\topen_risk_ratio\tsymbol_point\tday_trades\tpositions_pipe\torders_pipe\tbalance\tequity\tdrawdown_pct\tspread_points\tmargin\tfree_margin\tmargin_level_pct\tleverage\n"
    suffix = "\t\t\t10000\t10000\t0\t10\t0\t10000\t0\t100\n"
    audit.write_text(header + "e1\t1|2\t2025-01-01 00:01\t2025-01-01 00:00\t100\t2\t0\t0.01\t0" + suffix, encoding="utf-8")
    decisions = Model1OracleDataset(bridge, audit).decisions()
    assert decisions[0].raw_inputs.tolist() == [1.0, 2.0]
    audit.write_text(header + "e1\t1|3\t2025-01-01 00:01\t2025-01-01 00:00\t100\t2\t0\t0.01\t0" + suffix, encoding="utf-8")
    with pytest.raises(ValueError, match="bridge/audit input mismatch"):
        Model1OracleDataset(bridge, audit).decisions()


def test_canonical_input_composition_never_recalculates_mt5_market_blocks() -> None:
    raw = np.arange(137, dtype=float)
    market = canonical_market_vector(raw)
    result = compose_canonical_inputs(
        market,
        position_features=np.full(40, -2.0),
        account_features=np.full(8, -3.0),
        execution_position_features=np.full(10, -4.0),
    )
    assert np.array_equal(result[:29], raw[:29])
    assert np.array_equal(result[77:107], raw[77:107])
    assert np.array_equal(result[117:137], raw[117:137])
    assert np.array_equal(result[29:69], np.full(40, -2.0))
    assert np.array_equal(result[69:77], np.full(8, -3.0))
    assert np.array_equal(result[107:117], np.full(10, -4.0))


def test_dynamic_blocks_use_exact_open_time_not_rounded_snapshot_age() -> None:
    position = ReplayPositionSnapshot("1", 1, 0.1, 100.0, 95.0, 110.0, 5.0, 1_000)
    positions, account, execution = build_dynamic_input_blocks(
        positions=(position,), bar_time=pd.Timestamp(1_120, unit="s"),
        decision_time="2025-01-01 12:00", bar_close=105.0, atr=2.0,
        spread_points=10, point=0.01, balance=10_000, equity=10_100,
        open_risk_ratio=0.01, day_trades=1, max_positions=5,
        max_holding_bars=60, min_daily_trades=0, max_daily_trades=0,
    )
    assert positions[3] == pytest.approx(2.0 / 60.0)
    assert positions[7] == 1.0
    assert execution[0] == 5.0
    assert account[0] == 0.2


def test_oracle_compiles_to_csv_free_tape(tmp_path) -> None:
    bridge = tmp_path / "bridge.csv"
    audit = tmp_path / "audit.csv"
    inputs = "|".join(str(float(i)) for i in range(137))
    outputs = "|".join(str(float(i) / 10.0) for i in range(20))
    bridge.write_text(
        "event_id\tinputs_pipe\toutputs_pipe\taction\tbalance\tequity\tdrawdown_pct\tspread_points\tpositions_pipe\torders_pipe\tresponse_status\texecution_pipe\n"
        f"e1\t{inputs}\t{outputs}\t4\t10000\t10000\t0\t10\t\t\tEXECUTED\t\n",
        encoding="utf-8",
    )
    audit.write_text(
        "event_id\tinputs\tlogged_at\tbar_time\tbar_close\tatr\topen_risk_ratio\tsymbol_point\tday_trades\tpositions_pipe\torders_pipe\tbalance\tequity\tdrawdown_pct\tspread_points\tmargin\tfree_margin\tmargin_level_pct\tleverage\n"
        f"e1\t{inputs}\t2025-01-01 00:01\t2025-01-01 00:00\t100\t2\t0\t0.01\t0\t\t\t10000\t10000\t0\t10\t0\t10000\t0\t100\n",
        encoding="utf-8",
    )
    target = Model1OracleDataset(bridge, audit).compile(tmp_path / "oracle.npz")
    tape = Model1OracleTape(target)
    assert len(tape) == 1
    assert tape.inputs.shape == (1, 137)
    assert tape.outputs.shape == (1, 20)
    assert tape.actions.tolist() == [4]
    assert tape.metadata[0]["event_id"] == "e1"
    assert len(tape.account_margin_state_sha256) == 64
    assert len(tape.content_sha256) == 64


def test_training_state_hash_ignores_process_identity_but_not_weights() -> None:
    first = SimpleNamespace(weights=np.asarray([1.0, 2.0]), graph_instance_uuid="process-a")
    second = SimpleNamespace(weights=np.asarray([1.0, 2.0]), graph_instance_uuid="process-b")
    assert model1_training_state_hash(first) == model1_training_state_hash(second)
    second.weights[1] = 2.0000001
    assert model1_training_state_hash(first) != model1_training_state_hash(second)


def test_execution_contract_validates_action_trace_and_fixed_geometry(tmp_path) -> None:
    target = tmp_path / "tape.npz"
    metadata = json.dumps({
        "sequence": 1, "event_id": "1735689620-1", "positions_pipe": "",
        "orders_pipe": "", "execution_pipe": "", "balance": 10000,
        "equity": 10000, "drawdown_pct": 0, "spread_points": 10,
        "margin": 0, "free_margin": 10000, "margin_level_pct": 0, "leverage": 100,
        "day_trades": 0, "bar_close": 100, "atr": 2,
    }, separators=(",", ":"))
    header = "transaction_type\tack_sequence\trequested_action\ttrace_id\tgeometry_entry_pipe"
    geometry = "0|0|0|0|5|100.00|105.00|90.00|2.00|fixed_atr_v1|2.5|5.0|2.0"
    row = f"BATCH_RESOLUTION\t1\t4\tpolicy~model1-1\t{geometry}"
    np.savez_compressed(
        target, schema_version=np.asarray([1], dtype=np.int16),
        inputs=np.zeros((1, 137)), outputs=np.zeros((1, 20)),
        actions=np.asarray([4], dtype=np.int16), metadata=np.asarray([metadata]),
        execution_lines=np.asarray([header, row]),
    )
    report = validate_model1_execution_contract(Model1OracleTape(target))
    assert report.exact
    assert report.checked_entry_geometries == 1
