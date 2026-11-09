from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from self_growing_trader.execution_contract import (
    EXECUTION_CONTRACT_VERSION,
    FIXED_GEOMETRY_MODE,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    LEGACY_FACTORIZED_ACTION_MARKER_VALUE,
    LEGACY_LEARNED_GEOMETRY_MODE,
    PARAMETER_ACTIVATION_RAW_LOGITS_V2,
    activate_policy_output,
    decode_execution_geometry,
    pack_legacy_geometry_choices,
)
from self_growing_trader.mt5_runner import parse_mt5_execution_ledger


def test_legacy_marker_keeps_fixed_geometry_and_ignores_packed_row() -> None:
    low = np.zeros(20, dtype=float)
    high = np.zeros(20, dtype=float)
    low[13] = -100.0
    high[13] = 100.0
    low[16] = -100.0
    high[16] = 100.0

    left = decode_execution_geometry(
        low, min_risk_pct=0.001, max_risk_pct=0.005, technical_min_rr=1.2,
        fixed_sl_atr=2.5, fixed_tp_atr=5.0,
    )
    right = decode_execution_geometry(
        high, min_risk_pct=0.001, max_risk_pct=0.005, technical_min_rr=1.2,
        fixed_sl_atr=2.5, fixed_tp_atr=5.0,
    )

    assert EXECUTION_CONTRACT_VERSION == 7
    assert left.mode == right.mode == FIXED_GEOMETRY_MODE
    assert (left.sl_atr, left.tp_atr, left.rr, left.trailing_stop_atr) == (2.5, 5.0, 2.0, 0.0)
    assert (right.sl_atr, right.tp_atr, right.rr, right.trailing_stop_atr) == (2.5, 5.0, 2.0, 0.0)
    assert left.risk_pct == right.risk_pct


def test_marker_6_packet_keeps_exact_legacy_geometry_codebook() -> None:
    outputs = np.zeros(20, dtype=float)
    outputs[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] = LEGACY_FACTORIZED_ACTION_MARKER_VALUE
    outputs[GEOMETRY_PACKED_OUTPUT_INDEX] = pack_legacy_geometry_choices(5, 3, 4)

    geometry = decode_execution_geometry(
        outputs, min_risk_pct=0.001, max_risk_pct=0.005, technical_min_rr=2.0,
        fixed_sl_atr=2.5, fixed_tp_atr=5.0,
    )

    assert geometry.mode == LEGACY_LEARNED_GEOMETRY_MODE
    assert (geometry.sl_atr, geometry.rr, geometry.tp_atr, geometry.trailing_stop_atr) == (
        4.0, 4.0, 16.0, 2.0,
    )


def test_raw_parameter_logits_keep_risk_decoder_but_not_fixed_sl_tp() -> None:
    raw = np.full(20, 3.0, dtype=float)
    activated = activate_policy_output(raw, parameter_activation_mode=PARAMETER_ACTIVATION_RAW_LOGITS_V2)
    assert np.allclose(activated[:12], np.tanh(raw[:12]))
    assert np.allclose(activated[12:], raw[12:])
    geometry = decode_execution_geometry(
        activated, min_risk_pct=0.001, max_risk_pct=0.005, technical_min_rr=1.2,
        fixed_sl_atr=2.5, fixed_tp_atr=5.0,
    )
    assert 0.001 < geometry.risk_pct < 0.005
    assert (geometry.sl_atr, geometry.tp_atr, geometry.rr) == (2.5, 5.0, 2.0)


def test_fixed_geometry_rejects_rr_below_technical_minimum() -> None:
    with np.testing.assert_raises(ValueError):
        decode_execution_geometry(
            np.zeros(20), min_risk_pct=0.001, max_risk_pct=0.005, technical_min_rr=1.2,
            fixed_sl_atr=2.5, fixed_tp_atr=2.0,
        )


def test_v13_ledger_retains_immutable_fixed_geometry_and_policy_exit(tmp_path: Path) -> None:
    fields = [
        "schema_version", "logged_at", "ack_sequence", "trace_id", "requested_action", "source",
        "transaction_type", "filled", "closed", "position_final_close", "position_ticket",
        "profit", "commission", "swap", "initial_risk", "age_bars", "price", "close_reason",
        "entry_price", "initial_sl_price", "initial_tp_price", "initial_risk_distance_price",
        "final_sl_price", "final_tp_price", "exit_kind", "exit_controller", "exit_trace_id",
        "exit_requested_action", "geometry_mode", "fixed_sl_atr", "fixed_tp_atr", "fixed_rr",
        "position_owner_trace_id",
    ]
    ledger = tmp_path / "execution.tsv"
    rows = [
        {
            "schema_version": 13, "logged_at": "2026.07.30 10:00:00", "ack_sequence": 1,
            "trace_id": "entry-trace", "requested_action": 1, "source": "policy",
            "transaction_type": "DEAL_ADD", "filled": 1, "position_final_close": 0,
            "position_ticket": 77, "initial_risk": 100, "entry_price": 100, "price": 100,
            "initial_sl_price": 97.5, "initial_tp_price": 105, "initial_risk_distance_price": 2.5,
            "final_sl_price": 97.5, "final_tp_price": 105, "geometry_mode": "fixed_atr_v1",
            "fixed_sl_atr": 2.5, "fixed_tp_atr": 5.0, "fixed_rr": 2.0,
            "position_owner_trace_id": "entry-trace",
        },
        {
            "schema_version": 13, "logged_at": "2026.07.30 10:05:00", "ack_sequence": 4,
            "trace_id": "close-trace", "requested_action": 9, "source": "policy",
            "transaction_type": "DEAL_ADD", "closed": 1, "position_final_close": 1,
            "position_ticket": 77, "profit": 50, "initial_risk": 100, "age_bars": 5,
            "price": 101.25, "close_reason": "EXPERT", "entry_price": 100,
            "initial_sl_price": 97.5, "initial_tp_price": 105, "initial_risk_distance_price": 2.5,
            "final_sl_price": 97.5, "final_tp_price": 105, "exit_kind": "POLICY_CLOSE_BEST",
            "exit_controller": "policy", "exit_trace_id": "close-trace", "exit_requested_action": 9,
            "geometry_mode": "fixed_atr_v1", "fixed_sl_atr": 2.5, "fixed_tp_atr": 5.0,
            "fixed_rr": 2.0, "position_owner_trace_id": "entry-trace",
        },
    ]
    with ledger.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    lifecycle = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0).lifecycles[0]
    assert lifecycle.trace_id == "entry-trace"
    assert lifecycle.initial_risk == 100.0
    assert lifecycle.realized_r == 0.5
    assert lifecycle.initial_sl_price == lifecycle.final_sl_price == 97.5
    assert lifecycle.initial_tp_price == lifecycle.final_tp_price == 105.0
    assert lifecycle.metadata["geometry_mode"] == "fixed_atr_v1"
    assert lifecycle.metadata["fixed_sl_atr"] == 2.5
    assert lifecycle.metadata["fixed_tp_atr"] == 5.0
    assert lifecycle.metadata["fixed_rr"] == 2.0
    assert lifecycle.exit_kind == "POLICY_CLOSE_BEST"


def test_v13_fixed_geometry_suffix_expands_to_auditable_fields(tmp_path: Path) -> None:
    ledger = tmp_path / "execution.tsv"
    ledger.write_text(
        "schema_version\tposition_ticket\tfilled\tclosed\tposition_final_close\t"
        "initial_risk\tentry_price\tinitial_sl_price\tinitial_tp_price\t"
        "initial_risk_distance_price\tgeometry_entry_pipe\n"
        "13\t99\t1\t1\t1\t100\t100\t97.5\t105\t2.5\t"
        "1.0|1|2.5|2.0|5.0|100|97.5|105|1.0|fixed_atr_v1|2.5|5.0|2.0\n",
        encoding="utf-8",
    )

    summary = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0)

    assert len(summary.lifecycles) == 1
    assert summary.lifecycles[0].metadata["geometry_mode"] == "fixed_atr_v1"
    assert summary.lifecycles[0].metadata["fixed_sl_atr"] == 2.5
    assert summary.lifecycles[0].metadata["fixed_tp_atr"] == 5.0
    assert summary.lifecycles[0].metadata["fixed_rr"] == 2.0


def test_v15_path_pipe_reconstructs_long_excursion_without_fake_zero(tmp_path: Path) -> None:
    ledger = tmp_path / "execution_v15.tsv"
    ledger.write_text(
        "schema_version\tlogged_at\tposition_ticket\tfilled\tclosed\tposition_final_close\t"
        "profit\tinitial_risk\tentry_price\tinitial_sl_price\tinitial_risk_distance_price\t"
        "geometry_entry_pipe\tgeometry_path_pipe\n"
        "15\t2026.08.16 10:00:00\t501\t1\t0\t0\t0\t100\t100\t99\t1\t"
        "1|1|1|2|2\t\n"
        "15\t2026.08.16 10:01:00\t501\t0\t0\t0\t0\t100\t100\t99\t1\t"
        "1|1|1|2|2\t100.8|99.5|100.5|1|0.1\n"
        "15\t2026.08.16 10:02:00\t501\t0\t1\t1\t50\t100\t100\t99\t1\t"
        "1|1|1|2|2\t\n",
        encoding="utf-8",
    )

    lifecycle = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0).lifecycles[0]

    assert lifecycle.excursion_available is True
    assert np.isclose(lifecycle.mfe_r, 0.8)
    assert np.isclose(lifecycle.mae_r, -0.5)
    assert np.isclose(float(lifecycle.trajectory[0]["unrealized_r"]), 0.5)
    assert lifecycle.metadata["excursion_evidence_source"] == "geometry_path_v15_reconstructed"


def test_v15_short_path_uses_ask_spread_for_excursion(tmp_path: Path) -> None:
    ledger = tmp_path / "execution_v15_short.tsv"
    ledger.write_text(
        "schema_version\tlogged_at\tposition_ticket\tfilled\tclosed\tposition_final_close\t"
        "profit\tinitial_risk\tentry_price\tinitial_sl_price\tinitial_risk_distance_price\t"
        "geometry_entry_pipe\tgeometry_path_pipe\n"
        "15\t2026.08.16 10:00:00\t502\t1\t0\t0\t0\t100\t100\t101\t1\t"
        "1|-1|1|2|2\t\n"
        "15\t2026.08.16 10:01:00\t502\t0\t0\t0\t0\t100\t100\t101\t1\t"
        "1|-1|1|2|2\t100.6|99.2|99.5|1|0.1\n"
        "15\t2026.08.16 10:02:00\t502\t0\t1\t1\t40\t100\t100\t101\t1\t"
        "1|-1|1|2|2\t\n",
        encoding="utf-8",
    )

    lifecycle = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0).lifecycles[0]

    assert lifecycle.excursion_available is True
    assert np.isclose(lifecycle.mfe_r, 0.7)
    assert np.isclose(lifecycle.mae_r, -0.7)
    assert np.isclose(float(lifecycle.trajectory[0]["unrealized_r"]), 0.4)


def test_missing_path_differs_from_explicit_true_zero_excursion(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tsv"
    missing.write_text(
        "schema_version\tposition_ticket\tfilled\tclosed\tposition_final_close\t"
        "profit\tinitial_risk\tentry_price\tinitial_sl_price\tgeometry_entry_pipe\n"
        "15\t601\t1\t0\t0\t0\t100\t100\t99\t1|1|1|2|2\n"
        "15\t601\t0\t1\t1\t-100\t100\t100\t99\t1|1|1|2|2\n",
        encoding="utf-8",
    )
    explicit = tmp_path / "explicit.tsv"
    explicit.write_text(
        "schema_version\tposition_ticket\tfilled\tclosed\tposition_final_close\t"
        "profit\tinitial_risk\tentry_price\tinitial_sl_price\tgeometry_entry_pipe\tgeometry_path_pipe\n"
        "16\t602\t1\t0\t0\t0\t100\t100\t99\t1|1|1|2|2\t\n"
        "16\t602\t0\t0\t0\t0\t100\t100\t99\t1|1|1|2|2\t100|100|100|1|0.1|0|0|0\n"
        "16\t602\t0\t1\t1\t-100\t100\t100\t99\t1|1|1|2|2\t\n",
        encoding="utf-8",
    )

    missing_lifecycle = parse_mt5_execution_ledger(missing, initial_equity=10_000.0).lifecycles[0]
    explicit_lifecycle = parse_mt5_execution_ledger(explicit, initial_equity=10_000.0).lifecycles[0]

    assert missing_lifecycle.excursion_available is False
    assert explicit_lifecycle.excursion_available is True
    assert explicit_lifecycle.mfe_r == 0.0
    assert explicit_lifecycle.mae_r == 0.0
    assert explicit_lifecycle.metadata["excursion_evidence_source"] == "ledger_v16"


def test_v15_reconstruction_ignores_prices_before_mid_bar_fill(tmp_path: Path) -> None:
    ledger = tmp_path / "mid_bar.tsv"
    ledger.write_text(
        "schema_version\tlogged_at\tposition_ticket\tfilled\tclosed\tposition_final_close\t"
        "profit\tinitial_risk\tentry_price\tinitial_sl_price\tinitial_risk_distance_price\t"
        "geometry_entry_pipe\tgeometry_path_pipe\n"
        "15\t2026.08.16 10:00:30\t701\t1\t0\t0\t0\t100\t100\t99\t1\t"
        "1|1|1|2|2\t\n"
        "15\t2026.08.16 10:01:00\t701\t0\t0\t0\t0\t100\t100\t99\t1\t"
        "1|1|1|2|2\t110|90|100|1|0.1\n"
        "15\t2026.08.16 10:02:00\t701\t0\t0\t0\t0\t100\t100\t99\t1\t"
        "1|1|1|2|2\t101|99.5|100.5|1|0.1\n"
        "15\t2026.08.16 10:03:00\t701\t0\t1\t1\t50\t100\t100\t99\t1\t"
        "1|1|1|2|2\t\n",
        encoding="utf-8",
    )

    lifecycle = parse_mt5_execution_ledger(ledger, initial_equity=10_000.0).lifecycles[0]

    assert lifecycle.excursion_available is True
    assert np.isclose(lifecycle.mfe_r, 1.0)
    assert np.isclose(lifecycle.mae_r, -0.5)
    assert float(lifecycle.trajectory[0]["mfe_r"]) == 0.0
    assert np.isclose(float(lifecycle.trajectory[1]["mfe_r"]), 1.0)
