from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.augment_seven_choice_state_audit import augment_dataset


def _write_tsv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_state_audit_recomputes_dynamic_slots_and_blocks_entries(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.npz"
    snapshot = tmp_path / "snapshots.tsv"
    audit = tmp_path / "audit.tsv"
    output = tmp_path / "augmented.npz"
    observations = np.zeros((4, 137), dtype=np.float32)
    observations[:, 0] = np.asarray([1.0, 1.0, 2.0, 2.0])
    arrays = {
        "observations": observations,
        "sample_times": np.asarray([
            "2025-01-01T00:00:00", "2025-01-01T00:00:00",
            "2025-01-01T00:01:00", "2025-01-01T00:01:00",
        ]),
        "allowed_action_mask": np.asarray([[True, True, False, True, False, True, False]] * 4),
        "best_actions": np.asarray([0, 0, 3, 4], dtype=np.int64),
        "outcome_best_actions": np.asarray([1, 2, 3, 4], dtype=np.int64),
        "causal_action_scores_r": np.zeros((4, 7), dtype=np.float32),
    }
    np.savez_compressed(dataset, **arrays)
    (dataset.with_suffix(dataset.suffix + ".json")).write_text(
        json.dumps({"schema_version": 14}), encoding="utf-8"
    )
    _write_tsv(snapshot, ["sample_time", "close_bid", "atr", "spread_price"], [
        {"sample_time": "2025.01.01 00:00:00", "close_bid": "100", "atr": "2", "spread_price": "0.5"},
        {"sample_time": "2025.01.01 00:01:00", "close_bid": "101", "atr": "2", "spread_price": "0.5"},
    ])
    _write_tsv(audit, [
        "bar_time", "positions_pipe", "orders_pipe", "balance", "equity",
        "drawdown_pct", "open_risk_ratio",
    ], [
        {"bar_time": "2025.01.01 00:00", "positions_pipe": "", "orders_pipe": "", "balance": "10000", "equity": "10000", "drawdown_pct": "0", "open_risk_ratio": "0"},
        {"bar_time": "2025.01.01 00:01", "positions_pipe": "7:-1:0.1:102:104:98:-0.5:3", "orders_pipe": "", "balance": "10000", "equity": "9990", "drawdown_pct": "0.1", "open_risk_ratio": "0.02"},
    ])

    report = augment_dataset(dataset, snapshot, audit, output)
    assert report["matched_unique_times"] == 2
    assert report["state_position_rows"] == 1
    with np.load(output, allow_pickle=False) as loaded:
        result = {name: loaded[name] for name in loaded.files}
    # The base market block is immutable; only position/account/execution
    # blocks are rebuilt from the audit.
    assert result["observations"][0, 0] == 1.0
    np.testing.assert_allclose(result["observations"][0, 29:77], 0.0)
    # The second time is duplicated for the two entry owners.
    assert result["observations"][2, 29] == 1.0
    assert result["observations"][2, 30] == -1.0
    assert result["observations"][2, 69] == 0.2
    assert result["allowed_action_mask"][2].tolist() == [True, False, False, False, False, False, False]
    assert result["best_actions"][2] == 0
    assert result["outcome_best_actions"][3] == 0
    assert result["causal_action_scores_r"][2, 0] == 0.0
    assert np.all(result["causal_action_scores_r"][2, 1:] == -5.0)


def test_state_augmentation_can_fail_on_mixed_feature_contracts(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.npz"
    snapshot = tmp_path / "snapshots.tsv"
    audit = tmp_path / "audit.tsv"
    output = tmp_path / "augmented.npz"
    np.savez_compressed(
        dataset,
        observations=np.zeros((2, 137), dtype=np.float32),
        sample_times=np.asarray(["2025-01-01T00:00:00"] * 2),
        allowed_action_mask=np.ones((2, 7), dtype=bool),
        best_actions=np.zeros(2, dtype=np.int64),
        outcome_best_actions=np.zeros(2, dtype=np.int64),
        causal_action_scores_r=np.zeros((2, 7), dtype=np.float32),
    )
    _write_tsv(snapshot, [
        "sample_time", "close_bid", "atr", "spread_price", "feature_schema_hash",
    ], [{
        "sample_time": "2025.01.01 00:00:00", "close_bid": "100", "atr": "2",
        "spread_price": "0.5", "feature_schema_hash": "dense-contract",
    }])
    _write_tsv(audit, [
        "bar_time", "positions_pipe", "orders_pipe", "balance", "equity",
        "drawdown_pct", "open_risk_ratio", "feature_schema_hash",
    ], [{
        "bar_time": "2025.01.01 00:00", "positions_pipe": "", "orders_pipe": "",
        "balance": "10000", "equity": "10000", "drawdown_pct": "0",
        "open_risk_ratio": "0", "feature_schema_hash": "state-contract",
    }])

    with pytest.raises(ValueError, match="khong cung hop dong"):
        augment_dataset(
            dataset, snapshot, audit, output,
            expected_feature_schema_hash="v193-contract",
            require_matching_feature_schema=True,
        )


def test_state_augmentation_reports_matching_feature_contract(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.npz"
    snapshot = tmp_path / "snapshots.tsv"
    audit = tmp_path / "audit.tsv"
    output = tmp_path / "augmented.npz"
    np.savez_compressed(
        dataset,
        observations=np.zeros((2, 137), dtype=np.float32),
        sample_times=np.asarray(["2025-01-01T00:00:00"] * 2),
        allowed_action_mask=np.ones((2, 7), dtype=bool),
        best_actions=np.zeros(2, dtype=np.int64),
        outcome_best_actions=np.zeros(2, dtype=np.int64),
        causal_action_scores_r=np.zeros((2, 7), dtype=np.float32),
    )
    _write_tsv(snapshot, [
        "sample_time", "close_bid", "atr", "spread_price", "feature_schema_hash",
    ], [{
        "sample_time": "2025.01.01 00:00:00", "close_bid": "100", "atr": "2",
        "spread_price": "0.5", "feature_schema_hash": "v193-contract",
    }])
    _write_tsv(audit, [
        "bar_time", "positions_pipe", "orders_pipe", "balance", "equity",
        "drawdown_pct", "open_risk_ratio", "feature_schema_hash",
    ], [{
        "bar_time": "2025.01.01 00:00", "positions_pipe": "", "orders_pipe": "",
        "balance": "10000", "equity": "10000", "drawdown_pct": "0",
        "open_risk_ratio": "0", "feature_schema_hash": "v193-contract",
    }])

    report = augment_dataset(
        dataset, snapshot, audit, output,
        expected_feature_schema_hash="v193-contract",
        require_matching_feature_schema=True,
    )
    assert report["feature_contract_match"] is True
    assert report["observed_feature_schema_hashes"] == {
        "snapshots": ["v193-contract"],
        "state_audit": ["v193-contract"],
    }
