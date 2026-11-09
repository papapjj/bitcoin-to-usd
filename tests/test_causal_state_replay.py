from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from self_growing_trader.causal_state_replay import (
    BEHAVIOUR_ACTION_CYCLE,
    CausalStateReplayConfig,
    V16_FIXED_ENTRY_OFFSET_ATR,
    augment_snapshot_file,
)


def _write_snapshot(path: Path, rows: int = 60) -> None:
    header = [
        "schema_version", "feature_schema_hash", "sample_time", "symbol", "timeframe",
        "open_bid", "high_bid", "low_bid", "close_bid", "atr", "spread_price",
        "point", "pending_min_distance", "digits", "causal_input", "decision_bar_time",
        "tick_count", "tick_time_msc_pipe", "tick_bid_pipe", "tick_ask_pipe", "inputs_pipe",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        for index in range(rows):
            when = datetime(2025, 1, 1) + timedelta(minutes=index)
            decision = when + timedelta(minutes=1)
            price = 100.0
            bid = np.asarray([price, price, price, price], dtype=float)
            ask = bid + 0.05
            inputs = np.zeros(137, dtype=float)
            writer.writerow([
                2, "five_observation_v10_137", when.strftime("%Y.%m.%d %H:%M:%S"),
                "TEST", "PERIOD_M1", price, price + 0.1, price - 0.1, price,
                1.0, 0.05, 0.01, 0.02, 2, 1,
                decision.strftime("%Y.%m.%d %H:%M:%S"), 4,
                "|".join(str(1_000_000 + index * 60_000 + 20_000 * tick) for tick in range(4)),
                "|".join(f"{value:.5f}" for value in bid),
                "|".join(f"{value:.5f}" for value in ask),
                "|".join(f"{value:.10f}" for value in inputs),
            ])


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_state_replay_covers_all_seven_actions_and_populates_past_state(tmp_path) -> None:
    source = tmp_path / "source.tsv"
    output = tmp_path / "stateful.tsv"
    _write_snapshot(source)
    report = augment_snapshot_file(
        source,
        output,
        config=CausalStateReplayConfig(
            horizon_bars=4, pending_expiry_bars=1, cycle_bars=7,
        ),
    )
    rows = _read(output)
    assert set(int(row["state_replay_action"]) for row in rows[:43]) == set(range(7))
    assert report["production_eligible"] is False
    assert report["causal_contract"]["future_ticks_used_for_current_input"] is False

    # Row 7 emits MARKET_BUY while flat; row 8 sees only the position created
    # by row 7's current tick sequence.  It cannot see row 8 or later ticks.
    before = np.asarray([float(x) for x in rows[7]["inputs_pipe"].split("|")])
    after = np.asarray([float(x) for x in rows[8]["inputs_pipe"].split("|")])
    assert int(rows[7]["state_replay_action"]) == 1
    assert before[29] == 0.0
    assert after[29] == 1.0
    assert after[30] == 1.0
    assert after[69] == 0.2
    assert report["filled_action_counts"]["MARKET_BUY"] == 2


def test_state_replay_is_causal_to_future_tick_changes(tmp_path) -> None:
    source_a = tmp_path / "source_a.tsv"
    source_b = tmp_path / "source_b.tsv"
    output_a = tmp_path / "stateful_a.tsv"
    output_b = tmp_path / "stateful_b.tsv"
    _write_snapshot(source_a)
    source_b.write_bytes(source_a.read_bytes())

    # Change ticks only after row 30.  Inputs through row 30 must remain byte
    # equal because replay emits before processing each row's ticks.
    rows = _read(source_b)
    for row in rows[31:]:
        row["tick_bid_pipe"] = "|".join(["200.0"] * 4)
        row["tick_ask_pipe"] = "|".join(["200.05"] * 4)
    with source_b.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    cfg = CausalStateReplayConfig(horizon_bars=4, pending_expiry_bars=1, cycle_bars=7)
    augment_snapshot_file(source_a, output_a, config=cfg)
    augment_snapshot_file(source_b, output_b, config=cfg)
    a, b = _read(output_a), _read(output_b)
    for index in range(31):
        assert a[index]["inputs_pipe"] == b[index]["inputs_pipe"]


def test_state_replay_uses_signed_short_position_encoding(tmp_path) -> None:
    source = tmp_path / "source.tsv"
    output = tmp_path / "stateful.tsv"
    _write_snapshot(source, rows=24)
    # With a two-bar finite lifetime, the second complete cycle opens a short
    # after the earlier long has already been closed.  This checks the exact
    # side sign and the account/execution blocks used by ComputeAllInputs.
    report = augment_snapshot_file(
        source,
        output,
        config=CausalStateReplayConfig(
            horizon_bars=2, pending_expiry_bars=1, cycle_bars=7,
        ),
    )
    rows = _read(output)
    short_state = np.asarray([float(x) for x in rows[15]["inputs_pipe"].split("|")])
    assert short_state.shape == (137,)
    assert short_state[29] == 1.0
    assert short_state[30] == -1.0
    assert short_state[69] == 0.2
    assert short_state[70] == 0.01
    # The distance-to-stop and distance-to-target slots are populated for the
    # signed short in the same positions as the EA wire contract.
    assert short_state[107] > 0.0
    assert short_state[108] > 0.0
    assert report["feature_count"] == 137


def test_state_replay_fails_closed_on_non_137_wire_input(tmp_path) -> None:
    source = tmp_path / "source.tsv"
    output = tmp_path / "stateful.tsv"
    _write_snapshot(source, rows=8)
    rows = _read(source)
    rows[0]["inputs_pipe"] = "|".join(["0"] * 136)
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="137"):
        augment_snapshot_file(
            source,
            output,
            config=CausalStateReplayConfig(horizon_bars=2, pending_expiry_bars=1, cycle_bars=7),
        )


def test_state_replay_default_uses_v16_effective_offset() -> None:
    assert CausalStateReplayConfig().fixed_entry_offset_atr == V16_FIXED_ENTRY_OFFSET_ATR
