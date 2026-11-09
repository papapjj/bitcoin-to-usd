from __future__ import annotations

import json

import numpy as np

from self_growing_trader.model1_python_backend import Model1ReplayRequestStream
from self_growing_trader.model1_replay import Model1OracleTape


def test_replay_stream_never_exposes_future_execution_rows(tmp_path) -> None:
    target = tmp_path / "tape.npz"
    metadata = json.dumps({
        "event_id": "1735689620-1", "positions_pipe": "", "orders_pipe": "",
        "execution_pipe": "", "balance": 10000, "equity": 10000,
        "margin": 0, "free_margin": 10000, "margin_level_pct": 0, "leverage": 100,
        "drawdown_pct": 0, "spread_points": 10, "day_trades": 0,
    }, separators=(",", ":"))
    header = "logged_at\tack_sequence\ttransaction_type"
    np.savez_compressed(
        target, schema_version=np.asarray([1], dtype=np.int16),
        inputs=np.zeros((1, 137)), outputs=np.zeros((1, 20)),
        actions=np.asarray([0], dtype=np.int16), metadata=np.asarray([metadata]),
        execution_lines=np.asarray([
            header,
            "2025.01.01 00:00:20\t1\tREQUEST",
            "2025.01.01 00:00:40\t1\tDEAL_ADD",
        ]),
    )
    tape = Model1OracleTape(target)
    ledger = tmp_path / "runtime" / "execution.csv"
    stream = Model1ReplayRequestStream(tape, ledger)
    next(stream)
    causal = ledger.read_text(encoding="utf-8")
    assert "00:00:20" in causal
    assert "00:00:40" not in causal
    try:
        next(stream)
    except StopIteration:
        pass
    final = ledger.read_text(encoding="utf-8")
    assert "00:00:40" in final
