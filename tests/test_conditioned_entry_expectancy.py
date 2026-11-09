from __future__ import annotations

import numpy as np

from self_growing_trader.conditioned_entry_expectancy import (
    ConditionedEntryExpectancyLedger,
    entry_condition,
)
from self_growing_trader.protocol_contract import INPUT_COUNT


def test_condition_distinguishes_chasing_market_entry_from_breakout_stop() -> None:
    wire = np.zeros(INPUT_COUNT, dtype=float)
    wire[11] = 0.9
    wire[3] = 1.2
    wire[5] = 0.3
    wire[119] = 1.4

    market = entry_condition(wire, action_index=1, context="CTX_TANG_MANH")
    stop = entry_condition(wire, action_index=5, context="CTX_TANG_MANH")

    assert market["action"] == "MARKET_BUY"
    assert stop["action"] == "BUY_STOP"
    assert market["trend"] == "up"
    assert market["location"] == "far"
    assert market["volatility"].startswith("expanding")


def test_conditioned_expectancy_is_shrunk_until_minimum_evidence() -> None:
    ledger = ConditionedEntryExpectancyLedger(min_samples=4, prior_strength=12.0)
    condition = {
        "context": "CTX_TANG_MANH", "action": "BUY_STOP", "action_index": 5,
        "trend": "up", "location": "near", "volatility": "expanding",
    }

    _key, early = ledger.observe(condition, 3.0, trace_id="one")
    assert early.eligible is False
    assert early.mean_net_r < 3.0
    for index in range(3):
        _key, final = ledger.observe(condition, 1.0, trace_id=f"more-{index}")

    assert final.sample_count == 4
    assert final.eligible is True
    assert ledger.report()["telemetry_only"] is True


def test_report_separates_observed_samples_from_eligible_condition_support() -> None:
    """A sparse condition must not be presented as zero observed evidence."""

    ledger = ConditionedEntryExpectancyLedger(min_samples=4, prior_strength=12.0)
    condition = {
        "context": "CTX_TANG_MANH", "action": "MARKET_BUY", "action_index": 1,
        "trend": "up", "location": "mid", "volatility": "expanding",
    }

    for index in range(3):
        ledger.observe(condition, 0.5, trace_id=f"sparse-{index}")

    sparse = ledger.report()
    assert sparse["record_count"] == 1
    assert sparse["sample_count"] == 3
    assert sparse["eligible_record_count"] == 0
    assert sparse["eligible_sample_count"] == 0
    assert sparse["max_record_sample_count"] == 3
    assert sparse["samples_to_first_eligible"] == 1

    ledger.observe(condition, 0.5, trace_id="mature")
    mature = ledger.report()
    assert mature["sample_count"] == 4
    assert mature["eligible_record_count"] == 1
    assert mature["eligible_sample_count"] == 4
    assert mature["samples_to_first_eligible"] == 0
