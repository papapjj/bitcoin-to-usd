from __future__ import annotations

import torch

from scripts.evaluate_five_temporal_v3 import (
    causal_intervention_audit,
    temporal_impulse_retention_audit,
)
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy


def test_future_intervention_and_agent_reset_cannot_leak_into_temporal_prefix() -> None:
    torch.manual_seed(1)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    result = causal_intervention_audit(network, steps=8, cutoff=4, seed=3)

    assert result["passed"] is True
    assert result["max_abs_prefix_policy_probability_delta"] == 0.0
    assert result["max_abs_prefix_lstm_hidden_delta"] == 0.0
    assert result["agent_boundary_reset"]["max_abs_temporal_token_delta_after_reset"] == 0.0


def test_temporal_ema_retains_recoverable_impulse_at_lag_32_and_128() -> None:
    torch.manual_seed(2)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    result = temporal_impulse_retention_audit(network)

    assert result["passed"] is True
    assert result["required_retention"]["lag_32_from_horizon_32"] > 0.35
    assert result["required_retention"]["lag_128_from_horizon_128"] > 0.35
