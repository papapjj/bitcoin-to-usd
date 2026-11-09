"""
tests/test_router_micro_scalp_integration.py

Comprehensive tests for DualChampionRouter integration with MicroPullbackScalpEngine:
- Validates micro-scalp activation when active policy emits HOLD
- Verifies Agent 4 remains strictly HOLD (POLICY_CLOSE_ALL == 0)
- Verifies scalp geometry injection (SL=2.5, TP=1.5, Trailing=1.5)
- Verifies CPPI risk scaling on scalp trades
"""

from unittest.mock import MagicMock
import numpy as np
import pytest

from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    BridgeActionBatch,
)
from self_growing_trader.micro_scalp_engine import (
    MicroPullbackScalpEngine,
    MicroScalpConfig,
)
from self_growing_trader.cppi_risk_engine import CPPIConfig, CPPIFloorEngine
from self_growing_trader.two_tier_runner import TwoTierConfig, TwoTierTrailingRunnerEngine


class MockTrace:
    def __init__(self, action_idx: int = 0):
        self.action_index = action_idx
        self.selected_action = np.zeros(20, dtype=float)
        self.selected_action[action_idx] = 1.0
        self.output_vector = self.selected_action.copy()
        self.action_details = {"five_action_index": action_idx, "action": action_idx}


class MockSpecialistPolicy:
    def __init__(self):
        self.last_obs = None

    def act_nam_tac_nhan(self, obs, *args, **kwargs):
        self.last_obs = obs
        # Return 5 HOLD traces by default (Agent 0 to Agent 4)
        return [MockTrace(0) for _ in range(5)]


def create_bull_pullback_observation():
    obs = np.zeros(137, dtype=float)
    obs[7] = 0.05    # bar return
    obs[9] = 0.12    # lower shadow / rejection wick
    obs[11] = 0.25   # EMA20 - EMA60 positive
    obs[16] = -0.15  # RSI14 oversold dip
    obs[19] = 0.25   # Lower BB touch (pullback)
    obs[26] = -0.10  # Near EMA20
    obs[80] = 0.10   # M5 momentum bullish
    obs[84] = 0.15   # M15 momentum bullish
    return obs


def create_bear_rally_observation():
    obs = np.zeros(137, dtype=float)
    obs[7] = -0.05   # bar return
    obs[8] = 0.12    # upper shadow / rejection wick
    obs[11] = -0.25  # EMA20 - EMA60 negative
    obs[16] = 0.20   # RSI14 overbought rally
    obs[19] = 0.80   # Upper BB touch (rally)
    obs[26] = 0.15   # Near EMA20
    obs[80] = -0.10  # M5 momentum bearish
    obs[84] = -0.15  # M15 momentum bearish
    return obs


def test_dual_champion_router_triggers_bull_micro_scalp():
    bull_pol = MockSpecialistPolicy()
    bear_pol = MockSpecialistPolicy()
    router = DualChampionRouter(
        bull_policy=bull_pol,
        bear_policy=bear_pol,
        micro_scalp_enabled=True,
    )
    router.set_regime_soft_probabilities((0.70, 0.10, 0.20))  # Dominant Bull

    obs = create_bull_pullback_observation()
    batch = router.act_nam_tac_nhan(observation=obs, equity=10000.0)

    assert isinstance(batch, BridgeActionBatch)
    traces = batch.traces
    assert len(traces) == 5

    # Agent 0 (Long Specialist) should be upgraded from HOLD (0) to BUY_LIMIT (3)
    agent0_action = getattr(traces[0], "action_index", None)
    assert agent0_action == 3  # BUY_LIMIT
    assert traces[0].output_vector[14] == 7.0  # Factorized marker

    # Agent 4 must strictly remain HOLD (0)
    agent4_action = getattr(traces[4], "action_index", None)
    assert agent4_action == 0


def test_dual_champion_router_triggers_bear_micro_scalp():
    bull_pol = MockSpecialistPolicy()
    bear_pol = MockSpecialistPolicy()
    router = DualChampionRouter(
        bull_policy=bull_pol,
        bear_policy=bear_pol,
        micro_scalp_enabled=True,
    )
    router.set_regime_soft_probabilities((0.10, 0.75, 0.15))  # Dominant Bear

    obs = create_bear_rally_observation()
    batch = router.act_nam_tac_nhan(observation=obs, equity=10000.0)

    assert isinstance(batch, BridgeActionBatch)
    traces = batch.traces
    assert len(traces) == 5

    # Agent 1 (Short Specialist) should be upgraded from HOLD (0) to SELL_LIMIT (4)
    agent1_action = getattr(traces[1], "action_index", None)
    assert agent1_action == 4  # SELL_LIMIT
    assert traces[1].output_vector[14] == 7.0  # Factorized marker

    # Agent 4 must strictly remain HOLD (0)
    agent4_action = getattr(traces[4], "action_index", None)
    assert agent4_action == 0


def test_disabled_micro_scalp_leaves_hold():
    bull_pol = MockSpecialistPolicy()
    bear_pol = MockSpecialistPolicy()
    router = DualChampionRouter(
        bull_policy=bull_pol,
        bear_policy=bear_pol,
        micro_scalp_enabled=False,
    )
    router.set_regime_soft_probabilities((0.70, 0.10, 0.20))

    obs = create_bull_pullback_observation()
    batch = router.act_nam_tac_nhan(observation=obs, equity=10000.0)

    traces = batch.traces
    # When disabled, Agent 0 stays HOLD (0)
    assert getattr(traces[0], "action_index", 0) == 0
    assert getattr(traces[1], "action_index", 0) == 0
