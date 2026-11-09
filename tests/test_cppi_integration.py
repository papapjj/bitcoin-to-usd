"""
tests/test_cppi_integration.py

Comprehensive Integration Test Suite for Milestone 3:
- Router wiring & CPPI/Two-Tier engine instantiation
- Dynamic risk application (CPPI Floor, Asymmetric Volatility Pumping, 50% cushion cap, 3-TF confluence)
- Agent 2 position management dispatch (Partial Close 50% at 10.0 ATR & Chandelier 3.0 ATR trailing stop)
- Agent 4 strict HOLD lock & permanent POLICY_CLOSE_ALL == 0 invariant
- MT5 IPC Contract: 137 in / 20 out, dynamic risk_pct in column 10 of self_growing_action.csv, < 5.0 ms SLA
- Inactive specialist recurrent memory isolation (L_inf = 0.0 bit-for-bit state freezing)
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from self_growing_trader.cppi_risk_engine import CPPIConfig, CPPIFloorEngine
from self_growing_trader.two_tier_runner import (
    ACTION_HOLD,
    ACTION_MANAGE_POSITION,
    MANAGEMENT_MODE_PARTIAL_CLOSE,
    CLOSE_BIN_50_PCT,
    TwoTierConfig,
    TwoTierTrailingRunnerEngine,
)
from self_growing_trader.execution_contract import (
    unpack_geometry_choices,
    pack_geometry_choices,
    TRAILING_ATR_BINS,
)
from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    BridgeActionBatch,
    INPUT_COUNT,
    OUTPUT_COUNT,
)
from self_growing_trader.five_shared_policy import FiveAgentMemory, FIVE_LSTM_SIZE
from self_growing_trader.five_agent_authority import HOLD_ACTION
from self_growing_trader.mt5_runner import (
    BridgeAgentAction,
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    RiskConfig,
    OrderConfig,
)
from self_growing_trader.orders import Action


# ==============================================================================
# TEST FIXTURES & REALISTIC RECURRENT POLICY MOCKS
# ==============================================================================

@dataclass
class IntegrationMockTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(OUTPUT_COUNT, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    confidence: float = 0.90
    action_details: dict[str, Any] = field(default_factory=dict)
    positions_pipe: str = ""
    dynamic_risk: float | None = None
    risk_pct: float | None = None


class IntegrationRecurrentSpecialist:
    """Specialist mock maintaining real 5-agent x 4096-D recurrent LSTM memory tensors."""

    def __init__(self, name: str, *, action_map: dict[int, int] | None = None, seed: int = 42, base_risk: float = 0.005) -> None:
        self.name = name
        self.input_size = INPUT_COUNT
        self.output_size = OUTPUT_COUNT
        self.step_count = 0
        self.action_map = action_map or {}  # Maps agent_id -> action_id to emit
        self.base_risk = float(base_risk)

        rng = np.random.RandomState(seed)
        self.initial_hidden = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)
        self.initial_cell = rng.randn(5, FIVE_LSTM_SIZE).astype(np.float32)

        self.hidden_state = self.initial_hidden.copy()
        self.cell_state = self.initial_cell.copy()

        self.five_agent_memories = [
            FiveAgentMemory(
                hidden=torch.from_numpy(self.hidden_state[i].copy()),
                cell=torch.from_numpy(self.cell_state[i].copy()),
                previous_action=torch.zeros((), dtype=torch.long),
            )
            for i in range(5)
        ]

        self.experts = {f"expert_{name}": MagicMock()}
        self.nhan_dien = MagicMock()
        self.registry = MagicMock()
        self.cau_hinh = MagicMock()
        self.cau_hinh.che_do_quyet_dinh = "nam_tac_nhan_dung_chung"

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[IntegrationMockTrace, ...]:
        self.step_count += 1

        # Advance recurrent state by observation-dependent delta
        obs_norm = float(np.linalg.norm(observation)) if observation.size else 1.0
        delta = 0.005 * (obs_norm + float(self.step_count))
        self.hidden_state += delta
        self.cell_state += delta * 2.0
        for mem in self.five_agent_memories:
            mem.hidden.add_(delta)
            mem.cell.add_(delta * 2.0)

        traces = []
        for agent_id in range(5):
            act_idx = self.action_map.get(agent_id, int(HOLD_ACTION))
            outputs = np.zeros(OUTPUT_COUNT, dtype=float)
            outputs[act_idx] = 2.0  # Highest logit for intended action
            # Default entry geometry: index 12 risk ~ 0.01
            outputs[12] = 0.0

            tr = IntegrationMockTrace(
                trace_id=f"{self.name}_step_{self.step_count}_agent_{agent_id}",
                selected_action=outputs,
                selected_idx=act_idx,
                confidence=0.88,
                action_details={
                    "five_agent_index": agent_id,
                    "five_action_index": act_idx,
                    "policy_name": self.name,
                    "base_risk": self.base_risk,
                },
            )
            traces.append(tr)

        return tuple(traces)

    def ensure_compatible(self, *args: Any, **kwargs: Any) -> bool:
        return True

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state = self.initial_hidden.copy()
        self.cell_state = self.initial_cell.copy()
        for i, mem in enumerate(self.five_agent_memories):
            mem.hidden.copy_(torch.from_numpy(self.hidden_state[i]))
            mem.cell.copy_(torch.from_numpy(self.cell_state[i]))


def make_test_request(
    equity: float = 10000.0,
    balance: float = 10000.0,
    sequence: int = 1,
    positions_pipe: str = "",
    h1_trend: float = 0.0,
    m15_trend: float = 0.0,
    m5_velocity: float = 0.0,
    regime_probabilities: tuple[float, float, float] | None = None,
) -> BridgeRequest:
    """Helper to build a deterministic BridgeRequest with 137 input features."""
    inputs = np.zeros(INPUT_COUNT, dtype=float)
    inputs[88] = float(h1_trend)      # H1 Trend
    inputs[84] = float(m15_trend)     # M15 Trend
    inputs[81] = float(m5_velocity)   # M5 Velocity

    return BridgeRequest(
        sequence=sequence,
        inputs=inputs,
        raw_inputs=inputs.copy(),
        event_id=f"evt_{sequence}",
        balance=float(balance),
        equity=float(equity),
        positions_pipe=str(positions_pipe),
        regime_probabilities=regime_probabilities or (0.70, 0.15, 0.15),
        regime_name="BULL",
        regime_index=0,
    )


# ==============================================================================
# SUITE 1: ROUTER WIRING & ENGINE INSTANTIATION
# ==============================================================================

class TestRouterWiringAndInstantiation:
    """Verify clean instantiation, alias exposure, and state synchronization."""

    def test_router_instantiates_cppi_and_two_tier_engines(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        assert isinstance(router.cppi_engine, CPPIFloorEngine)
        assert isinstance(router.two_tier_runner, TwoTierTrailingRunnerEngine)
        # Check aliases
        assert router.two_tier_engine is router.two_tier_runner
        assert router.runner_engine is router.two_tier_runner

    def test_custom_config_injection(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        cppi_cfg = CPPIConfig(max_allowed_dd=0.06, base_alpha=0.5, pumping_slope=3.5)
        runner_cfg = TwoTierConfig(partial_tp_atr=8.0, trailing_atr=2.5)

        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            cppi_config=cppi_cfg,
            two_tier_config=runner_cfg,
        )

        assert router.cppi_engine.config.max_allowed_dd == 0.06
        assert router.cppi_engine.config.base_alpha == 0.5
        assert router.two_tier_runner.config.partial_tp_atr == 8.0
        assert router.two_tier_runner.config.trailing_atr == 2.5

    def test_router_reset_runtime_state_resets_both_engines(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Update equity to ratchet peak equity
        router.set_account_context({"equity": 50000.0})
        assert router.cppi_engine.peak_equity == 50000.0

        # Register position in runner
        router.two_tier_runner.register_position("pos_1", entry_price=100.0, direction=1, entry_atr=2.0)
        assert router.two_tier_runner.active_position_count == 1

        # Reset runtime state
        router.reset_runtime_state()
        assert router.cppi_engine.peak_equity == 0.0
        assert router.two_tier_runner.active_position_count == 0

    def test_set_account_context_syncs_equity_to_cppi_engine(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        router.set_account_context({"equity": 25000.0}, positions_pipe="test_pipe")
        assert router.cppi_engine.peak_equity == 25000.0
        expected_floor = (1.0 - router.cppi_engine.config.max_allowed_dd) * 25000.0
        assert pytest.approx(router.cppi_engine.floor, rel=1e-5) == expected_floor
        assert router._last_positions_pipe == "test_pipe"


# ==============================================================================
# SUITE 2: DYNAMIC RISK COMPUTATION (CPPI ENGINE INTEGRATION)
# ==============================================================================

class TestDynamicRiskApplication:
    """Verify dynamic risk scaling on entry actions and strict 50% cushion cap."""

    def test_dynamic_risk_applied_on_agent0_buy_entry(self) -> None:
        # Agent 0 emits Action 1 (MARKET_BUY)
        bull = IntegrationRecurrentSpecialist("Bull", action_map={0: int(Action.MARKET_BUY)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={0: int(Action.MARKET_BUY)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        req = make_test_request(equity=10000.0, sequence=1)
        batch = router.act_nam_tac_nhan(req)

        assert isinstance(batch, BridgeActionBatch)
        assert batch.risk_pct is not None
        assert batch.risk_pct > 0.0

        # Check trace 0 has dynamic risk
        trace_0 = batch[0]
        assert trace_0.action_details.get("dynamic_risk") is not None
        assert trace_0.action_details["dynamic_risk"] == batch.risk_pct

        # Hard invariant: loss <= 50% cushion
        cushion = router.cppi_engine.cushion
        max_loss_pct = (0.50 * cushion) / req.equity
        assert batch.risk_pct <= max_loss_pct + 1e-9

    def test_dynamic_risk_applied_on_agent1_sell_entry(self) -> None:
        # Agent 1 emits Action 2 (MARKET_SELL)
        bull = IntegrationRecurrentSpecialist("Bull", action_map={1: int(Action.MARKET_SELL)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={1: int(Action.MARKET_SELL)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        req = make_test_request(equity=12000.0, sequence=2)
        batch = router.act_nam_tac_nhan(req)

        assert isinstance(batch, BridgeActionBatch)
        assert batch.risk_pct is not None
        assert batch.risk_pct > 0.0

        trace_1 = batch[1]
        assert trace_1.action_details.get("dynamic_risk") is not None
        assert trace_1.action_details["dynamic_risk"] == batch.risk_pct

    def test_dynamic_leverage_pumping_on_accumulated_profits(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull", action_map={0: int(Action.MARKET_BUY)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={0: int(Action.MARKET_BUY)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Establish peak at $10,000 (Floor at $9,200)
        router.set_account_context({"equity": 10000.0})

        # Drawdown state: equity = $9,350 (cushion = $150, cushion_ratio = 150/800 = 0.1875)
        # alpha = 0.3 + 4.7 * 0.1875 = 1.18125
        req_depressed = make_test_request(equity=9350.0, sequence=1)
        batch_depressed = router.act_nam_tac_nhan(req_depressed)
        risk_depressed = batch_depressed.risk_pct

        # Accumulated profit state: equity recovers to peak $10,000 (cushion = $800, cushion_ratio = 1.0)
        # alpha = 0.3 + 4.7 * 1.0 = 5.0 (leverage pumped > 4x above depressed state!)
        req_pumped = make_test_request(equity=10000.0, sequence=2)
        batch_pumped = router.act_nam_tac_nhan(req_pumped)
        risk_pumped = batch_pumped.risk_pct

        # Leverage pumping must expand risk significantly above depressed state
        assert risk_pumped is not None and risk_depressed is not None
        assert risk_pumped > risk_depressed
        assert risk_pumped >= 2.0 * risk_depressed

    def test_dynamic_risk_contraction_near_floor_prevents_breach(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull", action_map={0: int(Action.MARKET_BUY)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={0: int(Action.MARKET_BUY)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Establish peak at $10,000 (Floor at $9,200 for 8% DD)
        router.set_account_context({"equity": 10000.0})

        # Drawdown: equity collapses to $9,250 (only $50 cushion remaining!)
        req_near_floor = make_test_request(equity=9250.0, sequence=1)
        batch = router.act_nam_tac_nhan(req_near_floor)

        assert batch.risk_pct is not None
        # Max allowable risk is capped strictly by (0.50 * 50) / 9250 = 0.002702 (0.27%)
        hard_cushion_cap = (0.50 * 50.0) / 9250.0
        assert batch.risk_pct <= hard_cushion_cap + 1e-9

    def test_3tf_confluence_boost_applied_when_aligned(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull", action_map={0: int(Action.MARKET_BUY)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={0: int(Action.MARKET_BUY)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # 1. Neutral signals
        req_neutral = make_test_request(equity=10000.0, h1_trend=0.0, m15_trend=0.0, m5_velocity=0.0)
        batch_neutral = router.act_nam_tac_nhan(req_neutral)
        risk_neutral = batch_neutral.risk_pct

        # 2. Confluent signals (all positive for BUY)
        req_confluent = make_test_request(equity=10000.0, h1_trend=1.5, m15_trend=1.2, m5_velocity=0.8)
        batch_confluent = router.act_nam_tac_nhan(req_confluent)
        risk_confluent = batch_confluent.risk_pct

        assert risk_neutral is not None and risk_confluent is not None
        # 1.5x boost should expand risk
        assert risk_confluent > risk_neutral
        assert pytest.approx(risk_confluent, rel=1e-3) == 1.5 * risk_neutral


# ==============================================================================
# SUITE 3: AGENT 2 TWO-TIER RUNNER PARTIAL CLOSE DISPATCH
# ==============================================================================

class TestAgent2TwoTierRunnerDispatch:
    """Verify Agent 2 position management dispatch when excursion reaches 10.0 ATR."""

    def test_agent2_dispatches_partial_close_at_10_atr(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Pipe with open position at 2.0 R (= 10.0 ATR excursion with initial 5.0 ATR SL)
        # ticket=1001, side=1 (BUY), volume=1.0, entry=100.0, sl=95.0, tp=120.0, unrealized_r=2.0
        pipe = "1001:1:1.0:100.0:95.0:120.0:2.0:10"
        req = make_test_request(equity=10000.0, positions_pipe=pipe)

        batch = router.act_nam_tac_nhan(req)

        # Agent 2 must emit ACTION_MANAGE_POSITION (11)
        assert batch.actions[2] == int(ACTION_MANAGE_POSITION)

        trace_2 = batch[2]
        details = trace_2.action_details
        assert details["management_mode"] == 1  # Mode 1: Partial Close
        assert details["close_bin"] == 1        # CloseBin 1: 50% volume
        assert details["close_ratio"] == 0.50
        assert details["trailing_index"] == 6   # Index 6: 3.0 ATR Chandelier trailing
        assert details["trailing_atr"] == 3.0

        # Check geometry packed in output vector
        out = trace_2.selected_action
        assert out[11] == 1.0  # Logit for ACTION_MANAGE_POSITION
        assert out[14] >= 7.0  # Factorized contract marker
        sl_idx, tp_idx, trail_idx = unpack_geometry_choices(out[16])
        assert trail_idx == 6
        assert TRAILING_ATR_BINS[trail_idx] == 3.0

    def test_agent2_idempotency_after_tier1_partial_close(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        pipe = "1001:1:1.0:100.0:95.0:120.0:2.0:10"

        # Step 1: fires partial close
        req1 = make_test_request(equity=10000.0, sequence=1, positions_pipe=pipe)
        batch1 = router.act_nam_tac_nhan(req1)
        assert batch1.actions[2] == int(ACTION_MANAGE_POSITION)

        # Step 2: next bar, excursion is still >= 10.0 ATR (e.g. 2.2 R)
        pipe2 = "1001:1:0.5:100.0:101.0:120.0:2.2:11"
        req2 = make_test_request(equity=10000.0, sequence=2, positions_pipe=pipe2)
        batch2 = router.act_nam_tac_nhan(req2)

        # Partial close must NOT fire again (idempotent)!
        assert batch2.actions[2] == int(ACTION_HOLD)

    def test_agent2_remains_hold_when_excursion_below_threshold(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Unrealized R = 1.0 (= 5.0 ATR excursion < 10.0 ATR threshold)
        pipe = "1001:1:1.0:100.0:95.0:120.0:1.0:5"
        req = make_test_request(equity=10000.0, positions_pipe=pipe)
        batch = router.act_nam_tac_nhan(req)

        assert batch.actions[2] == int(ACTION_HOLD)


# ==============================================================================
# SUITE 4: AGENT 4 STRICT HOLD LOCK (POLICY_CLOSE_ALL == 0)
# ==============================================================================

class TestAgent4StrictHoldLock:
    """Verify permanent mathematical guarantee of POLICY_CLOSE_ALL == 0."""

    def test_agent4_intercepted_and_locked_to_hold_in_router(self) -> None:
        # Policy attempts to emit Action.CLOSE_ALL (10) on Agent 4
        bull = IntegrationRecurrentSpecialist("Bull", action_map={4: int(Action.CLOSE_ALL)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={4: int(Action.CLOSE_ALL)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        req = make_test_request(equity=10000.0)
        batch = router.act_nam_tac_nhan(req)

        # Agent 4 must be strictly locked to HOLD (0)
        assert batch.actions[4] == int(ACTION_HOLD)
        trace_4 = batch[4]
        assert trace_4.action_details.get("five_action_index") == 0
        assert trace_4.action_details.get("wire_action") == 0

        # Action 10 logit masked out in output vector
        out_4 = trace_4.selected_action
        assert out_4[0] == 1.0
        assert out_4[10] <= 0.0

    def test_agent4_wire_clamped_in_five_bridge_action_packet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            evaluator = MT5StrategyTesterEvaluator(
                MT5BacktestConfig(
                    runner_dir=Path(tmp_dir),
                    action_file="test_action.csv",
                    request_file="test_request.csv",
                    offline_replay=True,
                )
            )
            bull = IntegrationRecurrentSpecialist("Bull", action_map={4: int(Action.CLOSE_ALL)})
            bear = IntegrationRecurrentSpecialist("Bear", action_map={4: int(Action.CLOSE_ALL)})
            router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

            req = make_test_request(equity=10000.0)
            raw_traces = bull.act_nam_tac_nhan(req.inputs)

            # Pass traces where Agent 4 has action 10
            packet, _ = evaluator._five_bridge_action_packet(router, req, raw_traces)

            # In the resulting packet, Agent 4 must be clamped to HOLD
            agent_4_action = packet[4]
            assert agent_4_action.action == int(Action.HOLD)
            assert agent_4_action.outputs[10] <= 0.0

    def test_agent4_wire_clamped_in_write_action_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "self_growing_action.csv"
            req = make_test_request(equity=10000.0)

            # Construct packet where agent_id 4 maliciously has action 10
            actions = [
                BridgeAgentAction(agent_id=i, trace_id=f"t_{i}", action=int(Action.HOLD), score=0.0, confidence=0.9, outputs=np.zeros(20))
                for i in range(4)
            ]
            actions.append(
                BridgeAgentAction(agent_id=4, trace_id="t_4", action=int(Action.CLOSE_ALL), score=0.9, confidence=0.9, outputs=np.zeros(20))
            )

            MT5StrategyTesterEvaluator._write_action_batch(csv_path, req, actions)

            # Read back TSV
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                rows = list(reader)

            # Header is row 0; rows 1..5 are agent 0..4
            agent_4_row = rows[5]
            action_col = int(agent_4_row[6])  # 0-indexed column 6 is 'action'
            assert action_col == int(Action.HOLD)
            assert action_col != int(Action.CLOSE_ALL)


# ==============================================================================
# SUITE 5: MT5 IPC CONTRACT & COLUMN 10 RISK_PCT WRITING
# ==============================================================================

class TestMT5IPCContractAndInvariants:
    """Verify 137 in / 20 out contract, column 10 dynamic risk writing, and < 5.0 ms SLA."""

    def test_write_action_batch_writes_dynamic_risk_pct_to_column_10(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "self_growing_action.csv"
            req = make_test_request(equity=15000.0)

            # Agent 0 has dynamic risk_pct = 0.0487
            dynamic_risk = 0.0487
            actions = [
                BridgeAgentAction(
                    agent_id=0,
                    trace_id="t_0",
                    action=int(Action.MARKET_BUY),
                    score=1.5,
                    confidence=0.95,
                    outputs=np.zeros(20),
                    risk_pct=dynamic_risk,
                )
            ]
            for i in range(1, 5):
                actions.append(
                    BridgeAgentAction(
                        agent_id=i,
                        trace_id=f"t_{i}",
                        action=int(Action.HOLD),
                        score=0.0,
                        confidence=0.9,
                        outputs=np.zeros(20),
                    )
                )

            MT5StrategyTesterEvaluator._write_action_batch(csv_path, req, actions)

            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f, delimiter="\t")
                rows = list(reader)

            header = rows[0]
            # Column 10 (1-indexed) is index 9: 'risk_pct'
            assert header[9] == "risk_pct"

            agent_0_row = rows[1]
            written_risk = float(agent_0_row[9])
            assert pytest.approx(written_risk, rel=1e-5) == dynamic_risk

    def test_137_input_20_output_contract_intact(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        req = make_test_request()
        assert req.inputs.shape == (137,)

        batch = router.act_nam_tac_nhan(req)
        assert len(batch) == 5
        for item in batch:
            out = getattr(item, "selected_action", getattr(item, "output_vector", None))
            assert out is not None
            assert out.shape == (20,)

    def test_mean_ipc_latency_below_sla(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull")
        bear = IntegrationRecurrentSpecialist("Bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        req = make_test_request(equity=10000.0)

        # Warmup
        for _ in range(10):
            router.act_nam_tac_nhan(req)

        # Benchmark 200 consecutive cycles
        latencies = []
        for i in range(200):
            t0 = time.perf_counter()
            router.act_nam_tac_nhan(req)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)  # ms

        mean_latency = float(np.mean(latencies))
        p99_latency = float(np.percentile(latencies, 99))

        assert mean_latency < 1.0, f"Mean latency {mean_latency:.3f} ms exceeded 1.0 ms budget"
        assert p99_latency < 5.0, f"P99 latency {p99_latency:.3f} ms exceeded 5.0 ms SLA"


# ==============================================================================
# SUITE 6: INACTIVE SPECIALIST RECURRENT MEMORY ISOLATION (L_inf = 0.0)
# ==============================================================================

class TestRecurrentMemoryIsolation:
    """Verify inactive specialist hidden and cell states remain bit-for-bit frozen."""

    def test_recurrent_memory_strictly_frozen_during_cppi_routing(self) -> None:
        bull = IntegrationRecurrentSpecialist("Bull", action_map={0: int(Action.MARKET_BUY)})
        bear = IntegrationRecurrentSpecialist("Bear", action_map={1: int(Action.MARKET_SELL)})
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        # Baseline snapshots
        init_bear_hidden = bear.hidden_state.copy()
        init_bear_cell = bear.cell_state.copy()

        # Step 50 times in BULL regime (P_Bull = 0.80)
        for i in range(50):
            req = make_test_request(equity=10000.0 + i * 100.0, sequence=i + 1, regime_probabilities=(0.80, 0.10, 0.10))
            router.act_nam_tac_nhan(req)

        # Bull policy must have evolved
        assert bull.step_count == 50

        # Bear policy MUST be 100% frozen bit-for-bit (L_inf = 0.0)
        diff_hidden = np.max(np.abs(bear.hidden_state - init_bear_hidden))
        diff_cell = np.max(np.abs(bear.cell_state - init_bear_cell))
        assert diff_hidden == 0.0, f"Bear hidden state leaked: diff={diff_hidden}"
        assert diff_cell == 0.0, f"Bear cell state leaked: diff={diff_cell}"

        # Now step 50 times in BEAR regime (P_Bear = 0.80)
        curr_bull_hidden = bull.hidden_state.copy()
        curr_bull_cell = bull.cell_state.copy()

        for i in range(50):
            req = make_test_request(equity=15000.0 + i * 100.0, sequence=i + 51, regime_probabilities=(0.10, 0.80, 0.10))
            router.act_nam_tac_nhan(req)

        # Bear evolved
        assert bear.step_count == 50

        # Bull policy MUST now remain 100% frozen bit-for-bit (L_inf = 0.0)
        diff_bull_h = np.max(np.abs(bull.hidden_state - curr_bull_hidden))
        diff_bull_c = np.max(np.abs(bull.cell_state - curr_bull_cell))
        assert diff_bull_h == 0.0, f"Bull hidden state leaked: diff={diff_bull_h}"
        assert diff_bull_c == 0.0, f"Bull cell state leaked: diff={diff_bull_c}"
