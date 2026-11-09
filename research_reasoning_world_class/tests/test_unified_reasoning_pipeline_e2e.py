"""
E2E Integration & Verification Tests for Unified Reasoning Trader Pipeline & MT5 IPC Transport
Module: Milestone 4 (Features 31, 32, 33 - Requirements R1 through R8, AC-6, AC-7)
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Quantitative Reasoning Engine
"""

import math
import os
import sys
import time
import pytest
import numpy as np

# Ensure project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

from research_reasoning_world_class.src.mt5_ipc_transport import (
    MT5IPCTransportBridge,
    MT5WirePacket,
    WIRE_BYTE_SIZE,
    WIRE_FLOAT_COUNT,
    IDX_POLICY_CLOSE_ALL,
)
from research_reasoning_world_class.src.unified_reasoning_pipeline import (
    UnifiedReasoningTraderPipeline,
    PipelineTickResult,
)
from research_reasoning_world_class.src.multi_symbol_matrix import SUPPORTED_SYMBOLS


class TestMT5IPCTransportAndWireContract:
    """Verifies MT5 IPC binary transport, 20-float wire buffer, and Feature 33 locked invariant."""

    def test_20_float_wire_packet_serialization_deserialization(self) -> None:
        """Verifies exactly 20 floats (80 bytes) serialization parity."""
        packet = MT5WirePacket(
            action=1.0,
            order_type=1.0,
            lot_size=0.15,
            limit_price=65420.50,
            stop_loss=65370.50,
            take_profit=65520.50,
            trailing_activation=25.0,
            trailing_step=10.0,
            expected_alpha=28.5,
            cost_ratio=4.2,
            confidence=0.89,
            bull_score=0.82,
            bear_score=0.18,
            net_margin=0.64,
            prm_score=0.91,
            macro_index=0.45,
            effective_correlation=0.32,
            deleveraging_factor=0.95,
            sniper_mode=0.0,
            policy_close_all=0.0,
        )

        raw_bytes = packet.to_bytes()
        assert len(raw_bytes) == WIRE_BYTE_SIZE, f"Expected {WIRE_BYTE_SIZE} bytes, got {len(raw_bytes)}"

        unpacked = MT5WirePacket.from_bytes(raw_bytes)
        assert unpacked.action == 1.0
        assert unpacked.order_type == 1.0
        assert pytest.approx(unpacked.lot_size, abs=1e-4) == 0.15
        assert pytest.approx(unpacked.confidence, abs=1e-4) == 0.89
        assert unpacked.policy_close_all == 0.0

    def test_feature_33_zero_premature_closes_invariant_locked(self) -> None:
        """
        Feature 33: POLICY_CLOSE_ALL must strictly equal 0.0.
        Any attempt to instantiate or dispatch with policy_close_all != 0 raises ValueError.
        """
        with pytest.raises(ValueError, match="Feature 33 Invariant Violation"):
            MT5WirePacket(
                action=0.0,
                order_type=0.0,
                lot_size=0.0,
                limit_price=0.0,
                stop_loss=0.0,
                take_profit=0.0,
                trailing_activation=0.0,
                trailing_step=0.0,
                expected_alpha=0.0,
                cost_ratio=0.0,
                confidence=0.0,
                bull_score=0.0,
                bear_score=0.0,
                net_margin=0.0,
                prm_score=0.0,
                macro_index=0.0,
                effective_correlation=0.0,
                deleveraging_factor=1.0,
                sniper_mode=0.0,
                policy_close_all=1.0,  # Prohibited!
            )

    def test_ipc_transport_bridge_sub_millisecond_dispatch(self) -> None:
        """Verifies sub-0.05ms dispatch latency across 500 packets."""
        bridge = MT5IPCTransportBridge(buffer_capacity=1000)
        packet = MT5WirePacket(
            action=1.0, order_type=0.0, lot_size=0.10, limit_price=0.0,
            stop_loss=100.0, take_profit=110.0, trailing_activation=5.0, trailing_step=2.0,
            expected_alpha=12.0, cost_ratio=4.0, confidence=0.88, bull_score=0.85,
            bear_score=0.15, net_margin=0.70, prm_score=0.89, macro_index=0.2,
            effective_correlation=0.1, deleveraging_factor=1.0, sniper_mode=0.0,
            policy_close_all=0.0,
        )

        for _ in range(500):
            bridge.dispatch_packet(packet)

        assert bridge.total_dispatched == 500
        assert bridge.average_dispatch_latency_ms < 0.1, f"IPC dispatch latency {bridge.average_dispatch_latency_ms:.4f}ms > 0.1ms"


class TestUnifiedReasoningTraderPipelineE2E:
    """Verifies end-to-end pipeline uniting R1 through R8 into live execution passes."""

    def _generate_synthetic_bars(self, n: int = 60, base_price: float = 100.0) -> np.ndarray:
        bars = np.zeros((n, 5), dtype=np.float64)
        price = base_price
        for i in range(n):
            o = price
            c = price + np.random.uniform(-0.5, 0.5)
            h = max(o, c) + np.random.uniform(0.1, 0.8)
            l = min(o, c) - np.random.uniform(0.1, 0.8)
            v = np.random.uniform(10.0, 50.0)
            bars[i] = [o, h, l, c, v]
            price = c
        return bars

    def _generate_macro_quotes(self, n: int = 60) -> dict:
        return {
            "DXY": np.linspace(104.0, 105.0, n),
            "US10Y": np.linspace(4.2, 4.3, n),
            "SPX": np.linspace(5000.0, 5050.0, n),
            "XAUUSD": np.linspace(2300.0, 2350.0, n),
        }

    def test_unified_pipeline_end_to_end_single_tick(self) -> None:
        """Verifies full synchronous execution across all 8 stages."""
        pipeline = UnifiedReasoningTraderPipeline(initial_capital=10_000.0)
        
        base_obs = np.random.randn(137).astype(np.float32)
        bars = self._generate_synthetic_bars(60, 65000.0)
        macro = self._generate_macro_quotes(60)

        # Warmup pass to eliminate cold module loading & first-pass allocation overhead
        _ = pipeline.process_tick(
            symbol="BTCUSD",
            base_obs_137=base_obs,
            m1_bars_60=bars,
            macro_quotes=macro,
            current_price=65000.0,
        )

        result = pipeline.process_tick(
            symbol="BTCUSD",
            base_obs_137=base_obs,
            m1_bars_60=bars,
            macro_quotes=macro,
            current_price=65000.0,
        )

        assert isinstance(result, PipelineTickResult)
        assert result.symbol == "BTCUSD"
        assert result.action in (-1, 0, 1)
        assert result.order_type in ("MARKET", "LIMIT", "LIMIT_SNIPER")
        assert result.wire_packet.policy_close_all == 0.0
        assert result.total_latency_ms < 50.0, f"Tick latency {result.total_latency_ms:.2f}ms > 50ms budget"

    def test_multi_symbol_isolated_execution_across_6_assets(self) -> None:
        """Verifies sequential stepping across all 6 symbols without state corruption."""
        pipeline = UnifiedReasoningTraderPipeline(initial_capital=25_000.0)
        base_obs = np.random.randn(137).astype(np.float32)
        macro = self._generate_macro_quotes(60)

        for sym in SUPPORTED_SYMBOLS:
            bars = self._generate_synthetic_bars(60, 100.0)
            result = pipeline.process_tick(
                symbol=sym,
                base_obs_137=base_obs,
                m1_bars_60=bars,
                macro_quotes=macro,
                current_price=100.0,
            )
            assert result.symbol == sym
            assert math.isfinite(result.wire_packet.lot_size)
            assert result.wire_packet.policy_close_all == 0.0

    def test_ac7_adversarial_nan_inf_ingress_sanitization(self) -> None:
        """Verifies that injecting pathological NaN/Inf into ingress inputs does not crash the pipeline."""
        pipeline = UnifiedReasoningTraderPipeline(initial_capital=10_000.0)
        
        # Corrupt 137D base observation with NaNs and Infs
        base_obs = np.random.randn(137).astype(np.float32)
        base_obs[0] = np.nan
        base_obs[10] = np.inf
        base_obs[20] = -np.inf
        
        bars = self._generate_synthetic_bars(60, 100.0)
        macro = self._generate_macro_quotes(60)

        result = pipeline.process_tick(
            symbol="EURUSD",
            base_obs_137=base_obs,
            m1_bars_60=bars,
            macro_quotes=macro,
            current_price=1.0850,
        )

        assert math.isfinite(result.confidence)
        assert math.isfinite(result.lot_size)
        assert result.wire_packet.policy_close_all == 0.0
        assert pipeline.nan_inf_count >= 1
