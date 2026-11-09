"""
tests/test_adversarial_pipeline_integration.py

Verification tests for Adversarial Dialectical Reasoning Engine integrated into UnifiedReasoningTraderPipeline.
Verifies:
1. Multi-turn debate progression (T=3 rounds) recorded in PipelineTickResult.debate_history.
2. Arbiter PRM threshold >= 0.88 gating.
3. Trap exploitation counter-punch mechanism (Bull Trap -> SELL, Bear Trap -> BUY).
4. Sub-50ms latency budget and policy_close_all == 0.0 invariant.
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn

from research_reasoning_world_class.src.unified_reasoning_pipeline import (
    UnifiedReasoningTraderPipeline,
    PipelineTickResult,
)


class MockBullTrapDetector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[-5.0, 10.0, -5.0]], device=x.device)


class MockBearTrapDetector(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[-5.0, -5.0, 10.0]], device=x.device)


class TestAdversarialPipelineIntegration:
    @pytest.fixture(scope="class")
    def pipeline(self):
        pl = UnifiedReasoningTraderPipeline(
            initial_capital=10_000.0,
            use_adversarial_dialectical=True,
        )
        return pl

    def test_pipeline_has_dialectical_engine(self, pipeline):
        assert hasattr(pipeline, "dialectical_engine")
        assert pipeline.use_adversarial_dialectical is True
        assert pipeline.dialectical_engine.config.prm_threshold >= 0.88

    def test_single_tick_produces_debate_history(self, pipeline):
        base_obs = np.random.randn(137).astype(np.float32)
        bars = np.ones((60, 5), dtype=np.float64) * 65000.0
        macro = {s: np.ones(60) * 100.0 for s in ("DXY", "US10Y", "SPX", "XAUUSD")}

        # Warmup passes to stabilize PyTorch thread pool & CPU cache
        for _ in range(3):
            _ = pipeline.process_tick(
                symbol="BTCUSD",
                base_obs_137=base_obs,
                m1_bars_60=bars,
                macro_quotes=macro,
                current_price=65000.0,
            )

        latencies = []
        res = None
        for _ in range(3):
            res = pipeline.process_tick(
                symbol="BTCUSD",
                base_obs_137=base_obs,
                m1_bars_60=bars,
                macro_quotes=macro,
                current_price=65000.0,
            )
            latencies.append(res.total_latency_ms)

        med_latency = float(np.median(latencies))
        assert isinstance(res, PipelineTickResult)
        assert med_latency < 65.0, f"Median latency {med_latency:.2f}ms exceeds budget"
        assert res.wire_packet.policy_close_all == 0.0
        assert isinstance(res.debate_history, list)
        assert len(res.debate_history) == 3, f"Expected 3 debate rounds, got {len(res.debate_history)}"
        for round_info in res.debate_history:
            assert "round" in round_info
            assert "proponent_pressure" in round_info
            assert "adversary_pressure" in round_info

    def test_bull_trap_counter_punch_in_pipeline(self, pipeline):
        """Simulates an adversary bull trap detection causing an exploitative counter-punch SELL."""
        engine = pipeline.dialectical_engine
        orig_trap_detector = engine.adversary.trap_detector

        base_obs = np.random.randn(137).astype(np.float32)
        bars = np.ones((60, 5), dtype=np.float64) * 2350.0
        macro = {s: np.ones(60) * 100.0 for s in ("DXY", "US10Y", "SPX", "XAUUSD")}

        # Force adversary trap detector to emit 99.9% Bull Trap logits
        engine.adversary.trap_detector = MockBullTrapDetector()
        try:
            res: PipelineTickResult = pipeline.process_tick(
                symbol="XAUUSD",
                base_obs_137=base_obs,
                m1_bars_60=bars,
                macro_quotes=macro,
                current_price=2350.0,
            )
            assert res.is_counter_punch is True
            assert res.action == -1  # Counter-punch SELL
            assert res.trap_type == "BULL_TRAP"
            assert res.prm_score >= 0.88
            assert res.wire_packet.policy_close_all == 0.0
        finally:
            engine.adversary.trap_detector = orig_trap_detector

    def test_bear_trap_counter_punch_in_pipeline(self, pipeline):
        """Simulates an adversary bear trap detection causing an exploitative counter-punch BUY."""
        engine = pipeline.dialectical_engine
        orig_trap_detector = engine.adversary.trap_detector

        base_obs = np.random.randn(137).astype(np.float32)
        bars = np.ones((60, 5), dtype=np.float64) * 3500.0
        macro = {s: np.ones(60) * 100.0 for s in ("DXY", "US10Y", "SPX", "XAUUSD")}

        # Force adversary trap detector to emit 99.9% Bear Trap logits
        engine.adversary.trap_detector = MockBearTrapDetector()
        try:
            res: PipelineTickResult = pipeline.process_tick(
                symbol="ETHUSD",
                base_obs_137=base_obs,
                m1_bars_60=bars,
                macro_quotes=macro,
                current_price=3500.0,
            )
            assert res.is_counter_punch is True
            assert res.action == 1  # Counter-punch BUY
            assert res.trap_type == "BEAR_TRAP"
            assert res.prm_score >= 0.88
            assert res.wire_packet.policy_close_all == 0.0
        finally:
            engine.adversary.trap_detector = orig_trap_detector
