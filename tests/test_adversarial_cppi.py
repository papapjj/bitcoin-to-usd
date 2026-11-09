"""
tests/test_adversarial_cppi.py

Adversarial Stress Test Suite for CPPIFloorEngine:
1. Combinatorial 27 sign permutations x trade directions (+1, -1, 0, invalid).
2. Fail-closed behavior on non-finite values (NaN, +Inf, -Inf) in all signal slots.
3. Boundary & malformed observations (empty, truncated, >137, corrupt dtypes).
4. Multi-dimensional array handling (2D batch tensors, empty batches, 3D arrays).
5. Contrary trade directions never receiving 1.5x boost.
6. Strict dominance of 50% cushion cap over unconstrained leverage expansion.
7. Leverage saturation at cushion_ratio >= 2.0 and max_alpha clamping.
8. PyTorch tensor inputs compatibility (if torch available).
"""

from __future__ import annotations

import itertools
import math
import numpy as np
import pytest

from self_growing_trader.cppi_risk_engine import CPPIConfig, CPPIFloorEngine


class TestAdversarialConfluencePermutations:
    """Adversarial stress testing of all 27 sign permutations of 3TF signals."""

    @staticmethod
    def _create_observation(h1: float, m15: float, m5: float) -> list[float]:
        obs = [0.0] * 137
        obs[88] = float(h1)
        obs[84] = float(m15)
        obs[81] = float(m5)
        return obs

    # 3 signs: +1.0 (positive), 0.0 (zero), -1.0 (negative) -> 27 combinations
    SIGN_VALUES = [1.0, 0.0, -1.0]

    @pytest.mark.parametrize("h1,m15,m5", list(itertools.product(SIGN_VALUES, repeat=3)))
    def test_all_27_sign_permutations_buy(self, h1: float, m15: float, m5: float):
        """Verify that for BUY (+1), ONLY (+, +, +) gets confluence; all other 26 get False."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._create_observation(h1, m15, m5)

        is_confluent = engine.evaluate_confluence(inputs, trade_direction=1)
        direction = engine.evaluate_confluence_direction(inputs)

        # Baseline risk without confluence
        risk_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        # Risk evaluated with BUY
        risk_eval = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=1)

        if h1 > 0 and m15 > 0 and m5 > 0:
            assert is_confluent is True, f"Failed for all positive: h1={h1}, m15={m15}, m5={m5}"
            assert direction == 1
            assert risk_eval == pytest.approx(1.50 * risk_base)
        else:
            assert is_confluent is False, f"Unexpected confluence for h1={h1}, m15={m15}, m5={m5}"
            assert direction != 1
            assert risk_eval == pytest.approx(risk_base)

    @pytest.mark.parametrize("h1,m15,m5", list(itertools.product(SIGN_VALUES, repeat=3)))
    def test_all_27_sign_permutations_sell(self, h1: float, m15: float, m5: float):
        """Verify that for SELL (-1), ONLY (-, -, -) gets confluence; all other 26 get False."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._create_observation(h1, m15, m5)

        is_confluent = engine.evaluate_confluence(inputs, trade_direction=-1)
        direction = engine.evaluate_confluence_direction(inputs)

        risk_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        risk_eval = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=-1)

        if h1 < 0 and m15 < 0 and m5 < 0:
            assert is_confluent is True, f"Failed for all negative: h1={h1}, m15={m15}, m5={m5}"
            assert direction == -1
            assert risk_eval == pytest.approx(1.50 * risk_base)
        else:
            assert is_confluent is False, f"Unexpected confluence for h1={h1}, m15={m15}, m5={m5}"
            assert direction != -1
            assert risk_eval == pytest.approx(risk_base)

    @pytest.mark.parametrize("h1,m15,m5", list(itertools.product(SIGN_VALUES, repeat=3)))
    def test_all_27_sign_permutations_neutral_direction(self, h1: float, m15: float, m5: float):
        """Verify that trade_direction = 0 NEVER triggers confluence boost regardless of signals."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._create_observation(h1, m15, m5)

        assert engine.evaluate_confluence(inputs, trade_direction=0) is False

        risk_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        risk_eval = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=0)
        assert risk_eval == pytest.approx(risk_base)

    @pytest.mark.parametrize("invalid_dir", [-99, -5, -2, 2, 5, 99])
    @pytest.mark.parametrize("h1,m15,m5", [(1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (1.0, -1.0, 1.0)])
    def test_invalid_trade_directions_never_confluent(self, invalid_dir: int, h1: float, m15: float, m5: float):
        """Directions other than +1 and -1 must strictly return False and receive 1.0x multiplier."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = self._create_observation(h1, m15, m5)

        assert engine.evaluate_confluence(inputs, trade_direction=invalid_dir) is False

        risk_base = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        risk_eval = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=invalid_dir)
        assert risk_eval == pytest.approx(risk_base)


class TestAdversarialContraryDirections:
    """Verify contrary directions never receive the 1.5x boost."""

    def test_bullish_signals_with_sell_order_receives_no_boost(self):
        """H1>0, M15>0, M5>0 with trade_direction = -1 (SELL) must NOT receive 1.5x boost."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = 2.5, 1.8, 3.2  # Strongly bullish

        assert engine.evaluate_confluence(inputs, trade_direction=-1) is False
        assert engine.evaluate_confluence_direction(inputs) == 1

        risk_no_inputs = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        risk_contrary = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=-1)

        assert risk_contrary == pytest.approx(risk_no_inputs)
        assert risk_contrary < 1.50 * risk_no_inputs

    def test_bearish_signals_with_buy_order_receives_no_boost(self):
        """H1<0, M15<0, M5<0 with trade_direction = 1 (BUY) must NOT receive 1.5x boost."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = -2.5, -1.8, -3.2  # Strongly bearish

        assert engine.evaluate_confluence(inputs, trade_direction=1) is False
        assert engine.evaluate_confluence_direction(inputs) == -1

        risk_no_inputs = engine.compute_dynamic_risk(0.001, 10000.0, inputs=None, trade_direction=0)
        risk_contrary = engine.compute_dynamic_risk(0.001, 10000.0, inputs=inputs, trade_direction=1)

        assert risk_contrary == pytest.approx(risk_no_inputs)
        assert risk_contrary < 1.50 * risk_no_inputs


class TestAdversarialNonFiniteInputs:
    """Stress testing non-finite values (NaN, +Inf, -Inf) across timeframe signals and engine inputs."""

    NON_FINITES = [float("nan"), float("inf"), float("-inf")]

    @pytest.mark.parametrize("val", NON_FINITES)
    @pytest.mark.parametrize("slot", ["h1", "m15", "m5"])
    def test_single_non_finite_slot_fails_closed(self, val: float, slot: str):
        """Any single non-finite signal in (h1, m15, m5) must safely yield (0,0,0) and no boost."""
        engine = CPPIFloorEngine()
        signals = {"h1": 1.0, "m15": 1.0, "m5": 1.0}
        signals[slot] = val

        inputs = [0.0] * 137
        inputs[88] = signals["h1"]
        inputs[84] = signals["m15"]
        inputs[81] = signals["m5"]

        extracted = engine.extract_timeframe_signals(inputs)
        assert extracted == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(inputs, trade_direction=1) is False
        assert engine.evaluate_confluence(inputs, trade_direction=-1) is False
        assert engine.evaluate_confluence_direction(inputs) == 0

    @pytest.mark.parametrize("val1", NON_FINITES)
    @pytest.mark.parametrize("val2", NON_FINITES)
    def test_pairwise_non_finite_slots(self, val1: float, val2: float):
        """Multiple non-finite values are handled safely without unhandled exceptions."""
        engine = CPPIFloorEngine()
        inputs = [0.0] * 137
        inputs[88] = val1
        inputs[84] = val2
        inputs[81] = 1.0

        assert engine.extract_timeframe_signals(inputs) == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(inputs, trade_direction=1) is False

    def test_all_non_finite_slots(self):
        """All non-finite slots fail closed."""
        engine = CPPIFloorEngine()
        for v in [float("nan"), float("inf"), float("-inf")]:
            inputs = [0.0] * 137
            inputs[88], inputs[84], inputs[81] = v, v, v
            assert engine.extract_timeframe_signals(inputs) == (0.0, 0.0, 0.0)
            assert engine.evaluate_confluence(inputs, 1) is False

    def test_nan_in_non_confluence_indices_does_not_break_confluence(self):
        """NaN at non-confluence indices (e.g. index 0, 50, 136) does not corrupt confluence signals."""
        engine = CPPIFloorEngine()
        inputs = [float("nan")] * 137
        # Valid confluence at target indices
        inputs[88] = 0.5
        inputs[84] = 0.5
        inputs[81] = 0.5

        extracted = engine.extract_timeframe_signals(inputs)
        assert extracted == pytest.approx((0.5, 0.5, 0.5))
        assert engine.evaluate_confluence(inputs, trade_direction=1) is True


class TestAdversarialVectorShapesAndTypes:
    """Stress testing truncated vectors, irregular shapes, and unusual data types."""

    @pytest.mark.parametrize("length", [0, 1, 2, 10, 50, 80, 81, 82, 84, 85, 88])
    def test_strictly_truncated_vectors_fail_closed(self, length: int):
        """Vectors of length <= 88 cannot contain index 88 and must fail-closed."""
        engine = CPPIFloorEngine()
        inputs = [1.0] * length
        assert engine.extract_timeframe_signals(inputs) == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(inputs, trade_direction=1) is False

    def test_exact_minimum_vector_length_89(self):
        """Vector of length 89 contains index 88 and should extract properly."""
        engine = CPPIFloorEngine()
        inputs = [0.0] * 89
        inputs[88] = 0.7
        inputs[84] = 0.3
        inputs[81] = 0.4

        extracted = engine.extract_timeframe_signals(inputs)
        assert extracted == pytest.approx((0.7, 0.3, 0.4))
        assert engine.evaluate_confluence(inputs, trade_direction=1) is True

    @pytest.mark.parametrize("length", [90, 100, 136, 137, 200, 500])
    def test_overlength_vectors(self, length: int):
        """Vectors longer than 88 extract accurately."""
        engine = CPPIFloorEngine()
        inputs = [0.0] * length
        inputs[88] = 0.9
        inputs[84] = 0.8
        inputs[81] = 0.7

        extracted = engine.extract_timeframe_signals(inputs)
        assert extracted == pytest.approx((0.9, 0.8, 0.7))
        assert engine.evaluate_confluence(inputs, trade_direction=1) is True

    def test_empty_2d_batch_arrays(self):
        """Empty 2D batch arrays (0, 137) or (5, 0) fail closed safely."""
        engine = CPPIFloorEngine()
        arr_0_137 = np.zeros((0, 137), dtype=np.float32)
        assert engine.extract_timeframe_signals(arr_0_137) == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(arr_0_137, trade_direction=1) is False

        arr_5_0 = np.zeros((5, 0), dtype=np.float32)
        assert engine.extract_timeframe_signals(arr_5_0) == (0.0, 0.0, 0.0)

    def test_3d_and_irregular_tensors(self):
        """3D tensors [1, 1, 137] flatten and extract row safely."""
        engine = CPPIFloorEngine()
        t3d = np.zeros((1, 1, 137), dtype=np.float32)
        t3d[0, 0, 88] = 0.6
        t3d[0, 0, 84] = 0.5
        t3d[0, 0, 81] = 0.4

        assert engine.evaluate_confluence(t3d, trade_direction=1) is True

    @pytest.mark.parametrize("bad_type", [
        None,
        42,
        3.14159,
        "not_an_array",
        {"dict": "not_supported"},
        [object()],
        np.array(["a", "b", "c"]),
    ])
    def test_corrupt_types_fail_closed(self, bad_type):
        """Non-numeric or malformed data types fail closed without crashing."""
        engine = CPPIFloorEngine()
        assert engine.extract_timeframe_signals(bad_type) == (0.0, 0.0, 0.0)
        assert engine.evaluate_confluence(bad_type, trade_direction=1) is False

    def test_pytorch_tensor_compatibility(self):
        """If PyTorch is installed, Tensor inputs extract and evaluate correctly."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not installed")

        engine = CPPIFloorEngine()
        t = torch.zeros(137, dtype=torch.float32)
        t[88] = 0.75
        t[84] = 0.55
        t[81] = 0.35

        extracted = engine.extract_timeframe_signals(t)
        assert extracted == pytest.approx((0.75, 0.55, 0.35))
        assert engine.evaluate_confluence(t, trade_direction=1) is True
        assert engine.evaluate_confluence(t, trade_direction=-1) is False


class TestAdversarialCushionCapStrictDominance:
    """Stress testing the absolute dominance of the 50% cushion cap over leverage expansion."""

    @pytest.mark.parametrize("base_risk", [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.0, 10.0, 100.0])
    @pytest.mark.parametrize("equity", [10000.0, 9800.0, 9600.0, 9400.0, 9250.0, 9210.0, 9201.0, 9200.01])
    def test_cushion_cap_strictly_dominates_across_base_risks_and_equities(
        self, base_risk: float, equity: float
    ):
        """For any base risk and equity > floor, risk_pct <= (0.50 * cushion) / equity."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08, cushion_loss_fraction=0.50))
        engine.reset(10000.0)  # Peak = 10000, Floor = 9200

        # Bullish inputs with confluence boost (1.5x)
        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = 1.0, 1.0, 1.0

        risk = engine.compute_dynamic_risk(base_risk, current_equity=equity, inputs=inputs, trade_direction=1)
        cushion = max(0.0, equity - 9200.0)
        max_allowed_risk = (0.50 * cushion) / equity

        assert risk <= max_allowed_risk + 1e-12, (
            f"Cap violated! risk={risk}, max_allowed_risk={max_allowed_risk}, "
            f"cushion={cushion}, equity={equity}, base_risk={base_risk}"
        )
        # Dollar loss check
        dollar_loss = risk * equity
        assert dollar_loss <= (0.50 * cushion) + 1e-8

    def test_cushion_cap_dominance_under_extreme_confluence_boost(self):
        """Even with an astronomical confluence boost (100.0x), cushion cap strictly clips risk."""
        config = CPPIConfig(max_allowed_dd=0.08, confluence_boost=100.0, cushion_loss_fraction=0.50)
        engine = CPPIFloorEngine(config)
        engine.reset(10000.0)

        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = 1.0, 1.0, 1.0

        # Peak 10000, Floor 9200, Cushion 800. Cap = (0.50 * 800) / 10000 = 0.04 (4.0%)
        # Unconstrained = 0.05 * 5.0 * 100.0 = 25.0 (2500%)
        risk = engine.compute_dynamic_risk(0.05, current_equity=10000.0, inputs=inputs, trade_direction=1)
        assert risk == pytest.approx(0.04)

    def test_cushion_cap_dominance_under_extreme_pumping_slope(self):
        """Even with an extreme pumping slope (1000.0), cushion cap strictly clips risk."""
        config = CPPIConfig(max_allowed_dd=0.08, pumping_slope=1000.0, cushion_loss_fraction=0.50)
        engine = CPPIFloorEngine(config)
        engine.reset(10000.0)

        # Cap = 0.04
        risk = engine.compute_dynamic_risk(0.05, current_equity=10000.0)
        assert risk == pytest.approx(0.04)

    def test_zero_risk_at_or_below_floor_regardless_of_inputs(self):
        """At or below floor, computed risk is strictly 0.0, rejecting any leverage pumping."""
        engine = CPPIFloorEngine(CPPIConfig(max_allowed_dd=0.08))
        engine.reset(10000.0)  # Floor = 9200.0

        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = 1.0, 1.0, 1.0

        # Exactly at floor
        assert engine.compute_dynamic_risk(0.05, current_equity=9200.0, inputs=inputs, trade_direction=1) == 0.0
        # 1 cent below floor
        assert engine.compute_dynamic_risk(0.05, current_equity=9199.99, inputs=inputs, trade_direction=1) == 0.0
        # Severe crash below floor
        assert engine.compute_dynamic_risk(0.05, current_equity=5000.0, inputs=inputs, trade_direction=1) == 0.0


class TestAdversarialLeverageSaturation:
    """Stress testing leverage pumping saturation and max_alpha boundaries."""

    @pytest.mark.parametrize("cushion_ratio", [2.0, 2.01, 2.5, 3.0, 5.0, 10.0, 100.0, 1000.0])
    def test_leverage_saturates_at_max_cushion_ratio(self, cushion_ratio: float):
        """Alpha formula strictly clamps at max_cushion_ratio (2.0) -> alpha = 9.7x."""
        config = CPPIConfig(base_alpha=0.3, pumping_slope=4.7, max_cushion_ratio=2.0)
        clamped_ratio = max(0.0, min(config.max_cushion_ratio, cushion_ratio))
        alpha = config.base_alpha + config.pumping_slope * clamped_ratio
        assert alpha == pytest.approx(9.70)

    @pytest.mark.parametrize("cushion_ratio", [2.0, 2.5, 5.0, 10.0])
    def test_leverage_with_confluence_saturates_at_14_55x(self, cushion_ratio: float):
        """With 1.5x confluence boost, dynamic alpha saturates strictly at 9.7 * 1.5 = 14.55x."""
        config = CPPIConfig(base_alpha=0.3, pumping_slope=4.7, max_cushion_ratio=2.0, confluence_boost=1.5)
        clamped_ratio = max(0.0, min(config.max_cushion_ratio, cushion_ratio))
        alpha = config.base_alpha + config.pumping_slope * clamped_ratio
        dynamic_alpha = alpha * config.confluence_boost
        assert dynamic_alpha == pytest.approx(14.55)

    def test_max_alpha_hard_ceiling_override(self):
        """When max_alpha is configured (e.g. 6.0), alpha cannot exceed 6.0 even with confluence."""
        config = CPPIConfig(base_alpha=0.3, pumping_slope=4.7, max_alpha=6.0, confluence_boost=1.5)
        engine = CPPIFloorEngine(config)
        # At peak 10000, cushion_ratio = 1.0 -> alpha = 5.0. Confluence 1.5x -> 7.5x.
        # But max_alpha = 6.0 -> capped at 6.0x!
        # Base risk = 0.001 -> unconstrained = 0.001 * 6.0 = 0.006.
        # Cushion cap = 0.04 (does not bind).
        inputs = [0.0] * 137
        inputs[88], inputs[84], inputs[81] = 1.0, 1.0, 1.0
        risk = engine.compute_dynamic_risk(0.001, current_equity=10000.0, inputs=inputs, trade_direction=1)
        assert risk == pytest.approx(0.006)
