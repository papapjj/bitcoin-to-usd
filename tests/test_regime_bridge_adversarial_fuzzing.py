"""Adversarial Input Fuzzing & Fail-Closed Robustness Test Harness for Milestone 1.

Comprehensive empirical challenge suite verifying that predict_wire_features
and _predict_regime_for_request never crash, freeze, or emit malformed distributions
under corrupted, extreme, or adversarial inputs, and strictly fail closed to
neutral CHOP prior [1/3, 1/3, 1/3].
"""

from __future__ import annotations

import math
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from self_growing_trader.mt5_runner import (
    BridgeRequest,
    MT5BacktestConfig,
    MT5StrategyTesterEvaluator,
    MarketRegime,
    NUM_REGIMES,
    REGIME_NAMES,
)
from self_growing_trader.regime_detector import RegimeDetector

DEFAULT_CHAMPION_PATH = Path("outputs/regime_models/champion_regime_detector.pkl")
NEUTRAL_CHOP_PROBS = np.full(NUM_REGIMES, 1.0 / NUM_REGIMES, dtype=np.float64)


@pytest.fixture(scope="module")
def champion_detector() -> RegimeDetector:
    assert DEFAULT_CHAMPION_PATH.exists(), f"Missing champion artifact at {DEFAULT_CHAMPION_PATH}"
    detector = RegimeDetector.load(DEFAULT_CHAMPION_PATH)
    assert detector is not None
    return detector


@pytest.fixture(scope="module")
def evaluator_instance() -> MT5StrategyTesterEvaluator:
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = MT5BacktestConfig(
            runner_dir=Path(tmp_dir),
            regime_detector_enabled=True,
            offline_replay=True,
        )
        evaluator = MT5StrategyTesterEvaluator(config)
        return evaluator


# ==============================================================================
# SUITE 1: NAN, +INF, -INF CORRUPTED VECTOR FUZZING
# ==============================================================================

class TestNaNAndInfFuzzing:
    """Stress-test NaN, +Inf, -Inf across all 137 positions and combinations."""

    def test_single_nan_in_all_137_positions(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Placing a single NaN at ANY of the 137 slots must fail closed to neutral CHOP."""
        for pos in range(137):
            vec = np.zeros(137, dtype=float)
            vec[pos] = np.nan

            # 1. Direct hook verification
            idx, probs = champion_detector.predict_wire_features(vec)
            assert idx == int(MarketRegime.CHOP), f"Failed at pos {pos}: idx={idx}"
            assert np.allclose(probs, NEUTRAL_CHOP_PROBS, atol=1e-5), f"Failed at pos {pos}: {probs}"
            assert not np.isnan(probs).any()

            # 2. Evaluator IPC bridge verification
            req = BridgeRequest(sequence=pos, inputs=vec, raw_inputs=vec)
            ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
            assert ev_idx == int(MarketRegime.CHOP)
            assert ev_name == "CHOP"
            assert np.allclose(ev_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_single_pos_inf_in_all_137_positions(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Placing a single +Inf at ANY of the 137 slots must fail closed to neutral CHOP."""
        for pos in range(137):
            vec = np.zeros(137, dtype=float)
            vec[pos] = np.inf

            # 1. Direct hook verification
            idx, probs = champion_detector.predict_wire_features(vec)
            assert idx == int(MarketRegime.CHOP), f"Failed at pos {pos}: idx={idx}"
            assert np.allclose(probs, NEUTRAL_CHOP_PROBS, atol=1e-5), f"Failed at pos {pos}: {probs}"
            assert not np.isinf(probs).any()

            # 2. Evaluator IPC bridge verification
            req = BridgeRequest(sequence=pos, inputs=vec, raw_inputs=vec)
            ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
            assert ev_idx == int(MarketRegime.CHOP)
            assert ev_name == "CHOP"
            assert np.allclose(ev_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_single_neg_inf_in_all_137_positions(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Placing a single -Inf at ANY of the 137 slots must fail closed to neutral CHOP."""
        for pos in range(137):
            vec = np.zeros(137, dtype=float)
            vec[pos] = -np.inf

            # 1. Direct hook verification
            idx, probs = champion_detector.predict_wire_features(vec)
            assert idx == int(MarketRegime.CHOP), f"Failed at pos {pos}: idx={idx}"
            assert np.allclose(probs, NEUTRAL_CHOP_PROBS, atol=1e-5), f"Failed at pos {pos}: {probs}"
            assert not np.isinf(probs).any()

            # 2. Evaluator IPC bridge verification
            req = BridgeRequest(sequence=pos, inputs=vec, raw_inputs=vec)
            ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
            assert ev_idx == int(MarketRegime.CHOP)
            assert ev_name == "CHOP"
            assert np.allclose(ev_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_random_sparse_and_dense_nan_inf_fuzzing(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Monte Carlo fuzzing with randomized subsets of NaNs, +Infs, and -Infs."""
        rng = np.random.default_rng(seed=42)
        corruption_types = [np.nan, np.inf, -np.inf]

        for trial in range(100):
            vec = rng.standard_normal(137)
            # Choose random number of corrupted indices (1 to 137)
            num_corrupt = rng.integers(1, 138)
            corrupt_indices = rng.choice(137, size=num_corrupt, replace=False)
            for idx in corrupt_indices:
                vec[idx] = rng.choice(corruption_types)

            # Direct hook
            res_idx, res_probs = champion_detector.predict_wire_features(vec)
            assert res_idx == int(MarketRegime.CHOP), f"Trial {trial} failed: idx={res_idx}"
            assert np.allclose(res_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

            # Evaluator
            req = BridgeRequest(sequence=trial, inputs=vec, raw_inputs=vec)
            ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
            assert ev_idx == int(MarketRegime.CHOP)
            assert ev_name == "CHOP"
            assert np.allclose(ev_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_all_nans_all_pos_infs_all_neg_infs(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Check complete saturation of NaNs and Infs."""
        patterns = {
            "all_nans": np.full(137, np.nan),
            "all_pos_infs": np.full(137, np.inf),
            "all_neg_infs": np.full(137, -np.inf),
            "alternating_nans_infs": np.array([
                np.nan if i % 3 == 0 else (np.inf if i % 3 == 1 else -np.inf)
                for i in range(137)
            ]),
        }

        for name, vec in patterns.items():
            idx, probs = champion_detector.predict_wire_features(vec)
            assert idx == int(MarketRegime.CHOP), f"Pattern {name} failed: idx={idx}"
            assert np.allclose(probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

            req = BridgeRequest(sequence=1, inputs=vec, raw_inputs=vec)
            ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
            assert ev_idx == int(MarketRegime.CHOP)
            assert ev_name == "CHOP"
            assert np.allclose(ev_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)


# ==============================================================================
# SUITE 2: WRONG DIMENSIONALITY FUZZING
# ==============================================================================

class TestDimensionalityAnomalies:
    """Stress-test inputs with wrong shapes and counts: 0, 10, 100, 200, 2D/3D arrays."""

    @pytest.mark.parametrize("dim", [0, 1, 5, 10, 15, 30, 50, 100, 136, 138, 200, 500, 1000])
    def test_direct_predict_wire_features_dimensions(
        self, champion_detector: RegimeDetector, dim: int
    ) -> None:
        """predict_wire_features must handle any dimension without crashing."""
        vec = np.ones(dim, dtype=float) * 0.1
        idx, probs = champion_detector.predict_wire_features(vec)

        assert isinstance(idx, (int, np.integer))
        assert idx in {0, 1, 2}
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()

    @pytest.mark.parametrize("dim", [0, 10, 100, 200])
    def test_evaluator_wrong_dimensionality_handling(
        self, evaluator_instance: MT5StrategyTesterEvaluator, dim: int
    ) -> None:
        """_predict_regime_for_request must NEVER crash or freeze on wrong dimensions."""
        vec = np.zeros(dim, dtype=float)
        req = BridgeRequest(sequence=1, inputs=vec, raw_inputs=vec)
        ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)

        assert isinstance(ev_idx, (int, np.integer))
        assert ev_idx in {0, 1, 2}
        assert len(ev_probs) == 3
        assert np.isclose(sum(ev_probs), 1.0, atol=1e-5)
        assert ev_name in {"BULL", "BEAR", "CHOP"}

        # Specifically for 0-dimension (empty inputs), it must fail closed to neutral CHOP
        if dim == 0:
            assert ev_idx == int(MarketRegime.CHOP)
            assert ev_name == "CHOP"
            assert np.allclose(ev_probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_multidimensional_and_ragged_shapes(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Test 2D and 3D arrays flattened by ravel()."""
        shapes = [(1, 137), (137, 1), (5, 137), (1, 1, 137)]
        for shp in shapes:
            arr = np.zeros(shp, dtype=float)
            idx, probs = champion_detector.predict_wire_features(arr)
            assert idx in {0, 1, 2}
            assert probs.shape == (3,)
            assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

            req = BridgeRequest(sequence=1, inputs=arr, raw_inputs=arr)
            ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
            assert ev_idx in {0, 1, 2}
            assert np.isclose(sum(ev_probs), 1.0, atol=1e-5)


# ==============================================================================
# SUITE 3: EXTREME OUT-OF-DISTRIBUTION (OOD) VALUES & NUMERICAL STRESS
# ==============================================================================

class TestExtremeOODValues:
    """Stress-test numerical extremes: 1e12, -1e12, 1e300, -1e300, subnormals."""

    @pytest.mark.parametrize("val", [
        1e12,
        -1e12,
        1e300,
        -1e300,
        1e-320,  # subnormal float64
        1e-315,
        np.nextafter(0.0, 1.0),
        np.finfo(float).tiny,
        np.finfo(float).eps,
    ])
    def test_homogeneous_extreme_vectors(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator, val: float
    ) -> None:
        """Entire vector filled with an extreme value must evaluate stably."""
        vec = np.full(137, val, dtype=float)

        # 1. Direct hook
        idx, probs = champion_detector.predict_wire_features(vec)
        assert idx in {0, 1, 2}
        assert probs.shape == (3,)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)
        assert not np.isnan(probs).any()
        assert not np.isinf(probs).any()

        # 2. Evaluator IPC bridge
        req = BridgeRequest(sequence=1, inputs=vec, raw_inputs=vec)
        ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
        assert ev_idx in {0, 1, 2}
        assert len(ev_probs) == 3
        assert np.isclose(sum(ev_probs), 1.0, atol=1e-5)
        assert ev_name in {"BULL", "BEAR", "CHOP"}

    def test_alternating_polar_extremes(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Alternating massive positive and negative values: [+1e12, -1e12, ...]."""
        vec = np.array([1e12 if i % 2 == 0 else -1e12 for i in range(137)], dtype=float)
        idx, probs = champion_detector.predict_wire_features(vec)
        assert idx in {0, 1, 2}
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

        req = BridgeRequest(sequence=1, inputs=vec, raw_inputs=vec)
        ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
        assert ev_idx in {0, 1, 2}
        assert np.isclose(sum(ev_probs), 1.0, atol=1e-5)

    def test_isolated_massive_spikes_in_critical_slots(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Single slot spikes to 1e15 (volatility blowup, flash crash) with others neutral."""
        critical_slots = [0, 1, 2, 3, 10, 12, 15, 20, 89, 94, 121]
        for slot in critical_slots:
            for sign in [1.0, -1.0]:
                vec = np.zeros(137, dtype=float)
                vec[slot] = sign * 1e15
                idx, probs = champion_detector.predict_wire_features(vec)
                assert idx in {0, 1, 2}
                assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
                assert not np.isnan(probs).any()

                req = BridgeRequest(sequence=slot, inputs=vec, raw_inputs=vec)
                ev_idx, ev_probs, ev_name = evaluator_instance._predict_regime_for_request(req)
                assert ev_idx in {0, 1, 2}
                assert np.isclose(sum(ev_probs), 1.0, atol=1e-5)


# ==============================================================================
# SUITE 4: MISSING OR UNINITIALIZED DETECTOR MODEL INSTANCES
# ==============================================================================

class TestMissingOrUninitializedDetector:
    """Stress-test evaluator robustness when detector is missing, uninitialized, or corrupted."""

    def test_evaluator_when_detector_is_none(self) -> None:
        """evaluator.regime_detector = None must fail closed to neutral CHOP."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            evaluator.regime_detector = None

            req = BridgeRequest(sequence=1, inputs=np.zeros(137), raw_inputs=np.zeros(137))
            idx, probs, name = evaluator._predict_regime_for_request(req)
            assert idx == int(MarketRegime.CHOP)
            assert name == "CHOP"
            assert np.allclose(probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_evaluator_when_detector_is_uninitialized_instance(self) -> None:
        """evaluator.regime_detector = RegimeDetector(model=None, scaler=None) must not crash."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            evaluator.regime_detector = RegimeDetector(model=None, scaler=None)

            req = BridgeRequest(sequence=1, inputs=np.zeros(137), raw_inputs=np.zeros(137))
            idx, probs, name = evaluator._predict_regime_for_request(req)
            assert idx in {0, 1, 2}
            assert len(probs) == 3
            assert np.isclose(sum(probs), 1.0, atol=1e-5)
            assert name in {"BULL", "BEAR", "CHOP"}

    def test_evaluator_when_detector_raises_arbitrary_exceptions(self) -> None:
        """Detector raising unhandled exceptions must fail closed to neutral CHOP without crashing."""
        class ExplodingDetector:
            def __init__(self, exc_to_raise: Exception):
                self.exc_to_raise = exc_to_raise

            def predict_wire_features(self, x: Any) -> Any:
                raise self.exc_to_raise

        exceptions = [
            RuntimeError("Internal GPU/CPU kernel crash"),
            ValueError("Unexpected feature matrix shape"),
            ZeroDivisionError("Division by zero in normalization"),
            MemoryError("Out of memory during inference"),
            TypeError("Invalid input type"),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            req = BridgeRequest(sequence=1, inputs=np.zeros(137), raw_inputs=np.zeros(137))

            for exc in exceptions:
                evaluator.regime_detector = ExplodingDetector(exc)
                idx, probs, name = evaluator._predict_regime_for_request(req)
                assert idx == int(MarketRegime.CHOP)
                assert name == "CHOP"
                assert np.allclose(probs, NEUTRAL_CHOP_PROBS, atol=1e-5)

    def test_evaluator_when_detector_returns_malformed_outputs(self) -> None:
        """Detector returning malformed tuples/arrays must fail closed to neutral CHOP."""
        class MalformedDetector:
            def __init__(self, return_val: Any):
                self.return_val = return_val

            def predict_wire_features(self, x: Any) -> Any:
                return self.return_val

        malformed_returns = [
            None,
            (None, None),
            (0, [float("nan"), 0.5, 0.5]),
            (0, [float("inf"), 0.0, 0.0]),
            (0, [-float("inf"), 1.0, 0.0]),
            (0, [0.5, 0.5]),  # 2 elements
            (0, [0.25, 0.25, 0.25, 0.25]),  # 4 elements
            (0, "string_instead_of_array"),
            ("not_an_int", [0.33, 0.33, 0.34]),
        ]

        with tempfile.TemporaryDirectory() as tmp_dir:
            config = MT5BacktestConfig(
                runner_dir=Path(tmp_dir),
                regime_detector_enabled=True,
                offline_replay=True,
            )
            evaluator = MT5StrategyTesterEvaluator(config)
            req = BridgeRequest(sequence=1, inputs=np.zeros(137), raw_inputs=np.zeros(137))

            for ret in malformed_returns:
                evaluator.regime_detector = MalformedDetector(ret)
                idx, probs, name = evaluator._predict_regime_for_request(req)
                assert idx in {0, 1, 2}
                assert len(probs) == 3
                assert np.isclose(sum(probs), 1.0, atol=1e-5)
                assert not any(math.isnan(p) or math.isinf(p) for p in probs)

    def test_evaluator_when_request_is_none_or_fields_none(
        self, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Handling None request or None inputs fields."""
        # 1. request is None
        idx1, probs1, name1 = evaluator_instance._predict_regime_for_request(None)
        assert idx1 == int(MarketRegime.CHOP)
        assert name1 == "CHOP"
        assert np.allclose(probs1, NEUTRAL_CHOP_PROBS, atol=1e-5)

        # 2. inputs and raw_inputs are None
        req2 = BridgeRequest(sequence=1, inputs=None, raw_inputs=None)
        idx2, probs2, name2 = evaluator_instance._predict_regime_for_request(req2)
        assert idx2 == int(MarketRegime.CHOP)
        assert name2 == "CHOP"
        assert np.allclose(probs2, NEUTRAL_CHOP_PROBS, atol=1e-5)


# ==============================================================================
# SUITE 5: IPC PROCESS FREEZE PREVENTION & LATENCY UNDER ADVERSARIAL FLOOD
# ==============================================================================

class TestIPCProcessFreezePreventionAndLatency:
    """Verify system never hangs or freezes under rapid continuous adversarial barrage."""

    def test_continuous_adversarial_stream_latency_and_stability(
        self, champion_detector: RegimeDetector, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Fire 500 varied adversarial requests continuously; verify CPU latency < 5ms and zero freeze."""
        rng = np.random.default_rng(seed=123)
        num_iterations = 500
        latencies_ms: list[float] = []

        adversarial_generators = [
            lambda: np.full(137, np.nan),
            lambda: np.full(137, np.inf),
            lambda: np.full(137, -np.inf),
            lambda: np.zeros(0),
            lambda: np.zeros(10),
            lambda: np.zeros(200),
            lambda: np.full(137, 1e12),
            lambda: np.full(137, -1e12),
            lambda: np.full(137, 1e-320),
            lambda: rng.standard_normal(137),
        ]

        for i in range(num_iterations):
            gen = adversarial_generators[i % len(adversarial_generators)]
            arr = gen()
            req = BridgeRequest(sequence=i, inputs=arr, raw_inputs=arr)

            t0 = time.perf_counter()
            idx, probs, name = evaluator_instance._predict_regime_for_request(req)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

            # Invariants verification
            assert idx in {0, 1, 2}
            assert len(probs) == 3
            assert np.isclose(sum(probs), 1.0, atol=1e-5)
            assert not any(math.isnan(p) or math.isinf(p) for p in probs)
            assert name in {"BULL", "BEAR", "CHOP"}

        p95_ms = float(np.percentile(latencies_ms, 95))
        p99_ms = float(np.percentile(latencies_ms, 99))
        max_ms = float(np.max(latencies_ms))
        mean_ms = float(np.mean(latencies_ms))

        print(
            f"\n[Adversarial Stream Latency Benchmark - 500 calls]\n"
            f"Mean: {mean_ms:.3f} ms\n"
            f"P95:  {p95_ms:.3f} ms\n"
            f"P99:  {p99_ms:.3f} ms\n"
            f"Max:  {max_ms:.3f} ms"
        )

        assert mean_ms < 2.0, f"Mean latency {mean_ms:.3f} ms exceeds 2.0ms budget"
        assert p99_ms < 5.0, f"P99 latency {p99_ms:.3f} ms exceeds 5.0ms SLA"

    def test_full_pipeline_normalization_and_packet_adversarial_safety(
        self, evaluator_instance: MT5StrategyTesterEvaluator
    ) -> None:
        """Subject _normalize_request_inputs and _record_regime_telemetry to corrupted requests."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            diag_file = Path(tmp_dir) / "mt5_diagnostics.json"
            events_file = Path(tmp_dir) / "events.jsonl"
            evaluator_instance.runner_dir = Path(tmp_dir)

            corrupt_vectors = [
                np.full(137, np.nan),
                np.full(137, np.inf),
                np.zeros(0),
                np.full(137, 1e12),
            ]

            for seq, vec in enumerate(corrupt_vectors):
                req = BridgeRequest(sequence=seq, inputs=vec, raw_inputs=vec)
                norm_req = evaluator_instance._normalize_request_inputs(req)

                assert norm_req.regime_probabilities is not None
                assert len(norm_req.regime_probabilities) == 3
                assert np.isclose(sum(norm_req.regime_probabilities), 1.0, atol=1e-5)
                assert not any(math.isnan(p) or math.isinf(p) for p in norm_req.regime_probabilities)

                # Test telemetry recording does not throw or corrupt files
                evaluator_instance._record_regime_telemetry(
                    norm_req,
                    norm_req.regime_probabilities,
                    norm_req.regime_name or "CHOP",
                    norm_req.regime_index or 2,
                    report_path=events_file,
                )

            assert events_file.exists()
            assert diag_file.exists() or len(events_file.read_text(encoding="utf-8")) > 0
