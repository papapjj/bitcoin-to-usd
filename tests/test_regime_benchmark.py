"""
tests/test_regime_benchmark.py

Comprehensive Automated Test Suite for Milestone 3 Head-to-Head OOS Benchmark.
Verifies:
1. Mathematical correctness of all metrics (BA, Macro F1, Log-loss, Brier, MRTL, FWR).
2. Numerical stability on edge cases (empty inputs, constant labels, extreme flip rates).
3. Latency measurement profiling on CPU.
4. Candidate model instantiation and configuration integrity.
5. End-to-end benchmark execution on sliced / synthetic datasets.
6. Champion model selection, persistence, bit-for-bit reload, and report generation.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np
import pytest

from self_growing_trader.regime_benchmark import (
    compute_balanced_accuracy,
    compute_brier_score,
    compute_false_whipsaw_rate,
    compute_log_loss,
    compute_macro_f1,
    compute_mrtl,
    compute_pareto_score,
    evaluate_model_oos,
    get_candidate_models,
    measure_inference_latency,
    run_head_to_head_benchmark,
)
from self_growing_trader.regime_models import (
    BaseRegimeDetector,
    GaussianHMMRegimeDetector,
    GMMRegimeDetector,
    NUM_REGIMES,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
    SupervisedSoftmaxRegimeDetector,
)


# ==============================================================================
# 1. METRICS COMPUTATION UNIT TESTS
# ==============================================================================

class TestBenchmarkMetricsCalculations:
    """Verifies mathematical correctness of each individual benchmark metric."""

    def test_balanced_accuracy_perfect_and_imperfect(self) -> None:
        """Balanced accuracy should equal 1.0 for perfect predictions and average recall across classes."""
        # 1. Perfect match
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        assert compute_balanced_accuracy(y_true, y_pred) == 1.0

        # 2. Asymmetric recalls:
        # Class 0: 1/2 correct (50%)
        # Class 1: 2/2 correct (100%)
        # Class 2: 1/2 correct (50%)
        # Mean recall = (0.5 + 1.0 + 0.5) / 3 = 2.0 / 3 = 0.66666...
        y_pred_asym = np.array([0, 1, 1, 1, 2, 0])
        ba = compute_balanced_accuracy(y_true, y_pred_asym)
        assert np.isclose(ba, 2.0 / 3.0, atol=1e-5)

        # 3. Empty input
        assert compute_balanced_accuracy(np.array([]), np.array([])) == 0.0

    def test_macro_f1_score(self) -> None:
        """Macro F1 computes the unweighted mean of per-class F1 scores."""
        # Perfect
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        assert np.isclose(compute_macro_f1(y_true, y_pred), 1.0)

        # Class 0: TP=1, FP=1, FN=1 -> F1_0 = 2*1/(2+1+1) = 0.5
        # Class 1: TP=2, FP=0, FN=0 -> F1_1 = 1.0
        # Class 2: TP=1, FP=1, FN=1 -> F1_2 = 0.5
        # Macro F1 = (0.5 + 1.0 + 0.5) / 3 = 2/3
        y_pred_mix = np.array([0, 1, 2, 2, 1, 0])
        macro_f1 = compute_macro_f1(y_true, y_pred_mix)
        assert np.isclose(macro_f1, 2.0 / 3.0, atol=1e-5)

        # Empty
        assert compute_macro_f1(np.array([]), np.array([])) == 0.0

    def test_log_loss_multi_class(self) -> None:
        """Multi-class log loss matches theoretical cross entropy values."""
        # 1. Confident correct predictions: loss close to 0.0
        y_true = np.array([0, 1, 2])
        probs_perfect = np.array([
            [0.999, 0.0005, 0.0005],
            [0.0005, 0.999, 0.0005],
            [0.0005, 0.0005, 0.999],
        ])
        loss_perf = compute_log_loss(y_true, probs_perfect)
        assert loss_perf < 0.01

        # 2. Maximum entropy (uniform probability [1/3, 1/3, 1/3]): loss = ln(3) ~ 1.098612
        probs_uniform = np.full((3, 3), 1.0 / 3.0)
        loss_uniform = compute_log_loss(y_true, probs_uniform)
        assert np.isclose(loss_uniform, np.log(3.0), atol=1e-4)

        # 3. Bad predictions (confident wrong): high loss
        probs_wrong = np.array([
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.98, 0.01, 0.01],
        ])
        loss_wrong = compute_log_loss(y_true, probs_wrong)
        assert loss_wrong > 3.0

        # Empty
        assert compute_log_loss(np.array([]), np.zeros((0, 3))) == 0.0

    def test_brier_score_multi_class(self) -> None:
        """Multi-class Brier score computes mean squared probability errors."""
        # 1. Perfect predictions: Brier = 0.0
        y_true = np.array([0, 1, 2])
        probs_perfect = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        assert np.isclose(compute_brier_score(y_true, probs_perfect), 0.0)

        # 2. Completely incorrect deterministic: sum of squared error per sample = (0-1)^2 + (1-0)^2 = 2.0
        probs_worst = np.array([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ])
        assert np.isclose(compute_brier_score(y_true, probs_worst), 2.0)

        # 3. Uniform probabilities [1/3, 1/3, 1/3]:
        # Error per row = (1/3 - 1)^2 + (1/3 - 0)^2 + (1/3 - 0)^2 = 4/9 + 1/9 + 1/9 = 6/9 = 2/3 ~ 0.66667
        probs_uniform = np.full((3, 3), 1.0 / 3.0)
        assert np.isclose(compute_brier_score(y_true, probs_uniform), 2.0 / 3.0, atol=1e-5)

        # Empty
        assert compute_brier_score(np.array([]), np.zeros((0, 3))) == 0.0

    def test_mrtl_transition_latency_exact(self) -> None:
        """Verifies exact MRTL calculation under verified transition persistence."""
        h_min = 15
        w_max = 60

        # Scenario A: No transitions (constant regime)
        y_const = np.full(100, REGIME_CHOP)
        assert compute_mrtl(y_const, y_const, h_min=h_min, w_max=w_max) == 0.0

        # Scenario B: Transition with immediate detection at t_m = tau (0 bars delay)
        # 50 bars Chop -> 50 bars Bull
        y_true = np.array([REGIME_CHOP] * 50 + [REGIME_BULL] * 50)
        y_pred_immediate = np.array([REGIME_CHOP] * 50 + [REGIME_BULL] * 50)
        assert compute_mrtl(y_true, y_pred_immediate, h_min=h_min, w_max=w_max) == 0.0

        # Scenario C: Transition with exactly 3 bars delay
        # True transition at bar 50; detected at bar 53
        y_pred_3bars = np.array([REGIME_CHOP] * 53 + [REGIME_BULL] * 47)
        assert compute_mrtl(y_true, y_pred_3bars, h_min=h_min, w_max=w_max) == 3.0

        # Scenario D: Missed transition (never detects Bull)
        # Should be penalized at w_max = 60
        y_pred_missed = np.full(100, REGIME_CHOP)
        assert compute_mrtl(y_true, y_pred_missed, h_min=h_min, w_max=w_max) == 60.0

        # Scenario E: Transient flip that fails persistence condition (< h_min=15 bars)
        # 50 Chop -> 5 Bull -> 45 Chop. The 5 Bull bars should NOT be counted as a transition!
        y_transient = np.array([REGIME_CHOP] * 50 + [REGIME_BULL] * 5 + [REGIME_CHOP] * 45)
        # Here only the transition from Bull back to Chop at bar 55 persists for 45 bars (>= 15)
        # If model detects Chop immediately at bar 55, MRTL = 0
        y_pred_transient = np.array([REGIME_CHOP] * 100)
        mrtl_trans = compute_mrtl(y_transient, y_pred_transient, h_min=h_min, w_max=w_max)
        assert mrtl_trans == 0.0

    def test_false_whipsaw_rate_in_chop(self) -> None:
        """Verifies FWR in Chop under various prediction stability scenarios."""
        # 100 true Chop bars
        y_true = np.full(100, REGIME_CHOP)

        # 1. Perfectly stable Chop predictions: FWR = 0.0%
        y_pred_stable = np.full(100, REGIME_CHOP)
        assert compute_false_whipsaw_rate(y_true, y_pred_stable) == 0.0

        # 2. Every consecutive pair has a whipsaw flip (alternating Bull / Bear)
        # Total pairs = 99, Flips = 99 -> FWR = 100.0%
        y_pred_whipsaw = np.array([REGIME_BULL if i % 2 == 0 else REGIME_BEAR for i in range(100)])
        assert compute_false_whipsaw_rate(y_true, y_pred_whipsaw) == 100.0

        # 3. Known flip count: exactly 2 flips in 100 bars (99 pairs)
        # bar 20 flips to Bull (flip 1), bar 21 flips back to Chop (flip 2)
        y_pred_2flips = np.full(100, REGIME_CHOP)
        y_pred_2flips[20] = REGIME_BULL
        # Total pairs = 99. Flip at t=20 (pred[19]=Chop, pred[20]=Bull: flip=1)
        # Flip at t=21 (pred[20]=Bull, pred[21]=Chop: flip=1)
        # Flip at t=22 (pred[21]=Chop, pred[22]=Chop: flip=0)
        fwr = compute_false_whipsaw_rate(y_true, y_pred_2flips)
        expected_fwr = (2.0 / 99.0) * 100.0
        assert np.isclose(fwr, expected_fwr, atol=1e-4)

        # 4. Zero Chop bars
        y_non_chop = np.full(50, REGIME_BULL)
        assert compute_false_whipsaw_rate(y_non_chop, y_non_chop) == 0.0

    def test_pareto_composite_score(self) -> None:
        """Verifies exact Pareto decision function arithmetic."""
        score = compute_pareto_score(
            balanced_acc=0.80,
            macro_f1=0.75,
            fwr_chop=10.0,
            mrtl=3.0,
            latency_us=100.0,
        )
        # Formula: 0.80 + 0.75 - 0.5 * (10 / 100) - 0.2 * (3 / 60) - 0.1 * (100 / 5000)
        # = 1.55 - 0.05 - 0.01 - 0.002 = 1.4880
        assert np.isclose(score, 1.4880, atol=1e-5)


# ==============================================================================
# 2. LATENCY PROFILING & CANDIDATE REGISTRY TESTS
# ==============================================================================

class TestLatencyAndCandidateRegistry:
    """Tests latency measurement engine and candidate model instantiation."""

    def test_get_candidate_models_registry(self) -> None:
        """Ensures all 5 required candidate architectures are properly configured."""
        models = get_candidate_models(random_state=42)
        expected_keys = {
            "GaussianHMM_3State",
            "Pure_GMM",
            "GMM_EWMA_0.20",
            "Supervised_LightGBM_TempScaled",
            "Supervised_PyTorch_MLP_TempScaled",
        }
        assert set(models.keys()) == expected_keys
        for name, m in models.items():
            assert isinstance(m, BaseRegimeDetector), f"{name} must inherit BaseRegimeDetector"

    def test_measure_inference_latency_profiling(self) -> None:
        """Verifies single-step latency profiling executes and measures CPU timings."""
        np.random.seed(42)
        X_sample = np.random.randn(200, 15)
        hmm = GaussianHMMRegimeDetector(n_states=3, n_iter=5, random_state=42)
        hmm.fit(X_sample)

        metrics = measure_inference_latency(hmm, X_sample, n_warmup=20, n_iter=100)
        required_keys = ["p50_us", "p99_us", "mean_us", "p50_ms", "p99_ms", "mean_ms"]
        for k in required_keys:
            assert k in metrics
            assert metrics[k] >= 0.0

        # HMM on CPU must execute in < 5.0ms (5000 us)
        assert metrics["p50_us"] < 5000.0, f"HMM latency too high: {metrics['p50_us']} us"
        assert metrics["p99_us"] < 5000.0, f"HMM P99 latency too high: {metrics['p99_us']} us"


# ==============================================================================
# 3. END-TO-END BENCHMARK EXECUTION ON SYNTHETIC / SLICED SUBSETS
# ==============================================================================

class TestBenchmarkExecutionEndToEnd:
    """Verifies end-to-end benchmark execution, champion selection, and artifact serialization."""

    @pytest.fixture
    def synthetic_dataset(self) -> dict[str, Any]:
        """Generates synthetic dataset mimicking real BTCUSD M1 features and labels."""
        np.random.seed(123)
        n_train = 600
        n_oos = 400
        d = 15

        # Create 3 separable clusters for regimes
        def make_data(n: int) -> tuple[np.ndarray, np.ndarray]:
            n_per_regime = n // 3
            # Bull: positive momentum
            X_bull = np.random.normal(loc=1.5, scale=0.8, size=(n_per_regime, d))
            y_bull = np.full(n_per_regime, REGIME_BULL, dtype=np.int32)
            # Bear: negative momentum
            X_bear = np.random.normal(loc=-1.5, scale=0.8, size=(n_per_regime, d))
            y_bear = np.full(n_per_regime, REGIME_BEAR, dtype=np.int32)
            # Chop: zero momentum, lower volatility
            rem = n - 2 * n_per_regime
            X_chop = np.random.normal(loc=0.0, scale=0.5, size=(rem, d))
            y_chop = np.full(rem, REGIME_CHOP, dtype=np.int32)

            X = np.vstack([X_bull, X_bear, X_chop])
            y = np.concatenate([y_bull, y_bear, y_chop])
            return X, y

        X_tr, y_tr = make_data(n_train)
        X_oos, y_oos = make_data(n_oos)

        return {
            "X_train": X_tr,
            "y_train": y_tr,
            "X_oos": X_oos,
            "y_oos": y_oos,
            "feature_names": [f"feat_{i}" for i in range(d)],
        }

    def test_evaluate_single_model_oos(self, synthetic_dataset: dict[str, Any]) -> None:
        """Tests evaluate_model_oos on a single candidate model."""
        data = synthetic_dataset
        model = GaussianHMMRegimeDetector(n_states=3, n_iter=10, random_state=42)

        n_tr = len(data["X_train"])
        split = int(n_tr * 0.8)
        X_fit, y_fit = data["X_train"][:split], data["y_train"][:split]
        X_val, y_val = data["X_train"][split:], data["y_train"][split:]

        res = evaluate_model_oos(
            name="Test_HMM",
            model=model,
            X_train=X_fit,
            y_train=y_fit,
            X_val=X_val,
            y_val=y_val,
            X_oos=data["X_oos"],
            y_oos=data["y_oos"],
            latency_sample_size=100,
        )

        assert res.model_name == "Test_HMM"
        assert res.balanced_accuracy >= 0.0 and res.balanced_accuracy <= 1.0
        assert res.macro_f1 >= 0.0 and res.macro_f1 <= 1.0
        assert res.log_loss >= 0.0
        assert res.brier_score >= 0.0 and res.brier_score <= 2.0
        assert res.mrtl_bars >= 0.0
        assert res.false_whipsaw_rate >= 0.0 and res.false_whipsaw_rate <= 100.0
        assert res.latency_p50_ms < 5.0

    def test_run_head_to_head_benchmark_full_artifacts(self, synthetic_dataset: dict[str, Any]) -> None:
        """Tests full run_head_to_head_benchmark generating reports, json telemetry, and champion artifact."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir) / "regime_benchmark"
            champ_dir = Path(tmp_dir) / "regime_models"

            # Use lightweight test models for rapid execution
            test_models = {
                "Fast_HMM": GaussianHMMRegimeDetector(n_states=3, n_iter=5, random_state=42),
                "Fast_GMM": GMMRegimeDetector(n_components=3, ewma_alpha=0.20, random_state=42),
                "Fast_LightGBM": SupervisedSoftmaxRegimeDetector(
                    model_type="lightgbm", calibration="temperature", random_state=42
                ),
            }

            champ_name, summary = run_head_to_head_benchmark(
                output_dir=out_dir,
                champion_dir=champ_dir,
                models=test_models,
                dataset=synthetic_dataset,
                verbose=False,
            )

            assert champ_name in test_models
            assert "champion" in summary
            assert "models" in summary
            assert "rankings" in summary

            # Verify files were created
            results_json = out_dir / "benchmark_results.json"
            report_md = out_dir / "BENCHMARK_REPORT.md"
            champ_pkl = champ_dir / "champion_regime_detector.pkl"
            champ_meta = champ_dir / "champion_metadata.json"

            assert results_json.exists(), "benchmark_results.json missing"
            assert report_md.exists(), "BENCHMARK_REPORT.md missing"
            assert champ_pkl.exists(), "champion_regime_detector.pkl missing"
            assert champ_meta.exists(), "champion_metadata.json missing"

            # Verify benchmark_results.json parses correctly
            with open(results_json, "r", encoding="utf-8") as f:
                loaded_json = json.load(f)
            assert loaded_json["champion"]["champion_model_name"] == champ_name
            assert len(loaded_json["rankings"]) == len(test_models)

            # Verify BENCHMARK_REPORT.md contains expected sections
            report_content = report_md.read_text(encoding="utf-8")
            assert "# Head-to-Head Out-of-Sample Benchmark Report" in report_content
            assert "## 1. Executive Summary" in report_content
            assert "## 2. Head-to-Head Comparative Leaderboard" in report_content
            assert champ_name in report_content

            # Verify champion model loads and produces identical predictions
            reloaded_champ = BaseRegimeDetector.load(champ_pkl)
            assert isinstance(reloaded_champ, BaseRegimeDetector)

            sample_x = synthetic_dataset["X_oos"][:10]
            orig_probs = test_models[champ_name].predict_proba(sample_x)
            reloaded_probs = reloaded_champ.predict_proba(sample_x)
            np.testing.assert_allclose(orig_probs, reloaded_probs, atol=1e-6)
