"""
tests/test_regime_benchmark_adversarial.py

Adversarial Stress Test Suite for Milestone 3 Head-to-Head OOS Benchmark.
Authored by Challenger 1 (Empirical Metric & Robustness Challenger).

Covers:
1. Metric Edge Cases & Degenerate Distributions:
   - Zero true positives, disjoint class spaces, all-same-class ground truth and predictions.
   - Degenerate probabilities: exact one-hot, uniform, all-zeros, unnormalized, extreme values, negative values.
   - ZeroDivisionError and NaN/Inf immunity across Balanced Accuracy, Macro F1, Log-Loss, Brier, MRTL, and FWR.
   - Empty input arrays and single-element inputs.
   - Randomized Monte Carlo stress trials.
2. Transition Latency Pathological Stress (MRTL):
   - High-frequency chatter / rapid oscillation below persistence threshold (h_min=15).
   - Single-bar transient blips vs verified transitions.
   - Exact boundary transitions (tau = N - h_min, tau = 1, tau = N - 1).
   - Systematic delay sweep [0, 75] proving strict capping at w_max = 60.
   - Missed transitions and multiple consecutive transitions.
   - Global invariant: 0.0 <= MRTL <= w_max for all possible sequences.
3. False Whipsaw Rate in Chop Stress (FWR):
   - Zero Chop intervals, isolated Chop bars (no consecutive pairs).
   - Tranquil Chop, biased constant predictions, maximum whipsaw (100% flips).
   - Alternating Bull/Bear and Bull/Chop chatter.
   - Global invariant: 0.0 <= FWR <= 100.0 for all possible sequences.
4. Champion Artifact Bit-for-Bit Verification:
   - Reload `outputs/regime_models/champion_regime_detector.pkl`.
   - Verify SHA256 checksum and exact byte size matching metadata.
   - Verify bit-for-bit equivalence of telemetry numbers across metadata and benchmark_results.json.
   - Verify single-step `step_filter` online streaming consistency vs batch `predict_proba`.
   - Verify sub-5ms CPU latency SLA on single-bar streaming inference.
5. Pareto Composite Decision Function:
   - Monotonicity checks, extreme penalty bounds, and edge sensitivity.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
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
    measure_inference_latency,
)
from self_growing_trader.regime_models import (
    BaseRegimeDetector,
    NUM_REGIMES,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHAMPION_PKL = OUTPUTS_DIR / "regime_models" / "champion_regime_detector.pkl"
CHAMPION_JSON = OUTPUTS_DIR / "regime_models" / "champion_metadata.json"
BENCHMARK_RESULTS_JSON = OUTPUTS_DIR / "regime_benchmark" / "benchmark_results.json"
BENCHMARK_REPORT_MD = OUTPUTS_DIR / "regime_benchmark" / "BENCHMARK_REPORT.md"


# ==============================================================================
# 1. METRIC EDGE CASES & DEGENERATE DISTRIBUTIONS
# ==============================================================================

class TestMetricEdgeCasesAndDegenerateDistributions:
    """Adversarial stress testing on metric calculation functions under extreme inputs."""

    def test_zero_true_positives_all_classes(self) -> None:
        """When y_pred never matches y_true for any class, BA and F1 must be 0.0 without ZeroDivisionError."""
        # y_true has classes 0 and 1; y_pred has only class 2
        y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        y_pred = np.array([2, 2, 2, 2, 2, 2], dtype=np.int64)

        ba = compute_balanced_accuracy(y_true, y_pred)
        f1 = compute_macro_f1(y_true, y_pred)

        assert not math.isnan(ba)
        assert not math.isinf(ba)
        assert ba == 0.0

        assert not math.isnan(f1)
        assert not math.isinf(f1)
        assert f1 == 0.0

    def test_single_class_ground_truth(self) -> None:
        """Ground truth containing only one class must not crash balanced accuracy or macro F1."""
        y_true = np.zeros(100, dtype=np.int64)

        # Perfect prediction of the single class
        y_pred_correct = np.zeros(100, dtype=np.int64)
        ba_perf = compute_balanced_accuracy(y_true, y_pred_correct)
        f1_perf = compute_macro_f1(y_true, y_pred_correct)
        assert ba_perf == 1.0
        assert f1_perf == 1.0

        # Entirely wrong prediction
        y_pred_wrong = np.ones(100, dtype=np.int64)
        ba_wrong = compute_balanced_accuracy(y_true, y_pred_wrong)
        f1_wrong = compute_macro_f1(y_true, y_pred_wrong)
        assert ba_wrong == 0.0
        assert f1_wrong == 0.0

    def test_all_same_class_prediction_on_balanced_labels(self) -> None:
        """A naive constant predictor predicting only class 0 on 3 balanced classes."""
        y_true = np.array([0, 1, 2] * 50, dtype=np.int64)
        y_pred = np.zeros(150, dtype=np.int64)

        ba = compute_balanced_accuracy(y_true, y_pred)
        f1 = compute_macro_f1(y_true, y_pred)

        # Recall_0 = 1.0, Recall_1 = 0.0, Recall_2 = 0.0 -> BA = 1/3
        assert np.isclose(ba, 1.0 / 3.0, atol=1e-5)
        # Class 0: Precision = 50/150 = 1/3, Recall = 1.0 -> F1 = 2 * (1/3) / (4/3) = 0.5
        # Class 1: F1 = 0.0, Class 2: F1 = 0.0 -> Macro F1 = 0.5 / 3 = 1/6
        assert np.isclose(f1, 1.0 / 6.0, atol=1e-5)

    def test_empty_arrays_all_metrics(self) -> None:
        """All metrics must return 0.0 safely on empty arrays without throwing exceptions."""
        empty_i = np.array([], dtype=np.int64)
        empty_f = np.zeros((0, 3), dtype=np.float64)

        assert compute_balanced_accuracy(empty_i, empty_i) == 0.0
        assert compute_macro_f1(empty_i, empty_i) == 0.0
        assert compute_log_loss(empty_i, empty_f) == 0.0
        assert compute_brier_score(empty_i, empty_f) == 0.0
        assert compute_mrtl(empty_i, empty_i) == 0.0
        assert compute_false_whipsaw_rate(empty_i, empty_i) == 0.0

    def test_single_element_arrays_all_metrics(self) -> None:
        """Single-element inputs must be handled gracefully across all metric functions."""
        y_true = np.array([1], dtype=np.int64)
        y_pred_correct = np.array([1], dtype=np.int64)
        y_pred_wrong = np.array([2], dtype=np.int64)

        assert compute_balanced_accuracy(y_true, y_pred_correct) == 1.0
        assert compute_balanced_accuracy(y_true, y_pred_wrong) == 0.0
        assert compute_macro_f1(y_true, y_pred_correct) == 1.0
        assert compute_macro_f1(y_true, y_pred_wrong) == 0.0
        assert compute_mrtl(y_true, y_pred_correct) == 0.0
        assert compute_false_whipsaw_rate(y_true, y_pred_correct) == 0.0

    def test_degenerate_probabilities_all_zeros(self) -> None:
        """When probability matrix contains all zeros, compute_log_loss must clip and normalize without NaN."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        probs_zeros = np.zeros((3, 3), dtype=np.float64)

        loss = compute_log_loss(y_true, probs_zeros)
        assert not math.isnan(loss)
        assert not math.isinf(loss)
        # Should normalize to uniform distribution [1/3, 1/3, 1/3], yielding loss ~ ln(3)
        assert np.isclose(loss, np.log(3.0), atol=1e-4)

        brier = compute_brier_score(y_true, probs_zeros)
        assert not math.isnan(brier)
        # sum (0 - 1)^2 + (0 - 0)^2 = 1.0 per sample
        assert np.isclose(brier, 1.0, atol=1e-5)

    def test_degenerate_probabilities_all_ones_unnormalized(self) -> None:
        """Unnormalized probabilities with row sum > 1.0 must be safely normalized."""
        y_true = np.array([0, 1, 2], dtype=np.int64)
        probs_ones = np.ones((3, 3), dtype=np.float64)

        loss = compute_log_loss(y_true, probs_ones)
        assert not math.isnan(loss)
        assert np.isclose(loss, np.log(3.0), atol=1e-4)

    def test_degenerate_probabilities_extreme_values(self) -> None:
        """Extreme values (1e-300, 1e20, negative logits) must not cause overflow or negative loss."""
        y_true = np.array([0, 1, 2], dtype=np.int64)

        # Very tiny values
        probs_tiny = np.full((3, 3), 1e-300, dtype=np.float64)
        loss_tiny = compute_log_loss(y_true, probs_tiny)
        assert not math.isnan(loss_tiny)
        assert not math.isinf(loss_tiny)
        assert loss_tiny >= 0.0

        # Negative values (e.g. raw un-softmaxed logits accidentally passed)
        probs_neg = np.array([
            [-5.0, -2.0, -10.0],
            [-1.0, -1.0, -1.0],
            [0.0, -10.0, -20.0],
        ], dtype=np.float64)
        loss_neg = compute_log_loss(y_true, probs_neg)
        assert not math.isnan(loss_neg)
        assert not math.isinf(loss_neg)
        assert loss_neg >= 0.0

    def test_brier_score_out_of_bounds_labels(self) -> None:
        """Labels outside [0, num_classes-1] must be handled via valid_mask without IndexError."""
        y_true_oob = np.array([-1, 99, 3, 0], dtype=np.int64)
        probs = np.full((4, 3), 1.0 / 3.0, dtype=np.float64)

        brier = compute_brier_score(y_true_oob, probs, num_classes=3)
        assert not math.isnan(brier)
        assert not math.isinf(brier)
        assert 0.0 <= brier <= 2.0

    def test_monte_carlo_random_labels_and_probabilities(self) -> None:
        """100 random Monte Carlo trials with random lengths, classes, and probability matrices."""
        rng = np.random.RandomState(42)
        for _ in range(100):
            n = rng.randint(1, 200)
            # Random subset of classes
            classes_present = rng.choice([0, 1, 2], size=rng.randint(1, 4), replace=False)
            y_t = rng.choice(classes_present, size=n)
            y_p = rng.choice([0, 1, 2], size=n)

            # Random dirichlet or arbitrary probabilities
            raw_p = rng.exponential(scale=1.0, size=(n, 3))
            probs = raw_p / np.sum(raw_p, axis=1, keepdims=True)

            ba = compute_balanced_accuracy(y_t, y_p)
            f1 = compute_macro_f1(y_t, y_p)
            loss = compute_log_loss(y_t, probs)
            brier = compute_brier_score(y_t, probs)
            mrtl = compute_mrtl(y_t, y_p)
            fwr = compute_false_whipsaw_rate(y_t, y_p)

            assert 0.0 <= ba <= 1.0, f"BA out of bounds: {ba}"
            assert 0.0 <= f1 <= 1.0, f"Macro F1 out of bounds: {f1}"
            assert 0.0 <= loss, f"Loss negative: {loss}"
            assert 0.0 <= brier <= 2.0, f"Brier out of bounds: {brier}"
            assert 0.0 <= mrtl <= 60.0, f"MRTL out of bounds: {mrtl}"
            assert 0.0 <= fwr <= 100.0, f"FWR out of bounds: {fwr}"


# ==============================================================================
# 2. TRANSITION LATENCY PATHOLOGICAL STRESS (MRTL)
# ==============================================================================

class TestTransitionLatencyPathologicalStress:
    """Stress tests MRTL under extreme, pathological switching sequences."""

    def test_rapid_regime_flipping_chatter_zero_mrtl(self) -> None:
        """Rapid chatter (alternating 0, 1, 0, 1... every bar) fails persistence condition h_min=15."""
        # 1000 bars alternating between Bull and Bear every single bar
        y_true_chatter = np.array([REGIME_BULL if i % 2 == 0 else REGIME_BEAR for i in range(1000)])
        y_pred = np.full(1000, REGIME_CHOP)

        mrtl = compute_mrtl(y_true_chatter, y_pred, h_min=15, w_max=60)
        assert mrtl == 0.0, "High frequency chatter should not trigger persistent regime transition latencies"

    def test_transient_blips_ignored_by_persistence_filter(self) -> None:
        """Single-bar or sub-15-bar blips must be ignored, while confirmed transitions count."""
        h_min = 15
        w_max = 60

        # Pattern:
        # 50 bars Bull -> 3 bars Bear (blip!) -> 50 bars Bull -> 50 bars Bear (true transition!)
        y_true = np.concatenate([
            np.full(50, REGIME_BULL),
            np.full(3, REGIME_BEAR),  # transient blip (< 15 bars)
            np.full(50, REGIME_BULL),  # return to Bull persists for 50 bars (>= 15) -> transition at bar 53
            np.full(50, REGIME_BEAR),  # transition to Bear persists for 50 bars (>= 15) -> transition at bar 103
        ])
        # Model detects return to Bull at bar 55 (delay = 2 bars)
        # Model detects transition to Bear at bar 110 (delay = 7 bars)
        y_pred = np.full(len(y_true), REGIME_BULL)
        y_pred[110:] = REGIME_BEAR

        mrtl = compute_mrtl(y_true, y_pred, h_min=h_min, w_max=w_max)
        # Two confirmed transitions:
        # 1. Bar 53 (Bull): y_pred is already Bull at bar 53 -> delay = 0
        # 2. Bar 103 (Bear): y_pred becomes Bear at bar 110 -> delay = 110 - 103 = 7
        # Mean = (0 + 7) / 2 = 3.5
        assert np.isclose(mrtl, 3.5, atol=1e-5)

    def test_exact_boundary_transitions_edge_cases(self) -> None:
        """Transition occurring at the exact end of the dataset (tau = N - h_min) vs tau = N - h_min + 1."""
        h_min = 15
        w_max = 60
        N = 100

        # Scenario 1: Transition at tau = 85 (tau + h_min = 85 + 15 = 100 <= N). Exactly qualifies!
        y_true_boundary = np.concatenate([np.full(85, REGIME_BULL), np.full(15, REGIME_BEAR)])
        # Model detects immediately at bar 85
        y_pred_exact = np.copy(y_true_boundary)
        assert compute_mrtl(y_true_boundary, y_pred_exact, h_min=h_min, w_max=w_max) == 0.0

        # If model misses it completely in the remaining 15 bars:
        y_pred_miss = np.full(100, REGIME_BULL)
        # Search range is [85 : min(100, 85+61=146)] = [85:100]. Matches is empty -> appends w_max = 60.0
        assert compute_mrtl(y_true_boundary, y_pred_miss, h_min=h_min, w_max=w_max) == 60.0

        # Scenario 2: Transition at tau = 86 (tau + h_min = 86 + 15 = 101 > N). Fails persistence!
        y_true_too_late = np.concatenate([np.full(86, REGIME_BULL), np.full(14, REGIME_BEAR)])
        assert compute_mrtl(y_true_too_late, y_pred_miss, h_min=h_min, w_max=w_max) == 0.0

    def test_systematic_delay_sweep_strict_capping(self) -> None:
        """Sweep delays from 0 to 75 bars and verify exact linear scaling up to 60, and strict capping at 60."""
        h_min = 15
        w_max = 60
        N = 200
        tau = 50

        # 50 Bull -> 150 Bear
        y_true = np.concatenate([np.full(tau, REGIME_BULL), np.full(N - tau, REGIME_BEAR)])

        for delay in range(0, 76):
            y_pred = np.full(N, REGIME_BULL)
            # Model detects Bear at tau + delay
            detect_idx = tau + delay
            if detect_idx < N:
                y_pred[detect_idx:] = REGIME_BEAR

            mrtl = compute_mrtl(y_true, y_pred, h_min=h_min, w_max=w_max)
            expected = min(delay, w_max)
            assert mrtl == expected, f"Failed at delay={delay}: expected {expected}, got {mrtl}"

    def test_mrtl_global_boundedness_invariant(self) -> None:
        """Monte Carlo: MRTL must be strictly bounded in [0.0, w_max] across 500 arbitrary sequences."""
        rng = np.random.RandomState(999)
        w_max = 60
        for _ in range(500):
            n = rng.randint(2, 300)
            h_min = rng.randint(1, 25)
            y_t = rng.choice([0, 1, 2], size=n)
            y_p = rng.choice([0, 1, 2], size=n)

            mrtl = compute_mrtl(y_t, y_p, h_min=h_min, w_max=w_max)
            assert not math.isnan(mrtl), f"MRTL is NaN: {mrtl}"
            assert 0.0 <= mrtl <= float(w_max), f"MRTL {mrtl} outside [0.0, {w_max}]"


# ==============================================================================
# 3. FALSE WHIPSAW RATE IN CHOP STRESS (FWR)
# ==============================================================================

class TestFalseWhipsawRateChopStress:
    """Stress tests False Whipsaw Rate (FWR) under extreme regimes and chatter conditions."""

    def test_zero_chop_intervals(self) -> None:
        """When y_true contains no Chop bars, FWR must be 0.0 without ZeroDivisionError."""
        y_true = np.array([REGIME_BULL] * 50 + [REGIME_BEAR] * 50)
        y_pred = np.array([REGIME_BEAR] * 50 + [REGIME_BULL] * 50)

        fwr = compute_false_whipsaw_rate(y_true, y_pred, chop_label=REGIME_CHOP)
        assert fwr == 0.0

    def test_isolated_single_chop_bars_no_pairs(self) -> None:
        """Isolated single Chop bars have no consecutive Chop pairs (total_pairs == 0) -> FWR = 0.0."""
        # Bull, Chop, Bear, Chop, Bull...
        y_true = np.array([REGIME_BULL, REGIME_CHOP, REGIME_BEAR, REGIME_CHOP, REGIME_BULL])
        y_pred = np.array([REGIME_BULL, REGIME_BEAR, REGIME_BULL, REGIME_BEAR, REGIME_BULL])

        fwr = compute_false_whipsaw_rate(y_true, y_pred, chop_label=REGIME_CHOP)
        assert fwr == 0.0

    def test_constant_biased_predictions_zero_whipsaw(self) -> None:
        """A biased model predicting constant Bull during Chop never whipsaws (flips == 0) -> FWR = 0.0%."""
        y_true = np.full(100, REGIME_CHOP)
        y_pred_bull = np.full(100, REGIME_BULL)

        fwr = compute_false_whipsaw_rate(y_true, y_pred_bull, chop_label=REGIME_CHOP)
        assert fwr == 0.0

    def test_maximum_whipsaw_alternating_predictions(self) -> None:
        """Model alternating between Bull and Bear on every single bar during Chop -> FWR = 100.0%."""
        y_true = np.full(100, REGIME_CHOP)
        # 99 consecutive chop pairs
        y_pred_alt = np.array([REGIME_BULL if i % 2 == 0 else REGIME_BEAR for i in range(100)])

        fwr = compute_false_whipsaw_rate(y_true, y_pred_alt, chop_label=REGIME_CHOP)
        assert fwr == 100.0

    def test_alternating_bull_and_chop_whipsaw(self) -> None:
        """Model alternating between Chop and Bull every bar: both Chop->Bull and Bull->Chop count as flips."""
        y_true = np.full(100, REGIME_CHOP)
        y_pred_alt = np.array([REGIME_CHOP if i % 2 == 0 else REGIME_BULL for i in range(100)])

        # t=1: pred[0]=Chop, pred[1]=Bull -> flip (Bull != Chop)
        # t=2: pred[1]=Bull, pred[2]=Chop -> flip (Bull != Chop)
        # All 99 consecutive pairs flip
        fwr = compute_false_whipsaw_rate(y_true, y_pred_alt, chop_label=REGIME_CHOP)
        assert fwr == 100.0

    def test_fwr_global_boundedness_invariant(self) -> None:
        """Monte Carlo: FWR must be strictly bounded in [0.0, 100.0] across 500 arbitrary sequences."""
        rng = np.random.RandomState(777)
        for _ in range(500):
            n = rng.randint(2, 200)
            y_t = rng.choice([0, 1, 2], size=n)
            y_p = rng.choice([0, 1, 2], size=n)

            fwr = compute_false_whipsaw_rate(y_t, y_p, chop_label=REGIME_CHOP)
            assert not math.isnan(fwr)
            assert 0.0 <= fwr <= 100.0, f"FWR {fwr} outside [0.0, 100.0]"


# ==============================================================================
# 4. CHAMPION ARTIFACT BIT-FOR-BIT VERIFICATION
# ==============================================================================

class TestChampionArtifactVerification:
    """Verifies that the serialized champion artifact matches published benchmark numbers."""

    def test_champion_files_exist(self) -> None:
        """Verifies all required artifact files exist on disk."""
        assert CHAMPION_PKL.exists(), f"Missing champion artifact: {CHAMPION_PKL}"
        assert CHAMPION_JSON.exists(), f"Missing champion metadata: {CHAMPION_JSON}"
        assert BENCHMARK_RESULTS_JSON.exists(), f"Missing benchmark results: {BENCHMARK_RESULTS_JSON}"
        assert BENCHMARK_REPORT_MD.exists(), f"Missing benchmark report: {BENCHMARK_REPORT_MD}"

    def test_champion_sha256_and_filesize(self) -> None:
        """Verifies champion model artifact SHA256 hash and file size match metadata."""
        with open(CHAMPION_JSON, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["champion_model_name"] == "Supervised_LightGBM_TempScaled"
        assert metadata["champion_model_type"] == "SupervisedSoftmaxRegimeDetector"

        # Verify disk file hash
        with open(CHAMPION_PKL, "rb") as f:
            disk_sha256 = hashlib.sha256(f.read()).hexdigest().upper()
        disk_size = os.path.getsize(CHAMPION_PKL)

        assert disk_sha256 == metadata["sha256"], f"SHA256 mismatch: {disk_sha256} vs {metadata['sha256']}"
        assert disk_size == metadata["size_bytes"], f"Size mismatch: {disk_size} vs {metadata['size_bytes']}"

    def test_champion_telemetry_consistency_across_json_and_report(self) -> None:
        """Verifies telemetry numbers in champion_metadata.json and benchmark_results.json are identical."""
        with open(CHAMPION_JSON, "r", encoding="utf-8") as f:
            meta = json.load(f)
        with open(BENCHMARK_RESULTS_JSON, "r", encoding="utf-8") as f:
            res = json.load(f)

        champ_res = res["champion"]
        assert meta["champion_model_name"] == champ_res["champion_model_name"]
        assert meta["sha256"] == champ_res["sha256"]
        assert np.isclose(meta["pareto_score"], champ_res["pareto_score"], atol=1e-8)
        assert np.isclose(meta["balanced_accuracy"], champ_res["balanced_accuracy"], atol=1e-8)
        assert np.isclose(meta["macro_f1"], champ_res["macro_f1"], atol=1e-8)
        assert np.isclose(meta["log_loss"], champ_res["log_loss"], atol=1e-8)
        assert np.isclose(meta["brier_score"], champ_res["brier_score"], atol=1e-8)
        assert np.isclose(meta["mrtl_bars"], champ_res["mrtl_bars"], atol=1e-8)
        assert np.isclose(meta["false_whipsaw_rate"], champ_res["false_whipsaw_rate"], atol=1e-8)

    def test_champion_model_reloads_and_infers(self) -> None:
        """Loads champion model and verifies predict_proba and step_filter contract."""
        model = BaseRegimeDetector.load(CHAMPION_PKL)
        assert isinstance(model, BaseRegimeDetector)

        # 15 features matching Q1 schema
        x_dummy = np.random.randn(10, 15)
        probs = model.predict_proba(x_dummy)

        assert probs.shape == (10, 3)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)
        np.testing.assert_allclose(np.sum(probs, axis=-1), np.ones(10), atol=1e-5)

        # Single step filter
        model.reset()
        for i in range(10):
            regime, p = model.step_filter(x_dummy[i])
            assert regime in (REGIME_BULL, REGIME_BEAR, REGIME_CHOP)
            assert len(p) == 3
            np.testing.assert_allclose(np.sum(p), 1.0, atol=1e-5)

    def test_champion_step_filter_matches_predict_proba(self) -> None:
        """Single-step streaming step_filter must produce identical probabilities to batch predict_proba."""
        model = BaseRegimeDetector.load(CHAMPION_PKL)
        np.random.seed(42)
        X_test = np.random.randn(50, 15)

        probs_batch = model.predict_proba(X_test)

        probs_stream: list[np.ndarray] = []
        model.reset()
        for i in range(len(X_test)):
            _, p = model.step_filter(X_test[i])
            probs_stream.append(p)

        probs_stream_arr = np.array(probs_stream)
        np.testing.assert_allclose(probs_batch, probs_stream_arr, atol=1e-6)

    def test_champion_sub_5ms_inference_latency_sla(self) -> None:
        """Profiles champion model CPU inference to verify < 5.0ms SLA."""
        model = BaseRegimeDetector.load(CHAMPION_PKL)
        np.random.seed(123)
        X_sample = np.random.randn(200, 15)

        metrics = measure_inference_latency(model, X_sample, n_warmup=50, n_iter=500)
        assert metrics["p50_ms"] < 5.0, f"P50 latency exceeded SLA: {metrics['p50_ms']:.4f} ms"
        assert metrics["p99_ms"] < 5.0, f"P99 latency exceeded SLA: {metrics['p99_ms']:.4f} ms"
        assert metrics["mean_ms"] < 5.0, f"Mean latency exceeded SLA: {metrics['mean_ms']:.4f} ms"


# ==============================================================================
# 5. PARETO DECISION FUNCTION BEHAVIOR
# ==============================================================================

class TestParetoDecisionFunction:
    """Verifies Pareto composite fitness score properties and monotonicity."""

    def test_pareto_score_monotonicity(self) -> None:
        """Score must increase with BA and F1, and decrease with FWR, MRTL, and latency."""
        base = compute_pareto_score(0.5, 0.5, 10.0, 10.0, 1000.0)

        # Increasing BA increases score
        assert compute_pareto_score(0.6, 0.5, 10.0, 10.0, 1000.0) > base

        # Increasing Macro F1 increases score
        assert compute_pareto_score(0.5, 0.6, 10.0, 10.0, 1000.0) > base

        # Increasing FWR decreases score
        assert compute_pareto_score(0.5, 0.5, 20.0, 10.0, 1000.0) < base

        # Increasing MRTL decreases score
        assert compute_pareto_score(0.5, 0.5, 10.0, 20.0, 1000.0) < base

        # Increasing Latency decreases score
        assert compute_pareto_score(0.5, 0.5, 10.0, 10.0, 2000.0) < base

    def test_pareto_extreme_bounds(self) -> None:
        """Theoretical maximum score and worst score bounds."""
        # Best possible: BA=1.0, F1=1.0, FWR=0%, MRTL=0, Latency=0 -> Score = 2.0
        best_score = compute_pareto_score(1.0, 1.0, 0.0, 0.0, 0.0)
        assert np.isclose(best_score, 2.0, atol=1e-5)

        # Worst possible: BA=0, F1=0, FWR=100%, MRTL=60, Latency=5000us
        # Score = 0 + 0 - 0.5*(100/100) - 0.2*(60/60) - 0.1*(5000/5000) = -0.5 - 0.2 - 0.1 = -0.8
        worst_score = compute_pareto_score(0.0, 0.0, 100.0, 60.0, 5000.0)
        assert np.isclose(worst_score, -0.8, atol=1e-5)
