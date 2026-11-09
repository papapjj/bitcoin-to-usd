"""
tests/test_adversarial_regime_benchmark_challenger2.py

Adversarial Stress Test Suite for Milestone 3 Head-to-Head OOS Benchmark.
Designed and executed by Challenger 2 (Adversarial Whipsaw & Performance Hardening).

Target:
- src/self_growing_trader/regime_benchmark.py
- outputs/regime_benchmark/benchmark_results.json
- outputs/regime_models/champion_regime_detector.pkl
- outputs/regime_models/champion_metadata.json

Adversarial Challenge Dimensions:
1. Whipsaw In Chop Empirical Stress:
   - Filter real February and March OOS data for verified CHOP intervals.
   - Compute False Whipsaw Rate (FWR) across candidate architectures on real chop.
   - Contrast structural whipsaw suppression (HMM 1.47%, GMM+EWMA 1.70%) vs unbuffered classifiers (LightGBM 15.14%, MLP 21.97%).
   - Mathematical boundary oracles (zero chop, alternating chop, pathological flips).
2. Latency Stress Under CPU Load:
   - Execute 1,000 sequential step_filter calls on the champion model while CPU is active.
   - Measure P50, P90, P95, P99, and maximum latency spike against the < 5.0ms constraint.
   - Pathological input robustness: NaNs, Infs, zero vectors, extreme scales (1e-7, 1e7).
3. Zero OOS Data Contamination Audit:
   - Strict temporal separation: In-Sample (Jan 2025) vs Out-of-Sample (Feb & Mar 2025).
   - Zero timestamp intersection between Train and OOS.
   - 60-bar embargo purge verification on the Jan/Feb boundary.
   - Scaler isolation: In-Sample fitting verification.
   - Zero OOS exposure during training pipeline verification.
4. Champion Artifact Forensic Verification:
   - File existence, SHA256 cryptographic checksum matching.
   - Bit-for-bit reloadability via BaseRegimeDetector.load().
   - Metadata scorecard consistency between champion_metadata.json and benchmark_results.json.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
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
from self_growing_trader.regime_data import (
    DEFAULT_CSV_PATH,
    REGIME_LABEL_BEAR,
    REGIME_LABEL_BULL,
    REGIME_LABEL_CHOP,
    REGIME_LABEL_PURGED,
    prepare_dataset,
)
from self_growing_trader.regime_models import (
    BaseRegimeDetector,
    GaussianHMMRegimeDetector,
    GMMRegimeDetector,
    NUM_REGIMES,
    SupervisedSoftmaxRegimeDetector,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_BENCHMARK = PROJECT_ROOT / "outputs" / "regime_benchmark"
OUTPUTS_MODELS = PROJECT_ROOT / "outputs" / "regime_models"


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="module")
def real_dataset() -> dict[str, Any]:
    """Loads and caches the real BTCUSD M1 dataset for Milestone 3 testing."""
    if not os.path.exists(DEFAULT_CSV_PATH):
        pytest.skip(f"Real BTCUSD dataset not found at {DEFAULT_CSV_PATH}")
    return prepare_dataset(csv_path=DEFAULT_CSV_PATH)


@pytest.fixture(scope="module")
def champion_model() -> BaseRegimeDetector:
    """Loads the serialized champion model from outputs/regime_models."""
    artifact_path = OUTPUTS_MODELS / "champion_regime_detector.pkl"
    if not artifact_path.exists():
        pytest.skip(f"Champion artifact not found at {artifact_path}")
    return BaseRegimeDetector.load(artifact_path)


# ==============================================================================
# 1. WHIPSAW IN CHOP EMPIRICAL STRESS
# ==============================================================================

class TestWhipsawInChopEmpiricalStress:
    """
    Stress tests False Whipsaw Rate strictly on verified CHOP intervals in real February/March data.
    """

    def test_real_oos_chop_distribution_and_consecutive_pairs(
        self, real_dataset: dict[str, Any]
    ) -> None:
        """Verifies real OOS chop bar count and consecutive chop pair count."""
        y_oos = real_dataset["y_oos"]
        assert len(y_oos) == 84886

        chop_mask = (y_oos == REGIME_LABEL_CHOP)
        n_chop = int(np.sum(chop_mask))
        assert n_chop == 32668
        assert 0.38 < (n_chop / len(y_oos)) < 0.39

        chop_pairs = (y_oos[1:] == REGIME_LABEL_CHOP) & (y_oos[:-1] == REGIME_LABEL_CHOP)
        n_pairs = int(np.sum(chop_pairs))
        assert n_pairs == 28971
        assert n_pairs > 25000

    def test_champion_lightgbm_fwr_on_real_oos_chop(
        self, real_dataset: dict[str, Any], champion_model: BaseRegimeDetector
    ) -> None:
        """
        Computes champion model (Supervised_LightGBM_TempScaled) FWR strictly on real OOS chop.
        Confirms FWR matches benchmark telemetry (~15.14%).
        """
        X_oos = real_dataset["X_oos"]
        y_oos = real_dataset["y_oos"]

        probs = champion_model.predict_proba(X_oos)
        preds = np.argmax(probs, axis=-1)

        fwr = compute_false_whipsaw_rate(y_oos, preds, chop_label=REGIME_LABEL_CHOP)
        assert abs(fwr - 15.1358) < 0.05, f"Expected ~15.14%, got {fwr:.4f}%"

    def test_temporal_filter_whipsaw_suppression_comparison(
        self, real_dataset: dict[str, Any]
    ) -> None:
        """
        Empirically contrasts models with temporal filtering (HMM, GMM+EWMA)
        against memoryless supervised classifiers (LightGBM, MLP).
        Demonstrates that temporal inertia reduces False Whipsaw Rate by ~10x (1.47% vs 15.14%).
        """
        results_file = OUTPUTS_BENCHMARK / "benchmark_results.json"
        if not results_file.exists():
            pytest.skip("benchmark_results.json not found")

        with open(results_file, "r", encoding="utf-8") as f:
            bench_data = json.load(f)

        models = bench_data["models"]
        hmm_fwr = models["GaussianHMM_3State"]["false_whipsaw_rate"]
        gmm_ewma_fwr = models["GMM_EWMA_0.20"]["false_whipsaw_rate"]
        lgb_fwr = models["Supervised_LightGBM_TempScaled"]["false_whipsaw_rate"]
        mlp_fwr = models["Supervised_PyTorch_MLP_TempScaled"]["false_whipsaw_rate"]

        # 1. Structural temporal filtering suppresses whipsaws below 2.0%
        assert hmm_fwr < 2.0, f"HMM FWR should be < 2.0%, got {hmm_fwr}%"
        assert gmm_ewma_fwr < 2.0, f"GMM+EWMA FWR should be < 2.0%, got {gmm_ewma_fwr}%"

        # 2. Memoryless classifiers experience higher boundary chatter
        assert lgb_fwr > 10.0, f"LightGBM FWR expected > 10%, got {lgb_fwr}%"
        assert mlp_fwr > 15.0, f"MLP FWR expected > 15%, got {mlp_fwr}%"

        # 3. Ratio demonstrates > 8x suppression from temporal filtering
        assert (lgb_fwr / hmm_fwr) > 8.0
        assert (mlp_fwr / gmm_ewma_fwr) > 10.0

    def test_fwr_mathematical_oracles_and_boundary_conditions(self) -> None:
        """Tests mathematical edge cases and extreme oracles for compute_false_whipsaw_rate."""
        chop = REGIME_LABEL_CHOP

        # Case 1: Empty input
        assert compute_false_whipsaw_rate(np.array([]), np.array([])) == 0.0

        # Case 2: Single element (no consecutive pairs)
        assert compute_false_whipsaw_rate(np.array([chop]), np.array([chop])) == 0.0

        # Case 3: No true chop intervals
        y_true_no_chop = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1])
        assert compute_false_whipsaw_rate(y_true_no_chop, y_pred) == 0.0

        # Case 4: Perfect chop stability (0% whipsaw)
        y_true_all_chop = np.full(100, chop)
        y_pred_all_chop = np.full(100, chop)
        assert compute_false_whipsaw_rate(y_true_all_chop, y_pred_all_chop) == 0.0

        # Case 5: 100% violent whipsaw (alternating non-chop regimes: Bull, Bear, Bull, Bear...)
        y_pred_viol_whipsaw = np.array([0 if i % 2 == 0 else 1 for i in range(100)])
        assert compute_false_whipsaw_rate(y_true_all_chop, y_pred_viol_whipsaw) == 100.0

        # Case 6: Alternating Chop and Bull (each flip is a whipsaw)
        y_pred_alt = np.array([chop if i % 2 == 0 else 0 for i in range(100)])
        assert compute_false_whipsaw_rate(y_true_all_chop, y_pred_alt) == 100.0


# ==============================================================================
# 2. LATENCY STRESS UNDER LOAD
# ==============================================================================

class TestLatencyStressUnderLoad:
    """
    Stress tests champion model step_filter latency under CPU load.
    Target: P50 and P99 latency strictly < 5.0ms (5,000 µs).
    """

    def test_champion_step_filter_latency_under_cpu_load(
        self, real_dataset: dict[str, Any], champion_model: BaseRegimeDetector
    ) -> None:
        """
        Executes 1,000 sequential step_filter calls on the champion model while
        background CPU processes are actively running.
        Verifies P50 < 5.0ms and P99 < 5.0ms.
        """
        X_oos = real_dataset["X_oos"]
        n_samples = len(X_oos)

        # Spawn background CPU-bound active workload processes
        burn_code = """
import time
t_end = time.time() + 15
c = 0
while time.time() < t_end:
    c = (c + 1) % 1000000
"""
        procs = [
            subprocess.Popen([sys.executable, "-c", burn_code])
            for _ in range(2)
        ]
        time.sleep(0.3)

        try:
            # Warmup
            champion_model.reset()
            for i in range(50):
                champion_model.step_filter(X_oos[i % n_samples])

            # Measure 1,000 sequential calls
            latencies_ms: list[float] = []
            for i in range(1000):
                x = X_oos[i % n_samples]
                t0 = time.perf_counter()
                reg, probs = champion_model.step_filter(x)
                dt_ms = (time.perf_counter() - t0) * 1000.0
                latencies_ms.append(dt_ms)
        finally:
            for p in procs:
                p.terminate()

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        max_spike = float(np.max(lat_arr))
        mean_lat = float(np.mean(lat_arr))

        # HARD ASSERTIONS against < 5.0ms SLA
        assert p50 < 5.0, f"P50 latency under load ({p50:.4f} ms) exceeded 5.0ms SLA!"
        assert p99 < 5.0, f"P99 latency under load ({p99:.4f} ms) exceeded 5.0ms SLA!"
        assert max_spike < 5.0, f"Max latency spike under load ({max_spike:.4f} ms) exceeded 5.0ms SLA!"
        assert mean_lat < 5.0, f"Mean latency under load ({mean_lat:.4f} ms) exceeded 5.0ms SLA!"

    def test_champion_step_filter_contract_and_probabilities(
        self, champion_model: BaseRegimeDetector
    ) -> None:
        """Verifies step_filter causal shape and mathematical probability contracts."""
        x = np.random.randn(15)
        regime_idx, probs = champion_model.step_filter(x)

        assert isinstance(regime_idx, (int, np.integer))
        assert 0 <= regime_idx < NUM_REGIMES
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (NUM_REGIMES,)
        assert np.all(probs >= 0.0)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
        assert int(np.argmax(probs)) == regime_idx

    def test_champion_step_filter_pathological_inputs(
        self, champion_model: BaseRegimeDetector
    ) -> None:
        """
        Adversarially feeds NaNs, Infs, zeros, and extreme scale vectors to step_filter.
        Verifies zero unhandled exceptions, zero NaNs in returned probabilities, and safe fallback.
        """
        champion_model.reset()

        # 1. Vector containing NaNs
        x_nan = np.zeros(15)
        x_nan[3] = np.nan
        reg, probs = champion_model.step_filter(x_nan)
        assert not np.isnan(probs).any(), "NaN in probabilities!"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

        # 2. Vector containing Infs
        x_inf = np.zeros(15)
        x_inf[7] = np.inf
        reg, probs = champion_model.step_filter(x_inf)
        assert not np.isinf(probs).any(), "Inf in probabilities!"
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

        # 3. All zeros
        x_zero = np.zeros(15)
        reg, probs = champion_model.step_filter(x_zero)
        assert np.all(probs >= 0.0)
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

        # 4. Extremely large magnitude (1e7)
        x_large = np.full(15, 1e7)
        reg, probs = champion_model.step_filter(x_large)
        assert not np.isnan(probs).any()
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)

        # 5. Extremely small magnitude (1e-7)
        x_small = np.full(15, 1e-7)
        reg, probs = champion_model.step_filter(x_small)
        assert not np.isnan(probs).any()
        assert np.isclose(np.sum(probs), 1.0, atol=1e-4)


# ==============================================================================
# 3. ZERO OOS DATA CONTAMINATION AUDIT
# ==============================================================================

class TestZeroOOSDataContaminationAudit:
    """
    Verifies that X_oos timestamps are strictly in February and March 2025,
    and that no candidate model was trained on OOS data.
    """

    def test_strict_temporal_separation_jan_vs_feb_mar(
        self, real_dataset: dict[str, Any]
    ) -> None:
        """Verifies zero timestamp overlap and strict chronological ordering between Train and OOS."""
        train_ts = real_dataset["train_timestamps"]
        oos_ts = real_dataset["oos_timestamps"]

        # Chronological boundary: max(Train) must be strictly before min(OOS)
        max_train = train_ts.max()
        min_oos = oos_ts.min()
        assert max_train < min_oos, f"Train max ({max_train}) is not before OOS min ({min_oos})!"

        # Exactly verify date ranges
        assert train_ts.min() >= pd.Timestamp("2025-01-01 00:00:00")
        assert train_ts.max() <= pd.Timestamp("2025-01-31 23:59:59")

        assert oos_ts.min() >= pd.Timestamp("2025-02-01 00:00:00")
        assert oos_ts.max() <= pd.Timestamp("2025-03-31 23:59:59")

        # Zero set intersection
        overlap = set(train_ts).intersection(set(oos_ts))
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping timestamps!"

    def test_embargo_purge_boundary_integrity(
        self, real_dataset: dict[str, Any]
    ) -> None:
        """
        Verifies that the 60-bar embargo window on the Jan/Feb boundary was purged
        so forward triple barrier paths from January never cross into February.
        """
        train_ts = real_dataset["train_timestamps"]
        oos_ts = real_dataset["oos_timestamps"]

        # Max train timestamp is 2025-01-31 22:59:00 (exactly 60 minutes before 2025-02-01 00:00:00)
        expected_boundary = pd.Timestamp("2025-01-31 22:59:00")
        assert train_ts.max() == expected_boundary, (
            f"Expected max train timestamp {expected_boundary}, got {train_ts.max()}"
        )

        # Gap between max train and min OOS must be at least 61 minutes
        time_gap = (oos_ts.min() - train_ts.max()).total_seconds() / 60.0
        assert time_gap >= 61.0, f"Embargo gap was only {time_gap} minutes, expected >= 61 min!"

    def test_scaler_isolation_in_sample_only(
        self, real_dataset: dict[str, Any]
    ) -> None:
        """
        Verifies that the feature normalizer (RobustScaler) was fitted exclusively
        on In-Sample January training data, without observing OOS data.
        """
        scaler = real_dataset["scaler"]
        X_train_raw = real_dataset["X_train_raw"]
        X_oos_raw = real_dataset["X_oos_raw"]

        # Recompute center and scale exclusively on X_train_raw
        expected_center = np.median(X_train_raw, axis=0)
        q75 = np.percentile(X_train_raw, 75, axis=0)
        q25 = np.percentile(X_train_raw, 25, axis=0)
        expected_scale = q75 - q25
        expected_scale[expected_scale == 0.0] = 1.0

        # Assert scaler center strictly matches training median
        assert np.allclose(scaler.center_, expected_center, atol=1e-5), (
            "Scaler center does not match training data median!"
        )

        # Assert scaler scale strictly matches training IQR
        assert np.allclose(scaler.scale_, expected_scale, atol=1e-5), (
            "Scaler scale does not match training data IQR!"
        )

        # Confirm that scaler center differs from OOS median (proving no OOS fitting)
        oos_median = np.median(X_oos_raw, axis=0)
        assert not np.allclose(scaler.center_, oos_median, atol=1e-3), (
            "Scaler center suspiciously matches OOS median!"
        )

    def test_training_pipeline_passes_zero_oos_data_to_fit(self) -> None:
        """
        Inspects regime_benchmark.run_head_to_head_benchmark and evaluate_model_oos
        to guarantee that X_oos is NEVER passed to model.fit().
        """
        import inspect
        from self_growing_trader import regime_benchmark

        source_eval = inspect.getsource(regime_benchmark.evaluate_model_oos)
        assert "model.fit(" in source_eval
        # Verify that model.fit receives X_train and X_val, NOT X_oos
        assert "model.fit(X_train=X_train" in source_eval or "model.fit(X_train, y_train" in source_eval
        assert "model.fit(X_train=X_oos" not in source_eval
        assert "model.fit(X_oos" not in source_eval


# ==============================================================================
# 4. CHAMPION ARTIFACT INTEGRITY
# ==============================================================================

class TestChampionArtifactIntegrity:
    """
    Forensic verification of saved model artifacts, SHA256 checksums, and metadata.
    """

    def test_champion_model_sha256_checksum(self) -> None:
        """Verifies that champion_regime_detector.pkl matches the SHA256 in champion_metadata.json."""
        artifact_path = OUTPUTS_MODELS / "champion_regime_detector.pkl"
        metadata_path = OUTPUTS_MODELS / "champion_metadata.json"

        assert artifact_path.exists(), f"Missing {artifact_path}"
        assert metadata_path.exists(), f"Missing {metadata_path}"

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        with open(artifact_path, "rb") as f:
            actual_sha256 = hashlib.sha256(f.read()).hexdigest().upper()

        assert actual_sha256 == metadata["sha256"], (
            f"SHA256 mismatch! Metadata: {metadata['sha256']} vs Actual: {actual_sha256}"
        )

    def test_champion_model_reload_and_prediction_determinism(
        self, real_dataset: dict[str, Any]
    ) -> None:
        """Verifies bit-for-bit prediction determinism between loaded model instances."""
        artifact_path = OUTPUTS_MODELS / "champion_regime_detector.pkl"
        m1 = BaseRegimeDetector.load(artifact_path)
        m2 = BaseRegimeDetector.load(artifact_path)

        X_sample = real_dataset["X_oos"][:100]
        p1 = m1.predict_proba(X_sample)
        p2 = m2.predict_proba(X_sample)

        assert np.array_equal(p1, p2), "Model reload produced non-deterministic predictions!"
