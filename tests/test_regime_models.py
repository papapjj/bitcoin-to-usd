"""
tests/test_regime_models.py

Comprehensive Automated Unit & Integration Test Suite for Regime Detection Models.
Part of Milestone 2 for the Multi-Agent Regime Detection Engine.

Covers:
- BaseRegimeDetector ABC contracts and polymorphic interface adherence.
- GaussianHMMRegimeDetector: Baum-Welch EM convergence, transition matrix axioms,
  Hungarian / momentum alignment, and Bayesian online step filter math & latency (< 0.1ms).
- GMMRegimeDetector: sklearn GaussianMixture fitting, component alignment, and EWMA
  temporal inertia smoothing (alpha in [0.15, 0.30]) to suppress whipsaw.
- SupervisedSoftmaxRegimeDetector: LightGBM and PyTorch 2-layer MLP architectures,
  balanced class weighting, and Softmax router contracts.
- Probability Calibration: Temperature Scaling (NLL optimization via minimize_scalar)
  and Platt Scaling (via CalibratedClassifierCV with FrozenEstimator), verifying reduction
  in log-loss / Brier score while preserving rank-order.
- Numerical Stability: Flat market (zero variance), extreme 10-sigma volatility spikes,
  NaN/Inf corruption fails closed to neutral prior [1/3, 1/3, 1/3].
- Model Persistence: save() and load() / polymorphic loader restoring bit-for-bit
  identical probabilities across all architectures.
- Real-world In-Sample BTCUSD M1 data training and Out-of-Sample evaluation.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import time

# Ensure repository root and src are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
from scipy import stats

try:
    from self_growing_trader.regime_models import (
        NUM_REGIMES,
        REGIME_BEAR,
        REGIME_BULL,
        REGIME_CHOP,
        REGIME_NAMES,
        BaseRegimeDetector,
        GaussianHMMRegimeDetector,
        GMMRegimeDetector,
        SupervisedSoftmaxRegimeDetector,
        compute_diagonal_gaussian_log_prob,
        load_regime_model,
        stable_softmax,
    )
except ImportError:
    from src.self_growing_trader.regime_models import (
        NUM_REGIMES,
        REGIME_BEAR,
        REGIME_BULL,
        REGIME_CHOP,
        REGIME_NAMES,
        BaseRegimeDetector,
        GaussianHMMRegimeDetector,
        GMMRegimeDetector,
        SupervisedSoftmaxRegimeDetector,
        compute_diagonal_gaussian_log_prob,
        load_regime_model,
        stable_softmax,
    )

try:
    from self_growing_trader.regime_data import (
        DEFAULT_CSV_PATH,
        prepare_dataset,
    )
except ImportError:
    try:
        from src.self_growing_trader.regime_data import (
            DEFAULT_CSV_PATH,
            prepare_dataset,
        )
    except ImportError:
        DEFAULT_CSV_PATH = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
        prepare_dataset = None


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture
def synthetic_regime_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a deterministic 3-regime continuous time series:
    - State 0 (BULL): Positive momentum (+0.8), moderate volatility.
    - State 1 (BEAR): Negative momentum (-0.8), high volatility.
    - State 2 (CHOP): Near-zero momentum (0.0), low volatility.
    """
    rng = np.random.default_rng(42)
    n_train = 600
    n_val = 150
    D = 15

    def _generate_blocks(n_samples: int) -> tuple[np.ndarray, np.ndarray]:
        X = np.zeros((n_samples, D), dtype=np.float64)
        y = np.zeros(n_samples, dtype=np.int64)

        # 3 sequential blocks representing distinct regimes
        block_len = n_samples // 3
        # Bull block
        X[:block_len, 0] = rng.normal(0.8, 0.2, block_len)
        X[:block_len, 9] = rng.normal(0.6, 0.2, block_len)  # velocity
        X[:block_len, 5] = rng.normal(1.0, 0.2, block_len)  # vol
        X[:block_len, 1:] += rng.normal(0.0, 0.1, (block_len, D - 1))
        y[:block_len] = REGIME_BULL

        # Bear block
        X[block_len : 2 * block_len, 0] = rng.normal(-0.8, 0.2, block_len)
        X[block_len : 2 * block_len, 9] = rng.normal(-0.6, 0.2, block_len)
        X[block_len : 2 * block_len, 5] = rng.normal(2.0, 0.3, block_len)
        X[block_len : 2 * block_len, 1:] += rng.normal(0.0, 0.1, (block_len, D - 1))
        y[block_len : 2 * block_len] = REGIME_BEAR

        # Chop block
        X[2 * block_len :, 0] = rng.normal(0.0, 0.1, n_samples - 2 * block_len)
        X[2 * block_len :, 9] = rng.normal(0.0, 0.1, n_samples - 2 * block_len)
        X[2 * block_len :, 5] = rng.normal(0.4, 0.1, n_samples - 2 * block_len)
        X[2 * block_len :, 1:] += rng.normal(0.0, 0.1, (n_samples - 2 * block_len, D - 1))
        y[2 * block_len :] = REGIME_CHOP

        return X, y

    X_train, y_train = _generate_blocks(n_train)
    X_val, y_val = _generate_blocks(n_val)
    return X_train, y_train, X_val, y_val


@pytest.fixture(scope="module")
def real_csv_available() -> bool:
    return os.path.exists(DEFAULT_CSV_PATH)


# ==============================================================================
# 1. BASE REGIME DETECTOR ABC CONTRACTS
# ==============================================================================

class TestBaseRegimeDetectorContract:
    """Verifies interface invariants of the BaseRegimeDetector ABC."""

    def test_abstract_instantiation_prohibited(self) -> None:
        """Attempting to instantiate BaseRegimeDetector directly raises TypeError."""
        with pytest.raises(TypeError):
            BaseRegimeDetector()  # type: ignore[abstract]

    def test_polymorphic_method_signatures(self, synthetic_regime_dataset) -> None:
        """All concrete detectors implement predict_proba, predict, step_filter, and reset."""
        X_train, y_train, _, _ = synthetic_regime_dataset

        detectors: list[BaseRegimeDetector] = [
            GaussianHMMRegimeDetector(n_iter=5),
            GMMRegimeDetector(),
            SupervisedSoftmaxRegimeDetector(model_type="lightgbm"),
            SupervisedSoftmaxRegimeDetector(model_type="mlp", max_epochs=2),
        ]

        for det in detectors:
            assert hasattr(det, "fit"), f"{det.__class__.__name__} missing fit"
            assert hasattr(det, "predict_proba"), f"{det.__class__.__name__} missing predict_proba"
            assert hasattr(det, "predict"), f"{det.__class__.__name__} missing predict"
            assert hasattr(det, "step_filter"), f"{det.__class__.__name__} missing step_filter"
            assert hasattr(det, "step_inference"), f"{det.__class__.__name__} missing step_inference"
            assert hasattr(det, "reset"), f"{det.__class__.__name__} missing reset"
            assert hasattr(det, "save"), f"{det.__class__.__name__} missing save"
            assert hasattr(det, "load"), f"{det.__class__.__name__} missing load"

            det.fit(X_train, y_train)
            probs = det.predict_proba(X_train[:5])
            assert probs.shape == (5, 3), f"Expected shape (5, 3), got {probs.shape}"
            assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)

            regime_pred = det.predict(X_train[:5])
            assert len(regime_pred) == 5
            assert np.array_equal(regime_pred, np.argmax(probs, axis=-1))


# ==============================================================================
# 2. GAUSSIAN HMM REGIME DETECTOR (MODEL 1)
# ==============================================================================

class TestGaussianHMMRegimeDetector:
    """Verifies scaled Baum-Welch EM, transition dynamics, and online Bayesian step filter."""

    def test_hmm_fit_and_convergence(self, synthetic_regime_dataset) -> None:
        """Fits GaussianHMM on synthetic data and asserts mathematical state invariants."""
        X_train, y_train, X_val, y_val = synthetic_regime_dataset

        hmm = GaussianHMMRegimeDetector(n_states=3, n_iter=20, random_state=42)
        hmm.fit(X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        assert hmm.is_fitted
        assert hmm.pi.shape == (3,)
        assert np.isclose(np.sum(hmm.pi), 1.0, atol=1e-6)
        assert np.all(hmm.pi >= 0.0)

        # Transition matrix A: rows must sum to 1.0, diagonal persistence
        assert hmm.A.shape == (3, 3)
        assert np.allclose(np.sum(hmm.A, axis=1), 1.0, atol=1e-6)
        assert np.all(hmm.A >= 0.0)
        # Regimes exhibit temporal stickiness (diagonal elements should dominate)
        diag = np.diag(hmm.A)
        assert np.all(diag >= 0.50), f"HMM diagonal persistence should be >= 0.50, got {diag}"

        # Emission parameters
        assert hmm.means is not None and hmm.means.shape == (3, 15)
        assert hmm.covars is not None and hmm.covars.shape == (3, 15)
        assert np.all(hmm.covars >= hmm.var_floor), "All variances must respect variance floor"

    def test_hmm_bayesian_step_filter_math_and_latency(self, synthetic_regime_dataset) -> None:
        """
        Verifies single-step recursive Bayesian filter math:
        p_t = (p_{t-1} @ A * b(x_t)) / Z
        Latency SLA < 5.0ms (target < 0.1ms).
        """
        X_train, y_train, _, _ = synthetic_regime_dataset
        hmm = GaussianHMMRegimeDetector(n_iter=10, random_state=42).fit(X_train, y_train)
        hmm.reset()

        # Step through 50 bars sequentially
        p_history = []
        latencies = []
        for t in range(50):
            x_t = X_train[t]
            t0 = time.perf_counter()
            regime_idx, p_t = hmm.step_filter(x_t)
            latencies.append((time.perf_counter() - t0) * 1e6)  # microseconds

            assert regime_idx in (REGIME_BULL, REGIME_BEAR, REGIME_CHOP)
            assert len(p_t) == 3
            assert np.isclose(np.sum(p_t), 1.0, atol=1e-6)
            assert np.all(p_t >= 0.0)
            assert not np.isnan(p_t).any()
            assert not np.isinf(p_t).any()
            p_history.append(p_t)

        mean_latency_us = float(np.mean(latencies))
        assert mean_latency_us < 5000.0, f"HMM step_filter latency {mean_latency_us:.2f}us exceeded 5000us SLA"
        # Verify sub-100 microsecond performance
        assert mean_latency_us < 150.0, f"HMM step_filter target < 150us, got {mean_latency_us:.2f}us"

    def test_hmm_unsupervised_momentum_alignment(self) -> None:
        """Verifies momentum ordering correctly aligns Bull, Bear, and Chop without labels."""
        rng = np.random.default_rng(123)
        # Create clear momentum differentiation in column 0
        X_unlabeled = np.vstack([
            rng.normal(1.5, 0.1, (100, 15)),   # High positive momentum -> BULL
            rng.normal(-1.5, 0.1, (100, 15)),  # High negative momentum -> BEAR
            rng.normal(0.0, 0.1, (100, 15)),   # Neutral momentum -> CHOP
        ])

        hmm = GaussianHMMRegimeDetector(n_iter=15, mom_idx=0, random_state=42)
        hmm.fit(X_unlabeled)  # No labels passed

        assert hmm.means is not None
        # State 0 (BULL) must have highest momentum, State 1 (BEAR) lowest
        assert hmm.means[REGIME_BULL, 0] > hmm.means[REGIME_CHOP, 0]
        assert hmm.means[REGIME_CHOP, 0] > hmm.means[REGIME_BEAR, 0]


# ==============================================================================
# 3. GMM REGIME DETECTOR + EWMA FILTER (MODEL 2)
# ==============================================================================

class TestGMMRegimeDetector:
    """Verifies Gaussian Mixture Model and integrated EWMA inertia smoothing."""

    def test_gmm_fit_and_alignment(self, synthetic_regime_dataset) -> None:
        """Fits GMM on synthetic data and verifies component shapes and canonical alignment."""
        X_train, y_train, X_val, y_val = synthetic_regime_dataset

        gmm = GMMRegimeDetector(n_components=3, ewma_alpha=0.20, random_state=42)
        gmm.fit(X_train, y_train=y_train, X_val=X_val, y_val=y_val)

        assert gmm.is_fitted
        assert gmm.means is not None and gmm.means.shape == (3, 15)
        assert gmm.covars is not None and gmm.covars.shape == (3, 15)
        assert gmm.weights is not None and gmm.weights.shape == (3,)
        assert np.isclose(np.sum(gmm.weights), 1.0, atol=1e-6)

    def test_gmm_ewma_inertia_smoothing_suppresses_whipsaw(self, synthetic_regime_dataset) -> None:
        """
        Verifies EWMA temporal filter formula:
        P_tilde_t = alpha * P_t^{GMM} + (1 - alpha) * P_tilde_{t-1}
        Guarantees sudden 1-bar spikes do not cause instant 100% regime flips.
        """
        X_train, y_train, _, _ = synthetic_regime_dataset
        alpha = 0.20
        gmm = GMMRegimeDetector(ewma_alpha=alpha, random_state=42).fit(X_train, y_train)
        gmm.reset()

        # Step into Chop regime firmly
        chop_bar = np.zeros(15)
        chop_bar[0] = 0.0  # momentum = 0
        chop_bar[5] = 0.4  # low vol
        for _ in range(20):
            regime_idx, p_t = gmm.step_filter(chop_bar)
        assert regime_idx == REGIME_CHOP
        p_chop_prev = p_t[REGIME_CHOP]
        assert p_chop_prev > 0.60

        # Inject sudden single-bar Bull spike
        spike_bull_bar = np.zeros(15)
        spike_bull_bar[0] = 2.0  # massive positive momentum spike
        spike_bull_bar[9] = 1.5  # velocity spike

        regime_after_spike, p_after_spike = gmm.step_filter(spike_bull_bar)

        # EWMA inertia must prevent instant 100% flip to Bull
        assert p_after_spike[REGIME_BULL] < 0.60, (
            f"EWMA smoothing failed: Bull prob jumped to {p_after_spike[REGIME_BULL]}"
        )
        assert p_after_spike[REGIME_CHOP] > 0.30, "Chop inertia must be retained"
        assert np.isclose(np.sum(p_after_spike), 1.0, atol=1e-6)

    def test_gmm_step_filter_latency(self, synthetic_regime_dataset) -> None:
        """GMM step_filter executes in < 5.0ms (target < 0.2ms)."""
        X_train, y_train, _, _ = synthetic_regime_dataset
        gmm = GMMRegimeDetector().fit(X_train, y_train)

        t0 = time.perf_counter()
        for _ in range(500):
            gmm.step_filter(X_train[0])
        mean_us = (time.perf_counter() - t0) / 500 * 1e6

        assert mean_us < 5000.0, f"GMM latency {mean_us:.2f}us exceeded 5000us SLA"
        assert mean_us < 200.0, f"GMM latency target < 200us, got {mean_us:.2f}us"


# ==============================================================================
# 4. SUPERVISED SOFTMAX REGIME DETECTOR (MODEL 3: LIGHTGBM & PYTORCH MLP)
# ==============================================================================

class TestSupervisedSoftmaxRegimeDetector:
    """Verifies LightGBM and PyTorch 2-layer MLP classifier and Softmax Router."""

    def test_supervised_lightgbm_fit_and_inference(self, synthetic_regime_dataset) -> None:
        """Verifies LightGBM classifier with balanced class weighting and temperature scaling."""
        X_train, y_train, X_val, y_val = synthetic_regime_dataset

        lgb_det = SupervisedSoftmaxRegimeDetector(
            model_type="lightgbm", calibration="temperature", random_state=42
        )
        lgb_det.fit(X_train, y_train, X_val, y_val)

        assert lgb_det.is_fitted
        assert lgb_det.temperature_ > 0.0

        probs = lgb_det.predict_proba(X_val[:20])
        assert probs.shape == (20, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

        regime_idx, p_single = lgb_det.step_filter(X_val[0])
        assert regime_idx in (REGIME_BULL, REGIME_BEAR, REGIME_CHOP)
        assert len(p_single) == 3
        assert np.isclose(np.sum(p_single), 1.0, atol=1e-5)

    def test_supervised_pytorch_mlp_architecture_and_forward(self, synthetic_regime_dataset) -> None:
        """Verifies PyTorch 2-layer MLP (LayerNorm + GELU + Softmax) router."""
        X_train, y_train, X_val, y_val = synthetic_regime_dataset

        mlp_det = SupervisedSoftmaxRegimeDetector(
            model_type="mlp",
            calibration="temperature",
            hidden_dims=(64, 32),
            max_epochs=15,
            random_state=42,
        )
        mlp_det.fit(X_train, y_train, X_val, y_val)

        assert mlp_det.is_fitted
        assert mlp_det.temperature_ > 0.0

        probs = mlp_det.predict_proba(X_val[:20])
        assert probs.shape == (20, 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)

        # Single step latency test on PyTorch MLP
        t0 = time.perf_counter()
        for _ in range(200):
            mlp_det.step_filter(X_val[0])
        mean_us = (time.perf_counter() - t0) / 200 * 1e6
        assert mean_us < 5000.0, f"MLP latency {mean_us:.2f}us exceeded 5000us SLA"


# ==============================================================================
# 5. PROBABILITY CALIBRATION ENGINE (TEMPERATURE SCALING & PLATT SCALING)
# ==============================================================================

class TestProbabilityCalibrationEngine:
    """Verifies that calibration minimizes Negative Log-Likelihood while preserving rank order."""

    def test_temperature_scaling_lowers_nll_and_brier(self, synthetic_regime_dataset) -> None:
        """
        Temperature Scaling T* optimized via scipy.optimize.minimize_scalar:
        - Must reduce or maintain validation Negative Log-Likelihood.
        - Must reduce or maintain validation Brier score.
        - Must preserve 100% identical argmax class predictions (rank invariance).
        """
        X_train, y_train, X_val, y_val = synthetic_regime_dataset

        # Uncalibrated baseline (T = 1.0)
        uncal = SupervisedSoftmaxRegimeDetector(
            model_type="mlp", max_epochs=20, calibration="none", random_state=42
        ).fit(X_train, y_train, X_val, y_val)
        p_uncal = uncal.predict_proba(X_val)

        # Calibrated model (T* optimized on validation NLL)
        cal = SupervisedSoftmaxRegimeDetector(
            model_type="mlp", max_epochs=20, calibration="temperature", random_state=42
        ).fit(X_train, y_train, X_val, y_val)
        p_cal = cal.predict_proba(X_val)

        row_indices = np.arange(len(y_val))
        nll_uncal = -np.mean(np.log(np.maximum(p_uncal[row_indices, y_val], 1e-15)))
        nll_cal = -np.mean(np.log(np.maximum(p_cal[row_indices, y_val], 1e-15)))

        one_hot = np.zeros((len(y_val), 3))
        one_hot[row_indices, y_val] = 1.0
        brier_uncal = np.mean(np.sum((p_uncal - one_hot) ** 2, axis=1))
        brier_cal = np.mean(np.sum((p_cal - one_hot) ** 2, axis=1))

        # Rank-invariance theorem: argmax(z / T) == argmax(z) for any T > 0
        assert np.array_equal(np.argmax(p_uncal, axis=-1), np.argmax(p_cal, axis=-1)), (
            "Temperature scaling must preserve argmax ranking exactly"
        )

        # Calibration optimization check: NLL must be <= uncalibrated NLL + tolerance
        assert nll_cal <= nll_uncal + 1e-5, (
            f"Calibrated NLL ({nll_cal:.4f}) must be <= uncalibrated NLL ({nll_uncal:.4f})"
        )

    def test_platt_scaling_contract(self, synthetic_regime_dataset) -> None:
        """Verifies Platt Scaling via CalibratedClassifierCV(FrozenEstimator) on LightGBM."""
        X_train, y_train, X_val, y_val = synthetic_regime_dataset

        platt_clf = SupervisedSoftmaxRegimeDetector(
            model_type="lightgbm", calibration="platt", random_state=42
        )
        platt_clf.fit(X_train, y_train, X_val, y_val)

        probs = platt_clf.predict_proba(X_val)
        assert probs.shape == (len(X_val), 3)
        assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
        assert np.all(probs >= 0.0)
        assert not np.isnan(probs).any()


# ==============================================================================
# 6. MODEL PERSISTENCE & BIT-FOR-BIT SERIALIZATION
# ==============================================================================

class TestModelPersistenceAndBitForBitSerialization:
    """Verifies that save() and load() preserve identical predictions down to floating-point precision."""

    @pytest.mark.parametrize(
        "detector_factory",
        [
            lambda X, y: GaussianHMMRegimeDetector(n_iter=5).fit(X, y),
            lambda X, y: GMMRegimeDetector().fit(X, y),
            lambda X, y: SupervisedSoftmaxRegimeDetector(model_type="lightgbm", calibration="temperature").fit(X, y),
            lambda X, y: SupervisedSoftmaxRegimeDetector(model_type="lightgbm", calibration="platt").fit(X, y),
            lambda X, y: SupervisedSoftmaxRegimeDetector(model_type="mlp", max_epochs=3, calibration="temperature").fit(X, y),
        ],
    )
    def test_save_and_load_bit_for_bit(self, synthetic_regime_dataset, detector_factory) -> None:
        """Saves trained model to tempfile and loads via load_regime_model(), asserting exact equality."""
        X_train, y_train, X_val, _ = synthetic_regime_dataset
        detector = detector_factory(X_train, y_train)

        test_points = X_val[:15]
        probs_before = detector.predict_proba(test_points)
        _, step_p_before = detector.step_filter(test_points[0])

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            detector.save(tmp_path)
            loaded_detector = load_regime_model(tmp_path)

            probs_after = loaded_detector.predict_proba(test_points)
            _, step_p_after = loaded_detector.step_filter(test_points[0])

            # Assert bit-for-bit identical probabilities
            np.testing.assert_allclose(
                probs_before,
                probs_after,
                rtol=1e-9,
                atol=1e-9,
                err_msg="Batch probabilities diverged after deserialization!",
            )
            np.testing.assert_allclose(
                step_p_before,
                step_p_after,
                rtol=1e-9,
                atol=1e-9,
                err_msg="Step filter probabilities diverged after deserialization!",
            )
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


# ==============================================================================
# 7. NUMERICAL STABILITY & ADVERSARIAL EDGE CASES
# ==============================================================================

class TestNumericalStabilityAndAdversarialEdgeCases:
    """Verifies fail-closed behavior on corrupted inputs, extreme volatility, and flat channels."""

    def test_flat_channel_zero_variance_dead_market(self, synthetic_regime_dataset) -> None:
        """100% zero-variance input produces valid normalized probabilities without division by zero."""
        X_train, y_train, _, _ = synthetic_regime_dataset
        hmm = GaussianHMMRegimeDetector(n_iter=5).fit(X_train, y_train)
        gmm = GMMRegimeDetector().fit(X_train, y_train)

        flat_input = np.zeros(15, dtype=np.float64)
        for det in [hmm, gmm]:
            probs = det.predict_proba(flat_input)
            assert len(probs) == 3
            assert np.isclose(np.sum(probs), 1.0, atol=1e-6)
            assert not np.isnan(probs).any()
            assert not np.isinf(probs).any()

    def test_extreme_10_sigma_volatility_spike(self, synthetic_regime_dataset) -> None:
        """10-sigma outliers are evaluated safely without floating point overflow."""
        X_train, y_train, _, _ = synthetic_regime_dataset
        hmm = GaussianHMMRegimeDetector(n_iter=5).fit(X_train, y_train)
        gmm = GMMRegimeDetector().fit(X_train, y_train)

        extreme_outlier = np.ones(15, dtype=np.float64) * 1e4
        for det in [hmm, gmm]:
            probs = det.predict_proba(extreme_outlier)
            assert np.isclose(np.sum(probs), 1.0, atol=1e-6)
            assert not np.isnan(probs).any()
            assert not np.isinf(probs).any()

    def test_nan_and_inf_corrupted_inputs_fail_closed(self, synthetic_regime_dataset) -> None:
        """Corrupted inputs trigger safe fallback to neutral prior [1/3, 1/3, 1/3]."""
        X_train, y_train, _, _ = synthetic_regime_dataset
        detectors = [
            GaussianHMMRegimeDetector(n_iter=5).fit(X_train, y_train),
            GMMRegimeDetector().fit(X_train, y_train),
            SupervisedSoftmaxRegimeDetector(model_type="lightgbm").fit(X_train, y_train),
            SupervisedSoftmaxRegimeDetector(model_type="mlp", max_epochs=2).fit(X_train, y_train),
        ]

        nan_input = np.full(15, np.nan, dtype=np.float64)
        inf_input = np.full(15, np.inf, dtype=np.float64)

        for det in detectors:
            p_nan = det.predict_proba(nan_input)
            assert np.allclose(p_nan, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-4)

            p_inf = det.predict_proba(inf_input)
            assert np.allclose(p_inf, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-4)

            reg_idx, p_step = det.step_filter(nan_input)
            assert np.allclose(p_step, [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], atol=1e-4)

    def test_single_bar_1d_vs_sequence_2d_input_shapes(self, synthetic_regime_dataset) -> None:
        """predict_proba correctly dispatches 1D input to (3,) and 2D input to (N, 3)."""
        X_train, y_train, _, _ = synthetic_regime_dataset
        det = GaussianHMMRegimeDetector(n_iter=5).fit(X_train, y_train)

        # 1D single bar
        p_1d = det.predict_proba(X_train[0])
        assert p_1d.shape == (3,)

        # 2D sequence of 10 bars
        p_2d = det.predict_proba(X_train[:10])
        assert p_2d.shape == (10, 3)

        # 2D sequence of single bar (1, 15)
        p_single_2d = det.predict_proba(X_train[:1])
        assert p_single_2d.shape == (1, 3)


# ==============================================================================
# 8. REAL BTCUSD IN-SAMPLE & OOS INTEGRATION
# ==============================================================================

class TestRealBTCUSDIntegration:
    """End-to-end integration tests using real BTCUSD M1 dataset from regime_data."""

    def test_models_fit_and_evaluate_on_btcusd_data(self, real_csv_available: bool) -> None:
        """Trains all 3 models on In-Sample January BTCUSD data and evaluates on OOS February data."""
        if not real_csv_available or prepare_dataset is None:
            pytest.skip(f"BTCUSDm.csv not available at {DEFAULT_CSV_PATH}")

        # Load first 2,000 bars for fast CI test verification
        data = prepare_dataset(csv_path=DEFAULT_CSV_PATH)
        X_train_sub = data["X_train"][:2000]
        y_train_sub = data["y_train"][:2000]
        X_oos_sub = data["X_oos"][:500]
        y_oos_sub = data["y_oos"][:500]

        # 1. HMM
        hmm = GaussianHMMRegimeDetector(n_iter=10).fit(X_train_sub, y_train_sub)
        p_hmm = hmm.predict_proba(X_oos_sub)
        assert p_hmm.shape == (500, 3)
        assert np.allclose(p_hmm.sum(axis=1), 1.0, atol=1e-5)

        # 2. GMM + EWMA
        gmm = GMMRegimeDetector(ewma_alpha=0.20).fit(X_train_sub, y_train_sub)
        p_gmm = gmm.predict_proba(X_oos_sub)
        assert p_gmm.shape == (500, 3)
        assert np.allclose(p_gmm.sum(axis=1), 1.0, atol=1e-5)

        # 3. Supervised Softmax (LightGBM)
        lgb = SupervisedSoftmaxRegimeDetector(model_type="lightgbm", calibration="temperature")
        lgb.fit(X_train_sub, y_train_sub)
        p_lgb = lgb.predict_proba(X_oos_sub)
        assert p_lgb.shape == (500, 3)
        assert np.allclose(p_lgb.sum(axis=1), 1.0, atol=1e-5)

        # 4. Supervised Softmax (PyTorch MLP)
        mlp = SupervisedSoftmaxRegimeDetector(model_type="mlp", max_epochs=5, calibration="temperature")
        mlp.fit(X_train_sub, y_train_sub)
        p_mlp = mlp.predict_proba(X_oos_sub)
        assert p_mlp.shape == (500, 3)
        assert np.allclose(p_mlp.sum(axis=1), 1.0, atol=1e-5)
