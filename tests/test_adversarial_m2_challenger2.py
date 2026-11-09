"""
tests/test_adversarial_m2_challenger2.py

Adversarial Stress Test Suite for Milestone 2: Advanced Regime Models & Calibration.
Challenger 2 Empirical Verification Harness.

Covers:
1. Bit-for-bit serialization hardening across 10,000 samples for all architectures:
   - GaussianHMMRegimeDetector
   - GMMRegimeDetector (EWMA alpha=0.20)
   - SupervisedSoftmaxRegimeDetector (LightGBM, Temperature Scaling)
   - SupervisedSoftmaxRegimeDetector (LightGBM, Platt Scaling)
   - SupervisedSoftmaxRegimeDetector (MLP, Temperature Scaling)
   - SupervisedSoftmaxRegimeDetector (MLP, Platt Scaling)
   Asserts maximum absolute probability difference is exactly 0.0, and sequential
   step_filter is bit-for-bit identical across 500 steps.

2. Calibration efficacy verification:
   - Validates that Temperature Scaling on overconfident validation logits strictly
     improves (decreases) Negative Log-Likelihood and Brier score.
   - Evaluates both LightGBM and PyTorch MLP classifiers trained on noisy market-like
     distributions, proving NLL and Brier score strictly improve while ranking is 100% preserved.
   - Tests extreme boundary conditions (logits > 1000, logits < -1000, T -> 0, T -> inf).

3. Whipsaw suppression benchmark in simulated noisy chop:
   - Evaluates pure GMM (unmitigated, alpha=1.0) vs GMM+EWMA (alpha=0.20) vs GaussianHMM
     over 2,000 bars of mean-reverting chop with high-frequency noise.
   - Verifies unmitigated model suffers > 30% False Whipsaw Rate (FWR).
   - Verifies HMM and GMM+EWMA achieve < 15% FWR (> 50% whipsaw suppression).
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import warnings

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from self_growing_trader.regime_models import (
    NUM_REGIMES,
    REGIME_BEAR,
    REGIME_BULL,
    REGIME_CHOP,
    GaussianHMMRegimeDetector,
    GMMRegimeDetector,
    SupervisedSoftmaxRegimeDetector,
    load_regime_model,
    stable_softmax,
)


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="module")
def synthetic_training_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generates synthetic 15-dimensional regime training and validation data with realistic overlap."""
    rng = np.random.RandomState(42)
    n_train = 1200
    n_val = 400
    d = 15

    def make_samples(n: int) -> tuple[np.ndarray, np.ndarray]:
        X = rng.randn(n, d)
        logits = np.zeros((n, 3))
        logits[:, 0] = 1.4 * X[:, 0] + 0.5 * X[:, 1]
        logits[:, 1] = -1.4 * X[:, 0] + 0.5 * X[:, 2]
        logits[:, 2] = -0.5 * X[:, 1] - 0.5 * X[:, 2]
        probs = stable_softmax(logits, temperature=1.0)
        y = np.array([rng.choice([0, 1, 2], p=probs[i]) for i in range(n)], dtype=np.int64)
        return X, y

    X_train, y_train = make_samples(n_train)
    X_val, y_val = make_samples(n_val)
    return X_train, y_train, X_val, y_val


@pytest.fixture(scope="module")
def stress_10k_samples() -> np.ndarray:
    """Generates 10,000 diverse adversarial stress samples."""
    rng = np.random.RandomState(1337)
    n = 10000
    d = 15
    X = np.zeros((n, d), dtype=np.float64)

    # 1. Standard normal block (4,000 samples)
    X[:4000] = rng.randn(4000, d)

    # 2. Heavy-tailed Student-t block (2,000 samples)
    X[4000:6000] = rng.standard_t(df=3, size=(2000, d))

    # 3. High-correlation collinear block (2,000 samples)
    base = rng.randn(2000, 1)
    X[6000:8000] = base * np.ones((1, d)) + 0.05 * rng.randn(2000, d)

    # 4. Extreme boundary and repeated pattern block (2,000 samples)
    X[8000:8500] = rng.uniform(-10.0, 10.0, size=(500, d))
    X[8500:9000] = 0.0  # Zero flatline
    pattern = rng.randn(10, d)
    X[9000:10000] = np.tile(pattern, (100, 1))  # Repeated cyclical pattern

    return X


# ==============================================================================
# CHALLENGE 1: BIT-FOR-BIT SERIALIZATION HARDENING (10,000 SAMPLES)
# ==============================================================================

@pytest.mark.parametrize(
    "model_factory,model_name",
    [
        (lambda: GaussianHMMRegimeDetector(n_states=3, n_iter=10, random_state=42), "GaussianHMM"),
        (lambda: GMMRegimeDetector(n_components=3, ewma_alpha=0.20, random_state=42), "GMM_EWMA"),
        (lambda: SupervisedSoftmaxRegimeDetector(model_type="lightgbm", calibration="temperature", random_state=42), "LGBM_Temp"),
        (lambda: SupervisedSoftmaxRegimeDetector(model_type="lightgbm", calibration="platt", random_state=42), "LGBM_Platt"),
        (lambda: SupervisedSoftmaxRegimeDetector(model_type="mlp", calibration="temperature", max_epochs=6, random_state=42), "MLP_Temp"),
        (lambda: SupervisedSoftmaxRegimeDetector(model_type="mlp", calibration="platt", max_epochs=6, random_state=42), "MLP_Platt"),
    ],
)
def test_bit_for_bit_serialization_hardening_10000_samples(
    model_factory,
    model_name: str,
    synthetic_training_data,
    stress_10k_samples: np.ndarray,
) -> None:
    """
    Adversarial challenge: Save each model, reload via load_regime_model(),
    and verify bit-for-bit identity across 10,000 stress samples (difference == 0.0).
    Also verify online recursive step_filter bit-for-bit identity over 500 steps.
    """
    X_train, y_train, X_val, y_val = synthetic_training_data
    model = model_factory()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / f"{model_name}.pkl"
        model.save(ckpt_path)

        assert ckpt_path.exists()
        assert ckpt_path.stat().st_size > 0

        # Reload model via polymorphic loader
        loaded_model = load_regime_model(ckpt_path)

        # 1. Batch predict_proba on 10,000 samples
        orig_probs = model.predict_proba(stress_10k_samples)
        loaded_probs = loaded_model.predict_proba(stress_10k_samples)

        assert orig_probs.shape == (10000, 3)
        assert loaded_probs.shape == (10000, 3)

        # Bit-for-bit assertion: maximum absolute difference MUST be 0.0
        max_abs_diff = float(np.max(np.abs(orig_probs - loaded_probs)))
        print(f"[{model_name}] 10k batch max abs diff: {max_abs_diff}")
        assert max_abs_diff == 0.0, (
            f"Bit-for-bit serialization check failed for {model_name}: "
            f"max difference = {max_abs_diff} > 0.0"
        )
        assert np.array_equal(orig_probs, loaded_probs), (
            f"Probabilities are not bit-for-bit identical for {model_name}"
        )

        # Discrete predict check
        orig_preds = model.predict(stress_10k_samples)
        loaded_preds = loaded_model.predict(stress_10k_samples)
        assert np.array_equal(orig_preds, loaded_preds), (
            f"Discrete predictions differ after deserialization for {model_name}"
        )

        # 2. Sequential online step_filter over 500 steps
        model.reset()
        loaded_model.reset()

        step_samples = stress_10k_samples[:500]
        for t in range(len(step_samples)):
            r1, p1 = model.step_filter(step_samples[t])
            r2, p2 = loaded_model.step_filter(step_samples[t])
            assert r1 == r2, f"step_filter regime mismatch at step {t} for {model_name}"
            step_diff = float(np.max(np.abs(p1 - p2)))
            assert step_diff == 0.0, (
                f"step_filter prob difference {step_diff} at step {t} for {model_name}"
            )


# ==============================================================================
# CHALLENGE 2: CALIBRATION EFFICACY VERIFICATION
# ==============================================================================

def compute_nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Computes Negative Log-Likelihood (multiclass cross-entropy)."""
    n = len(y_true)
    probs_safe = np.clip(probs, 1e-15, 1.0)
    row_idx = np.arange(n)
    log_p = np.log(probs_safe[row_idx, y_true])
    return float(-np.mean(log_p))


def compute_brier_score(probs: np.ndarray, y_true: np.ndarray, n_classes: int = 3) -> float:
    """Computes multiclass Brier score: mean squared probability error."""
    n = len(y_true)
    one_hot = np.zeros((n, n_classes), dtype=np.float64)
    one_hot[np.arange(n), y_true] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=-1)))


def test_temperature_scaling_calibration_efficacy() -> None:
    """
    Adversarial verification of probability calibration:
    1. Tests on synthetic overconfident classifier logits:
       Verifies Temperature Scaling strictly lowers both NLL and Brier score.
    2. Verifies rank ordering (argmax) and full ranking (argsort) is 100% invariant.
    """
    rng = np.random.RandomState(42)
    n = 2000
    y_true = rng.choice([0, 1, 2], size=n, p=[0.35, 0.35, 0.30])

    # Construct synthetic overconfident logits
    base_logits = rng.randn(n, 3)
    for i in range(n):
        if rng.rand() < 0.70:
            base_logits[i, y_true[i]] += 2.5
    overconfident_logits = base_logits * 4.0

    uncalibrated_probs = stable_softmax(overconfident_logits, temperature=1.0)
    nll_uncal = compute_nll(uncalibrated_probs, y_true)
    brier_uncal = compute_brier_score(uncalibrated_probs, y_true)

    t_opt = SupervisedSoftmaxRegimeDetector._optimize_temperature(overconfident_logits, y_true)
    print(f"\n[Synthetic Calibration] Optimal Temperature: {t_opt:.4f}")
    assert t_opt > 1.0, f"Expected T* > 1.0 for overconfident logits, got {t_opt}"

    calibrated_probs = stable_softmax(overconfident_logits, temperature=t_opt)
    nll_cal = compute_nll(calibrated_probs, y_true)
    brier_cal = compute_brier_score(calibrated_probs, y_true)

    print(f"  Uncalibrated NLL: {nll_uncal:.4f} -> Calibrated NLL: {nll_cal:.4f} (diff = {nll_cal - nll_uncal:+.4f})")
    print(f"  Uncalibrated Brier: {brier_uncal:.4f} -> Calibrated Brier: {brier_cal:.4f} (diff = {brier_cal - brier_uncal:+.4f})")

    # Strict improvements
    assert nll_cal < nll_uncal, f"NLL did not strictly improve: {nll_cal} >= {nll_uncal}"
    assert brier_cal < brier_uncal, f"Brier score did not strictly improve: {brier_cal} >= {brier_uncal}"

    # Strict rank-order invariance
    uncal_rank = np.argsort(uncalibrated_probs, axis=-1)
    cal_rank = np.argsort(calibrated_probs, axis=-1)
    assert np.array_equal(uncal_rank, cal_rank), "Ranking was altered by temperature scaling!"

    uncal_argmax = np.argmax(uncalibrated_probs, axis=-1)
    cal_argmax = np.argmax(calibrated_probs, axis=-1)
    assert np.array_equal(uncal_argmax, cal_argmax), "Argmax classification changed!"


def test_temperature_scaling_on_trained_detectors(synthetic_training_data) -> None:
    """
    Verifies that for trained LightGBM and MLP detectors, calibration
    strictly improves both validation NLL and Brier score while preserving rank order.
    """
    X_train, y_train, X_val, y_val = synthetic_training_data

    # 1. Test LightGBM
    lgb_detector = SupervisedSoftmaxRegimeDetector(
        model_type="lightgbm", calibration="temperature", random_state=42
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lgb_detector.fit(X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    logits_val = lgb_detector._get_logits(X_val)
    uncal_p = stable_softmax(logits_val, temperature=1.0)
    cal_p = lgb_detector.predict_proba(X_val)

    nll_uncal = compute_nll(uncal_p, y_val)
    nll_cal = compute_nll(cal_p, y_val)
    brier_uncal = compute_brier_score(uncal_p, y_val)
    brier_cal = compute_brier_score(cal_p, y_val)

    print(f"\n[Trained LightGBM] T*={lgb_detector.temperature_:.4f}")
    print(f"  NLL  : {nll_uncal:.4f} -> {nll_cal:.4f} (diff = {nll_cal - nll_uncal:+.6f})")
    print(f"  Brier: {brier_uncal:.4f} -> {brier_cal:.4f} (diff = {brier_cal - brier_uncal:+.6f})")

    assert nll_cal < nll_uncal, f"LightGBM NLL did not strictly improve: {nll_cal} >= {nll_uncal}"
    assert brier_cal < brier_uncal, f"LightGBM Brier did not strictly improve: {brier_cal} >= {brier_uncal}"
    assert np.array_equal(np.argmax(uncal_p, axis=-1), np.argmax(cal_p, axis=-1))
    assert np.array_equal(np.argsort(uncal_p, axis=-1), np.argsort(cal_p, axis=-1))

    # 2. Test PyTorch MLP
    mlp_detector = SupervisedSoftmaxRegimeDetector(
        model_type="mlp", calibration="temperature", max_epochs=15, random_state=42
    )
    mlp_detector.fit(X_train, y_train=y_train, X_val=X_val, y_val=y_val)

    logits_val_mlp = mlp_detector._get_logits(X_val)
    uncal_p_mlp = stable_softmax(logits_val_mlp, temperature=1.0)
    cal_p_mlp = mlp_detector.predict_proba(X_val)

    nll_uncal_m = compute_nll(uncal_p_mlp, y_val)
    nll_cal_m = compute_nll(cal_p_mlp, y_val)
    brier_uncal_m = compute_brier_score(uncal_p_mlp, y_val)
    brier_cal_m = compute_brier_score(cal_p_mlp, y_val)

    print(f"\n[Trained MLP] T*={mlp_detector.temperature_:.4f}")
    print(f"  NLL  : {nll_uncal_m:.4f} -> {nll_cal_m:.4f} (diff = {nll_cal_m - nll_uncal_m:+.6f})")
    print(f"  Brier: {brier_uncal_m:.4f} -> {brier_cal_m:.4f} (diff = {brier_cal_m - brier_uncal_m:+.6f})")

    assert nll_cal_m < nll_uncal_m, f"MLP NLL did not strictly improve: {nll_cal_m} >= {nll_uncal_m}"
    assert brier_cal_m < brier_uncal_m, f"MLP Brier did not strictly improve: {brier_cal_m} >= {brier_uncal_m}"
    assert np.array_equal(np.argmax(uncal_p_mlp, axis=-1), np.argmax(cal_p_mlp, axis=-1))
    assert np.array_equal(np.argsort(uncal_p_mlp, axis=-1), np.argsort(cal_p_mlp, axis=-1))


def test_temperature_scaling_boundary_stability() -> None:
    """Stress tests stable_softmax numerical boundaries (extreme logits, extreme T)."""
    # 1. Extreme positive logits
    z_pos = np.array([1000.0, 1005.0, 995.0])
    p_pos = stable_softmax(z_pos, temperature=1.0)
    assert not np.isnan(p_pos).any() and not np.isinf(p_pos).any()
    assert abs(np.sum(p_pos) - 1.0) < 1e-9
    assert np.argmax(p_pos) == 1

    # 2. Extreme negative logits
    z_neg = np.array([-1000.0, -1005.0, -995.0])
    p_neg = stable_softmax(z_neg, temperature=1.0)
    assert not np.isnan(p_neg).any() and not np.isinf(p_neg).any()
    assert abs(np.sum(p_neg) - 1.0) < 1e-9
    assert np.argmax(p_neg) == 2

    # 3. Near-zero temperature (T -> 0)
    p_cold = stable_softmax(np.array([1.0, 2.0, 0.5]), temperature=1e-6)
    assert abs(p_cold[1] - 1.0) < 1e-4

    # 4. Infinite temperature (T -> inf)
    p_hot = stable_softmax(np.array([1.0, 2.0, 0.5]), temperature=1e6)
    assert np.allclose(p_hot, 1.0 / 3.0, atol=1e-4)


# ==============================================================================
# CHALLENGE 3: WHIPSAW SUPPRESSION BENCHMARK IN SIMULATED CHOP
# ==============================================================================

def compute_false_whipsaw_rate(predictions: np.ndarray) -> float:
    """
    Computes False Whipsaw Rate (FWR) as the proportion of consecutive bars
    where the predicted regime flips:
    FWR = (1 / (T - 1)) * sum(y_t != y_{t-1})
    """
    preds = np.asarray(predictions, dtype=np.int64)
    if len(preds) < 2:
        return 0.0
    flips = np.sum(preds[1:] != preds[:-1])
    return float(flips / (len(preds) - 1))


def test_whipsaw_suppression_simulated_noisy_chop() -> None:
    """
    Adversarial Whipsaw Suppression Benchmark:
    Feed simulated noisy mean-reverting data through pure GMM vs GMM+EWMA vs HMM.
    Verify:
    1. Pure unmitigated GMM (alpha=1.0) experiences high whipsaw rate (> 30%).
    2. Gaussian HMM and GMM+EWMA achieve < 15% whipsaw rate.
    3. Both mitigated models suppress whipsaws by > 50% relative to pure GMM.
    """
    rng = np.random.RandomState(42)
    d = 15

    # 1. Train models on persistent sequential regime data
    X_tr = []
    y_tr = []
    for _ in range(25):
        # Bull block (40 bars)
        for _ in range(40):
            X_tr.append([rng.normal(0.8, 0.3)] + list(rng.normal(0, 0.4, d - 1)))
            y_tr.append(REGIME_BULL)
        # Bear block (40 bars)
        for _ in range(40):
            X_tr.append([rng.normal(-0.8, 0.3)] + list(rng.normal(0, 0.4, d - 1)))
            y_tr.append(REGIME_BEAR)
        # Chop block (40 bars)
        for _ in range(40):
            X_tr.append([rng.normal(0.0, 0.25)] + list(rng.normal(0, 0.4, d - 1)))
            y_tr.append(REGIME_CHOP)

    X_train = np.array(X_tr, dtype=np.float64)
    y_train = np.array(y_tr, dtype=np.int64)

    # Train HMM (Baum-Welch learns state persistence A_ii ~ 0.98)
    hmm = GaussianHMMRegimeDetector(n_states=3, n_iter=25, random_state=42)
    hmm.fit(X_train, y_train=y_train)

    # Train Pure Unmitigated GMM (ewma_alpha=1.0 -> no temporal smoothing)
    gmm_pure = GMMRegimeDetector(n_components=3, ewma_alpha=1.0, random_state=42)
    gmm_pure.fit(X_train, y_train=y_train)

    # Train GMM with EWMA temporal smoothing (alpha=0.15)
    gmm_ewma = GMMRegimeDetector(n_components=3, ewma_alpha=0.15, random_state=42)
    gmm_ewma.fit(X_train, y_train=y_train)

    # 2. Simulate 2,000 bars of mean-reverting noisy chop
    # High-frequency oscillations across zero simulate price bouncing in a consolidation channel
    n_chop = 2000
    X_chop = np.zeros((n_chop, d), dtype=np.float64)

    val = 0.0
    for t in range(n_chop):
        val = 0.65 * val + rng.normal(0, 0.35)
        X_chop[t, 0] = val
        X_chop[t, 1:] = rng.normal(0, 0.3, d - 1)

    # 3. Evaluate causal online streaming step_filter for all 3 models
    hmm.reset()
    gmm_pure.reset()
    gmm_ewma.reset()

    preds_hmm = []
    preds_gmm_pure = []
    preds_gmm_ewma = []

    for t in range(n_chop):
        r_hmm, _ = hmm.step_filter(X_chop[t])
        r_pure, _ = gmm_pure.step_filter(X_chop[t])
        r_ewma, _ = gmm_ewma.step_filter(X_chop[t])

        preds_hmm.append(r_hmm)
        preds_gmm_pure.append(r_pure)
        preds_gmm_ewma.append(r_ewma)

    fwr_gmm_pure = compute_false_whipsaw_rate(np.array(preds_gmm_pure))
    fwr_gmm_ewma = compute_false_whipsaw_rate(np.array(preds_gmm_ewma))
    fwr_hmm = compute_false_whipsaw_rate(np.array(preds_hmm))

    print(f"\n--- Whipsaw Benchmark over {n_chop} noisy chop bars ---")
    print(f"Pure Unmitigated GMM (alpha=1.0) FWR : {fwr_gmm_pure * 100:.2f}%")
    print(f"GMM + EWMA Filter   (alpha=0.15) FWR : {fwr_gmm_ewma * 100:.2f}%")
    print(f"Gaussian HMM (Bayesian filter)   FWR : {fwr_hmm * 100:.2f}%")

    # Assertions per dispatch criteria:
    # 1. Unmitigated model whipsaw rate > 30%
    assert fwr_gmm_pure > 0.30, (
        f"Pure GMM was expected to have > 30% whipsaw rate in chop, but got {fwr_gmm_pure * 100:.2f}%"
    )

    # 2. Mitigated models achieve < 15% whipsaw rate
    assert fwr_gmm_ewma < 0.15, (
        f"GMM+EWMA failed whipsaw threshold (< 15%): got {fwr_gmm_ewma * 100:.2f}%"
    )
    assert fwr_hmm < 0.15, (
        f"Gaussian HMM failed whipsaw threshold (< 15%): got {fwr_hmm * 100:.2f}%"
    )

    # 3. Relative suppression > 50%
    suppression_ewma = 1.0 - (fwr_gmm_ewma / fwr_gmm_pure)
    suppression_hmm = 1.0 - (fwr_hmm / fwr_gmm_pure)
    print(f"GMM+EWMA Whipsaw Suppression Ratio: {suppression_ewma * 100:.1f}%")
    print(f"HMM Whipsaw Suppression Ratio     : {suppression_hmm * 100:.1f}%")

    assert suppression_ewma > 0.50, f"EWMA suppression ratio {suppression_ewma:.2f} <= 0.50"
    assert suppression_hmm > 0.50, f"HMM suppression ratio {suppression_hmm:.2f} <= 0.50"
