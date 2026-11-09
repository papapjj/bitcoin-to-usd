"""Empirical Adversarial Stress Harness for Milestone 2 Reservoir Buffer.

Challenger M2-1 (Adversarial Empirical Verification).
Focuses on:
1. 10,000 stream transitions across random regimes & memory safety.
2. Uniformity of Algorithm R reservoir sampling (asymptotic C/N probability).
3. 33.3% Mini-batch balance across 1,000 batches (size 96) under normal and skewed streams.
4. TD-error prioritization power-law fidelity and Markov transition weight analysis.
5. Importance Sampling (IS) weights positivity, finiteness, max-normalization (1.0), and beta annealing.
6. Edge cases: starvation deficit redistribution, boundary batch sizes, and priority update interface.
"""
from __future__ import annotations

import math
import random
import time
from typing import Any

import numpy as np
import pytest
import torch
from scipy import stats

from self_growing_trader.balanced_replay_buffer import (
    BalancedMultiRegimeReplayBuffer,
    BalancedMultiRegimeReservoirBuffer,
    normalize_regime,
)


def _make_dummy_step(
    step_id: int,
    regime: str = "UPTREND",
    realized_r: float = 0.5,
    td_error: float = 1.0,
    with_tensors: bool = True,
) -> dict[str, Any]:
    step: dict[str, Any] = {
        "step_id": step_id,
        "regime": regime,
        "realized_r": realized_r,
        "td_error": td_error,
        "lifecycle_stratum": ("entry", "pos", regime),
    }
    if with_tensors:
        step["obs"] = np.random.randn(5, 137).astype(np.float32)
        step["actions"] = np.zeros(5, dtype=np.int64)
        step["rewards"] = np.zeros(5, dtype=np.float32)
    return step


# ============================================================================
# 1. 10,000 Stream Transition Simulation Across Random Regimes
# ============================================================================

def test_10k_stream_simulation_across_random_regimes() -> None:
    """Simulate 10,000 transitions streaming into the reservoir across random regimes."""
    capacity = 128
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=capacity)
    regimes = ["UPTREND", "DOWNTREND", "SIDEWAY"]

    rng = np.random.default_rng(20261020)
    # Generate random stream characteristics
    regime_choices = rng.choice(regimes, size=10_000, p=[0.45, 0.35, 0.20])
    realized_rs = rng.normal(loc=0.05, scale=1.5, size=10_000)
    td_errors = rng.exponential(scale=2.0, size=10_000)

    t0 = time.perf_counter()
    for i in range(10_000):
        step = _make_dummy_step(
            step_id=i,
            regime=str(regime_choices[i]),
            realized_r=float(realized_rs[i]),
            td_error=float(td_errors[i]),
            with_tensors=False,
        )
        ok = buffer.add(
            step,
            regime=str(regime_choices[i]),
            realized_r=float(realized_rs[i]),
            td_error=float(td_errors[i]),
        )
        assert ok is True

    elapsed = time.perf_counter() - t0
    throughput = 10_000 / elapsed

    # Empirical invariants:
    assert buffer.total_admitted == 10_000
    assert sum(buffer.admitted_by_regime.values()) == 10_000
    assert throughput > 2000.0, f"Throughput too low: {throughput:.1f} items/sec"

    # Pool capacity bounds
    assert len(buffer.uptrend_pool) == capacity
    assert len(buffer.downtrend_pool) == capacity
    assert len(buffer.sideway_pool) == capacity
    assert buffer.stats()["total_samples"] == capacity * 3

    # Verification: Markov counts were accumulated
    assert buffer.markov_counts.sum() == 10_000 + 9.0  # 10k transitions + 9 Laplace priors


# ============================================================================
# 2. Uniformity of Algorithm R: Equal Asymptotic Probability C / N
# ============================================================================

def test_algorithm_r_uniform_asymptotic_probability_monte_carlo() -> None:
    """Verify via Monte Carlo (2,500 trials) that each transition has equal asymptotic prob C/N."""
    C = 30  # Capacity
    N = 300  # Stream length (10x capacity)
    M = 2500  # Number of independent Monte Carlo trials
    expected_p = C / N  # 0.10

    survival_counts = np.zeros(N, dtype=np.int64)

    for trial in range(M):
        buf = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=C)
        trial_rng = random.Random(trial + 1000)
        # Stream N items into single regime (UPTREND)
        for i in range(N):
            step = {"step_id": i}
            buf.add(step, regime="UPTREND", realized_r=1.0, rng=trial_rng)

        pool = buf.get_pool("UPTREND")
        assert len(pool) == C
        for exp in pool:
            survival_counts[exp.step_dict["step_id"]] += 1

    empirical_probs = survival_counts / float(M)

    # 1. Overall mean retention probability across all N items:
    mean_p = float(np.mean(empirical_probs))
    assert abs(mean_p - expected_p) < 0.0015, f"Mean probability {mean_p:.5f} deviates from expected {expected_p}"

    # 2. Stream Quintiles: items 0-59 (early), 60-119, 120-179, 180-239, 240-299 (late)
    quintile_means = [
        float(np.mean(empirical_probs[q * 60 : (q + 1) * 60]))
        for q in range(5)
    ]
    for q, q_mean in enumerate(quintile_means):
        assert abs(q_mean - expected_p) < 0.005, (
            f"Quintile {q} mean {q_mean:.4f} deviates significantly from expected {expected_p}. "
            "Algorithm R is not uniform!"
        )

    # 3. Chi-squared goodness of fit test across 10 deciles of the stream
    decile_observed = np.array([
        survival_counts[d * 30 : (d + 1) * 30].sum()
        for d in range(10)
    ], dtype=np.float64)
    decile_expected = np.full(10, (C * M) / 10.0, dtype=np.float64)

    chi2, p_val = stats.chisquare(decile_observed, decile_expected)
    assert p_val > 0.01, f"Chi-squared test rejected uniformity! chi2={chi2:.2f}, p_value={p_val:.4f}"


def test_algorithm_r_10k_stream_decile_presence() -> None:
    """Verify that after a 10,000 item stream, early stream items are not forgotten (anti-FIFO)."""
    C = 128
    N = 10_000
    buf = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=C)

    np.random.seed(42)
    for i in range(N):
        buf.add({"step_id": i}, regime="UPTREND", realized_r=0.5)

    pool = buf.get_pool("UPTREND")
    assert len(pool) == C
    surviving_ids = [e.step_dict["step_id"] for e in pool]

    # Verify that stream items from the first 1,000 steps survived
    early_items = [sid for sid in surviving_ids if sid < 1000]
    late_items = [sid for sid in surviving_ids if sid >= 9000]

    assert len(early_items) > 0, f"Catastrophic forgetting: 0 items survived from first 1,000 steps!"
    assert len(late_items) > 0, f"0 items survived from last 1,000 steps!"

    # In a pure sliding window (FIFO), all 128 items would have sid >= 9872.
    assert min(surviving_ids) < 5000, "Buffer behaved like FIFO, discarded older transitions!"


# ============================================================================
# 3. Mini-Batch Balance: 1,000 Batches (Size 96) ~33.3% Each (+-3%)
# ============================================================================

def test_minibatch_balance_1000_batches_size_96() -> None:
    """Sample 1,000 batches of size 96 and verify mean proportion of Bull, Bear, Sideway is 33.3% +-3%."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    # Populate each pool with 128 items
    for i in range(128):
        buffer.add(_make_dummy_step(i, "UPTREND", td_error=float(i + 1)), regime="UPTREND")
        buffer.add(_make_dummy_step(1000 + i, "DOWNTREND", td_error=float(i + 1)), regime="DOWNTREND")
        buffer.add(_make_dummy_step(2000 + i, "SIDEWAY", td_error=float(i + 1)), regime="SIDEWAY")

    num_batches = 1000
    batch_size = 96

    bull_counts = np.zeros(num_batches, dtype=np.int64)
    bear_counts = np.zeros(num_batches, dtype=np.int64)
    side_counts = np.zeros(num_batches, dtype=np.int64)

    for b in range(num_batches):
        batch = buffer.sample_balanced_batch(batch_size=batch_size)
        assert len(batch) == batch_size

        bull = sum(1 for item in batch if normalize_regime(item["regime"]) == "UPTREND")
        bear = sum(1 for item in batch if normalize_regime(item["regime"]) == "DOWNTREND")
        side = sum(1 for item in batch if normalize_regime(item["regime"]) == "SIDEWAY")

        bull_counts[b] = bull
        bear_counts[b] = bear
        side_counts[b] = side

    prop_bull = bull_counts / float(batch_size)
    prop_bear = bear_counts / float(batch_size)
    prop_side = side_counts / float(batch_size)

    mean_bull = float(np.mean(prop_bull))
    mean_bear = float(np.mean(prop_bear))
    mean_side = float(np.mean(prop_side))

    # Requirement: 33.3% +- 3% (i.e. [30.3%, 36.3%])
    assert abs(mean_bull - 0.3333) < 0.03, f"Bull proportion {mean_bull:.4f} outside [30.3%, 36.3%]"
    assert abs(mean_bear - 0.3333) < 0.03, f"Bear proportion {mean_bear:.4f} outside [30.3%, 36.3%]"
    assert abs(mean_side - 0.3333) < 0.03, f"Sideway proportion {mean_side:.4f} outside [30.3%, 36.3%]"

    # In fact, when pools have >= 32 items, quota is exact (32 each):
    assert np.all(bull_counts == 32)
    assert np.all(bear_counts == 32)
    assert np.all(side_counts == 32)


def test_minibatch_balance_under_extreme_stream_skew() -> None:
    """Verify that even under extreme 80/10/10 stream skew, mini-batches maintain 33.3% balance."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    rng = np.random.default_rng(777)

    # Stream 3,000 transitions with severe skew: 80% Bull, 10% Bear, 10% Sideway
    for i in range(3000):
        r = rng.choice(["UPTREND", "DOWNTREND", "SIDEWAY"], p=[0.80, 0.10, 0.10])
        buffer.add(_make_dummy_step(i, str(r)), regime=str(r))

    # All pools have at least 128 items (3000 * 0.10 = 300 >= 128)
    assert len(buffer.uptrend_pool) == 128
    assert len(buffer.downtrend_pool) == 128
    assert len(buffer.sideway_pool) == 128

    # Sample 500 batches
    for _ in range(500):
        batch = buffer.sample_balanced_batch(batch_size=96)
        bull = sum(1 for item in batch if normalize_regime(item["regime"]) == "UPTREND")
        bear = sum(1 for item in batch if normalize_regime(item["regime"]) == "DOWNTREND")
        side = sum(1 for item in batch if normalize_regime(item["regime"]) == "SIDEWAY")

        # Must NOT reflect the 80/10/10 stream skew!
        assert bull == 32
        assert bear == 32
        assert side == 32


# ============================================================================
# 4. TD-Error Prioritization and Markov Transition Weights
# ============================================================================

def test_td_error_prioritization_power_law_monotonicity() -> None:
    r"""Verify that sampling within a pool strictly obeys TD-error power law p_i \propto (|delta_i| + 1e-5)^0.6."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128, alpha=0.6)

    # Add 5 items with TD errors covering 4 orders of magnitude
    td_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    for idx, td in enumerate(td_values):
        buffer.add(_make_dummy_step(idx, "UPTREND", td_error=td), regime="UPTREND", td_error=td)

    # Populate other pools to satisfy batch quotas
    for i in range(10):
        buffer.add(_make_dummy_step(100 + i, "DOWNTREND"), regime="DOWNTREND")
        buffer.add(_make_dummy_step(200 + i, "SIDEWAY"), regime="SIDEWAY")

    # Repeatedly sample 1 item from UPTREND (batch_size=3 pulls 1 from each pool)
    np.random.seed(999)
    sample_counts = {idx: 0 for idx in range(5)}
    num_samples = 4000
    for _ in range(num_samples):
        batch = buffer.sample_balanced_batch(batch_size=3)
        for item in batch:
            sid = item["step_id"]
            if sid in sample_counts:
                sample_counts[sid] += 1

    # Monotonicity check
    counts_list = [sample_counts[i] for i in range(5)]
    for i in range(len(counts_list) - 1):
        assert counts_list[i] < counts_list[i + 1], (
            f"Non-monotonic priority selection: counts={counts_list}"
        )

    # Spearman rank correlation with theoretical priority
    theo_prios = [(td + 1e-5) ** 0.6 for td in td_values]
    corr, p_val = stats.spearmanr(theo_prios, counts_list)
    assert corr == pytest.approx(1.0, abs=1e-5), f"Rank correlation {corr} is not strictly 1.0"


def test_markov_transition_weights_mathematical_neutralization() -> None:
    """Adversarial Test: Verify the mathematical fact that markov_factor cancels out identically in probs.

    Finding: In sample_balanced_batch():
      markov_factor = 1.0 + 0.5 * next_regime_weights[r_idx]
      raw_prios = np.array(...) * markov_factor
      probs = raw_prios / raw_prios.sum()
    Since markov_factor is a scalar for pool r, (raw * markov_factor) / (sum(raw) * markov_factor)
    cancels out markov_factor identically (diff == 0.0 to machine precision).
    """
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    for i in range(10):
        buffer.add(_make_dummy_step(i, "UPTREND", td_error=float(i + 1)), regime="UPTREND", td_error=float(i + 1))

    pool = buffer.get_pool("UPTREND")
    N = len(pool)

    # Case 1: Extreme Markov transition weight P = 1.0 (markov_factor = 1.5)
    mf_high = 1.0 + 0.5 * 1.0
    raw_high = np.array([(abs(e.td_error) + 1e-5) ** buffer.alpha for e in pool]) * mf_high
    probs_high = raw_high / raw_high.sum()

    # Case 2: Extreme Markov transition weight P = 0.0 (markov_factor = 1.0)
    mf_low = 1.0 + 0.5 * 0.0
    raw_low = np.array([(abs(e.td_error) + 1e-5) ** buffer.alpha for e in pool]) * mf_low
    probs_low = raw_low / raw_low.sum()

    # Case 3: Arbitrary Markov transition weight P = 0.42
    mf_mid = 1.0 + 0.5 * 0.42
    raw_mid = np.array([(abs(e.td_error) + 1e-5) ** buffer.alpha for e in pool]) * mf_mid
    probs_mid = raw_mid / raw_mid.sum()

    max_diff = float(np.max(np.abs(probs_high - probs_low)))
    # Proves mathematically that markov_factor does NOT affect within-pool probabilities
    assert max_diff < 1e-15, f"Expected markov_factor to cancel out mathematically, got max_diff={max_diff}"
    assert np.allclose(probs_high, probs_mid, atol=1e-15)


# ============================================================================
# 5. Importance Sampling (IS) Weights Verification
# ============================================================================

def test_importance_sampling_weights_properties() -> None:
    """Verify IS weights: positive, finite, max-normalized to 1.0, beta annealing."""
    buffer = BalancedMultiRegimeReservoirBuffer(
        capacity_per_regime=128,
        beta_start=0.4,
        beta_end=1.0,
        beta_steps=500,
    )

    # Populate pools with wide TD error spectrum (0.001 to 100.0)
    for i in range(40):
        td = float(0.001 * (1.5 ** i))
        buffer.add(_make_dummy_step(i, "UPTREND", td_error=td), regime="UPTREND", td_error=td)
        buffer.add(_make_dummy_step(100 + i, "DOWNTREND", td_error=td), regime="DOWNTREND", td_error=td)
        buffer.add(_make_dummy_step(200 + i, "SIDEWAY", td_error=td), regime="SIDEWAY", td_error=td)

    # Sample 100 batches and check IS weights properties
    for s in range(100):
        batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=30)
        assert len(batch) == 30
        assert is_weights.shape == (30,)

        # 1. Strictly positive
        assert torch.all(is_weights > 0.0)
        # 2. Strictly finite
        assert torch.all(torch.isfinite(is_weights))
        # 3. Max-normalized to 1.0
        assert torch.max(is_weights).item() == pytest.approx(1.0, abs=1e-5)
        # 4. Embedded weights in step dictionaries match tensor
        for idx, item in enumerate(batch):
            assert item["importance_sampling_weight"] == pytest.approx(is_weights[idx].item(), abs=1e-5)
            assert item["is_weight"] == pytest.approx(is_weights[idx].item(), abs=1e-5)

    # 5. Annealing progression check
    assert buffer.step_count == 100
    assert buffer.beta == pytest.approx(0.4 + 100 / 500 * (1.0 - 0.4), abs=1e-4)

    # Run remaining steps to 500
    for _ in range(400):
        buffer.sample_balanced_batch(batch_size=3)
    assert buffer.beta == pytest.approx(1.0, abs=1e-4)

    # Beyond 500 steps, clamped to 1.0
    buffer.sample_balanced_batch(batch_size=3)
    assert buffer.beta == pytest.approx(1.0, abs=1e-4)


def test_importance_sampling_inverse_probability_bias_correction() -> None:
    r"""Verify that higher priority items receive LOWER IS weights (w_i \propto p_i^-beta)."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128, beta_start=1.0, beta_end=1.0)

    # Add 2 items: 1 low TD error (rarely sampled), 1 high TD error (frequently sampled)
    buffer.add(_make_dummy_step(1, "UPTREND", td_error=0.001), regime="UPTREND", td_error=0.001)
    buffer.add(_make_dummy_step(2, "UPTREND", td_error=100.0), regime="UPTREND", td_error=100.0)

    # Pad other pools
    for i in range(10):
        buffer.add(_make_dummy_step(100 + i, "DOWNTREND"), regime="DOWNTREND")
        buffer.add(_make_dummy_step(200 + i, "SIDEWAY"), regime="SIDEWAY")

    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=6)
    up_items = [(item, w) for item, w in zip(batch, is_weights) if normalize_regime(item["regime"]) == "UPTREND"]

    # In PER, the frequently sampled high-TD item must have a SMALLER or EQUAL IS weight than the rare low-TD item
    if len(up_items) == 2:
        item_low, w_low = [(it, w) for it, w in up_items if it["step_id"] == 1][0]
        item_high, w_high = [(it, w) for it, w in up_items if it["step_id"] == 2][0]
        assert w_low.item() >= w_high.item(), (
            f"Expected w_low >= w_high for bias correction, got w_low={w_low.item()}, w_high={w_high.item()}"
        )


# ============================================================================
# 6. Edge Cases, Starvation, and Priority Updates Interface
# ============================================================================

def test_buffer_starvation_and_graceful_redistribution() -> None:
    """Verify that buffer gracefully redistributes deficits when 2 regimes are completely empty."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    # Only UPTREND has items (50 items)
    for i in range(50):
        buffer.add(_make_dummy_step(i, "UPTREND"), regime="UPTREND")

    # Request batch of 45: quotas are 15 UP, 15 DOWN, 15 SIDEWAY
    # Since DOWN and SIDEWAY are empty, all 45 must be fulfilled by UPTREND
    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=45)
    assert len(batch) == 45
    assert len(is_weights) == 45
    assert all(normalize_regime(item["regime"]) == "UPTREND" for item in batch)
    assert torch.all(is_weights > 0.0)
    assert torch.max(is_weights).item() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("bs", [0, -1, -100])
def test_boundary_batch_sizes(bs: int) -> None:
    """Verify non-positive batch sizes return empty collections cleanly."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    buffer.add(_make_dummy_step(1, "UPTREND"), regime="UPTREND")

    batch_only = buffer.sample_balanced_batch(batch_size=bs)
    assert batch_only == []

    batch, weights = buffer.sample_balanced_reservoir_batch(batch_size=bs)
    assert batch == []
    assert weights.numel() == 0


def test_malformed_td_errors_and_r_values() -> None:
    """Verify robustness to NaN, Inf, and negative TD errors."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)

    # 1. NaN / Inf realized_r rejected
    assert buffer.add(_make_dummy_step(1), regime="UPTREND", realized_r=float("nan")) is False
    assert buffer.add(_make_dummy_step(2), regime="UPTREND", realized_r=float("inf")) is False

    # 2. NaN / Inf td_error replaced by 1.0
    assert buffer.add(_make_dummy_step(3), regime="UPTREND", realized_r=1.0, td_error=float("nan")) is True
    assert buffer.uptrend_pool[0].td_error == 1.0

    assert buffer.add(_make_dummy_step(4), regime="UPTREND", realized_r=1.0, td_error=float("inf")) is True
    assert buffer.uptrend_pool[1].td_error == 1.0

    # 3. Negative TD error is stored as given, but sampling takes abs() safely
    assert buffer.add(_make_dummy_step(5), regime="UPTREND", realized_r=1.0, td_error=-15.5) is True
    assert buffer.uptrend_pool[2].td_error == -15.5
    # Populate other pools and verify sample does not crash or produce complex/NaN numbers
    for i in range(5):
        buffer.add(_make_dummy_step(10 + i, "DOWNTREND"), regime="DOWNTREND")
        buffer.add(_make_dummy_step(20 + i, "SIDEWAY"), regime="SIDEWAY")
    batch, is_weights = buffer.sample_balanced_reservoir_batch(batch_size=3)
    assert len(batch) == 3
    assert torch.all(torch.isfinite(is_weights))


def test_update_priorities_functionality_and_index_limitation() -> None:
    """Verify update_priorities functionality and test pool index metadata preservation."""
    buffer = BalancedMultiRegimeReservoirBuffer(capacity_per_regime=128)
    for i in range(5):
        buffer.add(_make_dummy_step(i, "UPTREND", td_error=1.0), regime="UPTREND", td_error=1.0)

    # Direct priority update by pool index
    buffer.update_priorities("UPTREND", indices=[1, 3], td_errors=[20.0, 40.0])
    pool = buffer.get_pool("UPTREND")
    assert pool[1].td_error == pytest.approx(20.0)
    assert pool[3].td_error == pytest.approx(40.0)

    # Verify invalid indices or NaN deltas are safely ignored without crashing
    buffer.update_priorities("UPTREND", indices=[-1, 999, 0], td_errors=[5.0, 5.0, float("nan")])
    assert pool[0].td_error == pytest.approx(1.0)
