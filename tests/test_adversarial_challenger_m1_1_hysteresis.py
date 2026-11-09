"""
tests/test_adversarial_challenger_m1_1_hysteresis.py

Adversarial Stress Test Suite for Milestone 1 (challenger_m1_1):
Dual-Champion Router Hysteresis, Confirmation Buffer & Contest Zone Verification:
1. Rapid Regime Oscillation:
   - Alternating Bull/Bear oscillation suppression across confirmation_bars in [1, 2, 3, 5, 10]
   - Established state immunity to single-bar opposite-direction noise spikes
   - Quantitative measurement of whipsaw reduction under Markovian noise
2. Contest Zone Boundary Stress:
   - Precise floating-point boundaries at P=0.55 and margin=0.10 (strict inequality)
   - State retention (S_t = S_{t-1}) across 2D grid sweep in [0.45, 0.55] x [0.45, 0.55]
   - Intermittent contest zone poisoning (verifying pending counter resets to 0)
   - Long-duration contest zone wandering (1000 bars)
3. Sudden Chop Spike Dynamics:
   - Exact threshold boundary at P_Chop = 0.40
   - Counter clearance: near-complete confirmation buffer instantly aborted by Chop spike
   - Pure Chop regime sustained stability and Bear specialist routing
   - Empirical analysis of Chop vs Bull/Bear priority when both thresholds are met
4. Extreme & Malformed Inputs:
   - All zeros [0, 0, 0], negative probabilities, sums > 1.0, sums < 1.0
   - Non-finite inputs (NaN, +Inf, -Inf)
   - Shape variations (< 3 elements, > 3 elements, empty, numpy arrays, string floats)
5. Recurrent State Freezing & Exception Isolation:
   - 2,000-step continuous inference checking bit-for-bit inactive memory freeze
   - Fault isolation under injected sub-policy exceptions
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from self_growing_trader.dual_champion_router import DualChampionRouter


class AdversarialMockPolicy:
    """Mock policy that tracks 4096-D recurrent LSTM memory states and call counts."""

    def __init__(self, name: str, input_size: int = 137, output_size: int = 20) -> None:
        self.name = name
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_state = np.zeros((5, 4096), dtype=np.float32)
        self.cell_state = np.zeros((5, 4096), dtype=np.float32)
        self.step_count = 0
        self.account_context: dict[str, Any] = {}
        self.decision_context: dict[str, Any] = {}
        self.position_context: Any = None
        self.soft_probabilities: tuple[float, ...] | None = None
        self.experience_buffer: list[Any] = []
        self.ensure_compatible_called = 0

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[str, ...]:
        self.step_count += 1
        delta = float(self.step_count)
        self.hidden_state += delta
        self.cell_state += delta * 2.0
        return (f"{self.name}_trace_{self.step_count}",)

    def ensure_compatible(self, *args: Any, **kwargs: Any) -> bool:
        self.ensure_compatible_called += 1
        return True

    def reset_runtime_state(self) -> None:
        self.step_count = 0
        self.hidden_state.fill(0.0)
        self.cell_state.fill(0.0)

    def set_account_context(self, context: dict[str, Any], positions_pipe: str = "") -> None:
        self.account_context = dict(context)
        self.account_context["positions_pipe"] = positions_pipe

    def set_decision_context(self, **kwargs: Any) -> None:
        self.decision_context = dict(kwargs)

    def set_position_context(self, *args: Any, **kwargs: Any) -> None:
        self.position_context = (args, kwargs)

    def set_regime_soft_probabilities(
        self,
        probs: Any,
        regime_name: str | None = None,
        regime_idx: int | None = None,
    ) -> None:
        self.soft_probabilities = tuple(float(x) for x in probs)


# ==============================================================================
# 1. RAPID REGIME OSCILLATION & WHIPSAW SUPPRESSION
# ==============================================================================

class TestAdversarialRegimeOscillation:
    """Stress tests proving confirmation buffer suppresses whipsaws under high-frequency oscillation."""

    def test_alternating_bull_bear_zero_flips_with_confirmation_buffer(self) -> None:
        """Rapidly alternating Bull and Bear each bar for 1000 bars.

        With confirmation_bars = 2, zero regime switches must occur.
        """
        bull = AdversarialMockPolicy("bull")
        bear = AdversarialMockPolicy("bear")
        router = DualChampionRouter(
            bull_policy=bull,
            bear_policy=bear,
            bull_threshold=0.55,
            bear_threshold=0.55,
            chop_threshold=0.40,
            confirmation_bars=2,
        )

        assert router.current_regime == "CHOP"
        assert router.active_policy is bear

        bull_packet = [0.80, 0.10, 0.10]
        bear_packet = [0.10, 0.80, 0.10]

        for step in range(1000):
            packet = bull_packet if step % 2 == 0 else bear_packet
            router.set_regime_soft_probabilities(packet)

            # Invariant: Regime must remain locked in CHOP (zero flips)
            assert router.current_regime == "CHOP", f"Whipsaw flip at step {step}"
            assert router.current_regime_index == 2
            assert router.active_policy is bear

            # Invariant: Counter for candidate cannot exceed 1 and opposite counter is 0
            if step % 2 == 0:
                assert router._pending_bull_count == 1
                assert router._pending_bear_count == 0
            else:
                assert router._pending_bull_count == 0
                assert router._pending_bear_count == 1

    def test_established_bull_state_immune_to_single_bar_bear_glitches(self) -> None:
        """An established BULL state is immune to 500 alternating Bear noise bars."""
        bull = AdversarialMockPolicy("bull")
        bear = AdversarialMockPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=3)

        # Confirm BULL state via 3 consecutive bull bars
        for _ in range(3):
            router.set_regime_soft_probabilities([0.75, 0.10, 0.15])
        assert router.current_regime == "BULL"
        assert router.active_policy is bull

        # Bombard with alternating Bear and Bull ticks for 500 bars
        bear_tick = [0.05, 0.85, 0.10]
        bull_tick = [0.85, 0.05, 0.10]
        for step in range(500):
            tick = bear_tick if step % 2 == 0 else bull_tick
            router.set_regime_soft_probabilities(tick)
            assert router.current_regime == "BULL", f"Failed at noise step {step}"
            assert router.active_policy is bull
            assert router._pending_bear_count <= 1

    def test_established_bear_state_immune_to_single_bar_bull_glitches(self) -> None:
        """An established BEAR state is immune to 500 alternating Bull noise bars."""
        bull = AdversarialMockPolicy("bull")
        bear = AdversarialMockPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=3)

        # Confirm BEAR state
        for _ in range(3):
            router.set_regime_soft_probabilities([0.10, 0.75, 0.15])
        assert router.current_regime == "BEAR"
        assert router.active_policy is bear

        bull_tick = [0.85, 0.05, 0.10]
        bear_tick = [0.05, 0.85, 0.10]
        for step in range(500):
            tick = bull_tick if step % 2 == 0 else bear_tick
            router.set_regime_soft_probabilities(tick)
            assert router.current_regime == "BEAR", f"Failed at noise step {step}"
            assert router.active_policy is bear
            assert router._pending_bull_count <= 1

    def test_quantitative_whipsaw_suppression_ratio(self) -> None:
        """Empirically measure flip reduction across confirmation thresholds [1, 2, 3, 5]."""
        np.random.seed(42)
        n_steps = 1200
        synthetic_stream = []
        archetypes = [
            [0.75, 0.10, 0.15],  # Bull
            [0.10, 0.75, 0.15],  # Bear
            [0.20, 0.20, 0.60],  # Chop
        ]
        curr = 0
        for _ in range(n_steps):
            if np.random.rand() < 0.40:
                curr = np.random.choice([0, 1, 2])
            noise = np.random.normal(0, 0.02, 3)
            p = np.clip(np.array(archetypes[curr]) + noise, 0.01, 0.98)
            p = p / p.sum()
            synthetic_stream.append(p.tolist())

        flips = {}
        for conf in [1, 2, 3, 5]:
            router = DualChampionRouter(
                bull_policy=AdversarialMockPolicy("bull"),
                bear_policy=AdversarialMockPolicy("bear"),
                confirmation_bars=conf,
            )
            count = 0
            prev = router.current_regime
            for p in synthetic_stream:
                router.set_regime_soft_probabilities(p)
                if router.current_regime != prev:
                    count += 1
                    prev = router.current_regime
            flips[conf] = count

        # Verify monotonic decrease in flips
        assert flips[1] > flips[2] > flips[3] >= flips[5]
        # Calculate empirical suppression ratio
        ratio_3 = 1.0 - (flips[3] / flips[1])
        ratio_5 = 1.0 - (flips[5] / flips[1])
        assert ratio_3 >= 0.35, f"Expected >= 35% suppression for conf=3, got {ratio_3:.1%}"
        assert ratio_5 >= 0.50, f"Expected >= 50% suppression for conf=5, got {ratio_5:.1%}"


# ==============================================================================
# 2. CONTEST ZONE BOUNDARY STRESS
# ==============================================================================

class TestAdversarialContestZone:
    """Stress tests verifying contest zone [0.45, 0.55] and margin boundaries."""

    def test_strict_margin_inequality_at_boundary(self) -> None:
        """p_bull > p_bear + 0.10 requires strict inequality; exact 0.10 margin must not switch."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            bull_threshold=0.55,
            bear_threshold=0.55,
            confirmation_bars=1,
        )

        # Margin is exactly 0.10: 0.55 - 0.45 = 0.10
        router.set_regime_soft_probabilities([0.55, 0.45, 0.00])
        assert router.current_regime == "CHOP", "Exact 0.10 margin must not trigger Bull (requires strictly > 0.10)"

        # Margin is 0.1000000001 (strictly > 0.10)
        router.set_regime_soft_probabilities([0.5500000001, 0.4499999999, 0.00])
        assert router.current_regime == "BULL", "Strictly greater margin must trigger Bull"

    def test_contest_zone_grid_sweep_preserves_state(self) -> None:
        """Sweeping across contest zone grid points retains previous regime 100%."""
        for initial_regime in ["BULL", "BEAR", "CHOP"]:
            router = DualChampionRouter(
                bull_policy=AdversarialMockPolicy("bull"),
                bear_policy=AdversarialMockPolicy("bear"),
                confirmation_bars=1,
            )
            if initial_regime == "BULL":
                router.set_regime_soft_probabilities([0.80, 0.10, 0.10])
            elif initial_regime == "BEAR":
                router.set_regime_soft_probabilities([0.10, 0.80, 0.10])
            else:
                router.set_regime_soft_probabilities([0.20, 0.20, 0.60])
            assert router.current_regime == initial_regime

            # Grid of contest points where neither reaches 0.55+margin and Chop < 0.40
            contest_points = [
                (0.48, 0.48, 0.04),
                (0.50, 0.50, 0.00),
                (0.46, 0.46, 0.08),
                (0.52, 0.46, 0.02),  # margin 0.06 <= 0.10
                (0.46, 0.52, 0.02),  # margin 0.06 <= 0.10
                (0.54, 0.45, 0.01),  # margin 0.09 <= 0.10
                (0.45, 0.54, 0.01),  # margin 0.09 <= 0.10
                (0.49, 0.49, 0.02),
            ]
            for pb, pr, pc in contest_points:
                router.set_regime_soft_probabilities([pb, pr, pc])
                assert router.current_regime == initial_regime
                assert router._pending_bull_count == 0
                assert router._pending_bear_count == 0

    def test_intermittent_contest_zone_prevents_counter_accumulation(self) -> None:
        """Interleaved [Bull, Contest, Bull, Contest...] must never confirm Bull under conf=2."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=2,
        )

        for step in range(50):
            if step % 2 == 0:
                router.set_regime_soft_probabilities([0.75, 0.10, 0.15])
                assert router._pending_bull_count == 1
                assert router.current_regime == "CHOP"
            else:
                router.set_regime_soft_probabilities([0.48, 0.48, 0.04])
                assert router._pending_bull_count == 0
                assert router._pending_bear_count == 0
                assert router.current_regime == "CHOP"

        assert router.current_regime == "CHOP"

    def test_long_duration_contest_zone_wandering(self) -> None:
        """1000 consecutive steps in contest zone hold initial state with zero counter accumulation."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=1,
        )
        router.set_regime_soft_probabilities([0.70, 0.10, 0.20])
        assert router.current_regime == "BULL"

        np.random.seed(999)
        for _ in range(1000):
            pb = float(np.random.uniform(0.46, 0.52))
            diff = float(np.random.uniform(-0.06, 0.06))
            pr = float(np.clip(pb + diff, 0.45, 0.53))
            pc = max(0.0, 1.0 - pb - pr)
            router.set_regime_soft_probabilities([pb, pr, pc])
            assert router.current_regime == "BULL"
            assert router._pending_bull_count == 0
            assert router._pending_bear_count == 0


# ==============================================================================
# 3. SUDDEN CHOP SPIKE DYNAMICS
# ==============================================================================

class TestAdversarialChopSpike:
    """Stress tests for sudden Chop spikes, boundary transitions, and priority ordering."""

    def test_exact_chop_boundary_transition(self) -> None:
        """P_Chop = 0.400000 triggers CHOP immediately; 0.399999 in contest zone holds previous state."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            chop_threshold=0.40,
            confirmation_bars=1,
        )
        router.set_regime_soft_probabilities([0.70, 0.10, 0.20])
        assert router.current_regime == "BULL"

        # 0.399999 with neither reaching 0.55 -> contest zone holds BULL
        router.set_regime_soft_probabilities([0.31, 0.29, 0.399999])
        assert router.current_regime == "BULL"

        # 0.400000 -> switches immediately to CHOP
        router.set_regime_soft_probabilities([0.30, 0.30, 0.400000])
        assert router.current_regime == "CHOP"
        assert router.current_regime_index == 2
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0

    def test_chop_spike_aborts_near_complete_confirmation_buffer(self) -> None:
        """A near-complete confirmation buffer (4 of 5 bars) is instantly cleared by a Chop spike."""
        bull = AdversarialMockPolicy("bull")
        bear = AdversarialMockPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=5)

        # Accumulate 4 bars of Bull
        for step in range(1, 5):
            router.set_regime_soft_probabilities([0.80, 0.05, 0.15])
            assert router._pending_bull_count == step
            assert router.current_regime == "CHOP"

        # Sudden Chop spike
        router.set_regime_soft_probabilities([0.10, 0.10, 0.80])
        assert router.current_regime == "CHOP"
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0

        # Next Bull bar starts from 1, NOT 5
        router.set_regime_soft_probabilities([0.80, 0.05, 0.15])
        assert router._pending_bull_count == 1
        assert router.current_regime == "CHOP"

    def test_pure_chop_regime_sustained_stability(self) -> None:
        """In pure Chop distributions (P_Chop >= 0.40, P_Bull < 0.55, P_Bear < 0.55), state is 100% stable."""
        bull = AdversarialMockPolicy("bull")
        bear = AdversarialMockPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        np.random.seed(555)
        for _ in range(500):
            pc = float(np.random.uniform(0.40, 0.95))
            remainder = 1.0 - pc
            # Ensure both pb and pr are strictly below 0.55
            min_pb = max(0.0, remainder - 0.50)
            max_pb = min(0.50, remainder)
            pb = float(np.random.uniform(min_pb, max_pb))
            pr = remainder - pb
            assert pb < 0.55 and pr < 0.55, f"Generator error: pb={pb}, pr={pr}"
            router.set_regime_soft_probabilities([pb, pr, pc])

            assert router.current_regime == "CHOP"
            assert router.active_policy is bear
            assert router._pending_bull_count == 0
            assert router._pending_bear_count == 0

    def test_chop_vs_bull_priority_under_simultaneous_thresholds(self) -> None:
        """Empirical demonstration: When P_Chop >= 0.40 AND P_Bull >= 0.55 (e.g. [0.56, 0.04, 0.40]),

        due to if/elif evaluation order, Bull is evaluated before Chop in DualChampionRouter.
        """
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=1,
        )
        # Vector with both P_Bull >= 0.55 and P_Chop >= 0.40
        conflicting_probs = [0.56, 0.04, 0.40]

        router.set_regime_soft_probabilities(conflicting_probs)
        # Empirically verify that p_bull >= 0.55 branch executes first
        assert router.current_regime == "BULL", (
            "Empirical verification: p_bull is evaluated before p_chop in current code structure"
        )


# ==============================================================================
# 4. EXTREME & MALFORMED SOFT PROBABILITY INPUTS
# ==============================================================================

class TestAdversarialMalformedInputs:
    """Stress tests ensuring graceful handling of malformed and out-of-spec probability vectors."""

    def test_all_zeros_handled_gracefully(self) -> None:
        """[0, 0, 0] does not crash, holds previous state via contest zone, and clears counters."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
        )
        router.set_regime_soft_probabilities([0.70, 0.10, 0.20])
        assert router.current_regime == "BULL"

        router.set_regime_soft_probabilities([0.0, 0.0, 0.0])
        assert router.current_regime == "BULL"
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0
        assert router.current_probabilities == (0.0, 0.0, 0.0)

    def test_negative_probabilities_handled_without_exception(self) -> None:
        """Negative probability values do not crash the router."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
        )
        router.set_regime_soft_probabilities([-0.5, -0.2, -0.8])
        assert router.current_regime == "CHOP"
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0

    def test_sums_greater_than_one_deterministic(self) -> None:
        """Probabilities summing to > 1.0 execute deterministically."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=1,
        )
        # Bull dominant (> 0.55 and margin > 0.10)
        router.set_regime_soft_probabilities([10.0, 1.0, 0.0])
        assert router.current_regime == "BULL"

        # Bear dominant
        router.set_regime_soft_probabilities([1.0, 10.0, 0.0])
        assert router.current_regime == "BEAR"

        # Equal values: margin is 0 <= 0.10, Chop is 5.0 >= 0.40 -> CHOP
        router.set_regime_soft_probabilities([5.0, 5.0, 5.0])
        assert router.current_regime == "CHOP"

    def test_sums_less_than_one_deterministic(self) -> None:
        """Probabilities summing to < 1.0 execute deterministically."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=1,
        )
        router.set_regime_soft_probabilities([0.10, 0.70, 0.20])
        assert router.current_regime == "BEAR"

        # [0.1, 0.1, 0.1] sum=0.3 -> contest zone holds BEAR
        router.set_regime_soft_probabilities([0.1, 0.1, 0.1])
        assert router.current_regime == "BEAR"

    def test_nan_and_inf_inputs_handled(self) -> None:
        """Non-finite floats (NaN, +Inf, -Inf) do not raise unhandled exceptions."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=1,
        )
        router.set_regime_soft_probabilities([0.70, 0.10, 0.20])
        assert router.current_regime == "BULL"

        # NaN values fall safely into contest zone and hold state
        router.set_regime_soft_probabilities([float("nan"), float("nan"), float("nan")])
        assert router.current_regime == "BULL"
        assert router._pending_bull_count == 0
        assert router._pending_bear_count == 0

        # +Inf
        router.set_regime_soft_probabilities([float("inf"), 0.0, 0.0])
        assert router.current_regime == "BULL"

        # -Inf
        router.set_regime_soft_probabilities([float("-inf"), float("-inf"), float("-inf")])
        assert router.current_regime == "BULL"

    def test_variable_length_and_type_variations(self) -> None:
        """Accepts numpy arrays, string floats, extra-length vectors, and ignores under-length vectors."""
        router = DualChampionRouter(
            bull_policy=AdversarialMockPolicy("bull"),
            bear_policy=AdversarialMockPolicy("bear"),
            confirmation_bars=1,
        )

        # 4-element list (extra element ignored)
        router.set_regime_soft_probabilities([0.80, 0.10, 0.05, 0.05])
        assert router.current_regime == "BULL"

        # numpy array
        arr = np.array([0.10, 0.80, 0.10], dtype=np.float64)
        router.set_regime_soft_probabilities(arr)
        assert router.current_regime == "BEAR"

        # string floats
        router.set_regime_soft_probabilities(["0.20", "0.20", "0.60"])
        assert router.current_regime == "CHOP"

        # under-length vector (< 3 elements): ignored, state preserved
        router.set_regime_soft_probabilities([0.99, 0.01])
        assert router.current_regime == "CHOP"
        router.set_regime_soft_probabilities([])
        assert router.current_regime == "CHOP"


# ==============================================================================
# 5. RECURRENT STATE FREEZING & EXCEPTION ISOLATION UNDER STRESS
# ==============================================================================

class TestAdversarialMemoryAndExceptionIsolation:
    """Stress tests verifying zero bit drift in inactive specialist memory and fault-tolerant context sync."""

    def test_2000_step_random_walk_verifies_bit_level_memory_freezing(self) -> None:
        """2000 continuous random inference steps: inactive policy memory experiences 0 bit drift."""
        bull = AdversarialMockPolicy("bull")
        bear = AdversarialMockPolicy("bear")
        router = DualChampionRouter(bull_policy=bull, bear_policy=bear, confirmation_bars=2)

        dummy_obs = np.zeros(137, dtype=np.float32)
        np.random.seed(888)

        last_bull_h = bull.hidden_state.copy()
        last_bull_c = bull.cell_state.copy()
        last_bull_steps = bull.step_count

        last_bear_h = bear.hidden_state.copy()
        last_bear_c = bear.cell_state.copy()
        last_bear_steps = bear.step_count

        for step in range(2000):
            raw_p = np.random.dirichlet([1.0, 1.0, 1.0])
            router.set_regime_soft_probabilities(raw_p)
            router.act_nam_tac_nhan(dummy_obs)

            if router.active_policy is bull:
                # Bull active
                assert bull.step_count == last_bull_steps + 1
                assert not np.array_equal(bull.hidden_state, last_bull_h)
                last_bull_steps = bull.step_count
                last_bull_h = bull.hidden_state.copy()
                last_bull_c = bull.cell_state.copy()

                # Bear INACTIVE: MUST BE STRICTLY FROZEN
                assert bear.step_count == last_bear_steps, f"Bear leaked step at step {step}"
                assert np.array_equal(bear.hidden_state, last_bear_h), f"Bear hidden state mutated at step {step}"
                assert np.array_equal(bear.cell_state, last_bear_c), f"Bear cell state mutated at step {step}"
            else:
                # Bear active
                assert bear.step_count == last_bear_steps + 1
                assert not np.array_equal(bear.hidden_state, last_bear_h)
                last_bear_steps = bear.step_count
                last_bear_h = bear.hidden_state.copy()
                last_bear_c = bear.cell_state.copy()

                # Bull INACTIVE: MUST BE STRICTLY FROZEN
                assert bull.step_count == last_bull_steps, f"Bull leaked step at step {step}"
                assert np.array_equal(bull.hidden_state, last_bull_h), f"Bull hidden state mutated at step {step}"
                assert np.array_equal(bull.cell_state, last_bull_c), f"Bull cell state mutated at step {step}"

        # Both policies were stepped and verified
        assert bull.step_count > 100
        assert bear.step_count > 1000

    def test_injected_exception_fault_isolation(self) -> None:
        """Injected exceptions in one specialist during context sync do not disrupt the other."""
        bull = MagicMock()
        bear = MagicMock()
        bull.set_account_context.side_effect = ZeroDivisionError("Simulated Bull crash")
        bull.set_decision_context.side_effect = KeyError("Simulated Bull key error")
        bull.set_position_context.side_effect = ValueError("Simulated Bull value error")

        router = DualChampionRouter(bull_policy=bull, bear_policy=bear)

        router.set_account_context({"balance": 20000.0}, positions_pipe="BTC_TEST")
        bear.set_account_context.assert_called_once_with({"balance": 20000.0}, "BTC_TEST")

        router.set_decision_context(frame="FR_100")
        bear.set_decision_context.assert_called_once_with(frame="FR_100")

        router.set_position_context("pos_test")
        bear.set_position_context.assert_called_once_with("pos_test")
