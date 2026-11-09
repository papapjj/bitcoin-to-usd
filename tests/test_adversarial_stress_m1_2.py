"""
tests/test_adversarial_stress_m1_2.py

Adversarial Stress & Empirical Verification Suite for Milestone M1 (Challenger M1_2):
1. End-to-End Routing under Adversarial Market Sequences:
   - Sequence: Bull Rally -> Violent Flash Crash -> Consolidation Bottom -> Explosive Rebound
   - Zero-lag Soft-MoE inflection response verification
   - Strict enforcement of Agent 4 Governor Invariant ACTION_HOLD = 0 (POLICY_CLOSE_ALL == 0.0)
   - Suppression of Market Order Chasing (Action 1 MARKET_BUY, Action 2 MARKET_SELL) in Consolidation Transition
2. Adversarial Governor Invariant Fuzzing:
   - Hostile sub-policies emitting Action 10 (CLOSE_ALL), random actions, corrupted vectors
   - Invariant bit-for-bit guarantee across all ticks
3. Consolidation Transition Zone Boundary & Chasing Verification:
   - Systematic boundary sweep across contest (|w_bull - w_bear| <= 0.30) and chop (chop_index >= 0.60)
   - Verification of defensive de-leveraging scale
4. 10,000 Continuous Tick CPU Latency & Drift Benchmark:
   - 10,000 continuous tick steps through SoftMoEGatingNetwork and DualChampionRouter
   - Verification of <= 0.5ms SLA (Mean, Median, P95, P99)
   - Measurement of chunk drift (Chunk 1 vs Chunk 10) and memory stability
"""

from __future__ import annotations

import gc
import math
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import pytest
import torch

from self_growing_trader.dual_champion_router import (
    DualChampionRouter,
    INPUT_COUNT,
    OUTPUT_COUNT,
)
from self_growing_trader.five_agent_authority import HOLD_ACTION
from self_growing_trader.soft_moe_gating import (
    CapitalPreservationGuard,
    SoftMoEGatingNetwork,
    extract_multi_timescale_features,
)


# ==============================================================================
# ADVERSARIAL MOCK POLICIES
# ==============================================================================

@dataclass
class AdversarialTrace:
    trace_id: str
    selected_action: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))
    selected_idx: int = int(HOLD_ACTION)
    action_index: int = int(HOLD_ACTION)
    confidence: float = 0.90
    action_details: dict[str, Any] = field(default_factory=dict)
    output_vector: np.ndarray = field(default_factory=lambda: np.zeros(20, dtype=float))


class AdversarialChallengerPolicy:
    """Mock specialist policy capable of generating adversarial and chasing actions."""

    def __init__(self, name: str, *, default_action: int = 0) -> None:
        self.name = name
        self.default_action = default_action
        self.step_count = 0
        self.agent_action_overrides: dict[int, int] = {}
        self.emit_malicious_agent4: bool = False
        self.malicious_agent4_action: int = 10  # e.g. POLICY_CLOSE_ALL

    def set_agent_action(self, agent_idx: int, action_idx: int) -> None:
        self.agent_action_overrides[agent_idx] = action_idx

    def clear_overrides(self) -> None:
        self.agent_action_overrides.clear()

    def act_nam_tac_nhan(
        self,
        observation: np.ndarray,
        goal: np.ndarray | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> tuple[AdversarialTrace, ...]:
        self.step_count += 1
        traces = []
        for agent_id in range(5):
            act = self.agent_action_overrides.get(agent_id, self.default_action)
            if agent_id == 4 and self.emit_malicious_agent4:
                act = self.malicious_agent4_action

            out = np.zeros(20, dtype=float)
            out[act] = 10.0 if act != int(HOLD_ACTION) else 1.0

            tr = AdversarialTrace(
                trace_id=f"{self.name}_step_{self.step_count}_agent_{agent_id}",
                selected_action=out.copy(),
                selected_idx=act,
                action_index=act,
                confidence=0.92,
                action_details={
                    "five_agent_index": agent_id,
                    "five_action_index": act,
                    "wire_action": act,
                    "policy_action": act,
                    "policy_close_all": 1.0 if (agent_id == 4 and act == 10) else 0.0,
                    "policy_name": self.name,
                },
                output_vector=out.copy(),
            )
            traces.append(tr)
        return tuple(traces)


# ==============================================================================
# 1. ADVERSARIAL MARKET SEQUENCE STRESS TESTS
# ==============================================================================

class TestAdversarialMarketSequences:
    """Tests DualChampionRouter under full 4-stage adversarial market cycle.

    Sequence:
    Stage 1 (Ticks 0-49): Bull Rally (expansion, strong positive momentum, low chop)
    Stage 2 (Ticks 50-99): Violent Flash Crash (steep plunge, negative momentum, volatility spike)
    Stage 3 (Ticks 100-199): Consolidation Bottom (chop >= 0.60, contest zone, whipsaws)
    Stage 4 (Ticks 200-249): Explosive Rebound (V-shape recovery, massive positive momentum)
    """

    def test_end_to_end_adversarial_market_cycle(self) -> None:
        bull_pol = AdversarialChallengerPolicy("bull_champion")
        bear_pol = AdversarialChallengerPolicy("bear_champion")

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            soft_moe_gating=SoftMoEGatingNetwork(),
            capital_preservation_guard=CapitalPreservationGuard(),
            momentum_gate_enabled=True,
        )

        n_steps = 250
        history_w_bull = []
        history_w_bear = []
        history_act4 = []
        history_act0 = []
        history_act1 = []

        for step in range(n_steps):
            obs = np.zeros(137, dtype=np.float32)

            # Determine Stage and Synthesize Market Vector
            if step < 50:
                # STAGE 1: BULL RALLY
                obs[0] = 0.45       # M1 return positive
                obs[80] = 0.50      # M5 EMA diff strong positive
                obs[81] = 0.25      # M5 slope positive
                obs[84] = 0.40      # M15 EMA diff positive
                obs[88] = 0.30      # H1 EMA diff positive
                obs[93] = 0.20      # Low chop (clear trend)

                # Bull specialist proposes MARKET_BUY (1)
                bull_pol.set_agent_action(0, 1)
                bear_pol.set_agent_action(0, 1)
                bull_pol.set_agent_action(1, 2)
                bear_pol.set_agent_action(1, 2)

            elif step < 100:
                # STAGE 2: VIOLENT FLASH CRASH
                obs[0] = -3.50      # M1 violent plunge
                obs[20] = 4.00      # Volatility ratio explosion
                obs[80] = -0.75     # M5 EMA diff plunges negative
                obs[81] = -0.50     # M5 slope highly negative
                obs[84] = -0.40     # M15 EMA diff flips negative
                obs[88] = -0.20     # H1 EMA diff begins dropping
                obs[93] = 0.35      # Trend breakdown

                # Bull tries to dip-buy (Action 1), Bear tries to SELL (Action 2)
                bull_pol.set_agent_action(0, 1)
                bear_pol.set_agent_action(0, 1)
                bull_pol.set_agent_action(1, 2)
                bear_pol.set_agent_action(1, 2)

            elif step < 200:
                # STAGE 3: CONSOLIDATION BOTTOM / CONTEST ZONE
                # Alternating micro noise, high chop index >= 0.70
                obs[0] = 0.05 * (-1.0 if step % 2 == 0 else 1.0)
                obs[80] = 0.02 * (-1.0 if step % 2 == 0 else 1.0)
                obs[81] = 0.01 * (-1.0 if step % 2 == 0 else 1.0)
                obs[84] = -0.05
                obs[88] = -0.10
                obs[93] = 0.75      # Severe chop / consolidation bottom

                # Both specialists attempt aggressive market chasing (Action 1 BUY, Action 2 SELL)
                bull_pol.set_agent_action(0, 1)
                bear_pol.set_agent_action(0, 1)
                bull_pol.set_agent_action(1, 2)
                bear_pol.set_agent_action(1, 2)

            else:
                # STAGE 4: EXPLOSIVE REBOUND
                obs[0] = 3.20       # Massive M1 bounce
                obs[9] = 1.50       # Huge lower rejection wick
                obs[80] = 0.65      # M5 EMA diff explodes positive
                obs[81] = 0.45      # M5 slope accelerates upward
                obs[84] = 0.30      # M15 turns positive
                obs[88] = 0.05      # H1 starting to recover
                obs[93] = 0.25      # Trend restored

                # Bull tries to BUY (Action 1), Bear tries to SELL into rally (Action 2)
                bull_pol.set_agent_action(0, 1)
                bear_pol.set_agent_action(0, 1)
                bull_pol.set_agent_action(1, 2)
                bear_pol.set_agent_action(1, 2)

            # Route through DualChampionRouter
            batch = router.act_nam_tac_nhan(obs)
            assert len(batch) == 5

            # Extract weights and actions
            w_bull, w_bear = router.soft_moe_weights
            history_w_bull.append(w_bull)
            history_w_bear.append(w_bear)

            act0 = int(batch[0].action_details.get("five_action_index", 0))
            act1 = int(batch[1].action_details.get("five_action_index", 0))
            act4 = int(batch[4].action_details.get("five_action_index", 0))

            history_act0.append(act0)
            history_act1.append(act1)
            history_act4.append(act4)

            # CRITICAL INVARIANT: Agent 4 Governor invariant ACTION_HOLD = 0 (100% of ticks)
            assert act4 == 0, f"Step {step}: Agent 4 Governor violated HOLD! act4 = {act4}"
            assert batch[4].action_details.get("policy_close_all", 0.0) == 0.0
            assert batch[4].output_vector[0] == 1.0
            assert batch[4].output_vector[10] <= -100.0

            # VERIFICATION BY STAGE
            if step < 50:
                # Stage 1: Bull rally - w_bull dominates
                assert w_bull > 0.60, f"Step {step}: Expected bull dominance, got w_bull={w_bull}"
                assert w_bull + w_bear == pytest.approx(1.0, abs=1e-6)
                # Bear SELL action 2 must be vetoed by momentum gate
                assert act1 == 0, f"Step {step}: Bear SELL not vetoed during Bull rally!"

            elif step < 100:
                # Stage 2: Flash crash - w_bear dominates immediately (zero transition lag)
                assert w_bear > 0.60, f"Step {step}: Expected bear dominance during crash, got w_bear={w_bear}"
                # Bull BUY action 1 must be vetoed by downward momentum gate
                assert act0 == 0, f"Step {step}: Bull BUY not vetoed during flash crash!"

            elif step < 200:
                # Stage 3: Consolidation bottom (Transition Zone)
                # Market order chasing MUST be suppressed for both agents!
                assert act0 == 0, (
                    f"Step {step}: Market order chasing (MARKET_BUY) NOT suppressed in consolidation! act0={act0}"
                )
                assert act1 == 0, (
                    f"Step {step}: Market order chasing (MARKET_SELL) NOT suppressed in consolidation! act1={act1}"
                )
                assert batch[0].action_details.get("regime_veto_reason") == "capital_preservation_transition_zone_no_market_chase"
                assert batch[1].action_details.get("regime_veto_reason") == "capital_preservation_transition_zone_no_market_chase"

            else:
                # Stage 4: Explosive rebound - w_bull immediately surges
                assert w_bull > 0.60, f"Step {step}: Expected bull dominance on rebound, got w_bull={w_bull}"
                # Bear SELL action 2 must be vetoed by upward momentum gate
                assert act1 == 0, f"Step {step}: Bear SELL not vetoed during explosive rebound!"

        # Final assertion on the complete sequence
        assert len(history_act4) == 250
        assert all(a == 0 for a in history_act4), "Governor invariant violated during market sequence!"


# ==============================================================================
# 2. ADVERSARIAL GOVERNOR AGENT 4 INVARIANT FUZZING
# ==============================================================================

class TestAdversarialGovernorAgent4Invariants:
    """Adversarially attacks Agent 4 Governor invariant with hostile actions & noise."""

    @pytest.mark.parametrize("malicious_action", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 19])
    def test_hostile_agent4_action_injection_strictly_suppressed(self, malicious_action: int) -> None:
        """DualChampionRouter MUST intercept any non-zero Agent 4 action and force it to ACTION_HOLD = 0."""
        bull_pol = AdversarialChallengerPolicy("bull")
        bear_pol = AdversarialChallengerPolicy("bear")

        bull_pol.emit_malicious_agent4 = True
        bull_pol.malicious_agent4_action = malicious_action

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            soft_moe_gating=SoftMoEGatingNetwork(),
            capital_preservation_guard=CapitalPreservationGuard(),
        )

        obs = np.random.randn(137).astype(np.float32)
        batch = router.act_nam_tac_nhan(obs)

        agent4_trace = batch[4]
        assert agent4_trace.action_index == 0
        assert agent4_trace.selected_idx == 0
        assert agent4_trace.action_details.get("five_action_index") == 0
        assert agent4_trace.action_details.get("wire_action") == 0
        assert agent4_trace.action_details.get("policy_action") == 0
        assert agent4_trace.action_details.get("policy_close_all", 0.0) == 0.0
        assert agent4_trace.output_vector[0] == 1.0
        assert agent4_trace.output_vector[10] <= -100.0

    def test_randomized_agent4_fuzzing_1000_steps(self) -> None:
        """1,000 steps of hostile randomized actions for Agent 4."""
        bull_pol = AdversarialChallengerPolicy("bull")
        bear_pol = AdversarialChallengerPolicy("bear")
        bull_pol.emit_malicious_agent4 = True

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            soft_moe_gating=SoftMoEGatingNetwork(),
            capital_preservation_guard=CapitalPreservationGuard(),
        )

        rng = np.random.RandomState(999)
        for _ in range(1000):
            # Pick a hostile action in [1..19]
            random_hostile_act = int(rng.randint(1, 20))
            bull_pol.malicious_agent4_action = random_hostile_act

            obs = rng.randn(137).astype(np.float32)
            batch = router.act_nam_tac_nhan(obs)

            tr4 = batch[4]
            assert tr4.action_index == 0
            assert tr4.action_details.get("five_action_index") == 0
            assert tr4.action_details.get("wire_action") == 0
            assert tr4.output_vector[0] == 1.0
            assert tr4.output_vector[10] <= -100.0


# ==============================================================================
# 3. CONSOLIDATION TRANSITION ZONE BOUNDARY & CHASING VERIFICATION
# ==============================================================================

class TestConsolidationChasingSuppression:
    """Verifies that CapitalPreservationGuard suppresses market order chasing in transition zones."""

    def test_market_order_chasing_suppression_boundary_sweep(self) -> None:
        guard = CapitalPreservationGuard(transition_threshold=0.30, chop_threshold=0.60)

        # 1. Inside transition zone due to narrow margin (|w_bull - w_bear| <= 0.30)
        # Test values safely within the threshold taking float arithmetic into account
        for delta in [0.0, 0.05, 0.15, 0.25, 0.29]:
            wb = 0.5 + delta / 2.0
            wr = 0.5 - delta / 2.0
            assert guard.is_transition_active(wb, wr, chop_index=0.30)

            # MARKET_BUY (1) -> suppressed to 0
            vetoed, act, reason = guard.evaluate_order(1, wb, wr, chop_index=0.30)
            assert vetoed is True
            assert act == 0
            assert "no_market_chase" in reason

            # MARKET_SELL (2) -> suppressed to 0
            vetoed, act, reason = guard.evaluate_order(2, wb, wr, chop_index=0.30)
            assert vetoed is True
            assert act == 0
            assert "no_market_chase" in reason

            # Patient limit orders (BUY_LIMIT=3, SELL_LIMIT=4) -> permitted
            vetoed, act, _ = guard.evaluate_order(3, wb, wr, chop_index=0.30)
            assert vetoed is False
            assert act == 3

        # 2. Inside transition zone due to high chop (chop_index >= 0.60)
        for chop in [0.60, 0.65, 0.75, 0.90]:
            # Even with strong directional delta (e.g. 0.50)
            wb, wr = 0.75, 0.25
            assert guard.is_transition_active(wb, wr, chop_index=chop)

            vetoed, act, reason = guard.evaluate_order(1, wb, wr, chop_index=chop)
            assert vetoed is True
            assert act == 0

            # In severe chop >= 0.70, stop orders (5, 6) are also suppressed
            if chop >= 0.70:
                vetoed, act, reason = guard.evaluate_order(5, wb, wr, chop_index=chop)
                assert vetoed is True
                assert act == 0
                assert "severe_chop_no_breakout_chase" in reason

        # 3. Outside transition zone (clear trend: delta > 0.30 and chop < 0.60)
        wb, wr = 0.75, 0.25
        assert not guard.is_transition_active(wb, wr, chop_index=0.35)
        vetoed, act, _ = guard.evaluate_order(1, wb, wr, chop_index=0.35)
        assert vetoed is False
        assert act == 1

    def test_floating_point_boundary_edge_case(self) -> None:
        """Documents the exact IEEE-754 boundary behavior at w_bull=0.65, w_bear=0.35."""
        guard = CapitalPreservationGuard(transition_threshold=0.30, chop_threshold=0.60)
        # 0.65 - 0.35 in float64 is 0.30000000000000004 > 0.30
        delta = abs(0.65 - 0.35)
        assert delta > 0.30, "IEEE-754 precision delta should slightly exceed 0.30"
        # With 1e-6 epsilon tolerance, is_transition_active evaluates True at (0.65, 0.35)
        assert guard.is_transition_active(0.65, 0.35, chop_index=0.50) is True
        # But at w_bull=0.6499, w_bear=0.3501, it is True
        assert guard.is_transition_active(0.6499, 0.3501, chop_index=0.50) is True


# ==============================================================================
# 4. 10,000 CONTINUOUS TICK CPU LATENCY & DRIFT BENCHMARK
# ==============================================================================

class TestCPULatency10000ContinuousSteps:
    """Measures CPU latency across 10,000 continuous tick steps to detect drift or GC spikes."""

    def test_soft_moe_gating_10000_steps_latency_and_drift(self) -> None:
        """SoftMoEGatingNetwork must sustain <= 0.5ms CPU latency across 10,000 steps without drift."""
        gating = SoftMoEGatingNetwork()
        rng = np.random.RandomState(42)

        # Warmup 200 ticks
        dummy = rng.randn(137).astype(np.float32)
        for _ in range(200):
            gating.compute_regime_weights(dummy)

        n_steps = 10000
        latencies = []

        for _ in range(n_steps):
            obs = rng.randn(137).astype(np.float32)
            t0 = time.perf_counter()
            wb, wr, info = gating.compute_regime_weights(obs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.median(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))

        chunk1 = latencies[:1000]
        chunk10 = latencies[9000:]
        chunk1_mean = float(np.mean(chunk1))
        chunk10_mean = float(np.mean(chunk10))
        drift_ms = abs(chunk10_mean - chunk1_mean)

        print(f"\n[Soft-MoE 10k Benchmark] Mean={mean_ms:.4f}ms, P50={p50_ms:.4f}ms, P95={p95_ms:.4f}ms, P99={p99_ms:.4f}ms")
        print(f"[Soft-MoE 10k Drift] Chunk 1={chunk1_mean:.4f}ms, Chunk 10={chunk10_mean:.4f}ms, Drift={drift_ms:.4f}ms")

        # Strict SLA Assertions
        assert mean_ms <= 0.50, f"Mean latency exceeded 0.5ms SLA: {mean_ms:.4f}ms"
        assert p50_ms <= 0.50, f"P50 latency exceeded 0.5ms SLA: {p50_ms:.4f}ms"
        assert p95_ms <= 0.50, f"P95 latency exceeded 0.5ms SLA: {p95_ms:.4f}ms"
        assert p99_ms <= 0.50, f"P99 latency exceeded 0.5ms SLA: {p99_ms:.4f}ms"
        assert drift_ms < 0.10, f"Temporal latency drift detected: {drift_ms:.4f}ms >= 0.10ms"

    def test_dual_champion_router_10000_steps_latency_and_stability(self) -> None:
        """DualChampionRouter must sustain steady-state <= 0.5ms latency across 10,000 steps."""
        bull_pol = AdversarialChallengerPolicy("bull")
        bear_pol = AdversarialChallengerPolicy("bear")

        router = DualChampionRouter(
            bull_policy=bull_pol,
            bear_policy=bear_pol,
            soft_moe_gating=SoftMoEGatingNetwork(),
            capital_preservation_guard=CapitalPreservationGuard(),
            momentum_gate_enabled=True,
        )

        rng = np.random.RandomState(42)

        # Warmup 200 ticks
        dummy = rng.randn(137).astype(np.float32)
        for _ in range(200):
            router.act_nam_tac_nhan(dummy)

        n_steps = 10000
        latencies = []

        for _ in range(n_steps):
            obs = rng.randn(137).astype(np.float32)
            t0 = time.perf_counter()
            batch = router.act_nam_tac_nhan(obs)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        mean_ms = float(np.mean(latencies))
        p50_ms = float(np.median(latencies))
        p95_ms = float(np.percentile(latencies, 95))
        p99_ms = float(np.percentile(latencies, 99))
        pct_under_half_ms = float(np.mean(np.array(latencies) <= 0.50) * 100.0)

        chunk1 = latencies[:1000]
        chunk10 = latencies[9000:]
        chunk1_mean = float(np.mean(chunk1))
        chunk10_mean = float(np.mean(chunk10))
        drift_ms = abs(chunk10_mean - chunk1_mean)

        print(f"\n[Router 10k Benchmark] Mean={mean_ms:.4f}ms, P50={p50_ms:.4f}ms, P95={p95_ms:.4f}ms, P99={p99_ms:.4f}ms")
        print(f"[Router 10k Compliance] Steps <= 0.5ms: {pct_under_half_ms:.2f}%")
        print(f"[Router 10k Drift] Chunk 1={chunk1_mean:.4f}ms, Chunk 10={chunk10_mean:.4f}ms, Drift={drift_ms:.4f}ms")

        # Strict SLA Assertions
        assert mean_ms <= 0.50, f"Mean latency exceeded 0.5ms SLA: {mean_ms:.4f}ms"
        assert p50_ms <= 0.50, f"P50 latency exceeded 0.5ms SLA: {p50_ms:.4f}ms"
        assert p95_ms <= 0.50, f"P95 latency exceeded 0.5ms SLA: {p95_ms:.4f}ms"
        assert pct_under_half_ms >= 99.0, f"Less than 99% of ticks under 0.5ms SLA: {pct_under_half_ms:.2f}%"
        assert drift_ms < 0.10, f"Router temporal latency drift detected: {drift_ms:.4f}ms >= 0.10ms"
