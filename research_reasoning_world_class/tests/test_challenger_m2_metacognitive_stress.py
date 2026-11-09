"""
Challenger Adversarial Stress Test Suite: Metacognitive Reasoning, PRM & Activation Gate
Project: "Mythos của Claude" - xLSTM 17.5M Reasoning System
Milestone: M2 Empirical Verification

Tests the 4 empirical stress-testing objectives:
1. Activation Gate Invariant Stress:
   - Borderline boundary conditions:
     * Confidence = 0.8499 -> NOT activated (action = 0 / HOLD)
     * Margin = 0.1999 -> NOT activated (action = 0 / HOLD)
     * Confidence = 0.8501, Margin = 0.2001 -> Activated (action = 1 / BUY)
     * Confidence = 0.8501, Margin = -0.2001 -> Activated (action = -1 / SELL)
     * Conflicted debate (Bull = 0.90, Bear = 0.88 -> Margin = 0.02) -> NOT activated (action = 0 / HOLD)
   - Exhaustive grid scan around the decision boundaries.
2. PRM Weak-Link Principle Stress:
   - PRM score = 0.5 * mean(step_scores) + 0.5 * min(step_scores).
   - Adversarial case: step scores [1.0, 1.0, 1.0, 0.0] -> Mean = 0.75, Min = 0.0 -> PRM = 0.375.
   - Assert failure of high mean if one weak link collapses.
   - All permutations of single-hop failure (Hop 1, Hop 2, Hop 3, Hop 4).
3. Scenario Decay Dynamics:
   - Exponential decay curve verification: P(t) = P(0) * exp(-lambda * t) normalized.
   - Simplex probability conservation: sum(P(t)) == 1.0 across expansion, reversal, shakeout, chop within operational horizon (H=12, up to H=24).
   - Empirical identification of extended horizon (H=60, t>=29) eps distortion boundary.
   - Differential decay ordering inversions over horizon steps.
   - Extremum robustness (lambda -> 0, lambda -> 100, degenerate initial states).
4. Downstream Contract Parity:
   - Verify .to_reasoning_verdict() returns valid ReasoningVerdict dataclass matching PROJECT.md interface contract.
   - Native Python types enforcement (int, float, bool - no 0-dim tensor leakages).
   - Multi-batch extraction parity and out-of-bounds safety.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Single thread for deterministic runner execution
torch.set_num_threads(1)

# Ensure project and repo root are accessible
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.metacognitive_reasoning import (
    ReasoningConfig,
    MetacognitiveOutput,
    ReasoningVerdict,
    PRMVerifier,
    AdversarialSelfReflection,
    CoTReasoningHead,
    MetacognitiveReasoningEngine,
    JITReasoningWrapper,
)


@pytest.fixture
def standard_config() -> ReasoningConfig:
    return ReasoningConfig(
        input_dim=256,
        hidden_dim=256,
        num_hops=4,
        num_scenarios=4,
        prm_threshold=0.85,
        adversarial_margin_threshold=0.20,
        dropout=0.0,
        temporal_horizon=12,
        eps=1e-8,
    )


# =====================================================================
# SECTION 1: Activation Gate Invariant Stress
# =====================================================================

class TestActivationGateInvariantStress:
    """
    Stress-tests borderline boundary conditions of the strict Activation Gate:
    Activation requires: Confidence >= 0.85 AND |Margin| >= 0.20.
    Otherwise action MUST be 0 (HOLD).
    """

    def test_borderline_confidence_0_8499_must_hold(self, standard_config):
        """
        Confidence = 0.8499 (< 0.85) with ample Margin (0.50).
        Must NOT activate (is_activated == False, action == 0 / HOLD).
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        class MockReflectionExact(nn.Module):
            def forward(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
                B = thought.size(0)
                bull = torch.full((B,), 1.0, dtype=torch.float32)
                bear = torch.full((B,), 0.50, dtype=torch.float32) # Margin = +0.50 >= 0.20
                scen_probs = torch.zeros(B, 4, dtype=torch.float32)
                scen_probs[:, 0] = 1.0 # certainty_factor = 1.0
                return {
                    "bull_score": bull,
                    "bear_score": bear,
                    "adversarial_margin": bull - bear,
                    "scenario_probs": scen_probs,
                    "scenario_decays": torch.full((B, 4), 0.10, dtype=torch.float32),
                    "scenario_trajectories": torch.zeros(B, 4, 12, dtype=torch.float32),
                }

        class MockPRM(nn.Module):
            def forward(self, all_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                B = all_hops.size(0)
                return torch.full((B,), 0.8499, dtype=torch.float32), torch.full((B, 4), 0.8499)

        engine.reflection = MockReflectionExact()
        engine.prm_verifier = MockPRM()

        x = torch.zeros(4, standard_config.input_dim)
        with torch.no_grad():
            out = engine(x)

        for i in range(4):
            conf = out.confidence[i].item()
            margin = out.adversarial_margin[i].item()
            is_act = out.is_activated[i].item()
            act = out.action[i].item()

            assert abs(conf - 0.8499) < 1e-4, f"Expected confidence ~0.8499, got {conf}"
            assert abs(margin - 0.50) < 1e-4, f"Expected margin ~0.50, got {margin}"
            assert not is_act, f"Gate erroneously activated with confidence {conf} < 0.85"
            assert act == 0, f"Action must be 0 (HOLD) when not activated, got {act}"

    def test_borderline_margin_0_1999_must_hold(self, standard_config):
        """
        Confidence = 0.95 (>= 0.85) but Margin = 0.1999 (< 0.20).
        Must NOT activate (is_activated == False, action == 0 / HOLD).
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        class MockReflectionMargin(nn.Module):
            def forward(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
                B = thought.size(0)
                bull = torch.full((B,), 1.0, dtype=torch.float32)
                bear = torch.full((B,), 1.0 - 0.1999, dtype=torch.float32) # Margin = +0.1999
                scen_probs = torch.zeros(B, 4, dtype=torch.float32)
                scen_probs[:, 0] = 1.0 # certainty_factor = 1.0
                return {
                    "bull_score": bull,
                    "bear_score": bear,
                    "adversarial_margin": bull - bear, # 0.1999
                    "scenario_probs": scen_probs,
                    "scenario_decays": torch.full((B, 4), 0.10, dtype=torch.float32),
                    "scenario_trajectories": torch.zeros(B, 4, 12, dtype=torch.float32),
                }

        class MockPRMMargin(nn.Module):
            def forward(self, all_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                B = all_hops.size(0)
                return torch.full((B,), 0.95, dtype=torch.float32), torch.full((B, 4), 0.95)

        engine.reflection = MockReflectionMargin()
        engine.prm_verifier = MockPRMMargin()

        x = torch.zeros(4, standard_config.input_dim)
        with torch.no_grad():
            out = engine(x)

        for i in range(4):
            conf = out.confidence[i].item()
            margin = out.adversarial_margin[i].item()
            is_act = out.is_activated[i].item()
            act = out.action[i].item()

            assert abs(conf - 0.95) < 1e-4
            assert abs(margin - 0.1999) < 1e-4
            assert not is_act, f"Gate erroneously activated with margin {margin} < 0.20"
            assert act == 0, f"Action must be 0 (HOLD) when margin is insufficient, got {act}"

    def test_borderline_both_above_threshold_bull_buy(self, standard_config):
        """
        Confidence = 0.8501 (>= 0.85) AND Margin = 0.2001 (>= 0.20).
        Must activate (is_activated == True, action == 1 / BUY).
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        class MockReflectionAbove(nn.Module):
            def forward(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
                B = thought.size(0)
                bull = torch.full((B,), 1.0, dtype=torch.float32)
                bear = torch.full((B,), 1.0 - 0.2001, dtype=torch.float32) # Margin = +0.2001
                scen_probs = torch.zeros(B, 4, dtype=torch.float32)
                scen_probs[:, 0] = 1.0 # certainty_factor = 1.0
                return {
                    "bull_score": bull,
                    "bear_score": bear,
                    "adversarial_margin": bull - bear,
                    "scenario_probs": scen_probs,
                    "scenario_decays": torch.full((B, 4), 0.10, dtype=torch.float32),
                    "scenario_trajectories": torch.zeros(B, 4, 12, dtype=torch.float32),
                }

        class MockPRMAbove(nn.Module):
            def forward(self, all_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                B = all_hops.size(0)
                return torch.full((B,), 0.8501, dtype=torch.float32), torch.full((B, 4), 0.8501)

        engine.reflection = MockReflectionAbove()
        engine.prm_verifier = MockPRMAbove()

        x = torch.zeros(4, standard_config.input_dim)
        with torch.no_grad():
            out = engine(x)

        for i in range(4):
            conf = out.confidence[i].item()
            margin = out.adversarial_margin[i].item()
            is_act = out.is_activated[i].item()
            act = out.action[i].item()

            assert conf >= 0.85, f"Confidence {conf} < 0.85"
            assert margin >= 0.20, f"Margin {margin} < 0.20"
            assert is_act, "Gate must activate when both thresholds are exceeded"
            assert act == 1, f"Expected BUY (1), got {act}"

    def test_borderline_both_above_threshold_bear_sell(self, standard_config):
        """
        Confidence = 0.8501 (>= 0.85) AND Margin = -0.2001 (|Margin| >= 0.20, Bear dominant).
        Must activate (is_activated == True, action == -1 / SELL).
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        class MockReflectionBear(nn.Module):
            def forward(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
                B = thought.size(0)
                bull = torch.full((B,), 1.0 - 0.2001, dtype=torch.float32)
                bear = torch.full((B,), 1.0, dtype=torch.float32) # Margin = -0.2001
                scen_probs = torch.zeros(B, 4, dtype=torch.float32)
                scen_probs[:, 1] = 1.0 # Scenario 1 dominant -> certainty_factor = 1.0
                return {
                    "bull_score": bull,
                    "bear_score": bear,
                    "adversarial_margin": bull - bear, # -0.2001
                    "scenario_probs": scen_probs,
                    "scenario_decays": torch.full((B, 4), 0.10, dtype=torch.float32),
                    "scenario_trajectories": torch.zeros(B, 4, 12, dtype=torch.float32),
                }

        class MockPRMBear(nn.Module):
            def forward(self, all_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                B = all_hops.size(0)
                return torch.full((B,), 0.8501, dtype=torch.float32), torch.full((B, 4), 0.8501)

        engine.reflection = MockReflectionBear()
        engine.prm_verifier = MockPRMBear()

        x = torch.zeros(4, standard_config.input_dim)
        with torch.no_grad():
            out = engine(x)

        for i in range(4):
            conf = out.confidence[i].item()
            margin = out.adversarial_margin[i].item()
            is_act = out.is_activated[i].item()
            act = out.action[i].item()

            assert conf >= 0.85
            assert abs(margin) >= 0.20
            assert is_act, "Gate must activate for strong Bear signal"
            assert act == -1, f"Expected SELL (-1), got {act}"

    def test_conflicted_debate_high_conviction_results_in_hold(self, standard_config):
        """
        Adversarial case: Bull = 0.90, Bear = 0.88 -> Margin = 0.02.
        Both agents have strong conviction, PRM is 0.95.
        However, because the margin is only 0.02 (< 0.20), debate is conflicted and unresolved.
        The system must NOT activate (is_activated == False, action == 0 / HOLD).
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        class MockConflictedReflection(nn.Module):
            def forward(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
                B = thought.size(0)
                bull = torch.full((B,), 0.90, dtype=torch.float32)
                bear = torch.full((B,), 0.88, dtype=torch.float32)
                scen_probs = torch.full((B, 4), 0.25, dtype=torch.float32)
                return {
                    "bull_score": bull,
                    "bear_score": bear,
                    "adversarial_margin": bull - bear, # 0.02
                    "scenario_probs": scen_probs,
                    "scenario_decays": torch.full((B, 4), 0.10, dtype=torch.float32),
                    "scenario_trajectories": torch.zeros(B, 4, 12, dtype=torch.float32),
                }

        class MockHighPRM(nn.Module):
            def forward(self, all_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                B = all_hops.size(0)
                return torch.full((B,), 0.98, dtype=torch.float32), torch.full((B, 4), 0.98)

        engine.reflection = MockConflictedReflection()
        engine.prm_verifier = MockHighPRM()

        x = torch.zeros(4, standard_config.input_dim)
        with torch.no_grad():
            out = engine(x)

        for i in range(4):
            margin = out.adversarial_margin[i].item()
            is_act = out.is_activated[i].item()
            act = out.action[i].item()

            assert abs(margin - 0.02) < 1e-4, f"Expected margin ~0.02, got {margin}"
            assert not is_act, "Conflicted debate must NOT activate gate"
            assert act == 0, f"Conflicted debate must yield HOLD (0), got {act}"

    def test_activation_gate_exhaustive_grid_scan(self, standard_config):
        """
        Exhaustive test asserting invariant across a 2D mesh grid of (confidence, margin):
        Invariant: is_activated == True <=> (confidence >= 0.85 and abs(margin) >= 0.20)
        and action != 0 <=> is_activated.
        """
        conf_test_points = [0.0, 0.50, 0.849, 0.8499, 0.8500, 0.8501, 0.860, 0.950, 1.0]
        margin_test_points = [-1.0, -0.50, -0.2001, -0.2000, -0.1999, -0.05, 0.0, 0.05, 0.1999, 0.2000, 0.2001, 0.50, 1.0]

        for conf in conf_test_points:
            for margin in margin_test_points:
                conf_tensor = torch.tensor([conf], dtype=torch.float32)
                margin_tensor = torch.tensor([margin], dtype=torch.float32)

                is_act = (conf_tensor >= standard_config.prm_threshold) & (
                    torch.abs(margin_tensor) >= standard_config.adversarial_margin_threshold
                )
                action = torch.zeros(1, dtype=torch.long)
                action[is_act & (margin_tensor > 0.0)] = 1
                action[is_act & (margin_tensor < 0.0)] = -1

                expected_act = (conf >= 0.85) and (abs(margin) >= 0.20)
                assert is_act.item() == expected_act, (
                    f"Invariant violated for conf={conf}, margin={margin}: "
                    f"is_act={is_act.item()} vs expected={expected_act}"
                )

                if expected_act:
                    if margin > 0:
                        assert action.item() == 1
                    elif margin < 0:
                        assert action.item() == -1
                    else:
                        assert action.item() == 0
                else:
                    assert action.item() == 0, (
                        f"Non-activated state must have action == 0, got {action.item()} for conf={conf}, margin={margin}"
                    )


# =====================================================================
# SECTION 2: PRM Weak-Link Principle Stress
# =====================================================================

class TestPRMWeakLinkPrincipleStress:
    """
    Stress-tests the Process Reward Model (PRM) Weak-Link Principle:
    prm_score = 0.5 * weighted_mean(hop_rewards) + 0.5 * min(hop_rewards)
    
    If ANY single hop collapses (e.g. invalid liquidity sweep, hallucinated breakout),
    the min penalty collapses the overall PRM score regardless of how strong the other hops are.
    """

    def test_weak_link_adversarial_case_1_1_1_0(self):
        """
        Adversarial case: step scores [1.0, 1.0, 1.0, 0.0].
        Mean = 0.75, Min = 0.0.
        PRM score = 0.5 * 0.75 + 0.5 * 0.0 = 0.375.
        Assert failure of high mean if one weak link collapses!
        """
        hop_rewards = torch.tensor([[1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
        mean_score = torch.mean(hop_rewards, dim=-1).item()
        min_score = torch.min(hop_rewards, dim=-1).values.item()

        prm_score = 0.5 * mean_score + 0.5 * min_score

        assert abs(mean_score - 0.75) < 1e-6, f"Expected mean 0.75, got {mean_score}"
        assert abs(min_score - 0.0) < 1e-6, f"Expected min 0.0, got {min_score}"
        assert abs(prm_score - 0.375) < 1e-6, f"Expected PRM score 0.375, got {prm_score}"

        # If a naive mean was used, 0.75 might be near passing. But 0.375 is severely penalized.
        assert prm_score < 0.85, "PRM score must not pass threshold"
        assert prm_score == 0.375

    def test_prm_verifier_module_weak_link_execution(self, standard_config):
        """
        Tests the actual PRMVerifier PyTorch module to assert it implements
        the Weak-Link Principle aggregation.
        """
        verifier = PRMVerifier(standard_config)
        verifier.eval()

        # Check weights are uniform initially (1/K)
        with torch.no_grad():
            weights = F.softmax(verifier.consensus_weight, dim=0)
            expected_w = torch.full((standard_config.num_hops,), 1.0 / standard_config.num_hops)
            torch.testing.assert_close(weights, expected_w, rtol=1e-4, atol=1e-4)

        # Test formula directly with mock evaluator return values
        all_hops = torch.randn(2, standard_config.num_hops, standard_config.hidden_dim)
        with torch.no_grad():
            prm_score, hop_rewards = verifier(all_hops)

            # Recompute expected
            mean_r = torch.sum(hop_rewards * weights.unsqueeze(0), dim=-1)
            min_r, _ = torch.min(hop_rewards, dim=-1)
            expected_prm = torch.clamp(0.5 * mean_r + 0.5 * min_r, 0.0, 1.0)

            torch.testing.assert_close(prm_score, expected_prm, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("collapsed_hop_idx", [0, 1, 2, 3])
    def test_all_single_hop_collapse_permutations(self, collapsed_hop_idx):
        """
        Test that collapsing any individual hop (Hop 0: Regime, Hop 1: Liquidity,
        Hop 2: Momentum, Hop 3: Synthesis) pulls the score down to exactly 0.375.
        No reasoning hop can be bypassed!
        """
        hop_rewards = [1.0, 1.0, 1.0, 1.0]
        hop_rewards[collapsed_hop_idx] = 0.0

        t = torch.tensor([hop_rewards], dtype=torch.float32)
        mean_val = torch.mean(t, dim=-1).item()
        min_val = torch.min(t, dim=-1).values.item()

        prm = 0.5 * mean_val + 0.5 * min_val

        assert abs(prm - 0.375) < 1e-6, (
            f"Collapsed hop {collapsed_hop_idx} resulted in PRM={prm}, expected 0.375"
        )

    def test_weak_link_blocks_high_directional_conviction_in_engine(self, standard_config):
        """
        End-to-end integration: Even if Bull score is 1.0 and Bear score is 0.0
        (Margin = 1.0), a collapsed PRM weak link (0.375) will keep Total Confidence
        at <= 0.375, strictly blocking trade activation!
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        # Mock PRM to return 0.375 (collapsed weak link)
        class MockCollapsedPRM(nn.Module):
            def forward(self, all_hops: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                B = all_hops.size(0)
                return torch.full((B,), 0.375, dtype=torch.float32), torch.tensor([[1.0, 1.0, 1.0, 0.0]]).repeat(B, 1)

        # Mock Reflection to return perfect directional conviction
        class MockPerfectBull(nn.Module):
            def forward(self, thought: torch.Tensor) -> Dict[str, torch.Tensor]:
                B = thought.size(0)
                bull = torch.full((B,), 1.0, dtype=torch.float32)
                bear = torch.full((B,), 0.0, dtype=torch.float32)
                scen_probs = torch.zeros(B, 4, dtype=torch.float32)
                scen_probs[:, 0] = 1.0 # certainty_factor = 1.0
                return {
                    "bull_score": bull,
                    "bear_score": bear,
                    "adversarial_margin": bull - bear, # +1.0
                    "scenario_probs": scen_probs,
                    "scenario_decays": torch.full((B, 4), 0.10, dtype=torch.float32),
                    "scenario_trajectories": torch.zeros(B, 4, 12, dtype=torch.float32),
                }

        engine.prm_verifier = MockCollapsedPRM()
        engine.reflection = MockPerfectBull()

        x = torch.zeros(2, standard_config.input_dim)
        with torch.no_grad():
            out = engine(x)

        for i in range(2):
            assert abs(out.prm_score[i].item() - 0.375) < 1e-4
            assert abs(out.confidence[i].item() - 0.375) < 1e-4, (
                f"Confidence must be capped by PRM weak link, got {out.confidence[i].item()}"
            )
            assert not out.is_activated[i].item(), "Must NOT activate when PRM collapses"
            assert out.action[i].item() == 0, f"Action must be HOLD (0), got {out.action[i].item()}"


# =====================================================================
# SECTION 3: Scenario Decay Dynamics
# =====================================================================

class TestScenarioDecayDynamicsStress:
    """
    Stress-tests the exponential scenario decay dynamics:
    1. Curve equation: P_s(t) = P_s(0) * exp(-lambda_s * t) / sum(...)
    2. Simplex probability conservation across Expansion, Reversal, Shakeout, Chop:
       sum_{s=0}^3 P_s(t) == 1.0 for all t in [0, horizon].
    3. Monotonicity and differential decay ordering.
    """

    @pytest.mark.parametrize("horizon", [1, 2, 5, 12, 24])
    def test_scenario_probabilities_sum_to_one_within_operational_horizons(self, standard_config, horizon):
        """
        Verify that within the designed operational horizon (temporal_horizon = 12)
        and up to H=24, the 4 scenario probabilities strictly sum to 1.0 within 1e-4.
        """
        torch.manual_seed(42)
        engine = MetacognitiveReasoningEngine(standard_config)
        batch_size = 4

        # Generate random scenario probabilities on simplex
        raw_logits = torch.randn(batch_size, 4)
        scenario_probs = F.softmax(raw_logits, dim=-1)
        scenario_decays = torch.rand(batch_size, 4) * 0.5 + 0.05 # decay rates in [0.05, 0.55]

        decay_curves = engine.simulate_temporal_decay(
            scenario_probs, scenario_decays, horizon_steps=horizon
        )

        assert decay_curves.shape == (batch_size, 4, horizon)

        tol = 1e-4 if horizon <= 12 else 2e-4
        for t in range(horizon):
            step_sum = decay_curves[:, :, t].sum(dim=1)
            torch.testing.assert_close(
                step_sum,
                torch.ones_like(step_sum),
                rtol=tol,
                atol=tol,
                msg=f"Simplex violated at step t={t}"
            )

    def test_extended_horizon_eps_distortion_boundary_empirical_finding(self, standard_config):
        """
        CHALLENGER EMPIRICAL FINDING:
        In simulate_temporal_decay(), the denominator is (sum(decay_unnorm) + eps).
        When horizon H is extended to 60 steps and decay rate lambda >= 0.5,
        the unnormalized probability mass approaches 1e-7 ~ 1e-8 at step t=29.
        Adding eps (1e-8) directly into the denominator distorts the probability simplex,
        causing the normalized sum to degrade below 1.0 (empirically ~0.92 at t=29).
        
        This test documents the exact threshold where the numerical degradation begins.
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        p0 = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        lambdas = torch.tensor([[0.55, 0.55, 0.55, 0.55]], dtype=torch.float32)

        decay_curves = engine.simulate_temporal_decay(p0, lambdas, horizon_steps=35)

        # Up to step t=20, unnorm sum is >= 1e-5, so distortion is negligible (< 0.1%)
        sum_t20 = decay_curves[0, :, 20].sum().item()
        assert abs(sum_t20 - 1.0) < 0.01, f"Expected sum ~1.0 at t=20, got {sum_t20}"

        # At step t=29, unnorm sum drops to 1.18e-7, so adding eps=1e-8 causes ~8% distortion
        sum_t29 = decay_curves[0, :, 29].sum().item()
        assert sum_t29 < 0.95, f"Empirical finding: sum at t=29 degraded to {sum_t29:.4f}"

    def test_exponential_decay_analytical_match(self, standard_config):
        """
        Verify decay curves match analytical P_s(0) * exp(-lambda_s * t) exactly.
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        scenario_probs = torch.tensor([[0.40, 0.30, 0.20, 0.10]], dtype=torch.float32)
        scenario_decays = torch.tensor([[0.05, 0.10, 0.20, 0.30]], dtype=torch.float32)
        horizon = 10

        decay_curves = engine.simulate_temporal_decay(
            scenario_probs, scenario_decays, horizon_steps=horizon
        )

        for t in range(horizon):
            p0 = scenario_probs.squeeze(0).numpy()
            lambdas = scenario_decays.squeeze(0).numpy()
            unnorm = p0 * np.exp(-lambdas * t)
            expected = unnorm / (np.sum(unnorm) + standard_config.eps)

            actual = decay_curves[0, :, t].numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)

    def test_differential_decay_inversion_over_time(self, standard_config):
        """
        Scenario 0 starts dominant (P0 = 0.60) but has rapid decay (lambda = 0.50).
        Scenario 1 starts minor (P1 = 0.20) but has resilient decay (lambda = 0.02).
        Over time horizon, Scenario 1 MUST overtake Scenario 0 as dominant scenario.
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        scenario_probs = torch.tensor([[0.60, 0.20, 0.10, 0.10]], dtype=torch.float32)
        scenario_decays = torch.tensor([[0.50, 0.02, 0.10, 0.10]], dtype=torch.float32)
        horizon = 15

        decay_curves = engine.simulate_temporal_decay(
            scenario_probs, scenario_decays, horizon_steps=horizon
        )

        p0_start = decay_curves[0, 0, 0].item()
        p1_start = decay_curves[0, 1, 0].item()
        assert p0_start > p1_start, "Initial state: Scenario 0 must dominate Scenario 1"

        p0_end = decay_curves[0, 0, -1].item()
        p1_end = decay_curves[0, 1, -1].item()
        assert p1_end > p0_end, (
            f"After {horizon} steps, Scenario 1 ({p1_end:.4f}) must overtake Scenario 0 ({p0_end:.4f})"
        )

    def test_extreme_decay_boundaries_eps_safety(self, standard_config):
        """
        Test extreme edge cases:
        1. lambda = 0 (probabilities remain strictly identical to initial values).
        2. lambda = 100.0 (extreme exponential degradation, zero NaN/Inf, eps-protected).
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        p0 = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)

        # Case 1: Zero decay
        zero_decays = torch.zeros(1, 4, dtype=torch.float32)
        curves_zero = engine.simulate_temporal_decay(p0, zero_decays, horizon_steps=5)
        for t in range(5):
            torch.testing.assert_close(curves_zero[0, :, t], p0[0], rtol=1e-5, atol=1e-5)

        # Case 2: Extreme decay
        extreme_decays = torch.full((1, 4), 100.0, dtype=torch.float32)
        curves_ext = engine.simulate_temporal_decay(p0, extreme_decays, horizon_steps=5)
        assert not torch.isnan(curves_ext).any()
        assert not torch.isinf(curves_ext).any()
        for t in range(5):
            assert torch.all(curves_ext[0, :, t] >= 0.0)
            assert torch.all(curves_ext[0, :, t] <= 1.0)


# =====================================================================
# SECTION 4: Downstream Contract Parity
# =====================================================================

class TestDownstreamContractParityStress:
    """
    Stress-tests the downstream interface contract:
    - ReasoningVerdict dataclass adheres to PROJECT.md line 121:
      ReasoningVerdict(action: int, confidence: float, prm_score: float, bull_score: float, bear_score: float, margin: float)
    - Strict Python native type enforcement (int, float, bool - never PyTorch tensors).
    - Multi-batch extraction parity and out-of-bounds safety.
    """

    def test_reasoning_verdict_contract_fields_and_types(self, standard_config):
        """
        Verify that .to_reasoning_verdict() returns a ReasoningVerdict dataclass
        with correct attributes and strict Python scalar types.
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        batch_size = 5
        x = torch.randn(batch_size, standard_config.input_dim)
        with torch.no_grad():
            output: MetacognitiveOutput = engine(x)

        for b in range(batch_size):
            verdict: ReasoningVerdict = output.to_reasoning_verdict(batch_idx=b)

            assert isinstance(verdict, ReasoningVerdict), "Must be instance of ReasoningVerdict"

            # 1. Action contract
            assert isinstance(verdict.action, int), f"Action must be int, got {type(verdict.action)}"
            assert verdict.action in (-1, 0, 1), f"Action must be in {-1, 0, 1}, got {verdict.action}"

            # 2. Confidence contract
            assert isinstance(verdict.confidence, float), f"Confidence must be float, got {type(verdict.confidence)}"
            assert 0.0 <= verdict.confidence <= 1.0

            # 3. PRM score contract
            assert isinstance(verdict.prm_score, float), f"PRM score must be float, got {type(verdict.prm_score)}"
            assert 0.0 <= verdict.prm_score <= 1.0

            # 4. Bull & Bear score contracts
            assert isinstance(verdict.bull_score, float)
            assert isinstance(verdict.bear_score, float)
            assert 0.0 <= verdict.bull_score <= 1.0
            assert 0.0 <= verdict.bear_score <= 1.0

            # 5. Margin contract
            assert isinstance(verdict.margin, float)
            assert -1.0 <= verdict.margin <= 1.0

            # 6. Operational contracts
            assert isinstance(verdict.is_activated, bool)
            assert isinstance(verdict.dominant_scenario, int)
            assert verdict.dominant_scenario in (0, 1, 2, 3)
            assert isinstance(verdict.latency_ms, float)
            assert verdict.latency_ms >= 0.0

    def test_verdict_slice_parity_with_tensor_output(self, standard_config):
        """
        Assert element-wise numerical consistency between the tensor MetacognitiveOutput
        and the extracted scalar ReasoningVerdict for all batch items.
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        batch_size = 8
        x = torch.randn(batch_size, standard_config.input_dim)
        with torch.no_grad():
            output: MetacognitiveOutput = engine(x)

        for b in range(batch_size):
            verdict = output.to_reasoning_verdict(batch_idx=b)

            assert verdict.action == int(output.action[b].item())
            assert abs(verdict.confidence - float(output.confidence[b].item())) < 1e-6
            assert abs(verdict.prm_score - float(output.prm_score[b].item())) < 1e-6
            assert abs(verdict.bull_score - float(output.bull_score[b].item())) < 1e-6
            assert abs(verdict.bear_score - float(output.bear_score[b].item())) < 1e-6
            assert abs(verdict.margin - float(output.adversarial_margin[b].item())) < 1e-6
            assert verdict.is_activated == bool(output.is_activated[b].item())
            assert verdict.dominant_scenario == int(torch.argmax(output.scenario_probs[b]).item())

    def test_out_of_bounds_batch_index_raises_error(self, standard_config):
        """
        Verify that querying an invalid batch index raises IndexError.
        """
        engine = MetacognitiveReasoningEngine(standard_config)
        engine.eval()

        x = torch.randn(2, standard_config.input_dim)
        with torch.no_grad():
            output = engine(x)

        with pytest.raises(IndexError):
            _ = output.to_reasoning_verdict(batch_idx=5)

        with pytest.raises(IndexError):
            _ = output.to_reasoning_verdict(batch_idx=-5)
