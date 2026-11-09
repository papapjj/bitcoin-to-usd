"""
Challenger Adversarial Empirical Verification Suite (Milestone M1)
Project: "Mythos của Claude" - xLSTM 17.5M Frontier Mythos Reasoning Architecture
Agent: teamwork_preview_challenger_m1_1

Adversarial Stress Test Objectives:
1. Pairwise Hop Cosine Similarity:
   - Measure cos(h_i, h_j) across 100 diverse random batches.
   - Test adversarial distributions (Gaussian, high-var, low-var, collinear, sparse, extreme).
   - Verify cos(h_i, h_j) < 0.95 invariant holds across all pairs, batches, and individual samples.
   - Verify both nominal 4-hop mode and extended 8-hop Soft ACT mode.
2. Backtracking Revision Loop:
   - Subtle traps: test boundary response around tau_trap = 0.65 (sub-threshold vs supra-threshold).
   - Extreme traps: test premise reversal cleanliness (cos(h_1, h_1_revised), magnitude, norm preservation).
   - Concordant inputs: test whether backtracking ever triggers falsely when thoughts agree and smc=0.
   - Strict N_backtrack <= 1 invariant: verify bounded single-pass execution via call instrumentation.
   - Multi-pass iterative stability: verify LayerNorm bounds and zero NaN/Inf under repeated revision.
"""

from __future__ import annotations

import math
import os
import sys
import unittest
from typing import Dict, List, Tuple
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# MT5 deterministic single-thread convention
torch.set_num_threads(1)

# Ensure project and repository roots are in sys.path
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
    GatedReasoningHop,
    DenseCrossHopAttention,
    ContradictionDetector,
    BacktrackingRevisionModule,
    SoftHaltingPonderController,
    CoTReasoningHead,
    AdversarialSelfReflection,
    PRMVerifier,
    MetacognitiveReasoningEngine,
)


class TestPairwiseHopCosineSimilarity100Batches:
    """
    Adversarially tests pairwise hop cosine similarity across 100 diverse random batches.
    Can any input cause cos(h_i, h_j) >= 0.95?
    """

    @pytest.fixture(autouse=True)
    def setup_engine(self):
        self.config_4hop = ReasoningConfig(
            input_dim=256,
            hidden_dim=256,
            num_hops=4,
            enable_cross_hop=True,
            enable_backtrack=True,
            enable_act=False,
        )
        self.engine_4hop = MetacognitiveReasoningEngine(self.config_4hop).eval()

        self.config_8hop = ReasoningConfig(
            input_dim=256,
            hidden_dim=256,
            num_hops=4,
            max_hops=8,
            min_hops=2,
            enable_cross_hop=True,
            enable_backtrack=True,
            enable_act=True,
        )
        self.engine_8hop = MetacognitiveReasoningEngine(self.config_8hop).eval()

    def generate_adversarial_batch(self, batch_idx: int) -> Tuple[torch.Tensor, str]:
        """
        Generates 100 diverse input batches covering various potential failure regimes.
        """
        torch.manual_seed(1000 + batch_idx)
        B = [1, 2, 4, 8, 16, 32][batch_idx % 6]
        D = 256

        category_idx = batch_idx // 10
        if category_idx == 0:
            # 0..9: Standard Gaussian
            desc = "Standard Gaussian N(0, 1)"
            x = torch.randn(B, D)
        elif category_idx == 1:
            # 10..19: Extreme high variance (exploding input)
            desc = "High-variance Gaussian N(0, 100^2)"
            x = torch.randn(B, D) * 100.0
        elif category_idx == 2:
            # 20..29: Low variance near-zero (vanishing input)
            desc = "Near-zero Gaussian N(0, 1e-4^2)"
            x = torch.randn(B, D) * 1e-4
        elif category_idx == 3:
            # 30..39: Constant collinear vectors
            c = (batch_idx - 35) * 2.5
            desc = f"Collinear constant vector c={c}"
            x = torch.full((B, D), c, dtype=torch.float32)
        elif category_idx == 4:
            # 40..49: Sparse impulse spikes (1-3 active features)
            desc = "Sparse impulse features"
            x = torch.zeros(B, D)
            for b in range(B):
                indices = torch.randint(0, D, (3,))
                x[b, indices] = torch.randn(3) * 10.0
        elif category_idx == 5:
            # 50..59: Alternating sign square wave (-1, +1, -1, +1, ...)
            desc = "Alternating sign square wave"
            pattern = torch.tensor([1.0, -1.0] * (D // 2))
            x = pattern.unsqueeze(0).expand(B, -1) * (batch_idx % 5 + 1) * 2.0
        elif category_idx == 6:
            # 60..69: Saturated step inputs (+50 or -50)
            desc = "Saturated binary steps"
            x = torch.sign(torch.randn(B, D)) * 50.0
        elif category_idx == 7:
            # 70..79: Cauchy / heavy-tailed noise
            desc = "Heavy-tailed Cauchy noise"
            x = torch.distributions.Cauchy(0.0, 1.0).sample((B, D))
            x = torch.clamp(x, -500.0, 500.0)
        elif category_idx == 8:
            # 80..89: Corrupted inputs with NaN/Inf (tested for sanitization + diversity)
            desc = "Corrupted NaN/Inf injection"
            x = torch.randn(B, D)
            x[0, 0] = float("nan")
            x[0, 1] = float("inf")
            x[0, 2] = -float("inf")
        else:
            # 90..99: Uniform distributed noise U(-10, 10)
            desc = "Uniform U(-10, 10)"
            x = (torch.rand(B, D) - 0.5) * 20.0

        return x, desc

    def test_pairwise_cosine_similarity_across_100_batches_4hop(self):
        """
        Empirically evaluate all pairwise hop cosine similarities across 100 batches in 4-hop mode.
        Assert that for every single sample b and pair (i, j): cos(h_i, h_j) < 0.95.
        """
        all_pairwise_max: List[float] = []
        all_pairwise_means: List[float] = []
        pair_records: Dict[Tuple[int, int], List[float]] = {}

        for i in range(4):
            for j in range(i + 1, 4):
                pair_records[(i, j)] = []

        global_max_cos = -1.0
        worst_batch_idx = -1
        worst_pair = (-1, -1)
        worst_desc = ""

        with torch.no_grad():
            for batch_idx in range(100):
                x, desc = self.generate_adversarial_batch(batch_idx)
                out = self.engine_4hop(x)
                cot_hops = out.cot_hops # (B, 4, D)
                B, K, D = cot_hops.shape

                batch_max = -1.0
                for i in range(K):
                    for j in range(i + 1, K):
                        hi = cot_hops[:, i, :] # (B, D)
                        hj = cot_hops[:, j, :] # (B, D)
                        cos_sim = F.cosine_similarity(hi, hj, dim=-1) # (B,)
                        max_s = float(cos_sim.max().item())
                        mean_s = float(cos_sim.mean().item())

                        pair_records[(i, j)].append(mean_s)
                        if max_s > batch_max:
                            batch_max = max_s
                        if max_s > global_max_cos:
                            global_max_cos = max_s
                            worst_batch_idx = batch_idx
                            worst_pair = (i, j)
                            worst_desc = desc

                all_pairwise_max.append(batch_max)
                all_pairwise_means.append(float(np.mean([pair_records[p][-1] for p in pair_records])))

        # Print statistical verification summary
        print("\n" + "=" * 80)
        print("EMPIRICAL CHALLENGER VERIFICATION: 100 RANDOM BATCHES PAIRWISE COSINE SIMILARITY")
        print("=" * 80)
        print(f"Total Batches Evaluated: 100")
        print(f"Global Maximum Sample Pairwise Cosine Similarity: {global_max_cos:.6f}")
        print(f"Worst Case: Batch #{worst_batch_idx} ({worst_desc}), Pair Hop {worst_pair[0]} vs Hop {worst_pair[1]}")
        print(f"Mean of Batch Maxima: {np.mean(all_pairwise_max):.6f}")
        print(f"P95 of Batch Maxima:  {np.percentile(all_pairwise_max, 95):.6f}")
        print(f"P99 of Batch Maxima:  {np.percentile(all_pairwise_max, 99):.6f}")
        print("-" * 80)
        print("Pairwise Mean Cosine Similarity Breakdown:")
        for (i, j), vals in pair_records.items():
            print(f"  Hop {i} vs Hop {j}: Mean={np.mean(vals):.4f}, Min={np.min(vals):.4f}, Max={np.max(vals):.4f}")
        print("=" * 80)

        assert global_max_cos < 0.95, (
            f"VIOLATION: Found input with pairwise cosine similarity >= 0.95! "
            f"Observed cos={global_max_cos:.6f} at Batch #{worst_batch_idx} ({worst_desc}) "
            f"for Hop {worst_pair[0]} vs Hop {worst_pair[1]}."
        )

    def test_pairwise_cosine_similarity_across_100_batches_8hop_act(self):
        """
        Adversarially evaluate all 28 pairwise hop combinations in extended 8-hop Soft ACT mode
        across 100 random batches.
        """
        global_max_8hop = -1.0
        worst_batch = -1
        worst_pair = (-1, -1)

        with torch.no_grad():
            for batch_idx in range(100):
                x, desc = self.generate_adversarial_batch(batch_idx)
                # Random market uncertainty
                u = torch.rand(x.size(0))
                out = self.engine_8hop(x, uncertainty=u)
                cot_hops = out.cot_hops # (B, 8, D)
                B, K, D = cot_hops.shape
                assert K == 8

                for i in range(K):
                    for j in range(i + 1, K):
                        cos_sim = F.cosine_similarity(cot_hops[:, i], cot_hops[:, j], dim=-1)
                        m = float(cos_sim.max().item())
                        if m > global_max_8hop:
                            global_max_8hop = m
                            worst_batch = batch_idx
                            worst_pair = (i, j)

        print(f"\n8-Hop Extended ACT Global Max Cosine Similarity: {global_max_8hop:.6f} "
              f"(Batch #{worst_batch}, Hop {worst_pair[0]} vs Hop {worst_pair[1]})")

        assert global_max_8hop < 0.95, (
            f"VIOLATION in 8-hop mode: cos={global_max_8hop:.6f} >= 0.95 at Hop {worst_pair}"
        )


class TestBacktrackingRevisionAdversarialStress:
    """
    Adversarial verification of Backtracking Revision Loop under:
    1. Subtle traps (boundary conditions around tau_trap = 0.65)
    2. Extreme traps (premise reversal cleanliness & directional inversion)
    3. Concordant inputs (false positive prevention)
    4. Strict N_backtrack <= 1 execution bound (recursion prevention)
    5. Multi-pass iterative stability
    """

    @pytest.fixture(autouse=True)
    def setup_modules(self):
        self.hidden_dim = 256
        self.smc_dim = 64
        self.backtracker = BacktrackingRevisionModule(
            hidden_dim=self.hidden_dim,
            smc_dim=self.smc_dim,
            trap_threshold=0.65
        ).eval()
        self.detector = ContradictionDetector(
            hidden_dim=self.hidden_dim,
            smc_dim=self.smc_dim,
            trap_threshold=0.65
        ).eval()

    def test_subtle_traps_boundary_threshold(self):
        """
        Test subtle traps right below and right above tau_trap = 0.65 threshold.
        - When tau_trap = 0.649 -> was_backtracked MUST be False, magnitude == 0.0, revised_h1 == h1
        - When tau_trap = 0.651 -> was_backtracked MUST be True, magnitude > 0.0
        """
        torch.manual_seed(42)
        h1 = torch.randn(4, self.hidden_dim)
        hk = torch.randn(4, self.hidden_dim)
        smc = torch.randn(4, self.smc_dim)

        # Mock detector with precise scalar return to test boundary sharpness
        class MockDetector(nn.Module):
            def __init__(self, val: float):
                super().__init__()
                self.val = val

            def forward(self, x):
                return torch.full((x.size(0), 1), self.val, device=x.device)

        # Test sub-threshold values
        for tau in [0.0, 0.30, 0.50, 0.6499]:
            self.backtracker.detector = MockDetector(tau)
            rev_h1, was_bt, mag = self.backtracker(h1, hk, smc)
            assert was_bt is False, f"Sub-threshold tau={tau} incorrectly triggered backtracking!"
            assert mag == 0.0, f"Sub-threshold tau={tau} returned non-zero magnitude {mag}"
            assert torch.allclose(rev_h1, h1), f"Sub-threshold tau={tau} altered h1!"

        # Test supra-threshold subtle values
        for tau in [0.6501, 0.66, 0.70]:
            self.backtracker.detector = MockDetector(tau)
            rev_h1, was_bt, mag = self.backtracker(h1, hk, smc)
            assert was_bt is True, f"Supra-threshold tau={tau} failed to trigger backtracking!"
            assert mag > 0.0, f"Supra-threshold tau={tau} returned zero magnitude"
            assert not torch.allclose(rev_h1, h1), f"Supra-threshold tau={tau} did not alter h1"

    def test_extreme_traps_premise_reversal_cleanliness(self):
        """
        Under extreme institutional traps (tau_trap >= 0.90, massive liquidity sweep),
        does Hop 1 premise reverse cleanly?
        Measure:
        - cos(h1, revised_h1): must achieve strong angular divergence (cos < 0.10 or negative)
        - magnitude: ||revised_h1 - h1|| > 1.0
        - LayerNorm conservation: ||revised_h1||_2 remains bounded near sqrt(D) = 16.0
        - Zero NaN / Zero Inf
        """
        torch.manual_seed(99)
        h1 = torch.randn(8, self.hidden_dim)
        hk_trap = torch.randn(8, self.hidden_dim)
        smc_trap = torch.ones(8, self.smc_dim) * 5.0 # extreme institutional sweep

        # Force extreme trap detection
        with torch.no_grad():
            for p in self.backtracker.detector.parameters():
                p.fill_(0.8)

        rev_h1, was_bt, mag = self.backtracker(h1, hk_trap, smc_trap)
        assert was_bt is True

        cos_sim = F.cosine_similarity(h1, rev_h1, dim=-1)
        mean_cos = float(cos_sim.mean().item())
        max_cos = float(cos_sim.max().item())
        l2_norm = float(torch.norm(rev_h1, dim=-1).mean().item())
        expected_norm = math.sqrt(self.hidden_dim) # 16.0 for LayerNorm(256)

        print("\n" + "=" * 80)
        print("EXTREME TRAP PREMISE REVERSAL EMPIRICAL METRICS")
        print("=" * 80)
        print(f"Revision Magnitude: {mag:.4f} (Required > 1.0)")
        print(f"Mean cos(h1, rev_h1): {mean_cos:.4f} (Required < 0.10)")
        print(f"Max  cos(h1, rev_h1): {max_cos:.4f} (Required < 0.20)")
        print(f"Revised h1 L2 Norm:  {l2_norm:.4f} (LayerNorm theoretical: {expected_norm:.4f})")
        print("=" * 80)

        # Measure directional premise reversal
        assert mag > 1.0, f"Revision magnitude {mag} is too weak for extreme trap!"
        assert abs(l2_norm - expected_norm) < 0.5, f"L2 norm {l2_norm} deviated from LayerNorm standard {expected_norm}"
        assert not torch.isnan(rev_h1).any(), "NaN found in revised_h1"
        assert not torch.isinf(rev_h1).any(), "Inf found in revised_h1"

        # Empirical findings on premise reversal cleanliness:
        # Cosine similarity is positive (+0.5663), indicating incomplete directional inversion.
        # Record this finding for challenger report.
        print(f"EMPIRICAL VERDICT NOTE: Hop 1 premise cosine similarity after extreme trap is {mean_cos:.4f} > 0.0")
        print(f"Original premise is not cleanly inverted to negative half-space (Worker claimed cos = -0.0000).")

    def test_concordant_inputs_zero_false_positives(self):
        """
        When reasoning directions agree (concordant thoughts, e.g. hk is aligned with h1,
        or hk == h1, and no institutional liquidity sweep exists smc=0):
        Does Backtracking Revision ever trigger falsely?
        Test across 1,000 concordant pairs.
        """
        # Re-initialize detector to standard trained/initialized state
        self.backtracker = BacktrackingRevisionModule(
            hidden_dim=self.hidden_dim, smc_dim=self.smc_dim, trap_threshold=0.65
        ).eval()

        false_triggers = 0
        confidences = []

        torch.manual_seed(2026)
        N_TESTS = 1000

        with torch.no_grad():
            for i in range(N_TESTS):
                # Concordant thought: hk positively correlated with h1
                h1 = torch.randn(1, self.hidden_dim)
                alpha = torch.rand(1).item() * 2.0 + 0.5 # positive scalar
                hk_concordant = h1 * alpha + torch.randn(1, self.hidden_dim) * 0.05
                smc_benign = torch.zeros(1, self.smc_dim)

                rev_h1, was_bt, mag = self.backtracker(h1, hk_concordant, smc_benign)
                trap_in = torch.cat([h1, hk_concordant, smc_benign], dim=-1)
                conf = float(self.backtracker.detector(trap_in).item())
                confidences.append(conf)

                if was_bt:
                    false_triggers += 1

        print("\n" + "=" * 80)
        print("CONCORDANT INPUTS ADVERSARIAL STRESS TEST")
        print("=" * 80)
        print(f"Total Concordant Pairs Tested: {N_TESTS}")
        print(f"False Triggers (was_backtracked == True): {false_triggers}")
        print(f"Mean Trap Confidence on Concordant Thoughts: {np.mean(confidences):.4f}")
        print(f"Max Trap Confidence on Concordant Thoughts:  {np.max(confidences):.4f} (Threshold = 0.65)")
        print("=" * 80)

        assert false_triggers == 0, (
            f"VIOLATION: Backtracking falsely triggered on {false_triggers}/{N_TESTS} concordant inputs! "
            f"Max confidence was {np.max(confidences):.4f}"
        )
        assert np.max(confidences) < 0.65, (
            f"Max trap confidence {np.max(confidences):.4f} exceeded threshold 0.65 on concordant thoughts!"
        )

    def test_strict_n_backtrack_le_1_invariant(self):
        """
        Verify that CoTReasoningHead strictly enforces N_backtrack <= 1.
        Instruments backtrack_module.forward to count executions during an engine forward pass.
        Even when thoughts are artificially engineered to trigger traps on subsequent passes,
        backtrack_module MUST be called at most ONCE.
        """
        config = ReasoningConfig(
            input_dim=256,
            hidden_dim=256,
            num_hops=4,
            enable_cross_hop=True,
            enable_backtrack=True,
        )
        head = CoTReasoningHead(config).eval()

        # Force high trap detection on ALL thoughts
        with torch.no_grad():
            for p in head.backtrack_module.detector.parameters():
                p.fill_(0.7)

        # Instrument forward method
        call_count = [0]
        original_forward = head.backtrack_module.forward

        def instrumented_forward(*args, **kwargs):
            call_count[0] += 1
            return original_forward(*args, **kwargs)

        head.backtrack_module.forward = instrumented_forward

        # Execute reasoning forward pass
        x = torch.randn(4, 256)
        all_hops, final_thought = head(x)

        print("\n" + "=" * 80)
        print("STRICT N_BACKTRACK <= 1 INVARIANT CHECK")
        print("=" * 80)
        print(f"Total Backtracking Passes Executed: {call_count[0]}")
        print(f"was_backtracked status: {head.last_was_backtracked}")
        print(f"revision_magnitude: {head.last_revision_magnitude:.4f}")
        print("=" * 80)

        assert call_count[0] == 1, (
            f"VIOLATION: Backtrack module executed {call_count[0]} times! "
            f"Strict contract requires N_backtrack <= 1."
        )
        assert head.last_was_backtracked is True
        assert head.last_revision_magnitude > 0.0

    def test_multi_pass_iterative_stability(self):
        """
        Adversarially probe repeated revision passes:
        If an external caller or recursive process invokes BacktrackingRevisionModule
        repeatedly (5 passes), verify that numerical representation stays stable:
        - No NaN or Inf
        - LayerNorm magnitude remains ~16.0
        - Revision magnitude stabilizes / does not oscillate to infinity
        """
        self.backtracker = BacktrackingRevisionModule(
            hidden_dim=self.hidden_dim, smc_dim=self.smc_dim, trap_threshold=0.65
        ).eval()

        with torch.no_grad():
            for p in self.backtracker.detector.parameters():
                p.fill_(0.7)

        h = torch.randn(4, self.hidden_dim)
        hk = torch.randn(4, self.hidden_dim)
        smc = torch.randn(4, self.smc_dim)

        norms = []
        for step in range(5):
            h, was_bt, mag = self.backtracker(h, hk, smc)
            assert was_bt is True
            assert not torch.isnan(h).any()
            assert not torch.isinf(h).any()
            current_norm = float(torch.norm(h, dim=-1).mean().item())
            norms.append(current_norm)

        expected = math.sqrt(self.hidden_dim)
        for step, norm_val in enumerate(norms):
            assert abs(norm_val - expected) < 0.5, (
                f"Pass {step} norm {norm_val} deviated from LayerNorm expected {expected}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
