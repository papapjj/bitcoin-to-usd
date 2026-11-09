"""
Unit and Integration Tests for Metacognitive Multi-hop Reasoning Engine (R1 Module)
Project: "Mythos của Claude" for xLSTM 17.5M Quantitative System

Validates:
1. Shape contracts and Action/Confidence boundaries.
2. PRM Verifier strict >= 85% activation threshold.
3. Adversarial Bull/Bear debate and scenario probability decay simulation.
4. Absolute 0 NaN / 0 Inf arithmetic preservation (AC-7).
5. Real-time inference latency <= 50ms benchmark (AC-6).
"""

import sys
import os
import time
import pytest
import torch
import torch.nn as nn
import numpy as np

# Configure 1 thread for deterministic single-tick inference (MT5 runner convention)
torch.set_num_threads(1)

# Ensure module is accessible
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from research_reasoning_world_class.src.metacognitive_reasoning import (
        ReasoningConfig,
        MetacognitiveOutput,
        ReasoningVerdict,
        JITReasoningWrapper,
        CoTReasoningHead,
        AdversarialSelfReflection,
        PRMVerifier,
        MetacognitiveReasoningEngine,
    )
except (ImportError, ModuleNotFoundError):
    from src.metacognitive_reasoning import (
        ReasoningConfig,
        MetacognitiveOutput,
        ReasoningVerdict,
        JITReasoningWrapper,
        CoTReasoningHead,
        AdversarialSelfReflection,
        PRMVerifier,
        MetacognitiveReasoningEngine,
    )


@pytest.fixture
def config():
    return ReasoningConfig(
        input_dim=256,
        hidden_dim=256,
        num_hops=4,
        num_scenarios=4,
        prm_threshold=0.85,
        adversarial_margin_threshold=0.20,
        temporal_horizon=12,
    )


@pytest.fixture
def engine(config):
    model = MetacognitiveReasoningEngine(config)
    model.eval()
    return model


class TestMetacognitiveArchitecture:
    """Validates structural and mathematical contracts of the R1 module."""

    def test_output_tensor_shapes(self, engine, config):
        batch_size = 4
        x = torch.randn(batch_size, config.input_dim)
        
        with torch.no_grad():
            output: MetacognitiveOutput = engine(x)
            
        assert output.action.shape == (batch_size,)
        assert output.confidence.shape == (batch_size,)
        assert output.is_activated.shape == (batch_size,)
        assert output.prm_score.shape == (batch_size,)
        assert output.hop_rewards.shape == (batch_size, config.num_hops)
        assert output.cot_hops.shape == (batch_size, config.num_hops, config.hidden_dim)
        assert output.bull_score.shape == (batch_size,)
        assert output.bear_score.shape == (batch_size,)
        assert output.scenario_probs.shape == (batch_size, config.num_scenarios)
        assert output.scenario_decays.shape == (batch_size, config.num_scenarios)
        assert output.scenario_trajectories.shape == (
            batch_size, config.num_scenarios, config.temporal_horizon
        )
        assert output.adversarial_margin.shape == (batch_size,)

    def test_3d_temporal_input_support(self, engine, config):
        """Tests that engine can consume 3D sequential representations (B, T, D)."""
        batch_size = 2
        seq_len = 16
        x = torch.randn(batch_size, seq_len, config.input_dim)
        
        with torch.no_grad():
            output = engine(x)
            
        assert output.action.shape == (batch_size,)
        assert output.cot_hops.shape == (batch_size, config.num_hops, config.hidden_dim)

    def test_action_space_and_confidence_bounds(self, engine, config):
        x = torch.randn(16, config.input_dim)
        
        with torch.no_grad():
            output = engine(x)
            
        # Action must strictly be in {-1, 0, 1}
        unique_actions = torch.unique(output.action).tolist()
        for a in unique_actions:
            assert a in (-1, 0, 1), f"Invalid action value found: {a}"
            
        # Confidence must be in [0.0, 1.0]
        assert torch.all(output.confidence >= 0.0)
        assert torch.all(output.confidence <= 1.0)
        
        # PRM and hop scores in [0.0, 1.0]
        assert torch.all(output.prm_score >= 0.0)
        assert torch.all(output.prm_score <= 1.0)
        assert torch.all(output.hop_rewards >= 0.0)
        assert torch.all(output.hop_rewards <= 1.0)
        
        # Scenario probabilities sum to 1.0 across scenarios
        prob_sums = output.scenario_probs.sum(dim=-1)
        torch.testing.assert_close(
            prob_sums,
            torch.ones_like(prob_sums),
            rtol=1e-4,
            atol=1e-4
        )
        
        # Scenario decay rates must be positive
        assert torch.all(output.scenario_decays > 0.0)


class TestPRMVerificationAndActivationGate:
    """Verifies the Process Reward Model gating and 85% activation threshold."""

    def test_strict_85_percent_threshold_invariance(self, engine, config):
        """
        Signals CANNOT be activated if confidence < 0.85.
        When not activated, action MUST be 0 (HOLD).
        """
        x = torch.randn(32, config.input_dim)
        with torch.no_grad():
            output = engine(x)
            
        for i in range(len(output.confidence)):
            conf = output.confidence[i].item()
            is_act = output.is_activated[i].item()
            act = output.action[i].item()
            
            if conf < config.prm_threshold:
                assert not is_act, f"Sample {i} activated with sub-85% confidence: {conf:.4f}"
                assert act == 0, f"Sample {i} executed action {act} without activation!"
            else:
                if not is_act:
                    assert act == 0
                else:
                    assert act in (-1, 1), f"Activated signal must be directional: {act}"

    def test_synthetic_bullish_activation(self, config):
        """
        Construct a model state where Bull overwhelmingly dominates, PRM verification is high,
        and scenario consensus is established, proving BUY (+1) activation at >= 85% confidence.
        """
        model = MetacognitiveReasoningEngine(config)
        model.eval()
        
        with torch.no_grad():
            # PRM verification set to high confidence
            nn.init.constant_(model.prm_verifier.step_evaluator[-2].bias, 5.0)
            nn.init.constant_(model.prm_verifier.transition_evaluator[-2].bias, 5.0)
            
            # Bull head strongly positive, Bear head strongly negative
            nn.init.constant_(model.reflection.bull_head[-1].bias, 5.0)
            nn.init.constant_(model.reflection.bear_head[-1].bias, -5.0)
            
            # Scenario 0 (Bullish Expansion) dominant
            nn.init.constant_(model.reflection.scenario_head[-1].bias, -2.0)
            model.reflection.scenario_head[-1].bias.data[0] = 5.0
            
            x = torch.ones(2, config.input_dim) * 0.1
            output = model(x)
            
        assert torch.all(output.is_activated), "Expected signal activation"
        assert torch.all(output.action == 1), f"Expected BUY (1), got {output.action}"
        assert torch.all(output.confidence >= 0.85), f"Confidence below 85%: {output.confidence}"
        assert torch.all(output.bull_score > 0.90)
        assert torch.all(output.bear_score < 0.10)
        assert torch.all(output.adversarial_margin > 0.80)

    def test_synthetic_bearish_activation(self, config):
        """
        Construct a model state where Bear overwhelmingly dominates, PRM verification is high,
        and scenario consensus is established, proving SELL (-1) activation at >= 85% confidence.
        """
        model = MetacognitiveReasoningEngine(config)
        model.eval()
        
        with torch.no_grad():
            # PRM verification set to high confidence
            nn.init.constant_(model.prm_verifier.step_evaluator[-2].bias, 5.0)
            nn.init.constant_(model.prm_verifier.transition_evaluator[-2].bias, 5.0)
            
            # Bear head strongly positive, Bull head strongly negative
            nn.init.constant_(model.reflection.bear_head[-1].bias, 5.0)
            nn.init.constant_(model.reflection.bull_head[-1].bias, -5.0)
            
            # Scenario 1 (Bearish Reversal) dominant
            nn.init.constant_(model.reflection.scenario_head[-1].bias, -2.0)
            model.reflection.scenario_head[-1].bias.data[1] = 5.0
            
            x = torch.ones(2, config.input_dim) * 0.1
            output = model(x)
            
        assert torch.all(output.is_activated), "Expected signal activation"
        assert torch.all(output.action == -1), f"Expected SELL (-1), got {output.action}"
        assert torch.all(output.confidence >= 0.85), f"Confidence below 85%: {output.confidence}"
        assert torch.all(output.bear_score > 0.90)
        assert torch.all(output.bull_score < 0.10)
        assert torch.all(output.adversarial_margin < -0.80)

    def test_conflicted_debate_results_in_hold(self, config):
        """
        When Bull and Bear are equally matched (adversarial margin near 0),
        engine must output HOLD (0) even if individual confidence is moderate.
        """
        model = MetacognitiveReasoningEngine(config)
        model.eval()
        
        with torch.no_grad():
            # Zero out the final projection weights & set identical biases
            nn.init.zeros_(model.reflection.bull_head[-1].weight)
            nn.init.zeros_(model.reflection.bear_head[-1].weight)
            nn.init.constant_(model.reflection.bull_head[-1].bias, 0.0)
            nn.init.constant_(model.reflection.bear_head[-1].bias, 0.0)
                
            x = torch.randn(4, config.input_dim)
            output = model(x)
            
        assert torch.all(output.adversarial_margin.abs() < 1e-4)
        assert torch.all(output.action == 0), "Equally matched Bull/Bear debate must yield HOLD (0)"
        assert not torch.any(output.is_activated)


class TestScenarioDecaySimulation:
    """Verifies forward temporal probability decay across simulated market scenarios."""

    def test_exponential_decay_dynamics(self, engine, config):
        batch_size = 2
        x = torch.randn(batch_size, config.input_dim)
        
        with torch.no_grad():
            out = engine(x)
            decay_curves = engine.simulate_temporal_decay(
                out.scenario_probs,
                out.scenario_decays,
                horizon_steps=10
            )
            
        # Shape: (B, num_scenarios, horizon_steps)
        assert decay_curves.shape == (batch_size, config.num_scenarios, 10)
        
        # Verify probability simplex at each time horizon
        for t in range(10):
            step_sum = decay_curves[:, :, t].sum(dim=-1)
            torch.testing.assert_close(
                step_sum,
                torch.ones_like(step_sum),
                rtol=1e-4,
                atol=1e-4
            )


class TestNumericalRobustnessAC7:
    """AC-7 (Bảo Toàn Số Học): Zero NaN, Zero Inf across edge cases."""

    def test_extreme_inputs_numerical_stability(self, engine, config):
        """Test with zeros, extreme positives, extreme negatives, and explicit NaNs/Infs."""
        extreme_cases = [
            ("all_zeros", torch.zeros(4, config.input_dim)),
            ("large_positives", torch.full((4, config.input_dim), 1e4)),
            ("large_negatives", torch.full((4, config.input_dim), -1e4)),
            ("injected_nans", torch.randn(4, config.input_dim)),
            ("injected_infs", torch.randn(4, config.input_dim)),
        ]
        
        # Inject NaNs
        extreme_cases[3][1][0, :50] = float("nan")
        extreme_cases[3][1][2, 100:150] = float("nan")
        
        # Inject Infs
        extreme_cases[4][1][1, :50] = float("inf")
        extreme_cases[4][1][3, 50:100] = float("-inf")
        
        for name, tensor in extreme_cases:
            with torch.no_grad():
                out = engine(tensor)
                
            # Assert 0 NaN and 0 Inf in all outputs
            for field_name, val in [
                ("confidence", out.confidence),
                ("prm_score", out.prm_score),
                ("hop_rewards", out.hop_rewards),
                ("cot_hops", out.cot_hops),
                ("bull_score", out.bull_score),
                ("bear_score", out.bear_score),
                ("scenario_probs", out.scenario_probs),
                ("scenario_decays", out.scenario_decays),
                ("scenario_trajectories", out.scenario_trajectories),
                ("adversarial_margin", out.adversarial_margin),
            ]:
                has_nan = torch.isnan(val).any().item()
                has_inf = torch.isinf(val).any().item()
                assert not has_nan, f"NaN detected in {field_name} for case '{name}'"
                assert not has_inf, f"Inf detected in {field_name} for case '{name}'"


class TestLatencyBenchmarkAC6:
    """AC-6 (Tốc Độ Thực Thi): Real-time inference <= 50ms."""

    def test_inference_latency_under_50ms(self, engine, config):
        """Benchmark 100 consecutive single-tick forward passes (batch_size=1)."""
        num_warmup = 15
        num_iters = 100
        
        x = torch.randn(1, config.input_dim)
        
        # Warm-up
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = engine(x)
                
        # Benchmark
        latencies = []
        with torch.no_grad():
            for _ in range(num_iters):
                t0 = time.perf_counter()
                out = engine(x)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # ms
                
        latencies = np.array(latencies)
        mean_lat = float(np.mean(latencies))
        p50_lat = float(np.percentile(latencies, 50))
        p95_lat = float(np.percentile(latencies, 95))
        p99_lat = float(np.percentile(latencies, 99))
        max_lat = float(np.max(latencies))
        
        print("\n" + "=" * 60)
        print("LATENCY BENCHMARK REPORT (AC-6 Standard):")
        print(f"Iterations: {num_iters} ticks (Batch Size = 1)")
        print(f"Mean Latency : {mean_lat:.3f} ms")
        print(f"P50  Latency : {p50_lat:.3f} ms")
        print(f"P95  Latency : {p95_lat:.3f} ms")
        print(f"P99  Latency : {p99_lat:.3f} ms")
        print(f"Max  Latency : {max_lat:.3f} ms")
        print(f"Target Threshold: <= 50.000 ms")
        print("=" * 60)
        
        assert mean_lat <= 50.0, f"Mean latency exceeded 50ms: {mean_lat:.3f} ms"
        assert p99_lat <= 50.0, f"P99 latency exceeded 50ms: {p99_lat:.3f} ms"


class TestReasoningVerdictContractAndJITWrapper:
    """Verifies formal interface contract ReasoningVerdict and JITReasoningWrapper."""

    def test_to_reasoning_verdict_bridge(self, engine, config):
        batch_size = 3
        x = torch.randn(batch_size, config.input_dim)
        with torch.no_grad():
            output: MetacognitiveOutput = engine(x)

        for i in range(batch_size):
            verdict: ReasoningVerdict = output.to_reasoning_verdict(batch_idx=i)
            alias_verdict: ReasoningVerdict = output.to_verdict(batch_idx=i)

            assert isinstance(verdict, ReasoningVerdict)
            assert verdict.action in (-1, 0, 1)
            assert 0.0 <= verdict.confidence <= 1.0
            assert 0.0 <= verdict.prm_score <= 1.0
            assert 0.0 <= verdict.bull_score <= 1.0
            assert 0.0 <= verdict.bear_score <= 1.0
            assert -1.0 <= verdict.margin <= 1.0
            assert isinstance(verdict.is_activated, bool)
            assert verdict.dominant_scenario in (0, 1, 2, 3)
            assert verdict.latency_ms >= 0.0
            assert verdict.action == alias_verdict.action
            assert verdict.confidence == alias_verdict.confidence

    def test_jit_reasoning_wrapper_parity(self, engine, config):
        wrapper = JITReasoningWrapper(engine).eval()
        x = torch.randn(2, config.input_dim)

        with torch.no_grad():
            out_eager = engine(x)
            act, conf, is_act, prm, margin = wrapper(x)

        assert torch.equal(act, out_eager.action)
        assert torch.allclose(conf, out_eager.confidence, atol=1e-6)
        assert torch.equal(is_act, out_eager.is_activated)
        assert torch.allclose(prm, out_eager.prm_score, atol=1e-6)
        assert torch.allclose(margin, out_eager.adversarial_margin, atol=1e-6)

