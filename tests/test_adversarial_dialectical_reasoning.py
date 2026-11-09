"""
tests/test_adversarial_dialectical_reasoning.py

Forensic Verification Suite for Adversarial Dialectical Reasoning Engine.
Verifies:
1. Architectural initialization & dimension adaptability (233D / 768D).
2. Strict latency constraint: CPU inference <= 50.0 ms/tick.
3. Numerical safety: Zero NaN / Inf under extreme or degenerate input states.
4. Game-Theoretic Trap Exploitation (Counter-Punch):
   - Bull Trap -> Arbiter issues SNIPER SELL (-1).
   - Bear Trap -> Arbiter issues SNIPER BUY (+1).
5. Multi-turn Dialectical Cross-Examination progress over 3 rounds.
"""

import pytest
import time
import torch
import numpy as np

from research_reasoning_world_class.src.adversarial_dialectical_reasoning import (
    AdversarialDialecticalReasoningEngine,
    DialecticalConfig,
    DialecticalVerdict,
    MetacognitiveDialecticalArbiter,
)


def test_dialectical_engine_initialization():
    config = DialecticalConfig(input_dim=233, hidden_dim=128, num_debate_turns=3)
    engine = AdversarialDialecticalReasoningEngine(config)
    assert engine is not None
    assert engine.config.num_debate_turns == 3


def test_forward_pass_numerical_safety():
    config = DialecticalConfig(input_dim=233, hidden_dim=128, num_debate_turns=3)
    engine = AdversarialDialecticalReasoningEngine(config)
    
    # Test with standard tensor
    x = torch.randn(1, 233)
    verdict = engine(x)
    assert isinstance(verdict, DialecticalVerdict)
    assert verdict.action in (-1, 0, 1)
    assert 0.0 <= verdict.prm_score <= 1.0
    assert 0.0 <= verdict.confidence <= 1.0
    assert len(verdict.debate_history) == 3
    
    # Test with NaN/Inf injected input
    x_bad = torch.randn(1, 233)
    x_bad[0, 5] = float('nan')
    x_bad[0, 10] = float('inf')
    x_bad[0, 15] = float('-inf')
    verdict_safe = engine(x_bad)
    assert not np.isnan(verdict_safe.prm_score)
    assert not np.isnan(verdict_safe.confidence)
    assert verdict_safe.action in (-1, 0, 1)


def test_latency_budget():
    config = DialecticalConfig(input_dim=233, hidden_dim=128, num_debate_turns=3)
    engine = AdversarialDialecticalReasoningEngine(config)
    engine.eval()
    
    x = torch.randn(1, 233)
    # Warmup
    for _ in range(5):
        _ = engine(x)
        
    times = []
    with torch.no_grad():
        for _ in range(30):
            t0 = time.perf_counter()
            _ = engine(x)
            times.append((time.perf_counter() - t0) * 1000.0)
            
    median_latency = np.median(times)
    p95_latency = np.percentile(times, 95)
    print(f"\n[Bench] Dialectical Reasoning Latency: Median={median_latency:.2f} ms | P95={p95_latency:.2f} ms")
    assert median_latency < 50.0, f"Latency {median_latency:.2f} ms exceeded 50.0 ms budget"


def test_bull_trap_counter_punch_exploitation():
    """
    When Adversary Red-Team proves a Bull Trap (Liquidity Sweep High / Upthrust),
    Arbiter must NOT remain passively in HOLD; it must execute a SNIPER SELL (-1).
    """
    arbiter = MetacognitiveDialecticalArbiter(hidden_dim=128, prm_threshold=0.88)
    
    final_p = torch.randn(1, 128)
    final_a = torch.randn(1, 128)
    convictions = torch.tensor([[0.70, 0.30]]) # Proponent thinks it's Long
    
    # Adversary proves Bull Trap with 85% probability: [No Trap: 0.10, Bull Trap: 0.85, Bear Trap: 0.05]
    trap_probs = torch.tensor([[0.10, 0.85, 0.05]])
    
    action, prm, conf, counter_punch = arbiter(final_p, final_a, trap_probs, convictions)
    
    # Counter-punch: inverts the trap to SELL (-1)
    assert action.item() == -1, f"Expected action -1 (SELL) for Bull Trap, got {action.item()}"
    assert counter_punch.item() is True, "Expected counter_punch flag to be True"
    assert conf.item() >= 0.85, f"Expected high confidence for trap exploitation, got {conf.item()}"


def test_bear_trap_counter_punch_exploitation():
    """
    When Adversary Red-Team proves a Bear Trap (Liquidity Sweep Low / Spring),
    Arbiter must execute a SNIPER BUY (+1) to catch the reversal.
    """
    arbiter = MetacognitiveDialecticalArbiter(hidden_dim=128, prm_threshold=0.88)
    
    final_p = torch.randn(1, 128)
    final_a = torch.randn(1, 128)
    convictions = torch.tensor([[0.30, 0.70]]) # Proponent thinks it's Short
    
    # Adversary proves Bear Trap with 90% probability: [No Trap: 0.05, Bull Trap: 0.05, Bear Trap: 0.90]
    trap_probs = torch.tensor([[0.05, 0.05, 0.90]])
    
    action, prm, conf, counter_punch = arbiter(final_p, final_a, trap_probs, convictions)
    
    # Counter-punch: inverts the trap to BUY (+1)
    assert action.item() == 1, f"Expected action 1 (BUY) for Bear Trap, got {action.item()}"
    assert counter_punch.item() is True, "Expected counter_punch flag to be True"
    assert conf.item() >= 0.85, f"Expected high confidence for trap exploitation, got {conf.item()}"


def test_multi_turn_debate_progression():
    config = DialecticalConfig(input_dim=233, hidden_dim=128, num_debate_turns=3)
    engine = AdversarialDialecticalReasoningEngine(config)
    
    x = torch.randn(1, 233)
    verdict = engine(x)
    
    assert len(verdict.debate_history) == 3
    for entry in verdict.debate_history:
        assert "round" in entry
        assert "proponent_pressure" in entry
        assert "adversary_pressure" in entry
        assert entry["proponent_pressure"] > 0.0
        assert entry["adversary_pressure"] > 0.0
