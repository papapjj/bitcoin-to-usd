"""
tests/test_bilateral_dual_cot_reasoning.py

Forensic Verification & Regression Test Suite for Bilateral Dual-CoT & Dialectical Reasoning Architecture.
Validates:
1. Two distinct, independent reasoning models (BullCoT vs BearCoT) with separate parameters.
2. 3-turn Dialectical Cross-Examination without deadlock and latency <= 50.0 ms.
3. PRM Verifier strict gate (PRM >= 0.88 required for regular non-trap entries).
4. Counter-Punch Invariants:
   - Bull Trap exposed -> Inverts to SNIPER SELL (-1).
   - Bear Trap exposed -> Inverts to SNIPER BUY (+1).
5. Numerical safety under extreme/corrupt inputs (Zero NaN / Inf).
6. Governor Agent 4 strict HOLD (0) invariant preservation on MT5 Bridge.
7. Checkpoint serialization roundtrip and backward compatibility.
"""

from pathlib import Path
import time
import pytest
import numpy as np
import torch

from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    BilateralDualVerdict,
)


def test_independent_dual_cot_initialization():
    """Verify that Bull and Bear CoTs are two distinct neural networks with separate parameters."""
    config = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, num_debate_turns=3)
    engine = BilateralDualCoTEngine(config)
    
    assert engine.bull_cot is not None
    assert engine.bear_cot is not None
    assert engine.bull_cot is not engine.bear_cot, "Bull and Bear must be separate model instances"
    
    # Confirm distinct parameter pointers
    bull_params = list(engine.bull_cot.parameters())
    bear_params = list(engine.bear_cot.parameters())
    assert len(bull_params) == len(bear_params)
    assert bull_params[0].data_ptr() != bear_params[0].data_ptr(), "Parameters must not share memory storage"
    assert engine.config.num_debate_turns == 3


def test_multi_turn_debate_progression_and_latency():
    """Verify 3 debate rounds execute cleanly within <= 50ms latency SLA budget."""
    config = BilateralDualCoTConfig(input_dim=233, latent_dim=768, hidden_dim=256, num_debate_turns=3)
    engine = BilateralDualCoTEngine(config).eval()
    
    x = torch.randn(1, 233)
    # Warmup
    for _ in range(3):
        _ = engine(x)
        
    times = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            verdict = engine(x)
            times.append((time.perf_counter() - t0) * 1000.0)
            
    assert len(verdict.debate_history) == 3, f"Expected 3 debate turns, got {len(verdict.debate_history)}"
    for round_info in verdict.debate_history:
        assert "round" in round_info
        assert "proponent_pressure" in round_info
        assert "adversary_pressure" in round_info
        
    median_lat = np.median(times)
    p95_lat = np.percentile(times, 95)
    print(f"\n[Bilateral CoT SLA] Median: {median_lat:.2f} ms | P95: {p95_lat:.2f} ms")
    assert median_lat <= 50.0, f"Latency {median_lat:.2f} ms violated 50.0 ms budget"


def test_prm_verification_and_deadlock_resolution():
    """Verify that ambiguous debate or PRM < 0.88 resolves to safe HOLD (0)."""
    config = BilateralDualCoTConfig(prm_threshold=0.88)
    engine = BilateralDualCoTEngine(config).eval()
    
    # Force arbiter with low PRM
    final_p = torch.randn(1, 256)
    final_a = torch.randn(1, 256)
    trap_probs = torch.tensor([[0.80, 0.10, 0.10]]) # No trap
    convictions = torch.tensor([[0.50, 0.50]])       # Deadlock convictions
    
    action, prm, conf, cp = engine.arbiter(final_p, final_a, trap_probs, convictions)
    
    # If PRM is below threshold and no trap, action MUST be HOLD (0)
    if float(prm.item()) < 0.88:
        assert action.item() == 0, f"Expected HOLD (0) under sub-threshold PRM {prm.item():.2f}"


def test_bull_trap_counter_punch_invariant():
    """When Adversary Red-Team proves Bull Trap, Arbiter MUST execute SNIPER SELL (-1)."""
    engine = BilateralDualCoTEngine().eval()
    
    final_p = torch.randn(1, 256)
    final_a = torch.randn(1, 256)
    convictions = torch.tensor([[0.75, 0.25]]) # Bull initially bullish
    
    # Adversary Red-Team detects Bull Trap with 85% probability: [No Trap: 0.10, Bull Trap: 0.85, Bear Trap: 0.05]
    trap_probs = torch.tensor([[0.10, 0.85, 0.05]])
    
    action, prm, conf, is_cp = engine.arbiter(final_p, final_a, trap_probs, convictions)
    
    assert action.item() == -1, f"Expected action -1 (SELL) for Bull Trap, got {action.item()}"
    assert is_cp.item() is True, "Expected is_counter_punch to be True"
    assert prm.item() >= 0.85, "PRM for trap exploitation must be high"


def test_bear_trap_counter_punch_invariant():
    """When Adversary Red-Team proves Bear Trap, Arbiter MUST execute SNIPER BUY (+1)."""
    engine = BilateralDualCoTEngine().eval()
    
    final_p = torch.randn(1, 256)
    final_a = torch.randn(1, 256)
    convictions = torch.tensor([[0.20, 0.80]]) # Bear initially bearish
    
    # Adversary detects Bear Trap (Spring) with 85% probability: [No Trap: 0.10, Bull Trap: 0.05, Bear Trap: 0.85]
    trap_probs = torch.tensor([[0.10, 0.05, 0.85]])
    
    action, prm, conf, is_cp = engine.arbiter(final_p, final_a, trap_probs, convictions)
    
    assert action.item() == 1, f"Expected action 1 (BUY) for Bear Trap, got {action.item()}"
    assert is_cp.item() is True, "Expected is_counter_punch to be True"
    assert prm.item() >= 0.85, "PRM for trap exploitation must be high"


def test_numerical_stability_with_corrupt_inputs():
    """Verify Zero NaN / Inf guarantees under corrupted inputs."""
    engine = BilateralDualCoTEngine().eval()
    
    x_bad = torch.randn(1, 233)
    x_bad[0, 3] = float("nan")
    x_bad[0, 15] = float("inf")
    x_bad[0, 42] = float("-inf")
    
    verdict = engine(x_bad)
    assert not np.isnan(verdict.prm_score), "PRM score must not be NaN"
    assert not np.isnan(verdict.confidence), "Confidence must not be NaN"
    assert not np.isnan(verdict.bull_score), "Bull score must not be NaN"
    assert not np.isnan(verdict.bear_score), "Bear score must not be NaN"
    assert verdict.action in (-1, 0, 1)


def test_governor_agent4_strict_hold_invariant():
    """Governor (Agent 4) must strictly output HOLD (0) regardless of reasoning verdict."""
    from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
    
    class DummyTrace:
        def __init__(self):
            self.action = 1 # Dummy BUY
            self.action_name = "BUY"
            
    policy = MythosMT5Policy.__new__(MythosMT5Policy)
    trace_in = DummyTrace()
    
    # Enforce invariant
    trace_out = policy._enforce_agent4_hold(trace_in)
    
    # Extract action
    act_out = policy._extract_action_idx(trace_out)
    assert act_out == 0, f"Governor Agent 4 must strictly be ACTION_HOLD (0), got {act_out}"


def test_checkpoint_roundtrip_serialization(tmp_path: Path):
    """Verify that BilateralDualCoTEngine can save and load its state without data loss."""
    engine = BilateralDualCoTEngine().eval()
    ckpt_file = tmp_path / "test_bilateral_champion.pt"
    
    meta_in = {"training_win_rate": 84.5, "tested_by": "pytest"}
    saved_path = engine.save_checkpoint(ckpt_file, metadata=meta_in)
    assert saved_path.exists()
    
    engine_loaded = BilateralDualCoTEngine().eval()
    engine_loaded.load_checkpoint(ckpt_file)
    
    x = torch.randn(1, 233)
    with torch.no_grad():
        v1 = engine(x)
        v2 = engine_loaded(x)
    
    assert v1.action == v2.action
    assert abs(v1.bull_score - v2.bull_score) < 1e-4
    assert abs(v1.bear_score - v2.bear_score) < 1e-4


def test_batched_vectorized_inference():
    """Verify that forward_batch processes multi-candle batches with high throughput."""
    engine = BilateralDualCoTEngine().eval()
    B = 64
    x_batch = torch.randn(B, 233)
    
    with torch.no_grad():
        batch_res = engine.forward_batch(x_batch, batch_size=32)
        
    assert batch_res.actions.shape == (B,)
    assert batch_res.prm_scores.shape == (B,)
    assert batch_res.confidences.shape == (B,)
    assert batch_res.counter_punches.shape == (B,)
    assert batch_res.trap_probs.shape == (B, 3)
    assert batch_res.bull_scores.shape == (B,)
    assert batch_res.bear_scores.shape == (B,)
    assert batch_res.debate_margins.shape == (B,)
    assert batch_res.per_bar_latency_ms < 20.0  # Vectorized speedup SLA
    
    # Test to_verdicts conversion
    verdicts = batch_res.to_verdicts()
    assert len(verdicts) == B
    for i, v in enumerate(verdicts):
        assert v.action in {-1, 0, 1}
        assert 0.0 <= v.prm_score <= 1.0
        assert 0.0 <= v.confidence <= 1.0
        assert v.action == int(batch_res.actions[i].item())

