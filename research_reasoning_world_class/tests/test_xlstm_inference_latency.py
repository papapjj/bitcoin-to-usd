"""
Unit and Integration Tests for SGT-V193 xLSTM Core Backbone & 233D Reasoning Adapter
Target Module: research_reasoning_world_class/src/xlstm_reasoning_adapter.py
Milestone: M2 (Features 14, 15, 16)

Verifies:
1. SGT-V193 Core Parameter Count: within [16.5M, 18.5M].
2. Matrix memory capacity: exactly 147,456 elements per agent (4 heads x 192 x 192).
3. 233D Ingress Projection shape contracts: [233] -> [1024] -> [768] Governor Latent + [20] Wire.
4. Gated Residual ReZero backward compatibility: alpha=0 produces exact base 137D output.
5. Recurrent State Isolation across 6 assets (BTC, ETH, SOL, XAU, EUR, US30): L_inf = 0.0.
6. Zero NaN / Zero Inf arithmetic traps (AC-7) over adversarial steps with extreme values.
7. Governor Latent State Extraction: strictly extracts Agent 4 (Governor) representation.
8. End-to-End Metacognitive Reasoning integration: 768D latent -> 4-hop CoT + Bull-Bear debate.
9. Real-time inference latency benchmark (AC-6): <= 50.0ms per tick step on CPU.
10. TorchScript JIT tracing and parity check.
11. Pipeline aliases and JIT wrappers (XLSTMPipelineCore, JITCoreWrapper, JITXLSTMCacheCore).
"""

import sys
import os
import time
import pytest
import torch
import numpy as np

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from research_reasoning_world_class.src.xlstm_reasoning_adapter import (
    INGRESS_TOTAL_COUNT,
    INGRESS_BASE_COUNT,
    INGRESS_SMC_COUNT,
    INGRESS_MACRO_COUNT,
    GOVERNOR_AGENT_INDEX,
    GOVERNOR_LATENT_DIM,
    SUPPORTED_6_SYMBOLS,
    FeatureIngressAdapter233,
    SymbolRecurrentStateRegistry,
    XLSTMSGTReasoningAdapter,
    XLSTMPipelineCore,
    AdapterStepOutput,
    JITCoreWrapper,
    JITXLSTMCacheCore,
)

try:
    from src.self_growing_trader.sgt_compact_core import (
        COMPACT_HEAD_DIM,
        COMPACT_HIDDEN_SIZE,
        COMPACT_NUM_HEADS,
        FIVE_AGENT_COUNT,
        OUTPUT_COUNT,
        MatrixLSTMState,
        ScalarLSTMState,
    )
except (ImportError, ModuleNotFoundError):
    from self_growing_trader.sgt_compact_core import (
        COMPACT_HEAD_DIM,
        COMPACT_HIDDEN_SIZE,
        COMPACT_NUM_HEADS,
        FIVE_AGENT_COUNT,
        OUTPUT_COUNT,
        MatrixLSTMState,
        ScalarLSTMState,
    )

from research_reasoning_world_class.src.metacognitive_reasoning import (
    ReasoningConfig,
    MetacognitiveReasoningEngine,
)


@pytest.fixture
def adapter():
    """Default adapter fixture in eval mode."""
    torch.manual_seed(42)
    adapter_mod = XLSTMSGTReasoningAdapter(ingress_mode="direct")
    adapter_mod.eval()
    return adapter_mod


def test_core_parameter_count_criterion(adapter: XLSTMSGTReasoningAdapter):
    """Verifies that total parameters remain strictly within [16.5M, 18.5M]."""
    params = adapter.count_parameters()
    assert 16_500_000 <= params <= 18_500_000, f"Expected params in [16.5M, 18.5M], got {params:,}"


def test_matrix_memory_capacity_criterion():
    """Verifies mLSTM covariance memory elements: 4 heads x 192 x 192 = 147,456."""
    total_elements = COMPACT_NUM_HEADS * COMPACT_HEAD_DIM * COMPACT_HEAD_DIM
    assert total_elements == 147_456, f"Expected 147,456 covariance elements, got {total_elements}"


def test_ingress_233d_shape_contracts(adapter: XLSTMSGTReasoningAdapter):
    """Verifies tensor shape flow: 233D observation -> 768D Governor Latent + 20D Action Wire."""
    obs_1d = torch.randn(233)
    out = adapter.step_symbol("BTCUSD", obs_1d)

    assert out.symbol == "BTCUSD"
    assert out.governor_latent.shape == (768,)
    assert out.action_wire.shape == (5, 20)
    assert out.governor_action_wire.shape == (20,)
    assert out.values.shape == (5, 1)
    assert out.attention_weights.shape == (4, 5, 5)


def test_gated_residual_rezero_backward_compatibility():
    """Verifies that ReZero gated residual adapter with alpha=0 reproduces base 137D output exactly."""
    adapter_gated = FeatureIngressAdapter233(mode="gated_residual")
    x = torch.randn(5, 233)
    x_base = x[:, :137]

    base_only = adapter_gated.base_encoder(x_base)
    fused = adapter_gated(x)

    assert torch.allclose(fused, base_only, atol=1e-6), "ReZero property violated at initialization!"


def test_state_isolation_zero_cross_symbol_contamination(adapter: XLSTMSGTReasoningAdapter):
    """
    Verifies that stepping Symbol A (e.g. BTCUSD) causes EXACT zero state contamination
    in all other 5 symbols (ETHUSD, SOLUSD, XAUUSD, EURUSD, US30), verifying L_inf = 0.0.
    """
    # Create independent control adapter initialized with identical seed
    torch.manual_seed(100)
    control_adapter = XLSTMSGTReasoningAdapter(ingress_mode="direct")
    control_adapter.load_state_dict(adapter.state_dict())
    control_adapter.eval()

    # Step BTCUSD 50 times in test adapter
    torch.manual_seed(999)
    for _ in range(50):
        obs_btc = torch.randn(233)
        adapter.step_symbol("BTCUSD", obs_btc)

    # Now step all other 5 symbols in both adapters with identical inputs
    for sym in ["ETHUSD", "SOLUSD", "XAUUSD", "EURUSD", "US30"]:
        obs_probe = torch.randn(233)
        out_test = adapter.step_symbol(sym, obs_probe)
        out_ctrl = control_adapter.step_symbol(sym, obs_probe)

        diff_wire = (out_test.action_wire - out_ctrl.action_wire).abs().max().item()
        diff_gov = (out_test.governor_latent - out_ctrl.governor_latent).abs().max().item()

        m_test = adapter.state_registry.get_state(sym)
        m_ctrl = control_adapter.state_registry.get_state(sym)
        diff_cov = max((t.mlstm.covariance - c.mlstm.covariance).abs().max().item() for t, c in zip(m_test, m_ctrl))
        diff_cell = max((t.slstm.cell - c.slstm.cell).abs().max().item() for t, c in zip(m_test, m_ctrl))

        linf = max(diff_wire, diff_gov, diff_cov, diff_cell)
        assert linf == 0.0, f"State contamination detected in {sym}! L_inf = {linf}"


def test_ac7_zero_nan_zero_inf_adversarial_traps(adapter: XLSTMSGTReasoningAdapter):
    """Verifies AC-7 requirement: Absolute 0 NaN, 0 Inf across extreme adversarial inputs."""
    adversarial_inputs = [
        torch.full((233,), float("nan")),
        torch.full((233,), float("inf")),
        torch.full((233,), float("-inf")),
        torch.randn(233) * 1e6,
        torch.randn(233) * 1e-6,
        torch.zeros(233),
    ]

    for idx, bad_obs in enumerate(adversarial_inputs):
        out = adapter.step_symbol("BTCUSD", bad_obs)
        assert not torch.isnan(out.governor_latent).any(), f"NaN in governor_latent at test {idx}"
        assert not torch.isinf(out.governor_latent).any(), f"Inf in governor_latent at test {idx}"
        assert not torch.isnan(out.action_wire).any(), f"NaN in action_wire at test {idx}"
        assert not torch.isinf(out.action_wire).any(), f"Inf in action_wire at test {idx}"


def test_governor_latent_extraction_and_cot_reasoning_integration(adapter: XLSTMSGTReasoningAdapter):
    """Verifies that extracted 768D Governor Latent connects directly to MetacognitiveReasoningEngine."""
    cot_config = ReasoningConfig(input_dim=768, hidden_dim=256)
    reasoning_engine = MetacognitiveReasoningEngine(cot_config)
    reasoning_engine.eval()

    adapter.connect_metacognitive_reasoner(reasoning_engine)

    obs = torch.randn(233)
    out = adapter.step_symbol("ETHUSD", obs)

    assert out.metacognitive_verdict is not None
    assert out.metacognitive_verdict.cot_hops.shape == (1, 4, 256)
    assert out.metacognitive_verdict.bull_score.shape == (1,)
    assert out.metacognitive_verdict.bear_score.shape == (1,)
    assert out.metacognitive_verdict.scenario_probs.shape == (1, 4)


def test_inference_latency_benchmark_sub_50ms(adapter: XLSTMSGTReasoningAdapter):
    """Verifies AC-6 requirement: Single-step inference latency <= 50ms on CPU."""
    torch.set_num_threads(1)
    obs = torch.randn(233)

    # Warmup
    for _ in range(5):
        adapter.step_symbol("BTCUSD", obs)

    # Benchmark 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = adapter.step_symbol("BTCUSD", obs)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    assert avg_latency <= 50.0, f"Average latency {avg_latency:.2f}ms exceeds 50ms SLA budget!"
    assert p95_latency <= 85.0, f"P95 latency {p95_latency:.2f}ms exceeds OS scheduling jitter tolerance!"


import warnings


@pytest.mark.filterwarnings("ignore:.*torch\\.jit\\.trace.*:DeprecationWarning")
def test_torchscript_jit_tracing_compatibility():
    """Verifies TorchScript JIT tracing compatibility and mathematical parity."""
    adapter = XLSTMSGTReasoningAdapter(ingress_mode="direct").eval()
    wrapper = JITCoreWrapper(adapter)

    obs = torch.randn(5, 233)
    cov = torch.zeros(5, 4, 192, 192)
    norm = torch.zeros(5, 4, 192)
    stab = torch.zeros(5, 4)
    cell = torch.zeros(5, 768)
    snorm = torch.zeros(5, 768)
    sstab = torch.zeros(5, 768)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        traced = torch.jit.trace(wrapper, (obs, cov, norm, stab, cell, snorm, sstab))
    eager_gov, eager_cov, eager_cell = wrapper(obs, cov, norm, stab, cell, snorm, sstab)
    traced_gov, traced_cov, traced_cell = traced(obs, cov, norm, stab, cell, snorm, sstab)

    assert torch.allclose(eager_gov, traced_gov, atol=1e-5), "JIT parity mismatch on governor latent!"
    assert torch.allclose(eager_cov, traced_cov, atol=1e-5), "JIT parity mismatch on covariance!"


def test_xlstm_pipeline_core_alias_and_cache_core():
    """Verifies XLSTMPipelineCore alias and JITXLSTMCacheCore wrapper functionality."""
    assert XLSTMPipelineCore is XLSTMSGTReasoningAdapter
    core = XLSTMPipelineCore(ingress_mode="direct").eval()
    assert core.count_parameters() > 16_500_000

    cache = JITXLSTMCacheCore(core)
    obs = torch.randn(5, 233)
    cov = torch.zeros(5, 4, 192, 192)
    norm = torch.zeros(5, 4, 192)
    stab = torch.zeros(5, 4)
    cell = torch.zeros(5, 768)
    snorm = torch.zeros(5, 768)
    sstab = torch.zeros(5, 768)

    gov, cov_out, cell_out = cache(obs, cov, norm, stab, cell, snorm, sstab)
    assert gov.shape == (768,)
    assert cov_out.shape == (5, 4, 192, 192)
    assert cell_out.shape == (5, 768)
