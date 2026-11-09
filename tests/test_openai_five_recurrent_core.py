from __future__ import annotations

import numpy as np
import pytest
import torch

from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    ModuleState,
    OPENAI_FIVE_RECURRENT_CORE,
)
from self_growing_trader.config import load_config
from self_growing_trader.openai_five_lstm_core import lstm_step


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def test_lstm_step_matches_standard_gate_equations() -> None:
    x = np.array([0.25, -0.5, 0.75], dtype=np.float32)
    h0 = np.array([0.2, -0.1], dtype=np.float32)
    c0 = np.array([0.4, -0.3], dtype=np.float32)
    w_in = np.arange(8 * 3, dtype=np.float32).reshape(8, 3) / 50.0 - 0.2
    w_rec = np.arange(8 * 2, dtype=np.float32).reshape(8, 2) / 40.0 - 0.15
    bias = np.linspace(-0.1, 0.1, 8, dtype=np.float32)

    got = lstm_step(x, h0, c0, w_in, w_rec, bias)
    z = w_in @ x + w_rec @ h0 + bias
    i_raw, f_raw, g_raw, o_raw = np.split(z, 4)
    i = _sigmoid(i_raw)
    f = _sigmoid(f_raw)
    g = np.tanh(g_raw)
    o = _sigmoid(o_raw)
    c = f * c0 + i * g
    h = o * np.tanh(c)

    np.testing.assert_allclose(got.input_gate, i, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.forget_gate, f, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.candidate, g, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.output_gate, o, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.cell, c, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(got.hidden, h, rtol=1e-6, atol=1e-6)


def test_five_module_has_one_lstm_and_no_legacy_recurrent_stack() -> None:
    module = ModuleState.create(
        "five",
        input_size=7,
        hidden_size=8,
        output_size=5,
        rng=np.random.default_rng(7),
        hidden_layers=1,
        recurrent_core=OPENAI_FIVE_RECURRENT_CORE,
    )
    assert module.lstm_w_in.shape == (32, 7)
    assert module.lstm_w_rec.shape == (32, 8)
    assert module.lstm_b.shape == (32,)
    assert module.lstm_cell.shape == (8,)
    assert module.w_in.shape == (0, 0)
    assert module.w_rec.shape == (0, 0)
    assert module.w_lat is None
    assert module.deep_w == ()
    assert module.deep_rec == ()
    assert module.deep_b == ()
    assert module.attention_enabled is False
    assert module.hidden_normalization_enabled is False
    assert module.hidden_dropout_enabled is False
    assert module.p_stp is None


def test_five_forward_uses_cell_state_and_linear_policy_projection() -> None:
    module = ModuleState.create(
        "five",
        input_size=4,
        hidden_size=6,
        output_size=3,
        rng=np.random.default_rng(11),
        hidden_layers=1,
        recurrent_core=OPENAI_FIVE_RECURRENT_CORE,
    )
    x = np.array([0.3, -0.2, 0.1, 0.7], dtype=np.float32)
    gate, hidden, output = module.activate(x, np.zeros(6, dtype=np.float32))
    assert gate == 1.0
    trace = module.last_forward_trace
    assert trace is not None
    assert trace.recurrent_core == OPENAI_FIVE_RECURRENT_CORE
    assert trace.lstm_forward is not None
    np.testing.assert_allclose(module.lstm_cell, trace.lstm_forward.cell)
    expected = module.w_out @ np.asarray(hidden, dtype=np.float32) + module.b_out
    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-6)


def test_five_training_path_uses_shared_network_not_legacy_modules() -> None:
    from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy

    policy = ChuyenGiaPolicy(input_size=137, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    assert policy.modules == []
    assert policy.five_shared_network.lstm.hidden_size == 16
    assert policy.five_parameter_count() > 0


def test_five_core_fixed_topology_has_one_recurrent_layer() -> None:
    from self_growing_trader.five_shared_policy import (
        PostLstmCrossAgentTransformer,
        SharedFiveTradingPolicy,
    )

    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    assert network.lstm.hidden_size == 16
    assert isinstance(network.post_lstm_transformer, PostLstmCrossAgentTransformer)
    assert isinstance(
        network.post_lstm_transformer.encoder_layer,
        torch.nn.TransformerEncoderLayer,
    )
    assert not hasattr(network, "lstm2")
    assert not hasattr(network, "deep_layers")


def test_post_lstm_transformer_production_size_is_bounded_for_lstm_4096() -> None:
    from self_growing_trader.five_shared_policy import PostLstmCrossAgentTransformer

    block = PostLstmCrossAgentTransformer(4096)
    assert block.model_width == 512
    assert block.head_count == 8
    assert block.ffn_size == 2048
    assert sum(parameter.numel() for parameter in block.parameters()) == 12_610_050


def test_post_lstm_transformer_attention_mixes_five_agent_tokens() -> None:
    from self_growing_trader.five_shared_policy import PostLstmCrossAgentTransformer

    torch.manual_seed(17)
    block = PostLstmCrossAgentTransformer(12)
    hidden = torch.randn(5, 12, requires_grad=True)
    block(hidden)[1].sum().backward()
    assert hidden.grad is not None
    assert float(hidden.grad[0].abs().sum().item()) > 0.0


def test_temporal_tokens_are_causal_and_change_the_current_representation() -> None:
    from self_growing_trader.five_shared_policy import PostLstmCrossAgentTransformer

    torch.manual_seed(23)
    block = PostLstmCrossAgentTransformer(12)
    current = torch.randn(5, 12)
    empty, valid = block.empty_temporal_memory(device=current.device, dtype=current.dtype)
    without_history = block(current, empty, valid)
    remembered, remembered_valid = block.update_temporal_memory(
        torch.randn(5, 12), empty, valid
    )
    with_history = block(current, remembered, remembered_valid)

    assert remembered_valid.all()
    assert not torch.equal(with_history, without_history)


def test_temporal_tokens_reset_per_agent_at_episode_boundary() -> None:
    from self_growing_trader.five_shared_policy import (
        FIVE_AGENT_COUNT,
        SharedFiveTradingPolicy,
    )
    from self_growing_trader.protocol_contract import INPUT_COUNT

    torch.manual_seed(27)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    observations = torch.randn(2, FIVE_AGENT_COUNT, INPUT_COUNT)
    dones = torch.zeros((2, FIVE_AGENT_COUNT), dtype=torch.bool)
    dones[0, 0] = True
    memories = network.advance_history(observations, dones=dones)

    assert memories[0].temporal_valid is not None
    assert memories[0].temporal_valid.all()
    # Agent 0 was reset before the second observation, so every EMA horizon
    # starts from the same current token. Other agents retain two-step EMAs.
    tokens0 = memories[0].temporal_tokens
    assert tokens0 is not None
    torch.testing.assert_close(
        tokens0, tokens0[0].expand_as(tokens0), rtol=0.0, atol=0.0
    )
    tokens1 = memories[1].temporal_tokens
    assert tokens1 is not None
    assert not torch.equal(tokens1[0], tokens1[-1])


def test_v2_transformer_migration_is_output_neutral_until_temporal_training() -> None:
    from self_growing_trader.five_shared_policy import PostLstmCrossAgentTransformer

    torch.manual_seed(29)
    block = PostLstmCrossAgentTransformer(12)
    hidden = torch.randn(5, 12)
    projected = block.input_projection(hidden.unsqueeze(0))
    expected = hidden.unsqueeze(0) + block.residual_scale * block.output_projection(
        block.encoder_layer(projected)
    )
    for name in (
        "temporal_encoder_layer", "temporal_output_projection",
        "temporal_horizon_embedding", "temporal_agent_embedding",
    ):
        block._modules.pop(name)
    block._parameters.pop("temporal_scale")
    block._buffers.pop("temporal_decay")

    assert block.require_temporal_architecture() is True
    assert float(block.temporal_scale.item()) == 0.0
    torch.testing.assert_close(block(hidden), expected.squeeze(0), rtol=0.0, atol=0.0)


def test_old_policy_architecture_fails_fast_without_transformer_migration() -> None:
    from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy

    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.policy_architecture_version = 1
    with pytest.raises(RuntimeError, match="checkpoint Five cu"):
        network.require_current_architecture()


def test_production_profile_declares_five_width_and_trading_unroll_override() -> None:
    config = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")
    assert config.model.recurrent_core == OPENAI_FIVE_RECURRENT_CORE
    assert config.model.hidden_size == 4096
    assert config.model.hidden_layers == 1
    assert config.model.attention_before_each_hidden is False
    assert config.learning.direct_rnn_rl_temporal_bptt_steps == 16

    from self_growing_trader.five_shared_policy import public_profile_audit

    audit = public_profile_audit()
    assert audit["published_lstm_unroll"] == 16
    assert audit["lstm_unroll"] == 64
    assert audit["post_lstm_transformer_layers"] == 1
    assert audit["post_lstm_transformer_width"] == 512
    assert audit["post_lstm_transformer_heads"] == 8
    assert audit["post_lstm_transformer_ffn_size"] == 2048
    assert audit["temporal_token_horizons"] == [2, 8, 32, 128]
    assert audit["temporal_token_count_per_agent"] == 4
