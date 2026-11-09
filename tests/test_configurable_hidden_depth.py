from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from self_growing_trader.chuyen_gia_policy import (
    ChuyenGiaPolicy,
    OPENAI_FIVE_RECURRENT_CORE,
    _require_context_expert_architecture_match,
    tao_policy_chuyen_gia_doc_lap,
)
from self_growing_trader.config import INPUT_COUNT, OUTPUT_COUNT, ModelConfig, load_config


def test_active_context_expert_config_declares_five_recurrent_topology() -> None:
    config = load_config("configs/he_chuyen_gia_mot_context_mot_policy.yaml")

    assert config.model.recurrent_core == OPENAI_FIVE_RECURRENT_CORE
    assert config.model.hidden_size == 4096
    assert config.model.hidden_layers == 1
    assert config.model.attention_before_each_hidden is False
    assert config.learning.direct_rnn_rl_auxiliary_enabled is False
    assert config.learning.direct_rnn_rl_temporal_bptt_steps == 16
    assert config.learning.direct_rnn_state_memory_enabled is False


def test_depth_change_broadcasts_or_truncates_regularization_schedule(tmp_path: Path) -> None:
    path = tmp_path / "three_layers.yaml"
    path.write_text(
        """
model:
  hidden_size: 12
  hidden_layers: 3
learning:
  hidden_normalization_layers: [true]
  hidden_dropout_rates: [0.05, 0.15]
""".strip(),
        encoding="utf-8",
    )

    config = load_config(path)
    policy = tao_policy_chuyen_gia_doc_lap(config, seed=72)()
    module = policy.modules[0]

    assert config.learning.hidden_normalization_layers == (True, True, True)
    assert config.learning.hidden_dropout_rates == (0.05, 0.15, 0.15)
    assert module.hidden_size == 12
    assert len(module.deep_w) == 2
    assert module.hidden_dropout_rates == (0.05, 0.15, 0.15)


def test_model_depth_has_layerwise_schedule_limit() -> None:
    with pytest.raises(ValueError, match="model.hidden_layers"):
        ModelConfig(
            input_count=INPUT_COUNT,
            output_count=OUTPUT_COUNT,
            hidden_layers=65,
        )


def test_resume_architecture_mismatch_is_explicit_not_silent(tmp_path: Path) -> None:
    policy = ChuyenGiaPolicy.seed(
        input_size=INPUT_COUNT,
        output_size=OUTPUT_COUNT,
        hidden_size=8,
        hidden_layers=1,
        recurrent_core="legacy_rnn",
        seed=73,
    )
    requested = SimpleNamespace(
        model=SimpleNamespace(
            hidden_size=8, hidden_layers=1, recurrent_core=OPENAI_FIVE_RECURRENT_CORE
        )
    )
    system = SimpleNamespace(experts={"CTX_TEST": SimpleNamespace(policy=policy)})
    with pytest.raises(ValueError, match="Use --fresh-system"):
        _require_context_expert_architecture_match(
            system,
            requested,
            artifact_path="outputs/example.pkl",
        )

