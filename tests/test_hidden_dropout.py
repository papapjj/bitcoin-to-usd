from __future__ import annotations

import numpy as np
import pytest

from self_growing_trader.chuyen_gia_policy import ModuleState
from self_growing_trader.config import LearningIntegrityConfig


def _module(*, seed: int = 41, hidden_layers: int = 6) -> ModuleState:
    return ModuleState.create(
        "dropout_test",
        input_size=8,
        hidden_size=32,
        output_size=5,
        rng=np.random.default_rng(seed),
        hidden_layers=hidden_layers,
    )


def test_six_hidden_blocks_have_layerwise_regularization_slots() -> None:
    module = _module()
    assert 1 + len(module.deep_w) == 6
    assert len(module.hidden_normalization_layers) == 6
    assert len(module.hidden_dropout_rates) == 6
    assert module.hidden_normalization_mode == "layernorm_v1"
    assert module.hidden_normalization_enabled is True
    assert module.hidden_normalization_layers == (True,) * 6
    assert module.hidden_dropout_enabled is True
    assert module.hidden_dropout_rates == (0.05, 0.10, 0.10, 0.05, 0.10, 0.10)


def test_dropout_is_training_only_and_deterministic_per_seed() -> None:
    train_a = _module()
    train_b = _module()
    eval_module = _module()
    for module in (train_a, train_b, eval_module):
        module.hidden_dropout_enabled = True
        module.hidden_dropout_rates = (0.15, 0.20, 0.35, 0.50, 0.65, 0.80)
        module.hidden_dropout_seed = 1234
        module.hidden_dropout_train_only = True

    obs = np.linspace(-1.0, 1.0, 8)
    _, hidden_a, output_a = train_a.activate(obs, training=True)
    _, hidden_b, output_b = train_b.activate(obs, training=True)
    _, hidden_eval, output_eval = eval_module.activate(obs, training=False)

    # The mask is deterministic across cloned candidates but is not applied
    # to the persisted recurrent state itself.
    assert np.allclose(hidden_a, hidden_b)
    assert np.allclose(output_a, output_b)
    assert not np.allclose(output_a, output_eval)
    assert train_a.hidden_dropout_step == 1
    assert eval_module.hidden_dropout_step == 0
    assert train_a.hidden_dropout_rates == (0.15, 0.20, 0.35, 0.50, 0.65, 0.80)
    assert np.isfinite(hidden_a).all() and np.isfinite(output_a).all()


def test_forward_credit_vector_is_the_masked_output_path() -> None:
    module = _module(seed=43)
    module.hidden_dropout_enabled = True
    module.hidden_dropout_rates = (0.15, 0.20, 0.35, 0.50, 0.65, 0.80)
    module.hidden_dropout_seed = 9876
    module.hidden_dropout_train_only = True

    _, recurrent_hidden, _ = module.activate(np.linspace(-1.0, 1.0, 8), training=True)

    # The recurrent state remains unmasked for temporal memory, while the
    # output head and lifecycle/neuron credit must observe the exact masked
    # vector that produced the decision.
    assert module.last_output_hidden is not None
    assert module.last_output_hidden.shape == recurrent_hidden.shape
    assert not np.allclose(module.last_output_hidden, recurrent_hidden)


def test_hidden_regularization_contract_repairs_tampered_module() -> None:
    normalized = _module(seed=42, hidden_layers=6)
    tampered = _module(seed=42, hidden_layers=6)
    tampered.hidden_normalization_mode = "none"
    tampered.hidden_normalization_enabled = False
    tampered.hidden_normalization_layers = (False,) * 6
    tampered.hidden_dropout_enabled = False
    tampered.hidden_dropout_rates = (0.0,) * 6

    obs = np.full(8, 12.0)
    _, hidden_norm, output_norm = normalized.activate(obs)
    _, hidden_repaired, output_repaired = tampered.activate(obs)

    assert tampered.hidden_normalization_mode == "layernorm_v1"
    assert tampered.hidden_normalization_enabled is True
    assert tampered.hidden_normalization_layers == (True,) * 6
    assert tampered.hidden_dropout_enabled is True
    assert all(rate > 0.0 for rate in tampered.hidden_dropout_rates)
    assert np.allclose(hidden_norm, hidden_repaired)
    assert np.allclose(output_norm, output_repaired)
    assert np.isfinite(hidden_norm).all() and np.isfinite(output_norm).all()


@pytest.mark.parametrize(
    ("kwargs", "field"),
    [
        ({"hidden_normalization_mode": "none"}, "hidden_normalization_mode"),
        ({"hidden_normalization_enabled": False}, "hidden_normalization_enabled"),
        ({"hidden_normalization_layers": (True, False)}, "hidden_normalization_layers"),
        ({"hidden_dropout_enabled": False}, "hidden_dropout_enabled"),
        ({"hidden_dropout_rates": (0.10, 0.0)}, "hidden_dropout_rates"),
    ],
)
def test_config_rejects_attempts_to_disable_hidden_regularization(kwargs, field: str) -> None:
    with pytest.raises(ValueError, match=field):
        LearningIntegrityConfig(**kwargs)
