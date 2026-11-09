from __future__ import annotations

import numpy as np

from self_growing_trader.chuyen_gia_policy import ModuleState
from self_growing_trader.graph_plasticity import GraphCreditSignal, GraphPlasticityEngine
from self_growing_trader.neural_graph import EDGE_FORWARD, REGION_CRITIC, REGION_INPUT, NeuralGraph
from self_growing_trader.plasticity import PlasticityCoefficients


def _module(coeffs: PlasticityCoefficients) -> ModuleState:
    module = ModuleState.create("plasticity_test", 4, 3, 2, np.random.default_rng(11))
    module.w_in.fill(0.5)
    module.plasticity_coeffs = coeffs
    module.plasticity_input_normalization_mode = "soft_clip_v1"
    module.plasticity_update_clip = 10.0
    module.plasticity = 1.0
    return module


def test_negative_reward_cannot_reverse_oja_or_regularization() -> None:
    module = _module(
        PlasticityCoefficients(
            eta_grad=0.0,
            eta_hebb=0.0,
            eta_anti_hebb=0.0,
            eta_oja=0.10,
            lambda_reg=0.01,
        )
    )
    before = module.w_in.copy()
    module.apply_plasticity_rules(np.full(4, 20.0), np.ones(3), reward_modulator=-2.0)

    assert np.linalg.norm(module.w_in) < np.linalg.norm(before)
    assert module.plasticity_last_stats["oja_stability_norm"] > 0.0
    assert module.plasticity_last_stats["update_var"] >= 0.0


def test_loss_gradient_keeps_descent_direction_when_reward_is_negative() -> None:
    module = _module(
        PlasticityCoefficients(
            eta_grad=0.10,
            eta_hebb=0.0,
            eta_anti_hebb=0.0,
            eta_oja=0.0,
            lambda_reg=0.0,
        )
    )
    before = module.w_in.copy()
    module.apply_plasticity_rules(
        np.ones(4),
        np.ones(3),
        grad_loss_w=np.ones_like(module.w_in),
        reward_modulator=-2.0,
    )

    assert np.all(module.w_in < before)


def test_plasticity_normalizes_tail_and_tracks_update_variance() -> None:
    module = _module(PlasticityCoefficients())
    module.apply_plasticity_rules(np.full(4, 1_000.0), np.full(3, 50.0), reward_modulator=1.0)
    module.apply_plasticity_rules(np.full(4, -1_000.0), np.full(3, -50.0), reward_modulator=-1.0)

    stats = module.plasticity_last_stats
    assert stats["pre_max_abs"] < 2.0
    assert np.isfinite(module.w_in).all()
    assert np.isfinite(module.plasticity_update_var)
    assert module.plasticity_update_count == 2


def test_repeated_negative_credit_cannot_drift_module_norm() -> None:
    module = _module(PlasticityCoefficients())
    for step in range(200):
        module.apply_plasticity_rules(
            np.full(4, 1_000.0),
            np.ones(3),
            current_time=float(step),
            reward_modulator=-2.0,
        )

    assert np.isfinite(module.w_in).all()
    assert np.linalg.norm(module.w_in) <= module.plasticity_weight_norm_cap + 1e-9
    assert float(np.max(np.abs(module.w_in))) < 1.0


def test_graph_negative_credit_keeps_oja_correction_stable() -> None:
    graph = NeuralGraph()
    src = graph.add_neuron(REGION_INPUT, module_id="m1", lifecycle="active")
    dst = graph.add_neuron(REGION_CRITIC, module_id="critic", lifecycle="candidate")
    edge_id = graph.add_connection(src, dst, weight=0.5, gate=0.0, kind=EDGE_FORWARD, candidate_id="cand")
    graph.activate(np.asarray([1.0]))
    engine = GraphPlasticityEngine(
        gate_lr=0.0,
        weight_lr=1.0,
        coeffs=PlasticityCoefficients(eta_grad=0.0, eta_hebb=0.0, eta_anti_hebb=0.0, eta_oja=0.1, lambda_reg=0.0),
    )

    report = engine.apply_credit(
        graph,
        GraphCreditSignal(reward=-1.0),
        active_edge_ids=(edge_id,),
        candidate_mode=True,
    )

    assert graph.edges[edge_id].weight < 0.5
    assert report.oja_stability_abs_sum > 0.0
