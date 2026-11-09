from __future__ import annotations

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.nhat_ky_giai_thich import _responsibility
from self_growing_trader.protocol_contract import INPUT_COUNT, OUTPUT_COUNT
from self_growing_trader.graph_supernetwork_runtime import summarize_graph_trial_history
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_graph_observability_metrics_are_derived_from_trace_history() -> None:
    policy = SelfGrowingPolicy.seed(input_size=4, output_size=3, hidden_size=5, seed=401)
    policy.convert_m1_to_graph_backend()
    first = policy.act(np.array([0.2, -0.1, 0.3, 0.0]))
    policy.observe_transition(first, np.array([0.1, 0.0, 0.2, 0.1]), reward=0.3, cost=0.02, done=False)
    second = policy.act(np.array([-0.3, 0.4, 0.0, 0.1]))
    policy.observe_transition(second, np.array([-0.1, 0.2, 0.1, 0.0]), reward=0.1, cost=0.03, done=False)

    metrics = policy.graph_observability_metrics()
    assert set(metrics) == {
        "prediction_error_mean",
        "prediction_error_trend",
        "memory_usefulness",
        "selector_switch_rate",
        "credit_active_neurons",
        "credit_active_edges",
    }
    assert metrics["credit_active_neurons"] > 0
    assert metrics["credit_active_edges"] > 0
    assert 0.0 <= metrics["memory_usefulness"] <= 1.0


def test_graph_trial_summary_includes_observability_and_acceptance_metrics() -> None:
    summary = summarize_graph_trial_history([
        {
            "promoted": True,
            "graph_candidates": [{"role": "predictor_specialist", "reason": "high_prediction_error"}],
            "graph_observability": {
                "prediction_error_mean": 0.4,
                "prediction_error_trend": -0.1,
                "memory_usefulness": 0.6,
                "selector_switch_rate": 0.2,
                "credit_active_neurons": 5.0,
                "credit_active_edges": 7.0,
            },
        }
    ])
    assert summary["growth_accept_rate"] == 1.0
    assert summary["module_role_accept_rate:predictor_specialist"] == 1.0
    assert summary["cluster_to_module_accept_rate:high_prediction_error"] == 1.0
    assert summary["prediction_error_trend"] == -0.1
    assert summary["credit_active_edges"] == 7.0


def test_direct_rnn_feature_attribution_uses_actual_input_values_not_nested_graph_maps() -> None:
    policy = ChuyenGiaPolicy.seed(input_size=INPUT_COUNT, output_size=OUTPUT_COUNT, hidden_size=8, seed=405)
    trace = policy.act(np.linspace(-0.8, 0.8, INPUT_COUNT))
    _neurons, _edges, features = _responsibility(policy, trace, max_neurons=0, max_edges=0)
    assert features
    assert any(abs(float(item["gia_tri"])) > 0.01 for item in features)
    assert all(item["muc_do_quy_trach"] == "xap_xi_local_w_in_action_margin" for item in features)
