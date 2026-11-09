from __future__ import annotations

import numpy as np

from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_m1_graph_module_forward_matches_legacy_api_shape() -> None:
    module = M1GraphModule.seed(input_size=4, output_size=3, state_size=5, K=3, seed=7)

    output, error, novelty, conflict, confidence = module.forward(np.ones(4))

    assert output.shape == (3,)
    assert np.isfinite(output).all()
    assert error >= 0.0
    assert 0.0 <= novelty <= 2.0
    assert 0.0 <= conflict <= 1.0
    assert 0.0 <= confidence <= 1.0


def test_m1_graph_module_uses_action_bias_and_copy_is_deep() -> None:
    module = M1GraphModule.seed(input_size=4, output_size=3, state_size=5, K=3, seed=8)
    x = np.array([0.2, -0.1, 0.3, -0.4])

    baseline = module.forward(x)[0]
    biased = module.forward(x, context={"action_bias": np.array([1.0, -1.0, 0.0])})[0]

    assert not np.allclose(baseline, biased)

    copied = module.copy()
    edge_id = next(iter(copied.graph.edges))
    copied.graph.edges[edge_id].weight += 5.0
    assert not np.isclose(copied.graph.edges[edge_id].weight, module.graph.edges[edge_id].weight)


def test_m1_graph_module_reset_and_resize_compatibility() -> None:
    module = M1GraphModule.seed(input_size=4, output_size=3, state_size=5, K=3, seed=9)

    module.forward(np.ones(4))
    assert module.prev_output is not None

    module.reset_state()
    assert module.prev_output is None

    module.resize_hidden(7)
    assert module.state_size == 7
    assert module.forward(np.ones(4))[0].shape == (3,)


def test_high_account_stress_suppresses_graph_entry_logits_without_overriding_other_actions() -> None:
    
    policy = SelfGrowingPolicy.seed(input_size=6, output_size=20, hidden_size=5, seed=711)
    policy.convert_m1_to_graph_backend()
    baseline = policy.modules[0].m1_component.copy()
    stressed = policy.modules[0].m1_component.copy()
    observation = np.linspace(-0.2, 0.3, 6)

    low, *_ = baseline.forward(observation, context={"account_stress": 0.0})
    high, *_ = stressed.forward(observation, context={"account_stress": 1.0})

    assert np.mean(high[1:7]) < np.mean(low[1:7])
    assert stressed.last_context_summary["entry_gate"] < 0.2
    assert high.shape == low.shape == (20,)

def test_uncertainty_and_legacy_opportunity_cost_do_not_create_live_entry_probe() -> None:
    observation = np.linspace(-0.2, 0.3, 6)
    context = {
        "policy_entry_pressure": 1.0,
        "opportunity_cost": 1.0,
        "missed_opportunity": 1.0,
        "uncertainty": 0.8,
        "account_stress": 0.0,
    }
    module = M1GraphModule.seed(input_size=6, output_size=20, state_size=5, K=4, seed=10)
    output, *_ = module.forward(observation, context=context)

    # V8 permits no generic BUY/SELL proposal from scarcity, uncertainty or a
    # legacy opportunity_cost field. A naturally decoded graph action may still
    # happen to be an entry; the contract concerns the generator contribution.
    summary = module.last_action_generator_summary
    assert output.shape == (20,)
    assert summary["probe_strength"] == 0.0
    assert summary["probe_gain"] == 0.0
    assert summary["market_opportunity"] == 0.0
    assert summary["curiosity_live_action_delta"] == 0.0

    stressed = M1GraphModule.seed(input_size=6, output_size=20, state_size=5, K=4, seed=10)
    stressed_output, *_ = stressed.forward(observation, context={**context, "account_stress": 1.0})
    assert stressed_output.shape == (20,)
    assert stressed.last_context_summary["entry_gate"] < 0.2


def test_policy_records_runtime_starvation_as_learning_drive_without_forcing_entry() -> None:
    policy = SelfGrowingPolicy.seed(input_size=6, output_size=20, hidden_size=5, seed=20, m1_backend="graph")
    policy.set_experience_starvation({
        "no_trade_streak": 8.0,
        "unfilled_streak": 0.0,
        "cadence_deficit": 0.0,
        "starvation": 0.32,
        "intrinsic_reward": 0.0,
    })
    policy.set_account_context({
        "current_equity_drawdown": 0.0,
        "open_risk_ratio": 0.0,
        "equity_volatility": 0.0,
    })

    trace = policy.act(np.linspace(-0.2, 0.3, 6))

    assert trace.experience_starvation["runtime_starvation"] > 0.0
    assert np.max(trace.learning_drive_context) > 0.0
    # No-trade scarcity is a learning-drive signal, not fabricated BUY/SELL evidence.
    assert float(trace.action_details.get("market_opportunity", 0.0)) == 0.0

