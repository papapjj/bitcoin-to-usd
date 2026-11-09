"""
Empirical verification suite by challenger_m1_3.
Adversarial stress-testing of ChuyenGiaPolicy.apply_five_entry_lifecycle_reward
specifically targeting the elimination of UnboundLocalError when is_extended is True.
"""

from dataclasses import replace
import numpy as np
import pytest

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
from self_growing_trader.protocol_contract import INPUT_COUNT


def _create_test_policy_and_packet(target_agent: int = 0, action: int = 1):
    """Build a functional ChuyenGiaPolicy with one registered entry packet."""
    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=64, hidden_size=32)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    policy._learning_enabled = True
    policy.five_lifecycle_batch_size = 32

    rows = list(policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32)))
    target_id = str(rows[target_agent].trace_id)
    enriched = []
    for index, trace in enumerate(rows):
        details = dict(trace.action_details)
        details.update({
            "five_action_index": action if index == target_agent else 0,
            "five_economic_actor_eligible": bool(index == target_agent),
            "five_execution_filled": bool(index == target_agent),
        })
        enriched.append(replace(trace, action_details=details))

    reg_result = policy.register_five_entry_lifecycle_packet(tuple(enriched))
    assert reg_result["registered"] == 1
    return policy, target_id


class TestChuyenGiaPolicyUnboundLocalElimination:
    """Empirical verification that UnboundLocalError is 100% eliminated under all conditions."""

    @pytest.mark.parametrize("is_extended_flag", [True, False])
    @pytest.mark.parametrize("policy_credit_enabled", [True, False])
    @pytest.mark.parametrize("policy_credit_weight", [0.0, 0.5, 1.0])
    @pytest.mark.parametrize("v4_total", [-4.5, -1.0, -0.750002, -0.75, 0.0, 0.5, 0.75, 0.750002, 1.0, 4.5])
    def test_no_unbound_local_error_across_parameter_matrix(
        self, is_extended_flag, policy_credit_enabled, policy_credit_weight, v4_total
    ):
        policy, trace_id = _create_test_policy_and_packet(target_agent=1, action=1)
        policy.five_lifecycle_net_r_credit_enabled = policy_credit_enabled
        policy.five_lifecycle_net_r_credit_weight = policy_credit_weight

        components = {
            "realized_r": 1.5,
            "quadratic_holding_penalty_r": -0.05,
            "mae_flooding_penalty_r": -0.10,
            "mfe_harvest_reward_r": 0.05,
            "extended_settlement": is_extended_flag,
        }

        # Invocation must succeed without UnboundLocalError or any exception
        result = policy.apply_five_entry_lifecycle_reward(
            trace_id=trace_id,
            lifecycle_reward_r=v4_total,
            realized_r=1.5,
            components=components,
        )

        assert result["accepted"] is True
        assert "net_r_credit_enabled" in result
        assert "net_r_credit_weight" in result
        assert "net_r_credit_r" in result

        is_extended_effective = is_extended_flag or abs(v4_total) > 0.750001
        if is_extended_effective:
            assert result["net_r_credit_enabled"] is False
            assert result["net_r_credit_weight"] == 0.0
            assert result["net_r_credit_r"] == 0.0
            assert result["reward_r"] == pytest.approx(v4_total)
        else:
            assert result["net_r_credit_enabled"] == policy_credit_enabled
            if policy_credit_enabled and policy_credit_weight > 0.0:
                assert result["net_r_credit_weight"] == policy_credit_weight
                assert result["net_r_credit_r"] == pytest.approx(1.5 * policy_credit_weight)
                assert result["reward_r"] == pytest.approx(v4_total + 1.5 * policy_credit_weight)

    @pytest.mark.parametrize("target_agent", [0, 1, 2, 3, 4])
    def test_target_agent_positions(self, target_agent):
        """Verify that target agent at any position (first, middle, last) works cleanly."""
        policy, trace_id = _create_test_policy_and_packet(target_agent=target_agent, action=2)
        components = {"extended_settlement": True}

        result = policy.apply_five_entry_lifecycle_reward(
            trace_id=trace_id,
            lifecycle_reward_r=2.5,
            realized_r=2.0,
            components=components,
        )
        assert result["accepted"] is True
        assert result["agent_id"] == target_agent
        assert result["reward_r"] == pytest.approx(2.5)

    def test_non_finite_inputs_safely_rejected(self):
        """Verify rejection of invalid non-finite numbers before reaching processing."""
        policy, trace_id = _create_test_policy_and_packet(target_agent=1, action=1)
        components = {"extended_settlement": True}

        res_nan = policy.apply_five_entry_lifecycle_reward(
            trace_id=trace_id, lifecycle_reward_r=float("nan"), realized_r=1.0, components=components
        )
        assert res_nan["accepted"] is False
        assert res_nan["reason"] == "invalid_lifecycle_v4"

        # Re-register for inf test
        policy, trace_id = _create_test_policy_and_packet(target_agent=1, action=1)
        res_inf = policy.apply_five_entry_lifecycle_reward(
            trace_id=trace_id, lifecycle_reward_r=float("inf"), realized_r=1.0, components=components
        )
        assert res_inf["accepted"] is False
        assert res_inf["reason"] == "invalid_lifecycle_v4"

        # Re-register for overflow test
        policy, trace_id = _create_test_policy_and_packet(target_agent=1, action=1)
        res_overflow = policy.apply_five_entry_lifecycle_reward(
            trace_id=trace_id, lifecycle_reward_r=5.01, realized_r=1.0, components=components
        )
        assert res_overflow["accepted"] is False
        assert res_overflow["reason"] == "invalid_lifecycle_v4"
