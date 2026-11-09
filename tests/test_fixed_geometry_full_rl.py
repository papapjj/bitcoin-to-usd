from __future__ import annotations

from dataclasses import replace
import pickle

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.execution_contract import RR_OUTPUT_INDEX, SL_OUTPUT_INDEX


def _full_rl_policy(seed: int) -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy.seed(
        input_size=8, output_size=20, hidden_size=4, seed=seed, m1_backend="direct_rnn",
    )
    policy.direct_rnn_full_rl_enabled = True
    policy.direct_rnn_adamw_enabled = True
    policy._learning_enabled = True
    policy.execution_fixed_sl_atr = 2.5
    policy.execution_fixed_tp_atr = 5.0
    policy.ensure_compatible()
    return policy


def test_full_rl_updates_entry_and_value_but_freezes_compatibility_geometry_rows() -> None:
    policy = _full_rl_policy(7)
    trace = policy.act(np.zeros(8, dtype=float))
    selected = trace.selected_action.copy()
    selected[0] = -1.0
    selected[1] = 1.0
    trace = replace(trace, selected_action=selected)
    module = policy.modules[0]
    module.rnn_optimizer_m["w_out"] = np.ones_like(module.w_out)
    module.rnn_optimizer_v["w_out"] = np.ones_like(module.w_out)
    module.rnn_optimizer_m["b_out"] = np.ones_like(module.b_out)
    module.rnn_optimizer_v["b_out"] = np.ones_like(module.b_out)
    fixed_before = module.w_out[[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]].copy()
    fixed_bias_before = module.b_out[[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]].copy()
    entry_before = module.w_out[1].copy()

    updated = policy._apply_direct_rnn_full_rl_batch([{
        "trace": trace, "outcome": 0.5, "advantage": 0.5, "update_scale": 1.0,
    }])

    event = policy.growth_events[-1]
    assert updated == 1
    assert np.linalg.norm(module.w_out[1] - entry_before) > 0.0
    assert np.array_equal(module.w_out[[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]], fixed_before)
    assert np.array_equal(module.b_out[[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]], fixed_bias_before)
    assert np.all(module.rnn_optimizer_m["w_out"][[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]] == 0.0)
    assert np.all(module.rnn_optimizer_v["w_out"][[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]] == 0.0)
    assert np.all(module.rnn_optimizer_m["b_out"][[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]] == 0.0)
    assert np.all(module.rnn_optimizer_v["b_out"][[SL_OUTPUT_INDEX, RR_OUTPUT_INDEX]] == 0.0)
    assert event["fixed_geometry"]["mode"] == "fixed_atr_v1"
    assert event["fixed_geometry"]["compatibility_output_row_delta_norms"][module.name] == {
        "sl_row": 0.0, "rr_row": 0.0, "sl_bias": 0.0, "rr_bias": 0.0,
    }


def test_legacy_20_output_policy_loads_into_fixed_geometry_mode() -> None:
    policy = _full_rl_policy(11)
    policy.direct_rnn_rl_geometry_enabled = True
    policy.direct_rnn_rl_geometry_loss_weight = 0.2
    restored = pickle.loads(pickle.dumps(policy))
    restored.ensure_compatible()
    assert restored.output_size == 20
    assert restored.modules[0].w_out.shape[0] == 20
    assert restored.execution_fixed_sl_atr == 2.5
    assert restored.execution_fixed_tp_atr == 5.0
    assert not hasattr(restored, "direct_rnn_rl_geometry_enabled")
    assert not hasattr(restored, "direct_rnn_rl_geometry_loss_weight")
