from __future__ import annotations

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_new_policies_have_fixed_topology_without_alpha_or_develop_hook() -> None:
    for policy in (ChuyenGiaPolicy.seed(seed=71), SelfGrowingPolicy.seed(seed=71)):
        assert not hasattr(policy, "develop")
        assert all(not hasattr(module, "alpha_gate") for module in policy.modules)


def test_legacy_alpha_gate_is_folded_once_into_encoder_weight() -> None:
    policy = ChuyenGiaPolicy.seed(seed=72)
    module = policy.modules[0]
    original = module.w_in.copy()
    legacy_alpha = np.full_like(original, -0.7)
    module.alpha_gate = legacy_alpha

    policy.ensure_compatible()

    expected = original / (1.0 + np.exp(0.7))
    np.testing.assert_allclose(module.w_in, expected)
    assert not hasattr(module, "alpha_gate")

    # Re-running migration must not apply the factor a second time.
    policy.ensure_compatible()
    np.testing.assert_allclose(module.w_in, expected)
