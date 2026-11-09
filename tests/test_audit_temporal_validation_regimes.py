import numpy as np
import pytest

from scripts.audit_temporal_validation_regimes import match_rows


def test_cache_matching_uses_step_major_order_and_owner_mask():
    times = np.array(["a", "a", "b", "c", "d", "d"])
    roles = np.array([0, 1, 0, 1, 0, 1])
    sampled = {"sample_times": np.full((2, 130), "unused", dtype="U6"),
               "owner_masks": np.zeros((2, 130, 5), dtype=bool)}
    sampled["sample_times"][:, 128:] = [["a", "b"], ["c", "d"]]
    sampled["owner_masks"][0, 128, :2] = True
    sampled["owner_masks"][0, 129, 0] = True
    sampled["owner_masks"][1, 128, 1] = True
    sampled["owner_masks"][1, 129, :2] = True
    np.testing.assert_array_equal(match_rows(times, roles, sampled, 0), [0, 2, 4])
    np.testing.assert_array_equal(match_rows(times, roles, sampled, 1), [1, 3, 5])
    with pytest.raises(ValueError, match="duplicate"):
        match_rows(np.concatenate([times, times[:1]]), np.concatenate([roles, roles[:1]]), sampled, 0)
