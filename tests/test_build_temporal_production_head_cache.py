from types import SimpleNamespace
import numpy as np
import pytest

from scripts.build_temporal_production_head_cache import aligned_rows


def test_selected_windows_are_unique_full_and_production_aligned():
    times = np.repeat(np.arange(650), 2)
    data = SimpleNamespace(sample_times=times, actions=np.zeros(len(times)))
    rows, windows = aligned_rows(data, 3)
    assert windows.tolist() == [0, 4, 9]
    assert len(rows) == len(np.unique(rows)) == 3 * 64 * 2
    assert times[rows].reshape(3, 128)[:, 0].tolist() == [0, 256, 576]
    assert times[rows][-1] == 639  # Incomplete final window excluded.
    all_rows, _ = aligned_rows(data, 100)
    assert len(all_rows) == 1280
    with pytest.raises(ValueError):
        aligned_rows(data, 0)
