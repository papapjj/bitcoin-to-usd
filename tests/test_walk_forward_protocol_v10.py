from __future__ import annotations

import pytest

from self_growing_trader.self_growing_runner import walk_forward_windows, _kiem_tra_cua_so_walk_forward


def test_walk_forward_separates_train_and_oos_test_windows():
    windows = walk_forward_windows("2025.01.01", "2025.01.15", 7, 7, 7)
    assert windows == [{
        "from_date": "2025.01.01",
        "to_date": "2025.01.15",
        "train_from": "2025.01.01",
        "train_to": "2025.01.08",
        "test_from": "2025.01.08",
        "test_to": "2025.01.15",
    }]


def test_walk_forward_three_windows_with_step_equal_test_window():
    windows = walk_forward_windows("2025.01.01", "2025.01.29", 7, 7, 7)
    assert len(windows) == 3
    assert windows[0]["test_from"] == "2025.01.08"
    assert windows[1]["train_from"] == "2025.01.08"
    assert windows[2]["test_to"] == "2025.01.29"


def test_walk_forward_rejects_too_short_date_range_with_clear_message():
    with pytest.raises(ValueError, match="can toi thieu 14 ngay"):
        _kiem_tra_cua_so_walk_forward(
            from_date="2025.01.01",
            to_date="2025.01.02",
            train_window_days=7,
            test_window_days=7,
            step_days=7,
        )
