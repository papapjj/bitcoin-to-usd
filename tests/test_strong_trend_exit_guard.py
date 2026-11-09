from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from self_growing_trader.he_chuyen_gia_ngu_canh import (
    CauHinhHeChuyenGia,
    HeChuyenGiaMotContextMotPolicy,
    NhanDienNguCanh,
    NhiemVuNguCanh,
    _allowed_actions_for_context_task,
    _trend_exit_guard,
)
from self_growing_trader.mistake_memory import episode_from_trace
from self_growing_trader.mt5_runner import MT5StrategyTesterEvaluator


def _identification(task: NhiemVuNguCanh, *, stable: bool = True) -> NhanDienNguCanh:
    context = f"CTX_{task.value}"
    return NhanDienNguCanh(
        ma_context=context,
        ma_context_cha=context,
        nhiem_vu=task,
        toa_do=(0.8, 0.2),
        do_tin_cay=1.0,
        on_dinh=stable,
        ly_do=f"active={context};candidate={context};confirm=3/3",
    )


def _position(
    *,
    side: int = 1,
    current_r: float = 0.30,
    mfe_r: float = 0.35,
    bars_since_mfe: float = 0.0,
    near_tp: bool = False,
    giveback_r: float = 0.05,
) -> str:
    return ":".join([
        "777", str(side), "0.10", "100.0", "99.0", "103.0",
        str(current_r), "5", str(mfe_r), "-0.10", "1.0", "1.0",
        "0.5", "0.5", str(bars_since_mfe), "1" if near_tp else "0",
        str(giveback_r), "CTX_TANG_MANH", "CG_TANG_MANH", "CG_TANG_MANH",
        "trace-entry",
    ])


def test_strong_trend_contract_exposes_manage_position() -> None:
    up = _allowed_actions_for_context_task(NhiemVuNguCanh.TANG_MANH)
    down = _allowed_actions_for_context_task(NhiemVuNguCanh.GIAM_MANH)

    assert 11 in up
    assert 11 in down
    assert {1, 2, 3, 4, 5, 6}.issubset(up)
    assert {1, 2, 3, 4, 5, 6}.issubset(down)
    assert 7 in up and 7 in down


def test_strong_trend_guard_keeps_entry_and_manage_but_suppresses_premature_close() -> None:
    allowed = {0, 1, 8, 9, 10, 11}
    guarded, audit = _trend_exit_guard(
        _identification(NhiemVuNguCanh.TANG_MANH),
        allowed,
        _position(),
        CauHinhHeChuyenGia(),
    )

    assert {0, 1, 11}.issubset(guarded)
    assert {8, 9, 10}.isdisjoint(guarded)
    assert audit["trend_exit_guard_active"] is True
    assert audit["trend_exit_protected_tickets"] == ["777"]


def test_strong_trend_guard_reopens_close_after_confirmed_giveback_or_emergency() -> None:
    allowed = {0, 1, 8, 9, 10, 11}
    config = CauHinhHeChuyenGia()
    identified = _identification(NhiemVuNguCanh.TANG_MANH)

    giveback, giveback_audit = _trend_exit_guard(
        identified,
        allowed,
        _position(current_r=0.70, mfe_r=1.20, bars_since_mfe=3.0, giveback_r=0.50),
        config,
    )
    emergency, emergency_audit = _trend_exit_guard(
        identified,
        allowed,
        _position(current_r=-0.40, mfe_r=0.05),
        config,
    )

    assert giveback == allowed
    assert giveback_audit["trend_exit_guard_active"] is False
    assert emergency == allowed
    assert emergency_audit["trend_exit_guard_active"] is False


def test_strong_trend_manage_is_trailing_only() -> None:
    outputs = np.zeros(20, dtype=float)
    outputs[11] = 1.0
    outputs[16] = -2.0
    outputs[18] = 4.0
    trace = SimpleNamespace(selected_action=outputs, action_details={})

    adjusted = HeChuyenGiaMotContextMotPolicy._ap_dung_quan_ly_trend_manh(
        trace,
        {"trend_exit_guard_active": True},
    )

    assert adjusted.selected_action[16] >= 1.0
    assert adjusted.selected_action[18] <= -1.2
    assert adjusted.action_details["trend_management_mode"] == "trailing_only"


def test_wire_action_scope_intersects_mt5_and_context_contract() -> None:
    trace = SimpleNamespace(
        context_audit={"context_allowed_actions": [0, 1, 11]},
    )

    effective = MT5StrategyTesterEvaluator._effective_context_actions(
        trace,
        {0, 1, 2, 8, 9, 10, 11},
    )

    assert effective == {0, 1, 11}


def test_premature_strong_trend_close_becomes_manage_replacement_lesson() -> None:
    selected = np.zeros(20, dtype=float)
    selected[9] = 1.0
    trace = SimpleNamespace(
        trace_id="close-early",
        selected_action=selected,
        selected_idx=0,
        raw_obs=np.zeros(137, dtype=float),
        encoded_state=np.zeros(20, dtype=float),
        account_context={},
        experience_starvation={},
        action_details={"action_index": 9, "action_name": "CLOSE_BEST", "target_slot": 0},
        context_audit={"context_task": "TANG_MANH", "context_on_dinh": True},
        positions_pipe=_position(current_r=0.30, mfe_r=0.35, bars_since_mfe=0.0, giveback_r=0.05),
        positions_pipe_after="",
        execution_details={"close_reason": "EXPERT", "requested_action": 9},
        active_neuron_ids=(),
        active_edge_ids=(),
        graph_edge_gates={},
        graph_region_activity={},
        module_action_traces=(),
        lifecycle_tags=(),
    )

    episode = episode_from_trace(
        trace,
        np.zeros(137, dtype=float),
        reward=0.03,
        cost=0.0,
        signal={"td_delta": 0.0, "prediction_error": 0.0},
        metrics={"duration_bars": 5.0, "realized_r": 0.03},
    )

    assert "premature_strong_trend_exit" in episode.mistake_tags
    assert "replace_close_with_trailing_management" in episode.proposed_repairs
