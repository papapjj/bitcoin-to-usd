from __future__ import annotations

import numpy as np

from self_growing_trader.chuyen_gia_policy import ChuyenGiaPolicy
from self_growing_trader.five_agent_authority import (
    FIVE_AGENT_AUTHORITIES,
    FIVE_AGENT_AUTHORITY_VERSION,
    five_agent_authority,
    five_agent_authority_mask,
    five_agent_responsibility_active,
    resolve_opposite_entry_conflict,
)
from self_growing_trader.five_shared_policy import (
    SharedFiveTradingPolicy,
    public_profile_audit,
)
from self_growing_trader.orders import ACTION_COUNT, Action
from self_growing_trader.protocol_contract import INPUT_COUNT


def _small_policy() -> ChuyenGiaPolicy:
    policy = ChuyenGiaPolicy(input_size=INPUT_COUNT, output_size=20, modules=[])
    policy.five_shared_network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    policy.five_agent_memories = policy.five_shared_network.initial_memories()
    policy._learning_enabled = False
    return policy


def test_five_authority_khoa_dung_ma_action_python_ea() -> None:
    expected = (
        {Action.HOLD, Action.MARKET_BUY, Action.BUY_LIMIT, Action.BUY_STOP},
        {Action.HOLD, Action.MARKET_SELL, Action.SELL_LIMIT, Action.SELL_STOP},
        {Action.HOLD, Action.CLOSE_WORST, Action.CLOSE_BEST, Action.MANAGE_POSITION},
        {Action.HOLD, Action.CANCEL_ORDERS},
        {Action.HOLD, Action.CLOSE_ALL},
    )
    assert len(FIVE_AGENT_AUTHORITIES) == 5
    for index, actions in enumerate(expected):
        assert FIVE_AGENT_AUTHORITIES[index].agent_id == index
        assert FIVE_AGENT_AUTHORITIES[index].allowed_actions == {
            int(action) for action in actions
        }


def test_moi_action_khac_hold_chi_co_mot_chu_quyen() -> None:
    owners: dict[int, list[int]] = {action: [] for action in range(1, ACTION_COUNT)}
    for authority in FIVE_AGENT_AUTHORITIES:
        for action in authority.allowed_actions:
            if action != int(Action.HOLD):
                owners[action].append(authority.agent_id)
    assert set(owners) == set(range(1, ACTION_COUNT))
    assert all(len(action_owners) == 1 for action_owners in owners.values())


def test_authority_mask_giao_voi_trang_thai_va_giu_duong_hold_an_toan() -> None:
    mask = five_agent_authority_mask()
    assert mask.shape == (5, ACTION_COUNT)
    for authority in FIVE_AGENT_AUTHORITIES:
        assert set(np.flatnonzero(mask[authority.agent_id])) == authority.allowed_actions

    state_mask = np.zeros(ACTION_COUNT, dtype=bool)
    state_mask[int(Action.MARKET_BUY)] = True
    intersected = five_agent_authority_mask(state_mask)
    assert set(np.flatnonzero(intersected[0])) == {int(Action.MARKET_BUY)}
    for agent in range(1, 5):
        assert set(np.flatnonzero(intersected[agent])) == {int(Action.HOLD)}


def test_doi_nghich_entry_chi_giu_huong_co_xac_suat_cao_hon() -> None:
    assert resolve_opposite_entry_conflict(
        [int(Action.BUY_STOP), int(Action.MARKET_SELL), 0, 0, 0],
        [0.61, 0.72, 1.0, 1.0, 1.0],
    ) == (1, 0)
    assert resolve_opposite_entry_conflict(
        [int(Action.MARKET_BUY), int(Action.SELL_LIMIT), 0, 0, 0],
        [0.80, 0.40, 1.0, 1.0, 1.0],
    ) == (0, 1)


def test_doi_nghich_entry_tie_on_dinh_va_nan_khong_duoc_uu_tien() -> None:
    assert resolve_opposite_entry_conflict(
        [int(Action.MARKET_BUY), int(Action.MARKET_SELL), 0, 0, 0],
        [0.50, 0.50, 1.0, 1.0, 1.0],
    ) == (0, 1)
    assert resolve_opposite_entry_conflict(
        [int(Action.MARKET_BUY), int(Action.MARKET_SELL), 0, 0, 0],
        [float("nan"), 0.01, 1.0, 1.0, 1.0],
    ) == (1, 0)
    assert resolve_opposite_entry_conflict(
        [int(Action.HOLD), int(Action.MARKET_SELL), 0, 0, 0],
        [0.99, 0.01, 1.0, 1.0, 1.0],
    ) == (None, None)


def test_trach_nhiem_hold_phu_thuoc_dung_so_lenh() -> None:
    assert five_agent_responsibility_active(
        0, position_count=3, pending_order_count=1, owns_position=True
    )
    assert not five_agent_responsibility_active(
        0, position_count=3, pending_order_count=1, owns_position=False
    )
    assert five_agent_responsibility_active(
        2, position_count=1, pending_order_count=0, owns_position=False
    )
    assert five_agent_responsibility_active(
        3, position_count=0, pending_order_count=1, owns_position=False
    )
    assert five_agent_responsibility_active(
        4, position_count=0, pending_order_count=1, owns_position=False
    )
    assert not five_agent_responsibility_active(
        4, position_count=0, pending_order_count=0, owns_position=False
    )


def test_policy_chi_lay_mau_action_thuoc_quyen_va_ghi_role() -> None:
    policy = _small_policy()
    traces = policy.act_five_agents(np.zeros(INPUT_COUNT, dtype=np.float32))
    assert len(traces) == 5
    for agent_id, trace in enumerate(traces):
        details = trace.action_details
        authority = five_agent_authority(agent_id)
        assert int(trace.selected_idx) in authority.allowed_actions
        assert set(np.flatnonzero(trace.allowed_action_mask)) == authority.allowed_actions
        assert details["five_agent_authority_version"] == FIVE_AGENT_AUTHORITY_VERSION
        assert details["five_agent_authority_role"] == authority.role
        assert details["five_agent_authority_actions"] == sorted(authority.allowed_actions)


def test_public_profile_cong_bo_hop_dong_phan_quyen() -> None:
    audit = public_profile_audit()
    assert audit["agent_authority_version"] == FIVE_AGENT_AUTHORITY_VERSION
    assert [item["role"] for item in audit["agent_authorities"]] == [
        item.role for item in FIVE_AGENT_AUTHORITIES
    ]


def test_phan_quyen_khong_them_tensor_va_state_dict_cu_van_load_strict() -> None:
    source = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    checkpoint_state = source.state_dict()

    target = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    result = target.load_state_dict(checkpoint_state, strict=True)

    assert result.missing_keys == []
    assert result.unexpected_keys == []
    assert tuple(checkpoint_state) == tuple(target.state_dict())
    assert all(
        checkpoint_state[name].shape == target.state_dict()[name].shape
        for name in checkpoint_state
    )
