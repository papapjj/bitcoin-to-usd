from __future__ import annotations

import numpy as np

from self_growing_trader.config import OrderConfig, RiskConfig
from self_growing_trader.execution_contract import (
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
    GEOMETRY_PACKED_OUTPUT_INDEX,
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
    pack_geometry_choices,
    unpack_geometry_choices,
)
from self_growing_trader.orders import (
    Action,
    Position,
    PositionManagementMode,
    allowed_actions_for_state,
    conservative_exit,
    decode_outputs,
    pnl_for_exit,
)


def test_zero_position_limit_keeps_all_entry_actions_available() -> None:
    allowed = allowed_actions_for_state(
        position_count=100,
        pending_count=100,
        max_positions=0,
    )
    assert {
        Action.MARKET_BUY,
        Action.MARKET_SELL,
        Action.BUY_LIMIT,
        Action.SELL_LIMIT,
        Action.BUY_STOP,
        Action.SELL_STOP,
    } <= allowed


def test_action_argmax_uses_only_graph_action_logits() -> None:
    outputs = np.zeros(20)
    outputs[0] = 1.0
    # Index 11 is the final graph action (MANAGE_POSITION); parameters begin at 12.
    outputs[12] = 999.0
    outputs[13] = 999.0
    outputs[14] = 999.0
    outputs[15] = 999.0
    decision = decode_outputs(outputs, RiskConfig(), OrderConfig())
    assert decision.action == Action.HOLD


def test_decoded_tp_is_fixed_and_respects_technical_floor() -> None:
    outputs = np.zeros(15)
    outputs[1] = 1.0
    outputs[12] = 999.0
    outputs[13] = -999.0
    decision = decode_outputs(outputs, RiskConfig(), OrderConfig(technical_min_rr=0.2))
    assert decision.sl_atr == 5.0
    assert decision.tp_atr == 10.0
    assert decision.rr >= 0.2


def test_compatibility_rows_cannot_change_fixed_geometry() -> None:
    outputs = np.zeros(20)
    outputs[int(Action.MARKET_BUY)] = 1.0
    outputs[13] = 999.0
    outputs[14] = -999.0
    decision = decode_outputs(outputs, RiskConfig(), OrderConfig())
    assert (decision.sl_atr, decision.tp_atr, decision.rr) == (5.0, 10.0, 2.0)
    assert decision.action == Action.MARKET_BUY


def test_three_geometry_heads_pack_losslessly_into_unused_bridge_row() -> None:
    for sl_index in range(len(SL_ATR_BINS)):
        for tp_index in range(len(TP_RR_BINS)):
            for trailing_index in range(len(TRAILING_ATR_BINS)):
                packed = pack_geometry_choices(sl_index, tp_index, trailing_index)
                assert unpack_geometry_choices(packed) == (sl_index, tp_index, trailing_index)


def test_contract_v7_decodes_sub_2r_geometry_without_technical_rr_clamp() -> None:
    outputs = np.zeros(20)
    outputs[int(Action.MARKET_BUY)] = 1.0
    outputs[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] = FACTORIZED_ACTION_MARKER_VALUE
    outputs[GEOMETRY_PACKED_OUTPUT_INDEX] = pack_geometry_choices(6, 3, 4)

    decision = decode_outputs(outputs, RiskConfig(), OrderConfig(technical_min_rr=2.0))

    assert decision.sl_atr == 4.0
    assert decision.tp_atr == 4.0
    assert decision.trailing_stop_atr == 2.0
    assert decision.rr == 1.0


def test_action_mask_ignores_disallowed_high_score_actions() -> None:
    outputs = np.zeros(15)
    outputs[8] = 100.0
    outputs[1] = 1.0
    decision = decode_outputs(outputs, RiskConfig(), OrderConfig(), allowed_actions={Action.HOLD, Action.MARKET_BUY})
    assert decision.action == Action.MARKET_BUY


def test_decode_position_management_is_inert_in_fixed_geometry_mode() -> None:
    outputs = np.zeros(20)
    outputs[0] = 1.0
    outputs[18] = 2.0
    outputs[19] = 0.5
    decision = decode_outputs(outputs, RiskConfig(), OrderConfig(), max_positions=5)
    assert decision.action == Action.HOLD
    assert decision.target_slot == 0
    assert decision.management_mode == PositionManagementMode.NONE


def test_buy_uses_bid_exit_and_conservative_sl_first() -> None:
    position = Position(side=1, entry=101.0, sl=99.0, tp=105.0, risk_pct=0.01, risk_amount=100.0, risk_distance=2.0, size=50.0, opened_bar=0)
    should_close, exit_price, reason = conservative_exit(position, high_bid=106.0, low_bid=98.0, close_bid=104.0, spread_price=1.0)
    assert should_close
    assert exit_price == 99.0
    assert reason == "sl"
    assert pnl_for_exit(position, exit_price) == -100.0


def test_sell_uses_ask_exit_and_conservative_sl_first() -> None:
    position = Position(side=-1, entry=100.0, sl=102.0, tp=96.0, risk_pct=0.01, risk_amount=100.0, risk_distance=2.0, size=50.0, opened_bar=0)
    should_close, exit_price, reason = conservative_exit(position, high_bid=101.5, low_bid=95.0, close_bid=97.0, spread_price=1.0)
    assert should_close
    assert exit_price == 102.0
    assert reason == "sl"
    assert pnl_for_exit(position, exit_price) == -100.0



def test_v4_can_select_explicit_manage_position_without_reusing_risk_parameter() -> None:
    outputs = np.zeros(20)
    outputs[int(Action.HOLD)] = 1.0
    outputs[int(Action.MANAGE_POSITION)] = 10.0
    outputs[12] = 999.0  # v4 risk parameter, never a discrete action.

    decision = decode_outputs(outputs, RiskConfig(), OrderConfig(), max_positions=3)

    assert decision.action == Action.MANAGE_POSITION


def test_v4_management_output_cannot_change_fixed_position_contract() -> None:
    outputs = np.zeros(20)
    outputs[int(Action.HOLD)] = 2.0
    outputs[18] = 3.0
    outputs[19] = 999.0

    decision = decode_outputs(outputs, RiskConfig(), OrderConfig(), max_positions=5)

    assert decision.action == Action.HOLD
    assert decision.management_mode == PositionManagementMode.NONE
    assert decision.target_slot == 0
