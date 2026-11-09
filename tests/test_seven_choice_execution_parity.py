from __future__ import annotations

import numpy as np
import pytest

from self_growing_trader.config import OrderConfig, RiskConfig
from self_growing_trader.execution_contract import (
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    FACTORIZED_ACTION_MARKER_VALUE,
    SL_ATR_BINS,
    TP_RR_BINS,
    TRAILING_ATR_BINS,
)
from self_growing_trader.five_shared_policy import SharedFiveTradingPolicy
from self_growing_trader.orders import Action, PositionManagementMode, decode_outputs
from self_growing_trader.protocol_contract import INPUT_COUNT, POSITION_OFFSET


def test_fixed_seven_choice_geometry_keeps_entry_delay_and_manager_wire_contract() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    network = SharedFiveTradingPolicy(pre_lstm_size=16, hidden_size=8)
    network.eval()
    observations = torch.zeros((5, INPUT_COUNT))
    observations[:, POSITION_OFFSET] = 1.0
    observations[:, POSITION_OFFSET + 1] = 1.0
    actions = [Action.BUY_STOP, Action.SELL_LIMIT, Action.MANAGE_POSITION,
               Action.HOLD, Action.CLOSE_ALL]
    allowed = torch.zeros((5, 12), dtype=torch.bool)
    for role, action in enumerate(actions):
        allowed[role, int(action)] = True
    with torch.no_grad():
        mode_head = network.heads.parameter_heads["manage_mode"]
        mode_head.weight.zero_()
        mode_head.bias.fill_(-10.0)
        mode_head.bias[2] = 10.0
        ordinary, _, _ = network.forward_five(
            observations, network.initial_memories(),
            allowed_action_masks=allowed, deterministic=True,
        )
        network.heads.seven_choice_fixed_geometry_enabled = True
        network.heads.seven_choice_preserve_manager_wire = True
        network.heads.seven_choice_fixed_delay_index = 3
        network.heads.seven_choice_fixed_entry_offset_index = 7
        network.heads.seven_choice_fixed_sl_index = SL_ATR_BINS.index(1.5)
        network.heads.seven_choice_fixed_tp_index = TP_RR_BINS.index(2.0)
        network.heads.seven_choice_fixed_trailing_index = TRAILING_ATR_BINS.index(0.0)
        fixed, _, _ = network.forward_five(
            observations, network.initial_memories(),
            allowed_action_masks=allowed, deterministic=True,
        )

    # Deliberately disagree with the checkpoint geometry: the fixed bins on
    # the wire must win over profile defaults, just as EA marker-7 decoding does.
    orders = OrderConfig(timeframe="M5", fixed_sl_atr=2.5, fixed_tp_atr=5.0)
    decoded = [decode_outputs(row.output_vector, RiskConfig(), orders, max_positions=5)
               for row in fixed]
    for role in (0, 1):
        assert fixed[role].output_vector[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] == FACTORIZED_ACTION_MARKER_VALUE
        assert decoded[role].action == actions[role]
        assert decoded[role].factorized_contract
        assert decoded[role].action_delay_ticks == 3
        assert decoded[role].sl_atr == pytest.approx(1.5)
        assert decoded[role].tp_atr == pytest.approx(3.0)
        assert decoded[role].trailing_stop_atr == 0.0

    for role in (2, 3, 4):
        np.testing.assert_array_equal(fixed[role].output_vector, ordinary[role].output_vector)
    assert decoded[2].factorized_contract
    assert decoded[2].action == Action.MANAGE_POSITION
    assert decoded[2].management_mode == PositionManagementMode.FULL_CLOSE
    assert decoded[2].target_slot == 0
