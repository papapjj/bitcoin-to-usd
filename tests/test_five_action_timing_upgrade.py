from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from self_growing_trader.execution_contract import (
    ACTION_DELAY_OUTPUT_INDEX,
    EXECUTION_CONTRACT_VERSION,
    FACTORIZED_ACTION_MARKER_OUTPUT_INDEX,
    MANAGEMENT_MODE_OUTPUT_INDEX,
    PARTIAL_CLOSE_OUTPUT_INDEX,
    TARGET_SLOT_OUTPUT_INDEX,
)
from self_growing_trader.five_shared_policy import (
    FACTORIZED_ACTION_MARKER,
    PARAMETER_BINS,
    SharedFiveTradingPolicy,
)
from self_growing_trader.execution_contract import (
    LEGACY_SL_ATR_BINS,
    LEGACY_TP_RR_BINS,
    SL_ATR_BINS,
    TP_RR_BINS,
)


def test_giao_uoc_moi_danh_rieng_do_tre_va_quan_ly_vi_the() -> None:
    assert EXECUTION_CONTRACT_VERSION == 7
    assert ACTION_DELAY_OUTPUT_INDEX == 13
    assert FACTORIZED_ACTION_MARKER_OUTPUT_INDEX == 14
    assert PARTIAL_CLOSE_OUTPUT_INDEX == 17
    assert MANAGEMENT_MODE_OUTPUT_INDEX == 18
    assert TARGET_SLOT_OUTPUT_INDEX == 19
    assert PARAMETER_BINS["delay"] == 4


def test_migration_tp_v6_sang_v7_giu_nguyen_lua_chon_rr_cu() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    legacy = torch.nn.Linear(16, len(LEGACY_TP_RR_BINS))
    with torch.no_grad():
        legacy.weight.copy_(torch.arange(80, dtype=torch.float32).reshape(5, 16) / 100.0)
        legacy.bias.copy_(torch.tensor([-1.0, 0.0, 3.0, 1.0, 2.0]))
    network.heads.parameter_heads["tp"] = legacy
    probe = torch.randn(11, 16)
    old_logits = legacy(probe)
    old_choice = old_logits.argmax(dim=-1)

    assert network.require_current_architecture() is True
    migrated = network.heads.parameter_heads["tp"](probe)
    migrated_choice = migrated.argmax(dim=-1)
    mapped_choice = torch.as_tensor([
        TP_RR_BINS.index(LEGACY_TP_RR_BINS[int(index)]) for index in old_choice.tolist()
    ])

    torch.testing.assert_close(migrated_choice, mapped_choice)
    assert torch.all(migrated[:, :5] < -20.0)


def test_migration_sl_v6_sang_v7_giu_nguyen_lua_chon_atr_cu() -> None:
    network = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    legacy = torch.nn.Linear(16, len(LEGACY_SL_ATR_BINS))
    with torch.no_grad():
        legacy.weight.copy_(torch.arange(112, dtype=torch.float32).reshape(7, 16) / 100.0)
        legacy.bias.copy_(torch.tensor([-1.0, 0.0, 3.0, 1.0, 2.0, 0.5, -0.5]))
    network.heads.parameter_heads["sl"] = legacy
    probe = torch.randn(11, 16)
    old_choice = legacy(probe).argmax(dim=-1)

    assert network.require_current_architecture() is True
    migrated = network.heads.parameter_heads["sl"](probe)
    mapped_choice = torch.as_tensor([
        SL_ATR_BINS.index(LEGACY_SL_ATR_BINS[int(index)]) for index in old_choice.tolist()
    ])

    torch.testing.assert_close(migrated.argmax(dim=-1), mapped_choice)
    assert torch.all(migrated[:, 0] < -20.0)


def test_mang_phat_dau_hieu_giao_uoc_va_do_tre_0_den_3() -> None:
    net = SharedFiveTradingPolicy(pre_lstm_size=32, hidden_size=16)
    with torch.no_grad():
        results, _, _ = net.forward_five(
            torch.zeros(5, 137), net.initial_memories(), deterministic=True
        )
    for result in results:
        assert 0 <= result.delay_index <= 3
        assert result.output_vector[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] == np.float32(
            FACTORIZED_ACTION_MARKER
        )
        assert np.isfinite(result.output_vector[ACTION_DELAY_OUTPUT_INDEX])
        assert np.isfinite(result.output_vector[TARGET_SLOT_OUTPUT_INDEX])
        assert np.isfinite(result.output_vector[MANAGEMENT_MODE_OUTPUT_INDEX])
        assert np.isfinite(result.output_vector[PARTIAL_CLOSE_OUTPUT_INDEX])
        assert result.action_index not in {8, 9}
        assert int(np.argmax(result.output_vector[:12])) == int(result.action_index)


def test_ma_giao_dich_thuc_thi_do_tre_va_dong_vi_the_theo_muc_tieu() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "mt5_v2_runner" / "MQL5" / "Experts" / "SelfGrowing_MT5_Trainer_EA.mq5").read_text(
        encoding="utf-8"
    )
    assert "DecodeActionDelay(row.outputs)" in text
    assert "ScheduleDelayedPolicyAction" in text
    assert "AdvanceDelayedPolicyActions" in text
    assert "g_delayedPolicy[i].ticksRemaining" in text
    assert "TARGET_SLOT_OUTPUT_ENABLED = true" in text
    assert "DecodeTargetSlot(outputs[SG_TARGET_SLOT_OUTPUT_INDEX])" in text
    assert "DecodeDiscreteBin(outputs[SG_MANAGEMENT_MODE_OUTPUT_INDEX], 3)" in text
    assert "CloseSelectedPosition(closeRatio" in text
    assert "CloseSelectedPosition(1.0" in text
