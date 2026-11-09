from __future__ import annotations

from datetime import datetime

import pytest

from self_growing_trader.five_lifecycle_reward import (
    FiveLifecycleRewardConfig,
    lifecycle_path_dynamics,
    pop_stratified_lifecycle_batch,
    shape_five_entry_lifecycle,
)
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    MT5PositionLifecycle,
    MT5StrategyTesterEvaluator,
    _expand_geometry_pipes,
)


def _shape(**overrides):
    values = {
        "action_index": 1, "sl_atr": 2.5, "tp_atr": 5.0,
        "trailing_atr": 0.0, "trailing_applied": False, "locked_profit_r": 0.0,
        "mfe_r": 1.0, "mae_r": -0.25, "mfe_available": True,
        "realized_r": 1.0, "exit_kind": "TP", "config": FiveLifecycleRewardConfig(),
    }
    values.update(overrides)
    return shape_five_entry_lifecycle(**values)


def test_pure_net_r_profile_disables_v4_shaping_without_disabling_contract() -> None:
    config = FiveLifecycleRewardConfig.from_source(
        type("Learning", (), {
            "five_lifecycle_reward_enabled": False,
            "five_lifecycle_net_r_credit_enabled": True,
            "five_lifecycle_net_r_credit_weight": 1.0,
            "five_lifecycle_net_r_credit_clip_r": 3.0,
        })()
    )

    shaped = _shape(config=config, realized_r=-1.25, exit_kind="INITIAL_SL")

    assert config.enabled is False
    assert shaped.total_r == pytest.approx(0.0)
    assert shaped.unclipped_total_r == pytest.approx(0.0)
    # The signed net-R is routed by ChuyenGiaPolicy, not manufactured by the
    # V4 lifecycle shaper. This keeps the two causal channels distinguishable.
    assert shaped.initial_sl is False
    assert shaped.initial_sl_failure is False


def test_initial_sl_chi_danh_dau_loss_cuc_bo_va_khong_phat_scalar_trung() -> None:
    shaped = _shape(sl_atr=1.0, tp_atr=2.0, mfe_r=0.0, realized_r=-1.0, exit_kind="INITIAL_SL")
    assert shaped.initial_sl is True
    assert shaped.initial_sl_failure is True
    assert shaped.entry_excursion_quality_r == pytest.approx(0.0)
    assert shaped.geometry_quality_r == pytest.approx(-0.30)
    assert shaped.unclipped_total_r == pytest.approx(-0.30)
    assert shaped.total_r == pytest.approx(-0.30)
    assert "initial_sl_outcome_r" not in shaped.as_dict()


def test_initial_sl_khong_duoc_thuong_hinh_hoc_du_mfe_duong() -> None:
    shaped = _shape(
        action_index=3, sl_atr=2.0, tp_atr=4.0, mfe_r=1.0,
        realized_r=-1.0, exit_kind="INITIAL_SL",
        trailing_atr=1.5, trailing_applied=True, locked_profit_r=0.8,
        pending_fill_bars=1,
    )
    assert shaped.initial_sl is True
    assert shaped.initial_sl_failure is True
    assert shaped.geometry_quality_r == pytest.approx(0.0)
    assert shaped.entry_excursion_quality_r == pytest.approx(0.0)
    assert shaped.trailing_evaluated is False
    assert shaped.trailing_quality_r == pytest.approx(0.0)
    assert shaped.pending_fill_quality_r == pytest.approx(0.0)
    assert shaped.total_r == pytest.approx(0.0)


def test_initial_sl_net_duong_sau_chot_mot_phan_khong_bi_coi_la_entry_sai() -> None:
    shaped = _shape(
        sl_atr=2.0, tp_atr=4.0, mfe_r=1.0,
        realized_r=0.50, exit_kind="INITIAL_SL",
    )
    assert shaped.initial_sl is True
    assert shaped.initial_sl_failure is False
    assert shaped.geometry_quality_r == pytest.approx(0.05)
    assert shaped.entry_excursion_quality_r > 0.0
    assert shaped.total_r > 0.0


def test_thuong_hinh_hoc_chi_co_khi_mfe_dat_nguong_co_y_nghia() -> None:
    assert _shape(mfe_r=0.49, realized_r=0.0).geometry_quality_r == pytest.approx(0.0)
    assert _shape(mfe_r=0.50, realized_r=0.0).geometry_quality_r == pytest.approx(0.05)


@pytest.mark.parametrize(("mfe", "expected"), [(0.0, -0.40), (0.10, -0.30), (0.25, -0.15), (0.50, 0.0)])
def test_phat_mfe_mem_lien_tuc(mfe: float, expected: float) -> None:
    shaped = _shape(mfe_r=mfe, realized_r=0.0, sl_atr=0.0, tp_atr=0.0)
    assert shaped.entry_excursion_quality_r == pytest.approx(expected)


def test_trailing_chi_danh_gia_khi_duoc_chon_va_mfe_du_nguong() -> None:
    assert _shape(trailing_atr=0.0, mfe_r=2.0).trailing_evaluated is False
    assert _shape(trailing_atr=1.5, mfe_r=0.49).trailing_evaluated is False
    missed = _shape(trailing_atr=1.5, mfe_r=1.0, trailing_applied=False)
    assert missed.trailing_evaluated is True
    assert missed.trailing_quality_r == pytest.approx(-0.20)
    captured = _shape(trailing_atr=1.5, mfe_r=1.0, trailing_applied=True, locked_profit_r=0.8)
    assert captured.trailing_quality_r == pytest.approx(0.25)


def test_cac_thanh_phan_duong_va_gioi_han_tren() -> None:
    shaped = _shape(
        action_index=3, mfe_r=1.5, mae_r=-0.1, realized_r=1.2,
        trailing_atr=1.5, trailing_applied=True, locked_profit_r=1.2,
        recovery_fraction=1.0, positive_persistence=1.0, pending_fill_bars=1,
    )
    assert shaped.geometry_quality_r == pytest.approx(0.05)
    assert shaped.entry_excursion_quality_r == pytest.approx(0.30)
    assert shaped.trailing_quality_r == pytest.approx(0.25)
    assert shaped.exit_capture_quality_r == pytest.approx(0.30)
    assert shaped.recovery_persistence_quality_r == pytest.approx(0.15)
    assert shaped.pending_fill_quality_r > 0.05
    assert shaped.unclipped_total_r > 0.75
    assert shaped.total_r == pytest.approx(0.75)


def test_hoi_phuc_va_do_ben_duoc_tinh_tu_quy_dao_r() -> None:
    recovery, persistence = lifecycle_path_dynamics((
        {"age_bars": 1, "unrealized_r": -0.5},
        {"age_bars": 2, "unrealized_r": 0.2},
        {"age_bars": 3, "unrealized_r": 0.4},
    ))
    assert recovery == pytest.approx(1.0)
    assert persistence == pytest.approx(1.0)


def test_batch_phan_tang_khong_lap_mau_va_khong_sua_nhan() -> None:
    queue = [
        {"id": index, "reward": float(index), "lifecycle_stratum": (f"a{index % 3}", "duong" if index % 2 else "am", f"r{index % 2}")}
        for index in range(40)
    ]
    batch = pop_stratified_lifecycle_batch(queue, batch_size=32, max_wait=64)
    assert len(batch) == 32
    assert len({row["id"] for row in batch}) == 32
    assert {row["lifecycle_stratum"][1] for row in batch} == {"duong", "am"}
    assert all(row["reward"] == float(row["id"]) for row in batch)
    assert len(queue) == 8


def test_schema17_doc_tuoi_khop_lenh_cho_chinh_xac_tu_pipe() -> None:
    values = ["1", "1", "2", "2", "4", "100", "98", "104", "1", "learned", "2", "4", "2", "1.5", "3"]
    expanded = _expand_geometry_pipes({"geometry_entry_pipe": "|".join(values)})
    assert expanded["pending_fill_bars"] == "3"


def test_mt5_route_dung_mot_reward_v4_ve_entry_owner(tmp_path) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=tmp_path, offline_replay=True))
    evaluator._register_five_competitor_trace("entry-a", 2, 1)
    lifecycle = MT5PositionLifecycle(
        position_ticket="77", entry_time=datetime(2026, 8, 14, 10, 0),
        exit_time=datetime(2026, 8, 14, 10, 5), net_pnl=-100.0,
        gross_profit=-100.0, commission=0.0, swap=0.0, initial_risk=100.0,
        realized_r=-1.0, close_reason="SL", trace_id="entry-a",
        final_close_confirmed=True, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_atr=1.0, mfe_r=0.0, mae_r=-1.0,
        excursion_available=True, exit_kind="INITIAL_SL", trailing_applied=False,
        pending_fill_bars=2, metadata={"proposed_trailing_atr": 1.5},
    )

    class Sink:
        def __init__(self) -> None:
            self.calls = []

        def apply_five_entry_lifecycle_reward(self, trace_id, **payload):
            self.calls.append((trace_id, payload))
            return {
                "accepted": True,
                "updated": 1,
                "initial_sl_local_loss_value": 0.75,
                "initial_sl_local_samples": 8.0,
                "lifecycle_rollout_steps": 32.0,
            }

    sink = Sink()
    assert evaluator._record_five_competition_lifecycles([lifecycle], policy=sink) == 1
    assert evaluator._record_five_competition_lifecycles([lifecycle], policy=sink) == 0
    assert len(sink.calls) == 1
    _, payload = sink.calls[0]
    assert payload["lifecycle_reward_r"] == pytest.approx(-0.30)
    assert payload["realized_r"] == pytest.approx(-1.0)
    assert payload["components"]["entry_excursion_quality_r"] == pytest.approx(0.0)
    assert payload["components"]["initial_sl"] is True
    assert payload["components"]["initial_sl_failure"] is True
    assert "initial_sl_outcome_r" not in payload["components"]
    routing = evaluator._five_lifecycle_routing_snapshot()
    assert routing == {
        "attempts": 1,
        "accepted": 1,
        "misses": 0,
        "acceptance_ratio": 1.0,
        "initial_sl_attempts": 1,
        "initial_sl_accepted": 1,
        "optimizer_updates": 1,
        "initial_sl_optimizer_samples": 8.0,
        "last_initial_sl_loss": 0.75,
        "last_rollout_steps": 32.0,
        "v4_credit_sum_r": -0.30,
        "net_r_credit_sum_r": 0.0,
        "combined_credit_sum_r": -0.30,
    }


def test_mt5_route_uu_tien_owner_bat_bien_khi_trace_dong_la_manage(tmp_path) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=tmp_path, offline_replay=True))
    evaluator._register_five_competitor_trace("entry-owner", 2, 1)
    # The lifecycle trace represents the final MANAGE/CLOSE command.  The
    # immutable owner persisted by the EA is the only valid reward target.
    lifecycle = MT5PositionLifecycle(
        position_ticket="79", entry_time=datetime(2026, 8, 14, 10, 0),
        exit_time=datetime(2026, 8, 14, 10, 5), net_pnl=100.0,
        gross_profit=100.0, commission=0.0, swap=0.0, initial_risk=100.0,
        realized_r=1.0, close_reason="EXPERT", trace_id="close-manage",
        final_close_confirmed=True, entry_price=100.0, initial_sl_price=99.0,
        initial_tp_price=102.0, initial_atr=1.0, mfe_r=1.0, mae_r=-0.1,
        excursion_available=True, exit_kind="EXPERT",
        metadata={"position_owner_trace_id": "entry-owner"},
    )

    class Sink:
        def __init__(self) -> None:
            self.calls = []

        def apply_five_entry_lifecycle_reward(self, trace_id, **payload):
            self.calls.append((trace_id, payload))
            return {"accepted": True, "updated": 1, "lifecycle_rollout_steps": 1.0}

    sink = Sink()
    assert evaluator._record_five_competition_lifecycles([lifecycle], policy=sink) == 1
    assert len(sink.calls) == 1
    assert sink.calls[0][0] == "entry-owner"


def test_initial_sl_route_miss_duoc_dem_du_scalar_reward_bang_khong(tmp_path) -> None:
    evaluator = MT5StrategyTesterEvaluator(MT5BacktestConfig(runner_dir=tmp_path, offline_replay=True))
    evaluator._register_five_competitor_trace("entry-miss", 1, 2)
    lifecycle = MT5PositionLifecycle(
        position_ticket="78", entry_time=datetime(2026, 8, 14, 10, 0),
        exit_time=datetime(2026, 8, 14, 10, 5), net_pnl=-100.0,
        gross_profit=-100.0, commission=0.0, swap=0.0, initial_risk=100.0,
        realized_r=-1.0, close_reason="SL", trace_id="entry-miss",
        final_close_confirmed=True, entry_price=100.0, initial_sl_price=98.0,
        initial_tp_price=104.0, initial_atr=1.0, mfe_r=1.0, mae_r=-1.0,
        excursion_available=True, exit_kind="INITIAL_SL",
    )

    class MissSink:
        @staticmethod
        def apply_five_entry_lifecycle_reward(trace_id, **payload):
            assert payload["lifecycle_reward_r"] == pytest.approx(0.0)
            return {"accepted": False, "updated": 0, "reason": "trace_cache_miss"}

    assert evaluator._record_five_competition_lifecycles([lifecycle], policy=MissSink()) == 1
    routing = evaluator._five_lifecycle_routing_snapshot()
    assert routing["attempts"] == 1
    assert routing["accepted"] == 0
    assert routing["misses"] == 1
    assert routing["initial_sl_attempts"] == 1
    assert routing["initial_sl_accepted"] == 0
