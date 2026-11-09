from __future__ import annotations

"""Nhan hoc tre, xep hang 6 entry va credit HOLD-trong cho Five.

Du lieu tuong lai chi duoc dung sau khi path bat bien da hoan tat. Chung khong
bao gio di vao 137 quan sat, bo nho recurrent, lenh gui broker hay suy luan
live/frozen. Reward scalar o day co y loai bo cac khoan da co chu so huu:

* PnL, SL-truoc/TP-truoc: reward kinh te va loss phan loai cua cay quan sat.
* Toan bo MAE/MFE scalar cua entry that: reward vong doi entry V4.
* Lenh cho khong khop sau 5 bar: phat timeout cua so thuc thi.

Path counterfactual chi con hai nhiem vu: day nhan cho cay quan sat va xep hang
chat luong tong hop giua sau entry. Scalar o day chi danh gia HOLD khi tai khoan
trong; no khong thuong/phat lai entry da duoc lifecycle V4 xu ly.
"""

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

import numpy as np

from .entry_preference import (
    EntryPreferenceRecord,
    replay_entry_component_matrix,
    replay_entry_path_diagnostic_matrix,
)


ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})
PATH_REWARD_COMPONENTS = ("hold_bo_lo_edge", "hold_ne_tat_ca_entry_xau")


@dataclass(frozen=True)
class PatternDiscoveryRewardOracle:
    long_utility_atr: float
    short_utility_atr: float
    tree_min_edge_atr: float = 0.10
    long_mfe_atr: float = 0.0
    long_mae_atr: float = 0.0
    short_mfe_atr: float = 0.0
    short_mae_atr: float = 0.0
    tree_utility_clip_atr: float = 3.0
    auxiliary_loss_weight: float = 0.25
    utility_loss_weight: float = 1.0
    excursion_loss_weight: float = 0.50
    no_edge_loss_weight: float = 0.50
    direction_loss_weight: float = 1.0
    edge_hold_ceiling: float = 0.20
    hold_ceiling_loss_weight: float = 1.0

    flat_hold_missed_edge_penalty_r: float = 0.25
    flat_hold_confirmed_no_edge_bonus_r: float = 0.01
    flat_hold_edge_threshold_r: float = 0.10
    flat_hold_no_edge_threshold_r: float = -0.10
    flat_account_anchor: bool = False

    # Nhan path tre cho cay quan sat v4.
    horizon_returns_norm: tuple[float, ...] = (0.0,) * 5
    horizon_mask: tuple[float, ...] = (0.0,) * 5
    path_dynamics_norm: tuple[float, ...] = (0.0,) * 10
    first_hit_classes: tuple[int, ...] = (0, 0)
    action_path_norm: tuple[float, ...] = (0.0,) * 30
    action_outcome_classes: tuple[int, ...] = (0,) * 6
    action_utility_atr: tuple[float, ...] = (0.0,) * 6
    action_path_reward_r: tuple[float, ...] = (0.0,) * 6
    action_q_values_r: tuple[float, ...] = (0.0,) * 7
    # Signed, bounded path-only shaping for an entry that was actually
    # sampled.  Production keeps this at zero because lifecycle economic R is
    # the sole scalar for a filled entry; candidate experiments can enable a
    # small value to give positive credit to good counterfactual entries and
    # negative credit to bad ones without changing lifecycle R.
    action_quality_reward_scale_r: float = 0.0
    ranking_delta_r: float = 0.10
    ranking_margin_scale: float = 0.50
    ranking_margin_min: float = 0.05
    ranking_margin_max: float = 1.00
    ranking_loss_weight: float = 0.50
    hold_reward_components_r: tuple[float, ...] = (0.0, 0.0)
    rich_path_available: bool = False
    horizon_loss_weight: float = 0.50
    path_dynamics_loss_weight: float = 0.50
    first_hit_loss_weight: float = 0.50
    action_path_loss_weight: float = 0.50
    action_outcome_loss_weight: float = 0.50

    @classmethod
    def from_sample(
        cls,
        sample: Mapping[str, object],
        *,
        spread_cost_atr: float,
        tree_min_edge_atr: float = 0.10,
        tree_mfe_weight: float = 0.15,
        tree_mae_weight: float = 0.25,
        tree_utility_clip_atr: float = 3.0,
        auxiliary_loss_weight: float = 0.25,
        utility_loss_weight: float = 1.0,
        excursion_loss_weight: float = 0.50,
        no_edge_loss_weight: float = 0.50,
        direction_loss_weight: float = 1.0,
        edge_hold_ceiling: float = 0.20,
        hold_ceiling_loss_weight: float = 1.0,
        horizon_loss_weight: float = 0.50,
        path_dynamics_loss_weight: float = 0.50,
        first_hit_loss_weight: float = 0.50,
        action_path_loss_weight: float = 0.50,
        action_outcome_loss_weight: float = 0.50,
        flat_hold_missed_edge_penalty_r: float = 0.25,
        flat_hold_confirmed_no_edge_bonus_r: float = 0.01,
        flat_hold_edge_threshold_r: float = 0.10,
        flat_hold_no_edge_threshold_r: float = -0.10,
        ranking_delta_r: float = 0.10,
        ranking_margin_scale: float = 0.50,
        ranking_margin_min: float = 0.05,
        ranking_margin_max: float = 1.00,
        ranking_loss_weight: float = 0.50,
        action_quality_reward_scale_r: float = 0.0,
    ) -> "PatternDiscoveryRewardOracle":
        cap = max(float(tree_utility_clip_atr), 1e-6)
        long_mfe = float(np.clip(float(sample["long_mfe_atr"]), 0.0, cap))
        long_mae = float(np.clip(float(sample["long_mae_atr"]), -cap, 0.0))
        short_mfe = float(np.clip(float(sample["short_mfe_atr"]), 0.0, cap))
        short_mae = float(np.clip(float(sample["short_mae_atr"]), -cap, 0.0))

        def utility(side: str) -> float:
            final = float(sample[f"{side}_final_return_atr"])
            mfe = long_mfe if side == "long" else short_mfe
            mae = long_mae if side == "long" else short_mae
            value = final + float(tree_mfe_weight) * mfe + float(tree_mae_weight) * mae
            return float(np.clip(value - max(0.0, float(spread_cost_atr)), -cap, cap))

        non_negative = lambda value: max(0.0, float(value))
        return cls(
            long_utility_atr=utility("long"),
            short_utility_atr=utility("short"),
            tree_min_edge_atr=float(tree_min_edge_atr),
            long_mfe_atr=long_mfe,
            long_mae_atr=long_mae,
            short_mfe_atr=short_mfe,
            short_mae_atr=short_mae,
            tree_utility_clip_atr=cap,
            auxiliary_loss_weight=non_negative(auxiliary_loss_weight),
            utility_loss_weight=non_negative(utility_loss_weight),
            excursion_loss_weight=non_negative(excursion_loss_weight),
            no_edge_loss_weight=non_negative(no_edge_loss_weight),
            direction_loss_weight=non_negative(direction_loss_weight),
            edge_hold_ceiling=float(np.clip(float(edge_hold_ceiling), 0.0, 1.0)),
            hold_ceiling_loss_weight=non_negative(hold_ceiling_loss_weight),
            flat_hold_missed_edge_penalty_r=non_negative(flat_hold_missed_edge_penalty_r),
            flat_hold_confirmed_no_edge_bonus_r=non_negative(flat_hold_confirmed_no_edge_bonus_r),
            flat_hold_edge_threshold_r=non_negative(flat_hold_edge_threshold_r),
            flat_hold_no_edge_threshold_r=min(0.0, float(flat_hold_no_edge_threshold_r)),
            ranking_delta_r=non_negative(ranking_delta_r),
            ranking_margin_scale=non_negative(ranking_margin_scale),
            ranking_margin_min=non_negative(ranking_margin_min),
            ranking_margin_max=max(non_negative(ranking_margin_min), non_negative(ranking_margin_max)),
            ranking_loss_weight=non_negative(ranking_loss_weight),
            action_quality_reward_scale_r=non_negative(action_quality_reward_scale_r),
            horizon_loss_weight=non_negative(horizon_loss_weight),
            path_dynamics_loss_weight=non_negative(path_dynamics_loss_weight),
            first_hit_loss_weight=non_negative(first_hit_loss_weight),
            action_path_loss_weight=non_negative(action_path_loss_weight),
            action_outcome_loss_weight=non_negative(action_outcome_loss_weight),
        )

    @classmethod
    def from_entry_preference_record(
        cls,
        record: EntryPreferenceRecord,
        **config: float,
    ) -> "PatternDiscoveryRewardOracle":
        if not isinstance(record, EntryPreferenceRecord):
            raise TypeError("record phai la EntryPreferenceRecord")
        atr = float(record.atr)
        if not np.isfinite(atr) or atr <= 1e-12:
            raise ValueError("pattern MT5 record can ATR duong")
        bars = tuple(record.bars)
        if not bars:
            raise ValueError("pattern MT5 record chua co path da dong")
        bid, ask = float(record.bid), float(record.ask)
        if np.isfinite(bid) and np.isfinite(ask) and bid > 0.0 and ask >= bid:
            reference = 0.5 * (bid + ask)
            spread_cost_atr = max(0.0, ask - bid) / atr
        else:
            reference = float(bars[0].close)
            spread_cost_atr = 0.0
        highs = np.asarray([float(bar.high) for bar in bars], dtype=np.float64)
        lows = np.asarray([float(bar.low) for bar in bars], dtype=np.float64)
        closes = np.asarray([float(bar.close) for bar in bars], dtype=np.float64)
        if not (np.all(np.isfinite(highs)) and np.all(np.isfinite(lows)) and np.all(np.isfinite(closes))):
            raise ValueError("pattern MT5 path phai huu han")
        final_return = (float(closes[-1]) - reference) / atr
        base = cls.from_sample(
            {
                "long_final_return_atr": final_return,
                "long_mfe_atr": max(0.0, float(np.max(highs - reference)) / atr),
                "long_mae_atr": min(0.0, float(np.min(lows - reference)) / atr),
                "short_final_return_atr": -final_return,
                "short_mfe_atr": max(0.0, float(np.max(reference - lows)) / atr),
                "short_mae_atr": min(0.0, float(np.min(reference - highs)) / atr),
            },
            spread_cost_atr=spread_cost_atr,
            **config,
        )
        components = replay_entry_component_matrix(record)[1:]
        diagnostics = replay_entry_path_diagnostic_matrix(record)[1:]

        horizon_returns, horizon_mask = [], []
        for horizon in (4, 8, 16, 32, 48):
            if len(bars) >= horizon:
                raw = (float(closes[horizon - 1]) - reference) / atr
                horizon_returns.append(float(np.tanh(raw / max(0.5 * base.tree_utility_clip_atr, 1e-6))))
                horizon_mask.append(1.0)
            else:
                horizon_returns.append(0.0)
                horizon_mask.append(0.0)

        market_long, market_short = diagnostics[0], diagnostics[1]
        path_dynamics = tuple(
            float(value)
            for item in (market_long, market_short)
            for value in (
                item.time_to_mfe,
                item.time_to_mae,
                item.underwater_fraction,
                item.recovery_fraction,
                item.positive_persistence,
            )
        )

        def side_hit(outcome: int) -> int:
            return 1 if int(outcome) == 2 else 2 if int(outcome) == 3 else 0

        action_path: list[float] = []
        horizon_denominator = max(1, len(bars) - 1)
        for component, diagnostic in zip(components, diagnostics):
            action_path.extend((
                float(diagnostic.filled),
                float(np.clip(diagnostic.fill_bar / horizon_denominator, 0.0, 1.0)) if diagnostic.filled else 0.0,
                float(np.tanh(component.terminal_r / 2.0)),
                float(np.tanh(max(0.0, component.mfe_r) / 2.0)),
                float(np.tanh(max(0.0, -component.mae_r) / 2.0)),
            ))

        q_values = (0.0,) + tuple(float(item.quality_r) for item in components)
        best_entry = float(max(q_values[1:], default=0.0))
        missed = 0.0
        safe = 0.0
        if best_entry >= base.flat_hold_edge_threshold_r:
            missed = -base.flat_hold_missed_edge_penalty_r * float(np.clip(
                best_entry / max(4.0 * base.flat_hold_edge_threshold_r, 1e-6), 0.0, 1.0
            ))
        elif best_entry <= base.flat_hold_no_edge_threshold_r:
            safe = base.flat_hold_confirmed_no_edge_bonus_r * float(np.clip(
                best_entry / min(base.flat_hold_no_edge_threshold_r, -1e-6), 0.0, 1.0
            ))

        return replace(
            base,
            horizon_returns_norm=tuple(horizon_returns),
            horizon_mask=tuple(horizon_mask),
            path_dynamics_norm=path_dynamics,
            first_hit_classes=(side_hit(market_long.outcome_class), side_hit(market_short.outcome_class)),
            action_path_norm=tuple(action_path),
            action_outcome_classes=tuple(int(item.outcome_class) for item in diagnostics),
            action_utility_atr=tuple(float(item.quality_r * record.sl_atr) for item in components),
            action_path_reward_r=(0.0,) * 6,
            action_q_values_r=q_values,
            action_quality_reward_scale_r=float(base.action_quality_reward_scale_r),
            hold_reward_components_r=(float(missed), float(safe)),
            flat_account_anchor=bool(
                str(record.sample_scope).lower() == "flat_v1"
                and int(record.positions_before) == 0 and int(record.orders_before) == 0
            ),
            rich_path_available=True,
        )

    def auxiliary_targets(self) -> dict[str, object]:
        cap = max(float(self.tree_utility_clip_atr), 1e-6)
        best = max(float(self.long_utility_atr), float(self.short_utility_atr))
        no_edge = best <= float(self.tree_min_edge_atr)
        direction_class = 0 if no_edge else 1 if self.long_utility_atr >= self.short_utility_atr else 2
        return {
            "utility": np.asarray([
                np.clip(float(self.long_utility_atr) / cap, -1.0, 1.0),
                np.clip(float(self.short_utility_atr) / cap, -1.0, 1.0),
            ], dtype=np.float32),
            "excursion": np.asarray([
                np.clip(float(self.long_mfe_atr) / cap, 0.0, 1.0),
                np.clip(-float(self.long_mae_atr) / cap, 0.0, 1.0),
                np.clip(float(self.short_mfe_atr) / cap, 0.0, 1.0),
                np.clip(-float(self.short_mae_atr) / cap, 0.0, 1.0),
            ], dtype=np.float32),
            "no_edge": float(no_edge),
            "direction_class": int(direction_class),
            "auxiliary_loss_weight": float(self.auxiliary_loss_weight),
            "utility_loss_weight": float(self.utility_loss_weight),
            "excursion_loss_weight": float(self.excursion_loss_weight),
            "no_edge_loss_weight": float(self.no_edge_loss_weight),
            "direction_loss_weight": float(self.direction_loss_weight),
            "edge_hold_ceiling": float(self.edge_hold_ceiling),
            "hold_ceiling_loss_weight": float(self.hold_ceiling_loss_weight),
            "rich_path_available": bool(self.rich_path_available),
            "horizon_returns": np.asarray(self.horizon_returns_norm, dtype=np.float32),
            "horizon_mask": np.asarray(self.horizon_mask, dtype=np.float32),
            "path_dynamics": np.asarray(self.path_dynamics_norm, dtype=np.float32),
            "first_hit_classes": np.asarray(self.first_hit_classes, dtype=np.int64),
            "action_path": np.asarray(self.action_path_norm, dtype=np.float32).reshape(6, 5),
            "action_outcome_classes": np.asarray(self.action_outcome_classes, dtype=np.int64),
            "horizon_loss_weight": float(self.horizon_loss_weight if self.rich_path_available else 0.0),
            "path_dynamics_loss_weight": float(self.path_dynamics_loss_weight if self.rich_path_available else 0.0),
            "first_hit_loss_weight": float(self.first_hit_loss_weight if self.rich_path_available else 0.0),
            "action_path_loss_weight": float(self.action_path_loss_weight if self.rich_path_available else 0.0),
            "action_outcome_loss_weight": float(self.action_outcome_loss_weight if self.rich_path_available else 0.0),
            "action_q_values_r": np.asarray(self.action_q_values_r, dtype=np.float32),
            "ranking_delta_r": float(self.ranking_delta_r),
            "ranking_margin_scale": float(self.ranking_margin_scale),
            "ranking_margin_min": float(self.ranking_margin_min),
            "ranking_margin_max": float(self.ranking_margin_max),
            "ranking_loss_weight": float(self.ranking_loss_weight if self.rich_path_available else 0.0),
        }

    def rewards(self, actions: Sequence[int], actor_eligible: Sequence[bool]) -> np.ndarray:
        if len(actions) != 5 or len(actor_eligible) != 5:
            raise ValueError("pattern discovery reward requires five actions and masks")
        result = np.zeros(5, dtype=np.float32)
        if not self.rich_path_available:
            return result
        for index, (action, eligible) in enumerate(zip(actions, actor_eligible)):
            if not bool(eligible):
                continue
            action = int(action)
            if action in ENTRY_ACTIONS and self.action_quality_reward_scale_r > 0.0:
                q_values = np.asarray(self.action_q_values_r, dtype=np.float64)
                # Do not turn a genuinely no-edge path into a blanket
                # negative lesson for every entry style.  The tree/no-edge
                # target handles that case; signed entry shaping is reserved
                # for paths where at least one entry clears the edge gate.
                best_entry = float(np.max(q_values[1:])) if q_values.size > 1 else 0.0
                if action < q_values.size and best_entry >= float(self.flat_hold_edge_threshold_r):
                    result[index] = np.float32(
                        float(self.action_quality_reward_scale_r)
                        * np.tanh(float(q_values[action]) / max(float(self.tree_utility_clip_atr), 1.0))
                    )
            elif action == 0 and self.flat_account_anchor:
                result[index] = np.float32(sum(self.hold_reward_components_r))
        return result

    def reward_diagnostics(self) -> dict[str, object]:
        return {
            "rich_path_available": bool(self.rich_path_available),
            "action_path_reward_r": [float(value) for value in self.action_path_reward_r],
            "component_names": list(PATH_REWARD_COMPONENTS),
            "component_mean_r": {
                name: float(self.hold_reward_components_r[index])
                for index, name in enumerate(PATH_REWARD_COMPONENTS)
            },
            "hold_reward_r": float(sum(self.hold_reward_components_r)),
            "action_quality_reward_scale_r": float(self.action_quality_reward_scale_r),
            "action_q_values_r": [float(value) for value in self.action_q_values_r],
            "best_q_action": int(np.argmax(np.asarray(self.action_q_values_r, dtype=np.float64))),
        }
