from __future__ import annotations

"""Reward dinh hinh vong doi V4 cho dung entry da duoc khop."""

from dataclasses import dataclass
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np


ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})
PENDING_ACTIONS = frozenset({3, 4, 5, 6})
INITIAL_SL_EXITS = frozenset({"INITIAL_SL", "SL", "STOP_LOSS"})


@dataclass(frozen=True, init=False)
class FiveLifecycleRewardConfig:
    """Mot hop dong V4 duy nhat, thay the toan bo cac nguong V3 cu."""

    enabled: bool = True
    shape_clip_r: float = 0.75
    min_sl_atr: float = 1.5
    min_tp_atr: float = 3.0
    geometry_penalty_r: float = 0.30
    geometry_bonus_r: float = 0.05
    mfe_penalty_zero_r: float = 0.40
    mfe_penalty_010_r: float = 0.30
    mfe_penalty_025_r: float = 0.15
    mfe_penalty_neutral_at_r: float = 0.50
    excursion_bonus_at_r: float = 1.0
    excursion_good_mae_r: float = 0.25
    excursion_bad_mae_r: float = 1.0
    excursion_min_bonus_r: float = 0.15
    excursion_max_bonus_r: float = 0.30
    trailing_activation_mfe_r: float = 0.50
    trailing_capture_min_ratio: float = 0.50
    trailing_capture_target_ratio: float = 0.80
    trailing_missed_penalty_r: float = 0.20
    trailing_small_capture_penalty_r: float = 0.10
    trailing_min_bonus_r: float = 0.10
    trailing_max_bonus_r: float = 0.25
    exit_capture_min_ratio: float = 0.25
    exit_capture_target_ratio: float = 0.80
    exit_capture_min_bonus_r: float = 0.10
    exit_capture_max_bonus_r: float = 0.30
    recovery_min_bonus_r: float = 0.05
    recovery_max_bonus_r: float = 0.15
    pending_fill_max_bars: int = 5
    pending_fill_min_bonus_r: float = 0.05
    pending_fill_max_bonus_r: float = 0.10

    # Extended Lifecycle Settlement Reward (Milestone 1) - no annotations so they do not enter __dataclass_fields__
    extended_settlement_enabled = False
    extended_lifecycle_enabled = False
    extended_alpha_mae = 0.20
    extended_beta_time = 0.25
    extended_gamma_mfe = 0.10
    extended_timeout_penalty_r = 0.15
    extended_max_holding_bars = 120
    extended_clip_min_r = -5.0
    extended_clip_max_r = 5.0

    def __init__(
        self,
        enabled: bool = True,
        shape_clip_r: float = 0.75,
        min_sl_atr: float = 1.5,
        min_tp_atr: float = 3.0,
        geometry_penalty_r: float = 0.30,
        geometry_bonus_r: float = 0.05,
        mfe_penalty_zero_r: float = 0.40,
        mfe_penalty_010_r: float = 0.30,
        mfe_penalty_025_r: float = 0.15,
        mfe_penalty_neutral_at_r: float = 0.50,
        excursion_bonus_at_r: float = 1.0,
        excursion_good_mae_r: float = 0.25,
        excursion_bad_mae_r: float = 1.0,
        excursion_min_bonus_r: float = 0.15,
        excursion_max_bonus_r: float = 0.30,
        trailing_activation_mfe_r: float = 0.50,
        trailing_capture_min_ratio: float = 0.50,
        trailing_capture_target_ratio: float = 0.80,
        trailing_missed_penalty_r: float = 0.20,
        trailing_small_capture_penalty_r: float = 0.10,
        trailing_min_bonus_r: float = 0.10,
        trailing_max_bonus_r: float = 0.25,
        exit_capture_min_ratio: float = 0.25,
        exit_capture_target_ratio: float = 0.80,
        exit_capture_min_bonus_r: float = 0.10,
        exit_capture_max_bonus_r: float = 0.30,
        recovery_min_bonus_r: float = 0.05,
        recovery_max_bonus_r: float = 0.15,
        pending_fill_max_bars: int = 5,
        pending_fill_min_bonus_r: float = 0.05,
        pending_fill_max_bonus_r: float = 0.10,
        *,
        extended_settlement_enabled: bool = False,
        extended_lifecycle_enabled: bool = False,
        extended_alpha_mae: float = 0.20,
        extended_beta_time: float = 0.25,
        extended_gamma_mfe: float = 0.10,
        extended_timeout_penalty_r: float = 0.15,
        extended_max_holding_bars: int = 120,
        extended_clip_min_r: float = -5.0,
        extended_clip_max_r: float = 5.0,
        **kwargs: Any,
    ) -> None:
        set_attr = object.__setattr__
        set_attr(self, "enabled", bool(enabled))
        set_attr(self, "shape_clip_r", float(shape_clip_r))
        set_attr(self, "min_sl_atr", float(min_sl_atr))
        set_attr(self, "min_tp_atr", float(min_tp_atr))
        set_attr(self, "geometry_penalty_r", float(geometry_penalty_r))
        set_attr(self, "geometry_bonus_r", float(geometry_bonus_r))
        set_attr(self, "mfe_penalty_zero_r", float(mfe_penalty_zero_r))
        set_attr(self, "mfe_penalty_010_r", float(mfe_penalty_010_r))
        set_attr(self, "mfe_penalty_025_r", float(mfe_penalty_025_r))
        set_attr(self, "mfe_penalty_neutral_at_r", float(mfe_penalty_neutral_at_r))
        set_attr(self, "excursion_bonus_at_r", float(excursion_bonus_at_r))
        set_attr(self, "excursion_good_mae_r", float(excursion_good_mae_r))
        set_attr(self, "excursion_bad_mae_r", float(excursion_bad_mae_r))
        set_attr(self, "excursion_min_bonus_r", float(excursion_min_bonus_r))
        set_attr(self, "excursion_max_bonus_r", float(excursion_max_bonus_r))
        set_attr(self, "trailing_activation_mfe_r", float(trailing_activation_mfe_r))
        set_attr(self, "trailing_capture_min_ratio", float(trailing_capture_min_ratio))
        set_attr(self, "trailing_capture_target_ratio", float(trailing_capture_target_ratio))
        set_attr(self, "trailing_missed_penalty_r", float(trailing_missed_penalty_r))
        set_attr(self, "trailing_small_capture_penalty_r", float(trailing_small_capture_penalty_r))
        set_attr(self, "trailing_min_bonus_r", float(trailing_min_bonus_r))
        set_attr(self, "trailing_max_bonus_r", float(trailing_max_bonus_r))
        set_attr(self, "exit_capture_min_ratio", float(exit_capture_min_ratio))
        set_attr(self, "exit_capture_target_ratio", float(exit_capture_target_ratio))
        set_attr(self, "exit_capture_min_bonus_r", float(exit_capture_min_bonus_r))
        set_attr(self, "exit_capture_max_bonus_r", float(exit_capture_max_bonus_r))
        set_attr(self, "recovery_min_bonus_r", float(recovery_min_bonus_r))
        set_attr(self, "recovery_max_bonus_r", float(recovery_max_bonus_r))
        set_attr(self, "pending_fill_max_bars", int(pending_fill_max_bars))
        set_attr(self, "pending_fill_min_bonus_r", float(pending_fill_min_bonus_r))
        set_attr(self, "pending_fill_max_bonus_r", float(pending_fill_max_bonus_r))
        # Extended fields
        set_attr(self, "extended_settlement_enabled", bool(extended_settlement_enabled or extended_lifecycle_enabled))
        set_attr(self, "extended_lifecycle_enabled", bool(extended_lifecycle_enabled or extended_settlement_enabled))
        set_attr(self, "extended_alpha_mae", float(extended_alpha_mae))
        set_attr(self, "extended_beta_time", float(extended_beta_time))
        set_attr(self, "extended_gamma_mfe", float(extended_gamma_mfe))
        set_attr(self, "extended_timeout_penalty_r", float(extended_timeout_penalty_r))
        set_attr(self, "extended_max_holding_bars", int(extended_max_holding_bars))
        set_attr(self, "extended_clip_min_r", float(extended_clip_min_r))
        set_attr(self, "extended_clip_max_r", float(extended_clip_max_r))
        for k, v in kwargs.items():
            set_attr(self, k, v)

    @classmethod
    def from_source(cls, source: Any) -> "FiveLifecycleRewardConfig":
        defaults = cls()

        def read(field: str) -> Any:
            if field == "enabled":
                return getattr(source, "five_lifecycle_reward_enabled", defaults.enabled)
            return getattr(source, f"five_lifecycle_{field}", getattr(defaults, field))

        values: dict[str, Any] = {}
        for field in defaults.__dataclass_fields__:
            default = getattr(defaults, field)
            raw = read(field)
            values[field] = bool(raw) if isinstance(default, bool) else int(raw) if isinstance(default, int) else float(raw)

        for ext in (
            "extended_settlement_enabled", "extended_lifecycle_enabled",
            "extended_alpha_mae", "extended_beta_time", "extended_gamma_mfe",
            "extended_timeout_penalty_r", "extended_max_holding_bars",
            "extended_clip_min_r", "extended_clip_max_r",
        ):
            if hasattr(source, f"five_lifecycle_{ext}"):
                default_val = getattr(defaults, ext)
                raw = getattr(source, f"five_lifecycle_{ext}")
                values[ext] = bool(raw) if isinstance(default_val, bool) else int(raw) if isinstance(default_val, int) else float(raw)

        return cls(**values)


@dataclass(frozen=True)
class FiveLifecycleRewardBreakdown:
    """Tat ca thanh phan da co dau; ``total_r`` luon nam trong gioi han."""

    geometry_quality_r: float = 0.0
    entry_excursion_quality_r: float = 0.0
    trailing_quality_r: float = 0.0
    exit_capture_quality_r: float = 0.0
    recovery_persistence_quality_r: float = 0.0
    pending_fill_quality_r: float = 0.0
    unclipped_total_r: float = 0.0
    total_r: float = 0.0
    initial_sl: bool = False
    initial_sl_failure: bool = False
    trailing_evaluated: bool = False
    geometry_too_close: bool = False

    # Extended Lifecycle Settlement Components (Milestone 1)
    realized_r: float = 0.0
    holding_time_penalty_r: float = 0.0
    mae_flooding_penalty_r: float = 0.0
    mfe_harvest_bonus_r: float = 0.0
    timeout_penalty_r: float = 0.0
    extended_settlement: bool = False
    duration_bars: float = 0.0
    is_timeout: bool = False

    def as_dict(self) -> dict[str, float | bool]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def lifecycle_path_dynamics(
    trajectory: Sequence[Mapping[str, object]] | None,
) -> tuple[float, float]:
    """Tra ve (hoi_phuc_sau_MAE, do_ben_co_loi) tu cac diem R da xac nhan."""

    points: list[tuple[float, float]] = []
    for index, raw in enumerate(tuple(trajectory or ())):
        try:
            age = float(raw.get("age_bars", index))
            value = float(raw.get("unrealized_r", np.nan))
        except (TypeError, ValueError):
            continue
        if np.isfinite(age) and np.isfinite(value):
            points.append((age, value))
    if not points:
        return 0.0, 0.0
    points.sort(key=lambda item: item[0])
    values = np.asarray([item[1] for item in points], dtype=np.float64)
    mae_index = int(np.argmin(values))
    mae = float(values[mae_index])
    post_mae_peak = float(np.max(values[mae_index:]))
    full_span = max(1e-6, float(np.max(values)) - mae)
    recovery = float(np.clip((post_mae_peak - mae) / full_span, 0.0, 1.0))
    positive = np.flatnonzero(values > 0.0)
    persistence = float(np.mean(values[int(positive[0]):] > 0.0)) if positive.size else 0.0
    return recovery, persistence


def _linear(value: float, left: float, right: float, low: float, high: float) -> float:
    if right <= left:
        return float(high)
    scale = float(np.clip((value - left) / (right - left), 0.0, 1.0))
    return float(low + (high - low) * scale)


def _soft_mfe_quality(mfe: float, mae: float, config: FiveLifecycleRewardConfig) -> float:
    x = max(0.0, float(mfe))
    knots_x = np.asarray([0.0, 0.10, 0.25, config.mfe_penalty_neutral_at_r], dtype=float)
    knots_y = -np.asarray([
        config.mfe_penalty_zero_r,
        config.mfe_penalty_010_r,
        config.mfe_penalty_025_r,
        0.0,
    ], dtype=float)
    penalty = float(np.interp(min(x, knots_x[-1]), knots_x, knots_y))
    if x < config.excursion_bonus_at_r:
        return penalty
    adverse = max(0.0, -float(mae))
    mae_quality = 1.0 - float(np.clip(
        (adverse - config.excursion_good_mae_r)
        / max(config.excursion_bad_mae_r - config.excursion_good_mae_r, 1e-12),
        0.0,
        1.0,
    ))
    return _linear(
        mae_quality, 0.0, 1.0,
        config.excursion_min_bonus_r, config.excursion_max_bonus_r,
    )



def compute_holding_duration_penalty(
    duration_bars: float,
    beta: float = 0.25,
    max_bars: float = 120.0,
    clamp_max: bool = True,
) -> float:
    """Computes quadratic holding penalty beta * (min(dt, 120)/120)^2."""
    val = float(duration_bars) if np.isfinite(duration_bars) else 0.0
    dt = max(0.0, val)
    denom = max(1e-6, float(max_bars))
    ratio = min(dt / denom, 1.0) if clamp_max else (dt / denom)
    return float(beta * (ratio ** 2))


def compute_mae_flooding_penalty(mae_r: float, alpha: float = 0.20) -> float:
    """Computes internal MAE flooding penalty alpha * |mae_r|."""
    adverse = abs(float(mae_r)) if np.isfinite(mae_r) else 0.0
    return float(alpha * adverse)


def compute_mfe_harvest_reward(
    realized_r: float,
    mfe_r: float,
    gamma: float = 0.10,
    min_mfe_floor: float = 0.05,
) -> float:
    """Computes MFE harvest reward gamma * (realized_r / max(mfe_r, 0.05)) if realized_r > 0."""
    net_r = float(realized_r) if np.isfinite(realized_r) else 0.0
    if net_r <= 0.0:
        return 0.0
    mfe = max(0.0, float(mfe_r)) if np.isfinite(mfe_r) else 0.0
    denom = max(mfe, float(min_mfe_floor))
    efficiency = float(np.clip(net_r / denom, 0.0, 1.0))
    return float(gamma * efficiency)


def compute_timeout_penalty(is_timeout: bool, penalty_r: float = 0.15) -> float:
    """Computes additional timeout expiration penalty -0.15R when is_timeout = True."""
    return float(penalty_r) if bool(is_timeout) else 0.0


def calculate_extended_lifecycle_settlement(
    *,
    realized_r: float,
    mae_r: float = 0.0,
    duration_bars: float = 0.0,
    mfe_r: float = 0.0,
    is_timeout: bool = False,
    holding_bars: float | None = None,
    exit_kind: str = "",
    alpha: float = 0.20,
    beta: float = 0.25,
    gamma: float = 0.10,
    timeout_penalty_r: float = 0.15,
    max_holding_bars: float = 120.0,
    clip_low: float = -5.0,
    clip_high: float = 5.0,
) -> float:
    """Computes canonical Tier-1 Extended Lifecycle Settlement Reward.
    
    R* = realized_R - alpha * MAE_R - beta * (min(dt, 120) / 120)^2 + gamma * Eff_MFE - Penalty_timeout
    """
    bars = float(holding_bars if holding_bars is not None else duration_bars)
    net_r = float(realized_r) if np.isfinite(realized_r) else 0.0
    mae_pen = compute_mae_flooding_penalty(mae_r, alpha=alpha)
    hold_pen = compute_holding_duration_penalty(bars, beta=beta, max_bars=max_holding_bars)
    mfe_rew = compute_mfe_harvest_reward(realized_r, mfe_r, gamma=gamma)
    to_active = bool(is_timeout)
    if not to_active and exit_kind:
        clean_exit = str(exit_kind).strip().upper()
        if clean_exit == "TIMEOUT_EXPIRED" or (
            clean_exit in {"TIME_SAFETY", "TIMEOUT", "EXPIRED"} and bars >= max_holding_bars
        ):
            to_active = True
    to_pen = compute_timeout_penalty(to_active, penalty_r=timeout_penalty_r)

    raw_total = net_r - mae_pen - hold_pen + mfe_rew - to_pen
    return float(np.clip(raw_total, clip_low, clip_high))


class ExtendedLifecycleSettlementEngine:
    """Engine computing Extended Lifecycle Settlement Reward and breakdown dict."""

    def __init__(
        self,
        alpha_mae: float = 0.20,
        beta_time: float = 0.25,
        gamma_mfe: float = 0.10,
        timeout_penalty_r: float = 0.15,
        max_holding_bars: int = 120,
        clip_min_r: float = -5.0,
        clip_max_r: float = 5.0,
    ) -> None:
        self.alpha_mae = float(alpha_mae)
        self.beta_time = float(beta_time)
        self.gamma_mfe = float(gamma_mfe)
        self.timeout_penalty_r = float(timeout_penalty_r)
        self.max_holding_bars = max(1, int(max_holding_bars))
        self.clip_min_r = float(clip_min_r)
        self.clip_max_r = float(clip_max_r)

    def compute_settlement_reward(
        self,
        *,
        realized_r: float,
        mae_r: float = 0.0,
        mfe_r: float = 0.0,
        holding_bars: float | int = 0,
        duration_bars: float | int | None = None,
        exit_kind: str = "",
        is_timeout: bool | None = None,
    ) -> dict[str, float | bool]:
        bars = float(duration_bars if duration_bars is not None else holding_bars)
        realized = float(realized_r) if np.isfinite(realized_r) else 0.0
        time_penalty = compute_holding_duration_penalty(bars, beta=self.beta_time, max_bars=self.max_holding_bars)
        drawdown_penalty = compute_mae_flooding_penalty(mae_r, alpha=self.alpha_mae)
        mfe_bonus = compute_mfe_harvest_reward(realized, mfe_r, gamma=self.gamma_mfe)

        clean_exit = str(exit_kind or "").strip().upper()
        if is_timeout is not None:
            timeout_active = bool(is_timeout)
        else:
            timeout_active = bool(
                clean_exit == "TIMEOUT_EXPIRED"
                or (
                    clean_exit in {"TIME_SAFETY", "TIMEOUT", "EXPIRED"}
                    and bars >= float(self.max_holding_bars)
                )
            )
        timeout_penalty = compute_timeout_penalty(timeout_active, penalty_r=self.timeout_penalty_r)

        raw_reward = float(realized - drawdown_penalty - time_penalty + mfe_bonus - timeout_penalty)
        clipped_reward = float(np.clip(raw_reward, self.clip_min_r, self.clip_max_r))

        return {
            "reward_final": clipped_reward,
            "unclipped_reward": raw_reward,
            "realized_r": realized,
            "time_penalty": time_penalty,
            "drawdown_penalty": drawdown_penalty,
            "mfe_bonus": mfe_bonus,
            "timeout_penalty": timeout_penalty,
            "is_timeout": timeout_active,
            "duration_bars": bars,
        }


def compute_extended_lifecycle_settlement(
    *,
    realized_r: float,
    mae_r: float = 0.0,
    mfe_r: float = 0.0,
    holding_bars: float | int = 0,
    duration_bars: float | int | None = None,
    exit_kind: str = "",
    is_timeout: bool | None = None,
    alpha_mae: float = 0.20,
    beta_time: float = 0.25,
    gamma_mfe: float = 0.10,
    timeout_penalty_r: float = 0.15,
    max_holding_bars: int = 120,
    clip_min_r: float = -5.0,
    clip_max_r: float = 5.0,
) -> dict[str, float | bool]:
    """High-level utility returning detailed breakdown dict."""
    engine = ExtendedLifecycleSettlementEngine(
        alpha_mae=alpha_mae,
        beta_time=beta_time,
        gamma_mfe=gamma_mfe,
        timeout_penalty_r=timeout_penalty_r,
        max_holding_bars=max_holding_bars,
        clip_min_r=clip_min_r,
        clip_max_r=clip_max_r,
    )
    return engine.compute_settlement_reward(
        realized_r=realized_r,
        mae_r=mae_r,
        mfe_r=mfe_r,
        holding_bars=holding_bars,
        duration_bars=duration_bars,
        exit_kind=exit_kind,
        is_timeout=is_timeout,
    )


def shape_five_entry_lifecycle(
    *,
    action_index: int,
    sl_atr: float,
    tp_atr: float,
    trailing_atr: float,
    trailing_applied: bool,
    locked_profit_r: float,
    mfe_r: float,
    mae_r: float,
    mfe_available: bool,
    realized_r: float,
    exit_kind: str,
    recovery_fraction: float = 0.0,
    positive_persistence: float = 0.0,
    pending_fill_bars: int | None = None,
    holding_bars: float | int | None = None,
    duration_bars: float | int | None = None,
    is_timeout: bool | None = None,
    extended_settlement: bool | None = None,
    config: FiveLifecycleRewardConfig,
) -> FiveLifecycleRewardBreakdown:
    """Tinh reward V4 hoac Extended Settlement dung mot lan sau khi lifecycle da dong bat bien."""

    if not config.enabled or int(action_index) not in ENTRY_ACTIONS:
        return FiveLifecycleRewardBreakdown()

    use_extended = bool(
        extended_settlement
        if extended_settlement is not None
        else (config.extended_settlement_enabled or config.extended_lifecycle_enabled)
    )

    bars_held = (
        duration_bars if duration_bars is not None
        else holding_bars if holding_bars is not None
        else 0.0
    )

    if use_extended:
        engine = ExtendedLifecycleSettlementEngine(
            alpha_mae=config.extended_alpha_mae,
            beta_time=config.extended_beta_time,
            gamma_mfe=config.extended_gamma_mfe,
            timeout_penalty_r=config.extended_timeout_penalty_r,
            max_holding_bars=config.extended_max_holding_bars,
            clip_min_r=config.extended_clip_min_r,
            clip_max_r=config.extended_clip_max_r,
        )
        settlement = engine.compute_settlement_reward(
            realized_r=realized_r,
            mae_r=mae_r,
            mfe_r=mfe_r,
            holding_bars=bars_held,
            exit_kind=exit_kind,
            is_timeout=is_timeout,
        )
        initial_sl = str(exit_kind or "").strip().upper() in INITIAL_SL_EXITS
        realized = float(realized_r)
        initial_sl_failure = bool(initial_sl and realized <= 0.0)
        return FiveLifecycleRewardBreakdown(
            realized_r=float(settlement["realized_r"]),
            holding_time_penalty_r=-float(settlement["time_penalty"]),
            mae_flooding_penalty_r=-float(settlement["drawdown_penalty"]),
            mfe_harvest_bonus_r=float(settlement["mfe_bonus"]),
            timeout_penalty_r=-float(settlement["timeout_penalty"]),
            extended_settlement=True,
            duration_bars=float(bars_held),
            is_timeout=bool(settlement["is_timeout"]),
            unclipped_total_r=float(settlement["unclipped_reward"]),
            total_r=float(settlement["reward_final"]),
            initial_sl=bool(initial_sl),
            initial_sl_failure=bool(initial_sl_failure),
            trailing_evaluated=False,
            geometry_too_close=False,
        )

    mfe, mae = float(mfe_r), float(mae_r)
    has_excursion = bool(mfe_available and np.isfinite(mfe) and np.isfinite(mae))
    initial_sl = str(exit_kind or "").strip().upper() in INITIAL_SL_EXITS
    realized = float(realized_r)
    # Mot phan vi the co the da chot loi truoc khi phan con lai cham SL ban
    # dau. Chi outcome INITIAL_SL co net-R khong duong moi la loi entry.
    initial_sl_failure = bool(initial_sl and realized <= 0.0)

    sl_value, tp_value = float(sl_atr), float(tp_atr)
    too_close = bool(
        (np.isfinite(sl_value) and 0.0 < sl_value < config.min_sl_atr)
        or (np.isfinite(tp_value) and 0.0 < tp_value < config.min_tp_atr)
    )
    geometry_valid = bool(
        np.isfinite(sl_value) and sl_value >= config.min_sl_atr
        and np.isfinite(tp_value) and tp_value >= config.min_tp_atr
    )
    geometry_quality = (
        -config.geometry_penalty_r if too_close
        # Hinh hoc hop le chi dang thuong khi entry da tao duoc mot doan
        # thuan loi co y nghia. Tuyet doi khong thuong INITIAL_SL: neu khong,
        # mot entry sai van co scalar duong trong luc loss cuc bo dang co gang
        # ha chinh entry do.
        else config.geometry_bonus_r
        if (
            not initial_sl_failure
            and geometry_valid
            and has_excursion
            and mfe >= config.mfe_penalty_neutral_at_r
        )
        else 0.0
    )

    # INITIAL_SL va MFE thap la cung mot loi entry. INITIAL_SL duoc xu ly boi
    # loss cuc bo ben trong nhom sau entry, nen khong duoc dua them mot scalar
    # am vao PPO va lam tong xac suat entry chay sang HOLD/MANAGE.
    excursion_quality = (
        0.0 if initial_sl_failure or not has_excursion
        else _soft_mfe_quality(mfe, mae, config)
    )

    trailing_requested = bool(np.isfinite(float(trailing_atr)) and float(trailing_atr) > 0.0)
    trailing_evaluated = bool(
        not initial_sl_failure
        and trailing_requested
        and has_excursion
        and mfe >= config.trailing_activation_mfe_r
    )
    trailing_quality = 0.0
    if trailing_evaluated:
        if not bool(trailing_applied):
            trailing_quality = -config.trailing_missed_penalty_r
        else:
            capture = float(np.clip(float(locked_profit_r) / max(mfe, 1e-12), 0.0, 1.0))
            if capture < config.trailing_capture_min_ratio:
                trailing_quality = -config.trailing_small_capture_penalty_r * (
                    1.0 - capture / max(config.trailing_capture_min_ratio, 1e-12)
                )
            else:
                trailing_quality = _linear(
                    capture, config.trailing_capture_min_ratio, config.trailing_capture_target_ratio,
                    config.trailing_min_bonus_r, config.trailing_max_bonus_r,
                )

    exit_capture_quality = 0.0
    if has_excursion and mfe > 0.0 and realized > 0.0:
        capture = float(np.clip(realized / max(mfe, 1e-12), 0.0, 1.0))
        if capture >= config.exit_capture_min_ratio:
            exit_capture_quality = _linear(
                capture, config.exit_capture_min_ratio, config.exit_capture_target_ratio,
                config.exit_capture_min_bonus_r, config.exit_capture_max_bonus_r,
            )

    recovery = float(np.clip(recovery_fraction, 0.0, 1.0))
    persistence = float(np.clip(positive_persistence, 0.0, 1.0))
    recovery_quality = 0.0
    if realized > 0.0 and recovery > 0.0 and persistence > 0.0:
        recovery_quality = _linear(
            recovery * persistence, 0.0, 1.0,
            config.recovery_min_bonus_r, config.recovery_max_bonus_r,
        )

    pending_quality = 0.0
    if (
        not initial_sl_failure
        and int(action_index) in PENDING_ACTIONS
        and pending_fill_bars is not None
        and 0 <= int(pending_fill_bars) <= int(config.pending_fill_max_bars)
        and has_excursion and mfe >= config.mfe_penalty_neutral_at_r
    ):
        speed = 1.0 - float(np.clip(
            int(pending_fill_bars) / max(1, int(config.pending_fill_max_bars)), 0.0, 1.0
        ))
        pending_quality = _linear(
            speed, 0.0, 1.0,
            config.pending_fill_min_bonus_r, config.pending_fill_max_bonus_r,
        )

    raw_total = float(
        geometry_quality + excursion_quality + trailing_quality
        + exit_capture_quality + recovery_quality + pending_quality
    )
    clipped = float(np.clip(raw_total, -config.shape_clip_r, config.shape_clip_r))
    return FiveLifecycleRewardBreakdown(
        geometry_quality_r=float(geometry_quality),
        entry_excursion_quality_r=float(excursion_quality),
        trailing_quality_r=float(trailing_quality),
        exit_capture_quality_r=float(exit_capture_quality),
        recovery_persistence_quality_r=float(recovery_quality),
        pending_fill_quality_r=float(pending_quality),
        unclipped_total_r=raw_total,
        total_r=clipped,
        initial_sl=bool(initial_sl),
        initial_sl_failure=bool(initial_sl_failure),
        trailing_evaluated=bool(trailing_evaluated),
        geometry_too_close=bool(too_close),
    )


def lifecycle_stratum(
    *, action_index: int, realized_r: float, observation: object,
    neutral_band_r: float = 0.05,
) -> tuple[str, str, str]:
    """Nhom action x ket qua x che do; khong thay doi nhan hay reward."""

    action = int(action_index)
    family = "thi_truong" if action in {1, 2} else "gioi_han" if action in {3, 4} else "dung"
    value = float(realized_r)
    outcome = "duong" if value > neutral_band_r else "am" if value < -neutral_band_r else "trung_tinh"
    try:
        from .conditioned_entry_expectancy import entry_condition

        condition = entry_condition(observation, action_index=action)
        volatility = str(condition.get("volatility", "stable"))
        trend = str(condition.get("trend", "neutral"))
    except Exception:
        volatility, trend = "stable", "neutral"
    if trend == "up":
        regime = "xu_huong_tang"
    elif trend == "down":
        regime = "xu_huong_giam"
    elif volatility in {"expanding_high", "high_stable"}:
        regime = "bien_dong"
    else:
        regime = "di_ngang"
    side = "mua" if action in {1, 3, 5} else "ban"
    return f"{side}_{family}", outcome, regime


def pop_stratified_lifecycle_batch(
    queue: list[dict[str, object]], *, batch_size: int, max_wait: int,
) -> list[dict[str, object]]:
    """Lay round-robin theo stratum, khong lap mau va khong sua nhan."""

    target = max(1, int(batch_size))
    if len(queue) < target:
        return []
    outcome_groups = {
        tuple(item.get("lifecycle_stratum", ("", "", "")))[1]
        for item in queue
    }
    if len(outcome_groups) < 2 and len(queue) < max(target, int(max_wait)):
        return []
    buckets: dict[tuple[object, ...], list[int]] = {}
    for index, item in enumerate(queue):
        key = tuple(item.get("lifecycle_stratum", ("khac", "trung_tinh", "di_ngang")))
        buckets.setdefault(key, []).append(index)
    selected: list[int] = []
    ordered_keys = sorted(buckets, key=lambda key: (len(buckets[key]), tuple(map(str, key))))
    while len(selected) < target:
        moved = False
        for key in ordered_keys:
            if buckets[key] and len(selected) < target:
                selected.append(buckets[key].pop(0))
                moved = True
        if not moved:
            break
    chosen = [queue[index] for index in selected]
    selected_set = set(selected)
    queue[:] = [item for index, item in enumerate(queue) if index not in selected_set]
    return chosen
