"""Causal multi-task targets derived from an immutable entry path.

The targets are training-only.  They enrich the Direct-RNN representation and
never participate in action selection, risk sizing, routing or execution.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .entry_preference import (
    ENTRY_PREFERENCE_ACTIONS,
    EntryPreferenceRecord,
    replay_entry_component_matrix,
    replay_entry_timing_comparisons,
)


RETURN_HORIZONS_MINUTES = (20, 60, 240, 360, 600)
RETURN_HORIZON_NAMES = ("h1_3", "h1", "h4", "h6", "h10")
GLOBAL_AUXILIARY_NAMES = (
    tuple(f"future_direction_{name}" for name in RETURN_HORIZON_NAMES)
    + tuple(
        f"return_quantile_{name}_q{quantile}"
        for name in RETURN_HORIZON_NAMES
        for quantile in (10, 50, 90)
    )
    + tuple(f"realized_volatility_{name}" for name in RETURN_HORIZON_NAMES)
)
ACTION_AUXILIARY_NAMES = (
    "fill_probability",
    "time_to_positive",
    "expected_r",
    "sl_probability",
    "tp_probability",
    "timing_quality",
    "mfe_r",
    "mae_r",
    # Relative, action-local timing labels.  These compare the same entry style
    # now versus 1/2/4/8/12 bars later; they do not encode market direction.
    "enter_now_probability",
    "timing_advantage_r",
    "optimal_delay_fraction",
    # Categorical delay target.  It is masked when the best and second-best
    # delay utilities are too close, so ambiguous paths do not become a hard
    # timing label.
    "optimal_delay_0",
    "optimal_delay_1",
    "optimal_delay_2",
    "optimal_delay_4",
    "optimal_delay_8",
    "optimal_delay_12",
)
TIMING_DELAY_BUCKETS = (0, 1, 2, 4, 8, 12)
ENTRY_ACTIONS = tuple(action for action in ENTRY_PREFERENCE_ACTIONS if action != 0)
AUXILIARY_TARGET_NAMES = GLOBAL_AUXILIARY_NAMES + tuple(
    f"action_{action}_{name}" for action in ENTRY_ACTIONS for name in ACTION_AUXILIARY_NAMES
)
AUXILIARY_OUTPUT_COUNT = len(AUXILIARY_TARGET_NAMES)

# BCE outputs are trained as logits. Quantile outputs use pinball loss; all
# remaining outputs use bounded Huber regression.
AUXILIARY_BCE_INDICES = frozenset(
    tuple(range(len(RETURN_HORIZONS_MINUTES)))
    + tuple(
        len(GLOBAL_AUXILIARY_NAMES) + action_offset * len(ACTION_AUXILIARY_NAMES) + local
        for action_offset in range(len(ENTRY_ACTIONS))
        for local in (
            ACTION_AUXILIARY_NAMES.index("fill_probability"),
            ACTION_AUXILIARY_NAMES.index("sl_probability"),
            ACTION_AUXILIARY_NAMES.index("tp_probability"),
            ACTION_AUXILIARY_NAMES.index("enter_now_probability"),
        )
    )
)
AUXILIARY_QUANTILES = {
    len(RETURN_HORIZONS_MINUTES) + horizon * 3 + offset: quantile
    for horizon in range(len(RETURN_HORIZONS_MINUTES))
    for offset, quantile in enumerate((0.10, 0.50, 0.90))
}
AUXILIARY_CATEGORICAL_GROUPS = tuple(
    tuple(
        len(GLOBAL_AUXILIARY_NAMES)
        + action_offset * len(ACTION_AUXILIARY_NAMES)
        + ACTION_AUXILIARY_NAMES.index(f"optimal_delay_{delay}")
        for delay in TIMING_DELAY_BUCKETS
    )
    for action_offset in range(len(ENTRY_ACTIONS))
)


@dataclass(frozen=True)
class AuxiliaryTargets:
    values: np.ndarray
    mask: np.ndarray


def _future_close(record: EntryPreferenceRecord, horizon: int) -> float | None:
    try:
        index = RETURN_HORIZONS_MINUTES.index(int(horizon))
    except ValueError:
        return None
    if len(record.future_closes) == len(RETURN_HORIZONS_MINUTES):
        close = float(record.future_closes[index])
        return close if np.isfinite(close) and close > 0.0 else None
    if len(record.bars) < int(horizon):
        return None
    return float(record.bars[int(horizon) - 1].close)


def _future_return_r(record: EntryPreferenceRecord, horizon: int) -> float | None:
    bars = tuple(record.bars)
    if not bars or record.atr <= 1e-12:
        return None
    close = _future_close(record, horizon)
    if close is None:
        return None
    start = 0.5 * (float(record.bid) + float(record.ask))
    return float(np.clip((close - start) / float(record.atr), -5.0, 5.0))


def build_auxiliary_targets(record: EntryPreferenceRecord) -> AuxiliaryTargets:
    """Create masked dense targets without substituting unavailable horizons."""
    values = np.zeros(AUXILIARY_OUTPUT_COUNT, dtype=float)
    mask = np.zeros(AUXILIARY_OUTPUT_COUNT, dtype=bool)
    returns = [_future_return_r(record, horizon) for horizon in RETURN_HORIZONS_MINUTES]
    direction_start = 0
    quantile_start = len(RETURN_HORIZONS_MINUTES)
    volatility_start = quantile_start + 3 * len(RETURN_HORIZONS_MINUTES)
    for horizon_index, result in enumerate(returns):
        if result is None:
            continue
        values[direction_start + horizon_index] = float(result > 0.0)
        mask[direction_start + horizon_index] = True
        for quantile_offset in range(3):
            target_index = quantile_start + horizon_index * 3 + quantile_offset
            values[target_index] = float(result)
            mask[target_index] = True
        if len(record.future_volatilities) == len(RETURN_HORIZONS_MINUTES):
            volatility = float(record.future_volatilities[horizon_index])
        else:
            horizon = RETURN_HORIZONS_MINUTES[horizon_index]
            closes = np.asarray([float(bar.close) for bar in record.bars[:horizon]], dtype=float)
            volatility = (
                float(np.std(np.diff(np.log(np.maximum(closes, 1e-12)))) * np.sqrt(closes.size))
                if closes.size >= horizon and closes.size >= 2 else float("nan")
            )
        if np.isfinite(volatility):
            values[volatility_start + horizon_index] = float(np.clip(volatility, 0.0, 1.0))
            mask[volatility_start + horizon_index] = True

    components = {int(item.action): item for item in replay_entry_component_matrix(record)}
    timing = {
        int(item.action): item for item in replay_entry_timing_comparisons(record)
    }
    offset = len(GLOBAL_AUXILIARY_NAMES)
    horizon = max(1, len(record.bars))
    for action in ENTRY_ACTIONS:
        item = components[action]
        local = offset + (action - 1) * len(ACTION_AUXILIARY_NAMES)
        timing_item = timing[action]
        max_delay = max(1, max((1, 2, 4, 8, 12)))
        values[local:local + len(ACTION_AUXILIARY_NAMES)] = (
            float(item.filled),
            1.0 if item.bars_to_positive < 0 else float(np.clip(item.bars_to_positive / horizon, 0.0, 1.0)),
            float(np.clip(item.terminal_r / 3.0, -1.0, 1.0)),
            float(item.filled and item.terminal_r <= -0.999),
            float(item.filled and item.terminal_r >= float(record.rr) - 1e-6),
            float(np.clip(1.0 + item.timing_r / 2.0, 0.0, 1.0)),
            float(np.clip(item.mfe_r / 3.0, 0.0, 1.0)),
            float(np.clip(item.mae_r / 3.0, -1.0, 0.0)),
            float(timing_item.enter_now_probability),
            float(np.clip(timing_item.timing_advantage_r / 3.0, -1.0, 1.0)),
            float(np.clip(timing_item.best_delay_bars / max_delay, 0.0, 1.0)),
            *(float(timing_item.best_delay_bars == delay) for delay in TIMING_DELAY_BUCKETS),
        )
        # Fill is always observable. Remaining path-quality targets are valid
        # only after that hypothetical order actually fills.
        mask[local] = True
        if item.filled:
            mask[local + 1:local + 8] = True
        # Relative timing is observable even when a pending order never fills
        # now: a delayed version of that same order may still fill.  It only
        # requires at least one fully causal delayed decision path.
        if timing_item.valid_delays > 0:
            timing_start = local + ACTION_AUXILIARY_NAMES.index("enter_now_probability")
            mask[timing_start:timing_start + 3] = True
            if timing_item.delay_margin_r >= 0.05:
                delay_start = local + ACTION_AUXILIARY_NAMES.index("optimal_delay_0")
                mask[delay_start:delay_start + len(TIMING_DELAY_BUCKETS)] = True
    return AuxiliaryTargets(values=values, mask=mask)


def auxiliary_loss_and_ascent(
    logits: np.ndarray,
    targets: AuxiliaryTargets,
    *,
    huber_delta: float = 0.25,
    calibration_weight: float = 0.10,
) -> tuple[float, np.ndarray, dict[str, float]]:
    """Return mean loss and the ascent direction expected by policy AdamW."""
    prediction = np.asarray(logits, dtype=float).reshape(-1)
    values = np.asarray(targets.values, dtype=float).reshape(-1)
    mask = np.asarray(targets.mask, dtype=bool).reshape(-1)
    if prediction.size != AUXILIARY_OUTPUT_COUNT or values.size != prediction.size or mask.size != prediction.size:
        raise ValueError("auxiliary target/head shape mismatch")
    gradient = np.zeros_like(prediction)
    loss = 0.0
    counts = {"bce": 0.0, "quantile": 0.0, "huber": 0.0, "categorical": 0.0, "brier": 0.0}
    categorical_indices: set[int] = set()
    for group in AUXILIARY_CATEGORICAL_GROUPS:
        if not all(bool(mask[index]) for index in group):
            continue
        group_logits = np.clip(prediction[list(group)], -30.0, 30.0)
        shifted = group_logits - float(np.max(group_logits))
        probabilities = np.exp(shifted)
        probabilities /= max(float(np.sum(probabilities)), 1e-12)
        group_target = np.maximum(values[list(group)], 0.0)
        group_target /= max(float(np.sum(group_target)), 1e-12)
        loss += -float(np.sum(group_target * np.log(np.maximum(probabilities, 1e-12))))
        gradient[list(group)] = group_target - probabilities
        categorical_indices.update(group)
        counts["categorical"] += 1.0
    for index in np.flatnonzero(mask):
        if int(index) in categorical_indices:
            continue
        raw = float(np.clip(prediction[index], -30.0, 30.0))
        target = float(values[index])
        if index in AUXILIARY_BCE_INDICES:
            probability = 1.0 / (1.0 + float(np.exp(-raw)))
            loss += float(np.logaddexp(0.0, raw) - target * raw)
            gradient[index] = target - probability
            counts["bce"] += 1.0
            if "enter_now_probability" in AUXILIARY_TARGET_NAMES[index]:
                brier_weight = float(max(0.0, calibration_weight))
                loss += brier_weight * (probability - target) ** 2
                # ascent = -d(loss)/d(logit)
                gradient[index] += (
                    2.0 * brier_weight * (target - probability)
                    * probability * (1.0 - probability)
                )
                counts["brier"] += 1.0
        elif index in AUXILIARY_QUANTILES:
            residual = target - raw
            quantile = AUXILIARY_QUANTILES[index]
            loss += max(quantile * residual, (quantile - 1.0) * residual)
            gradient[index] = quantile if residual >= 0.0 else quantile - 1.0
            counts["quantile"] += 1.0
        else:
            residual = target - raw
            absolute = abs(residual)
            if absolute <= huber_delta:
                loss += 0.5 * residual * residual
                gradient[index] = residual
            else:
                loss += huber_delta * (absolute - 0.5 * huber_delta)
                gradient[index] = huber_delta * np.sign(residual)
            counts["huber"] += 1.0
    eligible = max(1, int(np.count_nonzero(mask)) - sum(
        len(group) - 1 for group in AUXILIARY_CATEGORICAL_GROUPS
        if all(bool(mask[index]) for index in group)
    ))
    return float(loss / eligible), gradient / float(eligible), counts
