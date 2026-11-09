"""Hard semantic boundaries for the 137-feature context-expert policy.

The MT5 wire vector intentionally contains market, position, account and
execution facts.  Feeding every field to the same recurrent encoder lets an
entry policy use a shortcut such as a previous loss, open position count or a
pending order as if it were market evidence.  This module is deliberately
small and deterministic: it routes the *existing* wire vector without fitting
statistics, adding a second optimiser, or changing the MT5 CSV contract.

``role_isolated_v1`` has three non-overlapping authorities:

* entry direction and HOLD: market + timing + Wyckoff only;
* order/position management: position + execution only;
* account/risk: deterministic size budget and hard safety veto only.

The policy owns recurrent-state separation and applies the returned route
before the Direct-RNN forward pass.  Keeping this pure module independent of
the policy makes the contract straightforward to unit-test and to audit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from .protocol_contract import (
    ACCOUNT_OFFSET,
    EXECUTION_OFFSET,
    EXTRA_MARKET_OFFSET,
    INPUT_COUNT,
    POSITION_OFFSET,
    TIMING_OFFSET,
)
from .context_history import CONTEXT_HISTORY_INPUT_COUNT


SEMANTIC_INPUT_ROUTING_LEGACY = "legacy"
SEMANTIC_INPUT_ROUTING_ROLE_ISOLATED_V1 = "role_isolated_v1"
SEMANTIC_INPUT_ROUTING_MODES = frozenset({
    SEMANTIC_INPUT_ROUTING_LEGACY,
    SEMANTIC_INPUT_ROUTING_ROLE_ISOLATED_V1,
})
SEMANTIC_INPUT_ROUTING_VERSION = 3

ENTRY_ACTIONS = frozenset({1, 2, 3, 4, 5, 6})
MANAGEMENT_ACTIONS = frozenset({7, 8, 9, 10, 11})
BUY_ENTRY_ACTIONS = frozenset({1, 3, 5})
SELL_ENTRY_ACTIONS = frozenset({2, 4, 6})


def _normalise_actions(values: object | None) -> frozenset[int]:
    """Return valid action ids from the bridge's mapping/list representation."""

    if values is None:
        return frozenset(range(12))
    result: set[int] = set()
    if isinstance(values, Mapping):
        for raw_index, enabled in values.items():
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if bool(enabled) and 0 <= index < 12:
                result.add(index)
    else:
        try:
            mask = np.asarray(values, dtype=bool).reshape(-1)
        except (TypeError, ValueError):
            mask = np.zeros(0, dtype=bool)
        result.update(index for index, enabled in enumerate(mask[:12]) if bool(enabled))
    return frozenset(result or {0})


def _finite_vector(values: object, *, size: int) -> np.ndarray:
    try:
        source = np.asarray(values, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        source = np.zeros(0, dtype=float)
    result = np.zeros(max(0, int(size)), dtype=float)
    result[: min(result.size, source.size)] = source[: result.size]
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


def _finite_mapping(values: Mapping[str, object] | None, key: str) -> float:
    if not isinstance(values, Mapping):
        return 0.0
    try:
        value = float(values.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        value = 0.0
    return value if np.isfinite(value) else 0.0


def _has_open_position(account_context: Mapping[str, object] | None, positions_pipe: object) -> bool:
    if str(positions_pipe or "").strip():
        return True
    return _finite_mapping(account_context, "position_count_norm") > 1e-9


def _mask_for_role(
    role: str,
    input_size: int,
    *,
    context_history_tail_count: int = CONTEXT_HISTORY_INPUT_COUNT,
    entry_local_tail_count: int = 0,
) -> np.ndarray:
    """Create an exact schema mask without assuming percentage offsets."""

    count = max(0, int(input_size))
    mask = np.ones(count, dtype=float)
    if count < INPUT_COUNT:
        # Compatibility artifacts below v10 do not have the semantic layout;
        # fail open rather than silently dropping unknown legacy columns.
        return mask
    mask[:] = 0.0
    if role == "management":
        mask[POSITION_OFFSET:ACCOUNT_OFFSET] = 1.0
        mask[EXECUTION_OFFSET:TIMING_OFFSET] = 1.0
    else:
        # Entry receives the two market blocks plus timing and Wyckoff.  It
        # cannot read position, account, execution or learning-drive fields.
        mask[:POSITION_OFFSET] = 1.0
        mask[EXTRA_MARKET_OFFSET:EXECUTION_OFFSET] = 1.0
        mask[TIMING_OFFSET:INPUT_COUNT] = 1.0
    # Python-local causal context tokens (t-1, t-2) precede the optional
    # interaction tail and are available to both entry and management.  The
    # explicit count matters when an isolated policy opts into interactions
    # without context-history; otherwise its first two interactions would be
    # misclassified as history and management could see them.
    context_tail_end = min(
        count,
        INPUT_COUNT + max(0, min(CONTEXT_HISTORY_INPUT_COUNT, int(context_history_tail_count))),
    )
    if context_tail_end > INPUT_COUNT:
        mask[INPUT_COUNT:context_tail_end] = 1.0
    # Candidate interaction fields are derived solely from the immutable
    # market wire immediately before Direct-RNN inference.  Grant them to the
    # entry route only; management must not acquire market information through
    # an otherwise unknown local tail.
    if role != "management":
        entry_tail_end = min(count, context_tail_end + max(0, int(entry_local_tail_count)))
        if entry_tail_end > context_tail_end:
            mask[context_tail_end:entry_tail_end] = 1.0
    return mask


@dataclass(frozen=True)
class SemanticInputRoute:
    """One deterministic feature route used by an RNN decision."""

    mode: str
    role: str
    allowed_actions: tuple[int, ...]
    observation: np.ndarray
    feature_mask: np.ndarray
    context_history_tail_count: int = CONTEXT_HISTORY_INPUT_COUNT

    @property
    def enabled(self) -> bool:
        return self.mode == SEMANTIC_INPUT_ROUTING_ROLE_ISOLATED_V1

    def audit(self) -> dict[str, object]:
        active = np.flatnonzero(self.feature_mask > 0.0)
        context_tail_end = min(
            self.feature_mask.size,
            INPUT_COUNT + max(0, min(CONTEXT_HISTORY_INPUT_COUNT, int(self.context_history_tail_count))),
        )
        return {
            "semantic_input_routing_mode": self.mode,
            "semantic_input_routing_version": SEMANTIC_INPUT_ROUTING_VERSION,
            "semantic_route": self.role,
            "semantic_route_allowed_actions": list(self.allowed_actions),
            "semantic_route_active_features": int(active.size),
            "semantic_route_masked_features": int(self.feature_mask.size - active.size),
            "semantic_route_market_active": int(np.count_nonzero(self.feature_mask[:POSITION_OFFSET])),
            "semantic_route_position_active": int(np.count_nonzero(self.feature_mask[POSITION_OFFSET:ACCOUNT_OFFSET])),
            "semantic_route_account_active": int(np.count_nonzero(self.feature_mask[ACCOUNT_OFFSET:EXTRA_MARKET_OFFSET])),
            "semantic_route_execution_active": int(np.count_nonzero(self.feature_mask[EXECUTION_OFFSET:TIMING_OFFSET])),
            "semantic_route_timing_active": int(np.count_nonzero(self.feature_mask[TIMING_OFFSET:min(INPUT_COUNT, self.feature_mask.size)])),
            "semantic_route_context_history_active": int(np.count_nonzero(
                self.feature_mask[INPUT_COUNT:context_tail_end]
            )),
            "semantic_route_entry_interaction_active": int(np.count_nonzero(
                self.feature_mask[context_tail_end:self.feature_mask.size]
            )),
        }


def build_semantic_input_route(
    observation: object,
    *,
    mode: str = SEMANTIC_INPUT_ROUTING_LEGACY,
    allowed_actions: object | None = None,
    account_context: Mapping[str, object] | None = None,
    positions_pipe: object = "",
    input_size: int = INPUT_COUNT,
    context_history_tail_count: int = CONTEXT_HISTORY_INPUT_COUNT,
    entry_local_tail_count: int = 0,
) -> SemanticInputRoute:
    """Route a complete observation by decision authority.

    An entry-capable decision always receives the market route.  The context
    facade creates a separate management-only proposal when it wants close or
    trailing actions, so this deterministic precedence prevents a mixed
    entry/management decision from leaking position state into BUY/SELL.
    """

    normalized_mode = str(mode or SEMANTIC_INPUT_ROUTING_LEGACY).strip().lower()
    if normalized_mode not in SEMANTIC_INPUT_ROUTING_MODES:
        raise ValueError(
            f"unsupported semantic input routing mode {mode!r}; "
            f"expected one of {sorted(SEMANTIC_INPUT_ROUTING_MODES)}"
        )
    vector = _finite_vector(observation, size=input_size)
    actions = _normalise_actions(allowed_actions)
    entry_capable = bool(actions.intersection(ENTRY_ACTIONS))
    management_capable = bool(actions.intersection(MANAGEMENT_ACTIONS))
    if normalized_mode == SEMANTIC_INPUT_ROUTING_LEGACY:
        role = "legacy"
        mask = np.ones(vector.size, dtype=float)
    elif entry_capable or not management_capable:
        role = "entry"
        mask = _mask_for_role(
            role,
            vector.size,
            context_history_tail_count=context_history_tail_count,
            entry_local_tail_count=entry_local_tail_count,
        )
    elif _has_open_position(account_context, positions_pipe):
        role = "management"
        mask = _mask_for_role(
            role,
            vector.size,
            context_history_tail_count=context_history_tail_count,
            entry_local_tail_count=entry_local_tail_count,
        )
    else:
        # A HOLD-only empty-book decision is still an entry-timing decision.
        role = "entry"
        mask = _mask_for_role(
            role,
            vector.size,
            context_history_tail_count=context_history_tail_count,
            entry_local_tail_count=entry_local_tail_count,
        )
    return SemanticInputRoute(
        mode=normalized_mode,
        role=role,
        allowed_actions=tuple(sorted(actions)),
        observation=vector * mask,
        feature_mask=mask,
        context_history_tail_count=max(0, min(CONTEXT_HISTORY_INPUT_COUNT, int(context_history_tail_count))),
    )


def parse_open_position_sides(positions_pipe: object) -> frozenset[int]:
    """Read +1/-1 sides from the canonical bridge position snapshot."""

    sides: set[int] = set()
    for item in str(positions_pipe or "").split("^"):
        parts = item.split(":")
        if len(parts) < 2:
            continue
        try:
            side = int(float(parts[1]))
        except (TypeError, ValueError):
            continue
        if side > 0:
            sides.add(1)
        elif side < 0:
            sides.add(-1)
    return frozenset(sides)


def duplicate_entry_blocked_actions(
    *,
    positions_pipe: object,
    execution_context: Mapping[str, object] | None,
    enabled: bool,
) -> frozenset[int]:
    """Return entry actions that would duplicate an open/pending order.

    The bridge does not yet include pending-order direction in the compact
    execution context.  A pending order therefore blocks all new entries;
    this is conservative but prevents duplicate pending intent.  Open
    positions block only their own side, keeping a permitted hedge/reversal
    visible to the ordinary broker action contract.
    """

    if not bool(enabled):
        return frozenset()
    pending = _finite_mapping(execution_context, "pending_orders")
    if pending > 1e-9:
        return ENTRY_ACTIONS
    sides = parse_open_position_sides(positions_pipe)
    blocked: set[int] = set()
    if 1 in sides:
        blocked.update(BUY_ENTRY_ACTIONS)
    if -1 in sides:
        blocked.update(SELL_ENTRY_ACTIONS)
    return frozenset(blocked)


@dataclass(frozen=True)
class AccountRiskDecision:
    """Non-learning risk budget applied after an entry direction is chosen.

    ``open_risk_ratio`` is the aggregate money-at-initial-SL risk divided by
    account equity. Position count is carried separately and must never become
    an implicit execution quota.
    """

    risk_fraction: float
    output_signal: float
    hard_veto: bool
    hard_veto_reasons: tuple[str, ...]
    scale: float

    def audit(self) -> dict[str, object]:
        return {
            "semantic_risk_fraction": float(self.risk_fraction),
            "semantic_risk_output_signal": float(self.output_signal),
            "semantic_risk_scale": float(self.scale),
            "semantic_risk_hard_veto": bool(self.hard_veto),
            "semantic_risk_hard_veto_reasons": list(self.hard_veto_reasons),
            "semantic_open_risk_measure": "money_at_initial_sl_ratio",
            "semantic_risk_writer": "deterministic_account_safety_only",
        }


def account_risk_decision(
    account_context: Mapping[str, object] | None,
    *,
    base_fraction: float,
    min_fraction: float,
    drawdown_soft_limit: float,
    drawdown_hard_limit: float,
    open_risk_soft_limit: float,
    open_risk_hard_limit: float,
) -> AccountRiskDecision:
    """Map account state to a bounded risk fraction and safety-only veto.

    ``output_signal`` is the desired post-``tanh`` decoder signal in ``[-1,
    1]``. The policy converts it to the corresponding pre-``tanh`` wire value
    at the single ContextFrame transport boundary. No model weight, EMA, or
    lifecycle credit is changed by this function. The
    ``open_risk_ratio`` is true aggregate money-at-initial-SL exposure, as
    documented on :class:`AccountRiskDecision`.
    """

    base = float(np.clip(base_fraction, 0.27, 0.73))
    minimum = float(np.clip(min_fraction, 0.27, base))
    drawdown = max(0.0, _finite_mapping(account_context, "current_equity_drawdown"))
    open_risk = max(0.0, _finite_mapping(account_context, "open_risk_ratio"))
    losing = float(np.clip(_finite_mapping(account_context, "losing_streak_norm"), 0.0, 1.0))
    volatility = max(0.0, _finite_mapping(account_context, "equity_volatility_20"))

    dd_soft = max(1e-9, float(drawdown_soft_limit))
    risk_soft = max(1e-9, float(open_risk_soft_limit))
    dd_scale = float(np.clip(1.0 - max(0.0, drawdown - 0.5 * dd_soft) / dd_soft, 0.25, 1.0))
    exposure_scale = float(np.clip(1.0 - max(0.0, open_risk - 0.5 * risk_soft) / risk_soft, 0.25, 1.0))
    loss_scale = float(np.clip(1.0 - 0.35 * losing, 0.55, 1.0))
    volatility_scale = float(np.clip(1.0 - 2.5 * volatility, 0.60, 1.0))
    scale = float(np.clip(dd_scale * exposure_scale * loss_scale * volatility_scale, 0.10, 1.0))
    fraction = float(np.clip(base * scale, minimum, 0.73))

    reasons: list[str] = []
    if drawdown >= max(drawdown_soft_limit, drawdown_hard_limit):
        reasons.append("drawdown_hard_limit")
    if open_risk >= max(open_risk_soft_limit, open_risk_hard_limit):
        reasons.append("open_risk_hard_limit")
    # The output decoder applies sigmoid to this already-tanh-bounded value.
    signal = float(np.clip(np.log(fraction / max(1e-12, 1.0 - fraction)), -1.0, 1.0))
    return AccountRiskDecision(
        risk_fraction=fraction,
        output_signal=signal,
        hard_veto=bool(reasons),
        hard_veto_reasons=tuple(reasons),
        scale=scale,
    )
