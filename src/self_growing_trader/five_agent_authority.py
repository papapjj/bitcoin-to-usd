"""Hợp đồng phân quyền cố định cho năm tác nhân dùng chung một policy.

Mỗi hành động khác HOLD chỉ có đúng một tác nhân có quyền phát. Hợp đồng này
không đổi kích thước quan sát 137, đầu ra 20 hoặc tham số checkpoint; agent
embedding hiện có là tín hiệu nhận dạng vai trò cho mạng.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Sequence

import numpy as np

from .execution_contract import ACTION_COUNT


FIVE_AGENT_AUTHORITY_VERSION: Final[int] = 1
FIVE_AGENT_COUNT: Final[int] = 5

# Mã wire 0..11 là hợp đồng Python/EA ổn định. Không import ``orders.Action``
# ở lớp thấp này vì orders -> config -> hệ chuyên gia sẽ tạo vòng import.
HOLD_ACTION: Final[int] = 0
MARKET_BUY_ACTION: Final[int] = 1
MARKET_SELL_ACTION: Final[int] = 2
BUY_LIMIT_ACTION: Final[int] = 3
SELL_LIMIT_ACTION: Final[int] = 4
BUY_STOP_ACTION: Final[int] = 5
SELL_STOP_ACTION: Final[int] = 6
CANCEL_ORDERS_ACTION: Final[int] = 7
CLOSE_WORST_ACTION: Final[int] = 8
CLOSE_BEST_ACTION: Final[int] = 9
CLOSE_ALL_ACTION: Final[int] = 10
MANAGE_POSITION_ACTION: Final[int] = 11
LONG_ENTRY_ACTIONS: Final[frozenset[int]] = frozenset({
    MARKET_BUY_ACTION, BUY_LIMIT_ACTION, BUY_STOP_ACTION,
})
SHORT_ENTRY_ACTIONS: Final[frozenset[int]] = frozenset({
    MARKET_SELL_ACTION, SELL_LIMIT_ACTION, SELL_STOP_ACTION,
})


@dataclass(frozen=True)
class FiveAgentAuthority:
    agent_id: int
    role: str
    allowed_actions: frozenset[int]
    responsibility: str

    def allows(self, action: int) -> bool:
        return int(action) in self.allowed_actions


FIVE_AGENT_AUTHORITIES: Final[tuple[FiveAgentAuthority, ...]] = (
    FiveAgentAuthority(
        agent_id=0,
        role="LONG_ENTRY_OWNER",
        allowed_actions=frozenset({
            HOLD_ACTION, MARKET_BUY_ACTION, BUY_LIMIT_ACTION, BUY_STOP_ACTION,
        }),
        responsibility="owned_position",
    ),
    FiveAgentAuthority(
        agent_id=1,
        role="SHORT_ENTRY_OWNER",
        allowed_actions=frozenset({
            HOLD_ACTION, MARKET_SELL_ACTION, SELL_LIMIT_ACTION, SELL_STOP_ACTION,
        }),
        responsibility="owned_position",
    ),
    FiveAgentAuthority(
        agent_id=2,
        role="POSITION_MANAGER",
        allowed_actions=frozenset({
            HOLD_ACTION, CLOSE_WORST_ACTION, CLOSE_BEST_ACTION,
            MANAGE_POSITION_ACTION,
        }),
        responsibility="any_position",
    ),
    FiveAgentAuthority(
        agent_id=3,
        role="PENDING_MANAGER",
        allowed_actions=frozenset({HOLD_ACTION, CANCEL_ORDERS_ACTION}),
        responsibility="any_pending_order",
    ),
    FiveAgentAuthority(
        agent_id=4,
        role="PORTFOLIO_GOVERNOR",
        allowed_actions=frozenset({HOLD_ACTION, CLOSE_ALL_ACTION}),
        responsibility="account_exposure",
    ),
)


def five_agent_authority(agent_id: int) -> FiveAgentAuthority:
    index = int(agent_id)
    if not 0 <= index < FIVE_AGENT_COUNT:
        raise ValueError(f"five agent_id phai nam trong [0,{FIVE_AGENT_COUNT - 1}]")
    return FIVE_AGENT_AUTHORITIES[index]


def five_agent_authority_mask(
    base_mask: np.ndarray | Iterable[bool] | None = None,
) -> np.ndarray:
    """Trả ma trận quyền 5x12, giao với mask trạng thái nếu được cung cấp."""

    authority = np.zeros((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=bool)
    for item in FIVE_AGENT_AUTHORITIES:
        authority[item.agent_id, tuple(sorted(item.allowed_actions))] = True
    if base_mask is None:
        return authority
    base = np.asarray(base_mask, dtype=bool)
    if base.shape == (ACTION_COUNT,):
        base = np.repeat(base.reshape(1, -1), FIVE_AGENT_COUNT, axis=0)
    if base.shape != authority.shape:
        raise ValueError(f"base mask phai co shape ({ACTION_COUNT},) hoac ({FIVE_AGENT_COUNT},{ACTION_COUNT})")
    result = authority & base
    # HOLD là đường an toàn bắt buộc nếu mask trạng thái loại hết quyền còn lại.
    empty = ~result.any(axis=1)
    result[empty, HOLD_ACTION] = True
    return result


def five_agent_responsibility_active(
    agent_id: int,
    *,
    position_count: int,
    pending_order_count: int,
    owns_position: bool,
) -> bool:
    responsibility = five_agent_authority(agent_id).responsibility
    if responsibility == "owned_position":
        return bool(owns_position)
    if responsibility == "any_position":
        return int(position_count) > 0
    if responsibility == "any_pending_order":
        return int(pending_order_count) > 0
    if responsibility == "account_exposure":
        return int(position_count) > 0 or int(pending_order_count) > 0
    raise RuntimeError(f"five responsibility khong duoc ho tro: {responsibility}")


def resolve_opposite_entry_conflict(
    actions: Sequence[int],
    selected_probabilities: Sequence[float],
) -> tuple[int | None, int | None]:
    """Choose one entry owner when both directions fire in the same state.

    Returns ``(winner, blocked)``.  Long/short orders created from one market
    snapshot are mutually exclusive hypotheses; emitting both paid spread on
    both sides and turned the packet into an accidental hedge.  The raw policy
    probability is the only decision-time score used here.  Ties resolve by
    stable agent id so replay is deterministic.
    """

    if len(actions) != FIVE_AGENT_COUNT or len(selected_probabilities) != FIVE_AGENT_COUNT:
        raise ValueError("opposite-entry arbitration requires exactly five actions and scores")
    long_agent = next(
        (index for index, action in enumerate(actions) if int(action) in LONG_ENTRY_ACTIONS),
        None,
    )
    short_agent = next(
        (index for index, action in enumerate(actions) if int(action) in SHORT_ENTRY_ACTIONS),
        None,
    )
    if long_agent is None or short_agent is None:
        return None, None
    long_score = float(selected_probabilities[long_agent])
    short_score = float(selected_probabilities[short_agent])
    if not np.isfinite(long_score):
        long_score = float("-inf")
    if not np.isfinite(short_score):
        short_score = float("-inf")
    winner = long_agent if (long_score, -long_agent) >= (short_score, -short_agent) else short_agent
    blocked = short_agent if winner == long_agent else long_agent
    return int(winner), int(blocked)


def _validate_authority_contract() -> None:
    if tuple(item.agent_id for item in FIVE_AGENT_AUTHORITIES) != tuple(range(FIVE_AGENT_COUNT)):
        raise RuntimeError("five authority phai co du agent_id 0..4 theo thu tu")
    assigned: dict[int, int] = {}
    for item in FIVE_AGENT_AUTHORITIES:
        if HOLD_ACTION not in item.allowed_actions:
            raise RuntimeError(f"five authority {item.role} thieu HOLD")
        for action in item.allowed_actions:
            if not 0 <= int(action) < ACTION_COUNT:
                raise RuntimeError(f"five authority {item.role} co action ngoai hop dong")
            if int(action) == HOLD_ACTION:
                continue
            if int(action) in assigned:
                raise RuntimeError(f"action {action} bi giao trung cho agent {assigned[int(action)]} va {item.agent_id}")
            assigned[int(action)] = item.agent_id
    expected = set(range(1, ACTION_COUNT))
    if set(assigned) != expected:
        raise RuntimeError(f"five authority thieu action: {sorted(expected - set(assigned))}")


_validate_authority_contract()
