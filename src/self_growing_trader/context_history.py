"""Causal context-history inputs for context-owner Direct-RNN policies.

The MT5 EA wire remains the immutable 137-feature schema. A candidate may
append two internal scalar tokens after that wire vector so a policy can
distinguish a fresh trend from one reached through a recent opposite regime.
Tokens are created by the Python facade only after the decision, so they never
contain the current/future M15 label or future PnL.

One scalar is assigned to each categorical token because this contract has
exactly two additional inputs: t-1 and t-2. The values are stable category
identifiers, not numerical trend scores; raw labels remain in decision audit.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from .protocol_contract import INPUT_COUNT


CONTEXT_HISTORY_INPUT_COUNT = 2
POLICY_CONTEXT_HISTORY_INPUT_SIZE = INPUT_COUNT + CONTEXT_HISTORY_INPUT_COUNT

# Zero is reserved for causal warm-up/unknown. Do not use Python's randomized
# hash(): a saved policy must see identical values after restart and replay.
CONTEXT_TOKEN_CODE: dict[str, float] = {
    "CTX_GIAM_MANH": -0.91,
    "CTX_GIAM_YEU": 0.37,
    "CTX_RANGING": -0.28,
    "CTX_VOLATILE": 0.82,
    "CTX_VOLATILE_SPIKE": -0.63,
    "CTX_VOLATILE_EXPANSION": 0.14,
    "CTX_VOLATILE_CHOP": 0.69,
    "CTX_TANG_YEU": -0.47,
    "CTX_TANG_MANH": 0.96,
}


def canonical_context_token(context: object) -> str:
    """Return a known market-context id, or ``""`` for any other label."""

    token = str(context or "").strip().upper()
    return token if token in CONTEXT_TOKEN_CODE else ""


def encode_context_history(
    contexts: Iterable[object],
    *,
    count: int = CONTEXT_HISTORY_INPUT_COUNT,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Encode oldest-to-newest stored contexts into newest-first t-1/t-2."""

    width = max(0, min(CONTEXT_HISTORY_INPUT_COUNT, int(count)))
    values = np.zeros(width, dtype=float)
    recent = [canonical_context_token(value) for value in contexts]
    recent = [value for value in recent if value][-width:][::-1]
    for index, token in enumerate(recent):
        values[index] = float(CONTEXT_TOKEN_CODE[token])
    labels = tuple(recent + [""] * max(0, width - len(recent)))
    return values, labels


def append_context_history_inputs(
    observation: object,
    contexts: Iterable[object],
    *,
    count: int = CONTEXT_HISTORY_INPUT_COUNT,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Append causal history without changing the EA/CSV wire contract."""

    width = max(0, min(CONTEXT_HISTORY_INPUT_COUNT, int(count)))
    try:
        raw = np.asarray(observation, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        raw = np.zeros(0, dtype=float)
    wire = np.zeros(INPUT_COUNT, dtype=float)
    wire[: min(wire.size, raw.size)] = raw[: wire.size]
    if width <= 0:
        return wire, ()
    encoded, labels = encode_context_history(contexts, count=width)
    return np.concatenate((wire, encoded)), labels
