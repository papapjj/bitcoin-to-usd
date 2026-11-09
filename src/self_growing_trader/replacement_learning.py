"""Stable, action-scoped evidence helpers for replacement learning.

This module deliberately contains no policy mutation.  It gives the policy a
small deterministic vocabulary for identifying a *specific decision context*
so SL/TP/risk replacements cannot leak from one action family into another.
"""
from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


# Negative lifecycle credit must move a saturated action margin within a few
# confirmed MT5 outcomes. Positive credit keeps the conservative base rate.
NEGATIVE_CONTEXT_CREDIT_MULTIPLIER = 8.0


def stable_identifier(namespace: str, *parts: object, length: int = 24) -> str:
    """Return a restart-stable compact identifier.

    Python's ``hash`` is salted per process; it must never be used for journal,
    checkpoint, or replay identifiers.
    """
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))
    digest = sha256(payload.encode("utf-8")).hexdigest()[: max(8, int(length))]
    return f"{namespace}:{digest}"


def _normalise_text(value: object) -> str:
    return str(value or "").strip().upper()


def action_scope(
    action_index: int,
    *,
    action_details: Mapping[str, Any] | None = None,
    positions_pipe: str | None = "",
) -> dict[str, object]:
    """Build a conservative compatibility scope for parameter replacements.

    Exact action index is mandatory.  Action family, trade side, lifecycle and
    position state add independent guards; a proposal is only reusable when all
    populated guards match the current decision.
    """
    details = dict(action_details or {})
    name = _normalise_text(details.get("action_name", ""))
    idx = int(action_index)
    if not name:
        names = {
            0: "HOLD", 1: "MARKET_BUY", 2: "MARKET_SELL", 3: "BUY_LIMIT",
            4: "SELL_LIMIT", 5: "BUY_STOP", 6: "SELL_STOP", 7: "CANCEL_ORDERS",
            8: "CLOSE_WORST", 9: "CLOSE_BEST", 10: "CLOSE_ALL", 11: "MANAGE_POSITION",
        }
        name = names.get(idx, f"ACTION_{idx}")

    if idx == 0 or name == "HOLD":
        family, lifecycle, side = "hold", "hold", "flat"
    elif idx in {1, 3, 5} or "BUY" in name:
        family, lifecycle, side = "entry", "entry", "long"
    elif idx in {2, 4, 6} or "SELL" in name:
        family, lifecycle, side = "entry", "entry", "short"
    elif idx in {8, 9, 10} or name.startswith("CLOSE"):
        family, lifecycle, side = "exit", "exit", "any"
    elif idx in {7, 11} or "MANAGE" in name or "CANCEL" in name:
        family, lifecycle, side = "management", "manage", "any"
    else:
        family, lifecycle, side = "other", "other", "any"

    raw_positions = str(positions_pipe or "").strip()
    position_state = "open" if raw_positions else "flat"
    target_slot = int(details.get("target_slot", 0) or 0)
    management_mode = _normalise_text(details.get("management_mode", ""))
    return {
        "action_index": idx,
        "action_name": name,
        "action_family": family,
        "side": side,
        "lifecycle": lifecycle,
        "position_state": position_state,
        "target_slot": target_slot if family in {"management", "exit"} else -1,
        "management_mode": management_mode if family == "management" else "",
    }


def scope_matches(recorded: Mapping[str, Any] | None, current: Mapping[str, Any] | None) -> bool:
    """Return whether a recorded proposal may influence this decision.

    Scope matching is intentionally exact for the discrete action and strict
    for every non-generic dimension.  This prevents a BUY SL/TP experiment from
    affecting a SELL entry, an exit, or a position-management action.
    """
    a = dict(recorded or {})
    b = dict(current or {})
    if not a or not b:
        return False
    if int(a.get("action_index", -1)) != int(b.get("action_index", -2)):
        return False
    for key in ("action_family", "lifecycle", "position_state"):
        left, right = str(a.get(key, "") or ""), str(b.get(key, "") or "")
        if left and right and left != right:
            return False
    for key in ("side", "management_mode"):
        left, right = str(a.get(key, "") or ""), str(b.get(key, "") or "")
        if left and right and left != "any" and right != "any" and left != right:
            return False
    left_slot, right_slot = int(a.get("target_slot", -1) or -1), int(b.get("target_slot", -1) or -1)
    if left_slot >= 0 and right_slot >= 0 and left_slot != right_slot:
        return False
    return True


def proposal_confidence(meta: Mapping[str, Any] | None) -> float:
    """Posterior-like confidence from paired replacement outcomes."""
    values = dict(meta or {})
    positive = max(0.0, float(values.get("evidence_for", 0.0) or 0.0))
    negative = max(0.0, float(values.get("evidence_against", 0.0) or 0.0))
    attempts = max(0, int(values.get("validation_attempts", 0) or 0))
    support = positive + negative
    direction = positive / max(1e-9, support) if support > 0.0 else 0.0
    coverage = 1.0 - pow(2.718281828, -float(attempts) / 3.0)
    return max(0.0, min(1.0, direction * coverage))
