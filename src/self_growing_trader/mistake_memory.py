from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .orders import Action


MISTAKE_EPISODE_SCHEMA_VERSION = 1
MISTAKE_EPISODE_STORE_SCHEMA_VERSION = 1

ENTRY_ACTION_NAMES = {
    "MARKET_BUY",
    "MARKET_SELL",
    "BUY_LIMIT",
    "SELL_LIMIT",
    "BUY_STOP",
    "SELL_STOP",
}


def _now() -> float:
    return float(time.time())


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.reshape(-1).tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _vector(value: Any, *, limit: int | None = None) -> list[float]:
    if value is None:
        return []
    arr = np.asarray(value, dtype=float).reshape(-1)
    if limit is not None:
        arr = arr[: max(0, int(limit))]
    if arr.size == 0:
        return []
    return [float(item) for item in arr.tolist()]


def _stable_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(encoded.encode("utf-8")).hexdigest()[:20]


def _action_name_from_index(action_index: int) -> str:
    try:
        return Action(int(action_index)).name
    except ValueError:
        return f"ACTION_{int(action_index)}"


def _action_index_from_trace(trace: Any) -> int:
    details = dict(getattr(trace, "action_details", {}) or {})
    for key in ("action_index", "action", "cmd"):
        if key in details:
            return _int(details.get(key), 0)
    execution_details = dict(getattr(trace, "execution_details", {}) or {})
    for key in ("bridge_action", "requested_action", "action"):
        if key in execution_details:
            return _int(execution_details.get(key), 0)
    outputs = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
    if outputs.size:
        return int(np.argmax(outputs[: min(outputs.size, len(Action))]))
    return 0


def _action_name_from_trace(trace: Any, action_index: int) -> str:
    details = dict(getattr(trace, "action_details", {}) or {})
    name = str(details.get("action_name", "") or "").strip()
    return name if name else _action_name_from_index(action_index)


def parse_positions_pipe(positions_pipe: str | None) -> list[dict[str, Any]]:
    if not positions_pipe:
        return []
    snapshots: list[dict[str, Any]] = []
    for slot, item in enumerate(str(positions_pipe).split("^")):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) < 8:
            continue
        try:
            snapshots.append({
                "slot": slot,
                "ticket": str(parts[0]),
                "side": _int(parts[1]),
                "volume": _float(parts[2]),
                "entry": _float(parts[3]),
                "sl": _float(parts[4]),
                "tp": _float(parts[5]),
                "unrealized_r": _float(parts[6]),
                "age_bars": _float(parts[7]),
                "mfe_r": _float(parts[8], _float(parts[6])) if len(parts) > 8 else _float(parts[6]),
                "mae_r": _float(parts[9], _float(parts[6])) if len(parts) > 9 else _float(parts[6]),
                "dist_tp_atr": _float(parts[10]) if len(parts) > 10 else 0.0,
                "dist_sl_atr": _float(parts[11]) if len(parts) > 11 else 0.0,
                "min_dist_tp_atr": _float(parts[12]) if len(parts) > 12 else 0.0,
                "min_dist_sl_atr": _float(parts[13]) if len(parts) > 13 else 0.0,
                "bars_since_mfe": _float(parts[14]) if len(parts) > 14 else 0.0,
                "near_tp": bool(_float(parts[15])) if len(parts) > 15 else False,
                "giveback_r": _float(parts[16]) if len(parts) > 16 else 0.0,
            })
        except (TypeError, ValueError):
            continue
    return snapshots


def _position_key_from_positions(positions: list[dict[str, Any]]) -> list[float]:
    if not positions:
        return [0.0] * 6
    unrealized = np.asarray([_float(item.get("unrealized_r")) for item in positions], dtype=float)
    ages = np.asarray([_float(item.get("age_bars")) for item in positions], dtype=float)
    sides = np.asarray([_float(item.get("side")) for item in positions], dtype=float)
    near_tp = np.asarray([1.0 if bool(item.get("near_tp", False)) else 0.0 for item in positions], dtype=float)
    giveback = np.asarray([_float(item.get("giveback_r")) for item in positions], dtype=float)
    return [
        float(len(positions) / 5.0),
        float(np.mean(unrealized) / 3.0),
        float(np.min(unrealized) / 3.0),
        float(np.mean(np.clip(ages / 60.0, 0.0, 2.0))),
        float(np.mean(sides)),
        float(np.mean(near_tp) + 0.25 * np.mean(np.clip(giveback, 0.0, 4.0))),
    ]


def _account_key_from_context(context: dict[str, Any]) -> list[float]:
    values = [
        _float(context.get("equity_to_balance")),
        _float(context.get("floating_pnl_r")),
        0.5 * (_float(context.get("current_equity_drawdown")) + _float(context.get("balance_drawdown"))),
        _float(context.get("open_risk_ratio")),
        _float(context.get("equity_return_1_bar")),
        _float(context.get("equity_return_5_bar")),
        _float(context.get("equity_volatility_20")),
        (
            0.5 * (_float(context.get("losing_streak_norm")) + _float(context.get("time_under_water_norm")))
            + 0.25 * _float(context.get("position_count_norm"))
            + 0.25 * _float(context.get("day_trade_pressure"))
        ),
    ]
    return [float(np.clip(item, -2.0, 2.0)) for item in values]


@dataclass
class MistakeEpisode:
    schema_version: int = MISTAKE_EPISODE_SCHEMA_VERSION
    episode_id: str = ""
    trace_id: str = ""
    source: str = "journal"
    raw_obs: list[float] = field(default_factory=list)
    market_key: list[float] = field(default_factory=list)
    account_key: list[float] = field(default_factory=list)
    position_key: list[float] = field(default_factory=list)
    action_index: int = 0
    action_name: str = "HOLD"
    action_details: dict[str, Any] = field(default_factory=dict)
    selected_output: list[float] = field(default_factory=list)
    confidence: float = 0.0
    risk_pct: float = 0.0
    sl_atr: float = 0.0
    tp_atr: float = 0.0
    trailing_stop_atr: float = 0.0
    partial_close_pct: float = 0.0
    management_mode: str = ""
    target_slot: int = 0
    module_action_traces: list[dict[str, Any]] = field(default_factory=list)
    active_neuron_ids: list[str] = field(default_factory=list)
    active_edge_ids: list[str] = field(default_factory=list)
    graph_edge_gates: dict[str, float] = field(default_factory=dict)
    graph_region_activity: dict[str, float] = field(default_factory=dict)
    positions_before: list[dict[str, Any]] = field(default_factory=list)
    positions_after: list[dict[str, Any]] = field(default_factory=list)
    execution_details: dict[str, Any] = field(default_factory=dict)
    mfe_r: float = 0.0
    mae_r: float = 0.0
    giveback_r: float = 0.0
    bars_since_mfe: float = 0.0
    close_reason: str = ""
    reward: float = 0.0
    td_delta: float = 0.0
    prediction_error: float = 0.0
    realized_r: float = 0.0
    net_profit_delta: float = 0.0
    drawdown_delta: float = 0.0
    spread_points: float = 0.0
    mistake_tags: list[str] = field(default_factory=list)
    tag_evidence: dict[str, dict[str, Any]] = field(default_factory=dict)
    suspected_causes: list[str] = field(default_factory=list)
    proposed_repairs: list[str] = field(default_factory=list)
    severity: float = 0.0
    confidence_score: float = 0.0
    support_count: int = 1
    contradiction_count: int = 0
    created_at: float = field(default_factory=_now)
    last_seen_at: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.schema_version = int(self.schema_version or MISTAKE_EPISODE_SCHEMA_VERSION)
        self.action_index = int(self.action_index)
        self.action_name = str(self.action_name or _action_name_from_index(self.action_index))
        self.mistake_tags = sorted(dict.fromkeys(str(tag) for tag in (self.mistake_tags or []) if str(tag)))
        self.active_neuron_ids = list(dict.fromkeys(str(item) for item in (self.active_neuron_ids or []) if str(item)))
        self.active_edge_ids = list(dict.fromkeys(str(item) for item in (self.active_edge_ids or []) if str(item)))
        self.support_count = max(1, int(self.support_count or 1))
        self.contradiction_count = max(0, int(self.contradiction_count or 0))
        if not self.episode_id:
            self.episode_id = self.build_episode_id()

    def build_episode_id(self) -> str:
        return "mistake-" + _stable_hash({
            "trace_id": self.trace_id,
            "source": self.source,
            "action_index": self.action_index,
            "action_name": self.action_name,
            "tags": self.mistake_tags,
            "market_key": [round(float(item), 4) for item in self.market_key[:12]],
            "account_key": [round(float(item), 4) for item in self.account_key[:8]],
            "position_key": [round(float(item), 4) for item in self.position_key[:6]],
        })

    def pattern_key(self) -> str:
        return _stable_hash({
            "action_index": self.action_index,
            "tags": self.mistake_tags,
            "account": [round(float(item), 2) for item in self.account_key[:8]],
            "position": [round(float(item), 2) for item in self.position_key[:6]],
            "market": [round(float(item), 1) for item in self.market_key[:8]],
        })

    def memory_key(self) -> np.ndarray:
        return np.asarray(list(self.market_key) + list(self.account_key) + list(self.position_key), dtype=float)

    def merge_support(self, other: "MistakeEpisode") -> None:
        total = max(1, int(self.support_count) + int(other.support_count))
        lr = float(other.support_count) / float(total)
        self.severity = float(max(self.severity, other.severity, (1.0 - lr) * self.severity + lr * other.severity))
        self.confidence_score = float((1.0 - lr) * self.confidence_score + lr * other.confidence_score)
        self.reward = float((1.0 - lr) * self.reward + lr * other.reward)
        self.td_delta = float((1.0 - lr) * self.td_delta + lr * other.td_delta)
        self.prediction_error = float((1.0 - lr) * self.prediction_error + lr * other.prediction_error)
        self.support_count = total
        self.contradiction_count += int(other.contradiction_count)
        self.last_seen_at = max(float(self.last_seen_at), float(other.last_seen_at))
        self.mistake_tags = sorted(set(self.mistake_tags) | set(other.mistake_tags))
        self.suspected_causes = sorted(set(self.suspected_causes) | set(other.suspected_causes))
        self.proposed_repairs = sorted(set(self.proposed_repairs) | set(other.proposed_repairs))
        merged_evidence = dict(self.tag_evidence)
        for tag, evidence in other.tag_evidence.items():
            merged_evidence.setdefault(str(tag), _jsonable(evidence))
        self.tag_evidence = merged_evidence

    def to_dict(self) -> dict[str, Any]:
        return _jsonable({
            "schema_version": int(self.schema_version),
            "episode_id": self.episode_id,
            "trace_id": self.trace_id,
            "source": self.source,
            "raw_obs": self.raw_obs,
            "market_key": self.market_key,
            "account_key": self.account_key,
            "position_key": self.position_key,
            "action_index": self.action_index,
            "action_name": self.action_name,
            "action_details": self.action_details,
            "selected_output": self.selected_output,
            "confidence": self.confidence,
            "risk_pct": self.risk_pct,
            "sl_atr": self.sl_atr,
            "tp_atr": self.tp_atr,
            "trailing_stop_atr": self.trailing_stop_atr,
            "partial_close_pct": self.partial_close_pct,
            "management_mode": self.management_mode,
            "target_slot": self.target_slot,
            "module_action_traces": self.module_action_traces,
            "active_neuron_ids": self.active_neuron_ids,
            "active_edge_ids": self.active_edge_ids,
            "graph_edge_gates": self.graph_edge_gates,
            "graph_region_activity": self.graph_region_activity,
            "positions_before": self.positions_before,
            "positions_after": self.positions_after,
            "execution_details": self.execution_details,
            "mfe_r": self.mfe_r,
            "mae_r": self.mae_r,
            "giveback_r": self.giveback_r,
            "bars_since_mfe": self.bars_since_mfe,
            "close_reason": self.close_reason,
            "reward": self.reward,
            "td_delta": self.td_delta,
            "prediction_error": self.prediction_error,
            "realized_r": self.realized_r,
            "net_profit_delta": self.net_profit_delta,
            "drawdown_delta": self.drawdown_delta,
            "spread_points": self.spread_points,
            "mistake_tags": self.mistake_tags,
            "tag_evidence": self.tag_evidence,
            "suspected_causes": self.suspected_causes,
            "proposed_repairs": self.proposed_repairs,
            "severity": self.severity,
            "confidence_score": self.confidence_score,
            "support_count": self.support_count,
            "contradiction_count": self.contradiction_count,
            "created_at": self.created_at,
            "last_seen_at": self.last_seen_at,
        })

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MistakeEpisode":
        payload = dict(data or {})
        version = int(payload.get("schema_version", 0) or 0)
        if version <= 0:
            payload["schema_version"] = MISTAKE_EPISODE_SCHEMA_VERSION
            payload.setdefault("source", payload.get("source_journal", "journal"))
            payload.setdefault("mistake_tags", payload.get("tags", []))
            payload.setdefault("reward", payload.get("net_r", 0.0))
            payload.setdefault("realized_r", payload.get("net_r", 0.0))
        allowed = set(cls.__dataclass_fields__)
        cleaned = {key: payload.get(key) for key in allowed if key in payload}
        episode = cls(**cleaned)
        if not episode.episode_id:
            episode.episode_id = episode.build_episode_id()
        return episode

    @classmethod
    def from_json(cls, line: str) -> "MistakeEpisode":
        return cls.from_dict(json.loads(line))


class MistakeTagger:
    """Evidence-only heuristic tagger for losing actions and rejected trials."""

    def tag(
        self,
        *,
        trace: Any | None = None,
        reward: float = 0.0,
        cost: float = 0.0,
        signal: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        reject_reasons: list[str] | tuple[str, ...] = (),
    ) -> tuple[list[str], dict[str, dict[str, Any]], list[str], list[str]]:
        signal = dict(signal or {})
        metrics = dict(metrics or {})
        execution_details = dict(getattr(trace, "execution_details", {}) or {}) if trace is not None else {}
        action_details = dict(getattr(trace, "action_details", {}) or {}) if trace is not None else {}
        account_context = dict(getattr(trace, "account_context", {}) or {}) if trace is not None else {}
        starvation = dict(getattr(trace, "experience_starvation", {}) or {}) if trace is not None else {}
        lifecycle_tags = set(str(item) for item in getattr(trace, "lifecycle_tags", ()) or ())
        lifecycle_tags.update(str(item) for item in execution_details.get("lifecycle_tags", []) or [])
        reason_blob = " ".join(str(item) for item in reject_reasons or ()).lower()
        tags: dict[str, dict[str, Any]] = {}

        def add(tag: str, **evidence: Any) -> None:
            tags.setdefault(tag, _jsonable(evidence))

        td_delta = _float(signal.get("td_delta"))
        prediction_error = _float(signal.get("prediction_error"))
        utility = td_delta + float(reward) - float(cost) - 0.1 * prediction_error
        action_index = _action_index_from_trace(trace) if trace is not None else _int(metrics.get("action_index"))
        action_name = _action_name_from_trace(trace, action_index) if trace is not None else _action_name_from_index(action_index)
        is_entry = action_name in ENTRY_ACTION_NAMES or action_index in {int(Action.MARKET_BUY), int(Action.MARKET_SELL), int(Action.BUY_LIMIT), int(Action.SELL_LIMIT), int(Action.BUY_STOP), int(Action.SELL_STOP)}

        if is_entry and (float(reward) < -0.05 or td_delta < -0.05 or _float(metrics.get("realized_r")) < -0.05):
            add("wrong_direction", reward=reward, td_delta=td_delta, realized_r=_float(metrics.get("realized_r")))
        drawdown = max(_float(account_context.get("current_equity_drawdown")), _float(account_context.get("balance_drawdown")), _float(metrics.get("drawdown_pct")))
        open_risk = _float(account_context.get("open_risk_ratio"))
        floating_loss = max(0.0, -_float(account_context.get("floating_pnl_r")))
        if is_entry and (drawdown >= 0.03 or open_risk >= 0.02 or floating_loss >= 0.5):
            add("stress_entry", drawdown=drawdown, open_risk_ratio=open_risk, floating_loss_r=floating_loss)
        spread_points = self._spread_points(execution_details, action_details, metrics)
        if is_entry and spread_points >= 20.0:
            add("high_spread_entry", spread_points=spread_points)
        day_pressure = max(_float(account_context.get("day_trade_pressure")), _float(starvation.get("day_trade_pressure")), _float(metrics.get("day_trade_pressure")))
        if day_pressure >= 0.85 or _int(metrics.get("day_trades")) >= _int(metrics.get("max_daily_trades"), 999999):
            add("overtrading", day_trade_pressure=day_pressure, day_trades=_int(metrics.get("day_trades")), max_daily_trades=_int(metrics.get("max_daily_trades")))
        risk_pct = self._risk_pct(action_details, metrics)
        if risk_pct >= max(2.0, _float(metrics.get("max_risk_pct"), 2.0)):
            add("bad_risk_pct", risk_pct=risk_pct, max_risk_pct=_float(metrics.get("max_risk_pct"), 2.0))

        positions_before = parse_positions_pipe(str(getattr(trace, "positions_pipe", "") or "")) if trace is not None else []
        selected_position = self._selected_position(positions_before, action_details)
        context_audit = dict(getattr(trace, "context_audit", {}) or {}) if trace is not None else {}
        if selected_position:
            mfe_r = _float(selected_position.get("mfe_r"), _float(selected_position.get("unrealized_r")))
            mae_r = _float(selected_position.get("mae_r"), _float(selected_position.get("unrealized_r")))
            giveback_r = _float(selected_position.get("giveback_r"))
            bars_since_mfe = _float(selected_position.get("bars_since_mfe"))
            near_tp = bool(selected_position.get("near_tp", False)) or _float(selected_position.get("min_dist_tp_atr"), 999.0) <= 0.25
            if "near_tp_giveback" in lifecycle_tags or (near_tp and giveback_r >= 0.5):
                add("near_tp_giveback", near_tp=near_tp, giveback_r=giveback_r, mfe_r=mfe_r)
            if "should_have_partial_or_trail" in lifecycle_tags or (mfe_r >= 1.0 and giveback_r >= 0.75):
                add("missed_trailing", mfe_r=mfe_r, giveback_r=giveback_r, bars_since_mfe=bars_since_mfe)
            if mfe_r >= 1.0 and giveback_r >= 0.5 and _float(action_details.get("partial_close_pct")) <= 0.0:
                add("missed_partial", mfe_r=mfe_r, giveback_r=giveback_r, partial_close_pct=_float(action_details.get("partial_close_pct")))
            if mae_r <= -0.15 and _float(reward) < 0.0:
                add("held_loser_too_long", mae_r=mae_r, reward=reward)
            context_task = str(context_audit.get("context_task", "") or "")
            context_stable = bool(context_audit.get("context_on_dinh", context_audit.get("on_dinh", False)))
            expected_side = 1 if context_task == "TANG_MANH" else -1 if context_task == "GIAM_MANH" else 0
            aligned = expected_side != 0 and _int(selected_position.get("side")) == expected_side
            exit_ready = (
                _float(selected_position.get("unrealized_r")) <= -0.75
                or near_tp
                or (giveback_r >= 0.35 and bars_since_mfe >= 2.0)
            )
            if action_name.startswith("CLOSE") and context_stable and aligned and not exit_ready:
                add(
                    "premature_strong_trend_exit",
                    context_task=context_task,
                    unrealized_r=_float(selected_position.get("unrealized_r")),
                    mfe_r=mfe_r,
                    giveback_r=giveback_r,
                    bars_since_mfe=bars_since_mfe,
                )
        close_reason = str(execution_details.get("close_reason", metrics.get("close_reason", "")) or "").upper()
        if close_reason in {"SL", "STOP_LOSS", "STOP_OUT"} and _float(metrics.get("duration_bars")) <= 3.0 and _float(metrics.get("duration_bars")) > 0.0:
            add("sl_too_tight", close_reason=close_reason, duration_bars=_float(metrics.get("duration_bars")))
        if "closed_after_near_tp_bad" in lifecycle_tags:
            add("closed_winner_too_early", lifecycle_tags=sorted(lifecycle_tags))
        if bool(execution_details.get("rejected", False)) or "reject" in reason_blob or "execution" in reason_blob or "reconciliation" in reason_blob:
            add("execution_reject", execution_details=execution_details, reject_reasons=list(reject_reasons or ()))
        if "no_trade" in reason_blob or "insufficient" in reason_blob or (_float(metrics.get("total_trades", metrics.get("trades", 0.0))) <= 0.0 and reject_reasons):
            add("no_trade_starvation", total_trades=_float(metrics.get("total_trades", metrics.get("trades", 0.0))), reject_reasons=list(reject_reasons or ()))
        if "reconciliation" in reason_blob or "ledger" in reason_blob or bool(metrics.get("ledger_integrity_bad", False)):
            add("ledger_integrity_bad", reject_reasons=list(reject_reasons or ()), ledger_integrity_bad=bool(metrics.get("ledger_integrity_bad", False)))
        if action_name.startswith("CLOSE") and float(reward) < -0.05 and _float(metrics.get("duration_bars")) <= 1.0 and _float(metrics.get("duration_bars")) > 0.0:
            add("immediate_exit_loop", action_name=action_name, reward=reward, duration_bars=_float(metrics.get("duration_bars")))
        if is_entry and _float(action_details.get("signal_age_bars")) >= 5.0:
            add("late_entry", signal_age_bars=_float(action_details.get("signal_age_bars")))
        if _int(action_details.get("target_slot"), -1) >= 0 and selected_position and _float(selected_position.get("unrealized_r")) > 0.5 and action_name in {"CLOSE_WORST", "MANAGE_POSITION"}:
            add("bad_target_slot", target_slot=_int(action_details.get("target_slot")), target_unrealized_r=_float(selected_position.get("unrealized_r")))

        tag_names = sorted(tags)
        causes = self._causes_for(tag_names)
        repairs = self._repairs_for(tag_names)
        if utility < -0.05 and not tag_names:
            add("loss_unclassified", utility=utility, reward=reward, td_delta=td_delta, cost=cost)
            tag_names = sorted(tags)
            causes = self._causes_for(tag_names)
            repairs = self._repairs_for(tag_names)
        return tag_names, tags, causes, repairs

    @staticmethod
    def _spread_points(*sources: dict[str, Any]) -> float:
        for source in sources:
            for key in ("spread_points", "spread", "pending_spread_points"):
                if key in source:
                    return _float(source.get(key))
        return 0.0

    @staticmethod
    def _risk_pct(action_details: dict[str, Any], metrics: dict[str, Any]) -> float:
        for source in (action_details, metrics):
            for key in ("risk_pct", "risk_percent", "risk"):
                if key in source:
                    return _float(source.get(key))
        return 0.0

    @staticmethod
    def _selected_position(positions: list[dict[str, Any]], action_details: dict[str, Any]) -> dict[str, Any]:
        if not positions:
            return {}
        slot = _int(action_details.get("target_slot"), 0)
        for item in positions:
            if _int(item.get("slot"), -1) == slot:
                return item
        return positions[0]

    @staticmethod
    def _causes_for(tags: list[str]) -> list[str]:
        mapping = {
            "wrong_direction": "entry_direction_caused_loss",
            "late_entry": "signal_decay_before_entry",
            "stress_entry": "account_stress_before_entry",
            "high_spread_entry": "expensive_spread_at_entry",
            "overtrading": "daily_cadence_pressure_high",
            "bad_risk_pct": "risk_sizing_too_large",
            "near_tp_giveback": "profit_protection_failed",
            "missed_partial": "partial_close_not_used",
            "missed_trailing": "trailing_management_not_used",
            "held_loser_too_long": "loss_management_too_slow",
            "premature_strong_trend_exit": "strong_trend_continuation_ignored",
            "execution_reject": "mt5_execution_not_confirmed",
            "no_trade_starvation": "candidate_failed_to_trade",
            "ledger_integrity_bad": "execution_ledger_incomplete",
        }
        return sorted({mapping[tag] for tag in tags if tag in mapping})

    @staticmethod
    def _repairs_for(tags: list[str]) -> list[str]:
        repairs: set[str] = set()
        if {"wrong_direction", "late_entry", "stress_entry", "high_spread_entry"} & set(tags):
            repairs.update({"tighten_entry_gate", "reduce_action_bias"})
        if "bad_risk_pct" in tags:
            repairs.add("reduce_risk_pct")
        if {"near_tp_giveback", "missed_trailing"} & set(tags):
            repairs.add("increase_trailing_gate")
        if "missed_partial" in tags:
            repairs.add("increase_partial_close_ratio")
        if "held_loser_too_long" in tags:
            repairs.add("increase_close_or_manage_bias")
        if "premature_strong_trend_exit" in tags:
            repairs.add("replace_close_with_trailing_management")
        if "execution_reject" in tags:
            repairs.add("lower_graph_edge_gate")
        if "no_trade_starvation" in tags:
            repairs.add("increase_hold_bias")
        return sorted(repairs)


def episode_from_trace(
    trace: Any,
    next_obs: Any | None,
    reward: float,
    cost: float,
    signal: dict[str, Any] | None,
    *,
    source: str = "journal",
    metrics: dict[str, Any] | None = None,
    reject_reasons: list[str] | tuple[str, ...] = (),
) -> MistakeEpisode:
    signal = dict(signal or {})
    metrics = dict(metrics or {})
    tagger = MistakeTagger()
    tags, evidence, causes, repairs = tagger.tag(
        trace=trace,
        reward=reward,
        cost=cost,
        signal=signal,
        metrics=metrics,
        reject_reasons=reject_reasons,
    )
    action_details = dict(getattr(trace, "action_details", {}) or {})
    account_context = dict(getattr(trace, "account_context", {}) or {})
    positions_before = parse_positions_pipe(str(getattr(trace, "positions_pipe", "") or ""))
    positions_after = parse_positions_pipe(str(getattr(trace, "positions_pipe_after", "") or ""))
    selected_position = MistakeTagger._selected_position(positions_before, action_details)
    action_index = _action_index_from_trace(trace)
    action_name = _action_name_from_trace(trace, action_index)
    td_delta = _float(signal.get("td_delta"))
    prediction_error = _float(signal.get("prediction_error"))
    utility = td_delta + float(reward) - float(cost) - 0.1 * prediction_error
    severity = float(np.clip(
        max(0.0, -utility)
        + 0.25 * len(tags)
        + 0.25 * max(0.0, _float(selected_position.get("giveback_r")))
        + 0.10 * max(0.0, prediction_error),
        0.05,
        5.0,
    ))
    confidence_score = float(np.clip(
        0.25
        + 0.08 * len(tags)
        + 0.10 * bool(getattr(trace, "active_neuron_ids", ()) or getattr(trace, "active_edge_ids", ()))
        + 0.10 * bool(getattr(trace, "execution_details", {}) or {}),
        0.05,
        1.0,
    ))
    module_action_traces = []
    for item in tuple(getattr(trace, "module_action_traces", ()) or ()):
        module_action_traces.append({
            "module_name": str(getattr(item, "module_name", "")),
            "gate": _float(getattr(item, "gate", 0.0)),
            "hidden": _vector(getattr(item, "hidden", [])),
            "raw_output": _vector(getattr(item, "raw_output", [])),
        })
    execution_details = dict(getattr(trace, "execution_details", {}) or {})
    close_reason = str(execution_details.get("close_reason", metrics.get("close_reason", "")) or "")
    return MistakeEpisode(
        trace_id=str(getattr(trace, "trace_id", "") or ""),
        source=str(source or "journal"),
        raw_obs=_vector(getattr(trace, "raw_obs", [])),
        market_key=_vector(getattr(trace, "encoded_state", [])),
        account_key=_account_key_from_context(account_context),
        position_key=_position_key_from_positions(positions_before),
        action_index=action_index,
        action_name=action_name,
        action_details=_jsonable(action_details),
        selected_output=_vector(getattr(trace, "selected_action", [])),
        confidence=_float(getattr(trace, "confidence", 0.0)),
        risk_pct=MistakeTagger._risk_pct(action_details, metrics),
        sl_atr=_float(action_details.get("sl_atr")),
        tp_atr=_float(action_details.get("tp_atr")),
        trailing_stop_atr=_float(action_details.get("trailing_stop_atr")),
        partial_close_pct=_float(action_details.get("partial_close_pct")),
        management_mode=str(action_details.get("management_mode", "") or ""),
        target_slot=_int(action_details.get("target_slot")),
        module_action_traces=module_action_traces,
        active_neuron_ids=[str(item) for item in tuple(getattr(trace, "active_neuron_ids", ()) or ())],
        active_edge_ids=[str(item) for item in tuple(getattr(trace, "active_edge_ids", ()) or ())],
        graph_edge_gates={str(key): _float(value) for key, value in dict(getattr(trace, "graph_edge_gates", {}) or {}).items()},
        graph_region_activity={str(key): _float(value) for key, value in dict(getattr(trace, "graph_region_activity", {}) or {}).items()},
        positions_before=_jsonable(positions_before),
        positions_after=_jsonable(positions_after),
        execution_details=_jsonable(execution_details),
        mfe_r=_float(selected_position.get("mfe_r")),
        mae_r=_float(selected_position.get("mae_r")),
        giveback_r=_float(selected_position.get("giveback_r")),
        bars_since_mfe=_float(selected_position.get("bars_since_mfe")),
        close_reason=close_reason,
        reward=float(reward),
        td_delta=td_delta,
        prediction_error=prediction_error,
        realized_r=_float(metrics.get("realized_r", reward)),
        net_profit_delta=_float(metrics.get("net_profit_delta", metrics.get("net_profit", 0.0))),
        drawdown_delta=_float(metrics.get("drawdown_delta", metrics.get("drawdown_pct", 0.0))),
        spread_points=MistakeTagger._spread_points(execution_details, action_details, metrics),
        mistake_tags=tags,
        tag_evidence=evidence,
        suspected_causes=causes,
        proposed_repairs=repairs,
        severity=severity,
        confidence_score=confidence_score,
    )


def episode_from_rejection(
    *,
    reject_reasons: list[str] | tuple[str, ...],
    metrics: dict[str, Any] | None,
    fitness_delta: float,
    source: str = "rejection",
) -> MistakeEpisode:
    metrics = dict(metrics or {})
    tagger = MistakeTagger()
    signal = {
        "td_delta": -abs(_float(metrics.get("expectancy", 0.0))),
        "prediction_error": 0.0,
    }
    tags, evidence, causes, repairs = tagger.tag(
        trace=None,
        reward=_float(metrics.get("net_profit", 0.0)),
        cost=max(0.0, _float(metrics.get("balance_drawdown_pct", 0.0))),
        signal=signal,
        metrics=metrics,
        reject_reasons=reject_reasons,
    )
    severity = float(np.clip(
        0.75 + 0.2 * len(tags) + 0.015 * max(0.0, -float(fitness_delta)),
        0.25,
        5.0,
    ))
    return MistakeEpisode(
        trace_id="",
        source=source,
        action_index=_int(metrics.get("action_index"), 0),
        action_name=_action_name_from_index(_int(metrics.get("action_index"), 0)),
        reward=_float(metrics.get("net_profit", 0.0)),
        td_delta=_float(signal.get("td_delta")),
        realized_r=_float(metrics.get("expectancy", 0.0)),
        net_profit_delta=_float(metrics.get("net_profit", 0.0)),
        drawdown_delta=max(_float(metrics.get("balance_drawdown_pct", 0.0)), _float(metrics.get("equity_drawdown_pct", 0.0))),
        mistake_tags=tags,
        tag_evidence=evidence,
        suspected_causes=causes,
        proposed_repairs=repairs,
        severity=severity,
        confidence_score=float(np.clip(0.35 + 0.08 * len(tags), 0.05, 1.0)),
    )


@dataclass
class MistakeEpisodeStore:
    schema_version: int = MISTAKE_EPISODE_STORE_SCHEMA_VERSION
    episodes: dict[str, MistakeEpisode] = field(default_factory=dict)
    max_episodes: int | None = None
    _by_trace: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _by_action: dict[int, set[str]] = field(default_factory=dict, init=False, repr=False)
    _by_tag: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _by_context: dict[str, set[str]] = field(default_factory=dict, init=False, repr=False)
    _by_pattern: dict[str, str] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.ensure_compatible()

    def ensure_compatible(self) -> None:
        if not hasattr(self, "schema_version"):
            self.schema_version = MISTAKE_EPISODE_STORE_SCHEMA_VERSION
        if not hasattr(self, "episodes") or self.episodes is None:
            self.episodes = {}
        migrated: dict[str, MistakeEpisode] = {}
        for key, value in dict(self.episodes).items():
            episode = value if isinstance(value, MistakeEpisode) else MistakeEpisode.from_dict(dict(value or {}))
            migrated[str(episode.episode_id or key)] = episode
        self.episodes = migrated
        # Persistent episode memory is open-ended by default.  A positive
        # value is accepted only for explicit deployment retention policies.
        if not hasattr(self, "max_episodes"):
            self.max_episodes = None
        self.max_episodes = None if self.max_episodes is None or int(self.max_episodes) <= 0 else int(self.max_episodes)
        self.rebuild_indexes()

    def copy(self) -> "MistakeEpisodeStore":
        self.ensure_compatible()
        return MistakeEpisodeStore(
            schema_version=int(self.schema_version),
            episodes={key: MistakeEpisode.from_dict(value.to_dict()) for key, value in self.episodes.items()},
            max_episodes=self.max_episodes,
        )

    def rebuild_indexes(self) -> None:
        self._by_trace = defaultdict(set)
        self._by_action = defaultdict(set)
        self._by_tag = defaultdict(set)
        self._by_context = defaultdict(set)
        self._by_pattern = {}
        for episode_id, episode in self.episodes.items():
            if episode.trace_id:
                self._by_trace[str(episode.trace_id)].add(episode_id)
            self._by_action[int(episode.action_index)].add(episode_id)
            for tag in episode.mistake_tags:
                self._by_tag[str(tag)].add(episode_id)
            self._by_context[self._context_bucket(episode)].add(episode_id)
            self._by_pattern.setdefault(episode.pattern_key(), episode_id)

    def add(self, episode: MistakeEpisode) -> MistakeEpisode:
        self.ensure_compatible()
        if not isinstance(episode, MistakeEpisode):
            episode = MistakeEpisode.from_dict(dict(episode or {}))
        episode.last_seen_at = max(float(episode.last_seen_at), _now())
        existing = self.episodes.get(episode.episode_id)
        if existing is None and episode.trace_id:
            for candidate_id in self._by_trace.get(episode.trace_id, set()):
                candidate = self.episodes.get(candidate_id)
                if candidate and candidate.action_index == episode.action_index and set(candidate.mistake_tags) == set(episode.mistake_tags):
                    existing = candidate
                    break
        if existing is None:
            pattern_id = self._by_pattern.get(episode.pattern_key())
            existing = self.episodes.get(pattern_id) if pattern_id else None
        if existing is not None:
            existing.merge_support(episode)
            self.rebuild_indexes()
            return existing
        self.episodes[episode.episode_id] = episode
        self._prune_if_needed()
        self.rebuild_indexes()
        return episode

    def query_recent_bad(self, limit: int = 32, *, min_severity: float = 0.0) -> list[MistakeEpisode]:
        self.ensure_compatible()
        candidates = [item for item in self.episodes.values() if float(item.severity) >= float(min_severity)]
        candidates.sort(key=lambda item: (float(item.last_seen_at), float(item.severity), int(item.support_count)), reverse=True)
        return candidates[: max(0, int(limit))]

    def query_by_action(self, action_index: int, limit: int = 32) -> list[MistakeEpisode]:
        self.ensure_compatible()
        ids = self._by_action.get(int(action_index), set())
        return self._rank(ids, limit)

    def query_by_tag(self, tag: str, limit: int = 32) -> list[MistakeEpisode]:
        self.ensure_compatible()
        ids = self._by_tag.get(str(tag), set())
        return self._rank(ids, limit)

    def query_similar_bad(
        self,
        market_key: Any,
        account_key: Any | None = None,
        position_key: Any | None = None,
        *,
        action_index: int | None = None,
        tags: list[str] | tuple[str, ...] = (),
        limit: int = 32,
    ) -> list[MistakeEpisode]:
        self.ensure_compatible()
        query = np.asarray(
            list(_vector(market_key))
            + list(_vector(account_key or []))
            + list(_vector(position_key or [])),
            dtype=float,
        )
        tag_set = {str(tag) for tag in tags}
        scored: list[tuple[float, MistakeEpisode]] = []
        for episode in self.episodes.values():
            if action_index is not None and int(episode.action_index) != int(action_index):
                continue
            if tag_set and not tag_set.intersection(episode.mistake_tags):
                continue
            key = episode.memory_key()
            size = max(query.size, key.size)
            if size == 0:
                sim = 0.0
            else:
                q = np.zeros(size, dtype=float)
                k = np.zeros(size, dtype=float)
                q[:query.size] = query
                k[:key.size] = key
                denom = float(np.linalg.norm(q) * np.linalg.norm(k))
                sim = 0.0 if denom <= 1e-12 else float((q @ k) / denom)
            scored.append((sim + 0.05 * float(episode.support_count) + 0.10 * float(episode.severity), episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [episode for _, episode in scored[: max(0, int(limit))]]

    def summary(self) -> dict[str, Any]:
        self.ensure_compatible()
        episodes = list(self.episodes.values())
        tag_counts = Counter(tag for episode in episodes for tag in episode.mistake_tags)
        action_counts = Counter(int(episode.action_index) for episode in episodes)
        repeated = [episode for episode in episodes if int(episode.support_count) > 1]
        severity_values = [float(episode.severity) for episode in episodes]
        return {
            "schema_version": int(self.schema_version),
            "count": len(episodes),
            "severity_mean": float(np.mean(severity_values)) if severity_values else 0.0,
            "top_tags": dict(tag_counts.most_common(8)),
            "by_action": {str(key): int(value) for key, value in action_counts.most_common(8)},
            "repeated_mistakes": len(repeated),
            "max_support": max((int(episode.support_count) for episode in episodes), default=0),
            "high_severity": sum(1 for episode in episodes if float(episode.severity) >= 2.0),
        }

    def to_jsonl(self, path: str | Path) -> None:
        self.ensure_compatible()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for episode in sorted(self.episodes.values(), key=lambda item: (float(item.created_at), item.episode_id)):
                handle.write(episode.to_json() + "\n")

    @classmethod
    def load_jsonl(cls, path: str | Path, *, max_episodes: int | None = None) -> "MistakeEpisodeStore":
        store = cls(max_episodes=max_episodes)
        input_path = Path(path)
        if not input_path.exists():
            return store
        with input_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    store.add(MistakeEpisode.from_json(text))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return store

    def _rank(self, ids: set[str], limit: int) -> list[MistakeEpisode]:
        candidates = [self.episodes[item] for item in ids if item in self.episodes]
        candidates.sort(key=lambda item: (float(item.severity), int(item.support_count), float(item.last_seen_at)), reverse=True)
        return candidates[: max(0, int(limit))]

    def _prune_if_needed(self) -> None:
        if self.max_episodes is None:
            return
        overflow = len(self.episodes) - int(self.max_episodes)
        if overflow <= 0:
            return
        candidates = sorted(
            self.episodes.values(),
            key=lambda item: (
                1 if float(item.severity) >= 3.0 and int(item.support_count) > 1 else 0,
                float(item.severity) * (1.0 + np.log1p(int(item.support_count))),
                float(item.last_seen_at),
            ),
        )
        removed = 0
        for episode in candidates:
            if removed >= overflow:
                break
            if float(episode.severity) >= 3.0 and int(episode.support_count) > 1 and len(self.episodes) - removed > overflow:
                continue
            self.episodes.pop(episode.episode_id, None)
            removed += 1

    @staticmethod
    def _context_bucket(episode: MistakeEpisode) -> str:
        return _stable_hash({
            "account": [round(float(item), 1) for item in episode.account_key[:8]],
            "position": [round(float(item), 1) for item in episode.position_key[:6]],
        })
