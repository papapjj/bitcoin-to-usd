from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .mistake_memory import ENTRY_ACTION_NAMES, MistakeEpisode
from .orders import ACTION_COUNT, PARAMETER_OFFSET, Action


MANAGEMENT_ACTIONS = {
    int(Action.CLOSE_WORST),
    int(Action.CLOSE_BEST),
    int(Action.CLOSE_ALL),
    int(Action.MANAGE_POSITION),
}
ENTRY_ACTIONS = {
    int(Action.MARKET_BUY),
    int(Action.MARKET_SELL),
    int(Action.BUY_LIMIT),
    int(Action.SELL_LIMIT),
    int(Action.BUY_STOP),
    int(Action.SELL_STOP),
}


@dataclass
class MistakeReplayItem:
    episode_id: str
    old_action: int
    new_action: int
    repeated_mistake: bool
    safer_alternative: bool
    action_score: float
    risk_score: float
    position_score: float
    total_score: float
    tags: list[str] = field(default_factory=list)
    mutated_edges: int = 0
    mutated_neurons: int = 0
    mutated_module_links: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "old_action": int(self.old_action),
            "new_action": int(self.new_action),
            "repeated_mistake": bool(self.repeated_mistake),
            "safer_alternative": bool(self.safer_alternative),
            "action_score": float(self.action_score),
            "risk_score": float(self.risk_score),
            "position_score": float(self.position_score),
            "total_score": float(self.total_score),
            "tags": list(self.tags),
            "mutated_edges": int(self.mutated_edges),
            "mutated_neurons": int(self.mutated_neurons),
            "mutated_module_links": int(self.mutated_module_links),
        }


class MistakeReplayEngine:
    """Replay bad episodes against a candidate policy and target repeated paths."""

    def __init__(self, *, max_items: int = 64, penalty_strength: float = 0.12, reward_strength: float = 0.04) -> None:
        self.max_items = max(1, int(max_items))
        self.penalty_strength = float(np.clip(penalty_strength, 0.0, 1.0))
        self.reward_strength = float(np.clip(reward_strength, 0.0, 1.0))

    def replay(
        self,
        policy: Any,
        episodes: list[MistakeEpisode] | tuple[MistakeEpisode, ...],
        *,
        hypotheses: list[Any] | tuple[Any, ...] = (),
        mutate: bool = False,
    ) -> dict[str, object]:
        items: list[MistakeReplayItem] = []
        old_account = dict(getattr(policy, "_account_context", {}) or {})
        old_positions = str(getattr(policy, "_positions_pipe_context", "") or "")
        reset_runtime = getattr(policy, "reset_runtime_state", None)
        try:
            for episode in sorted(episodes, key=lambda item: (float(item.severity), int(item.support_count)), reverse=True)[: self.max_items]:
                obs = self._observation_for(policy, episode)
                if callable(reset_runtime):
                    reset_runtime()
                set_account = getattr(policy, "set_account_context", None)
                if callable(set_account):
                    set_account(self._account_context_from_key(episode.account_key), self._positions_pipe_from_episode(episode))
                trace = policy.act(obs)
                new_action = self._action_from_trace(trace)
                old_action = int(episode.action_index)
                repeated = new_action == old_action
                safer = self._is_safer_alternative(episode, new_action)
                risk_score = self._risk_score(episode, trace)
                position_score = self._position_score(episode, new_action)
                action_score = -float(episode.severity) if repeated else (0.35 * float(episode.severity) if safer else 0.05 * float(episode.severity))
                total_score = action_score + risk_score + position_score
                item = MistakeReplayItem(
                    episode_id=str(episode.episode_id),
                    old_action=old_action,
                    new_action=new_action,
                    repeated_mistake=repeated,
                    safer_alternative=safer,
                    action_score=action_score,
                    risk_score=risk_score,
                    position_score=position_score,
                    total_score=total_score,
                    tags=list(episode.mistake_tags),
                )
                if mutate:
                    updates = self._apply_targeted_credit(policy, episode, new_action, repeated=repeated, safer=safer)
                    item.mutated_edges = int(updates.get("edges", 0))
                    item.mutated_neurons = int(updates.get("neurons", 0))
                    item.mutated_module_links = int(updates.get("module_links", 0))
                items.append(item)
        finally:
            set_account = getattr(policy, "set_account_context", None)
            if callable(set_account):
                set_account(old_account, old_positions)

        repeated_count = sum(1 for item in items if item.repeated_mistake)
        safer_count = sum(1 for item in items if item.safer_alternative)
        return {
            "episodes": len(items),
            "hypotheses": len(hypotheses or ()),
            "repeated_mistakes": repeated_count,
            "safer_alternatives": safer_count,
            "mean_score": float(np.mean([item.total_score for item in items])) if items else 0.0,
            "mutated_edges": sum(item.mutated_edges for item in items),
            "mutated_neurons": sum(item.mutated_neurons for item in items),
            "mutated_module_links": sum(item.mutated_module_links for item in items),
            "items": [item.to_dict() for item in items],
        }

    @staticmethod
    def _observation_for(policy: Any, episode: MistakeEpisode) -> np.ndarray:
        input_size = max(1, int(getattr(policy, "input_size", len(episode.raw_obs) or len(episode.market_key) or 1)))
        source = episode.raw_obs if episode.raw_obs else episode.market_key
        arr = np.asarray(source, dtype=float).reshape(-1)
        obs = np.zeros(input_size, dtype=float)
        obs[: min(input_size, arr.size)] = arr[:input_size]
        return obs

    @staticmethod
    def _action_from_trace(trace: Any) -> int:
        details = dict(getattr(trace, "action_details", {}) or {})
        for key in ("action_index", "action", "bridge_action"):
            if key in details:
                try:
                    return int(float(details.get(key)))
                except (TypeError, ValueError):
                    pass
        selected = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        if selected.size:
            return int(np.argmax(selected[: min(ACTION_COUNT, selected.size)]))
        return 0

    @staticmethod
    def _is_safer_alternative(episode: MistakeEpisode, new_action: int) -> bool:
        tags = set(episode.mistake_tags)
        old_is_entry = int(episode.action_index) in ENTRY_ACTIONS or str(episode.action_name) in ENTRY_ACTION_NAMES
        if old_is_entry and tags & {"wrong_direction", "stress_entry", "high_spread_entry", "late_entry", "bad_risk_pct"}:
            return int(new_action) not in ENTRY_ACTIONS
        if tags & {"near_tp_giveback", "missed_partial", "missed_trailing", "held_loser_too_long", "premature_strong_trend_exit"}:
            return int(new_action) in MANAGEMENT_ACTIONS or int(new_action) == int(Action.HOLD)
        if "no_trade_starvation" in tags:
            return int(new_action) in ENTRY_ACTIONS
        return int(new_action) != int(episode.action_index)

    @staticmethod
    def _risk_score(episode: MistakeEpisode, trace: Any) -> float:
        selected = np.asarray(getattr(trace, "selected_action", []), dtype=float).reshape(-1)
        if selected.size <= PARAMETER_OFFSET:
            return 0.0
        # Output parameter 0 maps to risk in the action schema.  Only score
        # relative improvement; exact lot conversion remains in orders.py/EA.
        candidate_risk_proxy = float(np.clip((selected[PARAMETER_OFFSET] + 1.0) * 0.5 * 5.0, 0.0, 5.0))
        old_risk = float(max(0.0, episode.risk_pct))
        if old_risk <= 0.0:
            return 0.0
        return float(np.clip((old_risk - candidate_risk_proxy) / max(0.1, old_risk), -1.0, 1.0) * 0.25 * max(0.1, episode.severity))

    @staticmethod
    def _position_score(episode: MistakeEpisode, new_action: int) -> float:
        tags = set(episode.mistake_tags)
        if tags & {"near_tp_giveback", "missed_partial", "missed_trailing", "premature_strong_trend_exit"}:
            return 0.25 * float(episode.severity) if int(new_action) in MANAGEMENT_ACTIONS else -0.15 * float(episode.severity)
        if "held_loser_too_long" in tags:
            return 0.20 * float(episode.severity) if int(new_action) in MANAGEMENT_ACTIONS else -0.10 * float(episode.severity)
        return 0.0

    @staticmethod
    def _account_context_from_key(account_key: list[float]) -> dict[str, float]:
        values = list(account_key or [])
        names = (
            "equity_to_balance",
            "floating_pnl_r",
            "current_equity_drawdown",
            "open_risk_ratio",
            "equity_return_1_bar",
            "equity_return_5_bar",
            "equity_volatility_20",
            "day_trade_pressure",
        )
        return {name: float(values[idx]) if idx < len(values) else 0.0 for idx, name in enumerate(names)}

    @staticmethod
    def _positions_pipe_from_episode(episode: MistakeEpisode) -> str:
        parts: list[str] = []
        for item in list(episode.positions_before or []):
            parts.append(":".join([
                str(item.get("ticket", "")),
                str(item.get("side", 0)),
                str(item.get("volume", 0.0)),
                str(item.get("entry", 0.0)),
                str(item.get("sl", 0.0)),
                str(item.get("tp", 0.0)),
                str(item.get("unrealized_r", 0.0)),
                str(item.get("age_bars", 0.0)),
                str(item.get("mfe_r", item.get("unrealized_r", 0.0))),
                str(item.get("mae_r", item.get("unrealized_r", 0.0))),
                str(item.get("dist_tp_atr", 0.0)),
                str(item.get("dist_sl_atr", 0.0)),
                str(item.get("min_dist_tp_atr", 0.0)),
                str(item.get("min_dist_sl_atr", 0.0)),
                str(item.get("bars_since_mfe", 0.0)),
                "1" if bool(item.get("near_tp", False)) else "0",
                str(item.get("giveback_r", 0.0)),
            ]))
        return "^".join(parts)

    def _apply_targeted_credit(
        self,
        policy: Any,
        episode: MistakeEpisode,
        new_action: int,
        *,
        repeated: bool,
        safer: bool,
    ) -> dict[str, int]:
        severity = float(np.clip(episode.severity * (1.0 + 0.15 * max(0, int(episode.support_count) - 1)), 0.05, 5.0))
        penalty = self.penalty_strength * severity
        reward = self.reward_strength * severity
        action_idx = int(episode.action_index)
        module_links = 0
        graph_edges = 0
        graph_neurons = 0
        if repeated:
            # V10 principle: repeated mistakes do not suppress the old action.
            # They create and strengthen a cause-conditioned replacement action.
            replacement = self._replacement_action_for_cause(episode, action_idx, int(new_action))
            module_links += self._reward_module_action(policy, replacement, max(reward, 0.35 * penalty))
            graph_edges += self._mark_graph_replacement_pressure(policy, episode, replacement, penalty)
        elif safer:
            module_links += self._reward_module_action(policy, int(new_action), reward)
        return {"module_links": module_links, "edges": graph_edges, "neurons": graph_neurons}


    @staticmethod
    def _replacement_action_for_cause(episode: MistakeEpisode, old_action: int, observed_new_action: int) -> int:
        """Return a cause-conditioned replacement without penalizing old_action.

        Direction mistakes generate opposite/confirmation alternatives; entry
        quality mistakes generate wait/management alternatives; risk/execution
        mistakes prefer HOLD/MANAGE. This is intentionally positive-only: it
        supplies a replacement target for reward, not a suppression target for
        the old action.
        """
        tags = set(str(tag) for tag in (episode.mistake_tags or []))
        old = int(old_action)
        if tags & {"wrong_direction", "direction_error"}:
            if old == int(Action.MARKET_BUY):
                return int(Action.MARKET_SELL)
            if old == int(Action.MARKET_SELL):
                return int(Action.MARKET_BUY)
            if old == int(Action.BUY_LIMIT):
                return int(Action.SELL_LIMIT)
            if old == int(Action.SELL_LIMIT):
                return int(Action.BUY_LIMIT)
            if old == int(Action.BUY_STOP):
                return int(Action.SELL_STOP)
            if old == int(Action.SELL_STOP):
                return int(Action.BUY_STOP)
        if tags & {"late_entry", "stress_entry", "high_spread_entry", "bad_risk_pct", "spread_slippage"}:
            return int(Action.HOLD)
        if tags & {"near_tp_giveback", "missed_partial", "missed_trailing", "held_loser_too_long", "premature_strong_trend_exit"}:
            return int(Action.MANAGE_POSITION)
        if tags & {"no_trade_starvation", "missed_opportunity"}:
            return int(Action.MARKET_BUY if old != int(Action.MARKET_BUY) else Action.MARKET_SELL)
        if int(observed_new_action) != old:
            return int(observed_new_action)
        return int(Action.HOLD)

    @staticmethod
    def _mark_graph_replacement_pressure(policy: Any, episode: MistakeEpisode, replacement_action: int, strength: float) -> int:
        """Mark graph nodes/edges for replacement learning without weakening them."""
        updated = 0
        active_edges = set(str(item) for item in episode.active_edge_ids)
        active_nodes = set(str(item) for item in episode.active_neuron_ids)
        for module in getattr(policy, "modules", []) or []:
            graph = getattr(getattr(module, "m1_component", None), "graph", None)
            if graph is None:
                continue
            for edge_id in active_edges:
                edge = getattr(graph, "edges", {}).get(edge_id)
                if edge is None:
                    continue
                edge.eligibility = float(np.clip(getattr(edge, "eligibility", 0.0) + 0.1 * strength, 0.0, 100.0))
                # Do not reduce gate/weight/utility here. Replacement learning is
                # expressed by strengthening the replacement output path elsewhere.
                updated += 1
            for node_id in active_nodes:
                node = getattr(graph, "neurons", {}).get(node_id)
                if node is None:
                    continue
                node.eligibility = float(np.clip(getattr(node, "eligibility", 0.0) + 0.1 * strength, 0.0, 100.0))
                updated += 1
        return updated

    @staticmethod
    def _penalize_module_action(policy: Any, episode: MistakeEpisode, action_idx: int, penalty: float) -> int:
        updated = 0
        traces = {str(item.get("module_name", "")): item for item in episode.module_action_traces}
        for module in getattr(policy, "modules", []) or []:
            if action_idx >= getattr(module, "b_out", np.zeros(0)).size:
                continue
            module.b_out[action_idx] = float(np.clip(module.b_out[action_idx] - 0.15 * penalty, -8.0, 8.0))
            trace = traces.get(str(getattr(module, "name", "")))
            if trace:
                hidden = np.asarray(trace.get("hidden", []), dtype=float).reshape(-1)
            else:
                hidden = np.asarray(getattr(module, "neuron_activation_ema", []), dtype=float).reshape(-1)
            if hidden.size and getattr(module, "w_out", np.zeros((0, 0))).shape[0] > action_idx:
                h = np.zeros(module.w_out.shape[1], dtype=float)
                h[: min(h.size, hidden.size)] = hidden[: h.size]
                active = np.abs(h) > 0.03
                if np.any(active):
                    module.w_out[action_idx, active] -= 0.01 * penalty * np.tanh(h[active])
                    module.w_out[action_idx, :] = np.clip(module.w_out[action_idx, :], -8.0, 8.0)
                    updated += int(np.count_nonzero(active))
            updated += 1
        return updated

    @staticmethod
    def _reward_module_action(policy: Any, action_idx: int, reward: float) -> int:
        updated = 0
        for module in getattr(policy, "modules", []) or []:
            if action_idx >= getattr(module, "b_out", np.zeros(0)).size:
                continue
            module.b_out[action_idx] = float(np.clip(module.b_out[action_idx] + 0.08 * reward, -8.0, 8.0))
            updated += 1
        return updated

    @staticmethod
    def _penalize_graph_edges(policy: Any, episode: MistakeEpisode, penalty: float) -> int:
        updated = 0
        active = set(str(item) for item in episode.active_edge_ids)
        if not active:
            return 0
        for module in getattr(policy, "modules", []) or []:
            graph = getattr(getattr(module, "m1_component", None), "graph", None)
            if graph is None:
                continue
            for edge_id in active:
                edge = getattr(graph, "edges", {}).get(edge_id)
                if edge is None:
                    continue
                edge.utility = float(np.clip(edge.utility - penalty, -5.0, 5.0))
                edge.gate = float(np.clip(edge.gate - 0.25 * penalty, -8.0, 8.0))
                edge.weight = float(np.clip(edge.weight * (1.0 - min(0.4, 0.10 * penalty)), -8.0, 8.0))
                edge.eligibility = float(np.clip(edge.eligibility + 0.1 * penalty, 0.0, 100.0))
                updated += 1
        return updated

    @staticmethod
    def _penalize_graph_nodes(policy: Any, episode: MistakeEpisode, action_idx: int, penalty: float) -> int:
        updated = 0
        active = set(str(item) for item in episode.active_neuron_ids)
        if not active:
            return 0
        for module in getattr(policy, "modules", []) or []:
            graph = getattr(getattr(module, "m1_component", None), "graph", None)
            if graph is None:
                continue
            for node_id in active:
                node = getattr(graph, "neurons", {}).get(node_id)
                if node is None:
                    continue
                node.utility = float(np.clip(node.utility - penalty, -5.0, 5.0))
                node.eligibility = float(np.clip(node.eligibility + 0.1 * penalty, 0.0, 100.0))
                if getattr(node, "region", "") == "output":
                    try:
                        output_idx = int(str(getattr(node, "id", "")).rsplit(":", 1)[-1])
                    except ValueError:
                        output_idx = -1
                    if output_idx == int(action_idx):
                        node.bias = float(np.clip(node.bias - 0.20 * penalty, -8.0, 8.0))
                updated += 1
        return updated
