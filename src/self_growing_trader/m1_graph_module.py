from __future__ import annotations

from typing import Mapping

import numpy as np

from .m1_graph_factory import seed_m1_graph
from .neural_graph import (
    REGION_CRITIC,
    REGION_INPUT,
    REGION_MEMORY,
    REGION_OUTPUT,
    REGION_PREDICTOR,
    REGION_SELECTOR,
    GraphActivationTrace,
    NeuralGraph,
    NeuralGraphBudget,
)


class M1GraphModule:
    """Graph-native M1 cognition module.

    The stable public action tuple is preserved for the bridge.  All internal
    state, learning and growth are graph-native; no retired-core compatibility path is
    instantiated or converted at runtime.
    """

    def __init__(self, graph: NeuralGraph, *, state_size: int, K: int, activation_steps: int = 3) -> None:
        self.graph = graph
        self.state_size = int(state_size)
        self.K = int(K)
        self.activation_steps = max(1, int(activation_steps))
        self.memory_bank: list[np.ndarray] = []
        self.memory_bank_max: int | None = None
        self.prev_output: np.ndarray | None = None
        self.prev_predicted_state: np.ndarray | None = None
        self.last_trace: GraphActivationTrace | None = None
        self.last_context_summary: dict[str, float] = {}
        # Semantic M1 evidence retained alongside the compatible return tuple.
        self.last_candidate_actions = np.zeros((max(1, self.K), self.output_size), dtype=float)
        self.last_candidate_scores = np.zeros(max(1, self.K), dtype=float)
        self.last_candidate_logits = np.zeros(max(1, self.K), dtype=float)
        self.last_selected_idx = 0
        self.last_selected_log_prob = 0.0
        self.last_predicted_state = np.zeros(max(1, self.state_size), dtype=float)
        self.last_critic_value = 0.0
        # V18: a bounded policy-level reasoning bias can enter the score path
        # before the selector chooses an M1 hypothesis.  It is diagnostic
        # state, not a persistent topology mutation.
        self.last_reasoning_score_bias = np.zeros(max(1, self.K), dtype=float)
        self.last_action_generator_summary: dict[str, float] = {}
        # Read-only snapshot of the cognitive pipeline for live observability.
        self.last_thinking_output: dict[str, object] = {}

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self.memory_bank = list(getattr(self, "memory_bank", []))
        raw_memory_limit = getattr(self, "memory_bank_max", None)
        self.memory_bank_max = None if raw_memory_limit in (None, 0) else max(1, int(raw_memory_limit))
        self.prev_output = getattr(self, "prev_output", None)
        self.prev_predicted_state = getattr(self, "prev_predicted_state", None)
        self.last_trace = getattr(self, "last_trace", None)
        self.last_context_summary = dict(getattr(self, "last_context_summary", {}) or {})
        self.last_candidate_actions = np.asarray(getattr(self, "last_candidate_actions", np.zeros((max(1, self.K), self.output_size))), dtype=float)
        self.last_candidate_scores = np.asarray(getattr(self, "last_candidate_scores", np.zeros(max(1, self.K))), dtype=float)
        self.last_candidate_logits = np.asarray(getattr(self, "last_candidate_logits", self.last_candidate_scores), dtype=float)
        self.last_selected_idx = int(getattr(self, "last_selected_idx", 0))
        # Raw hypothesis choice and final semantic action are retained
        # separately so a safety gate cannot look like an unexplained action
        # substitution in telemetry.
        self.last_selected_action_index = int(getattr(self, "last_selected_action_index", 0))
        self.last_selected_hypothesis_action_index = int(getattr(
            self,
            "last_selected_hypothesis_action_index",
            self.last_selected_action_index,
        ))
        self.last_selected_log_prob = float(getattr(self, "last_selected_log_prob", 0.0))
        self.last_predicted_state = np.asarray(getattr(self, "last_predicted_state", np.zeros(max(1, self.state_size))), dtype=float)
        self.last_critic_value = float(getattr(self, "last_critic_value", 0.0))
        self.last_reasoning_score_bias = np.asarray(getattr(self, "last_reasoning_score_bias", np.zeros(max(1, self.K))), dtype=float).reshape(-1)
        if self.last_reasoning_score_bias.size != max(1, self.K):
            self.last_reasoning_score_bias = np.resize(self.last_reasoning_score_bias, max(1, self.K)) if self.last_reasoning_score_bias.size else np.zeros(max(1, self.K), dtype=float)
        self.last_action_generator_summary = dict(getattr(self, "last_action_generator_summary", {}) or {})
        self.last_thinking_output = dict(getattr(self, "last_thinking_output", {}) or {})

    @classmethod
    def seed(
        cls,
        *,
        input_size: int,
        output_size: int,
        state_size: int,
        K: int,
        seed: int | None = None,
        activation_steps: int = 3,
    ) -> "M1GraphModule":
        return cls(
            seed_m1_graph(
                input_size=input_size,
                output_size=output_size,
                state_size=state_size,
                hypothesis_count=K,
                seed=seed,
            ),
            state_size=state_size,
            K=K,
            activation_steps=activation_steps,
        )

    @property
    def input_size(self) -> int:
        return len(self.graph.regions[REGION_INPUT].neuron_ids)

    @property
    def output_size(self) -> int:
        return len(self.graph.regions[REGION_OUTPUT].neuron_ids)

    def forward(
        self,
        x: np.ndarray,
        goal: np.ndarray | None = None,
        context: object | None = None,
    ) -> tuple[np.ndarray, float, float, float, float]:
        # Goal is a first-class decision context.  Earlier graph versions
        # discarded it here, which made online reasoning invariant to risk and
        # objective inputs despite the caller supplying them.
        obs = np.asarray(x, dtype=float).reshape(-1)
        if isinstance(context, Mapping):
            graph_context: dict[str, object] = dict(context)
        else:
            graph_context = {}
        if goal is not None:
            graph_context["goal_vector"] = np.asarray(goal, dtype=float).reshape(-1)
        region_context = _context_to_regions(graph_context, self.graph)
        _graph_output, trace = self.graph.activate(
            obs,
            steps=self.activation_steps,
            context_by_region=region_context,
            return_trace=True,
        )
        self.last_trace = trace
        state = self._region_state("encoder")
        memory = self._region_state(REGION_MEMORY)
        hypothesis = self._region_state("hypothesis")
        selector = self._region_state(REGION_SELECTOR)
        predicted = self._region_state(REGION_PREDICTOR)
        critic = self._region_state(REGION_CRITIC)
        candidate_actions = self._candidate_actions(hypothesis)
        # M1 v9: generate actions through neural replacement/success/probe
        # pathways.  Negative memories no longer suppress the losing action;
        # they create cause-conditioned replacement pressure that competes as a
        # positive proposal against the current hypothesis set.
        generator = _neural_action_generator_context(context, candidate_actions.shape[1] if candidate_actions.ndim == 2 else self.output_size)
        candidate_actions = self._augment_candidate_actions(candidate_actions, generator)
        # A context can forbid every raw neural hypothesis at startup.  In that
        # case the selector must receive one *legal* no-entry hypothesis rather
        # than choose an illegal candidate and rely on a later bridge repair.
        # This adds only the invariant HOLD transport action, not a trading
        # strategy or a fallback policy.
        candidate_actions = self._ensure_legal_hold_candidate(candidate_actions, graph_context)
        scores = self._candidate_scores(hypothesis, selector, critic, candidate_actions, graph_context)
        scores = self._apply_neural_action_generator_scores(scores, candidate_actions, generator)
        # V18: policy-level reasoning arrives before the M1 selector chooses
        # a hypothesis.  The vector is centred, bounded and then passed through
        # the ordinary legality gate below; it can prefer among M1 proposals but
        # cannot create a new or illegal action.
        raw_reasoning_bias = graph_context.get(
            "reasoning_candidate_bias",
            getattr(self, "_runtime_reasoning_candidate_bias", np.zeros(0, dtype=float)),
        )
        try:
            reasoning_bias = np.asarray(raw_reasoning_bias, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            reasoning_bias = np.zeros(0, dtype=float)
        if reasoning_bias.size:
            if reasoning_bias.size != scores.size:
                reasoning_bias = np.resize(reasoning_bias, scores.size)
            reasoning_bias = np.nan_to_num(reasoning_bias, nan=0.0, posinf=0.0, neginf=0.0)
            reasoning_bias = np.tanh(reasoning_bias - float(np.mean(reasoning_bias)))
            raw_gain = graph_context.get(
                "reasoning_candidate_gain",
                getattr(self, "_runtime_reasoning_candidate_gain", 0.0),
            )
            try:
                reasoning_gain = float(raw_gain)
            except (TypeError, ValueError):
                reasoning_gain = 0.0
            reasoning_gain = float(np.clip(reasoning_gain, 0.0, 0.18))
            applied_reasoning_bias = reasoning_gain * reasoning_bias
            scores = scores + applied_reasoning_bias
        else:
            applied_reasoning_bias = np.zeros_like(scores, dtype=float)
        self.last_reasoning_score_bias = np.asarray(applied_reasoning_bias, dtype=float).copy()
        # Legal action constraints are supplied before selection, not repaired
        # only by the bridge after a candidate has won.
        scores = self._apply_action_legality_to_scores(scores, candidate_actions, graph_context)
        probabilities = _softmax(scores)
        selected_idx = int(np.argmax(scores)) if scores.size else 0
        selected_hypothesis_action = int(
            np.argmax(candidate_actions[selected_idx, :min(12, candidate_actions.shape[1])])
        ) if candidate_actions.size else 0
        output = candidate_actions[selected_idx].copy() if candidate_actions.size else np.zeros(self.output_size, dtype=float)
        output = self._apply_neural_action_generator_output(output, generator, probabilities)
        output = _mask_invalid_action_logits(output, graph_context)
        entry_gate = _entry_gate(graph_context)
        if output.size > 1 and entry_gate < 1.0:
            end = min(output.size, 7)
            stress = 1.0 - entry_gate
            output = output.copy()
            output[1:end] = np.tanh(entry_gate * output[1:end] - 0.75 * stress)

        if self.prev_predicted_state is None:
            error = float(np.linalg.norm(_resize(state, memory.size) - memory))
        else:
            error = float(np.linalg.norm(_resize(state, self.prev_predicted_state.size) - self.prev_predicted_state))
        self.prev_predicted_state = predicted.copy()
        self.prev_output = output.copy()
        novelty = self._novelty(state)
        self.memory_bank.append(state.copy())
        if self.memory_bank_max is not None and len(self.memory_bank) > self.memory_bank_max:
            self.memory_bank = self.memory_bank[-self.memory_bank_max:]
        confidence = float(probabilities[selected_idx]) if probabilities.size else 0.0
        conflict = _conflict_from_probabilities(probabilities)
        self.last_candidate_actions = candidate_actions.copy()
        self.last_candidate_scores = scores.copy()
        self.last_candidate_logits = scores.copy()
        self.last_selected_idx = int(selected_idx)
        self.last_selected_hypothesis_action_index = int(selected_hypothesis_action)
        self.last_selected_action_index = int(np.argmax(output[:min(12, output.size)])) if output.size else 0
        self.last_selected_log_prob = float(np.log(max(confidence, 1e-12)))
        self.last_predicted_state = predicted.copy()
        self.last_critic_value = float(scores[selected_idx]) if scores.size else 0.0
        self.last_thinking_output = {
            "schema": "M1_THINKING_V8",
            "encoder": np.asarray(state, dtype=float).tolist(),
            "memory": np.asarray(memory, dtype=float).tolist(),
            "hypotheses": np.asarray(hypothesis, dtype=float).tolist(),
            "selector": np.asarray(selector, dtype=float).tolist(),
            "predictor": np.asarray(predicted, dtype=float).tolist(),
            "critic": np.asarray(critic, dtype=float).tolist(),
            "candidate_scores": np.asarray(scores, dtype=float).tolist(),
            "candidate_probabilities": np.asarray(probabilities, dtype=float).tolist(),
            "candidate_actions": np.asarray(candidate_actions, dtype=float).tolist(),
            "selected_hypothesis": int(selected_idx),
            "selected_confidence": float(confidence),
            "selected_critic_value": float(self.last_critic_value),
            "entry_gate": float(entry_gate),
            "prediction_error": float(error),
            "novelty": float(novelty),
            "conflict": float(conflict),
            "reasoning_score_bias": np.asarray(self.last_reasoning_score_bias, dtype=float).tolist(),
            "action_generator": dict(self.last_action_generator_summary),
            "trace": {
                "active_neurons": int(len(getattr(trace, "active_neuron_ids", ()) or ())),
                "active_edges": int(trace.active_edge_count),
                "recorded_edges": int(trace.traced_edge_count),
                "region_activity": {str(k): float(v) for k, v in dict(getattr(trace, "region_activity", {}) or {}).items()},
            },
        }
        self.last_context_summary = {
            "memory": float(np.mean(np.abs(region_context.get(REGION_MEMORY, np.zeros(1))))),
            "predictor": float(np.mean(np.abs(region_context.get(REGION_PREDICTOR, np.zeros(1))))),
            "critic": float(np.mean(np.abs(region_context.get(REGION_CRITIC, np.zeros(1))))),
            "selector": float(np.mean(np.abs(region_context.get(REGION_SELECTOR, np.zeros(1))))),
            "entry_gate": entry_gate,
            "selected_hypothesis": float(selected_idx),
            "trace_active_edges": float(trace.active_edge_count),
            "trace_recorded_edges": float(trace.traced_edge_count),
            "goal": _mean_abs(_vector(graph_context.get("goal_vector"))) if graph_context else 0.0,
            "portfolio": _mean_abs(_vector(graph_context.get("portfolio_context"))) if graph_context else 0.0,
            "reasoning_score_bias_abs": float(np.mean(np.abs(self.last_reasoning_score_bias))) if self.last_reasoning_score_bias.size else 0.0,
            "position": _mean_abs(_vector(graph_context.get("position_context"))) if graph_context else 0.0,
            "execution": _mean_abs(_vector(graph_context.get("execution_context"))) if graph_context else 0.0,
            "legal_action_fraction": float(np.mean(_allowed_action_mask(graph_context, self.output_size))) if graph_context else 1.0,
            "expectancy_best_lcb": float(_scalar(graph_context.get("expectancy_best_lcb", 0.0))) if graph_context else 0.0,
            "expectancy_regret": float(_scalar(graph_context.get("expectancy_regret", 0.0))) if graph_context else 0.0,
        }
        return output, error, novelty, conflict, confidence

    def thinking_snapshot(self) -> dict[str, object]:
        """Return the latest M1 cognitive output without mutating policy state."""
        return dict(getattr(self, "last_thinking_output", {}) or {})

    def _region_state(self, region: str) -> np.ndarray:
        ids = tuple(self.graph.regions[region].neuron_ids)
        return np.asarray([self.graph.neurons[node_id].activation for node_id in ids], dtype=float)

    def _candidate_actions(self, hypothesis: np.ndarray) -> np.ndarray:
        """Decode per-hypothesis actions without scanning edge objects in Python.

        The M1 graph can contain thousands of non-decoder edges. Candidate
        decoding only requires hypothesis→output paths, so this uses the graph's
        stable compiled index and a vectorised scatter add. It is intentionally
        separate from activation tracing: score candidates must be cheap enough
        to run on every bridge tick.
        """
        compiled = self.graph._compiled()
        node_count = len(compiled["node_ids"])
        output_ids = tuple(self.graph.regions[REGION_OUTPUT].neuron_ids)
        hypothesis_ids = tuple(self.graph.regions["hypothesis"].neuron_ids)
        k = max(1, min(self.K, len(hypothesis_ids), np.asarray(hypothesis).size))
        candidate = np.tile(
            np.asarray([self.graph.neurons[node_id].bias for node_id in output_ids], dtype=float),
            (max(1, self.K), 1),
        )
        if not output_ids or k <= 0 or not self.graph.edges:
            return np.tanh(np.nan_to_num(candidate, nan=0.0, posinf=1.0, neginf=-1.0))

        node_to_hyp = np.full(node_count, -1, dtype=int)
        node_to_output = np.full(node_count, -1, dtype=int)
        index = compiled["index"]
        for position, node_id in enumerate(hypothesis_ids[:k]):
            node_to_hyp[int(index[node_id])] = position
        for position, node_id in enumerate(output_ids):
            node_to_output[int(index[node_id])] = position
        edge_ids = tuple(compiled["edge_ids"])
        src = np.asarray(compiled["edge_src"], dtype=int)
        dst = np.asarray(compiled["edge_dst"], dtype=int)
        src_hyp = node_to_hyp[src]
        dst_output = node_to_output[dst]
        enabled = np.fromiter((bool(self.graph.edges[edge_id].enabled) for edge_id in edge_ids), dtype=bool, count=len(edge_ids))
        mask = enabled & (src_hyp >= 0) & (dst_output >= 0)
        if np.any(mask):
            selected_ids = np.flatnonzero(mask)
            weights = np.fromiter((float(self.graph.edges[edge_ids[idx]].weight) for idx in selected_ids), dtype=float, count=selected_ids.size)
            gates = np.fromiter((float(self.graph.edges[edge_ids[idx]].gate) for idx in selected_ids), dtype=float, count=selected_ids.size)
            source_values = np.asarray(hypothesis, dtype=float).reshape(-1)[src_hyp[selected_ids]]
            values = (1.0 / (1.0 + np.exp(-np.clip(gates, -60.0, 60.0)))) * weights * source_values
            np.add.at(candidate, (src_hyp[selected_ids], dst_output[selected_ids]), values)
        return np.tanh(np.nan_to_num(candidate, nan=0.0, posinf=1.0, neginf=-1.0))

    def _candidate_scores(
        self,
        hypothesis: np.ndarray,
        selector: np.ndarray,
        critic: np.ndarray,
        candidate_actions: np.ndarray,
        context: object | None,
    ) -> np.ndarray:
        candidate_count = int(candidate_actions.shape[0]) if np.asarray(candidate_actions).ndim == 2 else max(1, self.K)
        scores = _resize(selector, candidate_count) + 0.35 * _resize(hypothesis, candidate_count)
        # Critic influences the selection score rather than being a dead trace
        # region. Selector wiring remains the primary per-hypothesis signal.
        if critic.size:
            scores += 0.10 * float(np.mean(critic))
        action_bias = _extract_action_bias(context, candidate_actions.shape[1])
        if action_bias is not None and candidate_actions.size:
            scores += 0.06 * np.maximum(0.0, candidate_actions @ np.clip(action_bias, -1.0, 1.0))
        if isinstance(context, Mapping) and candidate_actions.size:
            # Robust lower-bound evidence is the only direct positive action
            # pressure.  A candidate may receive it only through the exact
            # action components it proposes.
            action_size = candidate_actions.shape[1]
            exp_lcb = _resize(_vector(context.get("expectancy_action_lcb")), action_size)
            exp_conf = _resize(_vector(context.get("expectancy_action_confidence")), action_size)
            positive_edge = np.maximum(0.0, exp_lcb) * np.clip(exp_conf, 0.0, 1.0)
            if np.any(positive_edge > 1e-9):
                scores += 0.24 * np.maximum(0.0, candidate_actions @ np.clip(positive_edge, 0.0, 1.0))

            # Negative evidence must inhibit only the same primary action in
            # the same retrieved context.  Do not subtract a whole action
            # family: a bad BUY record must not suppress a SELL alternative.
            action_count = min(action_size, 12)
            if action_count > 0:
                primary = np.argmax(candidate_actions[:, :action_count], axis=1)
                reliable_negative = _resize(_vector(context.get("expectancy_reliable_negative_mask")), action_count) > 0.5
                negative_strength = _resize(_vector(context.get("expectancy_negative_inhibition")), action_count)
                # Backward-safe derivation for artifacts created before V8.
                if not np.any(negative_strength > 0.0):
                    negative_strength = np.maximum(0.0, -exp_lcb[:action_count]) * np.clip(exp_conf[:action_count], 0.0, 1.0)
                negative_strength = np.clip(negative_strength, 0.0, 2.5)
                local_penalty = np.where(reliable_negative[primary], negative_strength[primary], 0.0)
                scores -= 0.32 * local_penalty
        goal = _vector(context.get("goal_vector")) if isinstance(context, Mapping) else np.zeros(0, dtype=float)
        if goal.size and candidate_actions.size:
            # The selector receives goal context through graph regions as well.
            # This bounded alignment makes the goal visibly affect candidate
            # comparison even before later graph growth has specialised paths.
            goal_projection = _resize(goal, candidate_actions.shape[1])
            scores += 0.035 * (candidate_actions @ np.clip(goal_projection, -1.0, 1.0))
        return np.nan_to_num(scores, nan=0.0, posinf=5.0, neginf=-5.0)


    def _ensure_legal_hold_candidate(
        self,
        candidate_actions: np.ndarray,
        context: object | None,
    ) -> np.ndarray:
        """Guarantee a legal no-entry candidate for fail-closed context routing.

        The candidate set remains neural. This method only ensures that when
        every neural proposal is outside the current context action contract,
        the selector can choose HOLD instead of an illegal action that would
        later be rewritten by legality or the MT5 bridge.
        """
        actions = np.asarray(candidate_actions, dtype=float)
        if actions.ndim != 2 or actions.size == 0:
            actions = np.zeros((0, self.output_size), dtype=float)
        width = actions.shape[1] if actions.ndim == 2 and actions.shape[1] else int(self.output_size)
        count = min(12, width)
        if count <= 0:
            return actions
        mask = _allowed_action_mask(context, count)
        if actions.shape[0] > 0:
            primary = np.argmax(actions[:, :count], axis=1)
            if np.any(mask[np.asarray(primary, dtype=int)]):
                return actions
        hold = np.zeros(width, dtype=float)
        hold[:count] = -1.0
        hold[0] = 1.0
        return np.vstack([actions, hold.reshape(1, -1)])

    def _apply_action_legality_to_scores(
        self,
        scores: np.ndarray,
        candidate_actions: np.ndarray,
        context: object | None,
    ) -> np.ndarray:
        result = np.asarray(scores, dtype=float).reshape(-1).copy()
        actions = np.asarray(candidate_actions, dtype=float)
        if result.size == 0 or actions.ndim != 2 or actions.shape[0] != result.size:
            return result
        mask = _allowed_action_mask(context, actions.shape[1])
        action_count = min(mask.size, actions.shape[1])
        if action_count <= 0:
            return result
        primary = np.argmax(actions[:, :action_count], axis=1)
        invalid = ~mask[np.asarray(primary, dtype=int)]
        # A legal HOLD candidate is always available through the frame's
        # fail-closed mask.  Use a decisive score, not a weak preference that a
        # later generator bonus can undo.
        result[invalid] = -50.0
        return result

    def _augment_candidate_actions(self, candidate_actions: np.ndarray, generator: dict[str, object]) -> np.ndarray:
        """Append only evidence-backed replacement hypotheses.

        Curiosity/probe vectors are intentionally excluded from live candidate
        construction.  They can request replay or shadow evaluation elsewhere,
        but uncertainty must never manufacture a BUY/SELL candidate.
        """
        actions = np.asarray(candidate_actions, dtype=float)
        if actions.ndim != 2 or actions.size == 0:
            actions = np.zeros((1, self.output_size), dtype=float)
        size = actions.shape[1]
        proposals: list[np.ndarray] = [row.copy() for row in actions]
        replacement = np.asarray(generator.get("replacement", np.zeros(size)), dtype=float).reshape(-1)
        success = np.asarray(generator.get("success", np.zeros(size)), dtype=float).reshape(-1)
        for vec, gain in ((replacement, 0.75), (success, 0.55)):
            if vec.size == size and np.any(np.abs(vec) > 1e-9):
                proposals.append(np.tanh(gain * np.clip(vec, -1.0, 1.0)))

        # A direct entry hypothesis is legal only when contextual expectancy
        # has verified one specific entry action with positive LCB, support and
        # confidence.  The side/type comes from the evidence record itself;
        # no side-prior, random alternation or uncertainty-derived probe is
        # allowed here.
        opportunity = float(np.clip(float(generator.get("opportunity", 0.0)), 0.0, 1.0))
        verified = bool(generator.get("verified_entry_evidence", False))
        verified_action = int(generator.get("verified_entry_action", 0) or 0)
        if verified and 1 <= verified_action <= min(6, size - 1) and opportunity > 0.0:
            evidence_vec = np.zeros(size, dtype=float)
            evidence_vec[verified_action] = float(np.clip(0.55 + 0.45 * opportunity, 0.0, 1.0))
            evidence_vec[0] = -float(np.clip(0.25 + 0.20 * opportunity, 0.0, 0.55))
            proposals.append(np.tanh(evidence_vec))
            generator["entry_evidence_start"] = int(len(proposals) - 1)
            generator["entry_evidence_count"] = 1
        return np.asarray(proposals, dtype=float)

    def _apply_neural_action_generator_scores(self, scores: np.ndarray, candidate_actions: np.ndarray, generator: dict[str, object]) -> np.ndarray:
        out = np.asarray(scores, dtype=float).reshape(-1).copy()
        actions = np.asarray(candidate_actions, dtype=float)
        if out.size == 0 or actions.ndim != 2 or actions.shape[0] != out.size:
            return out
        size = actions.shape[1]
        replacement = np.asarray(generator.get("replacement", np.zeros(size)), dtype=float).reshape(-1)
        success = np.asarray(generator.get("success", np.zeros(size)), dtype=float).reshape(-1)
        if replacement.size == size and np.any(np.abs(replacement) > 1e-9):
            out += 0.22 * np.maximum(0.0, actions @ np.clip(replacement, -1.0, 1.0))
        if success.size == size and np.any(np.abs(success) > 1e-9):
            out += 0.16 * np.maximum(0.0, actions @ np.clip(success, -1.0, 1.0))
        evidence_start = int(generator.get("entry_evidence_start", -1) or -1)
        if bool(generator.get("verified_entry_evidence", False)) and 0 <= evidence_start < out.size:
            # A bounded tie-breaker for an already verified evidence proposal;
            # it cannot override legality masks or create a proposal by itself.
            out[evidence_start] += 0.12 * float(np.clip(generator.get("opportunity", 0.0), 0.0, 1.0))
        return np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0)

    def _apply_neural_action_generator_output(self, output: np.ndarray, generator: dict[str, object], probabilities: np.ndarray) -> np.ndarray:
        result = np.asarray(output, dtype=float).copy()
        size = result.size
        replacement = np.asarray(generator.get("replacement", np.zeros(size)), dtype=float).reshape(-1)
        success = np.asarray(generator.get("success", np.zeros(size)), dtype=float).reshape(-1)
        confidence = float(np.max(probabilities)) if np.asarray(probabilities).size else 0.0
        trust = float(generator.get("trust", 0.0))
        replacement_gain = float(np.clip(0.10 + 0.25 * trust, 0.0, 0.35))
        success_gain = float(np.clip(0.08 + 0.18 * trust, 0.0, 0.28))
        if replacement.size == size and np.any(np.abs(replacement) > 1e-9):
            result = np.tanh(result + replacement_gain * np.maximum(0.0, np.clip(replacement, -1.0, 1.0)))
        if success.size == size and np.any(np.abs(success) > 1e-9):
            result = np.tanh(result + success_gain * np.clip(success, -1.0, 1.0))
        # No curiosity/probe delta is added to the live action vector.  Curiosity
        # is a shadow/replay mechanism and is retained only as diagnostics.
        self.last_action_generator_summary = {
            "replacement_strength": float(np.mean(np.abs(replacement))) if replacement.size else 0.0,
            "success_strength": float(np.mean(np.abs(success))) if success.size else 0.0,
            "probe_strength": 0.0,
            "probe_gain": 0.0,
            "market_opportunity": float(generator.get("opportunity", 0.0)),
            "entry_evidence_verified": float(bool(generator.get("verified_entry_evidence", False))),
            "trust": trust,
            "confidence_before_generator": confidence,
            "negative_action_local_inhibition": 1.0,
            "replacement_learning": 1.0,
            "curiosity_live_action_delta": 0.0,
        }
        return result

    def copy(self) -> "M1GraphModule":
        copied = M1GraphModule(self.graph.copy(), state_size=self.state_size, K=self.K, activation_steps=self.activation_steps)
        copied.memory_bank = [item.copy() for item in self.memory_bank]
        copied.memory_bank_max = self.memory_bank_max
        copied.prev_output = self.prev_output.copy() if self.prev_output is not None else None
        copied.prev_predicted_state = self.prev_predicted_state.copy() if self.prev_predicted_state is not None else None
        copied.last_trace = self.last_trace
        copied.last_context_summary = dict(self.last_context_summary)
        copied.last_candidate_actions = self.last_candidate_actions.copy()
        copied.last_candidate_scores = self.last_candidate_scores.copy()
        copied.last_candidate_logits = self.last_candidate_logits.copy()
        copied.last_selected_idx = int(self.last_selected_idx)
        copied.last_selected_log_prob = float(self.last_selected_log_prob)
        copied.last_predicted_state = self.last_predicted_state.copy()
        copied.last_critic_value = float(self.last_critic_value)
        copied.last_reasoning_score_bias = np.asarray(getattr(self, "last_reasoning_score_bias", np.zeros(max(1, self.K))), dtype=float).copy()
        copied.last_action_generator_summary = dict(getattr(self, "last_action_generator_summary", {}) or {})
        copied.last_thinking_output = dict(getattr(self, "last_thinking_output", {}) or {})
        return copied

    def reset_state(self) -> None:
        self.graph.reset_state()
        self.prev_output = None
        self.prev_predicted_state = None
        self.last_trace = None
        self.last_context_summary = {}
        self.last_action_generator_summary = {}
        self.last_thinking_output = {}
        self.last_candidate_actions = np.zeros((max(1, self.K), self.output_size), dtype=float)
        self.last_candidate_scores = np.zeros(max(1, self.K), dtype=float)
        self.last_candidate_logits = np.zeros(max(1, self.K), dtype=float)
        self.last_selected_idx = 0
        self.last_selected_log_prob = 0.0
        self.last_predicted_state = np.zeros(max(1, self.state_size), dtype=float)
        self.last_critic_value = 0.0
        self.last_reasoning_score_bias = np.zeros(max(1, self.K), dtype=float)

    def resize_hidden(self, hidden_size: int, rng: np.random.Generator | None = None) -> None:
        del rng
        # Capacity changes are structural candidate actions.  This compatibility
        # setter never silently changes graph topology on a baseline policy.
        self.state_size = max(1, int(hidden_size))

    def resize_input(self, input_size: int) -> None:
        target = max(1, int(input_size))
        current = self.input_size
        if target <= current:
            return
        self._ensure_neuron_budget(target - current)
        for idx in range(current, target):
            self.graph.add_neuron(REGION_INPUT, neuron_id=f"m1:input:{idx}", module_id="m1", lifecycle="active")

    def resize_output(self, output_size: int) -> None:
        target = max(1, int(output_size))
        current = self.output_size
        if target <= current:
            return
        self._ensure_neuron_budget(target - current)
        for idx in range(current, target):
            self.graph.add_neuron(REGION_OUTPUT, neuron_id=f"m1:output:{idx}", module_id="m1", lifecycle="active")

    def _ensure_neuron_budget(self, extra: int) -> None:
        needed = len(self.graph.neurons) + max(0, int(extra))
        if self.graph.budget.max_neurons <= 0 or needed <= self.graph.budget.max_neurons:
            return
        # A configured positive budget is a hard V7 candidate limit.  Do not
        # silently double it during a schema/topology resize.
        old = self.graph.budget
        if old.max_neurons > 0 and needed > old.max_neurons:
            raise ValueError("M1 graph neuron budget exceeded")
        self.graph.budget = NeuralGraphBudget(
            max_neurons=max(needed, old.max_neurons * 2),
            max_connections=old.max_connections,
            max_recurrent_edges=old.max_recurrent_edges,
            max_lateral_edges=old.max_lateral_edges,
            max_cross_module_edges=old.max_cross_module_edges,
            max_m1_module_edges=old.max_m1_module_edges,
            max_cross_module_recurrent_edges=old.max_cross_module_recurrent_edges,
            max_cross_module_lateral_edges=old.max_cross_module_lateral_edges,
        )

    def _novelty(self, obs: np.ndarray) -> float:
        if not self.memory_bank:
            return 1.0
        distances = [float(np.linalg.norm(obs - item)) for item in self.memory_bank if item.shape == obs.shape]
        if not distances:
            return 1.0
        denom = float(np.linalg.norm(obs)) + 1e-8
        return float(np.clip(min(distances) / denom, 0.0, 2.0))


def _project_blocks(size: int, *blocks: tuple[object, float]) -> np.ndarray:
    """Project semantically separate context blocks without dropping a block."""
    target = max(0, int(size))
    result = np.zeros(target, dtype=float)
    usable = [(np.asarray(_vector(values), dtype=float), max(0.0, float(weight))) for values, weight in blocks]
    usable = [(values, weight) for values, weight in usable if values.size and weight > 0.0]
    if target <= 0 or not usable:
        return result
    weights = np.asarray([weight for _, weight in usable], dtype=float)
    widths = np.maximum(1, np.floor(target * weights / max(float(weights.sum()), 1e-12)).astype(int))
    while int(widths.sum()) > target:
        index = int(np.argmax(widths))
        if widths[index] <= 1:
            break
        widths[index] -= 1
    while int(widths.sum()) < target:
        widths[int(np.argmax(weights))] += 1
    cursor = 0
    for (values, weight), width in zip(usable, widths):
        end = min(target, cursor + int(width))
        if end > cursor:
            result[cursor:end] = _pool(values, end - cursor) * float(np.clip(weight, 0.0, 2.0))
        cursor = end
    return np.clip(result, -2.0, 2.0)


def _context_to_regions(context: object | None, graph: NeuralGraph) -> dict[str, np.ndarray]:
    if not isinstance(context, Mapping):
        return {}
    memory = _vector(context.get("memory_context"))
    action_bias = _vector(context.get("action_bias"))
    critic_bias = _vector(context.get("critic_bias"))
    market = _vector(context.get("market_context"))
    portfolio = _vector(context.get("portfolio_context"))
    position = _vector(context.get("position_context"))
    execution = _vector(context.get("execution_context"))
    temporal = _vector(context.get("temporal_short"))
    temporal_delta = _vector(context.get("temporal_delta"))
    goal = _vector(context.get("goal_vector"))
    learning = _vector(context.get("learning_drive"))
    expectancy_critic = _vector(context.get("expectancy_critic_bias"))
    expectancy_lcb = _vector(context.get("expectancy_action_lcb"))
    market_memory = _vector(context.get("market_memory_context"))
    position_memory = _vector(context.get("position_lifecycle_memory_context"))
    execution_memory = _vector(context.get("execution_memory_context"))
    world_return = np.asarray([
        _scalar(context.get("world_model_return_lcb_1", 0.0)),
        _scalar(context.get("world_model_return_lcb_5", 0.0)),
        _scalar(context.get("world_model_volatility", 0.0)),
        _scalar(context.get("world_model_spread_atr", 0.0)),
        -abs(_scalar(context.get("world_model_uncertainty", 0.0))),
    ], dtype=float)
    trust = float(np.clip(max(
        _scalar(context.get("trust", 0.0)),
        _scalar(context.get("match_strength", 0.0)),
    ), 0.0, 1.0))
    memory_size = len(graph.regions[REGION_MEMORY].neuron_ids)
    predictor_size = len(graph.regions[REGION_PREDICTOR].neuron_ids)
    critic_size = len(graph.regions[REGION_CRITIC].neuron_ids)
    selector_size = len(graph.regions[REGION_SELECTOR].neuron_ids)
    # Every region receives a deliberate projection of relevant world context;
    # account/position/execution are no longer merely post-decision gates.
    return {
        REGION_MEMORY: _project_blocks(
            memory_size,
            (memory * (0.10 + 0.20 * trust), 1.00),
            (market_memory, 0.55), (position_memory, 0.35), (execution_memory, 0.30),
            (market, 0.45), (portfolio, 0.30), (position, 0.25),
            (temporal, 0.35), (temporal_delta, 0.20),
        ),
        REGION_PREDICTOR: _project_blocks(
            predictor_size,
            (market, 0.50), (position, 0.30), (execution, 0.35),
            (market_memory, 0.35), (position_memory, 0.25), (execution_memory, 0.25),
            (world_return, 0.30), (temporal, 0.25), (goal, 0.25), (memory * trust, 0.20),
        ),
        REGION_CRITIC: _project_blocks(
            critic_size,
            (market, 0.35), (portfolio, 0.45), (position, 0.35),
            (execution, 0.40), (goal, 0.40), (learning, 0.10),
            (position_memory, 0.25), (execution_memory, 0.35), (world_return, 0.45),
            (memory * trust, 0.25), (critic_bias, 0.30),
            (expectancy_critic, 0.45), (expectancy_lcb, 0.25),
        ),
        REGION_SELECTOR: _project_blocks(
            selector_size,
            (action_bias, 0.35), (critic_bias, 0.30), (goal, 0.45),
            (portfolio, 0.25), (execution, 0.30), (learning, 0.10),
            (np.maximum(0.0, expectancy_lcb), 0.35),
        ),
    }


def _pool(values: np.ndarray, size: int) -> np.ndarray:
    source = np.asarray(values, dtype=float).reshape(-1)
    result = np.zeros(max(0, int(size)), dtype=float)
    if result.size == 0 or source.size == 0:
        return result
    boundaries = np.linspace(0, source.size, result.size + 1)
    for index in range(result.size):
        start = int(np.floor(boundaries[index]))
        stop = min(source.size, max(start + 1, int(np.ceil(boundaries[index + 1]))))
        if start < stop:
            result[index] = float(np.mean(source[start:stop]))
    return np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=-1.0)


def _allowed_action_mask(context: object | None, size: int) -> np.ndarray:
    count = max(1, min(int(size), 12))
    result = np.ones(count, dtype=bool)
    if not isinstance(context, Mapping) or context.get("allowed_action_mask") is None:
        return result
    raw = np.asarray(context.get("allowed_action_mask"), dtype=bool).reshape(-1)
    if raw.size:
        result[:] = False
        result[: min(result.size, raw.size)] = raw[:result.size]
    if not np.any(result):
        result[0] = True
    return result


def _mask_invalid_action_logits(output: np.ndarray, context: object | None) -> np.ndarray:
    result = np.asarray(output, dtype=float).reshape(-1).copy()
    count = min(result.size, 12)
    if count <= 0:
        return result
    allowed = _allowed_action_mask(context, count)
    legal_values = result[:count][allowed]
    floor = float(np.min(legal_values) - 2.0) if legal_values.size else -2.0
    for index in range(count):
        if not bool(allowed[index]):
            result[index] = floor
    return result


def _neural_action_generator_context(context: object | None, size: int) -> dict[str, object]:
    """Extract only evidence-backed live action proposals.

    Missing data, uncertainty and legacy opportunity fields remain shadow
    diagnostics.  They are never reinterpreted as an instruction to enter a
    trade.  A direct entry proposal is possible only after the contextual
    expectancy ledger verifies LCB, support and confidence for one exact action.
    """
    target = max(1, int(size))
    replacement = np.zeros(target, dtype=float)
    success = np.zeros(target, dtype=float)
    trust = 0.0
    uncertainty = 0.0
    opportunity = 0.0
    learning_drive = 0.0
    replacement_action = 0
    verified_entry_evidence = False
    verified_entry_action = 0
    if isinstance(context, Mapping):
        trust = float(np.clip(max(_scalar(context.get("trust", 0.0)), _scalar(context.get("match_strength", 0.0))), 0.0, 1.0))
        uncertainty = float(np.clip(_scalar(context.get("uncertainty", 0.0)), 0.0, 1.0))
        drive_vec = _vector(context.get("learning_drive"))
        learning_drive = float(np.clip(np.mean(drive_vec) if drive_vec.size else _scalar(context.get("runtime_starvation", 0.0)) / 3.0, 0.0, 1.0))
        for key, dst, scale in (
            ("replacement_action_bias", replacement, 1.0),
            ("replacement_bias", replacement, 1.0),
            ("action_bias", replacement, 0.65),
            ("success_action_bias", success, 1.0),
            ("success_bias", success, 1.0),
        ):
            vec = _vector(context.get(key))
            if vec.size == target:
                dst += scale * np.nan_to_num(vec, nan=0.0, posinf=1.0, neginf=-1.0)
        try:
            replacement_action = int(context.get("replacement_action", 0) or 0)
        except (TypeError, ValueError):
            replacement_action = 0
        verified_entry_evidence = bool(context.get("expectancy_entry_evidence_verified", False))
        try:
            verified_entry_action = int(context.get("expectancy_verified_entry_index", 0) or 0)
        except (TypeError, ValueError):
            verified_entry_action = 0
        if verified_entry_evidence and 1 <= verified_entry_action <= min(6, target - 1):
            # Trust only named, verified contextual expectancy evidence.
            opportunity = float(np.clip(_scalar(context.get("expectancy_market_opportunity", 0.0)), 0.0, 1.0))
        else:
            verified_entry_evidence = False
            verified_entry_action = 0
    return {
        "replacement": np.clip(replacement, -1.0, 1.0),
        "success": np.clip(success, -1.0, 1.0),
        "probe": np.zeros(target, dtype=float),
        "probe_gain": 0.0,
        "opportunity": opportunity,
        "learning_drive": learning_drive,
        "uncertainty": uncertainty,
        "trust": trust,
        "replacement_action": replacement_action,
        "verified_entry_evidence": bool(verified_entry_evidence),
        "verified_entry_action": int(verified_entry_action),
        "shadow_replay_requested": bool(
            _scalar(context.get("world_model_shadow_replay_requested", 0.0)) > 0.0
            or uncertainty > 0.55
        ) if isinstance(context, Mapping) else False,
    }

def _entry_gate(context: object | None) -> float:
    if not isinstance(context, Mapping):
        return 1.0
    explicit = context.get("entry_gate")
    if explicit is not None:
        return float(np.clip(_scalar(explicit), 0.05, 1.0))
    stress = max(
        _scalar(context.get("account_stress", 0.0)),
        _scalar(context.get("drawdown", 0.0)),
        _scalar(context.get("open_risk_ratio", 0.0)),
    )
    return float(np.clip(1.0 - 0.90 * np.clip(stress, 0.0, 1.0), 0.05, 1.0))


def _extract_action_bias(context: object | None, size: int) -> np.ndarray | None:
    if not isinstance(context, Mapping):
        return None
    raw = _vector(context.get("action_bias"))
    if raw.size != size:
        return None
    return raw


def _mean_abs(values: object) -> float:
    vector = _vector(values)
    return float(np.mean(np.abs(vector))) if vector.size else 0.0


def _vector(value: object) -> np.ndarray:
    """Return a finite 1-D vector; absent optional context is empty, not NaN."""
    if value is None:
        return np.zeros(0, dtype=float)
    try:
        vector = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return np.zeros(0, dtype=float)
    return np.nan_to_num(vector, nan=0.0, posinf=1.0, neginf=-1.0)


def _scalar(value: object) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return 0.0
    return scalar if np.isfinite(scalar) else 0.0


def _resize(values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros(max(0, int(size)), dtype=float)
    if values.size:
        result[: min(result.size, values.size)] = np.nan_to_num(values[:result.size], nan=0.0, posinf=1.0, neginf=-1.0)
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    shifted = arr - float(np.max(arr))
    exp = np.exp(np.clip(shifted, -40.0, 40.0))
    return exp / (float(np.sum(exp)) + 1e-12)


def _conflict_from_probabilities(probabilities: np.ndarray) -> float:
    probs = np.sort(np.asarray(probabilities, dtype=float).reshape(-1))[::-1]
    if probs.size < 2:
        return 0.0
    return float(np.clip(1.0 - (probs[0] - probs[1]), 0.0, 1.0))


def _confidence(output: np.ndarray) -> float:
    if output.size == 0:
        return 0.0
    logits = output - np.max(output)
    exp = np.exp(logits)
    probs = exp / (np.sum(exp) + 1e-10)
    return float(np.max(probs))


def _conflict(output: np.ndarray) -> float:
    if output.size < 2:
        return 0.0
    logits = output - np.max(output)
    exp = np.exp(logits)
    probs = np.sort(exp / (np.sum(exp) + 1e-10))[::-1]
    return float(1.0 - (probs[0] - probs[1]))
