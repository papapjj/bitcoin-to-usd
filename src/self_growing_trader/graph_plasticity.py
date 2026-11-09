from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .neural_graph import (
    EDGE_CRITIC,
    EDGE_GATING,
    EDGE_MEMORY,
    EDGE_META,
    EDGE_PLASTIC,
    EDGE_PREDICTIVE,
    EDGE_RECURRENT,
    EDGE_SELECTOR,
    NEURON_LIFECYCLE_CANDIDATE,
    NEURON_LIFECYCLE_CONSOLIDATED,
    GraphActivationTrace,
    NeuralGraph,
)
from .plasticity import (
    PlasticityCoefficients,
    anti_hebbian_update,
    gradient_plasticity,
    hebbian_update,
    homeostatic_update_bias,
)


_LIFECYCLE_PENALTIES = {
    "near_tp_missed": 0.45,
    "gave_back_profit": 0.55,
    "near_tp_giveback": 0.55,
    "closed_after_near_tp_bad": 0.40,
    "should_have_partial_or_trail": 0.35,
    "stress_entry_loss": 0.35,
}

_EDGE_KIND_WEIGHT_SCALE = {
    EDGE_PREDICTIVE: 1.25,
    EDGE_CRITIC: 0.85,
    EDGE_SELECTOR: 0.95,
    EDGE_GATING: 0.55,
    EDGE_PLASTIC: 1.45,
    EDGE_META: 0.80,
    EDGE_MEMORY: 0.90,
    EDGE_RECURRENT: 0.75,
}

_EDGE_KIND_GATE_SCALE = {
    EDGE_GATING: 1.80,
    EDGE_SELECTOR: 1.25,
    EDGE_CRITIC: 1.15,
    EDGE_META: 1.10,
    EDGE_MEMORY: 0.85,
    EDGE_RECURRENT: 0.75,
}


@dataclass(frozen=True)
class GraphCreditSignal:
    reward: float = 0.0
    cost: float = 0.0
    td_delta: float = 0.0
    prediction_error: float = 0.0
    lifecycle_penalty: float = 0.0
    account_stress: float = 0.0
    lifecycle_tags: tuple[str, ...] = ()
    memory_match_quality: float = 1.0

    @property
    def tagged_lifecycle_penalty(self) -> float:
        return float(sum(_LIFECYCLE_PENALTIES.get(tag, 0.0) for tag in self.lifecycle_tags))

    @property
    def credit(self) -> float:
        return float(
            self.reward
            - self.cost
            + 0.5 * self.td_delta
            - 0.1 * abs(self.prediction_error)
            - self.lifecycle_penalty
            - self.tagged_lifecycle_penalty
            - 0.25 * np.clip(self.account_stress, 0.0, 1.0)
        )


@dataclass
class GraphPlasticityReport:
    credit: float
    updated_edges: int = 0
    updated_neurons: int = 0
    disabled_edges: int = 0
    protected_baseline_edges: int = 0
    lifecycle_penalty: float = 0.0
    active_edge_ids: tuple[str, ...] = ()
    active_neuron_ids: tuple[str, ...] = ()
    update_mode: str = "candidate"
    gradient_abs_sum: float = 0.0
    hebbian_abs_sum: float = 0.0
    anti_hebbian_abs_sum: float = 0.0
    oja_drive_abs_sum: float = 0.0
    oja_stability_abs_sum: float = 0.0
    update_abs_sum: float = 0.0
    notes: list[str] = field(default_factory=list)


class GraphPlasticityEngine:
    """Credit assignment with bounded online plasticity and candidate safety.

    ``candidate_mode=True`` is required for destructive actions such as
    disabling an edge.  A negative credit signal on a baseline records
    attribution but does not directly degrade its production parameters.
    """

    def __init__(
        self,
        *,
        utility_decay: float = 0.95,
        gate_lr: float = 0.02,
        weight_lr: float = 0.001,
        min_utility: float = -1.0,
        gate_clip: float = 8.0,
        weight_clip: float = 10.0,
        short_term_decay: float = 0.92,
        protect_baseline_from_negative_credit: bool = True,
        coeffs: PlasticityCoefficients | None = None,
        homeostatic_target: float = 0.35,
    ) -> None:
        self.utility_decay = float(np.clip(utility_decay, 0.0, 1.0))
        self.gate_lr = float(gate_lr)
        self.weight_lr = float(weight_lr)
        self.min_utility = float(min_utility)
        self.gate_clip = float(max(1.0, gate_clip))
        self.weight_clip = float(max(1.0, weight_clip))
        self.short_term_decay = float(np.clip(short_term_decay, 0.0, 1.0))
        self.protect_baseline_from_negative_credit = bool(protect_baseline_from_negative_credit)
        # Reuse the project-wide plasticity primitives so graph components
        # synapses share the same Hebbian, anti-Hebbian, Oja, gradient and
        # homeostatic vocabulary.
        self.coeffs = coeffs or PlasticityCoefficients()
        self.homeostatic_target = float(np.clip(homeostatic_target, 0.01, 0.95))

    def apply_credit(
        self,
        graph: NeuralGraph,
        signal: GraphCreditSignal,
        *,
        active_edge_ids: Iterable[str] | None = None,
        active_neuron_ids: Iterable[str] | None = None,
        trace: GraphActivationTrace | None = None,
        candidate_mode: bool = True,
        update_mode: str = "candidate",
    ) -> GraphPlasticityReport:
        if update_mode not in {"candidate", "replay", "online", "baseline"}:
            raise ValueError(f"unknown graph plasticity update mode: {update_mode}")
        if trace is not None:
            if active_edge_ids is None:
                active_edge_ids = trace.active_edge_ids
            if active_neuron_ids is None:
                active_neuron_ids = trace.active_neuron_ids
        edge_ids = tuple(dict.fromkeys(active_edge_ids or ()))
        neuron_ids = tuple(dict.fromkeys(active_neuron_ids or ()))
        credit = signal.credit
        report = GraphPlasticityReport(
            credit=credit,
            lifecycle_penalty=signal.lifecycle_penalty + signal.tagged_lifecycle_penalty,
            active_edge_ids=edge_ids,
            active_neuron_ids=neuron_ids,
            update_mode=update_mode,
        )
        stress_dampen = float(np.clip(1.0 - 0.75 * np.clip(signal.account_stress, 0.0, 1.0), 0.10, 1.0))
        baseline_protected = (
            not candidate_mode
            and self.protect_baseline_from_negative_credit
            and credit < 0.0
        )
        for edge_id in edge_ids:
            edge = graph.edges.get(edge_id)
            if edge is None or not edge.enabled:
                continue
            src_activation = float(np.tanh(np.clip(
                np.nan_to_num(graph.neurons[edge.src].activation if edge.src in graph.neurons else 0.0, nan=0.0),
                -20.0,
                20.0,
            )))
            dst_activation = float(np.tanh(np.clip(
                np.nan_to_num(graph.neurons[edge.dst].activation if edge.dst in graph.neurons else 0.0, nan=0.0),
                -20.0,
                20.0,
            )))
            edge.utility = self.utility_decay * edge.utility + (1.0 - self.utility_decay) * credit
            edge.eligibility = self.utility_decay * edge.eligibility + abs(src_activation * dst_activation)
            # A graph-level policy may be active while an attached specialist
            # edge is still a candidate.  Candidate credit must not be blocked
            # merely because the owning M1 module is production-active.
            edge_is_candidate = (
                edge.lifecycle == NEURON_LIFECYCLE_CANDIDATE
                and edge.candidate_id not in {None, "", "graph_seed"}
            )
            edge_protected = baseline_protected and not edge_is_candidate
            if edge_protected:
                report.protected_baseline_edges += 1
                continue
            effective_plasticity = float(np.clip(edge.plasticity * stress_dampen, 0.05, 3.0))
            # Reuse the scalar forms of the common plasticity rules.  The
            # local pre/post values come only from the active trace, avoiding
            # inadvertent credit on inactive edges.
            pre = np.asarray([src_activation], dtype=float)
            post = np.asarray([dst_activation], dtype=float)
            weight = np.asarray([[edge.weight]], dtype=float)
            pseudo_grad = np.asarray([[-credit * src_activation * dst_activation]], dtype=float)
            grad = float(gradient_plasticity(weight, pseudo_grad, self.coeffs.eta_grad)[0, 0])
            hebbian = float(hebbian_update(pre, post, self.coeffs.eta_hebb)[0, 0])
            anti_hebbian = float(anti_hebbian_update(pre, post, self.coeffs.eta_anti_hebb)[0, 0])
            oja_drive = float(self.coeffs.eta_oja * src_activation * dst_activation)
            # The Oja norm correction is always damping.  Keeping it separate
            # prevents negative credit from reversing the stability term.
            oja_stability = float(-self.coeffs.eta_oja * dst_activation * dst_activation * edge.weight)
            kind_credit = credit
            if edge.kind == EDGE_PREDICTIVE:
                kind_credit += 0.25 * signal.td_delta - 0.10 * abs(signal.prediction_error)
            elif edge.kind == EDGE_CRITIC:
                kind_credit -= 0.25 * signal.account_stress + 0.15 * signal.lifecycle_penalty
            elif edge.kind == EDGE_SELECTOR:
                kind_credit += 0.20 * signal.td_delta
            elif edge.kind == EDGE_MEMORY:
                kind_credit += 0.10 * (signal.memory_match_quality - 0.5)
            elif edge.kind == EDGE_META:
                kind_credit -= 0.10 * signal.tagged_lifecycle_penalty
            anti_scale = 1.0 - min(1.0, abs(kind_credit))
            local_delta = (
                kind_credit * hebbian
                + anti_scale * anti_hebbian
                + max(kind_credit, 0.0) * oja_drive
                + oja_stability
                + grad
            )
            report.gradient_abs_sum += abs(grad)
            report.hebbian_abs_sum += abs(kind_credit * hebbian)
            report.anti_hebbian_abs_sum += abs(anti_scale * anti_hebbian)
            report.oja_drive_abs_sum += abs(max(kind_credit, 0.0) * oja_drive)
            report.oja_stability_abs_sum += abs(oja_stability)
            report.update_abs_sum += abs(local_delta)
            gate_scale = _EDGE_KIND_GATE_SCALE.get(edge.kind, 1.0)
            weight_scale = _EDGE_KIND_WEIGHT_SCALE.get(edge.kind, 1.0)
            edge.gate = float(np.clip(
                edge.gate + self.gate_lr * kind_credit * effective_plasticity * gate_scale,
                -self.gate_clip,
                self.gate_clip,
            ))
            edge.weight = float(np.clip(
                edge.weight + self.weight_lr * local_delta * effective_plasticity * weight_scale,
                -self.weight_clip,
                self.weight_clip,
            ))
            if edge.kind in {"recurrent", "memory"}:
                edge.short_term_state = float(np.clip(
                    self.short_term_decay * edge.short_term_state + (1.0 - self.short_term_decay) * hebbian,
                    -1.0,
                    1.0,
                ))
            report.updated_edges += 1
            if (candidate_mode or edge_is_candidate) and edge.utility < self.min_utility:
                graph.disable_connection(edge_id)
                report.disabled_edges += 1
        for neuron_id in neuron_ids:
            neuron = graph.neurons.get(neuron_id)
            if neuron is None:
                continue
            neuron.utility = self.utility_decay * neuron.utility + (1.0 - self.utility_decay) * credit
            neuron.eligibility = self.utility_decay * neuron.eligibility + abs(neuron.activation)
            neuron_is_candidate = (
                neuron.lifecycle == NEURON_LIFECYCLE_CANDIDATE
                and str(getattr(neuron, "module_id", "")) != "m1"
            )
            if baseline_protected and not neuron_is_candidate:
                continue
            if credit > 0:
                neuron.plasticity = float(np.clip(neuron.plasticity * 0.99, 0.05, 3.0))
            else:
                neuron.plasticity = float(np.clip(neuron.plasticity * 1.01, 0.05, 3.0))
            # One-element homeostatic update preserves the utility of the
            # existing shared helper while keeping graph node biases bounded.
            target_bias = homeostatic_update_bias(
                np.asarray([neuron.bias], dtype=float),
                np.asarray([abs(neuron.activation)], dtype=float),
                self.homeostatic_target,
                self.gate_lr * 0.10,
            )[0]
            neuron.bias = float(np.clip(target_bias, -2.0, 2.0))
            report.updated_neurons += 1
        if baseline_protected:
            report.notes.append("negative credit attributed without mutating baseline parameters")
        graph.assert_invariants()
        return report

    def consolidate_high_utility(
        self,
        graph: NeuralGraph,
        *,
        threshold: float = 0.5,
        promotions: int = 1,
    ) -> int:
        """Slow high-utility paths after repeat promotion to reduce forgetting."""
        count = 0
        factor = float(np.clip(0.94 - 0.01 * max(0, promotions - 1), 0.75, 0.94))
        for edge in graph.edges.values():
            if edge.enabled and edge.utility >= threshold:
                edge.plasticity = float(np.clip(edge.plasticity * factor, 0.05, 3.0))
                edge.consolidation = float(np.clip(edge.consolidation + 0.1, 0.0, 1.0))
                edge.lifecycle = NEURON_LIFECYCLE_CONSOLIDATED
                count += 1
        for neuron in graph.neurons.values():
            if neuron.utility >= threshold:
                neuron.plasticity = float(np.clip(neuron.plasticity * factor, 0.05, 3.0))
                neuron.lifecycle = NEURON_LIFECYCLE_CONSOLIDATED
        return count

    def apply_decay_and_prune_pressure(
        self,
        graph: NeuralGraph,
        *,
        candidate_mode: bool,
        utility_threshold: float = -0.25,
    ) -> int:
        """Age low-utility candidates; baseline topology is never pruned here."""
        disabled = 0
        for edge in graph.edges.values():
            edge.utility *= self.utility_decay
            if candidate_mode and edge.enabled and edge.utility <= utility_threshold:
                graph.disable_connection(edge.id)
                disabled += 1
        return disabled
