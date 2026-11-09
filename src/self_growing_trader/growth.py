from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .chuyen_gia_policy import ModuleState


MODULE_LIFECYCLE_DORMANT = "dormant"
MODULE_LIFECYCLE_CANDIDATE = "candidate"
MODULE_LIFECYCLE_ACTIVE = "active"
MODULE_LIFECYCLE_CONSOLIDATED = "consolidated"
MODULE_LIFECYCLE_PRUNING = "pruning"

VALID_LIFECYCLES = (
    MODULE_LIFECYCLE_DORMANT,
    MODULE_LIFECYCLE_CANDIDATE,
    MODULE_LIFECYCLE_ACTIVE,
    MODULE_LIFECYCLE_CONSOLIDATED,
    MODULE_LIFECYCLE_PRUNING,
)


@dataclass
class ErrorCluster:
    indices: list[int]
    mean_error: float
    mean_novelty: float
    mean_conflict: float
    centroid: np.ndarray | None = None


@dataclass
class ShadowBranch:
    original_name: str
    shadow_name: str
    validation_score: float = 0.0
    ood_score: float = 0.0
    accepted: bool = False


@dataclass
class ModuleSpec:
    role: str
    input_sources: list[str] = field(default_factory=list)
    output_targets: list[str] = field(default_factory=list)
    initial_gate_bias: float = -1.5
    initial_plasticity: float = 1.5


ROLE_PREDICTOR_SPECIALIST = "predictor_specialist"
ROLE_MEMORY_SPECIALIST = "memory_specialist"
ROLE_CRITIC_SPECIALIST = "critic_specialist"
ROLE_HYPOTHESIS_SPECIALIST = "hypothesis_specialist"
ROLE_FEATURE_SPECIALIST = "feature_specialist"
ROLE_GENERAL_SPECIALIST = "general_specialist"


def diagnose_cluster_to_string(cluster: ErrorCluster) -> str:
    if cluster.mean_error > 0.7:
        return "prediction_error_high"
    elif cluster.mean_novelty > 0.7:
        return "novelty_high"
    elif cluster.mean_conflict > 0.5:
        return "ambiguous_actions"
    else:
        return "capacity_insufficient"


@dataclass
class GrowthCoefficients:
    a_error: float = 1.0
    b_novelty: float = 0.8
    c_conflict: float = 0.6
    d_representation: float = 0.5
    e_complexity: float = 0.3


def cosine_gradient_similarity(grad_a: np.ndarray, grad_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(grad_a))
    norm_b = float(np.linalg.norm(grad_b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(grad_a.flatten(), grad_b.flatten()) / (norm_a * norm_b))


def should_split_module(
    grad_a: np.ndarray,
    grad_b: np.ndarray,
    tau_split: float = -0.2,
) -> bool:
    return cosine_gradient_similarity(grad_a, grad_b) < tau_split


def compute_module_similarity(
    gate_a: np.ndarray,
    gate_b: np.ndarray,
    hidden_a: np.ndarray,
    hidden_b: np.ndarray,
    output_a: np.ndarray,
    output_b: np.ndarray,
    lambda_g: float = 0.3,
    lambda_h: float = 0.4,
    lambda_y: float = 0.3,
) -> float:
    def _cosine(x: np.ndarray, y: np.ndarray) -> float:
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx < 1e-10 or ny < 1e-10:
            return 0.0
        return float(np.dot(x.flatten(), y.flatten()) / (nx * ny))

    return (
        lambda_g * _cosine(gate_a, gate_b)
        + lambda_h * _cosine(hidden_a, hidden_b)
        + lambda_y * _cosine(output_a, output_b)
    )


def compute_utility(
    gate_ema: float,
    delta_loss: float,
    info_gain: float = 0.0,
    cost: float = 0.0,
    lambda_i: float = 0.3,
    lambda_c: float = 0.1,
) -> float:
    return gate_ema * delta_loss + lambda_i * info_gain - lambda_c * cost


def estimate_utility_by_ablation(
    baseline_loss: float,
    ablated_loss: float,
    gate_ema: float,
) -> float:
    delta_loss = ablated_loss - baseline_loss
    return gate_ema * delta_loss


def error_clustering(
    errors: np.ndarray,
    novelties: np.ndarray,
    latent_states: np.ndarray | None,
    action_conflicts: np.ndarray | None,
    n_clusters: int = 3,
    min_cluster_size: int = 5,
) -> list[ErrorCluster]:
    n = len(errors)
    if n < min_cluster_size:
        return []

    features_list = [errors.reshape(-1, 1), novelties.reshape(-1, 1)]
    if latent_states is not None:
        if latent_states.ndim == 1:
            latent_states = latent_states.reshape(-1, 1)
        features_list.append(latent_states)
    if action_conflicts is not None:
        features_list.append(action_conflicts.reshape(-1, 1))

    features = np.hstack(features_list)
    features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-10)

    actual_k = min(n_clusters, n // min_cluster_size)
    if actual_k < 2:
        high_error_mask = errors > np.percentile(errors, 75)
        indices = np.where(high_error_mask)[0].tolist()
        if len(indices) >= min_cluster_size:
            return [ErrorCluster(
                indices=indices,
                mean_error=float(np.mean(errors[indices])),
                mean_novelty=float(np.mean(novelties[indices])),
                mean_conflict=float(np.mean(action_conflicts[indices])) if action_conflicts is not None else 0.0,
                centroid=features[indices].mean(axis=0),
            )]
        return []

    rng = np.random.default_rng(42)
    centroids = features[rng.choice(n, actual_k, replace=False)]

    for _ in range(10):
        distances = np.linalg.norm(features[:, None] - centroids[None, :], axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([
            features[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else centroids[k]
            for k in range(actual_k)
        ])
        if np.allclose(centroids, new_centroids, atol=1e-6):
            break
        centroids = new_centroids

    clusters = []
    for k in range(actual_k):
        indices = np.where(labels == k)[0].tolist()
        if len(indices) >= min_cluster_size:
            clusters.append(ErrorCluster(
                indices=indices,
                mean_error=float(np.mean(errors[indices])),
                mean_novelty=float(np.mean(novelties[indices])),
                mean_conflict=float(np.mean(action_conflicts[indices])) if action_conflicts is not None else 0.0,
                centroid=centroids[k],
            ))

    return sorted(clusters, key=lambda c: c.mean_error, reverse=True)


def diagnose_cluster(
    cluster: ErrorCluster,
    global_loss: float,
    modules: list,
    tau_module: float = 0.1,
) -> str:
    cluster_loss = cluster.mean_error
    if cluster_loss - global_loss > tau_module:
        return "new_subtask"
    if cluster.mean_conflict > 0.5:
        return "gradient_conflict"
    if cluster.mean_novelty > 0.7:
        return "missing_path"
    return "capacity"


class GrowthController:
    def __init__(
        self,
        tau_grow: float = 0.3,
        tau_prune: float = 0.05,
        tau_module: float = 0.15,
        tau_split: float = -0.2,
        tau_merge: float = 0.8,
        H: int = 3,
        K_max: int = 8,
        N_max: int = 128,
        n_min: int = 5,
        coeffs: GrowthCoefficients | None = None,
    ) -> None:
        self.tau_grow = tau_grow
        self.tau_prune = tau_prune
        self.tau_module = tau_module
        self.tau_split = tau_split
        self.tau_merge = tau_merge
        self.H = H
        self.K_max = K_max
        self.N_max = N_max
        self.n_min = n_min
        self.coeffs = coeffs or GrowthCoefficients()
        self._growth_window: list[float] = []

    def compute_growth_signal(
        self,
        error: float,
        novelty: float,
        conflict: float,
        representation_need: float,
        complexity: float,
    ) -> float:
        c = self.coeffs
        return (
            c.a_error * error
            + c.b_novelty * novelty
            + c.c_conflict * conflict
            + c.d_representation * representation_need
            - c.e_complexity * complexity
        )

    def sustained_growth_check(self, signal: float) -> bool:
        self._growth_window.append(signal)
        if len(self._growth_window) > self.H:
            self._growth_window = self._growth_window[-self.H:]
        if len(self._growth_window) < self.H:
            return False
        return all(s > self.tau_grow for s in self._growth_window)

    def update_thresholds(self, recent_improvement: float) -> None:
        if recent_improvement > 0.1:
            self.tau_grow = min(0.8, self.tau_grow * 1.05)
            self.tau_prune = max(0.01, self.tau_prune * 0.95)
        elif recent_improvement < -0.05:
            self.tau_grow = max(0.1, self.tau_grow * 0.95)
            self.tau_prune = min(0.2, self.tau_prune * 1.05)

    def check_budget(self, n_modules: int, n_neurons: int) -> bool:
        return n_modules < self.K_max and n_neurons < self.N_max

    def validate_change(
        self,
        baseline_val_loss: float,
        new_val_loss: float,
        baseline_ood_loss: float | None = None,
        new_ood_loss: float | None = None,
        delta_keep: float = 0.01,
        epsilon_forget: float = 0.05,
    ) -> tuple[bool, list[str]]:
        reasons = []
        val_improved = (baseline_val_loss - new_val_loss) > delta_keep
        if not val_improved:
            reasons.append("validation_not_improved")
        if baseline_ood_loss is not None and new_ood_loss is not None:
            ood_ok = new_ood_loss <= baseline_ood_loss + epsilon_forget
            if not ood_ok:
                reasons.append("ood_degraded")
        return val_improved and len(reasons) == 0, reasons

    def spawn_module_shadow(
        self,
        parent_module: "ModuleState",
        rng: np.random.Generator,
        mutation_sigma: float,
        name_suffix: str = "_shadow",
    ) -> tuple["ModuleState", ShadowBranch]:
        shadow = parent_module.copy()
        shadow.name = parent_module.name + name_suffix
        shadow.role = "candidate"
        shadow.lifecycle = MODULE_LIFECYCLE_CANDIDATE
        shadow.gate_b -= 1.5
        shadow.plasticity = min(2.5, shadow.plasticity * 1.5)
        shadow.utility = 0.0
        shadow.gate_ema = 0.0
        shadow.age = 0
        shadow.mutate(rng, mutation_sigma * 2.0)
        branch = ShadowBranch(
            original_name=parent_module.name,
            shadow_name=shadow.name,
        )
        return shadow, branch

    def finalize_shadow(
        self,
        branch: ShadowBranch,
        val_score: float,
        ood_score: float | None = None,
        delta_keep: float = 0.01,
    ) -> ShadowBranch:
        branch.validation_score = val_score
        if ood_score is not None:
            branch.ood_score = ood_score
        branch.accepted = val_score > delta_keep
        return branch


@dataclass
class DevelopmentalOutput:
    delta_alpha: float = 0.0
    eta_plasticity: float = 1.0
    recruit_logit: float = 0.0
    prune_logit: float = 0.0
    consolidate_logit: float = 0.0
    memory_reset_logit: float = 0.0


class DevelopmentalControllerRNN:
    __slots__ = ("w_in", "w_rec", "b_hidden", "w_out", "b_out",
                 "hidden", "input_size", "hidden_size", "output_size",
                 "lr", "eligibility_decay", "eligibility")

    def __init__(self, input_size: int = 12, hidden_size: int = 16, output_size: int = 6, rng: np.random.Generator | None = None) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
        self.w_in = rng.normal(0.0, scale, size=(hidden_size, input_size))
        self.w_rec = rng.normal(0.0, 0.05, size=(hidden_size, hidden_size))
        self.b_hidden = np.zeros(hidden_size, dtype=float)
        self.w_out = rng.normal(0.0, 0.1, size=(output_size, hidden_size))
        self.b_out = np.zeros(output_size, dtype=float)
        self.hidden = np.zeros(hidden_size, dtype=float)
        self.lr = 0.001
        self.eligibility_decay = 0.95
        self.eligibility = np.zeros((output_size, hidden_size), dtype=float)

    def forward(self, q: np.ndarray) -> DevelopmentalOutput:
        q_padded = np.zeros(self.input_size, dtype=float)
        q_padded[: min(q.size, self.input_size)] = q[: self.input_size]
        self.hidden = np.tanh(self.w_in @ q_padded + self.w_rec @ self.hidden + self.b_hidden)
        raw = self.w_out @ self.hidden + self.b_out
        self.eligibility = self.eligibility_decay * self.eligibility + np.outer(
            np.ones(self.output_size), self.hidden
        )
        return DevelopmentalOutput(
            delta_alpha=float(np.clip(raw[0], -2.0, 2.0)),
            eta_plasticity=float(np.clip(0.5 + 0.5 * np.tanh(raw[1]), 0.01, 3.0)),
            recruit_logit=float(raw[2]),
            prune_logit=float(raw[3]),
            consolidate_logit=float(raw[4]),
            memory_reset_logit=float(raw[5]),
        )

    def update(self, meta_loss_grad: np.ndarray) -> None:
        grad = np.clip(meta_loss_grad, -1.0, 1.0)
        self.w_out -= self.lr * np.outer(grad, self.hidden)
        self.b_out -= self.lr * grad

    def compute_q_vector(
        self,
        gate_ema: float,
        activation_ema: float,
        prediction_error: float,
        td_delta: float,
        novelty: float,
        conflict: float,
        uncertainty: float,
        age: int,
        utility: float,
        plasticity: float,
        n_modules: int,
        k_max: int,
    ) -> np.ndarray:
        return np.array([
            gate_ema,
            activation_ema,
            np.tanh(prediction_error),
            np.tanh(td_delta),
            novelty,
            conflict,
            uncertainty,
            min(age / 100.0, 1.0),
            np.tanh(utility),
            plasticity / 2.5,
            n_modules / max(k_max, 1),
            0.0,
        ], dtype=float)

    def copy(self) -> "DevelopmentalControllerRNN":
        dc = DevelopmentalControllerRNN.__new__(DevelopmentalControllerRNN)
        dc.input_size = self.input_size
        dc.hidden_size = self.hidden_size
        dc.output_size = self.output_size
        dc.w_in = self.w_in.copy()
        dc.w_rec = self.w_rec.copy()
        dc.b_hidden = self.b_hidden.copy()
        dc.w_out = self.w_out.copy()
        dc.b_out = self.b_out.copy()
        dc.hidden = self.hidden.copy()
        dc.lr = float(self.lr)
        dc.eligibility_decay = float(self.eligibility_decay)
        dc.eligibility = self.eligibility.copy()
        return dc

    def mutate(self, rng: np.random.Generator, sigma: float) -> None:
        self.w_in += rng.normal(0.0, sigma, size=self.w_in.shape)
        self.w_rec += rng.normal(0.0, sigma * 0.3, size=self.w_rec.shape)
        self.w_out += rng.normal(0.0, sigma, size=self.w_out.shape)

    def score_edges(
        self,
        edge_q_vectors: np.ndarray,
    ) -> np.ndarray:
        n_edges = edge_q_vectors.shape[0]
        scores = np.zeros(n_edges, dtype=float)
        saved_hidden = self.hidden.copy()
        for k in range(n_edges):
            q = edge_q_vectors[k]
            q_padded = np.zeros(self.input_size, dtype=float)
            q_padded[: min(q.size, self.input_size)] = q[: self.input_size]
            h = np.tanh(self.w_in @ q_padded + self.w_rec @ saved_hidden + self.b_hidden)
            raw = self.w_out @ h + self.b_out
            scores[k] = float(raw[0])
        self.hidden = saved_hidden
        return scores

    def score_neurons(
        self,
        neuron_q_vectors: np.ndarray,
    ) -> np.ndarray:
        return self.score_edges(neuron_q_vectors)

    @staticmethod
    def build_edge_q_vector(
        pre_act: float,
        post_act: float,
        eligibility: float,
        local_error: float,
        current_alpha: float,
        current_g: float,
        weight_mag: float,
        gate_usage_src: float,
        gate_usage_dst: float,
    ) -> np.ndarray:
        return np.array([
            np.tanh(pre_act),
            np.tanh(post_act),
            np.tanh(eligibility),
            np.tanh(local_error),
            np.tanh(current_alpha),
            current_g,
            np.tanh(weight_mag),
            gate_usage_src,
            gate_usage_dst,
            0.0, 0.0, 0.0,
        ], dtype=float)

    @staticmethod
    def build_neuron_q_vector(
        activation_ema: float,
        gate_ema: float,
        age: float,
        utility: float,
        plasticity: float,
        is_dormant: float,
    ) -> np.ndarray:
        return np.array([
            np.tanh(activation_ema),
            gate_ema,
            min(age / 100.0, 1.0),
            np.tanh(utility),
            plasticity / 2.5,
            is_dormant,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ], dtype=float)
