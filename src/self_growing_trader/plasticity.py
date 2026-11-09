from __future__ import annotations

import numpy as np

from .input_normalization import normalize_input_vector


class PlasticityCoefficients:
    __slots__ = ("eta_grad", "eta_hebb", "eta_anti_hebb", "eta_oja", "lambda_reg")

    def __init__(
        self,
        eta_grad: float = 0.001,
        eta_hebb: float = 0.005,
        eta_anti_hebb: float = 0.002,
        eta_oja: float = 0.001,
        lambda_reg: float = 0.0001,
    ) -> None:
        self.eta_grad = eta_grad
        self.eta_hebb = eta_hebb
        self.eta_anti_hebb = eta_anti_hebb
        self.eta_oja = eta_oja
        self.lambda_reg = lambda_reg


class STDPParams:
    __slots__ = ("A_plus", "A_minus", "tau_plus", "tau_minus")

    def __init__(
        self,
        A_plus: float = 0.01,
        A_minus: float = 0.012,
        tau_plus: float = 20.0,
        tau_minus: float = 20.0,
    ) -> None:
        self.A_plus = A_plus
        self.A_minus = A_minus
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus


class ShortTermPlasticityState:
    __slots__ = ("p", "rho", "alpha")

    def __init__(self, size: int | tuple[int, ...], rho: float = 0.95, alpha: float = 0.1) -> None:
        if isinstance(size, int):
            self.p = np.ones(size, dtype=float)
        else:
            self.p = np.ones(size, dtype=float)
        self.rho = rho
        self.alpha = alpha

    def update(self, pre_activation: np.ndarray) -> np.ndarray:
        self.p = np.clip(self.rho * self.p + self.alpha * pre_activation, 0.01, 5.0)
        return self.p

    def copy(self) -> "ShortTermPlasticityState":
        state = ShortTermPlasticityState.__new__(ShortTermPlasticityState)
        state.p = self.p.copy()
        state.rho = self.rho
        state.alpha = self.alpha
        return state


def gradient_plasticity(
    weights: np.ndarray,
    grad_loss_w: np.ndarray,
    eta_g: float,
) -> np.ndarray:
    return -eta_g * grad_loss_w


def hebbian_update(
    pre_activation: np.ndarray,
    post_activation: np.ndarray,
    eta_h: float,
) -> np.ndarray:
    if pre_activation.ndim == 1 and post_activation.ndim == 1:
        return eta_h * np.outer(post_activation, pre_activation)
    return eta_h * post_activation * pre_activation


def anti_hebbian_update(
    pre_activation: np.ndarray,
    post_activation: np.ndarray,
    eta_a: float,
) -> np.ndarray:
    if pre_activation.ndim == 1 and post_activation.ndim == 1:
        return -eta_a * np.outer(post_activation, pre_activation)
    return -eta_a * post_activation * pre_activation


def oja_update(
    pre_activation: np.ndarray,
    post_activation: np.ndarray,
    weights: np.ndarray,
    eta_o: float,
) -> np.ndarray:
    if pre_activation.ndim == 1 and post_activation.ndim == 1:
        outer = np.outer(post_activation, pre_activation)
        correction = np.outer(post_activation * post_activation, np.ones_like(pre_activation)) * weights
        return eta_o * (outer - correction)
    return eta_o * (post_activation * pre_activation - post_activation * post_activation * weights)


def normalize_plasticity_vector(
    values: np.ndarray,
    *,
    size: int,
    mode: str = "soft_clip_v1",
) -> np.ndarray:
    """Sanitize and normalize an activation before long-term plasticity.

    Forward inference keeps its own feature contract.  This helper only
    bounds the correlation signal used by Hebbian/Oja updates, preventing a
    single raw feature tail from producing a disproportionately large weight
    change.  ``raw`` remains available for protected legacy artifacts.
    """

    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size != int(size):
        resized = np.zeros(int(size), dtype=float)
        resized[: min(arr.size, int(size))] = arr[: int(size)]
        arr = resized
    arr = np.nan_to_num(arr, nan=0.0, posinf=20.0, neginf=-20.0)
    return normalize_input_vector(arr, mode=str(mode or "soft_clip_v1"), size=int(size))


def compose_stable_plasticity_update(
    weights: np.ndarray,
    pre_activation: np.ndarray,
    post_activation: np.ndarray,
    grad_loss_w: np.ndarray | None,
    coeffs: PlasticityCoefficients,
    reward_modulator: float,
    *,
    input_mode: str = "soft_clip_v1",
    max_update_norm: float = 0.25,
    stdp_delta: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Compose a bounded, sign-safe plasticity update.

    Reward gating applies only to directed learning.  Anti-Hebbian
    decorrelation, the Oja stability correction, and L2 decay never reverse
    under negative reward.  This is the key guard against runaway overfit in
    short losing windows.
    """

    w = np.asarray(weights, dtype=float)
    if w.ndim != 2:
        raise ValueError(f"plasticity weights must be a matrix, got {w.shape}")
    hidden_size, input_size = w.shape
    pre = normalize_plasticity_vector(pre_activation, size=input_size, mode=input_mode)
    post_raw = np.asarray(post_activation, dtype=float).reshape(-1)
    if post_raw.size != hidden_size:
        resized = np.zeros(hidden_size, dtype=float)
        resized[: min(post_raw.size, hidden_size)] = post_raw[:hidden_size]
        post_raw = resized
    post = np.tanh(np.nan_to_num(post_raw, nan=0.0, posinf=20.0, neginf=-20.0))
    modulator = float(np.clip(np.nan_to_num(reward_modulator, nan=0.0), -2.0, 2.0))

    grad = np.zeros_like(w)
    if grad_loss_w is not None:
        candidate = np.asarray(grad_loss_w, dtype=float)
        if candidate.shape == w.shape and np.all(np.isfinite(candidate)):
            # A true loss gradient must keep descending even when the reward
            # signal is negative; multiplying it by reward would turn descent
            # into ascent exactly when the policy is already failing.
            grad = gradient_plasticity(w, candidate, coeffs.eta_grad)

    hebbian = hebbian_update(pre, post, coeffs.eta_hebb)
    anti_hebbian = anti_hebbian_update(pre, post, coeffs.eta_anti_hebb)
    oja_drive = coeffs.eta_oja * np.outer(post, pre)
    # This term is the actual Oja norm correction and must always damp large
    # weights.  It is deliberately kept outside reward gating.
    oja_stability = -coeffs.eta_oja * (post * post)[:, None] * w
    regularization = -coeffs.lambda_reg * w

    # Negative reward reverses Hebbian reinforcement, but never turns the
    # decorrelating/stability terms into reinforcement.  Oja drive is only
    # allowed on positive evidence; bad evidence relies on damping instead.
    directed = grad + modulator * hebbian + max(modulator, 0.0) * oja_drive
    anti_scale = max(0.0, 1.0 - min(1.0, abs(modulator)))
    stable = anti_scale * anti_hebbian + oja_stability + regularization
    delta = directed + stable
    if stdp_delta is not None:
        stdp_matrix = np.asarray(stdp_delta, dtype=float)
        if stdp_matrix.shape == w.shape and np.all(np.isfinite(stdp_matrix)):
            # STDP is associative, so its sign is still credit-gated, but it
            # cannot bypass the update norm guard below.
            delta = delta + modulator * stdp_matrix

    raw_norm = float(np.linalg.norm(delta))
    update_scale = 1.0
    limit = float(max(0.0, max_update_norm))
    if limit > 0.0 and raw_norm > limit:
        update_scale = limit / max(raw_norm, 1e-12)
        delta = delta * update_scale

    stats = {
        "reward_modulator": modulator,
        "pre_rms": float(np.sqrt(np.mean(pre * pre))) if pre.size else 0.0,
        "pre_max_abs": float(np.max(np.abs(pre), initial=0.0)),
        "post_rms": float(np.sqrt(np.mean(post * post))) if post.size else 0.0,
        "grad_norm": float(np.linalg.norm(grad)),
        "hebbian_norm": float(np.linalg.norm(hebbian)),
        "anti_hebbian_norm": float(np.linalg.norm(anti_scale * anti_hebbian)),
        "anti_hebbian_scale": anti_scale,
        "oja_drive_norm": float(np.linalg.norm(oja_drive)),
        "oja_stability_norm": float(np.linalg.norm(oja_stability)),
        "regularization_norm": float(np.linalg.norm(regularization)),
        "stdp_norm": float(np.linalg.norm(stdp_delta)) if stdp_delta is not None else 0.0,
        "raw_update_norm": raw_norm,
        "update_norm": float(np.linalg.norm(delta)),
        "update_scale": update_scale,
    }
    return delta, pre, stats


def stdp_update(
    pre_spike_times: np.ndarray,
    post_spike_time: float,
    params: STDPParams,
) -> np.ndarray:
    dt = post_spike_time - pre_spike_times
    delta = np.zeros_like(dt)
    pos_mask = dt > 0
    neg_mask = dt < 0
    if np.any(pos_mask):
        delta[pos_mask] = params.A_plus * np.exp(-dt[pos_mask] / params.tau_plus)
    if np.any(neg_mask):
        delta[neg_mask] = -params.A_minus * np.exp(dt[neg_mask] / params.tau_minus)
    return delta


def stdp_update_scalar(dt: float, params: STDPParams) -> float:
    if dt > 0:
        return params.A_plus * np.exp(-dt / params.tau_plus)
    elif dt < 0:
        return -params.A_minus * np.exp(dt / params.tau_minus)
    return 0.0


def short_term_plasticity_effect(
    weights: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    return weights * p


def homeostatic_update_bias(
    bias: np.ndarray,
    activation_ema: np.ndarray,
    a_target: float,
    eta_b: float,
) -> np.ndarray:
    return bias + eta_b * (a_target - activation_ema)


def homeostatic_rescale_weights(
    weights: np.ndarray,
    activation_ema: np.ndarray,
    a_target: float,
) -> np.ndarray:
    scale = a_target / (activation_ema + 1e-8)
    return weights * scale


def combined_synapse_update(
    weights: np.ndarray,
    pre_activation: np.ndarray,
    post_activation: np.ndarray,
    grad_loss_w: np.ndarray | None,
    coeffs: PlasticityCoefficients,
    mean_activation: np.ndarray | None = None,
    weight_decay: float = 0.0001,
) -> np.ndarray:
    # Retained as a compatibility helper for old callers, but with the same
    # sign-safe decomposition as the live ModuleState path.
    matrix = np.asarray(weights, dtype=float)
    hidden_size, input_size = matrix.shape
    pre = normalize_plasticity_vector(pre_activation, size=input_size, mode="soft_clip_v1")
    post_raw = np.asarray(post_activation, dtype=float).reshape(-1)
    post = np.zeros(hidden_size, dtype=float)
    post[: min(hidden_size, post_raw.size)] = post_raw[:hidden_size]
    post = np.tanh(np.nan_to_num(post, nan=0.0, posinf=20.0, neginf=-20.0))
    delta = np.zeros_like(matrix)
    if grad_loss_w is not None and np.asarray(grad_loss_w).shape == matrix.shape:
        delta += gradient_plasticity(matrix, np.asarray(grad_loss_w, dtype=float), coeffs.eta_grad)
    delta += hebbian_update(pre, post, coeffs.eta_hebb)
    delta += anti_hebbian_update(pre, post, coeffs.eta_anti_hebb)
    delta += coeffs.eta_oja * np.outer(post, pre)
    delta -= coeffs.eta_oja * (post * post)[:, None] * matrix
    if mean_activation is not None:
        mean = np.asarray(mean_activation, dtype=float).reshape(-1)
        mean_vec = np.zeros(hidden_size, dtype=float)
        mean_vec[: min(hidden_size, mean.size)] = mean[:hidden_size]
        delta -= coeffs.eta_anti_hebb * np.abs(mean_vec)[:, None] * matrix
    delta -= float(max(0.0, weight_decay)) * matrix
    return np.nan_to_num(delta)


def compute_total_loss(
    l_task: float,
    weights: list[np.ndarray],
    activations: list[np.ndarray],
    lambda_reg: float = 0.0001,
    lambda_sparse: float = 0.001,
    lambda_homeo: float = 0.01,
    lambda_plastic: float = 0.001,
    a_target: float = 0.35,
    gate_values: np.ndarray | None = None,
) -> tuple[float, dict[str, float]]:
    l_reg = 0.0
    for w in weights:
        l_reg += float(np.sum(w * w))

    l_sparse = 0.0
    if gate_values is not None:
        gate_mean = float(np.mean(gate_values))
        if gate_mean > 0 and gate_mean < 1:
            l_sparse = -(gate_mean * np.log(gate_mean + 1e-10) + (1 - gate_mean) * np.log(1 - gate_mean + 1e-10))

    l_homeo = 0.0
    for act in activations:
        if act.size > 0:
            mean_act = float(np.mean(np.abs(act)))
            l_homeo += (mean_act - a_target) ** 2

    l_plastic = 0.0
    for w in weights:
        l_plastic += float(np.sum(np.abs(w)))

    total = l_task + lambda_reg * l_reg + lambda_sparse * l_sparse + lambda_homeo * l_homeo + lambda_plastic * l_plastic
    components = {
        "l_task": l_task,
        "l_reg": lambda_reg * l_reg,
        "l_sparse": lambda_sparse * l_sparse,
        "l_homeo": lambda_homeo * l_homeo,
        "l_plastic": lambda_plastic * l_plastic,
    }
    return total, components
