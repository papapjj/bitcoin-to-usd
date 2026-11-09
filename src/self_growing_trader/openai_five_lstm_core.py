from __future__ import annotations

from dataclasses import dataclass
import numpy as np


OPENAI_FIVE_RECURRENT_CORE = "openai_five_lstm_v1"
OPENAI_FIVE_LSTM_SIZE = 4096


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))).astype(np.float32, copy=False)


@dataclass(frozen=True)
class OpenAIFiveLSTMForward:
    observation: np.ndarray
    previous_hidden: np.ndarray
    previous_cell: np.ndarray
    input_gate: np.ndarray
    forget_gate: np.ndarray
    candidate: np.ndarray
    output_gate: np.ndarray
    cell: np.ndarray
    hidden: np.ndarray


def initialize_lstm(
    input_size: int,
    hidden_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize one standard single-layer LSTM parameter set.

    OpenAI's public Five report specifies a single 4096-unit LSTM, but does
    not publish an exact parameter initializer.  The implementation therefore
    keeps the architecture exact while using a conventional variance-scaled
    initializer for fresh trading checkpoints.
    """
    input_size = int(input_size)
    hidden_size = int(hidden_size)
    input_scale = 1.0 / max(1.0, np.sqrt(float(input_size)))
    recurrent_scale = 1.0 / max(1.0, np.sqrt(float(hidden_size)))
    w_in = rng.normal(0.0, input_scale, size=(4 * hidden_size, input_size)).astype(np.float32)
    w_rec = rng.normal(0.0, recurrent_scale, size=(4 * hidden_size, hidden_size)).astype(np.float32)
    bias = np.zeros(4 * hidden_size, dtype=np.float32)
    return w_in, w_rec, bias


def lstm_step(
    observation: np.ndarray,
    previous_hidden: np.ndarray,
    previous_cell: np.ndarray,
    w_in: np.ndarray,
    w_rec: np.ndarray,
    bias: np.ndarray,
) -> OpenAIFiveLSTMForward:
    """Run one timestep of a standard input/forget/candidate/output LSTM."""
    x = np.asarray(observation, dtype=np.float32).reshape(-1)
    h_prev = np.asarray(previous_hidden, dtype=np.float32).reshape(-1)
    c_prev = np.asarray(previous_cell, dtype=np.float32).reshape(-1)
    hidden_size = h_prev.size
    z = np.asarray(w_in, dtype=np.float32) @ x + np.asarray(w_rec, dtype=np.float32) @ h_prev + np.asarray(bias, dtype=np.float32)
    i_raw, f_raw, g_raw, o_raw = np.split(z, 4)
    i = _sigmoid(i_raw)
    f = _sigmoid(f_raw)
    g = np.tanh(g_raw).astype(np.float32, copy=False)
    o = _sigmoid(o_raw)
    c = (f * c_prev + i * g).astype(np.float32, copy=False)
    h = (o * np.tanh(c)).astype(np.float32, copy=False)
    return OpenAIFiveLSTMForward(
        observation=x.copy(),
        previous_hidden=h_prev.copy(),
        previous_cell=c_prev.copy(),
        input_gate=i.copy(),
        forget_gate=f.copy(),
        candidate=g.copy(),
        output_gate=o.copy(),
        cell=c.copy(),
        hidden=h.copy(),
    )


def lstm_backward_step(
    forward: OpenAIFiveLSTMForward,
    hidden_direction: np.ndarray,
    cell_direction: np.ndarray,
    w_rec: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Backpropagate an ascent direction through one LSTM timestep."""
    dh = np.asarray(hidden_direction, dtype=np.float32).reshape(-1)
    dc_next = np.asarray(cell_direction, dtype=np.float32).reshape(-1)
    c = np.asarray(forward.cell, dtype=np.float32)
    c_prev = np.asarray(forward.previous_cell, dtype=np.float32)
    i = np.asarray(forward.input_gate, dtype=np.float32)
    f = np.asarray(forward.forget_gate, dtype=np.float32)
    g = np.asarray(forward.candidate, dtype=np.float32)
    o = np.asarray(forward.output_gate, dtype=np.float32)

    tanh_c = np.tanh(c).astype(np.float32, copy=False)
    dc = dc_next + dh * o * (1.0 - tanh_c * tanh_c)
    do = dh * tanh_c
    df = dc * c_prev
    di = dc * g
    dg = dc * i

    dz_i = di * i * (1.0 - i)
    dz_f = df * f * (1.0 - f)
    dz_g = dg * (1.0 - g * g)
    dz_o = do * o * (1.0 - o)
    dz = np.concatenate((dz_i, dz_f, dz_g, dz_o)).astype(np.float32, copy=False)

    directions = {
        "lstm_w_in": np.outer(dz, np.asarray(forward.observation, dtype=np.float32)),
        "lstm_w_rec": np.outer(dz, np.asarray(forward.previous_hidden, dtype=np.float32)),
        "lstm_b": dz.copy(),
    }
    previous_hidden_direction = np.asarray(w_rec, dtype=np.float32).T @ dz
    previous_cell_direction = dc * f
    return directions, previous_hidden_direction.astype(np.float32), previous_cell_direction.astype(np.float32)
