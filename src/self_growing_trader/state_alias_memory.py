"""Causal working memory for observations that alias at the current tick.

Two market snapshots can have nearly identical current values while arriving
through very different paths.  ``StateAliasingMemoryState`` keeps a bounded,
ordered buffer of past routed observations and performs a differentiable Q/K/V
read before the Direct-RNN encoder.  Learned parameters belong to the existing
Full-RL AdamW writer; the runtime buffer is data, never a second optimizer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import copy
import hashlib
from typing import Mapping

import numpy as np


STATE_ALIAS_MEMORY_SCHEMA_VERSION = 2
STATE_ALIAS_MEMORY_ARTIFACT_VERSION = 15
STATE_ALIAS_MEMORY_PARAMETER_PREFIX = "state_memory."
STATE_ALIAS_MEMORY_CONFIG_NAMES = (
    "direct_rnn_state_memory_enabled",
    "direct_rnn_state_memory_capacity",
    "direct_rnn_state_memory_size",
    "direct_rnn_state_memory_min_history",
    "direct_rnn_state_memory_temperature",
    "direct_rnn_state_memory_injection_scale",
    "direct_rnn_state_memory_path_delta_scale",
)


def _softmax(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if not array.size:
        return array
    shifted = np.clip(array - float(np.max(array)), -60.0, 60.0)
    weights = np.exp(shifted)
    return weights / max(float(np.sum(weights)), 1e-12)


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(float(value), -60.0, 60.0))))


def _sinusoidal_positions(capacity: int, width: int) -> np.ndarray:
    positions = np.arange(max(1, int(capacity)), dtype=float).reshape(-1, 1)
    dimensions = np.arange(max(1, int(width)), dtype=float).reshape(1, -1)
    scale = np.power(10000.0, -2.0 * np.floor(dimensions / 2.0) / max(1.0, float(width)))
    angles = positions * scale
    result = np.where((dimensions.astype(int) % 2) == 0, np.sin(angles), np.cos(angles))
    return 0.02 * result


@dataclass(frozen=True)
class StateAliasingMemoryForward:
    """Exact causal read cache retained until lifecycle settlement."""

    schema_version: int
    enabled: bool
    ready: bool
    history_length: int
    history_fingerprint: str
    observation: np.ndarray
    tokens: np.ndarray
    position_indices: np.ndarray
    query: np.ndarray
    keys: np.ndarray
    values: np.ndarray
    attention_weights: np.ndarray
    content: np.ndarray
    projected: np.ndarray
    gate: float
    injected: np.ndarray
    denominator: float

    def payload(self) -> dict[str, object]:
        weights = np.asarray(self.attention_weights, dtype=float).reshape(-1)
        positive = weights[weights > 1e-12]
        entropy = float(-np.sum(positive * np.log(positive))) if positive.size else 0.0
        normalized_entropy = float(entropy / max(np.log(max(2, weights.size)), 1e-12)) if weights.size else 0.0
        top_index = int(np.argmax(weights)) if weights.size else -1
        # age=1 is the most recent stored observation.
        top_age = int(weights.size - top_index) if top_index >= 0 else 0
        return {
            "schema_version": int(self.schema_version),
            "enabled": bool(self.enabled),
            "ready": bool(self.ready),
            "history_length": int(self.history_length),
            "history_fingerprint": str(self.history_fingerprint),
            "attention_entropy": entropy,
            "attention_entropy_normalized": normalized_entropy,
            "attention_effective_history": float(np.exp(entropy)) if weights.size else 0.0,
            "top_history_age": top_age,
            "top_attention_weight": float(weights[top_index]) if top_index >= 0 else 0.0,
            "gate": float(self.gate),
            "context_norm": float(np.linalg.norm(self.content)),
            "injection_norm": float(np.linalg.norm(self.injected)),
            "path_delta_norm": float(np.mean(np.linalg.norm(
                np.diff(np.asarray(self.tokens, dtype=float), axis=0), axis=1
            ))) if self.tokens.shape[0] > 1 else 0.0,
            "attention_weights": [float(value) for value in weights.tolist()],
        }


@dataclass
class StateAliasingMemoryState:
    """Per-expert ordered working memory with an AdamW-owned Q/K/V reader."""

    schema_version: int = STATE_ALIAS_MEMORY_SCHEMA_VERSION
    enabled: bool = False
    input_size: int = 137
    hidden_size: int = 64
    memory_size: int = 32
    capacity: int = 32
    min_history: int = 4
    temperature: float = 1.0
    injection_scale: float = 0.25
    path_delta_scale: float = 0.75
    query_w: np.ndarray | None = field(default=None, repr=False)
    query_b: np.ndarray | None = field(default=None, repr=False)
    key_w: np.ndarray | None = field(default=None, repr=False)
    key_b: np.ndarray | None = field(default=None, repr=False)
    value_w: np.ndarray | None = field(default=None, repr=False)
    value_b: np.ndarray | None = field(default=None, repr=False)
    output_w: np.ndarray | None = field(default=None, repr=False)
    output_b: np.ndarray | None = field(default=None, repr=False)
    relative_position: np.ndarray | None = field(default=None, repr=False)
    recency_bias: np.ndarray | None = field(default=None, repr=False)
    gate_w: np.ndarray | None = field(default=None, repr=False)
    gate_b: np.ndarray | None = field(default=None, repr=False)
    runtime_buffer: list[np.ndarray] = field(default_factory=list, repr=False)
    read_count: int = 0
    ready_read_count: int = 0

    @classmethod
    def seed(
        cls,
        *,
        input_size: int,
        hidden_size: int,
        memory_size: int = 32,
        capacity: int = 32,
        min_history: int = 4,
        temperature: float = 1.0,
        injection_scale: float = 0.25,
        path_delta_scale: float = 0.75,
        enabled: bool = False,
        seed: int = 0,
    ) -> "StateAliasingMemoryState":
        rng = np.random.default_rng(int(seed) % (2**32))
        input_width = max(1, int(input_size))
        hidden_width = max(1, int(hidden_size))
        memory_width = max(4, int(memory_size))
        length = max(2, int(capacity))
        # V190 used 0.20-scaled Q/K, a 0.02 output projection and gate=-2.
        # On real 137-wide observations that combination made every score
        # almost equal and reduced the injected history to ~1e-3.  The V192
        # seed keeps variance bounded while giving content and path deltas a
        # usable signal from the first Full-RL batch.
        input_scale = 0.80 / np.sqrt(float(input_width))
        value_scale = 0.70 / np.sqrt(float(input_width))
        output_scale = 0.25 / np.sqrt(float(memory_width))
        return cls(
            enabled=bool(enabled),
            input_size=input_width,
            hidden_size=hidden_width,
            memory_size=memory_width,
            capacity=length,
            min_history=int(np.clip(min_history, 1, length)),
            temperature=float(np.clip(temperature, 0.05, 10.0)),
            injection_scale=float(np.clip(injection_scale, 0.0, 2.0)),
            path_delta_scale=float(np.clip(path_delta_scale, 0.0, 4.0)),
            query_w=rng.normal(0.0, input_scale, size=(memory_width, input_width)),
            query_b=np.zeros(memory_width, dtype=float),
            key_w=rng.normal(0.0, input_scale, size=(memory_width, input_width)),
            key_b=np.zeros(memory_width, dtype=float),
            value_w=rng.normal(0.0, value_scale, size=(memory_width, input_width)),
            value_b=np.zeros(memory_width, dtype=float),
            output_w=rng.normal(0.0, output_scale, size=(hidden_width, memory_width)),
            output_b=np.zeros(hidden_width, dtype=float),
            relative_position=_sinusoidal_positions(length, memory_width),
            recency_bias=np.linspace(-0.20, 0.0, length, dtype=float),
            gate_w=np.zeros(input_width, dtype=float),
            gate_b=np.asarray([-0.5], dtype=float),
        )

    @staticmethod
    def deterministic_seed(owner: str, input_size: int, hidden_size: int, memory_size: int) -> int:
        payload = f"{owner}|{int(input_size)}|{int(hidden_size)}|{int(memory_size)}".encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")

    def ensure_compatible(self) -> None:
        self.schema_version = STATE_ALIAS_MEMORY_SCHEMA_VERSION
        self.input_size = max(1, int(getattr(self, "input_size", 137) or 137))
        self.hidden_size = max(1, int(getattr(self, "hidden_size", 64) or 64))
        self.memory_size = max(4, int(getattr(self, "memory_size", 32) or 32))
        self.capacity = max(2, int(getattr(self, "capacity", 32) or 32))
        self.min_history = int(np.clip(getattr(self, "min_history", 4), 1, self.capacity))
        self.temperature = float(np.clip(getattr(self, "temperature", 1.0), 0.05, 10.0))
        self.injection_scale = float(np.clip(getattr(self, "injection_scale", 0.25), 0.0, 2.0))
        self.path_delta_scale = float(np.clip(
            getattr(self, "path_delta_scale", 0.75), 0.0, 4.0
        ))
        shapes = {
            "query_w": (self.memory_size, self.input_size),
            "query_b": (self.memory_size,),
            "key_w": (self.memory_size, self.input_size),
            "key_b": (self.memory_size,),
            "value_w": (self.memory_size, self.input_size),
            "value_b": (self.memory_size,),
            "output_w": (self.hidden_size, self.memory_size),
            "output_b": (self.hidden_size,),
            "relative_position": (self.capacity, self.memory_size),
            "recency_bias": (self.capacity,),
            "gate_w": (self.input_size,),
            "gate_b": (1,),
        }
        fallback = StateAliasingMemoryState.seed(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            memory_size=self.memory_size,
            capacity=self.capacity,
            min_history=self.min_history,
            temperature=self.temperature,
            injection_scale=self.injection_scale,
            path_delta_scale=self.path_delta_scale,
            enabled=bool(getattr(self, "enabled", False)),
            seed=0,
        )
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name, None), dtype=float)
            if value.shape != shape:
                setattr(self, name, np.asarray(getattr(fallback, name), dtype=float).copy())
            else:
                setattr(self, name, np.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0))
        raw_buffer = list(getattr(self, "runtime_buffer", []) or [])
        self.runtime_buffer = [
            np.asarray(item, dtype=float).reshape(-1).copy()
            for item in raw_buffer
            if np.asarray(item).size == self.input_size
        ][-self.capacity:]
        self.read_count = max(0, int(getattr(self, "read_count", 0) or 0))
        self.ready_read_count = max(0, int(getattr(self, "ready_read_count", 0) or 0))

    def configure(self, config: Mapping[str, object]) -> None:
        self.enabled = bool(config.get("direct_rnn_state_memory_enabled", self.enabled))
        self.capacity = max(2, int(config.get("direct_rnn_state_memory_capacity", self.capacity) or self.capacity))
        self.memory_size = max(4, int(config.get("direct_rnn_state_memory_size", self.memory_size) or self.memory_size))
        self.min_history = int(np.clip(
            int(config.get("direct_rnn_state_memory_min_history", self.min_history) or self.min_history),
            1,
            self.capacity,
        ))
        self.temperature = float(np.clip(
            float(config.get("direct_rnn_state_memory_temperature", self.temperature) or self.temperature),
            0.05,
            10.0,
        ))
        self.injection_scale = float(np.clip(
            float(config.get("direct_rnn_state_memory_injection_scale", self.injection_scale) or 0.0),
            0.0,
            2.0,
        ))
        self.path_delta_scale = float(np.clip(
            float(config.get(
                "direct_rnn_state_memory_path_delta_scale", self.path_delta_scale
            ) or 0.0),
            0.0,
            4.0,
        ))
        self.ensure_compatible()

    def copy(self) -> "StateAliasingMemoryState":
        copied = copy.copy(self)
        for name, value in self.parameters().items():
            setattr(copied, name[len(STATE_ALIAS_MEMORY_PARAMETER_PREFIX):], value.copy())
        copied.runtime_buffer = [np.asarray(item, dtype=float).copy() for item in self.runtime_buffer]
        return copied

    def reset_runtime(self) -> None:
        self.runtime_buffer = []
        self.read_count = 0
        self.ready_read_count = 0

    def set_runtime_buffer(self, values: list[np.ndarray] | tuple[np.ndarray, ...]) -> None:
        self.runtime_buffer = [
            np.asarray(item, dtype=float).reshape(-1).copy()
            for item in values
            if np.asarray(item).size == self.input_size
        ][-self.capacity:]

    def remember(self, observation: np.ndarray) -> None:
        if not self.enabled:
            return
        vector = np.asarray(observation, dtype=float).reshape(-1)
        if vector.size != self.input_size:
            return
        self.runtime_buffer = [*self.runtime_buffer, vector.copy()][-self.capacity:]

    @staticmethod
    def _fingerprint(tokens: np.ndarray) -> str:
        values = np.ascontiguousarray(np.asarray(tokens, dtype=np.float32))
        return hashlib.sha256(values.tobytes(order="C")).hexdigest()[:20] if values.size else "cold"

    def read(self, observation: np.ndarray, *, count: bool = True) -> StateAliasingMemoryForward:
        self.ensure_compatible()
        vector = np.asarray(observation, dtype=float).reshape(-1)
        if vector.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[:min(vector.size, self.input_size)] = vector[:self.input_size]
            vector = padded
        tokens = np.asarray(self.runtime_buffer[-self.capacity:], dtype=float)
        if tokens.size == 0:
            tokens = np.zeros((0, self.input_size), dtype=float)
        return self._read_tokens(vector, tokens, count=count)

    def read_frozen(
        self,
        observation: np.ndarray,
        tokens: np.ndarray,
    ) -> StateAliasingMemoryForward:
        """Re-evaluate an immutable decision history with current parameters.

        Delayed credit must not read the *current* runtime buffer: that would
        leak bars that occurred after the decision.  The tokens captured in
        ``StateAliasingMemoryForward`` are causal data, so this method applies
        current Q/K/V weights to those frozen tokens without incrementing
        runtime counters or modifying the buffer.
        """
        self.ensure_compatible()
        vector = np.asarray(observation, dtype=float).reshape(-1)
        if vector.size != self.input_size:
            padded = np.zeros(self.input_size, dtype=float)
            padded[:min(vector.size, self.input_size)] = vector[:self.input_size]
            vector = padded
        frozen = np.asarray(tokens, dtype=float)
        if frozen.ndim != 2 or frozen.shape[1:] != (self.input_size,):
            frozen = np.zeros((0, self.input_size), dtype=float)
        frozen = frozen[-self.capacity:].copy()
        return self._read_tokens(vector, frozen, count=False)

    def _read_tokens(
        self,
        vector: np.ndarray,
        tokens: np.ndarray,
        *,
        count: bool,
    ) -> StateAliasingMemoryForward:
        length = int(tokens.shape[0])
        ready = bool(self.enabled and length >= self.min_history)
        if count:
            self.read_count += int(self.enabled)
            self.ready_read_count += int(ready)
        if not ready:
            zeros_m = np.zeros(self.memory_size, dtype=float)
            return StateAliasingMemoryForward(
                self.schema_version, bool(self.enabled), False, length,
                self._fingerprint(tokens), vector.copy(), tokens.copy(),
                np.zeros(0, dtype=int), zeros_m.copy(),
                np.zeros((0, self.memory_size), dtype=float),
                np.zeros((0, self.memory_size), dtype=float),
                np.zeros(0, dtype=float), zeros_m.copy(),
                np.zeros(self.hidden_size, dtype=float), 0.0,
                np.zeros(self.hidden_size, dtype=float),
                float(np.sqrt(self.memory_size) * self.temperature),
            )
        position_indices = np.arange(self.capacity - length, self.capacity, dtype=int)
        query_source, path_tokens = self._path_sources(vector, tokens)
        query = self.query_w @ query_source + self.query_b
        keys = path_tokens @ self.key_w.T + self.key_b + self.relative_position[position_indices]
        values = path_tokens @ self.value_w.T + self.value_b
        denominator = float(max(1e-6, np.sqrt(self.memory_size) * self.temperature))
        scores = (keys @ query) / denominator + self.recency_bias[position_indices]
        weights = _softmax(scores)
        content = weights @ values
        projected = self.output_w @ content + self.output_b
        gate = _sigmoid(float(self.gate_w @ vector + self.gate_b[0]))
        injected = float(self.injection_scale) * gate * projected
        return StateAliasingMemoryForward(
            self.schema_version, True, True, length, self._fingerprint(tokens),
            vector.copy(), tokens.copy(), position_indices.copy(), query.copy(),
            keys.copy(), values.copy(), weights.copy(), content.copy(),
            projected.copy(), gate, injected.copy(), denominator,
        )

    def _path_sources(
        self,
        observation: np.ndarray,
        tokens: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Encode ordered history without adding a hand-written action rule.

        Raw snapshot means are invariant to permutation, which allowed two
        opposite paths to alias.  Adding the causal first difference makes
        key/value tokens order-sensitive; the query uses the transition from
        the latest stored state into the current observation.  These are still
        inputs to learned Q/K/V tensors and never choose an action directly.
        """
        values = np.asarray(tokens, dtype=float)
        if values.ndim != 2 or not values.shape[0]:
            return np.asarray(observation, dtype=float).copy(), values.copy()
        previous = np.vstack((values[:1], values[:-1]))
        deltas = values - previous
        scale = float(self.path_delta_scale)
        path_tokens = values + scale * deltas
        query_source = np.asarray(observation, dtype=float) + scale * (
            np.asarray(observation, dtype=float) - values[-1]
        )
        return query_source, path_tokens

    def backward(
        self,
        forward: StateAliasingMemoryForward | None,
        upstream: np.ndarray,
    ) -> dict[str, np.ndarray]:
        if not isinstance(forward, StateAliasingMemoryForward) or not forward.ready:
            return {}
        direction = np.asarray(upstream, dtype=float).reshape(-1)
        if direction.size != self.hidden_size:
            return {}
        scale = float(self.injection_scale)
        gate = float(forward.gate)
        projected_direction = direction * scale * gate
        gate_direction = float(direction @ (scale * forward.projected))
        gate_logit_direction = gate_direction * gate * (1.0 - gate)
        content_direction = self.output_w.T @ projected_direction
        weight_direction = forward.values @ content_direction
        value_direction = np.outer(forward.attention_weights, content_direction)
        score_direction = forward.attention_weights * (
            weight_direction - float(forward.attention_weights @ weight_direction)
        )
        query_direction = forward.keys.T @ score_direction / forward.denominator
        key_direction = np.outer(score_direction, forward.query) / forward.denominator
        query_source, path_tokens = self._path_sources(
            forward.observation, forward.tokens
        )
        gradients = {
            "query_w": np.outer(query_direction, query_source),
            "query_b": query_direction,
            "key_w": key_direction.T @ path_tokens,
            "key_b": np.sum(key_direction, axis=0),
            "value_w": value_direction.T @ path_tokens,
            "value_b": np.sum(value_direction, axis=0),
            "output_w": np.outer(projected_direction, forward.content),
            "output_b": projected_direction,
            "relative_position": np.zeros_like(self.relative_position),
            "recency_bias": np.zeros_like(self.recency_bias),
            "gate_w": gate_logit_direction * forward.observation,
            "gate_b": np.asarray([gate_logit_direction], dtype=float),
        }
        gradients["relative_position"][forward.position_indices] = key_direction
        gradients["recency_bias"][forward.position_indices] = score_direction
        return {
            f"{STATE_ALIAS_MEMORY_PARAMETER_PREFIX}{name}": value
            for name, value in gradients.items()
        }

    def parameters(self) -> dict[str, np.ndarray]:
        self.ensure_compatible()
        names = (
            "query_w", "query_b", "key_w", "key_b", "value_w", "value_b",
            "output_w", "output_b", "relative_position", "recency_bias",
            "gate_w", "gate_b",
        )
        return {
            f"{STATE_ALIAS_MEMORY_PARAMETER_PREFIX}{name}": np.asarray(getattr(self, name), dtype=float)
            for name in names
        }

    def sanitize(self) -> None:
        for value in self.parameters().values():
            np.nan_to_num(value, copy=False, nan=0.0, posinf=5.0, neginf=-5.0)
            np.clip(value, -10.0, 10.0, out=value)

    def summary(self) -> dict[str, object]:
        return {
            "schema_version": int(self.schema_version),
            "enabled": bool(self.enabled),
            "capacity": int(self.capacity),
            "memory_size": int(self.memory_size),
            "min_history": int(self.min_history),
            "temperature": float(self.temperature),
            "injection_scale": float(self.injection_scale),
            "path_delta_scale": float(self.path_delta_scale),
            "runtime_history_length": int(len(self.runtime_buffer)),
            "read_count": int(self.read_count),
            "ready_read_count": int(self.ready_read_count),
            "ready_rate": float(self.ready_read_count / max(1, self.read_count)),
            "parameter_count": int(sum(value.size for value in self.parameters().values())),
        }


__all__ = [
    "STATE_ALIAS_MEMORY_ARTIFACT_VERSION",
    "STATE_ALIAS_MEMORY_CONFIG_NAMES",
    "STATE_ALIAS_MEMORY_PARAMETER_PREFIX",
    "STATE_ALIAS_MEMORY_SCHEMA_VERSION",
    "StateAliasingMemoryForward",
    "StateAliasingMemoryState",
]
