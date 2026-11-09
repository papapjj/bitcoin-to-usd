"""SGT-V193-Compact Sequence Core (xLSTM: mLSTM 768D + sLSTM 768D + Relational Graph Transformer).

Production-grade PyTorch implementation complying with:
- Beck et al. (NeurIPS 2024 / xLSTM: Extended Long Short-Term Memory)
- Top 3 Technical Design Docs: Design Doc 1 (SGT-V193-Compact Sequence Core)
- System Invariants: 137 input floats, 20 action wire output floats, 5-agent shared weights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Constants for SGT-V193-Compact
INPUT_COUNT = 137
OUTPUT_COUNT = 20
ACTION_COUNT = 12
FIVE_AGENT_COUNT = 5

COMPACT_PRE_SIZE = 1024
COMPACT_HIDDEN_SIZE = 768
COMPACT_NUM_HEADS = 4
COMPACT_HEAD_DIM = COMPACT_HIDDEN_SIZE // COMPACT_NUM_HEADS  # 192

PARAMETER_BINS = {
    "delay": 4,
    "risk": 8,
    "entry_offset": 17,
    "sl": 24,
    "tp": 24,
    "trailing": 16,
    "manage_mode": 3,
    "close_fraction": 10,
}

RISK_OUTPUT_INDEX = 12
ACTION_DELAY_OUTPUT_INDEX = 13
FACTORIZED_ACTION_MARKER_OUTPUT_INDEX = 14
ENTRY_OFFSET_OUTPUT_INDEX = 15
GEOMETRY_PACKED_OUTPUT_INDEX = 16
PARTIAL_CLOSE_OUTPUT_INDEX = 17
MANAGEMENT_MODE_OUTPUT_INDEX = 18
TARGET_SLOT_OUTPUT_INDEX = 19
FACTORIZED_ACTION_MARKER = 1.0
MAX_POSITION_SLOTS = 8
ENTRY_ACTIONS = (1, 2, 3, 4, 5, 6)
DEFAULT_SL_BIN_INDEX = 12
DEFAULT_TP_BIN_INDEX = 12
DEFAULT_TRAILING_BIN_INDEX = 0


@dataclass(frozen=True)
class MatrixLSTMState:
    """Recurrent state for one agent in Matrix LSTM (mLSTM)."""
    covariance: Tensor   # [H, d_h, d_h] -> [4, 192, 192] (147,456 memory elements per agent)
    normalizer: Tensor   # [H, d_h] -> [4, 192]
    stabilizer: Tensor   # [H] -> [4]

    def detached(self) -> "MatrixLSTMState":
        return MatrixLSTMState(
            covariance=self.covariance.detach().clone(),
            normalizer=self.normalizer.detach().clone(),
            stabilizer=self.stabilizer.detach().clone(),
        )


@dataclass(frozen=True)
class ScalarLSTMState:
    """Recurrent state for one agent in Scalar LSTM (sLSTM)."""
    cell: Tensor         # [H, d_h] -> [4, 192] (768 scalars)
    normalizer: Tensor   # [H, d_h] -> [4, 192]
    stabilizer: Tensor   # [H, d_h] -> [4, 192]

    def detached(self) -> "ScalarLSTMState":
        return ScalarLSTMState(
            cell=self.cell.detach().clone(),
            normalizer=self.normalizer.detach().clone(),
            stabilizer=self.stabilizer.detach().clone(),
        )


@dataclass(frozen=True)
class CompactAgentMemory:
    """Joint recurrent memory state for one agent in SGT-V193-Compact.
    
    Fully duck-type compatible with FiveAgentMemory.
    """
    mlstm: MatrixLSTMState
    slstm: ScalarLSTMState
    hidden: Tensor                                 # [768]
    previous_action: Tensor = torch.tensor(0, dtype=torch.long)
    temporal_tokens: Tensor | None = None
    temporal_valid: Tensor | None = None

    @property
    def cell(self) -> Tensor:
        return self.slstm.cell.reshape(-1)

    def detached(self) -> "CompactAgentMemory":
        return CompactAgentMemory(
            mlstm=self.mlstm.detached(),
            slstm=self.slstm.detached(),
            hidden=self.hidden.detach().clone(),
            previous_action=self.previous_action.detach().clone(),
            temporal_tokens=None if self.temporal_tokens is None else self.temporal_tokens.detach().clone(),
            temporal_valid=None if self.temporal_valid is None else self.temporal_valid.detach().clone(),
        )

    def with_previous_action(self, action: int | Tensor) -> "CompactAgentMemory":
        act_tensor = action if isinstance(action, Tensor) else torch.tensor(int(action), dtype=torch.long, device=self.hidden.device)
        return replace(self, previous_action=act_tensor)


class MatrixLSTMCell(nn.Module):
    """Matrix LSTM (mLSTM) Cell with covariance memory and log-space exponential gating.
    
    Adheres strictly to Beck et al. (NeurIPS 2024):
    - C_t = f_t * C_{t-1} + i_t * (v_t @ k_t^T)
    - n_t = f_t * n_{t-1} + i_t * k_t
    - log f_t = -softplus(-f_raw) <= 0
    - m_t = max(m_{t-1} + log f_t, i_raw)
    - i_t = exp(i_raw - m_t) in (0, 1]
    - f_t = exp(log f_t + m_{t-1} - m_t) in (0, 1]
    - h_raw = (C_t @ q_t) / max(|n_t^T @ q_t|, 1.0)
    """

    def __init__(
        self,
        in_features: int = COMPACT_PRE_SIZE,
        hidden_size: int = COMPACT_HIDDEN_SIZE,
        num_heads: int = COMPACT_NUM_HEADS,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim * self.num_heads != self.hidden_size:
            raise ValueError(f"hidden_size {self.hidden_size} must be divisible by num_heads {self.num_heads}")

        # Linear projections for Query, Key, Value
        self.proj_q = nn.Linear(self.in_features, self.hidden_size, bias=False)
        self.proj_k = nn.Linear(self.in_features, self.hidden_size, bias=False)
        self.proj_v = nn.Linear(self.in_features, self.hidden_size, bias=False)

        # Gating projections: Input gate, Forget gate (per head)
        self.proj_i = nn.Linear(self.in_features, self.num_heads, bias=True)
        self.proj_f = nn.Linear(self.in_features, self.num_heads, bias=True)

        # Output gate (per dimension)
        self.proj_o = nn.Linear(self.in_features, self.hidden_size, bias=True)

        # Feedforward expansion block (Beck et al. 2024 mLSTM block)
        self.ln_ffn = nn.LayerNorm(self.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, 1536),
            nn.GELU(),
            nn.Linear(1536, self.hidden_size),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.proj_q.weight)
        nn.init.orthogonal_(self.proj_k.weight)
        nn.init.orthogonal_(self.proj_v.weight)
        nn.init.zeros_(self.proj_i.bias)
        nn.init.zeros_(self.proj_f.bias)  # b_f = 0 per Beck et al. 2024
        nn.init.zeros_(self.proj_o.bias)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> MatrixLSTMState:
        cov = torch.zeros(batch_size, self.num_heads, self.head_dim, self.head_dim, device=device, dtype=dtype)
        norm = torch.zeros(batch_size, self.num_heads, self.head_dim, device=device, dtype=dtype)
        stab = torch.zeros(batch_size, self.num_heads, device=device, dtype=dtype)
        return MatrixLSTMState(covariance=cov, normalizer=norm, stabilizer=stab)

    def forward(
        self,
        x: Tensor,  # [B, in_features]
        state: MatrixLSTMState | None = None,
    ) -> tuple[Tensor, MatrixLSTMState]:
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if state is None:
            state = self.initial_state(batch_size, device, dtype)

        # Project Q, K, V -> [B, H, d_h]
        q = self.proj_q(x).view(batch_size, self.num_heads, self.head_dim)
        # Scaled key projection: 1 / sqrt(d_h)
        k = (self.proj_k(x) / math.sqrt(self.head_dim)).view(batch_size, self.num_heads, self.head_dim)
        v = self.proj_v(x).view(batch_size, self.num_heads, self.head_dim)

        # Gating: i_raw, f_raw -> [B, H]
        i_raw = self.proj_i(x).clamp(-30.0, 30.0)
        f_raw = self.proj_f(x).clamp(-30.0, 30.0)

        # Log-space stabilization: log f_t = -softplus(-f_raw) <= 0
        log_f = -F.softplus(-f_raw)

        # Dynamic stabilizer update: m_t = max(m_{t-1} + log f_t, i_raw)
        m_prev = state.stabilizer
        m_t = torch.maximum(m_prev + log_f, i_raw).clamp(-30.0, 30.0)

        # Normalized gate coefficients in (0, 1]
        i_t = torch.exp((i_raw - m_t).clamp(-30.0, 0.0))        # [B, H]
        f_t = torch.exp((log_f + m_prev - m_t).clamp(-30.0, 0.0)) # [B, H]

        # Reshape for broadcasting with covariance [B, H, 1, 1]
        f_t_cov = f_t.unsqueeze(-1).unsqueeze(-1)
        i_t_cov = i_t.unsqueeze(-1).unsqueeze(-1)

        # Outer product: v_t @ k_t^T -> [B, H, d_h, d_h]
        vk_outer = torch.matmul(v.unsqueeze(-1), k.unsqueeze(-2))

        # Update covariance matrix: C_t = f_t * C_{t-1} + i_t * (v_t k_t^T)
        C_t = torch.nan_to_num(f_t_cov * state.covariance + i_t_cov * vk_outer, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        # Reshape for broadcasting with normalizer vector [B, H, 1]
        f_t_norm = f_t.unsqueeze(-1)
        i_t_norm = i_t.unsqueeze(-1)

        # Update normalizer vector: n_t = f_t * n_{t-1} + i_t * k_t
        n_t = torch.nan_to_num(f_t_norm * state.normalizer + i_t_norm * k, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        # State retrieval: C_t @ q_t -> [B, H, d_h]
        C_q = torch.matmul(C_t, q.unsqueeze(-1)).squeeze(-1)    # [B, H, d_h]

        # Inner product n_t^T @ q_t -> [B, H, 1]
        nq = torch.sum(n_t * q, dim=-1, keepdim=True).abs()
        denom = torch.clamp(nq, min=1.0)                        # [B, H, 1]

        h_tilde = (C_q / denom).view(batch_size, self.hidden_size)

        # Output gate: o_t = sigmoid(W_o x + b_o)
        o_t = torch.sigmoid(self.proj_o(x))
        h_t = o_t * h_tilde

        # Feedforward block residual
        h_t = h_t + self.ffn(self.ln_ffn(h_t))

        next_state = MatrixLSTMState(covariance=C_t, normalizer=n_t, stabilizer=m_t)
        return h_t, next_state


class ScalarLSTMCell(nn.Module):
    """Scalar LSTM (sLSTM) Cell with exponential gating and dynamic memory stabilization."""

    def __init__(
        self,
        in_features: int = COMPACT_HIDDEN_SIZE,
        hidden_size: int = COMPACT_HIDDEN_SIZE,
        num_heads: int = COMPACT_NUM_HEADS,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.hidden_size // self.num_heads

        # Projections per head-dim: candidate z, input gate, forget gate, output gate
        self.proj_z = nn.Linear(self.in_features, self.hidden_size, bias=True)
        self.proj_i = nn.Linear(self.in_features, self.hidden_size, bias=True)
        self.proj_f = nn.Linear(self.in_features, self.hidden_size, bias=True)
        self.proj_o = nn.Linear(self.in_features, self.hidden_size, bias=True)

        # Feedforward expansion block (Beck et al. 2024 sLSTM block)
        self.ln_ffn = nn.LayerNorm(self.hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_size, 1024),
            nn.GELU(),
            nn.Linear(1024, self.hidden_size),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.proj_z.weight)
        nn.init.zeros_(self.proj_i.bias)
        nn.init.zeros_(self.proj_f.bias)
        nn.init.zeros_(self.proj_o.bias)

    def initial_state(self, batch_size: int, device: torch.device, dtype: torch.dtype = torch.float32) -> ScalarLSTMState:
        c = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        n = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        m = torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)
        return ScalarLSTMState(cell=c, normalizer=n, stabilizer=m)

    def forward(
        self,
        x: Tensor,  # [B, in_features]
        state: ScalarLSTMState | None = None,
    ) -> tuple[Tensor, ScalarLSTMState]:
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype

        if state is None:
            state = self.initial_state(batch_size, device, dtype)

        z_raw = self.proj_z(x)
        i_raw = self.proj_i(x).clamp(-30.0, 30.0)
        f_raw = self.proj_f(x).clamp(-30.0, 30.0)

        # Log-space stabilization
        log_f = -F.softplus(-f_raw)
        m_prev = state.stabilizer
        m_t = torch.maximum(m_prev + log_f, i_raw).clamp(-30.0, 30.0)

        i_t = torch.exp((i_raw - m_t).clamp(-30.0, 0.0))
        f_t = torch.exp((log_f + m_prev - m_t).clamp(-30.0, 0.0))
        z_t = torch.tanh(z_raw)

        c_t = torch.nan_to_num(f_t * state.cell + i_t * z_t, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)
        n_t = torch.nan_to_num(f_t * state.normalizer + i_t, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        h_tilde = c_t / torch.clamp(n_t.abs(), min=1.0)
        o_t = torch.sigmoid(self.proj_o(x))
        h_t = o_t * h_tilde

        # Residual feedforward block
        h_t = h_t + self.ffn(self.ln_ffn(h_t))

        next_state = ScalarLSTMState(cell=c_t, normalizer=n_t, stabilizer=m_t)
        return h_t, next_state


class RelationalCrossAgentGraphTransformer(nn.Module):
    """5-Agent Relational Cross-Agent Graph Transformer.
    
    Includes:
    - 5x5 Role Relational Prior Matrix R (supervision, peer, conflict)
    - Adversarial conflict penalty between Agent 0 (BUY) and Agent 1 (SELL)
    - Gated Residual Network (GRN) with ReZero parameter alpha = 0.0 at initialization
    """

    def __init__(
        self,
        hidden_size: int = COMPACT_HIDDEN_SIZE,
        num_heads: int = COMPACT_NUM_HEADS,
        num_agents: int = FIVE_AGENT_COUNT,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.num_agents = int(num_agents)
        self.head_dim = self.hidden_size // self.num_heads

        # Projections for Attention
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Learnable role relation embedding matrix (5x5 -> num_heads)
        # Prior relation classes: 0=self, 1=peer, 2=conflict (BUY vs SELL), 3=governor_supervision
        self.relation_types = nn.Parameter(self._create_relational_prior_matrix(), requires_grad=False)
        self.relation_embeddings = nn.Embedding(4, self.num_heads)

        # Conflict penalty coefficient
        self.conflict_penalty = nn.Parameter(torch.tensor(0.50, dtype=torch.float32))

        # Gated Residual Network (GRN)
        self.grn_dense1 = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.grn_gate = nn.Linear(self.hidden_size, self.hidden_size * 2)
        self.grn_dense2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.layer_norm = nn.LayerNorm(self.hidden_size)

        # ReZero scale parameter: strictly initialized to 0.0
        self.rezero_alpha = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def _create_relational_prior_matrix(self) -> Tensor:
        # 5 agents: 0: BUY, 1: SELL, 2: RISK_BUY, 3: RISK_SELL, 4: GOVERNOR
        mat = torch.ones(self.num_agents, self.num_agents, dtype=torch.long)  # Default: 1 (peer)
        # Diagonal: self (0)
        for i in range(self.num_agents):
            mat[i, i] = 0
        # Conflict: 0 vs 1 (2)
        mat[0, 1] = 2
        mat[1, 0] = 2
        # Governor supervision: Agent 4 oversees all (3)
        for i in range(self.num_agents - 1):
            mat[4, i] = 3
            mat[i, 4] = 3
        return mat

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for 5 agents.
        
        Args:
            x: [num_agents, hidden_size] -> [5, 768]
        Returns:
            out: [num_agents, hidden_size]
            attn_weights: [num_heads, num_agents, num_agents]
        """
        num_agents = x.shape[0]

        q = self.q_proj(x).view(num_agents, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d_h]
        k = self.k_proj(x).view(num_agents, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d_h]
        v = self.v_proj(x).view(num_agents, self.num_heads, self.head_dim).transpose(0, 1)  # [H, N, d_h]

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)  # [H, N, N]

        # Relational bias: [N, N] -> [N, N, H] -> [H, N, N]
        rel_bias = self.relation_embeddings(self.relation_types).permute(2, 0, 1)
        scores = scores + rel_bias

        # Adversarial conflict penalty on BUY (0) vs SELL (1)
        conflict_mask = (self.relation_types == 2).unsqueeze(0).to(dtype=x.dtype)  # [1, N, N]
        scores = scores - self.conflict_penalty * conflict_mask

        attn_weights = F.softmax(scores, dim=-1)  # [H, N, N]
        attn_out = torch.matmul(attn_weights, v)  # [H, N, d_h]
        attn_out = attn_out.transpose(0, 1).contiguous().view(num_agents, self.hidden_size)
        attn_out = self.out_proj(attn_out)

        # GRN (Gated Residual Network)
        gated = F.gelu(self.grn_dense1(attn_out)) * torch.sigmoid(self.grn_gate(attn_out))
        grn_out = self.grn_dense2(gated)

        # ReZero residual connection
        out = self.layer_norm(x + self.rezero_alpha * grn_out)
        return out, attn_weights


class CompactFactorizedActionHeads(nn.Module):
    """Compact Factorized Action and Value Heads for 768D hidden representation."""

    def __init__(
        self,
        hidden_size: int = COMPACT_HIDDEN_SIZE,
        action_count: int = ACTION_COUNT,
        five_agent_count: int = FIVE_AGENT_COUNT,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.action_count = int(action_count)
        self.five_agent_count = int(five_agent_count)

        # Role-specific calibration embeddings & network
        self.role_embedding = nn.Embedding(self.five_agent_count, self.hidden_size)
        self.role_calibrator = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.SiLU(),
            nn.Linear(256, self.hidden_size),
        )

        self.action_query = nn.Linear(self.hidden_size, 128)
        self.action_embedding = nn.Embedding(self.action_count, 128)

        # Geometry & continuous parameters
        self.geometry_net = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, OUTPUT_COUNT - ACTION_COUNT),  # 8 continuous slots
        )

        # Critic Value Head
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        # Factorized parameter heads for discrete execution bins
        self.parameter_heads = nn.ModuleDict({
            "delay": nn.Linear(self.hidden_size, PARAMETER_BINS["delay"]),
            "risk": nn.Linear(self.hidden_size, PARAMETER_BINS["risk"]),
            "entry_offset": nn.Linear(self.hidden_size, PARAMETER_BINS["entry_offset"]),
            "sl": nn.Linear(self.hidden_size, PARAMETER_BINS["sl"]),
            "tp": nn.Linear(self.hidden_size, PARAMETER_BINS["tp"]),
            "trailing": nn.Linear(self.hidden_size, PARAMETER_BINS["trailing"]),
            "manage_mode": nn.Linear(self.hidden_size, PARAMETER_BINS["manage_mode"]),
            "close_fraction": nn.Linear(self.hidden_size, PARAMETER_BINS["close_fraction"]),
        })

    def primary_action_logits(self, hidden: Tensor, agent_ids: Tensor) -> Tensor:
        calibrated = hidden + self.role_embedding(agent_ids)
        calibrated = calibrated + self.role_calibrator(calibrated)
        queries = self.action_query(calibrated)  # [5, 128]
        logits = torch.matmul(queries, self.action_embedding.weight.t())  # [5, 12]
        return logits

    def continuous_parameters(self, hidden: Tensor) -> Tensor:
        return self.geometry_net(hidden)  # [5, 8]

    def value(self, hidden: Tensor) -> Tensor:
        return self.value_head(hidden)  # [5, 1]

    def apply_internal_policy_state(
        self, action_logits: Tensor, hidden: Tensor
    ) -> tuple[Tensor, Tensor]:
        return action_logits, hidden

    def apply_codex_action_residual(
        self, action_logits: Tensor, hidden: Tensor
    ) -> Tensor:
        return action_logits

    def direct_action_logits(
        self, hidden: Tensor, agent_ids: Tensor, observations: Tensor | None = None
    ) -> Tensor:
        return self.primary_action_logits(hidden, agent_ids)

    def target_logits(
        self,
        hidden: Tensor,
        position_embeddings: Tensor | None = None,
        position_valid: Tensor | None = None,
        chosen_action: Tensor | None = None,
    ) -> Tensor:
        device = hidden.device
        batch_shape = hidden.shape[:-1]
        return torch.zeros((*batch_shape, MAX_POSITION_SLOTS), dtype=torch.float32, device=device)


class CompactObservationEncoder(nn.Module):
    """SGT-V193-Compact Observation Encoder with integrated Five Pattern Discovery."""

    def __init__(self, encoder: nn.Module, pattern_encoder: nn.Module | None = None) -> None:
        super().__init__()
        self.encoder = encoder
        if pattern_encoder is None:
            from .five_shared_policy import FiveObservationEncoder
            pattern_encoder = FiveObservationEncoder()
        self.pattern_encoder = pattern_encoder

    def forward(self, observations: Tensor, *args, **kwargs) -> Tensor:
        return self.encoder(observations)

    def pattern_predictions(self, observations: Tensor) -> dict[str, Tensor]:
        return self.pattern_encoder.pattern_predictions(observations)

    def encode_components(
        self, observations: Tensor, agent_ids: Tensor, previous_actions: Tensor
    ) -> dict[str, Tensor]:
        return self.pattern_encoder.encode_components(observations, agent_ids, previous_actions)


class SGTCompactTradingPolicy(nn.Module):
    """SGT-V193-Compact Trading Policy (~17.5M trainable parameters).
    
    Architecture:
    Observation (137) -> Observation Encoder -> 1024D ->
    Layer 1: mLSTMCell (768D, 4 heads, 147k matrix memory) ->
    Layer 2: sLSTMCell (768D, 4 heads, stabilized scalar memory) ->
    Relational Graph Transformer (768D, 5 agents) ->
    Factorized Action & Value Heads -> Action Wire (20) + Value (1).
    """

    def __init__(
        self,
        *,
        input_count: int = INPUT_COUNT,
        output_count: int = OUTPUT_COUNT,
        pre_lstm_size: int = COMPACT_PRE_SIZE,
        hidden_size: int = COMPACT_HIDDEN_SIZE,
    ) -> None:
        super().__init__()
        self.input_count = int(input_count)
        self.output_count = int(output_count)
        self.pre_lstm_size = int(pre_lstm_size)
        self.hidden_size = int(hidden_size)

        # Observation Encoder: 137 floats -> 1024D with multi-stage feature refinement
        self.observation_encoder = nn.Sequential(
            nn.Linear(self.input_count, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, self.pre_lstm_size),
            nn.LayerNorm(self.pre_lstm_size),
            nn.SiLU(),
            nn.Linear(self.pre_lstm_size, self.pre_lstm_size),
            nn.LayerNorm(self.pre_lstm_size),
            nn.SiLU(),
        )

        # 2-Layer xLSTM Core
        self.mlstm = MatrixLSTMCell(in_features=self.pre_lstm_size, hidden_size=self.hidden_size)
        self.slstm = ScalarLSTMCell(in_features=self.hidden_size, hidden_size=self.hidden_size)

        # Relational Cross-Agent Graph Transformer
        self.graph_transformer = RelationalCrossAgentGraphTransformer(hidden_size=self.hidden_size)

        # Factorized Action & Value Heads
        self.heads = CompactFactorizedActionHeads(hidden_size=self.hidden_size)

        # Architectural compatibility tags
        self.policy_architecture_version = 2
        self.reasoning_schema_version = 1
        self.reasoning_rl_enabled = False
        self.reasoning_steps = 3
        self.reasoning_action_scale = 0.0
        self.reasoning_action_residual_clip = 0.25

    def require_current_architecture(self) -> None:
        """Compatibility assertion and dynamic component binding for Five runtime."""
        if not hasattr(self.observation_encoder, "pattern_predictions"):
            from .five_shared_policy import FiveObservationEncoder
            ref = next(self.parameters())
            pattern_encoder = FiveObservationEncoder().to(device=ref.device, dtype=ref.dtype)
            self.observation_encoder = CompactObservationEncoder(
                encoder=self.observation_encoder,
                pattern_encoder=pattern_encoder,
            )

    def reset_memories(
        self,
        memories: Sequence[CompactAgentMemory],
        done_mask: Tensor | np.ndarray | Sequence[bool] | None = None,
    ) -> list[CompactAgentMemory]:
        """Reset recurrent state for finished agent trajectories."""
        if len(memories) != FIVE_AGENT_COUNT:
            raise ValueError(f"expected {FIVE_AGENT_COUNT} recurrent memories")
        device = next(self.parameters()).device
        if done_mask is None:
            mask = torch.ones(FIVE_AGENT_COUNT, dtype=torch.bool, device=device)
        else:
            mask = torch.as_tensor(done_mask, dtype=torch.bool, device=device).reshape(-1)
        fresh = self.initial_memories(FIVE_AGENT_COUNT, device=device)
        out: list[CompactAgentMemory] = []
        for i, memory in enumerate(memories):
            if bool(mask[i].item()):
                out.append(fresh[i])
            else:
                out.append(memory.detached())
        return out

    def advance_history(
        self,
        observations: Tensor,
        memories: Sequence[CompactAgentMemory] | None = None,
        *,
        actions: Tensor | None = None,
        dones: Tensor | None = None,
    ) -> list[CompactAgentMemory]:
        """Replay past observations through the recurrent xLSTM state without running action heads."""
        device = observations.device
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        steps = int(observations.shape[0])
        current = list(memories) if memories is not None else self.initial_memories(FIVE_AGENT_COUNT, device=device)
        for t in range(steps):
            obs_step = observations[t]
            if obs_step.ndim == 1:
                obs_step = obs_step.unsqueeze(0).repeat(FIVE_AGENT_COUNT, 1)
            with torch.no_grad():
                _, _, _, current = self.forward(obs_step, memories=current)
            if dones is not None:
                mask = dones[t]
                current = self.reset_memories(current, mask)
            if actions is not None:
                for i in range(FIVE_AGENT_COUNT):
                    current[i] = current[i].with_previous_action(actions[t, i])
        return current

    @staticmethod
    def _bin_to_logit(index: int, bins: int) -> float:
        if bins <= 1:
            return 0.0
        p = (float(index) + 0.5) / float(bins)
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return float(math.log(p / (1.0 - p)))

    def forward_five(
        self,
        observations: Tensor,
        memories: Sequence[CompactAgentMemory],
        *,
        allowed_action_masks: Tensor | None = None,
        deterministic: bool = False,
        behavior_uniform_mix: float | Tensor = 0.0,
        behavior_entry_probability_floor: float | None = None,
        sampling_generator: torch.Generator | None = None,
    ):
        """Full Five-agent forward step producing FiveActionResult and 20-slot wire vector."""
        from .five_shared_policy import (
            FiveActionResult,
            SharedFiveTradingPolicy,
            pack_geometry_choices,
            TARGETED_MANAGEMENT_ACTIONS,
        )

        self.require_current_architecture()
        device = observations.device
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        if observations.shape != (FIVE_AGENT_COUNT, INPUT_COUNT):
            raise ValueError(f"expected observation matrix ({FIVE_AGENT_COUNT},{INPUT_COUNT})")
        if len(memories) != FIVE_AGENT_COUNT:
            raise ValueError(f"expected {FIVE_AGENT_COUNT} recurrent memories")

        # Step full SGT-Compact forward pass
        action_wire, values, extra, next_memories = self.forward(observations, memories=memories)
        hidden = extra["hidden"]
        action_logits = torch.nan_to_num(extra["action_logits"], nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)

        # Handle allowed masks
        if allowed_action_masks is not None:
            allowed = allowed_action_masks.to(device=device, dtype=torch.bool)
        else:
            allowed = torch.ones_like(action_logits, dtype=torch.bool)
        hold = torch.zeros_like(allowed)
        hold[:, 0] = True
        allowed = torch.where(allowed.any(dim=-1, keepdim=True), allowed, hold)

        # Primary action sampling using standard Five distribution logic
        actions, action_log_prob, action_entropy, policy_action_probs, behavior_action_probs = (
            SharedFiveTradingPolicy._primary_action_distribution(
                action_logits,
                allowed=allowed,
                deterministic=deterministic,
                entry_probability_floor=(
                    0.005 if behavior_entry_probability_floor is None else float(behavior_entry_probability_floor)
                ),
                uniform_mix=behavior_uniform_mix,
                include_policy_probs=True,
                sampling_generator=sampling_generator,
            )
        )

        # Discrete parameter heads
        param_choices = {}
        param_lps = {}
        param_ents = {}
        for name, head in self.heads.parameter_heads.items():
            p_logits = head(hidden)
            choice, lp, ent = SharedFiveTradingPolicy._distribution(p_logits, deterministic, sampling_generator)
            param_choices[name] = choice
            param_lps[name] = lp
            param_ents[name] = ent

        results: list[FiveActionResult] = []
        updated_memories: list[CompactAgentMemory] = []

        for i in range(FIVE_AGENT_COUNT):
            act = int(actions[i].item())
            delay = int(param_choices["delay"][i].item())
            risk = int(param_choices["risk"][i].item())
            entry_offset = int(param_choices["entry_offset"][i].item())
            sl = int(param_choices["sl"][i].item())
            tp = int(param_choices["tp"][i].item())
            trailing = int(param_choices["trailing"][i].item())
            manage_mode = int(param_choices["manage_mode"][i].item())
            close_frac = int(param_choices["close_fraction"][i].item())

            output = np.zeros(OUTPUT_COUNT, dtype=np.float32)
            output[:ACTION_COUNT] = action_logits[i].detach().cpu().numpy().astype(np.float32)
            output[act] = np.float32(max(float(np.max(output[:ACTION_COUNT])), float(output[act])) + 1.0)
            output[RISK_OUTPUT_INDEX] = np.float32(self._bin_to_logit(risk, PARAMETER_BINS["risk"]))
            output[ACTION_DELAY_OUTPUT_INDEX] = np.float32(self._bin_to_logit(delay, PARAMETER_BINS["delay"]))
            output[FACTORIZED_ACTION_MARKER_OUTPUT_INDEX] = np.float32(FACTORIZED_ACTION_MARKER)
            output[ENTRY_OFFSET_OUTPUT_INDEX] = np.float32(self._bin_to_logit(entry_offset, PARAMETER_BINS["entry_offset"]))
            output[GEOMETRY_PACKED_OUTPUT_INDEX] = np.float32(pack_geometry_choices(sl, tp, trailing))
            output[PARTIAL_CLOSE_OUTPUT_INDEX] = np.float32(self._bin_to_logit(close_frac, PARAMETER_BINS["close_fraction"]))
            output[MANAGEMENT_MODE_OUTPUT_INDEX] = np.float32(self._bin_to_logit(manage_mode, PARAMETER_BINS["manage_mode"]))
            output[TARGET_SLOT_OUTPUT_INDEX] = np.float32(self._bin_to_logit(0, MAX_POSITION_SLOTS))

            head_lps = {"action": float(action_log_prob[i].item()), "delay": float(param_lps["delay"][i].item())}
            head_ents = {"action": float(action_entropy[i].item()), "delay": float(param_ents["delay"][i].item())}
            total_lp = action_log_prob[i] + param_lps["delay"][i]
            total_entropy = action_entropy[i] + param_ents["delay"][i]
            if act in ENTRY_ACTIONS:
                for name in ("risk", "entry_offset", "sl", "tp", "trailing"):
                    total_lp = total_lp + param_lps[name][i]
                    total_entropy = total_entropy + param_ents[name][i]
                    head_lps[name] = float(param_lps[name][i].item())
                    head_ents[name] = float(param_ents[name][i].item())

            h_prev = memories[i].hidden.detach().cpu().numpy().astype(np.float32)
            h_next = hidden[i].detach().cpu().numpy().astype(np.float32)
            c_prev = memories[i].cell.detach().cpu().numpy().astype(np.float32)
            c_next = next_memories[i].cell.detach().cpu().numpy().astype(np.float32)

            results.append(
                FiveActionResult(
                    action_index=act,
                    target_index=0,
                    delay_index=delay,
                    risk_index=risk,
                    entry_offset_index=entry_offset,
                    sl_index=sl,
                    tp_index=tp,
                    trailing_index=trailing,
                    manage_mode_index=manage_mode,
                    close_fraction_index=close_frac,
                    output_vector=output,
                    log_probability=float(total_lp.item()),
                    entropy=float(total_entropy.item()),
                    value=float(values[i].item()),
                    action_probabilities=behavior_action_probs[i].detach().cpu().numpy().astype(np.float32),
                    policy_action_probabilities=policy_action_probs[i].detach().cpu().numpy().astype(np.float32),
                    behavior_action_probabilities=behavior_action_probs[i].detach().cpu().numpy().astype(np.float32),
                    policy_action_entropy=float(action_entropy[i].item()),
                    behavior_action_entropy=float(action_entropy[i].item()),
                    encoded_observation=extra["encoded"][i].detach().cpu().numpy().astype(np.float32),
                    hidden_before=h_prev,
                    hidden_after=h_next,
                    cell_before=c_prev,
                    cell_after=c_next,
                    head_log_probabilities=head_lps,
                    head_entropies=head_ents,
                )
            )
            updated_memories.append(next_memories[i].with_previous_action(act))

        return results, updated_memories, extra

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initial_memories(self, batch_size: int = FIVE_AGENT_COUNT, device: torch.device | None = None) -> list[CompactAgentMemory]:
        if device is None:
            device = next(self.parameters()).device
        mlstm_init = self.mlstm.initial_state(batch_size, device)
        slstm_init = self.slstm.initial_state(batch_size, device)
        h_init = torch.zeros(batch_size, self.hidden_size, device=device)

        memories = []
        for i in range(batch_size):
            memories.append(
                CompactAgentMemory(
                    mlstm=MatrixLSTMState(
                        covariance=mlstm_init.covariance[i],
                        normalizer=mlstm_init.normalizer[i],
                        stabilizer=mlstm_init.stabilizer[i],
                    ),
                    slstm=ScalarLSTMState(
                        cell=slstm_init.cell[i],
                        normalizer=slstm_init.normalizer[i],
                        stabilizer=slstm_init.stabilizer[i],
                    ),
                    hidden=h_init[i],
                )
            )
        return memories

    def forward(
        self,
        observations: Tensor,  # [5, 137]
        agent_ids: Tensor | None = None,
        memories: Sequence[CompactAgentMemory] | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor], list[CompactAgentMemory]]:
        """Forward pass for 5-agent shared policy.
        
        Returns:
            action_wire: [5, 20]
            values: [5, 1]
            extra: dict of intermediate activations
            next_memories: updated agent memories
        """
        device = observations.device
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        dtype = observations.dtype
        batch_size = observations.shape[0]

        if agent_ids is None:
            agent_ids = torch.arange(batch_size, dtype=torch.long, device=device)

        if memories is None:
            memories = self.initial_memories(batch_size, device)

        # 1. Observation encoding: [5, 137] -> [5, 1024]
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        encoded = self.observation_encoder(observations)

        # Reconstruct batched state for mLSTM and sLSTM
        cov_batch = torch.stack([m.mlstm.covariance for m in memories], dim=0)
        norm_batch = torch.stack([m.mlstm.normalizer for m in memories], dim=0)
        stab_batch = torch.stack([m.mlstm.stabilizer for m in memories], dim=0)
        mlstm_state = MatrixLSTMState(covariance=cov_batch, normalizer=norm_batch, stabilizer=stab_batch)

        # Layer 1: mLSTM forward
        h1, next_mlstm_state = self.mlstm(encoded, mlstm_state)

        # Reconstruct sLSTM state
        cell_batch = torch.stack([m.slstm.cell for m in memories], dim=0)
        snorm_batch = torch.stack([m.slstm.normalizer for m in memories], dim=0)
        sstab_batch = torch.stack([m.slstm.stabilizer for m in memories], dim=0)
        slstm_state = ScalarLSTMState(cell=cell_batch, normalizer=snorm_batch, stabilizer=sstab_batch)

        # Layer 2: sLSTM forward + residual connection
        h2_delta, next_slstm_state = self.slstm(h1, slstm_state)
        h2 = h1 + h2_delta

        # Layer 3: Relational Graph Transformer across 5 agents
        hidden, attn_weights = self.graph_transformer(h2)

        # Action and Value heads
        action_logits = self.heads.primary_action_logits(hidden, agent_ids)  # [5, 12]
        continuous_params = self.heads.continuous_parameters(hidden)         # [5, 8]
        values = self.heads.value(hidden)                                    # [5, 1]

        # Assemble [5, 20] Action Wire
        action_wire = torch.cat([action_logits, continuous_params], dim=-1)

        # Pack next memories
        next_memories = []
        for i in range(batch_size):
            next_memories.append(
                CompactAgentMemory(
                    mlstm=MatrixLSTMState(
                        covariance=next_mlstm_state.covariance[i],
                        normalizer=next_mlstm_state.normalizer[i],
                        stabilizer=next_mlstm_state.stabilizer[i],
                    ),
                    slstm=ScalarLSTMState(
                        cell=next_slstm_state.cell[i],
                        normalizer=next_slstm_state.normalizer[i],
                        stabilizer=next_slstm_state.stabilizer[i],
                    ),
                    hidden=hidden[i],
                )
            )

        extra = {
            "encoded": encoded,
            "h1": h1,
            "h2": h2,
            "hidden": hidden,
            "action_logits": action_logits,
            "attn_weights": attn_weights,
        }
        return action_wire, values, extra, next_memories

    @property
    def lstm_size(self) -> int:
        return self.hidden_size

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))

    def _prepare_initial_states(
        self, initial_memories: Sequence[Any], device: torch.device
    ) -> tuple[MatrixLSTMState, ScalarLSTMState, Tensor, Tensor]:
        num_agents = len(initial_memories)
        if any(not hasattr(m, "mlstm") for m in initial_memories):
            mlstm_state = self.mlstm.initial_state(num_agents, device)
            slstm_state = self.slstm.initial_state(num_agents, device)
        else:
            mlstm_cov = torch.stack([m.mlstm.covariance for m in initial_memories], dim=0).to(device=device)
            mlstm_norm = torch.stack([m.mlstm.normalizer for m in initial_memories], dim=0).to(device=device)
            mlstm_stab = torch.stack([m.mlstm.stabilizer for m in initial_memories], dim=0).to(device=device)
            mlstm_state = MatrixLSTMState(covariance=mlstm_cov, normalizer=mlstm_norm, stabilizer=mlstm_stab)

            slstm_cell = torch.stack([m.slstm.cell for m in initial_memories], dim=0).to(device=device)
            slstm_norm = torch.stack([m.slstm.normalizer for m in initial_memories], dim=0).to(device=device)
            slstm_stab = torch.stack([m.slstm.stabilizer for m in initial_memories], dim=0).to(device=device)
            slstm_state = ScalarLSTMState(cell=slstm_cell, normalizer=slstm_norm, stabilizer=slstm_stab)

        h = torch.stack([
            m.hidden if isinstance(m.hidden, Tensor) else torch.as_tensor(m.hidden, dtype=torch.float32, device=device)
            for m in initial_memories
        ], dim=0).to(device=device)

        prev_actions = torch.stack([
            m.previous_action if isinstance(m.previous_action, Tensor) else torch.tensor(int(m.previous_action), dtype=torch.long, device=device)
            for m in initial_memories
        ], dim=0).to(device=device)

        return mlstm_state, slstm_state, h, prev_actions

    def evaluate_sequence(
        self,
        observations: Tensor,
        initial_memories: Sequence[Any],
        *,
        actions: Tensor,
        targets: Tensor,
        delay_indices: Tensor,
        risk_indices: Tensor,
        entry_offset_indices: Tensor,
        manage_mode_indices: Tensor,
        close_fraction_indices: Tensor,
        sl_indices: Tensor | None = None,
        tp_indices: Tensor | None = None,
        trailing_indices: Tensor | None = None,
        allowed_action_masks: Tensor | None = None,
        episode_dones: Tensor | None = None,
        behavior_uniform_mixes: Tensor | None = None,
        behavior_entry_probability_floor: float | None = None,
    ) -> dict[str, Tensor]:
        """Re-evaluate one truncated recurrent sample without resampling actions.

        Shapes use [time, five, ...]. Recurrent state is carried through the
        whole sample so gradients flow through exactly the unroll window. A
        terminal marker after step t resets only that agent before t+1
        so memory never leaks between independent episodes.
        """
        from .five_shared_policy import (
            SharedFiveTradingPolicy,
            LEGACY_HEURISTIC_CLOSE_ACTIONS,
            TARGETED_MANAGEMENT_ACTIONS,
            FIVE_LSTM_UNROLL,
        )

        self.require_current_architecture()
        if observations.ndim != 3 or observations.shape[1:] != (FIVE_AGENT_COUNT, INPUT_COUNT):
            raise ValueError(f"expected observations [T,{FIVE_AGENT_COUNT},{INPUT_COUNT}]")
        steps = int(observations.shape[0])
        if steps <= 0 or steps > FIVE_LSTM_UNROLL:
            raise ValueError(f"sequence length must be in [1,{FIVE_LSTM_UNROLL}]")
        if len(initial_memories) != FIVE_AGENT_COUNT:
            raise ValueError("expected five initial memories")
        if episode_dones is not None and episode_dones.shape != (steps, FIVE_AGENT_COUNT):
            raise ValueError("episode_dones must have shape [T,5]")
        if behavior_uniform_mixes is not None and behavior_uniform_mixes.shape != (steps, FIVE_AGENT_COUNT):
            raise ValueError("behavior_uniform_mixes must have shape [T,5]")

        device = observations.device
        observations = torch.nan_to_num(observations, nan=0.0, posinf=0.0, neginf=0.0)
        if sl_indices is None:
            sl_indices = torch.full((steps, FIVE_AGENT_COUNT), DEFAULT_SL_BIN_INDEX, dtype=torch.long, device=device)
        if tp_indices is None:
            tp_indices = torch.full((steps, FIVE_AGENT_COUNT), DEFAULT_TP_BIN_INDEX, dtype=torch.long, device=device)
        if trailing_indices is None:
            trailing_indices = torch.full((steps, FIVE_AGENT_COUNT), DEFAULT_TRAILING_BIN_INDEX, dtype=torch.long, device=device)

        mlstm_state, slstm_state, h, previous_actions = self._prepare_initial_states(initial_memories, device)
        agent_ids = torch.arange(FIVE_AGENT_COUNT, dtype=torch.long, device=device)

        log_probs: list[Tensor] = []
        policy_log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        primary_action_log_probs: list[Tensor] = []
        policy_action_log_probs: list[Tensor] = []
        primary_action_entropies: list[Tensor] = []
        primary_entry_probability_mass: list[Tensor] = []
        primary_action_probabilities: list[Tensor] = []
        raw_action_logits_seq: list[Tensor] = []
        policy_action_entropies: list[Tensor] = []
        policy_action_probabilities: list[Tensor] = []
        values: list[Tensor] = []
        hidden_seq: list[Tensor] = []
        policy_hidden_seq: list[Tensor] = []
        internal_policy_state_seq: list[Tensor] = []
        cell_seq: list[Tensor] = []

        for t in range(steps):
            if t > 0 and episode_dones is not None:
                ended = episode_dones[t - 1].to(device=device, dtype=torch.bool)
                if bool(ended.any().item()):
                    # Boundary reset for finished agents
                    ended_4d = ended.view(FIVE_AGENT_COUNT, 1, 1, 1)
                    ended_3d = ended.view(FIVE_AGENT_COUNT, 1, 1)
                    ended_2d = ended.view(FIVE_AGENT_COUNT, 1)

                    fresh_mlstm = self.mlstm.initial_state(FIVE_AGENT_COUNT, device)
                    fresh_slstm = self.slstm.initial_state(FIVE_AGENT_COUNT, device)

                    mlstm_state = MatrixLSTMState(
                        covariance=torch.where(ended_4d, fresh_mlstm.covariance, mlstm_state.covariance),
                        normalizer=torch.where(ended_3d, fresh_mlstm.normalizer, mlstm_state.normalizer),
                        stabilizer=torch.where(ended_2d, fresh_mlstm.stabilizer, mlstm_state.stabilizer),
                    )
                    slstm_state = ScalarLSTMState(
                        cell=torch.where(ended_2d, fresh_slstm.cell, slstm_state.cell),
                        normalizer=torch.where(ended_2d, fresh_slstm.normalizer, slstm_state.normalizer),
                        stabilizer=torch.where(ended_2d, fresh_slstm.stabilizer, slstm_state.stabilizer),
                    )
                    h = torch.where(ended_2d, torch.zeros_like(h), h)
                    previous_actions = torch.where(ended, torch.zeros_like(previous_actions), previous_actions)

            # 1. Observation encoding
            encoded = self.observation_encoder(observations[t])

            # 2. mLSTM layer
            h1, mlstm_state = self.mlstm(encoded, mlstm_state)

            # 3. sLSTM layer + residual
            h2_delta, slstm_state = self.slstm(h1, slstm_state)
            h2 = h1 + h2_delta

            # 4. Cross-agent Relational Graph Transformer
            policy_hidden, _ = self.graph_transformer(h2)

            # 5. Heads
            action_logits = self.heads.primary_action_logits(policy_hidden, agent_ids)
            value_step = self.heads.value(policy_hidden).squeeze(-1)

            # Allowed action masking
            if allowed_action_masks is not None:
                allowed = allowed_action_masks[t].to(device=device, dtype=torch.bool)
                hold = torch.zeros_like(allowed)
                hold[:, 0] = True
                allowed = torch.where(allowed.any(dim=-1, keepdim=True), allowed, hold)
                for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                    action_logits[:, legacy_action] = torch.where(
                        allowed[:, legacy_action],
                        action_logits[:, legacy_action],
                        torch.full_like(action_logits[:, legacy_action], -30.0),
                    )
            else:
                allowed = torch.ones_like(action_logits, dtype=torch.bool)
                for legacy_action in LEGACY_HEURISTIC_CLOSE_ACTIONS:
                    action_logits[:, legacy_action] = -30.0
                    allowed[:, legacy_action] = False

            raw_action_logits_seq.append(action_logits)

            _, _, _, policy_probs, action_probs = SharedFiveTradingPolicy._primary_action_distribution(
                action_logits,
                allowed=allowed,
                deterministic=False,
                entry_probability_floor=(
                    0.005 if behavior_entry_probability_floor is None else float(behavior_entry_probability_floor)
                ),
                uniform_mix=(
                    0.0 if behavior_uniform_mixes is None else behavior_uniform_mixes[t]
                ),
                include_policy_probs=True,
            )

            # Defensive guard ensuring Simplex constraint is always satisfied
            action_probs = torch.nan_to_num(action_probs, nan=0.0)
            act_sum = action_probs.sum(dim=-1, keepdim=True)
            action_probs = torch.where(
                act_sum > 1e-12,
                action_probs / act_sum.clamp_min(1e-12),
                allowed.to(dtype=action_probs.dtype) / allowed.sum(dim=-1, keepdim=True).clamp_min(1.0),
            )

            policy_probs = torch.nan_to_num(policy_probs, nan=0.0)
            pol_sum = policy_probs.sum(dim=-1, keepdim=True)
            policy_probs = torch.where(
                pol_sum > 1e-12,
                policy_probs / pol_sum.clamp_min(1e-12),
                allowed.to(dtype=policy_probs.dtype) / allowed.sum(dim=-1, keepdim=True).clamp_min(1.0),
            )

            action_dist = torch.distributions.Categorical(probs=action_probs)
            policy_dist = torch.distributions.Categorical(probs=policy_probs)
            chosen_action = actions[t].to(device=device, dtype=torch.long)

            total_lp = action_dist.log_prob(chosen_action)
            policy_total_lp = policy_dist.log_prob(chosen_action)
            primary_action_log_probs.append(total_lp)
            policy_action_log_probs.append(policy_total_lp)
            primary_action_entropies.append(action_dist.entropy())
            primary_entry_probability_mass.append(action_dist.probs[:, 1:7].sum(dim=-1))
            primary_action_probabilities.append(action_dist.probs)

            total_ent = policy_dist.entropy()
            policy_action_entropies.append(policy_dist.entropy())
            policy_action_probabilities.append(policy_probs)

            # Discrete parameter heads
            selected_parameter_indices = {
                "delay": delay_indices[t],
                "risk": risk_indices[t],
                "entry_offset": entry_offset_indices[t],
                "sl": sl_indices[t],
                "tp": tp_indices[t],
                "trailing": trailing_indices[t],
                "manage_mode": manage_mode_indices[t],
                "close_fraction": close_fraction_indices[t],
            }
            target_active = torch.as_tensor(
                [int(a.item()) in TARGETED_MANAGEMENT_ACTIONS for a in chosen_action],
                dtype=torch.bool,
                device=device,
            )
            for name, head in self.heads.parameter_heads.items():
                logits = head(policy_hidden)
                dist = torch.distributions.Categorical(logits=logits)
                choice = selected_parameter_indices[name].to(device=device, dtype=torch.long)
                if name == "delay":
                    active = torch.ones(FIVE_AGENT_COUNT, dtype=torch.bool, device=device)
                elif name in {"risk", "entry_offset", "sl", "tp", "trailing"}:
                    active = torch.as_tensor(
                        [int(a.item()) in ENTRY_ACTIONS for a in chosen_action],
                        dtype=torch.bool,
                        device=device,
                    )
                elif name == "manage_mode":
                    active = target_active
                else:  # close_fraction
                    manage_choice = manage_mode_indices[t].to(device=device, dtype=torch.long)
                    active = target_active & (manage_choice == 1)

                total_lp = total_lp + torch.where(active, dist.log_prob(choice), torch.zeros_like(total_lp))
                policy_total_lp = policy_total_lp + torch.where(
                    active, dist.log_prob(choice), torch.zeros_like(policy_total_lp)
                )
                total_ent = total_ent + torch.where(active, dist.entropy(), torch.zeros_like(total_ent))

            log_probs.append(total_lp)
            policy_log_probs.append(policy_total_lp)
            entropies.append(total_ent)
            values.append(value_step)
            hidden_seq.append(policy_hidden)
            policy_hidden_seq.append(policy_hidden)
            internal_policy_state_seq.append(policy_hidden)
            cell_seq.append(slstm_state.cell.reshape(FIVE_AGENT_COUNT, -1))
            previous_actions = chosen_action

        return {
            "log_probability": torch.stack(log_probs, dim=0),
            "policy_log_probability": torch.stack(policy_log_probs, dim=0),
            "entropy": torch.stack(entropies, dim=0),
            "primary_action_log_probability": torch.stack(primary_action_log_probs, dim=0),
            "policy_action_log_probability": torch.stack(policy_action_log_probs, dim=0),
            "primary_action_entropy": torch.stack(primary_action_entropies, dim=0),
            "primary_entry_probability_mass": torch.stack(primary_entry_probability_mass, dim=0),
            "primary_action_probabilities": torch.stack(primary_action_probabilities, dim=0),
            "raw_action_logits": torch.stack(raw_action_logits_seq, dim=0),
            "policy_action_entropy": torch.stack(policy_action_entropies, dim=0),
            "policy_action_probabilities": torch.stack(policy_action_probabilities, dim=0),
            "value": torch.stack(values, dim=0),
            "hidden": torch.stack(hidden_seq, dim=0),
            "policy_hidden": torch.stack(policy_hidden_seq, dim=0),
            "internal_policy_state": torch.stack(internal_policy_state_seq, dim=0),
            "cell": torch.stack(cell_seq, dim=0),
        }
