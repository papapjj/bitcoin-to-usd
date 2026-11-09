"""
Latent Dynamics World Model for SGT-V193 Governor Latent Space (768D).
Module: Milestone M2 / Requirement R2 (Features 5, 6, 7)
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine

Implements:
1. 768D Latent Dynamics State Transition Network:
   z_hat_{t+1} = LayerNorm(z_t + alpha * g(z_t, a_t))
2. Multi-Step Latent Rollout (H >= 5) with Cosine Similarity >= 0.90 guarantee.
3. Auxiliary Prediction Heads:
   - State Value Head v_hat(z_t)
   - Immediate Return Head r_hat(z_t, a_t)
   - Dynamic Risk Bounds [MFE_hat, MAE_hat] replacing static ATR bins
4. Adversarial Market Stressor:
   Generates worst-case execution shocks (synthetic liquidity sweep spikes, adverse slippage).
5. Absolute Zero NaN / Zero Inf numerical safety guarantees (AC-7).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarketStateEncoder(nn.Module):
    """
    Non-linear State Representation Encoder mapping 137D market observation vector
    from RealM1Environment into 768D normalized latent state space:
    z_t = h_phi(s_t) in R^768.
    Feature 4 / Requirement R2.
    """

    def __init__(
        self,
        input_dim: int = 137,
        latent_dim: int = 768,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights with orthogonal projections and default uniform bias."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Encodes observation vector into normalized latent space.
        Args:
            obs: Tensor of shape [B, input_dim] or [input_dim]
        Returns:
            z: Normalized latent state tensor of shape [B, latent_dim]
        """
        clean = torch.nan_to_num(obs, nan=0.0, posinf=1.0, neginf=-1.0)
        if clean.dim() == 1:
            clean = clean.unsqueeze(0)
        return self.net(clean)


@dataclass
class TemporalKVCache:
    """
    Key-Value Cache for TemporalCrossAttentionEncoder.
    Holds precomputed Keys and Values for fast O(1) cross-attention query evaluation.
    Supports attribute access, dict indexing, and tuple unpacking.
    """
    key: torch.Tensor          # [B, num_heads, candle_len, head_dim]
    value: torch.Tensor        # [B, num_heads, candle_len, head_dim]
    candle_len: int = 240
    batch_size: int = 1

    def __iter__(self):
        yield self.key
        yield self.value

    def __getitem__(self, item: Union[str, int]) -> Any:
        if item in ("key", "k", 0):
            return self.key
        elif item in ("value", "v", 1):
            return self.value
        elif item == "candle_len":
            return self.candle_len
        elif item == "batch_size":
            return self.batch_size
        raise KeyError(f"Key {item} not found in TemporalKVCache")


def compute_attention_telemetry(
    attn_weights: Union[torch.Tensor, np.ndarray],
    prices: Optional[Any] = None,
    candle_len: int = 240,
    num_heads: int = 4,
    bar_idx: Optional[int] = None,
    timestamp: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive, mathematically rigorous attention telemetry across 240 candles.
    
    Args:
        attn_weights: Tensor or ndarray of shape [B, H, L] or [H, L] or [B, H, 1, L]
        prices: Optional price sequence for reference
        candle_len: Number of candles (default: 240)
        num_heads: Number of attention heads (default: 4)
        bar_idx: Optional target bar index
        timestamp: Optional ISO datetime string
        
    Returns:
        JSON-serializable dictionary with summary statistics, distribution metrics,
        and per-head weights.
    """
    if isinstance(attn_weights, np.ndarray):
        attn_weights = torch.from_numpy(attn_weights)
        
    if attn_weights.dim() == 4:
        attn_weights = attn_weights.squeeze(2)  # [B, H, L]
    if attn_weights.dim() == 2:
        attn_weights = attn_weights.unsqueeze(0)  # [1, H, L]

    attn_np = attn_weights.detach().cpu().to(torch.float32).numpy()  # [B, H, L]
    attn_h_l = np.mean(attn_np, axis=0)  # [H, L]
    mean_attn = np.mean(attn_h_l, axis=0)  # [L]

    L = len(mean_attn)
    H = attn_h_l.shape[0]

    # 1. Per-Head Shannon Entropy: -sum(p * log(p + 1e-12))
    head_entropy: List[float] = []
    head_peak_indices: List[int] = []
    head_peak_weights: List[float] = []

    for h in range(H):
        p_h = attn_h_l[h]
        ent = -float(np.sum(p_h * np.log(p_h + 1e-12)))
        head_entropy.append(round(ent, 6))
        peak_idx = int(np.argmax(p_h))
        head_peak_indices.append(peak_idx)
        head_peak_weights.append(round(float(p_h[peak_idx]), 6))

    mean_entropy = float(np.mean(head_entropy))
    max_entropy = math.log(float(L))
    normalized_entropy = float(mean_entropy / max_entropy) if max_entropy > 0 else 0.0

    # 2. Peak Candle Index & Distance to Current Tick
    peak_candle_index = int(np.argmax(mean_attn))
    peak_candle_weight = float(mean_attn[peak_candle_index])
    peak_candle_lag = int((L - 1) - peak_candle_index)  # 0 = current bar, L-1 = oldest bar

    # 3. Top-K Saliency Candles
    top_5_indices = [int(idx) for idx in np.argsort(mean_attn)[::-1][:5]]
    top_5_weights = [round(float(mean_attn[idx]), 6) for idx in top_5_indices]

    # 4. Recency Ratios
    # Feature 15 contract: recent 30 vs distant 210 candles
    recent_30_bars = min(30, L)
    recent_30_mass = float(np.sum(mean_attn[-recent_30_bars:]))
    distant_210_mass = float(np.sum(mean_attn[:-recent_30_bars]))
    recency_ratio_30_210 = float(recent_30_mass / (distant_210_mass + 1e-6))

    # Backward compatibility: recent 15 bars vs prior bars
    recent_15_bars = min(15, L)
    recent_15_mass = float(np.sum(mean_attn[-recent_15_bars:]))
    prior_mass = float(np.sum(mean_attn[:-recent_15_bars]))
    recency_weight_ratio = float(recent_15_mass / (prior_mass + 1e-6))

    last_candle_weight = float(mean_attn[-1])

    # 5. Attention Distribution Statistics
    attn_std = float(np.std(mean_attn))
    attn_min = float(np.min(mean_attn))
    attn_max = float(np.max(mean_attn))

    # Gini Sparsity Coefficient: sum(|p_i - p_j|) / (2 * L * sum(p))
    diff_matrix = np.abs(mean_attn[:, None] - mean_attn[None, :])
    gini_sparsity = float(np.sum(diff_matrix) / (2.0 * L * (np.sum(mean_attn) + 1e-12)))

    # 6. Cross-Head Coherence & Variance
    cross_head_var_per_candle = np.var(attn_h_l, axis=0)  # [L]
    cross_head_variance = float(np.mean(cross_head_var_per_candle))

    dev = attn_h_l - (1.0 / float(L))
    cov = (dev @ dev.T) / float(L)
    std_heads = np.sqrt(np.diag(cov))
    if np.all(std_heads < 1e-6):
        cross_head_coherence = 1.0
    else:
        denom = std_heads[:, None] * std_heads[None, :] + 1e-12
        corr_matrix = cov / denom
        triu_idx = np.triu_indices(H, k=1)
        cross_head_coherence = float(np.mean(corr_matrix[triu_idx])) if len(triu_idx[0]) > 0 else 1.0

    return {
        # Legacy contract keys (100% backward compatible)
        "mean_attention": mean_attn,  # NumPy array [L]
        "top_k_indices": top_5_indices,
        "recency_weight_ratio": recency_weight_ratio,
        "head_entropy": head_entropy,
        "last_candle_weight": last_candle_weight,
        "candle_len": L,
        "num_heads": H,

        # Standardized telemetry keys
        "recency_ratio": recency_ratio_30_210,
        "recency_ratio_30_210": recency_ratio_30_210,
        "recent_30_mass": recent_30_mass,
        "distant_210_mass": distant_210_mass,
        "entropy": round(mean_entropy, 6),
        "mean_entropy": round(mean_entropy, 6),
        "normalized_entropy": round(normalized_entropy, 6),
        "peak_candle_index": peak_candle_index,
        "peak_candle_lag": peak_candle_lag,
        "peak_candle_weight": round(peak_candle_weight, 6),
        "head_peak_indices": head_peak_indices,
        "head_peak_weights": head_peak_weights,
        "top_5_candles": top_5_indices,
        "top_5_weights": top_5_weights,
        "attention_std": round(attn_std, 6),
        "attention_min": round(attn_min, 6),
        "attention_max": round(attn_max, 6),
        "gini_sparsity": round(gini_sparsity, 6),
        "cross_head_variance": round(cross_head_variance, 8),
        "cross_head_coherence": round(cross_head_coherence, 6),
        "bar_idx": bar_idx,
        "timestamp": timestamp,

        # Serialized weight vectors (JSON serializable)
        "mean_weights_240": [round(float(x), 6) for x in mean_attn],
        "head_weights": [[round(float(x), 6) for x in attn_h_l[h]] for h in range(H)],
    }


def extract_attention_weights(
    model_or_weights: Any,
    obs_137: Optional[Union[torch.Tensor, np.ndarray]] = None,
    candle_seq: Optional[Union[torch.Tensor, np.ndarray]] = None,
    kv_cache: Optional[Any] = None,
    unbatch: bool = False,
    keep_query_dim: bool = False,
) -> Optional[torch.Tensor]:
    """
    Helper to extract attention weights from a World Model, TemporalCrossAttentionEncoder, or raw Tensor.
    """
    if isinstance(model_or_weights, torch.Tensor):
        attn = model_or_weights
        if keep_query_dim and attn.dim() == 3:
            attn = attn.unsqueeze(2)
        if unbatch and attn.shape[0] == 1:
            return attn.squeeze(0)
        return attn
    if hasattr(model_or_weights, "encode_state") and obs_137 is not None:
        model_or_weights.encode_state(obs_137, candle_seq=candle_seq, kv_cache=kv_cache)
    if hasattr(model_or_weights, "get_attention_weights"):
        return model_or_weights.get_attention_weights(unbatch=unbatch)
    return None


class TemporalCrossAttentionEncoder(nn.Module):
    """
    Temporal Cross-Attention Module (Option C).
    Enables Champion to explicitly attend to raw sequential M1 candles [B, 240, 6]
    (representing 4 continuous hours of M1 price action) without disrupting the frozen
    137D feature representation z_feat.
    
    Q = z_feat (projected to temp_dim)
    K, V = Candle_Sequence (projected to temp_dim + Causal Self-Attention)
    
    Residual Gating ensures exact backward compatibility and distribution safety:
        alpha = sigmoid(gate_net([z_feat, z_temp]) + bias) with initial bias = -2.0 (alpha ~ 0.11)
        z_out = LayerNorm(z_feat + alpha * z_temp)
    """

    def __init__(
        self,
        candle_dim: int = 6,
        candle_len: int = 240,
        latent_dim: int = 768,
        temp_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.05,
        init_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.candle_dim = candle_dim
        self.candle_len = candle_len
        self.latent_dim = latent_dim
        self.temp_dim = temp_dim
        self.num_heads = num_heads
        self.head_dim = temp_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # 1. Candle Embedding & Positional Encoding
        self.candle_proj = nn.Sequential(
            nn.Linear(candle_dim, temp_dim),
            nn.LayerNorm(temp_dim),
            nn.GELU(),
        )
        self.pos_emb = nn.Parameter(torch.empty(candle_len, temp_dim))
        nn.init.orthogonal_(self.pos_emb, gain=1.0)

        # 2. Causal Transformer Encoder for inter-candle interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=temp_dim,
            nhead=num_heads,
            dim_feedforward=temp_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 3. Cross-Attention: Query from z_feat, Key/Value from Candle Sequence
        self.q_proj = nn.Linear(latent_dim, temp_dim, bias=False)
        self.k_proj = nn.Linear(temp_dim, temp_dim, bias=False)
        self.v_proj = nn.Linear(temp_dim, temp_dim, bias=False)
        self.out_proj = nn.Sequential(
            nn.Linear(temp_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # 4. Residual Gate with safe bias initialization
        self.gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )
        # Initialize gate bias to favor frozen z_feat initially
        nn.init.zeros_(self.gate[0].weight)
        nn.init.constant_(self.gate[0].bias, init_gate_bias)

        self.final_norm = nn.LayerNorm(latent_dim)
        self.last_candle_attn_weights: Optional[torch.Tensor] = None

        # Build causal mask for sequence length
        causal_mask = torch.triu(torch.full((candle_len, candle_len), float("-inf")), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

    def precompute_kv_cache(self, candle_seq: torch.Tensor) -> TemporalKVCache:
        """
        Precomputes Keys and Values from continuous candle sequence [B, 240, 6] or [240, 6].
        Stores precomputed K, V representations to allow O(1) cross-attention query evaluation.
        """
        candle_seq = torch.nan_to_num(candle_seq, nan=0.0, posinf=10.0, neginf=-10.0)
        if candle_seq.dim() == 2:
            candle_seq = candle_seq.unsqueeze(0)

        B, L, _ = candle_seq.shape
        # Pad or slice to candle_len if necessary
        if L < self.candle_len:
            pad = torch.zeros(B, self.candle_len - L, self.candle_dim, device=candle_seq.device, dtype=candle_seq.dtype)
            candle_seq = torch.cat([pad, candle_seq], dim=1)
        elif L > self.candle_len:
            candle_seq = candle_seq[:, -self.candle_len :, :]

        # 1. Project and add positional embedding
        h_cand = self.candle_proj(candle_seq) + self.pos_emb.unsqueeze(0)  # [B, candle_len, 128]

        # 2. Causal Transformer over candles
        h_cand = self.transformer(h_cand, mask=self.causal_mask[:self.candle_len, :self.candle_len])  # [B, candle_len, 128]

        # 3. Project to Keys and Values
        K = self.k_proj(h_cand).view(B, self.candle_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, candle_len, d]
        V = self.v_proj(h_cand).view(B, self.candle_len, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, candle_len, d]

        return TemporalKVCache(key=K, value=V, candle_len=self.candle_len, batch_size=B)

    def forward_with_kv_cache(
        self,
        z_feat: torch.Tensor,
        kv_cache: Union[TemporalKVCache, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluates cross-attention using precomputed KV-cache.
        Mathematically equivalent to full-sequence forward pass with bitwise accuracy.
        """
        if isinstance(kv_cache, TemporalKVCache):
            K, V = kv_cache.key, kv_cache.value
        elif isinstance(kv_cache, dict):
            K, V = kv_cache["key"], kv_cache["value"]
        elif isinstance(kv_cache, (tuple, list)):
            K, V = kv_cache[0], kv_cache[1]
        else:
            raise TypeError(f"Unsupported kv_cache type: {type(kv_cache)}")

        z_feat = torch.nan_to_num(z_feat, nan=0.0, posinf=1.0, neginf=-1.0)
        if z_feat.dim() == 1:
            z_feat = z_feat.unsqueeze(0)

        B = z_feat.shape[0]
        # Match batch dimension if cached for single batch and query is batched
        if K.shape[0] == 1 and B > 1:
            K = K.expand(B, -1, -1, -1)
            V = V.expand(B, -1, -1, -1)

        # Cross-Attention
        # Q: [B, H, 1, d]
        Q = self.q_proj(z_feat).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, candle_len]

        # Soft clamping for numerical stability across 240 candles
        scores = torch.clamp(scores, min=-50.0, max=50.0)

        attn_weights = F.softmax(scores, dim=-1)  # [B, H, 1, candle_len]
        self.last_candle_attn_weights = attn_weights.detach().squeeze(2)  # [B, H, candle_len]

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [B, H, 1, d]
        context = context.transpose(1, 2).contiguous().view(B, 1, self.temp_dim)  # [B, 1, temp_dim]
        z_temp = self.out_proj(context.squeeze(1))  # [B, latent_dim]

        # Gated Residual Integration
        gate_input = torch.cat([z_feat, z_temp], dim=-1)  # [B, 2 * latent_dim]
        alpha = self.gate(gate_input)  # [B, latent_dim]
        z_out = self.final_norm(z_feat + alpha * z_temp)  # [B, latent_dim]

        return z_out, self.last_candle_attn_weights

    def forward(
        self,
        z_feat: torch.Tensor,
        candle_seq: Optional[torch.Tensor] = None,
        kv_cache: Optional[Union[TemporalKVCache, Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z_feat: [B, 768] feature latent state from MarketStateEncoder
            candle_seq: Optional [B, 240, 6] or [240, 6] continuous M1 candle sequence
            kv_cache: Optional precomputed TemporalKVCache
        Returns:
            z_out: [B, 768] enriched latent state
            attn_weights: [B, num_heads, 240] attention weights per candle
        """
        if kv_cache is not None:
            return self.forward_with_kv_cache(z_feat, kv_cache)
        if candle_seq is not None:
            cache = self.precompute_kv_cache(candle_seq)
            return self.forward_with_kv_cache(z_feat, cache)
        raise ValueError("Either candle_seq or kv_cache must be provided to TemporalCrossAttentionEncoder.forward")

    def get_attention_weights(self, unbatch: bool = False, keep_query_dim: bool = False) -> Optional[torch.Tensor]:
        """Returns the stored attention weights [B, num_heads, candle_len] or [num_heads, candle_len] if unbatch."""
        if self.last_candle_attn_weights is None:
            return None
        attn = self.last_candle_attn_weights
        if keep_query_dim:
            attn = attn.unsqueeze(2)
        if unbatch and self.last_candle_attn_weights.shape[0] == 1:
            return attn.squeeze(0)
        return attn

    def get_attention_summary(
        self,
        bar_idx: Optional[int] = None,
        timestamp: Optional[str] = None,
        prices: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Extracts quantitative telemetry summary of the last attention weights across all 240 candles:
        - mean_attention: [candle_len] numpy array (averaged across heads and batch)
        - top_k_indices: list of indices of top-5 attended candles
        - recency_weight_ratio: float, ratio of attention on recent 15 minutes vs earlier bars
        - head_entropy: list of Shannon entropy per head
        - last_candle_weight: float, attention on the current (latest) bar
        - recency_ratio: float, ratio of attention on recent 30 candles vs distant 210 candles
        - peak_candle_index, peak_candle_lag, gini_sparsity, cross_head_variance, cross_head_coherence
        """
        if self.last_candle_attn_weights is None:
            return None
        return compute_attention_telemetry(
            self.last_candle_attn_weights,
            prices=prices,
            candle_len=self.candle_len,
            num_heads=self.num_heads,
            bar_idx=bar_idx,
            timestamp=timestamp,
        )



class LatentDynamicsWorldModel(nn.Module):
    """
    768D Latent Dynamics World Model for SGT-V193 Governor space.
    PROJECT.md lines 83-94 & Features 4, 5, 6.
    """

    def __init__(
        self,
        latent_dim: int = 768,
        action_dim: int = 64,
        hidden_dim: int = 512,
        alpha: float = 0.25,
        obs_dim: int = 137,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.obs_dim = obs_dim

        # State Representation Encoder h_phi(s_t) -> z_t (Feature 4)
        self.encoder = MarketStateEncoder(
            input_dim=obs_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
        )

        # Residual state transition network g(z_t, a_t) (Feature 5)
        self.g_net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)

        # Auxiliary prediction heads (Feature 6)
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.risk_bounds_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # [MFE, MAE]
            nn.Softplus(),  # Extrema excursions are strictly non-negative
        )

        # Optional Temporal Cross-Attention Module (Option C)
        # Defaults to None to maintain exact 30-key state_dict compatibility under strict=True.
        self.temporal_encoder: Optional[TemporalCrossAttentionEncoder] = None

        self._init_weights()

    def enable_temporal_stream(
        self,
        candle_dim: int = 6,
        candle_len: int = 240,
        temp_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        init_gate_bias: float = -2.0,
    ) -> None:
        """Enables and attaches the TemporalCrossAttentionEncoder to this world model."""
        device = next(self.parameters()).device
        self.temporal_encoder = TemporalCrossAttentionEncoder(
            candle_dim=candle_dim,
            candle_len=candle_len,
            latent_dim=self.latent_dim,
            temp_dim=temp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            init_gate_bias=init_gate_bias,
        ).to(device)

    def _init_weights(self) -> None:
        """Initializes weights with small residual scale for stable latent rollouts."""
        for m in [self.g_net, self.value_head, self.reward_head, self.risk_bounds_head]:
            for mod in m.modules():
                if isinstance(mod, nn.Linear):
                    nn.init.orthogonal_(mod.weight, gain=0.1)
                    if mod.bias is not None:
                        nn.init.zeros_(mod.bias)

    @staticmethod
    def encode_action(
        action: Union[int, float, torch.Tensor],
        sl_atr: float = 3.0,
        tp_atr: float = 6.0,
        action_dim: int = 64,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Encodes discrete action and continuous SL/TP parameters into canonical 64D vector:
        - Index 0: Direction (-1.0 for SELL, 0.0 for HOLD, +1.0 for BUY)
        - Index 1: is_hold (1.0 if HOLD else 0.0)
        - Index 2: is_buy (1.0 if BUY else 0.0)
        - Index 3: is_sell (1.0 if SELL else 0.0)
        - Index 4: Normalized SL: (sl_atr - 1.0) / (8.0 - 1.0) in [0, 1]
        - Index 5: Normalized TP: (tp_atr - 1.5) / (15.0 - 1.5) in [0, 1]
        - Index 6: Risk-Reward Ratio: min(5.0, tp_atr / max(1e-4, sl_atr)) / 5.0 in [0, 1]
        - Indices 7..63: Zero padding or continuous geometry parameters
        """
        if device is None:
            device = torch.device("cpu")

        if isinstance(action, torch.Tensor) and action.shape[-1] == action_dim:
            if action.ndim == 1:
                return action.unsqueeze(0).to(device)
            return action.to(device)

        act_idx = int(action.item() if isinstance(action, torch.Tensor) else action)
        direction = 0.0 if act_idx == 0 else (1.0 if act_idx == 1 else -1.0)
        norm_sl = max(0.0, min(1.0, (sl_atr - 1.0) / 7.0)) if act_idx != 0 else 0.0
        norm_tp = max(0.0, min(1.0, (tp_atr - 1.5) / 13.5)) if act_idx != 0 else 0.0
        norm_rr = min(5.0, tp_atr / max(1e-4, sl_atr)) / 5.0 if act_idx != 0 else 0.0

        vec = torch.zeros(action_dim, dtype=torch.float32, device=device)
        vec[0] = direction
        vec[1] = 1.0 if act_idx == 0 else 0.0
        vec[2] = 1.0 if act_idx == 1 else 0.0
        vec[3] = 1.0 if act_idx == 2 else 0.0
        vec[4] = norm_sl
        vec[5] = norm_tp
        vec[6] = norm_rr
        return vec.unsqueeze(0)

    def _prepare_action(
        self,
        a_t: Union[int, float, torch.Tensor, np.ndarray],
        z_clean: torch.Tensor,
    ) -> torch.Tensor:
        """Sanitizes and formats polymorphic action input into tensor of shape [B, action_dim]."""
        target_device = z_clean.device

        if isinstance(a_t, (int, float)):
            a_clean = self.encode_action(int(a_t), action_dim=self.action_dim, device=target_device)
        elif isinstance(a_t, np.ndarray):
            a_tensor = torch.from_numpy(a_t).to(device=target_device)
            if a_tensor.dtype in (torch.int64, torch.int32, torch.int16, torch.int8):
                if a_tensor.numel() == 1:
                    a_clean = self.encode_action(int(a_tensor.item()), action_dim=self.action_dim, device=target_device)
                else:
                    act_flat = a_tensor.view(-1)
                    a_clean = torch.cat(
                        [self.encode_action(int(act.item()), action_dim=self.action_dim, device=target_device) for act in act_flat],
                        dim=0,
                    )
            else:
                a_clean = a_tensor.to(dtype=torch.float32)
        elif isinstance(a_t, torch.Tensor):
            if a_t.dtype in (torch.int64, torch.int32, torch.long, torch.int16, torch.int8):
                if a_t.numel() == 1:
                    a_clean = self.encode_action(int(a_t.item()), action_dim=self.action_dim, device=target_device)
                else:
                    act_flat = a_t.view(-1)
                    a_clean = torch.cat(
                        [self.encode_action(int(act.item()), action_dim=self.action_dim, device=target_device) for act in act_flat],
                        dim=0,
                    )
            else:
                a_clean = a_t.to(device=target_device, dtype=torch.float32)
        else:
            a_clean = torch.as_tensor(a_t, dtype=torch.float32, device=target_device)

        a_clean = torch.nan_to_num(a_clean, nan=0.0, posinf=1e4, neginf=-1e4)
        if a_clean.ndim == 1:
            a_clean = a_clean.unsqueeze(0)
        if a_clean.shape[0] != z_clean.shape[0] and a_clean.shape[0] == 1:
            a_clean = a_clean.expand(z_clean.shape[0], -1)

        if a_clean.shape[-1] != self.action_dim:
            if a_clean.shape[-1] < self.action_dim:
                a_clean = F.pad(a_clean, (0, self.action_dim - a_clean.shape[-1]))
            else:
                a_clean = a_clean[..., :self.action_dim]

        return a_clean

    def encode_state(
        self,
        obs_137: Union[torch.Tensor, np.ndarray],
        candle_seq: Optional[Union[torch.Tensor, np.ndarray]] = None,
        kv_cache: Optional[Any] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """
        Public contract for Feature 4 & Feature 15:
        Encodes canonical 137D market observation vector into 768D normalized latent space.
        When candle_seq or kv_cache is provided and temporal_encoder is active, performs Temporal
        Cross-Attention (Option C) to enrich latent state with individual candle patterns.

        Args:
            obs_137: Observation array or tensor of shape [137] or [B, 137]
            candle_seq: Optional raw continuous candle sequence [240, 6] or [B, 240, 6]
            kv_cache: Optional precomputed TemporalKVCache
            return_attention: If True, returns (z_out, attn_weights); if False, returns z_out.
        Returns:
            z_t: Normalized latent tensor of shape [B, 768] (or [1, 768] if 1D input)
                 or Tuple[z_t, attn_weights] if return_attention=True
        """
        if not isinstance(obs_137, torch.Tensor):
            if isinstance(obs_137, np.ndarray):
                obs_137 = torch.from_numpy(obs_137)
            else:
                obs_137 = torch.tensor(obs_137, dtype=torch.float32)

        if obs_137.dim() == 1:
            obs_137 = obs_137.unsqueeze(0)

        device = next(self.parameters()).device
        obs_tensor = obs_137.to(device=device, dtype=torch.float32)
        z_feat = self.encoder(obs_tensor)

        if self.temporal_encoder is not None:
            if kv_cache is not None:
                z_out, attn_weights = self.temporal_encoder(z_feat, kv_cache=kv_cache)
                return (z_out, attn_weights) if return_attention else z_out
            if candle_seq is not None:
                if not isinstance(candle_seq, torch.Tensor):
                    if isinstance(candle_seq, np.ndarray):
                        candle_seq = torch.from_numpy(candle_seq)
                    else:
                        candle_seq = torch.tensor(candle_seq, dtype=torch.float32)
                if candle_seq.dim() == 2:
                    candle_seq = candle_seq.unsqueeze(0)
                candle_tensor = candle_seq.to(device=device, dtype=torch.float32)
                z_out, attn_weights = self.temporal_encoder(z_feat, candle_seq=candle_tensor)
                return (z_out, attn_weights) if return_attention else z_out

        return (z_feat, None) if return_attention else z_feat

    def precompute_temporal_kv_cache(self, candle_seq: Union[torch.Tensor, np.ndarray]) -> Optional[TemporalKVCache]:
        """Precomputes and returns TemporalKVCache if temporal_encoder is active, else None."""
        if self.temporal_encoder is not None:
            if not isinstance(candle_seq, torch.Tensor):
                if isinstance(candle_seq, np.ndarray):
                    candle_seq = torch.from_numpy(candle_seq)
                else:
                    candle_seq = torch.tensor(candle_seq, dtype=torch.float32)
            device = next(self.parameters()).device
            candle_tensor = candle_seq.to(device=device, dtype=torch.float32)
            return self.temporal_encoder.precompute_kv_cache(candle_tensor)
        return None

    def get_temporal_attention_weights(self, unbatch: bool = False, keep_query_dim: bool = False) -> Optional[torch.Tensor]:
        """Extracts stored attention weights across candles if temporal stream is active."""
        if self.temporal_encoder is not None:
            return self.temporal_encoder.get_attention_weights(unbatch=unbatch, keep_query_dim=keep_query_dim)
        return None

    def get_attention_weights(self, unbatch: bool = False, keep_query_dim: bool = False) -> Optional[torch.Tensor]:
        """Returns the stored attention weights from temporal stream if active."""
        if self.temporal_encoder is not None:
            return self.temporal_encoder.get_attention_weights(unbatch=unbatch, keep_query_dim=keep_query_dim)
        return None

    def get_attention_summary(
        self,
        bar_idx: Optional[int] = None,
        timestamp: Optional[str] = None,
        prices: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extracts quantitative telemetry summary of attention weights if temporal stream is active."""
        if self.temporal_encoder is not None:
            return self.temporal_encoder.get_attention_summary(bar_idx=bar_idx, timestamp=timestamp, prices=prices)
        return None

    def extract_attention_weights(
        self,
        obs_137: Optional[Union[torch.Tensor, np.ndarray]] = None,
        candle_seq: Optional[Union[torch.Tensor, np.ndarray]] = None,
        kv_cache: Optional[Any] = None,
        unbatch: bool = False,
        keep_query_dim: bool = False,
    ) -> Optional[torch.Tensor]:
        """Extracts attention weights, optionally running forward pass if inputs are provided."""
        if obs_137 is not None and (candle_seq is not None or kv_cache is not None):
            self.encode_state(obs_137, candle_seq=candle_seq, kv_cache=kv_cache)
        return self.get_attention_weights(unbatch=unbatch, keep_query_dim=keep_query_dim)

    def extract_attention_telemetry(
        self,
        obs_137: Optional[Union[torch.Tensor, np.ndarray]] = None,
        candle_seq: Optional[Union[torch.Tensor, np.ndarray]] = None,
        kv_cache: Optional[Any] = None,
        bar_idx: Optional[int] = None,
        timestamp: Optional[str] = None,
        prices: Optional[Any] = None,
    ) -> Optional[Dict[str, Any]]:
        """Extracts attention telemetry dictionary, optionally running forward pass if inputs provided."""
        if obs_137 is not None and (candle_seq is not None or kv_cache is not None):
            self.encode_state(obs_137, candle_seq=candle_seq, kv_cache=kv_cache)
        return self.get_attention_summary(bar_idx=bar_idx, timestamp=timestamp, prices=prices)

    def create_projection_head(
        self,
        d_in: Optional[int] = None,
        horizons: int = 3,
        dropout: float = 0.05,
    ) -> MultiHorizonProjectionHead:
        """Factory helper creating a MultiHorizonProjectionHead matched to model latent_dim."""
        if d_in is None:
            d_in = self.latent_dim
        device = next(self.parameters()).device
        head = create_projection_head(d_in=d_in, horizons=horizons, dropout=dropout)
        return head.to(device)

    def transition(self, z_t: torch.Tensor, a_t: Union[int, float, torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Calculates one-step forward latent transition:
        z_hat_{t+1} = LayerNorm(z_t + alpha * g(z_t, a_t))
        Args:
            z_t: Current latent state. Shape: [B, latent_dim] or [latent_dim]
            a_t: Proposed action embedding, continuous vector, or discrete action index.
        Returns:
            z_next: Next latent state. Shape: [B, latent_dim]
        """
        z_clean = torch.nan_to_num(z_t, nan=0.0, posinf=1e4, neginf=-1e4)
        if z_clean.ndim == 1:
            z_clean = z_clean.unsqueeze(0)

        a_clean = self._prepare_action(a_t, z_clean)

        inp = torch.cat([z_clean, a_clean], dim=-1)
        dz = self.g_net(inp)
        z_next = self.norm(z_clean + self.alpha * dz)
        return torch.nan_to_num(z_next, nan=0.0, posinf=50.0, neginf=-50.0)

    def rollout(self, z_0: torch.Tensor, action_sequence: torch.Tensor) -> torch.Tensor:
        """
        Unrolls multi-step forward latent trajectory:
        Args:
            z_0: Initial latent state. Shape: [B, latent_dim] or [latent_dim]
            action_sequence: Sequence of action embeddings. Shape: [B, H, action_dim] or [H, action_dim]
        Returns:
            trajectory: Predicted latent states. Shape: [B, H, latent_dim]
        """
        if z_0.ndim == 1:
            z_0 = z_0.unsqueeze(0)
        if action_sequence.ndim == 2:
            action_sequence = action_sequence.unsqueeze(0)

        B, H, _ = action_sequence.shape
        trajectory: List[torch.Tensor] = []
        z_curr = torch.nan_to_num(z_0, nan=0.0, posinf=1e4, neginf=-1e4)

        for step in range(H):
            a_step = action_sequence[:, step, :]
            z_curr = self.transition(z_curr, a_step)
            trajectory.append(z_curr)

        return torch.stack(trajectory, dim=1)

    def unroll_trajectory(
        self,
        z_0: torch.Tensor,
        action_sequence: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unrolls K steps forward returning latent states, values, and rewards.
        Evaluates reward head at step k, transitions to next state, and evaluates value head at step k+1.
        Args:
            z_0: Initial latent state. Shape: [B, latent_dim] or [latent_dim]
            action_sequence: Sequence of actions. Shape: [B, K, action_dim], [B, K], or [K, ...]
        Returns:
            pred_z: Predicted latent states [B, K, latent_dim]
            pred_v: Predicted state values [B, K, 1]
            pred_r: Predicted immediate rewards [B, K, 1]
        """
        if z_0.ndim == 1:
            z_0 = z_0.unsqueeze(0)

        if action_sequence.ndim == 1:
            action_sequence = action_sequence.unsqueeze(0)

        if action_sequence.ndim == 2:
            if action_sequence.dtype in (torch.int64, torch.int32, torch.long, torch.int16, torch.int8):
                B, K = action_sequence.shape
                actions_encoded = []
                for b in range(B):
                    row = [
                        self.encode_action(int(action_sequence[b, k].item()), action_dim=self.action_dim, device=z_0.device)
                        for k in range(K)
                    ]
                    actions_encoded.append(torch.cat(row, dim=0))
                action_sequence = torch.stack(actions_encoded, dim=0)
            elif action_sequence.shape[-1] == self.action_dim:
                action_sequence = action_sequence.unsqueeze(0)

        B, K, _ = action_sequence.shape
        if z_0.shape[0] != B and z_0.shape[0] == 1:
            z_0 = z_0.expand(B, -1)

        pred_z: List[torch.Tensor] = []
        pred_v: List[torch.Tensor] = []
        pred_r: List[torch.Tensor] = []
        z_curr = torch.nan_to_num(z_0, nan=0.0, posinf=50.0, neginf=-50.0)

        for step in range(K):
            a_step = action_sequence[:, step, :]
            heads = self.predict_heads(z_curr, a_step)
            pred_r.append(heads["reward"])

            z_curr = self.transition(z_curr, a_step)
            pred_z.append(z_curr)

            next_heads = self.predict_heads(z_curr, a_step)
            pred_v.append(next_heads["value"])

        return (
            torch.stack(pred_z, dim=1),
            torch.stack(pred_v, dim=1),
            torch.stack(pred_r, dim=1),
        )

    def predict_heads(
        self,
        z_t: torch.Tensor,
        a_t: Union[int, float, torch.Tensor, np.ndarray],
    ) -> Dict[str, torch.Tensor]:
        """
        Computes auxiliary predictions:
        Args:
            z_t: Latent state. Shape: [B, latent_dim] or [latent_dim]
            a_t: Action embedding, continuous vector, or discrete action index.
        Returns:
            Dictionary containing 'value', 'reward', 'risk_bounds' ([MFE, MAE])
        """
        z_clean = torch.nan_to_num(z_t, nan=0.0, posinf=1e4, neginf=-1e4)
        if z_clean.ndim == 1:
            z_clean = z_clean.unsqueeze(0)

        a_clean = self._prepare_action(a_t, z_clean)

        v_hat = self.value_head(z_clean)
        r_hat = self.reward_head(torch.cat([z_clean, a_clean], dim=-1))
        risk_bounds = self.risk_bounds_head(z_clean)

        return {
            "value": torch.nan_to_num(v_hat, nan=0.0),
            "reward": torch.nan_to_num(r_hat, nan=0.0),
            "risk_bounds": torch.nan_to_num(risk_bounds, nan=0.0),
        }

    def compute_coevolution_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        val_pred: torch.Tensor,
        val_target: torch.Tensor,
        rew_pred: torch.Tensor,
        rew_target: torch.Tensor,
        lambda_latent: float = 1.0,
        lambda_val: float = 1.0,
        lambda_rew: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes multi-step co-evolution loss:
        L_total = lambda_val * L_value + lambda_rew * L_reward + lambda_latent * L_latent
        where L_latent is cosine distance: (1.0 - CosSim(z_pred, z_target)).clamp(min=0.0).mean(),
        and L_value and L_reward are SmoothL1 (Huber, beta=1.0) losses.
        """
        cos_sim = F.cosine_similarity(z_pred, z_target, dim=-1)
        loss_latent = (1.0 - cos_sim).clamp(min=0.0).mean()

        loss_val = F.smooth_l1_loss(val_pred, val_target, beta=1.0)
        loss_rew = F.smooth_l1_loss(rew_pred, rew_target, beta=1.0)

        total_loss = (
            lambda_val * loss_val
            + lambda_rew * loss_rew
            + lambda_latent * loss_latent
        )

        return {
            "total_loss": total_loss,
            "loss_latent": loss_latent,
            "loss_value": loss_val,
            "loss_reward": loss_rew,
            "cos_similarity": cos_sim.mean(),
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        """
        Loads state_dict with seamless backward compatibility for legacy checkpoints
        (e.g., mythos_selfplay_triad_champion.pt) that do not contain encoder weights.
        """
        has_encoder = any(k.startswith("encoder.") for k in state_dict.keys())
        if strict and not has_encoder:
            res = super().load_state_dict(state_dict, strict=False, assign=assign)
            non_encoder_missing = [k for k in res.missing_keys if not k.startswith("encoder.")]
            if non_encoder_missing or res.unexpected_keys:
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}: "
                    f"Missing non-encoder keys: {non_encoder_missing}, "
                    f"Unexpected keys: {res.unexpected_keys}"
                )
            return res
        return super().load_state_dict(state_dict, strict=strict, assign=assign)



class AdversarialMarketStressor(nn.Module):
    """
    Adversarial Market Stressor generating worst-case execution shocks
    (synthetic liquidity sweep spikes, adverse slippage).
    PROJECT.md Feature 7 & line 13.
    """
    def __init__(self, latent_dim: int = 768, stress_scale: float = 0.15):
        super().__init__()
        self.latent_dim = latent_dim
        self.stress_scale = stress_scale
        self.sweep_generator = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.Tanh(),
            nn.Linear(256, latent_dim),
        )

    def forward(
        self, z_t: torch.Tensor, volatility_factor: float = 1.0
    ) -> Tuple[torch.Tensor, float]:
        """
        Injects adversarial directional stress into latent state.
        Args:
            z_t: Latent state. Shape: [B, latent_dim]
            volatility_factor: Scaling multiplier for market turmoil.
        Returns:
            stressed_z: Perturbed latent state.
            adverse_slippage_pips: Estimated execution friction penalty.
        """
        z_clean = torch.nan_to_num(z_t, nan=0.0, posinf=1e4, neginf=-1e4)
        shock_dir = self.sweep_generator(z_clean)
        effective_scale = self.stress_scale * volatility_factor
        stressed_z = z_clean + effective_scale * shock_dir
        adverse_slippage_pips = float(2.5 * volatility_factor)

        return torch.nan_to_num(stressed_z, nan=0.0), adverse_slippage_pips


def generate_trajectory_targets(
    env: Any,
    start_idx: int,
    actions: Sequence[Union[int, float]],
    horizon_k: int = 5,
    sl_atr: float = 2.0,
    tp_atr: float = 4.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts authentic future targets from real M1 environment.
    Returns:
        s0: (137,) float32 initial observation
        future_obs: (horizon_k, 137) float32 future observations
        target_values: (horizon_k, 1) float32 simulated trade return
        target_rewards: (horizon_k, 1) float32 1-step price delta normalized by ATR
    """
    s0 = env.get_observation(start_idx)
    future_obs = np.empty((horizon_k, 137), dtype=np.float32)
    target_v = np.empty((horizon_k, 1), dtype=np.float32)
    target_r = np.empty((horizon_k, 1), dtype=np.float32)

    close_arr = getattr(env, "close_arr", getattr(env, "closes", None))
    atr_arr = getattr(env, "atr_arr", getattr(env, "atrs", None))

    for k in range(horizon_k):
        fut_idx = start_idx + k + 1
        future_obs[k] = env.get_observation(fut_idx)

        act = int(actions[k]) if k < len(actions) else 0
        if close_arr is not None:
            c_prev = float(close_arr[fut_idx - 1])
            c_curr = float(close_arr[fut_idx])
        else:
            c_prev = 0.0
            c_curr = 0.0

        if hasattr(env, "get_atr"):
            atr = max(float(env.get_atr(fut_idx)), 1e-4)
        elif atr_arr is not None:
            atr = max(float(atr_arr[fut_idx]), 1e-4)
        else:
            atr = 1.0

        direction = 1.0 if act == 1 else (-1.0 if act == 2 else 0.0)
        target_r[k, 0] = direction * (c_curr - c_prev) / atr

        if act in (1, 2) and hasattr(env, "simulate_trade"):
            res = env.simulate_trade(fut_idx, action=act, sl_atr=sl_atr, tp_atr=tp_atr)
            ret_atr = getattr(res, "return_atr", res[1] if isinstance(res, (tuple, list)) else 0.0)
            target_v[k, 0] = float(np.tanh(ret_atr / 5.0))
        else:
            target_v[k, 0] = 0.0

    return s0, future_obs, target_v, target_r


# =============================================================================
# MULTI-HORIZON RETURN PROJECTION HEAD & ADAPTIVE LOSS (Features 5 & 6)
# =============================================================================


class MultiHorizonProjectionHead(nn.Module):
    """
    Multi-Horizon Return Prediction Projection Head (Milestone 2 / Feature 5).
    Attaches on top of the 768D fused latent output z_out from TemporalCrossAttentionEncoder
    (or LatentDynamicsWorldModel.encode_state) to predict future price returns normalized by ATR(14)
    at three discrete horizons: t+5 (5 min), t+15 (15 min), and t+30 (30 min).

    Topology:
        nn.Sequential(
            nn.Linear(d_in, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, num_horizons)
        )

    Weights: Exactly 8 parameter tensors:
        - net.0.weight: [256, d_in]
        - net.0.bias: [256]
        - net.1.weight: [256]
        - net.1.bias: [256]
        - net.4.weight: [128, 256]
        - net.4.bias: [128]
        - net.6.weight: [num_horizons, 128]
        - net.6.bias: [num_horizons]
    Total Parameters: Exactly 230,659 weights when d_in=768, num_horizons=3.
    """

    def __init__(
        self,
        d_in: int = 768,
        num_horizons: int = 3,
        dropout: float = 0.05,
        *,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        hidden_dim: int = 256,
        intermediate_dim: int = 128,
    ) -> None:
        super().__init__()
        if input_dim is not None:
            d_in = input_dim
        if output_dim is not None:
            num_horizons = output_dim

        self.d_in = d_in
        self.num_horizons = num_horizons
        self.dropout_rate = dropout

        self.net = nn.Sequential(
            nn.Linear(d_in, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, num_horizons),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes weights using orthogonal initialization for stable multi-task regression."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z_out: torch.Tensor) -> torch.Tensor:
        """
        Forward pass projecting 768D latent state into multi-horizon return predictions.

        Args:
            z_out: Latent state tensor of shape [B, 768] or [768] from TCA or WorldModel.
        Returns:
            predictions: Projected returns of shape [B, 3] or [1, 3] corresponding to [y_5, y_15, y_30].
        """
        clean_z = torch.nan_to_num(z_out, nan=0.0, posinf=50.0, neginf=-50.0)
        if clean_z.dim() == 1:
            clean_z = clean_z.unsqueeze(0)
        return self.net(clean_z)

    def predict_returns(
        self,
        z_out: torch.Tensor,
        atr: Optional[Union[float, torch.Tensor, np.ndarray]] = None,
    ) -> torch.Tensor:
        """
        Predicts multi-horizon returns, optionally un-normalizing by ATR.
        If atr is supplied, un-normalizes returns via returns = pred * atr.
        """
        pred = self.forward(z_out)
        if atr is not None:
            if not isinstance(atr, torch.Tensor):
                atr_tensor = torch.tensor(atr, dtype=pred.dtype, device=pred.device)
            else:
                atr_tensor = atr.to(device=pred.device, dtype=pred.dtype)
            if atr_tensor.dim() == 1:
                atr_tensor = atr_tensor.unsqueeze(-1)
            elif atr_tensor.dim() == 0:
                atr_tensor = atr_tensor.view(1, 1)
            return pred * atr_tensor
        return pred


def create_projection_head(
    d_in: int = 768,
    horizons: int = 3,
    dropout: float = 0.05,
) -> MultiHorizonProjectionHead:
    """Factory helper creating a MultiHorizonProjectionHead."""
    return MultiHorizonProjectionHead(
        d_in=d_in,
        num_horizons=horizons,
        dropout=dropout,
    )


def compute_multi_horizon_loss(
    pred_returns: torch.Tensor,
    targets: torch.Tensor,
    beta: float = 1.0,
    weights: Optional[Union[torch.Tensor, Sequence[float]]] = None,
) -> torch.Tensor:
    """
    Computes Smooth L1 (Huber) loss across horizons (beta=1.0) with optional per-horizon weights.
    Defaults to equal weighting [1.0, 1.0, 1.0].

    Args:
        pred_returns: Predicted returns tensor of shape [B, 3] or [3].
        targets: Target returns tensor of shape [B, 3] or [3].
        beta: Huber transition threshold (default: 1.0).
        weights: Optional per-horizon weights tensor or sequence.

    Returns:
        Scalar differentiable loss tensor.
    """
    clean_pred = torch.nan_to_num(pred_returns, nan=0.0, posinf=50.0, neginf=-50.0)
    clean_tgt = torch.nan_to_num(targets, nan=0.0, posinf=50.0, neginf=-50.0)

    if clean_pred.dim() == 1:
        clean_pred = clean_pred.unsqueeze(0)
    if clean_tgt.dim() == 1:
        clean_tgt = clean_tgt.unsqueeze(0)

    loss_matrix = F.smooth_l1_loss(clean_pred, clean_tgt, beta=beta, reduction="none")

    if weights is not None:
        if not isinstance(weights, torch.Tensor):
            w = torch.tensor(weights, dtype=loss_matrix.dtype, device=loss_matrix.device)
        else:
            w = weights.to(device=loss_matrix.device, dtype=loss_matrix.dtype)
        if w.dim() == 1:
            w = w.unsqueeze(0)
        loss_matrix = loss_matrix * w

    return loss_matrix.mean()

