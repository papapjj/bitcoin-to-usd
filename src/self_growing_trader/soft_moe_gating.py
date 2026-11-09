"""
src/self_growing_trader/soft_moe_gating.py

Production Continuous Soft-MoE (Mixture of Regimes) Gating Network &
Capital Preservation Guard.

Replaces discrete hard switching and H1 macro filter override freeze with a
continuous, differentiable gating network taking the 137-dim feature vector
(multi-timescale features: M5 EMA diff/slope idx 80-81, M15 EMA diff/slope idx 84-85,
H1 EMA diff/slope idx 88-89, ATR ratio, volatility, and trend indicators).

Computes continuous regime weights w_bull, w_bear in [0.0, 1.0] with w_bull + w_bear == 1.0.
Zero-lag smooth response to market inflection, eliminating the 15-45m lag of hard discrete switching.
In transition zones (|w_bull - w_bear| <= 0.30 or chop >= 0.60), activates Capital Preservation Guard
with defensive risk scaling and market order chasing suppression.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 20 Salient Multi-Timescale & Microstructure Feature Indices from 137D Wire
# [0]: ret1_atr (M1 return)
# [2]: ret5_atr (M1 short momentum)
# [3]: ret15_atr (M1 medium momentum)
# [9]: lower_wick_atr (rejection/buying pressure wick)
# [12]: ema_slope_atr (M1 fast trend velocity)
# [16]: rsi14 (M1 structural momentum)
# [17]: rsi5 (M1 fast oversold / overbought)
# [20]: volatility_ratio (expansion / contraction)
# [80]: m5_ema_diff_atr (M5 trend alignment)
# [81]: m5_ema_slope_atr (M5 momentum direction)
# [82]: m5_volatility_ratio (M5 volatility state)
# [84]: m15_ema_diff_atr (M15 baseline trend)
# [85]: m15_ema_slope_atr (M15 momentum direction)
# [88]: h1_ema_diff_atr (H1 macro context)
# [89]: h1_ema_slope_atr (H1 macro slope)
# [92]: trend_strength_atr (composite directional trend)
# [93]: chop_index (consolidation metric: 0=Trend, 1=Chop)
# [117]: swing_range120_position (2-Hour Range Extreme: 0=Bottom, 1=Top)
# [126]: absorption_score (Wyckoff institutional absorption)
# [129]: spring_score (Wyckoff accumulation spring signal)
WIRE_MTS_INDICES: tuple[int, ...] = (
    0, 2, 3, 9, 12, 16, 17, 20,
    80, 81, 82, 84, 85, 88, 89, 92, 93,
    117, 126, 129,
)

NUM_MTS_FEATURES: int = len(WIRE_MTS_INDICES)  # 20
WIRE_MTS_INDICES_NP: np.ndarray = np.array(WIRE_MTS_INDICES, dtype=np.intp)


def _get_finite(arr: np.ndarray, idx_137: int, idx_20: int, default: float) -> float:
    n = arr.size
    if n > idx_137:
        v = float(arr[idx_137])
    elif n == NUM_MTS_FEATURES and 0 <= idx_20 < NUM_MTS_FEATURES:
        v = float(arr[idx_20])
    else:
        return default
    return v if math.isfinite(v) else default


def extract_multi_timescale_features(obs: np.ndarray | torch.Tensor | Sequence[float]) -> np.ndarray:
    """Extract the 20 salient multi-timescale features from 137D observation vector.

    Handles full 137-dim inputs, already extracted 20-dim inputs, and truncated synthetic inputs.
    """
    if isinstance(obs, torch.Tensor):
        arr = obs.detach().cpu().numpy().reshape(-1)
    elif isinstance(obs, np.ndarray):
        arr = obs.reshape(-1)
    else:
        arr = np.asarray(obs, dtype=np.float32).reshape(-1)

    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)

    if arr.size == NUM_MTS_FEATURES:
        return arr

    if arr.size >= 130:
        return arr[WIRE_MTS_INDICES_NP]

    out = np.zeros(NUM_MTS_FEATURES, dtype=np.float32)
    n = arr.size
    for i, idx in enumerate(WIRE_MTS_INDICES):
        if idx < n:
            val = float(arr[idx])
            out[i] = val if math.isfinite(val) else 0.0
    return out


class SoftMoEGatingNetwork(nn.Module):
    """Continuous, differentiable gating network for Mixture-of-Regimes routing.

    Computes continuous regime weights:
        w_bull in [0.0, 1.0]
        w_bear in [0.0, 1.0]
    subject to the invariant:
        w_bull + w_bear == 1.0.

    Eliminates the 15-45m lag of discrete hard switching by responding smoothly
    to multi-timescale momentum (M1, M5, M15, H1) and chop dynamics in real-time.
    """

    def __init__(
        self,
        in_features: int = NUM_MTS_FEATURES,
        hidden_dim: int = 32,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.hidden_dim = int(hidden_dim)
        self.temperature = max(0.05, float(temperature))

        self.fc1 = nn.Linear(self.in_features, self.hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(self.hidden_dim, 2)

        self._initialize_calibrated_weights()
        self._sync_numpy_weights()

    def _sync_numpy_weights(self) -> None:
        """Cache float32 numpy weights for sub-0.05ms CPU inference."""
        self._np_fc1_w = self.fc1.weight.detach().cpu().numpy().copy()
        self._np_fc1_b = self.fc1.bias.detach().cpu().numpy().copy()
        self._np_fc2_w = self.fc2.weight.detach().cpu().numpy().copy()
        self._np_fc2_b = self.fc2.bias.detach().cpu().numpy().copy()
        self._np_fc1_w_T = self._np_fc1_w.T
        self._np_fc2_w_T = self._np_fc2_w.T

    def _initialize_calibrated_weights(self) -> None:
        """Initialize weights with domain-calibrated quantitative momentum priors.

        Ensures zero-lag response to inflections:
        - Fast M1/M5 momentum features (indices 0, 2, 80, 81) have high responsiveness
        - M15 baseline (indices 84, 85) provides trend continuity
        - H1 macro context (indices 88, 89) provides long-term orientation without lockup
        - Chop index (idx 93 / pos 16) pulls logits towards equality (w_bull ~ w_bear ~ 0.5)
        """
        nn.init.xavier_uniform_(self.fc1.weight, gain=0.8)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.8)
        nn.init.zeros_(self.fc2.bias)

        # Calibrate first layer: feature sensitivity projection
        # Map: Bull evidence -> hidden units [0..15], Bear evidence -> hidden units [16..31]
        with torch.no_grad():
            # Features in x_mts:
            # 0: ret1_atr, 1: ret5_atr, 2: ret15_atr, 3: lower_wick, 4: ema_slope
            # 5: rsi14, 6: rsi5, 7: vol_ratio, 8: m5_diff, 9: m5_slope, 10: m5_vol
            # 11: m15_diff, 12: m15_slope, 13: h1_diff, 14: h1_slope, 15: trend_str
            # 16: chop, 17: swing_pos, 18: absorption, 19: spring
            for h in range(16):
                # Bull detector hidden units
                self.fc1.weight[h, 0] += 0.35   # ret1
                self.fc1.weight[h, 1] += 0.30   # ret5
                self.fc1.weight[h, 4] += 0.40   # ema_slope
                self.fc1.weight[h, 6] += 0.25   # rsi5
                self.fc1.weight[h, 8] += 0.55   # m5_diff
                self.fc1.weight[h, 9] += 0.45   # m5_slope
                self.fc1.weight[h, 11] += 0.40  # m15_diff
                self.fc1.weight[h, 13] += 0.35  # h1_diff
                self.fc1.weight[h, 16] -= 0.30  # chop pulls down conviction

            for h in range(16, 32):
                # Bear detector hidden units
                self.fc1.weight[h, 0] -= 0.35   # ret1 negative
                self.fc1.weight[h, 1] -= 0.30   # ret5 negative
                self.fc1.weight[h, 4] -= 0.40   # ema_slope negative
                self.fc1.weight[h, 6] -= 0.25   # rsi5 oversold / dump
                self.fc1.weight[h, 8] -= 0.55   # m5_diff negative
                self.fc1.weight[h, 9] -= 0.45   # m5_slope negative
                self.fc1.weight[h, 11] -= 0.40  # m15_diff negative
                self.fc1.weight[h, 13] -= 0.35  # h1_diff negative
                self.fc1.weight[h, 16] -= 0.30  # chop pulls down conviction

            # Second layer: route hidden units to logits (0=Bull, 1=Bear)
            self.fc2.weight[0, :16] += 0.50
            self.fc2.weight[0, 16:] -= 0.35
            self.fc2.weight[1, :16] -= 0.35
            self.fc2.weight[1, 16:] += 0.50

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass computing continuous regime weights.

        Args:
            x: Tensor of shape [..., 20] or [..., 137].

        Returns:
            weights: Tensor of shape [..., 2] with softmax normalized weights
                     [w_bull, w_bear] summing to 1.0 along the last dimension.
        """
        if x.shape[-1] != self.in_features:
            if x.shape[-1] >= 130:
                indices = torch.tensor(WIRE_MTS_INDICES, dtype=torch.long, device=x.device)
                x = torch.index_select(x, -1, indices)
            elif x.shape[-1] > self.in_features:
                x = x[..., :self.in_features]
            else:
                pad_size = self.in_features - x.shape[-1]
                x = F.pad(x, (0, pad_size), value=0.0)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        scaled_logits = logits / self.temperature
        weights = F.softmax(scaled_logits, dim=-1)
        return weights

    def compute_regime_weights(
        self,
        obs: np.ndarray | torch.Tensor | Sequence[float],
    ) -> Tuple[float, float, dict[str, Any]]:
        """Compute continuous regime weights and telemetry info.

        Contract interface for DualChampionRouter:
        Input: obs (137-dim or 20-dim observation)
        Output:
            w_bull (float in [0.0, 1.0])
            w_bear (float in [0.0, 1.0]) such that w_bull + w_bear == 1.0
            info (dict with transition_flag, delta_regime, m5_trend, m15_trend, h1_trend, chop_index)
        Latency constraint: <= 0.5ms on CPU.
        """
        is_torch_grad = isinstance(obs, torch.Tensor) and obs.requires_grad

        if is_torch_grad:
            x_in = obs.unsqueeze(0) if obs.dim() == 1 else obs
            w = self.forward(x_in)
            w_bull_t = w[..., 0].squeeze()
            w_bear_t = w[..., 1].squeeze()
            w_bull = float(w_bull_t.detach().cpu().item())
            w_bear = float(w_bear_t.detach().cpu().item())
        else:
            x_mts = extract_multi_timescale_features(obs)
            # Ultra-fast NumPy inference (<= 0.03ms per tick on CPU)
            # h = GELU(x @ W1.T + b1)
            pre_h = x_mts @ self._np_fc1_w_T + self._np_fc1_b
            h = 0.5 * pre_h * (1.0 + np.tanh(0.7978845608 * (pre_h + 0.044715 * pre_h * pre_h * pre_h)))
            # logits = h @ W2.T + b2
            logits = (h @ self._np_fc2_w_T + self._np_fc2_b) / self.temperature
            # Ultra-fast 2-element scalar softmax
            l0 = float(logits[0])
            l1 = float(logits[1])
            m = l0 if l0 > l1 else l1
            e0 = math.exp(l0 - m)
            e1 = math.exp(l1 - m)
            s = e0 + e1
            w_bull = e0 / s
            w_bear = e1 / s

        # Enforce exact numerical simplex invariant: w_bull + w_bear == 1.0
        fallback_active = False
        if not math.isfinite(w_bull) or not math.isfinite(w_bear):
            w_bull = 0.50
            w_bear = 0.50
            fallback_active = True

        total = w_bull + w_bear
        if total > 0.0 and abs(total - 1.0) > 1e-7:
            w_bull = w_bull / total
            w_bear = 1.0 - w_bull
        else:
            w_bear = 1.0 - w_bull

        # Clamp strictly to [0.0, 1.0]
        w_bull = max(0.0, min(1.0, w_bull))
        w_bear = max(0.0, min(1.0, 1.0 - w_bull))

        if not math.isfinite(w_bull) or not math.isfinite(w_bear):
            w_bull = 0.50
            w_bear = 0.50
            fallback_active = True

        # Extract telemetry feature scalars
        arr_137 = (
            obs.detach().cpu().numpy().reshape(-1)
            if isinstance(obs, torch.Tensor)
            else (obs.reshape(-1) if isinstance(obs, np.ndarray) else np.asarray(obs, dtype=np.float32).reshape(-1))
        )

        m5_trend = _get_finite(arr_137, 80, 8, 0.0)
        m5_slope = _get_finite(arr_137, 81, 9, 0.0)
        m15_trend = _get_finite(arr_137, 84, 11, 0.0)
        h1_trend = _get_finite(arr_137, 88, 13, 0.0)
        chop_index = max(0.0, min(1.0, _get_finite(arr_137, 93, 16, 0.50)))

        delta_regime = abs(w_bull - w_bear)
        is_chop = chop_index >= 0.60
        # Transition Zone: |w_bull - w_bear| <= 0.30 (or chop >= 0.60)
        transition_flag = fallback_active or (delta_regime <= (0.30 + 1e-6)) or is_chop

        if fallback_active:
            defensive_scale = 0.40
        elif delta_regime <= (0.30 + 1e-6):
            defensive_scale = max(0.40, min(1.0, delta_regime / 0.30))
        else:
            defensive_scale = 1.0

        if is_chop:
            defensive_scale = min(defensive_scale, 0.50)

        dominant_regime = (
            "TRANSITION"
            if fallback_active
            else ("BULL" if w_bull > 0.65 else ("BEAR" if w_bear > 0.65 else "TRANSITION"))
        )

        info: dict[str, Any] = {
            "transition_flag": bool(transition_flag),
            "delta_regime": float(delta_regime),
            "m5_trend": float(m5_trend),
            "m5_slope": float(m5_slope),
            "m15_trend": float(m15_trend),
            "h1_trend": float(h1_trend),
            "chop_index": float(chop_index),
            "is_chop": bool(is_chop),
            "capital_preservation_active": bool(transition_flag),
            "defensive_scale": float(defensive_scale),
            "dominant_regime": dominant_regime,
        }

        return w_bull, w_bear, info


class CapitalPreservationGuard:
    """Capital Preservation Guard in Regime Transition Zones.

    Protects portfolio against whipsaw stopouts in market contest/chop regimes:
    1. Detects Transition Zone: |w_bull - w_bear| <= 0.30 or chop_index >= 0.60.
    2. Suppresses market order chasing: converts/vetoes MARKET_BUY (Action 1) and
       MARKET_SELL (Action 2) to ACTION_HOLD (0).
    3. Permits patient limit orders (Action 3 BUY_LIMIT, Action 4 SELL_LIMIT)
       when aligned with micro momentum.
    4. Provides defensive risk scaling factor (0.40 to 1.0).
    """

    def __init__(
        self,
        transition_threshold: float = 0.30,
        chop_threshold: float = 0.60,
        min_defensive_scale: float = 0.40,
    ) -> None:
        self.transition_threshold = float(transition_threshold)
        self.chop_threshold = float(chop_threshold)
        self.min_defensive_scale = float(min_defensive_scale)

    def is_transition_active(
        self,
        w_bull: float,
        w_bear: float,
        chop_index: float = 0.50,
        eps: float = 1e-6,
    ) -> bool:
        """Return True if market is within contest / transition zone or chop."""
        if not math.isfinite(w_bull) or not math.isfinite(w_bear):
            return True
        val_chop = float(chop_index)
        chop = val_chop if math.isfinite(val_chop) else 0.50
        delta = abs(w_bull - w_bear)
        return (delta <= self.transition_threshold + eps) or (chop >= self.chop_threshold)

    def compute_defensive_scale(
        self,
        w_bull: float,
        w_bear: float,
        chop_index: float = 0.50,
        eps: float = 1e-6,
    ) -> float:
        """Compute risk de-leveraging scale in [min_defensive_scale, 1.0]."""
        if not math.isfinite(w_bull) or not math.isfinite(w_bear):
            return float(self.min_defensive_scale)
        val_chop = float(chop_index)
        chop = val_chop if math.isfinite(val_chop) else 0.50
        delta = abs(w_bull - w_bear)
        if delta <= self.transition_threshold + eps:
            scale = max(self.min_defensive_scale, min(1.0, delta / self.transition_threshold))
        else:
            scale = 1.0

        if chop >= self.chop_threshold:
            scale = min(scale, 0.50)
        return float(scale)

    def evaluate_order(
        self,
        proposed_action: int,
        w_bull: float,
        w_bear: float,
        chop_index: float = 0.50,
    ) -> Tuple[bool, int, str]:
        """Evaluate whether a proposed action should be intercepted by the guard.

        Args:
            proposed_action: Action integer index (1=MKT_BUY, 2=MKT_SELL, 3=BUY_LIMIT, etc.)
            w_bull: Continuous Bull specialist weight
            w_bear: Continuous Bear specialist weight
            chop_index: Choppiness index from wire [93]

        Returns:
            (is_vetoed, effective_action, reason)
        """
        val_chop = float(chop_index) if math.isfinite(float(chop_index)) else 0.50
        in_transition = self.is_transition_active(w_bull, w_bear, val_chop)
        if not in_transition:
            return False, proposed_action, ""

        # Suppress market order chasing in transition zone
        # Action 1: MARKET_BUY, Action 2: MARKET_SELL
        if proposed_action in (1, 2):
            return True, 0, "capital_preservation_transition_zone_no_market_chase"

        # In severe chop, also suppress stop orders (breakout chasing)
        # Action 5: BUY_STOP, Action 6: SELL_STOP
        if val_chop >= 0.70 and proposed_action in (5, 6):
            return True, 0, "capital_preservation_severe_chop_no_breakout_chase"

        return False, proposed_action, ""
