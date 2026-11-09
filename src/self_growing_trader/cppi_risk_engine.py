"""
src/self_growing_trader/cppi_risk_engine.py

Constant Proportion Portfolio Insurance (CPPI) & Asymmetric Volatility Pumping Risk Engine.
Provides real-time PeakEquity ratchet, Floor protection, Cushion calculation,
Dynamic Leverage Pumping on accumulated house money, and 3-Timeframe Confluence Multiplier.
Strictly guarantees the mathematical invariant:
    risk_pct <= (0.50 * cushion) / current_equity
making it mathematically impossible for trade Stop Losses to penetrate the capital floor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional, Sequence, Tuple
import numpy as np


@dataclass(frozen=True)
class CPPIConfig:
    """Configuration parameters for CPPIFloorEngine."""

    # Floor & Drawdown Parameters (Requirement R1, Feature 1)
    max_allowed_dd: float = 0.08          # Maximum permitted drawdown from peak (strictly capped at 8.0%)
    cushion_loss_fraction: float = 0.50   # Hard barrier: single trade loss capped at 50% of remaining cushion

    # Asymmetric Leverage Pumping Parameters (Requirement R2, Feature 4)
    base_alpha: float = 0.3               # Baseline leverage contraction factor at floor (0.3x)
    pumping_slope: float = 4.7            # Linear leverage expansion slope
    max_cushion_ratio: float = 2.0        # Maximum cushion ratio ceiling before saturation

    # 3-Timeframe Confluence Parameters (Requirement R2, Feature 5)
    confluence_boost: float = 1.50        # Multiplier applied when H1, M15, M5 align with trade direction (1.5x)
    confluence_threshold: float = 0.0     # Minimum signal magnitude to establish non-neutral directional bias
    max_alpha: Optional[float] = None     # Optional hard upper bound on leverage multiplier

    # 137-dim Input Feature Vector Indices (Wire Parity)
    h1_trend_idx: int = 88                # Index in 137-dim vector: h1_ema_diff_atr (H1 Trend)
    m15_trend_idx: int = 84               # Index in 137-dim vector: m15_ema_diff_atr (M15 Trend)
    m5_velocity_idx: int = 81             # Index in 137-dim vector: m5_ema_slope_atr (M5 Velocity)

    # Safety Bounds
    max_risk_cap: float = 0.25            # Absolute ceiling on risk_pct per single trade (25% safety cap)
    min_risk_pct: float = 0.0             # Minimum non-zero risk fraction (default 0.0)
    initial_equity: float = 0.0           # Initial baseline equity (if known upfront)

    def __post_init__(self) -> None:
        """Validate parameter invariants strictly upon instantiation."""
        if not (0.0 < float(self.max_allowed_dd) < 1.0):
            raise ValueError(f"max_allowed_dd must be in (0, 1), got {self.max_allowed_dd}")
        if not (0.0 < float(self.cushion_loss_fraction) <= 1.0):
            raise ValueError(f"cushion_loss_fraction must be in (0, 1], got {self.cushion_loss_fraction}")
        if float(self.base_alpha) < 0.0:
            raise ValueError(f"base_alpha must be non-negative, got {self.base_alpha}")
        if float(self.pumping_slope) < 0.0:
            raise ValueError(f"pumping_slope must be non-negative, got {self.pumping_slope}")
        if float(self.max_cushion_ratio) <= 0.0:
            raise ValueError(f"max_cushion_ratio must be positive, got {self.max_cushion_ratio}")
        if float(self.confluence_boost) < 1.0:
            raise ValueError(f"confluence_boost must be >= 1.0, got {self.confluence_boost}")
        if self.max_alpha is not None and float(self.max_alpha) < float(self.base_alpha):
            raise ValueError(f"max_alpha ({self.max_alpha}) cannot be less than base_alpha ({self.base_alpha})")
        if not (0.0 < float(self.max_risk_cap) <= 1.0):
            raise ValueError(f"max_risk_cap must be in (0, 1], got {self.max_risk_cap}")
        for name, idx in (
            ("h1_trend_idx", self.h1_trend_idx),
            ("m15_trend_idx", self.m15_trend_idx),
            ("m5_velocity_idx", self.m5_velocity_idx),
        ):
            if not (0 <= int(idx) < 137):
                raise ValueError(f"{name} must be within [0, 136], got {idx}")


class CPPIFloorEngine:
    """Continuous Constant Proportion Portfolio Insurance (CPPI) & Dynamic Risk Engine.

    Tracks peak equity and dynamic floor in real-time, calculates risk cushions,
    pumps leverage on house money, and enforces the hard 50% cushion loss limit.
    """

    def __init__(
        self,
        config: Optional[CPPIConfig] = None,
        initial_equity: float = 0.0,
    ) -> None:
        self.config: CPPIConfig = config if config is not None else CPPIConfig()
        init_eq = float(initial_equity) if initial_equity > 0.0 else float(self.config.initial_equity)
        self._peak_equity: float = float(max(0.0, init_eq))
        self._floor: float = (1.0 - self.config.max_allowed_dd) * self._peak_equity if self._peak_equity > 0.0 else 0.0
        self._cushion: float = 0.0
        self._cushion_ratio: float = 0.0
        self._last_equity: float = self._peak_equity
        self._trade_eval_count: int = 0

        if self._peak_equity > 0.0:
            self.update_equity(self._peak_equity)

    # --- Public Observable Properties ---
    @property
    def peak_equity(self) -> float:
        """Historical highest equity mark observed."""
        return self._peak_equity

    @property
    def floor(self) -> float:
        """Immutable capital floor: (1 - max_allowed_dd) * peak_equity."""
        return self._floor

    @property
    def cushion(self) -> float:
        """Current dollar risk buffer: max(0.0, current_equity - floor)."""
        return self._cushion

    @property
    def cushion_ratio(self) -> float:
        """Normalized cushion ratio: cushion / (max_allowed_dd * peak_equity)."""
        return self._cushion_ratio

    @property
    def current_equity(self) -> float:
        """Latest reported equity value."""
        return self._last_equity

    # --- State Management ---
    def reset(self, initial_equity: float = 0.0) -> None:
        """Reset internal high-water mark and state (e.g., at backtest start)."""
        init_eq = float(max(0.0, initial_equity))
        self._peak_equity = init_eq
        self._floor = (1.0 - self.config.max_allowed_dd) * self._peak_equity if self._peak_equity > 0.0 else 0.0
        self._cushion = 0.0
        self._cushion_ratio = 0.0
        self._last_equity = init_eq
        self._trade_eval_count = 0
        if self._peak_equity > 0.0:
            self.update_equity(self._peak_equity)

    def update_equity(self, current_equity: float) -> Tuple[float, float, float]:
        """Update continuous equity state.

        Ratchets peak equity monotonically when new high is reached,
        and computes current floor and dollar cushion.

        Returns:
            Tuple of (peak_equity, floor, cushion), all guaranteed finite and >= 0.0.
        """
        # Guard 1: Non-finite or non-positive equity
        if not math.isfinite(current_equity) or current_equity <= 0.0:
            floor = (1.0 - self.config.max_allowed_dd) * self._peak_equity if self._peak_equity > 0.0 else 0.0
            cushion = max(0.0, float(current_equity) - floor) if (math.isfinite(current_equity) and current_equity > 0.0) else 0.0
            return (float(self._peak_equity), float(floor), float(cushion))

        eq = float(current_equity)
        # Ratchet peak equity monotonically
        if eq > self._peak_equity:
            self._peak_equity = eq

        self._floor = (1.0 - self.config.max_allowed_dd) * self._peak_equity
        self._cushion = max(0.0, eq - self._floor)
        self._last_equity = eq

        denom = self.config.max_allowed_dd * self._peak_equity
        self._cushion_ratio = (self._cushion / denom) if denom > 1e-12 else 0.0

        return (float(self._peak_equity), float(self._floor), float(self._cushion))

    def compute_cushion_ratio(self, current_equity: Optional[float] = None) -> float:
        """Compute the normalized cushion ratio: Cushion / (MaxDD * PeakEquity).

        If current_equity is provided, computes for that equity value.
        Otherwise returns the current cached cushion ratio.
        """
        if current_equity is not None:
            if not math.isfinite(current_equity) or current_equity <= 0.0:
                return 0.0
            eq = float(current_equity)
            peak = max(self._peak_equity, eq)
            floor = (1.0 - self.config.max_allowed_dd) * peak
            cushion = max(0.0, eq - floor)
            denom = self.config.max_allowed_dd * peak
            return float(cushion / denom) if denom > 1e-12 else 0.0
        return float(self._cushion_ratio)

    # --- 3-Timeframe Confluence Extraction & Evaluation ---
    def extract_timeframe_signals(
        self,
        inputs: Any,
    ) -> Tuple[float, float, float]:
        """Safely extract (h1_trend, m15_trend, m5_velocity) from 137-dim observation.

        Handles 1D arrays, 2D batch tensors [batch_size, 137], lists, and PyTorch tensors.
        Returns (0.0, 0.0, 0.0) fail-closed on malformed, truncated, or non-finite inputs.
        """
        if inputs is None:
            return (0.0, 0.0, 0.0)
        try:
            if hasattr(inputs, "detach") and hasattr(inputs, "cpu"):
                arr = inputs.detach().cpu().numpy()
            else:
                arr = np.asarray(inputs)

            if arr.ndim > 1:
                arr = arr[0]
            flat = arr.reshape(-1)

            max_idx = max(self.config.h1_trend_idx, self.config.m15_trend_idx, self.config.m5_velocity_idx)
            if flat.size <= max_idx:
                return (0.0, 0.0, 0.0)

            h1 = float(flat[self.config.h1_trend_idx])
            m15 = float(flat[self.config.m15_trend_idx])
            m5 = float(flat[self.config.m5_velocity_idx])

            if not (math.isfinite(h1) and math.isfinite(m15) and math.isfinite(m5)):
                return (0.0, 0.0, 0.0)

            return (h1, m15, m5)
        except (TypeError, ValueError, IndexError):
            return (0.0, 0.0, 0.0)

    def evaluate_confluence(
        self,
        inputs: Any,
        trade_direction: int,
    ) -> bool:
        """Evaluate whether H1 Trend, M15 Trend, and M5 Velocity align with trade direction."""
        if trade_direction not in (-1, 1):
            return False
        h1, m15, m5 = self.extract_timeframe_signals(inputs)
        tau = self.config.confluence_threshold
        if trade_direction == 1:
            return (h1 > tau) and (m15 > tau) and (m5 > tau)
        elif trade_direction == -1:
            return (h1 < -tau) and (m15 < -tau) and (m5 < -tau)
        return False

    def check_confluence(
        self,
        inputs: Any,
        trade_direction: int,
    ) -> bool:
        """Alias for evaluate_confluence."""
        return self.evaluate_confluence(inputs, trade_direction)

    def evaluate_confluence_direction(
        self,
        inputs: Any,
    ) -> int:
        """Evaluate 3TF agreement: returns +1 (BUY), -1 (SELL), or 0 (neutral/mixed)."""
        h1, m15, m5 = self.extract_timeframe_signals(inputs)
        tau = self.config.confluence_threshold
        if (h1 > tau) and (m15 > tau) and (m5 > tau):
            return 1
        if (h1 < -tau) and (m15 < -tau) and (m5 < -tau):
            return -1
        return 0

    # --- Core Dynamic Risk Computation ---
    def compute_dynamic_risk(
        self,
        base_risk_pct: float,
        current_equity: float,
        inputs: Optional[Any] = None,
        trade_direction: int = 0,
    ) -> float:
        """Compute dynamically scaled risk_pct with asymmetric pumping and hard 50% cushion cap.

        Guaranteed to return a finite float in [0.0, 1.0], and strictly <= (0.50 * cushion) / current_equity.
        """
        self._trade_eval_count += 1

        # Guard 1: Validate equity
        if not math.isfinite(current_equity) or current_equity <= 0.0:
            return 0.0

        # Guard 2: Validate base risk percentage
        if not math.isfinite(base_risk_pct) or base_risk_pct <= 0.0:
            return 0.0

        # Update and query CPPI state
        peak, floor, cushion = self.update_equity(current_equity)

        # Guard 3: Zero or negative cushion (equity <= floor)
        if cushion <= 0.0:
            return 0.0

        # Invariant 1: Hard 50% Cushion Cap (loss <= 0.50 * Cushion)
        cushion_cap = (self.config.cushion_loss_fraction * cushion) / float(current_equity)

        # Guard 4: Zero peak equity denominator guard
        denominator = self.config.max_allowed_dd * peak
        if denominator <= 1e-12:
            cushion_ratio = 0.0
        else:
            cushion_ratio = cushion / denominator

        # Invariant 2: Cushion Ratio Clamping [0.0, max_cushion_ratio]
        clamped_cushion_ratio = max(0.0, min(self.config.max_cushion_ratio, cushion_ratio))
        alpha = self.config.base_alpha + self.config.pumping_slope * clamped_cushion_ratio

        # Invariant 3: 3-Timeframe Confluence Boost (Indices 88, 84, 81)
        confluence_mult = 1.0
        if inputs is not None and trade_direction in (-1, 1):
            if self.evaluate_confluence(inputs, trade_direction):
                confluence_mult = self.config.confluence_boost

        dynamic_alpha = alpha * confluence_mult
        if self.config.max_alpha is not None:
            dynamic_alpha = min(self.config.max_alpha, dynamic_alpha)

        unconstrained_risk = float(base_risk_pct) * dynamic_alpha

        # Invariant 4: Hard Cushion Cap Dominance
        final_risk = min(unconstrained_risk, cushion_cap)

        # Invariant 5: Global Ceilings
        if self.config.max_risk_cap is not None:
            final_risk = min(final_risk, self.config.max_risk_cap)

        return float(max(0.0, min(1.0, final_risk)))

    def get_telemetry(self) -> dict[str, Any]:
        """Return audit dictionary of current state, metrics, and diagnostics."""
        return {
            "peak_equity": self._peak_equity,
            "floor": self._floor,
            "cushion": self._cushion,
            "cushion_ratio": self._cushion_ratio,
            "current_equity": self._last_equity,
            "trade_eval_count": self._trade_eval_count,
            "max_allowed_dd": self.config.max_allowed_dd,
            "base_alpha": self.config.base_alpha,
            "pumping_slope": self.config.pumping_slope,
            "max_cushion_ratio": self.config.max_cushion_ratio,
            "cushion_loss_fraction": self.config.cushion_loss_fraction,
            "confluence_boost": self.config.confluence_boost,
        }


# ============================================================================
# Milestone M3: Dynamic Bidirectional CPPI & Concurrency Position Governance
# ============================================================================

@dataclass(frozen=True)
class BidirectionalCPPIConfig:
    """Configuration for Bidirectional CPPI Risk Engine & Concurrency Governance."""

    max_allowed_dd: float = 0.08          # Maximum permitted drawdown from peak (8.0% default per dispatch R3)
    min_risk_pct: float = 0.0015          # 0.15% Fractional Kelly base risk
    max_risk_pct: float = 0.0025          # 0.25% Fractional Kelly max risk
    cushion_loss_fraction: float = 0.50   # Hard barrier: single trade loss capped at 50% of remaining cushion
    max_concurrent_positions: int = 5     # Maximum concurrent positions global
    max_positions_per_direction: int = 2  # Maximum positions per direction (BUY or SELL)
    min_spacing_bars: int = 3             # Minimum bars spacing between entries
    initial_equity: float = 0.0           # Initial baseline equity

    def __post_init__(self) -> None:
        """Validate parameter invariants strictly upon instantiation."""
        if not (0.0 < float(self.max_allowed_dd) < 1.0):
            raise ValueError(f"max_allowed_dd must be in (0, 1), got {self.max_allowed_dd}")
        if not (0.0 < float(self.min_risk_pct) <= float(self.max_risk_pct) <= 1.0):
            raise ValueError(
                f"Invalid risk bounds: min_risk_pct={self.min_risk_pct}, max_risk_pct={self.max_risk_pct}"
            )
        if not (0.0 < float(self.cushion_loss_fraction) <= 1.0):
            raise ValueError(f"cushion_loss_fraction must be in (0, 1], got {self.cushion_loss_fraction}")
        if int(self.max_concurrent_positions) <= 0:
            raise ValueError(f"max_concurrent_positions must be positive, got {self.max_concurrent_positions}")
        if int(self.max_positions_per_direction) <= 0:
            raise ValueError(f"max_positions_per_direction must be positive, got {self.max_positions_per_direction}")
        if int(self.min_spacing_bars) < 0:
            raise ValueError(f"min_spacing_bars must be non-negative, got {self.min_spacing_bars}")


class BidirectionalCPPIEngine:
    """Dynamic Bidirectional Constant Proportion Portfolio Insurance (CPPI) & Concurrency Risk Engine.

    Symmetrically manages portfolio risk for BUY and SELL orders:
    - Floor: (1 - max_allowed_dd) * peak_equity
    - Cushion: max(0.0, equity - floor)
    - Cushion ratio: cushion / (max_allowed_dd * peak_equity)
    - Dynamic risk scaling: Scales continuously between min_risk_pct (0.15%) and max_risk_pct (0.25%)
      based on cushion ratio (Fractional Kelly).
    - Hard single-trade loss cap: Maximum loss <= cushion_loss_fraction * cushion (50%).
    - Concurrency gating: Max 5 total open positions, max 2 directional (BUY or SELL).
    """

    def __init__(
        self,
        config: Optional[BidirectionalCPPIConfig] = None,
        initial_equity: float = 0.0,
    ) -> None:
        self.config: BidirectionalCPPIConfig = config if config is not None else BidirectionalCPPIConfig()
        init_eq = float(initial_equity) if initial_equity > 0.0 else float(self.config.initial_equity)
        self._peak_equity: float = float(max(0.0, init_eq))
        self._floor: float = (1.0 - self.config.max_allowed_dd) * self._peak_equity if self._peak_equity > 0.0 else 0.0
        self._cushion: float = 0.0
        self._cushion_ratio: float = 0.0
        self._last_equity: float = self._peak_equity
        self._eval_count: int = 0

        if self._peak_equity > 0.0:
            self.update_equity(self._peak_equity)

    @property
    def peak_equity(self) -> float:
        return self._peak_equity

    @property
    def floor(self) -> float:
        return self._floor

    @property
    def cushion(self) -> float:
        return self._cushion

    @property
    def cushion_ratio(self) -> float:
        return self._cushion_ratio

    @property
    def current_equity(self) -> float:
        return self._last_equity

    def reset(self, initial_equity: float = 0.0) -> None:
        """Reset engine state cleanly."""
        init_eq = float(max(0.0, initial_equity)) if initial_equity > 0.0 else float(self.config.initial_equity)
        self._peak_equity = init_eq
        self._floor = (1.0 - self.config.max_allowed_dd) * self._peak_equity if self._peak_equity > 0.0 else 0.0
        self._cushion = 0.0
        self._cushion_ratio = 0.0
        self._last_equity = self._peak_equity
        self._eval_count = 0
        if self._peak_equity > 0.0:
            self.update_equity(self._peak_equity)

    def update_equity(self, current_equity: float) -> Tuple[float, float, float]:
        """Update continuous equity, ratcheting peak_equity and updating floor & cushion."""
        if not math.isfinite(current_equity) or current_equity <= 0.0:
            return self._peak_equity, self._floor, self._cushion

        eq = float(current_equity)
        if eq > self._peak_equity:
            self._peak_equity = eq

        self._last_equity = eq
        self._floor = (1.0 - self.config.max_allowed_dd) * self._peak_equity
        self._cushion = max(0.0, self._last_equity - self._floor)

        denom = self.config.max_allowed_dd * self._peak_equity
        self._cushion_ratio = (self._cushion / denom) if denom > 1e-12 else 0.0
        return self._peak_equity, self._floor, self._cushion

    @staticmethod
    def _classify_position_direction(pos: Mapping[str, Any]) -> int:
        """Classify position as BUY (+1), SELL (-1), or 0 (unknown)."""
        if "side" in pos:
            val = pos["side"]
            if isinstance(val, (int, float)):
                return 1 if val > 0 else (-1 if val < 0 else 0)
            if isinstance(val, str):
                s = val.strip().upper()
                if s in ("BUY", "LONG", "1", "+1"):
                    return 1
                if s in ("SELL", "SHORT", "-1"):
                    return -1
        if "direction" in pos:
            val = pos["direction"]
            if isinstance(val, (int, float)):
                return 1 if val > 0 else (-1 if val < 0 else 0)
            if isinstance(val, str):
                s = val.strip().upper()
                if s in ("BUY", "LONG", "1", "+1"):
                    return 1
                if s in ("SELL", "SHORT", "-1"):
                    return -1
        if "is_long" in pos:
            return 1 if bool(pos["is_long"]) else -1
        if "type" in pos:
            val = pos["type"]
            if isinstance(val, str):
                s = val.strip().upper()
                if s in ("BUY", "LONG"):
                    return 1
                if s in ("SELL", "SHORT"):
                    return -1
            if isinstance(val, (int, float)):
                return 1 if int(val) == 0 else (-1 if int(val) == 1 else 0)
        return 0

    def evaluate_order_risk(
        self,
        equity: float,
        peak_equity: float,
        proposed_direction: str,
        current_open_positions: Sequence[Mapping[str, Any]],
        atr_m1: float = 0.0,
    ) -> Tuple[bool, float, str]:
        """Evaluate order risk under CPPI floor protection and concurrency governance.

        Returns:
            allowed: bool (False if concurrency cap or floor breach reached)
            risk_pct: float (0.15% - 0.25% Fractional Kelly or cushion cap, 0.0 if not allowed)
            reason: str ('order_admitted', 'max_concurrent_positions_reached', etc.)
        """
        self._eval_count += 1

        # 1. Global Concurrency Cap Check (Max 5)
        if len(current_open_positions) >= self.config.max_concurrent_positions:
            return False, 0.0, "max_concurrent_positions_reached"

        # 2. Directional Concurrency Cap Check (Max 2 BUY, Max 2 SELL)
        dir_clean = str(proposed_direction).strip().upper()
        is_buy = dir_clean in ("BUY", "LONG", "1", "+1")
        is_sell = dir_clean in ("SELL", "SHORT", "-1")

        if not is_buy and not is_sell:
            return False, 0.0, f"unknown_direction_{proposed_direction}"

        open_buys = sum(1 for p in current_open_positions if self._classify_position_direction(p) > 0)
        open_sells = sum(1 for p in current_open_positions if self._classify_position_direction(p) < 0)

        if is_buy and open_buys >= self.config.max_positions_per_direction:
            return False, 0.0, "max_directional_buys_reached"
        if is_sell and open_sells >= self.config.max_positions_per_direction:
            return False, 0.0, "max_directional_sells_reached"

        # 3. Numeric validity
        if not math.isfinite(equity) or equity <= 0.0:
            return False, 0.0, "invalid_equity"

        eff_peak = max(float(peak_equity) if math.isfinite(peak_equity) else 0.0, float(equity), self._peak_equity)
        if eff_peak <= 0.0:
            eff_peak = float(equity)
        self._peak_equity = eff_peak
        self._last_equity = float(equity)

        # 4. CPPI Floor & Cushion
        floor = (1.0 - self.config.max_allowed_dd) * eff_peak
        self._floor = floor
        cushion = max(0.0, float(equity) - floor)
        self._cushion = cushion

        if cushion <= 0.0:
            return False, 0.0, "cushion_depleted_capital_floor_reached"

        denom = self.config.max_allowed_dd * eff_peak
        cushion_ratio = (cushion / denom) if denom > 1e-12 else 0.0
        self._cushion_ratio = cushion_ratio

        # 5. Continuous Fractional Kelly Risk Scaling (0.15% - 0.25%)
        clamped_ratio = max(0.0, min(1.0, cushion_ratio))
        kelly_risk = self.config.min_risk_pct + (self.config.max_risk_pct - self.config.min_risk_pct) * clamped_ratio

        # 6. Hard 50% Cushion Cap (loss on single stopped-out trade <= 0.50 * cushion)
        cushion_cap = (self.config.cushion_loss_fraction * cushion) / float(equity)

        # Final risk is bounded by cushion cap to guarantee capital floor protection
        final_risk = min(kelly_risk, cushion_cap)
        if final_risk <= 0.0:
            return False, 0.0, "cushion_cap_zero_risk"

        # Global ceiling is max_risk_pct
        final_risk = min(float(self.config.max_risk_pct), float(final_risk))

        return True, float(final_risk), "order_admitted"

    def get_telemetry(self) -> dict[str, Any]:
        """Return audit dictionary of current state, metrics, and diagnostics."""
        return {
            "peak_equity": self._peak_equity,
            "floor": self._floor,
            "cushion": self._cushion,
            "cushion_ratio": self._cushion_ratio,
            "current_equity": self._last_equity,
            "eval_count": self._eval_count,
            "max_allowed_dd": self.config.max_allowed_dd,
            "min_risk_pct": self.config.min_risk_pct,
            "max_risk_pct": self.config.max_risk_pct,
            "cushion_loss_fraction": self.config.cushion_loss_fraction,
            "max_concurrent_positions": self.config.max_concurrent_positions,
            "max_positions_per_direction": self.config.max_positions_per_direction,
            "min_spacing_bars": self.config.min_spacing_bars,
        }

