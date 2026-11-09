"""Causal M15 market-regime detector — six regimes plus volatile sub-regimes.

The detector deliberately owns only the regime *state*.  It reads the latest
closed M15 candle features supplied by the MT5 bridge and emits one immediate
label; it does not debounce M1 candidates and it does not pretend that an
unobserved MLP update is a learned routing signal.  TimingNet remains the
separate learned predictor of a future transition.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Mapping, Tuple

REGIME_TANG_MANH = 0
REGIME_TANG_YEU = 1
REGIME_GIAM_MANH = 2
REGIME_GIAM_YEU = 3
REGIME_RANGING = 4
REGIME_VOLATILE = 5

REGIME_NAMES = {
    0: "TANG_MANH", 1: "TANG_YEU", 2: "GIAM_MANH",
    3: "GIAM_YEU", 4: "RANGING", 5: "VOLATILE",
}

REGIME_IDS = {
    "CTX_TANG_MANH": 0, "CTX_TANG_YEU": 1, "CTX_GIAM_MANH": 2,
    "CTX_GIAM_YEU": 3, "CTX_RANGING": 4, "CTX_VOLATILE": 5,
}

VOLATILE_SUBREGIMES = ("SPIKE", "EXPANSION", "CHOP")


@dataclass
class MarketRegimeClassifier:
    """Immediate detector over the latest *closed* M15 feature tuple.

    The former MLP was updated with the same rule label that was used for
    routing, while its prediction was never consumed.  It was therefore dead
    learning code and made the detector look adaptive when it was not.  The
    replacement is explicit: M15 data decides the current regime, and the
    independently trained TimingNet predicts upcoming changes.
    """

    timeframe: str = "M15"
    # The M15/H1 slots have two valid wire encodings.  In the historical
    # legacy contract they are divided by M1 ATR; in ``native_tf_atr_v2`` the
    # EMA/gap slots are already expressed in their own timeframe's ATR and
    # the volatility slot is a symmetric ATR ratio.  Treating one as the
    # other makes an M15 label change when only M1 ATR changes.
    mtf_scaling_mode: str = "legacy_m1_atr_v1"

    def __post_init__(self) -> None:
        if str(self.timeframe).upper() != "M15":
            raise ValueError("MarketRegimeClassifier chi ho tro M15")
        self.timeframe = "M15"
        mode = str(self.mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        if mode not in {"legacy_m1_atr_v1", "native_tf_atr_v2"}:
            raise ValueError("MarketRegimeClassifier mtf_scaling_mode khong hop le")
        self.mtf_scaling_mode = mode

    def __setstate__(self, state: Mapping[str, object]) -> None:
        """Discard obsolete M1/MLP state when loading a previous artifact."""

        _ = state
        self.timeframe = "M15"
        self.mtf_scaling_mode = "legacy_m1_atr_v1"

    @staticmethod
    def _at(market: np.ndarray, index: int, default: float = 0.0, limit: float = 10.0) -> float:
        if not (0 <= index < market.size and np.isfinite(market[index])):
            return float(default)
        return float(np.clip(float(market[index]), -limit, limit))

    @classmethod
    def m15_signature(
        cls,
        market_features: np.ndarray,
        *,
        mtf_scaling_mode: str = "legacy_m1_atr_v1",
    ) -> tuple[float, float, float, float]:
        """Stable identity of the bridge's last closed M15 bar.

        The old MTF wire fields are divided by the M1 ATR.  Caching them raw
        would therefore still churn on every M1 bar.  Ratios below cancel that
        denominator: M15 EMA/gap quantities are divided by the M15 ATR field,
        while M15 ATR is compared with H1 ATR.  Every member consequently uses
        only closed higher-timeframe values and stays constant until an M15 or
        H1 bar closes.
        """

        market = np.asarray(market_features, dtype=float).reshape(-1)
        mode = str(mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        if mode == "native_tf_atr_v2":
            # Native EMA/gap values are already M15-ATR normalized and remain
            # constant until the closed M15 candle changes.  The two
            # volatility slots still contain M1 in a reversible symmetric
            # ratio, therefore reconstruct their M15/H1 relation before
            # caching.  Rounding prevents harmless floating-point drift in a
            # repeatedly exported M1 ATR from fabricating an M15 transition.
            m15_to_h1 = cls._native_tf_ratio(
                cls._at(market, 38, default=0.0, limit=0.999999),
                cls._at(market, 42, default=0.0, limit=0.999999),
            )
            return (
                round(cls._at(market, 36), 6),
                round(cls._at(market, 37), 6),
                round(cls._at(market, 39), 6),
                round(m15_to_h1, 4),
            )
        m15_atr_over_m1 = max(0.10, abs(cls._at(market, 38, default=1.0)))
        h1_atr_over_m1 = max(0.10, abs(cls._at(market, 42, default=1.0)))
        return (
            round(cls._at(market, 36) / m15_atr_over_m1, 10),
            round(cls._at(market, 37) / m15_atr_over_m1, 10),
            round(cls._at(market, 39) / m15_atr_over_m1, 10),
            round(m15_atr_over_m1 / h1_atr_over_m1, 10),
        )

    @staticmethod
    def _native_atr_over_m1(value: float) -> float:
        """Invert ``(ATR_tf-ATR_m1)/(ATR_tf+ATR_m1)`` safely."""

        ratio = float(np.clip(value, -0.999999, 0.999999))
        return float((1.0 + ratio) / max(1.0 - ratio, 1e-6))

    @classmethod
    def _native_tf_ratio(cls, m15_ratio: float, h1_ratio: float) -> float:
        m15_over_m1 = cls._native_atr_over_m1(m15_ratio)
        h1_over_m1 = cls._native_atr_over_m1(h1_ratio)
        return float(np.clip(m15_over_m1 / max(h1_over_m1, 1e-6), 0.02, 20.0))

    @classmethod
    def directional_score(
        cls,
        market_features: np.ndarray,
        *,
        mtf_scaling_mode: str = "legacy_m1_atr_v1",
    ) -> float:
        """M15 trend strength from M15 EMA spread, slope and closed-bar gap."""

        market = np.asarray(market_features, dtype=float).reshape(-1)
        mode = str(mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        if mode == "native_tf_atr_v2":
            ema_spread = cls._at(market, 36)
            ema_slope = cls._at(market, 37)
            closed_gap = cls._at(market, 39)
        else:
            m15_atr_over_m1 = max(0.10, abs(cls._at(market, 38, default=1.0)))
            ema_spread = cls._at(market, 36) / m15_atr_over_m1
            ema_slope = cls._at(market, 37) / m15_atr_over_m1
            closed_gap = cls._at(market, 39) / m15_atr_over_m1
        return float(np.tanh((0.65 * ema_spread + 0.25 * ema_slope + 0.10 * closed_gap) / 1.40))

    @classmethod
    def timing_directional_score(
        cls,
        market_features: np.ndarray,
        *,
        mtf_scaling_mode: str = "legacy_m1_atr_v1",
    ) -> float:
        """M1/MTF projection retained solely for TimingNet deceleration labels."""

        market = np.asarray(market_features, dtype=float).reshape(-1)

        def at(index: int, default: float = 0.0, limit: float = 6.0) -> float:
            return cls._at(market, index, default=default, limit=limit)

        mode = str(mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        if mode == "native_tf_atr_v2":
            m15_direction = 0.12 * at(36) + 0.08 * at(37)
            h1_direction = 0.12 * at(40) + 0.08 * at(41)
        else:
            m15_atr = max(0.10, abs(at(38, default=1.0)))
            h1_atr = max(0.10, abs(at(42, default=1.0)))
            m15_direction = 0.12 * at(36) / m15_atr + 0.08 * at(37) / m15_atr
            h1_direction = 0.12 * at(40) / h1_atr + 0.08 * at(41) / h1_atr
        components = [
            0.08 * at(0), 0.10 * at(1), 0.12 * at(2), 0.16 * at(3),
            0.14 * at(4), 0.10 * at(5), 0.14 * at(11), 0.10 * at(12),
            0.04 * at(16), 0.03 * at(17), 0.015 * at(18, limit=2.0),
            0.10 * at(21), 0.12 * at(22), 0.08 * at(25), 0.05 * at(26),
            0.05 * at(27), 0.10 * at(32), 0.06 * at(33), m15_direction,
            h1_direction, 0.08 * at(47),
            0.08 * at(48),
        ]
        return float(np.tanh(np.sum(components) / 1.8))

    def deterministic_regime(
        self,
        market_features: np.ndarray,
        *,
        volatile_max_directional_strength: float = 0.18,
    ) -> Tuple[int, np.ndarray]:
        """Return the current, inspectable regime from the closed M15 bar."""
        market = np.asarray(market_features, dtype=float).reshape(-1)
        regime = self.rule_based_label(
            market,
            volatile_max_directional_strength=volatile_max_directional_strength,
        )
        probs = np.full(len(REGIME_NAMES), 0.02, dtype=float)
        confidence = self.rule_based_confidence(market, regime)
        probs[regime] = max(float(confidence), 0.55)
        probs /= probs.sum() + 1e-12
        return int(regime), probs

    def rule_based_confidence(self, market_features: np.ndarray, regime: int | None = None) -> float:
        market = np.asarray(market_features, dtype=float).reshape(-1)
        if market.size == 0:
            return 0.50
        regime_id = self.rule_based_label(market) if regime is None else int(regime)
        trend_score = self.directional_score(market, mtf_scaling_mode=self.mtf_scaling_mode)
        volatility = self.volatility_score(market, mtf_scaling_mode=self.mtf_scaling_mode)
        chop = self.chop_score(market, mtf_scaling_mode=self.mtf_scaling_mode)
        if regime_id == REGIME_VOLATILE:
            return float(np.clip(0.55 + 0.25 * volatility, 0.55, 0.90))
        if regime_id == REGIME_RANGING:
            return float(np.clip(0.50 + 0.35 * chop - 0.15 * abs(trend_score), 0.45, 0.85))
        return float(np.clip(0.50 + 0.25 * abs(trend_score) + 0.10 * max(0.0, 1.0 - chop), 0.50, 0.92))

    @classmethod
    def volatility_score(
        cls,
        market_features: np.ndarray,
        *,
        mtf_scaling_mode: str = "legacy_m1_atr_v1",
    ) -> float:
        """Closed-M15 ATR relative to closed-H1 ATR, independent of M1 ATR."""

        market = np.asarray(market_features, dtype=float).reshape(-1)
        mode = str(mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        raw_m15_atr_over_m1 = abs(cls._at(market, 38))
        raw_h1_atr_over_m1 = abs(cls._at(market, 42))
        # Empty/warm-up vectors must be neutral. Treating two missing values
        # as a 1:1 ATR ratio would fabricate VOLATILE from zero input.
        if raw_m15_atr_over_m1 < 1e-9 or raw_h1_atr_over_m1 < 1e-9:
            return 0.0
        if mode == "native_tf_atr_v2":
            relative_atr = cls._native_tf_ratio(
                cls._at(market, 38, limit=0.999999),
                cls._at(market, 42, limit=0.999999),
            )
            closed_gap = abs(cls._at(market, 39))
        else:
            m15_atr_over_m1 = max(0.10, raw_m15_atr_over_m1)
            h1_atr_over_m1 = max(0.10, raw_h1_atr_over_m1)
            relative_atr = m15_atr_over_m1 / h1_atr_over_m1
            closed_gap = abs(cls._at(market, 39) / m15_atr_over_m1)
        return float(np.clip(
            0.10 + 2.00 * max(0.0, relative_atr - 0.45) + 0.10 * min(2.0, closed_gap),
            0.0,
            2.0,
        ))

    def volatile_subregime(
        self,
        market_features: np.ndarray,
        *,
        spike_threshold: float = 0.55,
        expansion_threshold: float = 1.15,
    ) -> str:
        """Split broad volatile state into deterministic learnable owners."""
        market = np.asarray(market_features, dtype=float).reshape(-1)

        def at(index: int, default: float = 0.0) -> float:
            if not (0 <= index < market.size and np.isfinite(market[index])):
                return float(default)
            return float(market[index])

        # On M15 the closed-bar gap is the causal shock proxy.  Do not mix in
        # M1 breakout slots: that would silently reintroduce M1 regime churn.
        if self.mtf_scaling_mode == "native_tf_atr_v2":
            shock = abs(at(39))
        else:
            m15_atr = max(0.10, abs(at(38, 1.0)))
            shock = abs(at(39)) / m15_atr
        breakout_pressure = abs(self.directional_score(market, mtf_scaling_mode=self.mtf_scaling_mode))
        volatility = self.volatility_score(market, mtf_scaling_mode=self.mtf_scaling_mode)
        spike_limit = float(np.clip(spike_threshold, 0.0, 2.0))
        expansion_limit = float(np.clip(expansion_threshold, 0.0, 2.0))
        if shock >= spike_limit or (breakout_pressure >= 0.80 and volatility >= expansion_limit):
            return "SPIKE"
        if volatility >= expansion_limit:
            return "EXPANSION"
        return "CHOP"

    @classmethod
    def chop_score(
        cls,
        market_features: np.ndarray,
        *,
        mtf_scaling_mode: str = "legacy_m1_atr_v1",
    ) -> float:
        market = np.asarray(market_features, dtype=float).reshape(-1)
        mode = str(mtf_scaling_mode or "legacy_m1_atr_v1").strip().lower()
        trend = abs(cls.directional_score(market, mtf_scaling_mode=mode))
        if mode == "native_tf_atr_v2":
            slope = abs(cls._at(market, 37))
            gap = abs(cls._at(market, 39))
        else:
            m15_atr_over_m1 = max(0.10, abs(cls._at(market, 38, default=1.0)))
            slope = abs(cls._at(market, 37) / m15_atr_over_m1)
            gap = abs(cls._at(market, 39) / m15_atr_over_m1)
        return float(np.clip(0.86 - 0.50 * trend - 0.05 * min(2.0, slope) - 0.04 * min(2.0, gap), 0.05, 0.95))

    def rule_based_label(
        self,
        market_features: np.ndarray,
        *,
        volatile_max_directional_strength: float = 0.18,
    ) -> int:
        trend = self.directional_score(market_features, mtf_scaling_mode=self.mtf_scaling_mode)
        chop = self.chop_score(market_features, mtf_scaling_mode=self.mtf_scaling_mode)
        volatility = self.volatility_score(market_features, mtf_scaling_mode=self.mtf_scaling_mode)
        trend_strength = abs(trend)

        # Volatile is reserved for high-energy, directionally neutral states.
        # A weak but coherent trend belongs to TANG_YEU/GIAM_YEU so its expert
        # can learn directional timing instead of inheriting every noisy trend
        # sample from the volatile owner.
        volatile_direction_limit = float(np.clip(volatile_max_directional_strength, 0.0, 1.0))
        if volatility > 0.85 and trend_strength < volatile_direction_limit:
            return REGIME_VOLATILE
        if chop > 0.68 and trend_strength < 0.30:
            return REGIME_RANGING
        if trend > 0.16:
            return REGIME_TANG_MANH if trend_strength > 0.45 else REGIME_TANG_YEU
        if trend < -0.16:
            return REGIME_GIAM_MANH if trend_strength > 0.45 else REGIME_GIAM_YEU
        return REGIME_RANGING

    def copy(self) -> "MarketRegimeClassifier":
        return MarketRegimeClassifier(timeframe="M15", mtf_scaling_mode=self.mtf_scaling_mode)
