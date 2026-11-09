"""
Module R3: Cross-Asset Macro Reasoning Engine
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Reasoning Engine
Milestone: M1 (Cross-Asset Macro Reasoning)

Responsibilities:
1. Cross-Asset Ingestion: Continuous tracking of DXY, US10Y, SPX, XAUUSD.
2. Multi-Horizon Momentum & Velocity: Returns over 5, 15, and 60 bars.
3. Cross-Asset Correlation Topology: 6 pairwise rolling Pearson correlations.
4. Global Macro Regime Scoring: Risk-On vs. Risk-Off index I_macro in [-1.0, 1.0].
5. Dense 32-Float Context Embedding: Observation features for xLSTM & CoT reasoning.
6. Zero NaN / Zero Inf Invariants (AC-7) & Sub-50ms Execution (AC-6).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np


class MacroAsset(Enum):
    DXY = "DXY"        # US Dollar Index (Reserve currency liquidity)
    US10Y = "US10Y"    # 10Y US Treasury Yield (Discount rate / Cost of capital)
    SPX = "SPX"        # S&P 500 Index (Global institutional risk appetite)
    XAUUSD = "XAUUSD"  # Gold (Monetary debasement / Safe haven)


class MacroRegime(Enum):
    STRONG_RISK_ON = "STRONG_RISK_ON"    # I_macro > 0.35
    MILD_RISK_ON = "MILD_RISK_ON"        # 0.10 < I_macro <= 0.35
    NEUTRAL = "NEUTRAL"                  # -0.10 <= I_macro <= 0.10
    MILD_RISK_OFF = "MILD_RISK_OFF"      # -0.35 <= I_macro < -0.10
    STRONG_RISK_OFF = "STRONG_RISK_OFF"  # I_macro < -0.35


SUPPORTED_MACRO_ASSETS: List[str] = ["DXY", "US10Y", "SPX", "XAUUSD"]

DEFAULT_MACRO_BASELINES: Dict[str, float] = {
    "DXY": 104.0,
    "US10Y": 4.25,
    "SPX": 5000.0,
    "XAUUSD": 2350.0,
}


@dataclass
class MacroConfig:
    """Configuration parameters for Cross-Asset Macro Reasoning Engine."""
    lookback_window: int = 60            # Bar buffer depth for slow horizon and correlation
    fast_horizon: int = 5                # Fast return lookback
    mid_horizon: int = 15                # Medium return lookback
    slow_horizon: int = 60               # Slow return lookback
    return_scaling: float = 10.0         # Tanh scaling factor for returns (avoids float32 saturation)
    vol_scaling: float = 500.0           # Tanh scaling factor for volatility
    risk_on_threshold: float = 0.35      # Threshold for STRONG_RISK_ON
    risk_off_threshold: float = -0.35    # Threshold for STRONG_RISK_OFF
    mild_risk_on_threshold: float = 0.10 # Threshold for MILD_RISK_ON
    mild_risk_off_threshold: float = -0.10
    dense_latent_dim: int = 4
    eps: float = 1e-8                    # Arithmetic safeguard epsilon


@dataclass
class MacroQuote:
    """Individual real-time tick or bar quote for a macro instrument."""
    symbol: str
    price: float
    timestamp: float = 0.0
    volume: float = 0.0


@dataclass
class MacroEngineOutput:
    """Detailed output and telemetry container from CrossAssetMacroEngine."""
    features: np.ndarray                 # Shape (32,), dtype float32
    macro_regime_index: float            # I_macro in [-1.0, 1.0]
    regime: MacroRegime                  # Categorical regime classification
    dollar_pressure_score: float         # Dollar drag in [-1.0, 1.0]
    yield_pressure_score: float          # Rate drag in [-1.0, 1.0]
    equity_sentiment_score: float        # Equity risk appetite in [-1.0, 1.0]
    flight_to_safety_score: float        # Safe-haven divergence in [-1.0, 1.0]
    crypto_macro_alignment: float        # Directional crypto bias in [-1.0, 1.0]
    pairwise_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    data_integrity: float = 1.0          # Feed completeness in [0.0, 1.0]
    latency_ms: float = 0.0

    def __array__(self, dtype=None):
        if dtype is None:
            return self.features
        return self.features.astype(dtype)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx]


MACRO_FEATURE_NAMES: List[str] = [
    # 0..3: DXY Features
    "dxy_ret_5", "dxy_ret_15", "dxy_ret_60", "dxy_vol_60",
    # 4..7: US10Y Features
    "us10y_ret_5", "us10y_ret_15", "us10y_ret_60", "us10y_vol_60",
    # 8..11: SPX Features
    "spx_ret_5", "spx_ret_15", "spx_ret_60", "spx_vol_60",
    # 12..15: XAUUSD Features
    "xau_ret_5", "xau_ret_15", "xau_ret_60", "xau_vol_60",
    # 16..21: Pairwise Rolling Pearson Correlations
    "rho_dxy_us10y", "rho_dxy_spx", "rho_dxy_xau",
    "rho_us10y_spx", "rho_us10y_xau", "rho_spx_xau",
    # 22..27: Macro Regime & Sub-Factors
    "macro_regime_index", "dollar_pressure_score", "yield_pressure_score",
    "equity_sentiment_score", "flight_to_safety_score", "crypto_macro_alignment",
    # 28..31: Systemic Macro Metrics
    "velocity_spread", "macro_stress_vol", "global_liquidity_expansion",
    "data_integrity_indicator"
]

assert len(MACRO_FEATURE_NAMES) == 32, f"Registry must have exactly 32 features, got {len(MACRO_FEATURE_NAMES)}"


class CrossAssetMacroEngine:
    """
    R3 Engine: Cross-Asset Macro Reasoning Engine.
    Tracks DXY, US10Y, SPX, XAUUSD over 5, 15, and 60-bar windows,
    synthesizing a dense 32-float embedding and Risk-On/Off scoring.
    """

    FEATURE_COUNT: int = 32
    FEATURE_NAMES: List[str] = MACRO_FEATURE_NAMES

    def __init__(
        self,
        lookback_window: int = 60,
        symbols: Optional[List[str]] = None,
        config: Optional[MacroConfig] = None,
    ) -> None:
        self.lookback_window: int = int(lookback_window)
        self.symbols: List[str] = symbols or list(SUPPORTED_MACRO_ASSETS)
        self.config: MacroConfig = config or MacroConfig(lookback_window=self.lookback_window)

        # Internal rolling price storage
        self._prices: Dict[str, List[float]] = {s: [] for s in self.symbols}
        self._timestamps: Dict[str, List[float]] = {s: [] for s in self.symbols}

    def update_quote(
        self, symbol: str, price: float, timestamp: float = 0.0
    ) -> None:
        """Register a real-time quote for a macro instrument with sanitization."""
        if symbol not in self._prices:
            return

        price_flt = float(price)
        # Sanitize input: reject NaN / Inf / non-positive prices
        if math.isnan(price_flt) or math.isinf(price_flt) or price_flt <= 0.0:
            price_flt = (
                self._prices[symbol][-1]
                if self._prices[symbol]
                else DEFAULT_MACRO_BASELINES.get(symbol, 1.0)
            )

        buf = self._prices[symbol]
        buf.append(price_flt)
        self._timestamps[symbol].append(float(timestamp))

        # Bounded buffer: keep up to 4x lookback window
        max_len = self.lookback_window * 4
        if len(buf) > max_len:
            del buf[: len(buf) - max_len]
            del self._timestamps[symbol][: len(self._timestamps[symbol]) - max_len]

    def set_price_history(self, symbol: str, prices: Sequence[float]) -> None:
        """Helper to inject price history for testing or historical warmup."""
        if symbol not in self._prices:
            return
        cleaned = []
        last_valid = DEFAULT_MACRO_BASELINES.get(symbol, 1.0)
        for p in prices:
            p_flt = float(p)
            if not math.isnan(p_flt) and not math.isinf(p_flt) and p_flt > 0.0:
                cleaned.append(p_flt)
                last_valid = p_flt
            else:
                cleaned.append(last_valid)
        self._prices[symbol] = cleaned
        self._timestamps[symbol] = [float(i) for i in range(len(cleaned))]

    def ingest_matrix(self, matrix: np.ndarray) -> None:
        """
        Direct ingestion of 2D matrix shape (4, N) where rows correspond to
        [DXY, US10Y, SPX, XAUUSD].
        """
        if matrix.ndim != 2 or matrix.shape[0] != len(self.symbols):
            raise ValueError(
                f"Expected matrix of shape ({len(self.symbols)}, N), got {matrix.shape}"
            )
        for i, sym in enumerate(self.symbols):
            self.set_price_history(sym, matrix[i, :].tolist())

    def get_returns(self, symbol: str, window: Optional[int] = None) -> np.ndarray:
        """Compute rolling returns array for a symbol with zero division safeguards."""
        w = window or self.lookback_window
        series = self._prices.get(symbol, [])
        if len(series) < 2:
            return np.array([], dtype=np.float64)

        closes = np.array(series[-(w + 1):], dtype=np.float64)
        if len(closes) < 2:
            return np.array([], dtype=np.float64)

        prev_closes = closes[:-1]
        valid_mask = prev_closes > 1e-12
        if not np.all(valid_mask):
            prev_closes = np.where(valid_mask, prev_closes, 1.0)

        with np.errstate(all="ignore"):
            returns = (closes[1:] - prev_closes) / prev_closes
            return np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_pair_correlation(
        self, sym_a: str, sym_b: str, window: Optional[int] = None
    ) -> float:
        """Computes Pearson correlation between two macro assets with zero-variance protection."""
        if sym_a == sym_b:
            return 1.0

        r_a = self.get_returns(sym_a, window)
        r_b = self.get_returns(sym_b, window)
        min_len = min(len(r_a), len(r_b))
        if min_len < 5:
            return 0.0

        r_a_trimmed = r_a[-min_len:]
        r_b_trimmed = r_b[-min_len:]

        with np.errstate(all="ignore"):
            std_a = float(np.std(r_a_trimmed))
            std_b = float(np.std(r_b_trimmed))
            if std_a < 1e-8 or std_b < 1e-8:
                return 0.0

            c = np.corrcoef(r_a_trimmed, r_b_trimmed)[0, 1]
            if math.isnan(c) or math.isinf(c):
                return 0.0
            return float(np.clip(c, -1.0, 1.0))

    def extract_features(
        self, matrix: Optional[np.ndarray] = None
    ) -> MacroEngineOutput:
        """
        Extract the complete 32-float dense context embedding vector and regime diagnostics.
        Guaranteed zero NaN, zero Inf, fully bounded.
        """
        t0 = time.perf_counter()

        if matrix is not None:
            self.ingest_matrix(matrix)

        feats = np.zeros(self.FEATURE_COUNT, dtype=np.float32)
        valid_assets_count = 0

        scale = self.config.return_scaling
        vol_scale = self.config.vol_scaling

        mom_weighted: Dict[str, float] = {}
        velocities_5: Dict[str, float] = {}
        velocities_60: Dict[str, float] = {}

        # Precompute returns arrays for all symbols once
        ret_dict: Dict[str, np.ndarray] = {
            sym: self.get_returns(sym, self.lookback_window) for sym in self.symbols
        }

        # -------------------------------------------------------------
        # Features 0..15: 4 Assets x 4 Metrics (ret_5, ret_15, ret_60, vol_60)
        # -------------------------------------------------------------
        for idx, sym in enumerate(self.symbols):
            base_idx = idx * 4
            series = self._prices.get(sym, [])
            if len(series) >= 2:
                valid_assets_count += 1
                curr_p = series[-1]

                # Window 5
                p5 = series[-6] if len(series) > 5 else series[0]
                r5 = (curr_p - p5) / max(p5, 1e-6)

                # Window 15
                p15 = series[-16] if len(series) > 15 else series[0]
                r15 = (curr_p - p15) / max(p15, 1e-6)

                # Window 60
                p60 = series[-61] if len(series) > 60 else series[0]
                r60 = (curr_p - p60) / max(p60, 1e-6)

                ret_arr = ret_dict[sym]
                with np.errstate(all="ignore"):
                    vol60 = float(np.std(ret_arr)) if len(ret_arr) >= 2 else 0.0
            else:
                r5, r15, r60, vol60 = 0.0, 0.0, 0.0, 0.0

            tr5 = float(np.tanh(scale * r5))
            tr15 = float(np.tanh(scale * r15))
            tr60 = float(np.tanh(scale * r60))
            tvol60 = float(np.tanh(vol_scale * vol60))

            feats[base_idx] = tr5
            feats[base_idx + 1] = tr15
            feats[base_idx + 2] = tr60
            feats[base_idx + 3] = tvol60

            # Multi-horizon weighted momentum
            m_w = 0.20 * tr5 + 0.35 * tr15 + 0.45 * tr60
            mom_weighted[sym] = m_w
            velocities_5[sym] = tr5 / 5.0
            velocities_60[sym] = tr60 / 60.0

        # -------------------------------------------------------------
        # Features 16..21: 6 Pairwise Rolling Pearson Correlations
        # -------------------------------------------------------------
        pairs = [
            ("DXY", "US10Y"),
            ("DXY", "SPX"),
            ("DXY", "XAUUSD"),
            ("US10Y", "SPX"),
            ("US10Y", "XAUUSD"),
            ("SPX", "XAUUSD"),
        ]
        pairwise_corrs: Dict[Tuple[str, str], float] = {}

        min_ret_len = min((len(ret_dict[s]) for s in self.symbols), default=0)
        corr_matrix: Optional[np.ndarray] = None
        if min_ret_len >= 5:
            R = np.vstack([ret_dict[s][-min_ret_len:] for s in self.symbols])
            with np.errstate(all="ignore"):
                row_stds = np.std(R, axis=1)
                c_mat = np.corrcoef(R)
                c_mat = np.nan_to_num(c_mat, nan=0.0)
                zero_mask = row_stds < 1e-8
                c_mat[zero_mask, :] = 0.0
                c_mat[:, zero_mask] = 0.0
                np.fill_diagonal(c_mat, 1.0)
                corr_matrix = c_mat

        sym_idx_map = {s: i for i, s in enumerate(self.symbols)}
        for p_idx, (s1, s2) in enumerate(pairs):
            if corr_matrix is not None and s1 in sym_idx_map and s2 in sym_idx_map:
                rho = float(corr_matrix[sym_idx_map[s1], sym_idx_map[s2]])
            else:
                rho = self.compute_pair_correlation(s1, s2, self.lookback_window)
            feats[16 + p_idx] = rho
            pairwise_corrs[(s1, s2)] = rho

        # -------------------------------------------------------------
        # Features 22..27: Global Macro Regime & Sub-Factors
        # -------------------------------------------------------------
        m_spx = mom_weighted.get("SPX", 0.0)
        m_dxy = mom_weighted.get("DXY", 0.0)
        m_us10y = mom_weighted.get("US10Y", 0.0)
        m_xau = mom_weighted.get("XAUUSD", 0.0)

        # Raw composite macro score
        s_macro = (1.0 / 3.0) * m_spx - (1.0 / 3.0) * m_dxy - 0.20 * m_us10y + 0.1333 * m_xau
        i_macro = float(np.clip(np.tanh(1.5 * s_macro), -1.0, 1.0))

        dollar_pressure = float(np.tanh(-1.5 * m_dxy))
        yield_pressure = float(np.tanh(-1.5 * m_us10y))
        equity_sentiment = float(np.tanh(1.5 * m_spx))

        # Safe-haven flight-to-safety divergence: Gold return minus SPX return over 15 bars
        xau_r15 = feats[13]
        spx_r15 = feats[9]
        flight_to_safety = float(np.tanh(1.5 * (xau_r15 - spx_r15)))

        crypto_macro_alignment = float(np.clip(np.tanh(2.0 * s_macro), -1.0, 1.0))

        feats[22] = i_macro
        feats[23] = dollar_pressure
        feats[24] = yield_pressure
        feats[25] = equity_sentiment
        feats[26] = flight_to_safety
        feats[27] = crypto_macro_alignment

        # -------------------------------------------------------------
        # Features 28..31: Systemic Macro Metrics
        # -------------------------------------------------------------
        v_spread = sum(velocities_5[s] - velocities_60[s] for s in self.symbols) / float(len(self.symbols))
        feats[28] = float(np.tanh(50.0 * v_spread))

        stress_vol = sum(feats[idx * 4 + 3] for idx in range(4)) / 4.0
        feats[29] = float(stress_vol)

        liquidity_exp = float(np.tanh(-1.2 * (m_dxy + 0.8 * m_us10y)))
        feats[30] = liquidity_exp

        data_integrity = float(valid_assets_count / float(len(self.symbols)))
        feats[31] = data_integrity

        # AC-7: Absolute NaN/Inf sanitization trap
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)
        feats = np.clip(feats, -1.0, 1.0)

        # Categorical regime determination
        if i_macro > self.config.risk_on_threshold:
            regime = MacroRegime.STRONG_RISK_ON
        elif i_macro > self.config.mild_risk_on_threshold:
            regime = MacroRegime.MILD_RISK_ON
        elif i_macro >= self.config.mild_risk_off_threshold:
            regime = MacroRegime.NEUTRAL
        elif i_macro >= self.config.risk_off_threshold:
            regime = MacroRegime.MILD_RISK_OFF
        else:
            regime = MacroRegime.STRONG_RISK_OFF

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return MacroEngineOutput(
            features=feats,
            macro_regime_index=i_macro,
            regime=regime,
            dollar_pressure_score=dollar_pressure,
            yield_pressure_score=yield_pressure,
            equity_sentiment_score=equity_sentiment,
            flight_to_safety_score=flight_to_safety,
            crypto_macro_alignment=crypto_macro_alignment,
            pairwise_correlations=pairwise_corrs,
            data_integrity=data_integrity,
            latency_ms=latency_ms,
        )

    def extract_features_array(
        self, matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Direct extraction returning strictly 1D float32 numpy array of shape (32,)."""
        return self.extract_features(matrix=matrix).features

    def get_feature_dict(
        self, features: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Maps 32-D feature vector to named dictionary for telemetry and inspection."""
        arr = features if features is not None else self.extract_features().features
        if arr.shape[0] != self.FEATURE_COUNT:
            raise ValueError(f"Expected array of length {self.FEATURE_COUNT}, got {arr.shape[0]}")
        return {name: float(val) for name, val in zip(self.FEATURE_NAMES, arr)}
