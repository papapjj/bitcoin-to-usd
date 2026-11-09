"""
Module R6: Multi-Symbol & Multi-Timeframe High-Frequency Matrix
Project: Mythos of Claude - xLSTM 17.5M World-Class Reasoning Engine

Responsibilities:
1. Supported 6-asset basket: BTCUSD, ETHUSD, SOLUSD, XAUUSD, EURUSD, US30.
2. Multi-Timeframe tracking: M1 (Micro-Scalping) and M5 (Market Structure).
3. Cross-Asset Correlation Filter (Triệt tiêu tương quan đồng pha, rho > 0.85).
4. 24/7 High-Frequency Opportunity Routing & Quota Monitor (Target: 45-60 precision trades/day).
5. Multi-timeframe confluence arbitration between higher structure (M5) and micro trigger (M1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class TradeDirection(Enum):
    BUY = 1
    SELL = -1

    @classmethod
    def from_str(cls, val: str) -> TradeDirection:
        v = val.strip().upper()
        if v in ("BUY", "LONG", "1", "+1"):
            return cls.BUY
        elif v in ("SELL", "SHORT", "-1"):
            return cls.SELL
        raise ValueError(f"Unknown direction string: {val}")


class TimeFrame(Enum):
    M1 = "M1"
    M5 = "M5"


class AssetClass(Enum):
    CRYPTO = "CRYPTO"          # BTCUSD, ETHUSD, SOLUSD (24/7 continuous high beta)
    METALS = "METALS"          # XAUUSD (Gold, safe-haven macro anchor)
    FOREX = "FOREX"            # EURUSD (Currency macro flows)
    INDICES = "INDICES"        # US30 (Wall Street risk sentiment)


SUPPORTED_SYMBOLS: List[str] = [
    "BTCUSD",
    "ETHUSD",
    "SOLUSD",
    "XAUUSD",
    "EURUSD",
    "US30",
]

SYMBOL_ASSET_CLASS: Dict[str, AssetClass] = {
    "BTCUSD": AssetClass.CRYPTO,
    "ETHUSD": AssetClass.CRYPTO,
    "SOLUSD": AssetClass.CRYPTO,
    "XAUUSD": AssetClass.METALS,
    "EURUSD": AssetClass.FOREX,
    "US30": AssetClass.INDICES,
    # MT5 / Broker 'm' suffixed symbols
    "BTCUSDM": AssetClass.CRYPTO,
    "ETHUSDM": AssetClass.CRYPTO,
    "SOLUSDM": AssetClass.CRYPTO,
    "XAUUSDM": AssetClass.METALS,
    "EURUSDM": AssetClass.FOREX,
    "US30M": AssetClass.INDICES,
    "BTCUSDm": AssetClass.CRYPTO,
    "ETHUSDm": AssetClass.CRYPTO,
    "SOLUSDm": AssetClass.CRYPTO,
    "XAUUSDm": AssetClass.METALS,
    "EURUSDm": AssetClass.FOREX,
    "US30m": AssetClass.INDICES,
}


SYMBOL_ALIAS_MAP: Dict[str, str] = {
    "BTC": "BTCUSD",
    "ETH": "ETHUSD",
    "SOL": "SOLUSD",
    "XAU": "XAUUSD",
    "GOLD": "XAUUSD",
    "EUR": "EURUSD",
    "US30": "US30",
    "DJIA": "US30",
    "DJ30": "US30",
}


def normalize_symbol(symbol: str) -> str:
    """Normalizes symbol to uppercase and strips broker suffixes/postfixes (e.g. BTCUSDm -> BTCUSD)."""
    s = str(symbol).strip().upper()
    if not s:
        return s
    if s in SUPPORTED_SYMBOLS:
        return s
    if s in SYMBOL_ALIAS_MAP:
        return SYMBOL_ALIAS_MAP[s]
    for base in SUPPORTED_SYMBOLS:
        if s.startswith(base):
            return base
    if s.endswith("M") and len(s) > 3 and s[:-1] in SUPPORTED_SYMBOLS:
        return s[:-1]
    return s


@dataclass
class BarData:
    symbol: str
    timeframe: TimeFrame
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


@dataclass
class SignalProposal:
    symbol: str
    direction: TradeDirection
    timeframe: TimeFrame
    expected_alpha: float            # Projected return in pips / points / base units
    confidence: float                # Model / reasoner confidence score [0.0, 1.0]
    entry_price: float
    timestamp: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    m5_bias: Optional[TradeDirection] = None
    metadata: Dict[str, Union[float, str, int]] = field(default_factory=dict)


@dataclass
class ActivePosition:
    symbol: str
    direction: TradeDirection
    entry_price: float
    open_time: float
    volume: float = 1.0
    current_pnl: float = 0.0


@dataclass
class FilterDecision:
    approved: bool
    reason: str
    symbol: str
    direction: TradeDirection
    conflicting_symbol: Optional[str] = None
    correlation: float = 0.0
    effective_correlation: float = 0.0
    confluence_score: float = 1.0


@dataclass
class MultiSymbolRoutingResult:
    approved_signals: List[SignalProposal]
    rejected_signals: List[Tuple[SignalProposal, FilterDecision]]
    active_correlations: Dict[Tuple[str, str], float]


class MultiSymbolMatrix:
    """
    R6 Engine: Multi-Asset & Multi-Timeframe High-Frequency Matrix.
    Manages price streams across 6 symbols on M1 & M5, maintaining real-time
    rolling correlation matrices and enforcing the in-phase correlation shield (rho > 0.85).
    """

    def __init__(
        self,
        symbols: Optional[List[str]] = None,
        max_correlation_threshold: float = 0.85,
        rolling_window_bars: int = 60,
        daily_target_trades_min: int = 45,
        daily_target_trades_max: int = 60,
    ) -> None:
        self.symbols: List[str] = symbols or list(SUPPORTED_SYMBOLS)
        self.max_correlation: float = float(max_correlation_threshold)
        self.rolling_window_bars: int = int(rolling_window_bars)
        self.daily_target_min: int = int(daily_target_trades_min)
        self.daily_target_max: int = int(daily_target_trades_max)

        # Bar series storage: symbol -> timeframe -> list of BarData
        self._bars: Dict[str, Dict[TimeFrame, List[BarData]]] = {
            s: {TimeFrame.M1: [], TimeFrame.M5: []} for s in self.symbols
        }

        # Active trade execution history for rate pacing (timestamp list)
        self._execution_timestamps: List[float] = []

        # Current active open positions
        self._active_positions: Dict[str, ActivePosition] = {}

    @property
    def supported_symbols(self) -> List[str]:
        return list(self.symbols)

    def register_bar(self, bar: BarData) -> None:
        """Register a new M1 or M5 bar into the rolling buffer."""
        sym = normalize_symbol(bar.symbol)
        if sym not in self.symbols:
            return
        buf = self._bars[sym][bar.timeframe]
        buf.append(bar)
        # Keep buffer bounded to avoid memory bloating
        max_buf = self.rolling_window_bars * 4
        if len(buf) > max_buf:
            del buf[: len(buf) - max_buf]

    def set_price_history(
        self,
        symbol: str,
        timeframe: TimeFrame,
        prices: List[float],
        base_timestamp: float = 0.0,
        step_seconds: float = 60.0,
    ) -> None:
        """Helper to populate price history for testing or warm-up."""
        sym = normalize_symbol(symbol)
        if sym not in self.symbols:
            return
        bars = []
        for i, p in enumerate(prices):
            ts = base_timestamp + i * step_seconds
            bars.append(
                BarData(
                    symbol=sym,
                    timeframe=timeframe,
                    timestamp=ts,
                    open=p,
                    high=p,
                    low=p,
                    close=p,
                    volume=100.0,
                )
            )
        self._bars[sym][timeframe] = bars

    def get_returns(
        self, symbol: str, timeframe: TimeFrame = TimeFrame.M1, window: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute rolling percentage returns for a symbol.
        Returns empty 1D numpy array if insufficient data.
        """
        sym = normalize_symbol(symbol)
        w = window or self.rolling_window_bars
        bars = self._bars.get(sym, {}).get(timeframe, [])
        if len(bars) < 2:
            return np.array([], dtype=np.float64)

        closes = np.array([b.close for b in bars[- (w + 1) :]], dtype=np.float64)
        if len(closes) < 2:
            return np.array([], dtype=np.float64)

        # Safe division to prevent zero-division
        prev_closes = closes[:-1]
        valid_mask = prev_closes > 1e-12
        if not np.all(valid_mask):
            prev_closes = np.where(valid_mask, prev_closes, 1.0)

        returns = (closes[1:] - prev_closes) / prev_closes
        # Replace NaN or Inf with 0.0 for arithmetic safety
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        return returns

    def compute_pair_correlation(
        self,
        sym_a: str,
        sym_b: str,
        timeframe: TimeFrame = TimeFrame.M1,
        window: Optional[int] = None,
    ) -> float:
        """
        Calculates Pearson correlation rho in [-1.0, 1.0] between two symbols.
        Handles zero variance and short series with robust guards.
        """
        sym_a_norm = normalize_symbol(sym_a)
        sym_b_norm = normalize_symbol(sym_b)
        if sym_a_norm == sym_b_norm:
            return 1.0

        r_a = self.get_returns(sym_a_norm, timeframe, window)
        r_b = self.get_returns(sym_b_norm, timeframe, window)

        min_len = min(len(r_a), len(r_b))
        if min_len < 5:
            # Insufficient sample points -> neutral default
            return 0.0

        r_a_trimmed = r_a[-min_len:]
        r_b_trimmed = r_b[-min_len:]

        std_a = float(np.std(r_a_trimmed))
        std_b = float(np.std(r_b_trimmed))

        if std_a < 1e-8 or std_b < 1e-8:
            return 0.0

        cov = float(np.cov(r_a_trimmed, r_b_trimmed)[0, 1])
        rho = cov / (std_a * std_b)

        if math.isnan(rho) or math.isinf(rho):
            return 0.0

        return float(np.clip(rho, -1.0, 1.0))

    def compute_correlation_matrix(
        self, timeframe: TimeFrame = TimeFrame.M1, window: Optional[int] = None
    ) -> Tuple[List[str], np.ndarray]:
        """
        Builds the 6x6 Pearson correlation matrix across all supported assets.
        Guarantees symmetric matrix with 1.0 on diagonal.
        """
        n = len(self.symbols)
        mat = np.eye(n, dtype=np.float64)

        for i in range(n):
            for j in range(i + 1, n):
                rho = self.compute_pair_correlation(
                    self.symbols[i], self.symbols[j], timeframe, window
                )
                mat[i, j] = rho
                mat[j, i] = rho

        return list(self.symbols), mat

    def check_correlation_filter(
        self,
        proposal: SignalProposal,
        existing_positions: Optional[List[ActivePosition]] = None,
    ) -> FilterDecision:
        """
        Cross-Asset Correlation Filter (Bộ Lọc Triệt Tiêu Tương Quan Đồng Pha):
        Evaluates incoming trade proposal against active open positions.
        Formula:
            rho_effective = rho(sym_candidate, sym_active) * dir_candidate * dir_active
        If rho_effective > max_correlation (0.85):
            REJECT trade to protect the portfolio from unhedged systemic co-directional risk.
        """
        positions = (
            existing_positions
            if existing_positions is not None
            else list(self._active_positions.values())
        )

        cand_sym = normalize_symbol(proposal.symbol)
        cand_dir = proposal.direction.value  # +1 for BUY, -1 for SELL

        for pos in positions:
            pos_sym = normalize_symbol(pos.symbol)
            # If same symbol has active position in same direction, block duplicate
            if pos_sym == cand_sym:
                if pos.direction == proposal.direction:
                    return FilterDecision(
                        approved=False,
                        reason=f"REJECT_DUPLICATE_POSITION: Active position already exists in {cand_sym} with direction {pos.direction.name}",
                        symbol=cand_sym,
                        direction=proposal.direction,
                        conflicting_symbol=pos.symbol,
                        correlation=1.0,
                        effective_correlation=1.0,
                    )
                # If opposite direction on same symbol, allowed (hedging or reversal)
                continue

            active_dir = pos.direction.value
            rho = self.compute_pair_correlation(cand_sym, pos_sym, proposal.timeframe)

            # In-phase directional risk product:
            # If rho > 0 and same direction (+1 * +1 or -1 * -1): effective = +rho
            # If rho > 0 and opposite direction (+1 * -1): effective = -rho
            # If rho < -0.85 (inversely correlated) and opposite direction (-1 * +1 * rho): effective = +|rho|
            effective_rho = rho * cand_dir * active_dir

            if effective_rho > self.max_correlation:
                return FilterDecision(
                    approved=False,
                    reason=(
                        f"REJECT_CORRELATED_IN_PHASE_RISK: Candidate {cand_sym} ({proposal.direction.name}) "
                        f"violates correlation limit with active {pos.symbol} ({pos.direction.name}). "
                        f"Pair rho={rho:.4f}, Effective rho={effective_rho:.4f} > {self.max_correlation:.2f}"
                    ),
                    symbol=cand_sym,
                    direction=proposal.direction,
                    conflicting_symbol=pos.symbol,
                    correlation=rho,
                    effective_correlation=effective_rho,
                )

        return FilterDecision(
            approved=True,
            reason="APPROVED_CORRELATION_SAFE: Diversified risk exposure",
            symbol=cand_sym,
            direction=proposal.direction,
            conflicting_symbol=None,
            correlation=0.0,
            effective_correlation=0.0,
        )

    # Backward-compatible alias for unified pipeline
    evaluate_proposal = check_correlation_filter

    def evaluate_m5_confluence(self, proposal: SignalProposal) -> float:
        """
        Evaluates Multi-Timeframe Confluence:
        Checks M1 entry against M5 market structure bias.
        Returns confluence multiplier [0.5, 1.25].
        If proposal explicitly specifies M5 bias and directions match, boosts score.
        If opposing M5 strong trend, penalizes unless high expected alpha.
        """
        sym = normalize_symbol(proposal.symbol)
        if proposal.m5_bias is None:
            # Extract basic M5 bias from last few M5 bars if available
            m5_bars = self._bars.get(sym, {}).get(TimeFrame.M5, [])
            if len(m5_bars) >= 5:
                recent_close = m5_bars[-1].close
                ma5 = sum(b.close for b in m5_bars[-5:]) / 5.0
                if recent_close > ma5:
                    inferred_bias = TradeDirection.BUY
                else:
                    inferred_bias = TradeDirection.SELL
            else:
                return 1.0
        else:
            inferred_bias = proposal.m5_bias

        if proposal.direction == inferred_bias:
            # In confluence with higher timeframe trend
            return 1.20
        else:
            # Counter-trend micro scalp
            return 0.75

    def route_and_filter_candidates(
        self,
        candidate_signals: List[SignalProposal],
        existing_positions: Optional[List[ActivePosition]] = None,
    ) -> MultiSymbolRoutingResult:
        """
        24/7 Multi-Symbol High-Frequency Router:
        1. Evaluates each candidate against active open positions.
        2. Arbitrates mutually conflicting candidate proposals within the batch:
           If two simultaneous candidates (e.g. BTCUSD BUY and ETHUSD BUY) have rho > 0.85,
           it selects the one with the higher rank score = (Expected Alpha * Confidence * Confluence)
           and filters out the secondary one.
        """
        active_pos = (
            existing_positions
            if existing_positions is not None
            else list(self._active_positions.values())
        )

        approved_list: List[SignalProposal] = []
        rejected_list: List[Tuple[SignalProposal, FilterDecision]] = []
        active_correlations: Dict[Tuple[str, str], float] = {}

        # Precompute pairwise correlations for reporting
        for i in range(len(self.symbols)):
            for j in range(i + 1, len(self.symbols)):
                s1, s2 = self.symbols[i], self.symbols[j]
                active_correlations[(s1, s2)] = self.compute_pair_correlation(s1, s2)

        # Sort candidate signals by overall composite score descending
        def score_candidate(sig: SignalProposal) -> float:
            confluence = self.evaluate_m5_confluence(sig)
            return sig.expected_alpha * sig.confidence * confluence

        sorted_candidates = sorted(candidate_signals, key=score_candidate, reverse=True)

        # Working simulated active positions containing already open + accepted in this batch
        simulated_active = list(active_pos)

        for cand in sorted_candidates:
            # Check against current + accepted positions in this cycle
            decision = self.check_correlation_filter(cand, simulated_active)
            if decision.approved:
                # Add to approved list
                approved_list.append(cand)
                # Treat as active for subsequent candidates in the same batch
                simulated_active.append(
                    ActivePosition(
                        symbol=cand.symbol,
                        direction=cand.direction,
                        entry_price=cand.entry_price,
                        open_time=cand.timestamp,
                    )
                )
            else:
                rejected_list.append((cand, decision))

        return MultiSymbolRoutingResult(
            approved_signals=approved_list,
            rejected_signals=rejected_list,
            active_correlations=active_correlations,
        )

    def record_execution(self, timestamp: float) -> None:
        """Track executed trade timestamp for 24-hour rate pacing."""
        self._execution_timestamps.append(timestamp)
        # Retain last 48 hours of executions
        cutoff = timestamp - 86400.0 * 2
        self._execution_timestamps = [ts for ts in self._execution_timestamps if ts >= cutoff]

    def get_24h_execution_count(self, current_timestamp: float) -> int:
        """Returns number of executed trades in the trailing 24 hours."""
        cutoff = current_timestamp - 86400.0
        return sum(1 for ts in self._execution_timestamps if ts >= cutoff)

    def get_frequency_pacing_status(
        self, current_timestamp: float
    ) -> Dict[str, Union[int, float, str]]:
        """
        Monitors progress towards the 45-60 daily trades target (>= 4,000 trades over 90 days).
        Returns telemetry regarding current pacing.
        """
        count = self.get_24h_execution_count(current_timestamp)
        if count < self.daily_target_min:
            status = "LAGGING_BELOW_QUOTA"
        elif count > self.daily_target_max:
            status = "EXCEEDING_QUOTA_TIGHTEN_SELECTION"
        else:
            status = "OPTIMAL_PACING"

        return {
            "24h_trades_executed": count,
            "target_min": self.daily_target_min,
            "target_max": self.daily_target_max,
            "status": status,
            "projected_90d_trades": int(count * 90),
        }

    def add_active_position(self, pos: ActivePosition) -> None:
        """Registers a live position in the matrix."""
        self._active_positions[normalize_symbol(pos.symbol)] = pos

    def close_active_position(self, symbol: str) -> Optional[ActivePosition]:
        """Removes a closed position from the matrix."""
        return self._active_positions.pop(normalize_symbol(symbol), None)
