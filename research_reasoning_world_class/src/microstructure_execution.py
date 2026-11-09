"""
Module R7: Order Microstructure Execution & Slippage-Compensation Shield
Project: Mythos of Claude - xLSTM 17.5M World-Class Reasoning Engine

Responsibilities:
1. Cost Control Shield: Strict enforcement of
       Expected Alpha / (Spread + Commission + Slippage) >= 3.5
   Rejects trades where institutional fee/spread decay erodes edge.
2. Microstructure Liquidity & Rotation Pivot Detector:
   Identifies Order Book Imbalance (OBI), micro-price shifts,
   and liquidity sweep rotations (BSL/SSL sweeps, FVG retests).
3. Sniper / Maker-like Limit Entry Simulator:
   Simulates passive limit entries at liquidity rotation pivots
   to eliminate taker spread and achieve zero or negative slippage (price improvement).
4. Slippage Compensation & Edge Preservation:
   Dynamically estimates volatility-induced slippage and adjusts entry price buffers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

try:
    from research_reasoning_world_class.src.multi_symbol_matrix import normalize_symbol
except ImportError:
    try:
        from .multi_symbol_matrix import normalize_symbol
    except ImportError:
        def normalize_symbol(symbol: str) -> str:
            s = str(symbol).strip().upper()
            for base in ("BTCUSD", "ETHUSD", "SOLUSD", "XAUUSD", "EURUSD", "US30"):
                if s == base or s.startswith(base):
                    return base
            if s.endswith("M") and len(s) > 3:
                return s[:-1]
            return s


class ExecutionMode(Enum):
    MARKET = "MARKET"
    LIMIT_SNIPER = "LIMIT_SNIPER"


class RotationPivotType(Enum):
    BSL_SWEEP_REVERSAL = "BSL_SWEEP_REVERSAL"  # Buy-side liquidity swept -> Short sniper
    SSL_SWEEP_REVERSAL = "SSL_SWEEP_REVERSAL"  # Sell-side liquidity swept -> Long sniper
    FVG_ABSORPTION = "FVG_ABSORPTION"          # Fair value gap micro-pullback limit fill
    MID_QUOTE_PASSIVE = "MID_QUOTE_PASSIVE"    # Inside-spread maker entry
    NO_ROTATION = "NO_ROTATION"


@dataclass
class CostProfile:
    symbol: str
    point_size: float              # e.g., 0.00001 for EURUSD, 0.01 for XAUUSD, 1.0 for BTCUSD
    typical_spread_points: float   # in points
    typical_commission_points: float  # equivalent points round-turn
    typical_market_slippage_points: float  # in points under normal volatility
    sniper_slippage_points: float = 0.0    # passive maker-like limit slippage (<= 0)


# Baseline cost parameters for the 6 core assets
DEFAULT_COST_PROFILES: Dict[str, CostProfile] = {
    "BTCUSD": CostProfile(
        symbol="BTCUSD",
        point_size=1.0,
        typical_spread_points=4.0,
        typical_commission_points=1.5,
        typical_market_slippage_points=2.0,
        sniper_slippage_points=0.0,
    ),
    "ETHUSD": CostProfile(
        symbol="ETHUSD",
        point_size=0.1,
        typical_spread_points=0.3,
        typical_commission_points=0.15,
        typical_market_slippage_points=0.2,
        sniper_slippage_points=0.0,
    ),
    "SOLUSD": CostProfile(
        symbol="SOLUSD",
        point_size=0.01,
        typical_spread_points=0.03,
        typical_commission_points=0.015,
        typical_market_slippage_points=0.02,
        sniper_slippage_points=0.0,
    ),
    "XAUUSD": CostProfile(
        symbol="XAUUSD",
        point_size=0.01,
        typical_spread_points=0.18,
        typical_commission_points=0.07,
        typical_market_slippage_points=0.08,
        sniper_slippage_points=0.0,
    ),
    "EURUSD": CostProfile(
        symbol="EURUSD",
        point_size=0.00001,
        typical_spread_points=0.00008,     # 0.8 pip
        typical_commission_points=0.00004, # 0.4 pip
        typical_market_slippage_points=0.00003, # 0.3 pip
        sniper_slippage_points=0.0,
    ),
    "US30": CostProfile(
        symbol="US30",
        point_size=1.0,
        typical_spread_points=1.8,
        typical_commission_points=0.8,
        typical_market_slippage_points=1.2,
        sniper_slippage_points=0.0,
    ),
}

# Register common Exness broker suffixes ('m', 'M') into DEFAULT_COST_PROFILES for direct lookup compatibility
for _base_sym in list(DEFAULT_COST_PROFILES.keys()):
    _prof = DEFAULT_COST_PROFILES[_base_sym]
    DEFAULT_COST_PROFILES[f"{_base_sym}m"] = _prof
    DEFAULT_COST_PROFILES[f"{_base_sym}M"] = _prof


@dataclass
class OrderBookSnapshot:
    symbol: str
    bid_price: float
    ask_price: float
    bid_volume: float
    ask_volume: float
    timestamp: float

    @property
    def spread(self) -> float:
        return max(0.0, self.ask_price - self.bid_price)

    @property
    def mid_price(self) -> float:
        return (self.ask_price + self.bid_price) / 2.0

    @property
    def micro_price(self) -> float:
        """
        Volume-weighted micro-price:
        P_micro = (V_bid * P_ask + V_ask * P_bid) / (V_bid + V_ask)
        Reflects instantaneous order book pressure.
        """
        tot_vol = self.bid_volume + self.ask_volume
        if tot_vol < 1e-12:
            return self.mid_price
        return (self.bid_volume * self.ask_price + self.ask_volume * self.bid_price) / tot_vol

    @property
    def order_book_imbalance(self) -> float:
        """
        Normalized Order Book Imbalance (OBI) in [-1.0, 1.0]:
        OBI = (V_bid - V_ask) / (V_bid + V_ask)
        Positive: strong buying pressure; Negative: selling pressure.
        """
        tot = self.bid_volume + self.ask_volume
        if tot < 1e-12:
            return 0.0
        return (self.bid_volume - self.ask_volume) / tot


@dataclass
class LiquidityRotationPivot:
    detected: bool
    pivot_type: RotationPivotType
    pivot_price: float
    recommended_entry: float
    absorption_ratio: float = 1.0
    confidence: float = 0.0
    description: str = ""


@dataclass
class ExecutionVerdict:
    approved: bool
    symbol: str
    direction_val: int                  # +1 = BUY, -1 = SELL
    execution_mode: ExecutionMode
    target_entry_price: float
    spread: float
    commission: float
    estimated_slippage: float
    total_friction: float
    expected_alpha: float
    cost_ratio: float                   # Expected Alpha / Total Friction
    min_cost_ratio_required: float = 3.5
    fill_probability: float = 1.0
    rejection_reason: Optional[str] = None
    telemetry: Dict[str, Union[float, str, bool]] = field(default_factory=dict)


class MicrostructureExecutionShield:
    """
    R7 Engine: Order Microstructure Execution & Slippage-Compensation Shield.
    Guarantees cost control: Expected Alpha / (Spread + Commission + Slippage) >= 3.5.
    Employs Sniper / Maker-like entry at liquidity rotation pivots to eradicate slippage decay.
    """

    MIN_COST_RATIO: float = 3.5

    def __init__(
        self,
        min_cost_ratio: float = 3.5,
        custom_cost_profiles: Optional[Dict[str, CostProfile]] = None,
    ) -> None:
        self.min_cost_ratio: float = float(min_cost_ratio)
        self.cost_profiles: Dict[str, CostProfile] = dict(DEFAULT_COST_PROFILES)
        if custom_cost_profiles:
            self.cost_profiles.update(custom_cost_profiles)

    def get_cost_profile(self, symbol: str) -> CostProfile:
        # 1. Exact raw match (handles direct or custom profiles)
        if symbol in self.cost_profiles:
            return self.cost_profiles[symbol]
        # 2. Normalized base symbol match (e.g. BTCUSD.m, BTCUSD_raw -> BTCUSD)
        norm_sym = normalize_symbol(symbol)
        if norm_sym in self.cost_profiles:
            return self.cost_profiles[norm_sym]
        # 3. Generic fallback profile
        return CostProfile(
            symbol=symbol,
            point_size=0.01,
            typical_spread_points=0.2,
            typical_commission_points=0.1,
            typical_market_slippage_points=0.1,
            sniper_slippage_points=0.0,
        )

    def calculate_cost_ratio(
        self,
        expected_alpha: float,
        spread: float,
        commission: float,
        slippage: float,
    ) -> float:
        """
        Institutional Cost Formula:
        Ratio = Expected Alpha / (Spread + Commission + Slippage)
        Guards against zero-division and negative friction values.
        """
        total_friction = max(1e-12, spread + commission + slippage)
        if expected_alpha <= 0.0:
            return 0.0
        ratio = expected_alpha / total_friction
        if math.isnan(ratio) or math.isinf(ratio):
            return 0.0
        return float(ratio)

    def estimate_market_slippage(
        self,
        symbol: str,
        book: OrderBookSnapshot,
        lot_size: float = 1.0,
        volatility_factor: float = 1.0,
        direction_val: int = 1,
    ) -> float:
        """
        Dynamic adverse slippage model for Market Taker orders:
        Slippage increases with:
        1. Volatility factor (> 1.0 during news or rapid bursts)
        2. Order size vs available opposite book depth
        3. Adverse Order Book Imbalance (e.g. buying when OBI is negative)
        """
        profile = self.get_cost_profile(symbol)
        base_slippage = profile.typical_market_slippage_points

        # Impact of order book imbalance against our trade
        obi = book.order_book_imbalance
        # If buying (+1) and OBI is negative (selling pressure), liquidity is thinner on asks
        adverse_pressure = max(0.0, -obi * direction_val)

        # Scale by volume and volatility
        vol_scale = math.sqrt(max(0.1, lot_size))
        dynamic_slippage = base_slippage * max(1.0, volatility_factor) * (1.0 + 0.5 * adverse_pressure) * vol_scale

        return float(dynamic_slippage)

    def detect_liquidity_rotation_pivot(
        self,
        symbol: str,
        direction_val: int,
        book: OrderBookSnapshot,
        recent_high: float,
        recent_low: float,
        fvg_level: Optional[float] = None,
    ) -> LiquidityRotationPivot:
        """
        Detects Institutional Liquidity Rotation Pivots (SMC/ICT Sweeps & Absorption):
        1. BSL Sweep Reversal (for SELL): Price swept above recent swing high,
           exhausted liquidity, and is rotating downward into the spread.
        2. SSL Sweep Reversal (for BUY): Price swept below recent swing low,
           exhausted sell stops, and is rotating upward into the spread.
        3. FVG Absorption: Micro-pullback into a fair value gap / imbalance zone.
        4. Mid-Quote Passive: Resting maker limit order placed at bid or mid-spread.
        """
        current_bid = book.bid_price
        current_ask = book.ask_price
        spread = book.spread

        if direction_val == 1:  # BUY
            # Check if recent low was swept (SSL sweep) and price rejected back above
            if current_bid <= recent_low + spread * 1.5:
                # SSL Sweep detected: Smart money absorbed sell liquidity at discount
                return LiquidityRotationPivot(
                    detected=True,
                    pivot_type=RotationPivotType.SSL_SWEEP_REVERSAL,
                    pivot_price=recent_low,
                    recommended_entry=book.bid_price,  # Maker limit placed at bid (zero taker spread)
                    absorption_ratio=1.45,
                    confidence=0.92,
                    description=f"SSL Sweep at {recent_low:.5f} with bid resting fill",
                )
            elif fvg_level is not None and current_bid <= fvg_level + spread:
                return LiquidityRotationPivot(
                    detected=True,
                    pivot_type=RotationPivotType.FVG_ABSORPTION,
                    pivot_price=fvg_level,
                    recommended_entry=min(current_ask, fvg_level + spread * 0.5),
                    absorption_ratio=1.20,
                    confidence=0.88,
                    description=f"FVG Retest Absorption at {fvg_level:.5f}",
                )
            else:
                # Default passive maker entry inside book
                passive_entry = book.bid_price + (spread * 0.25)
                return LiquidityRotationPivot(
                    detected=True,
                    pivot_type=RotationPivotType.MID_QUOTE_PASSIVE,
                    pivot_price=book.mid_price,
                    recommended_entry=passive_entry,
                    absorption_ratio=1.0,
                    confidence=0.80,
                    description="Passive Maker Entry inside spread",
                )

        else:  # SELL
            # Check if recent high was swept (BSL sweep)
            if current_ask >= recent_high - spread * 1.5:
                return LiquidityRotationPivot(
                    detected=True,
                    pivot_type=RotationPivotType.BSL_SWEEP_REVERSAL,
                    pivot_price=recent_high,
                    recommended_entry=book.ask_price,  # Maker limit placed at ask
                    absorption_ratio=1.45,
                    confidence=0.92,
                    description=f"BSL Sweep at {recent_high:.5f} with ask resting fill",
                )
            elif fvg_level is not None and current_ask >= fvg_level - spread:
                return LiquidityRotationPivot(
                    detected=True,
                    pivot_type=RotationPivotType.FVG_ABSORPTION,
                    pivot_price=fvg_level,
                    recommended_entry=max(current_bid, fvg_level - spread * 0.5),
                    absorption_ratio=1.20,
                    confidence=0.88,
                    description=f"FVG Retest Absorption at {fvg_level:.5f}",
                )
            else:
                passive_entry = book.ask_price - (spread * 0.25)
                return LiquidityRotationPivot(
                    detected=True,
                    pivot_type=RotationPivotType.MID_QUOTE_PASSIVE,
                    pivot_price=book.mid_price,
                    recommended_entry=passive_entry,
                    absorption_ratio=1.0,
                    confidence=0.80,
                    description="Passive Maker Entry inside spread",
                )

    def evaluate_execution(
        self,
        symbol: str,
        direction_val: int,
        expected_alpha: float,
        book: OrderBookSnapshot,
        recent_high: Optional[float] = None,
        recent_low: Optional[float] = None,
        fvg_level: Optional[float] = None,
        lot_size: float = 1.0,
        volatility_factor: float = 1.0,
        force_mode: Optional[ExecutionMode] = None,
    ) -> ExecutionVerdict:
        """
        R7 Core Evaluation Pipeline:
        1. Checks Market order feasibility:
           Friction_market = Spread + Commission + Slippage_market
           Ratio_market = Alpha / Friction_market
        2. If Ratio_market < 3.5 or Sniper mode preferred:
           Applies Sniper / Maker-like entry at Liquidity Rotation Pivot.
           In Sniper mode:
           - Spread friction is reduced by entering at the bid/ask or mid-quote (saving 50-100% of spread).
           - Slippage is eliminated (slippage <= 0, maker price improvement).
           - Ratio_sniper is recalculated.
        3. Enforces the strict rule:
           If final Ratio < 3.5 -> REJECT trade outright.
        """
        profile = self.get_cost_profile(symbol)
        spread = book.spread
        comm = profile.typical_commission_points

        r_high = recent_high if recent_high is not None else (book.ask_price + spread * 5.0)
        r_low = recent_low if recent_low is not None else (book.bid_price - spread * 5.0)

        # 1. Market Order Cost Analysis
        mkt_slippage = self.estimate_market_slippage(
            symbol=symbol,
            book=book,
            lot_size=lot_size,
            volatility_factor=volatility_factor,
            direction_val=direction_val,
        )
        mkt_total_friction = spread + comm + mkt_slippage
        mkt_ratio = self.calculate_cost_ratio(expected_alpha, spread, comm, mkt_slippage)

        # 2. Check if Market Execution is requested and passes
        if force_mode == ExecutionMode.MARKET:
            approved = (mkt_ratio >= self.min_cost_ratio)
            target_price = book.ask_price if direction_val == 1 else book.bid_price
            return ExecutionVerdict(
                approved=approved,
                symbol=symbol,
                direction_val=direction_val,
                execution_mode=ExecutionMode.MARKET,
                target_entry_price=target_price,
                spread=spread,
                commission=comm,
                estimated_slippage=mkt_slippage,
                total_friction=mkt_total_friction,
                expected_alpha=expected_alpha,
                cost_ratio=mkt_ratio,
                min_cost_ratio_required=self.min_cost_ratio,
                fill_probability=1.0,
                rejection_reason=None if approved else f"REJECT_COST_DRAG_SHIELD: Market ratio {mkt_ratio:.2f} < {self.min_cost_ratio}",
                telemetry={
                    "mode": "FORCED_MARKET",
                    "friction_saved": 0.0,
                    "normalized_symbol": normalize_symbol(symbol),
                },
            )

        # 3. Sniper / Maker-like Entry at Liquidity Rotation Pivot
        pivot = self.detect_liquidity_rotation_pivot(
            symbol=symbol,
            direction_val=direction_val,
            book=book,
            recent_high=r_high,
            recent_low=r_low,
            fvg_level=fvg_level,
        )

        # In Sniper entry:
        # Effective spread cost is near zero or fraction (maker limit entry inside the book)
        # Slippage is eliminated (0.0 or negative/price improvement)
        if pivot.pivot_type in (RotationPivotType.BSL_SWEEP_REVERSAL, RotationPivotType.SSL_SWEEP_REVERSAL):
            # Perfect sweep limit entry: maker capture, spread cost = 0, slippage = 0
            effective_spread = 0.0
            sniper_slippage = 0.0
            fill_prob = 0.95
        elif pivot.pivot_type == RotationPivotType.FVG_ABSORPTION:
            effective_spread = spread * 0.25
            sniper_slippage = 0.0
            fill_prob = 0.90
        else:
            effective_spread = spread * 0.50
            sniper_slippage = 0.0
            fill_prob = 0.85

        sniper_total_friction = effective_spread + comm + sniper_slippage
        sniper_ratio = self.calculate_cost_ratio(expected_alpha, effective_spread, comm, sniper_slippage)

        # Check if Sniper meets or exceeds threshold
        if sniper_ratio >= self.min_cost_ratio:
            friction_saved = mkt_total_friction - sniper_total_friction
            return ExecutionVerdict(
                approved=True,
                symbol=symbol,
                direction_val=direction_val,
                execution_mode=ExecutionMode.LIMIT_SNIPER,
                target_entry_price=pivot.recommended_entry,
                spread=effective_spread,
                commission=comm,
                estimated_slippage=sniper_slippage,
                total_friction=sniper_total_friction,
                expected_alpha=expected_alpha,
                cost_ratio=sniper_ratio,
                min_cost_ratio_required=self.min_cost_ratio,
                fill_probability=fill_prob,
                rejection_reason=None,
                telemetry={
                    "pivot_type": pivot.pivot_type.value,
                    "pivot_confidence": pivot.confidence,
                    "market_cost_ratio": mkt_ratio,
                    "friction_saved": friction_saved,
                    "pivot_description": pivot.description,
                    "normalized_symbol": normalize_symbol(symbol),
                },
            )

        # If even Sniper entry fails to satisfy ratio >= 3.5, the edge is fundamentally insufficient
        return ExecutionVerdict(
            approved=False,
            symbol=symbol,
            direction_val=direction_val,
            execution_mode=ExecutionMode.LIMIT_SNIPER,
            target_entry_price=pivot.recommended_entry,
            spread=effective_spread,
            commission=comm,
            estimated_slippage=sniper_slippage,
            total_friction=sniper_total_friction,
            expected_alpha=expected_alpha,
            cost_ratio=sniper_ratio,
            min_cost_ratio_required=self.min_cost_ratio,
            fill_probability=fill_prob,
            rejection_reason=(
                f"REJECT_COST_DRAG_SHIELD: Sniper ratio {sniper_ratio:.2f} < {self.min_cost_ratio:.2f}. "
                f"Alpha {expected_alpha:.4f} insufficient against friction {sniper_total_friction:.4f}"
            ),
            telemetry={
                "market_cost_ratio": mkt_ratio,
                "sniper_cost_ratio": sniper_ratio,
                "pivot_type": pivot.pivot_type.value,
                "normalized_symbol": normalize_symbol(symbol),
            },
        )
