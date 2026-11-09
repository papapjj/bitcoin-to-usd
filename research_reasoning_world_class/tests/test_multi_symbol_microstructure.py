"""
Unit Tests for Module R6 (Multi-Symbol Matrix) and Module R7 (Microstructure Execution)
Project: Mythos of Claude - xLSTM 17.5M World-Class Reasoning Engine
"""

import math
import numpy as np
import pytest

from research_reasoning_world_class.src.multi_symbol_matrix import (
    ActivePosition,
    AssetClass,
    BarData,
    FilterDecision,
    MultiSymbolMatrix,
    SignalProposal,
    SUPPORTED_SYMBOLS,
    SYMBOL_ASSET_CLASS,
    TimeFrame,
    TradeDirection,
)
from research_reasoning_world_class.src.microstructure_execution import (
    CostProfile,
    ExecutionMode,
    ExecutionVerdict,
    LiquidityRotationPivot,
    MicrostructureExecutionShield,
    OrderBookSnapshot,
    RotationPivotType,
)


# =====================================================================
# R6: Multi-Symbol & Multi-Timeframe High-Frequency Matrix Tests
# =====================================================================


class TestMultiSymbolMatrixR6:
    """Test suite for Module R6."""

    def test_supported_symbols_and_asset_classes(self):
        """Verify basket of 6 assets and correct asset class taxonomy."""
        matrix = MultiSymbolMatrix()
        assert len(matrix.supported_symbols) == 6
        expected = ["BTCUSD", "ETHUSD", "SOLUSD", "XAUUSD", "EURUSD", "US30"]
        for sym in expected:
            assert sym in matrix.supported_symbols
            assert sym in SYMBOL_ASSET_CLASS

        assert SYMBOL_ASSET_CLASS["BTCUSD"] == AssetClass.CRYPTO
        assert SYMBOL_ASSET_CLASS["ETHUSD"] == AssetClass.CRYPTO
        assert SYMBOL_ASSET_CLASS["SOLUSD"] == AssetClass.CRYPTO
        assert SYMBOL_ASSET_CLASS["XAUUSD"] == AssetClass.METALS
        assert SYMBOL_ASSET_CLASS["EURUSD"] == AssetClass.FOREX
        assert SYMBOL_ASSET_CLASS["US30"] == AssetClass.INDICES

    def test_rolling_returns_and_correlation_precision(self):
        """Test Pearson correlation calculations with known mathematical inputs."""
        matrix = MultiSymbolMatrix(rolling_window_bars=50)

        # Generate synthetic series
        np.random.seed(42)
        base = np.cumsum(np.random.normal(0, 1, 100)) + 1000.0
        # Perfectly correlated
        series_a = base
        series_b = base * 2.0 + 50.0
        # Inversely correlated
        series_c = -base + 3000.0

        matrix.set_price_history("BTCUSD", TimeFrame.M1, list(series_a))
        matrix.set_price_history("ETHUSD", TimeFrame.M1, list(series_b))
        matrix.set_price_history("EURUSD", TimeFrame.M1, list(series_c))

        # Test A vs B (should be ~1.0)
        rho_ab = matrix.compute_pair_correlation("BTCUSD", "ETHUSD")
        assert rho_ab > 0.98

        # Test A vs C (should be ~ -1.0)
        rho_ac = matrix.compute_pair_correlation("BTCUSD", "EURUSD")
        assert rho_ac < -0.98

        # Test self-correlation is identically 1.0
        assert matrix.compute_pair_correlation("BTCUSD", "BTCUSD") == 1.0

    def test_numerical_stability_zero_variance_and_short_series(self):
        """Guarantees 0 NaN, 0 Inf on flatline price series or empty buffers."""
        matrix = MultiSymbolMatrix()

        # Flatline constant price (zero variance)
        flat = [100.0] * 50
        matrix.set_price_history("BTCUSD", TimeFrame.M1, flat)
        matrix.set_price_history("ETHUSD", TimeFrame.M1, flat)

        rho = matrix.compute_pair_correlation("BTCUSD", "ETHUSD")
        assert not math.isnan(rho)
        assert not math.isinf(rho)
        assert rho == 0.0

        # Short series (< 5 bars)
        matrix.set_price_history("SOLUSD", TimeFrame.M1, [10.0, 10.5, 10.2])
        rho_short = matrix.compute_pair_correlation("BTCUSD", "SOLUSD")
        assert rho_short == 0.0

    def test_correlation_matrix_6x6_structure(self):
        """Verifies full 6x6 correlation matrix properties: symmetric, diagonal == 1.0."""
        matrix = MultiSymbolMatrix()
        symbols, mat = matrix.compute_correlation_matrix()
        assert len(symbols) == 6
        assert mat.shape == (6, 6)

        # Diagonal check
        for i in range(6):
            assert mat[i, i] == 1.0

        # Symmetry check
        assert np.allclose(mat, mat.T)

    def test_cross_asset_correlation_filter_in_phase_positive_rho(self):
        """
        Cross-Asset Correlation Filter (rho > 0.85):
        When BTCUSD and ETHUSD have rho = 0.95:
        - Active BUY BTCUSD blocks candidate BUY ETHUSD (co-directional in-phase risk).
        - Active BUY BTCUSD allows candidate SELL ETHUSD (cross-market hedge).
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)

        # Seed synthetic series with rho > 0.85
        t = np.linspace(0, 10, 80)
        p1 = 50000 + 100 * np.sin(t) + np.arange(80) * 10
        p2 = 3000 + 10 * np.sin(t) + np.arange(80) * 1
        matrix.set_price_history("BTCUSD", TimeFrame.M1, list(p1))
        matrix.set_price_history("ETHUSD", TimeFrame.M1, list(p2))

        rho = matrix.compute_pair_correlation("BTCUSD", "ETHUSD")
        assert rho > 0.85

        # Register active position: BUY BTCUSD
        matrix.add_active_position(
            ActivePosition(
                symbol="BTCUSD",
                direction=TradeDirection.BUY,
                entry_price=50000.0,
                open_time=1000.0,
            )
        )

        # Candidate 1: BUY ETHUSD (In-phase risk -> REJECT)
        proposal_buy_eth = SignalProposal(
            symbol="ETHUSD",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=25.0,
            confidence=0.90,
            entry_price=3000.0,
            timestamp=1050.0,
        )
        decision1 = matrix.check_correlation_filter(proposal_buy_eth)
        assert decision1.approved is False
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in decision1.reason
        assert decision1.conflicting_symbol == "BTCUSD"
        assert decision1.effective_correlation > 0.85

        # Candidate 2: SELL ETHUSD (Hedge / Pairs trade -> APPROVED)
        proposal_sell_eth = SignalProposal(
            symbol="ETHUSD",
            direction=TradeDirection.SELL,
            timeframe=TimeFrame.M1,
            expected_alpha=25.0,
            confidence=0.90,
            entry_price=3000.0,
            timestamp=1050.0,
        )
        decision2 = matrix.check_correlation_filter(proposal_sell_eth)
        assert decision2.approved is True
        assert decision2.effective_correlation <= 0.85

    def test_cross_asset_correlation_filter_inverse_correlation(self):
        """
        When two assets have negative correlation rho = -0.90 (e.g. EURUSD vs DXY proxy):
        - BUY EURUSD and SELL DXY are in-phase risk -> REJECT!
        - BUY EURUSD and BUY DXY are hedged -> APPROVED!
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)

        t = np.linspace(0, 10, 80)
        p_eur = 1.0800 + 0.0050 * np.sin(t) + np.arange(80) * 0.0001
        p_us30 = 40000.0 - 500.0 * np.sin(t) - np.arange(80) * 10.0  # inverse
        matrix.set_price_history("EURUSD", TimeFrame.M1, list(p_eur))
        matrix.set_price_history("US30", TimeFrame.M1, list(p_us30))

        rho = matrix.compute_pair_correlation("EURUSD", "US30")
        assert rho < -0.85

        # Active BUY EURUSD
        matrix.add_active_position(
            ActivePosition(
                symbol="EURUSD",
                direction=TradeDirection.BUY,
                entry_price=1.0800,
                open_time=100.0,
            )
        )

        # Candidate: SELL US30
        # rho < -0.85, dir_candidate = -1, dir_active = +1
        # effective_rho = (-0.90) * (-1) * (+1) = +0.90 > 0.85 -> REJECT
        cand_sell_us30 = SignalProposal(
            symbol="US30",
            direction=TradeDirection.SELL,
            timeframe=TimeFrame.M1,
            expected_alpha=40.0,
            confidence=0.88,
            entry_price=39000.0,
            timestamp=120.0,
        )
        dec = matrix.check_correlation_filter(cand_sell_us30)
        assert dec.approved is False
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in dec.reason

        # Candidate: BUY US30 (Effective rho is negative -> APPROVED)
        cand_buy_us30 = SignalProposal(
            symbol="US30",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=40.0,
            confidence=0.88,
            entry_price=39000.0,
            timestamp=120.0,
        )
        dec_buy = matrix.check_correlation_filter(cand_buy_us30)
        assert dec_buy.approved is True

    def test_duplicate_symbol_same_direction_blocked(self):
        """Cannot stack identical direction on the same symbol."""
        matrix = MultiSymbolMatrix()
        matrix.add_active_position(
            ActivePosition(
                symbol="BTCUSD",
                direction=TradeDirection.BUY,
                entry_price=60000.0,
                open_time=1.0,
            )
        )
        cand = SignalProposal(
            symbol="BTCUSD",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=100.0,
            confidence=0.9,
            entry_price=60100.0,
            timestamp=2.0,
        )
        dec = matrix.check_correlation_filter(cand)
        assert dec.approved is False
        assert "REJECT_DUPLICATE_POSITION" in dec.reason

    def test_multi_symbol_batch_router_arbitration(self):
        """
        Simultaneous candidate signals across multiple assets:
        Router selects the highest scoring signal in a correlated cluster
        and drops the redundant ones.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)

        t = np.linspace(0, 10, 80)
        p_btc = 60000 + 500 * np.sin(t)
        p_eth = 3000 + 25 * np.sin(t)
        p_sol = 150 + 1.25 * np.sin(t)
        p_gold = 2000 + 10 * np.cos(t)  # uncorrelated with crypto

        matrix.set_price_history("BTCUSD", TimeFrame.M1, list(p_btc))
        matrix.set_price_history("ETHUSD", TimeFrame.M1, list(p_eth))
        matrix.set_price_history("SOLUSD", TimeFrame.M1, list(p_sol))
        matrix.set_price_history("XAUUSD", TimeFrame.M1, list(p_gold))

        # 3 correlated Crypto BUY signals + 1 uncorrelated Gold BUY signal
        cands = [
            SignalProposal("BTCUSD", TradeDirection.BUY, TimeFrame.M1, expected_alpha=80.0, confidence=0.85, entry_price=60000.0, timestamp=10.0),
            SignalProposal("ETHUSD", TradeDirection.BUY, TimeFrame.M1, expected_alpha=95.0, confidence=0.92, entry_price=3000.0, timestamp=10.0), # Higher score!
            SignalProposal("SOLUSD", TradeDirection.BUY, TimeFrame.M1, expected_alpha=50.0, confidence=0.80, entry_price=150.0, timestamp=10.0),
            SignalProposal("XAUUSD", TradeDirection.BUY, TimeFrame.M1, expected_alpha=60.0, confidence=0.88, entry_price=2000.0, timestamp=10.0),
        ]

        result = matrix.route_and_filter_candidates(cands)

        approved_symbols = [sig.symbol for sig in result.approved_signals]
        # ETHUSD had the highest score among cryptos, so it is accepted
        assert "ETHUSD" in approved_symbols
        # XAUUSD is uncorrelated, so it is also accepted
        assert "XAUUSD" in approved_symbols
        # BTCUSD and SOLUSD should be filtered out due to correlation with approved ETHUSD
        rejected_symbols = [sig.symbol for sig, dec in result.rejected_signals]
        assert "BTCUSD" in rejected_symbols
        assert "SOLUSD" in rejected_symbols

    def test_m5_confluence_evaluation(self):
        """Confluence multiplier boosts trend-aligned trades and penalizes counter-trend."""
        matrix = MultiSymbolMatrix()
        # Create M5 upward trend
        m5_closes = [100.0, 101.0, 102.0, 103.0, 105.0]
        matrix.set_price_history("BTCUSD", TimeFrame.M5, m5_closes)

        # Propose BUY (aligned with trend)
        prop_buy = SignalProposal("BTCUSD", TradeDirection.BUY, TimeFrame.M1, 50.0, 0.9, 105.0, 1.0)
        assert matrix.evaluate_m5_confluence(prop_buy) == 1.20

        # Propose SELL (counter-trend)
        prop_sell = SignalProposal("BTCUSD", TradeDirection.SELL, TimeFrame.M1, 50.0, 0.9, 105.0, 1.0)
        assert matrix.evaluate_m5_confluence(prop_sell) == 0.75

    def test_24h_execution_rate_and_pacing_monitor(self):
        """Monitors 45-60 daily target and projected 90d trade count (> 4,000 trades)."""
        matrix = MultiSymbolMatrix(daily_target_trades_min=45, daily_target_trades_max=60)
        now = 100000.0

        # Record 50 executions in the last 24h
        for i in range(50):
            matrix.record_execution(now - (i * 1200))  # every 20 minutes

        status = matrix.get_frequency_pacing_status(now)
        assert status["24h_trades_executed"] == 50
        assert status["status"] == "OPTIMAL_PACING"
        assert status["projected_90d_trades"] == 50 * 90  # 4,500 trades >= 4,000 target


# =====================================================================
# R7: Order Microstructure Execution & Slippage Shield Tests
# =====================================================================


class TestMicrostructureExecutionR7:
    """Test suite for Module R7."""

    def test_cost_ratio_formula_precision(self):
        """
        Verify: Ratio = Expected Alpha / (Spread + Commission + Slippage) >= 3.5.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        # Alpha = 10.0, Spread = 1.0, Comm = 0.5, Slippage = 0.5 -> Friction = 2.0 -> Ratio = 5.0 >= 3.5 -> Pass
        r1 = shield.calculate_cost_ratio(expected_alpha=10.0, spread=1.0, commission=0.5, slippage=0.5)
        assert pytest.approx(r1, rel=1e-5) == 5.0
        assert r1 >= 3.5

        # Alpha = 5.0, Friction = 2.0 -> Ratio = 2.5 < 3.5 -> Fail
        r2 = shield.calculate_cost_ratio(expected_alpha=5.0, spread=1.0, commission=0.5, slippage=0.5)
        assert pytest.approx(r2, rel=1e-5) == 2.5
        assert r2 < 3.5

        # Zero or negative alpha safe handling
        assert shield.calculate_cost_ratio(expected_alpha=0.0, spread=1.0, commission=0.5, slippage=0.5) == 0.0
        assert shield.calculate_cost_ratio(expected_alpha=-5.0, spread=1.0, commission=0.5, slippage=0.5) == 0.0

    def test_order_book_microstructure_and_imbalance(self):
        """Verifies spread, mid-price, micro-price, and Order Book Imbalance (OBI)."""
        book = OrderBookSnapshot(
            symbol="BTCUSD",
            bid_price=60000.0,
            ask_price=60004.0,
            bid_volume=10.0,
            ask_volume=2.0,
            timestamp=100.0,
        )
        assert book.spread == 4.0
        assert book.mid_price == 60002.0

        # OBI = (10 - 2) / 12 = 8 / 12 = 0.6667
        assert pytest.approx(book.order_book_imbalance, rel=1e-3) == 0.6667

        # Micro-price = (10 * 60004 + 2 * 60000) / 12 = (600040 + 120000) / 12 = 720040 / 12 = 60003.3333
        assert pytest.approx(book.micro_price, rel=1e-3) == 60003.3333

    def test_liquidity_rotation_pivot_detection(self):
        """
        Detects Institutional Liquidity Sweeps:
        - SSL sweep when price sweeps below recent low for BUY.
        - BSL sweep when price sweeps above recent high for SELL.
        """
        shield = MicrostructureExecutionShield()

        # SSL Sweep scenario (BUY)
        book_buy = OrderBookSnapshot("ETHUSD", bid_price=2999.0, ask_price=2999.5, bid_volume=5.0, ask_volume=2.0, timestamp=1.0)
        pivot_buy = shield.detect_liquidity_rotation_pivot(
            symbol="ETHUSD",
            direction_val=1,  # BUY
            book=book_buy,
            recent_high=3050.0,
            recent_low=3000.0,  # Swept below 3000
        )
        assert pivot_buy.detected is True
        assert pivot_buy.pivot_type == RotationPivotType.SSL_SWEEP_REVERSAL
        assert pivot_buy.recommended_entry == 2999.0  # Resting maker limit at bid

        # BSL Sweep scenario (SELL)
        book_sell = OrderBookSnapshot("ETHUSD", bid_price=3050.5, ask_price=3051.0, bid_volume=1.0, ask_volume=6.0, timestamp=2.0)
        pivot_sell = shield.detect_liquidity_rotation_pivot(
            symbol="ETHUSD",
            direction_val=-1,  # SELL
            book=book_sell,
            recent_high=3050.0,  # Swept above 3050
            recent_low=2980.0,
        )
        assert pivot_sell.detected is True
        assert pivot_sell.pivot_type == RotationPivotType.BSL_SWEEP_REVERSAL
        assert pivot_sell.recommended_entry == 3051.0  # Resting maker limit at ask

    def test_sniper_maker_mode_salvages_viable_trades_blocked_by_market_slippage(self):
        """
        Key Institutional Value Proposition:
        A trade with moderate alpha fails Market execution due to full spread + market slippage (ratio < 3.5),
        but PASSES in Sniper / Maker-like mode by eliminating slippage and capturing spread at liquidity pivot (ratio >= 3.5).
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        # BTCUSD order book snapshot
        book = OrderBookSnapshot(
            symbol="BTCUSD",
            bid_price=65000.0,
            ask_price=65004.0,  # Spread = 4.0
            bid_volume=5.0,
            ask_volume=5.0,
            timestamp=500.0,
        )
        # Commission is 1.5, Market Slippage is ~2.0
        # Total Market Friction = 4.0 + 1.5 + 2.0 = 7.5
        # If Expected Alpha = 18.0:
        # Market Ratio = 18.0 / 7.5 = 2.40 < 3.5 -> Would be REJECTED!

        # 1. Test Forced Market mode -> MUST REJECT
        verdict_mkt = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=18.0,
            book=book,
            recent_high=65200.0,
            recent_low=65002.0,  # Swept
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict_mkt.approved is False
        assert verdict_mkt.execution_mode == ExecutionMode.MARKET
        assert verdict_mkt.cost_ratio < 3.5
        assert "REJECT_COST_DRAG_SHIELD" in verdict_mkt.rejection_reason

        # 2. Test Default Sniper Execution mode -> MUST PASS
        # Sniper at SSL sweep: effective spread = 0, slippage = 0
        # Total Sniper Friction = 0.0 + 1.5 + 0.0 = 1.5
        # Sniper Ratio = 18.0 / 1.5 = 12.0 >= 3.5 -> APPROVED!
        verdict_sniper = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=18.0,
            book=book,
            recent_high=65200.0,
            recent_low=65002.0,
        )
        assert verdict_sniper.approved is True
        assert verdict_sniper.execution_mode == ExecutionMode.LIMIT_SNIPER
        assert verdict_sniper.cost_ratio >= 3.5
        assert verdict_sniper.estimated_slippage == 0.0
        assert verdict_sniper.telemetry["friction_saved"] > 0.0

    def test_cost_shield_strictly_rejects_sub_par_edge(self):
        """
        When Expected Alpha is tiny (e.g. 1.0 pt on BTCUSD), even Sniper entry
        cannot overcome broker commission (1.5 pt). Ratio = 1.0 / 1.5 = 0.67 < 3.5.
        Must strictly REJECT.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        book = OrderBookSnapshot("BTCUSD", 65000.0, 65004.0, 5.0, 5.0, 500.0)

        verdict = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=1.0,  # Tiny edge
            book=book,
        )
        assert verdict.approved is False
        assert verdict.cost_ratio < 3.5
        assert "REJECT_COST_DRAG_SHIELD" in verdict.rejection_reason

    def test_all_six_symbols_cost_profiles(self):
        """
        Ensures all 6 assets in the Mythos basket (BTCUSD, ETHUSD, SOLUSD, XAUUSD, EURUSD, US30)
        have valid cost configurations and can be evaluated without any error.
        """
        shield = MicrostructureExecutionShield()
        test_prices = {
            "BTCUSD": (65000.0, 65004.0, 30.0),
            "ETHUSD": (3000.0, 3000.3, 2.0),
            "SOLUSD": (150.0, 150.03, 0.25),
            "XAUUSD": (2400.0, 2400.18, 1.5),
            "EURUSD": (1.08500, 1.08508, 0.00060),
            "US30": (40000.0, 40001.8, 15.0),
        }

        for sym in SUPPORTED_SYMBOLS:
            bid, ask, alpha = test_prices[sym]
            book = OrderBookSnapshot(
                symbol=sym,
                bid_price=bid,
                ask_price=ask,
                bid_volume=10.0,
                ask_volume=10.0,
                timestamp=100.0,
            )
            profile = shield.get_cost_profile(sym)
            assert profile.symbol == sym
            assert profile.point_size > 0.0

            verdict = shield.evaluate_execution(
                symbol=sym,
                direction_val=1,
                expected_alpha=alpha,
                book=book,
            )
            assert isinstance(verdict, ExecutionVerdict)
            assert not math.isnan(verdict.cost_ratio)
            assert not math.isinf(verdict.cost_ratio)


    def test_adversarial_extreme_volatility_and_obi_skew(self):
        """
        Adversarial: Extreme volatility (e.g., 10x normal during CPI/NFP)
        and strong opposing order book pressure (OBI = -0.99 against a BUY).
        Market slippage explodes, but Sniper Limit entry safely contains friction or rejects.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        book = OrderBookSnapshot(
            symbol="BTCUSD",
            bid_price=60000.0,
            ask_price=60005.0,
            bid_volume=0.1,    # Tiny bid depth
            ask_volume=50.0,   # Massive ask wall (OBI ~= -0.996)
            timestamp=2000.0,
        )

        # Market slippage under 10x volatility
        slip_mkt = shield.estimate_market_slippage(
            symbol="BTCUSD",
            book=book,
            lot_size=5.0,
            volatility_factor=10.0,
            direction_val=1,
        )
        assert slip_mkt > 20.0  # Exploded slippage

        # Market order evaluation with alpha = 50.0
        # Friction = spread (5.0) + comm (1.5) + slippage (>20) > 26.5
        # Ratio = 50 / >26.5 < 1.9 < 3.5 -> MUST REJECT in market mode
        verdict_mkt = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=50.0,
            book=book,
            lot_size=5.0,
            volatility_factor=10.0,
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict_mkt.approved is False
        assert "REJECT_COST_DRAG_SHIELD" in verdict_mkt.rejection_reason

    def test_adversarial_empty_and_unknown_candidates(self):
        """
        Tests router handling of empty list, unknown symbols, and zero bars.
        """
        matrix = MultiSymbolMatrix()
        # Empty candidate list
        res = matrix.route_and_filter_candidates([])
        assert len(res.approved_signals) == 0
        assert len(res.rejected_signals) == 0

        # Bar for unknown symbol ignored gracefully
        unknown_bar = BarData("UNKNOWN", TimeFrame.M1, 100.0, 1.0, 1.0, 1.0, 1.0)
        matrix.register_bar(unknown_bar)
        assert "UNKNOWN" not in matrix._bars

        # Unknown symbol proposal
        unk_prop = SignalProposal("UNKNOWN", TradeDirection.BUY, TimeFrame.M1, 50.0, 0.9, 10.0, 100.0)
        dec = matrix.check_correlation_filter(unk_prop)
        assert dec.approved is True  # No conflict with empty matrix
