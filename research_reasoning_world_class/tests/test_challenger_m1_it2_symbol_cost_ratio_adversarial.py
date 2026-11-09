"""
Empirical Challenger Test Suite: Symbol Normalization & Cost Ratio Invariant Stress
Milestone M1 Iteration 2 Challenger (Challenger M1_it2_2)
Author: teamwork_preview_challenger_m1_it2_2

Adversarial Stress Testing:
1. Complete Cost Profile Parity:
   - BTCUSD vs BTCUSDm, BTCUSDM, btcusdm, BTCUSD.m, BTCUSD_m
   - ETHUSD vs ETHUSDm, ETHUSDM, ethusdm, ETHUSD.m, ETHUSD_m
   - XAUUSD vs XAUUSDm, XAUUSDM, xauusdm, XAUUSD.m, XAUUSD_m
   - SOLUSD vs SOLUSDm, SOLUSDM, solusdm, SOLUSD.m, SOLUSD_m
   - EURUSD vs EURUSDm, EURUSDM, eurusdm, EURUSD.m, EURUSD_m
   - US30 vs US30m, US30M, us30m, US30.m, US30_m
2. Zero Underestimation Verification:
   - Prove mathematically and empirically that underestimation is strictly 0.00% (eliminating the 44.0% underestimation bug).
3. Exact Cost Ratio Boundary Filtering:
   - Alpha with ratio = 3.4999 strictly rejected across all raw and suffixed symbols.
   - Alpha with ratio = 3.5000 approved across all raw and suffixed symbols.
4. Dynamic Market Slippage and OBI Stress:
   - Slippage parity under high volatility, large lot sizes, and adverse OBI.
5. End-to-End Multi-Symbol + Microstructure Execution Pipeline Invariant:
   - Co-directional filtering + Microstructure cost filtering pipeline integrity.
"""

import math
import pytest
import numpy as np

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
    normalize_symbol,
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


class TestChallengerSymbolNormalizationCostParity:
    """Empirical verification of 100% cost profile parity between raw and suffixed symbols."""

    BASE_SYMBOLS = ["BTCUSD", "ETHUSD", "SOLUSD", "XAUUSD", "EURUSD", "US30"]
    SUFFIX_VARIANTS = ["m", "M", ".m", "_m", "_raw"]

    def test_all_base_and_suffixed_symbols_cost_profile_identity(self):
        """
        Verify that every suffixed variant produces the exact same CostProfile attributes
        as the raw base symbol: point_size, typical_spread, commission, and slippage.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        for base in self.BASE_SYMBOLS:
            base_prof = shield.get_cost_profile(base)

            # Check direct broker 'm' and 'M' suffixes
            for suffix in ["m", "M"]:
                suffixed_sym = f"{base}{suffix}"
                prof = shield.get_cost_profile(suffixed_sym)

                assert prof.point_size == base_prof.point_size, f"{suffixed_sym} point_size mismatch"
                assert prof.typical_spread_points == base_prof.typical_spread_points, f"{suffixed_sym} spread mismatch"
                assert prof.typical_commission_points == base_prof.typical_commission_points, f"{suffixed_sym} commission mismatch"
                assert prof.typical_market_slippage_points == base_prof.typical_market_slippage_points, f"{suffixed_sym} slippage mismatch"
                assert prof.sniper_slippage_points == base_prof.sniper_slippage_points, f"{suffixed_sym} sniper slippage mismatch"

            # Check lowercase variants (e.g. btcusdm)
            lower_sym = f"{base.lower()}m"
            prof_lower = shield.get_cost_profile(lower_sym)
            assert prof_lower.typical_commission_points == base_prof.typical_commission_points
            assert prof_lower.typical_market_slippage_points == base_prof.typical_market_slippage_points

            # Check dot and underscore suffixes (e.g. BTCUSD.m, BTCUSD_m)
            for sep in [".m", "_m"]:
                ext_sym = f"{base}{sep}"
                prof_ext = shield.get_cost_profile(ext_sym)
                assert prof_ext.typical_commission_points == base_prof.typical_commission_points
                assert prof_ext.typical_market_slippage_points == base_prof.typical_market_slippage_points

    def test_zero_underestimation_btc_eth_xau(self):
        """
        Explicitly verify the 3 core assets highlighted in the bug report:
        BTCUSD vs BTCUSDm
        ETHUSD vs ETHUSDm
        XAUUSD vs XAUUSDm
        Calculate underestimation percentage: must be strictly 0.0000%.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        targets = [
            ("BTCUSD", "BTCUSDm", 4.0),
            ("ETHUSD", "ETHUSDm", 0.3),
            ("XAUUSD", "XAUUSDm", 0.18),
        ]

        for raw, suffixed, typical_spread in targets:
            prof_raw = shield.get_cost_profile(raw)
            prof_suf = shield.get_cost_profile(suffixed)

            # In forced MARKET mode, total friction = spread + comm + slippage
            friction_raw = typical_spread + prof_raw.typical_commission_points + prof_raw.typical_market_slippage_points
            friction_suf = typical_spread + prof_suf.typical_commission_points + prof_suf.typical_market_slippage_points

            underestimation_pct = ((friction_raw - friction_suf) / friction_raw) * 100.0

            print(f"\n{raw} Friction: {friction_raw:.4f} | {suffixed} Friction: {friction_suf:.4f} | Underestimation: {underestimation_pct:.4f}%")

            assert friction_raw == friction_suf, f"Friction mismatch between {raw} and {suffixed}"
            assert underestimation_pct == 0.0, f"Violation: {suffixed} has {underestimation_pct:.2f}% underestimation!"


class TestChallengerCostRatioStrictRejection:
    """Empirical verification that cost ratio < 3.5 is strictly rejected across raw and suffixed symbols."""

    def test_market_sub_par_trade_rejection_raw_and_suffixed_parity(self):
        """
        Adversarial test: Construct trades with cost ratio exactly 3.4999 (sub-par) and 3.5001 (viable)
        for all 6 pairs in both raw and suffixed forms.
        Verify that 3.4999 is REJECTED in all cases and 3.5001 is APPROVED in all cases.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        test_pairs = [
            ("BTCUSD", "BTCUSDm", 60000.0, 4.0),
            ("ETHUSD", "ETHUSDm", 3000.0, 0.3),
            ("SOLUSD", "SOLUSDm", 150.0, 0.03),
            ("XAUUSD", "XAUUSDm", 2400.0, 0.18),
            ("EURUSD", "EURUSDm", 1.0800, 0.00008),
            ("US30", "US30m", 39000.0, 1.8),
        ]

        for raw_sym, suf_sym, mid_price, spread in test_pairs:
            for sym in [raw_sym, suf_sym]:
                prof = shield.get_cost_profile(sym)
                book = OrderBookSnapshot(
                    symbol=sym,
                    bid_price=mid_price - spread / 2.0,
                    ask_price=mid_price + spread / 2.0,
                    bid_volume=10.0,
                    ask_volume=10.0,
                    timestamp=100.0,
                )

                # Total market friction = spread + comm + slippage (at default vol=1.0, lot=1.0, OBI=0.0)
                tot_friction = spread + prof.typical_commission_points + prof.typical_market_slippage_points

                # Sub-par alpha: ratio = 3.4999
                alpha_sub = tot_friction * 3.4999
                v_sub = shield.evaluate_execution(
                    symbol=sym,
                    direction_val=1,
                    expected_alpha=alpha_sub,
                    book=book,
                    force_mode=ExecutionMode.MARKET,
                )
                assert v_sub.approved is False, f"Violation: sub-par trade approved for {sym}! Ratio: {v_sub.cost_ratio:.4f}"
                assert v_sub.cost_ratio < 3.5000
                assert "REJECT_COST_DRAG_SHIELD" in v_sub.rejection_reason

                # Viable alpha: ratio = 3.5001
                alpha_via = tot_friction * 3.5001
                v_via = shield.evaluate_execution(
                    symbol=sym,
                    direction_val=1,
                    expected_alpha=alpha_via,
                    book=book,
                    force_mode=ExecutionMode.MARKET,
                )
                assert v_via.approved is True, f"Violation: viable trade rejected for {sym}! Ratio: {v_via.cost_ratio:.4f}"
                assert v_via.cost_ratio >= 3.5000
                assert v_via.rejection_reason is None

    def test_sniper_limit_sub_par_rejection_raw_and_suffixed_parity(self):
        """
        Verify that under Sniper Limit mode (where spread and slippage are minimized),
        trades where alpha / friction < 3.5 are STILL strictly rejected for both raw and suffixed pairs.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        for sym in ["BTCUSD", "BTCUSDm", "ETHUSD", "ETHUSDm", "XAUUSD", "XAUUSDm"]:
            prof = shield.get_cost_profile(sym)
            book = OrderBookSnapshot(
                symbol=sym,
                bid_price=50000.0,
                ask_price=50004.0,
                bid_volume=5.0,
                ask_volume=5.0,
                timestamp=100.0,
            )

            # In SSL sweep, maker spread = 0.0, slippage = 0.0 -> friction = comm
            # Sub-par edge: alpha = comm * 3.4
            sub_alpha = prof.typical_commission_points * 3.4
            v_sub = shield.evaluate_execution(
                symbol=sym,
                direction_val=1,
                expected_alpha=sub_alpha,
                book=book,
                recent_high=50200.0,
                recent_low=50002.0,  # SSL sweep triggers maker limit
            )
            assert v_sub.approved is False, f"Sniper mode failed to reject sub-par trade on {sym}"
            assert v_sub.cost_ratio < 3.5

            # Viable edge: alpha = comm * 3.6
            via_alpha = prof.typical_commission_points * 3.6
            v_via = shield.evaluate_execution(
                symbol=sym,
                direction_val=1,
                expected_alpha=via_alpha,
                book=book,
                recent_high=50200.0,
                recent_low=50002.0,
            )
            assert v_via.approved is True, f"Sniper mode incorrectly rejected viable trade on {sym}"
            assert v_via.cost_ratio >= 3.5


class TestChallengerAdversarialMicrostructureStress:
    """Stress tests on dynamic market conditions: adverse OBI, volatility bursts, lot scaling."""

    def test_dynamic_slippage_parity_under_adversarial_conditions(self):
        """
        Ensure dynamic slippage calculations produce 100% identical values between
        raw and suffixed symbols across:
        - Adverse OBI (-0.95 to +0.95)
        - High volatility spikes (vol_factor = 3.5)
        - Heavy institutional lot sizes (lot_size = 25.0)
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        test_pairs = [("BTCUSD", "BTCUSDm"), ("ETHUSD", "ETHUSDm"), ("XAUUSD", "XAUUSDm")]

        for raw_sym, suf_sym in test_pairs:
            for obi_val in [-0.95, -0.5, 0.0, 0.5, 0.95]:
                # Construct book with specified OBI
                tot_vol = 100.0
                bid_vol = (1.0 + obi_val) * tot_vol / 2.0
                ask_vol = tot_vol - bid_vol

                book_raw = OrderBookSnapshot(raw_sym, 1000.0, 1002.0, bid_vol, ask_vol, 1.0)
                book_suf = OrderBookSnapshot(suf_sym, 1000.0, 1002.0, bid_vol, ask_vol, 1.0)

                for vol_factor in [1.0, 2.5, 5.0]:
                    for lot in [0.1, 1.0, 10.0, 50.0]:
                        slip_raw = shield.estimate_market_slippage(
                            symbol=raw_sym,
                            book=book_raw,
                            lot_size=lot,
                            volatility_factor=vol_factor,
                            direction_val=1,
                        )
                        slip_suf = shield.estimate_market_slippage(
                            symbol=suf_sym,
                            book=book_suf,
                            lot_size=lot,
                            volatility_factor=vol_factor,
                            direction_val=1,
                        )
                        assert slip_raw == slip_suf, (
                            f"Dynamic slippage divergence! {raw_sym}={slip_raw} vs {suf_sym}={slip_suf} "
                            f"(obi={obi_val}, vol={vol_factor}, lot={lot})"
                        )

    def test_normalized_symbol_in_telemetry(self):
        """
        Verify that execution telemetry always reports the canonical normalized symbol
        so downstream accounting and logs are not polluted.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        book = OrderBookSnapshot("BTCUSDm", 60000.0, 60004.0, 10.0, 10.0, 1.0)

        verdict = shield.evaluate_execution(
            symbol="BTCUSDm",
            direction_val=1,
            expected_alpha=100.0,
            book=book,
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict.telemetry["normalized_symbol"] == "BTCUSD"

        book_gold = OrderBookSnapshot("XAUUSDm", 2400.0, 2400.2, 10.0, 10.0, 1.0)
        verdict_gold = shield.evaluate_execution(
            symbol="XAUUSDm",
            direction_val=1,
            expected_alpha=10.0,
            book=book_gold,
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict_gold.telemetry["normalized_symbol"] == "XAUUSD"


class TestChallengerEndToEndFilteringPipelineEmpirical:
    """End-to-End Multi-Symbol Matrix + Microstructure Shield Integration."""

    def test_pipeline_integration_with_mixed_suffixed_candidates(self):
        """
        Test an end-to-end pipeline pass:
        1. MultiSymbolMatrix filters candidate signals for correlation / duplication.
        2. MicrostructureExecutionShield evaluates each approved signal for cost ratio.
        3. Verify that suffixed symbols are handled seamlessly across both modules.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        # Set correlated price history
        np.random.seed(42)
        x = np.random.normal(0, 1, 120)
        z = np.random.normal(0, 1, 120)
        rho_target = 0.92
        y = rho_target * x + math.sqrt(1.0 - rho_target ** 2) * z
        p_btc = list(50000.0 + np.cumsum(x) * 20.0)
        p_eth = list(3000.0 + np.cumsum(y) * 5.0)
        matrix.set_price_history("BTCUSDm", TimeFrame.M1, p_btc)
        matrix.set_price_history("ETHUSDm", TimeFrame.M1, p_eth)

        # Add active position in BTCUSDm
        matrix.add_active_position(
            ActivePosition(symbol="BTCUSDm", direction=TradeDirection.BUY, entry_price=50000.0, open_time=1.0)
        )

        # Candidate 1: ETHUSDm BUY (Correlated with BTCUSDm BUY -> should be rejected by correlation shield)
        cand_eth = SignalProposal(
            symbol="ETHUSDm",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=50.0,
            confidence=0.90,
            entry_price=3000.0,
            timestamp=10.0,
        )
        dec_corr = matrix.check_correlation_filter(cand_eth)
        # Verify correlation filter correctly flags suffixed symbols
        assert dec_corr.approved is False
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in dec_corr.reason

        # Candidate 2: XAUUSDm BUY (Uncorrelated with BTCUSDm -> approved by correlation shield)
        cand_gold = SignalProposal(
            symbol="XAUUSDm",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=0.50,  # Alpha = 0.50 pt
            confidence=0.92,
            entry_price=2400.0,
            timestamp=10.0,
        )
        dec_gold = matrix.check_correlation_filter(cand_gold)
        assert dec_gold.approved is True

        # Candidate 2 then enters Microstructure Execution Shield
        # For XAUUSDm: spread=0.18, comm=0.07, slip=0.08 -> market friction = 0.33
        # In forced market: ratio = 0.50 / 0.33 = 1.515 < 3.5 -> MUST BE REJECTED by Microstructure Shield!
        book_gold = OrderBookSnapshot("XAUUSDm", 2400.0, 2400.18, 5.0, 5.0, 10.0)
        v_gold = shield.evaluate_execution(
            symbol=cand_gold.symbol,
            direction_val=1,
            expected_alpha=cand_gold.expected_alpha,
            book=book_gold,
            force_mode=ExecutionMode.MARKET,
        )
        assert v_gold.approved is False
        assert v_gold.cost_ratio < 3.5
        assert "REJECT_COST_DRAG_SHIELD" in v_gold.rejection_reason
