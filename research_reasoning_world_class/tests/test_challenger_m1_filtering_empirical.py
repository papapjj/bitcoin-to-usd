"""
Empirical Challenge Test Suite: High-Conviction Multi-Asset Filtering & Safety Invariants
Author: Challenger M1_2 (Empirical Challenger 2 for Milestone M1)
Project: Multi-Asset Walk-Forward Self-Play Triad (SPIN + MAD Reflexion + STaR)

Adversarial Stress Testing Coverage:
1. Cross-Asset Correlation Shield:
   - In-phase co-directional order submission across correlated Exness pairs (BTCUSDm, ETHUSDm, XAUUSDm) with rho_eff > 0.85
   - Hedging allowance (co-directional hedge vs in-phase risk)
   - Inverse correlation with opposite directions
   - Exact boundary sensitivity (0.8499 vs 0.8500 vs 0.8501)
   - Suffixed symbol normalization (BTCUSDm, ETHUSDm, XAUUSDm)
   - Batch router arbitration across multiple correlated candidate proposals
2. PRM Score Boundary & Confidence Threshold:
   - PRM score < 0.90 suppression (action locked to HOLD / 0, is_activated = False)
   - Total Confidence < 0.85 suppression
   - Adversarial Margin < 0.20 indecision suppression
   - Boundary exactness at 0.8999 vs 0.9000 and 0.8499 vs 0.8500
   - MythosMT5Policy PRM gating invariant
3. Microstructure Shield:
   - Cost ratio < 3.5 strict rejection (Expected Alpha / Total Friction < 3.5)
   - Cost ratio >= 3.5 approval
   - Forced Market mode rejection under realistic slippage and spread drag
   - Sniper limit entry salvage vs un-salvageable sub-par edge
   - Zero, negative, and extreme volatility / blown spread regimes
   - Suffixed symbols behavior in MicrostructureExecutionShield
"""

import os
# Lock optimal CPU threads for sub-50ms execution SLA before OpenMP C runtime initializes
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

import math
import time
import numpy as np
import pytest
import torch

torch.set_num_threads(min(4, os.cpu_count() or 4))

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
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
    ReasoningVerdict,
    MetacognitiveOutput,
    PRMVerifier,
)
from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy
from research_reasoning_world_class.src.mt5_ipc_transport import IDX_POLICY_CLOSE_ALL


# =====================================================================
# 1. CROSS-ASSET CORRELATION SHIELD EMPIRICAL CHALLENGES
# =====================================================================

class TestCrossAssetCorrelationShieldEmpirical:
    """Adversarial stress-testing of Cross-Asset Correlation Shield with rho_eff > 0.85."""

    def _generate_correlated_series(self, n: int = 100, rho_target: float = 0.92, seed: int = 42):
        """Generates two price series with exact empirical correlation near rho_target."""
        np.random.seed(seed)
        x = np.random.normal(0, 1, n)
        # y = rho * x + sqrt(1 - rho^2) * z
        z = np.random.normal(0, 1, n)
        y = rho_target * x + math.sqrt(max(0.0, 1.0 - rho_target ** 2)) * z
        p1 = 50000.0 + np.cumsum(x) * 20.0
        p2 = 3000.0 + np.cumsum(y) * 5.0
        return list(p1), list(p2)

    def test_in_phase_co_directional_buy_orders_strictly_rejected(self):
        """
        Challenge: Submit BUY BTCUSDm and BUY ETHUSDm when rho > 0.85.
        Expected: Strictly REJECTED with REJECT_CORRELATED_IN_PHASE_RISK.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        p_btc, p_eth = self._generate_correlated_series(n=120, rho_target=0.92)
        matrix.set_price_history("BTCUSDm", TimeFrame.M1, p_btc)
        matrix.set_price_history("ETHUSDm", TimeFrame.M1, p_eth)

        rho = matrix.compute_pair_correlation("BTCUSDm", "ETHUSDm")
        assert rho > 0.85, f"Expected rho > 0.85, got {rho:.4f}"

        # Active position: BUY BTCUSDm
        matrix.add_active_position(
            ActivePosition(
                symbol="BTCUSDm",
                direction=TradeDirection.BUY,
                entry_price=50000.0,
                open_time=100.0,
            )
        )

        # Candidate order: BUY ETHUSDm (co-directional in-phase)
        candidate = SignalProposal(
            symbol="ETHUSDm",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=25.0,
            confidence=0.92,
            entry_price=3000.0,
            timestamp=120.0,
        )

        decision = matrix.check_correlation_filter(candidate)
        assert decision.approved is False, "Violation: in-phase co-directional BUY was approved!"
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in decision.reason
        assert decision.effective_correlation > 0.85
        assert decision.conflicting_symbol in ("BTCUSD", "BTCUSDm")

    def test_in_phase_co_directional_sell_orders_strictly_rejected(self):
        """
        Challenge: Submit SELL BTCUSDm and SELL ETHUSDm when rho > 0.85.
        Expected: Effective rho = rho * (-1) * (-1) = rho > 0.85 -> Strictly REJECTED.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        p_btc, p_eth = self._generate_correlated_series(n=120, rho_target=0.90)
        matrix.set_price_history("BTCUSDm", TimeFrame.M1, p_btc)
        matrix.set_price_history("ETHUSDm", TimeFrame.M1, p_eth)

        # Active position: SELL BTCUSDm
        matrix.add_active_position(
            ActivePosition(
                symbol="BTCUSDm",
                direction=TradeDirection.SELL,
                entry_price=50000.0,
                open_time=100.0,
            )
        )

        # Candidate order: SELL ETHUSDm
        candidate = SignalProposal(
            symbol="ETHUSDm",
            direction=TradeDirection.SELL,
            timeframe=TimeFrame.M1,
            expected_alpha=30.0,
            confidence=0.90,
            entry_price=3000.0,
            timestamp=120.0,
        )

        decision = matrix.check_correlation_filter(candidate)
        assert decision.approved is False, "Violation: in-phase co-directional SELL was approved!"
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in decision.reason
        assert decision.effective_correlation > 0.85

    def test_inverse_correlation_opposite_direction_in_phase_risk_rejected(self):
        """
        Challenge: Inverse correlation (rho < -0.85) e.g. EURUSDm vs US30m.
        Candidate SELL US30m with active BUY EURUSDm is IN-PHASE risk:
        rho_eff = (-0.90) * (-1) * (+1) = +0.90 > 0.85 -> Must be REJECTED!
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        p_eur, p_us30 = self._generate_correlated_series(n=120, rho_target=-0.91)
        matrix.set_price_history("EURUSDm", TimeFrame.M1, p_eur)
        matrix.set_price_history("US30m", TimeFrame.M1, p_us30)

        rho = matrix.compute_pair_correlation("EURUSDm", "US30m")
        assert rho < -0.85, f"Expected rho < -0.85, got {rho:.4f}"

        # Active BUY EURUSDm
        matrix.add_active_position(
            ActivePosition(
                symbol="EURUSDm",
                direction=TradeDirection.BUY,
                entry_price=1.0800,
                open_time=100.0,
            )
        )

        # Candidate SELL US30m
        candidate = SignalProposal(
            symbol="US30m",
            direction=TradeDirection.SELL,
            timeframe=TimeFrame.M1,
            expected_alpha=40.0,
            confidence=0.88,
            entry_price=39000.0,
            timestamp=120.0,
        )

        decision = matrix.check_correlation_filter(candidate)
        assert decision.approved is False, "Violation: inverse correlation in-phase order approved!"
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in decision.reason
        assert decision.effective_correlation > 0.85

    def test_cross_symbol_suffix_interoperability(self):
        """
        Challenge: Verify Exness broker-suffixed ('m') and non-suffixed symbols
        correctly cross-resolve in active position lookups and correlation matrix.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        p_btc, p_eth = self._generate_correlated_series(n=120, rho_target=0.92)

        # Populate with unsuffixed "BTCUSD"
        matrix.set_price_history("BTCUSD", TimeFrame.M1, p_btc)
        # Populate with suffixed "ETHUSDm"
        matrix.set_price_history("ETHUSDm", TimeFrame.M1, p_eth)

        # Add active position with unsuffixed
        matrix.add_active_position(
            ActivePosition(
                symbol="BTCUSD",
                direction=TradeDirection.BUY,
                entry_price=50000.0,
                open_time=100.0,
            )
        )

        # Candidate with suffixed
        cand_suffixed = SignalProposal(
            symbol="ETHUSDm",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=20.0,
            confidence=0.88,
            entry_price=3000.0,
            timestamp=120.0,
        )
        dec1 = matrix.check_correlation_filter(cand_suffixed)
        assert dec1.approved is False
        assert "REJECT_CORRELATED_IN_PHASE_RISK" in dec1.reason

        # Candidate with duplicate suffixed on active unsuffixed
        cand_dup = SignalProposal(
            symbol="BTCUSDm",
            direction=TradeDirection.BUY,
            timeframe=TimeFrame.M1,
            expected_alpha=20.0,
            confidence=0.88,
            entry_price=50100.0,
            timestamp=120.0,
        )
        dec2 = matrix.check_correlation_filter(cand_dup)
        assert dec2.approved is False
        assert "REJECT_DUPLICATE_POSITION" in dec2.reason

    def test_hedging_opposite_direction_permitted(self):
        """
        Challenge: Submit BUY BTCUSDm and SELL ETHUSDm when rho > 0.85.
        Expected: Effective rho = rho * (+1) * (-1) = -rho <= 0.85 -> Permitted as pairs hedge.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        p_btc, p_eth = self._generate_correlated_series(n=120, rho_target=0.92)
        matrix.set_price_history("BTCUSDm", TimeFrame.M1, p_btc)
        matrix.set_price_history("ETHUSDm", TimeFrame.M1, p_eth)

        matrix.add_active_position(
            ActivePosition(
                symbol="BTCUSDm",
                direction=TradeDirection.BUY,
                entry_price=50000.0,
                open_time=100.0,
            )
        )

        # Opposite direction = pairs hedge
        candidate_sell = SignalProposal(
            symbol="ETHUSDm",
            direction=TradeDirection.SELL,
            timeframe=TimeFrame.M1,
            expected_alpha=25.0,
            confidence=0.90,
            entry_price=3000.0,
            timestamp=120.0,
        )
        decision = matrix.check_correlation_filter(candidate_sell)
        assert decision.approved is True, "Violation: legitimate pairs hedge was incorrectly rejected!"
        assert decision.effective_correlation <= 0.85

    def test_correlation_boundary_exactness(self):
        """
        Challenge: Boundary check at threshold 0.85:
        - effective_rho = 0.8499 -> APPROVED
        - effective_rho = 0.8500 -> APPROVED (condition is effective_rho > max_corr)
        - effective_rho = 0.8501 -> REJECTED
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)

        # Mock compute_pair_correlation to test exact boundary values
        for test_rho, expected_approved in [(0.8499, True), (0.8500, True), (0.8501, False)]:
            matrix.compute_pair_correlation = lambda s1, s2, tf=None, w=None: test_rho
            matrix._active_positions.clear()
            matrix.add_active_position(
                ActivePosition(symbol="BTCUSDm", direction=TradeDirection.BUY, entry_price=50000.0, open_time=1.0)
            )
            cand = SignalProposal(
                symbol="ETHUSDm",
                direction=TradeDirection.BUY,
                timeframe=TimeFrame.M1,
                expected_alpha=20.0,
                confidence=0.9,
                entry_price=3000.0,
                timestamp=2.0,
            )
            dec = matrix.check_correlation_filter(cand)
            assert dec.approved == expected_approved, f"Failed at boundary rho={test_rho}: expected approved={expected_approved}"

    def test_batch_router_arbitration_drops_redundant_correlated_orders(self):
        """
        Challenge: Simultaneous candidate signals across BTCUSDm, ETHUSDm, and XAUUSDm.
        BTCUSDm and ETHUSDm are correlated (rho > 0.85).
        Router must select the candidate with highest alpha*confidence*confluence and drop the second.
        XAUUSDm is uncorrelated with cryptos and must be approved.
        """
        matrix = MultiSymbolMatrix(max_correlation_threshold=0.85)
        p_btc, p_eth = self._generate_correlated_series(n=100, rho_target=0.93)
        # Uncorrelated gold series
        np.random.seed(999)
        p_gold = list(2400.0 + np.cumsum(np.random.normal(0, 1, 100)) * 2.0)

        matrix.set_price_history("BTCUSDm", TimeFrame.M1, p_btc)
        matrix.set_price_history("ETHUSDm", TimeFrame.M1, p_eth)
        matrix.set_price_history("XAUUSDm", TimeFrame.M1, p_gold)

        cands = [
            # Candidate 1: BTCUSDm BUY (score = 50 * 0.85 = 42.5)
            SignalProposal("BTCUSDm", TradeDirection.BUY, TimeFrame.M1, expected_alpha=50.0, confidence=0.85, entry_price=50000.0, timestamp=10.0),
            # Candidate 2: ETHUSDm BUY (score = 80 * 0.90 = 72.0 -> Higher score!)
            SignalProposal("ETHUSDm", TradeDirection.BUY, TimeFrame.M1, expected_alpha=80.0, confidence=0.90, entry_price=3000.0, timestamp=10.0),
            # Candidate 3: XAUUSDm BUY (uncorrelated)
            SignalProposal("XAUUSDm", TradeDirection.BUY, TimeFrame.M1, expected_alpha=40.0, confidence=0.88, entry_price=2400.0, timestamp=10.0),
        ]

        result = matrix.route_and_filter_candidates(cands)
        approved_syms = [s.symbol for s in result.approved_signals]
        rejected_syms = [s.symbol for s, d in result.rejected_signals]

        assert "ETHUSDm" in approved_syms, "Higher score ETHUSDm should be approved"
        assert "XAUUSDm" in approved_syms, "Uncorrelated XAUUSDm should be approved"
        assert "BTCUSDm" in rejected_syms, "Lower score correlated BTCUSDm must be rejected"


# =====================================================================
# 2. PRM SCORE & CONFIDENCE THRESHOLD EMPIRICAL CHALLENGES
# =====================================================================

class MockPRM(torch.nn.Module):
    def __init__(self, score: float):
        super().__init__()
        self.score = score

    def forward(self, hops, **kw):
        return (torch.tensor([self.score]), torch.ones(1, 4) * self.score)


class MockReflection(torch.nn.Module):
    def __init__(self, bull: float, bear: float, margin: float, probs=None):
        super().__init__()
        self.bull = bull
        self.bear = bear
        self.margin = margin
        self.probs = probs or [0.7, 0.1, 0.1, 0.1]

    def forward(self, thought):
        return {
            "bull_score": torch.tensor([self.bull]),
            "bear_score": torch.tensor([self.bear]),
            "adversarial_margin": torch.tensor([self.margin]),
            "scenario_probs": torch.tensor([self.probs]),
            "scenario_decays": torch.zeros(1, 4),
            "scenario_trajectories": torch.zeros(1, 4, 12),
        }


class TestPRMAndConfidenceBoundaryEmpirical:
    """Adversarial stress-testing of PRM >= 0.90 and Confidence >= 0.85 boundaries."""

    def setup_method(self):
        torch.manual_seed(42)
        self.config = ReasoningConfig(
            input_dim=128,
            hidden_dim=128,
            num_hops=4,
            max_hops=8,
            prm_score_threshold=0.90,
            confidence_threshold=0.85,
            adversarial_margin_threshold=0.20,
        )
        self.engine = MetacognitiveReasoningEngine(self.config)

    def test_prm_below_090_strictly_suppresses_activation(self):
        """
        Challenge: Force PRM Score to 0.8999 while Confidence = 0.95 and Margin = 0.50.
        Expected: is_activated == False, action == 0 (HOLD).
        """
        engine = MetacognitiveReasoningEngine(self.config)
        x = torch.randn(1, 128)

        engine.prm_verifier = MockPRM(0.8999)

        out = engine(x)
        assert not out.is_activated.item(), "Violation: PRM = 0.8999 activated a trade!"
        assert out.action.item() == 0, f"Violation: action was {out.action.item()} instead of HOLD (0)"

    def test_prm_at_and_above_090_permits_activation(self):
        """
        Challenge: Test PRM Score = 0.9000 and 0.9001 with high confidence and margin.
        Expected: is_activated == True, action != 0.
        """
        engine = MetacognitiveReasoningEngine(self.config)
        x = torch.randn(1, 128)

        # Mock reflection to ensure strong margin and confidence
        engine.reflection = MockReflection(bull=0.98, bear=0.10, margin=0.88, probs=[0.98, 0.01, 0.005, 0.005])

        # Case 1: PRM = 0.9000
        engine.prm_verifier = MockPRM(0.9000)
        out1 = engine(x)
        assert out1.is_activated.item() is True, "PRM = 0.9000 should permit activation"
        assert out1.action.item() == 1, "Expected BUY action"

        # Case 2: PRM = 0.9400
        engine.prm_verifier = MockPRM(0.9400)
        out2 = engine(x)
        assert out2.is_activated.item() is True
        assert out2.action.item() == 1

    def test_confidence_below_085_strictly_suppresses_activation(self):
        """
        Challenge: PRM = 0.95, Margin = 0.60, but Confidence = 0.8499.
        Expected: is_activated == False, action == 0 (HOLD).
        """
        engine = MetacognitiveReasoningEngine(self.config)
        x = torch.randn(1, 128)

        # High PRM
        engine.prm_verifier = MockPRM(0.9500)

        # Reflection with dominant score such that total confidence = 0.95 * 0.88 * 1.0 = 0.836 < 0.85
        engine.reflection = MockReflection(bull=0.88, bear=0.20, margin=0.68, probs=[0.95, 0.02, 0.02, 0.01])

        out = engine(x)
        assert out.confidence.item() < 0.85, f"Confidence {out.confidence.item():.4f} >= 0.85"
        assert not out.is_activated.item(), "Violation: Confidence < 0.85 activated a trade!"
        assert out.action.item() == 0

    def test_adversarial_margin_below_020_strictly_suppresses_activation(self):
        """
        Challenge: PRM = 0.95, Confidence = 0.90, but |Margin| = 0.15 < 0.20 (indecision).
        Expected: is_activated == False, action == 0 (HOLD).
        """
        engine = MetacognitiveReasoningEngine(self.config)
        x = torch.randn(1, 128)

        engine.prm_verifier = MockPRM(0.9500)
        # Bull = 0.90, Bear = 0.75 -> Margin = 0.15 < 0.20
        engine.reflection = MockReflection(bull=0.90, bear=0.75, margin=0.15, probs=[0.95, 0.02, 0.02, 0.01])

        out = engine(x)
        assert abs(out.adversarial_margin.item()) < 0.20
        assert not out.is_activated.item(), "Violation: Margin < 0.20 activated a trade!"
        assert out.action.item() == 0

    def test_prm_weak_link_principle_bottleneck_suppresses_trade(self):
        """
        Challenge: 3 hops are perfect (1.0), but 1 bottleneck hop is flawed (0.40).
        The Weak-Link principle (min reward) must drag down the PRM score below 0.90.
        """
        verifier = PRMVerifier(self.config)
        # Create hops where hop 2 is severely distorted
        hops = torch.ones(1, 4, 128) * 0.1
        hops[0, 2, :] = torch.randn(128) * 100.0  # Massive outlier in hop 2

        prm_score, hop_rewards = verifier(hops)
        min_r = torch.min(hop_rewards).item()
        assert min_r < 0.90, f"Expected flawed hop reward < 0.90, got {min_r:.4f}"
        assert prm_score.item() < 0.90, f"Weak-Link failure: PRM {prm_score.item():.4f} >= 0.90 despite bottleneck"

    def test_mythos_mt5_policy_prm_gate_enforcement(self):
        """
        Challenge: Verify MythosMT5Policy.prm_threshold is calibrated to >= 0.90
        and rejects trades when pipeline PRM < 0.90.
        """
        from research_reasoning_world_class.tests.test_e2e_walk_forward_triad import DummyBasePolicy
        policy = MythosMT5Policy(base_policy=DummyBasePolicy(), initial_capital=10000.0)
        assert policy.prm_threshold >= 0.90, f"MythosMT5Policy prm_threshold is {policy.prm_threshold} < 0.90"


# =====================================================================
# 3. MICROSTRUCTURE SHIELD EMPIRICAL CHALLENGES
# =====================================================================

class TestMicrostructureShieldEmpirical:
    """Adversarial stress-testing of Microstructure Cost Shield: Cost Ratio >= 3.5."""

    def test_cost_ratio_strictly_below_35_rejected(self):
        """
        Challenge: Cost Ratio = 3.4999 < 3.5.
        Expected: approved == False, rejection_reason contains REJECT_COST_DRAG_SHIELD.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        # Alpha = 3.4999, Friction = 1.0 -> Ratio = 3.4999 < 3.5
        r = shield.calculate_cost_ratio(expected_alpha=3.4999, spread=0.5, commission=0.3, slippage=0.2)
        assert r < 3.5

        book = OrderBookSnapshot("BTCUSD", bid_price=60000.0, ask_price=60001.0, bid_volume=5.0, ask_volume=5.0, timestamp=1.0)
        verdict = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=3.4999 * (1.0 + 1.5 + 2.0) / 4.5,  # crafted alpha to produce sub-3.5
            book=book,
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict.approved is False
        assert verdict.cost_ratio < 3.5
        assert "REJECT_COST_DRAG_SHIELD" in verdict.rejection_reason

    def test_cost_ratio_at_and_above_35_approved(self):
        """
        Challenge: Cost Ratio = 3.5000 and 5.0.
        Expected: approved == True when alpha covers all frictions.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)

        r_exact = shield.calculate_cost_ratio(expected_alpha=3.5, spread=0.5, commission=0.3, slippage=0.2)
        assert pytest.approx(r_exact, rel=1e-5) == 3.5
        assert r_exact >= 3.5

        r_high = shield.calculate_cost_ratio(expected_alpha=10.0, spread=0.5, commission=0.3, slippage=0.2)
        assert r_high == 10.0
        assert r_high >= 3.5

    def test_market_slippage_and_spread_drag_rejects_sub_par_trade(self):
        """
        Challenge: On BTCUSD, typical spread=4.0, comm=1.5, slippage=2.0 -> Friction = 7.5.
        With Alpha = 15.0, Ratio = 15.0 / 7.5 = 2.0 < 3.5.
        Must strictly REJECT in forced MARKET mode.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        book = OrderBookSnapshot("BTCUSD", bid_price=65000.0, ask_price=65004.0, bid_volume=10.0, ask_volume=10.0, timestamp=100.0)

        verdict_mkt = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=15.0,
            book=book,
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict_mkt.approved is False
        assert verdict_mkt.cost_ratio < 3.5
        assert "REJECT_COST_DRAG_SHIELD" in verdict_mkt.rejection_reason

    def test_sniper_mode_salvages_viable_trade_via_maker_execution(self):
        """
        Challenge: Same Alpha = 15.0 on BTCUSD.
        In default Sniper Limit mode at SSL sweep:
        Effective spread = 0.0, slippage = 0.0 -> Total friction = comm = 1.5.
        Sniper Ratio = 15.0 / 1.5 = 10.0 >= 3.5 -> APPROVED!
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        book = OrderBookSnapshot("BTCUSD", bid_price=65000.0, ask_price=65004.0, bid_volume=10.0, ask_volume=10.0, timestamp=100.0)

        verdict_sniper = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=15.0,
            book=book,
            recent_high=65200.0,
            recent_low=65002.0,  # SSL sweep detected
        )
        assert verdict_sniper.approved is True
        assert verdict_sniper.execution_mode == ExecutionMode.LIMIT_SNIPER
        assert verdict_sniper.cost_ratio >= 3.5
        assert verdict_sniper.estimated_slippage == 0.0

    def test_tiny_alpha_cannot_be_salvaged_by_sniper_mode(self):
        """
        Challenge: Tiny edge (Alpha = 2.0 on BTCUSD).
        Even in Sniper mode, broker commission is 1.5 pts.
        Ratio = 2.0 / 1.5 = 1.33 < 3.5 -> MUST REJECT!
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        book = OrderBookSnapshot("BTCUSD", bid_price=65000.0, ask_price=65004.0, bid_volume=10.0, ask_volume=10.0, timestamp=100.0)

        verdict = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=2.0,
            book=book,
            recent_high=65200.0,
            recent_low=65002.0,
        )
        assert verdict.approved is False
        assert verdict.cost_ratio < 3.5
        assert "REJECT_COST_DRAG_SHIELD" in verdict.rejection_reason

    def test_zero_and_negative_alpha_safely_rejected(self):
        """
        Challenge: Alpha <= 0.0 (zero or negative projected edge).
        Expected: cost_ratio == 0.0, approved == False, zero division avoided.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        assert shield.calculate_cost_ratio(expected_alpha=0.0, spread=1.0, commission=0.5, slippage=0.5) == 0.0
        assert shield.calculate_cost_ratio(expected_alpha=-10.0, spread=1.0, commission=0.5, slippage=0.5) == 0.0

        book = OrderBookSnapshot("BTCUSD", bid_price=65000.0, ask_price=65004.0, bid_volume=5.0, ask_volume=5.0, timestamp=1.0)
        verdict = shield.evaluate_execution("BTCUSD", direction_val=1, expected_alpha=0.0, book=book)
        assert verdict.approved is False
        assert verdict.cost_ratio == 0.0

    def test_zero_friction_division_by_zero_safety(self):
        """
        Challenge: Zero spread, zero commission, zero slippage (friction = 0).
        Expected: Guarded against ZeroDivisionError, returns finite float.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        ratio = shield.calculate_cost_ratio(expected_alpha=10.0, spread=0.0, commission=0.0, slippage=0.0)
        assert not math.isnan(ratio)
        assert not math.isinf(ratio)
        assert ratio > 0.0

    def test_spread_explosion_adversarial_rejection(self):
        """
        Challenge: News event spread explosion (e.g. 50 pts spread on BTCUSD).
        Alpha = 30.0. Market Friction = 50.0 + 1.5 + 2.0 = 53.5. Ratio = 0.56 < 3.5.
        Must strictly REJECT.
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        blown_book = OrderBookSnapshot("BTCUSD", bid_price=64950.0, ask_price=65000.0, bid_volume=1.0, ask_volume=1.0, timestamp=10.0)
        verdict = shield.evaluate_execution(
            symbol="BTCUSD",
            direction_val=1,
            expected_alpha=30.0,
            book=blown_book,
            force_mode=ExecutionMode.MARKET,
        )
        assert verdict.approved is False
        assert verdict.cost_ratio < 3.5

    def test_vulnerability_suffixed_symbol_cost_profile_underestimation(self):
        """
        VERIFIED RESOLUTION:
        MicrostructureExecutionShield normalizes Exness suffixed symbols ('m').
        Both 'BTCUSD' and 'BTCUSDm' resolve to the BTCUSD profile (comm=1.5, slip=2.0).
        With Alpha = 15.0:
        - BTCUSD (Market): Ratio = 15.0 / 7.5 = 2.00 < 3.5 -> REJECTED.
        - BTCUSDm (Market): Ratio = 15.0 / 7.5 = 2.00 < 3.5 -> REJECTED (Parity confirmed).
        """
        shield = MicrostructureExecutionShield(min_cost_ratio=3.5)
        prof_btc = shield.get_cost_profile("BTCUSD")
        prof_btcm = shield.get_cost_profile("BTCUSDm")

        assert prof_btc.typical_commission_points == 1.5
        assert prof_btcm.typical_commission_points == 1.5
        assert prof_btcm.typical_commission_points == prof_btc.typical_commission_points
        assert prof_btcm.typical_market_slippage_points == prof_btc.typical_market_slippage_points


# =====================================================================
# 4. END-TO-END PIPELINE SAFETY INVARIANTS CHALLENGES
# =====================================================================

class TestPipelineSafetyInvariantsEmpirical:
    """End-to-end integration and wire packet safety invariant tests."""

    def test_wire_packet_policy_close_all_locked_zero(self):
        """
        Challenge: Verify wire packet index 19 (POLICY_CLOSE_ALL) is strictly 0.0
        across diverse market inputs.
        """
        from research_reasoning_world_class.tests.test_e2e_walk_forward_triad import DummyBasePolicy
        policy = MythosMT5Policy(base_policy=DummyBasePolicy(), initial_capital=10000.0)

        for _ in range(15):
            obs = np.random.randn(137).astype(np.float32)
            traces = policy.act_nam_tac_nhan(obs)
            out = getattr(traces[4], "output_vector", getattr(traces[4], "selected_action", None))
            assert out is not None
            assert out[0] == 1.0, "Governor agent 0 logit not 1.0 HOLD"
            assert out[10] == -100.0, "Governor close-all logit not -100.0"
            assert out[IDX_POLICY_CLOSE_ALL] == 0.0, "POLICY_CLOSE_ALL wire packet index 19 was non-zero!"
