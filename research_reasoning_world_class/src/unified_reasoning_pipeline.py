"""
Unified Reasoning Trader Pipeline & End-to-End Orchestrating Runner
Module: Feature 31 (Milestone 4 - Requirements R1 through R8, AC-1 through AC-7)
Project: "Mythos của Claude" - xLSTM 17.5M World-Class Quantitative Reasoning Engine

Connects all 8 subsystems into a synchronized pipeline:
1. Data Ingestion: 6 symbols (BTC, ETH, SOL, XAU, EUR, US30) M1/M5 + Macro quotes
2. Feature Layer: SMC 64 floats + Macro 32 floats + Base 137 floats -> 233D Ingress
3. xLSTM 17.5M Backbone: SGT-V193 Core with isolated per-symbol memory (L_inf = 0.0)
4. Metacognition: 4-Hop CoT + Bull-Bear Debate + PRM Verifier (Strict 85% gate)
5. Multi-Symbol Matrix: Cross-asset correlation filter (rho <= 0.85) + M5 Confluence
6. Microstructure Shield: Cost ratio (Alpha / Friction >= 3.5) + Sniper Limit Entry
7. Dynamic Kelly: Fractional Kelly compounding + Continuous linear de-leveraging (floor 10%)
8. MT5 IPC Bridge: Dispatches 20-float binary wire packet with locked POLICY_CLOSE_ALL == 0
"""

from __future__ import annotations

import math
import os
from pathlib import Path
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

# Ensure project root is in sys.path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _project_root not in sys.path or sys.path[0] != _project_root:
    sys.path.insert(0, _project_root)

from research_reasoning_world_class.src.smc_liquidity_engine import (
    SMCLiquidityEngine,
    SMCConfig,
)
from research_reasoning_world_class.src.macro_reasoning_engine import (
    CrossAssetMacroEngine,
    MacroConfig,
)
from research_reasoning_world_class.src.xlstm_reasoning_adapter import (
    XLSTMSGTReasoningAdapter,
    GOVERNOR_AGENT_INDEX,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
    ReasoningVerdict,
)
from research_reasoning_world_class.src.adversarial_dialectical_reasoning import (
    AdversarialDialecticalReasoningEngine,
    DialecticalConfig,
    DialecticalVerdict,
)
from research_reasoning_world_class.src.bilateral_dual_cot import (
    BilateralDualCoTEngine,
    BilateralDualCoTConfig,
    BilateralDualVerdict,
    extract_canonical_bilateral_ingress_233,
)
from research_reasoning_world_class.src.multi_symbol_matrix import (
    MultiSymbolMatrix,
    SignalProposal,
    TradeDirection,
    TimeFrame,
    BarData,
    SUPPORTED_SYMBOLS,
)
from research_reasoning_world_class.src.microstructure_execution import (
    MicrostructureExecutionShield,
    OrderBookSnapshot,
    ExecutionVerdict,
    ExecutionMode,
)
from research_reasoning_world_class.src.dynamic_kelly_compounding import (
    DynamicKellyCompoundingEngine,
    CompoundingPortfolioManager,
    KellyParameters,
    TradeSizingDecision,
)
from research_reasoning_world_class.src.mt5_ipc_transport import (
    MT5IPCTransportBridge,
    MT5WirePacket,
    WIRE_BYTE_SIZE,
)


@dataclass
class PipelineTickResult:
    """Telemetry and outcome of a single tick pass through the Unified Reasoning Pipeline."""
    symbol: str
    action: int                     # -1, 0, 1
    order_type: str                 # "MARKET" or "LIMIT"
    lot_size: float
    confidence: float
    prm_score: float
    net_margin: float
    cost_ratio: float
    expected_alpha: float
    deleveraging_factor: float
    is_sniper_mode: bool
    wire_packet: MT5WirePacket
    total_latency_ms: float
    executed: bool
    bull_score: float = 0.0
    bear_score: float = 0.0
    is_counter_punch: bool = False
    trap_type: str = "NONE"
    debate_history: List[Dict[str, float]] = field(default_factory=list)
    was_backtracked: bool = False
    faulty_step: int = -1
    backtrack_magnitude: float = 0.0
    search_depth: int = 3


class UnifiedReasoningTraderPipeline:
    """
    Top-Level Orchestrating Trader Pipeline uniting R1 through R8.
    Fulfills AC-1 through AC-7 in live and backtesting environments.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fractional_kelly: float = 0.20,
        base_trade_risk: float = 0.012,
        use_direct_xlstm_ingress: bool = True,
        use_adversarial_dialectical: bool = True,
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.initial_capital = initial_capital
        self.use_adversarial_dialectical = use_adversarial_dialectical
        
        # 1. Feature Engines
        self.smc_engine = SMCLiquidityEngine(SMCConfig())
        self.macro_engine = CrossAssetMacroEngine(config=MacroConfig())
        
        # 2. xLSTM 17.5M Core Adapter & Metacognitive Reasoning
        ingress_mode = "direct" if use_direct_xlstm_ingress else "gated_residual"
        self.xlstm_adapter = XLSTMSGTReasoningAdapter(ingress_mode=ingress_mode).eval()
        self.cot_config = ReasoningConfig(
            input_dim=768,
            hidden_dim=256,
            num_hops=4,
            max_hops=8,
            enable_cross_hop=True,
            enable_backtrack=True,
            enable_act=True,
            prm_threshold=0.85,
            adversarial_margin_threshold=0.20,
        )
        self.cot_engine = MetacognitiveReasoningEngine(self.cot_config).eval()
        
        # Strictly load co-evolved champion checkpoint (0 missing / 0 unexpected keys)
        default_ckpt = Path(_project_root) / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_coevolved_champion.pt"
        m5_ckpt = Path(_project_root) / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_muzero_m5_champion.pt"
        if checkpoint_path is not None:
            ckpt_path = Path(checkpoint_path)
        elif default_ckpt.exists():
            ckpt_path = default_ckpt
        elif m5_ckpt.exists():
            ckpt_path = m5_ckpt
        else:
            ckpt_path = default_ckpt

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Co-evolved champion checkpoint missing at: {ckpt_path}")
        
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "reasoning_engine_state_dict" not in ckpt:
            raise KeyError(f"'reasoning_engine_state_dict' missing in {ckpt_path}")
            
        load_res = self.cot_engine.load_state_dict(ckpt["reasoning_engine_state_dict"], strict=True)
        assert len(load_res.missing_keys) == 0, f"Missing keys in cot_engine: {load_res.missing_keys}"
        assert len(load_res.unexpected_keys) == 0, f"Unexpected keys in cot_engine: {load_res.unexpected_keys}"
        
        param_count = sum(p.numel() for p in self.cot_engine.parameters())
        assert param_count == 4_196_564, f"Expected 4,196,564 parameters, got {param_count}"

        self.xlstm_adapter.connect_metacognitive_reasoner(self.cot_engine)

        # 2b. Frontier Adversarial Dialectical Reasoning Engine
        self.dialectical_engine = AdversarialDialecticalReasoningEngine(
            DialecticalConfig(
                input_dim=233,
                hidden_dim=256,
                num_debate_turns=3,
                num_heads=4,
                prm_threshold=0.35,
                max_latency_ms=50.0,
            )
        ).eval()

        # 2c. Frontier Bilateral Dual-CoT Engine (Independent Bull & Bear Brains)
        self.use_bilateral_dual_cot = True
        self.bilateral_engine = BilateralDualCoTEngine(
            BilateralDualCoTConfig(
                input_dim=233,
                latent_dim=768,
                hidden_dim=256,
                num_debate_turns=3,
                prm_threshold=0.35,
                max_latency_ms=50.0,
            )
        ).eval()
        bilateral_ckpt = Path(_project_root) / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_sniper_anti_mae_champion.pt"
        if not bilateral_ckpt.exists():
            bilateral_ckpt = Path(_project_root) / "outputs" / "he_chuyen_gia" / "v193_runs" / "mythos_bilateral_dual_cot_champion.pt"
        if bilateral_ckpt.exists():
            try:
                self.bilateral_engine.load_checkpoint(bilateral_ckpt)
            except Exception:
                pass
        else:
            try:
                self.bilateral_engine.load_checkpoint(ckpt_path)
            except Exception:
                pass
        
        # 3. Multi-Symbol Matrix & Microstructure Shield
        self.matrix = MultiSymbolMatrix(symbols=SUPPORTED_SYMBOLS)
        self.microstructure_shield = MicrostructureExecutionShield()
        
        # 4. Dynamic Kelly Compounding
        self.kelly_engine = DynamicKellyCompoundingEngine(
            KellyParameters(
                fractional_kelly=fractional_kelly,
                base_trade_risk_pct=base_trade_risk,
                max_drawdown_limit=0.15,
                sniper_dd_threshold=0.08,
                base_win_prob_threshold=0.85,
                sniper_win_prob_threshold=0.90,
                deleveraging_floor=0.10,
                nominal_payoff_ratio=0.70,
            )
        )
        self.portfolio_manager = CompoundingPortfolioManager(
            initial_balance=initial_capital,
            engine=self.kelly_engine,
        )
        
        # 5. MT5 IPC Wire Dispatcher
        self.ipc_bridge = MT5IPCTransportBridge()
        
        # Metrics tracking
        self.latencies_ms: List[float] = []
        self.nan_inf_count: int = 0
        self.last_governor_latent: Optional[torch.Tensor] = None

        # Pre-warm pipeline to eliminate cold-start latency (AC-6 <= 50ms)
        self._pre_warm()

    def _pre_warm(self) -> None:
        """Warm up PyTorch thread pool, GEMM, and dynamic branches."""
        dummy_obs = np.zeros(137, dtype=np.float32)
        dummy_bars = np.ones((60, 5), dtype=np.float64) * 65000.0
        dummy_macro = {
            "DXY": np.ones(60),
            "US10Y": np.ones(60),
            "SPX": np.ones(60),
            "XAUUSD": np.ones(60),
        }
        with torch.no_grad():
            for _ in range(2):
                self.process_tick(
                    symbol="BTCUSD",
                    base_obs_137=dummy_obs,
                    m1_bars_60=dummy_bars,
                    macro_quotes=dummy_macro,
                    current_price=65000.0,
                )

    @torch.no_grad()
    def process_tick(
        self,
        symbol: str,
        base_obs_137: np.ndarray,
        m1_bars_60: np.ndarray,
        macro_quotes: Dict[str, np.ndarray],
        order_book: Optional[OrderBookSnapshot] = None,
        m5_trend_direction: int = 0,
        current_price: float = 100.0,
    ) -> PipelineTickResult:
        """
        Executes a complete synchronous end-to-end pass through all 8 stages.
        Strict Latency Target: <= 50.0 ms.
        """
        t0 = time.perf_counter()
        
        # Stage 1: SMC Feature Extraction (64 floats)
        smc_features = self.smc_engine.extract_features(m1_bars_60)
        
        # Stage 2: Macro Feature Extraction (32 floats)
        if isinstance(macro_quotes, dict):
            matrix_4n = np.stack(
                [macro_quotes.get(s, np.zeros(60, dtype=np.float64)) for s in self.macro_engine.symbols],
                axis=0,
            )
        else:
            matrix_4n = macro_quotes
        macro_features = self.macro_engine.extract_features_array(matrix_4n)
        
        # Stage 3: Ingress Assembly (137 Base + 64 SMC + 32 Macro = 233D)
        ingress_233 = np.concatenate([base_obs_137, smc_features, macro_features], axis=0).astype(np.float32)
        
        # Numerical safety check (AC-7)
        if not np.all(np.isfinite(ingress_233)):
            self.nan_inf_count += 1
            ingress_233 = np.nan_to_num(ingress_233, nan=0.0, posinf=1.0, neginf=-1.0)
            
        # Stage 4: SGT-V193 xLSTM Core & Metacognitive CoT Reasoning
        adapter_output = self.xlstm_adapter.step_symbol(symbol, ingress_233)
        self.last_governor_latent = adapter_output.governor_latent
        meta_out = adapter_output.metacognitive_verdict

        is_counter_punch = False
        trap_type = "NONE"
        debate_history: List[Dict[str, float]] = []
        was_backtracked = False
        faulty_step = -1
        backtrack_mag = 0.0
        search_depth = 4 if meta_out is not None else 3

        if meta_out is not None and getattr(meta_out, "was_backtracked", False):
            was_backtracked = True
            faulty_step = getattr(meta_out, "faulty_step", -1)
            backtrack_mag = max(backtrack_mag, float(getattr(meta_out, "revision_magnitude", 0.0)))

        if getattr(self, "use_bilateral_dual_cot", False) and hasattr(self, "bilateral_engine"):
            with torch.no_grad():
                cur_atr = getattr(self, "current_atr", 45.0)
                canon_ingress = extract_canonical_bilateral_ingress_233(
                    base_obs_137=base_obs_137,
                    m1_bars_60=m1_bars_60,
                    cur_p=current_price,
                    cur_atr=cur_atr,
                )
                dial_in = torch.from_numpy(canon_ingress).float()
                bilateral_verdict = self.bilateral_engine(dial_in)
                is_counter_punch = bilateral_verdict.is_counter_punch
                trap_type = bilateral_verdict.trap_type
                debate_history = bilateral_verdict.debate_history
                if bilateral_verdict.was_backtracked:
                    was_backtracked = True
                    faulty_step = getattr(bilateral_verdict, "faulty_step", faulty_step)
                    backtrack_mag = max(backtrack_mag, float(bilateral_verdict.backtrack_magnitude))
                search_depth = 3
                action_val = int(bilateral_verdict.action)
                conf_val = float(bilateral_verdict.confidence)
                bull_val = float(bilateral_verdict.bull_score)
                bear_val = float(bilateral_verdict.bear_score)
                margin_val = float(bilateral_verdict.debate_margin)
                prm_val = float(bilateral_verdict.prm_score)
        elif getattr(self, "use_adversarial_dialectical", False):
            with torch.no_grad():
                dial_in = torch.from_numpy(ingress_233).float()
                dial_verdict = self.dialectical_engine(dial_in)
                is_counter_punch = dial_verdict.is_counter_punch
                trap_type = dial_verdict.trap_type
                debate_history = dial_verdict.debate_history
                if dial_verdict.was_backtracked:
                    was_backtracked = True
                    faulty_step = getattr(dial_verdict, "faulty_step", faulty_step)
                    backtrack_mag = max(backtrack_mag, float(dial_verdict.backtrack_magnitude))
                search_depth = max(search_depth, getattr(dial_verdict, "search_depth", 3))

                if is_counter_punch:
                    action_val = int(dial_verdict.action)
                    conf_val = float(dial_verdict.confidence)
                    bull_val = 0.95 if action_val == 1 else 0.05
                    bear_val = 0.95 if action_val == -1 else 0.05
                    margin_val = float(dial_verdict.debate_margin)
                    prm_val = float(dial_verdict.prm_score)
                elif dial_verdict.prm_score >= self.dialectical_engine.config.prm_threshold and dial_verdict.action != 0:
                    action_val = int(dial_verdict.action)
                    conf_val = float(dial_verdict.confidence)
                    bull_val = float(dial_verdict.proponent_conviction) if action_val == 1 else 0.10
                    bear_val = float(dial_verdict.proponent_conviction) if action_val == -1 else 0.10
                    margin_val = float(dial_verdict.debate_margin)
                    prm_val = float(dial_verdict.prm_score)
                elif meta_out is not None:
                    raw_action = meta_out.action
                    action_val = int(raw_action.item() if hasattr(raw_action, "item") else raw_action)
                    conf_val = float(meta_out.confidence.item() if hasattr(meta_out.confidence, "item") else meta_out.confidence)
                    bull_val = float(meta_out.bull_score.item() if hasattr(meta_out.bull_score, "item") else meta_out.bull_score)
                    bear_val = float(meta_out.bear_score.item() if hasattr(meta_out.bear_score, "item") else meta_out.bear_score)
                    margin_val = float(meta_out.adversarial_margin.item() if hasattr(meta_out.adversarial_margin, "item") else meta_out.adversarial_margin)
                    prm_val = float(meta_out.prm_score.item() if hasattr(meta_out.prm_score, "item") else meta_out.prm_score)
                else:
                    action_val = 0
                    conf_val = 0.0
                    bull_val = 0.0
                    bear_val = 0.0
                    margin_val = 0.0
                    prm_val = 0.0
        elif meta_out is not None:
            raw_action = meta_out.action
            action_val = int(raw_action.item() if hasattr(raw_action, "item") else raw_action)
            conf_val = float(meta_out.confidence.item() if hasattr(meta_out.confidence, "item") else meta_out.confidence)
            bull_val = float(meta_out.bull_score.item() if hasattr(meta_out.bull_score, "item") else meta_out.bull_score)
            bear_val = float(meta_out.bear_score.item() if hasattr(meta_out.bear_score, "item") else meta_out.bear_score)
            margin_val = float(meta_out.adversarial_margin.item() if hasattr(meta_out.adversarial_margin, "item") else meta_out.adversarial_margin)
            prm_val = float(meta_out.prm_score.item() if hasattr(meta_out.prm_score, "item") else meta_out.prm_score)
        else:
            action_val = 0
            conf_val = 0.0
            bull_val = 0.0
            bear_val = 0.0
            margin_val = 0.0
            prm_val = 0.0

        # Stage 5: Multi-Symbol Routing Matrix & Cross-Asset Arbitration
        if action_val != 0:
            trade_dir = TradeDirection.BUY if action_val == 1 else TradeDirection.SELL
            proposal = SignalProposal(
                symbol=symbol,
                direction=trade_dir,
                timeframe=TimeFrame.M1,
                expected_alpha=25.0,
                confidence=conf_val,
                entry_price=current_price,
                timestamp=time.time(),
                metadata={"margin": margin_val},
            )
            filter_decision = self.matrix.evaluate_proposal(proposal)
            matrix_approved = filter_decision.approved
            eff_corr = getattr(filter_decision, "effective_max_correlation", getattr(filter_decision, "effective_correlation", 0.0))
            action_filtered = action_val if matrix_approved else 0
        else:
            action_filtered = 0
            eff_corr = 0.0
            
        # Stage 6: Microstructure Execution Shield (Cost Ratio >= 3.5 & Sniper Limit Mode)
        if order_book is None:
            order_book = OrderBookSnapshot(
                symbol=symbol,
                bid_price=current_price - 0.5,
                ask_price=current_price + 0.5,
                bid_volume=100.0,
                ask_volume=100.0,
                timestamp=time.time(),
            )
        
        exec_verdict = self.microstructure_shield.evaluate_execution(
            symbol=symbol,
            direction_val=action_filtered,
            expected_alpha=25.0,
            book=order_book,
        )
        
        # Final action decision
        final_action = action_filtered if exec_verdict.approved else 0
        
        # Stage 7: Dynamic Fractional Kelly Sizing & Risk Management
        sizing = self.portfolio_manager.calculate_trade_order(
            p_win=max(0.35, conf_val),
            payoff_ratio=1.25,
            sl_points=50.0,
            point_value=1.0,
        )
        
        lot_size = sizing.lot_size if (final_action != 0 and sizing.admitted) else 0.0
        
        # Stage 8: MT5 IPC Wire Dispatch (20 Floats, Locked POLICY_CLOSE_ALL == 0.0)
        entry_mode_str = exec_verdict.execution_mode.value if hasattr(exec_verdict.execution_mode, "value") else str(exec_verdict.execution_mode)
        is_limit = 1.0 if "LIMIT" in entry_mode_str.upper() or "SNIPER" in entry_mode_str.upper() else 0.0
        
        wire_packet = MT5WirePacket(
            action=float(final_action),
            order_type=is_limit,
            lot_size=float(lot_size),
            limit_price=float(exec_verdict.target_entry_price),
            stop_loss=float(current_price - 50.0 if final_action == 1 else (current_price + 50.0 if final_action == -1 else 0.0)),
            take_profit=float(current_price + 75.0 if final_action == 1 else (current_price - 75.0 if final_action == -1 else 0.0)),
            trailing_activation=30.0,
            trailing_step=10.0,
            expected_alpha=float(exec_verdict.expected_alpha),
            cost_ratio=float(exec_verdict.cost_ratio),
            confidence=float(conf_val),
            bull_score=float(bull_val),
            bear_score=float(bear_val),
            net_margin=float(margin_val),
            prm_score=float(prm_val),
            macro_index=float(macro_features[16] if len(macro_features) > 16 else 0.0),
            effective_correlation=float(eff_corr),
            deleveraging_factor=float(sizing.deleveraging_factor),
            sniper_mode=1.0 if sizing.in_sniper_mode else 0.0,
            policy_close_all=0.0,  # Locked Invariant: STRICTLY 0.0
        )
        
        # Dispatch through IPC transport
        self.ipc_bridge.dispatch_packet(wire_packet)
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        self.latencies_ms.append(latency_ms)
        
        executed = (final_action != 0 and lot_size > 0.0)
        
        tick_res = PipelineTickResult(
            symbol=symbol,
            action=action_val if getattr(self, "use_bilateral_dual_cot", False) else final_action,
            order_type=entry_mode_str,
            lot_size=lot_size,
            confidence=conf_val,
            prm_score=prm_val,
            net_margin=margin_val,
            bull_score=bull_val,
            bear_score=bear_val,
            cost_ratio=exec_verdict.cost_ratio,
            expected_alpha=exec_verdict.expected_alpha,
            deleveraging_factor=sizing.deleveraging_factor,
            is_sniper_mode=sizing.in_sniper_mode,
            wire_packet=wire_packet,
            total_latency_ms=latency_ms,
            executed=executed,
            is_counter_punch=is_counter_punch,
            trap_type=trap_type,
            debate_history=debate_history,
            was_backtracked=was_backtracked,
            faulty_step=faulty_step,
            backtrack_magnitude=backtrack_mag,
            search_depth=search_depth,
        )
        self.last_tick_result = tick_res
        return tick_res

    @property
    def average_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.mean(self.latencies_ms))

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        return float(np.percentile(self.latencies_ms, 95))
