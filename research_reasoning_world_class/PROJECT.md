# Project: "Mythos của Claude" xLSTM 17.5M Reasoning System

## Architecture
The system elevates the SGT-V193-Compact 17.5M xLSTM architecture into an institutional-grade, multi-asset reasoning engine ("Mythos của Claude") capable of delivering hyper-growth (>= 50x in 90 days) while strictly controlling drawdown (< 15.0%).

```
[MT5 Tick & Bar Feed (6 Assets: BTC, ETH, SOL, XAU, EUR, US30 M1/M5) + Macro (DXY, US10Y, SPX)]
                                        │
                                        ▼
      ┌─────────────────────────────────┴─────────────────────────────────┐
      │                   Feature Extraction Layer (96 floats)             │
      │  - SMC/ICT Liquidity Engine (64 floats: Sweeps, FVG, Order Blocks) │
      │  - Cross-Asset Macro Engine (32 floats: Risk-On/Off, Velocities)  │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │ (137 Base + 96 Features = 233 floats)
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │         SGT-V193 xLSTM 17.5M Sequence Core + Torch JIT Engine     │
      │  - 233D -> 1024D Observation Projection                           │
      │  - Layer 1: mLSTMCell (768D, 4 heads, 147k covariance memory)     │
      │  - Layer 2: sLSTMCell (768D, 4 heads, stabilized scalar memory)   │
      │  - Relational Cross-Agent Graph Transformer (5 agents, 768D)      │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │ (768D Latent State)
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │     Metacognitive Multi-hop CoT & Bull-Bear Adversarial Debate    │
      │  - 4-Hop Gated Reasoning Head (Regime, Liquidity, Momentum, Act)  │
      │  - Adversarial Self-Reflection: Bull vs. Bear Margin (|Δ| >= 0.2) │
      │  - 4-Scenario Forward Projection with Temporal Probability Decay  │
      │  - Process Reward Model (PRM Verifier): Step & Coherence Scoring  │
      │  - Activation Gate: Confidence >= 0.85 AND |Δ_adv| >= 0.20        │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │ (Trade Signal Proposal)
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │       Multi-Symbol Matrix & Microstructure Execution Shield       │
      │  - 6-Asset Cross-Correlation: Block in-phase risk (ρ_eff > 0.85)  │
      │  - Cost Ratio Shield: Expected Alpha / Friction >= 3.5            │
      │  - Sniper / Maker Limit Order Execution at Liquidity Pivots       │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │ (Approved Execution Signal)
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │     Fractional Kelly Compounding & Continuous Linear De-leveraging│
      │  - Fractional Kelly f* in [0.15, 0.25] for (1+r)^N Hyper-Growth   │
      │  - Continuous De-leveraging: max(0.10, 1 - DD / 15%) [24/7 trades]│
      │  - Dynamic Sniper A+ Escalation: P(Win) >= 90% when DD > 8%       │
      │  - Broker Lot Step Quantization & Clamping                        │
      └─────────────────────────────────┬─────────────────────────────────┘
                                        │
                                        ▼
      ┌───────────────────────────────────────────────────────────────────┐
      │               MT5 IPC Wire Dispatcher (20 Floats)                 │
      │  - Action (12), Lots, SL, TP, Trailing, Policy Lock == 0          │
      │  - Sub-50ms SLA Execution via Low-Latency IPC Transport           │
      └───────────────────────────────────────────────────────────────────┘
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 4-Hop CoT Neural Reasoning Head | Sequentially evolves cognitive thoughts across regime, liquidity, momentum, action | M2 | R1 |
| 2 | Gated Reasoning Hop | Residual gating module combining past thoughts and incoming observation context | M2 | R1 |
| 3 | Dual Bull-Bear Adversarial Debate | Simultaneous generation of independent Bull and Bear conviction scores | M2 | R1, R8 |
| 4 | 4-Scenario Forward Simulation | Projects market trajectories across Expansion, Reversal, Shakeout, Chop | M2 | R1 |
| 5 | Scenario Temporal Probability Decay | Exponential degradation curves of scenario likelihoods over forward horizon | M2 | R1 |
| 6 | Process Reward Model (PRM Verifier) | Granular validation of step correctness and inter-hop transition coherence | M2 | R1, R8 |
| 7 | Strict 85% Activation Gate | Filters out signals where Calibrated Confidence < 0.85 or Margin < 0.20 | M2 | R1, R8 |
| 8 | SMC Liquidity Sweep Detector | Identifies BSL/SSL stop-hunts and price exhaustion sweeps | M1 | R2 |
| 9 | FVG Imbalance Absorption Retest | Identifies price pullbacks into Fair Value Gaps for passive entry | M1 | R2 |
| 10 | Institutional Order Block (OB) Engine | Detects high-volume displacement origin candles as institutional anchors | M1 | R2 |
| 11 | False Breakout & Trap Neutralizer | Filters out breakout entries lacking prior liquidity sweeps | M1 | R2 |
| 12 | Cross-Asset Macro Quant Engine | Tracks DXY, US10Y, SPX, XAU; computes rolling velocity & returns | M1 | R3 |
| 13 | Global Macro Risk-On/Off Classifier | Synthesizes macro state into Risk-On / Risk-Off score and 32D embedding | M1 | R3 |
| 14 | SGT-V193 xLSTM Core Integration | 17.5M parameter backbone (mLSTM + sLSTM + Graph Transformer) with 233D input | M2 | R4 |
| 15 | JIT / ONNX Sub-50ms Optimization | TorchScript & quantized execution guaranteeing <= 50ms per step | M2 | R4 |
| 16 | Zero NaN / Zero Inf Arithmetic Traps | Strict tensor clamping, log-space gating, nan_to_num, and epsilon safeguards | M2 | AC-7 |
| 17 | 6-Asset Basket Tracker | Manages price streams and buffers across BTC, ETH, SOL, XAU, EUR, US30 (M1/M5) | M3 | R6 |
| 18 | Rolling Pearson Correlation Matrix | Computes symmetric 6x6 correlation matrix across supported symbols | M3 | R6 |
| 19 | In-Phase Correlation Shield | Rejects candidate trades if effective correlation with active position > 0.85 | M3 | R6 |
| 20 | Multi-Timeframe M5 Confluence Scorer | Boosts trend-aligned signals (1.20x) and penalizes counter-trend (0.75x) | M3 | R6 |
| 21 | High-Frequency Quota & Rate Pacer | Monitors 24h execution rate against 45-60 daily target (>4,000 in 90d) | M3 | R6 |
| 22 | Multi-Symbol Batch Router Arbitration | Selects top-scoring signal in a correlated candidate cluster | M3 | R6 |
| 23 | Institutional Cost Ratio Shield | Enforces Expected Alpha / (Spread + Commission + Slippage) >= 3.5 | M3 | R7 |
| 24 | Dynamic Adverse Slippage Model | Estimates taker slippage based on volatility, order volume, and book imbalance | M3 | R7 |
| 25 | Sniper / Maker Limit Entry Mode | Converts taker orders to maker limit orders at liquidity pivots (0 slippage) | M3 | R7 |
| 26 | Order Book Imbalance (OBI) & Micro-Price | Volume-weighted order book pressure and micro-price computation | M3 | R7 |
| 27 | Full & Fractional Kelly Calculator | Computes optimal geometric growth fraction f* in [0.15, 0.25] | M3 | R5 |
| 28 | Continuous Linear De-leveraging | Scales risk down linearly to 10% floor as drawdown approaches 15% (24/7 active)| M3 | R5 |
| 29 | Dynamic Sniper A+ Admission Hurdle | Escalates required P(Win) from 85% to 90% when DD > 8% | M3 | R5 |
| 30 | MT5 Broker Lot Size Sizer | Calculates and quantizes lot sizes according to broker lot step | M3 | R5 |
| 31 | Unified Reasoning Trader Pipeline | Integrates R1-R8 into unified runtime runner emitting 20-float wire packets | M4 | R1-R8 |
| 32 | Headless MT5 IPC Transport Bridge | Low-latency binary IPC bridge communicating with MetaTrader 5 | M4 | R4 |
| 33 | Zero Premature Closes Invariant | Locks POLICY_CLOSE_ALL == 0 for natural trade progression (SL/TP/Runner) | M4 | R4 |
| 34 | 3-Month Multi-Cycle Walk-Forward Replay | Backtests 90 days across 6 symbols with realistic spreads, comms, slippage | M5 | AC-1-7 |
| 35 | 10,000-Run Monte Carlo Risk Verifier | Statistically validates 50x hyper-growth, PF >= 3.0, WR >= 85%, DD < 15% | M5 | AC-1-5 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | SMC/ICT Liquidity & Macro Reasoning | Deep SMC/ICT Engine (sweeps, FVG, OB, traps) + Cross-Asset Macro Engine (DXY, US10Y, SPX, XAU, Risk-On/Off) | None | IN_PROGRESS |
| M2 | xLSTM 17.5M Core & Metacognitive Reasoning | 233D Feature Ingress, SGT-V193 Core, 4-Hop CoT Head, Bull-Bear Debate, PRM Verifier, JIT <=50ms & 0 NaN/Inf | M1 | PLANNED |
| M3 | Multi-Asset Matrix, Cost Shield & Kelly Compounding | 6-Asset M1/M5 Matrix, In-Phase Shield (ρ <= 0.85), Cost Shield (>=3.5), Sniper Limit Entry, Fractional Kelly & Linear De-leveraging | M2 | PLANNED |
| M4 | Unified Pipeline Integration & MT5 IPC Transport | UnifiedReasoningTrader pipeline, 20-float wire packing, POLICY_CLOSE_ALL == 0, head-to-tail integration | M1, M2, M3 | PLANNED |
| M5 | 3-Month Walk-Forward Benchmark & Final E2E Verification | 90-day multi-asset backtest (>4,000 trades), 10,000-run Monte Carlo, AC-1 to AC-7 validation, Adversarial Hardening | M4 | PLANNED |

## Interface Contracts
### Data Ingestion ↔ SMC / Macro Feature Extractor
- Ingress: 6 symbols (BTC, ETH, SOL, XAU, EUR, US30) M1/M5 OHLCV bar buffers `[6, 2, 60, 5]` + Macro quotes `[4, 60]`
- Output: `SMCFeatures [64 floats]`, `MacroFeatures [32 floats]`
- Invariant: Zero NaN / Inf; values clamped to finite ranges.

### Feature Assembly ↔ SGT-V193 xLSTM Core
- Ingress: `Observation [233 floats]` = 137 Base + 64 SMC + 32 Macro
- Output: `LatentState [768 floats]` (Governor agent state from Relational Graph Transformer)
- Invariant: Recurrent memory isolation between symbols; log-space exponential gating with stabilization $m_t \in [-30, 30]$.

### SGT-V193 Latent State ↔ Metacognitive Reasoning & PRM
- Ingress: `LatentState [768 floats]`
- Output: `ReasoningVerdict(action: int in {-1, 0, 1}, confidence: float, prm_score: float, bull_score: float, bear_score: float, margin: float)`
- Invariant: Activated iff `confidence >= 0.85` AND `|margin| >= 0.20`. Otherwise `action = 0` (HOLD).

### Reasoning Verdict ↔ Multi-Symbol Matrix & Microstructure Shield
- Ingress: `ReasoningVerdict`, `OrderBookSnapshot`, `RollingCorrelationMatrix`
- Output: `ExecutionVerdict(approved: bool, entry_mode: str, limit_price: float, expected_alpha: float, cost_ratio: float)`
- Invariant: Rejects if effective in-phase correlation $\rho_{\text{eff}} > 0.85$ or $\text{CostRatio} < 3.5$.

### Execution Verdict ↔ Fractional Kelly Sizing Engine
- Ingress: `ExecutionVerdict`, `CurrentEquity`, `PeakEquity`, `WinProbability`, `PayoffRatio`, `SLPoints`
- Output: `TradeSizingDecision(lot_size: float, risk_pct: float, deleveraging_factor: float, in_sniper_mode: bool)`
- Invariant: `deleveraging_factor >= 0.10` (10% continuous risk floor); requires $P(\text{Win}) \ge 90\%$ when $\text{DD} > 8\%$; lot clamped to `[min_lot, max_lot]`.

### Sizing Decision ↔ MT5 IPC Transport
- Ingress: Action, quantized lot size, SL/TP levels, trailing parameters
- Output: Packed 20-float wire buffer
- Invariant: `POLICY_CLOSE_ALL == 0` locked.

## Code Layout
- `research_reasoning_world_class/src/`:
  - `smc_liquidity_engine.py`: SMC/ICT BSL/SSL sweeps, FVG imbalances, Order Blocks, false breakout traps (M1).
  - `macro_reasoning_engine.py`: Cross-Asset Macro quantitative tracking & Risk-On/Off embedding (M1).
  - `metacognitive_reasoning.py`: CoT 4-hop reasoning, Bull-Bear debate, PRM verifier, temporal decay (M2).
  - `xlstm_reasoning_adapter.py`: Ingress 233D projection, SGT-V193 xLSTM binding, TorchScript JIT compiler (M2).
  - `multi_symbol_matrix.py`: 6-symbol tracking, correlation matrix, in-phase shield, M5 confluence (M3).
  - `microstructure_execution.py`: OBI, micro-price, cost ratio shield (>=3.5), Sniper Limit entry (M3).
  - `dynamic_kelly_compounding.py`: Kelly math, continuous linear de-leveraging, Sniper A+, lot sizer, Monte Carlo (M3).
  - `unified_reasoning_pipeline.py`: Top-level orchestrating runner connecting all modules into MT5 IPC bridge (M4).
  - `mt5_ipc_transport.py`: Binary shared memory / pipe IPC transport layer (M4).
- `research_reasoning_world_class/tests/`:
  - `test_smc_liquidity_engine.py` (M1)
  - `test_macro_reasoning_engine.py` (M1)
  - `test_metacognitive_reasoning.py` (M2)
  - `test_xlstm_inference_latency.py` (M2)
  - `test_multi_symbol_microstructure.py` (M3)
  - `test_dynamic_kelly_compounding.py` (M3)
  - `test_unified_reasoning_pipeline_e2e.py` (M4)
  - `test_walkforward_benchmark_validation.py` (M5)
  - `test_e2e_opaque_box.py` (E2E Testing Track)
