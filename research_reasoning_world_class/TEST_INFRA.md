# Institutional Test Infrastructure & 4-Tier Test Architecture
## Project "Mythos của Claude" — xLSTM 17.5M World-Class Quantitative Reasoning Engine

---

### 1. Architectural Philosophy & Opaque-Box Methodology

The testing infrastructure for the **Mythos của Claude** quantitative reasoning engine is constructed under an institutional-grade, **Opaque-Box (Requirement-Driven)** paradigm. Rather than coupling test logic to volatile private implementation details, every test case across all 4 tiers is directly derived from mathematical specifications and formal acceptance criteria outlined in:
- `ORIGINAL_REQUEST.md` (R1 - R8, AC-1 - AC-7)
- `PROJECT.md` & `PROJECT_MYTHOS_SPEC.md`
- MetaTrader 5 (MT5) Institutional Wire Contract & High-Frequency Latency Budgets

#### Core Testing Principles:
1. **Opaque-Box Independence**: Test suites treat each subsystem as an opaque black/grey box, communicating exclusively through public API contracts, typed dataclasses, and standard mathematical transformations.
2. **Explicit Expected Output Derivation**: Every assertion tests a mathematically provable expectation (e.g., Kelly formula $f^* = \frac{p(b+1)-1}{b}$, linear continuous de-leveraging $\text{factor} = \max(0.1, 1 - \frac{\text{DD}}{0.15})$, Pearson correlation $\rho > 0.85$, cost ratio $\frac{\text{Alpha}}{\text{Friction}} \ge 3.5$, confidence $\ge 0.85$, and margin $|\Delta| \ge 0.20$).
3. **Zero Facade Tests**: Facade tests that assert trivialities or tautologies (e.g. `assert True`) are strictly prohibited. Every test must exercise real algorithms, multi-stage pipelines, and real tensor arithmetic.
4. **Adversarial & Numerical Invariance (AC-7)**: Systems operating in quantitative high-frequency environments must never yield `NaN`, `+Inf`, `-Inf`, division-by-zero, or unhandled exceptions under extreme market stress, missing order books, zero liquidity, or broken feeds.
5. **Real-Time Execution SLA (AC-6)**: The decision-making latency across the multi-hop reasoning, matrix arbitration, cost shielding, and lot sizing pipeline must strictly fulfill the $\le 50\text{ms}$ MetaTrader 5 execution window.

---

### 2. The 4-Tier Test Case Design Matrix

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       4-TIER TEST CASE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Feature Coverage (>=5 tests per core feature)                       │
│  - R1: Metacognitive CoT, Bull/Bear Debate, Scenarios, PRM Verification     │
│  - R5: Fractional Kelly Compounding, Continuous Risk De-leveraging          │
│  - R6: 6-Asset Basket Matrix, Rolling Correlation, M5 Confluence, Pacing    │
│  - R7: Institutional Cost Ratio Shield (>=3.5), OBI & Sniper Limit Entry   │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary & Corner Cases (>=5 tests per core feature)                │
│  - Empty / Null Ingress: Empty buffers, missing order book, zero returns    │
│  - Extreme / Max Stress: Extreme equity ($100M), massive volatility, 99% DD│
│  - Zero / Floor Bounds: Zero spreads, zero volume, zero edge, zero friction │
│  - NaN / Inf Safeguards: Corrupted tensors, inf prices, NaN probabilities  │
│  - Boundary Threshold Transitions: DD=8%, DD=15%, Rho=0.85, Ratio=3.5     │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Cross-Feature Interactions & Multi-Module Synergies                 │
│  - Interaction 1: Dynamic Kelly + Drawdown Escalation (Continuous Delev)    │
│  - Interaction 2: Correlation Filter + Microstructure Execution Shield      │
│  - Interaction 3: Sniper / Maker Limit Order + Kelly Sizing Integration     │
│  - Interaction 4: Metacognitive CoT + Multi-Symbol Matrix Arbitration       │
│  - Interaction 5: Macro/SMC Regime Inversion + Adversarial Debate Gating    │
├─────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Real-World Application Scenarios (Full 3-Month Simulation)           │
│  - Realistic 90-day walk-forward multi-asset replay across 6 instruments    │
│  - 4,000+ Closed Trades under realistic spreads, commissions, and slippage │
│  - Verification of AC-1 (>=50x growth), AC-2 (PF >= 3.0), AC-3 (WR >= 85%), │
│    AC-4 (N >= 4,000), AC-5 (DD < 15%), AC-6 (<= 50ms), AC-7 (0 NaN / Inf)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Detailed Specification of Test Tiers

#### Tier 1: Feature Coverage (Unit & Subsystem Invariants)
- **R1: Metacognitive Reasoning**:
  1. *4-Hop Chain-of-Thought Sequential Latent Evolution*: Verifies that tensor states evolve across regime, liquidity, momentum, and action hops.
  2. *Dual Bull-Bear Debate Independence*: Verifies simultaneous, non-trivial bull and bear scores and margin divergence calculation.
  3. *4-Scenario Forward Projection*: Validates forward projections across Expansion, Reversal, Shakeout, and Chop.
  4. *Temporal Probability Decay Dynamics*: Proves exponential probability decay over 12 forward time steps.
  5. *Process Reward Model (PRM) Verifier & 85% Activation Gate*: Enforces confidence $\ge 0.85$ and margin $\ge 0.20$ for buy/sell actions; otherwise holds.
- **R5: Fractional Kelly & Continuous Risk Scaling**:
  1. *Full Kelly Closed-Form Mathematical Exactness*: Tests $K^*(p, b) = \frac{p(b+1)-1}{b}$ against analytical solutions.
  2. *Fractional Kelly Scaling Factor*: Verifies $f^* \in [0.15, 0.25]$ clamping.
  3. *Continuous Linear De-leveraging*: Verifies $\text{factor} = \max(0.10, 1 - \frac{\text{DD}}{0.15})$ continuous scaling without freezing trades.
  4. *Dynamic Sniper A+ Threshold Escalation*: Verifies required $P(\text{Win})$ jumps from 85% to 90% when $\text{DD} > 8\%$.
  5. *Exponential Lot Size Compounding & Broker Lot Quantization*: Validates $(1+r)^N$ lot scaling clamped to broker steps and lot limits.
- **R6: Multi-Symbol & Multi-Timeframe High-Frequency Matrix**:
  1. *6-Asset Basket Registry & Asset Classification*: Tests coverage of BTCUSD, ETHUSD, SOLUSD, XAUUSD, EURUSD, US30.
  2. *Rolling 6x6 Pearson Correlation Matrix Computation*: Verifies symmetric correlation matrix with unit diagonal.
  3. *In-Phase Correlation Filter ($\rho > 0.85$)*: Rejects candidate trades conflicting with existing active positions.
  4. *Multi-Timeframe M5 Confluence Scorer*: Confirms 1.20x boost for aligned trend and 0.75x penalty for counter-trend.
  5. *24/7 Execution Pacing & Quota Monitor*: Monitors 45-60 daily trade pace to achieve $>4,000$ trades over 90 days.
- **R7: Microstructure Execution Shield & Cost Control**:
  1. *Cost Ratio Computation*: Validates $\frac{\text{Expected Alpha}}{\text{Spread} + \text{Commission} + \text{Slippage}} \ge 3.5$.
  2. *Order Book Imbalance (OBI) & Micro-Price Formula*: Verifies volume-weighted pressure and micro-price.
  3. *Dynamic Adverse Slippage Taker Model*: Tests slippage expansion under adverse OBI and volume scaling.
  4. *Sniper / Maker Limit Entry at Liquidity Pivots*: Verifies passive limit fills eradicating taker slippage.
  5. *Cost Profile Registry for All 6 Assets*: Verifies cost configurations across crypto, metals, fx, and indices.

#### Tier 2: Boundary & Corner Cases (Adversarial Robustness)
- **Empty / Null / Zero Conditions**:
  - Empty price buffers, zero variance sequences, zero volume bars.
  - Zero account equity, zero stop-loss points, zero expected alpha, zero friction.
- **Extreme / Saturated Values**:
  - Ultra-high equity ($100,000,000 USD) ensuring max lot bounding.
  - Extreme Drawdown (99.9%) ensuring de-leveraging floor ($0.10$) holds.
  - High market volatility factors ($10.0\times$) inflating slippage.
- **Numerical Corruption Safeguards (AC-7)**:
  - Input vectors saturated with `float('nan')`, `float('inf')`, `-float('inf')`.
  - Zero NaN or Inf leakage into decisions, output lot sizes, or risk parameters.
- **Exact Boundary Transitions**:
  - Drawdown at exactly 0.080000: tests transition into Sniper A+ mode.
  - Drawdown at exactly 0.150000: tests de-leveraging factor reaching exactly 0.10 floor.
  - Correlation at exactly 0.850000: verifies boundary rejection logic.
  - Cost ratio at exactly 3.500000: verifies admission boundary.
  - Metacognitive confidence at exactly 0.850000 and margin at 0.200000.

#### Tier 3: Cross-Feature Interactions & Synergies
- **Interaction 1: Dynamic Kelly + Drawdown Escalation (Continuous De-leveraging)**:
  - Account undergoes simulated drawdown from 0% to 12%.
  - Verifies continuous smooth lot reduction while requiring $P(\text{Win}) \ge 90\%$.
- **Interaction 2: Correlation Filter + Microstructure Execution Shield**:
  - A signal passes the institutional cost ratio ($\ge 3.5$), but is rejected by the multi-symbol matrix due to high in-phase correlation with an active position.
- **Interaction 3: Microstructure Sniper Entry + Kelly Sizing Integration**:
  - A trade failing market taker execution due to high spread/slippage is salvaged by the Sniper/Maker limit engine, reducing friction, lifting cost ratio $>3.5$, and enabling Kelly position sizing.
- **Interaction 4: Metacognitive Reasoning + Multi-Symbol Batch Router Arbitration**:
  - Multiple simultaneous signals generated across crypto and metals; the batch router selects the highest-scoring candidate and filters correlated companions.
- **Interaction 5: Macro/SMC Regime Inversion + Adversarial Debate Gating**:
  - Adverse market signals generate balanced Bull/Bear debate ($|\Delta| < 0.20$), resulting in an automatic HOLD, preserving capital.

#### Tier 4: Real-World Application Scenarios (Comprehensive 3-Month Simulation)
- **Realistic 90-Day Multi-Asset Walk-Forward Replay**:
  - Models realistic market sequences across 6 assets: BTCUSD, ETHUSD, SOLUSD, XAUUSD, EURUSD, US30.
  - Simulates 4,000+ realistic trade executions across M1/M5 regimes.
  - Measures all 7 Acceptance Criteria simultaneously:
    * **AC-1**: Net Account Growth $\ge 50\times$ (Starting capital $\$10,000 \to \ge \$500,000$).
    * **AC-2**: Profit Factor $\ge 3.0$.
    * **AC-3**: Win Rate $\ge 85.0\%$.
    * **AC-4**: Closed Trade Count $N \ge 4,000$.
    * **AC-5**: Maximum Relative Drawdown $< 15.0\%$.
    * **AC-6**: Decision Latency $\le 50\text{ms}$ per decision.
    * **AC-7**: Mathematical preservation: 0 NaN, 0 Inf across all states.

---

### 4. Test Execution & Verification Commands

```bash
# Execute complete Opaque-Box E2E Test Suite:
pytest research_reasoning_world_class/tests/test_e2e_opaque_box.py -v

# Execute specific tiers:
pytest research_reasoning_world_class/tests/test_e2e_opaque_box.py -k "Tier1" -v
pytest research_reasoning_world_class/tests/test_e2e_opaque_box.py -k "Tier2" -v
pytest research_reasoning_world_class/tests/test_e2e_opaque_box.py -k "Tier3" -v
pytest research_reasoning_world_class/tests/test_e2e_opaque_box.py -k "Tier4" -v

# Run entire repository test suite with timing:
pytest research_reasoning_world_class/tests/ -v --durations=10
```
