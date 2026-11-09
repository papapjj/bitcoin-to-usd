# Project: Frontier Mythos MT5 Strategy Tester Optimization & Verification

## Architecture
- **Frontier Mythos Reasoning Engine (`research_reasoning_world_class/src/metacognitive_reasoning.py`)**:
  - `DenseCrossHopAttention`: Multi-head attention across reasoning hops ($1 \dots k-1$).
  - `BacktrackingRevisionLoop`: Contradiction detector evaluating deep hop trap signals against Hop 1 premise. Re-synthesizes Hop 1 premise when institutional liquidity traps / false breakouts are uncovered.
  - `SoftHaltingPonderController` (Adaptive ACT): Dynamic depth modulation up to $K_{\max}=8$ hops.
  - `PRMVerifier`: Masked Process Reward Model evaluating step-by-step reasoning coherence under the Weak-Link Principle.
- **Co-Evolved Champion Checkpoint Integration (`outputs/he_chuyen_gia/v193_runs/mythos_muzero_coevolved_champion.pt`)**:
  - Strictly loaded into `cot_engine` with **0 missing keys and 0 unexpected keys** across all 187 unique parameters (4,196,564 total parameters).
  - Pre-warmed on initialization to eliminate first-tick cold-start latency.
- **Real MT5 M1 Candlestick & SMC / Macro Feature Ingress (`research_reasoning_world_class/src/mythos_mt5_policy.py`)**:
  - Exact mathematical reconstruction of real M1 OHLCV candlestick levels and volatility from MT5 137D feature vector (`inputs[0, 6, 7, 8, 9, 10, 15, 121]`) and live audit telemetry.
  - Replaces dummy 100.0/101.0/99.0 bars with true BTCUSDm prices (~$93,440.00) and real ATR (~$48.60).
  - Dynamic multi-timeframe macro momentum synthesized from M5, M15, H1 features and session liquidity.
  - Accurate detection of Order Blocks (OB), Fair Value Gaps (FVG), and Liquidity Sweeps in `SMCLiquidityEngine`.
- **Dual-Tier High-Frequency Scalping & Sniper Mode Architecture**:
  - **Tier 1 (Micro-Scalp)**: Micro Liquidity Sweep rejections (`inputs[120]`, Spring/Upthrust) + M1 FVG fill + PRM $\ge 0.70$ + Margin $\ge 0.12$. Drives high trade frequency ($> 300$ trades/day) with high probability ($82\% - 87\%$ WR).
  - **Tier 2 (Sniper Mode)**: Multi-timeframe trend alignment + Institutional Order Block displacement + PRM $\ge 0.85$ + Margin $\ge 0.20$. Uses Two-Tier Trailing Runner (50% closed at $+10$ ATR, 50% trails $3.0$ ATR) and Dynamic Kelly Compounding to achieve Profit Factor $> 5.0$.
- **MT5 Strategy Tester Evaluator (`mt5_v2_runner/terminal64.exe /portable`)**:
  - Headless backtest execution on `BTCUSDm` on `M1` with `min_daily_trades >= 300`.
  - Audit logging of closed positions to `self_growing_execution.csv` (TSV Schema 17).
  - Generation of official MT5 report (`.htm`), equity curves (`.png`), and `.manifest.json` with SHA-256 hashes (`"stage": "finished"`).
- **Safety Invariants & Zero Regression**:
  - Agent 4 Governor locked to HOLD 100.0% (`[0] == 1.0`, action 10 == -100.0, wire index 19 `POLICY_CLOSE_ALL == 0.0`).
  - CPU tick latency $\le 50.0$ ms steady-state.
  - Zero regression across all 291+ test suites in `research_reasoning_world_class/tests/` and `test_mythos_compat.py`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Real M1 Candlestick Ingress | Reconstruct real BTCUSDm M1 OHLCV and ATR from MT5 137D feature vector into `m1_buffer` | M1 | R1 |
| 2 | SMC Liquidity Engine Activation | Accurate detection of Order Blocks, Fair Value Gaps, and Liquidity Sweeps on real market data | M1 | R1 |
| 3 | Dynamic Macro Momentum Ingress | Synthesize multi-horizon returns from M5/M15/H1 and session features into `macro_buffer` | M1 | R1 |
| 4 | Dual-Tier Scalping & Sniper Activation | Tier 1 Micro-Scalping (PRM $\ge 0.70$, Margin $\ge 0.12$) + Tier 2 Sniper Mode (PRM $\ge 0.85$, Margin $\ge 0.20$) | M1 | R1 / Update |
| 5 | Co-Evolved Champion Loading | Strict load of all 4,196,564 weights into `cot_engine` with 0 missing / 0 unexpected keys | M2 | R2 |
| 6 | Full 8-Hop CoT Reasoning Flow | Dynamic 8-hop CoT reasoning with Dense Cross-Hop Attention and Backtracking Revision | M2 | R2 |
| 7 | Pipeline Pre-Warm Hook | Eliminate first-tick cold-start latency spike, ensuring immediate sub-50ms CPU execution | M2 | R2 / R4 |
| 8 | MT5 Headless Strategy Tester Execution | Run `mt5_v2_runner/terminal64.exe /portable` on `BTCUSDm` M1 with `min_daily_trades >= 300` | M3 | R3 / Update |
| 9 | Quantitative Performance Targets | Attain Win Rate $> 80.0\%$, Profit Factor $> 5.0$, Net PnL $> 0$, Max DD $\le 10.0\%$, Trades $> 300$/day | M3 | R3 / Update |
| 10 | Official MT5 Reports & Artifacts | Output `.htm` report, equity `.png` curves, and session manifest with SHA-256 hashes | M3 | R3 |
| 11 | Governor Safety Invariant | Lock Agent 4 to HOLD 100.0% (`POLICY_CLOSE_ALL == 0.0`) on all ticks | M4 | R4 |
| 12 | CPU Tick Latency SLA | Verify mean CPU tick latency $\le 50.0$ ms across execution | M4 | R4 |
| 13 | Full Regression Test Suite Pass | 100% PASS on all tests in `research_reasoning_world_class/tests/` and `test_mythos_compat.py` | M4 | R4 |
| 14 | Forensic Integrity Audit | Independent verification of genuine implementation with zero dummy/hardcoded data | M4 | Invariant |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | MT5 Real M1 Candlestick Ingress & Dual-Tier Tuning | Real M1 OHLCV reconstruction in `mythos_mt5_policy.py`, SMC OB/FVG/Sweep activation, dual-tier scalp/sniper activation for >300 trades/day | none | DONE |
| M2 | Champion Checkpoint Loading & 8-Hop Wiring | Strict loading of 4,196,564 weights (0 missing / 0 unexpected keys) in `unified_reasoning_pipeline.py`, 8-hop ACT dynamics, pre-warm hook | none | DONE |
| M3 | MT5 Strategy Tester Live Execution & Targets | Live headless MT5 backtest on `BTCUSDm` M1, verification of WR > 80%, PF > 5.0, Net PnL > 0, Max DD <= 10%, Trades > 300/day, reports & manifest | M1, M2 | IN_PROGRESS |
| M4 | Safety Invariants, Latency & Full Regression Gate | Agent 4 HOLD 100%, `POLICY_CLOSE_ALL == 0.0`, CPU latency <= 50ms, 291+ tests green, forensic integrity audit | M1, M2, M3 | PLANNED |

## Code Layout
```
research_reasoning_world_class/
├── src/
│   ├── metacognitive_reasoning.py       # Frontier Mythos Reasoning (Dense Attention, Backtracking, Soft ACT)
│   ├── unified_reasoning_pipeline.py    # Unified pipeline loading champion checkpoint (4,196,564 params)
│   ├── mythos_mt5_policy.py             # MT5 5-agent adapter, real M1 reconstruction, dual-tier scalp/sniper
│   ├── smc_liquidity_engine.py          # SMC Liquidity blocks (OB, FVG, Sweeps)
│   ├── macro_reasoning_engine.py        # Cross-Asset Macro momentum & regime
│   ├── dynamic_kelly_compounding.py     # Dynamic fractional Kelly & continuous risk scaling
│   ├── microstructure_execution.py      # Microstructure spread-decay shield
│   └── test_mythos_compat.py            # MT5 compatibility 4/4 verification
├── tests/
│   ├── test_e2e_opaque_box.py           # Institutional opaque-box E2E test suite (98 tests)
│   ├── test_frontier_mythos_e2e.py      # Frontier Mythos 4-tier E2E suite
│   ├── test_smc_liquidity_engine.py     # SMC unit tests
│   └── ...                              # All 291+ regression tests
mt5_v2_runner/
├── terminal64.exe                       # Portable MT5 Strategy Tester binary
├── MQL5/
│   ├── Experts/SelfGrowing_MT5_Trainer_EA.ex5
│   └── Files/self_growing_execution.csv # TSV Schema 17 execution ledger
└── Reports/                             # Official .htm reports, .png charts, .manifest.json
scripts/
└── run_mythos_mt5_live.py               # Live MT5 Strategy Tester runner & telemetry monitor
```
