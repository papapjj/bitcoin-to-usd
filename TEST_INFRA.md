# Institutional Test Infrastructure & 4-Tier Test Architecture
## Project: Multi-Asset Walk-Forward Self-Play Triad (SPIN + MAD Reflexion + STaR)

---

### 1. Architectural Philosophy & Opaque-Box Methodology

The testing infrastructure for the **Multi-Asset Walk-Forward Self-Play Triad** quantitative reasoning and trading engine is constructed under an institutional-grade, **Opaque-Box (Requirement-Driven)** verification paradigm. Rather than coupling test logic to volatile private implementation internals, every test case across all 4 tiers is directly derived from mathematical specifications and formal acceptance criteria outlined in:
- `ORIGINAL_REQUEST.md` (Section `## 2026-10-26T04:40:47Z`, Requirements R1 – R4, Quantitative Multi-Asset Performance)
- `PROJECT.md` (Architecture, Feature Inventory 1–16, Interface Contracts, Milestones M1 – M4)
- MetaTrader 5 (MT5) Institutional Wire Contract & High-Frequency Latency Budgets ($\le 50\text{ms}$ SLA)

#### Core Testing Principles:
1. **Opaque-Box Independence**: Test suites treat each subsystem as an opaque black/grey box, communicating exclusively through public API contracts, typed dataclasses, and standard mathematical transformations.
2. **Explicit Expected Output Derivation**: Every assertion tests a mathematically provable expectation:
   - **Process Reward Model (PRM)**: Score $\ge 0.90$ with step validity and transition coherence.
   - **Metacognitive Confidence**: Entropy-discounted confidence $\ge 0.85$ with multi-scenario certainty factor.
   - **Adversarial Polarized Margin**: $|p_{\text{bull}} - p_{\text{bear}}| \ge 0.20$ separating decisive entries from indecision zones.
   - **Strict Zero-Sum Payoff (Minimax)**: $R_{\text{adversary}} = -R_{\text{challenger}}$ with differentiable inactivity penalty $w_{\text{inact}} = 2.0$ penalizing mutual passivity ($zs_c + zs_{\text{adv}} = -4.0$).
   - **Governor HOLD Invariant**: Agent 4 locked 100.0% HOLD (`selected_action[0] == 1.0`, `selected_action[10] == -100.0`), wire index 19 `POLICY_CLOSE_ALL == 0.0`.
   - **Single-Tick Latency SLA**: Steady-state CPU execution latency strictly $\le 50.0\text{ms}$.
   - **Trajectory Schema 17**: 63-column TSV ledger parsing, incremental file synchronization, and closed deal extraction.
   - **12-Week Rolling Schedule**: Discrete partition of 2024.10.01 – 2025.01.01 across BTCUSDm, ETHUSDm, and XAUUSDm with champion promotion lineage.
3. **Zero Facade Tests**: Facade tests that assert trivialities or tautologies (e.g. `assert True`) are strictly prohibited. Every test exercises real neural layers, debate graphs, STaR rationalization loops, and risk compounding arithmetic.
4. **Adversarial & Numerical Invariance**: Systems operating in high-frequency trading must never produce `NaN`, `+Inf`, `-Inf`, division-by-zero, or unhandled exceptions under extreme market stress, missing order books, zero liquidity, or broken feeds.
5. **Real-Time Execution SLA**: The end-to-end inference latency across ingress, reasoning, correlation matrix arbitration, and MT5 wire packet dispatch must strictly satisfy the $\le 50.0\text{ms}$ SLA on CPU.

---

### 2. The 4-Tier Test Case Design Matrix

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                            4-TIER TEST CASE ARCHITECTURE                                    │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tier 1: Feature Coverage (>=5 test cases per core feature, 40 tests)                        │
│  - 1.1: PRM Score >= 0.90 (Calculation, Rejection, Acceptance, Weak-Link, Boundary)        │
│  - 1.2: Confidence >= 0.85 (Entropy Calibration, Rejection, Acceptance, Collapse, Exactness)│
│  - 1.3: Adversarial Margin >= 0.20 (Polarization, Indecision Rejection, BUY, SELL, Boundary)│
│  - 1.4: Anti-Collusion Zero-Sum Payoff (Zero-Sum Invariant, Inactivity Penalty, Divergence) │
│  - 1.5: Governor HOLD Invariant (Agent 4 Action, Wire Index 19, Random Noise, 100 Ticks)    │
│  - 1.6: Single-Tick Latency SLA <= 50ms (P50 SLA, Mean SLA, Pre-Warm, Regimes, Heap Check) │
│  - 1.7: Trajectory Schema 17 TSV (63 Columns, Winning Deal, Losing Deal, File Pointer, Crash)│
│  - 1.8: 12-Week Rolling Schedule (12 Weeks, Chronological Order, Multi-Asset Basket, Linage)│
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tier 2: Boundary & Corner Cases (6 tests)                                                   │
│  - 2.1: Flash Crash & Volatility Shock (-15% instant shock, Governor HOLD preserved)       │
│  - 2.2: Zero Liquidity & Frozen Order Book (Zero depth / tiny alpha -> Microstructure Veto) │
│  - 2.3: Spread Explosion & Friction Gate (50 ATR spread -> Cost Ratio < 3.5 -> Veto)       │
│  - 2.4: Extreme Drawdowns & Continuous Kelly Deleveraging (90% DD -> 0.10 Floor)            │
│  - 2.5: Mutual HOLD Inactivity Penalty Enforcement (Both models HOLD -> w_inact = 2.0)     │
│  - 2.6: Missing Bars & Corrupted Feed (NaN/Inf injection cleaned via nan_to_num guard)     │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tier 3: Cross-Feature Combinations (5 tests)                                                │
│  - 3.1: High Correlation + High PRM Rejection (rho_eff > 0.85 blocks correlated asset)      │
│  - 3.2: Volatility Shock + Two-Tier Trailing Runner + Locked Invariant (TP 10 ATR, SL +1.0)│
│  - 3.3: Negative MT5 Fill + STaR Rationalization + SPIN Preference Update                  │
│  - 3.4: MAD Reflexion Memory Trap Veto + Counter-Trap Sniper Inversion (sim > 0.88 veto)    │
│  - 3.5: Dynamic Kelly Continuous Scaling + Microstructure Cost Ratio Synergy               │
├─────────────────────────────────────────────────────────────────────────────────────────────┤
│ Tier 4: Real-World Walk-Forward Scenarios (4 tests)                                         │
│  - 4.1: Multi-Week Simulated Market Regime Shifts (Trending -> Sideway -> Sweep -> Breakout)│
│  - 4.2: High-Frequency 300+ Trades/Day Pacing (WR > 80.0%, PF > 5.0, Max DD <= 10.0%)      │
│  - 4.3: 5-10 Epochs Bilateral Co-Evolution Parameter Convergence (Non-zero weight updates)  │
│  - 4.4: 12-Week Rolling Deliverables & Forensic Integrity Audit (HTML, SHA-256 Manifest)   │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### 3. Detailed Specification of Test Tiers

#### Tier 1: Feature Coverage (40 Test Cases)
- **1.1: Process Reward Model (PRM $\ge 0.90$)**:
  1. `test_tier1_1_prm_calculation_and_step_scoring`: Verifies step validity and transition coherence scoring.
  2. `test_tier1_1_prm_rejection_below_threshold`: Verifies PRM $< 0.90$ forces HOLD (`is_activated = False`).
  3. `test_tier1_1_prm_acceptance_above_threshold`: Verifies PRM $\ge 0.90$ unlocks execution eligibility.
  4. `test_tier1_1_prm_weak_link_principle`: Verifies bottleneck hop suppresses aggregate reward ($prm \le mean$).
  5. `test_tier1_1_prm_boundary_threshold_sensitivity`: Verifies boundary sensitivity at PRM $= 0.9000$.

- **1.2: Metacognitive Confidence ($\ge 0.85$)**:
  1. `test_tier1_2_conf_entropy_discounted_derivation`: Verifies entropy-discounted confidence calibration.
  2. `test_tier1_2_conf_rejection_below_threshold`: Verifies confidence $< 0.85$ results in HOLD.
  3. `test_tier1_2_conf_acceptance_above_threshold`: Verifies confidence $\ge 0.85$ permits activation.
  4. `test_tier1_2_conf_scenario_disagreement_collapse`: Verifies uniform scenario distribution forces certainty factor $= 0.90$.
  5. `test_tier1_2_conf_boundary_exactness`: Verifies threshold precision at $0.8500$.

- **1.3: Adversarial Polarized Margin ($|\text{Margin}| \ge 0.20$)**:
  1. `test_tier1_3_margin_polarization_calculation`: Verifies Bull - Bear margin difference.
  2. `test_tier1_3_margin_indecision_zone_rejection`: Verifies indecision zone ($|\text{Margin}| < 0.20$) forces HOLD.
  3. `test_tier1_3_margin_strong_bull_triggers_buy`: Verifies Margin $\ge +0.20$ selects BUY (action $= 1$).
  4. `test_tier1_3_margin_strong_bear_triggers_sell`: Verifies Margin $\le -0.20$ selects SELL (action $= 2$).
  5. `test_tier1_3_margin_boundary_threshold_precision`: Verifies threshold boundary at $|\text{Margin}| = 0.2000$.

- **1.4: Non-Cooperative Anti-Collusion Zero-Sum Payoff**:
  1. `test_tier1_4_anticollusion_zero_sum_mathematical_relation`: Verifies $R_{\text{c}} + R_{\text{adv}} == 0.0$ on active trades.
  2. `test_tier1_4_anticollusion_mutual_inactivity_penalty`: Verifies $P_{\text{inactivity}} = 2.0$ penalizes mutual passivity ($zs_c + zs_{\text{adv}} == -4.0$).
  3. `test_tier1_4_anticollusion_unilateral_advantage`: Verifies active trader earns edge while idle player incurs penalty.
  4. `test_tier1_4_anticollusion_bilateral_gradients_non_zero`: Verifies non-zero gradients for both models.
  5. `test_tier1_4_anticollusion_arms_race_divergence`: Verifies bilateral updates drive trade exploration.

- **1.5: Governor HOLD Invariant**:
  1. `test_tier1_5_governor_action_locked_to_hold`: Verifies Agent 4 `selected_action[0] == 1.0` and `selected_action[10] == -100.0`.
  2. `test_tier1_5_governor_wire_packet_close_all_zero`: Verifies wire index 19 `POLICY_CLOSE_ALL == 0.0`.
  3. `test_tier1_5_governor_robust_under_random_inputs`: Verifies HOLD invariant across 20 randomized inputs.
  4. `test_tier1_5_governor_100_consecutive_ticks`: Verifies 100 consecutive ticks maintain 100.0% Governor HOLD rate.
  5. `test_tier1_5_governor_zero_override_contract`: Verifies external parameters cannot override locked invariant.

- **1.6: Single-Tick CPU Inference Latency SLA ($\le 50.0\text{ms}$)**:
  1. `test_tier1_6_latency_single_tick_p50_sla`: Verifies median latency P50 $\le 50.0\text{ms}$.
  2. `test_tier1_6_latency_mean_sla`: Verifies mean latency $\le 50.0\text{ms}$ steady-state.
  3. `test_tier1_6_latency_pre_warm_stability`: Verifies pipeline pre-warming prevents cold-start spikes.
  4. `test_tier1_6_latency_varying_inputs_bounded`: Verifies latency remains bounded across input regimes.
  5. `test_tier1_6_latency_zero_memory_leakage`: Verifies zero memory accumulation over consecutive inferences.

- **1.7: Trajectory Schema 17 TSV Ingestion**:
  1. `test_tier1_7_schema17_tsv_column_count_and_header`: Verifies 63 tab-separated Schema 17 columns.
  2. `test_tier1_7_schema17_winning_trade_parsing`: Verifies parsing of profitable trades and win metrics.
  3. `test_tier1_7_schema17_losing_trade_parsing`: Verifies parsing of stop-loss trades and loss metrics.
  4. `test_tier1_7_schema17_incremental_file_pointer`: Verifies `_last_exec_pos` avoids duplicate processing.
  5. `test_tier1_7_schema17_malformed_lines_graceful_recovery`: Verifies parser recovers from corrupted lines without crashing.

- **1.8: 12-Week Rolling Schedule Matrix**:
  1. `test_tier1_8_schedule_12_discrete_weeks_generation`: Verifies exactly 12 rolling weekly periods.
  2. `test_tier1_8_schedule_chronological_continuity`: Verifies chronological monotonicity across 12 weeks.
  3. `test_tier1_8_schedule_multi_asset_basket`: Verifies basket of BTCUSDm, ETHUSDm, and XAUUSDm each week.
  4. `test_tier1_8_schedule_checkpoint_promotion_lineage`: Verifies champion checkpoint succession lineage.
  5. `test_tier1_8_schedule_deliverables_manifest_spec`: Verifies weekly deliverables specification.

#### Tier 2: Boundary & Corner Cases (6 Test Cases)
- `test_tier2_1_flash_crash_and_extreme_volatility_shock`: -15% instant market drop preserves Governor HOLD and numeric stability.
- `test_tier2_2_zero_liquidity_and_frozen_order_book`: Zero depth with tiny alpha is rejected by Microstructure Shield.
- `test_tier2_3_spread_explosion_friction_gate`: Spread expands to 50 ATR -> Cost Ratio drops $< 3.5$ -> trade vetoed.
- `test_tier2_4_extreme_drawdown_continuous_deleveraging_floor`: Drawdown of 90% respects $0.10$ risk floor.
- `test_tier2_5_mutual_hold_inactivity_penalty_enforcement`: Mutual passivity incurs $-2.0$ penalty on both sides.
- `test_tier2_6_missing_bars_and_nan_inf_adversarial_injection`: Corrupted tensors sanitized via `nan_to_num`.

#### Tier 3: Cross-Feature Combinations (5 Test Cases)
- `test_tier3_1_high_correlation_plus_high_prm_rejection`: Cross-Asset Correlation Shield vetoes proposal if $\rho_{\text{eff}} > 0.85$ despite PRM $\ge 0.90$.
- `test_tier3_2_volatility_shock_two_tier_trailing_runner_invariant`: Two-Tier Runner locks 50% at 10 ATR, moves SL to $+1.0$ ATR, maintaining `POLICY_CLOSE_ALL == 0.0`.
- `test_tier3_3_negative_mt5_fill_star_rationalization_spin_update`: Realized MT5 loss streams to STaR -> rationalization bootstrapping -> SPIN preference update.
- `test_tier3_4_mad_reflexion_episodic_trap_veto_and_counter_trap_inversion`: Episodic memory queries trap similarity ($> 0.88$) and inverts into Counter-Trap Sniper Entry.
- `test_tier3_5_dynamic_kelly_continuous_scaling_microstructure_synergy`: Dynamic Kelly de-leveraging factor integrates smoothly with Microstructure Cost Ratio.

#### Tier 4: Real-World Walk-Forward Scenarios (4 Test Cases)
- `test_tier4_1_multi_week_market_regime_shifts_simulation`: Full 4-regime walk-forward simulation (Trending -> Sideway -> Volatile Sweep -> Breakout Expansion).
- `test_tier4_2_high_frequency_300_trades_per_day_pacing`: High-frequency pacing achieving $> 300$ trades/day with Win Rate $> 80.0\%$ and Profit Factor $> 5.0$.
- `test_tier4_3_coevolution_5_to_10_epochs_parameter_convergence`: 6-epoch Bilateral Co-Evolution verifying loss convergence and non-zero weight evolution ($||\Delta \theta|| > 0$).
- `test_tier4_4_12_week_rolling_deliverables_and_forensic_integrity_audit`: Verifies SHA-256 manifests, HTML reports, and TSV ledgers with strict zero-mock standards.

---

### 4. Test Execution & Verification

Run the entire 55-test E2E suite via the standard unit test runner:

```bash
python -m unittest research_reasoning_world_class/tests/test_e2e_walk_forward_triad.py
```

#### Official Verification Output:
```text
Ran 55 tests in 36.372s

OK
```
All 55 tests passed with 100% pass rate.
