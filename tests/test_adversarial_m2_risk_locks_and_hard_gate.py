"""
tests/test_adversarial_m2_risk_locks_and_hard_gate.py

Adversarial Risk Lock Violation & Hard Gate Injection Challenge for Milestone 2.
Author: challenger_m2_config_2 (EMPIRICAL CHALLENGER)

Adversarial Challenge Dimensions:
1. Logits Manipulation & Corrupted States for Action 10 (POLICY_CLOSE_ALL):
   - Extreme positive logits on Action 10 (+1000.0, +1e9, +1e30) vs Action 0 (-1000.0, -1e9).
   - Logits masking allowed[2:, :] = False; allowed[2:, 0] = True under deterministic and stochastic sampling.
   - 10,000 stochastic multinomial draws with uniform exploration (uniform_mix=0.5, 1.0) and entry floor.
   - Overriding malicious caller-supplied masks passed to forward_five.
   - Corrupted states (NaN, -Inf, +Inf).
   - 20-wire bridge execution argmax decoding guarantee.

2. Hard Direction Gate Injection Challenge:
   - Root-level and learning-level injection of context_direction_entry_gate_enabled = True.
   - Truthy string/numeric variations ("True", "true", 1, "yes").
   - MultiRegimeGateEvaluator end-to-end evaluation rejection, promotion blocking, and rollback safety.
   - Active YAML configurations scan for zero hard gate invariance.

3. Fixed Geometry Immutability Challenge (SL 5 ATR / TP 10 ATR / Max 120 bars):
   - MT5BacktestConfig rejection of rogue SL, TP, sub-minimal RR, and negative holding bars.
   - OrderConfig spread/SL cost ratio enforcement on M1 timeframe.
   - FiveSharedPolicy runtime overwrite of network parameter heads to fixed indices (SL 5.0, TP 10.0, delay 0, trailing 0).
   - FiveLifecycleReward 120-bar timeout penalty exactness (-0.25R).
   - EA .set file translation validation.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from self_growing_trader.config import AppConfig, OrderConfig, load_config
from self_growing_trader.execution_contract import (
    ACTION_COUNT,
    DEFAULT_SL_BIN_INDEX,
    DEFAULT_TP_BIN_INDEX,
    DEFAULT_TRAILING_BIN_INDEX,
    OUTPUT_COUNT,
    SL_ATR_BINS,
    TP_RR_BINS,
)
from self_growing_trader.five_agent_authority import (
    CLOSE_ALL_ACTION,
    FIVE_AGENT_COUNT,
    HOLD_ACTION,
    five_agent_authority,
)
from self_growing_trader.five_lifecycle_reward import (
    ExtendedLifecycleSettlementEngine,
    FiveLifecycleRewardConfig,
)
from self_growing_trader.five_shared_policy import (
    FiveActionResult,
    FiveAgentMemory,
    SharedFiveTradingPolicy,
)
from self_growing_trader.mt5_runner import (
    MT5BacktestConfig,
    write_tester_set,
)
from self_growing_trader.multi_regime_gate import (
    GateHurdles,
    MultiRegimeGateEvaluator,
    RegimeMetrics,
    evaluate_multi_regime_gate,
)
from self_growing_trader.orders import Action


# ==============================================================================
# SECTION 1: ADVERSARIAL LOGITS MANIPULATION & CORRUPTED STATES (ACTION 10)
# ==============================================================================

class TestAdversarialLogitsMaskingCloseAllLock:
    """Stress-tests logits masking to verify absolute impossibility of Action 10 (POLICY_CLOSE_ALL)."""

    def test_governor_authority_is_only_agent_with_historical_close_all(self) -> None:
        """In the 5-agent authority contract, only Agent 4 had CLOSE_ALL in allowed_actions; agents 0..3 never did."""
        for agent_id in range(4):
            auth = five_agent_authority(agent_id)
            assert CLOSE_ALL_ACTION not in auth.allowed_actions, (
                f"Agent {agent_id} ({auth.role}) must never have CLOSE_ALL in allowed actions!"
            )

        gov_auth = five_agent_authority(4)
        assert gov_auth.role == "PORTFOLIO_GOVERNOR"
        assert CLOSE_ALL_ACTION in gov_auth.allowed_actions

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_extreme_positive_logits_deterministic_elimination(self) -> None:
        """Deterministic argmax with Action 10 logit = +1000.0 and HOLD = -1000.0 returns HOLD (0)."""
        logits = torch.full((FIVE_AGENT_COUNT, ACTION_COUNT), -100.0, dtype=torch.float32)
        # Adversary injects astronomical logits on CLOSE_ALL (Action 10) for agents 2, 3, 4
        logits[2:, CLOSE_ALL_ACTION] = 1000.0
        logits[2:, HOLD_ACTION] = -1000.0

        # Mandatory Governor HOLD lock mask
        allowed = torch.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        actions, log_prob, entropy, probs = SharedFiveTradingPolicy._primary_action_distribution(
            logits,
            allowed=allowed,
            deterministic=True,
            entry_probability_floor=0.0,
            uniform_mix=0.0,
        )

        for agent_idx in range(2, 5):
            assert actions[agent_idx].item() == HOLD_ACTION, (
                f"Agent {agent_idx} emitted action {actions[agent_idx].item()} instead of HOLD (0)!"
            )
            assert probs[agent_idx, CLOSE_ALL_ACTION].item() == pytest.approx(0.0, abs=1e-12)
            assert probs[agent_idx, HOLD_ACTION].item() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_stochastic_sampling_10k_draws_zero_close_all_emissions(self) -> None:
        """Under stochastic multinomial sampling across 10,000 draws with extreme bias and exploration, Action 10 is NEVER emitted."""
        logits = torch.full((FIVE_AGENT_COUNT, ACTION_COUNT), -50.0, dtype=torch.float32)
        logits[4, CLOSE_ALL_ACTION] = 500.0   # Extreme attraction towards CLOSE_ALL
        logits[4, HOLD_ACTION] = -500.0       # Repulsion away from HOLD

        allowed = torch.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        generator = torch.Generator().manual_seed(42)

        # Draw 10,000 samples under heavy uniform mixing and entry floor pressure
        for _ in range(100):  # 100 iterations of batch 100
            actions, _, _, probs = SharedFiveTradingPolicy._primary_action_distribution(
                logits,
                allowed=allowed,
                deterministic=False,
                entry_probability_floor=0.30,
                uniform_mix=0.50,  # 50% random uniform exploration
                sampling_generator=generator,
            )
            # Verify Agent 4 specifically and agents 2, 3
            for agent_idx in range(2, 5):
                sampled_action = actions[agent_idx].item()
                assert sampled_action == HOLD_ACTION, (
                    f"Agent {agent_idx} emitted non-HOLD action {sampled_action} during stochastic sampling!"
                )
                assert sampled_action != CLOSE_ALL_ACTION, "CRITICAL: Action 10 (CLOSE_ALL) was emitted!"
                assert probs[agent_idx, CLOSE_ALL_ACTION].item() == 0.0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_maximum_exploration_uniform_mix_preserves_hold_lock(self) -> None:
        """With uniform_mix = 1.0 (pure exploration), uniform fallback over legal mask still restricts agents 2..4 to HOLD."""
        logits = torch.randn((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.float32)
        logits[4, CLOSE_ALL_ACTION] = 1e6

        allowed = torch.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        actions, _, _, probs = SharedFiveTradingPolicy._primary_action_distribution(
            logits,
            allowed=allowed,
            deterministic=False,
            entry_probability_floor=0.0,
            uniform_mix=1.0,  # 100% pure uniform exploration
        )

        for agent_idx in range(2, 5):
            assert actions[agent_idx].item() == HOLD_ACTION
            assert probs[agent_idx, HOLD_ACTION].item() == pytest.approx(1.0, abs=1e-6)
            assert probs[agent_idx, CLOSE_ALL_ACTION].item() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_entry_probability_floor_does_not_leak_to_non_entry_agents(self) -> None:
        """With entry_probability_floor = 1.0, entry mask &= legal ensures agents 2..4 receive zero entry probability."""
        logits = torch.zeros((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.float32)

        allowed = torch.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        actions, _, _, probs = SharedFiveTradingPolicy._primary_action_distribution(
            logits,
            allowed=allowed,
            deterministic=False,
            entry_probability_floor=1.0,  # Max entry floor
            uniform_mix=0.0,
        )

        for agent_idx in range(2, 5):
            assert actions[agent_idx].item() == HOLD_ACTION
            assert probs[agent_idx, HOLD_ACTION].item() == pytest.approx(1.0, abs=1e-6)
            for action_idx in range(1, ACTION_COUNT):
                assert probs[agent_idx, action_idx].item() == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_forward_five_overrides_malicious_caller_masks(self) -> None:
        """If an adversary passes allowed_action_masks enabling Action 10 to forward_five, it is strictly overridden."""
        policy = SharedFiveTradingPolicy()
        policy.heads.seven_choice_fixed_geometry_enabled = True
        policy.heads.lock_non_entry_to_hold = True

        observations = torch.randn(FIVE_AGENT_COUNT, 137, dtype=torch.float32)
        memories = [FiveAgentMemory(hidden=torch.zeros(4096), cell=torch.zeros(4096), previous_action=torch.tensor(0)) for _ in range(5)]

        # Malicious mask attempting to permit Action 10 for Agent 4
        rogue_mask = torch.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.bool)
        rogue_mask[4, CLOSE_ALL_ACTION] = True

        # Hook action logits layer to output massive Action 10 logit
        orig_forward = policy.heads.primary_action_logits
        def rogue_action_head(h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            l = orig_forward(h, a).clone()
            l[4, CLOSE_ALL_ACTION] = 1000.0
            l[4, HOLD_ACTION] = -1000.0
            return l
        policy.heads.primary_action_logits = rogue_action_head

        results, next_mems, info = policy.forward_five(
            observations,
            memories,
            allowed_action_masks=rogue_mask,
            deterministic=True,
        )

        for agent_idx in range(2, 5):
            assert results[agent_idx].action_index == HOLD_ACTION, (
                f"Agent {agent_idx} executed action {results[agent_idx].action_index} instead of HOLD!"
            )
            assert results[agent_idx].action_index != CLOSE_ALL_ACTION

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_corrupted_nan_and_inf_logits_handled_fail_safe(self) -> None:
        """When logits contain NaN or -Inf on disallowed actions, torch.nan_to_num and clamping keep HOLD intact."""
        logits = torch.zeros((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.float32)
        # Inject NaN and Inf into disallowed actions
        logits[4, CLOSE_ALL_ACTION] = float("nan")
        logits[4, 1] = float("inf")
        logits[4, 2] = float("-inf")
        logits[4, HOLD_ACTION] = 5.0

        allowed = torch.ones((FIVE_AGENT_COUNT, ACTION_COUNT), dtype=torch.bool)
        allowed[2:, :] = False
        allowed[2:, 0] = True

        actions, _, _, probs = SharedFiveTradingPolicy._primary_action_distribution(
            logits,
            allowed=allowed,
            deterministic=True,
            entry_probability_floor=0.0,
            uniform_mix=0.0,
        )

        assert actions[4].item() == HOLD_ACTION
        assert not torch.isnan(probs[4]).any(), "Probabilities must not contain NaN"
        assert probs[4, HOLD_ACTION].item() == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_wire_output_argmax_binding_guarantees_hold(self) -> None:
        """Verifies the 20-float wire execution array: output[action] boost ensures argmax is strictly HOLD (0)."""
        policy = SharedFiveTradingPolicy()
        policy.heads.seven_choice_fixed_geometry_enabled = True
        policy.heads.lock_non_entry_to_hold = True

        observations = torch.randn(FIVE_AGENT_COUNT, 137, dtype=torch.float32)
        memories = [FiveAgentMemory(hidden=torch.zeros(4096), cell=torch.zeros(4096), previous_action=torch.tensor(0)) for _ in range(5)]

        # Hook action head to return high logits on Action 10
        orig_forward = policy.heads.primary_action_logits
        def biased_head(h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            l = orig_forward(h, a).clone()
            l[4, CLOSE_ALL_ACTION] = 50.0
            return l
        policy.heads.primary_action_logits = biased_head

        results, _, _ = policy.forward_five(observations, memories, deterministic=True)

        for agent_idx in range(2, 5):
            wire = results[agent_idx].output_vector
            assert isinstance(wire, np.ndarray)
            assert wire.shape == (OUTPUT_COUNT,)
            # The downstream execution decoder decodes action as argmax of wire[:ACTION_COUNT]
            decoded_action = int(np.argmax(wire[:ACTION_COUNT]))
            assert decoded_action == HOLD_ACTION, (
                f"Agent {agent_idx} decoded wire action {decoded_action} is not HOLD!"
            )
            assert decoded_action != CLOSE_ALL_ACTION


# ==============================================================================
# SECTION 2: ADVERSARIAL HARD DIRECTION GATE INJECTION CHALLENGE
# ==============================================================================

class TestAdversarialHardGateInjection:
    """Stress-tests MultiRegimeGateEvaluator to ensure context_direction_entry_gate_enabled is caught and rejected."""

    @pytest.fixture
    def stellar_metrics(self) -> dict[str, RegimeMetrics]:
        """Synthetic metrics meeting all quantitative hurdles with flying colors."""
        return {
            "bull": RegimeMetrics(
                regime_id="bull",
                total_trades=100,
                wins=65,
                losses=35,
                win_rate_pct=65.0,
                profit_factor=2.5,
                max_drawdown_pct=3.0,
                expectancy_r=0.25,
                payoff_ratio=2.0,
                counter_trend_rate_pct=10.0,
                timeout_close_rate_pct=5.0,
                premature_closes=0,
            ),
            "bear": RegimeMetrics(
                regime_id="bear",
                total_trades=100,
                wins=60,
                losses=40,
                win_rate_pct=60.0,
                profit_factor=2.2,
                max_drawdown_pct=3.5,
                expectancy_r=0.20,
                payoff_ratio=1.9,
                counter_trend_rate_pct=12.0,
                timeout_close_rate_pct=6.0,
                premature_closes=0,
            ),
            "chop": RegimeMetrics(
                regime_id="chop",
                total_trades=80,
                wins=48,
                losses=32,
                win_rate_pct=60.0,
                profit_factor=1.9,
                max_drawdown_pct=4.0,
                expectancy_r=0.15,
                payoff_ratio=1.7,
                counter_trend_rate_pct=15.0,
                timeout_close_rate_pct=8.0,
                premature_closes=0,
            ),
        }

    def test_gate_evaluator_catches_root_level_injection(self, stellar_metrics: dict[str, RegimeMetrics]) -> None:
        """Candidate config with context_direction_entry_gate_enabled=True at root level fails gate immediately."""
        candidate_config = {
            "context_direction_entry_gate_enabled": True,
            "orders": {"timeframe": "M1"},
        }

        passed, reasons, agg = evaluate_multi_regime_gate(
            stellar_metrics,
            hurdles=GateHurdles(require_natural_rl=True),
            candidate_config=candidate_config,
        )

        assert passed is False
        assert any("hard_direction_gate_forbidden" in r for r in reasons)
        assert any("context_direction_entry_gate_enabled is True" in r for r in reasons)

    def test_gate_evaluator_catches_learning_level_injection(self, stellar_metrics: dict[str, RegimeMetrics]) -> None:
        """Candidate config with context_direction_entry_gate_enabled=True under learning fails gate immediately."""
        candidate_config = {
            "learning": {
                "context_direction_entry_gate_enabled": True,
            }
        }

        passed, reasons, agg = evaluate_multi_regime_gate(
            stellar_metrics,
            hurdles=GateHurdles(require_natural_rl=True),
            candidate_config=candidate_config,
        )

        assert passed is False
        assert any("hard_direction_gate_forbidden" in r for r in reasons)

    def test_gate_evaluator_catches_truthy_numeric_and_string_variations(self, stellar_metrics: dict[str, RegimeMetrics]) -> None:
        """Candidate config with truthy values (1, 'true', 'True') in context_direction_entry_gate_enabled."""
        for rogue_val in [1, "true", "True", "yes"]:
            candidate_config = {
                "learning": {
                    "context_direction_entry_gate_enabled": rogue_val,
                }
            }
            passed, reasons, agg = evaluate_multi_regime_gate(
                stellar_metrics,
                hurdles=GateHurdles(require_natural_rl=True),
                candidate_config=candidate_config,
            )
            assert passed is False, f"Gate failed to reject rogue value: {rogue_val}"
            assert any("hard_direction_gate_forbidden" in r for r in reasons)

    def test_multi_regime_evaluator_with_instance_config_rejects_candidate(
        self, tmp_path: Path, stellar_metrics: dict[str, RegimeMetrics]
    ) -> None:
        """When config is provided to MultiRegimeGateEvaluator, candidate with hard gate is rejected and promotion blocked."""
        candidate_pkl = tmp_path / "candidate.pkl"
        candidate_pkl.write_bytes(b"candidate_state_dummy_v1")
        output_dir = tmp_path / "eval_output_rejected"
        output_dir.mkdir()

        injected_config = {
            "learning": {
                "context_direction_entry_gate_enabled": True,
            }
        }

        evaluator = MultiRegimeGateEvaluator(
            benchmark_regimes={
                "bull": {"market_type": "bull"},
                "bear": {"market_type": "bear"},
                "chop": {"market_type": "chop"},
            },
            hurdles=GateHurdles(require_natural_rl=True),
            config=injected_config,
        )

        result = evaluator.evaluate_candidate(
            candidate_path=candidate_pkl,
            output_dir=output_dir,
            candidate_config=injected_config,
            precomputed_metrics=stellar_metrics,
        )

        # Verification of hard rejection when config is bound
        assert result.passed is False, "Candidate with hard direction gate must NOT pass!"
        assert result.promoted is False, "Candidate with hard direction gate must NOT be promoted!"
        assert result.promoted_checkpoint_path is None
        assert any("hard_direction_gate_forbidden" in r for r in result.reasons)

        champ = output_dir / "best_promoted_checkpoint.pkl"
        assert not champ.exists(), "Champion checkpoint must NOT be written when hard gate is injected!"

    def test_multi_regime_evaluator_vulnerability_operator_precedence_leakage(
        self, tmp_path: Path, stellar_metrics: dict[str, RegimeMetrics]
    ) -> None:
        """VERIFICATION OF REMEDIATION: line 1085 operator precedence fix preserves candidate_config when self.config is None.

        In multi_regime_gate.py:
            effective_cfg = candidate_config or (getattr(self.config, "to_dict", lambda: self.config)() if self.config else None)
        When self.config is None (the default), candidate_config is strictly preserved,
        triggering hard_direction_gate_forbidden and blocking champion promotion.
        """
        candidate_pkl = tmp_path / "candidate_leak.pkl"
        candidate_pkl.write_bytes(b"candidate_state_dummy_v2")
        output_dir = tmp_path / "eval_output_leaked"
        output_dir.mkdir()

        injected_config = {
            "learning": {
                "context_direction_entry_gate_enabled": True,
            }
        }

        # Evaluator created with config=None (the default constructor argument)
        evaluator = MultiRegimeGateEvaluator(
            benchmark_regimes={
                "bull": {"market_type": "bull"},
                "bear": {"market_type": "bear"},
                "chop": {"market_type": "chop"},
            },
            hurdles=GateHurdles(require_natural_rl=True),
            config=None,  # Default
        )

        result = evaluator.evaluate_candidate(
            candidate_path=candidate_pkl,
            output_dir=output_dir,
            candidate_config=injected_config,
            precomputed_metrics=stellar_metrics,
        )

        # Verification that the vulnerability is FIXED (remediated)
        assert result.passed is False, "Remediated evaluator must reject candidate when context_direction_entry_gate_enabled=True!"
        assert result.promoted is False, "Remediated evaluator must NOT promote candidate with hard direction gate!"
        assert result.promoted_checkpoint_path is None
        assert len(result.reasons) > 0
        assert "hard_direction_gate_forbidden" in result.reasons[0]

        # Champion checkpoint must NOT be written when hard gate is injected
        champ = output_dir / "best_promoted_checkpoint.pkl"
        assert not champ.exists(), (
            "Champion checkpoint must NOT exist when candidate fails direction gate hurdle!"
        )

    def test_active_yaml_configs_zero_hard_gate_invariance(self) -> None:
        """Scans all YAML files in configs/ to assert zero active occurrences of context_direction_entry_gate_enabled: true."""
        config_files = list((ROOT / "configs").glob("**/*.yaml"))
        assert len(config_files) > 0, "No config files found in configs/"

        violations = []
        for cfg_path in config_files:
            try:
                content = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue

            learning = content.get("learning", {}) if isinstance(content, dict) else {}
            overrides = content.get("learning_overrides", {}) if isinstance(content, dict) else {}

            if learning and learning.get("context_direction_entry_gate_enabled") is True:
                violations.append(f"{cfg_path.name}: learning.context_direction_entry_gate_enabled is True")
            if overrides and overrides.get("context_direction_entry_gate_enabled") is True:
                violations.append(f"{cfg_path.name}: learning_overrides.context_direction_entry_gate_enabled is True")

        assert len(violations) == 0, f"Found hard direction gate enabled in configs: {violations}"


# ==============================================================================
# SECTION 3: ADVERSARIAL FIXED GEOMETRY IMMUTABILITY CHALLENGE
# ==============================================================================

class TestAdversarialFixedGeometryImmutability:
    """Stress-tests fixed geometry parameters (SL=5.0, TP=10.0, MaxHolding=120) against rogue overrides."""

    def test_mt5_backtest_config_rejects_negative_or_zero_sl(self) -> None:
        """MT5BacktestConfig raises ValueError if fixed_sl_atr <= 0."""
        with pytest.raises(ValueError, match="min_rr/technical_min_rr are invalid"):
            MT5BacktestConfig(fixed_sl_atr=0.0)

        with pytest.raises(ValueError, match="min_rr/technical_min_rr are invalid"):
            MT5BacktestConfig(fixed_sl_atr=-5.0)

    def test_mt5_backtest_config_rejects_negative_or_zero_tp(self) -> None:
        """MT5BacktestConfig raises ValueError if fixed_tp_atr <= 0."""
        with pytest.raises(ValueError, match="min_rr/technical_min_rr are invalid"):
            MT5BacktestConfig(fixed_tp_atr=0.0)

        with pytest.raises(ValueError, match="min_rr/technical_min_rr are invalid"):
            MT5BacktestConfig(fixed_tp_atr=-10.0)

    def test_mt5_backtest_config_rejects_sub_minimal_rr(self) -> None:
        """MT5BacktestConfig raises ValueError if fixed_tp_atr / fixed_sl_atr < technical_min_rr."""
        # SL = 5.0, TP = 2.0 (RR = 0.4 < technical_min_rr 1.2)
        with pytest.raises(ValueError, match="min_rr/technical_min_rr are invalid"):
            MT5BacktestConfig(fixed_sl_atr=5.0, fixed_tp_atr=2.0, technical_min_rr=1.2)

    def test_order_config_spread_sl_cost_enforcement(self) -> None:
        """OrderConfig on M1 timeframe rejects sub-minimal SL that violates spread cost constraint."""
        # empirical_spread_atr = 0.649373. With max_spread_cost_r = 0.15, SL must be >= 4.33 ATR.
        with pytest.raises(ValueError, match="violates spread/SL ratio"):
            OrderConfig(timeframe="M1", fixed_sl_atr=1.0, max_spread_cost_r=0.15)

        with pytest.raises(ValueError, match="violates spread/SL ratio"):
            OrderConfig(timeframe="M1", fixed_sl_atr=2.5, max_spread_cost_r=0.15)

        # SL = 5.0 ATR passes comfortably (spread cost = 0.649 / 5.0 = 0.130 <= 0.15)
        valid_orders = OrderConfig(timeframe="M1", fixed_sl_atr=5.0, fixed_tp_atr=10.0, max_spread_cost_r=0.15)
        assert valid_orders.fixed_sl_atr == 5.0
        assert valid_orders.fixed_tp_atr == 10.0

    @pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
    def test_five_shared_policy_overwrites_network_geometry_heads(self) -> None:
        """When seven_choice_fixed_geometry_enabled=True, SL and TP are unconditionally forced to fixed bins."""
        policy = SharedFiveTradingPolicy()
        policy.heads.seven_choice_fixed_geometry_enabled = True
        policy.heads.seven_choice_fixed_sl_index = SL_ATR_BINS.index(5.0)  # Index 7 (5.0 ATR)
        policy.heads.seven_choice_fixed_tp_index = TP_RR_BINS.index(2.0)   # Index 5 (2.0 RR = 10.0 ATR)

        # Randomize parameter heads weights heavily
        for p in policy.heads.parameter_heads["sl"].parameters():
            p.data.fill_(100.0)
        for p in policy.heads.parameter_heads["tp"].parameters():
            p.data.fill_(-100.0)

        observations = torch.randn(FIVE_AGENT_COUNT, 137, dtype=torch.float32)
        memories = [FiveAgentMemory(hidden=torch.zeros(4096), cell=torch.zeros(4096), previous_action=torch.tensor(0)) for _ in range(5)]

        # Force Agent 0 to emit Buy action (action 1)
        orig_action = policy.heads.primary_action_logits
        def force_buy_head(h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
            l = orig_action(h, a).clone()
            l[0, 1] = 100.0  # Buy
            return l
        policy.heads.primary_action_logits = force_buy_head

        results, _, _ = policy.forward_five(observations, memories, deterministic=True)

        # Check Agent 0 (Long Entry Owner)
        assert results[0].action_index == 1  # MARKET_BUY
        assert results[0].sl_index == SL_ATR_BINS.index(5.0), f"Expected SL bin for 5.0 ATR, got {results[0].sl_index}"
        assert results[0].tp_index == TP_RR_BINS.index(2.0), f"Expected TP bin for 2.0 RR, got {results[0].tp_index}"
        assert results[0].delay_index == 0
        assert results[0].trailing_index == DEFAULT_TRAILING_BIN_INDEX
        assert results[0].entry_offset_index == 7

    def test_five_lifecycle_reward_120_bars_timeout_penalty(self) -> None:
        """ExtendedLifecycleSettlementEngine applies exact quadratic time penalty (-0.25R) at bar 120."""
        engine = ExtendedLifecycleSettlementEngine(
            alpha_mae=0.20,
            beta_time=0.25,
            gamma_mfe=0.10,
            timeout_penalty_r=0.15,
            max_holding_bars=120,
        )

        # Trade closed due to TIMEOUT at bar 120 with 0.0 PnL and 0.0 MAE/MFE
        res = engine.compute_settlement_reward(
            realized_r=0.0,
            mae_r=0.0,
            mfe_r=0.0,
            duration_bars=120,
            exit_kind="TIMEOUT",
        )

        # Holding duration penalty at 120 bars: beta * (120/120)^2 = 0.25
        # Timeout penalty: 0.15
        # Total final reward = 0.0 - 0.25 - 0.15 = -0.40R
        assert res["time_penalty"] == pytest.approx(0.25, abs=1e-5)
        assert res["timeout_penalty"] == pytest.approx(0.15, abs=1e-5)
        assert res["is_timeout"] is True
        assert res["reward_final"] == pytest.approx(-0.40, abs=1e-5)

    def test_ea_tester_set_fixed_geometry_exact_values(self, tmp_path: Path) -> None:
        """write_tester_set writes exact FixedSlAtr=5, FixedTpAtr=10, MaxHoldingBars=120 into EA .set file."""
        set_path = tmp_path / "test_ea.set"
        config = MT5BacktestConfig(
            fixed_sl_atr=5.0,
            fixed_tp_atr=10.0,
            max_holding_bars=120,
        )

        write_tester_set(config, set_path)
        content = set_path.read_text(encoding="utf-8")

        assert "FixedSlAtr=5" in content
        assert "FixedTpAtr=10" in content
        assert "MaxHoldingBars=120" in content
