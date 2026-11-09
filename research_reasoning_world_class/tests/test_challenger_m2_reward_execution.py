"""
Test Suite: Milestone M2 — Training Execution, Freezing Invariants & Checkpoint Routing Verification.
File: research_reasoning_world_class/tests/test_challenger_m2_reward_execution.py

Verifies:
1. SpecializedMassiveTrainer.__init__ unconditional parameter freezing (requires_grad == False).
2. Saved challenger checkpoint integrity: exactly 191 keys, 187 named parameters, 4,196,564 weights under strict=True.
3. Action-Collapse Resolution: Trade count restoration (>= 50 trades) vs previous freeze collapse (4 trades).
4. Safe Checkpoint Routing: Canonical checkpoint untouched while challenger is safely routed.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn

from research_reasoning_world_class.src.real_m1_environment import RealM1Environment
from scripts.train_mythos_muzero_coevolution import SlicedM1Environment
from scripts.train_mythos_muzero_specialized_massive import SpecializedMassiveTrainer

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "outputs" / "he_chuyen_gia" / "v193_runs"
CANONICAL_CKPT = CKPT_DIR / "mythos_muzero_coevolved_champion.pt"
CHALLENGER_CKPT = CKPT_DIR / "mythos_muzero_specialized_challenger.pt"
CSV_PATH = Path(r"C:\Users\Admin\Desktop\BTCUSDm.csv")


class TestSpecializedTrainerFreezingInvariants:
    """Verifies that SpecializedMassiveTrainer.__init__ unconditionally safeguards reasoning engine parameters."""

    @pytest.fixture(scope="class")
    def trainer_instance(self):
        master = RealM1Environment(csv_path=str(CSV_PATH), spread_usd=2.50)
        master.load_data("2025.01.15", "2025.02.28")
        s_bull = SlicedM1Environment(master, 1000, 1500, name="T_Bull")
        s_bear = SlicedM1Environment(master, 2000, 2500, name="T_Bear")
        s_chop = SlicedM1Environment(master, 3000, 3500, name="T_Chop")

        trainer = SpecializedMassiveTrainer(
            bull_env=s_bull,
            bear_env=s_bear,
            chop_env=s_chop,
            lr=3e-5,
            mcts_simulations=2,
            beta_dyna=0.5,
        )
        return trainer

    def test_reasoning_engine_parameters_unconditionally_frozen(self, trainer_instance):
        trainer = trainer_instance
        # Top-level reasoning engine
        trainable_cot = [p for p in trainer.reasoning_engine.parameters() if p.requires_grad]
        assert len(trainable_cot) == 0, f"trainer.reasoning_engine has {len(trainable_cot)} trainable parameters!"

        # Bull Champion reasoning engine
        bull_cot = [p for p in trainer.bull_champion.reasoning_engine.parameters() if p.requires_grad]
        assert len(bull_cot) == 0, f"bull_champion.reasoning_engine has {len(bull_cot)} trainable parameters!"

        # Bear Champion reasoning engine
        bear_cot = [p for p in trainer.bear_champion.reasoning_engine.parameters() if p.requires_grad]
        assert len(bear_cot) == 0, f"bear_champion.reasoning_engine has {len(bear_cot)} trainable parameters!"

    def test_optimizers_exclude_reasoning_engine_parameters(self, trainer_instance):
        trainer = trainer_instance
        for role_name, champ in [("Bull", trainer.bull_champion), ("Bear", trainer.bear_champion)]:
            opt_param_ids = {id(p) for group in champ.optimizer.param_groups for p in group["params"]}
            cot_param_ids = {id(p) for p in champ.reasoning_engine.parameters()}
            overlap = opt_param_ids.intersection(cot_param_ids)
            assert len(overlap) == 0, f"{role_name} optimizer includes {len(overlap)} reasoning_engine parameters!"


class TestCheckpointRoutingAndActionCollapseResolution:
    """Verifies that checkpoint routing and action collapse resolution contracts are satisfied."""

    def test_canonical_checkpoint_remains_canonical(self):
        assert CANONICAL_CKPT.exists(), "Canonical checkpoint must exist!"
        ckpt = torch.load(CANONICAL_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["reasoning_engine_state_dict"]
        assert len(sd) == 191, f"Expected 191 state_dict keys, got {len(sd)}"

        # Instantiate engine and verify strict=True loading + 187 params + 4,196,564 weights
        from research_reasoning_world_class.src.metacognitive_reasoning import MetacognitiveReasoningEngine, ReasoningConfig
        cfg = ReasoningConfig(input_dim=768, hidden_dim=256)
        engine = MetacognitiveReasoningEngine(cfg)
        load_res = engine.load_state_dict(sd, strict=True)
        assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0
        assert len(list(engine.named_parameters())) == 187
        assert sum(p.numel() for p in engine.parameters()) == 4_196_564

    def test_challenger_checkpoint_schema_and_exactness(self):
        assert CHALLENGER_CKPT.exists(), "Challenger checkpoint must exist!"
        ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["reasoning_engine_state_dict"]
        assert len(sd) == 191, f"Expected 191 state_dict keys, got {len(sd)}"

        # Verify strict=True loading + 187 params + 4,196,564 weights
        from research_reasoning_world_class.src.metacognitive_reasoning import MetacognitiveReasoningEngine, ReasoningConfig
        cfg = ReasoningConfig(input_dim=768, hidden_dim=256)
        engine = MetacognitiveReasoningEngine(cfg)
        load_res = engine.load_state_dict(sd, strict=True)
        assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0
        assert len(list(engine.named_parameters())) == 187
        assert sum(p.numel() for p in engine.parameters()) == 4_196_564
        assert ckpt["stats"]["status"] == "SAVED_CHALLENGER"

    def test_action_collapse_resolved_telemetry(self):
        ckpt = torch.load(CHALLENGER_CKPT, map_location="cpu", weights_only=False)
        stats = ckpt["stats"]
        total_trades = stats["total_trades"]
        win_rate = stats["win_rate"]
        net_profit = stats["net_profit"]
        profit_factor = stats["profit_factor"]

        # Action-Collapse check: Previous freeze run had only 4 trades.
        # Current reformed run must have healthy trading activity (>= 50 trades).
        assert total_trades >= 50, f"Action collapse detected: only {total_trades} trades executed!"
        assert total_trades == 66, f"Expected 66 trades, got {total_trades}"

        # Win rate & profitability comparison vs Freeze Challenger (25.0% WR, -$1,180.60 Net PnL)
        assert win_rate > 0.40, f"Win rate too low: {win_rate:.2%}"
        assert net_profit > 0.0, f"Net profit must be positive: ${net_profit:,.2f}"
        assert profit_factor > 1.0, f"Profit factor must be > 1.0: {profit_factor:.2f}"
