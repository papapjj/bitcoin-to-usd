"""
Unit tests for Frontier Self-Play Triad:
1. Self-Play Fine-Tuning (SPIN)
2. Multi-Agent Debate & Reflexion (MAD)
3. Self-Taught Reasoner (STaR)
"""

import unittest
import torch
import numpy as np

from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from research_reasoning_world_class.src.frontier_selfplay_triad import (
    SelfPlayFineTuningSPIN,
    MultiAgentDebateAndReflexion,
    SelfTaughtReasonerSTaR,
)


class TestFrontierSelfPlayTriad(unittest.TestCase):
    def setUp(self):
        self.config = ReasoningConfig(
            input_dim=768,
            hidden_dim=256,
            num_hops=4,
            max_hops=8,
        )
        self.engine = MetacognitiveReasoningEngine(self.config)

    def test_spin_pairwise_preference_loss(self):
        spin = SelfPlayFineTuningSPIN(beta=0.1)

        # Collect 5 pairs
        dummy_state = torch.randn(768)
        for i in range(5):
            spin.collect_pair(
                state_z=dummy_state + torch.randn(768) * 0.05,
                action_challenger=1, # BUY (Winner)
                pnl_challenger=45.0,
                action_reference=2,  # SELL (Loser)
                pnl_reference=-30.0,
            )

        self.assertEqual(len(spin.buffer), 5)

        # Reference model (identical copy for test)
        ref_engine = MetacognitiveReasoningEngine(self.config)
        loss, metrics = spin.compute_spin_loss(self.engine, ref_engine, batch_size=4)

        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        self.assertIn("spin_loss", metrics)
        self.assertIn("accuracy", metrics)

    def test_multi_agent_debate_and_reflexion_veto(self):
        mad = MultiAgentDebateAndReflexion(hidden_dim=256, debate_turns=3)
        dummy_thought = torch.randn(1, 256)

        # 1. Run multi-turn debate
        action, conf, trace = mad.run_debate(dummy_thought, initial_bull_score=0.75, initial_bear_score=0.40)
        self.assertIn(action, [0, 1, 2])
        self.assertTrue(0.0 <= conf <= 1.0)
        self.assertEqual(len(trace["turns"]), 3)

        # 2. Record reflection on trapped trade
        mad.record_reflection(
            state_embedding=dummy_thought,
            trigger_price=95000.0,
            regime="volatile_sweep",
            action_taken=1, # Trapped BUY
            pnl=-50.0,
            mae=2.5,
        )
        self.assertEqual(len(mad.memory), 1)

        # 3. Query reflection veto on identical trap state
        vetoed, reason = mad.query_reflection_veto(dummy_thought, proposed_action=1)
        self.assertTrue(vetoed, f"Expected veto on identical trap, got {reason}")
        self.assertIn("REFLEXION_VETO", reason)

        # Query different action (e.g. SELL or HOLD) -> should NOT veto
        vetoed_sell, _ = mad.query_reflection_veto(dummy_thought, proposed_action=2)
        self.assertFalse(vetoed_sell)

    def test_star_rationale_generation_and_bootstrapping(self):
        star = SelfTaughtReasonerSTaR(self.engine, lr=1e-3)
        dummy_z = torch.randn(1, 768)

        # Case 1: Successful rationale
        sample_succ = star.process_tick_trajectory(dummy_z, real_future_pnl=50.0, real_future_action=1)
        self.assertTrue(sample_succ.generated_hops.shape[0] >= 4)

        # Case 2: Failed rationale -> trigger rationalization bootstrapping
        sample_fail = star.process_tick_trajectory(dummy_z, real_future_pnl=-35.0, real_future_action=2)
        self.assertEqual(len(star.gold_buffer), 2)

        # Add more samples to test fine-tuning step
        for i in range(4):
            star.process_tick_trajectory(torch.randn(1, 768), real_future_pnl=20.0, real_future_action=i % 3)

        train_metrics = star.train_step(batch_size=4)
        self.assertGreater(train_metrics["star_loss"], 0.0)
        self.assertTrue(np.isfinite(train_metrics["star_loss"]))

    def test_bilateral_coevolution_zero_sum_and_anti_collusion(self):
        spin = SelfPlayFineTuningSPIN(beta=0.1, inactivity_penalty=2.0)
        challenger = MetacognitiveReasoningEngine(self.config)
        adversary = MetacognitiveReasoningEngine(self.config)

        opt_c = torch.optim.AdamW(challenger.parameters(), lr=1e-4)
        opt_adv = torch.optim.AdamW(adversary.parameters(), lr=1e-4)

        # 1. Collect non-collusive interaction (Challenger wins)
        dummy_state = torch.randn(768)
        s1 = spin.collect_bilateral_interaction(
            state_z=dummy_state,
            action_challenger=1, # BUY
            pnl_challenger=45.0,
            action_adversary=2,  # SELL
            pnl_adversary=-30.0,
            market_ground_truth=1,
            inactivity_penalty=2.0,
        )
        self.assertFalse(s1.is_mutual_inactivity)
        # Verify strict zero-sum relation: zs_c + zs_adv == 0 when no mutual inactivity penalty
        self.assertAlmostEqual(s1.zero_sum_challenger + s1.zero_sum_adversary, 0.0, places=4)
        self.assertGreater(s1.zero_sum_challenger, 0.0)
        self.assertLess(s1.zero_sum_adversary, 0.0)

        # 2. Collect mutual inactivity (Collusion attempt: both choose HOLD)
        s2 = spin.collect_bilateral_interaction(
            state_z=dummy_state + torch.randn(768) * 0.1,
            action_challenger=0, # HOLD
            pnl_challenger=0.0,
            action_adversary=0,  # HOLD
            pnl_adversary=0.0,
            market_ground_truth=1, # Market was moving!
            inactivity_penalty=2.0,
        )
        self.assertTrue(s2.is_mutual_inactivity)
        # Both must suffer negative penalty: sum is -2 * penalty
        self.assertAlmostEqual(s2.zero_sum_challenger + s2.zero_sum_adversary, -4.0, places=4)
        self.assertLess(s2.zero_sum_challenger, 0.0)
        self.assertLess(s2.zero_sum_adversary, 0.0)

        # 3. Add more interactions for batch training
        for i in range(4):
            spin.collect_bilateral_interaction(
                state_z=torch.randn(768),
                action_challenger=(i % 2) + 1,
                pnl_challenger=20.0 if i % 2 == 0 else -15.0,
                action_adversary=((i + 1) % 2) + 1,
                pnl_adversary=20.0 if i % 2 == 1 else -15.0,
                market_ground_truth=(i % 2) + 1,
                inactivity_penalty=2.0,
            )

        # 4. Simultaneous Bilateral Training Step
        metrics = spin.bilateral_train_step(
            challenger_model=challenger,
            adversary_model=adversary,
            opt_challenger=opt_c,
            opt_adversary=opt_adv,
            batch_size=4,
            inactivity_weight=0.5,
            prm_weight=0.2,
        )
        self.assertIn("loss_challenger", metrics)
        self.assertIn("loss_adversary", metrics)
        self.assertIn("loss_inactivity_c", metrics)
        self.assertIn("mutual_inactivity_rate", metrics)
        self.assertTrue(np.isfinite(metrics["loss_challenger"]))
        self.assertTrue(np.isfinite(metrics["loss_adversary"]))


if __name__ == "__main__":
    unittest.main()
