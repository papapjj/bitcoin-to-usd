"""
Unit and Integration Tests for Frontier Mythos MT5 Online Co-Evolution & observe_transition.
Verifies Architectural Directive 2026-10-24T20:21:12Z:
Connecting live MT5 execution stream (self_growing_execution.csv -> observe_transition -> Policy, PRM, Value Heads).
"""
import copy
import os
import tempfile
import unittest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from research_reasoning_world_class.src.mythos_mt5_policy import MythosMT5Policy


class DummyBasePolicy:
    """Mock 5-Agent base policy exposing act_nam_tac_nhan and observe_transition."""
    def __init__(self):
        self.input_size = 137
        self.output_size = 20
        self.observed_transitions = []

    def act_nam_tac_nhan(self, observation, goal=None, *args, **kwargs):
        class DummyTrace:
            def __init__(self, agent_idx):
                self.action_details = {
                    "five_agent_index": agent_idx,
                    "five_action_index": 0,
                    "five_reward_sample_eligible": True,
                }
                self.selected_idx = 0
                self.selected_action = np.zeros(20, dtype=np.float64)

        return tuple(DummyTrace(i) for i in range(5))

    def observe_transition(self, trace, next_obs, reward, cost, done, **kwargs):
        self.observed_transitions.append((trace, reward, cost, done))
        return {"base_credit": 1.0}


class TestMythosMT5OnlineCoEvolution(unittest.TestCase):
    def setUp(self):
        self.base_policy = DummyBasePolicy()
        self.policy = MythosMT5Policy(base_policy=self.base_policy, initial_capital=10_000.0)

    def test_observe_transition_positive_trade(self):
        """Verifies closed-loop RL adaptation on profitable trade fill."""
        initial_adapt_count = self.policy.adaptation_count
        # Simulate profitable trade: +180.50 USD, MFE 2.8 R, MAE 0.2 R
        result = self.policy.observe_transition(
            trace="five:mock_trace_1",
            next_obs=np.ones(137, dtype=np.float32) * 0.01,
            reward=180.50,
            cost=2.50,
            done=True,
            pnl=180.50,
            exit_type="TAKE_PROFIT",
            mfe=2.8,
            mae=0.2,
        )

        self.assertEqual(result["status"], "ADAPTED")
        self.assertEqual(self.policy.adaptation_count, initial_adapt_count + 1)
        self.assertAlmostEqual(self.policy.adapted_pnl, 180.50, places=2)
        self.assertGreater(result["loss"], 0.0)
        self.assertTrue(np.isfinite(result["loss"]))

    def test_observe_transition_negative_trade(self):
        """Verifies closed-loop RL adaptation on adverse trade fill (stop-loss)."""
        initial_adapt_count = self.policy.adaptation_count
        # Simulate loss trade: -65.20 USD, MFE 0.3 R, MAE 1.0 R
        result = self.policy.observe_transition(
            trace="five:mock_trace_2",
            next_obs=np.ones(137, dtype=np.float32) * -0.01,
            reward=-65.20,
            cost=2.50,
            done=True,
            pnl=-65.20,
            exit_type="STOP_LOSS",
            mfe=0.3,
            mae=1.0,
        )

        self.assertEqual(result["status"], "ADAPTED")
        self.assertEqual(self.policy.adaptation_count, initial_adapt_count + 1)
        self.assertAlmostEqual(self.policy.adapted_pnl, -65.20, places=2)
        self.assertGreater(result["loss"], 0.0)

    def test_sync_closed_deals_from_execution_ledger(self):
        """Verifies parsing of TSV Schema 17 self_growing_execution.csv into observe_transition."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            exec_file = Path(tmp_dir) / "self_growing_execution.csv"
            # Schema 17 TSV Header
            header = (
                "schema_version\tlogged_at\tack_sequence\tevent_id\ttrace_id\trequested_action\t"
                "action_name\tsource\ttransaction_type\tstatus\taccepted\tplaced\tfilled\tcancelled\t"
                "modified\tclosed\trejected\tretcode\torder_ticket\tdeal_ticket\tposition_ticket\t"
                "intent_context\tintent_expert\tintent_policy\tintent_state\tintent_reason\t"
                "position_owner_trace_id\tposition_owner_context\tposition_owner_expert\t"
                "position_owner_policy\tposition_owner_state\tposition_owner_reason\tdeal_entry\t"
                "deal_reason\tclose_reason\tvolume\tprice\tprofit\tcommission\tswap\t"
                "position_final_close\tinitial_risk\tage_bars\tentry_price\tinitial_sl_price\t"
                "initial_tp_price\tinitial_risk_distance_price\tfinal_sl_price\tfinal_tp_price\t"
                "old_sl_price\told_tp_price\tnew_sl_price\tnew_tp_price\tstop_controller\tstop_reason\t"
                "exit_kind\texit_controller\texit_trace_id\texit_requested_action\tpost_exit_horizon_bars\t"
                "post_exit_hold_r\tgeometry_entry_pipe\tgeometry_path_pipe\n"
            )
            # Row 1: Winning trade (+245.50 profit)
            row_win = (
                "17\t2025.01.06 10:15:00\t1\tevt-1\ttrace:1001\t1\tBUY\tpolicy\tDEAL_ADD\tCLOSED\t"
                "1\t0\t1\t0\t0\t1\t0\t0\t101\t201\t301\tCTX\tEXP\tPOL\tST\tRSN\t"
                "trace:1001\tCTX\tEXP\tPOL\tST\tRSN\t0\t0\tTAKE_PROFIT\t"
                "0.10\t94250.00\t245.50\t-1.50\t0.00\t1\t50.00\t15\t94000.00\t93950.00\t94250.00\t50.00\t"
                "93950.00\t94250.00\t0.00\t0.00\t0.00\t0.00\tCTRL\tRSN\tTAKE_PROFIT\tCTRL\ttrace:1001\t1\t"
                "0\t0.00\t50.00|BUY\t94300.00|93980.00|94250.00|45.00|15.00|2.45|3.20|0.40\n"
            )
            # Row 2: Second trade (+112.00 profit)
            row_win2 = (
                "17\t2025.01.06 11:30:00\t2\tevt-2\ttrace:1002\t2\tSELL\tpolicy\tDEAL_ADD\tCLOSED\t"
                "1\t0\t1\t0\t0\t1\t0\t0\t102\t202\t302\tCTX\tEXP\tPOL\tST\tRSN\t"
                "trace:1002\tCTX\tEXP\tPOL\tST\tRSN\t0\t0\tTRAILING_STOP\t"
                "0.10\t93800.00\t112.00\t-1.50\t0.00\t1\t50.00\t22\t94000.00\t94050.00\t93750.00\t50.00\t"
                "93850.00\t93750.00\t0.00\t0.00\t0.00\t0.00\tCTRL\tRSN\tTRAILING_STOP\tCTRL\ttrace:1002\t2\t"
                "0\t0.00\t50.00|SELL\t94020.00|93780.00|93800.00|45.00|15.00|1.12|2.10|0.20\n"
            )
            exec_file.write_text(header + row_win + row_win2, encoding="utf-8")

            # Temporarily redirect ledger path
            import research_reasoning_world_class.src.mythos_mt5_policy as m_pol
            orig_root = m_pol._project_root
            m_pol._project_root = str(Path(tmp_dir).parent)

            # Create the nested directory structure mt5_v2_runner/MQL5/Files/
            ledger_dir = Path(tmp_dir).parent / "mt5_v2_runner" / "MQL5" / "Files"
            ledger_dir.mkdir(parents=True, exist_ok=True)
            target_ledger = ledger_dir / "self_growing_execution.csv"
            target_ledger.write_text(header + row_win + row_win2, encoding="utf-8")

            try:
                self.policy._last_exec_pos = 0
                self.policy._sync_closed_deals()
                self.assertEqual(self.policy.closed_trades, 2)
                self.assertEqual(self.policy.wins, 2)
                self.assertAlmostEqual(self.policy.gross_profit, 357.50, places=2)
                self.assertEqual(self.policy.adaptation_count, 2)
                self.assertAlmostEqual(self.policy.adapted_pnl, 357.50, places=2)
            finally:
                m_pol._project_root = orig_root
                if target_ledger.exists():
                    target_ledger.unlink()


if __name__ == "__main__":
    unittest.main()
