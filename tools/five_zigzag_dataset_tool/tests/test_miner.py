import unittest
from zigzag_miner.synthetic import repeated_swing_market
from zigzag_miner.structures import PatternSpec
from zigzag_miner.miner import mine_dataframe


class TestMiner(unittest.TestCase):
    def test_mines_repeated_structure_and_shapes(self):
        df = repeated_swing_market(repeats=60)
        spec = PatternSpec.from_dict({
            "name": "synthetic",
            "history_bars": 64,
            "future_bars": 32,
            "min_gap_bars": 8,
            "legs": [
                {"direction": "up", "min_atr": 1.0, "max_atr": 30.0, "min_bars": 2, "max_bars": 30},
                {"direction": "down", "min_atr": 0.3, "max_atr": 20.0, "min_bars": 2, "max_bars": 30, "min_ratio_prev": 0.05, "max_ratio_prev": 0.95},
                {"direction": "up", "min_atr": 0.5, "max_atr": 30.0, "min_bars": 2, "max_bars": 30, "min_ratio_prev": 0.5, "max_ratio_prev": 4.0},
            ],
        })
        samples = list(mine_dataframe(df, spec, zigzag_atr_mult=1.0))
        self.assertGreater(len(samples), 20)
        s = samples[0]
        self.assertEqual(s["history"].shape, (64, 5))
        self.assertEqual(s["future"].shape, (32, 5))
        self.assertGreaterEqual(s["anchor_index"], s["pivots"][-1]["confirm_index"])

if __name__ == "__main__":
    unittest.main()
