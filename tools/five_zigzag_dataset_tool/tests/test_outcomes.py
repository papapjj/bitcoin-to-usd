from __future__ import annotations

import unittest

from zigzag_miner.miner import mine_dataframe
from zigzag_miner.structures import PatternSpec
from zigzag_miner.synthetic import repeated_swing_market


class TestDirectionalOutcomes(unittest.TestCase):
    def test_long_and_short_outcomes_are_explicit_and_symmetric(self):
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
        sample = next(mine_dataframe(df, spec))
        o = sample["outcome"]
        self.assertAlmostEqual(o["short_mfe_atr"], -o["long_mae_atr"])
        self.assertAlmostEqual(o["short_mae_atr"], -o["long_mfe_atr"])
        self.assertAlmostEqual(o["short_final_return_atr"], -o["long_final_return_atr"])


if __name__ == "__main__":
    unittest.main()
