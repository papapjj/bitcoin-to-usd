import unittest
import numpy as np
from zigzag_miner.synthetic import repeated_swing_market
from zigzag_miner.causal_zigzag import causal_zigzag


class TestCausality(unittest.TestCase):
    def test_confirmed_pivots_do_not_change_when_future_appended(self):
        df = repeated_swing_market(repeats=30)
        cut = len(df) // 2
        a = causal_zigzag(df.high.values[:cut], df.low.values[:cut], df.close.values[:cut], atr_mult=1.0)
        b = causal_zigzag(df.high.values, df.low.values, df.close.values, atr_mult=1.0)
        b_prefix = [p for p in b if p.confirm_index < cut]
        self.assertEqual(a, b_prefix)

    def test_confirmation_never_precedes_pivot(self):
        df = repeated_swing_market(repeats=10)
        pivs = causal_zigzag(df.high.values, df.low.values, df.close.values, atr_mult=1.0)
        self.assertTrue(pivs)
        self.assertTrue(all(p.confirm_index > p.pivot_index for p in pivs))

if __name__ == "__main__":
    unittest.main()
