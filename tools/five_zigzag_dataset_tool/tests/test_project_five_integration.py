from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from zigzag_miner.causal_zigzag import compute_atr
from zigzag_miner.project_five_adapter import ProjectFiveContract, ProjectFiveDataset
from zigzag_miner.io import write_indexed_five_shards
from zigzag_miner.synthetic import repeated_swing_market


PROJECT_ROOT = os.environ.get("FIVE_PROJECT_ROOT", "").strip()
PROJECT_CONFIG = os.environ.get("FIVE_PROJECT_CONFIG", "configs/he_chuyen_gia_latest_ui_v193.yaml")


@unittest.skipUnless(PROJECT_ROOT, "dat FIVE_PROJECT_ROOT de chay kiem tra noi du an")
class TestProjectFiveIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = ProjectFiveContract.load(PROJECT_ROOT, PROJECT_CONFIG)

    def _raw(self, repeats: int = 260) -> pd.DataFrame:
        df = repeated_swing_market(repeats=repeats).copy()
        df["spread"] = 2.0
        df["real_volume"] = 0.0
        return df

    def test_tool_atr_mt5_sma_matches_project_reference(self):
        df = self._raw(repeats=8)
        period = 30
        ours = compute_atr(df.high.values, df.low.values, df.close.values, period, mode="mt5_sma")
        project_df = pd.DataFrame({
            "High": df.high.values,
            "Low": df.low.values,
            "Close": df.close.values,
        })
        reference = self.contract.modules["features"].compute_atr(project_df, period).to_numpy(dtype=float)
        np.testing.assert_allclose(ours[period - 1:], reference[period - 1:], rtol=0.0, atol=1e-12)

    def test_indexed_tape_reconstructs_exact_137_environment_input(self):
        raw = self._raw()
        project_raw = self.contract.prepare_project_raw(raw)
        feature_frame = self.contract.build_feature_frame(project_raw)
        self.assertGreater(len(feature_frame), 1000)

        times = pd.to_datetime(feature_frame["Datetime"], utc=True)
        anchor_time = times.iloc[len(times) // 2]
        raw_times = pd.DatetimeIndex(raw.index).tz_convert("UTC")
        anchor_raw = int(np.flatnonzero(raw_times == anchor_time)[0])
        start_raw = anchor_raw - 64
        end_raw = anchor_raw + 32
        self.assertGreaterEqual(start_raw, 0)
        self.assertLess(end_raw, len(raw))

        sample = {
            "pattern": "kiem_tra",
            "history_start_index": start_raw,
            "anchor_index": anchor_raw,
            "future_end_index": end_raw,
            "anchor_time": str(anchor_time),
            "anchor_close": float(raw.close.iloc[anchor_raw]),
            "anchor_atr": 1.0,
            "pivots": [],
            "legs": [],
            "outcome": {
                "mfe_atr": 1.0, "mae_atr": -1.0, "final_return_atr": 0.0,
                "long_mfe_atr": 1.0, "long_mae_atr": -1.0, "long_final_return_atr": 0.0,
                "short_mfe_atr": 1.0, "short_mae_atr": -1.0, "short_final_return_atr": 0.0,
            },
        }
        enriched = list(self.contract.enrich_index_samples([sample], raw, feature_frame))
        self.assertEqual(len(enriched), 1)
        item = enriched[0]
        self.assertEqual(item["context_steps"], 64)
        self.assertEqual(item["future_steps"], 32)

        with tempfile.TemporaryDirectory() as td:
            manifest = self.contract.write_tape(feature_frame, td)
            saved_times = np.load(Path(td) / "five_market_tape" / "time_ns.npy", allow_pickle=False)
            self.assertGreater(abs(int(saved_times[0])), 10**17)
            write_indexed_five_shards([item], td, shard_size=10, dataset_metadata=self.contract.metadata(tape_manifest=manifest))
            with ProjectFiveDataset(td, PROJECT_ROOT, PROJECT_CONFIG) as dataset:
                context, anchor = dataset.context_and_anchor(item)
                self.assertEqual(context.shape, (64, 137))
                self.assertEqual(anchor.shape, (137,))
                env = dataset.environment_for_sample(item)
                np.testing.assert_allclose(anchor, env.observation(64), rtol=0.0, atol=1e-6)

            five = self.contract.modules["five"]
            import torch
            encoder = five.FiveObservationEncoder(pre_lstm_size=32)
            obs = torch.as_tensor(np.repeat(anchor[None, :], 5, axis=0), dtype=torch.float32)
            ids = torch.arange(5, dtype=torch.long)
            previous = torch.zeros(5, dtype=torch.long)
            encoded, _, _ = encoder(obs, ids, previous)
            self.assertEqual(tuple(encoded.shape), (5, 32))

    def test_project_features_at_anchor_do_not_change_when_future_removed(self):
        raw = self._raw()
        project_raw = self.contract.prepare_project_raw(raw)
        full = self.contract.build_feature_frame(project_raw)
        times = pd.to_datetime(full["Datetime"], utc=True)
        anchor_time = times.iloc[len(times) // 2]
        raw_times = pd.DatetimeIndex(raw.index).tz_convert("UTC")
        anchor_raw = int(np.flatnonzero(raw_times == anchor_time)[0])
        prefix_raw = raw.iloc[:anchor_raw + 1].copy()
        prefix_project = self.contract.prepare_project_raw(prefix_raw)
        prefix = self.contract.build_feature_frame(prefix_project)
        row_full = full.loc[pd.to_datetime(full["Datetime"], utc=True) == anchor_time, list(self.contract.environment_columns)].iloc[0]
        row_prefix = prefix.loc[pd.to_datetime(prefix["Datetime"], utc=True) == anchor_time, list(self.contract.environment_columns)].iloc[0]
        np.testing.assert_allclose(row_full.to_numpy(float), row_prefix.to_numpy(float), rtol=1e-10, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
