import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from qrcit import QRCIT, QRCITConfig, test_independence
from qrcit.datasets import SDB_VARIANTS, pnl, sdb, trial_seed

ROOT = Path(__file__).resolve().parents[1]


class CoreChecks(unittest.TestCase):
    def test_recovered_seed_identity(self):
        d = pd.read_csv(ROOT / "results/synthetic/trials.csv")
        seeds = [
            trial_seed(r.benchmark.lower(), r.n, r.d_z, r.beta, r.trial_id, r.variant)
            for r in d.itertuples(index=False)
        ]
        self.assertEqual(seeds, d.seed.tolist())

    def test_sdb_generator_patterns(self):
        for variant in SDB_VARIANTS:
            d = sdb(n=60, d_z=5, beta=0.5, seed=123, variant=variant)
            self.assertEqual(d.X.shape, (60, 1))
            self.assertTrue(np.isfinite(d.X).all())
            self.assertEqual(d.Z.shape, (60, 5))

    def test_real_fit_and_determinism(self):
        torch.set_num_threads(1)
        data = pnl(n=80, d_z=3, beta=0.5, seed=173, trial_id=0)
        cfg = QRCITConfig(
            epochs=2,
            patience=2,
            hidden_dims=(8, 8),
            n_folds=2,
            batch_size=32,
            n_null_draws=3,
            u_grid_size=5,
            lambda_mono=0.0,
        )
        a = QRCIT(cfg).safe_test(data)
        b = QRCIT(cfg).safe_test(data)
        self.assertIsNone(a.error, a.error)
        self.assertIsNone(b.error, b.error)
        self.assertTrue(np.isfinite(a.statistic))
        self.assertIn(a.p_value, [0.25, 0.5, 0.75, 1.0])
        self.assertEqual(a.p_value, b.p_value)
        self.assertEqual(a.statistic, b.statistic)
        result = test_independence(data.X, data.Y, data.Z, config=cfg)
        self.assertIsNone(result.error)
        self.assertTrue(0 <= result.p_value <= 1)

    def test_real_regional_calibration(self):
        torch.set_num_threads(1)
        d = sdb(n=100, d_z=3, beta=0.5, seed=123, variant="tail_only")
        cfg = QRCITConfig(
            epochs=2,
            patience=2,
            hidden_dims=(8, 8),
            n_folds=2,
            batch_size=32,
            n_null_draws=3,
            u_grid_size=5,
            lambda_mono=0.0,
            sdb_region_pvalue_enabled=True,
            sdb_region_pvalue_statistics=(
                "deformation_energy",
                "predictive_gain_region",
            ),
        )
        r = QRCIT(cfg).safe_test(d)
        self.assertIsNone(r.error, r.error)
        for key in [
            "p_value_sdb_region_deformation_energy",
            "p_value_sdb_region_predictive_gain_region",
        ]:
            self.assertIn(r.metadata[key], [0.25, 0.5, 0.75, 1.0])


if __name__ == "__main__":
    unittest.main()
