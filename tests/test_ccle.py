"""Check saved CCLE p-values, fold arrays and the default reporting profile."""

import copy
import json
import unittest

import numpy as np

from experiments.ccle import REFERENCE, validate
from experiments.plot_ccle import read_records
from experiments.report_ccle import make_report


class CCLEChecks(unittest.TestCase):
    def test_saved_records_and_calibration(self):
        records = list(REFERENCE.glob("*_seed0.json"))
        self.assertEqual(len(records), 10)
        for path in records:
            record = json.loads(path.read_text())
            validate(record)
        broken = copy.deepcopy(record)
        broken["result"]["p_global"] = 0.5
        with self.assertRaises(ValueError):
            validate(broken)
        broken = copy.deepcopy(record)
        broken["fold_contrast"].pop()
        with self.assertRaises(ValueError):
            validate(broken)

    def test_seed0_report_and_plot_selection(self):
        report = make_report()
        self.assertEqual(
            report.gene.tolist(),
            [
                "BRAF.V600E",
                "BRAF.MC",
                "HIP1",
                "CDC42BPA",
                "MAP3K5",
                "PRKD1",
                "FLT3",
                "PIP5K1A",
                "DNMT1",
                "THBS3",
            ],
        )
        self.assertTrue(report.seed.eq(0).all())
        np.testing.assert_array_equal(
            np.round(report.p_value, 6),
            [
                0.013986,
                0.007992,
                0.074925,
                0.027972,
                1.0,
                0.012987,
                0.912088,
                0.117882,
                0.956044,
                0.901099,
            ],
        )
        records = read_records([REFERENCE])
        self.assertEqual(set(records), {"HIP1_MUT", "PIP5K1A_MUT"})
        self.assertEqual({r["identity"]["seed"] for r in records.values()}, {0})
        self.assertAlmostEqual(
            records["HIP1_MUT"]["result"]["p_global"], 0.07492507492507493
        )


if __name__ == "__main__":
    unittest.main()
