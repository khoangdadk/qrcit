"""Guard the recovered Sachs result against accidental data or threshold changes."""

import unittest

from experiments.verify_sachs import verify


class SachsArchiveChecks(unittest.TestCase):
    def test_original_paper_row(self):
        result = verify()
        self.assertEqual(
            (result["TP"], result["FP"], result["FN"], result["TN"]), (44, 16, 6, 24)
        )
        self.assertEqual(result["F1"], 0.8)


if __name__ == "__main__":
    unittest.main()
