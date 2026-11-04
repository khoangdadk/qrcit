"""Exercise real fitting, atomic output and resume through the public CLI."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class RunnerChecks(unittest.TestCase):
    def test_cli_resume_preserves_completed_trial(self):
        config = json.loads((ROOT / "configs/scalar.json").read_text())
        config.update(
            epochs=1,
            patience=1,
            hidden_dims=[8, 8],
            n_folds=2,
            n_null_draws=3,
            batch_size=32,
            lambda_mono=0.0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            cfg = tmp / "config.json"
            cfg.write_text(json.dumps(config))
            output = tmp / "run"
            command = [
                sys.executable,
                "-m",
                "experiments.run",
                "--benchmark",
                "pnl",
                "--n",
                "80",
                "--d-z",
                "3",
                "--K",
                "5",
                "--trials",
                "1",
                "--betas",
                "0.5",
                "--config",
                str(cfg),
                "--out",
                str(output),
            ]
            first = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            before = (output / "trials.csv").read_bytes()
            second = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(before, (output / "trials.csv").read_bytes())
            frame = pd.read_csv(output / "trials.csv")
            self.assertEqual(len(frame), 1)
            self.assertTrue(frame.error.isna().all())
            self.assertTrue(frame.p_global.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
