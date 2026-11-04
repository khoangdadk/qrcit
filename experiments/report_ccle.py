"""Summarize CCLE fits after checking their saved null statistics."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ._utils import ROOT, atomic_json
from .ccle import CONFIG, REFERENCE, validate


def make_report(runs=REFERENCE, config_path=CONFIG, seed=None):
    """Read one completed fit per configured gene, in report order."""
    config = json.loads(Path(config_path).read_text())
    seed = config["model"]["seed"] if seed is None else seed
    rows = []
    for feature in config["default_features"]:
        path = Path(runs) / f"{feature}_seed{seed}.json"
        record = json.loads(path.read_text())
        validate(record)
        identity = record["identity"]
        if identity["feature"] != feature or identity["seed"] != seed:
            raise ValueError(f"Wrong feature/seed in {path}")
        expected = dict(config["model"], seed=seed, device="cpu")
        if identity["model_config"] != expected:
            raise ValueError(f"Configuration mismatch in {path}")
        p = record["result"]["p_global"]
        rows.append(
            dict(
                gene=feature.removesuffix("_MUT"),
                feature=feature,
                seed=seed,
                p_value=p,
                reject_at_005=p < 0.05,
                n_null_draws=expected["n_null_draws"],
                K=expected["u_grid_size"],
                n_folds=expected["n_folds"],
                source_file=(
                    path.relative_to(ROOT).as_posix()
                    if path.is_relative_to(ROOT)
                    else str(path)
                ),
            )
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        type=Path,
        default=REFERENCE,
        help="Completed fits; defaults to bundled seed-0 results",
    )
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out", type=Path, default=Path("artifacts/ccle"))
    args = parser.parse_args()
    report = make_report(args.runs, args.config, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out / "pvalues.csv", index=False, float_format="%.6f")
    atomic_json(
        args.out / "summary.json",
        dict(
            source="saved fits; p-values recomputed from retained null statistics",
            results=report.to_dict("records"),
        ),
    )
    print(report[["gene", "seed", "p_value", "reject_at_005"]].to_string(index=False))
    print(f"Saved {args.out / 'pvalues.csv'}")


if __name__ == "__main__":
    main()
