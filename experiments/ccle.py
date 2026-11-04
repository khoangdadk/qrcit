"""Test mutation indicators against PLX4720 response in CCLE."""

import argparse
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from threadpoolctl import threadpool_limits

from . import _ccle_model as model
from ._utils import ROOT, atomic_json, digest

CONFIG = ROOT / "configs/ccle.json"
DATA = ROOT / "data/ccle/ccle_plx4720.npz"
REFERENCE = ROOT / "results/ccle/seed0"


def validate(record):
    """Check complete fold arrays and recompute the randomization p-value."""
    cfg, result = record["identity"]["model_config"], record["result"]
    null = np.asarray(record["null_global"], dtype=float)
    contrast = np.asarray(record["fold_contrast"], dtype=float)
    factual = np.asarray(record["fold_factual"], dtype=float)
    shape = (cfg["n_folds"], cfg["u_grid_size"])
    if (
        null.shape != (cfg["n_null_draws"],)
        or contrast.shape != shape
        or factual.shape != shape
    ):
        raise ValueError("Incomplete CCLE null draws or fold curves")
    if not all(
        np.isfinite(x).all() for x in (null, contrast, factual)
    ) or not np.isfinite(result["T_global_obs"]):
        raise ValueError("Non-finite CCLE output")
    p = (1 + int(np.sum(null >= result["T_global_obs"]))) / (len(null) + 1)
    if p != result["p_global"] or result["n"] != 474:
        raise ValueError("Invalid CCLE calibration/cohort")
    if (
        len(record["train_logs"]) != cfg["n_folds"]
        or len(record["curves"]) != cfg["u_grid_size"]
    ):
        raise ValueError("Incomplete CCLE training logs or curve summary")
    frame = pd.DataFrame(record["curves"])
    for key, expected in [
        ("contrast_mean", contrast.mean(axis=0)),
        ("contrast_std_across_folds", contrast.std(axis=0)),
    ]:
        np.testing.assert_allclose(frame[key], expected, rtol=2e-6, atol=2e-6)


def run_one(feature, config, data, data_path, out, threads):
    """Fit one feature, or reuse a complete result with the same identity."""
    seed = int(config["seed"])
    core = ROOT / "qrcit"
    identity = dict(
        feature=feature,
        seed=seed,
        model_config=config,
        data_sha256=digest(data_path),
        model_sha256=digest(model.__file__),
        runner_sha256=digest(__file__),
        core_sha256={
            name: digest(core / name)
            for name in [
                "models.py",
                "training.py",
                "preprocessing.py",
                "losses.py",
                "calibration.py",
            ]
        },
        python=platform.python_version(),
        torch=torch.__version__,
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        scipy=scipy.__version__,
        pandas=pd.__version__,
        threads=threads,
    )
    path = out / f"{feature}_seed{seed}.json"
    if path.exists():
        record = json.loads(path.read_text())
        if record["identity"] != identity:
            raise ValueError(f"Incompatible existing CCLE output: {path}")
        validate(record)
        print(f"Reusing {path}", flush=True)
        return record

    row, curves, contrast, factual, null, logs = model.run_feature(
        data["X_selected"],
        data["y_response"],
        data["feature_names"],
        feature,
        config,
        seed=seed,
        verbose=True,
    )
    record = dict(
        status="fresh_model_fit",
        identity=identity,
        result=row,
        curves=curves.to_dict("records"),
        fold_contrast=contrast.tolist(),
        fold_factual=factual.tolist(),
        null_global=null.tolist(),
        train_logs=logs,
    )
    validate(record)
    atomic_json(path, record)
    print(f"{feature} seed={seed}: p={row['p_global']:.6f}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument(
        "--features",
        nargs="+",
        help="Mutation features; defaults to all ten report genes",
    )
    parser.add_argument(
        "--seed", type=int, help="Override the seed for every selected feature"
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("runs/ccle"))
    args = parser.parse_args()
    plan = json.loads(args.config.read_text())
    features = args.features or plan["default_features"]
    if args.threads < 1 or len(features) != len(set(features)):
        parser.error("Require positive threads and distinct feature names")

    config = dict(plan["model"])
    if args.seed is not None:
        config["seed"] = args.seed
    config["device"] = "cpu"
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    with (
        np.load(args.data, allow_pickle=False) as data,
        threadpool_limits(limits=args.threads),
    ):
        if data["X_selected"].shape != (474, 75) or len(data["feature_names"]) != 75:
            raise ValueError("Expected the 474 x 75 CCLE matrix")
        if not set(features).issubset(data["feature_names"]):
            parser.error("Unknown mutation feature")
        rows = []
        for feature in features:
            record = run_one(feature, config, data, args.data, args.out, args.threads)
            rows.append(
                dict(
                    feature=feature,
                    seed=config["seed"],
                    p_value=record["result"]["p_global"],
                )
            )
    report = pd.DataFrame(rows)
    report.to_csv(args.out / "pvalues.csv", index=False)
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
