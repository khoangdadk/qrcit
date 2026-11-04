"""Screen METABRIC gene expression with cross-fitted quantile models."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from . import _metabric_model as model
from ._utils import ROOT, atomic_json, digest

DATA = ROOT / "data/metabric/cohort.csv"
GENES = ROOT / "data/metabric/genes.txt"
CONFIG = ROOT / "configs/metabric.json"
THRESHOLDS = {
    "Neural_CFIT_p001": 0.01,
    "Neural_CFIT_p005": 0.05,
    "Neural_CFIT_p010": 0.10,
}


def load_config(path):
    cfg = json.loads(path.read_text())
    for key in [
        "n_splits",
        "n_null_draws",
        "quantiles",
        "batch_size",
        "q0_epochs",
        "deformation_epochs",
        "q0_patience",
        "deformation_patience",
        "n_fourier",
        "n_rbf",
    ]:
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"Invalid {key}")
    if (
        cfg["n_splits"] < 2
        or cfg["quantiles"] < 2
        or not 0 < cfg["validation_fraction"] < 1
    ):
        raise ValueError("Invalid folds, quantiles or validation fraction")
    return cfg


def make_identity(config, data, genes, device, threads):
    import platform

    import pandas
    import scipy
    import sklearn

    return dict(
        config_sha256=digest(config),
        data_sha256=digest(data),
        genes_sha256=digest(genes),
        runner_sha256=digest(__file__),
        model_sha256=digest(model.__file__),
        torch=torch.__version__,
        numpy=np.__version__,
        sklearn=sklearn.__version__,
        pandas=pandas.__version__,
        scipy=scipy.__version__,
        python=platform.python_version(),
        device=str(device),
        threads=threads,
    )


def validate_record(record, gene, gene_index, cfg):
    r = record["result"]
    if r["gene"] != gene or r["gene_index"] != gene_index or r["status"] != "ok":
        raise ValueError("Wrong gene identity/status")
    if any(r[k] != cfg[k] for k in ["seed", "n_splits", "n_null_draws"]):
        raise ValueError("Wrong gene configuration")
    null = np.asarray(r["null_stats"], dtype=float)
    if (
        null.shape != (cfg["n_null_draws"],)
        or not np.isfinite(null).all()
        or not np.isfinite(r["neural_T_obs"])
    ):
        raise ValueError("Missing or invalid null statistics")
    p = (1 + np.sum(null >= r["neural_T_obs"])) / (len(null) + 1)
    if abs(p - r["neural_cfit_p_value"]) > 1e-14 or r["n"] != 1491 or r["dim_z"] != 180:
        raise ValueError("Invalid calibration or cohort")
    if len(r["train_logs"]) != cfg["n_splits"]:
        raise ValueError("Incomplete fold training logs")
    curves = record["curves"]
    expected = {
        (fold, float(u))
        for fold in range(cfg["n_splits"])
        for u in np.linspace(0.05, 0.95, cfg["quantiles"])
    }
    observed = {(v["fold_id"], v["u"]) for v in curves}
    if len(curves) != len(expected) or observed != expected:
        raise ValueError("Incomplete or duplicate fold curves")
    if any(v["gene"] != gene or not np.isfinite(v["delta"]) for v in curves):
        raise ValueError("Invalid fold curve")


def select_genes(rows):
    # Alphabetical ties keep downstream feature order deterministic.
    ranked = sorted(rows, key=lambda r: (r["neural_cfit_p_value"], r["gene"]))
    return {
        name: [r["gene"] for r in ranked if r["neural_cfit_p_value"] < p]
        for name, p in THRESHOLDS.items()
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--genes", type=Path, default=GENES)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=169)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, default=ROOT / "runs/metabric/genes")
    args = p.parse_args()
    if not 0 <= args.start < args.end <= 169 or args.threads < 1:
        p.error("Expected 0 <= start < end <= 169 and threads >= 1")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(args.threads)
    cfg = load_config(args.config)
    device = torch.device(args.device)
    df, genes = model.load_data(args.data, args.genes)
    identity = make_identity(args.config, args.data, args.genes, device, args.threads)
    for i in range(args.start, args.end):
        path = args.out / f"gene_{i:03}.json"
        if path.exists():
            record = json.loads(path.read_text())
            if record["identity"] != identity:
                raise ValueError(f"Incompatible existing output: {path}")
            validate_record(record, genes[i], i, cfg)
            print(f"{i}: {genes[i]} reused", flush=True)
            continue
        with threadpool_limits(limits=args.threads):
            result, curves = model.run_gene(df, genes, genes[i], cfg, device)
        result["gene_index"] = i
        record = dict(identity=identity, config=cfg, result=result, curves=curves)
        validate_record(record, genes[i], i, cfg)
        atomic_json(path, record)
        print(
            f"{i}: {genes[i]} p={result['neural_cfit_p_value']:.6f} ({result['runtime_sec']:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
