"""Run one synthetic setting, with native trial seeds and atomic resumable CSVs."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--benchmark",
        choices=["pnl", "sdb", "role-swap", "multivariate"],
        required=True,
    )
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--d-z", type=int, default=5)
    ap.add_argument("--K", type=int, default=9)
    ap.add_argument("--trials", type=int)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int)
    ap.add_argument("--betas", nargs="+", type=float, default=[0.0, 0.5])
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--sweep", choices=["N", "dz"], default="N")
    ap.add_argument("--config", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import numpy as np
    import pandas as pd

    from qrcit import QRCIT, QRCITConfig, SlicedQRCIT, SlicedQRCITConfig
    from qrcit.datasets import SDB_VARIANTS, multivariate, pnl, sdb, trial_seed

    count = (
        args.trials
        if args.trials is not None
        else (20 if args.benchmark == "sdb" else 100)
    )
    end = count if args.end is None else args.end
    if not 0 <= args.start < end <= count:
        ap.error("Require 0 <= start < end <= trials")
    variants = (
        args.variants or list(SDB_VARIANTS) if args.benchmark == "sdb" else [None]
    )
    if args.benchmark == "sdb" and not set(variants) <= set(SDB_VARIANTS):
        ap.error("Unknown SDB variant")
    name = "multivariate.json" if args.benchmark == "multivariate" else "scalar.json"
    config = json.loads((args.config or ROOT / "configs" / name).read_text())
    base = config["base_config"] if args.benchmark == "multivariate" else config
    base["u_grid_size"] = args.K
    if args.benchmark == "sdb":
        base.update(
            sdb_region_pvalue_enabled=True,
            sdb_region_pvalue_decision=False,
            sdb_region_pvalue_statistics=[
                "deformation_energy",
                "predictive_gain_region",
            ],
        )
    code_hash = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted((ROOT / "qrcit").rglob("*.py")))
        + Path(__file__).read_bytes()
    ).hexdigest()
    identity = dict(
        benchmark=args.benchmark,
        n=args.n,
        d_z=args.d_z,
        K=args.K,
        trials=count,
        betas=args.betas,
        variants=variants,
        sweep=args.sweep,
        config=config,
        code_sha256=code_hash,
    )
    environment = {
        k: importlib.metadata.version(k)
        for k in ["numpy", "scipy", "pandas", "torch", "scikit-learn"]
    }
    environment["python"] = sys.version
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = args.out / "manifest.json"
        record = dict(scientific=identity, environment=environment)
        if manifest.exists() and json.loads(manifest.read_text()) != record:
            raise RuntimeError(
                "Resume identity/environment mismatch; use a new output directory"
            )
        manifest.write_text(json.dumps(record, indent=2) + "\n")
        path = args.out / "trials.csv"
        rows = pd.read_csv(path).to_dict("records") if path.exists() else []
        if any(pd.notna(r.get("error")) for r in rows):
            raise RuntimeError(
                "Existing failed trials require inspection before resume"
            )
        done = {
            (float(r["beta"]), int(r["trial_id"]), str(r["variant_key"])) for r in rows
        }
        for variant in variants:
            for beta in args.betas:
                for trial in range(args.start, end):
                    variant_key = variant if variant is not None else "default"
                    key = (float(beta), trial, variant_key)
                    if key in done:
                        continue
                    seed = trial_seed(
                        args.benchmark, args.n, args.d_z, beta, trial, variant
                    )
                    if args.benchmark == "sdb":
                        data = sdb(
                            n=args.n,
                            d_z=args.d_z,
                            beta=beta,
                            seed=seed,
                            variant=variant,
                            trial_id=trial,
                        )
                    elif args.benchmark == "multivariate":
                        data = multivariate(
                            case="multivariate_x_multivariate_y",
                            n=args.n,
                            d_x=5,
                            d_y=5,
                            d_z=args.d_z,
                            beta=beta,
                            seed=seed,
                        )
                    else:
                        data = pnl(
                            n=args.n,
                            d_z=args.d_z,
                            beta=beta,
                            seed=seed,
                            trial_id=trial,
                            families=["pnl_analytic", "pnl_detection"],
                        )
                        if args.benchmark == "role-swap":
                            data.X, data.Y = data.Y, data.X
                            data.delta_true = data.delta_struct = None
                            data.meta.update(
                                swap_xy=True, cfit_target_role="Ytarget_swapped"
                            )
                    method = (
                        SlicedQRCIT(SlicedQRCITConfig(**config))
                        if args.benchmark == "multivariate"
                        else QRCIT(QRCITConfig(**config))
                    )
                    result = method.safe_test(data)
                    row = dict(
                        benchmark=args.benchmark.upper(),
                        sweep=args.sweep,
                        x_value=args.n if args.sweep == "N" else args.d_z,
                        n=args.n,
                        d_x=data.d_x,
                        d_y=data.d_y,
                        d_z=args.d_z,
                        K=args.K,
                        beta=beta,
                        trial_id=trial,
                        seed=seed,
                        variant=data.variant,
                        variant_key=variant_key,
                        method=result.method,
                        p_global=result.p_value,
                        p_region_pg=result.metadata.get(
                            "p_value_sdb_region_predictive_gain_region", np.nan
                        ),
                        p_region_de=result.metadata.get(
                            "p_value_sdb_region_deformation_energy", np.nan
                        ),
                        fit_time_sec=result.fit_time_sec,
                        test_time_sec=result.test_time_sec,
                        calibration_time_sec=result.calibration_time_sec,
                        error=result.error,
                    )
                    rows.append(row)
                    temp = path.with_suffix(".tmp")
                    pd.DataFrame(rows).to_csv(temp, index=False, float_format="%.17g")
                    temp.replace(path)
                    if result.error:
                        raise RuntimeError(result.error)
                    done.add(key)
                    print(
                        f"{args.benchmark} variant={variant} beta={beta} trial={trial} p={result.p_value}",
                        flush=True,
                    )


if __name__ == "__main__":
    main()
