"""Tune XGBoost once per gene set and evaluate 50 train/test splits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ._metabric_model import CDCIT_GENES, CLINICAL, load_data
from ._utils import ROOT, atomic_json, digest
from .metabric import DATA, GENES, select_genes

DOWNSTREAM_CONFIG = ROOT / "configs/metabric_downstream.json"
HISTORICAL = ROOT / "results/metabric/downstream_summary.csv"


def prediction_frame(df, genes):
    if len(set(genes)) != len(genes) or not set(genes).issubset(df.columns):
        raise ValueError("Unknown or duplicate selected gene")
    x = df[genes + CLINICAL].copy()
    for gene in genes:
        x[gene] = pd.to_numeric(x[gene], errors="coerce")
    y = pd.to_numeric(df.tumor_size, errors="raise").to_numpy(dtype=float)
    return x, y


def pipeline(genes, params, seed, threads):
    from xgboost import XGBRegressor

    numeric = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    prep = ColumnTransformer([("num", numeric, genes), ("cat", categorical, CLINICAL)])
    return Pipeline(
        [
            ("prep", prep),
            (
                "model",
                XGBRegressor(**{**params, "random_state": seed, "n_jobs": threads}),
            ),
        ]
    )


def standardize(train, test):
    mean, std = float(np.mean(train)), max(float(np.std(train)), 1e-8)
    return (train - mean) / std, (test - mean) / std, mean, std


def tune_once(df, genes, cfg, threads):
    x, y = prediction_frame(df, genes)
    xt, xu, yt, yu = train_test_split(
        x, y, train_size=cfg["tune_train_fraction"], random_state=cfg["tune_seed"]
    )
    yt, _, _, _ = standardize(yt, yu)
    # Keep the total parallel budget bounded: parallel CV fits, one XGB thread each.
    grid = GridSearchCV(
        pipeline(genes, cfg["base_params"], cfg["tune_seed"], 1),
        param_grid={"model__" + k: v for k, v in cfg["grid"].items()},
        scoring="neg_mean_squared_error",
        cv=cfg["cv_folds"],
        n_jobs=threads,
        return_train_score=True,
    )
    grid.fit(xt, yt)
    params = {
        **cfg["base_params"],
        **{k.removeprefix("model__"): v for k, v in grid.best_params_.items()},
    }
    return params, float(-grid.best_score_), pd.DataFrame(grid.cv_results_)


def evaluate_repeat(df, genes, params, repeat, cfg, threads):
    x, y = prediction_frame(df, genes)
    seed = cfg["seed"] + repeat
    tr, te = train_test_split(
        np.arange(len(x)), test_size=cfg["test_fraction"], random_state=seed
    )
    yt, ytest, mean, std = standardize(y[tr], y[te])
    fitted = pipeline(genes, params, seed, threads).fit(x.iloc[tr], yt)
    prediction = fitted.predict(x.iloc[te])
    mse = float(mean_squared_error(ytest, prediction))
    baseline = float(mean_squared_error(ytest, np.full_like(ytest, float(np.mean(yt)))))
    return dict(
        repeat=repeat,
        seed=seed,
        mse=mse,
        baseline_mse=baseline,
        mse_over_baseline=mse / baseline,
        raw_mse=float(mean_squared_error(y[te], prediction * std + mean)),
        target_train_mean=mean,
        target_train_std=std,
        test_indices=te.tolist(),
        y_test=ytest.tolist(),
        prediction=prediction.tolist(),
    )


def validate_repeat(row, repeat, cfg):
    if row["repeat"] != repeat or row["seed"] != cfg["seed"] + repeat:
        raise ValueError("Repeat/seed mismatch")
    y, pred = np.asarray(row["y_test"]), np.asarray(row["prediction"])
    if y.shape != pred.shape or y.ndim != 1 or len(y) != len(row["test_indices"]):
        raise ValueError("Invalid prediction shape")
    if not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("Nonfinite prediction")
    if abs(float(np.mean((y - pred) ** 2)) - row["mse"]) > 1e-10:
        raise ValueError("Saved predictions do not match MSE")


def main():
    import sklearn
    import xgboost

    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--screen", type=Path, help="Fresh screen.json from collect_metabric"
    )
    source.add_argument("--archived-selection", action="store_true")
    p.add_argument(
        "--parameters", choices=["tune-once", "archived"], default="tune-once"
    )
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--genes", type=Path, default=GENES)
    p.add_argument("--config", type=Path, default=DOWNSTREAM_CONFIG)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.threads < 1 or (
        args.parameters == "archived" and not args.archived_selection
    ):
        p.error(
            "threads must be positive; archived parameters require --archived-selection"
        )
    cfg = json.loads(args.config.read_text())
    if cfg["n_repeats"] < 2:
        p.error("At least two repeats are required to report SD")
    df, all_genes = load_data(args.data, args.genes)
    historical = pd.read_csv(HISTORICAL).set_index("gene_set")
    if args.archived_selection:
        sets = {
            name: row.selected_genes.split(";") for name, row in historical.iterrows()
        }
        selection_source = dict(kind="archived", sha256=digest(HISTORICAL))
    else:
        screen = json.loads(args.screen.read_text())
        if screen["kind"] != "fresh_metabric_screen" or screen["identity"][
            "data_sha256"
        ] != digest(args.data):
            raise ValueError("Selection does not match this cohort")
        values = screen["p_values"]
        if len(values) != 169 or {v["gene"] for v in values} != set(all_genes):
            raise ValueError("Incomplete fresh gene screen")
        if any(
            not np.isfinite(v["p_value"]) or not 0 <= v["p_value"] <= 1 for v in values
        ):
            raise ValueError("Invalid screening p-values")
        sets = select_genes(
            [dict(gene=v["gene"], neural_cfit_p_value=v["p_value"]) for v in values]
        )
        if sets != screen["selected_genes"]:
            raise ValueError("Selected sets do not match screening p-values")
        sets["CDCIT_reported_8"] = CDCIT_GENES
        selection_source = dict(kind="fresh", sha256=digest(args.screen))
    identity = dict(
        data_sha256=digest(args.data),
        genes_sha256=digest(args.genes),
        config_sha256=digest(args.config),
        code_sha256=digest(__file__),
        loader_sha256=digest(Path(__file__).with_name("_metabric_model.py")),
        selection=selection_source,
        parameters=args.parameters,
        threads=args.threads,
        xgboost=xgboost.__version__,
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    summaries = []
    for name, genes in sets.items():
        directory = args.out / name
        tuning_path = directory / "parameters.json"
        if tuning_path.exists():
            tuning = json.loads(tuning_path.read_text())
            if (
                tuning["identity"] != identity
                or tuning["genes"] != genes
                or tuning["config"] != cfg
            ):
                raise ValueError(f"Incompatible existing parameters: {tuning_path}")
            params = tuning["params"]
        else:
            if args.parameters == "archived":
                params = json.loads(historical.loc[name, "xgb_params_used"])
                cv_mse = float(historical.loc[name, "xgb_tune_once_best_cv_mse"])
            else:
                print(f"{name}: tuning once", flush=True)
                params, cv_mse, cv_results = tune_once(df, genes, cfg, args.threads)
                directory.mkdir(parents=True, exist_ok=True)
                cv_results.to_csv(directory / "cv_results.csv", index=False)
            params.pop("n_jobs", None)  # Runtime thread budget is recorded separately.
            tuning = dict(
                identity=identity,
                config=cfg,
                genes=genes,
                params=params,
                best_cv_mse=cv_mse,
            )
            atomic_json(tuning_path, tuning)
        rows = []
        for repeat in range(cfg["n_repeats"]):
            path = directory / f"repeat_{repeat:03}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if (
                    record["identity"] != identity
                    or record["params"] != params
                    or record["genes"] != genes
                ):
                    raise ValueError(f"Incompatible existing repeat: {path}")
                row = record["result"]
            else:
                row = evaluate_repeat(df, genes, params, repeat, cfg, args.threads)
                validate_repeat(row, repeat, cfg)
                atomic_json(
                    path,
                    dict(identity=identity, params=params, genes=genes, result=row),
                )
            validate_repeat(row, repeat, cfg)
            rows.append(row)
            print(
                f"{name} {repeat + 1}/{cfg['n_repeats']}: MSE={row['mse']:.6f}",
                flush=True,
            )
        frame = pd.DataFrame(rows)
        frame.drop(columns=["test_indices", "y_test", "prediction"]).to_csv(
            directory / "repeats.csv", index=False
        )
        summaries.append(
            dict(
                gene_set=name,
                n_genes=len(genes),
                n_repeats=len(rows),
                mean_mse=float(frame.mse.mean()),
                std_mse=float(frame.mse.std(ddof=1)),
                archived_mean_mse=float(historical.loc[name, "mean_mse"]),
                difference_from_archive=float(
                    frame.mse.mean() - historical.loc[name, "mean_mse"]
                ),
                parameters=args.parameters,
                selection=selection_source["kind"],
            )
        )
    result = pd.DataFrame(summaries)
    result.to_csv(args.out / "summary.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
