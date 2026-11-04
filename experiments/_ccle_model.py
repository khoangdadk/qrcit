"""CCLE joint-residual quantile model and Bernoulli randomization sampler."""

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from qrcit.calibration import (
    deformation_energy_stat,
    predictive_gain_region_stat,
    predictive_gain_stat,
)
from qrcit.models import QuantileModelConfig
from qrcit.preprocessing import Standardizer
from qrcit.training import (
    TrainConfig,
    predict_q1_for_y,
    predict_quantiles,
    train_quantile_model,
)


@dataclass
class BernoulliConfig:
    """Use an empirical fallback when either training class is too small."""

    min_binary_class_count: int = 5
    p_clip: float = 1e-4


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D/2D array, got shape {x.shape}")
    return x


def _as_scalar_target(x: np.ndarray, name: str) -> np.ndarray:
    x = _as_2d(x)
    if x.shape[1] != 1:
        raise ValueError(
            f"PredictiveGainCRT currently expects scalar {name}; got shape {x.shape}"
        )
    return x.reshape(-1)


class BernoulliResampler:
    def __init__(self, cfg: BernoulliConfig, seed: int = 0):
        self.cfg = cfg
        self.seed = seed
        self.model = None
        self.empirical_p = 0.5
        self.status = "unfit"

    def fit(self, Z: np.ndarray, Y: np.ndarray):
        Z = _as_2d(Z)
        y = _as_scalar_target(Y, "Y")
        vals = np.unique(y[np.isfinite(y)])
        if len(vals) < 2:
            self.empirical_p = float(np.mean(y)) if len(y) else 0.0
            self.model = None
            self.status = "one_class_empirical"
            return self
        # Map the larger raw level to 1 if needed.
        y01 = (y == np.sort(vals)[-1]).astype(int)
        counts = np.bincount(y01, minlength=2)
        self.empirical_p = float(np.mean(y01))
        if np.min(counts) < int(self.cfg.min_binary_class_count):
            self.model = None
            self.status = "rare_class_empirical"
            return self
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=5000,
                random_state=self.seed,
            ),
        )
        self.model.fit(Z, y01)
        self.status = "logistic"
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        Z = _as_2d(Z)
        if self.model is None:
            p = np.full(Z.shape[0], self.empirical_p, dtype=float)
        else:
            p = self.model.predict_proba(Z)[:, 1]
        return np.clip(p, float(self.cfg.p_clip), 1.0 - float(self.cfg.p_clip))

    def sample(
        self, Z: np.ndarray, n_draws: int = 1, seed: int | None = None
    ) -> np.ndarray:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        p = self.predict_proba(Z)
        draws = rng.binomial(n=1, p=p[None, :], size=(int(n_draws), len(p)))
        return draws[:, :, None].astype(np.float64)

    def diagnostics(self, Z: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
        y = _as_scalar_target(Y, "Y")
        p = self.predict_proba(Z)
        return {
            "resampler_type": "bernoulli",
            "resampler_status": self.status,
            "obs_mean": float(np.mean(y)),
            "pred_mean": float(np.mean(p)),
            "pred_min": float(np.min(p)),
            "pred_max": float(np.max(p)),
        }


def feature_arrays(X_selected, y_response, feature_names_selected, feature_name):
    feature_names_selected = list(map(str, feature_names_selected))
    if feature_name not in feature_names_selected:
        raise KeyError(f"{feature_name} not found in selected feature matrix.")
    j = feature_names_selected.index(feature_name)
    X_resp = np.asarray(y_response, dtype=np.float32).reshape(-1, 1)
    Y_feat = np.asarray(X_selected[:, j], dtype=np.float32).reshape(-1, 1)
    Z_ctx = np.delete(np.asarray(X_selected, dtype=np.float32), j, axis=1)
    return X_resp, Y_feat, Z_ctx


def is_binary_vector(y):
    vals = np.unique(np.asarray(y).reshape(-1))
    return len(vals) == 2 and set(vals.astype(int).tolist()).issubset({0, 1})


def _model_cfg(d_z, cfg):
    return QuantileModelConfig(
        d_z=int(d_z),
        d_y=1,
        model_type=cfg["model_type"],
        hidden_dims=tuple(cfg["hidden_dims"]),
        dropout=float(cfg["dropout"]),
        u_feature_mode=cfg["u_feature_mode"],
        n_u_fourier=int(cfg["n_u_fourier"]),
        n_u_rbf=int(cfg["n_u_rbf"]),
        u_rbf_width=float(cfg["u_rbf_width"]),
    )


def _train_cfg(seed, cfg):
    return TrainConfig(
        epochs=int(cfg["epochs"]),
        batch_size=int(cfg["batch_size"]),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
        dropout=float(cfg["dropout"]),
        hidden_dims=tuple(cfg["hidden_dims"]),
        lambda_D=float(cfg["lambda_D"]),
        lambda_mono=float(cfg["lambda_mono"]),
        patience=int(cfg["patience"]),
        device=str(cfg["device"]),
        v2_q0_pretrain_epochs=int(cfg["v2_q0_pretrain_epochs"]),
        v2_d_train_epochs=int(cfg["v2_d_train_epochs"]),
        v2_q0_pretrain_patience=int(cfg["v2_q0_pretrain_patience"]),
        v2_d_train_patience=int(cfg["v2_d_train_patience"]),
        v2_freeze_q0=bool(cfg["v2_freeze_q0"]),
        v2_d_l2_penalty=float(cfg["v2_d_l2_penalty"]),
        v2_mono_penalty=float(cfg["v2_mono_penalty"]),
        seed=int(seed),
        deterministic=bool(cfg["deterministic"]),
    )


def _sample_bernoulli_y_std(fd, rng):
    """Sample raw binary Y|Z from fold Bernoulli model, then standardize to fold Y scale."""
    raw = (
        fd["resampler"]
        .sample(
            fd["Z_test_std"],
            n_draws=1,
            seed=int(rng.integers(0, 2**31 - 1)),
        )[0]
        .astype(np.float32)
    )
    stdz = fd["standardizer"]
    return ((raw - stdz.y_mean) / stdz.y_std).astype(np.float32)


def _region_mask(n, u_grid, kind, cfg):
    u = np.asarray(u_grid)
    if kind == "lower_tail":
        m_u = u <= float(cfg["lower_u_max"])
    elif kind == "upper_tail":
        m_u = u >= float(cfg["upper_u_min"])
    else:
        raise ValueError(kind)
    return np.repeat(m_u.reshape(1, -1), int(n), axis=0)


def run_feature(
    X_selected, y_drug, feature_names_selected, feature_name, cfg, seed=0, verbose=False
):
    """Fit Q0 and the residual head jointly, then randomize the mutation feature."""
    X, Y, Z = feature_arrays(X_selected, y_drug, feature_names_selected, feature_name)
    if not is_binary_vector(Y):
        raise ValueError(f"{feature_name} is not binary.")

    y01 = Y.reshape(-1).astype(int)
    counts = np.bincount(y01, minlength=2)
    if counts.min() < 5:
        raise ValueError(
            f"{feature_name}: too few samples in one class: n0={counts[0]}, n1={counts[1]}"
        )

    n = X.shape[0]
    u_grid = np.linspace(
        float(cfg["u_grid_min"]),
        float(cfg["u_grid_max"]),
        int(cfg["u_grid_size"]),
        dtype=np.float32,
    )
    q0_full = np.zeros((n, len(u_grid)), dtype=np.float32)
    q1_full = np.zeros((n, len(u_grid)), dtype=np.float32)
    delta_full = np.zeros((n, len(u_grid)), dtype=np.float32)
    Xstd_full = np.zeros_like(X, dtype=np.float32)
    folds = []

    fold_curve_contrast = []
    fold_curve_factual = []
    train_logs = []

    fit_time_total = 0.0
    test_time_total = 0.0

    kf = KFold(n_splits=int(cfg["n_folds"]), shuffle=True, random_state=int(seed))
    for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(n))):
        tr_idx, val_idx = train_test_split(
            trainval_idx,
            test_size=float(cfg["inner_val_frac"]),
            random_state=int(seed) + fold_id,
        )
        stdz = Standardizer.fit(X[tr_idx], Y[tr_idx], Z[tr_idx])
        Xtr, Ytr, Ztr = stdz.transform(X[tr_idx], Y[tr_idx], Z[tr_idx])
        Xv, Yv, Zv = stdz.transform(X[val_idx], Y[val_idx], Z[val_idx])
        Xte, Yte, Zte = stdz.transform(X[test_idx], Y[test_idx], Z[test_idx])

        # Binary mutation conditional-randomization model, fitted on standardized Z and raw 0/1 Y.
        resampler = BernoulliResampler(
            BernoulliConfig(), seed=int(seed) + 50021 * fold_id
        )
        resampler.fit(Ztr, Y[tr_idx])

        fold_seed = int(seed) + 1009 * fold_id
        model, info = train_quantile_model(
            Xtr,
            Ytr,
            Ztr,
            Xv,
            Yv,
            Zv,
            _model_cfg(d_z=Z.shape[1], cfg=cfg),
            _train_cfg(seed=fold_seed, cfg=cfg),
        )
        train_logs.append(dict(fold_id=fold_id, **info))
        fit_time_total += float(info.get("fit_time_sec", 0.0))

        t0 = time.perf_counter()
        pred = predict_quantiles(model, Xte, Yte, Zte, u_grid, device=cfg["device"])
        test_time_total += time.perf_counter() - t0

        q0_full[test_idx] = pred["q0"]
        q1_full[test_idx] = pred["q1"]
        delta_full[test_idx] = pred["delta"]
        Xstd_full[test_idx] = (X[test_idx] - stdz.x_mean) / stdz.x_std

        # Amax-scale curves.
        fold_curve_factual.append(
            (pred["delta"] * stdz.x_std.reshape(1, -1)[0, 0]).mean(axis=0)
        )
        Y0_std = np.full(
            (len(test_idx), 1),
            (0.0 - stdz.y_mean.reshape(-1)[0]) / stdz.y_std.reshape(-1)[0],
            dtype=np.float32,
        )
        Y1_std = np.full(
            (len(test_idx), 1),
            (1.0 - stdz.y_mean.reshape(-1)[0]) / stdz.y_std.reshape(-1)[0],
            dtype=np.float32,
        )
        q1_y0 = predict_q1_for_y(model, Y0_std, Zte, u_grid, device=cfg["device"])
        q1_y1 = predict_q1_for_y(model, Y1_std, Zte, u_grid, device=cfg["device"])
        fold_curve_contrast.append(
            ((q1_y1 - q1_y0) * stdz.x_std.reshape(1, -1)[0, 0]).mean(axis=0)
        )

        folds.append(
            dict(
                fold_id=fold_id,
                test_idx=test_idx,
                q0=pred["q0"],
                q1=pred["q1"],
                delta=pred["delta"],
                model=model,
                standardizer=stdz,
                resampler=resampler,
                Z_test_std=Zte,
            )
        )

        if verbose:
            print(
                f'  fold {fold_id}: fit_time={info.get("fit_time_sec", np.nan):.2f}s, resampler={resampler.status}'
            )

    T_obs = predictive_gain_stat(Xstd_full, q0_full, q1_full, u_grid)
    lower_mask = _region_mask(n, u_grid, "lower_tail", cfg)
    upper_mask = _region_mask(n, u_grid, "upper_tail", cfg)
    region_masks = {"lower_tail": lower_mask, "upper_tail": upper_mask}

    T_region_obs = {}
    for region, mask in region_masks.items():
        T_region_obs[f"{region}_pg_obs"] = predictive_gain_region_stat(
            Xstd_full, q0_full, q1_full, u_grid, mask
        )
        T_region_obs[f"{region}_energy_obs"] = deformation_energy_stat(
            q1_full - q0_full, mask
        )

    rng = np.random.default_rng(int(seed) + 99991)
    null_global = []
    null_region_pg = {r: [] for r in region_masks}
    null_region_energy = {r: [] for r in region_masks}

    cal_t0 = time.perf_counter()
    for b in range(int(cfg["n_null_draws"])):
        q1_null_full = np.zeros_like(q1_full)
        for fd in folds:
            idx = fd["test_idx"]
            y_null_std = _sample_bernoulli_y_std(fd, rng)
            q1_null_full[idx] = predict_q1_for_y(
                fd["model"], y_null_std, fd["Z_test_std"], u_grid, device=cfg["device"]
            )
        null_global.append(
            predictive_gain_stat(Xstd_full, q0_full, q1_null_full, u_grid)
        )
        for region, mask in region_masks.items():
            null_region_pg[region].append(
                predictive_gain_region_stat(
                    Xstd_full, q0_full, q1_null_full, u_grid, mask
                )
            )
            null_region_energy[region].append(
                deformation_energy_stat(q1_null_full - q0_full, mask)
            )
    calibration_time = time.perf_counter() - cal_t0

    null_global = np.asarray(null_global, dtype=np.float32)
    p_global = float((1.0 + np.sum(null_global >= T_obs)) / (1.0 + len(null_global)))

    row = {
        "feature": feature_name,
        "gene": feature_name.removesuffix("_MUT").split(".")[0],
        "n": int(n),
        "n0": int(counts[0]),
        "n1": int(counts[1]),
        "u_grid_size": int(len(u_grid)),
        "n_folds": int(cfg["n_folds"]),
        "n_null_draws": int(cfg["n_null_draws"]),
        "T_global_obs": float(T_obs),
        "p_global": p_global,
        "null_global_mean": float(np.mean(null_global)),
        "null_global_std": float(np.std(null_global)),
        "fit_time_sec": float(fit_time_total),
        "test_time_sec": float(test_time_total),
        "calibration_time_sec": float(calibration_time),
        "reported_time_sec": float(fit_time_total + test_time_total),
    }

    for region, mask in region_masks.items():
        arr_pg = np.asarray(null_region_pg[region], dtype=np.float32)
        arr_en = np.asarray(null_region_energy[region], dtype=np.float32)
        obs_pg = T_region_obs[f"{region}_pg_obs"]
        obs_en = T_region_obs[f"{region}_energy_obs"]
        row[f"{region}_pg_obs"] = float(obs_pg)
        row[f"{region}_pg_p"] = float(
            (1.0 + np.sum(arr_pg >= obs_pg)) / (1.0 + len(arr_pg))
        )
        row[f"{region}_pg_null_mean"] = float(np.mean(arr_pg))
        row[f"{region}_pg_null_std"] = float(np.std(arr_pg))
        row[f"{region}_energy_obs"] = float(obs_en)
        row[f"{region}_energy_p"] = float(
            (1.0 + np.sum(arr_en >= obs_en)) / (1.0 + len(arr_en))
        )
        row[f"{region}_energy_null_mean"] = float(np.mean(arr_en))
        row[f"{region}_energy_null_std"] = float(np.std(arr_en))

    contrast = np.vstack(fold_curve_contrast)
    factual = np.vstack(fold_curve_factual)
    curve_df = pd.DataFrame(
        {
            "feature": feature_name,
            "u": u_grid,
            "contrast_mean": contrast.mean(axis=0),
            "contrast_std_across_folds": contrast.std(axis=0),
            "factual_mean": factual.mean(axis=0),
            "factual_std_across_folds": factual.std(axis=0),
        }
    )

    return row, curve_df, contrast, factual, null_global, train_logs
