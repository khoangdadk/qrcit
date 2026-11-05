"""Cross-fitted conditional quantile testing and structural diagnostics."""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np
from sklearn.model_selection import KFold, train_test_split

from qrcit.base import CITMethod, MethodResult
from qrcit.data import CITData
from qrcit.datasets.sdb import sdb_region_mask_grid

from .calibration import (
    conditional_randomization_pvalues_from_folds,
    predictive_gain_stat,
)
from .diagnostics import deformation_diagnostics, sdb_struct_diagnostics
from .models import QuantileModelConfig
from .preprocessing import Standardizer
from .sampling import build_y_sampler
from .training import TrainConfig, predict_quantiles, train_quantile_model


@dataclass
class QuantileTestConfig:
    """Network, cross-fitting and conditional-randomization settings."""

    model_type: str = "residual"
    eval_mode: str = "crossfit"
    test_stat_mode: str = "predictive_gain"
    calibration_mode: str = "conditional_randomization"
    y_resampler: str = "learned_gaussian"
    n_folds: int = 3
    inner_val_frac: float = 0.20
    n_null_draws: int = 20
    u_grid_min: float = 0.05
    u_grid_max: float = 0.95
    u_grid_size: int = 19
    seed: int = 0
    deterministic: bool = True
    # Shared optimizer and network settings.
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.05
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    lambda_D: float = 1e-3
    lambda_mono: float = 1e-2
    patience: int = 20
    device: str = "auto"
    # Quantile-level features.
    u_feature_mode: str = "qnorm"
    n_u_fourier: int = 4
    n_u_rbf: int = 12
    u_rbf_width: float = 0.10
    # Used only when model_type is residual_deformation_v2.
    v2_q0_pretrain_epochs: int = 60
    v2_d_train_epochs: int = 100
    v2_q0_pretrain_patience: int = 15
    v2_d_train_patience: int = 20
    v2_freeze_q0: bool = True
    v2_d_l2_penalty: float = 1e-4
    v2_mono_penalty: float = 1e-3
    # SDB diagnostics use the known affected region of the synthetic generator.
    sdb_region_pvalue_enabled: bool = False
    sdb_region_pvalue_decision: bool = False
    # Single-statistic default when no statistics are listed below.
    sdb_region_pvalue_statistic: str = "deformation_energy"
    # Evaluate several region statistics with the same randomization draws.
    sdb_region_pvalue_statistics: tuple[str, ...] = ()
    # Optional region statistic used for the diagnostic decision.
    sdb_region_pvalue_decision_statistic: str = "predictive_gain_region"
    sdb_region_alpha: float = 0.05


class QuantileTest(CITMethod):
    """Learn baseline/augmented quantiles on training folds and test held-out gain."""

    name = "cfit"

    def __init__(
        self, config: QuantileTestConfig | dict | None = None, alpha: float = 0.05
    ):
        super().__init__(alpha=alpha)
        if config is None:
            self.config = QuantileTestConfig()
        elif isinstance(config, dict):
            self.config = QuantileTestConfig(**config)
        else:
            self.config = config
        self.name = f"cfit_{self.config.model_type}"

    @property
    def u_grid(self) -> np.ndarray:
        c = self.config
        return np.linspace(
            float(c.u_grid_min),
            float(c.u_grid_max),
            int(c.u_grid_size),
            dtype=np.float32,
        )

    def _train_cfg(self) -> TrainConfig:
        c = self.config
        return TrainConfig(
            epochs=int(c.epochs),
            batch_size=int(c.batch_size),
            lr=float(c.lr),
            weight_decay=float(c.weight_decay),
            dropout=float(c.dropout),
            hidden_dims=tuple(c.hidden_dims),
            lambda_D=float(c.lambda_D),
            lambda_mono=float(c.lambda_mono),
            patience=int(c.patience),
            device=str(c.device),
            v2_q0_pretrain_epochs=int(c.v2_q0_pretrain_epochs),
            v2_d_train_epochs=int(c.v2_d_train_epochs),
            v2_q0_pretrain_patience=int(c.v2_q0_pretrain_patience),
            v2_d_train_patience=int(c.v2_d_train_patience),
            v2_freeze_q0=bool(c.v2_freeze_q0),
            v2_d_l2_penalty=float(c.v2_d_l2_penalty),
            v2_mono_penalty=float(c.v2_mono_penalty),
            seed=int(c.seed),
            deterministic=bool(c.deterministic),
        )

    def _model_cfg(self, d_z: int, d_y: int = 1) -> QuantileModelConfig:
        c = self.config
        return QuantileModelConfig(
            d_z=int(d_z),
            d_y=int(d_y),
            model_type=str(c.model_type),
            hidden_dims=tuple(c.hidden_dims),
            dropout=float(c.dropout),
            u_feature_mode=str(c.u_feature_mode),
            n_u_fourier=int(c.n_u_fourier),
            n_u_rbf=int(c.n_u_rbf),
            u_rbf_width=float(c.u_rbf_width),
        )

    def test(self, data: CITData) -> MethodResult:
        if data.d_x != 1:
            raise ValueError("X must be scalar; use SlicedQRCIT for multivariate X.")
        if self.config.eval_mode != "crossfit":
            raise NotImplementedError("Only eval_mode='crossfit' is supported.")
        if self.config.test_stat_mode != "predictive_gain":
            raise NotImplementedError(
                "Only test_stat_mode='predictive_gain' is supported."
            )
        if self.config.calibration_mode != "conditional_randomization":
            raise NotImplementedError(
                "Only conditional-randomization calibration is supported."
            )

        result = self._empty_result(data)
        result.method = self.name
        X, Y, Z = (
            data.X.astype(np.float32),
            data.Y.astype(np.float32),
            data.Z.astype(np.float32),
        )
        n = data.n
        u_grid = self.u_grid
        folds = []
        q0_full = np.zeros((n, len(u_grid)), dtype=np.float32)
        q1_full = np.zeros((n, len(u_grid)), dtype=np.float32)
        delta_full = np.zeros((n, len(u_grid)), dtype=np.float32)
        fit_time_total = 0.0
        test_time_total = 0.0
        train_cfg = self._train_cfg()

        kf = KFold(
            n_splits=int(self.config.n_folds),
            shuffle=True,
            random_state=int(self.config.seed) + int(data.seed),
        )
        for fold_id, (trainval_idx, test_idx) in enumerate(kf.split(np.arange(n))):
            tr_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=float(self.config.inner_val_frac),
                random_state=int(self.config.seed) + int(data.seed) + fold_id,
            )
            stdz = Standardizer.fit(X[tr_idx], Y[tr_idx], Z[tr_idx])
            Xtr, Ytr, Ztr = stdz.transform(X[tr_idx], Y[tr_idx], Z[tr_idx])
            Xv, Yv, Zv = stdz.transform(X[val_idx], Y[val_idx], Z[val_idx])
            Xte, Yte, Zte = stdz.transform(X[test_idx], Y[test_idx], Z[test_idx])

            fold_seed = int(self.config.seed) + int(data.seed) + 1009 * int(fold_id)
            sampler = build_y_sampler(
                self.config.y_resampler,
                seed=fold_seed + 50_003,
                data=data,
                standardizer=stdz,
            )
            sampler.fit(Ytr, Ztr)

            mc = self._model_cfg(d_z=data.d_z, d_y=data.d_y)
            fold_train_cfg = replace(train_cfg, seed=fold_seed)
            model, info = train_quantile_model(
                Xtr, Ytr, Ztr, Xv, Yv, Zv, mc, fold_train_cfg
            )
            fit_time_total += float(info.get("fit_time_sec", 0.0))
            t0 = time.perf_counter()
            pred = predict_quantiles(
                model, Xte, Yte, Zte, u_grid, device=fold_train_cfg.device
            )
            test_time_total += time.perf_counter() - t0
            q0_full[test_idx] = pred["q0"]
            q1_full[test_idx] = pred["q1"]
            delta_full[test_idx] = pred["delta"]
            folds.append(
                {
                    "fold_id": fold_id,
                    "test_idx": test_idx,
                    "q0": pred["q0"],
                    "q1": pred["q1"],
                    "delta": pred["delta"],
                    "model": model,
                    "standardizer": stdz,
                    "sampler": sampler,
                    "Z_test_std": Zte,
                }
            )

        # X is standardized per fold; recreate fold-standardized X vector for statistic.
        Xstd_full = np.zeros_like(X)
        for fd in folds:
            idx = fd["test_idx"]
            # Each held-out row uses its own training fold's X scale.
            Xstd_full[idx] = (X[idx] - fd["standardizer"].x_mean) / fd[
                "standardizer"
            ].x_std

        # Held-out deformation in the original units of X (undo each fold's X scale).
        deformation = np.zeros_like(delta_full)
        for fd in folds:
            deformation[fd["test_idx"]] = (
                np.asarray(fd["delta"], dtype=np.float32) * fd["standardizer"].x_std
            )

        stat = predictive_gain_stat(Xstd_full, q0_full, q1_full, u_grid)
        region_mask = None
        region_info = {}
        if bool(self.config.sdb_region_pvalue_enabled) and str(data.benchmark) == "sdb":
            region_mask, region_info = sdb_region_mask_grid(data, u_grid)

        region_statistics = self.config.sdb_region_pvalue_statistics
        if isinstance(region_statistics, str):
            region_statistics = tuple(
                x.strip() for x in region_statistics.split(",") if x.strip()
            )
        elif region_statistics is None:
            region_statistics = ()
        else:
            region_statistics = tuple(region_statistics)
        if len(region_statistics) == 0:
            region_statistics = (str(self.config.sdb_region_pvalue_statistic),)

        decision_region_statistic = str(
            self.config.sdb_region_pvalue_decision_statistic
        )
        if decision_region_statistic == "" or decision_region_statistic == "None":
            decision_region_statistic = region_statistics[0]
        if bool(self.config.sdb_region_pvalue_enabled) and str(data.benchmark) == "sdb":
            if decision_region_statistic not in region_statistics:
                region_statistics = tuple(
                    list(region_statistics) + [decision_region_statistic]
                )

        cal = conditional_randomization_pvalues_from_folds(
            Xstd_full,
            Z,
            folds,
            None,
            u_grid,
            int(self.config.n_null_draws),
            seed=int(data.seed) + int(self.config.seed),
            device=train_cfg.device,
            region_mask=region_mask,
            region_statistics=region_statistics,
            decision_region_statistic=decision_region_statistic,
        )
        pval = float(cal["p_value"])
        null_stats = np.asarray(cal["null_stats"], dtype=np.float32)
        cal_time = float(cal["calibration_time_sec"])

        p_value_sdb_region = float(cal.get("p_value_sdb_region", float("nan")))
        decision_statistic = str(cal.get("sdb_region_decision_statistic", "global"))
        decision_source = "global"
        decision_p_value = pval
        if (
            bool(self.config.sdb_region_pvalue_enabled)
            and bool(self.config.sdb_region_pvalue_decision)
            and np.isfinite(p_value_sdb_region)
        ):
            decision_source = "sdb_region"
            decision_p_value = p_value_sdb_region

        # Flatten all requested SDB-region p-values and null summaries.
        region_meta = {}
        for st in cal.get("sdb_region_statistics", []):
            safe = str(st)
            p_region = cal.get("sdb_region_p_values", {}).get(st, float("nan"))
            obs_region = cal.get("sdb_region_obs", {}).get(st, float("nan"))
            null_arr = np.asarray(
                cal.get("sdb_region_nulls_by_statistic", {}).get(st, []),
                dtype=np.float32,
            )
            region_meta[f"p_value_sdb_region_{safe}"] = float(p_region)
            region_meta[f"reject_sdb_region_{safe}_alpha_005"] = (
                bool(float(p_region) <= float(self.config.sdb_region_alpha))
                if np.isfinite(float(p_region))
                else None
            )
            region_meta[f"T_sdb_region_{safe}_obs_calib"] = float(obs_region)
            region_meta[f"T_sdb_region_{safe}_null_mean"] = (
                float(np.mean(null_arr)) if len(null_arr) else float("nan")
            )
            region_meta[f"T_sdb_region_{safe}_null_std"] = (
                float(np.std(null_arr)) if len(null_arr) else float("nan")
            )
            region_meta[f"T_sdb_region_{safe}_null_min"] = (
                float(np.min(null_arr)) if len(null_arr) else float("nan")
            )
            region_meta[f"T_sdb_region_{safe}_null_max"] = (
                float(np.max(null_arr)) if len(null_arr) else float("nan")
            )
            region_meta[f"T_sdb_region_{safe}_n_null"] = int(len(null_arr))

        result.statistic = stat
        result.p_value = pval
        result.fit_time_sec = fit_time_total
        result.test_time_sec = test_time_total
        result.calibration_time_sec = cal_time
        result.reported_time_sec = fit_time_total + test_time_total
        result.metadata = {
            "eval_mode": self.config.eval_mode,
            "test_stat_mode": self.config.test_stat_mode,
            "calibration_mode": self.config.calibration_mode,
            "y_resampler": self.config.y_resampler,
            "n_folds": int(self.config.n_folds),
            "n_null_draws": int(self.config.n_null_draws),
            "u_grid_size": int(len(u_grid)),
            "null_mean": (
                float(np.mean(null_stats)) if len(null_stats) else float("nan")
            ),
            "null_std": float(np.std(null_stats)) if len(null_stats) else float("nan"),
            "sdb_region_pvalue_enabled": bool(self.config.sdb_region_pvalue_enabled)
            and str(data.benchmark) == "sdb",
            "sdb_region_pvalue_decision": bool(self.config.sdb_region_pvalue_decision),
            "sdb_region_pvalue_statistic": str(self.config.sdb_region_pvalue_statistic),
            "sdb_region_pvalue_statistics": ",".join(
                str(x) for x in cal.get("sdb_region_statistics", region_statistics)
            ),
            "sdb_region_pvalue_decision_statistic": decision_statistic,
            "p_value_sdb_region": p_value_sdb_region,
            "reject_sdb_region_alpha_005": (
                bool(p_value_sdb_region <= float(self.config.sdb_region_alpha))
                if np.isfinite(p_value_sdb_region)
                else None
            ),
            "decision_p_value": float(decision_p_value),
            "decision_source": decision_source,
            "decision_statistic": (
                decision_statistic if decision_source == "sdb_region" else "global"
            ),
            "reject_decision_alpha_005": bool(
                decision_p_value <= float(self.config.sdb_region_alpha)
            ),
            "T_sdb_region_obs_calib": float(
                cal.get("T_sdb_region_obs_calib", float("nan"))
            ),
            "T_sdb_region_null_mean": (
                float(np.mean(cal.get("T_sdb_region_nulls", [])))
                if len(cal.get("T_sdb_region_nulls", []))
                else float("nan")
            ),
            "T_sdb_region_null_std": (
                float(np.std(cal.get("T_sdb_region_nulls", [])))
                if len(cal.get("T_sdb_region_nulls", []))
                else float("nan")
            ),
            "T_sdb_region_null_min": (
                float(np.min(cal.get("T_sdb_region_nulls", [])))
                if len(cal.get("T_sdb_region_nulls", []))
                else float("nan")
            ),
            "T_sdb_region_null_max": (
                float(np.max(cal.get("T_sdb_region_nulls", [])))
                if len(cal.get("T_sdb_region_nulls", []))
                else float("nan")
            ),
            "T_sdb_region_n_null": int(len(cal.get("T_sdb_region_nulls", []))),
            **region_meta,
            **region_info,
            "T_delta": float(np.mean(delta_full**2)),
            **deformation_diagnostics(delta_full, None),
            **sdb_struct_diagnostics(delta_full, data, u_grid),
        }
        result.u_grid = np.asarray(u_grid, dtype=np.float32).copy()
        result.deformation = deformation
        return result
