"""Summary statistics for learned quantile deformations."""

from __future__ import annotations

import numpy as np


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    aa, bb = a[mask], b[mask]
    if aa.std() < 1e-12 or bb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def deformation_diagnostics(
    delta_hat: np.ndarray, delta_true: np.ndarray | None = None
) -> dict:
    out = {
        "T_delta": float(np.nanmean(np.asarray(delta_hat) ** 2)),
        "T_sup": float(np.nanmax(np.abs(delta_hat))),
        "MSE_delta": float("nan"),
        "NMSE_delta": float("nan"),
        "Corr_delta": float("nan"),
    }
    if delta_true is not None:
        dt = np.asarray(delta_true)
        dh = np.asarray(delta_hat)
        if dt.shape == dh.shape:
            mse = float(np.nanmean((dh - dt) ** 2))
            denom = float(np.nanmean(dt**2)) + 1e-12
            out.update(
                {
                    "MSE_delta": mse,
                    "NMSE_delta": mse / denom,
                    "Corr_delta": safe_corr(dh, dt),
                }
            )
    return out


def sdb_struct_diagnostics(
    delta_hat: np.ndarray, data, u_grid: np.ndarray | None = None
) -> dict:
    """Compare quantile-averaged predictions with injected sample deformations.

    The injected value is evaluated at each sample's latent U; it is not a
    ground-truth conditional-quantile difference on the full quantile grid.
    """
    out = {
        "Corr_delta_struct_sample": float("nan"),
        "MSE_delta_struct_sample": float("nan"),
        "T_sdb_region_ratio": float("nan"),
    }
    if getattr(data, "delta_struct", None) is None:
        return out
    d_true = np.asarray(data.delta_struct).reshape(-1)
    d_hat = np.nanmean(np.asarray(delta_hat), axis=1).reshape(-1)
    out["Corr_delta_struct_sample"] = safe_corr(d_hat, d_true)
    out["MSE_delta_struct_sample"] = float(np.nanmean((d_hat - d_true) ** 2))
    if u_grid is not None and str(getattr(data, "benchmark", "")) == "sdb":
        try:
            from qrcit.datasets.sdb import sdb_region_mask_grid

            mask, _ = sdb_region_mask_grid(data, u_grid)
            dh = np.asarray(delta_hat, dtype=np.float32)
            if mask.shape == dh.shape and int(mask.sum()) > 0:
                t_global = float(np.nanmean(dh**2))
                t_region = float(np.nanmean(dh[mask] ** 2))
                out["T_sdb_region_ratio"] = float(t_region / (t_global + 1e-12))
        except Exception:
            pass
    return out
