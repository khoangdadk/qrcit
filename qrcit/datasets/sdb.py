"""Structural Dependency Benchmark: X = Q0(Z, U) + beta * d(Y, Z, U).

The null beta=0 makes X conditionally independent of Y given Z.
Injected deformations and affected-region metadata accompany each sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.special import expit, ndtri

from qrcit.data import CITData

SDB_VARIANTS = (
    "context_local",
    "sign_reversing",
    "tail_only",
    "mean_neutral_shape",
    "moving_u_ridge",
)


@dataclass
class SDBConfig:
    n: int = 2000
    d_z: int = 5
    beta: float = 0.5
    seed: int = 0
    variant: Literal[
        "context_local",
        "sign_reversing",
        "tail_only",
        "mean_neutral_shape",
        "moving_u_ridge",
    ] = "context_local"
    amplitude: float = 1.0
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    u_eps: float = 1e-4
    gate_steepness: float = 5.0
    tail_u0: float = 0.90
    shape_u0: float = 0.65
    shape_width: float = 0.18
    moving_width: float = 0.10


def _unit(rng: np.random.Generator, d: int) -> np.ndarray:
    v = rng.normal(size=d)
    return (v / (np.linalg.norm(v) + 1e-12)).astype(np.float32)


def _base_q0(Z: np.ndarray, u: np.ndarray, sigma_x: float = 1.0) -> np.ndarray:
    """Baseline conditional quantile Q0(Z,u)."""
    z0 = Z[:, 0]
    z1 = Z[:, 1] if Z.shape[1] > 1 else 0.0
    z2 = Z[:, 2] if Z.shape[1] > 2 else 0.0
    loc = 0.7 * np.sin(z0) + 0.35 * z1 + 0.20 * np.cos(z2)
    scale = float(sigma_x) * (0.65 + 0.25 * expit(z0 - 0.5))
    return (loc + scale * ndtri(u)).astype(np.float32)


def _generate_y(
    Z: np.ndarray, rng: np.random.Generator, sigma_y: float
) -> tuple[np.ndarray, dict]:
    d = Z.shape[1]
    a_y = _unit(rng, d)
    t = Z @ a_y
    m_y = np.cos(t) + 0.3 * t**2 + 0.2 * np.sin(Z[:, 0])
    y = (m_y + float(sigma_y) * rng.normal(size=Z.shape[0])).astype(np.float32)
    return y, {"a_y": a_y.tolist(), "y_mean": float(y.mean()), "y_std": float(y.std())}


def _gaussian_bump(
    u: np.ndarray, center: np.ndarray | float, width: float
) -> np.ndarray:
    return np.exp(-0.5 * ((u - center) / max(float(width), 1e-6)) ** 2).astype(
        np.float32
    )


def _mean_neutral_shape(u: np.ndarray, center: float, width: float) -> np.ndarray:
    """A positive ridge minus its numerical mean over the unit interval."""
    bump = _gaussian_bump(u, center, width)
    grid = np.linspace(1e-4, 1 - 1e-4, 2000, dtype=np.float32)
    bump_mean = float(_gaussian_bump(grid, center, width).mean())
    return (bump - bump_mean).astype(np.float32)


def _moving_center(Z: np.ndarray) -> np.ndarray:
    z0 = Z[:, 0]
    z1 = Z[:, 1] if Z.shape[1] > 1 else 0.0
    # keep center away from boundaries: roughly [0.2, 0.8]
    return (0.20 + 0.60 * expit(0.9 * z0 - 0.5 * z1)).astype(np.float32)


def structural_deformation(
    y: np.ndarray, Z: np.ndarray, u: np.ndarray, cfg: SDBConfig
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return injected deformation d(Y,Z,U), region mask, and metadata."""
    y_std = ((y - y.mean()) / (y.std() + 1e-8)).astype(np.float32)
    z0 = Z[:, 0]
    z1 = Z[:, 1] if Z.shape[1] > 1 else 0.0
    k = float(cfg.gate_steepness)

    if cfg.variant == "context_local":
        # localized slab in Z_0, smoothed by a product of two sigmoids
        left = expit(k * (z0 + 0.75))
        right = expit(k * (0.75 - z0))
        gate = (left * right).astype(np.float32)
        shape = (0.75 + 0.25 * np.cos(2 * np.pi * u)).astype(np.float32)
        d = y_std * gate * shape
        region = np.abs(z0) <= 0.75
        region_name = "active_z_slab"
        extra = {"active_z_threshold": 0.75}

    elif cfg.variant == "sign_reversing":
        sign = np.tanh(1.25 * z0).astype(np.float32)
        shape = (0.75 + 0.25 * np.cos(2 * np.pi * u)).astype(np.float32)
        d = y_std * sign * shape
        region = np.abs(z0) >= 0.35
        region_name = "large_abs_z_sign_regions"
        extra = {
            "sign_mean": float(sign.mean()),
            "sign_abs_mean": float(np.abs(sign).mean()),
        }

    elif cfg.variant == "tail_only":
        tail_gate = expit(k * (u - float(cfg.tail_u0))).astype(np.float32)
        d = y_std * tail_gate
        region = u >= float(cfg.tail_u0)
        region_name = "upper_tail"
        extra = {"tail_u0": float(cfg.tail_u0)}

    elif cfg.variant == "mean_neutral_shape":
        shape = _mean_neutral_shape(
            u, center=float(cfg.shape_u0), width=float(cfg.shape_width)
        )
        # normalize for comparable amplitude across widths
        shape = shape / (np.std(shape) + 1e-8)
        d = y_std * shape
        region = np.abs(u - float(cfg.shape_u0)) <= 2.0 * float(cfg.shape_width)
        region_name = "mean_neutral_shape_band"
        extra = {
            "shape_u0": float(cfg.shape_u0),
            "shape_width": float(cfg.shape_width),
            "shape_mean_sample": float(shape.mean()),
        }

    elif cfg.variant == "moving_u_ridge":
        u_star = _moving_center(Z)
        ridge = _gaussian_bump(u, u_star, float(cfg.moving_width))
        ridge = ridge / (np.std(ridge) + 1e-8)
        context_amp = (0.75 + 0.25 * np.sin(z0 + 0.5 * z1)).astype(np.float32)
        d = y_std * context_amp * ridge
        region = np.abs(u - u_star) <= 2.0 * float(cfg.moving_width)
        region_name = "moving_u_ridge_band"
        extra = {
            "moving_width": float(cfg.moving_width),
            "u_star_mean": float(u_star.mean()),
            "u_star_std": float(u_star.std()),
        }

    else:
        raise ValueError(
            f"Unknown SDB variant {cfg.variant!r}. Choices: {SDB_VARIANTS}"
        )

    d = (float(cfg.amplitude) * d).astype(np.float32)
    region = np.asarray(region, dtype=bool)
    meta = {"sdb_region_name": region_name, **extra}
    return d, region, meta


def generate_sdb(
    n: int = 2000,
    d_z: int = 5,
    beta: float = 0.5,
    seed: int = 0,
    variant: str = "context_local",
    trial_id: int | None = None,
    amplitude: float = 1.0,
) -> CITData:
    if variant not in SDB_VARIANTS:
        raise ValueError(f"Unknown variant={variant!r}. Choices: {SDB_VARIANTS}")
    cfg = SDBConfig(
        n=int(n),
        d_z=int(d_z),
        beta=float(beta),
        seed=int(seed),
        variant=variant,
        amplitude=float(amplitude),
    )
    rng = np.random.default_rng(int(seed))
    Z = rng.normal(size=(cfg.n, cfg.d_z)).astype(np.float32)
    Y, y_meta = _generate_y(Z, rng, sigma_y=cfg.sigma_y)
    U = rng.uniform(cfg.u_eps, 1.0 - cfg.u_eps, size=cfg.n).astype(np.float32)
    Q0 = _base_q0(Z, U, sigma_x=cfg.sigma_x)
    d_struct, region_mask, region_meta = structural_deformation(Y, Z, U, cfg)
    X = (Q0 + float(cfg.beta) * d_struct).astype(np.float32)

    region_n = int(region_mask.sum())
    t_struct_global = float(np.mean(d_struct**2))
    t_struct_region = (
        float(np.mean(d_struct[region_mask] ** 2)) if region_n > 0 else float("nan")
    )
    meta = {
        "source": "sdb",
        "merged_group": "sdb_mixed",
        "sub_benchmark": "sdb",
        "sdb_variant": variant,
        "variant_detail": variant,
        "trial_id": None if trial_id is None else int(trial_id),
        "n": int(n),
        "d_z": int(d_z),
        "u_eps": float(cfg.u_eps),
        "amplitude": float(cfg.amplitude),
        "sigma_x": float(cfg.sigma_x),
        "sigma_y": float(cfg.sigma_y),
        "region_fraction": float(np.mean(region_mask)),
        "region_n": region_n,
        "T_struct_global": t_struct_global,
        "T_struct_region": t_struct_region,
        "T_struct_region_ratio": (
            float(t_struct_region / (t_struct_global + 1e-12))
            if region_n > 0
            else float("nan")
        ),
        "mean_abs_delta_struct": float(np.mean(np.abs(d_struct))),
        "mean_abs_delta_struct_region": (
            float(np.mean(np.abs(d_struct[region_mask])))
            if region_n > 0
            else float("nan")
        ),
        "U_mean": float(U.mean()),
        "U_std": float(U.std()),
        **y_meta,
        **region_meta,
    }
    return CITData(
        X=X[:, None],
        Y=Y[:, None],
        Z=Z,
        benchmark="sdb",
        variant=variant,
        beta=float(beta),
        seed=int(seed),
        meta=meta,
        delta_struct=d_struct[:, None],
        u_grid=None,
    )


def sdb_region_mask_grid(data: CITData, u_grid: np.ndarray) -> tuple[np.ndarray, dict]:
    """Return the oracle SDB affected-region mask on a CFIT quantile grid.

    The returned mask has shape ``(n, len(u_grid))`` and is defined only from
    the known SDB structural region.  It is intended for synthetic SDB
    diagnostics / oracle-region decision rules, not for real-data CFIT.
    """
    if str(getattr(data, "benchmark", "")) != "sdb":
        raise ValueError("sdb_region_mask_grid requires data.benchmark == 'sdb'.")
    variant = str(data.meta.get("sdb_variant", data.variant))
    if variant not in SDB_VARIANTS:
        raise ValueError(f"Unknown SDB variant {variant!r}. Choices: {SDB_VARIANTS}")

    Z = np.asarray(data.Z, dtype=np.float32)
    u = np.asarray(u_grid, dtype=np.float32).reshape(1, -1)
    n = Z.shape[0]
    z0 = Z[:, [0]]
    z1 = Z[:, [1]] if Z.shape[1] > 1 else np.zeros((n, 1), dtype=np.float32)

    if variant == "context_local":
        thr = float(data.meta.get("active_z_threshold", 0.75))
        sample_mask = np.abs(z0) <= thr
        mask = np.repeat(sample_mask, u.shape[1], axis=1)
        region_name = "active_z_slab"

    elif variant == "sign_reversing":
        sample_mask = np.abs(z0) >= 0.35
        mask = np.repeat(sample_mask, u.shape[1], axis=1)
        region_name = "large_abs_z_sign_regions"

    elif variant == "tail_only":
        tail_u0 = float(data.meta.get("tail_u0", 0.90))
        mask = np.repeat((u >= tail_u0), n, axis=0)
        region_name = "upper_tail"

    elif variant == "mean_neutral_shape":
        center = float(data.meta.get("shape_u0", 0.65))
        width = float(data.meta.get("shape_width", 0.18))
        mask = np.repeat((np.abs(u - center) <= 2.0 * width), n, axis=0)
        region_name = "mean_neutral_shape_band"

    elif variant == "moving_u_ridge":
        width = float(data.meta.get("moving_width", 0.10))
        u_star = (0.20 + 0.60 * expit(0.9 * z0 - 0.5 * z1)).astype(np.float32)
        mask = np.abs(u - u_star) <= 2.0 * width
        region_name = "moving_u_ridge_band"

    else:  # defensive; covered above
        raise ValueError(f"Unknown SDB variant {variant!r}.")

    mask = np.asarray(mask, dtype=bool)
    info = {
        "sdb_region_mask_source": "oracle_sdb_variant",
        "sdb_region_name_grid": region_name,
        "sdb_region_grid_fraction": float(mask.mean()) if mask.size else float("nan"),
        "sdb_region_grid_n": int(mask.sum()),
    }
    return mask, info
