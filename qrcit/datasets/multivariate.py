"""Simple multivariate synthetic conditional-independence benchmarks.

These generators are intentionally simple.  Their role is to check whether
methods handle multivariate X and/or multivariate Y, not to stress subtle
conditional-law deformation as SDB does.

All cases satisfy X _||_ Y | Z when beta=0.  For beta>0, X depends on Y
after conditioning on Z.
"""

from __future__ import annotations

import numpy as np

from qrcit.data import CITData

VALID_CASES = {
    "scalar_x_multivariate_y",
    "multivariate_x_scalar_y",
    "multivariate_x_multivariate_y",
}


def _standardize(a: np.ndarray) -> np.ndarray:
    return (
        (a - a.mean(axis=0, keepdims=True)) / (a.std(axis=0, keepdims=True) + 1e-8)
    ).astype(np.float32)


def _make_low_rank_matrix(
    rng: np.random.Generator, rows: int, cols: int, scale: float = 0.5
) -> np.ndarray:
    return (scale * rng.normal(size=(rows, cols))).astype(np.float32)


def generate_multivariate_simple(
    case: str,
    n: int = 2000,
    d_x: int | None = None,
    d_y: int | None = None,
    d_z: int = 5,
    beta: float = 0.5,
    seed: int = 0,
    noise_x: float = 1.0,
    noise_y: float = 1.0,
    nonlinear: bool = True,
) -> CITData:
    """Generate one multivariate CIT dataset.

    Cases:
        scalar_x_multivariate_y: d_x=1, d_y defaults to 5.
        multivariate_x_scalar_y: d_x defaults to 5, d_y=1.
        multivariate_x_multivariate_y: d_x=d_y defaults to 5.
    """
    case = str(case)
    if case not in VALID_CASES:
        raise ValueError(
            f"Unknown multivariate case {case!r}; expected one of {sorted(VALID_CASES)}"
        )
    if d_x is None:
        d_x = 1 if case == "scalar_x_multivariate_y" else 5
    if d_y is None:
        d_y = 1 if case == "multivariate_x_scalar_y" else 5
    if case == "scalar_x_multivariate_y":
        d_x = 1
    if case == "multivariate_x_scalar_y":
        d_y = 1
    rng = np.random.default_rng(int(seed))
    Z = rng.normal(size=(int(n), int(d_z))).astype(np.float32)

    Ay = _make_low_rank_matrix(rng, int(d_z), int(d_y), scale=0.55)
    Ax = _make_low_rank_matrix(rng, int(d_z), int(d_x), scale=0.55)
    Ey = rng.normal(size=(int(n), int(d_y))).astype(np.float32) * float(noise_y)
    Ex = rng.normal(size=(int(n), int(d_x))).astype(np.float32) * float(noise_x)

    Y_base = Z @ Ay + Ey
    if nonlinear:
        Y = np.tanh(Y_base) + 0.25 * Y_base
    else:
        Y = Y_base
    Y = _standardize(Y)

    X_base = Z @ Ax + Ex
    if nonlinear:
        X_base = np.sin(0.65 * X_base) + 0.35 * X_base

    # Low-rank Y -> X dependence.  For d_y=1 this reduces to a vector effect.
    B = _make_low_rank_matrix(rng, int(d_y), int(d_x), scale=1.0 / np.sqrt(max(1, d_y)))
    signal = Y @ B
    signal = _standardize(signal)
    X = X_base + float(beta) * signal
    X = _standardize(X)

    meta = {
        "case": case,
        "mechanism": "multivariate_simple_low_rank",
        "nonlinear": bool(nonlinear),
        "noise_x": float(noise_x),
        "noise_y": float(noise_y),
        "effect_rank": int(min(d_x, d_y)),
        "description": "beta=0 gives X independent of Y given Z; beta>0 adds low-rank Y->X dependence.",
    }
    return CITData(
        X=X.astype(np.float32),
        Y=Y.astype(np.float32),
        Z=Z.astype(np.float32),
        benchmark="multivariate_simple",
        variant=case,
        beta=float(beta),
        seed=int(seed),
        meta=meta,
    )
