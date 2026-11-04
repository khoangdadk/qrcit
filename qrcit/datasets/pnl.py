"""Post-nonlinear synthetic data with analytic and detection families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from qrcit.data import CITData

ALL_STANDARD_PNL_FAMILIES = ("additive_gaussian", "pnl_analytic", "pnl_detection")
# The benchmark mixes the two nonlinear families.
STANDARD_PNL_FAMILIES = ("pnl_analytic", "pnl_detection")
PNL_MONOTONE_LINKS = ("identity", "tanh", "asinh", "cube")
PNL_DETECTION_LINKS = ("identity", "square", "cube", "tanh", "cos")


@dataclass
class StandardPNLMixedConfig:
    n: int = 1000
    d_z: int = 5
    beta: float = 0.5
    seed: int = 0
    family: str | None = None
    sigma_x: float = 1.0
    sigma_y: float = 1.0
    link_scale_x: float = 1.0
    link_scale_y: float = 1.0
    analytic_link_scale_x: float = 0.75
    analytic_link_scale_y: float = 0.75
    detection_dep_input: Literal["x_raw", "x_latent"] = "x_raw"
    detection_center_signal: bool = True
    detection_standardize_signal: bool = True


def make_unit_vector(rng: np.random.Generator, d: int) -> np.ndarray:
    a = rng.normal(size=d)
    return (a / (np.linalg.norm(a) + 1e-12)).astype(np.float32)


def structural_functions(
    Z: np.ndarray, a_x: np.ndarray, a_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tx, ty = Z @ a_x, Z @ a_y
    m_x = np.sin(tx) + 0.3 * tx**2
    m_y = np.cos(ty) + 0.3 * ty**2
    return m_x.astype(np.float32), m_y.astype(np.float32)


def pnl_link(t: np.ndarray, name: str = "identity", scale: float = 1.0) -> np.ndarray:
    """Monotone link for the analytic PNL family."""
    t = np.asarray(t, dtype=np.float32)
    scale = float(scale)
    if name == "identity":
        return t.astype(np.float32)
    if name == "tanh":
        return np.tanh(scale * t).astype(np.float32)
    if name == "asinh":
        return np.arcsinh(scale * t).astype(np.float32)
    if name == "cube":
        return np.power(t, 3).astype(np.float32)
    raise ValueError(f"Unknown PNL analytic link: {name}")


def pnl_detection_link(
    t: np.ndarray, name: str = "identity", scale: float = 1.0
) -> np.ndarray:
    """Possibly non-monotone link for the PNL-detection family."""
    t = np.asarray(t, dtype=np.float32)
    tt = float(scale) * t
    if name == "identity":
        return tt.astype(np.float32)
    if name == "square":
        return np.square(tt).astype(np.float32)
    if name == "cube":
        return np.power(tt, 3).astype(np.float32)
    if name == "tanh":
        return np.tanh(tt).astype(np.float32)
    if name == "cos":
        return np.cos(tt).astype(np.float32)
    raise ValueError(f"Unknown PNL detection link: {name}")


def normalize_detection_signal(
    signal: np.ndarray, center: bool = True, standardize: bool = True
) -> tuple[np.ndarray, float, float]:
    signal = np.asarray(signal, dtype=np.float32).reshape(-1)
    mean = float(np.mean(signal)) if center else 0.0
    centered = signal - mean
    std = float(np.std(centered)) if standardize else 1.0
    std = max(std, 1e-8)
    out = centered / std if standardize else centered
    return out.astype(np.float32), mean, std


def _base_meta(cfg: StandardPNLMixedConfig, family: str, rng_seed: int) -> dict:
    return {
        "sub_benchmark": family,
        "synthetic_family": family,
        "source": "standard_pnl_mixed",
        "generator_seed": int(rng_seed),
        "exact_delta_available": False,
    }


def generate_additive_gaussian(
    cfg: StandardPNLMixedConfig, rng: np.random.Generator
) -> CITData:
    Z = rng.normal(size=(cfg.n, cfg.d_z)).astype(np.float32)
    a_x, a_y = make_unit_vector(rng, cfg.d_z), make_unit_vector(rng, cfg.d_z)
    m_x, m_y = structural_functions(Z, a_x, a_y)
    eps_y = rng.normal(size=cfg.n).astype(np.float32)
    eps_x = rng.normal(size=cfg.n).astype(np.float32)
    Y = (m_y + eps_y).astype(np.float32)
    X = (m_x + float(cfg.beta) * Y + eps_x).astype(np.float32)
    meta = _base_meta(cfg, "additive_gaussian", cfg.seed)
    meta.update(
        {
            "variant_detail": "additive_gaussian",
            "sigma_x": 1.0,
            "sigma_y": 1.0,
            "a_x": a_x.tolist(),
            "a_y": a_y.tolist(),
            "exact_delta_available": True,
        }
    )
    return CITData(
        X=X[:, None],
        Y=Y[:, None],
        Z=Z,
        benchmark="standard_pnl_mixed",
        variant="additive_gaussian",
        beta=float(cfg.beta),
        seed=int(cfg.seed),
        meta=meta,
    )


def generate_pnl_analytic(
    cfg: StandardPNLMixedConfig, rng: np.random.Generator
) -> CITData:
    link_x = str(rng.choice(PNL_MONOTONE_LINKS))
    link_y = str(rng.choice(PNL_MONOTONE_LINKS))
    Z = rng.normal(size=(cfg.n, cfg.d_z)).astype(np.float32)
    a_x, a_y = make_unit_vector(rng, cfg.d_z), make_unit_vector(rng, cfg.d_z)
    axz = (Z @ a_x).astype(np.float32)
    ayz = (Z @ a_y).astype(np.float32)
    eps_y = rng.normal(size=cfg.n).astype(np.float32)
    eps_x = rng.normal(size=cfg.n).astype(np.float32)
    nu_y = (ayz + cfg.sigma_y * eps_y).astype(np.float32)
    Y = pnl_link(nu_y, name=link_y, scale=cfg.analytic_link_scale_y)
    nu_x = (axz + float(cfg.beta) * nu_y + cfg.sigma_x * eps_x).astype(np.float32)
    X = pnl_link(nu_x, name=link_x, scale=cfg.analytic_link_scale_x)
    variant = f"pnl_analytic__x_{link_x}__y_{link_y}"
    meta = _base_meta(cfg, "pnl_analytic", cfg.seed)
    meta.update(
        {
            "variant_detail": variant,
            "link_x": link_x,
            "link_y": link_y,
            "link_scale_x": float(cfg.analytic_link_scale_x),
            "link_scale_y": float(cfg.analytic_link_scale_y),
            "sigma_x": float(cfg.sigma_x),
            "sigma_y": float(cfg.sigma_y),
            "a_x": a_x.tolist(),
            "a_y": a_y.tolist(),
            "exact_delta_available": True,
        }
    )
    return CITData(
        X=X[:, None],
        Y=Y[:, None],
        Z=Z,
        benchmark="standard_pnl_mixed",
        variant=variant,
        beta=float(cfg.beta),
        seed=int(cfg.seed),
        meta=meta,
    )


def generate_pnl_detection(
    cfg: StandardPNLMixedConfig, rng: np.random.Generator
) -> CITData:
    link_x = str(rng.choice(PNL_DETECTION_LINKS))
    link_y = str(rng.choice(PNL_DETECTION_LINKS))
    Z = rng.normal(size=(cfg.n, cfg.d_z)).astype(np.float32)
    a_x, a_y = make_unit_vector(rng, cfg.d_z), make_unit_vector(rng, cfg.d_z)
    axz = (Z @ a_x).astype(np.float32)
    ayz = (Z @ a_y).astype(np.float32)
    eps_x = rng.normal(size=cfg.n).astype(np.float32)
    eps_y = rng.normal(size=cfg.n).astype(np.float32)
    eta_x = (axz + cfg.sigma_x * eps_x).astype(np.float32)
    x_raw = pnl_detection_link(eta_x, name=link_x, scale=cfg.link_scale_x)
    raw_signal = x_raw if cfg.detection_dep_input == "x_raw" else eta_x
    signal, signal_mean, signal_std = normalize_detection_signal(
        raw_signal,
        center=cfg.detection_center_signal,
        standardize=cfg.detection_standardize_signal,
    )
    eta_y = (ayz + float(cfg.beta) * signal + cfg.sigma_y * eps_y).astype(np.float32)
    y_raw = pnl_detection_link(eta_y, name=link_y, scale=cfg.link_scale_y)
    variant = f"pnl_detection__x_{link_x}__y_{link_y}"
    meta = _base_meta(cfg, "pnl_detection", cfg.seed)
    meta.update(
        {
            "variant_detail": variant,
            "link_x": link_x,
            "link_y": link_y,
            "link_scale_x": float(cfg.link_scale_x),
            "link_scale_y": float(cfg.link_scale_y),
            "sigma_x": float(cfg.sigma_x),
            "sigma_y": float(cfg.sigma_y),
            "dep_input": cfg.detection_dep_input,
            "signal_mean": float(signal_mean),
            "signal_std": float(signal_std),
            "a_x": a_x.tolist(),
            "a_y": a_y.tolist(),
            "exact_delta_available": False,
        }
    )
    return CITData(
        X=x_raw[:, None],
        Y=y_raw[:, None],
        Z=Z,
        benchmark="standard_pnl_mixed",
        variant=variant,
        beta=float(cfg.beta),
        seed=int(cfg.seed),
        meta=meta,
    )


def generate_standard_pnl_mixed(
    n: int = 1000,
    d_z: int = 5,
    beta: float = 0.5,
    seed: int = 0,
    family: str | None = None,
    trial_id: int | None = None,
    families: Iterable[str] | None = None,
) -> CITData:
    """Generate one trial from the standard PNL benchmark.

    If ``family`` is provided, that exact mechanism is used.  This keeps
    ``additive_gaussian`` available for controlled smoke tests.

    If ``family`` is None, the mechanism is sampled uniformly from ``families``
    when provided, otherwise from ``STANDARD_PNL_FAMILIES``.  The
    default excludes ``additive_gaussian`` because it is usually too easy.
    """
    cfg = StandardPNLMixedConfig(
        n=int(n), d_z=int(d_z), beta=float(beta), seed=int(seed), family=family
    )
    rng = np.random.default_rng(seed)
    pool = tuple(families) if families is not None else STANDARD_PNL_FAMILIES
    if not pool:
        raise ValueError("families must contain at least one mechanism")
    for fam in pool:
        if fam not in ALL_STANDARD_PNL_FAMILIES:
            raise ValueError(
                f"Unknown family in pool: {fam!r}. Choices: {ALL_STANDARD_PNL_FAMILIES}"
            )
    selected = family or str(rng.choice(pool))
    if selected not in ALL_STANDARD_PNL_FAMILIES:
        raise ValueError(
            f"Unknown family={selected!r}. Choices: {ALL_STANDARD_PNL_FAMILIES}"
        )
    if selected == "additive_gaussian":
        data = generate_additive_gaussian(cfg, rng)
    elif selected == "pnl_analytic":
        data = generate_pnl_analytic(cfg, rng)
    elif selected == "pnl_detection":
        data = generate_pnl_detection(cfg, rng)
    else:  # pragma: no cover
        raise RuntimeError(selected)
    data.meta.update(
        {
            "trial_id": None if trial_id is None else int(trial_id),
            "merged_group": "standard_pnl_mixed",
            "n": int(n),
            "d_z": int(d_z),
            "mechanism_family_pool": list(pool),
        }
    )
    return data
