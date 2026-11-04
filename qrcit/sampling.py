"""Y|Z resamplers for conditional-randomization calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV


class YGivenZSampler:
    name = "base"

    def fit(self, Y: np.ndarray, Z: np.ndarray):
        raise NotImplementedError

    def sample(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LearnedGaussianSampler(YGivenZSampler):
    """Conditional Gaussian with a ridge mean and constant residual scale."""

    alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
    min_std: float = 1e-3
    name: str = "learned_gaussian"

    def fit(self, Y: np.ndarray, Z: np.ndarray):
        self.model_ = RidgeCV(alphas=np.asarray(self.alphas, dtype=np.float64)).fit(
            Z, Y
        )
        mu = self.model_.predict(Z).reshape(Y.shape)
        resid = Y - mu
        self.std_ = np.maximum(resid.std(axis=0, keepdims=True), self.min_std).astype(
            np.float32
        )
        return self

    def sample(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.model_.predict(Z).astype(np.float32)
        if mu.ndim == 1:
            mu = mu[:, None]
        return (mu + rng.normal(size=mu.shape).astype(np.float32) * self.std_).astype(
            np.float32
        )


@dataclass
class StandardizedResidualBootstrapSampler(YGivenZSampler):
    """Random-forest mean/scale model with standardized residual bootstrap."""

    n_estimators: int = 100
    min_std: float = 1e-3
    random_state: int = 0
    name: str = "standardized_residual_bootstrap"

    def fit(self, Y: np.ndarray, Z: np.ndarray):
        y = Y.reshape(Y.shape[0], -1)
        if y.shape[1] != 1:
            raise ValueError(
                "StandardizedResidualBootstrapSampler currently supports scalar Y only."
            )
        self.mean_model_ = RandomForestRegressor(
            n_estimators=int(self.n_estimators),
            min_samples_leaf=5,
            random_state=int(self.random_state),
            n_jobs=-1,
        ).fit(Z, y[:, 0])
        mu = self.mean_model_.predict(Z).reshape(-1, 1).astype(np.float32)
        abs_res = np.abs(y - mu).reshape(-1)
        self.scale_model_ = RandomForestRegressor(
            n_estimators=int(self.n_estimators),
            min_samples_leaf=5,
            random_state=int(self.random_state) + 17,
            n_jobs=-1,
        ).fit(Z, abs_res)
        scale = self.scale_model_.predict(Z).reshape(-1, 1).astype(np.float32)
        scale = np.maximum(scale, self.min_std)
        self.std_resid_ = ((y - mu) / scale).reshape(-1).astype(np.float32)
        return self

    def sample(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mu = self.mean_model_.predict(Z).reshape(-1, 1).astype(np.float32)
        scale = self.scale_model_.predict(Z).reshape(-1, 1).astype(np.float32)
        scale = np.maximum(scale, self.min_std)
        idx = rng.integers(0, self.std_resid_.shape[0], size=Z.shape[0])
        return (mu + scale * self.std_resid_[idx, None]).astype(np.float32)


def _pnl_link(t: np.ndarray, name: str = "identity", scale: float = 1.0) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    tt = float(scale) * t
    if name == "identity":
        return t.astype(np.float32)
    if name == "tanh":
        return np.tanh(tt).astype(np.float32)
    if name == "asinh":
        return np.arcsinh(tt).astype(np.float32)
    if name == "cube":
        return np.power(t, 3).astype(np.float32)
    raise ValueError(f"Unknown oracle PNL monotone link {name!r}")


def _pnl_detection_link(
    t: np.ndarray, name: str = "identity", scale: float = 1.0
) -> np.ndarray:
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
    raise ValueError(f"Unknown oracle PNL detection link {name!r}")


class OracleYGivenZSampler(YGivenZSampler):
    """Generator-aware Y|Z sampler for synthetic benchmarks.

    The CFIT training loop standardizes Y and Z inside each fold.  This sampler
    therefore receives the fold standardizer, converts standardized Z back to
    raw Z, samples raw Y from the known synthetic generator, then returns Y in
    the fold-standardized scale.

    Supported generators:
      - standard_pnl_mixed/additive_gaussian
      - standard_pnl_mixed/pnl_analytic
      - standard_pnl_mixed/pnl_detection (Monte Carlo over latent X noise)
      - sdb (same Y|Z structural model)

    For unsupported data, construction fails explicitly rather than silently
    falling back to a learned sampler.
    """

    name = "oracle"

    def __init__(self, data=None, standardizer=None):
        if data is None or standardizer is None:
            raise ValueError(
                "OracleYGivenZSampler requires data and fold standardizer."
            )
        self.data = data
        self.meta = getattr(data, "meta", {}) or {}
        self.standardizer = standardizer
        self.beta = float(getattr(data, "beta", self.meta.get("beta", 0.0)))
        self.benchmark = str(getattr(data, "benchmark", self.meta.get("source", "")))
        self.variant = str(getattr(data, "variant", self.meta.get("sub_benchmark", "")))

    def fit(self, Y: np.ndarray, Z: np.ndarray):
        return self

    def _raw_z(self, Z_std: np.ndarray) -> np.ndarray:
        return (
            np.asarray(Z_std, dtype=np.float32) * self.standardizer.z_std
            + self.standardizer.z_mean
        ).astype(np.float32)

    def _standardize_y(self, Y_raw: np.ndarray) -> np.ndarray:
        y = np.asarray(Y_raw, dtype=np.float32)
        if y.ndim == 1:
            y = y[:, None]
        return ((y - self.standardizer.y_mean) / self.standardizer.y_std).astype(
            np.float32
        )

    def _unit_from_meta(self, key: str) -> np.ndarray:
        if key not in self.meta:
            raise ValueError(f"Oracle sampler metadata missing {key!r}.")
        return np.asarray(self.meta[key], dtype=np.float32)

    @staticmethod
    def _structural_m_y(Z: np.ndarray, a_y: np.ndarray) -> np.ndarray:
        ty = Z @ a_y
        return (np.cos(ty) + 0.3 * ty**2).astype(np.float32)

    @staticmethod
    def _sdb_m_y(Z: np.ndarray, a_y: np.ndarray) -> np.ndarray:
        t = Z @ a_y
        return (np.cos(t) + 0.3 * t**2 + 0.2 * np.sin(Z[:, 0])).astype(np.float32)

    def _sample_standard_pnl(
        self, Z: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        sub = str(self.meta.get("sub_benchmark", self.variant))
        a_y = self._unit_from_meta("a_y")
        sigma_y = float(self.meta.get("sigma_y", 1.0))
        eps_y = rng.normal(size=Z.shape[0]).astype(np.float32)

        if sub == "additive_gaussian":
            m_y = self._structural_m_y(Z, a_y)
            return (m_y + sigma_y * eps_y).astype(np.float32)

        if sub == "pnl_analytic":
            ayz = (Z @ a_y).astype(np.float32)
            nu_y = (ayz + sigma_y * eps_y).astype(np.float32)
            return _pnl_link(
                nu_y,
                name=str(self.meta.get("link_y", "identity")),
                scale=float(self.meta.get("link_scale_y", 0.75)),
            )

        if sub == "pnl_detection":
            a_x = self._unit_from_meta("a_x")
            axz = (Z @ a_x).astype(np.float32)
            ayz = (Z @ a_y).astype(np.float32)
            sigma_x = float(self.meta.get("sigma_x", 1.0))
            eps_x = rng.normal(size=Z.shape[0]).astype(np.float32)
            eta_x = (axz + sigma_x * eps_x).astype(np.float32)
            x_raw = _pnl_detection_link(
                eta_x,
                name=str(self.meta.get("link_x", "identity")),
                scale=float(self.meta.get("link_scale_x", 1.0)),
            )
            dep_input = str(self.meta.get("dep_input", "x_raw"))
            raw_signal = x_raw if dep_input == "x_raw" else eta_x
            mean = float(self.meta.get("signal_mean", 0.0))
            std = max(float(self.meta.get("signal_std", 1.0)), 1e-8)
            signal = ((raw_signal - mean) / std).astype(np.float32)
            eta_y = (ayz + self.beta * signal + sigma_y * eps_y).astype(np.float32)
            return _pnl_detection_link(
                eta_y,
                name=str(self.meta.get("link_y", "identity")),
                scale=float(self.meta.get("link_scale_y", 1.0)),
            )

        raise ValueError(
            f"Oracle sampler does not support standard_pnl sub_benchmark={sub!r}"
        )

    def _sample_sdb(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        a_y = self._unit_from_meta("a_y")
        sigma_y = float(self.meta.get("sigma_y", 1.0))
        eps = rng.normal(size=Z.shape[0]).astype(np.float32)
        return (self._sdb_m_y(Z, a_y) + sigma_y * eps).astype(np.float32)

    def sample(self, Z: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        Z_raw = self._raw_z(Z)
        source = str(self.meta.get("source", self.benchmark))
        if source == "standard_pnl_mixed" or self.benchmark == "standard_pnl_mixed":
            y_raw = self._sample_standard_pnl(Z_raw, rng)
        elif source == "sdb" or self.benchmark == "sdb":
            y_raw = self._sample_sdb(Z_raw, rng)
        else:
            raise ValueError(
                f"Oracle sampler unsupported for benchmark/source {self.benchmark!r}/{source!r}"
            )
        return self._standardize_y(y_raw)


def build_y_sampler(
    name: str, seed: int = 0, data=None, standardizer=None
) -> YGivenZSampler:
    name = str(name).lower()
    if name in {"learned_gaussian", "gaussian"}:
        return LearnedGaussianSampler()
    if name in {
        "standardized_residual_bootstrap",
        "std_residual_bootstrap",
        "residual_bootstrap",
    }:
        return StandardizedResidualBootstrapSampler(random_state=int(seed))
    if name == "oracle":
        return OracleYGivenZSampler(data=data, standardizer=standardizer)
    raise ValueError(f"Unknown Y|Z sampler {name!r}")
