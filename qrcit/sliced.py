"""Project multivariate X onto scalar targets and aggregate their p-values."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from qrcit.base import CITMethod, MethodResult
from qrcit.data import CITData
from qrcit.method import QuantileTest, QuantileTestConfig


@dataclass
class SlicedQRCITConfig:
    base_config: dict = field(default_factory=dict)
    n_projections: int = 16
    aggregation: str = "cauchy"  # cauchy | fisher | min_p | mean_p
    projection_seed: int = 123
    include_coordinate_projections: bool = True
    max_metadata_projection_values: int = 32


def _random_unit_directions(
    d: int, n_proj: int, rng: np.random.Generator, include_coordinate: bool = True
) -> np.ndarray:
    dirs = []
    if include_coordinate:
        for j in range(min(d, n_proj)):
            e = np.zeros(d, dtype=np.float32)
            e[j] = 1.0
            dirs.append(e)
    while len(dirs) < int(n_proj):
        v = rng.normal(size=d).astype(np.float32)
        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            dirs.append(v / norm)
    return np.stack(dirs, axis=0).astype(np.float32)


def combine_pvalues(pvals: np.ndarray, method: str = "cauchy") -> float:
    p = np.asarray(pvals, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    method = str(method).lower()
    if method == "min_p":
        return float(min(1.0, len(p) * np.min(p)))
    if method == "mean_p":
        return float(np.mean(p))
    if method == "fisher":
        try:
            from scipy.stats import chi2

            stat = -2.0 * np.sum(np.log(p))
            return float(1.0 - chi2.cdf(stat, 2 * len(p)))
        except Exception:
            # fallback to min-p if scipy is unavailable
            return float(min(1.0, len(p) * np.min(p)))
    if method == "cauchy":
        t = np.tan((0.5 - p) * math.pi)
        cct = float(np.mean(t))
        return float(np.clip(0.5 - math.atan(cct) / math.pi, 0.0, 1.0))
    raise ValueError(f"Unknown p-value aggregation method {method!r}")


class SlicedQRCIT(CITMethod):
    """Run scalar tests on unit projections of X and combine their p-values."""

    name = "sliced_cfit"

    def __init__(
        self, config: SlicedQRCITConfig | dict | None = None, alpha: float = 0.05
    ):
        super().__init__(alpha=alpha)
        if config is None:
            self.config = SlicedQRCITConfig()
        elif isinstance(config, dict):
            self.config = SlicedQRCITConfig(**config)
        else:
            self.config = config
        self.name = "sliced_cfit"

    def test(self, data: CITData) -> MethodResult:
        result = self._empty_result(data)
        rng = np.random.default_rng(int(self.config.projection_seed) + int(data.seed))
        dirs = _random_unit_directions(
            data.d_x,
            int(self.config.n_projections),
            rng,
            include_coordinate=bool(self.config.include_coordinate_projections),
        )
        pvals, stats = [], []
        fit_time = test_time = cal_time = 0.0
        errors = []
        for proj_id, theta in enumerate(dirs):
            x_proj = data.X @ theta.reshape(-1, 1)
            proj_data = CITData(
                X=x_proj.astype(np.float32),
                Y=data.Y,
                Z=data.Z,
                benchmark=data.benchmark,
                variant=data.variant,
                beta=data.beta,
                seed=int(data.seed) + 10007 * int(proj_id),
                meta={**data.meta, "sliced_projection_id": int(proj_id)},
            )
            base_cfg = dict(self.config.base_config)
            base_cfg.setdefault("model_type", "residual_deformation_v2")
            base_method = QuantileTest(QuantileTestConfig(**base_cfg), alpha=self.alpha)
            r = base_method.safe_test(proj_data)
            fit_time += float(r.fit_time_sec)
            test_time += float(r.test_time_sec)
            cal_time += float(r.calibration_time_sec)
            if r.error:
                errors.append(f"projection {proj_id}: {r.error[:300]}")
                continue
            if r.p_value is not None and np.isfinite(float(r.p_value)):
                pvals.append(float(r.p_value))
            if r.statistic is not None and np.isfinite(float(r.statistic)):
                stats.append(float(r.statistic))

        result.fit_time_sec = fit_time
        result.test_time_sec = test_time
        result.calibration_time_sec = cal_time
        result.reported_time_sec = fit_time + test_time
        if pvals:
            result.p_value = combine_pvalues(np.asarray(pvals), self.config.aggregation)
        else:
            result.p_value = None
        result.statistic = float(np.mean(stats)) if stats else None
        keep = int(self.config.max_metadata_projection_values)
        result.metadata = {
            "base_method": "QuantileTest",
            "base_config": dict(self.config.base_config),
            "n_projections_requested": int(self.config.n_projections),
            "n_projections_successful": int(len(pvals)),
            "aggregation": str(self.config.aggregation),
            "projection_seed": int(self.config.projection_seed),
            "projection_p_values": [float(x) for x in pvals[:keep]],
            "projection_statistics": [float(x) for x in stats[:keep]],
            "directions_shape": tuple(dirs.shape),
            "note": "Aggregated projection-wise scalar p-values.",
        }
        if errors:
            result.metadata["projection_errors"] = errors[:5]
            if not pvals:
                result.error = "All sliced CFIT projections failed."
        return result
