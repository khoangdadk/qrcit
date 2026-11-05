"""Common method interface and result helpers."""

from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any

from qrcit.data import CITData


@dataclass
class MethodResult:
    method: str
    benchmark: str
    variant: str
    beta: float
    seed: int
    n: int
    d_x: int
    d_y: int
    d_z: int
    statistic: float | None = None
    p_value: float | None = None
    reject_alpha_005: bool | None = None
    fit_time_sec: float = 0.0
    test_time_sec: float = 0.0
    calibration_time_sec: float = 0.0
    total_time_sec: float = 0.0
    reported_time_sec: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    # Held-out deformation field, set by quantile tests for scalar X:
    # u_grid has shape (K,), deformation[i, k] = Q1(Y_i, Z_i, u_k) - Q0(Z_i, u_k)
    # in the units of X, with shape (n, K). Not included in to_row().
    u_grid: Any = field(default=None, repr=False, compare=False)
    deformation: Any = field(default=None, repr=False, compare=False)

    def to_row(self) -> dict[str, Any]:
        row = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("u_grid", "deformation")
        }
        row["metadata"] = repr(row["metadata"])
        return row


class CITMethod:
    name: str = "base_method"

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def test(self, data: CITData) -> MethodResult:
        raise NotImplementedError

    def _empty_result(self, data: CITData) -> MethodResult:
        return MethodResult(
            method=self.name,
            benchmark=data.benchmark,
            variant=data.variant,
            beta=data.beta,
            seed=data.seed,
            n=data.n,
            d_x=data.d_x,
            d_y=data.d_y,
            d_z=data.d_z,
        )

    def safe_test(self, data: CITData) -> MethodResult:
        start = time.perf_counter()
        try:
            result = self.test(data)
            result.total_time_sec = time.perf_counter() - start
            if result.reported_time_sec == 0.0:
                result.reported_time_sec = result.fit_time_sec + result.test_time_sec
            if result.p_value is not None:
                result.reject_alpha_005 = bool(result.p_value < self.alpha)
            return result
        except Exception:
            result = self._empty_result(data)
            result.error = traceback.format_exc(limit=5)
            result.total_time_sec = time.perf_counter() - start
            return result
