"""Train/test standardization utilities for CFIT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Standardizer:
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    z_mean: np.ndarray
    z_std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> "Standardizer":
        def ms(a):
            m = a.mean(axis=0, keepdims=True)
            s = a.std(axis=0, keepdims=True) + 1e-8
            return m.astype(np.float32), s.astype(np.float32)

        xm, xs = ms(X)
        ym, ys = ms(Y)
        zm, zs = ms(Z)
        return Standardizer(xm, xs, ym, ys, zm, zs)

    def transform(
        self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            ((X - self.x_mean) / self.x_std).astype(np.float32),
            ((Y - self.y_mean) / self.y_std).astype(np.float32),
            ((Z - self.z_mean) / self.z_std).astype(np.float32),
        )
