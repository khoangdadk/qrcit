"""Common data container for conditional independence experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class CITData:
    """A unified data object for testing X _||_ Y | Z.

    Shape convention:
        X: (n, d_x), always two-dimensional
        Y: (n, d_y), always two-dimensional
        Z: (n, d_z), always two-dimensional
    """

    X: np.ndarray
    Y: np.ndarray
    Z: np.ndarray
    benchmark: str
    variant: str
    beta: float
    seed: int
    meta: dict[str, Any] = field(default_factory=dict)
    delta_true: Optional[np.ndarray] = None
    delta_struct: Optional[np.ndarray] = None
    u_grid: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.X = ensure_2d_float(self.X)
        self.Y = ensure_2d_float(self.Y)
        self.Z = ensure_2d_float(self.Z)
        n = self.X.shape[0]
        if self.Y.shape[0] != n or self.Z.shape[0] != n:
            raise ValueError(
                f"X,Y,Z must have the same number of rows; got "
                f"{self.X.shape[0]}, {self.Y.shape[0]}, {self.Z.shape[0]}"
            )

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def d_x(self) -> int:
        return self.X.shape[1]

    @property
    def d_y(self) -> int:
        return self.Y.shape[1]

    @property
    def d_z(self) -> int:
        return self.Z.shape[1]


def ensure_2d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(f"Expected 1D or 2D array, got shape {x.shape}")
    return x
