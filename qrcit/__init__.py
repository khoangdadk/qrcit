"""Quantile-resolved conditional independence testing."""

from dataclasses import dataclass

from .data import CITData
from .method import QuantileTest, QuantileTestConfig
from .sliced import SlicedQRCIT, SlicedQRCITConfig

__version__ = "1.0.0"


@dataclass
class QRCITConfig(QuantileTestConfig):
    """Default scalar-test configuration with joint residual training."""

    n_folds: int = 5
    n_null_draws: int = 20
    u_grid_size: int = 9
    hidden_dims: tuple[int, ...] = (32, 32)
    batch_size: int = 64
    weight_decay: float = 1e-3
    y_resampler: str = "standardized_residual_bootstrap"
    device: str = "cpu"


class QRCIT(QuantileTest):
    """Test scalar X against Y conditional on Z using cross-fitted quantiles."""

    def __init__(self, config=None, alpha=0.05):
        super().__init__(QRCITConfig() if config is None else config, alpha=alpha)


def test_independence(X, Y, Z, *, config=None, alpha=0.05):
    """Return a MethodResult for scalar X; arrays have shape (n, dimension)."""
    cfg = config or QRCITConfig()
    if isinstance(cfg, dict):
        cfg = QRCITConfig(**cfg)
    data = CITData(
        X=X,
        Y=Y,
        Z=Z,
        benchmark="custom",
        variant="custom",
        beta=0.0,
        seed=int(cfg.seed),
    )
    result = QRCIT(cfg, alpha=alpha).safe_test(data)
    if result.error:
        raise RuntimeError(result.error)
    return result


__all__ = [
    "QRCIT",
    "QRCITConfig",
    "CITData",
    "SlicedQRCITConfig",
    "SlicedQRCIT",
    "test_independence",
]
