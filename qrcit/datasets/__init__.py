"""Synthetic data generators and deterministic trial seed formulas."""

from .multivariate import generate_multivariate_simple as multivariate
from .pnl import generate_standard_pnl_mixed as pnl
from .sdb import SDB_VARIANTS
from .sdb import generate_sdb as sdb

__all__ = ["pnl", "sdb", "multivariate", "SDB_VARIANTS", "trial_seed"]


def trial_seed(benchmark, n, d_z, beta, trial_id, variant=None):
    if benchmark == "multivariate":
        return (
            9173 * trial_id
            + int(round(beta * 1000))
            + 11 * len("multivariate_x_multivariate_y")
        )
    seed = n * 17 + d_z * 101 + int(round(beta * 1000)) * 10000 + trial_id
    if benchmark == "sdb":
        order = {
            "context_local": 1,
            "sign_reversing": 2,
            "tail_only": 3,
            "mean_neutral_shape": 4,
            "moving_u_ridge": 5,
        }
        seed += 20000000 + order[variant] * 1000000
    return seed
