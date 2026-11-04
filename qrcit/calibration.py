"""Predictive-gain statistics and conditional-randomization p-values."""

from __future__ import annotations

import time

import numpy as np

from .training import predict_q1_for_y


def pinball_np(x: np.ndarray, q: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """Return pinball losses for x:(n,1), q:(n,m), u_grid:(m,)."""
    x1 = np.asarray(x, dtype=np.float32).reshape(-1, 1)
    q = np.asarray(q, dtype=np.float32)
    u = np.asarray(u_grid, dtype=np.float32).reshape(1, -1)
    r = x1 - q
    return np.maximum(u * r, (u - 1.0) * r)


def predictive_gain_stat(
    x: np.ndarray, q0: np.ndarray, q1: np.ndarray, u_grid: np.ndarray
) -> float:
    """Mean pinball-loss reduction from Q0 to Q1."""
    l0 = pinball_np(x, q0, u_grid)
    l1 = pinball_np(x, q1, u_grid)
    return float(np.mean(l0 - l1))


def predictive_gain_region_stat(
    x: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    u_grid: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Region-restricted predictive-gain statistic.

    This is the SDB oracle-region analogue of the global CFIT statistic:

        E_{(i,u) in region} [rho_u(X_i - Q0_iu) - rho_u(X_i - Q1_iu)].

    Larger values mean that including Y improves the conditional-quantile
    prediction within the known SDB affected region.
    """
    l0 = pinball_np(x, q0, u_grid)
    l1 = pinball_np(x, q1, u_grid)
    gain = l0 - l1
    m = np.asarray(mask, dtype=bool)
    if m.shape != gain.shape:
        raise ValueError(
            f"Region mask shape {m.shape} does not match gain shape {gain.shape}."
        )
    if int(m.sum()) == 0:
        return float("nan")
    return float(np.nanmean(gain[m]))


def deformation_energy_stat(delta: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Mean squared deformation, optionally restricted by a boolean mask."""
    d = np.asarray(delta, dtype=np.float32)
    if mask is None:
        return float(np.nanmean(d**2))
    m = np.asarray(mask, dtype=bool)
    if m.shape != d.shape:
        raise ValueError(
            f"Region mask shape {m.shape} does not match delta shape {d.shape}."
        )
    if int(m.sum()) == 0:
        return float("nan")
    return float(np.nanmean(d[m] ** 2))


def _normalize_region_statistics(
    region_statistics=None, region_statistic: str | None = None
) -> list[str]:
    """Normalize old and new SDB-region statistic config formats."""
    if region_statistics is None:
        if region_statistic is None:
            region_statistics = ["deformation_energy"]
        else:
            region_statistics = [region_statistic]
    if isinstance(region_statistics, str):
        # Accept comma-separated strings for CLI/YAML convenience.
        region_statistics = [
            x.strip() for x in region_statistics.split(",") if x.strip()
        ]
    out = []
    for st in region_statistics:
        st = str(st).strip()
        if not st:
            continue
        if st not in {"deformation_energy", "predictive_gain_region"}:
            raise NotImplementedError(
                "Unsupported region statistic={!r}. Supported values are "
                "'deformation_energy' and 'predictive_gain_region'.".format(st)
            )
        if st not in out:
            out.append(st)
    if not out:
        out = ["deformation_energy"]
    return out


def _compute_region_statistic(
    statistic: str,
    x: np.ndarray,
    q0: np.ndarray,
    q1: np.ndarray,
    u_grid: np.ndarray,
    region_mask: np.ndarray,
) -> float:
    """Compute one SDB-region statistic for observed or null predictions."""
    if statistic == "deformation_energy":
        return deformation_energy_stat(q1 - q0, region_mask)
    if statistic == "predictive_gain_region":
        return predictive_gain_region_stat(x, q0, q1, u_grid, region_mask)
    raise NotImplementedError(f"Unsupported region statistic={statistic!r}.")


def conditional_randomization_pvalues_from_folds(
    x: np.ndarray,
    z: np.ndarray,
    folds: list[dict],
    y_sampler,
    u_grid: np.ndarray,
    n_null_draws: int,
    seed: int,
    device: str = "auto",
    region_mask: np.ndarray | None = None,
    region_statistic: str = "deformation_energy",
    region_statistics=None,
    decision_region_statistic: str | None = None,
) -> dict:
    """Cross-fit conditional-randomization p-values for global and region statistics.

    Always computes the global predictive-gain p-value. If
    ``region_mask`` is provided, also computes one or more SDB oracle-region
    p-values in a *single calibration pass*. This avoids retraining or
    resampling twice when comparing ``deformation_energy`` and
    ``predictive_gain_region``.

    ``region_statistic`` supplies the default when ``region_statistics`` is empty.
    ``p_value_sdb_region`` selects one region p-value for diagnostic decisions.
    """
    rng = np.random.default_rng(int(seed))
    n = x.shape[0]
    m = len(u_grid)
    q0_full = np.zeros((n, m), dtype=np.float32)
    q1_full = np.zeros((n, m), dtype=np.float32)
    for fd in folds:
        idx = fd["test_idx"]
        q0_full[idx] = fd["q0"]
        q1_full[idx] = fd["q1"]

    t_obs = predictive_gain_stat(x, q0_full, q1_full, u_grid)

    use_region = region_mask is not None
    if use_region:
        region_mask = np.asarray(region_mask, dtype=bool)
        if region_mask.shape != q1_full.shape:
            raise ValueError(
                f"region_mask shape {region_mask.shape} does not match prediction shape {q1_full.shape}."
            )
        if int(region_mask.sum()) == 0:
            raise ValueError("region_mask is empty; cannot compute SDB-region p-value.")
        stats = _normalize_region_statistics(
            region_statistics=region_statistics, region_statistic=region_statistic
        )
        if decision_region_statistic is None:
            decision_region_statistic = stats[0]
        decision_region_statistic = str(decision_region_statistic)
        if decision_region_statistic not in stats:
            stats.append(decision_region_statistic)
        # Validate any decision stat appended above.
        stats = _normalize_region_statistics(region_statistics=stats)
        if decision_region_statistic not in stats:
            raise ValueError(
                f"decision_region_statistic={decision_region_statistic!r} is not available in {stats}."
            )
        region_obs = {
            st: _compute_region_statistic(st, x, q0_full, q1_full, u_grid, region_mask)
            for st in stats
        }
        region_null_stats = {st: [] for st in stats}
    else:
        stats = []
        decision_region_statistic = "none"
        region_obs = {}
        region_null_stats = {}

    start = time.perf_counter()
    null_stats = []
    for _ in range(int(n_null_draws)):
        q1_null_full = np.zeros_like(q1_full)
        for fd in folds:
            idx = fd["test_idx"]
            y_null = fd["sampler"].sample(fd["Z_test_std"], rng)
            q1_null_full[idx] = predict_q1_for_y(
                fd["model"], y_null, fd["Z_test_std"], u_grid, device=device
            )
        null_stats.append(predictive_gain_stat(x, q0_full, q1_null_full, u_grid))
        if use_region:
            for st in stats:
                region_null_stats[st].append(
                    _compute_region_statistic(
                        st, x, q0_full, q1_null_full, u_grid, region_mask
                    )
                )

    null_stats = np.asarray(null_stats, dtype=np.float32)
    p_value = float((1.0 + np.sum(null_stats >= t_obs)) / (1.0 + len(null_stats)))

    region_p_values = {}
    region_null_arrays = {}
    if use_region:
        for st in stats:
            arr = np.asarray(region_null_stats[st], dtype=np.float32)
            region_null_arrays[st] = arr
            region_p_values[st] = float(
                (1.0 + np.sum(arr >= region_obs[st])) / (1.0 + len(arr))
            )
        p_value_region = region_p_values.get(decision_region_statistic, float("nan"))
        selected_region_obs = region_obs.get(decision_region_statistic, float("nan"))
        selected_region_nulls = region_null_arrays.get(
            decision_region_statistic, np.asarray([], dtype=np.float32)
        )
    else:
        p_value_region = float("nan")
        selected_region_obs = float("nan")
        selected_region_nulls = np.asarray([], dtype=np.float32)

    cal_time = time.perf_counter() - start
    out = {
        "p_value": p_value,
        "null_stats": null_stats,
        "calibration_time_sec": cal_time,
        "t_obs": float(t_obs),
        # Selected statistic for the optional region decision.
        "p_value_sdb_region": float(p_value_region),
        "T_sdb_region_obs_calib": float(selected_region_obs),
        "T_sdb_region_nulls": selected_region_nulls,
        "sdb_region_statistic": (
            str(decision_region_statistic) if use_region else "none"
        ),
        # All requested statistics share the same randomization draws.
        "sdb_region_statistics": list(stats),
        "sdb_region_decision_statistic": (
            str(decision_region_statistic) if use_region else "none"
        ),
        "sdb_region_p_values": region_p_values,
        "sdb_region_obs": region_obs,
        "sdb_region_nulls_by_statistic": region_null_arrays,
    }
    return out
