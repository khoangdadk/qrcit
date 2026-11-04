"""Power across structural dependency patterns."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NS = [250, 500, 1000, 1500]

DZS = [5, 10, 20, 50]

ALPHA = 0.05

VARIANTS = [
    ("context_local", "Context-local"),
    ("mean_neutral_shape", "Mean-neutral"),
    ("moving_u_ridge", r"Moving-$u$"),
    ("sign_reversing", "Sign-reversing"),
    ("tail_only", "Tail-only"),
]

METHOD_ORDER = [
    "QRCIT",
    "QRCIT+PG",
    "QRCIT+DE",
    "GCM",
    "SGMCIT",
    "CDCIT",
    "NNLSCIT",
    "KCI",
    "PM-KCI",
]

METHOD_MAP = {
    "cfit_residual": "QRCIT",
    "gcm": "GCM",
    "sgmcit": "SGMCIT",
    "cdcit": "CDCIT",
    "nnlscit": "NNLSCIT",
    "standard_kci": "KCI",
    "powermax_kci": "PM-KCI",
}

STYLE = {
    "QRCIT": dict(color="#111111", lw=0.92, marker="o", ls="-", ms=1.90, zorder=10),
    "QRCIT+PG": dict(color="#8B1E3F", lw=0.88, marker="o", ls="--", ms=1.85, zorder=9),
    "QRCIT+DE": dict(color="#8C6D1F", lw=0.88, marker="o", ls="-", ms=1.85, zorder=8),
    "GCM": dict(color="#F2B134", lw=0.78, marker="o", ls="-", ms=1.75, zorder=5),
    "SGMCIT": dict(color="#2F80ED", lw=0.78, marker="o", ls=":", ms=1.75, zorder=5),
    "CDCIT": dict(color="#9B5DE5", lw=0.78, marker="o", ls="--", ms=1.75, zorder=4),
    "NNLSCIT": dict(color="#F28CA8", lw=0.80, marker="o", ls="-", ms=1.80, zorder=6),
    "KCI": dict(color="#2E8B57", lw=0.74, marker="o", ls="-.", ms=1.70, zorder=2),
    "PM-KCI": dict(color="#F15BB5", lw=0.74, marker="o", ls="-.", ms=1.70, zorder=2),
}

SUMMARY_COLS = [
    "summary_group",
    "variant",
    "method",
    "beta",
    "n_runs",
    "rejection_rate",
    "mean_p_value",
    "decision_rejection_rate",
    "mean_decision_p_value",
    "mean_reported_time_sec",
    "std_reported_time_sec",
    "mean_fit_time_sec",
    "mean_test_time_sec",
    "mean_calibration_time_sec",
    "mean_region_fraction",
    "mean_T_struct_region_ratio",
    "reported_variant",
    "metric_role",
    "type2_error",
    "decision_type2_error",
    "alpha",
]

LONG_COLS = [
    "benchmark",
    "sdb_variant",
    "variant",
    "sub_benchmark",
    "method",
    "n",
    "d_x",
    "d_y",
    "d_z",
    "beta",
    "n_valid",
    "reject_rate",
    "mean_p_value",
    "median_p_value",
    "mean_train_time_report_sec",
    "mean_fit_time_sec",
    "mean_test_time_sec",
    "mean_reported_time_sec",
    "mean_calibration_time_sec",
    "train_time_report_definition",
    "mean_statistic",
    "median_statistic",
    "n_attempted",
    "n_failed",
    "failure_rate",
]


def plot_power_only(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot separated SDB power only: 2 rows x 5 structural patterns.

    Row 1: N sweep, fixed d_z=5.
    Row 2: d_z sweep, fixed N=1000.
    Columns: five SDB structural patterns.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = df[df["metric"] == "Power"].copy()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 5.8,
            "axes.titlesize": 6.4,
            "axes.labelsize": 5.8,
            "xtick.labelsize": 5.1,
            "ytick.labelsize": 5.1,
            "legend.fontsize": 5.2,
            "axes.linewidth": 0.58,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 5, figsize=(7.45, 3.45), sharey=True)
    fig.subplots_adjust(
        left=0.048, right=0.996, top=0.84, bottom=0.155, wspace=0.14, hspace=0.24
    )

    for col, (variant, vlabel) in enumerate(VARIANTS):
        for row, (sweep, x_order, xlabel) in enumerate(
            [("N", NS, "N"), ("dz", DZS, r"$d_Z$")]
        ):
            ax = axes[row, col]
            sub = df[(df["variant"] == variant) & (df["sweep"] == sweep)]

            for method in METHOD_ORDER:
                ss = sub[sub["method"] == method]
                if ss.empty:
                    continue

                y = []
                for xv in x_order:
                    vals = ss[ss["x_value"] == xv]["value"].values
                    y.append(float(vals[0]) if len(vals) else np.nan)
                if all(np.isnan(v) for v in y):
                    continue

                ax.plot(x_order, y, label=method, **STYLE[method])

            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks(x_order)
            ax.set_xticklabels([str(x) for x in x_order])
            ax.set_xlabel(xlabel, labelpad=0.4)

            if col == 0:
                ax.set_ylabel("Power (↑)")

            if row == 0:
                ax.set_title(f"({chr(97+col)}) {vlabel}", pad=4.0, fontweight="bold")

            # Clean boxed style: no internal grid, all four spines visible.
            ax.grid(False)
            ax.margins(x=0.055)
            ax.tick_params(length=1.7, width=0.5, direction="out", pad=0.7)
            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_linewidth(0.58)

    # Common legend
    handle_map = {}
    for ax in axes.flat:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            handle_map[l] = h
    labels = [m for m in METHOD_ORDER if m in handle_map]
    handles = [handle_map[m] for m in labels]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.038),
        ncol=len(labels),
        frameon=False,
        handlelength=1.15,
        columnspacing=0.58,
        handletextpad=0.25,
        borderpad=0.0,
    )

    base = out_dir / "sdb_2x5_power_separated_metrics"
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
