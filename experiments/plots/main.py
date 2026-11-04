"""Main synthetic comparison figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

NS = [250, 500, 1000, 1500]

DZS = [5, 10, 20, 50]

ALPHA = 0.05

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
    # Colorful, publication-style palette; small circular markers for all methods.
    "QRCIT": dict(color="#111111", lw=1.10, marker="o", ls="-", ms=2.45, zorder=10),
    "QRCIT+PG": dict(color="#8B1E3F", lw=1.10, marker="o", ls="--", ms=2.40, zorder=9),
    "QRCIT+DE": dict(color="#8C6D1F", lw=1.10, marker="o", ls="-", ms=2.40, zorder=8),
    "GCM": dict(color="#F2B134", lw=0.95, marker="o", ls="-", ms=2.20, zorder=5),
    "SGMCIT": dict(color="#2F80ED", lw=0.95, marker="o", ls=":", ms=2.20, zorder=5),
    "CDCIT": dict(color="#9B5DE5", lw=0.95, marker="o", ls="--", ms=2.20, zorder=4),
    "NNLSCIT": dict(color="#F28CA8", lw=1.00, marker="o", ls="-", ms=2.35, zorder=6),
    "KCI": dict(color="#2E8B57", lw=0.90, marker="o", ls="-.", ms=2.10, zorder=2),
    "PM-KCI": dict(color="#F15BB5", lw=0.90, marker="o", ls="-.", ms=2.10, zorder=2),
}


def _group_center_x(fig, axes_group):
    """Return horizontal center of a group of axes in figure coordinates."""
    boxes = [ax.get_position() for ax in axes_group]
    left = min(b.x0 for b in boxes)
    right = max(b.x1 for b in boxes)
    return 0.5 * (left + right)


def plot_8panel(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Single-column configuration only.
    # The 4-column layout is too dense at a strict 3.55-inch width,
    # so use a slightly wider single-column export and controlled spacing.
    figsize = (6.95, 3.02)
    fs_base = 5.9
    legend_fs = 5.2
    lw_scale = 0.88
    ms_scale = 0.88
    ncol_legend = len(METHOD_ORDER)
    legend_y = 0.023
    left_adj, right_adj, top_adj, bottom_adj = 0.048, 0.992, 0.835, 0.145
    wspace_adj, hspace_adj = 0.20, 0.28

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": fs_base,
            "axes.titlesize": fs_base + 0.4,
            "axes.labelsize": fs_base,
            "xtick.labelsize": fs_base - 0.4,
            "ytick.labelsize": fs_base - 0.4,
            "legend.fontsize": legend_fs,
            "axes.linewidth": 0.62,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(2, 4, figsize=figsize, sharey=False)
    specs = [
        (0, 0, "PNL", "N", "Power", "Power (↑)"),
        (0, 1, "PNL", "N", "Type-I", "Type-I (↓)"),
        (0, 2, "SDB", "N", "Power", "Power (↑)"),
        (0, 3, "SDB", "N", "Type-I", "Type-I (↓)"),
        (1, 0, "PNL", "dz", "Power", ""),
        (1, 1, "PNL", "dz", "Type-I", ""),
        (1, 2, "SDB", "dz", "Power", ""),
        (1, 3, "SDB", "dz", "Type-I", ""),
    ]
    handles = {}
    for row, col, bench, sweep, metric, title in specs:
        ax = axes[row, col]
        sub = df[
            (df["benchmark"] == bench)
            & (df["sweep"] == sweep)
            & (df["metric"] == metric)
        ]
        x_order = NS if sweep == "N" else DZS
        for method in METHOD_ORDER:
            ss = sub[sub["method"].astype(str) == method]
            if ss.empty:
                continue
            yvals = []
            for xv in x_order:
                vals = ss[ss["x_value"] == xv]["value"].dropna().values
                yvals.append(vals[0] if len(vals) else np.nan)
            if all(np.isnan(v) for v in yvals):
                continue
            st = STYLE[method].copy()
            st["lw"] *= lw_scale
            st["ms"] *= ms_scale
            (line,) = ax.plot(x_order, yvals, label=method, **st)
            handles[method] = line

        if metric == "Type-I":
            ax.axhline(ALPHA, color="0.2", ls=":", lw=0.65)
            ax.set_ylim(-0.02, 0.50)
        else:
            ax.set_ylim(-0.03, 1.05)

        ax.set_xticks(x_order)
        # Compact labels avoid crowding when exporting the single-column version.
        if sweep == "N":
            ax.set_xticklabels(["250", "500", "1000", "1500"])
        else:
            ax.set_xticklabels(["5", "10", "20", "50"])
        ax.set_xlabel("N" if sweep == "N" else r"$d_Z$", labelpad=1.0)
        ax.set_ylabel("")
        if title:
            ax.set_title(title, pad=3.0, loc="center")

        # Match the clean boxed style in the reference: no internal grid, all four spines visible.
        ax.grid(False)
        ax.margins(x=0.055)
        ax.tick_params(length=2.1, width=0.58, direction="out", pad=1.0)
        for spine in ["top", "bottom", "left", "right"]:
            ax.spines[spine].set_visible(True)
            ax.spines[spine].set_linewidth(0.62)

    ordered_handles = [handles[m] for m in METHOD_ORDER if m in handles]
    ordered_labels = [m for m in METHOD_ORDER if m in handles]
    fig.legend(
        ordered_handles,
        ordered_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=ncol_legend,
        frameon=False,
        fancybox=False,
        handlelength=1.18,
        columnspacing=0.60,
        handletextpad=0.28,
        borderpad=0.0,
        labelspacing=0.0,
    )

    fig.subplots_adjust(
        left=left_adj,
        right=right_adj,
        top=top_adj,
        bottom=bottom_adj,
        wspace=wspace_adj,
        hspace=hspace_adj,
    )

    # Add centered group titles after layout is fixed, so they remain centered over the panel blocks.
    fig.canvas.draw()
    y_title = max(ax.get_position().y1 for ax in axes[0, :]) + 0.065
    fig.text(
        _group_center_x(fig, [axes[0, 0], axes[0, 1]]),
        y_title,
        "(a) Synthetic PNL",
        ha="center",
        va="bottom",
        fontsize=fs_base + 0.9,
        fontweight="bold",
    )
    fig.text(
        _group_center_x(fig, [axes[0, 2], axes[0, 3]]),
        y_title,
        "(b) Merged SDB",
        ha="center",
        va="bottom",
        fontsize=fs_base + 0.9,
        fontweight="bold",
    )

    base = out_dir / "pnl_sdb_8panel_metrics_single_column_only_v14"
    fig.savefig(
        base.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.025
    )
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.025)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)
