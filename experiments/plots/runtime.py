import matplotlib.pyplot as plt


def plot_runtime(runtime, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    METHOD_ORDER = ["QRCIT", "GCM", "SGMCIT", "CDCIT", "NNLSCIT", "KCI", "PM-KCI"]
    STYLE = {
        "QRCIT": dict(color="#111111", lw=1.10, marker="o", ls="-", ms=2.45, zorder=10),
        "GCM": dict(color="#F2B134", lw=0.95, marker="o", ls="-", ms=2.20, zorder=5),
        "SGMCIT": dict(color="#2F80ED", lw=0.95, marker="o", ls=":", ms=2.20, zorder=5),
        "CDCIT": dict(color="#9B5DE5", lw=0.95, marker="o", ls="--", ms=2.20, zorder=4),
        "NNLSCIT": dict(
            color="#F28CA8", lw=1.00, marker="o", ls="-", ms=2.35, zorder=6
        ),
        "KCI": dict(color="#2E8B57", lw=0.90, marker="o", ls="-.", ms=2.10, zorder=2),
        "PM-KCI": dict(
            color="#F15BB5", lw=0.90, marker="o", ls="-.", ms=2.10, zorder=2
        ),
    }
    figsize = (2.0, 1.70)
    fs_base = 5.9
    legend_fs = 4.2
    lw_scale = 0.88
    ms_scale = 0.82
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": fs_base,
            "axes.titlesize": fs_base + 0.4,
            "axes.labelsize": fs_base,
            "xtick.labelsize": fs_base - 0.5,
            "ytick.labelsize": fs_base - 0.5,
            "legend.fontsize": legend_fs,
            "axes.linewidth": 0.62,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=300)
    fig.subplots_adjust(left=0.24, right=0.995, top=0.985, bottom=0.235)
    x_order = [250, 500, 1000, 1500]
    plot_df = runtime[runtime["fit_runtime_for_report_sec"] > 0].copy()
    handles = {}
    for method in METHOD_ORDER:
        ss = plot_df[plot_df["method"] == method].sort_values("N")
        if ss.empty:
            continue
        yvals = []
        for x in x_order:
            vals = ss.loc[ss["N"] == x, "fit_runtime_for_report_sec"].values
            yvals.append(float(vals[0]) if len(vals) else float("nan"))
        st = STYLE[method].copy()
        st["lw"] *= lw_scale
        st["ms"] *= ms_scale
        (line,) = ax.plot(x_order, yvals, label=method, **st)
        handles[method] = line
    ax.set_yscale("log")
    ax.set_xlabel(r"$N$", labelpad=0.25)
    ax.set_ylabel("Runtime (s, log)", labelpad=0.6)
    ax.set_xticks(x_order)
    ax.set_xticklabels(["250", "500", "1000", "1500"])
    ax.grid(False)
    ax.margins(x=0.055)
    ax.tick_params(length=2.0, width=0.56, direction="out", pad=0.9)
    for spine in ["top", "bottom", "left", "right"]:
        ax.spines[spine].set_visible(True)
        ax.spines[spine].set_linewidth(0.62)
    ordered_labels = [m for m in METHOD_ORDER if m in handles]
    ordered_handles = [handles[m] for m in ordered_labels]
    ax.legend(
        ordered_handles,
        ordered_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=len(ordered_labels),
        frameon=False,
        fancybox=False,
        handlelength=0.95,
        columnspacing=0.30,
        handletextpad=0.12,
        borderpad=0.0,
        labelspacing=0.0,
    )
    base = out_dir / "pnl_runtime_vs_N_figsize_2_0"
    fig.savefig(base.with_suffix(".png"), dpi=500, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
