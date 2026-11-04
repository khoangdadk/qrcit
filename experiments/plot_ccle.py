"""Plot PIP5K1A and HIP1 contrasts from saved cross-fitted quantile models."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._utils import atomic_json
from .ccle import REFERENCE, validate

FEATURES = ("PIP5K1A_MUT", "HIP1_MUT")
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 6.5,
    "axes.titlesize": 7.3,
    "axes.labelsize": 6.5,
    "xtick.labelsize": 6.0,
    "ytick.labelsize": 6.0,
    "axes.linewidth": 0.75,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def read_records(paths, hip1_seed=0, pip5k1a_seed=0):
    selected = {}
    for path in paths:
        files = [path] if path.is_file() else sorted(path.glob("*.json"))
        for file in files:
            rec = json.loads(file.read_text())
            if rec.get("status") != "fresh_model_fit":
                continue
            feature = rec["identity"]["feature"]
            if feature not in FEATURES or (
                feature == "HIP1_MUT" and rec["identity"]["seed"] != hip1_seed
            ):
                continue
            if feature == "PIP5K1A_MUT" and rec["identity"]["seed"] != pip5k1a_seed:
                continue
            validate(rec)
            if feature in selected and selected[feature] != rec:
                raise ValueError(f"Multiple conflicting records for {feature}")
            selected[feature] = rec
    if set(selected) != set(FEATURES):
        raise ValueError(f"Need PIP5K1A seed {pip5k1a_seed} and HIP1 seed {hip1_seed}")
    return selected


def build_figure(records):
    width, height = 3.61, 1.35
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width, height), dpi=220)
        for feature, x in zip(FEATURES, [0.40, 2.12]):
            ax = fig.add_axes([x / width, 0.29 / height, 1.29 / width, 0.60 / height])
            rec = records[feature]
            u = np.asarray([row["u"] for row in rec["curves"]])
            folds = np.asarray(rec["fold_contrast"], dtype=float)
            mean, sd = folds.mean(axis=0), folds.std(axis=0)
            ax.fill_between(
                u, mean - sd, mean + sd, color="#d9d9d9", alpha=0.78, linewidth=0
            )
            ax.plot(u, mean, "o-", color="#111111", lw=1.35, ms=2.0)
            ax.axhline(0, color="#777777", ls="--", lw=0.8)
            ax.set_title(feature, fontsize=7, pad=4)
            ax.set(xlim=(0, 1), xticks=[0, 0.5, 1])
            ax.set_xlabel(r"Quantile level $u$", labelpad=0.8)
            if feature == "PIP5K1A_MUT":
                ax.set_ylabel(r"Deformation $\Delta(u)$", labelpad=0.8)
            # Fixed axes make plots from separate runs directly comparable.
            limits = (-0.30, 5.75) if feature == "PIP5K1A_MUT" else (-13.0, 0.5)
            if (mean - sd).min() < limits[0] or (mean + sd).max() > limits[1]:
                raise ValueError("Curve exceeds the figure axis limits")
            ax.set_ylim(*limits)
            for spine in ax.spines.values():
                spine.set_linewidth(0.75)
                spine.set_color("#222222")
            ax.tick_params(pad=1.5, length=2, width=0.7)
        fig.text(
            0.40 / width,
            1.22 / height,
            "B. CCLE / PLX4720",
            fontsize=8.5,
            weight="bold",
        )
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs="+", default=[REFERENCE])
    p.add_argument("--hip1-seed", type=int, default=0)
    p.add_argument("--pip5k1a-seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("artifacts/ccle"))
    args = p.parse_args()
    records = read_records(args.runs, args.hip1_seed, args.pip5k1a_seed)
    fig = build_figure(records)
    args.out.mkdir(parents=True, exist_ok=True)
    stem = "figure1b"
    for ext in ["pdf", "png", "svg"]:
        with plt.rc_context(STYLE):
            fig.savefig(args.out / f"{stem}.{ext}", dpi=300)
    plt.close(fig)
    metadata = dict(
        source="saved fold contrasts",
        band="mean +/- population SD across folds",
        results={
            f: dict(
                seed=records[f]["identity"]["seed"],
                p_value=records[f]["result"]["p_global"],
            )
            for f in FEATURES
        },
    )
    atomic_json(args.out / f"{stem}.json", metadata)
    print(f"Saved {stem} to {args.out}")


if __name__ == "__main__":
    main()
