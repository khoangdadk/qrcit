"""Build the mutation-only CCLE cohort from mutation.txt and response.csv."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ._utils import ROOT

REPORT_FEATURES = [
    "BRAF.V600E_MUT",
    "BRAF.MC_MUT",
    "HIP1_MUT",
    "CDC42BPA_MUT",
    "THBS3_MUT",
    "DNMT1_MUT",
    "PRKD1_MUT",
    "FLT3_MUT",
    "PIP5K1A_MUT",
    "MAP3K5_MUT",
]
MAPK_GENES = {
    "BRAF",
    "RAF1",
    "ARAF",
    "KRAS",
    "NRAS",
    "HRAS",
    "MAP2K1",
    "MAP2K2",
    "MAPK1",
    "MAPK3",
    "NF1",
    "SPRY1",
    "SPRY2",
    "DUSP4",
    "DUSP6",
    "SOS1",
    "SOS2",
    "GRB2",
    "SHC1",
    "MAP3K1",
    "MAP3K5",
    "MAP3K7",
    "MAPK8",
    "MAPK9",
    "MAPK14",
}


def load_raw(raw_dir):
    """Align PLX4720 Amax responses with binary mutation features."""
    mutations = pd.read_csv(
        raw_dir / "mutation.txt", sep="\t", header=2, index_col=1
    ).iloc[:, 1:]
    mutations = mutations.loc[[str(name).endswith("_MUT") for name in mutations.index]]
    response = pd.read_csv(raw_dir / "response.csv", index_col=[0, 2])

    # MultiIndex levels retain the cell order used by the experiment.
    cells = [
        cell
        for cell in response.index.levels[0]
        if cell in mutations.columns and (cell, "PLX4720") in response.index
    ]
    x = np.asarray([mutations[cell].values for cell in cells])
    y = np.asarray(
        [response.loc[(cell, "PLX4720"), "Amax"] for cell in cells], dtype=float
    )
    return x, y, mutations.index.astype(str), cells


def prepare(raw_dir, out):
    """Apply the correlation screen and retain the specified report/MAPK genes."""
    x, y, names, cells = load_raw(Path(raw_dir))
    correlations = np.array(
        [
            abs(np.corrcoef(column, y)[0, 1]) if np.std(column) > 0 else 0.0
            for column in x.T
        ]
    )
    correlations = np.nan_to_num(correlations, nan=0.0, posinf=0.0, neginf=0.0)
    pearson_mask = correlations >= 0.10

    # Add required genes without changing the raw mutation-feature order.
    forced = np.array(
        [
            name in REPORT_FEATURES
            or name.removesuffix("_MUT").split(".")[0] in MAPK_GENES
            for name in names
        ]
    )
    selected = pearson_mask | forced
    matrix, selected_names = x[:, selected], names[selected]
    if matrix.shape != (474, 75):
        raise ValueError(f"Expected a 474 x 75 matrix; got {matrix.shape}")

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "ccle_plx4720.npz",
        X_selected=matrix,
        y_response=y,
        cell_names=np.asarray(cells, dtype=str),
        feature_names=np.asarray(selected_names, dtype=str),
        pearson_mask=pearson_mask,
        augmented_mask=selected,
        all_mutation_features=np.asarray(names, dtype=str),
        correlations=correlations,
    )
    pd.DataFrame(matrix, index=cells, columns=selected_names).assign(
        PLX4720_Amax=y
    ).to_csv(out / "cohort.csv", index_label="cell_line")
    pd.DataFrame(
        dict(
            feature=selected_names,
            mutation_count=matrix.sum(axis=0),
            pearson_selected=pearson_mask[selected],
        )
    ).to_csv(out / "feature_inventory.csv", index=False)
    return matrix.shape


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=ROOT / "data/ccle/raw")
    parser.add_argument("--out", type=Path, default=Path("artifacts/ccle-data"))
    args = parser.parse_args()
    shape = prepare(args.raw, args.out)
    print(f"Saved {shape[0]} cell lines and {shape[1]} mutation features to {args.out}")


if __name__ == "__main__":
    main()
