"""Validate saved METABRIC screening statistics and selected gene sets."""

import json

import numpy as np
import pandas as pd

from ._utils import ROOT, digest
from .metabric import THRESHOLDS

RESULTS = ROOT / "results/metabric"


def verify():
    """Recompute all screening p-values and check the saved downstream summary."""
    provenance = json.loads((RESULTS / "provenance.json").read_text())
    for name, expected in provenance["archived_checksums"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"METABRIC checksum mismatch: {name}")

    genes = pd.read_csv(RESULTS / "gene_results.csv")
    summary = pd.read_csv(RESULTS / "downstream_summary.csv")
    paper = pd.read_csv(RESULTS / "paper_summary.csv")
    cohort = pd.read_csv(ROOT / "data/metabric/cohort.csv")
    names = (ROOT / "data/metabric/genes.txt").read_text().splitlines()
    assert len(genes) == len(names) == genes.gene.nunique() == 169
    assert set(genes.gene) == set(names)
    assert len(cohort) == 1491 and set(names).issubset(cohort.columns)
    assert genes.status.eq("ok").all()
    assert genes.n.eq(1491).all() and genes.dim_z.eq(180).all()

    for row in genes.itertuples():
        null = np.asarray(json.loads(row.null_stats), dtype=float)
        logs = json.loads(row.train_logs)
        assert len(null) == 200 and len(logs) == 3, row.gene
        p = (1 + int(np.sum(null >= row.neural_T_obs))) / 201
        assert np.isclose(p, row.neural_cfit_p_value, rtol=0, atol=1e-14), row.gene
        assert np.isclose(null.mean(), row.neural_T_null_mean, rtol=0, atol=1e-13)
        assert np.isclose(null.std(), row.neural_T_null_std, rtol=0, atol=1e-13)

    selected = {}
    for name, threshold in THRESHOLDS.items():
        selected[name] = genes.loc[
            genes.neural_cfit_p_value < threshold, "gene"
        ].tolist()
        row = summary.loc[summary.gene_set == name].iloc[0]
        assert selected[name] == row.selected_genes.split(";"), name
        assert len(selected[name]) == row.n_selected_genes, name
    assert [len(selected[name]) for name in THRESHOLDS] == [10, 19, 26]
    assert genes.cutoff_0_05_selected.equals(genes.neural_cfit_p_value < 0.05)
    assert genes.cutoff_0_10_selected.equals(genes.neural_cfit_p_value < 0.10)

    for name, label in {
        "Neural_CFIT_p010": "QRCIT p<0.10",
        "Neural_CFIT_p005": "QRCIT p<0.05",
        "Neural_CFIT_p001": "QRCIT p<0.01",
        "CDCIT_reported_8": "CDCIT",
    }.items():
        row = summary.loc[summary.gene_set == name].iloc[0]
        reference = paper.loc[paper.method == label].iloc[0]
        assert row.n_repeats == 50 and row.target_transform == "standardize"
        assert row.xgb_search_mode == "tune_once_then_fixed_all_repeats"
        assert np.isclose(row.mean_mse, reference.mean_mse, rtol=0, atol=5e-7)
        assert np.isclose(row.std_mse, reference.std_mse, rtol=0, atol=5e-7)

    return dict(
        n_samples=len(cohort),
        n_genes=len(genes),
        n_null_draws=200,
        selected_counts={name: len(value) for name, value in selected.items()},
        downstream_summary=summary[
            ["gene_set", "n_selected_genes", "mean_mse", "std_mse"]
        ].to_dict("records"),
        downstream_status="saved summary; original per-split MSE rows unavailable",
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
