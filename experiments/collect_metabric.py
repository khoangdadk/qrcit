"""Collect all 169 fresh METABRIC gene fits and derive the three selected sets."""

import argparse
import json
from pathlib import Path

import pandas as pd

from ._utils import atomic_json, digest
from .metabric import (
    CONFIG,
    DATA,
    GENES,
    load_config,
    select_genes,
    validate_record,
)


def collect(directories, config=CONFIG, data=DATA, gene_path=GENES):
    cfg = load_config(config)
    genes = gene_path.read_text().splitlines()
    paths = [p for d in directories for p in Path(d).glob("gene_*.json")]
    records = [json.loads(p.read_text()) for p in paths]
    if len(records) != 169 or sorted(
        r["result"]["gene_index"] for r in records
    ) != list(range(169)):
        raise ValueError(
            "Expected exactly 169 unique gene records; partial or duplicate campaigns are rejected"
        )
    if len({json.dumps(r["identity"], sort_keys=True) for r in records}) != 1:
        raise ValueError("Mixed configuration, code, inputs or execution environment")
    identity = records[0]["identity"]
    for key, path in [
        ("config_sha256", config),
        ("data_sha256", data),
        ("genes_sha256", gene_path),
    ]:
        if identity[key] != digest(path):
            raise ValueError(f"Input/configuration mismatch: {key}")
    records.sort(key=lambda r: r["result"]["gene_index"])
    for i, record in enumerate(records):
        if record["config"] != cfg:
            raise ValueError("Inconsistent embedded configuration")
        validate_record(record, genes[i], i, cfg)
    rows = [r["result"] for r in records]
    screen = dict(
        kind="fresh_metabric_screen",
        identity=identity,
        config=cfg,
        n_genes=169,
        p_values=[dict(gene=r["gene"], p_value=r["neural_cfit_p_value"]) for r in rows],
        selected_genes=select_genes(rows),
    )
    return screen, rows, [c for r in records for c in r["curves"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directories", nargs="+", type=Path)
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--data", type=Path, default=DATA)
    p.add_argument("--genes", type=Path, default=GENES)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    screen, rows, curves = collect(args.directories, args.config, args.data, args.genes)
    atomic_json(args.out / "screen.json", screen)
    for r in rows:
        for key in ["null_stats", "train_logs"]:
            r[key] = json.dumps(r[key])
    pd.DataFrame(rows).to_csv(args.out / "gene_results.csv", index=False)
    pd.DataFrame(curves).to_csv(args.out / "fold_curves.csv", index=False)
    print(
        json.dumps(
            {name: len(genes) for name, genes in screen["selected_genes"].items()},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
