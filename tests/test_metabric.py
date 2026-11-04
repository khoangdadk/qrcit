import unittest

import pandas as pd

from experiments.metabric import ROOT, select_genes
from experiments.verify_metabric import verify


class MetabricArchiveTest(unittest.TestCase):
    def test_archived_evidence_and_selection_order(self):
        result = verify()
        self.assertEqual(result["n_genes"], 169)
        self.assertEqual(result["n_null_draws"], 200)
        rows = pd.read_csv(ROOT / "results/metabric/gene_results.csv").to_dict(
            "records"
        )
        selected = select_genes(rows)
        summary = pd.read_csv(
            ROOT / "results/metabric/downstream_summary.csv"
        ).set_index("gene_set")
        for name, genes in selected.items():
            self.assertEqual(genes, summary.loc[name, "selected_genes"].split(";"))


if __name__ == "__main__":
    unittest.main()
