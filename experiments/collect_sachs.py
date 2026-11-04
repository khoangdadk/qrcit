"""Validate and summarize a complete fresh 90-query Sachs campaign.

Pass one or more directories containing case_000.json ... case_089.json.
The archived paper result is never used as a substitute for missing fresh cases.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from .verify_sachs import EXPECTED_HASHES, read

ROOT = Path(__file__).resolve().parents[1]


def collect(directories):
    paths = [
        p for directory in directories for p in Path(directory).glob("case_*.json")
    ]
    if len(paths) != 90:
        raise ValueError(f"Expected 90 fresh case files; found {len(paths)}")
    records = [json.loads(p.read_text()) for p in paths]
    identities = {json.dumps(item["identity"], sort_keys=True) for item in records}
    if len(identities) != 1:
        raise ValueError("Mixed runner, input, configuration, device or environment")
    identity = records[0]["identity"]
    original_config = hashlib.sha256(
        (ROOT / "configs/sachs.json").read_bytes()
    ).hexdigest()
    if identity["config_sha256"] != original_config:
        raise ValueError(
            "Expected the checked-in paper recipe, not a smoke or modified configuration"
        )
    for key, path in (
        ("data_sha256", "data/sachs/sachs_1755.csv"),
        ("relations_sha256", "data/sachs/relations.csv"),
    ):
        if identity[key] != EXPECTED_HASHES[path]:
            raise ValueError(f"Wrong historical input: {path}")
    ids = [int(item["result"]["case_id"]) for item in records]
    if sorted(ids) != list(range(90)):
        raise ValueError("Missing or duplicate Sachs case IDs")
    rows = [
        item["result"]
        for item in sorted(records, key=lambda item: item["result"]["case_id"])
    ]
    relations = read("data/sachs/relations.csv")
    counts = dict(TP=0, FP=0, FN=0, TN=0)
    predictions = []
    for row in rows:
        rel = relations[row["case_id"]]
        if any(
            str(row[key]) != rel[key]
            for key in ("x_name", "y_name", "z_names", "label_ci")
        ):
            raise ValueError(f"Wrong relation for case {row['case_id']}")
        if (
            row["status"] != "ok"
            or row["error"]
            or row["seed"] != 60123 + row["case_id"]
        ):
            raise ValueError(f"Failed or wrong-seed case {row['case_id']}")
        if row["n_splits"] != 3 or row["n_null_draws"] != 200:
            raise ValueError(f"Wrong calibration recipe for case {row['case_id']}")
        p = (row["neural_count_ge_obs"] + 1) / 201
        if abs(p - row["neural_p_value"]) > 1e-14:
            raise ValueError(f"Wrong plus-one p-value for case {row['case_id']}")
        truth = row["label_ci"] == 1
        predicted = p > 1 / 201
        counts[
            (
                "TP"
                if truth and predicted
                else "FN" if truth else "FP" if predicted else "TN"
            )
        ] += 1
        predictions.append(
            dict(
                case_id=row["case_id"],
                label_ci=row["label_ci"],
                p_global=p,
                pred_ci=int(predicted),
            )
        )
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    summary = dict(
        n_cases=90,
        alpha="1/201",
        positive_class="CI",
        identity=identity,
        **counts,
        precision=tp / (tp + fp) if tp + fp else 0,
        recall=tp / (tp + fn),
        F1=2 * tp / (2 * tp + fp + fn),
    )
    return summary, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary, rows = collect(args.directories)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.out / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: v for k, v in summary.items() if k != "identity"}, indent=2))


if __name__ == "__main__":
    main()
