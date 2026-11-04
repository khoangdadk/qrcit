"""Check the archived Sachs result against the published CI-positive metrics.

This reads historical output only. It does not fit a new model.
"""

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_HASHES = {
    "data/sachs/sachs_1755.csv": "0dda72e37fc4d6e6a13bc1cbda6ebf2474093bc27dac7f34f845c7496391cdbe",
    "data/sachs/relations.csv": "9ee7472e10877e7d250caf339ab4c72a8ceecfab9a5935f27e3f4588355e402b",
    "results/sachs/predictions.csv": "f39778f3b7714a669fcba737d9cfd36a4d7cf2173503a5ee6bd0a740ff470ec3",
}


def read(path):
    with (ROOT / path).open(newline="") as f:
        return list(csv.DictReader(f))


def verify():
    for name, expected in EXPECTED_HASHES.items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Historical source checksum mismatch: {name}")
    rows = read("results/sachs/predictions.csv")
    relations = read("data/sachs/relations.csv")
    if len(rows) != 90 or len(relations) != 90:
        raise ValueError("Expected 90 predictions and relations")
    counts = dict(TP=0, FP=0, FN=0, TN=0)
    for case_id, (row, rel) in enumerate(zip(rows, relations)):
        if int(row["case_id"]) != case_id or int(rel["author_case_id"]) != case_id:
            raise ValueError(f"Query order mismatch: {case_id}")
        if any(row[k] != rel[k] for k in ("x_name", "y_name", "z_names", "label_ci")):
            raise ValueError(f"Query content mismatch: {case_id}")
        if row["status"] != "ok" or row["error"] or int(row["seed"]) != 60123 + case_id:
            raise ValueError(f"Incomplete or wrong-seed case: {case_id}")
        if int(row["n_splits"]) != 3 or int(row["n_null_draws"]) != 200:
            raise ValueError(f"Wrong calibration recipe: {case_id}")
        p = (int(row["neural_count_ge_obs"]) + 1) / 201
        if abs(p - float(row["neural_p_value"])) > 1e-14:
            raise ValueError(f"Wrong plus-one p-value: {case_id}")
        truth = int(row["label_ci"]) == 1
        prediction = p > 1 / 201
        key = (
            "TP"
            if truth and prediction
            else "FN" if truth else "FP" if prediction else "TN"
        )
        counts[key] += 1
    if counts != dict(TP=44, FP=16, FN=6, TN=24):
        raise ValueError(f"Paper metrics do not match: {counts}")
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    return dict(
        n_cases=len(rows),
        positive_class="CI",
        alpha="1/201",
        **counts,
        precision=tp / (tp + fp),
        recall=tp / (tp + fn),
        F1=2 * tp / (2 * tp + fp + fn),
    )


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
