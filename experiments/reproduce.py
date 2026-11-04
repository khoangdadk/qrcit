"""Recompute synthetic tables and figures from bundled trial results."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ["benchmark", "sweep", "x_value", "K", "beta", "variant"]


def summarize(trials, alpha=0.05, tau=0.2):
    rows = []
    if trials.p_global.isna().any() or not trials.p_global.between(0, 1).all():
        raise ValueError("Invalid trial p-values")
    for key, merged in trials.groupby(
        ["benchmark", "sweep", "x_value", "K", "beta"], sort=False
    ):
        if len(merged) != 100:
            raise ValueError(
                f"Expected 100 trials per setting/beta: {key}: {len(merged)}"
            )
        ids = ["trial_id", "variant"] if key[0] == "SDB" else ["trial_id"]
        if merged.duplicated(ids).any():
            raise ValueError(f"Repeated trial identity in {key}")
        groups = [("merged", merged)] + (
            list(merged.groupby("variant")) if key[0] == "SDB" else []
        )
        for variant, sub in groups:
            row = dict(zip(KEYS[:-1], key), variant=variant)
            global_rej = sub.p_global <= alpha
            row.update(
                n_trials=len(sub),
                n=int(sub.n.iloc[0]),
                d_z=int(sub.d_z.iloc[0]),
                metric="Type-I" if key[-1] == 0 else "Power",
                global_rate=float(global_rej.mean()),
                mean_p_global=float(sub.p_global.mean()),
            )
            if key[0] == "SDB":
                for region in ["pg", "de"]:
                    p = sub["p_region_" + region]
                    if p.isna().any():
                        raise ValueError("Missing regional p-values")
                    row["gated_" + region + "_rate"] = float(
                        (
                            global_rej
                            | (
                                (sub.p_global > alpha)
                                & (sub.p_global <= tau)
                                & (p < alpha)
                            )
                        ).mean()
                    )
                    row["mean_p_" + region] = float(p.mean())
                    row["region_" + region + "_rate"] = float((p < alpha).mean())
            rows.append(row)
    return pd.DataFrame(rows)


def figure_data(paper, rates, pattern=False):
    result = paper.copy()
    result["evidence"] = "archived_baseline_summary"
    for idx, p in result.iterrows():
        if not p.method.startswith("QRCIT"):
            continue
        variant = p.variant if pattern else "merged"
        r = rates[
            (rates.benchmark == p.benchmark)
            & (rates.sweep == p.sweep)
            & (rates.x_value == p.x_value)
            & (rates.K == 9)
            & (rates.metric == p.metric)
            & (rates.variant == variant)
        ]
        if len(r) != 1:
            raise ValueError("Ambiguous or missing raw setting")
        col = (
            "global_rate"
            if p.method == "QRCIT"
            else "gated_" + p.method[-2:].lower() + "_rate"
        )
        result.loc[idx, "value"] = r.iloc[0][col]
        result.loc[idx, "evidence"] = "recovered_raw"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--view", choices=["verified", "archived"], default="verified")
    ap.add_argument("--out", type=Path, default=Path("artifacts/verified"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rates = summarize(pd.read_csv(ROOT / "results/synthetic/trials.csv"))
    expected = pd.read_csv(ROOT / "results/synthetic/expected_scalar_rates.csv")
    actual = rates.set_index(KEYS).sort_index()
    expected = expected.set_index(KEYS).sort_index()
    assert actual.index.equals(expected.index)
    cols = [
        "global_rate",
        "gated_pg_rate",
        "gated_de_rate",
        "mean_p_global",
        "mean_p_pg",
        "mean_p_de",
    ]
    assert np.allclose(actual[cols], expected[cols], atol=1e-9, rtol=0, equal_nan=True)
    rates.to_csv(args.out / "scalar_rates.csv", index=False)
    merged = rates[
        (rates.benchmark == "SDB") & (rates.variant == "merged") & (rates.sweep == "N")
    ]
    power = merged[merged.beta == 0.5].pivot(
        index="x_value", columns="K", values="global_rate"
    )
    power.index.name = "N"
    power.to_csv(args.out / "table2_resolution.csv")
    table1 = []
    for n in [250, 500, 1000]:
        g = merged[(merged.x_value == n) & (merged.K == 9)].set_index("beta")
        table1.append(
            dict(
                N=n,
                null_mean_p_global=g.loc[0, "mean_p_global"],
                alt_mean_p_global=g.loc[0.5, "mean_p_global"],
                alt_mean_p_R_PG=g.loc[0.5, "mean_p_pg"],
                alt_mean_p_R_DE=g.loc[0.5, "mean_p_de"],
            )
        )
    t1 = pd.DataFrame(table1)
    assert np.allclose(
        t1, pd.read_csv(ROOT / "results/reference/table1.csv"), atol=1e-6, rtol=0
    )
    t1.to_csv(args.out / "table1_pvalues.csv", index=False)
    comparison = []
    tables = []
    for fig in [3, 4]:
        paper = pd.read_csv(ROOT / f"results/reference/figure{fig}.csv")
        verified = figure_data(paper, rates, pattern=fig == 4)
        q = paper.method.str.startswith("QRCIT")
        for idx in paper.index[q]:
            p, v = paper.loc[idx], verified.loc[idx]
            comparison.append(
                dict(
                    figure=fig,
                    benchmark=p.benchmark,
                    sweep=p.sweep,
                    x_value=p.x_value,
                    variant=p.get("variant", "merged"),
                    method=p.method,
                    metric=p.metric,
                    archived_value=p.value,
                    verified_value=v.value,
                    matches=np.isclose(p.value, v.value),
                )
            )
        table = paper if args.view == "archived" else verified
        table.to_csv(args.out / f"figure{fig}.csv", index=False)
        tables.append(table)
    comparison = pd.DataFrame(comparison)
    comparison.to_csv(args.out / "archived_vs_verified.csv", index=False)
    global_rows = comparison[comparison.method == "QRCIT"]
    assert global_rows.matches.all()
    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from .plots.main import plot_8panel
        from .plots.patterns import plot_power_only
        from .plots.runtime import plot_runtime

        plot_8panel(tables[0], args.out)
        plot_power_only(tables[1], args.out)
        plot_runtime(pd.read_csv(ROOT / "results/reference/runtime.csv"), args.out)
    print(
        f"Verified {sum(global_rows.figure == 3)}/32 Figure 3 global values and "
        f"{sum(global_rows.figure == 4)}/40 Figure 4 global values; Tables I–II recomputed."
    )
    print(
        "Regional curves differ from the archived summary proxy; see archived_vs_verified.csv."
    )


if __name__ == "__main__":
    main()
