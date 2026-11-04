"""Run the 90 Sachs conditional-independence queries with staged quantile models."""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ._utils import ROOT, atomic_json, digest

COLUMNS = {
    "Raf": "praf",
    "Mek": "pmek",
    "PLCg": "plcg",
    "PIP2": "PIP2",
    "PIP3": "PIP3",
    "Erk": "p44/42",
    "Akt": "pakts473",
    "PKA": "PKA",
    "PKC": "PKC",
    "P38": "P38",
    "JNK": "pjnk",
}
DTYPE = torch.float64


def load_sachs(path):
    raw = pd.read_csv(path)
    missing = set(COLUMNS.values()) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing Sachs columns: {sorted(missing)}")
    data = raw[list(COLUMNS.values())].rename(
        columns={v: k for k, v in COLUMNS.items()}
    )
    data = (
        data.apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    # The experiment standardizes all 1,755 rows before cross-fitting.
    return pd.DataFrame(
        StandardScaler().fit_transform(data.to_numpy(dtype=float)), columns=data.columns
    )


def load_relations(path):
    relations = pd.read_csv(path).rename(columns={"author_case_id": "case_id"})
    if len(relations) != 90 or relations.case_id.tolist() != list(range(90)):
        raise ValueError("Expected the 90 exact author relations in case-ID order")
    if relations.label_ci.sum() != 50:
        raise ValueError("Expected 50 CI and 40 CD relations")
    return relations


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def u_features(u, cfg):
    u = u.reshape(-1, 1)
    parts = [u]
    for k in range(1, cfg["n_fourier"] + 1):
        parts += [torch.sin(2 * math.pi * k * u), torch.cos(2 * math.pi * k * u)]
    centers = torch.linspace(
        0.05, 0.95, cfg["n_rbf"], device=u.device, dtype=u.dtype
    ).reshape(1, -1)
    parts.append(torch.exp(-0.5 * ((u - centers) / cfg["rbf_width"]) ** 2))
    return torch.cat(parts, dim=1)


class MLP(nn.Module):
    def __init__(self, in_dim, cfg):
        super().__init__()
        layers = []
        for h in cfg["hidden_dims"]:
            layers += [nn.Linear(in_dim, h), nn.SiLU(), nn.Dropout(cfg["dropout"])]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualQuantileModel(nn.Module):
    def __init__(self, z_dim, y_dim, cfg):
        super().__init__()
        self.cfg = cfg
        u_dim = 1 + 2 * cfg["n_fourier"] + cfg["n_rbf"]
        self.q0 = MLP(z_dim + u_dim, cfg)
        self.d = MLP(y_dim + z_dim + u_dim, cfg)
        nn.init.zeros_(self.d.net[-1].weight)
        nn.init.zeros_(self.d.net[-1].bias)

    def baseline(self, z, u):
        return self.q0(torch.cat([z, u_features(u, self.cfg)], dim=1))

    def deformation(self, y, z, u):
        return self.d(torch.cat([y, z, u_features(u, self.cfg)], dim=1))

    def augmented(self, y, z, u):
        return self.baseline(z, u) + self.deformation(y, z, u)


def pinball(pred, x, u):
    error = x - pred
    return torch.maximum(u * error, (u - 1) * error).mean()


def monotonicity(model, y, z, cfg, mode):
    n, dim = z.shape
    u = torch.tensor(
        np.linspace(0.05, 0.95, cfg["quantiles"]), dtype=DTYPE, device=z.device
    )
    ur = u.reshape(1, -1).repeat(n, 1).reshape(-1, 1)
    zr = z[:, None, :].repeat(1, len(u), 1).reshape(-1, dim)
    if mode == "q0":
        q = model.baseline(zr, ur).reshape(n, -1)
    else:
        yr = y[:, None, :].repeat(1, len(u), 1).reshape(-1, y.shape[1])
        q = model.augmented(yr, zr, ur).reshape(n, -1)
    return torch.relu(-(q[:, 1:] - q[:, :-1])).mean()


def train_relation(x, y, z, seed, cfg, device):
    set_seeds(seed)
    order = np.random.default_rng(seed).permutation(len(x))
    n_val = max(1, int(cfg["validation_fraction"] * len(x)))
    val, tr = order[:n_val], order[n_val:]

    def tensor(a, index):
        return torch.tensor(a[index], dtype=DTYPE, device=device)

    xt, yt, zt = tensor(x, tr), tensor(y, tr), tensor(z, tr)
    xv, yv, zv = tensor(x, val), tensor(y, val), tensor(z, val)
    loader = DataLoader(
        TensorDataset(xt, yt, zt),
        batch_size=min(cfg["batch_size"], len(tr)),
        shuffle=True,
    )
    model = ResidualQuantileModel(z.shape[1], y.shape[1], cfg).to(
        device=device, dtype=DTYPE
    )

    def draw_u(n):
        return torch.rand(n, 1, dtype=DTYPE, device=device).clamp(0.001, 0.999)

    for stage, epochs, patience_limit in (
        ("q0", cfg["q0_epochs"], cfg["q0_patience"]),
        ("d", cfg["deformation_epochs"], cfg["deformation_patience"]),
    ):
        if stage == "d":
            for p in model.q0.parameters():
                p.requires_grad_(False)
        params = (
            model.q0.parameters()
            if stage == "q0"
            else (p for p in model.parameters() if p.requires_grad)
        )
        opt = torch.optim.AdamW(
            params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
        )
        best, best_state, patience = float("inf"), None, 0
        for _ in range(epochs):
            model.train()
            for xb, yb, zb in loader:
                ub = draw_u(len(xb))
                pred = (
                    model.baseline(zb, ub)
                    if stage == "q0"
                    else model.augmented(yb, zb, ub)
                )
                loss = pinball(pred, xb, ub)
                if stage == "d" and cfg["deformation_l2_penalty"]:
                    loss = (
                        loss
                        + cfg["deformation_l2_penalty"]
                        * model.deformation(yb, zb, ub).square().mean()
                    )
                if cfg["monotonicity_penalty"]:
                    loss = loss + cfg["monotonicity_penalty"] * monotonicity(
                        model, yb, zb, cfg, stage
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                uv = draw_u(len(xv))
                pred = (
                    model.baseline(zv, uv)
                    if stage == "q0"
                    else model.augmented(yv, zv, uv)
                )
                val_loss = float(pinball(pred, xv, uv).detach().cpu())
            if val_loss < best - 1e-5:
                best = val_loss
                state = model.q0 if stage == "q0" else model
                best_state = {
                    k: v.detach().cpu().clone() for k, v in state.state_dict().items()
                }
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    break
        if best_state is not None:
            (model.q0 if stage == "q0" else model).load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    return model


class RidgeResidualBootstrap:
    """Ridge conditional mean with a standardized-residual bootstrap."""

    def fit(self, z, y):
        self.model = RidgeCV(alphas=np.logspace(-4, 4, 25)).fit(
            np.asarray(z, np.float64), np.asarray(y, np.float64).ravel()
        )
        residuals = y.ravel() - self.model.predict(z)
        mu, sigma = float(np.mean(residuals)), max(
            float(np.std(residuals, ddof=1)), 1e-6
        )
        self.resid_std = ((residuals - mu) / sigma).astype(np.float32)
        self.sigma = sigma
        return self

    def sample(self, z, n_draws, seed):
        rng = np.random.default_rng(seed)
        mean = self.model.predict(z).astype(np.float32)
        eps = rng.choice(
            self.resid_std, size=(n_draws, len(mean)), replace=True
        ).astype(np.float32)
        return (mean[None, :] + self.sigma * eps)[:, :, None].astype(np.float32)


def statistic(model, x, y, z, levels, device, kind="gain"):
    model.eval()
    with torch.no_grad():
        xt, yt, zt = (torch.tensor(v, dtype=DTYPE, device=device) for v in (x, y, z))
        values = []
        for u in levels:
            ut = torch.full((len(x), 1), float(u), dtype=DTYPE, device=device)
            if kind == "gain":
                v = pinball(model.baseline(zt, ut), xt, ut) - pinball(
                    model.augmented(yt, zt, ut), xt, ut
                )
            else:
                v = model.deformation(yt, zt, ut).square().mean()
            values.append(float(v.detach().cpu()))
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_case(data, row, cfg, device):
    start = time.time()
    case_id = int(row.case_id)
    seed = cfg["base_seed"] + case_id
    zs = row.z_names.split(";")
    # Preserve the experiment's float32 input rounding before float64 fitting.
    x, y, z = (
        data[names].to_numpy(dtype=np.float32)
        for names in ([row.x_name], [row.y_name], zs)
    )
    folds = KFold(n_splits=cfg["n_splits"], shuffle=True, random_state=seed)
    n, f = cfg["n_null_draws"], cfg["n_splits"]
    observed = {k: [] for k in ("gain", "gain_lower_tail", "D_energy_lower_tail")}
    null = {k: np.zeros((n, f), dtype=np.float64) for k in observed}
    grids = {
        "gain": np.linspace(0.05, 0.95, cfg["quantiles"], dtype=np.float64),
        "gain_lower_tail": np.array([0.01], dtype=np.float64),
        "D_energy_lower_tail": np.array([0.01], dtype=np.float64),
    }
    for fold, (tr, te) in enumerate(folds.split(z)):
        model = train_relation(x[tr], y[tr], z[tr], seed + 37 * fold, cfg, device)
        for kind, grid in grids.items():
            observed[kind].append(
                statistic(
                    model,
                    x[te],
                    y[te],
                    z[te],
                    grid,
                    device,
                    "energy" if kind.startswith("D_") else "gain",
                )
            )
        sampler = RidgeResidualBootstrap().fit(
            z[tr].astype(np.float64), y[tr].astype(np.float64)
        )
        draws = sampler.sample(z[te].astype(np.float64), n, seed + 6000 + fold)
        for b in range(n):
            for kind, grid in grids.items():
                null[kind][b, fold] = statistic(
                    model,
                    x[te],
                    draws[b],
                    z[te],
                    grid,
                    device,
                    "energy" if kind.startswith("D_") else "gain",
                )
    result = {
        "case_id": case_id,
        "x_name": row.x_name,
        "y_name": row.y_name,
        "z_names": row.z_names,
        "z_dim": len(zs),
        "label_ci": int(row.label_ci),
        "n_splits": f,
        "n_null_draws": n,
        "seed": seed,
        "runtime_seconds": time.time() - start,
        "status": "ok",
        "error": "",
    }
    for kind in observed:
        obs, draws = float(np.mean(observed[kind])), null[kind].mean(axis=1)
        count = int(np.sum(draws >= obs))
        suffix = (
            ""
            if kind == "gain"
            else (
                "_gain_lower_tail"
                if kind == "gain_lower_tail"
                else "_D_energy_lower_tail"
            )
        )
        result.update(
            {
                "neural_stat" + suffix: obs,
                "neural_null_mean" + suffix: float(np.mean(draws)),
                "neural_null_std" + suffix: float(np.std(draws, ddof=1)),
                "neural_p_value" + suffix: (count + 1) / (n + 1),
                "neural_count_ge_obs" + suffix: count,
            }
        )
    result["pred_ci"] = int(
        result["neural_p_value"] > 0.05
    )  # Original saved notebook column.
    result["pred_cd"] = 1 - result["pred_ci"]
    result["pred_ci_paper"] = int(result["neural_p_value"] > cfg["alpha"])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sachs/sachs_1755.csv")
    parser.add_argument(
        "--relations", type=Path, default=ROOT / "data/sachs/relations.csv"
    )
    parser.add_argument("--config", type=Path, default=ROOT / "configs/sachs.json")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/sachs")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=90)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not 0 <= args.start < args.end <= 90:
        parser.error("Expected 0 <= start < end <= 90")
    cfg = json.loads(args.config.read_text())
    torch.set_default_dtype(DTYPE)  # Set before creating any model parameters.
    device = torch.device(args.device)
    data, relations = load_sachs(args.data), load_relations(args.relations)
    if len(data) != 1755:
        raise ValueError("Historical run requires exactly 1755 complete rows")
    identity = {
        "config_sha256": digest(args.config),
        "data_sha256": digest(args.data),
        "relations_sha256": digest(args.relations),
        "runner_sha256": digest(__file__),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "sklearn": __import__("sklearn").__version__,
        "device": str(device),
    }
    for _, row in relations.iloc[args.start : args.end].iterrows():
        target = args.out / f"case_{int(row.case_id):03}.json"
        if target.exists():
            prior = json.loads(target.read_text())
            if prior["identity"] != identity or prior["result"]["status"] != "ok":
                raise ValueError(
                    f"Existing output has incompatible identity/status: {target}"
                )
            print(f"case {row.case_id}: reused", flush=True)
            continue
        result = run_case(data, row, cfg, device)
        atomic_json(target, {"identity": identity, "result": result})
        print(
            f"case {row.case_id}: p={result['neural_p_value']:.6f} time={result['runtime_seconds']:.1f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
