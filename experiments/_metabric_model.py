"""Two-stage quantile regression for continuous METABRIC expression features."""

import math
import time

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLINICAL = [
    "HER2_status",
    "hormone_therapy",
    "menopausal_status",
    "survival_status",
    "radiation_therapy",
]
CDCIT_GENES = ["DNAH11", "FOXP1", "FOXO3", "MLLT4", "NR3C1", "PPP2R2A", "SF3B1", "TP53"]


def load_data(path, gene_path):
    # Reconstruction converted clinical categories to str before CSV export.
    # Preserve the literal "nan" category and its one-hot column order.
    df = pd.read_csv(path, keep_default_na=False, float_precision="round_trip")
    genes = [g for g in gene_path.read_text().splitlines() if g]
    if len(df) != 1491 or len(genes) != 169 or len(set(genes)) != 169:
        raise ValueError("Expected the reconstructed 1491-sample, 169-gene cohort")
    if not set(["tumor_size"] + CLINICAL + genes).issubset(df.columns):
        raise ValueError("Missing reconstructed METABRIC columns")
    return df, genes


def make_task(df, genes, gene):
    if gene not in genes:
        raise ValueError(f"Unknown gene: {gene}")
    x = pd.to_numeric(df.tumor_size, errors="coerce").to_numpy().reshape(-1, 1)
    y = pd.to_numeric(df[gene], errors="coerce").to_numpy().reshape(-1, 1)
    zg = df[[g for g in genes if g != gene]].apply(pd.to_numeric, errors="coerce")
    zc = pd.get_dummies(
        df[CLINICAL].fillna("__MISSING__").astype(str), drop_first=False
    )
    valid = np.isfinite(x.ravel()) & np.isfinite(y.ravel())
    x, y, zg, zc = x[valid], y[valid], zg.loc[valid], zc.loc[valid]
    values = np.unique(y)
    if len(values) <= 2 or (
        len(values) <= 10 and np.all(np.abs(values - np.round(values)) < 1e-8)
    ):
        raise ValueError(
            "The recovered METABRIC recipe expects continuous gene expression"
        )
    zg = SimpleImputer(strategy="median").fit_transform(zg)
    z = np.hstack([zg, zc.to_numpy(dtype=np.float32)]).astype(np.float32)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    z[:, : zg.shape[1]] = StandardScaler().fit_transform(z[:, : zg.shape[1]])
    y = StandardScaler().fit_transform(y.astype(float)).astype(np.float32)
    finite = (
        np.isfinite(x).all(axis=1)
        & np.isfinite(y).all(axis=1)
        & np.isfinite(z).all(axis=1)
    )
    return x[finite], y[finite], z[finite]


def set_seed(seed):
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def qnorm(u):
    u = torch.clamp(u, 1e-4, 1 - 1e-4)
    return math.sqrt(2) * torch.erfinv(2 * u - 1)


def u_features(u, cfg):
    u = torch.clamp(u, 1e-4, 1 - 1e-4)
    features = [qnorm(u), u, u**2]
    for m in range(1, cfg["n_fourier"] + 1):
        features += [torch.sin(2 * math.pi * m * u), torch.cos(2 * math.pi * m * u)]
    centers = torch.linspace(
        0.05, 0.95, cfg["n_rbf"], device=u.device, dtype=u.dtype
    ).view(1, -1)
    features.append(torch.exp(-0.5 * ((u - centers) / cfg["rbf_width"]) ** 2))
    return torch.cat(features, dim=1)


class MLP(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        layers = []
        for h in cfg["hidden_dims"]:
            layers += [nn.Linear(dim, h), nn.SiLU()]
            if cfg["dropout"] > 0:
                layers.append(nn.Dropout(cfg["dropout"]))
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Model(nn.Module):
    def __init__(self, z_dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.q0 = MLP(z_dim + 1, cfg)
        self.D = MLP(1 + z_dim + 3 + 2 * cfg["n_fourier"] + cfg["n_rbf"], cfg)
        nn.init.zeros_(self.D.net[-1].weight)
        nn.init.zeros_(self.D.net[-1].bias)

    def forward_q0(self, z, u):
        return self.q0(torch.cat([z, qnorm(u)], dim=1))

    def forward_D(self, y, z, u):
        return self.D(torch.cat([y, z, u_features(u, self.cfg)], dim=1))

    def forward_q1(self, y, z, u):
        return self.forward_q0(z, u) + self.forward_D(y, z, u)


def pinball(x, q, u):
    r = x - q
    return torch.maximum(u * r, (u - 1) * r).mean()


def train_stage(model, train, val, seed, cfg, stage):
    set_seed(seed)
    if stage == "d":
        for p in model.q0.parameters():
            p.requires_grad_(False)
    params = (
        model.q0.parameters()
        if stage == "q0"
        else [p for p in model.parameters() if p.requires_grad]
    )
    opt = torch.optim.AdamW(
        params, lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )
    ds = (
        TensorDataset(train["X"], train["Z"])
        if stage == "q0"
        else TensorDataset(train["X"], train["Y"], train["Z"])
    )
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    best, state, patience = np.inf, None, 0
    for _ in range(cfg["q0_epochs" if stage == "q0" else "deformation_epochs"]):
        model.train()
        for batch in loader:
            x, z = batch if stage == "q0" else (batch[0], batch[2])
            y = None if stage == "q0" else batch[1]
            u = torch.rand((len(x), 1), device=x.device).clamp(1e-4, 1 - 1e-4)
            q = model.forward_q0(z, u) if stage == "q0" else model.forward_q1(y, z, u)
            loss = pinball(x, q, u)
            if stage == "d":
                if cfg["deformation_l2_penalty"] > 0:
                    loss = (
                        loss
                        + cfg["deformation_l2_penalty"]
                        * model.forward_D(y, z, u).square().mean()
                    )
                if cfg["monotonicity_penalty"] > 0:
                    u2 = torch.clamp(u + 0.02, 1e-4, 1 - 1e-4)
                    low = model.forward_q1(y, z, torch.minimum(u, u2))
                    high = model.forward_q1(y, z, torch.maximum(u, u2))
                    loss = (
                        loss
                        + cfg["monotonicity_penalty"] * torch.relu(low - high).mean()
                    )
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            u = torch.rand((len(val["X"]), 1), device=val["X"].device).clamp(
                1e-4, 1 - 1e-4
            )
            q = (
                model.forward_q0(val["Z"], u)
                if stage == "q0"
                else model.forward_q1(val["Y"], val["Z"], u)
            )
            score = pinball(val["X"], q, u).item()
        if score < best - 1e-5:
            best, patience = score, 0
            target = model.q0 if stage == "q0" else model
            state = {
                k: v.detach().cpu().clone() for k, v in target.state_dict().items()
            }
        else:
            patience += 1
            if (
                patience
                >= cfg["q0_patience" if stage == "q0" else "deformation_patience"]
            ):
                break
    if state is not None:
        (model.q0 if stage == "q0" else model).load_state_dict(state)
    return float(best)


class RidgeBootstrap:
    def fit(self, z, y):
        self.model = make_pipeline(
            StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13))
        )
        self.model.fit(z, y.reshape(-1))
        self.residuals = (y.reshape(-1) - self.model.predict(z)).astype(float)
        return self

    def sample(self, z, draws, seed):
        mean = self.model.predict(z).reshape(-1)
        eps = np.random.default_rng(seed).choice(
            self.residuals, size=(draws, len(mean)), replace=True
        )
        return (mean[None, :] + eps)[..., None].astype(np.float32)


def gain_matrix(model, tens, levels, y_override=None):
    model.eval()
    x, z = tens["X"], tens["Z"]
    y = (
        tens["Y"]
        if y_override is None
        else torch.tensor(y_override, dtype=torch.float32, device=x.device)
    )
    gains = []
    with torch.no_grad():
        for level in levels:
            u = torch.full((len(x), 1), float(level), device=x.device)
            q0, q1 = model.forward_q0(z, u), model.forward_q1(y, z, u)
            loss0 = torch.maximum(u * (x - q0), (u - 1) * (x - q0))
            loss1 = torch.maximum(u * (x - q1), (u - 1) * (x - q1))
            gains.append((loss0 - loss1).cpu().numpy().reshape(-1))
    return np.vstack(gains).T


def run_gene(df, genes, gene, cfg, device):
    start = time.time()
    x, y, z = make_task(df, genes, gene)
    seed, draws = cfg["seed"], cfg["n_null_draws"]
    levels = np.linspace(0.05, 0.95, cfg["quantiles"])
    obs, null, logs, curves = [], [], [], []
    for fold, (train_idx, test_idx) in enumerate(
        KFold(cfg["n_splits"], shuffle=True, random_state=seed).split(z)
    ):
        tr = np.array(train_idx)
        np.random.default_rng(seed + fold).shuffle(tr)
        n_val = max(1, int(round(cfg["validation_fraction"] * len(tr))))

        def tensors(index):
            return {
                k: torch.tensor(a[index], dtype=torch.float32, device=device)
                for k, a in [("X", x), ("Y", y), ("Z", z)]
            }

        fit, val, test = tensors(tr[n_val:]), tensors(tr[:n_val]), tensors(test_idx)
        set_seed(seed + fold)
        model = Model(z.shape[1], cfg).to(device)
        best0 = train_stage(model, fit, val, seed + fold, cfg, "q0")
        bestd = train_stage(model, fit, val, seed + 100 + fold, cfg, "d")
        logs.append(dict(fold_id=fold, q0_best_val=best0, D_best_val=bestd))
        obs.append(float(np.mean(gain_matrix(model, test, levels))))
        sampler = RidgeBootstrap().fit(z[train_idx], y[train_idx])
        samples = sampler.sample(z[test_idx], draws, seed + 2000 + fold)
        null.append(
            [
                float(np.mean(gain_matrix(model, test, levels, samples[b])))
                for b in range(draws)
            ]
        )
        low, high = float(np.quantile(y[train_idx], 0.25)), float(
            np.quantile(y[train_idx], 0.75)
        )
        model.eval()
        with torch.no_grad():
            y0 = torch.full((len(test_idx), 1), low, device=device)
            y1 = torch.full((len(test_idx), 1), high, device=device)
            for level in levels:
                u = torch.full((len(test_idx), 1), float(level), device=device)
                delta = float(
                    (
                        model.forward_q1(y1, test["Z"], u)
                        - model.forward_q1(y0, test["Z"], u)
                    )
                    .mean()
                    .cpu()
                )
                curves.append(
                    dict(
                        gene=gene,
                        fold_id=fold,
                        u=float(level),
                        delta=delta,
                        contrast_type="q75_minus_q25",
                        y_low=low,
                        y_high=high,
                    )
                )
    observed = float(np.mean(obs))
    null = np.asarray(null, dtype=float).mean(axis=0)
    mean_curve = pd.DataFrame(curves).groupby("u").delta.mean()
    result = dict(
        gene=gene,
        status="ok",
        neural_cfit_p_value=float((1 + np.sum(null >= observed)) / (draws + 1)),
        neural_T_obs=observed,
        neural_T_null_mean=float(null.mean()),
        neural_T_null_std=float(null.std()),
        y_type="continuous",
        n=len(x),
        dim_z=z.shape[1],
        runtime_sec=time.time() - start,
        train_logs=logs,
        null_stats=null.tolist(),
        seed=seed,
        n_splits=cfg["n_splits"],
        n_null_draws=draws,
    )
    for q in [0.1, 0.5, 0.9]:
        result[f"delta_u{int(q * 100)}"] = float(
            mean_curve.iloc[np.argmin(np.abs(mean_curve.index.values - q))]
        )
    result["curve_area"] = float(np.trapz(mean_curve.values, mean_curve.index.values))
    result["curve_abs_area"] = float(
        np.trapz(np.abs(mean_curve.values), mean_curve.index.values)
    )
    result["curve_sign_fraction_positive"] = float(np.mean(mean_curve.values > 0))
    return result, curves
