"""Training and prediction utilities for scalar Quantile-CFIT."""

from __future__ import annotations

import copy
import os
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import monotonicity_penalty, pinball_loss
from .models import QuantileModelConfig, build_quantile_model


@dataclass
class TrainConfig:
    epochs: int = 80
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.05
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    lambda_D: float = 1e-3
    lambda_mono: float = 1e-2
    patience: int = 20
    train_u_min: float = 0.02
    train_u_max: float = 0.98
    device: str = "auto"
    # v2 staged training
    v2_q0_pretrain_epochs: int = 60
    v2_d_train_epochs: int = 100
    v2_q0_pretrain_patience: int = 15
    v2_d_train_patience: int = 20
    v2_freeze_q0: bool = True
    v2_d_l2_penalty: float = 1e-4
    v2_mono_penalty: float = 1e-3
    # Reproducibility controls.  When deterministic=True, model initialization,
    # minibatch order, sampled quantile levels, and CUDA backend flags are seeded.
    seed: int = 0
    deterministic: bool = True


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible CFIT training.

    This does not guarantee bitwise identity for every CUDA operation on every
    hardware/software stack, but it removes the major uncontrolled randomness
    sources in the CFIT pipeline: model initialization, torch.rand calls for u,
    and DataLoader shuffling.
    """
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        except Exception:
            pass


def get_device(device: str = "auto") -> torch.device:
    if str(device) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def tensors_from_arrays(
    X: np.ndarray, Y: np.ndarray, Z: np.ndarray, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.as_tensor(X, dtype=torch.float32, device=device),
        torch.as_tensor(Y, dtype=torch.float32, device=device),
        torch.as_tensor(Z, dtype=torch.float32, device=device),
    )


def _loader(
    X, Y, Z, batch_size: int, shuffle: bool, seed: int | None = None
) -> DataLoader:
    generator = None
    if bool(shuffle) and seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        TensorDataset(X, Y, Z),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        generator=generator,
        num_workers=0,
    )


def _sample_u(n: int, cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    lo, hi = float(cfg.train_u_min), float(cfg.train_u_max)
    return lo + (hi - lo) * torch.rand(n, 1, device=device)


@torch.no_grad()
def validation_loss(model, X, Y, Z, cfg: TrainConfig) -> float:
    model.eval()
    u = _sample_u(X.shape[0], cfg, X.device)
    q0 = model.q0(Z, u)
    q1 = model.q1(Y, Z, u)
    loss = pinball_loss(X, q0, u) + pinball_loss(X, q1, u)
    return float(loss.detach().cpu().item())


def _train_joint(model, train_t, val_t, cfg: TrainConfig) -> tuple[object, dict]:
    Xtr, Ytr, Ztr = train_t
    Xv, Yv, Zv = val_t
    opt = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay)
    )
    best_state = copy.deepcopy(model.state_dict())
    best = float("inf")
    bad = 0
    history = []
    for epoch in range(int(cfg.epochs)):
        model.train()
        losses = []
        for xb, yb, zb in _loader(
            Xtr, Ytr, Ztr, cfg.batch_size, True, seed=int(cfg.seed) + 10_000 + epoch
        ):
            u = _sample_u(xb.shape[0], cfg, xb.device)
            q0 = model.q0(zb, u)
            q1 = model.q1(yb, zb, u)
            loss = pinball_loss(xb, q0, u) + pinball_loss(xb, q1, u)
            if hasattr(model, "delta") and float(cfg.lambda_D) > 0:
                loss = loss + float(cfg.lambda_D) * model.delta(yb, zb, u).pow(2).mean()
            if float(cfg.lambda_mono) > 0:
                u2 = torch.clamp(u + 0.02, max=0.98)
                loss = loss + float(cfg.lambda_mono) * (
                    monotonicity_penalty(model.q0(zb, u), model.q0(zb, u2))
                    + monotonicity_penalty(model.q1(yb, zb, u), model.q1(yb, zb, u2))
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        vl = validation_loss(model, Xv, Yv, Zv, cfg)
        history.append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vl}
        )
        if vl < best - 1e-5:
            best = vl
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= int(cfg.patience):
            break
    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best,
        "epochs_run": len(history),
        "history": history,
    }


def _train_v2(model, train_t, val_t, cfg: TrainConfig) -> tuple[object, dict]:
    """Two-stage v2 training: Q0 pretrain followed by D-stage training."""
    Xtr, Ytr, Ztr = train_t
    Xv, Yv, Zv = val_t
    hist = {"q0": [], "d": []}

    # Stage 1: train only Q0.
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.q0_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        model.q0_net.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    best, bad = float("inf"), 0
    best_state = copy.deepcopy(model.q0_net.state_dict())
    for epoch in range(int(cfg.v2_q0_pretrain_epochs)):
        model.train()
        losses = []
        for xb, yb, zb in _loader(
            Xtr, Ytr, Ztr, cfg.batch_size, True, seed=int(cfg.seed) + 20_000 + epoch
        ):
            u = _sample_u(xb.shape[0], cfg, xb.device)
            q0 = model.q0(zb, u)
            loss = pinball_loss(xb, q0, u)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        with torch.no_grad():
            model.eval()
            u = _sample_u(Xv.shape[0], cfg, Xv.device)
            vl = float(pinball_loss(Xv, model.q0(Zv, u), u).cpu().item())
        hist["q0"].append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vl}
        )
        if vl < best - 1e-5:
            best = vl
            best_state = copy.deepcopy(model.q0_net.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= int(cfg.v2_q0_pretrain_patience):
            break
    model.q0_net.load_state_dict(best_state)

    # Stage 2: train deformation head, optionally freeze Q0.
    for p in model.parameters():
        p.requires_grad_(False)
    if not bool(cfg.v2_freeze_q0):
        for p in model.q0_net.parameters():
            p.requires_grad_(True)
    for p in model.d_net.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    best, bad = float("inf"), 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(int(cfg.v2_d_train_epochs)):
        model.train()
        losses = []
        for xb, yb, zb in _loader(
            Xtr, Ytr, Ztr, cfg.batch_size, True, seed=int(cfg.seed) + 30_000 + epoch
        ):
            u = _sample_u(xb.shape[0], cfg, xb.device)
            q1 = model.q1(yb, zb, u)
            loss = pinball_loss(xb, q1, u)
            if float(cfg.v2_d_l2_penalty) > 0:
                loss = (
                    loss
                    + float(cfg.v2_d_l2_penalty) * model.delta(yb, zb, u).pow(2).mean()
                )
            if float(cfg.v2_mono_penalty) > 0:
                u2 = torch.clamp(u + 0.02, max=0.98)
                loss = loss + float(cfg.v2_mono_penalty) * monotonicity_penalty(
                    model.q1(yb, zb, u), model.q1(yb, zb, u2)
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        with torch.no_grad():
            model.eval()
            u = _sample_u(Xv.shape[0], cfg, Xv.device)
            vl = float(pinball_loss(Xv, model.q1(Yv, Zv, u), u).cpu().item())
        hist["d"].append(
            {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vl}
        )
        if vl < best - 1e-5:
            best = vl
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= int(cfg.v2_d_train_patience):
            break
    model.load_state_dict(best_state)
    for p in model.parameters():
        p.requires_grad_(True)
    return model, {
        "best_val_loss": best,
        "epochs_run": len(hist["q0"]) + len(hist["d"]),
        "history": hist,
    }


def train_quantile_model(
    X_train,
    Y_train,
    Z_train,
    X_val,
    Y_val,
    Z_val,
    model_cfg: QuantileModelConfig,
    train_cfg: TrainConfig,
):
    set_global_seed(int(train_cfg.seed), deterministic=bool(train_cfg.deterministic))
    device = get_device(train_cfg.device)
    model = build_quantile_model(model_cfg).to(device)
    train_t = tensors_from_arrays(X_train, Y_train, Z_train, device)
    val_t = tensors_from_arrays(X_val, Y_val, Z_val, device)
    start = time.perf_counter()
    if str(model_cfg.model_type).lower() in {
        "residual_deformation_v2",
        "residual_v2",
        "deformation_v2",
    }:
        model, info = _train_v2(model, train_t, val_t, train_cfg)
    else:
        model, info = _train_joint(model, train_t, val_t, train_cfg)
    info["fit_time_sec"] = time.perf_counter() - start
    return model, info


@torch.no_grad()
def predict_quantiles(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    u_grid: np.ndarray,
    device: str = "auto",
) -> dict[str, np.ndarray]:
    dev = get_device(device)
    model.eval().to(dev)
    Xt, Yt, Zt = tensors_from_arrays(X, Y, Z, dev)
    q0s, q1s, ds = [], [], []
    for u in np.asarray(u_grid, dtype=np.float32).reshape(-1):
        ut = torch.full((X.shape[0], 1), float(u), dtype=torch.float32, device=dev)
        q0 = model.q0(Zt, ut).detach().cpu().numpy()
        q1 = model.q1(Yt, Zt, ut).detach().cpu().numpy()
        d = model.delta(Yt, Zt, ut).detach().cpu().numpy()
        q0s.append(q0)
        q1s.append(q1)
        ds.append(d)
    return {
        "q0": np.stack(q0s, axis=1).squeeze(-1),
        "q1": np.stack(q1s, axis=1).squeeze(-1),
        "delta": np.stack(ds, axis=1).squeeze(-1),
    }


@torch.no_grad()
def predict_q1_for_y(
    model, Y_new: np.ndarray, Z: np.ndarray, u_grid: np.ndarray, device: str = "auto"
) -> np.ndarray:
    dev = get_device(device)
    model.eval().to(dev)
    Yt = torch.as_tensor(Y_new, dtype=torch.float32, device=dev)
    Zt = torch.as_tensor(Z, dtype=torch.float32, device=dev)
    q1s = []
    for u in np.asarray(u_grid, dtype=np.float32).reshape(-1):
        ut = torch.full((Z.shape[0], 1), float(u), dtype=torch.float32, device=dev)
        q1s.append(model.q1(Yt, Zt, ut).detach().cpu().numpy())
    return np.stack(q1s, axis=1).squeeze(-1)
