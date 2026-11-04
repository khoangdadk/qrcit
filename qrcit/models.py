"""Conditional quantile networks with independent or residual heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn


@dataclass
class QuantileModelConfig:
    d_z: int
    d_y: int = 1
    model_type: str = "residual"
    hidden_dims: tuple[int, ...] = (128, 128, 64)
    dropout: float = 0.05
    u_feature_mode: str = "qnorm"  # qnorm | fourier | rbf | fourier_rbf
    n_u_fourier: int = 4
    n_u_rbf: int = 12
    u_rbf_width: float = 0.10
    zero_init_deformation: bool = True


def torch_qnorm(u: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(u.dtype).eps
    u = torch.clamp(u, eps, 1.0 - eps)
    return math.sqrt(2.0) * torch.erfinv(2.0 * u - 1.0)


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        hidden_dims: Sequence[int] = (128, 128, 64),
        dropout: float = 0.05,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = int(in_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(prev, int(h)))
            layers.append(nn.SiLU())
            if dropout and dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev = int(h)
        layers.append(nn.Linear(prev, int(out_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _zero_init_last_linear(module: nn.Module) -> None:
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is not None:
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)


def u_features(u: torch.Tensor, cfg: QuantileModelConfig) -> torch.Tensor:
    """Encode u:(n,1) with a normal quantile and optional Fourier/RBF features."""
    mode = str(cfg.u_feature_mode).lower()
    feats = []
    if mode in {"qnorm", "fourier", "rbf", "fourier_rbf"}:
        feats.append(torch_qnorm(u))
    else:
        raise ValueError(f"Unknown u_feature_mode={cfg.u_feature_mode!r}")

    if mode in {"fourier", "fourier_rbf"}:
        for k in range(1, int(cfg.n_u_fourier) + 1):
            kk = float(k)
            feats.append(torch.sin(2.0 * math.pi * kk * u))
            feats.append(torch.cos(2.0 * math.pi * kk * u))

    if mode in {"rbf", "fourier_rbf"}:
        centers = torch.linspace(
            0.05, 0.95, int(cfg.n_u_rbf), device=u.device, dtype=u.dtype
        ).view(1, -1)
        width = max(float(cfg.u_rbf_width), 1e-6)
        feats.append(torch.exp(-0.5 * ((u - centers) / width) ** 2))

    return torch.cat(feats, dim=1)


def u_feature_dim(cfg: QuantileModelConfig) -> int:
    mode = str(cfg.u_feature_mode).lower()
    dim = 1
    if mode in {"fourier", "fourier_rbf"}:
        dim += 2 * int(cfg.n_u_fourier)
    if mode in {"rbf", "fourier_rbf"}:
        dim += int(cfg.n_u_rbf)
    return dim


class IndependentQuantileModel(nn.Module):
    def __init__(self, cfg: QuantileModelConfig):
        super().__init__()
        self.cfg = cfg
        uf = u_feature_dim(cfg)
        self.q0_net = MLP(cfg.d_z + uf, 1, cfg.hidden_dims, cfg.dropout)
        self.q1_net = MLP(int(cfg.d_y) + cfg.d_z + uf, 1, cfg.hidden_dims, cfg.dropout)

    def q0(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.q0_net(torch.cat([z, u_features(u, self.cfg)], dim=1))

    def q1(self, y: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.q1_net(torch.cat([y, z, u_features(u, self.cfg)], dim=1))

    def delta(self, y: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.q1(y, z, u) - self.q0(z, u)


class ResidualQuantileModel(nn.Module):
    def __init__(self, cfg: QuantileModelConfig):
        super().__init__()
        self.cfg = cfg
        uf = u_feature_dim(cfg)
        self.q0_net = MLP(cfg.d_z + uf, 1, cfg.hidden_dims, cfg.dropout)
        self.d_net = MLP(int(cfg.d_y) + cfg.d_z + uf, 1, cfg.hidden_dims, cfg.dropout)

    def q0(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.q0_net(torch.cat([z, u_features(u, self.cfg)], dim=1))

    def deformation(
        self, y: torch.Tensor, z: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        return self.d_net(torch.cat([y, z, u_features(u, self.cfg)], dim=1))

    def q1(self, y: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.q0(z, u) + self.deformation(y, z, u)

    def delta(self, y: torch.Tensor, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.deformation(y, z, u)


class StagedResidualQuantileModel(ResidualQuantileModel):
    def __init__(self, cfg: QuantileModelConfig):
        super().__init__(cfg)
        if cfg.zero_init_deformation:
            _zero_init_last_linear(self.d_net)


def build_quantile_model(cfg: QuantileModelConfig) -> nn.Module:
    mt = str(cfg.model_type).lower()
    if mt == "independent":
        return IndependentQuantileModel(cfg)
    if mt == "residual":
        return ResidualQuantileModel(cfg)
    if mt in {"residual_deformation_v2", "residual_v2", "deformation_v2"}:
        # Feature selection is controlled by the experiment configuration.
        return StagedResidualQuantileModel(cfg)
    raise ValueError(f"Unknown CFIT model_type={cfg.model_type!r}")
