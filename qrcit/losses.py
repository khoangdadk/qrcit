"""Losses and regularizers for scalar Quantile-CFIT."""

from __future__ import annotations

import torch


def pinball_loss(x: torch.Tensor, q: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    r = x - q
    return torch.maximum(u * r, (u - 1.0) * r).mean()


def pinball_loss_no_reduce(
    x: torch.Tensor, q: torch.Tensor, u: torch.Tensor
) -> torch.Tensor:
    r = x - q
    return torch.maximum(u * r, (u - 1.0) * r)


def monotonicity_penalty(q_low: torch.Tensor, q_high: torch.Tensor) -> torch.Tensor:
    return torch.relu(q_low - q_high).pow(2).mean()
