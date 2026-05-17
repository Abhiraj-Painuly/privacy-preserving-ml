"""Shared building blocks for the four MLP backbones.

The paper's ensemble (§III-B) uses ResNet/DenseNet/WideResNet/PreResNet on the
same perturbed tabular feature matrix. We therefore implement *tabular*
analogues of those four architectures: residual / densely-connected MLPs that
honour each architecture's distinctive design choice while remaining
DP-compatible (GroupNorm or LayerNorm, never BatchNorm).
"""
from __future__ import annotations

import torch
from torch import nn


def _norm(channels: int) -> nn.Module:
    """Use GroupNorm if divisible by 8 else LayerNorm — both DP-safe."""
    if channels >= 8 and channels % 8 == 0:
        return nn.GroupNorm(num_groups=8, num_channels=channels)
    return nn.LayerNorm(channels)


class ResidualBlock(nn.Module):
    """Standard residual MLP block (used by ResNetGN)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.norm1 = _norm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.norm2 = _norm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = self.fc2(h)
        h = self.norm2(h)
        return self.act(h + x)


class PreActResidualBlock(nn.Module):
    """Pre-activation residual block (PreResNet style: norm/act *before* conv)."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm1 = _norm(channels)
        self.fc1 = nn.Linear(channels, channels)
        self.norm2 = _norm(channels)
        self.fc2 = nn.Linear(channels, channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.fc1(h)
        h = self.act(self.norm2(h))
        h = self.fc2(h)
        return h + x


class WideResidualBlock(nn.Module):
    """WideResNet block: wider hidden dim and dropout (kept off under DP)."""

    def __init__(self, channels: int, widen: int = 4) -> None:
        super().__init__()
        wide = channels * widen
        self.fc1 = nn.Linear(channels, wide)
        self.norm1 = _norm(wide)
        self.fc2 = nn.Linear(wide, channels)
        self.norm2 = _norm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.fc1(x)))
        h = self.act(self.norm2(self.fc2(h)))
        return h + x


class DenseLayer(nn.Module):
    """One dense-layer of a DenseNet block (input is concatenated each step)."""

    def __init__(self, in_channels: int, growth: int) -> None:
        super().__init__()
        self.fc = nn.Linear(in_channels, growth)
        self.norm = _norm(growth)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.act(self.norm(self.fc(x)))
        return torch.cat([x, out], dim=-1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c = in_channels
        for _ in range(n_layers):
            layers.append(DenseLayer(c, growth))
            c += growth
        self.layers = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
