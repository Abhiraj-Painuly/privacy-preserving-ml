"""Tabular ResNet with GroupNorm (DP-compatible)."""
from __future__ import annotations

import torch
from torch import nn

from ._blocks import ResidualBlock, _norm


class ResNetGN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 128,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden), _norm(hidden), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
