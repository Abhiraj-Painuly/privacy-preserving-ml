"""Tabular WideResNet with GroupNorm (DP-compatible)."""
from __future__ import annotations

import torch
from torch import nn

from ._blocks import WideResidualBlock, _norm


class WideResNetGN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 64,
        widen: int = 4,
        n_blocks: int = 3,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(nn.Linear(input_dim, hidden), _norm(hidden), nn.GELU())
        self.blocks = nn.Sequential(*[WideResidualBlock(hidden, widen=widen) for _ in range(n_blocks)])
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.blocks(self.stem(x)))
