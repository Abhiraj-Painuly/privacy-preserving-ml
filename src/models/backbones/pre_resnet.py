"""Tabular Pre-Activation ResNet (PreResNet) with GroupNorm."""
from __future__ import annotations

import torch
from torch import nn

from ._blocks import PreActResidualBlock, _norm


class PreResNetGN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 128,
        n_blocks: int = 4,
    ) -> None:
        super().__init__()
        self.stem = nn.Linear(input_dim, hidden)
        self.blocks = nn.Sequential(*[PreActResidualBlock(hidden) for _ in range(n_blocks)])
        self.norm = _norm(hidden)
        self.act = nn.GELU()
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.blocks(h)
        h = self.act(self.norm(h))
        return self.head(h)
