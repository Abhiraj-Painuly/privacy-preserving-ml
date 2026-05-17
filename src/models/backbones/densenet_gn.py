"""Tabular DenseNet with GroupNorm (DP-compatible)."""
from __future__ import annotations

import torch
from torch import nn

from ._blocks import DenseBlock, _norm


class DenseNetGN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden: int = 64,
        growth: int = 32,
        n_layers_per_block: int = 4,
        n_blocks: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden), _norm(hidden), nn.GELU()]
        c = hidden
        for _ in range(n_blocks):
            block = DenseBlock(c, growth=growth, n_layers=n_layers_per_block)
            layers.append(block)
            c = block.out_channels
            # Transition: project back down to ``hidden`` channels.
            layers.append(nn.Linear(c, hidden))
            layers.append(_norm(hidden))
            layers.append(nn.GELU())
            c = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))
