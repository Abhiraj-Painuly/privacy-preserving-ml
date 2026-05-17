"""Vanilla deterministic autoencoder (paper Table II row 1).

The Vanilla AE is the simplest of the three architectures: a symmetric
encoder/decoder pair trained to minimise reconstruction MSE. Privacy is
enforced *only* through DP-SGD on all parameters — there is no latent-space
noise. As §V-A ¶1 of the paper points out, this design has

    sensitivity ∝ batch size B

because gradient clipping is applied batch-wise, so it collapses for ε < 1.
"""
from __future__ import annotations

import torch
from torch import nn


class VanillaAE(nn.Module):
    """Symmetric MLP autoencoder.

    Parameters
    ----------
    input_dim:
        Number of input columns (after preprocessing).
    latent_dim:
        Latent representation size (defaults to input_dim // 2).
    hidden:
        Width of the two hidden layers.
    """

    def __init__(self, input_dim: int, latent_dim: int | None = None, hidden: int = 128) -> None:
        super().__init__()
        latent_dim = latent_dim or max(8, input_dim // 2)
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.GroupNorm(num_groups=8, num_channels=hidden) if hidden % 8 == 0 else nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GroupNorm(num_groups=8, num_channels=hidden) if hidden % 8 == 0 else nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.forward(x)
        return torch.mean((x_hat - x) ** 2)
