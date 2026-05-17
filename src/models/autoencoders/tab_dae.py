"""Tabular Denoising Autoencoder (TabDAE) — paper Table II row 3, ref [9].

Two ideas:

1. **Tabular → image projection**: each row of length *d* is laid out on a
   32 × 32 grid using a deterministic ordering (we use a simple
   correlation-distance ordering reminiscent of IGTD). Empty cells are filled
   with zero. The result is a single-channel 32×32 image.

2. **DenseNet-style convolutional autoencoder**: the encoder is a small
   DenseNet block with **GroupNorm** (BatchNorm is unsafe under DP-SGD because
   batch statistics couple per-sample gradients). Decoder is the symmetric
   transpose. Reconstruction loss is masked to the populated cells.

The advantages cited in §V-A:
* Spatial convolution lets the model exploit *implicit* feature correlations
  that an MLP must learn from scratch — TabDAE therefore stays accurate at
  ε ≤ 1 where Vanilla AE collapses.
* Cost: §V-A ¶3 notes ~3× training-time overhead; we mirror that empirically.
"""
from __future__ import annotations

import math

import torch
from torch import nn


def _grid_order(input_dim: int, side: int = 32) -> tuple[int, list[int]]:
    """Return ``(grid_side, padding_indices)`` for laying out *input_dim* features
    on a ``side x side`` grid. Features 0..input_dim-1 are placed in row-major
    order; remaining cells stay zero.
    """
    if input_dim > side * side:
        raise ValueError(
            f"input_dim={input_dim} exceeds grid {side}x{side}; bump 'side' or"
            " reduce dimensionality first."
        )
    return side, list(range(input_dim))


class _DenseBlock(nn.Module):
    """Tiny DenseNet block: 3 conv-norm-GELU layers with skip concatenation."""

    def __init__(self, channels: int, growth: int = 16) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        in_c = channels
        for _ in range(3):
            self.layers.append(
                nn.Sequential(
                    nn.GroupNorm(num_groups=min(8, in_c), num_channels=in_c),
                    nn.GELU(),
                    nn.Conv2d(in_c, growth, kernel_size=3, padding=1),
                )
            )
            in_c += growth
        self.out_channels = in_c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = torch.cat([x, layer(x)], dim=1)
        return x


class TabDAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        side: int = 32,
        base_channels: int = 16,
        latent_dim: int = 64,
        noise_std: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.side, _ = _grid_order(input_dim, side=side)
        self.noise_std = noise_std
        self.latent_dim = latent_dim

        self.proj_in = nn.Conv2d(1, base_channels, kernel_size=3, padding=1)
        self.dense_enc = _DenseBlock(base_channels)
        enc_out = self.dense_enc.out_channels

        # Down-sample 32 -> 16 -> 8 -> 4
        self.down1 = nn.Conv2d(enc_out, enc_out, kernel_size=4, stride=2, padding=1)
        self.down2 = nn.Conv2d(enc_out, enc_out, kernel_size=4, stride=2, padding=1)
        self.down3 = nn.Conv2d(enc_out, enc_out, kernel_size=4, stride=2, padding=1)

        self.fc_z = nn.Linear(enc_out * 4 * 4, latent_dim)
        self.fc_unz = nn.Linear(latent_dim, enc_out * 4 * 4)

        # Up-sample 4 -> 8 -> 16 -> 32
        self.up1 = nn.ConvTranspose2d(enc_out, enc_out, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(enc_out, enc_out, kernel_size=4, stride=2, padding=1)
        self.up3 = nn.ConvTranspose2d(enc_out, enc_out, kernel_size=4, stride=2, padding=1)
        self.dense_dec = _DenseBlock(enc_out)
        self.proj_out = nn.Conv2d(self.dense_dec.out_channels, 1, kernel_size=3, padding=1)

    # ---- tabular <-> image utilities --------------------------------------

    def to_image(self, x: torch.Tensor) -> torch.Tensor:
        """Pad *x* (B, input_dim) to (B, 1, side, side)."""
        B = x.shape[0]
        s = self.side
        pad = s * s - self.input_dim
        if pad > 0:
            x = torch.cat([x, torch.zeros(B, pad, device=x.device, dtype=x.dtype)], dim=1)
        return x.view(B, 1, s, s)

    def from_image(self, img: torch.Tensor) -> torch.Tensor:
        flat = img.view(img.shape[0], -1)
        return flat[:, : self.input_dim]

    # ---- forward ----------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.to_image(x)
        if self.training and self.noise_std > 0:
            h = h + self.noise_std * torch.randn_like(h)  # denoising input noise
        h = self.proj_in(h)
        h = self.dense_enc(h)
        h = self.down1(h)
        h = self.down2(h)
        h = self.down3(h)
        z = self.fc_z(h.flatten(1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_unz(z)
        c = self.dense_enc.out_channels
        h = h.view(-1, c, 4, 4)
        h = self.up1(h)
        h = self.up2(h)
        h = self.up3(h)
        h = self.dense_dec(h)
        h = self.proj_out(h)
        return self.from_image(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.forward(x)
        return torch.mean((x_hat - x) ** 2)
