"""Differentially Private Variational Autoencoder (paper Table II row 2).

Implements term-wise gradient aggregation per Takahashi et al. (2020) — paper
ref [11]. Key ideas:

* The ELBO factorises into two terms:
      L_recon (sample-wise)   +   β · L_KL (batch-wise)
  In a vanilla DP-VAE, both terms are pushed through the same per-sample
  gradient clipper, so noise scales as O(B). Term-wise aggregation clips them
  separately, yielding O(1) sensitivity.
* The latent space is probabilistic ``q(z|x) = N(μ(x), σ(x)²)``. Sampling adds
  intrinsic stochasticity that reinforces DP.
* Output reuses the Vanilla AE decoder.

This is the design that makes DP-VAE close to TabDAE on Table III despite
having dramatically less compute than TabDAE.
"""
from __future__ import annotations

import torch
from torch import nn


class DPVAE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int | None = None,
        hidden: int = 128,
        beta: float = 1.0,
    ) -> None:
        super().__init__()
        latent_dim = latent_dim or max(8, input_dim // 2)
        self.latent_dim = latent_dim
        self.beta = beta

        def _norm(channels: int) -> nn.Module:
            if channels % 8 == 0:
                return nn.GroupNorm(8, channels)
            return nn.LayerNorm(channels)

        self.encoder_trunk = nn.Sequential(
            nn.Linear(input_dim, hidden), _norm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), _norm(hidden), nn.GELU(),
        )
        self.fc_mu = nn.Linear(hidden, latent_dim)
        self.fc_logvar = nn.Linear(hidden, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), _norm(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), _norm(hidden), nn.GELU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder_trunk(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return {"x_hat": x_hat, "z": z, "mu": mu, "logvar": logvar}

    def loss_terms(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the reconstruction and KL terms separately.

        The trainer applies *separate* gradient clipping to each (Takahashi
        et al. 2020) so the per-sample sensitivity stays O(1) regardless of
        batch size.
        """
        out = self.forward(x)
        recon = torch.mean((out["x_hat"] - x) ** 2, dim=-1)  # per-sample
        kl = -0.5 * torch.mean(
            1 + out["logvar"] - out["mu"].pow(2) - out["logvar"].exp(),
            dim=-1,
        )  # per-sample as well; aggregator may average vs. sum
        return {"recon": recon, "kl": kl, **out}

    def total_loss(self, x: torch.Tensor) -> torch.Tensor:
        terms = self.loss_terms(x)
        return terms["recon"].mean() + self.beta * terms["kl"].mean()
