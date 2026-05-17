"""Gaussian mechanism: approximate (epsilon, delta)-differential privacy.

Reference: Dwork & Roth (2014); Dong & Roth (2019, 2022) — paper refs [1],
[5], [6]. The Gaussian mechanism is preferred in the paper for
high-dimensional features (e.g. CIFAR-10 96-bin histograms) because its
light-tailed noise is friendlier to deep-learning optimisation (Table I).

For an L2-sensitivity Δf and δ ∈ (0, 1), noise scale

    σ = Δf · sqrt(2 · ln(1.25 / δ)) / ε

guarantees (ε, δ)-DP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class GaussianMechanism:
    epsilon: float
    delta: float = 1e-5
    sensitivity: float | np.ndarray | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if not (0 < self.delta < 1):
            raise ValueError("delta must lie in (0, 1)")
        self._rng = np.random.default_rng(self.seed)

    @staticmethod
    def sensitivity_l2(min_vals: np.ndarray, max_vals: np.ndarray) -> float:
        """L2 sensitivity bound for a record-level neighbouring relation."""
        return float(np.linalg.norm(np.maximum(max_vals - min_vals, 1e-8)))

    def sigma(self, sensitivity: float) -> float:
        return sensitivity * math.sqrt(2.0 * math.log(1.25 / self.delta)) / self.epsilon

    def perturb(self, x: np.ndarray, sensitivity: float | None = None) -> np.ndarray:
        if sensitivity is None:
            if self.sensitivity is None:
                sensitivity = self.sensitivity_l2(x.min(axis=0), x.max(axis=0))
            else:
                sensitivity = float(self.sensitivity)
        s = self.sigma(float(sensitivity))
        noise = self._rng.normal(0.0, s, size=x.shape)
        return x + noise

    def __repr__(self) -> str:  # pragma: no cover
        return f"GaussianMechanism(epsilon={self.epsilon}, delta={self.delta})"
