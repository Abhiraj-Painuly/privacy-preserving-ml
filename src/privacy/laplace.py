"""Laplace mechanism: pure (epsilon, 0)-differential privacy.

Reference: Dwork & Roth (2014) — paper ref [1]. Used in the paper for
low-dimensional sensitive features (paper Table I).

For a query *f* with global L1 sensitivity Δf, the Laplace mechanism releases

    M(x) = f(x) + Lap(0, Δf / ε)

which satisfies pure (ε, 0)-DP.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LaplaceMechanism:
    """Per-feature Laplace perturbation with explicit sensitivity bookkeeping.

    Parameters
    ----------
    epsilon:
        Privacy budget consumed by **each** call to :meth:`perturb`. Smaller ε
        means stronger privacy and noisier output.
    sensitivity:
        Either a scalar Δf used for every feature, or a per-feature numpy
        array. We compute it from a clipping range when not provided.
    seed:
        Optional RNG seed for reproducibility (used in tests).
    """

    epsilon: float
    sensitivity: float | np.ndarray | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self._rng = np.random.default_rng(self.seed)

    @staticmethod
    def sensitivity_from_range(min_vals: np.ndarray, max_vals: np.ndarray) -> np.ndarray:
        """L1 sensitivity = (max - min) per column when each user contributes
        one record (paper §III-D ¶1)."""
        return np.maximum(max_vals - min_vals, 1e-8)

    def perturb(self, x: np.ndarray, sensitivity: np.ndarray | None = None) -> np.ndarray:
        """Apply Laplace noise to a 2-D array.

        Each column ``j`` receives noise ~ Laplace(0, Δf_j / ε).
        """
        if sensitivity is None:
            sensitivity = self.sensitivity
        if sensitivity is None:
            # Estimate from the data itself (note: this leaks information; only
            # used for benchmarking when caller hasn't pre-clipped).
            sensitivity = self.sensitivity_from_range(x.min(axis=0), x.max(axis=0))
        scale = np.asarray(sensitivity, dtype=float) / float(self.epsilon)
        noise = self._rng.laplace(0.0, scale, size=x.shape)
        return x + noise

    def __repr__(self) -> str:  # pragma: no cover
        return f"LaplaceMechanism(epsilon={self.epsilon})"
