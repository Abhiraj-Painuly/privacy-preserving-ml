"""Confident GNMax aggregation (PATE — Papernot et al. 2018, paper ref [15]).

For an ensemble of *T* teachers and a query *x* with class votes
``v_j = #{teachers predicting class j}`` (j = 1..K):

1. **Confidence gate** — release the answer only if
       max_j v_j + N(0, σ_th²) ≥ τ.
   Otherwise abstain (this saves privacy budget on uncertain queries).
2. **Aggregation** — output
       arg max_j v_j + N(0, σ_agg²).

Both noise additions are **Gaussian** with per-query L2 sensitivity 1, so
each released answer consumes a small ε via the Gaussian Differential
Privacy composition (Dong et al. 2019, paper refs [5], [6]).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ConfidentGNMaxResult:
    labels: np.ndarray            # (N,), -1 where the gate abstained
    answered_mask: np.ndarray     # (N,), bool
    n_answered: int
    n_total: int


def confident_gnmax(
    teacher_predictions: np.ndarray,
    *,
    n_classes: int,
    sigma_threshold: float = 5.0,
    sigma_aggregator: float = 1.0,
    tau: float | None = None,
    seed: int = 42,
) -> ConfidentGNMaxResult:
    """Aggregate a (T, N) matrix of teacher hard predictions via Confident GNMax.

    Parameters
    ----------
    teacher_predictions:
        Array of shape ``(T, N)`` of integer labels in ``[0, n_classes)``.
    sigma_threshold:
        Std-dev of the noise added during the confidence gate. Larger ⇒ more
        abstentions but smaller ε per query.
    sigma_aggregator:
        Std-dev of the noise added during final aggregation.
    tau:
        Abstention threshold. Defaults to ``T / 2`` (majority must exist).
    seed:
        RNG seed.
    """
    if teacher_predictions.ndim != 2:
        raise ValueError("teacher_predictions must have shape (T, N)")
    T, N = teacher_predictions.shape
    if tau is None:
        tau = T / 2.0

    # Vote tally: counts (N, K)
    counts = np.zeros((N, n_classes), dtype=np.int32)
    for t in range(T):
        for j in range(n_classes):
            counts[:, j] += (teacher_predictions[t] == j).astype(np.int32)

    rng = np.random.default_rng(seed)
    max_votes = counts.max(axis=1)
    gate = max_votes + rng.normal(0.0, sigma_threshold, size=N)
    answered = gate >= tau

    noisy_counts = counts + rng.normal(0.0, sigma_aggregator, size=counts.shape)
    labels = noisy_counts.argmax(axis=1)
    labels = np.where(answered, labels, -1)
    return ConfidentGNMaxResult(
        labels=labels,
        answered_mask=answered,
        n_answered=int(answered.sum()),
        n_total=N,
    )
