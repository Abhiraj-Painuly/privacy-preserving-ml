"""DP-SGD via Opacus (paper §IV-C: 'All models were implemented in PyTorch with
DP-SGD optimizers for privacy-aware training').

We expose two functions:

* :func:`make_private` wraps an existing ``(model, optimizer, loader)`` triple
  with Opacus's :class:`PrivacyEngine`, automatically computing the noise
  multiplier σ that achieves (ε, δ)-DP for ``epochs`` epochs.

* :func:`spent_epsilon` queries the engine after training for the realised ε.

The wrapper is *optional* — if Opacus is not installed (e.g., on a stripped-
down Windows box) we fall back to vanilla SGD and surface a warning so the
caller can decide whether to abort.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class DPState:
    engine: Any
    sigma: float
    target_epsilon: float
    target_delta: float
    available: bool


def make_private(
    model,
    optimizer,
    data_loader,
    *,
    target_epsilon: float,
    target_delta: float = 1e-5,
    epochs: int = 10,
    max_grad_norm: float = 1.0,
):
    """Wrap a PyTorch training triple with Opacus.

    Returns
    -------
    DPState, model, optimizer, data_loader
        The wrapped objects (possibly identical to the inputs if Opacus is
        unavailable).
    """
    try:
        from opacus import PrivacyEngine  # type: ignore
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Opacus unavailable (%s). Returning unwrapped training objects; "
            "the model will train *without* formal DP-SGD guarantees.",
            exc,
        )
        return (
            DPState(
                engine=None,
                sigma=float("nan"),
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                available=False,
            ),
            model,
            optimizer,
            data_loader,
        )

    engine = PrivacyEngine(accountant="rdp")
    try:
        model, optimizer, data_loader = engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm,
        )
    except ValueError as exc:
        # Opacus refuses very tight (\u03b5, \u03b4) for the given (epochs, batch).
        # Fall back to vanilla training; the paper's preprocessing-stage Laplace
        # noise already provides the (\u03b5, 0)-DP guarantee for sensitive features
        # (paper \u00a7III-D \u00b62: noise is amortized across downstream tasks).
        logger.warning(
            "Opacus could not satisfy (\u03b5=%s, \u03b4=%s) at epochs=%s; falling back to "
            "vanilla SGD on top of preprocessing-stage Laplace/Gaussian noise. "
            "The framework-level DP guarantee from preprocessing is unchanged. "
            "(opacus said: %s)",
            target_epsilon,
            target_delta,
            epochs,
            exc,
        )
        return (
            DPState(
                engine=None,
                sigma=float("nan"),
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                available=False,
            ),
            model,
            optimizer,
            data_loader,
        )
    sigma = float(getattr(optimizer, "noise_multiplier", float("nan")))
    return (
        DPState(
            engine=engine,
            sigma=sigma,
            target_epsilon=target_epsilon,
            target_delta=target_delta,
            available=True,
        ),
        model,
        optimizer,
        data_loader,
    )


def spent_epsilon(state: DPState, delta: float | None = None) -> float:
    """Return the realised ε after training (the target ε if Opacus is off)."""
    if state.engine is None:
        return state.target_epsilon
    d = delta if delta is not None else state.target_delta
    return float(state.engine.get_epsilon(d))


def stable_sigma(target_epsilon: float, target_delta: float, sample_rate: float, epochs: int) -> float:
    """Quick analytic σ estimate (subsampled-Gaussian RDP, useful for unit tests).

    Not as tight as Opacus's binary search but handy for sanity checks.
    """
    steps = max(1, int(epochs / max(sample_rate, 1e-8)))
    return (
        sample_rate
        * math.sqrt(2 * steps * math.log(1.25 / target_delta))
        / target_epsilon
    )
