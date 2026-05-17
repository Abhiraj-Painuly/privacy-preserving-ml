"""A small RDP-style privacy accountant used to track the cumulative epsilon.

For the preprocessing-stage Laplace / Gaussian noise it suffices to compose
budgets via *basic composition*: ε_total = Σ ε_i. For DP-SGD training we
delegate to Opacus's RDP accountant in :mod:`src.privacy.dp_sgd`. This module
exposes a uniform interface so the pipeline reports one ε per dataset/run.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PrivacyAccountant:
    """Simple basic-composition accountant.

    Use :meth:`spend` to log an event, then :meth:`epsilon` / :meth:`delta`
    to obtain the cumulative budget.
    """

    history: list[tuple[str, float, float]] = field(default_factory=list)

    def spend(self, name: str, epsilon: float, delta: float = 0.0) -> None:
        if epsilon < 0 or delta < 0:
            raise ValueError("epsilon and delta must be non-negative")
        self.history.append((name, epsilon, delta))

    def epsilon(self) -> float:
        return sum(e for _, e, _ in self.history)

    def delta(self) -> float:
        # Basic composition adds δ linearly.
        return sum(d for _, _, d in self.history)

    def summary(self) -> dict:
        return {
            "events": [
                {"name": n, "epsilon": e, "delta": d} for n, e, d in self.history
            ],
            "total_epsilon": self.epsilon(),
            "total_delta": self.delta(),
        }
