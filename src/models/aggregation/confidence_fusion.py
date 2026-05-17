"""Confidence-based fusion (paper §III-B).

Given an ensemble's per-model softmax outputs, return a single fused softmax.
Two strategies are supported:

* ``mode="weighted"`` — weight each model by its average top-1 confidence on
  the input batch (so confident models dominate), with optional **temperature
  scaling** to flatten over-confident peaks (paper §V-B "Confidence
  calibration methods, such as temperature scaling…").

* ``mode="mean"`` — uniform soft-vote; useful baseline.
"""
from __future__ import annotations

from typing import Literal

import numpy as np


def confidence_fusion(
    proba_stack: np.ndarray,
    *,
    mode: Literal["mean", "weighted"] = "weighted",
    temperature: float = 1.0,
) -> np.ndarray:
    """Fuse a stack of softmax predictions.

    Parameters
    ----------
    proba_stack:
        Array of shape ``(T, N, K)`` — T teachers, N queries, K classes.
    mode:
        Fusion strategy.
    temperature:
        Applied as ``softmax(log_p / temperature)`` per teacher before fusion.
    """
    if proba_stack.ndim != 3:
        raise ValueError("proba_stack must have shape (T, N, K)")

    if temperature != 1.0:
        log_p = np.log(np.clip(proba_stack, 1e-12, None))
        log_p = log_p / temperature
        m = np.max(log_p, axis=-1, keepdims=True)
        proba_stack = np.exp(log_p - m)
        proba_stack = proba_stack / proba_stack.sum(axis=-1, keepdims=True)

    if mode == "mean":
        return proba_stack.mean(axis=0)

    # weighted: weight by per-teacher top-1 confidence
    top1 = proba_stack.max(axis=-1)  # (T, N)
    w = top1 / top1.sum(axis=0, keepdims=True)  # normalise across teachers
    return (proba_stack * w[..., None]).sum(axis=0)
