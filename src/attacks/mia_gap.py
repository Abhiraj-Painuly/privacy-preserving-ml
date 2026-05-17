"""GAP MIA — the simplest attack baseline (paper Fig 5).

The 'GAP attack' from Yeom et al. (2018) merely thresholds the per-sample
loss: if a sample's loss is below the median of training losses, predict
"member"; else predict "non-member". The AUC of this score against true
membership is the GAP MIA AUC.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from .mia_aggregated import MIAResult


def gap_mia_auc(
    proba_member: np.ndarray,
    y_member: np.ndarray,
    proba_non: np.ndarray,
    y_non: np.ndarray,
) -> MIAResult:
    """AUC of "loss-as-membership-score"."""
    proba_m = np.clip(proba_member, 1e-12, 1.0)
    proba_n = np.clip(proba_non, 1e-12, 1.0)
    loss_m = -np.log(proba_m[np.arange(len(y_member)), y_member])
    loss_n = -np.log(proba_n[np.arange(len(y_non)), y_non])
    # Lower loss => more likely a member
    scores = -np.concatenate([loss_m, loss_n])
    labels = np.concatenate([np.ones(len(loss_m)), np.zeros(len(loss_n))])
    auc = float(roc_auc_score(labels, scores))
    return MIAResult(auc=auc, n_members=len(loss_m), n_non_members=len(loss_n))
