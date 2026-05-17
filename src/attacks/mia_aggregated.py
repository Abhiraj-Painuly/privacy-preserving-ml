"""Aggregated black-box Membership Inference Attack (paper Fig 4, Fig 7).

We follow the standard threshold-on-confidence formulation:

    Given a target model f and (x, y),
        score(x, y) = -CrossEntropy(f(x), y)   (higher = more confident)

    A logistic-regression "attack model" is fit on (score, was_member) over a
    held-out *attack train* split where membership labels are known. The
    attack's AUC on a separate *attack test* split is reported.

This is the "Aggregated MIA AUC" curve in Figs 4 and 7. We additionally
report the per-class softmax entropy as an auxiliary feature, mirroring the
"shadow-model" attack of Shokri et al. (2017) but at much smaller cost.

Inputs are NumPy arrays so the function works for both PyTorch backbones and
the DP-XGBoost wrapper.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def _confidence_features(proba: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Build (loss, max_prob, entropy, true_class_prob) features."""
    proba = np.clip(proba, 1e-12, 1.0)
    loss = -np.log(proba[np.arange(len(y)), y])
    max_p = proba.max(axis=1)
    entropy = -(proba * np.log(proba)).sum(axis=1)
    true_p = proba[np.arange(len(y)), y]
    return np.stack([loss, max_p, entropy, true_p], axis=1)


@dataclass
class MIAResult:
    auc: float
    n_members: int
    n_non_members: int


def aggregated_mia_auc(
    proba_member: np.ndarray,
    y_member: np.ndarray,
    proba_non: np.ndarray,
    y_non: np.ndarray,
    *,
    seed: int = 42,
) -> MIAResult:
    """Compute the aggregated MIA AUC.

    Parameters
    ----------
    proba_member, y_member:
        Softmax probabilities and true labels for samples that were *in* the
        target model's training set.
    proba_non, y_non:
        Same, but for samples that were *not* used to train the target.
    """
    feat_m = _confidence_features(proba_member, y_member)
    feat_n = _confidence_features(proba_non, y_non)
    X = np.concatenate([feat_m, feat_n], axis=0)
    y = np.concatenate([np.ones(len(feat_m)), np.zeros(len(feat_n))])
    # Train/test split for the attack model itself
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    half = len(y) // 2
    train_idx, test_idx = idx[:half], idx[half:]
    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_idx], y[train_idx])
    score = clf.predict_proba(X[test_idx])[:, 1]
    auc = float(roc_auc_score(y[test_idx], score))
    return MIAResult(auc=auc, n_members=len(feat_m), n_non_members=len(feat_n))
