"""Shared preprocessing path used by both pipelines.

Steps (paper Fig 1):
    raw -> impute -> categorical-encode -> kanon split ->
        (sensitive: Laplace/Gaussian)  (non-sensitive: passthrough)
        -> normalise continuous columns
        -> optional correlation-PCA on sensitive block
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from ..data.preprocessing import (
    FeatureSplit,
    correlation_pca,
    encode_categoricals,
    impute,
    kanon_split,
    normalise_continuous,
)
from ..privacy.gaussian import GaussianMechanism
from ..privacy.laplace import LaplaceMechanism

DPMechanismName = Literal["laplace", "gaussian", "none"]


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    target_name: str
    sensitive_idx: list[int]
    non_sensitive_idx: list[int]
    pca_meta: dict


def _train_test_split(df: pd.DataFrame, target: str, test_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_test = int(len(df) * test_frac)
    test_df = df.iloc[idx[:n_test]].reset_index(drop=True)
    train_df = df.iloc[idx[n_test:]].reset_index(drop=True)
    return train_df, test_df


def prepare(
    df: pd.DataFrame,
    *,
    target: str,
    declared_sensitive: list[str],
    epsilon: float,
    mechanism: DPMechanismName = "laplace",
    delta: float = 1e-5,
    test_frac: float = 0.2,
    use_pca: bool = False,
    seed: int = 42,
) -> PreparedData:
    """Run the full sensitivity-aware preprocessing path (paper §III-A/§III-C)."""
    df = impute(df)
    df, _ = encode_categoricals(df, target=target, mode="onehot")

    train_df, test_df = _train_test_split(df, target=target, test_frac=test_frac, seed=seed)

    split: FeatureSplit = kanon_split(
        train_df, target=target, declared_sensitive=declared_sensitive, k=5
    )

    feature_cols = split.sensitive + split.non_sensitive
    # We keep sensitive columns first so we can index them as a contiguous block.
    X_train_df = train_df[feature_cols].astype(float)
    X_test_df = test_df[feature_cols].astype(float)
    y_train = train_df[target].astype(int).values
    y_test = test_df[target].astype(int).values

    pca_meta: dict = {"pca_groups": []}
    if use_pca and split.sensitive:
        # Apply PCA only on the sensitive block, then re-merge.
        sub_train, pca_meta = correlation_pca(X_train_df, columns=split.sensitive)
        # Use the same fitted ordering on test by appending zero-fill if mismatch
        # (we re-fit per-call for simplicity; deterministic given seed).
        sub_test, _ = correlation_pca(X_test_df, columns=split.sensitive)
        new_sensitive = [c for c in sub_train.columns if c not in split.non_sensitive]
        feature_cols = new_sensitive + split.non_sensitive
        X_train_df = sub_train[feature_cols].astype(float)
        X_test_df = sub_test.reindex(columns=feature_cols, fill_value=0.0).astype(float)
        split = FeatureSplit(
            sensitive=new_sensitive, non_sensitive=split.non_sensitive, target=target
        )

    # Normalise the continuous block (this includes the sensitive columns;
    # categorical one-hots are already in [0, 1] so the StandardScaler call
    # leaves them well-behaved).
    X_train_df, _norm_meta = normalise_continuous(X_train_df, columns=list(X_train_df.columns), mode="zscore")
    X_test_df, _ = normalise_continuous(X_test_df, columns=list(X_test_df.columns), mode="zscore")

    X_train = X_train_df.values.astype(np.float32)
    X_test = X_test_df.values.astype(np.float32)

    sensitive_idx = list(range(len(split.sensitive)))
    non_sensitive_idx = list(range(len(split.sensitive), len(feature_cols)))

    if mechanism != "none" and split.sensitive:
        sens_train = X_train[:, sensitive_idx]
        sens_test = X_test[:, sensitive_idx]

        if mechanism == "laplace":
            mech = LaplaceMechanism(epsilon=epsilon, seed=seed)
            sens_train = mech.perturb(
                sens_train,
                sensitivity=mech.sensitivity_from_range(
                    sens_train.min(axis=0), sens_train.max(axis=0)
                ),
            )
            sens_test = mech.perturb(
                sens_test,
                sensitivity=mech.sensitivity_from_range(
                    sens_test.min(axis=0), sens_test.max(axis=0)
                ),
            )
        else:
            mech = GaussianMechanism(epsilon=epsilon, delta=delta, seed=seed)
            sens_train = mech.perturb(sens_train)
            sens_test = mech.perturb(sens_test)

        X_train[:, sensitive_idx] = sens_train.astype(np.float32)
        X_test[:, sensitive_idx] = sens_test.astype(np.float32)

    return PreparedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
        target_name=target,
        sensitive_idx=sensitive_idx,
        non_sensitive_idx=non_sensitive_idx,
        pca_meta=pca_meta,
    )
