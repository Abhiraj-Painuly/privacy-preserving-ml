"""Sensitivity-aware preprocessing (paper §III-C, §IV-B).

This module implements the four preprocessing stages described in the paper:

1. **k-anonymity classifier** (``kanon_split``): given a list of candidate
   sensitive columns, identify which are *quasi-identifiers* by checking whether
   their unique-value count fits inside ``k``-sized equivalence classes.
   Columns failing the check are tagged "sensitive" and routed to DP
   perturbation; the rest are kept as-is.

2. **Correlation-driven dimensionality reduction** (``correlation_pca``):
   collapses dense multicollinear blocks (e.g., Credit Card BILL_AMT/PAY_AMT,
   ρ > 0.9) to PCA components before DP injection so noise is not duplicated
   across redundant features (paper §III-C ¶4).

3. **Categorical handling** (``encode_categoricals``): one-hot encoding by
   default; ``mode="embedding"`` returns dense integer codes for downstream
   embedding lookups.

4. **Continuous handling** (``normalise_continuous``): standardisation, with an
   optional Gaussian-Mixture model for heavy-tailed columns
   (``mode="gmm"``, paper §III-C ¶3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureSplit:
    sensitive: list[str]
    non_sensitive: list[str]
    target: str


def kanon_split(
    df: pd.DataFrame,
    target: str,
    declared_sensitive: Iterable[str],
    k: int = 5,
) -> FeatureSplit:
    """k-anonymity heuristic to classify columns as sensitive vs non-sensitive.

    A column is marked **sensitive** when:

        * the user has flagged it (``declared_sensitive``), OR
        * its number of distinct values is large enough that grouping by it
          would produce equivalence classes smaller than ``k``.

    The intuition (paper §III-C ¶5) is that high-cardinality columns are
    quasi-identifiers and therefore re-identification risks; they need DP
    perturbation. Low-cardinality columns can be safely retained.
    """
    declared = set(declared_sensitive)
    sensitive: list[str] = []
    non_sensitive: list[str] = []
    n = len(df)

    for col in df.columns:
        if col == target:
            continue
        if col in declared:
            sensitive.append(col)
            continue
        n_unique = df[col].nunique(dropna=True)
        # "Equivalence class size" approximation: average rows per unique value.
        if n_unique == 0:
            non_sensitive.append(col)
            continue
        avg_class_size = n / max(n_unique, 1)
        if avg_class_size < k:
            sensitive.append(col)
        else:
            non_sensitive.append(col)
    return FeatureSplit(sensitive=sensitive, non_sensitive=non_sensitive, target=target)


def correlation_pca(
    df: pd.DataFrame,
    columns: list[str],
    threshold: float = 0.85,
    keep_variance: float = 0.95,
) -> tuple[pd.DataFrame, dict]:
    """Collapse multicollinear blocks via PCA (paper §III-C ¶4).

    Returns a DataFrame containing the original columns *minus* any group whose
    pairwise correlations exceed ``threshold``, replaced with PCA components
    that retain ``keep_variance`` of the variance.
    """
    if not columns:
        return df.copy(), {"pca_groups": []}

    sub = df[columns].astype(float)
    corr = sub.corr().abs()
    visited: set[str] = set()
    groups: list[list[str]] = []

    for col in columns:
        if col in visited:
            continue
        cluster = [c for c in columns if (corr.loc[col, c] >= threshold and c not in visited)]
        if len(cluster) >= 2:
            groups.append(cluster)
            visited.update(cluster)
        else:
            visited.add(col)

    new_df = df.copy()
    pca_meta: list[dict] = []
    for i, cluster in enumerate(groups):
        pca = PCA(n_components=keep_variance, svd_solver="full")
        block = sub[cluster].values
        comps = pca.fit_transform(block)
        new_cols = [f"pca{i}_c{j}" for j in range(comps.shape[1])]
        for j, name in enumerate(new_cols):
            new_df[name] = comps[:, j]
        new_df = new_df.drop(columns=cluster)
        pca_meta.append({"members": cluster, "components": new_cols, "explained": pca.explained_variance_ratio_.tolist()})
    return new_df, {"pca_groups": pca_meta}


def encode_categoricals(
    df: pd.DataFrame,
    target: str,
    mode: Literal["onehot", "embedding"] = "onehot",
) -> tuple[pd.DataFrame, dict]:
    """One-hot or integer-code categorical columns.

    Numeric columns are passed through untouched. The target column is
    preserved as-is.
    """
    cat_cols = [c for c in df.columns if df[c].dtype == "object" and c != target]
    meta: dict[str, list[str]] = {"categoricals": cat_cols}

    if mode == "onehot":
        encoded = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=float)
        return encoded, meta
    # Integer codes for embedding lookups
    out = df.copy()
    code_maps: dict[str, dict] = {}
    for c in cat_cols:
        codes, uniques = pd.factorize(out[c])
        out[c] = codes
        code_maps[c] = {v: i for i, v in enumerate(uniques)}
    meta["code_maps"] = code_maps
    return out, meta


def normalise_continuous(
    df: pd.DataFrame,
    columns: list[str],
    mode: Literal["zscore", "gmm"] = "zscore",
    gmm_components: int = 3,
) -> tuple[pd.DataFrame, dict]:
    """Normalise continuous columns.

    ``zscore`` is the standard StandardScaler (paper §III-C "continuous
    variables were normalized"); ``gmm`` fits a 3-component Gaussian Mixture
    per column and replaces each value with its responsibility-weighted mean,
    matching the paper's "modeled with Gaussian mixtures" option.
    """
    out = df.copy()
    meta: dict = {"mode": mode, "params": {}}
    if not columns:
        return out, meta

    if mode == "zscore":
        scaler = StandardScaler()
        out[columns] = scaler.fit_transform(out[columns].astype(float))
        meta["params"]["mean"] = scaler.mean_.tolist()
        meta["params"]["scale"] = scaler.scale_.tolist()
        return out, meta

    # GMM mode
    for col in columns:
        gmm = GaussianMixture(n_components=gmm_components, random_state=0)
        x = out[[col]].astype(float).values
        gmm.fit(x)
        resp = gmm.predict_proba(x)
        means = gmm.means_.flatten()
        out[col] = (resp * means).sum(axis=1)
        meta["params"][col] = {"means": means.tolist(), "weights": gmm.weights_.tolist()}
    return out, meta


def impute(df: pd.DataFrame) -> pd.DataFrame:
    """Reversible mean/mode imputation (paper §III-C bullet 3)."""
    out = df.copy()
    for col in out.columns:
        if out[col].isna().any():
            if out[col].dtype.kind in "biufc":
                out[col] = out[col].fillna(out[col].mean())
            else:
                out[col] = out[col].fillna(out[col].mode().iloc[0])
    return out
