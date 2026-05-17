"""UCI Adult Income loader (paper §IV-A bullet 2).

The Adult dataset contains demographic and employment attributes; the binary
label is whether income > $50K.

We try ``ucimlrepo`` first (clean, online), and fall back to the canonical
URLs if it is unavailable.
"""
from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
CACHE.mkdir(parents=True, exist_ok=True)


# Per the UCI Adult schema. Order matters because the raw CSV has no header.
_COLS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
_TRAIN_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
)
_TEST_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
)


@dataclass
class AdultData:
    df: pd.DataFrame
    target: str = "income"
    sensitive: tuple[str, ...] = (
        "age",
        "sex",
        "race",
        "native-country",
        "marital-status",
        "relationship",
    )
    name: str = "adult"


def _read_url(url: str, has_header_skip: bool = False) -> pd.DataFrame:
    """Fetch and parse one of the two raw Adult files."""
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310
        text = r.read().decode("utf-8")
    if has_header_skip:
        # adult.test starts with a junk first line "|1x3 Cross validator"
        text = "\n".join(text.split("\n")[1:])
    df = pd.read_csv(
        StringIO(text), header=None, names=_COLS, skipinitialspace=True, na_values="?"
    )
    df["income"] = df["income"].str.rstrip(".").map({">50K": 1, "<=50K": 0})
    return df


def load_adult(
    cache: bool = True,
    n_rows: int | None = None,
    seed: int = 42,
) -> AdultData:
    """Load the Adult dataset, dropping rows with missing values.

    Parameters
    ----------
    cache:
        If True, store / read a parquet cache under ``data_cache/``.
    n_rows:
        Optional sub-sample size (used by ``--quick`` runs).
    seed:
        Sub-sampling seed.
    """
    cache_path = CACHE / "adult.parquet"
    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        train = _read_url(_TRAIN_URL)
        test = _read_url(_TEST_URL, has_header_skip=True)
        df = pd.concat([train, test], ignore_index=True)
        df = df.dropna().reset_index(drop=True)
        if cache:
            df.to_parquet(cache_path, index=False)
    if n_rows is not None and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return AdultData(df=df)
