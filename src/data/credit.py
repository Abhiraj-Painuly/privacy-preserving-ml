"""UCI Credit Card Default of Clients loader (paper §IV-A bullet 4).

Financial dataset with 23 attributes (LIMIT_BAL, demographics, repayment status
PAY_0..PAY_6, bill amounts BILL_AMT1..6, payment amounts PAY_AMT1..6) and a
binary default-payment label.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
CACHE.mkdir(parents=True, exist_ok=True)

_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/"
    "default%20of%20credit%20card%20clients.xls"
)


@dataclass
class CreditData:
    df: pd.DataFrame
    target: str = "default"
    sensitive: tuple[str, ...] = ("SEX", "AGE", "MARRIAGE", "EDUCATION", "LIMIT_BAL")
    name: str = "credit"


def load_credit(
    cache: bool = True,
    n_rows: int | None = None,
    seed: int = 42,
) -> CreditData:
    """Download (or cache) UCI ``default of credit card clients``.

    The original Excel file uses the first row as a section header and the
    second row as column names; pandas handles this with ``header=1``.
    """
    cache_path = CACHE / "credit.parquet"
    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        df = pd.read_excel(_URL, header=1)
        df = df.rename(columns={"default payment next month": "default"})
        if "ID" in df.columns:
            df = df.drop(columns=["ID"])
        if cache:
            df.to_parquet(cache_path, index=False)
    if n_rows is not None and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return CreditData(df=df)
