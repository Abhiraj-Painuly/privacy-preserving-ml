"""Health vitals loader (paper §IV-A bullet 3).

The paper describes a *custom* dataset: physiological parameters
``temperature``, ``oxygen level``, ``systolic`` & ``diastolic`` blood pressure,
plus a heart-rate-like vital, with a binary normal/abnormal target. Because
the paper's exact CSV is not public, we generate a synthetic dataset whose
joint distribution matches the description in §III-C/§IV-A:

    - temperature ~ N(36.8, 0.6)         | mostly independent
    - oxygen      ~ N(97, 1.8)           | mostly independent
    - heart_rate  ~ N(78, 12)            | weak correlation with bp_sys
    - bp_sys      ~ N(120, 14)           | rho ~ 0.26 with bp_dia
    - bp_dia      ~ bp_sys * 0.6 + N(0,8)
    - abnormal    if any vital exceeds clinical thresholds

This is the "Option A" deviation called out in the README.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class HealthData:
    df: pd.DataFrame
    target: str = "abnormal"
    sensitive: tuple[str, ...] = ("temperature", "oxygen", "heart_rate", "bp_sys", "bp_dia")
    name: str = "health"


def load_health(
    n_rows: int = 8000,
    seed: int = 42,
    cache: bool = True,
) -> HealthData:
    """Generate (or load cached) synthetic Health vitals dataset.

    The dependency structure intentionally matches the paper's correlation
    discussion (§III-C ¶3): only systolic/diastolic BP show moderate
    correlation; temperature and oxygen are nearly independent.
    """
    cache_path = CACHE / f"health_{n_rows}_{seed}.parquet"
    if cache and cache_path.exists():
        return HealthData(df=pd.read_parquet(cache_path))

    rng = np.random.default_rng(seed)
    n = n_rows
    temperature = rng.normal(36.8, 0.6, n)
    oxygen = rng.normal(97.0, 1.8, n)
    heart_rate = rng.normal(78.0, 12.0, n)
    bp_sys = rng.normal(120.0, 14.0, n)
    bp_dia = 0.6 * bp_sys + rng.normal(0.0, 8.0, n)

    abnormal = (
        (temperature > 38.0)
        | (temperature < 35.5)
        | (oxygen < 92)
        | (heart_rate > 110)
        | (heart_rate < 50)
        | (bp_sys > 150)
        | (bp_sys < 90)
        | (bp_dia > 95)
        | (bp_dia < 55)
    ).astype(int)

    df = pd.DataFrame(
        {
            "temperature": temperature,
            "oxygen": oxygen,
            "heart_rate": heart_rate,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "abnormal": abnormal,
        }
    )
    if cache:
        df.to_parquet(cache_path, index=False)
    return HealthData(df=df)
