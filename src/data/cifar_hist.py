"""CIFAR-10 → 96-bin colour histogram loader (paper §IV-A bullet 1).

Each 32×32 RGB image is converted into a tabular feature vector by computing a
32-bin histogram **per RGB channel** (32 × 3 = 96 bins). This matches the
paper's description: "Each image is converted into 96-bin color histograms,
producing structured numerical features suitable for tabular representation."
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

CACHE = Path(__file__).resolve().parents[2] / "data_cache"
CACHE.mkdir(parents=True, exist_ok=True)


@dataclass
class CifarHistData:
    df: pd.DataFrame
    target: str = "label"
    # In an image-histogram dataset every bin is "sensitive"; the structure of
    # the paper's intra-channel correlations (Fig 2a) guides the DP strategy.
    sensitive: tuple[str, ...] = tuple(
        f"{ch}_bin{i}" for ch in ("R", "G", "B") for i in range(32)
    )
    name: str = "cifar_hist"


def _images_to_hist(images: np.ndarray, n_bins: int = 32) -> np.ndarray:
    """Vectorised per-channel histogram extraction.

    Parameters
    ----------
    images:
        Shape (N, 32, 32, 3), uint8 values in [0, 255].
    """
    N = images.shape[0]
    out = np.zeros((N, 3 * n_bins), dtype=np.float32)
    edges = np.linspace(0, 256, n_bins + 1)
    for c in range(3):
        for i in range(N):
            h, _ = np.histogram(images[i, :, :, c], bins=edges)
            out[i, c * n_bins : (c + 1) * n_bins] = h
    # Normalise so each row sums to ~1 per channel; keeps DP sensitivity bounded.
    out = out / (32 * 32)
    return out


def _load_torchvision(split: Literal["train", "test"]) -> tuple[np.ndarray, np.ndarray]:
    import torchvision

    ds = torchvision.datasets.CIFAR10(
        root=str(CACHE / "cifar10"), train=(split == "train"), download=True
    )
    images = ds.data  # (N, 32, 32, 3)
    labels = np.array(ds.targets)
    return images, labels


def load_cifar_hist(
    n_rows: int | None = None,
    seed: int = 42,
    cache: bool = True,
) -> CifarHistData:
    """Build the 96-bin tabular representation of CIFAR-10."""
    cache_path = CACHE / "cifar_hist.parquet"
    if cache and cache_path.exists():
        df = pd.read_parquet(cache_path)
    else:
        train_imgs, train_lbl = _load_torchvision("train")
        test_imgs, test_lbl = _load_torchvision("test")
        imgs = np.concatenate([train_imgs, test_imgs], axis=0)
        lbl = np.concatenate([train_lbl, test_lbl], axis=0)
        feats = _images_to_hist(imgs)
        cols = [f"{ch}_bin{i}" for ch in ("R", "G", "B") for i in range(32)]
        df = pd.DataFrame(feats, columns=cols)
        df["label"] = lbl
        if cache:
            df.to_parquet(cache_path, index=False)
    if n_rows is not None and n_rows < len(df):
        df = df.sample(n=n_rows, random_state=seed).reset_index(drop=True)
    return CifarHistData(df=df)
