"""Pearson correlation heatmaps (paper Fig 2 a-d)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    df: pd.DataFrame,
    *,
    title: str,
    out_path: str | Path,
    drop: list[str] | None = None,
    annot: bool = False,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> None:
    """Render a Pearson-ρ heatmap and save it as PNG.

    The heatmap is what Fig 2 of the paper shows; the four files
    ``results/fig2_<dataset>.png`` reproduce panels (a)-(d).
    """
    drop = drop or []
    cols = [c for c in df.columns if c not in drop]
    numeric = df[cols].select_dtypes(include=[np.number])
    corr = numeric.corr(method="pearson")

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=annot,
        fmt=".2f",
        cmap="vlag",
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        square=False,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
