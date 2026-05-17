"""Reproductions of paper Figs 3-7."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def plot_acc_vs_epsilon(
    df: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
    method_col: str = "method",
    eps_col: str = "epsilon",
    acc_col: str = "accuracy",
) -> None:
    """Reproduce Fig 3 / Fig 6 layout: accuracy vs ε per method."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, sub in df.sort_values(eps_col).groupby(method_col):
        ax.plot(sub[eps_col], sub[acc_col], marker="o", label=name)
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel("Accuracy")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mia_vs_epsilon(
    df: pd.DataFrame,
    *,
    out_path: str | Path,
    title: str,
    method_col: str = "method",
    eps_col: str = "epsilon",
    auc_col: str = "mia_auc",
) -> None:
    """Reproduce Fig 4 / Fig 5 / Fig 7 layout: MIA AUC vs ε per method."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for name, sub in df.sort_values(eps_col).groupby(method_col):
        ax.plot(sub[eps_col], sub[auc_col], marker="s", label=name)
    ax.axhline(0.5, color="grey", linestyle="--", alpha=0.7, label="random")
    ax.set_xlabel(r"Privacy budget $\varepsilon$")
    ax.set_ylabel("MIA AUC")
    ax.set_xscale("log")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
