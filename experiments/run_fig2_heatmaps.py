"""Reproduce paper Fig 2 — Pearson correlation heatmaps for all 4 datasets.

Outputs
-------
results/fig2_adult.png, fig2_credit.png, fig2_health.png, fig2_cifar_hist.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.analysis.correlation import plot_correlation_heatmap  # noqa: E402
from src.data import load_adult, load_credit, load_health  # noqa: E402
from src.utils import RESULTS_DIR  # noqa: E402


def _adult_panel(quick: bool) -> None:
    data = load_adult(n_rows=2000 if quick else None)
    plot_correlation_heatmap(
        data.df,
        title="(b) Adult Income — Pearson correlation",
        out_path=RESULTS_DIR / "fig2_adult.png",
        drop=[data.target],
    )


def _credit_panel(quick: bool) -> None:
    try:
        data = load_credit(n_rows=2000 if quick else None)
    except Exception as exc:
        print(f"[fig2_credit] skipped: {exc}")
        return
    plot_correlation_heatmap(
        data.df,
        title="(d) Credit Card Default — Pearson correlation",
        out_path=RESULTS_DIR / "fig2_credit.png",
        drop=[data.target],
    )


def _health_panel(quick: bool) -> None:
    data = load_health(n_rows=2000 if quick else 8000)
    plot_correlation_heatmap(
        data.df,
        title="(c) Health Vitals — Pearson correlation",
        out_path=RESULTS_DIR / "fig2_health.png",
        drop=[data.target],
        annot=True,
        figsize=(6.0, 5.0),
    )


def _cifar_panel(quick: bool) -> None:
    """CIFAR-10 96-bin histogram heatmap (Fig 2a). Requires torchvision download."""
    try:
        from src.data import load_cifar_hist
    except Exception as exc:  # pragma: no cover
        print(f"[fig2_cifar] import failed: {exc}")
        return
    try:
        data = load_cifar_hist(n_rows=3000 if quick else 10000)
    except Exception as exc:
        print(f"[fig2_cifar] skipped (download failed?): {exc}")
        return
    plot_correlation_heatmap(
        data.df,
        title="(a) CIFAR-10 96-bin Histogram — Pearson correlation",
        out_path=RESULTS_DIR / "fig2_cifar_hist.png",
        drop=[data.target],
        annot=False,
        figsize=(8.0, 7.0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="use sub-samples for speed")
    p.add_argument(
        "--skip-cifar",
        action="store_true",
        help="skip the CIFAR panel (heavy torchvision download)",
    )
    args = p.parse_args()

    _adult_panel(args.quick)
    _credit_panel(args.quick)
    _health_panel(args.quick)
    if not args.skip_cifar:
        _cifar_panel(args.quick)
    print("Wrote heatmaps to", RESULTS_DIR)


if __name__ == "__main__":
    main()
