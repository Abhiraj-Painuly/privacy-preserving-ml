"""Reproduce paper Figs 6, 7 — Ensemble accuracy and MIA AUC vs ε.

Sweeps ε ∈ {0.1, 0.5, 1.0, 5.0, 10.0} on the Adult dataset using the full
five-backbone ensemble (paper §III-B) with **Laplace-DP perturbation** at
the preprocessing stage and **confidence-based fusion** at the aggregator
(the configuration the paper reports as the best, hitting 88.97% acc).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.analysis.plots import plot_acc_vs_epsilon, plot_mia_vs_epsilon  # noqa: E402
from src.data import load_adult  # noqa: E402
from src.pipelines import run_ensemble_pipeline  # noqa: E402
from src.utils import RESULTS_DIR, set_seed  # noqa: E402

EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-dpsgd", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-rows", type=int, default=None)
    p.add_argument("--aggregator", choices=["confidence", "gnmax"], default="confidence")
    args = p.parse_args()

    epochs = args.epochs or (2 if args.quick else 8)
    n_rows = args.n_rows or (4000 if args.quick else None)

    set_seed(42)
    data = load_adult(n_rows=n_rows)
    rows: list[dict] = []
    for eps in EPSILONS:
        print(f"[fig67] epsilon={eps}")
        res = run_ensemble_pipeline(
            data.df,
            target=data.target,
            declared_sensitive=list(data.sensitive),
            epsilon=eps,
            mechanism="laplace",
            aggregator=args.aggregator,
            epochs=epochs,
            batch_size=128,
            use_dpsgd=not args.no_dpsgd,
        )
        row = {
            "epsilon": eps,
            "method": f"ensemble_laplace_{args.aggregator}",
            "accuracy": round(res.accuracy, 4),
            "mia_auc_aggregated": round(res.mia_aggregated_auc, 4),
            "mia_auc_gap": round(res.mia_gap_auc, 4),
        }
        for k, v in res.per_backbone_accuracy.items():
            row[f"acc_{k}"] = round(v, 4)
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "fig6_7_ensemble_sweep.csv", index=False)

    plot_acc_vs_epsilon(
        df,
        out_path=RESULTS_DIR / "fig6_ensemble_acc_vs_eps.png",
        title="Fig 6 — Ensemble accuracy vs ε (Laplace DP)",
    )
    plot_mia_vs_epsilon(
        df.rename(columns={"mia_auc_aggregated": "mia_auc"}),
        out_path=RESULTS_DIR / "fig7_ensemble_mia_vs_eps.png",
        title="Fig 7 — Ensemble MIA AUC vs ε (Laplace DP)",
    )
    print(f"\nWrote {RESULTS_DIR / 'fig6_7_ensemble_sweep.csv'} and Figs 6/7 PNGs.")
    print("\nHeadline accuracy:", df["accuracy"].max())


if __name__ == "__main__":
    main()
