"""Reproduce paper Figs 3, 4, 5 — AE accuracy and MIA AUCs vs ε.

Sweeps ε ∈ {0.1, 0.5, 1.0, 5.0, 10.0} for each of the three autoencoders on
the Adult dataset.

Outputs
-------
results/fig3_5_ae_sweep.csv      tidy long-form table
results/fig3_acc_vs_eps.png      accuracy panel  (Fig 3)
results/fig4_mia_aggregated.png  Aggregated MIA AUC panel (Fig 4)
results/fig5_mia_gap.png         GAP MIA AUC panel (Fig 5)
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
from src.pipelines import run_ae_pipeline  # noqa: E402
from src.utils import RESULTS_DIR, set_seed  # noqa: E402

EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0]
AES = ["tab_dae", "dp_vae", "vanilla_ae"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--no-dpsgd", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-rows", type=int, default=None)
    args = p.parse_args()

    epochs = args.epochs or (2 if args.quick else 8)
    n_rows = args.n_rows or (4000 if args.quick else None)

    set_seed(42)
    data = load_adult(n_rows=n_rows)
    rows: list[dict] = []
    for eps in EPSILONS:
        for ae in AES:
            print(f"[fig345] AE={ae}, epsilon={eps}")
            res = run_ae_pipeline(
                data.df,
                target=data.target,
                declared_sensitive=list(data.sensitive),
                ae_name=ae,
                epsilon=eps,
                mechanism="laplace",
                epochs=epochs,
                batch_size=128,
                use_dpsgd=not args.no_dpsgd,
            )
            rows.append(
                {
                    "epsilon": eps,
                    "method": ae,
                    "accuracy": round(res.accuracy, 4),
                    "mia_auc_aggregated": round(res.mia_aggregated_auc, 4),
                    "mia_auc_gap": round(res.mia_gap_auc, 4),
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "fig3_5_ae_sweep.csv", index=False)

    plot_acc_vs_epsilon(
        df,
        out_path=RESULTS_DIR / "fig3_acc_vs_eps.png",
        title="Fig 3 — Autoencoder accuracy vs ε",
        method_col="method",
        eps_col="epsilon",
        acc_col="accuracy",
    )
    plot_mia_vs_epsilon(
        df.rename(columns={"mia_auc_aggregated": "mia_auc"}),
        out_path=RESULTS_DIR / "fig4_mia_aggregated.png",
        title="Fig 4 — Aggregated MIA AUC vs ε (autoencoders)",
    )
    plot_mia_vs_epsilon(
        df.rename(columns={"mia_auc_gap": "mia_auc"}),
        out_path=RESULTS_DIR / "fig5_mia_gap.png",
        title="Fig 5 — GAP MIA AUC vs ε (autoencoders)",
    )
    print(f"\nWrote {RESULTS_DIR / 'fig3_5_ae_sweep.csv'} and Figs 3/4/5 PNGs.")


if __name__ == "__main__":
    main()
