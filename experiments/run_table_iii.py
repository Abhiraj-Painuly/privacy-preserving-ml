"""Reproduce Table III — Adult dataset, 3 AEs × ε ∈ {0.1, 0.5, 1.0}.

Outputs ``results/table_iii.csv`` with columns ``epsilon, ae, accuracy,
mia_aggregated_auc, mia_gap_auc, realised_epsilon``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data import load_adult  # noqa: E402
from src.pipelines import run_ae_pipeline  # noqa: E402
from src.utils import RESULTS_DIR, set_seed  # noqa: E402

EPSILONS = [0.1, 0.5, 1.0]
AES = ["tab_dae", "dp_vae", "vanilla_ae"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="train fewer epochs / smaller subset")
    p.add_argument("--no-dpsgd", action="store_true", help="disable Opacus (faster, NOT private)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--n-rows", type=int, default=None)
    args = p.parse_args()

    epochs = args.epochs or (3 if args.quick else 10)
    n_rows = args.n_rows or (4000 if args.quick else None)

    set_seed(42)
    data = load_adult(n_rows=n_rows)
    rows: list[dict] = []

    for eps in EPSILONS:
        for ae in AES:
            print(f"[table_iii] AE={ae}, epsilon={eps}")
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
                    "ae": ae,
                    "accuracy": round(res.accuracy, 4),
                    "mia_aggregated_auc": round(res.mia_aggregated_auc, 4),
                    "mia_gap_auc": round(res.mia_gap_auc, 4),
                    "realised_epsilon": round(res.realised_epsilon, 4),
                }
            )
    df = pd.DataFrame(rows)
    out = RESULTS_DIR / "table_iii.csv"
    df.to_csv(out, index=False)
    pivot = df.pivot(index="epsilon", columns="ae", values="accuracy")
    pivot = pivot.reindex(columns=AES)  # match paper column order
    print("\n=== Reproduced Table III (accuracy) ===")
    print(pivot.to_string())
    print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
