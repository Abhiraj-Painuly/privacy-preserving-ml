#!/usr/bin/env bash
# Reproduce every numeric result and figure from the ICOIN paper.
#
# Pass --quick to use sub-samples and a small number of epochs (~5-10 min on a
# laptop CPU). Without it, the full sweep can take several hours on CPU; a
# CUDA GPU is recommended.

set -euo pipefail
cd "$(dirname "$0")/.."

QUICK=""
if [[ "${1:-}" == "--quick" ]]; then
  QUICK="--quick"
fi

echo "[1/4] Fig 2: Pearson correlation heatmaps"
python experiments/run_fig2_heatmaps.py $QUICK

echo "[2/4] Table III: AE accuracy at strict ε"
python experiments/run_table_iii.py $QUICK

echo "[3/4] Figs 3-5: AE sweep across ε ∈ {0.1, 0.5, 1.0, 5.0, 10.0}"
python experiments/run_fig3_5_ae_sweep.py $QUICK

echo "[4/4] Figs 6-7: Ensemble sweep (Laplace DP, confidence fusion)"
python experiments/run_fig6_7_ensemble_sweep.py $QUICK

echo ""
echo "All results written to ./results/"
