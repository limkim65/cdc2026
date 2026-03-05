#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash experiments/reacher/run_adaptive_rho_kmin_grid.sh
# Optional env overrides:
#   SEEDS=10 MAX_STEPS=200 N_ITER=1 KMAX=320 KSTEP=1 GAMMA_MIN=1e-6 DATASET=iid \
#   bash experiments/reacher/run_adaptive_rho_kmin_grid.sh
#   PYTHON_BIN="/c/Users/yelim/anaconda3/envs/InvPen_DD/python.exe" \
#   bash experiments/reacher/run_adaptive_rho_kmin_grid.sh

SEEDS="${SEEDS:-10}"
MAX_STEPS="${MAX_STEPS:-200}"
N_ITER="${N_ITER:-1}"
KMAX="${KMAX:-320}"
KSTEP="${KSTEP:-1}"
GAMMA_MIN="${GAMMA_MIN:-1e-6}"
DATASET="${DATASET:-iid}"
PYTHON_BIN="${PYTHON_BIN:-}"

if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[error] python not found in PATH. Set PYTHON_BIN to your interpreter, e.g." >&2
    echo "  PYTHON_BIN=\"/c/Users/yelim/anaconda3/envs/InvPen_DD/python.exe\" bash $0" >&2
    exit 127
  fi
fi

RHO_LIST=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)
KMIN_LIST=(20 40 60)

for KMIN in "${KMIN_LIST[@]}"; do
  OUTDIR="logs/reacher/adaptive_k_dir/reacher_cdc_adapt_rho_seed${SEEDS}_Kmin${KMIN}"
  echo "[run] Kmin=${KMIN} -> ${OUTDIR}"
  "${PYTHON_BIN}" experiments/reacher/run_cdc_reacher_benchmark.py \
    --mode adaptive_rho \
    --outdir "${OUTDIR}" \
    --seeds "${SEEDS}" \
    --max_steps "${MAX_STEPS}" \
    --n_iter "${N_ITER}" \
    --Kmin "${KMIN}" --Kmax "${KMAX}" --Kstep "${KSTEP}" \
    --rho "${RHO_LIST[@]}" \
    --gamma_min "${GAMMA_MIN}" \
    --dataset "${DATASET}"

  "${PYTHON_BIN}" make_figures.py \
    --adaptive_dir "${OUTDIR}" \
    --out_dir "logs/reacher/figure/figs_cdc_adapt_only_seed${SEEDS}_Kmin${KMIN}" \
    --reach_eps 0.01 --reach_hold 20 --sigma_threshold 1e-9
done

echo "[done] Adaptive rho sweep grid finished."
