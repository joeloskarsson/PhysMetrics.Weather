#!/bin/bash

#SBATCH --job-name=physmetrics_ref_balance
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -A ab016
#SBATCH -c 72
#SBATCH --time=1:00:00
#SBATCH --output=logs/physmetrics_ref_balance_%j.out

# Precompute ERA5 hydrostatic/geostrophic balance RMSEs at every valid time reached by 10-day
# forecasts initialized in 2020, used as reference in scripts/plot_physmetrics.py.
#
# Usage:
#   sbatch scripts/submit_ref_balance.sh [extra physmetrics-ref-balance args]
#
# Writes results/ucast/reference_balance_2020.csv

set -euo pipefail

PHYS_DIR=/capstor/store/cscs/swissai/ab016/ojoel/physmetrics
# ERA5 on the same 1.5deg grid the forecasts were made on
REF_ZARR=/iopsstor/datacache/cscs/swissai/ab016/era5_1.5deg.zarr
OUTPUT_DIR=results/ucast

ARGS=(
    --ref-zarr "$REF_ZARR"
    --year 2020
    --max-lead 10d
    --workers 72
    --output-dir "$OUTPUT_DIR"
    "$@"
)

echo "Computing ERA5 reference balance from $REF_ZARR"
srun --cpu-bind=none --container-writable -A ab016 --environment=cuda129_ub2404 \
    bash -c "source $PHYS_DIR/.venv/bin/activate && cd $PHYS_DIR && \
        physmetrics-ref-balance $(printf '%q ' "${ARGS[@]}")"
