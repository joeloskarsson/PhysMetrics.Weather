#!/bin/bash

#SBATCH --job-name=physmetrics
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -A ab016
#SBATCH -c 72
#SBATCH --time=1:00:00
#SBATCH --output=logs/physmetrics_%j.out

# Run physmetrics on a saved 2020 forecast zarr (e.g. from ucast's scripts/eval_run.sh).
#
# Usage:
#   sbatch scripts/submit_physmetrics.sh <model_name> <prediction_zarr> [extra physmetrics-run args]
#   sbatch scripts/submit_physmetrics.sh mcrps_spectral /iopsstor/scratch/cscs/ojoel/ucast_fc/mcrps_spectral_2020.zarr
#
# Writes results/ucast/{physics_evaluation,time_series,spectra,lapse_rate_dist}_<model_name>_2020.csv

set -euo pipefail

PHYS_DIR=/capstor/store/cscs/swissai/ab016/ojoel/physmetrics
# ERA5 on the same 1.5deg grid the forecasts were made on
REF_ZARR=/capstor/store/cscs/swissai/ab016/ERA5_coarse/era5_1.5deg.zarr
OUTPUT_DIR=results/ucast

if [[ $# -lt 2 ]]; then
    echo "Usage: sbatch scripts/submit_physmetrics.sh <model_name> <prediction_zarr> [extra args]" >&2
    exit 1
fi
MODEL=$1
PREDICTION_ZARR=$2
shift 2

ARGS=(
    --model "$MODEL"
    --prediction-zarr "$PREDICTION_ZARR"
    --ref-zarr "$REF_ZARR"
    --year 2020
    --workers 72
    --lead-times 12h,1d,5d,10d
    --output-dir "$OUTPUT_DIR"
    "$@"
)

echo "Running physmetrics for $MODEL on $PREDICTION_ZARR"
srun --cpu-bind=none --container-writable -A ab016 --environment=cuda129_ub2404 \
    bash -c "source $PHYS_DIR/.venv/bin/activate && cd $PHYS_DIR && \
        physmetrics-run $(printf '%q ' "${ARGS[@]}")"
