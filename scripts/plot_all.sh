#!/bin/bash
# Generate all paper plots of this repo: physmetrics time series and power spectra.
#
# Usage:
#   scripts/plot_all.sh [--results-dir DIR] [--out-dir DIR] [--legend]
# Defaults: results from results/ucast, plots to result_plots/paper

set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

python scripts/plot_physmetrics.py "$@"
python scripts/plot_spectra.py "$@"
