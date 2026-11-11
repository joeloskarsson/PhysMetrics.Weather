#!/bin/bash
# Generate all plots
#
# Usage:
#   scripts/plot_all.sh [--results-dir DIR] [--out-dir DIR] [--legend]

set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg

python scripts/plot_physmetrics.py "$@"
python scripts/plot_spectra.py "$@"
