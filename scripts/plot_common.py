"""
Loading of physmetrics result CSVs for the plotting scripts.

Results of run <run> are in <results_dir>/<kind>_<run>_<YEAR>.csv, with kind one of
physics_evaluation, time_series, spectra.
"""

import warnings
from pathlib import Path

import pandas as pd

import plot_style

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results" / "ucast"
DEFAULT_OUT_DIR = REPO_ROOT / "result_plots" / "paper"
YEAR = 2020

# Run name used in the filename, for runs whose results are not in <kind>_<run>_<YEAR>.csv.
# reprod_train_12e_2020.csv files are from an older physmetrics version (fewer leads, only
# KE_500 spectra truncated at l=59); the up-to-date re-run got a doubled year suffix.
RUN_FILE_OVERRIDES = {
    "reprod_train_12e": "reprod_train_12e_2020",
}

HOURS_PER_DAY = 24


def result_path(results_dir, kind, run):
    return Path(results_dir) / f"{kind}_{RUN_FILE_OVERRIDES.get(run, run)}_{YEAR}.csv"


def read_run_csvs(results_dir, kind, model, **read_csv_kwargs):
    """Read the result CSV of each seed of model, as {run: DataFrame}, skipping missing files."""
    frames = {}
    for run in plot_style.seed_run_names(model):
        path = result_path(results_dir, kind, run)
        if not path.exists():
            warnings.warn(f"Missing {path}, averaging {model} over the remaining seeds")
            continue
        frames[run] = pd.read_csv(path, **read_csv_kwargs)
    return frames


def lead_label(lead_hours):
    return f"{lead_hours}h"


def add_common_args(parser):
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--legend", action="store_true", help="Also save plots with legend (<name>_with_legend)"
    )
    return parser
