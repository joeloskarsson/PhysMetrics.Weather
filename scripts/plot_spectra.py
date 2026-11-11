"""
Paper plots of spherical-harmonic power spectra and their relative error w.r.t. ERA5.

For each spectra variable and lead time, writes
    spectra/spectra_<variable>_<lead>.{png,pdf}            power vs. wavelength, models + ERA5
    spectra/spectra_rel_error_<variable>_<lead>.{png,pdf}  (pred - ERA5) / ERA5
Spectra are averaged over init dates per seed, then shown as mean +- 1 std over seeds.

Usage:
    python scripts/plot_spectra.py [--results-dir DIR] [--out-dir DIR] [--legend] [--leads 12 120 240]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style

EARTH_RADIUS_KM = 6371.0
MIN_WAVENUMBER = 1
SPECTRA_SUBDIR = "spectra"
SPECTRA_COLUMNS = ["lead_hours", "variable", "wavenumber", "power_pred", "power_ref"]
GROUP_COLUMNS = ["lead_hours", "variable", "wavenumber"]

KINETIC_ENERGY_LABEL = "KE"
# Spectra variable -> (plot_style variable, pressure level); KE_<level> is handled separately
SPECTRA_VARIABLES = {
    "Q_700": ("specific_humidity", 700),
    "T_850": ("temperature", 850),
    "Z_500": ("geopotential", 500),
    "T2M": ("2m_temperature", None),
    "MSL": ("mean_sea_level_pressure", None),
}
WAVELENGTH_LABEL = r"Wavelength [km]"
REL_ERROR_LABEL_SUFFIX = "rel. error"
# In-plot legend placement per plot kind (matplotlib loc)
LEGEND_PLACEMENTS = {"power": "lower left", "rel_error": "upper left"}


def variable_axis_labels(variable):
    """Return (power label, relative error label) of a spectra variable."""
    if variable.startswith("KE_"):
        name = KINETIC_ENERGY_LABEL + variable.removeprefix("KE_")
    else:
        style_variable, level = SPECTRA_VARIABLES[variable]
        name = plot_style.variable_label(style_variable, level, with_units=False)
    return name, f"{name} {REL_ERROR_LABEL_SUFFIX}"


def mean_over_dates(path):
    """Spectra of one run averaged over init dates (and members)."""
    df = pd.read_csv(path, usecols=SPECTRA_COLUMNS)
    df = df[df["wavenumber"] >= MIN_WAVENUMBER]
    return df.groupby(GROUP_COLUMNS)[["power_pred", "power_ref"]].mean()


def load_spectra(results_dir):
    """Return {model: {run: date-mean spectra indexed by (lead, variable, wavenumber)}}."""
    spectra = {}
    for model in plot_style.MODEL_ORDER:
        spectra[model] = {}
        for run in plot_style.seed_run_names(model):
            path = Path(results_dir) / plot_style.RESULT_FILENAME.format(kind="spectra", run=run)
            spectra[model][run] = mean_over_dates(path)
    return spectra


def reference_spectra(spectra):
    """ERA5 spectra, averaged over all runs (they are evaluated on the same dates)."""
    all_runs = pd.concat(
        [df["power_ref"] for runs in spectra.values() for df in runs.values()]
    )
    return all_runs.groupby(level=GROUP_COLUMNS).mean()


def seed_spectra(runs, lead, variable, wavenumbers):
    """Model power per seed on wavenumbers, as array (n_seeds, n_wavenumbers)."""
    per_seed = []
    for run, df in runs.items():
        key = (lead, variable)
        if key not in df.index.droplevel("wavenumber"):
            raise ValueError(f"{run}: no {variable} spectra at {lead}h")
        per_seed.append(df.loc[key, "power_pred"].reindex(wavenumbers).to_numpy())
    return np.array(per_seed)


def plot_reference(ax, x, y):
    ax.plot(
        x,
        y,
        color=plot_style.REFERENCE_COLOR,
        linestyle=plot_style.REFERENCE_LINESTYLE,
        linewidth=plot_style.REFERENCE_LINEWIDTH,
        label=plot_style.REFERENCE_LABEL,
        zorder=5,
    )


def finish_axes(ax, wavelengths, ylabel):
    ax.set_xscale("log")
    # Large scales on the left
    ax.set_xlim(wavelengths.max(), wavelengths.min())
    ax.set_xlabel(WAVELENGTH_LABEL)
    ax.set_ylabel(ylabel)


def plot_variable_lead(spectra, reference, variable, lead, out_dir):
    ref = reference.loc[(lead, variable)]
    wavenumbers = ref.index.to_numpy()
    wavelengths = 2.0 * np.pi * EARTH_RADIUS_KM / wavenumbers
    power_label, rel_error_label = variable_axis_labels(variable)
    out_name = f"{variable.lower()}_{lead}h"

    fig, ax = plt.subplots(figsize=(plot_style.FIG_WIDTH, plot_style.LINE_PLOT_HEIGHT))
    fig_err, ax_err = plt.subplots(figsize=(plot_style.FIG_WIDTH, plot_style.LINE_PLOT_HEIGHT))
    plot_reference(ax, wavelengths, ref.to_numpy())
    plot_reference(ax_err, wavelengths, np.zeros_like(wavelengths))
    for model in plot_style.MODEL_ORDER:
        per_seed = seed_spectra(spectra[model], lead, variable, wavenumbers)
        plot_style.plot_mean_std(ax, wavelengths, per_seed, model)
        plot_style.plot_mean_std(ax_err, wavelengths, (per_seed - ref.to_numpy()) / ref.to_numpy(), model)

    ax.set_yscale("log")
    finish_axes(ax, wavelengths, power_label)
    finish_axes(ax_err, wavelengths, rel_error_label)
    handles = plot_style.legend_handles(include_reference=True)
    plot_style.save_line_plot(
        fig, ax, out_dir / f"spectra_{out_name}", handles, LEGEND_PLACEMENTS["power"]
    )
    plot_style.save_line_plot(
        fig_err, ax_err, out_dir / f"spectra_rel_error_{out_name}", handles, LEGEND_PLACEMENTS["rel_error"]
    )


def run(results_dir, out_dir, leads=None, save_with_legend=False):
    plot_style.apply_style(save_with_legend=save_with_legend)
    spectra = load_spectra(results_dir)
    reference = reference_spectra(spectra)
    available_leads = sorted(reference.index.get_level_values("lead_hours").unique())
    for lead in leads or available_leads:
        if lead not in available_leads:
            raise ValueError(f"Lead {lead}h not in spectra results")
        variables = reference.loc[lead].index.get_level_values("variable").unique()
        for variable in sorted(variables):
            plot_variable_lead(
                spectra, reference, variable, lead, out_dir / SPECTRA_SUBDIR
            )
    plot_style.save_legend(
        out_dir / "legend_with_reference", plot_style.legend_handles(include_reference=True)
    )


def main():
    parser = plot_style.add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--leads", type=int, nargs="+", help="Lead times [h], default all")
    args = parser.parse_args()
    run(args.results_dir, args.out_dir, args.leads, args.legend)

if __name__ == "__main__":
    main()
