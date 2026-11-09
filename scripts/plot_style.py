"""
Shared plot style for the paper figures (ICML 2-column template).

Imported by the plotting scripts in this directory. An identical copy lives in
physmetrics/scripts/plot_style.py; keep the two in sync.

Line plots are sized so that two fit side-by-side within one column, with
text at roughly caption size. Full-page figures (e.g. map grids) are sized so
two fit side-by-side across the page.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# --- Sizes, in inches ---
COLUMN_WIDTH = 3.25
FIG_WIDTH = 1.6
LINE_PLOT_HEIGHT = 1.3
LEGEND_WIDTH = COLUMN_WIDTH

# --- Font sizes, in points (ICML captions are 9pt) ---
FONTSIZE = 8
AXIS_LABEL_FONTSIZE = 8
TICK_LABEL_FONTSIZE = 6.5
LEGEND_FONTSIZE = 8
COLORBAR_LABEL_FONTSIZE = 8

# --- Output ---
PLOT_FORMATS = ("png", "pdf")
DPI = 300
SAVE_PAD_INCHES = 0.01

# --- Lines and uncertainty bands ---
LINEWIDTH = 1.4
BAND_ALPHA = 0.35
# Standard deviation over seeds, delta degrees of freedom
STD_DDOF = 1
GRID_COLOR = "0.88"
GRID_LINEWIDTH = 0.6
REFERENCE_COLOR = "black"
REFERENCE_LINESTYLE = "-"
REFERENCE_LINEWIDTH = 1.0

# --- Models ---
# Run names of the 4 seeds of each model are <name>, <name>_1, <name>_2, <name>_3
SEED_SUFFIXES = ("", "_1", "_2", "_3")
MODEL_ORDER = ("reprod_train_12e", "mcrps_spectral", "spectral_sht_exp", "var_es_unweight")
MODEL_LABELS = {
    "reprod_train_12e": r"CRPS",
    "mcrps_spectral": r"$\spectralcrps$ (FT)",
    "spectral_sht_exp": r"$\spectralcrps$ (SHT)",
    "var_es_unweight": r"MSVS",
}
_SET2 = plt.get_cmap("Set2").colors
MODEL_STYLES = {
    "reprod_train_12e": {"color": _SET2[0], "linestyle": "-"},
    "mcrps_spectral": {"color": _SET2[1], "linestyle": "--"},
    "spectral_sht_exp": {"color": _SET2[2], "linestyle": "-."},
    "var_es_unweight": {"color": _SET2[3], "linestyle": ":"},
}
REFERENCE_LABEL = "ERA5"

# --- Variables ---
SURFACE_VARIABLES = (
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "sea_surface_temperature",
)
LEVEL_VARIABLES = (
    "geopotential",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "specific_humidity",
    "vertical_velocity",
)
PLOT_LEVELS = (200, 500, 700, 850)
SHORT_NAMES = {
    "2m_temperature": "2t",
    "10m_u_component_of_wind": "10u",
    "10m_v_component_of_wind": "10v",
    "mean_sea_level_pressure": "msl",
    "sea_surface_temperature": "sst",
    "geopotential": "z",
    "temperature": "t",
    "u_component_of_wind": "u",
    "v_component_of_wind": "v",
    "specific_humidity": "q",
    "vertical_velocity": "w",
}
VARIABLE_UNITS = {
    "2m_temperature": r"K",
    "10m_u_component_of_wind": r"m/s",
    "10m_v_component_of_wind": r"m/s",
    "mean_sea_level_pressure": r"Pa",
    "sea_surface_temperature": r"K",
    "geopotential": r"m$^2$/s$^2$",
    "temperature": r"K",
    "u_component_of_wind": r"m/s",
    "v_component_of_wind": r"m/s",
    "specific_humidity": r"kg/kg",
    "vertical_velocity": r"Pa/s",
}

# --- Text ---
# Macros from the paper's doc_macros.tex needed for labels
LATEX_PREAMBLE = r"""
\usepackage{amsmath}
\newcommand{\spectralcrps}{\lambda\text{-CRPS}}
"""


def apply_style():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            "font.serif": ["Times"],
            "font.sans-serif": ["Helvetica"],
            "font.monospace": ["Courier"],
            "text.latex.preamble": LATEX_PREAMBLE,
            "font.size": FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": TICK_LABEL_FONTSIZE,
            "ytick.labelsize": TICK_LABEL_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linewidth": GRID_LINEWIDTH,
            "axes.axisbelow": True,
            "lines.linewidth": LINEWIDTH,
            "savefig.dpi": DPI,
        }
    )


def variable_label(variable, level=None, with_units=True):
    """ECMWF-style short name in upright text, e.g. t850 or 2t, optionally with units."""
    label = SHORT_NAMES[variable] + ("" if level is None else str(level))
    if with_units:
        label += f" [{VARIABLE_UNITS[variable]}]"
    return label


def seed_run_names(model):
    return [model + suffix for suffix in SEED_SUFFIXES]


def plot_mean_std(ax, x, values_per_seed, model, label=None):
    """Plot mean over seeds (axis 0 of values_per_seed) with a +-1 std band."""
    values_per_seed = np.asarray(values_per_seed, dtype=float)
    mean = np.nanmean(values_per_seed, axis=0)
    style = MODEL_STYLES[model]
    (line,) = ax.plot(x, mean, label=label or MODEL_LABELS[model], **style)
    if values_per_seed.shape[0] > 1:
        std = np.nanstd(values_per_seed, axis=0, ddof=STD_DDOF)
        ax.fill_between(x, mean - std, mean + std, color=style["color"], alpha=BAND_ALPHA, linewidth=0)
    return line


def legend_handles(models=MODEL_ORDER, include_reference=False):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], label=MODEL_LABELS[m], **MODEL_STYLES[m]) for m in models]
    if include_reference:
        handles.insert(
            0,
            Line2D(
                [],
                [],
                label=REFERENCE_LABEL,
                color=REFERENCE_COLOR,
                linestyle=REFERENCE_LINESTYLE,
                linewidth=REFERENCE_LINEWIDTH,
            ),
        )
    return handles


def save_figure(fig, out_path_without_suffix):
    out_path_without_suffix = Path(out_path_without_suffix)
    out_path_without_suffix.parent.mkdir(parents=True, exist_ok=True)
    for fmt in PLOT_FORMATS:
        fig.savefig(
            out_path_without_suffix.with_suffix(f".{fmt}"),
            bbox_inches="tight",
            pad_inches=SAVE_PAD_INCHES,
            dpi=DPI,
        )
    plt.close(fig)


def save_legend(out_path_without_suffix, handles, ncol=None):
    """Render a legend on its own, in a separate file."""
    fig = plt.figure(figsize=(LEGEND_WIDTH, 0.3))
    fig.legend(handles=handles, loc="center", ncol=ncol or len(handles), frameon=False)
    save_figure(fig, out_path_without_suffix)
