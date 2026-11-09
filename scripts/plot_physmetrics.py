"""
Paper plots of physmetrics time series vs. lead time (as in physmetrics-plot, without lapse
rate distributions and tables), shown as mean +- 1 std over seeds.

Writes to the output dir
    ts_<conserved quantity>.{png,pdf}   relative change [%] w.r.t. the first lead, mean over dates
    ts_<balance>_rmse.{png,pdf}         balance error RMSE, models + ERA5 at the same valid times
    ts_<balance>_rmse_diff.{png,pdf}    model RMSE - ERA5 RMSE at the same valid times

Usage:
    python scripts/plot_physmetrics.py [--results-dir DIR] [--out-dir DIR] [--legend]
"""

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_common
import plot_style

LEAD_LABEL = r"Lead time [d]"
PERCENT = 100.0

# Time series column -> y label; plotted as relative change w.r.t. the first forecast lead
CONSERVED_QUANTITIES = {
    "dry_mass_Eg": r"Dry air mass change [\%]",
    "water_mass_kg": r"Water mass change [\%]",
    "total_energy_J": r"Total energy change [\%]",
}
# Time series column (also metric_name in physics_evaluation) -> (RMSE label, unit)
BALANCE_RMSES = {
    "hydrostatic_rmse": (r"Hydrostatic RMSE", r"m$^2$/s$^2$"),
    "geostrophic_rmse": (r"Geostrophic RMSE", r"m/s"),
}
DIFF_LABEL_PREFIX = r"$\Delta$"
SERIES_KEYS = ["date", "ensemble_member"]
TIME_SERIES_COLUMNS = (
    ["date", "forecast_hour", "ensemble_member"]
    + list(CONSERVED_QUANTITIES)
    + list(BALANCE_RMSES)
)
PHYSICS_EVALUATION_COLUMNS = ["date", "lead_time_hours", "metric_name", "ref_value"]


def load_time_series(results_dir):
    """Return {model: {run: time series DataFrame}}."""
    time_series = {}
    for model in plot_style.MODEL_ORDER:
        runs = plot_common.read_run_csvs(
            results_dir, "time_series", model, usecols=TIME_SERIES_COLUMNS
        )
        for df in runs.values():
            df["date"] = pd.to_datetime(df["date"])
        time_series[model] = runs
    return time_series


def load_reference_rmse(results_dir):
    """ERA5 balance RMSEs indexed by valid time, with a column per metric.

    ERA5 RMSEs only exist in physics_evaluation, at its (sparser) lead times; all runs are
    combined to cover as many valid times as possible.
    """
    frames = []
    for model in plot_style.MODEL_ORDER:
        runs = plot_common.read_run_csvs(
            results_dir, "physics_evaluation", model, usecols=PHYSICS_EVALUATION_COLUMNS
        )
        frames.extend(runs.values())
    df = pd.concat(frames)
    df = df[df["metric_name"].isin(BALANCE_RMSES)].dropna(subset=["ref_value"])
    df["valid_time"] = pd.to_datetime(df["date"]) + pd.to_timedelta(
        df["lead_time_hours"], unit="h"
    )
    return df.pivot_table(index="valid_time", columns="metric_name", values="ref_value")


def relative_change(df, column):
    """Change [%] w.r.t. the value at the first forecast lead of each forecast."""
    df = df.dropna(subset=[column]).sort_values("forecast_hour")
    first = df.groupby(SERIES_KEYS)[column].transform("first")
    return (df[column] - first) / first * PERCENT


def per_seed_curves(runs, values_fn, forecast_hours, model, quantity):
    """Mean over dates of values_fn(df) per forecast hour, as array (n_seeds, n_hours)."""
    curves = []
    for df in runs.values():
        values = values_fn(df)
        curve = values.groupby(df.loc[values.index, "forecast_hour"]).mean()
        missing = set(forecast_hours) - set(curve.index)
        if missing:
            warnings.warn(f"{model}: {quantity} missing at forecast hours {sorted(missing)}")
        curves.append(curve.reindex(forecast_hours).to_numpy())
    n_seeds = len(plot_style.SEED_SUFFIXES)
    if len(curves) < n_seeds:
        warnings.warn(f"{model}: {quantity} available for {len(curves)}/{n_seeds} seeds")
    return np.array(curves)


def with_reference(df, reference_rmse, column):
    """ERA5 values of column at the valid time of each row (NaN where not evaluated)."""
    valid_time = df["date"] + pd.to_timedelta(df["forecast_hour"], unit="h")
    return pd.Series(
        reference_rmse[column].reindex(valid_time).to_numpy(), index=df.index
    )


def new_axes():
    return plt.subplots(figsize=(plot_style.FIG_WIDTH, plot_style.LINE_PLOT_HEIGHT))


def plot_reference(ax, lead_days, values):
    ax.plot(
        lead_days,
        values,
        color=plot_style.REFERENCE_COLOR,
        linestyle=plot_style.REFERENCE_LINESTYLE,
        linewidth=plot_style.REFERENCE_LINEWIDTH,
        label=plot_style.REFERENCE_LABEL,
        zorder=5,
    )


def plot_models(ax, curves, lead_days):
    for model, per_seed in curves.items():
        if per_seed.size == 0:
            continue
        plot_style.plot_mean_std(ax, lead_days, per_seed, model)


def finish_and_save(fig, ax, ylabel, out_path, show_legend, include_reference):
    ax.set_xlabel(LEAD_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, None)
    if show_legend:
        ax.legend(handles=plot_style.legend_handles(include_reference=include_reference))
    plot_style.save_figure(fig, out_path)


def plot_conserved_quantity(time_series, forecast_hours, column, ylabel, out_dir, show_legend):
    curves = {
        model: per_seed_curves(
            runs, lambda df: relative_change(df, column), forecast_hours, model, column
        )
        for model, runs in time_series.items()
    }
    fig, ax = new_axes()
    plot_models(ax, curves, forecast_hours / plot_common.HOURS_PER_DAY)
    finish_and_save(fig, ax, ylabel, out_dir / f"ts_{column}", show_legend, False)


def plot_balance_rmse(
    time_series, reference_rmse, forecast_hours, column, label, unit, out_dir, show_legend
):
    """Absolute RMSE with ERA5, and difference to ERA5, both only at valid times with ERA5."""

    def matched(df):
        reference = with_reference(df, reference_rmse, column)
        keep = reference.notna() & df[column].notna()
        return df.loc[keep, column], reference[keep]

    curves = {}
    diff_curves = {}
    reference_runs = {}
    for model, runs in time_series.items():
        curves[model] = per_seed_curves(
            runs, lambda df: matched(df)[0], forecast_hours, model, column
        )
        diff_curves[model] = per_seed_curves(
            runs, lambda df: matched(df)[0] - matched(df)[1], forecast_hours, model, column
        )
        reference_runs.update(runs)
    reference_curve = np.nanmean(
        per_seed_curves(
            reference_runs, lambda df: matched(df)[1], forecast_hours, "ERA5", column
        ),
        axis=0,
    )

    lead_days = forecast_hours / plot_common.HOURS_PER_DAY
    fig, ax = new_axes()
    plot_reference(ax, lead_days, reference_curve)
    plot_models(ax, curves, lead_days)
    finish_and_save(fig, ax, f"{label} [{unit}]", out_dir / f"ts_{column}", show_legend, True)

    fig, ax = new_axes()
    plot_reference(ax, lead_days, np.zeros_like(lead_days))
    plot_models(ax, diff_curves, lead_days)
    finish_and_save(
        fig,
        ax,
        f"{DIFF_LABEL_PREFIX} {label} [{unit}]",
        out_dir / f"ts_{column}_diff",
        show_legend,
        True,
    )


def run(results_dir, out_dir, show_legend=False):
    plot_style.apply_style()
    time_series = load_time_series(results_dir)
    reference_rmse = load_reference_rmse(results_dir)
    forecast_hours = np.array(
        sorted(
            set().union(
                *(df["forecast_hour"].unique() for runs in time_series.values() for df in runs.values())
            )
        )
    )
    for column, ylabel in CONSERVED_QUANTITIES.items():
        plot_conserved_quantity(time_series, forecast_hours, column, ylabel, out_dir, show_legend)
    for column, (label, unit) in BALANCE_RMSES.items():
        plot_balance_rmse(
            time_series, reference_rmse, forecast_hours, column, label, unit, out_dir, show_legend
        )
    plot_style.save_legend(out_dir / "legend", plot_style.legend_handles())
    plot_style.save_legend(
        out_dir / "legend_with_reference", plot_style.legend_handles(include_reference=True)
    )


def main():
    parser = plot_common.add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    run(args.results_dir, args.out_dir, args.legend)


if __name__ == "__main__":
    main()
