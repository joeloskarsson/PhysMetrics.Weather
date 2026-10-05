"""Unit tests for plot_neurips_metrics.py visualization utilities."""

import matplotlib
import numpy as np
import pandas as pd

from physmetrics_weather.plot_neurips_metrics import (
    PhysicsPlotter,
    PlotterConfig,
    get_model_baselines,
    infer_reference_label,
    load_summaries,
    model_name_from_csv,
    pretty_region_name,
)


def test_infer_reference_label(tmp_path):
    """Test reference label inference from directory path name."""
    era5_dir = tmp_path / "results_era5"
    ifs_dir = tmp_path / "results_ifs"

    assert infer_reference_label(era5_dir) == "ERA5"
    assert infer_reference_label(ifs_dir) == "IFS"


def test_pretty_region_name():
    """Test region name formatting."""
    assert pretty_region_name("tropics") == "Tropics"
    assert pretty_region_name("nh_mid") == "Nor. HS"
    assert pretty_region_name("custom_region") == "Custom Region"


def test_load_summaries_and_baselines(tmp_path):
    """Test loading model summary CSV files and calculating baselines."""
    df_aurora = pd.DataFrame({
        "date": ["2022-01-01"],
        "lead_time_hours": [12],
        "metric_name": ["hydrostatic_rmse"],
        "model_value": [10.5],
        "ref_value": [8.0],
        "ensemble_member": [0],
    })

    file_path = tmp_path / "physics_evaluation_aurora_2022.csv"
    df_aurora.to_csv(file_path, index=False)

    summaries = load_summaries(tmp_path)
    assert "aurora" in summaries
    assert len(summaries["aurora"]) == 1

    baselines = get_model_baselines(summaries, "hydrostatic_rmse")
    assert "aurora" in baselines
    assert baselines["aurora"] == 8.0


def test_model_name_from_csv():
    """Test model name parsing from results CSV filenames, including underscores."""
    assert model_name_from_csv("spectra_gc_plus_ucast_2020.csv", "spectra_") == "gc_plus_ucast"
    assert model_name_from_csv("spectra_pangu_2020.csv", "spectra_") == "pangu"
    assert model_name_from_csv("physics_evaluation_ifs_ens_2020.csv", "physics_evaluation_") == "ifs_ens"
    assert model_name_from_csv("spectra_pangu_unknown.csv", "spectra_") == "pangu"


def test_plot_spectra(tmp_path):
    """Test that spectra are plotted per variable and lead for models outside MODELS."""
    matplotlib.use("Agg")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    wavenumbers = np.arange(0, 20)
    for model in ["pangu", "gc_plus_ucast"]:
        rows = [
            {
                "date": "2020-01-01",
                "lead_hours": lead,
                "variable": variable,
                "wavenumber": k,
                "power_pred": 1.0 / (k + 1) ** 3,
                "power_ref": 1.1 / (k + 1) ** 3,
                "ensemble_member": 0,
            }
            for lead in [12, 120]
            for variable in ["KE_500", "Q_500"]
            for k in wavenumbers
        ]
        pd.DataFrame(rows).to_csv(results_dir / f"spectra_{model}_2020.csv", index=False)

    outdir = tmp_path / "plots"
    plotter = PhysicsPlotter(PlotterConfig(results_dir=results_dir, outdir=outdir, dpi=50))
    generated = plotter.plot_spectra(leads=[12, 120])

    expected = {
        outdir / f"spectra_{v}_{lt}h.png" for v in ["ke_500", "q_500"] for lt in [12, 120]
    }
    assert set(generated) == expected
    assert all(p.exists() for p in expected)
