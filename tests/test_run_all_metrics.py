"""Unit tests for run_all_metrics.py CLI runner and helper functions."""

import argparse
import numpy as np
import pytest
import xarray as xr

from physmetrics_weather.run_all_metrics import (
    _align_ref_to_model,
    _grids_match,
    _parse_lead_times,
    _parse_spectra_spec,
    _resolve_dates,
    _spectrum_label,
)


def test_parse_lead_times():
    """Test parsing of lead-time specification strings."""
    parsed = _parse_lead_times("12h,5d,10d")
    assert len(parsed) == 3
    assert parsed[0] == ("12h", np.timedelta64(12, "h"))
    assert parsed[1] == ("5d", np.timedelta64(120, "h"))
    assert parsed[2] == ("10d", np.timedelta64(240, "h"))

    with pytest.raises(ValueError):
        _parse_lead_times("invalid_time")


def test_resolve_dates():
    """Test resolution of date arguments."""
    args = argparse.Namespace(dates=["2022-01-01", "2022-01-02"], month=None, year=2022)
    resolved = _resolve_dates(args)
    assert resolved == ["2022-01-01T00:00:00", "2022-01-02T00:00:00"]

    args_month = argparse.Namespace(dates=None, month="2022-02", year=2022)
    resolved_month = _resolve_dates(args_month)
    assert len(resolved_month) == 28


def test_grids_match():
    """Test dataset grid matching comparison."""
    lats = np.linspace(90, -90, 32)
    lons = np.linspace(0, 350, 64)

    ds_a = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    ds_b = xr.Dataset(coords={"latitude": lats, "longitude": lons})
    ds_c = xr.Dataset(coords={"latitude": np.linspace(90, -90, 16), "longitude": lons})

    assert _grids_match(ds_a, ds_b) is True
    assert _grids_match(ds_a, ds_c) is False


def test_align_ref_to_model():
    """Test grid alignment between reference and model datasets."""
    lats_ref = np.linspace(90, -90, 33)
    lats_mod = np.linspace(90, -90, 32)
    lons = np.linspace(0, 350, 64)

    ds_ref = xr.Dataset(
        data_vars={"temp": (["latitude", "longitude"], np.zeros((33, 64)))},
        coords={"latitude": lats_ref, "longitude": lons},
    )
    ds_mod = xr.Dataset(
        data_vars={"temp": (["latitude", "longitude"], np.zeros((32, 64)))},
        coords={"latitude": lats_mod, "longitude": lons},
    )

    aligned = _align_ref_to_model(ds_ref, ds_mod)
    assert aligned.sizes["latitude"] == 32


def test_parse_spectra_spec():
    """Bare tokens have no level; explicit levels are parsed as floats."""
    assert _parse_spectra_spec("KE:850,t2m, Z") == [("KE", 850.0), ("T2M", None), ("Z", None)]


def test_spectrum_label():
    """3D variables are labelled with their level (default 500 hPa), surface variables without."""
    shape = (2, 4, 8)
    coords = {"level": [500.0, 850.0], "latitude": np.linspace(90, -90, 4), "longitude": np.arange(8) * 45.0}
    ds = xr.Dataset(
        {
            "u_component_of_wind": (["level", "latitude", "longitude"], np.zeros(shape)),
            "temperature": (["level", "latitude", "longitude"], np.zeros(shape)),
            "mean_sea_level_pressure": (["latitude", "longitude"], np.zeros(shape[1:])),
        },
        coords=coords,
    )
    assert _spectrum_label(ds, "KE", None) == ("KE_500", 500.0)
    assert _spectrum_label(ds, "t", 850.0) == ("T_850", 850.0)
    assert _spectrum_label(ds, "MSL", None)[0] == "MSL"
    assert _spectrum_label(ds, "MSL", 850.0)[0] == "MSL"
    with pytest.raises(KeyError):
        _spectrum_label(ds, "Q", None)
