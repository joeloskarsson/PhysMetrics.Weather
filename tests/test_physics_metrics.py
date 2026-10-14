"""Unit tests for physics_metrics.py using pytest and mock xarray datasets."""

import numpy as np
import pyshtools as pysh
import pytest
import xarray as xr

from physmetrics_weather.physics_metrics import (
    EARTH_RADIUS,
    _detect_ensemble_dim,
    _detect_level_dim,
    _integrate_column,
    compute_conservation_scalars,
    compute_drift_percentages,
    compute_drift_slope,
    compute_dry_air_mass,
    compute_geostrophic_imbalance,
    compute_hydrostatic_imbalance,
    compute_ke_spectrum,
    compute_lapse_rate_wasserstein,
    compute_pure_tcwv,
    compute_q_spectrum,
    compute_scalar_spectrum,
    compute_spectral_scores,
    compute_total_energy,
    compute_water_mass,
    compute_zonal_spectrum,
    derive_surface_pressure,
    get_grid_cell_area,
)


@pytest.fixture
def mock_grid_coords():
    """Create standard Driscoll-Healy latitude/longitude grid coordinates (32x64)."""
    lats = np.linspace(90, -90, 32)
    lons = np.linspace(0, 354.375, 64)
    levels = np.array([1000.0, 850.0, 500.0, 250.0, 100.0])
    return lats, lons, levels


@pytest.fixture
def mock_deterministic_ds(mock_grid_coords):
    """Create a deterministic mock xarray dataset without ensemble dimension."""
    lats, lons, levels = mock_grid_coords
    nlat, nlon, nlev = len(lats), len(lons), len(levels)

    np.random.seed(42)
    shape_3d = (nlev, nlat, nlon)
    shape_2d = (nlat, nlon)

    temp = 280.0 + np.random.randn(*shape_3d) * 5.0
    q = np.abs(np.random.randn(*shape_3d) * 0.005)
    u = np.random.randn(*shape_3d) * 10.0
    v = np.random.randn(*shape_3d) * 10.0
    phi = (
        np.broadcast_to(levels[:, None, None] * 100.0, shape_3d)
        + np.random.randn(*shape_3d) * 10.0
    )
    msl = 101325.0 + np.random.randn(*shape_2d) * 500.0
    sp = 100000.0 + np.random.randn(*shape_2d) * 1000.0

    ds = xr.Dataset(
        data_vars={
            "temperature": (["level", "latitude", "longitude"], temp),
            "specific_humidity": (["level", "latitude", "longitude"], q),
            "u_component_of_wind": (["level", "latitude", "longitude"], u),
            "v_component_of_wind": (["level", "latitude", "longitude"], v),
            "geopotential": (["level", "latitude", "longitude"], phi),
            "mean_sea_level_pressure": (["latitude", "longitude"], msl),
            "surface_pressure": (["latitude", "longitude"], sp),
        },
        coords={
            "latitude": lats,
            "longitude": lons,
            "level": levels,
        },
    )
    return ds


@pytest.fixture
def mock_ensemble_ds(mock_deterministic_ds):
    """Create an ensemble mock xarray dataset with an 'ens' dimension (3 members)."""
    ds = mock_deterministic_ds
    members = [0, 1, 2]

    data_vars = {}
    for var_name, da in ds.data_vars.items():
        shape = (len(members),) + da.shape
        var_data = np.stack([da.values + i * 0.1 for i in members], axis=0)
        dims = ["ens"] + list(da.dims)
        data_vars[var_name] = (dims, var_data)

    coords = dict(ds.coords)
    coords["ens"] = members

    return xr.Dataset(data_vars=data_vars, coords=coords)


@pytest.fixture
def mock_static_ds(mock_grid_coords):
    """Create a static dataset containing surface geopotential."""
    lats, lons, _ = mock_grid_coords
    shape_2d = (len(lats), len(lons))
    z_sfc = np.clip(np.random.randn(*shape_2d) * 500.0, 0, None)

    return xr.Dataset(
        data_vars={"geopotential_at_surface": (["latitude", "longitude"], z_sfc)},
        coords={"latitude": lats, "longitude": lons},
    )


# ============================================================================
# Unit Tests
# ============================================================================

def test_detect_ensemble_dim(mock_deterministic_ds, mock_ensemble_ds):
    """Test auto-detection of ensemble dimension."""
    assert _detect_ensemble_dim(mock_deterministic_ds) is None
    assert _detect_ensemble_dim(mock_ensemble_ds) == "ens"


def test_detect_level_dim(mock_deterministic_ds):
    """Test auto-detection of pressure level dimension."""
    assert _detect_level_dim(mock_deterministic_ds) == "level"


def test_get_grid_cell_area(mock_deterministic_ds):
    """Test computation of grid cell areas."""
    area = get_grid_cell_area(mock_deterministic_ds)
    assert isinstance(area, xr.DataArray)
    assert area.shape == (32, 64)
    assert (area.values > 0).all()
    # Total surface area of Earth ≈ 5.1e14 m²
    assert np.isclose(float(area.sum()), 4 * np.pi * (6.371e6) ** 2, rtol=1e-2)


def test_derive_surface_pressure(mock_deterministic_ds, mock_static_ds):
    """Test surface pressure derivation from MSL and surface geopotential."""
    sp = derive_surface_pressure(mock_deterministic_ds, mock_static_ds)
    assert isinstance(sp, xr.DataArray)
    assert sp.shape == (32, 64)
    assert (sp.values > 50000.0).all()


def test_integrate_column(mock_grid_coords):
    """Test 3D column integration helper."""
    lats, lons, levels = mock_grid_coords
    nlat, nlon, nlev = len(lats), len(lons), len(levels)
    field_3d = np.ones((nlev, nlat, nlon))
    ps_2d = np.full((nlat, nlon), 100000.0)

    integrated = _integrate_column(field_3d, levels, ps_2d)
    assert integrated.shape == (nlat, nlon)
    assert (integrated > 0).all()


def test_compute_dry_air_mass_deterministic(mock_deterministic_ds):
    """Test dry air mass calculation for deterministic dataset."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]

    dry_mass = compute_dry_air_mass(mock_deterministic_ds, ps, area)
    assert isinstance(dry_mass, float)
    assert 4.0 < dry_mass < 6.0  # Earth dry air mass ≈ 5.1 Eg


def test_compute_dry_air_mass_ensemble(mock_ensemble_ds):
    """Test dry air mass calculation for ensemble dataset."""
    area = get_grid_cell_area(mock_ensemble_ds)
    ps = mock_ensemble_ds["surface_pressure"]

    dry_mass = compute_dry_air_mass(mock_ensemble_ds, ps, area)
    assert isinstance(dry_mass, dict)
    assert len(dry_mass) == 3
    for m, val in dry_mass.items():
        assert isinstance(val, float)
        assert 4.0 < val < 6.0


def test_compute_water_mass(mock_deterministic_ds, mock_ensemble_ds):
    """Test global water mass calculation for deterministic and ensemble datasets."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]

    water_det = compute_water_mass(mock_deterministic_ds, ps, area)
    assert isinstance(water_det, float)
    assert water_det > 0

    water_ens = compute_water_mass(mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area)
    assert isinstance(water_ens, dict)
    assert len(water_ens) == 3


def test_compute_total_energy(mock_deterministic_ds, mock_static_ds, mock_ensemble_ds):
    """Test total energy calculation for deterministic and ensemble datasets."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]
    z_sfc = mock_static_ds["geopotential_at_surface"]

    te_det = compute_total_energy(mock_deterministic_ds, ps, area, z_sfc=z_sfc)
    assert isinstance(te_det, float)
    assert te_det > 0

    te_ens = compute_total_energy(
        mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area, z_sfc=z_sfc
    )
    assert isinstance(te_ens, dict)
    assert len(te_ens) == 3


def test_compute_ke_spectrum(mock_deterministic_ds, mock_ensemble_ds):
    """Test KE spectrum calculation."""
    res_det = compute_ke_spectrum(mock_deterministic_ds, level=500.0)
    assert isinstance(res_det, tuple)
    k, e = res_det
    assert len(k) == len(e)

    res_ens = compute_ke_spectrum(mock_ensemble_ds, level=500.0)
    assert isinstance(res_ens, dict)
    assert len(res_ens) == 3


def test_compute_q_spectrum(mock_deterministic_ds, mock_ensemble_ds):
    """Test Q spectrum calculation."""
    res_det = compute_q_spectrum(mock_deterministic_ds, level=500.0)
    assert isinstance(res_det, tuple)

    res_ens = compute_q_spectrum(mock_ensemble_ds, level=500.0)
    assert isinstance(res_ens, dict)
    assert len(res_ens) == 3


def test_compute_spectral_scores():
    """Test spectral divergence and residual calculations."""
    e_true = np.exp(-np.linspace(0, 5, 20))
    e_pred = e_true * 1.1

    s_div, s_res = compute_spectral_scores(e_pred, e_true)
    assert isinstance(s_div, float)
    assert isinstance(s_res, float)
    assert s_div >= 0
    assert s_res >= 0


def test_compute_hydrostatic_imbalance(mock_deterministic_ds, mock_ensemble_ds):
    """Test hydrostatic balance RMSE calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)

    hydro_det = compute_hydrostatic_imbalance(mock_deterministic_ds, area)
    assert isinstance(hydro_det, float)
    assert hydro_det >= 0

    hydro_ens = compute_hydrostatic_imbalance(mock_ensemble_ds, area)
    assert isinstance(hydro_ens, dict)
    assert len(hydro_ens) == 3


def test_compute_geostrophic_imbalance(mock_deterministic_ds, mock_ensemble_ds):
    """Test geostrophic balance RMSE calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)

    geo_det = compute_geostrophic_imbalance(mock_deterministic_ds, area)
    assert isinstance(geo_det, float)

    geo_ens = compute_geostrophic_imbalance(mock_ensemble_ds, area)
    assert isinstance(geo_ens, dict)
    assert len(geo_ens) == 3


def test_compute_lapse_rate_wasserstein(mock_deterministic_ds):
    """Test lapse rate Wasserstein distance calculation."""
    area = get_grid_cell_area(mock_deterministic_ds)
    res = compute_lapse_rate_wasserstein(mock_deterministic_ds, mock_deterministic_ds, area)
    assert isinstance(res, dict)
    assert "lapse_rate_w1_tropics" in res


def test_compute_drift_slope_and_percentages():
    """Test drift slope and drift percentage calculations."""
    hours = np.array([12, 24, 36, 48])
    values = np.array([100.0, 101.0, 102.0, 103.0])

    slope = compute_drift_slope(hours, values)
    assert np.isclose(slope, 2.0)  # 1 unit per 12h = 2 units per 24h (day)

    drift = compute_drift_percentages(
        hours, values, values, values,
        hours, values, values
    )
    assert isinstance(drift, dict)
    assert "dry_mass_drift_pct_per_day" in drift


def test_compute_conservation_scalars(mock_deterministic_ds, mock_static_ds, mock_ensemble_ds):
    """Test conservation scalars wrapper function."""
    area = get_grid_cell_area(mock_deterministic_ds)
    ps = mock_deterministic_ds["surface_pressure"]
    z_sfc = mock_static_ds["geopotential_at_surface"]

    scalars_det = compute_conservation_scalars(mock_deterministic_ds, ps, area, z_sfc=z_sfc)
    assert isinstance(scalars_det, tuple)
    assert len(scalars_det) == 3

    scalars_ens = compute_conservation_scalars(
        mock_ensemble_ds, mock_ensemble_ds["surface_pressure"], area, z_sfc=z_sfc
    )
    assert isinstance(scalars_ens, dict)
    assert len(scalars_ens) == 3


def test_compute_pure_tcwv(mock_deterministic_ds):
    """Test fixed-level pure TCWV integration."""
    tcwv_pure = compute_pure_tcwv(mock_deterministic_ds)
    assert isinstance(tcwv_pure, xr.DataArray)
    assert tcwv_pure.shape == (32, 64)


def test_compute_scalar_spectrum(mock_deterministic_ds):
    """Test spherical harmonic spectrum calculation for arbitrary scalar fields."""
    k, spec = compute_scalar_spectrum(mock_deterministic_ds, var_name="temperature", level=850.0)
    assert isinstance(k, np.ndarray)
    assert isinstance(spec, np.ndarray)
    assert len(k) == len(spec)
    assert len(k) > 0


def _band_limited_grid(lats, lons, lmax, seed=0):
    """Random 4pi-normalized SH field band-limited to lmax, its power per degree, and grid values."""
    clm = pysh.SHCoeffs.from_random(np.ones(lmax + 1), lmax=lmax, normalization="4pi", seed=seed)
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    return clm.spectrum(), clm.expand(lat=lat2d, lon=lon2d)


def _scalar_ds(field, lats, lons, name="temperature"):
    return xr.Dataset({name: (["latitude", "longitude"], field)}, coords={"latitude": lats, "longitude": lons})


def test_ducc0_spectrum_recovers_band_limited_field(mock_grid_coords):
    """ducc0 CC analysis is exact up to lmax = nlat - 2 on a grid with both poles."""
    lats, lons, _ = mock_grid_coords
    lmax = len(lats) - 2
    power_true, field = _band_limited_grid(lats, lons, lmax)

    k, power = compute_scalar_spectrum(_scalar_ds(field, lats, lons), "temperature", sh_backend="ducc0")
    np.testing.assert_array_equal(k, np.arange(lmax + 1))
    np.testing.assert_allclose(power, power_true, rtol=1e-10, atol=1e-12 * power_true.max())


def test_ducc0_matches_dh_and_is_orientation_independent(mock_grid_coords):
    """For a field band-limited below the DH limit both backends agree; S->N input gives the same result."""
    _, lons, _ = mock_grid_coords
    # 33 rows with both poles: DH drops the south pole row (32 x 64, lmax 15), CC uses all (lmax 31)
    lats = np.linspace(90, -90, 33)
    _, field = _band_limited_grid(lats, lons, lmax=10, seed=1)
    ds = _scalar_ds(field, lats, lons)

    k_dh, power_dh = compute_scalar_spectrum(ds, "temperature", sh_backend="dh")
    k_ducc, power_ducc = compute_scalar_spectrum(ds, "temperature", sh_backend="ducc0")
    assert len(k_ducc) == len(lats) - 1 > len(k_dh)
    np.testing.assert_allclose(power_ducc[: len(k_dh)], power_dh, rtol=1e-8, atol=1e-12)

    _, power_flipped = compute_scalar_spectrum(ds.isel(latitude=slice(None, None, -1)), "temperature", sh_backend="ducc0")
    np.testing.assert_allclose(power_flipped, power_ducc, rtol=1e-12)


def test_ducc0_rejects_grid_without_poles(mock_grid_coords):
    """The ducc0 backend only supports equiangular grids including both poles."""
    _, lons, _ = mock_grid_coords
    lats = np.linspace(87.1875, -87.1875, 32)
    ds = _scalar_ds(np.zeros((len(lats), len(lons))), lats, lons)
    with pytest.raises(ValueError, match="both poles"):
        compute_scalar_spectrum(ds, "temperature", sh_backend="ducc0")


@pytest.mark.parametrize("sh_backend", ["dh", "ducc0"])
def test_ke_spectrum_is_mean_of_component_spectra(mock_deterministic_ds, mock_ensemble_ds, sh_backend):
    """KE spectrum equals (S_u + S_v) / 2 for both backends, and handles ensembles."""
    _, e = compute_ke_spectrum(mock_deterministic_ds, level=500.0, sh_backend=sh_backend)
    _, s_u = compute_scalar_spectrum(mock_deterministic_ds, "u_component_of_wind", level=500.0, sh_backend=sh_backend)
    _, s_v = compute_scalar_spectrum(mock_deterministic_ds, "v_component_of_wind", level=500.0, sh_backend=sh_backend)
    np.testing.assert_allclose(e, 0.5 * (s_u + s_v))

    res_ens = compute_ke_spectrum(mock_ensemble_ds, level=500.0, sh_backend=sh_backend)
    assert isinstance(res_ens, dict) and len(res_ens) == 3


def test_zonal_spectrum_single_wave(mock_grid_coords):
    """A pure zonal wave gives a single peak matching the WB2 ZonalEnergySpectrum definition."""
    lats, lons, levels = mock_grid_coords
    k0, amplitude = 3, 2.0
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    field = amplitude * np.cos(k0 * np.deg2rad(lon2d)) * np.cos(np.deg2rad(lat2d))
    # Outside the band the field must not contribute
    field[np.abs(lat2d) > 55] = 1e6

    k, power, wavelength_km = compute_zonal_spectrum(_scalar_ds(field, lats, lons), "temperature", lat_band=(25.0, 55.0))

    in_band = (np.abs(lats) >= 25) & (np.abs(lats) <= 55)
    cos_lat = np.cos(np.deg2rad(lats[in_band]))
    weights = cos_lat / cos_lat.sum()
    # rfft(norm="forward") of A cos(k0 λ) is A/2 at k0, doubled power: A²/2, times circumference
    expected = np.sum(weights * (amplitude * cos_lat) ** 2 / 2 * 2 * np.pi * EARTH_RADIUS * cos_lat)
    assert np.argmax(power) == k0
    np.testing.assert_allclose(power[k0], expected, rtol=1e-10)
    np.testing.assert_allclose(np.delete(power, k0), 0.0, atol=1e-10 * expected)
    assert np.isinf(wavelength_km[0])
    np.testing.assert_allclose(wavelength_km[k0], 2 * np.pi * EARTH_RADIUS * np.sum(weights * cos_lat) / k0 / 1000)


def test_zonal_spectrum_uses_both_hemispheres_and_ke(mock_grid_coords, mock_deterministic_ds, mock_ensemble_ds):
    """Rings from both hemispheres enter the zonal spectrum; KE is (S_u + S_v) / 2; ensembles give dicts."""
    lats, lons, _ = mock_grid_coords
    lat2d, lon2d = np.meshgrid(lats, lons, indexing="ij")
    field = np.where(lat2d < 0, np.cos(2 * np.deg2rad(lon2d)), 0.0)
    _, power, _ = compute_zonal_spectrum(_scalar_ds(field, lats, lons), "temperature")
    assert power[2] > 0

    _, e, _ = compute_zonal_spectrum(mock_deterministic_ds, "KE", level=500.0)
    _, s_u, _ = compute_zonal_spectrum(mock_deterministic_ds, "u_component_of_wind", level=500.0)
    _, s_v, _ = compute_zonal_spectrum(mock_deterministic_ds, "v_component_of_wind", level=500.0)
    np.testing.assert_allclose(e, 0.5 * (s_u + s_v))

    res_ens = compute_zonal_spectrum(mock_ensemble_ds, "Q", level=500.0)
    assert isinstance(res_ens, dict) and len(res_ens) == 3
