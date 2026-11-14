import numpy as np
import pandas as pd
import pytest
import xarray as xr

from flood.ingest import extract_thresholds, grib_points


def dataset(variable, timestamps, values):
    return xr.Dataset(
        {
            variable: (
                ("time", "latitude", "longitude"),
                np.asarray(values, dtype=float).reshape(-1, 1, 1),
            )
        },
        coords={
            "time": timestamps,
            "valid_time": ("time", timestamps),
            "latitude": [30.525],
            "longitude": [70.875],
        },
    )


def test_discharge_uses_previous_calendar_day(monkeypatch):
    data = dataset("avg_dis", pd.date_range("2020-01-02", periods=3), [100, 200, 300])
    monkeypatch.setattr("flood.ingest.cfgrib.open_datasets", lambda *args, **kwargs: [data])
    frame = grib_points(
        "unused.grib",
        {"dis24": "discharge"},
        [{"id": "reach", "latitude": 30.52, "longitude": 70.85}],
        0.1,
    )
    assert frame.date.tolist() == list(pd.date_range("2020-01-01", periods=3))
    assert frame.discharge.tolist() == [100, 200, 300]


def test_single_day_discharge_retains_valid_time(monkeypatch):
    data = dataset("dis24", pd.date_range("2023-01-01", periods=1), [500])
    monkeypatch.setattr("flood.ingest.cfgrib.open_datasets", lambda *args, **kwargs: [data])
    frame = grib_points(
        "unused.grib",
        {"dis24": "discharge"},
        [{"id": "reach", "latitude": 30.52, "longitude": 70.85}],
        0.1,
    )
    assert frame.date.tolist() == [pd.Timestamp("2022-12-31")]
    assert frame.discharge.tolist() == [500]


def test_era5_requires_full_days(monkeypatch):
    data = dataset("t2m", pd.date_range("2020-01-01", periods=23, freq="h"), np.ones(23) * 290)
    monkeypatch.setattr("flood.ingest.cfgrib.open_datasets", lambda *args, **kwargs: [data])
    with pytest.raises(ValueError, match="24 hourly"):
        grib_points(
            "unused.grib",
            {"t2m": "temperature"},
            [{"id": "reach", "latitude": 30.52, "longitude": 70.85}],
            0.1,
        )


def test_era5_daily_mean(monkeypatch):
    data = dataset("t2m", pd.date_range("2020-01-01", periods=48, freq="h"), np.arange(48) + 270)
    monkeypatch.setattr("flood.ingest.cfgrib.open_datasets", lambda *args, **kwargs: [data])
    frame = grib_points(
        "unused.grib",
        {"t2m": "temperature"},
        [{"id": "reach", "latitude": 30.52, "longitude": 70.85}],
        0.1,
    )
    np.testing.assert_allclose(frame.temperature, [281.5, 305.5])


def test_threshold_extraction_preserves_provenance(tmp_path, cfg):
    grid = xr.Dataset(
        {"rl_2.0": (("lat", "lon"), np.array([[1234.0]]))},
        coords={"lat": [30.525], "lon": [70.875]},
    )
    target = tmp_path / "threshold.nc"
    grid.to_netcdf(target)
    options = cfg["data"].copy()
    options["reaches"] = [options["reaches"][0]]
    frame = extract_thresholds(target, options)
    assert frame.threshold.tolist() == [1234.0]
    assert frame.latitude.tolist() == [30.525]
    assert frame.source_url.tolist() == [options["threshold_url"]]
