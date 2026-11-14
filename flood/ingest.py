import hashlib
import json
from pathlib import Path

import cdsapi
import cfgrib
import numpy as np
import pandas as pd
import requests
import xarray as xr

from flood.common import config, digest, required, write_json
from flood.rainfall import chirps_points


def download(url, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        with temporary.open("wb") as stream:
            for block in response.iter_content(1024 * 1024):
                stream.write(block)
    temporary.replace(target)


def retrieve(client, dataset, request, target):
    if target.exists():
        return
    temporary = target.with_suffix(".partial")
    job_path = target.with_suffix(".job.json")
    if job_path.exists():
        remote = client.client.get_remote(json.loads(job_path.read_text())["id"])
    else:
        remote = client.client.submit(dataset, request)
        write_json(job_path, {"id": remote.request_id, "dataset": dataset, "request": request})
    remote.download(str(temporary))
    temporary.replace(target)


def request_path(raw, prefix, month, request):
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
    return raw / f"{prefix}-{month}-{signature}.grib"


def source_path(raw, prefix, period, request, cache_areas):
    preferred = request_path(raw, prefix, period, request)
    if preferred.exists() or preferred.with_suffix(".job.json").exists():
        return preferred
    for area in cache_areas:
        candidate = request_path(raw, prefix, period, {**request, "area": area})
        if candidate.exists():
            return candidate
    return preferred


def extract_thresholds(path, data):
    rows = []
    if data["glofas_version"] != "version_4_0" or data["threshold_return_period"] != 2:
        raise ValueError(
            "The configured auxiliary file requires GloFAS v4 and a 2-year return period"
        )
    with xr.open_dataset(path) as dataset:
        field = dataset[data["threshold_variable"]]
        for reach in data["reaches"]:
            point = field.sel(lat=reach["latitude"], lon=reach["longitude"], method="nearest")
            rows.append(
                {
                    "reach_id": reach["id"],
                    "threshold": float(point),
                    "return_period": data["threshold_return_period"],
                    "glofas_version": data["glofas_version"],
                    "latitude": float(point.lat),
                    "longitude": float(point.lon),
                    "source_url": data["threshold_url"],
                }
            )
    return pd.DataFrame(rows)


def grib_points(path, variables, reaches, maximum_distance):
    datasets = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    result = []
    try:
        for reach in reaches:
            values = []
            for variable, output in variables.items():
                aliases = ("dis24", "avg_dis") if variable == "dis24" else (variable,)
                matches = [
                    (dataset, alias)
                    for dataset in datasets
                    for alias in aliases
                    if alias in dataset
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one GRIB field {variable}: {path}")
                dataset, alias = matches[0]
                field = dataset[alias]
                point = field.sel(
                    latitude=reach["latitude"], longitude=reach["longitude"], method="nearest"
                )
                if (
                    max(
                        abs(float(point.latitude) - reach["latitude"]),
                        abs(float(point.longitude) - reach["longitude"]),
                    )
                    > maximum_distance
                ):
                    raise ValueError(f"Grid point too far from {reach['id']}")
                if "step" in point.dims and point.sizes["step"] > 1:
                    raise ValueError("Forecast-step GRIB is not historical daily data")
                time = np.asarray(point.valid_time.values).reshape(-1)
                series = pd.Series(np.asarray(point.values).reshape(-1), index=pd.to_datetime(time))
                if series.index.duplicated().any():
                    raise ValueError("Duplicate GRIB timestamps")
                if output != "discharge":
                    if not (series.resample("D").count() == 24).all():
                        raise ValueError("ERA5 daily aggregation requires 24 hourly values")
                    series = series.resample("D").mean()
                else:
                    if series.index.normalize().duplicated().any():
                        raise ValueError("Duplicate daily discharge")
                    if not (series.index == series.index.normalize()).all():
                        raise ValueError("Expected midnight end-of-period discharge timestamps")
                    series.index = series.index.normalize() - pd.Timedelta(days=1)
                values.append(series.rename(output))
            frame = pd.concat(values, axis=1)
            frame.index.name = "date"
            frame = frame.reset_index()
            frame["reach_id"] = reach["id"]
            result.append(frame)
    finally:
        for dataset in datasets:
            dataset.close()
    return pd.concat(result, ignore_index=True)


def ingest():
    cfg = config()
    data = cfg["data"]
    required("CDSAPI_KEY")
    required("EWDS_API_KEY")
    raw = Path("data/raw")
    raw.mkdir(parents=True, exist_ok=True)
    downloads = Path("data/downloads")
    downloads.mkdir(parents=True, exist_ok=True)
    cds = cdsapi.Client(
        url="https://cds.climate.copernicus.eu/api",
        key=required("CDSAPI_KEY"),
        timeout=180,
        retry_max=5,
        sleep_max=30,
    )
    ewds = cdsapi.Client(
        url="https://ewds.climate.copernicus.eu/api",
        key=required("EWDS_API_KEY"),
        timeout=180,
        retry_max=5,
        sleep_max=30,
    )
    dates = pd.date_range(data["start"], data["end"], freq="D")
    discharge, weather = [], []
    discharge_dates = dates
    for month in discharge_dates.to_period("Y").unique():
        selected = discharge_dates[discharge_dates.to_period("Y") == month]
        days = sorted(set(selected.strftime("%d")))
        shared = {
            "year": [str(month.year)],
            "month": sorted(set(selected.strftime("%m"))),
            "day": days,
            "area": data["area"],
            "data_format": "grib",
            "download_format": "unarchived",
        }
        request = {
            **shared,
            "system_version": [data["glofas_version"]],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["average_river_discharge_in_the_last_24_hours"],
            "timespan": "time_mean",
        }
        glofas = source_path(downloads, "glofas", month, request, data.get("cache_areas", []))
        retrieve(ewds, "cems-glofas-historical", request, glofas)
        discharge.append(
            grib_points(
                glofas, {"dis24": "discharge"}, data["reaches"], data["maximum_grid_distance"]
            )
        )
    for month in dates.to_period("Y").unique():
        selected = dates[dates.to_period("Y") == month]
        days = sorted(set(selected.strftime("%d")))
        shared = {
            "year": [str(month.year)],
            "month": sorted(set(selected.strftime("%m"))),
            "day": days,
            "area": data["area"],
            "data_format": "grib",
            "download_format": "unarchived",
        }
        request = {
            **shared,
            "product_type": ["reanalysis"],
            "variable": ["2m_temperature", "volumetric_soil_water_layer_1"],
            "time": [f"{hour:02d}:00" for hour in range(24)],
        }
        era5 = source_path(downloads, "era5", month, request, data.get("cache_areas", []))
        retrieve(cds, "reanalysis-era5-single-levels", request, era5)
        weather.append(
            grib_points(
                era5, {"t2m": "temperature", "swvl1": "soil_moisture"}, data["reaches"], 0.26
            )
        )
    rainfall = chirps_points(data, raw)
    keys = ["date", "reach_id"]
    frame = pd.concat(discharge).merge(pd.concat(weather), on=keys, validate="one_to_one")
    frame = frame.merge(rainfall, on=keys, validate="one_to_one")
    frame = frame[frame.date.between(data["start"], data["end"])].sort_values(keys)
    frame.to_parquet(raw / "series.parquet", index=False)
    write_json(
        raw / "manifest.json",
        {
            "sources": {
                "glofas": "cems-glofas-historical",
                "era5": "reanalysis-era5-single-levels",
                "chirps": data["chirps_url"],
                "thresholds": data["threshold_url"],
            },
            "glofas_version": data["glofas_version"],
            "thresholds_sha256": digest("data/auxiliary/thresholds.csv"),
            "downloads": {p.name: digest(p) for p in sorted(downloads.glob("*.grib"))},
            "files": {
                p.name: digest(p)
                for p in sorted(raw.iterdir())
                if p.is_file() and p.name != "manifest.json"
            },
        },
    )


if __name__ == "__main__":
    ingest()
