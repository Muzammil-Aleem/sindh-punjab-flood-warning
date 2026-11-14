from pathlib import Path

import numpy as np
import pandas as pd

from flood.common import config, digest, write_json


VARIABLES = ["discharge", "rainfall", "temperature", "soil_moisture"]


def validate_series(frame, thresholds, cfg):
    required = {"date", "reach_id", *VARIABLES}
    if set(frame.columns) != required:
        raise ValueError(f"Series schema must be {sorted(required)}")
    if frame.isna().any().any() or not np.isfinite(frame[VARIABLES].to_numpy()).all():
        raise ValueError("Series contains null or nonfinite values")
    if frame.duplicated(["date", "reach_id"]).any():
        raise ValueError("Duplicate date/reach rows")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError("Dates must be daily midnight timestamps")
    expected_reaches = {reach["id"] for reach in cfg["data"]["reaches"]}
    if set(frame.reach_id) != expected_reaches:
        raise ValueError("Reach coverage differs from configuration")
    expected_dates = pd.date_range(cfg["data"]["start"], cfg["data"]["end"])
    for reach_id, group in frame.groupby("reach_id"):
        if not pd.DatetimeIndex(group.date.sort_values()).equals(expected_dates):
            raise ValueError(f"Incomplete daily coverage for {reach_id}")
    bounds = cfg["validation"]
    limits = {
        "discharge": (0, bounds["discharge_max"]),
        "rainfall": (0, bounds["rainfall_max"]),
        "temperature": (bounds["temperature_min"], bounds["temperature_max"]),
        "soil_moisture": (0, bounds["soil_moisture_max"]),
    }
    for variable, (lower, upper) in limits.items():
        if not frame[variable].between(lower, upper).all():
            raise ValueError(f"Out-of-range {variable}")
    threshold_fields = {
        "reach_id",
        "threshold",
        "return_period",
        "glofas_version",
        "latitude",
        "longitude",
        "source_url",
    }
    if set(thresholds.columns) != threshold_fields or thresholds.isna().any().any():
        raise ValueError(f"Threshold CSV schema must be {sorted(threshold_fields)}")
    if thresholds.reach_id.duplicated().any() or set(thresholds.reach_id) != expected_reaches:
        raise ValueError("Thresholds must contain exactly one row per configured reach")
    if not thresholds.threshold.between(0.001, bounds["discharge_max"]).all():
        raise ValueError("Invalid discharge thresholds")
    if not thresholds.return_period.eq(cfg["data"]["threshold_return_period"]).all():
        raise ValueError("Unexpected threshold return period")
    if not thresholds.glofas_version.eq(cfg["data"]["glofas_version"]).all():
        raise ValueError("Threshold and discharge GloFAS versions differ")
    if not thresholds.source_url.str.startswith("https://").all():
        raise ValueError("Thresholds require HTTPS provenance URLs")
    for reach in cfg["data"]["reaches"]:
        threshold = thresholds.set_index("reach_id").loc[reach["id"]]
        if (
            max(
                abs(threshold.latitude - reach["latitude"]),
                abs(threshold.longitude - reach["longitude"]),
            )
            > cfg["data"]["maximum_grid_distance"]
        ):
            raise ValueError("Threshold grid cell differs from reach")
    return frame.sort_values(["reach_id", "date"]).reset_index(drop=True)


def validate():
    cfg = config()
    frame = pd.read_parquet("data/raw/series.parquet")
    thresholds = pd.read_csv("data/auxiliary/thresholds.csv")
    frame = validate_series(frame, thresholds, cfg)
    Path("data/validated").mkdir(parents=True, exist_ok=True)
    frame.to_parquet("data/validated/series.parquet", index=False)
    thresholds.to_csv("data/validated/thresholds.csv", index=False)
    write_json(
        "reports/validation.json",
        {
            "rows": len(frame),
            "reaches": int(frame.reach_id.nunique()),
            "start": str(frame.date.min().date()),
            "end": str(frame.date.max().date()),
            "nulls": 0,
            "duplicates": 0,
            "series_sha256": digest("data/raw/series.parquet"),
            "thresholds_sha256": digest("data/auxiliary/thresholds.csv"),
        },
    )


if __name__ == "__main__":
    validate()
