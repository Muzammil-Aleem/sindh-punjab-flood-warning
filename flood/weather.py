import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from flood.common import write_json


def weather_frame(payload, reach, start, end):
    if payload.get("utc_offset_seconds") != 0:
        raise ValueError("ERA5 point data must use UTC")
    units = payload["hourly_units"]
    if units.get("temperature_2m") != "°C" or units.get("soil_moisture_0_to_7cm") != "m³/m³":
        raise ValueError("Unexpected ERA5 point-data units")
    if (
        max(
            abs(payload["latitude"] - reach["latitude"]),
            abs(payload["longitude"] - reach["longitude"]),
        )
        > 0.26
    ):
        raise ValueError("ERA5 grid cell is too far from the reach")
    hourly = payload["hourly"]
    index = pd.DatetimeIndex(pd.to_datetime(hourly["time"]))
    expected = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(hours=23), freq="h")
    if not index.equals(expected):
        raise ValueError("ERA5 requires complete, unique hourly coverage")
    frame = pd.DataFrame(
        {
            "temperature": np.asarray(hourly["temperature_2m"], dtype=float) + 273.15,
            "soil_moisture": np.asarray(hourly["soil_moisture_0_to_7cm"], dtype=float),
        },
        index=index,
    )
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("ERA5 hourly data contains null or nonfinite values")
    if not frame.temperature.between(220, 340).all() or not frame.soil_moisture.between(0, 1).all():
        raise ValueError("ERA5 hourly data is outside physical ranges")
    frame = frame.resample("D").mean()
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["reach_id"] = reach["id"]
    return frame


def era5_points(data, raw):
    source = data["era5_url"]
    frames = []
    days = pd.date_range(data["start"], data["end"])
    for reach in data["reaches"]:
        for year in days.year.unique():
            selected = days[days.year == year]
            start, end = selected.min().strftime("%Y-%m-%d"), selected.max().strftime("%Y-%m-%d")
            request = {
                "latitude": reach["latitude"],
                "longitude": reach["longitude"],
                "start_date": start,
                "end_date": end,
                "hourly": "temperature_2m,soil_moisture_0_to_7cm",
                "models": "era5",
                "timezone": "GMT",
                "temperature_unit": "celsius",
                "cell_selection": "nearest",
                "elevation": "nan",
            }
            fingerprint = hashlib.sha256(
                json.dumps({"source": source, **request}, sort_keys=True).encode()
            ).hexdigest()[:16]
            destination = Path(raw) / f"era5-point-{reach['id']}-{year}-{fingerprint}.json"
            if not destination.exists():
                for attempt in range(5):
                    response = requests.get(source, params=request, timeout=(30, 180))
                    if response.status_code in {429, 500, 502, 503, 504} and attempt < 4:
                        time.sleep(
                            min(120, max(10, int(response.headers.get("Retry-After", "60"))))
                        )
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    weather_frame(payload, reach, start, end)
                    write_json(destination, {"source": source, "request": request, "data": payload})
                    break
                time.sleep(3)
            payload = json.loads(destination.read_text())["data"]
            frames.append(weather_frame(payload, reach, start, end))
            print(f"ERA5 ready: {reach['id']} {year}", flush=True)
    return pd.concat(frames, ignore_index=True)
