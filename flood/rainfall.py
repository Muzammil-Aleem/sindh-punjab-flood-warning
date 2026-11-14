import ast
import hashlib
import json
import math
import time
from pathlib import Path

import pandas as pd
import requests

from flood.common import write_json


def response_data(response):
    response.raise_for_status()
    try:
        value = response.json()
    except ValueError:
        value = response.text
    while isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = ast.literal_eval(value)
        if parsed == value:
            break
        value = parsed
    return value


def cell_polygon(latitude, longitude):
    x = -180 + (math.floor((longitude + 180) / 0.05 + 1e-9) + 0.5) * 0.05
    y = 50 - (math.floor((50 - latitude) / 0.05 + 1e-9) + 0.5) * 0.05
    radius = 0.024999
    return {
        "type": "Polygon",
        "coordinates": [
            [
                [x - radius, y - radius],
                [x + radius, y - radius],
                [x + radius, y + radius],
                [x - radius, y + radius],
                [x - radius, y - radius],
            ]
        ],
    }


def rainfall_frame(payload, reach_id, start, end):
    rows = []
    for item in payload["data"]:
        date = (
            pd.to_datetime(float(item["epochTime"]), unit="s", utc=True)
            .tz_localize(None)
            .normalize()
        )
        rows.append({"date": date, "reach_id": reach_id, "rainfall": float(item["value"]["avg"])})
    frame = (
        pd.DataFrame(rows, columns=["date", "reach_id", "rainfall"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    if frame.empty or not pd.DatetimeIndex(frame.date).equals(pd.date_range(start, end)):
        raise ValueError(f"Incomplete CHIRPS daily coverage for {reach_id}")
    if not frame.rainfall.between(0, 2000).all():
        raise ValueError(f"Invalid CHIRPS rainfall for {reach_id}")
    return frame


def chirps_points(data, raw):
    base = data["chirps_url"].rstrip("/")
    frames = []
    for reach in data["reaches"]:
        request = {
            "datatype": 0,
            "begintime": pd.Timestamp(data["start"]).strftime("%m/%d/%Y"),
            "endtime": pd.Timestamp(data["end"]).strftime("%m/%d/%Y"),
            "intervaltype": 0,
            "operationtype": 5,
            "geometry": json.dumps(cell_polygon(reach["latitude"], reach["longitude"])),
        }
        key = hashlib.sha256(
            json.dumps({"base": base, **request}, sort_keys=True).encode()
        ).hexdigest()[:16]
        destination = Path(raw) / f"chirps-{reach['id']}-{key}.json"
        job_path = destination.with_suffix(".job.json")
        if not destination.exists():
            if job_path.exists():
                job_id = json.loads(job_path.read_text())["id"]
            else:
                job_id = response_data(
                    requests.get(f"{base}/submitDataRequest/", params=request, timeout=(30, 120))
                )[0]
                write_json(job_path, {"id": job_id, "request": request, "source": base})
            deadline = time.monotonic() + 3600
            while True:
                progress = response_data(
                    requests.get(
                        f"{base}/getDataRequestProgress/", params={"id": job_id}, timeout=(30, 120)
                    )
                )[0]
                if progress < 0:
                    raise RuntimeError(f"CHIRPS processing failed for {reach['id']}: {job_id}")
                if progress >= 100:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError(f"CHIRPS request is still processing: {job_id}")
                time.sleep(10)
            payload = response_data(
                requests.get(
                    f"{base}/getDataFromRequest/", params={"id": job_id}, timeout=(30, 120)
                )
            )
            rainfall_frame(payload, reach["id"], data["start"], data["end"])
            write_json(destination, payload)
        frames.append(
            rainfall_frame(
                json.loads(destination.read_text()), reach["id"], data["start"], data["end"]
            )
        )
        print(f"CHIRPS ready: {reach['id']}", flush=True)
    return pd.concat(frames, ignore_index=True)
