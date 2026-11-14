import copy

import numpy as np
import pandas as pd
import pytest

from flood.weather import weather_frame


@pytest.fixture
def weather():
    return {
        "latitude": 30.5,
        "longitude": 70.75,
        "utc_offset_seconds": 0,
        "hourly_units": {"temperature_2m": "°C", "soil_moisture_0_to_7cm": "m³/m³"},
        "hourly": {
            "time": pd.date_range("2010-01-01", periods=48, freq="h").astype(str).tolist(),
            "temperature_2m": list(np.arange(48) / 2),
            "soil_moisture_0_to_7cm": [0.25] * 48,
        },
    }


def convert(weather):
    return weather_frame(
        weather,
        {"id": "reach", "latitude": 30.52, "longitude": 70.85},
        "2010-01-01",
        "2010-01-02",
    )


def test_daily_means_convert_celsius_to_kelvin(weather):
    result = convert(weather)
    np.testing.assert_allclose(result.temperature, [278.9, 290.9])
    assert result.soil_moisture.tolist() == [0.25, 0.25]
    assert result.reach_id.tolist() == ["reach", "reach"]


@pytest.mark.parametrize(
    "fault", ["missing_hour", "duplicate_hour", "null", "offset", "units", "grid"]
)
def test_weather_rejects_invalid_source_data(weather, fault):
    payload = copy.deepcopy(weather)
    if fault == "missing_hour":
        payload["hourly"]["time"].pop()
    elif fault == "duplicate_hour":
        payload["hourly"]["time"][1] = payload["hourly"]["time"][0]
    elif fault == "null":
        payload["hourly"]["temperature_2m"][0] = None
    elif fault == "offset":
        payload["utc_offset_seconds"] = 3600
    elif fault == "units":
        payload["hourly_units"]["temperature_2m"] = "°F"
    else:
        payload["latitude"] = 32.0
    with pytest.raises(ValueError):
        convert(payload)
