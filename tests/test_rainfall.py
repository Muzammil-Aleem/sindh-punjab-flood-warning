import json

import pandas as pd
import pytest

from flood.rainfall import cell_polygon, rainfall_frame, response_data


def payload(values):
    return {
        "data": [
            {"epochTime": str(int(day.timestamp())), "value": {"avg": value}}
            for day, value in zip(
                pd.date_range("2020-01-01", periods=len(values), tz="UTC"), values
            )
        ]
    }


def test_preserves_daily_rainfall_values():
    frame = rainfall_frame(payload([0, 12.5, 20]), "reach", "2020-01-01", "2020-01-03")
    assert frame.rainfall.tolist() == [0, 12.5, 20]
    assert frame.reach_id.tolist() == ["reach"] * 3


@pytest.mark.parametrize("values", [[], [1, 2], [-9999, 1, 2], [float("nan"), 1, 2]])
def test_rejects_incomplete_or_invalid_rainfall(values):
    with pytest.raises(ValueError):
        rainfall_frame(payload(values), "reach", "2020-01-01", "2020-01-03")


def test_polygon_selects_the_containing_chirps_cell():
    polygon = cell_polygon(30.52, 70.85)
    ring = polygon["coordinates"][0]
    assert ring[0] == ring[-1]
    assert max(point[0] for point in ring) < 70.9
    assert min(point[0] for point in ring) > 70.85
    assert min(point[1] for point in ring) > 30.5
    assert max(point[1] for point in ring) < 30.55


@pytest.mark.parametrize("encoded", [[100.0], "[100.0]", json.dumps("[100.0]")])
def test_normalizes_climateserv_response_encodings(encoded):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return encoded

    assert response_data(Response()) == [100.0]
