import numpy as np
import pandas as pd
import pytest

from flood.validate import validate_series


def test_accepts_complete_data(observations, thresholds, cfg):
    assert len(validate_series(observations, thresholds, cfg)) == len(observations)


@pytest.mark.parametrize(
    "corruption", ["null", "range", "gap", "duplicate", "schema", "infinity", "date"]
)
def test_rejects_corrupt_data(observations, thresholds, cfg, corruption):
    frame = observations.copy()
    if corruption == "null":
        frame.loc[0, "rainfall"] = np.nan
    elif corruption == "range":
        frame.loc[0, "discharge"] = -1
    elif corruption == "gap":
        frame = frame.iloc[1:]
    elif corruption == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif corruption == "schema":
        frame = frame.drop(columns="temperature")
    elif corruption == "infinity":
        frame.loc[0, "soil_moisture"] = np.inf
    elif corruption == "date":
        frame.loc[0, "date"] += pd.Timedelta(hours=1)
    with pytest.raises(ValueError):
        validate_series(frame, thresholds, cfg)


@pytest.mark.parametrize(
    "field,value",
    [
        ("threshold", -1),
        ("return_period", 10),
        ("glofas_version", "version_3_1"),
        ("latitude", 0),
        ("source_url", "unknown"),
    ],
)
def test_rejects_invalid_thresholds(observations, thresholds, cfg, field, value):
    thresholds.loc[0, field] = value
    with pytest.raises(ValueError):
        validate_series(observations, thresholds, cfg)
