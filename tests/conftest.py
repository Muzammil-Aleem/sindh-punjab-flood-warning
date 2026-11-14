from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def cfg():
    value = yaml.safe_load(Path("params.yaml").read_text())
    value["data"].update(start="2018-01-01", end="2022-12-31")
    value["split"].update(train_end="2019-12-31", validation_end="2021-12-31")
    value["models"]["random_forest"].update(n_estimators=10, max_depth=4)
    return value


@pytest.fixture
def observations(cfg):
    days = pd.date_range(cfg["data"]["start"], cfg["data"]["end"])
    frames = []
    for i, reach in enumerate(cfg["data"]["reaches"]):
        t = np.arange(len(days))
        frames.append(
            pd.DataFrame(
                {
                    "date": days,
                    "reach_id": reach["id"],
                    "discharge": 1000 + 700 * np.sin(2 * np.pi * (t + i * 3) / 90),
                    "rainfall": 10 + 10 * np.sin(2 * np.pi * t / 20),
                    "temperature": 290 + 10 * np.sin(2 * np.pi * t / 365),
                    "soil_moisture": 0.25 + 0.1 * np.sin(2 * np.pi * t / 60),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def thresholds(cfg):
    return pd.DataFrame(
        [
            {
                "reach_id": reach["id"],
                "threshold": 1400.0,
                "return_period": 2,
                "glofas_version": cfg["data"]["glofas_version"],
                "latitude": reach["latitude"],
                "longitude": reach["longitude"],
                "source_url": "https://example.org/synthetic-test-fixture",
            }
            for reach in cfg["data"]["reaches"]
        ]
    )
