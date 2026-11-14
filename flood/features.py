from pathlib import Path

import numpy as np
import pandas as pd

from flood.common import config, write_json
from flood.validate import VARIABLES


def split_rows(frame, cfg):
    train_end = pd.Timestamp(cfg["split"]["train_end"])
    validation_end = pd.Timestamp(cfg["split"]["validation_end"])
    if train_end >= validation_end:
        raise ValueError("train_end must precede validation_end")
    result = {}
    masks = {
        "train": frame.target_date.le(train_end),
        "validation": frame.date.gt(train_end) & frame.target_date.le(validation_end),
        "test": frame.date.gt(validation_end),
    }
    for name, mask in masks.items():
        result[name] = frame.loc[mask].copy().reset_index(drop=True)
        if result[name].empty:
            raise ValueError(f"Empty {name} split")
    return result


def engineer(frame, thresholds, cfg):
    options = cfg["features"]
    feature_names = list(VARIABLES)
    feature_names += [f"{v}_lag_{lag}" for v in VARIABLES for lag in options["lags"]]
    feature_names += [
        f"{v}_sum_{w}" for v in ["rainfall", "discharge"] for w in options["rolling_windows"]
    ]
    feature_names += ["discharge_change", "day_sin", "day_cos", "horizon", "threshold"]
    threshold_map = thresholds.set_index("reach_id").threshold
    frames, sequences, sequence_keys = [], [], []
    lookback = options["sequence_length"]
    for reach_id, group in frame.groupby("reach_id", sort=True):
        group = group.sort_values("date").reset_index(drop=True).copy()
        if not group.date.diff().iloc[1:].eq(pd.Timedelta(days=1)).all():
            raise ValueError("Feature engineering requires consecutive daily observations")
        for variable in VARIABLES:
            for lag in options["lags"]:
                group[f"{variable}_lag_{lag}"] = group[variable].shift(lag)
        for variable in ["rainfall", "discharge"]:
            for window in options["rolling_windows"]:
                group[f"{variable}_sum_{window}"] = (
                    group[variable].rolling(window, min_periods=window).sum()
                )
        group["discharge_change"] = group.discharge.diff()
        group["day_sin"] = np.sin(2 * np.pi * group.date.dt.dayofyear / 365.25)
        group["day_cos"] = np.cos(2 * np.pi * group.date.dt.dayofyear / 365.25)
        group["threshold"] = float(threshold_map.loc[reach_id])
        group["persistence"] = (group.discharge >= group.threshold).astype(int)
        for horizon in options["horizons"]:
            part = group.copy()
            future = part.discharge.shift(-horizon)
            part["label"] = (future >= part.threshold).astype(int)
            part["target_date"] = part.date + pd.to_timedelta(horizon, unit="D")
            part["horizon"] = horizon
            valid = future.notna() & part[feature_names].notna().all(axis=1)
            valid &= part.index >= lookback - 1
            frames.append(
                part.loc[
                    valid,
                    ["reach_id", "date", "target_date", "label", "persistence", *feature_names],
                ]
            )
        for end in range(lookback - 1, len(group) - max(options["horizons"])):
            if group.loc[end, [name for name in feature_names if name != "horizon"]].isna().any():
                continue
            sequences.append(
                group.loc[end - lookback + 1 : end, VARIABLES].to_numpy(dtype=np.float32)
            )
            sequence_keys.append(
                {
                    "reach_id": reach_id,
                    "date": group.at[end, "date"],
                    "target_date": group.at[end, "date"]
                    + pd.Timedelta(days=max(options["horizons"])),
                    **{
                        f"label_{h}": int(
                            group.at[end + h, "discharge"] >= group.at[end, "threshold"]
                        )
                        for h in options["horizons"]
                    },
                }
            )
    if not sequences:
        raise ValueError("Insufficient history to construct sequences")
    return (
        pd.concat(frames, ignore_index=True),
        feature_names,
        np.stack(sequences),
        pd.DataFrame(sequence_keys),
    )


def build_features():
    cfg = config()
    frame = pd.read_parquet("data/validated/series.parquet")
    thresholds = pd.read_csv("data/validated/thresholds.csv")
    table, names, sequences, keys = engineer(frame, thresholds, cfg)
    target = Path("data/features")
    target.mkdir(parents=True, exist_ok=True)
    for name, split in split_rows(table, cfg).items():
        if split.label.nunique() != 2:
            raise ValueError(f"{name} requires positive and negative flood examples")
        split.to_parquet(target / f"{name}.parquet", index=False)
    for name, split in split_rows(keys.assign(sequence_index=np.arange(len(keys))), cfg).items():
        indices = split.sequence_index.to_numpy()
        np.savez_compressed(
            target / f"sequences_{name}.npz",
            X=sequences[indices],
            y=split[[f"label_{h}" for h in cfg["features"]["horizons"]]].to_numpy(dtype=np.int64),
        )
        split.to_parquet(target / f"sequence_keys_{name}.parquet", index=False)
    write_json(
        target / "schema.json",
        {
            "features": names,
            "sequence_variables": VARIABLES,
            "sequence_length": cfg["features"]["sequence_length"],
            "horizons": cfg["features"]["horizons"],
            "issue_time": "end_of_day_UTC",
        },
    )
    write_json(
        "reports/splits.json",
        {
            name: {
                "rows": len(split),
                "positive_fraction": float(split.label.mean()),
                "first_issue": str(split.date.min().date()),
                "last_target": str(split.target_date.max().date()),
            }
            for name, split in split_rows(table, cfg).items()
        },
    )


if __name__ == "__main__":
    build_features()
