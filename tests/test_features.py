import numpy as np
import pandas as pd

from flood.features import engineer, split_rows


def test_labels_and_horizons(observations, thresholds, cfg):
    table, names, _, _ = engineer(observations, thresholds, cfg)
    truth = observations.set_index(["reach_id", "date"]).discharge
    for horizon in cfg["features"]["horizons"]:
        rows = table[table.horizon.eq(horizon)].head(100)
        for row in rows.itertuples():
            assert row.target_date - row.date == pd.Timedelta(days=horizon)
            assert row.label == int(truth.loc[(row.reach_id, row.target_date)] >= 1400)
    assert "label" not in names
    assert "target_date" not in names
    assert not table[names].isna().any().any()


def test_future_changes_cannot_change_features(observations, thresholds, cfg):
    cutoff = pd.Timestamp("2020-06-01")
    before, names, sequences_before, keys_before = engineer(observations, thresholds, cfg)
    altered = observations.copy()
    altered.loc[
        altered.date.gt(cutoff), ["discharge", "rainfall", "temperature", "soil_moisture"]
    ] *= 2
    after, _, sequences_after, keys_after = engineer(altered, thresholds, cfg)
    pd.testing.assert_frame_equal(
        before.loc[before.date.le(cutoff), names], after.loc[after.date.le(cutoff), names]
    )
    np.testing.assert_array_equal(
        sequences_before[keys_before.date.le(cutoff)], sequences_after[keys_after.date.le(cutoff)]
    )


def test_split_purges_target_overlap(observations, thresholds, cfg):
    table, _, _, _ = engineer(observations, thresholds, cfg)
    splits = split_rows(table, cfg)
    assert splits["train"].target_date.max() < splits["validation"].date.min()
    assert splits["validation"].target_date.max() < splits["test"].date.min()
    for left, right in [("train", "validation"), ("validation", "test")]:
        left_keys = set(zip(splits[left].reach_id, splits[left].date))
        right_keys = set(zip(splits[right].reach_id, splits[right].date))
        assert not left_keys.intersection(right_keys)


def test_sequences_align_with_history_and_targets(observations, thresholds, cfg):
    _, _, sequences, keys = engineer(observations, thresholds, cfg)
    horizons = cfg["features"]["horizons"]
    assert sequences.shape == (len(keys), 30, 4)
    assert sequences.dtype == np.float32
    assert keys[[f"label_{h}" for h in horizons]].shape == (len(keys), 5)
    row = keys.iloc[0]
    reach = observations[observations.reach_id.eq(row.reach_id)].set_index("date")
    expected = reach.loc[
        row.date - pd.Timedelta(days=29) : row.date,
        ["discharge", "rainfall", "temperature", "soil_moisture"],
    ]
    np.testing.assert_allclose(sequences[0], expected.to_numpy(), rtol=1e-6)
    for horizon in horizons:
        assert row[f"label_{horizon}"] == int(
            reach.loc[row.date + pd.Timedelta(days=horizon), "discharge"] >= 1400
        )


def test_lags_and_rolling_use_current_and_past_only(observations, thresholds, cfg):
    table, _, _, _ = engineer(observations, thresholds, cfg)
    row = table.iloc[100]
    reach = observations[observations.reach_id.eq(row.reach_id)].set_index("date")
    assert row.discharge_lag_7 == reach.loc[row.date - pd.Timedelta(days=7), "discharge"]
    expected = reach.loc[row.date - pd.Timedelta(days=6) : row.date, "rainfall"].sum()
    np.testing.assert_allclose(row.rainfall_sum_7, expected)
