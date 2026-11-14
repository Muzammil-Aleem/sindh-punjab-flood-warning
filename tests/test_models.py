import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flood.common import remote_uri
from flood.features import engineer, split_rows
from flood.train import PersistenceClassifier, ThresholdClassifier, metrics


def test_models_run_on_engineered_features(observations, thresholds, cfg):
    table, names, _, _ = engineer(observations, thresholds, cfg)
    splits = split_rows(table, cfg)
    train, test = splits["train"], splits["test"]
    estimators = [
        PersistenceClassifier(),
        make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        RandomForestClassifier(n_estimators=10, max_depth=4, random_state=42, n_jobs=1),
    ]
    for model in estimators:
        model.fit(train[names], train.label)
        probability = model.predict_proba(test[names])
        assert probability.shape == (len(test), 2)
        assert np.isfinite(probability).all()
        assert np.logical_and(probability >= 0, probability <= 1).all()
        scores = metrics(test.label, probability[:, 1])
        assert all(0 <= value <= 1 for value in scores.values())


def test_persistence_uses_only_current_discharge():
    frame = pd.DataFrame({"discharge": [99, 100, 101], "threshold": [100, 100, 100]})
    model = PersistenceClassifier().fit(frame)
    np.testing.assert_array_equal(model.predict(frame), [0, 1, 1])


def test_single_class_subgroup_has_no_pr_auc():
    result = metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert result["pr_auc"] is None
    assert result["positive_fraction"] == 0
    assert result["f1"] == 0


def test_registered_classifier_preserves_selected_threshold():
    frame = pd.DataFrame({"x": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]})
    estimator = LogisticRegression().fit(frame, [0, 0, 0, 0, 1, 1])
    wrapper = ThresholdClassifier(estimator, 0.25)
    probability = wrapper.predict_proba(frame)[:, 1]
    np.testing.assert_array_equal(wrapper.predict(frame), probability >= 0.25)
    assert not np.array_equal(wrapper.predict(frame), estimator.predict(frame))


@pytest.mark.parametrize(
    "uri",
    [
        "",
        "file:///tmp/mlruns",
        "http://localhost:5000",
        "https://127.0.0.1",
        "https://user:password@example.org",
    ],
)
def test_rejects_missing_local_or_embedded_credentials(monkeypatch, uri):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    with pytest.raises(ValueError):
        remote_uri("MLFLOW_TRACKING_URI")
