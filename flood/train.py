import json
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from flood.common import config, digest, remote_uri, write_json


class PersistenceClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.classes_ = np.array([0, 1])
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def predict_proba(self, X):
        probability = (X["discharge"] >= X["threshold"]).to_numpy(dtype=float)
        return np.column_stack([1 - probability, probability])

    def predict(self, X):
        return self.predict_proba(X)[:, 1].astype(int)


class ThresholdClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, estimator, threshold=0.5):
        self.estimator = estimator
        self.threshold = threshold

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def n_features_in_(self):
        return self.estimator.n_features_in_

    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def metrics(y, probability, threshold=0.5):
    prediction = np.asarray(probability) >= threshold
    return {
        "f1": float(f1_score(y, prediction, zero_division=0)),
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probability)),
        "positive_fraction": float(np.mean(y)),
    }


def setup_tracking(cfg):
    mlflow.set_tracking_uri(remote_uri("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(cfg["tracking"]["experiment"])


def train():
    cfg = config()
    setup_tracking(cfg)
    schema = json.loads(Path("data/features/schema.json").read_text())
    names = schema["features"]
    train_frame = pd.read_parquet("data/features/train.parquet")
    validation = pd.read_parquet("data/features/validation.parquet")
    seed = cfg["models"]["seed"]
    candidates = {
        "persistence": PersistenceClassifier(),
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                **cfg["models"]["logistic_regression"],
                class_weight="balanced",
                random_state=seed,
            ),
        ),
        "random_forest": RandomForestClassifier(
            **cfg["models"]["random_forest"],
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=1,
        ),
    }
    destination = Path("models")
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
    )
    for name, estimator in candidates.items():
        with mlflow.start_run(run_name=name) as run:
            mlflow.set_tags(
                {
                    "model_family": name,
                    "dataset_sha256": digest("data/features/train.parquet"),
                    "git_commit": commit or "uncommitted",
                    "git_dirty": str(dirty).lower(),
                    "label_source": "GloFAS_reanalysis_return_period_exceedance",
                    "selection_split": "validation",
                }
            )
            parameters = {
                key: value
                for key, value in estimator.get_params(deep=True).items()
                if isinstance(value, (str, int, float, bool)) or value is None
            }
            mlflow.log_params(parameters)
            mlflow.log_params(
                {
                    "seed": seed,
                    "train_rows": len(train_frame),
                    "validation_rows": len(validation),
                    "horizons": json.dumps(schema["horizons"]),
                    "train_end": cfg["split"]["train_end"],
                    "validation_end": cfg["split"]["validation_end"],
                }
            )
            estimator.fit(train_frame[names], train_frame.label)
            probabilities = estimator.predict_proba(validation[names])[:, 1]
            threshold = 0.5
            if name != "persistence":
                grid = np.linspace(0.05, 0.95, 19)
                threshold = float(
                    max(
                        grid,
                        key=lambda value: (
                            f1_score(validation.label, probabilities >= value, zero_division=0),
                            -abs(value - 0.5),
                        ),
                    )
                )
            score = metrics(validation.label, probabilities, threshold)
            mlflow.log_param("decision_threshold", threshold)
            mlflow.log_metrics({f"validation_{key}": value for key, value in score.items()})
            estimator = ThresholdClassifier(estimator, threshold)
            sample = train_frame[names].head(5)
            model_info = mlflow.sklearn.log_model(
                estimator,
                artifact_path="model",
                input_example=sample,
                signature=infer_signature(sample, estimator.predict(sample)),
                pip_requirements="requirements.lock",
                code_paths=["flood"],
            )
            mlflow.log_artifact("params.yaml")
            mlflow.log_artifact("requirements.txt")
            mlflow.log_artifact("requirements.lock")
            mlflow.log_artifact(".python-version")
            mlflow.log_artifact("dvc.yaml")
            mlflow.log_artifact("dvc.lock")
            mlflow.log_artifact("data/features/schema.json")
            mlflow.log_artifact("data/raw/manifest.json")
            mlflow.log_artifact("reports/validation.json")
            mlflow.log_artifact("reports/splits.json")
            joblib.dump(estimator, destination / f"{name}.joblib")
            records[name] = {
                "run_id": run.info.run_id,
                "model_uri": model_info.model_uri,
                "threshold": threshold,
                "validation": score,
            }
    eligible = [name for name in records if name != "persistence"]
    winner = max(eligible, key=lambda name: records[name]["validation"]["f1"])
    write_json("reports/training.json", {"winner": winner, "models": records})


if __name__ == "__main__":
    train()
