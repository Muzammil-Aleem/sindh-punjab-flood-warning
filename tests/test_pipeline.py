import json
import shutil
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import yaml

from flood.evaluate import evaluate
from flood.features import build_features
from flood.register import register
from flood.train import train
from flood.validate import validate


@pytest.fixture
def isolated_pipeline(tmp_path, monkeypatch, observations, thresholds, cfg):
    requirements = Path("requirements.txt").resolve()
    monkeypatch.chdir(tmp_path)
    Path("params.yaml").write_text(yaml.safe_dump(cfg))
    shutil.copy(requirements, "requirements.txt")
    raw = Path("data/raw")
    raw.mkdir(parents=True)
    observations.to_parquet(raw / "series.parquet", index=False)
    Path("data/auxiliary").mkdir(parents=True)
    thresholds.to_csv("data/auxiliary/thresholds.csv", index=False)
    (raw / "manifest.json").write_text(json.dumps({"source": "synthetic_test_fixture"}))
    runs = []

    @contextmanager
    def start_run(run_name=None, run_id=None):
        identifier = run_id or f"test-{run_name}"
        runs.append(identifier)
        yield SimpleNamespace(info=SimpleNamespace(run_id=identifier))

    monkeypatch.setattr(mlflow, "start_run", start_run)
    for method in [
        "set_tracking_uri",
        "set_experiment",
        "set_tags",
        "log_params",
        "log_param",
        "log_metrics",
        "log_artifact",
    ]:
        monkeypatch.setattr(mlflow, method, lambda *args, **kwargs: None)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://example.org/synthetic-test-tracking")
    monkeypatch.setattr(
        mlflow.sklearn,
        "log_model",
        lambda *args, **kwargs: SimpleNamespace(model_uri="runs:/synthetic-test/model"),
    )
    return runs


def test_pipeline_validates_trains_evaluates_and_registers(isolated_pipeline, monkeypatch):
    class Registry:
        def search_model_versions(self, query):
            return []

        def set_model_version_tag(self, *args):
            pass

        def transition_model_version_stage(self, *args, **kwargs):
            assert kwargs["stage"] == "Staging"

        def get_model_version(self, name, version):
            return SimpleNamespace(version="1", current_stage="Staging")

    monkeypatch.setattr("flood.register.MlflowClient", Registry)
    monkeypatch.setattr(
        mlflow, "register_model", lambda *args, **kwargs: SimpleNamespace(version="1")
    )
    validate()
    build_features()
    train()
    evaluate()
    register()
    report = json.loads(Path("reports/evaluation.json").read_text())
    receipt = json.loads(Path("reports/registry.json").read_text())
    assert set(report["models"]) == {"persistence", "logistic_regression", "random_forest"}
    assert report["selection_split"] == "validation"
    assert receipt["winner"] == report["winner"]
    assert receipt["stage"] == "Staging"
    assert len(set(isolated_pipeline)) == 3
    assert len(list(Path("reports/figures").glob("*.png"))) == 6
    assert all(len(value["per_horizon"]) == 5 for value in report["models"].values())
