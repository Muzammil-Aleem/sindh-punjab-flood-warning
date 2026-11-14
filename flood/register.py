import json
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from flood.common import config, write_json
from flood.train import setup_tracking


def register():
    cfg = config()
    setup_tracking(cfg)
    training = json.loads(Path("reports/training.json").read_text())
    evaluation = json.loads(Path("reports/evaluation.json").read_text())
    winner = training["winner"]
    record = training["models"][winner]
    if evaluation["winner"] != winner:
        raise ValueError("Training and evaluation winners differ")
    client = MlflowClient()
    name = cfg["tracking"]["registered_model"]
    existing = [
        version
        for version in client.search_model_versions(f"name='{name}'")
        if version.run_id == record["run_id"]
    ]
    version = (
        existing[0]
        if existing
        else mlflow.register_model(record["model_uri"], name, await_registration_for=300)
    )
    client.set_model_version_tag(
        name, version.version, "decision_threshold", str(record["threshold"])
    )
    client.set_model_version_tag(name, version.version, "selection_split", "validation")
    client.transition_model_version_stage(
        name, version.version, stage=cfg["tracking"]["stage"], archive_existing_versions=True
    )
    checked = client.get_model_version(name, version.version)
    if checked.current_stage != cfg["tracking"]["stage"]:
        raise RuntimeError("Registry stage verification failed")
    write_json(
        "reports/registry.json",
        {
            "name": name,
            "version": checked.version,
            "stage": checked.current_stage,
            "run_id": record["run_id"],
            "winner": winner,
            "decision_threshold": record["threshold"],
        },
    )


if __name__ == "__main__":
    register()
