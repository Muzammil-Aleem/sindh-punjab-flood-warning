import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from flood.common import config, write_json
from flood.train import metrics, setup_tracking

plt.switch_backend("Agg")


def evaluate():
    cfg = config()
    setup_tracking(cfg)
    training = json.loads(Path("reports/training.json").read_text())
    names = json.loads(Path("data/features/schema.json").read_text())["features"]
    test = pd.read_parquet("data/features/test.parquet")
    result = {"winner": training["winner"], "selection_split": "validation", "models": {}}
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    predictions = test[["date", "target_date", "reach_id", "horizon", "label"]].copy()
    for name, record in training["models"].items():
        model = joblib.load(f"models/{name}.joblib")
        probability = model.predict_proba(test[names])[:, 1]
        predictions[f"{name}_probability"] = probability
        predictions[f"{name}_prediction"] = (probability >= record["threshold"]).astype(int)
        scores = metrics(test.label, probability, record["threshold"])
        horizons = {}
        for horizon, rows in test.groupby("horizon"):
            subset = probability[rows.index.to_numpy()]
            if rows.label.nunique() != 2:
                raise ValueError(f"Test horizon {horizon} requires both classes")
            horizons[str(horizon)] = metrics(rows.label, subset, record["threshold"])
        reaches = {}
        for reach, rows in test.groupby("reach_id"):
            if rows.label.nunique() != 2:
                raise ValueError(f"Test reach {reach} requires both classes")
            reaches[reach] = metrics(
                rows.label, probability[rows.index.to_numpy()], record["threshold"]
            )
        result["models"][name] = {
            "test": scores,
            "per_horizon": horizons,
            "per_reach": reaches,
            "validation": record["validation"],
            "threshold": record["threshold"],
            "run_id": record["run_id"],
        }
        with mlflow.start_run(run_id=record["run_id"]):
            mlflow.log_metrics({f"test_{key}": value for key, value in scores.items()})
            for horizon, values in horizons.items():
                mlflow.log_metrics(
                    {f"test_h{horizon}_{key}": value for key, value in values.items()}
                )
            for reach, values in reaches.items():
                mlflow.log_metrics({f"test_{reach}_{key}": value for key, value in values.items()})
            for kind in ["precision_recall", "confusion_matrix"]:
                fig, axis = plt.subplots(figsize=(6, 5))
                if kind == "precision_recall":
                    PrecisionRecallDisplay.from_predictions(test.label, probability, ax=axis)
                else:
                    ConfusionMatrixDisplay.from_predictions(
                        test.label, probability >= record["threshold"], labels=[0, 1], ax=axis
                    )
                axis.set_title(name)
                fig.tight_layout()
                path = f"reports/figures/{name}_{kind}.png"
                fig.savefig(path, dpi=160)
                plt.close(fig)
                mlflow.log_artifact(path, "evaluation")
    predictions.to_parquet("reports/test_predictions.parquet", index=False)
    write_json("reports/evaluation.json", result)
    for record in training["models"].values():
        with mlflow.start_run(run_id=record["run_id"]):
            mlflow.log_artifact("reports/evaluation.json", "evaluation")
            mlflow.log_artifact("reports/test_predictions.parquet", "evaluation")


if __name__ == "__main__":
    evaluate()
