import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
from dotenv import load_dotenv

import matplotlib.image as mpimg
from matplotlib.backends.backend_pdf import PdfPages
from mlflow.tracking import MlflowClient

from flood.common import remote_uri

plt.switch_backend("Agg")


def paragraph(fig, value, y, size=10):
    fig.text(
        0.08, y, "\n".join(textwrap.wrap(value, width=95)), va="top", fontsize=size, linespacing=1.5
    )


def page(title, number):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top")
    fig.text(0.92, 0.04, str(number), fontsize=9, ha="right")
    return fig


def screenshot(fig, path, box):
    axis = fig.add_axes(box)
    axis.imshow(mpimg.imread(path))
    axis.axis("off")


def main():
    load_dotenv()
    tracking = remote_uri("MLFLOW_TRACKING_URI")
    training = json.loads(Path("reports/training.json").read_text())
    evaluation = json.loads(Path("reports/evaluation.json").read_text())
    registry = json.loads(Path("reports/registry.json").read_text())
    validation = json.loads(Path("reports/validation.json").read_text())
    evidence = Path("submission/evidence")
    required = [evidence / name for name in ["runs.png", "registry.png", "dvc_dag.png"]]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(", ".join(missing))
    client = MlflowClient(tracking_uri=tracking)
    for record in training["models"].values():
        run = client.get_run(record["run_id"])
        if run.info.status != "FINISHED" or "test_f1" not in run.data.metrics:
            raise ValueError("Remote experiment evidence is incomplete")
    version = client.get_model_version(registry["name"], registry["version"])
    if version.current_stage != "Staging" or version.run_id != registry["run_id"]:
        raise ValueError("Remote registry evidence does not match the local receipt")
    github = (
        subprocess.run(
            ["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True
        )
        .stdout.strip()
        .removesuffix(".git")
    )
    if github.startswith("git@github.com:"):
        github = github.replace("git@github.com:", "https://github.com/")
    if not github.startswith("https://github.com/"):
        raise ValueError("A GitHub origin URL is required")
    destination = Path("submission/flood-warning.pdf")
    with PdfPages(destination) as pdf:
        fig = page("Flood Early Warning: Sindh and Punjab", 1)
        paragraph(fig, github, 0.87)
        paragraph(fig, tracking, 0.82)
        paragraph(
            fig,
            f"The core pipeline processes daily discharge, rainfall, temperature and soil moisture for {validation['reaches']} Indus river reaches. The validated dataset contains {validation['rows']} reach-day records spanning {validation['start']} to {validation['end']}. Predictions classify GloFAS two-year discharge threshold exceedances at exact lead times of three through seven days.",
            0.75,
        )
        paragraph(
            fig,
            "GloFAS historical reanalysis and hourly ERA5 are retrieved using authenticated Copernicus APIs. CHIRPS daily rainfall and official GloFAS threshold grids are downloaded from their publishers. GloFAS discharge is aligned to the preceding calendar day, matching its 24-hour averaging interval. The threshold file is pinned by URL version and SHA256.",
            0.59,
        )
        paragraph(
            fig,
            "Validation rejects schema changes, missing or infinite observations, duplicate reach-days, gaps in daily coverage, invalid physical ranges and inconsistent threshold versions. Raw sources, validated point series, engineered datasets and trained artifacts are tracked as DVC pipeline outputs; credentials, data and model weights are excluded from Git.",
            0.41,
        )
        paragraph(
            fig,
            "This is a retrospective classification study using reanalysis and model-derived labels. Its results do not establish operational warning skill or agreement with independently observed inundation. A comparison against archived GloFAS forecasts remains outside this core deliverable.",
            0.23,
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("Features and experiment tracking", 2)
        paragraph(
            fig,
            "The pipeline builds per-reach lag, rolling rainfall and discharge sums, discharge changes and seasonal features. It also exports 30-day, four-variable sequence tensors with five aligned labels for later temporal models. Chronological train, validation and test splits purge targets crossing split boundaries. Scalers fit only on training observations; decision thresholds and the winning trained classifier are selected on validation F1.",
            0.87,
        )
        screenshot(fig, evidence / "runs.png", [0.05, 0.28, 0.9, 0.42])
        paragraph(
            fig,
            "Persistence, class-weighted logistic regression and a class-weighted random forest are evaluated. Each remote MLflow run records estimator parameters, split settings, dataset fingerprints, validation and test metrics, the model signature, dependency versions, source manifests and evaluation plots.",
            0.22,
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("Evaluation and registered model", 3)
        rows = []
        for name, result in evaluation["models"].items():
            scores = result["test"]
            rows.append(
                [name, *[f"{scores[key]:.4f}" for key in ["f1", "precision", "recall", "pr_auc"]]]
            )
        axis = fig.add_axes([0.06, 0.69, 0.88, 0.17])
        axis.axis("off")
        table = axis.table(
            cellText=rows,
            colLabels=["Model", "F1", "Precision", "Recall", "PR-AUC"],
            loc="center",
            cellLoc="center",
            colWidths=[0.34, 0.16, 0.16, 0.16, 0.16],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.8)
        paragraph(
            fig,
            f"The validation-selected classifier is {registry['winner']}. Model {registry['name']}, version {registry['version']}, is verified in remote Staging. Its serialized classifier preserves the selected decision threshold ({registry['decision_threshold']:.2f}). Test results are also broken down by lead time and river reach.",
            0.64,
        )
        screenshot(fig, evidence / "registry.png", [0.05, 0.12, 0.9, 0.39])
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("Reproducibility and DVC pipeline", 4)
        paragraph(
            fig,
            "Named DVC stages ingest thresholds and source observations, validate data, engineer features, train models, evaluate the hold-out period and register the validation winner. Source configurations, dependencies and stage outputs define the dependency graph. Fixed model seeds and a pinned environment support repeatable runs; DVC caches and the configured remote preserve dataset and model versions.",
            0.87,
        )
        screenshot(fig, evidence / "dvc_dag.png", [0.12, 0.27, 0.76, 0.43])
        paragraph(
            fig,
            "The pipeline runs with make run, which executes dvc repro. make push uploads tracked outputs to the configured DVC remote. make test verifies validation, future-label alignment, chronology, sequence shapes, feature causality, estimator predictions and orchestration using isolated synthetic test fixtures. These fixtures are excluded from the reported real-data results.",
            0.21,
        )
        pdf.savefig(fig)
        plt.close(fig)
    print(destination)


if __name__ == "__main__":
    main()
