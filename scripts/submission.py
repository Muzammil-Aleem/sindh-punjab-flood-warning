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


def paragraph(fig, value, y, size=10, url=None):
    fig.text(
        0.08,
        y,
        "\n".join(textwrap.wrap(value, width=95)),
        va="top",
        fontsize=size,
        linespacing=1.5,
        url=url,
    )


def page(title, number):
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.08, 0.94, title, fontsize=18, weight="bold", va="top", color="#1f4e79")
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
    splits = json.loads(Path("reports/splits.json").read_text())
    evidence = Path("submission/evidence")
    capture = json.loads((evidence / "capture.json").read_text())
    viewer_note = ""
    if capture["viewer"] == "official_mlflow_client":
        viewer_note = " Screenshots use the official MLflow frontend connected read-only to the hosted DagsHub APIs; no local tracking datastore is used."
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
        paragraph(fig, github, 0.87, url=github)
        dagshub = tracking.removesuffix(".mlflow")
        paragraph(fig, dagshub, 0.83, url=dagshub)
        paragraph(fig, tracking, 0.79, url=tracking)
        fig.text(0.08, 0.755, "Group members", fontsize=11, weight="bold", color="#1f4e79")
        members_axis = fig.add_axes([0.08, 0.65, 0.84, 0.08])
        members_axis.axis("off")
        members_table = members_axis.table(
            cellText=[["", ""] for _ in range(4)],
            colLabels=["Name", "Roll No."],
            cellLoc="left",
            colWidths=[0.7, 0.3],
            bbox=[0, 0, 1, 1],
        )
        members_table.auto_set_font_size(False)
        members_table.set_fontsize(8)
        paragraph(
            fig,
            f"The core pipeline processes daily discharge, rainfall, temperature and soil moisture for {validation['reaches']} Indus river reaches. The validated dataset contains {validation['rows']} reach-day records spanning {validation['start']} to {validation['end']}. Predictions classify GloFAS two-year discharge threshold exceedances at exact lead times of three through seven days.",
            0.62,
        )
        paragraph(
            fig,
            "GloFAS historical reanalysis is retrieved as GRIB through Copernicus. Hourly ERA5 temperature and topsoil moisture use Open-Meteo's ERA5 archive (https://open-meteo.com/en/docs/historical-weather-api), with ERA5 explicitly selected and elevation downscaling disabled. CHIRPS point series use ClimateSERV. Official ECMWF thresholds are pinned by URL version and SHA256. All observations are aligned to UTC calendar days.",
            0.49,
        )
        paragraph(
            fig,
            "Validation rejects schema changes, missing or infinite observations, duplicate reach-days, gaps in daily coverage, invalid physical ranges and inconsistent threshold versions. Source point observations, validated series, engineered datasets and trained artifacts are tracked as DVC pipeline outputs. Original GRIB archives are retained in a reusable download cache and fingerprinted in the source manifest; credentials, data and model weights are excluded from Git.",
            0.34,
        )
        paragraph(
            fig,
            "This is a retrospective classification study using reanalysis and model-derived labels. Its results do not establish operational warning skill or agreement with independently observed inundation. A comparison against archived GloFAS forecasts remains outside this core deliverable.",
            0.18,
        )
        pdf.savefig(fig)
        plt.close(fig)
        fig = page("Features and experiment tracking", 2)
        paragraph(
            fig,
            f"The pipeline builds per-reach lag, rolling rainfall and discharge sums, discharge changes and seasonal features, plus 30-day sequence tensors with five labels. Training ends {splits['train']['last_target']}; validation ends {splits['validation']['last_target']}; testing begins {splits['test']['first_issue']}. Targets crossing split boundaries are purged. Scalers fit only on training data; decision thresholds and the winning classifier are selected on validation F1.",
            0.87,
        )
        screenshot(fig, evidence / "runs.png", [0.05, 0.28, 0.9, 0.42])
        paragraph(
            fig,
            "Persistence, class-weighted logistic regression and a class-weighted random forest are evaluated. Each remote MLflow run records estimator parameters, split settings, dataset fingerprints, validation and test metrics, the model signature, dependency versions, source manifests and evaluation plots."
            + viewer_note,
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
            f"The validation-selected classifier is {registry['winner']}. Model {registry['name']}, version {registry['version']}, is verified in remote Staging, preserving decision threshold {registry['decision_threshold']:.2f}. Persistence outperforms the registered classifier on 2022 test F1 ({evaluation['models']['persistence']['test']['f1']:.3f} versus {evaluation['models'][registry['winner']]['test']['f1']:.3f}); promotion uses validation results. Only Kotri exceeds the fixed thresholds in 2022. Per-reach reports mark PR-AUC undefined for single-class groups.",
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
