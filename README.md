# Flood Early Warning for Sindh and Punjab

This project looks at flood risk along four Indus river reaches: Taunsa, Guddu, Sukkur and Kotri. It combines daily river discharge, rainfall, temperature and soil moisture to predict whether discharge will exceed a GloFAS two-year return-period threshold three to seven days ahead.

The current milestone covers data collection, validation, feature engineering, model training, evaluation and registration. The pipeline runs locally, while DagsHub stores the DVC data versions and MLflow experiments. Deployment and the warning service are planned for a later milestone.

## Data

The dataset covers 2010–2022 and contains 18,992 reach-day records.

| Source | What we use | How it is collected |
| --- | --- | --- |
| GloFAS v4 | Daily discharge and two-year return-period thresholds | Copernicus GRIB downloads and the official ECMWF threshold file |
| CHIRPS | Daily rainfall | ClimateSERV point requests |
| ERA5 | Temperature and topsoil moisture | Open-Meteo historical API, with ERA5 selected explicitly and elevation downscaling disabled |

The direct Copernicus ERA5 download adapter is also available through `era5_provider: cds` in `params.yaml`. The default uses the point archive to avoid downloading large hourly grids for four locations.

Before training, validation checks the schema, daily coverage, duplicate records, nulls, physical ranges and threshold provenance. Point observations and processed datasets are versioned in DVC. Downloaded GRIB archives stay in a reusable local cache, and their checksums are recorded in the source manifest.

## Setup

You need Git, Make and a working `python3` installation to bootstrap the environment. The Makefile installs the project's pinned Python 3.11.14 environment and dependencies.

```bash
git clone https://github.com/Muzammil-Aleem/sindh-punjab-flood-warning.git
cd sindh-punjab-flood-warning
make install
cp .env.example .env
```

Fill in `.env` with your own credentials. The existing private project requires collaborator access.

```dotenv
CDSAPI_KEY=your_copernicus_cds_key
EWDS_API_KEY=your_copernicus_ewds_key
MLFLOW_TRACKING_URI=https://dagshub.com/Muzammil-Aleem/sindh-punjab-flood-warning.mlflow
MLFLOW_TRACKING_USERNAME=your_dagshub_username
MLFLOW_TRACKING_PASSWORD=your_dagshub_token
DVC_REMOTE_URL=https://dagshub.com/Muzammil-Aleem/sindh-punjab-flood-warning.dvc
```

Accept the relevant Copernicus dataset licences before making downloads. `GITHUB_TOKEN` and `DAGSHUB_TOKEN` are used by the repository setup and publishing scripts; they are not needed for normal training when tracking and storage are already configured. Keep `.env` out of Git.

```bash
make configure
make check
make run
```

`make configure` saves DVC authentication in the ignored `.dvc/config.local` file. `make run` calls `dvc repro` and runs the stages that need updating. A first download can take a while because Copernicus requests are queued. Completed downloads are cached, and interrupted Copernicus requests retain their job IDs so they can be resumed.

To restore existing versioned outputs instead of downloading and training again:

```bash
.venv/bin/dvc pull
make run
```

MLflow remains a remote service, so training, evaluation and registration need network access and valid tracking credentials.

## Pipeline and models

The stages in `dvc.yaml` are `thresholds`, `ingest`, `validate`, `features`, `train`, `evaluate` and `register`.

Features include lags, rolling discharge and rainfall sums, discharge changes and seasonality. Training uses 2010–2014, validation uses 2015–2021, and 2022 is held out for testing. Targets crossing split boundaries are removed. Scaling is fitted on training data, and decision thresholds are selected on validation F1.

The current comparison includes a persistence baseline, logistic regression and random forest. The two trained classifiers use class weights. The pipeline also exports 30-day sequence windows for the later LSTM and Transformer work; those models have not been trained in this milestone.

Each MLflow run logs parameters, metrics, model artifacts, dependency versions and data fingerprints. The trained classifier with the highest validation F1 is registered in Staging, with its selected decision threshold preserved in the saved model.

## Current results

These are the 2022 test results pooled across the five lead times.

| Model | F1 | Precision | Recall | PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| Persistence | 0.4444 | 0.4444 | 0.4444 | 0.2010 |
| Logistic regression | 0.0919 | 0.0499 | 0.5778 | 0.0519 |
| Random forest | 0.4000 | 0.4000 | 0.4000 | 0.3071 |

Random forest was selected on validation F1 and registered as version 1 in Staging. Persistence performed better on test F1. We kept the validation-based selection rather than changing the winner after looking at the test results.

Only Kotri exceeds the fixed threshold in the 2022 data. Reports include results by reach and lead time, and mark PR-AUC as undefined for single-class groups. These labels come from GloFAS reanalysis, not independently observed inundation. The results should be read as a retrospective classification experiment, not evidence that an operational warning system is ready.

## Repository layout

```text
flood/                    Data, features, training, evaluation and registration
tests/                    Validation, feature, model and pipeline tests
scripts/                  Setup, publishing and submission document scripts
params.yaml               Sources, reaches, dates, features and model settings
dvc.yaml                  Pipeline stages
dvc.lock                  Versioned stage inputs and outputs
requirements.lock         Pinned runtime dependencies
requirements-evidence.txt Optional screenshot and Word document tools
submission/               Submission PDF and editable Word document
```

Running the pipeline creates `data/`, `models/` and `reports/`. These are excluded from Git; the pipeline's versioned outputs are stored through DVC.

## Useful commands

```bash
make test          # Run the tests
make lint          # Check Python source formatting rules
make dag           # Display the DVC dependency graph
make push          # Upload DVC outputs to the configured remote
make evidence      # Capture experiment, registry and pipeline screenshots
make submission    # Generate the four-page PDF
make document      # Generate the editable Word document
```

Document generation uses the local reports and evidence files. `make evidence` needs remote MLflow access and installs its browser tools separately from the training environment. Fill in the group names and roll numbers in the Word document before exporting the final submission PDF.

[DagsHub project](https://dagshub.com/Muzammil-Aleem/sindh-punjab-flood-warning) · [MLflow tracking server](https://dagshub.com/Muzammil-Aleem/sindh-punjab-flood-warning.mlflow)
