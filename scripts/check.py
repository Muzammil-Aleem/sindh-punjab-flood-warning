import json
import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def main():
    load_dotenv()
    names = [
        "CDSAPI_KEY",
        "EWDS_API_KEY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "DVC_REMOTE_URL",
    ]
    missing = [name for name in names if not os.environ.get(name)]
    invalid = []
    for name in ["MLFLOW_TRACKING_URI", "DVC_REMOTE_URL"]:
        value = os.environ.get(name)
        if value:
            parsed = urlparse(value)
            allowed = {"https"}
            if (
                parsed.scheme not in allowed
                or not parsed.netloc
                or parsed.username
                or parsed.password
            ):
                invalid.append(name)
    status = {
        "missing_environment": missing,
        "invalid_environment": invalid,
        "git_initialized": Path(".git").exists(),
        "dvc_initialized": Path(".dvc/config").exists(),
        "raw_data_available": Path("data/raw/series.parquet").exists(),
        "trained_models_available": Path("reports/training.json").exists(),
        "registry_receipt_available": Path("reports/registry.json").exists(),
    }
    print(json.dumps(status, indent=2))
    if missing or invalid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
