import os
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from dvc.repo import Repo


def run(*arguments):
    if arguments[0] == "dvc":
        arguments = (str(Path(sys.executable).parent / "dvc"), *arguments[1:])
    subprocess.run(arguments, check=True)


def main():
    load_dotenv()
    remote = os.environ.get("DVC_REMOTE_URL", "")
    parsed = urlparse(remote)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError("DVC_REMOTE_URL must be a DagsHub HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError("DVC remote credentials must be supplied separately")
    run("dvc", "remote", "add", "--force", "--default", "storage", remote)
    if parsed.scheme == "https":
        with Repo().config.edit(level="local") as configuration:
            configuration.setdefault("remote", {}).setdefault("storage", {}).update(
                {
                    "auth": "basic",
                    "user": os.environ["MLFLOW_TRACKING_USERNAME"],
                    "password": os.environ["MLFLOW_TRACKING_PASSWORD"],
                }
            )
        Path(".dvc/config.local").chmod(0o600)


if __name__ == "__main__":
    main()
