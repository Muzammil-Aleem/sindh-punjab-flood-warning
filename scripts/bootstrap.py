import os
import subprocess
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv


def checked(response):
    if not response.ok:
        raise RuntimeError(f"Remote setup failed with HTTP {response.status_code}")
    return response.json()


def main():
    load_dotenv()
    name = "sindh-punjab-flood-warning"
    github_token = os.environ["GITHUB_TOKEN"]
    dagshub_token = os.environ["DAGSHUB_TOKEN"]
    if not github_token or not dagshub_token:
        raise ValueError("GITHUB_TOKEN and DAGSHUB_TOKEN must be populated in .env")
    github_headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    github_user = checked(
        requests.get("https://api.github.com/user", headers=github_headers, timeout=30)
    )["login"]
    github_url = f"https://api.github.com/repos/{github_user}/{name}"
    response = requests.get(github_url, headers=github_headers, timeout=30)
    if response.status_code == 404:
        checked(
            requests.post(
                "https://api.github.com/user/repos",
                headers=github_headers,
                json={"name": name, "private": True, "auto_init": False},
                timeout=30,
            )
        )
    else:
        checked(response)
    dagshub_headers = {"Authorization": f"Bearer {dagshub_token}"}
    dagshub_user = checked(
        requests.get("https://dagshub.com/api/v1/user", headers=dagshub_headers, timeout=30)
    )["login"]
    response = requests.get(
        f"https://dagshub.com/api/v1/repos/{dagshub_user}/{name}",
        headers=dagshub_headers,
        timeout=30,
    )
    if response.status_code == 404:
        checked(
            requests.post(
                "https://dagshub.com/api/v1/user/repos",
                headers=dagshub_headers,
                json={"name": name, "private": True, "auto_init": False},
                timeout=30,
            )
        )
    else:
        checked(response)
    from dotenv import set_key

    environment = Path(".env")
    environment.touch(mode=0o600, exist_ok=True)
    environment.chmod(0o600)
    values = {
        "GITHUB_REPOSITORY": f"{github_user}/{name}",
        "MLFLOW_TRACKING_URI": f"https://dagshub.com/{dagshub_user}/{name}.mlflow",
        "MLFLOW_TRACKING_USERNAME": dagshub_user,
        "MLFLOW_TRACKING_PASSWORD": dagshub_token,
        "DVC_REMOTE_URL": f"https://dagshub.com/{dagshub_user}/{name}.dvc",
    }
    for key, value in values.items():
        set_key(environment, key, value)
    os.environ.update(values)
    origin = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True)
    if origin.returncode:
        subprocess.run(
            ["git", "remote", "add", "origin", f"https://github.com/{github_user}/{name}.git"],
            check=True,
        )
    subprocess.run([sys.executable, "scripts/configure.py"], check=True)
    print(values["GITHUB_REPOSITORY"])
    print(values["MLFLOW_TRACKING_URI"])


if __name__ == "__main__":
    main()
