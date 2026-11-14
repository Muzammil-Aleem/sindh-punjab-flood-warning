import base64
import os
import subprocess

import requests
from dotenv import load_dotenv


def run(*arguments, **kwargs):
    return subprocess.run(arguments, check=True, **kwargs)


def main():
    load_dotenv()
    token = os.environ["GITHUB_TOKEN"]
    response = requests.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {token}"},
        timeout=30,
    )
    if not response.ok:
        raise RuntimeError(f"GitHub identity verification failed: HTTP {response.status_code}")
    user = response.json()
    login = user["login"]
    origin = run(
        "git", "remote", "get-url", "origin", capture_output=True, text=True
    ).stdout.strip()
    expected = f"https://github.com/{login}/sindh-punjab-flood-warning.git"
    if origin != expected:
        raise ValueError("Git origin does not match the authenticated project owner")
    for field, value in {
        "user.name": user.get("name") or login,
        "user.email": f"{user['id']}+{login}@users.noreply.github.com",
    }.items():
        current = subprocess.run(["git", "config", "--get", field], capture_output=True, text=True)
        if current.returncode:
            run("git", "config", "--local", field, value)
    paths = [
        "README.md",
        ".env.example",
        ".python-version",
        ".gitignore",
        ".dvc",
        ".dvcignore",
        "Makefile",
        "dvc.yaml",
        "dvc.lock",
        "params.yaml",
        "pyproject.toml",
        "requirements.txt",
        "requirements.lock",
        "requirements-evidence.txt",
        "flood",
        "scripts",
        "tests",
        "submission/flood-warning.pdf",
        "submission/flood-warning.docx",
    ]
    run("git", "add", "--", *paths)
    staged = run(
        "git", "diff", "--cached", "--name-only", capture_output=True, text=True
    ).stdout.splitlines()
    if any(
        path.startswith(("data/", "models/", ".env", ".dvc/config.local"))
        and path != ".env.example"
        for path in staged
    ):
        raise ValueError("Refusing to publish data, model weights or local credentials")
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"])
    if changed.returncode == 1:
        run("git", "commit", "-m", "Build reproducible flood classification pipeline")
    elif changed.returncode != 0:
        raise RuntimeError("Unable to inspect staged changes")
    environment = os.environ.copy()
    authorization = base64.b64encode(f"{login}:{token}".encode()).decode()
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_COUNT": "2",
            "GIT_CONFIG_KEY_0": "credential.helper",
            "GIT_CONFIG_VALUE_0": "",
            "GIT_CONFIG_KEY_1": "http.https://github.com/.extraheader",
            "GIT_CONFIG_VALUE_1": f"AUTHORIZATION: basic {authorization}",
        }
    )
    run("git", "push", "--set-upstream", "origin", "HEAD", env=environment)


if __name__ == "__main__":
    main()
