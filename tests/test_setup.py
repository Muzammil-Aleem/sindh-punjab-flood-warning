import os
from types import SimpleNamespace

from scripts.bootstrap import main


def test_bootstrap_passes_generated_configuration_to_child(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ["GITHUB_TOKEN", "DAGSHUB_TOKEN"]:
        monkeypatch.setenv(name, "synthetic-test-token")
    generated = [
        "GITHUB_REPOSITORY",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_TRACKING_USERNAME",
        "MLFLOW_TRACKING_PASSWORD",
        "DVC_REMOTE_URL",
    ]
    for name in generated:
        monkeypatch.setenv(name, "")

    class Response:
        ok = True
        status_code = 200

        def json(self):
            return {"login": "test-owner"}

    calls = []

    def process(arguments, **kwargs):
        calls.append(arguments)
        if "scripts/configure.py" in arguments:
            assert (
                os.environ["DVC_REMOTE_URL"]
                == "https://dagshub.com/test-owner/sindh-punjab-flood-warning.dvc"
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.bootstrap.requests.get", lambda *args, **kwargs: Response())
    monkeypatch.setattr("scripts.bootstrap.subprocess.run", process)
    main()
    assert len(calls) == 2
