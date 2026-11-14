import base64
import html
import json
import os
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


def main():
    load_dotenv()
    root = Path(__file__).resolve().parents[1]
    tracking = os.environ["MLFLOW_TRACKING_URI"].rstrip("/")
    if urllib.parse.urlparse(tracking).hostname != "dagshub.com":
        raise ValueError("Screenshot capture requires the configured DagsHub host")
    username = os.environ["MLFLOW_TRACKING_USERNAME"]
    password = os.environ["MLFLOW_TRACKING_PASSWORD"]
    training = json.loads((root / "reports/training.json").read_text())
    registry = json.loads((root / "reports/registry.json").read_text())
    run_id = training["models"][training["winner"]]["run_id"]
    authorization = base64.b64encode(f"{username}:{password}".encode()).decode()
    request = urllib.request.Request(
        tracking + "/api/2.0/mlflow/runs/get?" + urllib.parse.urlencode({"run_id": run_id}),
        headers={"Authorization": "Basic " + authorization},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        experiment_id = json.load(response)["run"]["info"]["experiment_id"]
    destination = root / "submission/evidence"
    destination.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(
            viewport={"width": 1600, "height": 1000},
            device_scale_factor=1,
            http_credentials={
                "username": username,
                "password": password,
                "origin": "https://dagshub.com",
            },
        )
        page = context.new_page()
        page.goto(
            tracking + f"/#/experiments/{experiment_id}",
            wait_until="domcontentloaded",
            timeout=120000,
        )
        page.get_by_text("persistence", exact=False).first.wait_for(timeout=120000)
        page.get_by_text("random_forest", exact=False).first.wait_for(timeout=120000)
        page.screenshot(path=str(destination / "runs.png"), full_page=True)
        model = urllib.parse.quote(registry["name"], safe="")
        page.goto(
            tracking + f"/#/models/{model}/versions/{registry['version']}",
            wait_until="domcontentloaded",
            timeout=120000,
        )
        page.get_by_text("Staging", exact=False).first.wait_for(timeout=120000)
        page.screenshot(path=str(destination / "registry.png"), full_page=True)
        command = subprocess.run(
            [str(root / ".venv/bin/dvc"), "dag"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        diagram = html.escape("$ dvc dag\n\n" + command.stdout)
        page.set_content(
            f"<html><body style='margin:40px;background:#111827;color:#f9fafb'><pre style='font-size:20px;line-height:1.1'>{diagram}</pre></body></html>"
        )
        page.screenshot(path=str(destination / "dvc_dag.png"), full_page=True)
        browser.close()
    print(destination)


if __name__ == "__main__":
    main()
