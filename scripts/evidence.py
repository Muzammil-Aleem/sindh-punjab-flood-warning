import base64
import html
import json
import mimetypes
import os
import subprocess
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path

from dotenv import load_dotenv
from playwright.sync_api import sync_playwright


class RemoteViewer:
    def __init__(self, root, tracking, authorization):
        self.build = next(
            (root / ".venv/lib").glob("python*/site-packages/mlflow/server/js/build")
        ).resolve()
        self.tracking = tracking
        self.authorization = authorization

    def handle(self, route):
        parsed = urllib.parse.urlparse(route.request.url)
        if parsed.hostname != "mlflow-viewer.local":
            route.abort()
            return
        path = parsed.path
        if path.startswith(("/api/", "/ajax-api/")):
            path = path.replace("/ajax-api/", "/api/", 1)
            operation = path.rstrip("/").rsplit("/", 1)[-1]
            allowed = route.request.method == "GET" or (
                route.request.method == "POST"
                and operation
                in {"search", "get-metric-history-bulk", "get-metric-history-bulk-interval"}
            )
            if not allowed:
                route.fulfill(status=403, body="Read-only evidence viewer")
                return
            request = urllib.request.Request(
                self.tracking + path + ("?" + parsed.query if parsed.query else ""),
                data=route.request.post_data_buffer,
                headers={
                    "Authorization": "Basic " + self.authorization,
                    "Content-Type": "application/json",
                },
                method=route.request.method,
            )
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    route.fulfill(
                        status=response.status,
                        body=response.read(),
                        content_type=response.headers.get_content_type(),
                    )
            except urllib.error.HTTPError as error:
                route.fulfill(status=error.code, body=error.read(), content_type="application/json")
            return
        relative = "index.html" if path == "/" else path.removeprefix("/static-files/")
        candidate = (self.build / relative).resolve()
        if not candidate.is_relative_to(self.build) or not candidate.is_file():
            route.fulfill(status=404, body="Not found")
            return
        route.fulfill(
            status=200,
            body=candidate.read_bytes(),
            content_type=mimetypes.guess_type(candidate.name)[0] or "application/octet-stream",
        )


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
        context.route(
            "https://dagshub.com/**",
            lambda route: route.continue_(
                headers={**route.request.headers, "authorization": "Basic " + authorization}
            ),
        )
        page = context.new_page()
        response = page.goto(
            tracking + f"/#/experiments/{experiment_id}",
            wait_until="domcontentloaded",
            timeout=120000,
        )
        viewer = "hosted"
        base = tracking
        if not response.ok:
            viewer = "official_mlflow_client"
            base = "http://mlflow-viewer.local"
            adapter = RemoteViewer(root, tracking, authorization)
            context.route("**/*", adapter.handle)
            page.goto(
                base + f"/#/experiments/{experiment_id}",
                wait_until="domcontentloaded",
                timeout=120000,
            )
            print("Using the official MLflow frontend with the hosted DagsHub APIs", flush=True)
        page.get_by_text("persistence", exact=False).first.wait_for(timeout=120000)
        page.get_by_text("random_forest", exact=False).first.wait_for(timeout=120000)
        page.screenshot(path=str(destination / "runs.png"), full_page=True)
        model = urllib.parse.quote(registry["name"], safe="")
        page.goto(
            base + f"/#/models/{model}/versions/{registry['version']}",
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
    (destination / "capture.json").write_text(
        json.dumps(
            {
                "viewer": viewer,
                "tracking_uri": tracking,
                "source": "MLflow 2.22.2 official frontend",
                "local_tracking_store": False,
            },
            indent=2,
        )
        + "\n"
    )
    print(destination)


if __name__ == "__main__":
    main()
