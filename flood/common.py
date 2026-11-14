import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml
from dotenv import load_dotenv


def config():
    load_dotenv()
    return yaml.safe_load(Path("params.yaml").read_text())


def required(name):
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Required environment variable is missing: {name}")
    return value


def remote_uri(name):
    value = required(name)
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{name} must be an HTTPS remote URI")
    if parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        raise ValueError(f"{name} must use a remote server")
    if parsed.username or parsed.password:
        raise ValueError(f"Supply credentials separately from {name}")
    return value


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
