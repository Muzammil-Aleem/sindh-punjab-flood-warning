from pathlib import Path

from flood.common import config, digest, write_json
from flood.ingest import download, extract_thresholds


def ingest_thresholds():
    data = config()["data"]
    destination = Path("data/auxiliary")
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "thresholds.nc"
    if path.exists() and digest(path) != data["threshold_sha256"]:
        path.unlink()
    if not path.exists():
        download(data["threshold_url"], path)
    if digest(path) != data["threshold_sha256"]:
        raise ValueError("GloFAS return-period threshold SHA256 mismatch")
    frame = extract_thresholds(path, data)
    if not frame.threshold.between(0.001, 150000).all():
        raise ValueError("Configured reach does not map to a valid GloFAS threshold cell")
    frame.to_csv(destination / "thresholds.csv", index=False)
    write_json(
        destination / "manifest.json",
        {
            "url": data["threshold_url"],
            "sha256": digest(path),
            "return_period": data["threshold_return_period"],
            "glofas_version": data["glofas_version"],
        },
    )


if __name__ == "__main__":
    ingest_thresholds()
