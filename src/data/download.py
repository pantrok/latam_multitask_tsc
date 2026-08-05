from __future__ import annotations

import os
import zipfile
from pathlib import Path

import requests

from src.data.dataset_list import get_sensor_datasets

UCR_ARCHIVE_ZIP_URL = "https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip"


def download_file(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f"Download failed ({r.status_code}) for: {url}")

    with dst.open("wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)


def extract_sensor_only(zip_path: Path, out_dir: Path, sensor_names: list[str], password: str) -> None:
    pwd = password.encode("utf-8")
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        wanted_prefixes = [f"UCRArchive_2018/{name}/" for name in sensor_names]
        to_extract = [m for m in members if any(m.startswith(p) for p in wanted_prefixes)]

        if not to_extract:
            raise RuntimeError(
                "No files matched for Sensor datasets. "
                "Possible causes: CSV names differ from archive folder names, or archive structure changed."
            )

        for m in to_extract:
            zf.extract(m, path=out_dir, pwd=pwd)


def verify(out_dir: Path, sensor_names: list[str]) -> list[str]:
    base = out_dir / "UCRArchive_2018"
    missing = []
    for name in sensor_names:
        d = base / name
        if not (d / f"{name}_TRAIN.tsv").exists() or not (d / f"{name}_TEST.tsv").exists():
            missing.append(name)
    return missing


def run(dataset_root: Path = Path("data/raw/UCRArchive_2018"), force: bool = False) -> None:
    sensor_names = get_sensor_datasets("assets/DataSummary.csv")
    print(f"Sensor datasets expected: {len(sensor_names)}")

    if not force:
        missing = verify(dataset_root.parent, sensor_names)
        if not missing:
            print(f"Already downloaded and verified in: {dataset_root}. Skipping (use --force to redo).")
            return
        print(f"Incomplete or missing ({len(missing)} datasets); downloading.")

    password = os.environ.get("UCR2018_ZIP_PASSWORD")
    if not password:
        raise SystemExit(
            "Set env var UCR2018_ZIP_PASSWORD (published in BriefingDocument2018.pdf on the UCR 2018 page)."
        )

    zip_path = Path("data/raw/UCRArchive_2018.zip")
    if not zip_path.exists() or force:
        print("Downloading UCRArchive_2018.zip ...")
        download_file(UCR_ARCHIVE_ZIP_URL, zip_path)

    print("Extracting Sensor datasets only ...")
    extract_sensor_only(zip_path, Path("data/raw"), sensor_names, password)

    print("Verifying ...")
    missing = verify(Path("data/raw"), sensor_names)
    if missing:
        raise RuntimeError(f"Verification failed for datasets (first 10): {missing[:10]}")
    print(f"Verified {len(sensor_names)} Sensor datasets in: {dataset_root}")


if __name__ == "__main__":
    run()
