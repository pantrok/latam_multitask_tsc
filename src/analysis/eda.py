from __future__ import annotations

"""Descriptive per-dataset profiling. Ported from origin's sensor_eda.py."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset_list import get_sensor_datasets
from src.data.ucr_io import load_ucr_dataset


def summarize_sensor_dataset(name: str, x_train: np.ndarray, y_train: np.ndarray,
                              x_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    X_train = x_train[:, 0, :]
    X_test = x_test[:, 0, :]
    series_length = X_train.shape[1]
    train_flat = X_train.ravel()
    test_flat = X_test.ravel()

    diff_train = np.diff(X_train, axis=1)
    mean_abs_change = float(np.nanmean(np.abs(diff_train))) if diff_train.size else np.nan

    classes_train, counts_train = np.unique(y_train, return_counts=True)
    classes_test, counts_test = np.unique(y_test, return_counts=True)

    return {
        "dataset": name,
        "train_n": int(len(X_train)),
        "test_n": int(len(X_test)),
        "n_classes": int(len(classes_train)),
        "series_length": int(series_length),
        "train_mean": float(np.nanmean(train_flat)),
        "train_std": float(np.nanstd(train_flat)),
        "test_mean": float(np.nanmean(test_flat)),
        "test_std": float(np.nanstd(test_flat)),
        "mean_abs_change": mean_abs_change,
        "mean_shift_train_test": float(np.nanmean(train_flat) - np.nanmean(test_flat)),
        "train_class_imbalance": float(counts_train.max() / counts_train.min()) if counts_train.min() > 0 else 1.0,
        "train_to_test_ratio": float(len(X_train) / len(X_test)) if len(X_test) else float("nan"),
    }


def run(dataset_root: str = "data/raw/UCRArchive_2018", out_dir: Path = Path("artifacts/eda")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = get_sensor_datasets("assets/DataSummary.csv")
    rows = []
    for name in names:
        x_train, y_train, x_test, y_test = load_ucr_dataset(dataset_root, name)
        rows.append(summarize_sensor_dataset(name, x_train, y_train, x_test, y_test))
    df = pd.DataFrame(rows).sort_values("dataset")
    out_path = out_dir / "dataset_profiles.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} datasets)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Descriptive EDA over the 30 Sensor datasets.")
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.dataset_root)


if __name__ == "__main__":
    main()
