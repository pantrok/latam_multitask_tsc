from __future__ import annotations

"""
Two distinct kinds of baseline, kept in separate artifacts:

1. Published reference results (HIVE-COTE 2.0, ResNet, ROCKET, InceptionTime),
   fetched via aeon's ReferenceResults loader -- not locally fitted models.
   Stored as "ROCKET_published" to keep it unambiguous against the
   locally-fitted MiniRocket below. aeon's reference table covers 20 of the
   30 Sensor datasets used in this study; the other 10 (mostly
   *GestureWiimote*/*GesturePebble*/DodgerLoop*) have no reference row and
   are left blank rather than guessed.

2. Locally-fitted MiniRocket, over all 30 datasets, original UCR split, no
   resampling -- a genuine local fit, not a reference lookup. Kept in
   `baselines_local_minirocket.csv`, never merged into
   `baselines_reference.csv`.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ESTIMATORS = ["HC2", "ResNet", "InceptionTime", "ROCKET"]


def fetch_live(dataset_names: list[str], out_path: Path) -> pd.DataFrame:
    from aeon.benchmarking.results_loaders import get_estimator_results

    results = get_estimator_results(
        estimators=ESTIMATORS, datasets=dataset_names, task="classification",
        measure="accuracy", num_resamples=1,
    )
    table = pd.DataFrame(results).reset_index().rename(columns={"index": "dataset"})
    table.to_csv(out_path, index=False, encoding="utf-8")
    return table


def run(
    dataset_names: list[str],
    checked_in_csv: Path = Path("assets/acc_sensor_hc2_resnet_refs.csv"),
    out_dir: Path = Path("artifacts/results"),
    try_live: bool = True,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    live_path = out_dir / "acc_sensor_hc2_resnet_refs_live.csv"

    table = None
    if try_live:
        try:
            table = fetch_live(dataset_names, live_path)
            print(f"Fetched live reference results from aeon -> {live_path}")
        except Exception as e:
            print(f"[warn] live aeon fetch failed ({e}); falling back to checked-in {checked_in_csv}")

    if table is None:
        table = pd.read_csv(checked_in_csv)

    table["dataset"] = table["dataset"].astype(str).str.strip()
    missing = sorted(set(dataset_names) - set(table["dataset"]))
    if missing:
        print(f"[warn] {len(missing)} datasets have no published reference row (kept blank): {missing}")

    if "ROCKET" in table.columns:
        table = table.rename(columns={"ROCKET": "ROCKET_published"})

    table["source"] = "aeon.benchmarking.results_loaders.get_estimator_results (published reference, resample 0)"
    ref_path = out_dir / "baselines_reference.csv"
    table.to_csv(ref_path, index=False)
    print(f"Wrote {ref_path}")

    if "HC2" in table.columns:
        hc2 = table[["dataset", "HC2"]].rename(columns={"HC2": "acc"}).dropna()
        hc2_path = out_dir / "hc2_baseline.csv"
        hc2.to_csv(hc2_path, index=False)
        print(f"Wrote {hc2_path} ({len(hc2)} datasets) for use as --baseline in the correlation stage")


def fit_minirocket_local(
    dataset_names: list[str],
    dataset_root: str = "data/raw/UCRArchive_2018",
    out_dir: Path = Path("artifacts/results"),
    random_state: int = 42,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Fits aeon's MiniRocketClassifier on each dataset's own original UCR
    train/test split (no resampling, no validation carve-out -- MiniRocket
    isn't early-stopped, so there's nothing to validate against). A failure
    on one dataset (e.g. a variable-length series MiniRocket can't handle)
    is caught, logged, and left out of the CSV -- never silently imputed or
    substituted with a workaround."""
    from aeon.classification.convolution_based import MiniRocketClassifier

    from src.data.ucr_io import load_ucr_dataset

    rows = []
    failed = []
    for name in dataset_names:
        t0 = time.time()
        try:
            x_train, y_train, x_test, y_test = load_ucr_dataset(dataset_root, name)
            clf = MiniRocketClassifier(random_state=random_state, n_jobs=n_jobs)
            clf.fit(x_train, y_train)
            preds = clf.predict(x_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average="macro")
            fit_seconds = time.time() - t0
            rows.append({"dataset": name, "acc": acc, "f1_macro": f1, "fit_seconds": fit_seconds})
            print(f"[minirocket-local] {name}: acc={acc:.4f} f1={f1:.4f} ({fit_seconds:.1f}s)")
        except Exception as e:
            failed.append((name, str(e)))
            print(f"[minirocket-local] FAILED on {name}: {e}")

    if failed:
        print(f"[warn] MiniRocket failed on {len(failed)} dataset(s), not imputed, left out of the CSV: "
              f"{[n for n, _ in failed]}")

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "baselines_local_minirocket.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)}/{len(dataset_names)} datasets fit successfully)")
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch/refresh published TSC reference baselines, or fit MiniRocket locally.")
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--no-live", action="store_true", help="Skip live aeon fetch, use checked-in CSV only.")
    p.add_argument("--local-minirocket", action="store_true", help="Fit MiniRocket locally instead of fetching references.")
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--out-dir", type=str, default="artifacts/results")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs", type=int, default=-1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.local_minirocket:
        fit_minirocket_local(args.dataset_names, args.dataset_root, Path(args.out_dir), args.random_state, args.n_jobs)
    else:
        run(args.dataset_names, try_live=not args.no_live)


if __name__ == "__main__":
    main()
