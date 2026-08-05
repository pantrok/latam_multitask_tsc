from __future__ import annotations

"""
Seed-level dispersion (mean, std, CV%, 95% CI) for C3-head-BN, C3-head,
C3-full, and C4, matching how C2/C5's own dispersion is already computed in
evaluate.py's _seed_level_stats: for each seed, the mean accuracy over all
30 datasets; then mean/std/CV/CI across those per-seed means.

Pure aggregation over already-computed raw per-seed results -- no new runs.

C3-head-BN is not computed for the LSTM: the LSTM backbone has no
BatchNorm layers, so "recalibrated normalization statistics" is undefined
for it. That row is omitted rather than filled with a degenerate number.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _stats(x: np.ndarray) -> dict:
    n = len(x)
    mean = float(x.mean())
    std = float(x.std(ddof=1)) if n > 1 else 0.0
    return {
        "n_seeds": n, "mean": mean, "std": std,
        "cv_pct": 100.0 * std / mean if mean else 0.0,
        "ci95": 1.96 * std / np.sqrt(n) if n > 0 else 0.0,
    }


def _per_seed_global_acc(df: pd.DataFrame) -> np.ndarray:
    return df.groupby("seed")["test_acc"].mean().to_numpy()


def run(results_dir: Path, out_path: Path) -> pd.DataFrame:
    finetune_runs = pd.read_csv(results_dir / "finetune_runs_summary.csv")
    scratch_runs = pd.read_csv(results_dir / "scratch_runs_summary.csv")

    rows: list[dict] = []

    for model in ("cnn", "lstm"):
        conditions = [
            ("head_bn_eval", finetune_runs, {"model": model, "mode": "head_bn_eval"}),
            ("head", finetune_runs, {"model": model, "mode": "head"}),
            ("full", finetune_runs, {"model": model, "mode": "full"}),
            ("scratch", scratch_runs, {"model": model, "lr_label": "primary"}),
        ]
        for condition_name, source_df, filters in conditions:
            if condition_name == "head_bn_eval" and model == "lstm":
                continue  # undefined: LSTM has no BatchNorm layers to recalibrate
            sub = source_df.copy()
            for col, val in filters.items():
                sub = sub[sub[col] == val]
            if sub.empty:
                print(f"[warn] no rows for model={model} condition={condition_name} -- skipping")
                continue
            acc = _per_seed_global_acc(sub)
            rows.append({"model": model, "condition": condition_name, **_stats(acc)})

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(df.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed-level dispersion for the fine-tuning conditions.")
    p.add_argument("--results-dir", type=Path, default=Path("artifacts/results"))
    p.add_argument("--out", type=Path, default=Path("artifacts/results/condition_dispersion.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.results_dir, args.out)


if __name__ == "__main__":
    main()
