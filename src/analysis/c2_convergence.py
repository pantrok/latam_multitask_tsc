from __future__ import annotations

"""
Per-epoch training/validation curves for C2 (joint training), aggregated
over the 8 seeds. Read directly off each seed's already-recorded
metrics.json history -- no retraining needed.

Answers whether C2's low absolute accuracy reflects an early cutoff (still
improving when training stopped) or a genuine plateau (already converged,
gap is about capacity/sharing, not under-training).
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def load_seed_histories(runs_dir: Path, seeds: list[int]) -> pd.DataFrame:
    rows = []
    for seed in seeds:
        metrics_path = runs_dir / f"seed_{seed}" / "metrics.json"
        if not metrics_path.exists():
            print(f"[warn] {metrics_path} not found -- skipping seed {seed}")
            continue
        with metrics_path.open("r", encoding="utf-8") as f:
            m = json.load(f)
        for h in m["history"]:
            rows.append({"seed": seed, **h})
    return pd.DataFrame(rows)


def aggregate_curves(df: pd.DataFrame) -> pd.DataFrame:
    # groupby naturally drops out any seed that already early-stopped before
    # a given epoch -- n_seeds per row shows exactly how many contributed.
    return (
        df.groupby("epoch")
        .agg(
            n_seeds=("seed", "nunique"),
            train_loss_mean=("train_loss", "mean"), train_loss_std=("train_loss", "std"),
            val_loss_mean=("val_loss_mean", "mean"), val_loss_std=("val_loss_mean", "std"),
            val_acc_mean=("val_acc_mean", "mean"), val_acc_std=("val_acc_mean", "std"),
            val_f1_mean=("val_f1_macro_mean", "mean"), val_f1_std=("val_f1_macro_mean", "std"),
        )
        .reset_index()
    )


def run(runs_dir: Path, seeds: list[int], out_path: Path) -> pd.DataFrame:
    df = load_seed_histories(runs_dir, seeds)
    curves = aggregate_curves(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    curves.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    return curves


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate C2 per-epoch training/validation curves across seeds.")
    p.add_argument("--runs-dir", type=Path, default=Path("artifacts/runs/cnn"))
    p.add_argument("--seeds", type=int, nargs="+", default=[7, 21, 42, 63, 84, 105, 126, 147])
    p.add_argument("--out", type=Path, default=Path("artifacts/results/c2_training_curves.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.runs_dir, args.seeds, args.out)


if __name__ == "__main__":
    main()
