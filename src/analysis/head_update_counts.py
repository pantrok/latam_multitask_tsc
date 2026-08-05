from __future__ import annotations

"""
Per-dataset count of head-specific gradient updates during C2 (joint
training, interleaved batches across all 30 datasets) versus C3-head
(frozen-backbone, one dataset at a time). Tests whether C3-head's recovery
is explained by a starved head -- during C2 a small dataset's head receives
far fewer updates over the shared run than a large dataset's, while
C3-head gives every dataset up to its own full epoch budget regardless of
size.

Every training batch for a given dataset triggers exactly one optimizer
step touching that dataset's head, in both conditions (C3-head additionally
freezes the backbone, but the head update itself is identical). So the
count is batches_per_epoch(dataset) x epochs actually trained, read
directly off the already-completed C2/C3-head runs -- no retraining needed.

Outputs
-------
  artifacts/results/head_update_counts.csv     dataset, train_size,
                                                batches_per_epoch,
                                                updates_c2, updates_c3head
  artifacts/analysis/head_update_correlation.json/.txt
                                                Spearman correlation between
                                                C3-head's recovery and the
                                                updates_c3head/updates_c2 ratio
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data.ucr_io import load_ucr_dataset, split_train_validation

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42


def _train_size_and_batches(dataset_root: str, dataset_name: str, val_size: float,
                             seed: int, batch_size: int) -> tuple[int, int]:
    x_train, y_train, _x_test, _y_test = load_ucr_dataset(dataset_root, dataset_name)
    x_tr, _x_val, _y_tr, _y_val = split_train_validation(x_train, y_train, val_size, seed)
    train_size = int(x_tr.shape[0])
    batches_per_epoch = -(-train_size // batch_size)  # ceil division, matches DataLoader(drop_last=False)
    return train_size, batches_per_epoch


def build_update_counts(
    dataset_names: list[str], dataset_root: str, joint_runs_summary_csv: Path,
    finetune_runs_summary_csv: Path, model: str = "cnn", batch_size: int = 48,
    val_size: float = 0.2, split_seed: int = 7,
) -> pd.DataFrame:
    joint_runs = pd.read_csv(joint_runs_summary_csv)
    joint_runs = joint_runs[joint_runs["model"] == model]
    finetune_runs = pd.read_csv(finetune_runs_summary_csv)
    head_runs = finetune_runs[(finetune_runs["model"] == model) & (finetune_runs["mode"] == "head")]

    rows = []
    for dataset_name in dataset_names:
        train_size, batches_per_epoch = _train_size_and_batches(
            dataset_root, dataset_name, val_size, split_seed, batch_size,
        )
        c2_epochs = joint_runs.loc[joint_runs["dataset"] == dataset_name, "epochs_trained"].mean()
        c3head_epochs = head_runs.loc[head_runs["dataset"] == dataset_name, "epochs_trained"].mean()

        rows.append({
            "dataset": dataset_name,
            "train_size": train_size,
            "batches_per_epoch": batches_per_epoch,
            "updates_c2": batches_per_epoch * c2_epochs,
            "updates_c3head": batches_per_epoch * c3head_epochs,
        })

    return pd.DataFrame(rows)


def permutation_spearman(x: np.ndarray, y: np.ndarray, n_perm: int = N_PERMUTATIONS,
                          seed: int = RANDOM_SEED) -> tuple[float, float]:
    rho_obs = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    y_perm = y.copy()
    for i in range(n_perm):
        rng.shuffle(y_perm)
        null[i] = stats.spearmanr(x, y_perm).statistic
    p = (np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_perm + 1)
    return float(rho_obs), float(p)


def run(
    dataset_names: list[str], dataset_root: str,
    joint_results_csv: Path, joint_runs_summary_csv: Path, finetune_head_csv: Path,
    finetune_runs_summary_csv: Path, results_dir: Path, analysis_dir: Path, model: str = "cnn",
) -> None:
    counts = build_update_counts(
        dataset_names, dataset_root, joint_runs_summary_csv, finetune_runs_summary_csv, model=model,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "head_update_counts.csv"
    counts.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    joint_acc = pd.read_csv(joint_results_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "joint_acc"})
    head_acc = pd.read_csv(finetune_head_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "head_acc"})
    df = counts.merge(joint_acc, on="dataset").merge(head_acc, on="dataset")
    df["recovery"] = df["head_acc"] - df["joint_acc"]
    df["update_ratio"] = df["updates_c3head"] / df["updates_c2"]

    rho, p = permutation_spearman(df["update_ratio"].to_numpy(), df["recovery"].to_numpy())

    analysis_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "model": model,
        "n_datasets": int(len(df)),
        "n_permutations": N_PERMUTATIONS,
        "spearman_rho": rho,
        "permutation_p": p,
        "per_dataset": df.round(6).to_dict("records"),
    }
    (analysis_dir / "head_update_correlation.json").write_text(json.dumps(result, indent=2))

    report = (
        "Head-update-count vs. C3-head recovery\n"
        "=======================================\n"
        f"model: {model}\n"
        f"N: {len(df)} datasets\n"
        f"updates_c3head / updates_c2 vs. (head_acc - joint_acc)\n"
        f"  Spearman rho = {rho:+.4f}   permutation p = {p:.4f}\n"
    )
    (analysis_dir / "head_update_correlation.txt").write_text(report)
    print(report)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Head-specific gradient-update counts, C2 vs. C3-head.")
    p.add_argument("--dataset-names", type=str, nargs="+", required=True)
    p.add_argument("--dataset-root", type=str, default="data/raw/UCRArchive_2018")
    p.add_argument("--joint-results-csv", type=Path, default=Path("artifacts/results/per_dataset_results.csv"))
    p.add_argument("--joint-runs-summary-csv", type=Path, default=Path("artifacts/results/runs_summary.csv"))
    p.add_argument("--finetune-head-csv", type=Path, default=Path("artifacts/results/finetune_head_cnn.csv"))
    p.add_argument("--finetune-runs-summary-csv", type=Path,
                    default=Path("artifacts/results/finetune_runs_summary.csv"))
    p.add_argument("--results-dir", type=Path, default=Path("artifacts/results"))
    p.add_argument("--analysis-dir", type=Path, default=Path("artifacts/analysis"))
    p.add_argument("--model", choices=["cnn", "lstm"], default="cnn")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.dataset_names, args.dataset_root, args.joint_results_csv, args.joint_runs_summary_csv,
        args.finetune_head_csv, args.finetune_runs_summary_csv, args.results_dir, args.analysis_dir, args.model)


if __name__ == "__main__":
    main()
