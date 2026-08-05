from __future__ import annotations

"""
Does the BatchNorm-recalibration effect (C3-head minus C3-head-BN, i.e. the
gain from letting normalization statistics adapt on top of an already
frozen-stats head fine-tune) depend on training-set size?

Several Sensor datasets have fewer training instances than the batch size,
so "recalibrate statistics" means estimating per-channel mean/variance from
a single small, repeated batch -- a high-variance estimator. If the effect
correlates with training-set size, part of it is estimation regime, not
mechanism.

Runs entirely on results already computed -- no retraining.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

N_PERMUTATIONS = 10_000
RANDOM_SEED = 42


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
    head_csv: Path, head_bn_eval_csv: Path, train_size_csv: Path, out_path: Path,
) -> dict:
    head = pd.read_csv(head_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "c3_head"})
    head_bn = pd.read_csv(head_bn_eval_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "c3_head_bn"})
    sizes = pd.read_csv(train_size_csv)[["dataset", "train_size"]]

    df = head.merge(head_bn, on="dataset").merge(sizes, on="dataset")
    df["delta"] = df["c3_head"] - df["c3_head_bn"]
    df["log_train_size"] = np.log(df["train_size"])

    rho, p = permutation_spearman(df["log_train_size"].to_numpy(), df["delta"].to_numpy())

    result = {
        "n_datasets": int(len(df)),
        "n_permutations": N_PERMUTATIONS,
        "spearman_rho": rho,
        "permutation_p": p,
        "per_dataset": df.round(6).to_dict("records"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {out_path}")
    print(f"Spearman rho = {rho:+.4f}   permutation p = {p:.4f}   N = {len(df)}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlate BN-recalibration effect with training-set size.")
    p.add_argument("--head-csv", type=Path, default=Path("artifacts/results/finetune_head_cnn.csv"))
    p.add_argument("--head-bn-eval-csv", type=Path, default=Path("artifacts/results/finetune_head_bn_eval_cnn.csv"))
    p.add_argument("--train-size-csv", type=Path, default=Path("artifacts/results/head_update_counts.csv"),
                    help="Any CSV with dataset,train_size columns -- head_update_counts.csv already has both.")
    p.add_argument("--out", type=Path, default=Path("artifacts/results/bn_effect_vs_trainsize.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.head_csv, args.head_bn_eval_csv, args.train_size_csv, args.out)


if __name__ == "__main__":
    main()
