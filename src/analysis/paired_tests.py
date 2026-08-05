from __future__ import annotations

"""
Paired tests over the 8 shared seeds: C3-full vs. C3-head (does unfreezing
the backbone add anything beyond noise), and C3-head vs. BN-only (does
retraining the head add anything once BatchNorm statistics are already
recalibrated). Each seed's own mean test accuracy across the 30 datasets is
one paired observation -- same seed, same data splits, different
fine-tuning condition on top, so a paired test is more powerful than
treating the two conditions as independent samples.

Wilcoxon signed-rank is the primary test (N=8, no normality assumption);
the paired t-test is reported alongside for comparison.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def _per_seed_mean(df: pd.DataFrame) -> pd.Series:
    return df.groupby("seed")["test_acc"].mean().sort_index()


def _paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """a - b, paired."""
    diff = a - b
    wilcoxon = stats.wilcoxon(a, b)
    ttest = stats.ttest_rel(a, b)
    return {
        "n_seeds": int(len(a)),
        "mean_diff": float(diff.mean()),
        "std_diff": float(diff.std(ddof=1)),
        "wilcoxon_statistic": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "paired_t_statistic": float(ttest.statistic),
        "paired_t_p": float(ttest.pvalue),
    }


def run(finetune_runs_csv: Path, bn_only_runs_csv: Path, out_path: Path) -> dict:
    finetune = pd.read_csv(finetune_runs_csv)
    bn_only = pd.read_csv(bn_only_runs_csv)

    head = _per_seed_mean(finetune[(finetune["model"] == "cnn") & (finetune["mode"] == "head")])
    full = _per_seed_mean(finetune[(finetune["model"] == "cnn") & (finetune["mode"] == "full")])
    bn = _per_seed_mean(bn_only[bn_only["model"] == "cnn"])

    seeds_hf = sorted(set(head.index) & set(full.index))
    seeds_bh = sorted(set(bn.index) & set(head.index))

    result = {
        "primary_test": "wilcoxon_signed_rank",
        "unit": "each value is one seed's own mean test accuracy across all 30 datasets",
        "c3_full_vs_c3_head": {
            "seeds": seeds_hf,
            "mean_c3_full": float(full.loc[seeds_hf].mean()),
            "mean_c3_head": float(head.loc[seeds_hf].mean()),
            **_paired_test(full.loc[seeds_hf].to_numpy(), head.loc[seeds_hf].to_numpy()),
        },
        "c3_head_vs_bn_only": {
            "seeds": seeds_bh,
            "mean_c3_head": float(head.loc[seeds_bh].mean()),
            "mean_bn_only": float(bn.loc[seeds_bh].mean()),
            **_paired_test(head.loc[seeds_bh].to_numpy(), bn.loc[seeds_bh].to_numpy()),
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paired tests: C3-full vs C3-head, C3-head vs BN-only.")
    p.add_argument("--finetune-runs-csv", type=Path, default=Path("artifacts/results/finetune_runs_summary.csv"))
    p.add_argument("--bn-only-runs-csv", type=Path, default=Path("artifacts/results/bn_only_runs_summary.csv"))
    p.add_argument("--out", type=Path, default=Path("artifacts/results/paired_tests.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.finetune_runs_csv, args.bn_only_runs_csv, args.out)


if __name__ == "__main__":
    main()
