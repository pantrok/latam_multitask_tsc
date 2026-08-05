#!/usr/bin/env python3
"""
geometry_gap_correlation.py
===========================

Does inter-dataset signal geometry predict where multi-task sharing hurts?

Tests the association between:

    iota_i  -- isolation of dataset i in the inter-dataset DTW distance
               matrix (mean distance to its k nearest neighbours), and
    delta_i -- accuracy penalty of the shared multi-task model relative to
               a per-dataset specialist (HIVE-COTE 2.0 by default).

Everything here runs on results you already have. No retraining.

METHOD IS PRE-SPECIFIED. Do not change the isolation metric, k, or the
statistic after seeing the output -- that is metric shopping and a reviewer
will call it. The defaults below (k=5, Spearman, 10^4 permutations) are the
ones written into Section III-G of the manuscript. If you must change one,
change it in the manuscript too, and say so.

A weak or non-significant result IS a result. It downgrades the geometric
account from a predictive claim to a qualitative one. That interpretation is
pre-committed in the manuscript; report the number either way.

Outputs
-------
  artifacts/analysis/geometry_gap_correlation.json   machine-readable
  artifacts/analysis/geometry_gap_correlation.txt    numbers for the LaTeX
  artifacts/analysis/geometry_gap_scatter.png        diagnostic figure

Usage
-----
  uv run python scripts/geometry_gap_correlation.py \
      --distance-matrix artifacts/eda/dtw_distance_matrix.csv \
      --results artifacts/runs/per_dataset_results.csv \
      --baseline artifacts/comparison/hive_cote2.csv \
      --metadata assets/DataSummary.csv

Input formats (adapt the loaders below to whatever your files actually look
like -- that is the only part of this script you should need to touch):

  --distance-matrix : square CSV, 30x30, first column and header row are
                      dataset names. Symmetric, zero diagonal.
  --results         : CSV with columns [dataset, acc]  (shared model, mean
                      over the 8 seeds). Extra columns are ignored.
  --baseline        : CSV with columns [dataset, acc]  (per-dataset
                      specialist).
  --metadata        : CSV with columns [dataset, train_size, n_classes].
                      UCR DataSummary.csv works with --metadata-ucr.

Dependencies: numpy, pandas, scipy, matplotlib.

--------------------------------------------------------------------------
Connected to this repo's real artifacts (run_all.py invokes this stage as):

  python -m src.analysis.geometry_gap_correlation \
      --distance-matrix artifacts/eda/dtw_distance_matrix.csv \
      --results artifacts/results/per_dataset_results.csv \
      --baseline artifacts/results/hc2_baseline.csv \
      --metadata assets/DataSummary.csv --metadata-ucr

per_dataset_results.csv has a "dataset" column and an "acc_mean" column
(matches load_accuracy's column search below). hc2_baseline.csv has
"dataset"/"acc" (produced by src/analysis/baselines.py from the published
HC2 reference results). k, the statistic, and the permutation count must
stay fixed at their pre-specified values.
--------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --------------------------------------------------------------------------
# Pre-specified analysis parameters. Changing these after seeing results is
# exactly what the permutation test cannot protect you from.
# --------------------------------------------------------------------------
K_NEIGHBOURS = 5
N_PERMUTATIONS = 10_000
RANDOM_SEED = 42
N_HIGH_LEVERAGE_DROP = 2  # robustness check: drop the 2 largest penalties


# ==========================================================================
# Loading
# ==========================================================================

def load_distance_matrix(path: Path) -> pd.DataFrame:
    """Square inter-dataset distance matrix, indexed by dataset name."""
    d = pd.read_csv(path, index_col=0)
    d.index = d.index.astype(str).str.strip()
    d.columns = d.columns.astype(str).str.strip()

    if not d.index.equals(d.columns):
        common = d.index.intersection(d.columns)
        if len(common) < 3:
            raise ValueError(
                f"{path}: row and column labels do not match "
                f"(only {len(common)} in common). Check the export."
            )
        print(f"  [warn] reindexing distance matrix to {len(common)} common labels")
        d = d.loc[common, common]

    a = d.to_numpy(dtype=float)
    if not np.allclose(a, a.T, equal_nan=True, atol=1e-8):
        print("  [warn] distance matrix is not symmetric; symmetrizing as (A+A')/2")
        a = (a + a.T) / 2.0
        d = pd.DataFrame(a, index=d.index, columns=d.columns)

    return d


def load_accuracy(path: Path, label: str) -> pd.Series:
    """CSV with a dataset column and an accuracy column -> Series."""
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}

    ds_col = next((cols[c] for c in
                   ("dataset", "name", "dataset_name", "set") if c in cols), None)
    acc_col = next((cols[c] for c in
                    ("acc", "accuracy", "test_acc", "acc_mean", "mean_acc",
                     "test_accuracy") if c in cols), None)

    if ds_col is None or acc_col is None:
        raise ValueError(
            f"{path}: could not find dataset/accuracy columns. "
            f"Found: {list(df.columns)}. Rename them or edit load_accuracy()."
        )

    s = df.groupby(df[ds_col].astype(str).str.strip())[acc_col].mean()
    s.name = label

    if s.max() > 1.5:
        print(f"  [warn] {label}: values exceed 1.0; assuming percent, dividing by 100")
        s = s / 100.0

    return s


def load_metadata(path: Path, ucr_format: bool) -> pd.DataFrame:
    """Confound variables: training set size and number of classes."""
    df = pd.read_csv(path)

    if ucr_format:
        # UCR DataSummary.csv column names
        ren = {"Name": "dataset", "Train ": "train_size", "Train": "train_size",
               "Class": "n_classes"}
        df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    cols = {c.lower().strip(): c for c in df.columns}
    ds_col = next((cols[c] for c in ("dataset", "name") if c in cols), None)
    tr_col = next((cols[c] for c in ("train_size", "train", "n_train") if c in cols), None)
    cl_col = next((cols[c] for c in ("n_classes", "class", "classes", "n_class")
                   if c in cols), None)

    if not all([ds_col, tr_col, cl_col]):
        raise ValueError(
            f"{path}: need dataset/train_size/n_classes. Found: {list(df.columns)}"
        )

    out = df[[ds_col, tr_col, cl_col]].copy()
    out.columns = ["dataset", "train_size", "n_classes"]
    out["dataset"] = out["dataset"].astype(str).str.strip()
    return out.set_index("dataset")


# ==========================================================================
# The isolation index
# ==========================================================================

def isolation_index(dist: pd.DataFrame, k: int = K_NEIGHBOURS) -> pd.Series:
    """
    iota_i = mean distance from dataset i to its k nearest neighbours.

    Self-distance is excluded. Larger iota = farther from the dense regions
    of the corpus. This is deliberately the simplest defensible definition:
    it uses the raw distance matrix, not the t-SNE embedding, so it cannot be
    accused of inheriting structure from the very projection whose story it
    is being used to test.
    """
    a = dist.to_numpy(dtype=float).copy()
    np.fill_diagonal(a, np.inf)
    a.sort(axis=1)

    k_eff = min(k, a.shape[1] - 1)
    if k_eff != k:
        print(f"  [warn] only {a.shape[1]} datasets; using k={k_eff}")

    return pd.Series(a[:, :k_eff].mean(axis=1), index=dist.index, name="iota")


# ==========================================================================
# Statistics
# ==========================================================================

def permutation_spearman(x: np.ndarray, y: np.ndarray,
                         n_perm: int = N_PERMUTATIONS,
                         seed: int = RANDOM_SEED) -> tuple[float, float, np.ndarray]:
    """
    Spearman rho with a two-sided permutation p-value.

    With N=30 the asymptotic p-value is not trustworthy; shuffling y against
    a fixed x rebuilds the null distribution directly. p is computed with the
    (r+1)/(n+1) correction so it can never be reported as exactly zero.
    """
    rho_obs = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(seed)

    null = np.empty(n_perm)
    y_perm = y.copy()
    for i in range(n_perm):
        rng.shuffle(y_perm)
        null[i] = stats.spearmanr(x, y_perm).statistic

    p = (np.sum(np.abs(null) >= abs(rho_obs)) + 1) / (n_perm + 1)
    return float(rho_obs), float(p), null


def partial_spearman(x: np.ndarray, y: np.ndarray,
                     controls: np.ndarray,
                     n_perm: int = N_PERMUTATIONS,
                     seed: int = RANDOM_SEED) -> tuple[float, float]:
    """
    Spearman correlation between x and y after linearly removing `controls`
    from both, on ranks.

    Answers the obvious reviewer objection: "isn't isolation just a proxy for
    dataset size or class count?" If the partial correlation survives, the
    geometry carries information those two do not.
    """
    def rank(v):
        return stats.rankdata(v, axis=0)

    xr, yr = rank(x), rank(y)
    cr = rank(controls)
    design = np.column_stack([np.ones(len(xr)), cr])

    res_x = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
    res_y = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]

    return permutation_spearman(res_x, res_y, n_perm, seed)[:2]


# ==========================================================================
# Main
# ==========================================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Correlate inter-dataset geometry with multi-task accuracy penalty.")
    ap.add_argument("--distance-matrix", type=Path, default=Path("artifacts/eda/dtw_distance_matrix.csv"))
    ap.add_argument("--results", type=Path, default=Path("artifacts/results/per_dataset_results.csv"),
                    help="Shared multi-task model, per-dataset accuracy.")
    ap.add_argument("--baseline", type=Path, default=Path("artifacts/results/hc2_baseline.csv"),
                    help="Per-dataset specialist accuracy (HIVE-COTE 2.0).")
    ap.add_argument("--metadata", type=Path, default=Path("assets/DataSummary.csv"),
                    help="Optional: enables the partial-correlation control.")
    ap.add_argument("--metadata-ucr", action="store_true", default=True,
                    help="Metadata file is the UCR DataSummary.csv.")
    ap.add_argument("--baseline-name", default="HIVE-COTE 2.0")
    ap.add_argument("-k", "--k-neighbours", type=int, default=K_NEIGHBOURS)
    ap.add_argument("--out-dir", type=Path, default=Path("artifacts/analysis"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    print("Loading inputs")
    dist = load_distance_matrix(args.distance_matrix)
    shared = load_accuracy(args.results, "acc_shared")
    special = load_accuracy(args.baseline, "acc_specialist")

    common = dist.index.intersection(shared.index).intersection(special.index)
    if len(common) < 10:
        print(f"ERROR: only {len(common)} datasets in common across inputs.",
              file=sys.stderr)
        print(f"  distance matrix : {sorted(dist.index)[:5]} ...", file=sys.stderr)
        print(f"  results         : {sorted(shared.index)[:5]} ...", file=sys.stderr)
        print(f"  baseline        : {sorted(special.index)[:5]} ...", file=sys.stderr)
        print("Dataset names must match exactly across files.", file=sys.stderr)
        return 1

    common = sorted(common)
    print(f"  {len(common)} datasets in common")
    if len(common) < len(dist.index):
        missing = sorted(set(dist.index) - set(common))
        print(f"  [warn] dropped {len(missing)}: {missing}")

    dist = dist.loc[common, common]
    iota = isolation_index(dist, args.k_neighbours)
    delta = (special.loc[common] - shared.loc[common]).rename("delta")

    df = pd.DataFrame({"iota": iota.loc[common], "delta": delta}).dropna()
    n = len(df)

    print(f"\nPrimary test (Spearman, {N_PERMUTATIONS} permutations, N={n})")
    rho, p, null = permutation_spearman(df["iota"].to_numpy(), df["delta"].to_numpy())
    print(f"  rho = {rho:+.4f}   p = {p:.4f}")

    # -- robustness 1: drop the highest-penalty datasets -------------------
    dropped = df.nlargest(N_HIGH_LEVERAGE_DROP, "delta").index.tolist()
    df_rob = df.drop(index=dropped)
    rho_r, p_r, _ = permutation_spearman(df_rob["iota"].to_numpy(),
                                         df_rob["delta"].to_numpy())
    print(f"\nRobustness: without {dropped} (N={len(df_rob)})")
    print(f"  rho = {rho_r:+.4f}   p = {p_r:.4f}")

    # -- robustness 2: partial correlation ---------------------------------
    rho_p = p_p = None
    if args.metadata:
        meta = load_metadata(args.metadata, args.metadata_ucr)
        meta = meta.reindex(df.index).dropna()
        if len(meta) >= 10:
            sub = df.loc[meta.index]
            controls = np.column_stack([
                np.log(meta["train_size"].to_numpy(dtype=float)),
                np.log(meta["n_classes"].to_numpy(dtype=float)),
            ])
            rho_p, p_p = partial_spearman(sub["iota"].to_numpy(),
                                          sub["delta"].to_numpy(), controls)
            print(f"\nPartial correlation, controlling log(train size) and "
                  f"log(n classes) (N={len(sub)})")
            print(f"  rho = {rho_p:+.4f}   p = {p_p:.4f}")
        else:
            print("\n[warn] metadata covers too few datasets; skipping partial correlation")

    # -- outputs -----------------------------------------------------------
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "n_datasets": int(n),
        "k_neighbours": int(args.k_neighbours),
        "n_permutations": int(N_PERMUTATIONS),
        "baseline": args.baseline_name,
        "spearman_rho": rho,
        "permutation_p": p,
        "robust_dropped": dropped,
        "robust_rho": rho_r,
        "robust_p": p_r,
        "partial_rho": rho_p,
        "partial_p": p_p,
        "per_dataset": df.assign(
            acc_shared=shared.loc[df.index],
            acc_specialist=special.loc[df.index],
        ).round(6).reset_index().rename(columns={"index": "dataset"}).to_dict("records"),
    }

    (args.out_dir / "geometry_gap_correlation.json").write_text(
        json.dumps(result, indent=2))

    def fmt(v, nd=4):
        return "n/a" if v is None else f"{v:+.{nd}f}"

    report = f"""Geometry-penalty correlation
============================
Isolation index : mean DTW distance to k={args.k_neighbours} nearest neighbours
Penalty         : {args.baseline_name} accuracy minus shared-model accuracy
N               : {n} datasets

  Spearman rho          {fmt(rho)}    permutation p = {p:.4f}
  Without {str(dropped):<22}{fmt(rho_r)}    permutation p = {p_r:.4f}
  Partial (size, class) {fmt(rho_p)}    permutation p = {'n/a' if p_p is None else f'{p_p:.4f}'}

Paste into the manuscript, Section V-F:
  \\PENDING{{rho}}         -> {rho:+.3f}
  \\PENDING{{p}}           -> {p:.4f}
  \\PENDING{{rho_robust}}  -> {rho_r:+.3f}
  \\PENDING{{p_robust}}    -> {p_r:.4f}
  \\PENDING{{rho_partial}} -> {fmt(rho_p, 3)}
  \\PENDING{{p_partial}}   -> {'n/a' if p_p is None else f'{p_p:.4f}'}

Reminder: a weak or non-significant rho is reportable. The manuscript
pre-commits (Section III-G) to downgrading the geometric account to a
qualitative one in that case. Do not re-run with a different k to find a
better number.
"""
    (args.out_dir / "geometry_gap_correlation.txt").write_text(report)
    print("\n" + report)

    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

            ax1.scatter(df["iota"], df["delta"], s=45, alpha=0.8,
                        edgecolor="black", linewidth=0.5)
            for name, row in df.iterrows():
                ax1.annotate(name, (row["iota"], row["delta"]), fontsize=6,
                             xytext=(3, 3), textcoords="offset points")
            ax1.set_xlabel(f"Isolation index (mean DTW distance to {args.k_neighbours}-NN)")
            ax1.set_ylabel(f"Accuracy penalty vs. {args.baseline_name}")
            ax1.set_title(f"Spearman rho = {rho:+.3f}  (p = {p:.4f}, N = {n})")
            ax1.axhline(0, color="grey", linewidth=0.8, linestyle="--")
            ax1.grid(alpha=0.3)

            ax2.hist(null, bins=50, alpha=0.75, edgecolor="black", linewidth=0.3)
            ax2.axvline(rho, color="red", linewidth=2,
                        label=f"observed = {rho:+.3f}")
            ax2.set_xlabel("Spearman rho under the null")
            ax2.set_ylabel("Frequency")
            ax2.set_title(f"Permutation null ({N_PERMUTATIONS} resamples)")
            ax2.legend()
            ax2.grid(alpha=0.3)

            fig.tight_layout()
            out = args.out_dir / "geometry_gap_scatter.png"
            fig.savefig(out, dpi=300)
            print(f"Figure written to {out}")
        except ImportError:
            print("[warn] matplotlib not available; skipping figure")

    print(f"Results written to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
