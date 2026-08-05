from __future__ import annotations

"""
Quantitative check of whether the DTW-distance corpus geometry has any
macro-cluster structure, rather than relying on a t-SNE plot read by eye.

Two independent checks:
  - Silhouette coefficient (k=2, k=3 hierarchical clustering) directly on
    the DTW distance matrix (precomputed metric, no embedding involved).
  - Hopkins statistic on the existing t-SNE embedding coordinates -- this
    is literally the space the disputed figure shows, so it answers "does
    the thing being visualized have cluster tendency" directly rather than
    through a separate re-embedding. H ~ 0.5 means uniform/no cluster
    tendency; H -> 1 means highly clusterable; H -> 0 means regularly
    spaced (anti-clustered).

Both are reported regardless of outcome -- this script does not exist to
rescue the "two macro-clusters" claim, it exists to check it.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def silhouette_by_k(distance_matrix: pd.DataFrame, ks: tuple[int, ...] = (2, 3)) -> dict[int, dict]:
    D = distance_matrix.to_numpy(dtype=float)
    results = {}
    for k in ks:
        model = AgglomerativeClustering(metric="precomputed", linkage="average", n_clusters=k)
        labels = model.fit_predict(D)
        sil = silhouette_score(D, labels, metric="precomputed")
        sizes = np.bincount(labels).tolist()
        results[k] = {"silhouette": float(sil), "cluster_sizes": sizes}
    return results


def hopkins_statistic(X: np.ndarray, rng: np.random.Generator, m: int | None = None, n_repeats: int = 200) -> tuple[float, float]:
    """Standard Hopkins statistic: compare nearest-neighbour distances of a
    real sample against nearest-neighbour distances of uniform random points
    in the same bounding box. Averaged over n_repeats resamples since n=30
    is small enough that a single draw is noisy."""
    n, d = X.shape
    if m is None:
        m = max(2, n // 2)
    mins, maxs = X.min(axis=0), X.max(axis=0)

    h_values = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=m, replace=False)
        u = np.array([
            np.linalg.norm(X[np.arange(n) != i] - X[i], axis=1).min()
            for i in idx
        ])
        rand_pts = rng.uniform(mins, maxs, size=(m, d))
        w = np.array([np.linalg.norm(X - yp, axis=1).min() for yp in rand_pts])
        h_values.append(w.sum() / (u.sum() + w.sum()))

    return float(np.mean(h_values)), float(np.std(h_values))


def run(
    distance_matrix_csv: Path = Path("artifacts/eda/dtw_distance_matrix.csv"),
    tsne_coords_csv: Path = Path("artifacts/eda/tsne_dtw_coords.csv"),
    out_path: Path = Path("artifacts/eda/geometry_structure_check.json"),
    seed: int = 42,
) -> dict:
    dist = pd.read_csv(distance_matrix_csv, index_col=0)
    coords = pd.read_csv(tsne_coords_csv, index_col=0)

    silhouette = silhouette_by_k(dist)
    rng = np.random.default_rng(seed)
    h_mean, h_std = hopkins_statistic(coords[["x", "y"]].to_numpy(dtype=float), rng)

    result = {
        "n_datasets": len(dist),
        "silhouette_by_k": silhouette,
        "hopkins": {"mean": h_mean, "std": h_std, "n_repeats": 200,
                    "interpretation": "~0.5 = uniform/no cluster tendency; ->1 = highly clusterable; ->0 = regularly spaced"},
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distance-matrix-csv", type=Path, default=Path("artifacts/eda/dtw_distance_matrix.csv"))
    p.add_argument("--tsne-coords-csv", type=Path, default=Path("artifacts/eda/tsne_dtw_coords.csv"))
    p.add_argument("--out", type=Path, default=Path("artifacts/eda/geometry_structure_check.json"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.distance_matrix_csv, args.tsne_coords_csv, args.out, args.seed)


if __name__ == "__main__":
    main()
