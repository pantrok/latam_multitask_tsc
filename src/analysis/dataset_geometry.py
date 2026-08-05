from __future__ import annotations

"""
Inter-dataset distance geometry: DTW / Soft-DTW / CID between per-dataset
prototypes (each dataset's train split averaged into one Z-normalized,
200-point-resampled series), plus t-SNE / hierarchical clustering on top.

Runs in seconds -- distances are computed between the 30 per-dataset
prototypes, not between individual instances.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE

from src.data.dataset_list import get_sensor_datasets
from src.data.ucr_io import load_ucr_dataset


@dataclass(frozen=True)
class DatasetPrototypes:
    series: Mapping[str, np.ndarray]
    split: str
    target_length: int


def _resample_series(values: np.ndarray, target_length: int) -> np.ndarray | None:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return None
    original_idx = np.linspace(0.0, 1.0, num=clean.size, dtype=float)
    target_idx = np.linspace(0.0, 1.0, num=target_length, dtype=float)
    return np.interp(target_idx, original_idx, clean)


def build_dataset_prototypes(
    sensor_sets: Mapping[str, np.ndarray],
    *,
    target_length: int = 200,
) -> DatasetPrototypes:
    """sensor_sets: {dataset_name: x_train array shaped (N, 1, L)}"""
    prototypes: MutableMapping[str, np.ndarray] = {}
    for name, x_train in sensor_sets.items():
        resampled: list[np.ndarray] = []
        for row in x_train[:, 0, :]:
            res = _resample_series(row, target_length=target_length)
            if res is None:
                continue
            res = res - res.mean()
            std = res.std()
            if std > 1e-8:
                res = res / std
            resampled.append(res)

        if not resampled:
            prototypes[name] = np.zeros(target_length, dtype=float)
            continue

        mean_series = np.mean(resampled, axis=0)
        mean_series = mean_series - mean_series.mean()
        std = mean_series.std()
        if std > 1e-8:
            mean_series = mean_series / std
        prototypes[name] = mean_series.astype(float, copy=False)

    return DatasetPrototypes(series=prototypes, split="train", target_length=target_length)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    n, m = len(a), len(b)
    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i - 1] - b[j - 1])
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def _softmin(values: Sequence[float], gamma: float) -> float:
    scaled = -np.asarray(values, dtype=float) / gamma
    max_scaled = np.max(scaled)
    return float(-gamma * (np.log(np.sum(np.exp(scaled - max_scaled))) + max_scaled))


def _soft_dtw_distance(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    if gamma <= 0:
        raise ValueError("gamma must be positive for Soft-DTW.")
    n, m = len(a), len(b)
    R = np.full((n + 1, m + 1), np.inf, dtype=float)
    R[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            R[i, j] = cost + _softmin((R[i - 1, j], R[i, j - 1], R[i - 1, j - 1]), gamma)
    return float(R[n, m] / max(n, m))


def _cid_distance(a: np.ndarray, b: np.ndarray) -> float:
    euclidean = np.linalg.norm(a - b)
    diff_a = np.diff(a)
    diff_b = np.diff(b)
    ca = np.sqrt(np.sum(diff_a**2))
    cb = np.sqrt(np.sum(diff_b**2))
    correction = 1.0 if min(ca, cb) <= 1e-12 else max(ca, cb) / min(ca, cb)
    return float(euclidean * correction)


def compute_distance_matrices(
    prototypes: DatasetPrototypes,
    *,
    metrics: Iterable[str] = ("dtw", "soft_dtw", "cid"),
    soft_dtw_gamma: float = 1.0,
) -> Dict[str, pd.DataFrame]:
    valid_metrics = {"dtw", "soft_dtw", "cid"}
    requested = [m.lower() for m in metrics]
    for metric in requested:
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric: {metric}")

    names = sorted(prototypes.series.keys())
    series = prototypes.series
    matrices: Dict[str, pd.DataFrame] = {}

    for metric in requested:
        data = np.zeros((len(names), len(names)), dtype=float)
        for i, name_i in enumerate(names):
            for j in range(i + 1, len(names)):
                name_j = names[j]
                a, b = series[name_i], series[name_j]
                if metric == "dtw":
                    dist = _dtw_distance(a, b)
                elif metric == "soft_dtw":
                    dist = _soft_dtw_distance(a, b, gamma=soft_dtw_gamma)
                else:
                    dist = _cid_distance(a, b)
                data[i, j] = data[j, i] = dist
        matrices[metric] = pd.DataFrame(data, index=names, columns=names)

    return matrices


def compute_tsne(distance_df: pd.DataFrame, *, random_state: int = 42, perplexity: float | None = None) -> pd.DataFrame:
    distance_matrix = distance_df.to_numpy(dtype=float)
    labels = distance_df.index.to_list()
    n_samples = len(labels)
    perplex = perplexity or max(5, min(30, (n_samples - 1) / 3))
    tsne = TSNE(
        metric="precomputed", perplexity=perplex, learning_rate="auto",
        init="random", random_state=random_state, n_components=2,
    )
    coords = tsne.fit_transform(distance_matrix)
    return pd.DataFrame(coords, index=labels, columns=["x", "y"])


def cluster_datasets(distance_df: pd.DataFrame, *, n_clusters: int | None = None, linkage: str = "average") -> pd.Series:
    if n_clusters is None:
        n_clusters = max(2, len(distance_df) // 4)
    try:
        model = AgglomerativeClustering(metric="precomputed", n_clusters=n_clusters, linkage=linkage)
    except TypeError:
        model = AgglomerativeClustering(affinity="precomputed", n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(distance_df.to_numpy(dtype=float))
    return pd.Series(labels, index=distance_df.index, name="cluster")


def run(dataset_root: str = "data/raw/UCRArchive_2018", out_dir: Path = Path("artifacts/eda")) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = get_sensor_datasets("assets/DataSummary.csv")

    sensor_train: dict[str, np.ndarray] = {}
    for name in names:
        x_train, _, _, _ = load_ucr_dataset(dataset_root, name)
        sensor_train[name] = x_train

    prototypes = build_dataset_prototypes(sensor_train)
    matrices = compute_distance_matrices(prototypes)

    filenames = {"dtw": "dtw_distance_matrix.csv", "soft_dtw": "softdtw_distance_matrix.csv",
                 "cid": "cid_distance_matrix.csv"}
    for metric, df in matrices.items():
        path = out_dir / filenames[metric]
        df.to_csv(path)
        print(f"Wrote {path}")

    tsne_coords = compute_tsne(matrices["dtw"])
    tsne_coords.to_csv(out_dir / "tsne_dtw_coords.csv")
    print(f"Wrote {out_dir / 'tsne_dtw_coords.csv'}")


if __name__ == "__main__":
    run()
