from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def _impute_missing_values(x: np.ndarray) -> np.ndarray:
    if not np.isnan(x).any():
        return x

    x = x.copy()
    for row in x:
        missing = np.isnan(row)
        if not missing.any():
            continue

        valid_idx = np.flatnonzero(~missing)
        if valid_idx.size == 0:
            row[:] = 0.0
            continue

        missing_idx = np.flatnonzero(missing)
        row[missing_idx] = np.interp(missing_idx, valid_idx, row[valid_idx])

    return x


def load_ucr_dataset(dataset_root: str, dataset_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = Path(dataset_root) / dataset_name
    train_file = dataset_dir / f"{dataset_name}_TRAIN.tsv"
    test_file = dataset_dir / f"{dataset_name}_TEST.tsv"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {dataset_dir}. "
            f"Expected {train_file.name} and {test_file.name}."
        )

    train_data = np.loadtxt(train_file, delimiter="\t")
    test_data = np.loadtxt(test_file, delimiter="\t")

    y_train_raw = train_data[:, 0]
    x_train = _impute_missing_values(train_data[:, 1:])
    y_test_raw = test_data[:, 0]
    x_test = _impute_missing_values(test_data[:, 1:])

    classes = np.unique(np.concatenate([y_train_raw, y_test_raw]))
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    y_train = np.array([class_to_idx[v] for v in y_train_raw], dtype=np.int64)
    y_test = np.array([class_to_idx[v] for v in y_test_raw], dtype=np.int64)

    x_train = np.expand_dims(x_train.astype(np.float32), axis=1)
    x_test = np.expand_dims(x_test.astype(np.float32), axis=1)
    return x_train, y_train, x_test, y_test


def split_train_validation(
    x: np.ndarray,
    y: np.ndarray,
    val_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        return train_test_split(x, y, test_size=val_size, random_state=seed, stratify=y)
    except ValueError:
        return train_test_split(x, y, test_size=val_size, random_state=seed, stratify=None)


def _worker_init_fn(worker_id: int, base_seed: int) -> None:
    seed = base_seed + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    seed: int = 0,
) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    generator = torch.Generator()
    generator.manual_seed(seed)
    worker_init = (lambda wid: _worker_init_fn(wid, seed)) if num_workers > 0 else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=worker_init,
        generator=generator if shuffle else None,
    )


def class_weights_from_labels(y: np.ndarray, num_classes: int) -> torch.Tensor:
    """num_classes must be the model head's true class count (train union test),
    not derived from y alone: a stratified train/val split can leave a rare
    class with zero training instances, which previously sized the weight
    tensor off np.unique(y).max()+1 and produced a too-short tensor whenever
    the highest-index class happened to be the one missing (hit in practice
    on Phoneme, 39 classes). Classes absent from y get a neutral weight of
    1.0 -- they contribute no samples to this split's loss either way, so the
    value only matters for keeping the tensor's length correct."""
    classes, counts = np.unique(y, return_counts=True)
    weights = np.ones(num_classes, dtype=np.float32)
    inv_freq = 1.0 / counts.astype(np.float32)
    inv_freq = inv_freq / inv_freq.sum() * len(classes)
    for cls, weight in zip(classes, inv_freq, strict=False):
        weights[int(cls)] = weight
    return torch.from_numpy(weights)
