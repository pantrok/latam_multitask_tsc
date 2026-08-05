from __future__ import annotations

"""
fig_distance_matrices -- DTW / Soft-DTW / CID heatmaps.

Each panel gets its own independent min-max normalization to [0,1] and its
own colorbar: the three metrics are not on the same physical scale (CID's
range reaches the hundreds, DTW/Soft-DTW stay in the single digits), so a
shared color axis would not be meaningful for comparing them directly.

Repo-only diagnostic, not part of the manuscript's figure set.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.figure_style import apply_style, save_figure

METRICS = ("dtw", "soft_dtw", "cid")
TITLES = {"dtw": "DTW", "soft_dtw": "Soft-DTW", "cid": "CID"}
FILENAMES = {"dtw": "dtw_distance_matrix.csv", "soft_dtw": "softdtw_distance_matrix.csv",
             "cid": "cid_distance_matrix.csv"}

# Not IEEE column-constrained -- this figure is repo-only, out of the
# manuscript's figure set, so it's sized for actual legibility of 30
# dataset labels per axis instead.
FIG_WIDTH_IN = 14.0
FIG_HEIGHT_IN = 5.2
LABEL_FONTSIZE = 6.0


def run(eda_dir: Path = Path("artifacts/eda"), out_dir: Path = Path("figures")) -> plt.Figure:
    apply_style()
    matrices = {m: pd.read_csv(eda_dir / FILENAMES[m], index_col=0) for m in METRICS}
    names = matrices["dtw"].index.tolist()
    n = len(names)

    fig, axes = plt.subplots(1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    for ax, metric in zip(axes, METRICS):
        df = matrices[metric]
        arr = df.to_numpy(dtype=float)
        lo, hi = arr.min(), arr.max()
        normed = (arr - lo) / (hi - lo) if hi > lo else arr * 0.0

        im = ax.imshow(normed, cmap="viridis", vmin=0.0, vmax=1.0, aspect="equal")
        ax.set_title(f"{TITLES[metric]} (range {lo:.2g}-{hi:.2g})", fontsize=9)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=LABEL_FONTSIZE)
        ax.set_yticklabels(names, fontsize=LABEL_FONTSIZE)

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("Normalized distance [0,1]", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_distance_matrices")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--eda-dir", type=Path, default=Path("artifacts/eda"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.eda_dir, args.out_dir))
