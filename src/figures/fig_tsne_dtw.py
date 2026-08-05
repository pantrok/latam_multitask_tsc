from __future__ import annotations

"""fig_tsne_dtw -- t-SNE embedding of the 30-dataset DTW distance matrix."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.figure_style import IEEE_COLUMN_WIDTH_IN, PALETTE, apply_style, save_figure

ISOLATED_LABELS = {"Trace", "Earthquakes", "FordA", "FordB", "CinCECGTorso", "Phoneme"}


def run(coords_csv: Path = Path("artifacts/eda/tsne_dtw_coords.csv"), out_dir: Path = Path("figures")) -> plt.Figure:
    apply_style()
    df = pd.read_csv(coords_csv, index_col=0)

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, 3.0))
    ax.scatter(df["x"], df["y"], s=28, color=PALETTE[0], edgecolor="black", linewidth=0.4, alpha=0.85)

    for name, row in df.iterrows():
        if name in ISOLATED_LABELS:
            ax.annotate(name, (row["x"], row["y"]), fontsize=6.0, fontweight="bold",
                        xytext=(3, 3), textcoords="offset points", color=PALETTE[1])

    ax.set_xlabel("t-SNE dim. 1")
    ax.set_ylabel("t-SNE dim. 2")
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_tsne_dtw")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--coords-csv", type=Path, default=Path("artifacts/eda/tsne_dtw_coords.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.coords_csv, args.out_dir))
