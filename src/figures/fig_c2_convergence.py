from __future__ import annotations

"""
fig_c2_convergence -- C2's per-epoch loss and validation accuracy,
aggregated over the 8 seeds, +-1 std band. Shows directly whether training
stopped at a plateau or was still improving, instead of leaving that to be
inferred from the single final accuracy number.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.figure_style import IEEE_PAGE_WIDTH_IN, PALETTE, apply_style, save_figure


def _draw_loss(ax, df: pd.DataFrame) -> None:
    ax.plot(df["epoch"], df["train_loss_mean"], color=PALETTE[0], marker="o", markersize=3, label="Train loss")
    ax.fill_between(df["epoch"], df["train_loss_mean"] - df["train_loss_std"],
                     df["train_loss_mean"] + df["train_loss_std"], color=PALETTE[0], alpha=0.15)
    ax.plot(df["epoch"], df["val_loss_mean"], color=PALETTE[1], marker="s", markersize=3, linestyle="--",
            label="Validation loss")
    ax.fill_between(df["epoch"], df["val_loss_mean"] - df["val_loss_std"],
                     df["val_loss_mean"] + df["val_loss_std"], color=PALETTE[1], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (mean over seeds)")
    ax.legend(fontsize=6)


def _draw_accuracy(ax, df: pd.DataFrame) -> None:
    ax.plot(df["epoch"], df["val_acc_mean"], color=PALETTE[2], marker="^", markersize=3)
    ax.fill_between(df["epoch"], df["val_acc_mean"] - df["val_acc_std"],
                     df["val_acc_mean"] + df["val_acc_std"], color=PALETTE[2], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (mean over seeds)")
    n_min, n_max = int(df["n_seeds"].min()), int(df["n_seeds"].max())
    if n_min != n_max:
        ax.text(0.02, 0.02, f"n seeds per epoch: {n_max} -> {n_min}", transform=ax.transAxes,
                fontsize=6, ha="left", va="bottom", color="#555555")


def run(curves_csv: Path = Path("artifacts/results/c2_training_curves.csv"),
        out_dir: Path = Path("figures")) -> plt.Figure:
    apply_style()
    df = pd.read_csv(curves_csv)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH_IN, 3.2))
    _draw_loss(ax_loss, df)
    _draw_accuracy(ax_acc, df)
    fig.tight_layout()
    save_figure(fig, out_dir, "fig_c2_convergence")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--curves-csv", type=Path, default=Path("artifacts/results/c2_training_curves.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.curves_csv, args.out_dir))
