from __future__ import annotations

"""
fig_control_decomposition (F2) -- decomposes the fine-tuning recovery into
its two candidate explanations: adapted features (C3-head) vs. simply more
training iterations (C4, trained from scratch at the same budget).

Two panels:
  (a) The four conditions' aggregate accuracy as horizontal bars, with two
      specialist reference lines: the locally-fit MiniRocket over all 30
      datasets (same scope as the bars) and the published HC2 reference,
      which only covers 20 of the 30 and is labeled as such.
  (b) C3-head vs. C4, per dataset, against the y=x diagonal. C3-head never
      touches the backbone, so if it beats C4 (equal budget, random init)
      on most datasets, the recovery is about the joint features
      specifically, not just extra epochs.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

from src.figures.figure_style import IEEE_PAGE_WIDTH_IN, PALETTE, apply_style, save_figure

LABEL_DATASETS = {
    "Plane", "Trace", "SonyAIBORobotSurface1",  # largest C3-head - C4 gaps
    "ChlorineConcentration", "Car", "Earthquakes",  # the 3 below the diagonal
}
# Trace and Car's labels otherwise land directly on top of their own point --
# nudged right before adjustText runs so they start from a readable position
# instead of relying on the solver to separate label from marker.
RIGHT_NUDGE = {"Trace", "Car"}


def _draw_bars(ax, joint_csv, head_csv, full_csv, scratch_csv, baselines_csv, minirocket_local_csv) -> None:
    joint = pd.read_csv(joint_csv)["acc_mean"].mean()
    head = pd.read_csv(head_csv)["acc_mean"].mean()
    full = pd.read_csv(full_csv)["acc_mean"].mean()
    scratch = pd.read_csv(scratch_csv)["acc_mean"].mean()
    base = pd.read_csv(baselines_csv)
    hc2 = base["HC2"].dropna().mean() if "HC2" in base.columns else float("nan")
    minirocket_local = (
        pd.read_csv(minirocket_local_csv)["acc"].mean() if minirocket_local_csv.exists() else float("nan")
    )

    labels = ["C2\n(joint)", "C4\n(scratch)", "C3-head\n(frozen)", "C3-full\n(unfrozen)"]
    values = [joint, scratch, head, full]
    colors = [PALETTE[0], PALETTE[2], PALETTE[1], PALETTE[4]]
    # Four bars, several pairs close in grayscale luminance (vermillion vs
    # green vs purple all sit within ~0.04-0.07 of each other) -- hatching
    # gives each bar a distinct silhouette so color alone isn't load-bearing.
    hatches = [None, "//", "xx", ".."]
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5, height=0.6, hatch=hatches)
    for bar, v in zip(bars, values):
        ax.text(v + 0.015, bar.get_y() + bar.get_height() / 2, f"{v:.3f}", va="center", fontsize=7)

    # Two specialist reference lines: MiniRocket-local matches the bars' own
    # 30-dataset scope (the primary comparison); HC2 only covers 20 of the
    # 30 and is labeled as such rather than implying it's the same scope.
    ax.axvline(minirocket_local, color="#888888", linewidth=1.2, linestyle="-", zorder=1)
    ax.text(minirocket_local, -0.85, f"MiniRocket (local) {minirocket_local:.3f}\n(30 datasets)",
            fontsize=6, ha="center", va="bottom")
    ax.axvline(hc2, color="#888888", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(hc2, 3.75, f"HC2 {hc2:.3f}\n(20 datasets)", fontsize=6, ha="center", va="top")

    # The "91% of the full fine-tuning gain" point is made in the manuscript
    # text instead of on the figure -- annotating it here kept overlapping
    # the C3-full bar however it was sized/placed, and reads as cramped.

    ax.set_xlim(0, 1.12)
    ax.set_ylim(4.15, -1.15)
    ax.set_xlabel("Mean test accuracy (30 datasets)")
    ax.tick_params(axis="y", labelsize=7)


def _draw_scatter(ax, head_csv, scratch_csv) -> None:
    head = pd.read_csv(head_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "C3head"})
    scratch = pd.read_csv(scratch_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "C4"})
    df = head.merge(scratch, on="dataset", how="inner")

    above = df["C3head"] > df["C4"]
    n_above = int(above.sum())

    ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.8, linestyle=":", zorder=1)
    ax.scatter(df.loc[above, "C4"], df.loc[above, "C3head"], color=PALETTE[0], s=24,
               edgecolor="black", linewidth=0.3, zorder=2, label="C3-head above C4")
    ax.scatter(df.loc[~above, "C4"], df.loc[~above, "C3head"], color=PALETTE[3], marker="X", s=40,
               edgecolor="black", linewidth=0.4, zorder=3, label="C4 above C3-head")

    texts = []
    for _, row in df[df["dataset"].isin(LABEL_DATASETS)].iterrows():
        x0 = row["C4"] + (0.06 if row["dataset"] in RIGHT_NUDGE else 0.0)
        texts.append(ax.text(x0, row["C3head"], row["dataset"], fontsize=6.0))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("From-scratch accuracy (C4)")
    ax.set_ylabel("Head-only fine-tune accuracy (C3-head)")
    # Placed in data coordinates at a spot verified empty of both labels and
    # markers (checked directly against every text/point bbox, not guessed)
    # -- the point cloud roughly follows the diagonal, so most "corner"
    # positions are occupied by either the legend, a dataset label, or both.
    ax.text(0.03, 0.72, f"{n_above} of {len(df)}\nabove the diagonal", fontsize=7,
            ha="left", va="top", fontweight="bold")
    ax.legend(loc="lower right", fontsize=6)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")


def run(
    per_dataset_cnn_csv: Path = Path("artifacts/results/per_dataset_results.csv"),
    finetune_head_cnn_csv: Path = Path("artifacts/results/finetune_head_cnn.csv"),
    finetune_full_cnn_csv: Path = Path("artifacts/results/finetune_full_cnn.csv"),
    scratch_cnn_csv: Path = Path("artifacts/results/scratch_cnn.csv"),
    baselines_csv: Path = Path("artifacts/results/baselines_reference.csv"),
    minirocket_local_csv: Path = Path("artifacts/results/baselines_local_minirocket.csv"),
    out_dir: Path = Path("figures"),
) -> plt.Figure:
    apply_style()
    fig, (ax_bars, ax_scatter) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH_IN, 3.4))

    _draw_bars(ax_bars, per_dataset_cnn_csv, finetune_head_cnn_csv, finetune_full_cnn_csv,
               scratch_cnn_csv, baselines_csv, minirocket_local_csv)
    _draw_scatter(ax_scatter, finetune_head_cnn_csv, scratch_cnn_csv)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_control_decomposition")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset-cnn-csv", type=Path, default=Path("artifacts/results/per_dataset_results.csv"))
    p.add_argument("--finetune-head-cnn-csv", type=Path, default=Path("artifacts/results/finetune_head_cnn.csv"))
    p.add_argument("--finetune-full-cnn-csv", type=Path, default=Path("artifacts/results/finetune_full_cnn.csv"))
    p.add_argument("--scratch-cnn-csv", type=Path, default=Path("artifacts/results/scratch_cnn.csv"))
    p.add_argument("--baselines-csv", type=Path, default=Path("artifacts/results/baselines_reference.csv"))
    p.add_argument("--minirocket-local-csv", type=Path,
                    default=Path("artifacts/results/baselines_local_minirocket.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.per_dataset_cnn_csv, args.finetune_head_cnn_csv, args.finetune_full_cnn_csv,
                   args.scratch_cnn_csv, args.baselines_csv, args.minirocket_local_csv, args.out_dir))
