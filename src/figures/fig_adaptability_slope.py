from __future__ import annotations

"""
fig_adaptability_slope (F3) -- CNN and LSTM, joint -> head-only fine-tune ->
full fine-tune, as a three-point slope chart. For the CNN, head-only
fine-tuning (frozen backbone) already recovers almost all of what full
fine-tuning does; for the LSTM it barely moves, and most of its recovery
needs the backbone to actually change. The crossing point between the two
models' lines is the headline: the LSTM is the better joint model but far
less adaptable, so the CNN ends up ahead after fine-tuning either way.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.figure_style import IEEE_COLUMN_WIDTH_IN, PALETTE, apply_style, save_figure


def run(three_condition_csv: Path = Path("artifacts/results/three_condition_table.csv"), out_dir: Path = Path("figures")) -> plt.Figure:
    apply_style()
    df = pd.read_csv(three_condition_csv)
    df30 = df[df["scope"] == 30]

    # HC2 only exists at scope=20 -- it has no published reference for the other 10 datasets.
    hc2_mean = df.loc[(df["condition"] == "specialist") & (df["model"] == "HC2") & (df["scope"] == 20), "acc_mean"].iloc[0]

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, 3.2))
    x = [0, 1, 2]

    # marker + linestyle differ between CNN/LSTM, not just color -- PALETTE[0]
    # (blue) and PALETTE[1] (vermillion) are colorblind-distinguishable but
    # sit close enough in grayscale luminance that color alone isn't
    # reliable once printed without color.
    series = [("cnn", "CNN", PALETTE[0], "o", "-"), ("lstm", "LSTM", PALETTE[1], "s", "--")]

    # Manual offsets, not adjustText: the two models sit within 0.01 of each
    # other at "joint" (x=0) and would collide there, but are well separated
    # at x=1/x=2 -- known in advance, so placed directly rather than left to
    # a general-purpose solver. CNN labels go above its marker, LSTM below,
    # consistently at all three x positions.
    dy = {"cnn": 9, "lstm": -9}
    va = {"cnn": "bottom", "lstm": "top"}
    for model, label, color, marker, linestyle in series:
        joint = df30.loc[(df30["condition"] == "joint") & (df30["model"] == model), "acc_mean"].iloc[0]
        head = df30.loc[(df30["condition"] == "head") & (df30["model"] == model), "acc_mean"].iloc[0]
        full = df30.loc[(df30["condition"] == "finetuned") & (df30["model"] == model), "acc_mean"].iloc[0]
        ax.plot(x, [joint, head, full], marker=marker, markersize=7, linewidth=2.0, color=color, linestyle=linestyle,
                zorder=3)
        for xi, v in zip(x, [joint, head, full]):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, dy[model]), textcoords="offset points",
                        ha="center", va=va[model], fontsize=6.5)
        # Model identified directly at the line's end, not in a legend --
        # the legend previously spelled out the same value sequence already
        # printed on the plot, so it was dropped; this is the only remaining
        # way to tell which line is which.
        ax.annotate(label, (x[-1], full), xytext=(8, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=7.5, color=color, fontweight="bold")

    ax.axhline(hc2_mean, color="grey", linewidth=1.0, linestyle="--", zorder=1,
               label=f"HC2 specialist ({hc2_mean:.3f})")

    ax.set_xticks(x)
    ax.set_xticklabels(["C2", "C3-head", "C3-full"])
    ax.set_xlim(-0.4, 2.4)
    ax.set_ylim(0.30, 0.92)
    ax.set_ylabel("Mean test accuracy")
    ax.legend(loc="upper center", fontsize=6.5, bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_adaptability_slope")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--three-condition-csv", type=Path, default=Path("artifacts/results/three_condition_table.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.three_condition_csv, args.out_dir))
