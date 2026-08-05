from __future__ import annotations

"""
fig_four_conditions (F1) -- specialist / joint (C2) / head-only fine-tune
(C3-head) / from-scratch (C4) accuracy per dataset. C3-head, not C3-full, is
the condition this repository recommends: it is the only one of the four
that keeps a single shared backbone in memory (a fully fine-tuned model per
dataset gives up the deployment advantage the whole study is about).
C3-full is deliberately not plotted here -- it sits within 0.02 of C3-head
on average and would saturate the figure without changing the
recommendation; it stays in the results table instead.

The specialist marker is HC2 where a published reference exists (20 of 30
datasets), and a locally-fit MiniRocket for the remaining 10 -- every
dataset gets a specialist point, not just the HC2 subset. The 10 filled in
this way keep a "(local MiniRocket)" suffix so the source is never
ambiguous.

Horizontal, datasets on the y-axis sorted ascending by joint (C2) accuracy,
markers only (never a line connecting different datasets, since they are
categories not a series), legend outside the data area.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.figure_style import IEEE_PAGE_WIDTH_IN, PALETTE, apply_style, save_figure


def run(
    per_dataset_cnn_csv: Path = Path("artifacts/results/per_dataset_results.csv"),
    finetune_head_cnn_csv: Path = Path("artifacts/results/finetune_head_cnn.csv"),
    scratch_cnn_csv: Path = Path("artifacts/results/scratch_cnn.csv"),
    baselines_csv: Path = Path("artifacts/results/baselines_reference.csv"),
    minirocket_local_csv: Path = Path("artifacts/results/baselines_local_minirocket.csv"),
    out_dir: Path = Path("figures"),
) -> plt.Figure:
    apply_style()
    joint = pd.read_csv(per_dataset_cnn_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "C2"})
    head = pd.read_csv(finetune_head_cnn_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "C3head"})
    scratch = pd.read_csv(scratch_cnn_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "C4"})
    base = pd.read_csv(baselines_csv)

    df = joint.merge(head, on="dataset", how="inner").merge(scratch, on="dataset", how="inner")
    if "HC2" in base.columns:
        df = df.merge(base[["dataset", "HC2"]], on="dataset", how="left")
    else:
        df["HC2"] = float("nan")

    df["from_hc2"] = df["HC2"].notna()
    if minirocket_local_csv.exists():
        mr = pd.read_csv(minirocket_local_csv)[["dataset", "acc"]].rename(columns={"acc": "MiniRocketLocal"})
        df = df.merge(mr, on="dataset", how="left")
        df["Specialist"] = df["HC2"].where(df["from_hc2"], df["MiniRocketLocal"])
    else:
        df["Specialist"] = df["HC2"]

    df = df.sort_values("C2", ascending=True).reset_index(drop=True)
    n = len(df)

    fig, ax = plt.subplots(figsize=(IEEE_PAGE_WIDTH_IN, max(4.2, n * 0.13)))
    y = range(n)

    # The new headline jump: C2 -> C3-head, not C2 -> C3-full.
    for i, row in df.iterrows():
        ax.annotate(
            "", xy=(row["C3head"], i), xytext=(row["C2"], i),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.0, alpha=0.75, shrinkA=0, shrinkB=0),
            zorder=1,
        )

    ax.scatter(df["C2"], y, color=PALETTE[0], marker="o", s=24, zorder=2,
               label="Joint (C2)", edgecolor="black", linewidth=0.3)
    ax.scatter(df["C3head"], y, color=PALETTE[1], marker="s", s=24, zorder=2,
               label="Head-only fine-tune (C3-head)", edgecolor="black", linewidth=0.3)
    # "P" (filled plus), not "D" -- a diamond is just a rotated square and
    # has the same vertex count as C3-head's square marker, so the two are
    # not reliably distinguishable by shape alone once color is unreliable.
    ax.scatter(df["C4"], y, color=PALETTE[2], marker="P", s=26, zorder=2,
               label="From scratch (C4)", edgecolor="black", linewidth=0.3)
    has_spec = df["Specialist"].notna()
    ax.scatter(df.loc[has_spec, "Specialist"], [i for i in y if has_spec.iloc[i]], color=PALETTE[3], marker="^",
               s=26, zorder=2, label="Specialist (HC2 or local MiniRocket)", edgecolor="black", linewidth=0.3)

    labels = [name + ("" if ref else "  (local MiniRocket)") for name, ref in zip(df["dataset"], df["from_hc2"])]
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=6.0)
    for tick, ref in zip(ax.get_yticklabels(), df["from_hc2"]):
        if not ref:
            tick.set_color("#888888")

    ax.set_xlabel("Test accuracy")
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-1, n)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=4, fontsize=6.5, frameon=False)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_four_conditions")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-dataset-cnn-csv", type=Path, default=Path("artifacts/results/per_dataset_results.csv"))
    p.add_argument("--finetune-head-cnn-csv", type=Path, default=Path("artifacts/results/finetune_head_cnn.csv"))
    p.add_argument("--scratch-cnn-csv", type=Path, default=Path("artifacts/results/scratch_cnn.csv"))
    p.add_argument("--baselines-csv", type=Path, default=Path("artifacts/results/baselines_reference.csv"))
    p.add_argument("--minirocket-local-csv", type=Path,
                    default=Path("artifacts/results/baselines_local_minirocket.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.per_dataset_cnn_csv, args.finetune_head_cnn_csv, args.scratch_cnn_csv,
                   args.baselines_csv, args.minirocket_local_csv, args.out_dir))
