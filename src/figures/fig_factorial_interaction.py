from __future__ import annotations

"""
fig_factorial_interaction -- the CNN's fine-tuning recovery is a 2x2
factorial (normalization statistics frozen/recalibrated x head
frozen/retrained) with a large interaction, not a sequence of additive
steps. An interaction plot is the right shape for that: two nearly-flat,
widely-separated lines. The vertical gap between them is the effect of
recalibrating normalization statistics; the (tiny) horizontal slope of each
line is the effect of retraining the head.

Reference lines on the same accuracy axis: the per-dataset specialist
(locally-fit MiniRocket), C3-full (backbone also unfrozen), and the frozen
*random* backbone (C5) -- below every point in the factorial, since a
trained backbone beats no real training behind it at all.

Optional second panel: C3-head vs. C5 per dataset against the y=x diagonal,
the decisive control from the same data -- unchanged from before.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.figures.figure_style import IEEE_COLUMN_WIDTH_IN, IEEE_PAGE_WIDTH_IN, PALETTE, apply_style, save_figure


def _load_values(
    joint_csv: Path, head_bn_eval_csv: Path, bn_only_csv: Path, head_csv: Path, full_csv: Path,
    random_backbone_csv: Path, minirocket_local_csv: Path,
) -> dict[str, float]:
    return {
        "c2": pd.read_csv(joint_csv)["acc_mean"].mean(),
        "head_bn_eval": pd.read_csv(head_bn_eval_csv)["acc_mean"].mean(),
        "bn_only": pd.read_csv(bn_only_csv)["acc_mean"].mean(),
        "head": pd.read_csv(head_csv)["acc_mean"].mean(),
        "full": pd.read_csv(full_csv)["acc_mean"].mean(),
        "c5": pd.read_csv(random_backbone_csv)["acc_mean"].mean(),
        "minirocket_local": pd.read_csv(minirocket_local_csv)["acc"].mean(),
    }


def _draw_interaction(ax, v: dict[str, float]) -> None:
    x = [0, 1]
    y_frozen = [v["c2"], v["head_bn_eval"]]
    y_recal = [v["bn_only"], v["head"]]

    ax.plot(x, y_frozen, marker="o", color=PALETTE[0], linestyle="-", linewidth=1.6, markersize=6,
            label="Normalization stats frozen", zorder=3)
    ax.plot(x, y_recal, marker="s", color=PALETTE[1], linestyle="--", linewidth=1.6, markersize=6,
            label="Normalization stats recalibrated", zorder=3)

    for xi, yi in zip(x, y_frozen):
        ax.annotate(f"{yi:.4f}", (xi, yi), xytext=(0, 9), textcoords="offset points",
                    fontsize=6.5, ha="center", va="bottom", color=PALETTE[0])
    for xi, yi in zip(x, y_recal):
        ax.annotate(f"{yi:.4f}", (xi, yi), xytext=(0, 9), textcoords="offset points",
                    fontsize=6.5, ha="center", va="bottom", color=PALETTE[1])

    ax.annotate(f"+{y_frozen[1] - y_frozen[0]:.4f}", (0.5, (y_frozen[0] + y_frozen[1]) / 2),
                xytext=(0, -13), textcoords="offset points", fontsize=6.5, ha="center",
                color=PALETTE[0], fontweight="bold")
    ax.annotate(f"+{y_recal[1] - y_recal[0]:.4f}", (0.5, (y_recal[0] + y_recal[1]) / 2),
                xytext=(0, -13), textcoords="offset points", fontsize=6.5, ha="center", va="top",
                color=PALETTE[1], fontweight="bold")

    refs = [
        ("full", "C3-full", "#555555", "-.", "bottom"),
        ("minirocket_local", "MiniRocket", "#333333", "-", "bottom"),
        ("c5", "C5", "#888888", ":", "top"),
    ]
    for key, label, color, ls, va in refs:
        ax.axhline(v[key], color=color, linewidth=0.9, linestyle=ls, zorder=1)
        offset = 4 if va == "bottom" else -4
        ax.annotate(f"{label} {v[key]:.4f}", (1.75, v[key]), xytext=(0, offset), textcoords="offset points",
                    fontsize=6, ha="right", va=va, color=color)

    ax.set_xlim(-0.4, 2.05)
    ax.set_ylim(v["c5"] - 0.05, v["minirocket_local"] + 0.05)
    ax.set_xticks(x)
    ax.set_xticklabels(["Head frozen", "Head retrained"])
    ax.set_ylabel("Mean test accuracy (30 datasets)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), fontsize=6.5, ncol=1, frameon=False)


def _draw_scatter(ax, head_csv: Path, random_backbone_csv: Path) -> None:
    head = pd.read_csv(head_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "c3head"})
    c5 = pd.read_csv(random_backbone_csv)[["dataset", "acc_mean"]].rename(columns={"acc_mean": "c5"})
    df = head.merge(c5, on="dataset")
    above = int((df["c3head"] > df["c5"]).sum())

    ax.plot([0, 1], [0, 1], color="#888888", linewidth=0.8, linestyle=":", zorder=1)
    ax.scatter(df["c5"], df["c3head"], color=PALETTE[0], s=22, edgecolor="black", linewidth=0.3, zorder=2)
    # Bottom-right quadrant is empty by construction (every point sits above
    # the diagonal, and the highest-C5 points also have high C3-head) --
    # verified directly against the data rather than guessed.
    ax.text(0.97, 0.03, f"{above} of {len(df)}\nabove the diagonal", fontsize=7, ha="right", va="bottom",
            fontweight="bold", transform=ax.transAxes)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Random frozen backbone (C5)")
    ax.set_ylabel("Frozen-backbone head fine-tune (C3-head)")
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")


def run(
    joint_csv: Path = Path("artifacts/results/per_dataset_results.csv"),
    head_bn_eval_csv: Path = Path("artifacts/results/finetune_head_bn_eval_cnn.csv"),
    bn_only_csv: Path = Path("artifacts/results/bn_only_cnn.csv"),
    head_csv: Path = Path("artifacts/results/finetune_head_cnn.csv"),
    full_csv: Path = Path("artifacts/results/finetune_full_cnn.csv"),
    random_backbone_csv: Path = Path("artifacts/results/random_backbone_cnn.csv"),
    minirocket_local_csv: Path = Path("artifacts/results/baselines_local_minirocket.csv"),
    out_dir: Path = Path("figures"),
    two_panel: bool = True,
) -> plt.Figure:
    apply_style()
    v = _load_values(joint_csv, head_bn_eval_csv, bn_only_csv, head_csv, full_csv,
                      random_backbone_csv, minirocket_local_csv)

    if two_panel:
        fig, (ax_interaction, ax_scatter) = plt.subplots(1, 2, figsize=(IEEE_PAGE_WIDTH_IN, 3.4),
                                                          gridspec_kw={"width_ratios": [1.15, 1]})
        _draw_interaction(ax_interaction, v)
        _draw_scatter(ax_scatter, head_csv, random_backbone_csv)
    else:
        fig, ax_interaction = plt.subplots(1, 1, figsize=(IEEE_COLUMN_WIDTH_IN, 3.0))
        _draw_interaction(ax_interaction, v)

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_factorial_interaction")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--joint-csv", type=Path, default=Path("artifacts/results/per_dataset_results.csv"))
    p.add_argument("--head-bn-eval-csv", type=Path, default=Path("artifacts/results/finetune_head_bn_eval_cnn.csv"))
    p.add_argument("--bn-only-csv", type=Path, default=Path("artifacts/results/bn_only_cnn.csv"))
    p.add_argument("--head-csv", type=Path, default=Path("artifacts/results/finetune_head_cnn.csv"))
    p.add_argument("--full-csv", type=Path, default=Path("artifacts/results/finetune_full_cnn.csv"))
    p.add_argument("--random-backbone-csv", type=Path, default=Path("artifacts/results/random_backbone_cnn.csv"))
    p.add_argument("--minirocket-local-csv", type=Path,
                    default=Path("artifacts/results/baselines_local_minirocket.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    p.add_argument("--two-panel", action="store_true", default=True)
    p.add_argument("--single-panel", dest="two_panel", action="store_false")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.joint_csv, args.head_bn_eval_csv, args.bn_only_csv, args.head_csv, args.full_csv,
                   args.random_backbone_csv, args.minirocket_local_csv, args.out_dir, args.two_panel))
