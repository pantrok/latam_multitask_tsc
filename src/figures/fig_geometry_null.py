from __future__ import annotations

"""
fig_geometry_null (F4) -- isolation index vs. accuracy penalty. No fit
line: the correlation is null, and drawing a trend line through it would
misrepresent that. rho/p/N are caption material, not drawn on the canvas --
printed to stdout instead.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from adjustText import adjust_text

from src.figures.figure_style import IEEE_COLUMN_WIDTH_IN, PALETTE, apply_style, save_figure


def run(
    correlation_json: Path = Path("artifacts/analysis/geometry_gap_correlation.json"),
    out_dir: Path = Path("figures"),
) -> plt.Figure:
    apply_style()
    result = json.loads(correlation_json.read_text())
    df = pd.DataFrame(result["per_dataset"])

    fig, ax = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, 2.8))
    ax.scatter(df["iota"], df["delta"], s=26, color=PALETTE[0], edgecolor="black",
               linewidth=0.4, alpha=0.85, zorder=2)
    ax.axhline(0, color="grey", linewidth=0.6, linestyle="--", zorder=1)

    texts = []
    threshold = df["delta"].abs().quantile(0.8)
    for _, row in df.iterrows():
        if abs(row["delta"]) > threshold:
            texts.append(ax.text(row["iota"], row["delta"], row["dataset"], fontsize=6.0))
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="grey", lw=0.5))

    ax.set_xlabel(f"Isolation index (mean DTW dist. to {result['k_neighbours']}-NN)")
    ax.set_ylabel(f"Accuracy penalty vs. {result['baseline']}")
    # rho/p/N deliberately not drawn on the canvas -- caption material.
    # Printed to stdout instead, for whoever writes the caption to copy exactly.
    print(f"Caption values: Spearman rho={result['spearman_rho']:+.3f}, "
          f"p={result['permutation_p']:.4f}, N={result['n_datasets']}")

    fig.tight_layout()
    save_figure(fig, out_dir, "fig_geometry_null")
    return fig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--correlation-json", type=Path, default=Path("artifacts/analysis/geometry_gap_correlation.json"))
    p.add_argument("--out-dir", type=Path, default=Path("figures"))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plt.close(run(args.correlation_json, args.out_dir))
