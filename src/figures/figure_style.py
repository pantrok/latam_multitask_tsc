from __future__ import annotations

"""
Shared IEEE-compliant matplotlib style for every figure in this repo.

IEEE column width: 3.5 in (single) / 7.16 in (double). Figures are sized so
they read at that final width without upscaling. Okabe-Ito palette (colorblind-
safe, distinguishable in grayscale); markers/linestyles vary too, since color
alone is not enough once printed in grayscale. No in-figure titles -- the
caption in the manuscript carries that. Every figure is saved as both a
vector PDF (for \\includegraphics) and a 400 dpi PNG (submission + backup).
"""

from pathlib import Path

import matplotlib.pyplot as plt

IEEE_COLUMN_WIDTH_IN = 3.5
IEEE_PAGE_WIDTH_IN = 7.16

# Okabe-Ito, colorblind-safe.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky_blue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
PALETTE = [OKABE_ITO[k] for k in
           ("blue", "vermillion", "green", "orange", "purple", "sky_blue", "yellow", "black")]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":"]

FONT_SIZE_PT = 7
DPI_PNG = 400


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": FONT_SIZE_PT,
        "axes.labelsize": FONT_SIZE_PT,
        "axes.titlesize": FONT_SIZE_PT,
        "xtick.labelsize": FONT_SIZE_PT - 1,
        "ytick.labelsize": FONT_SIZE_PT - 1,
        "legend.fontsize": FONT_SIZE_PT - 1,
        "figure.titlesize": FONT_SIZE_PT,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": DPI_PNG,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    png_path = out_dir / f"{name}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=DPI_PNG, bbox_inches="tight")
    print(f"Wrote {pdf_path} and {png_path}")
