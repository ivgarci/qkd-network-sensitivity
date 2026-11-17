"""Shared figure style for the manuscript.

Design constraints, in priority order:

1. Print. Figures are read on paper, so every palette is validated against a
   white surface, not a screen surface.
2. Greyscale legibility. Colour never carries meaning alone: series are also
   separated by line style, marker, or hatch, and heatmap cells carry printed
   values.
3. At most three categorical series in any one panel. Where the data has seven
   levels (the parameters), the figure facets instead of cycling hues.

Palette slots and the sequential ramp are taken unchanged from the reference
design system. Validated with its own checker at
`validate_palette.js "#2a78d6,#eb6834,#1baf7a" --mode light --surface #ffffff
--pairs all` -> all checks pass (worst all-pairs CVD dE 9.2, normal-vision 24.0).
The aqua slot sits at 2.82:1 against white, below the 3:1 bar, so every use of
it carries a visible label or legend text in ink - the documented relief.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- categorical slots (fixed order, never cycled) --------------------------
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
SERIES = (BLUE, ORANGE, AQUA)

# --- sequential ramp, blue 100 -> 700 ---------------------------------------
SEQ_BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
            "#0d366b"]

# --- diverging pair, blue <-> red with neutral grey midpoint ----------------
DIV_LOW, DIV_MID, DIV_HIGH = "#2a78d6", "#f0efec", "#d03b3b"

# --- chrome and ink ---------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

SCENARIO_STYLE = {
    "cyl": dict(color=BLUE, ls="-", marker="o", label="Castilla y Leon"),
    "spain": dict(color=ORANGE, ls="--", marker="s", label="Mainland Spain"),
}


def sequential_cmap(name="seq_blue"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, SEQ_BLUE)


def diverging_cmap(name="div_blue_red"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, [DIV_LOW, DIV_MID, DIV_HIGH])


def apply():
    """Install the manuscript rcParams. Call once before plotting."""
    mpl.rcParams.update({
        # Type 42 embeds TrueType outlines, which MDPI requires; the default
        # Type 3 subsetting is rejected by several production workflows.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Palatino", "Palatino Linotype", "URW Palladio L",
                       "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,

        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,

        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "lines.linewidth": 1.4,
        "lines.markersize": 3.5,
        "legend.frameon": False,
        "errorbar.capsize": 2,
    })


# MDPI single-column text width is about 88 mm; full width about 180 mm.
MM = 1.0 / 25.4
W1 = 88 * MM
W2 = 180 * MM


def save(fig, path):
    # dpi applies only to rasterised artists (the heatmap meshes); all text,
    # lines and contours remain vector.
    fig.savefig(path, format="pdf", dpi=600)
    plt.close(fig)
    return path
