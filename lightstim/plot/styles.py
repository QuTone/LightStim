"""
LightStim Plot Styles
=====================
Single source of truth for publication-quality figure styling.

Color Palette (Dark2 / ColorBrewer-derived)
--------------------------------------------
Desaturated, high-contrast, colorblind-friendly.  These are the canonical
colors for all LightStim figures.

  RUST     #a63603  — Rotated SC d=3  / bb_72_12_6
  TEAL     #1b9e77  — Rotated SC d=5  / bb_108_8_10
  VIOLET   #7570b3  — Rotated SC d=7  / bb_144_12_12
  ORANGE   #d95f02  — Rotated SC d=9
  MAGENTA  #e7298a  — extra group 5
  OLIVE    #66a61e  — extra group 6
  GOLD     #e6ab02  — extra group 7
  SLATE    #666666  — extra group 8 / neutral

Usage
-----
from lightstim.plot.styles import apply_paper_style, PALETTE, CODES, bold_ticks

fig, ax = plt.subplots(...)
apply_paper_style()          # call once per script/notebook
...
bold_ticks(ax)               # call on each Axes after plotting
fig.tight_layout(pad=0.4)    # keep whitespace tight
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt


# ── Color palette ─────────────────────────────────────────────────────────────

#: Canonical 8-color palette (Dark2 / ColorBrewer-derived).
PALETTE: List[str] = [
    "#a63603",  # 0  RUST
    "#1b9e77",  # 1  TEAL
    "#7570b3",  # 2  VIOLET
    "#d95f02",  # 3  ORANGE
    "#e7298a",  # 4  MAGENTA
    "#66a61e",  # 5  OLIVE
    "#e6ab02",  # 6  GOLD
    "#666666",  # 7  SLATE
]

#: Map code distance → color (first 4 distances use RUST / TEAL / VIOLET / DUSTY-ROSE).
PALETTE_DISTANCE: Dict[int, str] = {
    3:  PALETTE[0],   # RUST
    5:  PALETTE[1],   # TEAL
    7:  PALETTE[2],   # VIOLET
    9:  "#d4607a",    # ROSE RED — medium rose, distinct from violet
    11: PALETTE[4],
    13: PALETTE[5],
    15: PALETTE[6],
}

#: Per-code-family colors for overlay plots (Fig 3 style).
CODES: Dict[str, str] = {
    # Surface codes — blue family
    "rotated_sc":   "#2166ac",
    "unrotated_sc": "#4dac26",
    "toric":        "#d01c8b",
    "color_code":   "#f4a582",
    # BB codes — PALETTE order
    "bb_72_12_6":   PALETTE[0],
    "bb_108_8_10":  PALETTE[1],
    "bb_144_12_12": PALETTE[2],
    "bb_288_12_18": PALETTE[3],
}


# ── rcParams ───────────────────────────────────────────────────────────────────

#: Full rcParams dict for paper-quality figures.
#: Apply once per script with ``apply_paper_style()``.
PAPER_RC: Dict[str, object] = {
    # Font — everything bold
    "font.family":        "sans-serif",
    "font.weight":        "bold",
    "font.size":          14,
    # Axes labels & title
    "axes.labelsize":     17,
    "axes.labelweight":   "bold",
    "axes.titlesize":     18,
    "axes.titleweight":   "bold",
    "axes.linewidth":     1.3,
    # Ticks
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "xtick.major.width":  1.2,
    "ytick.major.width":  1.2,
    # Legend
    "legend.fontsize":    12,
    "legend.title_fontsize": 12,
    "legend.frameon":     True,
    "legend.edgecolor":   "0.7",
    # Lines & markers
    "lines.linewidth":    2.2,
    "lines.markersize":   8.0,
    # Grid
    "axes.grid":          True,
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.5,
    # Figure
    "figure.dpi":         150,
    "figure.autolayout":  False,  # use tight_layout(pad=0.4) explicitly
}


# ── Public API ─────────────────────────────────────────────────────────────────

def apply_paper_style() -> None:
    """Apply PAPER_RC to matplotlib global rcParams.

    Call once at the top of any plotting script or notebook cell.
    """
    mpl.rcParams.update(PAPER_RC)


def bold_ticks(ax: mpl.axes.Axes) -> None:
    """Set fontweight='bold' on all tick labels of *ax*.

    Tick label boldness cannot be set via rcParams alone — call this
    after all plotting is done on the Axes.
    """
    for tick in ax.xaxis.get_major_ticks() + ax.yaxis.get_major_ticks():
        tick.label1.set_fontweight("bold")


def get_color(key: Union[int, str]) -> str:
    """Return a canonical color.

    Args:
        key: integer index into PALETTE, code-family name from CODES,
             or integer distance for PALETTE_DISTANCE.
    """
    if isinstance(key, int):
        if key in PALETTE_DISTANCE:
            return PALETTE_DISTANCE[key]
        return PALETTE[key % len(PALETTE)]
    if isinstance(key, str) and key in CODES:
        return CODES[key]
    raise KeyError(f"Unknown color key: {key!r}")


def get_palette(n: int) -> List[str]:
    """Return the first *n* colors from PALETTE (cycles if n > 8)."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


# ── Legacy shim (keep old callers working) ────────────────────────────────────

# ── Code / decoder metadata ────────────────────────────────────────────────────

#: Human-readable labels for code families and specific BB codes.
CODE_LABELS: Dict[str, str] = {
    "rotated_sc":    "Rotated SC",
    "unrotated_sc":  "Unrotated SC",
    "toric":         "Toric",
    "color_code":    "Color (6-6-6)",
    "bb_72_12_6":    r"$[[72,12,6]]$",
    "bb_108_8_10":   r"$[[108,8,10]]$",
    "bb_144_12_12":  r"$[[144,12,12]]$",
    "bb_288_12_18":  r"$[[288,12,18]]$",
}

#: Line styles per code family (distinguish on same axes).
CODE_LINESTYLES: Dict[str, str] = {
    "rotated_sc":   "-",
    "unrotated_sc": "--",
    "toric":        ":",
    "color_code":   "-.",
}

#: Markers per code family.
CODE_MARKERS: Dict[str, str] = {
    "rotated_sc":   "o",
    "unrotated_sc": "s",
    "toric":        "^",
    "color_code":   "D",
}

#: Line styles per decoder label.
DECODER_LINESTYLES: Dict[str, str] = {
    "gpu_bposd": "-",
    "mwpf":      "--",
}

#: Markers per decoder label.
DECODER_MARKERS: Dict[str, str] = {
    "gpu_bposd": "o",
    "mwpf":      "X",
}


# ── Legacy shim (keep old callers working) ────────────────────────────────────

def apply_theme(figsize=(7, 5), font_size=12, **_):
    """Deprecated shim — use apply_paper_style() instead."""
    apply_paper_style()
    mpl.rcParams["figure.figsize"] = list(figsize)
