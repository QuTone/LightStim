"""Styling: color palettes, rcParams, theme for publication-ready figures."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

# Tech-oriented, distinct, colorblind-friendly palette (extended from test_planar)
PALETTE_DISTANCE: Dict[int, str] = {
    3: "#a63603",
    5: "#1b9e77",
    7: "#7570b3",
    9: "#d95f02",
    11: "#e7298a",
    13: "#66a61e",
    15: "#e6ab02",
}

# Default palette for up to 8 groups (husl-based, distinct)
DEFAULT_PALETTE_HEX: List[str] = [
    "#a63603", "#1b9e77", "#7570b3", "#d95f02",
    "#e7298a", "#66a61e", "#e6ab02", "#666666",
]


def get_palette(n: int, palette: Optional[Dict | str] = None):
    """Get color list for n groups. palette can be a dict (value->color) or preset name."""
    if palette is None:
        return DEFAULT_PALETTE_HEX[:n] if n <= len(DEFAULT_PALETTE_HEX) else sns.color_palette("husl", n).as_hex()
    if isinstance(palette, dict):
        values = sorted(palette.keys())
        return [palette[v] for v in values]
    if isinstance(palette, str) and palette == "distance":
        return [PALETTE_DISTANCE.get(k, DEFAULT_PALETTE_HEX[i % len(DEFAULT_PALETTE_HEX)])
                for i, k in enumerate(range(3, 3 + n))]
    return sns.color_palette(palette, n_colors=n).as_hex()


def apply_theme(
    figsize: tuple = (7, 5),
    font_size: int = 11,
    font_family: str = "sans-serif",
    grid_alpha: float = 0.3,
    dpi: int = 150,
) -> None:
    """Apply professional theme to matplotlib."""
    plt.rcParams.update({
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "font.size": font_size,
        "font.family": font_family,
        "axes.labelsize": font_size + 1,
        "axes.titlesize": font_size + 2,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "grid.alpha": grid_alpha,
    })
    sns.set_style("whitegrid", {"grid.alpha": grid_alpha})


def ensure_theme_applied():
    """Apply default theme if not already customized."""
    if plt.rcParams.get("figure.figsize") == [6.0, 4.0]:  # matplotlib default
        apply_theme()
