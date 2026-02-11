"""Plot module for QEC simulation results."""

from .config import PlotConfig
from .plotter import (
    plot_custom,
    plot_ler_vs_distance,
    plot_ler_vs_p,
    plot_simulation_results,
)
from .styles import apply_theme, get_palette, PALETTE_DISTANCE
from .utils import add_error_bars, compute_error_bars, sanitize_df

__all__ = [
    "PlotConfig",
    "plot_custom",
    "plot_simulation_results",
    "plot_ler_vs_p",
    "plot_ler_vs_distance",
    "apply_theme",
    "get_palette",
    "PALETTE_DISTANCE",
    "add_error_bars",
    "compute_error_bars",
    "sanitize_df",
]
