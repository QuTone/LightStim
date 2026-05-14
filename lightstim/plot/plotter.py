"""Plotting functions for QEC simulation results."""

from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import PlotConfig
from .styles import apply_theme, get_palette, PALETTE_DISTANCE
from .utils import add_error_bars, sanitize_df


def plot_custom(
    df: pd.DataFrame,
    cfg: PlotConfig,
    save_path: Optional[str] = None,
):
    """
    Create a configurable plot from simulation results.
    Supports line plots with x, y, hue, optional facets, and error bars.
    """
    apply_theme(cfg.figsize)
    df = sanitize_df(df, cfg.x, cfg.y, cfg.hue)
    if cfg.error_bars:
        df = add_error_bars(df, cfg.y)

    hue_order = None
    palette = None
    if cfg.hue:
        hue_vals = list(df[cfg.hue].unique())
        hue_order = sorted(hue_vals, key=lambda x: (x,))
        if isinstance(cfg.palette, dict):
            palette = cfg.palette
        elif cfg.palette == "distance":
            palette = {int(k): PALETTE_DISTANCE.get(int(k), "#666666") for k in hue_vals}
        else:
            palette = dict(zip(hue_order, get_palette(len(hue_vals), cfg.palette)))

    use_facets = cfg.facet_col or cfg.facet_row

    if use_facets:
        g = sns.relplot(
            data=df,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            col=cfg.facet_col,
            row=cfg.facet_row,
            kind="line",
            hue_order=hue_order,
            palette=palette,
            markers=cfg.marker,
            linewidth=cfg.linewidth,
            facet_kws={"sharex": True, "sharey": True},
        )
        for ax in g.axes.flat:
            if cfg.x_scale == "log":
                ax.set_xscale("log")
            if cfg.y_scale == "log":
                ax.set_yscale("log")
        fig = g.fig
    else:
        fig, ax = plt.subplots(figsize=cfg.figsize)
        sns.lineplot(
            data=df,
            x=cfg.x,
            y=cfg.y,
            hue=cfg.hue,
            ax=ax,
            hue_order=hue_order,
            palette=palette,
            marker=cfg.marker,
            linewidth=cfg.linewidth,
        )
        if cfg.x_scale == "log":
            ax.set_xscale("log")
        if cfg.y_scale == "log":
            ax.set_yscale("log")

    if cfg.title:
        plt.title(cfg.title)
    if cfg.x_label:
        plt.xlabel(cfg.x_label)
    if cfg.y_label:
        plt.ylabel(cfg.y_label)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
    return fig


def plot_simulation_results(
    df: pd.DataFrame,
    x: str = "p1",
    y: str = "logical_error_rate",
    hue: Optional[str] = "d",
    x_scale: str = "log",
    y_scale: str = "log",
    save_path: Optional[str] = None,
    **kwargs,
):
    """
    One-liner to plot LER vs physical error rate.
    Uses p1 or p as x-axis and logical_error_rate as y by default.
    """
    if "p" in df.columns and x == "p1" and "p1" not in df.columns:
        x = "p"
    cfg = PlotConfig(
        x=x,
        y=y,
        hue=hue,
        x_scale=x_scale,
        y_scale=y_scale,
        palette="distance",
        **kwargs,
    )
    return plot_custom(df, cfg, save_path=save_path)


def plot_ler_vs_p(
    df: pd.DataFrame,
    hue: str = "d",
    x_col: str = "p1",
    save_path: Optional[str] = None,
    **kwargs,
):
    """Preset: LER vs physical error rate (p or p1), colored by distance."""
    defaults = {
        "x_label": "Physical Error Rate (p)",
        "y_label": "Logical Error Rate (LER)",
        "title": "LER vs Physical Error Rate",
    }
    opts = {**defaults, **kwargs}
    return plot_simulation_results(
        df,
        x=x_col,
        y="logical_error_rate",
        hue=hue,
        save_path=save_path,
        **opts,
    )


def plot_ler_vs_distance(
    df: pd.DataFrame,
    hue: str = "decoder",
    x_col: str = "d",
    save_path: Optional[str] = None,
    **kwargs,
):
    """Preset: LER vs code distance, colored by decoder."""
    defaults = {
        "x_label": "Code Distance (d)",
        "y_label": "Logical Error Rate (LER)",
        "title": "LER vs Code Distance",
    }
    opts = {**defaults, **kwargs}
    return plot_simulation_results(
        df,
        x=x_col,
        y="logical_error_rate",
        hue=hue,
        save_path=save_path,
        **opts,
    )
