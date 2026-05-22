"""Configuration for plot module."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


@dataclass
class PlotConfig:
    """Configuration for customizable simulation result plots."""

    x: str
    y: str
    hue: Optional[str] = None
    style: Optional[str] = None
    facet_col: Optional[str] = None
    facet_row: Optional[str] = None
    x_scale: Literal["linear", "log"] = "log"
    y_scale: Literal["linear", "log"] = "log"
    palette: Optional[Dict[Any, str] | str] = None
    title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    error_bars: bool = True
    figsize: tuple = (7, 5)
    marker: Optional[str] = "o"
    linewidth: float = 2.5
