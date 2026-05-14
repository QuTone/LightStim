"""Utilities for plot module: error bar computation, column validation."""

from typing import Optional

import numpy as np
import pandas as pd


def compute_error_bars(
    df: pd.DataFrame,
    errors_col: str = "errors",
    shots_col: str = "post_selected_shots",
    y_col: str = "logical_error_rate",
    confidence: float = 1.96,
) -> pd.DataFrame:
    """
    Add error bar column for binomial LER: y_err = z * sqrt(p*(1-p)/n).
    Returns a copy of df with {y_col}_err column added.
    """
    df = df.copy()
    if errors_col not in df.columns or shots_col not in df.columns:
        return df
    n = df[shots_col].values
    p = df[errors_col].values / np.maximum(n, 1)
    p = np.clip(p, 1e-10, 1 - 1e-10)
    err = confidence * np.sqrt(p * (1 - p) / np.maximum(n, 1))
    df[f"{y_col}_err"] = err
    return df


def add_error_bars(df: pd.DataFrame, y_col: str = "logical_error_rate") -> pd.DataFrame:
    """Add y_col_err from errors and post_selected_shots if not present."""
    if f"{y_col}_err" in df.columns:
        return df
    if "errors" in df.columns and "post_selected_shots" in df.columns:
        return compute_error_bars(df, y_col=y_col)
    return df


def sanitize_df(df: pd.DataFrame, x: str, y: str, hue: Optional[str] = None) -> pd.DataFrame:
    """Drop rows with NaN in essential columns."""
    cols = [x, y]
    if hue:
        cols.append(hue)
    return df.dropna(subset=[c for c in cols if c in df.columns])
