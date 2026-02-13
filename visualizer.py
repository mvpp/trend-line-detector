"""Candlestick chart with pivot markers, trend lines, and MACD using mplfinance."""

from typing import List

import mplfinance as mpf
import numpy as np
import pandas as pd

from config import (
    FIGURE_SIZE,
    LINE_ALPHA,
    LINE_WIDTH,
    MACD_FAST,
    MACD_SIGNAL,
    MACD_SLOW,
    PANEL_RATIOS,
    PIVOT_MARKER_SIZE,
    SAVE_DPI,
)
from trend_fitter import TrendLine


def _compute_macd(
    close: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """Compute MACD line, signal line, and histogram using pandas EWM."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def plot_trend_lines(
    df: pd.DataFrame,
    support_pivots: pd.DataFrame,
    resistance_pivots: pd.DataFrame,
    support_lines: List[TrendLine],
    resistance_lines: List[TrendLine],
    title: str = "",
    savefig: str | None = None,
    show_pivots: bool = False,
    dash_lines: bool = False,
) -> None:
    """Plot a candlestick chart overlaid with pivots, trend lines, and MACD.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        support_pivots: DataFrame with [date, price] for support pivots.
        resistance_pivots: DataFrame with [date, price] for resistance pivots.
        support_lines: Fitted support TrendLine objects.
        resistance_lines: Fitted resistance TrendLine objects.
        title: Chart title.
        savefig: If provided, save chart to this file path instead of showing.
        show_pivots: If True, draw pivot-point scatter markers on the chart.
        dash_lines: If True, draw trend lines as dashed instead of solid.
    """
    addplots = []

    # --- Optional pivot markers ---
    if show_pivots:
        # Support pivot markers (green upward triangles)
        if not support_pivots.empty:
            sup_series = pd.Series(np.nan, index=df.index, dtype=float)
            for _, pivot in support_pivots.iterrows():
                sup_series.loc[pivot["date"]] = pivot["price"]
            if sup_series.notna().any():
                addplots.append(mpf.make_addplot(
                    sup_series, type="scatter", marker="^",
                    markersize=PIVOT_MARKER_SIZE, color="green",
                ))

        # Resistance pivot markers (red downward triangles)
        if not resistance_pivots.empty:
            res_series = pd.Series(np.nan, index=df.index, dtype=float)
            for _, pivot in resistance_pivots.iterrows():
                res_series.loc[pivot["date"]] = pivot["price"]
            if res_series.notna().any():
                addplots.append(mpf.make_addplot(
                    res_series, type="scatter", marker="v",
                    markersize=PIVOT_MARKER_SIZE, color="red",
                ))

    # --- MACD subplot (panel 2) ---
    macd_line, signal_line, histogram = _compute_macd(df["Close"])

    addplots.append(mpf.make_addplot(
        macd_line, panel=2, color="blue", width=0.8, ylabel="MACD",
    ))
    addplots.append(mpf.make_addplot(
        signal_line, panel=2, color="orange", width=0.8,
    ))

    # Split histogram into positive (green) and negative (red)
    hist_pos = histogram.where(histogram >= 0, np.nan)
    hist_neg = histogram.where(histogram < 0, np.nan)
    addplots.append(mpf.make_addplot(
        hist_pos, panel=2, type="bar", color="green", width=0.7,
    ))
    addplots.append(mpf.make_addplot(
        hist_neg, panel=2, type="bar", color="red", width=0.7,
    ))

    # --- Trend lines via alines ---
    all_alines = []
    aline_colors = []

    for line in support_lines:
        start_date = df.index[line.start_bar]
        end_date = df.index[line.end_bar]
        start_price = line.slope * line.start_bar + line.intercept
        end_price = line.slope * line.end_bar + line.intercept
        all_alines.append([(start_date, start_price), (end_date, end_price)])
        aline_colors.append("green")

    for line in resistance_lines:
        start_date = df.index[line.start_bar]
        end_date = df.index[line.end_bar]
        start_price = line.slope * line.start_bar + line.intercept
        end_price = line.slope * line.end_bar + line.intercept
        all_alines.append([(start_date, start_price), (end_date, end_price)])
        aline_colors.append("red")

    kwargs = dict(
        type="candle",
        style="charles",
        title=title,
        volume=True,
        figsize=FIGURE_SIZE,
        tight_layout=True,
        panel_ratios=PANEL_RATIOS,
    )

    if addplots:
        kwargs["addplot"] = addplots

    if all_alines:
        kwargs["alines"] = dict(
            alines=all_alines,
            colors=aline_colors,
            linewidths=LINE_WIDTH,
            alpha=LINE_ALPHA,
            linestyle="--" if dash_lines else "-",
        )

    if savefig:
        kwargs["savefig"] = dict(fname=savefig, dpi=SAVE_DPI, bbox_inches="tight")

    mpf.plot(df, **kwargs)
